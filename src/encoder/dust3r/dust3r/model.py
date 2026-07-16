# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import torch.nn as nn
import os
from packaging import version
import huggingface_hub
from typing import NamedTuple, Optional

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed
from dust3r.cls_token import Block
from functools import partial
from timm.layers import Mlp

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa
from RADIO.utils import get_radio_model

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AdaptorInput(NamedTuple):
    images: torch.Tensor
    summary: torch.Tensor
    features: torch.Tensor
    feature_fmt: str
    patch_size: int

class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)
        
        # dust3r specific initialization
        self.encoder = None

        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)
        print(freeze)

        self.radio_model_teacher = None

        self.adaptors = None

        self.fc1 = nn.Linear(1024, 1024)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(1024, 1024)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            # 'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        summary, spatial_features = self.encoder(image)

        _, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # # now apply the transformer encoder and normalization
        # for blk in self.enc_blocks:
        #     x = blk(x, pos)

        # x = self.enc_norm(x)
        return summary, spatial_features, pos

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out_summary, out_spatial_features, pos = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out_summary, out_summary2 = out_summary.chunk(2, dim=0)
            out_spatial_features, out_spatial_features2 = out_spatial_features.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out_summary, out_spatial_features, pos = self._encode_image(img1, true_shape1)
            out_summary2, out_spatial_features2, pos2 = self._encode_image(img2, true_shape2)
        return out_summary, out_summary2,out_spatial_features,out_spatial_features2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2): # instance表示图像id
            # computing half of forward pass!'
            summary1, summary2, feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            summary1, summary2 = interleave(summary1, summary2)
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            summary1, summary2, feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (summary1, summary2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def cls_token_adaptors(self, view1):
            input = view1['img'].clone() # N*3*384*512
            bacth = input.shape[0]

            summary1_radio, spatial_features1_radio = self.encoder(view1['img'])

            cls_feat1 = spatial_features1_radio
            cls_summary1 = summary1_radio


            bb_summary = cls_summary1.view(bacth, 3, 1024)
            all_summary = bb_summary
            all_feat = cls_feat1

            ret = dict()
            for name, adaptor in self.adaptors.items():
                if all_summary.ndim == 3:
                    summary = all_summary[:, adaptor.head_idx]
                else:
                    summary = all_summary
                ada_input = AdaptorInput(images=input, summary=summary.float(), features=all_feat, feature_fmt='NLC', patch_size=16)
                v = adaptor(ada_input).to(torch.float32)
                ret[name] = v

            return ret

    def forward(self, view1, view2, type = None):
        # encode the two images --> B,S,D
        if type is None:
            (shape1, shape2), (summary1, summary2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

            summary1_radio, spatial_features1_radio = self.radio_model_teacher(view1['img'])
            summary2_radio, spatial_features2_radio = self.radio_model_teacher(view2['img'])
            
            cls_feat1 = feat1
            cls_feat2 = feat2

            cls_summary1 = summary1
            cls_summary2 = summary2

            feat1_dec = self.fc2(self.gelu(self.fc1(feat1)))
            feat2_dec = self.fc2(self.gelu(self.fc1(feat2)))
            dec1, dec2 = self._decoder(feat1_dec, pos1, feat2_dec, pos2)

            with torch.cuda.amp.autocast(enabled=False):
                res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
                res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

            res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame

            res1["radio_feat"] = spatial_features1_radio.detach()
            res2["radio_feat"] = spatial_features2_radio.detach()
            res1["cls_feat"] = cls_feat1
            res2["cls_feat"] = cls_feat2

            res1["radio_summary"] = summary1_radio.detach()
            res2["radio_summary"] = summary2_radio.detach()
            res1["cls_summary"] = cls_summary1
            res2["cls_summary"] = cls_summary2

            return res1, res2

        else:
            res1 = dict()
            res2 = dict()
            cls_summary1, cls_feat1 = self.encoder(view1['img'])
            cls_summary2, cls_feat2 = self.encoder(view2['img'])
            
            summary1_radio, spatial_features1_radio = self.radio_model_teacher(view1['img'])
            summary2_radio, spatial_features2_radio = self.radio_model_teacher(view2['img'])

            res1["radio_feat"] = spatial_features1_radio.detach()
            res2["radio_feat"] = spatial_features2_radio.detach()
            res1["cls_feat"] = cls_feat1
            res2["cls_feat"] = cls_feat2

            res1["radio_summary"] = summary1_radio.detach()
            res2["radio_summary"] = summary2_radio.detach()
            res1["cls_summary"] = cls_summary1
            res2["cls_summary"] = cls_summary2


            return res1, res2