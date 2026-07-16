# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of MASt3R training losses
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_msssim
import lpips
from sklearn.metrics import average_precision_score

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.losses import BaseCriterion, Criterion, MultiLoss, Sum, ConfLoss
from dust3r.losses import Regr3D as Regr3D_dust3r
from dust3r.utils.geometry import (geotrf, inv, normalize_pointcloud, geotrf_xyzrgb)
from dust3r.inference import get_pred_pts3d
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale


lpips_loss_fn = lpips.LPIPS(net='alex')
lpips_loss_fn.eval()

def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz

def normalize_to_clip(rgb_pred):
    mean = torch.tensor([0.4815, 0.4578, 0.4082], device=rgb_pred.device)[None,  None, None,:]
    std = torch.tensor([0.2686, 0.2613, 0.2758], device=rgb_pred.device)[None, None, None,:]
    return rgb_pred * std + mean 

class Regr3D (Regr3D_dust3r):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, opt_fit_gt=False,
                 sky_loss_value=2, max_metric_scale=False, loss_in_log=False, color_loss = False):
        self.loss_in_log = loss_in_log
        if norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        super().__init__(criterion, self.norm_mode, gt_scale)

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale
        self.color_loss = color_loss

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf_xyzrgb(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf_xyzrgb(in_camera1, gt2['pts3d'])  # B,H,W,3

        gt_pts1[...,3:6] = normalize_to_clip(gt_pts1[...,3:6])
        gt_pts2[...,3:6] = normalize_to_clip(gt_pts2[...,3:6])

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            # dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            # dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            dis1 = gt_pts1[...,:3].norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2[...,:3].norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        if self.loss_in_log == 'before':
            # this only make sense when depth_mode == 'linear'
            gt_pts1[...,:3] = apply_log_to_norm(gt_pts1[...,:3])
            gt_pts2[...,:3] = apply_log_to_norm(gt_pts2[...,:3])

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()

        if not self.norm_all:
            if self.max_metric_scale:
                B = valid1.shape[0]
                # valid1: B, H, W
                # torch.linalg.norm(gt_pts1, dim=-1) -> B, H, W
                # dist1_to_cam1 -> reshape to B, H*W
                dist1_to_cam1 = torch.where(valid1, torch.linalg.norm(gt_pts1[...,:3], dim=-1), 0).view(B, -1)
                dist2_to_cam1 = torch.where(valid2, torch.linalg.norm(gt_pts2[...,:3], dim=-1), 0).view(B, -1)

                # is_metric_scale: B
                # dist1_to_cam1.max(dim=-1).values -> B
                gt1['is_metric_scale'] = gt1['is_metric_scale'] \
                    & (dist1_to_cam1.max(dim=-1).values < self.max_metric_scale) \
                    & (dist2_to_cam1.max(dim=-1).values < self.max_metric_scale)
                gt2['is_metric_scale'] = gt1['is_metric_scale']

            mask = ~gt1['is_metric_scale']
        else:
            mask = torch.ones_like(gt1['is_metric_scale'])
        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts1[mask], pr_pts2[mask] = normalize_pointcloud(pr_pts1[mask], pr_pts2[mask], self.norm_mode,
                                                                valid1[mask], valid2[mask])

        if self.norm_mode and not self.gt_scale:
            gt_pts1[...,:3], gt_pts2[...,:3], norm_factor = normalize_pointcloud(gt_pts1[...,:3], gt_pts2[...,:3], self.norm_mode,
                                                                 valid1, valid2, ret_factor=True)
            # apply the same normalization to prediction
            pr_pts1[~mask] = pr_pts1[~mask] / norm_factor[~mask]
            pr_pts2[~mask] = pr_pts2[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1['sky_mask'] & (~valid1)
        sky2 = gt2['sky_mask'] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts_rgb1, gt_pts_rgb2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)

        color_rgb1 = pred1["color_rgb"].clone()
        color_rgb2 = pred2["color_rgb"].clone()

        gt_pts1 = gt_pts_rgb1[...,:3]
        gt_pts2 = gt_pts_rgb2[...,:3]
        gt_rgb1 = gt_pts_rgb1[...,3:6]
        gt_rgb2 = gt_pts_rgb2[...,3:6]

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # add the sky pixel as "valid" pixels...
            mask1 = mask1 | sky1
            mask2 = mask2 | sky2

        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        if self.loss_in_log and self.loss_in_log != 'before':
            # this only make sense when depth_mode == 'exp'
            pred_pts1 = apply_log_to_norm(pred_pts1)
            gt_pts1 = apply_log_to_norm(gt_pts1)
        l1 = self.criterion(pred_pts1, gt_pts1)

        # loss on gt2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        if self.loss_in_log and self.loss_in_log != 'before':
            pred_pts2 = apply_log_to_norm(pred_pts2)
            gt_pts2 = apply_log_to_norm(gt_pts2)
        l2 = self.criterion(pred_pts2, gt_pts2)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # ... but force the loss to be high there
            l1 = torch.where(sky1[mask1], self.sky_loss_value, l1)
            l2 = torch.where(sky2[mask2], self.sky_loss_value, l2)
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        
        if not self.color_loss:
            return Sum((l1, mask1), (l2, mask2)), (details | monitoring)
        
        else:
            l1_loss1 = F.l1_loss(color_rgb1, gt_rgb1)
            l1_loss2 = F.l1_loss(color_rgb2, gt_rgb2)
            ssim_loss1 = 1 - pytorch_msssim.ssim(color_rgb1, gt_rgb1, data_range=1.0, size_average=True)
            ssim_loss2 = 1 - pytorch_msssim.ssim(color_rgb2, gt_rgb2, data_range=1.0, size_average=True)
            lpips_loss1 = lpips_loss_fn.to(color_rgb1.device)(color_rgb1.permute(0,3,1,2), gt_rgb1.permute(0,3,1,2)).mean()
            lpips_loss2 = lpips_loss_fn.to(color_rgb2.device)(color_rgb2.permute(0,3,1,2), gt_rgb2.permute(0,3,1,2)).mean()
            l1_rgb = l1_loss1*1.0 + ssim_loss1*0.5 + lpips_loss1*0.2
            l2_rgb = l1_loss2*1.0 + ssim_loss2*0.5 + lpips_loss2*0.2

            details[self_name + "_rgb_1"] = float(l1_rgb)
            details[self_name + "_rgb_2"] = float(l2_rgb)
            
            return Sum((l1, mask1), (l2, mask2)), (l1_rgb, l2_rgb), (details | monitoring)

class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)
        gt_pts1 = gt_pts1[...,:3]
        gt_pts2 = gt_pts2[...,:3]

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth scale.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)
        
        gt_pts1 = gt_pts1[...,:3]
        gt_pts2 = gt_pts2[...,:3]
    
        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass


def get_similarities(desc1, desc2, euc=False):
    if euc:  # euclidean distance in same range than similarities
        dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
        sim = 1 / (1 + dists)
    else:
        # Compute similarities
        sim = desc1 @ desc2.transpose(-2, -1)
    return sim


class MatchingCriterion(BaseCriterion):
    def __init__(self, reduction='mean', fp=torch.float32):
        super().__init__(reduction)
        self.fp = fp

    def forward(self, a, b, valid_matches=None, euc=False):
        assert a.ndim >= 2 and 1 <= a.shape[-1], f'Bad shape = {a.shape}'
        dist = self.loss(a.to(self.fp), b.to(self.fp), valid_matches, euc=euc)
        # one dimension less or reduction to single value
        assert (valid_matches is None and dist.ndim == a.ndim -
                1) or self.reduction in ['mean', 'sum', '1-mean', 'none']
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        if self.reduction == '1-mean':
            return 1. - dist.mean() if dist.numel() > 0 else dist.new_ones(())
        raise ValueError(f'bad {self.reduction=} mode')

    def loss(self, a, b, valid_matches=None):
        raise NotImplementedError


class InfoNCE(MatchingCriterion):
    def __init__(self, temperature=0.07, eps=1e-8, mode='all', **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.eps = eps
        assert mode in ['all', 'proper', 'dual']
        self.mode = mode

    def loss(self, desc1, desc2, valid_matches=None, euc=False):
        # valid positives are along diagonals
        B, N, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2
        if valid_matches is None:
            valid_matches = torch.ones([B, N], dtype=bool)
        # torch.all(valid_matches.sum(dim=-1) > 0) some pairs have no matches????
        # print("#"*100)
        # print(B)
        # print(N)
        # print(valid_matches.shape)
        # print(desc1.shape)
        # print(valid_matches.sum())
        # print(valid_matches.shape == torch.Size([B, N]))
        # print(valid_matches.sum() > 0)
        # print(valid_matches.shape == torch.Size([B, N]) and valid_matches.sum() > 0)
        # print("#"*100)

        # assert valid_matches.shape == torch.Size([B, N]) and valid_matches.sum() > 0

        # Tempered similarities
        sim = get_similarities(desc1, desc2, euc) / self.temperature
        sim[sim.isnan()] = -torch.inf  # ignore nans
        # Softmax of positives with temperature
        sim = sim.exp_()  # save peak memory
        positives = sim.diagonal(dim1=-2, dim2=-1)

        # Loss
        if self.mode == 'all':            # Previous InfoNCE
            loss = -torch.log((positives / sim.sum(dim=-1).sum(dim=-1, keepdim=True)).clip(self.eps))
        elif self.mode == 'proper':  # Proper InfoNCE
            loss = -(torch.log((positives / sim.sum(dim=-2)).clip(self.eps)) +
                     torch.log((positives / sim.sum(dim=-1)).clip(self.eps)))
        elif self.mode == 'dual':  # Dual Softmax
            loss = -(torch.log((positives**2 / sim.sum(dim=-1) / sim.sum(dim=-2)).clip(self.eps)))
        else:
            raise ValueError("This should not happen...")
        return loss[valid_matches]


class APLoss (MatchingCriterion):
    """ AP loss.
    """

    def __init__(self, nq='torch', min=0, max=1, euc=False, **kw):
        super().__init__(**kw)
        # Exact/True AP loss (not differentiable)
        if nq == 0:
            nq = 'sklearn'  # special case
        try:
            self.compute_AP = eval('self.compute_true_AP_' + nq)
        except:
            raise ValueError("Unknown mode %s for AP loss" % nq)

    @staticmethod
    def compute_true_AP_sklearn(scores, labels):
        def compute_AP(label, score):
            return average_precision_score(label, score)

        aps = scores.new_zeros((scores.shape[0], scores.shape[1]))
        label_np = labels.cpu().numpy().astype(bool)
        scores_np = scores.cpu().numpy()
        for bi in range(scores_np.shape[0]):
            for i in range(scores_np.shape[1]):
                labels = label_np[bi, i, :]
                if labels.sum() < 1:
                    continue
                aps[bi, i] = compute_AP(labels, scores_np[bi, i, :])
        return aps

    @staticmethod
    def compute_true_AP_torch(scores, labels):
        assert scores.shape == labels.shape
        B, N, M = labels.shape
        dev = labels.device
        with torch.no_grad():
            # sort scores
            _, order = scores.sort(dim=-1, descending=True)
            # sort labels accordingly
            labels = labels[torch.arange(B, device=dev)[:, None, None].expand(order.shape),
                            torch.arange(N, device=dev)[None, :, None].expand(order.shape),
                            order]
            # compute number of positives per query
            npos = labels.sum(dim=-1)
            assert torch.all(torch.isclose(npos, npos[0, 0])
                             ), "only implemented for constant number of positives per query"
            npos = int(npos[0, 0])
            # compute precision at each recall point
            posrank = labels.nonzero()[:, -1].view(B, N, npos)
            recall = torch.arange(1, 1 + npos, dtype=torch.float32, device=dev)[None, None, :].expand(B, N, npos)
            precision = recall / (1 + posrank).float()
            # average precision values at all recall points
            aps = precision.mean(dim=-1)

        return aps

    def loss(self, desc1, desc2, valid_matches=None, euc=False):  # if matches is None, positives are the diagonal
        B, N1, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2

        scores = get_similarities(desc1, desc2, euc)

        labels = torch.zeros([B, N1, N2], dtype=scores.dtype, device=scores.device)

        # allow all diagonal positives and only mask afterwards
        labels.diagonal(dim1=-2, dim2=-1)[...] = 1.
        apscore = self.compute_AP(scores, labels)
        if valid_matches is not None:
            apscore = apscore[valid_matches]
        return apscore


class MatchingLoss (Criterion, MultiLoss):
    """ 
    Matching loss per image 
    only compare pixels inside an image but not in the whole batch as what would be done usually
    """

    def __init__(self, criterion, withconf=False, use_pts3d=False, negatives_padding=0, blocksize=4096):
        super().__init__(criterion)
        self.negatives_padding = negatives_padding
        self.use_pts3d = use_pts3d
        self.blocksize = blocksize
        self.withconf = withconf

    def add_negatives(self, outdesc2, desc2, batchid, x2, y2):
        if self.negatives_padding:
            B, H, W, D = desc2.shape
            negatives = torch.ones([B, H, W], device=desc2.device, dtype=bool)
            negatives[batchid, y2, x2] = False
            sel = negatives & (negatives.view([B, -1]).cumsum(dim=-1).view(B, H, W)
                               <= self.negatives_padding)  # take the N-first negatives
            outdesc2 = torch.cat([outdesc2, desc2[sel].view([B, -1, D])], dim=1)
        return outdesc2

    def get_confs(self, pred1, pred2, sel1, sel2):
        if self.withconf:
            if self.use_pts3d:
                outconfs1 = pred1['conf'][sel1]
                outconfs2 = pred2['conf'][sel2]
            else:
                outconfs1 = pred1['desc_conf'][sel1]
                outconfs2 = pred2['desc_conf'][sel2]
        else:
            outconfs1 = outconfs2 = None
        return outconfs1, outconfs2

    def get_descs(self, pred1, pred2):
        if self.use_pts3d:
            desc1, desc2 = pred1['pts3d'], pred2['pts3d_in_other_view']
        else:
            desc1, desc2 = pred1['desc'], pred2['desc']
        return desc1, desc2

    def get_matching_descs(self, gt1, gt2, pred1, pred2, **kw):
        outdesc1 = outdesc2 = outconfs1 = outconfs2 = None
        # Recover descs, GT corres and valid mask
        desc1, desc2 = self.get_descs(pred1, pred2)

        (x1, y1), (x2, y2) = gt1['corres'].unbind(-1), gt2['corres'].unbind(-1)
        valid_matches = gt1['valid_corres']

        # Select descs that have GT matches
        B, N = x1.shape
        batchid = torch.arange(B)[:, None].repeat(1, N)  # B, N
        outdesc1, outdesc2 = desc1[batchid, y1, x1], desc2[batchid, y2, x2]  # B, N, D

        # Padd with unused negatives
        outdesc2 = self.add_negatives(outdesc2, desc2, batchid, x2, y2)

        # Gather confs if needed
        sel1 = batchid, y1, x1
        sel2 = batchid, y2, x2
        outconfs1, outconfs2 = self.get_confs(pred1, pred2, sel1, sel2)

        return outdesc1, outdesc2, outconfs1, outconfs2, valid_matches, {'use_euclidean_dist': self.use_pts3d}

    def blockwise_criterion(self, descs1, descs2, confs1, confs2, valid_matches, euc, rng=np.random, shuffle=True):
        loss = None
        details = {}
        B, N, D = descs1.shape

        if N <= self.blocksize:  # Blocks are larger than provided descs, compute regular loss
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
        else:  # Compute criterion on the blockdiagonal only, after shuffling
            # Shuffle if necessary
            matches_perm = slice(None)
            if shuffle:
                matches_perm = np.stack([rng.choice(range(N), size=N, replace=False) for _ in range(B)])
                batchid = torch.tile(torch.arange(B), (N, 1)).T
                matches_perm = batchid, matches_perm

            descs1 = descs1[matches_perm]
            descs2 = descs2[matches_perm]
            valid_matches = valid_matches[matches_perm]

            assert N % self.blocksize == 0, "Error, can't chunk block-diagonal, please check blocksize"
            n_chunks = N // self.blocksize
            descs1 = descs1.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            descs2 = descs2.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            valid_matches = valid_matches.view([B * n_chunks, self.blocksize])
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
            if self.withconf:
                confs1, confs2 = map(lambda x: x[matches_perm], (confs1, confs2))  # apply perm to confidences if needed

        if self.withconf:
            # split confidences between positives/negatives for loss computation
            details['conf_pos'] = map(lambda x: x[valid_matches.view(B, -1)], (confs1, confs2))
            details['conf_neg'] = map(lambda x: x[~valid_matches.view(B, -1)], (confs1, confs2))
            details['Conf1_std'] = confs1.std()
            details['Conf2_std'] = confs2.std()

        return loss, details

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # Gather preds and GT
        descs1, descs2, confs1, confs2, valid_matches, monitoring = self.get_matching_descs(
            gt1, gt2, pred1, pred2, **kw)

        # loss on matches
        loss, details = self.blockwise_criterion(descs1, descs2, confs1, confs2,
                                                 valid_matches, euc=monitoring.pop('use_euclidean_dist', False))

        details[type(self).__name__] = float(loss.mean())
        return loss, (details | monitoring)


class ConfMatchingLoss(ConfLoss):
    """ Weight matching by learned confidence. Same as ConfLoss but for a matching criterion
        Assuming the input matching_loss is a match-level loss.
    """

    def __init__(self, pixel_loss, alpha=1., confmode='prod', neg_conf_loss_quantile=False):
        super().__init__(pixel_loss, alpha)
        self.pixel_loss.withconf = True
        self.confmode = confmode
        self.neg_conf_loss_quantile = neg_conf_loss_quantile

    def aggregate_confs(self, confs1, confs2):  # get the confidences resulting from the two view predictions
        if self.confmode == 'prod':
            confs = confs1 * confs2 if confs1 is not None and confs2 is not None else 1.
        elif self.confmode == 'mean':
            confs = .5 * (confs1 + confs2) if confs1 is not None and confs2 is not None else 1.
        else:
            raise ValueError(f"Unknown conf mode {self.confmode}")
        return confs

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        loss, details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        # Recover confidences for positive and negative samples
        conf1_pos, conf2_pos = details.pop('conf_pos')
        conf1_neg, conf2_neg = details.pop('conf_neg')
        conf_pos = self.aggregate_confs(conf1_pos, conf2_pos)

        # weight Matching loss by confidence on positives
        conf_pos, log_conf_pos = self.get_conf_log(conf_pos)
        conf_loss = loss * conf_pos - self.alpha * log_conf_pos
        # average + nan protection (in case of no valid pixels at all)
        conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
        # Add negative confs loss to give some supervision signal to confidences for pixels that are not matched in GT
        if self.neg_conf_loss_quantile:
            conf_neg = torch.cat([conf1_neg, conf2_neg])
            conf_neg, log_conf_neg = self.get_conf_log(conf_neg)

            # recover quantile that will be used for negatives loss value assignment
            neg_loss_value = torch.quantile(loss, self.neg_conf_loss_quantile).detach()
            neg_loss = neg_loss_value * conf_neg - self.alpha * log_conf_neg

            neg_loss = neg_loss.mean() if neg_loss.numel() > 0 else 0
            conf_loss = conf_loss + neg_loss

        return conf_loss, dict(matching_conf_loss=float(conf_loss), **details)

class SemanticLossFeat(MultiLoss):
    def __init__(self, loss_weight_alpha, loss_weight_beta, random_choose = True):
        super().__init__()
        self.loss_weight_alpha = loss_weight_alpha
        self.loss_weight_beta = loss_weight_beta
        self.random_choose = random_choose
        self.num_sample = 256 

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # Select a subset of features for loss calculation
        student_token1 = pred1["cls_feat"]
        student_token2 = pred2["cls_feat"]
        student_token = torch.cat([student_token1,student_token2], dim=0)
        teacher_token1 = pred1["radio_feat"]
        teacher_token2 = pred2["radio_feat"]
        teacher_token = torch.cat([teacher_token1,teacher_token2], dim=0)

        if self.random_choose:
            batch_size, num_tokens, feature_dim = student_token.shape[0], student_token.shape[1], student_token.shape[2]
            self.num_sample = int((256 / 768) * num_tokens)
            # subset_indices = torch.randint(0, num_tokens, (batch_size, num_tokens // 2))
            subset_indices = torch.stack([torch.randperm(num_tokens)[:self.num_sample] for _ in range(batch_size)])
            subset_indices = subset_indices.unsqueeze(-1).expand(-1, -1, feature_dim).cuda()
            student_token = torch.gather(student_token, dim=1, index=subset_indices)
            teacher_token = torch.gather(teacher_token, dim=1, index=subset_indices)

        # Cosine similarity lossstudent_token
        cosine_loss = 1 - F.cosine_similarity(student_token, teacher_token, dim=-1).mean()

        # Smooth L1 loss
        l1_loss = F.smooth_l1_loss(student_token, teacher_token)

        # Combined loss
        semantic_loss = self.loss_weight_alpha * cosine_loss + self.loss_weight_beta * l1_loss
        return semantic_loss, dict(feat_semantic_loss=float(semantic_loss))
    

class SemanticLossSummary(MultiLoss):
    def __init__(self, loss_weight_alpha, loss_weight_beta):
        super().__init__()
        self.loss_weight_alpha = loss_weight_alpha
        self.loss_weight_beta = loss_weight_beta

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # Select a subset of features for loss calculation
        student_token1 = pred1["cls_summary"]
        student_token2 = pred2["cls_summary"]
        student_token = torch.cat([student_token1,student_token2], dim=0)
        teacher_token1 = pred1["radio_summary"]
        teacher_token2 = pred2["radio_summary"]
        teacher_token = torch.cat([teacher_token1,teacher_token2], dim=0)

        # Cosine similarity loss
        cosine_loss = 1 - F.cosine_similarity(student_token, teacher_token, dim=-1).mean()

        # Smooth L1 loss
        l1_loss = F.smooth_l1_loss(student_token, teacher_token)

        # Combined loss
        semantic_loss = self.loss_weight_alpha * cosine_loss + self.loss_weight_beta * l1_loss
        return semantic_loss, dict(summary_semantic_loss=float(semantic_loss))
