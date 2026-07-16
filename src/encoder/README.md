## ✨Codebase
Our implementation is built upon the following excellent open-source projects:
- https://github.com/NVlabs/RADIO.git  
- https://github.com/naver/mast3r.git  

Our encoder design is inspired by [RADIO](https://github.com/NVlabs/RADIO) and [MASt3R](https://github.com/naver/mast3r):
- RADIO → semantic representation
- MASt3R → geometric modeling
- Ours → unified encoder for spatial reasoning and 3D generation

The code has been modified and extended to support our task and framework. We thank the original authors for their valuable contributions.


## ⚙️ Installation

Please follow the instructions below to install the repo and dependencies.
```bash
git clone https://github.com/fudan-zvg/UniUGG.git
cd UniUGG
```

### Setup the environment

```bash
# Create conda environment
conda create -f envs/train_encoder.yaml
conda activate encoder
```
If you encounter issues during installation, please refer to the official repository and accompanying guidance. The pretrained weights for RADIO and MASt3R can also be downloaded from this repository. The `.path` file should be placed in the `./checkpoints` folder.

## 📂 Dataset & Data preprocessing
The encoder is trained on two types of data:

- **3D data for geometric representation learning**, sourced from DUSt3R.  
  The input consists of preprocessed image pairs. Please refer to the official repository for data download and preprocessing: https://github.com/naver/dust3r.git  

- **Natural images for semantic guidance**, sourced from the LAION and ImageNet datasets. The input consists of single images.

## Training
We provide the commands used for training our models:

```bash
torchrun --nproc_per_node=8 train.py \
    --train_dataset "200_000 @ ScanNetpp(split='train', ROOT='data/scannetpp_processed', resolution=[(512, 384), (384, 384)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 200_000 @ ARKitScenes(split='train',ROOT='data/arkitscenes_processed', resolution=[(512, 384), (384, 384)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 400_000 @ Laion(ROOT='data/laion400m/images', resolution=[(512, 384), (384, 384)], aug_crop='auto', transform=ColorJitter) + 400_000 @ ImageNet(ROOT='imagente-1k/train', resolution=[(512, 384), (384, 384)], aug_crop='auto', transform=ColorJitter)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis', color_loss=False), alpha=0.2, color_loss=False) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean') + 10*SemanticLossFeat(0.9, 0.1)+ 10*SemanticLossSummary(0.9, 0.1)" \
    --train_criterion_nature "10*SemanticLossFeat(0.9,0.1)+ 10*SemanticLossSummary(0.9,0.1)" \
    --pretrained "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 0.2 --epochs 1 --batch_size 1 --accum_iter 2 \
    --save_freq 1 --keep_freq 1 --eval_freq -1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "checkpoints/encoder"
```

## Checkpoints
We provide two encoder checkpoints:

- A geometry-semantic encoder
- A geometry-semantic encoder with an additional color head for color decoding

Model weights are available on [Hugging Face](https://huggingface.co/ming82871/uniugg-encoder/tree/main)
