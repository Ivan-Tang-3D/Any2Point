#!/bin/sh
PATH_1=./ckpts/dinov2_vitb14_pretrain.pth
FORMAL_DIR=try


pretrained_path="$PATH_1"
mode="finetune_encoder"
formal_dirs="$FORMAL_DIR"

python examples/classification/main.py --cfg ./cfgs/scanobjectnn/Any2Point.yaml --num_view 6 --trans=-1.8 --patchknn 64 --lastdim 32 --coef_pro 0.7 --coef_2dgird 26 --scale_factor 0.3  --coef_3dgird 0.16 --attn2d_dim 12 --formal_dirs $formal_dirs --mode $mode --pretrained_path $pretrained_path >./log/try.log 2>&1 &