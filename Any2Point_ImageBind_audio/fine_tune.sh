#!/bin/bash
PATH_1=./ckpts/imagebind_audio.pth
FORMAL_DIR=try

pretrained_path="$PATH_1"
mode="finetune_encoder"
formal_dirs="$FORMAL_DIR"

python examples/classification/main.py --cfg ./cfgs/scanobjectnn/Any2point.yaml --coef_pro 0.5 --coef_2dgird 16 --coef_3dgird 0.16 --attn2d_dim 12 --adapter_dim 12 --trans=-1.4 --patchknn 64 --scale_factor 0.3 --formal_dirs $formal_dirs --mode $mode --pretrained_path $pretrained_path >./log/try.log 2>&1 &