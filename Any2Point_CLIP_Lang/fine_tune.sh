#!/bin/bash
PATH_1=./ckpts/ViT-L-14.pt
FORMAL_DIR=try

pretrained_path="$PATH_1"
mode="finetune_encoder"
formal_dirs="$FORMAL_DIR"

python examples/classification/main.py --num_view 6 --pos_cor -1.0 0.0 1.0 --coef_pro 0.3 --coef_1dgird 2 --coef_3dgird 0.08 --scale_factor 0.3 --attn1d_dim 12 --formal_dirs $formal_dirs --mode $mode --pretrained_path $pretrained_path --cfg ./cfgs/scanobjectnn/Any2Point.yaml >./log/try.log 2>&1 & 
