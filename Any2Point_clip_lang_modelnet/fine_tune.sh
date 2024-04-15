#!/bin/bash
PATH_1=./ckpts/ViT-L-14.pt
FORMAL_DIR=try

pretrained_path="$PATH_1"
formal_dirs="$FORMAL_DIR"

python main.py --config cfgs/finetune_modelnet.yaml --coef_pro 0.5 --scale_factor 0.3 --coef_1dgird 2 --coef_3dgird 0.16 --adapter_dim 16 --patchknn 64 --pos_cor 1.0 -1.0 0.0 --finetune_model --exp_name $FORMAL_DIR  --ckpts $PATH_1 >./log/try.log 2>&1 &
