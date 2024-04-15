#!/bin/bash
PATH_1=./ckpts/dinov2_vitb14_pretrain.pth
FORMAL_DIR=try

pretrained_path="$PATH_1"
formal_dirs="$FORMAL_DIR"

srun -p optimal --quotatype=spot --gres=gpu:1 -J LLM python main.py --config cfgs/finetune_modelnet.yaml --coef_pro 0.1 --scale_factor 1.0 --coef_2dgird 26 --coef_3dgird 0.16 --adapter_dim 8 --patchknn 81 --trans=-1.4 --finetune_model --exp_name $FORMAL_DIR  --ckpts $PATH_1 >./log/try.log 2>&1 &