#!/bin/bash
. ~/anaconda3/etc/profile.d/conda.sh
export DBSR_ROOT=$PWD
export HAT_MODEL=RealESRGAN_x4plus
export HAT_REPO_PATH=$DBSR_ROOT/hat
export HAT_TRAIN_CONFIG=$HAT_REPO_PATH/options/train/train_custom.yaml
conda activate dbsr-HAT
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 \
    $HAT_REPO_PATH/hat/train.py -opt $HAT_TRAIN_CONFIG --launcher pytorch
