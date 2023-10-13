#!/bin/bash
. ~/anaconda3/etc/profile.d/conda.sh

conda activate dbsr-ESRGAN
START_TIME=$(date +%s)
export DBSR_ROOT=$PWD
export ESRGAN_MODEL=RealESRGAN_x4plus
export ESRGAN_REPO_PATH=$DBSR_ROOT/Real-ESRGAN
export ESRGAN_OUTPUT=$OUTPUT/ESRGAN
export ESRGAN_STAGE1_CONFIG=options/train_realesrnet_x4plus.yml
# wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -P $ESRGAN_REPO_PATH/experiments/pretrained_models

# # ######################## First Trage Training ######################## 
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 $ESRGAN_REPO_PATH/realesrgan/train.py \
    -opt $ESRGAN_REPO_PATH/$ESRGAN_STAGE1_CONFIG --launcher pytorch
