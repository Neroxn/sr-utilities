#!/bin/bash
. ~/anaconda3/etc/profile.d/conda.sh
conda activate dbsr-ESRGAN
export DBSR_ROOT=$PWD
export DATASET_ROOT=$DBSR_ROOT/datasets/dboss_youtube
export HR_FOLDER_INPUT=$DATASET_ROOT/val_images_hr_p002
export HR_FOLDER_OUTPUT=$DATASET_ROOT/val_images_hr_p002
export LR_FOLDER_OUTPUT=$DATASET_ROOT/val_images_lr_p002
export META_INFO_PATH=$DATASET_ROOT/meta_info_val_p002.txt

# # You can use and combine folders as you wish, for now, use only one
# Create LQ dataset from orginal HR folder
python $DBSR_ROOT/.dev_scripts/dataset_formatter/folder_to_lq.py $HR_FOLDER_INPUT $LR_FOLDER_OUTPUT \
    --scale 4 --interpolation_type cubic

export LR_FOLDER_INPUT=$LR_FOLDER_OUTPUT
export LR_FOLDER_OUTPUT=$DATASET_ROOT/val_images_lr_pr
# python $DBSR_ROOT/.dev_scripts/dataset_formatter/folder_to_esrgan.py --input $HR_FOLDER_INPUT --output $HR_FOLDER_OUTPUT --process subimages --crop_size 512 --step 256
# python $DBSR_ROOT/.dev_scripts/dataset_formatter/folder_to_esrgan.py --input $LR_FOLDER_INPUT --output $LR_FOLDER_OUTPUT --process subimages --crop_size 128 --step 64
python $DBSR_ROOT/.dev_scripts/dataset_formatter/folder_to_esrgan.py --meta_input $HR_FOLDER_OUTPUT \
    --process meta --meta_root $DATASET_ROOT --meta_info $META_INFO_PATH

# sed -i "s,dataroot_gt: .*,dataroot_gt: $DATASET_ROOT," "$ESRGAN_REPO_PATH/$ESRGAN_STAGE1_CONFIG"
# sed -i "s,meta_info: .*,meta_info: $META_INFO_PATH," "$ESRGAN_REPO_PATH/$ESRGAN_STAGE1_CONFIG"
# sed -i "s,pretrain_network_g: .*,pretrain_network_g: $ESRGAN_REPO_PATH/experiments/pretrained_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth," "$ESRGAN_REPO_PATH/$ESRGAN_STAGE1_CONFIG" 
