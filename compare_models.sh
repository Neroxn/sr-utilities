#!/bin/bash
. ~/anaconda3/etc/profile.d/conda.sh

# source /home/b051/anaconda3/etc/profile.d/conda.sh
export DBSR_ROOT=$PWD
export FILE_FOLDER=$DBSR_ROOT/data/youtube
export FILENAME=a_gt.png
export OUTPUT=$DBSR_ROOT/outputs
export CUDA_VISIBLE_DEVICES=2,3
export LOG_FILE=$DBSR_ROOT/"script_log.txt"
rm $LOG_FILE
echo "The logs will be logged to the $LOG_FILE"
# exec >> "$LOG_FILE" 2>&1
echo "Script started at $(date)" 
# Add or remove using models as you wish

#Linear
# conda activate dbsr-diffbir
# export GT_OUTPUT=$OUTPUT/GT
# mkdir $GT_OUTPUT
# START_TIME=$(date +%s)
# python .dev_scripts/dataset_cropper.py $FILE_FOLDER --resize_value 4 --folder-mode --output-folder $GT_OUTPUT --interpolation cubic
# END_TIME=$(date +%s)
# echo "Time taken for Linear: $(($END_TIME - $START_TIME)) seconds" 

# #ESRGAN
# conda activate dbsr-ESRGAN
# START_TIME=$(date +%s)
# export ESRGAN_MODEL=RealESRGAN_x4plus
# export ESRGAN_REPO_PATH=$DBSR_ROOT/Real-ESRGAN
# export ESRGAN_OUTPUT=$OUTPUT/ESRGAN
# mkdir $ESRGAN_OUTPUT
# python $ESRGAN_REPO_PATH/inference_realesrgan.py -i "$FILE_FOLDER" -o $ESRGAN_OUTPUT -n $ESRGAN_MODEL --suffix real-esrgan --tile 512 

# END_TIME=$(date +%s)
# echo "Time taken for ESRGAN: $(($END_TIME - $START_TIME)) seconds"

# #hat
# conda activate dbsr-HAT
# START_TIME=$(date +%s)
# export HAT_REPO_PATH=$DBSR_ROOT/hat
# export HAT_MODEL_CONFIG=options/test/HAT_GAN_Real_SRx4_script.yml
# export HAT_MODEL_PATH=experiments/pretrained_models/Real_HAT_GAN_SRx4.pth
# export HAT_OUTPUT=$OUTPUT/hat
# mkdir $HAT_OUTPUT
# sed -i "s,dataroot_lq: .*,dataroot_lq: $FILE_FOLDER," "$HAT_REPO_PATH/$HAT_MODEL_CONFIG"
# sed -i "s,pretrain_network_g: .*,pretrain_network_g: $HAT_REPO_PATH/$HAT_MODEL_PATH," "$HAT_REPO_PATH/$HAT_MODEL_CONFIG"
# python $HAT_REPO_PATH/hat/test.py -opt $HAT_REPO_PATH/$HAT_MODEL_CONFIG
# # Get the last folder that is updated from $HAT_REPO_PATH/results
# LAST_UPDATED_VISUALIZATION=$(ls -td "$HAT_REPO_PATH/results"/*/visualization/custom/ | head -n 1)
# # Move it to the $OUTPUT folder
# PREFIX="hat_"

# # Iterate over each image inside the visualization folder
# for img in "$LAST_UPDATED_VISUALIZATION"*
# do
#     # Extract the image filename
#     img_name=$(basename "$img")
    
#     # Move (and rename with a prefix) each image to the $OUTPUT directory
#     mv "$img" "$HAT_OUTPUT/$PREFIX$img_name"

# done
# END_TIME=$(date +%s)
# echo "Time taken for hat: $(($END_TIME - $START_TIME)) seconds"

# #  Run DiffBIR model. Note that this model requires
# #a large amount of GPU space
# conda activate dbsr-diffbir
# START_TIME=$(date +%s)
# export DIFFBIR_PATH=DiffBIR/
# export DIFFBIR_OUTPUT=$OUTPUT/DiffBIR
# mkdir $DIFFBIR_OUTPUT
# python $DIFFBIR_PATH/inference.py --input $DIFFBIR_PATH/inputs/demo/general \
# --config $DIFFBIR_PATH/configs/model/cldm.yaml \
# --ckpt $DIFFBIR_PATH/weights/general_full_v1.ckpt \
# --reload_swinir --swinir_ckpt $DIFFBIR_PATH/weights/general_swinir_v1.ckpt \
# --steps 25 \
# --sr_scale 4 \
# --input $FILE_FOLDER \
# --color_fix_type wavelet \
# --output $DIFFBIR_OUTPUT \
# --device cuda 
# END_TIME=$(date +%s)
# echo "Time taken for diffbir: $(($END_TIME - $START_TIME)) seconds"

# ########################## DAT MODEL ##########################
# conda activate dbsr-dat
# export DAT_REPO_PATH=$DBSR_ROOT/dat
# export DAT_MODEL_CONFIG=options/Test/test_single_x4.yml
# export DAT_MODEL_PATH=experiments/pretrained_models/DAT_x4.pth
# export DAT_OUTPUT=$OUTPUT/dat
# mkdir $DAT_OUTPUT
# sed -i "s,dataroot_lq: .*,dataroot_lq: $FILE_FOLDER," "$DAT_REPO_PATH/$DAT_MODEL_CONFIG"
# sed -i "s,pretrain_network_g: .*,pretrain_network_g: $DAT_REPO_PATH/$DAT_MODEL_PATH," "$DAT_REPO_PATH/$DAT_MODEL_CONFIG"

# START_TIME=$(date +%s)
# python $DAT_REPO_PATH/basicsr/test.py -opt $DAT_REPO_PATH/$DAT_MODEL_CONFIG
# # Get the last folder that is updated from $HAT_REPO_PATH/results
# LAST_UPDATED_VISUALIZATION=$(ls -td "$DAT_REPO_PATH/results"/*/visualization/Single/ | head -n 1)
# # Move it to the $OUTPUT folder
# PREFIX="dat_"

# # Iterate over each image inside the visualization folder
# for img in "$LAST_UPDATED_VISUALIZATION"*
# do
#     # Extract the image filename
#     img_name=$(basename "$img")
    
#     # Move (and rename with a prefix) each image to the $OUTPUT directory
#     mv "$img" "$DAT_OUTPUT/$PREFIX$img_name"

# done
# END_TIME=$(date +%s)
# echo "Time taken for dat: $(($END_TIME - $START_TIME)) seconds"
########################## RealBasicVSR Model ##########################
# RealBasicVSR is a video model so it requires images/video frames with same size
conda activate dbsr-realbasicvsr
export REALBASICVSR_REPO_PATH=$DBSR_ROOT/RealBasicVSR
export REALBASICVSR_CONFIG=$REALBASICVSR_REPO_PATH/configs/realbasicvsr_x4.py
export REALBASICVSR_MODEL_PATH=$REALBASICVSR_REPO_PATH/checkpoints/RealBasicVSR_x4.pth
export REALBASICVSR_OUTPUT=$OUTPUT/RealBasicVSR
mkdir $REALBASICVSR_OUTPUT

# for img in "$FILE_FOLDER"/*
# do
#     img_name=$(basename "$img")
#     python $REALBASICVSR_REPO_PATH/inference_realbasicvsr.py $REALBASICVSR_CONFIG $REALBASICVSR_MODEL_PATH $img $REALBASICVSR_OUTPUT/realbasicvsr_$img_name
# done

python $REALBASICVSR_REPO_PATH/inference_realbasicvsr.py $REALBASICVSR_CONFIG $REALBASICVSR_MODEL_PATH $FILE_FOLDER $OUTPUT/RealBasicVSR --max_seq_len 8


echo "Script ended at $(date)"
