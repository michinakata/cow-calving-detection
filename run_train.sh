#!/usr/bin/env bash
BATCH_SIZE=32
EPOCH=20
GPU_NO=1
LR=0.001 #学習率
USE_WEIGHT=0 #
PATIENCE=2
MAX_LR_CHANGES=3 #
HIST=0 #ヒストグラム平坦化を利用するか否か
AUG=1 #データオーギュメンテーションの実施可否
PARARELL=0 #学習時にGPU並列化を用いるか否か

#use batchnorm or not
USE_BN=1
TRAIN_FILE_PATH="data/20200116_extractor_train.csv"
VALID_FILE_PATH="data/20200116_extractor_valid.csv"
# OUTPUT_DIR="20200117-adam-bceloss_cooccurloss-${BATCH_SIZE}_hit-${HIST}_initlr-${LR}_use-weight-${USE_WEIGHT}_bn-${USE_BN}"
RESULT_ROOT_DIR="/mnt/iot-qnap3/nakata/cow-calving-detection/cow-calving-detection/20_feature_extraction/multitask_posture_position/result/"
# RESULT_PATH="vit_efficientnet_exp2"
RESULT_PATH="ViT_Pytorch_DenseNet"
# OUTPUT_DIR="sample_model"
# OUTPUT_PATH="output_densenet_and_vit_exp5"
CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size $BATCH_SIZE \
                                             --epoch_size $EPOCH \
                                             --result_path $RESULT_ROOT_DIR/$RESULT_PATH \
                                             --gpu $GPU_NO \
                                             --train_filepath $TRAIN_FILE_PATH \
                                             --valid_filepath $VALID_FILE_PATH \
                                             --lr $LR \
                                             --loss_weight $USE_WEIGHT \
                                             --patience $PATIENCE \
                                             --hist $HIST \
                                             --aug $AUG \
                                             --parallel $PARARELL \
                                             --use_bn $USE_BN \
                                             --max_lr_changes $MAX_LR_CHANGES
