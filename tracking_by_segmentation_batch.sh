#!/usr/bin/env bash
export FFMPEG_LOG_LEVEL_QUIET=1

DEVICE="cuda:0"
CONFIG_NAME="mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py"
CHECKPOINT_NAME="mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth"
THR=0
BATCH_SIZE=128

DAYS=("20220823")
FARM_ID="233293045097862"
for DAY in ${DAYS[@]}; do

# 実行中の日付を出力
echo $DAY

python3 /mnt/.../tracking_by_segmentation_batch.py \
    --farm_ID $FARM_ID \
    --day $DAY \
    --device $DEVICE \
    --config_name $CONFIG_NAME \
    --checkpoint_name $CHECKPOINT_NAME \
    --thr $THR \
    --batch_size $BATCH_SIZE \

done