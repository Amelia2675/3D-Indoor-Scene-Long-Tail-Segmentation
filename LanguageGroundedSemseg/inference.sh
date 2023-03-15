#!/bin/bash

# Exit script when a command returns nonzero state
export PYTHONUNBUFFERED="True"

export BATCH_SIZE=1
export POSTFIX=$1
export ARGS=$2

export MODEL=Res16UNet34D
export DATASET=Scannet200Voxelization2cmDataset

export DATA_ROOT="./dataset/"
export OUTPUT_DIR_ROOT="./output/"
export PRETRAINED_WEIGHTS="./checkpoint_best.ckpt"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$POSTFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --visualize True \
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 1 \
    --balanced_category_sampling True \
    --is_train False \
    --weights $PRETRAINED_WEIGHTS \
    --val_phase test \
    $ARGS \
    2>&1 | tee -a "$LOG"
