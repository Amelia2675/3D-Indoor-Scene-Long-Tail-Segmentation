#!/bin/bash

# Exit script when a command returns nonzero state
export PYTHONUNBUFFERED="True"

export BATCH_SIZE=$1
export POSTFIX=$2
export ARGS=$3

export MODEL=Res16UNet34D
export DATASET=Scannet200Textual2cmDataset

export DATA_ROOT="./dataset/"
export OUTPUT_DIR_ROOT="./output/"
# export PRETRAINED_WEIGHTS="<path_to_ckpt_file>"
# export RESUME_DIR="<path_to_ckpt_dir>"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$POSTFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

CUDA_VISIBLE_DEVICES="0, 1" python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --train_limit_numpoints 1400000 \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --num_gpu 2 \
    --balanced_category_sampling False \
    --use_embedding_loss True \
    --max_epoch 200 \
    $ARGS \
    2>&1 | tee -a "$LOG"

#    --resume $RESUME_DIR \
#    --weights $PRETRAINED_WEIGHTS \
