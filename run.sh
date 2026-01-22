#!/bin/bash

# Run flow matching model on 2D toy dataset
# Available datasets: checkerboard, crescent, spiral

DATASET=${1:-checkerboard}
EPOCHS=${2:-60}
OUTPUT_DIR=${3:-./models/}

python toyfm.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size 4096 \
    --lr 0.003 \
    --h_dim 128 \
    --num_points 8000 \
    --epoch_save_freq 10 \
    --num_steps 100 \
    --output_dir $OUTPUT_DIR