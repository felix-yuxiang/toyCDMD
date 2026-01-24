#!/bin/bash

# Run Cached CDMD (Consistency Distribution Matching Distillation) training
# Uses trajectory caching for more efficient training
# Requires a pre-trained flow matching teacher model

DATASET=${1:-spiral}
STUDENT_STEPS=${2:-1}
EPOCHS=${3:-100}
OUTPUT_DIR=${4:-./models}

# Cached CDMD specific parameters
NUM_TRAJECTORIES=${5:-1024}    # K: Number of trajectories in cache
MAX_STEPS=${6:-10}             # T: Maximum steps per trajectory
MAX_LIFESPAN=${7:-5}           # J_max: Maximum lifespan before reset

python toycdmd_cached.py \
    --teacher_checkpoint ${OUTPUT_DIR}/flow_model_${DATASET}_epoch_100.pt \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size 256 \
    --lr_student 1e-4 \
    --h_dim 128 \
    --student_steps $STUDENT_STEPS \
    --num_trajectories $NUM_TRAJECTORIES \
    --max_steps $MAX_STEPS \
    --max_lifespan $MAX_LIFESPAN \
    --epoch_save_freq 10 \
    --output_dir $OUTPUT_DIR \
    --num_points 4000


# Example usage:
# ./run_cdmd_cached.sh spiral 1 100 ./models 1024 10 5
