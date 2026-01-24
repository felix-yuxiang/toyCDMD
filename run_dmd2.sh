#!/bin/bash

# Run DMD2 (Distribution Matching Distillation 2) training
# Requires a pre-trained flow matching teacher model

DATASET=${1:-spiral}
STUDENT_STEPS=${2:-1}
EPOCHS=${3:-100}
OUTPUT_DIR=${4:-./models}

python toydmd2.py \
    --teacher_checkpoint ${OUTPUT_DIR}/flow_model_${DATASET}_epoch_100.pt \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size 4096 \
    --lr_student 1e-4 \
    --lr_proxy 1e-4 \
    --h_dim 128 \
    --student_steps $STUDENT_STEPS \
    --proxy_update_ratio 10 \
    --epoch_save_freq 10 \
    --output_dir $OUTPUT_DIR \
    --num_points 4000




# python toydmd2.py --visualize_only --dataset spiral --epochs 100 --student_steps 1 --vis_epoch_step 20
