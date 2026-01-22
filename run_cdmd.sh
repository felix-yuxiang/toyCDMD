#!/bin/bash

# Run CDMD (Consistency Distribution Matching Distillation) training
# Requires a pre-trained flow matching teacher model

DATASET=${1:-spiral}
STUDENT_STEPS=${2:-1}
EPOCHS=${3:-100}
OUTPUT_DIR=${4:-./models}

python toycdmd.py \
    --teacher_checkpoint ${OUTPUT_DIR}/flow_model_${DATASET}_epoch_100.pt \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size 2048 \
    --lr_student 1e-4 \
    --h_dim 128 \
    --student_steps $STUDENT_STEPS \
    --epoch_save_freq 10 \
    --output_dir $OUTPUT_DIR \
    --num_points 4000


# Example visualization only:
# python toycdmd.py --visualize_only --dataset spiral --epochs 100 --student_steps 1 --vis_epoch_step 20
