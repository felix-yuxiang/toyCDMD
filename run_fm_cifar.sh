#!/bin/bash

# Run Flow Matching training on CIFAR-10

EPOCHS=${1:-100}
BATCH_SIZE=${2:-128}
LR=${3:-1e-4}
CHECKPOINT_DIR=${4:-./checkpoints/fm_cifar10}
DATA_DIR=${5:-./data}

python fm_cifar10.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --data_dir $DATA_DIR \
    --save_freq 10 \
    --log_freq 100 \
    --num_workers 4 \
    --num_samples 64 \
    --num_steps 100 \
    --output_image ./imgs/fm_cifar10_samples.png


# Example usage:
# ./run_fm_cifar.sh                          # Run with defaults
# ./run_fm_cifar.sh 200                      # Train for 200 epochs
# ./run_fm_cifar.sh 100 256 2e-4             # Custom epochs, batch size, and learning rate

# To resume training:
# python fm_cifar10.py --resume ./checkpoints/fm_cifar10/fm_cifar10_epoch_50.pt --epochs 100

# To generate samples only:
# python fm_cifar10.py --generate_only --resume ./checkpoints/fm_cifar10/fm_cifar10_epoch_100.pt --num_samples 64 --num_steps 100
