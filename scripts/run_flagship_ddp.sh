#!/bin/bash

# Flagship Experiment: Multi-task with Cross-Attention (Single Token) on 4 GPUs
# Optimized for 4x A800 GPUs, 20 Epochs

OUTPUT_DIR="output/flagship_ddp"
mkdir -p $OUTPUT_DIR

echo "Starting Flagship Experiment (DDP)"
echo "Using 4 GPUs (0,1,2,3). Batch Size: 128 per GPU -> 512 Total."

# Set CUDA_VISIBLE_DEVICES to ensure we use the right cards
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Use torchrun for distributed launch
# Note: val_freq_split=10 ensures frequent validation for monitoring
torchrun --nproc_per_node=4 --master_port=29501 main.py \
  --config configs/pretrain.yaml \
  --opts \
    model.pretrain_tasks=['reconstruction','feature_pred'] \
    model.feature_token_type='cross_attn' \
    model.feature_token_strategy='single' \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-flagship-ddp" \
    epochs=20 \
    dataset.batch_size=128 \
    optimizer.lr=1e-3 \
    val_freq_split=10

echo "Flagship DDP Experiment Finished."
