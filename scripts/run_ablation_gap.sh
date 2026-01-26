#!/bin/bash

# Ablation A: Multi-task with GAP Strategy
OUTPUT_DIR="output/ablation_gap"
mkdir -p $OUTPUT_DIR

echo "Starting Ablation A: Multi-task (GAP)"
python3 main.py \
  --config configs/pretrain.yaml \
  --opts \
    model.feature_token_type='gap' \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-ablation"

echo "Ablation A Finished."
