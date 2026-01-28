#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# Baseline: Reconstruction Only
OUTPUT_DIR="output/baseline_recon"
mkdir -p $OUTPUT_DIR

echo "Starting Baseline: Reconstruction Only"
python3 main.py \
  --config configs/pretrain.yaml \
  --opts \
    model.pretrain_tasks=['reconstruction'] \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-baseline"

echo "Baseline Finished."
