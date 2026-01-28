#!/bin/bash

# Debug Visualization Script
# Goal: Verify that images (reconstruction vs original) are uploaded to WandB
OUTPUT_DIR="output/debug_viz"
mkdir -p $OUTPUT_DIR

echo "Starting Visualization Debug on GPU 1"

CUDA_VISIBLE_DEVICES=1 python3 main.py \
  --config configs/pretrain.yaml \
  --tiny \
  --opts \
    model.pretrain_tasks=['reconstruction','feature_pred'] \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-debug-viz" \
    epochs=1 \
    val_freq_split=5 \
    dataset.batch_size=4

echo "Viz Debug Finished. Check WandB!"
