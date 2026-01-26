#!/bin/bash

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate labram

OUTPUT_DIR="output/verify_changes"
mkdir -p $OUTPUT_DIR

echo "Starting Verification..."
python3 main.py \
  --config configs/pretrain.yaml \
  --tiny \
  --opts \
    model.pretrain_tasks=['feature_pred'] \
    output_dir=$OUTPUT_DIR \
    enable_wandb=false \
    epochs=1 \
    dataset.batch_size=2 \
    val_freq_split=2 \
    optimizer.lr=0.001

echo "Verification Finished."
ls -l $OUTPUT_DIR
