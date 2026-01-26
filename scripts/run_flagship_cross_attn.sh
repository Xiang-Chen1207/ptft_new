#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# Flagship Experiment: Multi-task with Cross-Attention (Single Token)
# Single GPU Version (GPU 1)
# stdbuf -oL -eL conda run --no-capture-output -n labram python eval_features.py --config configs/pretrain.yaml --checkpoint /vePFS-0x0d/home/chen/ptft/output/flagship_cross_attn/best.pth --output feature_metrics_eval_zero_mask.csv
OUTPUT_DIR="output/flagship_cross_attn"
mkdir -p $OUTPUT_DIR

echo "Starting Flagship Experiment: Multi-task (Cross-Attention Single)"

RESUME_ARG=""
if [ "$1" == "resume" ]; then
    CHECKPOINT_PATH="${OUTPUT_DIR}/latest.pth"
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Resuming from checkpoint: $CHECKPOINT_PATH"
        RESUME_ARG="--resume $CHECKPOINT_PATH"
    else
        echo "Warning: Checkpoint not found at $CHECKPOINT_PATH, starting from scratch"
    fi
elif [ -n "$RESUME_PATH" ]; then
    echo "Resuming from checkpoint: $RESUME_PATH"
    RESUME_ARG="--resume $RESUME_PATH"
fi

# Using GPU 1
CUDA_VISIBLE_DEVICES=1 python3 main.py \
  --config configs/pretrain.yaml \
  $RESUME_ARG \
  --opts \
    model.pretrain_tasks=['reconstruction','feature_pred'] \
    model.feature_token_type='cross_attn' \
    model.feature_token_strategy='single' \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-flagship" \
    epochs=20 \
    dataset.batch_size=1024 \
    optimizer.lr=1e-3 \
    val_freq_split=10

echo "Flagship Experiment Finished."
