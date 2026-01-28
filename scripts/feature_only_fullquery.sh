#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# Ablation B: Feature Prediction Only (Sanity Check)
OUTPUT_DIR="output/sanity_feat_only_all_60s"
mkdir -p $OUTPUT_DIR

echo "Starting Ablation B: Feature Prediction Only"

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

python3 main.py \
  --config configs/pretrain.yaml \
  $RESUME_ARG \
  --opts \
    model.pretrain_tasks=['feature_pred'] \
    model.feature_token_strategy='all' \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-sanity"\
    epochs=20 \
    dataset.batch_size=256 \
    dataset.input_size=12000 \
    optimizer.lr=5e-5 \
    val_freq_split=10

echo "Ablation B Finished."