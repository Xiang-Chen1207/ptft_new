#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# Flagship Experiment (Dynamic Loss): Multi-task with Cross-Attention + Dynamic Loss Balancing
# Multi-GPU Version (Using 4 GPUs via DataParallel)

OUTPUT_DIR="output/flagship_dynamic_loss"
mkdir -p $OUTPUT_DIR

echo "Starting Flagship Experiment (Dynamic Loss)"

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

# Start Training
# Note: feature_loss_weight=1.0 is used as the BASE weight for dynamic balancing.
# The dynamic scaler will multiply this base weight.
python3 main.py \
  --config configs/pretrain.yaml \
  $RESUME_ARG \
  --opts \
    model.pretrain_tasks=['reconstruction','feature_pred'] \
    model.feature_token_type='cross_attn' \
    model.feature_token_strategy='single' \
    loss.feature_loss_weight=1.0 \
    loss.use_dynamic_loss=true \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-flagship-dynamic" \
    epochs=20 \
    dataset.batch_size=512 \
    optimizer.lr=4e-3 \
    val_freq_split=10 \
    dataset.input_size=12000 \
    model.seq_len=60 \
    dataset.cache_path="output/dataset_index_60s.json"

echo "Experiment Finished."
