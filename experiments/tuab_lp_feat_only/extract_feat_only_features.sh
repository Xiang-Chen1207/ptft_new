#!/usr/bin/env bash
# Extract features from the feature-prediction-only pretrained model (sanity_feat_only)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${DATASET_DIR:=/vepfs-0x0d/eeg-data/TUAB}"
: "${WEIGHTS:=output_old/sanity_feat_only/best.pth}"
: "${DEVICE:=cuda}"
: "${BATCH_SIZE:=256}"
: "${NUM_WORKERS:=16}"
: "${OUT_DIR:=experiments/tuab_lp_feat_only/features}"

export CUDA_VISIBLE_DEVICES=2,3


/vepfs-0x0d/home/cx/miniconda3/envs/labram/bin/python experiments/tuab_lp_feat_only/extract_features.py \
  --model_type feat_only \
  --dataset_dir "$DATASET_DIR" \
  --weights_path "$WEIGHTS" \
  --output_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS"

echo "Saved features to $OUT_DIR/feat_only_features.npz"
