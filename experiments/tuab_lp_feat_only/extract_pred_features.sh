#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${DATASET_DIR:=/vepfs-0x0d/eeg-data/TUAB}"
: "${WEIGHTS:=output_old/flagship_cross_attn/best.pth}"
: "${DEVICE:=cuda}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=8}"
: "${OUT_DIR:=experiments/tuab_lp/features}"

python experiments/tuab_lp/extract_features.py \
  --model_type neuro_ke \
  --dataset_dir "$DATASET_DIR" \
  --weights_path "$WEIGHTS" \
  --output_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS"

echo "Saved features (including train_pred/test_pred) to $OUT_DIR/neuro_ke_features.npz"

