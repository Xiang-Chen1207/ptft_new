#!/usr/bin/env bash
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CHECKPOINT:=/vepfs-0x0d/home/cx/ptft/output/flagship_fixed/checkpoint_epoch_16.pth}"
: "${CONFIG:=configs/pretrain.yaml}"
: "${OUTPUT:=/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/60sfeature_metrics_eval_full.csv}"
: "${BATCH_SIZE:=256}"
: "${DEVICE:=cuda}"

python eval_features.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --dataset TUEG \
  --split val

echo "Done. Metrics saved to $OUTPUT"
