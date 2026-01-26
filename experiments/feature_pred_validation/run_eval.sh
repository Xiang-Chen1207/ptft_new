#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CHECKPOINT:=output_old/flagship_cross_attn/best.pth}"
: "${CONFIG:=configs/pretrain.yaml}"
: "${OUTPUT:=feature_metrics_eval_full.csv}"
: "${BATCH_SIZE:=256}"
: "${DEVICE:=cuda}"

python eval_features.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"

echo "Done. Metrics saved to $OUTPUT"

