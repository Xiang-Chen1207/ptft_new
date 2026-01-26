#!/usr/bin/env bash
# Evaluate feature prediction performance using the feature-prediction-only pretrained model
# (sanity_feat_only: only feature prediction task, no reconstruction)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CHECKPOINT:=output_old/sanity_feat_only/best.pth}"
: "${CONFIG:=configs/pretrain.yaml}"
: "${OUTPUT:=feature_metrics_eval_feat_only.csv}"
: "${BATCH_SIZE:=256}"
: "${DEVICE:=cuda}"

python eval_features.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"

echo "Done. Metrics saved to $OUTPUT"
