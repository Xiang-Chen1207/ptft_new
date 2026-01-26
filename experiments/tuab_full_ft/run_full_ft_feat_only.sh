#!/usr/bin/env bash
# Full fine-tuning with Feat-Only (feature prediction only) pretrained weights
export CUDA_VISIBLE_DEVICES=2
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CONFIG:=configs/finetune.yaml}"
: "${EPOCHS:=10}"
: "${PRETRAINED:=output_old/sanity_feat_only/best.pth}"
: "${LR:=0.001}"
: "${WD:=0.05}"
: "${OUTPUT_DIR:=output/finetune_feat_only}"

python main.py \
  --config "$CONFIG" \
  --opts model.use_pretrained=true model.pretrained_path="$PRETRAINED" epochs="$EPOCHS" optimizer.lr="$LR" optimizer.weight_decay="$WD" output_dir="$OUTPUT_DIR"

echo "Full fine-tune (Feat-only init) finished. Check outputs under $OUTPUT_DIR/"
