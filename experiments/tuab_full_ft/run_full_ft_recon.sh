#!/usr/bin/env bash
# Full fine-tuning with Recon-only (reconstruction baseline) pretrained weights
export CUDA_VISIBLE_DEVICES=1
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CONFIG:=configs/finetune.yaml}"
: "${EPOCHS:=10}"
: "${PRETRAINED:=output_old/baseline_recon/best.pth}"
: "${LR:=0.001}"
: "${WD:=0.05}"
: "${OUTPUT_DIR:=output/finetune_recon}"

python main.py \
  --config "$CONFIG" \
  --opts model.use_pretrained=true model.pretrained_path="$PRETRAINED" epochs="$EPOCHS" optimizer.lr="$LR" optimizer.weight_decay="$WD" output_dir="$OUTPUT_DIR"

echo "Full fine-tune (Recon-only init) finished. Check outputs under $OUTPUT_DIR/"
