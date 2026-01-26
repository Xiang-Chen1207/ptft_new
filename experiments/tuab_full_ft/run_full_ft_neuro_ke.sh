#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CONFIG:=configs/finetune.yaml}"
: "${EPOCHS:=10}"
: "${PRETRAINED:=output_old/flagship_cross_attn/best.pth}"
: "${LR:=0.001}"
: "${WD:=0.05}"

python main.py \
  --config "$CONFIG" \
  --opts model.use_pretrained=true model.pretrained_path="$PRETRAINED" epochs="$EPOCHS" optimizer.lr="$LR" optimizer.weight_decay="$WD"

echo "Full fine-tune (Neuro-KE init) finished. Check outputs under output/finetune/"

