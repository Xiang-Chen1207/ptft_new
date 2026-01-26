#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=2
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CONFIG:=configs/finetune.yaml}"
: "${EPOCHS:=10}"
: "${PRETRAINED:=output_old/sanity_feat_only/best.pth}"
: "${LR:=0.001}"
: "${WD:=0.05}"

python main.py \
  --config "$CONFIG" \
  --opts model.use_pretrained=true model.pretrained_path="$PRETRAINED" epochs="$EPOCHS" optimizer.lr="$LR" optimizer.weight_decay="$WD"

echo "Full fine-tune (Feat-only init) finished. Check outputs under output/finetune/"

