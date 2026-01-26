#!/usr/bin/env bash
# Full fine-tuning with Neuro-KE pretrained weights using Feature token head only (cross-attention)
export CUDA_VISIBLE_DEVICES=3
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CONFIG:=configs/finetune.yaml}"
: "${EPOCHS:=10}"
: "${PRETRAINED:=output_old/flagship_cross_attn/best.pth}"
: "${LR:=0.001}"
: "${WD:=0.05}"
: "${OUTPUT_DIR:=output/finetune_neuro_ke_feat_head}"

python main.py \
  --config "$CONFIG" \
  --opts model.use_pretrained=true model.pretrained_path="$PRETRAINED" model.cls_head_type=feat epochs="$EPOCHS" optimizer.lr="$LR" optimizer.weight_decay="$WD" output_dir="$OUTPUT_DIR"

echo "Full fine-tune (Neuro-KE init, Feature head only) finished. Check outputs under $OUTPUT_DIR/"
