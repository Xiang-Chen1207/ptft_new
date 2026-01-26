#!/usr/bin/env bash
# Full fine-tuning with Neuro-KE pretrained weights using EEG + Feature token head (full head)
export CUDA_VISIBLE_DEVICES=3
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CONFIG:=configs/finetune.yaml}"
: "${EPOCHS:=10}"
: "${PRETRAINED:=output_old/flagship_cross_attn/best.pth}"
: "${LR:=0.001}"
: "${WD:=0.05}"
: "${OUTPUT_DIR:=output/finetune_neuro_ke_full_head}"

python main.py \
  --config "$CONFIG" \
  --opts model.use_pretrained=true model.pretrained_path="$PRETRAINED" model.cls_head_type=full epochs="$EPOCHS" optimizer.lr="$LR" optimizer.weight_decay="$WD" output_dir="$OUTPUT_DIR"

echo "Full fine-tune (Neuro-KE init, EEG+Feature head) finished. Check outputs under $OUTPUT_DIR/"
