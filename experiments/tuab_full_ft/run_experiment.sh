#!/bin/bash
# Unified script to run full fine-tuning experiments
# Usage: bash run_experiment.sh [model_type] [gpu_id]
# model_type: recon, feat_only, neuro_ke, all

MODEL_TYPE=${1:-all}
GPU_ID=${2:-0,1,2,3}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/finetune.yaml"
EPOCHS=10
LR=0.001
WD=0.05

run_ft() {
    local name=$1
    local pretrained=$2
    local output_dir=$3
    local gpu=$4

    echo "========================================================"
    echo "Starting Full Fine-tuning for $name"
    echo "Pretrained Weights: $pretrained"
    echo "Output Directory: $output_dir"
    echo "GPU: $gpu"
    echo "========================================================"
    
    export CUDA_VISIBLE_DEVICES=$gpu
    
    python main.py \
      --config "$CONFIG" \
      --opts model.use_pretrained=true \
             model.pretrained_path="$pretrained" \
             epochs="$EPOCHS" \
             optimizer.lr="$LR" \
             optimizer.weight_decay="$WD" \
             output_dir="$output_dir"
             
    echo "Finished $name. Check results in $output_dir/log.csv"
}

# Define paths
# 1. Recon (Baseline): From external project or fallback
PATH_RECON="/vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth"
# 2. Feat-Only (Sanity): From output_old
PATH_FEAT="output_old/sanity_feat_only/best.pth"
# 3. Neuro-KE (Flagship): From output_old
PATH_NEURO="output_old/flagship_cross_attn/best.pth"

case $MODEL_TYPE in
    recon)
        run_ft "Recon-Baseline" "$PATH_RECON" "output/finetune_recon1" "$GPU_ID"
        ;;
    feat_only)
        run_ft "Feat-Only" "$PATH_FEAT" "output/finetune_feat_only1" "$GPU_ID"
        ;;
    neuro_ke)
        run_ft "Neuro-KE" "$PATH_NEURO" "output/finetune_neuro_ke1" "$GPU_ID"
        ;;
    all)
        echo "Running ALL experiments sequentially on GPU $GPU_ID..."
        run_ft "Recon-Baseline" "$PATH_RECON" "output/finetune_recon1" "$GPU_ID"
        run_ft "Feat-Only" "$PATH_FEAT" "output/finetune_feat_only1" "$GPU_ID"
        run_ft "Neuro-KE" "$PATH_NEURO" "output/finetune_neuro_ke1" "$GPU_ID"
        ;;
    *)
        echo "Usage: $0 {recon|feat_only|neuro_ke|all} [gpu_id]"
        echo "Example: bash experiments/tuab_full_ft/run_experiment.sh neuro_ke 0"
        exit 1
        ;;
esac
