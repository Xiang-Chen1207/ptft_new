#!/bin/bash
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram

# Paths
BASELINE_WEIGHTS="/vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth"
FLAGSHIP_WEIGHTS="/vePFS-0x0d/home/cx/ptft/output_old/flagship_cross_attn/checkpoint_epoch_6.pth"
FEATONLY_WEIGHTS="/vePFS-0x0d/home/cx/ptft/output_old/sanity_feat_only/checkpoint_epoch_5.pth" # Assuming relative path works or fix it if needed

BASELINE_FEAT="experiments/tuab_lp/features/recon_features.npz"
FLAGSHIP_FEAT="experiments/tuab_lp/features/neuro_ke_features.npz"
FEATONLY_FEAT="experiments/tuab_lp_feat_only/features/feat_only_features.npz"

OUTPUT_REPORT="experiments/tuab_lp/results_final.md"

echo "=== Starting Full Comparative Experiment (Baseline vs Flagship vs FeatOnly) ==="

# 1. Feature Extraction - Baseline
echo "Extracting Baseline features..."
python experiments/tuab_lp/extract_features.py \
    --model_type recon \
    --weights_path "$BASELINE_WEIGHTS" \
    --output_dir experiments/tuab_lp/features \
    --device cuda

# 2. Feature Extraction - Flagship
echo "Extracting Flagship features..."
python experiments/tuab_lp/extract_features.py \
    --model_type neuro_ke \
    --weights_path "$FLAGSHIP_WEIGHTS" \
    --output_dir experiments/tuab_lp/features \
    --device cuda

# 3. Feature Extraction - FeatOnly
echo "Extracting FeatOnly features..."
mkdir -p experiments/tuab_lp_feat_only/features
python experiments/tuab_lp/extract_features.py \
    --model_type feat_only \
    --weights_path "$FEATONLY_WEIGHTS" \
    --output_dir experiments/tuab_lp_feat_only/features \
    --device cuda

# 3. Run Comparative Linear Probing
echo "Running Comparative Linear Probing (Incremental Subjects)..."
# Using tee to show output in terminal AND save to file
python experiments/tuab_lp/run_lp_compare.py \
    --baseline_path "$BASELINE_FEAT" \
    --flagship_path "$FLAGSHIP_FEAT" \
    --featonly_path "$FEATONLY_FEAT" \
    --seed 42 \
    | tee "$OUTPUT_REPORT"

echo "Done. Final results saved to $OUTPUT_REPORT"
