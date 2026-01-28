#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram

# Paths
BASELINE_WEIGHTS="/vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth"
FLAGSHIP_FULL_WEIGHTS="/vepfs-0x0d/home/cx/ptft/output/flagship_cross_attn_all_60s_fullquery/checkpoint_epoch_4.pth"
FEATONLY_WEIGHTS="/vepfs-0x0d/home/cx/ptft/output/sanity_feat_only_all_60s/checkpoint_epoch_5.pth"

BASELINE_FEAT="experiments/tuab_lp/features/recon_features.npz"
FLAGSHIP_FULL_FEAT="experiments/tuab_lp/features/neuro_ke_full_features.npz"
FLAGSHIP_POOLED_FEAT="experiments/tuab_lp/features/neuro_ke_pooled_features.npz"
FEATONLY_FEAT="experiments/tuab_lp_feat_only/features/feat_only_features.npz"

OUTPUT_REPORT="experiments/tuab_lp/results_60s_final.md"

echo "=== Starting Full Comparative Experiment (Baseline vs Flagship Full vs Flagship Pooled vs FeatOnly) ==="

# 1. Feature Extraction - Baseline
# echo "Extracting Baseline features..."
# python experiments/tuab_lp/extract_features.py \
#    --model_type recon \
#    --weights_path "$BASELINE_WEIGHTS" \
#    --output_dir experiments/tuab_lp/features \
#    --device cuda

# 2. Feature Extraction - Flagship Full (12400 dim)
# echo "Extracting Flagship Full features..."
# python experiments/tuab_lp/extract_features.py \
#     --model_type neuro_ke \
#     --weights_path "$FLAGSHIP_FULL_WEIGHTS" \
#     --output_dir experiments/tuab_lp/features \
#     --output_name neuro_ke_full_features.npz \
#     --device cuda

# 3. Feature Extraction - Flagship Pooled (200 dim)
echo "Extracting Flagship Pooled features..."
python experiments/tuab_lp/extract_features.py \
    --model_type neuro_ke_pooled \
    --weights_path "$FLAGSHIP_FULL_WEIGHTS" \
    --output_dir experiments/tuab_lp/features \
    --output_name neuro_ke_pooled_features.npz \
    --device cuda

# 4. Feature Extraction - FeatOnly
echo "Extracting FeatOnly features..."
mkdir -p experiments/tuab_lp_feat_only/features
python experiments/tuab_lp/extract_features.py \
    --model_type feat_only \
    --weights_path "$FEATONLY_WEIGHTS" \
    --output_dir experiments/tuab_lp_feat_only/features \
    --output_name feat_only_features.npz \
    --device cuda

# 5. Run Comparative Linear Probing
echo "Running Comparative Linear Probing (Incremental Subjects)..."
# Using tee to show output in terminal AND save to file
python experiments/tuab_lp/run_lp_compare.py \
    --baseline_path "$BASELINE_FEAT" \
    --flagship_path "$FLAGSHIP_POOLED_FEAT" \
    --flagship_pooled_path "$FLAGSHIP_POOLED_FEAT" \
    --featonly_path "$FEATONLY_FEAT" \
    --seed 42 \
    | tee "$OUTPUT_REPORT"

echo "Done. Final results saved to $OUTPUT_REPORT"
