#!/bin/bash
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram

# 1. Extract features for Baseline (Recon)
# Using the specific weights provided by user
echo "Running feature extraction for Baseline (Recon)..."
python experiments/tuab_lp/extract_features.py \
    --model_type recon \
    --weights_path /vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth \
    --device cuda

# 2. Run Linear Probing and save to baseline report
echo "Running linear probing for Baseline..."
python experiments/tuab_lp/run_lp_offline.py --features_path experiments/tuab_lp/features/recon_features.npz | tee experiments/tuab_lp/results_baseline.md

echo "Done. Baseline results saved to experiments/tuab_lp/results_baseline.md"
