#!/bin/bash
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram

# 1. Extract features
# Using default weights_path which I just modified
echo "Running feature extraction..."
python experiments/tuab_lp/extract_features.py --model_type neuro_ke --device cuda

# 2. Run Linear Probing and save to new report
echo "Running linear probing..."
python experiments/tuab_lp/run_lp_offline.py --features_path experiments/tuab_lp/features/neuro_ke_features.npz | tee experiments/tuab_lp/results_new.md

echo "Done. Results saved to experiments/tuab_lp/results_new.md"
