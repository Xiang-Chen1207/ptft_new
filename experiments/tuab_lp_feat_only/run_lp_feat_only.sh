#!/usr/bin/env bash
# Run linear probing using features from the feature-prediction-only model (sanity_feat_only)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${FEATURES:=experiments/tuab_lp_feat_only/features/feat_only_features.npz}"
: "${DEVICE:=cuda}"

echo "=== Linear Probing with Feat-Only Model Features ==="
echo "Feature types: eeg, feat, full, pred"
echo ""

/vepfs-0x0d/home/cx/miniconda3/envs/labram/bin/python experiments/tuab_lp_feat_only/run_lp.py \
  --features_path "$FEATURES" \
  --device "$DEVICE" \
  --feature_types eeg feat full pred
