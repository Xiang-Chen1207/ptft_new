#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${FEATURES:=experiments/tuab_lp/features/neuro_ke_features.npz}"
: "${DEVICE:=cuda}"

python experiments/tuab_lp/run_lp_offline.py \
  --features_path "$FEATURES" \
  --device "$DEVICE" \
  --feature_types pred

