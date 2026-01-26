# Experimental Plan: EEG Pretraining with Feature Guidance (Flagship First)

This document outlines the validation and ablation experiments, prioritizing the most promising "Flagship" architecture.

## 1. Flagship Model (Primary)
**Architecture**: Multi-task (Reconstruction + Feature Prediction) with **Cross-Attention Single Token**.
- **Objective**: Verify that decoupling feature extraction via a dedicated query token improves representation learning compared to baselines.
- **Config**:
  - `pretrain_tasks`: `['reconstruction', 'feature_pred']`
  - `feature_token_type`: `cross_attn`
  - `feature_token_strategy`: `single`
- **Script**: `scripts/run_flagship_cross_attn.sh`

## 2. Ablation Studies (Secondary)
If the flagship model works, we compare it against simpler or different strategies to prove its effectiveness.

### Ablation A: GAP Strategy
- **Objective**: Check if the complex Cross-Attention is necessary, or if simple Global Average Pooling suffices.
- **Config**: `feature_token_type: 'gap'`
- **Script**: `scripts/run_ablation_gap.sh`

### Ablation B: Feature Prediction Only (Sanity Check)
- **Objective**: Ensure features are actually predictable from the data (R2 > 0). If this fails, Multi-task will fail.
- **Config**: `pretrain_tasks: ['feature_pred']`
- **Script**: `scripts/run_sanity_feat_only.sh`

## 3. Baselines (Optional)
### Reconstruction Only
- **Objective**: Standard MAE baseline. Useful only if we suspect Multi-tasking is hurting performance.
- **Config**: `pretrain_tasks: ['reconstruction']`
- **Script**: `scripts/run_baseline_recon.sh`
