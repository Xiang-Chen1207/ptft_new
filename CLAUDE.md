# CLAUDE.md - AI Assistant Guide for PTFT

## Project Overview

**PTFT (Pretrain-Finetune)** implements **Neuro-KE (Neuro-Knowledge Engine)**, a label-free knowledge integration framework for EEG foundation models. The project enhances EEG pretraining by integrating domain-specific signal priors (morphometric, spectral, non-linear features) into deep learning architectures.

### Core Concept
Neuro-KE acts as a "knowledge engine" that forces the model to encode clinically meaningful patterns by predicting z-scored EEG features (band powers, Hjorth parameters, spectral entropy, aperiodic exponents) as an auxiliary training signal.

## Codebase Structure

```
ptft/
├── main.py                    # Unified CLI entry point for pretraining/finetuning
├── configs/
│   ├── pretrain.yaml          # Pretraining configuration (TUEG dataset, multi-task)
│   └── finetune.yaml          # Finetuning configuration (TUAB classification)
├── core/
│   ├── trainer.py             # Trainer class: optimization loop, checkpointing
│   ├── engine.py              # train_one_epoch() and evaluate() functions
│   └── loss.py                # LossFactory with MSE, CrossEntropy, BCEWithLogits
├── models/
│   ├── wrapper.py             # CBraModWrapper: task heads (recon, feature, classification)
│   ├── backbone.py            # CBraModBackbone: Transformer encoder with patch embedding
│   └── criss_cross_transformer.py  # Custom spatial-temporal cross-attention
├── datasets/
│   ├── builder.py             # Dataset factory (TUEG/TUAB), supports --tiny mode
│   ├── tueg.py                # TUEG pretraining dataset (95/5 train/val split)
│   └── tuab.py                # TUAB classification dataset (80/10/10 split)
├── utils/
│   └── util.py                # MetricLogger, WandB integration, visualization
├── scripts/
│   ├── run_flagship_cross_attn.sh  # Primary experiment script
│   ├── run_ablation_gap.sh         # GAP strategy ablation
│   ├── run_sanity_feat_only.sh     # Feature-only sanity check
│   └── run_baseline_recon.sh       # Reconstruction-only baseline
├── eval_features.py           # Feature prediction evaluation (R², PCC per feature)
├── viz_features.py            # Bar plot visualization for metrics
├── viz_compare.py             # Compare metrics across experiments
└── verify_changes.sh          # Quick sanity test with tiny dataset
```

## Quick Start Commands

```bash
# Pretraining (from scratch)
python main.py --config configs/pretrain.yaml

# Resume training from checkpoint
python main.py --config configs/pretrain.yaml --resume output/flagship_cross_attn/latest.pth

# Debugging with tiny dataset (10 files, fast iteration)
python main.py --config configs/pretrain.yaml --tiny --opts epochs=1

# Finetuning on TUAB
python main.py --config configs/finetune.yaml

# Override config options via CLI
python main.py --config configs/pretrain.yaml --opts model.pretrain_tasks=['reconstruction'] epochs=50

# Evaluate feature prediction
python eval_features.py --config configs/pretrain.yaml --checkpoint output/best.pth --output metrics.csv

# Quick sanity check
./verify_changes.sh
```

## Configuration System

YAML configs support nested key overrides via `--opts`:

```bash
--opts key1.key2=value list_key=['a','b'] numeric=100
```

### Key Configuration Options

| Key | Values | Description |
|-----|--------|-------------|
| `task_type` | `pretraining`, `classification` | Training mode |
| `model.pretrain_tasks` | `['reconstruction']`, `['feature_pred']`, `['reconstruction','feature_pred']` | Pretraining objectives |
| `model.feature_token_type` | `gap`, `cross_attn` | Feature extraction method |
| `model.feature_token_strategy` | `single`, `all`, `group` | Cross-attention token strategy |
| `loss.feature_loss_weight` | float | Weight for feature prediction loss |
| `val_freq_split` | int | Intra-epoch validation frequency (0 = disabled) |
| `dataset.tiny` | bool | Use 10-file subset for debugging |

## Code Conventions

### Python Style
- **Type hints**: Not consistently used; follow existing patterns in file
- **Docstrings**: Minimal; code is self-documenting
- **Imports**: Standard library → third-party → local modules
- **Config access**: Use `config.get('key', default)` pattern throughout

### Model Architecture Pattern
```python
# Wrapper pattern: CBraModWrapper contains backbone + task-specific heads
class CBraModWrapper(nn.Module):
    def __init__(self, config):
        self._init_backbone(model_config)      # Shared encoder
        self._init_task_heads(config, model_config)  # Task-specific heads
```

### Training Loop Pattern
- `train_one_epoch()` handles forward/backward, logging, intra-epoch validation
- `evaluate()` computes metrics, generates visualizations
- Trainer orchestrates epochs, checkpointing, scheduler stepping

### Checkpoint Structure
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,  # Optional
    'config': ...,
    'epoch': int,
    'best_metric': float
}
```

## Important Patterns

### Multi-task Forward Pass (Pretraining)
```python
# Returns tuple: (reconstruction_output, mask, feature_prediction)
outputs, mask, feature_pred = model(x, mask=mask)
```

### Loss Computation
```python
# Reconstruction uses masked MSE
loss = criterion(outputs, x, mask=mask)

# Feature loss added with weight
if feature_pred is not None:
    loss += feature_loss_weight * F.mse_loss(feature_pred, target_features)
```

### Metrics Tracking
- Uses `MetricLogger` with `SmoothedValue` for running averages
- WandB logging via `WandbLogger` wrapper
- CSV logs written to `output_dir/log.csv`

## Testing & Validation

### Quick Sanity Test
```bash
./verify_changes.sh  # 2 samples, 1 epoch, verifies code runs
```

### Tiny Mode
```bash
python main.py --config configs/pretrain.yaml --tiny --opts epochs=2
```
Loads only 10 H5 files for fast iteration during development.

### Intra-Epoch Validation
Set `val_freq_split: N` to validate N times per epoch (uses 50-batch mini-validation for efficiency).

## Experiment Scripts

| Script | Purpose |
|--------|---------|
| `run_flagship_cross_attn.sh` | **Primary**: Multi-task + cross-attention feature tokens |
| `run_ablation_gap.sh` | Ablation: Global average pooling for features |
| `run_sanity_feat_only.sh` | Sanity: Feature prediction only (verify features are learnable) |
| `run_baseline_recon.sh` | Baseline: Reconstruction-only (standard MAE) |

All scripts support `resume` argument: `./run_flagship_cross_attn.sh resume`

## Data Paths (Production)

- **TUEG (Pretraining)**: `/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset`
- **TUAB (Finetuning)**: `/vepfs-0x0d/eeg-data/TUAB`
- **Features CSV**: `/vePFS-0x0d/pretrain-clip/feature_analysis/features_final_zscore.csv`

## Key Dependencies

- `torch` (2.9+) - Deep learning framework
- `einops` - Tensor operations
- `h5py` - EEG data I/O
- `wandb` - Experiment tracking
- `scikit-learn` - Metrics (balanced_accuracy, ROC-AUC, etc.)
- `pyyaml` - Configuration
- `mne` - EEG processing utilities

## Common Development Tasks

### Adding a New Pretraining Task
1. Add task name to `model.pretrain_tasks` list in config
2. Implement head in `models/wrapper.py:_init_pretraining_heads()`
3. Handle forward pass in `_forward_pretraining()`
4. Add loss computation in `core/engine.py:train_one_epoch()`

### Adding a New Dataset
1. Create dataset class in `datasets/` following `tueg.py` pattern
2. Register in `datasets/builder.py:build_dataloader()`
3. Implement `__getitem__` returning `(x, features)` for pretraining or `(x, label)` for classification

### Modifying Model Architecture
- Backbone changes: `models/backbone.py`
- Task heads: `models/wrapper.py`
- Attention mechanism: `models/criss_cross_transformer.py`

### Debugging Training Issues
1. Enable tiny mode: `--tiny`
2. Check feature alignment: `python eval_features.py --checkpoint ... --output metrics.csv`
3. Inspect checkpoint: `python inspect_checkpoint.py path/to/checkpoint.pth`
4. Check `output/*/log.csv` for training curves

## Git Workflow

- Main experiments tracked in `output/` directory (gitignored)
- Configuration changes should be committed
- Use descriptive branch names for experiments
- Checkpoint resume support allows interruption/restart

## WandB Integration

- Project: `eeg-knowledge-engine` (pretraining) / `ptft-project` (finetuning)
- Entity: `bci-foundation-model`
- Logs: loss, metrics, learning rate, gradient norm, reconstruction visualizations
- Set `enable_wandb: false` in config to disable

## Architecture Details

### CBraModBackbone
- **Patch Embedding**: Spectral + spatial convolutions with FFT analysis
- **Transformer**: 12 layers, 8 heads, d_model=200
- **Output**: `(B, C, N, D)` where C=channels, N=patches, D=d_model

### Feature Prediction Strategies
1. **GAP**: Global Average Pool → MLP → features
2. **Cross-Attention (single)**: 1 learnable query → attention → all features
3. **Cross-Attention (all)**: N queries (N=feature_dim) → attention → 1 output each
4. **Cross-Attention (group)**: K queries → attention → MLP → features
