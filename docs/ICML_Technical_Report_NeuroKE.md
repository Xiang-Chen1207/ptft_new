# Neuro-KE: Feature-Guided Masked Modeling for EEG Foundation Models

## Technical Report for ICML Submission

---

# Main Paper Content

## 1. Introduction and Problem Formulation

### 1.1 Masked Modeling for EEG

Masked modeling has emerged as a powerful self-supervised paradigm for learning representations from unlabeled data. In the context of electroencephalography (EEG), we frame the pretraining objective as learning a mapping $f_\theta: \mathcal{X} \rightarrow \mathcal{X}$ that reconstructs masked portions of the input signal while simultaneously predicting domain-specific features.

Let $\mathbf{X} \in \mathbb{R}^{C \times T}$ denote a multi-channel EEG recording with $C = 19$ channels and $T$ time samples. We partition $\mathbf{X}$ into $N$ non-overlapping patches $\{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$ where each patch $\mathbf{x}_i \in \mathbb{R}^{C \times P}$ contains $P$ samples per channel.

### 1.2 Feature-Guided Reconstruction Motivation

Standard masked autoencoders optimize reconstruction loss alone, which may not capture clinically meaningful structure in EEG signals. We introduce **Neuro-KE (Neuro-Knowledge Engine)**, a label-free knowledge integration framework that augments reconstruction with feature prediction as an auxiliary task.

The key insight is that domain-specific EEG features—including spectral band powers, Hjorth parameters, and aperiodic exponents—encode clinically relevant information that raw reconstruction may not explicitly capture. By jointly optimizing both objectives, we force the encoder to learn representations that are simultaneously faithful to the raw signal and aligned with interpretable domain knowledge.

### 1.3 Problem Definition

**Input:** Raw EEG signal $\mathbf{X} \in \mathbb{R}^{C \times N \times P}$ (channels × patches × patch_size)

**Masking:** Binary mask $\mathbf{M} \in \{0, 1\}^{C \times N}$ where $M_{c,n} = 1$ indicates patch $(c, n)$ is masked

**Targets:**
- Reconstruction target: $\mathbf{X}$ (original signal)
- Feature target: $\mathbf{z} \in \mathbb{R}^{D_f}$ where $D_f = 62$ is the dimension of z-scored EEG features

**Objective:** Learn encoder parameters $\theta$ such that:
$$\min_\theta \mathcal{L}_{\text{recon}}(\hat{\mathbf{X}}, \mathbf{X}, \mathbf{M}) + \lambda \cdot \mathcal{L}_{\text{feat}}(\hat{\mathbf{z}}, \mathbf{z})$$

---

## 2. Masked Modeling Mechanism

### 2.1 Masking Strategy

The masking operates at the **patch level** with the following specification:

**Granularity:** Each patch spans $P = 200$ time samples (1 second at 200 Hz sampling rate) across all $C = 19$ channels.

**Mask Sampling:** Random Bernoulli masking with ratio $p = 0.5$:
```
M_{c,n} ~ Bernoulli(0.5)  ∀ c ∈ [1,C], n ∈ [1,N]
```

**Implementation Detail (from `wrapper.py:195-198`):**
```python
def _generate_mask(self, x):
    B, C, N, P = x.shape
    return torch.bernoulli(torch.full((B, C, N), 0.5, device=x.device))
```

The mask is generated independently for each spatial-temporal patch, meaning a patch at channel $c$ and temporal position $n$ may be masked while adjacent patches remain visible.

### 2.2 Visible Token Processing

For masked patches, the input is replaced with a learnable mask token (initialized to zeros):

```python
# From backbone.py:27
self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
```

The patch embedding module processes masked inputs as follows:

1. **Mask Application:** Replace masked patch values with `mask_encoding`
   ```python
   mask_bool = (mask == 1).unsqueeze(-1)
   mask_x = torch.where(mask_bool, self.mask_encoding.view(1,1,1,-1), mask_x)
   ```

2. **Temporal Convolution:** Three-layer 1D convolution stack
   - Layer 1: Conv2d(1 → 25, kernel=49, stride=25) + GroupNorm + GELU
   - Layer 2: Conv2d(25 → 25, kernel=3, stride=1) + GroupNorm + GELU
   - Layer 3: Conv2d(25 → 25, kernel=3, stride=1) + GroupNorm + GELU

3. **Spectral Embedding:** FFT-based frequency encoding
   ```python
   spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
   spectral = torch.abs(spectral)  # Magnitude spectrum
   spectral_emb = self.spectral_proj(spectral)  # Linear(freq_bins → d_model)
   ```

4. **Positional Encoding:** Depthwise separable convolution
   ```python
   self.positional_encoding = nn.Conv2d(d_model, d_model, kernel_size=(19, 7),
                                         stride=1, padding=(9, 3), groups=d_model)
   ```

5. **Feature Fusion:**
   ```python
   patch_emb = patch_emb + spectral_emb + positional_embedding
   ```

### 2.3 Masked Token Reconstruction

The encoder processes all tokens (both visible and masked) through the Transformer. Reconstruction is performed via a linear projection head:

```python
# From wrapper.py:44
self.head = nn.Linear(self.d_model, model_config.get('out_dim', 200))
```

The reconstruction output $\hat{\mathbf{X}} \in \mathbb{R}^{B \times C \times N \times P}$ has the same shape as the input.

### 2.4 Feature-Guided Supervision

The feature prediction module extracts a global representation and predicts z-scored EEG features:

**Cross-Attention Feature Extraction (Single Token Strategy):**

1. Initialize learnable query token: $\mathbf{Q}_{\text{feat}} \in \mathbb{R}^{1 \times d}$
   ```python
   self.feat_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
   nn.init.normal_(self.feat_query, std=0.02)
   ```

2. Flatten encoder output: $\mathbf{H} \in \mathbb{R}^{B \times (C \cdot N) \times d}$

3. Cross-attention:
   ```python
   attn_output, _ = self.feat_attn(query, feats_flat, feats_flat)
   # feat_attn: MultiheadAttention(embed_dim=200, num_heads=4)
   ```

4. MLP prediction head:
   ```python
   self.feature_head = nn.Sequential(
       nn.Linear(self.d_model, self.d_model),
       nn.ReLU(),
       nn.Linear(self.d_model, self.feature_dim)  # feature_dim=62
   )
   ```

**Feature Target Details:**

The 62 z-scored features span four categories:
- **Time-Domain:** RMS, peak-to-peak amplitude, skewness, kurtosis, zero-crossing rate
- **Frequency-Domain:** Band powers (delta, theta, alpha, beta, gamma), band ratios
- **Spectral & Aperiodic:** Spectral entropy, centroid frequency, aperiodic exponent
- **Non-Linear & Complexity:** Hjorth parameters (activity, mobility, complexity)

Features are pre-computed using standard signal processing libraries (scipy, mne) and z-scored across the dataset. During training, z-scores are clipped to $[-6, 6]$ for robustness:
```python
features_np = np.clip(features_np, -6.0, 6.0)
```

---

## 3. Model Architecture

### 3.1 Encoder Architecture

The encoder follows a Transformer architecture with specialized spatial-temporal attention:

| Component | Specification |
|-----------|--------------|
| Input dimension | $d_{\text{model}} = 200$ |
| Feed-forward dimension | $d_{\text{ff}} = 800$ |
| Number of layers | $L = 12$ |
| Attention heads | $H = 8$ |
| Patch size | $P = 200$ (1 second) |
| Sequence length | $N = 60$ patches (60 seconds) |

### 3.2 Criss-Cross Spatial-Temporal Attention

Each Transformer layer employs a custom criss-cross attention mechanism that factorizes attention into spatial and temporal components:

**Self-Attention Block (from `criss_cross_transformer.py:95-113`):**

```python
def _sa_block(self, x):
    # x: (B, C, N, D) where C=channels, N=patches, D=d_model

    # Split embedding into spatial/temporal halves
    xs = x[:, :, :, :D//2]  # Spatial component
    xt = x[:, :, :, D//2:]  # Temporal component

    # Spatial attention: across channels within each time patch
    xs = xs.transpose(1,2).view(B*N, C, D//2)
    xs = self.self_attn_s(xs, xs, xs)  # MHA(d_model//2, nhead//2)

    # Temporal attention: across time patches within each channel
    xt = xt.view(B*C, N, D//2)
    xt = self.self_attn_t(xt, xt, xt)  # MHA(d_model//2, nhead//2)

    # Concatenate
    x = torch.cat([xs, xt], dim=-1)
    return self.dropout(x)
```

This factorization reduces computational complexity from $O((C \cdot N)^2)$ to $O(C^2 \cdot N + N^2 \cdot C)$ while explicitly modeling both spatial (cross-channel) and temporal dependencies.

### 3.3 Complete Forward Pass

```
Input: X ∈ ℝ^(B×C×N×P)
    ↓
[Mask Generation] → M ∈ {0,1}^(B×C×N)
    ↓
[Patch Embedding]
  - Temporal Conv: (B,C,N,P) → (B,C,N,D)
  - FFT + Linear: (B,C,N,P) → (B,C,N,D)
  - Positional Encoding: (B,C,N,D) → (B,C,N,D)
    ↓
[12× Criss-Cross Transformer Layer]
  - Spatial-Temporal Self-Attention
  - Feed-Forward Network (D→4D→D)
    ↓
Encoder Output: H ∈ ℝ^(B×C×N×D)
    ↓
┌──────────────────────┬────────────────────────────┐
│ Reconstruction Head  │ Feature Prediction Head    │
│ Linear(D→P)         │ CrossAttn + MLP            │
│ X̂ ∈ ℝ^(B×C×N×P)    │ ẑ ∈ ℝ^(B×62)              │
└──────────────────────┴────────────────────────────┘
```

---

## 4. Training Objective

### 4.1 Loss Formulation

The total loss combines masked reconstruction and feature prediction:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{feat}}$$

**Reconstruction Loss (Masked MSE):**

Only masked patches contribute to the reconstruction loss:
```python
# From loss.py:34-40
loss = (pred - target) ** 2
loss = loss.mean(dim=-1)  # Mean over patch dimension
loss = (loss * mask).sum() / (mask.sum() + 1e-6)
```

Mathematically:
$$\mathcal{L}_{\text{recon}} = \frac{\sum_{c,n} M_{c,n} \cdot \|\hat{\mathbf{x}}_{c,n} - \mathbf{x}_{c,n}\|_2^2}{\sum_{c,n} M_{c,n}}$$

**Feature Prediction Loss:**

Standard MSE between predicted and target z-scored features:
```python
feat_loss = F.mse_loss(feature_pred, target_features)
```

$$\mathcal{L}_{\text{feat}} = \frac{1}{D_f} \sum_{d=1}^{D_f} (\hat{z}_d - z_d)^2$$

### 4.2 Dynamic Loss Balancing

An optional dynamic weighting scheme balances gradient magnitudes:

```python
# From engine.py:72-81
if use_dynamic_loss:
    with torch.no_grad():
        dynamic_weight = (recon_loss.detach() / (feat_loss.detach() + 1e-8))
        dynamic_weight = torch.clamp(dynamic_weight, 0.5, 50.0)
        dynamic_weight = dynamic_weight * feature_loss_weight
    loss = recon_loss + dynamic_weight * feat_loss
else:
    loss = recon_loss + feature_loss_weight * feat_loss
```

This prevents either loss from dominating training by scaling the feature loss weight proportionally to the reconstruction loss magnitude, clamped to $[0.5\lambda, 50\lambda]$.

---

## 5. Training Protocol

### 5.1 Flagship Configuration

Derived from `run_flagship_fixed.sh`:

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | AdamW |
| **Learning rate** | $3 \times 10^{-4}$ |
| **Weight decay** | 0.05 |
| **Batch size** | 512 |
| **Epochs** | 50 |
| **Scheduler** | CosineAnnealingLR ($\eta_{\min} = 10^{-6}$) |
| **Gradient clipping** | max_norm = 1.0 |
| **Masking ratio** | 50% |
| **Feature loss weight ($\lambda$)** | 2.0 |
| **Input duration** | 60 seconds (12,000 samples @ 200 Hz) |
| **GPUs** | 4× (DataParallel) |

### 5.2 Ablation: Feature-Only Model

The `run_sanity_feat_only.sh` script tests feature prediction without reconstruction:

| Change | Value |
|--------|-------|
| `pretrain_tasks` | `['feature_pred']` (reconstruction disabled) |
| Learning rate | $5 \times 10^{-5}$ (reduced) |
| Epochs | 30 |

**Hypothesis Tested:** This ablation validates that EEG features are learnable directly from raw signals, isolating the contribution of the feature prediction branch. If the feature-only model achieves high feature R² but poor downstream performance compared to the full model, it demonstrates that reconstruction provides complementary learning signals.

### 5.3 Intra-Epoch Validation

To monitor training stability, validation is performed multiple times per epoch:
```python
val_freq_split = 10  # Validate 10 times per epoch
```

Each validation uses a mini-batch of 50 samples for efficiency:
```python
evaluate(..., limit_batches=50, verbose=False)
```

---

## 6. Downstream Experiments

### 6.1 TUAB Linear Probe (experiments/tuab_lp)

**Task:** Binary classification of abnormal EEG recordings

**Dataset:** Temple University Abnormal EEG Corpus (TUAB)
- Train/Val/Test: 80/10/10 subject-level split
- Labels: 0 = normal, 1 = abnormal

**Feature Extraction Protocol:**
1. Load pretrained encoder with frozen weights
2. For each 60-second EEG segment:
   - Pass through encoder: $\mathbf{H} = f_\theta(\mathbf{X})$
   - Extract two representations:
     - **EEG Token:** Global average pooling: $\mathbf{h}_{\text{eeg}} = \text{mean}_{c,n}(\mathbf{H}) \in \mathbb{R}^{200}$
     - **Feat Token:** Cross-attention query output: $\mathbf{h}_{\text{feat}} \in \mathbb{R}^{200}$

```python
# From extract_features.py:118-144
eeg_feat = feats.mean(dim=[1, 2])  # GAP
query = model.feat_query.expand(B, -1, -1)
attn_output, _ = model.feat_attn(query, feats_flat, feats_flat)
feat_token = attn_output.squeeze(1)
```

**Classifier:** Logistic regression with LBFGS optimizer
```python
model = LogisticRegressionTorch(input_dim, num_classes=2)
optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=100)
```

**Evaluation Metrics:**
- Accuracy
- Balanced Accuracy (primary metric)
- AUC-ROC

**Data Efficiency Regimes:** Evaluated at 0.5%, 1%, 5%, 10%, 20%, 50%, 100% of training subjects

**Representations Evaluated:**
| Feature Type | Description | Dimension |
|--------------|-------------|-----------|
| `eeg` | GAP-pooled encoder output | 200 |
| `feat` | Cross-attention feature token | 200 |
| `full` | Concatenation of eeg + feat | 400 |
| `pred` | Predicted 62 features | 62 |

### 6.2 TUEV Linear Probe (experiments/tuev_lp)

**Task:** 6-class event classification

**Dataset:** TUH Events Corpus (TUEV)
- Train/Val/Test: 60/20/20 subject-level split
- Classes: 6 event types (one-hot encoded in data)

**Key Differences from TUAB:**
- Segment duration: 1 second (200 samples)
- Number of patches: $N = 1$
- Multiclass classification (6 classes)

**Evaluation:**
- Same feature extraction and linear probe protocol as TUAB
- Metrics: Accuracy, Balanced Accuracy
- LBFGS with increased iterations (max_iter=500) for convergence

**Implementation Note:**
```python
# From tuev_lp/run_lp_compare.py:52
optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, history_size=20)
```

StandardScaler normalization is applied before training:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 6.3 Feature Prediction Validation (experiments/feature_pred_validation)

**Purpose:** Quantify how well the model predicts individual EEG features, validating that masked modeling captures domain-relevant information.

**Protocol:**
1. Run pretrained model on validation set with zero masking
2. Compute per-feature metrics:
   - R² (coefficient of determination)
   - PCC (Pearson correlation coefficient)
   - MSE/RMSE

**Global R² Calculation (from `eval_features.py:235-241`):**
```python
# Accumulate across all samples
sst_per_channel = sum_y_sq - count * (mean_y ** 2)
r2_per_channel = 1 - (sum_sq_err / sst_per_channel)
```

**Feature Categories Analyzed:**
- Time-Domain (RMS, kurtosis, skewness, zero-crossing)
- Frequency-Domain (band powers, ratios)
- Spectral & Aperiodic (entropy, centroid, exponent)
- Non-Linear & Complexity (Hjorth parameters)
- Dynamics (standard deviation variants)

**Visualization:**
- Per-feature scatter plots (predicted vs. true)
- Radar charts comparing feature profiles
- Grouped bar charts by category

**What This Validates:**
High R² on interpretable features demonstrates that the learned representations encode clinically meaningful patterns, not just low-level signal statistics. The comparison between the full multi-task model and feature-only ablation isolates the contribution of joint training.

---

# Appendix

## A. Implementation Details

### A.1 Data Preprocessing

**Input Normalization (per-channel, per-segment):**
```python
# From tueg.py:333-335
mean = tensor.mean(dim=1, keepdim=True)
std = tensor.std(dim=1, keepdim=True)
tensor = (tensor - mean) / (std + 1e-6)
```

**Channel Selection:**
Standard 10-20 system with 19 channels:
```python
target_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                   'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                   'FZ', 'CZ', 'PZ']
```

**Sampling Rate:** 200 Hz

**Segment Duration:**
- Pretraining (TUEG): 60 seconds (12,000 samples)
- TUAB downstream: 60 seconds
- TUEV downstream: 1 second (200 samples)

### A.2 Patch Embedding Details

**Temporal Convolution Stack:**
```python
self.proj_in = nn.Sequential(
    nn.Conv2d(1, 25, kernel=(1, 49), stride=(1, 25), padding=(0, 24)),
    nn.GroupNorm(5, 25),
    nn.GELU(),
    nn.Conv2d(25, 25, kernel=(1, 3), stride=(1, 1), padding=(0, 1)),
    nn.GroupNorm(5, 25),
    nn.GELU(),
    nn.Conv2d(25, 25, kernel=(1, 3), stride=(1, 1), padding=(0, 1)),
    nn.GroupNorm(5, 25),
    nn.GELU(),
)
```

**FFT Parameters:**
- Normalization: 'forward' (divides by $N$)
- Frequency bins: $P/2 + 1 = 101$ for $P = 200$

**Spectral Projection:**
```python
self.spectral_proj = nn.Sequential(
    nn.Linear(self.freq_bins, d_model),
    nn.Dropout(0.1),
)
```

### A.3 Weight Initialization

```python
def _weights_init(self, m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

Feature query token: `nn.init.normal_(self.feat_query, std=0.02)`

### A.4 Regression Metrics Implementation

**Memory-Efficient Chunked Computation (from `util.py:211-292`):**
```python
def calc_regression_metrics(preds, target, chunk_size=256):
    # Global mean for R² calculation
    target_mean = torch.mean(target, dim=0)

    for chunk in chunks:
        # PCC: Cov(x,y) / (Std(x) * Std(y))
        vx = p_chunk - torch.mean(p_chunk, dim=1, keepdim=True)
        vy = t_chunk - torch.mean(t_chunk, dim=1, keepdim=True)
        pcc = torch.sum(vx * vy, dim=1) / (std_x * std_y)

        # R²: 1 - SS_res / SS_tot
        ss_res_accum += torch.sum((t_chunk - p_chunk) ** 2, dim=0)
        ss_tot_accum += torch.sum((t_chunk - target_mean) ** 2, dim=0)

    r2 = 1 - ss_res_accum / ss_tot_accum
    # Clamp to [-100, 1] for robustness
    r2 = torch.clamp(r2, min=-100.0, max=1.0)
```

### A.5 Feature Token Strategies

Three strategies are implemented for the cross-attention feature head:

| Strategy | Query Tokens | Output Dim | Description |
|----------|-------------|------------|-------------|
| `single` | 1 | 62 | Single query → MLP → all features |
| `all` | 62 | 62 | One query per feature → squeeze |
| `group` | K | 62 | K queries → concat → MLP |

**Implementation:**
```python
if self.feature_token_strategy == 'single':
    num_tokens, out_dim = 1, self.feature_dim
elif self.feature_token_strategy == 'all':
    num_tokens, out_dim = self.feature_dim, 1
elif self.feature_token_strategy == 'group':
    num_tokens, out_dim = self.feature_group_count, None
```

### A.6 Checkpoint Structure

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'config': config,
    'epoch': epoch,
    'best_metric': best_metric
}
```

## B. Dataset Statistics

### B.1 TUEG Pretraining Corpus

- **Source:** Temple University EEG Corpus
- **Split:** 95/5 (train/val) at subject level
- **Segment Duration:** 60 seconds
- **Sampling Rate:** 200 Hz
- **Channels:** 19 (10-20 montage)
- **Feature File:** 62 z-scored features per segment

### B.2 Feature Dimensions

| Category | Features | Examples |
|----------|----------|----------|
| Time-Domain | 6 | rms, peak_to_peak, skewness, kurtosis |
| Frequency-Domain | 12 | delta_power, alpha_power, alpha_theta_ratio |
| Spectral & Aperiodic | 8 | spectral_entropy, aperiodic_exponent |
| Non-Linear | 3 | hjorth_activity, hjorth_mobility, hjorth_complexity |
| Dynamics (std) | 33 | *_std variants of above |
| **Total** | **62** | |

## C. Hyperparameter Sensitivity

### C.1 Feature Loss Weight

The feature loss weight $\lambda$ controls the balance between reconstruction and feature prediction:

| $\lambda$ | Configuration |
|-----------|---------------|
| 1.0 | Default (pretrain.yaml) |
| 2.0 | Flagship fixed |
| 10.0 | Flagship cross-attention |
| Dynamic | $\lambda_{\text{dyn}} = \lambda \cdot \text{clamp}(L_{\text{recon}}/L_{\text{feat}}, 0.5, 50)$ |

### C.2 Learning Rate

| Configuration | Learning Rate |
|---------------|---------------|
| Flagship | $3 \times 10^{-4}$ |
| Feature-only | $5 \times 10^{-5}$ |
| Default | $1 \times 10^{-3}$ |

## D. Visualization Examples

### D.1 Reconstruction Visualization

The `visualize_eeg_batch` function generates comparison plots:
- Blue: Original signal
- Orange: Reconstructed signal
- Gray shading: Masked regions

### D.2 Feature Prediction Scatter Plots

Generated per-feature with:
- X-axis: True z-scored value
- Y-axis: Predicted value
- Red dashed line: Identity ($y = x$)
- Title: Feature name, R², PCC

---

## E. Reproducibility Checklist

- [x] Random seeds configurable (`seed=42` default)
- [x] Subject-level splits prevent data leakage
- [x] Dataset index caching for reproducible sample ordering
- [x] Checkpoint saving/resumption support
- [x] Configuration fully specified via YAML
- [x] WandB logging for experiment tracking

## F. Computational Requirements

| Resource | Specification |
|----------|---------------|
| GPUs | 4× NVIDIA GPU (DataParallel) |
| Batch Size | 512 |
| Training Duration | 50 epochs |
| Workers | 16 (persistent, prefetch=4) |
| Precision | FP32 |

---

*End of Technical Report*
