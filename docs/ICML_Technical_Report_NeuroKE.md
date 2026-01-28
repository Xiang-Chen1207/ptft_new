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

$$M_{c,n} \sim \text{Bernoulli}(0.5) \quad \forall c \in [1,C], n \in [1,N]$$

The mask is generated independently for each spatial-temporal patch, meaning a patch at channel $c$ and temporal position $n$ may be masked while adjacent patches remain visible.

### 2.2 Patch Embedding

For masked patches, the input is replaced with a learnable mask token $\mathbf{e}_{\text{mask}} \in \mathbb{R}^{P}$ (initialized to zeros). The patch embedding module processes inputs through three parallel pathways:

**Temporal Embedding:** A three-layer 1D convolution stack with GroupNorm and GELU activation:

$$\mathbf{h}_{\text{temp}} = \text{Conv}_3 \circ \text{Conv}_2 \circ \text{Conv}_1(\tilde{\mathbf{x}})$$

where $\tilde{\mathbf{x}}$ denotes the masked input.

**Spectral Embedding:** FFT-based frequency encoding:

$$\mathbf{h}_{\text{spec}} = \mathbf{W}_{\text{spec}} \cdot |\mathcal{F}(\tilde{\mathbf{x}})|$$

where $\mathcal{F}$ denotes the discrete Fourier transform and $|\cdot|$ extracts the magnitude spectrum.

**Positional Encoding:** Depthwise separable convolution with kernel size $(19, 7)$:

$$\mathbf{h}_{\text{pos}} = \text{DWConv}(\mathbf{h}_{\text{temp}})$$

**Feature Fusion:**

$$\mathbf{h} = \mathbf{h}_{\text{temp}} + \mathbf{h}_{\text{spec}} + \mathbf{h}_{\text{pos}}$$

### 2.3 Reconstruction Head

The encoder processes all tokens through the Transformer. Reconstruction is performed via a linear projection:

$$\hat{\mathbf{X}} = \mathbf{W}_{\text{recon}} \cdot \mathbf{H} + \mathbf{b}_{\text{recon}}$$

where $\mathbf{H} \in \mathbb{R}^{B \times C \times N \times d}$ is the encoder output and $\hat{\mathbf{X}} \in \mathbb{R}^{B \times C \times N \times P}$ is the reconstructed signal.

### 2.4 Feature-Guided Supervision

The feature prediction module employs cross-attention to extract a global representation:

**Cross-Attention Feature Extraction:**

1. Initialize learnable query token: $\mathbf{Q}_{\text{feat}} \in \mathbb{R}^{1 \times d}$

2. Flatten encoder output: $\mathbf{H}_{\text{flat}} \in \mathbb{R}^{B \times (C \cdot N) \times d}$

3. Cross-attention with 4 heads:

$$\mathbf{h}_{\text{feat}} = \text{MultiHead}(\mathbf{Q}_{\text{feat}}, \mathbf{H}_{\text{flat}}, \mathbf{H}_{\text{flat}})$$

4. MLP prediction:

$$\hat{\mathbf{z}} = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{h}_{\text{feat}} + \mathbf{b}_1) + \mathbf{b}_2$$

**Feature Target Details:**

The 62 z-scored features span four categories:
- **Time-Domain:** RMS, peak-to-peak amplitude, skewness, kurtosis, zero-crossing rate
- **Frequency-Domain:** Band powers (delta, theta, alpha, beta, gamma), band ratios
- **Spectral & Aperiodic:** Spectral entropy, centroid frequency, aperiodic exponent
- **Non-Linear & Complexity:** Hjorth parameters (activity, mobility, complexity)

Features are pre-computed using standard signal processing libraries and z-scored across the dataset, with values clipped to $[-6, 6]$ for robustness.

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

Each Transformer layer employs a custom criss-cross attention mechanism that factorizes attention into spatial and temporal components.

Given input $\mathbf{X} \in \mathbb{R}^{B \times C \times N \times d}$, we split the embedding dimension:

$$\mathbf{X}_s = \mathbf{X}[:,:,:,:d/2], \quad \mathbf{X}_t = \mathbf{X}[:,:,:,d/2:]$$

**Spatial Attention** (across channels within each time patch):

$$\mathbf{X}_s' = \text{MHA}_s(\mathbf{X}_s^{(B \cdot N, C, d/2)})$$

**Temporal Attention** (across time patches within each channel):

$$\mathbf{X}_t' = \text{MHA}_t(\mathbf{X}_t^{(B \cdot C, N, d/2)})$$

**Concatenation:**

$$\mathbf{X}' = [\mathbf{X}_s'; \mathbf{X}_t']$$

This factorization reduces computational complexity from $O((C \cdot N)^2)$ to $O(C^2 \cdot N + N^2 \cdot C)$ while explicitly modeling both spatial (cross-channel) and temporal dependencies.

### 3.3 Complete Forward Pass

$$\mathbf{X} \xrightarrow{\text{Mask}} \tilde{\mathbf{X}} \xrightarrow{\text{Embed}} \mathbf{H}^{(0)} \xrightarrow{\text{12× Transformer}} \mathbf{H}^{(L)} \xrightarrow[\text{Feature Head}]{\text{Recon Head}} (\hat{\mathbf{X}}, \hat{\mathbf{z}})$$

---

## 4. Training Objective

### 4.1 Loss Formulation

The total loss combines masked reconstruction and feature prediction:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{feat}}$$

**Reconstruction Loss (Masked MSE):**

Only masked patches contribute to the reconstruction loss:

$$\mathcal{L}_{\text{recon}} = \frac{\sum_{c,n} M_{c,n} \cdot \|\hat{\mathbf{x}}_{c,n} - \mathbf{x}_{c,n}\|_2^2}{\sum_{c,n} M_{c,n}}$$

**Feature Prediction Loss:**

Standard MSE between predicted and target z-scored features:

$$\mathcal{L}_{\text{feat}} = \frac{1}{D_f} \sum_{d=1}^{D_f} (\hat{z}_d - z_d)^2$$

### 4.2 Dynamic Loss Balancing

An optional dynamic weighting scheme balances gradient magnitudes:

$$\lambda_{\text{dyn}} = \lambda \cdot \text{clamp}\left(\frac{\mathcal{L}_{\text{recon}}}{\mathcal{L}_{\text{feat}} + \epsilon}, 0.5, 50.0\right)$$

This prevents either loss from dominating training by scaling the feature loss weight proportionally to the reconstruction loss magnitude.

---

## 5. Training Protocol

### 5.1 Flagship Configuration

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

The feature-only ablation tests feature prediction without reconstruction:

| Change | Value |
|--------|-------|
| Pretrain tasks | Feature prediction only (reconstruction disabled) |
| Learning rate | $5 \times 10^{-5}$ (reduced) |
| Epochs | 30 |

**Hypothesis Tested:** This ablation validates that EEG features are learnable directly from raw signals, isolating the contribution of the feature prediction branch. If the feature-only model achieves high feature $R^2$ but poor downstream performance compared to the full model, it demonstrates that reconstruction provides complementary learning signals.

### 5.3 Intra-Epoch Validation

To monitor training stability, validation is performed 10 times per epoch using mini-batches of 50 samples for efficiency.

---

## 6. Downstream Experiments

### 6.1 TUAB Linear Probe

**Task:** Binary classification of abnormal EEG recordings

**Dataset:** Temple University Abnormal EEG Corpus (TUAB)
- Train/Val/Test: 80/10/10 subject-level split
- Labels: 0 = normal, 1 = abnormal

**Feature Extraction Protocol:**
1. Load pretrained encoder with frozen weights
2. For each 60-second EEG segment, extract:
   - **EEG Token:** $\mathbf{h}_{\text{eeg}} = \frac{1}{C \cdot N}\sum_{c,n} \mathbf{H}_{c,n} \in \mathbb{R}^{200}$ (global average pooling)
   - **Feat Token:** $\mathbf{h}_{\text{feat}} \in \mathbb{R}^{200}$ (cross-attention query output)

**Classifier:** Logistic regression with L-BFGS optimizer

**Evaluation Metrics:** Accuracy, Balanced Accuracy (primary), AUC-ROC

**Data Efficiency Regimes:** Evaluated at 0.5%, 1%, 5%, 10%, 20%, 50%, 100% of training subjects

**Representations Evaluated:**

| Feature Type | Description | Dimension |
|--------------|-------------|-----------|
| `eeg` | GAP-pooled encoder output | 200 |
| `feat` | Cross-attention feature token | 200 |
| `full` | Concatenation of eeg + feat | 400 |
| `pred` | Predicted 62 features | 62 |

### 6.2 TUEV Linear Probe

**Task:** 6-class event classification

**Dataset:** TUH Events Corpus (TUEV)
- Train/Val/Test: 60/20/20 subject-level split
- Classes: 6 event types

**Key Differences from TUAB:**
- Segment duration: 1 second (200 samples)
- Number of patches: $N = 1$
- Multiclass classification (6 classes)

**Evaluation:** Same feature extraction and linear probe protocol as TUAB, with L-BFGS iterations increased to 500 for convergence. StandardScaler normalization applied before training.

### 6.3 Feature Prediction Validation

**Purpose:** Quantify how well the model predicts individual EEG features, validating that masked modeling captures domain-relevant information.

**Protocol:**
1. Run pretrained model on validation set with zero masking
2. Compute per-feature metrics:
   - $R^2$ (coefficient of determination)
   - PCC (Pearson correlation coefficient)
   - MSE/RMSE

**Global $R^2$ Calculation:**

$$R^2_d = 1 - \frac{\sum_i (\hat{z}_{i,d} - z_{i,d})^2}{\sum_i (z_{i,d} - \bar{z}_d)^2}$$

**Feature Categories Analyzed:**
- Time-Domain (RMS, kurtosis, skewness, zero-crossing)
- Frequency-Domain (band powers, ratios)
- Spectral & Aperiodic (entropy, centroid, exponent)
- Non-Linear & Complexity (Hjorth parameters)

**What This Validates:**
High $R^2$ on interpretable features demonstrates that the learned representations encode clinically meaningful patterns, not just low-level signal statistics.

---

# Appendix

## A. Implementation Details

### A.1 Data Preprocessing

**Input Normalization:** Per-channel, per-segment z-score normalization:

$$\tilde{x}_{c,t} = \frac{x_{c,t} - \mu_c}{\sigma_c + \epsilon}$$

where $\mu_c$ and $\sigma_c$ are the mean and standard deviation of channel $c$.

**Channel Selection:** Standard 10-20 system with 19 channels: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ

**Sampling Rate:** 200 Hz

**Segment Duration:**
- Pretraining (TUEG): 60 seconds (12,000 samples)
- TUAB downstream: 60 seconds
- TUEV downstream: 1 second (200 samples)

### A.2 Patch Embedding Details

**Temporal Convolution Stack:**
- Layer 1: kernel=49, stride=25, channels: 1→25
- Layer 2: kernel=3, stride=1, channels: 25→25
- Layer 3: kernel=3, stride=1, channels: 25→25
- Each layer followed by GroupNorm(5) and GELU

**FFT Parameters:**
- Normalization: 'forward' (divides by $N$)
- Frequency bins: $\lfloor P/2 \rfloor + 1 = 101$ for $P = 200$

### A.3 Weight Initialization

- Linear/Conv layers: Kaiming normal initialization
- Feature query token: $\mathcal{N}(0, 0.02)$
- BatchNorm: weight=1, bias=0

### A.4 Feature Token Strategies

Three strategies implemented for the cross-attention feature head:

| Strategy | Query Tokens | Output Dim | Description |
|----------|-------------|------------|-------------|
| `single` | 1 | 62 | Single query → MLP → all features |
| `all` | 62 | 62 | One query per feature → squeeze |
| `group` | $K$ | 62 | $K$ queries → concat → MLP |

## B. Dataset Statistics

### B.1 TUEG Pretraining Corpus

- **Source:** Temple University EEG Corpus
- **Split:** 95/5 (train/val) at subject level
- **Segment Duration:** 60 seconds
- **Sampling Rate:** 200 Hz
- **Channels:** 19 (10-20 montage)
- **Feature Dimension:** 62 z-scored features per segment

### B.2 Feature Dimensions

| Category | Count | Examples |
|----------|-------|----------|
| Time-Domain | 6 | rms, peak_to_peak, skewness, kurtosis |
| Frequency-Domain | 12 | delta_power, alpha_power, alpha_theta_ratio |
| Spectral & Aperiodic | 8 | spectral_entropy, aperiodic_exponent |
| Non-Linear | 3 | hjorth_activity, hjorth_mobility, hjorth_complexity |
| Dynamics (std) | 33 | *_std variants of above |
| **Total** | **62** | |

## C. Hyperparameter Sensitivity

### C.1 Feature Loss Weight

| $\lambda$ | Configuration |
|-----------|---------------|
| 1.0 | Default |
| 2.0 | Flagship fixed |
| 10.0 | Flagship cross-attention |
| Dynamic | $\lambda_{\text{dyn}} = \lambda \cdot \text{clamp}(\mathcal{L}_{\text{recon}}/\mathcal{L}_{\text{feat}}, 0.5, 50)$ |

### C.2 Learning Rate

| Configuration | Learning Rate |
|---------------|---------------|
| Flagship | $3 \times 10^{-4}$ |
| Feature-only | $5 \times 10^{-5}$ |
| Default | $1 \times 10^{-3}$ |

## D. Reproducibility Checklist

- [x] Random seeds configurable (seed=42 default)
- [x] Subject-level splits prevent data leakage
- [x] Dataset index caching for reproducible sample ordering
- [x] Checkpoint saving/resumption support
- [x] Configuration fully specified via YAML
- [x] WandB logging for experiment tracking

## E. Computational Requirements

| Resource | Specification |
|----------|---------------|
| GPUs | 4× NVIDIA GPU (DataParallel) |
| Batch Size | 512 |
| Training Duration | 50 epochs |
| Workers | 16 (persistent, prefetch=4) |
| Precision | FP32 |

---

*End of Technical Report*
