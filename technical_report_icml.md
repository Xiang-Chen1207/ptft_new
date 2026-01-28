# Technical Report: Neuro-KE (ICML Submission)

**Title:** Neuro-KE: Scaling EEG Foundation Models via Label-Free Knowledge Integration

This report details the methodology, architecture, and experimental setup for **Neuro-KE**, a framework designed to enhance EEG foundation models by integrating domain-specific signal priors. This document is structured to support the ICML submission, providing core narrative elements for the main text and rigorous implementation details for the appendix.

---

## Part 1: Main Text Essentials

### 1. Abstract
Foundational Models (FMs) for Electroencephalography (EEG) have shown promise in learning generalized representations from large-scale datasets. However, current approaches primarily rely on raw signal reconstruction or scarce supervised labels, often neglecting the rich, domain-specific prior knowledge encapsulated in decades of signal processing research. In this work, we introduce **Neuro-KE (Neuro-Knowledge Engine)**, a plug-and-play, label-free framework designed to seamlessly integrate comprehensive signal characteristics into the pre-training of EEG foundation models. Neuro-KE aggregates a vast array of morphometric, spectral, non-linear, and aperiodic features—effectively distilling historical domain expertise into a unified knowledge base. We demonstrate that Neuro-KE significantly enhances model generalization and robustness, particularly in label-scarce downstream tasks, offering a rigorous pathway to embed domain-invariant signal dynamics into modern deep learning architectures.

### 2. Methodology: The Neuro-KE Framework

**2.1. The Knowledge Engine (Target Features)**
Unlike standard reconstruction targets, Neuro-KE forces the model to encode clinically and physically meaningful patterns. We define a **comprehensive signal descriptor manifold** comprising 62 z-scored features across four categories:
*   **Time-Domain Dynamics**: Amplitude statistics (RMS, Peak-to-Peak, Mean Abs Amplitude), Statistical Moments (Skewness, Kurtosis), and Zero-Crossing Rates.
*   **Frequency-Domain Power**: Absolute and Relative Band Powers (Delta, Theta, Alpha, Beta, Gamma) and key Power Ratios (Theta/Beta, Delta/Theta).
*   **Spectral & Aperiodic Structure**: Spectral Entropy, Spectral Centroid, Peak Frequency, Individual Alpha Frequency (IAF), and Aperiodic Exponents (capturing 1/f background activity).
*   **Non-Linear Complexity**: Hjorth Parameters (Activity, Mobility, Complexity).

**2.2. Architecture: Decoupled Feature Guidance**
To integrate these priors without interfering with the primary signal reconstruction task, we employ a **Multi-task Architecture with Cross-Attention Decoupling**:

1.  **Backbone Encoder ($E$)**: A standard Transformer processes masked EEG patches ($X_{masked}$) to produce latent representations $Z \in \mathbb{R}^{N \times D}$.
2.  **Feature Query Token ($Q_{feat}$)**: A learnable query vector initialized to represent the global signal state.
3.  **Cross-Attention Head**: The Feature Token queries the patch-level representations $Z$ to aggregate global information:
    $$ h_{feat} = \text{CrossAttn}(Q_{feat}, Z, Z) $$
    This vector $h_{feat}$ is then projected to predict the 62-dimensional feature vector $\hat{y}$.

**2.3. Optimization Objective**
The total pretraining loss is a weighted sum of the Masked Reconstruction Loss ($\mathcal{L}_{rec}$) and the Feature Prediction Loss ($\mathcal{L}_{feat}$):
$$ \mathcal{L} = \mathcal{L}_{rec} + \lambda \cdot \mathcal{L}_{feat} $$
In our flagship configuration, we set $\lambda = 2.0$ to emphasize the importance of learning high-level signal semantics alongside low-level reconstruction.

---

## Part 2: Appendix - Implementation Details

### A. Model Hyperparameters

The backbone is a `CBraModBackbone` (Transformer Encoder) with the following specifications:

| Component | Hyperparameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Input** | `input_size` | 12,000 | 60 seconds @ 200 Hz |
| | `patch_size` | 200 | Time-domain patch size |
| | `seq_len` | 60 | Number of patches ($12000 / 200$) |
| **Encoder** | `d_model` | 200 | Hidden dimension |
| | `n_layer` | 12 | Transformer layers |
| | `nhead` | 8 | Attention heads |
| | `dim_feedforward` | 800 | FFN expansion dimension |
| | `dropout` | 0.1 | Regularization rate |
| **Head** | `feature_token_type` | `cross_attn` | Decoupling mechanism |
| | `feature_token_strategy` | `single` | Single global query token |

**Patch Embedding Details**:
The input processing pipeline uses a hybrid approach:
1.  **Convolutional Projection**: 2D Convolutions with GroupNorms to project time-domain signals.
2.  **Spectral Embedding**: FFT-based spectral magnitude projection added to temporal embeddings to capture frequency priors explicitly.

### B. Pretraining Experimental Setup

**Dataset**:
*   **Source**: Temple University EEG Corpus (TUEG).
*   **Preprocessing**: Signals are segmented into **60-second clips**, resampled to 200 Hz, and z-score normalized.

**Training Configuration ("Flagship Fixed")**:
*   **Batch Size**: 512 (distributed across 4 GPUs).
*   **Optimizer**: AdamW (`lr=3e-4`, `weight_decay=0.05`).
*   **Scheduler**: Cosine Annealing (50 epochs).
*   **Masking**: Random masking with a **50% ratio**.
*   **Loss Weight**: $\lambda=2.0$ for feature prediction.

### C. Ablation Studies & Baselines

We compare the Neuro-KE Flagship model against the following baselines to validate our design choices:

1.  **Baseline: Reconstruction Only**
    *   **Config**: `pretrain_tasks=['reconstruction']`.
    *   **Purpose**: Standard MAE baseline to measure the added value of knowledge integration.

2.  **Ablation: Global Average Pooling (GAP)**
    *   **Config**: `feature_token_type='gap'`.
    *   **Mechanism**: Replaces Cross-Attention with simple averaging of patch tokens.
    *   **Hypothesis**: Cross-Attention provides better decoupling than rigid pooling.

3.  **Sanity Check: Feature Prediction Only**
    *   **Config**: `pretrain_tasks=['feature_pred']`.
    *   **Hyperparameters**: Lower learning rate (`lr=5e-5`), 30 epochs.
    *   **Purpose**: Validates that the handcrafted features are deterministically predictable from the raw signal (verifying $R^2 > 0$).
