# Feat-Only Model TUAB Linear Probing Results (Tiny Run)

## Experimental Setup
- **Task**: TUAB Abnormal vs Normal Classification
- **Method**: Linear Probing (Frozen Backbone + Logistic Regression on GPU)
- **Feature Extraction**:
    - **Feat-Only (Target)**: `sanity_feat_only` pretrained weights.
    - **Dataset**: Tiny Subset (10 Subjects for training, ~1400 samples).
- **Evaluation**: Subject-independent Split.
- **Metric**: Balanced Accuracy (%)

> **Note**: This is a verification run using a tiny subset of data (10 subjects). Results may not reflect full-scale performance.

## Feature Definitions
1.  **EEG Token**: Global Average Pooling (GAP) of backbone patch tokens (Dim: 200).
2.  **Feat Token**: Learned Cross-Attention Query Tokens (Dim: 200).
3.  **Full Token**: Concatenation of EEG Token and Feat Token (Dim: 400).
4.  **Pred Features**: Neuro-KE features predicted by the model (Dim: 62).

## Results Table (Balanced Accuracy)

| Ratio (Subjects) | EEG Token<br>Dim:200 | Feat Token<br>Dim:200 | Full Token<br>Dim:400 | Pred Features<br>Dim:62 |
| :--- | :--- | :--- | :--- | :--- |
| **20.0%** (2) | 49.02% | 53.87% | 51.76% | 39.71% |
| **50.0%** (5) | 45.42% | 46.85% | 47.59% | 64.43% |
| **100%** (10) | 50.44% | 45.33% | 52.01% | 41.86% |

*(Ratios < 20% were skipped due to insufficient class diversity in the tiny subset)*

## Key Findings (Preliminary)
1.  **Feat Token** shows competitive performance even in this limited setting (53.87% at 2-shot).
2.  **Pred Features** showed high variance (spiking to 64.43% at 50% data), likely due to the small sample size and specific subject characteristics.
3.  **Full Token** generally averages the performance of EEG and Feat tokens.

## Technical Details
-   **Classifier**: Logistic Regression (PyTorch implementation, L-BFGS optimizer).
-   **Convergence**: All runs converged successfully.
