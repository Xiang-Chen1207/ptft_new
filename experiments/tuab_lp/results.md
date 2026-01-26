# Flagship Model TUAB Linear Probing Results

## Experimental Setup
- **Task**: TUAB Abnormal vs Normal Classification
- **Method**: Linear Probing (Frozen Backbone + Logistic Regression on GPU)
- **Feature Extraction**:
    - **Recon (Baseline)**: `cbramod` pretrained weights.
    - **Neuro-KE (Flagship)**: `flagship_cross_attn` weights.
- **Evaluation**: Subject-independent Split (Train on N subjects, Test on held-out subjects)
- **Metric**: Balanced Accuracy (%)

## Feature Definitions
1.  **EEG Token**: Global Average Pooling (GAP) of backbone patch tokens (Dim: 200).
2.  **Feat Token**: Learned Cross-Attention Query Tokens (Dim: 200).
3.  **Full Token**: Concatenation of EEG Token and Feat Token (Dim: 400).

## Results Table (Balanced Accuracy)

| Ratio (Subjects) | Recon (EEG)<br>Dim:200 | Neuro-KE (EEG)<br>Dim:200 | Neuro-KE (Feat)<br>Dim:200 | Neuro-KE (Full)<br>Dim:400 |
| :--- | :--- | :--- | :--- | :--- |
| **0.5%** (~9) | 61.40% | 64.34% | 66.50% | 65.60% |
| **1.0%** (~18) | 61.25% | 67.53% | 66.95% | 68.64% |
| **5.0%** (~92) | 66.24% | 70.75% | 71.83% | 71.17% |
| **10.0%** (~185) | 67.76% | 75.49% | 74.80% | 74.87% |
| **20.0%** (~370) | 70.28% | 78.14% | 77.33% | 78.26% |
| **50.0%** (~925) | 69.76% | 78.64% | 78.05% | 78.75% |
| **100%** (~1851) | 70.02% | 79.22% | 78.28% | 79.20% |

## Key Findings
1.  **Neuro-KE significantly outperforms Recon Baseline** across all data regimes, with a gap of ~9% in the full-shot setting.
2.  **Few-Shot Efficiency**: At 10% data (~185 subjects), Neuro-KE achieves ~75.5% accuracy, significantly higher than Baseline's full-shot performance (70.0%).
3.  **Feature Contribution**:
    -   **Feat Token** performs comparably to EEG Token, showing it captures strong discriminatory information.
    -   **Full Token** (concatenation) provides marginal gains in some regimes, suggesting high redundancy between EEG and Feat tokens, or that a simple linear classifier cannot fully exploit the combined information.

## Technical Details
-   **Classifier**: Logistic Regression (PyTorch implementation, L-BFGS optimizer).
-   **Convergence**: All runs converged successfully.
-   **Dimensions**: EEG (200), Feat (200), Full (400).
