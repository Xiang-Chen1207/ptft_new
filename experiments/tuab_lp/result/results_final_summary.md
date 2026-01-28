# Linear Probing Results: Model Comparison (Baseline vs Neuro-KE vs FeatOnly)

## Overview
- **Task**: TUAB Abnormal/Normal Classification
- **Method**: Linear Probing (Logistic Regression)
- **Metric**: Balanced Accuracy (%)
- **Models**:
  - **Baseline**: Standard Reconstruction (MAE)
  - **Neuro-KE**: Reconstruction + Feature Prediction (Flagship)
  - **FeatOnly**: Feature Prediction Only (No Reconstruction)

## Feature Types Explanation
- **EEG** (Dim: 200): Standard Global Average Pooling (GAP) features from the Transformer encoder backbone. Represents the general signal encoding.
- **Feat** (Dim: 200): The learned 'Feature Token' from the Cross-Attention Head. This token is specifically trained to attend to clinical features.
- **Pred** (Dim: 62): The raw predicted values for the 62 clinical features (regression output). Highly interpretable and low-dimensional.
- **Full** (Dim: 400): Concatenation of **EEG** and **Feat** vectors. Combines general signal representation with targeted clinical knowledge (Reference).

## Comparative Results Table
*Note: **Bold** indicates the best performance among all models (Global Best). The best performance among comparable low-dimensional features (Dim ≤ 200) is also **bolded** if it differs from the global best.*

| Ratio | Samples | **Baseline** | **Neuro-KE** (EEG) | **Neuro-KE** (Feat) | **Neuro-KE** (Pred) | **FeatOnly** (EEG) | **FeatOnly** (Feat) | **FeatOnly** (Pred) | **Neuro-KE** (Full) | **FeatOnly** (Full) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.5%** | 9 | **61.62%** | 60.70% | 57.01% | 59.41% | 51.90% | 56.77% | 54.11% | **62.10%** | 52.43% |
| **1.0%** | 18 | 64.95% | 66.70% | **68.20%** | 65.68% | 63.84% | 64.83% | 63.21% | **68.91%** | 65.16% |
| **5.0%** | 92 | 68.31% | 73.88% | 73.90% | 74.22% | 74.03% | **74.78%** | 72.54% | 74.24% | 74.04% |
| **10.0%** | 185 | 69.57% | 76.79% | **76.96%** | 75.83% | 76.34% | 76.65% | 74.56% | **77.15%** | 76.43% |
| **20.0%** | 370 | 69.28% | 77.08% | **77.09%** | 76.10% | 76.95% | 76.81% | 74.89% | **77.82%** | 76.98% |
| **50.0%** | 925 | 69.29% | 77.04% | 77.03% | 76.10% | 77.45% | **77.46%** | 75.29% | **78.07%** | 77.51% |
| **100%** | 1851 | 69.80% | 78.27% | 78.17% | 77.05% | 78.57% | **78.64%** | 76.01% | **79.36%** | 78.82% |

## Key Observations

1. **Neuro-KE Dominance**: Neuro-KE (Full) consistently achieves top-tier performance, reaching **79.36%** at 100% data, a **+9.56%** improvement over Baseline.
2. **FeatOnly Competitiveness**: The FeatOnly model (without reconstruction) performs extremely well, suggesting feature prediction is a strong supervisory signal.
3. **Feature Efficacy**: The `Full` (concatenated) features generally offer the highest performance (Global Best), while individual `Feat` or `EEG` features (Dim 200) often achieve the best low-dimensional performance.
4. **Low Data Regime**: In lower data regimes (e.g., 5-10%), feature prediction based models maintain a significant lead over the Baseline.
