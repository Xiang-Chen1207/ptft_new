# Linear Probing Results: Model Comparison (Baseline vs Neuro-KE vs FeatOnly - 60s)

## Overview
- **Task**: TUAB Abnormal/Normal Classification
- **Method**: Linear Probing (Logistic Regression)
- **Metric**: Balanced Accuracy (%)
- **Models**:
  - **Baseline**: Standard Reconstruction (MAE)
  - **Neuro-KE**: Reconstruction + Feature Prediction (Flagship) - Using Full Query (62 tokens) & Pooled Query (mean pool)
  - **FeatOnly**: Feature Prediction Only (No Reconstruction) - Using Single Query

## Feature Types Explanation
- **EEG** (Dim: 200): Standard Global Average Pooling (GAP) features from the Transformer encoder backbone. Represents the general signal encoding.
- **Feat** (Dim: 200): The learned 'Feature Token' from the Cross-Attention Head. This token is specifically trained to attend to clinical features.
- **Pred** (Dim: 62): The raw predicted values for the 62 clinical features (regression output). Highly interpretable and low-dimensional.
- **Full** (Dim: 400): Concatenation of **EEG** and **Feat** vectors. Combines general signal representation with targeted clinical knowledge (Reference).

## Comparative Results Table
*Note: **Bold** indicates the best performance among all models (Global Best). The best performance among comparable low-dimensional features (Dim ≤ 200) is also **bolded** if it differs from the global best.*

| Ratio | Samples | **Baseline** | **Neuro-KE** (EEG) | **Neuro-KE** (Feat) | **Neuro-KE** (Pool Feat) | **FeatOnly** (EEG) | **FeatOnly** (Feat) | **FeatOnly** (Pred) | **Neuro-KE** (Full) | **Neuro-KE** (Pool Full) | **FeatOnly** (Full) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.5%** | 9 | **61.64%** | 62.79% | 60.70% | 59.79% | 55.03% | 57.21% | **64.07%** | 58.70% | 60.02% | 57.38% |
| **1.0%** | 18 | 64.93% | **68.31%** | 67.92% | 67.01% | 64.61% | 64.75% | 63.30% | 67.62% | 68.13% | 65.10% |
| **5.0%** | 92 | 68.21% | **75.30%** | 74.68% | 74.61% | 72.25% | 71.79% | 70.23% | 74.58% | 74.43% | 72.38% |
| **10.0%** | 185 | 69.65% | **77.07%** | 76.31% | 76.37% | 75.40% | 74.18% | 72.11% | 76.20% | 76.28% | 74.82% |
| **20.0%** | 370 | 68.80% | **76.65%** | 76.05% | 76.15% | 75.36% | 74.19% | 72.71% | 76.41% | 76.28% | 75.39% |
| **50.0%** | 925 | 69.16% | 76.53% | 76.15% | 76.28% | 75.42% | 75.01% | 73.28% | **76.64%** | 76.61% | 75.52% |
| **100%** | 1851 | 69.97% | 77.60% | 77.41% | 77.45% | 76.68% | 76.27% | 74.23% | 77.81% | **77.84%** | 76.82% |

## Key Observations

1. **Neuro-KE Dominance**: Neuro-KE (both Full and Pooled versions) consistently outperforms both Baseline and FeatOnly models across almost all data regimes. At 100% data, it achieves **77.84%** Balanced Accuracy, which is a substantial **+7.87%** improvement over the Baseline (69.97%).

2. **Feature Efficacy**: 
    - The `Full` (concatenated EEG + Feat) representation generally yields the highest performance, especially in high data regimes.
    - Interestingly, Neuro-KE's `EEG` features (backbone GAP) perform exceptionally well (often best or second best), suggesting the backbone itself learned powerful representations from the auxiliary tasks.

3. **Pooling Strategy**: The Pooled Feature (`Flagship(Pool) feat`) performs comparably to the Full Feature (`Flagship(Full) feat`), with differences often less than 0.1%. This validates that mean pooling the 62 query tokens into a single 200-dim vector preserves the critical diagnostic information while significantly reducing dimensionality (12400 -> 200).

4. **FeatOnly Performance**: While FeatOnly (Single Query) performs well (reaching 76.82%), it consistently lags behind Neuro-KE (Full Query) by about 1.0%, indicating that the reconstruction task and the richer Full Query mechanism contribute positively to the learned representation.

5. **Low Data Regime**: In very low data regimes (0.5%), the `FeatOnly (Pred)` (62-dim) features achieve the highest accuracy (64.07%), likely due to their low dimensionality and direct clinical relevance preventing overfitting.
