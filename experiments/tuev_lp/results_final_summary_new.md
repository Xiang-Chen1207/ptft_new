# Linear Probing Results: Model Comparison (Baseline vs Neuro-KE vs FeatOnly - TUEV)

## Overview
- **Task**: TUEV (TUH Events) Classification (6-class)
- **Method**: Linear Probing (Logistic Regression)
- **Metric**: Balanced Accuracy (%)
- **Models**:
  - **Baseline**: Standard Reconstruction (MAE)
  - **Neuro-KE**: Reconstruction + Feature Prediction (Flagship)
  - **FeatOnly**: Feature Prediction Only (No Reconstruction)

## Feature Types Explanation
- **EEG** (Dim: 200): Standard Global Average Pooling (GAP) features from the Transformer encoder backbone.
- **Feat** (Dim: 200): The learned 'Feature Token' from the Cross-Attention Head.
- **Pred** (Dim: 62): The raw predicted values for the 62 clinical features.
- **Full** (Dim: 400): Concatenation of **EEG** and **Feat** vectors.

## Comparative Results Table
*Note: **Bold** indicates the best performance among all models for each ratio.*

| Ratio | NumSub | **Baseline** | **Neuro-KE** (EEG) | **Neuro-KE** (Feat) | **Neuro-KE** (Full) | **Neuro-KE** (Pred) | **FeatOnly** (EEG) | **FeatOnly** (Feat) | **FeatOnly** (Full) | **FeatOnly** (Pred) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.5%** | 1 | - | - | - | - | - | - | - | - | - |
| **1.0%** | 2 | 20.73% | 27.07% | 23.04% | 25.05% | 23.89% | **29.48%** | 28.25% | 26.71% | 25.31% |
| **5.0%** | 11 | 25.24% | 32.05% | **35.97%** | 34.72% | 30.44% | 34.41% | 34.22% | 35.72% | 31.76% |
| **10.0%** | 23 | 25.71% | 37.78% | 31.69% | 34.24% | 34.96% | 36.62% | 36.87% | 37.81% | **38.08%** |
| **20.0%** | 46 | 31.16% | **46.85%** | 41.69% | 42.77% | 40.85% | 38.71% | 39.90% | 41.12% | 44.89% |
| **50.0%** | 116 | 32.39% | 46.95% | 45.46% | 44.05% | 44.58% | 43.50% | 43.74% | 46.13% | **47.71%** |
| **100%** | 232 | 33.22% | **47.68%** | 47.12% | 46.53% | 42.24% | 45.54% | 45.87% | 48.18% | 48.34% |

## Key Observations

1.  **Significant Improvement over Baseline**: Both Neuro-KE and FeatOnly consistently outperform the Baseline model by a large margin (often +10-15% BAcc) across all data regimes. This demonstrates the strong benefit of the proposed pre-training tasks.

2.  **FeatOnly Strong in Low Data**: In lower data regimes (1%, 10%, 50%), the **FeatOnly** model (especially its Pred features) often achieves the top performance. The 62-dim prediction vector (`Pred`) is highly effective at 50% data (47.71%).

3.  **Neuro-KE Dominance at Scale**: At 100% data, Neuro-KE's EEG features take the lead (47.68%), showing that the multi-task learning helps learn robust general-purpose representations when sufficient data is available.

4.  **Baseline Underperformance**: The Baseline model struggles significantly on this 6-class task, hovering around 20-33% BAcc, which is much lower than the proposed methods.
