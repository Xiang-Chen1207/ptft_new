# Linear Probing Results: Model Comparison (Baseline vs Neuro-KE vs FeatOnly - 60s)

## Overview
- **Task**: TUAB Abnormal/Normal Classification
- **Method**: Linear Probing (Logistic Regression)
- **Metric**: Balanced Accuracy (%)
- **Models**:
  - **Baseline**: Standard Reconstruction (MAE)
  - **Neuro-KE**: Reconstruction + Feature Prediction (Flagship)
  - **FeatOnly**: Feature Prediction Only (No Reconstruction)

## Feature Types Explanation
- **EEG**: Standard Global Average Pooling (GAP) features from the Transformer encoder backbone.
- **Feat**: The learned 'Feature Token' from the Cross-Attention Head.
- **Pred**: The raw predicted values for the clinical features.
- **Full**: Concatenation of **EEG** and **Feat** vectors.

## Comparative Results Table
*Note: **Bold** indicates the best performance in each row.*

| Ratio | Samples | **Baseline** | **Neuro-KE** (EEG) | **Neuro-KE** (Feat) | **Neuro-KE** (Pred) | **Neuro-KE** (Full) | **FeatOnly** (EEG) | **FeatOnly** (Feat) | **FeatOnly** (Pred) | **FeatOnly** (Full) |
|:---:|:---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.5%** | 9 | **62.47%** | 60.75% | 61.92% | 58.18% | 60.00% | 53.67% | 56.83% | 59.10% | 54.86% |
| **1.0%** | 18 | **67.75%** | 64.26% | 66.15% | 64.32% | 64.94% | 63.71% | 63.78% | 64.40% | 63.80% |
| **5.0%** | 92 | 70.35% | 69.89% | 70.33% | 70.22% | 69.65% | 71.27% | 71.01% | **71.87%** | 71.42% |
| **10.0%** | 185 | 70.59% | 71.75% | 72.15% | 71.66% | 72.06% | 74.65% | 74.44% | 73.65% | **75.51%** |
| **20.0%** | 370 | 66.74% | 72.53% | 72.56% | 71.81% | 72.84% | 74.93% | 75.11% | 74.04% | **75.55%** |
| **50.0%** | 925 | 70.74% | 72.81% | 73.06% | 71.78% | 73.43% | 75.06% | 75.14% | 74.20% | **75.79%** |
| **100%** | 1851 | 71.65% | 74.27% | 74.17% | 72.99% | 74.57% | 76.41% | 76.33% | 75.74% | **77.05%** |

## Key Observations
*(Automatically generated placeholder - please update with analysis)*
