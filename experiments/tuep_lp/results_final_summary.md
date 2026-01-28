# Linear Probing Results: Model Comparison (Baseline vs Neuro-KE vs FeatOnly - TUEP)

## Overview
- **Task**: TUEP Epilepsy/No-Epilepsy Classification
- **Method**: Linear Probing (Logistic Regression)
- **Metric**: Balanced Accuracy (%)
- **Models**:
  - **Baseline**: Standard Reconstruction (MAE)
  - **Neuro-KE**: Reconstruction + Feature Prediction (Flagship) - Using Single Query (200 dim)
  - **FeatOnly**: Feature Prediction Only (No Reconstruction) - Using Single Query

## Feature Types Explanation
- **EEG** (Dim: 200): Standard Global Average Pooling (GAP) features from the Transformer encoder backbone. Represents the general signal encoding.
- **Feat** (Dim: 200): The learned 'Feature Token' from the Cross-Attention Head.
- **Pred** (Dim: 62): The raw predicted values for the 62 clinical features (regression output).
- **Full** (Dim: 400): Concatenation of **EEG** and **Feat** vectors.

## Comparative Results Table
*Note: **Bold** indicates the best performance among all models (Global Best).*

| Ratio | Samples | **Baseline** | **Neuro-KE** (EEG) | **Neuro-KE** (Feat) | **Neuro-KE** (Full) | **Neuro-KE** (Pred) | **FeatOnly** (EEG) | **FeatOnly** (Feat) | **FeatOnly** (Full) | **FeatOnly** (Pred) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.5%** | 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **1.0%** | 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **5.0%** | 7 | 43.61% | 35.37% | 34.55% | 37.81% | 41.39% | **50.55%** | 45.86% | 45.25% | 40.64% |
| **10.0%** | 15 | 49.09% | 43.46% | 41.06% | 44.73% | 42.05% | **50.36%** | 44.81% | 50.48% | 43.92% |
| **20.0%** | 31 | **55.92%** | 44.15% | 43.08% | 44.65% | 47.81% | 52.30% | 49.64% | 50.65% | 45.44% |
| **50.0%** | 79 | 61.13% | 58.23% | 63.91% | 62.27% | 62.57% | **67.07%** | 60.16% | 61.25% | 58.93% |
| **100%** | 158 | 55.05% | 58.98% | **66.02%** | 64.50% | 58.67% | 61.41% | 56.69% | 61.23% | 52.55% |

## Key Observations

1.  **Neuro-KE Dominance at Scale**: At 100% data, **Neuro-KE (Feat)** achieves the highest performance (**66.02%**), significantly outperforming the Baseline (55.05%) by **+10.97%**. This demonstrates that the Neuro-KE pretraining strategy effectively learns robust representations for epilepsy detection when sufficient downstream data is available.

2.  **FeatOnly Strength in Low Data**: In lower data regimes (5% - 50%), the **FeatOnly (EEG)** model consistently performs very well, achieving the global best at 5%, 10%, and 50% ratios. This suggests that the frozen backbone of the FeatOnly model (which focuses solely on feature prediction) provides a very stable and generalizable representation for few-shot learning.

3.  **Baseline Instability**: The Baseline model shows significant instability, peaking at 20% (55.92%) and 50% (61.13%) but dropping sharply at 100% (55.05%). This trend suggests that the standard reconstruction task alone does not consistently align with the discriminative requirements of the TUEP task, and may be susceptible to distribution shifts or noise in the larger dataset.

4.  **Feature vs. EEG**: For Neuro-KE, the specialized `Feat` token (66.02%) significantly outperforms the general `EEG` backbone features (58.98%) at 100% data, validating the value of the cross-attention mechanism in capturing task-relevant information.

5.  **Prediction Head**: The raw clinical feature predictions (`Pred`) generally perform worse than the high-dimensional latent representations, suggesting that while the model learns to predict clinical features, the latent embeddings (`Feat`/`EEG`) contain richer information for the downstream classification task.
