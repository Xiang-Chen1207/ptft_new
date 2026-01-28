# Radar Charts Visualization Guide

The images in this directory are **Radar Charts** (also known as Spider Charts) designed to visually evaluate the model's ability to reconstruct/predict clinical EEG features for individual samples.

## What do these charts show?

Each chart compares the **Ground Truth** (True Feature Values) vs. **Prediction** (Model Output) for a single EEG sample across a subset of features.

- **Lines**:
  - **<span style="color:blue">Blue Line</span>**: Ground Truth (The actual calculated features from the EEG signal).
  - **<span style="color:orange">Orange/Red Line</span>**: Prediction (The values predicted by the model's feature head).
- **Overlap**: A high degree of overlap between the two polygons indicates high prediction accuracy. Discrepancies show where the model is underestimating or overestimating specific features.

## Sample Selection

To provide a representative view of performance, the script selects 4 specific samples from the validation set based on their total Mean Squared Error (MSE):

1.  **Best #1, #2, #3**: The top 3 samples where the model had the lowest error. These show the model's "upper bound" capability.
2.  **Median**: A sample with median error. This represents the "typical" performance you can expect.

## Feature Grouping

The 60+ clinical features are grouped into 4 categories to make the charts readable:

1.  **Time & Complexity**:
    - Includes statistical metrics (Mean, Std, Skewness, Kurtosis) and complexity measures (Hjorth Mobility/Complexity, LZ Complexity).
    - *Interpretation*: Reflects the signal's amplitude distribution and chaotic nature.

2.  **Spectral Power**:
    - Absolute power in standard frequency bands (Delta, Theta, Alpha, Beta, Gamma).
    - *Interpretation*: Reflects the energy intensity in different brain rhythms.

3.  **Relative Power**:
    - The percentage of total power contributed by each frequency band.
    - *Interpretation*: Crucial for identifying dominant rhythms (e.g., Alpha dominance in resting state).

4.  **Spectral Features**:
    - Advanced spectral metrics like Spectral Edge Frequency (SEF), Peak Frequency, and Band Ratios.
    - *Interpretation*: Describes the shape and shifts of the power spectrum.

## How to Read

- **Perfect Prediction**: The colored shapes perfectly align.
- **Systematic Bias**: If the Prediction shape is consistently smaller/larger than the Ground Truth, the model may have a scaling issue.
- **Pattern Matching**: Even if values aren't exact, check if the *shape* is similar. This means the model correctly identified which features are relatively high or low for that patient (e.g., correctly predicting "high Alpha, low Delta" even if the absolute values are slightly off).
