# ICML Abstract Draft

## Project Name Ideas
* **Neuro-KE:** A Universal Knowledge Engine for EEG Foundation Models
* **Knowledge-Guided EEG Foundation Models:** Unlocking Historical Signal Priors
* **NeuroPriors:** Label-Free Knowledge Integration for Robust EEG Representation Learning

## Suggested Title
**Neuro-KE: Scaling EEG Foundation Models via Label-Free Knowledge Integration**

## TL;DR
We introduce Neuro-KE, a label-free knowledge engine that enhances EEG foundation models by integrating diverse signal priors. It boosts generalization in data-scarce regimes across masked modeling, contrastive learning, and LLM paradigms.

## Abstract
Foundational Models (FMs) for Electroencephalography (EEG) have shown promise in learning generalized representations from large-scale datasets. However, current approaches primarily rely on raw signal reconstruction or scarce supervised labels, often neglecting the rich, domain-specific prior knowledge encapsulated in decades of signal processing research. In this work, we introduce Neuro-KE (Neuro-Knowledge Engine), a plug-and-play, label-free framework designed to seamlessly integrate comprehensive signal characteristics into the pre-training of EEG foundation models. Neuro-KE aggregates a vast array of morphometric, spectral, non-linear, and aperiodic features—spanning time-domain statistics, band powers, Hjorth parameters, and spectral entropy—effectively distilling historical domain expertise into a unified knowledge base. We demonstrate the versatility and effectiveness of Neuro-KE across three mainstream technical paradigms of EEG FMs: (1) Masked Modeling, where features guide the reconstruction process; (2) Contrastive Learning, where features serve as dense "semantic anchors" to align representations without human annotation; and (3) EEG Language Models, using feature priors to strengthen supervision signals in EEG-LLM adaptation. Extensive experiments show that Neuro-KE significantly enhances model generalization and robustness, particularly in label-scarce downstream tasks, offering a rigorous pathway to embed domain-invariant signal dynamics into modern deep learning architectures.

## Keywords
Foundation Models, EEG, Knowledge-Guided Learning, Self-Supervised Learning, Time-Series Analysis

---

## Detailed Breakdown (For Discussion)

### What is Neuro-KE? (The "Knowledge Engine")
We define Neuro-KE not just as a feature extractor, but as a **comprehensive signal descriptor manifold**. It moves beyond simple statistics to capture the full physical dynamics of the EEG signal.
*   **Components:** 
    *   **Time-Domain:** Amplitude dynamics (RMS, Peak-to-Peak, Mean Abs Amplitude), Statistical Moments (Skewness, Kurtosis), and Zero-Crossing Rates.
    *   **Frequency-Domain:** Absolute and Relative Band Powers (Delta, Theta, Alpha, Beta, Gamma, Low/High Gamma), and Power Ratios (Theta/Beta, Delta/Theta, Low/High).
    *   **Spectral & Aperiodic:** Spectral Entropy, Spectral Centroid, Peak Frequency, Individual Alpha Frequency (IAF), and Aperiodic Exponents (capturing the 1/f background activity).
    *   **Non-Linear & Complexity:** Hjorth Parameters (Activity, Mobility, Complexity).
    *   **Dynamics:** Captures both the **mean** and **temporal variability (std)** of all metrics across windows.
*   **Role:** It acts as a "teacher" or "prior" that forces the Foundation Model to pay attention to clinically and physically meaningful patterns, rather than just matching raw sample values.

### Validation Routes
1.  **Masked Modeling (PTFT):** *Current active run.* We use the features as auxiliary targets or conditioning inputs during the masked modeling phase, ensuring the latent space encodes these expert-defined properties.
2.  **Contrastive Learning:** Unlike traditional CLIP which relies on scarce text reports, we treat the Neuro-KE feature vector as a "semantic description" of the signal. The model learns to align the raw signal embedding with its corresponding "Knowledge Embedding," effectively performing self-supervised grounding.
3.  **EEG Language Models:** In Instruction Tuning (SFT), we inject these features to reinforce supervision signals, providing objective guidance for the LLM's reasoning and reducing hallucinations.

### Key Advantages
*   **Label-Free:** Requires zero human annotation; scales with any raw dataset.
*   **Universal Compatibility:** Can be added to Masked Autoencoders (MAE), Contrastive (CLIP), or Autoregressive (GPT) architectures.
*   **Data Efficiency:** Provides strong inductive biases that are crucial when training data is limited (common in EEG).
