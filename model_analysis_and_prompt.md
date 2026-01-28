**Subject**: Professional Neural Network Architecture Diagram for ICML Paper

**Role**: You are an expert scientific illustrator creating a high-quality, vector-style architecture diagram for a top-tier machine learning conference (ICML).

**Task**: Generate a comprehensive schematic of the "PTFT EEG Foundation Model" with a focus on its Criss-Cross Attention mechanism and Multi-Task learning flow.

**Visual Style**:
*   **Clean & Academic**: Use a flat, modern design with thin, precise lines. White background.
*   **Color Palette**: Use distinct but harmonious colors for different data flows.
    *   *Blue/Teal*: Time-domain processing.
    *   *Purple/Violet*: Frequency-domain processing.
    *   *Orange/Gold*: Attention mechanisms (Spatial & Temporal).
    *   *Green*: Output heads and objectives.
*   **Layout**: Horizontal flow from Left (Input) to Right (Outputs).

**Diagram Components & Layout**:

1.  **Input Block (Left)**:
    *   Show a stack of EEG signals labeled **"Raw EEG (Multi-channel)"**.
    *   Show a step of **"Patching"** dividing signals into small blocks.
    *   **Embedding Fusion Module**: Show three parallel arrows merging into one:
        *   Arrow 1: **"Time Conv"** (CNN icon).
        *   Arrow 2: **"FFT + Linear"** (Frequency icon).
        *   Arrow 3: **"Positional Enc"**.
        *   Result: **"Patch Embeddings"**.

2.  **Backbone: Criss-Cross Transformer (Center - Large Box)**:
    *   Show a stack of **"N x Transformer Blocks"**.
    *   **Zoom-in Detail on One Block**:
        *   Show the input feature splitting into two parallel branches:
        *   **Branch A (Top)**: Labeled **"Spatial Attention (Channel-dim)"**. Visual: Attention map connecting different rows (channels).
        *   **Branch B (Bottom)**: Labeled **"Temporal Attention (Patch-dim)"**. Visual: Attention map connecting different columns (time steps).
        *   Show the two branches merging (Concatenation) -> **"Feed Forward"** -> Output.

3.  **Multi-Task Outputs (Right)**:
    *   Split the backbone output into two paths:
    *   **Path 1 (Top) - Reconstruction**:
        *   Arrow labeled "Latent Features".
        *   Block: **"Linear Projection"**.
        *   Output: **"Reconstructed EEG"** (Visual: Blurry EEG traces becoming clear).
        *   Loss: **"MSE Loss (Reconstruction)"**.
    *   **Path 2 (Bottom) - Feature Prediction**:
        *   Block: **"Cross-Attention Aggregator"**.
        *   Input to Block: **"Learnable Query"** (Small token icon) interacting with Backbone Features.
        *   Block: **"MLP Head"**.
        *   Output: **"Predicted Features"** (Visual: Bar chart or vector).
        *   Loss: **"MSE Loss (Neuro-KE)"**.

**Annotations**:
*   Label dimensions where appropriate: $B \times C \times N \times D$.
*   Use solid arrows for data flow, dashed arrows for Loss calculation.
*   Add a legend for "Spatial Path" vs "Temporal Path".

**Output Format**: High-resolution SVG or PNG (300 DPI). Aspect ratio 16:9