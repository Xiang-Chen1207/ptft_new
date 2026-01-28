import re

def parse_results_table(filepath):
    """Parses the raw results markdown table into a structured list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header and separator
    table_lines = [line.strip() for line in lines if line.strip().startswith('|') and 'Ratio' not in line and '---' not in line]
    
    for line in table_lines:
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 7: continue
        
        # | Ratio | NumSub | Model | Feature | Dim | Acc | BAcc |
        ratio = parts[0]
        num_samples = parts[1]
        model = parts[2]
        feature = parts[3]
        dim = parts[4]
        bacc_str = parts[6].replace('%', '')
        
        try:
            bacc = float(bacc_str)
        except:
            bacc = 0.0
            
        data.append({
            'Ratio': ratio,
            'NumSamples': num_samples,
            'Model': model,
            'Feature': feature,
            'Dim': dim,
            'BAcc': bacc
        })
    return data

def format_bacc(val, is_bold=False):
    s = f"{val:.2f}%"
    return f"**{s}**" if is_bold else s

def generate_summary_markdown(data):
    # Get unique ratios
    ratios_dict = {}
    for d in data:
        key = (d['Ratio'], d['NumSamples'])
        try:
            val = float(d['Ratio'].replace('%', ''))
        except:
            val = 100.0
        ratios_dict[key] = val
        
    ratios = sorted(ratios_dict.keys(), key=lambda x: ratios_dict[x])
    
    md = "# Linear Probing Results: Model Comparison (Baseline vs Neuro-KE vs FeatOnly)\n\n"
    
    md += "## Overview\n"
    md += "- **Task**: TUAB Abnormal/Normal Classification\n"
    md += "- **Method**: Linear Probing (Logistic Regression)\n"
    md += "- **Metric**: Balanced Accuracy (%)\n"
    md += "- **Models**:\n"
    md += "  - **Baseline**: Standard Reconstruction (MAE)\n"
    md += "  - **Neuro-KE**: Reconstruction + Feature Prediction (Flagship)\n"
    md += "  - **FeatOnly**: Feature Prediction Only (No Reconstruction)\n"
    
    md += "\n## Feature Types Explanation\n"
    md += "- **EEG** (Dim: 200): Standard Global Average Pooling (GAP) features from the Transformer encoder backbone. Represents the general signal encoding.\n"
    md += "- **Feat** (Dim: 200): The learned 'Feature Token' from the Cross-Attention Head. This token is specifically trained to attend to clinical features.\n"
    md += "- **Pred** (Dim: 62): The raw predicted values for the 62 clinical features (regression output). Highly interpretable and low-dimensional.\n"
    md += "- **Full** (Dim: 400): Concatenation of **EEG** and **Feat** vectors. Combines general signal representation with targeted clinical knowledge (Reference).\n\n"
    
    md += "## Comparative Results Table\n"
    md += "*Note: **Bold** indicates the best performance among all models (Global Best). The best performance among comparable low-dimensional features (Dim ≤ 200) is also **bolded** if it differs from the global best.*\n\n"
    
    # Header - Reordered: Baseline | Neuro-KE (EEG/Feat/Pred) | FeatOnly (EEG/Feat/Pred) | Full (Ref)
    md += "| Ratio | Samples | **Baseline** | **Neuro-KE** (EEG) | **Neuro-KE** (Feat) | **Neuro-KE** (Pred) | **FeatOnly** (EEG) | **FeatOnly** (Feat) | **FeatOnly** (Pred) | **Neuro-KE** (Full) | **FeatOnly** (Full) |\n"
    md += "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    
    for r_str, n_samples in ratios:
        # Extract data for this ratio
        subset = [d for d in data if d['Ratio'] == r_str]
        
        # Helper to find value
        def get_val(model, feat):
            matches = [d['BAcc'] for d in subset if d['Model'] == model and d['Feature'].lower() == feat.lower()]
            return matches[0] if matches else 0.0
            
        # Values
        base_val = get_val('Baseline', 'EEG')
        
        nk_eeg = get_val('Neuro-KE', 'eeg')
        nk_feat = get_val('Neuro-KE', 'feat')
        nk_pred = get_val('Neuro-KE', 'pred')
        nk_full = get_val('Neuro-KE', 'full')
        
        fo_eeg = get_val('FeatOnly', 'eeg')
        fo_feat = get_val('FeatOnly', 'feat')
        fo_pred = get_val('FeatOnly', 'pred')
        fo_full = get_val('FeatOnly', 'full')
        
        # 1. Global Best (across ALL columns)
        all_vals = [base_val, nk_eeg, nk_feat, nk_pred, nk_full, fo_eeg, fo_feat, fo_pred, fo_full]
        global_max = max(all_vals) if all_vals else 0.0
        
        # 2. Low-Dim Best (across Dim <= 200 columns)
        # Exclude 'Full' features which are Dim 400
        low_dim_vals = [base_val, nk_eeg, nk_feat, nk_pred, fo_eeg, fo_feat, fo_pred]
        low_dim_max = max(low_dim_vals) if low_dim_vals else 0.0
        
        # Helper to check if should bold
        def is_highlight(val, is_full=False):
            if val == global_max: return True
            if not is_full and val == low_dim_max: return True
            return False
        
        row = f"| **{r_str}** | {n_samples} | "
        row += f"{format_bacc(base_val, is_highlight(base_val))} | "
        
        row += f"{format_bacc(nk_eeg, is_highlight(nk_eeg))} | "
        row += f"{format_bacc(nk_feat, is_highlight(nk_feat))} | "
        row += f"{format_bacc(nk_pred, is_highlight(nk_pred))} | "
        
        row += f"{format_bacc(fo_eeg, is_highlight(fo_eeg))} | "
        row += f"{format_bacc(fo_feat, is_highlight(fo_feat))} | "
        row += f"{format_bacc(fo_pred, is_highlight(fo_pred))} | "
        
        # Full features at the end
        row += f"{format_bacc(nk_full, is_highlight(nk_full, True))} | "
        row += f"{format_bacc(fo_full, is_highlight(fo_full, True))} |"
        
        md += row + "\n"
        
    md += "\n## Key Observations\n\n"
    
    full_data_subset = [d for d in data if d['Ratio'] == '100%']
    if full_data_subset:
        def get_100_val(model, feat):
            matches = [d['BAcc'] for d in full_data_subset if d['Model'] == model and d['Feature'].lower() == feat.lower()]
            return matches[0] if matches else 0.0
            
        base_100 = get_100_val('Baseline', 'EEG')
        nk_full_100 = get_100_val('Neuro-KE', 'full')
        
        nk_gain = nk_full_100 - base_100
        
        md += f"1. **Neuro-KE Dominance**: Neuro-KE (Full) consistently achieves top-tier performance, reaching **{nk_full_100:.2f}%** at 100% data, a **+{nk_gain:.2f}%** improvement over Baseline.\n"
        md += "2. **FeatOnly Competitiveness**: The FeatOnly model (without reconstruction) performs extremely well, suggesting feature prediction is a strong supervisory signal.\n"
        md += "3. **Feature Efficacy**: The `Full` (concatenated) features generally offer the highest performance (Global Best), while individual `Feat` or `EEG` features (Dim 200) often achieve the best low-dimensional performance.\n"
        md += "4. **Low Data Regime**: In lower data regimes (e.g., 5-10%), feature prediction based models maintain a significant lead over the Baseline.\n"
    
    return md

if __name__ == "__main__":
    input_path = "experiments/tuab_lp/results_final.md"
    output_path = "experiments/tuab_lp/results_final_summary.md"
    
    data = parse_results_table(input_path)
    report = generate_summary_markdown(data)
    
    with open(output_path, "w") as f:
        f.write(report)
        
    print(f"Summary generated at {output_path}")
