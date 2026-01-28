
import os
import re

def parse_results(file_path):
    data = {}
    ratios = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Find table start
    start_idx = -1
    for i, line in enumerate(lines):
        if '| Ratio' in line and '| NumSub' in line:
            start_idx = i + 2 # Skip header and separator
            break
            
    if start_idx == -1:
        return None, None

    for line in lines[start_idx:]:
        line = line.strip()
        if not line.startswith('|'):
            continue
            
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 7:
            continue
            
        ratio = parts[0]
        num_sub = parts[1]
        model = parts[2]
        feature = parts[3]
        # dim = parts[4]
        # acc = parts[5]
        bacc = parts[6]
        
        key = (ratio, num_sub)
        if key not in ratios:
            ratios.append(key)
            
        if key not in data:
            data[key] = {}
            
        if model not in data[key]:
            data[key][model] = {}
            
        data[key][model][feature] = bacc

    return data, ratios

def generate_markdown(data, ratios, output_path):
    # Define columns
    # Based on what we saw in results_final.md
    columns = [
        ('Baseline', 'EEG'),
        ('Neuro-KE', 'eeg'),
        ('Neuro-KE', 'feat'),
        ('Neuro-KE', 'pred'),
        ('Neuro-KE', 'full'),
        ('FeatOnly', 'eeg'),
        ('FeatOnly', 'feat'),
        ('FeatOnly', 'pred'),
        ('FeatOnly', 'full'),
    ]
    
    headers = [
        "**Baseline**",
        "**Neuro-KE** (EEG)",
        "**Neuro-KE** (Feat)",
        "**Neuro-KE** (Pred)",
        "**Neuro-KE** (Full)",
        "**FeatOnly** (EEG)",
        "**FeatOnly** (Feat)",
        "**FeatOnly** (Pred)",
        "**FeatOnly** (Full)"
    ]

    with open(output_path, 'w') as f:
        f.write("# Linear Probing Results: Model Comparison (Baseline vs Neuro-KE vs FeatOnly - 60s)\n\n")
        
        f.write("## Overview\n")
        f.write("- **Task**: TUAB Abnormal/Normal Classification\n")
        f.write("- **Method**: Linear Probing (Logistic Regression)\n")
        f.write("- **Metric**: Balanced Accuracy (%)\n")
        f.write("- **Models**:\n")
        f.write("  - **Baseline**: Standard Reconstruction (MAE)\n")
        f.write("  - **Neuro-KE**: Reconstruction + Feature Prediction (Flagship)\n")
        f.write("  - **FeatOnly**: Feature Prediction Only (No Reconstruction)\n\n")
        
        f.write("## Feature Types Explanation\n")
        f.write("- **EEG**: Standard Global Average Pooling (GAP) features from the Transformer encoder backbone.\n")
        f.write("- **Feat**: The learned 'Feature Token' from the Cross-Attention Head.\n")
        f.write("- **Pred**: The raw predicted values for the clinical features.\n")
        f.write("- **Full**: Concatenation of **EEG** and **Feat** vectors.\n\n")
        
        f.write("## Comparative Results Table\n")
        f.write("*Note: **Bold** indicates the best performance in each row.*\n\n")
        
        # Table Header
        header_line = "| Ratio | Samples | " + " | ".join(headers) + " |"
        sep_line = "|:---:|:---:| " + " | ".join([":---:"] * len(headers)) + " |"
        
        f.write(header_line + "\n")
        f.write(sep_line + "\n")
        
        for ratio, num_sub in ratios:
            row_data = []
            row_values = []
            
            # First pass to get values and find max
            for model, feat in columns:
                val_str = data.get((ratio, num_sub), {}).get(model, {}).get(feat, "-")
                val_float = -1.0
                if val_str != "-":
                    try:
                        val_float = float(val_str.replace('%', ''))
                    except:
                        pass
                row_values.append(val_float)
            
            max_val = max(row_values) if row_values else -1
            
            # Second pass to format
            for i, (model, feat) in enumerate(columns):
                val_str = data.get((ratio, num_sub), {}).get(model, {}).get(feat, "-")
                val_float = row_values[i]
                
                if val_float == max_val and max_val > 0:
                    row_data.append(f"**{val_str}**")
                else:
                    row_data.append(val_str)
            
            line = f"| **{ratio}** | {num_sub} | " + " | ".join(row_data) + " |"
            f.write(line + "\n")
            
        f.write("\n")
        f.write("## Key Observations\n")
        f.write("*(Automatically generated placeholder - please update with analysis)*\n")

if __name__ == "__main__":
    input_file = "/vePFS-0x0d/home/cx/ptft/experiments/tuev_lp/results_final.md"
    output_file = "/vePFS-0x0d/home/cx/ptft/experiments/tuev_lp/results_final_summary.md"
    
    data, ratios = parse_results(input_file)
    if data:
        generate_markdown(data, ratios, output_file)
        print(f"Summary generated at {output_file}")
    else:
        print("Failed to parse results.")
