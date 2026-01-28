import pandas as pd
import os
import glob

# Define experiments and their paths
experiments = {
    "Recon (EEG)": "output/finetune_recon1",
    "Feat-Only (EEG)": "output/finetune_feat_only1",
    "Neuro-KE (EEG)": "output/finetune_neuro_ke1"
}

# Linear Probe Baseline (from previous results.md)
lp_results = {
    "Recon (EEG)": 70.02,
    "Neuro-KE (EEG)": 79.22,
    "Neuro-KE (Feat)": 78.28
}

def get_best_metric(log_path, metric='val_balanced_acc'):
    if not os.path.exists(log_path):
        return "N/A"
    
    try:
        df = pd.read_csv(log_path)
        if metric not in df.columns:
            # Try finding a similar column
            cols = [c for c in df.columns if 'balanced_acc' in c and 'val' in c]
            if cols:
                metric = cols[0]
            else:
                return "N/A"
        
        # Get max value
        best_val = df[metric].max()
        return best_val * 100 # Convert to percentage
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return "Error"

def generate_markdown():
    markdown = "# TUAB Full Fine-tuning Results\n\n"
    markdown += "## Experiment Overview\n"
    markdown += "- **Task**: Abnormal vs Normal Classification (TUAB)\n"
    markdown += "- **Method**: Full Fine-tuning (End-to-End)\n"
    markdown += "- **Batch Size**: 256\n"
    markdown += "- **GPUs**: 4 (DataParallel)\n\n"
    
    markdown += "## Results Table\n\n"
    markdown += "| Model | Pretraining | Linear Probe BAcc (%) | Full Fine-Tune BAcc (%) |\n"
    markdown += "|-------|-------------|-----------------------|-------------------------|\n"
    
    for name, path in experiments.items():
        log_file = os.path.join(path, "log.csv")
        ft_acc = get_best_metric(log_file)
        
        lp_acc = lp_results.get(name, "N/A")
        if isinstance(lp_acc, float):
            lp_acc = f"{lp_acc:.2f}"
            
        if isinstance(ft_acc, float):
            ft_acc_str = f"**{ft_acc:.2f}**"
        else:
            ft_acc_str = str(ft_acc)
            
        # Determine pretraining task description
        if "Recon" in name:
            pt_task = "Reconstruction Only"
        elif "Feat-Only" in name:
            pt_task = "Feature Pred Only"
        elif "Neuro-KE" in name:
            pt_task = "Recon + Feature Pred"
        else:
            pt_task = "-"
            
        markdown += f"| {name} | {pt_task} | {lp_acc} | {ft_acc_str} |\n"
        
    markdown += "\n## Analysis\n\n"
    markdown += "*(This section will be populated with analysis once results are available)*\n"
    
    return markdown

if __name__ == "__main__":
    report = generate_markdown()
    
    output_file = "experiments/tuab_full_ft/results_full_ft.md"
    with open(output_file, "w") as f:
        f.write(report)
        
    print(f"Report generated at {output_file}")
    print(report)
