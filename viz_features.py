import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize Feature Metrics")
    parser.add_argument("--csv", type=str, required=True, help="Path to metrics CSV")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for plots")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    args = parser.parse_args()

    # Load Data
    df = pd.read_csv(args.csv)
    
    # Sort by R2 for R2 plot
    df_r2 = df.sort_values('R2', ascending=False)
    
    # Sort by PCC for PCC plot
    df_pcc = df.sort_values('PCC', ascending=False)
    
    # Clean Feature Names for better display (optional)
    # Remove underscores, title case
    df_r2['Display Name'] = df_r2['Feature Name'].str.replace('_', ' ').str.title()
    df_pcc['Display Name'] = df_pcc['Feature Name'].str.replace('_', ' ').str.title()
    
    # --- Plot R2 ---
    plt.figure(figsize=(12, 14))
    sns.set_theme(style="whitegrid")
    
    # Use a diverging palette or sequential? R2 is max 1. 
    # Let's use 'viridis' or 'coolwarm'
    ax = sns.barplot(x="R2", y="Display Name", data=df_r2, palette="viridis")
    
    # Add values to bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        # Handle negative values for positioning
        if width < 0:
            x_pos = width - 0.05
            ha = 'right'
        else:
            x_pos = width + 0.01
            ha = 'left'
            
        ax.text(x_pos, p.get_y() + p.get_height()/2, f'{width:.3f}', 
                va='center', ha=ha, fontsize=9)

    plt.title(f"Feature Prediction R2 Score (Zero Mask)", fontsize=16)
    plt.xlabel("R2 Score", fontsize=14)
    plt.ylabel("")
    plt.xlim(min(0, df_r2['R2'].min() - 0.2), 1.15) # Expand limits for labels
    plt.tight_layout()
    
    fname = f"feature_r2_scores{args.suffix}.png"
    r2_path = os.path.join(args.output_dir, fname)
    plt.savefig(r2_path, dpi=300)
    print(f"Saved R2 plot to {r2_path}")
    plt.close()
    
    # --- Plot PCC ---
    plt.figure(figsize=(12, 14))
    ax = sns.barplot(x="PCC", y="Display Name", data=df_pcc, palette="magma")
    
    # Add values to bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if width < 0:
            x_pos = width - 0.05
            ha = 'right'
        else:
            x_pos = width + 0.01
            ha = 'left'
            
        ax.text(x_pos, p.get_y() + p.get_height()/2, f'{width:.3f}', 
                va='center', ha=ha, fontsize=9)

    plt.title(f"Feature Prediction Pearson Correlation (Zero Mask)", fontsize=16)
    plt.xlabel("Pearson Correlation Coefficient (r)", fontsize=14)
    plt.ylabel("")
    plt.xlim(min(0, df_pcc['PCC'].min() - 0.2), 1.15)
    plt.tight_layout()
    
    fname = f"feature_pcc_scores{args.suffix}.png"
    pcc_path = os.path.join(args.output_dir, fname)
    plt.savefig(pcc_path, dpi=300)
    print(f"Saved PCC plot to {pcc_path}")
    plt.close()

def plot_scatter(csv_path, output_dir):
    """
    Generate scatter plots for each feature (Predicted vs Target).
    Note: Since we don't save raw predictions in CSV, we can't do this from summary CSV.
    We need to modify eval_features.py to save raw data or plot on the fly.
    
    WAIT: The user asked to "generate scatter plot for each feature r".
    If they mean scatter plot of R values? No, "scatter plot of each feature r" implies Pred vs True.
    
    Since we only have summary stats in CSV, we cannot generate scatter plots of Pred vs True.
    We need to modify eval_features.py to either:
    1. Save all predictions (Huge file: 28k samples * 66 features * 2 floats) ~ 30MB. Feasible.
    2. Or plot inside eval_features.py.
    
    Let's assume we need to modify eval_features.py to save a separate file with raw data for plotting.
    """
    pass

if __name__ == "__main__":
    main()
