import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Compare Feature Metrics")
    parser.add_argument("--csv1", type=str, required=True, help="Path to first metrics CSV (e.g. Mask 50%)")
    parser.add_argument("--name1", type=str, required=True, help="Name for first dataset")
    parser.add_argument("--csv2", type=str, required=True, help="Path to second metrics CSV (e.g. Zero Mask)")
    parser.add_argument("--name2", type=str, required=True, help="Name for second dataset")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    # Load Data
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)
    
    # Merge on Feature Name
    # Assume Feature Name is unique key
    if 'Feature Name' not in df1.columns or 'Feature Name' not in df2.columns:
        print("Error: CSVs must contain 'Feature Name' column")
        return

    # Add source column
    df1['Condition'] = args.name1
    df2['Condition'] = args.name2
    
    # Concatenate
    # We want a long format for seaborn: Feature Name | R2 | Condition
    cols = ['Feature Name', 'R2', 'PCC', 'Condition']
    df_long = pd.concat([df1[cols], df2[cols]], axis=0)
    
    # Clean Names
    df_long['Display Name'] = df_long['Feature Name'].str.replace('_', ' ').str.title()
    
    # Sort order: Use the order of the first dataset (usually sorted by R2)
    # Get order from df1
    df1_sorted = df1.sort_values('R2', ascending=False)
    order = df1_sorted['Feature Name'].str.replace('_', ' ').str.title().tolist()
    
    # --- Plot Comparison R2 ---
    plt.figure(figsize=(12, 18))
    sns.set_theme(style="whitegrid")
    
    # Grouped Bar Plot
    # hue='Condition' creates the side-by-side bars
    ax = sns.barplot(x="R2", y="Display Name", hue="Condition", data=df_long, 
                     order=order, palette="muted")
    
    plt.title("Feature Prediction R2 Score Comparison", fontsize=16)
    plt.xlabel("R2 Score", fontsize=14)
    plt.ylabel("")
    plt.legend(title="Condition", loc="lower right")
    plt.xlim(min(0, df_long['R2'].min() - 0.1), 1.0)
    plt.tight_layout()
    
    out_path = os.path.join(args.output_dir, "feature_r2_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved Comparison R2 plot to {out_path}")
    plt.close()
    
    # --- Plot Comparison PCC ---
    # Re-sort for PCC if desired? Or keep same order?
    # Usually better to keep same order for direct comparison, or sort by PCC of first.
    # Let's sort by PCC of first dataset for the PCC plot.
    df1_pcc_sorted = df1.sort_values('PCC', ascending=False)
    pcc_order = df1_pcc_sorted['Feature Name'].str.replace('_', ' ').str.title().tolist()

    plt.figure(figsize=(12, 18))
    sns.barplot(x="PCC", y="Display Name", hue="Condition", data=df_long, 
                order=pcc_order, palette="muted")
    
    plt.title("Feature Prediction PCC Comparison", fontsize=16)
    plt.xlabel("Pearson Correlation Coefficient (r)", fontsize=14)
    plt.ylabel("")
    plt.legend(title="Condition", loc="lower right")
    plt.xlim(min(0, df_long['PCC'].min() - 0.1), 1.0)
    plt.tight_layout()
    
    out_path = os.path.join(args.output_dir, "feature_pcc_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved Comparison PCC plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
