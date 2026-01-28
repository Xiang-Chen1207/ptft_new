
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_data(path, label):
    df = pd.read_csv(path)
    df['Model'] = label
    return df

def categorize_features(feature_name):
    """
    Categorize features based on the abstract description.
    Priority: Dynamics (std) -> Specific Categories (mean/others)
    """
    name = feature_name.lower()
    
    # 5. Dynamics (Standard Deviation)
    if 'std' in name:
        return 'Dynamics'

    # 1. Time-Domain
    if any(x in name for x in ['rms', 'peak_to_peak', 'mean_abs_amplitude', 'skewness', 'kurtosis', 'zero_crossing']):
        return 'Time-Domain'
    
    # 2. Frequency-Domain (Band Powers & Ratios)
    if 'ratio' in name:
        return 'Frequency-Domain'
    if any(x in name for x in ['delta', 'theta', 'alpha', 'beta', 'gamma']) and 'ratio' not in name and 'individual' not in name:
        return 'Frequency-Domain'
    if 'total_power' in name:
         return 'Frequency-Domain'

    # 3. Spectral & Aperiodic
    if any(x in name for x in ['spectral', 'peak_frequency', 'individual_alpha', 'aperiodic', 'centroid']):
        return 'Spectral & Aperiodic'
    
    # 4. Non-Linear & Complexity
    if 'hjorth' in name:
        return 'Non-Linear & Complexity'
        
    # Default fallback
    return 'Other'

def plot_horizontal_bars(df, metric, title, output_path, color='skyblue'):
    """
    Plot horizontal bar chart for a single dataframe sorted by metric.
    """
    df_sorted = df.sort_values(metric, ascending=True) # Ascending for horiz bar (top is highest)
    
    plt.figure(figsize=(10, 15))
    bars = plt.barh(df_sorted['Feature Name'], df_sorted[metric], color=color)
    
    plt.xlabel(metric)
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add values
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', 
                 ha='left', va='center', fontsize=8)
                 
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_comparison_horizontal(df1, df2, metric, title, output_path):
    """
    Plot comparison horizontal bars.
    """
    # Merge dataframes
    merged = pd.merge(df1[['Feature Name', metric]], df2[['Feature Name', metric]], 
                      on='Feature Name', suffixes=('_FeatOnly', '_Full'))
    
    # Sort by FeatOnly for consistency
    merged = merged.sort_values(f'{metric}_FeatOnly', ascending=True)
    
    y_pos = np.arange(len(merged))
    height = 0.35
    
    plt.figure(figsize=(12, 18))
    
    # Plot FeatOnly
    bars1 = plt.barh(y_pos + height/2, merged[f'{metric}_FeatOnly'], height, label='Feature Only', color='skyblue')
    # Plot Full
    bars2 = plt.barh(y_pos - height/2, merged[f'{metric}_Full'], height, label='Full Model', color='orange')
    
    plt.yticks(y_pos, merged['Feature Name'])
    plt.xlabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add values (optional, might be too crowded)
    # Let's add values only if requested or careful
    # User asked for values.
    
    def add_labels(bars):
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}', 
                     ha='left', va='center', fontsize=6)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_grouped_vertical(df1, df2, metric, title, output_path):
    """
    Plot grouped vertical bars based on categories.
    """
    # Add categories
    df1['Category'] = df1['Feature Name'].apply(categorize_features)
    df2['Category'] = df2['Feature Name'].apply(categorize_features)
    
    # Merge for easier plotting
    combined = pd.concat([df1[['Feature Name', metric, 'Category', 'Model']], 
                          df2[['Feature Name', metric, 'Category', 'Model']]])
    
    # Get unique categories
    categories = ['Time-Domain', 'Frequency-Domain', 'Spectral & Aperiodic', 'Non-Linear & Complexity', 'Dynamics']
    
    # Create subplots for each category
    # 5 Categories -> 1x5
    fig, axes = plt.subplots(1, 5, figsize=(30, 8), sharey=True)
    
    colors = {'Feature Only': 'skyblue', 'Full Model': 'orange'}
    
    for i, cat in enumerate(categories):
        ax = axes[i]
        subset = combined[combined['Category'] == cat]
        
        if subset.empty:
            continue
            
        # Pivot for plotting: Feature Name x Model
        # We need to ensure consistent order
        features = subset['Feature Name'].unique()
        # Sort features by metric mean to look nice
        feat_mean = subset.groupby('Feature Name')[metric].mean().sort_values(ascending=False)
        order = feat_mean.index.tolist()
        
        sns.barplot(data=subset, x='Feature Name', y=metric, hue='Model', 
                    palette=colors, ax=ax, order=order)
        
        ax.set_title(cat)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel('')
        if i == 0:
            ax.set_ylabel(metric)
        else:
            ax.set_ylabel('')
            
        # Add values on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8, rotation=90)
            
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    feat_only_path = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_feat_only.csv"
    full_path = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_full.csv"
    output_dir = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/comparison_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df_feat = load_data(feat_only_path, "Feature Only")
    df_full = load_data(full_path, "Full Model")
    
    # 1. Individual Horizontal Bars (R2 & PCC)
    print("Generating individual plots...")
    plot_horizontal_bars(df_feat, 'R2', 'Feature Only - R2 Score', os.path.join(output_dir, 'feat_only_R2_bar.png'))
    plot_horizontal_bars(df_feat, 'PCC', 'Feature Only - PCC Score', os.path.join(output_dir, 'feat_only_PCC_bar.png'))
    plot_horizontal_bars(df_full, 'R2', 'Full Model - R2 Score', os.path.join(output_dir, 'full_R2_bar.png'))
    plot_horizontal_bars(df_full, 'PCC', 'Full Model - PCC Score', os.path.join(output_dir, 'full_PCC_bar.png'))
    
    # 2. Comparison Horizontal Bars
    print("Generating comparison plots...")
    plot_comparison_horizontal(df_feat, df_full, 'R2', 'R2 Score Comparison', os.path.join(output_dir, 'comparison_R2_bar.png'))
    plot_comparison_horizontal(df_feat, df_full, 'PCC', 'PCC Score Comparison', os.path.join(output_dir, 'comparison_PCC_bar.png'))
    
    # 3. Grouped Vertical Bars
    print("Generating grouped plots...")
    plot_grouped_vertical(df_feat, df_full, 'R2', 'R2 Score by Category', os.path.join(output_dir, 'grouped_R2_bar.png'))
    plot_grouped_vertical(df_feat, df_full, 'PCC', 'PCC Score by Category', os.path.join(output_dir, 'grouped_PCC_bar.png'))
    
    print(f"Done. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
