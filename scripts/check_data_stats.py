import pandas as pd
import numpy as np
import os

# Path to feature file
FEATURE_FILE = "/vePFS-0x0d/pretrain-clip/output_features/features_zscored.csv"
OUTPUT_REPORT = "data_quality_report.txt"

def check_stats():
    print(f"Loading {FEATURE_FILE}...")
    # Load more data for robust skew/kurtosis (50k rows)
    try:
        df = pd.read_csv(FEATURE_FILE, nrows=50000)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Identify feature columns (exclude meta)
    meta_cols = ['aug_type', 'segment_id', 'window_idx', 'start_time', 'subject_id', 'source_file']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"Analyzing {len(feature_cols)} features over {len(df)} samples...")
    
    # Calculate stats
    desc = df[feature_cols].describe().transpose()
    skew = df[feature_cols].skew()
    kurt = df[feature_cols].kurtosis()
    
    desc['skew'] = skew
    desc['kurt'] = kurt
    
    # Flags
    desc['FLAG_LOW_VAR'] = desc['std'] < 0.1
    desc['FLAG_BIAS'] = desc['mean'].abs() > 0.5
    desc['FLAG_SKEW'] = desc['skew'].abs() > 5.0
    
    # Sort by 'std' to see the most problematic ones first
    desc = desc.sort_values(by='std')
    
    # Write report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(f"Data Quality Report for {FEATURE_FILE}\n")
        f.write(f"Samples Analyzed: {len(df)}\n")
        f.write("="*80 + "\n\n")
        
        # Section 1: Problematic Features
        problems = desc[desc['FLAG_LOW_VAR'] | desc['FLAG_BIAS'] | desc['FLAG_SKEW']]
        f.write(f"Found {len(problems)} Potentially Problematic Features:\n")
        f.write(problems[['mean', 'std', 'min', 'max', 'skew', 'kurt']].to_string())
        f.write("\n\n" + "="*80 + "\n\n")
        
        # Section 2: Full Stats
        f.write("Full Statistics (Sorted by Std):\n")
        f.write(desc[['mean', 'std', 'min', 'max', 'skew', 'kurt']].to_string())
        
    print(f"Report saved to {OUTPUT_REPORT}")
    
    # Print summary to console
    print(f"\n--- Summary of Issues ---")
    print(f"Low Variance (Std < 0.1): {desc['FLAG_LOW_VAR'].sum()}")
    print(f"High Bias (Mean > 0.5): {desc['FLAG_BIAS'].sum()}")
    print(f"High Skewness (> 5.0): {desc['FLAG_SKEW'].sum()}")
    
    if len(problems) > 0:
        print("\nTop 10 Most Problematic Features (by Low Variance):")
        print(problems[['mean', 'std', 'skew']].head(10))

if __name__ == "__main__":
    check_stats()
