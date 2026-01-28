import pandas as pd
import os

FEATURE_FILE = "/vePFS-0x0d/pretrain-clip/output_features/features_zscored.csv"
OUTPUT_FILE = "output/bad_features.txt"

def generate_blacklist():
    print(f"Loading {FEATURE_FILE}...")
    # Load 50k rows for robust stats
    df = pd.read_csv(FEATURE_FILE, nrows=50000)
    
    # Identify feature columns
    meta_cols = ['aug_type', 'segment_id', 'window_idx', 'start_time', 'subject_id', 'source_file']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Calculate stats
    std = df[feature_cols].std()
    skew = df[feature_cols].skew()
    
    # Define criteria
    # 1. Low Variance: Std < 0.1 (Too flat)
    # 2. Extreme Skewness: Abs(Skew) > 10.0 (Too long-tailed)
    bad_feats = []
    
    for feat in feature_cols:
        if std[feat] < 0.1:
            bad_feats.append(feat)
            print(f"Exclude {feat}: Low Std ({std[feat]:.4f})")
        elif abs(skew[feat]) > 10.0:
            bad_feats.append(feat)
            print(f"Exclude {feat}: High Skew ({skew[feat]:.4f})")
            
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        for feat in bad_feats:
            f.write(f"{feat}\n")
            
    print(f"\nTotal excluded features: {len(bad_feats)} / {len(feature_cols)}")
    print(f"Blacklist saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    generate_blacklist()
