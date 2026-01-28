
import os
import h5py
import pandas as pd
from tqdm import tqdm
from glob import glob

def check_segment_consistency(h5_dir, csv_path):
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Group CSV by source_file to get segment counts
    # We assume 'source_file' contains the full path or at least the filename
    # Let's extract basename to be safe
    df['basename'] = df['source_file'].apply(os.path.basename)
    
    # Count segments per file in CSV
    # Filter for window_idx == 0 if multiple windows per segment exist (as per previous code logic)
    if 'window_idx' in df.columns:
        csv_counts = df[df['window_idx'] == 0].groupby('basename').size()
    else:
        csv_counts = df.groupby('basename').size()
        
    print(f"CSV contains {len(csv_counts)} unique files.")
    
    # Get list of H5 files
    # We need to match the files in CSV. Let's look for them in the h5_dir.
    # Or better, iterate over the files present in the CSV to check them.
    
    mismatches = []
    missing_files = []
    checked_count = 0
    
    # Limit check to a subset if too many, or check all if feasible
    # Let's check all unique files in CSV that exist in h5_dir
    
    print("Checking H5 files...")
    
    # We need to find where the H5 files are actually located.
    
    # The CSV 'source_file' column might have absolute paths.
    # Let's try to use those first.
    
    unique_files = df['source_file'].unique()
    
    # Check logic moved to check_file function below
    
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def check_file(file_path):
        # Check if file exists at path
        target_path = file_path
        if not os.path.exists(target_path):
            basename = os.path.basename(file_path)
            potential_path = os.path.join(h5_dir, basename)
            if os.path.exists(potential_path):
                target_path = potential_path
            else:
                return {'status': 'missing', 'file': basename}

        try:
            h5_seg_count = 0
            with h5py.File(target_path, 'r') as f:
                # Count segments logic from TUEGDataset
                for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                    trial_group = f[trial_key]
                    segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                    if segment_keys:
                        h5_seg_count += len(segment_keys)
                    else:
                        h5_seg_count += 1
            
            basename = os.path.basename(target_path)
            csv_count = csv_counts.get(basename, 0)
            
            if h5_seg_count != csv_count:
                return {
                    'status': 'mismatch',
                    'file': basename,
                    'h5_count': h5_seg_count,
                    'csv_count': csv_count
                }
            
            return {'status': 'ok'}
            
        except Exception as e:
            return {'status': 'error', 'file': target_path, 'error': str(e)}

    # Use ThreadPoolExecutor for concurrent file checking
    print(f"Starting concurrent check with {min(32, os.cpu_count() or 1)} workers...")
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        futures = [executor.submit(check_file, f) for f in unique_files]
        
        for future in tqdm(as_completed(futures), total=len(unique_files)):
            result = future.result()
            if result['status'] == 'ok':
                checked_count += 1
            elif result['status'] == 'missing':
                missing_files.append(result['file'])
            elif result['status'] == 'mismatch':
                mismatches.append(result)
                checked_count += 1
            elif result['status'] == 'error':
                print(f"Error checking {result['file']}: {result['error']}")

    print(f"\n--- Consistency Check Report ---")
    print(f"Checked Files: {checked_count}")
    print(f"Missing Files (in CSV but not found on disk): {len(missing_files)}")
    print(f"Mismatched Files: {len(mismatches)}")
    
    if mismatches:
        print("\nTop 10 Mismatches:")
        for m in mismatches[:10]:
            print(f"File: {m['file']} | H5: {m['h5_count']} vs CSV: {m['csv_count']}")
    
    if missing_files:
        print(f"\nExample Missing Files: {missing_files[:5]}")

if __name__ == "__main__":
    # Config
    H5_DIR = "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset" # From previous context
    CSV_PATH = "/vePFS-0x0d/pretrain-clip/feature_analysis/features_final_zscore.csv"
    
    check_segment_consistency(H5_DIR, CSV_PATH)
