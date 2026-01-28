
import os
import glob
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def check_file(h5_path):
    short_count = 0
    total_count = 0
    try:
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                for seg_key in segment_keys:
                    seg_group = trial_group[seg_key]
                    
                    if 'eeg' in seg_group:
                        dset = seg_group['eeg']
                        # Check shape. Usually (Channels, Time) or (Time, Channels)
                        # Based on previous checks, it seems to be (21, N) or similar, but let's check shape[1] if shape[0]==21
                        shape = dset.shape
                        
                        # Assuming (Channels, Time) based on previous logs: "Data Shape: (21, 7377)"
                        # If shape[0] is 21, then shape[1] is time.
                        # But let's be robust.
                        
                        length = 0
                        if shape[0] == 21:
                            length = shape[1]
                        elif len(shape) > 1 and shape[1] == 21:
                            length = shape[0]
                        else:
                            # Fallback, assume last dim is time if not 21
                            length = shape[-1]
                            
                        total_count += 1
                        if length < 2000: # 10s * 200Hz
                            short_count += 1
    except Exception as e:
        # print(f"Error reading {h5_path}: {e}")
        pass
        
    return total_count, short_count

def main():
    dataset_dir = "/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure"
    print("Scanning files...")
    all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True))
    print(f"Found {len(all_h5_files)} files.")
    
    total_segments = 0
    short_segments = 0
    
    # Use parallel processing to speed up
    max_workers = min(32, os.cpu_count() or 1)
    
    print(f"Processing with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(check_file, all_h5_files), total=len(all_h5_files)))
        
    for t, s in results:
        total_segments += t
        short_segments += s
        
    print("\n=== Statistics ===")
    print(f"Total Segments: {total_segments}")
    print(f"Segments < 10s (2000 samples): {short_segments}")
    if total_segments > 0:
        print(f"Percentage: {short_segments/total_segments*100:.2f}%")

if __name__ == '__main__':
    main()
