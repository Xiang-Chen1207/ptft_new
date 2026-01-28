
import os
import glob
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

LABELS = [f"Label {i}" for i in range(13)]

def check_file_stats(h5_path):
    # Returns {label_idx: {'total': count, 'short': count}}
    stats = {}
    try:
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                for seg_key in segment_keys:
                    seg_group = trial_group[seg_key]
                    
                    if 'eeg' in seg_group:
                        dset = seg_group['eeg']
                        
                        # Get Length
                        shape = dset.shape
                        length = 0
                        if shape[0] == 21:
                            length = shape[1]
                        elif len(shape) > 1 and shape[1] == 21:
                            length = shape[0]
                        else:
                            length = shape[-1]
                        
                        # Get Label
                        label = -1
                        if 'label' in dset.attrs:
                            label_vec = dset.attrs['label']
                            if hasattr(label_vec, '__len__'):
                                label = np.argmax(label_vec)
                            else:
                                label = int(label_vec)
                        
                        if label != -1 and label < 13:
                            if label not in stats:
                                stats[label] = {'total': 0, 'short': 0}
                            
                            stats[label]['total'] += 1
                            if length < 2000: # 10s * 200Hz
                                stats[label]['short'] += 1
    except Exception:
        pass
    return stats

def merge_stats(results):
    final_stats = {i: {'total': 0, 'short': 0} for i in range(13)}
    for res in results:
        for label, counts in res.items():
            final_stats[label]['total'] += counts['total']
            final_stats[label]['short'] += counts['short']
    return final_stats

def main():
    dataset_dir = "/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure"
    
    print("Scanning files...")
    all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True))
    
    print(f"Processing {len(all_h5_files)} files...")
    max_workers = min(32, os.cpu_count() or 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(check_file_stats, all_h5_files), total=len(all_h5_files)))
        
    final_stats = merge_stats(results)
    
    print("\n=== Short Segment Statistics (< 10s) per Label ===")
    print(f"{'Label ID':<10} | {'Total':<10} | {'Short':<10} | {'Ratio (%)':<10}")
    print("-" * 46)
    
    for i in range(13):
        total = final_stats[i]['total']
        short = final_stats[i]['short']
        ratio = (short / total * 100) if total > 0 else 0.0
        
        print(f"{i:<10} | {total:<10} | {short:<10} | {ratio:<10.2f}")

if __name__ == '__main__':
    main()
