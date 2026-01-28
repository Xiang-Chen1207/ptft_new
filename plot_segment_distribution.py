
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import seaborn as sns

# Label mapping for TUSZ (Assuming standard 13 classes if available, otherwise just indices)
# Usually: 0: 'GNSZ', 1: 'FNSZ', 2: 'SPSZ', 3: 'CPSZ', 4: 'ABSZ', 5: 'TNSZ', 6: 'SNSZ', 7: 'TCSZ', 8: 'ATSZ', 9: 'MYSZ', 10: 'NESZ', 11: 'INsz', 12: 'FNSZ' (Duplicate?) or Background
# Let's use indices 0-12 if exact map is unknown.

LABELS = [f"Label {i}" for i in range(13)]

def collect_stats(h5_path):
    stats = {i: [] for i in range(13)}
    try:
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                for seg_key in segment_keys:
                    seg_group = trial_group[seg_key]
                    
                    label = -1
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
                        if 'label' in dset.attrs:
                            label_vec = dset.attrs['label']
                            if hasattr(label_vec, '__len__'):
                                label = np.argmax(label_vec)
                            else:
                                label = int(label_vec)
                        
                        if label != -1 and label < 13:
                            stats[label].append(length / 200.0) # Convert to seconds
    except Exception:
        pass
    return stats

def merge_stats(results):
    final_stats = {i: [] for i in range(13)}
    for res in results:
        for k, v in res.items():
            final_stats[k].extend(v)
    return final_stats

def plot_distributions(stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Use seaborn style
    sns.set_theme(style="whitegrid")
    
    for label_idx, lengths in stats.items():
        if not lengths:
            print(f"No data for Label {label_idx}")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Kernel Density Estimate plot
        sns.histplot(lengths, kde=True, stat="probability", bins=50)
        
        plt.title(f"Segment Length Distribution - {LABELS[label_idx]} (N={len(lengths)})")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Probability")
        plt.xlim(0, max(lengths) + 10) # Set x limit reasonable
        
        # Add vertical line for 10s and 60s
        plt.axvline(x=10, color='r', linestyle='--', label='10s Cutoff')
        plt.axvline(x=60, color='g', linestyle='--', label='60s Cutoff')
        plt.legend()
        
        output_path = os.path.join(output_dir, f"dist_label_{label_idx}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot for Label {label_idx} to {output_path}")

def main():
    dataset_dir = "/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure"
    output_dir = "tusz_plots"
    
    print("Scanning files...")
    all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True))
    
    print(f"Processing {len(all_h5_files)} files...")
    max_workers = min(32, os.cpu_count() or 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(collect_stats, all_h5_files), total=len(all_h5_files)))
        
    final_stats = merge_stats(results)
    
    print("Generating plots...")
    plot_distributions(final_stats, output_dir)
    print("Done.")

if __name__ == '__main__':
    main()
