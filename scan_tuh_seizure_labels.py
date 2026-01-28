
import h5py
import numpy as np
import os
from tqdm import tqdm

def scan_labels(dataset_dir, num_files=20):
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')][:num_files]
    labels = set()
    segment_lengths = []
    
    for filename in tqdm(files):
        path = os.path.join(dataset_dir, filename)
        try:
            with h5py.File(path, 'r') as f:
                for trial_key in f.keys():
                    if not trial_key.startswith('trial'): continue
                    trial = f[trial_key]
                    
                    # Try to find labels
                    # In Seizure dataset, labels might be segment level or trial level
                    # The previous output showed task_name='seizure_detection'
                    # Let's check segment attributes
                    
                    for seg_key in trial.keys():
                        if not seg_key.startswith('segment'): continue
                        seg = trial[seg_key]
                        
                        # Check segment attributes for label
                        # Common keys: 'label', 'target', 'class'
                        # print(dict(seg.attrs))
                        
                        # Collect lengths
                        if 'eeg' in seg:
                            shape = seg['eeg'].shape
                            segment_lengths.append(shape[1])
                        
                        # To find classification labels, we might need to look deeper
                        # Sometimes label is in the group name or a specific dataset
                        
                        # Let's dump all attributes of one segment to be sure
                        if len(labels) == 0 and len(segment_lengths) == 1:
                            print(f"Sample Segment Attrs: {dict(seg.attrs)}")
                            
                        # Assuming 'label' attribute exists
                        if 'label' in seg.attrs:
                            labels.add(seg.attrs['label'])
                        elif 'seizure_type' in seg.attrs:
                            labels.add(seg.attrs['seizure_type'])
                        
        except Exception as e:
            pass
            
    return labels, segment_lengths

dataset_dir = "/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure"
labels, lengths = scan_labels(dataset_dir)

print(f"\nUnique Labels Found: {labels}")
if lengths:
    print(f"Segment Lengths (Samples): Min={min(lengths)}, Max={max(lengths)}, Mean={np.mean(lengths):.2f}")
    print(f"Segment Lengths (Seconds @ 200Hz?): Min={min(lengths)/200:.2f}s, Max={max(lengths)/200:.2f}s")
