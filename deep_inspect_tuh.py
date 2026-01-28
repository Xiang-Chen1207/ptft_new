
import h5py
import numpy as np
import os
import glob

# Try to find a file with labels, maybe metadata is not in every segment or file structure is different
# TUH Seizure usually has seizure vs background labels.

def deep_inspect(file_path):
    print(f"Deep inspecting {file_path}")
    with h5py.File(file_path, 'r') as f:
        # Check trial attributes again
        for trial_key in f.keys():
            if not trial_key.startswith('trial'): continue
            trial = f[trial_key]
            print(f"  Trial {trial_key} attrs: {dict(trial.attrs)}")
            
            # Maybe label is a dataset inside segment?
            for seg_key in trial.keys():
                if not seg_key.startswith('segment'): continue
                seg = trial[seg_key]
                print(f"    Segment {seg_key} keys: {list(seg.keys())}")
                if 'label' in seg:
                    print(f"      Found 'label' dataset: {seg['label'][()]}")
                if 'target' in seg:
                    print(f"      Found 'target' dataset: {seg['target'][()]}")
                break # Just check first segment
            break # Just check first trial

# Pick a few random files
files = glob.glob("/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure/*.h5")[:3]
for f in files:
    deep_inspect(f)
