
import h5py
import numpy as np
import os

def inspect_h5(file_path):
    print(f"Inspecting {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            # Print structure
            print(f"Keys: {list(f.keys())}")
            
            trial_key = list(f.keys())[0]
            print(f"Sample Trial Key: {trial_key}")
            
            trial_group = f[trial_key]
            print(f"Trial Group Keys: {list(trial_group.keys())}")
            print(f"Trial Attributes: {dict(trial_group.attrs)}")
            
            # Check for data/eeg
            data = None
            if 'data' in trial_group:
                data = trial_group['data']
            elif 'eeg' in trial_group:
                data = trial_group['eeg']
            
            # If not in trial group, check segment
            seg_key = [k for k in trial_group.keys() if k.startswith('segment')]
            if seg_key:
                print(f"Found Segments: {len(seg_key)}")
                seg_group = trial_group[seg_key[0]]
                print(f"Sample Segment Group Keys: {list(seg_group.keys())}")
                if 'data' in seg_group:
                    data = seg_group['data']
                elif 'eeg' in seg_group:
                    data = seg_group['eeg']
            
            if data is not None:
                print(f"Data Shape: {data.shape}")
                
            # Check for labels in other trials/files if possible, or print attributes
            
    except Exception as e:
        print(f"Error: {e}")

file_path = "/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure/sub_aaaaaaac.h5"
inspect_h5(file_path)
