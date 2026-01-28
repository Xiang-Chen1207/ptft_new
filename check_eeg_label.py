
import h5py
import numpy as np
import os

def check_eeg_label(file_path):
    print(f"Inspecting {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            trial_key = list(f.keys())[0]
            print(f"Checking {trial_key}...")
            trial_group = f[trial_key]
            
            # Find a segment
            seg_keys = [k for k in trial_group.keys() if k.startswith('segment')]
            if not seg_keys:
                print("No segments found.")
                return

            seg_key = seg_keys[0]
            print(f"Checking {seg_key}...")
            seg_group = trial_group[seg_key]
            
            # Check keys in segment
            print(f"Segment keys: {list(seg_group.keys())}")
            
            if 'eeg' in seg_group:
                eeg_obj = seg_group['eeg']
                print(f"'eeg' object type: {type(eeg_obj)}")
                
                if isinstance(eeg_obj, h5py.Group):
                    print(f"'eeg' is a Group. Keys: {list(eeg_obj.keys())}")
                    if 'label' in eeg_obj:
                        print(f"Found 'label' in 'eeg' group: {eeg_obj['label'][()]}")
                    if 'data' in eeg_obj:
                         print(f"Found 'data' in 'eeg' group. Shape: {eeg_obj['data'].shape}")
                elif isinstance(eeg_obj, h5py.Dataset):
                    print(f"'eeg' is a Dataset. Shape: {eeg_obj.shape}")
                    print(f"'eeg' Dataset Attributes: {dict(eeg_obj.attrs)}")
            else:
                print("'eeg' key not found in segment.")

            # Check for other potential label keys in segment
            for key in seg_group.keys():
                if key not in ['eeg', 'data']:
                    obj = seg_group[key]
                    if isinstance(obj, h5py.Dataset):
                        print(f"Found other dataset '{key}': {obj[()]}")
                    elif isinstance(obj, h5py.Group):
                        print(f"Found other group '{key}': {list(obj.keys())}")

    except Exception as e:
        print(f"Error: {e}")

# Use one of the files we found earlier
file_path = "/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure/sub_aaaaaaac.h5"
check_eeg_label(file_path)
