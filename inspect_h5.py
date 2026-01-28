import h5py
import numpy as np
import sys

file_path = '/vePFS-0x0d/pretrain-clip/benchmark_dataloader/hdf5_output/TUH_Events/sub_aaaaaaar.h5'

def print_structure(name, obj):
    indent = "  " * (name.count('/') + 1)
    print(f"{indent}Name: {name}")
    
    # Print attributes
    if len(obj.attrs) > 0:
        print(f"{indent}Attributes:")
        for key, val in obj.attrs.items():
            print(f"{indent}  {key}: {val}")
    
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}Type: Dataset, Shape: {obj.shape}, Dtype: {obj.dtype}")
        
        # Heuristics to print content
        if obj.size < 50:
            try:
                data = obj[:]
                if isinstance(data[0], bytes):
                    print(f"{indent}Content: {[d.decode('utf-8') if isinstance(d, bytes) else d for d in data]}")
                else:
                    print(f"{indent}Content: {data}")
            except Exception as e:
                print(f"{indent}Could not print content: {e}")
        elif 'channel' in name.lower():
             try:
                data = obj[:min(obj.size, 25)]
                print(f"{indent}Content (first {len(data)}): {[d.decode('utf-8') if isinstance(d, bytes) else d for d in data]}")
             except:
                pass

print(f"Inspecting {file_path}...")

try:
    with h5py.File(file_path, 'r') as f:
        print(f"Root Keys: {list(f.keys())}")
        # Print root attributes
        if len(f.attrs) > 0:
            print("Root Attributes:")
            for key, val in f.attrs.items():
                print(f"  {key}: {val}")
                
        f.visititems(print_structure)
        
except Exception as e:
    print(f"Error: {e}")
