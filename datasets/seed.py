
import os
import glob
import h5py
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import OrderedDict

# --- Splitting Logic ---

def get_seed_file_list(dataset_dir, mode='train', seed=42):
    """
    Get file list for SEED dataset.
    Splitting: First 12 subjects for train, last 3 for test.
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
    all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, '*.h5')))
    
    # Sort by subject ID to ensure consistent order (sub_1, sub_2, ... sub_10, ...)
    def extract_id(fname):
        basename = os.path.basename(fname)
        # Assumes sub_X.h5
        try:
            return int(basename.split('_')[1].split('.')[0])
        except:
            return 999
            
    all_h5_files.sort(key=extract_id)
    
    print(f"Found {len(all_h5_files)} H5 files in {dataset_dir}")
    
    # Split logic: 12 train, 3 test
    # Files are sorted by ID: sub_1, sub_2, ..., sub_15
    if len(all_h5_files) < 15:
        print(f"Warning: Expected at least 15 subjects for SEED, found {len(all_h5_files)}")
    
    n_train = 12
    
    splits = {
        'train': all_h5_files[:n_train],
        'val': all_h5_files[n_train:], # Use test set for validation as well
        'test': all_h5_files[n_train:]
    }
    
    file_list = splits.get(mode, [])
    print(f"[{mode}] Selected {len(file_list)} files")
    return file_list

# --- Indexing Logic ---

def _index_worker(h5_path):
    samples = []
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            # Structure: trialX / segmentY / eeg
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                if not isinstance(trial_group, h5py.Group):
                    continue
                    
                for seg_key in [k for k in trial_group.keys() if k.startswith('segment')]:
                    seg_group = trial_group[seg_key]
                    if not isinstance(seg_group, h5py.Group):
                        continue
                        
                    if 'eeg' in seg_group:
                        dset = seg_group['eeg']
                        # Extract label from attributes
                        label = -1
                        if 'label' in dset.attrs:
                            lbl_attr = dset.attrs['label']
                            if hasattr(lbl_attr, '__iter__') and len(lbl_attr) == 1:
                                label = int(lbl_attr[0])
                            elif hasattr(lbl_attr, '__iter__'):
                                # Fallback if it's a longer array, though usually it's single int
                                label = int(lbl_attr[0]) 
                            else:
                                label = int(lbl_attr)
                        
                        samples.append({
                            'file_path': h5_path,
                            'trial_key': trial_key,
                            'segment_key': seg_key,
                            'label': label
                        })
    except Exception as e:
        print(f"Error indexing {h5_path}: {e}")
    return samples

class SEEDDataset(Dataset):
    def __init__(self, file_list, input_size=400, transform=None, cache_path='dataset_index_seed.json', mode='train', **kwargs):
        super().__init__()
        self.input_size = input_size # Default to 2s (400 samples @ 200Hz)
        self.transform = transform
        self.mode = mode
        
        # 62 channels
        self.num_channels = 62
        
        self.samples = self._load_or_generate_index(file_list, cache_path)
        
        # File handle cache
        self.file_cache = OrderedDict()
        self.cache_size = 128

    def _load_or_generate_index(self, file_list, cache_path):
        full_index = []
        input_files_set = set(file_list)
        reindex = True
        
        if os.path.exists(cache_path):
            print(f"Loading index from {cache_path}...")
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    full_index = data if isinstance(data, list) else data.get('samples', [])
                
                cached_files = set(s['file_path'] for s in full_index)
                # Check if our input files are covered by cache
                # Since SEED is small, we can probably just rely on the cache if it contains our files
                if input_files_set.issubset(cached_files):
                     reindex = False
                else:
                     print("Cache incomplete. Re-indexing...")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        if reindex:
            print(f"Indexing {len(file_list)} files...")
            max_workers = min(16, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(_index_worker, file_list), total=len(file_list)))
            new_samples = [s for res in results for s in res]
            
            # Merge with existing
            existing_samples_map = { (s['file_path'], s['trial_key'], s['segment_key']): s for s in full_index }
            for s in new_samples:
                key = (s['file_path'], s['trial_key'], s['segment_key'])
                existing_samples_map[key] = s
            
            full_index = list(existing_samples_map.values())
            
            try:
                with open(cache_path, 'w') as f:
                    json.dump(full_index, f)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
                
        filtered_samples = [s for s in full_index if s['file_path'] in input_files_set]
        print(f"Dataset initialized: {len(filtered_samples)} samples.")
        
        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        h5_path, trial_key, seg_key = info['file_path'], info['trial_key'], info['segment_key']
        label = info['label']
        
        data = np.zeros((self.num_channels, self.input_size), dtype=np.float32)
        
        try:
            # Manage File Cache
            if h5_path in self.file_cache:
                f = self.file_cache[h5_path]
                self.file_cache.move_to_end(h5_path)
            else:
                if len(self.file_cache) >= self.cache_size:
                    old_path, old_f = self.file_cache.popitem(last=False)
                    try:
                        old_f.close()
                    except:
                        pass
                
                f = h5py.File(h5_path, 'r', rdcc_nbytes=4*1024*1024, libver='latest')
                self.file_cache[h5_path] = f

            dset = f[trial_key][seg_key]['eeg']
            
            # Read all
            raw = dset[:] # Shape (62, 400)
            
            # Ensure shape is (Channels, Time)
            if raw.shape[0] != 62 and raw.shape[1] == 62:
                raw = raw.T
            
            # Handle time dimension
            current_len = raw.shape[1]
            if current_len >= self.input_size:
                # Typically it matches exactly 400, but if larger, crop
                if self.mode == 'train':
                    start = np.random.randint(0, current_len - self.input_size + 1)
                else:
                    start = (current_len - self.input_size) // 2
                data = raw[:, start:start+self.input_size]
            else:
                # Pad if shorter
                pad = self.input_size - current_len
                data = np.pad(raw, ((0,0), (0, pad)), 'constant')
                    
        except Exception as e:
            # If cache error, try to reopen
            if h5_path in self.file_cache:
                try:
                    self.file_cache.pop(h5_path).close()
                except:
                    pass
            print(f"Error loading {h5_path}: {e}")
            pass

        tensor = torch.from_numpy(data).float()
        
        # Normalize
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / (std + 1e-6)
        
        # Patchify
        # Input: (62, 400) -> Output: (62, 2, 200)
        patch_size = 200
        if self.input_size % patch_size == 0:
            num_patches = self.input_size // patch_size
            tensor = tensor.view(tensor.shape[0], num_patches, patch_size)

        return tensor, label
