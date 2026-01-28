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

def _group_files_by_subject(all_h5_files):
    subject_files = {}
    for f in all_h5_files:
        basename = os.path.basename(f)
        try:
            # Assumes sub_aaaaaaar.h5 format where 'aaaaaaar' is subject ID
            subject_id = basename.split('_')[1].split('.')[0]
        except IndexError:
            subject_id = basename.split('.')[0]
            
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(f)
    return subject_files

def get_tusz_file_list(dataset_dir, mode='train', seed=42):
    """
    Get file list for TUSZ (TUH Seizure).
    """
    # Look for files in TUH_Seizure subdirectory if it exists, otherwise just recursively search
    search_dir = os.path.join(dataset_dir, 'TUH_Seizure')
    if not os.path.exists(search_dir):
        search_dir = dataset_dir
        
    all_h5_files = sorted(glob.glob(os.path.join(search_dir, '**', '*.h5'), recursive=True))
    all_h5_files = [f for f in all_h5_files if 'sub_' in os.path.basename(f)]
    
    print(f"Found {len(all_h5_files)} H5 files in {search_dir}")
    
    rng = np.random.RandomState(seed)
    
    subject_files = _group_files_by_subject(all_h5_files)
    unique_subjects = sorted(list(subject_files.keys()))
    rng.shuffle(unique_subjects)
    
    n_subjects = len(unique_subjects)
    
    # 60/20/20 split
    n_train = int(n_subjects * 0.60)
    n_val = int(n_subjects * 0.20)
    # Remaining 20% for test
    
    splits = {
        'train': unique_subjects[:n_train],
        'val': unique_subjects[n_train:n_train+n_val],
        'test': unique_subjects[n_train+n_val:]
    }
    
    target_subjects = splits.get(mode, [])
    file_list = []
    for s in target_subjects:
        file_list.extend(subject_files[s])
        
    print(f"[{mode}] Selected {len(file_list)} files from {len(target_subjects)} subjects")
    return file_list

# --- Indexing Logic ---

def _index_worker(h5_path):
    samples = []
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                for seg_key in segment_keys:
                    seg_group = trial_group[seg_key]
                    
                    # Extract label
                    label = -1
                    if 'eeg' in seg_group:
                        dset = seg_group['eeg']
                        if 'label' in dset.attrs:
                            label_vec = dset.attrs['label']
                            # TUSZ is 13-class classification
                            # Label is usually one-hot vector or distribution
                            if hasattr(label_vec, '__len__'):
                                label = np.argmax(label_vec)
                            else:
                                label = int(label_vec)
                                
                    samples.append({
                        'file_path': h5_path,
                        'trial_key': trial_key,
                        'segment_key': seg_key,
                        'label': int(label)
                    })
    except Exception:
        pass
    return samples

class TUSZDataset(Dataset):
    def __init__(self, file_list, input_size=2000, transform=None, cache_path='dataset_index_tusz.json', mode='train', **kwargs):
        super().__init__()
        self.input_size = input_size # Default to 10s (2000 samples @ 200Hz)
        self.transform = transform
        
        # Standard 19 channels
        self.target_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
        # TUSZ source channels might vary, but usually follow 10-20
        self.source_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2', 'A1', 'A2']
        
        self.channel_indices = []
        for target in self.target_channels:
            # Handle standard renaming
            mapped_target = target
            if target == 'T3': mapped_target = 'T7'
            if target == 'T4': mapped_target = 'T8'
            if target == 'T5': mapped_target = 'P7'
            if target == 'T6': mapped_target = 'P8'
            
            try:
                idx = self.source_channels.index(mapped_target)
                self.channel_indices.append(idx)
            except ValueError:
                # Fallback to direct name if mapped failed (e.g. if file uses T3 instead of T7)
                try:
                    idx = self.source_channels.index(target)
                    self.channel_indices.append(idx)
                except ValueError:
                    print(f"Warning: Channel {target} (mapped to {mapped_target}) not found in source map.")
        
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
                if input_files_set.issubset(cached_files):
                     reindex = False
                else:
                     print("Cache incomplete. Re-indexing...")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        if reindex:
            print(f"Indexing {len(file_list)} files...")
            max_workers = min(32, os.cpu_count() or 1)
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
        
        data = np.zeros((len(self.channel_indices), self.input_size), dtype=np.float32)
        
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

            group = f[trial_key][seg_key]
            dset = group['eeg']
            
            # Read all
            raw = dset[:]
            
            # Check orientation
            if raw.shape[0] != 21 and raw.shape[1] == 21:
                raw = raw.T
            
            # Select channels
            if raw.shape[0] >= 21: # Ensure we have enough channels
                raw = raw[self.channel_indices, :]
            
            # Handle time dimension
            current_len = raw.shape[1]
            if current_len >= self.input_size:
                if self.mode == 'train':
                    # Random crop for training
                    start = np.random.randint(0, current_len - self.input_size + 1)
                else:
                    # Center crop for val/test
                    start = (current_len - self.input_size) // 2
                
                data = raw[:, start:start+self.input_size]
            else:
                pad = self.input_size - current_len
                data = np.pad(raw, ((0,0), (0, pad)), 'constant')
                    
        except Exception as e:
            # If cache error, try to reopen
            if h5_path in self.file_cache:
                try:
                    self.file_cache.pop(h5_path).close()
                except:
                    pass
            # print(f"Error loading {h5_path}: {e}")
            pass

        tensor = torch.from_numpy(data).float()
        
        # Normalize
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / (std + 1e-6)
        
        # Patchify
        patch_size = 200
        if self.input_size % patch_size == 0:
            num_patches = self.input_size // patch_size
            tensor = tensor.view(tensor.shape[0], num_patches, patch_size)

        return tensor, label
