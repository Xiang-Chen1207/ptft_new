import os
import glob
import h5py
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Splitting Logic (adapted from TUAB) ---

def _group_files_by_subject(all_h5_files):
    subject_files = {}
    for f in all_h5_files:
        basename = os.path.basename(f)
        try:
            # Assumes sub_00000001.h5 or similar format
            # If TUEG has different naming, this might need adjustment, 
            # but generally "sub_ID" is the pattern.
            subject_id = basename.split('_')[1].split('.')[0]
        except IndexError:
            subject_id = basename.split('.')[0]
            
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(f)
    return subject_files

def get_tueg_file_list(dataset_dir, mode='train', seed=42):
    """
    Get file list for TUEG.
    For pretraining, we typically use ALL data as 'train'.
    If mode is 'train', we might return everything.
    But to support optional validation, we can still do a split.
    """
    all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True))
    # Filter for sub_*.h5 if needed, or just all h5
    all_h5_files = [f for f in all_h5_files if 'sub_' in os.path.basename(f)]
    
    print(f"Found {len(all_h5_files)} H5 files in {dataset_dir}")
    
    rng = np.random.RandomState(seed)
    rng.shuffle(all_h5_files)
    
    # If the user wants NO separation (all for training), we return all for train.
    # However, having a small val set is usually good for monitoring reconstruction.
    # Let's use a 95/5 split for pretraining validation if requested.
    
    subject_files = _group_files_by_subject(all_h5_files)
    unique_subjects = sorted(list(subject_files.keys()))
    rng.shuffle(unique_subjects)
    
    n_subjects = len(unique_subjects)
    
    # Simple split: 95% train, 5% val for monitoring
    n_train = int(n_subjects * 0.95)
    
    splits = {
        'train': unique_subjects[:n_train],
        'val': unique_subjects[n_train:],
        'test': [] # Usually no test set for pretraining
    }
    
    # If mode is not recognized or we want everything, just return everything?
    # But standard trainer expects 'train' and 'val' loaders.
    
    target_subjects = splits.get(mode, [])
    file_list = []
    for s in target_subjects:
        file_list.extend(subject_files[s])
        
    return file_list

# --- Indexing Logic ---

def _index_worker(h5_path):
    samples = []
    # For TUEG pretraining, we might not care about labels (normal/abnormal),
    # but we need valid segments.
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                
                # We can store label if available, or just dummy
                label = -1 
                
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                if segment_keys:
                    for seg_key in segment_keys:
                        samples.append({
                            'file_path': h5_path,
                            'trial_key': trial_key,
                            'segment_key': seg_key,
                            'label': label
                        })
                else:
                    samples.append({
                        'file_path': h5_path,
                        'trial_key': trial_key,
                        'segment_key': None,
                        'label': label
                    })
    except Exception:
        pass
    return samples

class TUEGDataset(Dataset):
    def __init__(self, file_list, input_size=2000, transform=None, cache_path='dataset_index.json', feature_path=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.transform = transform
        
        # Load features if provided
        self.feature_map = None
        self.feature_dim = 0
        if feature_path and os.path.exists(feature_path):
            print(f"Loading features from {feature_path}...")
            try:
                import pandas as pd
                df = pd.read_csv(feature_path)
                # Filter for window_idx == 0 (start of segment)
                if 'window_idx' in df.columns:
                    df = df[df['window_idx'] == 0]
                
                # Identify feature columns
                meta_cols = ['aug_type', 'segment_id', 'window_idx', 'start_time', 'subject_id', 'source_file']
                feature_cols = [c for c in df.columns if c not in meta_cols]
                
                self.feature_dim = len(feature_cols)
                self.feature_names = feature_cols # Expose names
                print(f"Found {self.feature_dim} features.")
                
                # Build index
                self.feature_map = {}
                # Pre-compute basename for matching
                df['basename'] = df['source_file'].apply(os.path.basename)
                
                # Use numpy for faster iteration or vectorized ops if possible, but iterrows is safe
                # Create a dict directly might be faster
                # (basename, segment_id) -> values
                # Zip is faster than iterrows
                keys = zip(df['basename'], df['segment_id'])
                vals = df[feature_cols].values.astype(np.float32)
                
                for i, (fname, seg_id) in enumerate(keys):
                    self.feature_map[(fname, int(seg_id))] = vals[i]
                
                print(f"Indexed {len(self.feature_map)} feature entries.")
                    
            except Exception as e:
                print(f"Error loading features: {e}")
                self.feature_map = None
        
        # Standard 19 channels (Same as TUAB/CBraMod)
        self.target_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
        self.source_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2', 'A1', 'A2']
        
        self.channel_indices = []
        for target in self.target_channels:
            try:
                idx = self.source_channels.index(target)
                self.channel_indices.append(idx)
            except ValueError:
                print(f"Warning: Channel {target} not found in source map.")
        
        self.samples = self._load_or_generate_index(file_list, cache_path)

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
        print(f"Dataset initialized: {len(filtered_samples)} samples (from {len(file_list)} files).")
        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        h5_path, trial_key, seg_key = info['file_path'], info['trial_key'], info['segment_key']
        
        data = np.zeros((len(self.channel_indices), self.input_size), dtype=np.float32)
        try:
            with h5py.File(h5_path, 'r') as f:
                group = f[trial_key][seg_key] if seg_key else f[trial_key]
                if 'eeg' in group:
                    raw = group['eeg'][:]
                elif 'data' in group:
                    raw = group['data'][:]
                else:
                    raise KeyError("No data")

                if raw.shape[0] != 21 and raw.shape[1] == 21:
                    raw = raw.T
                
                raw = raw[self.channel_indices, :]
                
                if raw.shape[1] < self.input_size:
                    pad = self.input_size - raw.shape[1]
                    data = np.pad(raw, ((0,0), (0, pad)), 'constant')
                else:
                    data = raw[:, :self.input_size]
        except Exception as e:
            # print(f"Error loading {h5_path}: {e}")
            pass

        tensor = torch.from_numpy(data).float()
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / (std + 1e-6)
        
        patch_size = 200
        if self.input_size % patch_size == 0:
            num_patches = self.input_size // patch_size
            tensor = tensor.view(tensor.shape[0], num_patches, patch_size)
            
        # Retrieve features if map exists
        features = None
        if self.feature_map is not None:
            basename = os.path.basename(h5_path)
            # Parse segment id from key (e.g. 'segment20' -> 20)
            seg_id = 0
            if seg_key and seg_key.startswith('segment'):
                try:
                    seg_id = int(seg_key.replace('segment', ''))
                except ValueError:
                    seg_id = 0
            
            features_np = self.feature_map.get((basename, seg_id))
            if features_np is not None:
                # Robust Scaling: Clip to [-6, 6] (6 stds) to handle extreme outliers
                # This prevents R2 explosion due to long-tailed artifacts
                features_np = np.clip(features_np, -6.0, 6.0)
                features = torch.from_numpy(features_np)
            else:
                # Missing features for this segment
                features = torch.zeros(self.feature_dim, dtype=torch.float32)
        
        if features is not None:
            return tensor, features
        else:
            return tensor, 0 # Dummy label for pretraining
