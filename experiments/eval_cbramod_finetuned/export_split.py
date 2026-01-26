import os
import glob
import json
import numpy as np

def _group_files_by_subject(all_h5_files):
    subject_files = {}
    for f in all_h5_files:
        basename = os.path.basename(f)
        # sub_00000001.h5 -> 00000001
        subject_id = basename.split('_')[1].split('.')[0]
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(f)
    return subject_files

def _split_subjects(subject_files, seed, ratios=(0.8, 0.1)):
    unique_subjects = sorted(list(subject_files.keys()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjects)
    
    n_subjects = len(unique_subjects)
    n_train = int(n_subjects * ratios[0])
    n_val = int(n_subjects * ratios[1])
    
    splits = {
        'train': unique_subjects[:n_train],
        'val': unique_subjects[n_train:n_train+n_val],
        'test': unique_subjects[n_train+n_val:]
    }
    
    files_split = {}
    for key in ['train', 'val', 'test']:
        files = []
        for s in splits[key]:
            files.extend(subject_files[s])
        files_split[key] = files
        
    print(f"Subject Split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    return files_split

def main():
    data_root = "/vepfs-0x0d/eeg-data/TUAB"
    output_dir = os.path.dirname(os.path.abspath(__file__))
    seed = 0 # Match evaluation seed
    
    print(f"Data Root: {data_root}")
    print(f"Seed: {seed}")
    
    # Get all H5 files
    all_h5_files = sorted(glob.glob(os.path.join(data_root, 'sub_*.h5')))
    print(f"Found {len(all_h5_files)} H5 files.")
    
    if len(all_h5_files) == 0:
        print("Error: No H5 files found! Check data_root path.")
        return

    # Cross-Subject Split (Randomly split files)
    # Shuffle first (with fixed seed for reproducibility across ranks)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_h5_files)
    
    # 1. Group by subject ID
    subject_files = _group_files_by_subject(all_h5_files)
    
    # 2. Split subjects (Train/Val/Test)
    # Note: _split_subjects returns a dict here, unlike list in original script, adapted for clarity
    files_split = _split_subjects(subject_files, seed)
    
    print(f"Total Files: Train={len(files_split['train'])}, Val={len(files_split['val'])}, Test={len(files_split['test'])}")

    # Save dataset split to file
    split_info = {
        'train': files_split['train'],
        'val': files_split['val'],
        'test': files_split['test']
    }
    split_path = os.path.join(output_dir, 'dataset_split.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=4)
    print(f"Dataset split saved to {split_path}")

if __name__ == '__main__':
    main()
