import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import glob

# Add cbramod_tuab to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../cbramod_tuab')))

from util.datasets import EEGDataset
from models.modeling_finetune import CBraModClassifier

def _group_files_by_subject(all_h5_files):
    subject_files = {}
    for f in all_h5_files:
        basename = os.path.basename(f)
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
        
    return files_split

def main():
    data_root = "/vepfs-0x0d/eeg-data/TUAB"
    checkpoint_path = "/vePFS-0x0d/home/chen/cbramod_tuab/output_dir/checkpoint-best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_workers = 8
    seed = 42  # Match other experiments (finetune.yaml, extract_features.py)
    
    print(f"Device: {device}")
    
    # 1. Dataset Split
    all_h5_files = sorted(glob.glob(os.path.join(data_root, 'sub_*.h5')))
    print(f"Found {len(all_h5_files)} H5 files.")
    
    rng = np.random.RandomState(seed)
    rng.shuffle(all_h5_files)
    
    subject_files = _group_files_by_subject(all_h5_files)
    splits = _split_subjects(subject_files, seed)
    test_files = splits['test']
    print(f"Test files: {len(test_files)}")
    
    target_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
    
    test_dataset = EEGDataset(
        file_list=test_files,
        target_channels=target_channels
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 2. Model
    print("Loading model...")
    model = CBraModClassifier(
        in_dim=200, 
        out_dim=200, 
        d_model=200, 
        dim_feedforward=800, 
        seq_len=30, 
        n_layer=12, 
        nhead=8, 
        num_classes=1 # Config said nb_classes: 1
    )
    
    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # The checkpoint from finetuning (save_model in utils.py) typically saves 'model' key.
    # Also, DDP wraps it.
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
        
    msg = model.load_state_dict(new_state_dict, strict=True)
    print(f"Load status: {msg}")
    
    model.to(device)
    model.eval()
    
    # 3. Evaluation
    print("Evaluating...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # EEGDataset returns: data, label
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # Binary classification (nb_classes=1) -> Logits
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    
    print(f"--- Results ---")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {bacc*100:.2f}%")

if __name__ == '__main__':
    main()
