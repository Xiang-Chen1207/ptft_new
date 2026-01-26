import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from models.wrapper import CBraModWrapper
from datasets.tuab import TUABDataset, get_tuab_file_list

def get_config(model_type):
    config = {
        'model': {
            'in_dim': 200,
            'd_model': 200,
            'dim_feedforward': 800,
            'seq_len': 30,
            'n_layer': 12,
            'nhead': 8,
            'dropout': 0.1,
            'num_classes': 1,
            'pretrain_tasks': ['reconstruction'],
            'feature_token_type': 'gap',
            'feature_token_strategy': 'single',
            'feature_dim': 200,
        },
        'task_type': 'pretraining'
    }

    if model_type == 'neuro_ke':
        # Multi-task: reconstruction + feature prediction
        config['model']['pretrain_tasks'] = ['reconstruction', 'feature_pred']
        config['model']['feature_token_type'] = 'cross_attn'
        config['model']['feature_token_strategy'] = 'single'
        config['model']['feature_dim'] = 62
    elif model_type == 'feat_only':
        # Single-task: feature prediction only (sanity_feat_only)
        config['model']['pretrain_tasks'] = ['feature_pred']
        config['model']['feature_token_type'] = 'cross_attn'
        config['model']['feature_token_strategy'] = 'single'
        config['model']['feature_dim'] = 62

    return config

def load_model(model_type, weights_path, device):
    config = get_config(model_type)
    model = CBraModWrapper(config)
    
    print(f"Loading weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    
    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs for feature extraction.")
        model.backbone = nn.DataParallel(model.backbone)
        # Note: We only parallelize the backbone as it's the heaviest part.
        # The heads (GAP, CrossAttn) will run on the main GPU, which is efficient enough.
        
    return model

def get_subject_id(file_path):
    basename = os.path.basename(file_path)
    try:
        return basename.split('_')[1].split('.')[0]
    except:
        return basename.split('.')[0]

def extract_features_dataset(model, dataset, device, batch_size, num_workers, model_type):
    # Optimize num_workers based on CPU count
    # num_workers = min(16, os.cpu_count() or 1)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    eeg_feats_list = []
    feat_tokens_list = []
    pred_features_list = []
    labels_list = []
    
    subject_ids = [get_subject_id(s['file_path']) for s in dataset.samples]
    
    print(f"Extracting features for {len(dataset)} samples...")
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data = data.to(device)
            
            # Backbone (Parallelized)
            feats = model.backbone(data) # (B, C, N, D)
            
            # 1. EEG Token (GAP)
            # feats is on device (likely gpu:0 if using DataParallel)
            eeg_feat = feats.mean(dim=[1, 2])
            eeg_feats_list.append(eeg_feat.cpu().numpy())
            
            # 2. Feat Token & Predicted Features (Cross Attn) - For Neuro-KE and Feat-Only
            if model_type in ['neuro_ke', 'feat_only']:
                B, C, N, D = feats.shape
                feats_flat = feats.view(B, C * N, D)
                query = model.feat_query.expand(B, -1, -1)
                attn_output, _ = model.feat_attn(query, feats_flat, feats_flat)
                if model.feature_token_strategy == 'single':
                    feat_token = attn_output.squeeze(1)
                else:
                    feat_token = attn_output.reshape(B, -1)
                feat_tokens_list.append(feat_token.cpu().numpy())
                # Predicted features via feature_head
                pred_feat = model.feature_head(feat_token)
                pred_features_list.append(pred_feat.detach().cpu().numpy())
            
            labels_list.append(target.numpy())
            
    eeg_feats = np.concatenate(eeg_feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    subject_ids = np.array(subject_ids)
    
    feat_tokens = None
    if feat_tokens_list:
        feat_tokens = np.concatenate(feat_tokens_list, axis=0)
    pred_features = None
    if pred_features_list:
        pred_features = np.concatenate(pred_features_list, axis=0)
        
    return {
        'eeg': eeg_feats,
        'feat': feat_tokens,
        'pred': pred_features,
        'labels': labels,
        'subjects': subject_ids
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['recon', 'neuro_ke', 'feat_only'], required=True)
    parser.add_argument('--dataset_dir', type=str, default='/vepfs-0x0d/eeg-data/TUAB')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='experiments/tuab_lp/features')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading Dataset Index...")
    train_files = get_tuab_file_list(args.dataset_dir, 'train', seed=42)
    test_files = get_tuab_file_list(args.dataset_dir, 'test', seed=42)
    
    train_dataset = TUABDataset(train_files)
    test_dataset = TUABDataset(test_files)
    
    # 2. Load Model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_type, args.weights_path, device)
    
    # 3. Extract
    print("Extracting Train Features...")
    train_data = extract_features_dataset(model, train_dataset, device, args.batch_size, args.num_workers, args.model_type)
    
    print("Extracting Test Features...")
    test_data = extract_features_dataset(model, test_dataset, device, args.batch_size, args.num_workers, args.model_type)
    
    # 4. Save
    output_path = os.path.join(args.output_dir, f"{args.model_type}_features.npz")
    print(f"Saving to {output_path}...")
    
    save_dict = {
        'train_eeg': train_data['eeg'],
        'train_labels': train_data['labels'],
        'train_subjects': train_data['subjects'],
        'test_eeg': test_data['eeg'],
        'test_labels': test_data['labels'],
        'test_subjects': test_data['subjects']
    }
    
    if args.model_type in ['neuro_ke', 'feat_only']:
        save_dict['train_feat'] = train_data['feat']
        save_dict['test_feat'] = test_data['feat']
        if train_data['pred'] is not None and test_data['pred'] is not None:
            save_dict['train_pred'] = train_data['pred']
            save_dict['test_pred'] = test_data['pred']
        
    np.savez_compressed(output_path, **save_dict)
    print("Done.")

if __name__ == '__main__':
    main()
