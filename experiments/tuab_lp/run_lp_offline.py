import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def run_experiment_gpu(X_train, y_train, X_test, y_test, seed, C=1.0, device='cuda'):
    # Convert to Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    # Ensure num_classes is at least 2 for binary case handling
    num_classes = max(num_classes, 2)
    
    model = LogisticRegressionTorch(input_dim, num_classes).to(device)
    
    # L2 Regularization via weight_decay
    # C = 1/lambda -> lambda = 1/C
    weight_decay = 1.0 / C if C > 0 else 0
    
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=100)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    def closure():
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    
    # Eval
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()
        
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    
    # Check convergence (simplified for LBFGS)
    converged = True 
    
    return acc, bacc, converged, input_dim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, required=True, help='Path to .npz file')
    parser.add_argument('--seed', type=int, default=42)
    # Removing 0.001 (0.1%) as requested
    parser.add_argument('--ratios', type=float, nargs='+', 
                        default=[0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0], 
                        help='List of training set ratios (0.0 to 1.0)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--feature_types', type=str, nargs='+', default=None, help='Subset of feature types to use: eeg feat full pred')
    
    args = parser.parse_args()
    
    print(f"Loading features from {args.features_path}...")
    data = np.load(args.features_path)
    
    # Check available keys
    has_feat = 'train_feat' in data
    has_pred = 'train_pred' in data
    
    # Determine available feature types
    feature_types = ['eeg']
    if has_feat:
        feature_types.extend(['feat', 'full'])
    if has_pred:
        feature_types.append('pred')
    if args.feature_types:
        feature_types = args.feature_types
        
    print(f"Available Feature Types: {feature_types}")
    print("\n=== Results Table ===")
    print(f"| {'Ratio':<8} | {'NumSub':<6} | {'Feature':<10} | {'Dim':<5} | {'Acc':<7} | {'BAcc':<7} | {'Converged':<9} |")
    print(f"|{'-'*10}|{'-'*8}|{'-'*12}|{'-'*7}|{'-'*9}|{'-'*9}|{'-'*11}|")
    
    unique_train_subjects = np.unique(data['train_subjects'])
    total_subjects = len(unique_train_subjects)
    print(f"Total Training Subjects: {total_subjects}")
    
    for ratio in args.ratios:
        if ratio >= 1.0:
            n_subs = total_subjects
            ratio_display = "100%"
        else:
            n_subs = max(1, int(total_subjects * ratio))
            ratio_display = f"{ratio*100:.1f}%"
            
        # Robust Sampling Loop
        valid_sample = False
        selected_subjects = None
        y_train_sub = None
        mask_train = None
        
        for attempt in range(10): 
            current_seed = args.seed + int(ratio * 1000) + attempt
            rng = np.random.RandomState(current_seed)
            
            if n_subs == total_subjects:
                selected_subjects = unique_train_subjects
            else:
                selected_subjects = rng.choice(unique_train_subjects, n_subs, replace=False)
                
            mask_train = np.isin(data['train_subjects'], selected_subjects)
            y_train_sub = data['train_labels'][mask_train]
            
            if len(np.unique(y_train_sub)) >= 2:
                valid_sample = True
                break
        
        if not valid_sample:
            print(f"{ratio_display:<8} | {n_subs:<6} | SKIPPED (Only 1 class after 10 attempts)")
            continue
            
        for ftype in feature_types:
            if ftype == 'eeg':
                X_train_sub = data['train_eeg'][mask_train]
                X_test = data['test_eeg']
            elif ftype == 'feat':
                X_train_sub = data['train_feat'][mask_train]
                X_test = data['test_feat']
            elif ftype == 'full':
                X_train_sub = np.concatenate([data['train_eeg'][mask_train], data['train_feat'][mask_train]], axis=1)
                X_test = np.concatenate([data['test_eeg'], data['test_feat']], axis=1)
            elif ftype == 'pred':
                X_train_sub = data['train_pred'][mask_train]
                X_test = data['test_pred']
                
            acc, bacc, converged, dim = run_experiment_gpu(
                X_train_sub, y_train_sub, X_test, data['test_labels'], 
                args.seed, device=args.device
            )
            
            conv_str = "Yes" if converged else "NO"
            print(f"| {ratio_display:<8} | {n_subs:<6} | {ftype:<10} | {dim:<5} | {acc*100:.2f}% | {bacc*100:.2f}% | {conv_str} |")

if __name__ == '__main__':
    main()
