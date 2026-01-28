import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
import logging

# Add project root to path
import sys
sys.path.append(os.getcwd())

from models.wrapper import CBraModWrapper
from datasets.tuep import TUEPDataset, get_tuep_file_list

# Define load_config directly to resolve import error
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Define setup_logger directly to resolve import error
def setup_logger(output_dir, log_filename='training.log'):
    # Remove existing handlers to avoid duplicate logs when switching models
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, log_filename)),
            logging.StreamHandler()
        ]
    )

def load_pretrained_model(config, weights_path, device):
    # Initialize model wrapper
    # Note: Full fine-tuning uses the classification head
    # The config should specify num_classes, etc.
    model = CBraModWrapper(config)
    
    # Load backbone weights
    print(f"Loading pretrained weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        # Filter state dict to load only backbone or matching keys
        # For full FT, we generally load the backbone. 
        # If the keys match, we load them.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            new_state_dict[k] = v
            
        # Load strictly=False because the classification head might be new or different
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Proceeding might be dangerous if weights are crucial, but let's allow it for debugging if needed
        # sys.exit(1)
        
    model.to(device)
    
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(dataloader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Check output type
        if isinstance(output, dict):
            logits = output.get('logits', output.get('cls_pred', None))
        else:
            logits = output
            
        if logits is None:
            raise ValueError("Model did not return logits!")
            
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            if isinstance(output, dict):
                logits = output.get('logits', output.get('cls_pred', None))
            else:
                logits = output
                
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    bacc = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return total_loss / len(dataloader), bacc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/finetune.yaml')
    parser.add_argument('--baseline_path', type=str, required=True)
    parser.add_argument('--flagship_path', type=str, required=True)
    parser.add_argument('--featonly_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset')
    parser.add_argument('--output_dir', type=str, default='experiments/tuep_full_ft/results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=float, nargs='+', 
                        default=[1.0])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=16)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load Dataset Index
    print("Loading Dataset Index...")
    # Load BOTH train and test to merge them for custom splitting
    train_files_orig = get_tuep_file_list(args.dataset_dir, 'train', seed=args.seed)
    test_files_orig = get_tuep_file_list(args.dataset_dir, 'test', seed=args.seed)
    
    all_files = train_files_orig + test_files_orig
    print(f"Total files loaded: {len(all_files)}")

    print("Grouping all files by subject...")
    # Directly extract subject_id from the file info dict
    all_file_subjects = [f['subject_id'] for f in all_files]
    unique_subjects = np.unique(all_file_subjects)
    total_subjects = len(unique_subjects)
    print(f"Total Unique Subjects (Train + Test): {total_subjects}")
    
    # Global Shuffle
    rng = np.random.RandomState(args.seed)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)
    
    # Load Config template
    base_config = load_config(args.config)
    # Ensure classification task
    base_config['task_type'] = 'classification'
    base_config['model']['num_classes'] = 2 # No Epilepsy vs With Epilepsy
    
    # Results storage
    results = []
    
    # Iterate Ratios
    for ratio in sorted(args.ratios):
        if ratio >= 1.0:
            n_subs = total_subjects
            current_subjects = shuffled_subjects
            ratio_display = "100%"
        else:
            n_subs = max(1, int(total_subjects * ratio))
            current_subjects = shuffled_subjects[:n_subs]
            ratio_display = f"{ratio*100:.1f}%"
            
        print(f"\n=== Ratio: {ratio_display} ({n_subs} Subjects) ===")
        
        # Split 80/10/10
        n_train = int(0.8 * n_subs)
        n_val = int(0.1 * n_subs)
        # Ensure at least 1 subject in each if possible, or handle edge cases
        if n_train == 0: n_train = 1
        # Remaining goes to test, but check val
        if n_val == 0 and n_subs > 1: n_val = 1
        
        # Adjust indices
        train_subs = current_subjects[:n_train]
        val_subs = current_subjects[n_train:n_train+n_val]
        test_subs = current_subjects[n_train+n_val:]
        
        # If dataset is too small, fallback (overlap or warn)
        # But assuming reasonable size here.
        
        print(f"Split Sizes (Subjects): Train={len(train_subs)}, Val={len(val_subs)}, Test={len(test_subs)}")
        
        # Create Subsets of files
        train_subs_set = set(train_subs)
        val_subs_set = set(val_subs)
        test_subs_set = set(test_subs)
        
        train_files_subset = [f for f in all_files if f['subject_id'] in train_subs_set]
        val_files_subset = [f for f in all_files if f['subject_id'] in val_subs_set]
        test_files_subset = [f for f in all_files if f['subject_id'] in test_subs_set]
        
        if len(train_files_subset) == 0:
            print("Warning: No train files selected!")
            continue

        train_dataset = TUEPDataset(train_files_subset)
        val_dataset = TUEPDataset(val_files_subset) if val_files_subset else None
        test_dataset = TUEPDataset(test_files_subset) if test_files_subset else None
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True, prefetch_factor=4, pin_memory=True) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True, prefetch_factor=4, pin_memory=True) if test_dataset else None
        
        # Define Models to Run
        models_to_run = [
            ('Baseline', args.baseline_path),
            ('Flagship', args.flagship_path),
            ('FeatOnly', args.featonly_path)
        ]
        
        for model_name, weights_path in models_to_run:
            print(f"--- Training {model_name} ---")
            
            # Setup logger for this model
            log_filename = f"training_{model_name.lower()}_{ratio_display.replace('%','')}.log"
            setup_logger(args.output_dir, log_filename)
            logging.info(f"Starting training for model: {model_name}, Ratio: {ratio_display}")
            
            # Reset Model for each run
            model = load_pretrained_model(base_config, weights_path, device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
            criterion = nn.CrossEntropyLoss()
            
            best_val_bacc = 0.0
            best_val_f1 = 0.0
            best_epoch = -1
            
            # Path to save best model
            best_model_path = os.path.join(args.output_dir, f"best_model_{model_name}_{ratio_display.replace('%','')}.pth")
            
            # Training Loop
            for epoch in range(args.epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                
                # Validate
                if val_loader:
                    val_loss, val_bacc, val_f1 = evaluate(model, val_loader, criterion, device)
                    
                    if val_bacc > best_val_bacc:
                        best_val_bacc = val_bacc
                        best_val_f1 = val_f1
                        best_epoch = epoch
                        # Save best model
                        torch.save(model.state_dict(), best_model_path)
                        
                    log_msg = f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} BAcc: {val_bacc:.4f} F1: {val_f1:.4f} (Best BAcc: {best_val_bacc:.4f})"
                else:
                    log_msg = f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | No Validation Set"
                    
                print(log_msg)
                logging.info(log_msg)
            
            # Testing
            final_test_bacc = 0.0
            final_test_f1 = 0.0
            
            if test_loader and os.path.exists(best_model_path):
                print(f"Loading best model from epoch {best_epoch+1} for testing...")
                model.load_state_dict(torch.load(best_model_path))
                test_loss, test_bacc, test_f1 = evaluate(model, test_loader, criterion, device)
                final_test_bacc = test_bacc
                final_test_f1 = test_f1
                print(f"TEST RESULTS: Loss: {test_loss:.4f} BAcc: {test_bacc:.4f} F1: {test_f1:.4f}")
                logging.info(f"TEST RESULTS: Loss: {test_loss:.4f} BAcc: {test_bacc:.4f} F1: {test_f1:.4f}")
            elif test_loader:
                print("Warning: No best model saved (maybe validation failed?), testing with last model.")
                test_loss, test_bacc, test_f1 = evaluate(model, test_loader, criterion, device)
                final_test_bacc = test_bacc
                final_test_f1 = test_f1
            else:
                print("No test set available.")

            results.append({
                'ratio': ratio_display,
                'n_subs': n_subs,
                'model': model_name,
                'val_bacc': best_val_bacc,
                'test_bacc': final_test_bacc,
                'test_f1': final_test_f1
            })
            
            print(f"Final Test BAcc for {model_name} @ {ratio_display}: {final_test_bacc*100:.2f}%, F1: {final_test_f1*100:.2f}%")

    # Generate Report
    print("\n=== Final Results Summary ===")
    print(f"| {'Ratio':<8} | {'NumSub':<6} | {'Model':<10} | {'Test BAcc':<10} | {'Test F1':<10} |")
    print(f"|{'-'*10}|{'-'*8}|{'-'*12}|{'-'*12}|{'-'*12}|")
    
    report_path = os.path.join(args.output_dir, 'results_final_ft.md')
    with open(report_path, 'w') as f:
        f.write("# TUEP Full Fine-tuning Comparative Results (80/10/10 Split)\n\n")
        f.write(f"| Ratio | NumSub | Model | Test BAcc (%) | Test F1 (%) | Val Best BAcc (%) |\n")
        f.write(f"|-------|--------|-------|---------------|-------------|-------------------|\n")
        
        for r in results:
            line = f"| {r['ratio']} | {r['n_subs']} | {r['model']} | {r['test_bacc']*100:.2f} | {r['test_f1']*100:.2f} | {r['val_bacc']*100:.2f} |"
            print(f"| {r['ratio']:<8} | {r['n_subs']:<6} | {r['model']:<10} | {r['test_bacc']*100:.2f}%    | {r['test_f1']*100:.2f}%    |")
            f.write(line + "\n")
            
    print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()
