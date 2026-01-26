import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
import os
from tqdm import tqdm
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.wrapper import CBraModWrapper
from datasets.builder import build_dataloader
import utils.util as utils

def main():
    parser = argparse.ArgumentParser(description="Evaluate Feature Prediction Metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--output", type=str, default="feature_metrics.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size

    # Ensure task type is pretraining
    config['task_type'] = 'pretraining'
    config['model']['task_type'] = 'pretraining'
    
    # Load Model
    print("Building model...")
    model = CBraModWrapper(config)
    model.to(args.device)
    model.eval()

    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Handle state dict keys if wrapped in 'model' or DDP
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove DDP prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)

    # Build Validation Loader
    print("Building validation dataloader...")
    # Disable validation limit for full evaluation
    config['dataset']['limit_val'] = False
    val_loader = build_dataloader(config['dataset']['name'], config['dataset'], mode='val')
    
    # Feature Names
    feature_names = None
    if hasattr(val_loader.dataset, 'feature_names'):
        feature_names = val_loader.dataset.feature_names
    elif hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'feature_names'):
         feature_names = val_loader.dataset.dataset.feature_names
    
    if feature_names is None:
        print("Warning: Could not find feature names in dataset. Using indices.")

    # Prepare for Scatter Plots (Raw Data Collection)
    # To avoid OOM with 28k samples, we can subsample or just store all (28k * 66 * 4 bytes ~ 7MB, very small)
    all_preds = []
    all_targets = []

    # Evaluation Loop
    print("Starting evaluation...")
    
    # Accumulators
    mse_accum = None
    var_accum = None
    count = 0
    
    # For R2: 1 - SSE/SST
    # SSE = sum((pred - target)^2)
    # SST = sum((target - mean)^2) -> This requires global mean.
    # Alternative R2 calculation: 
    # Use online Welford algorithm or just sum(target) and sum(target^2) to get variance?
    # Simpler: Accumulate sum_squared_error and sum_squared_total (using batch mean approximation or 2-pass?)
    # 2-pass is expensive. 
    # Let's use the implementation in utils.calc_regression_metrics which does batch-wise averaging?
    # No, batch-wise averaging of R2 is not mathematically correct for global R2.
    # Correct way: Accumulate SSE and SST globally.
    # SST = sum((y - y_global_mean)^2). 
    # We need y_global_mean first. 
    # Approximation: Use batch mean as proxy? No, that's bad.
    # Better: Accumulate sum(y) and sum(y^2) to compute global variance later?
    # Var(y) = E[y^2] - (E[y])^2
    # SST = N * Var(y)
    
    sum_y = None
    sum_y_sq = None
    sum_sq_err = None
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if len(batch) == 2:
                x, y = batch
                mask = None
            else:
                x, y, mask = batch
            
            x = x.to(args.device).float()
            
            # y contains features in pretraining
            if isinstance(y, torch.Tensor):
                target_features = y.to(args.device).float()
            else:
                # Should not happen in this codebase for pretraining
                target_features = y.to(args.device).float()

            if mask is not None:
                mask = mask.to(args.device)

            # Forward
            # Explicitly create a zero-mask (no masking) for full signal inference
            # Mask shape should be (B, C, N) with all zeros
            # We need patch count N. 
            # x shape: (B, C, N, P) or similar? 
            # Let's check wrapper.py: _generate_mask uses x.shape -> B, C, N, P
            if x.ndim == 4:
                B, C, N, P = x.shape
                zero_mask = torch.zeros((B, C, N), device=x.device, dtype=torch.long)
            elif x.ndim == 3: # Maybe (B, N, D) or something?
                # wrapper says: B, C, N, P = x.shape. So assume 4D.
                B, C, N = x.shape
                zero_mask = torch.zeros((B, C, N), device=x.device, dtype=torch.long)
            else:
                 # Fallback if shape is weird, let model generate it? No, we want zero mask.
                 # Just pass None and hope? No, wrapper generates 50% mask if None.
                 # We MUST generate zero mask.
                 # Let's trust wrapper's expectation of 4D input for now.
                 if x.ndim == 4:
                     B, C, N, P = x.shape
                     zero_mask = torch.zeros((B, C, N), device=x.device, dtype=torch.long)
                 else:
                     # Attempt to construct valid mask
                     # Try to match shape of mask if it was passed by loader
                     if mask is not None:
                         zero_mask = torch.zeros_like(mask)
                     else:
                         # Risky fallback
                         print("Warning: Input x is not 4D and no mask provided. Cannot infer N.")
                         zero_mask = None # This will trigger random mask in model

            # model(x, mask) -> out, mask, feature_pred
            outputs = model(x, mask=zero_mask)
            
            feature_pred = None
            if isinstance(outputs, tuple):
                 if len(outputs) == 3:
                     _, _, feature_pred = outputs
                 elif len(outputs) == 2:
                     _, _ = outputs # No feature pred?
            
            if feature_pred is None:
                print("Error: Model did not return feature predictions. Check config 'pretrain_tasks'.")
                return

            # Initialize accumulators
            B, D = feature_pred.shape
            if sum_sq_err is None:
                sum_sq_err = torch.zeros(D, device=args.device)
                sum_y = torch.zeros(D, device=args.device)
                sum_y_sq = torch.zeros(D, device=args.device)
            
            # Accumulate stats
            # SSE
            diff = feature_pred - target_features
            sum_sq_err += torch.sum(diff ** 2, dim=0)
            
            # Stats for SST
            sum_y += torch.sum(target_features, dim=0)
            sum_y_sq += torch.sum(target_features ** 2, dim=0)
            
            # Stats for PCC: Need E[xy], E[x], E[y], E[x^2], E[y^2]
            # We already have E[y] (sum_y) and E[y^2] (sum_y_sq)
            # Need sum_x, sum_x_sq, sum_xy
            if 'sum_x' not in locals():
                sum_x = torch.zeros(D, device=args.device)
                sum_x_sq = torch.zeros(D, device=args.device)
                sum_xy = torch.zeros(D, device=args.device)
            
            sum_x += torch.sum(feature_pred, dim=0)
            sum_x_sq += torch.sum(feature_pred ** 2, dim=0)
            sum_xy += torch.sum(feature_pred * target_features, dim=0)
            
            count += B
            
            # Store raw data for scatter plots (CPU to save GPU mem)
            all_preds.append(feature_pred.cpu())
            all_targets.append(target_features.cpu())
            
    # Compute Metrics
    # MSE = SSE / N
    mse_per_channel = sum_sq_err / count
    rmse_per_channel = torch.sqrt(mse_per_channel)
    
    # Global Mean
    mean_y = sum_y / count
    mean_x = sum_x / count
    
    # SST = sum(y^2) - 2*mean*sum(y) + N*mean^2
    #     = sum_y_sq - count*mean_y^2 (Simplified)
    sst_per_channel = sum_y_sq - count * (mean_y ** 2)
    
    # R2 = 1 - SSE / SST
    # Handle division by zero (constant features)
    valid_mask = sst_per_channel > 1e-6
    r2_per_channel = torch.zeros_like(sst_per_channel)
    r2_per_channel[valid_mask] = 1 - (sum_sq_err[valid_mask] / sst_per_channel[valid_mask])
    
    # PCC Calculation
    # PCC = (E[xy] - E[x]E[y]) / (std_x * std_y)
    # std_x = sqrt(E[x^2] - (E[x])^2)
    
    # Numerator: N * Cov(x,y) = sum_xy - N * mean_x * mean_y
    numerator = sum_xy - count * mean_x * mean_y
    
    # Denominator
    var_x_n = sum_x_sq - count * (mean_x ** 2)
    var_y_n = sum_y_sq - count * (mean_y ** 2) # This is SST
    
    denominator = torch.sqrt(var_x_n * var_y_n)
    
    pcc_per_channel = torch.zeros_like(r2_per_channel)
    # Avoid div by zero
    valid_pcc = denominator > 1e-8
    pcc_per_channel[valid_pcc] = numerator[valid_pcc] / denominator[valid_pcc]

    # Prepare DataFrame
    metrics_data = {
        'Feature Index': range(len(r2_per_channel)),
        'R2': r2_per_channel.cpu().numpy(),
        'PCC': pcc_per_channel.cpu().numpy(),
        'RMSE': rmse_per_channel.cpu().numpy(),
        'MSE': mse_per_channel.cpu().numpy()
    }
    
    if feature_names:
        # Ensure length matches
        if len(feature_names) == len(r2_per_channel):
            metrics_data['Feature Name'] = feature_names
        else:
            print(f"Warning: Feature names count ({len(feature_names)}) != output dim ({len(r2_per_channel)})")
            
    df = pd.DataFrame(metrics_data)
    
    # Reorder columns if Name exists
    if 'Feature Name' in df.columns:
        cols = ['Feature Index', 'Feature Name', 'R2', 'PCC', 'RMSE', 'MSE']
        df = df[cols]
        
    # Sort by R2 descending
    df = df.sort_values('R2', ascending=False)
    
    print(f"Saving metrics to {args.output}")
    df.to_csv(args.output, index=False)
    
    print("Top 10 Features by R2:")
    print(df.head(10))

    # --- Generate Scatter Plots ---
    print("Generating Scatter Plots...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    scatter_dir = "feature_scatter_plots"
    os.makedirs(scatter_dir, exist_ok=True)
    
    # Concatenate all data
    all_preds = torch.cat(all_preds, dim=0).numpy() # (N, D)
    all_targets = torch.cat(all_targets, dim=0).numpy() # (N, D)
    
    # Subsample for plotting if too large (e.g. max 5000 points)
    num_samples = all_preds.shape[0]
    if num_samples > 5000:
        indices = np.random.choice(num_samples, 5000, replace=False)
        plot_preds = all_preds[indices]
        plot_targets = all_targets[indices]
    else:
        plot_preds = all_preds
        plot_targets = all_targets
        
    num_features = all_preds.shape[1]
    
    for i in tqdm(range(num_features), desc="Plotting"):
        fname = feature_names[i] if feature_names else f"Feature {i}"
        safe_fname = fname.replace('/', '_').replace(' ', '_')
        
        # Get metrics for title
        r2_val = r2_per_channel[i].item()
        pcc_val = pcc_per_channel[i].item()
        
        plt.figure(figsize=(6, 6))
        # Scatter with alpha
        plt.scatter(plot_targets[:, i], plot_preds[:, i], alpha=0.3, s=10, edgecolors='none')
        
        # Identity line
        min_val = min(plot_targets[:, i].min(), plot_preds[:, i].min())
        max_val = max(plot_targets[:, i].max(), plot_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        plt.title(f"{fname}\nR2={r2_val:.3f}, r={pcc_val:.3f}")
        plt.xlabel("True Value (Z-scored)")
        plt.ylabel("Predicted Value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(scatter_dir, f"{safe_fname}.png"), dpi=150)
        plt.close()
        
    print(f"Scatter plots saved to {scatter_dir}/")

if __name__ == "__main__":
    main()
