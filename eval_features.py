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
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name (e.g. TUEG, TUAB)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on (train/val/test)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config['dataset']['name'] = args.dataset
        print(f"Overriding dataset to {args.dataset}")

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
    print(f"Building dataloader for split: {args.split}...")
    # Disable validation limit for full evaluation
    config['dataset']['limit_val'] = False
    val_loader = build_dataloader(config['dataset']['name'], config['dataset'], mode=args.split)
    
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
    print("Generating Scatter Plots and Radar Charts...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from math import pi

    output_dir = os.path.dirname(args.output)
    if not output_dir:
        output_dir = "."

    # Create a unique visualization directory based on the output CSV filename
    output_stem = os.path.splitext(os.path.basename(args.output))[0]
    viz_base_dir = os.path.join(output_dir, f"{output_stem}_viz")
    
    scatter_dir = os.path.join(viz_base_dir, "scatter_plots")
    os.makedirs(scatter_dir, exist_ok=True)
    
    # Concatenate all data
    all_preds_tensor = torch.cat(all_preds, dim=0) # (N, D)
    all_targets_tensor = torch.cat(all_targets, dim=0) # (N, D)
    
    # Calculate per-sample MSE to find best/worst examples
    # (N, D) -> (N,)
    per_sample_mse = torch.mean((all_preds_tensor - all_targets_tensor)**2, dim=1)
    
    # Select indices: Best 3, Median 1, Worst 1
    sorted_indices = torch.argsort(per_sample_mse)
    n_samples = len(sorted_indices)
    
    best_indices = sorted_indices[:3].cpu().numpy()
    median_index = sorted_indices[n_samples // 2].cpu().numpy()
    worst_index = sorted_indices[-1].cpu().numpy()
    
    indices_to_plot = np.concatenate([best_indices, [median_index]])
    labels_to_plot = ["Best #1", "Best #2", "Best #3", "Median"]
    
    # Radar Chart Function
    def plot_radar_chart(categories, values_list, labels_list, title, filename):
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        # plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
        # plt.ylim(0, 40)
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        
        for i, (values, label) in enumerate(zip(values_list, labels_list)):
            values = list(values)
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=label, color=colors[i % len(colors)])
            ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
            
        plt.title(title, size=20, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

    # Normalize data for radar chart visualization?
    # Z-scores are roughly -3 to 3. Shift them to be positive for radar chart area?
    # Or just use raw values and set appropriate limits. 
    # Radar charts with negative values are tricky. 
    # Strategy: Min-Max normalize each feature based on the GLOBAL range of that feature in the dataset.
    # This preserves relative magnitude.
    
    all_preds_np = all_preds_tensor.numpy()
    all_targets_np = all_targets_tensor.numpy()
    
    feat_min = all_targets_np.min(axis=0)
    feat_max = all_targets_np.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1e-6
    
    # We will plot separate charts for different feature groups to avoid overcrowding
    # Based on feature names
    if feature_names:
        groups = {
            "Time & Complexity": [],
            "Spectral Power": [],
            "Relative Power": [],
            "Spectral Features": []
        }
        
        for idx, name in enumerate(feature_names):
            if "relative_power" in name:
                groups["Relative Power"].append(idx)
            elif "power" in name and "relative" not in name and "ratio" not in name:
                 groups["Spectral Power"].append(idx)
            elif "spectral" in name or "frequency" in name or "ratio" in name or "exponent" in name:
                groups["Spectral Features"].append(idx)
            else:
                groups["Time & Complexity"].append(idx)
                
        # Generate Radar Charts for selected samples
        radar_dir = os.path.join(viz_base_dir, "radar_charts")
        os.makedirs(radar_dir, exist_ok=True)
        
        for i, idx in enumerate(indices_to_plot):
            sample_label = labels_to_plot[i]
            sample_mse = per_sample_mse[idx].item()
            
            # Normalize this sample's target and pred
            # But wait, we want to see if Pred matches Target.
            # We should normalize BOTH using the Global Min/Max of Target (to keep scale consistent)
            
            target_raw = all_targets_np[idx]
            pred_raw = all_preds_np[idx]
            
            # Plot for each group
            for group_name, feat_indices in groups.items():
                if not feat_indices:
                    continue
                    
                cats = [feature_names[fi] for fi in feat_indices]
                # Shorten names for display
                short_cats = [c.replace('_mean', '').replace('_std', ' (std)').replace('mean_', '').replace('spectral_', '').replace('_power', '') for c in cats]
                
                # Extract values
                t_vals = target_raw[feat_indices]
                p_vals = pred_raw[feat_indices]
                
                # Normalize to [0, 1] for visualization using global min/max
                mins = feat_min[feat_indices]
                maxs = feat_max[feat_indices]
                ranges = feat_range[feat_indices]
                
                t_norm = (t_vals - mins) / ranges
                p_norm = (p_vals - mins) / ranges
                
                # Clip to [0, 1] just in case pred is out of bound
                p_norm = np.clip(p_norm, 0, 1.2) # Allow slight overshoot
                
                # Also include the mean of all targets as a reference baseline?
                # Maybe too cluttered. Just True vs Pred.
                
                fname = f"radar_{sample_label.replace(' ', '_')}_{group_name.replace(' ', '_')}.png"
                fpath = os.path.join(radar_dir, fname)
                
                plot_radar_chart(
                    short_cats, 
                    [t_norm, p_norm], 
                    ["Ground Truth", "Prediction"], 
                    f"{sample_label} (MSE={sample_mse:.2f}) - {group_name}",
                    fpath
                )
        
        print(f"Radar charts saved to {radar_dir}/")

    # Scatter Plots Logic (Keep existing)
    all_preds = all_preds_np
    all_targets = all_targets_np
    
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
