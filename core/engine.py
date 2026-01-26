import torch
import torch.nn.functional as F
import math
import sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import utils.util as utils

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scheduler=None, max_norm=1.0, log_writer=None, val_loader=None, val_freq=50, task_type="classification", feature_loss_weight=0.0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        # Unpack batch
        if len(batch) == 2:
            x, y = batch
            mask = None
        else:
            x, y, mask = batch 

        x = x.to(device).float()
        
        # Handle features in y if present (for pretraining auxiliary task)
        target_features = None
        if task_type == "pretraining":
             if isinstance(y, torch.Tensor):
                 target_features = y.to(device)
                 # y is just features, not used for recon loss (x is target)
             else:
                 y = y.to(device)
        else:
            y = y.to(device)
        
        if mask is not None:
             mask = mask.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        if task_type == "pretraining":
             outputs = model(x, mask=mask)
             feature_pred = None
             
             if isinstance(outputs, tuple):
                 # Check if we have feature prediction (recon, mask, feat_pred)
                 if len(outputs) == 3:
                     outputs, gen_mask, feature_pred = outputs
                 else:
                     outputs, gen_mask = outputs
                 
                 if mask is None: mask = gen_mask
             
             # Initialize total loss
             loss = 0.0
             
             # Reconstruction loss
             if outputs is not None:
                 recon_loss = criterion(outputs, x, mask=mask)
                 loss = loss + recon_loss
                 metric_logger.update(recon_loss=recon_loss.item())
             
             # Feature Loss
             if feature_pred is not None and target_features is not None:
                 loss_feat = F.mse_loss(feature_pred, target_features)
                 loss = loss + feature_loss_weight * loss_feat
                 metric_logger.update(loss_feat=loss_feat.item())
                 
             # Lightweight Training Metrics (No Viz)
             with torch.no_grad():
                 if outputs is not None:
                     # Recon Correlation
                     # outputs: (B, C, N, P), x: (B, C, N, P)
                     # Flatten for correlation: (B, -1)
                     B = x.shape[0]
                     recon_flat = outputs.view(B, -1)
                     x_flat = x.view(B, -1)
                     metrics_recon = utils.calc_regression_metrics(recon_flat, x_flat)
                     metric_logger.update(recon_pcc=metrics_recon['pcc'])
                     metric_logger.update(recon_r2=metrics_recon['r2'])
                 
                 if feature_pred is not None and target_features is not None:
                     metrics_feat = utils.calc_regression_metrics(feature_pred, target_features)
                     metric_logger.update(feat_pcc=metrics_feat['pcc'])
                     metric_logger.update(feat_r2=metrics_feat['r2'])
        else:
            # Finetuning
            outputs = model(x)
            
            # --- Standardize Shapes for Binary Classification ---
            if task_type == "classification":
                # Ensure outputs is (B,) if binary (1 output unit)
                if outputs.ndim == 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                # Ensure y is (B,)
                if y.ndim == 2 and y.shape[1] == 1:
                    y = y.squeeze(1)
            
            loss = criterion(outputs, y)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Warning: Loss is {}".format(loss_value))
            # Continue or stop? labram stops on NaN usually, or just warns
            # sys.exit(1)

        loss.backward()
        
        if max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_norm = 0.0
            
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Metrics logging
        metric_logger.update(loss=loss_value)
        metric_logger.update(grad_norm=grad_norm)
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        
        # Calculate training accuracy for classification tasks if possible
        if mask is None and task_type == "classification": # Finetuning
            with torch.no_grad():
                # Check for binary classification based on output shape (now standardized to 1D or 2D > 1 col)
                is_binary = outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1)
                
                if is_binary:
                    # Both outputs and y are (B,)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    # Calculate accuracy on GPU
                    acc = (preds == y).float().mean()
                    
                    # Calculate balanced accuracy (needs CPU numpy)
                    balanced_acc = balanced_accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
                    
                else:
                    # Multiclass
                    preds_cls = outputs.max(-1)[-1]
                    acc = (preds_cls == y).float().mean()
                    
                    # Calculate balanced accuracy
                    balanced_acc = balanced_accuracy_score(y.cpu().numpy(), preds_cls.cpu().numpy())
                    
                metric_logger.update(acc=acc.item())
                metric_logger.update(balanced_acc=balanced_acc)
        
        if log_writer is not None:
            log_data = {
                'train_loss': loss_value,
                'lr': max_lr,
                'grad_norm': grad_norm,
                'epoch': epoch,
                'step': epoch * len(dataloader) + data_iter_step
            }
            # Dynamically add all other metrics tracked in metric_logger
            # Use .median or .avg for smoothing
            for k, meter in metric_logger.meters.items():
                if k not in ['loss']: # loss already added as train_loss
                    log_data[f'train_{k}'] = meter.median 
            
            log_writer.log(log_data)
            
        # Intra-epoch Validation
        if val_loader is not None and val_freq > 0 and (data_iter_step + 1) % val_freq == 0:
             print(f"Intra-epoch Validation at step {data_iter_step + 1}")
             # Use verbose=False to suppress intermediate logs
             evaluate(model, val_loader, criterion, device, task_type=task_type, log_writer=log_writer, epoch=epoch, header='Val (Intra):', log_prefix='val_intra', limit_batches=50, verbose=False)
             model.train(True)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, task_type="classification", log_writer=None, epoch=None, header='Test:', log_prefix=None, limit_batches=None, verbose=True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    preds = []
    targets = []
    
    last_batch_viz = None # Initialize to avoid scope issues
    
    # Use log_every or simple iterator depending on context? 
    if verbose:
        iterator = metric_logger.log_every(dataloader, 50, header)
    else:
        iterator = dataloader
    
    # Check if dataloader is empty
    if len(dataloader) == 0:
        print(f"WARNING: {header} Dataloader is empty!")
        return {}

    # Feature Names for Top-K Logging
    feature_names = None
    if hasattr(dataloader.dataset, 'feature_names'):
        feature_names = dataloader.dataset.feature_names
    elif hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset.dataset, 'feature_names'):
        # Handle Subset wrapper if present
        feature_names = dataloader.dataset.dataset.feature_names

    # Accumulators for Feature R2
    # We want to average R2 across batches. 
    # NOTE: Averaging R2 is an approximation. Correct is global R2. 
    # But for monitoring, average is okay.
    acc_r2_per_channel = None
    num_batches = 0

    for i, batch in enumerate(iterator):
        if limit_batches is not None and i >= limit_batches:
            break
            
        if len(batch) == 2:
            x, y = batch
            mask = None
        else:
            x, y, mask = batch
            
        x = x.to(device).float()
        
        # Handle features in y if present (for pretraining auxiliary task)
        target_features = None
        if task_type == "pretraining":
             if isinstance(y, torch.Tensor):
                 target_features = y.to(device)
             else:
                 y = y.to(device)
        else:
            y = y.to(device)
            
        if mask is not None:
             mask = mask.to(device)
        
        # Forward pass
        if task_type == "pretraining":
             outputs = model(x, mask=mask)
             feature_pred = None
             
             if isinstance(outputs, tuple):
                 if len(outputs) == 3:
                     outputs, gen_mask, feature_pred = outputs
                 else:
                     outputs, gen_mask = outputs
                 if mask is None: mask = gen_mask
             
             loss = 0.0
             if outputs is not None:
                loss = loss + criterion(outputs, x, mask=mask)
             
             # Calculate Pretraining Metrics
             with torch.no_grad():
                 if outputs is not None:
                     B = x.shape[0]
                     metrics_recon = utils.calc_regression_metrics(outputs.view(B, -1), x.view(B, -1))
                     metric_logger.update(recon_pcc=metrics_recon['pcc'])
                     metric_logger.update(recon_r2=metrics_recon['r2'])
                 
                 if feature_pred is not None and target_features is not None:
                     loss_feat = F.mse_loss(feature_pred, target_features)
                     loss = loss + 0.0 * loss_feat # Just to track loss, not backward
                     metrics_feat = utils.calc_regression_metrics(feature_pred, target_features)
                     metric_logger.update(feat_pcc=metrics_feat['pcc'])
                     metric_logger.update(feat_r2=metrics_feat['r2'])
                     metric_logger.update(loss_feat=loss_feat.item())
                     
                     # Accumulate per-channel R2
                     r2_pc = metrics_feat['r2_per_channel'] # (D,)
                     if acc_r2_per_channel is None:
                         acc_r2_per_channel = torch.zeros_like(r2_pc)
                     acc_r2_per_channel += r2_pc
                     num_batches += 1
                     
             # --- Rich Visualization (Last Batch) ---
             # We store the last batch data for visualization after loop
             last_batch_viz = {
                 'x': x,
                 'x_hat': outputs,
                 'mask': mask
             }
             # print(f"Stored viz data for step {i}") # Debug print
        else:
             outputs = model(x)
             
             # --- Standardize Shapes for Binary Classification ---
             if task_type == "classification":
                 if outputs.ndim == 2 and outputs.shape[1] == 1:
                     outputs = outputs.squeeze(1)
                 if y.ndim == 2 and y.shape[1] == 1:
                     y = y.squeeze(1)
             
             loss = criterion(outputs, y)
        
        metric_logger.update(loss=loss.item())
        
        # Only collect preds/targets for classification
        if task_type == "classification":
            preds.append(outputs.cpu())
            targets.append(y.cpu())
            
            # Batch accuracy
            is_binary = outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1)
            if is_binary:
                 p = (torch.sigmoid(outputs) > 0.5).float()
                 acc = (p == y).float().mean()
            else:
                 acc = (outputs.max(-1)[-1] == y).float().mean()
            metric_logger.meters['acc'].update(acc.item(), n=x.shape[0])

    # Synchronize
    metric_logger.synchronize_between_processes()
    if verbose:
        print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    
    metrics = {
        "loss": metric_logger.loss.global_avg,
    }
    if hasattr(metric_logger, 'acc'):
         metrics["acc"] = metric_logger.acc.global_avg
    if hasattr(metric_logger, 'recon_pcc'):
         metrics["recon_pcc"] = metric_logger.recon_pcc.global_avg
    if hasattr(metric_logger, 'feat_r2'):
         metrics["feat_r2"] = metric_logger.feat_r2.global_avg
    
    # Top-K Feature Logging
    if acc_r2_per_channel is not None and num_batches > 0:
        avg_r2_per_channel = acc_r2_per_channel / num_batches
        # Get Top 5
        k = min(5, len(avg_r2_per_channel))
        topk_vals, topk_inds = torch.topk(avg_r2_per_channel, k)
        
        topk_dict = {}
        for idx, val in zip(topk_inds, topk_vals):
            fname = feature_names[idx.item()] if feature_names else f"Feat_{idx.item()}"
            topk_dict[fname] = val.item()
            # Add to wandb metrics if needed, e.g. "feat_top1_r2"
        
        print(f"Top-{k} Predicted Features: {topk_dict}")
        
        # Log to metric_logger (and thus WandB)
        # Strategy:
        # 1. Log Scalars for Ranks (e.g. Rank 1 R2) to track best performance
        # 2. Log Table for detailed Name/R2 mapping
        import wandb
        
        # 1. Scalars
        for i, val in enumerate(topk_vals):
            metrics[f"feat_rank{i+1}_r2"] = val.item()
            
        # 2. Table
        if log_writer is not None:
            # Create a table with columns: Rank, Feature Name, R2
            table_data = []
            for i, (idx, val) in enumerate(zip(topk_inds, topk_vals)):
                fname = feature_names[idx.item()] if feature_names else f"Feat_{idx.item()}"
                table_data.append([i+1, fname, val.item()])
            
            # We can't put Table in 'metrics' directly if 'metrics' is used for other things
            # But we can put it in a special key that log_writer handles, or just log it here?
            # Wait, if we log here, we need 'step'. 'evaluate' has 'epoch'.
            # Better to put it in 'metrics' and let the final loop handle it.
            metrics["top_features_table"] = wandb.Table(data=table_data, columns=["Rank", "Feature", "R2"])

    # Pretraining Visualization (WandB)
    if task_type == "pretraining" and log_writer is not None:
        if last_batch_viz is not None:
            try:
                print(f"Generating visualization for {header}...")
                fig = utils.visualize_eeg_batch(
                    last_batch_viz['x'], 
                    last_batch_viz['x_hat'], 
                    last_batch_viz['mask']
                )
                
                # Unified Logging: Put Image in metrics
                import wandb
                metrics["reconstruction_viz"] = wandb.Image(fig)
                
            except Exception as e:
                print(f"Visualization failed: {e}")
        else:
            print(f"WARNING: last_batch_viz is None! Validation loop might have been skipped or failed to store data.")

    if task_type == "classification":
        # Global metrics
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        
        # Check binary vs multiclass
        is_binary = preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1)
        
        if is_binary: # Binary
            # Standardize to 1D numpy
            if preds.ndim == 2: preds = preds.squeeze(1)
            if targets.ndim == 2: targets = targets.squeeze(1)
            
            pred_labels = (torch.sigmoid(preds) > 0.5).long().numpy()
            probs = torch.sigmoid(preds).numpy()
            targets_np = targets.numpy()
            
            try:
                metrics["auc"] = roc_auc_score(targets_np, probs)
            except:
                metrics["auc"] = 0.0
            
            # Confusion Matrix & Balanced Acc
            cm = confusion_matrix(targets_np, pred_labels)
            balanced_acc = balanced_accuracy_score(targets_np, pred_labels)
            metrics["balanced_acc"] = balanced_acc
            
            if verbose:
                print(f'{header} balanced accuracy {balanced_acc:.3f}')
                print(f'{header} Confusion Matrix:\n{cm}')
            
        else: # Multiclass
            pred_labels = torch.argmax(preds, dim=1).numpy()
            targets_np = targets.numpy()
            
            metrics["kappa"] = cohen_kappa_score(targets_np, pred_labels)
            metrics["f1"] = f1_score(targets_np, pred_labels, average='weighted')
            
            balanced_acc = balanced_accuracy_score(targets_np, pred_labels)
            metrics["balanced_acc"] = balanced_acc
            if verbose:
                print(f'{header} balanced accuracy {balanced_acc:.3f}')

    if log_writer is not None and epoch is not None:
        if log_prefix is not None:
            prefix = log_prefix
        else:
            prefix = "val" if "Val" in header else "test"
            
        log_data = {'epoch': epoch}
        # Log all metrics dynamically
        for k, v in metrics.items():
            # Handle potential naming conflicts or cleaner grouping
            # If k already starts with prefix, don't double it
            key = f"{prefix}/{k}" if not k.startswith(f"{prefix}") else k
            log_data[key] = v
            
        log_writer.log(log_data)
            
    return metrics
