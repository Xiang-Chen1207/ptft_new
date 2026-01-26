import torch
import sys

ckpt_path = "/vePFS-0x0d/home/chen/ptft/output/flagship_cross_attn/best.pth"

try:
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("\n--- Configuration Found ---")
        
        # Check Task Type
        task_type = config.get('task_type', 'Unknown')
        print(f"Task Type: {task_type}")
        
        # Check Model Config
        if 'model' in config:
            model_config = config['model']
            pretrain_tasks = model_config.get('pretrain_tasks', [])
            print(f"Pretrain Tasks: {pretrain_tasks}")
            
            feature_token_type = model_config.get('feature_token_type', 'N/A')
            print(f"Feature Token Type: {feature_token_type}")
            
            mask_ratio = config.get('dataset', {}).get('mask_ratio', 'Not in dataset config')
            # Check model config for mask ratio too if not in dataset
            if mask_ratio == 'Not in dataset config':
                 mask_ratio = model_config.get('mask_ratio', 'Default (0.5 in code?)')
            print(f"Mask Ratio: {mask_ratio}")
            
        else:
            print("No 'model' section in config.")
            
    else:
        print("No 'config' key in checkpoint.")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
