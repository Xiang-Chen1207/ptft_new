from torch.utils.data import DataLoader

def build_dataset(name, config, mode='train'):
    # config contains all keys from the 'dataset' section of yaml, including 'dataset_dir'
    if name == 'TUEG':
        from .tueg import TUEGDataset, get_tueg_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files using TUEG logic (95/5 for pretraining)
        file_list = get_tueg_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 10 files.")
            file_list = file_list[:10]
        elif mode == 'val' and config.get('limit_val', True):
            # For validation efficiency, limit to a small subset (Mini-Val)
            # 50 files is enough for metric stability (~few hundred samples)
            print(f"[{mode}] Limiting validation set to 50 files for efficiency.")
            file_list = file_list[:50]
        
        # 2. Initialize Dataset
        return TUEGDataset(file_list=file_list, **config)
    elif name == 'TUAB':
        from .tuab import TUABDataset, get_tuab_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files using reference logic
        file_list = get_tuab_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 10 files.")
            file_list = file_list[:10]
        
        # 2. Initialize Dataset
        return TUABDataset(file_list=file_list, **config)
    else:
        raise ValueError(f"Dataset {name} not supported")

def build_dataloader(name, config, mode='train'):
    dataset = build_dataset(name, config, mode)
    
    collate_fn = getattr(dataset, 'collate', None)
    
    # Shuffle for train AND val (as per cbramod_tuab reference for intra-epoch val)
    # Test usually remains unshuffle, but val should be shuffled for random sampling during intra-validation
    shuffle = (mode == 'train' or mode == 'val')
    
    return DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=shuffle,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn
    )
