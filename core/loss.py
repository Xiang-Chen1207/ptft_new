import torch.nn as nn
from typing import Dict, Any, Callable

class LossFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(loss_class):
            cls._registry[name] = loss_class
            return loss_class
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Loss function {name} not found in registry")
        return cls._registry[name](**kwargs)

@LossFactory.register("mse")
class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, target, mask=None, **kwargs):
        if mask is None:
            return self.criterion(pred, target)
        else:
            # Masked MSE Loss
            # pred, target: (B, C, N, P)
            # mask: (B, C, N) with 1 indicating masked patches (reconstruct these)
            
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1) # Mean over patch dimension -> (B, C, N)
            
            # Only compute loss on masked patches
            # mask is 0 or 1. If 1 means masked, we want loss * mask.
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            return loss

@LossFactory.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)

    def forward(self, pred, target, **kwargs):
        return self.criterion(pred, target)

@LossFactory.register("bce_with_logits")
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, pred, target, **kwargs):
        return self.criterion(pred, target.float())
