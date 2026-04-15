from __future__ import annotations
import torch

def create_optimizer(model, learning_rate: float, weight_decay: float = 0.01):
    """Create an optimizer for training."""
    # Typically we use AdamW for PEFT/Transformers
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
