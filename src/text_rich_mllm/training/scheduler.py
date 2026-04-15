from __future__ import annotations
from transformers import get_scheduler

def create_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int, scheduler_type: str = "cosine"):
    """Create a learning rate scheduler for training."""
    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
