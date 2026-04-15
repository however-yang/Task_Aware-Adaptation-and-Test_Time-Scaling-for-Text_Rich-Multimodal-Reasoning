from __future__ import annotations
from pathlib import Path
import json

class CheckpointManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model, processor, step: int, metrics: dict):
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        model.save_pretrained(ckpt_dir)
        try:
            processor.save_pretrained(ckpt_dir)
        except AttributeError:
            pass # Handle processor without save_pretrained if any
        
        # Save metrics
        with open(ckpt_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Saved checkpoint to {ckpt_dir}")
        return ckpt_dir
