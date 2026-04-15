from __future__ import annotations
import json
from pathlib import Path

def plot_metrics(summary: dict[str, float], save_path: str | Path) -> None:
    """Generate visualization plots for evaluation metrics."""
    print(f"Plotting metrics to {save_path}...")
    # Add actual matplotlib/seaborn code as needed.

def export_qualitative_cases(records: list, save_path: str | Path) -> None:
    """Export error analysis cases for qualitative review."""
    print(f"Exporting qualitative cases to {save_path}...")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
