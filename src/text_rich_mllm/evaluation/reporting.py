from __future__ import annotations

from text_rich_mllm.evaluation.slicing import build_sliced_summary
from text_rich_mllm.schemas import PredictionRecord


def build_evaluation_report(
    records: list[PredictionRecord],
    metric_summary: dict[str, float],
    *,
    metadata_keys: list[str] | None = None,
) -> dict:
    report = dict(metric_summary)
    report["num_predictions"] = len(records)
    report["slices"] = build_sliced_summary(records, metadata_keys=metadata_keys)
    return report
