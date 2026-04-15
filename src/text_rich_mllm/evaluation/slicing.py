from __future__ import annotations

from collections import defaultdict

from text_rich_mllm.schemas import PredictionRecord


def _aggregate(records: list[PredictionRecord], key_fn) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        if key in {"", None}:
            continue
        grouped[str(key)].append(record)
    return {
        key: {
            "count": len(items),
            "mean_score": sum(item.score for item in items) / len(items),
        }
        for key, items in grouped.items()
    }


def build_sliced_summary(
    records: list[PredictionRecord],
    *,
    metadata_keys: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    summary = {
        "by_dataset": _aggregate(records, lambda record: record.dataset_name),
        "by_task_type": _aggregate(records, lambda record: record.task_type),
        "by_answer_type": _aggregate(records, lambda record: record.answer_type),
        "by_split": _aggregate(records, lambda record: record.split),
        "by_error_type": _aggregate(records, lambda record: record.error_type),
    }
    for metadata_key in metadata_keys or []:
        summary[f"by_metadata.{metadata_key}"] = _aggregate(
            records,
            lambda record, key=metadata_key: record.metadata.get(key),
        )
    return summary
