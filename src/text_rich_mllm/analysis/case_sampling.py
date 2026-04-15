from __future__ import annotations

from collections import defaultdict

from text_rich_mllm.schemas import PredictionRecord


def sample_cases(records: list[PredictionRecord], *, limit_per_error: int = 5) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        error_type = record.error_type or "unknown"
        if len(grouped[error_type]) >= limit_per_error:
            continue
        grouped[error_type].append(record.to_dict())
    return dict(grouped)
