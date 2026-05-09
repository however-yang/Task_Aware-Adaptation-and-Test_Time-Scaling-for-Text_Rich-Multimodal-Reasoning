from __future__ import annotations

from collections import deque
from itertools import zip_longest
from math import sqrt

from text_rich_mllm.schemas import UnifiedSample


def group_samples_by_dataset(samples: list[UnifiedSample]) -> dict[str, list[UnifiedSample]]:
    grouped: dict[str, list[UnifiedSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.dataset_name, []).append(sample)
    return grouped


def balanced_interleave(samples: list[UnifiedSample]) -> list[UnifiedSample]:
    grouped = group_samples_by_dataset(samples)
    queues = {name: deque(items) for name, items in grouped.items()}
    mixed: list[UnifiedSample] = []
    while queues:
        exhausted = []
        for name, queue in queues.items():
            if queue:
                mixed.append(queue.popleft())
            if not queue:
                exhausted.append(name)
        for name in exhausted:
            queues.pop(name, None)
    return mixed


def square_root_interleave(samples: list[UnifiedSample]) -> list[UnifiedSample]:
    grouped = group_samples_by_dataset(samples)
    weighted: list[tuple[str, int]] = []
    for name, items in grouped.items():
        repeat = max(1, round(sqrt(len(items))))
        weighted.append((name, repeat))
    expanded_groups: list[list[UnifiedSample]] = []
    for name, repeat in weighted:
        # 关键修复：每次生成独立副本，而非 [list_ref] * repeat（引用复制）
        # 若用引用复制，zip_longest 会并行遍历同一对象，square_root 加权失效
        for _ in range(repeat):
            expanded_groups.append(list(grouped[name]))

    mixed: list[UnifiedSample] = []
    for chunk in zip_longest(*expanded_groups):
        for sample in chunk:
            if sample is not None:
                mixed.append(sample)
    seen: set[str] = set()
    deduped: list[UnifiedSample] = []
    for sample in mixed:
        if sample.sample_id in seen:
            continue
        seen.add(sample.sample_id)
        deduped.append(sample)
    return deduped


def mix_training_samples(samples: list[UnifiedSample], *, strategy: str = "balanced") -> list[UnifiedSample]:
    if strategy == "balanced":
        return balanced_interleave(samples)
    if strategy in {"sqrt", "square_root"}:
        return square_root_interleave(samples)
    if strategy in {"sequential", "none"}:
        return list(samples)
    raise ValueError(f"Unsupported sampling strategy: {strategy}")
