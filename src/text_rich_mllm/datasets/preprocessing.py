from __future__ import annotations

from pathlib import Path

from text_rich_mllm.datasets import build_dataset_adapter
from text_rich_mllm.utils import load_yaml, read_json, read_jsonl, write_json, write_jsonl


def load_raw_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    for key in ("data", "samples", "items", "questions", "annotations"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Could not infer record list from {path}")


def clean_unified_samples(
    samples,
    *,
    check_image_paths: bool = False,
    drop_missing_images: bool = False,
):
    cleaned = []
    stats = {
        "input_samples": len(samples),
        "kept_samples": 0,
        "dropped_empty_question": 0,
        "dropped_duplicate_id": 0,
        "missing_image_path": 0,
        "missing_image_file": 0,
    }
    seen_ids: set[str] = set()

    for sample in samples:
        sample.question = sample.question.strip()
        sample.gold_answer = sample.gold_answer.strip()
        sample.image_path = sample.image_path.strip()

        if not sample.question:
            stats["dropped_empty_question"] += 1
            continue
        if sample.sample_id in seen_ids:
            stats["dropped_duplicate_id"] += 1
            continue
        seen_ids.add(sample.sample_id)

        if not sample.image_path:
            stats["missing_image_path"] += 1
            if drop_missing_images:
                continue
        elif check_image_paths and not sample.image_path.startswith(("http://", "https://")):
            if not Path(sample.image_path).exists():
                stats["missing_image_file"] += 1
                if drop_missing_images:
                    continue

        cleaned.append(sample)

    stats["kept_samples"] = len(cleaned)
    return cleaned, stats


def convert_raw_records(
    *,
    dataset_name: str,
    input_path: str | Path,
    output_path: str | Path,
    split: str,
    image_root: str | None = None,
    check_image_paths: bool = False,
    drop_missing_images: bool = False,
    stats_path: str | Path | None = None,
) -> tuple[int, dict]:
    records = load_raw_records(input_path)
    adapter = build_dataset_adapter(dataset_name)
    samples = adapter.convert_records(records, split=split, image_root=image_root)
    samples, stats = clean_unified_samples(
        samples,
        check_image_paths=check_image_paths,
        drop_missing_images=drop_missing_images,
    )
    write_jsonl([sample.to_dict() for sample in samples], output_path)
    if stats_path:
        write_json(stats, stats_path)
    return len(samples), stats


def preprocess_from_dataset_config(config_path: str | Path, *, split: str) -> tuple[str, int]:
    config = load_yaml(config_path)
    input_key = f"raw_{split}"
    output_key = f"processed_{split}"
    stats_key = f"stats_{split}"
    count, _stats = convert_raw_records(
        dataset_name=config["name"],
        input_path=config[input_key],
        output_path=config[output_key],
        split=split,
        image_root=config.get("image_root"),
        check_image_paths=config.get("check_image_paths", False),
        drop_missing_images=config.get("drop_missing_images", False),
        stats_path=config.get(stats_key),
    )
    return str(config[output_key]), count
