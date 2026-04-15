from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.utils import load_yaml, write_jsonl
from text_rich_mllm.utils import get_logger


def _serialize_value(value, *, image_dir: Path, key: str, index: int):
    if hasattr(value, "save"):
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{key}_{index}.png"
        value.save(image_path)
        return str(image_path)
    if isinstance(value, dict):
        if "path" in value and value["path"]:
            return value["path"]
        return {
            sub_key: _serialize_value(sub_value, image_dir=image_dir, key=f"{key}_{sub_key}", index=index)
            for sub_key, sub_value in value.items()
        }
    if isinstance(value, list):
        return [
            _serialize_value(item, image_dir=image_dir, key=key, index=item_index)
            for item_index, item in enumerate(value)
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_hf_split(config: dict, split: str) -> str:
    specific_key = f"hf_{split}_split"
    return str(config.get(specific_key, split))


def _resolve_hf_subsets(config: dict) -> list[str | None]:
    subsets = config.get("hf_subsets")
    if subsets:
        return list(subsets)
    return [config.get("hf_subset")]


def export_hf_split(config_path: str, split: str, *, limit: int | None = None) -> tuple[str, int]:
    from datasets import load_dataset

    logger = get_logger("download_data")
    config = load_yaml(config_path)
    output_path = Path(config[f"raw_{split}"])
    image_dir = Path(config.get("image_root", output_path.parent / "images"))
    records = []
    hf_split = _resolve_hf_split(config, split)
    remaining_limit = limit
    global_index = 0

    for subset in _resolve_hf_subsets(config):
        dataset = load_dataset(
            config["hf_dataset_name"],
            name=subset,
            split=hf_split,
            cache_dir=config.get("hf_cache_dir"),
        )
        if remaining_limit is not None:
            dataset = dataset.select(range(min(remaining_limit, len(dataset))))
        for row in dataset:
            serialized = {
                key: _serialize_value(value, image_dir=image_dir, key=key, index=global_index)
                for key, value in row.items()
            }
            serialized["_hf_dataset"] = config["hf_dataset_name"]
            serialized["_hf_split"] = hf_split
            serialized["_hf_subset"] = subset
            records.append(serialized)
            global_index += 1
        if remaining_limit is not None:
            remaining_limit -= min(remaining_limit, len(dataset))
            if remaining_limit <= 0:
                break
    write_jsonl(records, output_path)
    logger.info("Exported %s records for %s split %s (HF split: %s)", len(records), config["name"], split, hf_split)
    return str(output_path), len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output_path, count = export_hf_split(args.config, args.split, limit=args.limit)
    print(f"Downloaded {count} records to {output_path}")


if __name__ == "__main__":
    main()
