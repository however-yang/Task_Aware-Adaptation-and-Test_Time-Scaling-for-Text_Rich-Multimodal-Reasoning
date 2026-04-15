from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.datasets.preprocessing import convert_raw_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--check-image-paths", action="store_true")
    parser.add_argument("--drop-missing-images", action="store_true")
    parser.add_argument("--stats-output")
    args = parser.parse_args()

    count, stats = convert_raw_records(
        dataset_name=args.dataset,
        input_path=args.input,
        output_path=args.output,
        split=args.split,
        image_root=args.image_root,
        check_image_paths=args.check_image_paths,
        drop_missing_images=args.drop_missing_images,
        stats_path=args.stats_output,
    )
    print(f"Wrote {count} unified samples to {args.output}")
    if args.stats_output:
        print(f"Saved preprocessing stats to {args.stats_output}: {stats}")


if __name__ == "__main__":
    main()
