from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.analysis.case_sampling import sample_cases
from text_rich_mllm.schemas import PredictionRecord
from text_rich_mllm.utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-error", type=int, default=5)
    args = parser.parse_args()

    records = [PredictionRecord(**record) for record in read_jsonl(args.tagged_predictions)]
    sampled = sample_cases(records, limit_per_error=args.limit_per_error)
    write_json(sampled, args.output)
    print(f"Saved sampled cases to {args.output}")


if __name__ == "__main__":
    main()
