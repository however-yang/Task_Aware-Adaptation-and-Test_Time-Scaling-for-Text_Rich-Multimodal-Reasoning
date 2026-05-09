from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.training.checkpointing import composite_validation_score
from text_rich_mllm.utils import read_json, write_json


def parse_weights(weight_items: list[str]) -> dict[str, float]:
    weights = {}
    for item in weight_items:
        dataset_name, value = item.split("=", 1)
        weights[dataset_name] = float(value)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()

    weights = parse_weights(args.weights)
    scored = []
    for report_path in args.reports:
        report = read_json(report_path)
        score = composite_validation_score(report, dataset_weights=weights or None)
        scored.append({"report_path": report_path, "composite_score": score})

    best = max(scored, key=lambda item: item["composite_score"])
    payload = {"best": best, "all_scores": scored}
    if args.output:
        write_json(payload, args.output)

    # ── 格式化输出（论文日志存档） ───────────────────────────────────────
    line = "=" * 70
    print(f"\n{line}")
    print("  BEST CHECKPOINT SELECTION")
    print(line)
    print(f"  Best  : {Path(best['report_path']).name}")
    print(f"  Score : {best['composite_score']:.6f}")
    print(f"  Path  : {best['report_path']}")
    print("  --- All checkpoints (sorted by score) ---")
    for item in sorted(scored, key=lambda x: -x["composite_score"]):
        marker = " <-- BEST" if item["report_path"] == best["report_path"] else ""
        print(f"    {item['composite_score']:.6f}  {Path(item['report_path']).name}{marker}")
    print(f"{line}\n")


if __name__ == "__main__":
    main()
