from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.analysis.report_export import evaluation_report_to_markdown
from text_rich_mllm.utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = read_json(args.report)
    markdown = evaluation_report_to_markdown(report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(markdown, encoding="utf-8")
    print(f"Exported markdown tables to {args.output}")


if __name__ == "__main__":
    main()
