import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.analysis.report_export import evaluation_report_to_markdown

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to the JSON report file")
    parser.add_argument("--output", required=True, help="Path to the output Markdown file")
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    markdown_text = evaluation_report_to_markdown(report_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Exported Markdown report to {output_path}")

if __name__ == "__main__":
    main()
