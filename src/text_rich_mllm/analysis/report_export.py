from __future__ import annotations


def _format_slice_table(title: str, rows: dict[str, dict[str, float]]) -> list[str]:
    lines = [f"## {title}", "", "| Key | Count | Mean Score |", "| --- | ---: | ---: |"]
    for key, payload in rows.items():
        lines.append(f"| {key} | {payload['count']} | {payload['mean_score']:.4f} |")
    lines.append("")
    return lines


def evaluation_report_to_markdown(report: dict) -> str:
    lines = ["# Evaluation Report", ""]
    lines.extend(
        [
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in report.items():
        if key == "slices":
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")

    invalid_output_rate = report.get("invalid_output_rate", {})
    if invalid_output_rate:
        lines.extend(["## Invalid Output Rate", "", "| Dataset | Rate |", "| --- | ---: |"])
        for dataset_name, value in invalid_output_rate.items():
            lines.append(f"| {dataset_name} | {value:.4f} |")
        lines.append("")

    error_counts = report.get("error_counts", {})
    if error_counts:
        lines.extend(["## Error Counts", "", "| Error Type | Count |", "| --- | ---: |"])
        for error_type, count in error_counts.items():
            lines.append(f"| {error_type} | {count} |")
        lines.append("")

    slices = report.get("slices", {})
    for title, rows in slices.items():
        if rows:
            lines.extend(_format_slice_table(title, rows))
    return "\n".join(lines).strip() + "\n"
