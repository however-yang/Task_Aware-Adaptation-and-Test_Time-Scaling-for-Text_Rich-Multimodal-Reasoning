from __future__ import annotations

from pathlib import Path

from text_rich_mllm.models.generation_utils import run_generation
from text_rich_mllm.prompts import PromptBuilder


def generate_predictions(
    *,
    samples,
    model,
    processor,
    prompt_style: str,
    generation_config: dict,
    output_path: str | Path | None = None,
    existing_predictions: dict[str, str] | None = None,
    limit: int | None = None,
    continue_on_error: bool = False,
):
    from text_rich_mllm.utils import write_jsonl

    builder = PromptBuilder(style=prompt_style)
    prediction_map = dict(existing_predictions or {})
    output_records = [{"sample_id": key, "prediction": value} for key, value in prediction_map.items()]

    iterable = list(samples)
    if limit is not None:
        iterable = iterable[:limit]

    for sample in iterable:
        if sample.sample_id in prediction_map:
            continue
        try:
            prediction = run_generation(
                model,
                processor,
                sample.image_path,
                builder.build(sample),
                generation_config,
            )
        except Exception:
            if not continue_on_error:
                raise
            prediction = ""
        prediction_map[sample.sample_id] = prediction
        output_records.append({"sample_id": sample.sample_id, "prediction": prediction})
        if output_path:
            write_jsonl(output_records, output_path)

    return prediction_map
