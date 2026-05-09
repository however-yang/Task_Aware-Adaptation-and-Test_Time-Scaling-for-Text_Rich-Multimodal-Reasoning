from __future__ import annotations

import time
from pathlib import Path

from text_rich_mllm.models.generation_utils import run_generation
from text_rich_mllm.prompts import PromptBuilder


def _append_jsonl(records: list[dict], path: str | Path, *, mode: str = "a") -> None:
    """mode='w' 就是全量覆写，mode='a' 就是追加。"""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    from tqdm.auto import tqdm

    from text_rich_mllm.utils import write_jsonl

    builder = PromptBuilder(style=prompt_style)
    prediction_map = dict(existing_predictions or {})
    output_records = [{"sample_id": key, "prediction": value} for key, value in prediction_map.items()]

    iterable = list(samples)
    if limit is not None:
        iterable = iterable[:limit]

    to_process = [sample for sample in iterable if sample.sample_id not in prediction_map]
    resumed = len(iterable) - len(to_process)
    total_slots = len(iterable)

    print(
        "\n[inference] "
        f"prompt_style={prompt_style!r} "
        f"total_samples={total_slots}"
        + (f" limit={limit}" if limit is not None else "")
        + f" resume_skipped={resumed} "
        f"to_generate={len(to_process)} "
        f"output={output_path!s}",
        flush=True,
    )

    if not to_process:
        print("[inference] nothing to generate (all sample_ids already in predictions).", flush=True)
        return prediction_map

    t0 = time.perf_counter()
    pbar = tqdm(
        to_process,
        desc=f"infer[{prompt_style}]",
        unit="sample",
        total=len(to_process),
        dynamic_ncols=True,
        mininterval=0.5,
    )
    FLUSH_EVERY = 10   # 每 10 条 append 一次，避免 O(N²) 全量重写
    pending_records: list[dict] = []

    # 如果是 resume 模式，文件已有内容，用 append 模式继续追加
    write_mode = "a" if (existing_predictions and output_path and Path(output_path).exists()) else "w"

    for i, sample in enumerate(pbar, start=1):
        try:
            prediction = run_generation(
                model,
                processor,
                sample.image_path,
                builder.build(sample),
                generation_config,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"[inference] WARNING: sample_id={sample.sample_id!r} error={exc}", flush=True)
            prediction = ""
        prediction_map[sample.sample_id] = prediction
        new_record = {"sample_id": sample.sample_id, "prediction": prediction}
        output_records.append(new_record)
        pending_records.append(new_record)

        # 增量 append，每 10 条 flush 一次
        if output_path and (len(pending_records) >= FLUSH_EVERY or i == len(to_process)):
            _append_jsonl(pending_records, output_path, mode=write_mode)
            pending_records.clear()
            write_mode = "a"   # 首次写完后后续都是 append

        elapsed = time.perf_counter() - t0
        if elapsed > 0:
            pbar.set_postfix(rate=f"{i / elapsed:.2f}/s", refresh=False)

    elapsed_total = time.perf_counter() - t0
    rate_mean = len(to_process) / elapsed_total if elapsed_total > 0 else 0.0
    print(
        f"\n[inference] finished in {elapsed_total:.1f}s "
        f"({rate_mean:.3f} samples/s on generated subset; "
        f"{elapsed_total / max(len(to_process), 1):.2f}s/sample wall avg).",
        flush=True,
    )

    return prediction_map
