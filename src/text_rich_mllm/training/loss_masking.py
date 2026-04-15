from __future__ import annotations


def build_answer_only_labels(
    prompt_input_ids: list[int],
    answer_input_ids: list[int],
    *,
    ignore_index: int = -100,
) -> list[int]:
    return [ignore_index] * len(prompt_input_ids) + list(answer_input_ids)


def tokenize_prompt_answer_pair(
    tokenizer,
    *,
    prompt: str,
    answer: str,
    max_length: int | None = None,
    ignore_index: int = -100,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)
    labels = build_answer_only_labels(prompt_ids, answer_ids, ignore_index=ignore_index)
    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
