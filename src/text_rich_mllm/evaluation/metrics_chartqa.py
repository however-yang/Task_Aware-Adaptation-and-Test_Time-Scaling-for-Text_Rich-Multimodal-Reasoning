from text_rich_mllm.evaluation.normalization import normalize_answer, numeric_equal


def chartqa_score(prediction: str, gold_answer: str, *, answer_type: str) -> float:
    pred = normalize_answer(prediction, answer_type)
    gold = normalize_answer(gold_answer, answer_type)
    if pred == gold:
        return 1.0
    if numeric_equal(prediction, gold_answer):
        return 1.0
    return 0.0
