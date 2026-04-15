from text_rich_mllm.evaluation.normalization import normalize_answer


def parse_prediction(raw_prediction: str, *, answer_type: str) -> str:
    return normalize_answer(raw_prediction, answer_type)
