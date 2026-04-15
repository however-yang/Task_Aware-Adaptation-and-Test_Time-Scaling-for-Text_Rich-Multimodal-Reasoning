from text_rich_mllm.evaluation.normalization import extract_choice_label


def multiple_choice_accuracy(prediction: str, gold_answer: str) -> float:
    return float(extract_choice_label(prediction) == extract_choice_label(gold_answer))
