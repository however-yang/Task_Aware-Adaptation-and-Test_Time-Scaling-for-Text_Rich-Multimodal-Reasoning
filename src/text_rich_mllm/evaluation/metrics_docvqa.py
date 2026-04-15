from text_rich_mllm.evaluation.normalization import normalize_text


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def anls_score(prediction: str, gold_answer: str, *, threshold: float = 0.5) -> float:
    pred = normalize_text(prediction)
    gold = normalize_text(gold_answer)
    if not pred and not gold:
        return 1.0
    max_len = max(len(pred), len(gold), 1)
    distance = _levenshtein_distance(pred, gold)
    similarity = 1.0 - distance / max_len
    return similarity if similarity >= threshold else 0.0
