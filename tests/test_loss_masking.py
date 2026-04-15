from text_rich_mllm.training.loss_masking import build_answer_only_labels


def test_answer_only_labels_mask_prompt_tokens() -> None:
    labels = build_answer_only_labels([1, 2, 3], [4, 5])
    assert labels == [-100, -100, -100, 4, 5]
