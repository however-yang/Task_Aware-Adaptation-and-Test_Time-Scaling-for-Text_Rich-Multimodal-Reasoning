from text_rich_mllm.datasets import build_dataset_adapter


def test_docvqa_adapter_uses_hf_fields() -> None:
    adapter = build_dataset_adapter("docvqa")
    sample = adapter.convert_record(
        {
            "question_id": 7,
            "question": "Invoice number?",
            "answers": ["12345"],
            "image": "doc.png",
            "ocr_results": {"tokens": ["12345"]},
            "other_metadata": {"doc_id": "d1", "page_id": "p2"},
        },
        index=0,
        split="train",
    )
    assert sample.sample_id == "7"
    assert sample.question == "Invoice number?"
    assert sample.gold_answer == "12345"
    assert sample.metadata["doc_id"] == "d1"


def test_chartqa_adapter_maps_query_and_label() -> None:
    adapter = build_dataset_adapter("chartqa")
    sample = adapter.convert_record(
        {
            "query": "What is the average?",
            "label": ["58"],
            "image": "chart.png",
            "human_or_machine": "human",
        },
        index=0,
        split="train",
    )
    assert sample.question == "What is the average?"
    assert sample.gold_answer == "58"
    assert sample.answer_type == "numeric"
    assert sample.metadata["source"] == "human"


def test_scienceqa_adapter_parses_option_string_and_maps_answer_to_letter() -> None:
    adapter = build_dataset_adapter("scienceqa")
    sample = adapter.convert_record(
        {
            "question": "[QUESTION]Which continent is highlighted?",
            "choices": "[OPTIONS](A) Europe (B) Antarctica (C) North America (D) Africa",
            "answer": "North America",
            "solution": "This continent is North America.",
            "CTH": False,
            "image": "map.png",
        },
        index=0,
        split="validation",
    )
    assert sample.question == "Which continent is highlighted?"
    assert sample.choices == ["Europe", "Antarctica", "North America", "Africa"]
    assert sample.gold_answer == "C"
    assert sample.metadata["solution"] == "This continent is North America."


def test_mmmu_adapter_uses_options_and_subset_metadata() -> None:
    adapter = build_dataset_adapter("mmmu")
    sample = adapter.convert_record(
        {
            "id": "validation_Accounting_1",
            "question": "Question: <image 1> Which option is correct?",
            "options": ["10", "20", "30"],
            "answer": "B",
            "image_1": "m1.png",
            "_hf_subset": "Accounting",
            "subfield": "Financial Accounting",
            "topic_difficulty": "Easy",
            "question_type": "multiple-choice",
        },
        index=0,
        split="validation",
    )
    assert sample.sample_id == "validation_Accounting_1"
    assert sample.choices == ["10", "20", "30"]
    assert sample.gold_answer == "B"
    assert sample.metadata["hf_subset"] == "Accounting"
    assert sample.metadata["topic_difficulty"] == "Easy"
