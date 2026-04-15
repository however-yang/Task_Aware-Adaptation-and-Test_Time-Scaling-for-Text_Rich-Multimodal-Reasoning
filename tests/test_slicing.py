from text_rich_mllm.evaluation.slicing import build_sliced_summary
from text_rich_mllm.schemas import PredictionRecord


def test_sliced_summary_aggregates_by_dataset_and_metadata() -> None:
    records = [
        PredictionRecord(
            sample_id="1",
            dataset_name="docvqa",
            gold_answer="a",
            raw_prediction="a",
            parsed_prediction="a",
            normalized_prediction="a",
            answer_type="open_text",
            score=1.0,
            task_type="document_qa",
            split="validation",
            error_type="correct",
            metadata={"page_id": "p1"},
        ),
        PredictionRecord(
            sample_id="2",
            dataset_name="docvqa",
            gold_answer="b",
            raw_prediction="c",
            parsed_prediction="c",
            normalized_prediction="c",
            answer_type="open_text",
            score=0.0,
            task_type="document_qa",
            split="validation",
            error_type="text_reading_failure",
            metadata={"page_id": "p1"},
        ),
    ]
    summary = build_sliced_summary(records, metadata_keys=["page_id"])
    assert summary["by_dataset"]["docvqa"]["count"] == 2
    assert summary["by_metadata.page_id"]["p1"]["mean_score"] == 0.5
