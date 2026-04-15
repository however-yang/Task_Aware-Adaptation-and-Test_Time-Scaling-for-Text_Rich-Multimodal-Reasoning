from __future__ import annotations

from typing import Any

from text_rich_mllm.utils.constants import AnswerType, DatasetName, TaskType
from text_rich_mllm.datasets.base import BaseDatasetAdapter
from text_rich_mllm.schemas import UnifiedSample


class DocVQAAdapter(BaseDatasetAdapter):
    dataset_name = DatasetName.DOCVQA.value
    task_type = TaskType.DOCUMENT_QA.value
    answer_type = AnswerType.OPEN_TEXT.value

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
        image_root: str | None = None,
    ) -> UnifiedSample:
        answers = record.get("answers") or []
        gold_answer = record.get("answer") or (answers[0] if answers else "")
        image_path = record.get("image") or record.get("image_path") or ""
        other_metadata = record.get("other_metadata") if isinstance(record.get("other_metadata"), dict) else {}
        sample_id = record.get("question_id") or record.get("sample_id") or f"docvqa-{split}-{index}"
        return UnifiedSample(
            sample_id=str(sample_id),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            image_path=self._join_image_path(image_root, image_path),
            question=str(record["question"]),
            gold_answer=str(gold_answer),
            answer_type=self.answer_type,
            split=split,
            metadata={
                "page_id": other_metadata.get("page_id") or record.get("page_id"),
                "doc_id": other_metadata.get("doc_id") or other_metadata.get("ucsf_document_id"),
                "document_page_no": other_metadata.get("ucsf_document_page_no"),
                "other_metadata": other_metadata,
                "ocr_results": record.get("ocr_results"),
                "original_answers": answers,
            },
        )

