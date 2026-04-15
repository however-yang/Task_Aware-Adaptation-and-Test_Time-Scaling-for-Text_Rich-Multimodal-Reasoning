from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any

from text_rich_mllm.schemas import UnifiedSample


class BaseDatasetAdapter(ABC):
    dataset_name: str
    task_type: str
    answer_type: str

    def convert_records(
        self,
        records: list[dict[str, Any]],
        *,
        split: str,
        image_root: str | None = None,
    ) -> list[UnifiedSample]:
        return [
            self.convert_record(record, index=index, split=split, image_root=image_root)
            for index, record in enumerate(records)
        ]

    @abstractmethod
    def convert_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
        image_root: str | None = None,
    ) -> UnifiedSample:
        raise NotImplementedError

    @staticmethod
    def _join_image_path(image_root: str | None, image_path: str | None) -> str:
        if not image_path:
            return ""
        if image_root and not image_path.startswith(("http://", "https://")):
            return f"{image_root.rstrip('/\\\\')}/{image_path.lstrip('/\\\\')}"
        return image_path

    def _extract_image_paths(self, record: dict[str, Any], *, image_root: str | None = None) -> list[str]:
        image_paths: list[str] = []
        for key, value in record.items():
            if "image" not in key.lower():
                continue
            if isinstance(value, str) and value:
                image_paths.append(self._join_image_path(image_root, value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item:
                        image_paths.append(self._join_image_path(image_root, item))
                    elif isinstance(item, dict) and item.get("path"):
                        image_paths.append(self._join_image_path(image_root, item["path"]))
            elif isinstance(value, dict) and value.get("path"):
                image_paths.append(self._join_image_path(image_root, value["path"]))
        deduped: list[str] = []
        seen: set[str] = set()
        for path in image_paths:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    @staticmethod
    def _strip_prefix(text: str, prefix: str) -> str:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
        return text.strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _parse_mcq_string(self, options_text: str) -> list[str]:
        text = options_text.strip()
        text = self._strip_prefix(text, "[OPTIONS]")
        pattern = re.compile(r"\(([A-F])\)\s*(.*?)(?=\s*\([A-F]\)\s*|$)", re.DOTALL)
        choices = [match.group(2).strip() for match in pattern.finditer(text)]
        if choices:
            return choices
        if text:
            return [segment.strip() for segment in text.split("||") if segment.strip()]
        return []

    def _find_choice_label(self, answer_text: str, choices: list[str]) -> str | None:
        normalized_answer = self._normalize_text(answer_text)
        for index, choice in enumerate(choices):
            if self._normalize_text(choice) == normalized_answer:
                return chr(ord("A") + index)
        return None
