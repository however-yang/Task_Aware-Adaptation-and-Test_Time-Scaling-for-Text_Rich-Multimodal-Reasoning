from enum import Enum


class StrEnum(str, Enum):
    """Small compatibility shim for Python < 3.11."""


class DatasetName(StrEnum):
    DOCVQA = "docvqa"
    CHARTQA = "chartqa"
    SCIENCEQA = "scienceqa"
    MMMU = "mmmu"


class TaskType(StrEnum):
    DOCUMENT_QA = "document_qa"
    CHART_QA = "chart_qa"
    SCIENTIFIC_QA = "scientific_qa"


class AnswerType(StrEnum):
    OPEN_TEXT = "open_text"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"


class PromptStyle(StrEnum):
    DIRECT = "direct"
    STRUCTURED = "structured"


MULTIPLE_CHOICE_LABELS = ("A", "B", "C", "D", "E", "F")

DATASET_TO_TASK = {
    DatasetName.DOCVQA: TaskType.DOCUMENT_QA,
    DatasetName.CHARTQA: TaskType.CHART_QA,
    DatasetName.SCIENCEQA: TaskType.SCIENTIFIC_QA,
    DatasetName.MMMU: TaskType.SCIENTIFIC_QA,
}
