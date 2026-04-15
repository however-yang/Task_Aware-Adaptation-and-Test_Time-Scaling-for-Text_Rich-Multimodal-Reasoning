from __future__ import annotations

from text_rich_mllm.utils.constants import DatasetName
from text_rich_mllm.datasets.base import BaseDatasetAdapter
from text_rich_mllm.datasets.chartqa import ChartQAAdapter
from text_rich_mllm.datasets.docvqa import DocVQAAdapter
from text_rich_mllm.datasets.mmmu import MMMUAdapter
from text_rich_mllm.datasets.scienceqa import ScienceQAAdapter


_REGISTRY: dict[str, type[BaseDatasetAdapter]] = {
    DatasetName.DOCVQA.value: DocVQAAdapter,
    DatasetName.CHARTQA.value: ChartQAAdapter,
    DatasetName.SCIENCEQA.value: ScienceQAAdapter,
    DatasetName.MMMU.value: MMMUAdapter,
}


def build_dataset_adapter(dataset_name: str) -> BaseDatasetAdapter:
    try:
        return _REGISTRY[dataset_name.lower()]()
    except KeyError as exc:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {supported}") from exc

