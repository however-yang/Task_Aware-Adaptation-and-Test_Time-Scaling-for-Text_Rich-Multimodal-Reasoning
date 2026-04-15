from __future__ import annotations

from text_rich_mllm.training.collator import TrainingExample


class SupervisedTrainingDataset:
    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]
