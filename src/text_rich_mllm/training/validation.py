from __future__ import annotations

from text_rich_mllm.evaluation import UnifiedEvaluator


def run_validation(samples, predictions):
    evaluator = UnifiedEvaluator()
    return evaluator.evaluate(samples, predictions)
