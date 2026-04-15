from enum import Enum


class ErrorType(str, Enum):
    TEXT_READING_FAILURE = "text_reading_failure"
    LAYOUT_GROUNDING_FAILURE = "layout_grounding_failure"
    CHART_REASONING_FAILURE = "chart_reasoning_failure"
    SCIENTIFIC_REASONING_FAILURE = "scientific_figure_reasoning_failure"
    OUTPUT_MISMATCH = "output_mismatch"
    CORRECT = "correct"
    UNKNOWN = "unknown"
