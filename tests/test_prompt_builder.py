from text_rich_mllm.utils.constants import AnswerType, PromptStyle, TaskType
from text_rich_mllm.prompts import PromptBuilder
from text_rich_mllm.schemas import UnifiedSample


def test_prompt_builder_uses_mcq_template_when_choices_exist() -> None:
    sample = UnifiedSample(
        sample_id="1",
        dataset_name="scienceqa",
        task_type=TaskType.SCIENTIFIC_QA.value,
        image_path="demo.png",
        question="What is shown?",
        choices=["cat", "dog", "bird"],
        gold_answer="A",
        answer_type=AnswerType.MULTIPLE_CHOICE.value,
        split="validation",
    )
    prompt = PromptBuilder().build(sample)
    assert "Options:" in prompt
    assert "Return only the option letter." in prompt


def test_direct_prompt_is_less_constrained() -> None:
    sample = UnifiedSample(
        sample_id="2",
        dataset_name="docvqa",
        task_type=TaskType.DOCUMENT_QA.value,
        image_path="demo.png",
        question="What is the invoice number?",
        gold_answer="123",
        answer_type=AnswerType.OPEN_TEXT.value,
        split="validation",
    )
    prompt = PromptBuilder(style=PromptStyle.DIRECT.value).build(sample)
    assert "Return only" not in prompt
    assert "Question: What is the invoice number?" in prompt

