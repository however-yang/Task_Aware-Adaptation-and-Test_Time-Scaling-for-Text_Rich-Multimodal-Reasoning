from __future__ import annotations

from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.models.peft_adapter import attach_lora_adapter
from text_rich_mllm.training.collator import (
    build_training_examples,
    build_training_examples_with_tra,
)
from text_rich_mllm.training.hf_trainer import (
    train_with_hf_trainer,
    train_with_hf_trainer_tra,
)
from text_rich_mllm.training.mixing import mix_training_samples


def prepare_training_run(
    model,
    samples,
    peft_config: dict,
    *,
    sampling_strategy: str = "balanced",
    prompt_style: str = PromptStyle.STRUCTURED.value,
):
    samples = mix_training_samples(samples, strategy=sampling_strategy)
    model = attach_lora_adapter(model, peft_config)
    examples = build_training_examples(samples, prompt_style=prompt_style)
    return model, examples


def run_training(
    *,
    model,
    processor,
    train_samples,
    peft_config: dict,
    train_config: dict,
    eval_samples=None,
    resume_from_checkpoint: str | None = None,
):
    """
    标准训练入口（LoRA / DoRA SFT）。
    对应实验 E3（LoRA）和 E4（DoRA，peft_config 中 use_dora=True）。
    """
    model, train_examples = prepare_training_run(
        model,
        train_samples,
        peft_config,
        sampling_strategy=train_config.get("sampling", "balanced"),
        prompt_style=train_config.get("prompt_style", PromptStyle.STRUCTURED.value),
    )
    eval_examples = None
    if eval_samples:
        eval_examples = build_training_examples(
            eval_samples,
            prompt_style=train_config.get("prompt_style", PromptStyle.STRUCTURED.value),
        )
    trainer = train_with_hf_trainer(
        model=model,
        processor=processor,
        train_examples=train_examples,
        train_config=train_config,
        eval_examples=eval_examples,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return model, train_examples, trainer


def run_training_with_tra(
    *,
    model,
    processor,
    train_samples,
    peft_config: dict,
    train_config: dict,
    tra_config_path: str,
    eval_samples=None,
    resume_from_checkpoint: str | None = None,
):
    """
    TRA-light 训练入口（Stage 2）。
    对应实验 E5：在 LoRA checkpoint 基础上续训，注入 TRA-light。

    额外步骤（相比 run_training）：
      1. 从 tra_config_path 加载 TRAConfig
      2. 调用 inject_tra(model, tra_config) 注入 TRABlock hooks
      3. 使用 build_training_examples_with_tra 填充 task_id 字段
      4. 使用 train_with_hf_trainer_tra（内含 TRATrainer）训练
    """
    from text_rich_mllm.adapters.text_rich_adapter import TRAConfig
    from text_rich_mllm.models.qwen_with_tra import inject_tra, load_tra_state, save_tra_state
    import os

    # 1. 加载 TRAConfig
    tra_config = TRAConfig.from_yaml(tra_config_path)

    # 2. 判断是【从 DoRA 开始注入 TRA】还是【恢复一个中断的 TRA 训练】
    is_resuming_tra = False
    if resume_from_checkpoint and os.path.exists(os.path.join(resume_from_checkpoint, "tra_state.pt")):
        is_resuming_tra = True

    # 3. 挂载 LoRA / DoRA
    if peft_config:
        if resume_from_checkpoint:
            # 无论是恢复 TRA 还是从 DoRA 起步，此时 resume_from_checkpoint 均包含 PEFT adapter 权重
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=True)
        else:
            model = attach_lora_adapter(model, peft_config)

    # 4. 注入 TRA hooks（在 LoRA 挂载后，兼容 PeftModel 封装）
    model = inject_tra(model, tra_config)

    # 如果是恢复中断的 TRA 训练，加载 TRA 的权重
    if is_resuming_tra:
        load_tra_state(model, os.path.join(resume_from_checkpoint, "tra_state.pt"))

    # 4. 数据混合 + 构建带 task_id 的 TrainingExamples
    train_samples = mix_training_samples(
        train_samples,
        strategy=train_config.get("sampling", "balanced"),
    )
    prompt_style = train_config.get("prompt_style", PromptStyle.STRUCTURED.value)
    train_examples = build_training_examples_with_tra(
        train_samples,
        prompt_style=prompt_style,
        task_name_to_id=tra_config.task_name_to_id,
    )
    eval_examples = None
    if eval_samples:
        eval_examples = build_training_examples_with_tra(
            eval_samples,
            prompt_style=prompt_style,
            task_name_to_id=tra_config.task_name_to_id,
        )

    # 5. 训练（TRA 专用 Trainer）
    # 如果只是从 DoRA 开始，决不能把 resume_from_checkpoint 传给 Trainer，否则会因为优化器参数维度不匹配（多了 TRA 参数）而崩溃！
    trainer_resume_arg = resume_from_checkpoint if is_resuming_tra else None

    trainer = train_with_hf_trainer_tra(
        model=model,
        processor=processor,
        train_examples=train_examples,
        train_config=train_config,
        eval_examples=eval_examples,
        resume_from_checkpoint=trainer_resume_arg,
    )
    return model, train_examples, trainer

