from __future__ import annotations

from pathlib import Path

from text_rich_mllm.training.hf_dataset import SupervisedTrainingDataset


class MultimodalSupervisedCollator:
    def __init__(self, processor, *, max_length: int | None = None, ignore_index: int = -100):
        self.processor = processor
        self.max_length = max_length
        self.ignore_index = ignore_index

    def __call__(self, examples):
        from PIL import Image

        images = [Image.open(example.image_path).convert("RGB") for example in examples]
        prompts = [example.prompt for example in examples]
        full_texts = [f"{example.prompt} {example.target_answer}".strip() for example in examples]

        processor_kwargs = {
            "images": images,
            "text": full_texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": self.max_length is not None,
        }
        if self.max_length is not None:
            processor_kwargs["max_length"] = self.max_length
        full_batch = self.processor(**processor_kwargs)

        prompt_kwargs = dict(processor_kwargs)
        prompt_kwargs["text"] = prompts
        prompt_batch = self.processor(**prompt_kwargs)

        labels = full_batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, :prompt_length] = self.ignore_index
        labels[full_batch["attention_mask"] == 0] = self.ignore_index
        full_batch["labels"] = labels
        return full_batch


def _build_training_arguments(output_dir: str, train_config: dict):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_config.get("batch_size", 1),
        per_device_eval_batch_size=train_config.get("eval_batch_size", train_config.get("batch_size", 1)),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        learning_rate=train_config.get("learning_rate", 1e-4),
        num_train_epochs=train_config.get("num_train_epochs", 1),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        logging_strategy=train_config.get("logging_strategy", "steps"),
        logging_steps=train_config.get("logging_steps", 10),
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 100),
        evaluation_strategy=train_config.get("eval_strategy", "no"),
        eval_steps=train_config.get("eval_steps"),
        save_total_limit=train_config.get("save_total_limit", 2),
        remove_unused_columns=False,
        report_to=[],
        bf16=train_config.get("bf16", False),
        fp16=train_config.get("fp16", False),
        dataloader_num_workers=train_config.get("dataloader_num_workers", 0),
        load_best_model_at_end=train_config.get("load_best_model_at_end", False),
        metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_config.get("greater_is_better", False),
    )


def train_with_hf_trainer(
    *,
    model,
    processor,
    train_examples,
    train_config: dict,
    eval_examples=None,
    resume_from_checkpoint: str | None = None,
):
    from transformers import Trainer

    output_dir = train_config.get("output_dir", "outputs/checkpoints/default")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = _build_training_arguments(output_dir, train_config)
    collator = MultimodalSupervisedCollator(
        processor,
        max_length=train_config.get("max_seq_length"),
        ignore_index=train_config.get("ignore_index", -100),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SupervisedTrainingDataset(train_examples),
        eval_dataset=SupervisedTrainingDataset(eval_examples) if eval_examples else None,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)
    return trainer
