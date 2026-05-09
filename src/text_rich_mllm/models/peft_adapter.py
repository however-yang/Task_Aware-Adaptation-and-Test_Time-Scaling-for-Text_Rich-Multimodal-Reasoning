from __future__ import annotations


def attach_lora_adapter(model, peft_config: dict):
    """
    挂载 LoRA 或 DoRA adapter。

    peft_config 说明：
      - use_dora: true   → 使用 DoRA（Weight-Decomposed LoRA, Liu et al. 2024）
      - use_dora: false  → 标准 LoRA（默认）
      - 其余字段透传给 LoraConfig

    参考：
      DoRA: arxiv.org/abs/2402.09353
      PEFT library: huggingface.co/docs/peft
    """
    from peft import LoraConfig, get_peft_model

    cfg = dict(peft_config)  # 浅拷贝，避免修改原始 dict
    use_dora = cfg.pop("use_dora", False)  # 从 config 提取，不传给 LoraConfig

    config = LoraConfig(use_dora=use_dora, **cfg)
    model = get_peft_model(model, config)

    # 打印可训练参数量（便于实验记录）
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model
