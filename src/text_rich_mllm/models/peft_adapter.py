from __future__ import annotations


def attach_lora_adapter(model, peft_config: dict):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(**peft_config)
    return get_peft_model(model, config)
