from __future__ import annotations


def _normalize_torch_dtype(torch_dtype):
    if torch_dtype is None:
        return None
    if not isinstance(torch_dtype, str):
        return torch_dtype
    import torch

    lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return lookup.get(torch_dtype.lower(), torch_dtype)


def load_model_bundle(model_name: str, *, processor_name: str | None = None, trust_remote_code: bool = True, **kwargs):
    from transformers import AutoModelForCausalLM, AutoProcessor

    kwargs = dict(kwargs)
    kwargs["torch_dtype"] = _normalize_torch_dtype(kwargs.get("torch_dtype"))
    kwargs.setdefault("low_cpu_mem_usage", True)
    processor = AutoProcessor.from_pretrained(processor_name or model_name, trust_remote_code=trust_remote_code)
    model = None
    model_classes = []
    try:
        from transformers import AutoModelForImageTextToText

        model_classes.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        model_classes.append(AutoModelForVision2Seq)
    except ImportError:
        pass
    model_classes.append(AutoModelForCausalLM)

    last_error = None
    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(model_name, trust_remote_code=trust_remote_code, **kwargs)
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        raise RuntimeError(f"Unable to load model {model_name}") from last_error
    return processor, model
