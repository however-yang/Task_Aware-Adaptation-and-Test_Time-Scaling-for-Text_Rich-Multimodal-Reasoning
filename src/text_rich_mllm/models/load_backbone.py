from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _resolve_device_map(device_map):
    """
    NPU 适配：当 torch_npu 已安装且 NPU 可用时，自动将 device_map='auto' 覆盖为 None。
    原因：accelerates 的 dispatch_model 与 torch_npu 的内核冲突，会导致 RuntimeError。
    CPU/CUDA 环境下不受影响。
    """
    try:
        from text_rich_mllm.utils.npu_utils import is_npu_available
        if is_npu_available() and device_map == "auto":
            logger.warning(
                "NPU detected: overriding device_map='auto' → None. "
                "Model will be moved to NPU device manually after loading."
            )
            return None
    except ImportError:
        pass
    return device_map


def _move_to_npu_if_needed(model) -> None:
    """
    NPU 适配：若 NPU 可用且模型仍在 CPU（device_map=None 时的默认状态），
    将模型整体移动到 npu:0（单卡）或当前 torch_npu 设置的卡。
    多卡训练由外层 DDP/accelerate 包装处理，此处只处理单卡情况。
    """
    try:
        from text_rich_mllm.utils.npu_utils import is_npu_available, get_device
        if not is_npu_available():
            return
        device = get_device(local_rank=0)
        model.to(device)
        logger.info("Model moved to %s", device)
    except ImportError:
        pass


def _is_peft_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").is_file()


def _load_pretrained_model_classes(model_name: str, *, trust_remote_code: bool = True, **kwargs):
    """依次尝试 VL / CausalLM，与 Hub 全量权重加载逻辑一致。"""
    from transformers import AutoModelForCausalLM

    model = None
    last_error = None
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

    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(model_name, trust_remote_code=trust_remote_code, **kwargs)
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        raise RuntimeError(f"Unable to load base model {model_name}") from last_error
    return model


def load_model_bundle(model_name: str, *, processor_name: str | None = None, trust_remote_code: bool = True, **kwargs):
    """
    加载 processor + model。
    - Hub / 本地「完整权重」目录：直接 from_pretrained。
    - **PEFT LoRA 目录**（含 adapter_config.json）：先按 adapter 内 base_model_name_or_path 加载底座，
      再用 PeftModel.from_pretrained 挂载适配器（训练产物推理必需）。
    """
    from transformers import AutoProcessor

    kwargs = dict(kwargs)
    raw_dt = kwargs.pop("torch_dtype", None)
    if raw_dt is None:
        raw_dt = kwargs.pop("dtype", None)
    norm_dt = _normalize_torch_dtype(raw_dt)
    if norm_dt is not None:
        kwargs["dtype"] = norm_dt
    kwargs.setdefault("low_cpu_mem_usage", True)

    # NPU 适配：解决 device_map 冲突
    if "device_map" in kwargs:
        kwargs["device_map"] = _resolve_device_map(kwargs["device_map"])

    path = Path(model_name)
    if _is_peft_adapter_dir(path):
        cfg_path = path / "adapter_config.json"
        acfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        base_id = acfg.get("base_model_name_or_path") or acfg.get("model_name_or_path")
        if not base_id:
            raise ValueError(f"{cfg_path} 缺少 base_model_name_or_path，无法挂载 LoRA")

        proc_src = None
        if (path / "tokenizer_config.json").is_file() or (path / "processor_config.json").is_file():
            proc_src = str(path)
        elif processor_name:
            proc_src = processor_name
        else:
            proc_src = base_id

        processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=trust_remote_code)
        base_model = _load_pretrained_model_classes(base_id, trust_remote_code=trust_remote_code, **kwargs)

        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, str(path))

        # NPU 适配：若 device_map=None，手动移动到 NPU 设备
        _move_to_npu_if_needed(model)
        return processor, model

    processor = AutoProcessor.from_pretrained(processor_name or model_name, trust_remote_code=trust_remote_code)
    model = _load_pretrained_model_classes(model_name, trust_remote_code=trust_remote_code, **kwargs)

    # NPU 适配：若 device_map=None，手动移动到 NPU 设备
    _move_to_npu_if_needed(model)
    return processor, model


def load_model_bundle_with_optional_checkpoint(*, checkpoint: str | None, model_config: dict):
    """
    供推理/验证脚本使用：传入 checkpoint（Hub 全量、合并权重目录或 LoRA adapter 目录）时，
    仍合并 backbone yaml 里的 dtype / device_map / trust_remote_code，与 train_peft 加载条件对齐。
    """
    if checkpoint:
        extras = {k: v for k, v in model_config.items() if k != "model_name"}
        return load_model_bundle(checkpoint, **extras)
    return load_model_bundle(**model_config)
