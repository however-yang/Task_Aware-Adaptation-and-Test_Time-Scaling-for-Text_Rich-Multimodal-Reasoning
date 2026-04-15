from __future__ import annotations

from contextlib import nullcontext

def _move_to_device(payload, device):
    moved = {}
    for key, value in payload.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def strip_prompt_from_output(decoded: str, prompt: str) -> str:
    decoded = decoded.strip()
    prompt = prompt.strip()
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded


def run_generation(model, processor, image_path: str, prompt: str, generation_config: dict) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = _move_to_device(inputs, model_device)
    if hasattr(model, "eval"):
        model.eval()
    try:
        import torch

        context = torch.inference_mode()
    except ImportError:
        context = nullcontext()
    with context:
        generated = model.generate(**inputs, **generation_config)
    decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return strip_prompt_from_output(decoded, prompt)
