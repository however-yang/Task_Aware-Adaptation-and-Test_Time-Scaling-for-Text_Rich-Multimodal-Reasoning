from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    import platform

    print(f"python: {platform.python_version()}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch: missing")

    for module_name in ["transformers", "datasets", "accelerate", "peft", "PIL", "yaml"]:
        try:
            module = __import__(module_name)
            print(f"{module_name}: {getattr(module, '__version__', 'ok')}")
        except ImportError:
            print(f"{module_name}: missing")


if __name__ == "__main__":
    main()
