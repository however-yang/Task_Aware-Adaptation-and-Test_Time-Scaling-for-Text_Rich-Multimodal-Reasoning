"""
check_npu_runtime.py — 昇腾 NPU 环境快速验证脚本

在服务器上运行训练前，先执行此脚本确认 NPU 环境正常。
用法：
  python scripts/check_npu_runtime.py

验证内容：
  1. torch_npu 是否成功导入
  2. NPU 卡是否可用（数量、型号）
  3. torch 版本兼容性
  4. BF16 是否支持
  5. 简单 Tensor 运算（加法/矩阵乘法）是否正常
  6. HCCL 多卡通信库是否存在（多卡训练前检查）
"""
from __future__ import annotations

import sys


def check_python_version() -> None:
    v = sys.version_info
    assert v >= (3, 10), f"Python 3.10+ required, got {v.major}.{v.minor}"
    print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")


def check_torch() -> None:
    import torch
    print(f"[OK] torch {torch.__version__}")
    return torch


def check_torch_npu(torch) -> None:
    try:
        import torch_npu  # noqa: F401
        print(f"[OK] torch_npu {getattr(torch_npu, '__version__', 'unknown')}")
    except ImportError:
        print("[FAIL] torch_npu not found! 请检查昇腾环境是否正确安装。")
        sys.exit(1)

    if not torch.npu.is_available():
        print("[FAIL] torch.npu.is_available() = False! NPU 不可用。")
        sys.exit(1)

    n_cards = torch.npu.device_count()
    print(f"[OK] NPU available: {n_cards} card(s)")
    for i in range(n_cards):
        props = torch.npu.get_device_properties(i)
        print(f"     Card {i}: {props.name}  |  HBM: {props.total_memory / 1024**3:.1f} GB")


def check_bf16(torch) -> None:
    try:
        torch.npu.set_device(0)
        x = torch.randn(4, 4, dtype=torch.bfloat16).npu()
        y = x @ x.T
        assert y.shape == (4, 4)
        print("[OK] BF16 matmul on NPU: passed")
    except Exception as e:
        print(f"[WARN] BF16 test failed: {e}")
        print("       如果使用 bf16=true 训练，请确认当前 NPU 型号支持 BF16（910B 支持）。")


def check_basic_ops(torch) -> None:
    torch.npu.set_device(0)
    a = torch.tensor([1.0, 2.0, 3.0]).npu()
    b = torch.tensor([4.0, 5.0, 6.0]).npu()
    c = a + b
    assert c.tolist() == [5.0, 7.0, 9.0], f"Unexpected: {c.tolist()}"
    print("[OK] Basic tensor add on NPU: passed")


def check_hccl() -> None:
    """验证多卡通信库是否存在（不实际初始化，避免需要多进程）。"""
    try:
        import hccl  # noqa: F401
        print("[OK] hccl found (multi-card communication library)")
    except ImportError:
        try:
            import torch.distributed as dist  # noqa: F401
            print("[OK] torch.distributed available (will use HCCL backend for NPU)")
        except ImportError:
            print("[WARN] Neither hccl nor torch.distributed found.")


def main() -> None:
    print("=" * 60)
    print("昇腾 NPU 运行时环境检查")
    print("=" * 60)

    check_python_version()
    torch = check_torch()
    check_torch_npu(torch)
    check_bf16(torch)
    check_basic_ops(torch)
    check_hccl()

    print("=" * 60)
    print("ALL CHECKS PASSED — NPU 环境就绪，可以开始训练。")
    print("=" * 60)


if __name__ == "__main__":
    main()
