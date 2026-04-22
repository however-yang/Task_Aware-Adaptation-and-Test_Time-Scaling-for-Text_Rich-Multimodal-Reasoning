"""
npu_utils.py — 昇腾 NPU 适配工具模块

提供统一的设备检测、device 字符串获取、以及 torch_npu 的安全导入。
所有需要区分 CUDA / NPU 的地方都通过此模块的接口，不直接写 "cuda" 或 "npu"。

用法：
  from text_rich_mllm.utils.npu_utils import get_device, is_npu_available, setup_npu

支持的运行环境（自动检测）：
  1. 昇腾 NPU（Ascend 910B，torch_npu 已安装）
  2. NVIDIA GPU（torch.cuda 可用）
  3. CPU（fallback）
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# ── torch_npu 安全导入 ────────────────────────────────────────────────────────
# 必须在 import torch 之前 import torch_npu，才能让 torch.npu.* 正确注册
_NPU_AVAILABLE = False
try:
    import torch_npu  # noqa: F401  # 注册 torch.npu 命名空间
    import torch
    _NPU_AVAILABLE = torch.npu.is_available()
    if _NPU_AVAILABLE:
        logger.info("torch_npu detected. Ascend NPU is available.")
except ImportError:
    import torch
    logger.debug("torch_npu not found. Falling back to CUDA/CPU.")


def is_npu_available() -> bool:
    """返回当前环境是否有可用的昇腾 NPU。"""
    return _NPU_AVAILABLE


def get_device(local_rank: int = 0) -> "torch.device":
    """
    自动选择最优设备。

    优先级：NPU > CUDA > CPU
    local_rank：多卡训练时的本地卡号（0-based）。
    """
    if _NPU_AVAILABLE:
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def setup_npu(local_rank: int = 0) -> None:
    """
    昇腾 NPU 环境初始化。

    - 设置当前进程使用的 NPU 卡号
    - 设置昇腾推荐的内存分配策略
    - 打印 NPU 版本信息（便于 debug）

    在训练脚本入口调用一次即可（在 import torch 之后）。
    """
    if not _NPU_AVAILABLE:
        logger.debug("setup_npu() called but NPU not available, skipping.")
        return

    torch.npu.set_device(local_rank)

    # 推荐的内存分配配置（类比 CUDA 的 expandable_segments）
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    # HCCL 多卡通信超时（昇腾集合通信库）
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")

    # 异步执行（0=启用，提升吞吐；调试时可设为 1）
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "0")

    logger.info(
        "NPU setup complete: device=npu:%d, torch_npu=%s",
        local_rank,
        getattr(torch_npu, "__version__", "unknown"),
    )


def device_str() -> str:
    """返回设备类型字符串：'npu' / 'cuda' / 'cpu'，用于日志。"""
    if _NPU_AVAILABLE:
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize() -> None:
    """跨平台的设备同步（对应 torch.cuda.synchronize() / torch.npu.synchronize()）。"""
    if _NPU_AVAILABLE:
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
