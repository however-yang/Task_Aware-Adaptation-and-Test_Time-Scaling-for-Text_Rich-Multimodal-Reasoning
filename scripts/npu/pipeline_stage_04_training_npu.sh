#!/usr/bin/env bash
# [NPU 版] 阶段 4：LoRA SFT 训练（昇腾 910B）
#
# 与 pipeline_stage_04_training.sh 的区别：
#   1. 使用 backbone_npu.yaml（device_map=null，不走 accelerate dispatch）
#   2. 使用 train_joint_npu.yaml（bf16=true，dataloader_num_workers=0）
#   3. 训练前自动调用 check_npu_runtime.py 验证环境
#   4. 设置昇腾相关环境变量（ASCEND_RT_VISIBLE_DEVICES, HCCL 等）
#
# 用法（单卡）：
#   bash scripts/npu/pipeline_stage_04_training_npu.sh
#
# 用法（多卡，例如 4 卡）：
#   NPU_CARDS=4 bash scripts/npu/pipeline_stage_04_training_npu.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ── 昇腾环境变量 ──────────────────────────────────────────────────────────
export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_DISK}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_DISK}/huggingface_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_DISK}/transformers_cache}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# 昇腾专用环境变量
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"   # 使用的 NPU 卡号
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-120}"            # 集合通信超时
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"          # 异步执行
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# 禁用 tokenizers 并行（NPU 环境下的已知稳定性问题）
export TOKENIZERS_PARALLELISM=false

NPU_CARDS="${NPU_CARDS:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/___TRAINING_NPU_PIPELINE_LOGS___/run_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/00_MASTER__${TS}.log"
touch "${MASTER_LOG}"
append_master() { tee -a "${MASTER_LOG}"; }

log_step() { echo "[NPU STAGE-04] $*" | append_master; }

log_step "=================================================="
log_step "NPU LoRA SFT Training | run_id=${TS}"
log_step "NPU cards: ${NPU_CARDS} | ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
log_step "=================================================="

# ── Step 0: 验证 NPU 运行时 ───────────────────────────────────────────────
log_step "Step 0: Checking NPU runtime..."
python scripts/check_npu_runtime.py 2>&1 | append_master
log_step "NPU runtime check done."

# ── Step 1: 训练 ─────────────────────────────────────────────────────────
log_step "Step 1: Starting LoRA SFT training on NPU..."

if [[ "${NPU_CARDS}" -gt 1 ]]; then
    # 多卡：使用 torchrun（torch_npu 支持）
    MASTER_PORT="${MASTER_PORT:-29500}"
    log_step "Multi-card training: ${NPU_CARDS} NPUs, port=${MASTER_PORT}"
    torchrun \
        --nproc_per_node="${NPU_CARDS}" \
        --master_port="${MASTER_PORT}" \
        scripts/train_peft.py \
            --train-config  configs/train/train_joint_npu.yaml \
            --model-config  configs/model/backbone_npu.yaml \
            --peft-config   configs/model/peft.yaml \
        2>&1 | tee "${LOG_DIR}/train_npu_${TS}.log" | append_master
else
    # 单卡：直接 python
    python scripts/train_peft.py \
        --train-config  configs/train/train_joint_npu.yaml \
        --model-config  configs/model/backbone_npu.yaml \
        --peft-config   configs/model/peft.yaml \
    2>&1 | tee "${LOG_DIR}/train_npu_${TS}.log" | append_master
fi

log_step "=================================================="
log_step "Training complete. Log: ${LOG_DIR}/train_npu_${TS}.log"
log_step "=================================================="
