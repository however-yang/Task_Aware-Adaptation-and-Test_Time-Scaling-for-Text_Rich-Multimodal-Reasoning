#!/usr/bin/env bash
# [NPU 版] 阶段 9：TRA-light Stage 2 训练（昇腾 910B）
#
# 对应实验 E5。从 E3/E4 LoRA checkpoint 续训，注入 TRA-light hooks。
#
# 用法：
#   bash scripts/npu/pipeline_stage_09_tra_training_npu.sh \
#     --checkpoint outputs/checkpoints/joint_docvqa_chartqa_npu/checkpoint-best
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

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-120}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# ── 参数解析 ──────────────────────────────────────────────────────────────
RESUME_CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) RESUME_CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${RESUME_CHECKPOINT}" ]]; then
    echo "错误: --checkpoint <path> 是必须的（E3/E4 LoRA checkpoint 路径）" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/___TRAINING_NPU_PIPELINE_LOGS___/run_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/00_MASTER__${TS}.log"
touch "${MASTER_LOG}"
append_master() { tee -a "${MASTER_LOG}"; }

echo "[NPU STAGE-09] TRA-light Stage 2 Training | run_id=${TS}" | append_master
echo "[NPU STAGE-09] Resume from: ${RESUME_CHECKPOINT}" | append_master

python scripts/train_peft.py \
    --train-config      configs/train/train_joint_tra.yaml \
    --model-config      configs/model/backbone_npu.yaml \
    --peft-config       configs/model/peft.yaml \
    --tra-config        configs/model/tra.yaml \
    --resume-checkpoint "${RESUME_CHECKPOINT}" \
2>&1 | tee "${LOG_DIR}/tra_npu_${TS}.log" | append_master

echo "[NPU STAGE-09] Done. Log: ${LOG_DIR}/tra_npu_${TS}.log" | append_master
