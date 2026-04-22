#!/usr/bin/env bash
# [NPU 版] 阶段 10：TS-GRPO 训练（昇腾 910B）
#
# 对应实验 E8。从 E5 TRA checkpoint 续训，RL 对齐。
#
# 用法：
#   bash scripts/npu/pipeline_stage_10_grpo_npu.sh \
#     --checkpoint outputs/checkpoints/joint_tra_light/checkpoint-best \
#     --tra-config configs/model/tra.yaml
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

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

CHECKPOINT=""
TRA_CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)  CHECKPOINT="$2";  shift 2 ;;
        --tra-config)  TRA_CONFIG="$2";  shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${CHECKPOINT}" ]] && { echo "错误: --checkpoint 必须指定" >&2; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/___TRAINING_NPU_PIPELINE_LOGS___/run_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/00_MASTER__${TS}.log"
touch "${MASTER_LOG}"
append_master() { tee -a "${MASTER_LOG}"; }

echo "[NPU STAGE-10] TS-GRPO Training | run_id=${TS}" | append_master
echo "[NPU STAGE-10] Checkpoint: ${CHECKPOINT}" | append_master

TRA_ARG=""
[[ -n "${TRA_CONFIG}" ]] && TRA_ARG="--tra-config ${TRA_CONFIG}"

python scripts/train_grpo.py \
    --train-config configs/train/train_joint_grpo_npu.yaml \
    --model-config configs/model/backbone_npu.yaml \
    --peft-config  configs/model/peft.yaml \
    --checkpoint   "${CHECKPOINT}" \
    ${TRA_ARG} \
2>&1 | tee "${LOG_DIR}/grpo_npu_${TS}.log" | append_master

echo "[NPU STAGE-10] Done. Log: ${LOG_DIR}/grpo_npu_${TS}.log" | append_master
