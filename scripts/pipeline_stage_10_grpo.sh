#!/usr/bin/env bash
# [阶段 9] Task-Stratified GRPO 训练（实验 E8）
#
# 功能：
#   在 SFT checkpoint（E3 LoRA 或 E5 TRA）基础上，
#   用 Task-Stratified GRPO 进行 RL 对齐。
#   每个 GRPO group 只包含同一 task 的样本，
#   确保 Advantage 在 task-homogeneous 的 reward scale 内计算。
#
# 前置条件：
#   - pipeline_stage_08_tra_training.sh（E5）或
#     pipeline_stage_04_training.sh（E3）已成功执行
#   - CHECKPOINT 指向有效 checkpoint 目录
#
# 用法：
#   bash scripts/pipeline_stage_09_grpo.sh \
#     --checkpoint <ckpt_dir> \
#     [--peft-config configs/model/peft.yaml] \
#     [--tra-config  configs/model/tra.yaml] \
#     [--dry-run]
#
# 日志：
#   logs/___GRPO_TRAINING_LOGS___/run_<TS>/
#
# 产物：
#   outputs/checkpoints/E4_grpo_joint_docvqa_chartqa/
#   outputs/checkpoints/E4_grpo_joint_docvqa_chartqa/grpo_step_log.json
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── 环境变量 ──────────────────────────────────────────────────────────────
export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export TEXT_RICH_MLLM_CHECKPOINT_ROOT="${TEXT_RICH_MLLM_CHECKPOINT_ROOT:-${DATA_DISK}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_DISK}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_DISK}/huggingface_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_DISK}/transformers_cache}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# ── 时间戳与日志 ──────────────────────────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
TS_ISO="$(date -Iseconds)"
LOG_DIR="${ROOT}/logs/___GRPO_TRAINING_LOGS___/run_${TS}"
mkdir -p "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"
append_master() { tee -a "$MASTER_LOG"; }

log_step() {
    local msg="$1"
    echo "[${TS_ISO}] [STAGE-10] ${msg}" | append_master
}

# ── 参数解析 ──────────────────────────────────────────────────────────────
CHECKPOINT=""
PEFT_CONFIG="configs/model/peft.yaml"
TRA_CONFIG=""
TRAIN_CONFIG="configs/train/train_joint_grpo.yaml"
MODEL_CONFIG="configs/model/backbone_main.yaml"
DRY_RUN_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)  CHECKPOINT="$2";   shift 2 ;;
        --peft-config) PEFT_CONFIG="$2";  shift 2 ;;
        --tra-config)  TRA_CONFIG="$2";   shift 2 ;;
        --dry-run)     DRY_RUN_FLAG="--dry-run"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── 前置检查 ──────────────────────────────────────────────────────────────
log_step "=== Task-Stratified GRPO Training (E8) START ==="
log_step "Checkpoint   : ${CHECKPOINT:-<none>}"
log_step "PEFT config  : ${PEFT_CONFIG}"
log_step "TRA config   : ${TRA_CONFIG:-<none>}"
log_step "Train config : ${TRAIN_CONFIG}"
log_step "Dry run      : ${DRY_RUN_FLAG:-false}"

if [[ -z "${CHECKPOINT}" && -z "${DRY_RUN_FLAG}" ]]; then
    log_step "WARNING: No --checkpoint provided. RL will start from fresh LoRA init (not recommended)."
fi

if [[ -n "${CHECKPOINT}" && ! -d "${CHECKPOINT}" ]]; then
    log_step "ERROR: Checkpoint directory not found: ${CHECKPOINT}"
    exit 1
fi

if [[ ! -f "${TRAIN_CONFIG}" ]]; then
    log_step "ERROR: Train config not found: ${TRAIN_CONFIG}"
    exit 1
fi

# ── 构建 TRA 参数（数组形式，防止路径含空格时断开）────────────────────
TRA_ARGS=()
if [[ -n "${TRA_CONFIG}" ]]; then
    if [[ ! -f "${TRA_CONFIG}" ]]; then
        log_step "ERROR: TRA config not found: ${TRA_CONFIG}"
        exit 1
    fi
    TRA_ARGS=("--tra-config" "${TRA_CONFIG}")
    log_step "TRA mode: hooks will be re-injected from ${TRA_CONFIG}"
fi

# ── 构建 checkpoint 参数（数组形式）──────────────────────────────────────
CKPT_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
    CKPT_ARGS=("--checkpoint" "${CHECKPOINT}")
fi

# ── 运行 GRPO 训练 ────────────────────────────────────────────────────────
log_step "Launching TS-GRPO training..."

python scripts/train_grpo.py \
    --train-config "${TRAIN_CONFIG}"  \
    --model-config "${MODEL_CONFIG}"  \
    --peft-config  "${PEFT_CONFIG}"   \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
    ${TRA_ARGS[@]+"${TRA_ARGS[@]}"}   \
    ${DRY_RUN_FLAG}                   \
    2>&1 | tee "${LOG_DIR}/01_grpo_training.log" | append_master

EXIT_CODE=${PIPESTATUS[0]}

# ── 结果记录 ──────────────────────────────────────────────────────────────
if [[ ${EXIT_CODE} -eq 0 ]]; then
    log_step "TS-GRPO training SUCCEEDED."
    echo "{\"stage\": \"09_grpo\", \"status\": \"success\", \"ts\": \"${TS_ISO}\", \"checkpoint\": \"${CHECKPOINT}\"}" \
        > "${LOG_DIR}/manifest.json"
else
    log_step "TS-GRPO training FAILED (exit code: ${EXIT_CODE})."
    echo "{\"stage\": \"09_grpo\", \"status\": \"failed\", \"exit_code\": ${EXIT_CODE}, \"ts\": \"${TS_ISO}\"}" \
        > "${LOG_DIR}/manifest.json"
    exit ${EXIT_CODE}
fi

log_step "=== STAGE 10 COMPLETE. Output: outputs/checkpoints/E4_grpo_joint_docvqa_chartqa ==="
log_step "Step log: outputs/checkpoints/E4_grpo_joint_docvqa_chartqa/grpo_step_log.json"

# ── GRPO step log 数值摘要（论文日志存档）────────────────────────────────
STEP_LOG="${TEXT_RICH_MLLM_CHECKPOINT_ROOT:-${DATA_DISK}}/outputs/checkpoints/E4_grpo_joint_docvqa_chartqa/grpo_step_log.json"
if [[ -f "${STEP_LOG}" ]]; then
    log_step "=== GRPO Reward Trend Summary ==="
    python - <<PYEOF 2>&1 | tee -a "${MASTER_LOG}"
import json, pathlib
data = json.loads(pathlib.Path("${STEP_LOG}").read_text(encoding="utf-8"))
if not data:
    print("[grpo_summary] No step data found.")
else:
    # 每 50 步打印一次均值（窗口大小=50）
    WINDOW = 50
    print(f"  {'Step range':<20} {'mean_reward':>12}  {'std_reward':>12}  {'mean_loss':>10}")
    print(f"  {'-'*20} {'-'*12}  {'-'*12}  {'-'*10}")
    for i in range(0, len(data), WINDOW):
        chunk = data[i:i+WINDOW]
        mean_r = sum(e["mean_reward"] for e in chunk) / len(chunk)
        std_r = (sum((e["mean_reward"] - mean_r)**2 for e in chunk) / max(len(chunk)-1, 1))**0.5
        mean_l = sum(e["loss"] for e in chunk) / len(chunk)
        s_start = chunk[0]["step"]
        s_end = chunk[-1]["step"]
        print(f"  step {s_start:>4}-{s_end:<4}           {mean_r:>12.4f}  {std_r:>12.4f}  {mean_l:>10.4f}")
    # 最终 overall
    all_r = [e["mean_reward"] for e in data]
    print(f"\n  Total steps: {len(data)}  Overall mean_reward: {sum(all_r)/len(all_r):.4f}")
PYEOF
fi

