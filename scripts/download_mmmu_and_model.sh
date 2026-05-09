#!/usr/bin/env bash
# 仅下载 MMMU（validation 全量，与 pipeline 阶段 1 中 MMMU 段一致）+ 预下载模型权重。
# 适用于主流程在 MMMU 或模型步骤前中断、怀疑卡死时单独重跑。
#
# 用法：
#   cd /root/Final && bash scripts/download_mmmu_and_model.sh
#
# 环境变量（与 pipeline 一致）：
#   MODEL_NAME              默认 Qwen/Qwen3-VL-8B-Instruct
#   DATA_DISK / HF_ENDPOINT / HF_TOKEN 等见 pipeline_env_data_disk.sh
#
# 若上次 MMMU 下到一半（有 images 无 validation.jsonl），建议先清空再跑，避免图片索引与 jsonl 不一致：
#   CLEAN_MMMU_RAW=1 bash scripts/download_mmmu_and_model.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_env_data_disk.sh"

TS="$(date +%Y%m%d_%H%M%S)"
TS_ISO="$(date -Iseconds)"
LOG_DIR="${ROOT}/logs/___MMMU_MODEL_ONLY___/run_${TS}"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/00_MASTER.log"
touch "$MASTER_LOG"
append() { tee -a "$MASTER_LOG"; }

{
  echo "================================================================================"
  echo "MMMU + 模型预下载（单独脚本）| run_id=${TS}"
  echo "ISO 时间  : ${TS_ISO}"
  echo "工作目录  : ${ROOT}"
  echo "主控日志  : ${MASTER_LOG}"
  echo "DATA_DISK : ${DATA_DISK}"
  echo "HF 镜像   : ${HF_ENDPOINT}"
  echo "MODEL_NAME: ${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
  echo "================================================================================"
} | append

if [[ "${CLEAN_MMMU_RAW:-0}" == "1" ]]; then
  echo "[INFO] CLEAN_MMMU_RAW=1：删除 data/raw/mmmu 后重新下载" | append
  rm -rf "${ROOT}/data/raw/mmmu"
fi

run_step() {
  local tag="$1"
  shift
  local step_log="${LOG_DIR}/step__${tag}__${TS}.log"
  {
    echo ""
    echo "################################################################################"
    echo "# STEP [${tag}] 开始  $(date -Iseconds)"
    echo "################################################################################"
  } | append | tee "$step_log"
  "$@" 2>&1 | tee -a "$step_log" | append
  echo "# STEP [${tag}] 结束  $(date -Iseconds)" | append | tee -a "$step_log"
}

run_step "mmmu_download_validation" \
  python scripts/download_data.py \
    --config configs/data/mmmu.yaml \
    --split validation

run_step "mmmu_preprocess_validation" \
  python scripts/preprocess_mmmu.py --split validation

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
export MODEL_NAME
run_step "model_prefetch" \
  python - <<'PYEOF'
import os, sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub 未安装，请 pip install huggingface_hub", flush=True)
    sys.exit(1)

model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
print(f"[model_prefetch] 下载模型: {model_name}", flush=True)
print(f"[model_prefetch] 缓存目录: {cache_dir}", flush=True)
path = snapshot_download(
    repo_id=model_name,
    cache_dir=cache_dir,
    ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
)
print(f"[model_prefetch] 完成！本地路径: {path}", flush=True)
PYEOF

{
  echo ""
  echo "=== MMMU / 模型 自检 ==="
  wc -l "${ROOT}/data/raw/mmmu/validation.jsonl" 2>/dev/null || echo "raw mmmu validation.jsonl 缺失"
  wc -l "${ROOT}/data/processed/mmmu/validation.jsonl" 2>/dev/null || echo "processed mmmu 缺失"
  echo "日志目录: ${LOG_DIR}"
  echo "================================================================================"
} | append

echo "完成。主控日志: ${MASTER_LOG}"
