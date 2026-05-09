#!/usr/bin/env bash
# [阶段 1] 构建数据集 + 模型预下载
#
# 训练数据集（下载 train + val，8:2 切分）：
#   - ChartQA   默认 train=15000 / val=3750（可用 CHARTQA_TRAIN_LIMIT 覆盖）
#   - DocVQA    默认与 TRAIN_LIMIT 一致（可用 DOCVQA_TRAIN_LIMIT 单独设为如 10000）
#
# 外部评测数据集（仅下载 validation，不参与训练）：
#   - ScienceQA  validation only
#   - MMMU       validation only（设 SKIP_MMMU=1 可跳过）
#
# 模型预下载（脚本末尾）：
#   - Qwen/Qwen3-VL-8B-Instruct（由 MODEL_NAME 变量控制）
#
# 日志:
#   logs/___DATA_BUILD_PIPELINE_LOGS___/run_<YYYYMMDD_HHMMSS>/
#
# 可覆盖的环境变量：
#   TRAIN_LIMIT=15000         ChartQA / DocVQA 的默认训练条数（二者未单独设时）
#   CHARTQA_TRAIN_LIMIT       仅 ChartQA 训练条数（默认 = TRAIN_LIMIT）
#   DOCVQA_TRAIN_LIMIT        仅 DocVQA 训练条数（默认 = TRAIN_LIMIT）
#   TRAIN_VAL_RATIO=8:2       训练:验证 比例（验证条数 = train×val/train）
#   SKIP_MMMU=1               跳过 MMMU 下载（网络受限时使用）
#   MODEL_NAME                预下载的模型名（默认 Qwen/Qwen3-VL-8B-Instruct）
#   DATA_DISK                 未设置时自动选 /root/autodl-tmp 或 /autodl-tmp（若目录存在）
#   SKIP_DATA_SYMLINK_CHECK=1 若坚持保留仓库内「实体 data 目录」（不推荐）
#   TEXT_RICH_MLLM_DATA_ROOT    数据集物理目录（默认 ${DATA_DISK}/text_rich_mllm_data，仓库内 data/ 会链到此）
#   HF_ENDPOINT               HuggingFace 镜像（默认 hf-mirror.com）
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_env_data_disk.sh"

# ── 数据量控制 ──────────────────────────────────────────────────────────────
TRAIN_LIMIT="${TRAIN_LIMIT:-15000}"
TRAIN_VAL_RATIO="${TRAIN_VAL_RATIO:-8:2}"
CHARTQA_TRAIN_LIMIT="${CHARTQA_TRAIN_LIMIT:-${TRAIN_LIMIT}}"
DOCVQA_TRAIN_LIMIT="${DOCVQA_TRAIN_LIMIT:-${TRAIN_LIMIT}}"

TS="$(date +%Y%m%d_%H%M%S)"
TS_ISO="$(date -Iseconds)"

LOG_DIR="${ROOT}/logs/___DATA_BUILD_PIPELINE_LOGS___/run_${TS}"
RESULT_DIR="${ROOT}/outputs/___DATA_BUILD_FINAL_RESULTS___/run_${TS}"
MANIFEST="${RESULT_DIR}/DATA_BUILD_MANIFEST__${TS}.txt"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"

append_master() { tee -a "$MASTER_LOG"; }

{
  echo "================================================================================"
  echo "DATA BUILD PIPELINE（阶段 1）| run_id=${TS}"
  echo "ISO 时间  : ${TS_ISO}"
  echo "工作目录  : ${ROOT}"
  echo "主控日志  : ${MASTER_LOG}"
  echo "DATA_DISK : ${DATA_DISK}"
  echo "DATA_ROOT : ${TEXT_RICH_MLLM_DATA_ROOT} (data/ -> 此处)"
  echo "HF 镜像   : ${HF_ENDPOINT}"
  echo "训练数据  : ChartQA train=${CHARTQA_TRAIN_LIMIT} + DocVQA train=${DOCVQA_TRAIN_LIMIT}（比例 ${TRAIN_VAL_RATIO}）"
  echo "评测数据  : ScienceQA（val）+ MMMU（val，SKIP_MMMU=${SKIP_MMMU:-0}）"
  echo "模型预载  : ${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
  if [[ -f "${HF_HOME}/token" ]] || [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Hub 认证  : 已设置"
  else
    echo "提示      : 建议先执行 huggingface-cli login 或 export HF_TOKEN=xxx"
  fi
  echo "================================================================================"
} | append_master

run_cmd() {
  local tag="$1"; shift
  local step_log="${LOG_DIR}/data_build_step__${tag}__${TS}.log"
  {
    echo ""
    echo "################################################################################"
    echo "# STEP [${tag}] 开始  $(date -Iseconds)"
    echo "################################################################################"
  } | append_master | tee "$step_log"
  "$@" 2>&1 | tee -a "$step_log" | append_master
  {
    echo "# STEP [${tag}] 结束  $(date -Iseconds)"
    echo ""
  } | append_master | tee -a "$step_log"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: ChartQA
#   --with-matched-validation 一条命令同时完成 train（15000）和
#   validation（3750，由 8:2 自动计算）的下载
# ══════════════════════════════════════════════════════════════════════════════
echo "" | append_master
echo ">>>>>> PHASE — ChartQA  train=${CHARTQA_TRAIN_LIMIT}  ratio=${TRAIN_VAL_RATIO} <<<<<<" | append_master

run_cmd "chartqa_download_train_and_val" \
  python scripts/download_data.py \
    --config configs/data/chartqa.yaml \
    --split train \
    --limit "${CHARTQA_TRAIN_LIMIT}" \
    --with-matched-validation \
    --train-val-ratio "${TRAIN_VAL_RATIO}"

run_cmd "chartqa_preprocess_train" \
  python scripts/preprocess_chartqa.py --split train

run_cmd "chartqa_preprocess_validation" \
  python scripts/preprocess_chartqa.py --split validation

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: DocVQA
# ══════════════════════════════════════════════════════════════════════════════
echo "" | append_master
echo ">>>>>> PHASE — DocVQA   train=${DOCVQA_TRAIN_LIMIT}  ratio=${TRAIN_VAL_RATIO} <<<<<<" | append_master

run_cmd "docvqa_download_train_and_val" \
  python scripts/download_data.py \
    --config configs/data/docvqa.yaml \
    --split train \
    --limit "${DOCVQA_TRAIN_LIMIT}" \
    --with-matched-validation \
    --train-val-ratio "${TRAIN_VAL_RATIO}"

run_cmd "docvqa_preprocess_train" \
  python scripts/preprocess_docvqa.py --split train

run_cmd "docvqa_preprocess_validation" \
  python scripts/preprocess_docvqa.py --split validation

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ScienceQA（仅 validation，外部评测集，不参与训练）
# ══════════════════════════════════════════════════════════════════════════════
echo "" | append_master
echo ">>>>>> PHASE — ScienceQA  (validation only) <<<<<<" | append_master

run_cmd "scienceqa_download_validation" \
  python scripts/download_data.py \
    --config configs/data/scienceqa.yaml \
    --split validation

run_cmd "scienceqa_preprocess_validation" \
  python scripts/preprocess_scienceqa.py --split validation

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: MMMU（仅 validation，外部评测集，可用 SKIP_MMMU=1 跳过）
# ══════════════════════════════════════════════════════════════════════════════
echo "" | append_master
if [[ "${SKIP_MMMU:-0}" == "1" ]]; then
  echo ">>>>>> PHASE — MMMU  [SKIP_MMMU=1，已跳过] <<<<<<" | append_master
else
  echo ">>>>>> PHASE — MMMU  (validation only) <<<<<<" | append_master

  run_cmd "mmmu_download_validation" \
    python scripts/download_data.py \
      --config configs/data/mmmu.yaml \
      --split validation

  run_cmd "mmmu_preprocess_validation" \
    python scripts/preprocess_mmmu.py --split validation
fi

# ══════════════════════════════════════════════════════════════════════════════
# 最终行数自检（全部 4 个数据集）
# ══════════════════════════════════════════════════════════════════════════════
run_cmd "line_count_self_check" bash -c "
  echo '=== raw JSONL 行数 ===';
  wc -l data/raw/chartqa/*.jsonl data/raw/docvqa/*.jsonl \
       data/raw/scienceqa/*.jsonl data/raw/mmmu/*.jsonl 2>/dev/null | sort -n || true;
  echo '=== processed JSONL 行数 ===';
  wc -l data/processed/chartqa/*.jsonl data/processed/docvqa/*.jsonl \
       data/processed/scienceqa/*.jsonl data/processed/mmmu/*.jsonl 2>/dev/null | sort -n || true
"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: 模型预下载（Qwen3-VL-8B-Instruct）
#   使用 huggingface_hub.snapshot_download 将模型权重缓存到 HF_HUB_CACHE。
#   后续训练脚本直接从缓存加载，避免训练时临时下载阻塞 GPU 时间。
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
echo "" | append_master
echo ">>>>>> PHASE — 模型预下载: ${MODEL_NAME} <<<<<<" | append_master

run_cmd "model_prefetch" \
  python - <<'PYEOF'
import os, sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub 未安装，尝试 pip install huggingface_hub", flush=True)
    sys.exit(1)

model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
cache_dir  = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
print(f"[model_prefetch] 下载模型: {model_name}", flush=True)
print(f"[model_prefetch] 缓存目录: {cache_dir}", flush=True)
path = snapshot_download(
    repo_id=model_name,
    cache_dir=cache_dir,
    ignore_patterns=["*.msgpack", "*.h5", "flax_*"],  # 跳过 Flax/TF 权重
)
print(f"[model_prefetch] 完成！本地路径: {path}", flush=True)
PYEOF

{
  echo "DATA_BUILD_MANIFEST | run_id=${TS}"
  echo "finished_at=$(date -Iseconds)"
  echo "log_dir=${LOG_DIR}"
  echo ""
  echo "[训练数据集]"
  echo "  chartqa   : train=${CHARTQA_TRAIN_LIMIT}, val=auto(${TRAIN_VAL_RATIO})"
  echo "  docvqa    : train=${DOCVQA_TRAIN_LIMIT}, val=auto(${TRAIN_VAL_RATIO})"
  echo "[外部评测集]"
  echo "  scienceqa : validation only"
  echo "  mmmu      : validation only (SKIP_MMMU=${SKIP_MMMU:-0})"
  echo "[模型]"
  echo "  ${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
  echo ""
  echo "[raw/processed line counts]"
  wc -l data/raw/chartqa/*.jsonl data/raw/docvqa/*.jsonl \
       data/raw/scienceqa/*.jsonl data/raw/mmmu/*.jsonl 2>/dev/null | sort -n || true
  wc -l data/processed/chartqa/*.jsonl data/processed/docvqa/*.jsonl \
       data/processed/scienceqa/*.jsonl data/processed/mmmu/*.jsonl 2>/dev/null | sort -n || true
} > "$MANIFEST"

cp -f "$MANIFEST" "${LOG_DIR}/DATA_BUILD_MANIFEST__${TS}.copy.txt" 2>/dev/null || true

{
  echo ""
  echo "================================================================================"
  echo "DATA BUILD PIPELINE 结束（阶段 1）| run_id=${TS}"
  echo "清单 : ${MANIFEST}"
  echo "主控日志 : ${MASTER_LOG}"
  echo "================================================================================"
} | append_master

exit 0
