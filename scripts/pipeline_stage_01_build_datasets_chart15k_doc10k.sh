#!/usr/bin/env bash
# ChartQA：训练 15000 + 按 TRAIN_VAL_RATIO 匹配的 validation（默认 8:2 → 3750）
# DocVQA：训练 10000 + 匹配 validation（8:2 → 2500），减轻 WDS 下载压力
# ScienceQA / MMMU：仅 validation，全量（无 --limit，与主 pipeline 一致）
# 模型：末尾仍预下载 MODEL_NAME（默认 Qwen3-VL-8B-Instruct）
#
# 用法：
#   bash scripts/pipeline_stage_01_build_datasets_chart15k_doc10k.sh
#
# 可选覆盖（再传给主脚本逻辑）：
#   TRAIN_VAL_RATIO=8:2  CHARTQA_TRAIN_LIMIT=15000  DOCVQA_TRAIN_LIMIT=10000
#   SKIP_MMMU=1  DATA_DISK=...  等，与 pipeline_stage_01_build_datasets.sh 相同
#
set -euo pipefail

export CHARTQA_TRAIN_LIMIT="${CHARTQA_TRAIN_LIMIT:-15000}"
export DOCVQA_TRAIN_LIMIT="${DOCVQA_TRAIN_LIMIT:-10000}"
export TRAIN_LIMIT="${TRAIN_LIMIT:-15000}" # 仅作未单独设置时的回退；Chart/Doc 已显式 export

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/pipeline_stage_01_build_datasets.sh"
