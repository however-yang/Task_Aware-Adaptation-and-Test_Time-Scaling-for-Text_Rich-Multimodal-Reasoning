#!/usr/bin/env bash
# 由各 pipeline_stage_*.sh 在「已 cd 到仓库根且已设置 ROOT」之后 source。
#
# 作用：
#   - 默认 DATA_DISK 指向 AutoDL 数据盘（可用环境变量覆盖）
#   - 将 HuggingFace / Transformers 缓存放到数据盘
#   - 将仓库内 data/ 链到数据盘上的 TEXT_RICH_MLLM_DATA_ROOT（数据集与图片）
#   - 与 Python 中 resolve_training_output_dir 对齐：checkpoint 相对路径写到数据盘
#
# 环境变量（均可选）：
#   DATA_DISK                  未设置时自动选 /root/autodl-tmp 或 /autodl-tmp（若目录存在）
#   SKIP_DATA_SYMLINK_CHECK=1  保留仓库内「实体 data/」时不退出（不推荐）
#   TEXT_RICH_MLLM_REQUIRE_DATA_SYMLINK=0  不强制 data 为软链（仅不访问 data/ 的脚本，如阶段 6）
#   TEXT_RICH_MLLM_CHECKPOINT_ROOT  checkpoint 根（默认 ${DATA_DISK}，即 outputs/checkpoints/... 在数据盘下）
#   TEXT_RICH_MLLM_PROJECT_ROOT  仓库根（默认已设置的 ROOT）
#
# shellcheck shell=bash

: "${ROOT:?pipeline_env_data_disk.sh 要求先设置 ROOT 并 cd 到仓库根}"

# 未显式设置时：优先常见 AutoDL 数据盘挂载点
if [[ -z "${DATA_DISK:-}" ]]; then
  if [[ -d "/root/autodl-tmp" ]]; then
    DATA_DISK="/root/autodl-tmp"
  elif [[ -d "/autodl-tmp" ]]; then
    DATA_DISK="/autodl-tmp"
  else
    DATA_DISK="/root/autodl-tmp"
  fi
fi
export DATA_DISK

export TEXT_RICH_MLLM_PROJECT_ROOT="${TEXT_RICH_MLLM_PROJECT_ROOT:-${ROOT}}"
export TEXT_RICH_MLLM_DATA_ROOT="${TEXT_RICH_MLLM_DATA_ROOT:-${DATA_DISK}/text_rich_mllm_data}"
export TEXT_RICH_MLLM_CHECKPOINT_ROOT="${TEXT_RICH_MLLM_CHECKPOINT_ROOT:-${DATA_DISK}}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_DISK}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DATA_DISK}/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_DISK}/huggingface_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_DISK}/transformers_cache}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
mkdir -p "${TEXT_RICH_MLLM_DATA_ROOT}/raw" "${TEXT_RICH_MLLM_DATA_ROOT}/processed"

if [[ -e "${ROOT}/data" && ! -L "${ROOT}/data" ]]; then
  echo "[pipeline_env_data_disk] ERROR: ${ROOT}/data 为普通目录，数据集会继续写在系统盘。" >&2
  echo "  解决：将目录迁到数据盘后改为软链，例如：" >&2
  echo "    mv \"${ROOT}/data\" \"${TEXT_RICH_MLLM_DATA_ROOT}\"" >&2
  echo "    ln -sfn \"${TEXT_RICH_MLLM_DATA_ROOT}\" \"${ROOT}/data\"" >&2
  echo "  若确需保留本地目录（不推荐），可 export SKIP_DATA_SYMLINK_CHECK=1 后重跑。" >&2
  _req="${TEXT_RICH_MLLM_REQUIRE_DATA_SYMLINK:-1}"
  if [[ "${SKIP_DATA_SYMLINK_CHECK:-0}" != "1" && "${_req}" == "1" ]]; then
    exit 1
  fi
elif [[ ! -e "${ROOT}/data" ]]; then
  ln -sfn "${TEXT_RICH_MLLM_DATA_ROOT}" "${ROOT}/data"
fi
