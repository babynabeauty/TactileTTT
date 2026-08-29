#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/env/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRIPT_DIR}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRIPT_DIR}/.uv-python}"

CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/compute_norm_stats.py --config-name pi0_libero


CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-sync scripts/train.py pi0_libero --exp-name=pi0_libero --overwrite

# 多卡训练
CUDA_VISIBLE_DEVICES=4,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-sync scripts/train.py pi0_seer_0409 --exp-name=pi0_seer_0409  --fsdp-devices 2   --overwrite 


# 训练 probe
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python scripts/train_future_query_probe.py \
  --config-name pi0_latent_flow_noise \
  --exp-name probe_from_30k_2gpu \
  --pretrained-params checkpoints/pi0_latent_force_flow_noise/pi0_latent_flow_noise_0428/29999/params \
  --batch-size 16 \
  --probe-layer 12 \
  --overwrite
