#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/env/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRIPT_DIR}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRIPT_DIR}/.uv-python}"

uv run --no-sync python scripts/compute_lerobot_future_flow_video.py \
  --repo-id tpy/forge_all_0413 \
  --output-dir ./forge_all_0413 \
  --future-step 32 \
  --overwrite
