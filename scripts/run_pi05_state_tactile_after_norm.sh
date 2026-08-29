#!/usr/bin/env bash
set -Eeuo pipefail

# Example:
#   cd /workspace/mnt/sqzhang26/FactileLDM
#   setsid nohup env \
#     RUN_TAG=task12345_2_pi05_state_tactile_$(date +%m%d_%H%M) \
#     bash scripts/run_pi05_state_tactile_after_norm.sh \
#       data/task12345-2 \
#     > logs/pi05_state_tactile_scheduler_$(date +%m%d_%H%M).log 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/task12345-2}"
if [[ ! -f "$DATA_REPO/meta/info.json" ]]; then
  echo "ERROR: LeRobot dataset not found: $DATA_REPO/meta/info.json" >&2
  exit 2
fi

ASSET_ID="${ASSET_ID:-$(basename "$DATA_REPO")}"
export DATA_ASSET_ID="${DATA_ASSET_ID:-$ASSET_ID}"

PI0_BASE_PARAMS="${PI0_BASE_PARAMS:-checkpoints/pi0_base/params}"
PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-/workspace/mnt/sqzhang26/gaoyuxuan/openpi/.cache/openpi-assets/checkpoints/pi05_base/params}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export TRAIN_STEPS="${TRAIN_STEPS:-60000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export FSDP_DEVICES="${FSDP_DEVICES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export KEEP_PERIOD="${KEEP_PERIOD:-10000}"
export GPU_SLOTS="${GPU_SLOTS:-0,1,2,3;4,5,6,7}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Stage 1/2: normalization for ${DATA_REPO} (asset=${ASSET_ID})"
NORM_CONFIGS="pi05_xhand_full_finetune_h16 pi0_xhand_state_tactile_finetune_h16 pi05_xhand_state_tactile_finetune_h16" \
  ASSET_ID="$ASSET_ID" \
  bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"

log "Stage 2/2: queued training for pi0/pi0.5 current-tactile baselines"
log "pi0 base params: ${PI0_BASE_PARAMS}"
log "pi0.5 base params: ${PI05_BASE_PARAMS}"

JOB_LABELS=(
  "A_pi05_full_h16"
  "B_pi0_state_tactile_h16"
  "C_pi05_state_tactile_h16"
)
JOB_CONFIGS=(
  "pi05_xhand_full_finetune_h16"
  "pi0_xhand_state_tactile_finetune_h16"
  "pi05_xhand_state_tactile_finetune_h16"
)
JOB_ASSET_DIRS=(
  "assets/pi05_xhand_full_finetune_h16"
  "assets/pi0_xhand_state_tactile_finetune_h16"
  "assets/pi05_xhand_state_tactile_finetune_h16"
)
JOB_WEIGHT_PATHS=(
  "$PI05_BASE_PARAMS"
  "$PI0_BASE_PARAMS"
  "$PI05_BASE_PARAMS"
)

source scripts/four_gpu_training_queue.sh
run_four_gpu_training_queue "$DATA_REPO"
