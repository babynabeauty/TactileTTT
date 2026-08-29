#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/taskall-2}"
ASSET_ID="${ASSET_ID:-$(basename "$DATA_REPO")}"
PI0_BASE_PARAMS="${PI0_BASE_PARAMS:-checkpoints/pi0_base/params}"
PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-/workspace/mnt/sqzhang26/gaoyuxuan/openpi/.cache/openpi-assets/checkpoints/pi05_base/params}"

export TRAIN_STEPS="${TRAIN_STEPS:-80000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export FSDP_DEVICES="${FSDP_DEVICES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export KEEP_PERIOD="${KEEP_PERIOD:-10000}"
export GPU_SLOTS="${GPU_SLOTS:-0,1,2,3;4,5,6,7}"
export RUN_TAG="${RUN_TAG:-${ASSET_ID}_baseline_80k_$(date +%m%d_%H%M)}"

PI0_ASSETS_DIR="${PI0_ASSETS_DIR:-assets/pi0_xhand_full_finetune_h16}"
PI05_ASSETS_DIR="${PI05_ASSETS_DIR:-assets/pi05_xhand_full_finetune_h16}"
PI0_TACTILE_ASSETS_DIR="${PI0_TACTILE_ASSETS_DIR:-assets/pi0_xhand_state_tactile_finetune_h16}"
PI05_TACTILE_ASSETS_DIR="${PI05_TACTILE_ASSETS_DIR:-assets/pi05_xhand_state_tactile_finetune_h16}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: ${message}: ${path}" >&2
    exit 2
  fi
}

require_path "$DATA_REPO/meta/info.json" "dataset not found"
require_path "$PI0_BASE_PARAMS" "pi0 base params not found"
require_path "$PI05_BASE_PARAMS" "pi05 base params not found"

log "Stage 1/2: normalization for four baseline configs"
NORM_CONFIGS="pi0_xhand_full_finetune_h16 pi05_xhand_full_finetune_h16 pi0_xhand_state_tactile_finetune_h16 pi05_xhand_state_tactile_finetune_h16" \
ASSET_ID="$ASSET_ID" \
bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"

JOB_LABELS=(
  "A_pi0_full_h16"
  "B_pi05_full_h16"
  "C_pi0_raw_state_tactile_h16"
  "D_pi05_raw_state_tactile_h16"
)
JOB_CONFIGS=(
  "pi0_xhand_full_finetune_h16"
  "pi05_xhand_full_finetune_h16"
  "pi0_xhand_state_tactile_finetune_h16"
  "pi05_xhand_state_tactile_finetune_h16"
)
JOB_ASSET_DIRS=(
  "$PI0_ASSETS_DIR"
  "$PI05_ASSETS_DIR"
  "$PI0_TACTILE_ASSETS_DIR"
  "$PI05_TACTILE_ASSETS_DIR"
)
JOB_WEIGHT_PATHS=(
  "$PI0_BASE_PARAMS"
  "$PI05_BASE_PARAMS"
  "$PI0_BASE_PARAMS"
  "$PI05_BASE_PARAMS"
)

log "Stage 2/2: queued baseline training"
log "Dataset: $DATA_REPO"
log "Asset ID: $ASSET_ID"
log "Steps: $TRAIN_STEPS, save interval: $SAVE_INTERVAL, batch: $GLOBAL_BATCH_SIZE, fsdp: $FSDP_DEVICES"
log "GPU slots: $GPU_SLOTS"

source scripts/four_gpu_training_queue.sh
DATA_ASSET_ID="$ASSET_ID" run_four_gpu_training_queue "$DATA_REPO"
