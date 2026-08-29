#!/usr/bin/env bash


# setsid nohup env \
#   HF_LEROBOT_HOME="$PWD" \
#   HF_DATASETS_CACHE=.hf_datasets_cache \
#   HF_HUB_OFFLINE=1 \
#   RUN_TAG=task123_pool_ablation_$(date +%m%d_%H%M) \
#   TRAIN_STEPS=30000 \
#   GLOBAL_BATCH_SIZE=8 \
#   FSDP_DEVICES=1 \
#   NUM_WORKERS=2 \
#   bash scripts/run_xhand_pool_ablation_after_norm.sh \
#     data/task12345 \
#   > logs/pool_ablation_scheduler_$(date +%m%d_%H%M).log 2>&1 &


set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-}"
if [[ -z "$DATA_REPO" ]]; then
  echo "Usage: bash scripts/run_xhand_pool_ablation_after_norm.sh data/your_dataset" >&2
  exit 2
fi
if [[ ! -f "$DATA_REPO/meta/info.json" ]]; then
  echo "ERROR: LeRobot dataset not found: $DATA_REPO/meta/info.json" >&2
  exit 2
fi

ASSET_ID="${ASSET_ID:-$(basename "$DATA_REPO")}"
export DATA_ASSET_ID="${DATA_ASSET_ID:-$ASSET_ID}"
CALC_TACTILE_ASSETS_DIR="${CALC_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_dual_ae}"
RAW_TACTILE_ASSETS_DIR="${RAW_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
NO_TACTILE_ASSETS_DIR="${NO_TACTILE_ASSETS_DIR:-$CALC_TACTILE_ASSETS_DIR}"
PATCH_RAW_TACTILE_ASSETS_DIR="${PATCH_RAW_TACTILE_ASSETS_DIR:-$RAW_TACTILE_ASSETS_DIR}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Stage 1/2: normalization for ${DATA_REPO} (asset=${ASSET_ID})"
NORM_CONFIGS="pi0_xhand_tactile_structured_dual_ae pi0_xhand_tactile_structured_raw_dual_ae" \
  ASSET_ID="$ASSET_ID" \
  bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"

log "Stage 2/2: queued training for history/future pooled ablations"
log "No-tactile assets: ${NO_TACTILE_ASSETS_DIR}"
log "Raw tactile assets: ${RAW_TACTILE_ASSETS_DIR}"
log "Patch raw tactile assets: ${PATCH_RAW_TACTILE_ASSETS_DIR}"

JOB_LABELS=(
  "A_pi0_full"
  "C_raw_dual_ae_history_future_pool"
  "D_adaptive_patch_raw_dual_ae_history_future_pool"
)
JOB_CONFIGS=(
  "pi0_xhand_full_finetune"
  "pi0_xhand_tactile_structured_raw_dual_ae_history_future_pool"
  "pi0_xhand_tactile_structured_adaptive_patch_raw_dual_ae_history_future_pool"
)
JOB_ASSET_DIRS=(
  "$NO_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
  "$PATCH_RAW_TACTILE_ASSETS_DIR"
)

source scripts/four_gpu_training_queue.sh
run_four_gpu_training_queue "$DATA_REPO"
