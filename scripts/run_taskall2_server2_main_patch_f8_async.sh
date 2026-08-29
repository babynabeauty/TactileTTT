#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/taskall-2}"
ASSET_ID="${ASSET_ID:-$(basename "$DATA_REPO")}"
RAW_TACTILE_ASSETS_DIR="${RAW_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
BASE_PARAMS="${BASE_PARAMS:-checkpoints/pi0_base/params}"

export RUN_TAG="${RUN_TAG:-${ASSET_ID}_patch_pretrained_f8_async_90k_$(date +%m%d_%H%M)}"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-64}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export KEEP_PERIOD="${KEEP_PERIOD:-10000}"
export STAGE1_STEPS="${STAGE1_STEPS:-20000}"
export STAGE1_SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL:-5000}"
export STAGE2A_STEPS="${STAGE2A_STEPS:-5000}"
export STAGE2A_SAVE_INTERVAL="${STAGE2A_SAVE_INTERVAL:-5000}"
export STAGE2B_STEPS="${STAGE2B_STEPS:-90000}"
export STAGE2B_SAVE_INTERVAL="${STAGE2B_SAVE_INTERVAL:-10000}"
export STAGE2_BASE_CONFIG="${STAGE2_BASE_CONFIG:-pi0_xhand_dual_patch_f8_h16_async}"
export ASSET_DIR="$RAW_TACTILE_ASSETS_DIR"
export DATA_ASSET_ID="$ASSET_ID"
export BASE_PARAMS="$BASE_PARAMS"

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
require_path "$BASE_PARAMS" "pi0 base params not found"

RAW_NORM_PATH="$RAW_TACTILE_ASSETS_DIR/$ASSET_ID/norm_stats.json"
if [[ ! -f "$RAW_NORM_PATH" ]]; then
  log "Raw tactile norm stats not found; computing first: $RAW_NORM_PATH"
  NORM_CONFIGS="pi0_xhand_tactile_structured_raw_dual_ae" \
  ASSET_ID="$ASSET_ID" \
  bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"
else
  log "Raw tactile norm stats already exist: $RAW_NORM_PATH"
fi

require_path "$RAW_NORM_PATH" "raw tactile norm stats not found after normalization"

log "Starting main patch-pretrained f8 async pipeline"
log "Dataset: $DATA_REPO"
log "Asset ID: $ASSET_ID"
log "Run tag: $RUN_TAG"
log "Stage2 base config: $STAGE2_BASE_CONFIG"
log "Stage2B steps: $STAGE2B_STEPS, save interval: $STAGE2B_SAVE_INTERVAL"
log "GPUs: $GPUS"

bash scripts/run_patch_encoder_pretrain_pipeline.sh "$DATA_REPO"
