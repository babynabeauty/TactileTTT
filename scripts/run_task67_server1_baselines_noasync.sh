#!/usr/bin/env bash
set -Eeuo pipefail

# Server 1 for the task6+7 experiment:
#   1. Compute missing norm stats with the fast local-parquet path.
#   2. Run the server1 jobs in this order:
#      pi0 tactile -> pi05 tactile -> no-async freeze -> no-async -> pi0 full -> pi05 full.
#
# The no-async ablation needs a pretrained patch encoder. Set
# PATCH_ENCODER_PARAMS to the Stage-1 checkpoint path. If the path does not
# exist yet, the script waits for it, so this can be launched while server2 is
# still pretraining the encoder.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/task6_7}"
ASSET_ID="${ASSET_ID:-$(basename "$DATA_REPO")}"
PYTHON="${PYTHON:-env/.venv/bin/python}"
PI0_BASE_PARAMS="${PI0_BASE_PARAMS:-checkpoints/pi0_base/params}"
PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-/workspace/mnt/sqzhang26/gaoyuxuan/openpi/.cache/openpi-assets/checkpoints/pi05_base/params}"
PATCH_ENCODER_PARAMS="${PATCH_ENCODER_PARAMS:-}"
RAW_TACTILE_ASSETS_DIR="${RAW_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"

PI0_ASSETS_DIR="${PI0_ASSETS_DIR:-assets/pi0_xhand_full_finetune_h16}"
PI05_ASSETS_DIR="${PI05_ASSETS_DIR:-assets/pi05_xhand_full_finetune_h16}"
PI0_TACTILE_ASSETS_DIR="${PI0_TACTILE_ASSETS_DIR:-assets/pi0_xhand_state_tactile_finetune_h16}"
PI05_TACTILE_ASSETS_DIR="${PI05_TACTILE_ASSETS_DIR:-assets/pi05_xhand_state_tactile_finetune_h16}"

export TRAIN_STEPS="${TRAIN_STEPS:-40000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export FSDP_DEVICES="${FSDP_DEVICES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export KEEP_PERIOD="${KEEP_PERIOD:-10000}"
NOASYNC_FREEZE_STEPS="${NOASYNC_FREEZE_STEPS:-5000}"
NOASYNC_FREEZE_SAVE_INTERVAL="${NOASYNC_FREEZE_SAVE_INTERVAL:-5000}"
export GPU_SLOTS="${GPU_SLOTS:-0,1,2,3;4,5,6,7}"
export NORM_BATCH_SIZE="${NORM_BATCH_SIZE:-64}"
export NORM_NUM_WORKERS="${NORM_NUM_WORKERS:-2}"
export RUN_TAG="${RUN_TAG:-${ASSET_ID}_task67_server1_$(date +%m%d_%H%M%S)}"

norm_file() {
  local assets_dir="$1"
  printf '%s/%s/norm_stats.json\n' "$assets_dir" "$ASSET_ID"
}

wait_for_path() {
  local path="$1"
  local label="$2"
  local timeout="${WAIT_FOR_PATCH_ENCODER_TIMEOUT:-0}"
  local poll="${WAIT_FOR_PATCH_ENCODER_POLL:-60}"
  local start now
  start="$(date +%s)"
  while [[ ! -e "$path" ]]; do
    if [[ "$timeout" != "0" ]]; then
      now="$(date +%s)"
      if (( now - start >= timeout )); then
        echo "ERROR: timed out waiting for ${label}: ${path}" >&2
        exit 2
      fi
    fi
    printf '[%s] Waiting for %s: %s\n' "$(date '+%F %T')" "$label" "$path"
    sleep "$poll"
  done
}

if [[ ! -f "$DATA_REPO/meta/info.json" ]]; then
  echo "ERROR: dataset not found: $DATA_REPO/meta/info.json" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -e "$PI0_BASE_PARAMS" ]]; then
  echo "ERROR: pi0 base params not found: $PI0_BASE_PARAMS" >&2
  exit 2
fi
if [[ ! -e "$PI05_BASE_PARAMS" ]]; then
  echo "ERROR: pi05 base params not found: $PI05_BASE_PARAMS" >&2
  exit 2
fi

mkdir -p logs

echo "[server1] Stage 1/5: normalization for baselines and no-async assets"
NORM_CONFIGS="pi0_xhand_full_finetune_h16 pi05_xhand_full_finetune_h16 pi0_xhand_state_tactile_finetune_h16 pi0_xhand_tactile_structured_raw_dual_ae" \
ASSET_ID="$ASSET_ID" \
PYTHON="$PYTHON" \
bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"

# pi0/pi05 state-tactile use the same pre-model state+tactile vector, so reuse.
src_norm="$(norm_file "$PI0_TACTILE_ASSETS_DIR")"
dst_norm="$(norm_file "$PI05_TACTILE_ASSETS_DIR")"
if [[ ! -f "$src_norm" ]]; then
  echo "ERROR: expected pi0 tactile norm stats not found: $src_norm" >&2
  exit 2
fi
mkdir -p "$(dirname "$dst_norm")"
cp "$src_norm" "$dst_norm"
echo "[server1] Reused tactile norm stats: $src_norm -> $dst_norm"

source scripts/four_gpu_training_queue.sh

JOB_LABELS=(
  "A_pi0_raw_state_tactile_h16"
  "B_pi05_raw_state_tactile_h16"
)
JOB_CONFIGS=(
  "pi0_xhand_state_tactile_finetune_h16"
  "pi05_xhand_state_tactile_finetune_h16"
)
JOB_ASSET_DIRS=(
  "$PI0_TACTILE_ASSETS_DIR"
  "$PI05_TACTILE_ASSETS_DIR"
)
JOB_WEIGHT_ARGS=(
  "--weight-loader.params-path $PI0_BASE_PARAMS"
  "--weight-loader.params-path $PI05_BASE_PARAMS"
)

echo "[server1] Stage 2/5: queued state+tactile baselines, steps=${TRAIN_STEPS}, batch=${GLOBAL_BATCH_SIZE}, fsdp=${FSDP_DEVICES}"
DATA_ASSET_ID="$ASSET_ID" run_four_gpu_training_queue "$DATA_REPO"

if [[ -z "$PATCH_ENCODER_PARAMS" ]]; then
  echo "ERROR: PATCH_ENCODER_PARAMS is required for pi0_xhand_dual_patch_pretrained_f8_h16_no_async." >&2
  echo "Example: PATCH_ENCODER_PARAMS=checkpoints/xhand_patch_tactile_encoder_pretrain/<exp>/19999/params" >&2
  exit 2
fi
wait_for_path "$PATCH_ENCODER_PARAMS" "Stage-1 patch encoder params"

NOASYNC_FREEZE_CONFIG="pi0_xhand_dual_patch_pretrained_f8_h16_no_async_freeze"
NOASYNC_FREEZE_LABEL="C_patch_pretrained_f8_h16_no_async_freeze"
TAIL_RUN_TAG="${RUN_TAG}_tail"
NOASYNC_FREEZE_PARAMS="checkpoints/${NOASYNC_FREEZE_CONFIG}/${NOASYNC_FREEZE_LABEL}_${TAIL_RUN_TAG}/$((NOASYNC_FREEZE_STEPS - 1))/params"

JOB_LABELS=(
  "$NOASYNC_FREEZE_LABEL"
  "E_pi0_full_h16"
  "D_patch_pretrained_f8_h16_no_async"
  "F_pi05_full_h16"
)
JOB_CONFIGS=(
  "$NOASYNC_FREEZE_CONFIG"
  "pi0_xhand_full_finetune_h16"
  "pi0_xhand_dual_patch_pretrained_f8_h16_no_async"
  "pi05_xhand_full_finetune_h16"
)
JOB_ASSET_DIRS=(
  "$RAW_TACTILE_ASSETS_DIR"
  "$PI0_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
  "$PI05_ASSETS_DIR"
)
JOB_WEIGHT_ARGS=(
  "--weight-loader.pi0-params-path $PI0_BASE_PARAMS --weight-loader.encoder-params-path $PATCH_ENCODER_PARAMS"
  "--weight-loader.params-path $PI0_BASE_PARAMS"
  "--weight-loader.pi0-params-path $NOASYNC_FREEZE_PARAMS --weight-loader.encoder-params-path $PATCH_ENCODER_PARAMS"
  "--weight-loader.params-path $PI05_BASE_PARAMS"
)
JOB_READY_PATHS=(
  ""
  ""
  "$NOASYNC_FREEZE_PARAMS"
  ""
)
JOB_TRAIN_STEPS=(
  "$NOASYNC_FREEZE_STEPS"
  "$TRAIN_STEPS"
  "$TRAIN_STEPS"
  "$TRAIN_STEPS"
)
JOB_SAVE_INTERVALS=(
  "$NOASYNC_FREEZE_SAVE_INTERVAL"
  "$SAVE_INTERVAL"
  "$SAVE_INTERVAL"
  "$SAVE_INTERVAL"
)

echo "[server1] Stage 3/3: queued no-async freeze/no-async and no-tactile baselines"
DATA_ASSET_ID="$ASSET_ID" \
RUN_TAG="$TAIL_RUN_TAG" \
run_four_gpu_training_queue "$DATA_REPO"
