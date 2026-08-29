#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RAW_TACTILE_ASSETS_DIR="${RAW_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
PI0_BASE_PARAMS="${PI0_BASE_PARAMS:-checkpoints/pi0_base/params}"
PATCH_POLICY_PARAMS="${PATCH_POLICY_PARAMS:-$PI0_BASE_PARAMS}"
PATCH_ENCODER_PARAMS="${PATCH_ENCODER_PARAMS:-}"

export TRAIN_STEPS="${TRAIN_STEPS:-60000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export FSDP_DEVICES="${FSDP_DEVICES:-4}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export KEEP_PERIOD="${KEEP_PERIOD:-10000}"
export GPU_SLOTS="${GPU_SLOTS:-0,1,2,3,4,5,6,7}"

if [[ -z "$PATCH_ENCODER_PARAMS" ]]; then
  echo "ERROR: PATCH_ENCODER_PARAMS is required for patch-pretrained ablations." >&2
  echo "Example: PATCH_ENCODER_PARAMS=checkpoints/xhand_patch_tactile_encoder_pretrain/<exp>/<step>/params" >&2
  exit 2
fi
if [[ ! -e "$PATCH_ENCODER_PARAMS" ]]; then
  echo "ERROR: PATCH_ENCODER_PARAMS not found: $PATCH_ENCODER_PARAMS" >&2
  exit 2
fi
if [[ ! -e "$PI0_BASE_PARAMS" ]]; then
  echo "ERROR: PI0_BASE_PARAMS not found: $PI0_BASE_PARAMS" >&2
  exit 2
fi
if [[ ! -e "$PATCH_POLICY_PARAMS" ]]; then
  echo "ERROR: PATCH_POLICY_PARAMS not found: $PATCH_POLICY_PARAMS" >&2
  exit 2
fi

JOB_LABELS=(
  "A_no_future"
  "B_no_future_update"
  "C_raw_mlp"
  "D_direct_align"
  "E_no_async"
)
JOB_CONFIGS=(
  "pi0_xhand_patch_pretrained_f8_h16_async_no_future"
  "pi0_xhand_dual_patch_pretrained_f8_h16_async_no_future_update"
  "pi0_xhand_dual_raw_mlp_f8_h16_async"
  "pi0_xhand_patch_pretrained_f8_h16_async_direct_align"
  "pi0_xhand_dual_patch_pretrained_f8_h16_no_async"
)
JOB_ASSET_DIRS=(
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
)
JOB_WEIGHT_ARGS=(
  "--weight-loader.pi0-params-path $PATCH_POLICY_PARAMS --weight-loader.encoder-params-path $PATCH_ENCODER_PARAMS"
  "--weight-loader.pi0-params-path $PATCH_POLICY_PARAMS --weight-loader.encoder-params-path $PATCH_ENCODER_PARAMS"
  "--weight-loader.params-path $PI0_BASE_PARAMS"
  "--weight-loader.pi0-params-path $PATCH_POLICY_PARAMS --weight-loader.encoder-params-path $PATCH_ENCODER_PARAMS"
  "--weight-loader.pi0-params-path $PATCH_POLICY_PARAMS --weight-loader.encoder-params-path $PATCH_ENCODER_PARAMS"
)

source scripts/four_gpu_training_queue.sh
run_four_gpu_training_queue "${1:-data/task12345-2}"
