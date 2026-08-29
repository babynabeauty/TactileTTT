#!/usr/bin/env bash
set -Eeuo pipefail

# Full fresh-reset async training pipeline for taskall-2.
#
# Default stages:
#   1. Reuse an existing patch encoder checkpoint.
#   2. Train fresh async policy with the patch encoder frozen.
#   3. Continue fresh async policy finetuning with the patch encoder unfrozen.
#
# Default target:
#   data/taskall-2, f4/h16 async-fresh, 8 GPUs, global batch 8.
#
# Usage:
#   bash scripts/run_taskall2_fresh_async_pipeline.sh
#
# Background:
#   setsid nohup env RUN_TAG=taskall2_fresh_f4_$(date +%m%d_%H%M) \
#     bash scripts/run_taskall2_fresh_async_pipeline.sh \
#     > logs/taskall2_fresh_async_pipeline.log 2>&1 &
#
# Resume:
#   setsid nohup env RUN_TAG=<same_tag> RESUME=1 \
#     bash scripts/run_taskall2_fresh_async_pipeline.sh \
#     > logs/taskall2_fresh_async_pipeline_resume.log 2>&1 &
#
# Useful overrides:
#   DATA_REPO=data/taskall-2
#   FRESH_VARIANT=f4 or f8
#   SKIP_STAGE1=0 to train a new patch encoder first.
#   PATCH_ENCODER_PARAMS=checkpoints/xhand_patch_tactile_encoder_pretrain/<exp>/<step>/params
#   STAGE1_CONFIG=xhand_patch_tactile_encoder_pretrain
#   STAGE1_CONFIG=xhand_patch_mean_force_encoder_pretrain
#   STAGE1_STEPS=20000
#   STAGE2A_STEPS=5000
#   STAGE2B_STEPS=90000

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_REPO="${1:-${DATA_REPO:-data/taskall-2}}"
ASSET_ID="${DATA_ASSET_ID:-$(basename "$DATA_REPO")}"
ASSET_DIR="${ASSET_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
PYTHON_BIN="${PYTHON:-env/.venv/bin/python}"
BASE_PARAMS="${BASE_PARAMS:-checkpoints/pi0_base/params}"
FRESH_VARIANT="${FRESH_VARIANT:-f4}"
RUN_TAG="${RUN_TAG:-${ASSET_ID}_fresh_${FRESH_VARIANT}_$(date +%m%d_%H%M%S)}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-64}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RESUME="${RESUME:-0}"

STAGE1_CONFIG="${STAGE1_CONFIG:-xhand_patch_mean_force_encoder_pretrain}"
SKIP_STAGE1="${SKIP_STAGE1:-1}"
PATCH_ENCODER_PARAMS="${PATCH_ENCODER_PARAMS:-checkpoints/xhand_patch_tactile_encoder_pretrain/xhand_patch_tactile_encoder_pretrain_taskall2_patch_pretrained_f8_async_90k_0717/19999/params}"
STAGE1_STEPS="${STAGE1_STEPS:-20000}"
STAGE1_SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL:-5000}"
STAGE1_KEEP_PERIOD="${STAGE1_KEEP_PERIOD:-5000}"
STAGE1_HISTORY_TIME_SAMPLES="${STAGE1_HISTORY_TIME_SAMPLES:-2}"
STAGE1_FUTURE_TIME_SAMPLES="${STAGE1_FUTURE_TIME_SAMPLES:-2}"

STAGE2A_STEPS="${STAGE2A_STEPS:-5000}"
STAGE2B_STEPS="${STAGE2B_STEPS:-90000}"
STAGE2A_SAVE_INTERVAL="${STAGE2A_SAVE_INTERVAL:-5000}"
STAGE2B_SAVE_INTERVAL="${STAGE2B_SAVE_INTERVAL:-10000}"
STAGE2_KEEP_PERIOD="${STAGE2_KEEP_PERIOD:-10000}"
STAGE2_FSDP_DEVICES="${STAGE2_FSDP_DEVICES:-4}"

case "$FRESH_VARIANT" in
  f4)
    CONFIG_STAGE2A="pi0_xhand_dual_patch_pretrained_f4_h16_async_fresh_freeze"
    CONFIG_STAGE2B="pi0_xhand_dual_patch_pretrained_f4_h16_async_fresh"
    ;;
  f8)
    CONFIG_STAGE2A="pi0_xhand_dual_patch_pretrained_f8_h16_async_fresh_freeze"
    CONFIG_STAGE2B="pi0_xhand_dual_patch_pretrained_f8_h16_async_fresh"
    ;;
  *)
    echo "ERROR: FRESH_VARIANT must be f4 or f8, got: $FRESH_VARIANT" >&2
    exit 2
    ;;
esac

EXP_STAGE1="${STAGE1_CONFIG}_${RUN_TAG}"
EXP_STAGE2A="${CONFIG_STAGE2A}_${RUN_TAG}"
EXP_STAGE2B="${CONFIG_STAGE2B}_${RUN_TAG}"

FINAL_STAGE1=$((STAGE1_STEPS - 1))
FINAL_STAGE2A=$((STAGE2A_STEPS - 1))
FINAL_STAGE2B=$((STAGE2B_STEPS - 1))

CKPT_STAGE1="checkpoints/${STAGE1_CONFIG}/${EXP_STAGE1}/${FINAL_STAGE1}/params"
if [[ "$SKIP_STAGE1" == "1" ]]; then
  CKPT_STAGE1="$PATCH_ENCODER_PARAMS"
fi
CKPT_STAGE2A="checkpoints/${CONFIG_STAGE2A}/${EXP_STAGE2A}/${FINAL_STAGE2A}/params"
CKPT_STAGE2B="checkpoints/${CONFIG_STAGE2B}/${EXP_STAGE2B}/${FINAL_STAGE2B}/params"

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

check_output_dir() {
  local config="$1"
  local exp="$2"
  local dir="checkpoints/${config}/${exp}"
  if [[ "$RESUME" == "1" ]]; then
    return 0
  fi
  if [[ -e "$dir" && "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "ERROR: checkpoint dir already exists: $dir" >&2
    echo "Use a new RUN_TAG, set RESUME=1, or set ALLOW_OVERWRITE=1." >&2
    exit 2
  fi
}

run_train() {
  local config="$1"
  local exp="$2"
  local steps="$3"
  local save_interval="$4"
  local keep_period="$5"
  local fsdp_devices="$6"
  local batch_size="$7"
  local log_file="$8"
  shift 8

  local checkpoint_dir="checkpoints/${config}/${exp}"
  local final_params="${checkpoint_dir}/$((steps - 1))/params"
  if [[ -e "$final_params" ]]; then
    log "Skipping ${config}: final checkpoint already exists at ${final_params}"
    return 0
  fi

  local run_mode_args=()
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    run_mode_args=(--overwrite)
  elif [[ "$RESUME" == "1" && -e "$checkpoint_dir" ]]; then
    run_mode_args=(--resume)
  fi

  log "Starting ${config}: exp=${exp}, steps=${steps}, batch=${batch_size}, fsdp=${fsdp_devices}, mode=${run_mode_args[*]:-new}"
  log "Log file: ${log_file}"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_datasets_cache}" \
  HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON_BIN" scripts/train.py "$config" \
    --exp-name "$exp" \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir "$ASSET_DIR" \
    --num-train-steps "$steps" \
    --batch-size "$batch_size" \
    --fsdp-devices "$fsdp_devices" \
    --num-workers "$NUM_WORKERS" \
    --save-interval "$save_interval" \
    --keep-period "$keep_period" \
    --no-wandb-enabled \
    "${run_mode_args[@]}" \
    "$@" \
    > "$log_file" 2>&1
  log "Finished ${config}: exp=${exp}"
}

cd "$PROJECT_ROOT"
mkdir -p logs .hf_datasets_cache .cache/huggingface

if [[ "$ALLOW_OVERWRITE" != "0" && "$ALLOW_OVERWRITE" != "1" ]]; then
  echo "ERROR: ALLOW_OVERWRITE must be 0 or 1, got: $ALLOW_OVERWRITE" >&2
  exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "ERROR: RESUME must be 0 or 1, got: $RESUME" >&2
  exit 2
fi
if [[ "$SKIP_STAGE1" != "0" && "$SKIP_STAGE1" != "1" ]]; then
  echo "ERROR: SKIP_STAGE1 must be 0 or 1, got: $SKIP_STAGE1" >&2
  exit 2
fi
if [[ "$ALLOW_OVERWRITE" == "1" && "$RESUME" == "1" ]]; then
  echo "ERROR: ALLOW_OVERWRITE=1 and RESUME=1 cannot be used together." >&2
  exit 2
fi
if (( STAGE1_STEPS <= 0 || STAGE2A_STEPS <= 0 || STAGE2B_STEPS <= 0 )); then
  echo "ERROR: all stage step counts must be positive." >&2
  exit 2
fi

require_path "$PYTHON_BIN" "python not found"
require_path "$DATA_REPO/meta/info.json" "dataset not found"
require_path "$BASE_PARAMS" "pi0 base params not found"
if [[ "$SKIP_STAGE1" == "1" ]]; then
  require_path "$PATCH_ENCODER_PARAMS" "patch encoder params not found"
fi

RAW_NORM_PATH="$ASSET_DIR/$ASSET_ID/norm_stats.json"
if [[ ! -f "$RAW_NORM_PATH" ]]; then
  log "Raw tactile norm stats not found; computing first: $RAW_NORM_PATH"
  NORM_CONFIGS="pi0_xhand_tactile_structured_raw_dual_ae" \
  ASSET_ID="$ASSET_ID" \
  bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"
else
  log "Raw tactile norm stats already exist: $RAW_NORM_PATH"
fi
require_path "$RAW_NORM_PATH" "raw tactile norm stats not found after normalization"

if [[ "$SKIP_STAGE1" == "0" ]]; then
  check_output_dir "$STAGE1_CONFIG" "$EXP_STAGE1"
fi
check_output_dir "$CONFIG_STAGE2A" "$EXP_STAGE2A"
check_output_dir "$CONFIG_STAGE2B" "$EXP_STAGE2B"

log "Dataset: $DATA_REPO"
log "Asset ID: $ASSET_ID"
log "Asset dir: $ASSET_DIR"
log "GPUs: $GPUS"
log "Run tag: $RUN_TAG"
log "Fresh variant: $FRESH_VARIANT"
log "Skip Stage1: $SKIP_STAGE1"
log "Stage1 config: $STAGE1_CONFIG"
log "Patch encoder params: $CKPT_STAGE1"
log "Stage2A config: $CONFIG_STAGE2A"
log "Stage2B config: $CONFIG_STAGE2B"
log "Stage2A final policy params: $CKPT_STAGE2A"
log "Stage2B final policy params: $CKPT_STAGE2B"

if [[ "$SKIP_STAGE1" == "0" ]]; then
  run_train \
    "$STAGE1_CONFIG" \
    "$EXP_STAGE1" \
    "$STAGE1_STEPS" \
    "$STAGE1_SAVE_INTERVAL" \
    "$STAGE1_KEEP_PERIOD" \
    1 \
    "$STAGE1_BATCH_SIZE" \
    "logs/${EXP_STAGE1}.log" \
    --model.pretrain-history-time-samples "$STAGE1_HISTORY_TIME_SAMPLES" \
    --model.pretrain-future-time-samples "$STAGE1_FUTURE_TIME_SAMPLES"
else
  log "Skipping Stage1 and reusing patch encoder: $CKPT_STAGE1"
fi

require_path "$CKPT_STAGE1" "stage1 encoder checkpoint was not produced"

run_train \
  "$CONFIG_STAGE2A" \
  "$EXP_STAGE2A" \
  "$STAGE2A_STEPS" \
  "$STAGE2A_SAVE_INTERVAL" \
  "$STAGE2_KEEP_PERIOD" \
  "$STAGE2_FSDP_DEVICES" \
  "$STAGE2_BATCH_SIZE" \
  "logs/${EXP_STAGE2A}.log" \
  --weight-loader.pi0-params-path "$BASE_PARAMS" \
  --weight-loader.encoder-params-path "$CKPT_STAGE1"

require_path "$CKPT_STAGE2A" "stage2A policy checkpoint was not produced"

run_train \
  "$CONFIG_STAGE2B" \
  "$EXP_STAGE2B" \
  "$STAGE2B_STEPS" \
  "$STAGE2B_SAVE_INTERVAL" \
  "$STAGE2_KEEP_PERIOD" \
  "$STAGE2_FSDP_DEVICES" \
  "$STAGE2_BATCH_SIZE" \
  "logs/${EXP_STAGE2B}.log" \
  --weight-loader.pi0-params-path "$CKPT_STAGE2A" \
  --weight-loader.encoder-params-path "$CKPT_STAGE1"

log "All stages completed."
log "Final checkpoint: $CKPT_STAGE2B"
