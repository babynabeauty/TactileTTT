#!/usr/bin/env bash
set -euo pipefail

# Three-stage training pipeline:
#   1. xhand_patch_tactile_encoder_pretrain
#   2. pi0_xhand_dual_patch_pretrained_f4_h16_async_freeze by default
#   3. pi0_xhand_dual_patch_pretrained_f4_h16_async by default
# Set STAGE2_BASE_CONFIG=pi0_xhand_dual_patch_f8_h16_async to run the f8 policy variant.
#
# Usage:
#   bash scripts/run_patch_encoder_pretrain_pipeline.sh data/stage12345-2
#
# Background usage:
#   setsid nohup env RUN_TAG=stage12345_$(date +%m%d_%H%M) \
#     bash scripts/run_patch_encoder_pretrain_pipeline.sh data/stage12345-2 \
#     > logs/patch_encoder_pretrain_pipeline.log 2>&1 &
#
# Resume an interrupted run:
#   setsid nohup env RUN_TAG=stage12345_... RESUME=1 \
#     bash scripts/run_patch_encoder_pretrain_pipeline.sh data/stage12345-2 \
#     > logs/patch_encoder_pretrain_pipeline_resume.log 2>&1 &

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_REPO="${1:-data/stage12345-2}"
ASSET_ID="${DATA_ASSET_ID:-$(basename "$DATA_REPO")}"
ASSET_DIR="${ASSET_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
PYTHON_BIN="${PYTHON:-env/.venv/bin/python}"
BASE_PARAMS="${BASE_PARAMS:-checkpoints/pi0_base/params}"
RUN_TAG="${RUN_TAG:-${ASSET_ID}_patch_pretrain_$(date +%m%d_%H%M%S)}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-$GLOBAL_BATCH_SIZE}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-$GLOBAL_BATCH_SIZE}"
STAGE1_HISTORY_TIME_SAMPLES="${STAGE1_HISTORY_TIME_SAMPLES:-2}"
STAGE1_FUTURE_TIME_SAMPLES="${STAGE1_FUTURE_TIME_SAMPLES:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RESUME="${RESUME:-0}"

STAGE1_STEPS="${STAGE1_STEPS:-20000}"
STAGE2A_STEPS="${STAGE2A_STEPS:-5000}"
STAGE2B_STEPS="${STAGE2B_STEPS:-60000}"
STAGE1_SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL:-5000}"
STAGE2A_SAVE_INTERVAL="${STAGE2A_SAVE_INTERVAL:-5000}"
STAGE2B_SAVE_INTERVAL="${STAGE2B_SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"

CONFIG_STAGE1="xhand_patch_tactile_encoder_pretrain"
STAGE2_BASE_CONFIG="${STAGE2_BASE_CONFIG:-pi0_xhand_dual_patch_f4_h16_async}"
CONFIG_STAGE2A="${STAGE2_BASE_CONFIG/pi0_xhand_dual_patch_/pi0_xhand_dual_patch_pretrained_}_freeze"
CONFIG_STAGE2B="${STAGE2_BASE_CONFIG/pi0_xhand_dual_patch_/pi0_xhand_dual_patch_pretrained_}"

EXP_STAGE1="xhand_patch_tactile_encoder_pretrain_${RUN_TAG}"
EXP_STAGE2A="${CONFIG_STAGE2A}_${RUN_TAG}"
EXP_STAGE2B="${CONFIG_STAGE2B}_${RUN_TAG}"

FINAL_STAGE1=$((STAGE1_STEPS - 1))
FINAL_STAGE2A=$((STAGE2A_STEPS - 1))

CKPT_STAGE1="checkpoints/${CONFIG_STAGE1}/${EXP_STAGE1}/${FINAL_STAGE1}/params"
CKPT_STAGE2A="checkpoints/${CONFIG_STAGE2A}/${EXP_STAGE2A}/${FINAL_STAGE2A}/params"
CKPT_STAGE2B="checkpoints/${CONFIG_STAGE2B}/${EXP_STAGE2B}/$((STAGE2B_STEPS - 1))/params"

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
    echo "Use a new RUN_TAG or set ALLOW_OVERWRITE=1." >&2
    exit 2
  fi
}

run_train() {
  local config="$1"
  local exp="$2"
  local steps="$3"
  local save_interval="$4"
  local fsdp_devices="$5"
  local batch_size="$6"
  local log_file="$7"
  shift 7

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

  log "Starting ${config}: exp=${exp}, steps=${steps}, batch=${batch_size}, fsdp=${fsdp_devices}, mode=${run_mode_args[*]:-new}, log=${log_file}"
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
    --keep-period "$KEEP_PERIOD" \
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
require_path "$ASSET_DIR/$ASSET_ID/norm_stats.json" "norm stats not found"
require_path "$BASE_PARAMS" "pi0 base params not found"

check_output_dir "$CONFIG_STAGE1" "$EXP_STAGE1"
check_output_dir "$CONFIG_STAGE2A" "$EXP_STAGE2A"
check_output_dir "$CONFIG_STAGE2B" "$EXP_STAGE2B"

log "Dataset: $DATA_REPO"
log "Asset ID: $ASSET_ID"
log "Asset dir: $ASSET_DIR"
log "GPUs: $GPUS"
log "Global batch size: $GLOBAL_BATCH_SIZE"
log "Stage1 batch size: $STAGE1_BATCH_SIZE"
log "Stage1 tactile time samples: history=${STAGE1_HISTORY_TIME_SAMPLES}, future=${STAGE1_FUTURE_TIME_SAMPLES}"
log "Stage2 batch size: $STAGE2_BATCH_SIZE"
log "Stage2 base config: $STAGE2_BASE_CONFIG"
log "Stage2A config: $CONFIG_STAGE2A"
log "Stage2B config: $CONFIG_STAGE2B"
log "Run tag: $RUN_TAG"
log "Resume: $RESUME"
log "Stage1 final encoder params will be: $CKPT_STAGE1"
log "Stage2A final policy params will be: $CKPT_STAGE2A"
log "Stage2B final policy params will be: $CKPT_STAGE2B"

run_train \
  "$CONFIG_STAGE1" \
  "$EXP_STAGE1" \
  "$STAGE1_STEPS" \
  "$STAGE1_SAVE_INTERVAL" \
  1 \
  "$STAGE1_BATCH_SIZE" \
  "logs/${EXP_STAGE1}.log" \
  --model.pretrain-history-time-samples "$STAGE1_HISTORY_TIME_SAMPLES" \
  --model.pretrain-future-time-samples "$STAGE1_FUTURE_TIME_SAMPLES"

require_path "$CKPT_STAGE1" "stage1 encoder checkpoint was not produced"

run_train \
  "$CONFIG_STAGE2A" \
  "$EXP_STAGE2A" \
  "$STAGE2A_STEPS" \
  "$STAGE2A_SAVE_INTERVAL" \
  4 \
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
  4 \
  "$STAGE2_BATCH_SIZE" \
  "logs/${EXP_STAGE2B}.log" \
  --weight-loader.pi0-params-path "$CKPT_STAGE2A" \
  --weight-loader.encoder-params-path "$CKPT_STAGE1"

log "All stages completed."
log "Stage1 log: logs/${EXP_STAGE1}.log"
log "Stage2A log: logs/${EXP_STAGE2A}.log"
log "Stage2B log: logs/${EXP_STAGE2B}.log"
