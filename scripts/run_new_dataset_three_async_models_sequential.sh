#!/usr/bin/env bash
set -Eeuo pipefail

# Normalize one complete LeRobot dataset, optionally pretrain one shared
# three-head patch encoder, then train three policy variants sequentially:
#   1. pi0_xhand_dual_patch_pretrained_f4_h16_async
#   2. pi0_xhand_dual_patch_pretrained_f8_h16_async
#   3. pi0_xhand_dual_patch_pretrained_f4_h16_async_fresh
#
# All dataset episodes are used for training. No train/val/test filter is
# created or passed to train.py.
#
# Default policy schedule:
#   Stage2A: 5k steps with the patch encoder frozen.
#   Stage2B: 30k steps with the patch encoder unfrozen.
#   Stage2B checkpoints: every 5k steps, plus the final step 29999.
#
# Usage:
#   setsid nohup env \
#     RUN_TAG=my_dataset_async_$(date +%m%d_%H%M) \
#     bash scripts/run_new_dataset_three_async_models_sequential.sh data/my_dataset \
#     > logs/my_dataset_three_async_models.pipeline.log 2>&1 &
#
# Reuse an existing encoder instead of pretraining a new one:
#   setsid nohup env \
#     RUN_TAG=my_dataset_async_$(date +%m%d_%H%M) \
#     SKIP_ENCODER_PRETRAIN=1 \
#     PATCH_ENCODER_PARAMS=checkpoints/xhand_patch_force_three_head_encoder_pretrain/<exp>/<step>/params \
#     bash scripts/run_new_dataset_three_async_models_sequential.sh data/my_dataset \
#     > logs/my_dataset_three_async_models.pipeline.log 2>&1 &
#
# Resume an interrupted pipeline with the same RUN_TAG:
#   setsid nohup env \
#     RUN_TAG=<existing_tag> RESUME=1 \
#     bash scripts/run_new_dataset_three_async_models_sequential.sh data/my_dataset \
#     > logs/my_dataset_three_async_models.resume.pipeline.log 2>&1 &

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_REPO="${1:-${DATA_REPO:-}}"
if [[ -z "$DATA_REPO" ]]; then
  echo "Usage: bash scripts/run_new_dataset_three_async_models_sequential.sh data/your_dataset" >&2
  exit 2
fi

ASSET_ID="${DATA_ASSET_ID:-$(basename "$DATA_REPO")}"
ASSET_DIR="${ASSET_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
PYTHON_BIN="${PYTHON_BIN:-env/.venv/bin/python}"
BASE_PARAMS="${BASE_PARAMS:-checkpoints/pi0_base/params}"
RUN_TAG="${RUN_TAG:-${ASSET_ID}_three_async_$(date +%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

OVERWRITE_NORM="${OVERWRITE_NORM:-0}"
NORM_BATCH_SIZE="${NORM_BATCH_SIZE:-64}"
NORM_NUM_WORKERS="${NORM_NUM_WORKERS:-2}"
NORM_MIN_FREE_MEMORY_MB="${NORM_MIN_FREE_MEMORY_MB:-10000}"
NORM_MAX_GPU_UTIL="${NORM_MAX_GPU_UTIL:-100}"
NORM_POLL_INTERVAL="${NORM_POLL_INTERVAL:-10}"

SKIP_ENCODER_PRETRAIN="${SKIP_ENCODER_PRETRAIN:-0}"
PATCH_ENCODER_PARAMS="${PATCH_ENCODER_PARAMS:-}"
ENCODER_CONFIG="${ENCODER_CONFIG:-xhand_patch_force_three_head_encoder_pretrain}"
ENCODER_STEPS="${ENCODER_STEPS:-20000}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-32}"
ENCODER_FSDP_DEVICES="${ENCODER_FSDP_DEVICES:-1}"
ENCODER_SAVE_INTERVAL="${ENCODER_SAVE_INTERVAL:-5000}"

STAGE2A_STEPS="${STAGE2A_STEPS:-5000}"
STAGE2B_STEPS="${STAGE2B_STEPS:-30000}"
STAGE2A_SAVE_INTERVAL="${STAGE2A_SAVE_INTERVAL:-5000}"
STAGE2B_SAVE_INTERVAL="${STAGE2B_SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"

ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RESUME="${RESUME:-0}"

CONFIG_F4_FREEZE="pi0_xhand_dual_patch_pretrained_f4_h16_async_freeze"
CONFIG_F4="pi0_xhand_dual_patch_pretrained_f4_h16_async"
CONFIG_F8_FREEZE="pi0_xhand_dual_patch_pretrained_f8_h16_async_freeze"
CONFIG_F8="pi0_xhand_dual_patch_pretrained_f8_h16_async"
CONFIG_F4_FRESH_FREEZE="pi0_xhand_dual_patch_pretrained_f4_h16_async_fresh_freeze"
CONFIG_F4_FRESH="pi0_xhand_dual_patch_pretrained_f4_h16_async_fresh"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_path() {
  local path="$1"
  local description="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: ${description} not found: ${path}" >&2
    exit 2
  fi
}

validate_boolean() {
  local name="$1"
  local value="$2"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "ERROR: ${name} must be 0 or 1; got ${value}." >&2
    exit 2
  fi
}

check_output_dir() {
  local config="$1"
  local exp="$2"
  local output="checkpoints/${config}/${exp}"
  if [[ "$RESUME" == "1" ]]; then
    return 0
  fi
  if [[ -e "$output" && "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "ERROR: checkpoint directory already exists: ${output}" >&2
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

  local checkpoint_root="checkpoints/${config}/${exp}"
  local final_params="${checkpoint_root}/$((steps - 1))/params"
  if [[ -e "$final_params" ]]; then
    log "Skipping completed run: config=${config}, final=${final_params}"
    return 0
  fi

  local run_mode_args=()
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    run_mode_args=(--overwrite)
  elif [[ "$RESUME" == "1" && -e "$checkpoint_root" ]]; then
    run_mode_args=(--resume)
  fi

  log "Starting config=${config}"
  log "  exp=${exp}"
  log "  steps=${steps}, batch=${batch_size}, fsdp=${fsdp_devices}"
  log "  log=${log_file}"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_datasets_cache}" \
  HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" -u scripts/train.py "$config" \
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
    --eval-interval 0 \
    --no-wandb-enabled \
    "${run_mode_args[@]}" \
    "$@" \
    > "$log_file" 2>&1

  require_path "$final_params" "final checkpoint"
  log "Finished config=${config}; final=${final_params}"
}

run_policy_pair() {
  local label="$1"
  local freeze_config="$2"
  local finetune_config="$3"

  local freeze_exp="${freeze_config}_${RUN_TAG}"
  local finetune_exp="${finetune_config}_${RUN_TAG}"
  local freeze_final="checkpoints/${freeze_config}/${freeze_exp}/$((STAGE2A_STEPS - 1))/params"

  check_output_dir "$freeze_config" "$freeze_exp"
  check_output_dir "$finetune_config" "$finetune_exp"

  log "===== ${label}: frozen encoder warmup ====="
  run_train \
    "$freeze_config" \
    "$freeze_exp" \
    "$STAGE2A_STEPS" \
    "$STAGE2A_SAVE_INTERVAL" \
    "$KEEP_PERIOD" \
    "$FSDP_DEVICES" \
    "$GLOBAL_BATCH_SIZE" \
    "logs/${freeze_exp}.log" \
    --weight-loader.pi0-params-path "$BASE_PARAMS" \
    --weight-loader.encoder-params-path "$PATCH_ENCODER_PARAMS"

  require_path "$freeze_final" "${label} frozen-stage checkpoint"

  log "===== ${label}: end-to-end finetune ====="
  run_train \
    "$finetune_config" \
    "$finetune_exp" \
    "$STAGE2B_STEPS" \
    "$STAGE2B_SAVE_INTERVAL" \
    "$KEEP_PERIOD" \
    "$FSDP_DEVICES" \
    "$GLOBAL_BATCH_SIZE" \
    "logs/${finetune_exp}.log" \
    --weight-loader.pi0-params-path "$freeze_final" \
    --weight-loader.encoder-params-path "$PATCH_ENCODER_PARAMS"
}

cd "$PROJECT_ROOT"
mkdir -p logs .hf_datasets_cache .cache/huggingface

validate_boolean "OVERWRITE_NORM" "$OVERWRITE_NORM"
validate_boolean "SKIP_ENCODER_PRETRAIN" "$SKIP_ENCODER_PRETRAIN"
validate_boolean "ALLOW_OVERWRITE" "$ALLOW_OVERWRITE"
validate_boolean "RESUME" "$RESUME"
if [[ "$ALLOW_OVERWRITE" == "1" && "$RESUME" == "1" ]]; then
  echo "ERROR: ALLOW_OVERWRITE=1 and RESUME=1 cannot be used together." >&2
  exit 2
fi

require_path "$PYTHON_BIN" "Python interpreter"
require_path "$DATA_REPO/meta/info.json" "dataset metadata"
require_path "$BASE_PARAMS" "Pi0 base params"

IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
if (( ${#GPU_ID_LIST[@]} != 8 )); then
  echo "ERROR: GPU_IDS must contain exactly eight GPU IDs; got ${GPU_IDS}." >&2
  exit 2
fi
if (( 8 % FSDP_DEVICES != 0 )); then
  echo "ERROR: FSDP_DEVICES=${FSDP_DEVICES} must divide eight visible GPUs." >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % 8 != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by eight GPUs." >&2
  exit 2
fi
if (( ENCODER_BATCH_SIZE % 8 != 0 )); then
  echo "ERROR: ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE} must be divisible by eight GPUs." >&2
  exit 2
fi

log "Pipeline configuration"
log "  dataset=${DATA_REPO}"
log "  all episodes are used for training; no filter/split is enabled"
log "  asset=${ASSET_DIR}/${ASSET_ID}"
log "  GPUs=${GPU_IDS}, policy FSDP=${FSDP_DEVICES}, global batch=${GLOBAL_BATCH_SIZE}"
log "  policy schedule=${STAGE2A_STEPS} freeze + ${STAGE2B_STEPS} finetune"
log "  Stage2B save interval=${STAGE2B_SAVE_INTERVAL}"
log "  run tag=${RUN_TAG}"

log "===== Step 0: normalization ====="
NORM_CONFIGS="pi0_xhand_tactile_structured_raw_dual_ae" \
ASSET_ID="$ASSET_ID" \
GPU_IDS="$GPU_IDS" \
PYTHON="$PYTHON_BIN" \
NORM_BATCH_SIZE="$NORM_BATCH_SIZE" \
NORM_NUM_WORKERS="$NORM_NUM_WORKERS" \
MIN_FREE_MEMORY_MB="$NORM_MIN_FREE_MEMORY_MB" \
MAX_GPU_UTIL="$NORM_MAX_GPU_UTIL" \
POLL_INTERVAL="$NORM_POLL_INTERVAL" \
OVERWRITE_NORM="$OVERWRITE_NORM" \
RUN_TAG="${RUN_TAG}_norm" \
bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"

NORM_STATS_PATH="$ASSET_DIR/$ASSET_ID/norm_stats.json"
require_path "$NORM_STATS_PATH" "normalization statistics"
log "Normalization ready: ${NORM_STATS_PATH}"

if [[ "$SKIP_ENCODER_PRETRAIN" == "0" ]]; then
  ENCODER_EXP="${ENCODER_CONFIG}_${RUN_TAG}"
  check_output_dir "$ENCODER_CONFIG" "$ENCODER_EXP"

  log "===== Step 1: shared three-head patch encoder ====="
  run_train \
    "$ENCODER_CONFIG" \
    "$ENCODER_EXP" \
    "$ENCODER_STEPS" \
    "$ENCODER_SAVE_INTERVAL" \
    "$ENCODER_SAVE_INTERVAL" \
    "$ENCODER_FSDP_DEVICES" \
    "$ENCODER_BATCH_SIZE" \
    "logs/${ENCODER_EXP}.log"

  PATCH_ENCODER_PARAMS="checkpoints/${ENCODER_CONFIG}/${ENCODER_EXP}/$((ENCODER_STEPS - 1))/params"
else
  if [[ -z "$PATCH_ENCODER_PARAMS" ]]; then
    echo "ERROR: SKIP_ENCODER_PRETRAIN=1 requires PATCH_ENCODER_PARAMS." >&2
    exit 2
  fi
  require_path "$PATCH_ENCODER_PARAMS" "pretrained patch encoder params"
  log "Reusing patch encoder: ${PATCH_ENCODER_PARAMS}"
fi

require_path "$PATCH_ENCODER_PARAMS" "shared patch encoder params"
log "Shared patch encoder: ${PATCH_ENCODER_PARAMS}"

log "===== Step 2/4: F4 recursive policy ====="
run_policy_pair "F4 recursive" "$CONFIG_F4_FREEZE" "$CONFIG_F4"

log "===== Step 3/4: F8 recursive policy ====="
run_policy_pair "F8 recursive" "$CONFIG_F8_FREEZE" "$CONFIG_F8"

log "===== Step 4/4: F4 Fresh policy ====="
run_policy_pair "F4 Fresh" "$CONFIG_F4_FRESH_FREEZE" "$CONFIG_F4_FRESH"

log "All three policy variants completed successfully."
log "Final checkpoint roots:"
log "  checkpoints/${CONFIG_F4}/${CONFIG_F4}_${RUN_TAG}/$((STAGE2B_STEPS - 1))/params"
log "  checkpoints/${CONFIG_F8}/${CONFIG_F8}_${RUN_TAG}/$((STAGE2B_STEPS - 1))/params"
log "  checkpoints/${CONFIG_F4_FRESH}/${CONFIG_F4_FRESH}_${RUN_TAG}/$((STAGE2B_STEPS - 1))/params"
