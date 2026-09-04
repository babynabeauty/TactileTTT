#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential TactileTTT v1/v2 training (select with TTT_VERSION):
#   stage 1: train only TTT parameters
#   stage 2: load stage-1 params into a fresh run and jointly train all parameters
#
# Run this script from anywhere.  It deliberately never uses --resume between
# stages because the warm-up and joint optimizer state trees are different.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/mnt/sqzhang26/FactileLDM/env/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

DATA_REPO="${DATA_REPO:-data/press_button_4_times}"
ASSET_ID="${ASSET_ID:-press_button_4_times}"
ASSET_DIR="${ASSET_DIR:-assets/pi05_tactile_current}"
TRAIN_SPLIT="${TRAIN_SPLIT:-outputs/episode_splits/press_button_4_times/train_episodes.json}"
VAL_SPLIT="${VAL_SPLIT:-outputs/episode_splits/press_button_4_times/val_episodes.json}"
PATCH_ENCODER_PARAMS="${PATCH_ENCODER_PARAMS:-/workspace/mnt/sqzhang26/FactileLDM/checkpoints/xhand_patch_tactile_encoder_pretrain/patch_informed_full_heads_taskall2_encoder_final_20k_0722/19999/params}"

RUN_TAG="${RUN_TAG:-studentonly_$(date +%m%d_%H%M%S)}"
TTT_VERSION="${TTT_VERSION:-v1}"
if [[ "$TTT_VERSION" != "v1" && "$TTT_VERSION" != "v2" ]]; then
  printf 'ERROR: TTT_VERSION must be v1 or v2, got %s\n' "$TTT_VERSION" >&2
  exit 2
fi
WARMUP_CONFIG="pi05_tactile_ttt_${TTT_VERSION}_warmup"
JOINT_CONFIG="pi05_tactile_ttt_${TTT_VERSION}"
WARMUP_EXP="${WARMUP_EXP:-pi05_tactile_ttt_${TTT_VERSION}_warmup_${RUN_TAG}}"
JOINT_EXP="${JOINT_EXP:-pi05_tactile_ttt_${TTT_VERSION}_twostage_joint_${RUN_TAG}}"

WARMUP_STEPS="${WARMUP_STEPS:-250}"
JOINT_STEPS="${JOINT_STEPS:-750}"
WARMUP_LR_STEPS="${WARMUP_LR_STEPS:-25}"
JOINT_LR_STEPS="${JOINT_LR_STEPS:-75}"
PEAK_LR="${PEAK_LR:-2.5e-5}"
DECAY_LR="${DECAY_LR:-2.5e-6}"
WARMUP_SAVE_INTERVAL="${WARMUP_SAVE_INTERVAL:-125}"
JOINT_SAVE_INTERVAL="${JOINT_SAVE_INTERVAL:-250}"
WARMUP_EVAL_INTERVAL="${WARMUP_EVAL_INTERVAL:-125}"
JOINT_EVAL_INTERVAL="${JOINT_EVAL_INTERVAL:-250}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-2}"

WARMUP_FINAL_STEP=$((WARMUP_STEPS - 1))
JOINT_FINAL_STEP=$((JOINT_STEPS - 1))
WARMUP_DIR="$PROJECT_ROOT/checkpoints/$WARMUP_CONFIG/$WARMUP_EXP"
JOINT_DIR="$PROJECT_ROOT/checkpoints/$JOINT_CONFIG/$JOINT_EXP"
WARMUP_PARAMS="$WARMUP_DIR/$WARMUP_FINAL_STEP/params"
JOINT_PARAMS="$JOINT_DIR/$JOINT_FINAL_STEP/params"
WARMUP_LOG="$PROJECT_ROOT/logs/${WARMUP_EXP}.log"
JOINT_LOG="$PROJECT_ROOT/logs/${JOINT_EXP}.log"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_datasets_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "ERROR: required file does not exist: $path" >&2
    exit 2
  fi
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log "ERROR: required path does not exist: $path" >&2
    exit 2
  fi
}

check_new_output_dir() {
  local path="$1"
  if [[ -e "$path" ]]; then
    log "ERROR: checkpoint directory already exists: $path" >&2
    log "Use a new RUN_TAG. This script will not overwrite or resume it." >&2
    exit 2
  fi
}

run_training_stage() {
  local stage_name="$1"
  local log_file="$2"
  shift 2

  log "Starting $stage_name; detailed log: $log_file"
  if CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" scripts/train.py "$@" >"$log_file" 2>&1; then
    log "Completed $stage_name"
  else
    local exit_code=$?
    log "ERROR: $stage_name failed with exit code $exit_code"
    log "Last 80 log lines follow:"
    tail -n 80 "$log_file" || true
    return "$exit_code"
  fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  log "ERROR: Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if ((WARMUP_STEPS <= 0 || JOINT_STEPS <= 0)); then
  log "ERROR: WARMUP_STEPS and JOINT_STEPS must both be positive." >&2
  exit 2
fi

require_file "$TRAIN_SPLIT"
require_file "$VAL_SPLIT"
require_path "$ASSET_DIR"
require_path "$PATCH_ENCODER_PARAMS"
check_new_output_dir "$WARMUP_DIR"
check_new_output_dir "$JOINT_DIR"
mkdir -p logs "$HF_DATASETS_CACHE"

log "Run tag: $RUN_TAG"
log "TactileTTT version: $TTT_VERSION"
log "GPUs: $GPU_IDS; FSDP devices: $FSDP_DEVICES; global batch: $BATCH_SIZE"
log "Stage 1: $WARMUP_CONFIG/$WARMUP_EXP ($WARMUP_STEPS steps)"
log "Stage 2: $JOINT_CONFIG/$JOINT_EXP ($JOINT_STEPS steps)"

run_training_stage "stage 1 TTT-only warm-up" "$WARMUP_LOG" \
  "$WARMUP_CONFIG" \
  --exp-name "$WARMUP_EXP" \
  --data.repo-id "$DATA_REPO" \
  --data.assets.asset-id "$ASSET_ID" \
  --data.assets.assets-dir "$ASSET_DIR" \
  --train-filter-path "$TRAIN_SPLIT" \
  --weight-loader.encoder-params-path "$PATCH_ENCODER_PARAMS" \
  --num-train-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --lr-schedule.warmup-steps "$WARMUP_LR_STEPS" \
  --lr-schedule.peak-lr "$PEAK_LR" \
  --lr-schedule.decay-steps "$WARMUP_STEPS" \
  --lr-schedule.decay-lr "$DECAY_LR" \
  --save-interval "$WARMUP_SAVE_INTERVAL" \
  --keep-period "$WARMUP_SAVE_INTERVAL" \
  --eval-interval "$WARMUP_EVAL_INTERVAL" \
  --eval-num-batches "$EVAL_NUM_BATCHES" \
  --eval-batch-size "$BATCH_SIZE" \
  --eval-num-workers "$NUM_WORKERS" \
  --eval-repo-id "$DATA_REPO" \
  --eval-asset-id "$ASSET_ID" \
  --eval-assets-dir "$ASSET_DIR" \
  --eval-filter-path "$VAL_SPLIT" \
  --no-wandb-enabled

require_path "$WARMUP_PARAMS"
log "Warm-up params verified: $WARMUP_PARAMS"
log "Starting a fresh joint run (intentionally not using --resume)"

run_training_stage "stage 2 joint training" "$JOINT_LOG" \
  "$JOINT_CONFIG" \
  --exp-name "$JOINT_EXP" \
  --data.repo-id "$DATA_REPO" \
  --data.assets.asset-id "$ASSET_ID" \
  --data.assets.assets-dir "$ASSET_DIR" \
  --train-filter-path "$TRAIN_SPLIT" \
  --weight-loader.pi0-params-path "$WARMUP_PARAMS" \
  --weight-loader.encoder-params-path "$PATCH_ENCODER_PARAMS" \
  --num-train-steps "$JOINT_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --lr-schedule.warmup-steps "$JOINT_LR_STEPS" \
  --lr-schedule.peak-lr "$PEAK_LR" \
  --lr-schedule.decay-steps "$JOINT_STEPS" \
  --lr-schedule.decay-lr "$DECAY_LR" \
  --save-interval "$JOINT_SAVE_INTERVAL" \
  --keep-period "$JOINT_SAVE_INTERVAL" \
  --eval-interval "$JOINT_EVAL_INTERVAL" \
  --eval-num-batches "$EVAL_NUM_BATCHES" \
  --eval-batch-size "$BATCH_SIZE" \
  --eval-num-workers "$NUM_WORKERS" \
  --eval-repo-id "$DATA_REPO" \
  --eval-asset-id "$ASSET_ID" \
  --eval-assets-dir "$ASSET_DIR" \
  --eval-filter-path "$VAL_SPLIT" \
  --no-wandb-enabled

require_path "$JOINT_PARAMS"
log "Two-stage TactileTTT training completed successfully."
log "Final params: $JOINT_PARAMS"
