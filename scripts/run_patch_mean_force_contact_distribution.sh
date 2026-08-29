#!/usr/bin/env bash
set -Eeuo pipefail

# Train the physically consistent patch mean-force/contact/distribution encoder,
# then evaluate its final checkpoint on the held-out episode split.
#
# Usage:
#   bash scripts/run_patch_mean_force_contact_distribution.sh data/taskall-2
#
# Background:
#   setsid nohup env GPU=0 RUN_TAG=taskall2_mfcd_seed42 \
#     bash scripts/run_patch_mean_force_contact_distribution.sh data/taskall-2 \
#     > logs/taskall2_mfcd_seed42.pipeline.log 2>&1 &

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_REPO="${1:-data/taskall-2}"
ASSET_ID="${DATA_ASSET_ID:-$(basename "$DATA_REPO")}"
ASSET_DIR="${ASSET_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
SPLIT_DIR="${SPLIT_DIR:-outputs/episode_splits/taskall-2_encoder_final_10pct_seed42}"
TRAIN_FILTER_PATH="${TRAIN_FILTER_PATH:-${SPLIT_DIR}/train_episodes.json}"
EVAL_FILTER_PATH="${EVAL_FILTER_PATH:-${SPLIT_DIR}/val_episodes.json}"
PYTHON_BIN="${PYTHON:-env/.venv/bin/python}"
GPU="${GPU:-0}"

CONFIG="xhand_patch_mean_force_contact_distribution_encoder_pretrain"
RUN_TAG="${RUN_TAG:-taskall2_mfcd_seed42_$(date +%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-${CONFIG}_${RUN_TAG}}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-20000}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RESUME="${RESUME:-0}"

FINAL_STEP=$((TRAIN_STEPS - 1))
CHECKPOINT_DIR="checkpoints/${CONFIG}/${EXP_NAME}"
FINAL_PARAMS="${CHECKPOINT_DIR}/${FINAL_STEP}/params"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/patch_mean_force_encoder_eval/${EXP_NAME}_step${FINAL_STEP}_val}"
TRAIN_LOG="${TRAIN_LOG:-logs/${EXP_NAME}.train.log}"
EVAL_LOG="${EVAL_LOG:-logs/${EXP_NAME}.eval.log}"

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

cd "$PROJECT_ROOT"
mkdir -p logs .hf_datasets_cache .cache/huggingface

require_path "$PYTHON_BIN" "python not found"
require_path "$DATA_REPO/meta/info.json" "dataset not found"
require_path "$ASSET_DIR/$ASSET_ID/norm_stats.json" "normalization statistics not found"
require_path "$TRAIN_FILTER_PATH" "training split not found"
require_path "$EVAL_FILTER_PATH" "validation/test split not found"

if [[ "$ALLOW_OVERWRITE" != "0" && "$ALLOW_OVERWRITE" != "1" ]]; then
  echo "ERROR: ALLOW_OVERWRITE must be 0 or 1." >&2
  exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "ERROR: RESUME must be 0 or 1." >&2
  exit 2
fi
if [[ "$ALLOW_OVERWRITE" == "1" && "$RESUME" == "1" ]]; then
  echo "ERROR: ALLOW_OVERWRITE and RESUME cannot both be 1." >&2
  exit 2
fi
if [[ -e "$CHECKPOINT_DIR" && "$ALLOW_OVERWRITE" != "1" && "$RESUME" != "1" ]]; then
  echo "ERROR: checkpoint directory already exists: $CHECKPOINT_DIR" >&2
  echo "Use a new RUN_TAG, or set RESUME=1." >&2
  exit 2
fi

run_mode_args=()
if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
  run_mode_args=(--overwrite)
elif [[ "$RESUME" == "1" && -e "$CHECKPOINT_DIR" ]]; then
  run_mode_args=(--resume)
fi

log "Training ${CONFIG} for ${TRAIN_STEPS} steps."
log "Train split: ${TRAIN_FILTER_PATH}"
log "Held-out split: ${EVAL_FILTER_PATH}"

CUDA_VISIBLE_DEVICES="$GPU" \
HF_LEROBOT_HOME="$PROJECT_ROOT" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_datasets_cache}" \
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$PYTHON_BIN" scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --data.repo-id "$DATA_REPO" \
  --data.assets.asset-id "$ASSET_ID" \
  --data.assets.assets-dir "$ASSET_DIR" \
  --train-filter-path "$TRAIN_FILTER_PATH" \
  --num-train-steps "$TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices 1 \
  --num-workers "$NUM_WORKERS" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-num-batches "$EVAL_NUM_BATCHES" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-num-workers "$NUM_WORKERS" \
  --eval-repo-id "$DATA_REPO" \
  --eval-asset-id "$ASSET_ID" \
  --eval-assets-dir "$ASSET_DIR" \
  --eval-filter-path "$EVAL_FILTER_PATH" \
  --no-wandb-enabled \
  "${run_mode_args[@]}" \
  > "$TRAIN_LOG" 2>&1

require_path "$FINAL_PARAMS" "final checkpoint not found after training"

log "Running final held-out evaluation from ${FINAL_PARAMS}."
CUDA_VISIBLE_DEVICES="$GPU" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$PYTHON_BIN" scripts/eval_patch_mean_force_encoder.py \
  --repo-id "$DATA_REPO" \
  --params "$FINAL_PARAMS" \
  --config-name "$CONFIG" \
  --filter-path "$EVAL_FILTER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-frames "$MAX_EVAL_FRAMES" \
  --batch-size 256 \
  > "$EVAL_LOG" 2>&1

log "Done. Checkpoint: ${FINAL_PARAMS}"
log "Evaluation: ${OUTPUT_DIR}/metrics.json"
