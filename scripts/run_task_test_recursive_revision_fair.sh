#!/usr/bin/env bash
set -Eeuo pipefail

# Fair recursive-revision analysis on one fixed ReTouch checkpoint.
#
# The three inference behaviors are evaluated together in one process so they
# share model weights, samples, preprocessing, Teacher latents, and diffusion
# noise. At offset 0 their outputs must be identical; the evaluator enforces
# this as a sanity check.
#
# Server usage:
#   setsid nohup env GPU_IDS=0,1,2,3 \
#     bash scripts/run_task_test_recursive_revision_fair.sh \
#     > logs/task_test_recursive_revision_fair.pipeline.log 2>&1 &
#
# Useful overrides:
#   EVAL_REPO=data/task-test
#   CHECKPOINT=checkpoints/.../<step>
#   CONFIG_NAME=pi0_xhand_dual_patch_pretrained_f8_h16_async
#   TRAIN_ASSET_ID=taskall-2
#   OUTPUT_DIR=outputs/task_test_recursive_revision_fair
#   FILTER_PATH=outputs/.../test_episodes.json
#   MAX_BATCHES=10
#   INCLUDE_ACTION_ONLY=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

EVAL_REPO="${EVAL_REPO:-data/task-test}"
CONFIG_NAME="${CONFIG_NAME:-pi0_xhand_dual_patch_pretrained_f8_h16_async}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pi0_xhand_dual_patch_pretrained_f8_h16_async/pi0_xhand_dual_patch_pretrained_f8_h16_async_taskall2_patch_pretrained_f8_async_90k_0717/80000}"
MODEL_LABEL="${MODEL_LABEL:-fixed_retouch_f8_taskall2_80k}"
TRAIN_ASSET_ID="${TRAIN_ASSET_ID:-taskall-2}"
ASSETS_DIR="${ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/task_test_recursive_revision_fair}"
FILTER_PATH="${FILTER_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-env/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_BATCHES="${MAX_BATCHES:-0}"
NUM_STEPS="${NUM_STEPS:-10}"
SEED="${SEED:-42}"
INCLUDE_ACTION_ONLY="${INCLUDE_ACTION_ONLY:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

CONTACT_THRESHOLD="${CONTACT_THRESHOLD:-1.0}"
CONTACT_MIN_TAXELS="${CONTACT_MIN_TAXELS:-1}"
CONTACT_MIN_CONSECUTIVE_FRAMES="${CONTACT_MIN_CONSECUTIVE_FRAMES:-1}"

require_path() {
  local path="$1"
  local description="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: ${description} not found: ${path}" >&2
    exit 2
  fi
}

require_path "$PYTHON_BIN" "Python interpreter"
require_path "$EVAL_REPO/meta/info.json" "task-test metadata"
require_path "$CHECKPOINT" "fixed ReTouch checkpoint"
require_path "$ASSETS_DIR/$TRAIN_ASSET_ID/norm_stats.json" "training normalization stats"
if [[ -n "$FILTER_PATH" ]]; then
  require_path "$FILTER_PATH" "test episode filter"
fi
if [[ "$INCLUDE_ACTION_ONLY" != "0" && "$INCLUDE_ACTION_ONLY" != "1" ]]; then
  echo "ERROR: INCLUDE_ACTION_ONLY must be 0 or 1." >&2
  exit 2
fi
if [[ "$ALLOW_OVERWRITE" != "0" && "$ALLOW_OVERWRITE" != "1" ]]; then
  echo "ERROR: ALLOW_OVERWRITE must be 0 or 1." >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR/metrics.json" && "$ALLOW_OVERWRITE" != "1" ]]; then
  echo "ERROR: output already exists: $OUTPUT_DIR/metrics.json" >&2
  echo "Use a new OUTPUT_DIR or set ALLOW_OVERWRITE=1." >&2
  exit 2
fi

modes=(one_shot fresh_reinfer retouch)
if [[ "$INCLUDE_ACTION_ONLY" == "1" ]]; then
  modes=(one_shot action_only fresh_reinfer retouch)
fi

filter_args=()
if [[ -n "$FILTER_PATH" ]]; then
  filter_args=(--filter-path "$FILTER_PATH")
fi

mkdir -p "$OUTPUT_DIR" logs
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[$(date '+%F %T')] Fixed-checkpoint recursive revision analysis"
echo "  checkpoint=$CHECKPOINT"
echo "  config=$CONFIG_NAME"
echo "  dataset=$EVAL_REPO"
echo "  training_asset=$ASSETS_DIR/$TRAIN_ASSET_ID"
echo "  modes=${modes[*]}"
echo "  offsets=0 4 8 12"
echo "  output=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
"$PYTHON_BIN" scripts/eval_recursive_revision_analysis.py \
  --config-name "$CONFIG_NAME" \
  --pretrained-params "$CHECKPOINT" \
  --model-label "$MODEL_LABEL" \
  --modes "${modes[@]}" \
  --repo-id "$EVAL_REPO" \
  --asset-id "$TRAIN_ASSET_ID" \
  --assets-dir "$ASSETS_DIR" \
  "${filter_args[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --max-batches "$MAX_BATCHES" \
  --seed "$SEED" \
  --num-steps "$NUM_STEPS" \
  --offsets 0 4 8 12 \
  --latent-action-condition zero \
  --contact-threshold "$CONTACT_THRESHOLD" \
  --contact-min-taxels "$CONTACT_MIN_TAXELS" \
  --contact-min-consecutive-frames "$CONTACT_MIN_CONSECUTIVE_FRAMES" \
  2>&1 | tee "$OUTPUT_DIR/eval.log"

echo "[$(date '+%F %T')] Done: $OUTPUT_DIR"
echo "  metrics: $OUTPUT_DIR/metrics.json"
echo "  table source: $OUTPUT_DIR/metrics_long.csv"
