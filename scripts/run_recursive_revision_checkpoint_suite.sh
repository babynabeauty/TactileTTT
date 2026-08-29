#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

EVAL_REPO="${1:-data/task3-eval}"
RUN_TAG="${RUN_TAG:-task3_eval_recursive_revision_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/recursive_revision/${RUN_TAG}}"
RAW_ASSETS_DIR="${RAW_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
PYTHON_BIN="${PYTHON_BIN:-env/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_BATCHES="${MAX_BATCHES:-0}"
NUM_STEPS="${NUM_STEPS:-10}"
CONTACT_THRESHOLD="${CONTACT_THRESHOLD:-1.0}"
CONTACT_MIN_TAXELS="${CONTACT_MIN_TAXELS:-1}"
CONTACT_MIN_CONSECUTIVE_FRAMES="${CONTACT_MIN_CONSECUTIVE_FRAMES:-1}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

mkdir -p "$OUTPUT_ROOT"

run_eval() {
  local label="$1"
  local config_name="$2"
  local behavior="$3"
  local checkpoint="$4"
  local training_asset_id="$5"
  local output_dir="$OUTPUT_ROOT/$label"

  if [[ ! -e "$checkpoint" ]]; then
    echo "ERROR: checkpoint not found: $checkpoint" >&2
    return 1
  fi
  if [[ ! -f "$RAW_ASSETS_DIR/$training_asset_id/norm_stats.json" ]]; then
    echo "ERROR: norm stats not found: $RAW_ASSETS_DIR/$training_asset_id/norm_stats.json" >&2
    return 1
  fi

  mkdir -p "$output_dir"
  echo "[$(date '+%F %T')] Evaluating $label"
  echo "  config=$config_name behavior=$behavior asset=$training_asset_id checkpoint=$checkpoint"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" scripts/eval_recursive_revision_analysis.py \
    --config-name "$config_name" \
    --pretrained-params "$checkpoint" \
    --model-label "$label" \
    --modes "$behavior" \
    --repo-id "$EVAL_REPO" \
    --asset-id "$training_asset_id" \
    --assets-dir "$RAW_ASSETS_DIR" \
    --output-dir "$output_dir" \
    --batch-size "$BATCH_SIZE" \
    --fsdp-devices "$FSDP_DEVICES" \
    --num-workers "$NUM_WORKERS" \
    --max-batches "$MAX_BATCHES" \
    --num-steps "$NUM_STEPS" \
    --offsets 0 4 8 12 \
    --contact-threshold "$CONTACT_THRESHOLD" \
    --contact-min-taxels "$CONTACT_MIN_TAXELS" \
    --contact-min-consecutive-frames "$CONTACT_MIN_CONSECUTIVE_FRAMES" \
    2>&1 | tee "$output_dir/eval.log"
}

# "one_shot_future" still refreshes the action from fresh tactile at each
# offset, but keeps the t0 future-contact prediction fixed. This matches the
# no_future_update training ablation and is evaluated as action_only.
run_eval \
  "one_shot_future_f8_task12345_60k" \
  "pi0_xhand_dual_patch_pretrained_f8_h16_async_no_future_update" \
  "action_only" \
  "checkpoints/pi0_xhand_dual_patch_pretrained_f8_h16_async_no_future_update/B_no_future_update_task12345_2_ablation_f8_0712/59999" \
  "task12345-2"

run_eval \
  "fresh_f4_taskall2_80k" \
  "pi0_xhand_dual_patch_pretrained_f4_h16_async_fresh" \
  "fresh_reinfer" \
  "checkpoints/pi0_xhand_dual_patch_pretrained_f4_h16_async_fresh/pi0_xhand_dual_patch_pretrained_f4_h16_async_fresh_taskall2_fresh_f4_0722_0000/80000" \
  "taskall-2"

run_eval \
  "retouch_f4_task12345_60k" \
  "pi0_xhand_dual_patch_pretrained_f4_h16_async" \
  "retouch" \
  "checkpoints/pi0_xhand_dual_patch_pretrained_f4_h16_async/pi0_xhand_dual_patch_pretrained_f4_h16_async_task12345_2_patch_pretrain_fast_0709/59999" \
  "task12345-2"

run_eval \
  "retouch_f8_taskall2_80k" \
  "pi0_xhand_dual_patch_pretrained_f8_h16_async" \
  "retouch" \
  "checkpoints/pi0_xhand_dual_patch_pretrained_f8_h16_async/pi0_xhand_dual_patch_pretrained_f8_h16_async_taskall2_patch_pretrained_f8_async_90k_0717/80000" \
  "taskall-2"

run_eval \
  "retouch_f8_taskall2_90k" \
  "pi0_xhand_dual_patch_pretrained_f8_h16_async" \
  "retouch" \
  "checkpoints/pi0_xhand_dual_patch_pretrained_f8_h16_async/pi0_xhand_dual_patch_pretrained_f8_h16_async_taskall2_patch_pretrained_f8_async_90k_0717/89999" \
  "taskall-2"

run_eval \
  "retouch_f8_task12345_60k" \
  "pi0_xhand_dual_patch_pretrained_f8_h16_async" \
  "retouch" \
  "checkpoints/pi0_xhand_dual_patch_pretrained_f8_h16_async/pi0_xhand_dual_patch_pretrained_f8_h16_async_task12345_2_patch_pretrain_f8_0711_1607/59999" \
  "task12345-2"

"$PYTHON_BIN" scripts/summarize_recursive_revision_suite.py \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT"

echo "[$(date '+%F %T')] Recursive revision suite complete: $OUTPUT_ROOT"
