#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate full-head, mean-force/contact/distribution, and compact three-head
# checkpoints on the exact same uniformly sampled held-out frames.
#
# FULL_HEAD_PARAMS must point to the selected full-head checkpoint because that
# checkpoint root may contain multiple experiments.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON:-env/.venv/bin/python}"
DATA_REPO="${DATA_REPO:-data/taskall-2}"
ASSET_ID="${ASSET_ID:-taskall-2}"
ASSETS_DIR="${ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
FILTER_PATH="${FILTER_PATH:-outputs/episode_splits/taskall-2_encoder_final_10pct_seed42/val_episodes.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/patch_encoder_eval/three_checkpoint_comparison}"
GPU="${GPU:-2}"
MAX_FRAMES="${MAX_FRAMES:-20000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"

FULL_HEAD_PARAMS="${FULL_HEAD_PARAMS:-}"
MFCD_PARAMS="${MFCD_PARAMS:-}"
THREE_HEAD_PARAMS="${THREE_HEAD_PARAMS:-}"
if [[ -z "$MFCD_PARAMS" ]]; then
  MFCD_ROOT="checkpoints/xhand_patch_mean_force_contact_distribution_encoder_pretrain"
  MFCD_EXP="xhand_patch_mean_force_contact_distribution_encoder_pretrain_taskall2_mfcd_seed42"
  MFCD_PARAMS="${MFCD_ROOT}/${MFCD_EXP}/19999/params"
fi
if [[ -z "$THREE_HEAD_PARAMS" ]]; then
  THREE_HEAD_ROOT="checkpoints/xhand_patch_force_three_head_encoder_pretrain"
  THREE_HEAD_PARAMS="${THREE_HEAD_ROOT}/force_three_head_taskall2_force_three_head_seed42/19999/params"
fi

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: ${message}: ${path}" >&2
    exit 2
  fi
}

if [[ -z "$FULL_HEAD_PARAMS" ]]; then
  echo "ERROR: set FULL_HEAD_PARAMS to the full-head .../params checkpoint." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
require_path "$PYTHON_BIN" "python not found"
require_path "$DATA_REPO/meta/info.json" "dataset not found"
require_path "$FILTER_PATH" "held-out split not found"
require_path "$ASSETS_DIR/$ASSET_ID/norm_stats.json" "normalization statistics not found"
require_path "$FULL_HEAD_PARAMS" "full-head checkpoint not found"
require_path "$MFCD_PARAMS" "MFCD checkpoint not found"
require_path "$THREE_HEAD_PARAMS" "three-head checkpoint not found"
mkdir -p "$OUTPUT_ROOT"

run_eval() {
  local label="$1"
  local config="$2"
  local params="$3"
  local output_dir="$OUTPUT_ROOT/$label"
  echo "Evaluating ${label}: ${params}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON_BIN" scripts/eval_patch_tactile_encoder.py \
    --repo-id "$DATA_REPO" \
    --params "$params" \
    --config-name "$config" \
    --filter-path "$FILTER_PATH" \
    --assets-dir "$ASSETS_DIR" \
    --asset-id "$ASSET_ID" \
    --output-dir "$output_dir" \
    --max-frames "$MAX_FRAMES" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"
}

run_eval \
  full_head \
  xhand_patch_tactile_encoder_pretrain \
  "$FULL_HEAD_PARAMS"
run_eval \
  mean_force_contact_distribution \
  xhand_patch_mean_force_contact_distribution_encoder_pretrain \
  "$MFCD_PARAMS"
run_eval \
  force_three_head \
  xhand_patch_force_three_head_encoder_pretrain \
  "$THREE_HEAD_PARAMS"

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
models = (
    "full_head",
    "mean_force_contact_distribution",
    "force_three_head",
)
metric_names = (
    "contact_f1",
    "contact_precision",
    "contact_recall",
    "contact_bce",
    "distribution_kl",
    "force_distribution_kl",
    "active_force_vector_l2",
    "active_pred_force_magnitude_mean",
    "active_target_force_magnitude_mean",
    "active_force_magnitude_ratio",
    "active_force_direction_cosine",
    "active_force_x_mae",
    "active_force_y_mae",
    "active_force_z_mae",
    "inactive_force_magnitude_mean",
    "strength_mae",
    "strength_pearson",
    "eval_frames",
)
rows = []
for model in models:
    metrics = json.loads((root / model / "metrics.json").read_text())
    row = {"model": model}
    row.update({name: metrics.get(name) for name in metric_names})
    rows.append(row)

output = root / "comparison.csv"
with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=("model", *metric_names))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {output}")
PY

echo "Comparison complete: ${OUTPUT_ROOT}/comparison.csv"
