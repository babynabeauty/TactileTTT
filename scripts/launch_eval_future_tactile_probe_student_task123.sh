#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

GPU_IDS="${GPU_IDS:-0,1,2,3}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCHES="${MAX_BATCHES:-200}"
NUM_PLOT_SAMPLES="${NUM_PLOT_SAMPLES:-12}"
NUM_WORKERS="${NUM_WORKERS:-0}"

PROBE_DIR="${PROBE_DIR:-checkpoints/future_tactile_token_probe/pi0_xhand_tactile_structured_dual_ae_history_future_pool/probe_student_task123_valsplit/student}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/future_tactile_probe_eval/student_task123_valsplit}"
LOG_FILE="${LOG_FILE:-logs/eval_future_tactile_probe_student_task123_valsplit.log}"

if [[ ! -d "$PROBE_DIR" ]]; then
  echo "ERROR: probe dir not found: $PROBE_DIR" >&2
  exit 1
fi

setsid nohup env \
  HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  PYTHONUNBUFFERED=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python -u scripts/eval_future_tactile_token_probe.py \
    --probe-dir "$PROBE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-batches "$MAX_BATCHES" \
    --num-plot-samples "$NUM_PLOT_SAMPLES" \
    --num-workers "$NUM_WORKERS" \
    --fsdp-devices "$FSDP_DEVICES" \
    --batch-size "$BATCH_SIZE" \
  > "$LOG_FILE" 2>&1 &

echo "Started student probe eval: PID=$!"
echo "Log: $LOG_FILE"
echo "Output: $OUTPUT_DIR"
