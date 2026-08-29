#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-}"
if [[ -z "$DATA_REPO" ]]; then
  echo "Usage: bash scripts/run_xhand_norm_stats_parallel.sh data/your_dataset" >&2
  exit 2
fi
if [[ ! -f "$DATA_REPO/meta/info.json" ]]; then
  echo "ERROR: LeRobot dataset not found: $DATA_REPO/meta/info.json" >&2
  exit 2
fi

ASSET_ID="${ASSET_ID:-$(basename "$DATA_REPO")}"
PYTHON="${PYTHON:-env/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-50000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
OVERWRITE_NORM="${OVERWRITE_NORM:-0}"
RUN_TAG="${RUN_TAG:-${ASSET_ID}_$(date +%m%d_%H%M%S)}"
NORM_BATCH_SIZE="${NORM_BATCH_SIZE:-64}"
NORM_NUM_WORKERS="${NORM_NUM_WORKERS:-2}"

if [[ -n "${NORM_CONFIGS:-}" ]]; then
  # Space-separated override, e.g.
  # NORM_CONFIGS="pi0_xhand_tactile_structured_dual_ae pi0_xhand_tactile_structured_raw_dual_ae"
  read -r -a CONFIGS <<< "$NORM_CONFIGS"
else
  CONFIGS=(
    pi0_xhand_tactile_structured_dual_ae
    pi0_xhand_tactile_forceonly_full_finetune
    pi0_xhand_tactile_structured_raw_dual_ae
  )
fi

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p logs "$HF_DATASETS_CACHE" "$HF_HOME"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required for GPU scheduling." >&2
  exit 2
fi

declare -A ALLOWED_GPU=()
declare -A RESERVED_GPU=()
declare -A PID_CONFIG=()
declare -A PID_GPU=()
IFS=',' read -ra GPU_ID_LIST <<< "$GPU_IDS"
for gpu in "${GPU_ID_LIST[@]}"; do
  ALLOWED_GPU["$gpu"]=1
done

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

norm_path() {
  local config="$1"
  printf 'assets/%s/%s/norm_stats.json\n' "$config" "$ASSET_ID"
}

find_available_gpu() {
  local best_gpu=""
  local best_free=-1
  local gpu free total util

  while IFS=',' read -r gpu free total util; do
    gpu="${gpu//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [[ -n "${ALLOWED_GPU[$gpu]:-}" ]] || continue
    [[ -z "${RESERVED_GPU[$gpu]:-}" ]] || continue
    (( free >= MIN_FREE_MEMORY_MB )) || continue
    (( util <= MAX_GPU_UTIL )) || continue
    if (( free > best_free )); then
      best_gpu="$gpu"
      best_free="$free"
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
  )

  printf '%s\n' "$best_gpu"
}

launch_job() {
  local config="$1"
  local gpu="$2"
  local log_file="logs/norm_${config}_${RUN_TAG}.log"

  log "Starting ${config} on GPU ${gpu}; log=${log_file}"
  {
    printf '[%s] config=%s gpu=%s repo=%s asset=%s\n' "$(date '+%F %T')" "$config" "$gpu" "$DATA_REPO" "$ASSET_ID"
    printf '[%s] python=%s\n' "$(date '+%F %T')" "$PYTHON"
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON" -u scripts/compute_norm_stats.py \
        --config-name "$config" \
        --repo-id "$DATA_REPO" \
        --asset-id "$ASSET_ID" \
        --batch-size "$NORM_BATCH_SIZE" \
        --num-workers "$NORM_NUM_WORKERS"
  } > "$log_file" 2>&1 &

  local pid=$!
  PID_CONFIG["$pid"]="$config"
  PID_GPU["$pid"]="$gpu"
  RESERVED_GPU["$gpu"]="$pid"
  log "Launched ${config}: PID=${pid}, GPU=${gpu}"
}

cleanup_children() {
  local pid
  for pid in "${!PID_CONFIG[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup_children INT TERM

PENDING=()
for config in "${CONFIGS[@]}"; do
  output="$(norm_path "$config")"
  if [[ -f "$output" && "$OVERWRITE_NORM" != "1" ]]; then
    log "Skipping existing stats: ${output}"
  else
    PENDING+=("$config")
  fi
done

if (( ${#PENDING[@]} == 0 )); then
  log "All requested normalization files already exist for asset ${ASSET_ID}."
  exit 0
fi

log "Dataset=${DATA_REPO}, asset=${ASSET_ID}"
log "GPU pool=${GPU_IDS}, min free=${MIN_FREE_MEMORY_MB} MiB, max util=${MAX_GPU_UTIL}%"
log "Norm loader: batch_size=${NORM_BATCH_SIZE}, num_workers=${NORM_NUM_WORKERS}"

FAILED=0
while (( ${#PENDING[@]} > 0 || ${#PID_CONFIG[@]} > 0 )); do
  progress=0

  for pid in "${!PID_CONFIG[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      config="${PID_CONFIG[$pid]}"
      gpu="${PID_GPU[$pid]}"
      if wait "$pid"; then
        log "Completed ${config} on GPU ${gpu}: $(norm_path "$config")"
      else
        status=$?
        log "ERROR: ${config} failed with exit code ${status}; check logs/norm_${config}_${RUN_TAG}.log"
        FAILED=1
      fi
      unset 'PID_CONFIG[$pid]' 'PID_GPU[$pid]' 'RESERVED_GPU[$gpu]'
      progress=1
    fi
  done

  while (( ${#PENDING[@]} > 0 )); do
    gpu="$(find_available_gpu)"
    [[ -n "$gpu" ]] || break
    config="${PENDING[0]}"
    PENDING=("${PENDING[@]:1}")
    launch_job "$config" "$gpu"
    progress=1
  done

  if (( ${#PENDING[@]} > 0 && ${#PID_CONFIG[@]} == 0 )); then
    log "No GPU currently satisfies the threshold; polling again in ${POLL_INTERVAL}s."
  fi
  if (( progress == 0 || ${#PID_CONFIG[@]} > 0 )); then
    sleep "$POLL_INTERVAL"
  fi
done

if (( FAILED != 0 )); then
  exit 1
fi
log "All requested normalization jobs completed successfully."
