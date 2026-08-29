#!/usr/bin/env bash
set -Eeuo pipefail

# Sequentially run:
#   Stage 1: future_tactile_encoder_pretrain_flare_dit
#   Stage 2: pi0_xhand_tactile_action_aware_flare_single_ae
#
# Usage from the FactileLDM repo root:
#
#   setsid nohup env \
#     RUN_TAG=106ep_0621 \
#     bash scripts/run_action_aware_stage1_stage2.sh \
#       data/grasp_pipette_and_press_button_106ep \
#     > logs/action_aware_stage1_stage2_106ep_0621.scheduler.log 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/grasp_pipette_and_press_button_106ep}"
DATA_ASSET_ID="${DATA_ASSET_ID:-$(basename "$DATA_REPO")}"
RUN_TAG="${RUN_TAG:-$(basename "$DATA_REPO")_$(date +%m%d_%H%M)}"

WEIGHT_PATH="${WEIGHT_PATH:-checkpoints/pi0_base/params}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

STAGE1_CONFIG="${STAGE1_CONFIG:-future_tactile_encoder_pretrain_flare_dit}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
STAGE1_STEPS="${STAGE1_STEPS:-30000}"
STAGE1_SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL:-5000}"
STAGE1_KEEP_PERIOD="${STAGE1_KEEP_PERIOD:-5000}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-future_tactile_encoder_${RUN_TAG}_${STAGE1_STEPS}steps}"

STAGE2_CONFIG="${STAGE2_CONFIG:-pi0_xhand_tactile_action_aware_flare_single_ae}"
STAGE2_STEPS="${STAGE2_STEPS:-20000}"
STAGE2_SAVE_INTERVAL="${STAGE2_SAVE_INTERVAL:-5000}"
STAGE2_KEEP_PERIOD="${STAGE2_KEEP_PERIOD:-5000}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-action_aware_single_ae_${RUN_TAG}_${STAGE2_STEPS}steps}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p logs "$HF_DATASETS_CACHE" "$HF_HOME"

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if [[ -x "$PYTHON" ]] && "$PYTHON" -V >/dev/null 2>&1; then
      printf '%s\n' "$PYTHON"
      return
    fi
    cat >&2 <<EOF
ERROR: PYTHON is set but is not runnable: $PYTHON

If the virtualenv was copied from another machine, its python symlink may point to
a missing interpreter. Check with:
  ls -lah env/.venv/bin/python* .venv/bin/python* 2>/dev/null || true
  readlink -f env/.venv/bin/python .venv/bin/python 2>/dev/null || true
EOF
    exit 4
  fi

  local candidate
  for candidate in env/.venv/bin/python .venv/bin/python python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -V >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  cat >&2 <<'EOF'
ERROR: Could not find a runnable Python interpreter.

Tried:
  env/.venv/bin/python
  .venv/bin/python
  python3.11
  python3

If env/.venv was copied from another server, recreate it on this machine with:
  uv sync --frozen

Or pass a known-good interpreter explicitly:
  PYTHON=/path/to/python bash scripts/run_action_aware_stage1_stage2.sh data/...
EOF
  exit 4
}

PYTHON="$(resolve_python)"

if [[ ! -f "$DATA_REPO/meta/info.json" ]]; then
  cat >&2 <<EOF
ERROR: Local LeRobot dataset not found: $DATA_REPO
Expected file: $DATA_REPO/meta/info.json

Pass an existing dataset path relative to FactileLDM, for example:
  bash scripts/run_action_aware_stage1_stage2.sh data/grasp_pipette_and_press_button_106ep
EOF
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

latest_step_dir() {
  local root="$1"
  find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1
}

ensure_norm_stats() {
  local source_config="$1"
  local target_config="$2"
  local source_dir="assets/${source_config}/${DATA_ASSET_ID}"
  local target_dir="assets/${target_config}/${DATA_ASSET_ID}"

  if [[ -d "$target_dir" ]]; then
    log "Norm stats already exist: ${target_dir}"
    return
  fi
  if [[ -d "$source_dir" ]]; then
    log "Copying norm stats: ${source_dir} -> ${target_dir}"
    mkdir -p "$(dirname "$target_dir")"
    cp -a "$source_dir" "$target_dir"
    return
  fi

  log "Computing norm stats for ${target_config}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" \
    "$PYTHON" scripts/compute_norm_stats.py \
      --config-name "$target_config" \
      --repo-id "$DATA_REPO" \
      --asset-id "$DATA_ASSET_ID" \
    > "logs/norm_${target_config}_${RUN_TAG}.log" 2>&1
}

log "Dataset repo: ${DATA_REPO}"
log "Asset id: ${DATA_ASSET_ID}"
log "Run tag: ${RUN_TAG}"
log "Stage 1 config: ${STAGE1_CONFIG}"
log "Python: ${PYTHON} ($("$PYTHON" -V 2>&1))"
log "GPUs: ${GPU_IDS}, batch: ${GLOBAL_BATCH_SIZE}, fsdp devices: ${FSDP_DEVICES}"

# These three configs use the same structured calc-force data transform.
ensure_norm_stats "pi0_xhand_tactile_structured_single_ae" "$STAGE1_CONFIG"
ensure_norm_stats "$STAGE1_CONFIG" "$STAGE2_CONFIG"

STAGE1_LOG="logs/${STAGE1_EXP_NAME}.log"
STAGE1_CKPT_ROOT="checkpoints/${STAGE1_CONFIG}/${STAGE1_EXP_NAME}"
if [[ "$SKIP_STAGE1" == "1" ]]; then
  log "Stage 1 skipped; reusing checkpoint under ${STAGE1_CKPT_ROOT}"
else
  log "Stage 1 starting: exp=${STAGE1_EXP_NAME}, steps=${STAGE1_STEPS}, log=${STAGE1_LOG}"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    "$PYTHON" scripts/train.py "$STAGE1_CONFIG" \
      --exp-name "$STAGE1_EXP_NAME" \
      --data.repo-id "$DATA_REPO" \
      --data.assets.asset-id "$DATA_ASSET_ID" \
      --num-train-steps "$STAGE1_STEPS" \
      --batch-size "$GLOBAL_BATCH_SIZE" \
      --fsdp-devices "$FSDP_DEVICES" \
      --num-workers "$NUM_WORKERS" \
      --save-interval "$STAGE1_SAVE_INTERVAL" \
      --keep-period "$STAGE1_KEEP_PERIOD" \
      --no-wandb-enabled \
      --overwrite \
    > "$STAGE1_LOG" 2>&1
fi

STAGE1_STEP="$(latest_step_dir "$STAGE1_CKPT_ROOT")"
if [[ -z "${STAGE1_STEP:-}" ]]; then
  log "ERROR: Stage 1 completed but no numeric checkpoint was found under ${STAGE1_CKPT_ROOT}"
  exit 3
fi
STAGE1_ENCODER_PARAMS="${STAGE1_CKPT_ROOT}/${STAGE1_STEP}/params"
if [[ ! -d "$STAGE1_ENCODER_PARAMS" ]]; then
  log "ERROR: Stage 1 encoder params not found: ${STAGE1_ENCODER_PARAMS}"
  exit 3
fi
log "Stage 1 completed. Using encoder params: ${STAGE1_ENCODER_PARAMS}"

STAGE2_LOG="logs/${STAGE2_EXP_NAME}.log"
log "Stage 2 starting: exp=${STAGE2_EXP_NAME}, steps=${STAGE2_STEPS}, log=${STAGE2_LOG}"
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  "$PYTHON" scripts/train.py "$STAGE2_CONFIG" \
    --exp-name "$STAGE2_EXP_NAME" \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$DATA_ASSET_ID" \
    --num-train-steps "$STAGE2_STEPS" \
    --batch-size "$GLOBAL_BATCH_SIZE" \
    --fsdp-devices "$FSDP_DEVICES" \
    --num-workers "$NUM_WORKERS" \
    --save-interval "$STAGE2_SAVE_INTERVAL" \
    --keep-period "$STAGE2_KEEP_PERIOD" \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.pi0-params-path "$WEIGHT_PATH" \
    --weight-loader.encoder-params-path "$STAGE1_ENCODER_PARAMS" \
  > "$STAGE2_LOG" 2>&1

log "Stage 2 completed successfully."
log "Done. Stage1=${STAGE1_CKPT_ROOT}, Stage2=checkpoints/${STAGE2_CONFIG}/${STAGE2_EXP_NAME}"
