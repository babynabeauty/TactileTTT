#!/usr/bin/env bash
set -Eeuo pipefail

# Server 2 for the task6+7 experiment:
#   1. Ensure raw tactile norm stats exist.
#   2. Run the main pretrained patch f8/h16 async pipeline:
#        Stage1 encoder pretrain -> Stage2A freeze -> Stage2B async finetune.
#   3. Run ablation-specific Stage2A freeze warmups.
#   4. Run async ablations from their own freeze checkpoint.
#
# raw_mlp is intentionally not queued here; use the standalone command printed
# by Codex so it can be launched later with the same settings.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/task6_7}"
ASSET_ID="${ASSET_ID:-$(basename "$DATA_REPO")}"
PYTHON="${PYTHON:-env/.venv/bin/python}"
PI0_BASE_PARAMS="${PI0_BASE_PARAMS:-checkpoints/pi0_base/params}"
RAW_TACTILE_ASSETS_DIR="${RAW_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"

export RUN_TAG="${RUN_TAG:-${ASSET_ID}_task67_server2_$(date +%m%d_%H%M%S)}"
export NORM_BATCH_SIZE="${NORM_BATCH_SIZE:-64}"
export NORM_NUM_WORKERS="${NORM_NUM_WORKERS:-2}"

export STAGE1_STEPS="${STAGE1_STEPS:-20000}"
export STAGE2A_STEPS="${STAGE2A_STEPS:-5000}"
export STAGE2B_STEPS="${STAGE2B_STEPS:-50000}"
export STAGE1_SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL:-10000}"
export STAGE2A_SAVE_INTERVAL="${STAGE2A_SAVE_INTERVAL:-5000}"
export STAGE2B_SAVE_INTERVAL="${STAGE2B_SAVE_INTERVAL:-10000}"
export KEEP_PERIOD="${KEEP_PERIOD:-10000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-64}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export STAGE2_BASE_CONFIG="${STAGE2_BASE_CONFIG:-pi0_xhand_dual_patch_f8_h16_async}"
export ASSET_DIR="$RAW_TACTILE_ASSETS_DIR"
export BASE_PARAMS="$PI0_BASE_PARAMS"

if [[ ! -f "$DATA_REPO/meta/info.json" ]]; then
  echo "ERROR: dataset not found: $DATA_REPO/meta/info.json" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -e "$PI0_BASE_PARAMS" ]]; then
  echo "ERROR: pi0 base params not found: $PI0_BASE_PARAMS" >&2
  exit 2
fi

mkdir -p logs

run_train_once() {
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
    echo "[server2] Skipping ${config}: final checkpoint already exists at ${final_params}"
    return 0
  fi
  if [[ -e "$checkpoint_dir" && "${RESUME:-0}" != "1" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "ERROR: checkpoint dir already exists: $checkpoint_dir" >&2
    echo "Use a new RUN_TAG, set RESUME=1, or set ALLOW_OVERWRITE=1." >&2
    exit 2
  fi

  local run_mode_args=()
  if [[ "${ALLOW_OVERWRITE:-0}" == "1" ]]; then
    run_mode_args=(--overwrite)
  elif [[ "${RESUME:-0}" == "1" && -e "$checkpoint_dir" ]]; then
    run_mode_args=(--resume)
  fi

  echo "[server2] Starting ${config}: exp=${exp}, steps=${steps}, batch=${batch_size}, fsdp=${fsdp_devices}, log=${log_file}"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}" \
  HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" scripts/train.py "$config" \
    --exp-name "$exp" \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir "$RAW_TACTILE_ASSETS_DIR" \
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
  echo "[server2] Finished ${config}: exp=${exp}"
}

echo "[server2] Stage 1/4: ensure raw tactile normalization"
NORM_CONFIGS="pi0_xhand_tactile_structured_raw_dual_ae" \
ASSET_ID="$ASSET_ID" \
PYTHON="$PYTHON" \
bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"

echo "[server2] Stage 2/4: main patch-pretrained f8/h16 async pipeline"
bash scripts/run_patch_encoder_pretrain_pipeline.sh "$DATA_REPO"

CONFIG_STAGE1="xhand_patch_tactile_encoder_pretrain"
CONFIG_STAGE2A="pi0_xhand_dual_patch_pretrained_f8_h16_async_freeze"
EXP_STAGE1="xhand_patch_tactile_encoder_pretrain_${RUN_TAG}"
EXP_STAGE2A="${CONFIG_STAGE2A}_${RUN_TAG}"
CKPT_STAGE1="checkpoints/${CONFIG_STAGE1}/${EXP_STAGE1}/$((STAGE1_STEPS - 1))/params"
CKPT_STAGE2A="checkpoints/${CONFIG_STAGE2A}/${EXP_STAGE2A}/$((STAGE2A_STEPS - 1))/params"

if [[ ! -e "$CKPT_STAGE1" ]]; then
  echo "ERROR: Stage1 encoder checkpoint not found: $CKPT_STAGE1" >&2
  exit 2
fi
if [[ ! -e "$CKPT_STAGE2A" ]]; then
  echo "ERROR: Stage2A policy checkpoint not found: $CKPT_STAGE2A" >&2
  exit 2
fi

echo "[server2] Stage1 encoder params: $CKPT_STAGE1"
echo "[server2] Stage2A policy params: $CKPT_STAGE2A"

CONFIG_NO_FUTURE="pi0_xhand_patch_pretrained_f8_h16_async_no_future"
CONFIG_NO_FUTURE_UPDATE="pi0_xhand_dual_patch_pretrained_f8_h16_async_no_future_update"
CONFIG_NO_FUTURE_FREEZE="${CONFIG_NO_FUTURE}_freeze"
CONFIG_NO_FUTURE_UPDATE_FREEZE="${CONFIG_NO_FUTURE_UPDATE}_freeze"
EXP_NO_FUTURE_FREEZE="${CONFIG_NO_FUTURE_FREEZE}_${RUN_TAG}"
EXP_NO_FUTURE_UPDATE_FREEZE="${CONFIG_NO_FUTURE_UPDATE_FREEZE}_${RUN_TAG}"
CKPT_NO_FUTURE_FREEZE="checkpoints/${CONFIG_NO_FUTURE_FREEZE}/${EXP_NO_FUTURE_FREEZE}/$((STAGE2A_STEPS - 1))/params"
CKPT_NO_FUTURE_UPDATE_FREEZE="checkpoints/${CONFIG_NO_FUTURE_UPDATE_FREEZE}/${EXP_NO_FUTURE_UPDATE_FREEZE}/$((STAGE2A_STEPS - 1))/params"

echo "[server2] Stage 3/4: ablation-specific freeze warmups"
run_train_once \
  "$CONFIG_NO_FUTURE_FREEZE" \
  "$EXP_NO_FUTURE_FREEZE" \
  "$STAGE2A_STEPS" \
  "$STAGE2A_SAVE_INTERVAL" \
  4 \
  "$STAGE2_BATCH_SIZE" \
  "logs/${EXP_NO_FUTURE_FREEZE}.log" \
  --weight-loader.pi0-params-path "$PI0_BASE_PARAMS" \
  --weight-loader.encoder-params-path "$CKPT_STAGE1"

run_train_once \
  "$CONFIG_NO_FUTURE_UPDATE_FREEZE" \
  "$EXP_NO_FUTURE_UPDATE_FREEZE" \
  "$STAGE2A_STEPS" \
  "$STAGE2A_SAVE_INTERVAL" \
  4 \
  "$STAGE2_BATCH_SIZE" \
  "logs/${EXP_NO_FUTURE_UPDATE_FREEZE}.log" \
  --weight-loader.pi0-params-path "$PI0_BASE_PARAMS" \
  --weight-loader.encoder-params-path "$CKPT_STAGE1"

if [[ ! -e "$CKPT_NO_FUTURE_FREEZE" ]]; then
  echo "ERROR: no_future freeze checkpoint not found: $CKPT_NO_FUTURE_FREEZE" >&2
  exit 2
fi
if [[ ! -e "$CKPT_NO_FUTURE_UPDATE_FREEZE" ]]; then
  echo "ERROR: no_future_update freeze checkpoint not found: $CKPT_NO_FUTURE_UPDATE_FREEZE" >&2
  exit 2
fi

export TRAIN_STEPS="${ABLATION_TRAIN_STEPS:-50000}"
export GLOBAL_BATCH_SIZE="${ABLATION_GLOBAL_BATCH_SIZE:-8}"
export FSDP_DEVICES="${ABLATION_FSDP_DEVICES:-4}"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-2}"
export SAVE_INTERVAL="${ABLATION_SAVE_INTERVAL:-10000}"
export KEEP_PERIOD="${ABLATION_KEEP_PERIOD:-10000}"
export GPU_SLOTS="${ABLATION_GPU_SLOTS:-0,1,2,3,4,5,6,7}"

JOB_LABELS=(
  "B_no_future"
  "C_no_future_update"
)
JOB_CONFIGS=(
  "pi0_xhand_patch_pretrained_f8_h16_async_no_future"
  "pi0_xhand_dual_patch_pretrained_f8_h16_async_no_future_update"
)
JOB_ASSET_DIRS=(
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
)
JOB_WEIGHT_ARGS=(
  "--weight-loader.pi0-params-path $CKPT_NO_FUTURE_FREEZE --weight-loader.encoder-params-path $CKPT_STAGE1"
  "--weight-loader.pi0-params-path $CKPT_NO_FUTURE_UPDATE_FREEZE --weight-loader.encoder-params-path $CKPT_STAGE1"
)

echo "[server2] Stage 4/4: queued async ablations"
source scripts/four_gpu_training_queue.sh
DATA_ASSET_ID="$ASSET_ID" RUN_TAG="${RUN_TAG}_ablation" run_four_gpu_training_queue "$DATA_REPO"
