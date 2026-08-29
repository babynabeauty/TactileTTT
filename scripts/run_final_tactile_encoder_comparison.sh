#!/usr/bin/env bash
set -Eeuo pipefail

# Final, controlled Stage-1 encoder comparison:
#   1. create one stratified 90/10 episode split (10% held out per task),
#   2. train Raw MLP, Raw Spatial, and Patch-informed Full-head encoders,
#   3. independently evaluate all three checkpoints on exactly the same split.
#
# Usage:
#   setsid nohup env \
#     RUN_TAG=taskall2_encoder_final_20k_0722 \
#     GPU_POOL=0,1,2 \
#     bash scripts/run_final_tactile_encoder_comparison.sh data/taskall-2 \
#     > logs/taskall2_encoder_final_20k_0722.scheduler.log 2>&1 &

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_REPO="${1:-data/taskall-2}"
ASSET_ID="${DATA_ASSET_ID:-$(basename "$DATA_REPO")}"
ASSET_DIR="${ASSET_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
PYTHON_BIN="${PYTHON:-env/.venv/bin/python}"

RUN_TAG="${RUN_TAG:-${ASSET_ID}_encoder_final_20k_$(date +%m%d_%H%M%S)}"
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
SPLIT_DIR="${SPLIT_DIR:-outputs/episode_splits/${ASSET_ID}_encoder_final_10pct_seed${SPLIT_SEED}}"
RECREATE_SPLIT="${RECREATE_SPLIT:-0}"

GPU_POOL="${GPU_POOL:-0,1,2}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
TRAIN_EVAL_NUM_BATCHES="${TRAIN_EVAL_NUM_BATCHES:-100}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-64}"
FINAL_EVAL_BATCH_SIZE="${FINAL_EVAL_BATCH_SIZE:-256}"
FINAL_EVAL_MAX_FRAMES="${FINAL_EVAL_MAX_FRAMES:-20000}"
RUN_PATCH_VISUALIZATION="${RUN_PATCH_VISUALIZATION:-1}"
PATCH_VIS_SELECTION="${PATCH_VIS_SELECTION:-max_contact}"
PATCH_VIS_FRAME_STRIDE="${PATCH_VIS_FRAME_STRIDE:-1}"
PATCH_VIS_DPI="${PATCH_VIS_DPI:-150}"
PATCH_VIS_MAKE_VIDEO="${PATCH_VIS_MAKE_VIDEO:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RESUME="${RESUME:-0}"

CONFIGS=(
  "xhand_raw_mlp_tactile_encoder_pretrain"
  "xhand_raw_spatial_tactile_encoder_pretrain"
  "xhand_patch_tactile_encoder_pretrain"
)
LABELS=(
  "raw_mlp_full_heads"
  "raw_spatial_full_heads"
  "patch_informed_full_heads"
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

cd "$PROJECT_ROOT"
mkdir -p logs "$SPLIT_DIR" .hf_datasets_cache .cache/huggingface

[[ -x "$PYTHON_BIN" ]] || die "python not found or not executable: $PYTHON_BIN"
[[ -f "$DATA_REPO/meta/info.json" ]] || die "dataset not found: $DATA_REPO"
[[ -f "$ASSET_DIR/$ASSET_ID/norm_stats.json" ]] || \
  die "normalization stats not found: $ASSET_DIR/$ASSET_ID/norm_stats.json"
[[ "$ALLOW_OVERWRITE" == "0" || "$ALLOW_OVERWRITE" == "1" ]] || die "ALLOW_OVERWRITE must be 0 or 1"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || die "RESUME must be 0 or 1"
[[ "$RECREATE_SPLIT" == "0" || "$RECREATE_SPLIT" == "1" ]] || die "RECREATE_SPLIT must be 0 or 1"
[[ "$RUN_PATCH_VISUALIZATION" == "0" || "$RUN_PATCH_VISUALIZATION" == "1" ]] || \
  die "RUN_PATCH_VISUALIZATION must be 0 or 1"
[[ "$PATCH_VIS_MAKE_VIDEO" == "0" || "$PATCH_VIS_MAKE_VIDEO" == "1" ]] || \
  die "PATCH_VIS_MAKE_VIDEO must be 0 or 1"
[[ ! ("$ALLOW_OVERWRITE" == "1" && "$RESUME" == "1") ]] || \
  die "ALLOW_OVERWRITE=1 and RESUME=1 cannot be enabled together"

TRAIN_FILTER_PATH="$SPLIT_DIR/train_episodes.json"
EVAL_FILTER_PATH="$SPLIT_DIR/val_episodes.json"

if [[ "$RECREATE_SPLIT" == "1" || ! -f "$TRAIN_FILTER_PATH" || ! -f "$EVAL_FILTER_PATH" ]]; then
  log "Creating stratified episode split: val_fraction=$VAL_FRACTION seed=$SPLIT_SEED"
  "$PYTHON_BIN" - "$DATA_REPO" "$SPLIT_DIR" "$VAL_FRACTION" "$SPLIT_SEED" <<'PY'
import json
import math
import pathlib
import random
import sys
from collections import defaultdict

repo = pathlib.Path(sys.argv[1]).resolve()
split_dir = pathlib.Path(sys.argv[2]).resolve()
val_fraction = float(sys.argv[3])
seed = int(sys.argv[4])
if not 0.0 < val_fraction < 1.0:
    raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

episodes_path = repo / "meta" / "episodes.jsonl"
tasks_path = repo / "meta" / "tasks.jsonl"
if not episodes_path.exists():
    raise FileNotFoundError(episodes_path)

task_name_to_index = {}
if tasks_path.exists():
    with tasks_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "task" in row and "task_index" in row:
                task_name_to_index[str(row["task"])] = int(row["task_index"])

by_task = defaultdict(list)
with episodes_path.open() as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        episode = int(row.get("episode_index", row.get("index")))
        if "task_index" in row:
            task_key = int(row["task_index"])
        elif row.get("tasks"):
            task_name = str(row["tasks"][0])
            task_key = task_name_to_index.get(task_name, task_name)
        elif "task" in row:
            task_name = str(row["task"])
            task_key = task_name_to_index.get(task_name, task_name)
        else:
            task_key = 0
        by_task[task_key].append(episode)

rng = random.Random(seed)
train = []
val = []
summary = {}
for task_key, task_episodes in sorted(by_task.items(), key=lambda item: str(item[0])):
    task_episodes = sorted(set(task_episodes))
    if len(task_episodes) == 1:
        val_count = 1
    else:
        val_count = min(len(task_episodes) - 1, max(1, int(math.ceil(len(task_episodes) * val_fraction))))
    task_val = sorted(rng.sample(task_episodes, val_count))
    task_val_set = set(task_val)
    task_train = [episode for episode in task_episodes if episode not in task_val_set]
    train.extend(task_train)
    val.extend(task_val)
    summary[str(task_key)] = {
        "total": len(task_episodes),
        "train_count": len(task_train),
        "val_count": len(task_val),
        "train_episodes": task_train,
        "val_episodes": task_val,
    }

train = sorted(set(train))
val = sorted(set(val))
if set(train) & set(val):
    raise RuntimeError("train/val episode leakage detected")

split_dir.mkdir(parents=True, exist_ok=True)
(split_dir / "train_episodes.json").write_text(
    json.dumps({"episodes": train}, indent=2, ensure_ascii=False)
)
(split_dir / "val_episodes.json").write_text(
    json.dumps({"episodes": val}, indent=2, ensure_ascii=False)
)
(split_dir / "summary.json").write_text(
    json.dumps(
        {
            "repo": str(repo),
            "val_fraction_per_task": val_fraction,
            "seed": seed,
            "num_train_episodes": len(train),
            "num_val_episodes": len(val),
            "tasks": summary,
        },
        indent=2,
        ensure_ascii=False,
    )
)
print(f"Created split: train={len(train)}, val={len(val)}, tasks={len(by_task)}", flush=True)
PY
else
  log "Reusing existing split: $SPLIT_DIR"
fi

[[ -f "$TRAIN_FILTER_PATH" ]] || die "missing train split: $TRAIN_FILTER_PATH"
[[ -f "$EVAL_FILTER_PATH" ]] || die "missing validation split: $EVAL_FILTER_PATH"

IFS=',' read -r -a GPUS <<< "$GPU_POOL"
(( ${#GPUS[@]} >= ${#CONFIGS[@]} )) || \
  die "GPU_POOL needs at least ${#CONFIGS[@]} GPUs; got: $GPU_POOL"

run_train() {
  local gpu="$1"
  local config="$2"
  local label="$3"
  local exp="${label}_${RUN_TAG}"
  local checkpoint_dir="checkpoints/${config}/${exp}"
  local log_file="logs/${exp}.log"
  local mode_args=()

  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    mode_args=(--overwrite)
  elif [[ "$RESUME" == "1" && -d "$checkpoint_dir" ]]; then
    mode_args=(--resume)
  elif [[ -e "$checkpoint_dir" ]]; then
    die "checkpoint already exists: $checkpoint_dir; use a new RUN_TAG or RESUME=1"
  fi

  log "Training $label on GPU $gpu: config=$config exp=$exp"
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.hf_datasets_cache}" \
  HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON_BIN" scripts/train.py "$config" \
    --exp-name "$exp" \
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
    --eval-num-batches "$TRAIN_EVAL_NUM_BATCHES" \
    --eval-batch-size "$TRAIN_EVAL_BATCH_SIZE" \
    --eval-num-workers "$NUM_WORKERS" \
    --eval-repo-id "$DATA_REPO" \
    --eval-asset-id "$ASSET_ID" \
    --eval-assets-dir "$ASSET_DIR" \
    --eval-filter-path "$EVAL_FILTER_PATH" \
    --no-wandb-enabled \
    "${mode_args[@]}" \
    > "$log_file" 2>&1
  log "Training finished: $label"
}

run_final_eval() {
  local gpu="$1"
  local config="$2"
  local label="$3"
  local exp="${label}_${RUN_TAG}"
  local final_step=$((TRAIN_STEPS - 1))
  local params="checkpoints/${config}/${exp}/${final_step}/params"
  local output_dir="outputs/patch_encoder_eval_normalized/${label}_${RUN_TAG}"
  local log_file="logs/eval_${label}_${RUN_TAG}.log"

  [[ -d "$params" ]] || die "final checkpoint params not found: $params"
  log "Independent held-out eval for $label on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON_BIN" scripts/eval_patch_tactile_encoder.py \
    --config-name "$config" \
    --repo-id "$DATA_REPO" \
    --params "$params" \
    --assets-dir "$ASSET_DIR" \
    --asset-id "$ASSET_ID" \
    --filter-path "$EVAL_FILTER_PATH" \
    --output-dir "$output_dir" \
    --batch-size "$FINAL_EVAL_BATCH_SIZE" \
    --max-frames "$FINAL_EVAL_MAX_FRAMES" \
    > "$log_file" 2>&1
  log "Independent eval finished: $label -> $output_dir"
}

log "Dataset: $DATA_REPO"
log "Normalization: $ASSET_DIR/$ASSET_ID/norm_stats.json"
log "Train split: $TRAIN_FILTER_PATH"
log "Held-out split: $EVAL_FILTER_PATH"
log "Training: steps=$TRAIN_STEPS batch=$BATCH_SIZE GPUs=$GPU_POOL"

train_pids=()
for i in "${!CONFIGS[@]}"; do
  run_train "${GPUS[$i]}" "${CONFIGS[$i]}" "${LABELS[$i]}" &
  train_pids+=("$!")
done

train_failed=0
for pid in "${train_pids[@]}"; do
  if ! wait "$pid"; then
    train_failed=1
  fi
done
(( train_failed == 0 )) || die "at least one encoder training job failed; independent eval was not started"

eval_pids=()
for i in "${!CONFIGS[@]}"; do
  run_final_eval "${GPUS[$i]}" "${CONFIGS[$i]}" "${LABELS[$i]}" &
  eval_pids+=("$!")
done

eval_failed=0
for pid in "${eval_pids[@]}"; do
  if ! wait "$pid"; then
    eval_failed=1
  fi
done
(( eval_failed == 0 )) || die "at least one independent encoder evaluation failed"

if [[ "$RUN_PATCH_VISUALIZATION" == "1" ]]; then
  patch_config="xhand_patch_tactile_encoder_pretrain"
  patch_label="patch_informed_full_heads"
  patch_exp="${patch_label}_${RUN_TAG}"
  patch_step=$((TRAIN_STEPS - 1))
  patch_params="checkpoints/${patch_config}/${patch_exp}/${patch_step}/params"
  patch_output="outputs/patch_reconstruction_visualization/${RUN_TAG}"
  patch_log="logs/patch_reconstruction_visualization_${RUN_TAG}.log"
  video_args=()
  if [[ "$PATCH_VIS_MAKE_VIDEO" == "1" ]]; then
    video_args=(--make-video)
  fi

  log "Rendering held-out patch reconstruction episodes with $patch_params"
  CUDA_VISIBLE_DEVICES="${GPUS[2]}" \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON_BIN" scripts/visualize_patch_reconstruction_episodes.py \
    --repo-id "$DATA_REPO" \
    --params "$patch_params" \
    --config-name "$patch_config" \
    --filter-path "$EVAL_FILTER_PATH" \
    --assets-dir "$ASSET_DIR" \
    --asset-id "$ASSET_ID" \
    --output-dir "$patch_output" \
    --selection "$PATCH_VIS_SELECTION" \
    --seed "$SPLIT_SEED" \
    --batch-size "$FINAL_EVAL_BATCH_SIZE" \
    --frame-stride "$PATCH_VIS_FRAME_STRIDE" \
    --dpi "$PATCH_VIS_DPI" \
    "${video_args[@]}" \
    > "$patch_log" 2>&1
  log "Patch reconstruction visualization finished: $patch_output"
fi

log "Final three-encoder comparison completed successfully."
log "Metrics root: outputs/patch_encoder_eval_normalized"
