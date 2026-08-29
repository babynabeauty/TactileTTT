#!/usr/bin/env bash
set -Eeuo pipefail

# Train the three Stage-1 patch tactile encoder objectives under the same
# data/eval split. Each run saves and evaluates every 10k steps by default.
#
# Usage:
#   setsid nohup env \
#     RUN_TAG=taskall2_stage1_objective_$(date +%m%d_%H%M) \
#     EVAL_FILTER_PATH=outputs/episode_splits/taskall-2_recursive_revision/val_episodes.json \
#     bash scripts/run_patch_encoder_stage1_objective_sweep.sh data/taskall-2 \
#     > logs/patch_encoder_stage1_objective_sweep.log 2>&1 &
#
# Useful knobs:
#   GPU_POOL=0,1,2,3            # one GPU per objective in parallel
#   PARALLEL=0                  # run objectives sequentially
#   TRAIN_STEPS=100000
#   SAVE_INTERVAL=10000
#   EVAL_INTERVAL=10000
#   EVAL_NUM_BATCHES=100
#   BATCH_SIZE=64
#   NUM_WORKERS=2
#   SWEEP_GROUP=nonpatch         # train only raw-spatial and raw-MLP controls
#   SWEEP_GROUP=all              # train all four objectives plus two controls
#   RESUME=1                    # resume existing RUN_TAG dirs
#   ALLOW_OVERWRITE=1           # overwrite existing RUN_TAG dirs

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_REPO="${1:-data/taskall-2}"
ASSET_ID="${DATA_ASSET_ID:-$(basename "$DATA_REPO")}"
ASSET_DIR="${ASSET_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"
PYTHON_BIN="${PYTHON:-env/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-${ASSET_ID}_stage1_objective_$(date +%m%d_%H%M%S)}"

GPU_POOL="${GPU_POOL:-0,1,2,3}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-$NUM_WORKERS}"
EVAL_REPO_ID="${EVAL_REPO_ID:-$DATA_REPO}"
EVAL_ASSET_ID="${EVAL_ASSET_ID:-$ASSET_ID}"
EVAL_ASSETS_DIR="${EVAL_ASSETS_DIR:-$ASSET_DIR}"
EVAL_FILTER_PATH="${EVAL_FILTER_PATH:-}"
TRAIN_FILTER_PATH="${TRAIN_FILTER_PATH:-}"
SPLIT_DIR="${SPLIT_DIR:-outputs/episode_splits/${ASSET_ID}_stage1_objective}"
EVAL_EPISODES_PER_TASK="${EVAL_EPISODES_PER_TASK:-10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RESUME="${RESUME:-0}"
SWEEP_GROUP="${SWEEP_GROUP:-objectives}"

case "$SWEEP_GROUP" in
  force_three_head)
    CONFIGS=("xhand_patch_force_three_head_encoder_pretrain")
    LABELS=("force_three_head")
    ;;
  objectives)
    CONFIGS=(
      "xhand_patch_tactile_encoder_pretrain"
      "xhand_patch_mean_force_encoder_pretrain"
      "xhand_patch_mean_force_contact_encoder_pretrain"
      "xhand_patch_strength_contact_encoder_pretrain"
    )
    LABELS=("full_heads" "mean_force" "mean_force_contact_zero" "strength_contact_zero")
    ;;
  nonpatch)
    CONFIGS=("xhand_raw_spatial_tactile_encoder_pretrain" "xhand_raw_mlp_tactile_encoder_pretrain")
    LABELS=("raw_spatial_full_heads" "raw_mlp_full_heads")
    ;;
  all)
    CONFIGS=(
      "xhand_patch_tactile_encoder_pretrain"
      "xhand_patch_mean_force_encoder_pretrain"
      "xhand_patch_mean_force_contact_encoder_pretrain"
      "xhand_patch_strength_contact_encoder_pretrain"
      "xhand_raw_spatial_tactile_encoder_pretrain"
      "xhand_raw_mlp_tactile_encoder_pretrain"
    )
    LABELS=(
      "full_heads" "mean_force" "mean_force_contact_zero" "strength_contact_zero"
      "raw_spatial_full_heads" "raw_mlp_full_heads"
    )
    ;;
  *)
    echo "ERROR: SWEEP_GROUP must be force_three_head, objectives, nonpatch, or all; got: $SWEEP_GROUP" >&2
    exit 2
    ;;
esac

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

check_output_dir() {
  local config="$1"
  local exp="$2"
  local dir="checkpoints/${config}/${exp}"
  if [[ "$RESUME" == "1" ]]; then
    return 0
  fi
  if [[ -e "$dir" && "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "ERROR: checkpoint dir already exists: $dir" >&2
    echo "Use a new RUN_TAG, set RESUME=1, or set ALLOW_OVERWRITE=1." >&2
    exit 2
  fi
}

run_one() {
  local gpu="$1"
  local config="$2"
  local label="$3"
  local exp="${label}_${RUN_TAG}"
  local log_file="logs/${exp}.log"

  local run_mode_args=()
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    run_mode_args=(--overwrite)
  elif [[ "$RESUME" == "1" && -e "checkpoints/${config}/${exp}" ]]; then
    run_mode_args=(--resume)
  fi

  local eval_args=(
    --eval-interval "$EVAL_INTERVAL"
    --eval-num-batches "$EVAL_NUM_BATCHES"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --eval-num-workers "$EVAL_NUM_WORKERS"
    --eval-repo-id "$EVAL_REPO_ID"
    --eval-asset-id "$EVAL_ASSET_ID"
    --eval-assets-dir "$EVAL_ASSETS_DIR"
  )
  if [[ -n "$EVAL_FILTER_PATH" ]]; then
    eval_args+=(--eval-filter-path "$EVAL_FILTER_PATH")
  fi

  local train_filter_args=()
  if [[ -n "$TRAIN_FILTER_PATH" ]]; then
    train_filter_args=(--train-filter-path "$TRAIN_FILTER_PATH")
  fi

  log "Starting ${label}: config=${config}, gpu=${gpu}, exp=${exp}, log=${log_file}"
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
    --num-train-steps "$TRAIN_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --fsdp-devices 1 \
    --num-workers "$NUM_WORKERS" \
    --save-interval "$SAVE_INTERVAL" \
    --keep-period "$KEEP_PERIOD" \
    "${train_filter_args[@]}" \
    --no-wandb-enabled \
    "${eval_args[@]}" \
    "${run_mode_args[@]}" \
    > "$log_file" 2>&1
  log "Finished ${label}: config=${config}, exp=${exp}"
}

cd "$PROJECT_ROOT"
mkdir -p logs .hf_datasets_cache .cache/huggingface

require_path "$PYTHON_BIN" "python not found"
require_path "$DATA_REPO/meta/info.json" "dataset not found"
require_path "$ASSET_DIR/$ASSET_ID/norm_stats.json" "norm stats not found"

if [[ "$ALLOW_OVERWRITE" != "0" && "$ALLOW_OVERWRITE" != "1" ]]; then
  echo "ERROR: ALLOW_OVERWRITE must be 0 or 1, got: $ALLOW_OVERWRITE" >&2
  exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "ERROR: RESUME must be 0 or 1, got: $RESUME" >&2
  exit 2
fi
if [[ "$ALLOW_OVERWRITE" == "1" && "$RESUME" == "1" ]]; then
  echo "ERROR: ALLOW_OVERWRITE=1 and RESUME=1 cannot be used together." >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_POOL"
if [[ "$PARALLEL" == "1" && ${#GPUS[@]} -lt ${#CONFIGS[@]} ]]; then
  echo "ERROR: GPU_POOL must contain at least ${#CONFIGS[@]} GPUs for the default parallel run." >&2
  echo "       Current GPU_POOL=${GPU_POOL}. Set PARALLEL=0 to run sequentially." >&2
  exit 2
fi

if [[ -z "$TRAIN_FILTER_PATH" || -z "$EVAL_FILTER_PATH" ]]; then
  mkdir -p "$SPLIT_DIR"
  log "Creating per-task split: eval=${EVAL_EPISODES_PER_TASK}/task, seed=${SPLIT_SEED}, output=${SPLIT_DIR}"
  "$PYTHON_BIN" - "$DATA_REPO" "$SPLIT_DIR" "$EVAL_EPISODES_PER_TASK" "$SPLIT_SEED" <<'PY'
import json
import pathlib
import random
import sys
from collections import defaultdict

repo = pathlib.Path(sys.argv[1])
split_dir = pathlib.Path(sys.argv[2])
eval_per_task = int(sys.argv[3])
seed = int(sys.argv[4])

episodes_path = repo / "meta" / "episodes.jsonl"
tasks_path = repo / "meta" / "tasks.jsonl"
if not episodes_path.exists():
    raise FileNotFoundError(f"missing {episodes_path}")

task_name_to_index = {}
if tasks_path.exists():
    with tasks_path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if "task" in row and "task_index" in row:
                    task_name_to_index[str(row["task"])] = int(row["task_index"])

by_task = defaultdict(list)
all_eps = []
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
        all_eps.append(episode)

rng = random.Random(seed)
val = []
summary = {}
for task_key, episodes in sorted(by_task.items(), key=lambda kv: str(kv[0])):
    episodes = sorted(set(episodes))
    chosen_count = min(eval_per_task, len(episodes))
    chosen = sorted(rng.sample(episodes, chosen_count))
    val.extend(chosen)
    summary[str(task_key)] = {
        "total": len(episodes),
        "val": chosen,
        "train": len(episodes) - len(chosen),
    }

val_set = set(val)
train = sorted(ep for ep in sorted(set(all_eps)) if ep not in val_set)
val = sorted(val_set)
split_dir.mkdir(parents=True, exist_ok=True)
(split_dir / "train_episodes.json").write_text(json.dumps({"episodes": train}, indent=2, ensure_ascii=False))
(split_dir / "val_episodes.json").write_text(json.dumps({"episodes": val}, indent=2, ensure_ascii=False))
(split_dir / "summary.json").write_text(
    json.dumps(
        {
            "repo": str(repo),
            "eval_episodes_per_task": eval_per_task,
            "seed": seed,
            "num_train_episodes": len(train),
            "num_val_episodes": len(val),
            "tasks": summary,
        },
        indent=2,
        ensure_ascii=False,
    )
)
print(f"wrote train={len(train)} val={len(val)} to {split_dir}", flush=True)
PY
  TRAIN_FILTER_PATH="${TRAIN_FILTER_PATH:-${SPLIT_DIR}/train_episodes.json}"
  EVAL_FILTER_PATH="${EVAL_FILTER_PATH:-${SPLIT_DIR}/val_episodes.json}"
fi

require_path "$TRAIN_FILTER_PATH" "train split json not found"
require_path "$EVAL_FILTER_PATH" "eval split json not found"

for i in "${!CONFIGS[@]}"; do
  check_output_dir "${CONFIGS[$i]}" "${LABELS[$i]}_${RUN_TAG}"
done

log "Dataset: $DATA_REPO"
log "Asset: $ASSET_DIR/$ASSET_ID"
log "Train split: $TRAIN_FILTER_PATH"
log "Eval repo: $EVAL_REPO_ID"
log "Eval split: $EVAL_FILTER_PATH"
log "Train steps: $TRAIN_STEPS, save interval: $SAVE_INTERVAL"
log "Eval interval: $EVAL_INTERVAL, eval batches: $EVAL_NUM_BATCHES"
log "Batch size: $BATCH_SIZE, eval batch size: $EVAL_BATCH_SIZE"
log "Parallel: $PARALLEL, GPU pool: $GPU_POOL"
log "Sweep group: $SWEEP_GROUP"
log "Run tag: $RUN_TAG"

if [[ "$PARALLEL" == "1" ]]; then
  pids=()
  for i in "${!CONFIGS[@]}"; do
    run_one "${GPUS[$i]}" "${CONFIGS[$i]}" "${LABELS[$i]}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "ERROR: at least one Stage-1 objective run failed. Check logs/${RUN_TAG} related logs." >&2
    exit 1
  fi
else
  for i in "${!CONFIGS[@]}"; do
    run_one "${GPUS[$i]:-${GPUS[0]}}" "${CONFIGS[$i]}" "${LABELS[$i]}"
  done
fi

log "All Stage-1 objective runs completed."
