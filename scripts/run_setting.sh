#!/usr/bin/env bash

# Shared two-slot scheduler for 8-GPU training servers.
# The caller must define JOB_LABELS, JOB_CONFIGS, and JOB_ASSET_DIRS.

run_four_gpu_training_queue() {
  if (( ${#JOB_LABELS[@]} == 0 )); then
    echo "ERROR: No training jobs were configured." >&2
    return 2
  fi
  if (( ${#JOB_LABELS[@]} != ${#JOB_CONFIGS[@]} || ${#JOB_LABELS[@]} != ${#JOB_ASSET_DIRS[@]} )); then
    echo "ERROR: JOB_LABELS, JOB_CONFIGS, and JOB_ASSET_DIRS must have the same length." >&2
    return 2
  fi

  local data_repo="${1:-data/task1_2_206ep}"
  local data_asset_id="${DATA_ASSET_ID:-$(basename "$data_repo")}" 
  local train_steps="${TRAIN_STEPS:-20000}"
  local global_batch_size="${GLOBAL_BATCH_SIZE:-8}"
  # 1 disables parameter sharding. All four visible GPUs are still used for
  # data parallelism, with a full model/optimizer replica on every GPU.
  local fsdp_devices="${FSDP_DEVICES:-1}"
  local num_workers="${NUM_WORKERS:-2}"
  local save_interval="${SAVE_INTERVAL:-5000}"
  local keep_period="${KEEP_PERIOD:-5000}"
  local run_tag="${RUN_TAG:-task1_2_206ep_20k_$(date +%m%d_%H%M%S)}"
  local overwrite_existing="${ALLOW_OVERWRITE:-0}"
  local python_bin="${PYTHON:-env/.venv/bin/python}"
  local default_weight_path="${WEIGHT_PATH:-checkpoints/pi0_base/params}"
  local has_job_weight_paths=0
  local has_job_weight_args=0
  local has_job_ready_paths=0
  local has_job_train_steps=0
  local has_job_save_intervals=0
  if declare -p JOB_WEIGHT_PATHS >/dev/null 2>&1; then
    has_job_weight_paths=1
    if (( ${#JOB_WEIGHT_PATHS[@]} != ${#JOB_LABELS[@]} )); then
      echo "ERROR: JOB_WEIGHT_PATHS must have the same length as JOB_LABELS when provided." >&2
      return 2
    fi
  fi
  if declare -p JOB_TRAIN_STEPS >/dev/null 2>&1; then
    has_job_train_steps=1
    if (( ${#JOB_TRAIN_STEPS[@]} != ${#JOB_LABELS[@]} )); then
      echo "ERROR: JOB_TRAIN_STEPS must have the same length as JOB_LABELS when provided." >&2
      return 2
    fi
  fi
  if declare -p JOB_SAVE_INTERVALS >/dev/null 2>&1; then
    has_job_save_intervals=1
    if (( ${#JOB_SAVE_INTERVALS[@]} != ${#JOB_LABELS[@]} )); then
      echo "ERROR: JOB_SAVE_INTERVALS must have the same length as JOB_LABELS when provided." >&2
      return 2
    fi
  fi
  if declare -p JOB_READY_PATHS >/dev/null 2>&1; then
    has_job_ready_paths=1
    if (( ${#JOB_READY_PATHS[@]} != ${#JOB_LABELS[@]} )); then
      echo "ERROR: JOB_READY_PATHS must have the same length as JOB_LABELS when provided." >&2
      return 2
    fi
  fi
  if declare -p JOB_WEIGHT_ARGS >/dev/null 2>&1; then
    has_job_weight_args=1
    if (( ${#JOB_WEIGHT_ARGS[@]} != ${#JOB_LABELS[@]} )); then
      echo "ERROR: JOB_WEIGHT_ARGS must have the same length as JOB_LABELS when provided." >&2
      return 2
    fi
  fi
  local gpu_wait_enabled="${GPU_WAIT_ENABLED:-1}"
  local gpu_min_free_mib="${GPU_MIN_FREE_MIB:-70000}"
  local gpu_poll_seconds="${GPU_POLL_SECONDS:-60}"
  local gpu_status_log_seconds="${GPU_STATUS_LOG_SECONDS:-300}"
  local gpu_slots_spec="${GPU_SLOTS:-0,1,2,3;4,5,6,7}"
  local -a gpu_slots=()
  IFS=';' read -r -a gpu_slots <<< "$gpu_slots_spec"
  if (( ${#gpu_slots[@]} == 0 )); then
    echo "ERROR: GPU_SLOTS must define at least one slot, got: $gpu_slots_spec" >&2
    return 2
  fi

  if [[ ! "$gpu_wait_enabled" =~ ^[01]$ ]]; then
    echo "ERROR: GPU_WAIT_ENABLED must be 0 or 1, got: $gpu_wait_enabled" >&2
    return 2
  fi
  if [[ ! "$overwrite_existing" =~ ^[01]$ ]]; then
    echo "ERROR: ALLOW_OVERWRITE must be 0 or 1, got: $overwrite_existing" >&2
    return 2
  fi
  if [[ ! "$gpu_min_free_mib" =~ ^[0-9]+$ || ! "$gpu_poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GPU_MIN_FREE_MIB and GPU_POLL_SECONDS must be positive integers." >&2
    return 2
  fi
  if [[ ! "$gpu_status_log_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GPU_STATUS_LOG_SECONDS must be a positive integer." >&2
    return 2
  fi

  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}"
  export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false

  mkdir -p logs "$HF_DATASETS_CACHE" "$HF_HOME"

  if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: Python is not executable: $python_bin" >&2
    return 2
  fi
  if [[ ! -f "$data_repo/meta/info.json" ]]; then
    echo "ERROR: Dataset not found: $data_repo/meta/info.json" >&2
    return 2
  fi
  if (( has_job_weight_paths == 0 && has_job_weight_args == 0 )) && [[ ! -e "$default_weight_path" ]]; then
    echo "ERROR: Base checkpoint not found: $default_weight_path" >&2
    return 2
  fi

  local asset_dir
  for asset_dir in "${JOB_ASSET_DIRS[@]}"; do
    if [[ ! -f "$asset_dir/$data_asset_id/norm_stats.json" ]]; then
      echo "ERROR: Norm stats not found: $asset_dir/$data_asset_id/norm_stats.json" >&2
      return 2
    fi
  done

  local job_index
  local checkpoint_dir
  for job_index in "${!JOB_LABELS[@]}"; do
    checkpoint_dir="checkpoints/${JOB_CONFIGS[$job_index]}/${JOB_LABELS[$job_index]}_${run_tag}"
    if [[ -e "$checkpoint_dir" && "$overwrite_existing" == "0" ]]; then
      echo "ERROR: Checkpoint directory already exists: $checkpoint_dir" >&2
      echo "Use a new RUN_TAG to keep old checkpoints, or set ALLOW_OVERWRITE=1 to delete and rerun it." >&2
      return 2
    fi
    if (( has_job_weight_args == 0 )); then
      local job_weight_path="$default_weight_path"
      if (( has_job_weight_paths == 1 )); then
        job_weight_path="${JOB_WEIGHT_PATHS[$job_index]}"
      fi
      if [[ ! -e "$job_weight_path" ]]; then
        echo "ERROR: Base checkpoint not found for ${JOB_LABELS[$job_index]}: $job_weight_path" >&2
        return 2
      fi
    fi
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi -L | wc -l)"
    local max_gpu_id=-1
    local slot_spec
    local gpu_id
    local -a slot_gpu_ids=()
    for slot_spec in "${gpu_slots[@]}"; do
      IFS=',' read -r -a slot_gpu_ids <<< "$slot_spec"
      for gpu_id in "${slot_gpu_ids[@]}"; do
        gpu_id="${gpu_id//[[:space:]]/}"
        if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
          echo "ERROR: Invalid GPU id in GPU_SLOTS: $gpu_slots_spec" >&2
          return 2
        fi
        (( gpu_id > max_gpu_id )) && max_gpu_id="$gpu_id"
      done
    done
    if (( gpu_count <= max_gpu_id )); then
      echo "ERROR: GPU_SLOTS references GPU${max_gpu_id}, but only ${gpu_count} visible GPUs were found." >&2
      return 2
    fi
  elif (( gpu_wait_enabled == 1 )); then
    echo "ERROR: nvidia-smi is required when GPU_WAIT_ENABLED=1." >&2
    return 2
  fi

  log_queue() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
  }

  declare -A pid_to_slot=()
  declare -A pid_to_label=()
  declare -A gpu_free_mib=()
  local -a active_pids=()
  local -a slot_pid=()
  local slot_init_index
  for slot_init_index in "${!gpu_slots[@]}"; do
    slot_pid[$slot_init_index]=""
  done
  local next_job=0
  local failed=0
  local last_gpu_status_log=0

  remove_active_pid() {
    local target_pid="$1"
    local -a remaining=()
    local pid
    for pid in "${active_pids[@]}"; do
      if [[ "$pid" != "$target_pid" ]]; then
        remaining+=("$pid")
      fi
    done
    active_pids=("${remaining[@]}")
  }

  refresh_gpu_free_memory() {
    local output
    local gpu_id
    local free_mib

    gpu_free_mib=()
    if ! output="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
      return 1
    fi
    while IFS=',' read -r gpu_id free_mib; do
      gpu_id="${gpu_id//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      if [[ "$gpu_id" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]]; then
        gpu_free_mib["$gpu_id"]="$free_mib"
      fi
    done <<< "$output"
    (( ${#gpu_free_mib[@]} >= 8 ))
  }

  slot_has_enough_memory() {
    local slot_index="$1"
    local gpu_id
    local -a slot_gpu_ids=()

    if (( gpu_wait_enabled == 0 )); then
      return 0
    fi
    IFS=',' read -r -a slot_gpu_ids <<< "${gpu_slots[$slot_index]}"
    for gpu_id in "${slot_gpu_ids[@]}"; do
      if [[ -z "${gpu_free_mib[$gpu_id]+present}" ]]; then
        return 1
      fi
      if (( gpu_free_mib[$gpu_id] < gpu_min_free_mib )); then
        return 1
      fi
    done
    return 0
  }

  gpu_slot_status() {
    local slot_index="$1"
    local gpu_id
    local -a slot_gpu_ids=()
    local -a values=()

    IFS=',' read -r -a slot_gpu_ids <<< "${gpu_slots[$slot_index]}"
    for gpu_id in "${slot_gpu_ids[@]}"; do
      values+=("GPU${gpu_id}=${gpu_free_mib[$gpu_id]:-?}MiB")
    done
    local IFS=', '
    printf '%s' "${values[*]}"
  }

  terminate_active_jobs() {
    local pid
    if (( ${#active_pids[@]} == 0 )); then
      return
    fi
    log_queue "Stopping active training jobs..."
    for pid in "${active_pids[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
  }
  trap 'terminate_active_jobs; exit 130' INT TERM

  launch_job() {
    local job_index="$1"
    local slot_index="$2"
    local label="${JOB_LABELS[$job_index]}"
    local config="${JOB_CONFIGS[$job_index]}"
    local assets_dir="${JOB_ASSET_DIRS[$job_index]}"
    local job_train_steps="$train_steps"
    local job_save_interval="$save_interval"
    if (( has_job_train_steps == 1 )) && [[ -n "${JOB_TRAIN_STEPS[$job_index]}" ]]; then
      job_train_steps="${JOB_TRAIN_STEPS[$job_index]}"
    fi
    if (( has_job_save_intervals == 1 )) && [[ -n "${JOB_SAVE_INTERVALS[$job_index]}" ]]; then
      job_save_interval="${JOB_SAVE_INTERVALS[$job_index]}"
    fi
    local job_weight_path="$default_weight_path"
    if (( has_job_weight_paths == 1 )); then
      job_weight_path="${JOB_WEIGHT_PATHS[$job_index]}"
    fi
    local -a weight_args=()
    if (( has_job_weight_args == 1 )) && [[ -n "${JOB_WEIGHT_ARGS[$job_index]}" ]]; then
      # shellcheck disable=SC2206
      weight_args=(${JOB_WEIGHT_ARGS[$job_index]})
    else
      weight_args=(--weight-loader.params-path "$job_weight_path")
    fi
    local gpu_ids="${gpu_slots[$slot_index]}"
    local exp_name="${label}_${run_tag}"
    local log_file="logs/${exp_name}.log"
    local -a overwrite_args=()
    if (( overwrite_existing == 1 )); then
      overwrite_args=(--overwrite)
    fi

    log_queue "Starting ${label}: config=${config}, GPUs=${gpu_ids}, assets=${assets_dir}, weights=${weight_args[*]}"
    setsid --wait env \
      HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
      HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
      HF_HOME="$HF_HOME" \
      HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
      CUDA_VISIBLE_DEVICES="$gpu_ids" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "$python_bin" scripts/train.py "$config" \
        --exp-name "$exp_name" \
        --data.repo-id "$data_repo" \
        --data.assets.asset-id "$data_asset_id" \
        --data.assets.assets-dir "$assets_dir" \
        --num-train-steps "$job_train_steps" \
        --batch-size "$global_batch_size" \
        --fsdp-devices "$fsdp_devices" \
        --num-workers "$num_workers" \
        --save-interval "$job_save_interval" \
        --keep-period "$keep_period" \
        --no-wandb-enabled \
        "${overwrite_args[@]}" \
        "${weight_args[@]}" \
      > "$log_file" 2>&1 &

    local pid=$!
    pid_to_slot["$pid"]="$slot_index"
    pid_to_label["$pid"]="$label"
    active_pids+=("$pid")
    log_queue "Started ${label}: PID=${pid}, log=${log_file}"
  }

  log_queue "Dataset: $data_repo"
  log_queue "Asset ID: $data_asset_id"
  log_queue "Jobs: ${#JOB_LABELS[@]}, steps=${train_steps}, batch=${global_batch_size}, FSDP=${fsdp_devices}"
  log_queue "Run tag: ${run_tag}"
  if (( overwrite_existing == 1 )); then
    log_queue "Checkpoint overwrite is ENABLED by ALLOW_OVERWRITE=1."
  else
    log_queue "Checkpoint overwrite is disabled; existing checkpoint dirs will stop the scheduler."
  fi
  log_queue "GPU slots: ${gpu_slots[*]}"
  if (( gpu_wait_enabled == 1 )); then
    log_queue "GPU gate: every GPU in a slot must have at least ${gpu_min_free_mib} MiB free."
    log_queue "GPU polling interval: ${gpu_poll_seconds}s"
  else
    log_queue "GPU gate disabled by GPU_WAIT_ENABLED=0."
  fi

  while (( next_job < ${#JOB_LABELS[@]} || ${#active_pids[@]} > 0 )); do
    local made_progress=0
    local pid
    local status
    local finished_slot
    local finished_label
    local -a active_snapshot=("${active_pids[@]}")

    # Poll jobs launched by this scheduler and release their four-GPU slot
    # only after the actual Python process has exited.
    for pid in "${active_snapshot[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      set +e
      wait "$pid"
      status=$?
      set -e
      finished_slot="${pid_to_slot[$pid]}"
      finished_label="${pid_to_label[$pid]}"
      slot_pid[$finished_slot]=""
      remove_active_pid "$pid"
      unset 'pid_to_slot[$pid]' 'pid_to_label[$pid]'
      made_progress=1

      if (( status == 0 )); then
        log_queue "Completed ${finished_label} successfully; GPU slot ${gpu_slots[$finished_slot]} released."
      else
        log_queue "ERROR: ${finished_label} failed with exit code ${status}. No queued jobs will be started."
        failed=1
      fi
    done

    if (( failed == 0 && next_job < ${#JOB_LABELS[@]} )); then
      local memory_query_ok=1
      local readiness_blocked=0
      local readiness_wait_label=""
      local readiness_wait_path=""
      if (( gpu_wait_enabled == 1 )) && ! refresh_gpu_free_memory; then
        memory_query_ok=0
      fi

      local slot_index
      for slot_index in "${!gpu_slots[@]}"; do
        if (( next_job >= ${#JOB_LABELS[@]} )); then
          break
        fi
        if [[ -n "${slot_pid[$slot_index]}" ]]; then
          continue
        fi
        if (( has_job_ready_paths == 1 )) && [[ -n "${JOB_READY_PATHS[$next_job]}" && ! -e "${JOB_READY_PATHS[$next_job]}" ]]; then
          readiness_blocked=1
          readiness_wait_label="${JOB_LABELS[$next_job]}"
          readiness_wait_path="${JOB_READY_PATHS[$next_job]}"
          break
        fi
        if (( memory_query_ok == 1 )) && slot_has_enough_memory "$slot_index"; then
          launch_job "$next_job" "$slot_index"
          slot_pid[$slot_index]="${active_pids[-1]}"
          ((next_job += 1))
          made_progress=1
        fi
      done

      if (( made_progress == 0 )); then
        local now
        now="$(date +%s)"
        if (( now - last_gpu_status_log >= gpu_status_log_seconds )); then
          if (( readiness_blocked == 1 )); then
            log_queue "Waiting for dependency before ${readiness_wait_label}: ${readiness_wait_path}"
          elif (( memory_query_ok == 0 )); then
            log_queue "Waiting: nvidia-smi memory query failed; retrying in ${gpu_poll_seconds}s."
          else
            local -a slot_statuses=()
            local status_slot_index
            for status_slot_index in "${!gpu_slots[@]}"; do
              slot_statuses+=("slot${status_slot_index}[$(gpu_slot_status "$status_slot_index")]")
            done
            log_queue "Waiting for a free GPU slot (threshold=${gpu_min_free_mib}MiB): ${slot_statuses[*]}"
          fi
          last_gpu_status_log="$now"
        fi
      fi
    fi

    if (( failed != 0 && ${#active_pids[@]} == 0 )); then
      break
    fi
    if (( made_progress == 0 )); then
      sleep "$gpu_poll_seconds"
    fi
  done

  trap - INT TERM
  if (( failed != 0 )); then
    terminate_active_jobs
    return 1
  fi

  log_queue "All queued experiments completed successfully."
}
