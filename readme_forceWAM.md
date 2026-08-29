{"task_index":0,"task":"pick up the orange cob"}
{"task_index":1,"task":"pick up the pipette and press the pipette button"}
{"task_index":2,"task":"pick up the transparent water bottle"}
{"task_index":3,"task":"press the red button"}
{"task_index":4,"task":"wipe the black marks off the white surface with the sponge"}


# ForceWAM / FactileLDM 实验命令

## 每次运行前先设置

```bash
cd /workspace/mnt/sqzhang26/FactileLDM
source env/.venv/bin/activate

export PROJECT_ROOT=/workspace/mnt/sqzhang26/FactileLDM
export HF_LEROBOT_HOME="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=.hf_datasets_cache

DATA_REPO="data/task12345-2"
ASSET_ID="$(basename "$DATA_REPO")"
mkdir -p logs
```

说明：
- 统一使用 `HF_LEROBOT_HOME=/workspace/mnt/sqzhang26/FactileLDM`
- 统一使用 `DATA_REPO=data/xxx`
- 不要再使用 `HF_LEROBOT_HOME=/workspace/mnt/sqzhang26/FactileLDM/data`，否则容易变成 `data/data/xxx`

## 归一化

### 自动并行生成三套归一化文件

脚本会轮询空闲 GPU，并行生成 structured calc-force、非结构化 calc-force 和 structured raw 三套统计。
默认要求空闲显存至少 50000 MiB、GPU 利用率不高于 20%。已有文件会自动跳过。

```bash
setsid nohup env \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  MIN_FREE_MEMORY_MB=50000 \
  MAX_GPU_UTIL=20 \
  POLL_INTERVAL=30 \
  bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO" \
  > "logs/norm_scheduler_${ASSET_ID}.log" 2>&1 &
```

强制重新计算已有统计时增加：

```bash
OVERWRITE_NORM=1 bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"
```

### 5x3 structured dual AE

```bash
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_structured_dual_ae \
  --repo-id "$DATA_REPO" \
  --asset-id "$ASSET_ID"
```

### 5x120x3 raw structured dual AE

```bash
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_structured_raw_dual_ae \
  --repo-id "$DATA_REPO" \
  --asset-id "$ASSET_ID"
```

### structured single AE

```bash
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_structured_single_ae \
  --repo-id "$DATA_REPO" \
  --asset-id "$ASSET_ID"
```

## 推荐训练命令

### A. pi0 no tactile

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_full_finetune_h16 \
    --exp-name pi0_xhand_full_finetune_h16_task12345_0706 \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_raw_dual_ae \
    --num-train-steps 50000 \
    --batch-size 8 \
    --fsdp-devices 1 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_full_finetune_h16_task12345_0706.log 2>&1 &
```

### B. current 5x3 tactile observation tokens

obs-AE 已统一使用 `effort [5,3]`，直接复用 structured dual-AE 的归一化文件。

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_obs_ae_full_finetune \
    --exp-name pi0_xhand_tactile_obs_ae_full_finetune_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_dual_ae \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_tactile_obs_ae_full_finetune_106ep_20k.log 2>&1 &
```

### C. structured single AE

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_single_ae \
    --exp-name structured_single_ae_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_dual_ae \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_single_ae_106ep_20k.log 2>&1 &
```

### D. structured dual AE 5x3

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_dual_ae \
    --exp-name structured_dual_ae_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_dual_ae_106ep_20k.log 2>&1 &
```

### E. structured dual AE 5x3 + arm/future/hand mask

这里复用 `pi0_xhand_tactile_structured_dual_ae` 的归一化文件，所以保留 `--data.assets.assets-dir`。

DATA_REPO="data/grasp_pipette_and_press_button_106ep"
ASSET_ID="$(basename "$DATA_REPO")"

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_dual_ae_arm_future_hand_mask \
    --exp-name structured_dual_ae_arm_future_hand_mask_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_dual_ae \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_dual_ae_arm_future_hand_mask_106ep_20k.log 2>&1 &
```

### F. structured raw dual AE 5x120x3

raw 点阵触觉需要单独归一化，不要复用 5x3 的 assets。

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_raw_dual_ae \
    --exp-name pi0_xhand_tactile_structured_raw_dual_ae_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_tactile_structured_raw_dual_ae_106ep_20k.log 2>&1 &
```

### G. Patch-informed raw dual AE

保持每根手指 1 个外部 token，但在每根手指内部先按 5 个 patch 聚合点阵力，再融合成 finger token。复用 raw tactile 的 assets。

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_dual_patch_f4_h16 \
    --exp-name pi0_xhand_dual_patch_f4_h16 \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_raw_dual_ae \
    --num-train-steps 30000 \
    --batch-size 8 \
    --fsdp-devices 1 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_dual_patch_f4_h16.log 2>&1 &
```

### H. Patch-informed cached VLM async AE

异步训练版本：action horizon=16，history tactile=10 tokens，future tactile=4 segments，对应部署里 cached VLM + fresh tactile 更新 AE 的主线。

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_dual_patch_f4_h16_async \
    --exp-name pi0_xhand_dual_patch_f4_h16_async \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_raw_dual_ae \
    --num-train-steps 50000 \
    --batch-size 8 \
    --fsdp-devices 1 \
    --num-workers 2 \
    --save-interval 10000 \
    --keep-period 10000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_dual_patch_f4_h16_async.log 2>&1 &
```

# config总结：
当前主线只看 dual-AE。single-AE / flow / mask / refiner 旧配置先放到历史归档，不作为当前实验主线。

## Canonical dual-AE configs

共同设置：

- action horizon = 16
- history tactile = 10 tokens，即 5 个历史摘要 finger tokens + 5 个当前帧 finger tokens
- tokenizer 外部输出都是每个 tactile step 5 个 finger tokens
- patch-informed 复用 `assets/pi0_xhand_tactile_structured_raw_dual_ae`

| config | tokenizer | future segments | future tokens | async training | fast offsets |
|---|---|---:|---:|---|---|
| `pi0_xhand_dual_patch_f4_h16` | patch-informed | 4 | 20 | 否 | - |
| `pi0_xhand_dual_patch_f4_h16_async` | patch-informed | 4 | 20 | 是 | 4,8,12 |
| `pi0_xhand_dual_patch_f8_h16` | patch-informed | 8 | 40 | 否 | - |
| `pi0_xhand_dual_patch_f8_h16_async` | patch-informed | 8 | 40 | 是 | 2,4,6,8,10 |

baseline:

- `pi0_xhand_full_finetune_h16`：原始 pi0，action horizon = 16，不输入 tactile。

旧配置仍保留在代码中，主要用于复现实验和读取旧 checkpoint，不建议新实验继续优先使用。


## 历史内容归档

下面是之前的命令和杂项记录，保留备用；日常跑实验优先复制上面的推荐命令。

```bash
cd /workspace/mnt/sqzhang26/FactileLDM
source env/.venv/bin/activate
export PROJECT_ROOT=/workspace/mnt/sqzhang26/FactileLDM
export HF_LEROBOT_HOME="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=.hf_datasets_cache
DATA_REPO="data/grasp_pipette_and_press_button_106ep"
ASSET_ID="$(basename "$DATA_REPO")"
```

# 合并数据集
python scripts/merge_lerobot_v21_datasets.py \
  --sources \
    /workspace/mnt/sqzhang26/FactileLDM/data/47ep \
    /workspace/mnt/sqzhang26/FactileLDM/data/grasp_pipette_and_press_button_0616_59ep \
  --output /workspace/mnt/sqzhang26/FactileLDM/data/grasp_pipette_and_press_button_106ep \
  --overwrite
python scripts/merge_lerobot_v21_datasets.py \
  --sources \
    /workspace/mnt/sqzhang26/FactileLDM/data/task12345 \
    /workspace/mnt/sqzhang26/FactileLDM/data/0706_grasp_bottle \
  --output /workspace/mnt/sqzhang26/FactileLDM/data/task12345-2 \
  --overwrite

# 计算光流图像

```bash
    CUDA_VISIBLE_DEVICES=0 \
env/.venv/bin/python scripts/compute_lerobot_future_flow_video.py \
  --repo-id data/grasp_pipette_and_press_button_0616_59ep_flow \
  --output-dir ./flow_videos \
  --future-step 32 \
  --overwrite
后台挂起
  setsid nohup env CUDA_VISIBLE_DEVICES=4 \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/compute_lerobot_future_flow_video.py \
  --repo-id /data/workspace/zhangshiqi/forceWAM/grasp_pipette_and_press_button_26ep \
  --output-dir /data/workspace/zhangshiqi/forceWAM/flow_videos \
  --future-step 32 \
  > flow_videos/compute_future_flow.log 2>&1 &
```

# 将光流视频添加到 LeRobot 数据集
```bash
  env/.venv/bin/python scripts/add_flow_videos_to_lerobot.py \
  --repo-id grasp_pipette_and_press_0614_7ep \
  --flow-videos-dir flow_videos/videos \
  --map \
    cam_front=observation.future_flow.cam_front \
    cam_right=observation.future_flow.cam_right \
  --overwrite
```


# 计算3D偏移点
```bash
DATA=grasp_pipette_and_press_button
OUT=outputs/front_scene_flow_grasp_pipette_sam3_tracked_npz

mkdir -p "$OUT"
for EP in $(seq 0 48); do
  echo "===== episode ${EP} ====="
  CUDA_VISIBLE_DEVICES=4 env/.venv/bin/python scripts/visualize_front_scene_flow_episode_sam3_tracked_object.py \
    --repo-id "$DATA" \
    --episode-index "$EP" \
    --future-step 32 \
    --stride 3 \
    --max-depth 2.5 \
    --pair-step 1 \
    --overlay-step 1000000 \
    --sam3-checkpoint /data/shared_workspace/zhangshiqi/hf/SAM/sam3/sam3.pt \
    --sam3-device cuda \
    --sam3-confidence-threshold 0.35 \
    --object-points '238,215;242,250;246,275' \
    --object-negative-points '249,326;259,361;248,301' \
    --save-flow-npz \
    --skip-ply \
    --output-dir "$OUT"
done 2>&1 | tee "$OUT/batch.log"
```



# 模型推理
```bash
CUDA_VISIBLE_DEVICES=0 env/.venv/bin/python scripts/serve_policy.py --port=8990 policy:checkpoint \
--policy.config=pi0_xhand_tactile_flow_full_finetune \
--policy.dir=checkpoints/pi0_xhand_tactile_flow_full_finetune/pi0_xhand_tactile_flow_full_finetune/49999

CUDA_VISIBLE_DEVICES=0 env/.venv/bin/python scripts/serve_policy.py --port=8990 policy:checkpoint \
--policy.config=pi0_xhand_tactile_3dflow_full_finetune \
--policy.dir=FactileLDM/checkpoints/pi0_xhand_tactile_3dflow_full_finetune/pi0_xhand_tactile_3dflow_full_finetune/49999

CUDA_VISIBLE_DEVICES=0 env/.venv/bin/python scripts/serve_policy.py --port=8990 policy:checkpoint \
--policy.config=pi0_xhand_full_finetune \
--policy.dir=checkpoints/pi0_xhand_full_finetune/pi0_xhand_full_finetune_30k_2gpu/49999

```

# 压缩
tar -czvf data.tar.gz grasp_pipette_and_press_button_26ep_26ep/
# 解压
tar -xzvf 文件名.tar.gz

pip install awscli

aws configure
AWS Access Key ID [None]:
AWS Secret Access Key [None]:
Default region name [None]: huhehaote-1
Default output format [None]: json

aws s3 ls s3://sqzhang26-2   --endpoint-url https://eos-huhehaote-1.cmecloud.cn

#下载文件
aws s3 cp s3://sqzhang26-2/data.tar.gz data.tar.gz \
  --endpoint-url https://eos-huhehaote-1.cmecloud.cn

#下载整个目录
aws s3 cp s3://sqzhang26-2/path/to/folder ./folder \
  --recursive \
  --endpoint-url https://eos-huhehaote-1.cmecloud.cn

#上载整个目录
aws s3 cp /Users/babyna/FactileLDM/data/press_button_4_times/press_button_1/meta s3://sqzhang26-2/meta_1  \
  --recursive \
  --endpoint-url https://eos-huhehaote-1.cmecloud.cn

AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
AWS_RESPONSE_CHECKSUM_VALIDATION=when_required \
aws s3 cp \
"/Users/babyna/FactileLDM/data/press_button_4_times/press_button_0/meta" \
"s3://sqzhang26-2/meta_0/" \
--recursive \
--endpoint-url "https://eos-huhehaote-1.cmecloud.cn"

# 大文件上传  mac支持\
s3cmd put /Users/babyna/FactileLDM/data/press_button_4_times/press_button_2.tar.gz  \
  s3://sqzhang26-2/press_button_2.tar.gz



# 划分验证集
cd /workspace/mnt/sqzhang26/FactileLDM

env/.venv/bin/python scripts/create_task_stratified_episode_split.py \
  --repo-id data/taskall-2 \
  --output-dir outputs/episode_splits/taskall-2_recursive_revision \
  --val-ratio 0.10 \
  --min-val-per-task 3 \
  --seed 42

# 验证encoder
cd /workspace/mnt/sqzhang26/FactileLDM

PATCH_ENCODER_PARAMS=checkpoints/xhand_patch_tactile_encoder_pretrain/xhand_patch_tactile_encoder_pretrain_taskall2_patch_pretrained_f8_async_90k_0717/19999/params

env/.venv/bin/python scripts/eval_patch_tactile_encoder.py \
  --repo-id data/taskall-2 \
  --params "$PATCH_ENCODER_PARAMS" \
  --filter-path outputs/episode_splits/taskall-2_recursive_revision/val_episodes.json \
  --output-dir outputs/patch_encoder_eval/taskall-2 \
  --batch-size 256 \
  --max-frames 20000


# 验证未来触觉预测
cd /workspace/mnt/sqzhang26/FactileLDM

CHECKPOINT="checkpoints/你的新ckpt目录/step"
MODEL_LABEL="new_retouch_checkpoint"
OUTPUT_DIR="outputs/task_test_recursive_revision_new_ckpt"

mkdir -p "$OUTPUT_DIR" logs

CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
HF_LEROBOT_HOME="$PWD" \
HF_HUB_OFFLINE=1 \
env/.venv/bin/python scripts/eval_recursive_revision_analysis.py \
  --config-name pi0_xhand_dual_patch_pretrained_f8_h16_async \
  --pretrained-params "$CHECKPOINT" \
  --model-label "$MODEL_LABEL" \
  --modes one_shot fresh_reinfer retouch \
  --repo-id data/task-test \
  --asset-id taskall-2 \
  --assets-dir assets/pi0_xhand_tactile_structured_raw_dual_ae \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 4 \
  --fsdp-devices 4 \
  --num-workers 2 \
  --max-batches 0 \
  --seed 42 \
  --num-steps 10 \
  --offsets 0 4 8 12 \
  --latent-action-condition zero \
  --contact-threshold 1.0 \
  --contact-min-taxels 1 \
  --contact-min-consecutive-frames 1 \
  2>&1 | tee "$OUTPUT_DIR/eval.log"
# 可视化 PatchTactileEncoder的结果
cd /workspace/mnt/sqzhang26/FactileLDM

PATCH_ENCODER_PARAMS=checkpoints/xhand_patch_tactile_encoder_pretrain/patch_informed_full_heads_taskall2_encoder_final_20k_0722/19999/params

CUDA_VISIBLE_DEVICES=2 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
env/.venv/bin/python scripts/visualize_patch_reconstruction_episodes.py \
  --repo-id data/taskall-2 \
  --params "$PATCH_ENCODER_PARAMS" \
  --config-name xhand_patch_tactile_encoder_pretrain \
  --filter-path outputs/episode_splits/taskall-2_encoder_final_10pct_seed42/val_episodes.json \
  --assets-dir assets/pi0_xhand_tactile_structured_raw_dual_ae \
  --asset-id taskall-2 \
  --output-dir outputs/patch_reconstruction_visualization/taskall2_encoder_final_20k_0722 \
  --selection max_contact \
  --batch-size 256 \
  --frame-stride 1 \
  --dpi 150 \
  --make-video