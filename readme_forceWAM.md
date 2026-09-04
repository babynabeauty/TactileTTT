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

# 合并数据集
python scripts/merge_lerobot_v21_datasets.py \
  --sources \
    /workspace/mnt/sqzhang26/FactileLDM/data/0828_press_button/press_button_0 \
    /workspace/mnt/sqzhang26/FactileLDM/data/0828_press_button/press_button_1 \
    /workspace/mnt/sqzhang26/FactileLDM/data/0828_press_button/press_button_2 \
  --output /workspace/mnt/sqzhang26/FactileLDM/data/press_button_4_times \
  --overwrite
  
# 划分0.1的验证集
python scripts/create_task_stratified_episode_split.py   --repo-id data/press_button_4_times   --output-dir outputs/episode_splits/press_button_4_times   --val-ratio 0.10   --min-val-per-task 1   --seed 42

# 计算归一化
/workspace/mnt/sqzhang26/FactileLDM/env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_tactile_current \
  --repo-id data/press_button_4_times \
  --asset-id press_button_4_times \
  --batch-size 64 \
  --num-workers 4

# 训练

export PROJECT_ROOT=$PWD
export HF_LEROBOT_HOME=$PROJECT_ROOT
export HF_DATASETS_CACHE=$PROJECT_ROOT/.hf_datasets_cache
export HF_HUB_OFFLINE=1

DATA_REPO=data/press_button_4_times
ASSET_ID=press_button_4_times
ASSET_DIR=assets/pi05_tactile_current
TRAIN_SPLIT=outputs/episode_splits/press_button_4_times/train_episodes.json
VAL_SPLIT=outputs/episode_splits/press_button_4_times/val_episodes.json
PATCH_ENCODER_PARAMS=/workspace/mnt/sqzhang26/FactileLDM/checkpoints/xhand_patch_tactile_encoder_pretrain/patch_informed_full_heads_taskall2_encoder_final_20k_0722/19999/params

mkdir -p logs

## TactileTTT v1：阶段一250step，仅 warm-up TTT 参数,第二阶段联合训练750step


setsid nohup env \
GPU_IDS=0,1,2,3,4,5,6,7 \
FSDP_DEVICES=4 \
BATCH_SIZE=8 \
NUM_WORKERS=0 \
WARMUP_STEPS=250 \
JOINT_STEPS=750 \
DATA_REPO=data/press_button_4_times \
ASSET_ID=press_button_4_times \
ASSET_DIR=assets/pi05_tactile_current \
TRAIN_SPLIT=outputs/episode_splits/press_button_4_times/train_episodes.json \
VAL_SPLIT=outputs/episode_splits/press_button_4_times/val_episodes.json \
PATCH_ENCODER_PARAMS=/workspace/mnt/sqzhang26/FactileLDM/checkpoints/xhand_patch_tactile_encoder_pretrain/patch_informed_full_heads_taskall2_encoder_final_20k_0722/19999/params \
bash scripts/run_tactile_ttt_v1_two_stage.sh \
> "logs/tactile_ttt_v1_two_stage_0831.scheduler.log" 2>&1 &


# 单阶段整体训练 1000steps
setsid nohup env \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/workspace/mnt/sqzhang26/FactileLDM/env/.venv/bin/python scripts/train.py \
  pi05_tactile_ttt_v1 \
  --exp-name pi05_tactile_ttt_v1_0831 \
  --data.repo-id "$DATA_REPO" \
  --data.assets.asset-id "$ASSET_ID" \
  --data.assets.assets-dir "$ASSET_DIR" \
  --train-filter-path "$TRAIN_SPLIT" \
  --weight-loader.encoder-params-path "$PATCH_ENCODER_PARAMS" \
  --num-train-steps 1000 \
  --batch-size 8 \
  --fsdp-devices 4 \
  --num-workers 0 \
  --lr-schedule.warmup-steps 100 \
  --lr-schedule.peak-lr 2.5e-5 \
  --lr-schedule.decay-steps 1000 \
  --lr-schedule.decay-lr 2.5e-6 \
  --save-interval 250 \
  --keep-period 250 \
  --eval-interval 250 \
  --eval-num-batches 2 \
  --eval-batch-size 8 \
  --eval-num-workers 0 \
  --eval-repo-id "$DATA_REPO" \
  --eval-asset-id "$ASSET_ID" \
  --eval-assets-dir "$ASSET_DIR" \
  --eval-filter-path "$VAL_SPLIT" \
  --no-wandb-enabled \
  > logs/pi05_tactile_ttt_v1_0831.log 2>&1 &

### pi05
  setsid nohup env \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/workspace/mnt/sqzhang26/FactileLDM/env/.venv/bin/python scripts/train.py \
  pi05_xhand_full_finetune_h16 \
  --exp-name pi05_press_0829 \
  --data.repo-id "$DATA_REPO" \
  --data.assets.asset-id "$ASSET_ID" \
  --data.assets.assets-dir "$ASSET_DIR" \
  --train-filter-path "$TRAIN_SPLIT" \
  --num-train-steps 30000 \
  --batch-size 4 \
  --fsdp-devices 1 \
  --num-workers 0 \
  --lr-schedule.warmup-steps 1000 \
  --lr-schedule.peak-lr 2.5e-5 \
  --lr-schedule.decay-steps 30000 \
  --lr-schedule.decay-lr 2.5e-6 \
  --save-interval 2500 \
  --keep-period 5000 \
  --eval-interval 1000 \
  --eval-num-batches 2 \
  --eval-batch-size 4 \
  --eval-num-workers 0 \
  --eval-repo-id "$DATA_REPO" \
  --eval-asset-id "$ASSET_ID" \
  --eval-assets-dir "$ASSET_DIR" \
  --eval-filter-path "$VAL_SPLIT" \
  --no-wandb-enabled \
  > logs/pi05_press_0829.log 2>&1 &


## pi05+16帧历史
setsid nohup env \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/workspace/mnt/sqzhang26/FactileLDM/env/.venv/bin/python scripts/train.py \
  pi05_tactile_direct16 \
  --exp-name pi05_tactile_direct16_press_0829 \
  --data.repo-id "$DATA_REPO" \
  --data.assets.asset-id "$ASSET_ID" \
  --data.assets.assets-dir "$ASSET_DIR" \
  --train-filter-path "$TRAIN_SPLIT" \
  --weight-loader.encoder-params-path "$PATCH_ENCODER_PARAMS" \
  --num-train-steps 30000 \
  --batch-size 4 \
  --fsdp-devices 1 \
  --num-workers 0 \
  --lr-schedule.warmup-steps 1000 \
  --lr-schedule.peak-lr 2.5e-5 \
  --lr-schedule.decay-steps 30000 \
  --lr-schedule.decay-lr 2.5e-6 \
  --save-interval 2500 \
  --keep-period 5000 \
  --eval-interval 1000 \
  --eval-num-batches 2 \
  --eval-batch-size 4 \
  --eval-num-workers 0 \
  --eval-repo-id "$DATA_REPO" \
  --eval-asset-id "$ASSET_ID" \
  --eval-assets-dir "$ASSET_DIR" \
  --eval-filter-path "$VAL_SPLIT" \
  --no-wandb-enabled \
  > logs/pi05_tactile_direct16_press_0829.log 2>&1 &



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
aws s3 cp /Users/babyna/FactileLDM/data/press_button_4_times s3://sqzhang26-2/press_button_4_times  \
  --recursive \
  --endpoint-url https://eos-huhehaote-1.cmecloud.cn


# 大文件上传  mac支持\
s3cmd put /Users/babyna/TactileTTT/data/press_button_4_times_merged_filtered.tar.gz \
  s3://sqzhang26-2/press_button_4_times_merged_filtered.tar.gz



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
