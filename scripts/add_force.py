from datasets import Dataset, concatenate_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.compute_stats import compute_episode_stats, aggregate_stats
from lerobot.common.datasets.utils import (
    write_info, 
    write_stats,
    write_episode_stats
)
from datasets import load_dataset, Dataset
import argparse
import sys, os
file_path = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(file_path)+"/")

FORCE_THRESHOLD = 2.5
MAX_FREQ = 4.0           #VLM:AE opration freq, need to be same as model code
ACTION_HORIZON = 64
VOL_SMOOTH_SPAN= 40  # if None will not do EMA on force var

def debug():
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("✅ Waiting for debugger to attach on port 5678...")
    debugpy.wait_for_client()
    print("Start to debugging")
    
def modify_dataset_inplace(repo_id: str):
    """
    原地修改 LeRobot 数据集：
    1. 在 info.json 中注册新特征。
    2. 遍历所有 episode，计算 effort 和 is_contact，覆盖 Parquet 文件。
    3. 重新计算并保存所有统计数据 (stats.json)。
    """
    print(f"Loading metadata for {repo_id}...")
    # 加载现有元数据
    meta = LeRobotDatasetMetadata(repo_id)
    
    old_episodes_stats = {k: v.copy() for k, v in meta.episodes_stats.items()}
    print(f"Loaded existing stats for {len(old_episodes_stats)} episodes.")
    # ---------------------------------------------------------
    # 1. 更新 info.json 中的 features 定义
    # ---------------------------------------------------------
    print("Updating feature definitions in metadata...")
    
    # 定义 effort 特征 (假定是 6 维: Fx, Fy, Fz, Tx, Ty, Tz)
    # 如果你的机器人自由度不同，请修改 shape 和 names
    new_features = {
        "observation.effort": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["effort_0", "effort_1", "effort_2", "effort_3", "effort_4", "effort_5"]
        },
        "observation.is_contact": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_contact"]
        },
        "observation.fut_aefreq": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["fut_aefreq", "fut_force_mean"]
        }
    }
    
    for k, v in new_features.items():
        meta.info["features"][k] = v
    # 兼容历史版本：移除独立字段定义，改为并入 observation.fut_aefreq 的第二维
    meta.info["features"].pop("observation.fut_force_mean", None)
    
    # 立即把新的 info 写回磁盘，这样后续流程最稳妥
    write_info(meta.info, meta.root)

    # ---------------------------------------------------------
    # 2. 遍历处理 Parquet 并重新计算 Stats
    # ---------------------------------------------------------
    all_merged_stats = {}
    
    # 删除旧文件，防止追加重复内容
    ep_stats_path = meta.root / "meta/episodes_stats.jsonl"
    if ep_stats_path.exists():
        print(f"Removing existing {ep_stats_path} to rebuild it with merged data...")
        ep_stats_path.unlink()
    
    print(f"Processing {meta.total_episodes} episodes...")
    
    for ep_idx in tqdm(range(meta.total_episodes), desc="Modifying Episodes"):
        parquet_path = meta.root / meta.get_data_file_path(ep_idx)
        
        # --- 读取数据 ---
        hf_ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
        data_dict = hf_ds.to_dict()
        
        # --- 计算新字段 ---
        state_data = np.array(data_dict["observation.state"])
        effort = state_data[:, 7:13] # (T, 6)
        
        current_forces = effort[:, :3]
        init_force = current_forces[0]
        force_change_magnitude = np.linalg.norm(current_forces - init_force, axis=1)
        is_contact = force_change_magnitude > FORCE_THRESHOLD
        
        # 更新数据字典用于保存 Parquet
        data_dict["observation.effort"] = effort.astype(np.float32)
        data_dict["observation.is_contact"] = is_contact

        
        # --- 保存 Parquet ---
        new_hf_ds = Dataset.from_dict(data_dict)
        new_hf_ds.to_parquet(parquet_path)
        
        # --- 【关键步骤 2】 只计算新增字段的统计数据 ---
        # 我们只构建一个包含新字段的局部字典
        partial_data_numpy = {
            "observation.effort": effort.astype(np.float32),
            "observation.is_contact": is_contact,
        }
        
        # 对应的特征定义也只取这两个
        partial_features_def = {
            k: meta.info["features"][k] for k in partial_data_numpy.keys()
        }
        
        # 计算增量统计 (incremental stats)
        # 这样就不会涉及到 image/video，也就不会报错，也不会覆盖旧数据
        incremental_stats = compute_episode_stats(partial_data_numpy, partial_features_def)
        
        # --- 【关键步骤 3】 合并旧统计数据和新统计数据 ---
        # 获取当前 episode 的旧统计 (如果是个新数据集可能为空，给个默认 {})
        merged_stats = old_episodes_stats.get(ep_idx, {})
        
        # 将新算出来的 effort 和 is_contact 的统计 update 进去
        merged_stats.update(incremental_stats)
        
        # 记录到内存列表用于最后聚合
        all_merged_stats[ep_idx] = merged_stats
        
        # 写入文件 (包含：旧的图片stats + 新的力控stats)
        write_episode_stats(ep_idx, merged_stats, meta.root)

    # ---------------------------------------------------------
    # 3. 聚合全局 Stats
    # ---------------------------------------------------------
    print("Aggregating statistics...")
    # aggregate_stats 会自动处理所有的 key
    stats = aggregate_stats(list(all_merged_stats.values()))
    write_stats(stats, meta.root)
    
    print("\n✅ Dataset modification complete!")
    print("Checked: Old image stats should be preserved in episodes_stats.jsonl")


if __name__ == "__main__":
    # debug()
    # repo_id = "llly/usbinsert_v2_10_28"
    # repo_id = "llly/ChargerPlug"
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id")
    args = parser.parse_args()
    repo_id = args.repo_id
    modify_dataset_inplace(repo_id)
