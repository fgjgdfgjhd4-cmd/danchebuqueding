import os
# 设置环境变量，允许 OpenMP 库重复加载 (防止 Windows 下 Numpy/Torch 冲突报错)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import gymnasium as gym
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# --- 导入自定义模块 ---
# 确保这些文件在同一目录下
from uncertain_env import UncertainComplexEnv 
from rmmf_model import RMMF_ActorCritic
from dapo_agent import DAPO_Algorithm 

# ==========================================
# 1. 辅助函数
# ==========================================

def make_env(rank, max_steps):
    """环境工厂函数，用于创建并行环境"""
    def _thunk():
        env = UncertainComplexEnv() 
        env.max_steps = max_steps
        # 注意：RMMF 模型内部已集成动作缩放逻辑，不需要 ActionScalingWrapper
        return env
    return _thunk

def pad_collate(batch_data, device):
    """
    数据整理函数：将变长的轨迹列表填充(Pad)为张量 Batch。
    
    [DAPO 核心适配]:
    1. log_probs: PPO 裁剪计算的基础。
    2. success: 用于 DAPO 的动态采样 (Dynamic Sampling)，过滤掉全成/全败的组。
    3. lengths: 用于 DAPO 的软性超长惩罚 (Soft Overlong Punishment)。
    4. rewards: 整条轨迹的总回报 (Outcome Reward) 用于组内优势计算。
    """
    lengths = [len(traj['rewards']) for traj in batch_data]
    max_len = max(lengths)
    batch_size = len(batch_data)
    
    # 初始化 Padded Tensors
    padded_obs = torch.zeros(batch_size, max_len, 24).to(device)
    padded_actions = torch.zeros(batch_size, max_len, 2).to(device)
    padded_old_log_probs = torch.zeros(batch_size, max_len).to(device) # [必须]
    
    # 获取整条轨迹的总得分 (Outcome Reward)
    episode_returns = torch.zeros(batch_size).to(device)
    
    # 获取成功标记 (用于 DAPO 动态采样过滤)
    success_flags = torch.zeros(batch_size, dtype=torch.bool).to(device)
    
    # 获取轨迹长度 (用于 DAPO 软性超长惩罚)
    traj_lengths = torch.tensor(lengths, dtype=torch.float32).to(device)
    
    # 初始 hidden state: (1, Batch, Hidden)
    start_hiddens = torch.cat([traj['start_hidden'] for traj in batch_data], dim=1).to(device)
    
    # 填充数据
    for i, traj in enumerate(batch_data):
        l = lengths[i]
        padded_obs[i, :l, :] = torch.stack(traj['obs'])
        padded_actions[i, :l, :] = torch.stack(traj['actions'])
        padded_old_log_probs[i, :l] = torch.stack(traj['log_probs']) # [必须] 填充 log_probs
        
        episode_returns[i] = sum(traj['rewards'])
        success_flags[i] = traj['success']
        
    return {
        'obs': padded_obs,              # (Batch, Seq, 24)
        'actions': padded_actions,      # (Batch, Seq, 2)
        'log_probs': padded_old_log_probs, # [修复] 补上缺失的 log_probs，键名必须与 Agent 匹配
        'rewards': episode_returns,     # (Batch, ) -> 稍后需 Reshape 为 (Groups, Group_Size)
        'hidden_states': start_hiddens, # (1, Batch, Hidden)
        'success': success_flags,       # (Batch, ) -> 稍后需 Reshape
        'lengths': traj_lengths         # (Batch, ) -> 稍后需 Reshape
    }

# ==========================================
# 2. DAPO 训练主循环
# ==========================================
def train():
    # --- A. 初始化配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练参数配置 
    CONFIG = {
        "NUM_ENVS": 16,             # 并行环境数量 (即组大小 Group Size, G)
        "TOTAL_EPISODES": 3000,     # 总训练轮数 (适配四阶段课程)
        "MAX_EPISODE_STEPS": 500,   # 单个 Episode 最大步数 (L_max)
        
        "UPDATE_FREQ": 2,           # 更新频率: 每收集 2 个 Episode (即 2 个 Group) 更新一次
        "UPDATE_EPOCHS": 4,         # 每次更新的迭代次数
        
        "HIDDEN_DIM": 128,          # RMMF 隐层维度
        
        # DAPO 特有参数
        "LR": 1e-6,                 # 学习率 (DAPO 论文建议较小)
        "EPS_LOW": 0.2,             # 下限裁剪
        "EPS_HIGH": 0.28,           # 上限裁剪 (鼓励探索)
        "CACHE_LEN": 200,           # 软性惩罚的缓冲长度 (L_cache)
        
        # 课程阶段阈值 (完全对齐 AC-GDPO)
        "STAGE_ONE_EPISODE": 100,
        "STAGE_TWO_EPISODE": 400,
        "STAGE_THREE_EPISODE": 900
    }

    # 路径与日志
    run_name = f"DAPO_Curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"results/{run_name}"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"--- DAPO Training Started: {run_name} using {device} ---")

    # --- B. 环境与模型初始化 ---
    
    # 1. 创建并行训练环境
    # AsyncVectorEnv 允许我们在同一组参数下并行采集 G 条轨迹
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, CONFIG["MAX_EPISODE_STEPS"]) for i in range(CONFIG["NUM_ENVS"])]
    )

    # 2. 创建可视化/验证环境 (单进程)
    eval_env = UncertainComplexEnv(render_mode="human")
    eval_env.max_steps = CONFIG["MAX_EPISODE_STEPS"]

    # 3. 初始化 RMMF 模型
    model = RMMF_ActorCritic(observation_dim=24, action_dim=2, hidden_dim=CONFIG["HIDDEN_DIM"])
    
    # 4. 初始化 DAPO 代理
    algo = DAPO_Algorithm(
        model=model,
        lr=CONFIG["LR"],
        eps_low=CONFIG["EPS_LOW"],
        eps_high=CONFIG["EPS_HIGH"],
        max_steps=CONFIG["MAX_EPISODE_STEPS"],
        cache_len=CONFIG["CACHE_LEN"],
        device=device
    )
    
    # 将模型送入 GPU
    algo.model.to(device)

    # --- C. 状态变量初始化 ---
    hidden_actor = torch.zeros(1, CONFIG["NUM_ENVS"], CONFIG["HIDDEN_DIM"]).to(device)
    batch_buffer = [] 
    
    best_reward = -float('inf')
    best_stage3_reward = -float('inf')

    # 环境复位
    obs, _ = envs.reset()

    # --- D. 主循环 ---
    for episode_idx in range(1, CONFIG["TOTAL_EPISODES"] + 1):
        
        # =================================================
        # 1. 课程学习：四阶段设置 (与 AC-GDPO 一致)
        # =================================================
        dist_range = (10.0, 30.0)
        mode_name = "Init"

        if episode_idx <= CONFIG["STAGE_ONE_EPISODE"]:
            # Stage 1: Easy Fixed (10-30m range effectively)
            cur_start_pos = [65.0, 40.0]
            cur_target_pos = [55.0, 40.0]
            mode_name = "Stage 1: Easy"
            
        elif episode_idx <= CONFIG["STAGE_TWO_EPISODE"]:
            # Stage 2: Medium Fixed
            cur_start_pos = [65.0, 45.0] 
            cur_target_pos = [30.0, 40.0]
            mode_name = "Stage 2: Medium"
            
        elif episode_idx <= CONFIG["STAGE_THREE_EPISODE"]:
            # Stage 3: Hard Fixed (Cross Map)
            cur_start_pos = [10.0, 10.0]
            cur_target_pos = [90.0, 96.0]
            mode_name = "Stage 3: Hard"
            
        else:
            # Stage 4: Random Long Distance (>100m)
            dist_range = (100.0, 130.0) 
            mode_name = "Stage 4: Random Long (>100m)"
            
            # 利用 eval_env 寻找合法点 (避开障碍物)
            while True:
                s_x = np.random.uniform(5.0, eval_env.unwrapped.width - 5.0)
                s_y = np.random.uniform(5.0, eval_env.unwrapped.height - 5.0)
                
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(dist_range[0], dist_range[1])
                
                t_x = s_x + dist * np.cos(angle)
                t_y = s_y + dist * np.sin(angle)
                
                start_valid = not eval_env.unwrapped._check_static_collision(np.array([s_x, s_y]))
                target_valid = not eval_env.unwrapped._check_static_collision(np.array([t_x, t_y]))
                
                if start_valid and target_valid:
                    break 

            cur_start_pos = [s_x, s_y]
            cur_target_pos = [t_x, t_y]

        task_dist = np.linalg.norm(np.array(cur_start_pos) - np.array(cur_target_pos))

        # 广播任务配置：确保这一轮的 G 个环境使用相同的起点和终点，
        # 这样它们构成的 Group 才是针对同一个 Problem 的。
        obs, _ = envs.reset(seed=episode_idx+100, options={
            'start_pos': cur_start_pos,
            'target_pos': cur_target_pos
        })
        
        # 每轮清理 RNN 隐状态
        hidden_actor.fill_(0.0)

        # =================================================
        # 2. 数据收集 (Collect Rollouts)
        # =================================================
        current_group_trajectories = [
            {'obs': [], 'actions': [], 'rewards': [], 'log_probs': [], 'start_hidden': None, 'active': True, 'success': False} 
            for _ in range(CONFIG["NUM_ENVS"])
        ]
        
        # 记录初始 Hidden State (用于 RNN 训练)
        for i in range(CONFIG["NUM_ENVS"]):
            current_group_trajectories[i]['start_hidden'] = hidden_actor[:, i:i+1, :].clone()
        
        active_envs_count = CONFIG["NUM_ENVS"]
        current_steps = 0
        success_count = 0
        
        while active_envs_count > 0 and current_steps < CONFIG["MAX_EPISODE_STEPS"]:
            current_steps += 1
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            with torch.no_grad():
                # 使用 RMMF 模型推理
                scaled_action, raw_action, log_prob, next_h_a, _ = algo.model.get_action(
                    obs_tensor, hidden_actor, deterministic=False
                )
            
            next_obs, rewards, terminations, truncations, infos = envs.step(scaled_action.cpu().numpy())
            dones = np.logical_or(terminations, truncations)
            
            for i in range(CONFIG["NUM_ENVS"]):
                if current_group_trajectories[i]['active']:
                    traj = current_group_trajectories[i]
                    
                    traj['obs'].append(obs_tensor[i].clone())
                    traj['actions'].append(raw_action[i].clone())
                    traj['log_probs'].append(log_prob[i].clone()) # [关键] 记录 log_prob
                    traj['rewards'].append(rewards[i])
                    
                    if dones[i]:
                        traj['active'] = False
                        active_envs_count -= 1
                        
                        is_success = False
                        if "final_info" in infos:
                            final_info = infos["final_info"][i]
                            if final_info:
                                is_success = final_info.get("is_success", False)
                        
                        traj['success'] = is_success
                        if is_success:
                            success_count += 1

            obs = next_obs
            hidden_actor = next_h_a
            
        # =================================================
        # 3. 数据存入缓冲区
        # =================================================
        batch_buffer.extend(current_group_trajectories)
        
        # 计算当轮统计指标
        current_rewards = [sum(t['rewards']) for t in current_group_trajectories]
        avg_ep_reward = np.mean(current_rewards)
        avg_ep_len = np.mean([len(t['rewards']) for t in current_group_trajectories])
        success_rate = (success_count / CONFIG["NUM_ENVS"]) * 100.0

        # =================================================
        # 4. 触发 DAPO 网络更新
        # =================================================
        update_happened = False
        update_info = {}

        if episode_idx % CONFIG["UPDATE_FREQ"] == 0:
            # 1. 整理数据为张量 (包含 log_probs, success, lengths)
            batch = pad_collate(batch_buffer, device)
            
            # 2. [关键步骤] 手动 Reshape 组相关数据
            # DAPO 需要知道哪些数据属于同一个 Group (同一个 Task 的并行采样)
            # 形状变换: (Total_Batch, ) -> (Num_Groups, Group_Size)
            num_groups = batch['rewards'].shape[0] // CONFIG["NUM_ENVS"]
            
            grouped_rewards = batch['rewards'].view(num_groups, CONFIG["NUM_ENVS"])
            grouped_success = batch['success'].view(num_groups, CONFIG["NUM_ENVS"])
            grouped_lengths = batch['lengths'].view(num_groups, CONFIG["NUM_ENVS"])
            
            # 3. 构造存储字典
            storage = {
                'obs': batch['obs'],             # Flat: (B, T, D)
                'actions': batch['actions'],     # Flat: (B, T, A)
                'log_probs': batch['log_probs'], # Flat: (B, T) [确保存在]
                'hidden_states': batch['hidden_states'], # Flat
                'rewards': grouped_rewards,      # Reshaped: (G_num, G_size)
                'success': grouped_success,      # Reshaped: (G_num, G_size)
                'lengths': grouped_lengths       # Reshaped: (G_num, G_size)
            }
            
            for epoch in range(CONFIG["UPDATE_EPOCHS"]):
                # 调用 DAPO update
                # 注意：确保您的 dapo_agent.py 已修复了 soft_overlong_punishment 中的 RuntimeError
                info = algo.update(storage)
                
                # update 可能返回 None (如果所有组都被动态采样过滤掉了)
                if info:
                    update_info = info
            
            batch_buffer = [] 
            update_happened = True

        # =================================================
        # 5. 日志与保存
        # =================================================
        log_str = f"Ep: {episode_idx}/{CONFIG['TOTAL_EPISODES']} | {mode_name} | Rew: {avg_ep_reward:.2f} | Len: {avg_ep_len:.1f} | SR: {success_rate:.1f}%"
        
        if update_happened:
            if update_info:
                loss_val = update_info.get('loss', 0)
                log_str += f" | Loss: {loss_val:.3f}"
                writer.add_scalar("Train/Loss", loss_val, episode_idx)
            else:
                log_str += " | Skipped (No valid groups)"
        else:
             log_str += " | Collecting..."

        print(log_str)
        
        writer.add_scalar("Train/Episode_Reward", avg_ep_reward, episode_idx)
        writer.add_scalar("Train/Success_Rate", success_rate, episode_idx)
        writer.add_scalar("Curriculum/Distance", task_dist, episode_idx)

        # --- 模型保存 ---
        # 1. 保存当前最佳
        if avg_ep_reward > best_reward:
            best_reward = avg_ep_reward
            save_path = os.path.join(model_dir, "best_model.pth")
            torch.save(algo.model.state_dict(), save_path)

        # 2. 保存 Stage 4 (长距离) 最佳
        if episode_idx > CONFIG["STAGE_THREE_EPISODE"]:
            if avg_ep_reward > best_stage3_reward:
                best_stage3_reward = avg_ep_reward
                save_path = os.path.join(model_dir, "best_stage4_model.pth")
                torch.save(algo.model.state_dict(), save_path)
                print(f"  >>> New Best Stage 4 Model! Reward: {best_stage3_reward:.2f}")

        # 3. 定期保存检查点
        if episode_idx % 200 == 0:
            save_path = os.path.join(model_dir, f"ckpt_ep_{episode_idx}.pth")
            torch.save(algo.model.state_dict(), save_path)

        # =================================================
        # 6. 可视化演示
        # =================================================
        if episode_idx % 20 == 0:
            print(f"\n[Visual] Rendering {mode_name} (Dist: {task_dist:.1f}m)...")
            
            vis_obs, _ = eval_env.reset(seed=42, options={
                'start_pos': cur_start_pos,
                'target_pos': cur_target_pos
            })
            
            vis_h_a = torch.zeros(1, 1, CONFIG["HIDDEN_DIM"]).to(device)
            vis_done = False
            vis_steps = 0
            
            while not vis_done and vis_steps < CONFIG["MAX_EPISODE_STEPS"]:
                eval_env.render()
                vis_obs_tensor = torch.FloatTensor(vis_obs).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # 可视化时使用 deterministic=True，展示策略的最优解
                    vis_action, _, _, vis_next_h_a, _ = algo.model.get_action(
                        vis_obs_tensor, vis_h_a, deterministic=True
                    )
                
                vis_obs, _, term, trunc, _ = eval_env.step(vis_action[0].cpu().numpy())
                
                vis_h_a = vis_next_h_a
                vis_done = term or trunc
                vis_steps += 1
            print(f"[Visual] Finished in {vis_steps} steps.\n")

    envs.close()
    eval_env.close()
    writer.close()
    print("Training finished!")

if __name__ == "__main__":
    train()