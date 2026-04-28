
import os
# 设置环境变量，允许 OpenMP 库重复加载 (解决 Windows 下 Numpy/Torch 冲突报错)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import gymnasium as gym
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# --- 导入自定义模块 ---
# 确保这些文件在同一目录下
from uncertain_env import UncertainComplexEnv
from rmmf_model import RMMF_ActorCritic
from ac_gdpo_agent import AC_GDPO_Agent
from dsa_config import build_dsa_masker_from_config, make_dsa_config, save_dsa_config

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    # --- 训练规模 ---
    "TOTAL_EPISODES": 1500,       # 总训练轮数 (适配三阶段课程)
    "MAX_STEPS": 500,            # 单个 Episode 最大步数
    "GROUP_SIZE": 16,            # 组大小 (并行环境数量)，AC-GDPO 核心参数
    
    # --- 算法参数 ---
    "LR": 3e-4,                  # 学习率
    "GAMMA": 0.99,               # 折扣因子
    "CLIP_LOW": 0.2,             # DAPO 参数: 负优势裁剪下界
    "CLIP_HIGH": 0.28,           # DAPO 参数: 正优势裁剪上界 (鼓励探索)
    "UPDATE_EPOCHS": 4,          # 每次采集后网络更新次数
    "UPDATE_FREQ": 2,            # 更新频率: 每 2 个 Episode 更新一次
    
    # --- 模型参数 ---
    "HIDDEN_DIM": 128,           # RMMF 模型 GRU 隐层维度
    
    # --- 系统 ---
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "VISUALIZE_FREQ": 10,         # 可视化频率
    "SAVE_DIR": "./results",
    "USE_DSA_MASK": True,
    "DSA_CONFIG": make_dsa_config(floor_gain=0.25),

    "STAGE_ONE_EPISODE": 100,
    "STAGE_TWO_EPISODE": 400,
    "STAGE_THREE_EPISODE": 900

}

# ==========================================
# 2. 辅助函数
# ==========================================

def make_env(rank):
    """环境工厂函数，用于创建并行环境"""
    def _thunk():
        env = UncertainComplexEnv()
        # 注意：不需要 ActionScalingWrapper，RMMF 模型内部已集成缩放逻辑
        return env
    return _thunk

def build_dsa_masker():
    return build_dsa_masker_from_config(CONFIG["DSA_CONFIG"])


def pad_collate(batch_data, device):
    """
    数据整理函数：将变长的轨迹列表填充(Pad)为张量 Batch
    用于 RNN 模型的全序列训练
    """
    # 获取最大长度
    lengths = [len(traj['rewards']) for traj in batch_data]
    max_len = max(lengths)
    batch_size = len(batch_data)
    
    # 初始化 Padded Tensors
    padded_obs = torch.zeros(batch_size, max_len, 24).to(device)
    padded_actions = torch.zeros(batch_size, max_len, 2).to(device)
    padded_old_log_probs = torch.zeros(batch_size, max_len).to(device)
    padded_returns = torch.zeros(batch_size, max_len).to(device)
    padded_advantages = torch.zeros(batch_size, max_len).to(device)
    
    # 初始 hidden state: (1, Batch, Hidden)
    # 注意：这里需要拼接所有轨迹的初始 hidden
    start_hiddens = torch.cat([traj['start_hidden'] for traj in batch_data], dim=1).to(device)
    
    # 填充数据
    for i, traj in enumerate(batch_data):
        l = lengths[i]
        padded_obs[i, :l, :] = torch.stack(traj['obs'])
        padded_actions[i, :l, :] = torch.stack(traj['actions'])
        padded_old_log_probs[i, :l] = torch.stack(traj['log_probs'])
        padded_returns[i, :l] = traj['returns']
        padded_advantages[i, :l] = traj['norm_advantages']
        
    # 打包成 DataLoader 格式 (这里手动构造一个全量 Batch)
    return [{
        'obs': padded_obs,
        'actions': padded_actions,
        'log_probs': padded_old_log_probs,
        'returns': padded_returns,
        'norm_advantages': padded_advantages,
        'hidden_states': start_hiddens
    }]

# ==========================================
# 3. 训练主程序
# ==========================================
def train():
    # --- 初始化 ---
    run_name = f"AC_GDPO_Curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(CONFIG["SAVE_DIR"], run_name)
    os.makedirs(log_dir, exist_ok=True)
    if CONFIG["USE_DSA_MASK"]:
        dsa_config_path = save_dsa_config(CONFIG["DSA_CONFIG"], log_dir)
        print(f"--- DSA config saved to: {dsa_config_path} ---")
    
    writer = SummaryWriter(log_dir)
    print(f"--- Training Started: {run_name} using {CONFIG['DEVICE']} ---")
    print(f"--- Update Frequency: Every {CONFIG['UPDATE_FREQ']} Episodes ---")

    # 1. 创建并行环境 (AsyncVectorEnv)
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(CONFIG["GROUP_SIZE"])])
    
    # 2. 创建可视化/验证环境 (单进程)
    eval_env = UncertainComplexEnv(render_mode="human")

    # 3. 创建 RMMF 模型与 AC-GDPO 智能体
    model = RMMF_ActorCritic(
        observation_dim=24, 
        action_dim=2, 
        hidden_dim=CONFIG["HIDDEN_DIM"]
    )
    
    agent = AC_GDPO_Agent(
        model=model,
        lr=CONFIG["LR"],
        gamma=CONFIG["GAMMA"],
        clip_low=CONFIG["CLIP_LOW"],
        clip_high=CONFIG["CLIP_HIGH"],
        device=CONFIG["DEVICE"]
    )
    dsa_masker = build_dsa_masker() if CONFIG["USE_DSA_MASK"] else None

    # 记录最佳模型
    best_reward = -float('inf')
    best_stage3_reward = -float('inf') # 专门记录 Stage 3 的最佳表现
    
    # GRU 状态初始化: (Num_Layers=1, Batch=Group_Size, Hidden)
    hidden_states = torch.zeros(1, CONFIG["GROUP_SIZE"], CONFIG["HIDDEN_DIM"]).to(CONFIG["DEVICE"])

    # 数据缓冲区：用于积累多个 Episode 的组数据
    batch_groups_buffer = []

    # --- 主循环 ---
    for episode in range(1, CONFIG["TOTAL_EPISODES"] + 1):
        
        # =================================================
        # [核心逻辑] 4阶段课程设置 (Task Configuration)
        # =================================================
        if episode <= CONFIG["STAGE_ONE_EPISODE"]:
            cur_start_pos = [65.0, 40.0]
            cur_target_pos = [55.0, 40.0]
            mode_name = "Stage 1: Easy"
            # --- 阶段一: 短距离随机 (10m - 30m) ---
            # 目的：让智能体学会基本的移动和短程避障
            # dist_range = (10.0, 30.0)
            # mode_name = "Stage 1: Random Short (10-30m)"
            
        elif episode <= CONFIG["STAGE_TWO_EPISODE"]:
            cur_start_pos = [65.0, 45.0] # 修正后的起点，避开障碍物
            cur_target_pos = [30.0, 40.0]
            # cur_start_pos = [50.0, 35.0]
            # cur_target_pos = [10.0, 90.0]
            mode_name = "Stage 2: Medium"
            # --- 阶段二: 中距离随机 (50m - 70m) ---
            # 目的：增加导航难度，可能会遇到更多障碍物
            # dist_range = (50.0, 70.0)
            # mode_name = "Stage 2: Random Medium (50-70m)"
            
        elif episode <= CONFIG["STAGE_THREE_EPISODE"]:
            cur_start_pos = [10.0, 10.0]
            cur_target_pos = [90.0, 96.0]
            mode_name = "Stage 3: Hard"
            # --- 阶段三: 长距离挑战 (>100m) ---
            # 目的：在 100x100 的地图中，>100m 意味着必须从一端穿越到另一端 (最大对角线约 141m)
            # 这极大概率会遇到复杂的陷阱和死胡同
            # dist_range = (100.0, 130.0) 
            # mode_name = "Stage 3: Random Long (>100m)"
        else:
            dist_range = (100.0, 130.0) 
            mode_name = "Stage 4: Random Long (>100m)"

            while True:
                s_x = np.random.uniform(5.0, eval_env.width - 5.0)
                s_y = np.random.uniform(5.0, eval_env.height - 5.0)
                
                # 2. 随机生成方向和距离
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(dist_range[0], dist_range[1])
                
                # 3. 计算潜在终点
                t_x = s_x + dist * np.cos(angle)
                t_y = s_y + dist * np.sin(angle)
                
                if not eval_env._check_static_collision(np.array([s_x, s_y])) and not eval_env._check_static_collision(np.array([t_x, t_y])):
                    break

            cur_start_pos = [s_x, s_y]
            cur_target_pos = [t_x, t_y]

        # 构建任务配置
        task_config = {
            'start_pos': cur_start_pos,
            'target_pos': cur_target_pos
        }
        
        task_dist = np.linalg.norm(np.array(cur_start_pos) - np.array(cur_target_pos))

        # =================================================
        # 1. 环境重置与任务广播
        # =================================================
        obs, _ = envs.reset(options=task_config)
        hidden_states.fill_(0.0)
        if dsa_masker is not None:
            dsa_masker.reset(CONFIG["GROUP_SIZE"], torch.device(CONFIG["DEVICE"]))
        
        # 存储本轮 Group 的数据容器
        current_group_trajectories = [
            {'obs': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': [], 'dones': [], 
             'start_hidden': None, 'active': True, 'success': False,
             'uncertainty_penalties': [], 'risk_min_clearances': [], 'risk_margins': []} 
            for _ in range(CONFIG["GROUP_SIZE"])
        ]
        
        for i in range(CONFIG["GROUP_SIZE"]):
            current_group_trajectories[i]['start_hidden'] = hidden_states[:, i:i+1, :].clone()

        step_count = 0
        active_envs_count = CONFIG["GROUP_SIZE"]
        
        # =================================================
        # 2. 数据采集 (Rollout)
        # =================================================
        while active_envs_count > 0 and step_count < CONFIG["MAX_STEPS"]:
            step_count += 1
            
            obs_tensor = torch.from_numpy(obs).float().to(CONFIG["DEVICE"])
            if dsa_masker is not None:
                obs_tensor = dsa_masker.apply(obs_tensor)
            
            with torch.no_grad():
                scaled_action, raw_action, log_prob, next_hidden, value = agent.get_action(
                    obs_tensor, hidden_states
                )
            if dsa_masker is not None:
                dsa_masker.update_action_history(scaled_action)
            
            cpu_actions = scaled_action.cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = envs.step(cpu_actions)
            
            dones = np.logical_or(terminations, truncations)
            
            for i in range(CONFIG["GROUP_SIZE"]):
                if current_group_trajectories[i]['active']:
                    traj = current_group_trajectories[i]
                    traj['obs'].append(obs_tensor[i].clone())
                    traj['actions'].append(raw_action[i].clone())
                    traj['log_probs'].append(log_prob[i].clone())
                    traj['values'].append(value[i].item())
                    traj['rewards'].append(rewards[i])
                    traj['dones'].append(dones[i])
                    if "uncertainty_penalty" in infos:
                        traj['uncertainty_penalties'].append(float(infos["uncertainty_penalty"][i]))
                    if "risk_min_clearance" in infos:
                        traj['risk_min_clearances'].append(float(infos["risk_min_clearance"][i]))
                    if "risk_margin" in infos:
                        traj['risk_margins'].append(float(infos["risk_margin"][i]))
                    
                    if dones[i]:
                        traj['active'] = False
                        active_envs_count -= 1
                        
                        is_success = False
                        # Gymnasium VectorEnv 会将结束时的 info 放入 "final_info" 中
                        if "final_info" in infos:
                            has_final_info = True
                            if "_final_info" in infos:
                                has_final_info = bool(infos["_final_info"][i])
                            if has_final_info:
                                final_info = infos["final_info"][i]
                            # 确保 info 不为空 (有些版本可能是 None)
                                if final_info is not None:
                                    is_success = final_info.get("is_success", False)
                        elif "is_success" in infos:
                            is_success = bool(infos["is_success"][i])
                        
                        # 将成功标记存入轨迹字典，供后续统计使用
                        traj['success'] = is_success

                        if truncations[i]:
                            with torch.no_grad():
                                next_obs_tensor = torch.from_numpy(next_obs[i]).float().to(CONFIG["DEVICE"]).unsqueeze(0)
                                if dsa_masker is not None:
                                    next_obs_tensor = dsa_masker.fork(i).apply(next_obs_tensor)
                                _, _, _, _, last_val = agent.get_action(
                                    next_obs_tensor,
                                    next_hidden[:, i:i+1, :]
                                )
                            traj['next_value'] = last_val.item()
                        else:
                            traj['next_value'] = 0.0
            
            obs = next_obs
            hidden_states = next_hidden
            
        # =================================================
        # 数据后处理：防止 KeyError 并处理截断
        # =================================================
        for i in range(CONFIG["GROUP_SIZE"]):
            traj = current_group_trajectories[i] 
            
            if traj['active']:
                # 仍在运行 (达到 Max Steps)，需要截断
                traj['truncated'] = False 
                with torch.no_grad():
                     final_obs_tensor = torch.from_numpy(obs[i]).float().to(CONFIG["DEVICE"]).unsqueeze(0)
                     if dsa_masker is not None:
                         final_obs_tensor = dsa_masker.fork(i).apply(final_obs_tensor)
                     _, _, _, _, last_val = agent.get_action(
                        final_obs_tensor,
                        hidden_states[:, i:i+1, :]
                    )
                traj['next_value'] = last_val.item()
            else:
                # 已经结束，确保 key 存在
                traj['truncated'] = False

        # 将本轮数据加入缓冲区
        batch_groups_buffer.append(current_group_trajectories)


        

        # =================================================
        # 3. 算法更新 (Update Logic)
        # =================================================
        update_happened = False
        train_stats = {} 

        if episode % CONFIG["UPDATE_FREQ"] == 0:
            update_happened = True
            flat_trajs = agent.compute_group_advantages(batch_groups_buffer)
            dataloader = pad_collate(flat_trajs, CONFIG["DEVICE"])
            # 执行更新并获取 Loss 字典
            train_stats = agent.update(dataloader, num_epochs=CONFIG["UPDATE_EPOCHS"])
            batch_groups_buffer = []

        # =================================================
        # 4. 统计与日志 (Statistics & Logging)
        # =================================================
        current_rewards = [sum(t['rewards']) for t in current_group_trajectories]
        avg_reward = np.mean(current_rewards)
        avg_len = np.mean([len(t['rewards']) for t in current_group_trajectories])
        # 简单判断成功率：假设到达终点的奖励必定大于 50 (根据环境设置)
        # success_rate = np.mean([1.0 if r[-1] > 50.0 else 0.0 for r in [t['rewards'] for t in current_group_trajectories]])
        success_rate = np.mean([1.0 if t.get('success', False) else 0.0 for t in current_group_trajectories]) * 100.0
        uncertainty_values = [v for t in current_group_trajectories for v in t['uncertainty_penalties']]
        clearance_values = [v for t in current_group_trajectories for v in t['risk_min_clearances']]
        margin_values = [v for t in current_group_trajectories for v in t['risk_margins']]
        avg_uncertainty_penalty = np.mean(uncertainty_values) if uncertainty_values else 0.0
        avg_min_clearance = np.mean(clearance_values) if clearance_values else 0.0
        avg_risk_margin = np.mean(margin_values) if margin_values else 0.0
        avg_safety_buffer = avg_min_clearance - avg_risk_margin

        # 基础日志字符串
        log_str = f"Ep: {episode}/{CONFIG['TOTAL_EPISODES']} | {mode_name} | Rew: {avg_reward:.2f} | Len: {avg_len:.1f} | SR: {success_rate:.1f}% | Safe: Pen={avg_uncertainty_penalty:.3f} Buf={avg_safety_buffer:.3f}"
        
        if update_happened:
            # [关键新增] 提取并显示训练指标
            # 注意：需确保 ac_gdpo_agent.py 返回的 keys 分别为 loss/actor, loss/critic, loss/entropy
            loss_a = train_stats.get('loss/actor', 0.0)
            loss_c = train_stats.get('loss/critic', 0.0)
            loss_e = train_stats.get('loss/entropy', 0.0)
            loss_kl = train_stats.get('loss/kl', 0.0)
            
            log_str += f" | Loss: A={loss_a:.3f} C={loss_c:.3f} Ent={loss_e:.3f} KL={loss_kl:.4f}"
            
            # 记录到 TensorBoard
            writer.add_scalar("Loss/Actor", loss_a, episode)
            writer.add_scalar("Loss/Critic", loss_c, episode)
            writer.add_scalar("Loss/Entropy", loss_e, episode)
            writer.add_scalar("Loss/KL", loss_kl, episode)
        else:
            log_str += " | Collecting..."

        print(log_str)
        
        # 记录环境指标 (每轮都记)
        writer.add_scalar("Train/Average_Reward", avg_reward, episode)
        writer.add_scalar("Train/Success_Rate", success_rate, episode)
        writer.add_scalar("Curriculum/Distance", task_dist, episode)
        writer.add_scalar("Safety/Uncertainty_Penalty", avg_uncertainty_penalty, episode)
        writer.add_scalar("Safety/Min_Clearance", avg_min_clearance, episode)
        writer.add_scalar("Safety/Risk_Margin", avg_risk_margin, episode)
        writer.add_scalar("Safety/Buffer", avg_safety_buffer, episode)
        
        # --- 保存最佳模型 ---
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(os.path.join(log_dir, "best_model.pth"))

        if episode > CONFIG["STAGE_THREE_EPISODE"] and avg_reward > best_stage3_reward:
            best_stage3_reward = avg_reward
            agent.save(os.path.join(log_dir, "best_stage4_model.pth"))
            print(f"  >>> New Best Stage 4 Model! Reward: {best_stage3_reward:.2f}")

        # =================================================
        # 5. 可视化演示 (Visualization)
        # =================================================
        if episode % CONFIG["VISUALIZE_FREQ"] == 0:
            print(f"\n[Visual] Rendering {mode_name} (Dist: {task_dist:.1f}m)...")
            
            vis_obs, _ = eval_env.reset(options=task_config)
            vis_masker = build_dsa_masker() if CONFIG["USE_DSA_MASK"] else None
            if vis_masker is not None:
                vis_masker.reset(1, torch.device(CONFIG["DEVICE"]))
            
            vis_hidden = torch.zeros(1, 1, CONFIG["HIDDEN_DIM"]).to(CONFIG["DEVICE"])
            vis_done = False
            vis_steps = 0
            
            while not vis_done and vis_steps < CONFIG["MAX_STEPS"]:
                eval_env.render()
                
                vis_obs_tensor = torch.from_numpy(vis_obs).float().to(CONFIG["DEVICE"]).unsqueeze(0)
                if vis_masker is not None:
                    vis_obs_tensor = vis_masker.apply(vis_obs_tensor)
                with torch.no_grad():
                    s_action, _, _, vis_next_hidden, _ = agent.get_action(
                        vis_obs_tensor, vis_hidden, deterministic=True
                    )
                if vis_masker is not None:
                    vis_masker.update_action_history(s_action)
                
                vis_action = s_action.cpu().numpy()[0]
                vis_obs, _, term, trunc, _ = eval_env.step(vis_action)
                
                vis_hidden = vis_next_hidden
                vis_done = term or trunc
                vis_steps += 1
            
            print(f"[Visual] Finished in {vis_steps} steps.\n")

    envs.close()
    eval_env.close()
    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    train()
