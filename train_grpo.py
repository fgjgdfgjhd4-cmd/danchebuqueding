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
from uncertain_env import UncertainComplexEnv 
from rmmf_model import RMMF_ActorCritic
from grpo_agent import GRPO_Algorithm 

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
    数据整理函数：将变长的轨迹列表填充(Pad)为张量 Batch
    [GRPO 特有适配]：直接提取每条轨迹的总奖励 (Outcome Reward) 供计算组优势使用。
    """
    lengths = [len(traj['rewards']) for traj in batch_data]
    max_len = max(lengths)
    batch_size = len(batch_data)
    
    # 初始化 Padded Tensors
    padded_obs = torch.zeros(batch_size, max_len, 24).to(device)
    padded_actions = torch.zeros(batch_size, max_len, 2).to(device)
    
    # 获取整条轨迹的总得分 (Batch, )，这是 GRPO 组内归一化的核心输入
    episode_returns = torch.zeros(batch_size).to(device)
    
    # 初始 hidden state: (1, Batch, Hidden)
    start_hiddens = torch.cat([traj['start_hidden'] for traj in batch_data], dim=1).to(device)
    
    # 填充数据
    for i, traj in enumerate(batch_data):
        l = lengths[i]
        padded_obs[i, :l, :] = torch.stack(traj['obs'])
        padded_actions[i, :l, :] = torch.stack(traj['actions'])  # 记录 raw_action 供 log_prob 计算
        
        # 将该轨迹的所有即时奖励求和，作为整条轨迹的结果回报 (Outcome Reward)
        episode_returns[i] = sum(traj['rewards'])
        
    return {
        'obs': padded_obs,
        'actions': padded_actions,
        'rewards': episode_returns,
        'hidden_states': start_hiddens
    }

# ==========================================
# 2. GRPO 训练主循环
# ==========================================
def train():
    # --- A. 初始化配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练参数配置 
    CONFIG = {
        "NUM_ENVS": 16,             # 并行环境数量 (即 GRPO 论文中的 Group Size - G)
        "TOTAL_EPISODES": 1500,     # 总训练轮数 (适配四阶段课程)
        "MAX_EPISODE_STEPS": 500,   # 单个 Episode 最大步数
        
        "UPDATE_FREQ": 2,           # 更新频率: 每收集 UPDATE_FREQ 个 Episode 更新一次
        "UPDATE_EPOCHS": 4,         # 每次更新时的数据复用迭代次数 (同 PPO)
        
        "HIDDEN_DIM": 128,          # RMMF 隐层维度
        "LR": 1e-4,                 # 学习率
        "BETA": 0.04,               # GRPO 特有: 无偏 KL 散度的惩罚系数
        "EPSILON": 0.2,             # PPO 裁剪范围
        
        # 课程阶段阈值 (完全对齐之前的设置)
        "STAGE_ONE_EPISODE": 100,
        "STAGE_TWO_EPISODE": 400,
        "STAGE_THREE_EPISODE": 900
    }

    # 路径与日志
    run_name = f"GRPO_Curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"results/{run_name}"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"--- GRPO Training Started: {run_name} using {device} ---")

    # --- B. 环境与模型初始化 ---
    
    # 1. 创建并行训练环境
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, CONFIG["MAX_EPISODE_STEPS"]) for i in range(CONFIG["NUM_ENVS"])]
    )

    # 2. 创建可视化/验证环境 (单进程)
    eval_env = UncertainComplexEnv(render_mode="human")
    eval_env.max_steps = CONFIG["MAX_EPISODE_STEPS"]

    # 3. 初始化 RMMF 模型与参考模型
    model = RMMF_ActorCritic(observation_dim=24, action_dim=2, hidden_dim=CONFIG["HIDDEN_DIM"])
    ref_model = copy.deepcopy(model) # 深拷贝作为初始参考模型
    
    # 4. 初始化 GRPO 代理
    algo = GRPO_Algorithm(
        model=model,
        ref_model=ref_model,
        lr=CONFIG["LR"],
        beta=CONFIG["BETA"],
        epsilon=CONFIG["EPSILON"],
        group_size=CONFIG["NUM_ENVS"], # 这里的组大小设定为环境的并行度，因为它们针对的是同一个 task
        device=device
    )
    
    # 将模型送入 GPU/CPU
    algo.model.to(device)
    algo.ref_model.to(device)

    # --- C. 状态变量初始化 ---
    # RMMF 的隐状态 (Layers=1, Batch=NUM_ENVS, Hidden)
    hidden_actor = torch.zeros(1, CONFIG["NUM_ENVS"], CONFIG["HIDDEN_DIM"]).to(device)
    
    batch_buffer = [] # 用于积累多轮的数据供网络更新使用
    
    best_reward = -float('inf')
    best_stage3_reward = -float('inf')

    # 环境复位
    obs, _ = envs.reset()

    # --- D. 主循环 ---
    for episode_idx in range(1, CONFIG["TOTAL_EPISODES"] + 1):
        
        # =================================================
        # 1. 课程学习：设置四阶段的起始点和目标点
        # =================================================
        dist_range = (10.0, 30.0)
        mode_name = "Init"

        if episode_idx <= CONFIG["STAGE_ONE_EPISODE"]:
            cur_start_pos = [65.0, 40.0]
            cur_target_pos = [55.0, 40.0]
            mode_name = "Stage 1: Easy"
            
        elif episode_idx <= CONFIG["STAGE_TWO_EPISODE"]:
            cur_start_pos = [65.0, 45.0] # 修正后的起点，避开障碍物
            cur_target_pos = [30.0, 40.0]
            mode_name = "Stage 2: Medium"
            
        elif episode_idx <= CONFIG["STAGE_THREE_EPISODE"]:
            cur_start_pos = [10.0, 10.0]
            cur_target_pos = [90.0, 96.0]
            mode_name = "Stage 3: Hard"
            
        else:
            dist_range = (100.0, 130.0) 
            mode_name = "Stage 4: Random Long (>100m)"
            
            # 利用 eval_env 寻找不与障碍物重叠的合法起始点/目标点
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

        # 广播任务配置，确保本组所有并行环境都挑战相同的起点和终点 (GRPO要求)
        obs, _ = envs.reset(seed=episode_idx+100, options={
            'start_pos': cur_start_pos,
            'target_pos': cur_target_pos
        })
        
        # 每轮清理 RNN 隐状态记忆
        hidden_actor.fill_(0.0)

        # =================================================
        # 2. 数据收集 (Collect Rollouts)
        # =================================================
        # 为每条并行轨迹创建存储结构
        current_group_trajectories = [
            {'obs': [], 'actions': [], 'rewards': [], 'start_hidden': None, 'active': True} 
            for _ in range(CONFIG["NUM_ENVS"])
        ]
        
        # 记录每条轨迹的初始 GRU 隐状态 (仅在轨迹首步需要)
        for i in range(CONFIG["NUM_ENVS"]):
            current_group_trajectories[i]['start_hidden'] = hidden_actor[:, i:i+1, :].clone()
        
        active_envs_count = CONFIG["NUM_ENVS"]
        current_steps = 0
        success_count = 0
        
        while active_envs_count > 0 and current_steps < CONFIG["MAX_EPISODE_STEPS"]:
            current_steps += 1
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            # 使用 GRPO 算法类内部挂载的模型进行前向推理
            # 注意: GRPO 不需要 Value (Critic)，但为了适配 RMMF 模型这里仍然解包出来
            with torch.no_grad():
                scaled_action, raw_action, _, next_h_a, _ = algo.model.get_action(
                    obs_tensor, hidden_actor, deterministic=False
                )
            
            # 与环境交互 (使用限制物理范围后的 scaled_action)
            next_obs, rewards, terminations, truncations, infos = envs.step(scaled_action.cpu().numpy())
            dones = np.logical_or(terminations, truncations)
            
            for i in range(CONFIG["NUM_ENVS"]):
                if current_group_trajectories[i]['active']:
                    traj = current_group_trajectories[i]
                    
                    # 保存数据。注意: 行动时保存的是 raw_action，这是计算策略概率需要的
                    traj['obs'].append(obs_tensor[i].clone())
                    traj['actions'].append(raw_action[i].clone())
                    traj['rewards'].append(rewards[i])
                    
                    # 统计结束条件和成功率
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
        # 3. 将本轮同组数据合并入缓冲区
        # =================================================
        batch_buffer.extend(current_group_trajectories)
        
        # 实时计算并打印当轮的结果日志
        current_rewards = [sum(t['rewards']) for t in current_group_trajectories]
        avg_ep_reward = np.mean(current_rewards)
        avg_ep_len = np.mean([len(t['rewards']) for t in current_group_trajectories])
        success_rate = (success_count / CONFIG["NUM_ENVS"]) * 100.0

        # =================================================
        # 4. 触发 GRPO 网络更新
        # =================================================
        update_happened = False
        update_info = {}

        if episode_idx % CONFIG["UPDATE_FREQ"] == 0:
            # 数据打补丁与整理 (支持变长轨迹，获得整条轨的总奖励)
            batch = pad_collate(batch_buffer, device)
            
            # 多 Epoch 迭代更新 (提升样本利用率)
            for epoch in range(CONFIG["UPDATE_EPOCHS"]):
                update_info = algo.update(
                    obs=batch['obs'], 
                    actions=batch['actions'], 
                    rewards=batch['rewards'], 
                    hidden_states=batch['hidden_states']
                )
            
            # [关键] 迭代结束后，将学到的当前策略参数同步更新到参考模型
            algo.sync_ref_model()
            
            batch_buffer = [] # 清空缓冲区
            update_happened = True

        # =================================================
        # 5. 日志写入与模型保存
        # =================================================
        log_str = f"Ep: {episode_idx}/{CONFIG['TOTAL_EPISODES']} | {mode_name} | Rew: {avg_ep_reward:.2f} | Len: {avg_ep_len:.1f} | SR: {success_rate:.1f}%"
        
        if update_happened:
            log_str += f" | Loss: {update_info['total_loss']:.3f} (Pol={update_info['policy_loss']:.3f} KL={update_info['kl_loss']:.3f}) | Adv_mean: {update_info['avg_advantage']:.2f}"
            
            writer.add_scalar("Train/Total_Loss", update_info['total_loss'], episode_idx)
            writer.add_scalar("Train/Policy_Loss", update_info['policy_loss'], episode_idx)
            writer.add_scalar("Train/KL_Loss", update_info['kl_loss'], episode_idx)
            writer.add_scalar("Train/Advantage_Mean", update_info['avg_advantage'], episode_idx)
        else:
            log_str += " | Collecting..."

        print(log_str)
        
        # 记录环境表现指标
        writer.add_scalar("Train/Episode_Reward", avg_ep_reward, episode_idx)
        writer.add_scalar("Train/Success_Rate", success_rate, episode_idx)
        writer.add_scalar("Curriculum/Distance", task_dist, episode_idx)

        # --- 模型检查点保存 ---
        if avg_ep_reward > best_reward:
            best_reward = avg_ep_reward
            save_path = os.path.join(model_dir, "best_model.pth")
            algo.save_model(save_path)

        if episode_idx > CONFIG["STAGE_THREE_EPISODE"]:
            if avg_ep_reward > best_stage3_reward:
                best_stage3_reward = avg_ep_reward
                save_path = os.path.join(model_dir, "best_stage4_model.pth")
                algo.save_model(save_path)
                print(f"  >>> New Best Stage 4 Model! Reward: {best_stage3_reward:.2f}")

        if episode_idx % 200 == 0:
            save_path = os.path.join(model_dir, f"ckpt_ep_{episode_idx}.pth")
            algo.save_model(save_path)

        # =================================================
        # 6. 可视化演示 (Visualization)
        # =================================================
        if episode_idx % 20 == 0:
            print(f"\n[Visual] Rendering {mode_name} (Dist: {task_dist:.1f}m)...")
            
            # 保证可视化使用跟这轮同等的任务坐标
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
                
                # 可视化评测阶段通常开启确定性策略 deterministic=True (取均值最稳)
                with torch.no_grad():
                    vis_action, _, _, vis_next_h_a, _ = algo.model.get_action(
                        vis_obs_tensor, vis_h_a, deterministic=True
                    )
                
                vis_obs, _, term, trunc, _ = eval_env.step(vis_action[0].cpu().numpy())
                
                vis_h_a = vis_next_h_a
                vis_done = term or trunc
                vis_steps += 1
            print(f"[Visual] Finished in {vis_steps} steps.\n")

    # 结束训练
    envs.close()
    eval_env.close()
    writer.close()
    print("Training finished!")

if __name__ == "__main__":
    train()