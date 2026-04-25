


import os
# 设置环境变量，允许 OpenMP 库重复加载 (防止 Windows 下 Numpy/Torch 冲突报错)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import gymnasium as gym
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# --- 导入自定义模块 ---
from uncertain_env import UncertainComplexEnv 
from CNN_GRU_policy import RobustGRPOPolicy
from PPO_algorithm import PPO 

# ==========================================
# 0. 辅助工具: 动作空间缩放包装器
# ==========================================
class ActionScalingWrapper(gym.ActionWrapper):
    """
    将神经网络输出的归一化动作 [-1, 1] 映射到环境真实物理动作
    v: [-1, 1] -> [-1, 2] (最大速度 2m/s)
    w: [-1, 1] -> [-1, 1] (最大角速度 1rad/s)
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def action(self, action):
        v_scaled, w_scaled = action[0], action[1]
        # 映射逻辑：(-1+1)/2 * 3 - 1 = 0 * 3 - 1 = -1; (1+1)/2 * 3 - 1 = 2
        v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
        w_real = w_scaled
        return np.array([v_real, w_real], dtype=np.float32)

# ==========================================
# 1. 环境工厂函数
# ==========================================
def make_env(rank, seed, max_steps):
    """创建单个环境的工厂函数"""
    def _thunk():
        env = UncertainComplexEnv() 
        env = ActionScalingWrapper(env)
        env.max_steps = max_steps 
        # 注意：这里不需要手动 reset，AsyncVectorEnv 会自动处理
        return env
    return _thunk

# ==========================================
# 2. PPO 训练主循环
# ==========================================
def train():
    # --- A. 初始化配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练参数配置 (对齐 AC-GDPO)
    CONFIG = {
        "NUM_ENVS": 16,             # 并行环境数量
        "TOTAL_EPISODES": 1500,     # 总训练轮数 (适配三阶段)
        "MAX_EPISODE_STEPS": 500,   # 单个 Episode 最大步数
        "UPDATE_FREQ": 2,           # 每 2 个 Episode 更新一次网络
        "HIDDEN_DIM": 256,          # PPO 网络隐层 (保持原 PPO 设置)
        
        # 课程阶段阈值
        "STAGE_ONE_EPISODE": 100,
        "STAGE_TWO_EPISODE": 400,
        "STAGE_THREE_EPISODE": 900
    }

    # 路径与日志
    run_name = f"PPO_Curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"results/{run_name}" # 统一放在 results 文件夹
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"--- Training Started: {run_name} using {device} ---")

    # --- B. 环境与模型初始化 ---
    
    # 1. 创建并行训练环境
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, 42+i, CONFIG["MAX_EPISODE_STEPS"]) for i in range(CONFIG["NUM_ENVS"])]
    )

    # 2. 创建可视化/验证环境 (单进程，用于生成合法坐标和渲染)
    eval_env = UncertainComplexEnv(render_mode="human")
    eval_env = ActionScalingWrapper(eval_env)
    eval_env.max_steps = CONFIG["MAX_EPISODE_STEPS"]

    # 3. 初始化网络 (Actor)
    actor_policy = RobustGRPOPolicy(obs_dim=24, action_dim=2, hidden_dim=CONFIG["HIDDEN_DIM"])
    
    # 4. 初始化 PPO 算法
    algo = PPO(
        actor_net=actor_policy,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01, # 稍微增加熵系数，鼓励早期探索
        device=device
    )

    # --- C. 状态变量初始化 ---
    global_step = 0
    
    # PPO 的隐状态 (Batch, Num_Layers, Hidden) -> 注意维度顺序取决于具体实现，通常是 (Layers, Batch, Hidden)
    # 这里假设 Policy 内部处理好了，我们维护初始状态即可
    hidden_actor = torch.zeros(1, CONFIG["NUM_ENVS"], CONFIG["HIDDEN_DIM"]).to(device)
    hidden_critic = torch.zeros(1, CONFIG["NUM_ENVS"], CONFIG["HIDDEN_DIM"]).to(device)
    
    # 数据缓冲区：用于积累 UPDATE_FREQ 个 Episode 的数据
    batch_buffer = [] 
    
    # 最佳模型记录
    best_reward = -float('inf')
    best_stage3_reward = -float('inf')

    # 先重置一次获取初始 obs
    obs, _ = envs.reset()

    # --- D. 主循环 ---
    for episode_idx in range(1, CONFIG["TOTAL_EPISODES"] + 1):
        
        # =================================================
        # 1. 课程学习：生成随机任务 (参考 AC-GDPO)
        # =================================================
        dist_range = (10.0, 30.0)
        mode_name = "Init"

        if episode_idx <= CONFIG["STAGE_ONE_EPISODE"]:
            # --- 阶段一: 短距离随机 (10m - 30m) ---
            # dist_range = (10.0, 30.0)
            # mode_name = "Stage 1: Random Short (10-30m)"
            cur_start_pos = [65.0, 40.0]
            cur_target_pos = [55.0, 40.0]
            mode_name = "Stage 1: Easy"
            
        elif episode_idx <= CONFIG["STAGE_TWO_EPISODE"]:
            # --- 阶段二: 中距离随机 (50m - 70m) ---
            # dist_range = (50.0, 70.0)
            # mode_name = "Stage 2: Random Medium (50-70m)"
            cur_start_pos = [65.0, 45.0] # 修正后的起点，避开障碍物
            cur_target_pos = [30.0, 40.0]
            # cur_start_pos = [50.0, 35.0]
            # cur_target_pos = [10.0, 90.0]
            mode_name = "Stage 2: Medium"
        elif episode_idx <= CONFIG["STAGE_THREE_EPISODE"]:
            cur_start_pos = [10.0, 10.0]
            cur_target_pos = [90.0, 96.0]
            mode_name = "Stage 3: Hard"
            
        else:
            # --- 阶段三: 长距离挑战 (>100m) ---
            dist_range = (100.0, 130.0) 
            mode_name = "Stage 4: Random Long (>100m)"

            # [新增] 随机生成合法的起点和终点
            # 利用 eval_env 的地图信息检测碰撞，防止生成的点在墙里
            while True:
                # 随机生成起点 (留出边界缓冲)
                s_x = np.random.uniform(5.0, eval_env.unwrapped.width - 5.0)
                s_y = np.random.uniform(5.0, eval_env.unwrapped.height - 5.0)
                
                # 随机生成方向和距离
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(dist_range[0], dist_range[1])
                
                # 计算潜在终点
                t_x = s_x + dist * np.cos(angle)
                t_y = s_y + dist * np.sin(angle)
                
                # 检查：
                # 1. 起点是否撞墙
                # 2. 终点是否撞墙
                # (注意：需要访问 unwrapped 环境的方法)
                start_valid = not eval_env.unwrapped._check_static_collision(np.array([s_x, s_y]))
                target_valid = not eval_env.unwrapped._check_static_collision(np.array([t_x, t_y]))
                
                if start_valid and target_valid:
                    break # 找到合法任务，退出循环

            cur_start_pos = [s_x, s_y]
            cur_target_pos = [t_x, t_y]

        task_dist = np.linalg.norm(np.array(cur_start_pos) - np.array(cur_target_pos))

        # 广播任务配置给所有并行环境
        obs, _ = envs.reset(seed=episode_idx+100, options={
            'start_pos': cur_start_pos,
            'target_pos': cur_target_pos
        })
        
        # 重置 RNN 记忆
        hidden_actor.fill_(0.0)
        hidden_critic.fill_(0.0)

        # =================================================
        # 2. 数据收集 (Collect Rollouts)
        # =================================================
        
        # 存储本轮数据的容器
        rollouts = [[] for _ in range(CONFIG["NUM_ENVS"])]
        active_envs = np.ones(CONFIG["NUM_ENVS"], dtype=bool)
        
        episode_rewards = np.zeros(CONFIG["NUM_ENVS"])
        episode_lengths = np.zeros(CONFIG["NUM_ENVS"])
        success_count = 0
        current_steps = 0
        
        while np.any(active_envs):
            current_steps += 1
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            # PPO 推理 (Action Selection)
            # 注意：PPO 需要在这里保存 value 和 log_prob 用于后续更新
            action, log_prob, value, next_h_a, next_h_c = algo.select_action(
                obs_tensor, hidden_actor, hidden_critic, deterministic=False
            )
            
            # 环境步进
            next_obs, rewards, terminations, truncations, infos = envs.step(action)
            dones = np.logical_or(terminations, truncations)
            
            for i in range(CONFIG["NUM_ENVS"]):
                if active_envs[i]:
                    # 存储 Transition
                    rollouts[i].append({
                        'obs': obs_tensor[i].clone(),
                        'action': torch.tensor(action[i]),
                        'reward': rewards[i],
                        'log_prob': log_prob[i],
                        'value': value[i],
                        'hidden_actor': hidden_actor[:, i:i+1, :].clone(),
                        'hidden_critic': hidden_critic[:, i:i+1, :].clone()
                    })
                    
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    
                    # [关键修改] 统计成功率 (逻辑对齐 AC-GDPO)
                    if terminations[i]:
                        is_success = False
                        # 尝试从 info 中获取真实状态
                        if "final_info" in infos:
                            final_info = infos["final_info"][i]
                            if final_info:
                                is_success = final_info.get("is_success", False)
                        
                        if is_success:
                            success_count += 1
                    
                    if dones[i]:
                        active_envs[i] = False
                        # 单个环境结束后，将其隐状态清零，防止干扰后续（虽然本轮不再使用）
                        next_h_a[:, i, :] = 0.0
                        next_h_c[:, i, :] = 0.0

            if current_steps >= CONFIG["MAX_EPISODE_STEPS"]:
                break
            
            obs = next_obs
            hidden_actor = next_h_a
            hidden_critic = next_h_c
            
        global_step += int(np.sum(episode_lengths))
        
        # =================================================
        # 3. 数据整理 & 存入缓冲区
        # =================================================
        # 将本轮数据整理为 Tensor 并加入 buffer
        for i in range(CONFIG["NUM_ENVS"]):
            if len(rollouts[i]) == 0: continue
            
            # 构建单条轨迹数据
            env_trace = {
                'obs': torch.stack([x['obs'] for x in rollouts[i]]),
                'action': torch.stack([x['action'] for x in rollouts[i]]),
                'reward': torch.tensor([x['reward'] for x in rollouts[i]], dtype=torch.float32).to(device),
                'log_prob': torch.tensor([x['log_prob'] for x in rollouts[i]], dtype=torch.float32).to(device),
                'value': torch.tensor([x['value'] for x in rollouts[i]], dtype=torch.float32).to(device),
                # RNN 只需初始状态
                'hidden_actor': rollouts[i][0]['hidden_actor'],
                'hidden_critic': rollouts[i][0]['hidden_critic']
            }
            batch_buffer.append(env_trace)

        # =================================================
        # 4. PPO 更新 (每 UPDATE_FREQ 轮触发一次)
        # =================================================
        update_happened = False
        update_info = {} 

        if episode_idx % CONFIG["UPDATE_FREQ"] == 0:
            # 执行 PPO 更新
            # update 函数内部会处理 GAE 计算和 Batch 切分
            update_info = algo.update(batch_buffer)
            batch_buffer = [] # 清空缓冲区
            update_happened = True

        # =================================================
        # 5. 日志与模型保存
        # =================================================
        avg_ep_reward = np.mean(episode_rewards)
        avg_ep_len = np.mean(episode_lengths)
        success_rate = (success_count / CONFIG["NUM_ENVS"]) * 100.0

        # 构建日志字符串 (风格对齐 AC-GDPO)
        log_str = f"Ep: {episode_idx}/{CONFIG['TOTAL_EPISODES']} | {mode_name} | Rew: {avg_ep_reward:.2f} | Len: {avg_ep_len:.1f} | SR: {success_rate:.1f}%"
        
        if update_happened:
            log_str += f" | Loss: {update_info['loss']:.3f} (A={update_info['actor_loss']:.3f} C={update_info['critic_loss']:.3f})"
            
            # 记录训练指标
            writer.add_scalar("Train/Total_Loss", update_info['loss'], episode_idx)
            writer.add_scalar("Train/Actor_Loss", update_info['actor_loss'], episode_idx)
            writer.add_scalar("Train/Critic_Loss", update_info['critic_loss'], episode_idx)
            writer.add_scalar("Train/KL", update_info['kl'], episode_idx)
        else:
            log_str += " | Collecting..."

        print(log_str)
        
        # 记录环境指标
        writer.add_scalar("Train/Average_Reward", avg_ep_reward, episode_idx)
        writer.add_scalar("Train/Success_Rate", success_rate, episode_idx)
        writer.add_scalar("Curriculum/Distance", task_dist, episode_idx)

        # --- 保存最佳模型 (Best Overall) ---
        if avg_ep_reward > best_reward:
            best_reward = avg_ep_reward
            save_path = os.path.join(log_dir, "best_model.pth")
            algo.save(save_path)
            # print(f"  >>> Best model saved! Reward: {best_reward:.2f}")

        # --- 保存 Stage 3 最佳模型 (Best Stage 3) ---
        if episode_idx > CONFIG["STAGE_THREE_EPISODE"]:
            if avg_ep_reward > best_stage3_reward:
                best_stage3_reward = avg_ep_reward
                save_path = os.path.join(log_dir, "best_stage4_model.pth")
                algo.save(save_path)
                print(f"  >>> New Best Stage 3 Model! Reward: {best_stage3_reward:.2f}")

        # --- 定期保存 Checkpoint ---
        if episode_idx % 200 == 0:
            save_path = os.path.join(log_dir, f"ckpt_ep_{episode_idx}.pth")
            algo.save(save_path)

        # =================================================
        # 6. 可视化演示 (Visualization)
        # =================================================
        if episode_idx % 20 == 0: # 每 20 轮可视化一次
            print(f"\n[Visual] Rendering {mode_name} (Dist: {task_dist:.1f}m)...")
            
            # 使用与当前 Episode 相同的任务配置进行可视化
            vis_obs, _ = eval_env.reset(seed=42, options={
                'start_pos': cur_start_pos,
                'target_pos': cur_target_pos
            })
            
            vis_h_a = torch.zeros(1, 1, CONFIG["HIDDEN_DIM"]).to(device)
            vis_h_c = torch.zeros(1, 1, CONFIG["HIDDEN_DIM"]).to(device)
            vis_done = False
            vis_steps = 0
            
            while not vis_done and vis_steps < CONFIG["MAX_EPISODE_STEPS"]:
                eval_env.render()
                vis_obs_tensor = torch.FloatTensor(vis_obs).unsqueeze(0).to(device)
                
                # 使用确定性策略进行演示
                vis_action, _, _, vis_next_h_a, vis_next_h_c = algo.select_action(
                    vis_obs_tensor, vis_h_a, vis_h_c, deterministic=True
                )
                
                vis_obs, _, term, trunc, _ = eval_env.step(vis_action[0])
                
                vis_h_a = vis_next_h_a
                vis_h_c = vis_next_h_c
                vis_done = term or trunc
                vis_steps += 1
            print(f"[Visual] Finished in {vis_steps} steps.\n")

    # 结束
    envs.close()
    eval_env.close()
    writer.close()
    print("Training finished!")

if __name__ == "__main__":
    train()