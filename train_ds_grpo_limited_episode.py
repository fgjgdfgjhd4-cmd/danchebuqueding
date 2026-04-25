


import os
# 设置环境变量，允许 OpenMP 库重复加载 (防止 Windows 下 Numpy/Torch 冲突报错)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import gymnasium as gym
import os
import time
from torch.utils.tensorboard import SummaryWriter

# --- 导入自定义模块 ---
# 请确保文件名与您实际保存的一致
from uncertain_env import UncertainComplexEnv 
from CNN_GRU_policy import RobustGRPOPolicy
from ds_grpo import DSGRPO

# ==========================================
# 0. 辅助工具: 动作空间缩放包装器
# ==========================================
class ActionScalingWrapper(gym.ActionWrapper):
    """
    将神经网络输出的归一化动作 [-1, 1] 映射到环境真实物理动作
    v: [-1, 1] -> [-1, 2] (根据环境设定)
    w: [-1, 1] -> [-1, 1]
    """
    def __init__(self, env):
        super().__init__(env)
        # 定义 Gym 标准化的动作空间 [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def action(self, action):
        v_scaled, w_scaled = action[0], action[1]
        # 线性映射: (-1 + 1)/2 * 3 - 1 = -1; (1 + 1)/2 * 3 - 1 = 2
        v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
        w_real = w_scaled
        return np.array([v_real, w_real], dtype=np.float32)

# ==========================================
# 1. 环境工厂函数 (Gym VectorEnv 专用)
# ==========================================
def make_env(env_id, seed, max_steps):
    def _thunk():
        # 训练环境不需要渲染，速度优先
        env = UncertainComplexEnv() 
        env = ActionScalingWrapper(env)
        # 强制设置环境内部的最大步数，防止死循环
        env.max_steps = max_steps 
        env.reset(seed=seed)
        return env
    return _thunk

# ==========================================
# 2. 训练主循环 (基于完整 Episode 的变长序列训练)
# ==========================================
def train():
    # --- A. 超参数设置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 训练配置
    NUM_ENVS = 16           # 并行环境数量 (Group Size)
    TOTAL_EPISODES = 500   # 总训练轮数 (Episode Based)
    MAX_EPISODE_STEPS = 500 # 单个 Episode 允许的最大步数 (长视距)
    
    VISUALIZE_FREQ = 20     # 可视化频率 (每20轮看一次)
    
    # DAPO 参数
    G_MIN = 4
    G_MAX = 16 # 最大不超过 NUM_ENVS
    
    # 算法参数
    LR = 3e-4
    HIDDEN_DIM = 256
    
    # 路径设置
    exp_name = f"DS_GRPO_Final_{int(time.time())}"
    log_dir = f"runs/{exp_name}"
    model_dir = f"models/{exp_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)

    # --- B. 初始化 ---
    
    # 1. 创建并行训练环境
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, i+100, MAX_EPISODE_STEPS) for i in range(NUM_ENVS)]
    )

    # 2. 创建可视化测试环境 (主进程运行)
    visual_env = UncertainComplexEnv(render_mode="human")
    visual_env = ActionScalingWrapper(visual_env)
    visual_env.max_steps = MAX_EPISODE_STEPS 

    # 3. 初始化网络与算法
    policy = RobustGRPOPolicy(obs_dim=24, action_dim=2, hidden_dim=HIDDEN_DIM).to(device)
    
    # 实例化算法类
    algo = DSGRPO(
        policy_net=policy,
        optimizer_lr=LR,
        beta_kl=0.01,    
        clip_ratio=0.2,
        entropy_coef=0.02, # 确保这里的系数设置正确，防止过早收敛
        device=device
    )

    # --- C. 训练状态变量 ---
    global_step = 0
    start_time = time.time()
    
    # 初始观测
    obs, _ = envs.reset() 
    
    # GRU 隐状态 (Batch=NUM_ENVS)
    hidden_states = torch.zeros(1, NUM_ENVS, HIDDEN_DIM).to(device)
    
    # 记录上一轮的平均熵
    last_avg_entropy = 1.0 

    print("Starting Long-Horizon Episode Training...")
    print(f"Config: {TOTAL_EPISODES} Episodes, Max Steps {MAX_EPISODE_STEPS}")

    # --- D. 主循环 (按 Episode 迭代) ---
    for episode_idx in range(1, TOTAL_EPISODES + 1):
        
        # [DAPO] 动态决定是否需要更多探索 
        # (此处仅计算建议值用于记录，实际采样仍使用 NUM_ENVS 个并行环境)
        target_group_size = algo.determine_group_size(last_avg_entropy, G_MIN, G_MAX)
        
        # =================================================
        # 1. 数据收集 (Collect Rollouts)
        #    逻辑：让所有环境跑完当前 Episode
        # =================================================
        
        # 临时存储每个环境的轨迹
        rollouts = [[] for _ in range(NUM_ENVS)]
        
        # 标记哪些环境还在运行 (True=Running, False=Done)
        active_envs = np.ones(NUM_ENVS, dtype=bool)
        
        episode_rewards = np.zeros(NUM_ENVS)
        episode_lengths = np.zeros(NUM_ENVS)
        
        # 本轮累计的熵 (用于计算平均值)
        ep_entropy_sum = 0.0
        ep_step_count = 0
        
        # 重置 GRU 记忆 (每轮 Episode 开始前清空)
        hidden_states.fill_(0.0) 
        
        current_steps = 0
        
        # 只要还有一个环境没跑完，就继续
        while np.any(active_envs):
            current_steps += 1
            
            # (1) 准备数据
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            # (2) 策略推理
            # 注意: select_action 返回的 entropy 已经是 float 标量 (16个环境的均值)
            with torch.no_grad():
                actions, log_probs, next_hidden, entropy_val = algo.select_action(
                    obs_tensor, hidden_states, deterministic=False
                )
            
            # 累加熵值用于统计
            ep_entropy_sum += entropy_val
            ep_step_count += 1
            
            # (3) 环境步进
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            dones = np.logical_or(terminations, truncations)
            
            # (4) 存储数据 (只存储 Active 环境的数据)
            for i in range(NUM_ENVS):
                if active_envs[i]:
                    rollouts[i].append({
                        'obs': obs_tensor[i].cpu(),         # 移回 CPU 节省显存
                        'action': torch.tensor(actions[i]), # 已经是 CPU tensor
                        'reward': rewards[i],               # Numpy float
                        'log_prob': log_probs[i],           # Numpy float
                        'hidden': hidden_states[:, i:i+1, :].clone().cpu() # 存当前步隐状态
                    })
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    
                    # 如果这一步 Done 了，停止记录该环境
                    if dones[i]:
                        active_envs[i] = False
                        # 清空该环境对应的隐状态，防止影响下一轮
                        next_hidden[:, i, :] = 0.0

            # (5) 强制退出条件
            if current_steps >= MAX_EPISODE_STEPS:
                break
            
            obs = next_obs
            hidden_states = next_hidden

        # 更新全局统计
        global_step += int(np.sum(episode_lengths))
        
        # 计算本轮 Episode 的平均熵
        last_avg_entropy = ep_entropy_sum / max(1, ep_step_count)
        
        # =================================================
        # 2. 数据整理 (Process Batch)
        # =================================================
        batch_data = []
        for i in range(NUM_ENVS):
            # 将列表堆叠成 Tensor
            # 注意：reward 和 log_prob 还是 numpy list，需要转 Tensor
            env_trace = {
                'obs': torch.stack([x['obs'] for x in rollouts[i]]),       # [T_i, 24]
                'action': torch.stack([x['action'] for x in rollouts[i]]), # [T_i, 2]
                'reward': torch.tensor([x['reward'] for x in rollouts[i]], dtype=torch.float32), # [T_i]
                'log_prob': torch.tensor([x['log_prob'] for x in rollouts[i]], dtype=torch.float32), # [T_i]
                'hidden': rollouts[i][0]['hidden'] # [1, 1, H]
            }
            batch_data.append(env_trace)

        # =================================================
        # 3. 算法更新 (Update)
        # =================================================
        # 调用修改后的 update 函数 (支持变长列表输入)
        update_info = algo.update(batch_data)
        
        # =================================================
        # 4. 日志记录
        # =================================================
        avg_ep_reward = np.mean(episode_rewards)
        avg_ep_len = np.mean(episode_lengths)
        
        if episode_idx % 5 == 0:
            print(f"Episode: {episode_idx}/{TOTAL_EPISODES} | "
                  f"Avg Len: {avg_ep_len:.1f} | "
                  f"Loss: {update_info.get('loss', 0):.4f} | "
                  f"Avg Reward: {avg_ep_reward:.2f} | "
                  f"Entropy: {last_avg_entropy:.4f} | "
                  f"Safety Trig: {update_info.get('safety_trigger_rate', 0)*100:.1f}%")
            
            writer.add_scalar("Train/Episode_Reward", avg_ep_reward, episode_idx)
            writer.add_scalar("Train/Episode_Length", avg_ep_len, episode_idx)
            writer.add_scalar("Train/Loss", update_info.get('loss', 0), episode_idx)
            writer.add_scalar("Train/Entropy", last_avg_entropy, episode_idx)
            writer.add_scalar("Safety/Trigger_Rate", update_info.get('safety_trigger_rate', 0), episode_idx)
            writer.add_scalar("DAPO/Target_Group", target_group_size, episode_idx)

        # =================================================
        # 5. 可视化演示 (Render)
        # =================================================
        if episode_idx % VISUALIZE_FREQ == 0:
            print(f"\n[Visualization] Rendering test episode...")
            vis_obs, _ = visual_env.reset(seed=42)
            # [关键] 每次重置必须清空记忆！
            vis_hidden = torch.zeros(1, 1, HIDDEN_DIM).to(device) 
            
            vis_done = False
            vis_steps = 0
            
            while not vis_done and vis_steps < MAX_EPISODE_STEPS:
                visual_env.render()
                # 增加 Batch 维度
                vis_obs_tensor = torch.FloatTensor(vis_obs).unsqueeze(0).to(device)
                
                # 使用确定性策略
                vis_action_all, _, vis_next_hidden, _ = algo.select_action(
                    vis_obs_tensor, vis_hidden, deterministic=True
                )
                vis_action = vis_action_all[0]
                vis_obs, _, term, trunc, _ = visual_env.step(vis_action)
                
                vis_hidden = vis_next_hidden
                vis_done = term or trunc
                vis_steps += 1
            print(f"[Visualization] Finished in {vis_steps} steps.\n")

        # =================================================
        # 6. 保存模型
        # =================================================
        if episode_idx % 200 == 0:
            save_path = os.path.join(model_dir, f"model_ep_{episode_idx}.pth")
            algo.save(save_path)
            print(f"Model saved to {save_path}")

    # 结束
    envs.close()
    visual_env.close()
    writer.close()
    print("Training finished!")

if __name__ == "__main__":
    train()