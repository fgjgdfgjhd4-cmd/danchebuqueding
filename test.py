import os
# 设置环境变量，允许 OpenMP 库重复加载 (解决 Windows 下 Numpy/Torch 冲突报错)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import gymnasium as gym
import time
import math

# --- 导入自定义模块 ---
# 确保 uncertain_env.py, rmmf_model.py, ac_gdpo_agent.py 在同一目录下
from uncertain_env import UncertainComplexEnv
from rmmf_model import RMMF_ActorCritic
from ac_gdpo_agent import AC_GDPO_Agent

# ==========================================
# 配置参数
# ==========================================
CONFIG = {
    "MODEL_PATH": "models/DAPO_Curriculum_20260426_131633/best_model.pth", # 请根据实际路径修改
    "HIDDEN_DIM": 128,           # 必须与训练时一致
    "MAX_STEPS": 1000,           # 测试时可以给多一点步数，防止因为走得慢被截断
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "RENDER_DELAY": 0.02,        # 渲染延迟，方便肉眼观察 (秒)
    
    # 任务设置
    "RANDOM_TASK": False,         # True: 随机生成; False: 使用下方固定坐标
    "FIXED_START": [10.0, 10.0],
    "FIXED_TARGET": [90.0, 96.0] # 之前那个困难的陷阱关卡
}

# ==========================================
# 辅助函数
# ==========================================
def generate_random_task(ref_env, min_dist=50.0, max_dist=120.0, margin=5.0):
    """随机生成合法的起点和终点"""
    map_size = 100.0
    grid = ref_env.grid
    
    for _ in range(2000):
        # 随机起点
        s_x = np.random.uniform(margin, map_size - margin)
        s_y = np.random.uniform(margin, map_size - margin)
        if grid[int(s_x), int(s_y)] == 1: continue

        # 随机终点
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(min_dist, max_dist)
        t_x = s_x + dist * np.cos(angle)
        t_y = s_y + dist * np.sin(angle)
        
        # 边界与碰撞检查
        if not (margin <= t_x <= map_size - margin and margin <= t_y <= map_size - margin): continue
        if grid[int(t_x), int(t_y)] == 1: continue
        
        return [s_x, s_y], [t_x, t_y]
    
    return [10.0, 10.0], [90.0, 90.0] # 兜底

def calculate_smoothness(trajectory, actions):
    """
    计算平滑度指标
    1. 路径平滑度 (Path Smoothness): 轨迹方向变化的累积 (越低越直)
    2. 控制平滑度 (Control Smoothness): 动作指令的变化率 (越低越稳)
    """
    if len(trajectory) < 3: return 0.0, 0.0
    
    # --- 1. 路径平滑度 (基于轨迹几何) ---
    # 计算每一步的位移向量
    traj = np.array(trajectory)
    diffs = traj[1:] - traj[:-1]
    # 计算每一步的朝向角
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    # 计算角度变化量 (并处理 -pi 到 pi 的跳变)
    angle_diffs = angles[1:] - angles[:-1]
    angle_diffs = np.unwrap(angle_diffs) 
    # 指标：平均绝对角度变化 (rad/step)
    path_smoothness = np.mean(np.abs(angle_diffs))

    # --- 2. 控制平滑度 (基于动作抖动) ---
    actions = np.array(actions)
    if len(actions) < 2: return path_smoothness, 0.0
    # 计算相邻动作的 L2 距离
    action_changes = np.linalg.norm(actions[1:] - actions[:-1], axis=1)
    control_smoothness = np.mean(action_changes)
    
    return path_smoothness, control_smoothness

# ==========================================
# 主测试逻辑
# ==========================================
def test():
    print(f"--- Loading Model from: {CONFIG['MODEL_PATH']} ---")
    
    # 1. 初始化环境 (开启渲染)
    env = UncertainComplexEnv(render_mode="human")
    
    # 2. 初始化模型与智能体
    model = RMMF_ActorCritic(observation_dim=24, action_dim=2, hidden_dim=CONFIG["HIDDEN_DIM"])
    agent = AC_GDPO_Agent(model, device=CONFIG["DEVICE"])
    
    # 3. 加载权重
    if os.path.exists(CONFIG["MODEL_PATH"]):
        agent.load(CONFIG["MODEL_PATH"])
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {CONFIG['MODEL_PATH']}")
        return

    # 4. 生成任务
    if CONFIG["RANDOM_TASK"]:
        # start_pos, target_pos = generate_random_task(env, min_dist=50.0, max_dist=120.0)
        while True:
            s_x = np.random.uniform(10.0, 90.0)
            s_y = np.random.uniform(10.0, 90.0)
            
            # 2. 随机生成方向和距离
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(50.0, 120.0)
            
            # 3. 计算潜在终点
            t_x = s_x + dist * np.cos(angle)
            t_y = s_y + dist * np.sin(angle)
            
            if not env._check_static_collision(np.array([s_x, s_y])) and not env._check_static_collision(np.array([t_x, t_y])):
                break
        
        start_pos = [s_x, s_y]
        target_pos = [t_x, t_y]

        print(f"Task: Random Generated | Dist: {np.linalg.norm(np.array(start_pos)-np.array(target_pos)):.1f}m")
    else:
        start_pos, target_pos = CONFIG["FIXED_START"], CONFIG["FIXED_TARGET"]
        print(f"Task: Fixed Configuration")

    print(f"Start: {start_pos} -> Target: {target_pos}")

    # 5. 运行测试 Episode
    obs, info = env.reset(options={'start_pos': start_pos, 'target_pos': target_pos})
    
    # 初始化 RNN 隐藏状态
    hidden_state = torch.zeros(1, 1, CONFIG["HIDDEN_DIM"]).to(CONFIG["DEVICE"])
    
    # 数据记录容器
    trajectory = [env.agent_pos.copy()] # 记录轨迹点
    action_history = []                 # 记录动作
    total_reward = 0
    steps = 0
    done = False
    
    print("\nSimulating...")
    
    while not done and steps < CONFIG["MAX_STEPS"]:
        # 渲染
        env.render()
        time.sleep(CONFIG["RENDER_DELAY"]) # 稍微减慢速度以便观察
        
        # 准备输入
        obs_tensor = torch.from_numpy(obs).float().to(CONFIG["DEVICE"]).unsqueeze(0)
        
        # 模型推理 (关键：使用 deterministic=True)
        # 测试时不需要随机探索，我们需要看模型的"真实实力"
        with torch.no_grad():
            s_action, _, _, next_hidden, _ = agent.get_action(
                obs_tensor, hidden_state, deterministic=True
            )
        
        # 执行动作
        action = s_action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录数据
        trajectory.append(env.agent_pos.copy())
        action_history.append(action)
        total_reward += reward
        steps += 1
        
        # 更新状态
        obs = next_obs
        hidden_state = next_hidden
        done = terminated or truncated

    # ==========================================
    # 6. 计算与打印性能指标
    # ==========================================
    print("\n" + "="*40)
    print("          PERFORMANCE REPORT          ")
    print("="*40)
    
    # A. 到达判定
    # 假设环境在成功时 info 中有 'is_success'，或者根据 reward/terminated 判断
    # 如果您的 env 没有 is_success，可以用距离阈值判断
    dist_to_goal = np.linalg.norm(env.agent_pos - np.array(target_pos))
    is_success = dist_to_goal < 2.0 # 假设 2米内算到达
    result_str = "SUCCESS [Target Reached]" if is_success else "FAILURE [Time out or Collision]"
    if steps >= CONFIG["MAX_STEPS"]: result_str = "FAILURE [Time Limit Exceeded]"
    
    print(f"Result          : {result_str}")
    
    # B. 时间消耗
    # 假设 env.dt = 0.1 (需要在 env 中确认，如果不确定通常是 0.1 或 0.05)
    dt = getattr(env, 'dt', 0.1) 
    time_cost = steps * dt
    print(f"Time Taken      : {time_cost:.1f} s ({steps} steps)")
    
    # C. 路径长度
    # 计算相邻轨迹点的欧氏距离之和
    traj_arr = np.array(trajectory)
    path_len = np.sum(np.linalg.norm(traj_arr[1:] - traj_arr[:-1], axis=1))
    print(f"Path Length     : {path_len:.2f} m")
    
    # D. 平滑度指标
    path_smooth, ctrl_smooth = calculate_smoothness(trajectory, action_history)
    print(f"Path Smoothness : {path_smooth:.4f} (rad/step, lower is better)")
    print(f"Ctrl Smoothness : {ctrl_smooth:.4f} (action diff, lower is stable)")
    
    # E. 综合评分 (仅供参考)
    # 比如：路径长度 / 直线距离 (越接近 1 越好)
    straight_dist = np.linalg.norm(np.array(start_pos) - np.array(target_pos))
    efficiency = path_len / straight_dist if straight_dist > 0 else 0
    print(f"Path Efficiency : {efficiency:.2f}x (Actual / Straight)")

    print("="*40)
    
    env.close()

if __name__ == "__main__":
    test()