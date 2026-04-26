import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import gymnasium as gym
import time

from uncertain_env import UncertainComplexEnv
from CNN_GRU_policy import RobustGRPOPolicy
from ds_grpo import DSGRPO

# ==========================================
# 动作空间缩放包装器 (与训练时一致)
# ==========================================
class ActionScalingWrapper(gym.ActionWrapper):
    """将神经网络输出的归一化动作 [-1, 1] 映射到环境真实物理动作"""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def action(self, action):
        v_scaled, w_scaled = action[0], action[1]
        v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0  # [-1, 1] -> [-1, 2]
        w_real = w_scaled                                  # [-1, 1] -> [-1, 1]
        return np.array([v_real, w_real], dtype=np.float32)

# ==========================================
# 配置参数
# ==========================================
CONFIG = {
    "MODEL_PATH": "results/PPO Curricuium_20260426_115942/best_model.pth", # 请根据实际路径修改
    "HIDDEN_DIM": 256,            # 必须与训练时一致
    "MAX_STEPS": 1000,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "RENDER_DELAY": 0.02,

    "RANDOM_TASK": False,
    "FIXED_START": [10.0, 10.0],
    "FIXED_TARGET": [90.0, 96.0],
}

# ==========================================
# 辅助函数
# ==========================================
def calculate_smoothness(trajectory, actions):
    if len(trajectory) < 3:
        return 0.0, 0.0

    traj = np.array(trajectory)
    diffs = traj[1:] - traj[:-1]
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_diffs = angles[1:] - angles[:-1]
    angle_diffs = np.unwrap(angle_diffs)
    path_smoothness = np.mean(np.abs(angle_diffs))

    actions = np.array(actions)
    if len(actions) < 2:
        return path_smoothness, 0.0
    action_changes = np.linalg.norm(actions[1:] - actions[:-1], axis=1)
    control_smoothness = np.mean(action_changes)

    return path_smoothness, control_smoothness

# ==========================================
# 主测试逻辑
# ==========================================
def test():
    print(f"--- Loading Model from: {CONFIG['MODEL_PATH']} ---")

    # 1. 初始化环境 (开启渲染) + 动作缩放包装器
    env = UncertainComplexEnv(render_mode="human")
    env = ActionScalingWrapper(env)

    # 2. 初始化模型与 DSGRPO 算法实例
    model = RobustGRPOPolicy(obs_dim=24, action_dim=2, hidden_dim=CONFIG["HIDDEN_DIM"])
    algo = DSGRPO(policy_net=model, device=CONFIG["DEVICE"])

    # 3. 加载权重
    if os.path.exists(CONFIG["MODEL_PATH"]):
        algo.load(CONFIG["MODEL_PATH"])
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {CONFIG['MODEL_PATH']}")
        return

    # 4. 生成任务
    if CONFIG["RANDOM_TASK"]:
        while True:
            s_x = np.random.uniform(10.0, 90.0)
            s_y = np.random.uniform(10.0, 90.0)
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(50.0, 120.0)
            t_x = s_x + dist * np.cos(angle)
            t_y = s_y + dist * np.sin(angle)
            if not env._check_static_collision(np.array([s_x, s_y])) and \
               not env._check_static_collision(np.array([t_x, t_y])):
                break
        start_pos = [s_x, s_y]
        target_pos = [t_x, t_y]
        print(f"Task: Random Generated | Dist: {np.linalg.norm(np.array(start_pos)-np.array(target_pos)):.1f}m")
    else:
        start_pos, target_pos = CONFIG["FIXED_START"], CONFIG["FIXED_TARGET"]
        print("Task: Fixed Configuration")

    print(f"Start: {start_pos} -> Target: {target_pos}")

    # 5. 运行测试 Episode
    obs, info = env.reset(options={'start_pos': start_pos, 'target_pos': target_pos})
    hidden_state = torch.zeros(1, 1, CONFIG["HIDDEN_DIM"]).to(CONFIG["DEVICE"])

    trajectory = [env.unwrapped.agent_pos.copy()]
    action_history = []
    total_reward = 0
    steps = 0
    done = False

    print("\nSimulating...")

    while not done and steps < CONFIG["MAX_STEPS"]:
        env.render()
        time.sleep(CONFIG["RENDER_DELAY"])

        obs_tensor = torch.FloatTensor(obs).to(CONFIG["DEVICE"])

        with torch.no_grad():
            # DSGRPO.select_action 返回 4 个值: (action, log_prob, next_hidden, entropy)
            action_all, _, next_hidden, _ = algo.select_action(
                obs_tensor, hidden_state, deterministic=True
            )

        action = action_all[0]  # 取第0个环境的动作
        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append(env.unwrapped.agent_pos.copy())
        action_history.append(action)
        total_reward += reward
        steps += 1

        obs = next_obs
        hidden_state = next_hidden
        done = terminated or truncated

    # ==========================================
    # 6. 计算与打印性能指标
    # ==========================================
    print("\n" + "=" * 40)
    print("          PERFORMANCE REPORT          ")
    print("=" * 40)

    dist_to_goal = np.linalg.norm(env.unwrapped.agent_pos - np.array(target_pos))
    is_success = dist_to_goal < 2.0
    result_str = "SUCCESS [Target Reached]" if is_success else "FAILURE [Time out or Collision]"
    if steps >= CONFIG["MAX_STEPS"]:
        result_str = "FAILURE [Time Limit Exceeded]"

    print(f"Result          : {result_str}")

    dt = getattr(env, 'dt', 0.1)
    time_cost = steps * dt
    print(f"Time Taken      : {time_cost:.1f} s ({steps} steps)")

    traj_arr = np.array(trajectory)
    path_len = np.sum(np.linalg.norm(traj_arr[1:] - traj_arr[:-1], axis=1))
    print(f"Path Length     : {path_len:.2f} m")

    path_smooth, ctrl_smooth = calculate_smoothness(trajectory, action_history)
    print(f"Path Smoothness : {path_smooth:.4f} (rad/step, lower is better)")
    print(f"Ctrl Smoothness : {ctrl_smooth:.4f} (action diff, lower is stable)")

    straight_dist = np.linalg.norm(np.array(start_pos) - np.array(target_pos))
    efficiency = path_len / straight_dist if straight_dist > 0 else 0
    print(f"Path Efficiency : {efficiency:.2f}x (Actual / Straight)")

    print("=" * 40)

    env.close()

if __name__ == "__main__":
    test()
