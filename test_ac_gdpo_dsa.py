import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

import gymnasium as gym
import numpy as np
import torch

from uncertain_env import UncertainComplexEnv
from rmmf_model import RMMF_ActorCritic
from ac_gdpo_agent import AC_GDPO_Agent
from dsa_mask import DSABeamMasker
from test import calculate_smoothness


CONFIG = {
    "MODEL_PATH": "results/AC_GDPO_Curriculum_20260427_104230/best_stage4_model.pth",
    "HIDDEN_DIM": 128,
    "MAX_STEPS": 1000,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "RENDER_DELAY": 0.02,
    "FIXED_START": [10.0, 10.0],
    "FIXED_TARGET": [90.0, 96.0],
    "DSA_FLOOR_GAIN": 0.35,
    "DSA_TARGET_SIGMA": np.pi / 4.0,
    "DSA_MOTION_SIGMA": np.pi / 3.5,
    "DSA_SURPRISE_GAIN": 0.60,
    "DSA_TURN_SIDE_GAIN": 0.18,
    "DSA_EMA_DECAY": 0.70,
}


def build_dsa_masker():
    return DSABeamMasker(
        floor_gain=CONFIG["DSA_FLOOR_GAIN"],
        target_sigma=CONFIG["DSA_TARGET_SIGMA"],
        motion_sigma=CONFIG["DSA_MOTION_SIGMA"],
        surprise_gain=CONFIG["DSA_SURPRISE_GAIN"],
        turn_side_gain=CONFIG["DSA_TURN_SIDE_GAIN"],
        ema_decay=CONFIG["DSA_EMA_DECAY"],
    )


def test_ac_gdpo_dsa():
    print(f"--- Loading AC-GDPO DSA Model from: {CONFIG['MODEL_PATH']} ---")

    env = UncertainComplexEnv(render_mode="human")
    model = RMMF_ActorCritic(observation_dim=24, action_dim=2, hidden_dim=CONFIG["HIDDEN_DIM"])
    agent = AC_GDPO_Agent(model, device=CONFIG["DEVICE"])

    if os.path.exists(CONFIG["MODEL_PATH"]):
        agent.load(CONFIG["MODEL_PATH"])
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {CONFIG['MODEL_PATH']}")
        return

    start_pos = CONFIG["FIXED_START"]
    target_pos = CONFIG["FIXED_TARGET"]
    print(f"Task: Fixed Configuration")
    print(f"Start: {start_pos} -> Target: {target_pos}")

    obs, _ = env.reset(options={"start_pos": start_pos, "target_pos": target_pos})

    hidden_state = torch.zeros(1, 1, CONFIG["HIDDEN_DIM"]).to(CONFIG["DEVICE"])
    masker = build_dsa_masker()
    masker.reset(1, torch.device(CONFIG["DEVICE"]))

    trajectory = [env.agent_pos.copy()]
    action_history = []
    total_reward = 0.0
    steps = 0
    done = False

    print("\nSimulating with DSA mask...")

    while not done and steps < CONFIG["MAX_STEPS"]:
        env.render()
        time.sleep(CONFIG["RENDER_DELAY"])

        obs_tensor = torch.from_numpy(obs).float().to(CONFIG["DEVICE"]).unsqueeze(0)
        obs_tensor = masker.apply(obs_tensor)

        with torch.no_grad():
            s_action, _, _, next_hidden, _ = agent.get_action(
                obs_tensor, hidden_state, deterministic=True
            )
        masker.update_action_history(s_action)

        action = s_action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, _ = env.step(action)

        trajectory.append(env.agent_pos.copy())
        action_history.append(action)
        total_reward += reward
        steps += 1

        obs = next_obs
        hidden_state = next_hidden
        done = terminated or truncated

    dist_to_goal = np.linalg.norm(env.agent_pos - np.array(target_pos))
    is_success = dist_to_goal < 2.0
    result_str = "SUCCESS [Target Reached]" if is_success else "FAILURE [Time out or Collision]"
    if steps >= CONFIG["MAX_STEPS"]:
        result_str = "FAILURE [Time Limit Exceeded]"

    dt = getattr(env, "dt", 0.1)
    time_cost = steps * dt
    traj_arr = np.array(trajectory)
    path_len = np.sum(np.linalg.norm(traj_arr[1:] - traj_arr[:-1], axis=1))
    path_smooth, ctrl_smooth = calculate_smoothness(trajectory, action_history)
    straight_dist = np.linalg.norm(np.array(start_pos) - np.array(target_pos))
    efficiency = path_len / straight_dist if straight_dist > 0 else 0.0

    print("\n" + "=" * 40)
    print("      AC-GDPO DSA PERFORMANCE REPORT      ")
    print("=" * 40)
    print(f"Result          : {result_str}")
    print(f"Time Taken      : {time_cost:.1f} s ({steps} steps)")
    print(f"Total Reward    : {total_reward:.2f}")
    print(f"Path Length     : {path_len:.2f} m")
    print(f"Path Smoothness : {path_smooth:.4f} (rad/step, lower is better)")
    print(f"Ctrl Smoothness : {ctrl_smooth:.4f} (action diff, lower is stable)")
    print(f"Path Efficiency : {efficiency:.2f}x (Actual / Straight)")
    print("=" * 40)

    env.close()


if __name__ == "__main__":
    test_ac_gdpo_dsa()
