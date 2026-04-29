import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from uncertain_env import UncertainComplexEnv


Task = Dict[str, object]
EpisodeResult = Dict[str, object]


@dataclass
class ExperimentConfig:
    model_path: str
    hidden_dim: int
    device: str
    max_steps: int = 1000
    render: bool = False
    render_delay: float = 0.02
    fixed_repeats: int = 100
    output_dir: str = "eval_reports"
    run_note: str = ""


FIXED_TASKS: List[Task] = [
    {"task_name": "left_bottom_to_right_top", "start_pos": [10.0, 10.0], "target_pos": [90.0, 90.0]},
    {"task_name": "right_top_to_left_bottom", "start_pos": [90.0, 90.0], "target_pos": [10.0, 10.0]},
    {"task_name": "left_top_to_right_bottom", "start_pos": [10.0, 90.0], "target_pos": [90.0, 10.0]},
    {"task_name": "right_bottom_to_left_top", "start_pos": [90.0, 10.0], "target_pos": [10.0, 90.0]},
]


def calculate_smoothness(trajectory, actions):
    if len(trajectory) < 3:
        return 0.0, 0.0

    traj = np.array(trajectory)
    diffs = traj[1:] - traj[:-1]
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_diffs = np.unwrap(angles[1:] - angles[:-1])
    path_smoothness = float(np.mean(np.abs(angle_diffs))) if len(angle_diffs) > 0 else 0.0

    actions = np.array(actions)
    if len(actions) < 2:
        return path_smoothness, 0.0

    action_changes = np.linalg.norm(actions[1:] - actions[:-1], axis=1)
    control_smoothness = float(np.mean(action_changes)) if len(action_changes) > 0 else 0.0
    return path_smoothness, control_smoothness


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _safe_min(values: List[float]) -> float:
    return float(np.min(values)) if values else 0.0


def build_evaluation_plan(config: ExperimentConfig) -> List[Task]:
    plan: List[Task] = []

    for task in FIXED_TASKS:
        for repeat_idx in range(config.fixed_repeats):
            seed = 1000 + repeat_idx
            plan.append(
                {
                    "suite": "fixed_suite",
                    "task_name": task["task_name"],
                    "repeat_idx": repeat_idx,
                    "seed": seed,
                    "start_pos": task["start_pos"],
                    "target_pos": task["target_pos"],
                }
            )

    return plan


def create_env(render: bool) -> UncertainComplexEnv:
    return UncertainComplexEnv(render_mode="human" if render else None)


def run_single_episode(
    task: Task,
    config: ExperimentConfig,
    policy_runner: Callable[[np.ndarray, torch.Tensor], Tuple[np.ndarray, torch.Tensor]],
    episode_setup: Optional[Callable[[], None]] = None,
    episode_step_hook: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> EpisodeResult:
    env = create_env(config.render)
    try:
        obs, _ = env.reset(
            seed=int(task["seed"]),
            options={"start_pos": task["start_pos"], "target_pos": task["target_pos"]},
        )
        if episode_setup is not None:
            episode_setup()

        hidden_state = torch.zeros(1, 1, config.hidden_dim, device=config.device)
        trajectory = [env.agent_pos.copy()]
        action_history = []
        total_reward = 0.0
        uncertainty_penalties = []
        risk_margins = []
        risk_clearances = []
        terminated = False
        truncated = False
        info = {}

        for step_idx in range(config.max_steps):
            policy_obs = obs
            if episode_step_hook is not None:
                policy_obs = episode_step_hook(policy_obs)

            action, next_hidden = policy_runner(policy_obs, hidden_state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.agent_pos.copy())
            action_history.append(action)
            total_reward += float(reward)

            uncertainty_penalties.append(float(info.get("uncertainty_penalty", 0.0)))
            risk_margins.append(float(info.get("risk_margin", 0.0)))
            risk_clearances.append(float(info.get("risk_min_clearance", float("inf"))))

            obs = next_obs
            hidden_state = next_hidden

            if config.render:
                env.render()
                if config.render_delay > 0:
                    import time

                    time.sleep(config.render_delay)

            if terminated or truncated:
                break

        steps = len(action_history)
        dt = float(getattr(env, "dt", 0.1))
        final_distance = float(np.linalg.norm(env.agent_pos - np.array(task["target_pos"])))
        is_success = bool(info.get("is_success", final_distance < 2.0))
        is_collision = bool(info.get("is_collision", False))
        is_timeout = bool((not is_success) and (truncated or steps >= config.max_steps))

        traj_arr = np.array(trajectory)
        path_length = float(np.sum(np.linalg.norm(traj_arr[1:] - traj_arr[:-1], axis=1))) if len(traj_arr) > 1 else 0.0
        path_smoothness, ctrl_smoothness = calculate_smoothness(trajectory, action_history)

        straight_distance = float(
            np.linalg.norm(np.array(task["start_pos"], dtype=np.float32) - np.array(task["target_pos"], dtype=np.float32))
        )
        path_efficiency = path_length / max(straight_distance, 1e-6)

        finite_clearances = [v for v in risk_clearances if np.isfinite(v)]

        return {
            "suite": task["suite"],
            "task_name": task["task_name"],
            "repeat_idx": int(task["repeat_idx"]),
            "seed": int(task["seed"]),
            "start_x": float(task["start_pos"][0]),
            "start_y": float(task["start_pos"][1]),
            "target_x": float(task["target_pos"][0]),
            "target_y": float(task["target_pos"][1]),
            "success": int(is_success),
            "collision": int(is_collision),
            "timeout": int(is_timeout),
            "steps": steps,
            "sim_time": steps * dt,
            "total_reward": total_reward,
            "final_distance": final_distance,
            "path_length": path_length,
            "path_efficiency": float(path_efficiency),
            "path_smoothness": float(path_smoothness),
            "control_smoothness": float(ctrl_smoothness),
            "mean_uncertainty_penalty": _safe_mean(uncertainty_penalties),
            "mean_risk_margin": _safe_mean(risk_margins),
            "min_risk_clearance": _safe_min(finite_clearances) if finite_clearances else 0.0,
        }
    finally:
        env.close()


def summarize_results(rows: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
    suites = sorted(set(row["suite"] for row in rows))
    summary: Dict[str, Dict[str, float]] = {}

    for suite in suites:
        suite_rows = [row for row in rows if row["suite"] == suite]
        summary[suite] = {
            "episodes": len(suite_rows),
            "success_rate": 100.0 * _safe_mean([row["success"] for row in suite_rows]),
            "collision_rate": 100.0 * _safe_mean([row["collision"] for row in suite_rows]),
            "timeout_rate": 100.0 * _safe_mean([row["timeout"] for row in suite_rows]),
            "avg_reward": _safe_mean([row["total_reward"] for row in suite_rows]),
            "reward_std": _safe_std([row["total_reward"] for row in suite_rows]),
            "avg_steps": _safe_mean([row["steps"] for row in suite_rows]),
            "avg_final_distance": _safe_mean([row["final_distance"] for row in suite_rows]),
            "avg_path_length": _safe_mean([row["path_length"] for row in suite_rows]),
            "avg_efficiency": _safe_mean([row["path_efficiency"] for row in suite_rows]),
            "avg_path_smoothness": _safe_mean([row["path_smoothness"] for row in suite_rows]),
            "avg_control_smoothness": _safe_mean([row["control_smoothness"] for row in suite_rows]),
            "avg_uncertainty_penalty": _safe_mean([row["mean_uncertainty_penalty"] for row in suite_rows]),
            "avg_risk_margin": _safe_mean([row["mean_risk_margin"] for row in suite_rows]),
            "worst_case_clearance": _safe_min([row["min_risk_clearance"] for row in suite_rows]),
        }

    for task_name in sorted(set(row["task_name"] for row in rows)):
        task_rows = [row for row in rows if row["task_name"] == task_name]
        summary[task_name] = {
            "episodes": len(task_rows),
            "success_rate": 100.0 * _safe_mean([row["success"] for row in task_rows]),
            "collision_rate": 100.0 * _safe_mean([row["collision"] for row in task_rows]),
            "timeout_rate": 100.0 * _safe_mean([row["timeout"] for row in task_rows]),
            "avg_reward": _safe_mean([row["total_reward"] for row in task_rows]),
            "reward_std": _safe_std([row["total_reward"] for row in task_rows]),
            "avg_steps": _safe_mean([row["steps"] for row in task_rows]),
            "avg_final_distance": _safe_mean([row["final_distance"] for row in task_rows]),
            "avg_path_length": _safe_mean([row["path_length"] for row in task_rows]),
            "avg_efficiency": _safe_mean([row["path_efficiency"] for row in task_rows]),
            "avg_path_smoothness": _safe_mean([row["path_smoothness"] for row in task_rows]),
            "avg_control_smoothness": _safe_mean([row["control_smoothness"] for row in task_rows]),
            "avg_uncertainty_penalty": _safe_mean([row["mean_uncertainty_penalty"] for row in task_rows]),
            "avg_risk_margin": _safe_mean([row["mean_risk_margin"] for row in task_rows]),
            "worst_case_clearance": _safe_min([row["min_risk_clearance"] for row in task_rows]),
        }

    summary["overall"] = {
        "episodes": len(rows),
        "success_rate": 100.0 * _safe_mean([row["success"] for row in rows]),
        "collision_rate": 100.0 * _safe_mean([row["collision"] for row in rows]),
        "timeout_rate": 100.0 * _safe_mean([row["timeout"] for row in rows]),
        "avg_reward": _safe_mean([row["total_reward"] for row in rows]),
        "reward_std": _safe_std([row["total_reward"] for row in rows]),
        "avg_steps": _safe_mean([row["steps"] for row in rows]),
        "avg_final_distance": _safe_mean([row["final_distance"] for row in rows]),
        "avg_path_length": _safe_mean([row["path_length"] for row in rows]),
        "avg_efficiency": _safe_mean([row["path_efficiency"] for row in rows]),
        "avg_path_smoothness": _safe_mean([row["path_smoothness"] for row in rows]),
        "avg_control_smoothness": _safe_mean([row["control_smoothness"] for row in rows]),
        "avg_uncertainty_penalty": _safe_mean([row["mean_uncertainty_penalty"] for row in rows]),
        "avg_risk_margin": _safe_mean([row["mean_risk_margin"] for row in rows]),
        "worst_case_clearance": _safe_min([row["min_risk_clearance"] for row in rows]),
    }
    return summary


def format_summary_table(summary: Dict[str, Dict[str, float]], title: str) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(title.center(72))
    lines.append("=" * 72)
    for suite_name, metrics in summary.items():
        lines.append(
            f"[{suite_name}] episodes={int(metrics['episodes'])} | "
            f"success={metrics['success_rate']:.1f}% | "
            f"collision={metrics['collision_rate']:.1f}% | "
            f"timeout={metrics['timeout_rate']:.1f}%"
        )
        lines.append(
            f"reward={metrics['avg_reward']:.2f} +- {metrics['reward_std']:.2f} | "
            f"steps={metrics['avg_steps']:.1f} | final_dist={metrics['avg_final_distance']:.2f}"
        )
        lines.append(
            f"path_len={metrics['avg_path_length']:.2f} | eff={metrics['avg_efficiency']:.2f}x | "
            f"path_smooth={metrics['avg_path_smoothness']:.4f} | ctrl_smooth={metrics['avg_control_smoothness']:.4f}"
        )
        lines.append(
            f"unc_penalty={metrics['avg_uncertainty_penalty']:.4f} | "
            f"risk_margin={metrics['avg_risk_margin']:.4f} | "
            f"worst_clearance={metrics['worst_case_clearance']:.4f}"
        )
        lines.append("-" * 72)
    return "\n".join(lines)


def write_csv(path: str, rows: List[EpisodeResult]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: str, summary: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["suite"] + list(next(iter(summary.values())).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for suite_name, metrics in summary.items():
            row = {"suite": suite_name}
            row.update(metrics)
            writer.writerow(row)


def run_experiment(
    experiment_name: str,
    config: ExperimentConfig,
    policy_runner: Callable[[np.ndarray, torch.Tensor], Tuple[np.ndarray, torch.Tensor]],
    episode_setup_factory: Optional[Callable[[], Optional[Callable[[], None]]]] = None,
    episode_step_hook_factory: Optional[Callable[[], Optional[Callable[[np.ndarray], np.ndarray]]]] = None,
) -> Tuple[List[EpisodeResult], Dict[str, Dict[str, float]], str]:
    os.makedirs(config.output_dir, exist_ok=True)
    plan = build_evaluation_plan(config)
    results: List[EpisodeResult] = []

    for episode_idx, task in enumerate(plan, start=1):
        setup = episode_setup_factory() if episode_setup_factory is not None else None
        step_hook = episode_step_hook_factory() if episode_step_hook_factory is not None else None
        row = run_single_episode(
            task,
            config,
            policy_runner=policy_runner,
            episode_setup=setup,
            episode_step_hook=step_hook,
        )
        results.append(row)

        if episode_idx % 10 == 0 or episode_idx == len(plan):
            print(
                f"[{experiment_name}] progress {episode_idx}/{len(plan)} | "
                f"latest suite={row['suite']} task={row['task_name']} success={row['success']}"
            )

    summary = summarize_results(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    episodes_path = os.path.join(run_dir, "episodes.csv")
    summary_csv_path = os.path.join(run_dir, "summary.csv")
    summary_txt_path = os.path.join(run_dir, "summary.txt")
    note_path = os.path.join(run_dir, "run_note.txt")

    write_csv(episodes_path, results)
    write_summary_csv(summary_csv_path, summary)

    summary_text = format_summary_table(summary, f"{experiment_name} EVALUATION REPORT")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    note = str(config.run_note).strip()
    if note:
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(note + "\n")
    return results, summary, run_dir
