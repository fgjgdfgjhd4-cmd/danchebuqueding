import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from ac_gdpo_agent import AC_GDPO_Agent
from dsa_config import (
    build_dsa_masker_from_config,
    load_dsa_config_for_model,
    make_dsa_config,
)
from experiment_eval import ExperimentConfig, run_experiment
from rmmf_model import build_rmmf_model_from_state_dict


CONFIG = {
    "MODEL_PATH": "results/AC_GDPO_Curriculum_20260427_230049/best_stage4_model.pth",
    "HIDDEN_DIM": 128,
    "MAX_STEPS": 1000,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "RENDER": False,
    "RENDER_DELAY": 0.02,
    "FIXED_REPEATS": 100,
    "OUTPUT_DIR": "eval_reports",
    "DSA_CONFIG": make_dsa_config(
        floor_gain=0.25,
    target_sigma=np.pi / 4.0,
    motion_sigma=np.pi / 3.5,
    surprise_gain=0.60,
    turn_side_gain=0.18,
    ema_decay=0.70,
    ),
}


def build_agent():
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        raise FileNotFoundError(f"Model file not found: {CONFIG['MODEL_PATH']}")

    state_dict = torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"])
    model, model_variant = build_rmmf_model_from_state_dict(
        state_dict,
        observation_dim=24,
        action_dim=2,
        hidden_dim=CONFIG["HIDDEN_DIM"],
    )
    model = model.to(CONFIG["DEVICE"])
    model.load_state_dict(state_dict)
    agent = AC_GDPO_Agent(model, device=CONFIG["DEVICE"])
    return agent, model_variant


def build_dsa_masker():
    return build_dsa_masker_from_config(CONFIG["DSA_CONFIG"])

def infer_experiment_name():
    model_dir_name = os.path.basename(os.path.dirname(os.path.abspath(CONFIG["MODEL_PATH"])))
    safe_name = model_dir_name.lower().replace(" ", "_")
    return f"eval_{safe_name}_dsa" if safe_name else "eval_model_dsa"

def main():
    print(f"--- Loading AC-GDPO DSA model from: {CONFIG['MODEL_PATH']} ---")
    agent, model_variant = build_agent()
    print(f"Model loaded successfully. Detected architecture: {model_variant}")
    resolved_dsa_config, loaded_config_path = load_dsa_config_for_model(
        CONFIG["MODEL_PATH"],
        fallback_config=CONFIG["DSA_CONFIG"],
    )
    CONFIG["DSA_CONFIG"] = resolved_dsa_config
    if loaded_config_path is not None:
        print(f"Loaded DSA config from: {loaded_config_path}")
    else:
        print("DSA config file not found beside model, using local fallback config.")

    experiment_config = ExperimentConfig(
        model_path=CONFIG["MODEL_PATH"],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        device=CONFIG["DEVICE"],
        max_steps=CONFIG["MAX_STEPS"],
        render=CONFIG["RENDER"],
        render_delay=CONFIG["RENDER_DELAY"],
        fixed_repeats=CONFIG["FIXED_REPEATS"],
        output_dir=CONFIG["OUTPUT_DIR"],
    )

    active_masker = {"obj": None}

    def policy_runner(obs: np.ndarray, hidden_state: torch.Tensor):
        masker = active_masker["obj"]
        if masker is None:
            raise RuntimeError("DSA masker has not been initialized for this episode.")

        obs_tensor = torch.from_numpy(obs).float().to(CONFIG["DEVICE"]).unsqueeze(0)
        obs_tensor = masker.apply(obs_tensor)
        with torch.no_grad():
            scaled_action, _, _, next_hidden, _ = agent.get_action(
                obs_tensor, hidden_state, deterministic=True
            )
        masker.update_action_history(scaled_action)
        action = scaled_action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return action, next_hidden

    def episode_setup_factory():
        def setup():
            masker = build_dsa_masker()
            masker.reset(1, torch.device(CONFIG["DEVICE"]))
            active_masker["obj"] = masker

        return setup

    experiment_name = infer_experiment_name()
    _, summary, run_dir = run_experiment(
        experiment_name=experiment_name,
        config=experiment_config,
        policy_runner=policy_runner,
        episode_setup_factory=episode_setup_factory,
    )

    print()
    for suite_name, metrics in summary.items():
        print(
            f"[{suite_name}] episodes={int(metrics['episodes'])} "
            f"success={metrics['success_rate']:.1f}% "
            f"collision={metrics['collision_rate']:.1f}% "
            f"timeout={metrics['timeout_rate']:.1f}% "
            f"avg_reward={metrics['avg_reward']:.2f}"
        )

    print(f"\nDetailed results saved to: {os.path.join(run_dir, 'episodes.csv')}")
    print(f"Summary saved to: {os.path.join(run_dir, 'summary.csv')}")
    print(f"Readable report saved to: {os.path.join(run_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()
