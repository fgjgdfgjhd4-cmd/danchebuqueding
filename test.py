import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from ac_gdpo_agent import AC_GDPO_Agent
from experiment_eval import ExperimentConfig, run_experiment
from rmmf_model import build_rmmf_model_from_state_dict


CONFIG = {
    "MODEL_PATH": "models/DAPO_Curriculum_20260426_131633/best_stage4_model.pth",
    "HIDDEN_DIM": 128,
    "MAX_STEPS": 1000,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "RENDER": False,
    "RENDER_DELAY": 0.02,
    "FIXED_REPEATS": 100,
    "OUTPUT_DIR": "eval_reports",
    "RUN_NOTE": "",
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

def infer_experiment_name():
    model_dir_name = os.path.basename(os.path.dirname(os.path.abspath(CONFIG["MODEL_PATH"])))
    safe_name = model_dir_name.lower().replace(" ", "_")
    return f"eval_{safe_name}" if safe_name else "eval_model"


def main():
    print(f"--- Loading AC-GDPO model from: {CONFIG['MODEL_PATH']} ---")
    agent, model_variant = build_agent()
    print(f"Model loaded successfully. Detected architecture: {model_variant}")

    experiment_config = ExperimentConfig(
        model_path=CONFIG["MODEL_PATH"],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        device=CONFIG["DEVICE"],
        max_steps=CONFIG["MAX_STEPS"],
        render=CONFIG["RENDER"],
        render_delay=CONFIG["RENDER_DELAY"],
        fixed_repeats=CONFIG["FIXED_REPEATS"],
        output_dir=CONFIG["OUTPUT_DIR"],
        run_note=CONFIG["RUN_NOTE"],
    )

    def policy_runner(obs: np.ndarray, hidden_state: torch.Tensor):
        obs_tensor = torch.from_numpy(obs).float().to(CONFIG["DEVICE"]).unsqueeze(0)
        with torch.no_grad():
            scaled_action, _, _, next_hidden, _ = agent.get_action(
                obs_tensor, hidden_state, deterministic=True
            )
        action = scaled_action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return action, next_hidden

    experiment_name = infer_experiment_name()
    _, summary, run_dir = run_experiment(
        experiment_name=experiment_name,
        config=experiment_config,
        policy_runner=policy_runner,
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
