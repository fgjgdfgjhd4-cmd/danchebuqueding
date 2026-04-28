import math
import json
import os

from dsa_mask import DSABeamMasker


DEFAULT_DSA_CONFIG = {
    "floor_gain": 0.35,
    "target_sigma": math.pi / 4.0,
    "motion_sigma": math.pi / 3.5,
    "surprise_gain": 0.60,
    "turn_side_gain": 0.18,
    "ema_decay": 0.70,
}


def make_dsa_config(**overrides):
    config = dict(DEFAULT_DSA_CONFIG)
    config.update(overrides)
    return config


def build_dsa_masker_from_config(config):
    return DSABeamMasker(
        floor_gain=config["floor_gain"],
        target_sigma=config["target_sigma"],
        motion_sigma=config["motion_sigma"],
        surprise_gain=config["surprise_gain"],
        turn_side_gain=config["turn_side_gain"],
        ema_decay=config["ema_decay"],
    )


def save_dsa_config(config, output_dir, filename="dsa_config.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path


def load_dsa_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return make_dsa_config(**data)


def load_dsa_config_for_model(model_path, fallback_config=None, filename="dsa_config.json"):
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(model_dir, filename)
    if os.path.exists(config_path):
        return load_dsa_config(config_path), config_path

    if fallback_config is None:
        fallback_config = make_dsa_config()
    return dict(fallback_config), None
