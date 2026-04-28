import math
from typing import Optional

import numpy as np
import torch


DEFAULT_FLOOR_GAIN = 0.35
DEFAULT_TARGET_SIGMA = math.pi / 4.0
DEFAULT_MOTION_SIGMA = math.pi / 3.5
DEFAULT_SURPRISE_GAIN = 0.60
DEFAULT_TURN_SIDE_GAIN = 0.18
DEFAULT_EMA_DECAY = 0.70


class DSABeamMasker:
    """
    Direction-sensitive soft lidar weighting for AC-GDPO only.

    The mask preserves the original 24-D observation shape and only rescales
    lidar beams. It uses goal direction, previous action tendency, lidar
    surprise, and a safety floor to avoid over-suppressing global awareness.
    """

    def __init__(
        self,
        lidar_start_idx: int = 8,
        lidar_dim: int = 16,
        lidar_range: float = 30.0,
        floor_gain: float = DEFAULT_FLOOR_GAIN,
        target_sigma: float = DEFAULT_TARGET_SIGMA,
        motion_sigma: float = DEFAULT_MOTION_SIGMA,
        surprise_gain: float = DEFAULT_SURPRISE_GAIN,
        turn_side_gain: float = DEFAULT_TURN_SIDE_GAIN,
        ema_decay: float = DEFAULT_EMA_DECAY,
    ) -> None:
        self.lidar_start_idx = lidar_start_idx
        self.lidar_dim = lidar_dim
        self.lidar_range = lidar_range
        self.floor_gain = floor_gain
        self.target_sigma = target_sigma
        self.motion_sigma = motion_sigma
        self.surprise_gain = surprise_gain
        self.turn_side_gain = turn_side_gain
        self.ema_decay = ema_decay

        rel_angles = torch.arange(lidar_dim, dtype=torch.float32) * (2.0 * math.pi / lidar_dim)
        self.rel_beam_angles = self._wrap_to_pi(rel_angles)

        self.prev_lidar: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None
        self.prev_mask: Optional[torch.Tensor] = None

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.prev_lidar = None
        self.prev_actions = torch.zeros(batch_size, 2, device=device)
        self.prev_mask = torch.ones(batch_size, self.lidar_dim, device=device)

    def _build_mask(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        device = obs_tensor.device
        beam_angles = self.rel_beam_angles.to(device).unsqueeze(0)

        angle_to_target = obs_tensor[:, 7].unsqueeze(1)
        lidar = obs_tensor[:, self.lidar_start_idx : self.lidar_start_idx + self.lidar_dim]

        target_focus = torch.exp(
            -0.5 * (self._wrap_to_pi(beam_angles - angle_to_target) / self.target_sigma) ** 2
        )

        prev_actions = self.prev_actions
        assert prev_actions is not None
        prev_v = prev_actions[:, 0].unsqueeze(1)
        prev_w = prev_actions[:, 1].unsqueeze(1)

        forward_motion = (prev_v > 0.15).float()
        motion_focus = torch.exp(-0.5 * (beam_angles / self.motion_sigma) ** 2)
        motion_focus = forward_motion * motion_focus + (1.0 - forward_motion) * 0.5

        front_sector = torch.exp(-0.5 * (beam_angles / (math.pi / 2.5)) ** 2)
        left_sector = torch.sigmoid(4.0 * beam_angles) * front_sector
        right_sector = torch.sigmoid(-4.0 * beam_angles) * front_sector
        turn_focus = torch.clamp(prev_w, -1.0, 1.0) * (left_sector - right_sector)

        if self.prev_lidar is None:
            surprise = torch.zeros_like(lidar)
        else:
            delta = torch.abs(lidar - self.prev_lidar) / self.lidar_range
            surprise = torch.clamp(delta * 3.0, 0.0, 1.0)

        proximity = 1.0 - torch.clamp(lidar / self.lidar_range, 0.0, 1.0)
        sentinel = 0.15 * (1.0 + torch.cos(beam_angles * 4.0))
        sentinel = torch.clamp(sentinel, 0.0, 1.0)

        raw_mask = (
            0.25
            + 0.40 * target_focus
            + 0.20 * motion_focus
            + self.turn_side_gain * turn_focus
            + self.surprise_gain * surprise
            + 0.20 * proximity
            + 0.15 * sentinel
        )
        raw_mask = torch.clamp(raw_mask, 0.0, 1.0)
        return self.floor_gain + (1.0 - self.floor_gain) * raw_mask

    def apply(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if obs_tensor.ndim != 2:
            raise ValueError(f"Expected 2D observation tensor, got shape {tuple(obs_tensor.shape)}")

        if self.prev_actions is None or self.prev_mask is None:
            self.reset(obs_tensor.shape[0], obs_tensor.device)

        lidar = obs_tensor[:, self.lidar_start_idx : self.lidar_start_idx + self.lidar_dim]
        new_mask = self._build_mask(obs_tensor)

        prev_mask = self.prev_mask
        assert prev_mask is not None
        smooth_mask = self.ema_decay * prev_mask + (1.0 - self.ema_decay) * new_mask
        smooth_mask = torch.clamp(smooth_mask, self.floor_gain, 1.0)

        masked_obs = obs_tensor.clone()
        masked_obs[:, self.lidar_start_idx : self.lidar_start_idx + self.lidar_dim] = lidar * smooth_mask

        self.prev_lidar = lidar.detach().clone()
        self.prev_mask = smooth_mask.detach().clone()
        return masked_obs

    def update_action_history(self, actions: torch.Tensor) -> None:
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(f"Expected action tensor shape [B, 2], got {tuple(actions.shape)}")
        self.prev_actions = actions.detach().clone()

    def fork(self, index: int) -> "DSABeamMasker":
        child = DSABeamMasker(
            lidar_start_idx=self.lidar_start_idx,
            lidar_dim=self.lidar_dim,
            lidar_range=self.lidar_range,
            floor_gain=self.floor_gain,
            target_sigma=self.target_sigma,
            motion_sigma=self.motion_sigma,
            surprise_gain=self.surprise_gain,
            turn_side_gain=self.turn_side_gain,
            ema_decay=self.ema_decay,
        )
        if self.prev_lidar is not None:
            child.prev_lidar = self.prev_lidar[index : index + 1].detach().clone()
        if self.prev_actions is not None:
            child.prev_actions = self.prev_actions[index : index + 1].detach().clone()
        if self.prev_mask is not None:
            child.prev_mask = self.prev_mask[index : index + 1].detach().clone()
        return child


def mask_numpy_observation(masker: DSABeamMasker, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.from_numpy(obs).float().to(device)
    masked = masker.apply(obs_tensor)
    return masked.detach().cpu().numpy()
