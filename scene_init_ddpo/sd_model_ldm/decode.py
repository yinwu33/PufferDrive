"""Output (un)normalisation vendored from scenario-dreamer ``utils/data_helpers.py``.

The LDM diffuses *normalised autoencoder latents*; decoding a sample therefore
needs two stages:
  1. ``unnormalize_latents`` : (latent - mean) / std  ->  raw AE latents
  2. AE decoder (see ``autoencoder.py``)               ->  normalised scene in [-1, 1]
  3. ``unnormalize_scene_with_goal`` : [-1, 1]         ->  physical units

Agent state layout after stage 3 (state_dim == 9):
    [pos_x, pos_y, speed, cos_theta, sin_theta, length, width, goal_x, goal_y]
"""

from typing import Tuple

import numpy as np
import torch


def reparameterize(mu, log_var):
    """Reparameterization trick to sample a latent from N(mu, exp(log_var))."""
    assert mu.shape == log_var.shape
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def normalize_latents(agent_latents, lane_latents, agent_latents_mean, agent_latents_std,
                      lane_latents_mean, lane_latents_std):
    """Normalize agent/lane latents with the per-dataset latent mean/std."""
    agent_latents = (agent_latents - agent_latents_mean) / agent_latents_std
    lane_latents = (lane_latents - lane_latents_mean) / lane_latents_std
    return agent_latents, lane_latents


def unnormalize_latents(agent_latents, lane_latents, agent_latents_mean, agent_latents_std,
                        lane_latents_mean, lane_latents_std):
    """Invert ``normalize_latents`` back to raw autoencoder latents."""
    agent_latents = agent_latents * agent_latents_std + agent_latents_mean
    lane_latents = lane_latents * lane_latents_std + lane_latents_mean
    return agent_latents, lane_latents


def unnormalize_scene(
    agent_states: np.ndarray,
    road_points: np.ndarray,
    fov: float,
    min_speed: float,
    max_speed: float,
    min_length: float,
    max_length: float,
    min_width: float,
    max_width: float,
    min_lane_x: float,
    max_lane_x: float,
    min_lane_y: float,
    max_lane_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unnormalize agent states and lane points from [-1, 1] to physical scale."""
    agent_states[:, 0] = ((torch.clip(agent_states[:, 0], -1, 1) + 1) / 2) * fov + (-1 * fov / 2)
    agent_states[:, 1] = ((torch.clip(agent_states[:, 1], -1, 1) + 1) / 2) * fov + (-1 * fov / 2)
    agent_states[:, 2] = ((torch.clip(agent_states[:, 2], -1, 1) + 1) / 2) * (max_speed - min_speed) + min_speed
    agent_states[:, 3] = torch.clip(agent_states[:, 3], -1, 1)
    agent_states[:, 4] = torch.clip(agent_states[:, 4], -1, 1)
    agent_states[:, 5] = ((torch.clip(agent_states[:, 5], -1, 1) + 1) / 2) * (max_length - min_length) + min_length
    agent_states[:, 6] = ((torch.clip(agent_states[:, 6], -1, 1) + 1) / 2) * (max_width - min_width) + min_width

    lower_clip, upper_clip = -1000, 1000
    road_points[:, :, 0] = (
        (torch.clip(road_points[:, :, 0], lower_clip, upper_clip) + 1) / 2
    ) * (max_lane_x - min_lane_x) + min_lane_x
    road_points[:, :, 1] = (
        (torch.clip(road_points[:, :, 1], lower_clip, upper_clip) + 1) / 2
    ) * (max_lane_y - min_lane_y) + min_lane_y
    return agent_states, road_points


def unnormalize_scene_with_goal(agent_states, lane_states, cfg_dataset):
    """Unnormalize a goal-augmented sample, including goal coords at indices 7, 8."""
    agent_states, lane_states = unnormalize_scene(
        agent_states,
        lane_states,
        fov=cfg_dataset.fov,
        min_speed=cfg_dataset.min_speed,
        max_speed=cfg_dataset.max_speed,
        min_length=cfg_dataset.min_length,
        max_length=cfg_dataset.max_length,
        min_width=cfg_dataset.min_width,
        max_width=cfg_dataset.max_width,
        min_lane_x=cfg_dataset.min_lane_x,
        min_lane_y=cfg_dataset.min_lane_y,
        max_lane_x=cfg_dataset.max_lane_x,
        max_lane_y=cfg_dataset.max_lane_y,
    )
    if agent_states.shape[-1] >= 9:
        agent_states[:, 7] = ((torch.clip(agent_states[:, 7], -1, 1) + 1) / 2) * cfg_dataset.fov - cfg_dataset.fov / 2
        agent_states[:, 8] = ((torch.clip(agent_states[:, 8], -1, 1) + 1) / 2) * cfg_dataset.fov - cfg_dataset.fov / 2
    return agent_states, lane_states
