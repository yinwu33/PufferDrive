"""dm_goal diffusion model as a DDPO ``SceneInitModel``.

Wraps the vendored ``DMGoal`` network (scenario-dreamer) and exposes it as a
stochastic policy whose MDP is the DDPM ancestral-sampling chain:

  * horizon  H = n_diffusion_timesteps (100)
  * state    s_t = (noisy agent latents x_t, fixed conditioning graph)
  * action   a_t = sampled x_{t-1}
  * policy   pi_theta(a_t | s_t) = N( mean_theta(x_t, t),  Sigma_t )
             where mean_theta is the classifier-free-guided posterior mean and
             Sigma_t is the (fixed) DDPM posterior variance.

Project-specific choices baked in here:
  * mode = lane-conditioned: the lane/map chain is held to the real map every step,
    so only the AGENT chain is stochastic -> the policy only acts on agents.
  * ego (local index 0 per scene) is held fixed via inpainting every step, so the
    ego is never part of the policy and never contributes to the log-prob. DDPO
    only perturbs the non-ego agents to make the scene adversarial for the planner.
  * the final step (t == 0) is deterministic (DDPM zeroes the noise) and is excluded
    from the set of differentiable "policy steps".
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch_geometric.data import Batch

from ..interfaces import GeneratedScenes, SamplingTrajectory, SceneInitModel
from ..sd_model import DMGoal, unnormalize_scene_with_goal

_LOG_2PI = math.log(2.0 * math.pi)


def _gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-node diagonal-Gaussian log-density, summed over feature dims.

    x / mean shape: [N, 1, D]; logvar broadcastable. Returns [N].
    """
    var = logvar.exp()
    per_elem = -0.5 * (((x - mean) ** 2) / var + logvar + _LOG_2PI)
    return per_elem.flatten(1).sum(dim=1)


class DMGoalSceneInitModel(SceneInitModel):
    def __init__(self, cfg, ckpt_path: str | None, device: str = "cuda", use_ema_weights: bool = True):
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset
        self.device = device

        self.net = DMGoal(cfg).to(device)
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path, use_ema_weights)

        # Frozen pretrained reference for the KL-to-base penalty.
        self.ref = copy.deepcopy(self.net).to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        self._H = int(self.net.n_timesteps)
        self.agent_latent_dim = self.cfg_model.agent_latent_dim
        self.lane_latent_dim = self.cfg_model.lane_latent_dim
        self.diffusion_clip = self.cfg_model.diffusion_clip

    # ------------------------------------------------------------------ load
    def _load_checkpoint(self, ckpt_path: str, use_ema_weights: bool) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        # Lightning module stores the network under ``diff_model.*`` and the EMA
        # shadow under ``ema.*``. Prefer EMA weights (used for sampling/eval).
        net_sd = {k[len("diff_model.") :]: v for k, v in sd.items() if k.startswith("diff_model.")}
        missing, unexpected = self.net.load_state_dict(net_sd, strict=False)
        if missing:
            print(f"[dm_goal] {len(missing)} missing keys on load (e.g. {missing[:3]})")
        if use_ema_weights:
            shadow = [v for k, v in sd.items() if k.startswith("ema.shadow_params")]
            params = [p for p in self.net.parameters() if p.requires_grad]
            if len(shadow) == len(params) and len(shadow) > 0:
                with torch.no_grad():
                    for p, s in zip(params, shadow):
                        p.copy_(s.to(p.device))
                print(f"[dm_goal] loaded {len(shadow)} EMA shadow params into net")
            else:
                print("[dm_goal] EMA shadow not found/mismatched; using raw weights")

    # --------------------------------------------------------- ego inpainting
    @staticmethod
    def _ego_mask(data) -> torch.Tensor:
        """Boolean [N_agents] marking the ego = first agent node of each scene."""
        batch = data["agent"].batch
        ego = torch.zeros_like(batch, dtype=torch.bool)
        # first occurrence of each scene id
        first = torch.ones(int(batch.max()) + 1, dtype=torch.bool, device=batch.device)
        for i in range(batch.shape[0]):
            b = int(batch[i])
            if first[b]:
                ego[i] = True
                first[b] = False
        return ego

    # ------------------------------------------------------------ properties
    @property
    def num_sampling_steps(self) -> int:
        return self._H

    def trainable_parameters(self):
        return (p for p in self.net.parameters() if p.requires_grad)

    def state_dict(self) -> dict:
        return self.net.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd)

    # ---------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(self, conditioning: Batch) -> tuple[GeneratedScenes, SamplingTrajectory]:
        data = conditioning.to(self.device)
        net = self.net
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        num_scenes = int(data.batch_size)

        n_agent = data["agent"].x.shape[0]
        x_agent = torch.randn((n_agent, 1, self.agent_latent_dim), device=self.device)
        x_lane = torch.randn((data["lane"].x.shape[0], 1, self.lane_latent_dim), device=self.device)

        # Fixed conditioning targets: full map + ego agent.
        target_lane = net._lane_target(data).to(self.device)
        target_agent = net._agent_target(data).to(self.device)
        ego_mask = self._ego_mask(data)
        x_lane = target_lane                       # lane chain fully fixed
        x_agent[ego_mask] = target_agent[ego_mask]  # ego fixed

        records = []  # (x_t, x_tm1, t_value) per step; agents only
        old_lp = torch.zeros((num_scenes, self._H), device=self.device)

        for j, i in enumerate(reversed(range(self._H))):
            t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
            t_agent = t[agent_batch]
            t_lane = t[lane_batch]
            mean_a, logvar_a, _, _ = net.p_mean_variance(x_agent, x_lane, data, t_agent, t_lane)

            x_t = x_agent
            if i > 0:
                noise = torch.randn_like(x_agent)
                x_next = mean_a + (0.5 * logvar_a).exp() * noise
            else:
                x_next = mean_a  # deterministic final step

            x_next = torch.clip(x_next, -self.diffusion_clip, self.diffusion_clip)
            x_next[ego_mask] = target_agent[ego_mask]  # keep ego fixed throughout
            x_lane = target_lane

            # per-scene log-prob of the non-ego action under N(mean, var)
            if i > 0:
                node_lp = _gaussian_logprob(x_next, mean_a, logvar_a)
                node_lp = node_lp.masked_fill(ego_mask, 0.0)
                old_lp[:, j].index_add_(0, agent_batch, node_lp)

            records.append((x_t.detach(), x_next.detach(), i))
            x_agent = x_next

        scenes = self._decode(x_agent, x_lane, data, ego_mask)
        traj = SamplingTrajectory(
            records={"steps": records, "ego_mask": ego_mask, "agent_batch": agent_batch,
                     "lane_batch": lane_batch},
            old_logprob=old_lp.detach(),
            num_steps=self._H,
            num_scenes=num_scenes,
        )
        return scenes, traj

    # ----------------------------------------------------------- scoring
    def trajectory_logprob(
        self,
        trajectory: SamplingTrajectory,
        conditioning: Batch,
        step_indices: torch.Tensor,
        *,
        use_reference: bool = False,
    ) -> torch.Tensor:
        data = conditioning.to(self.device)
        net = self.ref if use_reference else self.net
        steps = trajectory.records["steps"]
        ego_mask = trajectory.records["ego_mask"]
        agent_batch = trajectory.records["agent_batch"]
        lane_batch = trajectory.records["lane_batch"]
        num_scenes = trajectory.num_scenes
        target_lane = self.net._lane_target(data).to(self.device)

        out = torch.zeros((num_scenes, len(step_indices)), device=self.device)
        ctx = torch.no_grad() if use_reference else torch.enable_grad()
        with ctx:
            for col, s in enumerate(step_indices.tolist()):
                x_t, x_tm1, i = steps[s]
                if i == 0:
                    continue  # deterministic step carries no policy gradient
                t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
                mean_a, logvar_a, _, _ = net.p_mean_variance(
                    x_t, target_lane, data, t[agent_batch], t[lane_batch]
                )
                node_lp = _gaussian_logprob(x_tm1, mean_a, logvar_a)
                node_lp = node_lp.masked_fill(ego_mask, 0.0)
                out[:, col] = out[:, col].index_add(0, agent_batch, node_lp)
        return out

    # ----------------------------------------------------------- decode
    @torch.no_grad()
    def _decode(self, x_agent, x_lane, data, ego_mask) -> GeneratedScenes:
        agent_states, lane_states, agent_types, _, _ = self.net.decode_outputs(
            x_agent[:, 0], x_lane, data
        )
        agent_states, lane_states = unnormalize_scene_with_goal(
            agent_states, lane_states, self.cfg_dataset
        )
        return GeneratedScenes(
            agent_states=agent_states,
            agent_types=agent_types,
            agent_scene_idx=data["agent"].batch,
            lane_polylines=lane_states,
            num_scenes=int(data.batch_size),
            meta={"ego_mask": ego_mask},
        )
