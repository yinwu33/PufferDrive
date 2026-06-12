"""ldm_goal latent-diffusion model as a DDPO ``SceneInitModel``.

Wraps the vendored ``LDM`` (latent diffusion over autoencoder latents) plus the
frozen ``AutoEncoder`` decoder, and exposes the pair as a stochastic policy whose
MDP is the DDPM ancestral-sampling chain *in latent space*:

  * horizon  H = n_diffusion_timesteps (100)
  * state    s_t = (noisy agent latents x_t, fixed lane-latent conditioning)
  * action   a_t = sampled x_{t-1}  (agent latents only)
  * policy   pi_theta(a_t | s_t) = N( mean_theta(x_t, t),  Sigma_t )
             where mean_theta is the classifier-free-guided posterior mean and
             Sigma_t is the (fixed) DDPM posterior variance.

Differences from the data-space dm_goal adapter (``scene_models/dm_goal.py``):
  * the policy acts on *autoencoder latents* (agent_latent_dim, e.g. 8), not on
    raw agent states; the decode step runs a FROZEN autoencoder decoder, so only
    the LDM is optimised by DDPO and the log-prob lives entirely in latent space.
  * the lane conditioning is the AE-encoded ``data['lane'].latents`` (already
    normalised), held fixed every step (lane-conditioned mode), NOT raw geometry.
  * lanes written to the simulator are the REAL map polylines carried on the
    conditioning graph (``data['lane'].road_points``, physical units) — the map
    is never generated/decoded (project decision).

Project-specific choices baked in (shared with dm_goal):
  * ALL agents are generated, including the ego (local index 0 per scene) and its
    goal; nothing is inpainted from real agent data.
  * the final step (t == 0) is deterministic and excluded from the differentiable
    policy steps.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch_geometric.data import Batch

from ..interfaces import GeneratedScenes, SamplingTrajectory, SceneInitModel
from ..sd_model_ldm import AutoEncoder, LDM, unnormalize_latents, unnormalize_scene_with_goal

_LOG_2PI = math.log(2.0 * math.pi)


def _gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-node diagonal-Gaussian log-density, summed over feature dims.

    x / mean shape: [N, 1, D]; logvar broadcastable. Returns [N].
    """
    var = logvar.exp()
    per_elem = -0.5 * (((x - mean) ** 2) / var + logvar + _LOG_2PI)
    return per_elem.flatten(1).sum(dim=1)


class LdmGoalSceneInitModel(SceneInitModel):
    def __init__(
        self,
        ldm_cfg,
        ae_cfg,
        ldm_ckpt: str | None,
        ae_ckpt: str | None,
        device: str = "cuda",
        use_ema_weights: bool = True,
    ):
        self.cfg = ldm_cfg
        self.cfg_model = ldm_cfg.model
        self.cfg_dataset = ldm_cfg.dataset
        self.device = device

        # --- latent-diffusion policy network -----------------------------------
        self.net = LDM(ldm_cfg).to(device)
        if ldm_ckpt is not None:
            self._load_ldm_checkpoint(ldm_ckpt, use_ema_weights)
        # Disable dropout / label-dropout on the policy: the only intended
        # stochasticity is the diffusion sampling noise, so the recomputed log-prob
        # must be deterministic (otherwise the IS ratio drifts from 1 on epoch 0).
        # eval() does not stop gradients — DDPO still optimises net's weights.
        self.net.eval()

        # Frozen pretrained reference for the KL-to-base penalty.
        self.ref = copy.deepcopy(self.net).to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        # --- frozen autoencoder decoder (NOT part of the policy) ----------------
        self.ae = AutoEncoder(ae_cfg.model).to(device).eval()
        if ae_ckpt is not None:
            self._load_ae_checkpoint(ae_ckpt)
        for p in self.ae.parameters():
            p.requires_grad_(False)

        self._H = int(self.net.n_timesteps)
        self.agent_latent_dim = self.cfg_model.agent_latent_dim
        self.lane_latent_dim = self.cfg_model.lane_latent_dim
        self.diffusion_clip = self.cfg_model.diffusion_clip

        # latent normalisation stats (scalars), injected into the dumped ldm cfg
        self.agent_latents_mean = self.cfg_dataset.agent_latents_mean
        self.agent_latents_std = self.cfg_dataset.agent_latents_std
        self.lane_latents_mean = self.cfg_dataset.lane_latents_mean
        self.lane_latents_std = self.cfg_dataset.lane_latents_std

    # ------------------------------------------------------------------ load
    def _load_ldm_checkpoint(self, ckpt_path: str, use_ema_weights: bool) -> None:
        """Load LDM weights from a scenario-dreamer Lightning checkpoint.

        The LightningModule stores the diffusion network under ``diff_model.*`` and
        persists the EMA shadow separately in ``ckpt['ema_state_dict']`` (ordered as
        ``diff_model.parameters()``). Prefer EMA weights for sampling/eval.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        net_sd = {k[len("diff_model.") :]: v for k, v in sd.items() if k.startswith("diff_model.")}
        missing, unexpected = self.net.load_state_dict(net_sd, strict=False)
        if missing:
            print(f"[ldm_goal] {len(missing)} missing keys on LDM load (e.g. {missing[:3]})")
        if use_ema_weights:
            ema_sd = ckpt.get("ema_state_dict", {})
            shadow = ema_sd.get("shadow_params", [])
            if not shadow:
                shadow = [v for k, v in sd.items() if k.startswith("ema.shadow_params")]
            params = list(self.net.parameters())
            if len(shadow) == len(params) and len(shadow) > 0:
                with torch.no_grad():
                    for p, s in zip(params, shadow):
                        p.copy_(s.to(p.device))
                print(f"[ldm_goal] loaded {len(shadow)} EMA shadow params into LDM")
            else:
                print(
                    f"[ldm_goal] EMA shadow not found/mismatched "
                    f"(shadow={len(shadow)}, params={len(params)}); using raw LDM weights"
                )

    def _load_ae_checkpoint(self, ckpt_path: str) -> None:
        """Load autoencoder weights from a scenario-dreamer Lightning checkpoint.

        ``ScenarioDreamerAutoEncoder`` stores the network under ``model.*``.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        ae_sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = self.ae.load_state_dict(ae_sd, strict=False)
        if missing:
            print(f"[ldm_goal] {len(missing)} missing keys on AE load (e.g. {missing[:3]})")
        if unexpected:
            print(f"[ldm_goal] {len(unexpected)} unexpected keys on AE load (e.g. {unexpected[:3]})")

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

    # ----------------------------------------------------------- conditioning
    def _lane_latents(self, data: Batch) -> torch.Tensor:
        """Fixed lane-latent conditioning, shape [n_lane, 1, lane_latent_dim]."""
        return data["lane"].latents.float().to(self.device).unsqueeze(1)

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

        # Lane-conditioned: the lane latents are held to the real (encoded) map every
        # step; every agent latent (incl. ego and its goal) is generated from noise.
        x_lane = self._lane_latents(data)
        target_lane = x_lane

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
            x_lane = target_lane  # lane chain fully fixed

            # per-scene log-prob of the action over ALL agents under N(mean, var)
            if i > 0:
                node_lp = _gaussian_logprob(x_next, mean_a, logvar_a)
                old_lp[:, j].index_add_(0, agent_batch, node_lp)

            records.append((x_t.detach(), x_next.detach(), i))
            x_agent = x_next

        scenes = self._decode(x_agent, x_lane, data)
        traj = SamplingTrajectory(
            records={"steps": records, "agent_batch": agent_batch, "lane_batch": lane_batch},
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
        agent_batch = trajectory.records["agent_batch"]
        num_scenes = trajectory.num_scenes
        target_lane = self._lane_latents(data)

        out = torch.zeros((num_scenes, len(step_indices)), device=self.device)
        ctx = torch.no_grad() if use_reference else torch.enable_grad()
        with ctx:
            for col, s in enumerate(step_indices.tolist()):
                x_t, x_tm1, i = steps[s]
                if i == 0:
                    continue  # deterministic step carries no policy gradient
                t = torch.full((num_scenes,), i, device=self.device, dtype=torch.long)
                mean_a, logvar_a, _, _ = net.p_mean_variance(
                    x_t, target_lane, data, t[agent_batch], t[trajectory.records["lane_batch"]]
                )
                node_lp = _gaussian_logprob(x_tm1, mean_a, logvar_a)
                out[:, col] = out[:, col].index_add(0, agent_batch, node_lp)
        return out

    # ----------------------------------------------------------- decode
    @torch.no_grad()
    def _decode(self, x_agent, x_lane, data) -> GeneratedScenes:
        # latent space -> raw AE latents
        agent_latents, lane_latents = unnormalize_latents(
            x_agent[:, 0],
            x_lane[:, 0],
            self.agent_latents_mean,
            self.agent_latents_std,
            self.lane_latents_mean,
            self.lane_latents_std,
        )
        # raw AE latents -> normalised scene
        agent_states, _lane_states, agent_types, _, _ = self.ae.forward_decoder(
            agent_latents, lane_latents, data
        )
        # normalised scene -> physical units (incl. goal at indices 7, 8)
        agent_states, _ = unnormalize_scene_with_goal(
            agent_states, _lane_states, self.cfg_dataset
        )

        # Lanes are the REAL map polylines carried on the conditioning graph
        # (physical units, attached at dump time), NOT decoded from latents.
        lane_polylines = data["lane"].road_points

        return GeneratedScenes(
            agent_states=agent_states,
            agent_types=agent_types,
            agent_scene_idx=data["agent"].batch,
            lane_polylines=lane_polylines,
            num_scenes=int(data.batch_size),
            meta={"lane_scene_idx": data["lane"].batch},
        )
