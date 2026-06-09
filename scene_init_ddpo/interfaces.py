"""Generic scenario-init RL interface.

This module defines the contract that decouples *which generative model produces
scenes* from *the DDPO/RL machinery and the PufferDrive reward*. Today the only
implementation is the vendored dm_goal diffusion model
(``scene_models.dm_goal.DMGoalSceneInitModel``); future scene-init models (other
diffusion variants, flow matching, autoregressive, ...) only need to implement
``SceneInitModel`` to plug into the same trainer and reward oracle.

Design assumption
-----------------
DDPO treats the *generative sampling process* as an MDP whose terminal reward is
computed by rolling out the generated scene in PufferDrive. Any model that can be
expressed as a sequence of stochastic steps with a tractable per-step log-density
(diffusion ancestral sampling, stochastic DDIM, autoregressive token sampling,
...) fits this interface. The interface therefore exposes:

  * ``sample``            – draw a batch of scenes + record the sampling trajectory
  * ``trajectory_logprob``– (with grad) recompute per-step log-prob for a subset of
                            steps, for the current parameters [policy] and, frozen,
                            for the pretrained reference [KL regularisation]
  * ``decode``            – turn a sampled batch into simulator-ready scenes

The ``num_steps`` / random-k-subset mechanics are what keep DDPO's backward pass
affordable: we only differentiate through ``k`` randomly chosen sampling steps
per update instead of the full horizon.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GeneratedScenes:
    """Decoded, simulator-ready output of one ``sample`` call (a batch of scenes).

    All agent tensors are flattened across the batch (PyG style); ``agent_scene_idx``
    maps every agent row to its scene in ``[0, num_scenes)``. By convention the
    ego/SDC is local index 0 within each scene (see project decision: index-0 agent
    is the ego and is held fixed via inpainting).
    """

    agent_states: torch.Tensor      # [N_agents, state_dim] physical units; see decode.py layout
    agent_types: torch.Tensor       # [N_agents] int class ids
    agent_scene_idx: torch.Tensor   # [N_agents] -> scene id in [0, num_scenes)
    lane_polylines: Any             # per-scene fixed map geometry (conditioning, not generated)
    num_scenes: int
    meta: dict = field(default_factory=dict)


@dataclass
class SamplingTrajectory:
    """Opaque-to-DDPO record of a sampling rollout, sufficient to recompute log-probs.

    ``old_logprob`` is the per-scene, per-step log-density evaluated under the
    behaviour parameters at sampling time (used as the PPO/IS reference). The model
    keeps whatever internal ``records`` it needs (e.g. per-step (x_t, x_{t-1}, t)).
    """

    records: Any                    # model-specific; consumed only by the same model
    old_logprob: torch.Tensor       # [num_scenes, num_steps] detached
    num_steps: int                  # horizon length (e.g. number of denoising steps)
    num_scenes: int


class SceneInitModel(abc.ABC):
    """A stochastic scene generator usable as a DDPO policy.

    Implementations wrap a concrete generative model and expose the minimal surface
    DDPO needs. The trainer never imports model internals — only this interface.
    """

    # --- optimisation surface -------------------------------------------------
    @abc.abstractmethod
    def trainable_parameters(self):
        """Iterable of nn.Parameters optimised by DDPO (the policy)."""

    @abc.abstractmethod
    def state_dict(self) -> dict: ...

    @abc.abstractmethod
    def load_state_dict(self, sd: dict) -> None: ...

    # --- sampling -------------------------------------------------------------
    @abc.abstractmethod
    @torch.no_grad()
    def sample(self, conditioning: Any) -> tuple[GeneratedScenes, SamplingTrajectory]:
        """Draw a batch of scenes for the given conditioning (fixed real maps + fixed
        ego). Returns the decoded scenes and the recorded sampling trajectory.
        Must run without building a graph (collection phase is no_grad)."""

    # --- scoring (DDPO update phase) -----------------------------------------
    @abc.abstractmethod
    def trajectory_logprob(
        self,
        trajectory: SamplingTrajectory,
        conditioning: Any,
        step_indices: torch.Tensor,
        *,
        use_reference: bool = False,
    ) -> torch.Tensor:
        """Recompute per-scene log-prob for the sampling steps in ``step_indices``.

        With ``use_reference=False`` this runs the *current* (trainable) parameters
        and the result is differentiable -> policy gradient. With
        ``use_reference=True`` it runs the frozen pretrained model under no_grad ->
        used for the KL-to-base penalty that keeps generated scenes on-manifold.

        Returns a tensor of shape ``[num_scenes, len(step_indices)]``.
        """

    @property
    @abc.abstractmethod
    def num_sampling_steps(self) -> int:
        """Horizon length H; DDPO samples k <= H steps to differentiate per update."""
