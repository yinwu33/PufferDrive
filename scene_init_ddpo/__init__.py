"""scene_init_ddpo: a PufferDrive-hosted scenario-RL platform.

Trains scene-initialisation generative models with DDPO so they produce scenes
that are *critical* for a frozen PufferDrive planner (e.g. high ego-collision
rate). The generative model is a vendored, swappable ``SceneInitModel``; the
reward is an in-process PufferDrive rollout. See README.md for the architecture.
"""

from .interfaces import GeneratedScenes, SamplingTrajectory, SceneInitModel

__all__ = ["SceneInitModel", "GeneratedScenes", "SamplingTrajectory"]
