"""Vendored scenario-dreamer dm_goal inference closure (nn.Module level only).

This package is a self-contained copy of the *minimal* set of scenario-dreamer
modules needed to (a) instantiate the dm_goal diffusion network, (b) load its
trained weights, and (c) run/score the denoising chain. It deliberately excludes
scenario-dreamer's PyTorch-Lightning module, dataset/datamodule and viz code.

Source: github scenario-dreamer @ commit used for the dm_goal checkpoint.
Keep edits minimal so the closure can be re-synced from upstream.
"""

from .dm import DM
from .dm_goal import DMGoal
from .dit import DiT
from .decode import unnormalize_scene, unnormalize_scene_with_goal

__all__ = ["DM", "DMGoal", "DiT", "unnormalize_scene", "unnormalize_scene_with_goal"]
