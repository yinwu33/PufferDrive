"""Vendored scenario-dreamer ldm_goal inference closure (nn.Module level only).

Self-contained copy of the *minimal* set of scenario-dreamer modules needed to
(a) instantiate the latent-diffusion network ``LDM`` and the ``AutoEncoder``
decoder, (b) load their trained weights, and (c) run/score the latent denoising
chain + decode latents to physical scenes. Excludes scenario-dreamer's
PyTorch-Lightning modules, datasets/datamodules and viz code.

Kept separate from ``sd_model`` (the dm_goal closure) because the LDM's DiT and
the autoencoder differ from the data-space dm_goal network. Keep edits minimal so
the closure can be re-synced from upstream ``nn_modules`` / ``utils``.
"""

from .ldm import LDM
from .autoencoder import AutoEncoder
from .dit import DiT
from .decode import (
    normalize_latents,
    unnormalize_latents,
    unnormalize_scene,
    unnormalize_scene_with_goal,
    reparameterize,
)

__all__ = [
    "LDM",
    "AutoEncoder",
    "DiT",
    "normalize_latents",
    "unnormalize_latents",
    "unnormalize_scene",
    "unnormalize_scene_with_goal",
    "reparameterize",
]
