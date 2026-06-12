"""Offline conditioning dump for ldm_goal — RUN IN THE scenario-dreamer VENV.

Builds a pool of conditioning graphs for DDPO over the *latent diffusion* model
(ldm_goal) and exports the resolved LDM + autoencoder configs, so the PufferDrive
side never has to vendor scenario-dreamer's dataset/datamodule.

Unlike the dm_goal dump, each conditioning graph here carries:
  * ``data['lane'].latents``      – normalised AE lane latents (the fixed map
                                     conditioning the LDM denoises against), and
  * ``data['lane'].road_points``  – the REAL lane polylines in PHYSICAL units
                                     (used to write the simulator ``.bin``; the map
                                     is never generated/decoded — project decision).
Both repos pin torch_geometric 2.6.1, so the pickled ``ScenarioDreamerData`` graphs
load on either side.

Usage (from the scenario-dreamer repo root):
    .venv/bin/python /path/to/PufferDrive/scene_init_ddpo/tools/dump_conditioning_ldm.py \
        --num-scenes 2000 \
        --out /path/to/PufferDrive/scene_init_ddpo/data/cond_pool_ldm \
        --split val

Outputs:
    <out>/scene_00000.pt ...      # individual HeteroData conditioning graphs
    <out>/ldm_model_cfg.yaml      # resolved cfg.ldm_goal (incl. latent mean/std)
    <out>/ae_model_cfg.yaml       # resolved cfg.ae_goal (autoencoder decoder)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.train_helpers import set_latent_stats
from datasets.waymo.dataset_ldm_waymo import WaymoDatasetLDM


def _unnormalize_lane_polylines(road_points, cfg_dataset):
    """Normalised [-1, 1] lane points -> physical units. Shape [n_lane, n_pts, >=2]."""
    rp = torch.as_tensor(road_points, dtype=torch.float32).clone()
    rp[:, :, 0] = ((torch.clip(rp[:, :, 0], -1, 1) + 1) / 2) * (
        cfg_dataset.max_lane_x - cfg_dataset.min_lane_x
    ) + cfg_dataset.min_lane_x
    rp[:, :, 1] = ((torch.clip(rp[:, :, 1], -1, 1) + 1) / 2) * (
        cfg_dataset.max_lane_y - cfg_dataset.min_lane_y
    ) + cfg_dataset.min_lane_y
    return rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-scenes", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--config-name", default="config")
    ap.add_argument("--cfgs-dir", default=str(Path.cwd() / "cfgs"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=args.cfgs_dir, version_base=None):
        cfg = compose(config_name=args.config_name, overrides=["model_name=ldm_goal"])

    dataset_name = cfg.dataset_name.name
    cfg_ldm = cfg.ldm_goal
    cfg_ae = cfg.ae_goal
    OmegaConf.set_struct(cfg_ldm, False)
    OmegaConf.set_struct(cfg_ae, False)
    cfg_ldm.dataset_name = dataset_name
    cfg_ae.dataset_name = dataset_name

    # Inject the cached latent mean/std into the ldm dataset cfg; the dataset's
    # sample_latents() normalises with these, and the PufferDrive decode step needs
    # them to invert the normalisation, so they must be baked into the dumped cfg.
    cfg_ldm = set_latent_stats(cfg_ldm)

    OmegaConf.save(
        OmegaConf.create(OmegaConf.to_container(cfg_ldm, resolve=True)),
        args.out / "ldm_model_cfg.yaml",
    )
    OmegaConf.save(
        OmegaConf.create(OmegaConf.to_container(cfg_ae, resolve=True)),
        args.out / "ae_model_cfg.yaml",
    )

    dataset = WaymoDatasetLDM(cfg_ldm.dataset, split_name=args.split)
    cfg_dataset = cfg_ldm.dataset

    n = min(args.num_scenes, len(dataset))
    print(f"Dumping {n} ldm_goal conditioning graphs from the {args.split} split -> {args.out}")
    for i in range(n):
        d = dataset[i]  # ScenarioDreamerData: latents, edge indices, counts, ids

        # Attach the REAL lane polylines (physical units) for the simulator .bin.
        # The raw cache pkl holds normalised road_points; read & unnormalise them.
        with open(dataset.files[i], "rb") as f:
            raw = pickle.load(f)
        d["lane"].road_points = _unnormalize_lane_polylines(raw["road_points"], cfg_dataset)

        torch.save(d, args.out / f"scene_{i:05d}.pt")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}")
    print("done")


if __name__ == "__main__":
    main()
