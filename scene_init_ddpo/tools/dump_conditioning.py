"""Offline conditioning dump — RUN IN THE scenario-dreamer VENV, NOT PufferDrive.

Builds a pool of conditioning graphs (real maps + real ego) for DDPO and exports
the resolved dm_goal config, so the PufferDrive side never has to vendor
scenario-dreamer's dataset/datamodule. Both repos pin torch_geometric 2.6.1, so
the pickled HeteroData graphs load on either side.

Usage (from the scenario-dreamer repo root):
    .venv/bin/python /path/to/PufferDrive/scene_init_ddpo/tools/dump_conditioning.py \
        --num-scenes 2000 \
        --out /path/to/PufferDrive/scene_init_ddpo/data/cond_pool \
        --split val

Outputs:
    <out>/scene_00000.pt ...   # individual HeteroData conditioning graphs
    <out>/model_cfg.yaml       # resolved cfg.dm_goal for DMGoalSceneInitModel
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-scenes", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--config-name", default="config_dm_goal_train")
    ap.add_argument("--cfgs-dir", default=str(Path.cwd() / "cfgs"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=args.cfgs_dir, version_base=None):
        cfg = compose(config_name=args.config_name)

    # Resolved dm_goal config consumed by the vendored DMGoal on the PufferDrive side.
    OmegaConf.save(cfg.dm_goal, args.out / "model_cfg.yaml")

    datamodule = instantiate(cfg.dm_goal.datamodule, dataset_cfg=cfg.dm_goal.dataset)
    datamodule.setup()
    dataset = datamodule.val_dataset if args.split == "val" else datamodule.train_dataset

    n = min(args.num_scenes, len(dataset))
    print(f"Dumping {n} conditioning graphs from the {args.split} split -> {args.out}")
    for i in range(n):
        data = dataset[i]
        torch.save(data, args.out / f"scene_{i:05d}.pt")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}")
    print("done")


if __name__ == "__main__":
    main()
