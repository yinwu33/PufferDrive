"""DDPO training loop for scene-init models.

Wires together: conditioning pool (real maps + ego) -> SceneInitModel.sample
(record trajectory) -> scene_codec (.bin) -> PufferDriveReward (frozen planner
rollout, ego metrics) -> advantages -> DDPO policy-gradient update over a random
subset of k sampling steps (+ optional KL-to-base).

Run inside the PufferDrive venv:
    .venv/bin/python -m scene_init_ddpo.train --config scene_init_ddpo/config/ddpo_dm_goal.yaml
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .conditioning import ConditioningPool
from .ddpo import DDPOConfig, compute_advantages, ddpo_loss
from .reward import PufferDriveReward
from .scene_codec import write_scene_bins
from .scene_models import DMGoalSceneInitModel


def build_model(cfg) -> DMGoalSceneInitModel:
    model_cfg = OmegaConf.load(cfg.model_cfg)  # resolved dm_goal cfg dumped from SD
    return DMGoalSceneInitModel(
        model_cfg,
        ckpt_path=cfg.model_ckpt,
        device=cfg.device,
        use_ema_weights=cfg.get("use_ema_weights", True),
    )


def train(cfg):
    device = cfg.device
    model = build_model(cfg)
    pool = ConditioningPool(cfg.conditioning_pool, device=device, seed=cfg.seed)
    reward = PufferDriveReward(
        cfg.planner_ckpt,
        device=device,
        sim_steps=cfg.sim_steps,
        max_agents_per_scene=cfg.max_agents_per_scene,
    )
    ddpo_cfg = DDPOConfig(**OmegaConf.to_container(cfg.ddpo, resolve=True))
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    H = model.num_sampling_steps
    stochastic_steps = torch.arange(0, H - 1)  # exclude deterministic t=0 final step
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for it in range(cfg.num_iterations):
        # ---- rollout / collect -------------------------------------------------
        cond = pool.sample_batch(cfg.batch_size)
        scenes, traj = model.sample(cond)

        with tempfile.TemporaryDirectory() as tmp:
            n = write_scene_bins(scenes, tmp)
            metrics = reward.evaluate(tmp, n)

        rewards = torch.as_tensor(metrics["reward"], device=device)
        advantages = compute_advantages(rewards, ddpo_cfg)

        # ---- DDPO update over random-k steps ----------------------------------
        for _ in range(cfg.inner_epochs):
            k_idx = stochastic_steps[torch.randperm(len(stochastic_steps))[: cfg.k_steps]]
            new_lp = model.trajectory_logprob(traj, cond, k_idx)
            old_lp = traj.old_logprob[:, k_idx]
            ref_lp = (
                model.trajectory_logprob(traj, cond, k_idx, use_reference=True)
                if ddpo_cfg.kl_coef > 0
                else None
            )
            loss, log = ddpo_loss(new_lp, old_lp, advantages, ddpo_cfg, ref_lp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), cfg.grad_clip)
            opt.step()

        crit = float(rewards.gt(0).float().mean())
        inval = float(metrics["init_invalid"].mean())
        print(
            f"[it {it:04d}] reward={rewards.mean():.3f} critical_rate={crit:.3f} "
            f"init_invalid={inval:.3f} loss={log['loss']:.4f} "
            f"ratio={log.get('ratio_mean', 1.0):.3f} kl={log.get('kl_to_base', 0.0):.4f}"
        )

        if cfg.save_every and (it + 1) % cfg.save_every == 0:
            torch.save(model.state_dict(), out_dir / f"dm_goal_ddpo_{it + 1:05d}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. batch_size=16")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    train(cfg)


if __name__ == "__main__":
    main()
