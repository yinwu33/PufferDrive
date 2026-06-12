"""DDPO training loop for scene-init models.

Wires together: conditioning pool (real maps) -> SceneInitModel.sample (record
trajectory) -> scene_codec (.bin) -> PufferDriveReward (frozen planner rollout,
ego metrics) -> advantages -> DDPO policy-gradient update over a random subset of
k sampling steps (+ optional KL-to-base).

Key metrics are logged to wandb every iteration; every ``eval_every`` iterations a
held-out val pass rolls out ``eval_num_scenes`` scenes and logs their planner-rollout
trajectories as images.

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
from .scene_codec import ego_goals, write_scene_bins
from .scene_models import DMGoalSceneInitModel, LdmGoalSceneInitModel


def build_model(cfg):
    """Build the DDPO scene-init policy selected by ``cfg.model_type``.

    ``dm_goal`` (default): data-space diffusion (single network).
    ``ldm_goal``: latent diffusion over autoencoder latents + frozen AE decoder.
    """
    model_type = cfg.get("model_type", "dm_goal")
    if model_type == "ldm_goal":
        ldm_cfg = OmegaConf.load(cfg.ldm_model_cfg)  # resolved cfg.ldm_goal (incl. latent stats)
        ae_cfg = OmegaConf.load(cfg.ae_model_cfg)    # resolved cfg.ae_goal (decoder)
        return LdmGoalSceneInitModel(
            ldm_cfg,
            ae_cfg,
            ldm_ckpt=cfg.ldm_ckpt,
            ae_ckpt=cfg.ae_ckpt,
            device=cfg.device,
            use_ema_weights=cfg.get("use_ema_weights", True),
        )
    model_cfg = OmegaConf.load(cfg.model_cfg)  # resolved dm_goal cfg dumped from SD
    return DMGoalSceneInitModel(
        model_cfg,
        ckpt_path=cfg.model_ckpt,
        device=cfg.device,
        use_ema_weights=cfg.get("use_ema_weights", True),
    )


@torch.no_grad()
def evaluate_and_visualize(model, eval_pool, reward, cfg, it, wandb):
    """Roll out a fixed set of held-out val scenes and build trajectory media.

    ``save_gif`` (config): if True, log a per-step GIF of each rollout; otherwise log
    the static first-episode summary image. Each media's title carries the per-scene
    reward breakdown (total + collision / offroad / init_invalid).
    """
    import matplotlib.pyplot as plt

    from .viz import render_rollout, render_rollout_frames, save_gif

    n = min(int(cfg.eval_num_scenes), len(eval_pool))
    # Fixed conditioning + fixed noise seed -> visuals across iterations differ only
    # by how the policy has changed, not by the sampled map or noise.
    torch.manual_seed(int(cfg.seed))
    cond = eval_pool.batch_from_indices(list(range(n)))
    scenes, _ = model.sample(cond)

    with tempfile.TemporaryDirectory() as tmp:
        m = write_scene_bins(scenes, tmp)
        metrics = reward.evaluate(tmp, m, record_trajectories=True, ego_goals=ego_goals(scenes))

    lanes = scenes.lane_polylines
    if isinstance(lanes, torch.Tensor):
        lanes = lanes.detach().cpu().numpy()
    lane_scene_idx = scenes.meta["lane_scene_idx"].detach().cpu().numpy()
    states = scenes.agent_states.detach().cpu().numpy()
    types = scenes.agent_types.detach().cpu().numpy()
    agent_scene_idx = scenes.agent_scene_idx.detach().cpu().numpy()

    save_gif_mode = bool(cfg.get("save_gif", False))
    gif_dir = Path(cfg.output_dir) / "eval_media"
    if save_gif_mode:
        gif_dir.mkdir(parents=True, exist_ok=True)

    media = []
    if wandb is not None:
        for s in range(m):
            a_sel = agent_scene_idx == s
            kwargs = dict(
                agent_states=states[a_sel],
                agent_types=types[a_sel],
                reward=metrics["reward"][s],
                ego_collision=metrics["ego_collision"][s] > 0,
                ego_offroad=metrics["ego_offroad"][s] > 0,
                init_invalid=metrics["init_invalid"][s] > 0,
                title=f"it{it} scene{s}",
            )
            if save_gif_mode:
                frames = render_rollout_frames(
                    metrics["trajectories"][s], lanes[lane_scene_idx == s],
                    max_frames=int(cfg.get("gif_max_frames", 50)), **kwargs,
                )
                # [T,H,W,3] -> wandb.Video needs (T,C,H,W); pass via gif file (no moviepy)
                path = str(gif_dir / f"scene_{s}.gif")
                save_gif(frames, path, fps=int(cfg.get("gif_fps", 10)))
                media.append(wandb.Video(path, format="gif"))
            else:
                fig = render_rollout(metrics["trajectories"][s], lanes[lane_scene_idx == s], **kwargs)
                media.append(wandb.Image(fig))
                plt.close(fig)
    return metrics, media


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
    trainable_params = list(model.trainable_parameters())
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- wandb + eval pool ----------------------------------------------------
    wandb = None
    if cfg.get("wandb", {}).get("enabled", False):
        import wandb as _wandb

        _wandb.init(
            project=cfg.wandb.get("project", "scene_init_ddpo"),
            entity=cfg.wandb.get("entity", None),
            name=cfg.wandb.get("run_name", None),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb = _wandb

    eval_every = int(cfg.get("eval_every", 0))
    eval_pool = None
    if eval_every > 0:
        eval_pool = ConditioningPool(cfg.eval_pool, device=device, seed=cfg.seed)

    H = model.num_sampling_steps
    min_diffusion_t = int(cfg.get("min_diffusion_t", 5))
    if min_diffusion_t < 1 or min_diffusion_t >= H:
        raise ValueError(f"min_diffusion_t must be in [1, {H - 1}], got {min_diffusion_t}")
    # records are indexed in reverse diffusion order: index 0 is t=H-1 and
    # index H-1 is deterministic t=0. Skip very low-t steps because their DDPM
    # posterior variance is tiny, making log-prob ratios numerically brittle.
    stochastic_steps = torch.arange(0, H - min_diffusion_t)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for it in range(cfg.num_iterations):
        # ---- rollout / collect -------------------------------------------------
        cond = pool.sample_batch(cfg.batch_size)
        scenes, traj = model.sample(cond)

        with tempfile.TemporaryDirectory() as tmp:
            n = write_scene_bins(scenes, tmp)
            metrics = reward.evaluate(tmp, n, ego_goals=ego_goals(scenes))

        rewards = torch.as_tensor(metrics["reward"], device=device)
        advantages = compute_advantages(rewards, ddpo_cfg)

        # ---- DDPO update over random-k steps ----------------------------------
        skipped_updates = 0
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
            if not torch.isfinite(loss):
                skipped_updates += 1
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                cfg.grad_clip,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                skipped_updates += 1
                continue
            opt.step()

        crit = float(rewards.gt(0).float().mean())
        inval = float(metrics["init_invalid"].mean())
        print(
            f"[it {it:04d}] reward={rewards.mean():.3f} critical_rate={crit:.3f} "
            f"init_invalid={inval:.3f} loss={log['loss']:.4f} "
            f"ratio={log.get('ratio_mean', 1.0):.3f} kl={log.get('kl_to_base', 0.0):.4f} "
            f"skipped_updates={skipped_updates}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "train/reward": float(rewards.mean()),
                    "train/critical_rate": crit,
                    "train/init_invalid": inval,
                    "train/ego_collision_rate": float(metrics["ego_collision"].mean()),
                    "train/ego_offroad_rate": float(metrics["ego_offroad"].mean()),
                    "train/reached_goal_rate": float(metrics["reached_goal"].mean()),
                    "train/loss": log["loss"],
                    "train/pg_loss": log.get("pg_loss", log["loss"]),
                    "train/ratio_mean": log.get("ratio_mean", 1.0),
                    "train/kl_to_base": log.get("kl_to_base", 0.0),
                    "train/adv_std": float(advantages.std(unbiased=False)),
                    "train/skipped_updates": skipped_updates,
                },
                step=it,
            )

        # ---- periodic held-out eval + trajectory viz --------------------------
        if eval_pool is not None and (it + 1) % eval_every == 0:
            ev, images = evaluate_and_visualize(model, eval_pool, reward, cfg, it, wandb)
            ev_crit = float((ev["ego_collision"] > 0).mean())
            ev_inval = float((ev["init_invalid"] > 0).mean())
            print(f"   [eval it {it:04d}] critical_rate={ev_crit:.3f} init_invalid={ev_inval:.3f}")
            if wandb is not None:
                wandb.log(
                    {
                        "val/critical_rate": ev_crit,
                        "val/init_invalid": ev_inval,
                        "val/ego_offroad_rate": float(ev["ego_offroad"].mean()),
                        "val/reached_goal_rate": float(ev["reached_goal"].mean()),
                        "val/rollouts": images,
                    },
                    step=it,
                )

        if cfg.save_every and (it + 1) % cfg.save_every == 0:
            tag = cfg.get("model_type", "dm_goal")
            torch.save(model.state_dict(), out_dir / f"{tag}_ddpo_{it + 1:05d}.pt")

    if wandb is not None:
        wandb.finish()


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
