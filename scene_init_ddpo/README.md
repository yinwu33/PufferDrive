# scene_init_ddpo

A PufferDrive-hosted **scenario-RL platform**: train scene-initialisation
generative models with **DDPO** so they produce scenes that are *critical* for a
frozen PufferDrive planner (high ego-collision rate), while staying on the
realistic-scene manifold.

Today the only plugged-in generator is scenario-dreamer's **dm_goal** diffusion
model (vendored). Future scene-init models only need to implement
`interfaces.SceneInitModel` to reuse the same DDPO trainer and reward.

## Why it lives in PufferDrive (one venv, in-process)

The diffusion policy and the simulator reward run in the **same process and venv**.
scenario-dreamer's dm_goal **inference closure** (the `nn.Module`s only — no
Lightning / dataset / viz) is vendored into `sd_model/`. Its runtime deps
(`torch_geometric` core, `torch_ema`) install cleanly on PufferDrive's
torch 2.10 / py3.12 venv — verified, no compiled PyG companions needed. This
avoids the cross-venv IPC that a scenario-dreamer-hosted design would have forced
(SD is py3.10 / torch 2.2, incompatible).

## DDPO as applied here

The denoising chain is the MDP (Black et al. 2023):

| MDP piece | dm_goal |
|---|---|
| horizon H | `n_diffusion_timesteps` = 100 |
| state `s_t` | noisy agent latents `x_t` + fixed conditioning graph |
| action `a_t` | sampled `x_{t-1}` |
| policy `π_θ` | `N(guided_posterior_mean_θ(x_t,t), Σ_t)`, `Σ_t` fixed |
| reward | ego collision over a frozen-planner rollout |

Project decisions baked in:
- **Lane/map chain is fixed** (lane-conditioned mode) → policy only acts on agents.
- **Ego = local index 0 per scene is held fixed** via inpainting every step → DDPO
  only perturbs the *other* agents to be adversarial; ego never enters the policy.
- **Only the ego is scored.** Reward = `+1` if the planner makes the ego collide,
  `-1` if the scene is degenerate (ego already overlapping at t=0, i.e. reward
  hacking), else `0`. See `reward.default_reward_fn`.
- **Random-k step gradient**: each update differentiates through only `k≈8` of the
  ~100 denoising steps (the rest of the trajectory is replayed from records) to
  keep the backward pass affordable. The deterministic final step (t=0) is excluded.
- **KL-to-base** (`ddpo.kl_coef>0`) regularises against the frozen pretrained model
  to keep scenes on-manifold. Off by default; turn on once PG is stable.

## Data flow per iteration

```
ConditioningPool.sample_batch(B)                 # real maps + real ego (HeteroData)
        │
DMGoalSceneInitModel.sample(cond)                # record denoising trajectory + old_logprob
        │  → GeneratedScenes (agents+goals, physical units) + SamplingTrajectory
scene_codec.write_scene_bins(scenes, tmp)        # → PufferDrive .bin (load_map_binary fmt)
        │
PufferDriveReward.evaluate(tmp, B)               # frozen planner rollout, ego metrics
        │  → reward[B], ego_collision, ego_offroad, init_invalid
ddpo.compute_advantages(reward)                  # per-scene whitened advantage
        │
for inner_epochs:                                # pick k random denoising steps
    new_lp = model.trajectory_logprob(traj, cond, k_idx)        # with grad (policy)
    ref_lp = model.trajectory_logprob(..., use_reference=True)  # frozen base (KL)
    loss   = ddpo.ddpo_loss(new_lp, old_lp, adv, cfg, ref_lp)
    loss.backward(); opt.step()
```

## Layout

```
scene_init_ddpo/
  interfaces.py        # SceneInitModel ABC + GeneratedScenes / SamplingTrajectory
  scene_models/
    dm_goal.py         # dm_goal adapter: ckpt load, sample+logprob, random-k logprob, KL ref
  sd_model/            # VENDORED scenario-dreamer dm_goal inference closure (nn.Module only)
  conditioning.py      # load + batch dumped HeteroData conditioning graphs
  scene_codec.py       # GeneratedScenes -> PufferDrive .bin
  geometry.py          # numpy SAT collision / road-edge crossing (ego metrics)
  reward.py            # in-process Drive rollout w/ frozen planner -> ego metrics
  ddpo.py              # advantages + PG loss (REINFORCE / PPO-clip) + KL term
  train.py             # main loop
  config/ddpo_dm_goal.yaml
  tools/dump_conditioning.py   # RUN IN SD VENV: export conditioning pool + model_cfg
```

## Running

1. **Dump conditioning** (scenario-dreamer venv, one-time):
   ```bash
   cd /home/.../scenario-dreamer
   .venv/bin/python /path/to/PufferDrive/scene_init_ddpo/tools/dump_conditioning.py \
       --num-scenes 2000 --out /path/to/PufferDrive/scene_init_ddpo/data/cond_pool
   ```
2. **Fill `config/ddpo_dm_goal.yaml`**: `model_ckpt` (dm_goal .ckpt), `model_cfg`
   (`.../cond_pool/model_cfg.yaml`), `conditioning_pool` (`.../cond_pool`),
   `planner_ckpt` (frozen PufferDrive planner .pt).
3. **Train** (PufferDrive venv):
   ```bash
   .venv/bin/python -m scene_init_ddpo.train --config scene_init_ddpo/config/ddpo_dm_goal.yaml
   ```

## Status / integration points to confirm on first run

These are the spots that depend on live artifacts and should be checked against a
real checkpoint + map (they are isolated and commented in-code):

- `scene_models/dm_goal.py` `_load_checkpoint`: exact Lightning key prefixes
  (`diff_model.*`, `ema.shadow_params*`) for the specific checkpoint.
- `reward.py` `_build_env` / `_build_policy`: env kwargs and `pufferlib.pacific.torch.Drive`
  policy kwargs must match the planner checkpoint that was trained.
- `reward.py` `_road_edges_by_scene`: how `get_road_edge_polylines()` splits by scene.
- `conditioning.py`: the dumped graphs must carry every field `DiT.forward` reads
  (`condition`, `lg_type`, `map_id`, `num_agents`, `num_lanes`, the 3 edge sets).
- `scene_codec.py`: if conditioning batches >1 distinct map, pass `lane_scene_idx`
  in `GeneratedScenes.meta` so lanes are written per scene.
```
