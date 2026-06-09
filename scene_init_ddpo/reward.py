"""In-process PufferDrive reward oracle for DDPO.

Loads a batch of generated scenes (``.bin`` maps), rolls them out with the frozen
planner, and returns per-scene ego metrics. Runs in the SAME process/venv as the
diffusion policy (the whole point of vendoring): the reward is just a function
call, no IPC.

Reward signal (project decisions):
  * only the ego (local index 0 per scene) is scored;
  * ``init_invalid`` flags scenes where the ego already overlaps another agent at
    t=0 (before the planner acts) -> these are reward-hacking degenerate scenes and
    are penalised, not rewarded;
  * ``ego_collision`` / ``ego_offroad`` are read over the planner rollout.

The default ``reward_fn`` turns these into the DDPO scalar; tune freely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .geometry import ego_collides, ego_offroad


def default_reward_fn(ego_collision, ego_offroad_, init_invalid):
    """Critical-scene reward: +1 if the planner collides, but kill degenerate
    (already-overlapping-at-t0) scenes with a strong negative so the diffusion
    cannot reward-hack by spawning agents on top of the ego."""
    r = np.where(ego_collision > 0, 1.0, 0.0)
    r = np.where(init_invalid > 0, -1.0, r)
    return r.astype(np.float32)


class PufferDriveReward:
    def __init__(
        self,
        planner_ckpt: str,
        *,
        device: str = "cuda",
        sim_steps: int = 91,
        max_agents_per_scene: int = 32,
        policy_kwargs: dict | None = None,
        env_kwargs: dict | None = None,
        deterministic: bool = True,
        reward_fn=default_reward_fn,
    ):
        self.planner_ckpt = planner_ckpt
        self.device = device
        self.sim_steps = sim_steps
        self.max_agents_per_scene = max_agents_per_scene
        self.policy_kwargs = policy_kwargs or {"input_size": 64, "hidden_size": 256}
        self.env_kwargs = env_kwargs or {}
        self.deterministic = deterministic
        self.reward_fn = reward_fn
        self._policy = None  # built lazily once an env exists (needs obs space)

    # ------------------------------------------------------------------ build
    def _build_env(self, map_dir: str, num_scenes: int):
        from pufferlib.pacific.drive.drive import Drive  # lazy: pulls raylib/binding

        kwargs = dict(
            map_dir=str(map_dir),
            num_maps=num_scenes,
            num_agents=self.max_agents_per_scene * num_scenes,
            max_controlled_agents=self.max_agents_per_scene,
            render_mode=1,  # headless
            init_steps=0,
            control_mode="control_agents",
            init_mode="create_all_valid",
            action_type="discrete",
            dynamics_model="classic",
            collision_behavior=0,
            offroad_behavior=0,
            episode_length=self.sim_steps,
        )
        kwargs.update(self.env_kwargs)
        return Drive(**kwargs)

    def _build_policy(self, env):
        import pufferlib.pacific.torch as ptorch

        policy = ptorch.Drive(env, **self.policy_kwargs).to(self.device)
        sd = torch.load(self.planner_ckpt, map_location=self.device)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        policy.load_state_dict(sd)
        policy.eval()
        return policy

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, map_dir: str, num_scenes: int) -> dict:
        import pufferlib

        env = self._build_env(map_dir, num_scenes)
        policy = self._build_policy(env)
        offsets = np.asarray(env.agent_offsets)
        map_ids = np.asarray(env.map_ids)

        # The env packs a fixed agent budget across many env copies that *cycle*
        # the maps, so num_envs may exceed num_scenes. Score each scene once using
        # the first env that hosts its map; ego = that env's first agent.
        scene_env = {}
        for env_i, m in enumerate(map_ids):
            scene_env.setdefault(int(m), env_i)
        road_edges = self._road_edges_by_scene(env, num_scenes)

        obs, _ = env.reset()
        ego_collision = np.zeros(num_scenes, dtype=np.float32)
        ego_offroad_ = np.zeros(num_scenes, dtype=np.float32)
        init_invalid = np.zeros(num_scenes, dtype=np.float32)

        for t in range(self.sim_steps):
            st = env.get_global_agent_state()
            for s in range(num_scenes):
                env_i = scene_env.get(s)
                if env_i is None:
                    continue
                lo, hi = offsets[env_i], offsets[env_i + 1]
                e = lo  # ego = first agent of the scene's env
                ego = {k: st[k][e] for k in ("x", "y", "heading", "length", "width")}
                others_idx = [i for i in range(lo, hi) if i != e]
                others = {k: st[k][others_idx] for k in ("x", "y", "heading", "length", "width")}
                collided = ego_collides(ego, others)
                offroad = ego_offroad(ego, road_edges[s])
                if t == 0 and (collided or offroad):
                    init_invalid[s] = 1.0
                ego_collision[s] = max(ego_collision[s], float(collided))
                ego_offroad_[s] = max(ego_offroad_[s], float(offroad))

            ob = torch.as_tensor(obs).to(self.device)
            logits, _ = policy.forward_eval(ob, {})
            if self.deterministic:
                action = pufferlib.pytorch.deterministic_action(logits)
            else:
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(env.action_space.shape)
            obs, _, _, _, _ = env.step(action_np)

        env.close()
        rewards = self.reward_fn(ego_collision, ego_offroad_, init_invalid)
        return {
            "reward": rewards,
            "ego_collision": ego_collision,
            "ego_offroad": ego_offroad_,
            "init_invalid": init_invalid,
        }

    def _road_edges_by_scene(self, env, num_scenes):
        """Per-scene list of road-edge polylines for the off-road test."""
        try:
            polylines = env.get_road_edge_polylines()
        except Exception:
            return [[] for _ in range(num_scenes)]
        # get_road_edge_polylines returns geometry keyed/indexed by env; if it does
        # not split by scene, fall back to sharing all edges (conservative).
        if isinstance(polylines, dict) and num_scenes in (len(polylines), ):
            return [polylines[i] for i in range(num_scenes)]
        return [polylines for _ in range(num_scenes)]
