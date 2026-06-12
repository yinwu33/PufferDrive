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
        goal_radius: float = 2.0,
    ):
        self.planner_ckpt = planner_ckpt
        self.device = device
        self.sim_steps = sim_steps
        self.max_agents_per_scene = max_agents_per_scene
        self.policy_kwargs = policy_kwargs or {"input_size": 64, "hidden_size": 256}
        self.env_kwargs = env_kwargs or {}
        self.deterministic = deterministic
        self.reward_fn = reward_fn
        # Once the ego comes within this distance of its goal it has finished its
        # task; the planner's post-goal behaviour (respawn / off-map drift) is
        # undefined, so we stop scoring + recording that scene. Matches the env's
        # MIN_DISTANCE_TO_GOAL (drive.h).
        self.goal_radius = goal_radius
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
    def evaluate(self, map_dir: str, num_scenes: int, record_trajectories: bool = False,
                 ego_goals=None) -> dict:
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
        if ego_goals is not None:
            ego_goals = np.asarray(ego_goals, dtype=np.float32)

        obs, _ = env.reset()
        ego_collision = np.zeros(num_scenes, dtype=np.float32)
        ego_offroad_ = np.zeros(num_scenes, dtype=np.float32)
        init_invalid = np.zeros(num_scenes, dtype=np.float32)
        reached_goal = np.zeros(num_scenes, dtype=np.float32)
        # A scene is "finished" once its EGO reaches its goal (the env respawns it,
        # raising the ego's TERMINAL) or the ego's episode otherwise ends (TRUNCATION).
        # After that the planner's behaviour is undefined, so we stop both scoring and
        # trajectory recording for that scene. We key off the ego (local index 0) only
        # -- another agent finishing must not cut the ego's window short.
        finished = np.zeros(num_scenes, dtype=bool)

        # Per-scene rollout buffers (ego = local index 0) for visualisation.
        # "respawn" marks per-step which agents have teleported back to spawn after
        # reaching their goal, so viz can drop them instead of drawing a fake overlap.
        traj = [{"x": [], "y": [], "heading": [], "length": None, "width": None,
                 "done": [], "respawn": []}
                for _ in range(num_scenes)] if record_trajectories else None

        for t in range(self.sim_steps):
            st = env.get_global_agent_state()
            active_now = ~finished  # snapshot so state + done stay aligned per step
            for s in range(num_scenes):
                env_i = scene_env.get(s)
                if env_i is None or finished[s]:
                    continue
                lo, hi = offsets[env_i], offsets[env_i + 1]
                e = lo  # ego = first agent of the scene's env
                ego = {k: st[k][e] for k in ("x", "y", "heading", "length", "width")}
                # Exclude agents that have reached their goal and been teleported back
                # to their t=0 spawn pose (GOAL_RESPAWN). The sim itself drops these from
                # collision_check (respawn_timestep != -1), so counting their respawn-
                # teleport overlap with the ego would be a spurious collision.
                others_idx = [i for i in range(lo, hi) if i != e and not st["respawn"][i]]
                others = {k: st[k][others_idx] for k in ("x", "y", "heading", "length", "width")}
                collided = ego_collides(ego, others)
                offroad = ego_offroad(ego, road_edges[s])
                if t == 0 and (collided or offroad):
                    init_invalid[s] = 1.0
                ego_collision[s] = max(ego_collision[s], float(collided))
                ego_offroad_[s] = max(ego_offroad_[s], float(offroad))

                if traj is not None:
                    tr = traj[s]
                    tr["x"].append(st["x"][lo:hi].copy())
                    tr["y"].append(st["y"][lo:hi].copy())
                    tr["heading"].append(st["heading"][lo:hi].copy())
                    tr["respawn"].append(st["respawn"][lo:hi].copy())
                    if tr["length"] is None:
                        tr["length"] = st["length"][lo:hi].copy()
                        tr["width"] = st["width"][lo:hi].copy()

                # Fallback finish for non-respawn goal modes (e.g. GOAL_STOP, which
                # raises no terminal): stop if the ego is seen within goal_radius.
                if ego_goals is not None and np.isfinite(ego_goals[s]).all():
                    if np.hypot(ego["x"] - ego_goals[s, 0], ego["y"] - ego_goals[s, 1]) < self.goal_radius:
                        reached_goal[s] = 1.0
                        finished[s] = True

            ob = torch.as_tensor(obs).to(self.device)
            logits, _ = policy.forward_eval(ob, {})
            if self.deterministic:
                action = pufferlib.pytorch.deterministic_action(logits)
            else:
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(env.action_space.shape)
            obs, _, term, trunc, _ = env.step(action_np)

            terminals = np.asarray(term).astype(bool)   # ego terminal == reached its goal
            truncations = np.asarray(trunc).astype(bool)  # time-limit / episode reset
            for s in range(num_scenes):
                env_i = scene_env.get(s)
                if env_i is None or not active_now[s]:
                    continue
                lo = offsets[env_i]
                if terminals[lo]:
                    reached_goal[s] = 1.0  # env's authoritative ego-reached-goal signal
                ego_done = bool(terminals[lo] or truncations[lo])  # ego only
                if traj is not None:
                    traj[s]["done"].append(ego_done)
                if ego_done:
                    finished[s] = True

            if finished.all():
                break

        env.close()
        rewards = self.reward_fn(ego_collision, ego_offroad_, init_invalid)
        out = {
            "reward": rewards,
            "ego_collision": ego_collision,
            "ego_offroad": ego_offroad_,
            "init_invalid": init_invalid,
            "reached_goal": reached_goal,
        }
        if traj is not None:
            # Stack each scene's per-step slices into [T, n_agents] arrays.
            for tr in traj:
                tr["x"] = np.asarray(tr["x"], dtype=np.float32) if tr["x"] else np.zeros((0, 0), np.float32)
                tr["y"] = np.asarray(tr["y"], dtype=np.float32) if tr["y"] else np.zeros((0, 0), np.float32)
                tr["heading"] = np.asarray(tr["heading"], dtype=np.float32) if tr["heading"] else np.zeros((0, 0), np.float32)
                tr["respawn"] = np.asarray(tr["respawn"], dtype=bool) if tr["respawn"] else np.zeros((0, 0), bool)
                tr["done"] = np.asarray(tr["done"], dtype=bool)
            out["trajectories"] = traj
        return out

    def _road_edges_by_scene(self, env, num_scenes):
        """Per-scene list of road-edge polylines for the off-road test.

        ``get_road_edge_polylines`` returns a FLAT dict across all scenarios:
        ``x``/``y`` are point coords concatenated over every polyline, ``lengths``
        gives the point count per polyline, and ``scenario_id`` the owning scenario
        (16-char string) per polyline. We slice it back into polylines, group by
        scenario id, and map each scenario id to its scene index via the env (scene
        index == env map id; both sides share the ``gen_XXXXX`` scenario id written
        by scene_codec).
        """
        try:
            pl = env.get_road_edge_polylines()
        except Exception:
            return [[] for _ in range(num_scenes)]

        xs = np.asarray(pl["x"], dtype=np.float32)
        ys = np.asarray(pl["y"], dtype=np.float32)
        lengths = np.asarray(pl["lengths"], dtype=np.int64)
        sids = [str(s).rstrip("\x00") for s in pl["scenario_id"]]

        by_sid: dict[str, list] = {}
        off = 0
        for p, L in enumerate(lengths):
            L = int(L)
            pts = list(zip(xs[off : off + L].tolist(), ys[off : off + L].tolist()))
            off += L
            if len(pts) >= 2:
                by_sid.setdefault(sids[p], []).append(pts)

        # scene index s == env map id; look up that env's scenario id string.
        result = [[] for _ in range(num_scenes)]
        map_ids = np.asarray(env.map_ids)
        scenario_ids = env.scenario_ids
        seen = set()
        for env_i, m in enumerate(map_ids):
            m = int(m)
            if m < 0 or m >= num_scenes or m in seen:
                continue
            seen.add(m)
            result[m] = by_sid.get(scenario_ids[env_i], [])
        return result
