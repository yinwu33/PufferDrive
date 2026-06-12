"""Encode generated scenes into PufferDrive ``.bin`` maps.

Bridges a ``GeneratedScenes`` (diffusion output, agent-only, fixed real map) into
the on-disk binary format consumed by ``load_map_binary`` in ``drive.h``. Reuses
the low-level writers from ``data_utils/selfplay/convert_selfplay_to_drive.py`` so
the byte layout stays in one place.

Agent state layout from dm_goal decode (physical units):
    [x, y, speed, cos_theta, sin_theta, length, width, goal_x, goal_y]
PufferDrive needs per-agent x, y, vx, vy, heading + dims + goal; only the t=0
state is meaningful (the planner rolls the future forward), so the trajectory is
filled with the initial state and marked valid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import struct
import torch

from data_utils.selfplay.convert_selfplay_to_drive import (
    TRAJECTORY_LENGTH,
    TYPE_ROAD_LANE,
    TYPE_VEHICLE,
    _pack_string_16,
    _write_entity_header,
    _write_float_array,
    _write_int_array,
)

# dm_goal agent type ids (argmax order) map onto PufferDrive entity types.
_TYPE_MAP = {0: 1, 1: 2, 2: 3}  # vehicle, pedestrian, cyclist


def _agent_rows_to_objects(states: np.ndarray, types: np.ndarray) -> list[dict]:
    objects = []
    for idx in range(states.shape[0]):
        s = states[idx]
        x, y, speed, cos_t, sin_t, length, width, goal_x, goal_y = s[:9]
        heading = float(np.arctan2(sin_t, cos_t))
        vx, vy = float(speed * cos_t), float(speed * sin_t)
        traj_x = np.full(TRAJECTORY_LENGTH, float(x), dtype=np.float32)
        traj_y = np.full(TRAJECTORY_LENGTH, float(y), dtype=np.float32)
        objects.append(
            {
                "id": idx,
                "type": _TYPE_MAP.get(int(types[idx]), TYPE_VEHICLE),
                "x": traj_x,
                "y": traj_y,
                "vx": np.full(TRAJECTORY_LENGTH, vx, dtype=np.float32),
                "vy": np.full(TRAJECTORY_LENGTH, vy, dtype=np.float32),
                "heading": np.full(TRAJECTORY_LENGTH, heading, dtype=np.float32),
                "valid": np.ones(TRAJECTORY_LENGTH, dtype=np.int32),
                "length": float(max(length, 0.5)),
                "width": float(max(width, 0.5)),
                "height": 1.6,
                "goal_x": float(goal_x),
                "goal_y": float(goal_y),
                "mark_as_expert": 0,
            }
        )
    return objects


def _lanes_to_roads(lane_polylines: np.ndarray) -> list[dict]:
    """lane_polylines: [n_lane, n_points, 2] in the scene frame."""
    roads = []
    for idx, poly in enumerate(lane_polylines):
        pts = [(float(p[0]), float(p[1])) for p in poly if np.isfinite(p[:2]).all()]
        if len(pts) >= 2:
            roads.append({"id": idx, "type": TYPE_ROAD_LANE, "geometry": pts})
    return roads


def _write_one(objects: list[dict], roads: list[dict], out_file: Path, map_id: int, ego_index: int = 0) -> None:
    with out_file.open("wb") as f:
        f.write(struct.pack("16s", _pack_string_16(f"gen_{map_id:05d}")))
        f.write(struct.pack("i", ego_index))      # sdc_track_index
        f.write(struct.pack("i", 1))              # num_tracks_to_predict
        f.write(struct.pack("i", ego_index))      # tracks_to_predict[0]
        f.write(struct.pack("i", len(objects)))   # num_objects
        f.write(struct.pack("i", len(roads)))     # num_roads
        for obj in objects:
            _write_entity_header(f, map_id, obj["type"], obj["id"], TRAJECTORY_LENGTH)
            _write_float_array(f, obj["x"])
            _write_float_array(f, obj["y"])
            _write_float_array(f, np.zeros(TRAJECTORY_LENGTH, dtype=np.float32))  # z
            _write_float_array(f, obj["vx"])
            _write_float_array(f, obj["vy"])
            _write_float_array(f, np.zeros(TRAJECTORY_LENGTH, dtype=np.float32))  # vz
            _write_float_array(f, obj["heading"])
            _write_int_array(f, obj["valid"])
            f.write(struct.pack("f", obj["width"]))
            f.write(struct.pack("f", obj["length"]))
            f.write(struct.pack("f", obj["height"]))
            f.write(struct.pack("f", obj["goal_x"]))
            f.write(struct.pack("f", obj["goal_y"]))
            f.write(struct.pack("f", 0.0))        # goal_z
            f.write(struct.pack("i", obj["mark_as_expert"]))
        for road in roads:
            geo = road["geometry"]
            _write_entity_header(f, map_id, road["type"], road["id"], len(geo))
            _write_float_array(f, np.asarray([p[0] for p in geo], dtype=np.float32))
            _write_float_array(f, np.asarray([p[1] for p in geo], dtype=np.float32))
            _write_float_array(f, np.zeros(len(geo), dtype=np.float32))
            for _ in range(6):
                f.write(struct.pack("f", 0.0))
            f.write(struct.pack("i", 0))


def ego_goals(scenes) -> np.ndarray:
    """Per-scene ego goal (x, y) in the scene frame, ``[num_scenes, 2]``.

    Ego is the first agent (local index 0) of each scene; its goal is at agent-state
    indices 7:9 (dm_goal). Used to window the reward to the pre-goal rollout.
    """
    states = scenes.agent_states.detach().cpu().numpy()
    sidx = scenes.agent_scene_idx.detach().cpu().numpy()
    goals = np.full((scenes.num_scenes, 2), np.nan, dtype=np.float32)
    for s in range(scenes.num_scenes):
        rows = np.nonzero(sidx == s)[0]
        if len(rows):
            goals[s] = states[rows[0], 7:9]
    return goals


def write_scene_bins(scenes, out_dir: str | Path) -> int:
    """Write one ``map_XXX.bin`` per generated scene. Returns the number written.

    ``scenes`` is a ``GeneratedScenes``. Ego is local index 0 in every scene.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.bin"):
        p.unlink()

    states = scenes.agent_states.detach().cpu().numpy()
    types = scenes.agent_types.detach().cpu().numpy()
    scene_idx = scenes.agent_scene_idx.detach().cpu().numpy()
    lanes = scenes.lane_polylines
    if isinstance(lanes, torch.Tensor):
        lanes = lanes.detach().cpu().numpy()

    # lane -> scene mapping must be supplied in meta (lane batch); fall back to
    # "same map for every scene" if a single shared map was used.
    lane_scene_idx = scenes.meta.get("lane_scene_idx")
    if lane_scene_idx is not None and isinstance(lane_scene_idx, torch.Tensor):
        lane_scene_idx = lane_scene_idx.detach().cpu().numpy()

    for s in range(scenes.num_scenes):
        a_mask = scene_idx == s
        objects = _agent_rows_to_objects(states[a_mask], types[a_mask])
        if lane_scene_idx is not None:
            roads = _lanes_to_roads(lanes[lane_scene_idx == s])
        else:
            roads = _lanes_to_roads(lanes)
        _write_one(objects, roads, out_dir / f"map_{s:03d}.bin", map_id=s)
    return scenes.num_scenes
