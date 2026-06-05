from __future__ import annotations

import argparse
import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np


TRAJECTORY_LENGTH = 91

TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 3
TYPE_ROAD_LANE = 4


def _as_points(points: np.ndarray, mask: np.ndarray | None = None) -> list[dict[str, float]]:
    if mask is not None:
        points = points[np.asarray(mask, dtype=bool)]
    clean = []
    for point in np.asarray(points):
        if len(point) < 2 or not np.isfinite(point[:2]).all():
            continue
        clean.append({"x": float(point[0]), "y": float(point[1]), "z": 0.0})
    return clean


def _agent_type(onehot: np.ndarray, fallback_name: str | None = None) -> int:
    if fallback_name == "pedestrian":
        return TYPE_PEDESTRIAN
    if fallback_name == "cyclist":
        return TYPE_CYCLIST
    if np.asarray(onehot).size >= 3:
        idx = int(np.argmax(onehot))
        return (TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST)[idx]
    return TYPE_VEHICLE


def _pack_string_16(value: str) -> bytes:
    return value.encode("utf-8", errors="ignore")[:16].ljust(16, b"\0")


def _write_entity_header(f, map_id: int, entity_type: int, entity_id: int, array_size: int) -> None:
    f.write(struct.pack("i", int(map_id)))
    f.write(struct.pack("i", int(entity_type)))
    f.write(struct.pack("i", int(entity_id)))
    f.write(struct.pack("i", int(array_size)))


def _write_float_array(f, values: np.ndarray) -> None:
    f.write(struct.pack(f"{len(values)}f", *[float(v) for v in values]))


def _write_int_array(f, values: np.ndarray) -> None:
    f.write(struct.pack(f"{len(values)}i", *[int(v) for v in values]))


def _shift_trajectory(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    traj = np.asarray(data.get("clipped_trajectory", data["local_trajectory"]), dtype=np.float32)
    valid = np.asarray(data.get("clipped_valid", data["trajectory_valid"]), dtype=bool)
    scene_timestep = int(data.get("scene_timestep", 0))
    shifted = np.zeros((traj.shape[0], TRAJECTORY_LENGTH, traj.shape[2]), dtype=np.float32)
    shifted_valid = np.zeros((traj.shape[0], TRAJECTORY_LENGTH), dtype=bool)

    future = traj[:, scene_timestep : scene_timestep + TRAJECTORY_LENGTH]
    future_valid = valid[:, scene_timestep : scene_timestep + TRAJECTORY_LENGTH]
    horizon = future.shape[1]
    shifted[:, :horizon] = future
    shifted_valid[:, :horizon] = future_valid

    if horizon < TRAJECTORY_LENGTH and horizon > 0:
        shifted[:, horizon:] = future[:, -1:, :]

    agent_states = np.asarray(data["agent_states"], dtype=np.float32)
    shifted[:, 0, :7] = agent_states[:, :7]
    shifted_valid[:, 0] = True
    return shifted, shifted_valid


def _make_objects(data: dict[str, Any]) -> list[dict[str, Any]]:
    traj, valid = _shift_trajectory(data)
    goals = np.asarray(data.get("clipped_final_states", data["raw_final_states"]), dtype=np.float32)
    goal_valid = np.asarray(data.get("clipped_final_valid", np.ones(traj.shape[0], dtype=bool)), dtype=bool)
    agent_ids = np.asarray(data["agent_ids"])
    agent_types = np.asarray(data["agent_types"])
    agent_type_names = data.get("agent_type_names", [None] * traj.shape[0])

    objects = []
    for idx in range(int(data["num_agents"])):
        state = traj[idx]
        goal = goals[idx] if idx < len(goals) and goal_valid[idx] else state[-1]
        objects.append(
            {
                "id": int(agent_ids[idx]),
                "type": _agent_type(agent_types[idx], agent_type_names[idx]),
                "x": state[:, 0],
                "y": state[:, 1],
                "vx": state[:, 2],
                "vy": state[:, 3],
                "heading": state[:, 4],
                "length": float(max(state[0, 5], 0.1)),
                "width": float(max(state[0, 6], 0.1)),
                "height": 1.6,
                "valid": valid[idx],
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "mark_as_expert": 0,
            }
        )
    return objects


def _make_roads(data: dict[str, Any]) -> list[dict[str, Any]]:
    roads: list[dict[str, Any]] = []
    for idx, points in enumerate(np.asarray(data["road_points"])):
        mask = np.asarray(data["road_point_masks"])[idx]
        geometry = _as_points(points, mask)
        if len(geometry) >= 2:
            roads.append({"id": idx, "type": TYPE_ROAD_LANE, "geometry": geometry})

    return roads


def _write_binary(data: dict[str, Any], output_file: Path, map_id: int) -> None:
    objects = _make_objects(data)
    roads = _make_roads(data)
    ego_index = int(data.get("ego_index", 0))
    scenario_id = str(data.get("scenario_id", f"selfplay_{map_id:03d}"))

    with output_file.open("wb") as f:
        f.write(struct.pack("16s", _pack_string_16(scenario_id)))
        f.write(struct.pack("i", ego_index))
        f.write(struct.pack("i", 1))
        f.write(struct.pack("i", ego_index))
        f.write(struct.pack("i", len(objects)))
        f.write(struct.pack("i", len(roads)))

        for obj in objects:
            _write_entity_header(f, map_id, obj["type"], obj["id"], TRAJECTORY_LENGTH)
            _write_float_array(f, obj["x"])
            _write_float_array(f, obj["y"])
            _write_float_array(f, np.zeros(TRAJECTORY_LENGTH, dtype=np.float32))
            _write_float_array(f, obj["vx"])
            _write_float_array(f, obj["vy"])
            _write_float_array(f, np.zeros(TRAJECTORY_LENGTH, dtype=np.float32))
            _write_float_array(f, obj["heading"])
            _write_int_array(f, obj["valid"].astype(np.int32))
            f.write(struct.pack("f", obj["width"]))
            f.write(struct.pack("f", obj["length"]))
            f.write(struct.pack("f", obj["height"]))
            f.write(struct.pack("f", obj["goal_x"]))
            f.write(struct.pack("f", obj["goal_y"]))
            f.write(struct.pack("f", 0.0))
            f.write(struct.pack("i", obj["mark_as_expert"]))

        for road in roads:
            geometry = road["geometry"]
            _write_entity_header(f, map_id, road["type"], road["id"], len(geometry))
            for coord in ("x", "y", "z"):
                _write_float_array(f, np.asarray([point[coord] for point in geometry], dtype=np.float32))
            f.write(struct.pack("f", 0.0))
            f.write(struct.pack("f", 0.0))
            f.write(struct.pack("f", 0.0))
            f.write(struct.pack("f", 0.0))
            f.write(struct.pack("f", 0.0))
            f.write(struct.pack("f", 0.0))
            f.write(struct.pack("i", 0))


def convert_directory(input_dir: Path, output_dir: Path, limit: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("*.pkl"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    for map_id, path in enumerate(paths):
        with path.open("rb") as f:
            data = pickle.load(f)
        _write_binary(data, output_dir / f"map_{map_id:03d}.bin", map_id)
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert self-play pkl scenarios to PufferDrive binary maps.")
    parser.add_argument("--input-dir", type=Path, default=Path("data_selfplay/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("resources/drive/binaries/selfplay_train"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    count = convert_directory(args.input_dir, args.output_dir, args.limit)
    print(f"Converted {count} self-play maps to {args.output_dir}")


if __name__ == "__main__":
    main()
