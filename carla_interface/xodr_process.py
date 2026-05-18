from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XODR_DIR = "/home/tjhu78u/carlaCache/0.9.13-2-g0c41f167c-dirty/Carla/Maps/OpenDrive"
CACHE_ROOT = REPO_ROOT / "carla_interface" / "cache"


def restore_initial_velocities(map_data, min_speed=0.05):
    restored = 0
    for obj in map_data.get("objects", []):
        velocities = obj.get("velocity", [])
        if not velocities:
            continue
        first = velocities[0]
        first_speed = math.hypot(float(first.get("x", 0.0)), float(first.get("y", 0.0)))
        if first_speed >= min_speed:
            continue

        replacement = None
        for velocity in velocities[1:]:
            speed = math.hypot(float(velocity.get("x", 0.0)), float(velocity.get("y", 0.0)))
            if speed >= min_speed:
                replacement = {"x": float(velocity.get("x", 0.0)), "y": float(velocity.get("y", 0.0))}
                break
        if replacement is None:
            raise ValueError(f"Object id={obj.get('id')} has no nonzero velocity to restore velocity[0]")
        velocities[0] = replacement
        restored += 1
    return restored


def position_xy(position):
    if isinstance(position, dict):
        return float(position.get("x", 0.0)), float(position.get("y", 0.0))
    return float(position[0]), float(position[1])


def angle_delta(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def derive_heading_from_positions(positions, idx, min_distance):
    if idx >= len(positions):
        return None

    x, y = position_xy(positions[idx])
    for next_idx in range(idx + 1, len(positions)):
        next_x, next_y = position_xy(positions[next_idx])
        dx = next_x - x
        dy = next_y - y
        if math.hypot(dx, dy) >= min_distance:
            return math.atan2(dy, dx)

    for prev_idx in range(idx - 1, -1, -1):
        prev_x, prev_y = position_xy(positions[prev_idx])
        dx = x - prev_x
        dy = y - prev_y
        if math.hypot(dx, dy) >= min_distance:
            return math.atan2(dy, dx)

    return None


def repair_replay_headings(map_data, min_distance=1e-3):
    repaired = 0
    for obj in map_data.get("objects", []):
        positions = obj.get("position", [])
        headings = obj.get("heading", [])
        if not positions or not headings:
            continue

        for idx in range(min(len(positions), len(headings))):
            derived = derive_heading_from_positions(positions, idx, min_distance)
            if derived is None:
                continue
            if abs(angle_delta(float(headings[idx]), derived)) > 1e-6:
                headings[idx] = derived
                repaired += 1
    return repaired


def resolve_xodr_path(town, xodr_path, xodr_dir):
    if xodr_path:
        path = Path(xodr_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
    else:
        path = Path(xodr_dir)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path / f"{town}.xodr"

    if not path.exists():
        raise FileNotFoundError(f"Missing XODR input: {path}. Pass --xodr-path for this map.")
    return path


def build_drive_json_from_xodr(town, xodr_path, args):
    from data_utils.carla.generate_carla_agents import generate_drive_json_from_xodr

    return generate_drive_json_from_xodr(
        xodr_path,
        town,
        num_objects=args.xodr_num_objects,
        seed=args.seed,
        dt=args.dt,
        road_spacing=args.xodr_road_spacing,
        spawn_spacing=args.xodr_spawn_spacing,
        waypoint_spacing=args.xodr_waypoint_spacing,
        agent_speed=args.xodr_agent_speed,
        min_start_distance=args.xodr_min_start_distance,
        min_goal_distance=args.xodr_min_goal_distance,
        agent_length=args.xodr_agent_length,
        agent_width=args.xodr_agent_width,
        agent_height=args.xodr_agent_height,
        agent_z_offset=args.xodr_agent_z_offset,
    )


def regenerate_cached_map(town, args):
    from pufferlib.ocean.drive.drive import load_map

    xodr_path = resolve_xodr_path(town, args.xodr_path, args.xodr_dir)
    map_data = build_drive_json_from_xodr(town, xodr_path, args)
    restored_velocities = restore_initial_velocities(map_data) if args.restore_initial_velocity else 0
    repaired_headings = repair_replay_headings(map_data) if args.repair_replay_headings else 0

    cache_dir = CACHE_ROOT / town
    cache_json = cache_dir / f"{town}.json"
    cache_bin = cache_dir / "map_000.bin"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_json.exists():
        cache_json.unlink()
    if cache_bin.exists():
        cache_bin.unlink()

    with cache_json.open("w") as f:
        json.dump(map_data, f, indent=2)
    load_map(str(cache_json), 0, str(cache_bin))
    if not cache_bin.exists():
        raise RuntimeError(f"Failed to generate Drive binary: {cache_bin}")

    return cache_dir, xodr_path, cache_json, cache_bin, restored_velocities, repaired_headings
