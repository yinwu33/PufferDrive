from __future__ import annotations

import argparse
import ast
import configparser
import glob
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from carla_interface.pufferdrive_model import PufferModule
from carla_interface.xodr_process import CACHE_ROOT, DEFAULT_XODR_DIR, regenerate_cached_map


os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON38 = "/home/tjhu78u/miniconda3/envs/py3.8torch2.4.1/bin/python"
DEFAULT_PUFFER_PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
DEFAULT_MODEL = "experiments/puffer_drive_177878959462.pt"
TOWN_ORDER = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD")

DT_S = 0.1


def puffer_value(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config(path):
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(path)
    if not parser.sections():
        raise FileNotFoundError(f"Could not read config: {path}")

    config = {}
    for section in parser.sections():
        config[section] = {key: puffer_value(value) for key, value in parser[section].items()}
    return config


def resolve_model_path(path):
    if path == "latest":
        candidates = glob.glob(str(REPO_ROOT / "experiments" / "puffer_drive*.pt"))
        if not candidates:
            raise FileNotFoundError("No puffer_drive*.pt checkpoints found in experiments/")
        return Path(max(candidates, key=os.path.getctime))

    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return model_path


def town_to_index(town):
    normalized = town.lower()
    for idx, name in enumerate(TOWN_ORDER):
        if name.lower() == normalized:
            return idx
    raise ValueError(f"Unknown town {town!r}. Known towns: {', '.join(TOWN_ORDER)}")


def load_agent_metadata(cache_json):
    with Path(cache_json).open("r") as f:
        data = json.load(f)

    metadata = {}
    for obj in data.get("objects", []):
        goal = obj.get("goalPosition")
        if goal is None:
            continue
        velocity = obj.get("velocity", [{}])[0]
        speed = math.hypot(float(velocity.get("x", 0.0)), float(velocity.get("y", 0.0)))
        metadata[int(obj["id"])] = {
            "goal": (
                float(goal.get("x", 0.0)),
                float(goal.get("y", 0.0)),
                float(goal.get("z", 0.0)),
            ),
            "initial_speed": speed,
        }
    return metadata


def build_drive_env(config, map_dir, args):
    from pufferlib.ocean.drive.drive import Drive

    env_config = dict(config.get("env", {}))
    env_config.update(
        map_dir=str(map_dir),
        num_maps=1,
        num_agents=args.num_agents,
        sample_mode="sequential",
        control_mode=args.control_mode,
        init_mode=args.init_mode,
        goal_behavior=args.goal_behavior,
        collision_behavior=1,
        offroad_behavior=1,
        episode_length=args.max_episode_steps + 1,
        termination_mode=0,
        render_mode=args.drive_render_mode,
    )
    return Drive(**env_config)


def connect_carla(args):
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    if args.load_world:
        world = client.load_world(args.map)
    else:
        world = client.get_world()

    previous_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT_S
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)
    return client, world, traffic_manager, previous_settings


def carla_transform(carla, world_map, x, y, z, heading, args):
    location = carla_location(carla, world_map, x, y, z, args)
    yaw = -math.degrees(float(heading)) if args.flip_y else math.degrees(float(heading))
    return carla.Transform(
        carla.Location(x=location.x, y=location.y, z=location.z + args.spawn_z_offset),
        carla.Rotation(yaw=yaw),
    )


def resolve_controlled_agent_index(states, metadata_by_id, requested_index):
    num_states = len(states["x"])
    if requested_index >= 0:
        if requested_index >= num_states:
            raise ValueError(f"controlled_agent_index={requested_index} out of range for {num_states} controlled agents")
        return requested_index

    best_index = None
    best_speed = -1.0
    for idx in range(num_states):
        metadata = metadata_by_id.get(int(states["id"][idx]))
        if metadata is None:
            continue
        if metadata["initial_speed"] > best_speed:
            best_index = idx
            best_speed = metadata["initial_speed"]
    if best_index is None:
        raise RuntimeError("Could not auto-select a controlled agent because no metadata matched active agent IDs")
    return best_index


def mirrored_state_indices(num_states, max_actors, controlled_agent_index):
    if controlled_agent_index < 0 or controlled_agent_index >= num_states:
        raise ValueError(f"controlled_agent_index={controlled_agent_index} out of range for {num_states} controlled agents")
    indices = [controlled_agent_index]
    for idx in range(num_states):
        if len(indices) >= max_actors:
            break
        if idx != controlled_agent_index:
            indices.append(idx)
    return indices


def spawn_mirrored_actors(world, states, max_actors, controlled_agent_index, rng, args):
    import carla

    blueprints = world.get_blueprint_library().filter("vehicle.*")
    preferred = [
        bp
        for bp in blueprints
        if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4
    ]
    if not preferred:
        raise RuntimeError("No four-wheel vehicle blueprints found in CARLA")
    actors = []
    indices = mirrored_state_indices(len(states["x"]), max_actors, controlled_agent_index)
    for actor_slot, state_idx in enumerate(indices):
        blueprint = rng.choice(preferred)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "sdc" if actor_slot == 0 else "pufferdrive_mirror")
        transform = carla_transform(
            carla,
            world.get_map(),
            states["x"][state_idx],
            states["y"][state_idx],
            states["z"][state_idx],
            states["heading"][state_idx],
            args,
        )
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            actors.append(actor)
    if len(actors) != len(indices):
        raise RuntimeError(f"Spawned {len(actors)}/{len(indices)} mirrored CARLA actors")
    return actors, indices


def spawn_traffic_manager_actors(world, traffic_manager, count, seed):
    rng = random.Random(seed)
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    if not blueprints:
        raise RuntimeError("No vehicle blueprints found in CARLA")
    spawn_points = list(world.get_map().get_spawn_points())
    if len(spawn_points) < count:
        raise RuntimeError(f"Requested {count} Traffic Manager actors but map only has {len(spawn_points)} spawn points")
    rng.shuffle(spawn_points)
    actors = []
    for transform in spawn_points[:count]:
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError(f"Failed to spawn Traffic Manager actor at {transform.location}")
        actor.set_autopilot(True, traffic_manager.get_port())
        actors.append(actor)
    return actors


def add_collision_sensor(world, actor, collision_events):
    blueprint = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(blueprint, actor.get_transform(), attach_to=actor)
    sensor.listen(lambda event: collision_events.append(event))
    return sensor


def sync_actors_to_drive(world, actors, mirrored_indices, states, args):
    import carla

    for actor, state_idx in zip(actors, mirrored_indices):
        if state_idx >= len(states["x"]) or not actor.is_alive:
            continue
        transform = carla_transform(
            carla,
            world.get_map(),
            states["x"][state_idx],
            states["y"][state_idx],
            states["z"][state_idx],
            states["heading"][state_idx],
            args,
        )
        actor.set_transform(transform)


def set_spectator_to_actor(world, actor, distance, height, pitch):
    import carla

    if actor is None or not actor.is_alive:
        return

    transform = actor.get_transform()
    yaw_rad = math.radians(transform.rotation.yaw)
    camera_location = carla.Location(
        x=transform.location.x - distance * math.cos(yaw_rad),
        y=transform.location.y - distance * math.sin(yaw_rad),
        z=transform.location.z + height,
    )
    camera_rotation = carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(camera_location, camera_rotation))


def carla_location(carla, world_map, x, y, z, args):
    carla_x = float(x)
    carla_y = -float(y) if args.flip_y else float(y)
    carla_z = float(z)
    if args.snap_to_road_z:
        waypoint = world_map.get_waypoint(carla.Location(x=carla_x, y=carla_y, z=2.0), project_to_road=True)
        if waypoint is not None:
            carla_z = waypoint.transform.location.z
    return carla.Location(x=carla_x, y=carla_y, z=carla_z)


def draw_goal_marker(world, goal_xyz, args):
    import carla

    base = carla_location(carla, world.get_map(), goal_xyz[0], goal_xyz[1], goal_xyz[2], args)
    base.z += 0.15
    top = carla.Location(base.x, base.y, base.z + args.goal_marker_height)
    color = carla.Color(0, 255, 0)
    world.debug.draw_point(base, size=args.goal_marker_size, color=color, life_time=args.goal_marker_life_time)
    world.debug.draw_line(base, top, thickness=args.goal_marker_thickness, color=color, life_time=args.goal_marker_life_time)
    world.debug.draw_string(top, "GOAL", draw_shadow=True, color=color, life_time=args.goal_marker_life_time)


def action_summary(action, dynamics_model):
    action_val = int(np.asarray(action).reshape(-1)[0])
    if dynamics_model == "classic":
        acceleration_values = [-4.0, -2.667, -1.333, -0.0, 1.333, 2.667, 4.0]
        steering_values = [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
        accel_idx = action_val // len(steering_values)
        steer_idx = action_val % len(steering_values)
        return f"{action_val} accel={acceleration_values[accel_idx]:.3f} steer={steering_values[steer_idx]:.3f}"
    if dynamics_model == "jerk":
        jerk_long = [-15.0, -4.0, 0.0, 4.0]
        jerk_lat = [-4.0, 0.0, 4.0]
        long_idx = action_val // len(jerk_lat)
        lat_idx = action_val % len(jerk_lat)
        return f"{action_val} jerk_long={jerk_long[long_idx]:.3f} jerk_lat={jerk_lat[lat_idx]:.3f}"
    return str(action_val)


def termination_reason(step, rewards, terminals, truncations, logs, collision_events, world, sdc_actor, goal_reward, check_carla_offroad):
    if collision_events:
        return "collision"
    if check_carla_offroad and sdc_actor is not None and sdc_actor.is_alive:
        waypoint = world.get_map().get_waypoint(sdc_actor.get_location(), project_to_road=False)
        if waypoint is None:
            return "off-road"

    if len(rewards) and rewards[0] >= goal_reward:
        return "success"
    if len(terminals) and terminals[0]:
        if logs:
            log = logs[0] if isinstance(logs, list) else logs
            if log.get("completion_rate", 0) > 0:
                return "success"
            if log.get("collision_rate", 0) > 0:
                return "collision"
            if log.get("offroad_rate", 0) > 0:
                return "off-road"
        return "terminated"
    if len(truncations) and truncations[0]:
        return "timeout"
    return None


def run_episode(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    model_path = resolve_model_path(args.load_model_path)
    map_dir, xodr_path, cache_json, cache_bin, restored_velocities, repaired_headings = regenerate_cached_map(args.map, args)
    metadata_by_id = load_agent_metadata(cache_json)
    print(f"Generated cache JSON from XODR {xodr_path} -> {cache_json}")
    if restored_velocities:
        print(f"Restored velocity[0] from later trajectory velocities for {restored_velocities} agents")
    if repaired_headings:
        print(f"Rebuilt replay headings from trajectory positions for {repaired_headings} frames")
    print(f"Generated Drive binary {cache_bin}")

    env = build_drive_env(config, map_dir, args)
    puffer_model = PufferModule(config, env, model_path, deterministic=args.deterministic)

    client, world, traffic_manager, previous_settings = connect_carla(args)
    print(f"Connected to CARLA {client.get_server_version()} on {world.get_map().name}")

    actors = []
    sensors = []
    tm_actors = []
    mirrored_indices = []
    reason = "timeout"
    try:
        observations, _ = env.reset(seed=args.seed)
        states = env.get_global_agent_state()
        controlled_idx = resolve_controlled_agent_index(states, metadata_by_id, args.controlled_agent_index)
        rng = random.Random(args.seed)
        actors, mirrored_indices = spawn_mirrored_actors(world, states, args.mirror_agents, controlled_idx, rng, args)
        if not actors:
            raise RuntimeError("Could not spawn the mirrored SDC actor in CARLA")
        controlled_actor_id = int(states["id"][controlled_idx])
        controlled_metadata = metadata_by_id.get(controlled_actor_id)
        if controlled_metadata is None:
            raise RuntimeError(f"No goalPosition found for controlled agent id={controlled_actor_id}")
        controlled_goal = controlled_metadata["goal"]
        start_x = float(states["x"][controlled_idx])
        start_y = float(states["y"][controlled_idx])
        last_x = start_x
        last_y = start_y
        goal_distance = math.hypot(controlled_goal[0] - start_x, controlled_goal[1] - start_y)
        print(
            f"Controlled agent index={controlled_idx} id={controlled_actor_id} "
            f"initial_speed={controlled_metadata['initial_speed']:.3f}m/s "
            f"goal_distance={goal_distance:.3f}m deterministic={args.deterministic}"
        )
        if args.deterministic and controlled_metadata["initial_speed"] < 0.05:
            print("Note: deterministic argmax starts from rest here and often selects near-neutral acceleration; omit --deterministic to sample actions.")
        tm_actors = spawn_traffic_manager_actors(world, traffic_manager, args.traffic_manager_agents, args.seed)
        collision_events = []
        sensors.append(add_collision_sensor(world, actors[0], collision_events))
        if args.follow_camera:
            set_spectator_to_actor(world, actors[0], args.camera_distance, args.camera_height, args.camera_pitch)
        if args.show_goal:
            draw_goal_marker(world, controlled_goal, args)
        world.tick()

        final_logs = []
        final_step = 0
        last_action = None
        max_displacement = 0.0
        last_displacement = 0.0
        for step in range(args.max_episode_steps):
            action = puffer_model.step(observations)
            last_action = action[controlled_idx]
            observations, rewards, terminals, truncations, info = env.step(action, per_env_logs=True)
            states = env.get_global_agent_state()
            last_x = float(states["x"][controlled_idx])
            last_y = float(states["y"][controlled_idx])
            final_step = step + 1
            last_displacement = math.hypot(last_x - start_x, last_y - start_y)
            max_displacement = max(max_displacement, last_displacement)
            sync_actors_to_drive(world, actors, mirrored_indices, states, args)
            if args.follow_camera:
                set_spectator_to_actor(world, actors[0], args.camera_distance, args.camera_height, args.camera_pitch)
            if args.show_goal:
                draw_goal_marker(world, controlled_goal, args)
            world.tick()

            if info:
                final_logs = info
            reason = termination_reason(
                step,
                rewards,
                terminals,
                truncations,
                final_logs,
                collision_events,
                world,
                actors[0],
                float(config.get("env", {}).get("reward_goal", 1.0)),
                args.check_carla_offroad,
            )
            if reason:
                print(f"Episode stopped at step={step + 1}: {reason}")
                break
            if args.log_motion:
                print(
                    f"step={step + 1} controlled_displacement={last_displacement:.3f}m "
                    f"action={action_summary(action[controlled_idx], env.dynamics_model)}"
                )
            if True:
                # sleep for better visualization.
                time.sleep(DT_S)
        else:
            print(f"Episode stopped at step={args.max_episode_steps}: timeout")

        if final_logs:
            print(f"Drive log: {final_logs[0] if isinstance(final_logs, list) else final_logs}")
        print(
            f"Controlled displacement: last={last_displacement:.3f}m max={max_displacement:.3f}m over {final_step} steps; "
            f"last_action={action_summary(last_action, env.dynamics_model) if last_action is not None else 'n/a'}"
        )
        print(f"CARLA actors: mirrored={len(actors)} traffic_manager={len(tm_actors)}")
        return reason
    finally:
        for sensor in sensors:
            sensor.stop()
        for actor in sensors + actors + tm_actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        world.apply_settings(previous_settings)
        traffic_manager.set_synchronous_mode(False)
        env.close()


def run_puffer_eval(args):
    model_path = resolve_model_path(args.load_model_path)
    map_dir, xodr_path, cache_json, cache_bin, restored_velocities, repaired_headings = regenerate_cached_map(args.map, args)
    print(f"Generated cache JSON from XODR {xodr_path} -> {cache_json}")
    if restored_velocities:
        print(f"Restored velocity[0] from later trajectory velocities for {restored_velocities} agents")
    if repaired_headings:
        print(f"Rebuilt replay headings from trajectory positions for {repaired_headings} frames")
    print(f"Generated Drive binary {cache_bin}")

    bin_files = sorted(Path(map_dir).glob("*.bin"))
    if bin_files != [Path(cache_bin)]:
        raise RuntimeError(f"Expected exactly one bin in {map_dir}, found: {bin_files}")

    puffer_python = Path(args.puffer_python)
    if not puffer_python.exists():
        raise FileNotFoundError(f"Missing puffer eval Python: {puffer_python}")

    scenario_mp4 = REPO_ROOT / f"{args.map}.mp4"
    output_mp4 = Path(args.output_mp4) if args.output_mp4 else CACHE_ROOT / args.map / "puffer_eval.mp4"
    if not output_mp4.is_absolute():
        output_mp4 = REPO_ROOT / output_mp4
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    if scenario_mp4.exists():
        scenario_mp4.unlink()
    if output_mp4.exists():
        output_mp4.unlink()

    cmd = [
        str(puffer_python),
        "-m",
        "pufferlib.pufferl",
        "eval",
        "puffer_drive",
        "--load-model-path",
        str(model_path),
        "--vec.backend",
        "PufferEnv",
        "--train.device",
        "cpu",
        "--env.map-dir",
        str(map_dir),
        "--eval.map-dir",
        str(map_dir),
        "--env.num-maps",
        "1",
        "--env.num-agents",
        str(args.num_agents),
        "--eval.sample-mode",
        "sequential",
        "--env.control-mode",
        args.control_mode,
        "--env.init-mode",
        args.init_mode,
        "--env.goal-behavior",
        str(args.goal_behavior),
        "--env.collision-behavior",
        "1",
        "--env.offroad-behavior",
        "1",
        "--env.episode-length",
        str(args.max_episode_steps + 1),
        "--env.render-mode",
        "1",
    ]
    if args.deterministic:
        cmd.extend(["--eval.deterministic", "True"])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    if not scenario_mp4.exists():
        raise RuntimeError(f"puffer eval completed but did not create expected mp4: {scenario_mp4}")
    shutil.move(str(scenario_mp4), str(output_mp4))
    print(f"Wrote {output_mp4}")
    return output_mp4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a PufferDrive checkpoint in a CARLA-backed visualization loop.")
    parser.add_argument("--mode", choices=("carla", "puffer-eval"), default="carla")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "ocean" / "drive.ini"))
    parser.add_argument("--load-model-path", default=DEFAULT_MODEL)
    parser.add_argument("--map", default="Town01", choices=TOWN_ORDER)
    parser.add_argument("--xodr-path", default=None, help="Exact OpenDRIVE .xodr input. Overrides --xodr-dir.")
    parser.add_argument("--xodr-dir", default=DEFAULT_XODR_DIR, help="Directory containing <Town>.xodr files.")
    parser.add_argument("--xodr-road-spacing", type=float, default=2.0)
    parser.add_argument("--xodr-spawn-spacing", type=float, default=2.0)
    parser.add_argument("--xodr-waypoint-spacing", type=float, default=0.5)
    parser.add_argument("--xodr-agent-speed", type=float, default=2.0)
    parser.add_argument("--xodr-num-objects", type=int, default=32)
    parser.add_argument("--xodr-min-start-distance", type=float, default=8.0)
    parser.add_argument("--xodr-min-goal-distance", type=float, default=2.0)
    parser.add_argument("--xodr-agent-length", type=float, default=4.5)
    parser.add_argument("--xodr-agent-width", type=float, default=2.0)
    parser.add_argument("--xodr-agent-height", type=float, default=1.8)
    parser.add_argument("--xodr-agent-z-offset", type=float, default=0.9)
    parser.add_argument("--restore-initial-velocity", action="store_true", default=True)
    parser.add_argument("--no-restore-initial-velocity", action="store_false", dest="restore_initial_velocity")
    parser.add_argument("--repair-replay-headings", action="store_true", default=True)
    parser.add_argument("--no-repair-replay-headings", action="store_false", dest="repair_replay_headings")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=91)
    parser.add_argument("--num-agents", type=int, default=32)
    parser.add_argument("--mirror-agents", type=int, default=16)
    parser.add_argument("--controlled-agent-index", type=int, default=-1, help="-1 selects the active agent with highest initial speed.")
    parser.add_argument("--traffic-manager-agents", type=int, default=16)
    parser.add_argument("--control-mode", default="control_agents")
    parser.add_argument("--init-mode", default="create_all_valid")
    parser.add_argument("--goal-behavior", type=int, default=2)
    parser.add_argument("--drive-render-mode", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--load-world", action="store_true", default=True, help="Load --map in CARLA before running.")
    parser.add_argument("--no-load-world", action="store_false", dest="load_world", help="Use the currently loaded CARLA map.")
    parser.add_argument("--check-carla-offroad", action="store_true", help="Also stop if CARLA says the mirrored SDC is off road.")
    parser.add_argument("--follow-camera", action="store_true", default=True, help="Keep the CARLA spectator behind the controlled car.")
    parser.add_argument("--no-follow-camera", action="store_false", dest="follow_camera", help="Leave the CARLA spectator unchanged.")
    parser.add_argument("--camera-distance", type=float, default=9.0)
    parser.add_argument("--camera-height", type=float, default=4.0)
    parser.add_argument("--camera-pitch", type=float, default=-15.0)
    parser.add_argument("--flip-y", action="store_true", default=False, help="Convert Drive coordinates to CARLA by negating y.")
    parser.add_argument("--no-flip-y", action="store_false", dest="flip_y")
    parser.add_argument("--snap-to-road-z", action="store_true", default=True, help="Use CARLA waypoint height for mirrored actors.")
    parser.add_argument("--no-snap-to-road-z", action="store_false", dest="snap_to_road_z")
    parser.add_argument("--spawn-z-offset", type=float, default=0.6)
    parser.add_argument("--show-goal", action="store_true", default=True)
    parser.add_argument("--no-show-goal", action="store_false", dest="show_goal")
    parser.add_argument("--goal-marker-height", type=float, default=4.0)
    parser.add_argument("--goal-marker-size", type=float, default=0.25)
    parser.add_argument("--goal-marker-thickness", type=float, default=0.08)
    parser.add_argument("--goal-marker-life-time", type=float, default=0.2)
    parser.add_argument("--log-motion", action="store_true")
    parser.add_argument("--puffer-python", default=DEFAULT_PUFFER_PYTHON)
    parser.add_argument("--output-mp4", default=None)
    parser.add_argument("--sleep", type=float, default=None, help="Wall-clock delay between ticks. Defaults to --dt; use 0 for fastest.")
    parser.add_argument(
        "--print-python",
        action="store_true",
        help="Print the recommended Python executable for CARLA 0.9.13 + torch on this machine.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.print_python:
        print(DEFAULT_PYTHON38)
        return 0
    if args.mode == "puffer-eval":
        run_puffer_eval(args)
        return 0
    reason = run_episode(args)
    return 0 if reason in {"success", "collision", "off-road", "timeout", "terminated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
