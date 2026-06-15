from __future__ import annotations

import argparse
import ast
import configparser
import glob
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from carla_interface.pufferdrive_model import PufferModule
from waymax import config as waymax_config
from waymax import datatypes
from waymax import dynamics
from waymax.dataloader import womd_dataloader, womd_factories, womd_utils
from waymax.env import base_environment
from waymax.utils import test_utils
from waymax.visualization import viz
from waymax_interface.tfrecord_preprocess import (
    MAX_NUM_OBJECTS,
    MAX_NUM_RG_POINTS,
    PreprocessError,
    preprocess_scenario_to_tfexample,
    preprocess_serialized_scenario_to_tfexample,
)
from waymax_interface.json_preprocess import preprocess_json_to_tfexample
from waymax_interface.bin_preprocess import preprocess_bin_to_tfexample

# TODO:
from waymax_interface.viz_pufferdrive import VizPufferDrive


viz_model_tool  = VizPufferDrive()

DATAROOT_DIR = Path("/mnt/disk/data/public/waymo/motion_v_1_3_1/scenario")
DEFAULT_CACHE_PATH = REPO_ROOT / "waymax_interface" / "cache"
DEFAULT_TFEXAMPLE_CACHE_DIR = DEFAULT_CACHE_PATH / "tfexample"
DEFAULT_VIDEO_CACHE_DIR = DEFAULT_CACHE_PATH / "video"
REQUIRED_WAYMAX_FEATURES = frozenset(
    ("roadgraph_samples/dir", "state/current/x", "state/future/x", "state/past/x")
)

INIT_STEPS = 11
NUM_STEPS = 80
RENDER_FPS = 10
PARTNER_FEATURES = 7
ROAD_FEATURES = 7

# Per-policy specs. "pufferdrive" and "selfplay_drive" are *different* models with
# different observation layouts, so each carries its own config, checkpoint and
# obs geometry (these mirror the C constants in the respective drive.h):
#   - pufferdrive   : ocean puffer_drive, classic ego = 8 base features (no
#                     conditioning), MAX_AGENTS=32, 128 road segments, 50m range.
#   - selfplay_drive: pacific selfplay_drive, classic ego = 8 base + 3 conditioning
#                     features (collision_factor, offroad_factor, lane_width),
#                     MAX_AGENTS=64, 512 road segments, 64m range (full 64x64 map).
POLICY_SPECS: dict[str, dict[str, Any]] = {
    "pufferdrive": {
        "config_path": REPO_ROOT / "config" / "ocean" / "drive.ini",
        "default_model": REPO_ROOT / "experiments" / "puffer_drive.pt",
        "model_glob": "*puffer_drive*.pt",
        "ego_features": 8,
        "has_conditioning": False,
        "max_partner_objects": 31,
        "max_road_objects": 128,
        "obs_range_sq": 2500.0,  # 50m
    },
    "selfplay_drive": {
        "config_path": REPO_ROOT / "config" / "pacific" / "selfplay_drive.ini",
        "default_model": (
            REPO_ROOT
            / "experiments"
            / "selfplay_drive_178125169286"
            / "model_selfplay_drive_001000.pt"
        ),
        "model_glob": "*selfplay_drive*.pt",
        "ego_features": 11,
        "has_conditioning": True,
        "max_partner_objects": 63,
        "max_road_objects": 512,
        "obs_range_sq": 4096.0,  # 64m: full 64x64 map
    },
}
PUFFER_POLICIES = tuple(POLICY_SPECS)
MAX_SPEED = 100.0
MAX_VEH_WIDTH = 15.0
MAX_VEH_LEN = 30.0
MAX_ROAD_SEGMENT_LENGTH = 100.0
MAX_ROAD_SCALE = 100.0
TIME_INTERVAL = 0.1
# Conditioning inputs for the selfplay_drive model. These scale the collision /
# offroad penalties the policy was trained against; higher = more cautious. The
# values are fed into the ego observation raw (un-normalized), matching drive.h.
DEFAULT_COLLISION_FACTOR = 2.0
DEFAULT_OFFROAD_FACTOR = 2.0
DEFAULT_LANE_WIDTH = 3.5
ACCELERATION_VALUES = (-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0)
STEERING_VALUES = (
    -1.0,
    -0.833,
    -0.667,
    -0.5,
    -0.333,
    -0.167,
    0.0,
    0.167,
    0.333,
    0.5,
    0.667,
    0.833,
    1.0,
)
WAYMAX_METRICS_TO_RUN = (
    "log_divergence",
    "overlap",
    "offroad",
    "kinematic_infeasibility",
)


class UserInputError(RuntimeError):
    pass


def puffer_value(value: str) -> Any:
    """Parse a config string with ast.literal_eval, falling back to the raw string."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_puffer_config(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load a PufferDrive .ini config file and return a nested dict of parsed values."""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(path)
    if not parser.sections():
        raise UserInputError(f"Could not read PufferDrive config: {path}")
    return {
        section: {key: puffer_value(value) for key, value in parser[section].items()}
        for section in parser.sections()
    }


def resolve_puffer_model(path: str, model_glob: str = "puffer_drive*.pt") -> Path:
    """Resolve a PufferDrive checkpoint path, or find the newest one for 'latest'."""
    if path == "latest":
        candidates = glob.glob(str(REPO_ROOT / "experiments" / "**" / model_glob), recursive=True)
        if not candidates:
            raise UserInputError(
                f"No {model_glob} checkpoints found under experiments/"
            )
        return Path(max(candidates, key=os.path.getctime))

    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        raise UserInputError(f"PufferDrive checkpoint not found: {model_path}")
    return model_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Waymax runner."""
    parser = argparse.ArgumentParser(
        description="Run a small Waymax rollout on WOMD TFRecord scenarios."
    )
    parser.add_argument("--data-root", type=Path, default=DATAROOT_DIR)
    parser.add_argument(
        "--split", default="validation", choices=("training", "validation", "testing")
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run Waymax on a generated zero scenario instead of a TFRecord.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Run Waymax on a ScenarioMax/PufferDrive JSON scenario.",
    )
    parser.add_argument(
        "--bin",
        type=Path,
        default=None,
        help="Run Waymax on a PufferDrive .bin map (selfplay_drive training format).",
    )
    parser.add_argument(
        "-f",
        "--file-index",
        type=int,
        default=0,
        help="Sorted TFRecord index inside --data-root/--split.",
    )
    parser.add_argument(
        "-s",
        "--scenario-index",
        type=int,
        default=0,
        help="Index to consume from the selected TFRecord iterator.",
    )
    parser.add_argument(
        "--policy",
        choices=("expert", "constant_speed", "zero", "pufferdrive", "selfplay_drive"),
        default="pufferdrive",
        help="SDC controller. 'pufferdrive' (ocean) and 'selfplay_drive' (pacific) "
        "are different models with different observation layouts.",
    )
    parser.add_argument(
        "--load-model-path",
        type=str,
        default=None,
        help="Checkpoint path (or 'latest') for the puffer policy. Defaults to the "
        "selected policy's built-in model.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic action selection for puffer policies.",
    )
    parser.add_argument(
        "--collision-factor",
        type=float,
        default=DEFAULT_COLLISION_FACTOR,
        help="selfplay_drive conditioning input: collision penalty scale "
        "(higher = more cautious). Fed into the ego observation.",
    )
    parser.add_argument(
        "--offroad-factor",
        type=float,
        default=DEFAULT_OFFROAD_FACTOR,
        help="selfplay_drive conditioning input: offroad penalty scale "
        "(higher = more cautious). Fed into the ego observation.",
    )
    parser.add_argument(
        "--lane-width",
        type=float,
        default=DEFAULT_LANE_WIDTH,
        help="selfplay_drive conditioning input: lane width (m, 1.0-5.0) used as "
        "the offroad half-width tolerance. Fed into the ego observation.",
    )
    parser.add_argument(
        "--controlled-object", choices=("SDC", "MODELED", "VALID"), default="SDC"
    )
    parser.add_argument("--compute-reward", action="store_true")
    parser.add_argument(
        "--no-compute-reward", action="store_false", dest="compute_reward"
    )
    parser.set_defaults(compute_reward=False)
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render an mp4 to the default video cache path.",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate every scenario in --data-root/--split and aggregate metrics.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional max number of scenarios for --eval-all.",
    )
    parser.add_argument("--eval-start-file-index", type=int, default=0)
    parser.add_argument("--eval-output-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_tfrecord(args: argparse.Namespace) -> Path:
    """Return the path of the single TFRecord to load, resolved from args."""
    split_dir = args.data_root / args.split
    if not split_dir.is_dir():
        raise UserInputError(f"Missing WOMD split directory: {split_dir}")

    files = sorted(
        path
        for path in split_dir.iterdir()
        if path.is_file() and ".tfrecord" in path.name
    )
    if not files:
        raise UserInputError(f"No TFRecord files found in {split_dir}")
    if args.file_index < 0 or args.file_index >= len(files):
        raise UserInputError(
            f"--file-index {args.file_index} out of range for {len(files)} files in {split_dir}"
        )
    return files[args.file_index]


def split_tfrecords(args: argparse.Namespace) -> list[Path]:
    """Return the sorted list of TFRecord paths for the selected split."""
    split_dir = args.data_root / args.split
    if not split_dir.is_dir():
        raise UserInputError(f"Missing WOMD split directory: {split_dir}")
    files = sorted(
        path
        for path in split_dir.iterdir()
        if path.is_file() and ".tfrecord" in path.name
    )
    if not files:
        raise UserInputError(f"No TFRecord files found in {split_dir}")
    if args.eval_start_file_index < 0 or args.eval_start_file_index >= len(files):
        raise UserInputError(
            f"--eval-start-file-index {args.eval_start_file_index} out of range for {len(files)} files in {split_dir}"
        )
    return files[args.eval_start_file_index :]


def cached_tfexample_path(split: str, file_index: int, scenario_index: int) -> Path:
    """Return the standard cached TFExample path for a raw WOMD scenario."""
    return (
        DEFAULT_TFEXAMPLE_CACHE_DIR / f"{split}_{file_index}_{scenario_index}.tfrecord"
    )


def cached_video_path(split: str, file_index: int, scenario_index: int) -> Path:
    """Return the standard cached rollout video path for a raw WOMD scenario."""
    return DEFAULT_VIDEO_CACHE_DIR / f"{split}_{file_index}_{scenario_index}.mp4"


def cached_json_video_path(json_path: Path | str) -> Path:
    """Return the standard cached rollout video path for a JSON scenario."""
    return DEFAULT_VIDEO_CACHE_DIR / f"{Path(json_path).stem}.mp4"


def object_type_from_name(name: str) -> Any:
    """Return the Waymax ObjectType enum value for the given name string."""
    return getattr(waymax_config.ObjectType, name)


def validate_waymax_tfrecord(tfrecord: Path | str) -> None:
    """Raise UserInputError if tfrecord is not a valid Waymax TFExample TFRecord."""
    try:
        raw_record = next(iter(tf.data.TFRecordDataset([str(tfrecord)])))
    except StopIteration as exc:
        raise UserInputError(f"TFRecord is empty: {tfrecord}") from exc

    example = tf.train.Example.FromString(raw_record.numpy())
    keys = set(example.features.feature.keys())
    if not keys:
        raise UserInputError(
            f"{tfrecord} is a raw Waymo Scenario protobuf TFRecord, not a Waymax TFExample TFRecord. "
            "Convert it to Waymax/WOMD TFExample format before using this runner."
        )

    missing = REQUIRED_WAYMAX_FEATURES - keys
    if missing:
        shown = ", ".join(sorted(missing))
        raise UserInputError(
            f"{tfrecord} is missing required Waymax TFExample features: {shown}"
        )


def make_synthetic_scenario(args: argparse.Namespace) -> Any:
    """Create a zero-filled synthetic Waymax scenario for testing."""
    data_config = waymax_config.DatasetConfig(
        path="",
        max_num_objects=MAX_NUM_OBJECTS,
        batch_dims=(),
        drop_remainder=False,
    )
    return test_utils.make_zeros_state(data_config)


def load_tfexample_scenario(
    args: argparse.Namespace, tfrecord: Path | str, scenario_index: int
) -> Any:
    """Load and return a Waymax SimulatorState from a TFExample TFRecord."""
    try:
        raw_record = next(
            record
            for idx, record in enumerate(tf.data.TFRecordDataset([str(tfrecord)]))
            if idx == scenario_index
        )
    except StopIteration as exc:
        raise UserInputError(
            f"Scenario index {scenario_index} is out of range for {tfrecord}"
        ) from exc

    features = womd_utils.get_features_description(
        max_num_objects=MAX_NUM_OBJECTS,
        max_num_rg_points=MAX_NUM_RG_POINTS,
    )
    decoded = tf.io.parse_single_example(raw_record, features)
    processed = womd_dataloader.preprocess_womd_example(
        decoded,
        aggregate_timesteps=True,
        max_num_objects=MAX_NUM_OBJECTS,
    )
    jnp_inputs = jax.tree_util.tree_map(jnp.asarray, processed)
    return womd_factories.simulator_state_from_womd_dict(
        jnp_inputs, include_sdc_paths=False
    )


def load_scenario_from_path(
    args: argparse.Namespace, tfrecord: Path, scenario_index: int
) -> tuple[Path, Any]:
    """Preprocess a raw TFRecord, then load and return the scenario."""
    try:
        tfrecord, stats = preprocess_scenario_to_tfexample(
            source_tfrecord=tfrecord,
            output_tfrecord=cached_tfexample_path(
                args.split, args.file_index, scenario_index
            ),
            scenario_index=scenario_index,
            overwrite=True,
        )
    except PreprocessError as exc:
        raise UserInputError(str(exc)) from exc
    scenario_index = 0
    if args.verbose:
        print(f"Preprocessed raw Scenario -> {tfrecord}")
        print(f"Preprocess stats: {stats}")

    validate_waymax_tfrecord(tfrecord)
    scenario = load_tfexample_scenario(args, tfrecord, scenario_index)
    return tfrecord, scenario


def load_scenario_from_json(args: argparse.Namespace, json_path: Path) -> tuple[Path, Any]:
    """Preprocess a ScenarioMax/PufferDrive JSON, then load it as a Waymax scenario."""
    try:
        tfrecord, stats = preprocess_json_to_tfexample(
            source_json=json_path,
            overwrite=True,
        )
    except PreprocessError as exc:
        raise UserInputError(str(exc)) from exc
    if args.verbose:
        print(f"Preprocessed JSON -> {tfrecord}")
        print(f"Preprocess stats: {stats}")

    validate_waymax_tfrecord(tfrecord)
    scenario = load_tfexample_scenario(args, tfrecord, 0)
    return tfrecord, scenario


def load_scenario_from_bin(args: argparse.Namespace, bin_path: Path) -> tuple[Path, Any]:
    """Preprocess a PufferDrive .bin map, then load it as a Waymax scenario."""
    try:
        tfrecord, stats = preprocess_bin_to_tfexample(
            source_bin=bin_path,
            overwrite=True,
        )
    except PreprocessError as exc:
        raise UserInputError(str(exc)) from exc
    if args.verbose:
        print(f"Preprocessed .bin -> {tfrecord}")
        print(f"Preprocess stats: {stats}")

    validate_waymax_tfrecord(tfrecord)
    scenario = load_tfexample_scenario(args, tfrecord, 0)
    return tfrecord, scenario


def load_scenario(args: argparse.Namespace) -> tuple[Path, Any]:
    """Resolve the TFRecord path from args and load the requested scenario."""
    if args.bin is not None:
        return load_scenario_from_bin(args, args.bin)
    if args.json is not None:
        return load_scenario_from_json(args, args.json)
    return load_scenario_from_path(args, resolve_tfrecord(args), args.scenario_index)


def build_environment(args: argparse.Namespace) -> Any:
    """Construct and return a Waymax BaseEnvironment from args."""
    env_config = waymax_config.EnvironmentConfig(
        max_num_objects=MAX_NUM_OBJECTS,
        init_steps=INIT_STEPS,
        controlled_object=object_type_from_name(args.controlled_object),
        compute_reward=args.compute_reward,
        metrics=waymax_config.MetricsConfig(metrics_to_run=WAYMAX_METRICS_TO_RUN),
    )
    return base_environment.BaseEnvironment(
        dynamics_model=dynamics.StateDynamics(), config=env_config
    )


def control_mask(state: Any, env: Any) -> Any:
    """Return a boolean control mask for the currently controlled objects."""
    mask = datatypes.get_control_mask(
        state.object_metadata, env.config.controlled_object
    )
    if (
        env.config.controlled_object.name == "VALID"
        and not env.config.allow_new_objects_after_warmup
    ):
        mask = state.current_sim_trajectory.valid[..., 0]
    return mask.astype(jnp.bool_)


def expert_action(state: Any, env: Any) -> Any:
    """Return the expert (log-replay) action for the current timestep."""
    return env.dynamics.inverse(
        state.log_trajectory, state.object_metadata, timestep=int(state.timestep)
    )


def constant_speed_action(state: Any, env: Any, speed: float | None) -> Any:
    """Build a constant-speed (or current-speed) action for the controlled objects."""
    traj_t0 = datatypes.dynamic_index(
        state.sim_trajectory, state.timestep, axis=-1, keepdims=True
    )
    if speed is None:
        vel_x = traj_t0.vel_x
        vel_y = traj_t0.vel_y
    else:
        vel_x = speed * jnp.cos(traj_t0.yaw)
        vel_y = speed * jnp.sin(traj_t0.yaw)

    action = jnp.concatenate(
        [
            traj_t0.x + vel_x * datatypes.TIME_INTERVAL,
            traj_t0.y + vel_y * datatypes.TIME_INTERVAL,
            traj_t0.yaw,
            vel_x,
            vel_y,
        ],
        axis=-1,
    )
    valid = control_mask(state, env)[..., None] & traj_t0.valid
    return datatypes.Action(data=action, valid=valid)


def zero_action(state: Any, env: Any) -> Any:
    """Return a zero-velocity action for the controlled objects."""
    return constant_speed_action(state, env, speed=0.0)


class PufferDrivePolicyEnv:
    """Minimal env wrapper exposing observation/action spaces for the PufferDrive policy."""

    def __init__(self, spec: dict[str, Any]) -> None:
        """Initialise observation/action space dimensions from a policy spec."""
        self.num_agents = 1
        self.ego_features = spec["ego_features"]
        self.max_partner_objects = spec["max_partner_objects"]
        self.partner_features = PARTNER_FEATURES
        self.max_road_objects = spec["max_road_objects"]
        self.road_features = ROAD_FEATURES
        self.num_obs = (
            self.ego_features
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
        )
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(self.num_obs,), dtype="float32"
        )
        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            [len(ACCELERATION_VALUES) * len(STEERING_VALUES)]
        )


def build_puffer_model(args: argparse.Namespace) -> Any:
    """Load and return the puffer policy model for the selected --policy."""
    spec = POLICY_SPECS[args.policy]
    config = load_puffer_config(spec["config_path"])
    model_path = resolve_puffer_model(
        args.load_model_path or str(spec["default_model"]), spec["model_glob"]
    )
    # Keep the obs-visualizer's slicing layout in sync with the active policy.
    viz_model_tool.set_layout(
        spec["ego_features"], spec["max_partner_objects"], spec["max_road_objects"]
    )
    return PufferModule(
        config,
        PufferDrivePolicyEnv(spec),
        model_path,
        deterministic=args.deterministic,
    )


def sdc_index(state: Any) -> int:
    """Return the object slot index of the SDC in the scenario."""
    is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
    indices = np.flatnonzero(is_sdc)
    if indices.size == 0:
        raise UserInputError("Waymax scenario has no SDC object")
    return int(indices[0])


def current_trajectory_arrays(state: Any) -> dict[str, np.ndarray]:
    """Extract current-timestep trajectory fields as a dict of numpy arrays."""
    traj = state.current_sim_trajectory

    def current_or_flat(value):
        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[-1] == 1:
            return array[..., 0]
        return array

    return {
        "x": np.asarray(traj.x)[..., 0],
        "y": np.asarray(traj.y)[..., 0],
        "z": np.asarray(traj.z)[..., 0],
        "vx": np.asarray(traj.vel_x)[..., 0],
        "vy": np.asarray(traj.vel_y)[..., 0],
        "yaw": np.asarray(traj.yaw)[..., 0],
        "valid": np.asarray(traj.valid)[..., 0].astype(bool),
        "length": current_or_flat(traj.length),
        "width": current_or_flat(traj.width),
    }


def sdc_goal_xy(state: Any, idx: int) -> tuple[float, float]:
    """Return the last valid (x, y) log-trajectory position of object idx as its goal."""
    valid = np.asarray(state.log_trajectory.valid)[idx].astype(bool)
    x = np.asarray(state.log_trajectory.x)[idx]
    y = np.asarray(state.log_trajectory.y)[idx]
    future_indices = np.flatnonzero(valid)
    if future_indices.size == 0:
        raise UserInputError(
            f"SDC object has no valid log trajectory points for goal extraction"
        )
    last = int(future_indices[-1])
    return float(x[last]), float(y[last])


def road_category(road_type: int) -> int:
    """Map a Waymo road feature type integer to a PufferDrive road category index."""
    road_type = int(road_type)
    if 1 <= road_type <= 3:
        return 0
    if 5 <= road_type <= 13:
        return 1
    if 14 <= road_type <= 16:
        return 2
    if road_type == 17:
        return 3
    if road_type == 18:
        return 4
    if road_type == 19:
        return 5
    return 6


def pufferdrive_observation_from_waymax(
    state: Any,
    spec: dict[str, Any],
    collision_factor: float = DEFAULT_COLLISION_FACTOR,
    offroad_factor: float = DEFAULT_OFFROAD_FACTOR,
    lane_width: float = DEFAULT_LANE_WIDTH,
) -> np.ndarray:
    """Build a PufferDrive observation vector from the current Waymax simulator state."""
    ego_features = spec["ego_features"]
    max_partner_objects = spec["max_partner_objects"]
    max_road_objects = spec["max_road_objects"]
    obs_range_sq = spec["obs_range_sq"]
    obs = np.zeros(
        (
            1,
            ego_features
            + max_partner_objects * PARTNER_FEATURES
            + max_road_objects * ROAD_FEATURES,
        ),
        dtype=np.float32,
    )
    arrays = current_trajectory_arrays(state)
    ego_idx = sdc_index(state)
    ego_x = float(arrays["x"][ego_idx])
    ego_y = float(arrays["y"][ego_idx])
    ego_yaw = float(arrays["yaw"][ego_idx])
    ego_cos = math.cos(ego_yaw)
    ego_sin = math.sin(ego_yaw)
    ego_vx = float(arrays["vx"][ego_idx])
    ego_vy = float(arrays["vy"][ego_idx])
    ego_speed = math.hypot(ego_vx, ego_vy)
    ego_signed_speed = math.copysign(ego_speed, ego_vx * ego_cos + ego_vy * ego_sin)
    goal_x, goal_y = sdc_goal_xy(state, ego_idx)
    goal_dx = goal_x - ego_x
    goal_dy = goal_y - ego_y

    obs[0, 0] = (goal_dx * ego_cos + goal_dy * ego_sin) * 0.005
    obs[0, 1] = (-goal_dx * ego_sin + goal_dy * ego_cos) * 0.005
    obs[0, 2] = ego_signed_speed / MAX_SPEED
    obs[0, 3] = float(arrays["width"][ego_idx]) / MAX_VEH_WIDTH
    obs[0, 4] = float(arrays["length"][ego_idx]) / MAX_VEH_LEN
    obs[0, 5] = 0.0  # collision state, 0: no, 1: has collision
    obs[0, 6] = 0.0  # respawn state: 0: no, 1: just respawned
    object_types = np.asarray(state.object_metadata.object_types)
    obs[0, 7] = float(object_types[ego_idx]) / 3.0
    if spec["has_conditioning"]:
        # selfplay_drive conditioning inputs (raw, un-normalized; see drive.h)
        # Trailing 3 ego features: collision_factor, offroad_factor, lane_width
        obs[0, ego_features - 3] = collision_factor
        obs[0, ego_features - 2] = offroad_factor
        obs[0, ego_features - 1] = lane_width

    partner_start = ego_features
    partner_rows = []
    for idx in range(arrays["x"].shape[0]):
        if idx == ego_idx or not arrays["valid"][idx]:
            continue
        obj_type = int(object_types[idx])
        if obj_type <= 0 or obj_type > 3:
            continue
        dx = float(arrays["x"][idx]) - ego_x
        dy = float(arrays["y"][idx]) - ego_y
        dist_sq = dx * dx + dy * dy
        if dist_sq > obs_range_sq:
            continue
        rel_x = dx * ego_cos + dy * ego_sin
        rel_y = -dx * ego_sin + dy * ego_cos
        other_yaw = float(arrays["yaw"][idx])
        other_cos = math.cos(other_yaw)
        other_sin = math.sin(other_yaw)
        other_speed = math.hypot(float(arrays["vx"][idx]), float(arrays["vy"][idx]))
        other_signed_speed = math.copysign(
            other_speed,
            float(arrays["vx"][idx]) * other_cos + float(arrays["vy"][idx]) * other_sin,
        )
        partner_rows.append(
            (
                dist_sq,
                np.asarray(
                    [
                        rel_x * 0.02,
                        rel_y * 0.02,
                        float(arrays["width"][idx]) / MAX_VEH_WIDTH,
                        float(arrays["length"][idx]) / MAX_VEH_LEN,
                        math.cos(other_yaw - ego_yaw),
                        math.sin(other_yaw - ego_yaw),
                        other_signed_speed / MAX_SPEED,
                    ],
                    dtype=np.float32,
                ),
            )
        )
    for slot, (_, row) in enumerate(
        sorted(partner_rows, key=lambda item: item[0])[:max_partner_objects]
    ):
        obs[
            0,
            partner_start
            + slot * PARTNER_FEATURES : partner_start
            + (slot + 1) * PARTNER_FEATURES,
        ] = row

    road_start = partner_start + max_partner_objects * PARTNER_FEATURES
    rg = state.roadgraph_points
    xyz = np.asarray(rg.xyz)
    dirs = np.asarray(rg.dir_xyz)
    ids = np.asarray(rg.ids)
    types = np.asarray(rg.types)
    valid = np.asarray(rg.valid).astype(bool)
    segment_rows = []
    for idx in range(max(0, xyz.shape[0] - 1)):
        if not valid[idx] or not valid[idx + 1] or ids[idx] != ids[idx + 1]:
            continue
        mid = 0.5 * (xyz[idx] + xyz[idx + 1])
        dx = float(mid[0]) - ego_x
        dy = float(mid[1]) - ego_y
        dist_sq = dx * dx + dy * dy
        if dist_sq > obs_range_sq:
            continue
        direction = dirs[idx]
        direction_norm = float(np.linalg.norm(direction[:2]))
        if direction_norm <= 1e-6:
            edge = xyz[idx + 1] - xyz[idx]
            direction_norm = float(np.linalg.norm(edge[:2]))
            direction = (
                edge / direction_norm
                if direction_norm > 1e-6
                else np.asarray([1.0, 0.0, 0.0])
            )
        rel_x = dx * ego_cos + dy * ego_sin
        rel_y = -dx * ego_sin + dy * ego_cos
        dir_x = float(direction[0])
        dir_y = float(direction[1])
        length = float(np.linalg.norm((xyz[idx + 1] - xyz[idx])[:2]))
        segment_rows.append(
            (
                dist_sq,
                np.asarray(
                    [
                        rel_x * 0.02,
                        rel_y * 0.02,
                        length / MAX_ROAD_SEGMENT_LENGTH,
                        0.1 / MAX_ROAD_SCALE,
                        dir_x * ego_cos + dir_y * ego_sin,
                        -dir_x * ego_sin + dir_y * ego_cos,
                        road_category(types[idx]),
                    ],
                    dtype=np.float32,
                ),
            )
        )
    for slot, (_, row) in enumerate(
        sorted(segment_rows, key=lambda item: item[0])[:max_road_objects]
    ):
        obs[
            0,
            road_start + slot * ROAD_FEATURES : road_start + (slot + 1) * ROAD_FEATURES,
        ] = row
    return obs


def pufferdrive_action_to_waymax(action_value: Any, state: Any) -> Any:
    """Convert a PufferDrive discrete action index to a Waymax Action."""
    action_value = int(np.asarray(action_value).reshape(-1)[0])
    num_steer = len(STEERING_VALUES)
    acceleration = ACCELERATION_VALUES[action_value // num_steer]
    steering = STEERING_VALUES[action_value % num_steer]
    arrays = current_trajectory_arrays(state)
    ego_idx = sdc_index(state)
    x = float(arrays["x"][ego_idx])
    y = float(arrays["y"][ego_idx])
    yaw = float(arrays["yaw"][ego_idx])
    vx = float(arrays["vx"][ego_idx])
    vy = float(arrays["vy"][ego_idx])
    length = max(float(arrays["length"][ego_idx]), 1e-3)
    heading_x = math.cos(yaw)
    heading_y = math.sin(yaw)
    speed = math.hypot(vx, vy)
    signed_speed = math.copysign(speed, vx * heading_x + vy * heading_y)
    signed_speed += acceleration * TIME_INTERVAL
    signed_speed = max(-MAX_SPEED, min(MAX_SPEED, signed_speed))
    beta = math.tanh(0.5 * math.tan(steering))
    yaw_rate = signed_speed * math.cos(beta) * math.tan(steering) / length
    new_yaw = yaw + yaw_rate * TIME_INTERVAL
    new_vx = signed_speed * math.cos(yaw + beta)
    new_vy = signed_speed * math.sin(yaw + beta)
    new_x = x + new_vx * TIME_INTERVAL
    new_y = y + new_vy * TIME_INTERVAL

    data = np.zeros((state.num_objects, 5), dtype=np.float32)
    valid = np.zeros((state.num_objects, 1), dtype=bool)
    data[ego_idx] = (new_x, new_y, new_yaw, new_vx, new_vy)
    valid[ego_idx, 0] = bool(arrays["valid"][ego_idx])
    return datatypes.Action(data=jnp.asarray(data), valid=jnp.asarray(valid))


def pufferdrive_action(args: argparse.Namespace, state: Any, puffer_model: Any) -> Any:
    """Run one puffer policy step and return the resulting Waymax Action."""
    if puffer_model is None:
        raise UserInputError(f"--policy {args.policy} requires a loaded puffer model")
    obs = pufferdrive_observation_from_waymax(
        state,
        POLICY_SPECS[args.policy],
        collision_factor=args.collision_factor,
        offroad_factor=args.offroad_factor,
        lane_width=args.lane_width,
    )
    viz_model_tool.add_input(obs) # TODO
    action_value = puffer_model.step(obs)
    viz_model_tool.add_output(action_value) # TODO
    return pufferdrive_action_to_waymax(action_value, state)


def select_action(
    args: argparse.Namespace, state: Any, env: Any, puffer_model: Any | None = None
) -> Any:
    """Choose and return an action for the current timestep according to --policy."""
    if args.policy == "expert":
        action = expert_action(state, env)
        return action.replace(valid=action.valid & control_mask(state, env)[..., None])
    if args.policy == "constant_speed":
        return constant_speed_action(state, env, None)
    if args.policy == "zero":
        return zero_action(state, env)
    if args.policy in PUFFER_POLICIES:
        return pufferdrive_action(args, state, puffer_model)
    raise ValueError(f"Unsupported policy: {args.policy}")


def scalar_metric(value: Any, mask: Any | None = None) -> float:
    """Return the nan-mean of value, optionally filtered by a boolean mask."""
    arr = np.asarray(value)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == arr.shape:
            arr = arr[mask]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def summarize_metrics(
    metrics: Any, state: Any, object_mask: Any | None = None
) -> dict[str, float]:
    """Reduce per-object Waymax metrics to scalar means over valid controlled objects."""
    current_valid = np.asarray(state.current_sim_trajectory.valid)[..., 0].astype(bool)
    if object_mask is not None:
        current_valid = current_valid & np.asarray(object_mask, dtype=bool)
    summary = {}
    for name, value in metrics.items():
        metric_valid = np.asarray(value.valid).astype(bool)
        mask = (
            metric_valid & current_valid
            if metric_valid.shape == current_valid.shape
            else metric_valid
        )
        summary[name] = scalar_metric(value.value, mask)
    return summary


class MetricAccumulator:
    """Accumulate per-scenario metric values and compute their running mean."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, metrics: dict[str, float]) -> None:
        """Add one scenario's metric dict, ignoring NaN values."""
        for name, value in metrics.items():
            if math.isnan(value):
                continue
            self.sums[name] = self.sums.get(name, 0.0) + float(value)
            self.counts[name] = self.counts.get(name, 0) + 1

    def mean(self) -> dict[str, float]:
        """Return the mean of each metric over all added scenarios."""
        return {
            name: self.sums[name] / self.counts[name]
            for name in sorted(self.sums)
            if self.counts.get(name, 0) > 0
        }


def normalize_frame(frame: Any) -> np.ndarray:
    """Convert a Waymax rendered frame to a contiguous uint8 RGB numpy array."""
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def write_mp4_with_ffmpeg(
    frames: list[np.ndarray], output_path: Path | str, fps: int
) -> None:
    """Encode a list of RGB uint8 frames to an mp4 file via ffmpeg."""
    if not frames:
        raise ValueError("No frames to render")

    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    payload = bytearray()
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("All frames must have the same dimensions")
        payload.extend(frame.tobytes())

    process = subprocess.run(
        command,
        input=bytes(payload),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr = process.stderr
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while writing {output_path}:\n{stderr.decode('utf-8', errors='replace')}"
        )


def render_rollout_mp4(
    states: list[Any], output_path: Path, fps: int, use_log_traj: bool = False
) -> tuple[Path, int, float]:
    """Render simulator states to an mp4; returns (path, frame_count, duration)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        normalize_frame(viz.plot_simulator_state(state, use_log_traj=use_log_traj))
        for state in states
    ]
    write_mp4_with_ffmpeg(frames, output_path, fps)
    return output_path, len(frames), len(frames) / fps


def rollout_scenario(
    args: argparse.Namespace,
    env: Any,
    scenario: Any,
    puffer_model: Any | None = None,
    collect_states: bool = False,
) -> tuple[Any, dict[str, float], list[Any] | None, int]:
    """Run a full episode rollout; returns (final_state, rollout_mean_metrics, states, steps)."""
    state = env.reset(scenario)
    states = [state] if collect_states else None
    metric_object_mask = np.asarray(
        datatypes.get_control_mask(state.object_metadata, env.config.controlled_object)
    ).astype(bool)
    metric_accumulator = MetricAccumulator()
    step_count = 0

    max_steps = min(NUM_STEPS, int(state.remaining_timesteps))

    for _ in range(max_steps):
        action = select_action(args, state, env, puffer_model=puffer_model)
        state = env.step(state, action)
        metric_accumulator.add(
            summarize_metrics(
                env.metrics(state),
                state,
                object_mask=metric_object_mask,
            )
        )
        step_count += 1
        if states is not None:
            states.append(state)
        if bool(state.is_done):
            break

    metric_summary = metric_accumulator.mean()
    if not metric_summary:
        metric_summary = summarize_metrics(
            env.metrics(state), state, object_mask=metric_object_mask
        )
    return state, metric_summary, states, step_count


def run(args: argparse.Namespace) -> tuple[Any, dict[str, float]]:
    """Run a single scenario rollout and print results; returns (state, metrics)."""
    if args.synthetic and args.json is not None:
        raise UserInputError("--synthetic and --json are mutually exclusive")
    if args.synthetic:
        tfrecord = None
        scenario = make_synthetic_scenario(args)
    else:
        tfrecord, scenario = load_scenario(args)
    env = build_environment(args)
    puffer_model = build_puffer_model(args) if args.policy in PUFFER_POLICIES else None
    state, metric_summary, states, max_steps = rollout_scenario(
        args,
        env,
        scenario,
        puffer_model=puffer_model,
        collect_states=args.video,
    )
    

    if args.verbose:
        print(f"Input: {'synthetic zero scenario' if args.synthetic else tfrecord}")
        print(f"Scenario index: {args.scenario_index}")
        print(f"Objects: {state.num_objects}")
        print(f"Rollout steps: {max_steps}")

    print(f"Input: {'synthetic zero scenario' if args.synthetic else tfrecord}")
    print(f"Policy: {args.policy}")
    print(f"Final timestep: {int(state.timestep)}")
    print(f"Metrics: {metric_summary}")

    if args.video:
        video_path = (
            cached_json_video_path(args.json)
            if args.json is not None
            else cached_video_path(args.split, args.file_index, args.scenario_index)
        )
        output, frame_count, duration = render_rollout_mp4(
            states,
            video_path,
            fps=RENDER_FPS,
            use_log_traj=False,
        )
        print(
            f"Wrote {output} ({frame_count} frames, {duration:.2f}s at {RENDER_FPS} fps)"
        )
        
        # TODO
        model_video_path = video_path.with_name(video_path.stem + "_model.mp4")
        viz_model_tool.create_video(model_video_path)
        print(f"Wrote model video {model_video_path}")
        

    return state, metric_summary


def evaluate_all(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the policy over all scenarios in the split and aggregate metrics."""
    if args.synthetic:
        raise UserInputError("--eval-all does not support --synthetic")
    if args.video:
        raise UserInputError("--eval-all does not support --video")

    env = build_environment(args)
    puffer_model = build_puffer_model(args) if args.policy in PUFFER_POLICIES else None
    accumulator = MetricAccumulator()
    records = []
    scenario_count = 0

    for file_index, tfrecord in enumerate(
        split_tfrecords(args), start=args.eval_start_file_index
    ):
        dataset = tf.data.TFRecordDataset([str(tfrecord)])
        for scenario_index, raw_record in enumerate(dataset):
            if args.eval_limit is not None and scenario_count >= args.eval_limit:
                break
            try:
                cached_tfrecord, preprocess_stats = (
                    preprocess_serialized_scenario_to_tfexample(
                        source_tfrecord=tfrecord,
                        serialized_scenario=raw_record,
                        output_tfrecord=cached_tfexample_path(
                            args.split, file_index, scenario_index
                        ),
                        scenario_index=scenario_index,
                        overwrite=False,
                    )
                )
            except PreprocessError as exc:
                raise UserInputError(str(exc)) from exc
            scenario = load_tfexample_scenario(args, cached_tfrecord, 0)
            state, metric_summary, _, _ = rollout_scenario(
                args,
                env,
                scenario,
                puffer_model=puffer_model,
                collect_states=False,
            )
            accumulator.add(metric_summary)
            records.append(
                {
                    "source_tfrecord": str(tfrecord),
                    "cached_tfrecord": str(cached_tfrecord),
                    "file_index": file_index,
                    "scenario_index": scenario_index,
                    "scenario_id": preprocess_stats.get("scenario_id"),
                    "final_timestep": int(state.timestep),
                    "metrics": metric_summary,
                }
            )
            scenario_count += 1
            if args.verbose or scenario_count % 50 == 0:
                print(
                    f"Evaluated {scenario_count} scenarios; latest file={file_index} scenario={scenario_index} metrics={metric_summary}"
                )
        if args.eval_limit is not None and scenario_count >= args.eval_limit:
            break

    summary = {
        "policy": args.policy,
        "split": args.split,
        "num_scenarios": scenario_count,
        "mean_metrics": accumulator.mean(),
        "metric_counts": accumulator.counts,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.eval_output_json is not None:
        args.eval_output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.eval_output_json.open("w") as f:
            json.dump(
                {"summary": summary, "records": records}, f, indent=2, sort_keys=True
            )
        print(f"Wrote {args.eval_output_json}")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: dispatch to run() or evaluate_all() based on args."""
    args = parse_args(argv)
    try:
        if args.eval_all:
            evaluate_all(args)
        else:
            run(args)
    except UserInputError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
