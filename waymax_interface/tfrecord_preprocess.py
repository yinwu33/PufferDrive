from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

PAST_STEPS = 10
CURRENT_STEPS = 1
FUTURE_STEPS = 80
TOTAL_STEPS = PAST_STEPS + CURRENT_STEPS + FUTURE_STEPS
DEFAULT_CACHE_DIR = Path("waymax_interface/cache/tfexample")

LANE_TYPE_OFFSET = 0
ROAD_LINE_TYPE_OFFSET = 5
ROAD_EDGE_TYPE_OFFSET = 14
STOP_SIGN_TYPE = 17
CROSSWALK_TYPE = 18
SPEED_BUMP_TYPE = 19


MAX_NUM_OBJECTS = 128
MAX_NUM_RG_POINTS = 30000
NUM_TLS = 16


class PreprocessError(RuntimeError):
    pass


def _float_feature(values: list[float]) -> tf.train.Feature:
    """Wrap a list of floats as a TFRecord float Feature."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[float(v) for v in values])
    )


def _int64_feature(values: list[int]) -> tf.train.Feature:
    """Wrap a list of ints as a TFRecord int64 Feature."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int(v) for v in values])
    )


def _example_from_arrays(arrays: dict[str, np.ndarray]) -> tf.train.Example:
    """Convert a dict of named numpy arrays into a tf.train.Example."""
    features = {}
    for name, values in arrays.items():
        flat = values.reshape(-1)
        if values.dtype.kind in ("f",):
            features[name] = _float_feature(flat)
        else:
            features[name] = _int64_feature(flat)
    return tf.train.Example(features=tf.train.Features(feature=features))


def _read_scenario_at_index(tfrecord_path: Path | str, scenario_index: int) -> Any:
    """Read and parse the scenario at the given index from a TFRecord file."""
    if scenario_index < 0:
        raise PreprocessError("--scenario-index must be non-negative")
    dataset = tf.data.TFRecordDataset([str(tfrecord_path)])
    total_scenarios = sum(1 for _ in dataset)
    if scenario_index >= total_scenarios:
        raise PreprocessError(
            f"Scenario index {scenario_index} is out of range for {tfrecord_path} with {total_scenarios} scenarios."
        )
    for idx, raw_record in enumerate(dataset):
        if idx == scenario_index:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw_record.numpy())
            if not scenario.scenario_id:
                raise PreprocessError(
                    f"Record {scenario_index} did not parse as a raw Waymo Scenario protobuf."
                )
            return scenario
    raise PreprocessError(
        f"Scenario index {scenario_index} is out of range for {tfrecord_path}"
    )


def _selected_track_indices(scenario: Any, max_num_objects: int) -> list[int]:
    """
    Create a priority list of track indices: tracks_to_predict first, then SDC (if not
    already in ttp), then the rest. Cut off the list at max_num_objects.
    """
    if not (0 <= scenario.sdc_track_index < len(scenario.tracks)):
        raise PreprocessError(
            f"Invalid sdc_track_index {scenario.sdc_track_index} for scenario with {len(scenario.tracks)} tracks."
        )
    # Sort ttp tracks by object ID to match Waymax official ordering
    ttp_indices = sorted(
        [pred.track_index for pred in scenario.tracks_to_predict],
        key=lambda idx: scenario.tracks[idx].id,
    )
    priority = ttp_indices
    priority.append(scenario.sdc_track_index)
    priority.extend(range(len(scenario.tracks)))

    selected = []
    seen = set()
    for idx in priority:
        if idx < 0 or idx >= len(scenario.tracks) or idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) == max_num_objects:
            break
    return selected


def _time_slices(current_time_index: int) -> dict[str, list[int]]:
    """Return timestep index lists for past, current, and future windows."""
    past = list(range(current_time_index - PAST_STEPS, current_time_index))
    current = [current_time_index]
    future = list(range(current_time_index + 1, current_time_index + 1 + FUTURE_STEPS))
    return {"past": past, "current": current, "future": future}


def _safe_state(track: Any, timestep: int) -> Any | None:
    """Return the valid state at timestep, or None if out-of-bounds or invalid."""
    if timestep < 0 or timestep >= len(track.states):
        return None
    state = track.states[timestep]
    return state if state.valid else None


def _fill_state_features(
    arrays: dict[str, np.ndarray],
    prefix: str,
    object_slot: int,
    time_slot: int,
    state: Any | None,
    timestamp_micros: int,
) -> None:
    """Write one object state into the arrays at the given slot."""
    valid = 1 if state is not None else 0
    arrays[f"state/{prefix}/valid"][object_slot, time_slot] = valid
    arrays[f"state/{prefix}/timestamp_micros"][object_slot, time_slot] = (
        timestamp_micros if valid else -1
    )
    if state is None:
        return

    speed = math.hypot(state.velocity_x, state.velocity_y)
    vel_yaw = (
        math.atan2(state.velocity_y, state.velocity_x)
        if speed > 1e-6
        else state.heading
    )
    arrays[f"state/{prefix}/bbox_yaw"][object_slot, time_slot] = state.heading
    arrays[f"state/{prefix}/height"][object_slot, time_slot] = state.height
    arrays[f"state/{prefix}/length"][object_slot, time_slot] = state.length
    arrays[f"state/{prefix}/vel_yaw"][object_slot, time_slot] = vel_yaw
    arrays[f"state/{prefix}/speed"][object_slot, time_slot] = speed
    arrays[f"state/{prefix}/velocity_x"][object_slot, time_slot] = state.velocity_x
    arrays[f"state/{prefix}/velocity_y"][object_slot, time_slot] = state.velocity_y
    arrays[f"state/{prefix}/width"][object_slot, time_slot] = state.width
    arrays[f"state/{prefix}/x"][object_slot, time_slot] = state.center_x
    arrays[f"state/{prefix}/y"][object_slot, time_slot] = state.center_y
    arrays[f"state/{prefix}/z"][object_slot, time_slot] = state.center_z


def _timestamp_micros(scenario: Any, timestep: int) -> int:
    """Convert a scenario timestep to microseconds, or -1 if out of range."""
    if timestep < 0 or timestep >= len(scenario.timestamps_seconds):
        return -1
    # timestamps_seconds is stored as float32; preserve that precision to match official.
    return int(np.float32(scenario.timestamps_seconds[timestep]) * np.float32(1_000_000))


def _init_arrays(
    max_num_objects: int, max_num_rg_points: int, num_tls: int
) -> dict[str, np.ndarray]:
    """Allocate output arrays filled with sentinel values."""
    arrays = {
        # roadgraph_samples
        "roadgraph_samples/xyz": np.full(
            (max_num_rg_points, 3), -1.0, dtype=np.float32
        ),
        "roadgraph_samples/dir": np.full(
            (max_num_rg_points, 3), -1.0, dtype=np.float32
        ),
        "roadgraph_samples/type": np.full((max_num_rg_points, 1), -1, dtype=np.int64),
        "roadgraph_samples/valid": np.zeros((max_num_rg_points, 1), dtype=np.int64),
        "roadgraph_samples/id": np.full((max_num_rg_points, 1), -1, dtype=np.int64),
        # states
        "state/tracks_to_predict": np.full((max_num_objects,), -1, dtype=np.int64),
        "state/objects_of_interest": np.full((max_num_objects,), -1, dtype=np.int64),
        "state/difficulty_level": np.full((max_num_objects,), -1, dtype=np.int64),
        "state/id": np.full((max_num_objects,), -1.0, dtype=np.float32),
        "state/is_sdc": np.full((max_num_objects,), -1, dtype=np.int64),
        "state/type": np.full((max_num_objects,), -1.0, dtype=np.float32),
    }
    for prefix, steps in (
        ("past", PAST_STEPS),
        ("current", CURRENT_STEPS),
        ("future", FUTURE_STEPS),
    ):
        for name in (
            "x",
            "y",
            "z",
            "bbox_yaw",
            "length",
            "width",
            "height",
            "speed",
            "vel_yaw",
            "velocity_x",
            "velocity_y",
        ):
            arrays[f"state/{prefix}/{name}"] = np.full(
                (max_num_objects, steps), -1.0, dtype=np.float32
            )
        arrays[f"state/{prefix}/timestamp_micros"] = np.full(
            (max_num_objects, steps), -1, dtype=np.int64
        )
        arrays[f"state/{prefix}/valid"] = np.zeros(
            (max_num_objects, steps), dtype=np.int64
        )

        tl_steps = steps
        for name in ("state", "valid", "id"):
            arrays[f"traffic_light_state/{prefix}/{name}"] = np.full(
                (tl_steps, num_tls), -1, dtype=np.int64
            )
        arrays[f"traffic_light_state/{prefix}/valid"] = np.zeros(
            (tl_steps, num_tls), dtype=np.int64
        )
        for name in ("x", "y", "z"):
            arrays[f"traffic_light_state/{prefix}/{name}"] = np.full(
                (tl_steps, num_tls), -1.0, dtype=np.float32
            )
        arrays[f"traffic_light_state/{prefix}/timestamp_micros"] = np.full(
            (tl_steps,), -1, dtype=np.int64
        )
    return arrays


def _fill_tracks(
    arrays: dict[str, np.ndarray], scenario: Any, selected_indices: list[int]
) -> None:
    """Populate per-object state arrays for all selected track indices."""
    prediction_indices = {pred.track_index for pred in scenario.tracks_to_predict}
    difficulty_by_track = {
        pred.track_index: int(getattr(pred, "difficulty", 0))
        for pred in scenario.tracks_to_predict
    }
    interest_ids = set(scenario.objects_of_interest)
    slices = _time_slices(scenario.current_time_index)

    for slot, track_idx in enumerate(selected_indices):
        track = scenario.tracks[track_idx]
        arrays["state/id"][slot] = float(track.id)
        arrays["state/type"][slot] = float(track.object_type)
        arrays["state/is_sdc"][slot] = 1 if track_idx == scenario.sdc_track_index else 0
        arrays["state/tracks_to_predict"][slot] = (
            1 if track_idx in prediction_indices else 0
        )
        arrays["state/objects_of_interest"][slot] = 1 if track.id in interest_ids else 0
        arrays["state/difficulty_level"][slot] = difficulty_by_track.get(track_idx, 0)

        for prefix, timesteps in slices.items():
            for time_slot, timestep in enumerate(timesteps):
                _fill_state_features(
                    arrays,
                    prefix,
                    slot,
                    time_slot,
                    _safe_state(track, timestep),
                    _timestamp_micros(scenario, timestep),
                )


def _point_xyz(point: Any) -> tuple[float, float, float]:
    """Extract (x, y, z) from a proto point."""
    return (point.x, point.y, point.z)


def _direction(
    points: Any, idx: int, closed: bool = False
) -> tuple[float, float, float]:
    """Compute the unit direction vector at index idx along a polyline."""
    if len(points) <= 1:
        return (0.0, 0.0, 0.0)
    if idx + 1 < len(points):
        p0 = points[idx]
        p1 = points[idx + 1]
    elif closed:
        p0 = points[idx]
        p1 = points[0]
    else:
        p0 = points[idx - 1]
        p1 = points[idx]
    vec = np.asarray(_point_xyz(p1), dtype=np.float32) - np.asarray(
        _point_xyz(p0), dtype=np.float32
    )
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return (0.0, 0.0, 0.0)
    return tuple((vec / norm).tolist())


def _append_polyline(
    samples: list, feature_id: int, element_type: int, points: Any, closed: bool = False
) -> None:
    """Append (xyz, direction, id, type) tuples for each point in a polyline."""
    for idx, point in enumerate(points):
        samples.append(
            (
                _point_xyz(point),
                _direction(points, idx, closed=closed),
                feature_id,
                element_type,
            )
        )


def _fill_roadgraph(
    arrays: dict[str, np.ndarray], scenario: Any, max_num_rg_points: int
) -> int:
    """Fill roadgraph sample arrays from map features; returns total sample count."""
    samples = []
    for feature in scenario.map_features:
        feature_type = feature.WhichOneof("feature_data")
        if feature_type == "lane":
            _append_polyline(
                samples,
                feature.id,
                LANE_TYPE_OFFSET + feature.lane.type,
                feature.lane.polyline,
            )
        elif feature_type == "road_line":
            _append_polyline(
                samples,
                feature.id,
                ROAD_LINE_TYPE_OFFSET + feature.road_line.type,
                feature.road_line.polyline,
            )
        elif feature_type == "road_edge":
            _append_polyline(
                samples,
                feature.id,
                ROAD_EDGE_TYPE_OFFSET + feature.road_edge.type,
                feature.road_edge.polyline,
            )
        elif feature_type == "stop_sign":
            samples.append(
                (
                    _point_xyz(feature.stop_sign.position),
                    (0.0, 0.0, 0.0),
                    feature.id,
                    STOP_SIGN_TYPE,
                )
            )
        elif feature_type == "crosswalk":
            _append_polyline(
                samples,
                feature.id,
                CROSSWALK_TYPE,
                feature.crosswalk.polygon,
                closed=True,
            )
        elif feature_type == "speed_bump":
            _append_polyline(
                samples,
                feature.id,
                SPEED_BUMP_TYPE,
                feature.speed_bump.polygon,
                closed=True,
            )

    for idx, (xyz, direction, feature_id, element_type) in enumerate(
        samples[:max_num_rg_points]
    ):
        arrays["roadgraph_samples/xyz"][idx] = xyz
        arrays["roadgraph_samples/dir"][idx] = direction
        arrays["roadgraph_samples/id"][idx, 0] = feature_id
        arrays["roadgraph_samples/type"][idx, 0] = element_type
        arrays["roadgraph_samples/valid"][idx, 0] = 1
    return len(samples)


def _fill_traffic_lights(
    arrays: dict[str, np.ndarray], scenario: Any, num_tls: int
) -> None:
    """Fill traffic light state arrays for all time windows."""
    slices = _time_slices(scenario.current_time_index)
    for prefix, timesteps in slices.items():
        for time_slot, timestep in enumerate(timesteps):
            arrays[f"traffic_light_state/{prefix}/timestamp_micros"][time_slot] = (
                _timestamp_micros(scenario, timestep)
            )
            if timestep < 0 or timestep >= len(scenario.dynamic_map_states):
                continue
            lane_states = scenario.dynamic_map_states[timestep].lane_states[:num_tls]
            for tl_slot, lane_state in enumerate(lane_states):
                arrays[f"traffic_light_state/{prefix}/state"][
                    time_slot, tl_slot
                ] = lane_state.state
                arrays[f"traffic_light_state/{prefix}/valid"][time_slot, tl_slot] = 1
                arrays[f"traffic_light_state/{prefix}/id"][
                    time_slot, tl_slot
                ] = lane_state.lane
                arrays[f"traffic_light_state/{prefix}/x"][
                    time_slot, tl_slot
                ] = lane_state.stop_point.x
                arrays[f"traffic_light_state/{prefix}/y"][
                    time_slot, tl_slot
                ] = lane_state.stop_point.y
                arrays[f"traffic_light_state/{prefix}/z"][
                    time_slot, tl_slot
                ] = lane_state.stop_point.z


def scenario_to_tfexample(
    scenario: Any,
    max_num_objects: int = 128,
    max_num_rg_points: int = 30000,
    num_tls: int = 16,
) -> tuple[tf.train.Example, dict[str, Any]]:
    """Convert a Scenario proto to a tf.train.Example and return conversion stats."""
    arrays = _init_arrays(
        max_num_objects=max_num_objects,
        max_num_rg_points=max_num_rg_points,
        num_tls=num_tls,
    )
    selected_indices = _selected_track_indices(scenario, max_num_objects)
    _fill_tracks(arrays, scenario, selected_indices)
    roadgraph_count = _fill_roadgraph(arrays, scenario, max_num_rg_points)
    _fill_traffic_lights(arrays, scenario, num_tls)
    example = _example_from_arrays(arrays)
    example.features.feature["scenario/id"].bytes_list.value.append(
        scenario.scenario_id.encode()
    )
    return example, {
        "objects": len(selected_indices),
        "roadgraph_samples": min(roadgraph_count, max_num_rg_points),
    }


def default_cache_path(
    source_tfrecord: Path | str,
    scenario_index: int,
    max_num_objects: int,
    max_num_rg_points: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Return the default output path for a preprocessed TFRecord cache file."""
    source = Path(source_tfrecord)
    name = f"{source.stem}.scenario_{scenario_index:06d}.objects_{max_num_objects}.rg_{max_num_rg_points}.tfrecord"
    return Path(cache_dir) / name


def preprocess_scenario_to_tfexample(
    source_tfrecord: Path | str,
    output_tfrecord: Path | str | None = None,
    scenario_index: int = 0,
    max_num_objects: int = MAX_NUM_OBJECTS,
    max_num_rg_points: int = MAX_NUM_RG_POINTS,
    num_tls: int = NUM_TLS,
    overwrite: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Read one scenario from a TFRecord, convert it, and write to output_tfrecord."""
    source_tfrecord = Path(source_tfrecord)

    # prepare output path
    if output_tfrecord is None:
        output_tfrecord = default_cache_path(
            source_tfrecord,
            scenario_index,
            max_num_objects,
            max_num_rg_points,
        )
    output_tfrecord = Path(output_tfrecord)

    # return cache if exists
    if output_tfrecord.exists() and not overwrite:
        return output_tfrecord, {"cached": True}

    scenario = _read_scenario_at_index(source_tfrecord, scenario_index)
    example, stats = scenario_to_tfexample(
        scenario,
        max_num_objects=max_num_objects,
        max_num_rg_points=max_num_rg_points,
        num_tls=num_tls,
    )

    output_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(output_tfrecord)) as writer:
        writer.write(example.SerializeToString())
    stats.update({"cached": False, "scenario_id": scenario.scenario_id})
    return output_tfrecord, stats


def preprocess_serialized_scenario_to_tfexample(
    source_tfrecord: Path | str,
    serialized_scenario: bytes | Any,
    output_tfrecord: Path | str | None = None,
    scenario_index: int = 0,
    max_num_objects: int = 128,
    max_num_rg_points: int = 30000,
    num_tls: int = 16,
    overwrite: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Convert an already-serialized Scenario proto bytes and write to output_tfrecord."""
    source_tfrecord = Path(source_tfrecord)
    if output_tfrecord is None:
        output_tfrecord = default_cache_path(
            source_tfrecord, scenario_index, max_num_objects, max_num_rg_points
        )
    output_tfrecord = Path(output_tfrecord)
    if output_tfrecord.exists() and not overwrite:
        return output_tfrecord, {"cached": True}

    if hasattr(serialized_scenario, "numpy"):
        serialized_scenario = serialized_scenario.numpy()
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(bytes(serialized_scenario))
    if not scenario.scenario_id:
        raise PreprocessError(
            f"Record {scenario_index} did not parse as a raw Waymo Scenario protobuf."
        )
    example, stats = scenario_to_tfexample(
        scenario,
        max_num_objects=max_num_objects,
        max_num_rg_points=max_num_rg_points,
        num_tls=num_tls,
    )
    output_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(output_tfrecord)) as writer:
        writer.write(example.SerializeToString())
    stats.update({"cached": False, "scenario_id": scenario.scenario_id})
    return output_tfrecord, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert one raw Waymo Scenario TFRecord record to a Waymax TFExample TFRecord."
    )
    parser.add_argument("source_tfrecord", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scenario-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: convert one scenario and print the output path."""
    args = parse_args(argv)
    try:
        output, stats = preprocess_scenario_to_tfexample(
            source_tfrecord=args.source_tfrecord,
            output_tfrecord=args.output,
            scenario_index=args.scenario_index,
        )
    except PreprocessError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print(f"Wrote {output}")
    print(f"Stats: {stats}")
    return


def test_sum_scenarios() -> None:
    """Count and print the number of scenarios in each training TFRecord file."""
    data_root_dir = "/mnt/disk/data/public/waymo/motion_v_1_3_1/scenario/training/"
    tfrecord_paths = sorted(Path(data_root_dir).glob("*"))
    count_list = []
    for path in tfrecord_paths:
        count = sum(1 for _ in tf.data.TFRecordDataset([str(path)]))
        print(f"{path.name}: {count} scenarios")
        count_list.append(count)

    print(f"Total scenarios: {sum(count_list)}")


if __name__ == "__main__":
    test_sum_scenarios()
    # raise SystemExit(main())
