from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import tensorflow as tf

from waymax_interface.tfrecord_preprocess import (
    CURRENT_STEPS,
    FUTURE_STEPS,
    MAX_NUM_OBJECTS,
    MAX_NUM_RG_POINTS,
    NUM_TLS,
    PAST_STEPS,
    PreprocessError,
    _example_from_arrays,
    _init_arrays,
)

DEFAULT_CACHE_DIR = Path("waymax_interface/cache/tfexample")
CURRENT_TIME_INDEX = PAST_STEPS
TOTAL_STEPS = PAST_STEPS + CURRENT_STEPS + FUTURE_STEPS

OBJECT_TYPE_BY_NAME = {
    "vehicle": 1,
    "pedestrian": 2,
    "cyclist": 3,
}

ROAD_TYPE_BY_NAME = {
    "lane": 2,
    "road_line": 7,
    "road_edge": 15,
    "stop_sign": 17,
    "crosswalk": 18,
    "speed_bump": 19,
    "driveway": 20,
}


def _time_slices() -> dict[str, list[int]]:
    """Return WOMD/Waymax timestep windows for ScenarioMax 91-frame JSON."""
    return {
        "past": list(range(CURRENT_TIME_INDEX - PAST_STEPS, CURRENT_TIME_INDEX)),
        "current": [CURRENT_TIME_INDEX],
        "future": list(range(CURRENT_TIME_INDEX + 1, TOTAL_STEPS)),
    }


def _timestamp_micros(timestep: int) -> int:
    """Return a synthetic 10 Hz timestamp for the given 91-frame timestep."""
    if timestep < 0 or timestep >= TOTAL_STEPS:
        return -1
    return int(timestep * 100_000)


def _is_sentinel(value: float) -> bool:
    """Return true for ScenarioMax padded sentinel coordinates."""
    return not math.isfinite(value) or abs(value) >= 9999.0


def _json_point_xyz(point: dict[str, Any] | None) -> tuple[float, float, float]:
    if not point:
        return (0.0, 0.0, 0.0)
    return (
        float(point.get("x", 0.0)),
        float(point.get("y", 0.0)),
        float(point.get("z", 0.0)),
    )


def _state_is_valid(obj: dict[str, Any], timestep: int) -> bool:
    positions = obj.get("position", [])
    valids = obj.get("valid", [])
    if timestep < 0 or timestep >= len(positions):
        return False
    if timestep < len(valids) and not bool(valids[timestep]):
        return False
    x, y, z = _json_point_xyz(positions[timestep])
    return not (_is_sentinel(x) or _is_sentinel(y) or _is_sentinel(z))


def _object_type(obj: dict[str, Any]) -> int:
    obj_type = obj.get("type", 1)
    if isinstance(obj_type, str):
        return OBJECT_TYPE_BY_NAME.get(obj_type, 3)
    return int(obj_type)


def _selected_object_indices(data: dict[str, Any], max_num_objects: int) -> list[int]:
    objects = data.get("objects", [])
    metadata = data.get("metadata", {})
    sdc_index = int(metadata.get("sdc_track_index", -1))
    if not (0 <= sdc_index < len(objects)):
        raise PreprocessError(
            f"Invalid metadata.sdc_track_index {sdc_index} for {len(objects)} objects."
        )

    ttp_indices = [
        int(track.get("track_index", -1))
        for track in metadata.get("tracks_to_predict", [])
    ]
    ttp_indices = sorted(
        [idx for idx in ttp_indices if 0 <= idx < len(objects)],
        key=lambda idx: int(objects[idx].get("id", idx)),
    )
    priority = [*ttp_indices, sdc_index, *range(len(objects))]

    selected = []
    seen = set()
    for idx in priority:
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) == max_num_objects:
            break
    return selected


def _fill_json_objects(
    arrays: dict[str, np.ndarray],
    data: dict[str, Any],
    selected_indices: list[int],
) -> None:
    objects = data.get("objects", [])
    metadata = data.get("metadata", {})
    sdc_index = int(metadata.get("sdc_track_index", -1))
    tracks_to_predict = {
        int(track.get("track_index", -1)): int(track.get("difficulty", 0))
        for track in metadata.get("tracks_to_predict", [])
    }
    objects_of_interest = set(metadata.get("objects_of_interest", []))

    for slot, object_index in enumerate(selected_indices):
        obj = objects[object_index]
        obj_id = int(obj.get("id", object_index))
        arrays["state/id"][slot] = float(obj_id)
        arrays["state/type"][slot] = float(_object_type(obj))
        arrays["state/is_sdc"][slot] = 1 if object_index == sdc_index else 0
        arrays["state/tracks_to_predict"][slot] = (
            1 if object_index in tracks_to_predict else 0
        )
        arrays["state/objects_of_interest"][slot] = (
            1 if obj_id in objects_of_interest else 0
        )
        arrays["state/difficulty_level"][slot] = tracks_to_predict.get(object_index, 0)

        positions = obj.get("position", [])
        velocities = obj.get("velocity", [])
        headings = obj.get("heading", [])
        width = float(obj.get("width", 0.0))
        length = float(obj.get("length", 0.0))
        height = float(obj.get("height", 0.0))

        for prefix, timesteps in _time_slices().items():
            for time_slot, timestep in enumerate(timesteps):
                if not _state_is_valid(obj, timestep):
                    continue

                x, y, z = _json_point_xyz(positions[timestep])
                vx, vy, _ = _json_point_xyz(
                    velocities[timestep] if timestep < len(velocities) else None
                )
                heading = (
                    float(headings[timestep]) if timestep < len(headings) else 0.0
                )
                speed = math.hypot(vx, vy)
                vel_yaw = math.atan2(vy, vx) if speed > 1e-6 else heading

                arrays[f"state/{prefix}/valid"][slot, time_slot] = 1
                arrays[f"state/{prefix}/timestamp_micros"][
                    slot, time_slot
                ] = _timestamp_micros(timestep)
                arrays[f"state/{prefix}/bbox_yaw"][slot, time_slot] = heading
                arrays[f"state/{prefix}/height"][slot, time_slot] = height
                arrays[f"state/{prefix}/length"][slot, time_slot] = length
                arrays[f"state/{prefix}/width"][slot, time_slot] = width
                arrays[f"state/{prefix}/speed"][slot, time_slot] = speed
                arrays[f"state/{prefix}/vel_yaw"][slot, time_slot] = vel_yaw
                arrays[f"state/{prefix}/velocity_x"][slot, time_slot] = vx
                arrays[f"state/{prefix}/velocity_y"][slot, time_slot] = vy
                arrays[f"state/{prefix}/x"][slot, time_slot] = x
                arrays[f"state/{prefix}/y"][slot, time_slot] = y
                arrays[f"state/{prefix}/z"][slot, time_slot] = z


def _road_type(road: dict[str, Any]) -> int:
    if "map_element_id" in road:
        return int(road["map_element_id"])
    road_type = road.get("type", 0)
    if isinstance(road_type, str):
        return ROAD_TYPE_BY_NAME.get(road_type, 20)
    return int(road_type)


def _direction(
    points: list[dict[str, Any]], idx: int, closed: bool = False
) -> tuple[float, float, float]:
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
    vec = np.asarray(_json_point_xyz(p1), dtype=np.float32) - np.asarray(
        _json_point_xyz(p0), dtype=np.float32
    )
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return (0.0, 0.0, 0.0)
    return tuple((vec / norm).tolist())


def _valid_geometry(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for point in points:
        x, y, z = _json_point_xyz(point)
        if _is_sentinel(x) or _is_sentinel(y) or _is_sentinel(z):
            continue
        valid.append(point)
    return valid


def _fill_json_roadgraph(
    arrays: dict[str, np.ndarray], data: dict[str, Any], max_num_rg_points: int
) -> int:
    samples = []
    for road in data.get("roads", []):
        geometry = _valid_geometry(road.get("geometry", []))
        if not geometry:
            continue
        element_type = _road_type(road)
        feature_id = int(road.get("id", road.get("map_element_id", 0)))
        closed = str(road.get("type", "")) in {"crosswalk", "speed_bump", "driveway"}
        for idx, point in enumerate(geometry):
            samples.append(
                (
                    _json_point_xyz(point),
                    _direction(geometry, idx, closed=closed),
                    feature_id,
                    element_type,
                )
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


def _fill_empty_traffic_lights(arrays: dict[str, np.ndarray]) -> None:
    for prefix, timesteps in _time_slices().items():
        for time_slot, timestep in enumerate(timesteps):
            arrays[f"traffic_light_state/{prefix}/timestamp_micros"][
                time_slot
            ] = _timestamp_micros(timestep)


def scenariomax_json_to_tfexample(
    data: dict[str, Any],
    max_num_objects: int = MAX_NUM_OBJECTS,
    max_num_rg_points: int = MAX_NUM_RG_POINTS,
    num_tls: int = NUM_TLS,
) -> tuple[tf.train.Example, dict[str, Any]]:
    arrays = _init_arrays(
        max_num_objects=max_num_objects,
        max_num_rg_points=max_num_rg_points,
        num_tls=num_tls,
    )
    selected_indices = _selected_object_indices(data, max_num_objects)
    _fill_json_objects(arrays, data, selected_indices)
    roadgraph_count = _fill_json_roadgraph(arrays, data, max_num_rg_points)
    _fill_empty_traffic_lights(arrays)

    example = _example_from_arrays(arrays)
    scenario_id = str(data.get("scenario_id", data.get("name", "")))
    example.features.feature["scenario/id"].bytes_list.value.append(
        scenario_id.encode()
    )
    return example, {
        "objects": len(selected_indices),
        "roadgraph_samples": min(roadgraph_count, max_num_rg_points),
        "scenario_id": scenario_id,
    }


def default_json_cache_path(
    source_json: Path | str,
    max_num_objects: int,
    max_num_rg_points: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    source = Path(source_json)
    return (
        Path(cache_dir)
        / f"{source.stem}.json.objects_{max_num_objects}.rg_{max_num_rg_points}.tfrecord"
    )


def preprocess_json_to_tfexample(
    source_json: Path | str,
    output_tfrecord: Path | str | None = None,
    max_num_objects: int = MAX_NUM_OBJECTS,
    max_num_rg_points: int = MAX_NUM_RG_POINTS,
    num_tls: int = NUM_TLS,
    overwrite: bool = True,
) -> tuple[Path, dict[str, Any]]:
    source_json = Path(source_json)
    if not source_json.is_file():
        raise PreprocessError(f"ScenarioMax JSON not found: {source_json}")
    if output_tfrecord is None:
        output_tfrecord = default_json_cache_path(
            source_json, max_num_objects, max_num_rg_points
        )
    output_tfrecord = Path(output_tfrecord)
    if output_tfrecord.exists() and not overwrite:
        return output_tfrecord, {"cached": True}

    with source_json.open("r") as f:
        data = json.load(f)
    example, stats = scenariomax_json_to_tfexample(
        data,
        max_num_objects=max_num_objects,
        max_num_rg_points=max_num_rg_points,
        num_tls=num_tls,
    )

    output_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(output_tfrecord)) as writer:
        writer.write(example.SerializeToString())
    stats["cached"] = False
    return output_tfrecord, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one ScenarioMax/PufferDrive JSON to a Waymax TFExample TFRecord."
    )
    parser.add_argument("source_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        output, stats = preprocess_json_to_tfexample(
            source_json=args.source_json,
            output_tfrecord=args.output,
            overwrite=not args.no_overwrite,
        )
    except PreprocessError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print(f"Wrote {output}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
