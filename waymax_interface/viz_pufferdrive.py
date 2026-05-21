# Visualization of the PufferDrive raw model input and selected output action.
from pathlib import Path
import os
import subprocess
from typing import Any, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np


EGO_FEATURES_CLASSIC = 8
MAX_PARTNER_OBJECTS = 31
MAX_ROAD_OBJECTS = 128
PARTNER_FEATURES = 7
ROAD_FEATURES = 7

MAX_SPEED = 100.0
MAX_VEH_WIDTH = 15.0
MAX_VEH_LEN = 30.0
MAX_ROAD_SEGMENT_LENGTH = 100.0

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

ROAD_COLORS = {
    0: "#9aa0a6",  # lane-like features
    1: "#f2c94c",  # road lines
    2: "#56ccf2",  # road edge
    3: "#eb5757",
    4: "#bb6bd9",
    5: "#6fcf97",
    6: "#bdbdbd",
}


class VizPufferDrive:
    def __init__(self, fps: int = 10, dpi: int = 120):
        self.input_data_buffer: List[np.ndarray] = []
        self.output_data_buffer: List[Any] = []
        self.fps = fps
        self.dpi = dpi

    def add_input(self, input_data):
        self.input_data_buffer.append(np.asarray(input_data, dtype=np.float32).copy())

    def add_output(self, output_data):
        self.output_data_buffer.append(np.asarray(output_data).copy())

    def create_video(self, video_path: str = "./pufferdrive_vis.mp4"):
        """Render a side-by-side model-input/model-output video with ffmpeg."""
        if len(self.input_data_buffer) == 0:
            print("No data to create video")
            return

        frame_count = len(self.input_data_buffer)
        if len(self.output_data_buffer) != frame_count:
            print(
                "Only visualize the input data, since the output data is not complete"
            )

        frames = [
            self._draw_frame(
                self.input_data_buffer[idx],
                self.output_data_buffer[idx]
                if idx < len(self.output_data_buffer)
                else None,
                idx,
            )
            for idx in range(frame_count)
        ]
        self._write_mp4(frames, video_path)

    def draw_input(self, ax, input_data):
        """Draw one flattened PufferDrive observation on a matplotlib axis."""
        obs = np.asarray(input_data, dtype=np.float32).reshape(-1)
        self._setup_bev_axis(ax)

        goal_x = float(obs[0]) * 200.0
        goal_y = float(obs[1]) * 200.0
        speed = float(obs[2]) * MAX_SPEED
        ego_width = max(float(obs[3]) * MAX_VEH_WIDTH, 0.5)
        ego_length = max(float(obs[4]) * MAX_VEH_LEN, 1.0)

        self._draw_vehicle(ax, 0.0, 0.0, ego_length, ego_width, 0.0, "#2f80ed", 0.9)
        ax.arrow(
            0.0,
            0.0,
            5.0,
            0.0,
            width=0.18,
            head_width=1.2,
            head_length=1.4,
            color="#ffffff",
            length_includes_head=True,
            zorder=7,
        )

        ax.add_patch(Circle((goal_x, goal_y), radius=1.1, color="#eb5757", zorder=6))
        ax.add_patch(
            Circle(
                (goal_x, goal_y),
                radius=4.0,
                edgecolor="#eb5757",
                facecolor="none",
                linewidth=1.0,
                alpha=0.5,
                zorder=5,
            )
        )
        ax.plot([0.0, goal_x], [0.0, goal_y], color="#eb5757", alpha=0.25, linewidth=1)

        self._draw_partners(ax, obs)
        self._draw_roads(ax, obs)

        ax.text(
            0.01,
            0.99,
            f"ego speed {speed:.1f} m/s\n"
            f"goal ({goal_x:.1f}, {goal_y:.1f}) m\n"
            f"partners {self._count_partners(obs)} / roads {self._count_roads(obs)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="#f5f5f5",
            fontsize=9,
            bbox=dict(facecolor="#111111", alpha=0.72, edgecolor="none", pad=4),
        )

    def draw_output(self, ax, output_data):
        """Draw the selected discrete action as acceleration/steering controls."""
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#171717", edgecolor="none"))

        if output_data is None:
            ax.text(0.5, 0.5, "no output", color="#f5f5f5", ha="center", va="center")
            return

        action = int(np.asarray(output_data).reshape(-1)[0])
        num_steer = len(STEERING_VALUES)
        max_action = len(ACCELERATION_VALUES) * num_steer - 1
        action = max(0, min(action, max_action))
        accel_idx = action // num_steer
        steer_idx = action % num_steer
        acceleration = ACCELERATION_VALUES[accel_idx]
        steering = STEERING_VALUES[steer_idx]

        ax.text(
            0.08,
            0.92,
            "model output",
            color="#f5f5f5",
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        ax.text(0.08, 0.80, f"action index: {action}", color="#f5f5f5", fontsize=10)
        ax.text(0.08, 0.68, f"acceleration: {acceleration:+.3f}", color="#f5f5f5")
        ax.text(0.08, 0.58, f"steering: {steering:+.3f}", color="#f5f5f5")

        self._draw_discrete_bar(
            ax,
            ACCELERATION_VALUES,
            accel_idx,
            y=0.38,
            title="acceleration bins",
            color="#6fcf97",
        )
        self._draw_discrete_bar(
            ax,
            STEERING_VALUES,
            steer_idx,
            y=0.16,
            title="steering bins",
            color="#f2c94c",
        )

    def _draw_frame(
        self, input_data: np.ndarray, output_data: Any | None, frame_idx: int
    ) -> np.ndarray:
        fig, (input_ax, output_ax) = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            gridspec_kw={"width_ratios": [3.2, 1.0]},
            dpi=self.dpi,
        )
        fig.patch.set_facecolor("#171717")
        self.draw_input(input_ax, input_data)
        self.draw_output(output_ax, output_data)
        fig.suptitle(
            f"PufferDrive model view - step {frame_idx}",
            color="#f5f5f5",
            fontsize=13,
        )
        fig.tight_layout(pad=1.2)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape((height, width, 4))[..., :3].copy()
        plt.close(fig)
        return frame

    def _setup_bev_axis(self, ax) -> None:
        ax.set_facecolor("#202124")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-55, 75)
        ax.set_ylim(-55, 55)
        ax.grid(True, color="#3c4043", linewidth=0.5, alpha=0.8)
        ax.axhline(0, color="#5f6368", linewidth=0.8)
        ax.axvline(0, color="#5f6368", linewidth=0.8)
        ax.set_xlabel("forward relative x (m)", color="#f5f5f5")
        ax.set_ylabel("left relative y (m)", color="#f5f5f5")
        ax.tick_params(colors="#f5f5f5", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#5f6368")

    def _draw_partners(self, ax, obs: np.ndarray) -> None:
        start = EGO_FEATURES_CLASSIC
        for idx in range(MAX_PARTNER_OBJECTS):
            row = obs[start + idx * PARTNER_FEATURES : start + (idx + 1) * PARTNER_FEATURES]
            if row.size < PARTNER_FEATURES or self._is_empty_position(row[0], row[1]):
                continue
            x = float(row[0]) * 50.0
            y = float(row[1]) * 50.0
            width = max(float(row[2]) * MAX_VEH_WIDTH, 0.4)
            length = max(float(row[3]) * MAX_VEH_LEN, 0.8)
            yaw = float(np.arctan2(row[5], row[4]))
            speed = float(row[6]) * MAX_SPEED
            color = "#f2994a" if abs(speed) > 0.5 else "#c97b37"
            self._draw_vehicle(ax, x, y, length, width, yaw, color, 0.75)
            ax.arrow(
                x,
                y,
                np.cos(yaw) * 3.0,
                np.sin(yaw) * 3.0,
                width=0.08,
                head_width=0.8,
                head_length=0.9,
                color="#ffffff",
                alpha=0.8,
                length_includes_head=True,
                zorder=6,
            )

    def _draw_roads(self, ax, obs: np.ndarray) -> None:
        start = EGO_FEATURES_CLASSIC + MAX_PARTNER_OBJECTS * PARTNER_FEATURES
        for idx in range(MAX_ROAD_OBJECTS):
            row = obs[start + idx * ROAD_FEATURES : start + (idx + 1) * ROAD_FEATURES]
            if row.size < ROAD_FEATURES or self._is_empty_position(row[0], row[1]):
                continue
            x = float(row[0]) * 50.0
            y = float(row[1]) * 50.0
            length = max(float(row[2]) * MAX_ROAD_SEGMENT_LENGTH, 0.1)
            yaw = float(np.arctan2(row[5], row[4]))
            half = 0.5 * length
            dx = np.cos(yaw) * half
            dy = np.sin(yaw) * half
            road_type = int(round(float(row[6])))
            color = ROAD_COLORS.get(road_type, "#bdbdbd")
            ax.plot(
                [x - dx, x + dx],
                [y - dy, y + dy],
                color=color,
                linewidth=1.4 if road_type == 2 else 0.9,
                alpha=0.85,
                zorder=1,
            )

    def _draw_vehicle(
        self,
        ax,
        x: float,
        y: float,
        length: float,
        width: float,
        yaw: float,
        color: str,
        alpha: float,
    ) -> None:
        half_l = 0.5 * length
        half_w = 0.5 * width
        corners = np.asarray(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ],
            dtype=np.float32,
        )
        rot = np.asarray(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float32,
        )
        corners = corners @ rot.T + np.asarray([x, y], dtype=np.float32)
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor=color,
                edgecolor="#f5f5f5",
                linewidth=0.8,
                alpha=alpha,
                zorder=5,
            )
        )

    def _draw_discrete_bar(
        self, ax, values: tuple[float, ...], selected_idx: int, y: float, title: str, color: str
    ) -> None:
        ax.text(0.08, y + 0.11, title, color="#f5f5f5", fontsize=9)
        left = 0.08
        width = 0.84 / len(values)
        for idx, value in enumerate(values):
            face = color if idx == selected_idx else "#3c4043"
            ax.add_patch(
                Rectangle(
                    (left + idx * width, y),
                    width * 0.82,
                    0.08,
                    facecolor=face,
                    edgecolor="#5f6368",
                    linewidth=0.5,
                )
            )
            if idx == selected_idx:
                ax.text(
                    left + idx * width + width * 0.41,
                    y - 0.035,
                    f"{value:+.2f}",
                    color="#f5f5f5",
                    fontsize=7,
                    ha="center",
                )

    def _count_partners(self, obs: np.ndarray) -> int:
        start = EGO_FEATURES_CLASSIC
        return sum(
            not self._is_empty_position(
                obs[start + idx * PARTNER_FEATURES],
                obs[start + idx * PARTNER_FEATURES + 1],
            )
            for idx in range(MAX_PARTNER_OBJECTS)
        )

    def _count_roads(self, obs: np.ndarray) -> int:
        start = EGO_FEATURES_CLASSIC + MAX_PARTNER_OBJECTS * PARTNER_FEATURES
        return sum(
            not self._is_empty_position(
                obs[start + idx * ROAD_FEATURES],
                obs[start + idx * ROAD_FEATURES + 1],
            )
            for idx in range(MAX_ROAD_OBJECTS)
        )

    def _is_empty_position(self, x: float, y: float) -> bool:
        return abs(float(x)) < 1e-8 and abs(float(y)) < 1e-8

    def _write_mp4(self, frames: list[np.ndarray], video_path: str | Path) -> None:
        if not frames:
            raise ValueError("No frames to write")

        output = Path(video_path)
        output.parent.mkdir(parents=True, exist_ok=True)
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
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        payload = bytearray()
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All video frames must have the same size")
            payload.extend(np.ascontiguousarray(frame).tobytes())

        process = subprocess.run(
            command,
            input=bytes(payload),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed while writing {output}:\n{stderr}")
