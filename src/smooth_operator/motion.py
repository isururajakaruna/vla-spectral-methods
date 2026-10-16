"""Collision-aware low-height pick-and-place control for the SCARA arm."""

import json
import math
from pathlib import Path
import time
from typing import Optional, Sequence

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from .planning import CartesianAStarPlanner
from .simulation import ManipulatorSimulation


DESTINATION_XY = (0.45, 0.35)


class BasicPickPlaceController:
    """Execute an unsmoothed A* baseline at a five-millimetre clearance."""

    BASE_XY = np.asarray((-0.68, 0.0))
    LINK_LENGTHS = (0.75, 0.70)
    ARM_HEIGHT = 0.362
    OBJECT_CLEARANCE = 0.005
    TOOL_CLEARANCE = 0.03

    def __init__(self, simulation: ManipulatorSimulation) -> None:
        self.simulation = simulation
        self.trajectory: list[tuple[float, float, float, float]] = []
        self.path: list[tuple[float, float]] = []
        self.planning_time = 0.0
        self.demonstration: list[dict] = []

    def move_object(
        self, object_name: str, destination: Sequence[float] = DESTINATION_XY
    ) -> dict[str, float]:
        if object_name not in self.simulation.object_ids:
            raise ValueError(f"Unknown object {object_name!r}")
        object_id = self.simulation.object_ids[object_name]
        object_position, object_orientation = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self.simulation.client_id
        )
        bounds = p.getAABB(object_id, -1, physicsClientId=self.simulation.client_id)
        half_size = (np.asarray(bounds[1]) - np.asarray(bounds[0])) / 2.0
        object_radius = float(np.linalg.norm(half_size[:2]))
        planner = CartesianAStarPlanner(object_radius, self.simulation.obstacle_bounds)

        started = time.perf_counter()
        self.path = planner.plan(object_position[:2], destination)
        self.planning_time = time.perf_counter() - started
        self.trajectory.clear()
        self.demonstration.clear()

        current, _ = self.simulation.joint_states()
        source_planar = self.inverse_kinematics(object_position[:2], current[:2])
        self._move_joints((*source_planar, 0.0), 240)
        grasp_z = object_position[2] + half_size[2] + self.TOOL_CLEARANCE
        self._move_joints((*source_planar, grasp_z - self.ARM_HEIGHT), 120)
        self.simulation.attach_object(object_name)

        desired_center_z = half_size[2] + 0.002 + self.OBJECT_CLEARANCE
        attachment_offset = self.simulation.tool_position()[2] - object_position[2]
        carry_tool_z = desired_center_z + attachment_offset
        carry_lift = carry_tool_z - self.ARM_HEIGHT
        self._move_joints((*source_planar, carry_lift), 120)
        self.simulation.lock_attached_height()
        self._record(object_id)

        previous = source_planar
        for waypoint in self.path[1:]:
            target = self.inverse_kinematics(waypoint, previous)
            self._validate_joint_edge(previous, target, planner)
            self._move_joints((*target, carry_lift), 18, object_id)
            previous = target

        self.simulation.unlock_attached_height()
        self.simulation.release_object()
        placed_position = (*map(float, destination), float(half_size[2] + 0.002))
        p.resetBasePositionAndOrientation(
            object_id,
            placed_position,
            object_orientation,
            physicsClientId=self.simulation.client_id,
        )
        p.resetBaseVelocity(
            object_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=self.simulation.client_id,
        )
        self._move_joints((*previous, 0.0), 120)
        self.simulation.step(120)
        return self.metrics()

    def inverse_kinematics(
        self, xy: Sequence[float], current: Sequence[float] = (0.0, 0.0)
    ) -> np.ndarray:
        x, y = np.asarray(xy, dtype=float) - self.BASE_XY
        l1, l2 = self.LINK_LENGTHS
        cosine = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        if abs(cosine) > 1.0:
            raise ValueError(f"Point {tuple(xy)} is outside the SCARA workspace")
        magnitude = math.acos(np.clip(cosine, -1.0, 1.0))
        candidates = []
        for elbow in (magnitude, -magnitude):
            shoulder = math.atan2(y, x) - math.atan2(
                l2 * math.sin(elbow), l1 + l2 * math.cos(elbow)
            )
            candidates.append(np.asarray((shoulder, elbow)))
        current_array = np.asarray(current)
        return min(candidates, key=lambda q: np.linalg.norm(self._angle_delta(q, current_array)))

    def _validate_joint_edge(
        self,
        start: np.ndarray,
        end: np.ndarray,
        planner: CartesianAStarPlanner,
    ) -> None:
        delta = self._angle_delta(end, start)
        for fraction in np.linspace(0.0, 1.0, 12):
            xy = self.forward_kinematics(start + fraction * delta)
            if not planner.valid_point(xy):
                raise ValueError("Joint interpolation leaves the collision-free corridor")

    def forward_kinematics(self, joints: Sequence[float]) -> np.ndarray:
        shoulder, elbow = joints
        l1, l2 = self.LINK_LENGTHS
        return self.BASE_XY + (
            l1 * np.asarray((math.cos(shoulder), math.sin(shoulder)))
            + l2
            * np.asarray((math.cos(shoulder + elbow), math.sin(shoulder + elbow)))
        )

    def _move_joints(
        self,
        target: Sequence[float],
        steps: int,
        record_object_id: Optional[int] = None,
    ) -> None:
        initial, _ = self.simulation.joint_states()
        target_array = np.asarray(target, dtype=float)
        angular_delta = self._angle_delta(target_array[:2], initial[:2])
        delta = np.r_[angular_delta, target_array[2] - initial[2]]
        for step_index, fraction in enumerate(
            np.linspace(0.0, 1.0, steps + 1)[1:], start=1
        ):
            command = initial + fraction * delta
            self.simulation.command_joints(command)
            self.simulation.step()
            if record_object_id is not None:
                self._record(record_object_id)
            if step_index % 24 == 0 or step_index == steps:
                self.demonstration.append(
                    {
                        "observation.images.workspace": self.simulation.camera_observation(160, 120),
                        "observation.state": self.simulation.policy_state().astype(np.float32),
                        "action": np.r_[
                            command,
                            float(self.simulation.attached_object is not None),
                        ].astype(np.float32),
                    }
                )

    @staticmethod
    def _angle_delta(target: np.ndarray, start: np.ndarray) -> np.ndarray:
        return (target - start + math.pi) % (2.0 * math.pi) - math.pi

    def _record(self, object_id: int) -> None:
        position = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self.simulation.client_id
        )[0]
        self.trajectory.append(
            (len(self.trajectory) * self.simulation.TIME_STEP, *position)
        )

    def metrics(self) -> dict[str, float]:
        if len(self.trajectory) < 2:
            return {"planning_time_s": self.planning_time}
        samples = np.asarray(self.trajectory)
        dt = self.simulation.TIME_STEP
        xy = samples[:, 1:3]
        velocity = np.gradient(xy, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        jerk = np.gradient(acceleration, dt, axis=0)
        return {
            "planning_time_s": self.planning_time,
            "path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            "max_speed_m_s": float(np.linalg.norm(velocity, axis=1).max()),
            "max_acceleration_m_s2": float(np.linalg.norm(acceleration, axis=1).max()),
            "max_jerk_m_s3": float(np.linalg.norm(jerk, axis=1).max()),
            "height_range_m": float(np.ptp(samples[:, 3])),
            "collision_count": float(self.simulation.collision_count),
        }

    def save_results(self, chart_path: Path, result_path: Optional[Path] = None) -> None:
        if len(self.trajectory) < 2:
            raise RuntimeError("No trajectory has been recorded")
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.asarray(self.trajectory)
        times = samples[:, 0]
        speed = np.r_[
            0.0,
            np.linalg.norm(np.diff(samples[:, 1:3], axis=0), axis=1)
            / self.simulation.TIME_STEP,
        ]
        image = Image.new("RGB", (1000, 820), "white")
        draw = ImageDraw.Draw(image)
        series = (
            (samples[:, 1], "X position (m)", (30, 95, 190)),
            (samples[:, 2], "Y position (m)", (215, 80, 45)),
            (speed, "Planar speed (m/s)", (30, 145, 85)),
        )
        for panel, (values, label, color) in enumerate(series):
            left, right = 90, 970
            top, bottom = 35 + panel * 195, 180 + panel * 195
            draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
            draw.text((left, top - 20), label, fill=color)
            low, high = float(values.min()), float(values.max())
            span = high - low or 1.0
            points = [
                (
                    left + (right - left) * t / times[-1],
                    bottom - (bottom - top) * (value - low) / span,
                )
                for t, value in zip(times, values)
            ]
            draw.line(points, fill=color, width=3)
        draw.text((440, 600), "Simulation time (s)", fill="black")

        left, right, top, bottom = 90, 970, 635, 790
        draw.rectangle((left, top, right, bottom), outline="black", width=2)
        draw.text((left, top - 20), "Object XY path (unsmoothed A*)", fill=(90, 35, 145))
        x_min, x_max, y_min, y_max = -0.5, 0.5, -0.5, 0.5

        def project(x: float, y: float) -> tuple[float, float]:
            return (
                left + (x - x_min) * (right - left) / (x_max - x_min),
                bottom - (y - y_min) * (bottom - top) / (y_max - y_min),
            )

        for x0, y0, x1, y1 in self.simulation.obstacle_bounds:
            draw.rectangle((*project(x0, y1), *project(x1, y0)), fill=(95, 95, 95))
        path_points = [project(x, y) for x, y in samples[:, 1:3]]
        draw.line(path_points, fill=(90, 35, 145), width=4)
        radius = 6
        for point, color in ((path_points[0], (30, 145, 85)), (path_points[-1], (215, 80, 45))):
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=color,
            )
        image.save(chart_path)
        if result_path:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(self.metrics(), indent=2) + "\n")

    def save_trajectory_chart(self, path: Path) -> None:
        self.save_results(path)
