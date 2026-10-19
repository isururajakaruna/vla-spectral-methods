"""SO-101 simulation, planar A* controller, and zero-shot policy execution."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from .planning import CartesianAStarPlanner
from .simulation import ASSET_DIR, ManipulatorSimulation


SO101_ASSET_DIR = ASSET_DIR / "so101"


class SO101Simulation(ManipulatorSimulation):
    """Official SO-101 geometry in a workspace scaled to its physical reach."""

    JOINT_COUNT = 6
    ROBOT_BASE = (-0.18, 0.0, 0.002)
    ENVIRONMENT_SCALE = 0.4
    WORKSPACE = (-0.24, -0.20, 0.24, 0.20)
    DEFAULT_DESTINATION = (0.16, 0.12)
    JOINT_NAMES = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    # The base checkpoint contains several physical-arm calibrations.  This
    # canonical centre maps the URDF zero pose into the dominant `so100` frame.
    POLICY_ZERO_DEGREES = np.asarray(
        (1.595914, 119.944237, 109.769775, 56.705513, -27.422693, 12.002786)
    )
    POLICY_HZ = 30

    def __init__(self, gui: bool = False, training_seed: Optional[int] = None) -> None:
        super().__init__(gui=gui, training_seed=training_seed)
        self.tool_link_index = self._link_index("gripper_frame_link")
        scale = self.ENVIRONMENT_SCALE
        self.obstacle_bounds = [
            tuple(scale * value for value in bounds)
            for bounds in (
                (-0.34, 0.13, -0.22, 0.31),
                (0.23, -0.25, 0.37, -0.11),
            )
        ] if training_seed is None else self.obstacle_bounds
        self.goal_position = self.DEFAULT_DESTINATION

    def _load_assets(self) -> tuple[int, int]:
        environment_id = p.loadURDF(
            str(ASSET_DIR / "environment.urdf"),
            globalScaling=self.ENVIRONMENT_SCALE,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        robot_id = p.loadURDF(
            str(SO101_ASSET_DIR / "so101_new_calib.urdf"),
            basePosition=self.ROBOT_BASE,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id,
        )
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9,
                cameraYaw=45.0,
                cameraPitch=-35.0,
                cameraTargetPosition=(0.0, 0.0, 0.12),
                physicsClientId=self.client_id,
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id
            )
        return environment_id, robot_id

    def _load_objects(self) -> dict[str, int]:
        placements = {
            "apple": ("apple.urdf", (-0.18, -0.14, 0.024)),
            "coke_can": ("coke_can.urdf", (-0.10, -0.14, 0.038)),
            "pen": ("pen.urdf", (0.02, -0.13, 0.010)),
        }
        result = {
            name: p.loadURDF(
                str(ASSET_DIR / asset),
                basePosition=position,
                globalScaling=0.62,
                physicsClientId=self.client_id,
            )
            for name, (asset, position) in placements.items()
        }
        p.changeDynamics(
            result["pen"], -1, lateralFriction=1.0, rollingFriction=0.2,
            physicsClientId=self.client_id,
        )
        return result

    def _find_controlled_joints(self) -> tuple[int, ...]:
        by_name = {
            p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)[1]
            .decode("utf-8"): index
            for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id))
        }
        missing = [name for name in self.JOINT_NAMES if name not in by_name]
        if missing:
            raise RuntimeError(f"SO-101 URDF is missing joints: {missing}")
        return tuple(by_name[name] for name in self.JOINT_NAMES)

    def _link_index(self, name: str) -> int:
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
            if info[12].decode("utf-8") == name:
                return index
        raise RuntimeError(f"SO-101 URDF is missing link {name!r}")

    def _randomize_training_scene(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for link in self._environment_links("cell_markers"):
            p.changeVisualShape(
                self.environment_id, link, rgbaColor=(0, 0, 0, 0),
                physicsClientId=self.client_id,
            )
        for link in self.obstacle_link_indices:
            p.setCollisionFilterGroupMask(
                self.environment_id, link, 0, 0, physicsClientId=self.client_id
            )
            p.changeVisualShape(
                self.environment_id, link, rgbaColor=(0, 0, 0, 0),
                physicsClientId=self.client_id,
            )
        self.obstacle_bounds = []
        self.training_obstacle_ids = []
        for _ in range(int(rng.integers(1, 5))):
            for _attempt in range(100):
                width, depth = rng.uniform(0.035, 0.065, size=2)
                x, y = rng.uniform((-0.11, -0.12), (0.14, 0.12))
                bounds = (x - width / 2, y - depth / 2, x + width / 2, y + depth / 2)
                goal_x, goal_y = self.DEFAULT_DESTINATION
                if bounds[0] - 0.04 <= goal_x <= bounds[2] + 0.04 and bounds[1] - 0.04 <= goal_y <= bounds[3] + 0.04:
                    continue
                if any(
                    bounds[0] - 0.025 < other[2]
                    and bounds[2] + 0.025 > other[0]
                    and bounds[1] - 0.025 < other[3]
                    and bounds[3] + 0.025 > other[1]
                    for other in self.obstacle_bounds
                ):
                    continue
                collision = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=(width / 2, depth / 2, 0.045),
                    physicsClientId=self.client_id,
                )
                visual = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=(width / 2, depth / 2, 0.045),
                    rgbaColor=(0.45, 0.24, 0.10, 1), physicsClientId=self.client_id,
                )
                body = p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual, basePosition=(x, y, 0.046),
                    physicsClientId=self.client_id,
                )
                self.training_obstacle_ids.append(body)
                self.obstacle_bounds.append(tuple(map(float, bounds)))
                break
        cells = [
            (x, y)
            for x in np.arange(-0.18, 0.13, 0.06)
            for y in np.arange(-0.12, 0.13, 0.06)
            if not any(
                x0 - 0.07 <= x <= x1 + 0.07 and y0 - 0.07 <= y <= y1 + 0.07
                for x0, y0, x1, y1 in self.obstacle_bounds
            )
            and np.linalg.norm(np.asarray((x, y)) - self.DEFAULT_DESTINATION) > 0.06
            and np.linalg.norm(np.asarray((x, y)) - np.asarray(self.ROBOT_BASE[:2])) > 0.14
        ]
        rng.shuffle(cells)
        for body, xy in zip(self.object_ids.values(), cells):
            pose = p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)
            p.resetBasePositionAndOrientation(
                body, (xy[0], xy[1], pose[0][2]), pose[1], physicsClientId=self.client_id
            )

    def set_goal_marker(self, xy: Sequence[float]) -> None:
        if self.goal_marker_id is not None:
            p.removeBody(self.goal_marker_id, physicsClientId=self.client_id)
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=(0.019, 0.019, 0.001),
            rgbaColor=(0.10, 0.75, 0.20, 1), physicsClientId=self.client_id,
        )
        self.goal_marker_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=visual,
            basePosition=(float(xy[0]), float(xy[1]), 0.004),
            physicsClientId=self.client_id,
        )
        self.goal_position = (float(xy[0]), float(xy[1]))

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        info = [
            p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
            for index in self.joint_indices
        ]
        return np.asarray([item[8] for item in info]), np.asarray([item[9] for item in info])

    def simulator_to_policy(self, joints: Sequence[float]) -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.shape != (6,):
            raise ValueError("Expected six SO-101 simulator joints")
        return self.POLICY_ZERO_DEGREES + np.rad2deg(values)

    def policy_to_simulator(self, values: Sequence[float]) -> np.ndarray:
        policy = np.asarray(values, dtype=float)
        if policy.shape != (6,) or not np.all(np.isfinite(policy)):
            raise ValueError("Expected six finite SO-101 policy actions")
        return np.deg2rad(policy - self.POLICY_ZERO_DEGREES)

    def policy_state(self) -> np.ndarray:
        positions, _ = self.joint_states()
        return self.simulator_to_policy(positions).astype(np.float32)

    def command_joints(self, positions: Sequence[float]) -> None:
        targets = np.asarray(positions, dtype=float)
        if targets.shape != (6,) or not np.all(np.isfinite(targets)):
            raise ValueError("Expected six finite SO-101 joint positions")
        lower, upper = self.joint_limits()
        if np.any(targets < lower - 1e-6) or np.any(targets > upper + 1e-6):
            raise ValueError("SO-101 command violates joint limits")
        p.setJointMotorControlArray(
            self.robot_id, self.joint_indices, p.POSITION_CONTROL,
            targetPositions=targets.tolist(), forces=(18, 18, 18, 10, 8, 5),
            physicsClientId=self.client_id,
        )

    def camera_observations(self, width: int = 256, height: int = 256) -> dict[str, np.ndarray]:
        # The official mesh is expensive for PyBullet's CPU renderer. State is
        # provided separately, so hide the arm while capturing workspace views.
        visuals = p.getVisualShapeData(self.robot_id, physicsClientId=self.client_id)
        colors = {int(item[1]): item[7] for item in visuals}
        for link in colors:
            p.changeVisualShape(
                self.robot_id, link, rgbaColor=(0, 0, 0, 0),
                physicsClientId=self.client_id,
            )
        try:
            views = {
                "observation.images.camera1": ((0.0, 0.0, 0.72), (0.0, 0.0, 0.0), (0, 1, 0)),
                "observation.images.camera3": ((0.48, -0.48, 0.34), (0.0, 0.0, 0.07), (0, 0, 1)),
            }
            images = {
                key: self._render_camera(eye, target, up, width, height)
                for key, (eye, target, up) in views.items()
            }
            state = p.getLinkState(
                self.robot_id, self.tool_link_index, computeForwardKinematics=True,
                physicsClientId=self.client_id,
            )
            origin = np.asarray(state[4], dtype=float)
            rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
            forward = rotation @ np.asarray((1.0, 0.0, 0.0))
            eye = origin + 0.025 * forward
            target = origin + 0.25 * forward
            up = rotation @ np.asarray((0.0, 0.0, 1.0))
            images["observation.images.camera2"] = self._render_camera(
                eye, target, up, width, height
            )
            return images
        finally:
            for link, color in colors.items():
                p.changeVisualShape(
                    self.robot_id, link, rgbaColor=color,
                    physicsClientId=self.client_id,
                )

    def camera_observation(self, width: int = 320, height: int = 240) -> np.ndarray:
        return self.camera_observations(width, height)["observation.images.camera1"]

    def _render_camera(self, eye, target, up, width: int, height: int) -> np.ndarray:
        view = p.computeViewMatrix(eye, target, up)
        projection = p.computeProjectionMatrixFOV(60, width / height, 0.015, 1.5)
        rgba = p.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.client_id,
        )[2]
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


class SO101AStarController:
    """Low-height Cartesian A* converted to continuous SO-101 joint targets."""

    TRANSPORT_Z = 0.085
    APPROACH_Z = 0.15
    GRIPPER_OPEN = 1.2
    GRIPPER_CLOSED = 0.0
    PHYSICS_STEPS_PER_ACTION = 8

    def __init__(
        self, simulation: SO101Simulation, record_demonstration: bool = False
    ) -> None:
        self.simulation = simulation
        self.record_demonstration = record_demonstration
        self.demonstration: list[dict[str, object]] = []
        self.samples: list[tuple[float, float, float, float]] = []
        self.joint_samples: list[list[float]] = []
        self.path: list[tuple[float, float]] = []
        self.planning_time = 0.0

    def _planner(self, object_radius: float) -> CartesianAStarPlanner:
        return CartesianAStarPlanner(
            object_radius, self.simulation.obstacle_bounds,
            workspace=self.simulation.WORKSPACE,
            base_xy=self.simulation.ROBOT_BASE[:2],
            link_lengths=(0.20, 0.20), resolution=0.01,
        )

    def inverse_kinematics(self, xyz: Sequence[float], current: Optional[np.ndarray] = None) -> np.ndarray:
        if current is None:
            current = self.simulation.joint_states()[0]
        lower, upper = self.simulation.joint_limits()
        result = p.calculateInverseKinematics(
            self.simulation.robot_id, self.simulation.tool_link_index, xyz,
            lowerLimits=lower.tolist(), upperLimits=upper.tolist(),
            jointRanges=(upper - lower).tolist(), restPoses=current.tolist(),
            maxNumIterations=500, residualThreshold=1e-5,
            physicsClientId=self.simulation.client_id,
        )
        targets = np.asarray(result[:6], dtype=float)
        targets[4] = 0.0
        targets[5] = current[5]
        targets = np.clip(targets, lower, upper)
        achieved = self._tool_position_for(targets)
        if np.linalg.norm(achieved - np.asarray(xyz, dtype=float)) > 0.015:
            raise ValueError(f"SO-101 IK cannot reach {tuple(xyz)} within tolerance")
        return targets

    def move_object(self, object_name: str, destination: Sequence[float]) -> dict[str, float]:
        if object_name not in self.simulation.object_ids:
            raise ValueError(f"Unknown object {object_name!r}")
        body = self.simulation.object_ids[object_name]
        source = self.simulation.object_position(body)
        bounds = p.getAABB(body, -1, physicsClientId=self.simulation.client_id)
        half = (np.asarray(bounds[1]) - np.asarray(bounds[0])) / 2
        planner = self._planner(float(np.linalg.norm(half[:2])))
        started = time.perf_counter()
        self.path = planner.plan(source[:2], destination)
        self.planning_time = time.perf_counter() - started
        self.samples.clear()
        self.joint_samples.clear()
        self.demonstration.clear()

        current = self.simulation.joint_states()[0]
        current[5] = self.GRIPPER_OPEN
        approach = self.inverse_kinematics((*source[:2], self.APPROACH_Z), current)
        approach[5] = self.GRIPPER_OPEN
        self._move(approach, 20)
        grasp = self.inverse_kinematics((*source[:2], source[2] + half[2] + 0.035), approach)
        grasp[5] = self.GRIPPER_OPEN
        self._move(grasp, 12)
        grasp[5] = self.GRIPPER_CLOSED
        self._move(grasp, 6)
        self.simulation.attach_object(object_name)
        # Smoothly lift object into the closed gripper jaws during carry
        self.simulation._attachment_offset[2] = 0.020
        carry = self.inverse_kinematics((*source[:2], self.TRANSPORT_Z), grasp)
        carry[5] = self.GRIPPER_CLOSED
        self._move(carry, 12)
        self.simulation.lock_attached_height()
        self._record(body, carry)

        previous = carry
        for waypoint in self.path[1:]:
            target = self.inverse_kinematics((*waypoint, self.TRANSPORT_Z), previous)
            target[5] = self.GRIPPER_CLOSED
            self._validate_edge(previous, target, planner)
            self._move(target, 5, body)
            previous = target

        self.simulation.unlock_attached_height()
        lower = self.inverse_kinematics((*destination, source[2] + half[2] + 0.035), previous)
        lower[5] = self.GRIPPER_CLOSED
        self._move(lower, 10)
        lower[5] = self.GRIPPER_OPEN
        self._move(lower, 5)
        self.simulation.release_object()
        p.resetBasePositionAndOrientation(
            body, (*map(float, destination), float(half[2] + 0.002)),
            (0, 0, 0, 1), physicsClientId=self.simulation.client_id,
        )
        self.simulation.step(30)
        return self.metrics(destination, body)

    def _validate_edge(self, start: np.ndarray, end: np.ndarray, planner: CartesianAStarPlanner) -> None:
        for fraction in np.linspace(0, 1, 12):
            joints = start + fraction * (end - start)
            position = self._tool_position_for(joints)
            if not planner.valid_point(position[:2]):
                raise ValueError("SO-101 joint interpolation leaves the collision-free corridor")

    def _tool_position_for(self, joints: np.ndarray) -> np.ndarray:
        original = self.simulation.joint_states()[0]
        for index, value in zip(self.simulation.joint_indices, joints):
            p.resetJointState(
                self.simulation.robot_id, index, float(value), physicsClientId=self.simulation.client_id
            )
        position = self.simulation.tool_position()
        for index, value in zip(self.simulation.joint_indices, original):
            p.resetJointState(
                self.simulation.robot_id, index, float(value), physicsClientId=self.simulation.client_id
            )
        return position

    def _move(self, target: np.ndarray, action_steps: int, body: Optional[int] = None) -> None:
        initial = self.simulation.joint_states()[0]
        for fraction in np.linspace(0, 1, action_steps + 1)[1:]:
            command = initial + fraction * (target - initial)
            if self.record_demonstration:
                self.demonstration.append(
                    {
                        **self.simulation.camera_observations(),
                        "observation.state": self.simulation.policy_state(),
                        "action": self.simulation.simulator_to_policy(command).astype(
                            np.float32
                        ),
                    }
                )
            self.simulation.command_joints(command)
            self.simulation.step(self.PHYSICS_STEPS_PER_ACTION)
            if body is not None:
                self._record(body, command)

    def _record(self, body: int, joints: np.ndarray) -> None:
        position = self.simulation.object_position(body)
        self.samples.append((len(self.samples) / self.simulation.POLICY_HZ, *position))
        self.joint_samples.append(joints.tolist())

    def metrics(self, destination: Sequence[float], body: int) -> dict[str, float]:
        result = trajectory_metrics(self.samples, self.simulation.POLICY_HZ)
        final = self.simulation.object_position(body)
        result.update(
            planning_time_s=self.planning_time,
            endpoint_error_m=float(np.linalg.norm(final[:2] - np.asarray(destination))),
            collision_count=float(self.simulation.collision_count),
        )
        return result

    def save_results(self, chart: Path, result: Optional[Path] = None) -> None:
        save_comparison_chart(
            chart, self.samples, self.joint_samples, self.simulation.obstacle_bounds,
            "SO-101 A*",
        )
        if result:
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps({"trajectory": self.samples}, indent=2) + "\n")


class SO101ActionPolicy(Protocol):
    def predict_action_chunk(
        self, images: Mapping[str, np.ndarray], state: np.ndarray, instruction: str
    ) -> np.ndarray: ...


class SO101LearnedExecutor:
    """Execute native SO-101 checkpoint actions without an algorithmic fallback."""

    def __init__(self, simulation: SO101Simulation, policy: SO101ActionPolicy, checkpoint: str) -> None:
        self.simulation = simulation
        self.policy = policy
        self.checkpoint = checkpoint
        self.samples: list[tuple[float, float, float, float]] = []
        self.joint_samples: list[list[float]] = []
        self.raw_actions: list[list[float]] = []
        self.latencies: list[float] = []

    def execute(self, instruction: str, object_name: str, destination: Sequence[float], max_chunks: int = 20) -> dict:
        failure: Optional[str] = None
        body = self.simulation.object_ids[object_name]
        started_at = time.perf_counter()
        try:
            for _ in range(max_chunks):
                started = time.perf_counter()
                chunk = self.policy.predict_action_chunk(
                    self.simulation.camera_observations(), self.simulation.policy_state(), instruction
                )
                self.latencies.append(time.perf_counter() - started)
                actions = np.asarray(chunk, dtype=float)
                self._validate_chunk(actions)
                for policy_action in actions[:10]:
                    self.raw_actions.append(policy_action.tolist())
                    command = self.simulation.policy_to_simulator(policy_action)
                    self._validate_command(command)
                    self.simulation.command_joints(command)
                    self._update_grasp(object_name, policy_action[5])
                    before = self.simulation.collision_count
                    self.simulation.step(8)
                    if self.simulation.collision_count > before:
                        raise ValueError("SmolVLA action caused an obstacle collision")
                    position = self.simulation.object_position(body)
                    self.samples.append((len(self.samples) / 30.0, *position))
                    self.joint_samples.append(command.tolist())
                if self._complete(body, destination):
                    break
            if not self._complete(body, destination):
                failure = "SmolVLA did not complete the task within the action limit"
        except (ValueError, RuntimeError) as error:
            failure = str(error)
        metrics = trajectory_metrics(self.samples, 30)
        final = self.simulation.object_position(body)
        metrics.update(
            endpoint_error_m=float(np.linalg.norm(final[:2] - np.asarray(destination))),
            inference_latency_s=float(sum(self.latencies)),
            mean_chunk_latency_s=float(np.mean(self.latencies)) if self.latencies else 0.0,
            elapsed_s=time.perf_counter() - started_at,
            collision_count=float(self.simulation.collision_count),
        )
        return {
            "controller": "smolvla",
            "checkpoint": self.checkpoint,
            "prompt": instruction,
            "object": object_name,
            "destination": list(map(float, destination)),
            "task_success": failure is None and self._complete(body, destination),
            "failure_reason": failure,
            "raw_policy_actions": self.raw_actions,
            "trajectory": self.samples,
            "joint_trajectory": self.joint_samples,
            **metrics,
        }

    def _validate_chunk(self, actions: np.ndarray) -> None:
        if actions.ndim != 2 or actions.shape[1] != 6 or not 1 <= len(actions) <= 50:
            raise ValueError("SmolVLA returned an invalid SO-101 action chunk shape")
        if not np.all(np.isfinite(actions)):
            raise ValueError("SmolVLA returned a non-finite action")

    def _validate_command(self, command: np.ndarray) -> None:
        lower, upper = self.simulation.joint_limits()
        if np.any(command < lower) or np.any(command > upper):
            raise ValueError("SmolVLA action violates SO-101 joint limits")
        current = self.simulation.joint_states()[0]
        if np.any(np.abs(command - current) > np.deg2rad(35)):
            raise ValueError("SmolVLA action exceeds the per-action joint displacement limit")

    def _update_grasp(self, object_name: str, policy_gripper: float) -> None:
        distance = np.linalg.norm(
            self.simulation.tool_position() - self.simulation.object_position(self.simulation.object_ids[object_name])
        )
        if policy_gripper <= 25 and self.simulation.attached_object is None and distance < 0.07:
            self.simulation.attach_object(object_name)
        elif policy_gripper >= 40 and self.simulation.attached_object is not None:
            self.simulation.release_object()

    def _complete(self, body: int, destination: Sequence[float]) -> bool:
        return self.simulation.attached_object is None and np.linalg.norm(
            self.simulation.object_position(body)[:2] - np.asarray(destination)
        ) < 0.03


def trajectory_metrics(samples: Sequence[Sequence[float]], hz: int) -> dict[str, float]:
    if len(samples) < 2:
        return {"path_length_m": 0.0, "max_speed_m_s": 0.0,
                "max_acceleration_m_s2": 0.0, "max_jerk_m_s3": 0.0}
    values = np.asarray(samples, dtype=float)
    xy = values[:, 1:3]
    velocity = np.gradient(xy, 1 / hz, axis=0)
    acceleration = np.gradient(velocity, 1 / hz, axis=0)
    jerk = np.gradient(acceleration, 1 / hz, axis=0)
    return {
        "path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        "max_speed_m_s": float(np.linalg.norm(velocity, axis=1).max()),
        "max_acceleration_m_s2": float(np.linalg.norm(acceleration, axis=1).max()),
        "max_jerk_m_s3": float(np.linalg.norm(jerk, axis=1).max()),
        "height_range_m": float(np.ptp(values[:, 3])),
    }


def resolve_so101_task(simulation: SO101Simulation, instruction: str):
    """Ground a prompt and scale relation offsets to the SO-101 workspace."""
    from .vla import PromptResolver

    positions = {
        name: simulation.object_position(body)[:2]
        for name, body in simulation.object_ids.items()
    }
    resolver = PromptResolver()
    task = resolver.resolve(instruction, positions)
    text = instruction.lower()
    if any(word in text for word in ("target", "destination", "goal")):
        destination = np.asarray(simulation.DEFAULT_DESTINATION, dtype=float)
        simulation.set_goal_marker(destination)
    else:
        names = resolver._mentioned_objects(text)
        anchor = np.asarray(positions[names[1]])
        destination = anchor + 0.4 * (np.asarray(task.destination) - anchor)
        bounds = p.getAABB(
            simulation.object_ids[task.object_name], -1,
            physicsClientId=simulation.client_id,
        )
        half = (np.asarray(bounds[1]) - np.asarray(bounds[0])) / 2
        clearance = max(float(np.linalg.norm(half[:2])), 0.025) + 0.01
        xmin, ymin, xmax, ymax = simulation.WORKSPACE
        destination = np.clip(
            destination,
            (xmin + clearance, ymin + clearance),
            (xmax - clearance, ymax - clearance),
        )
    return task.object_name, tuple(map(float, destination))


def save_comparison_chart(
    path: Path,
    samples: Sequence[Sequence[float]],
    joints: Sequence[Sequence[float]],
    obstacles: Sequence[Sequence[float]],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1100, 850), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 18), title, fill="black")
    if not samples:
        draw.text((40, 55), "No executed trajectory", fill=(180, 30, 30))
        canvas.save(path)
        return
    values = np.asarray(samples, dtype=float)
    times, xy = values[:, 0], values[:, 1:3]
    dt = 1 / 30
    speed = np.linalg.norm(np.gradient(xy, dt, axis=0), axis=1)
    acceleration = np.linalg.norm(np.gradient(np.gradient(xy, dt, axis=0), dt, axis=0), axis=1)
    jerk = np.linalg.norm(np.gradient(np.gradient(np.gradient(xy, dt, axis=0), dt, axis=0), dt, axis=0), axis=1)
    for panel, (series, label, color) in enumerate(((speed, "Speed", "blue"), (acceleration, "Acceleration", "green"), (jerk, "Jerk", "red"))):
        left, right = 70, 1050
        top, bottom = 55 + panel * 170, 185 + panel * 170
        draw.rectangle((left, top, right, bottom), outline="black")
        draw.text((left + 5, top + 4), label, fill=color)
        maximum = max(float(np.max(series)), 1e-9)
        duration = max(float(times[-1]), dt)
        points = [(left + (right-left)*float(t)/duration, bottom-(bottom-top)*float(v)/maximum) for t,v in zip(times,series)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    left, right, top, bottom = 70, 540, 590, 820
    draw.rectangle((left, top, right, bottom), outline="black")
    def project(point):
        return (left + (point[0]+0.24)*(right-left)/0.48, bottom-(point[1]+0.20)*(bottom-top)/0.40)
    for bounds in obstacles:
        draw.rectangle((*project((bounds[0], bounds[3])), *project((bounds[2], bounds[1]))), fill="gray")
    draw.line([project(point) for point in xy], fill="purple", width=4)
    if joints:
        joint_values = np.asarray(joints)
        joint_left, joint_right, joint_top, joint_bottom = 600, 1050, 590, 820
        draw.rectangle(
            (joint_left, joint_top, joint_right, joint_bottom), outline="black"
        )
        draw.text((joint_left + 5, joint_top + 4), "Joint trajectories (rad)", fill="black")
        palette = ("red", "blue", "green", "purple", "orange", "brown")
        minimum = float(joint_values.min())
        span = max(float(joint_values.max()) - minimum, 1e-9)
        for index in range(joint_values.shape[1]):
            points = [
                (
                    joint_left + (joint_right - joint_left) * sample /
                    max(len(joint_values) - 1, 1),
                    joint_bottom - (joint_bottom - joint_top) *
                    (float(value) - minimum) / span,
                )
                for sample, value in enumerate(joint_values[:, index])
            ]
            if len(points) > 1:
                draw.line(points, fill=palette[index], width=2)
            draw.text(
                (joint_left + 8 + index * 68, joint_bottom - 18),
                f"q{index + 1}", fill=palette[index],
            )
    canvas.save(path)
