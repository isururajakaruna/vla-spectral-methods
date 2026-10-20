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
from .vla_models import LEGACY_OFFSET_DEGREES, LEGACY_SIGN


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
    # Legacy LeRobot `so100` degrees, the frame of the SO-100 community data most
    # VLAs were pretrained on: policy = sign * URDF degrees + offset. The URDF
    # matches the LeRobot >= 0.5 calibration; this is LeRobot's documented
    # backward-compatibility transform (huggingface.co/docs/lerobot/backwardcomp).
    POLICY_SIGN = np.asarray(LEGACY_SIGN)
    POLICY_OFFSET_DEGREES = np.asarray(LEGACY_OFFSET_DEGREES)
    POLICY_HZ = 30
    # Centre of the pocket between the open jaws, in gripper_frame_link axes. It sits
    # ~1 cm clear of the fixed jaw; closing the moving jaw pushes the object onto it.
    GRASP_POINT_OFFSET = np.asarray((-0.04, 0.0, 0.0))
    JAW_CONTACT_TOLERANCE = 0.004

    def __init__(self, gui: bool = False, training_seed: Optional[int] = None) -> None:
        self.training_seed = training_seed
        super().__init__(gui=gui, training_seed=training_seed)
        self.tool_link_index = self._link_index("gripper_frame_link")
        self.fixed_jaw_link_index = self._link_index("gripper_link")
        self.moving_jaw_link_index = self._link_index("moving_jaw_so101_v1_link")
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
        for name in ("apple", "coke_can"):
            p.changeDynamics(
                result[name], -1, lateralFriction=1.0, rollingFriction=0.01,
                spinningFriction=0.01, physicsClientId=self.client_id,
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
        return self.POLICY_SIGN * np.rad2deg(values) + self.POLICY_OFFSET_DEGREES

    def policy_to_simulator(self, values: Sequence[float]) -> np.ndarray:
        policy = np.asarray(values, dtype=float)
        if policy.shape != (6,) or not np.all(np.isfinite(policy)):
            raise ValueError("Expected six finite SO-101 policy actions")
        return np.deg2rad(self.POLICY_SIGN * (policy - self.POLICY_OFFSET_DEGREES))

    def policy_state(self) -> np.ndarray:
        positions, _ = self.joint_states()
        return self.simulator_to_policy(positions).astype(np.float32)

    def grasp_point(self) -> np.ndarray:
        state = p.getLinkState(
            self.robot_id, self.tool_link_index, computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        return np.asarray(state[4], dtype=float) + rotation @ self.GRASP_POINT_OFFSET

    def object_between_jaws(self, object_id: int) -> bool:
        """True when both the fixed and the moving jaw touch the object."""
        for link in (self.fixed_jaw_link_index, self.moving_jaw_link_index):
            points = p.getClosestPoints(
                self.robot_id, object_id, self.JAW_CONTACT_TOLERANCE,
                linkIndexA=link, physicsClientId=self.client_id,
            )
            if not points:
                return False
        return True

    def robot_clear_of_obstacles(self, margin: float = 0.001) -> bool:
        """True when no robot link is within `margin` of an obstacle at the current joint state."""
        p.performCollisionDetection(physicsClientId=self.client_id)
        # Training scenes disable the fixed obstacles with a collision filter, which
        # getClosestPoints ignores, so only query them when they are live.
        fixed_obstacles = self.obstacle_link_indices if self.training_seed is None else ()
        for link in fixed_obstacles:
            if p.getClosestPoints(
                self.robot_id, self.environment_id, margin, linkIndexB=link,
                physicsClientId=self.client_id,
            ):
                return False
        for obstacle in self.training_obstacle_ids:
            if p.getClosestPoints(self.robot_id, obstacle, margin, physicsClientId=self.client_id):
                return False
        return True

    _grasp_in_tool: Optional[tuple[Sequence[float], Sequence[float]]] = None
    _grasp_rest_z = 0.0
    _grip_limit: Optional[float] = None

    def attach_object(self, object_name: str) -> None:
        """Rigidly hold the object in the gripper frame at its current relative pose."""
        super().attach_object(object_name)
        object_position, object_orientation = p.getBasePositionAndOrientation(
            self.object_ids[object_name], physicsClientId=self.client_id
        )
        tool_position, tool_orientation = self._tool_pose()
        inverse = p.invertTransform(tool_position, tool_orientation)
        self._grasp_in_tool = p.multiplyTransforms(
            *inverse, object_position, object_orientation
        )
        self._grasp_rest_z = float(object_position[2])
        self._grip_limit = float(self.joint_states()[0][5])

    def release_object(self) -> None:
        super().release_object()
        self._grasp_in_tool = None
        self._grip_limit = None

    def _tool_pose(self) -> tuple[Sequence[float], Sequence[float]]:
        state = p.getLinkState(
            self.robot_id, self.tool_link_index, computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return state[4], state[5]

    def _attached_object_pose(self) -> tuple[np.ndarray, Sequence[float]]:
        if self._grasp_in_tool is None:
            return super()._attached_object_pose()
        position, orientation = p.multiplyTransforms(*self._tool_pose(), *self._grasp_in_tool)
        position = np.asarray(position, dtype=float)
        # The table still supports a held object, so it cannot be pushed below its resting height.
        position[2] = max(position[2], self._grasp_rest_z)
        if self._attachment_height is not None:
            position[2] = self._attachment_height
        return position, orientation

    def command_joints(self, positions: Sequence[float]) -> None:
        targets = np.asarray(positions, dtype=float)
        if targets.shape != (6,) or not np.all(np.isfinite(targets)):
            raise ValueError("Expected six finite SO-101 joint positions")
        lower, upper = self.joint_limits()
        if np.any(targets < lower - 1e-6) or np.any(targets > upper + 1e-6):
            raise ValueError("SO-101 command violates joint limits")
        if self._grip_limit is not None:
            # A held object stops the moving jaw where it made contact.
            targets[5] = max(targets[5], self._grip_limit)
        p.setJointMotorControlArray(
            self.robot_id, self.joint_indices, p.POSITION_CONTROL,
            targetPositions=targets.tolist(), forces=(18, 18, 18, 10, 8, 5),
            physicsClientId=self.client_id,
        )

    def camera_observations(self, width: int = 256, height: int = 256) -> dict[str, np.ndarray]:
        views = {
            "observation.images.camera1": ((0.0, 0.0, 0.62), (0.0, 0.0, 0.0), (0, 1, 0)),
            "observation.images.camera3": ((0.36, -0.38, 0.28), (-0.02, 0.0, 0.05), (0, 0, 1)),
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
        # In SO-101 URDF, +Z of gripper_frame_link points along the gripper jaws toward the object
        grasp_axis = rotation[:, 2]
        wrist_up = -rotation[:, 0]
        eye = origin - 0.04 * grasp_axis - 0.04 * rotation[:, 0]
        target = origin + 0.14 * grasp_axis
        images["observation.images.camera2"] = self._render_camera(
            eye.tolist(), target.tolist(), wrist_up.tolist(), width, height
        )
        return images

    def camera_observation(self, width: int = 320, height: int = 240) -> np.ndarray:
        return self.camera_observations(width, height)["observation.images.camera1"]

    def _render_camera(self, eye, target, up, width: int, height: int) -> np.ndarray:
        view = p.computeViewMatrix(eye, target, up)
        projection = p.computeProjectionMatrixFOV(60, width / height, 0.015, 1.5)
        rgba = p.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=projection,
            renderer=self.camera_renderer, physicsClientId=self.client_id,
        )[2]
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def _matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    """Rotation matrix to a PyBullet (x, y, z, w) quaternion."""
    m = np.asarray(matrix, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        quaternion = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        quaternion = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        quaternion = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        quaternion = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    return [float(value) for value in quaternion]


class SO101AStarController:
    """Low-height Cartesian A* converted to continuous SO-101 joint targets."""

    # Heights of the jaw grasp point (the held object's centre), not of gripper_frame_link.
    TRANSPORT_Z = 0.065
    APPROACH_Z = 0.09
    APPROACH_CLEARANCE = 0.04
    RELEASE_DROPS = (0.003, 0.013, 0.023, 0.033)
    # None means any orientation, the last resort for points close to the base.
    GRASP_TILTS_DEG = (0, 10, -10, 20, -20, 30, -30, 40, -40, 50, -50, 60, -60, None)
    SAFE_TRAVEL_Z = 0.15
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
        self._held_offset: Optional[np.ndarray] = None

    def _planner(self, object_radius: float) -> CartesianAStarPlanner:
        return CartesianAStarPlanner(
            object_radius, self.simulation.obstacle_bounds,
            workspace=self.simulation.WORKSPACE,
            base_xy=self.simulation.ROBOT_BASE[:2],
            # Reach annulus 0.14-0.40 m: closer than 0.14 m the jaws cannot reach table height.
            link_lengths=(0.27, 0.13), resolution=0.01,
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

    def grasp_inverse_kinematics(
        self,
        xyz: Sequence[float],
        current: Optional[np.ndarray] = None,
        avoid_obstacles: bool = False,
    ) -> np.ndarray:
        """Joint targets that put the jaw grasp point at `xyz` with the jaws pointing down.

        The 5-DoF arm cannot point straight down everywhere, so the approach is
        tilted in the arm plane by the smallest angle that is still reachable
        (and, with `avoid_obstacles`, keeps every link clear of obstacles).
        """
        if current is None:
            current = self.simulation.joint_states()[0]
        goal = np.asarray(xyz, dtype=float)
        for tilt in self.GRASP_TILTS_DEG:
            joints = self._solve_grasp_pose(goal, tilt, current)
            if joints is None:
                continue
            if avoid_obstacles and not self._pose_for(joints, self.simulation.robot_clear_of_obstacles):
                continue
            return joints
        raise ValueError(f"SO-101 IK cannot place the grasp point at {tuple(xyz)}")

    def _grasp_frame(self, xy: np.ndarray, tilt_deg: float) -> np.ndarray:
        radial = np.asarray((xy[0] - self.simulation.ROBOT_BASE[0], xy[1] - self.simulation.ROBOT_BASE[1], 0.0))
        radial /= max(np.linalg.norm(radial), 1e-9)
        tilt = np.deg2rad(tilt_deg)
        approach = np.sin(tilt) * radial - np.cos(tilt) * np.asarray((0.0, 0.0, 1.0))
        jaw = -np.cos(tilt) * radial - np.sin(tilt) * np.asarray((0.0, 0.0, 1.0))
        return np.column_stack((jaw, np.cross(approach, jaw), approach))

    def _solve_grasp_pose(
        self, goal: np.ndarray, tilt_deg: Optional[float], current: np.ndarray
    ) -> Optional[np.ndarray]:
        lower, upper = self.simulation.joint_limits()
        original = self.simulation.joint_states()[0]
        joints = np.asarray(current, dtype=float)
        if tilt_deg is None:
            frame, orientation, oriented_passes, free_passes = None, None, 0, 10
            frame_now = self._pose_for(joints, self._tool_rotation)
            tool_goal = goal - frame_now @ self.simulation.GRASP_POINT_OFFSET
        else:
            frame = self._grasp_frame(goal, tilt_deg)
            orientation, oriented_passes, free_passes = _matrix_to_quaternion(frame), 4, 4
            tool_goal = goal - frame @ self.simulation.GRASP_POINT_OFFSET
        try:
            # Orientation-constrained passes first, then position-only passes seeded from
            # that pose so the grasp point converges while the approach stays close.
            for iteration in range(oriented_passes + free_passes):
                self._set_joint_state(joints)
                kwargs = {"targetOrientation": orientation} if iteration < oriented_passes else {}
                result = p.calculateInverseKinematics(
                    self.simulation.robot_id, self.simulation.tool_link_index, tool_goal.tolist(),
                    lowerLimits=lower.tolist(), upperLimits=upper.tolist(),
                    jointRanges=(upper - lower).tolist(), restPoses=joints.tolist(),
                    maxNumIterations=800, residualThreshold=1e-6,
                    physicsClientId=self.simulation.client_id, **kwargs,
                )
                joints = np.clip(np.asarray(result[:6], dtype=float), lower, upper)
                joints[5] = current[5]
                self._set_joint_state(joints)
                error = goal - self.simulation.grasp_point()
                tool_goal = tool_goal + error
            approach = self._tool_rotation()[:, 2]
        finally:
            self._set_joint_state(original)
        if np.linalg.norm(error) > 0.005:
            return None
        if frame is not None:
            misalignment = np.degrees(np.arccos(np.clip(approach @ frame[:, 2], -1.0, 1.0)))
            if misalignment > 8.0:
                return None
        return joints

    def _tool_rotation(self) -> np.ndarray:
        orientation = p.getLinkState(
            self.simulation.robot_id, self.simulation.tool_link_index,
            computeForwardKinematics=True, physicsClientId=self.simulation.client_id,
        )[5]
        return np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

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

        approach_z = max(self.APPROACH_Z, source[2] + half[2] + self.APPROACH_CLEARANCE)
        current = self.simulation.joint_states()[0]
        current[5] = self.GRIPPER_OPEN
        above = self.grasp_inverse_kinematics(
            (*source[:2], self.SAFE_TRAVEL_Z), current, avoid_obstacles=True
        )
        above[5] = self.GRIPPER_OPEN
        self._move(above, 14)
        approach = self.grasp_inverse_kinematics(
            (*source[:2], approach_z), above, avoid_obstacles=True
        )
        approach[5] = self.GRIPPER_OPEN
        self._move(approach, 8)
        grasp = self.grasp_inverse_kinematics(source, approach)
        grasp[5] = self.GRIPPER_OPEN
        self._move(grasp, 12)
        grasp[5] = self.GRIPPER_CLOSED
        self._move(grasp, 6)
        if not self.simulation.object_between_jaws(body):
            raise ValueError(f"SO-101 gripper closed without enclosing the {object_name}")
        self.simulation.attach_object(object_name)
        self._held_offset = self._tool_rotation().T @ (
            self.simulation.object_position(body) - self.simulation.grasp_point()
        )
        carry = self._held_object_inverse_kinematics((*source[:2], self.TRANSPORT_Z), grasp)
        carry[5] = self.GRIPPER_CLOSED
        self._move(carry, 12)
        self.simulation.lock_attached_height()
        self._record(body, carry)

        previous = carry
        for waypoint in self.path[1:]:
            target = self._held_object_inverse_kinematics((*waypoint, self.TRANSPORT_Z), previous)
            target[5] = self.GRIPPER_CLOSED
            self._validate_edge(previous, target, planner)
            self._move(target, 5, body)
            previous = target

        self.simulation.unlock_attached_height()
        lower = self._release_pose(destination, source[2], previous)
        lower[5] = self.GRIPPER_CLOSED
        self._move(lower, 10)
        self.simulation.release_object()
        self._held_offset = None
        lower[5] = self.GRIPPER_OPEN
        self._move(lower, 5)
        retreat = self.grasp_inverse_kinematics((*destination, approach_z), lower)
        retreat[5] = self.GRIPPER_OPEN
        self._move(retreat, 10)
        self.simulation.step(30)
        return self.metrics(destination, body)

    def _held_object_inverse_kinematics(
        self, xyz: Sequence[float], current: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Joint targets that put the held object's centre, not the grasp point, at `xyz`."""
        goal = np.asarray(xyz, dtype=float)
        grasp_goal = goal.copy()
        for _ in range(3):
            joints = self.grasp_inverse_kinematics(grasp_goal, current, avoid_obstacles=True)
            grasp_goal = grasp_goal + goal - self._pose_for(joints, self._held_point)
        return joints

    def _release_pose(
        self, destination: Sequence[float], rest_z: float, current: np.ndarray
    ) -> np.ndarray:
        """Lower the object onto the table, releasing a little higher if the arm would hit an obstacle."""
        for drop in self.RELEASE_DROPS:
            try:
                return self._held_object_inverse_kinematics((*destination, rest_z + drop), current)
            except ValueError:
                continue
        raise ValueError(f"SO-101 cannot lower the object onto {tuple(destination)}")

    def _held_point(self) -> np.ndarray:
        point = self.simulation.grasp_point()
        if self._held_offset is None:
            return point
        return point + self._tool_rotation() @ self._held_offset

    def _validate_edge(self, start: np.ndarray, end: np.ndarray, planner: CartesianAStarPlanner) -> None:
        for fraction in np.linspace(0, 1, 12):
            joints = start + fraction * (end - start)
            position = self._pose_for(joints, self._held_point)
            if not planner.valid_point(position[:2]):
                raise ValueError("SO-101 joint interpolation leaves the collision-free corridor")
            if not self._pose_for(joints, lambda: self.simulation.robot_clear_of_obstacles(0.0)):
                raise ValueError("SO-101 arm would hit an obstacle between waypoints")

    def _tool_position_for(self, joints: np.ndarray) -> np.ndarray:
        return self._pose_for(joints, self.simulation.tool_position)

    def _grasp_point_for(self, joints: np.ndarray) -> np.ndarray:
        return self._pose_for(joints, self.simulation.grasp_point)

    def _pose_for(self, joints: np.ndarray, measure) -> np.ndarray:
        original = self.simulation.joint_states()[0]
        self._set_joint_state(joints)
        position = measure()
        self._set_joint_state(original)
        return position

    def _set_joint_state(self, joints: Sequence[float]) -> None:
        for index, value in zip(self.simulation.joint_indices, joints):
            p.resetJointState(
                self.simulation.robot_id, index, float(value), physicsClientId=self.simulation.client_id
            )

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
        body = self.simulation.object_ids[object_name]
        if (
            policy_gripper <= 25
            and self.simulation.attached_object is None
            and self.simulation.object_between_jaws(body)
        ):
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
    bounds = p.getAABB(
        simulation.object_ids[task.object_name], -1,
        physicsClientId=simulation.client_id,
    )
    half = (np.asarray(bounds[1]) - np.asarray(bounds[0])) / 2
    radius = float(np.linalg.norm(half[:2]))
    if any(word in text for word in ("target", "destination", "goal")):
        destination = np.asarray(simulation.DEFAULT_DESTINATION, dtype=float)
        simulation.set_goal_marker(destination)
    else:
        names = resolver._mentioned_objects(text)
        anchor = np.asarray(positions[names[1]])
        destination = anchor + 0.4 * (np.asarray(task.destination) - anchor)
        clearance = max(radius, 0.025) + 0.01
        xmin, ymin, xmax, ymax = simulation.WORKSPACE
        destination = np.clip(
            destination,
            (xmin + clearance, ymin + clearance),
            (xmax - clearance, ymax - clearance),
        )

    # Ensure destination has a fully valid, IK-reachable, edge-validated A* path
    controller = SO101AStarController(simulation)
    planner = controller._planner(radius)
    source_xy = np.asarray(positions[task.object_name], dtype=float)

    def _validate_full_path(src_xy: np.ndarray, dst_xy: np.ndarray) -> bool:
        try:
            path = planner.plan(src_xy, dst_xy)
            prev_q = controller.grasp_inverse_kinematics((*src_xy, controller.TRANSPORT_Z))
            for wp in path[1:]:
                next_q = controller.grasp_inverse_kinematics((*wp, controller.TRANSPORT_Z), prev_q)
                controller._validate_edge(prev_q, next_q, planner)
                prev_q = next_q
            return True
        except ValueError:
            return False

    if not _validate_full_path(source_xy, destination):
        # If source_xy itself is on a tight grid boundary, check a tiny 1-2cm nudge on source first
        for sx_off in (0.0, 0.015, -0.015, 0.025, -0.025):
            for sy_off in (0.0, 0.015, -0.015, 0.025, -0.025):
                cand_src = source_xy + np.asarray([sx_off, sy_off], dtype=float)
                try:
                    controller.grasp_inverse_kinematics((*cand_src, controller.APPROACH_Z))
                except ValueError:
                    continue
                if _validate_full_path(cand_src, destination):
                    source_xy = cand_src
                    body_id = simulation.object_ids[task.object_name]
                    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=simulation.client_id)
                    p.resetBasePositionAndOrientation(
                        body_id, (float(source_xy[0]), float(source_xy[1]), float(pos[2])), orn,
                        physicsClientId=simulation.client_id,
                    )
                    break
            else:
                continue
            break

    if not _validate_full_path(source_xy, destination):
        best_pt = None
        best_dist = float("inf")
        for gx in np.linspace(-0.04, 0.16, 15):
            for gy in np.linspace(-0.14, 0.14, 15):
                cand = np.asarray([gx, gy], dtype=float)
                dist = float(np.linalg.norm(cand - destination))
                if dist >= best_dist or float(np.linalg.norm(cand - source_xy)) < 0.05:
                    continue
                if _validate_full_path(source_xy, cand):
                    best_pt = cand
                    best_dist = dist
        if best_pt is not None:
            destination = best_pt
            if any(word in text for word in ("target", "destination", "goal")):
                simulation.set_goal_marker(destination)
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
