"""PyBullet lifecycle and joint control for the manipulator simulation."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pybullet as p


ASSET_DIR = Path(__file__).with_name("assets")
OBJECT_PLACEMENTS = {
    "apple": ("apple.urdf", (-0.45, -0.35, 0.037)),
    "coke_can": ("coke_can.urdf", (-0.25, -0.35, 0.060)),
    "pen": ("pen.urdf", (0.05, -0.35, 0.012)),
}


class ManipulatorSimulation:
    """Load and control the static environment and elevated SCARA arm."""

    TIME_STEP = 1.0 / 240.0
    JOINT_COUNT = 3
    ROBOT_BASE = (-0.68, 0.0, 0.002)

    def __init__(self, gui: bool = False, training_seed: Optional[int] = None) -> None:
        self.gui = gui
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Could not connect to PyBullet")

        try:
            self._configure_physics()
            self.environment_id, self.robot_id = self._load_assets()
            self.object_ids = self._load_objects()
            self.joint_indices = self._find_controlled_joints()
            if len(self.joint_indices) != self.JOINT_COUNT:
                raise RuntimeError(
                    f"Expected {self.JOINT_COUNT} controlled joints, "
                    f"found {len(self.joint_indices)}"
                )
            self.tool_link_index = p.getNumJoints(
                self.robot_id, physicsClientId=self.client_id
            ) - 1
            self.attached_object: Optional[str] = None
            self._attachment_offset = np.zeros(3)
            self._attachment_orientation = (0.0, 0.0, 0.0, 1.0)
            self._attachment_height: Optional[float] = None
            self.obstacle_link_indices = self._environment_links("obstacle_")
            self.obstacle_bounds = [
                (-0.34, 0.13, -0.22, 0.31),
                (0.23, -0.25, 0.37, -0.11),
            ]
            self.training_obstacle_ids: list[int] = []
            self.goal_marker_id: Optional[int] = None
            self.goal_position: Optional[tuple[float, float]] = None
            if training_seed is not None:
                self._randomize_training_scene(training_seed)
            self.collision_count = 0
            self.command_joints((0.0, 0.0, 0.0))
        except Exception:
            self.close()
            raise

    def _configure_physics(self) -> None:
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.TIME_STEP, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.TIME_STEP,
            numSolverIterations=100,
            physicsClientId=self.client_id,
        )
        if self.gui:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id
            )

    def _load_assets(self) -> tuple[int, int]:
        environment_id = p.loadURDF(
            str(ASSET_DIR / "environment.urdf"),
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        robot_id = p.loadURDF(
            str(ASSET_DIR / "robot.urdf"),
            basePosition=self.ROBOT_BASE,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8,
                cameraYaw=45.0,
                cameraPitch=-35.0,
                cameraTargetPosition=(0.0, 0.0, 0.25),
                physicsClientId=self.client_id,
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id
            )
        return environment_id, robot_id

    def _load_objects(self) -> dict[str, int]:
        object_ids = {}
        for name, (asset, position) in OBJECT_PLACEMENTS.items():
            object_ids[name] = p.loadURDF(
                str(ASSET_DIR / asset),
                basePosition=position,
                physicsClientId=self.client_id,
            )
        p.changeDynamics(
            object_ids["pen"],
            -1,
            lateralFriction=1.0,
            rollingFriction=0.2,
            spinningFriction=0.1,
            physicsClientId=self.client_id,
        )
        return object_ids

    def _find_controlled_joints(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(
                p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
            )
            if p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)[2]
            in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
        )

    def _environment_links(self, prefix: str) -> tuple[int, ...]:
        links = []
        for index in range(
            p.getNumJoints(self.environment_id, physicsClientId=self.client_id)
        ):
            name = p.getJointInfo(
                self.environment_id, index, physicsClientId=self.client_id
            )[12].decode("utf-8")
            if name.startswith(prefix):
                links.append(index)
        return tuple(links)

    def _randomize_training_scene(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for link in self._environment_links("cell_markers"):
            p.changeVisualShape(
                self.environment_id,
                link,
                rgbaColor=(0.0, 0.0, 0.0, 0.0),
                physicsClientId=self.client_id,
            )
        for link in self.obstacle_link_indices:
            p.setCollisionFilterGroupMask(
                self.environment_id, link, 0, 0, physicsClientId=self.client_id
            )
            p.changeVisualShape(
                self.environment_id,
                link,
                rgbaColor=(0.0, 0.0, 0.0, 0.0),
                physicsClientId=self.client_id,
            )
        self.obstacle_bounds = []
        obstacle_count = int(rng.integers(1, 5))
        attempts = 0
        while len(self.obstacle_bounds) < obstacle_count and attempts < 100:
            attempts += 1
            width, depth = rng.uniform(0.10, 0.19, size=2)
            x = float(rng.uniform(-0.38, 0.38))
            y = float(rng.uniform(-0.30, 0.30))
            bounds = (x - width / 2, y - depth / 2, x + width / 2, y + depth / 2)
            if any(
                bounds[0] - 0.04 < other[2]
                and bounds[2] + 0.04 > other[0]
                and bounds[1] - 0.04 < other[3]
                and bounds[3] + 0.04 > other[1]
                for other in self.obstacle_bounds
            ):
                continue
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=(width / 2, depth / 2, 0.08),
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=(width / 2, depth / 2, 0.08),
                rgbaColor=(0.45, 0.24, 0.10, 1.0),
                physicsClientId=self.client_id,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=(x, y, 0.082),
                physicsClientId=self.client_id,
            )
            self.training_obstacle_ids.append(body)
            self.obstacle_bounds.append(bounds)

        object_cells = [
            (x, y)
            for x in np.arange(-0.45, 0.46, 0.10)
            for y in np.arange(-0.35, 0.36, 0.10)
            if not any(
                x0 - 0.06 <= x <= x1 + 0.06 and y0 - 0.06 <= y <= y1 + 0.06
                for x0, y0, x1, y1 in self.obstacle_bounds
            )
        ]
        rng.shuffle(object_cells)
        for (name, body), xy in zip(self.object_ids.items(), object_cells):
            current = p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)
            p.resetBasePositionAndOrientation(
                body,
                (xy[0], xy[1], current[0][2]),
                current[1],
                physicsClientId=self.client_id,
            )

    def body_bounds(self, body_id: int) -> tuple[float, float, float, float]:
        lower, upper = p.getAABB(body_id, -1, physicsClientId=self.client_id)
        return float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])

    def set_goal_marker(self, xy: Sequence[float]) -> None:
        """Display a collision-free target cell for a randomized task."""
        if self.goal_marker_id is not None:
            p.removeBody(self.goal_marker_id, physicsClientId=self.client_id)
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(0.048, 0.048, 0.001),
            rgbaColor=(0.10, 0.75, 0.20, 1.0),
            physicsClientId=self.client_id,
        )
        self.goal_marker_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual,
            basePosition=(float(xy[0]), float(xy[1]), 0.004),
            physicsClientId=self.client_id,
        )
        self.goal_position = (float(xy[0]), float(xy[1]))

    def joint_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Return controlled joint positions and velocities."""
        states = p.getJointStates(
            self.robot_id, self.joint_indices, physicsClientId=self.client_id
        )
        return (
            np.asarray([state[0] for state in states], dtype=float),
            np.asarray([state[1] for state in states], dtype=float),
        )

    def policy_state(self) -> np.ndarray:
        positions, _ = self.joint_states()
        return np.r_[positions, float(self.attached_object is not None)]

    def tool_position(self) -> np.ndarray:
        """Return the tool position in world coordinates."""
        state = p.getLinkState(
            self.robot_id,
            self.tool_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return np.asarray(state[4], dtype=float)

    def object_position(self, object_id: int) -> np.ndarray:
        return np.asarray(
            p.getBasePositionAndOrientation(
                object_id, physicsClientId=self.client_id
            )[0],
            dtype=float,
        )

    def camera_observation(self, width: int = 320, height: int = 240) -> np.ndarray:
        """Capture an RGB observation of the complete workspace."""
        if width < 1 or height < 1:
            raise ValueError("Camera dimensions must be positive")
        view = p.computeViewMatrix(
            cameraEyePosition=(0.9, -1.0, 1.35),
            cameraTargetPosition=(0.0, 0.0, 0.15),
            cameraUpVector=(0.0, 0.0, 1.0),
        )
        projection = p.computeProjectionMatrixFOV(
            fov=55.0,
            aspect=width / height,
            nearVal=0.05,
            farVal=4.0,
        )
        image = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )[2]
        return np.asarray(image, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def attach_object(self, object_name: str) -> None:
        """Attach an object below the tool using a temporary fixed constraint."""
        if self.attached_object is not None:
            raise RuntimeError("An object is already attached")
        if object_name not in self.object_ids:
            raise ValueError(f"Unknown object {object_name!r}")

        object_id = self.object_ids[object_name]
        object_position = np.asarray(
            p.getBasePositionAndOrientation(
                object_id, physicsClientId=self.client_id
            )[0]
        )
        offset = self.tool_position() - object_position
        self._attachment_offset = offset
        self._attachment_orientation = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self.client_id
        )[1]
        self.set_object_robot_collision(object_id, enabled=False)
        self.attached_object = object_name

    def lock_attached_height(self) -> None:
        """Keep the attached object at its current world height."""
        if self.attached_object is None:
            raise RuntimeError("No object is attached")
        object_id = self.object_ids[self.attached_object]
        self._attachment_height = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self.client_id
        )[0][2]

    def unlock_attached_height(self) -> None:
        self._attachment_height = None

    def release_object(self) -> None:
        """Release the currently attached object, if present."""
        if self.attached_object is None:
            return
        object_id = self.object_ids[self.attached_object]
        self.set_object_robot_collision(object_id, enabled=True)
        self.attached_object = None
        self._attachment_height = None

    def set_object_robot_collision(self, object_id: int, enabled: bool) -> None:
        """Enable or disable collision between an object and every robot link."""
        for robot_link in range(-1, self.tool_link_index + 1):
            p.setCollisionFilterPair(
                self.robot_id,
                object_id,
                robot_link,
                -1,
                int(enabled),
                physicsClientId=self.client_id,
            )

    def command_joints(self, positions: Sequence[float]) -> None:
        """Set position targets for the three controllable joints."""
        targets = np.asarray(positions, dtype=float)
        if targets.shape != (self.JOINT_COUNT,):
            raise ValueError(f"Expected {self.JOINT_COUNT} joint positions")
        if not np.all(np.isfinite(targets)):
            raise ValueError("Joint positions must be finite")

        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=targets.tolist(),
            forces=(45.0, 35.0, 60.0),
            physicsClientId=self.client_id,
        )

    def step(self, count: int = 1) -> None:
        """Advance the simulation by a positive number of fixed steps."""
        if count < 1:
            raise ValueError("Step count must be positive")
        for _ in range(count):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.attached_object is not None:
                object_id = self.object_ids[self.attached_object]
                position = self.tool_position() - self._attachment_offset
                if self._attachment_height is not None:
                    position[2] = self._attachment_height
                p.resetBasePositionAndOrientation(
                    object_id,
                    position.tolist(),
                    self._attachment_orientation,
                    physicsClientId=self.client_id,
                )
                p.resetBaseVelocity(
                    object_id,
                    linearVelocity=(0.0, 0.0, 0.0),
                    angularVelocity=(0.0, 0.0, 0.0),
                    physicsClientId=self.client_id,
                )
            for obstacle_link in self.obstacle_link_indices:
                contacts = p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.environment_id,
                    linkIndexB=obstacle_link,
                    physicsClientId=self.client_id,
                )
                self.collision_count += int(bool(contacts))
                if self.attached_object is not None:
                    object_contacts = p.getContactPoints(
                        bodyA=self.object_ids[self.attached_object],
                        bodyB=self.environment_id,
                        linkIndexB=obstacle_link,
                        physicsClientId=self.client_id,
                    )
                    self.collision_count += int(bool(object_contacts))
            for obstacle_id in self.training_obstacle_ids:
                self.collision_count += int(
                    bool(
                        p.getContactPoints(
                            self.robot_id,
                            obstacle_id,
                            physicsClientId=self.client_id,
                        )
                    )
                )
                if self.attached_object is not None:
                    self.collision_count += int(
                        bool(
                            p.getContactPoints(
                                self.object_ids[self.attached_object],
                                obstacle_id,
                                physicsClientId=self.client_id,
                            )
                        )
                    )

    def close(self) -> None:
        """Disconnect this simulation from PyBullet."""
        if p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)

    def __enter__(self) -> "ManipulatorSimulation":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
