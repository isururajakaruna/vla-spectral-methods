"""Two powered revolute joints, a fixed base, and a magnetic tool.

SI units throughout. XY is the motion plane; Z is vertical.
Attachment is an abstraction, not a physical grasp or a learned policy.
"""
import math
import time
import secrets
from itertools import product

import numpy as np
import pybullet as p

from .geometry import footprint_coverage
from .planning import JointPlanner


class SimulationClosed(Exception):
    """The desktop simulation window was closed."""


class PlanarArm:
    lengths = (0.40, 0.32)
    dt = 1 / 240
    grid_origin = (-0.70, -0.65)
    grid_shape = (17, 13)  # columns, rows; indices increase with world X and Y
    cell_size = 0.10
    object_half_extents = (0.022, 0.022, 0.025)
    height = 0.12

    home_joints = np.array([-1.2, 1.4])

    def __init__(self, gui=False, seed=None):
        self.gui = gui
        self.seed = secrets.randbits(32) if seed is None else seed
        self.planner = JointPlanner()
        self._choose_cells()
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.attachment = None
        self.steps = 0
        self.obstacle_contact_steps = 0
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.client)
        self._scene()
        self._arm()
        self.command(self.home_joints)
        for j, value in enumerate(self.home_joints):
            p.resetJointState(self.robot, j, value, physicsClientId=self.client)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(1.25, 0, -75, (0.25, 0, 0), physicsClientId=self.client)
            for label, xy in (("A: attach", self.start), ("B: release", self.goal)):
                p.addUserDebugText(label, (*xy, 0.18), textSize=1.4, physicsClientId=self.client)

    def _choose_cells(self):
        rng = np.random.default_rng(self.seed)
        candidates = []
        for col in range(self.grid_shape[0]):
            for row in range(self.grid_shape[1]):
                cell = (col, row)
                xy = self.cell_center(cell)
                if not (0.20 <= np.linalg.norm(xy) <= 0.65):
                    continue
                q = self.inverse_kinematics(xy)
                if self.planner.valid(q):
                    candidates.append(cell)
        rng.shuffle(candidates)
        reachable = []
        for cell in candidates:
            try:
                self.planner.plan(self.home_joints, self.inverse_kinematics(self.cell_center(cell)))
            except ValueError:
                continue
            if not reachable or np.linalg.norm(np.array(self.cell_center(cell)) - self.cell_center(reachable[0])) >= 0.20:
                reachable.append(cell)
            if len(reachable) == 2:
                self.pickup_cell, self.goal_cell = reachable
                return
        raise RuntimeError("Could not sample two reachable cells with obstacle clearance")

    def _box(self, half, position, color, mass=0):
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client)
        return p.createMultiBody(mass, collision, visual, position, physicsClientId=self.client)

    def _scene(self):
        self._box((0.85, 0.65, 0.025), (0.15, 0, -0.025), (0.88, 0.90, 0.93, 1))
        # Visual-only tiles: no raised platforms or added collision surfaces.
        for col in range(self.grid_shape[0]):
            for row in range(self.grid_shape[1]):
                cell = (col, row)
                shade = 0.86 if (col + row) % 2 else 0.93
                color = (shade, shade, shade, 1)
                if cell == self.pickup_cell:
                    color = (0.95, 0.63, 0.12, 1)
                elif cell == self.goal_cell:
                    color = (0.15, 0.75, 0.42, 1)
                shape = p.createVisualShape(p.GEOM_BOX,
                    halfExtents=(self.cell_size/2 - 0.001, self.cell_size/2 - 0.001, 0.0001),
                    rgbaColor=color, physicsClientId=self.client)
                p.createMultiBody(baseVisualShapeIndex=shape,
                    basePosition=(*self.cell_center(cell), 0.0002), physicsClientId=self.client)
        self.object = self._box(self.object_half_extents, (*self.start, 0.025), (0.9, 0.25, 0.16, 1), 0.05)
        p.changeDynamics(self.object, -1, lateralFriction=0.35, physicsClientId=self.client)
        self.obstacle = self._box((*self.planner.obstacle_half, 0.10),
                                 (*self.planner.obstacle_center, 0.10), (0.48, 0.22, 0.65, 1))

    def cell_bounds(self, cell):
        col, row = cell
        if not (0 <= col < self.grid_shape[0] and 0 <= row < self.grid_shape[1]):
            raise ValueError("Cell is outside the workspace grid")
        x = self.grid_origin[0] + col*self.cell_size
        y = self.grid_origin[1] + row*self.cell_size
        return (x, y, x+self.cell_size, y+self.cell_size)

    def cell_center(self, cell):
        xmin, ymin, xmax, ymax = self.cell_bounds(cell)
        return ((xmin+xmax)/2, (ymin+ymax)/2)

    @property
    def start(self):
        return self.cell_center(self.pickup_cell)

    @property
    def goal(self):
        return self.cell_center(self.goal_cell)

    def placement_metrics(self):
        position, orientation = p.getBasePositionAndOrientation(self.object, physicsClientId=self.client)
        rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        corners = np.array(list(product(*[(-h, h) for h in self.object_half_extents])))
        world = corners @ rotation.T + position
        coverage = footprint_coverage(world[:, :2], self.cell_bounds(self.goal_cell))
        return {"goal_coverage_pct": coverage, "placement_error_pct": 100 - coverage,
                "success": self.attachment is None and coverage >= 99.0}

    def _arm(self):
        base_c = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.07, height=0.20, physicsClientId=self.client)
        base_v = p.createVisualShape(p.GEOM_CYLINDER, radius=0.07, length=0.20,
                                   rgbaColor=(0.18, 0.22, 0.30, 1), physicsClientId=self.client)
        collisions, visuals = [], []
        for length, color in zip(self.lengths, ((0.12, 0.48, 0.75, 1), (0.12, 0.72, 0.70, 1))):
            collisions.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=(length/2, 0.022, 0.022),
                               collisionFramePosition=(length/2, 0, 0), physicsClientId=self.client))
            visuals.append(p.createVisualShape(p.GEOM_BOX, halfExtents=(length/2, 0.022, 0.022),
                           visualFramePosition=(length/2, 0, 0), rgbaColor=color, physicsClientId=self.client))
        # Fixed tool extends down to the object; it adds no actuated degree of freedom.
        visuals.append(p.createVisualShape(p.GEOM_CYLINDER, radius=0.028, length=0.08,
                       visualFramePosition=(0, 0, -0.04), rgbaColor=(0.22, 0.25, 0.30, 1), physicsClientId=self.client))
        self.robot = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=base_c, baseVisualShapeIndex=base_v,
            basePosition=(0, 0, self.height),
            linkMasses=[0.6, 0.45, 0.05], linkCollisionShapeIndices=collisions + [-1],
            linkVisualShapeIndices=visuals,
            linkPositions=[(0, 0, 0), (self.lengths[0], 0, 0), (self.lengths[1], 0, 0)],
            linkOrientations=[(0, 0, 0, 1)] * 3,
            linkInertialFramePositions=[(self.lengths[0]/2, 0, 0), (self.lengths[1]/2, 0, 0), (0, 0, 0)],
            linkInertialFrameOrientations=[(0, 0, 0, 1)] * 3,
            linkParentIndices=[0, 1, 2], linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_FIXED],
            linkJointAxis=[(0, 0, 1)] * 3, physicsClientId=self.client)

    def joint_state(self):
        states = p.getJointStates(self.robot, [0, 1], physicsClientId=self.client)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def endpoint(self):
        return np.array(p.getLinkState(self.robot, 2, computeForwardKinematics=True, physicsClientId=self.client)[4])

    def object_position(self):
        return np.array(p.getBasePositionAndOrientation(self.object, physicsClientId=self.client)[0])

    def inverse_kinematics(self, xy):
        x, y = xy
        l1, l2 = self.lengths
        c = (x*x + y*y - l1*l1 - l2*l2) / (2*l1*l2)
        if not math.isfinite(c) or abs(c) > 1 + 1e-9:
            raise ValueError(f"Point {xy} is outside the reachable annulus")
        q2 = math.acos(max(-1, min(1, c)))
        return np.array([math.atan2(y, x) - math.atan2(l2*math.sin(q2), l1+l2*math.cos(q2)), q2])

    def command(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (2,) or not np.isfinite(joints).all():
            raise ValueError("Expected two finite joint angles")
        p.setJointMotorControlArray(self.robot, [0, 1], p.POSITION_CONTROL,
                                   targetPositions=joints, forces=[12, 12],
                                   positionGains=[0.15, 0.15], velocityGains=[1, 1], physicsClientId=self.client)

    def step(self):
        try:
            p.stepSimulation(physicsClientId=self.client)
        except p.error:
            if self.gui and not p.isConnected(self.client):
                raise SimulationClosed() from None
            raise
        self.steps += 1
        if (p.getContactPoints(self.robot, self.obstacle, physicsClientId=self.client) or
                p.getContactPoints(self.object, self.obstacle, physicsClientId=self.client)):
            self.obstacle_contact_steps += 1
        if self.gui:
            time.sleep(self.dt)

    def move_to(self, xy, seconds=2.0):
        """Plan around the obstacle, then execute motor-driven joint waypoints."""
        if seconds <= 0:
            raise ValueError("Motion duration must be positive")
        target = self.inverse_kinematics(xy)
        self.move_joints(target, seconds)

    def retreat(self):
        self.move_joints(self.home_joints, 2.0)

    def move_joints(self, target, seconds):
        initial, _ = self.joint_state()
        path = self.planner.plan(initial, target)
        for a, b in zip(path, path[1:]):
            duration = max(seconds/(len(path)-1), float(np.max(np.abs(b-a)))*2.0)
            n = max(1, round(duration / self.dt))
            for i in range(1, n + 1):
                u = i / n
                blend = 10*u**3 - 15*u**4 + 6*u**5
                self.command(a + blend*(b-a))
                self.step()
        for _ in range(120):
            self.step()

    def attach(self):
        if self.attachment is not None:
            return
        if np.linalg.norm(self.endpoint()[:2] - self.object_position()[:2]) > 0.035:
            raise ValueError("Tool must be within 3.5 cm of the object to attach")
        self.attachment = p.createConstraint(self.robot, 2, self.object, -1, p.JOINT_FIXED,
                          (0, 0, 0), (0, 0, -self.height + 0.026), (0, 0, 0), physicsClientId=self.client)
        p.changeConstraint(self.attachment, maxForce=30, physicsClientId=self.client)

    def release(self):
        if self.attachment is not None:
            p.removeConstraint(self.attachment, physicsClientId=self.client)
            self.attachment = None

    def camera(self, size=512):
        view = p.computeViewMatrix((0.20, 0, 1.65), (0.20, 0, 0), (0, 1, 0))
        projection = p.computeProjectionMatrixFOV(55, 1, 0.05, 3)
        image = p.getCameraImage(size, size, view, projection, renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)
        return np.asarray(image[2], dtype=np.uint8).reshape(size, size, 4)[:, :, :3]

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
