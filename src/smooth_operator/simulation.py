"""Two powered revolute joints, a fixed base, and a magnetic tool.

SI units throughout. XY is the motion plane; Z is vertical.
Attachment is an abstraction, not a physical grasp or a learned policy.
"""
import math
import time

import numpy as np
import pybullet as p


class SimulationClosed(Exception):
    """The desktop simulation window was closed."""


class PlanarArm:
    lengths = (0.40, 0.32)
    dt = 1 / 240
    start = (0.43, -0.25)
    goal = (0.43, 0.25)
    height = 0.12

    def __init__(self, gui=False):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.attachment = None
        self.steps = 0
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.client)
        self._scene()
        self._arm()
        self.command((0.0, 1.4))
        for j, value in enumerate((0.0, 1.4)):
            p.resetJointState(self.robot, j, value, physicsClientId=self.client)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(1.25, 0, -75, (0.25, 0, 0), physicsClientId=self.client)
            for label, xy in (("A: attach", self.start), ("B: release", self.goal)):
                p.addUserDebugText(label, (*xy, 0.18), textSize=1.4, physicsClientId=self.client)

    def _box(self, half, position, color, mass=0):
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client)
        return p.createMultiBody(mass, collision, visual, position, physicsClientId=self.client)

    def _scene(self):
        self._box((0.85, 0.65, 0.025), (0.15, 0, -0.025), (0.88, 0.90, 0.93, 1))
        # Markers have no collision geometry.
        for xy, color in ((self.start, (0.95, 0.63, 0.12, 1)), (self.goal, (0.15, 0.75, 0.42, 1))):
            shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.065, length=0.002,
                                        rgbaColor=color, physicsClientId=self.client)
            p.createMultiBody(baseVisualShapeIndex=shape, basePosition=(*xy, 0.002), physicsClientId=self.client)
        self.object = self._box((0.022, 0.022, 0.025), (*self.start, 0.025), (0.9, 0.25, 0.16, 1), 0.05)
        p.changeDynamics(self.object, -1, lateralFriction=0.35, physicsClientId=self.client)

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
        if self.gui:
            time.sleep(self.dt)

    def move_to(self, xy, seconds=2.0):
        """Scripted joint interpolation for setup validation, not a VLA baseline."""
        if seconds <= 0:
            raise ValueError("Motion duration must be positive")
        initial, _ = self.joint_state()
        target = self.inverse_kinematics(xy)
        n = max(1, round(seconds / self.dt))
        for i in range(1, n + 1):
            u = i / n
            blend = 10*u**3 - 15*u**4 + 6*u**5
            self.command(initial + blend*(target-initial))
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
