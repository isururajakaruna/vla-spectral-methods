import unittest

import numpy as np
import pybullet as p

from smooth_operator.simulation import PlanarArm


class SimulationTests(unittest.TestCase):
    def test_robot_and_pick_place(self):
        with PlanarArm() as sim:
            types = [p.getJointInfo(sim.robot, i, physicsClientId=sim.client)[2] for i in range(3)]
            self.assertEqual(types, [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_FIXED])
            self.assertEqual(p.getDynamicsInfo(sim.robot, -1, physicsClientId=sim.client)[0], 0)
            with self.assertRaises(ValueError):
                sim.attach()
            with self.assertRaises(ValueError):
                sim.inverse_kinematics((2, 0))
            sim.move_to(sim.start)
            self.assertLess(np.linalg.norm(sim.endpoint()[:2] - sim.start), 0.005)
            sim.attach()
            sim.move_to(sim.goal)
            self.assertLess(np.linalg.norm(sim.object_position()[:2] - sim.goal), 0.01)
            sim.release()
            sim.move_to((0.60, 0.0), seconds=1)
            for _ in range(240):
                sim.step()
            self.assertLess(np.linalg.norm(sim.object_position()[:2] - sim.goal), 0.03)
            self.assertIsNone(sim.attachment)
            image = sim.camera(64)
            self.assertEqual(image.shape, (64, 64, 3))
            self.assertGreater(image.std(), 10)


if __name__ == "__main__":
    unittest.main()
