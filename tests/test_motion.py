import unittest

import numpy as np
import pybullet as p

from smooth_operator.motion import BasicPickPlaceController, DESTINATION_XY
from smooth_operator.simulation import ManipulatorSimulation


class MotionTests(unittest.TestCase):
    def test_pick_and_place_each_object(self):
        for object_name in ("apple", "coke_can", "pen"):
            with self.subTest(object_name=object_name):
                with ManipulatorSimulation() as simulation:
                    controller = BasicPickPlaceController(simulation)
                    controller.move_object(object_name)
                    trajectory = np.asarray(controller.trajectory)
                    self.assertGreater(len(trajectory), 100)
                    self.assertLess(np.ptp(trajectory[:, 3]), 1e-6)
                    self.assertEqual(controller.metrics()["collision_count"], 0)
                    self.assertLess(controller.metrics()["height_range_m"], 0.001)
                    self.assertGreater(controller.metrics()["max_jerk_m_s3"], 0)
                    frame = controller.demonstration[0]
                    self.assertEqual(frame["observation.images.workspace"].shape, (120, 160, 3))
                    self.assertEqual(frame["observation.state"].shape, (4,))
                    self.assertEqual(frame["action"].shape, (4,))
                    self.assertTrue(np.isfinite(frame["action"]).all())
                    position = p.getBasePositionAndOrientation(
                        simulation.object_ids[object_name],
                        physicsClientId=simulation.client_id,
                    )[0]
                    np.testing.assert_allclose(
                        position[:2], DESTINATION_XY, atol=0.015
                    )
                    self.assertIsNone(simulation.attached_object)

    def test_unknown_object_is_rejected(self):
        with ManipulatorSimulation() as simulation:
            with self.assertRaises(ValueError):
                BasicPickPlaceController(simulation).move_object("unknown")


if __name__ == "__main__":
    unittest.main()
