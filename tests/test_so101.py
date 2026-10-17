import unittest

import numpy as np
import pybullet as p

from smooth_operator.so101 import (
    SO101AStarController,
    SO101LearnedExecutor,
    SO101Simulation,
)


class SO101Tests(unittest.TestCase):
    def test_official_joint_layout_cameras_and_calibration(self):
        with SO101Simulation() as simulation:
            names = [
                p.getJointInfo(
                    simulation.robot_id, index, physicsClientId=simulation.client_id
                )[1].decode("utf-8")
                for index in simulation.joint_indices
            ]
            self.assertEqual(names, list(simulation.JOINT_NAMES))
            self.assertEqual(len(simulation.joint_indices), 6)
            cameras = simulation.camera_observations()
            self.assertEqual(
                set(cameras),
                {
                    "observation.images.camera1",
                    "observation.images.camera2",
                    "observation.images.camera3",
                },
            )
            for image in cameras.values():
                self.assertEqual(image.shape, (256, 256, 3))
                self.assertEqual(image.dtype, np.uint8)
                self.assertGreater(float(image.std()), 5.0)
            joints = np.asarray((0.2, -0.3, 0.4, -0.5, 0.6, 0.7))
            np.testing.assert_allclose(
                simulation.policy_to_simulator(simulation.simulator_to_policy(joints)),
                joints,
                atol=1e-8,
            )

    def test_astar_completes_low_height_transport(self):
        with SO101Simulation() as simulation:
            controller = SO101AStarController(simulation)
            metrics = controller.move_object("apple", simulation.DEFAULT_DESTINATION)
            self.assertEqual(metrics["collision_count"], 0)
            self.assertLess(metrics["endpoint_error_m"], 0.001)
            self.assertLess(metrics["height_range_m"], 0.001)
            self.assertGreater(len(controller.samples), 2)

    def test_unsafe_policy_stops_without_astar_fallback(self):
        class UnsafePolicy:
            def predict_action_chunk(self, images, state, instruction):
                return np.full((50, 6), np.nan)

        with SO101Simulation() as simulation:
            executor = SO101LearnedExecutor(simulation, UnsafePolicy(), "base")
            result = executor.execute(
                "Move the apple to the target",
                "apple",
                simulation.DEFAULT_DESTINATION,
                max_chunks=1,
            )
            self.assertFalse(result["task_success"])
            self.assertEqual(result["controller"], "smolvla")
            self.assertIn("non-finite", result["failure_reason"])
            self.assertNotIn("fallback", result["controller"])

    def test_seed_reproduces_scene_and_observations(self):
        with SO101Simulation(training_seed=42) as first:
            first_obstacles = first.obstacle_bounds
            first_positions = {
                name: first.object_position(body).copy()
                for name, body in first.object_ids.items()
            }
            first_top = first.camera_observations()["observation.images.camera1"]
        with SO101Simulation(training_seed=42) as second:
            self.assertEqual(first_obstacles, second.obstacle_bounds)
            for name, body in second.object_ids.items():
                np.testing.assert_allclose(first_positions[name], second.object_position(body))
            np.testing.assert_array_equal(
                first_top,
                second.camera_observations()["observation.images.camera1"],
            )


if __name__ == "__main__":
    unittest.main()
