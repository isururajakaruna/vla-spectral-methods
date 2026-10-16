import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from smooth_operator.simulation import ManipulatorSimulation


class SimulationTests(unittest.TestCase):
    def test_randomized_training_scene_and_goal_marker(self):
        with ManipulatorSimulation(training_seed=42) as simulation:
            self.assertGreaterEqual(len(simulation.obstacle_bounds), 1)
            self.assertLessEqual(len(simulation.obstacle_bounds), 4)
            for body in simulation.object_ids.values():
                x, y = simulation.object_position(body)[:2]
                self.assertFalse(
                    any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in simulation.obstacle_bounds)
                )
            simulation.set_goal_marker((0.35, 0.35))
            self.assertIsNotNone(simulation.goal_marker_id)
            self.assertEqual(
                p.getNumJoints(simulation.goal_marker_id, physicsClientId=simulation.client_id),
                0,
            )

    def test_static_assets_and_joint_layout(self):
        with patch.object(
            p, "createCollisionShape", side_effect=AssertionError("runtime geometry")
        ), patch.object(
            p, "createVisualShape", side_effect=AssertionError("runtime geometry")
        ):
            with ManipulatorSimulation() as simulation:
                self.assertEqual(len(simulation.joint_indices), 3)
                self.assertEqual(
                    set(simulation.object_ids), {"apple", "coke_can", "pen"}
                )
                axes = [
                    p.getJointInfo(
                        simulation.robot_id,
                        index,
                        physicsClientId=simulation.client_id,
                    )[13]
                    for index in simulation.joint_indices
                ]
                self.assertEqual(
                    axes,
                    [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
                )
                types = [
                    p.getJointInfo(
                        simulation.robot_id,
                        index,
                        physicsClientId=simulation.client_id,
                    )[2]
                    for index in simulation.joint_indices
                ]
                self.assertEqual(
                    types, [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
                )

                collision_links = {}
                for index in range(
                    p.getNumJoints(
                        simulation.environment_id,
                        physicsClientId=simulation.client_id,
                    )
                ):
                    info = p.getJointInfo(
                        simulation.environment_id,
                        index,
                        physicsClientId=simulation.client_id,
                    )
                    collision_links[info[12].decode("utf-8")] = index

                for name in ("grid", "obstacle_left", "obstacle_right"):
                    self.assertIn(name, collision_links)
                    bounds = p.getAABB(
                        simulation.environment_id,
                        collision_links[name],
                        physicsClientId=simulation.client_id,
                    )
                    self.assertTrue(
                        np.all(np.subtract(bounds[1], bounds[0]) > 0.0)
                    )

                for body_id in simulation.object_ids.values():
                    self.assertGreater(
                        p.getDynamicsInfo(
                            body_id, -1, physicsClientId=simulation.client_id
                        )[0],
                        0.0,
                    )
                    bounds = p.getAABB(
                        body_id, -1, physicsClientId=simulation.client_id
                    )
                    self.assertTrue(
                        np.all(np.subtract(bounds[1], bounds[0]) > 0.0)
                    )

    def test_objects_settle_on_grid(self):
        with ManipulatorSimulation() as simulation:
            simulation.step(480)
            positions = {
                name: p.getBasePositionAndOrientation(
                    body_id, physicsClientId=simulation.client_id
                )[0]
                for name, body_id in simulation.object_ids.items()
            }
            self.assertAlmostEqual(positions["apple"][2], 0.037, places=2)
            self.assertAlmostEqual(positions["coke_can"][2], 0.060, places=2)
            self.assertAlmostEqual(positions["pen"][2], 0.012, places=2)

    def test_pen_stopper_prevents_rolling(self):
        with ManipulatorSimulation() as simulation:
            initial = p.getBasePositionAndOrientation(
                simulation.object_ids["pen"], physicsClientId=simulation.client_id
            )[0]
            simulation.step(2400)
            final = p.getBasePositionAndOrientation(
                simulation.object_ids["pen"], physicsClientId=simulation.client_id
            )[0]
            self.assertLess(abs(final[0]), 0.60)
            self.assertLess(abs(final[1]), 0.50)

    def test_obstacles_are_fixed(self):
        with ManipulatorSimulation() as simulation:
            obstacle_links = {}
            for index in range(
                p.getNumJoints(
                    simulation.environment_id,
                    physicsClientId=simulation.client_id,
                )
            ):
                info = p.getJointInfo(
                    simulation.environment_id,
                    index,
                    physicsClientId=simulation.client_id,
                )
                name = info[12].decode("utf-8")
                if name.startswith("obstacle_"):
                    self.assertEqual(info[2], p.JOINT_FIXED)
                    obstacle_links[name] = index

            self.assertEqual(set(obstacle_links), {"obstacle_left", "obstacle_right"})
            initial_positions = {
                name: p.getLinkState(
                    simulation.environment_id,
                    index,
                    physicsClientId=simulation.client_id,
                )[0]
                for name, index in obstacle_links.items()
            }
            simulation.step(480)
            for name, index in obstacle_links.items():
                position = p.getLinkState(
                    simulation.environment_id,
                    index,
                    physicsClientId=simulation.client_id,
                )[0]
                np.testing.assert_allclose(position, initial_positions[name])

    def test_each_joint_changes_tool_pose(self):
        with ManipulatorSimulation() as simulation:
            tool_link = p.getNumJoints(
                simulation.robot_id, physicsClientId=simulation.client_id
            ) - 1
            simulation.step(480)
            baseline = np.asarray(
                p.getLinkState(
                    simulation.robot_id,
                    tool_link,
                    physicsClientId=simulation.client_id,
                )[0]
            )

            targets = (0.35, 0.35, -0.20)
            for joint, value in enumerate(targets):
                target = np.zeros(3)
                target[joint] = value
                simulation.command_joints(target)
                simulation.step(480)
                position = np.asarray(
                    p.getLinkState(
                        simulation.robot_id,
                        tool_link,
                        physicsClientId=simulation.client_id,
                    )[0]
                )
                self.assertGreater(np.linalg.norm(position - baseline), 0.02)
                simulation.command_joints((0.0, 0.0, 0.0))
                simulation.step(480)

            positions, velocities = simulation.joint_states()
            self.assertEqual(positions.shape, (3,))
            self.assertEqual(velocities.shape, (3,))

    def test_command_validation(self):
        with ManipulatorSimulation() as simulation:
            with self.assertRaises(ValueError):
                simulation.command_joints((0.0, 0.0))
            with self.assertRaises(ValueError):
                simulation.command_joints((0.0, np.nan, 0.0))
            with self.assertRaises(ValueError):
                simulation.step(0)


if __name__ == "__main__":
    unittest.main()
