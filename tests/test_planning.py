import unittest

import numpy as np

from smooth_operator.motion import BasicPickPlaceController
from smooth_operator.planning import CartesianAStarPlanner
from smooth_operator.simulation import ManipulatorSimulation


class PlanningTests(unittest.TestCase):
    def test_astar_detours_around_inflated_obstacles(self):
        planner = CartesianAStarPlanner(0.04)
        path = planner.plan((-0.45, -0.35), (0.45, 0.35))
        self.assertGreater(len(path), 2)
        self.assertEqual(path[0], (-0.45, -0.35))
        self.assertEqual(path[-1], (0.45, 0.35))
        self.assertTrue(all(planner.valid_point(point) for point in path))
        self.assertTrue(
            all(planner.segment_valid(a, b) for a, b in zip(path, path[1:]))
        )

    def test_blocked_and_outside_goals_fail(self):
        planner = CartesianAStarPlanner(0.04)
        for goal in ((0.30, -0.18), (0.80, 0.0)):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                planner.plan((-0.45, -0.35), goal)

    def test_all_grid_centers_have_scara_ik(self):
        with ManipulatorSimulation() as simulation:
            controller = BasicPickPlaceController(simulation)
            for x in np.arange(-0.55, 0.56, 0.10):
                for y in np.arange(-0.45, 0.46, 0.10):
                    joints = controller.inverse_kinematics((x, y))
                    np.testing.assert_allclose(
                        controller.forward_kinematics(joints), (x, y), atol=1e-8
                    )


if __name__ == "__main__":
    unittest.main()
