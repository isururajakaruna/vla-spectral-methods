import unittest

import numpy as np

from smooth_operator.planning import JointPlanner
from smooth_operator.simulation import PlanarArm


class PlanningTests(unittest.TestCase):
    def test_detour_and_blocked_goal(self):
        planner = JointPlanner()
        start = np.array([-1.2, 1.4])
        goal = np.array([-0.8301298000665818, 2.5324551548699183])
        self.assertFalse(planner.edge_valid(start, goal))
        path = planner.plan(start, goal)
        self.assertGreater(len(path), 2)
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], goal)
        for a, b in zip(path, path[1:]):
            self.assertTrue(planner.edge_valid(a, b))
        with self.assertRaises(ValueError):
            planner.plan(start, np.array([0.0, 1.4]))

    def test_seed_reproducibility_and_variation(self):
        scenes = []
        for seed in (42, 42, 7):
            with PlanarArm(seed=seed) as sim:
                scenes.append((sim.pickup_cell, sim.goal_cell))
                self.assertNotEqual(sim.pickup_cell, sim.goal_cell)
        self.assertEqual(scenes[0], scenes[1])
        self.assertNotEqual(scenes[0], scenes[2])
