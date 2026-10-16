import unittest

import numpy as np

from smooth_operator.simulation import ManipulatorSimulation
from smooth_operator.vla import (
    BaselineVLAPolicy,
    PromptResolver,
    SafeVLAExecutor,
)


class VLAPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = BaselineVLAPolicy()
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
        self.joints = np.zeros(3)

    def test_supported_instructions(self):
        instructions = {
            "Move the apple to the green cell": "apple",
            "Pick up the Coke can and place it in the goal": "coke_can",
            "Put the pen in the destination": "pen",
        }
        for instruction, expected in instructions.items():
            with self.subTest(instruction=instruction):
                action = self.policy.predict(self.image, instruction, self.joints)
                self.assertEqual(action.object_name, expected)

    def test_rejects_incomplete_or_ambiguous_instructions(self):
        for instruction in (
            "Look at the apple",
            "Move an object to the goal",
            "Move the apple and pen to the goal",
        ):
            with self.subTest(instruction=instruction):
                with self.assertRaises(ValueError):
                    self.policy.predict(self.image, instruction, self.joints)

    def test_rejects_invalid_observations(self):
        with self.assertRaises(ValueError):
            self.policy.predict(np.zeros((10, 10)), "Move apple to goal", self.joints)
        with self.assertRaises(ValueError):
            self.policy.predict(self.image, "Move apple to goal", np.zeros(2))

    def test_target_and_relation_paraphrases(self):
        positions = {"apple": (-0.4, -0.3), "coke_can": (0.0, 0.0), "pen": (0.2, 0.1)}
        resolver = PromptResolver()
        self.assertEqual(
            resolver.resolve("Move the apple to the target", positions).destination,
            (0.45, 0.35),
        )
        self.assertEqual(
            resolver.resolve("Put the pen to the right of the Coke can", positions).destination,
            (0.2, 0.0),
        )

    def test_unsafe_learned_action_uses_astar_fallback(self):
        class UnsafePolicy:
            def predict_action_chunk(self, image, state, instruction):
                return np.asarray([[np.nan, 0.0, 0.0, 0.0]])

        with ManipulatorSimulation() as simulation:
            result = SafeVLAExecutor(
                simulation, UnsafePolicy(), "fake-checkpoint"
            ).execute("Move the apple to the target")
            self.assertEqual(result["controller"], "astar_fallback")
            self.assertIn("non-finite", result["fallback_reason"])
            self.assertTrue(result["task_success"])
            self.assertIn("inference_latency_s", result)

    def test_random_goal_marker_controls_fallback_destination(self):
        class UnsafePolicy:
            def predict_action_chunk(self, image, state, instruction):
                return np.asarray([[np.nan, 0.0, 0.0, 0.0]])

        with ManipulatorSimulation() as simulation:
            simulation.set_goal_marker((-0.35, 0.35))
            result = SafeVLAExecutor(simulation, UnsafePolicy(), "fake").execute(
                "Move the apple to the target"
            )
            self.assertEqual(result["destination"], [-0.35, 0.35])


if __name__ == "__main__":
    unittest.main()
