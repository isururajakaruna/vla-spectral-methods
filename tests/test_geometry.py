import math
import unittest

import numpy as np

from smooth_operator.geometry import footprint_coverage


class CoverageTests(unittest.TestCase):
    def test_full_partial_outside_and_touching(self):
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        for bounds, expected in [((-2, -2, 2, 2), 100), ((0, -2, 2, 2), 50),
                                 ((0, 0, 2, 2), 25), ((2, 2, 3, 3), 0),
                                 ((1, -1, 2, 1), 0)]:
            with self.subTest(bounds=bounds):
                self.assertAlmostEqual(footprint_coverage(square, bounds), expected)

    def test_rotated_footprint_not_bounding_box(self):
        diamond = np.array([(0, math.sqrt(2)), (math.sqrt(2), 0),
                            (0, -math.sqrt(2)), (-math.sqrt(2), 0)])
        self.assertAlmostEqual(footprint_coverage(diamond, (0, -2, 2, 2)), 50)
        expected = 100 * (1 - (math.sqrt(2) - 1)**2)
        self.assertAlmostEqual(footprint_coverage(diamond, (-1, -1, 1, 1)), expected)
