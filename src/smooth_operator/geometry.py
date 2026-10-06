"""Exact convex footprint overlap with an axis-aligned workspace cell."""
import numpy as np


def convex_hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def chain(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result
    return chain(points)[:-1] + chain(reversed(points))[:-1]


def area(polygon):
    if len(polygon) < 3:
        return 0.0
    a = np.asarray(polygon)
    return float(abs(np.dot(a[:, 0], np.roll(a[:, 1], 1)) -
                     np.dot(a[:, 1], np.roll(a[:, 0], 1))) / 2)


def footprint_coverage(points, bounds):
    """Percent of a projected object footprint inside (xmin, ymin, xmax, ymax).

    Uses the convex hull of projected box corners, including arbitrary rotation
    and tilt. This is geometric ground truth, not a camera-pixel estimate.
    """
    polygon = convex_hull(points)
    total = area(polygon)
    if total <= 1e-15:
        raise ValueError("Object footprint must have positive area")
    xmin, ymin, xmax, ymax = bounds
    for axis, edge, sign in ((0, xmin, 1), (0, xmax, -1), (1, ymin, 1), (1, ymax, -1)):
        clipped = []
        if not polygon:
            return 0.0
        previous = np.asarray(polygon[-1], dtype=float)
        for point in polygon:
            current = np.asarray(point, dtype=float)
            inside = sign * (current[axis] - edge) >= 0
            was_inside = sign * (previous[axis] - edge) >= 0
            if inside != was_inside:
                fraction = (edge - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(previous + fraction * (current - previous))
            if inside:
                clipped.append(current)
            previous = current
        polygon = clipped
    return float(np.clip(100 * area(polygon) / total, 0, 100))
