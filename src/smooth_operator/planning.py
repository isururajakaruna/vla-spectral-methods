"""Collision-aware Cartesian planning for the SCARA workspace."""

from heapq import heappop, heappush
import math
from typing import Iterable, Optional, Sequence

import numpy as np


WORKSPACE = (-0.60, -0.50, 0.60, 0.50)
OBSTACLES = (
    (-0.34, 0.13, -0.22, 0.31),
    (0.23, -0.25, 0.37, -0.11),
)


class CartesianAStarPlanner:
    """Plan an eight-connected path around inflated rectangular obstacles."""

    RESOLUTION = 0.02
    SAFETY_MARGIN = 0.01
    WRIST_RADIUS = 0.025
    BASE_XY = np.asarray((-0.68, 0.0))
    LINK_LENGTHS = (0.75, 0.70)

    def __init__(
        self,
        object_radius: float,
        obstacles: Iterable[Sequence[float]] = OBSTACLES,
    ) -> None:
        self.clearance = max(object_radius, self.WRIST_RADIUS) + self.SAFETY_MARGIN
        self.obstacles = tuple(tuple(map(float, obstacle)) for obstacle in obstacles)

    def plan(
        self, start: Sequence[float], goal: Sequence[float]
    ) -> list[tuple[float, float]]:
        start_cell = self._cell(start)
        goal_cell = self._cell(goal)
        if not self.valid_point(start) or not self.valid_point(goal):
            raise ValueError("Start or goal is occupied, unreachable, or outside the grid")

        frontier = [(0.0, start_cell)]
        previous = {start_cell: None}
        cost = {start_cell: 0.0}
        while frontier:
            _, current = heappop(frontier)
            if current == goal_cell:
                break
            for neighbor, step_cost in self._neighbors(current):
                point = self._point(neighbor)
                if not self.valid_point(point):
                    continue
                new_cost = cost[current] + step_cost
                if new_cost >= cost.get(neighbor, math.inf):
                    continue
                cost[neighbor] = new_cost
                previous[neighbor] = current
                heuristic = math.dist(point, self._point(goal_cell))
                heappush(frontier, (new_cost + heuristic, neighbor))

        if goal_cell not in previous:
            raise ValueError("No collision-free path exists")
        cells = []
        current: Optional[tuple[int, int]] = goal_cell
        while current is not None:
            cells.append(current)
            current = previous[current]
        cells.reverse()
        path = [tuple(map(float, start))]
        path.extend(self._point(cell) for cell in cells[1:-1])
        path.append(tuple(map(float, goal)))
        return path

    def valid_point(self, point: Sequence[float]) -> bool:
        x, y = map(float, point)
        xmin, ymin, xmax, ymax = WORKSPACE
        c = self.clearance
        if not (xmin + c <= x <= xmax - c and ymin + c <= y <= ymax - c):
            return False
        radius = float(np.linalg.norm(np.asarray((x, y)) - self.BASE_XY))
        inner = abs(self.LINK_LENGTHS[0] - self.LINK_LENGTHS[1])
        outer = sum(self.LINK_LENGTHS)
        if not inner <= radius <= outer:
            return False
        return not any(
            oxmin - c <= x <= oxmax + c and oymin - c <= y <= oymax + c
            for oxmin, oymin, oxmax, oymax in self.obstacles
        )

    def segment_valid(self, start: Sequence[float], end: Sequence[float]) -> bool:
        distance = math.dist(start, end)
        steps = max(1, math.ceil(distance / (self.RESOLUTION / 2.0)))
        return all(
            self.valid_point(np.asarray(start) + fraction * (np.asarray(end) - start))
            for fraction in np.linspace(0.0, 1.0, steps + 1)
        )

    def _cell(self, point: Sequence[float]) -> tuple[int, int]:
        return (
            round((float(point[0]) - WORKSPACE[0]) / self.RESOLUTION),
            round((float(point[1]) - WORKSPACE[1]) / self.RESOLUTION),
        )

    def _point(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            WORKSPACE[0] + cell[0] * self.RESOLUTION,
            WORKSPACE[1] + cell[1] * self.RESOLUTION,
        )

    @staticmethod
    def _neighbors(cell: tuple[int, int]):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    yield (cell[0] + dx, cell[1] + dy), math.hypot(dx, dy) * 0.02
