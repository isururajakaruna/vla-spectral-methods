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
        workspace: Sequence[float] = WORKSPACE,
        base_xy: Sequence[float] = BASE_XY,
        link_lengths: Sequence[float] = LINK_LENGTHS,
        resolution: float = RESOLUTION,
    ) -> None:
        self.clearance = max(object_radius, self.WRIST_RADIUS) + self.SAFETY_MARGIN
        self.obstacles = tuple(tuple(map(float, obstacle)) for obstacle in obstacles)
        self.workspace = tuple(map(float, workspace))
        self.base_xy = np.asarray(base_xy, dtype=float)
        self.link_lengths = tuple(map(float, link_lengths))
        self.resolution = float(resolution)
        if len(self.workspace) != 4 or len(self.link_lengths) != 2:
            raise ValueError("Invalid planner geometry")
        if self.resolution <= 0.0:
            raise ValueError("Planner resolution must be positive")

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
        xmin, ymin, xmax, ymax = self.workspace
        c = self.clearance
        if not (xmin + c <= x <= xmax - c and ymin + c <= y <= ymax - c):
            return False
        radius = float(np.linalg.norm(np.asarray((x, y)) - self.base_xy))
        inner = abs(self.link_lengths[0] - self.link_lengths[1])
        outer = sum(self.link_lengths)
        if not inner <= radius <= outer:
            return False
        return not any(
            oxmin - c <= x <= oxmax + c and oymin - c <= y <= oymax + c
            for oxmin, oymin, oxmax, oymax in self.obstacles
        )

    def segment_valid(self, start: Sequence[float], end: Sequence[float]) -> bool:
        distance = math.dist(start, end)
        steps = max(1, math.ceil(distance / (self.resolution / 2.0)))
        return all(
            self.valid_point(np.asarray(start) + fraction * (np.asarray(end) - start))
            for fraction in np.linspace(0.0, 1.0, steps + 1)
        )

    def _cell(self, point: Sequence[float]) -> tuple[int, int]:
        return (
            round((float(point[0]) - self.workspace[0]) / self.resolution),
            round((float(point[1]) - self.workspace[1]) / self.resolution),
        )

    def _point(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.workspace[0] + cell[0] * self.resolution,
            self.workspace[1] + cell[1] * self.resolution,
        )

    def _neighbors(self, cell: tuple[int, int]):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    yield (
                        cell[0] + dx,
                        cell[1] + dy,
                    ), math.hypot(dx, dy) * self.resolution
