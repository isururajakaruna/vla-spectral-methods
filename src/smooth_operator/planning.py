"""Conservative planar joint-space planning for the scripted demo."""
import heapq
import math

import numpy as np


class JointPlanner:
    lengths = (0.40, 0.32)
    obstacle_center = (0.35, 0.0)
    obstacle_half = (0.045, 0.065)
    resolution = 0.10
    lower = np.array([-math.pi, 0.25])
    upper = np.array([math.pi, 2.65])

    def valid(self, q):
        if np.any(q < self.lower) or np.any(q > self.upper):
            return False
        elbow = self.lengths[0] * np.array([math.cos(q[0]), math.sin(q[0])])
        tip = elbow + self.lengths[1] * np.array([math.cos(sum(q)), math.sin(sum(q))])
        # Inflate for link width, tool and the carried box's circumscribed radius,
        # plus tracking clearance. Applied to both links for conservatism.
        low = np.array(self.obstacle_center) - self.obstacle_half - 0.05
        high = np.array(self.obstacle_center) + self.obstacle_half + 0.05
        for a, b in ((np.zeros(2), elbow), (elbow, tip)):
            enter, leave = 0.0, 1.0
            for axis in range(2):
                delta = b[axis] - a[axis]
                if abs(delta) < 1e-12:
                    if a[axis] < low[axis] or a[axis] > high[axis]:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    t0, t1 = sorted(((low[axis]-a[axis])/delta, (high[axis]-a[axis])/delta))
                    enter, leave = max(enter, t0), min(leave, t1)
            if enter <= leave:
                return False
        return True

    def edge_valid(self, a, b):
        n = max(1, math.ceil(float(np.max(np.abs(b-a))) / 0.02))
        return all(self.valid(a + (b-a)*i/n) for i in range(n+1))

    def plan(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        if not self.valid(start) or not self.valid(goal):
            raise ValueError("Start or goal violates obstacle clearance or joint bounds")
        if self.edge_valid(start, goal):
            return [start, goal]
        def angles(node):
            return self.lower + self.resolution*np.array(node)
        def connector(q):
            nearest = np.rint((q-self.lower)/self.resolution).astype(int)
            candidates = [(int(nearest[0]+i), int(nearest[1]+j)) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            candidates.sort(key=lambda node: np.linalg.norm(angles(node)-q))
            return next((node for node in candidates if self.edge_valid(q, angles(node))), None)
        source, target = connector(start), connector(goal)
        if source is None or target is None:
            raise ValueError("No safe connection to joint-space grid")
        queue = [(0.0, source)]
        costs, parent = {source: 0.0}, {}
        closed = set()
        while queue:
            _, node = heapq.heappop(queue)
            if node in closed:
                continue
            if node == target:
                nodes = [node]
                while node in parent:
                    node = parent[node]
                    nodes.append(node)
                path = [start] + [angles(n) for n in reversed(nodes)] + [goal]
                # Shortcut only when the entire interpolated edge is checked.
                result, i = [path[0]], 0
                while i < len(path)-1:
                    j = len(path)-1
                    while not self.edge_valid(path[i], path[j]):
                        j -= 1
                    result.append(path[j])
                    i = j
                return result
            closed.add(node)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                neighbor = (node[0]+dx, node[1]+dy)
                cost = costs[node] + math.hypot(dx, dy)*self.resolution
                if cost < costs.get(neighbor, math.inf) and self.edge_valid(angles(node), angles(neighbor)):
                    costs[neighbor], parent[neighbor] = cost, node
                    estimate = float(np.linalg.norm(angles(neighbor)-angles(target)))
                    heapq.heappush(queue, (cost+estimate, neighbor))
        raise ValueError("No collision-free path on the joint-space grid")
