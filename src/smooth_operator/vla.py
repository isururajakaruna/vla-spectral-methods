"""Prompt grounding, SmolVLA inference, and collision-safe execution."""

from dataclasses import dataclass
import math
import re
import time
from typing import Mapping, Optional, Protocol

import numpy as np
import pybullet as p

from .motion import BasicPickPlaceController, DESTINATION_XY
from .planning import CartesianAStarPlanner
from .simulation import ManipulatorSimulation


TARGETS: Mapping[str, tuple[float, float]] = {
    "green": DESTINATION_XY,
    "blue": (-0.45, 0.35),
    "yellow": (0.45, -0.35),
}


@dataclass(frozen=True)
class PickPlaceAction:
    object_name: str
    destination: tuple[float, float] = DESTINATION_XY


class PromptResolver:
    """Ground broad target and spatial-relation prompts in the known scene."""

    OBJECT_PHRASES = {
        "apple": ("apple",),
        "coke_can": ("coke can", "coke", "can", "soda can"),
        "pen": ("pen",),
    }
    ACTION_WORDS = ("move", "pick", "place", "put", "carry", "transfer", "set")
    RELATIONS = {
        "left": (-0.20, 0.0),
        "right": (0.20, 0.0),
        "front": (0.0, 0.20),
        "behind": (0.0, -0.20),
        "beside": (0.20, 0.0),
        "next to": (0.20, 0.0),
    }

    def resolve(
        self,
        instruction: str,
        object_positions: Mapping[str, tuple[float, float]],
    ) -> PickPlaceAction:
        text = " ".join(instruction.lower().strip().split())
        if not text or not any(word in text for word in self.ACTION_WORDS):
            raise ValueError("Instruction must request a manipulation action")
        mentioned = self._mentioned_objects(text)
        if not mentioned:
            raise ValueError("Instruction must identify a known object")
        moving = mentioned[0]

        for color, destination in TARGETS.items():
            if color in text:
                if len(mentioned) != 1:
                    raise ValueError("Target instructions must identify one object")
                return PickPlaceAction(moving, destination)
        if any(word in text for word in ("target", "destination", "goal")):
            if len(mentioned) != 1:
                raise ValueError("Target instructions must identify one object")
            return PickPlaceAction(moving, DESTINATION_XY)

        relation = next((name for name in self.RELATIONS if name in text), None)
        if relation and len(mentioned) == 2:
            anchor = np.asarray(object_positions[mentioned[1]], dtype=float)
            destination = anchor + np.asarray(self.RELATIONS[relation])
            return PickPlaceAction(moving, tuple(map(float, destination)))
        raise ValueError("Instruction must identify a target or spatial relation")

    def _mentioned_objects(self, text: str) -> list[str]:
        mentions = []
        for object_name, phrases in self.OBJECT_PHRASES.items():
            positions = [
                match.start()
                for phrase in phrases
                for match in re.finditer(rf"\b{re.escape(phrase)}\b", text)
            ]
            if positions:
                mentions.append((min(positions), object_name))
        return [name for _, name in sorted(mentions)]


class ActionChunkPolicy(Protocol):
    def predict_action_chunk(
        self, image: np.ndarray, state: np.ndarray, instruction: str
    ) -> np.ndarray: ...


class SmolVLAPolicy:
    """Lazy adapter around a locally cached LeRobot SmolVLA checkpoint."""

    def __init__(self, checkpoint: str, device: str = "auto") -> None:
        try:
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
        except ImportError as error:
            raise RuntimeError(
                "SmolVLA requires Python 3.12 and `pip install 'lerobot[smolvla]'`"
            ) from error
        self.torch = torch
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.model = Policy.from_pretrained(checkpoint).to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def predict_action_chunk(
        self, image: np.ndarray, state: np.ndarray, instruction: str
    ) -> np.ndarray:
        torch = self.torch
        frame = {
            "observation.images.workspace": torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device),
            "observation.state": torch.from_numpy(state)
            .float()
            .unsqueeze(0)
            .to(self.device),
            "task": [instruction],
        }
        with torch.inference_mode():
            action = self.model.predict_action_chunk(self.preprocess(frame))
            action = self.postprocess(action)
        return np.asarray(action.detach().cpu()).reshape(-1, 4)[:10]


class SafeVLAExecutor:
    """Execute safe learned chunks and use A* after any unsafe prediction."""

    def __init__(
        self,
        simulation: ManipulatorSimulation,
        policy: ActionChunkPolicy,
        checkpoint: str,
    ) -> None:
        self.simulation = simulation
        self.policy = policy
        self.checkpoint = checkpoint
        self.resolver = PromptResolver()
        self.trajectory: list[tuple[float, float]] = []
        self.inference_latencies: list[float] = []

    def execute(self, instruction: str, max_chunks: int = 20) -> dict:
        positions = {
            name: p.getBasePositionAndOrientation(
                body, physicsClientId=self.simulation.client_id
            )[0][:2]
            for name, body in self.simulation.object_ids.items()
        }
        task = self.resolver.resolve(instruction, positions)
        if self.simulation.goal_position is not None and any(
            word in instruction.lower() for word in ("target", "destination", "goal")
        ):
            task = PickPlaceAction(task.object_name, self.simulation.goal_position)
        fallback_reason: Optional[str] = None
        self.trajectory.clear()
        self.inference_latencies.clear()
        try:
            for _ in range(max_chunks):
                started = time.perf_counter()
                chunk = self.policy.predict_action_chunk(
                    self.simulation.camera_observation(),
                    self.simulation.policy_state(),
                    instruction,
                )
                self.inference_latencies.append(time.perf_counter() - started)
                self._validate_chunk(chunk, task.object_name)
                self._execute_chunk(chunk, task.object_name)
                if self._task_complete(task):
                    break
            if not self._task_complete(task):
                fallback_reason = "learned policy did not complete the task"
        except (ValueError, RuntimeError) as error:
            fallback_reason = str(error)

        metrics = self._learned_metrics()
        if fallback_reason:
            try:
                controller = BasicPickPlaceController(self.simulation)
                metrics.update(controller.move_object(task.object_name, task.destination))
            except ValueError as error:
                fallback_reason = f"{fallback_reason}; A* fallback failed: {error}"
        success = self._task_complete(task)
        return {
            "controller": "astar_fallback" if fallback_reason else "smolvla",
            "fallback_reason": fallback_reason,
            "checkpoint": self.checkpoint,
            "prompt": instruction,
            "seed": None,
            "object": task.object_name,
            "destination": list(task.destination),
            "task_success": success,
            **metrics,
        }

    def _validate_chunk(self, chunk: np.ndarray, object_name: str) -> None:
        actions = np.asarray(chunk, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 4 or len(actions) > 10:
            raise ValueError("SmolVLA returned an invalid action chunk shape")
        if not np.all(np.isfinite(actions)):
            raise ValueError("SmolVLA returned a non-finite action")
        bounds = p.getAABB(
            self.simulation.object_ids[object_name],
            -1,
            physicsClientId=self.simulation.client_id,
        )
        half = (np.asarray(bounds[1]) - np.asarray(bounds[0])) / 2.0
        planner = CartesianAStarPlanner(
            float(np.linalg.norm(half[:2])), self.simulation.obstacle_bounds
        )
        controller = BasicPickPlaceController(self.simulation)
        previous = self.simulation.joint_states()[0][:2]
        for action in actions:
            if not (-math.pi <= action[0] <= math.pi and -math.pi <= action[1] <= math.pi):
                raise ValueError("SmolVLA action violates revolute joint limits")
            if not -0.34 <= action[2] <= 0.0 or not 0.0 <= action[3] <= 1.0:
                raise ValueError("SmolVLA action violates wrist or gripper limits")
            xy = controller.forward_kinematics(action[:2])
            if action[2] < -0.05:
                if not planner.valid_point(xy):
                    raise ValueError("SmolVLA action enters an occupied region")
                controller._validate_joint_edge(previous, action[:2], planner)
            previous = action[:2]

    def _execute_chunk(self, chunk: np.ndarray, object_name: str) -> None:
        for action in np.asarray(chunk):
            self.simulation.command_joints(action[:3])
            if action[3] >= 0.5 and self.simulation.attached_object is None:
                if np.linalg.norm(
                    self.simulation.tool_position()[:2]
                    - np.asarray(
                        p.getBasePositionAndOrientation(
                            self.simulation.object_ids[object_name],
                            physicsClientId=self.simulation.client_id,
                        )[0][:2]
                    )
                ) < 0.06:
                    self.simulation.attach_object(object_name)
            elif action[3] < 0.5 and self.simulation.attached_object is not None:
                self.simulation.release_object()
            self.simulation.step(24)
            self.trajectory.append(tuple(self.simulation.object_position(object_name)[:2]))

    def _learned_metrics(self) -> dict[str, float]:
        points = np.asarray(self.trajectory, dtype=float)
        metrics = {
            "inference_latency_s": float(sum(self.inference_latencies)),
            "mean_chunk_latency_s": float(np.mean(self.inference_latencies))
            if self.inference_latencies
            else 0.0,
            "collision_count": float(self.simulation.collision_count),
        }
        if len(points) < 2:
            return metrics
        velocity = np.gradient(points, 0.1, axis=0)
        acceleration = np.gradient(velocity, 0.1, axis=0)
        jerk = np.gradient(acceleration, 0.1, axis=0)
        metrics.update(
            path_length_m=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            max_speed_m_s=float(np.linalg.norm(velocity, axis=1).max()),
            max_acceleration_m_s2=float(np.linalg.norm(acceleration, axis=1).max()),
            max_jerk_m_s3=float(np.linalg.norm(jerk, axis=1).max()),
        )
        return metrics

    def _task_complete(self, task: PickPlaceAction) -> bool:
        position = p.getBasePositionAndOrientation(
            self.simulation.object_ids[task.object_name],
            physicsClientId=self.simulation.client_id,
        )[0]
        return np.linalg.norm(np.asarray(position[:2]) - task.destination) < 0.05


class BaselineVLAPolicy:
    """Compatibility wrapper for deterministic prompt grounding."""

    def predict(
        self, image: np.ndarray, instruction: str, joint_positions: np.ndarray
    ) -> PickPlaceAction:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB observation")
        if np.asarray(joint_positions).shape not in ((3,), (4,)):
            raise ValueError("Expected a valid robot state")
        dummy_positions = {name: (0.0, 0.0) for name in PromptResolver.OBJECT_PHRASES}
        return PromptResolver().resolve(instruction, dummy_positions)
