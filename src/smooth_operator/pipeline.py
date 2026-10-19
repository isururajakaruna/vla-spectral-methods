"""Scope 1: Modular Execution Pipeline with SceneSpec, Model Adapters, and Inverse Spectrum Module.

Connects:
1. Reproducible PyBullet Scene Setup (`SceneSpec` / Golden Data metadata or seed)
2. Model / Policy Adapter (`smolvla`, `fast_vla`, or `astar_oracle` chunk generator)
3. `fast_tokenization` toggle + Pluggable `InverseSpectrumModule` (`direct`, `fast_idct`,
   `band_idct`, `ifft`, `idwt_haar`, `idwt_db4`, `custom_hybrid`)
4. Real-Time GUI visualization (3D trajectory trace + HUD) and Headless execution
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import pybullet as p

from .inference_protocol import SocketActionChunkPolicy
from .metrics import compute_comprehensive_metrics, save_spectral_diagnostic_chart
from .motion import BasicPickPlaceController, DESTINATION_XY
from .planning import WORKSPACE as SCARA_WORKSPACE
from .simulation import ManipulatorSimulation
from .so101 import (
    SO101AStarController,
    SO101Simulation,
    resolve_so101_task,
)
from .spectrum import (
    InverseSpectrumModule,
    create_spectrum_module,
    parse_freq_band,
)
from .vla import PromptResolver


@dataclass
class SceneSpec:
    """Complete reproducible PyBullet environment and task specification."""

    seed: Optional[int]
    robot: Literal["so101", "scara"] = "so101"
    prompt: str = "Move the apple to the target"
    task_type: str = "target"
    target_object: str = "apple"
    destination_xy: tuple[float, float] = (0.16, 0.12)
    initial_object_poses: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    obstacles: list[list[float]] = field(default_factory=list)
    oracle_path_length_m: Optional[float] = None
    oracle_trajectory: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_episode_metadata(cls, metadata: Mapping[str, Any]) -> "SceneSpec":
        robot = str(metadata.get("robot", "so101"))
        scene_block = metadata.get("scene_spec") or {}
        poses = scene_block.get("initial_object_poses") or {}
        raw_obstacles = scene_block.get("obstacles") or metadata.get("obstacles") or []
        obstacles = []
        for item in raw_obstacles:
            if isinstance(item, Mapping) and "bounds_xy" in item:
                obstacles.append([float(v) for v in item["bounds_xy"]])
            else:
                obstacles.append([float(v) for v in item])
        dest = (
            scene_block.get("destination_xy")
            or metadata.get("destination")
            or ((0.16, 0.12) if robot == "so101" else DESTINATION_XY)
        )
        oracle_metrics = metadata.get("oracle_metrics") or metadata.get("metrics") or {}
        return cls(
            seed=metadata.get("seed"),
            robot="so101" if robot == "so101" else "scara",
            prompt=str(metadata.get("prompt", "Move the apple to the target")),
            task_type=str(metadata.get("task_type", "target")),
            target_object=str(
                scene_block.get("target_object") or metadata.get("object") or "apple"
            ),
            destination_xy=(float(dest[0]), float(dest[1])),
            initial_object_poses=dict(poses),
            obstacles=obstacles,
            oracle_path_length_m=(
                float(oracle_metrics["path_length_m"])
                if oracle_metrics.get("path_length_m") is not None
                else None
            ),
            oracle_trajectory=[
                [float(v) for v in row]
                for row in (metadata.get("oracle_trajectory") or [])
            ],
        )


def capture_scene_spec(
    simulation: ManipulatorSimulation,
    prompt: str,
    target_object: str,
    destination_xy: Sequence[float],
    seed: Optional[int] = None,
    task_type: str = "target",
    oracle_path_length_m: Optional[float] = None,
    oracle_trajectory: Optional[Sequence[Sequence[float]]] = None,
) -> SceneSpec:
    """Snapshot the exact current PyBullet object poses, obstacles, and task into a SceneSpec."""
    robot: Literal["so101", "scara"] = (
        "so101" if isinstance(simulation, SO101Simulation) else "scara"
    )
    poses: dict[str, dict[str, list[float]]] = {}
    for name, body_id in simulation.object_ids.items():
        pos, orn = p.getBasePositionAndOrientation(
            body_id, physicsClientId=simulation.client_id
        )
        poses[name] = {
            "xyz": [float(v) for v in pos],
            "orn_xyzw": [float(v) for v in orn],
        }
    return SceneSpec(
        seed=seed,
        robot=robot,
        prompt=prompt,
        task_type=task_type,
        target_object=target_object,
        destination_xy=(float(destination_xy[0]), float(destination_xy[1])),
        initial_object_poses=poses,
        obstacles=[[float(v) for v in b] for b in simulation.obstacle_bounds],
        oracle_path_length_m=oracle_path_length_m,
        oracle_trajectory=[list(map(float, row)) for row in (oracle_trajectory or [])],
    )


def apply_scene_spec(simulation: ManipulatorSimulation, spec: SceneSpec) -> None:
    """Restore exact object poses, obstacles, and goal markers onto a simulation instance."""
    current_obstacles = [[float(v) for v in b] for b in simulation.obstacle_bounds]
    if spec.obstacles and not np.allclose(current_obstacles, spec.obstacles, atol=1e-6):
        for link in simulation._environment_links("cell_markers"):
            p.changeVisualShape(
                simulation.environment_id,
                link,
                rgbaColor=(0.0, 0.0, 0.0, 0.0),
                physicsClientId=simulation.client_id,
            )
        for link in simulation.obstacle_link_indices:
            p.setCollisionFilterGroupMask(
                simulation.environment_id,
                link,
                0,
                0,
                physicsClientId=simulation.client_id,
            )
            p.changeVisualShape(
                simulation.environment_id,
                link,
                rgbaColor=(0.0, 0.0, 0.0, 0.0),
                physicsClientId=simulation.client_id,
            )
        for obs_id in simulation.training_obstacle_ids:
            p.removeBody(obs_id, physicsClientId=simulation.client_id)
        simulation.training_obstacle_ids.clear()
        simulation.obstacle_bounds = []
        half_z = 0.045 if isinstance(simulation, SO101Simulation) else 0.08
        base_z = 0.046 if isinstance(simulation, SO101Simulation) else 0.082
        for bounds in spec.obstacles:
            x0, y0, x1, y1 = map(float, bounds)
            width, depth = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=(width / 2.0, depth / 2.0, half_z),
                physicsClientId=simulation.client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=(width / 2.0, depth / 2.0, half_z),
                rgbaColor=(0.45, 0.24, 0.10, 1.0),
                physicsClientId=simulation.client_id,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=(cx, cy, base_z),
                physicsClientId=simulation.client_id,
            )
            simulation.training_obstacle_ids.append(body)
            simulation.obstacle_bounds.append((x0, y0, x1, y1))

    for name, pose in spec.initial_object_poses.items():
        if name in simulation.object_ids:
            body_id = simulation.object_ids[name]
            xyz = pose["xyz"]
            orn = pose.get("orn_xyzw", [0.0, 0.0, 0.0, 1.0])
            p.resetBasePositionAndOrientation(
                body_id, xyz, orn, physicsClientId=simulation.client_id
            )
            p.resetBaseVelocity(
                body_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                physicsClientId=simulation.client_id,
            )

    if any(word in spec.prompt.lower() for word in ("target", "destination", "goal")):
        simulation.set_goal_marker(spec.destination_xy)
    simulation.collision_count = 0


class PrecomputedChunkPolicy:
    """Stream 10-step action chunks from an expert/oracle action sequence for controlled spectral ablation."""

    def __init__(self, full_actions: np.ndarray, chunk_size: int = 10) -> None:
        self.full_actions = np.asarray(full_actions, dtype=float)
        self.chunk_size = chunk_size
        self.cursor = 0

    def predict_action_chunk(
        self, image: Any, state: np.ndarray, instruction: str
    ) -> np.ndarray:
        if self.cursor >= len(self.full_actions):
            return np.repeat(self.full_actions[-1:, :], self.chunk_size, axis=0)
        chunk = self.full_actions[self.cursor : self.cursor + self.chunk_size]
        self.cursor += len(chunk)
        if len(chunk) < self.chunk_size:
            pad = np.repeat(chunk[-1:, :], self.chunk_size - len(chunk), axis=0)
            chunk = np.vstack([chunk, pad])
        return chunk


def generate_oracle_action_sequence(
    spec: SceneSpec,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]], float]:
    """Run an isolated headless oracle planner to obtain the raw unsmoothed policy action stream."""
    sim_cls = SO101Simulation if spec.robot == "so101" else ManipulatorSimulation
    with sim_cls(gui=False, training_seed=spec.seed) as sim:
        apply_scene_spec(sim, spec)
        if isinstance(sim, SO101Simulation):
            controller = SO101AStarController(sim, record_demonstration=True)
            metrics = controller.move_object(spec.target_object, spec.destination_xy)
            actions = np.stack([frame["action"] for frame in controller.demonstration]).astype(float)
            return actions, list(controller.samples), float(metrics.get("path_length_m", 0.0))
        else:
            controller = BasicPickPlaceController(sim)
            # Record every control step for SCARA chunk replay
            metrics = controller.move_object(spec.target_object, spec.destination_xy)
            actions = np.stack([frame["action"] for frame in controller.demonstration]).astype(float)
            return actions, list(controller.trajectory), float(metrics.get("path_length_m", 0.0))


class ModularExecutionPipeline:
    """Unified execution engine supporting Direct Time-Domain and FAST/InverseSpectrumModule control."""

    def __init__(
        self,
        simulation: ManipulatorSimulation,
        policy: Any,
        spectrum_module: InverseSpectrumModule,
        model_name: str = "smolvla",
        checkpoint: str = "lerobot/smolvla_base",
        chunk_size: int = 10,
        realtime_gui: bool = False,
    ) -> None:
        self.simulation = simulation
        self.policy = policy
        self.spectrum_module = spectrum_module
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.chunk_size = chunk_size
        self.realtime_gui = realtime_gui

        self.object_samples: list[tuple[float, float, float, float]] = []
        self.ee_samples: list[tuple[float, float, float, float]] = []
        self.joint_samples: list[list[float]] = []
        self.raw_policy_actions: list[list[float]] = []
        self.reconstructed_actions: list[list[float]] = []
        self.model_latencies: list[float] = []
        self.spectrum_decode_latencies: list[float] = []
        self.compression_ratios: list[float] = []
        self._debug_items: list[int] = []

    def _render_hud(self, prompt: str) -> None:
        if not self.simulation.gui:
            return
        for item in self._debug_items:
            p.removeUserDebugItem(item, physicsClientId=self.simulation.client_id)
        self._debug_items.clear()
        hud = (
            f"Model: {self.model_name} | FAST: {self.spectrum_module.fast_tokenization} | "
            f"Spectrum: {self.spectrum_module.schema_name} {self.spectrum_module.freq_band}\n"
            f"Prompt: {prompt}"
        )
        text_pos = (0.0, 0.0, 0.28) if isinstance(self.simulation, SO101Simulation) else (0.0, 0.0, 0.55)
        item_id = p.addUserDebugText(
            hud,
            textPosition=text_pos,
            textColorRGB=(0.1, 0.1, 0.1),
            textSize=1.2,
            physicsClientId=self.simulation.client_id,
        )
        self._debug_items.append(item_id)

    def execute_task(
        self,
        spec: SceneSpec,
        max_chunks: int = 25,
    ) -> dict[str, Any]:
        """Execute a prompt-driven manipulation episode and return comprehensive metrics."""
        self.object_samples.clear()
        self.ee_samples.clear()
        self.joint_samples.clear()
        self.raw_policy_actions.clear()
        self.reconstructed_actions.clear()
        self.model_latencies.clear()
        self.spectrum_decode_latencies.clear()
        self.compression_ratios.clear()

        is_so101 = isinstance(self.simulation, SO101Simulation)
        hz = 30.0 if is_so101 else 10.0
        steps_per_action = 8 if is_so101 else 24
        object_name = spec.target_object
        destination = spec.destination_xy
        body_id = self.simulation.object_ids[object_name]

        self._render_hud(spec.prompt)
        started_wall = time.perf_counter()
        failure_reason: Optional[str] = None
        prev_chunk_tail: Optional[np.ndarray] = None
        prev_ee_pos: Optional[np.ndarray] = None

        try:
            for _ in range(max_chunks):
                t_inf0 = time.perf_counter()
                if self.model_name == "astar_oracle" and isinstance(self.policy, PrecomputedChunkPolicy):
                    obs_images: Any = None
                else:
                    obs_images = (
                        self.simulation.camera_observations()
                        if is_so101
                        else self.simulation.camera_observation()
                    )
                raw_chunk = np.asarray(
                    self.policy.predict_action_chunk(
                        obs_images,
                        self.simulation.policy_state(),
                        spec.prompt,
                    ),
                    dtype=float,
                )
                self.model_latencies.append(time.perf_counter() - t_inf0)

                if raw_chunk.ndim != 2 or not np.all(np.isfinite(raw_chunk)):
                    raise ValueError("Model returned a non-finite or malformed action chunk")
                raw_chunk = raw_chunk[: self.chunk_size]

                # Pass through InverseSpectrumModule if FAST / spectral mode is enabled
                t_dec0 = time.perf_counter()
                if self.spectrum_module.fast_tokenization:
                    encoded_payload = self.spectrum_module.encode(raw_chunk)
                    self.compression_ratios.append(
                        float(encoded_payload.get("non_zero_ratio", 1.0))
                    )
                    exec_chunk = self.spectrum_module.decode(
                        encoded_payload, prev_chunk_tail=prev_chunk_tail
                    )
                else:
                    self.compression_ratios.append(1.0)
                    exec_chunk = self.spectrum_module.decode(raw_chunk)
                self.spectrum_decode_latencies.append(time.perf_counter() - t_dec0)

                for raw_act, exec_act in zip(raw_chunk, exec_chunk):
                    self.raw_policy_actions.append(raw_act.tolist())
                    self.reconstructed_actions.append(exec_act.tolist())
                    if is_so101:
                        self._step_so101(exec_act, object_name, steps_per_action)
                    else:
                        self._step_scara(exec_act, object_name, steps_per_action)

                    t_now = len(self.object_samples) / hz
                    obj_pos = self.simulation.object_position(body_id)
                    ee_pos = self.simulation.tool_position()
                    self.object_samples.append((t_now, *obj_pos))
                    self.ee_samples.append((t_now, *ee_pos))

                    if self.simulation.gui and prev_ee_pos is not None:
                        p.addUserDebugLine(
                            prev_ee_pos.tolist(),
                            ee_pos.tolist(),
                            lineColorRGB=(0.55, 0.15, 0.75),
                            lineWidth=2.5,
                            lifeTime=0.0,
                            physicsClientId=self.simulation.client_id,
                        )
                    prev_ee_pos = ee_pos.copy()
                    if self.realtime_gui and self.simulation.gui:
                        time.sleep(steps_per_action * self.simulation.TIME_STEP)

                prev_chunk_tail = exec_chunk[-2:].copy() if len(exec_chunk) >= 2 else exec_chunk.copy()
                if self._is_task_complete(body_id, destination, is_so101):
                    break

            if not self._is_task_complete(body_id, destination, is_so101):
                failure_reason = "Policy did not complete the task within the action limit"
        except (ValueError, RuntimeError) as exc:
            failure_reason = str(exc)

        # If using the deterministic oracle stream and object settled just after release,
        # let physics settle 20 steps so final position is stable
        if failure_reason is None or "within the action limit" in (failure_reason or ""):
            if self.simulation.attached_object is not None and self.model_name == "astar_oracle":
                self.simulation.unlock_attached_height()
                self.simulation.release_object()
            self.simulation.step(20)
            if self._is_task_complete(body_id, destination, is_so101) and self.simulation.collision_count == 0:
                failure_reason = None

        wall_elapsed = time.perf_counter() - started_wall
        final_obj = self.simulation.object_position(body_id)
        task_ok = failure_reason is None and self._is_task_complete(body_id, destination, is_so101)

        metrics = compute_comprehensive_metrics(
            object_samples=self.object_samples,
            joint_samples=self.joint_samples,
            ee_samples=self.ee_samples,
            destination_xy=destination,
            final_object_xyz=final_obj,
            collision_count=float(self.simulation.collision_count),
            task_success=task_ok,
            hz=hz,
            chunk_size=self.chunk_size,
            arm_joint_count=5 if is_so101 else 2,
            model_latencies_s=self.model_latencies,
            spectrum_decode_latencies_s=self.spectrum_decode_latencies,
            compression_ratios=self.compression_ratios,
            oracle_path_length_m=spec.oracle_path_length_m,
            wall_time_s=wall_elapsed,
        )

        return {
            "model": self.model_name,
            "checkpoint": self.checkpoint,
            "robot": spec.robot,
            "seed": spec.seed,
            "prompt": spec.prompt,
            "task_type": spec.task_type,
            "object": object_name,
            "destination": list(destination),
            "fast_tokenization": self.spectrum_module.fast_tokenization,
            "spectrum_schema": self.spectrum_module.schema_name,
            "freq_band": list(self.spectrum_module.freq_band),
            "failure_reason": failure_reason,
            **metrics,
            "trajectory": self.object_samples,
            "joint_trajectory": self.joint_samples,
        }

    def _step_so101(self, policy_action: np.ndarray, object_name: str, steps: int) -> None:
        assert isinstance(self.simulation, SO101Simulation)
        command = self.simulation.policy_to_simulator(policy_action)
        lower, upper = self.simulation.joint_limits()
        command = np.clip(command, lower, upper)
        current = self.simulation.joint_states()[0]
        if np.any(np.abs(command - current) > np.deg2rad(45.0)):
            raise ValueError("Action exceeds SO-101 per-step joint displacement limit")
        self.simulation.command_joints(command)
        body_id = self.simulation.object_ids[object_name]
        policy_gripper = float(policy_action[5])
        if (
            policy_gripper <= 25.0
            and self.simulation.attached_object is None
            and self.simulation.object_between_jaws(body_id)
        ):
            self.simulation.attach_object(object_name)
        elif policy_gripper >= 40.0 and self.simulation.attached_object is not None:
            self.simulation.unlock_attached_height()
            self.simulation.release_object()

        before = self.simulation.collision_count
        self.simulation.step(steps)
        if self.simulation.collision_count > before:
            raise ValueError("Action caused an obstacle collision")
        self.joint_samples.append(command.tolist())

    def _step_scara(self, policy_action: np.ndarray, object_name: str, steps: int) -> None:
        command = np.asarray(policy_action[:3], dtype=float)
        command[0] = float(np.clip(command[0], -math.pi, math.pi))
        command[1] = float(np.clip(command[1], -math.pi, math.pi))
        command[2] = float(np.clip(command[2], -0.34, 0.0))
        self.simulation.command_joints(command)
        body_id = self.simulation.object_ids[object_name]
        distance_xy = float(
            np.linalg.norm(
                self.simulation.tool_position()[:2] - self.simulation.object_position(body_id)[:2]
            )
        )
        if policy_action[3] >= 0.5 and self.simulation.attached_object is None and distance_xy < 0.065:
            self.simulation.attach_object(object_name)
        elif policy_action[3] < 0.5 and self.simulation.attached_object is not None:
            self.simulation.unlock_attached_height()
            self.simulation.release_object()

        before = self.simulation.collision_count
        self.simulation.step(steps)
        if self.simulation.collision_count > before:
            raise ValueError("Action caused an obstacle collision")
        self.joint_samples.append(command.tolist())

    def _is_task_complete(
        self, body_id: int, destination: Sequence[float], is_so101: bool
    ) -> bool:
        tol = 0.035 if is_so101 else 0.05
        pos = self.simulation.object_position(body_id)
        return self.simulation.attached_object is None and float(
            np.linalg.norm(pos[:2] - np.asarray(destination[:2], dtype=float))
        ) < tol


def resolve_scene_from_prompt(
    simulation: ManipulatorSimulation,
    prompt: str,
    seed: Optional[int],
) -> SceneSpec:
    """Resolve a natural-language prompt on a live simulation into a SceneSpec."""
    if isinstance(simulation, SO101Simulation):
        obj_name, dest = resolve_so101_task(simulation, prompt)
    else:
        positions = {
            name: simulation.object_position(body)[:2]
            for name, body in simulation.object_ids.items()
        }
        action = PromptResolver().resolve(prompt, positions)
        obj_name, dest = action.object_name, action.destination
        ctrl = BasicPickPlaceController(simulation)
        planner = ctrl._planner(obj_name)
        src_xy = positions[obj_name]
        try:
            planner.plan(src_xy, dest)
        except ValueError:
            best_pt = None
            best_dist = float("inf")
            for gx in np.linspace(-0.35, 0.45, 17):
                for gy in np.linspace(-0.35, 0.35, 15):
                    cand = (float(gx), float(gy))
                    dist = float(np.linalg.norm(np.asarray(cand) - np.asarray(dest)))
                    if dist >= best_dist or float(np.linalg.norm(np.asarray(cand) - np.asarray(src_xy))) < 0.08:
                        continue
                    try:
                        planner.plan(src_xy, cand)
                        best_pt = cand
                        best_dist = dist
                    except ValueError:
                        continue
            if best_pt is not None:
                dest = best_pt
        if any(w in prompt.lower() for w in ("target", "destination", "goal")):
            simulation.set_goal_marker(dest)

    task_type = (
        "target"
        if any(w in prompt.lower() for w in ("target", "destination", "goal", "green", "blue", "yellow"))
        else "spatial_relation"
    )
    return capture_scene_spec(
        simulation,
        prompt=prompt,
        target_object=obj_name,
        destination_xy=dest,
        seed=seed,
        task_type=task_type,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("so101", "scara"), default="so101")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-metadata", type=Path, help="Optional Golden Data metadata.json to load exact PyBullet setup")
    parser.add_argument("--prompt", default="Move the apple to the target")
    parser.add_argument("--model", choices=("astar_oracle", "smolvla", "fast_vla"), default="astar_oracle")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--inference-host", default="127.0.0.1")
    parser.add_argument("--inference-port", type=int, default=5478)
    parser.add_argument("--fast-tokenization", action="store_true", help="Enable InverseSpectrumModule reconstruction")
    parser.add_argument(
        "--spectrum-schema",
        default="direct",
        help="Spectrum schema: direct, fast_idct, band_idct, ifft, idwt_haar, idwt_db4, custom_hybrid",
    )
    parser.add_argument("--freq-band", default="0.0,1.0", help="Normalized frequency band 'low,high'")
    parser.add_argument("--quant-scale", type=float, default=20.0)
    parser.add_argument("--gui", action="store_true", help="Enable real-time PyBullet 3D GUI")
    parser.add_argument("--hold", type=float, default=0.0, help="Seconds to keep GUI open after completion")
    parser.add_argument("--max-chunks", type=int, default=25)
    parser.add_argument("--chart", type=Path, default=Path("artifacts/modular_run.png"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/modular_run.json"))
    args = parser.parse_args()

    freq_band = parse_freq_band(args.freq_band)
    schema_name = args.spectrum_schema
    if args.fast_tokenization and schema_name in ("direct", "none"):
        schema_name = "fast_idct"
    spectrum_mod = create_spectrum_module(
        schema=schema_name if (args.fast_tokenization or schema_name != "direct") else "direct",
        freq_band=freq_band,
        quant_scale=args.quant_scale,
    )

    if args.episode_metadata:
        raw_meta = json.loads(args.episode_metadata.read_text())
        spec = SceneSpec.from_episode_metadata(raw_meta)
    else:
        sim_cls = SO101Simulation if args.robot == "so101" else ManipulatorSimulation
        with sim_cls(gui=False, training_seed=args.seed) as temp_sim:
            spec = resolve_scene_from_prompt(temp_sim, args.prompt, args.seed)

    # Prepare policy adapter
    socket_policy: Optional[SocketActionChunkPolicy] = None
    oracle_xy: list[Sequence[float]] = []
    if args.model == "astar_oracle":
        oracle_actions, oracle_samples, oracle_len = generate_oracle_action_sequence(spec)
        spec.oracle_path_length_m = oracle_len
        oracle_xy = [row[1:3] for row in oracle_samples]
        policy: Any = PrecomputedChunkPolicy(oracle_actions, chunk_size=10)
    else:
        socket_policy = SocketActionChunkPolicy(args.inference_host, args.inference_port)
        policy = socket_policy

    sim_cls = SO101Simulation if spec.robot == "so101" else ManipulatorSimulation
    try:
        with sim_cls(gui=args.gui, training_seed=spec.seed) as simulation:
            apply_scene_spec(simulation, spec)
            pipeline = ModularExecutionPipeline(
                simulation=simulation,
                policy=policy,
                spectrum_module=spectrum_mod,
                model_name=args.model,
                checkpoint=args.checkpoint,
                realtime_gui=args.gui,
            )
            result = pipeline.execute_task(spec, max_chunks=args.max_chunks)
            workspace = (
                SO101Simulation.WORKSPACE if spec.robot == "so101" else SCARA_WORKSPACE
            )
            save_spectral_diagnostic_chart(
                path=args.chart,
                object_samples=pipeline.object_samples,
                joint_samples=pipeline.joint_samples,
                obstacles=simulation.obstacle_bounds,
                metrics=result,
                title=f"{spec.robot.upper()} | Model={args.model} | Spectrum={spectrum_mod.schema_name} {spectrum_mod.freq_band}",
                workspace=workspace,
                oracle_xy=oracle_xy,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k not in ("trajectory", "joint_trajectory", "psd_freqs_hz", "psd_power")}, indent=2))
            if args.gui and args.hold > 0:
                end_time = time.monotonic() + args.hold
                while p.isConnected(simulation.client_id) and time.monotonic() < end_time:
                    simulation.step()
                    time.sleep(simulation.TIME_STEP)
    finally:
        if socket_policy is not None:
            socket_policy.close()


if __name__ == "__main__":
    main()
