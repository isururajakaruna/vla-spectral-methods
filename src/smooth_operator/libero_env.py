"""LIBERO (MuJoCo, Franka Panda) scenes and rollouts for LeRobot VLAs fine-tuned on LIBERO.

Runs inside the Python 3.12 LeRobot environment. The task catalog loader is dependency-free so
the PyBullet studio can list suites and tasks without importing LIBERO.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

TASKS_PATH = Path(__file__).with_name("assets") / "libero_tasks.json"
LIBERO_FPS = 20
SCENE_VIEW_SIZE = (480, 360)
LIVE_FRAME_NAME = "live.jpg"
CAMERA_IDS = ("agentview", "wrist", "frontview")


def load_libero_tasks() -> dict[str, dict[str, Any]]:
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def _upright(image: np.ndarray) -> np.ndarray:
    """MuJoCo renders bottom-up; LeRobot's LIBERO processor applies the same 180° flip for models."""
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


class LiberoScene:
    """One LIBERO task environment, rebuilt only when the suite or task changes."""

    def __init__(self) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.key: Optional[tuple[str, int]] = None
        self.cfg = None
        self.vec = None

    def open(self, suite: str, task_id: int, options: Optional[dict[str, Any]] = None):
        key = (suite, task_id, json.dumps(options or {}, sort_keys=True))
        if self.key == key:
            return self.vec
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env

        self.close()
        self.cfg = LiberoEnv(task=suite, task_ids=[task_id], **(options or {}))
        self.vec = make_env(self.cfg, n_envs=1)[suite][task_id]
        self.key = key
        return self.vec

    def close(self) -> None:
        if self.vec is not None:
            self.vec.close()
        self.key, self.cfg, self.vec = None, None, None

    @property
    def gym_env(self):
        return self.vec.envs[0].unwrapped

    @property
    def mujoco_env(self):
        return self.gym_env._env.env

    def reset(self, init_state: int):
        self.gym_env.init_state_id = int(init_state)
        return self.vec.reset(seed=int(init_state))

    def object_positions(self) -> dict[str, np.ndarray]:
        env = self.mujoco_env
        return {name: np.array(env.sim.data.body_xpos[bid], dtype=float) for name, bid in env.obj_body_id.items()}

    def render_scene(self) -> np.ndarray:
        width, height = SCENE_VIEW_SIZE
        return np.ascontiguousarray(
            self.mujoco_env.sim.render(width=width, height=height, camera_name="frontview")[::-1]
        )


def _views(obs: dict[str, Any], scene: LiberoScene) -> dict[str, np.ndarray]:
    return {
        "agentview": _upright(obs["pixels"]["image"][0]),
        "wrist": _upright(obs["pixels"]["image2"][0]),
        "frontview": scene.render_scene(),
    }


def _png_b64(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_preview(suite: str, task_id: int, init_state: int = 0) -> dict[str, Any]:
    scene = LiberoScene()
    try:
        scene.open(suite, task_id)
        obs, _ = scene.reset(init_state)
        views = _views(obs, scene)
        return {
            "suite": suite,
            "task_id": task_id,
            "init_state": init_state,
            "task": scene.gym_env.task_description,
            "camera_images": {cam: _png_b64(rgb) for cam, rgb in views.items()},
            "objects": {name: [round(float(v), 4) for v in xyz] for name, xyz in scene.object_positions().items()},
        }
    finally:
        scene.close()


def _write_live_frame(run_dir: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    tmp = run_dir / (LIVE_FRAME_NAME + ".tmp")
    Image.fromarray(rgb).save(tmp, format="JPEG", quality=82)
    tmp.replace(run_dir / LIVE_FRAME_NAME)


class LiberoRollout:
    """Closed-loop LIBERO episodes with a loaded `LeRobotVLAPolicy` and a selectable decoder."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.scene = LiberoScene()

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        import torch
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        from .metrics import compute_comprehensive_metrics, save_spectral_diagnostic_chart
        from .spectrum import create_spectrum_module, decode_model_spectrum
        from .video import encode_mp4_video

        suite = str(request["suite"])
        task_id = int(request["task_id"])
        init_state = int(request.get("init_state", 0))
        catalog = load_libero_tasks()[suite]
        instruction = str(request.get("instruction") or catalog["tasks"][task_id]["language"])
        max_steps = int(request.get("max_steps") or catalog["max_steps"])
        decoder = str(request.get("decoder", "direct"))
        run_dir = Path(request["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        module = create_spectrum_module(
            decoder,
            freq_band=tuple(request.get("freq_band", (0.0, 1.0))),
            quant_scale=float(request.get("quant_scale", 20.0)),
        )
        if decoder != "direct" and not self.policy.emits_dct:
            raise ValueError(f"Decoder {decoder!r} needs a model that predicts DCT coefficients")

        wall_start = time.monotonic()
        self.scene.open(suite, task_id, self.policy.spec.get("libero_env"))
        env_pre, env_post = make_env_pre_post_processors(self.scene.cfg, self.policy.config)
        if hasattr(self.policy.model, "reset"):
            self.policy.model.reset()
        obs, _ = self.scene.reset(init_state)
        n_exec = int(getattr(self.policy.config, "n_action_steps", 0) or 10)

        frames: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMERA_IDS}
        ee_samples: list[tuple[float, ...]] = []
        joint_samples: list[list[float]] = []
        object_tracks: dict[str, list[np.ndarray]] = {name: [] for name in self.scene.mujoco_env.obj_body_id}
        model_latencies: list[float] = []
        decode_latencies: list[float] = []
        compression_ratios: list[float] = []
        prev_tail: Optional[np.ndarray] = None
        step = 0
        chunks = 0
        success = False

        def record(observation: dict[str, Any]) -> None:
            views = _views(observation, self.scene)
            for cam, rgb in views.items():
                frames[cam].append(rgb)
            if step % 2 == 0:
                _write_live_frame(run_dir, views["frontview"])
            t = step / LIBERO_FPS
            robot = observation["robot_state"]
            ee_samples.append((t, *np.asarray(robot["eef"]["pos"][0], dtype=float)))
            joints = np.asarray(robot["joints"]["pos"][0], dtype=float)
            gripper = float(np.abs(np.asarray(robot["gripper"]["qpos"][0], dtype=float)).sum())
            joint_samples.append([*joints.tolist(), gripper])
            for name, xyz in self.scene.object_positions().items():
                object_tracks[name].append(xyz)

        record(obs)
        while step < max_steps and not success:
            batch = preprocess_observation(obs)
            batch["task"] = [instruction]
            batch = env_pre(batch)
            t0 = time.perf_counter()
            actions, spectrum = self.policy.predict_frame(batch)
            t1 = time.perf_counter()
            if decoder == "direct":
                chunk = actions
                compression_ratios.append(1.0)
            else:
                chunk = decode_model_spectrum(module, spectrum, prev_chunk_tail=prev_tail)[: len(actions)]
                compression_ratios.append(float(spectrum.get("non_zero_ratio", 1.0)))
            model_latencies.append(t1 - t0)
            decode_latencies.append(time.perf_counter() - t1)
            chunks += 1
            executed = chunk[:n_exec]
            for row in executed:
                action = torch.as_tensor(np.asarray(row, dtype=np.float32)).unsqueeze(0)
                action = env_post({ACTION: action})[ACTION]
                obs, _, terminated, truncated, info = self.vec_step(action)
                step += 1
                record(obs)
                success = bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0])
                if success or bool(terminated[0]) or bool(truncated[0]) or step >= max_steps:
                    break
            prev_tail = executed[-2:].copy()
            if bool(terminated[0]) or bool(truncated[0]):
                break

        tracks = {name: np.asarray(xyz) for name, xyz in object_tracks.items() if xyz}
        moved = max(tracks, key=lambda n: float(np.linalg.norm(tracks[n][-1] - tracks[n][0])))
        times = np.asarray([s[0] for s in ee_samples])
        object_samples = [(t, *xyz) for t, xyz in zip(times, tracks[moved])]
        final_xyz = tracks[moved][-1]
        metrics = compute_comprehensive_metrics(
            object_samples=object_samples,
            joint_samples=joint_samples,
            ee_samples=ee_samples,
            destination_xy=final_xyz[:2],
            final_object_xyz=final_xyz,
            collision_count=0,
            task_success=success,
            hz=LIBERO_FPS,
            chunk_size=n_exec,
            arm_joint_count=7,
            model_latencies_s=model_latencies,
            spectrum_decode_latencies_s=decode_latencies,
            compression_ratios=compression_ratios,
            wall_time_s=time.monotonic() - wall_start,
        )
        # LIBERO scores success with its own goal predicates; there is no target point or collision count.
        for key in ("endpoint_error_m", "collision_count", "collision_steps"):
            metrics.pop(key, None)
        metrics["manipulated_object"] = moved
        metrics["object_displacement_m"] = round(float(np.linalg.norm(final_xyz - tracks[moved][0])), 4)
        metrics["max_vertical_lift_cm"] = round(float((tracks[moved][:, 2].max() - tracks[moved][0, 2]) * 100.0), 2)
        metrics["env_steps"] = step
        metrics["chunks"] = chunks

        xy = np.concatenate([np.asarray(ee_samples)[:, 1:3], np.asarray(object_samples)[:, 1:3]])
        lo, hi = xy.min(axis=0) - 0.05, xy.max(axis=0) + 0.05
        save_spectral_diagnostic_chart(
            path=run_dir / "diagnostic_chart.png",
            object_samples=object_samples,
            joint_samples=joint_samples,
            obstacles=[],
            metrics=metrics,
            title=f"LIBERO {suite} #{task_id} | {request.get('model_name', '')} | {decoder} | {instruction}",
            workspace=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
            chunk_size=n_exec,
        )
        videos = {cam: encode_mp4_video(frames[cam], run_dir / f"{cam}.mp4", fps=LIBERO_FPS) for cam in CAMERA_IDS}
        (run_dir / LIVE_FRAME_NAME).unlink(missing_ok=True)
        return {
            "success": success,
            "instruction": instruction,
            "metrics": {k: v for k, v in metrics.items() if k not in ("trajectory", "joint_trajectory", "psd_freqs_hz", "psd_power")},
            "videos": {cam: {"ok": ok, "frame_count": len(frames[cam])} for cam, ok in videos.items()},
            "video_fps": LIBERO_FPS,
        }

    def vec_step(self, action):
        return self.scene.vec.step(np.asarray(action.detach().cpu() if hasattr(action, "detach") else action))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a LIBERO task preview as JSON.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = render_preview(args.suite, args.task_id, args.init_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
