"""Record one closed-loop LIBERO episode as a 16:9 showcase video.

The policy sees exactly the evaluation inputs (256x256 agentview + wrist, same pre/post processors); the video comes
from an extra render of an otherwise unused scene camera ("paperview", re-aimed at a three-quarter view of the arm,
table and target), so it does not change the rollout. Nothing is written to results/eval*.

Usage: python scripts/record_video.py --model ckpt:full_s0_dwt_fixed_slw/ckpt/step_43500 --suite libero_10 --task-id 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_policy import ENV_OPTIONS, ROOT, load_model  # noqa: E402

CAMERA = "paperview"
VIEWS = {"three_quarter": ([1.2, -0.75, 1.6], [-0.3, 0.0, 1.2]), "front": ([1.45, 0.0, 1.75], [-0.25, 0.0, 1.05])}


def look_at(pos, target) -> np.ndarray:
    """MuJoCo camera quaternion (w, x, y, z) looking from pos at target with world z up."""
    from scipy.spatial.transform import Rotation

    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    x, y, z, w = Rotation.from_matrix(np.stack([right, up, -f], axis=1)).as_quat()
    return np.array([w, x, y, z])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--init-state", type=int, default=0)
    ap.add_argument("--view", choices=sorted(VIEWS), default="three_quarter")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hold", type=float, default=1.0, help="seconds to hold the last frame")
    args = ap.parse_args()

    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    from smooth_operator.libero_env import LIBERO_FPS, LiberoScene, load_libero_tasks
    from smooth_operator.video import encode_mp4_video

    catalog = load_libero_tasks()[args.suite]
    instruction = catalog["tasks"][args.task_id]["language"]
    max_steps = int(catalog["max_steps"])
    label, predict, cfg, _, n_exec = load_model(args.model)
    scene = LiberoScene()
    scene.open(args.suite, args.task_id, ENV_OPTIONS)
    env_pre, env_post = make_env_pre_post_processors(scene.cfg, cfg)
    obs, _ = scene.reset(args.init_state)
    sim = scene.mujoco_env.sim
    cid = sim.model.camera_name2id(CAMERA)
    pos, target = VIEWS[args.view]
    sim.model.cam_pos[cid] = pos
    sim.model.cam_quat[cid] = look_at(pos, target)

    def frame() -> np.ndarray:
        return np.ascontiguousarray(sim.render(width=args.width, height=args.height, camera_name=CAMERA)[::-1])

    frames, step, success = [frame()], 0, False
    while step < max_steps and not success:
        batch = preprocess_observation(obs)
        batch["task"] = [instruction]
        chunk = predict(env_pre(batch))
        done = False
        for row in chunk[:n_exec]:
            act = env_post({ACTION: torch.as_tensor(np.asarray(row, dtype=np.float32)).unsqueeze(0)})[ACTION]
            act = np.asarray(act.detach().cpu() if hasattr(act, "detach") else act)
            obs, _, term, trunc, info = scene.vec.step(act)
            step += 1
            frames.append(frame())
            success = bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0])
            done = success or bool(term[0]) or bool(trunc[0]) or step >= max_steps
            if done:
                break
        if done and not success:
            break
    frames += [frames[-1]] * int(args.hold * LIBERO_FPS)
    out = ROOT / "results" / "videos" / (
        f"showcase_{label.replace(':', '_').replace('/', '_')}__{args.suite}_t{args.task_id}_init{args.init_state}"
        f"_{args.view}_{args.width}x{args.height}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    encode_mp4_video(frames, out, fps=LIBERO_FPS)
    scene.close()
    print(f"{instruction!r}: {'SUCCESS' if success else 'fail'} in {step} steps -> {out}", flush=True)


if __name__ == "__main__":
    main()
