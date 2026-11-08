"""Stage D: closed-loop LIBERO evaluation on fixed initial states (identical across models).

Models:
  ckpt:<run>/<checkpoint>   adapter checkpoint from scripts/train.py (fast_matched / dwt_fixed / dwt_adaptive)
  ref:pi0fast_libero        lerobot/pi0fast-libero, unmodified, its own processors and decoder
  ref:pi0_libero            lerobot/pi0_libero_finetuned_v044, unmodified

Writes results/eval/<label>__<suite>_t<task>.json (per-episode + per-chunk telemetry), optional
videos under results/videos/, and merges a summary row per (model, task) into results/eval.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

from lerobot_policy_adaptive_dwt.metrics import wilson_ci  # noqa: E402


def load_model(spec: str):
    """Returns (label, predict(batch_after_env_pre) -> (H, 7) env-unit actions, policy_config, telemetry_fn)."""
    if spec.startswith("ckpt:"):
        from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import load_checkpoint
        from lerobot_policy_adaptive_dwt.processor_adaptive_dwt import make_processors

        path = ROOT / "runs" / spec[5:]
        policy, meta = load_checkpoint(path)
        stats = meta["norm_stats"]
        ds_stats = {"action": {k: torch.tensor(stats["action"][k], dtype=torch.float32) for k in ("mean", "std")},
                    "observation.state": {k: torch.tensor(stats["state"][k], dtype=torch.float32) for k in ("mean", "std")}}
        pre, post = make_processors(policy.config, ds_stats)

        def predict(batch):
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(pre(dict(batch)))
                return np.asarray(post(chunk).float().cpu())[0]

        def telemetry():
            return dict(policy.decode_log[-1]) if policy.decode_log else {}

        label = f"{meta['variant']}@{spec[5:].replace('/ckpt/', ':')}"
        return label, predict, policy.config, telemetry, int(policy.config.n_action_steps)
    if spec.startswith("ref:"):
        from smooth_operator.lerobot_policy import LeRobotVLAPolicy
        from smooth_operator.vla_models import MODEL_SPECS

        mspec = next(m for m in MODEL_SPECS if m["id"] == spec[4:])
        # Without DCT capture: an undecodable FAST stream falls back to LeRobot's own zero chunk instead of raising.
        vla = LeRobotVLAPolicy({**mspec, "embodiment": "libero", "native_output": "continuous"}, device="cuda")
        last = {}

        def predict(batch):
            t0 = time.perf_counter()
            actions, _ = vla.predict_frame(batch)
            torch.cuda.synchronize()
            last.clear()
            last.update(t_total=time.perf_counter() - t0, valid=None)
            return np.asarray(actions)

        if mspec.get("libero_env"):
            raise ValueError(f"{spec} needs LIBERO env options {mspec['libero_env']}; not supported here")
        return f"ref_{spec[4:]}", predict, vla.config, lambda: dict(last), int(getattr(vla.config, "n_action_steps", 10) or 10)
    raise ValueError(spec)


# lerobot/libero frames are 256x256; every model sees the same render size.
ENV_OPTIONS = {"observation_height": 256, "observation_width": 256}


def run_episode(scene, predict, telemetry, env_pre, env_post, instruction, init_state, max_steps, n_exec, video_path=None):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    obs, _ = scene.reset(init_state)
    step, success, chunks = 0, False, []
    actions_log, frames = [], []
    while step < max_steps and not success:
        batch = preprocess_observation(obs)
        batch["task"] = [instruction]
        t0 = time.perf_counter()
        chunk = predict(env_pre(batch))
        wall = time.perf_counter() - t0
        tel = telemetry()
        tel["wall_s"] = wall
        chunks.append(tel)
        terminated = truncated = False
        for row in chunk[:n_exec]:
            act = env_post({ACTION: torch.as_tensor(np.asarray(row, dtype=np.float32)).unsqueeze(0)})[ACTION]
            act = np.asarray(act.detach().cpu() if hasattr(act, "detach") else act)
            actions_log.append(act.reshape(-1).tolist())
            obs, _, term, trunc, info = scene.vec.step(act)
            step += 1
            if video_path is not None:
                frames.append(np.ascontiguousarray(np.asarray(obs["pixels"]["image"][0])[::-1, ::-1]))
            success = bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0])
            terminated, truncated = bool(term[0]), bool(trunc[0])
            if success or terminated or truncated or step >= max_steps:
                break
        if terminated or truncated:
            break
    if video_path is not None and frames:
        from smooth_operator.video import encode_mp4_video

        video_path.parent.mkdir(parents=True, exist_ok=True)
        encode_mp4_video(frames, video_path, fps=20)
    a = np.asarray(actions_log)
    grip = (a[:, 6] > 0).astype(int) if len(a) else np.zeros(0, int)
    return {
        "init_state": init_state, "success": success, "steps": step, "chunks": len(chunks),
        "invalid_streams": int(sum(1 for c in chunks if c.get("valid") is False)),
        "tokens_per_chunk": float(np.mean([c["n_tokens"] for c in chunks if "n_tokens" in c])) if chunks and "n_tokens" in chunks[0] else None,
        "latency_ms": [1000 * c.get("t_total", c["wall_s"]) for c in chunks],
        "wall_ms": [1000 * c["wall_s"] for c in chunks],
        "grammar_violations_unconstrained": int(sum(c.get("violations", 0) for c in chunks)),
        "plans": [c.get("plan") for c in chunks if c.get("plan")],
        "fast_coef_counts": [c["n_coef"] for c in chunks if "n_coef" in c],
        "gripper_transitions": int((np.diff(grip) != 0).sum()) if len(grip) > 1 else 0,
        "action_norm_mean": float(np.linalg.norm(a[:, :6], axis=1).mean()) if len(a) else None,
        "action_jerk_mean": float(np.abs(np.diff(a[:, :6], 3, axis=0)).mean()) if len(a) > 3 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-ids", type=int, nargs="+", default=[0])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--first-init-state", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="run only N initial states per task, drawn without replacement from the --episodes range "
                         "(seeded by --sample-seed and the task id, so every model gets the same states)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = suite default")
    ap.add_argument("--videos", type=int, default=2, help="record the first N episodes per model/task")
    args = ap.parse_args()

    from lerobot.envs.factory import make_env_pre_post_processors

    from smooth_operator.libero_env import LiberoScene, load_libero_tasks

    catalog = load_libero_tasks()[args.suite]
    max_steps = args.max_steps or int(catalog["max_steps"])
    out_dir = ROOT / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "results" / "eval.json"
    for spec in args.models:
        label, predict, cfg, telemetry, n_exec = load_model(spec)
        scene = LiberoScene()
        for task_id in args.task_ids:
            instruction = catalog["tasks"][task_id]["language"]
            scene.open(args.suite, task_id, ENV_OPTIONS)
            env_pre, env_post = make_env_pre_post_processors(scene.cfg, cfg)
            episodes = []
            path = out_dir / f"{label.replace(':', '_').replace('/', '_')}__{args.suite}_t{task_id}.json"
            inits = [args.first_init_state + i for i in range(args.episodes)]
            if args.sample:
                rng = np.random.default_rng([args.sample_seed, task_id])
                inits = sorted(int(x) for x in rng.choice(inits, size=args.sample, replace=False))
            for i, init in enumerate(inits):
                vid = ROOT / "results" / "videos" / f"{path.stem}_init{init}.mp4" if i < args.videos else None
                ep = run_episode(scene, predict, telemetry, env_pre, env_post, instruction, init, max_steps, n_exec, vid)
                episodes.append(ep)
                print(f"{label} {args.suite}#{task_id} init {init}: {'SUCCESS' if ep['success'] else 'fail'} "
                      f"steps={ep['steps']} invalid={ep['invalid_streams']} tok={ep['tokens_per_chunk']} "
                      f"lat={np.mean(ep['latency_ms']):.0f}ms", flush=True)
                path.write_text(json.dumps({"model": label, "spec": spec, "suite": args.suite, "task_id": task_id,
                                            "instruction": instruction, "max_steps": max_steps, "n_exec": n_exec,
                                            "env_options": ENV_OPTIONS,
                                            "episodes": episodes}, indent=1))
            k, n = sum(e["success"] for e in episodes), len(episodes)
            lat = np.concatenate([e["latency_ms"] for e in episodes])
            lo, hi = wilson_ci(k, n)
            toks = [e["tokens_per_chunk"] for e in episodes if e["tokens_per_chunk"] is not None]
            row = {"model": label, "spec": spec, "task": f"{args.suite}#{task_id}", "instruction": instruction,
                   "successes": k, "episodes": n, "success_rate": k / n, "ci_low": lo, "ci_high": hi,
                   "init_states": [e["init_state"] for e in episodes],
                   "tokens_mean": float(np.mean(toks)) if toks else None,
                   "latency_ms_mean": float(lat.mean()), "latency_ms_p95": float(np.quantile(lat, 0.95)),
                   "invalid_streams": None if spec.startswith("ref:") else int(sum(e["invalid_streams"] for e in episodes)),
                   "chunks": int(sum(e["chunks"] for e in episodes)), "gpu": torch.cuda.get_device_name(0),
                   "measured": True, "time": time.time()}
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"rows": []}
            summary["rows"] = [r for r in summary["rows"] if not (r["model"] == label and r["task"] == row["task"])] + [row]
            summary_path.write_text(json.dumps(summary, indent=1))
            print(json.dumps({k2: row[k2] for k2 in ("model", "task", "successes", "episodes", "ci_low", "ci_high", "tokens_mean", "latency_ms_mean")}), flush=True)
        scene.close()
        del predict
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
