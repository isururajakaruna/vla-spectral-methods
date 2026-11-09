"""Fine-tune lerobot/pi0fast-base with LoRA + action-row adapters on LIBERO.

Variants: fast_matched (FAST libero tokenizer), dwt_fixed, dwt_adaptive, dwt_{fixed,adaptive}_bpe (BPE over the coefficients of the fixed / oracle plans),
dwt_{fixed,adaptive}_s4h[_bpe] (whole-window sym4→haar DWT, symmetric mode; fixed = full depth, adaptive = per-chunk depth),
dwt_fixed_compact[_bpe] (fixed plan without per-segment structure tokens; see --detail-scale/--deadzone),
dct_fixed (whole-chunk DCT-II, full resolution, same quantizer/grammar as DWT, no BPE),
dwt_fixed_lw / dwt_fixed_slw (dwt_fixed tokens; coefficient-token loss weighted by quantizer step / plus soft labels,
see token_forward). Same data, images, state,
prompt, normalization (training-split stats), horizon, optimizer schedule and effective batch.

Run directory runs/<name>/:
  config.json      run arguments, data episode ids, versions, trainable-parameter counts
  metrics.jsonl    one JSON line per log step (train) / eval step (val)  -> dashboard
  status.json      live progress (step, ETA, VRAM, state)
  ckpt/latest      adapters + optimizer (overwritten)          ~0.25 GB
  ckpt/best        adapters with the lowest val loss (overwritten)
  ckpt/step_XXXXX  adapters only, at --milestones
Nothing else is written: the 5.8 GB backbone is never copied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec  # noqa: E402
from lerobot_policy_adaptive_dwt.codec_compact import CompactDWTCodec  # noqa: E402
from lerobot_policy_adaptive_dwt.codec_window import WindowDWTCodec, dct_bands  # noqa: E402
from lerobot_policy_adaptive_dwt.data import load_index, norm_stats, split_episodes, window_starts, windows  # noqa: E402

REPO = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
LENGTHS = {10: (2, 4, 8), 32: (4, 8, 16, 32)}
WINDOW_LEVELS = {10: 3, 32: 4}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fast_matched", "dwt_fixed", "dwt_adaptive", "dwt_fixed_bpe", "dwt_adaptive_bpe", "dwt_fixed_s4h", "dwt_adaptive_s4h",
                                                       "dwt_fixed_s4h_bpe", "dwt_adaptive_s4h_bpe",
                                                       "dwt_fixed_compact", "dwt_fixed_compact_bpe", "dct_fixed",
                                                       "dwt_fixed_lw", "dwt_fixed_slw"])
    ap.add_argument("--soft-sigma", type=float, default=0.007,
                    help="dwt_fixed_slw: soft-label width in normalized action units (coefficient error)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[24])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--max-action-tokens", type=int, default=96)
    ap.add_argument("--detail-scale", type=float, default=1.0, help="multiply Haar detail quantization steps (1 = plain)")
    ap.add_argument("--detail-scheme", choices=["uniform", "fine"], default="uniform", help="fine: finest details coarsest")
    ap.add_argument("--deadzone", type=float, default=0.0, help="extra zero band for detail coefficients, in steps")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less VRAM, ~25%% slower)")
    ap.add_argument("--empty-cameras", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-windows", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--milestones", type=int, nargs="*", default=[2000, 5000, 10000])
    ap.add_argument("--overfit", type=int, default=0, help="train+validate on this many fixed training windows")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at", type=int, default=0, help="pause after this step (schedule still spans --steps)")
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    return ap.parse_args()


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class WindowDataset(torch.utils.data.Dataset):
    """LeRobotDataset restricted to frames whose H-step action window stays inside the episode."""

    def __init__(self, episodes: list[int], horizon: int, splits: dict, pick: np.ndarray | None = None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        kw = dict(revision=REVISION, episodes=episodes, delta_timestamps={"action": [i / splits["fps"] for i in range(horizon)]})
        try:
            self.ds = LeRobotDataset(REPO, **kw)
        except ValueError:
            # LeRobot only falls back to downloading on FileNotFoundError; episodes whose parquet
            # files are not cached yet surface as a datasets ValueError ("corresponds to no data").
            self.ds = LeRobotDataset(REPO, force_cache_sync=True, **kw)
        lengths = {}
        for t in splits["tasks"].values():
            lengths.update({int(k): v for k, v in t["episode_lengths"].items()})
        cols = self.ds.hf_dataset.select_columns(["episode_index", "frame_index"])
        ep = np.asarray(cols["episode_index"]).reshape(-1)
        fr = np.asarray(cols["frame_index"]).reshape(-1)
        ok = np.array([fr[i] + horizon <= lengths[int(ep[i])] for i in range(len(ep))])
        self.indices = np.nonzero(ok)[0]
        if pick is not None:
            self.indices = self.indices[pick]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.ds[int(self.indices[i])]
        assert not bool(item["action_is_pad"].any()), "window crosses the episode end"
        return {k: v for k, v in item.items() if k in (
            "observation.images.image", "observation.images.image2", "observation.state", "action", "task",
            "episode_index", "frame_index")}


def infinite(loader):
    while True:
        yield from loader


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = ROOT / "runs" / args.name
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    status_path, metrics_path = run / "status.json", run / "metrics.jsonl"

    def status(**kw):
        cur = json.loads(status_path.read_text()) if status_path.exists() else {}
        cur.update(kw, updated=time.time())
        status_path.write_text(json.dumps(cur, indent=1))

    status(state="starting", name=args.name, variant=args.variant, total_steps=args.steps, step=0, started=time.time())
    try:
        _train(args, run, status, metrics_path)
    except BaseException as e:  # noqa: BLE001
        status(state="failed", error=f"{type(e).__name__}: {e}")
        raise


def _train(args, run: Path, status, metrics_path: Path) -> None:
    from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import (
        build_policy, classify_targets, make_config, save_checkpoint, token_forward,
    )
    from lerobot_policy_adaptive_dwt.processor_adaptive_dwt import DWTActionTokenizerStep, make_processors

    idx, splits = load_index()
    train_eps = split_episodes(splits, args.tasks, "train")
    val_eps = split_episodes(splits, args.tasks, "val")
    stats = norm_stats(idx, train_eps)
    H = args.horizon
    a_mean, a_std = np.asarray(stats["action"]["mean"]), np.asarray(stats["action"]["std"])
    spec = CodecSpec(horizon=H, lengths=LENGTHS[H], gripper_low=(-1 - a_mean[6]) / a_std[6], gripper_high=(1 - a_mean[6]) / a_std[6])
    choice = json.loads((ROOT / "results" / f"codec_choice_H{H}.json").read_text())
    codec = None
    if args.variant != "fast_matched":
        cal = windows(idx, window_starts(idx, train_eps, H, stride=1), H, stats)
        if args.variant == "dct_fixed":
            codec = WindowDWTCodec.fit(spec, cal, ["dct"], "orthonormal", levels=len(dct_bands(H)) - 1)
            args.max_action_tokens = max(args.max_action_tokens, codec.max_stream_tokens + 8)
        elif "_s4h" in args.variant:
            codec = WindowDWTCodec.fit(spec, cal, ["sym4", "haar"], "symmetric", levels=WINDOW_LEVELS[H])
            args.max_action_tokens = max(args.max_action_tokens, codec.max_stream_tokens + 8)
        else:
            qkw = dict(detail_scale=args.detail_scale, detail_scheme=args.detail_scheme, deadzone=args.deadzone)
            codec = AdaptiveDWTCodec.fit(spec, cal, **qkw)
            if "_compact" in args.variant:
                codec = CompactDWTCodec(codec, choice["fixed"]["plan"])
    gamma = choice["adaptive"]["weights"]["gamma"]
    fixed_plan = []
    if args.variant.startswith(("dwt_fixed", "dct_fixed")):
        fixed_plan = [list(p) for p in (codec.full_plan() if hasattr(codec, "full_plan") else choice["fixed"]["plan"])]
    bpe_stats = None
    if args.variant.endswith("_bpe"):
        from lerobot_policy_adaptive_dwt.bpe import BPEDWTCodec
        from lerobot_policy_adaptive_dwt.oracle_dp import OracleWeights, solve

        w = OracleWeights(gamma=gamma)
        plans = [[tuple(p) for p in fixed_plan]] * len(cal) if fixed_plan else [solve(codec, x, w)[0] for x in cal]
        base_len = [len(codec.encode_plan(x, p)[0]) for x, p in zip(cal, plans)]
        codec = BPEDWTCodec.fit(codec, cal, plans)
        bpe_len = [len(codec.encode_plan(x, p)[0]) for x, p in zip(cal, plans)]
        bpe_stats = {"local_tokens": codec.n_local, "merged_tokens": len(codec.expansion),
                     "train_tokens_mean_before": float(np.mean(base_len)), "train_tokens_mean_after": float(np.mean(bpe_len))}
        print(f"BPE: {bpe_stats}", flush=True)
    # safe fallback = zero end-effector delta, gripper open (-1), in normalized units
    fallback = np.concatenate([(0 - a_mean[:6]) / a_std[:6], [(-1 - a_mean[6]) / a_std[6]]])

    cfg = make_config(args.variant, H, args.max_action_tokens, empty_cameras=args.empty_cameras,
                      gradient_checkpointing=args.grad_ckpt, fixed_plan=fixed_plan, oracle_gamma=gamma,
                      lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    t_load = time.time()
    policy, counts = build_policy(args.variant, cfg, codec, fallback, args.lora_rank, args.lora_alpha)
    t_load = time.time() - t_load
    ds_stats = {
        "action": {k: torch.tensor(stats["action"][k], dtype=torch.float32) for k in ("mean", "std")},
        "observation.state": {k: torch.tensor(stats["state"][k], dtype=torch.float32) for k in ("mean", "std")},
    }
    step_kw = dict(action_tokenizer_name=cfg.action_tokenizer_name, max_action_tokens=cfg.max_action_tokens,
                   fast_skip_tokens=cfg.fast_skip_tokens, paligemma_tokenizer_name=cfg.text_tokenizer_name)
    act_step = None
    if codec is not None:
        act_step = DWTActionTokenizerStep(**step_kw, codec_state=codec.to_dict(), dwt_mode=cfg.dwt_mode,
                                          fixed_plan=fixed_plan, oracle_gamma=gamma)
    pre, _ = make_processors(cfg, ds_stats, act_step)

    rng = np.random.default_rng(args.seed)
    if args.overfit:
        full = WindowDataset(train_eps, H, splits)
        pick = np.sort(rng.choice(len(full), size=args.overfit, replace=False))
        train_ds = WindowDataset(train_eps, H, splits, pick=pick)
        val_ds = train_ds
    else:
        train_ds = WindowDataset(train_eps, H, splits)
        vfull = WindowDataset(val_eps, H, splits)
        # the validation subset does not depend on --seed, so val curves are comparable across seeds
        val_rng = np.random.default_rng(0)
        val_ds = WindowDataset(val_eps, H, splits, pick=np.sort(val_rng.choice(len(vfull), size=min(args.val_windows, len(vfull)), replace=False)))
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True, drop_last=True,
                                         num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                                         pin_memory=True, prefetch_factor=4 if args.num_workers else None)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False, num_workers=2)

    tok = policy._paligemma_tokenizer
    loss_kw = dict(codec=codec, coef_weight=args.variant in ("dwt_fixed_lw", "dwt_fixed_slw"),
                   soft_sigma=args.soft_sigma if args.variant == "dwt_fixed_slw" else 0.0)
    prefix_ids = tok.encode("Action: ", add_special_tokens=False)
    bar_id = tok.convert_tokens_to_ids("|")
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    start_step, best_val = 0, float("inf")
    counters = {"chunks_seen": 0, "action_tokens_seen": 0, "gpu_seconds": 0.0}
    latest = run / "ckpt" / "latest"
    if args.resume and latest.exists():
        from safetensors.torch import load_file

        from lerobot_policy_adaptive_dwt.adapters import load_adapter_state_dict

        load_adapter_state_dict(policy, load_file(str(latest / "adapters.safetensors")))
        opt.load_state_dict(torch.load(latest / "optimizer.pt", map_location="cuda"))
        m = json.loads((latest / "meta.json").read_text())
        start_step, best_val = m["step"], m.get("best_val", best_val)
        counters.update(m.get("counters", {}))
        if metrics_path.exists():
            kept = [l for l in metrics_path.read_text().splitlines() if json.loads(l)["step"] <= start_step]
            metrics_path.write_text("".join(l + "\n" for l in kept))
        print(f"resumed from step {start_step}", flush=True)

    meta_base = {
        "variant": args.variant, "horizon": H, "max_action_tokens": args.max_action_tokens, "tasks": args.tasks,
        "empty_cameras": args.empty_cameras,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "fixed_plan": fixed_plan, "oracle_gamma": gamma,
        "codec": codec.to_dict() if codec else None, "norm_stats": stats, "fallback_normalized": fallback.tolist(),
        "loss": {"coef_weight": loss_kw["coef_weight"], "soft_sigma": loss_kw["soft_sigma"]},
        "action_tokenizer": cfg.action_tokenizer_name if args.variant == "fast_matched" else None,
        "base_checkpoint": "lerobot/pi0fast-base@308eb844dabdf46cd9dcd5138c1988b7a234f1d9",
        "dataset": f"{REPO}@{REVISION}",
    }
    import lerobot
    import transformers

    run_cfg = {
        "args": vars(args), "train_episodes": train_eps, "val_episodes": val_eps, "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds), "instructions": {t: splits["tasks"][str(t)]["instruction"] for t in args.tasks},
        "counts": counts, "bpe": bpe_stats, "model_load_s": t_load, "git": git_rev(), "gpu": torch.cuda.get_device_name(0),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "lerobot": lerobot.__version__,
                     "transformers": transformers.__version__},
        "effective_batch": args.micro_batch * args.grad_accum, **{k: meta_base[k] for k in ("base_checkpoint", "dataset", "oracle_gamma", "fixed_plan", "loss")},
    }
    (run / "config.json").write_text(json.dumps(run_cfg, indent=1))
    print(json.dumps({k: run_cfg[k] for k in ("n_train_windows", "n_val_windows", "counts", "effective_batch")}), flush=True)

    def log(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def class_stats(out, acc: dict):
        classes = classify_targets(out["targets"], args.variant, codec, prefix_ids, bar_id, tok.eos_token_id)
        m = out["mask"]
        correct = (out["pred"] == out["targets"]) & m
        for name, c in classes.items():
            c = c & m
            n = int(c.sum())
            if n:
                a = acc.setdefault(name, [0.0, 0, 0])
                a[0] += float(out["per_token"][c].sum())
                a[1] += int(correct[c].sum())
                a[2] += n
        a = acc.setdefault("all", [0.0, 0, 0])
        a[0] += float(out["per_token"][m].sum())
        a[1] += int(correct.sum())
        a[2] += int(m.sum())
        return int(m.sum())

    def flatten(acc: dict, prefix: str) -> dict:
        r = {}
        for name, (ls, cor, n) in acc.items():
            r[f"{prefix}loss_{name}"] = ls / max(1, n)
            r[f"{prefix}acc_{name}"] = cor / max(1, n)
        return r

    @torch.no_grad()
    def validate(step):
        policy.eval()
        acc: dict = {}
        plans = []
        for vb in val_loader:
            vb = pre(vb)
            out = token_forward(policy, vb)
            class_stats(out, acc)
            if act_step is not None:
                plans += [len(p) for p in act_step.last_plans]
        policy.train()
        rec = {"kind": "val", "step": step, "time": time.time(), **flatten(acc, "val_")}
        rec["val_loss"] = rec["val_loss_all"]
        log(rec)
        return rec["val_loss"]

    policy.train()
    it = infinite(loader)
    t_run = time.time()
    t_last, steps_since = time.time(), 0
    chunks_seen, tokens_seen, gpu_seconds = counters["chunks_seen"], counters["action_tokens_seen"], counters["gpu_seconds"]
    acc_win: dict = {}
    torch.cuda.reset_peak_memory_stats()
    end = min(args.steps, args.stop_at) if args.stop_at else args.steps
    status(state="running", step=start_step, stop_at=end, n_train_windows=len(train_ds),
           trainable_params=counts["trainable_params"])
    for step in range(start_step, end):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        t_data = 0.0
        for _ in range(args.grad_accum):
            td = time.time()
            batch = pre(next(it))
            t_data += time.time() - td
            out = token_forward(policy, batch, **loss_kw)
            (out["loss"] / args.grad_accum).backward()
            tokens_seen += class_stats(out, acc_win)
            chunks_seen += args.micro_batch
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps_since += 1
        done = step + 1
        if done % args.log_every == 0 or done == end:
            torch.cuda.synchronize()
            dt = (time.time() - t_last) / steps_since
            gpu_seconds += time.time() - t_last
            t_last, steps_since = time.time(), 0
            rec = {"kind": "train", "step": done, "time": time.time(), "lr": lr_at(step), "grad_norm": gnorm,
                   "sec_per_step": dt, "data_sec_per_step": t_data, "chunks_seen": chunks_seen,
                   "action_tokens_seen": tokens_seen, "vram_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "vram_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "gpu_hours": gpu_seconds / 3600,
                   **flatten(acc_win, "")}
            rec["loss"] = rec.get("loss_all")
            acc_win = {}
            log(rec)
            eta = dt * (end - done)
            status(step=done, sec_per_step=dt, eta_s=eta, loss=rec["loss"], vram_reserved_gb=rec["vram_reserved_gb"],
                   gpu_hours=rec["gpu_hours"])
            print(f"step {done}/{args.steps} loss {rec['loss']:.4f} acc {rec.get('acc_all', 0):.3f} "
                  f"{dt:.2f}s/step vram {rec['vram_reserved_gb']:.1f}GB eta {eta / 60:.1f}min", flush=True)
        if done % args.val_every == 0 or done == end:
            v = validate(done)
            status(val_loss=v)
            if v < best_val:
                best_val = v
                save_checkpoint(policy, run / "ckpt" / "best", {**meta_base, "step": done, "val_loss": v}, min_free_gb=args.min_free_gb)
        if done in args.milestones:
            save_checkpoint(policy, run / "ckpt" / f"step_{done:05d}", {**meta_base, "step": done}, min_free_gb=args.min_free_gb)
        if done % args.ckpt_every == 0 or done == end:
            counters = {"chunks_seen": chunks_seen, "action_tokens_seen": tokens_seen, "gpu_seconds": gpu_seconds}
            size = save_checkpoint(policy, run / "ckpt" / "latest",
                                   {**meta_base, "step": done, "best_val": best_val, "counters": counters},
                                   optimizer=opt, min_free_gb=args.min_free_gb)
            status(ckpt_latest_mb=size / 1e6)
    status(state="done" if end == args.steps else "paused", step=end, eta_s=0, wall_s=time.time() - t_run)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
