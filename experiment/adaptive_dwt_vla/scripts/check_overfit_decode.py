"""Gate 2: a tiny-overfit checkpoint must generate (constrained, KV-cached) the oracle stream of its training windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="overfit_dwt_adaptive")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS
    from train import WindowDataset

    from lerobot_policy_adaptive_dwt.data import load_index, split_episodes
    from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import load_checkpoint
    from lerobot_policy_adaptive_dwt.processor_adaptive_dwt import DWTActionTokenizerStep, make_processors, pali_to_local

    run = ROOT / "runs" / args.run
    cfg_run = json.loads((run / "config.json").read_text())
    policy, meta = load_checkpoint(run / "ckpt" / "best")
    stats = meta["norm_stats"]
    ds_stats = {"action": {k: torch.tensor(stats["action"][k]) for k in ("mean", "std")},
                "observation.state": {k: torch.tensor(stats["state"][k]) for k in ("mean", "std")}}
    step = DWTActionTokenizerStep(action_tokenizer_name=policy.config.action_tokenizer_name,
                                  max_action_tokens=meta["max_action_tokens"], codec_state=meta["codec"],
                                  dwt_mode=policy.config.dwt_mode, fixed_plan=meta["fixed_plan"],
                                  oracle_gamma=meta["oracle_gamma"])
    pre, _ = make_processors(policy.config, ds_stats, step)
    _, splits = load_index()
    eps = split_episodes(splits, meta["tasks"], "train")
    rng = np.random.default_rng(cfg_run["args"]["seed"])
    full = WindowDataset(eps, meta["horizon"], splits)
    pick = np.sort(rng.choice(len(full), size=cfg_run["args"]["overfit"], replace=False))
    ds = WindowDataset(eps, meta["horizon"], splits, pick=pick)
    exact = 0
    errs, lat = [], []
    for i in range(min(args.n, len(ds))):
        item = torch.utils.data.default_collate([ds[i]])
        batch = pre(dict(item))
        toks = batch[ACTION_TOKENS][0][batch[ACTION_TOKEN_MASK][0]].tolist()
        target = [int(pali_to_local(t)) for t in toks[4:]]  # drop BOS + "Action: "
        nb = {k: v for k, v in batch.items() if k not in (ACTION_TOKENS, ACTION_TOKEN_MASK, "action")}
        pred = policy.predict_action_chunk(nb)[0].cpu().numpy()
        info = policy.decode_log[-1]
        oracle = policy.codec.decode(target)
        exact += int(info["valid"] and np.allclose(pred, oracle, atol=1e-6))
        errs.append(float(np.abs(pred - oracle).max()))
        lat.append(info["t_total"])
        print(f"window {i}: valid={info['valid']} tokens={info['n_tokens']}/{len(target)} plan={info.get('plan')} "
              f"max|pred-oracle|={errs[-1]:.2e} latency={1000 * info['t_total']:.0f}ms", flush=True)
    res = {"run": args.run, "windows": len(errs), "exact_stream_reproduction": exact, "max_abs_err": max(errs),
           "latency_ms_mean": 1000 * float(np.mean(lat))}
    (run / "overfit_decode_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
