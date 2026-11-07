"""Gate 2 (GPU): loss wiring, first-token supervision, gradients, and save/reload reproducibility."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def setup():
    from train import LENGTHS, WindowDataset

    from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec
    from lerobot_policy_adaptive_dwt.data import load_index, norm_stats, split_episodes, window_starts, windows
    from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import build_policy, make_config
    from lerobot_policy_adaptive_dwt.processor_adaptive_dwt import DWTActionTokenizerStep, make_processors

    idx, splits = load_index()
    eps = split_episodes(splits, [24], "train")
    stats = norm_stats(idx, eps)
    m, s = np.asarray(stats["action"]["mean"]), np.asarray(stats["action"]["std"])
    spec = CodecSpec(horizon=10, lengths=LENGTHS[10], gripper_low=(-1 - m[6]) / s[6], gripper_high=(1 - m[6]) / s[6])
    codec = AdaptiveDWTCodec.fit(spec, windows(idx, window_starts(idx, eps, 10, stride=2), 10, stats))
    fallback = np.concatenate([-m[:6] / s[:6], [(-1 - m[6]) / s[6]]])
    cfg = make_config("dwt_adaptive", 10, 96, gradient_checkpointing=False)
    policy, counts = build_policy("dwt_adaptive", cfg, codec, fallback, 8, 8.0)
    ds_stats = {"action": {k: torch.tensor(stats["action"][k]) for k in ("mean", "std")},
                "observation.state": {k: torch.tensor(stats["state"][k]) for k in ("mean", "std")}}
    step = DWTActionTokenizerStep(action_tokenizer_name=cfg.action_tokenizer_name, max_action_tokens=96,
                                  codec_state=codec.to_dict(), oracle_gamma=0.005)
    pre, post = make_processors(cfg, ds_stats, step)
    ds = WindowDataset(eps[:2], 10, splits, pick=np.arange(2))
    raw = torch.utils.data.default_collate([ds[0], ds[1]])
    return {"policy": policy, "counts": counts, "pre": pre, "post": post, "raw": raw, "codec": codec, "cfg": cfg,
            "stats": stats, "fallback": fallback, "step": step}


def test_trainable_parameters_are_only_adapters(setup):
    p = setup["policy"]
    names = [n for n, x in p.named_parameters() if x.requires_grad]
    assert names and all(("lora_" in n) or n.startswith("action_rows.") for n in names)
    assert setup["counts"]["lora_modules"] == 18 * 7
    assert not any("vision_tower" in n for n in names)


def test_first_len_token_is_supervised(setup):
    from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import token_forward
    from lerobot_policy_adaptive_dwt.processor_adaptive_dwt import pali_to_local

    policy, codec = setup["policy"], setup["codec"]
    from lerobot.utils.constants import ACTION_TOKENS

    batch = setup["pre"](dict(setup["raw"]))
    toks = batch[ACTION_TOKENS]
    tok = policy._paligemma_tokenizer
    prefix = tok.encode("Action: ", add_special_tokens=False)
    assert toks[0, 0].item() == tok.bos_token_id and toks[0, 1 : 1 + len(prefix)].tolist() == prefix
    j = 1 + len(prefix)  # first action-control token position
    assert codec.vocab.token_class(int(pali_to_local(toks[0, j].item()))) == "len"
    policy.zero_grad(set_to_none=True)
    out = token_forward(policy, batch)
    # shifted targets: position j-1 (the last prefix token, ' ') must predict the first LEN token, with loss
    assert out["targets"][0, j - 1].item() == toks[0, j].item()
    assert bool(out["mask"][0, j - 1])
    assert torch.isfinite(out["loss"])
    # the gradient of that single loss term reaches the LEN token's output row
    _forward_with_grad(policy, batch)["per_token_grad"][0, j - 1].backward()
    rows = policy.action_rows
    r = int((rows.token_ids == toks[0, j]).nonzero()[0])
    assert rows.delta_out.grad[r].abs().sum() > 0


def _forward_with_grad(policy, batch):
    import torch.nn.functional as F
    from lerobot.policies.common.vla_utils import prepare_attention_masks_4d
    from lerobot.utils.constants import (ACTION_TOKEN_MASK, ACTION_TOKENS, OBS_LANGUAGE_ATTENTION_MASK,
                                         OBS_LANGUAGE_TOKENS)

    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    act = batch[ACTION_TOKENS]
    embs, pad, att, _, _ = model.embed_prefix_fast(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                                   batch[OBS_LANGUAGE_ATTENTION_MASK], act, batch[ACTION_TOKEN_MASK])
    pg = model.paligemma_with_expert
    embs = embs.to(torch.bfloat16)
    (out, _), _ = pg.forward(attention_mask=prepare_attention_masks_4d(att, dtype=embs.dtype),
                             position_ids=torch.cumsum(pad, 1) - 1, past_key_values=None,
                             inputs_embeds=[embs, None], use_cache=False, adarms_cond=[None, None])
    logits = pg.paligemma.lm_head(out[:, -act.shape[1]:])[:, :-1].float()
    per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), act[:, 1:].reshape(-1), reduction="none")
    return {"per_token_grad": per.reshape(act[:, 1:].shape)}


def test_backward_gives_nonzero_adapter_grads(setup):
    from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import token_forward

    policy = setup["policy"]
    policy.zero_grad(set_to_none=True)
    out = token_forward(policy, setup["pre"](dict(setup["raw"])))
    out["loss"].backward()
    grads = {n: x.grad for n, x in policy.named_parameters() if x.requires_grad}
    assert all(g is not None for g in grads.values())
    nonzero = [n for n, g in grads.items() if g.abs().sum() > 0]
    # lora_b starts at zero, so only lora_b receives gradient on the very first step
    assert any("lora_b" in n for n in nonzero)
    assert "action_rows.delta_out" in nonzero and "action_rows.delta_in" in nonzero
    frozen = [x for n, x in policy.named_parameters() if not x.requires_grad]
    assert all(x.grad is None for x in frozen[:50])


def test_save_reload_reproduces_actions(setup, tmp_path):
    from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import load_checkpoint, save_checkpoint

    policy = setup["policy"]
    torch.manual_seed(0)
    with torch.no_grad():
        for n, x in policy.named_parameters():
            if x.requires_grad:
                x.add_(torch.randn_like(x) * (0.02 if "lora_b" in n or "delta" in n else 0.0))
    batch = setup["pre"]({k: v[:1] if hasattr(v, "shape") else v[:1] for k, v in setup["raw"].items() if k != "action"})
    a1 = policy.predict_action_chunk(batch).cpu()
    meta = {"variant": "dwt_adaptive", "horizon": 10, "max_action_tokens": 96, "lora_rank": 8, "lora_alpha": 8.0,
            "fixed_plan": [], "oracle_gamma": 0.005, "codec": setup["codec"].to_dict(), "norm_stats": setup["stats"],
            "fallback_normalized": setup["fallback"].tolist()}
    save_checkpoint(policy, tmp_path / "ck", meta, min_free_gb=1.0)
    size_mb = sum(f.stat().st_size for f in (tmp_path / "ck").iterdir()) / 1e6
    assert size_mb < 200, f"adapter checkpoint unexpectedly large: {size_mb:.0f} MB"
    policy2, _ = load_checkpoint(tmp_path / "ck")
    a2 = policy2.predict_action_chunk(batch).cpu()
    assert a1.shape == (1, 10, 7)
    torch.testing.assert_close(a1, a2, rtol=0, atol=0)
    assert policy.decode_log[-1]["valid"] and policy2.decode_log[-1]["valid"]
    print(json.dumps({"checkpoint_mb": size_mb, "decode": policy2.decode_log[-1]}))
    del policy2
    torch.cuda.empty_cache()
