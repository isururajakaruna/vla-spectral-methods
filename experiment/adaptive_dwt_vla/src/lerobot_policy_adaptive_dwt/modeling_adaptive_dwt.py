"""Adaptive-DWT policy on the π₀-FAST PaliGemma backbone (no upstream modifications).

* ``token_forward``: teacher-forced next-token CE identical to ``PI0FastPytorch.forward`` but
  also returning per-position losses/predictions (per-token-class metrics).
* ``generate``: KV-cached greedy decoding. The prefix (images + prompt) is prefilled, then
  ``[BOS] "Action: "`` is fed causally exactly as in training, then action tokens are generated.
  DWT variants are constrained by the finite-state grammar at every step; the FAST baseline is
  decoded unconstrained until "|" (stock π₀-FAST behaviour).
* ``AdaptiveDWTPolicy``: ``PI0FastPolicy`` subclass whose ``predict_action_chunk`` uses the
  constrained decoder + inverse DWT and falls back to a logged safe action on failure.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.policies.common.vla_utils import prepare_attention_masks_4d
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from safetensors.torch import load_file, save_file

from .adapters import adapter_state_dict, attach_adapters, load_adapter_state_dict
from .codec import AdaptiveDWTCodec
from .configuration_adaptive_dwt import AdaptiveDWTConfig
from .processor_adaptive_dwt import ACTION_PREFIX_TEXT, local_to_pali, pali_to_local
from .token_grammar import GrammarError

BASE_CHECKPOINT = "lerobot/pi0fast-base"
BASE_REVISION = "308eb844dabdf46cd9dcd5138c1988b7a234f1d9"


# --------------------------------------------------------------------------- training loss
def token_forward(policy, batch) -> dict:
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    act_tokens = batch[ACTION_TOKENS]
    act_masks = batch[ACTION_TOKEN_MASK]
    embs, pad_masks, att_masks, _, _ = model.embed_prefix_fast(
        images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
        fast_action_tokens=act_tokens, fast_action_masks=act_masks,
    )
    pg = model.paligemma_with_expert
    if pg.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        embs = embs.to(torch.bfloat16)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (out, _), _ = pg.forward(
        attention_mask=prepare_attention_masks_4d(att_masks, dtype=embs.dtype), position_ids=position_ids,
        past_key_values=None, inputs_embeds=[embs, None], use_cache=False, adarms_cond=[None, None],
    )
    hidden = out[:, -act_tokens.shape[1] :, :]
    logits = pg.paligemma.lm_head(hidden)[:, :-1, :]  # position i predicts token i+1
    targets = act_tokens[:, 1:]
    mask = act_masks[:, 1:]
    per_tok = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    per_tok = per_tok.reshape(targets.shape)
    loss = (per_tok * mask).sum() / mask.sum().clamp(min=1)
    return {"loss": loss, "per_token": per_tok.detach(), "pred": logits.detach().argmax(-1), "targets": targets, "mask": mask}


def classify_targets(targets: torch.Tensor, variant: str, codec: AdaptiveDWTCodec | None, prefix_ids: list[int],
                     bar_id: int, eos_id: int) -> dict[str, torch.Tensor]:
    """Boolean masks per token class for the shifted targets (B, T-1)."""
    t = targets
    classes = {"prefix": torch.zeros_like(t, dtype=torch.bool)}
    for pid in prefix_ids:
        classes["prefix"] |= t == pid
    if variant == "fast_matched":
        loc = (t >= 256000) & (t <= 257023)
        classes["fast"] = loc
        classes["eos"] = (t == bar_id) | (t == eos_id)
        return classes
    local = pali_to_local(t.cpu().numpy())
    v = codec.vocab
    in_range = (local >= 0) & (local < v.size)
    cls = np.full(local.shape, "", dtype=object)
    cls[in_range] = np.vectorize(v.token_class)(local[in_range]) if in_range.any() else []
    for name in ("len", "res", "coef", "grip", "eos"):
        classes[name] = torch.from_numpy(cls == name).to(t.device)
    return classes


# --------------------------------------------------------------------------- decoding
@torch.no_grad()
def generate(policy, batch, *, fsm_factory=None, stop_ids: set[int] | None = None, max_new: int = 128,
             temperature: float = 0.0) -> dict:
    """Greedy KV-cached decoding. Returns generated PaliGemma ids (after the forced prefix) per sample."""
    model = policy.model
    pg = model.paligemma_with_expert
    tok = policy._paligemma_tokenizer
    device = next(policy.parameters()).device
    t0 = time.perf_counter()
    images, img_masks = policy._preprocess_images(batch)
    lang, lang_mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    bsize = lang.shape[0]
    embs, pad_masks, att_masks, _, _ = model.embed_prefix_fast(images, img_masks, lang, lang_mask)
    bf16 = pg.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    if bf16:
        embs = embs.to(torch.bfloat16)
    (_, _), past = pg.forward(
        attention_mask=prepare_attention_masks_4d(att_masks, dtype=embs.dtype),
        position_ids=torch.cumsum(pad_masks, dim=1) - 1, past_key_values=None,
        inputs_embeds=[embs, None], use_cache=True, adarms_cond=[None, None],
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    cur_mask = pad_masks
    forced = [tok.bos_token_id, *tok.encode(ACTION_PREFIX_TEXT, add_special_tokens=False)]
    fsms = [fsm_factory() for _ in range(bsize)] if fsm_factory else None
    out_ids: list[list[int]] = [[] for _ in range(bsize)]
    finished = [False] * bsize
    violations = [0] * bsize  # steps where the unconstrained argmax would have broken the grammar
    next_tok = torch.full((bsize, 1), forced[0], dtype=torch.long, device=device)
    step_i, n_forced = 0, len(forced)
    while True:
        emb = pg.embed_language_tokens(next_tok)
        if bf16:
            emb = emb.to(torch.bfloat16)
        cur_mask = torch.cat([cur_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        pos = (cur_mask.sum(dim=1, keepdim=True) - 1).long()
        (step_out, _), past = pg.forward(
            attention_mask=prepare_attention_masks_4d(cur_mask.unsqueeze(1), dtype=emb.dtype), position_ids=pos,
            past_key_values=past, inputs_embeds=[emb, None], use_cache=True, adarms_cond=[None, None],
        )
        step_i += 1
        if step_i < n_forced:
            next_tok = torch.full((bsize, 1), forced[step_i], dtype=torch.long, device=device)
            continue
        logits = pg.paligemma.lm_head(step_out[:, -1, :]).float()  # (B, V)
        if all(finished) or step_i - n_forced >= max_new:
            break
        chosen = []
        for b in range(bsize):
            if finished[b]:
                chosen.append(tok.pad_token_id)
                continue
            row = logits[b]
            if fsms is not None:
                allowed_local = fsms[b].allowed()
                allowed = torch.as_tensor(local_to_pali(allowed_local), device=device)
                if int(row.argmax()) not in set(allowed.tolist()):
                    violations[b] += 1
                masked = torch.full_like(row, float("-inf"))
                masked[allowed] = row[allowed]
                row = masked
            if temperature > 0:
                nid = int(torch.multinomial(torch.softmax(row / temperature, -1), 1))
            else:
                nid = int(row.argmax())
            out_ids[b].append(nid)
            if fsms is not None:
                fsms[b].step(int(pali_to_local(nid)))
                finished[b] = fsms[b].done
            elif stop_ids and nid in stop_ids:
                finished[b] = True
            chosen.append(nid)
        if all(finished):
            break
        next_tok = torch.tensor(chosen, dtype=torch.long, device=device).unsqueeze(1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"ids": out_ids, "finished": finished, "violations": violations,
            "t_prefill": t_prefill, "t_total": time.perf_counter() - t0}


def decode_fast_strict(policy, ids: list[int], horizon: int, action_dim: int) -> tuple[np.ndarray, bool, int]:
    """FAST decode of generated ids (cut at "|") -> (actions, valid, n_coefficients).

    valid=False if the stream has no "|" or does not decode to exactly H*D coefficients; the actions are
    still upstream's relaxed decode (zero-pad / truncate to H*D), which is what stock π₀-FAST executes.
    """
    bar = policy._paligemma_tokenizer.convert_tokens_to_ids("|")
    seq = ids[: ids.index(bar)] if bar in ids else ids
    local = [int(policy._paligemma_tokens_to_act_tokens(torch.tensor(i))) for i in seq]
    valid, n = bar in ids, -1
    try:
        n = len(policy.action_tokenizer.bpe_tokenizer.decode(local))
        valid = valid and n == horizon * action_dim
    except Exception:  # noqa: BLE001
        valid = False
    acts = policy.decode_actions_with_fast([local], time_horizon=horizon, action_dim=action_dim)[0]
    return np.asarray(acts, dtype=np.float64), valid, n


class AdaptiveDWTPolicy(PI0FastPolicy):
    config_class = AdaptiveDWTConfig
    name = "adaptive_dwt"

    def setup_codec(self, codec: AdaptiveDWTCodec, fallback_normalized: np.ndarray):
        self.codec = codec
        self.fallback_normalized = np.asarray(fallback_normalized, dtype=np.float64)
        self.decode_log: list[dict] = []

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        self.eval()
        horizon = self.codec.spec.horizon
        res = generate(self, batch, fsm_factory=self.codec.fsm, max_new=self.codec.max_stream_tokens + 4)
        out = []
        for b, ids in enumerate(res["ids"]):
            local = [int(x) for x in pali_to_local(np.asarray(ids))]
            info = {"n_tokens": len(ids), "violations": res["violations"][b], "t_prefill": res["t_prefill"]}
            t0 = time.perf_counter()
            try:
                acts = self.codec.decode(local)
                from .token_grammar import parse

                segs = parse(local, self.codec.vocab, horizon, tuple(self.codec.spec.lengths), self.codec.spec.n_cont)
                info.update(valid=True, plan=[[s.length, s.res] for s in segs])
            except GrammarError as e:
                logging.warning("invalid DWT stream, using safe fallback: %s", e)
                acts = np.tile(self.fallback_normalized, (horizon, 1))
                info.update(valid=False, error=str(e))
            info["t_total"] = res["t_total"] + (time.perf_counter() - t0)
            self.decode_log.append(info)
            out.append(acts)
        return torch.tensor(np.stack(out), dtype=torch.float32, device=batch[OBS_LANGUAGE_TOKENS].device)


class FastMatchedPolicy(PI0FastPolicy):
    """Stock π₀-FAST with our shared decoder (same prefix handling / timing as the DWT policy)."""

    def setup_codec(self, horizon: int, action_dim: int, fallback_normalized: np.ndarray):
        self.horizon, self.action_dim = horizon, action_dim
        self.fallback_normalized = np.asarray(fallback_normalized, dtype=np.float64)
        self.decode_log: list[dict] = []

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        self.eval()
        bar = self._paligemma_tokenizer.convert_tokens_to_ids("|")
        res = generate(self, batch, stop_ids={bar}, max_new=self.config.max_action_tokens)
        out = []
        for b, ids in enumerate(res["ids"]):
            acts, valid, n_coef = decode_fast_strict(self, ids, self.horizon, self.action_dim)
            self.decode_log.append({"n_tokens": len(ids), "valid": valid, "n_coef": n_coef,
                                    "t_prefill": res["t_prefill"], "t_total": res["t_total"]})
            out.append(acts)
        return torch.tensor(np.stack(out), dtype=torch.float32, device=batch[OBS_LANGUAGE_TOKENS].device)


# --------------------------------------------------------------------------- build / checkpoint
def make_config(variant: str, horizon: int, max_action_tokens: int, device: str = "cuda", empty_cameras: int = 0,
                **kw) -> AdaptiveDWTConfig:
    """``empty_cameras=0``: π₀-FAST masks empty-camera tokens out of attention and positions, so
    dropping the third (empty) camera is numerically equivalent and saves 256 sequence tokens."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    cfg = AdaptiveDWTConfig(
        chunk_size=horizon, n_action_steps=horizon, max_action_tokens=max_action_tokens,
        max_decoding_steps=max_action_tokens, action_tokenizer_name="jadechoghari/tokenizer-lib-mean",
        empty_cameras=empty_cameras, device=device, dtype="bfloat16",
        gradient_checkpointing=kw.pop("gradient_checkpointing", True),
        dwt_mode="fixed" if variant == "dwt_fixed" else "adaptive", **kw,
    )
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.MEAN_STD,
                                 "ACTION": NormalizationMode.MEAN_STD}
    cfg.__post_init__()
    return cfg


def action_token_ids(variant: str, codec: AdaptiveDWTCodec | None, policy) -> list[int]:
    tok = policy._paligemma_tokenizer
    special = [tok.convert_tokens_to_ids("|"), tok.eos_token_id]
    if variant == "fast_matched":
        n = policy.action_tokenizer.vocab_size
        return [int(i) for i in local_to_pali(np.arange(n))] + special
    return [int(i) for i in local_to_pali(np.arange(codec.vocab.size))]


def build_policy(variant: str, cfg: AdaptiveDWTConfig, codec: AdaptiveDWTCodec | None, fallback: np.ndarray,
                 rank: int, alpha: float):
    cls = FastMatchedPolicy if variant == "fast_matched" else AdaptiveDWTPolicy
    policy = cls.from_pretrained(BASE_CHECKPOINT, config=cfg, revision=BASE_REVISION, strict=False)
    if variant == "fast_matched":
        policy.setup_codec(cfg.chunk_size, 7, fallback)
    else:
        policy.setup_codec(codec, fallback)
    counts = attach_adapters(policy, rank, alpha, cfg.lora_targets, action_token_ids(variant, codec, policy))
    return policy, counts


def save_checkpoint(policy, path: Path, meta: dict, optimizer=None, min_free_gb: float = 10.0) -> int:
    """Adapter-only checkpoint (+ optional optimizer state). Refuses to write when disk is nearly full."""
    free = shutil.disk_usage(path.parent if path.parent.exists() else Path.cwd()).free / 1e9
    if free < min_free_gb:
        raise RuntimeError(f"refusing to save checkpoint: only {free:.1f} GB free (< {min_free_gb} GB)")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(adapter_state_dict(policy), str(tmp / "adapters.safetensors"))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    return sum(f.stat().st_size for f in path.iterdir())


def load_checkpoint(path: Path, device: str = "cuda"):
    meta = json.loads((Path(path) / "meta.json").read_text())
    variant = meta["variant"]
    codec = AdaptiveDWTCodec.from_dict(meta["codec"]) if meta.get("codec") else None
    cfg = make_config(variant, meta["horizon"], meta["max_action_tokens"], device=device,
                      empty_cameras=meta.get("empty_cameras", 0),
                      gradient_checkpointing=False, fixed_plan=meta.get("fixed_plan") or [],
                      oracle_gamma=meta.get("oracle_gamma", 0.005), lora_rank=meta["lora_rank"],
                      lora_alpha=meta["lora_alpha"])
    policy, _ = build_policy(variant, cfg, codec, np.asarray(meta["fallback_normalized"]), meta["lora_rank"], meta["lora_alpha"])
    load_adapter_state_dict(policy, load_file(str(Path(path) / "adapters.safetensors")))
    policy.eval()
    return policy, meta
