"""Parameter-efficient adapters for the PaliGemma/Gemma backbone of π₀-FAST.

* ``LoRALinear``: frozen base linear + trainable low-rank update (fp32 params).
* ``ActionRowAdapter``: trainable residual rows for a fixed set of vocabulary ids,
  applied to both the input embedding and the LM head (untied deltas), so the
  action-token embeddings and logits can adapt without training the 257k-row matrix.

Only these tensors are saved in checkpoints (tens of MB instead of the 5.8 GB backbone).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):  # some HF code paths inspect .weight.dtype
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = (x.to(torch.float32) @ self.lora_a.t()) @ self.lora_b.t()
        return y + (self.scale * delta).to(y.dtype)


class ActionRowAdapter(nn.Module):
    """Residual embedding/logit rows for ``token_ids`` (sorted, unique PaliGemma ids)."""

    def __init__(self, token_ids: list[int], hidden: int, vocab: int, device):
        super().__init__()
        ids = torch.tensor(sorted(set(int(i) for i in token_ids)), dtype=torch.long, device=device)
        self.register_buffer("token_ids", ids, persistent=True)
        lookup = torch.full((vocab,), -1, dtype=torch.long, device=device)
        lookup[ids] = torch.arange(len(ids), device=device)
        self.register_buffer("lookup", lookup, persistent=False)
        self.delta_in = nn.Parameter(torch.zeros(len(ids), hidden, dtype=torch.float32, device=device))
        self.delta_out = nn.Parameter(torch.zeros(len(ids), hidden, dtype=torch.float32, device=device))

    def embed_hook(self, module, inputs, output):
        ids = inputs[0]
        row = self.lookup[ids]
        hit = row >= 0
        if not bool(hit.any()):
            return output
        add = torch.zeros(output.shape, dtype=torch.float32, device=output.device)
        add[hit] = self.delta_in[row[hit]]
        return output + add.to(output.dtype)

    def head_hook(self, module, inputs, output):
        h = inputs[0]
        extra = (h.to(torch.float32) @ self.delta_out.t()).to(output.dtype)  # (..., n_rows)
        return output.index_add(-1, self.token_ids, extra)


def attach_adapters(policy, rank: int, alpha: float, targets: list[str], action_token_ids: list[int]) -> dict:
    """Freeze the backbone, add LoRA to Gemma LM layers and residual action rows. Returns counts."""
    pg = policy.model.paligemma_with_expert.paligemma
    for p in policy.parameters():
        p.requires_grad_(False)
    lm = pg.model.language_model
    n_wrapped = 0
    for layer in lm.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in targets:
                mod = getattr(parent, name, None)
                if isinstance(mod, nn.Linear):
                    setattr(parent, name, LoRALinear(mod, rank, alpha))
                    n_wrapped += 1
    emb = lm.get_input_embeddings()
    rows = ActionRowAdapter(action_token_ids, emb.weight.shape[1], emb.weight.shape[0], emb.weight.device)
    policy.action_rows = rows
    policy._adapter_hooks = [emb.register_forward_hook(rows.embed_hook), pg.lm_head.register_forward_hook(rows.head_hook)]
    trainable = [p for p in policy.parameters() if p.requires_grad]
    return {
        "lora_modules": n_wrapped,
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "total_params": int(sum(p.numel() for p in policy.parameters())),
        "action_rows": int(len(rows.token_ids)),
    }


def adapter_state_dict(policy) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in policy.named_parameters() if v.requires_grad} | {
        "action_rows.token_ids": policy.action_rows.token_ids.cpu()
    }


def load_adapter_state_dict(policy, state: dict[str, torch.Tensor]) -> None:
    params = dict(policy.named_parameters())
    ids = state["action_rows.token_ids"].to(policy.action_rows.token_ids.device)
    if not torch.equal(ids, policy.action_rows.token_ids):
        raise ValueError("checkpoint action-row ids differ from the policy's action vocabulary")
    missing = [k for k, p in params.items() if p.requires_grad and k not in state]
    if missing:
        raise KeyError(f"adapter checkpoint missing {len(missing)} tensors, e.g. {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            if k in params:
                params[k].copy_(v.to(params[k].device, params[k].dtype))
