"""Byte-pair merges over the coefficient tokens of DWT action streams (the FAST idea applied to DWT).

Only runs of coefficient bins are merged, and a run never crosses a segment: structure tokens
(LEN, RES), gripper tokens and A_END are unchanged. A merged token stands for a fixed sequence of
bins, so decoding is exact and the grammar stays enforceable: while the FSM expects ``n`` more
coefficients, any merged token whose expansion has at most ``n`` bins is legal.

Local ids: ``[0, base.vocab.size)`` are the base DWT tokens, merged tokens follow, up to 1024 in
total (the number of PaliGemma ``<loc>`` rows the action ids are mapped onto).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .codec import AdaptiveDWTCodec, Plan
from .codec_compact import CompactDWTCodec
from .codec_window import WindowDWTCodec
from .token_grammar import GrammarError

# bins are written as these unicode characters for the Hugging Face BPE trainer
CHAR_OFFSET = 0x4E00
MAX_LOCAL = 1024


def _to_str(bins) -> str:
    return "".join(chr(CHAR_OFFSET + int(b)) for b in bins)


class BPEDWTCodec:
    def __init__(self, base: AdaptiveDWTCodec, tokenizer_json: str):
        from tokenizers import Tokenizer

        self.base = base
        self.spec, self.vocab = base.spec, base.vocab
        self.tokenizer_json = tokenizer_json
        self._tok = Tokenizer.from_str(tokenizer_json)
        n_bins = base.vocab.n_bins
        self.hf_to_local: dict[int, int] = {}
        self.expansion: dict[int, tuple[int, ...]] = {}
        nxt = base.vocab.size
        for s, hf_id in sorted(self._tok.get_vocab().items(), key=lambda kv: kv[1]):
            bins = tuple(ord(c) - CHAR_OFFSET for c in s)
            if any(not 0 <= b < n_bins for b in bins):
                raise ValueError(f"BPE token {s!r} is not a coefficient sequence")
            if len(bins) == 1:
                self.hf_to_local[hf_id] = bins[0]
            else:
                self.hf_to_local[hf_id] = nxt
                self.expansion[nxt] = bins
                nxt += 1
        self.n_local = nxt
        if self.n_local > MAX_LOCAL:
            raise ValueError(f"{self.n_local} local tokens exceed {MAX_LOCAL}")

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, base: AdaptiveDWTCodec, windows: np.ndarray, plans: list[Plan], vocab_total: int = MAX_LOCAL,
            min_frequency: int = 2) -> "BPEDWTCodec":
        from tokenizers import Tokenizer, models, trainers

        blocks = []
        for w, plan in zip(windows, plans, strict=True):
            toks, _, _ = base.encode_plan(w, plan)
            blocks += [_to_str(r) for r in coef_runs(toks, base.vocab.n_bins)]
        n_bins = base.vocab.n_bins
        tok = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=n_bins + vocab_total - base.vocab.size, min_frequency=min_frequency,
                                      initial_alphabet=[chr(CHAR_OFFSET + b) for b in range(n_bins)],
                                      special_tokens=[], show_progress=False, max_token_length=None)
        tok.train_from_iterator(blocks, trainer=trainer)
        return cls(base, tok.to_str())

    def to_dict(self) -> dict:
        return {**self.base.to_dict(), "bpe": {"tokenizer": self.tokenizer_json, "n_local": self.n_local}}

    @classmethod
    def from_dict(cls, d: dict) -> "BPEDWTCodec":
        return cls(_load_base(d), d["bpe"]["tokenizer"])

    # ------------------------------------------------------------- stream I/O
    @property
    def max_stream_tokens(self) -> int:
        return self.base.max_stream_tokens

    def compress(self, tokens: list[int]) -> list[int]:
        out, run = [], []
        n_bins = self.vocab.n_bins
        for t in list(tokens) + [-1]:
            if 0 <= t < n_bins:
                run.append(t)
                continue
            if run:
                out += [self.hf_to_local[i] for i in self._tok.encode(_to_str(run)).ids]
                run = []
            if t >= 0:
                out.append(t)
        return out

    def expand(self, tokens: list[int]) -> list[int]:
        out = []
        for t in tokens:
            out.extend(self.expansion.get(int(t), (int(t),)))
        return out

    def encode_plan(self, window: np.ndarray, plan: Plan):
        toks, recon, info = self.base.encode_plan(window, plan)
        return self.compress(toks), recon, info

    def decode(self, tokens: list[int]) -> np.ndarray:
        return self.base.decode(self.expand(tokens))

    def segments(self, tokens: list[int]):
        return self.base.segments(self.expand(tokens))

    def solve(self, window: np.ndarray, w):
        from .oracle_dp import solve

        return solve(self.base, window, w)

    def fsm(self) -> "BPEGrammarFSM":
        return BPEGrammarFSM(self)

    @lru_cache(maxsize=128)
    def merged_up_to(self, n: int) -> tuple[int, ...]:
        return tuple(t for t, e in self.expansion.items() if len(e) <= n)


class BPEGrammarFSM:
    """Base grammar FSM that also accepts merged coefficient tokens that fit the remaining count."""

    def __init__(self, codec: BPEDWTCodec):
        self.codec = codec
        self.inner = codec.base.fsm()

    @property
    def done(self) -> bool:
        return self.inner.done

    @property
    def state(self) -> int:
        return self.inner.state

    def allowed(self) -> list[int]:
        allowed = self.inner.allowed()
        if self.inner.state == self.inner.COEF:
            allowed = allowed + list(self.codec.merged_up_to(self.inner.coef_left))
        return allowed

    def step(self, tok: int) -> None:
        tok = int(tok)
        exp = self.codec.expansion.get(tok)
        if exp is None:
            self.inner.step(tok)
            return
        if self.inner.state != self.inner.COEF or len(exp) > self.inner.coef_left:
            raise GrammarError(f"merged coefficient token {tok} ({len(exp)} bins) not allowed here")
        for b in exp:
            self.inner.step(b)


def coef_runs(tokens: list[int], n_bins: int) -> list[list[int]]:
    runs, run = [], []
    for t in list(tokens) + [-1]:
        if 0 <= t < n_bins:
            run.append(int(t))
        elif run:
            runs.append(run)
            run = []
    return runs


def _load_base(state: dict):
    kind = state.get("kind")
    if kind == "window":
        return WindowDWTCodec.from_dict(state)
    if kind == "compact":
        return CompactDWTCodec.from_dict(state)
    return AdaptiveDWTCodec.from_dict(state)


def load_codec(state: dict):
    return BPEDWTCodec.from_dict(state) if state.get("bpe") else _load_base(state)
