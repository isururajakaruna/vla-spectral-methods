"""Adaptive-DWT action token vocabulary, finite-state grammar and strict parser.

Local token layout (ids are *local*; the policy maps them onto PaliGemma ids):

    [0, n_bins)              coefficient bins (shared scalar vocabulary)
    LEN_L                    for L in LEN_VALUES (2, 4, 8, 16, 32)
    RES_k                    k = 0 .. MAX_RES (5 == log2 32)
    G0, G1                   initial binary gripper state of a segment
    GT_i                     gripper flip at offset i = 1 .. max_len-1 inside a segment
    G_END                    end of the segment's gripper events
    A_END                    end of the action stream

``<A_START>`` is the π₀-FAST prefix ``"Action: "`` which precedes every action
stream (shared with the FAST baseline), so it is not a local token.

Positions are implicit: given LEN and RES the decoder knows exactly how many
coefficient tokens follow and which (wavelet group, position, dimension) each
one belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

LEN_VALUES: tuple[int, ...] = (2, 4, 8, 16, 32)
MAX_RES = 5


class GrammarError(ValueError):
    """Raised when a token stream violates the action grammar."""


@dataclass(frozen=True)
class Vocab:
    n_bins: int = 256
    max_len: int = 32

    @property
    def len_offset(self) -> int:
        return self.n_bins

    @property
    def res_offset(self) -> int:
        return self.len_offset + len(LEN_VALUES)

    @property
    def g0(self) -> int:
        return self.res_offset + MAX_RES + 1

    @property
    def g1(self) -> int:
        return self.g0 + 1

    @property
    def gt_offset(self) -> int:
        # GT_i lives at gt_offset + i - 1, i = 1 .. max_len - 1
        return self.g1 + 1

    @property
    def g_end(self) -> int:
        return self.gt_offset + self.max_len - 1

    @property
    def a_end(self) -> int:
        return self.g_end + 1

    @property
    def size(self) -> int:
        return self.a_end + 1

    def len_token(self, length: int) -> int:
        return self.len_offset + LEN_VALUES.index(length)

    def res_token(self, k: int) -> int:
        if not 0 <= k <= MAX_RES:
            raise ValueError(f"resolution {k} out of range")
        return self.res_offset + k

    def gt_token(self, offset: int) -> int:
        if not 1 <= offset <= self.max_len - 1:
            raise ValueError(f"gripper offset {offset} out of range")
        return self.gt_offset + offset - 1

    def token_class(self, tok: int) -> str:
        if 0 <= tok < self.n_bins:
            return "coef"
        if self.len_offset <= tok < self.res_offset:
            return "len"
        if self.res_offset <= tok < self.g0:
            return "res"
        if self.g0 <= tok <= self.g_end:
            return "grip"
        if tok == self.a_end:
            return "eos"
        return "invalid"

    def describe(self, tok: int) -> str:
        cls = self.token_class(tok)
        if cls == "coef":
            return f"c{tok}"
        if cls == "len":
            return f"<LEN_{LEN_VALUES[tok - self.len_offset]}>"
        if cls == "res":
            return f"<RES_{tok - self.res_offset}>"
        if tok == self.g0:
            return "<G0>"
        if tok == self.g1:
            return "<G1>"
        if tok == self.g_end:
            return "<G_END>"
        if cls == "grip":
            return f"<GT_{tok - self.gt_offset + 1}>"
        if cls == "eos":
            return "<A_END>"
        return f"<INVALID_{tok}>"


@dataclass
class Segment:
    length: int
    res: int
    coef_bins: list[int]  # 6 * 2**res bins, coefficient-major then dimension
    gripper_init: int
    flips: list[int] = field(default_factory=list)


def n_coef_tokens(res: int, n_cont: int) -> int:
    return n_cont * (2**res)


def segment_token_count(res: int, n_flips: int, n_cont: int) -> int:
    """LEN + RES + coefficients + initial gripper + flips + G_END."""
    return 2 + n_coef_tokens(res, n_cont) + 1 + n_flips + 1


@lru_cache(maxsize=64)
def coverable(horizon: int, lengths: tuple[int, ...]) -> frozenset[int]:
    """Remaining-horizon values that can be covered exactly by ``lengths``."""
    ok = {0}
    for r in range(1, horizon + 1):
        if any(r - length in ok for length in lengths if length <= r):
            ok.add(r)
    return frozenset(ok)


class GrammarFSM:
    """Incremental grammar state used for both validation and constrained decoding."""

    SEG_START, RES, COEF, GRIP_INIT, GRIP_EVENTS, DONE = range(6)

    def __init__(self, vocab: Vocab, horizon: int, lengths: tuple[int, ...], n_cont: int = 6):
        if horizon not in coverable(horizon, tuple(lengths)):
            raise ValueError(f"horizon {horizon} cannot be covered by lengths {lengths}")
        for length in lengths:
            if length not in LEN_VALUES or length > vocab.max_len:
                raise ValueError(f"segment length {length} not supported")
        self.vocab = vocab
        self.horizon = horizon
        self.lengths = tuple(sorted(lengths))
        self.n_cont = n_cont
        self._cover = coverable(horizon, self.lengths)
        self.reset()

    def reset(self) -> None:
        self.state = self.SEG_START
        self.remaining = self.horizon
        self.cur_len = 0
        self.cur_res = 0
        self.coef_left = 0
        self.last_flip = 0
        self.segments: list[Segment] = []

    @property
    def done(self) -> bool:
        return self.state == self.DONE

    def allowed(self) -> list[int]:
        v = self.vocab
        if self.state == self.SEG_START:
            if self.remaining == 0:
                return [v.a_end]
            return [
                v.len_token(length)
                for length in self.lengths
                if length <= self.remaining and (self.remaining - length) in self._cover
            ]
        if self.state == self.RES:
            return [v.res_token(k) for k in range(self.cur_len.bit_length())]
        if self.state == self.COEF:
            return list(range(v.n_bins))
        if self.state == self.GRIP_INIT:
            return [v.g0, v.g1]
        if self.state == self.GRIP_EVENTS:
            return [v.gt_token(i) for i in range(self.last_flip + 1, self.cur_len)] + [v.g_end]
        return []

    def step(self, tok: int) -> None:
        v = self.vocab
        tok = int(tok)
        if self.state == self.DONE:
            raise GrammarError("token after <A_END>")
        allowed = self.allowed()
        if tok not in allowed:
            raise GrammarError(self._explain(tok))
        if self.state == self.SEG_START:
            if tok == v.a_end:
                self.state = self.DONE
                return
            self.cur_len = LEN_VALUES[tok - v.len_offset]
            self.state = self.RES
        elif self.state == self.RES:
            self.cur_res = tok - v.res_offset
            self.coef_left = n_coef_tokens(self.cur_res, self.n_cont)
            self.segments.append(Segment(self.cur_len, self.cur_res, [], 0))
            self.state = self.COEF
        elif self.state == self.COEF:
            self.segments[-1].coef_bins.append(tok)
            self.coef_left -= 1
            if self.coef_left == 0:
                self.state = self.GRIP_INIT
        elif self.state == self.GRIP_INIT:
            self.segments[-1].gripper_init = 0 if tok == v.g0 else 1
            self.last_flip = 0
            self.state = self.GRIP_EVENTS
        elif self.state == self.GRIP_EVENTS:
            if tok == v.g_end:
                self.remaining -= self.cur_len
                self.state = self.SEG_START
            else:
                offset = tok - v.gt_offset + 1
                self.segments[-1].flips.append(offset)
                self.last_flip = offset

    def _explain(self, tok: int) -> str:
        v = self.vocab
        cls = v.token_class(tok)
        name = v.describe(tok)
        if self.state == self.SEG_START:
            if self.remaining == 0:
                return f"expected <A_END> after {self.horizon} steps, got {name}"
            if tok == v.a_end:
                return f"premature <A_END>: {self.horizon - self.remaining}/{self.horizon} steps decoded"
            if cls == "len":
                return f"illegal {name}: {self.remaining} steps remaining (overrun or not coverable)"
            return f"expected <LEN_*> at segment start, got {name}"
        if self.state == self.RES:
            if cls == "res":
                return f"illegal {name} for <LEN_{self.cur_len}>"
            return f"expected <RES_*>, got {name}"
        if self.state == self.COEF:
            return f"expected coefficient ({self.coef_left} remaining), got {name}"
        if self.state == self.GRIP_INIT:
            return f"expected <G0>/<G1>, got {name}"
        if self.state == self.GRIP_EVENTS:
            if cls == "grip":
                return f"illegal {name}: offsets must increase within 1..{self.cur_len - 1}"
            return f"expected <GT_*> or <G_END>, got {name}"
        return f"unexpected {name}"


def parse(
    tokens: list[int], vocab: Vocab, horizon: int, lengths: tuple[int, ...], n_cont: int = 6
) -> list[Segment]:
    """Strictly parse a full stream ending in <A_END>. Raises GrammarError."""
    fsm = GrammarFSM(vocab, horizon, lengths, n_cont)
    for tok in tokens:
        fsm.step(tok)
    if not fsm.done:
        raise GrammarError(
            f"truncated stream: missing <A_END> ({fsm.horizon - fsm.remaining}/{fsm.horizon} steps decoded)"
        )
    return fsm.segments


def serialize(segments: list[Segment], vocab: Vocab) -> list[int]:
    out: list[int] = []
    for seg in segments:
        out.append(vocab.len_token(seg.length))
        out.append(vocab.res_token(seg.res))
        out.extend(int(b) for b in seg.coef_bins)
        out.append(vocab.g1 if seg.gripper_init else vocab.g0)
        out.extend(vocab.gt_token(i) for i in seg.flips)
        out.append(vocab.g_end)
    out.append(vocab.a_end)
    return out


def max_stream_tokens(horizon: int, lengths: tuple[int, ...], n_cont: int = 6) -> int:
    """Worst-case serialized length (all full resolution, max flips) for the token budget check."""
    best = {0: 0}
    for r in range(1, horizon + 1):
        cands = [
            best[r - length] + segment_token_count(length.bit_length() - 1, length - 1, n_cont)
            for length in lengths
            if length <= r and (r - length) in best
        ]
        if cands:
            best[r] = max(cands)
    return best[horizon] + 1
