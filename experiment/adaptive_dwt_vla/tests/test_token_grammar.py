import numpy as np
import pytest
from conftest import synthetic_windows

from lerobot_policy_adaptive_dwt.codec import enumerate_plans
from lerobot_policy_adaptive_dwt.token_grammar import (
    GrammarError,
    GrammarFSM,
    Vocab,
    coverable,
    max_stream_tokens,
    parse,
)


def _stream(fitted, seed=0, plan=None):
    win = synthetic_windows(fitted.spec.horizon, n=1, seed=seed)[0]
    win[:, 6] = -1.0
    win[3:, 6] = 1.0
    plan = plan or enumerate_plans(fitted.spec.horizon, fitted.spec.lengths)[5]
    return fitted.encode_plan(win, plan)[0]


def _parse(fitted, toks):
    return parse(toks, fitted.vocab, fitted.spec.horizon, fitted.spec.lengths)


def test_vocab_layout_is_disjoint():
    v = Vocab()
    classes = [v.token_class(t) for t in range(v.size)]
    assert classes.count("coef") == 256
    assert classes.count("len") == 5 and classes.count("res") == 6
    assert classes.count("eos") == 1 and "invalid" not in classes
    assert v.size == 256 + 5 + 6 + 2 + 31 + 1 + 1
    assert v.token_class(v.size) == "invalid"


def test_fsm_accepts_encoder_output(fitted):
    toks = _stream(fitted)
    fsm = fitted.fsm()
    for t in toks:
        assert t in fsm.allowed()
        fsm.step(t)
    assert fsm.done


def test_fsm_constrained_random_streams_always_parse(fitted):
    rng = np.random.default_rng(0)
    for _ in range(200):
        fsm = fitted.fsm()
        toks = []
        while not fsm.done:
            t = int(rng.choice(fsm.allowed()))
            fsm.step(t)
            toks.append(t)
        assert len(toks) <= fitted.max_stream_tokens
        assert fitted.decode(toks).shape == (fitted.spec.horizon, 7)


def test_max_stream_tokens_is_tight(fitted):
    spec = fitted.spec
    # all-full-resolution plan with a flip at every step reaches the bound exactly
    win = np.zeros((spec.horizon, 7))
    win[:, 6] = np.where(np.arange(spec.horizon) % 2, 1.0, -1.0)
    worst = max(len(fitted.encode_plan(win, [(L, L.bit_length() - 1) for L, _ in p])[0])
                for p in enumerate_plans(spec.horizon, spec.lengths, fixed_k=0))
    assert worst == max_stream_tokens(spec.horizon, spec.lengths)


def test_rejects_wrong_coefficient_count(fitted):
    toks = _stream(fitted)
    v = fitted.vocab
    i = toks.index(v.g0) if v.g0 in toks else toks.index(v.g1)
    with pytest.raises(GrammarError, match="expected coefficient"):
        _parse(fitted, toks[: i - 1] + toks[i:])  # one coefficient short
    with pytest.raises(GrammarError, match="expected <G0>/<G1>"):
        _parse(fitted, toks[:i] + [5] + toks[i:])  # one coefficient too many


def test_rejects_bad_token_ids(fitted):
    toks = _stream(fitted)
    with pytest.raises(GrammarError):
        _parse(fitted, [fitted.vocab.size + 3] + toks)
    with pytest.raises(GrammarError):
        _parse(fitted, [-1] + toks)


def test_rejects_illegal_len_res(fitted):
    v = fitted.vocab
    spec = fitted.spec
    with pytest.raises(GrammarError, match="RES"):
        _parse(fitted, [v.len_token(min(spec.lengths)), v.res_token(min(spec.lengths).bit_length())])
    illegal_len = [L for L in (2, 4, 8, 16, 32) if L not in spec.lengths][0]
    with pytest.raises(GrammarError, match="LEN"):
        _parse(fitted, [v.len_token(illegal_len)])


def test_rejects_gripper_offsets_outside_segment(fitted):
    v = fitted.vocab
    L = min(fitted.spec.lengths)
    prefix = [v.len_token(L), v.res_token(0)] + [0] * 6 + [v.g0]
    with pytest.raises(GrammarError, match="offsets"):
        _parse(fitted, prefix + [v.gt_token(L)])
    with pytest.raises(GrammarError, match="offsets"):
        _parse(fitted, prefix + [v.gt_token(1), v.gt_token(1)])


def test_rejects_overrun_underrun_premature_and_missing_eos(fitted):
    v = fitted.vocab
    toks = _stream(fitted)
    with pytest.raises(GrammarError, match="truncated"):
        _parse(fitted, toks[:-1])  # missing EOS
    with pytest.raises(GrammarError, match="truncated"):
        _parse(fitted, toks[: len(toks) // 2])  # underrun
    first_seg_end = toks.index(v.g_end) + 1
    with pytest.raises(GrammarError, match="premature"):
        _parse(fitted, toks[:first_seg_end] + [v.a_end])
    with pytest.raises(GrammarError):
        _parse(fitted, toks[:-1] + [v.len_token(min(fitted.spec.lengths))])  # overrun
    with pytest.raises(GrammarError, match="after <A_END>"):
        _parse(fitted, toks + [v.a_end])


def test_len_choices_only_fit_remaining_horizon():
    v = Vocab()
    fsm = GrammarFSM(v, horizon=10, lengths=(2, 4, 8))
    assert fsm.allowed() == [v.len_token(2), v.len_token(4), v.len_token(8)]
    for t in [v.len_token(8), v.res_token(0)] + [0] * 6 + [v.g0, v.g_end]:
        fsm.step(t)
    assert fsm.allowed() == [v.len_token(2)]
    assert v.a_end not in fsm.allowed()


def test_uncoverable_horizon_rejected():
    with pytest.raises(ValueError):
        GrammarFSM(Vocab(), horizon=10, lengths=(4, 8))
    assert 6 in coverable(10, (2, 4, 8)) and 5 not in coverable(10, (2, 4, 8))
