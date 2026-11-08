import numpy as np
import pytest
from conftest import synthetic_windows

from lerobot_policy_adaptive_dwt.bpe import MAX_LOCAL, BPEDWTCodec, load_codec
from lerobot_policy_adaptive_dwt.token_grammar import GrammarError


@pytest.fixture
def bpe(fitted):
    h = fitted.spec.horizon
    plan = [(8, 3), (2, 1)] if h == 10 else [(32, 5)]
    train = synthetic_windows(h, n=300, seed=2)
    return BPEDWTCodec.fit(fitted, train, [plan] * len(train)), plan


def test_compress_is_lossless_and_shorter(bpe):
    codec, plan = bpe
    assert codec.vocab.size < codec.n_local <= MAX_LOCAL
    lens_base, lens_bpe = [], []
    for w in synthetic_windows(codec.spec.horizon, n=40, seed=9):
        base, recon, _ = codec.base.encode_plan(w, plan)
        toks, recon2, _ = codec.encode_plan(w, plan)
        assert codec.expand(toks) == base
        np.testing.assert_allclose(codec.decode(toks), recon)
        np.testing.assert_allclose(recon2, recon)
        lens_base.append(len(base))
        lens_bpe.append(len(toks))
    assert np.mean(lens_bpe) < np.mean(lens_base)


def test_fsm_accepts_compressed_stream_and_constrains(bpe):
    codec, plan = bpe
    w = synthetic_windows(codec.spec.horizon, n=1, seed=4)[0]
    toks, _, _ = codec.encode_plan(w, plan)
    fsm = codec.fsm()
    for t in toks:
        allowed = fsm.allowed()
        assert t in allowed
        for m in allowed:
            if m in codec.expansion:
                assert len(codec.expansion[m]) <= fsm.inner.coef_left
        fsm.step(t)
    assert fsm.done


def test_fsm_rejects_merge_that_overruns(bpe):
    codec, _ = bpe
    fsm = codec.fsm()
    longest = max(codec.expansion, key=lambda t: len(codec.expansion[t]))
    with pytest.raises(GrammarError):
        fsm.step(longest)  # stream must start with a LEN token


def test_serialization_roundtrip(bpe):
    codec, plan = bpe
    again = load_codec(codec.to_dict())
    assert isinstance(again, BPEDWTCodec) and again.n_local == codec.n_local and again.expansion == codec.expansion
    w = synthetic_windows(codec.spec.horizon, n=1, seed=5)[0]
    assert again.encode_plan(w, plan)[0] == codec.encode_plan(w, plan)[0]
