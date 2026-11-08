import numpy as np
import pytest
from conftest import synthetic_windows

from lerobot_policy_adaptive_dwt.bpe import BPEDWTCodec, load_codec
from lerobot_policy_adaptive_dwt.codec import CodecSpec
from lerobot_policy_adaptive_dwt.codec_window import WindowDWTCodec
from lerobot_policy_adaptive_dwt.oracle_dp import OracleWeights, solve
from lerobot_policy_adaptive_dwt.token_grammar import GrammarError

LEVELS = {10: 3, 32: 4}


@pytest.fixture(params=[10, 32], ids=["H10", "H32"])
def window_codec(request):
    h = request.param
    spec = CodecSpec(horizon=h, lengths=(2, 4, 8) if h == 10 else (4, 8, 16, 32))
    return WindowDWTCodec.fit(spec, synthetic_windows(h, n=400, seed=1), ["sym4", "haar"], "symmetric", LEVELS[h])


def test_transform_is_perfect_reconstruction(window_codec):
    c = window_codec
    np.testing.assert_allclose(c.S @ c.A, np.eye(c.spec.horizon), atol=1e-9)
    assert c.coefs_at(c.levels) == c.n_coef > c.spec.horizon  # symmetric padding adds coefficients
    if c.spec.horizon == 10:
        assert c.group_sizes == [2, 2, 4, 8]


def test_encode_decode_and_fsm(window_codec):
    c = window_codec
    ws = synthetic_windows(c.spec.horizon, n=30, seed=5)
    for w in ws:
        for k in range(c.levels + 1):
            toks, recon, _ = c.encode_plan(w, [(c.spec.horizon, k)])
            assert len(toks) == c.token_count(k, len(c.segments(toks)[0].flips))
            np.testing.assert_allclose(c.decode(toks), recon)
            fsm = c.fsm()
            for t in toks:
                assert t in fsm.allowed()
                fsm.step(t)
            assert fsm.done
            assert len(toks) <= c.max_stream_tokens
    full = np.stack([c.encode_plan(w, c.full_plan())[1] for w in ws])
    assert ((full - ws)[..., c.spec.cont_dims] ** 2).mean() < 1e-3


def test_fsm_rejects_bad_streams(window_codec):
    c = window_codec
    toks, _, _ = c.encode_plan(synthetic_windows(c.spec.horizon, n=1, seed=3)[0], [(c.spec.horizon, 1)])
    with pytest.raises(GrammarError):
        c.decode(toks[:-1])
    with pytest.raises(GrammarError):
        c.decode([c.vocab.len_token(8), *toks])
    with pytest.raises(GrammarError):
        c.decode(toks + [c.vocab.a_end])


def test_adaptive_depth_tracks_gamma(window_codec):
    c = window_codec
    ws = synthetic_windows(c.spec.horizon, n=40, seed=7)
    ks = {g: np.mean([solve(c, w, OracleWeights(gamma=g))[0][0][1] for w in ws]) for g in (1e-6, 1.0)}
    assert ks[1e-6] > ks[1.0]
    assert ks[1e-6] == c.levels


def test_bpe_over_window_codec_roundtrip(window_codec):
    c = window_codec
    train = synthetic_windows(c.spec.horizon, n=300, seed=2)
    w = OracleWeights(gamma=0.005)
    plans = [solve(c, x, w)[0] for x in train]
    bpe = BPEDWTCodec.fit(c, train, plans)
    again = load_codec(bpe.to_dict())
    assert isinstance(again.base, WindowDWTCodec)
    assert isinstance(load_codec(c.to_dict()), WindowDWTCodec)
    for x in synthetic_windows(c.spec.horizon, n=20, seed=8):
        plan = solve(again, x, w)[0]
        base, recon, _ = c.encode_plan(x, plan)
        toks, _, _ = again.encode_plan(x, plan)
        assert again.expand(toks) == base and len(toks) <= len(base)
        np.testing.assert_allclose(again.decode(toks), recon)
        assert again.segments(toks)[0].res == plan[0][1]
        fsm = again.fsm()
        for t in toks:
            assert t in fsm.allowed()
            fsm.step(t)
        assert fsm.done


@pytest.mark.parametrize("h", [10, 32])
def test_dct_window_codec(h):
    import scipy.fft

    from lerobot_policy_adaptive_dwt.codec_window import dct_bands

    spec = CodecSpec(horizon=h, lengths=(2, 4, 8) if h == 10 else (4, 8, 16, 32))
    c = WindowDWTCodec.fit(spec, synthetic_windows(h, n=400, seed=1), ["dct"], "orthonormal", len(dct_bands(h)) - 1)
    assert sum(c.group_sizes) == c.n_coef == h
    x = np.random.default_rng(0).normal(size=h)
    np.testing.assert_allclose(c.A @ x, scipy.fft.dct(x, norm="ortho"))
    ws = synthetic_windows(h, n=30, seed=5)
    for w in ws:
        toks, recon, _ = c.encode_plan(w, c.full_plan())
        np.testing.assert_allclose(load_codec(c.to_dict()).decode(toks), recon)
    full = np.stack([c.encode_plan(w, c.full_plan())[1] for w in ws])
    # at H=32 a 16-coefficient band shares one step, so only the H=10 band layout is held to the tight bound
    assert ((full - ws)[..., c.spec.cont_dims] ** 2).mean() < (1e-3 if h == 10 else 5e-3)
