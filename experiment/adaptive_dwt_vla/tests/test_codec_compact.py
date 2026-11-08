import numpy as np
import pytest
from conftest import synthetic_windows

from lerobot_policy_adaptive_dwt.bpe import BPEDWTCodec, load_codec
from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec
from lerobot_policy_adaptive_dwt.codec_compact import CompactDWTCodec
from lerobot_policy_adaptive_dwt.quantizer import CoefficientQuantizer
from lerobot_policy_adaptive_dwt.token_grammar import GrammarError

PLAN = [(8, 3), (2, 1)]
SPEC = CodecSpec(horizon=10, lengths=(2, 4, 8))


@pytest.fixture
def coarse():
    seg = AdaptiveDWTCodec.fit(SPEC, synthetic_windows(10, n=400, seed=1), detail_scale=6, detail_scheme="fine", deadzone=0.3)
    return CompactDWTCodec(seg, PLAN)


def test_default_quantizer_unchanged():
    w = synthetic_windows(10, n=200, seed=1)
    x = w[..., SPEC.cont_dims]
    a = CoefficientQuantizer(256, 1.0).fit(x, SPEC.lengths)
    b = CoefficientQuantizer.from_dict({k: v for k, v in a.to_dict().items() if k not in ("detail_scale", "detail_scheme", "deadzone")})
    for length in SPEC.lengths:
        np.testing.assert_array_equal(a.params[length][1], b.params[length][1])
        c = np.random.default_rng(0).normal(size=(length, 6))
        np.testing.assert_array_equal(a.quantize(c, length)[0], b.quantize(c, length)[0])


def test_coarse_quantizer_scales_details_and_zeroes_small_ones():
    x = synthetic_windows(10, n=200, seed=1)[..., SPEC.cont_dims]
    plain = CoefficientQuantizer(256, 1.0).fit(x, SPEC.lengths)
    fine = CoefficientQuantizer(256, 1.0, detail_scale=8, detail_scheme="fine", deadzone=0.3).fit(x, SPEC.lengths)
    sp, sf = plain.params[8][1], fine.params[8][1]
    np.testing.assert_allclose(sf[0], sp[0])
    np.testing.assert_allclose(sf[1:] / sp[1:], np.broadcast_to((8 ** (np.arange(1, 4) / 3))[:, None], sf[1:].shape))
    c = np.zeros((8, 6))
    c[1:] = 0.75 * fine._expand(8)[1][1:]  # 0.75 steps: rounds to 1 normally, to 0 inside the dead zone
    assert (fine.quantize(c, 8)[0][1:] == fine.mid).all()
    again = CoefficientQuantizer.from_dict(fine.to_dict())
    np.testing.assert_array_equal(again.quantize(c, 8)[0], fine.quantize(c, 8)[0])


def test_compact_roundtrip_matches_segment_codec(coarse):
    for w in synthetic_windows(10, n=40, seed=5):
        toks, recon, _ = coarse.encode_plan(w, PLAN)
        seg_toks, seg_recon, _ = coarse.seg.encode_plan(w, PLAN)
        np.testing.assert_allclose(recon, seg_recon)
        np.testing.assert_allclose(coarse.decode(toks), coarse.seg.decode(seg_toks))
        bits = coarse.seg.gripper_bits(w)
        assert len(toks) == coarse.n_coef_tokens + 2 + int((bits[1:] != bits[:-1]).sum())
        assert len(toks) <= coarse.max_stream_tokens
        fsm = coarse.fsm()
        for t in toks:
            assert t in fsm.allowed()
            fsm.step(t)
        assert fsm.done
        assert [(s.length, s.res) for s in coarse.segments(toks)] == PLAN


def test_compact_gripper_flip_across_segment_boundary(coarse):
    w = synthetic_windows(10, n=1, seed=3)[0].copy()
    w[:, SPEC.gripper_dim] = np.where(np.arange(10) < 9, SPEC.gripper_low, SPEC.gripper_high)
    toks, recon, _ = coarse.encode_plan(w, PLAN)
    segs = coarse.segments(toks)
    assert segs[0].flips == [] and segs[1].gripper_init == 0 and segs[1].flips == [1]
    np.testing.assert_allclose(coarse.decode(toks)[:, SPEC.gripper_dim], w[:, SPEC.gripper_dim])


def test_compact_rejects_bad_streams(coarse):
    toks, _, _ = coarse.encode_plan(synthetic_windows(10, n=1, seed=3)[0], PLAN)
    with pytest.raises(GrammarError):
        coarse.decode(toks[:-1])
    with pytest.raises(GrammarError):
        coarse.decode([coarse.vocab.len_token(8), *toks])
    with pytest.raises(ValueError):
        coarse.encode_plan(synthetic_windows(10, n=1, seed=3)[0], [(2, 1), (8, 3)])


def test_bpe_over_compact_roundtrip(coarse):
    train = synthetic_windows(10, n=300, seed=2)
    bpe = BPEDWTCodec.fit(coarse, train, [PLAN] * len(train))
    again = load_codec(bpe.to_dict())
    assert isinstance(again.base, CompactDWTCodec)
    assert again.base.quantizer.deadzone == 0.3
    for w in synthetic_windows(10, n=20, seed=8):
        base, recon, _ = coarse.encode_plan(w, PLAN)
        toks, _, _ = again.encode_plan(w, PLAN)
        assert again.expand(toks) == base and len(toks) < len(base)
        np.testing.assert_allclose(again.decode(toks), recon)
        fsm = again.fsm()
        for t in toks:
            assert t in fsm.allowed()
            fsm.step(t)
        assert fsm.done
