import numpy as np
import pywt
import pytest
from conftest import SPECS, synthetic_windows

from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, enumerate_plans
from lerobot_policy_adaptive_dwt.codec_basis import haar_matrix
from lerobot_policy_adaptive_dwt.metrics import gripper_metrics, reconstruction_metrics


@pytest.mark.parametrize("length", [2, 4, 8, 16, 32])
def test_basis_matches_pywt(length):
    rng = np.random.default_rng(length)
    x = rng.normal(size=length)
    level = length.bit_length() - 1
    ref = np.concatenate(pywt.wavedec(x, "haar", mode="periodization", level=level))
    np.testing.assert_allclose(haar_matrix(length) @ x, ref, atol=1e-12)
    np.testing.assert_allclose(haar_matrix(length).T @ ref, x, atol=1e-12)
    coeffs = pywt.wavedec(x, "haar", mode="periodization", level=level)
    np.testing.assert_allclose(pywt.waverec(coeffs, "haar", mode="periodization"), x, atol=1e-12)


@pytest.mark.parametrize("horizon", [10, 32])
def test_lossless_full_resolution_every_length(horizon):
    spec = SPECS[horizon]
    codec = AdaptiveDWTCodec(spec, quantizer=None)
    w = synthetic_windows(horizon, n=5, seed=3)
    for length in spec.lengths:
        plan = [(length, length.bit_length() - 1)] * (horizon // length)
        rem = horizon - sum(L for L, _ in plan)
        if rem:
            small = min(spec.lengths)
            plan += [(small, small.bit_length() - 1)] * (rem // small)
        for win in w:
            np.testing.assert_allclose(codec.reconstruct_lossless(win, plan), np.where(
                np.arange(7) == 6, np.where(win[:, 6:7] > 0, 1.0, -1.0), win), atol=1e-10)


def test_roundtrip_shape_and_exact_tokens(fitted):
    spec = fitted.spec
    windows = synthetic_windows(spec.horizon, n=50, seed=7)
    plans = enumerate_plans(spec.horizon, spec.lengths)
    rng = np.random.default_rng(0)
    for win in windows:
        plan = plans[rng.integers(len(plans))]
        tokens, recon, _ = fitted.encode_plan(win, plan)
        assert len(tokens) == fitted.plan_token_count(win, plan)
        assert len(tokens) <= fitted.max_stream_tokens
        dec = fitted.decode(tokens)
        assert dec.shape == (spec.horizon, 7)
        np.testing.assert_allclose(dec, recon, atol=1e-12)
        assert np.isfinite(dec).all()


def test_every_length_and_resolution_decodes(fitted):
    spec = fitted.spec
    win = synthetic_windows(spec.horizon, n=1, seed=11)[0]
    for length in spec.lengths:
        for k in range(length.bit_length()):
            rest = spec.horizon - length
            fill = [p for p in enumerate_plans(rest, spec.lengths, fixed_k=0)][:1] if rest else [[]]
            if not fill:
                continue
            plan = [(length, k)] + fill[0]
            tokens, recon, _ = fitted.encode_plan(win, plan)
            np.testing.assert_allclose(fitted.decode(tokens), recon, atol=1e-12)


def test_quantized_error_is_measured_not_asserted(fitted):
    spec = fitted.spec
    w = synthetic_windows(spec.horizon, n=100, seed=5)
    full = [(max(spec.lengths), max(spec.lengths).bit_length() - 1)] * (spec.horizon // max(spec.lengths))
    if sum(L for L, _ in full) != spec.horizon:
        full = [(2, 1)] * (spec.horizon // 2)
    recon = np.stack([fitted.encode_plan(x, full)[1] for x in w])
    m = reconstruction_metrics(w, recon, spec.cont_dims)
    assert m["mse"] < 1e-3  # 256-bin full-resolution quantization is fine-grained on this data


def test_zero_and_constant_motion(fitted):
    spec = fitted.spec
    for value in (0.0, 0.37):
        win = np.full((spec.horizon, 7), value)
        win[:, 6] = -1.0
        plan = enumerate_plans(spec.horizon, spec.lengths, fixed_k=0)[0]
        tokens, recon, _ = fitted.encode_plan(win, plan)
        dec = fitted.decode(tokens)
        # coarsest resolution of a constant signal is exact up to cA quantization
        step = max(fitted.quantizer.params[L][1][0].max() for L, _ in plan)
        assert np.abs(dec[:, :6] - value).max() <= step
        if value == 0.0:
            # detail groups are centred at zero, so a resolution increase adds nothing
            plan_hi = [(L, L.bit_length() - 1) for L, _ in plan]
            dec_hi = fitted.decode(fitted.encode_plan(win, plan_hi)[0])
            np.testing.assert_allclose(dec_hi, dec, atol=1e-12)


def test_high_frequency_motion_needs_resolution(fitted):
    spec = fitted.spec
    t = np.arange(spec.horizon)
    win = np.zeros((spec.horizon, 7))
    win[:, :6] = 0.5 * ((-1) ** t)[:, None]
    win[:, 6] = 1.0
    lo = enumerate_plans(spec.horizon, spec.lengths, fixed_k=0)[0]
    hi = [(L, L.bit_length() - 1) for L, _ in lo]
    err_lo = np.abs(fitted.decode(fitted.encode_plan(win, lo)[0])[:, :6] - win[:, :6]).max()
    err_hi = np.abs(fitted.decode(fitted.encode_plan(win, hi)[0])[:, :6] - win[:, :6]).max()
    assert err_hi < err_lo


def test_abrupt_gripper_changes_exact(fitted):
    spec = fitted.spec
    rng = np.random.default_rng(2)
    plans = enumerate_plans(spec.horizon, spec.lengths, fixed_k=0)
    true_bits, rec_bits = [], []
    for _ in range(200):
        win = synthetic_windows(spec.horizon, n=1, seed=int(rng.integers(1e6)))[0]
        win[:, 6] = np.where(rng.random(spec.horizon) < 0.5, -1.0, 1.0)  # flip almost every step
        plan = plans[rng.integers(len(plans))]
        dec = fitted.decode(fitted.encode_plan(win, plan)[0])
        true_bits.append(fitted.gripper_bits(win))
        rec_bits.append(fitted.gripper_bits(dec))
    g = gripper_metrics(np.array(true_bits), np.array(rec_bits))
    assert g["grip_exact_rate"] == 1.0 and g["grip_precision"] == 1.0 and g["grip_recall"] == 1.0


def test_out_of_range_coefficients_saturate(fitted):
    spec = fitted.spec
    win = np.full((spec.horizon, 7), 50.0)
    win[:, 6] = 1.0
    plan = enumerate_plans(spec.horizon, spec.lengths, fixed_k=0)[0]
    tokens, recon, info = fitted.encode_plan(win, plan)
    assert info["clipped"] > 0
    assert all(0 <= t < fitted.vocab.size for t in tokens)
    assert np.isfinite(fitted.decode(tokens)).all()


def test_wrong_window_shape_rejected(fitted):
    spec = fitted.spec
    with pytest.raises(ValueError):
        fitted.encode_plan(np.zeros((spec.horizon - 1, 7)), [(2, 0)])
    with pytest.raises(ValueError):
        fitted.encode_plan(np.zeros((spec.horizon, 7)), [(2, 0)])


def test_serialization_roundtrip(fitted):
    clone = AdaptiveDWTCodec.from_dict(fitted.to_dict())
    win = synthetic_windows(fitted.spec.horizon, n=1, seed=9)[0]
    plan = enumerate_plans(fitted.spec.horizon, fitted.spec.lengths)[3]
    t1, r1, _ = fitted.encode_plan(win, plan)
    t2, r2, _ = clone.encode_plan(win, plan)
    assert t1 == t2
    np.testing.assert_array_equal(r1, r2)


def test_calibration_uses_only_given_training_windows():
    spec = SPECS[10]
    a = AdaptiveDWTCodec.fit(spec, synthetic_windows(10, seed=1))
    b = AdaptiveDWTCodec.fit(spec, synthetic_windows(10, seed=1))
    c = AdaptiveDWTCodec.fit(spec, synthetic_windows(10, seed=2))
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != c.to_dict()
