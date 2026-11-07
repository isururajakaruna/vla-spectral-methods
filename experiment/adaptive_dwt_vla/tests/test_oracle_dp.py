import numpy as np
import pytest
from conftest import synthetic_windows

from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec, enumerate_plans
from lerobot_policy_adaptive_dwt.oracle_dp import OracleWeights, plan_objective, solve


@pytest.mark.parametrize(
    "spec",
    [CodecSpec(horizon=8, lengths=(2, 4, 8)), CodecSpec(horizon=10, lengths=(2, 4, 8)), CodecSpec(horizon=6, lengths=(2, 4))],
    ids=["H8", "H10", "H6"],
)
@pytest.mark.parametrize(
    "w",
    [OracleWeights(gamma=0.0), OracleWeights(gamma=0.02), OracleWeights(gamma=1.0), OracleWeights(gamma=0.01, tau=0.05)],
    ids=["g0", "g002", "g1", "tau"],
)
def test_dp_matches_exhaustive(spec, w):
    codec = AdaptiveDWTCodec.fit(spec, synthetic_windows(spec.horizon, seed=4))
    plans = enumerate_plans(spec.horizon, spec.lengths)
    for win in synthetic_windows(spec.horizon, n=15, seed=8):
        plan, obj = solve(codec, win, w)
        brute = min(plan_objective(codec, win, p, w) for p in plans)
        assert obj <= brute + 1e-9
        assert abs(plan_objective(codec, win, plan, w) - obj) < 1e-9


def test_large_token_penalty_prefers_coarse(fitted):
    win = synthetic_windows(fitted.spec.horizon, n=1, seed=3)[0]
    cheap, _ = solve(fitted, win, OracleWeights(gamma=100.0))
    rich, _ = solve(fitted, win, OracleWeights(gamma=0.0))
    assert fitted.plan_token_count(win, cheap) < fitted.plan_token_count(win, rich)


def test_oracle_output_is_valid_stream_with_exact_gripper(fitted):
    for win in synthetic_windows(fitted.spec.horizon, n=30, seed=13):
        plan, _ = solve(fitted, win, OracleWeights())
        dec = fitted.decode(fitted.encode_plan(win, plan)[0])
        np.testing.assert_array_equal(fitted.gripper_bits(dec), fitted.gripper_bits(win))
