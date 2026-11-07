import numpy as np
import pytest

from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec

SPECS = {
    10: CodecSpec(horizon=10, lengths=(2, 4, 8)),
    32: CodecSpec(horizon=32, lengths=(4, 8, 16, 32)),
}


def synthetic_windows(horizon: int, n: int = 400, seed: int = 0) -> np.ndarray:
    """Smooth random-walk actions with occasional gripper flips (normalized units)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, horizon)
    out = np.zeros((n, horizon, 7))
    for i in range(n):
        for d in range(6):
            f = rng.uniform(0.2, 3.0)
            out[i, :, d] = rng.normal(0, 0.8) + 0.5 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + 0.05 * rng.normal(size=horizon)
        g = np.full(horizon, -1.0 if rng.random() < 0.5 else 1.0)
        if rng.random() < 0.3:
            g[rng.integers(1, horizon) :] *= -1
        out[i, :, 6] = g
    return out


@pytest.fixture(params=[10, 32], ids=["H10", "H32"])
def fitted(request):
    spec = SPECS[request.param]
    train = synthetic_windows(spec.horizon, seed=1)
    return AdaptiveDWTCodec.fit(spec, train)
