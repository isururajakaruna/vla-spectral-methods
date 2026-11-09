import numpy as np
import torch
import torch.nn.functional as F

from lerobot_policy_adaptive_dwt.codec import AdaptiveDWTCodec, CodecSpec
from lerobot_policy_adaptive_dwt.codec_basis import group_of
from lerobot_policy_adaptive_dwt.modeling_adaptive_dwt import action_token_loss, coef_targets
from lerobot_policy_adaptive_dwt.processor_adaptive_dwt import PALI_VOCAB, local_to_pali

PLAN = [(8, 3), (2, 1)]


def _codec_and_targets(n=3, seed=0):
    rng = np.random.default_rng(seed)
    spec = CodecSpec(horizon=10, lengths=(2, 4, 8), gripper_low=-1.0, gripper_high=1.0)
    train = np.cumsum(rng.normal(0, 0.1, (64, 10, 7)), axis=1)
    train[..., 6] = np.where(rng.random((64, 10)) > 0.5, 1.0, -1.0)
    codec = AdaptiveDWTCodec.fit(spec, train)
    streams = [codec.encode_plan(train[i], PLAN)[0] for i in range(n)]
    width = max(map(len, streams)) + 2
    pad = int(local_to_pali(0)) + 1  # any id outside the action vocabulary
    targets = np.full((n, width), pad, dtype=np.int64)
    mask = np.zeros((n, width), dtype=bool)
    for i, s in enumerate(streams):
        targets[i, : len(s)] = local_to_pali(np.asarray(s))
        mask[i, : len(s)] = True
    return codec, streams, torch.from_numpy(targets), torch.from_numpy(mask)


def test_coef_targets_bins_and_steps():
    codec, streams, targets, _ = _codec_and_targets()
    bins, steps = coef_targets(targets, codec)
    n_cont = codec.spec.n_cont
    for i, s in enumerate(streams):
        segs = codec.segments(s)
        expect_bins, expect_steps = [], []
        for seg in segs:
            g = group_of(seg.length)
            for k, b in enumerate(seg.coef_bins):
                c, d = divmod(k, n_cont)
                expect_bins.append(b)
                expect_steps.append(codec.quantizer.params[seg.length][1][g[c], d])
        got = bins[i][bins[i] >= 0].tolist()
        assert got == expect_bins
        np.testing.assert_allclose(steps[i][bins[i] >= 0].numpy(), expect_steps, rtol=1e-6)
        assert (steps[i][bins[i] < 0] == 0).all()


def test_plain_and_weighted_losses():
    codec, _, targets, mask = _codec_and_targets()
    torch.manual_seed(0)
    logits = torch.randn(*targets.shape, PALI_VOCAB)
    plain, per_tok = action_token_loss(logits, targets, mask)
    ref = F.cross_entropy(logits.reshape(-1, PALI_VOCAB), targets.reshape(-1), reduction="none").reshape(targets.shape)
    torch.testing.assert_close(per_tok, ref)
    torch.testing.assert_close(plain, (ref * mask).sum() / mask.sum())

    weighted, per_tok_w = action_token_loss(logits, targets, mask, codec, coef_weight=True)
    torch.testing.assert_close(per_tok_w, ref)  # logged loss stays the plain cross-entropy
    bins, steps = coef_targets(targets, codec)
    is_coef = (bins >= 0) & mask
    w = torch.ones_like(ref)
    w[is_coef] = steps[is_coef] / steps[is_coef].mean()
    torch.testing.assert_close(weighted, (ref * w * mask).sum() / mask.sum())
    assert torch.isclose(w[is_coef].mean(), torch.tensor(1.0))


def test_soft_labels_reduce_to_one_hot_for_tiny_sigma_and_credit_neighbours():
    codec, _, targets, mask = _codec_and_targets()
    torch.manual_seed(1)
    logits = torch.randn(*targets.shape, PALI_VOCAB)
    plain, _ = action_token_loss(logits, targets, mask)
    tiny, _ = action_token_loss(logits, targets, mask, codec, soft_sigma=1e-9)
    torch.testing.assert_close(tiny, plain)

    # put all mass one bin away from the target on the first coefficient position: soft labels with sigma = step
    # must score it better than one-hot cross-entropy does
    bins, steps = coef_targets(targets, codec)
    b, t = map(int, torch.nonzero((bins >= 0) & mask)[0])
    near = int(local_to_pali(min(int(bins[b, t]) + 1, codec.vocab.n_bins - 1)))
    logits[b, t] = -20.0
    logits[b, t, near] = 20.0
    one_pos = torch.zeros_like(mask)
    one_pos[b, t] = True
    hard, _ = action_token_loss(logits, targets, one_pos)
    soft, _ = action_token_loss(logits, targets, one_pos, codec, soft_sigma=float(steps[b, t]))
    assert soft < hard
