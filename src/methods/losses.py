import os
import sys
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------
# The loss of SimCLR: NT-Xent (normalized temperature-scaled cross entropy)
# ------------------------------------------------------------------------
def nt_xent_loss(z1, z2, temperature=0.2):
    """
    Contrastive loss over a batch of N images (2N views).
        For each view, its "positive" is the other view of the same image.
        The other 2N - 2 views in the batch are "negatives".
        The loss is a classification problem: "which of the other 2N-1 views
        is my partner?", solved with softmax + cross-entropy.

        Note: simclr paper use temperature 0.5 for the base config and 0.1 for the longer training
    """
    if z1.shape != z2.shape:
        raise ValueError(
            f"z1 and z2 must have the same shape, "
            f"received {tuple(z1.shape)} and {tuple(z2.shape)}"
        )

    if temperature <=0:
        raise ValueError(f"Temperature valude need to be greater than 0, current value is: {temperature}")

    batch_size = z1.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent loss requires batch_size >= 2")

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1, eps=1e-8)

    similarity = torch.matmul(z, z.T) / temperature

    # A view shouldn't be compared with iteself, so mask the diagonal out.
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    neg = torch.finfo(similarity.dtype).min
    similarity = similarity.masked_fill(self_mask, neg)

    # Row i (from the first half) 
    labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)
    loss = F.cross_entropy(similarity, labels)
    
    return loss


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
def _unit_noise(n, d, gen):
    """Random directions with unit L2 norm per row, so a scale `s` means
    'noise of norm s' regardless of the dimension d."""
    return F.normalize(torch.randn(n, d, generator=gen), dim=1)


def _make_views(base, sigma, gen):
    """Two augmented views: base + noise of norm sigma, then re-normalized."""
    n, d = base.shape
    z1 = F.normalize(base + sigma * _unit_noise(n, d, gen), dim=1)
    z2 = F.normalize(base + sigma * _unit_noise(n, d, gen), dim=1)
    return z1, z2


def _log_rand(n):
    """Expected loss when all similarities are ~0: log(2N - 1)."""
    return math.log(2 * n - 1)


def _log_min(n, t):
    """Loss when pos sim = 1 and all neg sims = 0: log(1 + (2N-2) e^{-1/T})."""
    return math.log(1 + (2 * n - 2) * math.exp(-1 / t))


# ------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------
def test_input_validation():
    z = torch.randn(4, 8)
    for bad_call in (
        lambda: nt_xent_loss(z, torch.randn(5, 8)),       # shape mismatch
        lambda: nt_xent_loss(z, z, temperature=0.0),      # non-positive temperature
        lambda: nt_xent_loss(z[:1], z[:1]),               # batch too small
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_nt_xent_realistic():
    N, D, T = 64, 128, 0.5
    g = torch.Generator().manual_seed(0)
    base = F.normalize(torch.randn(N, D, generator=g), dim=1)

    loss_high_noise = nt_xent_loss(*_make_views(base, 1.0, g), temperature=T).item()
    loss_low_noise = nt_xent_loss(*_make_views(base, 0.1, g), temperature=T).item()
    print(f"low noise: {loss_low_noise:.3f}, high noise: {loss_high_noise:.3f}")

    # Less noise -> views more similar -> lower loss
    assert loss_low_noise < loss_high_noise
    # Sanity range: [0, log(2N-1) + margin]
    for loss in (loss_low_noise, loss_high_noise):
        assert 0.0 < loss < _log_rand(N) + 0.6, loss


def test_smoke_realistic():
    N, D, T = 128, 128, 0.2
    g = torch.Generator().manual_seed(0)
    z1 = torch.randn(N, D, generator=g, requires_grad=True)
    z2 = torch.randn(N, D, generator=g, requires_grad=True)

    loss = nt_xent_loss(z1, z2, T)

    # For random embeddings, NT-Xent ≈ log(2N - 1) (slightly above, since
    # random cosine similarities / T have nonzero variance).
    upper = _log_rand(N)
    assert torch.isfinite(loss)
    assert abs(loss.item() - upper) < 0.6, f"loss {loss.item():.3f} vs log(2N-1)={upper:.3f}"

    loss.backward()
    for name, z in (("z1", z1), ("z2", z2)):
        assert z.grad is not None and torch.isfinite(z.grad).all(), name
        assert z.grad.abs().sum() > 0, f"{name} got no gradient"


def test_analytical_endpoints():
    T = 0.2
    for N in (4, 16):
        # (a) Perfect case: pos sim = 1, neg sim = 0 (exact closed form)
        z = torch.eye(N)                        # rows orthogonal, unit norm
        loss = nt_xent_loss(z, z, T).item()
        expected = _log_min(N, T)
        assert abs(loss - expected) < 1e-4, (loss, expected, N)

        # (b) Random case: all sims ≈ 0 -> loss ≈ log(2N-1)
        g = torch.Generator().manual_seed(1)
        D = 256                                 # big D -> near-orthogonal
        z1 = F.normalize(torch.randn(N, D, generator=g), dim=1)
        z2 = F.normalize(torch.randn(N, D, generator=g), dim=1)
        loss = nt_xent_loss(z1, z2, T).item()
        assert abs(loss - _log_rand(N)) < 0.5, (loss, _log_rand(N), N)


def test_monotonicity_in_augmentation():
    N, D, T = 64, 128, 0.2
    g = torch.Generator().manual_seed(2)
    base = F.normalize(torch.randn(N, D, generator=g), dim=1)

    # sigma is the L2 norm of the noise relative to the unit-norm signal
    sigmas = [0.05, 0.1, 0.2, 0.5, 1.0, 3.0]
    losses = [nt_xent_loss(*_make_views(base, s, g), T).item() for s in sigmas]

    print("sigma -> loss:", {s: f"{l:.3f}" for s, l in zip(sigmas, losses)})
    for a, b, sa, sb in zip(losses, losses[1:], sigmas, sigmas[1:]):
        assert b > a, f"loss should increase with noise: sigma={sa}:{a:.3f} -> sigma={sb}:{b:.3f}"

    # Nearly clean views should approach the analytical minimum ...
    assert abs(losses[0] - _log_min(N, T)) < 0.3, (losses[0], _log_min(N, T))
    # ... and very noisy views should approach log(2N-1) ≈ 4.84
    assert losses[-1] > 4.0, f"very noisy views should approach {_log_rand(N):.2f}, got {losses[-1]:.3f}"
    assert losses[-1] < _log_rand(N) + 0.6


def _perturb(x, eps, gen):
    """Add noise of L2 norm eps to each row."""
    return x + eps * _unit_noise(*x.shape, gen)


def test_simulated_training_curve():
    N, D, T = 64, 128, 0.2
    noise = 0.1
    g = torch.Generator().manual_seed(3)

    identity = F.normalize(torch.randn(N, D, generator=g), dim=1)   # target
    alphas = torch.linspace(0, 1, 11).tolist()
    losses = []
    for a in alphas:
        # At a=0: two independent random views (no shared signal).
        # At a=1: both views ≈ identity_i plus small noise.
        # All random terms are unit-norm so they are on the same scale as identity.
        r1, r2 = _unit_noise(N, D, g), _unit_noise(N, D, g)
        e1, e2 = _unit_noise(N, D, g), _unit_noise(N, D, g)
        z1 = F.normalize((1 - a) * r1 + a * identity + noise * e1, dim=1)
        z2 = F.normalize((1 - a) * r2 + a * identity + noise * e2, dim=1)
        z1 = F.normalize(_perturb(z1, eps=0.05, gen=g), dim=1)
        z2 = F.normalize(_perturb(z2, eps=0.05, gen=g), dim=1)
        losses.append(nt_xent_loss(z1, z2, T).item())

    print("alpha -> loss:", [f"{l:.3f}" for l in losses])
    log_rand, log_min = _log_rand(N), _log_min(N, T)

    # Endpoints land in the right place
    assert abs(losses[0] - log_rand) < 0.6, (losses[0], log_rand)
    assert abs(losses[-1] - log_min) < 0.3, (losses[-1], log_min)
    # Decreasing, allowing small noise-induced wiggles
    for prev, cur in zip(losses, losses[1:]):
        assert cur < prev + 0.2, losses
    assert losses[0] > losses[len(losses) // 2] > losses[-1]
    assert losses[0] - losses[-1] > 3.0


if __name__ == "__main__":
    test_input_validation()
    test_nt_xent_realistic()
    test_smoke_realistic()
    test_analytical_endpoints()
    test_monotonicity_in_augmentation()
    test_simulated_training_curve()
    print("All tests passed.")
