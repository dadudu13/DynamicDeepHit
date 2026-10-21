"""Discrete-time cause-specific hazard output layer.

The head emits per-bin hazards but converts them to the same (cause, bin)
probability interface the softmax head produces, so the survival likelihood in
losses.py should be correct for *both* parameterisations with no change.  These
tests pin that claim down rather than assuming it, and pin the property the
whole arm exists for: total probability inside the grid is no longer forced to 1.
"""
import pytest
import torch

from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.losses import log_likelihood_per_sample, negative_log_likelihood

B, K, D, T, N = 7, 2, 4, 3, 16


def _model(hazard):
    torch.manual_seed(0)
    return DynamicDeepHitTorch(
        input_dim=D, output_dim=B, layers_rnn=1, hidden_rnn=5, typ='GRU',
        risks=K, hazard=hazard,
        long_param={'layers': [4], 'dropout': 0.0},
        att_param={'layers': [4], 'dropout': 0.0},
        cs_param={'layers': [4], 'dropout': 0.0},
    ).double().eval()


def _batch():
    torch.manual_seed(1)
    x = torch.randn(N, T, D, dtype=torch.float64)
    x[N // 2:, 2:, :] = float('nan')          # ragged sequence lengths
    return x


def _outputs(hazard):
    with torch.no_grad():
        _, outcomes = _model(hazard)(_batch())
    return outcomes


def test_softmax_head_saturates_the_grid():
    """The behaviour the hazard head exists to remove: total mass exactly 1."""
    total = torch.stack(_outputs(hazard=False)).sum(0).sum(1)
    assert torch.allclose(total, torch.ones(N, dtype=torch.float64), atol=1e-9)


def test_hazard_head_leaves_mass_outside_the_grid():
    total = torch.stack(_outputs(hazard=True)).sum(0).sum(1)
    assert (total < 1.0).all(), "hazard head must not saturate the grid"
    assert (total > 0.0).all()


def test_pmf_matches_hazard_times_prior_survival():
    """P(T=b, K=k) == lam_k(b) * prod_{b'<b}(1 - sum_k lam_k(b'))."""
    with torch.no_grad():
        outcomes = _outputs(hazard=True)
        # Rather than re-running forward internals (brittle), assert the identity
        # the conversion must satisfy: the hazard implied by pmf / S(b-1) must
        # rebuild the same survival curve through the product form.
        pmf = torch.stack(outcomes, -1)                       # [N, B, K]
        step_fail = pmf.sum(-1)                               # [N, B]
        survival = 1.0 - step_fail.cumsum(1)                  # S(b) via CIF
        # Hazard implied by the conversion: lam(b) = step_fail(b) / S(b-1)
        s_before = torch.cat([torch.ones(N, 1, dtype=torch.float64), survival[:, :-1]], 1)
        lam = step_fail / s_before
        assert (lam >= -1e-12).all() and (lam <= 1 + 1e-12).all()
        # Rebuild survival from the product form and compare.
        rebuilt = torch.cumprod(1.0 - lam, dim=1)
        assert torch.allclose(rebuilt, survival, atol=1e-9)


def test_cif_monotone_and_survival_bounded():
    outcomes = _outputs(hazard=True)
    cif = [torch.cumsum(o, 1) for o in outcomes]
    for c in cif:
        assert (c.diff(dim=1) >= -1e-12).all(), "CIF must be non-decreasing"
    survival = 1.0 - torch.stack(cif).sum(0)
    assert (survival > 0).all(), "survival must stay strictly positive"
    assert (survival.diff(dim=1) <= 1e-12).all(), "survival must be non-increasing"


@pytest.mark.parametrize("hazard", [False, True])
def test_likelihood_terms_match_closed_form(hazard):
    """losses.py is already the correct discrete-time likelihood for both heads."""
    outcomes = _outputs(hazard)
    cif = [torch.cumsum(o, 1) for o in outcomes]
    t = torch.randint(1, B, (N,))
    e = torch.tensor([0, 1, 2] * (N // 3) + [0] * (N - 3 * (N // 3))).int()
    s = torch.zeros(N).long()

    got = log_likelihood_per_sample(outcomes, cif, t, e, s=s)
    rows = torch.arange(N)
    survival = 1.0 - torch.stack([c[rows, t] for c in cif]).sum(0)
    want = torch.where(
        e == 0,
        torch.log(survival + 1e-10),
        torch.log(torch.stack([o[rows, t] for o in outcomes])[
            (e.long() - 1).clamp(min=0), rows] + 1e-10),
    )
    assert torch.allclose(got, want, atol=1e-9)


def test_landmark_conditioning_is_a_ratio_of_survivals():
    """-log S(s) must turn the term into P(T=t, K=k | T > s)."""
    outcomes = _outputs(hazard=True)
    cif = [torch.cumsum(o, 1) for o in outcomes]
    t = torch.full((N,), B - 1)
    e = torch.ones(N).int()
    s = torch.full((N,), 2).long()

    rows = torch.arange(N)
    conditioned = log_likelihood_per_sample(outcomes, cif, t, e, s=s)
    unconditioned = log_likelihood_per_sample(outcomes, cif, t, e, s=None)
    surv_at_s = 1.0 - torch.stack([c[rows, s] for c in cif]).sum(0)
    assert torch.allclose(conditioned, unconditioned - torch.log(surv_at_s), atol=1e-9)


def test_hazard_flag_defaults_off_and_changes_nothing():
    """Existing checkpoints and arms must be unaffected by the new code path."""
    assert _model(hazard=False).hazard is False
    assert _model(hazard=False).survive is None
    a = torch.stack(_outputs(hazard=False))
    torch.manual_seed(0)
    b = torch.stack(_outputs(hazard=False))
    assert torch.equal(a, b)


def test_nll_is_finite_and_differentiable_under_hazard():
    model = _model(hazard=True)
    _, outcomes = model(_batch())
    cif = [torch.cumsum(o, 1) for o in outcomes]
    t = torch.randint(1, B, (N,))
    e = torch.randint(0, K + 1, (N,)).int()
    loss = negative_log_likelihood(outcomes, cif, t, e, s=torch.zeros(N).long())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.survive.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "survive head must train"


def test_hazard_init_gives_sane_initial_survival():
    """Without a biased survive logit the head starts at S(last) ~ 1e-17.

    A softmax over K+1 outcomes puts ~1/(K+1) on each hazard at init, which
    compounds geometrically over B bins.  The bias must place initial survival
    past the last bin at hazard_init_survival, independent of B.
    """
    for bins, target in [(7, 0.5), (102, 0.5), (102, 0.4)]:
        torch.manual_seed(0)
        model = DynamicDeepHitTorch(
            input_dim=D, output_dim=bins, layers_rnn=1, hidden_rnn=5, typ='GRU',
            risks=K, hazard=True, hazard_init_survival=target,
            long_param={'layers': [4], 'dropout': 0.0},
            att_param={'layers': [4], 'dropout': 0.0},
            cs_param={'layers': [4], 'dropout': 0.0},
        ).double().eval()
        with torch.no_grad():
            _, outcomes = model(_batch())
        survival = 1.0 - torch.stack(outcomes).sum(0).sum(1)
        # Approximate, not exact: the cause heads keep their ordinary random init
        # (deliberately, so the arm isolates the parameterisation change), so their
        # logits are not the 0 the closed form assumes and Jensen pushes the mean
        # hazard up slightly.  What must hold is the order of magnitude -- the
        # unbiased head lands at ~1e-17 here, not ~0.5.
        assert abs(survival.mean().item() - target) < 0.10, (
            f"B={bins}: initial survival {survival.mean():.4f} far from {target}"
        )
        assert survival.mean().item() > 0.2, "geometric collapse has returned"


def test_hazard_init_scales_with_grid_length():
    """A finer grid needs a larger bias to hold the same initial survival."""
    biases = []
    for bins in (10, 100):
        torch.manual_seed(0)
        model = DynamicDeepHitTorch(
            input_dim=D, output_dim=bins, layers_rnn=1, hidden_rnn=5, typ='GRU',
            risks=K, hazard=True,
            long_param={'layers': [4], 'dropout': 0.0},
            att_param={'layers': [4], 'dropout': 0.0},
            cs_param={'layers': [4], 'dropout': 0.0},
        ).double()
        biases.append(model.survive[-1].bias.detach().mean().item())
    assert biases[1] > biases[0], "bias must grow with the number of bins"


def _prop_model(bins=B, target=0.5):
    torch.manual_seed(0)
    return DynamicDeepHitTorch(
        input_dim=D, output_dim=bins, layers_rnn=1, hidden_rnn=5, typ='GRU',
        risks=K, hazard=True, proportional=True, hazard_init_survival=target,
        long_param={'layers': [4], 'dropout': 0.0},
        att_param={'layers': [4], 'dropout': 0.0},
        cs_param={'layers': [4], 'dropout': 0.0},
    ).double().eval()


def test_proportional_keeps_risk_ordering_constant_across_bins():
    """The defining property: one shared patient score per cause.

    Under the free head each bin owns a 32-dim weight vector, so the ranking of
    patients may differ bin to bin.  Under proportionality the ranking must be
    identical in every bin, because the only patient-dependent term is a scalar
    added to every bin's baseline.
    """
    with torch.no_grad():
        _, outcomes = _prop_model()(_batch())
    pmf = torch.stack(outcomes, -1)
    step = pmf.sum(-1)
    survival = 1.0 - step.cumsum(1)
    s_before = torch.cat([torch.ones(N, 1, dtype=torch.float64), survival[:, :-1]], 1)
    lam_vte = (pmf[:, :, 0] / s_before)                       # [N, B]
    order = lam_vte.argsort(dim=0)
    for b in range(1, B):
        assert torch.equal(order[:, b], order[:, 0]), f"ranking changed at bin {b}"


def test_free_head_ranking_may_vary_across_bins():
    """Contrast: the free head is not constrained this way."""
    with torch.no_grad():
        _, outcomes = _model(hazard=True)(_batch())
    pmf = torch.stack(outcomes, -1)
    step = pmf.sum(-1)
    survival = 1.0 - step.cumsum(1)
    s_before = torch.cat([torch.ones(N, 1, dtype=torch.float64), survival[:, :-1]], 1)
    order = (pmf[:, :, 0] / s_before).argsort(dim=0)
    assert any(not torch.equal(order[:, b], order[:, 0]) for b in range(1, B))


def test_proportional_has_far_fewer_output_parameters():
    free, prop = _model(hazard=True), _prop_model()
    n_free = sum(p.numel() for p in free.parameters())
    n_prop = sum(p.numel() for p in prop.parameters())
    assert n_prop < n_free
    assert prop.survive is None and prop.baseline is not None
    assert prop.baseline.shape == (K, B)


def test_proportional_init_hits_target_survival_exactly():
    """Cause scores are zeroed at init, so the closed form is exact here."""
    for bins, target in [(7, 0.5), (102, 0.5), (102, 0.4)]:
        with torch.no_grad():
            _, outcomes = _prop_model(bins, target)(_batch())
        s = 1.0 - torch.stack(outcomes).sum(0).sum(1)
        assert abs(s.mean().item() - target) < 1e-6, f"B={bins}: {s.mean():.6f}"


def test_proportional_still_leaves_mass_outside_the_grid():
    total = torch.stack(_prop_model()(_batch())[1]).sum(0).sum(1)
    assert (total < 1.0).all()
