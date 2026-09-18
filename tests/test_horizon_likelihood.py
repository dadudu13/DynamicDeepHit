"""The horizon-specific likelihood term must be (a) inert at gamma=0, (b) scored on
exactly the labels and weights V3's evaluation uses, and (c) minimised at the truth."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddh.losses import (
    build_landmark_prefixes,
    censoring_survival_by_bin,
    horizon_likelihood_loss,
    landmark_nll_loss,
    landmark_row_days,
)


class Toy(torch.nn.Module):
    """Returns a fixed per-bin hazard mass, so the loss can be checked by hand."""

    def __init__(self, mass, n):
        super().__init__()
        self.p = torch.nn.Parameter(torch.as_tensor(mass, dtype=torch.double))
        self.n = n

    def forward(self, x):
        r = x.shape[0]
        out = [self.p[k].unsqueeze(0).expand(r, -1) for k in range(self.p.shape[0])]
        return torch.zeros(r, x.shape[1], x.shape[2], dtype=torch.double), out


def _toy(n_bins=8, n_pat=6, n_vis=3, n_feat=2):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n_pat, n_vis, n_feat, generator=g, dtype=torch.double)
    x[-1, 2:] = float("nan")
    s_all = torch.arange(n_vis).repeat(n_pat, 1).long()
    visit_days = torch.arange(n_vis, dtype=torch.double).repeat(n_pat, 1) * 100.0
    t = torch.tensor([5, 6, 7, 4, 7, 6]).long()
    e = torch.tensor([1, 0, 2, 1, 0, 1]).int()
    return x, s_all, visit_days, t, e, n_bins


def test_bucketize_matches_digitize():
    """ddh_api bins landmark days with np.digitize(right=True); the torch
    equivalent is bucketize(right=False), NOT bucketize(right=True)."""
    edges = np.array([0.0, 30.0, 90.0, 180.0, 365.0, 1000.0])
    v = np.array([0.0, 15.0, 30.0, 31.0, 90.0, 400.0, 999.0])
    want = np.digitize(v, edges, right=True) - 1
    got = (torch.bucketize(torch.tensor(v), torch.tensor(edges), right=False) - 1).numpy()
    assert (want == got).all(), f"{want} != {got}"


def test_gamma_zero_is_bit_identical():
    x, s_all, visit_days, t, e, n_bins = _toy()
    mass = torch.full((2, n_bins), 0.04, dtype=torch.double)
    m = Toy(mass, n_bins)
    x_exp, s_exp, pidx, w = build_landmark_prefixes(x, s_all, visit_days=visit_days,
                                                    decay_days=180.0)
    te, ee = t[pidx], e[pidx]
    spec = dict(day=landmark_row_days(x_exp, pidx, visit_days),
                split_time=torch.linspace(0, 800, n_bins + 1).double(),
                horizons=[90.0, 180.0],
                g_bins=censoring_survival_by_bin(t, e, n_bins))
    base = landmark_nll_loss(m, x_exp, te, ee, s_exp, w)
    same = landmark_nll_loss(m, x_exp, te, ee, s_exp, w, horizon_spec=spec, gamma=0.0)
    assert base.item() == same.item()
    # and the weighted form reduces to beta * base
    scaled = landmark_nll_loss(m, x_exp, te, ee, s_exp, w, nll_weight=0.75)
    assert abs(scaled.item() - 0.75 * base.item()) < 1e-12


def test_gamma_nonzero_adds_exactly_gamma_times_horizon():
    x, s_all, visit_days, t, e, n_bins = _toy()
    m = Toy(torch.full((2, n_bins), 0.04, dtype=torch.double), n_bins)
    x_exp, s_exp, pidx, w = build_landmark_prefixes(x, s_all, visit_days=visit_days)
    te, ee = t[pidx], e[pidx]
    # Bins of 25 days, so a 90-day horizon clears the landmark's own bin.
    spec = dict(day=landmark_row_days(x_exp, pidx, visit_days),
                split_time=torch.linspace(0, 200, n_bins + 1).double(),
                horizons=[90.0], g_bins=censoring_survival_by_bin(t, e, n_bins))
    parts = {}
    tot = landmark_nll_loss(m, x_exp, te, ee, s_exp, w, nll_weight=0.75,
                            horizon_spec=spec, gamma=3.0, components=parts)
    assert abs(tot.item() - (0.75 * parts["nll"] + 3.0 * parts["horizon"])) < 1e-10
    assert parts["horizon"] > 0


def test_horizon_inside_the_landmark_bin_contributes_nothing():
    """When the time grid is coarser than the horizon, day_s + h lands in the
    landmark's own bin: the conditional risk is identically 0 there and the row
    carries no gradient.  Harmless, but it means the term's reach at late
    landmarks depends on how wide the bins are out there -- worth measuring on
    real data, not assuming."""
    x, s_all, visit_days, t, e, n_bins = _toy()
    m = Toy(torch.full((2, n_bins), 0.04, dtype=torch.double), n_bins)
    x_exp, s_exp, pidx, w = build_landmark_prefixes(x, s_all, visit_days=visit_days)
    spec = dict(day=landmark_row_days(x_exp, pidx, visit_days),
                split_time=torch.linspace(0, 800, n_bins + 1).double(),  # 100-day bins
                horizons=[90.0], g_bins=censoring_survival_by_bin(t, e, n_bins))
    parts = {}
    landmark_nll_loss(m, x_exp, t[pidx], e[pidx], s_exp, w,
                      horizon_spec=spec, gamma=1.0, components=parts)
    assert parts["horizon"] == 0.0


def test_labels_and_weights_match_v3_evaluation():
    """Graf/Gerds-Schumacher, as V3's parts(): cause-2 deaths before h are true
    zeros carrying 1/G(T-), survivors past h carry 1/G(h), censored-before-h
    carry nothing."""
    n_bins = 10
    # one patient, one visit, landmark day 0 -> horizon bin is known exactly
    x = torch.ones(4, 1, 2, dtype=torch.double)
    s_all = torch.zeros(4, 1).long()
    visit_days = torch.zeros(4, 1, dtype=torch.double)
    t = torch.tensor([3, 3, 9, 3]).long()          # bins
    e = torch.tensor([1, 2, 0, 0]).int()           # VTE, death, censored late, censored early
    split_time = torch.linspace(0, 1000, n_bins + 1).double()   # bin width 100
    # horizon 500 days -> bucketize(500) - 1 = bin 4
    mass = torch.full((2, n_bins), 0.02, dtype=torch.double)
    m = Toy(mass, n_bins)
    _, outcomes = m(x)
    cif = [torch.cumsum(o, 1) for o in outcomes]
    g = censoring_survival_by_bin(t, e, n_bins)
    w = torch.ones(4, dtype=torch.double)
    val = horizon_likelihood_loss(outcomes, cif, t, e, torch.zeros(4).long(), w,
                                  day=torch.zeros(4, dtype=torch.double),
                                  split_time=split_time, horizons=[500.0], g_bins=g)
    # recompute by hand
    b_h = int(torch.bucketize(torch.tensor([500.0]), split_time, right=False).item()) - 1
    p = float((cif[0][0, b_h] - cif[0][0, 0]) / (1 - cif[0][0, 0] - cif[1][0, 0]))
    want = 0.0
    want += -np.log(p) * float(1 / g[2])            # VTE at bin 3 <= 4, y=1, 1/G(T-)
    want += -np.log(1 - p) * float(1 / g[2])        # death at bin 3, y=0, 1/G(T-)
    want += -np.log(1 - p) * float(1 / g[b_h])      # alive past h, y=0, 1/G(h)
    want += 0.0                                     # censored at bin 3 < h: dropped
    assert abs(val.item() - want / 2) < 1e-9, (val.item(), want / 2)


def test_minimised_at_the_true_rate():
    """A proper scoring rule: sweeping the model's conditional risk, the term
    bottoms out at the observed event fraction, which the bin NLL does not do."""
    n_bins, n = 12, 400
    rate = 0.25
    rng = np.random.default_rng(3)
    y = (rng.random(n) < rate)
    t = torch.where(torch.as_tensor(y), torch.tensor(2), torch.tensor(11)).long()
    e = torch.where(torch.as_tensor(y), torch.tensor(1), torch.tensor(0)).int()
    split_time = torch.linspace(0, 1200, n_bins + 1).double()
    g = torch.ones(n_bins, dtype=torch.double)     # no censoring before the horizon
    best, arg = 1e18, None
    for q in np.arange(0.05, 0.96, 0.01):
        # a CIF that reaches q by the horizon bin and stays flat
        c1 = torch.zeros(n, n_bins, dtype=torch.double)
        c1[:, 1:] = q
        c2 = torch.zeros(n, n_bins, dtype=torch.double)
        o1 = torch.diff(c1, dim=1, prepend=torch.zeros(n, 1, dtype=torch.double))
        v = horizon_likelihood_loss([o1, c2], [c1, c2], t, e,
                                    torch.zeros(n).long(), torch.ones(n, dtype=torch.double),
                                    day=torch.zeros(n, dtype=torch.double),
                                    split_time=split_time, horizons=[500.0], g_bins=g)
        if v.item() < best:
            best, arg = v.item(), q
    assert abs(arg - rate) < 0.02, f"minimised at {arg}, true rate {rate}"


def test_gradient_flows_and_is_finite():
    x, s_all, visit_days, t, e, n_bins = _toy()
    m = Toy(torch.full((2, n_bins), 0.04, dtype=torch.double), n_bins)
    x_exp, s_exp, pidx, w = build_landmark_prefixes(x, s_all, visit_days=visit_days)
    spec = dict(day=landmark_row_days(x_exp, pidx, visit_days),
                split_time=torch.linspace(0, 800, n_bins + 1).double(),
                horizons=[90.0, 180.0], g_bins=censoring_survival_by_bin(t, e, n_bins))
    loss = landmark_nll_loss(m, x_exp, t[pidx], e[pidx], s_exp, w,
                             nll_weight=0.75, horizon_spec=spec, gamma=1.0)
    loss.backward()
    assert m.p.grad is not None and torch.isfinite(m.p.grad).all()
    assert m.p.grad.abs().sum() > 0
