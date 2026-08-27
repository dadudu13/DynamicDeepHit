from __future__ import annotations

import numpy as np
import torch

from ddh.losses import (
    RankRows,
    build_landmark_prefixes,
    conditional_cif,
    ranking_loss,
    total_loss,
    weighted_ranking_loss,
)
from ddh.utils import all_landmark_objective

from test_landmark_nll import _make_batch, _make_model

SIGMA = 0.5


def _rows_from(model, x, t, e, s, patient=None, weight=None):
    """RankRows for one row per patient, conditioned exactly as total_loss does."""
    _, outcomes = model(x)
    cif = [torch.cumsum(ok, 1) for ok in outcomes]
    n = x.shape[0]
    return RankRows(
        cif=conditional_cif(cif, s),
        t=t.long(),
        s=s,
        e=e.int(),
        weight=torch.ones(n).double() if weight is None else weight,
        patient=torch.arange(n) if patient is None else patient,
    ), cif


def test_matches_ranking_loss_when_every_patient_has_one_row() -> None:
    """Unit weights and one row per patient must reproduce the existing pair sum."""
    model = _make_model(seed=10).eval()
    x, t, e, s_all = _make_batch([1] * 8, seed=10)
    s = s_all[:, 0]

    rows, cif = _rows_from(model, x, t, e, s)
    weighted = weighted_ranking_loss(rows, rows, SIGMA)
    baseline = ranking_loss(conditional_cif(cif, s), t.long(), e.int(), SIGMA, s=s)

    assert abs(float(weighted) - float(baseline)) < 1e-10


def test_objective_reduces_to_total_loss_for_single_visit_patients() -> None:
    """With V_i = 1 the landmark-ranking objective is the final-visit objective."""
    model = _make_model(seed=11).eval()
    x, t, e, s_all = _make_batch([1] * 6, seed=11)
    args = dict(alpha=0.01, beta=0.75, sigma=SIGMA)

    combined = all_landmark_objective(
        model, x, t, e, s_all[:, 0], s_all,
        args["alpha"], args["beta"], args["sigma"], None,
        chunk_rows=2, rank_on_landmarks=True,
    )
    baseline = total_loss(model, x, t, e, s=s_all[:, 0], **args).item()

    assert abs(combined - baseline) < 1e-9


def test_a_patients_rows_carry_the_weight_of_one_patient() -> None:
    """V identical rows at weight 1/V rank exactly like the single row at weight 1."""
    torch.manual_seed(12)
    n_bins = 6
    hazard = torch.rand(3, n_bins).double() / (4 * n_bins)
    cif = [torch.cumsum(hazard, 1)]
    t = torch.tensor([5, 5, 5])
    s = torch.tensor([0, 0, 0])
    e = torch.tensor([1, 1, 0], dtype=torch.int32)

    single = RankRows(cif=cif, t=t, s=s, e=e,
                      weight=torch.ones(3).double(), patient=torch.arange(3))

    # Patient 0 observed at three landmarks that happen to look identical.
    V = 3
    take = torch.tensor([0, 0, 0, 1, 2])
    expanded = RankRows(
        cif=[c[take] for c in cif], t=t[take], s=s[take], e=e[take],
        weight=torch.tensor([1 / V, 1 / V, 1 / V, 1.0, 1.0]).double(),
        patient=torch.tensor([0, 0, 0, 1, 2]),
    )

    assert abs(float(weighted_ranking_loss(single, single, SIGMA))
               - float(weighted_ranking_loss(expanded, expanded, SIGMA))) < 1e-12


def test_landmarks_of_the_same_patient_are_not_compared() -> None:
    """Two landmarks of one patient differ only in s — not a discrimination signal."""
    torch.manual_seed(13)
    n_bins = 6
    cif = [torch.cumsum(torch.rand(2, n_bins).double() / (4 * n_bins), 1)]
    # One patient, two landmarks: the earlier one has the longer remaining time,
    # so without the patient guard it would be an eligible comparator.
    same = RankRows(cif=cif, t=torch.tensor([5, 5]), s=torch.tensor([0, 2]),
                    e=torch.tensor([1, 1], dtype=torch.int32),
                    weight=torch.ones(2).double(), patient=torch.tensor([0, 0]))
    assert abs(float(weighted_ranking_loss(same, same, SIGMA))) < 1e-12

    distinct = same._replace(patient=torch.tensor([0, 1]))
    assert float(weighted_ranking_loss(distinct, distinct, SIGMA)) > 1e-6


def test_cases_at_or_past_the_event_bin_are_dropped() -> None:
    torch.manual_seed(14)
    n_bins = 6
    cif = [torch.cumsum(torch.rand(3, n_bins).double() / (4 * n_bins), 1)]
    kwargs = dict(cif=cif, t=torch.tensor([5, 5, 5]),
                  e=torch.tensor([1, 1, 1], dtype=torch.int32),
                  weight=torch.ones(3).double(), patient=torch.arange(3))

    usable = RankRows(s=torch.tensor([0, 1, 2]), **kwargs)
    spent = RankRows(s=torch.tensor([5, 5, 5]), **kwargs)

    assert float(weighted_ranking_loss(usable, usable, SIGMA)) > 1e-6
    assert abs(float(weighted_ranking_loss(spent, spent, SIGMA))) < 1e-12


def test_chunking_changes_neither_the_loss_nor_the_gradient() -> None:
    """chunk_rows stays a memory knob: the checkpointed chunks are exact."""
    model = _make_model(seed=15).eval()
    x, t, e, s_all = _make_batch([1, 5, 3, 7], seed=15)
    args = (0.01, 0.75, SIGMA, None)

    def run(chunk_rows):
        model.zero_grad()
        value = all_landmark_objective(model, x, t, e, s_all[:, 0], s_all, *args,
                                       chunk_rows=chunk_rows, backward=True,
                                       rank_on_landmarks=True)
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        return value, grads

    whole, grad_whole = run(0)
    chunked, grad_chunked = run(3)

    # Same float-noise caveat as test_chunking_does_not_change_the_loss.
    assert abs(whole - chunked) < 1e-5
    assert len(grad_whole) == len(grad_chunked)
    for a, b in zip(grad_whole, grad_chunked):
        assert torch.allclose(a, b, atol=1e-6)


def test_expansion_moves_ranking_off_the_final_visit_only() -> None:
    """Multi-visit patients must give a different ranking term than V_i = 1 does."""
    model = _make_model(seed=16).eval()
    x, t, e, s_all = _make_batch([4, 4, 4, 4], seed=16)
    args = (0.01, 0.75, SIGMA, None)

    landmark = all_landmark_objective(model, x, t, e, s_all[:, 0], s_all, *args,
                                      chunk_rows=0, rank_on_landmarks=True)
    final_visit = all_landmark_objective(model, x, t, e, s_all[:, 0], s_all, *args,
                                         chunk_rows=0, rank_on_landmarks=False)
    # Only the ranking term moved; the survival and longitudinal terms are shared.
    assert abs(landmark - final_visit) > 1e-8


def test_prefix_rows_keep_one_unit_of_ranking_weight_per_patient() -> None:
    lengths = [1, 4, 2, 3]
    x, _, _, s_all = _make_batch(lengths, seed=17)
    _, _, patient_idx, weight = build_landmark_prefixes(x, s_all)
    for i in range(len(lengths)):
        assert abs(float(weight[patient_idx == i].sum()) - 1.0) < 1e-12
