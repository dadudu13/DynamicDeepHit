from __future__ import annotations

import numpy as np
import torch

from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.losses import (
    build_landmark_prefixes,
    landmark_nll_loss,
    negative_log_likelihood,
    total_loss,
)
from ddh.utils import all_landmark_objective


def _make_batch(lengths, n_features=3, n_bins=8, seed=0):
    """Padded [N, T, D] batch where patient i has lengths[i] observed visits."""
    rng = np.random.default_rng(seed)
    t_max = max(lengths)
    x = np.full((len(lengths), t_max, n_features), np.nan)
    days = np.zeros((len(lengths), t_max))
    for i, n in enumerate(lengths):
        x[i, :n] = rng.normal(size=(n, n_features))
        days[i, :n] = np.arange(n)
    x = torch.from_numpy(x).double()
    # One landmark bin per visit, kept well below the event bin so no row is dropped.
    s_all = torch.from_numpy(days).long()
    t = torch.full((len(lengths),), n_bins - 1).double()
    e = torch.from_numpy(rng.integers(0, 3, size=len(lengths))).double()
    return x, t, e, s_all


def _make_model(n_features=3, n_bins=8, seed=0):
    torch.manual_seed(seed)
    return DynamicDeepHitTorch(
        n_features, n_bins, layers_rnn=1, hidden_rnn=4, typ="GRU",
        long_param={"layers": [8], "dropout": 0.0},
        att_param={"layers": [8], "dropout": 0.0},
        cs_param={"layers": [8], "dropout": 0.0},
        risks=2,
    ).double()


def test_prefix_expansion_shapes_and_masking() -> None:
    lengths = [1, 3, 2]
    x, _, _, s_all = _make_batch(lengths)

    x_exp, s_exp, patient_idx, weight = build_landmark_prefixes(x, s_all)

    assert x_exp.shape[0] == sum(lengths)
    assert patient_idx.tolist() == [0, 1, 1, 1, 2, 2]
    # Every patient contributes total weight 1 regardless of visit count.
    for i, n in enumerate(lengths):
        assert abs(float(weight[patient_idx == i].sum()) - 1.0) < 1e-12
    # Row v keeps exactly v+1 visits, and its landmark bin is that visit's bin.
    observed = (~torch.isnan(x_exp[:, :, 0])).sum(dim=1)
    assert observed.tolist() == [1, 1, 2, 3, 1, 2]
    assert s_exp.tolist() == [0, 0, 1, 2, 0, 1]


def test_prefix_row_matches_directly_truncated_sequence() -> None:
    """The masked prefix must give the model the same input as a real truncation."""
    model = _make_model().eval()
    x, _, _, s_all = _make_batch([1, 4, 2], seed=1)
    x_exp, _, _, _ = build_landmark_prefixes(x, s_all)

    # Patient 1 (4 visits) truncated to its first 2 visits == expanded row index 2.
    truncated = x[1:2, :2].clone()
    _, direct = model(truncated)
    _, expanded = model(x_exp[2:3])
    for a, b in zip(direct, expanded):
        assert torch.allclose(a, b, atol=1e-12)


def test_reduces_to_final_visit_nll_for_single_visit_patients() -> None:
    """With V_i = 1 the all-landmark NLL is exactly the existing objective."""
    model = _make_model(seed=2).eval()
    x, t, e, s_all = _make_batch([1, 1, 1, 1], seed=2)

    x_exp, s_exp, patient_idx, weight = build_landmark_prefixes(x, s_all)
    landmark = landmark_nll_loss(model, x_exp, t.long()[patient_idx],
                                 e.int()[patient_idx], s_exp, weight)

    _, outcomes = model(x)
    cif = [torch.cumsum(ok, 1) for ok in outcomes]
    baseline = negative_log_likelihood(outcomes, cif, t.long(), e.int(), s=s_all[:, 0])

    assert abs(float(landmark) - float(baseline)) < 1e-10


def test_objective_reduces_to_total_loss_for_single_visit_patients() -> None:
    model = _make_model(seed=3).eval()
    x, t, e, s_all = _make_batch([1, 1, 1], seed=3)
    args = dict(alpha=0.01, beta=0.75, sigma=0.5)

    combined = all_landmark_objective(
        model, x, t, e, s_all[:, 0], s_all,
        args["alpha"], args["beta"], args["sigma"], None, chunk_rows=2,
    )
    baseline = total_loss(model, x, t, e, s=s_all[:, 0], **args).item()

    assert abs(combined - baseline) < 1e-10


def test_chunking_does_not_change_the_loss() -> None:
    model = _make_model(seed=4).eval()
    x, t, e, s_all = _make_batch([1, 5, 3, 7], seed=4)
    args = dict(alpha=0.01, beta=0.75, sigma=0.5)

    whole = all_landmark_objective(model, x, t, e, s_all[:, 0], s_all,
                                   args["alpha"], args["beta"], args["sigma"], None,
                                   chunk_rows=0)
    chunked = all_landmark_objective(model, x, t, e, s_all[:, 0], s_all,
                                     args["alpha"], args["beta"], args["sigma"], None,
                                     chunk_rows=2)
    # Not bit-identical: BLAS blocks differently for different batch shapes, and
    # log(p + 1e-10) amplifies a 1e-16 difference in a near-zero bin probability
    # by ~1e9.  1e-5 on a loss of ~25 is that noise, not a different objective.
    assert abs(whole - chunked) < 1e-5


def test_chunked_backward_matches_single_graph_gradients() -> None:
    """Per-chunk backward accumulates the same gradient as one big graph."""
    model = _make_model(seed=5).eval()
    x, t, e, s_all = _make_batch([2, 6, 4], seed=5)
    x_exp, s_exp, patient_idx, weight = build_landmark_prefixes(x, s_all)
    t_exp, e_exp = t.long()[patient_idx], e.int()[patient_idx]

    model.zero_grad()
    landmark_nll_loss(model, x_exp, t_exp, e_exp, s_exp, weight).backward()
    whole = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad()
    for start in range(0, x_exp.shape[0], 3):
        sl = slice(start, start + 3)
        landmark_nll_loss(model, x_exp[sl], t_exp[sl], e_exp[sl],
                          s_exp[sl], weight[sl]).backward()
    chunked = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert len(whole) == len(chunked)
    for a, b in zip(whole, chunked):
        # Same float-noise caveat as test_chunking_does_not_change_the_loss.
        assert torch.allclose(a, b, atol=1e-6)


def test_landmarks_at_or_past_the_event_bin_are_dropped() -> None:
    model = _make_model(seed=6).eval()
    x, t, e, s_all = _make_batch([3, 3], n_bins=8, seed=6)
    x_exp, s_exp, patient_idx, weight = build_landmark_prefixes(x, s_all)
    t_exp, e_exp = t.long()[patient_idx], e.int()[patient_idx]

    kept = landmark_nll_loss(model, x_exp, t_exp, e_exp, s_exp, weight).detach()
    # Push every landmark to the event bin: nothing is left to condition on.
    dropped = landmark_nll_loss(model, x_exp, t_exp, e_exp,
                                torch.full_like(s_exp, int(t[0])), weight).detach()
    assert abs(float(dropped)) < 1e-12
    assert abs(float(kept)) > 1e-6
