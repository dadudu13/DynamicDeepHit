"""Equal weight per visit vs the per-patient 1 / V_i weighting."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ddh.losses import build_landmark_prefixes

from test_landmark_nll import _make_batch


def test_patient_weighting_is_unchanged():
    """The default path must stay bit-identical: every earlier arm is a control."""
    x, _, _, s_all = _make_batch([1, 3, 7, 2])
    *_, w_default = build_landmark_prefixes(x, s_all)
    *_, w_explicit = build_landmark_prefixes(x, s_all, row_weighting="patient")
    assert torch.equal(w_default, w_explicit)
    # 1 / V_i per row, so each patient totals 1.
    assert np.allclose(w_default.numpy(), np.repeat([1 / 1, 1 / 3, 1 / 7, 1 / 2], [1, 3, 7, 2]))


def test_visit_weighting_is_equal_and_keeps_the_batch_total():
    lengths = [1, 3, 7, 2]
    x, _, _, s_all = _make_batch(lengths)
    _, _, patient_idx, w = build_landmark_prefixes(x, s_all, row_weighting="visit")
    assert len(w) == sum(lengths)
    assert torch.allclose(w, w[0].expand_as(w))            # every row equal
    assert w.sum().item() == pytest.approx(len(lengths))   # same total as 1 / V_i
    # A patient now contributes in proportion to their visits.
    totals = torch.zeros(len(lengths), dtype=w.dtype).index_add(0, patient_idx, w)
    assert totals.numpy().argmax() == int(np.argmax(lengths))


def test_visit_weighting_rejects_decay():
    """Decay redistributes weight inside a patient's unit; 'visit' removes that unit."""
    x, _, _, s_all = _make_batch([2, 4])
    days = torch.zeros(x.shape[0], x.shape[1]).double()
    with pytest.raises(ValueError, match="incompatible"):
        build_landmark_prefixes(x, s_all, visit_days=days, decay_days=180.0,
                                row_weighting="visit")


def test_unknown_weighting_is_rejected():
    x, _, _, s_all = _make_batch([2, 2])
    with pytest.raises(ValueError, match="row_weighting"):
        build_landmark_prefixes(x, s_all, row_weighting="per_row")


def test_given_weighting_uses_the_supplied_weights():
    lengths = [2, 3]
    x, _, _, s_all = _make_batch(lengths)
    rw = torch.full((x.shape[0], x.shape[1]), float("nan")).double()
    rw[0, :2] = torch.tensor([1.0, 3.0]).double()
    rw[1, :3] = torch.tensor([2.0, 2.0, 2.0]).double()
    _, _, _, w = build_landmark_prefixes(x, s_all, row_weighting="given", row_weights=rw)
    # Supplied ratios are kept; only the overall scale is set by the batch total.
    assert w.sum().item() == pytest.approx(len(lengths))
    assert (w[1] / w[0]).item() == pytest.approx(3.0)
    assert (w[2] / w[0]).item() == pytest.approx(2.0)


def test_given_weighting_needs_weights():
    x, _, _, s_all = _make_batch([2, 2])
    with pytest.raises(ValueError, match="requires row_weights"):
        build_landmark_prefixes(x, s_all, row_weighting="given")


def _days(lengths, per_patient_days):
    d = torch.zeros(len(lengths), max(lengths)).double()
    for i, days in enumerate(per_patient_days):
        d[i, :len(days)] = torch.tensor(days).double()
    return d


def test_band_weighting_collapses_to_one_over_v_with_a_single_band():
    """One band covering the whole axis is the per-patient scheme, up to the batch scale."""
    lengths = [1, 3, 4]
    x, _, _, s_all = _make_batch(lengths)
    days = _days(lengths, [[0], [0, 10, 20], [0, 5, 400, 900]])
    # A single edge below day 0 puts every row in one band.
    _, _, _, w_band = build_landmark_prefixes(x, s_all, visit_days=days,
                                              row_weighting="band", band_edges=[-1.0])
    *_, w_patient = build_landmark_prefixes(x, s_all, row_weighting="patient")
    scale = w_patient.sum() / w_band.sum()
    assert torch.allclose(w_band * scale, w_patient)


def test_band_weighting_collapses_to_per_visit_when_each_band_holds_one_visit():
    lengths = [1, 3]
    x, _, _, s_all = _make_batch(lengths)
    days = _days(lengths, [[0], [0, 100, 300]])
    edges = [0.0, 50.0, 200.0, 500.0]
    _, _, _, w_band = build_landmark_prefixes(x, s_all, visit_days=days,
                                              row_weighting="band", band_edges=edges)
    _, _, _, w_visit = build_landmark_prefixes(x, s_all, row_weighting="visit")
    assert torch.allclose(w_band, w_visit)


def test_band_weighting_counts_each_patient_once_per_band():
    """The baseline visit has its own band; the three inside 1-365 d count a third each."""
    lengths = [5]
    x, _, _, s_all = _make_batch(lengths)
    days = _days(lengths, [[0, 7, 14, 21, 400]])
    edges = [0.0, 1.0, 366.0]          # bands: day 0 | 1-365 | 366+
    _, _, _, w = build_landmark_prefixes(x, s_all, visit_days=days,
                                         row_weighting="band", band_edges=edges)
    ratios = (w / w.min()).numpy()
    assert np.allclose(ratios, [3, 1, 1, 1, 3])   # day 0 alone | 7, 14, 21 | 400
    assert w.sum().item() == pytest.approx(1.0)   # one patient -> batch total 1


def test_band_weighting_needs_edges_and_days():
    x, _, _, s_all = _make_batch([2, 2])
    with pytest.raises(ValueError, match="band_edges"):
        build_landmark_prefixes(x, s_all, row_weighting="band")
