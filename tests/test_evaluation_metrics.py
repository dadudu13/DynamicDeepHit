"""Metric and data-building helpers of the evaluation package, on hand-checkable inputs."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from evaluation.config import Columns
from evaluation.data import build_fixed_landmarks, build_landmark_table
from evaluation.metrics import (aalen_johansen, auc, benjamini_hochberg, bootstrap, evaluate,
                                ipcw_weights)
from evaluation.model import condition_cif_on_landmark


def cohort(seed=0, n=400):
    rng = np.random.default_rng(seed)
    t1, t2, c = rng.exponential(300, n), rng.exponential(500, n), rng.uniform(50, 600, n)
    t = np.ceil(np.minimum.reduce([t1, t2, c]))
    e = np.where(t1 <= np.minimum(t2, c), 1, np.where(t2 <= c, 2, 0))
    return t, e


def test_aalen_johansen_is_one_minus_kaplan_meier_without_competing_events():
    t, e = cohort()
    e = np.where(e == 2, 0, e)            # competing events become censoring
    order = np.argsort(t)
    surv, h = 1.0, 180
    for time in np.unique(t[order]):
        if time > h:
            break
        at_risk, d = (t >= time).sum(), ((t == time) & (e == 1)).sum()
        surv *= 1 - d / at_risk
    assert aalen_johansen(t, e, h) == pytest.approx(1 - surv, abs=1e-12)


def test_aalen_johansen_matches_scikit_survival():
    sksurv = pytest.importorskip("sksurv.nonparametric")
    t, e = cohort(1)
    times, cif = sksurv.cumulative_incidence_competing_risks(e, t)[:2]
    for h in (30, 90, 180, 365):
        idx = np.searchsorted(times, h, side="right") - 1
        assert aalen_johansen(t, e, h) == pytest.approx(float(cif[1][idx]), abs=1e-12)


def test_auc_counts_cases_by_h_against_controls_beyond_h():
    t = np.array([10, 20, 30, 200, 300, 400, 50, 60.])
    e = np.array([1, 1, 1, 0, 0, 2, 0, 2])      # last two: censored / competing before h
    p = np.array([.9, .8, .7, .1, .2, .3, .99, .99])
    assert auc(t, e, p, 100) == 1.0            # rows 6-7 must not enter either group
    assert auc(t, e, 1 - p, 100) == 0.0
    assert auc(t, e, np.full(8, .5), 100) == 0.5
    assert np.isnan(auc(t, e, p, 15))          # fewer than 3 cases


def test_auc_weights_cases_by_inverse_probability_of_censoring():
    # Reverse KM of censoring: one of five at risk is censored at day 20, so G = 0.8
    # from then on; the cases at 30 and 40 weigh 1/0.8 = 1.25, the case at 10 weighs 1.
    t = np.array([10, 20, 30, 40, 200, 300.])
    e = np.array([1, 0, 1, 1, 0, 0])
    p = np.array([.9, .4, .2, .6, .5, .3])     # case shares of controls outranked: 1, 0, 1
    assert auc(t, e, p, 100) == pytest.approx((1 * 1 + 1.25 * 0 + 1.25 * 1) / 3.5)


def test_auc_matches_timeroc():
    # R timeROC 0.4, weighting = "marginal", AUC_1 for cause 1, on this exact cohort
    # (continuous times, so no event falls on day h, where the two definitions differ).
    expected = {(0, 90): 0.670443158985, (0, 180): 0.686828729985,
                (1, 90): 0.650093708109, (1, 180): 0.727709021032}
    for (seed, h), value in expected.items():
        rng = np.random.default_rng(seed)
        x = rng.normal(size=400)
        t1, t2 = rng.exponential(300 * np.exp(-0.7 * x)), rng.exponential(500, 400)
        c = rng.uniform(30, 600, 400) * np.exp(-0.3 * x)
        t = np.minimum.reduce([t1, t2, c])
        e = np.where(t1 <= np.minimum(t2, c), 1, np.where(t2 <= c, 2, 0))
        p = np.round(1 / (1 + np.exp(-(x + rng.normal(0, .5, 400)))), 2)
        assert auc(t, e, p, h) == pytest.approx(value, abs=1e-11)


def test_ipcw_weights_without_censoring_are_one():
    t = np.array([10, 20, 200, 300.])
    e = np.array([1, 2, 1, 2])
    w, y = ipcw_weights(t, e, 100)
    assert np.allclose(w, 1.0)
    assert y.tolist() == [1, 0, 0, 0]


def test_ipa_is_zero_for_the_null_model_and_oe_is_observed_over_predicted():
    t, e = cohort(2)
    obs = aalen_johansen(t, e, 180)
    r = evaluate(t, e, {"null": np.full(len(t), obs), "half": np.full(len(t), obs / 2)}, 180)
    assert r["null"]["ipa"] == pytest.approx(0.0, abs=1e-12)
    assert r["null"]["op"] == pytest.approx(1.0)
    assert r["half"]["op"] == pytest.approx(2.0)


def test_bootstrap_is_paired():
    t, e = cohort(3)
    p = np.random.default_rng(0).uniform(size=len(t))
    b = bootstrap(t, e, {"a": p, "b": p.copy()}, 180, n_boot=50)
    assert np.array_equal(b["a"]["auc"], b["b"]["auc"])


def test_benjamini_hochberg():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20]))
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.20])


def test_conditioning_on_the_landmark():
    grid = np.array([10., 20., 30., 40.])
    f1 = np.array([[.1, .2, .3, .4]] * 3)
    f2 = np.array([[.0, .1, .1, .2]] * 3)
    c1, _ = condition_cif_on_landmark(f1, f2, grid, np.array([0., 20., 45.]))
    assert np.allclose(c1[0], f1[0])                       # s = 0: unchanged
    assert np.allclose(c1[1], np.clip((f1[1] - .2) / (1 - .2 - .1), 0, 1))
    assert np.allclose(c1[2], f1[2])                       # past the grid: raw CIF kept


def test_landmark_table_and_fixed_landmark_sequences():
    cols = Columns(categorical=None)
    visits = pd.DataFrame({
        "patient_id": [1, 1, 1, 2],
        "visit": [1, 2, 3, 1],
        "visit_date": pd.to_datetime(["2020-01-01", "2020-01-21", "2020-03-01", "2020-01-01"]),
        "time_to_event": [100.0, 100.0, 100.0, 40.0],
        "event": [1, 1, 1, 0],
        "lab": [1.0, 2.0, 3.0, 5.0],
        "days_since_x": [5.0, 25.0, 65.0, 0.0],
        "gap": [0.0, 20.0, 40.0, 0.0],
    })
    table = build_landmark_table(visits, cols, [0, 30, 90], reanchor_cols=["days_since_x"], gap_col="gap")
    snap = table[table.is_landmark == 1].set_index(["patient_id", "landmark_day"])
    # patient 2 is censored at day 40, so it has no day-90 snapshot
    assert sorted(snap.index) == [(1, 30), (1, 90), (2, 30)]
    assert snap.loc[(1, 30), "lab"] == 2.0                 # forward-filled from day 20
    assert snap.loc[(1, 30), "gap"] == 10.0                # days since the last real visit
    assert snap.loc[(1, 30), "days_since_x"] == 35.0       # 25 at day 20, advanced 10 days
    assert snap.loc[(1, 90), "lab"] == 3.0

    ds = build_fixed_landmarks(table, ["lab", "days_since_x", "gap"], cols)
    idx = ds.sample_index.set_index(["patient_id", "landmark_day"])
    lengths = dict(zip(idx.index, [len(x) for x in ds.x]))
    assert lengths == {(1, 0): 1, (1, 30): 3, (1, 90): 4, (2, 0): 1, (2, 30): 2}
    assert dict(zip(idx.index, ds.last_obs_days)) == {(1, 0): 0, (1, 30): 30, (1, 90): 90,
                                                      (2, 0): 0, (2, 30): 30}
