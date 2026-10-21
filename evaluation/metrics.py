"""Discrimination and calibration of a risk by horizon h, scored from a landmark.

Every function takes, per row (one patient at one landmark):

    t   time from the landmark to the event or censoring (days)
    e   0 censored, 1 event of interest, 2 competing event
    p   predicted risk of the event of interest by h

and treats the competing event as a competing event, never as censoring.

observed         Aalen-Johansen cumulative incidence of cause 1 at h.
O:E              observed / mean(p). 1 is calibrated; below 1 the model over-predicts.
AUC              IPCW cumulative/dynamic AUC with a competing event (Blanche et al.,
                 Stat Med 2013). Cases had the event of interest by h, each weighted
                 1/G(T-); controls were still event-free after h, weighted 1/G(h).
                 Rows censored, or with a competing event, before h are in neither
                 group; ties in p count 1/2. G is the reverse Kaplan-Meier censoring
                 distribution, with a day's events leaving its risk set before that
                 day's censorings (the prodlim / timeROC convention). Identical to R
                 timeROC's AUC_1 except that timeROC counts only events strictly
                 before h as cases.
IPCW weights     Graf et al.: 1/G(T-) for a row with any event by h, 1/G(h) for a row
                 still event-free at h, 0 for a row censored before h. G is the
                 Kaplan-Meier estimate of the censoring distribution (here a day's
                 events stay in its risk set, as in the original analysis code; this
                 differs from the AUC's G only on days with both).
Brier, IPA       IPCW Brier score of the status at h (y = 1 for the event of interest by
                 h, y = 0 otherwise, including a competing event by h). IPA = 1 - Brier
                 / Brier of a null model that predicts `observed` for every row.
slope, CITL      IPCW-weighted logistic regression of y on logit(p): its slope, and the
                 intercept with logit(p) as an offset (calibration-in-the-large).
top-decile O:E   O:E among the 10% of rows with the highest p.

These are the definitions the project's reports were built with; the functions are
NumPy ports of that code so the numbers reproduce exactly.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .config import EVENT_OF_INTEREST as EOI

EPS = 1e-6
MIN_CASES = 3


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def km_steps(t: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival of `event` (boolean) at each distinct time."""
    order = np.argsort(t, kind="mergesort")
    ts, ev = t[order], event[order].astype(float)
    uniq, first = np.unique(ts, return_index=True)
    at_risk = len(ts) - first
    return uniq, np.cumprod(1 - np.add.reduceat(ev, first) / at_risk)


def step_at(uniq: np.ndarray, surv: np.ndarray, x: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(uniq, x, side="right") - 1
    return np.where(pos >= 0, surv[np.clip(pos, 0, None)], 1.0)


def ipcw_weights(t: np.ndarray, e: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Graf weights and the binary status y at h (see module docstring)."""
    uniq, g = km_steps(t, e == 0)
    g_h = max(float(step_at(uniq, g, np.array([h], float))[0]), EPS)
    g_t = np.maximum(step_at(uniq, g, np.maximum(t - EPS, 0)), EPS)
    w = np.where((t <= h) & (e != 0), 1 / g_t, 0.0) + np.where(t > h, 1 / g_h, 0.0)
    return w, ((t <= h) & (e == EOI)).astype(float)


def aalen_johansen(t: np.ndarray, e: np.ndarray, h: float) -> float:
    """Cumulative incidence of the event of interest at h, competing events as such."""
    order = np.argsort(t, kind="mergesort")
    ts, es = t[order], e[order]
    uniq, first = np.unique(ts, return_index=True)
    at_risk = len(ts) - first
    d_eoi = np.add.reduceat((es == EOI).astype(float), first)
    d_any = np.add.reduceat((es != 0).astype(float), first)
    surv = np.cumprod(1 - d_any / at_risk)
    cif = np.cumsum(np.concatenate([[1.0], surv[:-1]]) * d_eoi / at_risk)
    pos = np.searchsorted(uniq, h, side="right") - 1
    return float(cif[pos]) if pos >= 0 else 0.0


def censoring_survival(t: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse Kaplan-Meier G(t) = P(C > t), a day's events leaving the risk set first."""
    order = np.argsort(t, kind="mergesort")
    ts, es = t[order], e[order]
    times, first = np.unique(ts, return_index=True)
    censored = np.add.reduceat((es == 0).astype(float), first)
    at_risk = (len(ts) - first) - np.add.reduceat((es != 0).astype(float), first)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(at_risk > 0, 1 - censored / at_risk, 1.0)
    return times, np.cumprod(step)


def auc(t: np.ndarray, e: np.ndarray, p: np.ndarray, h: float) -> float:
    """IPCW cumulative/dynamic AUC at h with a competing event (see module docstring)."""
    case, ctrl = (t <= h) & (e == EOI), t > h
    if case.sum() < MIN_CASES or ctrl.sum() == 0:
        return np.nan
    times, g = censoring_survival(t, e)
    w_case = 1.0 / np.maximum(step_at(times, g, np.maximum(t[case] - EPS, 0)), EPS)
    # Every control carries the same weight 1/G(h), which cancels from the ratio:
    # each case contributes the share of controls it outranks (ties 1/2), weighted.
    pk = np.sort(p[ctrl])
    below = np.searchsorted(pk, p[case], side="left")
    ties = np.searchsorted(pk, p[case], side="right") - below
    return float(np.sum(w_case * (below + 0.5 * ties) / len(pk)) / np.sum(w_case))


def slope_citl(w: np.ndarray, y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    import statsmodels.api as sm
    lp, k = logit(p), w > 0
    if k.sum() < 20 or y[k].sum() < 5:
        return np.nan, np.nan
    binomial = sm.families.Binomial()
    fit_citl = sm.GLM(y[k], np.ones((int(k.sum()), 1)), family=binomial, freq_weights=w[k],
                      offset=lp[k]).fit()
    citl = float(np.asarray(fit_citl.params)[0])
    if np.ptp(lp[k]) == 0:          # constant predictions: no slope to estimate
        return np.nan, citl
    fit_slope = sm.GLM(y[k], sm.add_constant(lp[k]), family=binomial, freq_weights=w[k]).fit()
    return float(np.asarray(fit_slope.params)[1]), citl


def top_decile_oe(t: np.ndarray, e: np.ndarray, p: np.ndarray, h: float) -> float:
    top = p >= np.quantile(p, 0.9)
    return aalen_johansen(t[top], e[top], h) / float(p[top].mean())


def evaluate(t: np.ndarray, e: np.ndarray, preds: dict[str, np.ndarray], h: float,
             with_slope: bool = True, with_top_decile: bool = True) -> dict[str, dict[str, float]]:
    """All metrics for every predictor in `preds`, on the same rows.

    The IPCW weights and the observed incidence depend only on (t, e), so they are
    computed once and shared, which keeps predictors strictly paired.
    """
    w, y = ipcw_weights(t, e, h)
    observed = aalen_johansen(t, e, h)
    brier_null = float(np.mean(w * (y - observed) ** 2))
    out = {}
    for name, p in preds.items():
        brier = float(np.mean(w * (y - p) ** 2))
        r = dict(auc=auc(t, e, p, h),
                 ipa=1 - brier / brier_null if brier_null > 0 else np.nan,
                 brier=brier, op=observed / float(p.mean()), obs=observed, pred=float(p.mean()))
        if with_slope:
            r["slope"], r["citl"] = slope_citl(w, y, p)
        if with_top_decile:
            r["top_op"] = top_decile_oe(t, e, p, h)
        out[name] = r
    return out


def bootstrap(t: np.ndarray, e: np.ndarray, preds: dict[str, np.ndarray], h: float,
              n_boot: int = 1000, seed: int = 7,
              metrics: tuple[str, ...] = ("auc", "ipa")) -> dict[str, dict[str, np.ndarray]]:
    """Bootstrap over patients, the same resampled rows for every predictor.

    Returns {name: {metric: one value per kept resample}}. Because all predictors
    share each resample, values[a] - values[b] is a paired difference. Resamples
    with fewer than MIN_CASES cases or no controls are skipped (the AUC is
    undefined there), for every predictor alike.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    values = {name: {m: [] for m in metrics} for name in preds}
    for _ in range(n_boot):
        k = rng.choice(np.arange(n), n, replace=True)
        tk, ek = t[k], e[k]
        if ((tk <= h) & (ek == EOI)).sum() < MIN_CASES or (tk > h).sum() == 0:
            continue
        r = evaluate(tk, ek, {name: p[k] for name, p in preds.items()}, h,
                     with_slope=False, with_top_decile=False)
        for name in preds:
            for m in metrics:
                values[name][m].append(r[name][m])
    return {name: {m: np.asarray(v, float) for m, v in d.items()} for name, d in values.items()}


def percentile_ci(v: np.ndarray) -> tuple[float, float, float]:
    """95% percentile interval and a two-sided bootstrap p-value for a zero difference."""
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))


def wilcoxon_paired(diffs: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value for per-split paired differences."""
    d = diffs[np.isfinite(diffs)]
    if len(d) == 0 or np.abs(d).sum() == 0:
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)
