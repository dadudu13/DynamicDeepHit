"""Turn a per-visit table into the ragged sequences DynamicDeepHit reads.

The input is one row per (patient, visit), already preprocessed (imputed, scaled),
with the outcome repeated on every row: time from the patient's FIRST visit to the
event or censoring, and the event code. Three views are built from it:

  sequences       one sample per patient, their full observed history; the landmark
                  s is the day of their last visit (training, and "score at last visit")
  every visit     one sample per (patient, visit), history truncated at that visit;
                  s is that visit's day ("score at every visit")
  fixed landmark  one sample per (patient, landmark day) for patients still at risk,
                  history up to the landmark plus a forward-filled snapshot row dated
                  on the landmark ("score everyone at day 0, 90, 180, ...")

Day 0 is each patient's first visit throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Columns


@dataclass
class SequenceDataset:
    x: np.ndarray              # (N,) object array of (visits_i, features) arrays
    t: np.ndarray              # (N,) object array of per-visit outcome times
    e: np.ndarray              # (N,) object array of per-visit event codes
    sample_index: pd.DataFrame  # one row per sample: who, which visit, outcome
    last_obs_days: np.ndarray  # landmark s per sample, days from the first visit
    visit_days: np.ndarray | None = None  # (N,) object array of each visit's day


def ragged(arrays: list[np.ndarray]) -> np.ndarray:
    """1-D object array of per-sample arrays.

    np.array(list, dtype=object) stacks the arrays into a dense block when every
    sample has the same number of visits (always, for a first-visit-only cohort),
    which the model's padding code cannot read.
    """
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def read_visits(path: str | Path, cols: Columns) -> pd.DataFrame:
    """Read one per-visit table, check the required columns, and sort it."""
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in (cols.id, cols.date, cols.visit, cols.time, cols.event) if c not in df]
    if missing:
        raise ValueError(f"{path} is missing required column(s) {missing}; "
                         "map your names with --columns.")
    df[cols.date] = pd.to_datetime(df[cols.date])
    return df.sort_values([cols.id, cols.date, cols.visit]).reset_index(drop=True)


def predictor_columns(df: pd.DataFrame, cols: Columns) -> list[str]:
    """Every column that is not pipeline metadata, in file order."""
    skip = cols.non_predictors()
    return [c for c in df.columns if c not in skip]


def _visit_days(patient: pd.DataFrame, cols: Columns) -> np.ndarray:
    dates = patient[cols.date]
    return (dates - dates.iloc[0]).dt.days.to_numpy(dtype=float)


def build_sequences(df: pd.DataFrame, features: list[str], cols: Columns) -> SequenceDataset:
    """One sample per patient, using their complete visit sequence."""
    x, t, e, last_obs, visit_days, rows = [], [], [], [], [], []
    for pid, patient in df.sort_values([cols.id, cols.date, cols.visit]).groupby(cols.id, sort=False):
        patient = patient.reset_index(drop=True)
        days = _visit_days(patient, cols)
        x.append(patient[features].to_numpy(dtype=float))
        t.append(patient[cols.time].to_numpy(dtype=float))
        e.append(patient[cols.event].to_numpy(dtype=float))
        visit_days.append(days)
        last_obs.append(days[-1])
        last = patient.iloc[-1]
        rows.append({cols.id: pid, cols.visit: last[cols.visit], cols.date: last[cols.date],
                     cols.time: float(last[cols.time]), cols.event: int(last[cols.event]),
                     "num_visits": len(patient)})
    return SequenceDataset(x=ragged(x), t=ragged(t), e=ragged(e),
                           sample_index=pd.DataFrame(rows),
                           last_obs_days=np.array(last_obs, dtype=float),
                           visit_days=ragged(visit_days))


def build_every_visit(df: pd.DataFrame, features: list[str], cols: Columns) -> SequenceDataset:
    """One sample per (patient, visit), with the input truncated at that visit.

    The outcome is the patient-level one (time from the first visit) on every row;
    entry_time is the visit's day and time_ref the time remaining from it.
    """
    x, t, e, last_obs, rows = [], [], [], [], []
    for pid, patient in df.sort_values([cols.id, cols.date, cols.visit]).groupby(cols.id, sort=False):
        patient = patient.reset_index(drop=True)
        full_x = patient[features].to_numpy(dtype=float)
        days = _visit_days(patient, cols)
        terminal_t = float(patient.iloc[-1][cols.time])
        terminal_e = int(patient.iloc[-1][cols.event])
        for n in range(len(patient)):
            x.append(full_x[: n + 1])
            t.append(np.array([terminal_t]))
            e.append(np.array([terminal_e]))
            last_obs.append(days[n])
            rows.append({cols.id: pid, cols.visit: patient.iloc[n][cols.visit],
                         cols.date: patient.iloc[n][cols.date],
                         cols.time: terminal_t, cols.event: terminal_e,
                         "entry_time": int(days[n]), "time_ref": terminal_t - days[n]})
    return SequenceDataset(x=ragged(x), t=ragged(t), e=ragged(e),
                           sample_index=pd.DataFrame(rows),
                           last_obs_days=np.array(last_obs, dtype=float))


def build_fixed_landmarks(
    table: pd.DataFrame,
    features: list[str],
    cols: Columns,
    landmark_days: list[int] | None = None,
    first_visit_only: bool = False,
) -> SequenceDataset:
    """One sample per (patient, landmark day) from a landmark table.

    The table (see build_landmark_table) holds each patient's real visits
    (is_landmark = 0; the first visit has landmark_day = 0) and one synthetic
    snapshot per later landmark day (is_landmark = 1), forward-filled from the most
    recent real visit. The input at landmark L is every real visit on or before
    day L, followed by the snapshot for L. s = L, so the CIF is conditioned on
    surviving to the landmark.

    first_visit_only gives every landmark the first visit alone as input (for a
    model trained on first visits only); the at-risk set and s are unchanged, so
    the rows pair one-for-one with a dynamic model's.
    """
    for c in ("landmark_day", "is_landmark"):
        if c not in table:
            raise ValueError(f"landmark table has no '{c}' column; build it with build_landmark_table")
    table = table.copy()
    table[cols.date] = pd.to_datetime(table[cols.date])
    table = table.sort_values([cols.id, cols.date]).reset_index(drop=True)
    wanted = None if landmark_days is None else {float(d) for d in landmark_days}

    x, t, e, last_obs, rows = [], [], [], [], []
    for pid, patient in table.groupby(cols.id, sort=False):
        patient = patient.reset_index(drop=True)
        real = patient[patient["is_landmark"] == 0].sort_values(cols.date).reset_index(drop=True)
        first_date = real.iloc[0][cols.date]
        terminal_t = float(patient[cols.time].iloc[0])
        terminal_e = int(patient[cols.event].iloc[0])
        days = sorted(patient.loc[patient["landmark_day"].notna(), "landmark_day"].unique())
        if wanted is not None:
            days = [d for d in days if float(d) in wanted]
        for day in days:
            day = float(day)
            landmark_date = first_date + pd.Timedelta(days=int(day))
            real_before = real[real[cols.date] <= landmark_date]
            snapshot = patient[(patient["landmark_day"] == day) & (patient["is_landmark"] == 1)]
            if first_visit_only:
                seq = real.iloc[[0]].copy()
            elif snapshot.empty:          # day 0: the first visit is the prediction point
                seq = real_before.copy()
            else:
                seq = pd.concat([real_before, snapshot], ignore_index=True)
            seq = seq.sort_values(cols.date).reset_index(drop=True)
            last_real = seq.iloc[-1] if first_visit_only else real_before.iloc[-1]

            x.append(seq[features].to_numpy(dtype=float))
            t.append(np.array([terminal_t]))
            e.append(np.array([terminal_e]))
            last_obs.append(day)
            rows.append({cols.id: pid, cols.visit: last_real[cols.visit],
                         cols.time: terminal_t, cols.event: terminal_e,
                         "landmark_day": int(day),
                         "landmark_date": landmark_date.date().isoformat(),
                         "num_visits_used": len(seq)})
    return SequenceDataset(x=ragged(x), t=ragged(t), e=ragged(e),
                           sample_index=pd.DataFrame(rows),
                           last_obs_days=np.array(last_obs, dtype=float))


def build_landmark_table(
    df: pd.DataFrame,
    cols: Columns,
    landmark_days: list[int],
    reanchor_cols: list[str] | None = None,
    gap_col: str | None = None,
    no_ffill: list[str] | None = None,
) -> pd.DataFrame:
    """Add one forward-filled snapshot row per patient and landmark day.

    For every patient and every L in landmark_days a row dated first_visit + L is
    merged into their visit history. Snapshot rows (is_landmark = 1; landmark day 0
    coincides with the first visit, which is real) take every covariate from the
    most recent real visit, except:

    - reanchor_cols, "days since X" columns, which are advanced by the gap between
      that visit and the landmark instead of being copied;
    - gap_col, if given, which becomes the number of days since the last real visit.

    Rows on or after the patient's event/censoring date are dropped, so a patient
    appears at a landmark only while still at risk. Run this on the per-visit table
    BEFORE scaling, so the day arithmetic is in days.
    """
    reanchor_cols = list(reanchor_cols or [])
    df = df.copy()
    df[cols.date] = pd.to_datetime(df[cols.date], errors="coerce")
    df = df.sort_values([cols.id, cols.date]).reset_index(drop=True)
    df["_date_orig"] = df[cols.date]
    if cols.event_date in df:
        event_date = pd.to_datetime(df[cols.event_date])
    else:
        first = df.groupby(cols.id)[cols.date].transform("min")
        event_date = first + pd.to_timedelta(df[cols.time], unit="D")
    df["_event_date"] = event_date

    first_dates = (df.groupby(cols.id, as_index=False)[cols.date].min()
                     .rename(columns={cols.date: "_first_date"}))
    frames = []
    for day in landmark_days:
        tmp = first_dates.copy()
        tmp["landmark_day"] = day
        tmp[cols.date] = tmp["_first_date"] + pd.to_timedelta(day, unit="D")
        frames.append(tmp)
    base = pd.concat(frames, ignore_index=True)
    base["is_prediction_row"] = 1

    keep_as_is = {cols.id, cols.date, cols.visit, "is_prediction_row", "is_landmark",
                  "landmark_day", *reanchor_cols, *(no_ffill or [])}
    if gap_col:
        keep_as_is.add(gap_col)
    ffill_cols = [c for c in df.columns if c not in keep_as_is]

    lm = base.merge(df, on=[cols.id, cols.date], how="outer")
    lm["is_prediction_row"] = lm["is_prediction_row"].fillna(0).astype(int)
    lm = lm.sort_values([cols.id, cols.date, "landmark_day"]).reset_index(drop=True)
    lm["is_landmark"] = lm[cols.visit].isna().astype(int)   # no real visit on that date
    lm[ffill_cols] = lm.groupby(cols.id, sort=False)[ffill_cols].ffill()

    lm = lm.loc[lm[cols.date] < lm["_event_date"]].copy()
    lm["_gap_to_landmark"] = (lm[cols.date] - lm["_date_orig"]).dt.days.astype(float)
    snap = lm["is_landmark"].eq(1)
    if gap_col:
        lm.loc[snap, gap_col] = lm.loc[snap, "_gap_to_landmark"]
    for c in reanchor_cols:
        at_visit = lm.groupby([cols.id, "_date_orig"])[c].transform("first")
        lm[c] = at_visit + lm["_gap_to_landmark"]
    return lm.drop(columns=["_first_date", "_event_date"])


def choose_time_grid(train: SequenceDataset, val: SequenceDataset | None,
                     horizons: list[int], num_quantiles: int = 100) -> list[float]:
    """Discretisation grid: quantiles of train (+ val) outcome times, plus the horizons.

    Including the validation times keeps validation outcomes inside the grid.
    """
    times = np.array([seq[-1] for seq in train.t], dtype=float)
    if val is not None:
        times = np.concatenate([times, np.array([seq[-1] for seq in val.t], dtype=float)])
    grid = set(float(v) for v in np.quantile(times, np.linspace(0.0, 1.0, num_quantiles + 1)[1:]))
    grid.update(float(h) for h in horizons)
    grid.add(float(times.max()))
    return [0.0] + sorted(v for v in grid if v > 0)
