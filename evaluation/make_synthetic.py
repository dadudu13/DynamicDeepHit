"""Write a synthetic cohort in the layout the evaluation scripts read.

No real patient is involved. Each simulated patient has a categorical type, an
age, and a lab value measured at every visit; the event of interest (1) and a
competing event (2) arrive at exponential times, and follow-up is censored
administratively. Two properties of real follow-up data are built in:

- visits stop at the event, so patients with an early event have few visits
  (the informative visit pattern that makes 1/V_i landmark weighting biased);
- the lab rises over the weeks before an event of interest, so later visits
  carry information the first visit does not (what a dynamic model can use).

    python -m evaluation.make_synthetic --out /tmp/ddh_synthetic

writes <out>/split_<k>/{train,val,test,train_landmark,test_landmark,external}.csv
and <out>/columns.json, which tells the other scripts the column names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Columns
from .data import build_landmark_table

CONTINUOUS = ["age", "lab", "days_since_last_obs"]
COLUMNS = Columns(id="patient_id", date="visit_date", visit="visit", time="time_to_event",
                  event="event", event_date="event_date", categorical="cancer_type")
LANDMARK_DAYS = [0, 30, 60, 90, 120, 150, 180, 270]


def simulate(n: int, rng: np.random.Generator, start: str, span_days: int, first_id: int) -> pd.DataFrame:
    type_effect = np.array([-0.6, -0.2, 0.0, 0.4, 0.9])
    rows = []
    for i in range(n):
        ctype = int(rng.integers(0, len(type_effect)))
        age = rng.normal(62, 12)
        lab0 = rng.normal(0, 1)
        frailty = rng.normal(0, 0.4)
        t1 = rng.exponential(1 / (0.0003 * np.exp(type_effect[ctype] + 0.5 * lab0 + frailty)))
        t2 = rng.exponential(1 / (0.0003 * np.exp(0.03 * (age - 62))))
        cens = rng.uniform(150, 900)
        T = min(t1, t2, cens)
        event = 1 if T == t1 else 2 if T == t2 else 0
        first = pd.Timestamp(start) + pd.Timedelta(days=int(rng.integers(0, span_days)))

        day, visit, last_day = 0.0, 1, 0.0
        while day < T:
            rise = 1.5 * np.exp(-(t1 - day) / 45) if event == 1 else 0.0
            rows.append({"patient_id": first_id + i, "visit": visit,
                         "visit_date": first + pd.Timedelta(days=int(day)),
                         "time_to_event": float(np.ceil(T)), "event": event,
                         "event_date": first + pd.Timedelta(days=int(np.ceil(T))),
                         "cancer_type": ctype, "age": age,
                         "lab": lab0 + rise + rng.normal(0, 0.3),
                         "days_since_last_obs": day - last_day})
            last_day = day
            day += 7 + rng.exponential(28)
            visit += 1
    return pd.DataFrame(rows)


def truncate_at_anchor(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Keep each patient's visits up to a random anchor visit (a temporal-validation cohort)."""
    keep = []
    for _, g in df.groupby("patient_id", sort=False):
        keep.append(g.iloc[: int(rng.integers(1, len(g) + 1))])
    return pd.concat(keep, ignore_index=True)


def scale(frames: dict[str, pd.DataFrame], reference: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mean, sd = reference[CONTINUOUS].mean(), reference[CONTINUOUS].std().replace(0, 1)
    out = {}
    for name, f in frames.items():
        f = f.copy()
        f[CONTINUOUS] = (f[CONTINUOUS] - mean) / sd
        out[name] = f
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Data root to create.")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--patients", type=int, default=1500)
    ap.add_argument("--external-patients", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)

    cohort = simulate(args.patients, rng, "2018-01-01", 4 * 365, first_id=1)
    external = truncate_at_anchor(
        simulate(args.external_patients, rng, "2023-01-01", 365, first_id=10**6), rng)
    ids = cohort.patient_id.unique()
    for k in range(args.splits):
        perm = np.random.default_rng(args.seed + 100 + k).permutation(ids)
        n_tr, n_va = int(0.7 * len(perm)), int(0.15 * len(perm))
        parts = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
        raw = {name: cohort[cohort.patient_id.isin(p)] for name, p in parts.items()}
        for name in ("train", "test"):
            raw[f"{name}_landmark"] = build_landmark_table(
                raw[name], COLUMNS, LANDMARK_DAYS, gap_col="days_since_last_obs")
        raw["external"] = external
        split_dir = out / f"split_{k}"
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in scale(raw, raw["train"]).items():
            frame.to_csv(split_dir / f"{name}.csv", index=False)
    (out / "columns.json").write_text(json.dumps(COLUMNS.to_dict(), indent=2))
    events = cohort.groupby("patient_id").event.first().value_counts().to_dict()
    print(f"{args.patients} patients ({events}), {len(cohort)} visits, "
          f"{args.splits} splits, {external.patient_id.nunique()} external patients -> {out}")


if __name__ == "__main__":
    main()
