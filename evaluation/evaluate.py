"""Score trained arms against each other on three populations, raw (no recalibration).

  landmarks  internal test split, every patient still at risk at each fixed landmark
             day; metrics per split, then mean and SD over splits, and each arm's
             per-split difference from the reference arm (Wilcoxon signed-rank,
             Benjamini-Hochberg within each horizon and metric)
  visits     internal test split, every visit, grouped by days since the first visit;
             metrics per split, then mean and SD over splits
  external   external cohort at each patient's anchor, the split models averaged into
             one ensemble; strata by history; patient bootstrap for each arm and for
             each arm's paired difference from the reference

Arms are compared on identical rows: a row any arm cannot score (NaN risk) is
dropped for every arm. A comparator (for example an existing clinical score) is
evaluated as one more arm: write its predicted risks into the same files an arm
writes (risk_<h>d columns) and pass its folder with --arm; rows it does not
cover then drop out for everyone, keeping the comparison paired.

Metric definitions: evaluation/metrics.py.

    python -m evaluation.evaluate --arm band --arm "1/V_i=patient_weighting" \\
        --reference band --stage landmarks visits external
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (EVENT_OF_INTEREST, EVERY_VISIT_FILE, EXTERNAL_ANCHOR_FILE,
                     FIXED_LANDMARK_FILE, Columns, add_path_arguments, paths_from_args)
from .metrics import benjamini_hochberg, bootstrap, evaluate, percentile_ci, wilcoxon_paired

VISIT_BANDS = [("first visit", 0, 0), ("days 1-90", 1, 90), ("days 91-180", 91, 180),
               ("days 181-365", 181, 365), (">365 days", 366, 10**6), ("all visits", 0, 10**6)]


def parse_arms(specs: list[str]) -> dict[str, str]:
    """'Label=folder' or 'folder' (label = folder name), in the order given."""
    arms = {}
    for spec in specs:
        label, _, folder = spec.rpartition("=")
        arms[label or folder] = folder
    return arms


def read_predictions(paths, folder: str, split: int, name: str) -> pd.DataFrame:
    df = pd.read_csv(paths.run_dir(folder, split) / name)
    return df[df.time_ref > 0].reset_index(drop=True)


def n_events(t, e, h):
    return int(((t <= h) & (e == EVENT_OF_INTEREST)).sum())


# ------------------------------------------------------------------ landmarks
def stage_landmarks(arms, paths, cols, splits, horizons, landmarks):
    rows = []
    for split in splits:
        frames = {a: read_predictions(paths, f, split, FIXED_LANDMARK_FILE.format(part="test"))
                  for a, f in arms.items()}
        ref = next(iter(frames.values()))
        for h in horizons:
            for lm in landmarks:
                base = (ref[ref.landmark_day == lm].drop_duplicates(cols.id).sort_values(cols.id))
                keys = base[cols.id].to_numpy()
                P = {a: f[f.landmark_day == lm].drop_duplicates(cols.id).set_index(cols.id)
                        .reindex(keys)[f"risk_{h}d"].to_numpy() for a, f in frames.items()}
                t, e = base.time_ref.to_numpy(float), base[cols.event].to_numpy()
                ok = np.all(np.isfinite(np.vstack(list(P.values()))), axis=0)
                for a, r in evaluate(t[ok], e[ok], {a: p[ok] for a, p in P.items()}, h).items():
                    rows.append(dict(split=split, horizon=h, landmark=lm, n=int(ok.sum()),
                                     events=n_events(t[ok], e[ok], h), arm=a, **r))
        print(f"landmarks: split {split}", flush=True)
    return pd.DataFrame(rows)


def summarise_splits(per_split: pd.DataFrame, keys: list[str], reference: str) -> pd.DataFrame:
    """Mean and SD over splits; per-split paired differences from the reference arm."""
    metrics = [m for m in ("auc", "ipa", "op", "slope", "citl", "top_op", "obs", "pred")
               if m in per_split]
    summary = per_split.groupby(keys + ["arm"], sort=False)[metrics + ["n", "events"]].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" if s == "std" else m for m, s in summary.columns]
    summary = summary.reset_index()
    tests = []
    for key, cell in per_split.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = cell[cell.arm == reference].set_index("split")
        for arm, g in cell[cell.arm != reference].groupby("arm", sort=False):
            g = g.set_index("split")
            for m in ("auc", "ipa"):
                d = (g[m] - r[m]).dropna().to_numpy()
                tests.append(dict(zip(keys, key), arm=arm, metric=m, delta=d.mean() if len(d) else np.nan,
                                  delta_sd=d.std(ddof=1) if len(d) > 1 else np.nan,
                                  splits_up=int((d > 0).sum()), p=wilcoxon_paired(d)))
    tests = pd.DataFrame(tests)
    if not tests.empty:
        family = [k for k in keys if k == "horizon"] + ["metric"]
        tests["q_bh"] = tests.groupby(family)["p"].transform(lambda p: benjamini_hochberg(p.to_numpy()))
        wide = tests.pivot_table(index=keys + ["arm"], columns="metric",
                                 values=["delta", "delta_sd", "splits_up", "p", "q_bh"], sort=False)
        wide.columns = [f"{m}_{s}_vs_ref" for s, m in wide.columns]
        summary = summary.merge(wide.reset_index(), on=keys + ["arm"], how="left")
    return summary


# ------------------------------------------------------------------ visits
def stage_visits(arms, paths, cols, splits, horizons):
    rows, key = [], [cols.id, cols.visit]
    for split in splits:
        frames = {a: read_predictions(paths, f, split, EVERY_VISIT_FILE.format(part="test"))
                  for a, f in arms.items()}
        base = next(iter(frames.values()))
        day = base.entry_time.to_numpy(float)
        t, e = base.time_ref.to_numpy(float), base[cols.event].to_numpy()
        for h in horizons:
            P = {a: base[key].merge(f[key + [f"risk_{h}d"]], on=key, how="left")[f"risk_{h}d"].to_numpy()
                 for a, f in frames.items()}
            for label, lo, hi in VISIT_BANDS:
                ok = (day >= lo) & (day <= hi) & np.all(np.isfinite(np.vstack(list(P.values()))), axis=0)
                for a, r in evaluate(t[ok], e[ok], {a: p[ok] for a, p in P.items()}, h).items():
                    rows.append(dict(split=split, horizon=h, stratum=label, n=int(ok.sum()),
                                     events=n_events(t[ok], e[ok], h), arm=a, **r))
        print(f"visits: split {split}", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ external
def stage_external(arms, paths, cols, splits, horizons, reference, n_boot, seed):
    first = next(iter(arms.values()))
    ref = read_predictions(paths, first, splits[0], EXTERNAL_ANCHOR_FILE).sort_values(cols.id)
    ids = ref[cols.id].to_numpy()
    t, e = ref.time_ref.to_numpy(float), ref[cols.event].to_numpy()
    day, n_visits = ref.entry_time.to_numpy(float), ref.num_visits.to_numpy()
    ensemble = {}
    for h in horizons:
        for a, folder in arms.items():
            ensemble[(h, a)] = np.vstack([
                read_predictions(paths, folder, s, EXTERNAL_ANCHOR_FILE).set_index(cols.id)
                .reindex(ids)[f"risk_{h}d"].to_numpy() for s in splits]).mean(0)
    covered = np.all(np.isfinite(np.vstack(list(ensemble.values()))), axis=0)
    strata = {"all": covered, "first visit": covered & (n_visits == 1),
              "history <=365 d": covered & (n_visits > 1) & (day <= 365),
              ">365 d": covered & (day > 365)}
    rows = []
    for h in horizons:
        for name, mask in strata.items():
            preds = {a: ensemble[(h, a)][mask] for a in arms}
            tm, em = t[mask], e[mask]
            point = evaluate(tm, em, preds, h)
            boots = bootstrap(tm, em, preds, h, n_boot=n_boot, seed=seed)
            for a, r in point.items():
                row = dict(horizon=h, stratum=name, n=int(mask.sum()), events=n_events(tm, em, h),
                           arm=a, **r)
                for m in ("auc", "ipa"):
                    row[f"{m}_lo"], row[f"{m}_hi"], _ = percentile_ci(boots[a][m])
                    if a != reference:
                        lo, hi, p = percentile_ci(boots[a][m] - boots[reference][m])
                        row.update({f"{m}_delta_vs_ref": r[m] - point[reference][m],
                                    f"{m}_delta_lo": lo, f"{m}_delta_hi": hi, f"{m}_delta_p": p})
                rows.append(row)
            print(f"external: horizon {h}, {name}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap, data=False)
    ap.add_argument("--arm", action="append", required=True,
                    help="'Label=folder' (folder under --runs-root, or absolute); repeatable.")
    ap.add_argument("--reference", default=None, help="Arm label every other arm is compared with "
                                                      "(default: the first --arm).")
    ap.add_argument("--stage", nargs="+", choices=["landmarks", "visits", "external"],
                    default=["landmarks", "visits", "external"])
    ap.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--horizons", type=int, nargs="+", default=[90, 180])
    ap.add_argument("--landmarks", type=int, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    cols, paths = Columns.load(args.columns), paths_from_args(args)
    arms = parse_arms(args.arm)
    reference = args.reference or next(iter(arms))
    if reference not in arms:
        raise SystemExit(f"--reference {reference!r} is not one of {list(arms)}")
    # The reference arm defines the rows and their order.
    arms = {reference: arms[reference], **{a: f for a, f in arms.items() if a != reference}}

    pd.set_option("display.width", 200)
    if "landmarks" in args.stage:
        per_split = stage_landmarks(arms, paths, cols, args.splits, args.horizons, args.landmarks)
        per_split.to_csv(paths.out("landmarks_per_split.csv"), index=False)
        summary = summarise_splits(per_split, ["horizon", "landmark"], reference)
        summary.to_csv(paths.out("landmarks_summary.csv"), index=False)
        print(summary.pivot_table(index=["horizon", "landmark"], columns="arm",
                                  values=["auc", "op"], sort=False).round(3).to_string())
    if "visits" in args.stage:
        per_split = stage_visits(arms, paths, cols, args.splits, args.horizons)
        per_split.to_csv(paths.out("visits_per_split.csv"), index=False)
        summary = summarise_splits(per_split, ["horizon", "stratum"], reference)
        summary.to_csv(paths.out("visits_summary.csv"), index=False)
        print(summary.pivot_table(index=["horizon", "stratum"], columns="arm",
                                  values=["auc", "op"], sort=False).round(3).to_string())
    if "external" in args.stage:
        ext = stage_external(arms, paths, cols, args.splits, args.horizons, reference,
                             args.n_boot, args.seed)
        ext.to_csv(paths.out("external.csv"), index=False)
        print(ext.pivot_table(index=["horizon", "stratum"], columns="arm",
                              values=["auc", "op"], sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
