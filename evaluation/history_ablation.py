"""Does the model actually use a patient's earlier visits? Within-patient ablation.

Comparing patients with long records against patients with short ones cannot answer
this: they are different patients. This scores the SAME patients with the SAME
trained models on a changed record:

  full        the patient's record up to the anchor, landmark s = anchor day
              (must equal external_anchor_predictions.csv; checked when present)
  masked      the anchor visit alone, s still the anchor day. What the encoder
              extracts from earlier visits, holding the time frame fixed. The anchor
              row keeps any history it already summarises (forward-filled labs, gaps).
  reanchored  the anchor visit alone, s = 0, and --gap-col reset to the value it has
              on first visits: the patient as if seen for the first time.
              Only with --gap-col.

Patients with at least --min-visits visits at the anchor are used. Outcomes come
from `full` and are shared, so every contrast is paired. Each variant is scored as
the ensemble mean of the split models, with a patient bootstrap of each contrast
(full - masked, full - reanchored, masked - reanchored).

Writes ablation_predictions.csv (patient level: keep it with the data) and
ablation_summary.csv / ablation_contrasts.csv (aggregates).

    python -m evaluation.history_ablation --arm band --gap-col days_since_last_obs
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import EXTERNAL_ANCHOR_FILE, Columns, add_path_arguments, paths_from_args
from .data import build_sequences, read_visits
from .metrics import bootstrap, evaluate, percentile_ci
from .model import load_run, predict_at_horizons, set_determinism

PAIRS = [("full", "masked"), ("full", "reanchored"), ("masked", "reanchored")]


def score(model, frame, predictors, horizons, cols, s=None):
    ds = build_sequences(frame, predictors, cols)
    if s is not None:
        ds.last_obs_days = ds.sample_index[cols.id].map(s).to_numpy(dtype=float)
    return predict_at_horizons(model, ds, horizons, cols).set_index(cols.id)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--min-visits", type=int, default=2)
    ap.add_argument("--gap-col", default=None, help="Days-since-last-visit predictor, for 'reanchored'.")
    ap.add_argument("--horizons", type=int, nargs="+", default=[90, 180])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=19)
    args = ap.parse_args(argv)
    cols, paths = Columns.load(args.columns), paths_from_args(args)
    variants = ["full", "masked"] + (["reanchored"] if args.gap_col else [])

    rows = []
    for split in args.splits:
        run_dir = paths.run_dir(args.arm, split)
        set_determinism(42)
        model, meta = load_run(run_dir, cols, read_visits(paths.split_file(split, "train"), cols))
        predictors, horizons = meta["predictors"], sorted(set(meta["horizons"]) | set(args.horizons))
        cohort = read_visits(paths.external_file(split), cols)
        n_visits = cohort.groupby(cols.id).size()
        hist = cohort[cohort[cols.id].isin(n_visits[n_visits >= args.min_visits].index)]
        anchor = hist.groupby(cols.id, sort=False).tail(1)

        out = {"full": score(model, hist, predictors, horizons, cols)}
        out["masked"] = score(model, anchor, predictors, horizons, cols, s=out["full"]["entry_time"])
        if args.gap_col:
            first_gap = cohort.groupby(cols.id, sort=False).head(1)[args.gap_col]
            if first_gap.round(6).nunique() != 1:
                raise SystemExit(f"--gap-col {args.gap_col} is not constant on first visits")
            fresh = anchor.copy()
            fresh[args.gap_col] = float(first_gap.iloc[0])
            out["reanchored"] = score(model, fresh, predictors, horizons, cols)

        saved = run_dir / EXTERNAL_ANCHOR_FILE
        if saved.exists():
            s = pd.read_csv(saved).set_index(cols.id).reindex(out["full"].index)
            drift = max(float(np.nanmax(np.abs(out["full"][f"risk_{h}d"] - s[f"risk_{h}d"])))
                        for h in meta["horizons"])
            if drift > 1e-6:
                raise SystemExit(f"split {split}: 'full' differs from {saved.name} by {drift:.2e}")
        for v in variants:
            frame = out[v].reindex(out["full"].index)
            for pid, r in frame.iterrows():
                rows.append({"split": split, "variant": v, cols.id: pid,
                             "n_visits": int(n_visits[pid]),
                             "time_ref": float(out["full"].loc[pid, "time_ref"]),
                             "event": int(out["full"].loc[pid, cols.event]),
                             **{f"risk_{h}d": float(r[f"risk_{h}d"]) for h in args.horizons}})
        print(f"split {split}: {len(out['full'])} patients with >= {args.min_visits} visits", flush=True)

    preds = pd.DataFrame(rows)
    preds.to_csv(paths.out("ablation_predictions.csv"), index=False)

    ref = preds[(preds.variant == "full") & (preds.split == args.splits[0])].sort_values(cols.id)
    ids, t, e = ref[cols.id].to_numpy(), ref.time_ref.to_numpy(float), ref.event.to_numpy()
    summary, contrasts = [], []
    for h in args.horizons:
        ens = {v: preds[preds.variant == v].pivot(index="split", columns=cols.id, values=f"risk_{h}d")
               [ids].to_numpy().mean(0) for v in variants}
        point = evaluate(t, e, ens, h)
        boots = bootstrap(t, e, ens, h, n_boot=args.n_boot, seed=args.seed, metrics=("auc", "ipa", "op"))
        for v, r in point.items():
            row = dict(horizon=h, variant=v, n=len(t), **r)
            for m in ("auc", "ipa", "op"):
                row[f"{m}_lo"], row[f"{m}_hi"], _ = percentile_ci(boots[v][m])
            summary.append(row)
        for a, b in PAIRS:
            if a not in ens or b not in ens:
                continue
            row = dict(horizon=h, contrast=f"{a} - {b}")
            for m in ("auc", "ipa", "op"):
                lo, hi, p = percentile_ci(boots[a][m] - boots[b][m])
                row.update({f"d_{m}": point[a][m] - point[b][m], f"d_{m}_lo": lo, f"d_{m}_hi": hi,
                            f"d_{m}_p": p})
            if (a, b) == ("full", "masked"):
                deciles = [pd.qcut(ens[v], 10, labels=False, duplicates="drop") for v in (a, b)]
                row.update(spearman=float(spearmanr(ens[a], ens[b]).correlation),
                           share_changing_decile=float(np.mean(deciles[0] != deciles[1])))
            contrasts.append(row)
    pd.DataFrame(summary).to_csv(paths.out("ablation_summary.csv"), index=False)
    pd.DataFrame(contrasts).to_csv(paths.out("ablation_contrasts.csv"), index=False)
    print(pd.DataFrame(contrasts).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
