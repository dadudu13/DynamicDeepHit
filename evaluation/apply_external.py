"""Score an external (e.g. later-calendar-period) cohort with every split's trained model.

The external table has the same layout as the training tables, preprocessed with
split k's own transforms (so it is read from <data-root>/split_k/). For a
temporal validation it should hold, per patient, every visit up to and including
the ANCHOR visit (e.g. the first visit inside the validation window) and nothing
after it, with the outcome measured from the patient's first visit.

Writes to <runs-root>/<arm>/split_k/:
    external_anchor_predictions.csv       each patient scored at the anchor (their last row)
    external_every_visit_predictions.csv  each patient scored at every retained visit

The evaluation averages the split models' anchor predictions into one ensemble.

    python -m evaluation.apply_external --arm band
"""
from __future__ import annotations

import argparse

from .config import (EXTERNAL_ANCHOR_FILE, EXTERNAL_EVERY_VISIT_FILE, Columns, add_path_arguments,
                     paths_from_args)
from .data import build_every_visit, build_sequences, read_visits
from .model import load_run, predict_at_horizons, set_determinism


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap, out=False)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    cols, paths = Columns.load(args.columns), paths_from_args(args)

    for split in args.splits:
        run_dir = paths.run_dir(args.arm, split)
        set_determinism(args.seed)
        train_frame = read_visits(paths.split_file(split, "train"), cols)
        model, meta = load_run(run_dir, cols, train_frame)
        cohort = read_visits(paths.external_file(split), cols)
        missing = [c for c in meta["predictors"] if c not in cohort]
        if missing:
            raise SystemExit(f"split {split}: external cohort lacks predictors {missing}")
        horizons = meta["horizons"]

        every = predict_at_horizons(model, build_every_visit(cohort, meta["predictors"], cols), horizons, cols)
        anchor = predict_at_horizons(model, build_sequences(cohort, meta["predictors"], cols), horizons, cols)
        for frame in (every, anchor):
            frame.insert(0, "split", split)
        every.to_csv(run_dir / EXTERNAL_EVERY_VISIT_FILE, index=False)
        anchor.to_csv(run_dir / EXTERNAL_ANCHOR_FILE, index=False)
        print(f"split {split}: {len(anchor)} patients, {len(every)} visits -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
