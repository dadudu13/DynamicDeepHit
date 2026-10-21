"""Train one arm (one training configuration) on each split and write its predictions.

For every split k this reads <data-root>/split_k/{train,val,test} (and the
fixed-landmark tables when they exist), trains on train with early stopping on
val, and writes to <runs-root>/<arm>/split_k/:

    model_state.pt, metadata.json             checkpoint and everything needed to rebuild it
    {train,val,test}_last_visit_predictions   each patient scored at their last visit
    {train,val,test}_every_visit_predictions  each patient scored at every visit
    {train,test}_fixed_landmark_predictions   each at-risk patient scored at each landmark day

The arm the project reports (hazard head, all-landmark likelihood, band weighting):

    python -m evaluation.train --arm band --all-landmark-nll --alpha 0 --hazard \\
        --landmark-row-weighting band
"""
from __future__ import annotations

import argparse
import json
import traceback

import pandas as pd

from .config import (EVERY_VISIT_FILE, FIXED_LANDMARK_FILE, LAST_VISIT_FILE, Columns,
                     add_path_arguments, paths_from_args)
from .data import (build_every_visit, build_fixed_landmarks, build_sequences, predictor_columns,
                   read_visits)
from .model import Architecture, Objective, categorical_embedding, predict_at_horizons, save_run, train


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap, out=False)
    ap.add_argument("--arm", required=True, help="Name of this configuration; its runs go to <runs-root>/<arm>.")
    ap.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--iters", type=int, default=64, help="Maximum epochs (early stopping on val).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizons", type=int, nargs="+", default=[30, 90, 180],
                    help="Days after the landmark at which risk is reported.")
    ap.add_argument("--landmark-days", type=int, nargs="+", default=[0, 30, 60, 90, 120, 150, 180, 270],
                    help="Fixed landmark days scored from the landmark tables.")

    obj = ap.add_argument_group("training objective (defaults reproduce the original loss)")
    obj.add_argument("--alpha", type=float, default=0.01, help="Ranking term weight; 0 removes it.")
    obj.add_argument("--beta", type=float, default=0.75, help="Survival likelihood weight.")
    obj.add_argument("--sigma", type=float, default=0.5, help="Ranking sharpness.")
    obj.add_argument("--all-landmark-nll", action="store_true",
                     help="Score the conditional likelihood at every (patient, visit) landmark.")
    obj.add_argument("--landmark-row-weighting", choices=["patient", "visit", "band"], default="patient",
                     help="With --all-landmark-nll: patient = 1/V_i per row; visit = equal rows; "
                          "band = each patient once per landmark-day band.")
    obj.add_argument("--landmark-band-edges", type=float, nargs="+", default=[0, 30, 90, 180, 365, 730],
                     help="Band edges in days for --landmark-row-weighting band.")
    obj.add_argument("--landmark-decay-days", type=float, default=None,
                     help="With --all-landmark-nll and patient weighting: weight landmarks by exp(-day/T).")
    obj.add_argument("--landmark-augment-prob", type=float, default=0.0,
                     help="Per-epoch probability of truncating a patient to a random earlier visit.")
    obj.add_argument("--gamma", type=float, default=0.0, help="Horizon-specific likelihood weight.")
    obj.add_argument("--horizon-days", type=int, nargs="+", default=[90, 180],
                     help="Horizons the --gamma term scores.")
    obj.add_argument("--horizon-ipcw-clip", type=float, default=10.0)
    obj.add_argument("--first-visit-only", action="store_true",
                     help="Static model: keep each patient's first visit only, in every part.")

    arch = ap.add_argument_group("architecture")
    arch.add_argument("--hidden-rnn", type=int, default=32)
    arch.add_argument("--hazard", action="store_true", help="Discrete-time cause-specific hazard head.")
    arch.add_argument("--proportional", action="store_true", help="With --hazard: proportional factorisation.")
    args = ap.parse_args(argv)

    if args.landmark_row_weighting != "patient" and not args.all_landmark_nll:
        ap.error("--landmark-row-weighting needs --all-landmark-nll")
    if args.all_landmark_nll and args.landmark_augment_prob > 0:
        ap.error("--all-landmark-nll already covers every landmark; drop --landmark-augment-prob")
    if args.gamma and not args.all_landmark_nll:
        ap.error("--gamma needs --all-landmark-nll")
    return args


def run_split(split: int, args: argparse.Namespace, cols: Columns) -> dict:
    paths = paths_from_args(args)
    parts = {part: paths.split_file(split, part) for part in ("train", "val", "test")}
    frames = {part: read_visits(path, cols) for part, path in parts.items()}
    if args.first_visit_only:
        frames = {part: f.groupby(cols.id, sort=False).head(1).reset_index(drop=True)
                  for part, f in frames.items()}

    predictors = predictor_columns(frames["train"], cols)
    vocab, cat_index = categorical_embedding(frames["train"], predictors, cols)
    sequences = {part: build_sequences(f, predictors, cols) for part, f in frames.items()}

    arch = Architecture(hidden_rnn=args.hidden_rnn, hazard=args.hazard, proportional=args.proportional)
    objective = Objective(
        alpha=args.alpha, beta=args.beta, sigma=args.sigma, all_landmark_nll=args.all_landmark_nll,
        landmark_row_weighting=args.landmark_row_weighting,
        landmark_band_edges=args.landmark_band_edges if args.landmark_row_weighting == "band" else None,
        landmark_decay_days=args.landmark_decay_days, landmark_augment_prob=args.landmark_augment_prob,
        gamma=args.gamma, horizon_days=args.horizon_days, horizon_ipcw_clip=args.horizon_ipcw_clip,
        first_visit_only=args.first_visit_only)
    model = train(sequences["train"], sequences["val"], args.horizons, arch, objective, vocab, cat_index,
                  seed=args.seed, iters=args.iters, batch_size=args.batch_size,
                  learning_rate=args.learning_rate)

    predictions = {}
    for part, ds in sequences.items():
        predictions[LAST_VISIT_FILE.format(part=part)] = predict_at_horizons(model, ds, args.horizons, cols)
    if not args.first_visit_only:
        for part, f in frames.items():
            ds = build_every_visit(f, predictors, cols)
            predictions[EVERY_VISIT_FILE.format(part=part)] = predict_at_horizons(model, ds, args.horizons, cols)
    inputs = dict(parts)
    for part in ("train", "test"):
        table_path = paths.split_file(split, f"{part}_landmark")
        if not table_path.exists():
            continue
        inputs[f"{part}_landmark"] = table_path
        ds = build_fixed_landmarks(pd.read_csv(table_path, low_memory=False), predictors, cols,
                                   args.landmark_days, first_visit_only=args.first_visit_only)
        predictions[FIXED_LANDMARK_FILE.format(part=part)] = predict_at_horizons(model, ds, args.horizons, cols)

    run_dir = paths.run_dir(args.arm, split)
    save_run(run_dir, model, predictors=predictors, horizons=args.horizons, arch=arch,
             objective=objective, cols=cols, vocab_size=vocab, categorical_index=cat_index,
             inputs=inputs, seed=args.seed, predictions=predictions)
    print(f"split {split}: {len(predictors)} predictors, "
          f"{len(sequences['train'].x)} train patients -> {run_dir}", flush=True)
    return {"split": split, "patients": {p: len(s.x) for p, s in sequences.items()}}


def main(argv=None) -> None:
    args = parse_args(argv)
    cols = Columns.load(args.columns)
    done, failed = [], []
    for split in args.splits:
        try:
            done.append(run_split(split, args, cols))
        except Exception as exc:
            print(f"[error] split {split}: {exc}\n{traceback.format_exc()}", flush=True)
            failed.append(split)
    print(json.dumps({"completed": done, "failed": failed}, indent=2))
    if failed:
        raise SystemExit(f"failed splits: {failed}")


if __name__ == "__main__":
    main()
