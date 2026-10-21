# Training and evaluating landmark Dynamic-DeepHit models

These scripts are the training and evaluation protocol used with this fork, with
every file path and column name turned into a setting. No data ships with them:
they read a cohort you keep wherever you keep it, and `make_synthetic` builds a
stand-in cohort so the whole workflow can be run and tested without patient data.

On the cohort they were written for, these scripts reproduce the project's
original analysis code exactly. Checked:

* a retrained model's weights;
* the predictions at fixed landmarks, at every visit and on the external cohort;
* every metric and bootstrap interval in its calibration and discrimination tables;
* the within-patient ablation.

## The question being evaluated

At a visit (or at a fixed day after the first visit), a patient who is still
event-free gets a predicted risk of the event of interest within *h* days. A
competing event can pre-empt it. Each evaluation row is one patient at one
landmark. Its outcome is the time from that landmark to the event or to
censoring, and which event it was.

## Data layout

```
$DDH_DATA_ROOT/
  split_0/  train.csv  val.csv  test.csv          one row per (patient, visit)
            train_landmark.csv  test_landmark.csv  fixed-landmark tables (see below)
            external.csv                          optional: external / later-period cohort
  split_1/  ...
```

Each `split_k` is one random patient-level split. Every file in it, including
`external.csv`, has been preprocessed with transforms fitted on that split's
training patients.

**Per-visit tables.** Column names are set with a JSON file (`--columns` or
`DDH_COLUMNS`); the defaults are in brackets.

| column | meaning |
|---|---|
| patient id [`patient_id`] | |
| visit date [`visit_date`] | day 0 is the patient's first visit |
| visit number [`visit`] | order within the patient; not a predictor |
| time [`time_to_event`] | days from the **first** visit to event or censoring, the same on every row |
| event [`event`] | 0 censored, 1 event of interest, 2 competing event, the same on every row |
| event date [`event_date`] | optional; derived from date + time if absent |
| categorical [none] | optional integer-coded column given a learned embedding |
| meta [none] | any other non-predictor columns |

Every other column is a predictor. Predictors must be numeric and already
imputed and scaled. **They must not contain NaN**: the model reads NaN in the first
predictor as padding.

**Fixed-landmark tables.** `data.build_landmark_table` adds, for each patient
still at risk on day *L* (*L* in the landmark days), a snapshot row dated *L* days
after the first visit. It is forward-filled from their latest visit. "Days since"
columns are advanced to the landmark date, and the gap since the last visit is
recomputed. Build these from the unscaled table, then apply the same scaling.

**External cohort.** For a temporal validation, keep each patient's visits up to
and including an anchor visit (for example their first visit inside the validation
window) and nothing after it. The outcome is still measured from the first visit.

## Workflow

```sh
export DDH_DATA_ROOT=~/ddh/data DDH_RUNS_ROOT=~/ddh/runs DDH_OUTPUT_DIR=~/ddh/results

# 0. a synthetic cohort (skip with real data); it writes the column mapping it uses
python -m evaluation.make_synthetic --out $DDH_DATA_ROOT
export DDH_COLUMNS=$DDH_DATA_ROOT/columns.json

# 1. train two configurations on every split
python -m evaluation.train --arm patient --splits 0 1 2 --all-landmark-nll --alpha 0 --hazard --proportional
python -m evaluation.train --arm band    --splits 0 1 2 --all-landmark-nll --alpha 0 --hazard --proportional \
    --landmark-row-weighting band

# 2. score the external cohort with each split's model
python -m evaluation.apply_external --arm patient --splits 0 1 2
python -m evaluation.apply_external --arm band    --splits 0 1 2

# 3. compare them; the first --arm (or --reference) is the reference
python -m evaluation.evaluate --arm band --arm "1/V_i=patient" --splits 0 1 2

# 4. does the model use earlier visits?
python -m evaluation.history_ablation --arm band --splits 0 1 2 --gap-col days_since_last_obs
```

The demo uses the proportional hazard head. On a cohort this small the free hazard
head (drop `--proportional`) over-fits within the first epochs: training loss falls
while validation loss rises from the start, because each of its ~100 time bins has its
own output weights and the synthetic training set has only a few events per bin. The
proportional head shares one patient score across bins. The configuration the project
reports uses the free head on a larger cohort, where validation loss kept improving for
several epochs.

If your files use other names, set `DDH_FILE_TEMPLATE` (for example
`'cohort_{part}.csv'`) and `DDH_EXTERNAL_TEMPLATE`. If your run folders use another
layout, set `DDH_SPLIT_DIR_TEMPLATE`. Every flag has `--help`.

## What `evaluate` reports

All models are scored **raw**, with no recalibration. Every arm is scored on the
same rows, because a row any arm cannot score is dropped for all of them.

| stage | population | uncertainty | files |
|---|---|---|---|
| `landmarks` | internal test patients still at risk at days 0, 90, 180, 270 | per-split metrics; mean and SD over splits; each arm's per-split difference from the reference, Wilcoxon signed-rank, Benjamini-Hochberg within horizon and metric | `landmarks_per_split.csv`, `landmarks_summary.csv` |
| `visits` | internal test patients at every visit, by days since the first visit | as above | `visits_per_split.csv`, `visits_summary.csv` |
| `external` | external cohort at the anchor; the split models averaged into one ensemble; all / first visit / history ≤ 365 d / > 365 d | patient bootstrap (1,000 resamples) for each arm, and for each arm's paired difference from the reference | `external.csv` |

The metrics (definitions in [`metrics.py`](metrics.py)):

| metric | definition |
|---|---|
| `obs` | Aalen-Johansen cumulative incidence of the event of interest at *h*, with the competing event as a competing event |
| `op` (O:E) | `obs` / mean predicted risk. Below 1 means the model over-predicts. |
| `auc` | IPCW cumulative/dynamic AUC with a competing event: cases (event of interest by *h*), each weighted by the inverse probability of being uncensored at their event time, against controls (still event-free after *h*). Rows censored, or with a competing event, before *h* are in neither group. Matches R timeROC's `AUC_1`, except that an event on day *h* counts as a case here. |
| `brier`, `ipa` | IPCW Brier score of the status at *h*, with Graf weights from the Kaplan-Meier censoring distribution; IPA = 1 − Brier / Brier of predicting `obs` for everyone |
| `slope`, `citl` | weighted logistic recalibration of the status at *h* on logit(risk): its slope, and its intercept with logit(risk) as an offset |
| `top_op` | O:E among the 10% highest-risk rows |

**Comparing with an existing score.** Write the score's predicted risks
(`risk_<h>d` columns) into the same files an arm writes:

* `split_k/test_fixed_landmark_predictions.csv` and `split_k/test_every_visit_predictions.csv` for the internal stages;
* `split_k/external_anchor_predictions.csv` for the external stage.

Then pass that folder with `--arm`. Rows the score does not cover drop out for every
arm, so the comparison stays paired.

## Keeping patient data private

Predictions and `ablation_predictions.csv` are **patient-level**. Keep
`DDH_RUNS_ROOT` and `DDH_OUTPUT_DIR` next to your data, not inside a repository.
The tables `evaluate` writes are aggregates. This repository's `.gitignore` refuses
`*.csv`, `*.parquet`, `*.pkl` and `*.pt` files, as a guard against committing any of
them.
