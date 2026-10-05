# Modifications to Dynamic-DeepHit

This fork starts from [Jeanselme/DynamicDeepHit](https://github.com/Jeanselme/DynamicDeepHit)
(PyTorch port of Lee et al., *Dynamic-DeepHit*, IEEE TBME 2020) and adapts it to
**landmark prediction from irregular clinic visits with a competing risk**: at any visit,
predict the risk of the event of interest within *h* days, given everything seen so far
and given that the patient is still event-free.

Every change below is one commit, and every commit in the history passes the test
suite that exists at that point. The changes come in two groups:

* **Foundation** (sections 1–2): fixes and infrastructure. Several of these change the
  model's default behaviour on purpose, because the upstream behaviour was a bug.
* **Modifications** (sections 3–7): new training objectives and output layers. Each
  one is behind an argument whose default leaves the model exactly as it was. This
  was checked at every commit by training a small model with default arguments, with
  and without landmarks, and comparing the predictions, which stayed bit-identical
  across all thirteen commits.

Notation: patient *i* has visits at days *s<sub>i1</sub> < … < s<sub>iV<sub>i</sub></sub>*
(day 0 = first visit). Cause 1 is the event of interest and cause 2 the competing
event. *F<sub>k</sub>(t)* is the cumulative incidence of cause *k*, and
*S(t) = 1 − Σ<sub>k</sub> F<sub>k</sub>(t)*.

---

## 1. Foundation: making the port correct and reproducible

| Change | Why | Effect on defaults |
|---|---|---|
| Import the DSM helpers from **auton-survival** through `ddh/dsm_compat.py` | The `DeepSurvivalMachines` package the code imported from was renamed and repackaged; its old import paths no longer resolve | none |
| **Clip discretised times** to the bins training can reach | A time equal to the first bin edge got index −1, which torch reads as the *last* bin, so the earliest events were scored as the latest | yes (bug fix) |
| **Training loop**: reshuffle each epoch, validate on every row, ReduceLROnPlateau, gradient clipping, configurable patience | Validation counted batches with one batch size and sliced with another, so early stopping watched only a fixed fraction of the validation set | yes |
| **Seed every RNG** in `fit` | `random_state` reached only the train/validation shuffle; weight initialisation and batch order were unseeded | yes |
| **Restore RNN bias terms** | With `bias=False` no gate can learn an offset, and an all-mean input from a zero state is stuck at a zero hidden state | yes |
| Optional **learned embedding** for one integer-coded categorical column (`cancer_vocab_size`, `cancer_embedding_dim`, `cancer_id_col`) | An integer-coded category fed as a number is treated as ordinal | none (off unless all three are set) |
| **σ inside the exponential** of the ranking loss | Upstream computed exp(Δ)/σ instead of the paper's exp(Δ/σ), so σ only rescaled the term | yes (bug fix) |
| Ranking loss **summed over pairs**, with **per-cause weights** α<sub>k</sub> (`cause_weights`) | The paper sums over comparable pairs with a weight per cause; the code averaged and fixed α<sub>k</sub> = 1/K | yes |

## 2. Foundation: conditioning on the landmark

A prediction made at day *s* must condition on survival to *s*:

  P(T = t, K = k | T > s, history up to s) = P(T = t, K = k | history) / S(s)

`fit(s=...)` takes one landmark per patient. The likelihood subtracts log *S(s<sub>i</sub>)*. The
ranking term compares patients on remaining time *t − s* and scores each comparator's
conditional CIF *r<sub>i</sub>* days after its own landmark. With `s=None` every function takes
its original path.

A self-review of this change found six problems, each fixed in its own commit:

1. Landmarks were clipped to a wider bin range than event times.
2. `compute_nll` ignored landmarks.
3. The embedded categorical column was still a reconstruction target.
4. The conditioning term's survival floor (1e-10) allowed gradients of order 1e10; it is now 1e-3.
5. Validation loss depended on the batch size; the whole validation set is now scored in one pass.
6. auton-survival was found only through one hard-coded relative path; it can now come from an environment variable, a nearby checkout, or an installed package.

---

## 3. Train the likelihood at every landmark (`all_landmark_nll=True`)

**Problem.** The model was trained with one row per patient: full history,
conditioned on the last visit. Evaluation, though, truncates *every* patient to each
landmark. Short-history inputs were therefore seen in training almost only for
patients whose follow-up really was short, a group with early events. So the model
learned "short history → high risk", and over-predicted at early landmarks.

**Change.** Every (patient, visit) prefix contributes a conditional likelihood term:

  L = (1/K) Σ<sub>i</sub> Σ<sub>v</sub> w<sub>iv</sub> · −log P(T<sub>i</sub>, K<sub>i</sub> | T<sub>i</sub> > s<sub>iv</sub>, X<sub>i,≤v</sub>),  with w<sub>iv</sub> = 1/V<sub>i</sub> by default

The expanded rows are back-propagated in chunks (`landmark_chunk_rows`). This is exact,
since the objective is a weighted sum. Validation uses the same objective. With
*V<sub>i</sub> = 1* the objective reduces to the previous loss, which the tests assert.

Before this, a cheaper variant was tried and kept: `landmark_augment_prob` truncates a
random subset of patients to a random earlier visit each epoch. The deterministic
objective above covers every landmark instead of a sample.

Also in this group: a batch in which no patient has a second visit no longer makes
the reconstruction loss NaN.

## 4. Discrete-time cause-specific hazard head (`hazard=True`, `proportional=True`)

**Problem.** The DeepHit output is one softmax over all (cause, bin) cells, so the
total event probability inside the time grid is forced to 1. Patients who reach the
end of follow-up event-free have nowhere to go, and the model piles probability into
the last bin.

**Change.** `hazard=True` emits per-bin competing-risks hazards: K cause logits plus a
"survive this bin" logit, softmaxed within each bin. These are converted back to the
same (cause, bin) probability interface, so every downstream consumer is unchanged,
but P(no event by the last bin) is now representable. The survive logit's bias is
initialised in closed form so initial survival is not ≈ 0 on long grids.
`proportional=True` additionally factorises logit λ<sub>k</sub>(b) = α<sub>k</sub>(b) + β<sub>k</sub>·h: a per-bin
baseline plus one shared patient score. This makes the risk ordering identical in
every bin. The tests assert that, and also assert it does *not* hold for the free head.

## 5. Re-weighting the all-landmark objective over time (`landmark_decay_days`, `entry_weight`)

* `landmark_decay_days=T` weights each landmark by exp(−day/T), renormalised within
  each patient. It is keyed on **day**, not visit index: visit index mixes up clinical
  timing with how often a patient is seen.
* `entry_weight` scales a patient's *total* contribution. It is applied outside the
  per-patient normalisation, because a per-patient constant folded inside would divide
  straight back out. A test pins that placement.

## 6. Horizon-specific likelihood (`gamma`, `horizon_days`, `horizon_ipcw_clip`)

**Problem.** The bin-level likelihood scores the bin an event fell in. Nothing in it
constrains the *cumulative* risk at the horizons the model is used at. The model
over-predicted the 90-day level even in sample, so this was the objective not asking,
rather than a failure to generalise.

**Change.** For every landmark row and every horizon *h*, add the IPCW-weighted binary
log-likelihood of the observed status at *s + h* under
p = (F<sub>1</sub>(s+h) − F<sub>1</sub>(s)) / S(s).

* Labels and weights follow the evaluation exactly (Graf et al.): a competing event
  before *h* is a true 0, and a row censored before *h* carries no weight.
* The censoring weights are conditioned on the landmark (G(·)/G(s)). Without this,
  the term inflated late landmarks.
* `gamma` is added on top of α and β, so `gamma=0` is bit-identical to the previous
  objective.

## 7. Who counts how much: landmark row weighting (`landmark_row_weighting`)

**Problem.** Under 1/V<sub>i</sub> weighting every patient contributes a total weight of 1, spread
over their visits. But visits stop at the event, so patients with an early event have
few visits, and each of their early rows carries a large weight. On the development
cohort this put the first-visit rows' 90-day event rate well above the true rate, while a
static model trained on first visits only, one row per patient, did not over-predict.

**Change.** Three alternatives to `"patient"` (1/V<sub>i</sub>):

* `"visit"`: every row weighs the same.
* `"given"`: per-row weights from outside the model (`row_weights`), e.g. inverse-intensity weights from a model of the visit process.
* `"band"`: each patient counts **once per landmark-day band** (`landmark_band_edges`). This is how the model is evaluated, with each patient still at risk scored once per landmark. A single band reproduces 1/V<sub>i</sub>, and bands holding one visit each reproduce equal rows; the tests assert both limits.

All three keep the batch's total weight equal to the per-patient scheme's. Only the
distribution changes, which matters because gradients are clipped to a fixed norm.

### How the weighting was chosen

The weighting was settled by a sequence of single-change comparisons on the development
cohort, every arm scored raw (no recalibration):

1. **1/V<sub>i</sub>** (section 3) fixed the last-visit-only training, but over-predicted at
   the first visits.
2. **Decay by landmark day** (section 5) was the deployed arm for a while, and needed a
   post-hoc recalibration to be calibrated.
3. **A static model** trained on first visits only, one row per patient, with the same
   output layer, was calibrated at the first visit. So the over-prediction came from the
   weighting, not from the hazard head.
4. **Equal weight per visit** removed the early over-prediction, as a one-factor change.
5. **Inverse-intensity weights** from a model of the visit process landed on the per-visit
   result: modelling the visit process said what the simple rule already assumed.
6. **Band weighting** weights rows the way the model is evaluated, each patient still at
   risk counted once per landmark. It was the best calibrated across landmarks without any
   recalibration, and replaced decay plus recalibration as the reported configuration.

No weighting changed discrimination; the choice is about calibration. An under-prediction
after the first year was a separate problem that none of the weightings addressed.

The synthetic cohort (`evaluation/make_synthetic.py`) is built with the property behind
step 1: visits stop at the event, so patients with an early event have few visits. It
reproduces the first-visit effect with the published code:

| arm, 90-day risk | O:E at the first visit | O:E at day 90 | AUC at the first visit |
|---|---|---|---|
| 1/V<sub>i</sub> (`patient`) | 0.53 | 0.69 | 0.82 |
| decay, T = 180 d | 0.85 | 0.98 | 0.81 |
| equal per visit (`visit`) | 1.35 | 1.10 | 0.79 |
| band (`band`) | 1.31 | 1.03 | 0.80 |
| static, first visit only | 1.21 | 1.11 | 0.82 |

O:E is observed (Aalen-Johansen) over mean predicted risk; below 1 is over-prediction.
1/V<sub>i</sub> over-predicts about twofold at the first visit; giving each patient equal
weight there removes it (on this small cohort the correction overshoots a little), and AUC
barely moves. The simulation is not built to mimic the real cohort beyond the first visits,
where every arm over-predicts, so read it for the first-visit contrast only. Means over three
splits, with the proportional hazard head (see [`evaluation/`](../evaluation/README.md));
reproduce with:

```sh
python -m evaluation.make_synthetic --out $DDH_DATA_ROOT
export DDH_COLUMNS=$DDH_DATA_ROOT/columns.json
COMMON="--splits 0 1 2 --iters 40 --alpha 0 --hazard --proportional"
python -m evaluation.train --arm patient $COMMON --all-landmark-nll
python -m evaluation.train --arm visit   $COMMON --all-landmark-nll --landmark-row-weighting visit
python -m evaluation.train --arm band    $COMMON --all-landmark-nll --landmark-row-weighting band
python -m evaluation.train --arm decay   $COMMON --all-landmark-nll --landmark-decay-days 180
python -m evaluation.train --arm static  $COMMON --first-visit-only
python -m evaluation.evaluate --arm band --arm "1/V_i=patient" --arm "per visit=visit" \
    --arm decay --arm "static=static" --splits 0 1 2 --stage landmarks
```

**Configuration the project reports:** `hazard=True`, `all_landmark_nll=True`,
`alpha=0` (no ranking term), `landmark_row_weighting="band"`. See
[`evaluation/`](../evaluation/README.md) for how it is trained and evaluated.

---

## Tried and not adopted

One change was tried and not merged; it is kept on its own branch.

| Branch | Idea | Why it was not adopted |
|---|---|---|
| `experiments/all-landmark-rank` | Move the ranking term onto the same (patient, visit) rows as the all-landmark likelihood, pairs weighted *w<sub>i</sub>w<sub>j</sub>*, within-patient pairs dropped | A three-split pilot against the all-landmark likelihood alone showed no discrimination gain (the interval included zero and was narrow next to the gain the all-landmark likelihood itself had bought). Calibration moved slightly the wrong way, and each epoch took 50% longer. Stopped after the pilot. |
