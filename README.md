# Dynamic-DeepHit for landmark prediction

A fork of [Jeanselme/DynamicDeepHit](https://github.com/Jeanselme/DynamicDeepHit), the PyTorch
implementation of [Dynamic-DeepHit](https://ieeexplore.ieee.org/document/8681104) (Lee et al.,
IEEE TBME 2020; original TensorFlow code at
[chl8856/Dynamic-DeepHit](https://github.com/chl8856/Dynamic-DeepHit)).

The fork adapts the model to **landmark prediction from irregular clinic visits with a
competing risk**. At any visit, it predicts the risk of the event of interest within
*h* days, given everything seen so far and given that the patient is still event-free.

## What changed

Each row is one or a few commits; [`docs/modifications.md`](docs/modifications.md) gives
the problem, the change and how it was tested.

**Foundation.** These change upstream defaults where the upstream behaviour was a bug.

| change | where |
|---|---|
| Port to auton-survival (the maintained successor of DeepSurvivalMachines) | `ddh/dsm_compat.py` |
| Discretisation index fix: the earliest events were scored in the last bin | `discretize` |
| Training loop: per-epoch shuffle, full validation, LR schedule, gradient clipping, seeding | `utils.train_ddh`, `fit` |
| Ranking loss as in the paper: σ inside the exponential, summed over pairs, per-cause weights | `losses.ranking_loss` |
| Likelihood and ranking conditioned on survival to a landmark, plus six fixes from a self-review | `fit(s=...)`, `losses` |

**Modifications.** Every one is opt-in; with default arguments, predictions are
bit-identical to the foundation.

| change | switch | problem it addresses |
|---|---|---|
| Likelihood at every landmark | `all_landmark_nll=True` | trained on last visits only, evaluated at every landmark |
| Discrete-time cause-specific hazard head | `hazard=True`, `proportional=True` | the softmax head must spend all probability inside the time grid |
| Landmark-day decay and between-patient entry weights | `landmark_decay_days`, `entry_weight` | where in follow-up the objective puts its weight |
| Horizon-specific likelihood | `gamma`, `horizon_days` | nothing in the bin-level likelihood constrains cumulative risk at *h* |
| Landmark row weighting | `landmark_row_weighting="visit"/"given"/"band"` | patients with early events have few visits, so 1/V<sub>i</sub> weighting over-weights them early |

The configuration the project reports is `hazard=True, all_landmark_nll=True, alpha=0,
landmark_row_weighting="band"`. One experiment that was tried and not adopted (the ranking
term at every landmark) is kept on the `experiments/all-landmark-rank` branch;
[`docs/modifications.md`](docs/modifications.md#tried-and-not-adopted) says why it was dropped.

## Install

```sh
git clone https://github.com/dadudu13/DynamicDeepHit && cd DynamicDeepHit
pip install torch
# auton-survival 0.1.0; --no-deps because its release metadata pins torch<2.0
pip install --no-deps "auton-survival @ git+https://github.com/autonlab/auton-survival@5dde465f7223601717abddc1d075e837707c403b"
pip install -e ".[test,eval]"
python -m pytest
```

Tested locally with Python 3.11 and torch 2.11 and 2.14; CI runs Python 3.11 and 3.12.

## Use

```python
import numpy as np
from ddh import DynamicDeepHit

# x: per-patient arrays (visits x features); t, e: per-visit outcome time (days from the
# first visit) and event (0 censored, 1 event of interest, 2 competing);
# visit_days: per-patient arrays of each visit's day; s: each patient's last visit day.
model = DynamicDeepHit(split=50, hidden_rnn=32, typ="GRU", alpha=0, beta=0.75, hazard=True)
model.fit(x, t, e, iters=50, s=s, visit_days=visit_days,
          all_landmark_nll=True, landmark_row_weighting="band",
          landmark_band_edges=[0, 30, 90, 180, 365, 730])
cif_event = 1 - model.predict_survival(x, [90.0, 180.0], risk=1)   # unconditional, from day 0
```

To get risks conditioned on a landmark, and the full train / external-cohort /
evaluation workflow, see [`evaluation/`](evaluation/README.md). It runs end to end on
the synthetic cohort it generates.

## How the history is organised

* One change per commit, each explaining the problem it solves. Every commit passes
  the tests that exist at that point.
* Upstream's history is kept intact underneath. The work is merged through one pull
  request per theme: foundation, landmark training, hazard head, landmark-time
  weighting, horizon term, row weighting, packaging, evaluation, documentation.
* Experiments that were not adopted stay on their own branches.

## Credit and license

The original PyTorch implementation is by Vincent Jeanselme (MIT); the
modifications are by Dulin Wang, under the same license ([LICENSE](LICENSE)).
