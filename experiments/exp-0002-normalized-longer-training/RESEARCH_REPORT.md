# Research Report: exp-0002-normalized-longer-training

## 1. Hypothesis

Same as `exp-0001`: does `NeuroTopologicalFinancialModel` (learned dynamic
topology + graph message passing) outperform a flat logistic-regression
baseline at same-day market regime classification? This run's added
purpose: does fixing `exp-0001`'s identified under-training (unnormalized
features, too few epochs) change the answer?

## 2–5. Dataset, information cutoff, features, graph construction

Identical to `exp-0001` — same seed (42), same 30-asset/900-day synthetic
market, same four features, same point-in-time-safety, same learned-topology-
only graph. The only *addition*: each feature is z-score standardized
([`feature_engine::Standardizer`]) using parameters fit on the train
window only, then applied unchanged to the test window — no leakage from
test into the fitted mean/scale.

## 6. Model architecture

Identical to `exp-0001`: `NeuroTopologicalFinancialModel` vs.
`LogisticRegressionBaseline` vs. `MajorityClassBaseline` vs.
`NaivePersistenceBaseline`.

## 7. Training procedure

Adam, **learning rate 0.02** (up from 0.01), **60 epochs** (up from 15),
batch size 8. Same walk-forward split as `exp-0001` (train 2015-01-02 to
2016-05-16, test 2016-09-03 to 2016-12-12).

## 8–9. Validation methodology, leakage controls

Identical to `exp-0001`, plus: standardizers fit on train-window raw
feature values only (pooled across all assets and all train days),
verified by construction (`fit_standardizers` only ever receives
`raw_train`, never `raw_test`).

## 10. Results

| Model | Test-window accuracy | Final train loss |
|---|---|---|
| `naive_persistence` | **0.98** | — (not a trained model) |
| `logistic_regression` | 0.34 | 0.734 |
| `neuro_model` | 0.30 | 0.619 |
| `majority_class` | 0.00 | — |

Reference: `ln(3) ≈ 1.099` is the loss of a uniform random guess —
`exp-0001`'s trainable models plateaued at 1.10–1.12 (no real learning);
here both dropped well below it, confirming real learning occurred this
time. Full numbers: `metrics.json`.

## 11. Statistical analysis

None — still a single walk-forward split, single seed. 100 test examples
is also a small sample for a 3-class accuracy comparison (a 4-point
accuracy gap on 100 examples is not statistically distinguishable without
a proper test, which hasn't been run). This is flagged as a limitation,
not glossed over.

## 12–14. Ablation, topology, backtest

Not performed — same reasons as `exp-0001`.

## 15. Limitations

- Same single-split, single-seed, no-hyperparameter-search caveats as
  `exp-0001`.
- `neuro_model` reaches a **lower training loss** than
  `logistic_regression` (0.619 vs. 0.734) — it fits the *training* data
  better, consistent with its strictly larger capacity (feature encoder +
  message passing + topology scorer vs. one linear layer) — but scores
  **lower test accuracy** (0.30 vs. 0.34). That combination (better train
  fit, worse test performance) is the textbook signature of **overfitting
  relative to the baseline**, not of the graph/topology mechanism adding
  useful inductive bias. This run does not distinguish "the topology
  mechanism itself overfits" from "the extra capacity overfits regardless
  of what that capacity is for" — a capacity-matched baseline (an MLP with
  a comparable parameter count but no graph) would be needed to tell those
  apart, and doesn't exist yet.
- Both trained models are dramatically behind `naive_persistence` (0.98).
  This synthetic market's regimes are highly autocorrelated day to day (by
  construction — see `data_engine::synthetic::regime::RegimeTransition`'s
  persistence parameter), and neither trained model was given persistence
  as an inductive bias (both predict each day independently from that
  day's features, with no memory of yesterday's prediction or label) — a
  fair per-day classifier is at a structural disadvantage against a
  baseline that directly encodes "assume no change" in a persistent
  process. This isn't a flaw in the comparison so much as a reminder that
  regime *classification* accuracy alone, without a persistence-aware
  baseline or a temporal component in the model (§21, not yet built), is
  an easy metric for a trivial baseline to win.

## 16. Conclusions

**Fixing the under-training issue changed the result from "uninformative"
(exp-0001: both models near-uniform) to "informative but negative"
(exp-0002): once both models actually learn, the graph/topology-augmented
model does not outperform the flat baseline on held-out data in this run —
it slightly underperforms it, while fitting the training data better,
consistent with overfitting rather than useful inductive bias from the
learned topology.** This is a real, disclosed result for project spec
§54's question 1 ("does the graph improve prediction?") on this specific
single split — not yet a general answer (needs more seeds, more splits, a
capacity-matched baseline, and ideally a temporal component before it's
conclusive), but it is evidence, and it points the same direction the
scientific-honesty rule requires: report it, don't discard it because it's
not the hoped-for outcome.

Concrete next steps: (1) a capacity-matched MLP baseline (same parameter
budget as `neuro_model`, no graph) to separate "extra capacity overfits"
from "the topology mechanism specifically overfits"; (2) multiple
walk-forward splits and seeds before treating any accuracy gap as more
than anecdotal; (3) a persistence-aware model or a temporal component,
since `naive_persistence`'s dominance here says at least as much about
this task's structure as about either trained model's quality.
