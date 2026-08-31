# Research Report: exp-0003-capacity-matched-multi-split

## 1. Hypothesis

`exp-0002` found `neuro_model` reached a lower training loss than
`logistic_regression` but a *higher* test error — consistent with
overfitting, but unable to distinguish "the topology mechanism overfits"
from "extra capacity overfits regardless of what it's for." This
experiment adds a **capacity-matched `MlpBaseline`** (no graph, same
parameter count as `neuro_model`) to separate those two explanations, and
evaluates across **every** walk-forward split from a longer synthetic run
instead of just the first, so one split's idiosyncrasies can't drive the
conclusion.

## 2. Dataset

Same generator/seed (42) and 30-asset universe as `exp-0001`/`exp-0002`,
extended to 1100 days specifically to produce multiple non-overlapping
walk-forward splits.

## 3–5. Information cutoff, features, graph construction

Unchanged from `exp-0002`: point-in-time-safe features via
`PointInTimeDataset::as_of`, standardizers fit per-split on that split's
train window only (never reused across splits — each split gets its own
fit, which is the correct point-in-time-safe behavior for a genuinely
rolling evaluation), learned-topology-only graph for `neuro_model`.

## 6. Model architecture

Four trained/fit models, evaluated identically per split:
`NeuroTopologicalFinancialModel` (~931 parameters at
`feature_dim=4, embed_dim=16, topology_proj_dim=8` — see the exact
breakdown in `crates/evaluation/examples/third_experiment.rs`),
`MlpBaseline` (one hidden layer, `hidden_dim=116`, verified via
`MlpBaseline::param_count(4, 116, 3) == 931` — an exact parameter-count
match), `LogisticRegressionBaseline` (51 parameters — much smaller,
kept as the "how much does capacity alone buy you" reference point),
`MajorityClassBaseline`, `NaivePersistenceBaseline`.

## 7. Training procedure

Identical to `exp-0002` for every trained model: Adam, lr=0.02, 60 epochs,
batch size 8 — trained independently per split (a fresh model per split,
never reused across splits).

## 8–9. Validation methodology, leakage controls

`WalkForwardValidator`, `Expanding` mode, same train/validation/embargo/test
lengths as `exp-0002` (500/100/5/100 days). 1100 days of data produced
**4 splits**; all 4 were evaluated (`validator.splits()` returns every
split; `exp-0002` had only used `splits[0]`). Each split's standardizer is
fit on that split's own train window only.

## 10. Results

| Model | Split 0 | Split 1 | Split 2 | Split 3 | **Mean** |
|---|---|---|---|---|---|
| `naive_persistence` | 0.98 | 0.95 | 0.98 | 0.94 | **0.9625** |
| `mlp_baseline` (capacity-matched) | 0.28 | 0.36 | 0.48 | 0.46 | **0.3950** |
| `logistic_regression` (51 params) | 0.34 | 0.32 | 0.46 | 0.39 | **0.3775** |
| `neuro_model` (~931 params, graph) | 0.30 | 0.29 | 0.48 | 0.35 | **0.3550** |
| `majority_class` | 0.00 | 0.00 | 0.49 | 0.04 | **0.1325** |

Full per-split numbers: `metrics.json`.

## 11. Statistical analysis

4 splits, 100 test examples each, still one underlying market seed (the
splits are non-overlapping time windows from the *same* generated
history, not independent re-generations) — this is more evidence than
`exp-0002`'s single split, but still not enough for a formal significance
test, and does not yet vary the random seed. `neuro_model` was the
worst-or-tied-worst of the three trained models in 3 of 4 splits (0, 1,
3), tied for best in split 2. That's a repeated pattern, not a single
lucky/unlucky draw, but it is still one market realization.

## 12. Ablation analysis

This experiment *is* a targeted ablation: `neuro_model` vs. `mlp_baseline`
isolates the graph/topology machinery specifically (both have ~931
parameters; only one has a graph). A full ablation ladder (§33: static
graph only, learned topology only, topology + memory, etc.) remains future
work — this is one rung of it.

## 13–14. Topology analysis, backtest

Not performed — out of scope for this experiment.

## 15. Limitations

- **`mlp_baseline` (same parameter count, no graph) outperforms
  `neuro_model` on average (0.395 vs. 0.355) across all 4 splits, and
  even the much smaller `logistic_regression` (51 params) edges out
  `neuro_model` too (0.3775 vs. 0.355).** This directly answers this
  experiment's question: the gap between `neuro_model` and the flat
  baselines in `exp-0002` was **not** simply "more capacity overfits" —
  a same-capacity flat model does *better*, not worse, than
  `neuro_model` here. That points at the graph/topology-specific
  machinery (the learned scorer, mutual top-k selection, message passing)
  as the source of the underperformance in this setup, not raw parameter
  count.
- All four trained/fit models remain dramatically behind
  `naive_persistence` (0.96 mean) — see `exp-0002`'s discussion of why a
  per-day classifier with no persistence prior is structurally
  disadvantaged against a baseline that directly encodes "assume no
  change" in a highly autocorrelated process.
- Splits are drawn from one seed's market history — not yet independent
  re-generations of the synthetic market. A finding that holds across
  seeds would be considerably stronger than one that holds across splits
  of a single history.
- Every trained model here uses the *same* four simple, unnormalized-until-
  z-scored features and the *same* training budget — this experiment
  cannot rule out that a different feature set, more training, or
  different hyperparameters would change the ranking. It specifically
  tests "does the graph help, holding capacity and everything else fixed,"
  and the answer on this data is "no, and it may actively hurt."

## 16. Conclusions

**With capacity controlled for, the dynamic-topology architecture
underperformed a plain MLP of equal size, and underperformed logistic
regression, on this synthetic market across 4 walk-forward splits.** This
is a more specific and more damaging (for the central hypothesis) finding
than `exp-0002`'s: it is not merely "the trained models both did poorly,"
it is "given equal capacity, the graph/topology structure did *worse* than
no graph at all." On the evidence gathered so far in this project, dynamic
topology has not demonstrated the incremental predictive value it set out
to test (§54, question 1) — the honest, current answer is **no evidence of
benefit, and some evidence of harm**, on this particular synthetic setup,
feature set, and training budget.

This does not close the question — the architecture could plausibly
benefit from richer features (the current four are minimal), a temporal
component (§21, not yet built — same-day classification with no memory
may be a genuinely poor fit for a graph/topology mechanism whose value
proposition is arguably about *how relationships evolve over time*, not
about one day's snapshot), or simply more training data/epochs. But
absent those, this project's own evidence currently argues against the
core hypothesis, and that is reported here as-is.
