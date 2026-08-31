# Research Report: exp-0004-independent-seed-replication

## 1. Hypothesis

`exp-0003` found `neuro_model` underperforming a capacity-matched MLP and
even logistic regression, consistently across 4 splits — but those 4
splits were windows of *one* generated market history. This experiment
asks the question `exp-0003` itself flagged as open: does that pattern
hold up across **independently-generated** markets, or was it an artifact
of one seed's particular history?

## 2. Dataset

Five independently-seeded synthetic markets (seeds 42, 123, 7, 2024, 999),
each 30 assets / 1100 days, otherwise identical
`SyntheticMarketConfig::default()`. `exp-0003`'s single market (seed 42) is
one of the five, so this experiment is a superset that includes the
original run.

## 3–6. Information cutoff, features, graph construction, model architecture

Unchanged from `exp-0003`: same four point-in-time-safe features, same
per-split standardization, same three trained models
(`NeuroTopologicalFinancialModel`, capacity-matched `MlpBaseline`,
`LogisticRegressionBaseline`) plus the two fit-only baselines.

## 7–9. Training procedure, validation, leakage controls

Unchanged from `exp-0003`: Adam, lr=0.02, 60 epochs, batch size 8;
`WalkForwardValidator` (`Expanding`, 500/100/5/100 days) applied
independently within each seed's market (4 splits per seed, 20
(seed, split) combinations total). `tensor_engine::seed(seed)` is called
once per market seed, before both market generation and model
initialization for that seed — see Limitations for what this
does and doesn't guarantee given the Milestone 6 Burn RNG caveat.

## 10. Results

**Overall mean, across all 20 (seed, split) combinations:**

| Model | Mean accuracy |
|---|---|
| `naive_persistence` | 0.9720 |
| `neuro_model` | **0.4320** |
| `mlp_baseline` | 0.4150 |
| `logistic_regression` | 0.4080 |
| `majority_class` | 0.2725 |

**Per-seed means (4 splits each):**

| Seed | `naive_persistence` | `neuro_model` | `mlp_baseline` | `logistic_regression` | `neuro_model` beat both flat baselines? |
|---|---|---|---|---|---|
| 42 | 0.963 | 0.355 | 0.395 | 0.378 | No |
| 123 | 0.978 | 0.350 | 0.330 | 0.295 | Yes |
| 7 | 0.980 | 0.388 | 0.413 | 0.482 | No |
| 2024 | 0.975 | 0.497 | 0.402 | 0.358 | Yes |
| 999 | 0.965 | 0.570 | 0.535 | 0.527 | Yes |

`neuro_model` beat *both* flat baselines (on that seed's mean) in **3 of 5
seeds**. Full per-split numbers: `metrics.json`.

## 11. Statistical analysis

Still no formal significance test (no confidence intervals, no paired
test across seeds) — flagged as a limitation, not performed here. What
can be said without one: `exp-0003`'s seed (42) is, on this larger sample,
one of the two seeds where `neuro_model` did *not* beat both flat
baselines — i.e., `exp-0003` happened to sample from the *less*
favorable-to-`neuro_model` side of the seed distribution seen here, not a
representative one. The per-seed spread is large (`neuro_model` ranges
0.350–0.570 across seeds; `naive_persistence`, notably, is far more
stable at 0.963–0.980), which on its own says this task is noisy enough at
this data scale that single-seed conclusions — including `exp-0003`'s —
should not have been trusted as strongly as they were.

## 12–14. Ablation, topology, backtest

Not performed.

## 15. Limitations

- **`exp-0003`'s specific conclusion — "neuro_model consistently
  underperforms a capacity-matched flat model" — does not replicate.**
  Across 5 independent seeds, `neuro_model` has the *highest* overall mean
  accuracy (0.432) of the three trained models, and wins on 3 of 5 seeds
  individually. This is a genuine correction to `exp-0003`'s conclusion,
  not a minor caveat to it.
- That said, **a 3-of-5 win rate with this much per-seed variance is weak
  evidence for "topology helps," too.** The honest reading is: this
  experiment does not support either "topology reliably hurts" (exp-0003's
  claim) or "topology reliably helps" — the result is noisy and
  seed-dependent at this sample size (5 seeds, 20 data points per model).
  More seeds and a proper statistical test are needed before either
  direction is defensible.
- The Milestone 6 Burn RNG-determinism caveat applies here more than in
  prior experiments: `tensor_engine::seed(seed)` was called once per
  market seed, intending to also vary model initialization by seed, but
  (per that caveat) weight initialization is not guaranteed
  bit-reproducible under normal multi-threaded execution. This doesn't
  invalidate the results (each seed still produced *some* well-defined,
  reported outcome), but it does mean "seed 999" isn't a fully controlled,
  exactly-reproducible experimental unit the way the market generation
  itself is — re-running this exact script may not reproduce identical
  numbers, only the same qualitative pattern (which is itself worth
  checking, and hasn't been, yet).
- `naive_persistence` remains the dominant model by a wide margin in every
  seed (0.963–0.980) — this part of every prior report's conclusion is
  robust and unchanged.

## 16. Conclusions

**This project's honest V0.1-stage answer to "does dynamic topology
improve prediction?" (§54, question 1) is: inconclusive, with high
seed-to-seed variance, on same-day regime classification with this
minimal feature set and training budget.** `exp-0003`'s apparently clean
negative result was an artifact of evaluating one market history; with
independent replication, `neuro_model` slightly *edges out* the flat
baselines on average and wins more often than it loses, but not by a
margin or a consistency that supports a confident claim either way.

This is itself a useful, disclosed finding — and a methodological lesson
this project is recording plainly: **`exp-0003` should not have been
treated as a settled result on the strength of one market history's 4
splits**, and this report exists specifically to correct that. Concrete
next steps: more seeds (10–20, not 5) with a proper paired statistical
test; and, per every prior report's discussion, a temporal component
(§21) before concluding anything about whether *dynamic* topology
specifically (as opposed to a same-day snapshot of it) has value — a
same-day classifier has no way to benefit from the "dynamic" part of
"dynamic topology" at all, which may be the more fundamental reason none
of these experiments have found a strong, stable signal in either
direction.
