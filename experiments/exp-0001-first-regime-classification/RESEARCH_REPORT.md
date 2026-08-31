# Research Report: exp-0001-first-regime-classification

## 1. Hypothesis

Does `NeuroTopologicalFinancialModel` (feature encoder → learned dynamic
topology → sparse graph message passing → regime classification head)
outperform a flat logistic-regression baseline and the simplest persistence
baseline at same-day market regime classification (`RiskOn` / `Neutral` /
`RiskOff`)?

This is the first experiment run in this project. Its purpose is **not**
to produce a headline result — it is to prove the full pipeline (synthetic
data → point-in-time-safe features → walk-forward split → training →
baseline comparison → written report) actually works end to end, and to
establish an honest, reproducible starting point.

## 2. Dataset

`data-engine`'s synthetic generator (seed 42): 30 assets, 10 sectors, 900
trading days starting 2015-01-02. See `docs/architecture` /
`crates/data-engine/src/synthetic/generator.rs` for the generative model
(GARCH volatility clustering, regime-dependent hot-sector factors,
cross-sector contagion in `RiskOff`).

## 3. Information cutoff

Every feature for day `t` is computed from `PointInTimeDataset::as_of(t)`
— bars strictly up to and including day `t`, nothing later. This is a
same-day nowcast (predict today's regime from data available today), not a
forecast; there is no leakage in either direction because the target label
(the generator's own regime state for day `t`) is not derived from any
future price data either.

## 4. Features

Four hand-picked features per asset, from `feature-engine`:
`last_return` (1-day log return), `rolling_volatility_20` (20-day, min 5
periods), `momentum_20` (20-day cumulative log return), `sma_deviation_20`
(deviation of close from its 20-day moving average). Deliberately narrow —
this experiment tests the pipeline, not feature engineering.

## 5. Graph construction

`neuro-model`'s learned topology only (`TopologyScorer` → mutual top-k,
`k=5`); no static sector/correlation graph is used as an input here (that
comparison — static vs. learned topology — is future work, §54 question
2/3, not this experiment).

## 6. Model architecture

`NeuroTopologicalFinancialModel(feature_dim=4, embed_dim=16,
topology_proj_dim=8)` vs. `LogisticRegressionBaseline(feature_dim=4,
num_classes=3)` (mean-pool → linear → softmax, no graph) vs.
`MajorityClassBaseline` vs. `NaivePersistenceBaseline`.

## 7. Training procedure

Adam, learning rate 0.01, batch size 8 (gradient-accumulation
mini-batching per `training-engine`'s design), 15 epochs, single walk-forward
split (no early stopping used this run — see Limitations).

## 8. Validation methodology

`WalkForwardValidator`, `Expanding` mode: train 500 days (2015-01-02 to
2016-05-16), embargo 5 days, validation 100 days (unused this run — see
Limitations), embargo 5 days, test 100 days (2016-09-03 to 2016-12-12).
475 train examples, 100 test examples (after a 25-day feature warmup).

## 9. Leakage controls

Point-in-time feature construction (§3 above); walk-forward chronological
split with embargo, never a random split; `MajorityClassBaseline` and the
trained models are fit only on the train window's labels/data.

## 10. Results

| Model | Test-window accuracy |
|---|---|
| `naive_persistence` | **0.98** |
| `neuro_model` | 0.00 |
| `logistic_regression` | 0.00 |
| `majority_class` | 0.00 |

Full numbers: `metrics.json` in this directory.

## 11. Statistical analysis

None performed — a single walk-forward split and a single seed give no
basis for a significance test. This is explicitly flagged as a limitation,
not glossed over.

## 12. Ablation analysis

Not applicable yet (§33 needs multiple architecture variants; only one
non-baseline architecture exists so far).

## 13. Topology analysis

Not performed this run — out of scope for a first pipeline-validation
experiment (§34 is its own future milestone).

## 14. Backtest

Not performed — no `backtester` crate exists yet.

## 15. Limitations

- **Training loss plateaued at ≈1.10–1.12, barely below `ln(3)≈1.0986`**
  (the loss of outputting a uniform 1/3-1/3-1/3 guess every time) for
  *both* trainable models. Neither model learned meaningfully
  discriminative regime signal in this run — 15 epochs at this learning
  rate, with 4 simple hand-picked features and no per-feature
  normalization, was not enough for either the flat logistic baseline or
  the topological model to move far from a near-constant prediction.
- Given that, the `0.00` accuracy for `neuro_model`, `logistic_regression`,
  and `majority_class` should be read as **"each settled on an
  (effectively arbitrary) constant class that didn't match this
  particular 100-day test window's actual (also persistent) regime,"** not
  as "the model actively learned the wrong thing." `naive_persistence`'s
  0.98 is a real, structural advantage from directly echoing yesterday's
  true label in a domain (this synthetic generator, and real financial
  regimes) where regimes are highly autocorrelated day to day — it is
  expected to be a strong baseline, and it is.
- Single walk-forward split, single seed, no early stopping, no
  hyperparameter search, no feature normalization. Every one of these is a
  plausible reason the trainable models underperformed, and none has been
  ruled out — this report does not claim the topological architecture is
  incapable of learning regime signal, only that it did not in this
  specific, deliberately minimal first run.
- Features were not normalized/standardized before feeding either trainable
  model — likely a real contributor to the near-uniform-output collapse
  (unnormalized feature scales can make gradient-based training slow to
  move away from the loss-minimizing constant-output starting point).

## 16. Conclusions

The pipeline works end to end: synthetic data, point-in-time-safe feature
construction, walk-forward splitting, training (with the Milestone 8
gradient-tracking bug now fixed), baseline comparison, and a written report
all ran successfully in one command. **The scientific question this
project exists to answer — does dynamic topology add value — is not yet
answered by this run**: neither the topological model nor its flat
baseline learned enough to be distinguishable from a majority-class guess
here, so this experiment cannot yet say whether the graph/topology
machinery helps, hurts, or is neutral. The honest, useful finding from
`exp-0001` is methodological: **more epochs, feature normalization, and
likely a higher learning rate or longer schedule are needed before this
comparison is meaningful** — that is the concrete next step, not
re-running with different assumptions until a positive result appears.
