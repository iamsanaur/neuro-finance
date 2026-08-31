# Project Status

Last updated: 2026-08-31 (Milestone 11)

## Current milestone

**Milestone 11: capacity-matched MLP baseline + multi-split evaluation (exp-0003) — DONE**

## exp-0003 result: with capacity controlled for, the graph model underperforms a plain MLP, across 4 splits

Added `MlpBaseline` (one hidden layer, no graph, `hidden_dim=116` chosen so
`param_count(4, 116, 3) == 931`, an exact match to `neuro_model`'s ~931
parameters at this configuration — verified numerically via
`MlpBaseline::param_count`, not eyeballed). Extended the synthetic run to
1100 days to get **4** non-overlapping walk-forward splits, and evaluated
every model on all 4 (`exp-0002` had only used the first).

**Mean test accuracy across 4 splits: `naive_persistence` 0.9625,
`mlp_baseline` 0.3950, `logistic_regression` 0.3775, `neuro_model` 0.3550,
`majority_class` 0.1325.** `neuro_model` was worst-or-tied-worst of the
three trained models in 3 of 4 splits. This directly answers `exp-0002`'s
open question: the gap wasn't just "more capacity overfits" — a
**same-capacity flat MLP does better, not worse**, than `neuro_model`
here, which points at the graph/topology machinery specifically (not raw
parameter count) as the source of the underperformance in this setup.
Full writeup, including everything this does and doesn't establish (one
seed's market history across 4 splits, not yet independent
re-generations; no temporal component in the model yet, which the report
flags as a plausible reason a same-day topology snapshot underdelivers):
`experiments/exp-0003-capacity-matched-multi-split/RESEARCH_REPORT.md`.

This is now the most specific and most informative result the project has
produced: **no evidence of benefit, and some evidence of harm, from
dynamic topology on this synthetic setup** — reported plainly, per the
project's own rule against manufacturing positive results.

## Prior: exp-0002 (superseded by exp-0003, kept for the record)

Added `feature_engine::Standardizer` (z-score, fit on train window only —
point-in-time-safe by construction) and re-ran with 60 epochs at lr=0.02
(up from 15 epochs at lr=0.01). This time training loss actually dropped
(neuro-model: 1.24 → 0.62; logistic-regression: 1.21 → 0.73 — both well
below the `ln(3)≈1.10` uniform-guess floor `exp-0001` was stuck at),
confirming real learning happened.

**Test-window accuracy: `naive_persistence` 0.98, `logistic_regression`
0.34, `neuro_model` 0.30, `majority_class` 0.00.** The graph/topology
model reaches a *lower* training loss than the flat baseline (0.62 vs.
0.73 — it fits training data better, as expected from its larger
capacity) but *lower* test accuracy (0.30 vs. 0.34) — the textbook
signature of overfitting relative to the baseline, not of the topology
mechanism adding useful signal. Full writeup, including what this result
does and doesn't establish (single split, no capacity-matched baseline
yet, naive-persistence's dominance says as much about the task as about
either model): `experiments/exp-0002-normalized-longer-training/RESEARCH_REPORT.md`.

This is the first experiment in this project that actually bears on its
central question — and the honest answer so far, on this one split, is
"no measurable benefit from the topology, and a sign of overfitting."
Reported as-is, per the project's own rule against manufacturing positive
results.

## Prior: exp-0001 (superseded by exp-0002, kept for the record)

Ran `neuro-model` vs. a flat logistic-regression baseline vs.
majority-class vs. naive-persistence on one walk-forward split of synthetic
data (30 assets, 900 days; train 500d / embargo 5d / validation 100d
[unused] / embargo 5d / test 100d). Full report:
`experiments/exp-0001-first-regime-classification/RESEARCH_REPORT.md`.

**Result: `naive_persistence` scored 0.98 test accuracy; `neuro_model`,
`logistic_regression`, and `majority_class` all scored 0.00.** Training
loss for both trainable models plateaued at ≈1.10–1.12 — barely below
`ln(3)≈1.0986`, the loss of a uniform random guess — meaning neither model
learned meaningfully discriminative regime signal in this run (15 epochs,
lr=0.01, 4 unnormalized hand-picked features, single split). This is **not**
evidence the topological architecture doesn't work; it's evidence this
particular quick first run under-trained both trainable models equally.
The scientific question this project exists to answer (does dynamic
topology add value over a flat baseline?) remains open — both models
failed to learn enough to be distinguishable from each other here.
Per the project's own rule ("never manufacture positive results"), this is
reported as-is rather than re-run with different assumptions until a
positive-looking number appears. Concrete next steps identified: feature
normalization, more epochs/a longer schedule, and only then a real
architecture comparison.

## Major finding this milestone: a real gradient-tracking bug, found and fixed

While building the training loop, the model trained on ten epochs of a
repeated example and the loss was **bit-for-bit identical** before and
after — not "barely decreasing," literally unchanged to 17 significant
digits. That's impossible from slow convergence; it meant zero gradient
was reaching any parameter.

Root cause, isolated with a series of minimal reproductions (see
`crates/training-engine/src/train.rs` git history for the debug tests used,
removed once the cause was confirmed): **every `#[derive(Module)]` struct in
`topology-engine` and `neuro-model` had its backend baked in as the
concrete `tensor_engine::Backend` type alias** (e.g. `Linear<Backend>`
directly in a field), rather than being generic over a type parameter
`B: Backend`. That compiles fine and forward passes work correctly — the
bug only shows up when you actually try to train: `GradientsParams::from_grads`
silently returns **zero entries** for such a struct, because Burn's
`#[derive(Module)]` macro needs a real generic type parameter in scope to
generate a working parameter-registration path. A bare `burn::nn::Linear<Backend>`
used directly (not wrapped in a custom struct) trains fine; the moment
it's wrapped in even a trivial one-field custom struct with a concrete
backend, training silently becomes a no-op.

**Fix**: every `Module`-deriving struct across `topology-engine`
(`TopologyScorer`) and `neuro-model` (`FeatureEncoder`, `GraphMessagePassing`,
`RegimeHead`, `NeuroTopologicalFinancialModel`) is now generic over
`<B: burn::tensor::backend::Backend>`, taking a `device: &B::Device`
constructor argument instead of calling `tensor_engine::device()`
internally. `tensor_engine::Backend` is still the one concrete type this
workspace actually uses — it's now supplied as the type argument at call
sites (`TopologyScorer<tensor_engine::Backend>`), not baked into the
struct definitions. All 134 tests (including the previously-broken training
test) pass with this fix. Documented at the top of `topology-engine::scorer`
(the first struct that needed it) and cross-referenced from every other
struct that needed the same fix, so the next crate that adds a `Module`
struct doesn't repeat the mistake.

Also required raising `burn`'s feature set to include `"std"` (needed for
`burn::record`'s file-based recorders, used by checkpointing) — omitted
originally since the minimal `ndarray + autodiff` feature set from
Milestone 6 didn't need it.

## Decision: real data source

User asked about pulling data from Yahoo Finance. Agreed sequencing:
finish V0.1 on synthetic data first (so the pipeline is validated against
known ground truth end to end), then add a Yahoo Finance
`MarketDataProvider` adapter as the **first step of V0.2** (§51). Also noted
for that future step: Yahoo covers price/volume well but not
fundamentals/macro with proper point-in-time revision history — FRED/ALFRED
(free, revision-vintage-aware) is the better source for the macro/fundamental
providers V0.2 also calls for. Not acted on yet — flagged here so it isn't
lost before V0.2 starts.

## Scope reminder

V0.1 (per project spec §50) is: 100 synthetic assets, 10 sectors, daily data,
30-day sequences, basic causal features, sector graph, correlation graph,
baseline Transformer, static graph model, dynamic topology model, regime
classification, walk-forward evaluation, basic backtest, tests, CLI, docs.
Nothing beyond that scope until V0.1 is scientifically validated.

## Completed

- **Environment inspected and documented** — `docs/environment.md`:
  Apple M4 Mac mini, 16GB RAM, Metal 4 GPU, **no CUDA**, ~648GB free on the
  project volume. Rust was not installed; installed via rustup
  (rustc/cargo 1.98.0). `~/.cargo/bin` is not yet on the default shell PATH
  (root-owned `~/.zshenv` blocked the installer) — every command in this repo
  must export it explicitly until the user fixes that themselves.
- **Tensor backend decision made and documented**: **Burn**, CPU
  (`burn-ndarray`) by default for V0.1, with `burn-wgpu` (Metal) as the
  benchmarked GPU option once there's a model to benchmark. Reasoning in
  `docs/environment.md` §"Tensor/ML backend decision". Not yet added as an
  actual dependency — `tensor-engine` is still a stub; Burn gets pulled in
  when that crate is implemented.
- **Cargo workspace created**: 18 crates under `crates/`, matching the full
  target structure (project spec §3). Root `Cargo.toml` centralizes shared
  dependency versions and workspace metadata.
- **All 18 crates scaffolded** with a real `Cargo.toml` and a documented stub
  `lib.rs`. 11 are active for V0.1 (`financial-types`, `data-engine`,
  `feature-engine`, `financial-graph`, `topology-engine`, `tensor-engine`,
  `neuro-model`, `training-engine`, `evaluation`, `backtester`, `cli`); the
  other 7 (`portfolio-engine`, `risk-engine`, `paper-trading`, `inference`,
  `llm-interface`, `api`, `monitoring`) are directory + stub only, targeted at
  V0.2/V0.3 per spec §51–§52.
- **Configuration system implemented**: `configs/default.toml` holds every
  numeric knob the spec calls out as configurable (topology top-k, EMA
  persistence lambda, regularization weights, training hyperparameters,
  walk-forward windows, backtest cost assumptions). `crates/cli/src/config.rs`
  defines a typed `Config` + `Config::load`, with a unit test that parses the
  real default file.
- **CLI skeleton**: `neurofinance` binary (`crates/cli/src/main.rs`) parses
  `--config` and loads/prints it. Full command surface (§42) is not yet
  implemented — one command lands per future milestone, alongside the crate
  it drives.
- Root docs: `README.md`, `LICENSE` (all-rights-reserved placeholder — no
  license decision has been made), `.gitignore`.
- Repo pushed to **github.com/iamsanaur/neuro-finance** (private), `main`
  branch. Pushes go over HTTPS via `gh`'s git credential helper (the local
  SSH key isn't registered with GitHub yet — see Known issues).

### Milestone 2 additions

- **`financial-types` implemented** (project spec §8–§9):
  - `Timestamp = DateTime<Utc>` — every timestamp in the system is
    timezone-aware by construction, never a raw string.
  - Newtype identifiers (`Symbol`, `EntityId`, `MacroSeriesId`, `MetricId`,
    `Source`, `EventType`) so e.g. a `Symbol` can't be passed where a
    `MacroSeriesId` is expected — a compile error instead of a silent bug.
  - `MarketBar`, `FundamentalObservation`, `MacroObservation`, `NewsEvent` —
    struct shapes match the spec exactly, each with a `validate()` that
    catches malformed data (OHLC inconsistency, non-finite values,
    publication-before-observation, out-of-range sentiment) rather than
    letting it flow downstream silently.
  - **`PointInTime` trait + `PointInTimeDataset<T>`** — the actual
    point-in-time access contract (§9). `PointInTimeDataset` exposes exactly
    one read path, `as_of(query_time)`, which is a binary search (records
    kept sorted by `availability_time`) and structurally cannot return a
    record whose `availability_time` exceeds the query time. There is no
    other way to read the underlying data through this type.
  - `MarketBar`'s `availability_time` is documented as equal to its
    `timestamp` in V0.1 (no publication lag modeled for market data) — an
    explicit, flagged assumption, not a silent one; revisit for intraday/real
    data (V0.2+).
  - Leakage test included at this layer already (§30):
    `as_of_excludes_records_available_after_query_time` — a record observed
    long ago but published late must not appear in an early `as_of` query.

### Milestone 3 additions

- **Synthetic market generator implemented** (`data-engine::synthetic`,
  project spec §10): 100 assets / 10 sectors by default, configurable via
  `SyntheticMarketConfig`. Generative model (documented in full in
  `generator.rs`'s module doc, since "no hidden data transformations" is a
  hard requirement):
  - A shared market factor following a real **GARCH(1,1)** process (mean-
    reverting to a regime-scaled long-run variance) — produces genuine
    **volatility clustering**, verified by a positive lag-1 autocorrelation
    test on squared market-wide returns.
  - Per-sector factors pulled toward the market factor by a regime-dependent
    **contagion loading** — small in `RiskOn`/`Neutral`, large in `RiskOff`
    — the explicit mechanism for **cross-sector contagion**.
  - A **hot-sector factor**, active only for one sector per regime (sector 0
    in `RiskOn`, sector 1 in `RiskOff`, none in `Neutral`) — the mechanism
    for **regime-specific topology** (§10's "tech strongly connected in one
    regime, financials in another").
  - A 3-state Markov regime chain (`RiskOn` / `Neutral` / `RiskOff`) with
    configurable persistence — these three states double as the ground-truth
    label for the eventual regime-classification task (§25), so the
    synthetic data already targets the right label space.
  - Occasional regime-independent sector shocks, and lognormal-noised
    volume that scales with `|return|`.
  - A macro "realized volatility" series published with a **nonzero
    publication lag** (configurable, default 1 day) — gives the
    point-in-time machinery from Milestone 2 a real lagged series to guard,
    not just a contrived test fixture.
- **The generator's core claim is empirically tested, not assumed** (§10:
  "This is a scientific validation step."): with a same-regime control
  sector as baseline (isolating the hot-sector-specific effect from generic
  regime-wide contagion), sector 0's pairwise correlation is verified to
  exceed the control sector's specifically during `RiskOn`, sector 1's
  specifically during `RiskOff`, and cross-sector correlation is verified to
  rise in `RiskOff` vs. `Neutral`. A first version of this test compared
  sector 0 across regimes directly and failed — `RiskOff` contagion turned
  out to raise correlation *everywhere*, including in sector 0, more than
  `RiskOn`'s isolated hot-factor effect did. That's a real, intentional
  property of the generator (crises make everything correlated), not a bug;
  the test was corrected to control for it rather than the generator being
  weakened to pass a flawed test.
- Every generated `MarketBar` passes `MarketBar::validate()` (checked in a
  dedicated test); generation is verified deterministic given a seed, and
  verified to differ given a different seed.
- Not yet implemented in `data-engine`: `MarketDataProvider` /
  `FundamentalDataProvider` / `MacroDataProvider` / `NewsDataProvider`
  traits (§7), CSV/Parquet adapters. The synthetic generator produces
  in-memory `financial-types` structs directly; nothing is persisted to
  `data/raw` or `data/processed` yet — that lands with the provider trait
  layer.

### Milestone 4 additions

- **`feature-engine` implemented** (project spec §11): log returns, rolling
  (multi-period) returns, momentum, rolling volatility, moving average,
  drawdown, rolling correlation, rolling beta, volume change, dollar-volume
  liquidity.
  - Every rolling function is built on one primitive, `rolling::rolling_apply`,
    which enforces trailing-only alignment, an explicit `window`, and an
    explicit `min_periods` in a single place (§11: "Every rolling operation
    must explicitly define: window, alignment, minimum observations. No
    centered windows.") — `drawdown` is the one exception, documented as an
    expanding (not fixed) window by necessity, still provably causal.
  - **Causality is tested as a property, not spot-checked**: every module
    has a `prefix_computation_matches_full_computation` test — computing a
    feature on a prefix of a series must reproduce, index-for-index, what
    computing on the full series gives for those same indices. This is the
    concrete form of §30's "future rolling statistics" leakage test.
  - An integration test (`tests/synthetic_pipeline.rs`) runs the same
    causality property against real `data-engine`-generated series (GARCH
    clustering, regime switches, sector shocks included) rather than only
    smooth hand-built fixtures, since irregular series are what would
    actually expose an off-by-one leak.
  - Explicitly deferred to V0.2 (§51, when real fundamental/macro data
    exists): valuation, revenue growth, earnings growth, interest-rate
    changes, yield curve slope — there is no data source for any of these
    yet, so stub functions would be untestable.

### Milestone 5 additions

- **`financial-graph` implemented** (project spec §12–§14):
  - `FinancialGraph`: sparse storage only — a flat edge `Vec` plus per-node
    adjacency lists of edge indices, no `N x N` matrix anywhere in the type
    (§13). Undirected (every V0.1 relation is symmetric); `add_edge`
    rejects out-of-range nodes and self-loops.
  - `RelationType` enum has all 7 variants from §12 (`Sector`, `Industry`,
    `Correlation`, `Fundamental`, `Macro`, `News`, `Learned`) even though
    V0.1 only produces `Sector`/`Correlation` — so downstream crates
    (`topology-engine`, `neuro-model`) can be written against the full
    relation space without a breaking change later.
  - Node identity is `EntityId` (not `Symbol`) — matches `financial-types`'
    own doc for `EntityId` as the general graph-node identity; `Symbol` ->
    `EntityId` conversion happens at each builder's boundary.
  - `build_sector_graph`: asset-asset edges between every pair sharing a
    sector, grouped by sector first (not an all-pairs scan) to stay
    `O(n * avg_sector_size)`.
  - `build_correlation_graph`: edges weighted by pairwise rolling return
    correlation (via `feature-engine::correlation::pearson`, newly exposed
    for this), thresholded by `min_abs_correlation` to stay sparse (an
    unthresholded correlation graph is close to complete). **Point-in-time
    safety is the caller's responsibility by contract** — the function
    trusts whatever `bars` it's given; a test
    (`correlation_graph_does_not_use_future_bars`) demonstrates the actual
    leak that results from passing unfiltered bars, and confirms correct
    usage (`PointInTimeDataset::as_of` truncation first) is deterministic —
    the concrete form of §30's "future graph edges" leakage test.
  - Not yet implemented: industry/fundamental/macro/news graphs (V0.2+, no
    data source yet); per-relation learnable importance (`alpha_k`, §15)
    and the dynamic/learned topology itself (§16) — those belong to
    `topology-engine`, not this crate.

### Milestone 6 additions

- **`tensor-engine` implemented** — the Burn-wrapping interface promised
  since Milestone 1's environment decision. Re-exports `burn` wholesale
  (`tensor_engine::burn`) plus two type aliases (`Backend`, `Device`) and a
  `seed()` helper; no other crate depends on `burn` directly. Backend:
  `Autodiff<NdArray<f32>>` — autodiff wrapped in now even though nothing
  trains gradients yet, specifically to avoid a breaking `Tensor<B, _>` type
  change across the workspace once `training-engine` needs it.
  - **First real check of the Milestone 1 backend decision**: added Burn as
    an actual dependency for the first time. MSRV had to be bumped from
    1.75 to 1.92 (root `Cargo.toml`) — the declared MSRV was capping Burn to
    an ancient 0.13.2 even though rustc 1.98.0 is actually installed; fixed
    by raising the declared MSRV to match reality, not by pinning an old
    Burn. Resolved to Burn 0.21.0 stable.
  - **Real finding, documented rather than hidden**: Burn 0.21's `NdArray`
    backend fills random tensors (including module weight init) via
    rayon-parallel chunks, and `seed()` does **not** guarantee bit-identical
    weights across separately-constructed modules under normal
    multi-threaded execution — confirmed by reproducing it with
    `RAYON_NUM_THREADS=1` (passes) vs. default threading (fails). Lower-level
    single-tensor RNG (`Tensor::random`) *is* reliably deterministic under
    seeding — only module-initialization's parallel fill path is affected.
    Documented in `tensor-engine`'s and `topology-engine::scorer`'s doc
    comments; the affected test is kept as `#[ignore]` (not deleted) so the
    limitation stays checkable. **This is an open item for
    `training-engine`** (a later milestone) to resolve, work around
    (`RAYON_NUM_THREADS=1`), or accept with a documented caveat before any
    claim of exact experiment reproducibility (§31/§32) is made.
- **`topology-engine` implemented** (project spec §16–§18):
  - `TopologyScorer`: `Q`/`K` linear projections producing a dense `[N, N]`
    score matrix (`s_ij = Q_i^T K_j / sqrt(d)`) — an intentional intermediate,
    never stored past top-k reduction (documented explicitly, since §13
    could otherwise be misread as forbidding this standard attention step).
  - `top_k_topology`: reduces scores to a sparse `FinancialGraph` via
    **mutual top-k** (edge survives only if each endpoint is in the other's
    top-k) — chosen specifically because it's what makes §47's "topology
    degree <= configured top-k" hold as a hard guarantee, not just usually
    true; a union rule was considered and rejected for exactly this reason.
  - `TopologyPersistence`: EMA blending (`A_t = lambda*A_{t-1} +
    (1-lambda)*A_new`) keyed by `EntityId` pairs (not `NodeId`, which isn't
    stable across separately-built graphs); near-zero persisted edges are
    dropped so decayed edges don't accumulate forever. `TopologyDiff`
    reports created/deleted/persisted edges.
  - `l_sparse`/`l_stability`: differentiable losses over the dense score
    matrix (so `training-engine` can later sum them into a real training
    loss and backprop through `W_q`/`W_k`). `l_relation`: a plain scalar
    (edge-set overlap against a reference graph) since there's no way to
    backprop through discrete graph membership; defaults to inactive
    (`lambda_relation = 0.0` in `configs/default.toml`) since no relation
    graph is wired into training yet.
  - `connected_components`: the one structural metric needing real graph
    traversal; full community detection deliberately deferred to the
    topology-research milestone (§34) where it's actually needed.

### Milestone 7 additions

- **`neuro-model` implemented**, scoped intentionally to the first slice of
  §19–§25, not the full architecture:
  - `FeatureEncoder`: one linear projection + ReLU, raw causal features
    (`[N, feature_dim]`) → node embeddings (`[N, embed_dim]`).
  - `GraphMessagePassing` (§19, "basic graph aggregation" variant only —
    graph attention and dynamic sparse attention explicitly deferred to a
    later milestone, §28's baseline-first discipline): `m_i = sum_{j in
    N(i)} alpha_ij * W_v * h_j`, `h'_i = h_i + MLP(m_i)`, `alpha_ij` = edge
    weight normalized by degree. Implemented via a dense `[N, N]` matmul —
    documented explicitly as a deliberate, scale-appropriate choice (N=100)
    and *not* a violation of §13's sparse-storage rule, which governs
    `FinancialGraph`'s storage, not a message-passing kernel's math.
  - `RegimeHead`: mean-pools node embeddings, linear + softmax → 3-class
    probabilities `[RiskOn, Neutral, RiskOff]` (§25's first model task),
    matching `data-engine::synthetic::MarketRegime`'s class order exactly.
  - `NeuroTopologicalFinancialModel`: wires encoder → `topology-engine`'s
    scorer/top-k → message passing → regime head. **Explicitly not
    implemented**: global attention/fusion (§20), temporal encoding (§21,
    so this model is single-day/cross-sectional only, no sequence notion
    yet), hierarchical graph (§22), financial memory (§23).
  - **Documented, not hidden, architectural limitation**: gradients don't
    flow through `top_k_topology`'s hard selection, so backpropagating a
    prediction loss through this model trains everything *except*
    `TopologyScorer`'s `W_q`/`W_k`. Those train separately, directly on the
    differentiable score matrix, via `topology-engine`'s `l_sparse`/
    `l_stability` — a standard way to handle hard top-k in graph learning,
    not a placeholder for something more sophisticated pending later.
  - **Integration test** (`tests/synthetic_pipeline.rs`): the first version
    of §48's synthetic end-to-end test — synthetic bars → point-in-time-safe
    features (via `PointInTimeDataset::as_of`, `feature-engine`) → model →
    regime probabilities, on real (not fixture) generated data, plus a
    point-in-time-safety sanity check (different `as_of` days produce
    different features from the same dataset).

### Milestone 8 additions

- **`training-engine` implemented** (project spec §29, §31):
  - `WalkForwardValidator`: expanding or rolling train windows, fixed-length
    validation/test windows, an embargo gap on both boundaries (§29:
    "purging, embargo"). Rolls forward by `test_period` between splits, never
    returns a partial trailing split. `from_years` matches
    `configs/default.toml`'s `[walk_forward]` section field names directly.
  - `train_epoch`/`evaluate`: the actual training loop. "Mini-batching"
    here means **gradient accumulation** over `batch_size` trading days
    (documented why: each day's topology graph has a different structure,
    so days can't be stacked into one same-shaped batched tensor the way
    i.i.d. examples usually are).
  - `nll_loss`: negative log-likelihood computed directly on `RegimeHead`'s
    already-softmax-normalized output (rather than reusing Burn's
    logit-based `CrossEntropyLoss`, which would have required restructuring
    `neuro-model` just to expose pre-softmax logits).
  - `EarlyStopping`: patience + minimum-improvement-delta, tested including
    the "improvement resets the strike counter" case.
  - `save_checkpoint`/`load_checkpoint`: round-trip tested (`NamedMpkFileRecorder`)
    to confirm a loaded model reproduces identical forward output.
  - Baseline models (§28) deliberately **not** here — they belong to
    `evaluation` per the crate's Milestone-1 description; adding them to
    `training-engine` would be scope creep this crate doesn't need.

### Milestone 11 additions

- **`evaluation::MlpBaseline`**: one hidden layer, no graph, exposes
  `param_count(feature_dim, hidden_dim, num_classes)` so a
  capacity-matched comparison can be verified numerically rather than
  eyeballed. Deliberately built to isolate "extra capacity overfits" from
  "the graph/topology mechanism specifically overfits" — the open question
  `exp-0002` left behind.
- **`examples/third_experiment.rs`** (`exp-0003`): evaluates all four
  trained/fit models (`neuro_model`, `mlp_baseline`, `logistic_regression`,
  `naive_persistence`, `majority_class`) across every walk-forward split
  from a longer (1100-day) synthetic run, not just the first split. See
  the result above.

### Milestone 10 additions

- **`feature_engine::Standardizer`**: z-score standardization, `fit` on
  train data only, `transform`/`transform_all` applied unchanged to
  held-out data — the point-in-time-safe pattern (never fit on
  validation/test). Constant-feature edge case (`std == 0`) maps to `0.0`
  rather than dividing by zero.
- **`examples/second_experiment.rs`** (`exp-0002`): same market/seed/split
  as `exp-0001`, with standardized features and a longer training
  schedule. See the result above.

### Milestone 9 additions

- **`evaluation` implemented** (project spec §28):
  - `NaivePersistenceBaseline` (echoes the last observed class),
    `MajorityClassBaseline` (fit once on train labels), and
    `LogisticRegressionBaseline` (mean-pool → linear → softmax, same
    input/output shape as `neuro_model::RegimeHead` but no graph/topology —
    the direct "does the graph earn its complexity" comparison point).
    Gradient boosting, MLP, LSTM/GRU, Transformer, static/graph-Transformer
    baselines deliberately deferred (§2: no point comparing against a GBM
    before the two cheapest baselines have even been checked).
  - `classification_report`: accuracy + per-class precision/recall/F1.
    AUC, directional accuracy, regression error deferred — no
    binary-direction or regression task exists yet (§26/§27).
  - `examples/first_experiment.rs`: the project's first genuine experiment
    runner — synthetic data → point-in-time-safe features → walk-forward
    split → train `neuro-model` and the logistic baseline → evaluate all
    four models on the held-out test window → write `config.json`/
    `metrics.json`/`RESEARCH_REPORT.md` to `experiments/exp-0001.../`.
    Run via `cargo run --release --example first_experiment -p evaluation`.
- **First experiment run and reported** — see the section above. Negative/
  inconclusive result, reported honestly rather than tuned until positive.

## Verification (cumulative, latest milestone)

```
cargo build --workspace     → success, 18 crates compiled
cargo test --workspace      → 148 passed, 0 failed, 1 ignored (documented)
                               (18 financial-types, 15 data-engine,
                                35 feature-engine, 13 financial-graph,
                                2 tensor-engine, 22 topology-engine,
                                9 neuro-model, 21 training-engine,
                                12 evaluation, 1 cli)
cargo clippy --workspace --all-targets → no issues found (incl. all 3 examples)
cargo fmt --all -- --check  → clean
cargo run -p cli -- --config configs/default.toml
  → Loaded config from configs/default.toml: 100 assets across 10 sectors,
    sequence_length=30, topology_top_k=8
cargo run --release --example third_experiment -p evaluation
  → see experiments/exp-0003-capacity-matched-multi-split/ for full output
    (exp-0001, exp-0002 kept for the record; superseded by exp-0003)
```

## Known issues / risks

- `~/.cargo/bin` PATH is not persisted in the user's shell profile. Anyone
  (including future Claude sessions) running `cargo` here needs
  `export PATH="$HOME/.cargo/bin:$PATH"` first, or the user needs to fix
  `~/.zshenv` (root-owned) themselves.
- Burn is not yet an actual dependency anywhere — the backend decision is
  documented but unverified against real tensor ops. First real check of
  that decision happens when `tensor-engine` is implemented and benchmarked.
- Local SSH key (`~/.ssh/id_ed25519`) is not registered with GitHub; pushes
  use `gh`'s HTTPS credential helper instead. Fine for now, but if the user
  wants SSH-based git going forward, the public key needs adding to their
  GitHub account.
- `MarketBar.availability_time() == timestamp` is a V0.1 simplification
  (see above) — must be revisited before this system is trusted with
  intraday or real vendor-latency data.

## Next milestone

**Milestone 12: independent-seed replication, then decide on V0.1's
scientific conclusion.** `exp-0003`'s splits are 4 windows of *one*
generated market history, not independent re-generations — the natural
next check is re-running the same comparison across a handful of different
`SyntheticMarketConfig` seeds, to see whether "MLP/logistic beat
neuro_model" holds up as a seed-independent pattern or was itself an
artifact of this one market path. If it holds across seeds, V0.1's honest
conclusion (per §54) is that this project's own evidence argues against
its central hypothesis on same-day classification with no temporal
component — worth writing up as the actual V0.1 research report (§55) at
that point, rather than continuing to add architecture before the
existing evidence is taken seriously. A temporal component (§21) is the
most plausible remaining lever (per `exp-0003`'s own discussion) if the
project continues past that point.

Still open: the Milestone 6 Burn RNG-determinism caveat
(`RAYON_NUM_THREADS=1` needed for exact weight-init reproducibility) — not
yet resolved, still just documented.
