# Project Status

Last updated: 2026-08-31 (Milestone 4)

## Current milestone

**Milestone 4: `feature-engine` — causal, point-in-time-safe features — DONE**

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

## Verification (cumulative, latest milestone)

```
cargo build --workspace     → success, 18 crates compiled
cargo test --workspace      → 68 passed, 0 failed
                               (18 financial-types, 15 data-engine,
                                33 feature-engine, 1 cli, 1 doctest)
cargo clippy --workspace --all-targets → no issues found
cargo fmt --all -- --check  → clean
cargo run -p cli -- --config configs/default.toml
  → Loaded config from configs/default.toml: 100 assets across 10 sectors,
    sequence_length=30, topology_top_k=8
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

**Milestone 5: `financial-graph`** (project spec §12–§15) — the
`FinancialGraph` type (nodes, edges, relation types), sparse adjacency
storage (not dense N×N), and the first two static graphs: a sector graph
(from the synthetic universe's sector assignment) and a correlation graph
(from `feature-engine`'s `rolling_correlation`, computed point-in-time-safe
as of each day). This is what the topology learner (Milestone 7+) will
eventually be compared against as a baseline.
