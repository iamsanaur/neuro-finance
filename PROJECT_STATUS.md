# Project Status

Last updated: 2026-08-31 (Milestone 2)

## Current milestone

**Milestone 2: `financial-types` — strongly typed, point-in-time data structures — DONE**

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

## Verification (cumulative, latest milestone)

```
cargo build --workspace     → success, 18 crates compiled
cargo test --workspace      → 19 passed, 0 failed (18 in financial-types + 1 in cli)
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

**Milestone 3: synthetic market generator** (`data-engine`, project spec
§10): 100 assets, 10 sectors, 3–5 market regimes, correlated assets with
time-varying correlation, volatility clustering, macro factors, sector
shocks, cross-sector contagion — with a *known* hidden topology (e.g. tech
assets tightly connected in one regime, financials in another) so later
milestones can test whether the topology-learning components actually
recover it. This is the first genuinely research-bearing component, not just
scaffolding.
