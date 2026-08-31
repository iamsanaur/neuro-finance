# Project Status

Last updated: 2026-08-31 (Milestone 1)

## Current milestone

**Milestone 1: Environment Inspection & Workspace Scaffold — DONE**

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
- Repo is **not yet a git repository** — nothing has been committed.

## Verification (this milestone)

```
cargo build --workspace     → success, 18 crates compiled
cargo test --workspace      → 1 passed, 0 failed (cli::config::tests::loads_default_config)
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
- No git repository yet. Nothing is version-controlled until `git init` +
  first commit, which should happen before further milestones accumulate
  more untracked work.

## Next milestone

**Milestone 2: `financial-types` — strongly typed, timezone-aware data
structures** (project spec §8–§9): `MarketBar`, `FundamentalObservation`,
`MacroObservation`, `NewsEvent`, and the point-in-time envelope types
(observation timestamp vs. availability timestamp) that the rest of the
system builds on, plus unit tests establishing the point-in-time access
contract from the start. `git init` + first commit happens as part of this
milestone, before new files pile up further.
