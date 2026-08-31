# NeuroTopological Financial AI

A research-grade platform investigating whether a **dynamically learned
topology of financial entities** provides predictive information that
conventional statistical models, Transformers, and static graph neural
networks do not capture.

This is a research project first, production platform second. The central
question — does learning financial topology improve out-of-sample forecasting
and risk-adjusted portfolio performance? — is treated as genuinely open. A
negative result is an acceptable, publishable outcome; results are never
manufactured to look positive. See `docs/research-methodology.md` (once
written) and `PROJECT_STATUS.md` for where things currently stand.

## Status

Early scaffolding (V0.1 in progress). See [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Layout

- `crates/` — the Cargo workspace (18 crates; see each crate's `Cargo.toml`
  description for what it owns). `financial-types`, `data-engine`,
  `feature-engine`, `financial-graph`, `topology-engine`, `tensor-engine`,
  `neuro-model`, `training-engine`, `evaluation`, `backtester`, and `cli` are
  the active V0.1 crates; the rest are scaffolded but not yet implemented
  (targeted at V0.2/V0.3, per `PROJECT_STATUS.md`).
- `configs/` — TOML configuration (see `configs/default.toml`).
- `data/`, `models/`, `experiments/` — generated artifacts, gitignored except
  for structure.
- `docs/` — architecture, data model, methodology, and per-subsystem docs.

## Getting started

```bash
# Rust toolchain (rustup) must be on PATH:
export PATH="$HOME/.cargo/bin:$PATH"

cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --all -- --check

cargo run -p cli -- --config configs/default.toml
```

## Non-negotiables

No look-ahead bias, no random train/test splits on financial data, no
zero-cost backtesting, no hidden data transformations, no hard-coded secrets,
no direct model-to-broker execution. See the project spec for the full list —
these are enforced by design (point-in-time data access, walk-forward
validation, an independent risk engine) and checked by dedicated leakage
tests, not by convention alone.
