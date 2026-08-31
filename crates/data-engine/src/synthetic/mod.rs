//! Synthetic financial market generator (project spec §10).
//!
//! Produces a market with a *known* hidden topology — which sector is
//! strongly interconnected in which regime, and when cross-sector
//! contagion spikes — so that later milestones (topology learning, §16;
//! topology research, §34) can be scientifically validated against ground
//! truth, not just judged on downstream prediction accuracy alone.

pub mod config;
pub mod generator;
pub mod regime;

pub use config::{SyntheticMarketConfig, SyntheticMarketConfigError};
pub use generator::{SyntheticMarketDataset, SyntheticMarketGenerator};
pub use regime::{MarketRegime, RegimeTransition};
