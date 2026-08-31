//! feature-engine
//!
//! Causal, point-in-time-safe feature computation (project spec §11):
//! log/rolling returns, volatility, momentum, moving averages, drawdown,
//! rolling correlation, beta, volume change, liquidity.
//!
//! Every rolling operation explicitly declares its window, its alignment
//! (always trailing — see [`rolling::rolling_apply`]), and its minimum
//! observations; none accepts or produces a centered window, and every
//! module's tests include a prefix/full-series equivalence check — the
//! concrete form project spec §30's "no future rolling statistics" leakage
//! test takes here.
//!
//! Not yet implemented (deferred to V0.2, when `data-engine` grows real
//! fundamental/macro providers, per project spec §51): valuation, revenue
//! growth, earnings growth, interest-rate changes, yield curve slope. There
//! is currently no data source for any of these — adding the functions now
//! would mean untestable code with nothing real to compute over.

pub mod beta;
pub mod correlation;
pub mod drawdown;
pub mod moving_average;
pub mod returns;
pub mod rolling;
pub mod series;
pub mod volatility;
pub mod volume;

pub use beta::rolling_beta;
pub use correlation::rolling_correlation;
pub use drawdown::drawdown;
pub use moving_average::moving_average;
pub use returns::{log_returns, momentum, rolling_return};
pub use rolling::rolling_apply;
pub use series::{bars_for_symbol, close_series, volume_series};
pub use volatility::rolling_volatility;
pub use volume::{dollar_volume, volume_change};
