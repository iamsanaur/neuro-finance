//! Configuration for the synthetic market generator.
//!
//! Every parameter of the generative model lives here, not inline in the
//! generator (project spec: "No hidden data transformations. All
//! transformations must be explicit."). See `generator.rs` module doc for
//! the exact equations these parameters feed into.

#[derive(Debug, Clone)]
pub struct SyntheticMarketConfig {
    pub num_assets: usize,
    pub num_sectors: usize,
    pub num_days: usize,
    pub seed: u64,

    /// Inclusive range for each asset's initial price, drawn uniformly.
    pub initial_price_range: (f64, f64),

    /// Loading of the common market factor on every asset's return.
    pub beta_market: f64,
    /// Loading of an asset's own sector factor on its return.
    pub beta_sector: f64,
    /// Loading of the regime's hot-sector factor on member assets' returns,
    /// active only while their sector is the current regime's hot sector.
    pub beta_hot: f64,

    /// Baseline (unconditional-variance target) market volatility; scaled
    /// per-day by [`crate::synthetic::regime::MarketRegime::vol_multiplier`]
    /// and by GARCH(1,1) clustering (see `generator.rs`).
    pub sigma_market_base: f64,
    /// Std. dev. of each sector factor (drawn fresh every day).
    pub sigma_sector: f64,
    /// Std. dev. of each asset's idiosyncratic daily noise.
    pub sigma_idio: f64,

    /// GARCH(1,1) `alpha`: weight on yesterday's squared market shock.
    pub garch_alpha: f64,
    /// GARCH(1,1) `beta`: weight on yesterday's conditional variance.
    /// `garch_alpha + garch_beta` must be `< 1` for the process to be
    /// stationary (mean-reverting) — enforced in
    /// [`SyntheticMarketConfig::validate`].
    pub garch_beta: f64,

    /// Probability, on the Markov regime chain, of remaining in the current
    /// regime on the next trading day. See [`crate::synthetic::regime::RegimeTransition`].
    pub regime_stay_probability: f64,

    /// Per-day probability of an idiosyncratic sector shock (a jump
    /// affecting one randomly chosen sector only, independent of regime).
    pub sector_shock_probability: f64,
    /// Std. dev. of a sector shock's magnitude, when one occurs.
    pub sector_shock_scale: f64,

    /// Fraction of daily volatility used to generate the high/low range
    /// around open/close (V0.1 has no overnight gap — see `generator.rs`).
    pub intraday_range_factor: f64,

    /// Baseline daily volume (shares), before the volume-vs-volatility
    /// relationship and lognormal noise are applied.
    pub base_volume: f64,

    /// Days between a macro observation's `observation_time` and its
    /// `publication_time` — deliberately nonzero so downstream point-in-time
    /// tests have a real publication lag to catch, not just observation
    /// time (see `financial-types::point_in_time`).
    pub macro_publication_lag_days: i64,
}

impl Default for SyntheticMarketConfig {
    fn default() -> Self {
        Self {
            num_assets: 100,
            num_sectors: 10,
            num_days: 750, // ~3 trading years
            seed: 42,
            initial_price_range: (20.0, 200.0),
            beta_market: 0.55,
            beta_sector: 0.8,
            beta_hot: 1.7,
            sigma_market_base: 0.010,
            sigma_sector: 0.008,
            sigma_idio: 0.012,
            garch_alpha: 0.08,
            garch_beta: 0.90,
            regime_stay_probability: 0.98,
            sector_shock_probability: 0.01,
            sector_shock_scale: 0.03,
            intraday_range_factor: 0.6,
            base_volume: 1_000_000.0,
            macro_publication_lag_days: 1,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SyntheticMarketConfigError {
    #[error(
        "num_assets ({num_assets}) must be a positive multiple of num_sectors ({num_sectors})"
    )]
    AssetsNotDivisibleBySectors {
        num_assets: usize,
        num_sectors: usize,
    },
    #[error("num_days must be at least 2 to compute a single return, got {0}")]
    TooFewDays(usize),
    #[error(
        "GARCH process is non-stationary: garch_alpha ({garch_alpha}) + garch_beta ({garch_beta}) >= 1"
    )]
    NonStationaryGarch { garch_alpha: f64, garch_beta: f64 },
}

impl SyntheticMarketConfig {
    pub fn validate(&self) -> Result<(), SyntheticMarketConfigError> {
        if self.num_sectors == 0 || self.num_assets % self.num_sectors != 0 {
            return Err(SyntheticMarketConfigError::AssetsNotDivisibleBySectors {
                num_assets: self.num_assets,
                num_sectors: self.num_sectors,
            });
        }
        if self.num_days < 2 {
            return Err(SyntheticMarketConfigError::TooFewDays(self.num_days));
        }
        if self.garch_alpha + self.garch_beta >= 1.0 {
            return Err(SyntheticMarketConfigError::NonStationaryGarch {
                garch_alpha: self.garch_alpha,
                garch_beta: self.garch_beta,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert!(SyntheticMarketConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_indivisible_assets() {
        let cfg = SyntheticMarketConfig {
            num_assets: 101,
            ..SyntheticMarketConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(SyntheticMarketConfigError::AssetsNotDivisibleBySectors { .. })
        ));
    }

    #[test]
    fn rejects_non_stationary_garch() {
        let cfg = SyntheticMarketConfig {
            garch_alpha: 0.6,
            garch_beta: 0.6,
            ..SyntheticMarketConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(SyntheticMarketConfigError::NonStationaryGarch { .. })
        ));
    }
}
