//! The synthetic market generator (project spec §10).
//!
//! # Generative model
//!
//! For each trading day `t`, one shared market factor, `num_sectors` sector
//! factors, and (while a hot sector is active) one hot-sector factor are
//! drawn, and every asset's daily log return is a fixed linear combination
//! of the factor(s) for its sector plus idiosyncratic noise:
//!
//! ```text
//! r_i,t = beta_market * F_market_t
//!       + beta_sector * F_sector[sector(i)],t
//!       + beta_hot     * H_t                      (only if sector(i) is the regime's hot sector)
//!       + shock_t                                  (only on a sector-shock day, if sector(i) is the shocked sector)
//!       + epsilon_i,t
//! ```
//!
//! - `F_market_t = sigma_t * z_t`, `z_t ~ N(0,1)`, with `sigma_t` following a
//!   GARCH(1,1) process (see [`Garch`]) scaled by the active regime's
//!   volatility multiplier — this is what produces **volatility
//!   clustering**: a large `|F_market_t|` today raises the *expected*
//!   `sigma` tomorrow.
//! - Each `F_sector_s,t ~ N(0, sigma_sector^2)` independently, then pulled
//!   toward the shared market factor by the active regime's contagion
//!   loading: `F_sector_s,t <- (1 - c) * F_sector_s,t + c * F_market_t`.
//!   `c` is small in `RiskOn`/`Neutral` and large in `RiskOff` — this is the
//!   explicit **cross-sector contagion** mechanism (§10).
//! - `H_t ~ N(0, sigma_sector^2)` is drawn only while a regime has a hot
//!   sector (see [`crate::synthetic::regime::MarketRegime::hot_sector_index`]);
//!   it is the mechanism behind **regime-specific topology**
//!   (tech assets tightly linked in `RiskOn`, financials in `RiskOff`).
//! - `epsilon_i,t ~ N(0, sigma_idio^2)` is independent across assets and
//!   days.
//! - Sector shocks are a second, regime-independent contagion mechanism: on
//!   a Bernoulli-triggered day, one randomly chosen sector's members all
//!   receive the same jump.
//!
//! Prices evolve as `close_t = close_{t-1} * exp(r_i,t)`. **V0.1 does not
//! model an overnight gap**: `open_t = close_{t-1}` exactly, so
//! `ln(close_t / close_{t-1})` recovers `r_i,t` exactly — this is what lets
//! the tests below validate regime-conditional correlation directly from
//! the public `MarketBar` output rather than from internal generator state.
//! High/low are `open`/`close` widened by a fraction of the day's realized
//! volatility (`intraday_range_factor`). Volume scales with `|r_i,t|`
//! (higher volume on bigger moves) times lognormal noise.

use crate::synthetic::config::SyntheticMarketConfig;
use crate::synthetic::regime::{MarketRegime, RegimeTransition};
use chrono::{Duration, TimeZone, Utc};
use financial_types::{EntityId, MacroObservation, MacroSeriesId, MarketBar, Symbol, Timestamp};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal, Normal};

/// GARCH(1,1) conditional-variance process for the market factor.
///
/// `sigma2_t = omega_t + garch_alpha * eps_{t-1}^2 + garch_beta * sigma2_{t-1}`,
/// where `omega_t = base_variance * (1 - alpha - beta) * regime_multiplier^2`
/// so that, absent shocks, `sigma2` mean-reverts toward
/// `base_variance * regime_multiplier^2` — i.e. the regime's target
/// volatility, not a fixed unconditional constant. This is a standard
/// GARCH(1,1) with a time-varying (regime-scaled) long-run variance target,
/// not full regime-switching GARCH — sufficient to produce genuine
/// volatility clustering without the added complexity of a regime-aware
/// likelihood (out of scope for a data generator).
struct Garch {
    alpha: f64,
    beta: f64,
    base_variance: f64,
    sigma2: f64,
    prev_eps: f64,
}

impl Garch {
    fn new(alpha: f64, beta: f64, base_sigma: f64) -> Self {
        let base_variance = base_sigma * base_sigma;
        Self {
            alpha,
            beta,
            base_variance,
            sigma2: base_variance,
            prev_eps: 0.0,
        }
    }

    /// Advances the process by one day under the given regime multiplier and
    /// returns the day's realized market factor `eps_t`.
    fn step(&mut self, regime_multiplier: f64, rng: &mut impl Rng) -> f64 {
        let omega = self.base_variance
            * (1.0 - self.alpha - self.beta)
            * regime_multiplier
            * regime_multiplier;
        self.sigma2 =
            (omega + self.alpha * self.prev_eps * self.prev_eps + self.beta * self.sigma2)
                .max(1e-12);
        let sigma = self.sigma2.sqrt();
        let z: f64 = Normal::new(0.0, 1.0).unwrap().sample(rng);
        let eps = sigma * z;
        self.prev_eps = eps;
        eps
    }
}

/// The full generated dataset, plus the ground truth a later milestone can
/// validate a learned topology against (§10, §34): which sector each asset
/// belongs to, and the regime path actually used to generate the data.
#[derive(Debug, Clone)]
pub struct SyntheticMarketDataset {
    pub bars: Vec<MarketBar>,
    pub sector_assignment: Vec<(Symbol, EntityId)>,
    /// One entry per trading day, in chronological order. This is the
    /// ground-truth label the regime-classification task (§25) predicts —
    /// it is exposed here for training/evaluation labeling, and must never
    /// be fed into `feature-engine` as an input feature.
    pub regime_schedule: Vec<(Timestamp, MarketRegime)>,
    pub macro_observations: Vec<MacroObservation>,
}

pub struct SyntheticMarketGenerator {
    config: SyntheticMarketConfig,
}

impl SyntheticMarketGenerator {
    pub fn new(config: SyntheticMarketConfig) -> Self {
        Self { config }
    }

    /// Deterministic given `config.seed` — same config in, byte-identical
    /// dataset out (see the `determinism` test). This is what makes
    /// experiments reproducible (project spec §31, §32).
    pub fn generate(&self) -> SyntheticMarketDataset {
        let cfg = &self.config;
        cfg.validate().expect("invalid SyntheticMarketConfig");

        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        let assets_per_sector = cfg.num_assets / cfg.num_sectors;

        let symbols: Vec<Symbol> = (0..cfg.num_assets)
            .map(|i| Symbol::from(format!("AST{i:04}")))
            .collect();
        let sector_of_asset: Vec<usize> =
            (0..cfg.num_assets).map(|i| i / assets_per_sector).collect();
        let sector_ids: Vec<EntityId> = (0..cfg.num_sectors)
            .map(|s| EntityId::from(format!("SECTOR{s:02}")))
            .collect();
        let sector_assignment: Vec<(Symbol, EntityId)> = symbols
            .iter()
            .zip(sector_of_asset.iter())
            .map(|(sym, &s)| (sym.clone(), sector_ids[s].clone()))
            .collect();

        let start_date = Utc.with_ymd_and_hms(2015, 1, 2, 0, 0, 0).unwrap();

        let price_dist = rand::distributions::Uniform::new_inclusive(
            cfg.initial_price_range.0,
            cfg.initial_price_range.1,
        );
        let mut prev_close: Vec<f64> = (0..cfg.num_assets)
            .map(|_| price_dist.sample(&mut rng))
            .collect();

        let sector_factor_dist = Normal::new(0.0, cfg.sigma_sector).unwrap();
        let idio_dist = Normal::new(0.0, cfg.sigma_idio).unwrap();
        let shock_size_dist = Normal::new(0.0, cfg.sector_shock_scale).unwrap();

        let mut garch = Garch::new(cfg.garch_alpha, cfg.garch_beta, cfg.sigma_market_base);
        let transition = RegimeTransition::new(cfg.regime_stay_probability);

        let mut bars = Vec::with_capacity(cfg.num_assets * cfg.num_days);
        let mut regime_schedule = Vec::with_capacity(cfg.num_days);
        let mut macro_observations = Vec::with_capacity(cfg.num_days);

        let mut regime = MarketRegime::Neutral;
        for day in 0..cfg.num_days {
            let timestamp = start_date + Duration::days(day as i64);
            regime = if day == 0 {
                regime
            } else {
                transition.next(regime, &mut rng)
            };
            regime_schedule.push((timestamp, regime));

            // Market factor, via GARCH(1,1) clustering scaled by regime.
            let market_factor = garch.step(regime.vol_multiplier(), &mut rng);

            // Sector factors, pulled toward the market factor by the
            // regime's contagion loading.
            let contagion = regime.contagion_loading();
            let sector_factors: Vec<f64> = (0..cfg.num_sectors)
                .map(|_| {
                    let raw = sector_factor_dist.sample(&mut rng);
                    (1.0 - contagion) * raw + contagion * market_factor
                })
                .collect();

            // Hot-sector factor, only while a regime has one active.
            let hot_factor = regime
                .hot_sector_index()
                .map(|_| sector_factor_dist.sample(&mut rng));

            // At most one sector-wide idiosyncratic shock per day.
            let shocked_sector = if rng.gen_bool(cfg.sector_shock_probability) {
                Some((
                    rng.gen_range(0..cfg.num_sectors),
                    shock_size_dist.sample(&mut rng),
                ))
            } else {
                None
            };

            let day_sigma = garch.sigma2.sqrt();
            let volume_dist = LogNormal::new(0.0, 0.3).unwrap();

            for asset_idx in 0..cfg.num_assets {
                let sector = sector_of_asset[asset_idx];
                let mut r =
                    cfg.beta_market * market_factor + cfg.beta_sector * sector_factors[sector];
                if regime.hot_sector_index() == Some(sector) {
                    r += cfg.beta_hot
                        * hot_factor.expect("hot_factor set when hot_sector_index is Some");
                }
                if let Some((shocked, magnitude)) = shocked_sector {
                    if shocked == sector {
                        r += magnitude;
                    }
                }
                r += idio_dist.sample(&mut rng);

                let open = prev_close[asset_idx];
                let close = open * r.exp();
                let range = day_sigma * cfg.intraday_range_factor * close;
                let extra_high = rng.gen_range(0.0..=range);
                let extra_low = rng.gen_range(0.0..=range);
                let high = open.max(close) + extra_high;
                let low = (open.min(close) - extra_low)
                    .max(close.min(open) * 0.5)
                    .max(0.01);
                let volume =
                    cfg.base_volume * (1.0 + r.abs() * 20.0) * volume_dist.sample(&mut rng);

                let bar = MarketBar {
                    timestamp,
                    symbol: symbols[asset_idx].clone(),
                    open,
                    high,
                    low,
                    close,
                    volume,
                };
                debug_assert!(
                    bar.validate().is_ok(),
                    "generator produced an invalid bar: {bar:?}"
                );
                bars.push(bar);

                prev_close[asset_idx] = close;
            }

            // A macro "market volatility" series, published with a lag —
            // exercises the point-in-time publication-lag path with a real
            // (not contrived) example: realized volatility is known
            // instantly to the generator but shouldn't be assumed knowable
            // to a model before some reporting delay.
            macro_observations.push(MacroObservation {
                series: MacroSeriesId::from("REALIZED_MARKET_VOL"),
                observation_time: timestamp,
                publication_time: timestamp + Duration::days(cfg.macro_publication_lag_days),
                value: day_sigma,
            });
        }

        SyntheticMarketDataset {
            bars,
            sector_assignment,
            regime_schedule,
            macro_observations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> SyntheticMarketConfig {
        SyntheticMarketConfig {
            num_assets: 20,
            num_sectors: 10,
            num_days: 600,
            ..SyntheticMarketConfig::default()
        }
    }

    fn full_config() -> SyntheticMarketConfig {
        SyntheticMarketConfig::default()
    }

    #[test]
    fn generation_is_deterministic_given_seed() {
        let cfg = small_config();
        let a = SyntheticMarketGenerator::new(cfg.clone()).generate();
        let b = SyntheticMarketGenerator::new(cfg).generate();
        assert_eq!(a.bars, b.bars);
        assert_eq!(a.regime_schedule, b.regime_schedule);
    }

    #[test]
    fn different_seeds_produce_different_data() {
        let mut cfg_a = small_config();
        let mut cfg_b = small_config();
        cfg_a.seed = 1;
        cfg_b.seed = 2;
        let a = SyntheticMarketGenerator::new(cfg_a).generate();
        let b = SyntheticMarketGenerator::new(cfg_b).generate();
        assert_ne!(a.bars, b.bars);
    }

    #[test]
    fn shapes_match_config() {
        let cfg = full_config();
        let dataset = SyntheticMarketGenerator::new(cfg.clone()).generate();
        assert_eq!(dataset.bars.len(), cfg.num_assets * cfg.num_days);
        assert_eq!(dataset.sector_assignment.len(), cfg.num_assets);
        assert_eq!(dataset.regime_schedule.len(), cfg.num_days);
        assert_eq!(dataset.macro_observations.len(), cfg.num_days);

        for sector in 0..cfg.num_sectors {
            let expected_sector = EntityId::from(format!("SECTOR{sector:02}"));
            let count = dataset
                .sector_assignment
                .iter()
                .filter(|(_, s)| *s == expected_sector)
                .count();
            assert_eq!(count, cfg.num_assets / cfg.num_sectors);
        }
    }

    #[test]
    fn every_generated_bar_is_valid() {
        let dataset = SyntheticMarketGenerator::new(small_config()).generate();
        for bar in &dataset.bars {
            assert!(
                bar.validate().is_ok(),
                "invalid bar: {bar:?}: {:?}",
                bar.validate()
            );
        }
    }

    #[test]
    fn macro_observations_carry_the_configured_publication_lag() {
        let cfg = small_config();
        let dataset = SyntheticMarketGenerator::new(cfg.clone()).generate();
        for obs in &dataset.macro_observations {
            let lag = obs.publication_time - obs.observation_time;
            assert_eq!(lag, Duration::days(cfg.macro_publication_lag_days));
            assert!(obs.validate().is_ok());
        }
    }

    /// Recovers per-asset daily log returns straight from the public
    /// `MarketBar` output (not internal generator state) — see the module
    /// doc's note on `open_t = close_{t-1}` making this exact.
    fn log_returns(bars: &[MarketBar], symbol: &Symbol) -> Vec<f64> {
        let mut closes: Vec<(Timestamp, f64)> = bars
            .iter()
            .filter(|b| &b.symbol == symbol)
            .map(|b| (b.timestamp, b.close))
            .collect();
        closes.sort_by_key(|(t, _)| *t);
        closes.windows(2).map(|w| (w[1].1 / w[0].1).ln()).collect()
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let n = a.len() as f64;
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;
        let cov: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - mean_a) * (y - mean_b))
            .sum::<f64>()
            / n;
        let std_a = (a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / n).sqrt();
        let std_b = (b.iter().map(|y| (y - mean_b).powi(2)).sum::<f64>() / n).sqrt();
        cov / (std_a * std_b)
    }

    fn mean_pairwise_correlation(bars: &[MarketBar], symbols: &[Symbol], day_mask: &[bool]) -> f64 {
        let series: Vec<Vec<f64>> = symbols
            .iter()
            .map(|s| {
                log_returns(bars, s)
                    .into_iter()
                    .zip(day_mask.iter().skip(1)) // returns start from day 1
                    .filter(|(_, &keep)| keep)
                    .map(|(r, _)| r)
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..series.len() {
            for j in (i + 1)..series.len() {
                sum += pearson(&series[i], &series[j]);
                count += 1;
            }
        }
        sum / count as f64
    }

    /// The core scientific-validation test for the generator (project spec
    /// §10: "The model should be tested on whether it can recover these
    /// known relationships. This is a scientific validation step.") —
    /// applied one level earlier, to the generator itself: does the "known
    /// hidden topology" the generator claims to produce actually show up in
    /// the realized data? If this test doesn't hold, no later topology
    /// model has a real signal to recover in the first place.
    ///
    /// This compares each hot sector against a **same-regime control
    /// sector** (sector 9, never hot) rather than across regimes directly.
    /// A first version of this test compared sector 0 across regimes and
    /// failed: `RiskOff`'s high contagion loading raises correlation
    /// *everywhere*, including in sector 0, enough to outweigh sector 0's
    /// own hot-factor boost from `RiskOn` — a real property of the
    /// generator, not a bug. Comparing against a control sector in the same
    /// regime isolates the hot-sector-specific effect from that generic,
    /// regime-wide contagion effect (which the contagion test below checks
    /// separately).
    #[test]
    fn hot_sector_correlation_exceeds_control_sector_in_its_own_regime() {
        let dataset = SyntheticMarketGenerator::new(full_config()).generate();
        let symbols_in = |sector: &str| -> Vec<Symbol> {
            dataset
                .sector_assignment
                .iter()
                .filter(|(_, s)| *s == EntityId::from(sector))
                .map(|(sym, _)| sym.clone())
                .collect()
        };
        let sector0_symbols = symbols_in("SECTOR00"); // hot in RiskOn
        let sector1_symbols = symbols_in("SECTOR01"); // hot in RiskOff
        let control_symbols = symbols_in("SECTOR09"); // never hot

        let riskon_mask: Vec<bool> = dataset
            .regime_schedule
            .iter()
            .map(|(_, r)| *r == MarketRegime::RiskOn)
            .collect();
        let riskoff_mask: Vec<bool> = dataset
            .regime_schedule
            .iter()
            .map(|(_, r)| *r == MarketRegime::RiskOff)
            .collect();

        let sector0_riskon =
            mean_pairwise_correlation(&dataset.bars, &sector0_symbols, &riskon_mask);
        let sector1_riskoff =
            mean_pairwise_correlation(&dataset.bars, &sector1_symbols, &riskoff_mask);
        let control_riskon =
            mean_pairwise_correlation(&dataset.bars, &control_symbols, &riskon_mask);
        let control_riskoff =
            mean_pairwise_correlation(&dataset.bars, &control_symbols, &riskoff_mask);

        assert!(
            sector0_riskon > control_riskon + 0.1,
            "sector0 RiskOn corr {sector0_riskon} not meaningfully above control {control_riskon}"
        );
        assert!(
            sector1_riskoff > control_riskoff + 0.1,
            "sector1 RiskOff corr {sector1_riskoff} not meaningfully above control {control_riskoff}"
        );

        // And each hot sector should lose its edge over the control once
        // its own regime isn't active — sector 0 has no hot factor in
        // RiskOff, so it should look like an ordinary sector there.
        let sector0_riskoff =
            mean_pairwise_correlation(&dataset.bars, &sector0_symbols, &riskoff_mask);
        assert!(
            sector0_riskoff - control_riskoff < sector0_riskon - control_riskon,
            "sector0's excess correlation over control should shrink outside RiskOn: \
             riskon excess {}, riskoff excess {}",
            sector0_riskon - control_riskon,
            sector0_riskoff - control_riskoff
        );
    }

    /// Cross-sector contagion: average pairwise correlation *between*
    /// different sectors' assets should be visibly higher in RiskOff (high
    /// contagion loading) than in Neutral (baseline contagion loading).
    #[test]
    fn cross_sector_correlation_rises_sharply_in_riskoff() {
        let dataset = SyntheticMarketGenerator::new(full_config()).generate();
        let sector2_symbols: Vec<Symbol> = dataset
            .sector_assignment
            .iter()
            .filter(|(_, s)| *s == EntityId::from("SECTOR02"))
            .map(|(sym, _)| sym.clone())
            .collect();
        let sector3_symbols: Vec<Symbol> = dataset
            .sector_assignment
            .iter()
            .filter(|(_, s)| *s == EntityId::from("SECTOR03"))
            .map(|(sym, _)| sym.clone())
            .collect();

        let neutral_mask: Vec<bool> = dataset
            .regime_schedule
            .iter()
            .map(|(_, r)| *r == MarketRegime::Neutral)
            .collect();
        let riskoff_mask: Vec<bool> = dataset
            .regime_schedule
            .iter()
            .map(|(_, r)| *r == MarketRegime::RiskOff)
            .collect();

        let cross_corr = |mask: &[bool]| -> f64 {
            let series2: Vec<Vec<f64>> = sector2_symbols
                .iter()
                .map(|s| {
                    log_returns(&dataset.bars, s)
                        .into_iter()
                        .zip(mask.iter().skip(1))
                        .filter(|(_, &keep)| keep)
                        .map(|(r, _)| r)
                        .collect::<Vec<_>>()
                })
                .collect();
            let series3: Vec<Vec<f64>> = sector3_symbols
                .iter()
                .map(|s| {
                    log_returns(&dataset.bars, s)
                        .into_iter()
                        .zip(mask.iter().skip(1))
                        .filter(|(_, &keep)| keep)
                        .map(|(r, _)| r)
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut sum = 0.0;
            let mut count = 0;
            for a in &series2 {
                for b in &series3 {
                    sum += pearson(a, b);
                    count += 1;
                }
            }
            sum / count as f64
        };

        let neutral_cross = cross_corr(&neutral_mask);
        let riskoff_cross = cross_corr(&riskoff_mask);
        assert!(
            riskoff_cross > neutral_cross + 0.1,
            "RiskOff cross-sector corr {riskoff_cross} not meaningfully above Neutral {neutral_cross}"
        );
    }

    /// Volatility clustering: squared market-wide average daily return
    /// should show positive autocorrelation at lag 1 — the hallmark GARCH
    /// signature ("big moves followed by big moves").
    #[test]
    fn market_volatility_shows_clustering() {
        let dataset = SyntheticMarketGenerator::new(full_config()).generate();
        let cfg = full_config();

        // Market-wide average return per day, from the raw bars.
        let mut by_day: Vec<Vec<f64>> = vec![vec![]; cfg.num_days];
        let mut symbol_to_returns = std::collections::HashMap::new();
        for (sym, _) in &dataset.sector_assignment {
            symbol_to_returns.insert(sym.clone(), log_returns(&dataset.bars, sym));
        }
        for returns in symbol_to_returns.values() {
            for (day, &r) in returns.iter().enumerate() {
                by_day[day + 1].push(r);
            }
        }
        let market_return: Vec<f64> = by_day
            .iter()
            .skip(1)
            .map(|day_returns| day_returns.iter().sum::<f64>() / day_returns.len() as f64)
            .collect();
        let squared: Vec<f64> = market_return.iter().map(|r| r * r).collect();

        let lag1 = pearson(&squared[..squared.len() - 1], &squared[1..]);
        assert!(
            lag1 > 0.05,
            "expected positive lag-1 autocorrelation of squared returns, got {lag1}"
        );
    }
}
