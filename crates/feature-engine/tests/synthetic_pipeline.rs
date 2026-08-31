//! Integration test: synthetic market bars -> feature-engine, end to end.
//!
//! Not a substitute for the full synthetic end-to-end test project spec §48
//! calls for (that also needs the graph, topology, and model layers, which
//! don't exist yet) — but it is the first real check that `data-engine`'s
//! output and `feature-engine`'s input actually fit together, and that
//! causality survives a real (not hand-crafted) price series.

use data_engine::synthetic::{SyntheticMarketConfig, SyntheticMarketGenerator};
use feature_engine::{
    close_series, drawdown, log_returns, momentum, moving_average, rolling_volatility,
};
use financial_types::Symbol;

#[test]
fn features_compute_cleanly_over_a_real_synthetic_asset() {
    let config = SyntheticMarketConfig {
        num_days: 300,
        ..SyntheticMarketConfig::default()
    };
    let dataset = SyntheticMarketGenerator::new(config).generate();
    let symbol = Symbol::from("AST0000");

    let (timestamps, closes) = close_series(&dataset.bars, &symbol);
    assert_eq!(timestamps.len(), 300);

    let returns = log_returns(&closes);
    assert_eq!(returns.len(), closes.len() - 1);
    assert!(returns.iter().all(|r| r.is_finite()));

    let vol = rolling_volatility(&returns, 20, 5);
    assert!(vol.iter().flatten().all(|v| v.is_finite() && *v >= 0.0));
    // Once past min_periods, every value should be Some.
    assert!(vol[19..].iter().all(Option::is_some));

    let sma = moving_average(&closes, 20, 5);
    assert!(sma[19..].iter().all(Option::is_some));

    let mom = momentum(&closes, 20);
    assert!(mom[20..].iter().all(Option::is_some));
    assert!(mom[..20].iter().all(Option::is_none));

    let dd = drawdown(&closes);
    assert!(dd.iter().all(|d| *d <= 0.0 && d.is_finite()));
}

/// The same prefix/full-series causality property tested per-function in
/// unit tests, now over real generator output rather than hand-built
/// fixtures — the generator's own randomness (GARCH clustering, regime
/// switches, sector shocks) is exactly the kind of irregular series most
/// likely to expose an off-by-one leak that a smooth hand-crafted fixture
/// would hide.
#[test]
fn causality_holds_on_real_synthetic_data() {
    let config = SyntheticMarketConfig {
        num_days: 200,
        ..SyntheticMarketConfig::default()
    };
    let dataset = SyntheticMarketGenerator::new(config).generate();
    let symbol = Symbol::from("AST0005");
    let (_, closes) = close_series(&dataset.bars, &symbol);

    let full_sma = moving_average(&closes, 15, 3);
    let full_mom = momentum(&closes, 10);
    let full_dd = drawdown(&closes);

    for prefix_len in [1, 5, 20, 50, 100, closes.len()] {
        let prefix = &closes[..prefix_len];
        assert_eq!(moving_average(prefix, 15, 3), full_sma[..prefix_len]);
        assert_eq!(momentum(prefix, 10), full_mom[..prefix_len]);
        assert_eq!(drawdown(prefix), full_dd[..prefix_len]);
    }
}
