//! Rolling market beta: `cov(asset, market) / var(market)` over a trailing
//! window.
//!
//! The caller supplies the market return series (e.g. an equal-weighted
//! average across the universe) — this crate has no notion of "the
//! universe" itself, that's `data-engine`'s/`financial-graph`'s concern.

use crate::rolling::mean;

/// `None` until `min_periods` (>= 2) observations are available.
pub fn rolling_beta(
    asset_returns: &[f64],
    market_returns: &[f64],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    assert_eq!(
        asset_returns.len(),
        market_returns.len(),
        "asset_returns and market_returns must be the same length"
    );
    assert!(
        min_periods >= 2,
        "rolling_beta requires min_periods >= 2, got {min_periods}"
    );
    assert!(window > 0, "window must be positive");
    assert!(
        min_periods <= window,
        "min_periods ({min_periods}) must be <= window ({window})"
    );

    (0..asset_returns.len())
        .map(|i| {
            let start = (i + 1).saturating_sub(window);
            let wa = &asset_returns[start..=i];
            let wm = &market_returns[start..=i];
            if wa.len() >= min_periods {
                let mean_a = mean(wa);
                let mean_m = mean(wm);
                let cov: f64 = wa
                    .iter()
                    .zip(wm)
                    .map(|(x, y)| (x - mean_a) * (y - mean_m))
                    .sum();
                let var_m: f64 = wm.iter().map(|y| (y - mean_m).powi(2)).sum();
                Some(cov / var_m)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_of_market_with_itself_is_one() {
        let market = vec![0.01, -0.02, 0.03, 0.01, -0.01];
        let beta = rolling_beta(&market, &market, 5, 2);
        assert!((beta[4].unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn double_leveraged_asset_has_beta_two() {
        let market = vec![0.01, -0.02, 0.03, 0.01, -0.01];
        let asset: Vec<f64> = market.iter().map(|r| r * 2.0).collect();
        let beta = rolling_beta(&asset, &market, 5, 2);
        assert!((beta[4].unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn prefix_computation_matches_full_computation() {
        let asset = vec![0.02, -0.01, 0.03, 0.00, 0.04, -0.02, 0.01];
        let market = vec![0.01, -0.02, 0.02, 0.01, 0.03, -0.01, 0.005];
        let full = rolling_beta(&asset, &market, 4, 2);
        for prefix_len in 1..=asset.len() {
            let prefix = rolling_beta(&asset[..prefix_len], &market[..prefix_len], 4, 2);
            assert_eq!(
                prefix,
                full[..prefix_len],
                "mismatch at prefix_len={prefix_len}"
            );
        }
    }
}
