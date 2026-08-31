//! Rolling volatility (sample standard deviation of returns).

use crate::rolling::{rolling_apply, sample_std};

/// Rolling sample standard deviation of a return series. `min_periods` must
/// be `>= 2` (a standard deviation is undefined for a single observation);
/// this is asserted, not silently clamped, so a misconfigured caller finds
/// out immediately rather than getting a suspiciously-early first value.
pub fn rolling_volatility(returns: &[f64], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    assert!(
        min_periods >= 2,
        "rolling_volatility requires min_periods >= 2, got {min_periods}"
    );
    rolling_apply(returns, window, min_periods, sample_std)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::returns::log_returns;

    #[test]
    fn constant_returns_have_zero_volatility() {
        let returns = vec![0.01; 10];
        let vol = rolling_volatility(&returns, 5, 2);
        for v in vol.iter().skip(1) {
            assert!((v.unwrap()).abs() < 1e-12);
        }
    }

    #[test]
    fn volatility_reflects_dispersion() {
        let low_vol = vec![0.01, -0.01, 0.01, -0.01, 0.01, -0.01];
        let high_vol = vec![0.10, -0.10, 0.10, -0.10, 0.10, -0.10];
        let vol_low = rolling_volatility(&low_vol, 6, 6);
        let vol_high = rolling_volatility(&high_vol, 6, 6);
        assert!(vol_high[5].unwrap() > vol_low[5].unwrap());
    }

    #[test]
    #[should_panic(expected = "min_periods >= 2")]
    fn rejects_min_periods_below_two() {
        rolling_volatility(&[0.01, 0.02], 3, 1);
    }

    #[test]
    fn prefix_computation_matches_full_computation() {
        let prices = vec![100.0, 101.0, 99.0, 103.0, 98.0, 107.0, 110.0, 105.0];
        let returns = log_returns(&prices);
        let full = rolling_volatility(&returns, 4, 2);
        for prefix_len in 1..=returns.len() {
            let prefix = rolling_volatility(&returns[..prefix_len], 4, 2);
            assert_eq!(
                prefix,
                full[..prefix_len],
                "mismatch at prefix_len={prefix_len}"
            );
        }
    }
}
