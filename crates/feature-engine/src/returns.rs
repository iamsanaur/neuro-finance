//! Log returns and rolling (multi-period) returns / momentum.

use crate::rolling::rolling_apply;

/// Single-period log returns: `ln(values[i] / values[i-1])` for `i >= 1`.
/// Output has length `values.len() - 1` (or `0` if input has fewer than 2
/// points) — there is no return defined for the first observation, and this
/// function does not invent one, unlike the `Option`-padded rolling
/// functions below.
pub fn log_returns(values: &[f64]) -> Vec<f64> {
    values.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
}

/// Cumulative log return over a trailing `window`-period span ending at
/// each index of `returns` (a series of single-period log returns, e.g.
/// from [`log_returns`]). `min_periods` behaves as in
/// [`crate::rolling::rolling_apply`] — note this differs from a price-based
/// "N-day return" by one period, since it operates on an already-differenced
/// return series; see [`momentum`] for the price-based framing.
pub fn rolling_return(returns: &[f64], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    rolling_apply(returns, window, min_periods, |w| w.iter().sum())
}

/// `N`-period momentum, computed directly from a price series: `ln(prices[i]
/// / prices[i-window])`, `None` until `window` full periods of price history
/// exist at `i` (equivalently, `min_periods = window` — momentum has no
/// meaningful "partial window" the way a moving average does, so unlike
/// [`rolling_return`] there is no separate `min_periods` parameter here).
///
/// This is arithmetically the trailing sum of single-period log returns
/// over the same span — the two are documented separately because the spec
/// (§11) lists "rolling returns" and "momentum" as distinct named features,
/// but they are the same construct under the hood; this function exists so
/// callers can go straight from prices without hand-computing
/// [`log_returns`] first.
pub fn momentum(prices: &[f64], window: usize) -> Vec<Option<f64>> {
    prices
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            if i >= window {
                Some((p / prices[i - window]).ln())
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
    fn log_returns_basic() {
        let prices = vec![100.0, 110.0, 99.0];
        let returns = log_returns(&prices);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (110.0_f64 / 100.0).ln()).abs() < 1e-12);
        assert!((returns[1] - (99.0_f64 / 110.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn momentum_matches_sum_of_log_returns() {
        let prices = vec![100.0, 105.0, 98.0, 120.0, 130.0];
        let window = 3;
        let mom = momentum(&prices, window);
        let returns = log_returns(&prices);
        let rr = rolling_return(&returns, window, window);

        // momentum[i] for i >= window should equal rolling_return computed
        // on the return series at index i-1 (returns are one shorter, and
        // offset by one relative to prices).
        for i in window..prices.len() {
            let expected = rr[i - 1].unwrap();
            assert!(
                (mom[i].unwrap() - expected).abs() < 1e-9,
                "mismatch at i={i}"
            );
        }
    }

    #[test]
    fn momentum_is_none_before_window_is_full() {
        let prices = vec![100.0, 105.0, 98.0];
        let mom = momentum(&prices, 5);
        assert_eq!(mom, vec![None, None, None]);
    }

    #[test]
    fn prefix_computation_matches_full_computation() {
        let prices = vec![10.0, 10.5, 9.8, 11.0, 12.0, 11.5, 13.0];
        let full = momentum(&prices, 2);
        for prefix_len in 1..=prices.len() {
            let prefix = momentum(&prices[..prefix_len], 2);
            assert_eq!(
                prefix,
                full[..prefix_len],
                "mismatch at prefix_len={prefix_len}"
            );
        }
    }
}
