//! The core causal-windowing primitive every rolling feature is built on.
//!
//! Project spec §11: "Every rolling operation must explicitly define:
//! window, alignment, minimum observations. No centered windows. No future
//! values."
//!
//! [`rolling_apply`] is the single place that discipline is enforced:
//! - **alignment**: the window ending at index `i` is
//!   `values[i + 1 - len .. = i]` — it never includes `values[j]` for
//!   `j > i`. This is trailing alignment; a centered window is not
//!   representable through this function at all.
//! - **window**: the maximum number of trailing observations used.
//! - **minimum observations**: `min_periods` — indices with fewer than
//!   `min_periods` values available so far produce `None` rather than a
//!   value computed from partial, potentially misleading data.
//!
//! Every other function in this crate is implemented in terms of this (or
//! follows the identical trailing-window discipline by hand, e.g.
//! [`crate::drawdown::drawdown`], which is an expanding rather than fixed
//! window but is equally causal). See `tests/causality.rs`-equivalent
//! property tests in each module: computing features on a prefix of a
//! series must reproduce, index-for-index, what computing on the full
//! series gives for those same indices — the concrete leakage test project
//! spec §30 asks for ("future rolling statistics").

/// Applies `f` to each trailing window of `values`, honoring `window` and
/// `min_periods`. Returns one `Option<f64>` per input value, in the same
/// order.
pub fn rolling_apply<F>(values: &[f64], window: usize, min_periods: usize, f: F) -> Vec<Option<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    assert!(window > 0, "window must be positive");
    assert!(min_periods > 0, "min_periods must be positive");
    assert!(
        min_periods <= window,
        "min_periods ({min_periods}) must be <= window ({window})"
    );

    values
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let start = (i + 1).saturating_sub(window);
            let slice = &values[start..=i];
            if slice.len() >= min_periods {
                Some(f(slice))
            } else {
                None
            }
        })
        .collect()
}

pub fn mean(slice: &[f64]) -> f64 {
    slice.iter().sum::<f64>() / slice.len() as f64
}

/// Sample standard deviation (ddof = 1). Callers must ensure `slice.len() >= 2`
/// — this crate never calls it with `min_periods < 2` for volatility-style
/// features (see [`crate::volatility::rolling_volatility`]).
pub fn sample_std(slice: &[f64]) -> f64 {
    let m = mean(slice);
    let n = slice.len() as f64;
    (slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_never_includes_future_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // f just returns the max of the window; if it ever saw a future
        // value this would exceed the running max seen so far.
        let result = rolling_apply(&values, 3, 1, |w| {
            w.iter().cloned().fold(f64::MIN, f64::max)
        });
        assert_eq!(
            result,
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]
        );
    }

    #[test]
    fn respects_window_size() {
        let values = vec![1.0, 1.0, 1.0, 100.0, 1.0, 1.0];
        // Sum over a window of 2: once the 100.0 falls out of the trailing
        // window, the sum must drop back down — proves old values are
        // correctly excluded once `window` is exceeded.
        let result = rolling_apply(&values, 2, 1, |w| w.iter().sum());
        assert_eq!(
            result,
            vec![
                Some(1.0),
                Some(2.0),
                Some(2.0),
                Some(101.0),
                Some(101.0),
                Some(2.0)
            ]
        );
    }

    #[test]
    fn below_min_periods_is_none() {
        let values = vec![1.0, 2.0, 3.0];
        let result = rolling_apply(&values, 3, 3, |w| w.iter().sum());
        assert_eq!(result, vec![None, None, Some(6.0)]);
    }

    /// The causality property test (project spec §30): computing on a
    /// prefix must reproduce, index-for-index, what computing on the full
    /// series gives for those same indices. If a future value ever leaked
    /// into a window, truncating the series would change an earlier result.
    #[test]
    fn prefix_computation_matches_full_computation() {
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let full = rolling_apply(&values, 3, 2, mean);
        for prefix_len in 1..=values.len() {
            let prefix = rolling_apply(&values[..prefix_len], 3, 2, mean);
            assert_eq!(
                prefix,
                full[..prefix_len],
                "mismatch at prefix_len={prefix_len}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "min_periods (5) must be <= window (3)")]
    fn rejects_min_periods_above_window() {
        rolling_apply(&[1.0, 2.0], 3, 5, mean);
    }

    #[test]
    fn sample_std_of_constant_series_is_zero() {
        assert_eq!(sample_std(&[2.0, 2.0, 2.0]), 0.0);
    }
}
