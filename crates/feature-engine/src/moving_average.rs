//! Simple moving average.

use crate::rolling::{mean, rolling_apply};

/// Trailing simple moving average of `values` over `window`, `None` until
/// `min_periods` observations are available.
pub fn moving_average(values: &[f64], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    rolling_apply(values, window, min_periods, mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_average() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = moving_average(&values, 3, 3);
        assert_eq!(sma, vec![None, None, Some(2.0), Some(3.0), Some(4.0)]);
    }

    #[test]
    fn prefix_computation_matches_full_computation() {
        let values = vec![5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0];
        let full = moving_average(&values, 3, 1);
        for prefix_len in 1..=values.len() {
            let prefix = moving_average(&values[..prefix_len], 3, 1);
            assert_eq!(
                prefix,
                full[..prefix_len],
                "mismatch at prefix_len={prefix_len}"
            );
        }
    }
}
