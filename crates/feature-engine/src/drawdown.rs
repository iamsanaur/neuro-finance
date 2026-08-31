//! Drawdown from the running historical peak.
//!
//! Unlike the other features in this crate, drawdown is naturally an
//! **expanding**, not fixed, window: "peak so far" means all history up to
//! and including today, by definition. It is still fully causal — the
//! running max at index `i` is computed only from `values[0..=i]`, never
//! `values[j]` for `j > i` — it simply isn't expressed through
//! [`crate::rolling::rolling_apply`], which assumes a bounded window.

/// `(values[i] - running_max(values[0..=i])) / running_max(values[0..=i])`
/// — always `<= 0.0`. No `min_periods` concept applies: index `0`'s
/// drawdown is trivially `0.0` (it is its own running peak), so every index
/// produces a value.
pub fn drawdown(values: &[f64]) -> Vec<f64> {
    let mut running_max = f64::MIN;
    values
        .iter()
        .map(|&v| {
            running_max = running_max.max(v);
            (v - running_max) / running_max
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonically_increasing_series_has_zero_drawdown() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(drawdown(&values), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn drawdown_reflects_decline_from_peak() {
        let values = vec![100.0, 120.0, 90.0, 60.0, 110.0];
        let dd = drawdown(&values);
        assert_eq!(dd[0], 0.0);
        assert_eq!(dd[1], 0.0); // new peak
        assert!((dd[2] - (90.0 - 120.0) / 120.0).abs() < 1e-12);
        assert!((dd[3] - (60.0 - 120.0) / 120.0).abs() < 1e-12);
        // 110.0 is still below the 120.0 peak set at index 1, so drawdown
        // is still negative here, not reset to zero.
        assert!((dd[4] - (110.0 - 120.0) / 120.0).abs() < 1e-12);
    }

    #[test]
    fn recovering_above_prior_peak_resets_drawdown_to_zero() {
        let values = vec![100.0, 90.0, 130.0];
        let dd = drawdown(&values);
        assert_eq!(dd[2], 0.0);
    }

    #[test]
    fn prefix_computation_matches_full_computation() {
        let values = vec![50.0, 55.0, 48.0, 60.0, 40.0, 65.0, 62.0];
        let full = drawdown(&values);
        for prefix_len in 1..=values.len() {
            let prefix = drawdown(&values[..prefix_len]);
            assert_eq!(
                prefix,
                full[..prefix_len],
                "mismatch at prefix_len={prefix_len}"
            );
        }
    }
}
