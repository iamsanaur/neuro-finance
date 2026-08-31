//! Rolling pairwise correlation between two return series.
//!
//! This is the feature that will drive `financial-graph`'s correlation
//! graph (§14) once that crate is implemented — kept here rather than
//! there, since it's a windowing/statistics concern like every other
//! feature in this crate, not a graph-construction concern.

use crate::rolling::mean;

/// Trailing Pearson correlation of `a` and `b` (equal length) over `window`,
/// `None` until `min_periods` (>= 2) observations are available at a given
/// index in *both* series.
pub fn rolling_correlation(
    a: &[f64],
    b: &[f64],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    assert_eq!(a.len(), b.len(), "a and b must be the same length");
    assert!(
        min_periods >= 2,
        "rolling_correlation requires min_periods >= 2, got {min_periods}"
    );
    assert!(window > 0, "window must be positive");
    assert!(
        min_periods <= window,
        "min_periods ({min_periods}) must be <= window ({window})"
    );

    (0..a.len())
        .map(|i| {
            let start = (i + 1).saturating_sub(window);
            let wa = &a[start..=i];
            let wb = &b[start..=i];
            if wa.len() >= min_periods {
                Some(pearson(wa, wb))
            } else {
                None
            }
        })
        .collect()
}

/// The underlying Pearson correlation computation `rolling_correlation`
/// windows over. Exposed directly (not just through the rolling wrapper) for
/// callers that already have a single window of data in hand and want to
/// avoid `rolling_correlation`'s O(n) computation over every trailing index
/// when only the last one is needed — e.g. `financial-graph`'s correlation
/// graph builder, which only ever wants "the" correlation as of one point in
/// time, not a full rolling series.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = mean(a);
    let mean_b = mean(b);
    let cov: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
    let var_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum();
    cov / (var_a.sqrt() * var_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfectly_correlated_series() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = rolling_correlation(&a, &b, 5, 2);
        assert!((corr[4].unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn perfectly_anti_correlated_series() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = rolling_correlation(&a, &b, 5, 2);
        assert!((corr[4].unwrap() + 1.0).abs() < 1e-9);
    }

    #[test]
    fn prefix_computation_matches_full_computation() {
        let a = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.5];
        let b = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 3.5];
        let full = rolling_correlation(&a, &b, 3, 2);
        for prefix_len in 1..=a.len() {
            let prefix = rolling_correlation(&a[..prefix_len], &b[..prefix_len], 3, 2);
            assert_eq!(
                prefix,
                full[..prefix_len],
                "mismatch at prefix_len={prefix_len}"
            );
        }
    }
}
