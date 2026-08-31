//! Volume-derived features: volume change and a dollar-volume liquidity
//! proxy.

/// Single-period log change in volume: `ln(volumes[i] / volumes[i-1])`.
/// Same shape as [`crate::returns::log_returns`] (one shorter than the
/// input) — kept as a separate, clearly-named function rather than reusing
/// `log_returns` silently, since "a return" and "a volume change" are
/// different concepts that happen to share a formula.
pub fn volume_change(volumes: &[f64]) -> Vec<f64> {
    volumes.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
}

/// Dollar volume (`close * volume`) as a liquidity proxy — no window, no
/// history required, so there is no `min_periods`/`window` here; every
/// index gets a value.
pub fn dollar_volume(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    assert_eq!(
        closes.len(),
        volumes.len(),
        "closes and volumes must be the same length"
    );
    closes.iter().zip(volumes).map(|(c, v)| c * v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_change_basic() {
        let volumes = vec![1_000.0, 1_500.0, 750.0];
        let changes = volume_change(&volumes);
        assert!((changes[0] - (1_500.0_f64 / 1_000.0).ln()).abs() < 1e-12);
        assert!((changes[1] - (750.0_f64 / 1_500.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn dollar_volume_basic() {
        let closes = vec![10.0, 20.0];
        let volumes = vec![100.0, 50.0];
        assert_eq!(dollar_volume(&closes, &volumes), vec![1_000.0, 1_000.0]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn dollar_volume_rejects_mismatched_lengths() {
        dollar_volume(&[1.0, 2.0], &[1.0]);
    }
}
