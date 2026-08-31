//! Z-score feature standardization — fit once on a training set, applied to
//! both train and held-out data with the same fitted parameters.
//!
//! This is the point-in-time-safe way to normalize (project spec §9, §30):
//! `fit` must only ever see the training window's values. Fitting on
//! validation/test data (or on train+test combined) would leak the
//! held-out distribution's mean/scale into training — a subtle form of
//! leakage this project's own first experiment (`exp-0001`) didn't guard
//! against, which is part of why its trainable models under-learned
//! (unnormalized features on wildly different scales — a return near
//! `0.01` next to a momentum value that can be an order of magnitude
//! larger — slow down gradient-based training).

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Standardizer {
    mean: f64,
    std: f64,
}

impl Standardizer {
    /// Fits `mean`/`std` from `values` — call this on training data only.
    /// A constant (zero-variance) input fits `std = 0.0`; see
    /// [`Standardizer::transform`] for how that's handled.
    pub fn fit(values: &[f64]) -> Self {
        assert!(
            !values.is_empty(),
            "cannot fit a standardizer on zero values"
        );
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        Self {
            mean,
            std: variance.sqrt(),
        }
    }

    /// `(value - mean) / std`. If `std == 0.0` (every fitted value was
    /// identical), returns `0.0` unconditionally rather than dividing by
    /// zero — a constant feature carries no discriminative information, so
    /// mapping it to a constant `0.0` is the correct behavior, not just a
    /// crash-avoidance hack.
    pub fn transform(&self, value: f64) -> f64 {
        if self.std == 0.0 {
            0.0
        } else {
            (value - self.mean) / self.std
        }
    }

    pub fn transform_all(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&v| self.transform(v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformed_train_data_has_zero_mean_and_unit_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let standardizer = Standardizer::fit(&values);
        let transformed = standardizer.transform_all(&values);
        let mean: f64 = transformed.iter().sum::<f64>() / transformed.len() as f64;
        let variance: f64 =
            transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / transformed.len() as f64;
        assert!(mean.abs() < 1e-9, "expected ~0 mean, got {mean}");
        assert!(
            (variance - 1.0).abs() < 1e-9,
            "expected ~1 variance, got {variance}"
        );
    }

    #[test]
    fn constant_input_maps_to_zero_without_dividing_by_zero() {
        let standardizer = Standardizer::fit(&[5.0, 5.0, 5.0]);
        assert_eq!(standardizer.transform(5.0), 0.0);
        assert_eq!(standardizer.transform(100.0), 0.0);
    }

    #[test]
    fn fitted_scale_applies_unchanged_to_unseen_values() {
        // Simulates the intended usage: fit on "train", apply to "test"
        // without refitting — the whole point of this type.
        let train = vec![10.0, 20.0, 30.0];
        let standardizer = Standardizer::fit(&train);
        let test_value = 25.0; // not in the fitted set
        let transformed = standardizer.transform(test_value);
        // mean=20, std=sqrt(((10-20)^2+(0)+(10)^2)/3)=sqrt(66.67)=8.165
        let expected = (25.0 - 20.0) / (200.0_f64 / 3.0).sqrt();
        assert!((transformed - expected).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "zero values")]
    fn rejects_empty_input() {
        Standardizer::fit(&[]);
    }
}
