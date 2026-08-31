//! `WalkForwardValidator` (project spec §29): the only sanctioned way this
//! project splits data into train/validation/test — never a random split
//! (§29: "Never use random train/test splitting for the primary financial
//! evaluation").
//!
//! Each [`WalkForwardSplit`] carries three non-overlapping, chronologically
//! ordered windows, each separated from the next by an `embargo` gap:
//!
//! ```text
//! [------- train -------][embargo][-- validation --][embargo][-- test --]
//! ```
//!
//! The embargo exists because a rolling feature computed *inside* the
//! validation window can still depend on the last few days of the training
//! window (e.g. a 20-day rolling volatility computed on validation day 1
//! reaches back into training data) — the embargo gap is what prevents that
//! near-boundary leakage from ever mattering, by simply not evaluating on
//! days close enough to the boundary for it to be a factor. Successive
//! splits are produced by rolling the whole train/validation/test block
//! forward by `test_period` each time — this project always evaluates on
//! a full test window before ever seeing the next one, unlike the "roll by
//! one day" walk-forward variant used in some literature.
//!
//! `train_years`/`validation_years`/`test_years`/`embargo_days` (as they
//! appear in `configs/default.toml`) map onto `train_period` etc. here as
//! `Duration::days(365 * years)` — an approximation (no leap years, no
//! trading-calendar awareness), acceptable for V0.1's synthetic daily data
//! and explicitly not assumed calendar-exact.

use chrono::Duration;
use financial_types::Timestamp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    /// Train window always starts at the validator's `start` and grows
    /// with each successive split.
    Expanding,
    /// Train window has a fixed length (`train_period`), sliding forward
    /// with each split.
    Rolling,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WalkForwardSplit {
    /// `(start, end)`, both inclusive-in-spirit (i.e. the last usable
    /// timestamp is `< end`, not `<= end`, when slicing a dataset — see
    /// this module's tests for the exact convention used).
    pub train: (Timestamp, Timestamp),
    pub validation: (Timestamp, Timestamp),
    pub test: (Timestamp, Timestamp),
}

#[derive(Debug, Clone)]
pub struct WalkForwardValidator {
    train_period: Duration,
    validation_period: Duration,
    test_period: Duration,
    embargo: Duration,
    mode: WindowMode,
}

impl WalkForwardValidator {
    pub fn new(
        train_period: Duration,
        validation_period: Duration,
        test_period: Duration,
        embargo: Duration,
        mode: WindowMode,
    ) -> Self {
        assert!(
            train_period > Duration::zero(),
            "train_period must be positive"
        );
        assert!(
            validation_period > Duration::zero(),
            "validation_period must be positive"
        );
        assert!(
            test_period > Duration::zero(),
            "test_period must be positive"
        );
        assert!(embargo >= Duration::zero(), "embargo must be non-negative");
        Self {
            train_period,
            validation_period,
            test_period,
            embargo,
            mode,
        }
    }

    /// Convenience constructor matching `configs/default.toml`'s
    /// `[walk_forward]` section field names directly.
    pub fn from_years(
        train_years: u32,
        validation_years: u32,
        test_years: u32,
        embargo_days: u32,
        mode: WindowMode,
    ) -> Self {
        Self::new(
            Duration::days(365 * i64::from(train_years)),
            Duration::days(365 * i64::from(validation_years)),
            Duration::days(365 * i64::from(test_years)),
            Duration::days(i64::from(embargo_days)),
            mode,
        )
    }

    /// All splits fully contained within `[start, end]`. Stops as soon as a
    /// split's test window would extend past `end` — a partial trailing
    /// split is never returned, since an incomplete test window would give
    /// a biased (and this project would rather be explicit than silently
    /// underpowered) evaluation.
    pub fn splits(&self, start: Timestamp, end: Timestamp) -> Vec<WalkForwardSplit> {
        let mut result = Vec::new();
        let mut train_end = start + self.train_period;

        loop {
            let train_start = match self.mode {
                WindowMode::Expanding => start,
                WindowMode::Rolling => train_end - self.train_period,
            };
            let validation_start = train_end + self.embargo;
            let validation_end = validation_start + self.validation_period;
            let test_start = validation_end + self.embargo;
            let test_end = test_start + self.test_period;

            if test_end > end {
                break;
            }

            result.push(WalkForwardSplit {
                train: (train_start, train_end),
                validation: (validation_start, validation_end),
                test: (test_start, test_end),
            });

            train_end += self.test_period;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn day(n: i64) -> Timestamp {
        Utc.with_ymd_and_hms(2015, 1, 1, 0, 0, 0).unwrap() + Duration::days(n)
    }

    #[test]
    fn windows_are_chronologically_ordered_and_embargoed() {
        let validator = WalkForwardValidator::new(
            Duration::days(100),
            Duration::days(20),
            Duration::days(20),
            Duration::days(5),
            WindowMode::Expanding,
        );
        let splits = validator.splits(day(0), day(400));
        assert!(!splits.is_empty());
        for split in &splits {
            assert!(split.train.0 < split.train.1);
            assert!(split.train.1 + Duration::days(5) <= split.validation.0);
            assert!(split.validation.0 < split.validation.1);
            assert!(split.validation.1 + Duration::days(5) <= split.test.0);
            assert!(split.test.0 < split.test.1);
        }
    }

    #[test]
    fn successive_splits_roll_forward_by_test_period() {
        let validator = WalkForwardValidator::new(
            Duration::days(100),
            Duration::days(20),
            Duration::days(20),
            Duration::days(0),
            WindowMode::Expanding,
        );
        let splits = validator.splits(day(0), day(400));
        assert!(splits.len() >= 2);
        for pair in splits.windows(2) {
            assert_eq!(pair[1].train.1, pair[0].train.1 + Duration::days(20));
        }
    }

    #[test]
    fn expanding_mode_grows_train_window_each_split() {
        let validator = WalkForwardValidator::new(
            Duration::days(100),
            Duration::days(20),
            Duration::days(20),
            Duration::days(0),
            WindowMode::Expanding,
        );
        let splits = validator.splits(day(0), day(400));
        assert!(splits.len() >= 2);
        for pair in splits.windows(2) {
            assert_eq!(pair[0].train.0, pair[1].train.0); // same start
            assert!(pair[1].train.1 - pair[1].train.0 > pair[0].train.1 - pair[0].train.0);
            // grew
        }
    }

    #[test]
    fn rolling_mode_keeps_train_window_length_fixed() {
        let validator = WalkForwardValidator::new(
            Duration::days(100),
            Duration::days(20),
            Duration::days(20),
            Duration::days(0),
            WindowMode::Rolling,
        );
        let splits = validator.splits(day(0), day(400));
        assert!(splits.len() >= 2);
        for split in &splits {
            assert_eq!(split.train.1 - split.train.0, Duration::days(100));
        }
        // And it actually slides (doesn't stay pinned at the same start).
        assert_ne!(splits[0].train.0, splits[1].train.0);
    }

    #[test]
    fn no_trailing_partial_split_is_returned() {
        let validator = WalkForwardValidator::new(
            Duration::days(100),
            Duration::days(20),
            Duration::days(20),
            Duration::days(0),
            WindowMode::Expanding,
        );
        let splits = validator.splits(day(0), day(400));
        for split in &splits {
            assert!(split.test.1 <= day(400));
        }
    }

    #[test]
    fn from_years_matches_configs_default_toml_semantics() {
        // configs/default.toml: train_years=8, validation_years=1,
        // test_years=1, embargo_days=5.
        let validator = WalkForwardValidator::from_years(8, 1, 1, 5, WindowMode::Expanding);
        let splits = validator.splits(day(0), day(365 * 12));
        assert!(!splits.is_empty());
        let first = &splits[0];
        assert_eq!(first.train.1 - first.train.0, Duration::days(365 * 8));
        assert_eq!(first.validation.1 - first.validation.0, Duration::days(365));
        assert_eq!(first.test.1 - first.test.0, Duration::days(365));
    }

    #[test]
    #[should_panic(expected = "train_period must be positive")]
    fn rejects_zero_train_period() {
        WalkForwardValidator::new(
            Duration::zero(),
            Duration::days(1),
            Duration::days(1),
            Duration::days(0),
            WindowMode::Expanding,
        );
    }
}
