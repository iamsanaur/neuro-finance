//! Baseline models (project spec §28): the reference points
//! `neuro-model`'s accuracy has to actually beat before any claim of
//! predictive value means anything. "All models must use equivalent
//! information" (§28) — every baseline here sees exactly the class-index
//! history a real caller would have as of each prediction day, nothing
//! more.
//!
//! V0.1 implements the two simplest baselines from §28's list (naive,
//! majority-class — a variant of "naive" for an unbalanced label
//! distribution) plus logistic regression. Gradient boosting, MLP,
//! LSTM/GRU, Transformer, static GNN, and graph Transformer baselines are
//! explicitly deferred: each is a real, separate implementation effort, and
//! per §2's incremental-build principle, they're only worth adding once the
//! two cheapest baselines have actually established whether `neuro-model`
//! clears even the lowest bar. A model that can't beat "predict yesterday's
//! class" has no use for a GBM comparison yet.

use std::collections::HashMap;

/// Predicts whatever class was seen last (project spec §28: "naive
/// baseline"). Financial regimes are persistent (this project's own
/// synthetic generator gives them multi-day runs — see
/// `data-engine::synthetic::regime::RegimeTransition`), so "no change" is a
/// genuinely competitive baseline here, not a strawman.
#[derive(Debug, Default, Clone)]
pub struct NaivePersistenceBaseline {
    last_seen: Option<usize>,
}

impl NaivePersistenceBaseline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Predicts the last class observed via [`NaivePersistenceBaseline::observe`],
    /// or `fallback` if nothing has been observed yet.
    pub fn predict(&self, fallback: usize) -> usize {
        self.last_seen.unwrap_or(fallback)
    }

    /// Records the true class for a day, for use in the *next* prediction —
    /// callers must call this only with the actual label, after their own
    /// prediction for that day, never before (this is what keeps the
    /// baseline point-in-time-safe: it can echo yesterday, never today).
    pub fn observe(&mut self, true_class: usize) {
        self.last_seen = Some(true_class);
    }
}

/// Predicts whatever class was most frequent in a fixed training set — a
/// second "naive" variant that accounts for class imbalance (persistence
/// alone can look artificially strong on a heavily imbalanced label
/// distribution even when it captures nothing about *changes*; majority-
/// class is the complementary check).
#[derive(Debug, Clone)]
pub struct MajorityClassBaseline {
    majority_class: usize,
}

impl MajorityClassBaseline {
    /// Fit once, on a training set of class indices (e.g. the walk-forward
    /// train window's labels only — this must never be fit on data
    /// including the evaluation window).
    pub fn fit(train_classes: &[usize]) -> Self {
        assert!(
            !train_classes.is_empty(),
            "cannot fit a majority-class baseline on zero examples"
        );
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &c in train_classes {
            *counts.entry(c).or_insert(0) += 1;
        }
        let majority_class = *counts
            .iter()
            .max_by_key(|(_, count)| **count)
            .map(|(class, _)| class)
            .unwrap();
        Self { majority_class }
    }

    pub fn predict(&self) -> usize {
        self.majority_class
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_persistence_echoes_last_observed_class() {
        let mut baseline = NaivePersistenceBaseline::new();
        assert_eq!(baseline.predict(1), 1); // nothing observed yet, uses fallback
        baseline.observe(0);
        assert_eq!(baseline.predict(1), 0);
        baseline.observe(2);
        assert_eq!(baseline.predict(1), 2);
    }

    #[test]
    fn majority_class_picks_the_most_frequent_label() {
        let baseline = MajorityClassBaseline::fit(&[0, 0, 1, 0, 2, 0]);
        assert_eq!(baseline.predict(), 0);
    }

    #[test]
    #[should_panic(expected = "zero examples")]
    fn majority_class_rejects_empty_training_set() {
        MajorityClassBaseline::fit(&[]);
    }
}
