//! Early stopping on validation loss (project spec §31).

/// Tracks the best validation loss seen so far and how many consecutive
/// checks have failed to improve on it. Call [`EarlyStopping::step`] once
/// per validation check (typically once per epoch); when it returns `true`,
/// training should stop.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    patience: usize,
    /// Minimum improvement over the best loss to count as "improved" —
    /// without this, floating-point noise around a plateau could reset the
    /// patience counter indefinitely and stopping would never trigger.
    min_delta: f64,
    best_loss: f64,
    strikes: usize,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        assert!(patience > 0, "patience must be positive");
        assert!(min_delta >= 0.0, "min_delta must be non-negative");
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            strikes: 0,
        }
    }

    /// Records one validation loss observation. Returns `true` if training
    /// should stop now (patience exhausted).
    pub fn step(&mut self, validation_loss: f64) -> bool {
        if validation_loss < self.best_loss - self.min_delta {
            self.best_loss = validation_loss;
            self.strikes = 0;
        } else {
            self.strikes += 1;
        }
        self.strikes >= self.patience
    }

    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn does_not_stop_while_improving() {
        let mut es = EarlyStopping::new(3, 0.0);
        for loss in [1.0, 0.8, 0.6, 0.4, 0.2] {
            assert!(!es.step(loss));
        }
    }

    #[test]
    fn stops_after_patience_exhausted() {
        let mut es = EarlyStopping::new(2, 0.0);
        assert!(!es.step(1.0)); // improves (best=inf -> 1.0)
        assert!(!es.step(1.1)); // strike 1
        assert!(es.step(1.2)); // strike 2 == patience -> stop
    }

    #[test]
    fn improvement_resets_strike_counter() {
        let mut es = EarlyStopping::new(2, 0.0);
        assert!(!es.step(1.0));
        assert!(!es.step(1.1)); // strike 1
        assert!(!es.step(0.5)); // improves, resets to 0 strikes
        assert!(!es.step(0.6)); // strike 1 (not 3 — the reset above is what's being tested)
        assert!(es.step(0.7)); // strike 2 == patience -> stop
    }

    #[test]
    fn min_delta_prevents_noise_from_counting_as_improvement() {
        let mut es = EarlyStopping::new(2, 0.05);
        assert!(!es.step(1.0));
        assert!(!es.step(0.99)); // improvement of 0.01 < min_delta 0.05 -> strike 1
        assert!(es.step(0.98)); // still under min_delta -> strike 2 -> stop
    }

    #[test]
    fn best_loss_tracks_the_minimum_seen() {
        let mut es = EarlyStopping::new(5, 0.0);
        for loss in [1.0, 0.5, 0.8, 0.3, 0.9] {
            es.step(loss);
        }
        assert!((es.best_loss() - 0.3).abs() < 1e-12);
    }
}
