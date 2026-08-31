//! Market regimes for the synthetic generator (project spec §10, §25).
//!
//! Three regimes are used, not the full 3–5 range the spec allows, because
//! these three double as the ground-truth label for the eventual regime
//! classification task (§25: `RiskOn` / `Neutral` / `RiskOff`) — reusing
//! them here means the synthetic data already exercises the exact target
//! space the first model milestone needs, instead of inventing a second,
//! unrelated regime taxonomy just for topology.
//!
//! Each regime carries two effects that together produce §10's "known
//! hidden topology":
//!
//! - a **hot sector**, whose member assets share an extra common factor in
//!   that regime only (so intra-sector correlation should visibly spike
//!   exactly when that regime is active, and only then), and
//! - a **contagion loading**, controlling how strongly every sector's
//!   factor gets pulled toward the shared market factor (so cross-sector
//!   correlation should visibly spike specifically in `RiskOff`).
//!
//! `RiskOn` → sector 0 hot (stands in for "tech-led"); `RiskOff` → sector 1
//! hot (stands in for "financials under stress") *and* high contagion;
//! `Neutral` → no hot sector, baseline contagion. This matches the spec's
//! illustrative example almost exactly (tech strongly connected in one
//! regime, financials in another, cross-sector connectivity rising sharply
//! in a third) while keeping the regime count minimal (YAGNI, project spec
//! §2).

use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    RiskOn,
    Neutral,
    RiskOff,
}

impl MarketRegime {
    pub const ALL: [MarketRegime; 3] = [
        MarketRegime::RiskOn,
        MarketRegime::Neutral,
        MarketRegime::RiskOff,
    ];

    /// Sector index (0-based) that carries an extra shared factor while this
    /// regime is active, or `None` for `Neutral`.
    pub fn hot_sector_index(self) -> Option<usize> {
        match self {
            MarketRegime::RiskOn => Some(0),
            MarketRegime::Neutral => None,
            MarketRegime::RiskOff => Some(1),
        }
    }

    /// Multiplier applied to baseline market volatility while this regime
    /// is active.
    pub fn vol_multiplier(self) -> f64 {
        match self {
            MarketRegime::RiskOn => 0.85,
            MarketRegime::Neutral => 1.0,
            MarketRegime::RiskOff => 1.9,
        }
    }

    /// How strongly every sector factor is pulled toward the shared market
    /// factor while this regime is active. High in `RiskOff` by design —
    /// this is the literal mechanism behind "cross-sector connectivity
    /// rises sharply" (project spec §10, regime C).
    pub fn contagion_loading(self) -> f64 {
        match self {
            MarketRegime::RiskOn => 0.05,
            MarketRegime::Neutral => 0.05,
            MarketRegime::RiskOff => 0.6,
        }
    }

    fn index(self) -> usize {
        match self {
            MarketRegime::RiskOn => 0,
            MarketRegime::Neutral => 1,
            MarketRegime::RiskOff => 2,
        }
    }
}

/// A Markov chain over [`MarketRegime`] with a configurable self-transition
/// (persistence) probability, split evenly between the two other regimes on
/// a transition. This is what produces realistic regime *duration* — a
/// day-by-day coin flip between three states with no persistence would
/// switch too fast to be a "regime" at all, and would give the hot-sector /
/// contagion tests below nothing to detect above noise.
#[derive(Debug, Clone, Copy)]
pub struct RegimeTransition {
    stay_probability: f64,
}

impl RegimeTransition {
    /// `stay_probability` is the probability of remaining in the current
    /// regime on the next step; must be in `(0.0, 1.0)`. The remaining
    /// probability mass is split evenly across the other two regimes.
    pub fn new(stay_probability: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&stay_probability),
            "stay_probability must be in [0, 1): got {stay_probability}"
        );
        Self { stay_probability }
    }

    pub fn next(self, current: MarketRegime, rng: &mut impl Rng) -> MarketRegime {
        let other_prob = (1.0 - self.stay_probability) / 2.0;
        let weights = MarketRegime::ALL.map(|r| {
            if r.index() == current.index() {
                self.stay_probability
            } else {
                other_prob
            }
        });
        let dist = WeightedIndex::new(weights).expect("weights are non-negative and sum > 0");
        MarketRegime::ALL[dist.sample(rng)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn hot_sectors_match_spec_example() {
        assert_eq!(MarketRegime::RiskOn.hot_sector_index(), Some(0));
        assert_eq!(MarketRegime::Neutral.hot_sector_index(), None);
        assert_eq!(MarketRegime::RiskOff.hot_sector_index(), Some(1));
    }

    #[test]
    fn riskoff_has_highest_contagion_and_volatility() {
        assert!(
            MarketRegime::RiskOff.contagion_loading() > MarketRegime::RiskOn.contagion_loading()
        );
        assert!(
            MarketRegime::RiskOff.contagion_loading() > MarketRegime::Neutral.contagion_loading()
        );
        assert!(MarketRegime::RiskOff.vol_multiplier() > MarketRegime::Neutral.vol_multiplier());
        assert!(MarketRegime::Neutral.vol_multiplier() > MarketRegime::RiskOn.vol_multiplier());
    }

    #[test]
    fn high_stay_probability_gives_long_average_runs() {
        let transition = RegimeTransition::new(0.98);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut regime = MarketRegime::Neutral;
        let mut run_lengths = vec![];
        let mut current_run = 1;
        for _ in 0..5_000 {
            let next = transition.next(regime, &mut rng);
            if next.index() == regime.index() {
                current_run += 1;
            } else {
                run_lengths.push(current_run);
                current_run = 1;
            }
            regime = next;
        }
        let mean_run = run_lengths.iter().sum::<i32>() as f64 / run_lengths.len() as f64;
        // Expected run length for a geometric process with stay_probability p
        // is 1 / (1 - p) = 50. Assert it's in a broad but meaningful band.
        assert!(
            mean_run > 20.0,
            "mean run length {mean_run} too short for stay_probability=0.98"
        );
    }

    #[test]
    #[should_panic(expected = "stay_probability must be in [0, 1)")]
    fn rejects_invalid_stay_probability() {
        RegimeTransition::new(1.0);
    }
}
