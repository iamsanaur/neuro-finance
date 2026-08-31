//! `MarketBar`: a single OHLCV observation for one symbol over one period.

use crate::identifiers::Symbol;
use crate::point_in_time::PointInTime;
use crate::timestamp::Timestamp;
use serde::{Deserialize, Serialize};

/// A single OHLCV bar.
///
/// Precision (project spec §46): prices and volume are `f64`. They are not
/// neural-network internals (where `f32` would be fine) and volume can be
/// fractional for some instruments (crypto, some ETFs) — `f64` is the safer
/// default. Do not narrow to `f32` downstream without a documented reason.
///
/// `timestamp` is the bar's *close* time. In V0.1, with only end-of-day
/// synthetic data, a bar's availability time is assumed equal to its
/// timestamp — see the [`PointInTime`] impl below and its doc comment for
/// why that assumption will need revisiting for intraday or real data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketBar {
    pub timestamp: Timestamp,
    pub symbol: Symbol,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MarketBarError {
    #[error("empty symbol")]
    EmptySymbol,
    #[error("high ({high}) is below low ({low})")]
    HighBelowLow { high: f64, low: f64 },
    #[error("open ({open}) is outside [low, high] = [{low}, {high}]")]
    OpenOutsideRange { open: f64, low: f64, high: f64 },
    #[error("close ({close}) is outside [low, high] = [{low}, {high}]")]
    CloseOutsideRange { close: f64, low: f64, high: f64 },
    #[error("negative volume ({volume})")]
    NegativeVolume { volume: f64 },
    #[error("non-finite price or volume")]
    NonFinite,
}

impl MarketBar {
    /// Validates OHLC consistency (`low <= open, close <= high`), a
    /// non-negative volume, and that no field is NaN/infinite. Construction
    /// with the bare struct literal is still allowed (e.g. for test
    /// fixtures) — this is the checked path data-engine adapters should use
    /// for anything ingested from outside the process.
    pub fn validate(&self) -> Result<(), MarketBarError> {
        if self.symbol.0.is_empty() {
            return Err(MarketBarError::EmptySymbol);
        }
        for v in [self.open, self.high, self.low, self.close, self.volume] {
            if !v.is_finite() {
                return Err(MarketBarError::NonFinite);
            }
        }
        if self.high < self.low {
            return Err(MarketBarError::HighBelowLow {
                high: self.high,
                low: self.low,
            });
        }
        if self.open < self.low || self.open > self.high {
            return Err(MarketBarError::OpenOutsideRange {
                open: self.open,
                low: self.low,
                high: self.high,
            });
        }
        if self.close < self.low || self.close > self.high {
            return Err(MarketBarError::CloseOutsideRange {
                close: self.close,
                low: self.low,
                high: self.high,
            });
        }
        if self.volume < 0.0 {
            return Err(MarketBarError::NegativeVolume {
                volume: self.volume,
            });
        }
        Ok(())
    }
}

impl PointInTime for MarketBar {
    fn observation_time(&self) -> Timestamp {
        self.timestamp
    }

    /// V0.1 assumption: a bar is available for modeling at its own close
    /// timestamp (no publication lag modeled for market data itself). This
    /// is optimistic for real intraday feeds (exchange dissemination,
    /// consolidation, and vendor latency all add delay) and will need a
    /// separate `availability_time` field once real/intraday data is
    /// wired in (V0.2+, see PROJECT_STATUS.md). Documented here rather than
    /// silently assumed, per project spec §9.
    fn availability_time(&self) -> Timestamp {
        self.timestamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(y: i32, m: u32, d: u32) -> Timestamp {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }
    use chrono::Utc;

    fn valid_bar() -> MarketBar {
        MarketBar {
            timestamp: ts(2020, 1, 2),
            symbol: Symbol::from("AAA"),
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1_000.0,
        }
    }

    #[test]
    fn valid_bar_passes_validation() {
        assert!(valid_bar().validate().is_ok());
    }

    #[test]
    fn rejects_empty_symbol() {
        let mut bar = valid_bar();
        bar.symbol = Symbol::from("");
        assert_eq!(bar.validate(), Err(MarketBarError::EmptySymbol));
    }

    #[test]
    fn rejects_high_below_low() {
        let mut bar = valid_bar();
        bar.high = 90.0;
        assert_eq!(
            bar.validate(),
            Err(MarketBarError::HighBelowLow {
                high: 90.0,
                low: 99.0
            })
        );
    }

    #[test]
    fn rejects_open_outside_range() {
        let mut bar = valid_bar();
        bar.open = 200.0;
        assert!(matches!(
            bar.validate(),
            Err(MarketBarError::OpenOutsideRange { .. })
        ));
    }

    #[test]
    fn rejects_negative_volume() {
        let mut bar = valid_bar();
        bar.volume = -1.0;
        assert!(matches!(
            bar.validate(),
            Err(MarketBarError::NegativeVolume { .. })
        ));
    }

    #[test]
    fn availability_equals_timestamp_in_v0_1() {
        let bar = valid_bar();
        assert_eq!(bar.availability_time(), bar.observation_time());
    }
}
