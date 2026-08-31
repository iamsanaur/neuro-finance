//! Extracting a single symbol's chronologically ordered price/volume series
//! from a flat `&[MarketBar]` slice (e.g. `data-engine`'s synthetic output).
//!
//! This is the boundary between "point-in-time-safe data access"
//! (`financial-types::PointInTimeDataset`, already enforced) and "causal
//! windowing" (this crate): a caller is expected to have already restricted
//! `bars` to what's knowable as of some prediction time — via
//! `PointInTimeDataset::as_of` — before calling any of these extractors or
//! feature functions. This module does not itself perform any point-in-time
//! filtering; it only sorts and reshapes.

use financial_types::{MarketBar, Symbol, Timestamp};

/// One symbol's bars, sorted ascending by `timestamp`.
pub fn bars_for_symbol<'a>(bars: &'a [MarketBar], symbol: &Symbol) -> Vec<&'a MarketBar> {
    let mut filtered: Vec<&MarketBar> = bars.iter().filter(|b| &b.symbol == symbol).collect();
    filtered.sort_by_key(|b| b.timestamp);
    filtered
}

/// Convenience: `(timestamps, closes)` for one symbol, chronologically
/// ordered. Timestamps are returned alongside the values because every
/// `Option<f64>` output in this crate is index-aligned to its input — the
/// caller zips these timestamps back onto feature output to know which day
/// each value belongs to.
pub fn close_series(bars: &[MarketBar], symbol: &Symbol) -> (Vec<Timestamp>, Vec<f64>) {
    let sorted = bars_for_symbol(bars, symbol);
    (
        sorted.iter().map(|b| b.timestamp).collect(),
        sorted.iter().map(|b| b.close).collect(),
    )
}

/// Convenience: `(timestamps, volumes)` for one symbol, chronologically
/// ordered.
pub fn volume_series(bars: &[MarketBar], symbol: &Symbol) -> (Vec<Timestamp>, Vec<f64>) {
    let sorted = bars_for_symbol(bars, symbol);
    (
        sorted.iter().map(|b| b.timestamp).collect(),
        sorted.iter().map(|b| b.volume).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn bar(day: u32, symbol: &str, close: f64) -> MarketBar {
        MarketBar {
            timestamp: Utc.with_ymd_and_hms(2020, 1, day, 0, 0, 0).unwrap(),
            symbol: Symbol::from(symbol),
            open: close,
            high: close,
            low: close,
            close,
            volume: 100.0,
        }
    }

    #[test]
    fn filters_and_sorts_by_symbol() {
        let bars = vec![
            bar(3, "AAA", 30.0),
            bar(1, "BBB", 10.0),
            bar(1, "AAA", 10.0),
            bar(2, "AAA", 20.0),
        ];
        let (timestamps, closes) = close_series(&bars, &Symbol::from("AAA"));
        assert_eq!(closes, vec![10.0, 20.0, 30.0]);
        assert!(timestamps.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn unknown_symbol_returns_empty() {
        let bars = vec![bar(1, "AAA", 10.0)];
        let (timestamps, closes) = close_series(&bars, &Symbol::from("ZZZ"));
        assert!(timestamps.is_empty());
        assert!(closes.is_empty());
    }
}
