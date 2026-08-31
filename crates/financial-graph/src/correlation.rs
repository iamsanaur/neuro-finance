//! Correlation graph builder (project spec §14): an edge between two assets
//! weighted by their rolling return correlation, thresholded to keep the
//! graph sparse.
//!
//! **Point-in-time safety is the caller's responsibility, by design**: this
//! function computes correlation only from the `bars` it's given — it does
//! not know what "today" is and does not filter anything itself. The
//! intended usage is:
//!
//! ```text
//! let pit_dataset = PointInTimeDataset::new(all_bars);   // financial-types
//! let visible = pit_dataset.as_of(as_of).cloned().collect::<Vec<_>>();
//! let graph = build_correlation_graph(&visible, &symbols, as_of, &config);
//! ```
//!
//! i.e. the point-in-time filtering happens once, upstream, using the
//! machinery Milestone 2 already built — this crate does not reimplement
//! it. The `correlation_graph_does_not_use_future_bars` test below exists
//! to catch a regression of that contract, not to add a second enforcement
//! mechanism.

use crate::graph::FinancialGraph;
use crate::types::{Edge, NodeId, RelationType};
use feature_engine::correlation::pearson;
use feature_engine::{close_series, log_returns};
use financial_types::{EntityId, MarketBar, Symbol, Timestamp};

#[derive(Debug, Clone)]
pub struct CorrelationGraphConfig {
    /// Trailing number of daily returns used to compute each pairwise
    /// correlation.
    pub window: usize,
    /// Minimum trailing returns required (for *both* assets) before an edge
    /// is considered at all; assets with less history than this are left
    /// isolated (degree 0) in the resulting graph rather than causing an
    /// error.
    pub min_periods: usize,
    /// Edges with `|correlation| < min_abs_correlation` are dropped — this
    /// is what keeps the graph sparse rather than near-complete (real
    /// return series are rarely exactly uncorrelated, so an unthresholded
    /// correlation graph on `n` assets is close to a complete graph on `n`
    /// nodes).
    pub min_abs_correlation: f64,
}

impl Default for CorrelationGraphConfig {
    fn default() -> Self {
        Self {
            window: 60,
            min_periods: 20,
            min_abs_correlation: 0.3,
        }
    }
}

/// Builds a `RelationType::Correlation` graph over `symbols` from `bars`.
/// See the module doc for the point-in-time-safety contract this function
/// relies on the caller to uphold.
pub fn build_correlation_graph(
    bars: &[MarketBar],
    symbols: &[Symbol],
    as_of: Timestamp,
    config: &CorrelationGraphConfig,
) -> FinancialGraph {
    assert!(
        config.min_periods >= 2,
        "min_periods must be >= 2, got {}",
        config.min_periods
    );
    assert!(
        config.min_periods <= config.window,
        "min_periods ({}) must be <= window ({})",
        config.min_periods,
        config.window
    );
    assert!(
        (0.0..=1.0).contains(&config.min_abs_correlation),
        "min_abs_correlation must be in [0, 1], got {}",
        config.min_abs_correlation
    );

    // Each symbol's trailing `window` log returns, computed only from the
    // bars it was given (no knowledge of "now" beyond that).
    let tails: Vec<Vec<f64>> = symbols
        .iter()
        .map(|symbol| {
            let (_, closes) = close_series(bars, symbol);
            let returns = log_returns(&closes);
            let start = returns.len().saturating_sub(config.window);
            returns[start..].to_vec()
        })
        .collect();

    let nodes: Vec<EntityId> = symbols
        .iter()
        .map(|s| EntityId::from(s.0.clone()))
        .collect();
    let mut graph = FinancialGraph::new(nodes);

    for i in 0..symbols.len() {
        for j in (i + 1)..symbols.len() {
            if tails[i].len() < config.min_periods || tails[j].len() < config.min_periods {
                continue;
            }
            // Align to the shorter of the two tails (from the end — both
            // are already trailing windows, so this keeps them causal).
            let n = tails[i].len().min(tails[j].len());
            let a = &tails[i][tails[i].len() - n..];
            let b = &tails[j][tails[j].len() - n..];
            let corr = pearson(a, b);
            if corr.abs() >= config.min_abs_correlation {
                graph
                    .add_edge(Edge {
                        source: NodeId(i as u32),
                        target: NodeId(j as u32),
                        relation: RelationType::Correlation,
                        weight: corr,
                        timestamp: as_of,
                    })
                    .expect("indices come from the graph's own node table");
            }
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_engine::synthetic::{SyntheticMarketConfig, SyntheticMarketGenerator};
    use financial_types::PointInTimeDataset;

    #[test]
    fn correlated_pair_produces_an_edge() {
        let mut bars = Vec::new();
        let base = chrono::Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        use chrono::TimeZone;
        let mut price_a = 100.0_f64;
        let mut price_b = 50.0_f64;
        for day in 0..30 {
            // Perfectly co-moving prices: identical daily log return.
            let r: f64 = 0.01 * if day % 2 == 0 { 1.0 } else { -1.0 };
            price_a *= r.exp();
            price_b *= r.exp();
            let ts = base + chrono::Duration::days(day);
            bars.push(MarketBar {
                timestamp: ts,
                symbol: Symbol::from("AAA"),
                open: price_a,
                high: price_a,
                low: price_a,
                close: price_a,
                volume: 1000.0,
            });
            bars.push(MarketBar {
                timestamp: ts,
                symbol: Symbol::from("BBB"),
                open: price_b,
                high: price_b,
                low: price_b,
                close: price_b,
                volume: 1000.0,
            });
        }
        let symbols = vec![Symbol::from("AAA"), Symbol::from("BBB")];
        let config = CorrelationGraphConfig {
            window: 30,
            min_periods: 10,
            min_abs_correlation: 0.5,
        };
        let graph =
            build_correlation_graph(&bars, &symbols, base + chrono::Duration::days(29), &config);

        assert_eq!(graph.num_edges(), 1);
        let a = graph.node_id(&EntityId::from("AAA")).unwrap();
        let edge = graph.neighbors(a, None).next().unwrap().1;
        assert!((edge.weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn threshold_drops_weak_correlations() {
        let mut bars = Vec::new();
        let base = chrono::Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        use chrono::TimeZone;
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let mut price_a = 100.0_f64;
        let mut price_b = 100.0_f64;
        for day in 0..30 {
            price_a *= (rng.gen_range(-0.01..0.01_f64)).exp();
            price_b *= (rng.gen_range(-0.01..0.01_f64)).exp();
            let ts = base + chrono::Duration::days(day);
            bars.push(MarketBar {
                timestamp: ts,
                symbol: Symbol::from("AAA"),
                open: price_a,
                high: price_a,
                low: price_a,
                close: price_a,
                volume: 1000.0,
            });
            bars.push(MarketBar {
                timestamp: ts,
                symbol: Symbol::from("BBB"),
                open: price_b,
                high: price_b,
                low: price_b,
                close: price_b,
                volume: 1000.0,
            });
        }
        let symbols = vec![Symbol::from("AAA"), Symbol::from("BBB")];
        // A near-1.0 threshold should drop this independently-random pair.
        let config = CorrelationGraphConfig {
            window: 30,
            min_periods: 10,
            min_abs_correlation: 0.95,
        };
        let graph =
            build_correlation_graph(&bars, &symbols, base + chrono::Duration::days(29), &config);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn insufficient_history_leaves_asset_isolated() {
        let base = chrono::Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        use chrono::TimeZone;
        let bars = vec![
            MarketBar {
                timestamp: base,
                symbol: Symbol::from("AAA"),
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1000.0,
            },
            MarketBar {
                timestamp: base,
                symbol: Symbol::from("BBB"),
                open: 50.0,
                high: 50.0,
                low: 50.0,
                close: 50.0,
                volume: 1000.0,
            },
        ];
        let symbols = vec![Symbol::from("AAA"), Symbol::from("BBB")];
        let config = CorrelationGraphConfig::default();
        let graph = build_correlation_graph(&bars, &symbols, base, &config);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_nodes(), 2);
    }

    /// The concrete "future graph edges" leakage test project spec §30 asks
    /// for: this function trusts its `bars` argument completely and does
    /// not filter by `as_of` itself (as the module doc warns) — so it is
    /// entirely possible to misuse it by passing bars beyond `as_of`. This
    /// test demonstrates that misuse concretely (graphs built from a
    /// correctly `PointInTimeDataset::as_of`-truncated subset differ from
    /// ones built by (incorrectly) passing the full dataset), and confirms
    /// that using the *same* truncated subset twice is deterministic — i.e.
    /// the graph is a pure function of exactly what it's given, so the only
    /// thing standing between a caller and a leak is truncating `bars`
    /// before the call, which is exactly what `PointInTimeDataset::as_of`
    /// is for.
    #[test]
    fn correlation_graph_does_not_use_future_bars() {
        let config = SyntheticMarketConfig {
            num_assets: 10,
            num_days: 200,
            ..SyntheticMarketConfig::default()
        };
        let dataset = SyntheticMarketGenerator::new(config).generate();
        let symbols: Vec<Symbol> = dataset
            .sector_assignment
            .iter()
            .map(|(s, _)| s.clone())
            .collect();
        let graph_config = CorrelationGraphConfig::default();

        let pit = PointInTimeDataset::new(dataset.bars.clone());
        let as_of = dataset.regime_schedule[99].0; // day 100 of 200

        // Correct usage: truncate to what's actually knowable as of day 100.
        let visible: Vec<MarketBar> = pit.as_of(as_of).cloned().collect();
        let graph_correct = build_correlation_graph(&visible, &symbols, as_of, &graph_config);

        // Misuse: pass the full dataset (including days 101-200) with the
        // same `as_of` label. Because this function does not filter by
        // `as_of` itself, its correlation windows end up anchored near the
        // *end* of the full series (day 200), not day 100 — a real leak,
        // not a hypothetical one.
        let graph_leaked = build_correlation_graph(&dataset.bars, &symbols, as_of, &graph_config);

        let edge_weights = |g: &FinancialGraph| -> Vec<f64> {
            let mut w: Vec<f64> = g
                .edges_of_relation(RelationType::Correlation)
                .map(|e| e.weight)
                .collect();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w
        };
        assert_ne!(
            edge_weights(&graph_correct),
            edge_weights(&graph_leaked),
            "correlation graph should differ when built from a truncated vs. leaked (full) bar set — \
             if this fails, the two inputs coincidentally produced the same edges and the test needs a \
             different day/window to be a meaningful check"
        );

        // Re-running with the correctly truncated subset a second time
        // (simulating a second `as_of` call at the same timestamp) must be
        // byte-identical: the graph is a pure function of its input.
        let graph_correct_again = build_correlation_graph(&visible, &symbols, as_of, &graph_config);
        assert_eq!(
            edge_weights(&graph_correct),
            edge_weights(&graph_correct_again)
        );
    }
}
