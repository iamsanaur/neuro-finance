//! Integration test: synthetic market data -> causal features ->
//! `NeuroTopologicalFinancialModel` -> regime prediction, end to end.
//!
//! This is a first, partial version of the synthetic end-to-end test
//! project spec §48 calls for as the primary CI integration test — "partial"
//! because the portfolio/backtest stages (§48's last two steps) don't exist
//! yet. What this test actually proves: every crate built so far
//! (`data-engine`, `feature-engine`, `financial-types`, `financial-graph`
//! via `topology-engine`, `neuro-model`) composes into one working forward
//! pass without a shape mismatch, a panic, or a NaN — on real generated
//! data, not synthetic-to-the-test-itself fixtures.

use chrono::Duration;
use data_engine::synthetic::{SyntheticMarketConfig, SyntheticMarketGenerator};
use feature_engine::{close_series, log_returns, moving_average, rolling_volatility};
use financial_types::{EntityId, PointInTimeDataset, Timestamp};
use neuro_model::NeuroTopologicalFinancialModel;
use tensor_engine::burn::tensor::Tensor;
use tensor_engine::{device, Backend};

/// Builds a `[N, 2]` feature matrix (rolling volatility, moving-average
/// deviation) for every asset, `as_of` some day, from bars restricted via
/// `PointInTimeDataset::as_of` — the point-in-time-safe path Milestone 2-5
/// established, exercised here as a caller rather than in isolation.
fn build_features(
    dataset: &data_engine::synthetic::SyntheticMarketDataset,
    as_of: Timestamp,
) -> (Vec<f32>, Vec<EntityId>) {
    let pit = PointInTimeDataset::new(dataset.bars.clone());
    let visible: Vec<_> = pit.as_of(as_of).cloned().collect();

    let mut features = Vec::new();
    let mut entities = Vec::new();
    for (symbol, _) in &dataset.sector_assignment {
        let (_, closes) = close_series(&visible, symbol);
        let returns = log_returns(&closes);
        let vol = rolling_volatility(&returns, 20, 5)
            .last()
            .copied()
            .flatten()
            .unwrap_or(0.0);
        let sma = moving_average(&closes, 20, 5).last().copied().flatten();
        let last_close = *closes.last().unwrap_or(&0.0);
        let sma_deviation = match sma {
            Some(sma) if sma != 0.0 => ((last_close - sma) / sma) as f32,
            _ => 0.0,
        };
        features.push(vol as f32);
        features.push(sma_deviation);
        entities.push(EntityId::from(symbol.0.clone()));
    }
    (features, entities)
}

#[test]
fn full_pipeline_runs_end_to_end_on_synthetic_data() {
    tensor_engine::seed(0);

    let config = SyntheticMarketConfig {
        num_assets: 30,
        num_days: 60,
        ..SyntheticMarketConfig::default()
    };
    let dataset = SyntheticMarketGenerator::new(config).generate();

    // Predict as of the last generated day (30 days after the sequence
    // start, so the rolling features have enough history to be Some).
    let as_of = dataset.regime_schedule.last().unwrap().0;
    let (feature_data, entities) = build_features(&dataset, as_of);

    let n = entities.len();
    let feature_dim = 2;
    assert_eq!(feature_data.len(), n * feature_dim);
    assert!(
        feature_data.iter().all(|v| v.is_finite()),
        "no NaN/Inf features from real synthetic data"
    );

    let features: Tensor<Backend, 2> =
        Tensor::<Backend, 1>::from_data(feature_data.as_slice(), &device())
            .reshape([n, feature_dim]);

    let model = NeuroTopologicalFinancialModel::new(feature_dim, 16, 8);
    let probs = model.forward(features, &entities, 4, as_of);

    assert_eq!(probs.dims(), [1, 3]);
    let sum: f32 = probs.clone().sum().into_scalar();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "regime probabilities should sum to 1.0, got {sum}"
    );
    let min: f32 = probs.min().into_scalar();
    assert!(min >= 0.0);
}

#[test]
fn pipeline_is_point_in_time_safe_across_two_prediction_days() {
    // A weaker, but still meaningful, sanity check: predicting on an
    // earlier day must only ever see bars up to that day — verified here
    // by confirming build_features (this test file's own helper) produces
    // different feature vectors for two different `as_of` days from the
    // same dataset, which would not be true if it accidentally used the
    // full dataset regardless of `as_of` (a mistake financial-graph's
    // Milestone 5 test already demonstrated is a real, not hypothetical,
    // failure mode for code with this exact shape).
    let config = SyntheticMarketConfig {
        num_assets: 10,
        num_days: 60,
        ..SyntheticMarketConfig::default()
    };
    let dataset = SyntheticMarketGenerator::new(config).generate();

    let start = dataset.regime_schedule.first().unwrap().0;
    let early = start + Duration::days(30);
    let late = start + Duration::days(59);
    assert_ne!(early, late);

    let (features_early, _) = build_features(&dataset, early);
    let (features_late, _) = build_features(&dataset, late);

    assert_ne!(features_early, features_late);
}
