//! The project's first real experiment (project spec §32, §55): train
//! `NeuroTopologicalFinancialModel` and every baseline in this crate on one
//! walk-forward split of synthetic data, evaluate all of them on the same
//! held-out test window, and write the result to `experiments/<id>/`.
//!
//! Run with: `cargo run --release --example first_experiment -p evaluation`
//!
//! This is deliberately small in scope (one walk-forward split, not a full
//! rolling evaluation across many; four hand-picked features, not the full
//! `feature-engine` catalog; a few dozen training epochs) — the point of
//! this experiment is to prove the whole pipeline (data -> features ->
//! point-in-time-safe splitting -> training -> evaluation -> baselines ->
//! written report) actually works end to end, not to produce a
//! publication-grade result. See the written report this produces
//! (`experiments/`) for the honest interpretation of the numbers.

use chrono::Duration;
use data_engine::synthetic::{MarketRegime, SyntheticMarketConfig, SyntheticMarketGenerator};
use evaluation::{
    classification_report, LogisticRegressionBaseline, MajorityClassBaseline,
    NaivePersistenceBaseline,
};
use feature_engine::{close_series, log_returns, momentum, moving_average, rolling_volatility};
use financial_types::{EntityId, MarketBar, PointInTimeDataset, Symbol, Timestamp};
use neuro_model::NeuroTopologicalFinancialModel;
use serde_json::json;
use tensor_engine::burn::optim::{AdamConfig, GradientsParams, Optimizer};
use tensor_engine::burn::tensor::Tensor;
use tensor_engine::{device, Backend};
use training_engine::{nll_loss, TrainerConfig, TrainingExample, WalkForwardValidator, WindowMode};

const FEATURE_DIM: usize = 4;
const NUM_CLASSES: usize = 3;
const WARMUP_DAYS: usize = 25; // rolling windows need history before producing Some

fn regime_class_index(regime: MarketRegime) -> usize {
    MarketRegime::ALL
        .iter()
        .position(|&r| r == regime)
        .expect("MarketRegime::ALL is exhaustive")
}

/// `[num_assets * FEATURE_DIM]` features as of `bars_as_of`'s latest day,
/// one asset's features contiguous per row: `[last_return, rolling_vol,
/// momentum_20, sma_deviation]`.
fn build_features(bars_as_of: &[MarketBar], symbols: &[Symbol]) -> Vec<f32> {
    let mut features = Vec::with_capacity(symbols.len() * FEATURE_DIM);
    for symbol in symbols {
        let (_, closes) = close_series(bars_as_of, symbol);
        let returns = log_returns(&closes);
        let last_return = returns.last().copied().unwrap_or(0.0);
        let vol = rolling_volatility(&returns, 20, 5)
            .last()
            .copied()
            .flatten()
            .unwrap_or(0.0);
        let mom = momentum(&closes, 20)
            .last()
            .copied()
            .flatten()
            .unwrap_or(0.0);
        let sma = moving_average(&closes, 20, 5).last().copied().flatten();
        let last_close = *closes.last().unwrap_or(&0.0);
        let sma_deviation = match sma {
            Some(sma) if sma.abs() > 1e-9 => (last_close - sma) / sma,
            _ => 0.0,
        };
        features.extend([
            last_return as f32,
            vol as f32,
            mom as f32,
            sma_deviation as f32,
        ]);
    }
    features
}

fn build_examples(
    dataset: &data_engine::synthetic::SyntheticMarketDataset,
    entities: &[EntityId],
    symbols: &[Symbol],
    window: (Timestamp, Timestamp),
) -> Vec<TrainingExample> {
    let pit = PointInTimeDataset::new(dataset.bars.clone());
    dataset
        .regime_schedule
        .iter()
        .enumerate()
        .filter(|(i, (ts, _))| *i >= WARMUP_DAYS && *ts >= window.0 && *ts < window.1)
        .map(|(_, (ts, regime))| {
            let visible: Vec<MarketBar> = pit.as_of(*ts).cloned().collect();
            let feature_data = build_features(&visible, symbols);
            let features: Tensor<Backend, 2> =
                Tensor::<Backend, 1>::from_data(feature_data.as_slice(), &device())
                    .reshape([symbols.len(), FEATURE_DIM]);
            TrainingExample {
                features,
                entities: entities.to_vec(),
                target_class: regime_class_index(*regime),
                timestamp: *ts,
            }
        })
        .collect()
}

fn train_neuro_model(
    examples: &[TrainingExample],
    epochs: usize,
) -> NeuroTopologicalFinancialModel<Backend> {
    let mut model: NeuroTopologicalFinancialModel<Backend> =
        NeuroTopologicalFinancialModel::new(FEATURE_DIM, 16, 8, &device());
    let mut optimizer = AdamConfig::new().init();
    let config = TrainerConfig {
        learning_rate: 0.01,
        top_k: 5,
        batch_size: 8,
    };
    for epoch in 0..epochs {
        let (updated, loss) =
            training_engine::train_epoch(model, &mut optimizer, examples, &config);
        model = updated;
        println!("  neuro-model epoch {epoch}: train loss = {loss:.4}");
    }
    model
}

fn train_logistic_baseline(
    examples: &[TrainingExample],
    epochs: usize,
) -> LogisticRegressionBaseline<Backend> {
    let mut model: LogisticRegressionBaseline<Backend> =
        LogisticRegressionBaseline::new(FEATURE_DIM, NUM_CLASSES, &device());
    let mut optimizer = AdamConfig::new().init();
    let learning_rate = 0.01;
    let batch_size = 8;
    for epoch in 0..epochs {
        let mut total_loss = 0.0_f64;
        let mut num_batches = 0usize;
        for batch in examples.chunks(batch_size) {
            let mut batch_loss: Option<Tensor<Backend, 1>> = None;
            for example in batch {
                let probs = model.forward(example.features.clone());
                let loss = nll_loss(probs, example.target_class);
                batch_loss = Some(match batch_loss {
                    Some(acc) => acc + loss,
                    None => loss,
                });
            }
            let batch_loss = batch_loss.unwrap() / (batch.len() as f32);
            let loss_value: f32 = batch_loss.clone().into_scalar();
            total_loss += f64::from(loss_value);
            num_batches += 1;
            let grads = batch_loss.backward();
            let grads_params = GradientsParams::from_grads::<Backend, _>(grads, &model);
            model = optimizer.step(learning_rate, model, grads_params);
        }
        println!(
            "  logistic-regression epoch {epoch}: train loss = {:.4}",
            total_loss / num_batches.max(1) as f64
        );
    }
    model
}

fn predict_class<F: Fn(Tensor<Backend, 2>) -> Tensor<Backend, 2>>(
    forward: F,
    features: Tensor<Backend, 2>,
) -> usize {
    let probs = forward(features);
    let data: Vec<f32> = probs.into_data().to_vec().unwrap();
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn main() {
    tensor_engine::seed(42);

    println!("=== Generating synthetic market ===");
    let market_config = SyntheticMarketConfig {
        num_assets: 30,
        num_days: 900,
        ..SyntheticMarketConfig::default()
    };
    let dataset = SyntheticMarketGenerator::new(market_config).generate();
    let symbols: Vec<Symbol> = dataset
        .sector_assignment
        .iter()
        .map(|(s, _)| s.clone())
        .collect();
    let entities: Vec<EntityId> = symbols
        .iter()
        .map(|s| EntityId::from(s.0.clone()))
        .collect();

    println!("=== Building walk-forward split ===");
    let validator = WalkForwardValidator::new(
        Duration::days(500),
        Duration::days(100),
        Duration::days(100),
        Duration::days(5),
        WindowMode::Expanding,
    );
    let start = dataset.regime_schedule.first().unwrap().0;
    let end = dataset.regime_schedule.last().unwrap().0;
    let splits = validator.splits(start, end);
    assert!(
        !splits.is_empty(),
        "expected at least one walk-forward split from 900 days of data"
    );
    let split = &splits[0];
    println!(
        "  train:      {} .. {}\n  validation: {} .. {}\n  test:       {} .. {}",
        split.train.0.date_naive(),
        split.train.1.date_naive(),
        split.validation.0.date_naive(),
        split.validation.1.date_naive(),
        split.test.0.date_naive(),
        split.test.1.date_naive(),
    );

    println!("=== Building features (point-in-time-safe) ===");
    let train_examples = build_examples(&dataset, &entities, &symbols, split.train);
    let test_examples = build_examples(&dataset, &entities, &symbols, split.test);
    println!(
        "  train examples: {}, test examples: {}",
        train_examples.len(),
        test_examples.len()
    );

    println!("=== Training neuro-model ===");
    let epochs = 15;
    let model = train_neuro_model(&train_examples, epochs);

    println!("=== Training logistic-regression baseline ===");
    let logistic = train_logistic_baseline(&train_examples, epochs);

    println!("=== Fitting naive baselines ===");
    let train_classes: Vec<usize> = train_examples.iter().map(|e| e.target_class).collect();
    let majority = MajorityClassBaseline::fit(&train_classes);
    let mut naive = NaivePersistenceBaseline::new();
    naive.observe(*train_classes.last().unwrap());

    println!("=== Evaluating on held-out test window ===");
    let actual: Vec<usize> = test_examples.iter().map(|e| e.target_class).collect();

    let neuro_predicted: Vec<usize> = test_examples
        .iter()
        .map(|e| {
            predict_class(
                |x| model.forward(x, &entities, 5, e.timestamp),
                e.features.clone(),
            )
        })
        .collect();

    let logistic_predicted: Vec<usize> = test_examples
        .iter()
        .map(|e| predict_class(|x| logistic.forward(x), e.features.clone()))
        .collect();

    let majority_predicted: Vec<usize> = test_examples.iter().map(|_| majority.predict()).collect();

    let naive_predicted: Vec<usize> = test_examples
        .iter()
        .map(|e| {
            let pred = naive.predict(majority.predict());
            naive.observe(e.target_class);
            pred
        })
        .collect();

    let neuro_report = classification_report(&neuro_predicted, &actual, NUM_CLASSES);
    let logistic_report = classification_report(&logistic_predicted, &actual, NUM_CLASSES);
    let majority_report = classification_report(&majority_predicted, &actual, NUM_CLASSES);
    let naive_report = classification_report(&naive_predicted, &actual, NUM_CLASSES);

    println!("\n=== Results (test-window accuracy) ===");
    println!("  neuro-model:          {:.4}", neuro_report.accuracy);
    println!("  logistic-regression:  {:.4}", logistic_report.accuracy);
    println!("  majority-class:       {:.4}", majority_report.accuracy);
    println!("  naive-persistence:    {:.4}", naive_report.accuracy);

    let experiment_dir = std::path::Path::new("experiments/exp-0001-first-regime-classification");
    std::fs::create_dir_all(experiment_dir).expect("failed to create experiment directory");

    let config = json!({
        "experiment_id": "exp-0001-first-regime-classification",
        "market": {"num_assets": 30, "num_days": 900, "seed": 42},
        "walk_forward": {"train_days": 500, "validation_days": 100, "test_days": 100, "embargo_days": 5, "mode": "Expanding"},
        "features": ["last_return", "rolling_volatility_20", "momentum_20", "sma_deviation_20"],
        "model": {"feature_dim": FEATURE_DIM, "embed_dim": 16, "topology_proj_dim": 8, "top_k": 5},
        "training": {"epochs": epochs, "learning_rate": 0.01, "batch_size": 8, "optimizer": "Adam"},
    });
    std::fs::write(
        experiment_dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .expect("failed to write config.json");

    let metrics = json!({
        "test_examples": test_examples.len(),
        "train_examples": train_examples.len(),
        "accuracy": {
            "neuro_model": neuro_report.accuracy,
            "logistic_regression": logistic_report.accuracy,
            "majority_class": majority_report.accuracy,
            "naive_persistence": naive_report.accuracy,
        },
    });
    std::fs::write(
        experiment_dir.join("metrics.json"),
        serde_json::to_string_pretty(&metrics).unwrap(),
    )
    .expect("failed to write metrics.json");

    println!("\nWrote results to {}", experiment_dir.display());
}
