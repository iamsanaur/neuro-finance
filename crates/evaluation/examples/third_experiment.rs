//! `exp-0003`: `exp-0002`'s own next step — a capacity-matched
//! `MlpBaseline` (no graph, roughly the same parameter count as
//! `neuro_model`) to separate "extra capacity overfits" from "the topology
//! mechanism specifically overfits" — evaluated across **every**
//! walk-forward split in a longer synthetic run, not just the first one,
//! so a single lucky/unlucky split can't drive the conclusion.
//!
//! Run with: `cargo run --release --example third_experiment -p evaluation`

use chrono::Duration;
use data_engine::synthetic::{MarketRegime, SyntheticMarketConfig, SyntheticMarketGenerator};
use evaluation::{
    classification_report, LogisticRegressionBaseline, MajorityClassBaseline, MlpBaseline,
    NaivePersistenceBaseline,
};
use feature_engine::{
    close_series, log_returns, momentum, moving_average, rolling_volatility, Standardizer,
};
use financial_types::{EntityId, MarketBar, PointInTimeDataset, Symbol, Timestamp};
use neuro_model::NeuroTopologicalFinancialModel;
use serde_json::json;
use tensor_engine::burn;
use tensor_engine::burn::optim::{AdamConfig, GradientsParams, Optimizer};
use tensor_engine::burn::tensor::Tensor;
use tensor_engine::{device, Backend};
use training_engine::{
    nll_loss, TrainerConfig, TrainingExample, WalkForwardSplit, WalkForwardValidator, WindowMode,
};

const FEATURE_DIM: usize = 4;
const NUM_CLASSES: usize = 3;
const WARMUP_DAYS: usize = 25;
const EPOCHS: usize = 60;
const LEARNING_RATE: f64 = 0.02;
const EMBED_DIM: usize = 16;
const TOPOLOGY_PROJ_DIM: usize = 8;
/// Chosen so `MlpBaseline::param_count(FEATURE_DIM, MLP_HIDDEN_DIM,
/// NUM_CLASSES) == 931`, matching `neuro_model`'s parameter count at
/// `(feature_dim=4, embed_dim=16, topology_proj_dim=8)`:
/// encoder 80 + query 128 + key 128 + message-passing value 272 +
/// message-passing update_mlp 272 + regime head 51 = 931.
/// `MlpBaseline::param_count(4, 116, 3) == 931` exactly.
const MLP_HIDDEN_DIM: usize = 116;

fn regime_class_index(regime: MarketRegime) -> usize {
    MarketRegime::ALL
        .iter()
        .position(|&r| r == regime)
        .expect("MarketRegime::ALL is exhaustive")
}

struct RawExample {
    features: Vec<f64>,
    target_class: usize,
    timestamp: Timestamp,
}

fn build_raw_features(bars_as_of: &[MarketBar], symbols: &[Symbol]) -> Vec<f64> {
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
        features.extend([last_return, vol, mom, sma_deviation]);
    }
    features
}

fn build_raw_examples(
    dataset: &data_engine::synthetic::SyntheticMarketDataset,
    symbols: &[Symbol],
    window: (Timestamp, Timestamp),
) -> Vec<RawExample> {
    let pit = PointInTimeDataset::new(dataset.bars.clone());
    dataset
        .regime_schedule
        .iter()
        .enumerate()
        .filter(|(i, (ts, _))| *i >= WARMUP_DAYS && *ts >= window.0 && *ts < window.1)
        .map(|(_, (ts, regime))| {
            let visible: Vec<MarketBar> = pit.as_of(*ts).cloned().collect();
            RawExample {
                features: build_raw_features(&visible, symbols),
                target_class: regime_class_index(*regime),
                timestamp: *ts,
            }
        })
        .collect()
}

fn fit_standardizers(train: &[RawExample], num_assets: usize) -> Vec<Standardizer> {
    (0..FEATURE_DIM)
        .map(|f| {
            let values: Vec<f64> = train
                .iter()
                .flat_map(|ex| (0..num_assets).map(move |a| ex.features[a * FEATURE_DIM + f]))
                .collect();
            Standardizer::fit(&values)
        })
        .collect()
}

fn to_training_examples(
    raw: &[RawExample],
    standardizers: &[Standardizer],
    entities: &[EntityId],
    num_assets: usize,
) -> Vec<TrainingExample> {
    raw.iter()
        .map(|ex| {
            let transformed: Vec<f32> = (0..num_assets)
                .flat_map(|a| {
                    (0..FEATURE_DIM).map(move |f| {
                        standardizers[f].transform(ex.features[a * FEATURE_DIM + f]) as f32
                    })
                })
                .collect();
            let features: Tensor<Backend, 2> =
                Tensor::<Backend, 1>::from_data(transformed.as_slice(), &device())
                    .reshape([num_assets, FEATURE_DIM]);
            TrainingExample {
                features,
                entities: entities.to_vec(),
                target_class: ex.target_class,
                timestamp: ex.timestamp,
            }
        })
        .collect()
}

fn train_neuro_model(examples: &[TrainingExample]) -> NeuroTopologicalFinancialModel<Backend> {
    let mut model: NeuroTopologicalFinancialModel<Backend> =
        NeuroTopologicalFinancialModel::new(FEATURE_DIM, EMBED_DIM, TOPOLOGY_PROJ_DIM, &device());
    let mut optimizer = AdamConfig::new().init();
    let config = TrainerConfig {
        learning_rate: LEARNING_RATE,
        top_k: 5,
        batch_size: 8,
    };
    for _ in 0..EPOCHS {
        let (updated, _) = training_engine::train_epoch(model, &mut optimizer, examples, &config);
        model = updated;
    }
    model
}

/// Shared training loop for any `forward(&self, Tensor) -> Tensor` module —
/// `LogisticRegressionBaseline` and `MlpBaseline` are trained identically,
/// differing only in architecture.
fn train_flat_model<M, F>(mut model: M, examples: &[TrainingExample], forward: F) -> M
where
    M: burn::module::AutodiffModule<Backend>,
    F: Fn(&M, Tensor<Backend, 2>) -> Tensor<Backend, 2>,
{
    let mut optimizer = AdamConfig::new().init::<Backend, M>();
    let batch_size = 8;
    for _ in 0..EPOCHS {
        for batch in examples.chunks(batch_size) {
            let mut batch_loss: Option<Tensor<Backend, 1>> = None;
            for example in batch {
                let probs = forward(&model, example.features.clone());
                let loss = nll_loss(probs, example.target_class);
                batch_loss = Some(match batch_loss {
                    Some(acc) => acc + loss,
                    None => loss,
                });
            }
            let batch_loss = batch_loss.unwrap() / (batch.len() as f32);
            let grads = batch_loss.backward();
            let grads_params = GradientsParams::from_grads::<Backend, _>(grads, &model);
            model = optimizer.step(LEARNING_RATE, model, grads_params);
        }
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

fn evaluate_split(
    dataset: &data_engine::synthetic::SyntheticMarketDataset,
    entities: &[EntityId],
    symbols: &[Symbol],
    split: &WalkForwardSplit,
) -> serde_json::Value {
    let num_assets = symbols.len();
    let raw_train = build_raw_examples(dataset, symbols, split.train);
    let raw_test = build_raw_examples(dataset, symbols, split.test);
    let standardizers = fit_standardizers(&raw_train, num_assets);
    let train_examples = to_training_examples(&raw_train, &standardizers, entities, num_assets);
    let test_examples = to_training_examples(&raw_test, &standardizers, entities, num_assets);

    let neuro = train_neuro_model(&train_examples);
    let logistic: LogisticRegressionBaseline<Backend> =
        LogisticRegressionBaseline::new(FEATURE_DIM, NUM_CLASSES, &device());
    let logistic = train_flat_model(logistic, &train_examples, |m, x| m.forward(x));
    let mlp: MlpBaseline<Backend> =
        MlpBaseline::new(FEATURE_DIM, MLP_HIDDEN_DIM, NUM_CLASSES, &device());
    let mlp = train_flat_model(mlp, &train_examples, |m, x| m.forward(x));

    let train_classes: Vec<usize> = train_examples.iter().map(|e| e.target_class).collect();
    let majority = MajorityClassBaseline::fit(&train_classes);
    let mut naive = NaivePersistenceBaseline::new();
    naive.observe(*train_classes.last().unwrap());

    let actual: Vec<usize> = test_examples.iter().map(|e| e.target_class).collect();
    let neuro_predicted: Vec<usize> = test_examples
        .iter()
        .map(|e| {
            predict_class(
                |x| neuro.forward(x, entities, 5, e.timestamp),
                e.features.clone(),
            )
        })
        .collect();
    let logistic_predicted: Vec<usize> = test_examples
        .iter()
        .map(|e| predict_class(|x| logistic.forward(x), e.features.clone()))
        .collect();
    let mlp_predicted: Vec<usize> = test_examples
        .iter()
        .map(|e| predict_class(|x| mlp.forward(x), e.features.clone()))
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

    json!({
        "train_examples": train_examples.len(),
        "test_examples": test_examples.len(),
        "accuracy": {
            "neuro_model": classification_report(&neuro_predicted, &actual, NUM_CLASSES).accuracy,
            "logistic_regression": classification_report(&logistic_predicted, &actual, NUM_CLASSES).accuracy,
            "mlp_baseline": classification_report(&mlp_predicted, &actual, NUM_CLASSES).accuracy,
            "majority_class": classification_report(&majority_predicted, &actual, NUM_CLASSES).accuracy,
            "naive_persistence": classification_report(&naive_predicted, &actual, NUM_CLASSES).accuracy,
        },
    })
}

fn main() {
    tensor_engine::seed(42);

    assert_eq!(
        MlpBaseline::<Backend>::param_count(FEATURE_DIM, MLP_HIDDEN_DIM, NUM_CLASSES),
        931,
        "MLP_HIDDEN_DIM should be chosen to match neuro_model's ~931 parameters"
    );

    println!("=== Generating synthetic market (longer run, for multiple splits) ===");
    let market_config = SyntheticMarketConfig {
        num_assets: 30,
        num_days: 1100,
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
    println!("=== {} walk-forward splits found ===", splits.len());

    let mut per_split_results = Vec::new();
    for (i, split) in splits.iter().enumerate() {
        println!(
            "--- split {i}: train {}..{}  test {}..{} ---",
            split.train.0.date_naive(),
            split.train.1.date_naive(),
            split.test.0.date_naive(),
            split.test.1.date_naive(),
        );
        let result = evaluate_split(&dataset, &entities, &symbols, split);
        println!("  {}", serde_json::to_string(&result["accuracy"]).unwrap());
        per_split_results.push(result);
    }

    let mean_accuracy = |key: &str| -> f64 {
        let sum: f64 = per_split_results
            .iter()
            .map(|r| r["accuracy"][key].as_f64().unwrap())
            .sum();
        sum / per_split_results.len() as f64
    };

    println!("\n=== Mean accuracy across {} splits ===", splits.len());
    for key in [
        "naive_persistence",
        "logistic_regression",
        "mlp_baseline",
        "neuro_model",
        "majority_class",
    ] {
        println!("  {key:22}{:.4}", mean_accuracy(key));
    }

    let experiment_dir = std::path::Path::new("experiments/exp-0003-capacity-matched-multi-split");
    std::fs::create_dir_all(experiment_dir).expect("failed to create experiment directory");

    let config = json!({
        "experiment_id": "exp-0003-capacity-matched-multi-split",
        "based_on": "exp-0002-normalized-longer-training",
        "changes_from_exp_0002": [
            "added MlpBaseline, capacity-matched to neuro_model (931 params each)",
            "evaluated across all walk-forward splits, not just the first",
            "num_days 900 -> 1100 (to fit multiple splits)",
        ],
        "market": {"num_assets": 30, "num_days": 1100, "seed": 42},
        "walk_forward": {"train_days": 500, "validation_days": 100, "test_days": 100, "embargo_days": 5, "mode": "Expanding", "num_splits": splits.len()},
        "model": {"feature_dim": FEATURE_DIM, "embed_dim": EMBED_DIM, "topology_proj_dim": TOPOLOGY_PROJ_DIM, "top_k": 5, "param_count": 931},
        "mlp_baseline": {"hidden_dim": MLP_HIDDEN_DIM, "param_count": 931},
        "training": {"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": 8, "optimizer": "Adam"},
    });
    std::fs::write(
        experiment_dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .expect("failed to write config.json");

    let metrics = json!({
        "per_split": per_split_results,
        "mean_accuracy": {
            "naive_persistence": mean_accuracy("naive_persistence"),
            "logistic_regression": mean_accuracy("logistic_regression"),
            "mlp_baseline": mean_accuracy("mlp_baseline"),
            "neuro_model": mean_accuracy("neuro_model"),
            "majority_class": mean_accuracy("majority_class"),
        },
    });
    std::fs::write(
        experiment_dir.join("metrics.json"),
        serde_json::to_string_pretty(&metrics).unwrap(),
    )
    .expect("failed to write metrics.json");

    println!("\nWrote results to {}", experiment_dir.display());
}
