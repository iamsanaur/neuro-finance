//! `exp-0002`: re-runs `exp-0001` with the two changes its
//! `RESEARCH_REPORT.md` identified as the concrete next step — point-in-
//! time-safe feature standardization (fit on the train window only,
//! applied unchanged to test) and a longer training schedule — to check
//! whether under-training, not the architecture, explains `exp-0001`'s
//! result.
//!
//! Run with: `cargo run --release --example second_experiment -p evaluation`

use chrono::Duration;
use data_engine::synthetic::{MarketRegime, SyntheticMarketConfig, SyntheticMarketGenerator};
use evaluation::{
    classification_report, LogisticRegressionBaseline, MajorityClassBaseline,
    NaivePersistenceBaseline,
};
use feature_engine::{
    close_series, log_returns, momentum, moving_average, rolling_volatility, Standardizer,
};
use financial_types::{EntityId, MarketBar, PointInTimeDataset, Symbol, Timestamp};
use neuro_model::NeuroTopologicalFinancialModel;
use serde_json::json;
use tensor_engine::burn::optim::{AdamConfig, GradientsParams, Optimizer};
use tensor_engine::burn::tensor::Tensor;
use tensor_engine::{device, Backend};
use training_engine::{nll_loss, TrainerConfig, TrainingExample, WalkForwardValidator, WindowMode};

const FEATURE_DIM: usize = 4;
const NUM_CLASSES: usize = 3;
const WARMUP_DAYS: usize = 25;
const EPOCHS: usize = 60;
const LEARNING_RATE: f64 = 0.02;

fn regime_class_index(regime: MarketRegime) -> usize {
    MarketRegime::ALL
        .iter()
        .position(|&r| r == regime)
        .expect("MarketRegime::ALL is exhaustive")
}

/// Raw (unstandardized) per-day features, kept separate from
/// `TrainingExample` so standardizers can be fit across the raw train set
/// before any tensor is built.
struct RawExample {
    /// `[num_assets * FEATURE_DIM]`, row-major `[asset][feature]`.
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

/// Fits one [`Standardizer`] per feature dimension, pooling that feature's
/// values across every asset and every day in `train` — train data only,
/// per this module's point-in-time-safety contract.
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

fn train_neuro_model(
    examples: &[TrainingExample],
    epochs: usize,
) -> (NeuroTopologicalFinancialModel<Backend>, f64) {
    let mut model: NeuroTopologicalFinancialModel<Backend> =
        NeuroTopologicalFinancialModel::new(FEATURE_DIM, 16, 8, &device());
    let mut optimizer = AdamConfig::new().init();
    let config = TrainerConfig {
        learning_rate: LEARNING_RATE,
        top_k: 5,
        batch_size: 8,
    };
    let mut final_loss = f64::INFINITY;
    for epoch in 0..epochs {
        let (updated, loss) =
            training_engine::train_epoch(model, &mut optimizer, examples, &config);
        model = updated;
        final_loss = loss;
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("  neuro-model epoch {epoch}: train loss = {loss:.4}");
        }
    }
    (model, final_loss)
}

fn train_logistic_baseline(
    examples: &[TrainingExample],
    epochs: usize,
) -> (LogisticRegressionBaseline<Backend>, f64) {
    let mut model: LogisticRegressionBaseline<Backend> =
        LogisticRegressionBaseline::new(FEATURE_DIM, NUM_CLASSES, &device());
    let mut optimizer = AdamConfig::new().init();
    let batch_size = 8;
    let mut final_loss = f64::INFINITY;
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
            model = optimizer.step(LEARNING_RATE, model, grads_params);
        }
        final_loss = total_loss / num_batches.max(1) as f64;
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("  logistic-regression epoch {epoch}: train loss = {final_loss:.4}");
        }
    }
    (model, final_loss)
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

    println!("=== Generating synthetic market (same seed/config as exp-0001) ===");
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
    let num_assets = symbols.len();

    println!("=== Building walk-forward split (same as exp-0001) ===");
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
    let split = &splits[0];

    println!("=== Building raw features, fitting standardizers on train only ===");
    let raw_train = build_raw_examples(&dataset, &symbols, split.train);
    let raw_test = build_raw_examples(&dataset, &symbols, split.test);
    let standardizers = fit_standardizers(&raw_train, num_assets);
    let train_examples = to_training_examples(&raw_train, &standardizers, &entities, num_assets);
    let test_examples = to_training_examples(&raw_test, &standardizers, &entities, num_assets);
    println!(
        "  train examples: {}, test examples: {}",
        train_examples.len(),
        test_examples.len()
    );

    println!("=== Training neuro-model ({EPOCHS} epochs, lr={LEARNING_RATE}) ===");
    let (model, neuro_final_loss) = train_neuro_model(&train_examples, EPOCHS);

    println!("=== Training logistic-regression baseline ({EPOCHS} epochs, lr={LEARNING_RATE}) ===");
    let (logistic, logistic_final_loss) = train_logistic_baseline(&train_examples, EPOCHS);

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
    println!(
        "  neuro-model:          {:.4} (final train loss {:.4})",
        neuro_report.accuracy, neuro_final_loss
    );
    println!(
        "  logistic-regression:  {:.4} (final train loss {:.4})",
        logistic_report.accuracy, logistic_final_loss
    );
    println!("  majority-class:       {:.4}", majority_report.accuracy);
    println!("  naive-persistence:    {:.4}", naive_report.accuracy);
    println!(
        "  (ln(3) = {:.4}, the uniform-guess loss exp-0001 plateaued near)",
        3.0_f64.ln()
    );

    let experiment_dir = std::path::Path::new("experiments/exp-0002-normalized-longer-training");
    std::fs::create_dir_all(experiment_dir).expect("failed to create experiment directory");

    let config = json!({
        "experiment_id": "exp-0002-normalized-longer-training",
        "based_on": "exp-0001-first-regime-classification",
        "changes_from_exp_0001": [
            "z-score feature standardization, fit on train window only",
            format!("epochs 15 -> {EPOCHS}"),
            format!("learning_rate 0.01 -> {LEARNING_RATE}"),
        ],
        "market": {"num_assets": 30, "num_days": 900, "seed": 42},
        "walk_forward": {"train_days": 500, "validation_days": 100, "test_days": 100, "embargo_days": 5, "mode": "Expanding"},
        "features": ["last_return", "rolling_volatility_20", "momentum_20", "sma_deviation_20"],
        "model": {"feature_dim": FEATURE_DIM, "embed_dim": 16, "topology_proj_dim": 8, "top_k": 5},
        "training": {"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": 8, "optimizer": "Adam"},
    });
    std::fs::write(
        experiment_dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .expect("failed to write config.json");

    let metrics = json!({
        "test_examples": test_examples.len(),
        "train_examples": train_examples.len(),
        "final_train_loss": {
            "neuro_model": neuro_final_loss,
            "logistic_regression": logistic_final_loss,
            "uniform_guess_reference": 3.0_f64.ln(),
        },
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
