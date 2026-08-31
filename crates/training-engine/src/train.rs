//! The mini-batch training loop (project spec §31) for
//! `NeuroTopologicalFinancialModel`.
//!
//! ## What "mini-batch" means here
//!
//! Each training example is one trading day: a `[N, feature_dim]` feature
//! matrix, an entity list, and a target regime class. Different days'
//! graphs (built fresh per day by `TopologyScorer`/`top_k_topology` inside
//! `NeuroTopologicalFinancialModel::forward`) aren't the same shape or
//! structure, so they can't be stacked into one batched tensor the way
//! same-shaped examples usually are. "Mini-batching" here instead means
//! **gradient accumulation**: `batch_size` days' losses are averaged before
//! a single optimizer step, rather than stepping after every single day.
//! This is a real, if simpler, form of mini-batching (it changes the
//! gradient-noise/step-count tradeoff exactly the way batching over
//! same-shaped examples would) — a redesign toward per-day tensor batching
//! is future work, not needed for V0.1's ~100-asset, ~day-granularity scale.
//!
//! ## Target labels are caller-supplied class indices, not `MarketRegime`
//!
//! `training-engine` has no dependency on `data-engine` (a data source) in
//! its non-test code — a training loop should not need to know where its
//! labels came from. Callers (see `tests/synthetic_pipeline.rs`) map
//! `data_engine::synthetic::MarketRegime` to a class index themselves,
//! using `RegimeHead`'s documented class order.

use crate::loss::nll_loss;
use burn::optim::{GradientsParams, Optimizer};
use financial_types::{EntityId, Timestamp};
use neuro_model::NeuroTopologicalFinancialModel;
use tensor_engine::burn;
use tensor_engine::Backend;

#[derive(Clone)]
pub struct TrainingExample {
    pub features: burn::tensor::Tensor<Backend, 2>,
    pub entities: Vec<EntityId>,
    pub target_class: usize,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub learning_rate: f64,
    pub top_k: usize,
    pub batch_size: usize,
}

/// Runs one epoch (one pass over `examples`, in order — no shuffling: these
/// are time-ordered trading days, and shuffling would mix days within a
/// gradient-accumulation batch in a way that has no principled meaning
/// here, unlike i.i.d. training data). Returns the updated model and the
/// mean per-example loss over the epoch.
pub fn train_epoch<O: Optimizer<NeuroTopologicalFinancialModel<Backend>, Backend>>(
    mut model: NeuroTopologicalFinancialModel<Backend>,
    optimizer: &mut O,
    examples: &[TrainingExample],
    config: &TrainerConfig,
) -> (NeuroTopologicalFinancialModel<Backend>, f64) {
    assert!(config.batch_size > 0, "batch_size must be positive");
    let mut total_loss = 0.0_f64;
    let mut num_batches = 0usize;

    for batch in examples.chunks(config.batch_size) {
        let mut batch_loss: Option<burn::tensor::Tensor<Backend, 1>> = None;
        for example in batch {
            let probs = model.forward(
                example.features.clone(),
                &example.entities,
                config.top_k,
                example.timestamp,
            );
            let loss = nll_loss(probs, example.target_class);
            batch_loss = Some(match batch_loss {
                Some(acc) => acc + loss,
                None => loss,
            });
        }
        let batch_loss =
            batch_loss.expect("chunks() never yields an empty batch") / (batch.len() as f32);
        let loss_value: f32 = batch_loss.clone().into_scalar();
        total_loss += f64::from(loss_value);
        num_batches += 1;

        let grads = batch_loss.backward();
        let grads_params = GradientsParams::from_grads::<Backend, _>(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads_params);
    }

    (model, total_loss / num_batches.max(1) as f64)
}

/// Forward-only pass over `examples` (no gradient step) — for computing
/// validation/test loss with [`crate::EarlyStopping`] or a walk-forward
/// test window, where the model must not be updated.
pub fn evaluate(
    model: &NeuroTopologicalFinancialModel<Backend>,
    examples: &[TrainingExample],
    top_k: usize,
) -> f64 {
    if examples.is_empty() {
        return 0.0;
    }
    let mut total = 0.0_f64;
    for example in examples {
        let probs = model.forward(
            example.features.clone(),
            &example.entities,
            top_k,
            example.timestamp,
        );
        let loss: f32 = nll_loss(probs, example.target_class).into_scalar();
        total += f64::from(loss);
    }
    total / examples.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::AdamConfig;
    use chrono::{TimeZone, Utc};

    fn ts() -> Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    fn dummy_example(target_class: usize) -> TrainingExample {
        let n = 10;
        let feature_dim = 4;
        let entities: Vec<EntityId> = (0..n).map(|i| EntityId::from(format!("E{i}"))).collect();
        let features: burn::tensor::Tensor<Backend, 2> = burn::tensor::Tensor::random(
            [n, feature_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &tensor_engine::device(),
        );
        TrainingExample {
            features,
            entities,
            target_class,
            timestamp: ts(),
        }
    }

    #[test]
    fn training_reduces_loss_on_a_fixed_repeated_example() {
        tensor_engine::seed(0);
        let model = NeuroTopologicalFinancialModel::new(4, 8, 4, &tensor_engine::device());
        let mut optimizer = AdamConfig::new().init();
        let config = TrainerConfig {
            learning_rate: 0.05,
            top_k: 3,
            batch_size: 1,
        };

        // Same example repeated: a model that's actually learning should
        // drive the loss down on it over successive epochs.
        let examples = vec![dummy_example(0); 8];

        let loss_before = evaluate(&model, &examples, config.top_k);
        let mut model = model;
        let mut loss_after = loss_before;
        for _ in 0..10 {
            let (updated_model, epoch_loss) =
                train_epoch(model, &mut optimizer, &examples, &config);
            model = updated_model;
            loss_after = epoch_loss;
        }

        assert!(
            loss_after < loss_before,
            "expected training loss to decrease: before={loss_before}, after={loss_after}"
        );
    }

    #[test]
    fn evaluate_does_not_change_the_model() {
        tensor_engine::seed(1);
        let model = NeuroTopologicalFinancialModel::new(4, 8, 4, &tensor_engine::device());
        let examples = vec![dummy_example(1), dummy_example(2)];

        let loss_a = evaluate(&model, &examples, 3);
        let loss_b = evaluate(&model, &examples, 3);
        assert!(
            (loss_a - loss_b).abs() < 1e-9,
            "evaluate should be deterministic and side-effect-free"
        );
    }

    #[test]
    fn evaluate_of_empty_examples_is_zero() {
        let model = NeuroTopologicalFinancialModel::new(4, 8, 4, &tensor_engine::device());
        assert_eq!(evaluate(&model, &[], 3), 0.0);
    }
}
