//! `LogisticRegressionBaseline`: mean-pooled raw features straight into a
//! linear + softmax classifier — no hidden layer, no graph, no topology.
//! The point of this baseline (project spec §28) is to answer "does the
//! graph/topology machinery earn its complexity," so it deliberately uses
//! the *same* input features and the *same* mean-pool-then-classify shape
//! as `neuro_model::RegimeHead`, differing only in having no
//! `FeatureEncoder`/`GraphMessagePassing`/`TopologyScorer` in between.
//!
//! Generic over `B: Backend` — see `topology_engine::scorer`'s doc comment
//! for why every `#[derive(Module)]` struct in this workspace must be
//! (Milestone 8's gradient-tracking bug).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use tensor_engine::burn;

#[derive(Module, Debug)]
pub struct LogisticRegressionBaseline<B: Backend> {
    classifier: Linear<B>,
}

impl<B: Backend> LogisticRegressionBaseline<B> {
    pub fn new(feature_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        Self {
            classifier: LinearConfig::new(feature_dim, num_classes).init(device),
        }
    }

    /// `x`: `[N, feature_dim]` raw per-asset features. Returns `[1,
    /// num_classes]` softmax probabilities — same shape and pooling
    /// convention as `neuro_model::RegimeHead::forward`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let pooled = x.mean_dim(0);
        softmax(self.classifier.forward(pooled), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_engine::{device, Backend as ConcreteBackend};

    #[test]
    fn forward_produces_a_valid_probability_distribution() {
        tensor_engine::seed(0);
        let model: LogisticRegressionBaseline<ConcreteBackend> =
            LogisticRegressionBaseline::new(5, 3, &device());
        let x: Tensor<ConcreteBackend, 2> = Tensor::random(
            [20, 5],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let probs = model.forward(x);
        assert_eq!(probs.dims(), [1, 3]);
        let sum: f32 = probs.clone().sum().into_scalar();
        assert!((sum - 1.0).abs() < 1e-5);
        let min: f32 = probs.min().into_scalar();
        assert!(min >= 0.0);
    }
}
