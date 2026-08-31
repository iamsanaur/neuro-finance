//! `MlpBaseline`: a capacity-matched non-graph baseline (project spec §28's
//! "MLP" entry), sized to have roughly the same parameter count as
//! `NeuroTopologicalFinancialModel` — the point of this baseline is
//! specifically to separate two hypotheses `exp-0002` couldn't distinguish
//! between: "the topology mechanism overfits" vs. "any model with this
//! much capacity overfits on this little data, whether or not it has a
//! graph in it." If `MlpBaseline` *also* overfits the way `neuro_model`
//! did in `exp-0002`, that points at capacity, not topology; if it doesn't,
//! that's evidence the graph/topology machinery specifically is the
//! problem.
//!
//! One hidden layer, mean-pooled raw features straight in (same
//! `mean_dim(0)` pooling convention as every other head in this
//! workspace). Hidden width is a constructor argument, not hardcoded — see
//! `examples/third_experiment.rs` for the specific width chosen to
//! approximately match `neuro_model`'s parameter count, and the arithmetic
//! behind that choice.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use tensor_engine::burn;

#[derive(Module, Debug)]
pub struct MlpBaseline<B: Backend> {
    hidden: Linear<B>,
    classifier: Linear<B>,
}

impl<B: Backend> MlpBaseline<B> {
    pub fn new(
        feature_dim: usize,
        hidden_dim: usize,
        num_classes: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            hidden: LinearConfig::new(feature_dim, hidden_dim).init(device),
            classifier: LinearConfig::new(hidden_dim, num_classes).init(device),
        }
    }

    /// Total learnable parameter count (weights + biases of both layers) —
    /// exposed so callers constructing a capacity-matched comparison can
    /// verify the match numerically rather than trusting arithmetic done by
    /// hand in a comment.
    pub fn param_count(feature_dim: usize, hidden_dim: usize, num_classes: usize) -> usize {
        let hidden_params = feature_dim * hidden_dim + hidden_dim; // weight + bias
        let classifier_params = hidden_dim * num_classes + num_classes;
        hidden_params + classifier_params
    }

    /// `x`: `[N, feature_dim]` raw per-asset features. Returns `[1,
    /// num_classes]` softmax probabilities.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let pooled = x.mean_dim(0);
        let hidden = relu(self.hidden.forward(pooled));
        softmax(self.classifier.forward(hidden), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_engine::{device, Backend as ConcreteBackend};

    #[test]
    fn forward_produces_a_valid_probability_distribution() {
        tensor_engine::seed(0);
        let model: MlpBaseline<ConcreteBackend> = MlpBaseline::new(5, 32, 3, &device());
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

    #[test]
    fn param_count_matches_hand_computed_arithmetic() {
        // feature_dim=4, hidden_dim=10, num_classes=3:
        // hidden: 4*10 + 10 = 50; classifier: 10*3 + 3 = 33; total 83.
        assert_eq!(MlpBaseline::<ConcreteBackend>::param_count(4, 10, 3), 83);
    }
}
