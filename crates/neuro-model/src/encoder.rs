//! `FeatureEncoder`: raw per-asset causal features → node embeddings
//! (project spec §24's `FeatureEncoder` stage — the "Input → Feature
//! Encoder → Node Embeddings" start of the forward path).
//!
//! V0.1 is one linear projection + ReLU — deliberately the simplest thing
//! that produces embeddings of the right shape for
//! `topology_engine::TopologyScorer` to consume. A deeper encoder (multiple
//! layers, normalization, per-feature-type handling) is a natural place to
//! grow this crate later, once there's a trained baseline to compare a
//! richer encoder against (§28's baseline-first discipline applies inside a
//! crate, not just across model architectures).
//!
//! Generic over `B: Backend`, not fixed to `tensor_engine::Backend` — see
//! `topology_engine::scorer`'s doc comment for the real gradient-tracking
//! bug (found in Milestone 8) that makes this a correctness requirement,
//! not a style choice, for every `#[derive(Module)]` struct in this crate.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use tensor_engine::burn;

#[derive(Module, Debug)]
pub struct FeatureEncoder<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> FeatureEncoder<B> {
    /// `input_dim` is the number of causal features per asset (from
    /// `feature-engine`); `embed_dim` is the node embedding size fed to the
    /// rest of the model.
    pub fn new(input_dim: usize, embed_dim: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(input_dim, embed_dim).init(device),
        }
    }

    /// `x`: `[N, input_dim]` raw features, one row per asset. Returns
    /// `[N, embed_dim]` node embeddings.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        relu(self.linear.forward(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_engine::{device, Backend as ConcreteBackend};

    #[test]
    fn forward_produces_expected_shape() {
        tensor_engine::seed(0);
        let encoder: FeatureEncoder<ConcreteBackend> = FeatureEncoder::new(6, 16, &device());
        let x: Tensor<ConcreteBackend, 2> = Tensor::zeros([100, 6], &device());
        let h = encoder.forward(x);
        assert_eq!(h.dims(), [100, 16]);
    }

    #[test]
    fn relu_clamps_negative_activations_at_zero() {
        // A linear layer with all-zero input still has a bias term that can
        // be negative; ReLU should clamp any negative output to exactly 0,
        // never leave a negative value through.
        tensor_engine::seed(0);
        let encoder: FeatureEncoder<ConcreteBackend> = FeatureEncoder::new(4, 8, &device());
        let x: Tensor<ConcreteBackend, 2> = Tensor::zeros([5, 4], &device());
        let h = encoder.forward(x);
        let min: f32 = h.min().into_scalar();
        assert!(
            min >= 0.0,
            "ReLU output should never be negative, got min={min}"
        );
    }
}
