//! `RegimeHead`: the first prediction task (project spec §25) — market
//! regime classification into `RiskOn` / `Neutral` / `RiskOff`, the same
//! three classes `data-engine::synthetic::MarketRegime` already generates
//! ground truth for.
//!
//! Node embeddings are mean-pooled into one market-level embedding before
//! classification — a market regime is a property of the whole market, not
//! of any one asset, so pooling across nodes before the final linear layer
//! is the natural shape for this task (as opposed to per-asset heads, which
//! is what the *second* task, §26's asset direction prediction, will need
//! instead).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::Tensor;
use tensor_engine::burn;
use tensor_engine::{device, Backend};

pub const NUM_REGIME_CLASSES: usize = 3;

/// Class order: `[RiskOn, Neutral, RiskOff]` — matches
/// `data_engine::synthetic::regime::MarketRegime::ALL`'s order exactly, so
/// index `i` here and `MarketRegime::ALL[i]` there always refer to the same
/// class without a separate lookup table.
#[derive(Module, Debug, Clone)]
pub struct RegimeHead {
    classifier: Linear<Backend>,
}

impl RegimeHead {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            classifier: LinearConfig::new(embed_dim, NUM_REGIME_CLASSES).init(&device()),
        }
    }

    /// `h`: `[N, embed_dim]` node embeddings. Returns `[1, 3]` class
    /// probabilities (softmax-normalized, sums to `1.0`).
    pub fn forward(&self, h: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        let pooled = h.mean_dim(0); // [1, embed_dim]
        let logits = self.classifier.forward(pooled); // [1, 3]
        softmax(logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_a_valid_probability_distribution() {
        tensor_engine::seed(0);
        let head = RegimeHead::new(10);
        let h: Tensor<Backend, 2> = Tensor::random(
            [50, 10],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let probs = head.forward(h);
        assert_eq!(probs.dims(), [1, 3]);

        let sum: f32 = probs.clone().sum().into_scalar();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probabilities should sum to 1.0, got {sum}"
        );

        let min: f32 = probs.min().into_scalar();
        assert!(
            min >= 0.0,
            "probabilities should never be negative, got {min}"
        );
    }
}
