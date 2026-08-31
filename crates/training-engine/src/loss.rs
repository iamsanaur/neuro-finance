//! Negative log-likelihood loss over `neuro-model::RegimeHead`'s
//! already-softmax-normalized output.
//!
//! Burn's built-in `CrossEntropyLoss` expects raw logits (it applies
//! log-softmax internally); `RegimeHead::forward` already returns
//! probabilities (its own doc comment: "softmax-normalized, sums to
//! `1.0`"), so this computes `-ln(p_target)` directly rather than
//! reorganizing `neuro-model` to expose pre-softmax logits just to reuse
//! Burn's loss module. `eps` guards against `ln(0.0)` if a probability
//! ever underflows to exactly zero.

use burn::tensor::Tensor;
use tensor_engine::burn;
use tensor_engine::Backend;

const EPS: f32 = 1e-7;

/// `probs`: `[1, num_classes]` (as `RegimeHead::forward` returns).
/// `target_class`: the correct class index. Returns a `[1]` scalar loss
/// tensor, differentiable back through `probs`.
pub fn nll_loss(probs: Tensor<Backend, 2>, target_class: usize) -> Tensor<Backend, 1> {
    let num_classes = probs.dims()[1];
    assert!(
        target_class < num_classes,
        "target_class {target_class} out of range (0..{num_classes})"
    );

    let target_prob = probs.slice([0..1, target_class..target_class + 1]);
    let target_prob = target_prob.reshape([1]);
    -((target_prob + EPS).log())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_engine::device;

    #[test]
    fn loss_is_near_zero_for_confident_correct_prediction() {
        let probs: Tensor<Backend, 2> =
            Tensor::<Backend, 1>::from_data([0.98_f32, 0.01, 0.01].as_slice(), &device())
                .reshape([1, 3]);
        let loss: f32 = nll_loss(probs, 0).into_scalar();
        assert!(
            loss < 0.05,
            "expected near-zero loss for a confident correct prediction, got {loss}"
        );
    }

    #[test]
    fn loss_is_large_for_confident_wrong_prediction() {
        let probs: Tensor<Backend, 2> =
            Tensor::<Backend, 1>::from_data([0.98_f32, 0.01, 0.01].as_slice(), &device())
                .reshape([1, 3]);
        let loss: f32 = nll_loss(probs, 2).into_scalar();
        assert!(
            loss > 3.0,
            "expected a large loss for a confidently wrong prediction, got {loss}"
        );
    }

    #[test]
    fn uniform_prediction_gives_ln_num_classes_loss() {
        let probs: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_data(
            [1.0_f32 / 3.0, 1.0 / 3.0, 1.0 / 3.0].as_slice(),
            &device(),
        )
        .reshape([1, 3]);
        let loss: f32 = nll_loss(probs, 1).into_scalar();
        let expected = (3.0_f32).ln();
        assert!(
            (loss - expected).abs() < 1e-3,
            "expected ~{expected}, got {loss}"
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rejects_out_of_range_target_class() {
        let probs: Tensor<Backend, 2> =
            Tensor::<Backend, 1>::from_data([0.5_f32, 0.5].as_slice(), &device()).reshape([1, 2]);
        nll_loss(probs, 5);
    }
}
