//! `TopologyScorer`: learnable relationship scores between every pair of
//! node embeddings (project spec §16).
//!
//! ```text
//! Q_i = W_q h_i
//! K_j = W_k h_j
//! s_ij = Q_i^T K_j / sqrt(d)
//! ```
//!
//! This produces a full `[N, N]` score matrix as an *intermediate* value —
//! standard scaled dot-product attention necessarily computes all pairwise
//! scores before any selection happens. What §13/§16 actually asks not to
//! do is *store or propagate* that dense matrix downstream: [`crate::topk`]
//! immediately reduces it to a sparse top-k edge set, and the dense
//! `Tensor` returned by `forward` is not retained past that (or past a loss
//! computation, e.g. [`crate::regularization::l_stability`], which also
//! only needs it transiently).
//!
//! ## Why this struct is generic over `B: Backend`, not fixed to
//! `tensor_engine::Backend`
//!
//! A real bug, found while building `training-engine` (Milestone 8), not a
//! stylistic preference: a `#[derive(Module)]` struct whose fields use a
//! *concrete* backend type (e.g. `Linear<tensor_engine::Backend>` written
//! directly) compiles and runs forward passes correctly, but
//! `GradientsParams::from_grads` silently returns **zero** entries for
//! it — Burn's derive macro needs an actual generic type parameter to
//! generate a working parameter-registration path for training; a
//! monomorphized field type produces a `Module` impl that *looks* complete
//! (forward works) but isn't wired for gradient extraction. Confirmed with
//! a minimal reproduction (a one-field wrapper around `Linear`) before
//! changing every `Module` struct in the workspace — see `PROJECT_STATUS.md`
//! and `docs/environment.md` for the full writeup. `tensor_engine::Backend`
//! is still the one concrete type actually *used* — just as the type
//! argument at the call site (`TopologyScorer<tensor_engine::Backend>`),
//! not baked into the struct definition.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use tensor_engine::burn;

#[derive(Module, Debug)]
pub struct TopologyScorer<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    #[module(skip)]
    proj_dim: usize,
}

impl<B: Backend> TopologyScorer<B> {
    /// `input_dim` is the node embedding size; `proj_dim` is the
    /// query/key projection size (`d` in the formula above). No bias term
    /// on either projection — a bias would shift every score by a constant
    /// per query/key, which does nothing useful for a similarity score
    /// that's about to be ranked (top-k is shift-invariant per row only if
    /// the shift is per-query, not per-key, so it isn't actually free to
    /// drop — but it is standard practice for attention-style scoring, and
    /// keeps the parameter count and the diff against the literal spec
    /// formula both smaller).
    pub fn new(input_dim: usize, proj_dim: usize, device: &B::Device) -> Self {
        Self {
            query: LinearConfig::new(input_dim, proj_dim)
                .with_bias(false)
                .init(device),
            key: LinearConfig::new(input_dim, proj_dim)
                .with_bias(false)
                .init(device),
            proj_dim,
        }
    }

    /// `h`: `[N, input_dim]` node embeddings. Returns the `[N, N]` score
    /// matrix (`scores[i][j] = s_ij`).
    pub fn forward(&self, h: Tensor<B, 2>) -> Tensor<B, 2> {
        let q = self.query.forward(h.clone());
        let k = self.key.forward(h);
        let scale = (self.proj_dim as f64).sqrt();
        q.matmul(k.transpose()) / scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_engine::{device, Backend as ConcreteBackend};

    #[test]
    fn forward_produces_n_by_n_scores() {
        tensor_engine::seed(0);
        let scorer: TopologyScorer<ConcreteBackend> = TopologyScorer::new(8, 4, &device());
        let h: Tensor<ConcreteBackend, 2> = Tensor::random(
            [10, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let scores = scorer.forward(h);
        assert_eq!(scores.dims(), [10, 10]);
    }

    /// `forward` itself is a pure function of `(weights, h)` with no RNG
    /// calls — this is the determinism guarantee `forward` actually needs
    /// to provide, and it holds regardless of how the weights were
    /// initialized. See `weight_initialization_is_only_deterministic_single_threaded`
    /// below for what is (and isn't) guaranteed about *initialization*.
    #[test]
    fn forward_is_deterministic_for_fixed_weights() {
        tensor_engine::seed(0);
        let scorer: TopologyScorer<ConcreteBackend> = TopologyScorer::new(5, 3, &device());
        let h: Tensor<ConcreteBackend, 2> = Tensor::random(
            [6, 5],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );

        let scores_a = scorer.forward(h.clone());
        let scores_b = scorer.forward(h);

        let diff: f32 = (scores_a - scores_b).abs().sum().into_scalar();
        assert!(
            diff < 1e-6,
            "forward should be deterministic for identical weights and input, diff={diff}"
        );
    }

    /// Documents a real limitation discovered while writing this crate,
    /// not a hypothetical one: Burn 0.21's `NdArray` backend fills random
    /// tensors (including `LinearConfig::init`'s weight initialization)
    /// using rayon-parallel chunks, and the RNG draw order — and therefore
    /// the resulting values — depends on how those chunks get scheduled
    /// across threads. `tensor_engine::seed` does reset the underlying RNG
    /// (see `tensor-engine`'s own `seeded_initialization_is_deterministic`
    /// test, which passes reliably because it's a single `Tensor::random`
    /// call with no parallel-module-init interaction), but two separately
    /// constructed `TopologyScorer`s with the same seed are only guaranteed
    /// to have identical weights when the process runs single-threaded
    /// (`RAYON_NUM_THREADS=1`) — confirmed by running this exact assertion
    /// both ways while developing this test.
    ///
    /// This matters for project spec §31/§32 ("deterministic seeds",
    /// reproducible experiments): as of this milestone, exactly
    /// reproducing a training run's *initial weights* across separate
    /// process runs requires pinning `RAYON_NUM_THREADS=1`, which is
    /// recorded in `PROJECT_STATUS.md` as a known issue for
    /// `training-engine` to either accept, work around, or fix upstream.
    /// This test is `#[ignore]`d rather than deleted, so the limitation
    /// stays documented and re-checkable, without failing CI on ordinary
    /// multi-threaded runs.
    #[test]
    #[ignore = "only deterministic with RAYON_NUM_THREADS=1 — see doc comment"]
    fn weight_initialization_is_only_deterministic_single_threaded() {
        let h: Tensor<ConcreteBackend, 2> = Tensor::random(
            [6, 5],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );

        tensor_engine::seed(123);
        let scores_a: Tensor<ConcreteBackend, 2> =
            TopologyScorer::new(5, 3, &device()).forward(h.clone());

        tensor_engine::seed(123);
        let scores_b: Tensor<ConcreteBackend, 2> = TopologyScorer::new(5, 3, &device()).forward(h);

        let diff: f32 = (scores_a - scores_b).abs().sum().into_scalar();
        assert!(
            diff < 1e-6,
            "same seed should give identical scores, diff={diff}"
        );
    }
}
