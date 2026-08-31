//! `NeuroTopologicalFinancialModel`: wires the pieces built so far into the
//! forward path project spec §24 describes, up through the point this
//! milestone actually reaches:
//!
//! ```text
//! Input -> Feature Encoder -> Node Embeddings -> Dynamic Topology ->
//! Sparse Graph Message Passing -> [Global Attention -> Fusion ->
//! Temporal Encoding -> Memory Retrieval] -> Prediction Head
//! ```
//!
//! The bracketed stages — global attention/fusion (§20), temporal encoding
//! (§21), hierarchical graph (§22), financial memory (§23) — are **not**
//! implemented yet. This model is single-day, cross-sectional only: it has
//! no notion of a time sequence at all yet, let alone a causal one. Adding
//! them without a working baseline to compare against would be exactly the
//! "premature complexity" project spec §2 warns against; see
//! `PROJECT_STATUS.md` for what's next.
//!
//! ## A real, disclosed limitation: gradients don't flow through topology
//! selection
//!
//! `top_k_topology` is a hard (non-differentiable) selection over the score
//! matrix `TopologyScorer` produces. This model's forward path uses that
//! selection to build the graph `GraphMessagePassing` aggregates over, so
//! backpropagating a prediction loss through this model trains the
//! encoder, message passing, and prediction head, but **does not** train
//! `TopologyScorer`'s `W_q`/`W_k` — no gradient reaches them through this
//! path. `TopologyScorer` is trained separately, directly on the (fully
//! differentiable) score matrix, via `topology_engine::l_sparse` and
//! `l_stability` — this is a standard, accepted way to handle hard
//! selection in graph learning (the alternative, a differentiable
//! relaxation like Gumbel-softmax top-k, is real added complexity with no
//! evidence yet that it's needed — again, §2).

use crate::encoder::FeatureEncoder;
use crate::message_passing::GraphMessagePassing;
use crate::regime_head::RegimeHead;
use burn::module::Module;
use burn::tensor::Tensor;
use financial_graph::RelationType;
use financial_types::{EntityId, Timestamp};
use tensor_engine::burn;
use tensor_engine::Backend;
use topology_engine::{top_k_topology, TopologyScorer};

#[derive(Module, Debug, Clone)]
pub struct NeuroTopologicalFinancialModel {
    encoder: FeatureEncoder,
    topology_scorer: TopologyScorer,
    message_passing: GraphMessagePassing,
    regime_head: RegimeHead,
}

impl NeuroTopologicalFinancialModel {
    pub fn new(feature_dim: usize, embed_dim: usize, topology_proj_dim: usize) -> Self {
        Self {
            encoder: FeatureEncoder::new(feature_dim, embed_dim),
            topology_scorer: TopologyScorer::new(embed_dim, topology_proj_dim),
            message_passing: GraphMessagePassing::new(embed_dim),
            regime_head: RegimeHead::new(embed_dim),
        }
    }

    /// `features`: `[N, feature_dim]` causal features, one row per asset,
    /// row order matching `entities`. `top_k`: neighbors per node in the
    /// learned topology (see `topology_engine::top_k_topology`).
    /// `timestamp`: stamped onto the constructed topology graph (for
    /// bookkeeping — this model has no temporal component yet, so it plays
    /// no role in the computation itself).
    ///
    /// Returns `[1, 3]` regime class probabilities
    /// (`[RiskOn, Neutral, RiskOff]`).
    pub fn forward(
        &self,
        features: Tensor<Backend, 2>,
        entities: &[EntityId],
        top_k: usize,
        timestamp: Timestamp,
    ) -> Tensor<Backend, 2> {
        let h = self.encoder.forward(features);
        let scores = self.topology_scorer.forward(h.clone());
        let topology = top_k_topology(scores, entities, top_k, timestamp);
        let h = self
            .message_passing
            .forward(h, &topology, RelationType::Learned);
        self.regime_head.forward(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn ts() -> Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    fn entities(n: usize) -> Vec<EntityId> {
        (0..n).map(|i| EntityId::from(format!("E{i}"))).collect()
    }

    #[test]
    fn forward_produces_a_valid_probability_distribution() {
        tensor_engine::seed(0);
        let n = 20;
        let feature_dim = 6;
        let model = NeuroTopologicalFinancialModel::new(feature_dim, 16, 8);
        let features: Tensor<Backend, 2> = Tensor::random(
            [n, feature_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &tensor_engine::device(),
        );

        let probs = model.forward(features, &entities(n), 4, ts());
        assert_eq!(probs.dims(), [1, 3]);

        let sum: f32 = probs.clone().sum().into_scalar();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "probabilities should sum to 1.0, got {sum}"
        );
        let min: f32 = probs.min().into_scalar();
        assert!(min >= 0.0);
    }

    #[test]
    fn forward_is_deterministic_for_fixed_weights() {
        tensor_engine::seed(0);
        let n = 15;
        let model = NeuroTopologicalFinancialModel::new(5, 12, 6);
        let features: Tensor<Backend, 2> = Tensor::random(
            [n, 5],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &tensor_engine::device(),
        );
        let ents = entities(n);

        let probs_a = model.forward(features.clone(), &ents, 3, ts());
        let probs_b = model.forward(features, &ents, 3, ts());

        let diff: f32 = (probs_a - probs_b).abs().sum().into_scalar();
        assert!(
            diff < 1e-6,
            "forward should be deterministic for fixed weights, diff={diff}"
        );
    }
}
