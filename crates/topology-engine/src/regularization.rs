//! Topology regularization losses (project spec §18):
//!
//! ```text
//! L = L_prediction + lambda_s * L_sparse + lambda_t * L_stability + lambda_r * L_relation
//! ```
//!
//! `L_prediction` doesn't exist yet — it comes from `neuro-model`, which
//! doesn't exist yet either (this is the first crate past the graph layer;
//! the model comes after it). `L_sparse` and `L_stability` are implemented
//! here as differentiable `Tensor` losses over the *dense score matrix*
//! (before top-k thresholding) specifically so `training-engine` can sum
//! them directly into a total loss and backpropagate through
//! `TopologyScorer`'s `W_q`/`W_k` once it exists — that's also why they
//! don't operate on the sparse post-top-k graph, whose hard selection isn't
//! differentiable. `L_relation` is the exception: it compares two already-
//! materialized graphs (the learned one and a reference, e.g. the sector or
//! correlation graph from `financial-graph`) and is necessarily a plain
//! scalar, not a `Tensor` — there's no way to backpropagate through "is
//! this edge in that other graph's edge set."

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use financial_graph::{FinancialGraph, RelationType};
use tensor_engine::burn;

/// Mean absolute score across the whole `[N, N]` matrix — discourages
/// large-magnitude scores in general, which (combined with top-k selection)
/// discourages the learner from committing strongly to unnecessary edges.
///
/// Generic over `B` (any backend) rather than fixed to
/// `tensor_engine::Backend`, matching `TopologyScorer<B>` — see that
/// struct's doc comment for why concrete-backend fields specifically (not
/// free functions like this one) were the actual source of the Milestone 8
/// gradient-tracking bug; this function was never affected, but stays
/// generic for symmetry with what it's meant to be summed against.
pub fn l_sparse<B: Backend>(scores: Tensor<B, 2>) -> Tensor<B, 1> {
    scores.abs().mean()
}

/// Mean squared difference between two consecutive score matrices —
/// discourages the topology from oscillating wildly step to step. Both
/// tensors must have the same `[N, N]` shape (i.e. the same node set/order
/// as each other).
pub fn l_stability<B: Backend>(scores_t: Tensor<B, 2>, scores_prev: Tensor<B, 2>) -> Tensor<B, 1> {
    (scores_t - scores_prev).powf_scalar(2.0).mean()
}

/// `1.0 - overlap`, where overlap is the fraction of `learned`'s edges that
/// are also present (any relation type) in `reference`. `0.0` when every
/// learned edge is corroborated by the reference graph, `1.0` when none
/// are — including when `learned` has no edges at all (there is nothing to
/// overlap, which is deliberately treated as "no evidence of consistency,"
/// not "trivially consistent").
///
/// This encourages, but does not force, agreement with known relationships
/// (project spec §18: "without forcing them") — it's one term among several
/// in the total loss, and `configs/default.toml` sets its weight
/// (`lambda_relation`) to `0.0` by default in V0.1, since no relation graph
/// is wired into training yet.
pub fn l_relation(learned: &FinancialGraph, reference: &FinancialGraph) -> f64 {
    let learned_edges: Vec<(financial_graph::NodeId, financial_graph::NodeId)> = learned
        .edges_of_relation(RelationType::Learned)
        .map(|e| (e.source, e.target))
        .collect();
    if learned_edges.is_empty() {
        return 1.0;
    }

    let mut overlap_count = 0usize;
    for (source, target) in &learned_edges {
        let a = learned
            .entity(*source)
            .expect("edge endpoint must be valid");
        let b = learned
            .entity(*target)
            .expect("edge endpoint must be valid");
        let (Some(ra), Some(rb)) = (reference.node_id(a), reference.node_id(b)) else {
            continue; // node doesn't exist in the reference graph at all
        };
        if reference.neighbors(ra, None).any(|(n, _)| n == rb) {
            overlap_count += 1;
        }
    }

    1.0 - (overlap_count as f64 / learned_edges.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use financial_graph::Edge;
    use financial_types::{EntityId, Timestamp};
    use tensor_engine::{device, Backend as ConcreteBackend};

    fn ts() -> Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn l_sparse_is_zero_for_all_zero_scores() {
        let scores: Tensor<ConcreteBackend, 2> = Tensor::zeros([5, 5], &device());
        let loss: f32 = l_sparse(scores).into_scalar();
        assert!(loss.abs() < 1e-9);
    }

    #[test]
    fn l_sparse_increases_with_score_magnitude() {
        let small: Tensor<ConcreteBackend, 2> = Tensor::ones([4, 4], &device()) * 0.1;
        let large: Tensor<ConcreteBackend, 2> = Tensor::ones([4, 4], &device()) * 2.0;
        let loss_small: f32 = l_sparse(small).into_scalar();
        let loss_large: f32 = l_sparse(large).into_scalar();
        assert!(loss_large > loss_small);
    }

    #[test]
    fn l_stability_is_zero_for_identical_matrices() {
        let scores: Tensor<ConcreteBackend, 2> = Tensor::ones([4, 4], &device());
        let loss: f32 = l_stability(scores.clone(), scores).into_scalar();
        assert!(loss.abs() < 1e-9);
    }

    #[test]
    fn l_stability_increases_with_divergence() {
        let a: Tensor<ConcreteBackend, 2> = Tensor::zeros([4, 4], &device());
        let b_small: Tensor<ConcreteBackend, 2> = Tensor::ones([4, 4], &device()) * 0.1;
        let b_large: Tensor<ConcreteBackend, 2> = Tensor::ones([4, 4], &device()) * 1.0;
        let loss_small: f32 = l_stability(a.clone(), b_small).into_scalar();
        let loss_large: f32 = l_stability(a, b_large).into_scalar();
        assert!(loss_large > loss_small);
    }

    fn graph_with_edges(pairs: &[(&str, &str)]) -> FinancialGraph {
        let mut entities: Vec<EntityId> = pairs
            .iter()
            .flat_map(|(a, b)| [*a, *b])
            .map(EntityId::from)
            .collect();
        entities.sort();
        entities.dedup();
        let mut g = FinancialGraph::new(entities);
        for (a, b) in pairs {
            let na = g.node_id(&EntityId::from(*a)).unwrap();
            let nb = g.node_id(&EntityId::from(*b)).unwrap();
            g.add_edge(Edge {
                source: na,
                target: nb,
                relation: RelationType::Learned,
                weight: 1.0,
                timestamp: ts(),
            })
            .unwrap();
        }
        g
    }

    #[test]
    fn l_relation_is_zero_when_fully_corroborated() {
        let learned = graph_with_edges(&[("A", "B"), ("B", "C")]);
        let mut reference = FinancialGraph::new(vec![
            EntityId::from("A"),
            EntityId::from("B"),
            EntityId::from("C"),
        ]);
        for (a, b) in [("A", "B"), ("B", "C")] {
            let na = reference.node_id(&EntityId::from(a)).unwrap();
            let nb = reference.node_id(&EntityId::from(b)).unwrap();
            reference
                .add_edge(Edge {
                    source: na,
                    target: nb,
                    relation: RelationType::Sector,
                    weight: 1.0,
                    timestamp: ts(),
                })
                .unwrap();
        }
        assert!((l_relation(&learned, &reference)).abs() < 1e-9);
    }

    #[test]
    fn l_relation_is_one_when_no_edges_agree() {
        let learned = graph_with_edges(&[("A", "B")]);
        let reference = FinancialGraph::new(vec![
            EntityId::from("A"),
            EntityId::from("B"),
            EntityId::from("C"),
        ]);
        // reference has A, B, C as nodes but no edges at all.
        assert!((l_relation(&learned, &reference) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn l_relation_partial_overlap() {
        let learned = graph_with_edges(&[("A", "B"), ("C", "D")]);
        let mut reference = FinancialGraph::new(vec![
            EntityId::from("A"),
            EntityId::from("B"),
            EntityId::from("C"),
            EntityId::from("D"),
        ]);
        let na = reference.node_id(&EntityId::from("A")).unwrap();
        let nb = reference.node_id(&EntityId::from("B")).unwrap();
        reference
            .add_edge(Edge {
                source: na,
                target: nb,
                relation: RelationType::Sector,
                weight: 1.0,
                timestamp: ts(),
            })
            .unwrap();
        // C-D is not in reference at all.
        assert!((l_relation(&learned, &reference) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn l_relation_is_one_for_an_empty_learned_graph() {
        let learned = FinancialGraph::new(vec![EntityId::from("A"), EntityId::from("B")]);
        let reference = FinancialGraph::new(vec![EntityId::from("A"), EntityId::from("B")]);
        assert!((l_relation(&learned, &reference) - 1.0).abs() < 1e-9);
    }
}
