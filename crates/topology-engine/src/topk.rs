//! Sparse top-k selection: reduces a dense `[N, N]` score matrix to a
//! bounded-degree undirected [`FinancialGraph`] with `RelationType::Learned`
//! edges (project spec §16).
//!
//! `s_ij` is a directed notion (`N(i) = TopK_j(s_ij)` need not equal
//! `N(j) = TopK_j(s_ji)`), but every graph this workspace deals with is
//! undirected (see `financial-graph`'s module doc). This module resolves
//! that with **mutual top-k**: an edge `(i, j)` survives only if `j` is in
//! `i`'s top-k *and* `i` is in `j`'s top-k. This is what gives the property
//! project spec §47 asks for directly — "topology degree <= configured
//! top-k" — a union rule (edge if *either* direction selects it) would not
//! guarantee that bound, since a popular node could be selected by more
//! than `k` others without ever selecting them back.

use burn::tensor::Tensor;
use financial_graph::{Edge, FinancialGraph, NodeId, RelationType};
use financial_types::{EntityId, Timestamp};
use std::collections::HashSet;
use tensor_engine::burn;
use tensor_engine::Backend;

/// `scores`: `[N, N]` from [`crate::scorer::TopologyScorer::forward`].
/// `entities[i]` is node `i`'s identity, in the same order the embeddings
/// that produced `scores` were in. `k` is the top-k neighbor count per node
/// (project spec §16: configurable, typically 4/8/16/32).
///
/// Panics if `entities.len()` doesn't match `scores`' dimensions, or if
/// `k >= entities.len()` (there's no meaningful "other" node to exclude
/// self-selection against otherwise).
pub fn top_k_topology(
    scores: Tensor<Backend, 2>,
    entities: &[EntityId],
    k: usize,
    timestamp: Timestamp,
) -> FinancialGraph {
    let n = entities.len();
    assert_eq!(
        scores.dims(),
        [n, n],
        "scores must be [N, N] with N = entities.len()"
    );
    assert!(k < n, "k ({k}) must be less than the number of nodes ({n})");

    let data: Vec<f32> = scores
        .into_data()
        .to_vec()
        .expect("score tensor should hold f32 values");

    let row = |i: usize| -> &[f32] { &data[i * n..(i + 1) * n] };

    // top_k_of[i] = the set of column indices in i's top-k (excluding i itself).
    let top_k_of: Vec<HashSet<usize>> = (0..n)
        .map(|i| {
            let mut scored: Vec<(usize, f32)> =
                (0..n).filter(|&j| j != i).map(|j| (j, row(i)[j])).collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("scores must not be NaN"));
            scored.into_iter().take(k).map(|(j, _)| j).collect()
        })
        .collect();

    let mut graph = FinancialGraph::new(entities.to_vec());
    for i in 0..n {
        for &j in &top_k_of[i] {
            if j <= i {
                continue; // undirected: consider each pair once, i < j
            }
            if top_k_of[j].contains(&i) {
                let weight = ((row(i)[j] + row(j)[i]) / 2.0) as f64;
                graph
                    .add_edge(Edge {
                        source: NodeId(i as u32),
                        target: NodeId(j as u32),
                        relation: RelationType::Learned,
                        weight,
                        timestamp,
                    })
                    .expect("indices come from the graph's own node table");
            }
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use tensor_engine::device;

    fn ts() -> Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    fn entities(n: usize) -> Vec<EntityId> {
        (0..n).map(|i| EntityId::from(format!("E{i}"))).collect()
    }

    /// The concrete form of project spec §47's property test: "topology
    /// degree <= configured top-k".
    #[test]
    fn degree_never_exceeds_k() {
        tensor_engine::seed(1);
        let n = 20;
        let k = 4;
        let scores: Tensor<Backend, 2> = Tensor::random(
            [n, n],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let ents = entities(n);
        let graph = top_k_topology(scores, &ents, k, ts());

        for (i, entity) in ents.iter().enumerate() {
            let node = graph.node_id(entity).unwrap();
            let degree = graph.degree(node, Some(RelationType::Learned));
            assert!(degree <= k, "node {i} has degree {degree} > k={k}");
        }
    }

    #[test]
    fn mutual_top_k_excludes_one_sided_selections() {
        // Construct scores by hand: node 0 strongly prefers node 1, but
        // node 1 strongly prefers node 2 (not node 0). With k=1, 0->1 is
        // not mutual (1's only top-1 is 2), so no edge should form there;
        // 1<->2 should form only if it's mutual too.
        let device = device();
        let n = 3;
        // scores[i][j]: row-major, i*n+j
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            0.0, 5.0, 1.0, // node 0: prefers 1 (5.0) over 2 (1.0)
            1.0, 0.0, 5.0, // node 1: prefers 2 (5.0) over 0 (1.0)
            5.0, 1.0, 0.0, // node 2: prefers 0 (5.0) over 1 (1.0)
        ];
        let scores: Tensor<Backend, 2> =
            Tensor::<Backend, 1>::from_data(data.as_slice(), &device).reshape([n, n]);
        let ents = entities(n);
        let graph = top_k_topology(scores, &ents, 1, ts());

        // It's a 3-cycle of one-directional preferences (0->1, 1->2, 2->0),
        // none mutual — no edges should survive at k=1.
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    #[should_panic(expected = "k (5) must be less than the number of nodes (5)")]
    fn rejects_k_at_or_above_node_count() {
        let device = device();
        let n = 5;
        let scores: Tensor<Backend, 2> = Tensor::zeros([n, n], &device);
        top_k_topology(scores, &entities(n), 5, ts());
    }
}
