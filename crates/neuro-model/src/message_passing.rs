//! `GraphMessagePassing`: basic sparse graph aggregation (project spec §19,
//! variant 1 — "basic graph aggregation"; graph attention and dynamic
//! sparse attention, §19's variants 2–3, are deferred to a later milestone
//! once there's a baseline result to compare them against, §28).
//!
//! ```text
//! m_i  = sum_{j in N(i)} alpha_ij * W_v * h_j
//! h'_i = h_i + MLP(m_i)
//! ```
//!
//! `alpha_ij` here is mean aggregation weighted by edge weight
//! (`edge_weight_ij / degree(i)`) — the simplest well-defined choice; a
//! learned attention weight (§19 variant 2) is exactly what's deferred.
//!
//! ## On the `[N, N]` adjacency this builds
//!
//! `financial-graph`'s storage is sparse (§13); this function still builds
//! a dense `[N, N]` tensor internally to express aggregation as one matmul.
//! That is a deliberate, scale-appropriate choice, not a violation of §13:
//! §13 is about not defaulting to dense storage in the *graph data
//! structure* (which stays sparse — this function reads it via
//! `graph.neighbors`, never assumes density), and dense-matmul aggregation
//! over a `[N, N]` matrix is completely standard for GNNs at the node
//! counts this project targets (V0.1: `N=100`; even V1.0's larger
//! universes are nowhere near where a `[N, N]` `f32` matrix stops being
//! trivial — 10,000 assets is 400MB, at the outer edge of what's still
//! fine). If `N` ever grows enough for this to matter, the fix is a sparse
//! matmul kernel, not a redesign of `FinancialGraph`'s storage.
//!
//! Generic over `B: Backend` — see `topology_engine::scorer`'s doc comment
//! for why (a real gradient-tracking bug affecting every concrete-backend
//! `#[derive(Module)]` struct, found in Milestone 8). The adjacency tensor
//! is built on `h`'s own device (`h.device()`), so `forward` needs no
//! separate device argument.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use financial_graph::{FinancialGraph, NodeId, RelationType};
use tensor_engine::burn;

#[derive(Module, Debug)]
pub struct GraphMessagePassing<B: Backend> {
    value: Linear<B>,
    update_mlp: Linear<B>,
}

impl<B: Backend> GraphMessagePassing<B> {
    pub fn new(embed_dim: usize, device: &B::Device) -> Self {
        Self {
            value: LinearConfig::new(embed_dim, embed_dim).init(device),
            update_mlp: LinearConfig::new(embed_dim, embed_dim).init(device),
        }
    }

    /// `h`: `[N, embed_dim]` node embeddings, in `NodeId` order (`h`'s row
    /// `i` is `graph`'s `NodeId(i)`) — this is guaranteed by construction
    /// whenever `graph` and `h` were built from the same asset ordering
    /// (e.g. both from `data-engine`'s `sector_assignment` order), which is
    /// the only usage this crate supports in V0.1; there is no runtime
    /// check that the ordering actually matches; a mismatch would silently
    /// aggregate the wrong node's messages against another's embedding.
    /// `graph.num_nodes()` must equal `h.dims()[0]`.
    pub fn forward(
        &self,
        h: Tensor<B, 2>,
        graph: &FinancialGraph,
        relation: RelationType,
    ) -> Tensor<B, 2> {
        let n = graph.num_nodes();
        assert_eq!(h.dims()[0], n, "h's row count must match graph.num_nodes()");

        let mut adjacency = vec![0f32; n * n];
        for i in 0..n {
            let neighbors: Vec<_> = graph.neighbors(NodeId(i as u32), Some(relation)).collect();
            let degree = neighbors.len();
            if degree == 0 {
                continue;
            }
            for (neighbor, edge) in neighbors {
                adjacency[i * n + neighbor.0 as usize] = (edge.weight as f32) / degree as f32;
            }
        }
        let adjacency: Tensor<B, 2> =
            Tensor::<B, 1>::from_data(adjacency.as_slice(), &h.device()).reshape([n, n]);

        let v = self.value.forward(h.clone());
        let messages = adjacency.matmul(v);
        let update = relu(self.update_mlp.forward(messages));
        h + update
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use financial_graph::Edge;
    use financial_types::EntityId;
    use tensor_engine::{device, Backend as ConcreteBackend};

    fn ts() -> financial_types::Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    fn chain_graph(n: usize) -> FinancialGraph {
        let entities: Vec<EntityId> = (0..n).map(|i| EntityId::from(format!("E{i}"))).collect();
        let mut graph = FinancialGraph::new(entities);
        for i in 0..n - 1 {
            graph
                .add_edge(Edge {
                    source: NodeId(i as u32),
                    target: NodeId((i + 1) as u32),
                    relation: RelationType::Learned,
                    weight: 1.0,
                    timestamp: ts(),
                })
                .unwrap();
        }
        graph
    }

    #[test]
    fn forward_preserves_shape() {
        tensor_engine::seed(0);
        let n = 6;
        let mp: GraphMessagePassing<ConcreteBackend> = GraphMessagePassing::new(8, &device());
        let h: Tensor<ConcreteBackend, 2> = Tensor::zeros([n, 8], &device());
        let graph = chain_graph(n);
        let out = mp.forward(h, &graph, RelationType::Learned);
        assert_eq!(out.dims(), [n, 8]);
    }

    /// An isolated node (no edges of the given relation) has no messages to
    /// aggregate, so `h'_i = h_i + MLP(0)` — not necessarily `h_i` exactly
    /// (the MLP has a bias), but the *messages* term must be exactly zero.
    /// Checked indirectly: an isolated node's embedding must differ from
    /// its input by exactly `relu(update_mlp.bias)`, the same delta
    /// regardless of what any other (connected) node's embedding is —
    /// i.e. it must not depend on the graph at all.
    #[test]
    fn isolated_node_output_is_independent_of_graph_structure() {
        tensor_engine::seed(1);
        let mp: GraphMessagePassing<ConcreteBackend> = GraphMessagePassing::new(4, &device());
        let h: Tensor<ConcreteBackend, 2> = Tensor::zeros([3, 4], &device());

        // Node 0 isolated in both graphs; graph_a has an edge 1-2, graph_b
        // does not (but node 0 is unaffected either way).
        let entities: Vec<EntityId> = (0..3).map(|i| EntityId::from(format!("E{i}"))).collect();
        let mut graph_a = FinancialGraph::new(entities.clone());
        graph_a
            .add_edge(Edge {
                source: NodeId(1),
                target: NodeId(2),
                relation: RelationType::Learned,
                weight: 1.0,
                timestamp: ts(),
            })
            .unwrap();
        let graph_b = FinancialGraph::new(entities);

        let out_a = mp.forward(h.clone(), &graph_a, RelationType::Learned);
        let out_b = mp.forward(h, &graph_b, RelationType::Learned);

        let row0_a = out_a.clone().slice([0..1, 0..4]);
        let row0_b = out_b.clone().slice([0..1, 0..4]);
        let diff: f32 = (row0_a - row0_b).abs().sum().into_scalar();
        assert!(
            diff < 1e-6,
            "isolated node's output should not depend on the rest of the graph, diff={diff}"
        );
    }
}
