//! Basic topology-structure metrics (project spec §17: "Track: ... degree,
//! graph density, connected components, community structure").
//!
//! `degree` and `density` already live on `FinancialGraph` itself
//! (`financial-graph`'s crate, Milestone 5) — this module adds
//! [`connected_components`], the one structural metric that needs a graph
//! traversal rather than a simple count. Full community detection (e.g.
//! Louvain) is deliberately deferred to the topology-research milestone
//! (§34), which is where it's actually needed for interpretation; adding it
//! here now, with no research report yet to feed, would be exactly the
//! premature complexity project spec §2 warns against.

use financial_graph::{FinancialGraph, RelationType};
use std::collections::HashSet;

/// Number of connected components in the subgraph induced by `relation`,
/// via union-find. An isolated node (no edges of that relation type) counts
/// as its own component.
pub fn connected_components(graph: &FinancialGraph, relation: RelationType) -> usize {
    let n = graph.num_nodes();
    if n == 0 {
        return 0;
    }

    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    for edge in graph.edges_of_relation(relation) {
        union(&mut parent, edge.source.0 as usize, edge.target.0 as usize);
    }

    let roots: HashSet<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    roots.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use financial_graph::Edge;
    use financial_types::EntityId;

    fn ts() -> financial_types::Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn all_isolated_nodes_are_separate_components() {
        let graph = FinancialGraph::new(vec![
            EntityId::from("A"),
            EntityId::from("B"),
            EntityId::from("C"),
        ]);
        assert_eq!(connected_components(&graph, RelationType::Learned), 3);
    }

    #[test]
    fn a_single_edge_merges_two_components() {
        let mut graph = FinancialGraph::new(vec![
            EntityId::from("A"),
            EntityId::from("B"),
            EntityId::from("C"),
        ]);
        let a = graph.node_id(&EntityId::from("A")).unwrap();
        let b = graph.node_id(&EntityId::from("B")).unwrap();
        graph
            .add_edge(Edge {
                source: a,
                target: b,
                relation: RelationType::Learned,
                weight: 1.0,
                timestamp: ts(),
            })
            .unwrap();
        // A-B merged, C still isolated.
        assert_eq!(connected_components(&graph, RelationType::Learned), 2);
    }

    #[test]
    fn a_spanning_chain_is_one_component() {
        let mut graph = FinancialGraph::new(vec![
            EntityId::from("A"),
            EntityId::from("B"),
            EntityId::from("C"),
        ]);
        let a = graph.node_id(&EntityId::from("A")).unwrap();
        let b = graph.node_id(&EntityId::from("B")).unwrap();
        let c = graph.node_id(&EntityId::from("C")).unwrap();
        for (s, t) in [(a, b), (b, c)] {
            graph
                .add_edge(Edge {
                    source: s,
                    target: t,
                    relation: RelationType::Learned,
                    weight: 1.0,
                    timestamp: ts(),
                })
                .unwrap();
        }
        assert_eq!(connected_components(&graph, RelationType::Learned), 1);
    }

    #[test]
    fn empty_graph_has_zero_components() {
        let graph = FinancialGraph::new(vec![]);
        assert_eq!(connected_components(&graph, RelationType::Learned), 0);
    }
}
