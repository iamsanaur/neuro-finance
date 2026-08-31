//! `FinancialGraph`: sparse, multi-relational, undirected graph over
//! financial entities (project spec §12–§13).
//!
//! **Sparse by construction**: storage is a flat edge list plus per-node
//! adjacency lists of edge indices — there is no `N x N` matrix anywhere in
//! this type, and none can appear by accident, because there is no API that
//! takes or returns one. For a 100-node universe this doesn't matter much,
//! but the point (§13: "Do NOT default to dense NxN matrices") is to not
//! build a habit that breaks at the entity counts a real universe reaches.

use crate::types::{Edge, EdgeId, GraphError, NodeId, RelationType};
use financial_types::EntityId;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct FinancialGraph {
    /// `NodeId(i).0 as usize` indexes this directly.
    nodes: Vec<EntityId>,
    node_index: HashMap<EntityId, NodeId>,
    edges: Vec<Edge>,
    /// `adjacency[i]` holds `(neighbor, edge_index)` pairs for node `i`,
    /// populated symmetrically for both endpoints of every edge (see the
    /// module doc: this graph is undirected).
    adjacency: Vec<Vec<(NodeId, usize)>>,
}

impl FinancialGraph {
    /// Node identity is [`EntityId`], not [`financial_types::Symbol`] —
    /// `EntityId` is the type `financial-types` already documents as the
    /// general graph-node identity (a company, a sector, or any other
    /// entity the graph references), so builders that produce
    /// non-tradable nodes (e.g. a future sector-rollup hierarchy, §22) fit
    /// the same graph without a second node-identity type. Builders that
    /// work from tradable assets (`sector.rs`, `correlation.rs`) convert
    /// `Symbol -> EntityId` at their own boundary.
    ///
    /// Panics if `nodes` contains a duplicate `EntityId` — node identity
    /// must be unique for `node_id` lookups to be well-defined.
    pub fn new(nodes: Vec<EntityId>) -> Self {
        let mut node_index = HashMap::with_capacity(nodes.len());
        for (i, entity) in nodes.iter().enumerate() {
            let prev = node_index.insert(entity.clone(), NodeId(i as u32));
            assert!(prev.is_none(), "duplicate node identity: {entity}");
        }
        let adjacency = vec![Vec::new(); nodes.len()];
        Self {
            nodes,
            node_index,
            edges: Vec::new(),
            adjacency,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn node_id(&self, entity: &EntityId) -> Option<NodeId> {
        self.node_index.get(entity).copied()
    }

    pub fn entity(&self, node: NodeId) -> Option<&EntityId> {
        self.nodes.get(node.0 as usize)
    }

    /// Adds an undirected edge. Rejects an out-of-range node id or a
    /// self-loop (project spec's property-test list, §47: "graph edges
    /// reference valid nodes").
    pub fn add_edge(&mut self, edge: Edge) -> Result<EdgeId, GraphError> {
        let n = self.nodes.len();
        if edge.source.0 as usize >= n {
            return Err(GraphError::NodeOutOfRange(edge.source, n));
        }
        if edge.target.0 as usize >= n {
            return Err(GraphError::NodeOutOfRange(edge.target, n));
        }
        if edge.source == edge.target {
            return Err(GraphError::SelfLoop(edge.source));
        }

        let index = self.edges.len();
        self.adjacency[edge.source.0 as usize].push((edge.target, index));
        self.adjacency[edge.target.0 as usize].push((edge.source, index));
        self.edges.push(edge);
        Ok(EdgeId(index as u32))
    }

    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(id.0 as usize)
    }

    /// Neighbors of `node`, optionally filtered to one [`RelationType`].
    pub fn neighbors(
        &self,
        node: NodeId,
        relation: Option<RelationType>,
    ) -> impl Iterator<Item = (NodeId, &Edge)> + '_ {
        self.adjacency
            .get(node.0 as usize)
            .into_iter()
            .flatten()
            .filter_map(move |&(nbr, idx)| {
                let edge = &self.edges[idx];
                match relation {
                    Some(r) if edge.relation != r => None,
                    _ => Some((nbr, edge)),
                }
            })
    }

    pub fn degree(&self, node: NodeId, relation: Option<RelationType>) -> usize {
        self.neighbors(node, relation).count()
    }

    pub fn edges_of_relation(&self, relation: RelationType) -> impl Iterator<Item = &Edge> {
        self.edges.iter().filter(move |e| e.relation == relation)
    }

    /// Graph density restricted to one relation type: realized edge count
    /// divided by the maximum possible (`n * (n-1) / 2` for an undirected
    /// simple graph). A basic topology-research metric (§34); more will be
    /// added once there's a learned topology to actually research.
    pub fn density(&self, relation: RelationType) -> f64 {
        let n = self.nodes.len();
        if n < 2 {
            return 0.0;
        }
        let max_edges = (n * (n - 1) / 2) as f64;
        self.edges_of_relation(relation).count() as f64 / max_edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn ts() -> financial_types::Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    fn small_graph() -> FinancialGraph {
        FinancialGraph::new(vec![
            EntityId::from("A"),
            EntityId::from("B"),
            EntityId::from("C"),
        ])
    }

    #[test]
    fn node_lookup_round_trips() {
        let graph = small_graph();
        let id = graph.node_id(&EntityId::from("B")).unwrap();
        assert_eq!(graph.entity(id), Some(&EntityId::from("B")));
    }

    #[test]
    #[should_panic(expected = "duplicate node identity")]
    fn rejects_duplicate_nodes() {
        FinancialGraph::new(vec![EntityId::from("A"), EntityId::from("A")]);
    }

    #[test]
    fn add_edge_is_visible_from_both_endpoints() {
        let mut graph = small_graph();
        let a = graph.node_id(&EntityId::from("A")).unwrap();
        let b = graph.node_id(&EntityId::from("B")).unwrap();
        graph
            .add_edge(Edge {
                source: a,
                target: b,
                relation: RelationType::Sector,
                weight: 1.0,
                timestamp: ts(),
            })
            .unwrap();

        let a_neighbors: Vec<NodeId> = graph.neighbors(a, None).map(|(n, _)| n).collect();
        let b_neighbors: Vec<NodeId> = graph.neighbors(b, None).map(|(n, _)| n).collect();
        assert_eq!(a_neighbors, vec![b]);
        assert_eq!(b_neighbors, vec![a]);
    }

    #[test]
    fn rejects_self_loop() {
        let mut graph = small_graph();
        let a = graph.node_id(&EntityId::from("A")).unwrap();
        let result = graph.add_edge(Edge {
            source: a,
            target: a,
            relation: RelationType::Sector,
            weight: 1.0,
            timestamp: ts(),
        });
        assert_eq!(result, Err(GraphError::SelfLoop(a)));
    }

    #[test]
    fn rejects_out_of_range_node() {
        let mut graph = small_graph();
        let a = graph.node_id(&EntityId::from("A")).unwrap();
        let bogus = NodeId(99);
        let result = graph.add_edge(Edge {
            source: a,
            target: bogus,
            relation: RelationType::Sector,
            weight: 1.0,
            timestamp: ts(),
        });
        assert_eq!(result, Err(GraphError::NodeOutOfRange(bogus, 3)));
    }

    #[test]
    fn neighbors_filters_by_relation() {
        let mut graph = small_graph();
        let a = graph.node_id(&EntityId::from("A")).unwrap();
        let b = graph.node_id(&EntityId::from("B")).unwrap();
        let c = graph.node_id(&EntityId::from("C")).unwrap();
        graph
            .add_edge(Edge {
                source: a,
                target: b,
                relation: RelationType::Sector,
                weight: 1.0,
                timestamp: ts(),
            })
            .unwrap();
        graph
            .add_edge(Edge {
                source: a,
                target: c,
                relation: RelationType::Correlation,
                weight: 0.5,
                timestamp: ts(),
            })
            .unwrap();

        let sector_neighbors: Vec<NodeId> = graph
            .neighbors(a, Some(RelationType::Sector))
            .map(|(n, _)| n)
            .collect();
        assert_eq!(sector_neighbors, vec![b]);
        assert_eq!(graph.degree(a, None), 2);
        assert_eq!(graph.degree(a, Some(RelationType::Correlation)), 1);
    }

    #[test]
    fn density_of_complete_graph_is_one() {
        let mut graph = small_graph(); // 3 nodes
        let a = graph.node_id(&EntityId::from("A")).unwrap();
        let b = graph.node_id(&EntityId::from("B")).unwrap();
        let c = graph.node_id(&EntityId::from("C")).unwrap();
        for (s, t) in [(a, b), (b, c), (a, c)] {
            graph
                .add_edge(Edge {
                    source: s,
                    target: t,
                    relation: RelationType::Sector,
                    weight: 1.0,
                    timestamp: ts(),
                })
                .unwrap();
        }
        assert!((graph.density(RelationType::Sector) - 1.0).abs() < 1e-12);
    }
}
