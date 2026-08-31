//! Core graph types: node/edge identifiers, relation types, and the edge
//! record itself.

use financial_types::Timestamp;

/// Index into a [`crate::graph::FinancialGraph`]'s node table. Node
/// identity is otherwise an [`financial_types::EntityId`] — see
/// [`crate::graph::FinancialGraph::new`] for why `EntityId` (not `Symbol`)
/// is the graph's node-identity type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Index into a [`crate::graph::FinancialGraph`]'s edge table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub u32);

/// The kind of relationship an edge represents (project spec §12).
///
/// V0.1 only ever constructs `Sector` and `Correlation` edges (per §50's
/// scope); the rest of the variants exist now so the graph's storage layer
/// and every downstream consumer (`topology-engine`, `neuro-model`) can be
/// written against the full relation space from the start, rather than
/// needing a breaking change when `Industry`/`Fundamental`/`Macro`/`News`
/// graphs are added in V0.2 (§51) or `Learned` edges are produced by the
/// topology learner (§16, later milestone).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationType {
    Sector,
    Industry,
    Correlation,
    Fundamental,
    Macro,
    News,
    Learned,
}

/// One edge between two nodes. Graphs in this crate are undirected — an
/// edge is stored once and is reachable from both endpoints' adjacency
/// lists (see [`crate::graph::FinancialGraph::neighbors`]) — because every
/// V0.1 relation type (sector co-membership, correlation) is symmetric by
/// construction. A directed relation (e.g. a supply-chain edge) would need
/// its own type; none is needed yet.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub relation: RelationType,
    /// Relation-specific: `1.0` for an unweighted sector co-membership
    /// edge, the correlation coefficient (`[-1.0, 1.0]`) for a correlation
    /// edge, etc. Documented per-builder (`sector.rs`, `correlation.rs`),
    /// not constrained generically here — different relation types have
    /// legitimately different weight semantics.
    pub weight: f64,
    /// When this edge was computed/observed. For a correlation edge this is
    /// the `as_of` timestamp the caller built it for — see
    /// `correlation.rs`'s point-in-time-safety test for why this must never
    /// be later than the data the edge was actually derived from.
    pub timestamp: Timestamp,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GraphError {
    #[error("node index {0:?} is out of range (graph has {1} nodes)")]
    NodeOutOfRange(NodeId, usize),
    #[error("self-loop rejected: source and target are both {0:?}")]
    SelfLoop(NodeId),
}
