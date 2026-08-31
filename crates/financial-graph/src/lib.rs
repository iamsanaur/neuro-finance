//! financial-graph
//!
//! A sparse, multi-relational graph over financial entities (project spec
//! §12–§15): `FinancialGraph` (nodes, edges, `RelationType`), plus builders
//! for the two static graphs V0.1 needs — sector co-membership and rolling
//! return correlation.
//!
//! Not yet implemented: industry, fundamental, macro, and news graphs
//! (V0.2+, once `data-engine` produces that data — see
//! `feature-engine::lib`'s doc for the same reasoning); the per-relation
//! learnable importance weights `alpha_k` (§15) and the dynamic/learned
//! topology itself (§16) belong to `topology-engine`, a later crate — this
//! one only builds and stores graphs, it doesn't learn them.

pub mod correlation;
pub mod graph;
pub mod sector;
pub mod types;

pub use correlation::{build_correlation_graph, CorrelationGraphConfig};
pub use graph::FinancialGraph;
pub use sector::build_sector_graph;
pub use types::{Edge, EdgeId, GraphError, NodeId, RelationType};
