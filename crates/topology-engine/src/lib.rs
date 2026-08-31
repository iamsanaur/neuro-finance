//! topology-engine
//!
//! The dynamic topology learner (project spec §16–§18):
//! - [`scorer::TopologyScorer`] — learnable `Q`/`K` projections producing a
//!   dense `[N, N]` relationship-score matrix from node embeddings.
//! - [`topk::top_k_topology`] — reduces that dense matrix to a sparse,
//!   bounded-degree (`<= k`) `FinancialGraph` via mutual top-k selection.
//! - [`persistence::TopologyPersistence`] — EMA-blends topology across
//!   time, and reports edge creation/deletion/persistence.
//! - [`regularization`] — `L_sparse`, `L_stability` (differentiable, over
//!   the dense score matrix) and `L_relation` (a plain scalar, over
//!   materialized graphs).
//! - [`analysis::connected_components`] — a basic structural metric; full
//!   community detection is deferred to the topology-research milestone
//!   (§34).
//!
//! This crate produces and scores topology; it does not train anything
//! (there is no `L_prediction` term yet — that requires `neuro-model` and
//! `training-engine`, both later milestones) and it does not decide what
//! node embeddings to score (that's `neuro-model`'s feature encoder).

pub mod analysis;
pub mod persistence;
pub mod regularization;
pub mod scorer;
pub mod topk;

pub use analysis::connected_components;
pub use persistence::{TopologyDiff, TopologyPersistence};
pub use regularization::{l_relation, l_sparse, l_stability};
pub use scorer::TopologyScorer;
pub use topk::top_k_topology;
