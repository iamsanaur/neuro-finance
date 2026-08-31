//! neuro-model
//!
//! `NeuroTopologicalFinancialModel` (project spec §24), built up to the
//! point this milestone reaches: feature encoder → dynamic topology
//! (`topology-engine`) → sparse graph message passing → regime
//! classification head (§25). See [`model`]'s module doc for exactly what
//! is and isn't implemented yet, and why.

pub mod encoder;
pub mod message_passing;
pub mod model;
pub mod regime_head;

pub use encoder::FeatureEncoder;
pub use message_passing::GraphMessagePassing;
pub use model::NeuroTopologicalFinancialModel;
pub use regime_head::{RegimeHead, NUM_REGIME_CLASSES};
