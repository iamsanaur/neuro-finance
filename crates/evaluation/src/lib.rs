//! evaluation
//!
//! Baseline models (project spec §28) and classification metrics needed to
//! judge whether `neuro-model` actually earns its complexity.
//!
//! Implemented: `NaivePersistenceBaseline`, `MajorityClassBaseline`,
//! `LogisticRegressionBaseline`, and `classification_report` (accuracy,
//! per-class precision/recall/F1). Deferred: gradient boosting, MLP,
//! LSTM/GRU, Transformer, static/graph-Transformer baselines; AUC,
//! directional accuracy, and regression error metrics (no binary-direction
//! or regression task exists yet — §26/§27); the ablation harness (§33) and
//! research report generation (§55), both of which need multiple trained
//! models to compare, not just one.

pub mod baseline;
pub mod logistic;
pub mod metrics;

pub use baseline::{MajorityClassBaseline, NaivePersistenceBaseline};
pub use logistic::LogisticRegressionBaseline;
pub use metrics::{classification_report, ClassMetrics, ClassificationReport};
