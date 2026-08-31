//! training-engine
//!
//! `WalkForwardValidator` (§29), mini-batch training loop, checkpointing,
//! and early stopping (§31) for `neuro-model`.
//!
//! Baseline models (§28) are **not** here — they belong to `evaluation`
//! (see that crate's description in the workspace root `Cargo.toml`),
//! which is where they're actually needed (comparing a trained model
//! against them), not here.

pub mod checkpoint;
pub mod early_stopping;
pub mod loss;
pub mod train;
pub mod walk_forward;

pub use checkpoint::{load_checkpoint, save_checkpoint, CheckpointError};
pub use early_stopping::EarlyStopping;
pub use loss::nll_loss;
pub use train::{evaluate, train_epoch, TrainerConfig, TrainingExample};
pub use walk_forward::{WalkForwardSplit, WalkForwardValidator, WindowMode};
