//! Model checkpointing (project spec §31): save/load
//! `NeuroTopologicalFinancialModel<Backend>` state to/from a file.

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use neuro_model::NeuroTopologicalFinancialModel;
use tensor_engine::burn;
use tensor_engine::{device, Backend};

type CheckpointRecorder = NamedMpkFileRecorder<FullPrecisionSettings>;

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("failed to save checkpoint to {path}: {source}")]
    Save {
        path: String,
        #[source]
        source: burn::record::RecorderError,
    },
    #[error("failed to load checkpoint from {path}: {source}")]
    Load {
        path: String,
        #[source]
        source: burn::record::RecorderError,
    },
}

/// Saves `model`'s parameters to `path` (no file extension — Burn's
/// recorder appends its own). Overwrites an existing checkpoint at that
/// path without warning; callers who care about not clobbering a prior
/// checkpoint should check the path themselves first.
pub fn save_checkpoint(
    model: NeuroTopologicalFinancialModel<Backend>,
    path: &std::path::Path,
) -> Result<(), CheckpointError> {
    model
        .save_file(path, &CheckpointRecorder::new())
        .map_err(|source| CheckpointError::Save {
            path: path.display().to_string(),
            source,
        })
}

/// Loads a checkpoint saved by [`save_checkpoint`] into a freshly
/// constructed model of the same architecture. `feature_dim`/`embed_dim`/
/// `topology_proj_dim` must match what the checkpoint was saved with — this
/// is not verified (Burn's `CompactRecorder` will simply fail to load if the
/// tensor shapes it finds don't match the module structure it's asked to
/// populate).
pub fn load_checkpoint(
    path: &std::path::Path,
    feature_dim: usize,
    embed_dim: usize,
    topology_proj_dim: usize,
) -> Result<NeuroTopologicalFinancialModel<Backend>, CheckpointError> {
    let fresh: NeuroTopologicalFinancialModel<Backend> =
        NeuroTopologicalFinancialModel::new(feature_dim, embed_dim, topology_proj_dim, &device());
    let record = CheckpointRecorder::new()
        .load(path.to_path_buf(), &device())
        .map_err(|source| CheckpointError::Load {
            path: path.display().to_string(),
            source,
        })?;
    Ok(fresh.load_record(record))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use financial_types::EntityId;

    fn ts() -> financial_types::Timestamp {
        use chrono::{TimeZone, Utc};
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn round_trip_preserves_forward_output() {
        tensor_engine::seed(0);
        let model: NeuroTopologicalFinancialModel<Backend> =
            NeuroTopologicalFinancialModel::new(4, 8, 4, &device());

        let n = 10;
        let entities: Vec<EntityId> = (0..n).map(|i| EntityId::from(format!("E{i}"))).collect();
        let features: Tensor<Backend, 2> = Tensor::random(
            [n, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );

        let probs_before = model.forward(features.clone(), &entities, 3, ts());

        let dir = std::env::temp_dir().join(format!(
            "neurofinance-checkpoint-test-{}",
            std::process::id()
        ));
        save_checkpoint(model, &dir).expect("save should succeed");

        let loaded = load_checkpoint(&dir, 4, 8, 4).expect("load should succeed");
        let probs_after = loaded.forward(features, &entities, 3, ts());

        let diff: f32 = (probs_before - probs_after).abs().sum().into_scalar();
        assert!(
            diff < 1e-6,
            "checkpoint round-trip should preserve forward output exactly, diff={diff}"
        );

        // Clean up: CompactRecorder appends its own extension.
        for ext in [".mpk", ".mpk.gz", ".bin", ""] {
            let _ = std::fs::remove_file(format!("{}{ext}", dir.display()));
        }
    }

    #[test]
    fn load_from_missing_path_returns_an_error() {
        let missing =
            std::env::temp_dir().join("neurofinance-checkpoint-definitely-does-not-exist");
        let result = load_checkpoint(&missing, 4, 8, 4);
        assert!(result.is_err());
    }
}
