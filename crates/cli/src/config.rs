//! Typed configuration loaded from `configs/*.toml`.
//!
//! Every numeric knob the project spec calls out as "must be configurable"
//! (topology top-k, persistence lambda, regularization weights, training
//! hyperparameters, walk-forward windows, backtest cost assumptions) lives
//! here — never hard-coded inside a model or engine crate.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub universe: UniverseConfig,
    pub data: DataConfig,
    pub sequence: SequenceConfig,
    pub topology: TopologyConfig,
    pub training: TrainingConfig,
    pub walk_forward: WalkForwardConfig,
    pub backtest: BacktestConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UniverseConfig {
    pub num_assets: usize,
    pub num_sectors: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataConfig {
    pub raw_dir: String,
    pub interim_dir: String,
    pub processed_dir: String,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SequenceConfig {
    pub length: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TopologyConfig {
    pub top_k: usize,
    pub lambda_persistence: f64,
    pub regularization: TopologyRegularizationConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TopologyRegularizationConfig {
    pub lambda_sparse: f64,
    pub lambda_stability: f64,
    pub lambda_relation: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub early_stopping_patience: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WalkForwardConfig {
    pub train_years: u32,
    pub validation_years: u32,
    pub test_years: u32,
    pub embargo_days: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BacktestConfig {
    pub transaction_cost_bps: f64,
    pub slippage_bps: f64,
    pub max_position_weight: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config file {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.display().to_string(),
            source,
        })?;
        toml::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.display().to_string(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_default_config() {
        // Path is relative to the workspace root (where cargo test runs from
        // for a workspace member unless otherwise configured — verified by
        // CARGO_MANIFEST_DIR below instead, so this test is robust to CWD).
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir)
            .join("../../configs/default.toml")
            .canonicalize()
            .expect("configs/default.toml should exist at the workspace root");

        let config = Config::load(&path).expect("default.toml should parse");

        assert_eq!(config.universe.num_assets, 100);
        assert_eq!(config.universe.num_sectors, 10);
        assert_eq!(config.sequence.length, 30);
        assert_eq!(config.topology.top_k, 8);
        assert!((config.topology.lambda_persistence - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.training.seed, 42);
    }
}
