//! `neurofinance` CLI entry point.
//!
//! V0.1 status: only config loading is wired up. The full command surface
//! (§42 — data ingest/validate, features build, graph build, train, evaluate,
//! backtest, topology inspect, paper-trade, serve, experiment run) lands in
//! later milestones, one command at a time, alongside the crate it drives.

use clap::Parser;
use cli::config::Config;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "neurofinance",
    version,
    about = "NeuroTopological Financial AI CLI"
)]
struct Args {
    /// Path to a TOML config file (see configs/default.toml).
    #[arg(long, default_value = "configs/default.toml")]
    config: PathBuf,
}

fn main() {
    let args = Args::parse();

    match Config::load(&args.config) {
        Ok(config) => {
            println!(
                "Loaded config from {}: {} assets across {} sectors, sequence_length={}, topology_top_k={}",
                args.config.display(),
                config.universe.num_assets,
                config.universe.num_sectors,
                config.sequence.length,
                config.topology.top_k,
            );
        }
        Err(err) => {
            eprintln!("Failed to load config: {err}");
            std::process::exit(1);
        }
    }
}
