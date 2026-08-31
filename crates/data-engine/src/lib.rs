//! data-engine
//!
//! Data provider traits (market/fundamental/macro/news) and the synthetic
//! market generator. CSV/Parquet/synthetic first; live providers are
//! adapters (project spec §7).
//!
//! Status: synthetic market generator implemented (`synthetic` module).
//! Provider traits (`MarketDataProvider`, etc.) and CSV/Parquet adapters are
//! not yet implemented — see /PROJECT_STATUS.md.

pub mod synthetic;
