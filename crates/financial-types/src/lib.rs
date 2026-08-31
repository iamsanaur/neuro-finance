//! financial-types
//!
//! Strongly-typed, timezone-aware financial data structures, and the
//! point-in-time access contract (`PointInTime` / `PointInTimeDataset`) that
//! every other crate in this workspace builds data access on top of.
//!
//! See `docs/data-model.md` (once written) for the full rationale; the short
//! version lives in each module's doc comment.

pub mod fundamental;
pub mod identifiers;
pub mod macro_obs;
pub mod market;
pub mod news;
pub mod point_in_time;
pub mod timestamp;

pub use fundamental::{FundamentalObservation, FundamentalObservationError};
pub use identifiers::{EntityId, EventType, MacroSeriesId, MetricId, Source, Symbol};
pub use macro_obs::{MacroObservation, MacroObservationError};
pub use market::{MarketBar, MarketBarError};
pub use news::{NewsEvent, NewsEventError};
pub use point_in_time::{PointInTime, PointInTimeDataset};
pub use timestamp::Timestamp;
