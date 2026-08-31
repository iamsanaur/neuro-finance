//! All financial timestamps in this system are timezone-aware `DateTime<Utc>`
//! values — never a naive datetime, never a raw string (project spec §8: "Do
//! not represent financial timestamps as arbitrary strings.").
//!
//! Data ingested in another timezone (e.g. an exchange-local session time)
//! must be converted to UTC at the ingestion boundary, not carried as-is —
//! that conversion point is where DST and exchange-calendar bugs are caught
//! once, rather than re-litigated at every downstream read site.

use chrono::{DateTime, Utc};

pub type Timestamp = DateTime<Utc>;
