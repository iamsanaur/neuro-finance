//! Newtype string identifiers.
//!
//! A tradable `Symbol`, a macro `MacroSeriesId` (e.g. "US10Y", "CPI"), a
//! fundamental `MetricId` (e.g. "revenue_growth"), and free-text `Source` /
//! `EventType` labels are all, structurally, strings — but they are never
//! interchangeable, and passing one where another is expected is a bug that
//! should fail to compile, not surface as a runtime data error. Hence
//! distinct newtypes instead of passing `String` everywhere (project spec
//! §4: prefer strong typing).

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! string_id {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub String);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

string_id!(
    Symbol,
    "A tradable instrument identifier (e.g. an equity ticker)."
);
string_id!(
    EntityId,
    "A financial entity identifier — a company, sector, or other node the graph can reference. \
     Distinct from `Symbol` because not every entity is directly tradable (e.g. a sector rollup)."
);
string_id!(
    MacroSeriesId,
    "A macroeconomic series identifier (e.g. \"US10Y\", \"CPI\")."
);
string_id!(
    MetricId,
    "A fundamental metric name (e.g. \"revenue_growth\")."
);
string_id!(
    Source,
    "A free-text provenance label for a news event (e.g. a wire service name)."
);
string_id!(
    EventType,
    "A free-text news/event category (e.g. \"earnings\", \"guidance\")."
);
