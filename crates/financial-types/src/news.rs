//! `NewsEvent`: a single news/event record about an entity.
//!
//! News has no meaningful observation/publication gap in the same sense as
//! fundamentals or macro data — the event and its publication are
//! effectively the same moment from a modeling standpoint. It still
//! implements [`PointInTime`] (with `observation_time == publication_time`)
//! so it can be used interchangeably with the other point-in-time types
//! wherever a caller works generically over `PointInTime` data (e.g. a
//! unified point-in-time query across relation types in `financial-graph`).

use crate::identifiers::{EntityId, EventType, Source};
use crate::point_in_time::PointInTime;
use crate::timestamp::Timestamp;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewsEvent {
    pub publication_time: Timestamp,
    pub source: Source,
    pub entity: EntityId,
    pub event_type: EventType,
    /// Sentiment score in `[-1.0, 1.0]`; see [`NewsEvent::validate`].
    pub sentiment: f64,
    /// A pointer (e.g. a document ID or URL) to the underlying text, kept
    /// out-of-band rather than storing full article text inline.
    pub text_reference: String,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum NewsEventError {
    #[error("sentiment ({sentiment}) is outside [-1.0, 1.0]")]
    SentimentOutOfRange { sentiment: f64 },
}

impl NewsEvent {
    pub fn validate(&self) -> Result<(), NewsEventError> {
        if !(-1.0..=1.0).contains(&self.sentiment) {
            return Err(NewsEventError::SentimentOutOfRange {
                sentiment: self.sentiment,
            });
        }
        Ok(())
    }
}

impl PointInTime for NewsEvent {
    fn observation_time(&self) -> Timestamp {
        self.publication_time
    }

    fn availability_time(&self) -> Timestamp {
        self.publication_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn ts(y: i32, m: u32, d: u32) -> Timestamp {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn valid_event() -> NewsEvent {
        NewsEvent {
            publication_time: ts(2020, 4, 1),
            source: Source::from("wire"),
            entity: EntityId::from("ACME"),
            event_type: EventType::from("earnings"),
            sentiment: 0.5,
            text_reference: "doc://1".to_string(),
        }
    }

    #[test]
    fn valid_event_passes() {
        assert!(valid_event().validate().is_ok());
    }

    #[test]
    fn rejects_sentiment_out_of_range() {
        let mut event = valid_event();
        event.sentiment = 1.5;
        assert!(matches!(
            event.validate(),
            Err(NewsEventError::SentimentOutOfRange { .. })
        ));
    }
}
