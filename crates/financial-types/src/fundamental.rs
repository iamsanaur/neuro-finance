//! `FundamentalObservation`: one reported metric value for one entity.
//!
//! This is the canonical example of why observation time and publication
//! time must be tracked separately (project spec §9): a Q1 metric is
//! *observed* as of quarter-end (e.g. 2020-03-31) but not *published* until
//! the filing date (e.g. 2020-05-15). A model predicting on 2020-04-15 must
//! not see this observation at all, even though the value it describes
//! already existed.

use crate::identifiers::{EntityId, MetricId};
use crate::point_in_time::PointInTime;
use crate::timestamp::Timestamp;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FundamentalObservation {
    pub entity: EntityId,
    /// The period the value describes (e.g. fiscal quarter end).
    pub observation_time: Timestamp,
    /// When this value became knowable to a model (e.g. the filing/release
    /// timestamp). Always `>= observation_time` for a well-formed record —
    /// see [`FundamentalObservation::validate`].
    pub publication_time: Timestamp,
    pub metric: MetricId,
    pub value: f64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum FundamentalObservationError {
    #[error(
        "publication_time ({publication_time}) precedes observation_time ({observation_time}) \
         — a value cannot be published before the period it describes exists"
    )]
    PublicationBeforeObservation {
        observation_time: Timestamp,
        publication_time: Timestamp,
    },
    #[error("non-finite value")]
    NonFinite,
}

impl FundamentalObservation {
    pub fn validate(&self) -> Result<(), FundamentalObservationError> {
        if !self.value.is_finite() {
            return Err(FundamentalObservationError::NonFinite);
        }
        if self.publication_time < self.observation_time {
            return Err(FundamentalObservationError::PublicationBeforeObservation {
                observation_time: self.observation_time,
                publication_time: self.publication_time,
            });
        }
        Ok(())
    }
}

impl PointInTime for FundamentalObservation {
    fn observation_time(&self) -> Timestamp {
        self.observation_time
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

    #[test]
    fn valid_observation_passes() {
        let obs = FundamentalObservation {
            entity: EntityId::from("ACME"),
            observation_time: ts(2020, 3, 31),
            publication_time: ts(2020, 5, 15),
            metric: MetricId::from("revenue_growth"),
            value: 0.12,
        };
        assert!(obs.validate().is_ok());
    }

    #[test]
    fn rejects_publication_before_observation() {
        let obs = FundamentalObservation {
            entity: EntityId::from("ACME"),
            observation_time: ts(2020, 5, 15),
            publication_time: ts(2020, 3, 31),
            metric: MetricId::from("revenue_growth"),
            value: 0.12,
        };
        assert!(matches!(
            obs.validate(),
            Err(FundamentalObservationError::PublicationBeforeObservation { .. })
        ));
    }

    #[test]
    fn availability_is_publication_time_not_observation_time() {
        let obs = FundamentalObservation {
            entity: EntityId::from("ACME"),
            observation_time: ts(2020, 3, 31),
            publication_time: ts(2020, 5, 15),
            metric: MetricId::from("revenue_growth"),
            value: 0.12,
        };
        assert_eq!(obs.availability_time(), ts(2020, 5, 15));
        assert_ne!(obs.availability_time(), obs.observation_time());
    }
}
