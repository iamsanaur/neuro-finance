//! `MacroObservation`: one macroeconomic series value.
//!
//! Same observation/publication split as [`crate::fundamental`], and for the
//! same reason: government macro statistics are routinely *revised* after
//! initial release. The value published for a given period on the initial
//! release date is not necessarily the value seen in a later revision — this
//! type only ever represents one specific vintage of the number, tagged with
//! the timestamp at which *that vintage* became available. A point-in-time
//! query must never silently swap in a later revision.

use crate::identifiers::MacroSeriesId;
use crate::point_in_time::PointInTime;
use crate::timestamp::Timestamp;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroObservation {
    pub series: MacroSeriesId,
    /// The period this value describes (e.g. the month a CPI print covers).
    pub observation_time: Timestamp,
    /// When this specific vintage of the value was released.
    pub publication_time: Timestamp,
    pub value: f64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MacroObservationError {
    #[error(
        "publication_time ({publication_time}) precedes observation_time ({observation_time})"
    )]
    PublicationBeforeObservation {
        observation_time: Timestamp,
        publication_time: Timestamp,
    },
    #[error("non-finite value")]
    NonFinite,
}

impl MacroObservation {
    pub fn validate(&self) -> Result<(), MacroObservationError> {
        if !self.value.is_finite() {
            return Err(MacroObservationError::NonFinite);
        }
        if self.publication_time < self.observation_time {
            return Err(MacroObservationError::PublicationBeforeObservation {
                observation_time: self.observation_time,
                publication_time: self.publication_time,
            });
        }
        Ok(())
    }
}

impl PointInTime for MacroObservation {
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
        let obs = MacroObservation {
            series: MacroSeriesId::from("CPI"),
            observation_time: ts(2020, 1, 31),
            publication_time: ts(2020, 2, 12),
            value: 2.3,
        };
        assert!(obs.validate().is_ok());
    }

    #[test]
    fn rejects_publication_before_observation() {
        let obs = MacroObservation {
            series: MacroSeriesId::from("CPI"),
            observation_time: ts(2020, 2, 12),
            publication_time: ts(2020, 1, 31),
            value: 2.3,
        };
        assert!(matches!(
            obs.validate(),
            Err(MacroObservationError::PublicationBeforeObservation { .. })
        ));
    }
}
