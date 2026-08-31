//! The point-in-time access contract (project spec §9).
//!
//! > The model may only access information whose availability timestamp is
//! > `<= prediction timestamp`.
//!
//! [`PointInTime`] separates *when a fact describes* (`observation_time`)
//! from *when a fact became knowable* (`availability_time`). Every record
//! type in this crate implements it. [`PointInTimeDataset`] is the one
//! sanctioned way to query a collection of such records "as of" a given
//! time — it does not expose its underlying storage at all, so there is no
//! accidental path to a record whose `availability_time` is in the future
//! relative to the query. Reaching future data through this type requires
//! deliberately misusing the API (e.g. constructing an `as_of` timestamp
//! that is itself wrong), not an accidental oversight in a downstream crate.

use crate::timestamp::Timestamp;

/// Implemented by every record type that can appear in a
/// [`PointInTimeDataset`].
pub trait PointInTime {
    /// The period/moment this record's value describes.
    fn observation_time(&self) -> Timestamp;
    /// The moment this record became knowable to a model. Must be
    /// `>= observation_time` for a well-formed record (each concrete type's
    /// `validate` enforces this where it applies).
    fn availability_time(&self) -> Timestamp;
}

/// A collection of point-in-time records, queryable only through
/// [`PointInTimeDataset::as_of`].
///
/// Records are kept sorted by `availability_time` internally so `as_of` is a
/// binary search, not a linear scan — this matters once a dataset holds
/// years of daily bars across a hundred-asset universe.
#[derive(Debug, Clone)]
pub struct PointInTimeDataset<T> {
    records: Vec<T>,
}

impl<T: PointInTime> PointInTimeDataset<T> {
    pub fn new(mut records: Vec<T>) -> Self {
        records.sort_by_key(|r| r.availability_time());
        Self { records }
    }

    /// All records with `availability_time <= as_of`, in ascending
    /// `availability_time` order. This is the *only* read path this type
    /// exposes — there is no way to reach a record whose availability time
    /// exceeds `as_of` through this API.
    pub fn as_of(&self, as_of: Timestamp) -> impl Iterator<Item = &T> {
        let idx = self
            .records
            .partition_point(|r| r.availability_time() <= as_of);
        self.records[..idx].iter()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl<T: PointInTime> Default for PointInTimeDataset<T> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[derive(Debug, Clone, PartialEq)]
    struct Fact {
        id: u32,
        observation_time: Timestamp,
        availability_time: Timestamp,
    }

    impl PointInTime for Fact {
        fn observation_time(&self) -> Timestamp {
            self.observation_time
        }
        fn availability_time(&self) -> Timestamp {
            self.availability_time
        }
    }

    fn ts(day: u32) -> Timestamp {
        Utc.with_ymd_and_hms(2020, 1, day, 0, 0, 0).unwrap()
    }

    fn fact(id: u32, observed_day: u32, available_day: u32) -> Fact {
        Fact {
            id,
            observation_time: ts(observed_day),
            availability_time: ts(available_day),
        }
    }

    /// The core leakage test (project spec §30): as_of() must never return
    /// a record whose availability_time exceeds the query time, even when a
    /// record's *observation_time* is well in the past — publication lag is
    /// exactly the case this whole abstraction exists to guard.
    #[test]
    fn as_of_excludes_records_available_after_query_time() {
        let dataset = PointInTimeDataset::new(vec![
            fact(1, 1, 1),  // observed day 1, available day 1
            fact(2, 1, 10), // observed day 1, available day 10 (lagged publication)
            fact(3, 5, 5),  // observed day 5, available day 5
        ]);

        let visible = dataset.as_of(ts(5)).map(|f| f.id).collect::<Vec<_>>();
        assert_eq!(visible, vec![1, 3]);

        let visible_later = dataset.as_of(ts(10)).map(|f| f.id).collect::<Vec<_>>();
        assert_eq!(visible_later, vec![1, 3, 2]);
    }

    #[test]
    fn as_of_before_any_availability_returns_empty() {
        let dataset = PointInTimeDataset::new(vec![fact(1, 5, 5)]);
        assert_eq!(dataset.as_of(ts(1)).count(), 0);
    }

    #[test]
    fn as_of_is_inclusive_of_exact_availability_time() {
        let dataset = PointInTimeDataset::new(vec![fact(1, 5, 5)]);
        assert_eq!(dataset.as_of(ts(5)).count(), 1);
    }

    #[test]
    fn results_are_ordered_by_availability_time() {
        let dataset = PointInTimeDataset::new(vec![fact(1, 1, 9), fact(2, 1, 3), fact(3, 1, 6)]);
        let ids = dataset.as_of(ts(9)).map(|f| f.id).collect::<Vec<_>>();
        assert_eq!(ids, vec![2, 3, 1]);
    }

    #[test]
    fn empty_dataset_is_empty() {
        let dataset: PointInTimeDataset<Fact> = PointInTimeDataset::default();
        assert!(dataset.is_empty());
        assert_eq!(dataset.as_of(ts(1)).count(), 0);
    }
}
