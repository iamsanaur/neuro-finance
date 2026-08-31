//! Topology persistence across time (project spec §17):
//!
//! ```text
//! A_t = lambda * A_{t-1} + (1 - lambda) * A_new
//! ```
//!
//! Implemented as an exponential moving average over edge *weights*, keyed
//! by node-pair identity rather than position (`NodeId` numbering isn't
//! guaranteed stable across separately-constructed graphs; `EntityId` is).
//! An edge missing from one side of the blend contributes `0.0` to that
//! side, matching what the dense-matrix version of this formula would do —
//! this is also why persisted edges below `WEIGHT_EPSILON` are dropped
//! afterward, so the persisted graph doesn't accumulate an ever-growing set
//! of vanishingly small "ghost" edges from every topology that has ever
//! existed.

use financial_graph::{Edge, FinancialGraph, RelationType};
use financial_types::{EntityId, Timestamp};
use std::collections::HashMap;

/// Below this absolute weight, a persisted edge is dropped rather than kept
/// as noise.
const WEIGHT_EPSILON: f64 = 1e-4;

type PairKey = (EntityId, EntityId);

fn canonical_key(a: &EntityId, b: &EntityId) -> PairKey {
    if a <= b {
        (a.clone(), b.clone())
    } else {
        (b.clone(), a.clone())
    }
}

/// Summary of what changed between two consecutive persisted topologies —
/// project spec §17's "track edge persistence, edge creation, edge
/// deletion".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TopologyDiff {
    pub created: Vec<PairKey>,
    pub deleted: Vec<PairKey>,
    pub persisted: Vec<PairKey>,
}

pub struct TopologyPersistence {
    lambda: f64,
    previous_weights: HashMap<PairKey, f64>,
}

impl TopologyPersistence {
    /// `lambda` is the persistence weight (project spec §17); must be in
    /// `[0.0, 1.0]`. `lambda = 0.0` means no persistence at all (each update
    /// fully replaces the topology with `A_new`); `lambda = 1.0` would mean
    /// the topology never updates (rejected — that's a degenerate,
    /// certainly-unintended configuration, not a useful edge case to
    /// support silently).
    pub fn new(lambda: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&lambda),
            "lambda must be in [0.0, 1.0), got {lambda}"
        );
        Self {
            lambda,
            previous_weights: HashMap::new(),
        }
    }

    /// Blends `new_topology`'s `RelationType::Learned` edges into the
    /// running persisted topology and returns the result. `new_topology`'s
    /// node set defines the persisted graph's node set going forward (V0.1
    /// assumes a static universe — see `financial-graph`'s crate doc on
    /// what's deferred to V0.2).
    pub fn update(
        &mut self,
        new_topology: &FinancialGraph,
        timestamp: Timestamp,
    ) -> FinancialGraph {
        let mut new_weights: HashMap<PairKey, f64> = HashMap::new();
        for edge in new_topology.edges_of_relation(RelationType::Learned) {
            let a = new_topology
                .entity(edge.source)
                .expect("edge endpoint must be a valid node");
            let b = new_topology
                .entity(edge.target)
                .expect("edge endpoint must be a valid node");
            new_weights.insert(canonical_key(a, b), edge.weight);
        }

        let mut blended: HashMap<PairKey, f64> = HashMap::new();
        let all_keys: std::collections::HashSet<&PairKey> = self
            .previous_weights
            .keys()
            .chain(new_weights.keys())
            .collect();
        for key in all_keys {
            let prev = *self.previous_weights.get(key).unwrap_or(&0.0);
            let new = *new_weights.get(key).unwrap_or(&0.0);
            let w = self.lambda * prev + (1.0 - self.lambda) * new;
            if w.abs() > WEIGHT_EPSILON {
                blended.insert(key.clone(), w);
            }
        }

        let mut graph = FinancialGraph::new(new_topology.nodes().to_vec());
        for ((a, b), weight) in &blended {
            if let (Some(source), Some(target)) = (graph.node_id(a), graph.node_id(b)) {
                graph
                    .add_edge(Edge {
                        source,
                        target,
                        relation: RelationType::Learned,
                        weight: *weight,
                        timestamp,
                    })
                    .expect("both endpoints come from new_topology's own node set");
            }
        }

        self.previous_weights = blended;
        graph
    }

    /// Diff between the persisted state *before* the most recent
    /// [`TopologyPersistence::update`] call and the persisted state after
    /// it. Call this by keeping the pre-update key set yourself (see the
    /// tests for the intended usage pattern) — kept as a free function
    /// rather than internal state so a diff can be computed against any two
    /// snapshots, not only consecutive ones.
    pub fn diff(before: &HashMap<PairKey, f64>, after: &HashMap<PairKey, f64>) -> TopologyDiff {
        let before_keys: std::collections::HashSet<&PairKey> = before.keys().collect();
        let after_keys: std::collections::HashSet<&PairKey> = after.keys().collect();
        TopologyDiff {
            created: after_keys
                .difference(&before_keys)
                .map(|k| (*k).clone())
                .collect(),
            deleted: before_keys
                .difference(&after_keys)
                .map(|k| (*k).clone())
                .collect(),
            persisted: before_keys
                .intersection(&after_keys)
                .map(|k| (*k).clone())
                .collect(),
        }
    }

    /// A snapshot of the current persisted weights, for computing a
    /// [`TopologyPersistence::diff`] against a later state.
    pub fn snapshot(&self) -> HashMap<PairKey, f64> {
        self.previous_weights.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn ts() -> Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    fn graph_with_edge(a: &str, b: &str, weight: f64) -> FinancialGraph {
        let mut g = FinancialGraph::new(vec![EntityId::from(a), EntityId::from(b)]);
        let na = g.node_id(&EntityId::from(a)).unwrap();
        let nb = g.node_id(&EntityId::from(b)).unwrap();
        g.add_edge(Edge {
            source: na,
            target: nb,
            relation: RelationType::Learned,
            weight,
            timestamp: ts(),
        })
        .unwrap();
        g
    }

    #[test]
    fn first_update_with_no_prior_state_uses_new_weight_scaled_by_one_minus_lambda() {
        let mut persistence = TopologyPersistence::new(0.9);
        let new_topology = graph_with_edge("A", "B", 1.0);
        let result = persistence.update(&new_topology, ts());

        let a = result.node_id(&EntityId::from("A")).unwrap();
        let edge = result.neighbors(a, None).next().unwrap().1;
        assert!((edge.weight - 0.1).abs() < 1e-9); // 0.9*0 + 0.1*1.0
    }

    #[test]
    fn steady_state_edge_converges_toward_new_weight() {
        let mut persistence = TopologyPersistence::new(0.5);
        let new_topology = graph_with_edge("A", "B", 1.0);
        let mut last_weight = 0.0;
        for _ in 0..20 {
            let result = persistence.update(&new_topology, ts());
            let a = result.node_id(&EntityId::from("A")).unwrap();
            last_weight = result.neighbors(a, None).next().unwrap().1.weight;
        }
        assert!(
            (last_weight - 1.0).abs() < 1e-4,
            "expected convergence to 1.0, got {last_weight}"
        );
    }

    #[test]
    fn edge_that_disappears_decays_and_is_eventually_dropped() {
        let mut persistence = TopologyPersistence::new(0.5);
        // First update establishes an edge with weight 1.0.
        let with_edge = graph_with_edge("A", "B", 1.0);
        persistence.update(&with_edge, ts());

        // Subsequent updates have no edge at all (an empty two-node graph).
        let without_edge = FinancialGraph::new(vec![EntityId::from("A"), EntityId::from("B")]);
        let mut still_present = true;
        for _ in 0..30 {
            let result = persistence.update(&without_edge, ts());
            still_present = result.num_edges() > 0;
            if !still_present {
                break;
            }
        }
        assert!(
            !still_present,
            "edge should eventually decay below WEIGHT_EPSILON and be dropped"
        );
    }

    #[test]
    fn diff_reports_created_deleted_and_persisted_edges() {
        let before: HashMap<PairKey, f64> = HashMap::from([(
            canonical_key(&EntityId::from("A"), &EntityId::from("B")),
            0.5,
        )]);
        let after: HashMap<PairKey, f64> = HashMap::from([
            (
                canonical_key(&EntityId::from("A"), &EntityId::from("B")),
                0.6,
            ), // persisted
            (
                canonical_key(&EntityId::from("C"), &EntityId::from("D")),
                0.3,
            ), // created
        ]);
        let diff = TopologyPersistence::diff(&before, &after);
        assert_eq!(
            diff.persisted,
            vec![canonical_key(&EntityId::from("A"), &EntityId::from("B"))]
        );
        assert_eq!(
            diff.created,
            vec![canonical_key(&EntityId::from("C"), &EntityId::from("D"))]
        );
        assert!(diff.deleted.is_empty());
    }

    #[test]
    #[should_panic(expected = "lambda must be in [0.0, 1.0)")]
    fn rejects_lambda_of_one() {
        TopologyPersistence::new(1.0);
    }
}
