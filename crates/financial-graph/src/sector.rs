//! Sector graph builder (project spec §14): company → sector, expressed as
//! asset-asset edges between every pair of assets sharing a sector.
//!
//! This is a static graph — sector membership doesn't change day to day in
//! this system, so there's no `as_of` parameter here the way there is for
//! [`crate::correlation::build_correlation_graph`]. The `timestamp` on each
//! produced edge just records when the graph was built, for bookkeeping
//! parity with the other relation types.

use crate::graph::FinancialGraph;
use crate::types::{Edge, NodeId, RelationType};
use financial_types::{EntityId, Symbol, Timestamp};
use std::collections::HashMap;

/// Builds a graph whose nodes are every `Symbol` in `sector_assignment`
/// (converted to [`EntityId`] — see [`FinancialGraph::new`]'s doc), with a
/// `RelationType::Sector` edge (weight `1.0`, unweighted co-membership)
/// between every pair of assets in the same sector.
///
/// Grouping by sector first (rather than an all-pairs `O(n^2)` scan checking
/// sector equality) keeps this `O(n * average_sector_size)` — for a 100
/// asset / 10 sector universe that's a small constant either way, but the
/// grouped approach is the one that doesn't quietly become expensive at
/// real universe sizes.
pub fn build_sector_graph(
    sector_assignment: &[(Symbol, EntityId)],
    timestamp: Timestamp,
) -> FinancialGraph {
    let nodes: Vec<EntityId> = sector_assignment
        .iter()
        .map(|(symbol, _)| EntityId::from(symbol.0.clone()))
        .collect();
    let mut graph = FinancialGraph::new(nodes);

    let mut by_sector: HashMap<&EntityId, Vec<NodeId>> = HashMap::new();
    for (i, (_, sector)) in sector_assignment.iter().enumerate() {
        by_sector.entry(sector).or_default().push(NodeId(i as u32));
    }

    for members in by_sector.values() {
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                graph
                    .add_edge(Edge {
                        source: members[i],
                        target: members[j],
                        relation: RelationType::Sector,
                        weight: 1.0,
                        timestamp,
                    })
                    .expect("indices come from the graph's own node table");
            }
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn ts() -> Timestamp {
        Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn connects_only_same_sector_pairs() {
        let assignment = vec![
            (Symbol::from("AAA"), EntityId::from("TECH")),
            (Symbol::from("BBB"), EntityId::from("TECH")),
            (Symbol::from("CCC"), EntityId::from("FIN")),
        ];
        let graph = build_sector_graph(&assignment, ts());

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 1); // only AAA-BBB, both TECH

        let aaa = graph.node_id(&EntityId::from("AAA")).unwrap();
        let ccc = graph.node_id(&EntityId::from("CCC")).unwrap();
        assert_eq!(graph.degree(aaa, Some(RelationType::Sector)), 1);
        assert_eq!(graph.degree(ccc, Some(RelationType::Sector)), 0);
    }

    #[test]
    fn edge_count_matches_combinatorics_for_synthetic_universe() {
        // 100 assets, 10 sectors of 10 each: C(10, 2) = 45 edges per sector,
        // 10 sectors -> 450 edges total.
        let mut assignment = Vec::new();
        for sector in 0..10 {
            for asset in 0..10 {
                assignment.push((
                    Symbol::from(format!("AST{sector}{asset}")),
                    EntityId::from(format!("SECTOR{sector}")),
                ));
            }
        }
        let graph = build_sector_graph(&assignment, ts());
        assert_eq!(graph.num_edges(), 450);
    }
}
