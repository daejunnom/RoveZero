use super::*;
use rz_search::pals::engine::RoleQueryPurpose;
use rz_search::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
};
use rz_search::pals::store::{PalsStores, StoreLimits};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

#[test]
fn selected_encoder_preserves_rows_relations_and_ignores_raw_query_records() {
    let position = Position::startpos();
    let legal = position.legal_moves();
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(position.snapshot()).unwrap();
    let state = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    let mut proposal = RoleRecord {
        revision: 1,
        parent_revision: None,
        supersedes_revision: None,
        origin_state: state,
        kind: RecordKind::Proposal,
        line: vec![
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ],
        value: None,
        completed_depth: 0,
        score_scope: None,
        cpu_observation: None,
        perspective: Color::White,
        critical: true,
    };
    index.publish(&stores, &proposal, None).unwrap();
    proposal.revision = 2;
    proposal.parent_revision = Some(1);
    proposal.kind = RecordKind::Counterexample;
    proposal.line = vec![
        BoardMove::from_uci("e2e4").unwrap(),
        BoardMove::from_uci("c7c5").unwrap(),
    ];
    index.publish(&stores, &proposal, None).unwrap();
    let view = index
        .select(
            &stores,
            state,
            ReviewQuerySequences {
                purpose: RoleQueryPurpose::ProposePolicy,
                prefix: &[],
                proposal: &[],
                counterexample: None,
                legal: &legal,
                divergences: &[],
            },
            ReviewSelectionPolicy::default(),
        )
        .unwrap();
    proposal.revision = 999;
    let raw = [proposal];
    let canceled = AtomicBool::new(false);
    let query = RoleQuery {
        position: &position,
        legal: &legal,
        prefix: &[],
        proposal: &[],
        counterexample: None,
        records: &raw,
        revision: 2,
        deadline: Instant::now() + Duration::from_secs(1),
        cancel: &canceled,
    };
    let config = PalsModelConfig::full_line_interaction_v2();
    let input = prepare_selected_role_input_for_config(
        &query,
        NativeQueryKind::Propose,
        [1; 32],
        &config,
        &view,
    )
    .unwrap();
    assert_eq!(input.records.len(), 2);
    assert_eq!(
        input
            .records
            .iter()
            .map(|row| row.revision)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let lines = input.full_line.unwrap();
    assert_eq!(lines.records[1].parent, Some(0));
    assert_eq!(
        lines.records[0].moves[0],
        candidate(BoardMove::from_uci("e2e4").unwrap())
    );
    assert_eq!(
        lines.records[1].moves[1],
        candidate(BoardMove::from_uci("c7c5").unwrap())
    );
    assert_eq!(stores.review_pin_count().unwrap(), 1);
    let independent = position.clone();
    let independent_query = RoleQuery {
        position: &independent,
        legal: &legal,
        prefix: &[],
        proposal: &[],
        counterexample: None,
        records: &raw,
        revision: 2,
        deadline: query.deadline,
        cancel: &canceled,
    };
    assert!(
        prepare_selected_role_input_for_config(
            &independent_query,
            NativeQueryKind::Propose,
            [1; 32],
            &config,
            &view
        )
        .is_ok()
    );
    let mut wrong = position.clone();
    wrong.make_uci("e2e4").unwrap();
    let bad = RoleQuery {
        position: &wrong,
        ..query
    };
    assert!(
        prepare_selected_role_input_for_config(
            &bad,
            NativeQueryKind::Propose,
            [1; 32],
            &config,
            &view
        )
        .is_err()
    );
    drop(view);
    assert_eq!(stores.review_pin_count().unwrap(), 0);
}

#[test]
fn selected_encoder_rejects_same_board_and_counters_with_different_history() {
    let mut position = Position::startpos();
    for movement in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        position.make_uci(movement).unwrap();
    }
    let legal = position.legal_moves();
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(position.snapshot()).unwrap();
    let state = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    let view = index
        .select(
            &stores,
            state,
            ReviewQuerySequences {
                purpose: RoleQueryPurpose::ProposePolicy,
                prefix: &[],
                proposal: &[],
                counterexample: None,
                legal: &legal,
                divergences: &[],
            },
            ReviewSelectionPolicy::default(),
        )
        .unwrap();
    let different = Position::from_fen(&position.to_fen()).unwrap();
    assert_eq!(different.to_fen(), position.to_fen());
    assert!(!different.snapshot().same_state(&position.snapshot()));
    let canceled = AtomicBool::new(false);
    let query = RoleQuery {
        position: &different,
        legal: &legal,
        prefix: &[],
        proposal: &[],
        counterexample: None,
        records: &[],
        revision: 0,
        deadline: Instant::now() + Duration::from_secs(1),
        cancel: &canceled,
    };
    assert!(
        prepare_selected_role_input_for_config(
            &query,
            NativeQueryKind::Propose,
            [1; 32],
            &PalsModelConfig::full_line_interaction_v2(),
            &view
        )
        .is_err()
    );
}
