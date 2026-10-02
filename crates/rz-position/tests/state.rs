use rz_position::{
    BoardMove, Color, HistoryCompleteness, HistoryOrigin, Piece, PieceKind, Position,
    PositionError, Square,
};

#[test]
fn fen_rejects_malformed_and_contradictory_rule_state() {
    for fen in [
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "7k/8/8/8/8/8/8/8 w - - 0 1",
        "7k/8/8/8/8/8/8/KK6 w - - 0 1",
        "8/8/8/8/8/8/k7/K7 w - - 0 1",
        "7k/8/8/8/8/8/8/KP6 w - - 0 1",
        "7k/8/8/8/8/8/8/K6p w - - 0 1",
        "4k3/8/8/8/8/8/8/K3R3 w - - 0 1",
        "7k/8/8/8/8/8/8/K7 w K - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KK - 0 1",
        "7k/8/8/8/8/8/8/K7 x - - 0 1",
        "7k/8/8/8/8/8/8/K7 w - - -1 1",
        "7k/8/8/8/8/8/8/K7 w - - 0 0",
        "7k/8/8/8/8/8/8/K7 w - - 4294967296 1",
        "7k/8/8/8/8/8/8/K7 w - - 0 4294967296",
        "7k/8/8/8/8/8/8/K7 w - - 0",
        "7k/8/8/8/8/8/8/K7 w - - 0 1 extra",
        "7k/8/8/8/8/8/8/K8 w - - 0 1",
        "7k/8/8/8/8/8/8/Kx6 w - - 0 1",
        "7k/8/8/8/8/8/8/K7 w - d6 0 1",
        "k7/8/8/3pP3/8/8/8/4K3 w - d3 0 2",
    ] {
        assert!(
            Position::from_fen(fen).is_err(),
            "unexpectedly accepted {fen}"
        );
    }
    // A moving king may be in check; only a check on the nonmoving king is a
    // contradiction in a position reached after a completed legal move.
    assert!(Position::from_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1")
        .unwrap()
        .in_check());
}

#[test]
fn fen_round_trip_preserves_raw_ep_counters_rights_and_unknown_prefix() {
    let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    let position = Position::from_fen(fen).unwrap();
    assert_eq!(position.to_fen(), fen);
    assert_eq!(
        position.history_completeness(),
        HistoryCompleteness::UnknownPrefix
    );
    assert_eq!(position.history_origin(), HistoryOrigin::Fen);
    let snapshot = position.snapshot();
    assert_eq!(snapshot.known_history_len(), 1);
    assert_eq!(snapshot.known_history_fens(), vec![fen.to_owned()]);

    let no_rights = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 87 42").unwrap();
    assert_eq!(no_rights.to_fen(), "r3k2r/8/8/8/8/8/8/R3K2R w - - 87 42");
    assert!(!no_rights
        .legal_moves()
        .contains(&BoardMove::from_uci("e1g1").unwrap()));
    assert!(!no_rights
        .legal_moves()
        .contains(&BoardMove::from_uci("e1c1").unwrap()));
}

#[test]
fn repetition_normalizes_only_legally_capturable_ep_and_ignores_counters() {
    let pinned = Position::from_fen("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 2").unwrap();
    let pinned_without_ep = Position::from_fen("k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 2").unwrap();
    assert!(!pinned
        .legal_moves()
        .contains(&BoardMove::from_uci("e5d6").unwrap()));
    assert_eq!(
        pinned.repetition_identity(),
        pinned_without_ep.repetition_identity()
    );
    assert_ne!(
        pinned.position_identity(),
        pinned_without_ep.position_identity()
    );

    let legal = Position::from_fen("k7/8/8/3pP3/8/8/8/4K3 w - d6 0 2").unwrap();
    let legal_without_ep = Position::from_fen("k7/8/8/3pP3/8/8/8/4K3 w - - 0 2").unwrap();
    assert!(legal
        .legal_moves()
        .contains(&BoardMove::from_uci("e5d6").unwrap()));
    assert_ne!(
        legal.repetition_identity(),
        legal_without_ep.repetition_identity()
    );

    let counters = Position::from_fen("k7/8/8/3pP3/8/8/8/4K3 w - - 89 400").unwrap();
    assert_eq!(
        counters.repetition_identity(),
        legal_without_ep.repetition_identity()
    );
    assert_ne!(
        counters.position_identity(),
        legal_without_ep.position_identity()
    );

    let horizontal = Position::from_fen("k7/8/8/r4pPK/8/8/8/8 w - f6 0 2").unwrap();
    let horizontal_without_ep = Position::from_fen("k7/8/8/r4pPK/8/8/8/8 w - - 0 2").unwrap();
    assert!(!horizontal
        .legal_moves()
        .contains(&BoardMove::from_uci("g5f6").unwrap()));
    assert_eq!(
        horizontal.repetition_identity(),
        horizontal_without_ep.repetition_identity()
    );
}

#[test]
fn castling_rights_and_side_to_move_are_part_of_repetition_identity() {
    let base = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    let no_white_rights = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w kq - 0 1").unwrap();
    let black_to_move = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1").unwrap();
    assert_ne!(
        base.repetition_identity(),
        no_white_rights.repetition_identity()
    );
    assert_ne!(
        base.repetition_identity(),
        black_to_move.repetition_identity()
    );
}

#[test]
fn special_transitions_have_exact_external_state_and_restore_all_known_history() {
    for (fen, uci, expected) in [
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "e1g1",
            "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1",
        ),
        (
            "k7/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
            "e5d6",
            "k7/8/3P4/8/8/8/8/4K3 b - - 0 2",
        ),
        (
            "1r5k/P7/8/8/8/8/8/7K w - - 0 1",
            "a7b8q",
            "1Q5k/8/8/8/8/8/8/7K b - - 0 1",
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 1",
            "a1a8",
            "R3k2r/8/8/8/8/8/8/4K2R b Kk - 0 1",
        ),
    ] {
        let mut position = Position::from_fen(fen).unwrap();
        let before = position.snapshot();
        let identity = position.position_identity();
        let undo = position.make_uci(uci).unwrap();
        assert_eq!(position.to_fen(), expected, "{uci}");
        assert_eq!(before.to_fen(), fen, "snapshot changed after {uci}");
        position.unmake(undo).unwrap();
        assert!(
            before.same_state(&position.snapshot()),
            "incomplete restoration after {uci}"
        );
        assert_eq!(position.position_identity(), identity);
        assert!(position.revision() > before.revision());
    }
}

#[test]
fn long_make_unmake_restores_repetition_model_history_and_source_evidence() {
    let mut position = Position::startpos();
    let before = position.snapshot();
    let mut undos = Vec::new();
    for _ in 0..5 {
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            undos.push(position.make_uci(uci).unwrap());
        }
    }
    assert_eq!(position.known_repetition_count(), 6);
    assert_eq!(position.snapshot().known_history_len(), 21);
    while let Some(undo) = undos.pop() {
        position.unmake(undo).unwrap();
    }
    assert!(before.same_state(&position.snapshot()));
    assert_eq!(position.known_repetition_count(), 1);
    assert_eq!(position.snapshot().known_history_len(), 1);
    assert_eq!(position.history_origin(), HistoryOrigin::StartPosition);
    assert_eq!(
        position.history_completeness(),
        HistoryCompleteness::Complete
    );
}

#[test]
fn failed_moves_and_atomic_move_traces_preserve_original_state_and_live_view() {
    let mut position = Position::startpos();
    let before = position.snapshot();
    let view = position.ordered_legal_moves();
    assert_eq!(
        position.make_uci("e2e5").unwrap_err(),
        PositionError::IllegalMove
    );
    assert!(position.apply_uci_moves(&["e2e4", "e7e5", "e1e3"]).is_err());
    assert!(before.same_state(&position.snapshot()));
    assert_eq!(position.revision(), before.revision());
    position
        .make_from_view(&view, BoardMove::from_uci("e2e4").unwrap())
        .unwrap();
}

#[test]
fn checked_counter_overflow_is_atomic() {
    for (fen, uci) in [
        ("7k/8/8/8/8/8/8/KR6 w - - 4294967295 1", "b1b2"),
        ("7k/8/8/8/8/8/8/KR6 b - - 0 4294967295", "h8h7"),
    ] {
        let mut position = Position::from_fen(fen).unwrap();
        let before = position.snapshot();
        assert_eq!(
            position.make_uci(uci).unwrap_err(),
            PositionError::CounterOverflow
        );
        assert!(before.same_state(&position.snapshot()));
        assert_eq!(position.revision(), before.revision());
    }
}

#[test]
fn owned_snapshots_and_legal_views_survive_mutation_but_stale_views_cannot_make() {
    let mut position = Position::startpos();
    let initial = position.snapshot();
    let view = position.ordered_legal_moves();
    let original_moves = view.moves().to_vec();
    let e4 = BoardMove::from_uci("e2e4").unwrap();
    let undo = position.make_from_view(&view, e4).unwrap();
    assert_eq!(initial.to_fen(), Position::startpos().to_fen());
    assert_eq!(view.moves(), original_moves.as_slice());
    assert_eq!(view.snapshot().to_fen(), initial.to_fen());
    assert_eq!(
        initial.piece_at("e2".parse().unwrap()),
        Some(Piece {
            color: Color::White,
            kind: PieceKind::Pawn
        })
    );
    position.unmake(undo).unwrap();
    assert!(initial.same_state(&position.snapshot()));
    let before_stale = position.snapshot();
    assert_eq!(
        position.make_from_view(&view, e4).unwrap_err(),
        PositionError::StaleView
    );
    assert!(before_stale.same_state(&position.snapshot()));
    assert_eq!(position.revision(), before_stale.revision());

    let mut independent = Position::startpos();
    assert_eq!(
        independent.make_from_view(&view, e4).unwrap_err(),
        PositionError::StaleView
    );
    let mut cloned = position.clone();
    let current_view = position.ordered_legal_moves();
    assert_eq!(
        cloned.make_from_view(&current_view, e4).unwrap_err(),
        PositionError::StaleView
    );
}

#[test]
fn undo_tokens_reject_foreign_owners_wrong_children_and_atomic_replacement() {
    let mut position = Position::startpos();
    let undo = position.make_uci("e2e4").unwrap();
    let mut other = position.clone();
    let before = other.snapshot();
    assert_eq!(other.unmake(undo).unwrap_err(), PositionError::UndoMismatch);
    assert!(before.same_state(&other.snapshot()));

    let mut position = Position::startpos();
    let undo = position.make_uci("e2e4").unwrap();
    position.make_uci("e7e5").unwrap();
    let before = position.snapshot();
    assert_eq!(
        position.unmake(undo).unwrap_err(),
        PositionError::UndoMismatch
    );
    assert!(before.same_state(&position.snapshot()));

    let mut position = Position::startpos();
    let undo = position.make_uci("e2e4").unwrap();
    position.apply_uci_moves(&["e7e5"]).unwrap();
    let before = position.snapshot();
    assert_eq!(
        position.unmake(undo).unwrap_err(),
        PositionError::UndoMismatch
    );
    assert!(before.same_state(&position.snapshot()));
}

#[test]
fn legal_order_distinguishes_all_promotions_and_returns_no_duplicates() {
    let position = Position::from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1").unwrap();
    let view = position.ordered_legal_moves();
    let promotions: Vec<_> = view
        .moves()
        .iter()
        .filter(|mv| mv.from == "a7".parse::<Square>().unwrap())
        .map(ToString::to_string)
        .collect();
    assert_eq!(promotions, ["a7a8q", "a7a8r", "a7a8b", "a7a8n"]);
    let mut unique = view.moves().to_vec();
    unique.sort_by_key(|mv| mv.to_string());
    unique.dedup();
    assert_eq!(unique.len(), view.moves().len());
}
