use rz_position::{
    Availability, BoardMove, ClaimEvidence, ClaimReason, Color, HistoryCompleteness, HistoryOrigin,
    PlayStatus, Position, PositionClassification, TerminalReason,
};

fn current(classification: &PositionClassification, reason: ClaimReason) -> Availability {
    classification
        .claim_availability
        .iter()
        .find(|claim| claim.reason == reason && claim.evidence == ClaimEvidence::CurrentPosition)
        .expect("each current claim condition has explicit evidence")
        .availability
}

fn intended(classification: &PositionClassification, reason: ClaimReason, uci: &str) -> bool {
    let mv = BoardMove::from_uci(uci).unwrap();
    classification.claim_availability.iter().any(|claim| {
        claim.reason == reason
            && claim.evidence == ClaimEvidence::IntendedMove(mv)
            && claim.availability == Availability::Available
    })
}

fn moves(position: &mut Position, line: &[&str]) {
    for uci in line {
        position.make_uci(uci).unwrap();
    }
}

const KNIGHT_CYCLE: [&str; 4] = ["g1f3", "g8f6", "f3g1", "f6g8"];

#[test]
fn search_play_status_agrees_with_full_classification_and_rejects_stale_view() {
    // Search needs automatic outcomes, while full classification also previews
    // intended draw claims. The cheap path must preserve their shared meaning.
    for fen in [
        "7k/6Q1/5K2/8/8/8/8/8 b - - 150 1", // mate before 75-move rule
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",   // stalemate
        "7k/8/8/8/8/8/8/K7 w - - 0 1",      // proven dead material
        "7k/8/8/8/8/8/P7/KR6 w - - 100 1",  // claim, still ongoing
        "7k/8/8/8/8/8/P7/KR6 w - - 150 1",  // automatic draw
    ] {
        let position = Position::from_fen(fen).unwrap();
        let view = position.ordered_legal_moves();
        assert_eq!(
            position.play_status_from_view(&view).unwrap(),
            position.classify_position().unwrap().play_status,
            "{fen}"
        );
    }
    let mut position = Position::startpos();
    for cycles in 0..=4 {
        let view = position.ordered_legal_moves();
        assert_eq!(
            position.play_status_from_view(&view).unwrap(),
            position.classify_position().unwrap().play_status
        );
        if cycles < 4 {
            moves(&mut position, &KNIGHT_CYCLE);
        }
    }
    let stale = position.ordered_legal_moves();
    position.make_uci("e2e4").unwrap();
    assert!(position.play_status_from_view(&stale).is_err());
}

#[test]
fn threefold_claim_before_and_after_the_intended_move_is_not_automatic() {
    let mut position = Position::startpos();
    moves(&mut position, &KNIGHT_CYCLE);
    moves(&mut position, &KNIGHT_CYCLE[..3]);
    let before = position.classify_position().unwrap();
    assert_eq!(before.play_status, PlayStatus::Ongoing);
    assert_eq!(
        current(&before, ClaimReason::ThreefoldRepetition),
        Availability::Unavailable
    );
    assert!(intended(&before, ClaimReason::ThreefoldRepetition, "f6g8"));

    position.make_uci("f6g8").unwrap();
    let after = position.classify_position().unwrap();
    assert_eq!(after.play_status, PlayStatus::Ongoing);
    assert_eq!(
        current(&after, ClaimReason::ThreefoldRepetition),
        Availability::Available
    );
    assert_eq!(after.history_evidence.known_repetition_count, 3);
    assert_eq!(
        after.history_evidence.fivefold_repetition,
        Availability::Unavailable
    );
}

#[test]
fn fifth_known_occurrence_is_an_automatic_draw() {
    let mut position = Position::startpos();
    for _ in 0..4 {
        moves(&mut position, &KNIGHT_CYCLE);
    }
    let classification = position.classify_position().unwrap();
    assert_eq!(
        classification.play_status,
        PlayStatus::Terminal {
            reason: TerminalReason::FivefoldRepetition,
            winner: None,
        }
    );
    assert_eq!(classification.history_evidence.known_repetition_count, 5);
    assert_eq!(
        classification.history_evidence.fivefold_repetition,
        Availability::Available
    );
    assert!(
        classification
            .claim_availability
            .iter()
            .all(|claim| { claim.evidence == ClaimEvidence::CurrentPosition })
    );
}

#[test]
fn unknown_prefix_preserves_uncertainty_but_known_occurrences_prove_repetition() {
    let initial = Position::startpos().to_fen();
    let mut position = Position::from_fen(&initial).unwrap();
    let initial_classification = position.classify_position().unwrap();
    assert_eq!(initial_classification.play_status, PlayStatus::Ongoing);
    assert_eq!(
        current(&initial_classification, ClaimReason::ThreefoldRepetition),
        Availability::Unknown
    );
    assert_eq!(
        initial_classification.history_evidence.completeness,
        HistoryCompleteness::UnknownPrefix
    );
    assert_eq!(
        initial_classification.history_evidence.origin,
        HistoryOrigin::Fen
    );
    assert_eq!(
        initial_classification.history_evidence.fivefold_repetition,
        Availability::Unknown
    );
    assert!(
        initial_classification
            .claim_availability
            .iter()
            .any(|claim| {
                claim.reason == ClaimReason::ThreefoldRepetition
                    && claim.evidence
                        == ClaimEvidence::IntendedMove(BoardMove::from_uci("g1f3").unwrap())
                    && claim.availability == Availability::Unknown
            })
    );
    assert!(
        !initial_classification
            .claim_availability
            .iter()
            .any(|claim| {
                claim.reason == ClaimReason::ThreefoldRepetition
                    && claim.evidence
                        == ClaimEvidence::IntendedMove(BoardMove::from_uci("e2e4").unwrap())
            })
    );

    for _ in 0..2 {
        moves(&mut position, &KNIGHT_CYCLE);
    }
    let claim = position.classify_position().unwrap();
    assert_eq!(claim.play_status, PlayStatus::Ongoing);
    assert_eq!(
        current(&claim, ClaimReason::ThreefoldRepetition),
        Availability::Available
    );
    assert_eq!(
        claim.history_evidence.completeness,
        HistoryCompleteness::UnknownPrefix
    );

    for _ in 0..2 {
        moves(&mut position, &KNIGHT_CYCLE);
    }
    assert_eq!(
        position.classify_position().unwrap().play_status,
        PlayStatus::Terminal {
            reason: TerminalReason::FivefoldRepetition,
            winner: None,
        }
    );
}

#[test]
fn irreversible_move_completes_repetition_evidence_without_inventing_game_history() {
    let mut position = Position::from_fen(&Position::startpos().to_fen()).unwrap();
    position.make_uci("e2e4").unwrap();
    let classification = position.classify_position().unwrap();
    assert_eq!(
        classification.history_evidence.completeness,
        HistoryCompleteness::UnknownPrefix
    );
    assert!(classification.history_evidence.repetition_is_complete);
    assert_eq!(
        current(&classification, ClaimReason::ThreefoldRepetition),
        Availability::Unavailable
    );
    assert_eq!(
        classification.history_evidence.fivefold_repetition,
        Availability::Unavailable
    );
}

#[test]
fn fifty_move_claim_has_current_and_intended_evidence_and_resets_on_pawn_moves() {
    let mut position = Position::from_fen("7k/8/8/8/8/8/P7/KR6 w - - 99 1").unwrap();
    let before = position.classify_position().unwrap();
    assert_eq!(before.play_status, PlayStatus::Ongoing);
    assert_eq!(
        current(&before, ClaimReason::FiftyMove),
        Availability::Unavailable
    );
    assert!(intended(&before, ClaimReason::FiftyMove, "b1b2"));
    assert!(!intended(&before, ClaimReason::FiftyMove, "a2a3"));
    assert!(!intended(&before, ClaimReason::FiftyMove, "a2a4"));

    position.make_uci("b1b2").unwrap();
    let after = position.classify_position().unwrap();
    assert_eq!(after.play_status, PlayStatus::Ongoing);
    assert_eq!(
        current(&after, ClaimReason::FiftyMove),
        Availability::Available
    );

    let pawn_child = Position::from_fen("7k/8/8/8/8/8/P7/KR6 w - - 99 1")
        .unwrap()
        .preview_move(BoardMove::from_uci("a2a3").unwrap())
        .unwrap();
    assert_eq!(pawn_child.halfmove_clock(), 0);
    assert_eq!(
        current(
            &pawn_child.classify_position().unwrap(),
            ClaimReason::FiftyMove
        ),
        Availability::Unavailable
    );
}

#[test]
fn capture_cannot_be_used_as_an_intended_fifty_move_claim() {
    let position = Position::from_fen("7k/8/8/8/8/1n6/8/KR6 w - - 99 1").unwrap();
    let classification = position.classify_position().unwrap();
    assert!(!intended(&classification, ClaimReason::FiftyMove, "b1b3"));
    let captured = position
        .preview_move(BoardMove::from_uci("b1b3").unwrap())
        .unwrap();
    assert_eq!(captured.halfmove_clock(), 0);
}

#[test]
fn seventy_five_move_boundary_is_automatic_and_mate_on_that_move_takes_precedence() {
    let mut quiet = Position::from_fen("7k/8/8/8/8/8/8/KR6 w - - 149 1").unwrap();
    assert_eq!(
        quiet.classify_position().unwrap().play_status,
        PlayStatus::Ongoing
    );
    quiet.make_uci("b1b2").unwrap();
    assert_eq!(quiet.halfmove_clock(), 150);
    assert_eq!(
        quiet.classify_position().unwrap().play_status,
        PlayStatus::Terminal {
            reason: TerminalReason::SeventyFiveMove,
            winner: None,
        }
    );

    let mut mate = Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 149 1").unwrap();
    mate.make_uci("g6g7").unwrap();
    assert_eq!(mate.halfmove_clock(), 150);
    let classification = mate.classify_position().unwrap();
    assert_eq!(
        classification.play_status,
        PlayStatus::Terminal {
            reason: TerminalReason::Checkmate,
            winner: Some(Color::White),
        }
    );
    assert_eq!(
        classification.history_evidence.completeness,
        HistoryCompleteness::UnknownPrefix
    );
}

#[test]
fn no_legal_moves_distinguishes_mate_from_stalemate_with_unknown_history() {
    for (fen, reason, winner) in [
        (
            "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
            TerminalReason::Checkmate,
            Some(Color::White),
        ),
        (
            "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
            TerminalReason::Stalemate,
            None,
        ),
    ] {
        let position = Position::from_fen(fen).unwrap();
        assert!(position.legal_moves().is_empty(), "{fen}");
        let classification = position.classify_position().unwrap();
        assert_eq!(
            classification.play_status,
            PlayStatus::Terminal { reason, winner },
            "{fen}"
        );
        assert_eq!(
            classification.history_evidence.completeness,
            HistoryCompleteness::UnknownPrefix
        );
    }
}

#[test]
fn dead_position_detector_certifies_only_its_proven_material_subset() {
    for fen in [
        "7k/8/8/8/8/8/8/K7 w - - 0 1",
        "7k/8/8/8/8/8/8/KB6 w - - 0 1",
        "7k/8/8/8/8/8/8/KN6 w - - 0 1",
        "7k/8/8/8/8/3b4/8/KB6 w - - 0 1",
    ] {
        assert_eq!(
            Position::from_fen(fen)
                .unwrap()
                .classify_position()
                .unwrap()
                .play_status,
            PlayStatus::Terminal {
                reason: TerminalReason::DeadPosition,
                winner: None
            },
            "{fen}",
        );
    }
    for fen in [
        "7k/8/8/8/8/8/8/KNN5 w - - 0 1",
        "7k/8/8/8/8/3n4/8/KN6 w - - 0 1",
        "7k/8/8/8/8/3n4/8/KB6 w - - 0 1",
        "7k/8/8/8/3b4/8/8/KB6 w - - 0 1",
    ] {
        assert_eq!(
            Position::from_fen(fen)
                .unwrap()
                .classify_position()
                .unwrap()
                .play_status,
            PlayStatus::Ongoing,
            "{fen}"
        );
    }
}

#[test]
fn intended_claim_preview_errors_are_not_hidden_as_success_or_unknown_evidence() {
    let position = Position::from_fen("7k/8/8/8/8/8/8/KR6 b - - 0 4294967295").unwrap();
    assert_eq!(
        position.classify_position().unwrap_err(),
        rz_position::PositionError::CounterOverflow
    );
}
