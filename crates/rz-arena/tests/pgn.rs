use rz_arena::{ArenaPlan, GameResult, PgnLimits, PlanLimits, audit_pair_pgn, opening_pgn};
use rz_experiments::{HistoryCompleteness, InitialPosition, RunManifest};

const FIXTURE: &str = include_str!("../../../experiments/baselines/fixtures/e01-input.json");
const MATE_MOVES: &str = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#";
const MATE_FEN: &str = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4";
fn input() -> RunManifest {
    {
        let mut input = RunManifest::from_json(FIXTURE).unwrap();
        input.contract_revision = Some("0.1".into());
        input
    }
}
fn plan(input: RunManifest) -> ArenaPlan {
    ArenaPlan::build(
        &input.lock().unwrap(),
        PlanLimits {
            max_pairs: 1,
            max_json_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
}
fn limits() -> PgnLimits {
    PgnLimits {
        max_bytes: 1048576,
        max_plies: 256,
    }
}
fn pair_pgn(
    plan: &ArenaPlan,
    moves: &str,
    result: &str,
    termination: &str,
    extra_tags: &str,
) -> String {
    let pair = &plan.pairs()[0];
    pair.execution_order.iter().map(|&index| {
        let game = &pair.games[index];
        format!("[Event \"CPU audit fixture only\"]\n[White \"{}\"]\n[Black \"{}\"]\n[Result \"{result}\"]\n[Termination \"{termination}\"]\n{extra_tags}\n{moves} {result}\n\n",game.white_engine,game.black_engine)
    }).collect()
}
fn audit(plan: &ArenaPlan, pgn: &str) -> Result<rz_arena::PairPgnAudit, rz_arena::ArenaError> {
    audit_pair_pgn(plan, &plan.pairs()[0].id, pgn, limits())
}

#[test]
fn opening_book_preserves_full_startpos_prefix_and_uses_a_legal_san() {
    let p = plan(input());
    let pgn = opening_pgn(&p, &p.pairs()[0].id).unwrap();
    assert!(pgn.ends_with("1. e4 e5 *\n"));
    assert!(!pgn.contains("FEN"));
    assert!(pgn.contains("[Result \"*\"]"));
}

#[test]
fn exact_mate_pair_has_real_fen_contract_identity_and_swapped_engine_names() {
    let p = plan(input());
    let result = audit(
        &p,
        &pair_pgn(&p, MATE_MOVES, "1-0", "normal", "[PlyCount \"7\"]\n"),
    )
    .unwrap();
    assert_eq!(result.engine_contract_revision, "0.1");
    assert_eq!(
        result.rules_source_commit,
        "118dc0311a88e143be285703940321dc16261f6a"
    );
    assert_eq!(
        result.contract_source_commit,
        "67284c4f66f7a7ae9f46fa63dfd50e7410eb6845"
    );
    assert_eq!(result.start_state_semantic_sha256.len(), 64);
    assert_ne!(
        result.start_state_semantic_sha256,
        result.opening_input_sha256
    );
    for game in &result.games {
        assert_eq!(
            game.uci_moves,
            ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
        );
        assert_eq!(game.final_fen, MATE_FEN);
        assert_eq!(game.result, GameResult::WhiteWin);
        assert_eq!(game.classification, "rules_terminal");
        assert_eq!(game.terminal_reason.as_deref(), Some("checkmate"));
    }
    assert_eq!(result.games[0].white_engine, "baseline-fixture");
    assert_eq!(result.games[1].white_engine, "candidate-fixture");
    let repeated = audit(&p, &pair_pgn(&p, MATE_MOVES, "1-0", "normal", "")).unwrap();
    assert_eq!(
        result.start_state_semantic_sha256,
        repeated.start_state_semantic_sha256
    );
}

#[test]
fn pinned_fastchess_uci_movetext_is_checked_with_the_same_a_rules() {
    let p = plan(input());
    let moves = "1. e2e4 {book} e7e5 {book} 2. d1h5 {0.00/1 0.001s} b8c6 3. f1c4 g8f6 4. h5f7 {White mates}";
    let result = audit(&p, &pair_pgn(&p, moves, "1-0", "normal", "")).unwrap();
    assert_eq!(result.games[0].final_fen, MATE_FEN);
}

#[test]
fn labels_and_prefix_cannot_hide_a_different_position_or_result() {
    let p = plan(input());
    let valid = pair_pgn(&p, MATE_MOVES, "1-0", "normal", "");
    for changed in [
        valid.replace("Qxf7#", "Qxf7+"),
        valid.replace("e4 e5", "d4 d5"),
        valid.replace("1-0", "0-1"),
        valid.replace("[Result \"1-0\"]", "[Result \"0-1\"]"),
        valid.replace(
            "[White \"baseline-fixture\"]",
            "[White \"candidate-fixture\"]",
        ),
        valid.replace("2. Qh5", "23. Qh5"),
        valid.replace("Qxf7#", "Qxf7# 4... a6"),
        valid.replace("1. e4 e5", "2. Qh5 Nc6"),
    ] {
        assert!(
            audit(&p, &changed).is_err(),
            "accepted changed PGN: {changed}"
        );
    }
}

#[test]
fn an_extra_missing_or_reordered_game_is_not_a_complete_pair() {
    let p = plan(input());
    let valid = pair_pgn(&p, MATE_MOVES, "1-0", "normal", "");
    let second_game = valid.rfind("[Event").unwrap();
    assert!(audit(&p, &valid[..second_game]).is_err());
    assert!(audit(&p, &(valid.clone() + &valid)).is_err());
    let reordered = format!("{}{}", &valid[second_game..], &valid[..second_game]);
    assert!(audit(&p, &reordered).is_err());
}

#[test]
fn unsupported_pgn_syntax_is_rejected_instead_of_skipped() {
    let p = plan(input());
    let valid = pair_pgn(&p, MATE_MOVES, "1-0", "normal", "");
    for added in [
        "(1. d4)",
        "$1",
        "; ignored line\n",
        "% ignored line\n",
        "{nested {comment}}",
    ] {
        assert!(
            audit(&p, &valid.replace("1. e4", &format!("{added} 1. e4"))).is_err(),
            "accepted {added}"
        );
    }
    assert!(
        audit(
            &p,
            &valid.replace(
                "[Event \"CPU audit fixture only\"]",
                "[Event \"first\"]\n[Event \"second\"]"
            )
        )
        .is_err()
    );
    assert!(audit(&p, &valid.replace("CPU audit fixture only", "bad\\nvalue")).is_err());
    assert!(
        audit(
            &p,
            &valid.replace("[Termination", "[Variant \"Chess960\"]\n[Termination")
        )
        .is_err()
    );
}

#[test]
fn byte_ply_comment_and_header_budgets_are_enforced() {
    let p = plan(input());
    let valid = pair_pgn(&p, MATE_MOVES, "1-0", "normal", "");
    for bounds in [
        PgnLimits {
            max_bytes: 1,
            max_plies: 256,
        },
        PgnLimits {
            max_bytes: 1048576,
            max_plies: 6,
        },
        PgnLimits {
            max_bytes: 0,
            max_plies: 1,
        },
        PgnLimits {
            max_bytes: 1048576,
            max_plies: 4096,
        },
    ] {
        assert!(audit_pair_pgn(&p, &p.pairs()[0].id, &valid, bounds).is_err());
    }
    assert!(
        audit(
            &p,
            &valid.replace("1. e4", &format!("{{{}}} 1. e4", "x".repeat(16385)))
        )
        .is_err()
    );
    assert!(
        audit(
            &p,
            &valid.replace(
                "[Event \"CPU audit fixture only\"]",
                &format!("[Event \"{}\"]", "x".repeat(8193))
            )
        )
        .is_err()
    );
}

#[test]
fn a_fen_origin_keeps_black_turn_rights_counters_and_unknown_history() {
    let mut i = input();
    let opening = &mut i.input.openings[0];
    opening.initial = InitialPosition::Fen;
    opening.fen = Some("r3k2r/8/8/3pP3/8/8/8/R3K2R b KQkq - 17 42".into());
    opening.history = HistoryCompleteness::UnknownPrefix;
    opening.history_origin = "imported-fen-unknown-prefix".into();
    opening.moves = vec!["e8g8".into(), "e1c1".into()];
    let p = plan(i);
    let book = opening_pgn(&p, &p.pairs()[0].id).unwrap();
    assert!(book.contains("[FEN \"r3k2r/8/8/3pP3/8/8/8/R3K2R b KQkq - 17 42\"]"));
    assert!(book.ends_with("42... O-O 43. O-O-O *\n"));
    let tags = "[SetUp \"1\"]\n[FEN \"r3k2r/8/8/3pP3/8/8/8/R3K2R b KQkq - 17 42\"]\n";
    let valid = pair_pgn(
        &p,
        "42... O-O 43. O-O-O {Black disconnects}",
        "1-0",
        "abandoned",
        tags,
    );
    let result = audit(&p, &valid).unwrap();
    assert_eq!(
        result.games[0].final_fen,
        "r4rk1/8/8/3pP3/8/8/8/2KR3R b - - 19 43"
    );
    assert_eq!(result.games[0].classification, "engine_loss");
    assert!(audit(&p, &valid.replace("17 42", "0 1")).is_err());
    assert!(audit(&p, &valid.replace("42...", "42.")).is_err());
}

#[test]
fn fen_startpos_origin_remains_unknown_and_has_a_different_semantic_identity() {
    let original = plan(input());
    let mut i = input();
    i.input.openings[0].initial = InitialPosition::Fen;
    i.input.openings[0].fen =
        Some("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into());
    assert!(
        i.clone().lock().is_err(),
        "FEN-only history may not be relabelled complete"
    );
    i.input.openings[0].history = HistoryCompleteness::UnknownPrefix;
    let p = plan(i);
    let book = opening_pgn(&p, &p.pairs()[0].id).unwrap();
    assert!(book.contains("[SetUp \"1\"]"));
    let tags =
        "[SetUp \"1\"]\n[FEN \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\"]\n";
    let restored = audit(&p, &pair_pgn(&p, MATE_MOVES, "1-0", "normal", tags)).unwrap();
    let complete = audit(
        &original,
        &pair_pgn(&original, MATE_MOVES, "1-0", "normal", ""),
    )
    .unwrap();
    assert_eq!(restored.games[0].final_fen, complete.games[0].final_fen);
    assert_ne!(
        restored.start_state_semantic_sha256,
        complete.start_state_semantic_sha256
    );
    assert!(audit(&p, &pair_pgn(&p, MATE_MOVES, "1-0", "normal", "")).is_err());
}

#[test]
fn en_passant_and_every_promotion_are_serialized_by_checked_a_moves() {
    let mut i = input();
    let o = &mut i.input.openings[0];
    o.initial = InitialPosition::Fen;
    o.history = HistoryCompleteness::UnknownPrefix;
    o.fen = Some("4k2r/8/8/3pP3/8/8/8/4K2R w Kk d6 0 2".into());
    o.moves = vec!["e5d6".into()];
    let p = plan(i);
    assert!(
        opening_pgn(&p, &p.pairs()[0].id)
            .unwrap()
            .ends_with("2. exd6 *\n")
    );
    for (uci, expected) in [
        ("a7a8q", "a8=Q+"),
        ("a7a8r", "a8=R+"),
        ("a7a8b", "a8=B"),
        ("a7a8n", "a8=N"),
    ] {
        let mut i = input();
        let o = &mut i.input.openings[0];
        o.initial = InitialPosition::Fen;
        o.history = HistoryCompleteness::UnknownPrefix;
        o.fen = Some("7k/P7/8/8/8/8/8/4KR2 w - - 0 2".into());
        o.moves = vec![uci.into()];
        let p = plan(i);
        assert!(
            opening_pgn(&p, &p.pairs()[0].id)
                .unwrap()
                .ends_with(&format!("2. {expected} *\n"))
        );
    }
}

#[test]
fn a_exact_stalemate_is_a_draw_and_a_threefold_claim_is_not_terminal() {
    let mut i = input();
    let o = &mut i.input.openings[0];
    o.initial = InitialPosition::Fen;
    o.history = HistoryCompleteness::UnknownPrefix;
    o.fen = Some("7k/5K2/6Q1/8/8/8/8/8 b - - 4 12".into());
    o.moves.clear();
    let p = plan(i);
    let tags = "[SetUp \"1\"]\n[FEN \"7k/5K2/6Q1/8/8/8/8/8 b - - 4 12\"]\n";
    let result = audit(
        &p,
        &pair_pgn(&p, "{Draw by stalemate}", "1/2-1/2", "normal", tags),
    )
    .unwrap();
    assert_eq!(
        result.games[0].terminal_reason.as_deref(),
        Some("stalemate")
    );
    assert!(opening_pgn(&p, &p.pairs()[0].id).is_err());
    let mut i = input();
    i.input.openings[0].moves.clear();
    let p = plan(i);
    let claim = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 {Draw by 3-fold repetition}";
    assert!(audit(&p, &pair_pgn(&p, claim, "1/2-1/2", "normal", "")).is_err());
}

#[test]
fn arbitrary_adjudication_timeout_text_and_ambiguous_abandoned_are_not_losses() {
    let p = plan(input());
    for (moves, result, termination) in [
        ("1. e4 e5 {Draw by adjudication}", "1/2-1/2", "adjudication"),
        ("1. e4 e5 {White's connection stalls}", "0-1", "abandoned"),
        ("1. e4 e5 {something timed out}", "0-1", "time forfeit"),
        ("1. e4 e5 {Black disconnects}", "0-1", "abandoned"),
        ("1. e4 e5 {White disconnects}", "1-0", "abandoned"),
    ] {
        assert!(audit(&p, &pair_pgn(&p, moves, result, termination, "")).is_err());
    }
    let timeout = audit(
        &p,
        &pair_pgn(
            &p,
            "1. e4 e5 {0.0/0 0.1s, White loses on time (12ms overrun)}",
            "0-1",
            "time forfeit",
            "",
        ),
    )
    .unwrap();
    assert_eq!(
        timeout.games[0].engine_failure,
        Some(rz_arena::EngineFailureKind::Timeout)
    );
}

#[test]
fn an_illegal_reply_is_not_applied_to_the_final_a_state() {
    let p = plan(input());
    let loss = pair_pgn(
        &p,
        "1. e4 e5 {0.00/0 0.001s, White makes an illegal move: e2e5}",
        "0-1",
        "illegal move",
        "[PlyCount \"3\"]\n",
    );
    let result = audit(&p, &loss).unwrap();
    assert_eq!(result.games[0].uci_moves, ["e2e4", "e7e5"]);
    assert_eq!(
        result.games[0].engine_failure,
        Some(rz_arena::EngineFailureKind::IllegalMove)
    );
    assert!(audit(&p, &loss.replace("e2e5", "g1f3")).is_err());
    assert!(audit(&p, &loss.replace("e2e5", "e2e5 extra")).is_err());
    assert!(audit(&p, &loss.replace("[PlyCount \"3\"]", "[PlyCount \"2\"]")).is_err());
    assert!(audit(&p, &loss.replace("White makes", "Black makes")).is_err());
}

#[test]
fn san_requires_the_disambiguation_that_a_legal_moves_establish() {
    let mut i = input();
    let o = &mut i.input.openings[0];
    o.initial = InitialPosition::Fen;
    o.history = HistoryCompleteness::UnknownPrefix;
    o.fen = Some("4k3/8/8/8/8/8/4K3/R6R w - - 0 1".into());
    o.moves = vec!["a1e1".into()];
    let p = plan(i);
    assert!(
        opening_pgn(&p, &p.pairs()[0].id)
            .unwrap()
            .ends_with("1. Rae1 *\n")
    );
    let tags = "[SetUp \"1\"]\n[FEN \"4k3/8/8/8/8/8/4K3/R6R w - - 0 1\"]\n";
    let valid = pair_pgn(&p, "1. Rae1 {Black disconnects}", "1-0", "abandoned", tags);
    assert!(audit(&p, &valid).is_ok());
    assert!(audit(&p, &valid.replace("Rae1", "Re1")).is_err());
    assert!(audit(&p, &valid.replace("Rae1", "Rhe1")).is_err());
}

#[test]
fn a_automatic_draws_and_mate_priority_are_preserved_without_claim_adjudication() {
    let mut i = input();
    i.input.openings[0].moves.clear();
    let p = plan(i);
    let fivefold =
        "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3 Nf6 6. Ng1 Ng8 7. Nf3 Nf6 8. Ng1 Ng8";
    let result = audit(&p, &pair_pgn(&p, fivefold, "1/2-1/2", "normal", "")).unwrap();
    assert_eq!(
        result.games[0].terminal_reason.as_deref(),
        Some("fivefold_repetition")
    );
    for (fen, moves, outcome, reason) in [
        (
            "4k2r/8/8/8/8/8/8/4K2R w Kk - 150 76",
            "",
            "1/2-1/2",
            "seventy_five_move",
        ),
        (
            "4k3/8/8/8/8/8/8/4K3 w - - 4 12",
            "",
            "1/2-1/2",
            "dead_position",
        ),
        (
            "7k/5K2/6Q1/8/8/8/8/8 w - - 149 76",
            "76. Qg7#",
            "1-0",
            "checkmate",
        ),
    ] {
        let mut i = input();
        let o = &mut i.input.openings[0];
        o.initial = InitialPosition::Fen;
        o.history = HistoryCompleteness::UnknownPrefix;
        o.fen = Some(fen.into());
        o.moves.clear();
        let p = plan(i);
        let tags = format!("[SetUp \"1\"]\n[FEN \"{fen}\"]\n");
        let result = audit(&p, &pair_pgn(&p, moves, outcome, "normal", &tags)).unwrap();
        assert_eq!(result.games[0].terminal_reason.as_deref(), Some(reason));
        if reason == "checkmate" {
            assert!(result.games[0].final_fen.ends_with("150 76"));
        }
    }
}

#[test]
fn missing_or_unmatched_shared_contract_revision_blocks_execution_audit() {
    let mut i = input();
    i.contract_revision = None;
    let p = plan(i);
    assert!(opening_pgn(&p, &p.pairs()[0].id).is_err());
    assert!(audit(&p, &pair_pgn(&p, MATE_MOVES, "1-0", "normal", "")).is_err());
    let mut i = input();
    i.contract_revision = Some("0.2".into());
    let p = plan(i);
    assert!(opening_pgn(&p, &p.pairs()[0].id).is_err());
}

#[test]
fn the_ply_ceiling_keeps_one_a_claim_inspection_without_admitting_extra_moves() {
    let p = plan(input());
    let valid = pair_pgn(
        &p,
        "1. e4 e5 {White loses on time (12ms overrun)}",
        "0-1",
        "time forfeit",
        "",
    );
    let result = audit_pair_pgn(
        &p,
        &p.pairs()[0].id,
        &valid,
        PgnLimits {
            max_bytes: 1048576,
            max_plies: 2,
        },
    )
    .unwrap();
    assert_eq!(result.games[0].uci_moves.len(), 2);
    let too_long = pair_pgn(
        &p,
        "1. e4 e5 2. Nf3 {Black disconnects}",
        "1-0",
        "abandoned",
        "",
    );
    assert!(
        audit_pair_pgn(
            &p,
            &p.pairs()[0].id,
            &too_long,
            PgnLimits {
                max_bytes: 1048576,
                max_plies: 2
            }
        )
        .is_err()
    );
}

#[test]
fn exact_max_plies_adjudication_is_incomplete_and_never_a_scored_draw() {
    let mut i = input();
    i.protocol.max_plies = 4;
    assert_eq!(
        i.protocol.max_plies_outcome,
        rz_experiments::OutcomePolicy::Incomplete
    );
    let p = plan(i);
    let cutoff = pair_pgn(
        &p,
        "1. e4 {book} e5 {book} 2. Nf3 Nc6 {0.00/1 0.001s, Draw by adjudication}",
        "1/2-1/2",
        "adjudication",
        "[PlyCount \"4\"]\n",
    );
    let observed = audit(&p, &cutoff).unwrap();
    for (game, &index) in observed
        .games
        .iter()
        .zip(p.pairs()[0].execution_order.iter())
    {
        assert_eq!(game.game_id, p.pairs()[0].games[index].id);
        assert_eq!(game.classification, "incomplete");
        assert_eq!(
            game.result,
            GameResult::Draw,
            "raw runner declaration is retained without score authority"
        );
        assert_eq!(game.uci_moves.len(), 4);
        assert_eq!(game.terminal_reason, None);
        assert_eq!(game.loser_engine, None);
        assert_eq!(game.engine_failure, None);
    }
    for changed in [
        cutoff.replace("[PlyCount \"4\"]", "[PlyCount \"3\"]"),
        cutoff.replace("2. Nf3 Nc6", "2. Nf3"),
        cutoff.replace("Draw by adjudication", "Draw by adjudication: SyzygyTB"),
        cutoff.replace("Draw by adjudication", "White wins by adjudication"),
        cutoff.replace("Draw by adjudication", "not Draw by adjudication"),
        cutoff.replace("1/2-1/2", "1-0"),
    ] {
        assert!(
            audit(&p, &changed).is_err(),
            "accepted unsupported cutoff {changed}"
        );
    }
    let mut i = input();
    i.protocol.max_plies = 7;
    let p = plan(i);
    let terminal = pair_pgn(
        &p,
        &format!("{MATE_MOVES} {{Draw by adjudication}}"),
        "1/2-1/2",
        "adjudication",
        "",
    );
    assert!(
        audit(&p, &terminal).is_err(),
        "an A terminal may not become a limit draw or incomplete result"
    );
}
