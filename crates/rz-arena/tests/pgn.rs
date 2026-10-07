use rz_arena::{
    ArenaPlan, GameResult, PgnLimits, PgnOutcomePolicy, PlanLimits, audit_pair_pgn,
    audit_pair_pgn_for_spec, opening_pgn, opening_pgn_for_spec, validate_opening_artifact_for_spec,
};
use rz_experiments::{
    ArtifactRef, ClaimPolicy, HistoryCompleteness, InitialPosition, OutcomePolicy, RunManifest,
};
use sha2::{Digest, Sha256};

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

fn policy(max_game_plies: u32) -> PgnOutcomePolicy {
    PgnOutcomePolicy {
        engine_failure: OutcomePolicy::Loss,
        max_plies_outcome: OutcomePolicy::Incomplete,
        claim_policy: ClaimPolicy::ExplicitClaim,
        max_game_plies,
    }
}

#[test]
fn opening_pin_rejects_same_position_with_manual_headers_or_changed_prefix() {
    let mut opening = plan(input()).pairs()[0].opening.clone();
    opening.moves.clear();
    let text = opening_pgn_for_spec(&opening, 256).unwrap();
    let mut artifact = ArtifactRef {
        path: "opening.pgn".into(),
        bytes: text.len() as u64,
        sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        source: "https://github.com/daejunnom/RoveZero".into(),
        license: "MIT".into(),
    };
    assert_eq!(
        validate_opening_artifact_for_spec(&opening, 256, &artifact).unwrap(),
        text
    );
    let manual = "[Event \"RoveZero Stockfish19 paired pilot\"]\n[Result \"*\"]\n\n*\n";
    artifact.bytes = manual.len() as u64;
    artifact.sha256 = format!("{:x}", Sha256::digest(manual.as_bytes()));
    assert!(validate_opening_artifact_for_spec(&opening, 256, &artifact).is_err());
    artifact.bytes = text.len() as u64;
    artifact.sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    opening.moves = vec!["e2e4".into(), "e7e5".into()];
    assert!(validate_opening_artifact_for_spec(&opening, 256, &artifact).is_err());
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
fn automatic_claim_policy_requires_current_a_evidence_and_pinned_reason() {
    let mut i = input();
    i.input.openings[0].moves.clear();
    i.protocol.claim_policy = ClaimPolicy::AutomaticAcceptance;
    let p = plan(i);
    let moves = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8";
    let valid = pair_pgn(
        &p,
        &format!("{moves} {{Draw by 3-fold repetition}}"),
        "1/2-1/2",
        "normal",
        "",
    );
    let result = audit(&p, &valid).unwrap();
    assert_eq!(result.games[0].classification, "accepted_claim");
    assert_eq!(
        result.games[0].terminal_reason.as_deref(),
        Some("threefold_repetition")
    );
    for invalid in [
        valid.replace("3. Nf3 Nf6 4. Ng1 Ng8", ""),
        valid.replace("Draw by 3-fold repetition", "Draw by fifty moves rule"),
        valid.replace("Draw by 3-fold repetition", "an assumed repetition"),
        valid.replace("[Termination \"normal\"]", "[Termination \"adjudication\"]"),
    ] {
        assert!(audit(&p, &invalid).is_err());
    }
}

#[test]
fn automatic_fifty_move_claim_rejects_an_intended_only_claim() {
    for (halfmoves, available) in [(99, false), (100, true)] {
        let mut i = input();
        let o = &mut i.input.openings[0];
        let fen = format!("4k2r/8/8/8/8/8/8/4K2R w Kk - {halfmoves} 51");
        o.initial = InitialPosition::Fen;
        o.history = HistoryCompleteness::UnknownPrefix;
        o.fen = Some(fen.clone());
        o.moves.clear();
        i.protocol.claim_policy = ClaimPolicy::AutomaticAcceptance;
        let p = plan(i);
        let tags = format!("[SetUp \"1\"]\n[FEN \"{fen}\"]\n");
        let result = audit(
            &p,
            &pair_pgn(&p, "{Draw by fifty moves rule}", "1/2-1/2", "normal", &tags),
        );
        assert_eq!(result.is_ok(), available);
        if let Ok(result) = result {
            assert_eq!(result.games[0].classification, "accepted_claim");
            assert_eq!(
                result.games[0].terminal_reason.as_deref(),
                Some("fifty_move")
            );
        }
    }
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
    assert!(timeout.games[0].failure_evidence.is_none());
    assert!(
        serde_json::to_value(&timeout.games[0])
            .unwrap()
            .get("failure_evidence")
            .is_none()
    );
}

fn rules_fen(moves: &[&str]) -> String {
    let mut position = rz_position::Position::startpos();
    for mv in moves {
        position.make_uci(mv).unwrap();
    }
    position.to_fen()
}

#[test]
fn recorded_timeout_reply_is_retained_but_not_applied_to_the_a_game_state() {
    let p = plan(input());
    for (moves, result, count, committed, reply, overrun, white_lost) in [
        (
            "1. e4 e5 2. Nf3 {0.00/0 1.015s, White loses on time (5ms overrun)}",
            "0-1",
            3,
            vec!["e2e4", "e7e5"],
            "g1f3",
            5,
            true,
        ),
        (
            "1. e4 e5 2. Nf3 Nf6 {0.00/0 1.018s, Black loses on time (7ms overrun)}",
            "1-0",
            4,
            vec!["e2e4", "e7e5", "g1f3"],
            "g8f6",
            7,
            false,
        ),
    ] {
        let raw = pair_pgn(
            &p,
            moves,
            result,
            "time forfeit",
            &format!("[PlyCount \"{count}\"]\n"),
        );
        let audited = audit(&p, &raw).unwrap();
        for game in &audited.games {
            assert_eq!(game.classification, "engine_loss");
            assert_eq!(
                game.engine_failure,
                Some(rz_arena::EngineFailureKind::Timeout)
            );
            assert_eq!(game.uci_moves, committed);
            assert_eq!(game.final_fen, rules_fen(&committed));
            assert_eq!(
                game.loser_engine.as_ref(),
                Some(if white_lost {
                    &game.white_engine
                } else {
                    &game.black_engine
                })
            );
            let failure = game.failure_evidence.as_ref().unwrap();
            assert_eq!(
                failure.profile,
                "fastchess_recorded_unplayed_timeout_reply_v1"
            );
            assert_eq!(failure.unplayed_uci_reply, reply);
            assert_eq!(failure.declared_ply_count, count);
            assert_eq!(failure.overrun_ms, overrun);
            let serialized = serde_json::to_value(game).unwrap();
            assert_eq!(serialized["failure_evidence"]["unplayed_uci_reply"], reply);
            assert_eq!(serialized["failure_evidence"]["declared_ply_count"], count);
        }
    }
}

#[test]
fn a_mating_timeout_candidate_cannot_replace_the_prior_failure_boundary() {
    let mut i = input();
    i.input.openings[0].moves.clear();
    let p = plan(i);
    for (moves, result, count, committed, reply) in [
        (
            "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# {White loses on time (5ms overrun)}",
            "0-1",
            7,
            vec!["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"],
            "h5f7",
        ),
        (
            "1. f3 e5 2. g4 Qh4# {Black loses on time (7ms overrun)}",
            "1-0",
            4,
            vec!["f2f3", "e7e5", "g2g4"],
            "d8h4",
        ),
    ] {
        let raw = pair_pgn(
            &p,
            moves,
            result,
            "time forfeit",
            &format!("[PlyCount \"{count}\"]\n"),
        );
        let audited = audit(&p, &raw).unwrap();
        for game in &audited.games {
            assert_eq!(game.classification, "engine_loss");
            assert_eq!(game.terminal_reason, None);
            assert_eq!(game.uci_moves, committed);
            assert_eq!(game.final_fen, rules_fen(&committed));
            assert_eq!(
                game.failure_evidence.as_ref().unwrap().unplayed_uci_reply,
                reply
            );
        }
        // The candidate's hypothetical mate winner cannot override the timeout.
        let inverted = if result == "0-1" { "1-0" } else { "0-1" };
        let contradictory = raw
            .replace(
                &format!("[Result \"{result}\"]"),
                &format!("[Result \"{inverted}\"]"),
            )
            .replace(&format!("}} {result}"), &format!("}} {inverted}"));
        assert!(audit(&p, &contradictory).is_err());
    }
}

#[test]
fn recorded_timeout_reply_requires_the_exact_failure_profile_and_original_ply_count() {
    let p = plan(input());
    let raw = pair_pgn(
        &p,
        "1. e4 e5 2. Nf3 {0.00/0 1.015s, White loses on time (5ms overrun)}",
        "0-1",
        "time forfeit",
        "[PlyCount \"3\"]\n",
    );
    for invalid in [
        raw.replace("[PlyCount \"3\"]\n", ""),
        raw.replace("[PlyCount \"3\"]", "[PlyCount \"2\"]"),
        raw.replace("[PlyCount \"3\"]", "[PlyCount \"4\"]"),
        raw.replace("White loses", "Black loses"),
        raw.replace("White loses", "white loses"),
        raw.replace("5ms overrun", "-5ms overrun"),
        raw.replace("5ms overrun", "5ms overrun extra"),
        raw.replace("5ms overrun", "18446744073709551616ms overrun"),
        raw.replace("Nf3", "e2e5"),
        raw.replace("time forfeit", "abandoned"),
        raw.replace("time forfeit", "normal"),
        raw.replace("[Result \"0-1\"]", "[Result \"1-0\"]")
            .replace("} 0-1", "} 1-0"),
    ] {
        assert!(audit(&p, &invalid).is_err());
    }
    let opening_reply = pair_pgn(
        &p,
        "1. e4 e5 {Black loses on time (7ms overrun)}",
        "1-0",
        "time forfeit",
        "[PlyCount \"2\"]\n",
    );
    assert!(audit(&p, &opening_reply).is_err());
    let terminal_before_failure = pair_pgn(
        &p,
        &format!("{MATE_MOVES} {{Black loses on time (7ms overrun)}}"),
        "1-0",
        "time forfeit",
        "[PlyCount \"7\"]\n",
    );
    assert!(audit(&p, &terminal_before_failure).is_err());
}

#[test]
fn no_reply_timeout_preserves_the_current_a_turn_for_both_colors() {
    let p = plan(input());
    for (moves, result, count, committed) in [
        (
            "1. e4 e5 {White loses on time (12ms overrun)}",
            "0-1",
            2,
            vec!["e2e4", "e7e5"],
        ),
        (
            "1. e4 e5 2. Nf3 {Black loses on time (12ms overrun)}",
            "1-0",
            3,
            vec!["e2e4", "e7e5", "g1f3"],
        ),
    ] {
        let audited = audit(
            &p,
            &pair_pgn(
                &p,
                moves,
                result,
                "time forfeit",
                &format!("[PlyCount \"{count}\"]\n"),
            ),
        )
        .unwrap();
        for game in &audited.games {
            assert_eq!(game.classification, "engine_loss");
            assert_eq!(game.uci_moves, committed);
            assert_eq!(game.final_fen, rules_fen(&committed));
            assert!(game.failure_evidence.is_none());
        }
    }
}

#[test]
fn the_recorded_timeout_candidate_still_consumes_the_movetext_ply_budget() {
    let p = plan(input());
    let raw = pair_pgn(
        &p,
        "1. e4 e5 2. Nf3 {White loses on time (5ms overrun)}",
        "0-1",
        "time forfeit",
        "[PlyCount \"3\"]\n",
    );
    for (max_plies, accepted) in [(2, false), (3, true)] {
        let result = audit_pair_pgn(
            &p,
            &p.pairs()[0].id,
            &raw,
            PgnLimits {
                max_bytes: 1048576,
                max_plies,
            },
        );
        assert_eq!(result.is_ok(), accepted);
        if let Ok(audited) = result {
            assert_eq!(audited.games[0].uci_moves, ["e2e4", "e7e5"]);
            assert_eq!(
                audited.games[0]
                    .failure_evidence
                    .as_ref()
                    .unwrap()
                    .declared_ply_count,
                3
            );
        }
    }
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

#[test]
fn common_pair_helpers_reuse_rules_identity_and_preserve_execution_order() {
    let p = plan(input());
    let pair = &p.pairs()[0];
    assert_eq!(
        opening_pgn_for_spec(&pair.opening, 256).unwrap(),
        opening_pgn(&p, &pair.id).unwrap()
    );
    let pgn = pair_pgn(&p, MATE_MOVES, "1-0", "normal", "");
    let wrapped = audit(&p, &pgn).unwrap();
    let common = audit_pair_pgn_for_spec(pair, &pgn, limits(), policy(256)).unwrap();
    assert_eq!(common.engine_contract_revision, "0.1");
    assert_eq!(common.opening_input_sha256, wrapped.opening_input_sha256);
    assert_eq!(
        common.start_state_semantic_sha256,
        wrapped.start_state_semantic_sha256
    );
    for (actual, expected) in common.games.iter().zip(&wrapped.games) {
        assert_eq!(actual.game_id, expected.game_id);
        assert_eq!(actual.white_engine, expected.white_engine);
        assert_eq!(actual.black_engine, expected.black_engine);
        assert_eq!(actual.uci_moves, expected.uci_moves);
        assert_eq!(actual.final_fen, expected.final_fen);
        assert_eq!(actual.classification, "rules_terminal");
        assert_eq!(actual.terminal_reason, expected.terminal_reason);
    }
    let mut reversed = pair.clone();
    reversed.execution_order.reverse();
    let split = pgn.rfind("[Event").unwrap();
    let reversed_pgn = format!("{}{}", &pgn[split..], &pgn[..split]);
    let result = audit_pair_pgn_for_spec(&reversed, &reversed_pgn, limits(), policy(256)).unwrap();
    assert_eq!(
        result.games[0].game_id,
        reversed.games[reversed.execution_order[0]].id
    );
    assert!(audit_pair_pgn_for_spec(&reversed, &pgn, limits(), policy(256)).is_err());
    for order in [[0, 0], [1, 1], [2, 0]] {
        let mut invalid = pair.clone();
        invalid.execution_order = order;
        assert!(audit_pair_pgn_for_spec(&invalid, &pgn, limits(), policy(256)).is_err());
    }
    let mut invalid = pair.clone();
    invalid.games[1].white_engine = invalid.games[0].white_engine.clone();
    assert!(audit_pair_pgn_for_spec(&invalid, &pgn, limits(), policy(256)).is_err());
    let mut invalid = pair.clone();
    invalid.games[1].id = invalid.games[0].id.clone();
    assert!(audit_pair_pgn_for_spec(&invalid, &pgn, limits(), policy(256)).is_err());
}

#[test]
fn common_opening_helper_preserves_full_fen_and_rejects_ambiguous_history_or_ceiling() {
    let p = plan(input());
    let mut opening = p.pairs()[0].opening.clone();
    opening.initial = InitialPosition::Fen;
    opening.fen = Some("r3k2r/8/8/3pP3/8/8/8/R3K2R b KQkq - 17 42".into());
    opening.history = HistoryCompleteness::UnknownPrefix;
    opening.history_origin = "imported-fen-unknown-prefix".into();
    opening.moves = vec!["e8g8".into(), "e1c1".into()];
    let book = opening_pgn_for_spec(&opening, 2).unwrap();
    assert!(book.contains("[FEN \"r3k2r/8/8/3pP3/8/8/8/R3K2R b KQkq - 17 42\"]"));
    assert!(book.ends_with("42... O-O 43. O-O-O *\n"));
    for max_plies in [0, 1, 4096] {
        assert!(opening_pgn_for_spec(&opening, max_plies).is_err());
    }
    opening.history = HistoryCompleteness::Complete;
    assert!(opening_pgn_for_spec(&opening, 2).is_err());
    opening.history = HistoryCompleteness::UnknownPrefix;
    opening.initial = InitialPosition::Startpos;
    assert!(opening_pgn_for_spec(&opening, 2).is_err());
}

#[test]
fn common_pair_cutoff_preserves_incomplete_and_never_admits_a_longer_terminal() {
    let p = plan(input());
    let pair = &p.pairs()[0];
    let cutoff = pair_pgn(
        &p,
        "1. e4 e5 2. Nf3 Nc6 {Draw by adjudication}",
        "1/2-1/2",
        "adjudication",
        "[PlyCount \"4\"]\n",
    );
    let result = audit_pair_pgn_for_spec(pair, &cutoff, limits(), policy(4)).unwrap();
    for game in &result.games {
        assert_eq!(game.classification, "incomplete");
        assert_eq!(
            game.result,
            GameResult::Draw,
            "raw declaration grants no scored draw"
        );
        assert_eq!(game.uci_moves.len(), 4);
        assert_eq!(game.terminal_reason, None);
        assert_eq!(game.engine_failure, None);
    }
    let terminal = pair_pgn(&p, MATE_MOVES, "1-0", "normal", "");
    assert!(audit_pair_pgn_for_spec(pair, &terminal, limits(), policy(4)).is_err());
    let mut unsupported_cutoff = policy(4);
    unsupported_cutoff.max_plies_outcome = OutcomePolicy::Loss;
    assert!(audit_pair_pgn_for_spec(pair, &cutoff, limits(), unsupported_cutoff).is_err());
    let loss = pair_pgn(&p, "1. e4 e5 {White disconnects}", "0-1", "abandoned", "");
    let result = audit_pair_pgn_for_spec(pair, &loss, limits(), policy(4)).unwrap();
    assert_eq!(result.games[0].classification, "engine_loss");
    let mut non_loss = policy(4);
    non_loss.engine_failure = OutcomePolicy::Incomplete;
    assert!(audit_pair_pgn_for_spec(pair, &loss, limits(), non_loss).is_err());
    assert!(audit_pair_pgn_for_spec(pair, &cutoff, limits(), policy(0)).is_err());
}
