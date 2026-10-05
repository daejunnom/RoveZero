#![cfg(feature = "native-cuda")]

use rz_arena::*;
use rz_experiments::*;

fn fixture() -> (OpeningSpec, PairPgnAudit, String) {
    fixture_clock(clock())
}

fn fixture_clock(clock: NativeGameClockV3) -> (OpeningSpec, PairPgnAudit, String) {
    let opening = OpeningSpec {
        id: "clock-test".into(),
        initial: InitialPosition::Startpos,
        fen: None,
        moves: vec!["e2e4".into(), "e7e5".into()],
        history: HistoryCompleteness::Complete,
        history_origin: "startpos-complete-trace".into(),
    };
    let games = [
        GameSpec {
            id: "clock-1".into(),
            white_engine: "base".into(),
            black_engine: "candidate".into(),
            engine_seeds: Default::default(),
            engine_slots: Default::default(),
        },
        GameSpec {
            id: "clock-2".into(),
            white_engine: "candidate".into(),
            black_engine: "base".into(),
            engine_seeds: Default::default(),
            engine_slots: Default::default(),
        },
    ];
    let pair = PairSpec {
        id: "clock-pair".into(),
        ordinal: 0,
        opening: opening.clone(),
        opening_input_sha256: canonical_sha256(&opening).unwrap(),
        games,
        execution_order: [0, 1],
    };
    let pgn=pair.games.iter().map(|g|format!("[White \"{}\"]\n[Black \"{}\"]\n[Result \"1-0\"]\n[Termination \"normal\"]\n[TimeControl \"{}\"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n\n",g.white_engine,g.black_engine,clock.pgn_time_control())).collect::<String>();
    let audit = audit_pair_pgn_for_spec(
        &pair,
        &pgn,
        PgnLimits {
            max_bytes: 65536,
            max_plies: 256,
        },
        PgnOutcomePolicy {
            engine_failure: OutcomePolicy::Loss,
            max_plies_outcome: OutcomePolicy::Incomplete,
            claim_policy: ClaimPolicy::AutomaticAcceptance,
            max_game_plies: 256,
        },
    )
    .unwrap();
    let mut trace = String::new();
    for g in &pair.games {
        let mut remaining = [clock.base_ms; 2];
        for ply in 2..7 {
            let s = ply % 2;
            let name = if s == 0 {
                &g.white_engine
            } else {
                &g.black_engine
            };
            let before = remaining[s];
            remaining[s] = before - 101 + clock.increment_ms;
            trace.push_str(&format!("[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- RZ_CLOCK_V1 engine={name} before={before} increment={} elapsed_ns=100000001 after={} valid=true\n",clock.increment_ms,remaining[s]));
        }
    }
    (opening, audit, trace)
}
fn clock() -> NativeGameClockV3 {
    NativeGameClockV3 {
        base_ms: 30000,
        increment_ms: 100,
    }
}

#[test]
fn whole_clock_replays_audited_pgn_and_resets_on_swapped_game() {
    let (o, p, t) = fixture();
    let c = validate_pilot_clock_trace(t.as_bytes(), &p, &o, clock()).unwrap();
    assert_eq!(c.games.len(), 2);
    for g in c.games {
        assert_eq!(g.searched_plies, 5);
        assert_eq!(g.charged_ms, [303, 202]);
        assert_eq!(g.white_remaining_ms, 29997);
        assert_eq!(g.black_remaining_ms, 29998);
    }
}

#[test]
fn blitz_clock_preserves_fischer_increment_color_reset_and_pgn_time_control() {
    let c = NativeGameClockV3 {
        base_ms: 120_000,
        increment_ms: 1_000,
    };
    let (o, mut p, t) = fixture_clock(c);
    let audit = validate_pilot_clock_trace(t.as_bytes(), &p, &o, c).unwrap();
    for game in audit.games {
        assert_eq!(game.white_remaining_ms, 122_697);
        assert_eq!(game.black_remaining_ms, 121_798);
        assert_eq!(game.charged_ms, [303, 202]);
    }
    for tag in [None, Some("60+1".into()), Some("120+0".into())] {
        p.games[0].time_control = tag;
        let audit = validate_pilot_clock_trace(t.as_bytes(), &p, &o, c).unwrap();
        assert_eq!(audit.expected_pgn_time_control, "120+1");
        assert_eq!(audit.pgn_time_control_warnings.len(), 1);
    }
}

#[test]
fn clock_rejects_grace_extra_increment_wrong_role_rounding_and_forged_logs() {
    let (o, p, t) = fixture();
    for altered in [
        t.replacen("before=30000", "before=30100", 1),
        t.replacen("after=29999", "after=30000", 1),
        t.replacen("engine=base", "engine=candidate", 1),
        t.replacen("increment=100", "increment=0", 1),
        t.replacen("valid=true", "valid=false", 1),
        t.replacen("elapsed_ns=100000001", "elapsed_ns=30000000001", 1),
        t.replacen("before=30000", "before=030000", 1),
        t.replace("[TRACE ]", "[Engine]"),
        t.trim_end_matches('\n').into(),
        format!("{t}{}\n", t.lines().next().unwrap()),
        t.lines().skip(1).map(|l| format!("{l}\n")).collect(),
    ] {
        assert!(validate_pilot_clock_trace(altered.as_bytes(), &p, &o, clock()).is_err());
    }
}

#[test]
fn clock_does_not_erase_observed_engine_loss_or_accept_changed_pgn_prefix() {
    let (o, mut p, t) = fixture();
    p.games[0].classification = "engine_loss".into();
    assert!(validate_pilot_clock_trace(t.as_bytes(), &p, &o, clock()).is_err());
    assert_eq!(p.games[0].classification, "engine_loss");
    let (mut o, p, t) = fixture();
    o.moves[1] = "c7c5".into();
    assert!(validate_pilot_clock_trace(t.as_bytes(), &p, &o, clock()).is_err());
}

#[test]
fn startup_loss_is_preserved_without_a_pgn_or_a_successful_provider_receipt() {
    let (opening, audit, _) = fixture();
    let pair = PairSpec {
        id: "failure-pair".into(),
        ordinal: 0,
        opening_input_sha256: canonical_sha256(&opening).unwrap(),
        opening,
        games: std::array::from_fn(|i| GameSpec {
            id: audit.games[i].game_id.clone(),
            white_engine: audit.games[i].white_engine.clone(),
            black_engine: audit.games[i].black_engine.clone(),
            engine_seeds: Default::default(),
            engine_slots: Default::default(),
        }),
        execution_order: [0, 1],
    };
    let trace = concat!(
        "[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- Game 1 between base and candidate starting\n",
        "[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- Game 1 between base and candidate finished\n",
        "[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- Game 2 between candidate and base starting\n",
        "[FATAL ] [10:51:44.181984] <                    > fastchess --- Fatal; base engine startup failure: \"Engine didn't respond to uciok after startup\"\n",
        "[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- Game 2 between candidate and base finished\n"
    );
    let a = audit_pilot_startup_failure_trace(trace.as_bytes(), &pair)
        .unwrap()
        .unwrap();
    assert_eq!(a.startup_losses.len(), 1);
    let loss = &a.startup_losses[0];
    assert_eq!(loss.game_id, "clock-2");
    assert_eq!(loss.loser_engine, "base");
    assert_eq!(loss.declared_result, GameResult::WhiteWin);
    assert_eq!(loss.failure_stage, "startup_uci_handshake");
    // Neither quoted stderr nor an unanchored copy of the fatal text is trusted.
    for forged in [
        trace.replace("[FATAL ]", "[Engine]"),
        trace
            .lines()
            .filter(|l| !l.starts_with("[FATAL"))
            .map(|l| format!("{l}\n"))
            .collect(),
    ] {
        assert!(
            audit_pilot_startup_failure_trace(forged.as_bytes(), &pair)
                .unwrap()
                .is_none()
        );
    }
    for malformed in [
        trace.replacen("Fatal; base", "Fatal; stranger", 1),
        trace.replacen(
            "Game 2 between candidate and base starting",
            "Game 2 between base and candidate starting",
            1,
        ),
        trace.replacen(
            "[FATAL ] [10:51:44.181984]",
            "[FATAL ] [25:51:44.181984]",
            1,
        ),
        trace.replacen("[FATAL ]", "[FATALX]", 1),
        trace
            .lines()
            .filter(|l| !l.ends_with("starting"))
            .map(|l| format!("{l}\n"))
            .collect(),
        trace.trim_end_matches('\n').into(),
        trace.replace(
            "[FATAL ]",
            "[FATAL ] [10:51:44.181984] <                    > fastchess --- unsupported\n[FATAL ]",
        ),
    ] {
        assert!(audit_pilot_startup_failure_trace(malformed.as_bytes(), &pair).is_err());
    }
}
