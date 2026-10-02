use rz_contracts::{Color, HistoryCompleteness, PlayStatus};
use rz_position::{BoardMove, PositionLimits};
use rz_search::driver::CheckedPosition;
use rz_uci::{
    engine::{OwnerRegistry, RulesUciPort},
    *,
};
use std::sync::Arc;

fn port() -> RulesUciPort {
    RulesUciPort::new(
        Arc::new(OwnerRegistry::default()),
        PositionLimits::default(),
    )
}

#[test]
fn actual_rules_trace_is_atomic_and_forks_preserve_frozen_state() {
    let port = port();
    let start = port.prepare(&PositionSpec::default()).unwrap().snapshot;
    let shared = start.clone();
    assert_eq!(
        start.state().snapshot().identity(),
        shared.state().snapshot().identity()
    );
    let movement = rz_contracts::Move::try_from(BoardMove::from_uci("e2e4").unwrap()).unwrap();
    let child = start.play(&movement).unwrap();
    assert_ne!(
        start.state().snapshot().identity().owner,
        child.state().snapshot().identity().owner
    );
    assert_eq!(start.state().snapshot().side_to_move(), Color::White);
    assert_eq!(child.state().snapshot().side_to_move(), Color::Black);
    assert_eq!(child.state().rules().snapshot().known_history_len(), 2);
    assert_eq!(start.state().rules().snapshot().known_history_len(), 1);
    let mut session = Session::new(
        port,
        EngineIdentity {
            name: "actual Rules integration".into(),
            author: "test".into(),
        },
        Vec::new(),
        ParserLimits::default(),
    )
    .unwrap();
    assert!(
        session
            .handle_line("position startpos moves e2e4 e7e5")
            .accepted
    );
    let before = session.snapshot().state().snapshot().identity();
    let rejected = session.handle_line("position startpos moves d2d4 d7d5 e1e3");
    assert!(!rejected.accepted);
    assert_eq!(session.snapshot().state().snapshot().identity(), before);
}

#[test]
fn exact_terminal_and_unknown_history_have_distinct_uci_behavior() {
    let port = port();
    let from_fen = port
        .prepare(&PositionSpec {
            base: PositionBase::Fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into(),
            ),
            moves: Vec::new(),
        })
        .unwrap();
    assert!(!from_fen.exact_terminal);
    assert_eq!(
        from_fen
            .snapshot
            .state()
            .snapshot()
            .classification()
            .history,
        HistoryCompleteness::UnknownPrefix
    );
    assert_eq!(from_fen.legal_moves.len(), 20);
    for (fen, value) in [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", -1.0),
        ("8/8/8/8/8/6k1/6q1/7K w - - 0 1", -1.0),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.0),
        ("7k/8/8/8/8/8/8/KR6 w - - 150 80", 0.0),
    ] {
        let prepared = port
            .prepare(&PositionSpec {
                base: PositionBase::Fen(fen.into()),
                moves: Vec::new(),
            })
            .unwrap();
        assert!(prepared.exact_terminal);
        assert!(prepared.legal_moves.is_empty());
        assert_eq!(prepared.snapshot.classify().unwrap(), Some(value));
        assert!(matches!(
            prepared
                .snapshot
                .state()
                .snapshot()
                .classification()
                .play_status,
            PlayStatus::Terminal { .. }
        ));
    }
    let claim = port
        .prepare(&PositionSpec {
            base: PositionBase::Fen("7k/8/8/8/8/8/8/KR6 w - - 100 80".into()),
            moves: Vec::new(),
        })
        .unwrap();
    assert!(!claim.exact_terminal);
    assert_eq!(claim.snapshot.classify().unwrap(), None);
    assert!(!claim.legal_moves.is_empty());
    assert!(
        port.prepare(&PositionSpec {
            base: PositionBase::Fen("7k/8/8/8/8/8/8/KR6 w - - 150 80".into()),
            moves: vec!["b1b2".into()]
        })
        .is_err()
    );
}

#[test]
fn promotion_texts_and_rules_order_round_trip_without_reordering() {
    let prepared = port()
        .prepare(&PositionSpec {
            base: PositionBase::Fen("7k/P7/8/8/8/8/8/7K w - - 0 1".into()),
            moves: Vec::new(),
        })
        .unwrap();
    let promotions: Vec<_> = prepared
        .legal_moves
        .iter()
        .filter(|text| text.starts_with("a7a8"))
        .cloned()
        .collect();
    assert_eq!(promotions, ["a7a8q", "a7a8r", "a7a8b", "a7a8n"]);
    assert_eq!(
        prepared.legal_moves,
        prepared
            .snapshot
            .legal_moves()
            .unwrap()
            .into_iter()
            .map(|movement| rz_uci::engine::move_text(movement).unwrap())
            .collect::<Vec<_>>()
    );
    for promotion in promotions {
        let movement =
            rz_contracts::Move::try_from(BoardMove::from_uci(&promotion).unwrap()).unwrap();
        let next = prepared.snapshot.play(&movement).unwrap();
        assert_eq!(next.state().snapshot().side_to_move(), Color::Black);
        assert!(next.state().rules().snapshot().to_fen().contains(" b "));
    }
}
