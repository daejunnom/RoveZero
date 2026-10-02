use rz_position::{Position, PositionError, PositionLimits};

#[test]
fn explicit_input_history_and_perft_budgets_fail_without_mutation() {
    let limits = PositionLimits {
        max_fen_bytes: 8,
        ..PositionLimits::default()
    };
    assert!(matches!(
        Position::from_fen_with_limits(&Position::startpos().to_fen(), limits),
        Err(PositionError::ResourceLimit(_))
    ));
    assert!(matches!(
        Position::startpos_with_limits(PositionLimits {
            max_history_positions: 0,
            ..PositionLimits::default()
        }),
        Err(PositionError::ResourceLimit(_))
    ));
    assert!(matches!(
        Position::startpos_with_limits(PositionLimits {
            max_perft_depth: 100_000,
            ..PositionLimits::default()
        }),
        Err(PositionError::ResourceLimit(_))
    ));

    let mut p = Position::startpos_with_limits(PositionLimits {
        max_history_positions: 2,
        max_perft_depth: 2,
        max_perft_nodes: 20,
        ..PositionLimits::default()
    })
    .unwrap();
    let root = p.snapshot();
    // Twenty leaves still need the root node; total traversal budget is enforced.
    assert!(matches!(p.perft(1), Err(PositionError::ResourceLimit(_))));
    assert!(matches!(p.divide(1), Err(PositionError::ResourceLimit(_))));
    assert!(matches!(p.perft(3), Err(PositionError::ResourceLimit(_))));
    assert!(p.snapshot().same_state(&root));
    let undo = p.make_uci("e2e4").unwrap();
    let child = p.snapshot();
    assert!(matches!(
        p.make_uci("e7e5"),
        Err(PositionError::ResourceLimit(_))
    ));
    assert!(p.snapshot().same_state(&child));
    p.unmake(undo).unwrap();
    assert!(p.snapshot().same_state(&root));
}

#[test]
fn asserted_promotions_and_ep_predecessor_must_be_locally_possible() {
    for fen in [
        "7k/8/8/8/8/8/PPPPPPPP/KQQ5 w - - 0 1",
        "7k/8/8/8/8/8/PPPPPPPP/K1B1B3 w - - 0 1",
        "k3r3/8/8/3p4/8/8/8/4K3 w - d6 0 2",
        "k7/8/8/8/1b6/5n2/8/4K2r w - - 0 1",
    ] {
        assert!(
            matches!(Position::from_fen(fen), Err(PositionError::InvalidFen(_))),
            "{fen}"
        );
    }
    // A missing pawn permits an additional queen; an EP pawn can newly give check.
    Position::from_fen("7k/8/8/8/8/8/1PPPPPPP/KQQ5 w - - 0 1").unwrap();
    let p = Position::from_fen("4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 2").unwrap();
    assert!(p.in_check());
}

#[test]
fn persistent_history_releases_iteratively_and_preserves_shared_prefixes() {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let mut p = Position::startpos_with_limits(PositionLimits {
                max_history_positions: 10_000,
                ..PositionLimits::default()
            })
            .unwrap();
            let initial = p.snapshot();
            for index in 0..8_000 {
                p.make_uci(["g1f3", "g8f6", "f3g1", "f6g8"][index % 4])
                    .unwrap();
            }
            let history = p.snapshot();
            assert_eq!(history.known_history_len(), 8_001);
            assert!(
                format!("{history:?}").len() < 2_000,
                "debug formatting must summarize history"
            );
            drop(p);
            assert_eq!(history.known_history().count(), 8_001);
            drop(history);
            assert_eq!(initial.known_history_len(), 1);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn immutable_views_can_cross_threads_while_the_live_position_changes() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<rz_position::PositionSnapshot>();
    assert_send_sync::<rz_position::LegalMoveView>();
    let mut p = Position::startpos();
    let view = p.ordered_legal_moves();
    let expected = p.to_fen();
    let (ready, wait) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        wait.recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        (
            view.snapshot().to_fen(),
            view.moves().len(),
            view.snapshot().known_history_len(),
        )
    });
    p.apply_uci_moves(&["e2e4", "e7e5", "g1f3"]).unwrap();
    ready.send(()).unwrap();
    assert_eq!(worker.join().unwrap(), (expected, 20, 1));
}
