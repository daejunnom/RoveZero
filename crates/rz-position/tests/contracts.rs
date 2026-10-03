//! A03 consumer checks against the coordinator's exact revision 0.1 contract.
//!
//! Rules supply real immutable states and legal moves. Model/encoding manifests,
//! input keys and clock ticks below are explicit CPU mock fixtures; these checks
//! do not certify a model-specific C policy map or a Runtime implementation.
#![cfg(feature = "contracts")]

use std::sync::Arc;

use rz_contracts as shared;
use rz_position::{
    contracts::{ContractPosition, ContractState, RulesState},
    BoardMove, Position,
};

const CLOCK: shared::ClockDomain = shared::ClockDomain(shared::ProcessEpoch(81));

fn digest(tag: u8) -> shared::Digest {
    shared::Digest([tag; 32])
}

fn mock_model() -> Arc<shared::ModelDescriptor> {
    Arc::new(
        shared::ModelDescriptor::try_new(
            shared::ModelHandle {
                owner: shared::OwnerId(9000),
                slot: 1,
                generation: shared::SlotGeneration(1),
                manifest: digest(11),
            },
            shared::EncodingDescriptor {
                handle: shared::EncodingHandle {
                    owner: shared::OwnerId(9001),
                    slot: 1,
                    generation: shared::SlotGeneration(1),
                    manifest: digest(12),
                },
                history_length: 8,
                action_map: digest(13),
                history_policy: digest(14),
            },
            vec![shared::PrecisionProfile::Fp32],
            1,
            1,
        )
        .unwrap(),
    )
}

fn context(state: &ContractState, model: &shared::ModelDescriptor) -> shared::EvalContext {
    shared::EvalContext {
        revision: shared::CONTRACT_REVISION,
        request: shared::RequestId::new(CLOCK.0, 1),
        selection: shared::SelectionId::new(CLOCK.0, 1),
        game: shared::GameGeneration(2),
        root: shared::RootGeneration(3),
        state: state.snapshot().identity(),
        legal_order: state.legal_moves().order(),
        input: shared::EvalInputKey(digest(15)),
        model: model.handle(),
        encoding: model.encoding().handle,
        precision: shared::PrecisionProfile::Fp32,
        compute: shared::ComputeBudget {
            min_steps: 1,
            max_steps: 1,
            require_full: true,
        },
        backend: digest(16),
    }
}

fn build_request(
    state: &ContractState,
    legal: shared::LegalMoveView,
    context: shared::EvalContext,
    model: Arc<shared::ModelDescriptor>,
) -> Result<shared::EvalRequest<RulesState>, shared::ContractError> {
    shared::EvalRequest::try_new(
        context,
        state.snapshot().clone(),
        legal,
        model,
        shared::Deadline {
            clock: CLOCK,
            at: shared::MonotonicTick(100),
        },
        shared::CancelToken::new(),
        shared::ByteBudget {
            host: 4096,
            device: 0,
            pinned: 0,
        },
    )
}

fn request(
    state: &ContractState,
) -> Result<shared::EvalRequest<RulesState>, shared::ContractError> {
    let model = mock_model();
    build_request(
        state,
        state.legal_moves().clone(),
        context(state, &model),
        model,
    )
}

fn scope(request: &shared::EvalRequest<RulesState>) -> shared::AcceptanceScope {
    let context = request.context();
    shared::AcceptanceScope {
        game: context.game,
        root: context.root,
        model: context.model,
        encoding: context.encoding,
        backend: context.backend,
    }
}

fn output(request: &shared::EvalRequest<RulesState>) -> shared::EvalOutput {
    let count = request.legal().moves().len();
    shared::EvalOutput {
        context: request.context(),
        legal: request.legal().clone(),
        policy: shared::LegalPolicy::try_new(vec![1.0 / count as f64; count], 0.001).unwrap(),
        wdl: shared::Wdl::try_new(0.75, 0.125, 0.125, 0.0).unwrap(),
        viewpoint: shared::Viewpoint::SideToMove,
        actual: shared::ActualCompute {
            precision: shared::PrecisionProfile::Fp32,
            steps: 1,
            full: true,
            backend: request.context().backend,
            execution: Some(shared::ExecutionId::new(CLOCK.0, 1)),
            provenance: shared::CacheProvenance::Computed,
        },
    }
}

fn shared_move(uci: &str) -> shared::Move {
    shared::Move::try_from(BoardMove::from_uci(uci).unwrap()).unwrap()
}

#[test]
fn attested_forks_match_reexported_children_for_special_moves_claims_and_history() {
    let mut history = Position::startpos();
    history
        .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"])
        .unwrap();
    let mut cases = vec![Position::startpos(), history];
    for fen in [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "7k/8/8/8/8/8/8/KR6 w - - 99 80",
        "7k/8/8/8/8/8/8/KR6 w - - 149 80",
    ] {
        cases.push(Position::from_fen(fen).unwrap());
    }
    // This test's local registry also never reissues an owner to another fork.
    let mut sequence = 20_000_u64;
    let mut allocate = || {
        let owner = shared::OwnerId(sequence);
        sequence = sequence.checked_add(1).unwrap();
        owner
    };
    for (index, position) in cases.into_iter().enumerate() {
        let parent = ContractPosition::new(shared::OwnerId(10_000 + index as u64), position);
        let frozen = parent.export().unwrap();
        for &movement in frozen.legal_moves().moves() {
            // Independent consumer path retained as the previous API baseline:
            // clone, export its own view, apply, then export the child.
            let mut previous = ContractPosition::new(allocate(), parent.position().clone());
            let previous_view = previous.export().unwrap();
            previous.make_from_view(&previous_view, movement).unwrap();
            let expected = previous.export().unwrap();
            let child = parent
                .fork_from_view(allocate(), &frozen, movement)
                .unwrap();
            let actual = child.export().unwrap();
            assert_eq!(
                actual.snapshot().identity().semantic,
                expected.snapshot().identity().semantic
            );
            assert_eq!(
                actual.snapshot().identity().revision,
                expected.snapshot().identity().revision
            );
            assert_eq!(actual.legal_moves().order(), expected.legal_moves().order());
            assert_eq!(actual.legal_moves().moves(), expected.legal_moves().moves());
            assert_eq!(
                actual.rules().classification(),
                expected.rules().classification()
            );
            assert_eq!(
                actual.snapshot().classification(),
                expected.snapshot().classification()
            );
            assert_eq!(actual.terminal_wdl(), expected.terminal_wdl());
            assert!(actual
                .rules()
                .snapshot()
                .same_state(expected.rules().snapshot()));
            assert_ne!(
                actual.snapshot().identity().owner,
                frozen.snapshot().identity().owner
            );
            assert!(parent
                .position()
                .matches_snapshot(frozen.rules().snapshot()));
        }
    }
}

#[test]
fn forks_reject_foreign_stale_illegal_terminal_and_reused_owner_without_mutation() {
    let mut parent = ContractPosition::new(shared::OwnerId(40_000), Position::startpos());
    let frozen = parent.export().unwrap();
    let foreign = ContractPosition::new(shared::OwnerId(40_001), parent.position().clone())
        .export()
        .unwrap();
    for (view, movement, owner, code) in [
        (
            &foreign,
            shared_move("e2e4"),
            shared::OwnerId(40_002),
            shared::ErrorCode::Stale,
        ),
        (
            &frozen,
            shared_move("e2e5"),
            shared::OwnerId(40_002),
            shared::ErrorCode::InvalidInput,
        ),
        (
            &frozen,
            shared_move("e2e4"),
            shared::OwnerId(40_000),
            shared::ErrorCode::IdentityMismatch,
        ),
    ] {
        assert_eq!(
            parent
                .fork_from_view(owner, view, movement)
                .unwrap_err()
                .code,
            code
        );
        assert!(parent
            .position()
            .matches_snapshot(frozen.rules().snapshot()));
    }
    let undo = parent.make_from_view(&frozen, shared_move("e2e4")).unwrap();
    parent.unmake(undo).unwrap();
    let current = parent.position().snapshot();
    assert_eq!(
        parent
            .fork_from_view(shared::OwnerId(40_002), &frozen, shared_move("e2e4"))
            .unwrap_err()
            .code,
        shared::ErrorCode::Stale
    );
    assert!(parent.position().matches_snapshot(&current));
    let terminal = imported(40_003, "7k/8/8/8/8/8/8/KR6 w - - 150 80");
    let view = terminal.export().unwrap();
    assert!(!view.legal_moves().moves().is_empty());
    let movement = view.legal_moves().moves()[0];
    assert_eq!(
        terminal
            .fork_from_view(shared::OwnerId(40_004), &view, movement)
            .unwrap_err()
            .code,
        shared::ErrorCode::InvalidInput
    );
    assert!(terminal
        .position()
        .matches_snapshot(view.rules().snapshot()));
}

fn play(position: &mut ContractPosition, uci: &str) -> rz_position::UndoToken {
    let state = position.export().unwrap();
    position.make_from_view(&state, shared_move(uci)).unwrap()
}

fn imported(owner: u64, fen: &str) -> ContractPosition {
    ContractPosition::new(shared::OwnerId(owner), Position::from_fen(fen).unwrap())
}

fn assert_digest(actual: shared::Digest, expected_hex: &str) {
    let actual_hex = actual
        .0
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(actual_hex, expected_hex);
}

#[test]
fn canonical_profile_state_and_legal_order_match_independent_sha256_goldens() {
    // Fixed with Python hashlib/struct.pack("<Q", n), from literal profile/FEN
    // strings and python-chess 1.11.2 moves sorted by (from,to,None/Q/R/B/N).
    // No Rust digest or Rules history traversal generated these expected bytes.
    assert_eq!(
        rz_position::contracts::CONTRACT_SOURCE_REVISION,
        "67284c4f66f7a7ae9f46fa63dfd50e7410eb6845"
    );
    assert_digest(
        rz_position::contracts::profile_digest(),
        "f1ad641bd87cd12c8c220fa9de05295d261ed7009ef97dfdb002b33c828eafc2",
    );
    let mut position = ContractPosition::new(shared::OwnerId(90), Position::startpos());
    let initial = position.export().unwrap();
    assert_digest(
        initial.snapshot().identity().semantic,
        "4bd4c152eb1b51ca21d2608e07b79f977ce6b9d8c45dde15ca947a0ff6bacf25",
    );
    assert_digest(
        initial.legal_moves().order().0,
        "da9f6000f36d381db53a4a563e6d3e3c53c8636407d3776df1a1d4c9a8a7b923",
    );
    assert_eq!(
        initial.snapshot().classification().rules_profile,
        rz_position::contracts::profile_digest()
    );

    for mv in ["e2e4", "e7e5", "g1f3"] {
        play(&mut position, mv);
    }
    let trace = position.export().unwrap();
    let expected_history = [
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ];
    assert_eq!(
        trace.rules().snapshot().known_history_fens(),
        expected_history
    );
    assert_digest(
        trace.snapshot().identity().semantic,
        "f260a8719e4eb40f48a2c194e902c36e77a2c517ce73868ca19558d86b16a364",
    );
    assert_digest(
        trace.legal_moves().order().0,
        "57eddeabbc0a23587f485f1671bc2472e7a838f92dbd8257601e75674cb49d28",
    );

    let promotion = imported(91, "1r5k/P7/8/8/8/8/8/7K w - - 0 1")
        .export()
        .unwrap();
    assert_digest(
        promotion.snapshot().identity().semantic,
        "2bd9c6aa412a59ccd3689508a8d45e9074ab4b96b9887dce17a69d880ef14cf2",
    );
    assert_digest(
        promotion.legal_moves().order().0,
        "f74aaf8aabf6312a79d3db34679d14f879812e3564d41c76fbff42de012f170f",
    );
}

#[test]
fn actual_snapshot_and_legal_order_are_owned_across_make_unmake_and_drop() {
    let mut position = ContractPosition::new(shared::OwnerId(100), Position::startpos());
    let state = position.export().unwrap();
    let request = request(&state).unwrap();
    let initial_fen = request.position().state().snapshot().to_fen();
    let initial_history = request.position().state().snapshot().known_history_fens();
    let initial_context = request.context();
    let initial_moves = request.legal().moves().to_vec();
    let undo = position
        .make_from_view(&state, shared_move("e2e4"))
        .unwrap();
    assert_ne!(position.position().to_fen(), initial_fen);
    assert_eq!(request.position().state().snapshot().to_fen(), initial_fen);
    assert_eq!(request.context(), initial_context);
    assert_eq!(request.legal().moves(), initial_moves);

    position.unmake(undo).unwrap();
    let restored = position.export().unwrap();
    assert_eq!(
        restored.snapshot().identity().semantic,
        initial_context.state.semantic
    );
    assert_ne!(restored.snapshot().identity(), initial_context.state);
    assert!(restored.snapshot().identity().revision > initial_context.state.revision);
    let fresh_request = self::request(&restored).unwrap();
    assert_eq!(
        output(&request)
            .validate_for(
                &fresh_request,
                scope(&fresh_request),
                CLOCK,
                shared::MonotonicTick(99)
            )
            .unwrap_err()
            .code,
        shared::ErrorCode::IdentityMismatch
    );
    let before = position.position().snapshot();
    assert_eq!(
        position
            .make_from_view(&state, shared_move("e2e4"))
            .unwrap_err()
            .code,
        shared::ErrorCode::Stale
    );
    assert!(position.position().snapshot().same_state(&before));
    drop(position);
    drop(state);
    let result = output(&request);
    std::thread::spawn(move || {
        assert_eq!(request.position().state().snapshot().to_fen(), initial_fen);
        assert_eq!(
            request.position().state().snapshot().known_history_fens(),
            initial_history
        );
        result
            .validate_for(&request, scope(&request), CLOCK, shared::MonotonicTick(99))
            .unwrap();
    })
    .join()
    .unwrap();
}

#[test]
fn startpos_order_and_all_promotions_cross_the_shared_move_boundary_exactly() {
    let position = ContractPosition::new(shared::OwnerId(110), Position::startpos());
    let state = position.export().unwrap();
    let expected = [
        "b1a3", "b1c3", "g1f3", "g1h3", "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3",
        "d2d4", "e2e3", "e2e4", "f2f3", "f2f4", "g2g3", "g2g4", "h2h3", "h2h4",
    ];
    let expected = expected.map(shared_move);
    assert_eq!(state.legal_moves().moves(), expected);
    assert_eq!(state.legal_moves().state(), state.snapshot().identity());
    let second = position.export().unwrap();
    assert_eq!(second.legal_moves(), state.legal_moves());

    let promoted = imported(111, "1r5k/P7/8/8/8/8/8/7K w - - 0 1")
        .export()
        .unwrap();
    let promotions = promoted
        .legal_moves()
        .moves()
        .iter()
        .copied()
        .filter(|movement| movement.from.index() == 48)
        .collect::<Vec<_>>();
    assert_eq!(
        promotions,
        ["a7a8q", "a7a8r", "a7a8b", "a7a8n", "a7b8q", "a7b8r", "a7b8b", "a7b8n"].map(shared_move)
    );
    assert_eq!(
        promotions
            .iter()
            .map(|movement| movement.promotion.unwrap())
            .collect::<Vec<_>>(),
        vec![
            shared::Promotion::Queen,
            shared::Promotion::Rook,
            shared::Promotion::Bishop,
            shared::Promotion::Knight,
            shared::Promotion::Queen,
            shared::Promotion::Rook,
            shared::Promotion::Bishop,
            shared::Promotion::Knight
        ]
    );
    for movement in promotions {
        let local = BoardMove::try_from(movement).unwrap();
        assert_eq!(shared::Move::try_from(local).unwrap(), movement);
    }
    assert!(request(&promoted).is_ok());
}

#[test]
fn semantic_digest_distinguishes_raw_globals_origin_and_complete_history() {
    let fen = Position::startpos().to_fen();
    let variants = [
        fen.clone(),
        fen.replace(" w ", " b "),
        fen.replace(" KQkq ", " - "),
        fen.replace(" 0 1", " 1 1"),
        fen.replace(" 0 1", " 0 2"),
    ];
    let identities = variants
        .iter()
        .enumerate()
        .map(|(index, fen)| {
            imported(120 + index as u64, fen)
                .export()
                .unwrap()
                .snapshot()
                .identity()
                .semantic
        })
        .collect::<Vec<_>>();
    for i in 0..identities.len() {
        for j in i + 1..identities.len() {
            assert_ne!(identities[i], identities[j], "globals variants {i}/{j}");
        }
    }
    let raw_ep = imported(125, "4k3/8/8/4p3/8/8/8/4K3 w - e6 0 2")
        .export()
        .unwrap();
    let no_ep = imported(126, "4k3/8/8/4p3/8/8/8/4K3 w - - 0 2")
        .export()
        .unwrap();
    assert_eq!(
        raw_ep.rules().snapshot().repetition_identity(),
        no_ep.rules().snapshot().repetition_identity()
    );
    assert_ne!(
        raw_ep.snapshot().identity().semantic,
        no_ep.snapshot().identity().semantic
    );

    let complete = ContractPosition::new(shared::OwnerId(127), Position::startpos())
        .export()
        .unwrap();
    let unknown = imported(128, &fen).export().unwrap();
    assert_eq!(
        complete.rules().snapshot().to_fen(),
        unknown.rules().snapshot().to_fen()
    );
    assert_eq!(
        complete.snapshot().classification().history,
        shared::HistoryCompleteness::Complete
    );
    assert_eq!(
        unknown.snapshot().classification().history,
        shared::HistoryCompleteness::UnknownPrefix
    );
    assert_ne!(
        complete.snapshot().identity().semantic,
        unknown.snapshot().identity().semantic
    );

    let mut first = ContractPosition::new(shared::OwnerId(129), Position::startpos());
    let mut second = ContractPosition::new(shared::OwnerId(130), Position::startpos());
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        play(&mut first, mv);
    }
    for mv in ["b1c3", "b8c6", "c3b1", "c6b8"] {
        play(&mut second, mv);
    }
    let a = first.export().unwrap();
    let b = second.export().unwrap();
    assert_eq!(a.rules().snapshot().to_fen(), b.rules().snapshot().to_fen());
    assert_eq!(
        a.rules().snapshot().known_history_len(),
        b.rules().snapshot().known_history_len()
    );
    assert_ne!(
        a.snapshot().identity().semantic,
        b.snapshot().identity().semantic
    );
}

#[test]
fn mismatched_state_and_order_are_rejected_before_evaluation_or_move_mutation() {
    let first = ContractPosition::new(shared::OwnerId(140), Position::startpos());
    let mut second = ContractPosition::new(shared::OwnerId(141), Position::startpos());
    let a = first.export().unwrap();
    let b = second.export().unwrap();
    assert_eq!(
        a.snapshot().identity().semantic,
        b.snapshot().identity().semantic
    );
    assert_ne!(a.snapshot().identity().owner, b.snapshot().identity().owner);
    let model = mock_model();
    assert_eq!(
        build_request(
            &a,
            b.legal_moves().clone(),
            context(&a, &model),
            model.clone()
        )
        .unwrap_err()
        .code,
        shared::ErrorCode::IdentityMismatch
    );
    let before = second.position().snapshot();
    assert_eq!(
        second
            .make_from_view(&a, shared_move("e2e4"))
            .unwrap_err()
            .code,
        shared::ErrorCode::Stale
    );
    assert!(second.position().snapshot().same_state(&before));

    let mut wrong_context = context(&a, &model);
    wrong_context.state.semantic = digest(200);
    assert_eq!(
        build_request(&a, a.legal_moves().clone(), wrong_context, model.clone())
            .unwrap_err()
            .code,
        shared::ErrorCode::IdentityMismatch
    );
    wrong_context = context(&a, &model);
    wrong_context.legal_order = shared::LegalOrderIdentity(digest(201));
    assert_eq!(
        build_request(&a, a.legal_moves().clone(), wrong_context, model.clone())
            .unwrap_err()
            .code,
        shared::ErrorCode::IdentityMismatch
    );
    // Request and selection agree with each other, but the deadline clock is
    // still CLOCK. Cross-process ticks must never enter a valid request.
    wrong_context = context(&a, &model);
    wrong_context.request.epoch = shared::ProcessEpoch(82);
    wrong_context.selection.epoch = shared::ProcessEpoch(82);
    let error = build_request(&a, a.legal_moves().clone(), wrong_context, model).unwrap_err();
    assert_eq!(
        (error.code, error.stage),
        (
            shared::ErrorCode::IdentityMismatch,
            shared::Stage::Admission
        )
    );

    let request = request(&a).unwrap();
    let mut changed = output(&request);
    let mut moves = changed.legal.moves().to_vec();
    moves.swap(0, 1);
    changed.legal =
        shared::LegalMoveView::try_new(changed.legal.state(), changed.legal.order(), moves, 256)
            .unwrap();
    assert_eq!(
        changed
            .validate_for(&request, scope(&request), CLOCK, shared::MonotonicTick(99))
            .unwrap_err()
            .code,
        shared::ErrorCode::IdentityMismatch
    );
    changed = output(&request);
    changed.context.state.revision = shared::StateRevision(changed.context.state.revision.0 + 1);
    assert_eq!(
        changed
            .validate_for(&request, scope(&request), CLOCK, shared::MonotonicTick(99))
            .unwrap_err()
            .code,
        shared::ErrorCode::IdentityMismatch
    );
}

#[test]
fn fifty_move_and_threefold_claims_stay_ongoing_and_can_request_evaluation() {
    for (owner, clock) in [(150, 99), (151, 100)] {
        let fen = format!("4k3/8/8/8/8/8/8/R3K3 w - - {clock} 1");
        let state = imported(owner, &fen).export().unwrap();
        let classification = state.snapshot().classification();
        assert_eq!(classification.play_status, shared::PlayStatus::Ongoing);
        assert!(classification.claims.iter().any(|claim| {
            claim.rule == shared::ClaimRule::FiftyMove
                && claim.availability == shared::ClaimAvailability::Available
                && matches!(
                    (clock, claim.source),
                    (99, shared::ClaimSource::IntendedMove(_))
                        | (100, shared::ClaimSource::CurrentPosition)
                )
        }));
        assert!(classification
            .claims
            .iter()
            .any(|claim| claim.rule == shared::ClaimRule::Threefold
                && claim.availability == shared::ClaimAvailability::Unknown));
        assert!(state.terminal_wdl().is_none());
        assert!(request(&state).is_ok());
    }
    let mut position = ContractPosition::new(shared::OwnerId(152), Position::startpos());
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"] {
        play(&mut position, mv);
    }
    let intended = position.export().unwrap();
    assert_eq!(
        intended.snapshot().classification().play_status,
        shared::PlayStatus::Ongoing
    );
    assert!(intended
        .snapshot()
        .classification()
        .claims
        .iter()
        .any(|claim| {
            claim.rule == shared::ClaimRule::Threefold
                && claim.source == shared::ClaimSource::IntendedMove(shared_move("f6g8"))
                && claim.availability == shared::ClaimAvailability::Available
        }));
    assert!(request(&intended).is_ok());
    play(&mut position, "f6g8");
    let state = position.export().unwrap();
    assert_eq!(
        state.snapshot().classification().play_status,
        shared::PlayStatus::Ongoing
    );
    assert!(state
        .snapshot()
        .classification()
        .claims
        .iter()
        .any(|claim| {
            claim.rule == shared::ClaimRule::Threefold
                && claim.source == shared::ClaimSource::CurrentPosition
                && claim.availability == shared::ClaimAvailability::Available
        }));
    assert!(request(&state).is_ok());
}

#[test]
fn exact_terminals_bypass_eval_even_when_board_legal_moves_exist() {
    let cases = [
        (
            "7k/6Q1/5K2/8/8/8/8/8 b - - 150 1",
            shared::TerminalReason::Checkmate,
            Some(shared::Color::White),
            [0.0, 0.0, 1.0],
        ),
        (
            "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
            shared::TerminalReason::Stalemate,
            None,
            [0.0, 1.0, 0.0],
        ),
        (
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            shared::TerminalReason::DeadPosition,
            None,
            [0.0, 1.0, 0.0],
        ),
        (
            "4k3/8/8/8/8/8/8/R3K3 w - - 150 1",
            shared::TerminalReason::SeventyFiveMove,
            None,
            [0.0, 1.0, 0.0],
        ),
    ];
    for (index, (fen, reason, winner, probabilities)) in cases.into_iter().enumerate() {
        let state = imported(160 + index as u64, fen).export().unwrap();
        assert_eq!(
            state.snapshot().classification().play_status,
            shared::PlayStatus::Terminal { reason, winner }
        );
        assert_eq!(state.terminal_wdl().unwrap().probabilities(), probabilities);
        if matches!(
            reason,
            shared::TerminalReason::DeadPosition | shared::TerminalReason::SeventyFiveMove
        ) {
            assert!(!state.legal_moves().moves().is_empty());
        }
        assert_eq!(
            request(&state).unwrap_err().code,
            shared::ErrorCode::InvalidInput
        );
    }
    let mut position = ContractPosition::new(shared::OwnerId(164), Position::startpos());
    for _ in 0..4 {
        for mv in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            play(&mut position, mv);
        }
    }
    let state = position.export().unwrap();
    assert_eq!(
        state.snapshot().classification().play_status,
        shared::PlayStatus::Terminal {
            reason: shared::TerminalReason::FivefoldRepetition,
            winner: None
        }
    );
    assert!(!state.legal_moves().moves().is_empty());
    assert_eq!(
        state.terminal_wdl().unwrap().probabilities(),
        [0.0, 1.0, 0.0]
    );
    assert_eq!(
        request(&state).unwrap_err().code,
        shared::ErrorCode::InvalidInput
    );
}

#[test]
fn black_viewpoint_stale_generations_deadline_and_cancel_are_checked_at_output() {
    let mut position = ContractPosition::new(shared::OwnerId(170), Position::startpos());
    play(&mut position, "e2e4");
    let state = position.export().unwrap();
    let request = request(&state).unwrap();
    let result = output(&request);
    assert_eq!(request.position().side_to_move(), shared::Color::Black);
    assert_eq!(result.viewpoint, shared::Viewpoint::SideToMove);
    assert_eq!(result.wdl.probabilities(), [0.75, 0.125, 0.125]);
    result
        .validate_for(&request, scope(&request), CLOCK, shared::MonotonicTick(99))
        .unwrap();

    let mut foreign_execution = output(&request);
    foreign_execution.actual.execution =
        Some(shared::ExecutionId::new(shared::ProcessEpoch(82), 1));
    let error = foreign_execution
        .validate_for(&request, scope(&request), CLOCK, shared::MonotonicTick(99))
        .unwrap_err();
    assert_eq!(
        (error.code, error.stage),
        (shared::ErrorCode::IdentityMismatch, shared::Stage::Output)
    );

    for kind in 0..5 {
        let mut stale = scope(&request);
        match kind {
            0 => stale.game.0 += 1,
            1 => stale.root.0 += 1,
            2 => stale.model.generation.0 += 1,
            3 => stale.encoding.generation.0 += 1,
            _ => stale.backend = digest(220),
        }
        assert_eq!(
            result
                .validate_for(&request, stale, CLOCK, shared::MonotonicTick(99))
                .unwrap_err()
                .code,
            shared::ErrorCode::Stale
        );
    }
    for now in [100, 101] {
        assert_eq!(
            result
                .validate_for(&request, scope(&request), CLOCK, shared::MonotonicTick(now))
                .unwrap_err()
                .code,
            shared::ErrorCode::Expired
        );
    }
    assert_eq!(
        result
            .validate_for(
                &request,
                scope(&request),
                shared::ClockDomain(shared::ProcessEpoch(82)),
                shared::MonotonicTick(99)
            )
            .unwrap_err()
            .code,
        shared::ErrorCode::InvalidInput
    );
    request.cancel_token().cancel();
    assert_eq!(
        result
            .validate_for(&request, scope(&request), CLOCK, shared::MonotonicTick(99))
            .unwrap_err()
            .code,
        shared::ErrorCode::Canceled
    );
}

#[test]
fn operational_and_malformed_input_failures_preserve_owned_state() {
    let limits = rz_position::PositionLimits {
        max_history_positions: 1,
        ..rz_position::PositionLimits::default()
    };
    let limited = ContractPosition::new(
        shared::OwnerId(180),
        Position::startpos_with_limits(limits).unwrap(),
    );
    let overflow = imported(181, "4k2r/8/8/8/8/8/8/4K2R b - - 0 4294967295");
    for position in [&limited, &overflow] {
        // These ongoing boards require an intended-move preview. An exhausted
        // history/counter must be an explicit failure rather than draw/unknown.
        assert!(!position.position().legal_moves().is_empty());
        let before = position.position().snapshot();
        let error = position.export().unwrap_err();
        assert_eq!(
            (error.code, error.stage),
            (
                shared::ErrorCode::ResourceExhausted,
                shared::Stage::Admission
            )
        );
        assert_eq!(position.position().revision(), before.revision());
        assert!(position.position().snapshot().same_state(&before));
        assert!(position.position().matches_snapshot(&before));
    }

    let mut position = ContractPosition::new(shared::OwnerId(182), Position::startpos());
    let state = position.export().unwrap();
    let before = position.position().snapshot();
    let same_square = shared::Square::try_new(12).unwrap();
    // Public shared fields can bypass Move::new; the Rules boundary must check.
    let malformed = shared::Move {
        from: same_square,
        to: same_square,
        promotion: None,
    };
    let error = position.make_from_view(&state, malformed).unwrap_err();
    assert_eq!(
        (error.code, error.stage),
        (shared::ErrorCode::InvalidInput, shared::Stage::Admission)
    );
    assert!(position.position().matches_snapshot(&before));
    assert!(position.position().snapshot().same_state(&before));
    assert_eq!(
        position.export().unwrap().snapshot().identity(),
        state.snapshot().identity()
    );

    for promotion in [rz_position::PieceKind::Pawn, rz_position::PieceKind::King] {
        // Local public fields likewise do not attest a valid promotion kind.
        let malformed = BoardMove {
            from: rz_position::Square::new(48).unwrap(),
            to: rz_position::Square::new(56).unwrap(),
            promotion: Some(promotion),
        };
        let error = shared::Move::try_from(malformed).unwrap_err();
        assert_eq!(
            (error.code, error.stage),
            (shared::ErrorCode::InvalidInput, shared::Stage::Admission)
        );
        assert!(position.position().matches_snapshot(&before));
        assert!(position.position().snapshot().same_state(&before));
    }
}

#[test]
fn shared_castling_and_en_passant_moves_have_exact_deltas_and_restore() {
    use rz_position::{Color, Piece, PieceChange, PieceKind, RuleMoveKind, Square};
    use Color::{Black, White};
    use PieceKind::{King, Pawn, Rook};

    struct Fixture {
        fen: &'static str,
        uci: &'static str,
        child_fen: &'static str,
        castling: bool,
        removals: &'static [(u8, Color, PieceKind)],
        additions: &'static [(u8, Color, PieceKind)],
    }
    fn changes(entries: &[(u8, Color, PieceKind)]) -> Vec<PieceChange> {
        entries
            .iter()
            .map(|&(square, color, kind)| PieceChange {
                square: Square::new(square).unwrap(),
                piece: Piece { color, kind },
            })
            .collect()
    }
    // Literal independent successor FENs and rank-major square changes. These
    // expected deltas do not come from applying/reversing another Rules move.
    let cases = [
        Fixture {
            fen: "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            uci: "e1g1",
            child_fen: "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1",
            castling: true,
            removals: &[(4, White, King), (7, White, Rook)],
            additions: &[(5, White, Rook), (6, White, King)],
        },
        Fixture {
            fen: "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            uci: "e8c8",
            child_fen: "2kr3r/8/8/8/8/8/8/R3K2R w KQ - 1 2",
            castling: true,
            removals: &[(56, Black, Rook), (60, Black, King)],
            additions: &[(58, Black, King), (59, Black, Rook)],
        },
        Fixture {
            fen: "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
            uci: "e5d6",
            child_fen: "4k3/8/3P4/8/8/8/8/4K3 b - - 0 2",
            castling: false,
            removals: &[(35, Black, Pawn), (36, White, Pawn)],
            additions: &[(43, White, Pawn)],
        },
        Fixture {
            fen: "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2",
            uci: "e4d3",
            child_fen: "4k3/8/8/8/8/3p4/8/4K3 w - - 0 3",
            castling: false,
            removals: &[(27, White, Pawn), (28, Black, Pawn)],
            additions: &[(19, Black, Pawn)],
        },
    ];
    for (index, fixture) in cases.into_iter().enumerate() {
        let mut position = imported(190 + index as u64, fixture.fen);
        let before = position.export().unwrap();
        let frozen_request = request(&before).unwrap();
        let mv = shared_move(fixture.uci);
        assert!(before.legal_moves().moves().contains(&mv));
        let token = position.make_from_view(&before, mv).unwrap();
        let delta = token.delta();
        assert_eq!(
            position.position().to_fen(),
            fixture.child_fen,
            "{}",
            fixture.uci
        );
        assert_eq!(delta.mv.to_string(), fixture.uci);
        assert_eq!(
            delta.kind,
            RuleMoveKind {
                capture: !fixture.castling,
                castling: fixture.castling,
                en_passant: !fixture.castling,
                promotion: false,
            }
        );
        assert_eq!(
            delta.removals,
            changes(fixture.removals),
            "{} removals",
            fixture.uci
        );
        assert_eq!(
            delta.additions,
            changes(fixture.additions),
            "{} additions",
            fixture.uci
        );
        assert!(delta.before.same_state(before.rules().snapshot()));
        assert_eq!(delta.after.to_fen(), fixture.child_fen);
        let child = position.export().unwrap();
        assert!(delta.after.same_state(child.rules().snapshot()));
        assert_eq!(
            child.snapshot().identity().owner,
            before.snapshot().identity().owner
        );
        assert_ne!(
            child.snapshot().identity().semantic,
            before.snapshot().identity().semantic
        );
        assert_eq!(
            frozen_request.position().state().snapshot().to_fen(),
            fixture.fen
        );
        output(&frozen_request)
            .validate_for(
                &frozen_request,
                scope(&frozen_request),
                CLOCK,
                shared::MonotonicTick(99),
            )
            .unwrap();
        position.unmake(token).unwrap();
        let restored = position.export().unwrap();
        assert_eq!(position.position().to_fen(), fixture.fen);
        assert!(restored
            .rules()
            .snapshot()
            .same_state(before.rules().snapshot()));
        assert_eq!(
            restored.snapshot().identity().semantic,
            before.snapshot().identity().semantic
        );
        assert_eq!(
            restored.snapshot().identity().owner,
            before.snapshot().identity().owner
        );
        assert!(restored.snapshot().identity().revision > child.snapshot().identity().revision);
        assert_eq!(restored.legal_moves().moves(), before.legal_moves().moves());
    }
}
