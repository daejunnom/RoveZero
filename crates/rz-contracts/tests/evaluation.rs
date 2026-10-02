//! Synthetic boundary fixtures: no chess legality, model inference, runtime
//! finalization, or physical backend lease is established by these tests.
use std::sync::Arc;

use rz_contracts::*;

fn digest(marker: u8) -> Digest {
    Digest([marker; 32])
}

fn square(index: u8) -> Square {
    Square::try_new(index).unwrap()
}

fn movements() -> Vec<Move> {
    vec![
        Move::new(square(12), square(28), None).unwrap(),
        Move::new(square(11), square(27), None).unwrap(),
    ]
}

fn classification() -> PositionClassification {
    PositionClassification {
        play_status: PlayStatus::Ongoing,
        claims: Vec::new().into(),
        history: HistoryCompleteness::Complete,
        rules_profile: digest(1),
    }
}

fn request() -> EvalRequest<()> {
    request_with_state(Arc::new(()), classification())
}

fn request_with_state<P>(state: Arc<P>, classification: PositionClassification) -> EvalRequest<P> {
    let epoch = ProcessEpoch(7);
    let identity = StateIdentity {
        owner: OwnerId(10),
        revision: StateRevision(4),
        semantic: digest(2),
    };
    let order = LegalOrderIdentity(digest(3));
    let model_handle = ModelHandle {
        owner: OwnerId(20),
        slot: 1,
        generation: SlotGeneration(2),
        manifest: digest(4),
    };
    let encoding_handle = EncodingHandle {
        owner: OwnerId(30),
        slot: 1,
        generation: SlotGeneration(3),
        manifest: digest(5),
    };
    let model = Arc::new(
        ModelDescriptor::try_new(
            model_handle,
            EncodingDescriptor {
                handle: encoding_handle,
                history_length: 8,
                action_map: digest(6),
                history_policy: digest(7),
            },
            vec![PrecisionProfile::Fp32],
            1,
            16,
        )
        .unwrap(),
    );
    let context = EvalContext {
        revision: CONTRACT_REVISION,
        request: RequestId::new(epoch, 100),
        selection: SelectionId::new(epoch, 200),
        game: GameGeneration(8),
        root: RootGeneration(9),
        state: identity,
        legal_order: order,
        input: EvalInputKey(digest(8)),
        model: model_handle,
        encoding: encoding_handle,
        precision: PrecisionProfile::Fp32,
        compute: ComputeBudget {
            min_steps: 1,
            max_steps: 1,
            require_full: true,
        },
        backend: digest(9),
    };
    EvalRequest::try_new(
        context,
        PositionSnapshot::try_new(identity, state, Color::White, classification).unwrap(),
        LegalMoveView::try_new(identity, order, movements(), 256).unwrap(),
        model,
        Deadline {
            clock: ClockDomain(epoch),
            at: MonotonicTick(1000),
        },
        CancelToken::new(),
        ByteBudget {
            host: 4096,
            device: 0,
            pinned: 0,
        },
    )
    .unwrap()
}

fn scope<P>(request: &EvalRequest<P>) -> AcceptanceScope {
    let context = request.context();
    AcceptanceScope {
        game: context.game,
        root: context.root,
        model: context.model,
        encoding: context.encoding,
        backend: context.backend,
    }
}

fn output<P>(request: &EvalRequest<P>) -> EvalOutput {
    let context = request.context();
    EvalOutput {
        context,
        legal: request.legal().clone(),
        policy: LegalPolicy::try_new(vec![0.25, 0.75], 1e-9).unwrap(),
        wdl: Wdl::try_new(0.5, 0.25, 0.25, 0.0).unwrap(),
        viewpoint: Viewpoint::SideToMove,
        actual: ActualCompute {
            precision: context.precision,
            steps: 1,
            full: true,
            backend: context.backend,
            execution: Some(ExecutionId::new(context.request.epoch, 300)),
            provenance: CacheProvenance::Computed,
        },
    }
}

fn validate<P>(output: &EvalOutput, request: &EvalRequest<P>) -> Result<(), ContractError> {
    output.validate_for(
        request,
        scope(request),
        request.deadline().clock,
        MonotonicTick(999),
    )
}

fn rebuild<P>(
    template: &EvalRequest<P>,
    context: EvalContext,
    position: PositionSnapshot<P>,
    legal: LegalMoveView,
) -> Result<EvalRequest<P>, ContractError> {
    EvalRequest::try_new(
        context,
        position,
        legal,
        Arc::clone(template.model()),
        template.deadline(),
        template.cancel_token().clone(),
        template.byte_budget(),
    )
}

#[test]
fn independent_reference_preserves_order_and_side_to_move_wdl() {
    let request = request();
    let evaluated = output(&request);
    assert!(validate(&evaluated, &request).is_ok());
    assert_eq!(evaluated.legal.moves(), movements());
    assert_eq!(evaluated.policy.probabilities(), [0.25, 0.75]);
    assert_eq!(evaluated.wdl.probabilities(), [0.5, 0.25, 0.25]);
    assert_eq!(evaluated.wdl.value(), 0.25);
    assert_eq!(request.position().side_to_move(), Color::White);
}

#[test]
fn response_must_echo_every_context_identity_and_configuration() {
    let request = request();
    let original = request.context();
    let mut variants = Vec::new();
    macro_rules! changed {
        ($label:literal, $($field:ident).+, $value:expr) => {{
            let mut context = original;
            context.$($field).+ = $value;
            variants.push(($label, context));
        }};
    }
    changed!("revision", revision.minor, 2);
    changed!("request sequence", request.sequence, 101);
    changed!("request epoch", request.epoch, ProcessEpoch(8));
    changed!("selection sequence", selection.sequence, 201);
    changed!("selection epoch", selection.epoch, ProcessEpoch(8));
    changed!("game", game, GameGeneration(10));
    changed!("root", root, RootGeneration(11));
    changed!("state owner", state.owner, OwnerId(11));
    changed!("state revision", state.revision, StateRevision(5));
    changed!("state digest", state.semantic, digest(11));
    changed!("legal order", legal_order, LegalOrderIdentity(digest(12)));
    changed!("actual input", input, EvalInputKey(digest(13)));
    changed!("model owner", model.owner, OwnerId(21));
    changed!("model slot", model.slot, 2);
    changed!("model generation", model.generation, SlotGeneration(4));
    changed!("model manifest", model.manifest, digest(14));
    changed!("encoding owner", encoding.owner, OwnerId(31));
    changed!("encoding slot", encoding.slot, 2);
    changed!(
        "encoding generation",
        encoding.generation,
        SlotGeneration(4)
    );
    changed!("encoding manifest", encoding.manifest, digest(15));
    changed!("precision", precision, PrecisionProfile::Fp16);
    changed!("compute minimum", compute.min_steps, 2);
    changed!("compute maximum", compute.max_steps, 2);
    changed!("require full", compute.require_full, false);
    changed!("backend", backend, digest(16));
    for (label, context) in variants {
        let mut evaluated = output(&request);
        evaluated.context = context;
        let error = validate(&evaluated, &request).unwrap_err();
        assert_eq!(error.code, ErrorCode::IdentityMismatch, "{label}");
        assert_eq!(error.stage, Stage::Output, "{label}");
    }
}

#[test]
fn matching_order_digest_does_not_hide_different_actual_move_payload() {
    let request = request();
    let mut evaluated = output(&request);
    let mut reversed = movements();
    reversed.reverse();
    evaluated.legal = LegalMoveView::try_new(
        request.context().state,
        request.context().legal_order,
        reversed,
        256,
    )
    .unwrap();
    assert_eq!(
        validate(&evaluated, &request).unwrap_err().code,
        ErrorCode::IdentityMismatch,
    );
    let mut changed = movements();
    changed[0] = Move::new(square(6), square(21), None).unwrap();
    evaluated.legal = LegalMoveView::try_new(
        request.context().state,
        request.context().legal_order,
        changed,
        256,
    )
    .unwrap();
    assert!(validate(&evaluated, &request).is_err());
    evaluated.legal = request.legal().clone();
    evaluated.policy = LegalPolicy::try_new(vec![1.0], 0.0).unwrap();
    assert_eq!(
        validate(&evaluated, &request).unwrap_err().code,
        ErrorCode::IdentityMismatch,
    );
}

#[test]
fn admission_requires_the_same_state_model_encoding_and_legal_identity() {
    let request = request();
    let original = request.context();
    for context in [
        EvalContext {
            state: StateIdentity {
                revision: StateRevision(99),
                ..original.state
            },
            ..original
        },
        EvalContext {
            legal_order: LegalOrderIdentity(digest(99)),
            ..original
        },
        EvalContext {
            model: ModelHandle {
                generation: SlotGeneration(99),
                ..original.model
            },
            ..original
        },
        EvalContext {
            encoding: EncodingHandle {
                generation: SlotGeneration(99),
                ..original.encoding
            },
            ..original
        },
        EvalContext {
            selection: SelectionId::new(ProcessEpoch(99), 200),
            ..original
        },
        EvalContext {
            request: RequestId::new(ProcessEpoch(99), 100),
            selection: SelectionId::new(ProcessEpoch(99), 200),
            ..original
        },
    ] {
        assert_eq!(
            rebuild(
                &request,
                context,
                request.position().clone(),
                request.legal().clone()
            )
            .unwrap_err()
            .code,
            ErrorCode::IdentityMismatch,
        );
    }
    let wrong_state = StateIdentity {
        owner: OwnerId(99),
        ..original.state
    };
    let wrong_legal =
        LegalMoveView::try_new(wrong_state, original.legal_order, movements(), 256).unwrap();
    assert_eq!(
        rebuild(&request, original, request.position().clone(), wrong_legal)
            .unwrap_err()
            .code,
        ErrorCode::IdentityMismatch,
    );
    let wrong_position =
        PositionSnapshot::try_new(wrong_state, Arc::new(()), Color::White, classification())
            .unwrap();
    assert_eq!(
        rebuild(&request, original, wrong_position, request.legal().clone())
            .unwrap_err()
            .code,
        ErrorCode::IdentityMismatch,
    );
}

#[test]
fn terminal_with_board_legal_moves_bypasses_evaluation_but_claims_do_not() {
    let request = request();
    for play_status in [
        PlayStatus::Terminal {
            reason: TerminalReason::DeadPosition,
            winner: None,
        },
        PlayStatus::Terminal {
            reason: TerminalReason::FivefoldRepetition,
            winner: None,
        },
        PlayStatus::Terminal {
            reason: TerminalReason::Checkmate,
            winner: Some(Color::Black),
        },
    ] {
        let classified = PositionClassification {
            play_status,
            ..classification()
        };
        let position = PositionSnapshot::try_new(
            request.context().state,
            Arc::new(()),
            Color::White,
            classified,
        )
        .unwrap();
        assert_eq!(
            rebuild(
                &request,
                request.context(),
                position,
                request.legal().clone()
            )
            .unwrap_err()
            .code,
            ErrorCode::InvalidInput,
        );
    }
    let empty = LegalMoveView::try_new(
        request.context().state,
        request.context().legal_order,
        vec![],
        256,
    )
    .unwrap();
    assert_eq!(
        rebuild(
            &request,
            request.context(),
            request.position().clone(),
            empty
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidInput,
    );
    let claims = PositionClassification {
        history: HistoryCompleteness::UnknownPrefix,
        claims: vec![
            ClaimEvidence {
                rule: ClaimRule::FiftyMove,
                source: ClaimSource::CurrentPosition,
                availability: ClaimAvailability::Available,
            },
            ClaimEvidence {
                rule: ClaimRule::Threefold,
                source: ClaimSource::IntendedMove(movements()[0]),
                availability: ClaimAvailability::Unknown,
            },
        ]
        .into(),
        ..classification()
    };
    let ongoing = request_with_state(Arc::new(()), claims);
    assert!(validate(&output(&ongoing), &ongoing).is_ok());
    assert_eq!(
        ongoing.position().classification().history,
        HistoryCompleteness::UnknownPrefix
    );
    assert_eq!(ongoing.position().classification().claims.len(), 2);
}

#[test]
fn acceptance_uses_current_scope_strict_deadline_and_shared_cancellation() {
    let request = request();
    let evaluated = output(&request);
    let current = scope(&request);
    let clock = request.deadline().clock;
    assert!(evaluated
        .validate_for(&request, current, clock, MonotonicTick(999))
        .is_ok());
    for now in [1000, 1001] {
        assert_eq!(
            evaluated
                .validate_for(&request, current, clock, MonotonicTick(now))
                .unwrap_err()
                .code,
            ErrorCode::Expired,
        );
    }
    assert_eq!(
        evaluated
            .validate_for(
                &request,
                current,
                ClockDomain(ProcessEpoch(99)),
                MonotonicTick(1)
            )
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput,
    );
    for changed in [
        AcceptanceScope {
            game: GameGeneration(99),
            ..current
        },
        AcceptanceScope {
            root: RootGeneration(99),
            ..current
        },
        AcceptanceScope {
            model: ModelHandle {
                generation: SlotGeneration(99),
                ..current.model
            },
            ..current
        },
        AcceptanceScope {
            encoding: EncodingHandle {
                generation: SlotGeneration(99),
                ..current.encoding
            },
            ..current
        },
        AcceptanceScope {
            backend: digest(99),
            ..current
        },
    ] {
        assert_eq!(
            evaluated
                .validate_for(&request, changed, clock, MonotonicTick(999))
                .unwrap_err()
                .code,
            ErrorCode::Stale,
        );
    }
    let cancellation = request.cancel_token().clone();
    cancellation.cancel();
    cancellation.cancel();
    assert_eq!(
        validate(&evaluated, &request).unwrap_err().code,
        ErrorCode::Canceled
    );
}

#[test]
fn numerical_validation_never_silently_repairs_policy_or_wdl() {
    for probabilities in [
        vec![],
        vec![0.0, 0.0],
        vec![f64::NAN, 1.0],
        vec![f64::INFINITY, 0.0],
        vec![-0.1, 1.1],
        vec![0.25, 0.25],
        vec![1.01, 0.0],
    ] {
        assert_eq!(
            LegalPolicy::try_new(probabilities, 1e-6).unwrap_err().code,
            ErrorCode::NumericalFailure,
        );
    }
    for tolerance in [f64::NAN, f64::INFINITY, -0.1, 0.011] {
        assert_eq!(
            LegalPolicy::try_new(vec![0.5, 0.5], tolerance)
                .unwrap_err()
                .code,
            ErrorCode::InvalidInput,
        );
    }
    for (win, draw, loss) in [(f32::NAN, 0.0, 1.0), (-0.1, 0.5, 0.6), (0.2, 0.2, 0.2)] {
        assert_eq!(
            Wdl::try_new(win, draw, loss, 1e-6).unwrap_err().code,
            ErrorCode::NumericalFailure
        );
    }
}

#[test]
fn explicit_normalization_handles_f32_roundoff_for_strict_b_consumer() {
    // Independent IEEE conversion vector: these two f32 values sum below one in f64.
    let promoted = vec![f64::from(0.1_f32), f64::from(0.9_f32)];
    assert!((promoted.iter().sum::<f64>() - 1.0).abs() > 1e-9);
    assert!(LegalPolicy::try_new(promoted.clone(), 1e-9).is_err());
    let admitted = LegalPolicy::try_new(promoted.clone(), 1e-6).unwrap();
    assert_eq!(admitted.probabilities(), promoted);
    let normalized = admitted.normalized();
    assert!((normalized.iter().sum::<f64>() - 1.0).abs() <= 1e-9);
    let strict = LegalPolicy::try_new(normalized, 1e-9).unwrap();
    assert!((strict.probabilities()[0] - 0.1).abs() < 1e-7);
    assert!((strict.probabilities()[1] - 0.9).abs() < 1e-7);
    let wdl = Wdl::try_new(0.1, 0.0, 0.9, 1e-6).unwrap();
    let raw = wdl.probabilities().map(f64::from);
    assert!((raw.iter().sum::<f64>() - 1.0).abs() > 1e-9);
    let normalized_wdl = wdl.normalized();
    assert!((normalized_wdl.iter().sum::<f64>() - 1.0).abs() <= 1e-9);
    assert_eq!(wdl.probabilities(), [0.1, 0.0, 0.9]);
    assert_eq!(
        wdl.flipped().normalized(),
        [normalized_wdl[2], normalized_wdl[1], normalized_wdl[0]]
    );
    let request = request();
    let mut evaluated = output(&request);
    evaluated.policy = strict;
    assert!(validate(&evaluated, &request).is_ok());
}

#[test]
fn terminal_metadata_cannot_invert_the_checkmated_side_or_award_a_draw() {
    let request = request();
    for side in [Color::White, Color::Black] {
        for (reason, winner, valid) in [
            (TerminalReason::Checkmate, Some(side), false),
            (TerminalReason::Checkmate, Some(side.opposite()), true),
            (TerminalReason::Checkmate, None, false),
            (TerminalReason::Stalemate, Some(side.opposite()), false),
            (TerminalReason::Stalemate, None, true),
        ] {
            let result = PositionSnapshot::try_new(
                request.context().state,
                Arc::new(()),
                side,
                PositionClassification {
                    play_status: PlayStatus::Terminal { reason, winner },
                    ..classification()
                },
            );
            if valid {
                assert!(result.is_ok());
            } else {
                assert_eq!(result.unwrap_err().code, ErrorCode::InvalidInput);
            }
        }
    }
}

#[test]
fn raw_cache_provenance_is_distinct_from_new_physical_execution() {
    let request = request();
    let mut evaluated = output(&request);
    let previous = Some(ExecutionId::new(ProcessEpoch(7), 250));
    for source_execution in [None, previous, Some(ExecutionId::new(ProcessEpoch(6), 250))] {
        evaluated.actual.provenance = CacheProvenance::RawEvalHit { source_execution };
        evaluated.actual.execution = None;
        assert!(validate(&evaluated, &request).is_ok());
        evaluated.actual.execution = Some(ExecutionId::new(ProcessEpoch(7), 300));
        assert_eq!(
            validate(&evaluated, &request).unwrap_err().code,
            ErrorCode::IdentityMismatch
        );
    }
    for provenance in [
        CacheProvenance::Computed,
        CacheProvenance::ExactFeatureReuse,
    ] {
        evaluated.actual.provenance = provenance;
        evaluated.actual.execution = None;
        assert_eq!(
            validate(&evaluated, &request).unwrap_err().code,
            ErrorCode::IdentityMismatch
        );
        evaluated.actual.execution = Some(ExecutionId::new(ProcessEpoch(7), 300));
        assert!(validate(&evaluated, &request).is_ok());
        evaluated.actual.execution = Some(ExecutionId::new(ProcessEpoch(99), 300));
        assert_eq!(
            validate(&evaluated, &request).unwrap_err().code,
            ErrorCode::IdentityMismatch
        );
    }
}

#[test]
fn actual_compute_must_match_declared_precision_backend_steps_and_fullness() {
    let request = request();
    let original = output(&request).actual;
    for actual in [
        ActualCompute {
            precision: PrecisionProfile::Fp16,
            ..original
        },
        ActualCompute {
            backend: digest(99),
            ..original
        },
        ActualCompute {
            steps: 0,
            full: false,
            ..original
        },
        ActualCompute {
            steps: 2,
            ..original
        },
        ActualCompute {
            full: false,
            ..original
        },
    ] {
        let mut evaluated = output(&request);
        evaluated.actual = actual;
        assert_eq!(
            validate(&evaluated, &request).unwrap_err().code,
            ErrorCode::UnsupportedContract
        );
    }
    // A recurrent research model can report a declared partial result only if
    // its request permits it; the initial one-step fixture cannot be partial.
    let descriptor = ModelDescriptor::try_new(
        request.context().model,
        request.model().encoding().clone(),
        vec![PrecisionProfile::Fp32],
        2,
        16,
    )
    .unwrap();
    let mut context = request.context();
    context.compute = ComputeBudget {
        min_steps: 1,
        max_steps: 2,
        require_full: false,
    };
    let partial_request = EvalRequest::try_new(
        context,
        request.position().clone(),
        request.legal().clone(),
        Arc::new(descriptor),
        request.deadline(),
        CancelToken::new(),
        request.byte_budget(),
    )
    .unwrap();
    let mut partial = output(&partial_request);
    partial.actual.full = false;
    assert!(validate(&partial, &partial_request).is_ok());
    partial.actual.full = true;
    assert_eq!(
        validate(&partial, &partial_request).unwrap_err().code,
        ErrorCode::UnsupportedContract
    );
    partial.actual.steps = 2;
    assert!(validate(&partial, &partial_request).is_ok());
    partial.actual.full = false;
    assert_eq!(
        validate(&partial, &partial_request).unwrap_err().code,
        ErrorCode::UnsupportedContract
    );
}

#[test]
fn admission_rejects_unsupported_precision_or_full_step_budget() {
    let request = request();
    let original = request.context();
    for context in [
        EvalContext {
            precision: PrecisionProfile::Fp16,
            ..original
        },
        EvalContext {
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 2,
                require_full: false,
            },
            ..original
        },
    ] {
        assert_eq!(
            rebuild(
                &request,
                context,
                request.position().clone(),
                request.legal().clone()
            )
            .unwrap_err()
            .code,
            ErrorCode::UnsupportedContract,
        );
    }
    let descriptor = ModelDescriptor::try_new(
        original.model,
        request.model().encoding().clone(),
        vec![PrecisionProfile::Fp32],
        2,
        16,
    )
    .unwrap();
    assert_eq!(
        EvalRequest::try_new(
            original,
            request.position().clone(),
            request.legal().clone(),
            Arc::new(descriptor),
            request.deadline(),
            CancelToken::new(),
            request.byte_budget(),
        )
        .unwrap_err()
        .code,
        ErrorCode::UnsupportedContract,
    );
}

#[test]
fn owned_state_survives_source_drop_and_logical_cancel_until_last_request_drop() {
    #[derive(Debug)]
    struct FrozenFixture;
    let state = Arc::new(FrozenFixture);
    let weak = Arc::downgrade(&state);
    let request = Arc::new(request_with_state(Arc::clone(&state), classification()));
    let logical_subscriber = Arc::clone(&request);
    drop(state);
    assert!(weak.upgrade().is_some());
    request.cancel_token().cancel();
    assert!(weak.upgrade().is_some());
    drop(request);
    assert!(weak.upgrade().is_some());
    drop(logical_subscriber);
    assert!(weak.upgrade().is_none());
    // This is request ownership evidence only, not a physical GPU lease test.
}

#[test]
fn legal_view_checks_semantic_duplicates_limits_geometry_and_all_promotions() {
    let request = request();
    let state = request.context().state;
    let order = request.context().legal_order;
    let moves = movements();
    for (candidate, limit, expected) in [
        (vec![moves[0], moves[0]], 256, ErrorCode::InvalidInput),
        (moves.clone(), 1, ErrorCode::ResourceExhausted),
        (vec![], 0, ErrorCode::ResourceExhausted),
        (
            vec![Move {
                from: square(12),
                to: square(12),
                promotion: None,
            }],
            256,
            ErrorCode::InvalidInput,
        ),
    ] {
        assert_eq!(
            LegalMoveView::try_new(state, order, candidate, limit)
                .unwrap_err()
                .code,
            expected
        );
    }
    let promotions: Vec<_> = [
        Promotion::Queen,
        Promotion::Rook,
        Promotion::Bishop,
        Promotion::Knight,
    ]
    .into_iter()
    .map(|promotion| Move::new(square(48), square(56), Some(promotion)).unwrap())
    .collect();
    let view = LegalMoveView::try_new(state, order, promotions.clone(), 4).unwrap();
    assert_eq!(view.moves(), promotions);
    assert_ne!(view.moves()[0], view.moves()[3]);
    assert!(LegalMoveView::try_new(state, order, movements(), 2).is_ok());
}
