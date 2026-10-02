#![cfg(feature = "contracts")]
use rz_contracts::*;
use rz_encoding::classical::{self, Frame, HistoryFill, Input};
use rz_eval::contracts::{
    encoding_manifest, input_key, MaiaBinding, PreparedBatch, HOST_BYTES_PER_ITEM,
};
use rz_eval::RawOutput;
use std::sync::Arc;

struct Fixture {
    frames: [Frame; 1],
    black: bool,
    binding: MaiaBinding,
}

impl Fixture {
    fn new(black: bool) -> Self {
        let mut pieces = [[0u64; 6]; 2];
        pieces[0][5] = 1 << 4;
        pieces[1][5] = 1 << 60;
        pieces[0][0] = 1 << 48;
        pieces[1][0] = 1 << 8;
        let encoding = EncodingHandle {
            owner: OwnerId(1),
            slot: 2,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(HistoryFill::No),
        };
        let model = ModelHandle {
            owner: OwnerId(1),
            slot: 3,
            generation: SlotGeneration(1),
            manifest: Digest([5; 32]),
        };
        Self {
            frames: [Frame {
                pieces,
                repeated: false,
                en_passant_target: None,
            }],
            black,
            binding: MaiaBinding::new(model, encoding, HistoryFill::No, Digest([7; 32]), 16)
                .unwrap(),
        }
    }
    fn input(&self) -> Input<'_> {
        Input {
            history: &self.frames,
            black_to_move: self.black,
            castling: [true; 4],
            halfmove_clock: 0,
            history_fill: HistoryFill::No,
        }
    }
    fn moves(&self) -> Vec<Move> {
        let flip = |square| Square::try_new(if self.black { square ^ 56 } else { square }).unwrap();
        let mut moves = vec![
            Move::new(flip(4), flip(6), None).unwrap(),
            Move::new(flip(4), flip(2), None).unwrap(),
        ];
        for p in [
            Promotion::Queen,
            Promotion::Rook,
            Promotion::Bishop,
            Promotion::Knight,
        ] {
            moves.push(Move::new(flip(48), flip(56), Some(p)).unwrap());
        }
        moves
    }
    fn request(&self, change: impl FnOnce(&mut EvalContext)) -> Arc<EvalRequest<()>> {
        let state = StateIdentity {
            owner: OwnerId(4),
            revision: StateRevision(5),
            semantic: Digest([6; 32]),
        };
        let position = PositionSnapshot::try_new(
            state,
            Arc::new(()),
            if self.black {
                Color::Black
            } else {
                Color::White
            },
            PositionClassification {
                play_status: PlayStatus::Ongoing,
                claims: Arc::from([]),
                history: HistoryCompleteness::UnknownPrefix,
                rules_profile: Digest([8; 32]),
            },
        )
        .unwrap();
        let legal = LegalMoveView::try_new(
            state,
            LegalOrderIdentity(Digest([9; 32])),
            self.moves(),
            256,
        )
        .unwrap();
        let mut context = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(ProcessEpoch(10), 1),
            selection: SelectionId::new(ProcessEpoch(10), 2),
            game: GameGeneration(3),
            root: RootGeneration(4),
            state,
            legal_order: legal.order(),
            input: input_key(
                self.binding.model().encoding().handle,
                &classical::encode(self.input()).unwrap(),
            ),
            model: self.binding.model().handle(),
            encoding: self.binding.model().encoding().handle,
            precision: PrecisionProfile::Fp32,
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            backend: self.binding.backend(),
        };
        change(&mut context);
        Arc::new(
            EvalRequest::try_new(
                context,
                position,
                legal,
                Arc::clone(self.binding.model()),
                Deadline {
                    clock: ClockDomain(ProcessEpoch(10)),
                    at: MonotonicTick(100),
                },
                CancelToken::new(),
                ByteBudget {
                    host: HOST_BYTES_PER_ITEM,
                    device: 0,
                    pinned: 0,
                },
            )
            .unwrap(),
        )
    }
}

fn raw() -> RawOutput {
    RawOutput {
        policy_logits: (0..1858).map(|i| i as f32 * 0.01).collect(),
        wdl: vec![0.5, 0.3, 0.2],
    }
}

fn scope(request: &EvalRequest<()>) -> AcceptanceScope {
    let c = request.context();
    AcceptanceScope {
        game: c.game,
        root: c.root,
        model: c.model,
        encoding: c.encoding,
        backend: c.backend,
    }
}

#[test]
fn both_colors_preserve_order_castling_promotions_context_and_physical_execution() {
    for black in [false, true] {
        let fixture = Fixture::new(black);
        let request = fixture.request(|_| {});
        let prepared = fixture
            .binding
            .prepare(Arc::clone(&request), fixture.input())
            .unwrap();
        assert_eq!(prepared.indices(), &[103, 97, 1792, 1793, 1794, 1401]);
        let execution = ExecutionId::new(ProcessEpoch(10), 11);
        let result = prepared.output(&raw(), execution).unwrap();
        assert_eq!(result.context, request.context());
        assert_eq!(result.legal, *request.legal());
        assert_eq!(result.actual.execution, Some(execution));
        assert_eq!(result.actual.provenance, CacheProvenance::Computed);
        assert_eq!(result.wdl.probabilities(), [0.5, 0.3, 0.2]);
        result
            .validate_for(
                &request,
                scope(&request),
                ClockDomain(ProcessEpoch(10)),
                MonotonicTick(99),
            )
            .unwrap();
        assert_eq!(
            result
                .validate_for(
                    &request,
                    scope(&request),
                    ClockDomain(ProcessEpoch(10)),
                    MonotonicTick(100)
                )
                .unwrap_err()
                .code,
            ErrorCode::Expired
        );
        let stale = AcceptanceScope {
            root: RootGeneration(99),
            ..scope(&request)
        };
        assert_eq!(
            result
                .validate_for(
                    &request,
                    stale,
                    ClockDomain(ProcessEpoch(10)),
                    MonotonicTick(1)
                )
                .unwrap_err()
                .code,
            ErrorCode::Stale
        );
        request.cancel_token().cancel();
        assert_eq!(
            result
                .validate_for(
                    &request,
                    scope(&request),
                    ClockDomain(ProcessEpoch(10)),
                    MonotonicTick(1)
                )
                .unwrap_err()
                .code,
            ErrorCode::Canceled
        );
        // Physical metadata survives cancellation; no new logical success is published by C.
        assert_eq!(
            prepared.output(&raw(), execution).unwrap().actual.execution,
            Some(execution)
        );
    }
}

#[test]
fn wrong_tensor_backend_side_or_epoch_is_rejected() {
    let fixture = Fixture::new(false);
    let request = fixture.request(|_| {});
    let changed_input = Input {
        halfmove_clock: 1,
        ..fixture.input()
    };
    assert!(fixture
        .binding
        .prepare(Arc::clone(&request), changed_input)
        .is_err());
    let wrong_side = Input {
        black_to_move: true,
        ..fixture.input()
    };
    assert!(fixture
        .binding
        .prepare(Arc::clone(&request), wrong_side)
        .is_err());
    let wrong_backend = fixture.request(|c| c.backend = Digest([44; 32]));
    assert!(fixture
        .binding
        .prepare(wrong_backend, fixture.input())
        .is_err());
    let wrong_key = fixture.request(|c| c.input = EvalInputKey(Digest([45; 32])));
    assert!(fixture.binding.prepare(wrong_key, fixture.input()).is_err());
    let prepared = fixture.binding.prepare(request, fixture.input()).unwrap();
    assert_eq!(
        prepared
            .output(&raw(), ExecutionId::new(ProcessEpoch(999), 1))
            .unwrap_err()
            .code,
        ErrorCode::IdentityMismatch
    );
    let mut invalid = raw();
    invalid.wdl[0] = f32::NAN;
    assert_eq!(
        prepared
            .output(&invalid, ExecutionId::new(ProcessEpoch(10), 1))
            .unwrap_err()
            .code,
        ErrorCode::NumericalFailure
    );
}

#[test]
fn batches_reject_duplicate_requests_mixed_backends_and_epoch_changes() {
    let fixture = Fixture::new(false);
    let prepare = || {
        fixture
            .binding
            .prepare(fixture.request(|_| {}), fixture.input())
            .unwrap()
    };
    let execution = ExecutionId::new(ProcessEpoch(10), 12);
    assert!(PreparedBatch::new(execution, vec![prepare(), prepare()]).is_err());
    assert!(PreparedBatch::new(ExecutionId::new(ProcessEpoch(11), 12), vec![prepare()]).is_err());
    assert!(PreparedBatch::<()>::new(execution, vec![]).is_err());
    let second = fixture
        .binding
        .prepare(fixture.request(|c| c.request.sequence = 2), fixture.input())
        .unwrap();
    let batch = PreparedBatch::new(execution, vec![prepare(), second]).unwrap();
    assert_eq!(batch.execution(), execution);
    assert_eq!(batch.requests().len(), 2);
}

#[test]
fn encoding_identity_distinguishes_history_profiles_and_registry_generations() {
    assert_ne!(
        encoding_manifest(HistoryFill::No),
        encoding_manifest(HistoryFill::RepeatOldest)
    );
    let fixture = Fixture::new(false);
    let encoded = classical::encode(fixture.input()).unwrap();
    let handle = fixture.binding.model().encoding().handle;
    assert_ne!(
        input_key(handle, &encoded),
        input_key(
            EncodingHandle {
                generation: SlotGeneration(2),
                ..handle
            },
            &encoded
        )
    );
}
