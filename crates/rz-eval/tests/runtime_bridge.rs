#![cfg(feature = "contracts")]

use rz_contracts::*;
use rz_encoding::classical::HistoryFill;
use rz_eval::contracts::{encoding_manifest, MaiaBinding, HOST_BYTES_PER_ITEM};
use rz_eval::mock::{Callback, Limits as MockLimits, ScriptIdentity, ScriptedBackend, Step};
use rz_eval::rules_projection::ClassicalProjection;
use rz_eval::runtime_bridge::{MockTicket, ScriptedLease, ScriptedRuntimeBackend};
use rz_eval::RawOutput;
use rz_position::contracts::{ContractPosition, RulesState};
use rz_position::Position;
use rz_runtime::contracts::{
    ContractClock, ContractEvaluator, ContractsAdapter, RuntimeRequest, SharedScope,
};
use rz_runtime::{Backend, Clock, Limits, Resources};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::task::Poll;
use std::time::Duration;

const EPOCH: ProcessEpoch = ProcessEpoch(71);

#[derive(Clone, Default)]
struct ManualClock(Arc<AtomicU64>);

impl ManualClock {
    fn set(&self, tick: u64) {
        let previous = self.0.load(Ordering::SeqCst);
        assert!(
            tick >= previous,
            "the shared test clock must remain monotonic"
        );
        self.0.store(tick, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    type Tick = MonotonicTick;
    fn now(&self) -> MonotonicTick {
        MonotonicTick(self.0.load(Ordering::SeqCst))
    }
    fn elapsed(&self, earlier: MonotonicTick, later: MonotonicTick) -> Duration {
        Duration::from_nanos(later.0.saturating_sub(earlier.0))
    }
}

impl ContractClock for ManualClock {
    fn domain(&self) -> ClockDomain {
        ClockDomain(EPOCH)
    }
}

struct Fixture {
    clock: ManualClock,
    projection: ClassicalProjection,
    backend: ScriptedRuntimeBackend<ManualClock>,
}

impl Fixture {
    fn new(steps: Vec<Step>) -> Self {
        Self::with_limits(steps, MockLimits::default())
    }
    fn with_limits(steps: Vec<Step>, limits: MockLimits) -> Self {
        let script = ScriptedBackend::<MockTicket>::new(
            ScriptIdentity {
                name: "actual-rules-runtime-boundaries".into(),
                seed: 71,
            },
            steps,
            limits,
        )
        .unwrap();
        let backend_identity = Digest(script.identity_digest());
        let encoding = EncodingHandle {
            owner: OwnerId(72),
            slot: 1,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(HistoryFill::No),
        };
        let model = ModelHandle {
            owner: OwnerId(72),
            slot: 2,
            generation: SlotGeneration(1),
            manifest: Digest([73; 32]),
        };
        let binding =
            MaiaBinding::new(model, encoding, HistoryFill::No, backend_identity, 1).unwrap();
        let projection = ClassicalProjection::new(binding);
        let clock = ManualClock::default();
        let backend = ScriptedRuntimeBackend::new(
            script,
            clock.clone(),
            projection.clone(),
            Arc::clone(projection.model()),
            backend_identity,
        )
        .unwrap();
        Self {
            clock,
            projection,
            backend,
        }
    }

    fn request(&self, sequence: u64, deadline: u64) -> Arc<EvalRequest<RulesState>> {
        self.request_for(Position::startpos(), sequence, deadline)
    }

    fn request_for(
        &self,
        position: Position,
        sequence: u64,
        deadline: u64,
    ) -> Arc<EvalRequest<RulesState>> {
        let live = ContractPosition::new(OwnerId(80 + sequence), position);
        let frozen = live.export().unwrap();
        let context = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(EPOCH, sequence),
            selection: SelectionId::new(EPOCH, sequence),
            game: GameGeneration(1),
            root: RootGeneration(1),
            state: frozen.snapshot().identity(),
            legal_order: frozen.legal_moves().order(),
            input: self
                .projection
                .input_key(frozen.rules(), frozen.legal_moves().moves())
                .unwrap(),
            model: self.projection.model().handle(),
            encoding: self.projection.model().encoding().handle,
            precision: PrecisionProfile::Fp32,
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            backend: self.projection.backend(),
        };
        Arc::new(
            EvalRequest::try_new(
                context,
                frozen.snapshot().clone(),
                frozen.legal_moves().clone(),
                Arc::clone(self.projection.model()),
                Deadline {
                    clock: self.clock.domain(),
                    at: MonotonicTick(deadline),
                },
                CancelToken::new(),
                ByteBudget {
                    host: HOST_BYTES_PER_ITEM + 4096,
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
        policy_logits: vec![0.0; 1858],
        wdl: vec![0.5, 0.3, 0.2],
    }
}

fn step(callbacks: &[u64], done: u64, cancel_ack: Option<u64>) -> Step {
    Step {
        callbacks: callbacks
            .iter()
            .map(|&after| Callback {
                after: Duration::from_nanos(after),
                reply: Ok(raw()),
            })
            .collect(),
        device_complete_after: Duration::from_nanos(done),
        cancel_ack_after: cancel_ack.map(Duration::from_nanos),
    }
}

fn scope(request: &EvalRequest<RulesState>) -> AcceptanceScope {
    let context = request.context();
    AcceptanceScope {
        game: context.game,
        root: context.root,
        model: context.model,
        encoding: context.encoding,
        backend: context.backend,
    }
}

fn launch(
    fixture: &mut Fixture,
    request: Arc<EvalRequest<RulesState>>,
    sequence: u64,
) -> ScriptedLease<ManualClock> {
    let request = Arc::new(RuntimeRequest::new(request));
    fixture
        .backend
        .dispatch(&ExecutionId::new(EPOCH, sequence), &[request])
        .unwrap()
}

fn physical_result(
    fixture: &mut Fixture,
    lease: &mut ScriptedLease<ManualClock>,
    expected: RequestId,
) -> Result<EvalOutput, ContractError> {
    let Poll::Ready(mut results) = fixture.backend.poll(lease) else {
        panic!("physical completion must be ready");
    };
    assert_eq!(
        results.len(),
        1,
        "one physical input has exactly one tagged owned result"
    );
    let result = results.pop().unwrap();
    assert_eq!(result.request_id, expected);
    result.output
}

fn runtime_limits() -> Limits {
    Limits {
        max_requests: 4,
        max_batch_items: 1,
        max_executions: 1,
        max_batch_wait: Duration::ZERO,
        max_queue_age: Duration::from_nanos(1000),
        deadline_reserve: Duration::ZERO,
        memory: Resources {
            host_bytes: 4 * (HOST_BYTES_PER_ITEM + 8192),
            device_bytes: 0,
            pinned_bytes: 0,
        },
    }
}

#[test]
fn callback_is_pending_until_device_completion_and_owned_result_is_once() {
    let mut fixture = Fixture::new(vec![step(&[3], 7, None)]);
    let request = fixture.request(1, 100);
    let expected = request.context();
    let weak = Arc::downgrade(&request);
    let mut lease = launch(&mut fixture, request, 1);
    fixture.clock.set(3);
    assert!(matches!(fixture.backend.poll(&mut lease), Poll::Pending));
    assert!(
        weak.upgrade().is_some(),
        "candidate callback cannot release request pins"
    );
    assert_eq!(
        fixture
            .backend
            .controller()
            .script_status()
            .unwrap()
            .physical_leases,
        1
    );
    fixture.clock.set(7);
    let result = physical_result(&mut fixture, &mut lease, expected.request).unwrap();
    assert_eq!(result.context, expected);
    assert_eq!(result.actual.execution, Some(ExecutionId::new(EPOCH, 1)));
    assert_eq!(result.wdl.probabilities(), [0.5, 0.3, 0.2]);
    assert!(
        matches!(fixture.backend.poll(&mut lease), Poll::Pending),
        "no duplicate Ready publication"
    );
    assert!(
        weak.upgrade().is_some(),
        "the lease still owns its prepared request until dropped"
    );
    drop(lease);
    assert!(
        weak.upgrade().is_none(),
        "the owned output must not borrow its request lease"
    );
    assert_eq!(result.context, expected);
}

#[test]
fn cancellation_acknowledgement_and_forward_failure_never_release_physical_pins() {
    for fail_forward in [false, true] {
        let mut limits = MockLimits::default();
        if fail_forward {
            limits.max_pending_events = 2;
        }
        let mut fixture = Fixture::with_limits(vec![step(&[3], 10, Some(2))], limits);
        let request = fixture.request(1, 100);
        let expected = request.context().request;
        let cancel = request.cancel_token().clone();
        let weak = Arc::downgrade(&request);
        let diagnostics = fixture.backend.diagnostics();
        let mut lease = launch(&mut fixture, request, 1);
        fixture.clock.set(2);
        cancel.cancel();
        assert!(matches!(fixture.backend.poll(&mut lease), Poll::Pending));
        let first = diagnostics.take().unwrap();
        assert_eq!(first.cancellation_attempts, 1);
        assert_eq!(first.cancellations_forwarded, u64::from(!fail_forward));
        assert_eq!(first.cancellation_failures, u64::from(fail_forward));
        assert_eq!(first.cancel_acknowledgements, 0);
        fixture.clock.set(4);
        assert!(matches!(fixture.backend.poll(&mut lease), Poll::Pending));
        let after_ack = diagnostics.take().unwrap();
        assert_eq!(
            after_ack.cancellation_attempts, 1,
            "forwarding is a bounded single attempt"
        );
        assert_eq!(after_ack.cancel_acknowledgements, u64::from(!fail_forward));
        assert!(weak.upgrade().is_some());
        fixture.clock.set(10);
        let result = physical_result(&mut fixture, &mut lease, expected).unwrap();
        let pinned = weak.upgrade().unwrap();
        assert_eq!(
            result
                .validate_for(
                    &pinned,
                    scope(&pinned),
                    fixture.clock.domain(),
                    fixture.clock.now()
                )
                .unwrap_err()
                .code,
            ErrorCode::Canceled,
            "physical success is not logical cancellation acceptance"
        );
        drop(pinned);
        drop(lease);
        assert!(weak.upgrade().is_none());
    }
    let mut fixture = Fixture::new(vec![step(&[3], 10, Some(0))]);
    let request = fixture.request(1, 100);
    let cancellation = request.cancel_token().clone();
    let weak = Arc::downgrade(&request);
    let mut lease = launch(&mut fixture, request, 1);
    fixture.clock.set(2);
    cancellation.cancel();
    assert!(matches!(fixture.backend.poll(&mut lease), Poll::Pending));
    let same_tick = fixture.backend.diagnostics().take().unwrap();
    assert_eq!(same_tick.cancel_acknowledgements, 1);
    assert_eq!(same_tick.cancellation_attempts, 1);
    assert!(
        weak.upgrade().is_some(),
        "same-tick acknowledgement is not physical completion"
    );
    assert!(matches!(fixture.backend.poll(&mut lease), Poll::Pending));
    assert_eq!(
        fixture
            .backend
            .diagnostics()
            .take()
            .unwrap()
            .cancel_acknowledgements,
        1
    );
    fixture.clock.set(10);
    physical_result(&mut fixture, &mut lease, RequestId::new(EPOCH, 1)).unwrap();
    drop(lease);
    assert!(weak.upgrade().is_none());
}

#[test]
fn actual_d_evaluator_rejects_deadline_and_root_changes_while_physical_pins_remain() {
    for root_change in [false, true] {
        let fixture = Fixture::new(vec![step(&[2], 10, None)]);
        let request = fixture.request(1, if root_change { 100 } else { 5 });
        let weak = Arc::downgrade(&request);
        let shared_scope = SharedScope::new(scope(&request));
        let controller = fixture.backend.controller();
        let clock = fixture.clock.clone();
        let adapter = ContractsAdapter::new(shared_scope.clone(), clock.clone(), 1).unwrap();
        let mut runtime =
            ContractEvaluator::new(adapter, fixture.backend, runtime_limits(), 16).unwrap();
        runtime.submit(request).unwrap();
        runtime.pump();
        assert_eq!(runtime.state().executions, 1);
        clock.set(2);
        assert!(
            runtime.poll().is_none(),
            "callback is still only a candidate"
        );
        if root_change {
            shared_scope.update(AcceptanceScope {
                root: RootGeneration(2),
                ..shared_scope.get()
            });
        } else {
            clock.set(5); // Exactly the deadline is outside acceptance.
        }
        let result = runtime.poll().expect("logical rejection must be delivered");
        assert!(if root_change {
            matches!(result, EvalResult::Stale(_))
        } else {
            matches!(result, EvalResult::Expired(_))
        });
        assert_eq!(runtime.state().executions, 1);
        assert!(runtime.state().reserved.host_bytes > 0);
        assert!(weak.upgrade().is_some());
        assert_eq!(controller.script_status().unwrap().physical_leases, 1);
        clock.set(10);
        runtime.pump();
        assert_eq!(runtime.state().executions, 0);
        assert_eq!(runtime.state().reserved.host_bytes, 0);
        assert!(weak.upgrade().is_none());
        assert!(
            runtime.poll().is_none(),
            "late physical completion cannot publish a second logical result"
        );
    }
}

#[test]
fn duplicate_missing_and_late_callbacks_are_typed_failures_and_ready_is_immutable() {
    for callbacks in [vec![1, 2], vec![], vec![5]] {
        let mut fixture = Fixture::new(vec![step(&callbacks, 3, None)]);
        let request = fixture.request(1, 100);
        let id = request.context().request;
        let mut lease = launch(&mut fixture, request, 1);
        fixture.clock.set(3);
        let failure = physical_result(&mut fixture, &mut lease, id).unwrap_err();
        assert_eq!(failure.code, ErrorCode::BackendFailure);
        assert_eq!(failure.stage, Stage::Output);
        if callbacks.len() == 2 {
            assert!(failure.detail.contains("duplicate"));
        } else {
            assert!(failure.detail.contains("without a callback"));
        }
        drop(lease);
        if callbacks == vec![5] {
            let controller = fixture.backend.controller();
            let status = controller.script_status().unwrap();
            assert_eq!(status.physical_leases, 0);
            assert_eq!(
                status.scheduled_tickets, 1,
                "late simulator event pins outlive physical lease"
            );
            fixture.clock.set(5);
            controller.pump().unwrap();
            assert_eq!(
                fixture.backend.diagnostics().take().unwrap().late_callbacks,
                1
            );
            assert_eq!(controller.script_status().unwrap().scheduled_events, 0);
            assert_eq!(controller.script_status().unwrap().scheduled_tickets, 0);
            assert!(
                failure.detail.contains("without a callback"),
                "late candidate cannot rewrite published failure"
            );
        }
    }
    let mut fixture = Fixture::new(vec![step(&[1, 5], 3, None)]);
    let request = fixture.request(1, 100);
    let id = request.context().request;
    let mut lease = launch(&mut fixture, request, 1);
    fixture.clock.set(3);
    let result = physical_result(&mut fixture, &mut lease, id).unwrap();
    let probabilities = result.policy.probabilities().to_vec();
    let context = result.context;
    drop(lease);
    fixture.clock.set(5);
    fixture.backend.pump().unwrap();
    assert_eq!(
        fixture.backend.diagnostics().take().unwrap().late_callbacks,
        1
    );
    assert_eq!(result.context, context);
    assert_eq!(result.policy.probabilities(), probabilities);
}

#[test]
fn rejected_physical_ids_are_not_reused_and_bridge_owners_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<RulesState>();
    assert_send::<ScriptedRuntimeBackend<ManualClock>>();
    assert_send::<ScriptedLease<ManualClock>>();
    let mut fixture = Fixture::new(vec![step(&[1], 2, None)]);
    let expired = Arc::new(RuntimeRequest::new(fixture.request(1, 0)));
    let first = fixture
        .backend
        .dispatch(&ExecutionId::new(EPOCH, 1), &[expired])
        .err()
        .unwrap();
    assert_eq!(first.code, ErrorCode::Expired);
    assert_eq!(
        fixture
            .backend
            .controller()
            .script_status()
            .unwrap()
            .scheduled_tickets,
        0
    );
    let reused = Arc::new(RuntimeRequest::new(fixture.request(1, 100)));
    let rejected = fixture
        .backend
        .dispatch(&ExecutionId::new(EPOCH, 1), &[reused])
        .err()
        .unwrap();
    assert_eq!(rejected.code, ErrorCode::IdentityMismatch);
    let request = fixture.request(2, 100);
    let expected = request.context().request;
    let mut lease = launch(&mut fixture, request, 2);
    fixture.clock.set(2);
    assert_eq!(
        physical_result(&mut fixture, &mut lease, expected)
            .unwrap()
            .actual
            .execution,
        Some(ExecutionId::new(EPOCH, 2))
    );
}

#[test]
fn actual_rules_projection_preserves_unknown_history_repetition_promotions_and_castling() {
    let fixture = Fixture::new(vec![step(&[1], 2, None)]);
    let mut repeated = Position::startpos();
    repeated
        .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
        .unwrap();
    let imported = Position::from_fen(&repeated.to_fen()).unwrap();
    let known = fixture.request_for(repeated, 1, 100);
    let unknown = fixture.request_for(imported, 2, 100);
    assert_eq!(
        known.position().state().snapshot().to_fen(),
        unknown.position().state().snapshot().to_fen()
    );
    assert_ne!(
        known.context().state.semantic,
        unknown.context().state.semantic
    );
    assert_ne!(known.context().input, unknown.context().input);
    assert_eq!(
        known.position().classification().history,
        HistoryCompleteness::Complete
    );
    assert_eq!(
        unknown.position().classification().history,
        HistoryCompleteness::UnknownPrefix
    );
    let repeated_input = fixture.projection.encode(known.position().state()).unwrap();
    let imported_input = fixture
        .projection
        .encode(unknown.position().state())
        .unwrap();
    assert!(repeated_input.values()[12 * 64..13 * 64]
        .iter()
        .all(|&value| value == 1.0));
    assert!(imported_input.values()[12 * 64..13 * 64]
        .iter()
        .all(|&value| value == 0.0));

    let promoted = fixture.request_for(
        Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap(),
        3,
        100,
    );
    let indices = fixture
        .projection
        .legal_indices(promoted.position().state(), promoted.legal().moves())
        .unwrap();
    let promotions: Vec<_> = promoted
        .legal()
        .moves()
        .iter()
        .zip(indices)
        .filter(|(mv, _)| mv.from.index() == 48 && mv.to.index() == 56)
        .map(|(mv, index)| (mv.promotion.unwrap(), index))
        .collect();
    assert_eq!(
        promotions,
        vec![
            (Promotion::Queen, 1792),
            (Promotion::Rook, 1793),
            (Promotion::Bishop, 1794),
            (Promotion::Knight, 1401)
        ]
    );

    let castle = fixture.request_for(
        Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap(),
        4,
        100,
    );
    let indices = fixture
        .projection
        .legal_indices(castle.position().state(), castle.legal().moves())
        .unwrap();
    let castles: Vec<_> = castle
        .legal()
        .moves()
        .iter()
        .zip(indices)
        .filter(|(mv, _)| mv.from.index() == 4 && (mv.to.index() == 2 || mv.to.index() == 6))
        .map(|(mv, index)| (mv.to.index(), index))
        .collect();
    assert_eq!(castles, vec![(2, 97), (6, 103)]);
}
