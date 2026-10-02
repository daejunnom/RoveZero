#![cfg(feature = "contracts")]

//! Real TASK-I01 contract fixtures with a controlled, CPU-only physical backend.
//! The frozen state is test-local; no concrete chess implementation is required.

use rz_contracts::{
    AcceptanceScope, ActualCompute, ByteBudget, CacheProvenance, CancelToken, ClockDomain, Color,
    ComputeBudget, ContractError, Deadline, Digest, EncodingDescriptor, EncodingHandle, ErrorCode,
    EvalContext, EvalInputKey, EvalOutput, EvalRequest, EvalResult, Evaluator, ExecutionId,
    GameGeneration, HistoryCompleteness, LegalMoveView, LegalOrderIdentity, LegalPolicy,
    ModelDescriptor, ModelHandle, MonotonicTick, Move, OwnerId, PlayStatus, PositionClassification,
    PositionSnapshot, PrecisionProfile, ProcessEpoch, RequestId, RootGeneration, SelectionId,
    SlotGeneration, Square, Stage, StateIdentity, StateRevision, Viewpoint, Wdl, CONTRACT_REVISION,
};
use rz_runtime::contracts::{
    ContractClock, ContractEvaluator, ContractsAdapter, RuntimeRequest, SharedScope,
};
use rz_runtime::{
    Backend, BackendResult, Clock, DrainState, Limits, ObservationKind, Resources, Scheduler,
};
use rz_telemetry::FinishKind;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Duration;

const EPOCH: ProcessEpoch = ProcessEpoch(7);
const CLOCK_DOMAIN: ClockDomain = ClockDomain(EPOCH);

fn digest(value: u8) -> Digest {
    Digest([value; 32])
}

fn scope() -> AcceptanceScope {
    AcceptanceScope {
        game: GameGeneration(1),
        root: RootGeneration(1),
        model: ModelHandle {
            owner: OwnerId(1),
            slot: 0,
            generation: SlotGeneration(1),
            manifest: digest(1),
        },
        encoding: EncodingHandle {
            owner: OwnerId(2),
            slot: 0,
            generation: SlotGeneration(1),
            manifest: digest(2),
        },
        backend: digest(3),
    }
}

fn request_in_domain(
    sequence: u64,
    epoch: ProcessEpoch,
    clock: ClockDomain,
    at: u64,
) -> Arc<EvalRequest<()>> {
    let scope = scope();
    let state = StateIdentity {
        owner: OwnerId(3),
        revision: StateRevision(1),
        semantic: digest(4),
    };
    let legal_order = LegalOrderIdentity(digest(5));
    let position = PositionSnapshot::try_new(
        state,
        Arc::new(()),
        Color::White,
        PositionClassification {
            play_status: PlayStatus::Ongoing,
            claims: Arc::from([]),
            history: HistoryCompleteness::Complete,
            rules_profile: digest(6),
        },
    )
    .unwrap();
    let legal = LegalMoveView::try_new(
        state,
        legal_order,
        vec![Move::new(
            Square::try_new(12).unwrap(),
            Square::try_new(28).unwrap(),
            None,
        )
        .unwrap()],
        1,
    )
    .unwrap();
    let model = Arc::new(
        ModelDescriptor::try_new(
            scope.model,
            EncodingDescriptor {
                handle: scope.encoding,
                history_length: 1,
                action_map: digest(7),
                history_policy: digest(8),
            },
            vec![PrecisionProfile::Fp32],
            4,
            4,
        )
        .unwrap(),
    );
    Arc::new(
        EvalRequest::try_new(
            EvalContext {
                revision: CONTRACT_REVISION,
                request: RequestId::new(epoch, sequence),
                selection: SelectionId::new(epoch, sequence),
                game: scope.game,
                root: scope.root,
                state,
                legal_order,
                input: EvalInputKey(digest(9)),
                model: scope.model,
                encoding: scope.encoding,
                precision: PrecisionProfile::Fp32,
                compute: ComputeBudget {
                    min_steps: 4,
                    max_steps: 4,
                    require_full: true,
                },
                backend: scope.backend,
            },
            position,
            legal,
            model,
            Deadline {
                clock,
                at: MonotonicTick(at),
            },
            CancelToken::new(),
            ByteBudget {
                host: 11,
                device: 13,
                pinned: 5,
            },
        )
        .unwrap(),
    )
}

fn request(sequence: u64, at: u64) -> Arc<EvalRequest<()>> {
    request_in_domain(sequence, EPOCH, CLOCK_DOMAIN, at)
}

fn output(request: &EvalRequest<()>, execution: ExecutionId) -> EvalOutput {
    EvalOutput {
        context: request.context(),
        legal: request.legal().clone(),
        policy: LegalPolicy::try_new(vec![1.0], 0.0).unwrap(),
        wdl: Wdl::try_new(0.5, 0.25, 0.25, 0.0).unwrap(),
        viewpoint: Viewpoint::SideToMove,
        actual: ActualCompute {
            precision: request.context().precision,
            steps: 4,
            full: true,
            backend: request.context().backend,
            execution: Some(execution),
            provenance: CacheProvenance::Computed,
        },
    }
}

#[derive(Clone, Default)]
struct ManualClock(Arc<AtomicU64>);

impl ManualClock {
    fn set(&self, tick: u64) {
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
        CLOCK_DOMAIN
    }
}

fn limits(max_requests: usize) -> Limits {
    Limits {
        max_requests,
        max_batch_items: 1,
        max_executions: 1,
        max_batch_wait: Duration::ZERO,
        max_queue_age: Duration::from_nanos(1000),
        deadline_reserve: Duration::ZERO,
        memory: Resources {
            host_bytes: 100,
            device_bytes: 100,
            pinned_bytes: 100,
        },
    }
}

#[derive(Clone, Copy)]
enum OutputMode {
    Valid,
    Failure(ContractError),
    WrongExecution,
    ForeignExecutionEpoch,
    MissingExecution,
    RawCacheHit,
    WrongContext,
    IncompleteCompute,
}

struct Plan {
    ready: Arc<AtomicBool>,
    mode: OutputMode,
    completed_at: Option<u64>,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(true)),
            mode: OutputMode::Valid,
            completed_at: None,
        }
    }
}

#[derive(Default)]
struct BackendControl {
    plans: Mutex<VecDeque<Plan>>,
    executions: Mutex<Vec<ExecutionId>>,
    lease_drops: Arc<AtomicUsize>,
}

struct ControlledBackend {
    control: Arc<BackendControl>,
    clock: ManualClock,
}

struct Lease {
    requests: Vec<Arc<RuntimeRequest<()>>>,
    execution: ExecutionId,
    plan: Plan,
    drops: Arc<AtomicUsize>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

type ContractAdapter = ContractsAdapter<(), ManualClock>;
type Runtime = ContractEvaluator<(), ControlledBackend, ManualClock>;

impl Backend<ContractAdapter> for ControlledBackend {
    type Lease = Lease;

    fn additional_resources(&self, _: &[Arc<RuntimeRequest<()>>]) -> Resources {
        Resources {
            host_bytes: 3,
            device_bytes: 17,
            pinned_bytes: 7,
        }
    }

    fn dispatch(
        &mut self,
        execution: &ExecutionId,
        requests: &[Arc<RuntimeRequest<()>>],
    ) -> Result<Lease, ContractError> {
        self.control.executions.lock().unwrap().push(*execution);
        Ok(Lease {
            requests: requests.to_vec(),
            execution: *execution,
            plan: self
                .control
                .plans
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_default(),
            drops: Arc::clone(&self.control.lease_drops),
        })
    }

    fn poll(&mut self, lease: &mut Lease) -> Poll<Vec<BackendResult<ContractAdapter>>> {
        if !lease.plan.ready.load(Ordering::SeqCst) {
            return Poll::Pending;
        }
        if let Some(tick) = lease.plan.completed_at {
            self.clock.set(tick);
        }
        Poll::Ready(
            lease
                .requests
                .iter()
                .map(|request| {
                    let request = request.eval();
                    let result = match lease.plan.mode {
                        OutputMode::Failure(error) => Err(error),
                        mode => {
                            let mut value = output(request, lease.execution);
                            match mode {
                                OutputMode::WrongExecution => {
                                    value.actual.execution =
                                        Some(ExecutionId::new(EPOCH, lease.execution.sequence + 1));
                                }
                                OutputMode::ForeignExecutionEpoch => {
                                    value.actual.execution = Some(ExecutionId::new(
                                        ProcessEpoch(8),
                                        lease.execution.sequence,
                                    ));
                                }
                                OutputMode::MissingExecution => value.actual.execution = None,
                                OutputMode::RawCacheHit => {
                                    value.actual.execution = None;
                                    value.actual.provenance = CacheProvenance::RawEvalHit {
                                        source_execution: Some(lease.execution),
                                    };
                                }
                                OutputMode::WrongContext => value.context.selection.sequence += 1,
                                OutputMode::IncompleteCompute => {
                                    value.actual.steps = 2;
                                    value.actual.full = false;
                                }
                                OutputMode::Valid => {}
                                OutputMode::Failure(_) => unreachable!(),
                            }
                            Ok(value)
                        }
                    };
                    BackendResult {
                        request_id: request.context().request,
                        output: result,
                    }
                })
                .collect(),
        )
    }
}

struct Fixture {
    runtime: Runtime,
    clock: ManualClock,
    scope: SharedScope,
    backend: Arc<BackendControl>,
}

fn fixture(max_requests: usize) -> Fixture {
    fixture_with_capacity(max_requests, 8)
}

fn fixture_with_capacity(max_requests: usize, observation_capacity: usize) -> Fixture {
    let clock = ManualClock::default();
    let shared_scope = SharedScope::new(scope());
    let backend = Arc::new(BackendControl::default());
    let adapter = ContractsAdapter::new(shared_scope.clone(), clock.clone(), 1).unwrap();
    Fixture {
        runtime: ContractEvaluator::new(
            adapter,
            ControlledBackend {
                control: Arc::clone(&backend),
                clock: clock.clone(),
            },
            limits(max_requests),
            observation_capacity,
        )
        .unwrap(),
        clock,
        scope: shared_scope,
        backend,
    }
}

fn complete(runtime: &mut Runtime) -> EvalResult {
    runtime.pump();
    runtime.pump();
    runtime
        .poll()
        .expect("admitted request publishes one finalized common result")
}

fn failed(result: EvalResult) -> rz_contracts::EvalFailure {
    match result {
        EvalResult::Failed(failure) => failure,
        other => panic!("expected explicit failure, got {other:?}"),
    }
}

#[test]
fn direct_scheduler_consumes_a_queue_full_contract_request_id() {
    let clock = ManualClock::default();
    let backend = Arc::new(BackendControl::default());
    let adapter = ContractsAdapter::new(SharedScope::new(scope()), clock.clone(), 1).unwrap();
    let mut scheduler = Scheduler::new(
        adapter,
        ControlledBackend {
            control: Arc::clone(&backend),
            clock: clock.clone(),
        },
        clock,
        limits(1),
        8,
    )
    .unwrap();
    let first = scheduler
        .submit(RuntimeRequest::new(request(1, 100)))
        .unwrap_or_else(|rejected| panic!("first admission failed: {:?}", rejected.error));
    let refused = scheduler
        .submit(RuntimeRequest::new(request(2, 100)))
        .err()
        .expect("full scheduler must refuse the second request");
    assert_eq!(refused.error.code, ErrorCode::ResourceExhausted);
    drop(refused);
    assert!(scheduler.cancel(&RequestId::new(EPOCH, 1)));
    assert!(matches!(first.try_recv().unwrap(), EvalResult::Canceled(_)));
    assert_eq!(scheduler.state().reserved_requests, 0);
    assert_eq!(scheduler.state().reserved, Resources::default());
    let reused = scheduler
        .submit(RuntimeRequest::new(request(2, 100)))
        .err()
        .expect("free capacity must not permit a refused ID to be reused");
    assert_eq!(reused.error.code, ErrorCode::IdentityMismatch);
    assert_eq!(reused.error.stage, Stage::Admission);
    drop(reused);
    assert_eq!(scheduler.state().reserved_requests, 0);
    assert!(backend.executions.lock().unwrap().is_empty());
    let fresh = scheduler
        .submit(RuntimeRequest::new(request(3, 100)))
        .unwrap_or_else(|rejected| panic!("fresh admission failed: {:?}", rejected.error));
    scheduler.pump();
    scheduler.pump();
    let EvalResult::Completed(output) = fresh.try_recv().unwrap() else {
        panic!("a fresh ID must still complete after a refused retry")
    };
    assert_eq!(output.context.request, RequestId::new(EPOCH, 3));
    assert_eq!(scheduler.state().reserved_requests, 0);
}

#[test]
fn common_observations_preserve_logical_cancel_before_physical_completion() {
    let mut fixture = fixture_with_capacity(1, 64);
    let ready = Arc::new(AtomicBool::new(false));
    fixture.backend.plans.lock().unwrap().push_back(Plan {
        ready: Arc::clone(&ready),
        ..Plan::default()
    });
    let id = RequestId::new(EPOCH, 1);
    fixture.runtime.submit(request(1, 100)).unwrap();
    fixture.runtime.pump();
    let execution = fixture.backend.executions.lock().unwrap()[0];
    fixture.runtime.cancel(id).unwrap();
    assert!(matches!(
        fixture.runtime.poll(),
        Some(EvalResult::Canceled(_))
    ));
    assert_eq!(fixture.backend.lease_drops.load(Ordering::SeqCst), 0);
    ready.store(true, Ordering::SeqCst);
    fixture.clock.set(1);
    fixture.runtime.pump();
    let observed = fixture.runtime.take_observations();
    assert_eq!(observed.dropped, 0);
    assert!(!observed.counter_overflow);
    let canceled = observed
        .events
        .iter()
        .position(|event| {
            matches!(
                &event.kind,
                ObservationKind::Finished {
                    request,
                    kind: FinishKind::Canceled,
                    delivered: true,
                } if *request == id
            )
        })
        .unwrap();
    let completed = observed
        .events
        .iter()
        .position(|event| {
            matches!(
                &event.kind,
                ObservationKind::PhysicalCompleted { execution: actual } if *actual == execution
            )
        })
        .unwrap();
    assert!(canceled < completed);
    assert!(!observed.events.iter().any(|event| {
        matches!(
            &event.kind,
            ObservationKind::ValidationStarted { .. } | ObservationKind::ValidationFinished { .. }
        )
    }));
    assert_eq!(fixture.backend.lease_drops.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert!(fixture.runtime.poll().is_none());
}

#[test]
fn common_observation_loss_is_bounded_and_does_not_change_completion() {
    for capacity in [0, 2] {
        let mut fixture = fixture_with_capacity(1, capacity);
        fixture.runtime.submit(request(1, 100)).unwrap();
        assert!(matches!(
            complete(&mut fixture.runtime),
            EvalResult::Completed(_)
        ));
        let observed = fixture.runtime.take_observations();
        assert!(observed.events.len() <= capacity);
        assert!(observed.dropped > 0);
        assert!(!observed.counter_overflow);
        assert_eq!(fixture.runtime.state().reserved_requests, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        let second = fixture.runtime.take_observations();
        assert!(second.events.is_empty());
        assert_eq!(second.dropped, 0);
        assert!(!second.counter_overflow);
    }
}

#[test]
fn common_shutdown_snapshot_separates_physical_drain_from_unread_delivery() {
    for completed in [false, true] {
        let mut fixture = fixture(1);
        assert_eq!(fixture.runtime.shutdown_snapshot().drain, DrainState::Open);
        fixture.runtime.submit(request(1, 100)).unwrap();
        if completed {
            fixture.runtime.pump();
            fixture.runtime.pump();
        }
        fixture
            .runtime
            .begin_shutdown(Deadline {
                clock: CLOCK_DOMAIN,
                at: MonotonicTick(100),
            })
            .unwrap();
        let snapshot = fixture.runtime.shutdown_snapshot();
        assert_eq!(snapshot.drain, DrainState::Drained);
        assert!(snapshot.state.closed);
        assert_eq!(snapshot.state.logical_requests, 0);
        assert_eq!(snapshot.state.executions, 0);
        assert_eq!(snapshot.state.reserved_requests, 1);
        assert_eq!(
            snapshot.state.reserved,
            Resources {
                host_bytes: 11,
                device_bytes: 13,
                pinned_bytes: 5,
            }
        );
        let result = fixture.runtime.poll().unwrap();
        assert!(matches!(
            (completed, result),
            (true, EvalResult::Completed(_)) | (false, EvalResult::Canceled(_))
        ));
        let released = fixture.runtime.shutdown_snapshot();
        assert_eq!(released.drain, DrainState::Drained);
        assert_eq!(released.state.reserved_requests, 0);
        assert_eq!(released.state.reserved, Resources::default());
        assert!(fixture.runtime.poll().is_none());
    }
}

#[test]
fn common_evaluator_success_echoes_the_actual_physical_execution() {
    let mut fixture = fixture(2);
    let request = request(1, 100);
    fixture.runtime.submit(Arc::clone(&request)).unwrap();
    let result = complete(&mut fixture.runtime);
    let EvalResult::Completed(value) = result else {
        panic!("valid mock evaluation must complete")
    };
    let dispatched = fixture.backend.executions.lock().unwrap()[0];
    assert_eq!(dispatched.epoch, EPOCH);
    assert_eq!(value.actual.execution, Some(dispatched));
    value
        .validate_for(&request, scope(), CLOCK_DOMAIN, MonotonicTick(0))
        .unwrap();
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(fixture.backend.lease_drops.load(Ordering::SeqCst), 1);
    assert!(fixture.runtime.poll().is_none());
}

#[test]
fn foreign_epoch_or_deadline_clock_is_rejected_without_a_completion() {
    let mut fixture = fixture(2);
    for request in [
        request_in_domain(1, ProcessEpoch(8), CLOCK_DOMAIN, 100),
        request_in_domain(2, EPOCH, ClockDomain(ProcessEpoch(8)), 100),
    ] {
        let error = fixture.runtime.submit(request).unwrap_err();
        assert!(matches!(
            error.code,
            ErrorCode::IdentityMismatch | ErrorCode::InvalidInput
        ));
        assert_eq!(fixture.runtime.state().reserved_requests, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert!(fixture.runtime.poll().is_none());
    }
    assert!(fixture.backend.executions.lock().unwrap().is_empty());
}

#[test]
fn request_identity_cannot_be_reused_after_completion() {
    let mut fixture = fixture(2);
    fixture.runtime.submit(request(2, 100)).unwrap();
    assert!(matches!(
        complete(&mut fixture.runtime),
        EvalResult::Completed(_)
    ));
    for sequence in [2, 1] {
        assert_eq!(
            fixture
                .runtime
                .submit(request(sequence, 100))
                .unwrap_err()
                .code,
            ErrorCode::IdentityMismatch
        );
        assert!(fixture.runtime.poll().is_none());
    }
    fixture.runtime.submit(request(3, 100)).unwrap();
    assert!(matches!(
        complete(&mut fixture.runtime),
        EvalResult::Completed(_)
    ));
}

#[test]
fn queue_full_submission_consumes_identity_before_admission() {
    let mut fixture = fixture(1);
    fixture.runtime.submit(request(1, 100)).unwrap();
    assert_eq!(
        fixture.runtime.submit(request(2, 100)).unwrap_err().code,
        ErrorCode::ResourceExhausted
    );
    for sequence in [2, 1] {
        assert_eq!(
            fixture
                .runtime
                .submit(request(sequence, 100))
                .unwrap_err()
                .code,
            ErrorCode::IdentityMismatch,
            "ID freshness must be checked even while admission is full"
        );
    }
    assert_eq!(fixture.runtime.state().logical_requests, 1);
    assert_eq!(fixture.runtime.state().reserved_requests, 1);
    assert!(fixture.backend.executions.lock().unwrap().is_empty());
    assert!(matches!(
        complete(&mut fixture.runtime),
        EvalResult::Completed(_)
    ));
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert_eq!(
        fixture.runtime.submit(request(2, 100)).unwrap_err().code,
        ErrorCode::IdentityMismatch,
        "free capacity must not make a rejected request ID reusable"
    );
    assert!(fixture.runtime.poll().is_none());
    fixture.runtime.submit(request(3, 100)).unwrap();
    let EvalResult::Completed(output) = complete(&mut fixture.runtime) else {
        panic!("a fresh request ID must remain admissible")
    };
    assert_eq!(output.context.request, RequestId::new(EPOCH, 3));
    assert!(fixture.runtime.poll().is_none());
}

#[test]
fn deadline_equal_to_completion_tick_is_expired_after_backend_poll() {
    let mut fixture = fixture(2);
    fixture.clock.set(99);
    fixture.backend.plans.lock().unwrap().push_back(Plan {
        completed_at: Some(100),
        ..Plan::default()
    });
    fixture.runtime.submit(request(1, 100)).unwrap();
    let EvalResult::Expired(context) = complete(&mut fixture.runtime) else {
        panic!("equal tick must expire")
    };
    assert_eq!(context.request.request, RequestId::new(EPOCH, 1));
    assert_eq!(
        context.execution,
        Some(fixture.backend.executions.lock().unwrap()[0])
    );
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert!(fixture.runtime.poll().is_none());
}

#[test]
fn malformed_physical_execution_or_output_context_never_completes() {
    for mode in [
        OutputMode::WrongExecution,
        OutputMode::ForeignExecutionEpoch,
        OutputMode::MissingExecution,
        OutputMode::RawCacheHit,
        OutputMode::WrongContext,
    ] {
        let mut fixture = fixture(2);
        fixture.backend.plans.lock().unwrap().push_back(Plan {
            mode,
            ..Plan::default()
        });
        fixture.runtime.submit(request(1, 100)).unwrap();
        let failure = failed(complete(&mut fixture.runtime));
        assert_eq!(failure.error.code, ErrorCode::IdentityMismatch);
        assert_eq!(failure.context.request.request, RequestId::new(EPOCH, 1));
        assert_eq!(
            failure.context.execution,
            Some(fixture.backend.executions.lock().unwrap()[0])
        );
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert!(fixture.runtime.poll().is_none());
    }
}

#[test]
fn partial_compute_is_an_explicit_contract_failure() {
    let mut fixture = fixture(2);
    fixture.backend.plans.lock().unwrap().push_back(Plan {
        mode: OutputMode::IncompleteCompute,
        ..Plan::default()
    });
    fixture.runtime.submit(request(1, 100)).unwrap();
    let failure = failed(complete(&mut fixture.runtime));
    assert_eq!(failure.error.code, ErrorCode::UnsupportedContract);
    assert_eq!(failure.error.stage, Stage::Output);
    assert_eq!(failure.context.request.request, RequestId::new(EPOCH, 1));
    assert_eq!(
        failure.context.execution,
        Some(fixture.backend.executions.lock().unwrap()[0])
    );
    assert!(fixture.runtime.poll().is_none());
}

#[test]
fn backend_failure_retains_its_stage_detail_and_physical_context() {
    let mut fixture = fixture(2);
    let error = ContractError::new(
        ErrorCode::BackendFailure,
        Stage::Backend,
        "controlled CPU provider failure",
    );
    fixture.backend.plans.lock().unwrap().push_back(Plan {
        mode: OutputMode::Failure(error),
        ..Plan::default()
    });
    fixture.runtime.submit(request(1, 100)).unwrap();
    let failure = failed(complete(&mut fixture.runtime));
    assert_eq!(failure.error, error);
    assert_eq!(failure.context.request.request, RequestId::new(EPOCH, 1));
    assert_eq!(
        failure.context.execution,
        Some(fixture.backend.executions.lock().unwrap()[0])
    );
    assert_eq!(
        failure.recovery,
        rz_contracts::RecoveryOutcome::NotAttempted
    );
    assert!(fixture.runtime.poll().is_none());
}

#[test]
fn execution_sequence_exhaustion_fails_without_launch_or_wrapping() {
    let clock = ManualClock::default();
    let backend = Arc::new(BackendControl::default());
    let adapter = ContractsAdapter::with_execution_high_water(
        SharedScope::new(scope()),
        clock.clone(),
        1,
        u64::MAX,
    )
    .unwrap();
    let mut runtime = ContractEvaluator::new(
        adapter,
        ControlledBackend {
            control: Arc::clone(&backend),
            clock,
        },
        limits(1),
        8,
    )
    .unwrap();
    runtime.submit(request(1, 100)).unwrap();
    let failure = failed(complete(&mut runtime));
    assert_eq!(failure.error.code, ErrorCode::ResourceExhausted);
    assert_eq!(failure.context.request.request, RequestId::new(EPOCH, 1));
    assert_eq!(failure.context.execution, None);
    assert!(backend.executions.lock().unwrap().is_empty());
    assert_eq!(runtime.state().reserved_requests, 0);
    assert_eq!(runtime.state().reserved, Resources::default());
    assert!(runtime.poll().is_none());
}

#[test]
fn cancellation_is_idempotent_and_pins_survive_logical_delivery() {
    let mut fixture = fixture(1);
    let ready = Arc::new(AtomicBool::new(false));
    fixture.backend.plans.lock().unwrap().push_back(Plan {
        ready: Arc::clone(&ready),
        ..Plan::default()
    });
    let id = RequestId::new(EPOCH, 1);
    fixture.runtime.submit(request(1, 100)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.cancel(id).unwrap();
    fixture.runtime.cancel(id).unwrap();
    let EvalResult::Canceled(context) = fixture.runtime.poll().unwrap() else {
        panic!("cancel must publish cancellation")
    };
    assert_eq!(context.request.request, id);
    assert_eq!(
        context.execution,
        Some(fixture.backend.executions.lock().unwrap()[0])
    );
    assert_eq!(fixture.runtime.state().reserved_requests, 1);
    assert_eq!(fixture.runtime.state().executions, 1);
    assert_eq!(
        fixture.runtime.state().reserved,
        Resources {
            host_bytes: 14,
            device_bytes: 30,
            pinned_bytes: 12
        }
    );
    assert_eq!(fixture.backend.lease_drops.load(Ordering::SeqCst), 0);
    assert_eq!(
        fixture.runtime.submit(request(2, 100)).unwrap_err().code,
        ErrorCode::ResourceExhausted
    );
    ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    assert_eq!(fixture.backend.lease_drops.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert!(fixture.runtime.poll().is_none());
    fixture.runtime.submit(request(3, 100)).unwrap();
    assert!(matches!(
        complete(&mut fixture.runtime),
        EvalResult::Completed(_)
    ));
}

#[test]
fn unread_common_completion_bounds_admission_until_consumed() {
    let mut fixture = fixture(1);
    fixture.runtime.submit(request(1, 100)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.state().logical_requests, 0);
    assert_eq!(fixture.runtime.state().reserved_requests, 1);
    assert_eq!(
        fixture.runtime.state().reserved,
        Resources {
            host_bytes: 11,
            device_bytes: 13,
            pinned_bytes: 5
        }
    );
    assert_eq!(
        fixture.runtime.submit(request(2, 100)).unwrap_err().code,
        ErrorCode::ResourceExhausted
    );
    assert!(matches!(
        fixture.runtime.poll(),
        Some(EvalResult::Completed(_))
    ));
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    fixture.runtime.submit(request(3, 100)).unwrap();
    assert!(matches!(
        complete(&mut fixture.runtime),
        EvalResult::Completed(_)
    ));
}

#[test]
fn common_poll_revalidates_a_buffered_success_after_authority_changes() {
    for change in ["root", "deadline", "cancel"] {
        let mut fixture = fixture(1);
        fixture.runtime.submit(request(1, 100)).unwrap();
        fixture.runtime.pump();
        fixture.runtime.pump();
        assert_eq!(fixture.runtime.state().logical_requests, 0);
        assert_eq!(fixture.runtime.state().executions, 0);
        assert_eq!(fixture.runtime.state().reserved_requests, 1);
        match change {
            "root" => {
                let mut new_scope = scope();
                new_scope.root = RootGeneration(2);
                fixture.scope.update(new_scope);
            }
            "deadline" => fixture.clock.set(100),
            "cancel" => fixture.runtime.cancel(RequestId::new(EPOCH, 1)).unwrap(),
            _ => unreachable!(),
        }
        let result = fixture.runtime.poll().unwrap();
        let context = match (change, result) {
            ("root", EvalResult::Stale(context))
            | ("deadline", EvalResult::Expired(context))
            | ("cancel", EvalResult::Canceled(context)) => context,
            (_, other) => {
                panic!("buffered candidate must lose acceptance after {change}: {other:?}")
            }
        };
        assert_eq!(context.request.request, RequestId::new(EPOCH, 1));
        assert_eq!(
            context.execution,
            Some(fixture.backend.executions.lock().unwrap()[0])
        );
        assert_eq!(fixture.runtime.state().reserved_requests, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert!(fixture.runtime.poll().is_none());
    }
}

#[test]
fn root_scope_update_stales_inflight_request_and_disallows_late_success() {
    let mut fixture = fixture(1);
    let ready = Arc::new(AtomicBool::new(false));
    fixture.backend.plans.lock().unwrap().push_back(Plan {
        ready: Arc::clone(&ready),
        ..Plan::default()
    });
    fixture.runtime.submit(request(1, 100)).unwrap();
    fixture.runtime.pump();
    let mut new_scope = scope();
    new_scope.root = RootGeneration(2);
    fixture.scope.update(new_scope);
    fixture.runtime.pump();
    let EvalResult::Stale(context) = fixture.runtime.poll().unwrap() else {
        panic!("old root is stale")
    };
    assert_eq!(context.request.root, RootGeneration(1));
    assert_eq!(
        context.execution,
        Some(fixture.backend.executions.lock().unwrap()[0])
    );
    assert_eq!(fixture.runtime.state().reserved_requests, 1);
    assert_eq!(fixture.backend.lease_drops.load(Ordering::SeqCst), 0);
    ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    assert!(fixture.runtime.poll().is_none());
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
}
