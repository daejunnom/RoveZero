#![cfg(feature = "contracts")]

//! Injected physical workers exercise the actual D bridge; these are not NN runs.
use rz_contracts::*;
use rz_encoding::classical::HistoryFill;
use rz_eval::contracts::{encoding_manifest, MaiaBinding, HOST_BYTES_PER_ITEM};
use rz_eval::error::{BackendError, FailureKind, FailureStage, OutputCause};
use rz_eval::native_runtime_bridge::{
    NativeAdmissionPolicy, NativeDiagnosticKind, NativeRuntimeBackend, NativeWorkerOrigin,
    NativeWorkerOwner, NATIVE_CUDA_ADMISSION_BYTES,
};
use rz_eval::rules_projection::ClassicalProjection;
use rz_eval::worker::SingleWorker;
use rz_eval::{output, RawOutput};
use rz_position::contracts::{ContractPosition, RulesState};
use rz_position::Position;
use rz_runtime::contracts::{
    ContractClock, ContractEvaluator, ContractsAdapter, RuntimeRequest, SharedScope,
};
use rz_runtime::{Backend, Clock, DrainState, Limits, Resources};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc,
};
use std::task::Poll;
use std::time::{Duration, Instant};

const EPOCH: ProcessEpoch = ProcessEpoch(171);

#[derive(Clone, Default)]
struct ManualClock(Arc<AtomicU64>);
impl Clock for ManualClock {
    type Tick = MonotonicTick;
    fn now(&self) -> MonotonicTick {
        MonotonicTick(self.0.load(Ordering::Acquire))
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

fn projection() -> ClassicalProjection {
    let binding = MaiaBinding::new(
        ModelHandle {
            owner: OwnerId(172),
            slot: 1,
            generation: SlotGeneration(1),
            manifest: Digest([173; 32]),
        },
        EncodingHandle {
            owner: OwnerId(172),
            slot: 2,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(HistoryFill::No),
        },
        HistoryFill::No,
        Digest([174; 32]),
        1,
    )
    .unwrap();
    ClassicalProjection::new(binding)
}

fn request(
    projection: &ClassicalProjection,
    sequence: u64,
    deadline: u64,
) -> Arc<EvalRequest<RulesState>> {
    let live = ContractPosition::new(OwnerId(200 + sequence), Position::startpos());
    let state = live.export().unwrap();
    let context = EvalContext {
        revision: CONTRACT_REVISION,
        request: RequestId::new(EPOCH, sequence),
        selection: SelectionId::new(EPOCH, sequence),
        game: GameGeneration(1),
        root: RootGeneration(sequence),
        state: state.snapshot().identity(),
        legal_order: state.legal_moves().order(),
        input: projection
            .input_key(state.rules(), state.legal_moves().moves())
            .unwrap(),
        model: projection.model().handle(),
        encoding: projection.model().encoding().handle,
        precision: PrecisionProfile::Fp32,
        compute: ComputeBudget {
            min_steps: 1,
            max_steps: 1,
            require_full: true,
        },
        backend: projection.backend(),
    };
    Arc::new(
        EvalRequest::try_new(
            context,
            state.snapshot().clone(),
            state.legal_moves().clone(),
            Arc::clone(projection.model()),
            Deadline {
                clock: ClockDomain(EPOCH),
                at: MonotonicTick(deadline),
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

fn raw() -> RawOutput {
    RawOutput {
        policy_logits: vec![0.0; rz_encoding::POLICY_SIZE],
        wdl: vec![0.4, 0.3, 0.3],
    }
}

fn scope(context: EvalContext) -> AcceptanceScope {
    AcceptanceScope {
        game: context.game,
        root: context.root,
        model: context.model,
        encoding: context.encoding,
        backend: context.backend,
    }
}

fn limits() -> Limits {
    Limits {
        max_requests: 1,
        max_batch_items: 1,
        max_executions: 1,
        max_batch_wait: Duration::ZERO,
        max_queue_age: Duration::from_secs(1),
        deadline_reserve: Duration::ZERO,
        memory: Resources {
            host_bytes: 2 * HOST_BYTES_PER_ITEM,
            device_bytes: 0,
            pinned_bytes: 0,
        },
    }
}

fn wait_until(mut check: impl FnMut() -> bool) {
    let until = Instant::now() + Duration::from_secs(3);
    while !check() {
        assert!(Instant::now() < until, "finite physical worker test budget");
        std::thread::yield_now();
    }
}

#[test]
fn successful_physical_results_reuse_diagnostic_capacity_across_many_roots() {
    let projection = projection();
    let worker = SingleWorker::spawn(|batch: &rz_eval::contracts::PreparedBatch<RulesState>| {
        batch
            .requests()
            .iter()
            .map(|item| item.physical_output(&raw(), batch.execution()))
            .collect()
    })
    .unwrap();
    let owner = NativeWorkerOwner::from_worker(worker, projection.clone(), 1).unwrap();
    assert_eq!(owner.origin(), NativeWorkerOrigin::Injected);
    assert_eq!(owner.admission_policy(), NativeAdmissionPolicy::Cpu);
    assert!(owner.cuda_metadata().is_none());
    assert_eq!(
        owner.admission_policy().execution_resources().device_bytes,
        0
    );
    let clock = ManualClock::default();
    // More successes than B's 129 evaluations per root cannot exhaust the owner.
    for sequence in 1..=160 {
        let mut backend = NativeRuntimeBackend::new(owner.clone(), clock.clone()).unwrap();
        let eval = request(&projection, sequence, 100);
        let requests = [Arc::new(RuntimeRequest::new(eval))];
        let mut lease = backend
            .dispatch(&ExecutionId::new(EPOCH, sequence), &requests)
            .unwrap();
        wait_until(|| match backend.poll(&mut lease) {
            Poll::Pending => false,
            Poll::Ready(mut results) => {
                assert_eq!(results.len(), 1);
                let output = results.pop().unwrap().output.unwrap();
                assert_eq!(output.context.root, RootGeneration(sequence));
                assert_eq!(
                    output.actual.execution,
                    Some(ExecutionId::new(EPOCH, sequence))
                );
                true
            }
        });
        assert!(matches!(backend.poll(&mut lease), Poll::Pending));
    }
    let batch = owner.take_diagnostics().unwrap();
    assert!(batch.entries.is_empty());
    assert!(batch.boundary_error.is_none());
    assert!(owner.admission_error().is_none());
    let mut backend = NativeRuntimeBackend::new(owner, clock).unwrap();
    assert!(backend
        .dispatch(
            &ExecutionId::new(EPOCH, 161),
            &[Arc::new(RuntimeRequest::new(request(
                &projection,
                160,
                100
            )))]
        )
        .is_err());
}

#[test]
fn injected_cuda_admission_refuses_insufficient_device_budget_before_physical_launch() {
    let projection = projection();
    let calls = Arc::new(AtomicU64::new(0));
    let count = Arc::clone(&calls);
    let worker = SingleWorker::spawn(move |_: &rz_eval::contracts::PreparedBatch<RulesState>| {
        count.fetch_add(1, Ordering::AcqRel);
        Ok(Vec::new())
    })
    .unwrap();
    let owner = NativeWorkerOwner::from_worker_with_admission(
        worker,
        projection.clone(),
        1,
        NativeAdmissionPolicy::CudaOneGiB,
    )
    .unwrap();
    assert_eq!(owner.origin(), NativeWorkerOrigin::Injected);
    assert!(owner.cuda_metadata().is_none());
    assert_eq!(
        owner
            .admission_policy()
            .session_resident_admission()
            .device_bytes,
        NATIVE_CUDA_ADMISSION_BYTES
    );
    let clock = ManualClock::default();
    let eval = request(&projection, 1, 100);
    let adapter =
        ContractsAdapter::new(SharedScope::new(scope(eval.context())), clock.clone(), 1).unwrap();
    let backend = NativeRuntimeBackend::new(owner.clone(), clock).unwrap();
    let mut budget = limits();
    budget.memory.device_bytes = NATIVE_CUDA_ADMISSION_BYTES - 1;
    let mut runtime = ContractEvaluator::new(adapter, backend, budget, 64).unwrap();
    runtime.submit(eval).unwrap();
    match runtime
        .poll()
        .expect("device reservation failure is finalized")
    {
        EvalResult::Failed(failure) => assert_eq!(failure.error.code, ErrorCode::ResourceExhausted),
        other => panic!("expected prelaunch admission failure: {other:?}"),
    }
    assert_eq!(calls.load(Ordering::Acquire), 0);
    assert!(!owner.try_status().unwrap().unwrap().active);
    assert_eq!(runtime.state().reserved, Resources::default());
    assert!(owner.take_diagnostics().unwrap().entries.is_empty());
}

#[test]
fn injected_cuda_unknown_completion_retains_device_reservation_and_original_cause() {
    use rz_eval::error::CauseCode;
    use rz_eval::worker::PhysicalRun;
    let projection = projection();
    let original = BackendError::new(
        FailureKind::BackendFailure,
        FailureStage::Backend,
        "injected CUDA completion is unconfirmed",
    )
    .with_external_cause(CauseCode::OrtRun, &"authored device error");
    let cause = original.clone();
    let worker = SingleWorker::spawn_with_outcome(
        move |_: &rz_eval::contracts::PreparedBatch<RulesState>| -> PhysicalRun<Result<Vec<EvalOutput>, rz_eval::contracts::PhysicalFailure>> {
            PhysicalRun::Quarantined(cause.clone())
        },
    ).unwrap();
    let owner = NativeWorkerOwner::from_worker_with_admission(
        worker,
        projection.clone(),
        1,
        NativeAdmissionPolicy::CudaOneGiB,
    )
    .unwrap();
    assert_eq!(owner.origin(), NativeWorkerOrigin::Injected);
    let clock = ManualClock::default();
    let eval = request(&projection, 1, 100);
    let context = eval.context();
    let weak = Arc::downgrade(&eval);
    let adapter =
        ContractsAdapter::new(SharedScope::new(scope(context)), clock.clone(), 1).unwrap();
    let backend = NativeRuntimeBackend::new(owner.clone(), clock.clone()).unwrap();
    let mut budget = limits();
    budget.memory.device_bytes = NATIVE_CUDA_ADMISSION_BYTES;
    let mut runtime = ContractEvaluator::new(adapter, backend, budget, 64).unwrap();
    runtime.submit(eval).unwrap();
    wait_until(|| {
        assert!(
            runtime.poll().is_none(),
            "unknown completion cannot publish an output"
        );
        owner.admission_error().is_some()
    });
    assert!(owner.try_status().unwrap().unwrap().active);
    assert_eq!(
        runtime.state().reserved.device_bytes,
        NATIVE_CUDA_ADMISSION_BYTES
    );
    let mut diagnostics = owner.take_diagnostics().unwrap();
    assert_eq!(diagnostics.entries.len(), 1);
    let receipt = diagnostics.entries.pop().unwrap();
    assert_eq!(receipt.kind, NativeDiagnosticKind::Quarantined);
    assert_eq!(receipt.context.request, context);
    assert_eq!(receipt.context.execution, Some(ExecutionId::new(EPOCH, 1)));
    assert_eq!(receipt.failure.backend, Some(original));
    runtime
        .begin_shutdown(Deadline {
            clock: clock.domain(),
            at: MonotonicTick(1),
        })
        .unwrap();
    assert!(matches!(runtime.poll(), Some(EvalResult::Canceled(_))));
    clock.0.store(1, Ordering::Release);
    assert!(runtime.poll().is_none());
    assert!(matches!(
        runtime.shutdown_snapshot().drain,
        DrainState::TimedOut { .. }
    ));
    assert_eq!(
        runtime.state().reserved.device_bytes,
        NATIVE_CUDA_ADMISSION_BYTES
    );
    assert!(runtime.state().reserved_requests > 0);
    drop(runtime);
    assert!(
        weak.upgrade().is_some(),
        "D retains physically uncertain input pins"
    );
}

#[test]
fn deadline_keeps_physical_pins_and_preserves_a_late_native_failure() {
    let projection = projection();
    let (entered, entering) = mpsc::sync_channel(1);
    let (release, released) = mpsc::sync_channel(1);
    let worker = SingleWorker::spawn(move |_: &rz_eval::contracts::PreparedBatch<RulesState>| {
        entered.send(()).unwrap();
        released.recv_timeout(Duration::from_secs(3)).unwrap();
        Err(BackendError::new(
            FailureKind::NumericalFailure,
            FailureStage::Output,
            "injected physical output failure",
        )
        .with_output_cause(&output::OutputError::NonFinite {
            head: output::Head::Policy,
            index: 1857,
        })
        .into())
    })
    .unwrap();
    let owner = NativeWorkerOwner::from_worker(worker, projection.clone(), 2).unwrap();
    let clock = ManualClock::default();
    let eval = request(&projection, 1, 2);
    let original = eval.context();
    let weak = Arc::downgrade(&eval);
    let shared_scope = SharedScope::new(scope(original));
    let adapter = ContractsAdapter::new(shared_scope, clock.clone(), 1).unwrap();
    let backend = NativeRuntimeBackend::new(owner.clone(), clock.clone()).unwrap();
    let mut runtime = ContractEvaluator::new(adapter, backend, limits(), 64).unwrap();
    runtime.submit(eval).unwrap();
    assert!(runtime.poll().is_none());
    entering.recv_timeout(Duration::from_secs(3)).unwrap();
    clock.0.store(2, Ordering::Release);
    assert!(matches!(runtime.poll(), Some(EvalResult::Expired(_))));
    assert!(weak.upgrade().is_some());
    assert!(runtime.state().reserved_requests > 0);
    runtime
        .begin_shutdown(Deadline {
            clock: clock.domain(),
            at: MonotonicTick(100),
        })
        .unwrap();
    release.send(()).unwrap();
    wait_until(|| {
        assert!(
            runtime.poll().is_none(),
            "physical completion grants no second logical result"
        );
        runtime.shutdown_snapshot().drain == DrainState::Drained
    });
    assert_eq!(runtime.state().reserved_requests, 0);
    // Ready attests native completion and D reservation release. The physical
    // thread drops its final Job Arc immediately after sending that owned result.
    let until = Instant::now() + Duration::from_secs(2);
    while weak.upgrade().is_some() {
        assert!(Instant::now() < until, "completed physical Job Arc cleanup");
        std::thread::yield_now();
    }
    let mut batch = owner.take_diagnostics().unwrap();
    assert!(batch.boundary_error.is_none());
    assert_eq!(batch.entries.len(), 1);
    let receipt = batch.entries.pop().unwrap();
    assert_eq!(receipt.kind, NativeDiagnosticKind::PhysicalFailure);
    assert_eq!(receipt.context.request, original);
    assert_eq!(receipt.context.execution, Some(ExecutionId::new(EPOCH, 1)));
    assert_eq!(
        receipt.failure.backend.unwrap().cause.unwrap().output,
        Some(OutputCause::NonFinite {
            head: rz_eval::error::OutputHead::Policy,
            index: 1857
        })
    );
}

#[test]
fn full_diagnostic_store_refuses_before_launch_and_retains_original() {
    let projection = projection();
    let calls = Arc::new(AtomicU64::new(0));
    let count = Arc::clone(&calls);
    let worker = SingleWorker::spawn(move |_: &rz_eval::contracts::PreparedBatch<RulesState>| {
        count.fetch_add(1, Ordering::AcqRel);
        Err(BackendError::new(
            FailureKind::BackendFailure,
            FailureStage::Backend,
            "injected native failure",
        )
        .into())
    })
    .unwrap();
    let owner = NativeWorkerOwner::from_worker(worker, projection.clone(), 1).unwrap();
    let mut backend = NativeRuntimeBackend::new(owner.clone(), ManualClock::default()).unwrap();
    let mut lease = backend
        .dispatch(
            &ExecutionId::new(EPOCH, 1),
            &[Arc::new(RuntimeRequest::new(request(&projection, 1, 100)))],
        )
        .unwrap();
    wait_until(|| matches!(backend.poll(&mut lease), Poll::Ready(_)));
    let error = backend
        .dispatch(
            &ExecutionId::new(EPOCH, 2),
            &[Arc::new(RuntimeRequest::new(request(&projection, 2, 100)))],
        )
        .err()
        .unwrap();
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert_eq!(calls.load(Ordering::Acquire), 1);
    let mut batch = owner.take_diagnostics().unwrap();
    assert_eq!(
        batch.boundary_error.unwrap().code,
        ErrorCode::ResourceExhausted
    );
    assert_eq!(
        batch.entries.pop().unwrap().failure.backend.unwrap().detail,
        "injected native failure"
    );
    assert!(owner.admission_error().is_some());
}

#[test]
fn quarantine_is_pending_and_keeps_original_cause_and_physical_reservation() {
    let projection = projection();
    let worker = SingleWorker::spawn(|_: &rz_eval::contracts::PreparedBatch<RulesState>| -> Result<Vec<EvalOutput>, rz_eval::contracts::PhysicalFailure> {
        panic!("injected native unwind")
    }).unwrap();
    let owner = NativeWorkerOwner::from_worker(worker, projection.clone(), 1).unwrap();
    let clock = ManualClock::default();
    let eval = request(&projection, 1, 100);
    let weak = Arc::downgrade(&eval);
    let adapter =
        ContractsAdapter::new(SharedScope::new(scope(eval.context())), clock.clone(), 1).unwrap();
    let backend = NativeRuntimeBackend::new(owner.clone(), clock.clone()).unwrap();
    let mut runtime = ContractEvaluator::new(adapter, backend, limits(), 64).unwrap();
    runtime.submit(eval).unwrap();
    wait_until(|| {
        assert!(runtime.poll().is_none());
        owner.admission_error().is_some()
    });
    runtime
        .begin_shutdown(Deadline {
            clock: clock.domain(),
            at: MonotonicTick(1),
        })
        .unwrap();
    assert!(matches!(runtime.poll(), Some(EvalResult::Canceled(_))));
    clock.0.store(1, Ordering::Release);
    assert!(runtime.poll().is_none());
    assert!(matches!(
        runtime.shutdown_snapshot().drain,
        DrainState::TimedOut { .. }
    ));
    assert!(runtime.state().reserved_requests > 0);
    let mut batch = owner.take_diagnostics().unwrap();
    let receipt = batch.entries.pop().unwrap();
    assert_eq!(receipt.kind, NativeDiagnosticKind::Quarantined);
    let error = receipt.failure.backend.unwrap();
    assert_eq!(
        error.cause.unwrap().code,
        rz_eval::error::CauseCode::RuntimePanic
    );
    assert!(!format!("{error:?}").contains("injected native unwind"));
    drop(runtime); // D quarantines the unknown native Lease rather than freeing it.
    assert!(weak.upgrade().is_some());
}

#[test]
fn normal_cancel_audits_transfer_without_lifetime_exhaustion_or_native_launch() {
    let projection = projection();
    let calls = Arc::new(AtomicU64::new(0));
    let count = Arc::clone(&calls);
    let worker = SingleWorker::spawn(move |_: &rz_eval::contracts::PreparedBatch<RulesState>| {
        count.fetch_add(1, Ordering::AcqRel);
        Ok(Vec::new())
    })
    .unwrap();
    let owner = NativeWorkerOwner::from_worker(worker, projection.clone(), 1).unwrap();
    let mut backend = NativeRuntimeBackend::new(owner.clone(), ManualClock::default()).unwrap();
    for sequence in 1..=160 {
        let eval = request(&projection, sequence, 100);
        let original = eval.context();
        eval.cancel_token().cancel();
        assert_eq!(
            backend
                .dispatch(
                    &ExecutionId::new(EPOCH, sequence),
                    &[Arc::new(RuntimeRequest::new(eval))]
                )
                .err()
                .unwrap()
                .code,
            ErrorCode::Canceled
        );
        let mut batch = owner.try_take_diagnostics(1).unwrap().unwrap();
        assert_eq!(batch.entries.len(), 1);
        let receipt = batch.entries.pop().unwrap();
        assert_eq!(receipt.kind, NativeDiagnosticKind::DispatchRefused);
        assert_eq!(receipt.failure.contract.code, ErrorCode::Canceled);
        assert_eq!(receipt.context.request, original);
        assert!(receipt.context.execution.is_none());
        assert!(batch.boundary_error.is_none());
    }
    assert_eq!(calls.load(Ordering::Acquire), 0);
    assert!(owner.admission_error().is_none());
}

#[test]
fn limited_collection_keeps_reserved_pin_and_explicit_close_allows_physical_drain() {
    let projection = projection();
    let (entered, entering) = mpsc::sync_channel(1);
    let (release, released) = mpsc::sync_channel(1);
    let worker = SingleWorker::spawn(
        move |batch: &rz_eval::contracts::PreparedBatch<RulesState>| {
            entered.send(()).unwrap();
            released.recv_timeout(Duration::from_secs(3)).unwrap();
            batch
                .requests()
                .iter()
                .map(|item| item.physical_output(&raw(), batch.execution()))
                .collect()
        },
    )
    .unwrap();
    let owner = NativeWorkerOwner::from_worker(worker, projection.clone(), 2).unwrap();
    let mut backend = NativeRuntimeBackend::new(owner.clone(), ManualClock::default()).unwrap();
    let mut lease = backend
        .dispatch(
            &ExecutionId::new(EPOCH, 1),
            &[Arc::new(RuntimeRequest::new(request(&projection, 1, 100)))],
        )
        .unwrap();
    entering.recv_timeout(Duration::from_secs(3)).unwrap();
    let snapshot = owner.try_status().unwrap().unwrap();
    assert!(snapshot.active);
    assert_eq!(snapshot.reserved_diagnostics, 1);
    assert!(owner
        .try_take_diagnostics(1)
        .unwrap()
        .unwrap()
        .entries
        .is_empty());
    assert_eq!(owner.try_status().unwrap().unwrap().reserved_diagnostics, 1);
    let close = ContractError::new(
        ErrorCode::BackendFailure,
        Stage::Backend,
        "embedding closed admission after retaining native evidence",
    );
    assert!(owner.try_close_admission(close).unwrap());
    assert!(backend
        .dispatch(
            &ExecutionId::new(EPOCH, 2),
            &[Arc::new(RuntimeRequest::new(request(&projection, 2, 100)))]
        )
        .is_err());
    release.send(()).unwrap();
    wait_until(|| match backend.poll(&mut lease) {
        Poll::Pending => false,
        Poll::Ready(mut results) => {
            results.pop().unwrap().output.unwrap();
            true
        }
    });
    let batch = owner.try_take_diagnostics(0).unwrap().unwrap();
    assert_eq!(batch.boundary_error, Some(close));
    let snapshot = owner.try_status().unwrap().unwrap();
    assert!(!snapshot.active);
    assert_eq!(snapshot.reserved_diagnostics, 0);
}
