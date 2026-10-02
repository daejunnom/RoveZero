//! C's simulator → D's physical Backend boundary. No new worker or scheduler.
//! A callback supplies owned candidate heads. Only DeviceCompleted releases a
//! lease; cancellation acknowledgement, failure and late callback do not.

use crate::{
    contracts::{backend_error, PreparedRequest},
    error::{BackendError, CauseCode, FailureKind as K, FailureStage as S},
    mock::{EventKind, InjectedFailure, MockError, ScriptedBackend},
    rules_projection::ClassicalProjection,
    RawOutput,
};
use rz_contracts::*;
use rz_position::contracts::RulesState;
use rz_runtime::contracts::{ContractClock, ContractsAdapter, RuntimeRequest};
use rz_runtime::{Backend, BackendResult, Resources};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    task::Poll,
    time::Duration,
};

pub type MockTicket = (ExecutionId, RequestId);
const DIAGNOSTIC_CAPACITY: usize = 128;

#[derive(Clone, Debug)]
pub struct BridgeDiagnostic {
    pub ticket: Option<MockTicket>,
    pub error: BackendError,
}
#[derive(Debug)]
pub struct BridgeDiagnosticSnapshot {
    pub entries: Vec<BridgeDiagnostic>,
    pub lost: u64,
    pub late_callbacks: u64,
    pub cancel_acknowledgements: u64,
    pub cancellation_attempts: u64,
    pub cancellations_forwarded: u64,
    pub cancellation_failures: u64,
}
#[derive(Default)]
struct DiagnosticStore {
    entries: VecDeque<BridgeDiagnostic>,
    lost: u64,
    late_callbacks: u64,
    cancel_acknowledgements: u64,
    cancellation_attempts: u64,
    cancellations_forwarded: u64,
    cancellation_failures: u64,
}
#[derive(Clone, Default)]
pub struct BridgeDiagnostics(Arc<Mutex<DiagnosticStore>>);
impl BridgeDiagnostics {
    pub fn take(&self) -> Result<BridgeDiagnosticSnapshot, ContractError> {
        let mut store = self.0.lock().map_err(|_| {
            failure(
                ErrorCode::BackendFailure,
                Stage::Backend,
                "mock diagnostic owner poisoned",
            )
        })?;
        Ok(BridgeDiagnosticSnapshot {
            entries: store.entries.drain(..).collect(),
            lost: store.lost,
            late_callbacks: store.late_callbacks,
            cancel_acknowledgements: store.cancel_acknowledgements,
            cancellation_attempts: store.cancellation_attempts,
            cancellations_forwarded: store.cancellations_forwarded,
            cancellation_failures: store.cancellation_failures,
        })
    }
    fn record(&self, ticket: Option<MockTicket>, error: BackendError) {
        if let Ok(mut store) = self.0.lock() {
            if store.entries.len() == DIAGNOSTIC_CAPACITY {
                store.lost = store.lost.saturating_add(1);
            } else {
                store.entries.push_back(BridgeDiagnostic { ticket, error });
            }
        }
    }
    fn late(&self) {
        if let Ok(mut store) = self.0.lock() {
            store.late_callbacks = store.late_callbacks.saturating_add(1);
        }
    }
    fn ack(&self) {
        if let Ok(mut store) = self.0.lock() {
            store.cancel_acknowledgements = store.cancel_acknowledgements.saturating_add(1);
        }
    }
    fn cancel_result(&self, result: &Result<bool, MockError>) {
        if let Ok(mut store) = self.0.lock() {
            store.cancellation_attempts = store.cancellation_attempts.saturating_add(1);
            match result {
                Ok(true) => {
                    store.cancellations_forwarded = store.cancellations_forwarded.saturating_add(1)
                }
                Err(_) => {
                    store.cancellation_failures = store.cancellation_failures.saturating_add(1)
                }
                Ok(false) => {}
            }
        }
    }
}

struct Receipt {
    ticket: MockTicket,
    candidate: Option<Result<RawOutput, InjectedFailure>>,
    duplicate: bool,
    device_done: bool,
    cancel_attempted: bool,
}
struct SharedState {
    script: ScriptedBackend<MockTicket>,
    receipts: Vec<Receipt>,
    last_execution: Option<u64>,
    seen_requests: Vec<RequestId>,
    max_identities: usize,
    max_receipts: usize,
}

#[derive(Clone)]
pub struct MockController<C: ContractClock + Send> {
    shared: Arc<Mutex<SharedState>>,
    clock: C,
    diagnostics: BridgeDiagnostics,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MockScriptStatus {
    pub scheduled_tickets: usize,
    pub scheduled_events: usize,
    pub physical_leases: usize,
}
impl<C: ContractClock + Send> MockController<C> {
    pub fn pump(&self) -> Result<(), ContractError> {
        let mut state = self.shared.lock().map_err(|_| {
            failure(
                ErrorCode::BackendFailure,
                Stage::Backend,
                "mock physical owner poisoned",
            )
        })?;
        drain_events(&mut state, self.clock.now(), &self.diagnostics)
    }
    pub fn script_status(&self) -> Result<MockScriptStatus, ContractError> {
        let state = self.shared.lock().map_err(|_| {
            failure(
                ErrorCode::BackendFailure,
                Stage::Backend,
                "mock physical owner poisoned",
            )
        })?;
        Ok(MockScriptStatus {
            scheduled_tickets: state.script.in_flight(),
            scheduled_events: state.script.pending_events(),
            physical_leases: state.receipts.len(),
        })
    }
}

/// Pins the prepared tensor and original Rules/request owner independently of
/// the Backend handle. Runtime quarantine can retain this exact owned lease.
pub struct ScriptedLease<C: ContractClock> {
    prepared: PreparedRequest<RulesState>,
    ticket: MockTicket,
    shared: Arc<Mutex<SharedState>>,
    results: Vec<BackendResult<ContractsAdapter<RulesState, C>>>,
    _clock: std::marker::PhantomData<C>,
    consumed: bool,
    quarantined: bool,
}

/// A single-item physical launch. Multi-item atomic launch is deliberately not
/// claimed: C's ScriptedBackend.submit atomically admits one ticket at a time.
pub struct ScriptedRuntimeBackend<C: ContractClock + Send> {
    shared: Arc<Mutex<SharedState>>,
    clock: C,
    projection: ClassicalProjection,
    diagnostics: BridgeDiagnostics,
}

impl<C: ContractClock + Send> ScriptedRuntimeBackend<C> {
    pub fn new(
        script: ScriptedBackend<MockTicket>,
        clock: C,
        projection: ClassicalProjection,
        model: Arc<ModelDescriptor>,
        backend: Digest,
    ) -> Result<Self, ContractError> {
        if model.handle() != projection.model().handle()
            || model.encoding() != projection.model().encoding()
            || model.full_steps() != 1
            || model.max_batch_items() != 1
            || !model.supports(PrecisionProfile::Fp32)
            || backend != projection.backend()
            || backend != Digest(script.identity_digest())
        {
            return Err(failure(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "mock script/model/encoding/backend binding differs",
            ));
        }
        let max_receipts = script.limits().max_in_flight;
        let max_identities = script.limits().max_steps;
        let mut seen_requests = Vec::new();
        seen_requests
            .try_reserve_exact(max_identities)
            .map_err(|_| {
                failure(
                    ErrorCode::ResourceExhausted,
                    Stage::Contract,
                    "mock identity ledger allocation failed",
                )
            })?;
        Ok(Self {
            shared: Arc::new(Mutex::new(SharedState {
                script,
                receipts: Vec::new(),
                last_execution: None,
                seen_requests,
                max_identities,
                max_receipts,
            })),
            clock,
            projection,
            diagnostics: BridgeDiagnostics::default(),
        })
    }
    pub fn diagnostics(&self) -> BridgeDiagnostics {
        self.diagnostics.clone()
    }
    pub fn controller(&self) -> MockController<C> {
        MockController {
            shared: Arc::clone(&self.shared),
            clock: self.clock.clone(),
            diagnostics: self.diagnostics.clone(),
        }
    }

    /// Nonlaunch maintenance for due late callbacks after D's last lease ended.
    /// D physical Drained does not assert that future simulator events drained.
    pub fn pump(&mut self) -> Result<(), ContractError> {
        self.controller().pump()
    }

    fn poll_inner(
        &mut self,
        lease: &mut ScriptedLease<C>,
    ) -> Poll<Vec<BackendResult<ContractsAdapter<RulesState, C>>>> {
        if lease.consumed || lease.quarantined {
            return Poll::Pending;
        }
        if !Arc::ptr_eq(&self.shared, &lease.shared) {
            lease.quarantined = true;
            return Poll::Pending;
        }
        let now = self.clock.now();
        let Ok(mut state) = lease.shared.lock() else {
            lease.quarantined = true;
            return Poll::Pending;
        };
        // First synchronize the simulator and observe already-due physical
        // events. cancel() schedules its acknowledgement relative to this now,
        // never relative to an earlier dispatch/poll tick.
        if let Err(error) = drain_events(&mut state, now, &self.diagnostics) {
            self.diagnostics.record(
                Some(lease.ticket),
                BackendError::new(K::BackendFailure, S::Backend, error.detail),
            );
            lease.quarantined = true;
            return Poll::Pending;
        }
        let Some(index) = state
            .receipts
            .iter()
            .position(|receipt| receipt.ticket == lease.ticket)
        else {
            lease.quarantined = true;
            return Poll::Pending;
        };
        if !state.receipts[index].device_done
            && !state.receipts[index].cancel_attempted
            && (lease.prepared.request().cancel_token().is_canceled()
                || lease
                    .prepared
                    .request()
                    .deadline()
                    .accepts(self.clock.domain(), now)
                    .is_err())
        {
            state.receipts[index].cancel_attempted = true;
            let result = state.script.cancel(&lease.ticket);
            self.diagnostics.cancel_result(&result);
            if let Err(error) = result {
                self.diagnostics.record(
                    Some(lease.ticket),
                    mock_failure(error, "mock cancellation forwarding failed"),
                );
            }
        }
        // Process zero-delay acknowledgements at the same actual tick. Logical
        // cancellation/deadline admission remains D/B's guard on every output.
        if let Err(error) = drain_events(&mut state, now, &self.diagnostics) {
            self.diagnostics.record(
                Some(lease.ticket),
                BackendError::new(K::BackendFailure, S::Backend, error.detail),
            );
            lease.quarantined = true;
            return Poll::Pending;
        }
        let Some(index) = state
            .receipts
            .iter()
            .position(|receipt| receipt.ticket == lease.ticket)
        else {
            lease.quarantined = true;
            return Poll::Pending;
        };
        if !state.receipts[index].device_done {
            return Poll::Pending;
        }
        let receipt = state.receipts.swap_remove(index);
        drop(state);
        let output = if receipt.duplicate {
            Err(failure(
                ErrorCode::BackendFailure,
                Stage::Output,
                "duplicate callback before physical completion",
            ))
        } else {
            match receipt.candidate {
                Some(Ok(raw)) => {
                    // Retain typed local numerical cause beside common result.
                    match crate::output::validate_maia(&raw, lease.prepared.indices()) {
                        Ok(_) => lease.prepared.output(&raw, lease.ticket.0),
                        Err(error) => {
                            let backend = BackendError::new(
                                K::NumericalFailure,
                                S::Output,
                                "invalid mock model heads",
                            )
                            .with_output_cause(&error);
                            let common = backend_error(&backend);
                            self.diagnostics.record(Some(lease.ticket), backend);
                            Err(common)
                        }
                    }
                }
                Some(Err(injected)) => {
                    let backend = injected_failure(&injected);
                    let common = backend_error(&backend);
                    self.diagnostics.record(Some(lease.ticket), backend);
                    Err(common)
                }
                None => Err(failure(
                    ErrorCode::BackendFailure,
                    Stage::Output,
                    "physical completion without a callback",
                )),
            }
        };
        lease.results.push(BackendResult {
            request_id: lease.ticket.1,
            output,
        });
        lease.consumed = true;
        Poll::Ready(std::mem::take(&mut lease.results))
    }
}

impl<C: ContractClock + Send> Backend<ContractsAdapter<RulesState, C>>
    for ScriptedRuntimeBackend<C>
{
    type Lease = ScriptedLease<C>;
    fn additional_resources(&self, requests: &[Arc<RuntimeRequest<RulesState>>]) -> Resources {
        Resources {
            host_bytes: (requests.len() as u64).saturating_mul(4096),
            device_bytes: 0,
            pinned_bytes: 0,
        }
    }
    fn dispatch(
        &mut self,
        execution: &ExecutionId,
        requests: &[Arc<RuntimeRequest<RulesState>>],
    ) -> Result<Self::Lease, ContractError> {
        if requests.len() != 1 {
            return Err(failure(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "mock bridge requires single-item dispatch",
            ));
        }
        let request = &requests[0];
        let id = request.eval().context().request;
        if execution.epoch != self.clock.domain().0 || id.epoch != execution.epoch {
            return Err(failure(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "mock execution/request epoch differs",
            ));
        }
        // Consume identities even if subsequent preparation rejects the request.
        let mut state = self.shared.lock().map_err(|_| {
            failure(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "mock physical owner poisoned",
            )
        })?;
        if state
            .last_execution
            .is_some_and(|last| execution.sequence <= last)
            || state.seen_requests.contains(&id)
        {
            return Err(failure(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "mock physical identity reused or moved backwards",
            ));
        }
        if state.seen_requests.len() >= state.max_identities {
            return Err(failure(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "mock lifetime request identity bound reached",
            ));
        }
        state.last_execution = Some(execution.sequence);
        state.seen_requests.push(id);
        drain_events(&mut state, self.clock.now(), &self.diagnostics)?;
        request
            .eval()
            .deadline()
            .accepts(self.clock.domain(), self.clock.now())?;
        if request.eval().cancel_token().is_canceled() {
            return Err(failure(
                ErrorCode::Canceled,
                Stage::Admission,
                "mock request canceled before launch",
            ));
        }
        if state.receipts.len() >= state.max_receipts {
            return Err(failure(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "mock receipt bound reached",
            ));
        }
        let prepared = self.projection.prepare(Arc::clone(request.eval()))?;
        state.receipts.try_reserve(1).map_err(|_| {
            failure(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "mock receipt allocation failed",
            )
        })?;
        let mut results = Vec::new();
        results.try_reserve_exact(1).map_err(|_| {
            failure(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "mock Ready storage allocation failed",
            )
        })?;
        let ticket = (*execution, id);
        drain_events(&mut state, self.clock.now(), &self.diagnostics)?;
        request
            .eval()
            .deadline()
            .accepts(self.clock.domain(), self.clock.now())?;
        if request.eval().cancel_token().is_canceled() {
            return Err(failure(
                ErrorCode::Canceled,
                Stage::Admission,
                "mock request canceled during preparation",
            ));
        }
        // All fallible preparation precedes this atomic simulator launch.
        state.script.submit(ticket).map_err(|error| {
            let backend = mock_failure(error, "mock simulator rejected launch");
            let common = backend_error(&backend);
            self.diagnostics.record(Some(ticket), backend);
            common
        })?;
        state.receipts.push(Receipt {
            ticket,
            candidate: None,
            duplicate: false,
            device_done: false,
            cancel_attempted: false,
        });
        Ok(ScriptedLease {
            prepared,
            ticket,
            shared: Arc::clone(&self.shared),
            results,
            _clock: std::marker::PhantomData,
            consumed: false,
            quarantined: false,
        })
    }
    fn poll(
        &mut self,
        lease: &mut Self::Lease,
    ) -> Poll<Vec<BackendResult<ContractsAdapter<RulesState, C>>>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.poll_inner(lease))) {
            Ok(result) => result,
            Err(_) => {
                lease.quarantined = true;
                Poll::Pending
            }
        }
    }
}

fn drain_events(
    state: &mut SharedState,
    now: MonotonicTick,
    diagnostics: &BridgeDiagnostics,
) -> Result<(), ContractError> {
    state
        .script
        .advance_to(Duration::from_nanos(now.0))
        .map_err(|error| {
            let backend = mock_failure(error, "mock clock cannot advance");
            let common = backend_error(&backend);
            diagnostics.record(None, backend);
            common
        })?;
    while let Some(event) = state.script.next_event() {
        match state
            .receipts
            .iter_mut()
            .find(|receipt| receipt.ticket == event.ticket)
        {
            Some(receipt) => match event.kind {
                EventKind::Callback(_) if receipt.device_done => diagnostics.late(),
                EventKind::Callback(candidate) if receipt.candidate.is_none() => {
                    receipt.candidate = Some(candidate)
                }
                EventKind::Callback(_) => receipt.duplicate = true,
                EventKind::DeviceCompleted => receipt.device_done = true,
                EventKind::CancelAcknowledged => diagnostics.ack(),
            },
            None => match event.kind {
                EventKind::Callback(_) => diagnostics.late(),
                EventKind::CancelAcknowledged => diagnostics.ack(),
                EventKind::DeviceCompleted => {}
            },
        }
    }
    Ok(())
}
fn failure(code: ErrorCode, stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(code, stage, detail)
}
fn mock_failure(error: MockError, detail: &'static str) -> BackendError {
    let kind = match error {
        MockError::ScriptExhausted
        | MockError::ScriptLimit
        | MockError::InFlightLimit
        | MockError::EventLimit
        | MockError::AllocationFailed
        | MockError::TimeOverflow
        | MockError::SequenceOverflow => K::ResourceExhausted,
        MockError::DuplicateTicket | MockError::UnknownTicket | MockError::ClockMovedBackwards => {
            K::IdentityMismatch
        }
        MockError::InvalidLimits | MockError::InvalidScriptIdentity => K::InvalidInput,
    };
    BackendError::new(kind, S::Backend, detail)
        .with_external_cause(CauseCode::MockLifecycle, &error)
}
fn injected_failure(error: &InjectedFailure) -> BackendError {
    let kind = match error.code.as_str() {
        "invalid_input" => K::InvalidInput,
        "unsupported_model" => K::UnsupportedModel,
        "identity_mismatch" => K::IdentityMismatch,
        "backend_unavailable" => K::BackendUnavailable,
        "resource_exhausted" => K::ResourceExhausted,
        "numerical_failure" => K::NumericalFailure,
        _ => K::BackendFailure,
    };
    let stage = match error.stage.as_str() {
        "asset" => S::Asset,
        "admission" => S::Admission,
        "output" => S::Output,
        _ => S::Backend,
    };
    struct Cause<'a>(&'a InjectedFailure);
    impl std::fmt::Display for Cause<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}:{}", self.0.code, self.0.stage)
        }
    }
    BackendError::new(kind, stage, "scripted evaluation failed")
        .with_diagnostic(&error.code, &error.stage)
        .with_external_cause(CauseCode::MockCallback, &Cause(error))
}
