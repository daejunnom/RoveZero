//! TASK-D01 adapter for coordinator-owned contract revision 0.1.
//!
//! Rules snapshots and all shared identifiers/results remain `rz-contracts`
//! types. This module adds scheduling metadata and a bounded evaluator mailbox.

use crate::observation::Observations;
use crate::{
    Adapter, Backend, Clock, CompletionReceiver, DrainState, Limits, Resources, RuntimeFault,
    Scheduler, SchedulerObservations, ShutdownSnapshot, State, TerminalEvent,
};
use rz_contracts::*;
use rz_telemetry::{FinishKind, Snapshot};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{mpsc, Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// Clones must share one immutable domain and the same monotonic origin.
pub trait ContractClock: Clock<Tick = MonotonicTick> + Clone + 'static {
    fn domain(&self) -> ClockDomain;
}

#[derive(Clone, Debug)]
pub struct ContractSystemClock {
    domain: ClockDomain,
    origin: Arc<Instant>,
}

impl ContractSystemClock {
    pub fn new(epoch: ProcessEpoch) -> Self {
        Self {
            domain: ClockDomain(epoch),
            origin: Arc::new(Instant::now()),
        }
    }

    pub fn try_now(&self) -> Result<MonotonicTick, ContractError> {
        u64::try_from(self.origin.elapsed().as_nanos())
            .map(MonotonicTick)
            .map_err(|_| {
                ContractError::new(
                    ErrorCode::ResourceExhausted,
                    Stage::Contract,
                    "monotonic clock epoch exhausted",
                )
            })
    }
}

impl Clock for ContractSystemClock {
    type Tick = MonotonicTick;

    fn now(&self) -> MonotonicTick {
        // Exhaustion closes every representable deadline rather than wrapping.
        self.try_now().unwrap_or(MonotonicTick(u64::MAX))
    }

    fn elapsed(&self, earlier: MonotonicTick, later: MonotonicTick) -> Duration {
        Duration::from_nanos(later.0.saturating_sub(earlier.0))
    }
}

impl ContractClock for ContractSystemClock {
    fn domain(&self) -> ClockDomain {
        self.domain
    }
}

/// Acceptance authority shared with the owner of root/game/registry changes.
#[derive(Clone, Debug)]
pub struct SharedScope(Arc<RwLock<AcceptanceScope>>);

impl SharedScope {
    pub fn new(scope: AcceptanceScope) -> Self {
        Self(Arc::new(RwLock::new(scope)))
    }

    pub fn get(&self) -> AcceptanceScope {
        *self.0.read().expect("acceptance scope")
    }

    /// Generation issuance and draining old model owners remain caller duties.
    pub fn update(&self, scope: AcceptanceScope) {
        *self.0.write().expect("acceptance scope") = scope;
    }
}

/// Runtime-owned physical metadata; the immutable common request is untouched.
pub struct RuntimeRequest<P> {
    eval: Arc<EvalRequest<P>>,
    execution: Arc<OnceLock<ExecutionId>>,
}

impl<P> Clone for RuntimeRequest<P> {
    fn clone(&self) -> Self {
        Self {
            eval: Arc::clone(&self.eval),
            execution: Arc::clone(&self.execution),
        }
    }
}

impl<P> RuntimeRequest<P> {
    pub fn new(eval: Arc<EvalRequest<P>>) -> Self {
        Self {
            eval,
            execution: Arc::new(OnceLock::new()),
        }
    }

    pub fn eval(&self) -> &Arc<EvalRequest<P>> {
        &self.eval
    }

    pub fn execution(&self) -> Option<ExecutionId> {
        self.execution.get().copied()
    }

    fn completion_context(&self) -> CompletionContext {
        CompletionContext {
            request: self.eval.context(),
            execution: self.execution(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContractBatchKey {
    model: ModelHandle,
    encoding: EncodingHandle,
    precision: PrecisionProfile,
    compute: ComputeBudget,
    backend: Digest,
    legal_count: usize,
}

/// Bounded ID issuance: one process epoch and two sequence high-water marks.
pub struct ContractsAdapter<P, C> {
    scope: SharedScope,
    clock: C,
    epoch: ProcessEpoch,
    last_request: Option<u64>,
    last_execution: u64,
    max_batch_items: usize,
    _position: PhantomData<fn() -> P>,
}

impl<P, C: ContractClock> ContractsAdapter<P, C> {
    pub fn new(
        scope: SharedScope,
        clock: C,
        max_batch_items: usize,
    ) -> Result<Self, ContractError> {
        Self::with_execution_high_water(scope, clock, max_batch_items, 0)
    }

    pub fn with_execution_high_water(
        scope: SharedScope,
        clock: C,
        max_batch_items: usize,
        last_execution: u64,
    ) -> Result<Self, ContractError> {
        if max_batch_items == 0 {
            return Err(fault_error(RuntimeFault::InvalidLimits));
        }
        Ok(Self {
            scope,
            epoch: clock.domain().0,
            clock,
            last_request: None,
            last_execution,
            max_batch_items,
            _position: PhantomData,
        })
    }

    pub fn scope(&self) -> &SharedScope {
        &self.scope
    }

    fn failure(&self, request: &RuntimeRequest<P>, error: ContractError) -> EvalResult {
        failure_result(request.completion_context(), error)
    }
}

impl<P: Send + Sync + 'static, C: ContractClock> Adapter for ContractsAdapter<P, C> {
    type RequestId = RequestId;
    type ExecutionId = ExecutionId;
    type Tick = MonotonicTick;
    type Request = RuntimeRequest<P>;
    type BatchKey = ContractBatchKey;
    type Output = EvalOutput;
    type Error = ContractError;
    type TerminalResult = EvalResult;

    fn request_id(&self, request: &Self::Request) -> RequestId {
        request.eval.context().request
    }

    fn deadline(&self, request: &Self::Request) -> MonotonicTick {
        request.eval.deadline().at
    }

    fn batch_key(&self, request: &Self::Request) -> ContractBatchKey {
        let context = request.eval.context();
        ContractBatchKey {
            model: context.model,
            encoding: context.encoding,
            precision: context.precision,
            compute: context.compute,
            backend: context.backend,
            legal_count: request.eval.legal().moves().len(),
        }
    }

    fn resources(&self, request: &Self::Request) -> Resources {
        let bytes = request.eval.byte_budget();
        Resources {
            host_bytes: bytes.host,
            device_bytes: bytes.device,
            pinned_bytes: bytes.pinned,
        }
    }

    fn validate_admission(&mut self, request: &Self::Request) -> Result<(), ContractError> {
        let eval = &request.eval;
        let context = eval.context();
        context.revision.validate()?;
        if context.request.epoch != self.epoch
            || context.selection.epoch != self.epoch
            || self.clock.domain() != ClockDomain(self.epoch)
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "request does not belong to runtime process epoch",
            ));
        }
        if self
            .last_request
            .is_some_and(|last| context.request.sequence <= last)
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "request sequence is reused or not strictly increasing",
            ));
        }
        // The direct Scheduler adapter also consumes refused IDs. Corrected
        // inputs and retries are new logical requests with new identifiers.
        self.last_request = Some(context.request.sequence);
        if eval.model().max_batch_items() < self.max_batch_items {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "configured batch limit exceeds validated model limit",
            ));
        }
        eval.validate_acceptance(self.scope.get(), self.clock.domain(), self.clock.now())?;
        Ok(())
    }

    fn is_current(&self, request: &Self::Request) -> bool {
        let context = request.eval.context();
        let scope = self.scope.get();
        context.game == scope.game
            && context.root == scope.root
            && context.model == scope.model
            && context.encoding == scope.encoding
            && context.backend == scope.backend
    }

    fn is_canceled(&self, request: &Self::Request) -> bool {
        request.eval.cancel_token().is_canceled()
    }

    fn validate_output(
        &mut self,
        request: &Self::Request,
        output: &EvalOutput,
    ) -> Result<(), ContractError> {
        output.validate_for(
            &request.eval,
            self.scope.get(),
            self.clock.domain(),
            self.clock.now(),
        )?;
        if output.actual.provenance != CacheProvenance::Computed {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "fresh runtime does not accept cache or feature reuse",
            ));
        }
        if request.execution().is_none() || output.actual.execution != request.execution() {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "response does not echo the bound physical execution",
            ));
        }
        Ok(())
    }

    fn validate_execution(
        &mut self,
        request: &Self::Request,
        expected: &ExecutionId,
        output: &EvalOutput,
    ) -> Result<(), ContractError> {
        if output.actual.execution != Some(*expected) {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "response physical execution identity differs",
            ));
        }
        self.validate_output(request, output)
    }

    #[cfg(feature = "experimental-raw-cache")]
    fn validate_reused_output(
        &mut self,
        request: &Self::Request,
        output: &EvalOutput,
    ) -> Result<(), ContractError> {
        output.validate_for(
            &request.eval,
            self.scope.get(),
            self.clock.domain(),
            self.clock.now(),
        )?;
        if request.execution().is_some()
            || output.actual.execution.is_some()
            || !matches!(output.actual.provenance, CacheProvenance::RawEvalHit { .. })
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "raw reuse must not bind a new physical execution",
            ));
        }
        Ok(())
    }

    fn bind_execution(&mut self, request: &Self::Request, id: &ExecutionId) {
        request
            .execution
            .set(*id)
            .expect("one physical execution per fresh request");
    }

    fn allocate_execution_id(&mut self) -> Result<ExecutionId, ContractError> {
        self.last_execution = self.last_execution.checked_add(1).ok_or_else(|| {
            ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Backend,
                "physical execution sequence exhausted",
            )
        })?;
        Ok(ExecutionId::new(self.epoch, self.last_execution))
    }

    fn runtime_error(&self, fault: RuntimeFault) -> ContractError {
        fault_error(fault)
    }

    fn execution_error(&self, _: &Self::Request, error: &ContractError) -> ContractError {
        *error
    }

    fn terminal(
        &mut self,
        request: &Self::Request,
        event: TerminalEvent<EvalOutput, ContractError>,
    ) -> EvalResult {
        let context = request.completion_context();
        match event {
            TerminalEvent::Success(output) => EvalResult::Completed(output),
            TerminalEvent::Canceled => EvalResult::Canceled(context),
            TerminalEvent::Expired => EvalResult::Expired(context),
            TerminalEvent::Stale => EvalResult::Stale(context),
            TerminalEvent::Failure(error) => self.failure(request, error),
        }
    }
}

struct Pending<P> {
    request: Arc<RuntimeRequest<P>>,
    receiver: CompletionReceiver<EvalResult>,
}

#[cfg(feature = "experimental-raw-cache")]
pub trait ExactRawReuse<P>: Send {
    fn lookup(&mut self, request: Arc<EvalRequest<P>>)
        -> Result<Option<EvalOutput>, ContractError>;
    /// Only the final runtime acceptance promotes a staged raw result.
    fn observe(&mut self, request: &EvalRequest<P>, output: Option<&EvalOutput>);
}

/// Facts at the common result transfer boundary, separate from Scheduler facts.
/// Accepted means the completed candidate passed this boundary's checks. It
/// grants no new physical completion, current-scope or Search backup authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliveryObservationKind {
    Accepted,
    /// A completed mailbox candidate failed the common delivery acceptance check.
    Rejected {
        error: ContractError,
    },
    /// A non-completed mailbox result was transferred without revalidation.
    /// Canceled/Expired/Stale carry their cause in `kind`; Failed preserves error.
    Terminal {
        kind: FinishKind,
        error: Option<ContractError>,
    },
    MailboxDisconnected {
        error: ContractError,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeliveryObservation {
    /// Owner time when recording the transfer decision, after acceptance checks.
    /// This is neither backend source time nor Search's final consume time.
    pub at: MonotonicTick,
    pub clock: ClockDomain,
    /// Original admitted request context and its bound execution, if launched.
    /// Does not retain the position, policy, model owner or physical pins.
    pub context: CompletionContext,
    pub kind: DeliveryObservationKind,
}

/// A separate bounded stream; loss counters describe this drain only.
/// Nonzero loss makes reconstruction of common delivery decisions incomplete.
#[derive(Debug)]
pub struct DeliveryObservationDrain {
    pub events: Vec<DeliveryObservation>,
    pub dropped: u64,
    pub counter_overflow: bool,
}

/// A bounded, exactly-once bridge to the common logical Evaluator interface.
///
/// CompletionReceiver pins stay in `pending` until common poll transfers the
/// payload to its caller. A second acceptance check at that transfer boundary
/// rejects a candidate completed before a root/deadline/cancel change. It never
/// changes the Scheduler's already finalized internal terminal accounting.
pub struct ContractEvaluator<P, B, C>
where
    P: Send + Sync + 'static,
    C: ContractClock,
    B: Backend<ContractsAdapter<P, C>>,
{
    scheduler: Scheduler<ContractsAdapter<P, C>, B, C>,
    pending: VecDeque<Pending<P>>,
    max_requests: usize,
    last_submitted_request: Option<u64>,
    scope: SharedScope,
    clock: C,
    delivery_observations: Observations<DeliveryObservation>,
    #[cfg(feature = "experimental-raw-cache")]
    raw_reuse: Option<Box<dyn ExactRawReuse<P>>>,
}

impl<P, B, C> ContractEvaluator<P, B, C>
where
    P: Send + Sync + 'static,
    C: ContractClock,
    B: Backend<ContractsAdapter<P, C>>,
{
    pub fn new(
        adapter: ContractsAdapter<P, C>,
        backend: B,
        limits: Limits,
        metric_sample_capacity: usize,
    ) -> Result<Self, ContractError> {
        if adapter.max_batch_items != limits.max_batch_items {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "adapter and scheduler batch limits differ",
            ));
        }
        let scope = adapter.scope.clone();
        let clock = adapter.clock.clone();
        let scheduler = Scheduler::new(
            adapter,
            backend,
            clock.clone(),
            limits,
            metric_sample_capacity,
        )
        .map_err(fault_error)?;
        let delivery_observations =
            Observations::try_new(metric_sample_capacity).map_err(fault_error)?;
        Ok(Self {
            scheduler,
            pending: VecDeque::new(),
            max_requests: limits.max_requests,
            last_submitted_request: None,
            scope,
            clock,
            delivery_observations,
            #[cfg(feature = "experimental-raw-cache")]
            raw_reuse: None,
        })
    }

    pub fn pump(&mut self) {
        self.scheduler.pump();
    }

    #[cfg(feature = "experimental-raw-cache")]
    pub fn set_raw_reuse(&mut self, reuse: Box<dyn ExactRawReuse<P>>) {
        self.raw_reuse = Some(reuse);
    }

    pub fn state(&self) -> State {
        self.scheduler.state()
    }

    pub fn metrics(&self) -> Snapshot {
        self.scheduler.metrics()
    }

    /// Drain bounded scheduler facts without consuming common result receipts.
    /// Event loss is explicit; these facts do not authorize search backup or
    /// describe the separate final acceptance performed by common `poll`.
    /// The caller records wrapper refusals that occur before scheduler admission.
    pub fn take_observations(&mut self) -> SchedulerObservations<ContractsAdapter<P, C>> {
        self.scheduler.take_observations()
    }

    /// Drain passive common-poll decisions without consuming any result receipt.
    /// Capacity equals metric_sample_capacity independently of Scheduler events.
    /// Overflow drops the oldest fact, including at zero capacity, and never
    /// changes result acceptance, reservations, cancellation or lease ownership.
    /// Submission refusals and Search backup remain the caller's own boundaries.
    pub fn take_delivery_observations(&mut self) -> DeliveryObservationDrain {
        let drain = self.delivery_observations.drain();
        DeliveryObservationDrain {
            events: drain.events,
            dropped: drain.dropped,
            counter_overflow: drain.counter_overflow,
        }
    }

    fn observe_delivery(&mut self, request: &RuntimeRequest<P>, kind: DeliveryObservationKind) {
        self.delivery_observations.push(DeliveryObservation {
            at: self.clock.now(),
            clock: self.clock.domain(),
            context: request.completion_context(),
            kind,
        });
    }

    pub fn begin_shutdown(&mut self, deadline: Deadline) -> Result<(), ContractError> {
        if deadline.clock != self.clock.domain() {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "drain deadline clock domain mismatch",
            ));
        }
        self.scheduler.begin_shutdown(deadline.at);
        Ok(())
    }

    pub fn drain_state(&self) -> DrainState {
        self.scheduler.drain_state()
    }

    /// Physical drain can finish while unread common results still own their
    /// delivery reservations. Report both facts from one scheduler snapshot.
    pub fn shutdown_snapshot(&self) -> ShutdownSnapshot {
        self.scheduler.shutdown_snapshot()
    }

    fn accept_delivery(&mut self, request: &RuntimeRequest<P>, result: EvalResult) -> EvalResult {
        let EvalResult::Completed(output) = result else {
            #[cfg(feature = "experimental-raw-cache")]
            if let Some(reuse) = self.raw_reuse.as_mut() {
                reuse.observe(&request.eval, None);
            }
            let (kind, error) = match &result {
                EvalResult::Canceled(_) => (FinishKind::Canceled, None),
                EvalResult::Expired(_) => (FinishKind::Expired, None),
                EvalResult::Stale(_) => (FinishKind::Stale, None),
                EvalResult::Failed(failure) => (FinishKind::Failed, Some(failure.error)),
                EvalResult::Completed(_) => unreachable!("completed candidate matched above"),
            };
            self.observe_delivery(request, DeliveryObservationKind::Terminal { kind, error });
            return result;
        };
        let checked = output
            .validate_for(
                &request.eval,
                self.scope.get(),
                self.clock.domain(),
                self.clock.now(),
            )
            .and_then(|()| {
                request.eval.validate_acceptance(
                    self.scope.get(),
                    self.clock.domain(),
                    self.clock.now(),
                )
            });
        match checked {
            Ok(()) => {
                #[cfg(feature = "experimental-raw-cache")]
                if let Some(reuse) = self.raw_reuse.as_mut() {
                    reuse.observe(&request.eval, Some(&output));
                }
                self.observe_delivery(request, DeliveryObservationKind::Accepted);
                EvalResult::Completed(output)
            }
            Err(error) => {
                #[cfg(feature = "experimental-raw-cache")]
                if let Some(reuse) = self.raw_reuse.as_mut() {
                    reuse.observe(&request.eval, None);
                }
                self.observe_delivery(request, DeliveryObservationKind::Rejected { error });
                failure_result(request.completion_context(), error)
            }
        }
    }
}

impl<P, B, C> Evaluator<P> for ContractEvaluator<P, B, C>
where
    P: Send + Sync + 'static,
    C: ContractClock,
    B: Backend<ContractsAdapter<P, C>>,
{
    fn submit(&mut self, request: Arc<EvalRequest<P>>) -> Result<(), ContractError> {
        let id = request.context().request;
        if id.epoch != self.clock.domain().0
            || self
                .last_submitted_request
                .is_some_and(|last| id.sequence <= last)
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "submitted request epoch or monotonic sequence is invalid",
            ));
        }
        // Even refused submissions consume their logical ID. Queue fullness
        // must not create a loophole for retries/reuse outside adapter admission.
        self.last_submitted_request = Some(id.sequence);
        if self.pending.len() >= self.max_requests {
            return Err(fault_error(RuntimeFault::QueueFull));
        }
        #[cfg(feature = "experimental-raw-cache")]
        let reused = if let Some(reuse) = self.raw_reuse.as_mut() {
            request.validate_acceptance(self.scope.get(), self.clock.domain(), self.clock.now())?;
            reuse.lookup(Arc::clone(&request))?
        } else {
            None
        };
        let request = Arc::new(RuntimeRequest::new(request));
        // Both owners share one physical bind, while the common request remains frozen.
        let scheduled = request.as_ref().clone();
        #[cfg(feature = "experimental-raw-cache")]
        let receiver = match match reused {
            Some(output) => self.scheduler.submit_reused(scheduled, output),
            None => self.scheduler.submit(scheduled),
        } {
            Ok(receiver) => receiver,
            Err(error) => {
                if let Some(reuse) = self.raw_reuse.as_mut() {
                    reuse.observe(request.eval(), None);
                }
                return Err(error.error);
            }
        };
        #[cfg(not(feature = "experimental-raw-cache"))]
        let receiver = self
            .scheduler
            .submit(scheduled)
            .map_err(|error| error.error)?;
        self.pending.push_back(Pending { request, receiver });
        Ok(())
    }

    fn poll(&mut self) -> Option<EvalResult> {
        self.scheduler.pump();
        for _ in 0..self.pending.len() {
            let pending = self.pending.pop_front().expect("bounded pending length");
            match pending.receiver.try_recv() {
                Ok(result) => return Some(self.accept_delivery(&pending.request, result)),
                Err(mpsc::TryRecvError::Empty) => self.pending.push_back(pending),
                Err(mpsc::TryRecvError::Disconnected) => {
                    #[cfg(feature = "experimental-raw-cache")]
                    if let Some(reuse) = self.raw_reuse.as_mut() {
                        reuse.observe(pending.request.eval(), None);
                    }
                    let error = ContractError::new(
                        ErrorCode::BackendFailure,
                        Stage::Output,
                        "runtime completion mailbox disconnected",
                    );
                    self.observe_delivery(
                        &pending.request,
                        DeliveryObservationKind::MailboxDisconnected { error },
                    );
                    return Some(failure_result(pending.request.completion_context(), error));
                }
            }
        }
        None
    }

    fn cancel(&mut self, request: RequestId) -> Result<(), ContractError> {
        if request.epoch != self.clock.domain().0 {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "cancel request belongs to another process epoch",
            ));
        }
        if let Some(pending) = self
            .pending
            .iter()
            .find(|pending| pending.request.eval.context().request == request)
        {
            pending.request.eval.cancel_token().cancel();
        }
        self.scheduler.cancel(&request);
        Ok(())
    }
}

fn failure_result(context: CompletionContext, error: ContractError) -> EvalResult {
    match error.code {
        ErrorCode::Canceled => EvalResult::Canceled(context),
        ErrorCode::Expired => EvalResult::Expired(context),
        ErrorCode::Stale => EvalResult::Stale(context),
        _ => EvalResult::Failed(EvalFailure {
            context,
            error,
            recovery: RecoveryOutcome::NotAttempted,
        }),
    }
}

fn fault_error(fault: RuntimeFault) -> ContractError {
    let (code, stage, detail) = match fault {
        RuntimeFault::InvalidLimits => (
            ErrorCode::InvalidInput,
            Stage::Admission,
            "invalid runtime limits",
        ),
        RuntimeFault::Closed => (
            ErrorCode::Canceled,
            Stage::Admission,
            "runtime admission is closed",
        ),
        RuntimeFault::QueueFull => (
            ErrorCode::ResourceExhausted,
            Stage::Admission,
            "runtime request budget is full",
        ),
        RuntimeFault::DuplicateRequest => (
            ErrorCode::IdentityMismatch,
            Stage::Admission,
            "request identity is already active",
        ),
        RuntimeFault::ResourceOverflow => (
            ErrorCode::ResourceExhausted,
            Stage::Admission,
            "runtime resource arithmetic overflow",
        ),
        RuntimeFault::ResourceLimit => (
            ErrorCode::ResourceExhausted,
            Stage::Admission,
            "runtime resource limit exceeded",
        ),
        RuntimeFault::QueueAgeExceeded => (
            ErrorCode::ResourceExhausted,
            Stage::Admission,
            "runtime queue age exceeded",
        ),
        RuntimeFault::Canceled => (
            ErrorCode::Canceled,
            Stage::Admission,
            "runtime request canceled",
        ),
        RuntimeFault::Expired => (
            ErrorCode::Expired,
            Stage::Admission,
            "runtime request deadline expired",
        ),
        RuntimeFault::Stale => (
            ErrorCode::Stale,
            Stage::Admission,
            "runtime request is stale",
        ),
        RuntimeFault::InvalidCompletion => (
            ErrorCode::IdentityMismatch,
            Stage::Output,
            "backend completion ID set is invalid",
        ),
    };
    ContractError::new(code, stage, detail)
}
