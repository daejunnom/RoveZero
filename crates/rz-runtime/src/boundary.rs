use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

/// Scheduler-owned byte reservations, not a shared model/config schema.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Resources {
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub pinned_bytes: u64,
}

impl Resources {
    pub(crate) fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            host_bytes: self.host_bytes.checked_add(other.host_bytes)?,
            device_bytes: self.device_bytes.checked_add(other.device_bytes)?,
            pinned_bytes: self.pinned_bytes.checked_add(other.pinned_bytes)?,
        })
    }

    pub(crate) fn subtract(self, other: Self) -> Self {
        // Every release corresponds to an owner that successfully reserved.
        Self {
            host_bytes: self
                .host_bytes
                .checked_sub(other.host_bytes)
                .expect("host reservation"),
            device_bytes: self
                .device_bytes
                .checked_sub(other.device_bytes)
                .expect("device reservation"),
            pinned_bytes: self
                .pinned_bytes
                .checked_sub(other.pinned_bytes)
                .expect("pinned reservation"),
        }
    }

    pub(crate) fn fits(self, limit: Self) -> bool {
        self.host_bytes <= limit.host_bytes
            && self.device_bytes <= limit.device_bytes
            && self.pinned_bytes <= limit.pinned_bytes
    }
}

/// Finite scheduling limits selected by the embedding execution manifest.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_requests: usize,
    pub max_batch_items: usize,
    pub max_executions: usize,
    pub max_batch_wait: Duration,
    pub max_queue_age: Duration,
    pub deadline_reserve: Duration,
    pub memory: Resources,
}

impl Limits {
    pub fn validate(self) -> Result<Self, RuntimeFault> {
        if self.max_requests == 0
            || self.max_batch_items == 0
            || self.max_batch_items > self.max_requests
            || self.max_executions == 0
            || self.max_queue_age.is_zero()
            || self.max_batch_wait > self.max_queue_age
        {
            return Err(RuntimeFault::InvalidLimits);
        }
        Ok(self)
    }
}

/// Local scheduler failure facts. The adapter maps these into shared errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeFault {
    InvalidLimits,
    Closed,
    QueueFull,
    DuplicateRequest,
    ResourceOverflow,
    ResourceLimit,
    QueueAgeExceeded,
    Canceled,
    Expired,
    Stale,
    InvalidCompletion,
}

/// Logical facts passed to the adapter, not an EvalResult wire schema.
#[derive(Debug)]
pub enum TerminalEvent<O, E> {
    Success(O),
    Canceled,
    Expired,
    Stale,
    Failure(E),
}

/// Shared types and semantics remain with TASK-I01 and its consumers.
///
/// Request metadata must be immutable. `validate_admission` must validate the
/// shared contract and provenance of a fresh RequestId; IDs must never be reused.
/// It is called exactly once for every `submit`, including submissions refused
/// because admission is closed, the ID is active, or the queue is full. It may
/// consume fresh IDs, but must remain bounded and must not reserve scheduler
/// resources, launch work, or publish a terminal result. Local scheduling faults
/// take precedence over its returned error when both refuse the submission.
/// `is_current` includes game/root/model/encoding/state/slot generations.
/// `validate_output` checks request identity, legal order, required heads and
/// numerical admissibility. `terminal` must be a short, infallible construction;
/// it must not perform search backup or invoke arbitrary user callbacks.
pub trait Adapter {
    type RequestId: Clone + Debug + Eq + Hash;
    type ExecutionId: Clone + Debug;
    type Tick: Copy + Debug + Ord;
    type Request: Send + Sync + 'static;
    type BatchKey: Clone + Eq;
    type Output: Send + 'static;
    type Error: Send + 'static;
    type TerminalResult: Send + 'static;

    fn request_id(&self, request: &Self::Request) -> Self::RequestId;
    fn deadline(&self, request: &Self::Request) -> Self::Tick;
    fn batch_key(&self, request: &Self::Request) -> Self::BatchKey;
    fn resources(&self, request: &Self::Request) -> Resources;
    fn validate_admission(&mut self, request: &Self::Request) -> Result<(), Self::Error>;
    fn is_current(&self, request: &Self::Request) -> bool;
    fn is_canceled(&self, request: &Self::Request) -> bool;
    fn validate_output(
        &mut self,
        request: &Self::Request,
        output: &Self::Output,
    ) -> Result<(), Self::Error>;
    /// Bind only after successful launch. Keep this infallible and bounded;
    /// failure to record metadata cannot undo already-running physical work.
    fn bind_execution(&mut self, _request: &Self::Request, _id: &Self::ExecutionId) {}
    /// Validate the actual physical identity as well as the logical payload.
    /// Adapters whose output has no execution identity may use the default.
    fn validate_execution(
        &mut self,
        request: &Self::Request,
        _expected: &Self::ExecutionId,
        output: &Self::Output,
    ) -> Result<(), Self::Error> {
        self.validate_output(request, output)
    }
    fn allocate_execution_id(&mut self) -> Result<Self::ExecutionId, Self::Error>;
    fn runtime_error(&self, fault: RuntimeFault) -> Self::Error;
    /// Attach each subscriber's context while preserving the physical failure.
    fn execution_error(&self, request: &Self::Request, error: &Self::Error) -> Self::Error;
    fn terminal(
        &mut self,
        request: &Self::Request,
        event: TerminalEvent<Self::Output, Self::Error>,
    ) -> Self::TerminalResult;
}

/// One owned host-side output, tagged with the original logical request ID.
pub struct BackendResult<A: Adapter> {
    pub request_id: A::RequestId,
    pub output: Result<A::Output, A::Error>,
}

/// Cooperative backend execution. No concrete tensor or device type is exposed.
///
/// `dispatch` and `poll` must return promptly. Dispatch Err guarantees no device
/// work was launched; once work is launched it must return a Lease, even if that
/// work will fail. Poll Ready guarantees physical completion, with one tagged
/// result for every input. Pending keeps all device/input/output/workspace pins
/// in the Lease. A Ready output must own its data without borrowing the Lease.
/// Additional resources cover batch workspace/staging not counted per request.
/// Once device work starts, provider code must contain panic/unwind and preserve
/// pins/context in a Lease until completion. An unwinding launch that loses its
/// Lease is a provider contract violation, not a recoverable scheduler error.
pub trait Backend<A: Adapter>: 'static {
    /// Own/pin its context; borrowed external owners cannot survive quarantine.
    type Lease: 'static;

    fn additional_resources(&self, requests: &[Arc<A::Request>]) -> Resources;
    fn dispatch(
        &mut self,
        execution_id: &A::ExecutionId,
        requests: &[Arc<A::Request>],
    ) -> Result<Self::Lease, A::Error>;
    fn poll(&mut self, lease: &mut Self::Lease) -> Poll<Vec<BackendResult<A>>>;
}

/// One monotonic clock domain, including the embedding request deadlines.
pub trait Clock {
    type Tick: Copy + Debug + Ord;
    fn now(&self) -> Self::Tick;
    fn elapsed(&self, earlier: Self::Tick, later: Self::Tick) -> Duration;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    type Tick = Instant;

    fn now(&self) -> Instant {
        Instant::now()
    }

    fn elapsed(&self, earlier: Instant, later: Instant) -> Duration {
        later.saturating_duration_since(earlier)
    }
}
