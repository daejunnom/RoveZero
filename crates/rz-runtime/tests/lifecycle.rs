//! Independent CPU fixtures exercise the scheduler's public lifecycle boundary.
//! All identifiers, errors and request/output types below are test-local; they
//! are not alternate definitions of TASK-I01's shared contract.

use rz_runtime::{
    Adapter, Backend, BackendResult, Clock, CompletionReceiver, DrainState, Limits,
    ObservationKind, Resources, RuntimeFault, Scheduler, TerminalEvent,
};
use rz_telemetry::FinishKind;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::task::Poll;
use std::thread;
use std::time::Duration;

#[derive(Clone, Default)]
struct ManualClock(Arc<AtomicU64>);

impl ManualClock {
    fn set(&self, tick: u64) {
        self.0.store(tick, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    type Tick = u64;

    fn now(&self) -> Self::Tick {
        self.0.load(Ordering::SeqCst)
    }

    fn elapsed(&self, earlier: u64, later: u64) -> Duration {
        Duration::from_millis(later.saturating_sub(earlier))
    }
}

#[derive(Default)]
struct Drops {
    inputs: AtomicUsize,
    leases: AtomicUsize,
    providers: AtomicUsize,
}

#[derive(Debug)]
struct Request {
    id: u64,
    root: u64,
    key: u8,
    deadline: u64,
    resources: Resources,
    canceled: Arc<AtomicBool>,
    drops: Arc<Drops>,
}

impl Drop for Request {
    fn drop(&mut self) {
        self.drops.inputs.fetch_add(1, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for Drops {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drops").finish_non_exhaustive()
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Output {
    request_id: u64,
    value: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Failure {
    Runtime(RuntimeFault),
    Admission,
    InvalidOutput,
    Backend(&'static str),
    ExecutionId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Error {
    request_id: Option<u64>,
    failure: Failure,
}

impl Error {
    fn backend(code: &'static str) -> Self {
        Self {
            request_id: None,
            failure: Failure::Backend(code),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Outcome {
    Success(Output),
    Canceled,
    Expired,
    Stale,
    Failure(Error),
}

#[derive(Debug, Eq, PartialEq)]
struct Reply {
    request_id: u64,
    outcome: Outcome,
}

enum ValidationHook {
    AdvanceClock(u64),
    ChangeRoot(u64),
    Cancel(Arc<AtomicBool>),
    AssertLeaseAlive,
}

struct AdapterControl {
    clock: ManualClock,
    root: AtomicU64,
    validation: Mutex<VecDeque<ValidationHook>>,
    admission_tick: AtomicU64,
    admission_ids: Mutex<Vec<u64>>,
    execution_tick: AtomicU64,
    fail_execution_id: AtomicBool,
    panic_binding: AtomicBool,
    terminal_ids: Mutex<Vec<u64>>,
    drops: Arc<Drops>,
}

struct FixtureAdapter {
    control: Arc<AdapterControl>,
    seen: HashSet<u64>,
    next_execution: u64,
}

impl Adapter for FixtureAdapter {
    type RequestId = u64;
    type ExecutionId = u64;
    type Tick = u64;
    type Request = Request;
    type BatchKey = u8;
    type Output = Output;
    type Error = Error;
    type TerminalResult = Reply;

    fn request_id(&self, request: &Request) -> u64 {
        request.id
    }

    fn deadline(&self, request: &Request) -> u64 {
        request.deadline
    }

    fn batch_key(&self, request: &Request) -> u8 {
        request.key
    }

    fn resources(&self, request: &Request) -> Resources {
        request.resources
    }

    fn validate_admission(&mut self, request: &Request) -> Result<(), Error> {
        self.control.admission_ids.lock().unwrap().push(request.id);
        let tick = self.control.admission_tick.swap(0, Ordering::SeqCst);
        if tick != 0 {
            self.control.clock.set(tick);
        }
        if request.id == 0 || !self.seen.insert(request.id) {
            return Err(Error {
                request_id: Some(request.id),
                failure: Failure::Admission,
            });
        }
        Ok(())
    }

    fn is_current(&self, request: &Request) -> bool {
        self.control.root.load(Ordering::SeqCst) == request.root
    }

    fn is_canceled(&self, request: &Request) -> bool {
        request.canceled.load(Ordering::SeqCst)
    }

    fn validate_output(&mut self, request: &Request, output: &Output) -> Result<(), Error> {
        if let Some(hook) = self.control.validation.lock().unwrap().pop_front() {
            match hook {
                ValidationHook::AdvanceClock(tick) => self.control.clock.set(tick),
                ValidationHook::ChangeRoot(root) => {
                    self.control.root.store(root, Ordering::SeqCst);
                }
                ValidationHook::Cancel(token) => token.store(true, Ordering::SeqCst),
                ValidationHook::AssertLeaseAlive => {
                    assert_eq!(self.control.drops.leases.load(Ordering::SeqCst), 0);
                    assert_eq!(self.control.drops.inputs.load(Ordering::SeqCst), 0);
                }
            }
        }
        if output.request_id != request.id || output.value != request.id as i64 * 10 {
            return Err(Error {
                request_id: Some(request.id),
                failure: Failure::InvalidOutput,
            });
        }
        Ok(())
    }

    fn allocate_execution_id(&mut self) -> Result<u64, Error> {
        let tick = self.control.execution_tick.swap(0, Ordering::SeqCst);
        if tick != 0 {
            self.control.clock.set(tick);
        }
        if self.control.fail_execution_id.swap(false, Ordering::SeqCst) {
            return Err(Error {
                request_id: None,
                failure: Failure::ExecutionId,
            });
        }
        self.next_execution += 1;
        Ok(self.next_execution)
    }

    fn bind_execution(&mut self, _request: &Request, _id: &u64) {
        assert!(
            !self.control.panic_binding.swap(false, Ordering::SeqCst),
            "fixture binding failure"
        );
    }

    fn runtime_error(&self, fault: RuntimeFault) -> Error {
        Error {
            request_id: None,
            failure: Failure::Runtime(fault),
        }
    }

    fn execution_error(&self, request: &Request, error: &Error) -> Error {
        Error {
            request_id: Some(request.id),
            failure: error.failure.clone(),
        }
    }

    fn terminal(&mut self, request: &Request, event: TerminalEvent<Output, Error>) -> Reply {
        self.control.terminal_ids.lock().unwrap().push(request.id);
        Reply {
            request_id: request.id,
            outcome: match event {
                TerminalEvent::Success(output) => Outcome::Success(output),
                TerminalEvent::Canceled => Outcome::Canceled,
                TerminalEvent::Expired => Outcome::Expired,
                TerminalEvent::Stale => Outcome::Stale,
                TerminalEvent::Failure(error) => Outcome::Failure(error),
            },
        }
    }
}

enum Results {
    InputOrder,
    Explicit(Vec<(u64, Result<i64, Error>)>),
}

struct Plan {
    ready: Arc<AtomicBool>,
    results: Results,
}

impl Plan {
    fn pending() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
            results: Results::InputOrder,
        }
    }
}

#[derive(Default)]
struct BackendControl {
    batches: Mutex<Vec<Vec<u64>>>,
    execution_ids: Mutex<Vec<u64>>,
    plans: Mutex<VecDeque<Plan>>,
    dispatch_error: Mutex<Option<Error>>,
}

struct ControlledBackend {
    control: Arc<BackendControl>,
    overhead: Resources,
    drops: Arc<Drops>,
}

struct OpaqueLease {
    inputs: Vec<Arc<Request>>,
    plan: Plan,
    drops: Arc<Drops>,
}

impl Drop for OpaqueLease {
    fn drop(&mut self) {
        self.drops.leases.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for ControlledBackend {
    fn drop(&mut self) {
        self.drops.providers.fetch_add(1, Ordering::SeqCst);
    }
}

impl Backend<FixtureAdapter> for ControlledBackend {
    type Lease = OpaqueLease;

    fn additional_resources(&self, _requests: &[Arc<Request>]) -> Resources {
        self.overhead
    }

    fn dispatch(
        &mut self,
        execution_id: &u64,
        requests: &[Arc<Request>],
    ) -> Result<OpaqueLease, Error> {
        if let Some(error) = self.control.dispatch_error.lock().unwrap().take() {
            return Err(error);
        }
        self.control
            .batches
            .lock()
            .unwrap()
            .push(requests.iter().map(|request| request.id).collect());
        self.control
            .execution_ids
            .lock()
            .unwrap()
            .push(*execution_id);
        let plan = self
            .control
            .plans
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Plan {
                ready: Arc::new(AtomicBool::new(true)),
                results: Results::InputOrder,
            });
        Ok(OpaqueLease {
            inputs: requests.to_vec(),
            plan,
            drops: Arc::clone(&self.drops),
        })
    }

    fn poll(&mut self, lease: &mut OpaqueLease) -> Poll<Vec<BackendResult<FixtureAdapter>>> {
        if !lease.plan.ready.load(Ordering::SeqCst) {
            return Poll::Pending;
        }
        let results = std::mem::replace(&mut lease.plan.results, Results::InputOrder);
        let outputs = match results {
            Results::InputOrder => lease
                .inputs
                .iter()
                .map(|request| BackendResult {
                    request_id: request.id,
                    output: Ok(Output {
                        request_id: request.id,
                        value: request.id as i64 * 10,
                    }),
                })
                .collect(),
            Results::Explicit(results) => results
                .into_iter()
                .map(|(id, value)| BackendResult {
                    request_id: id,
                    output: value.map(|value| Output {
                        request_id: id,
                        value,
                    }),
                })
                .collect(),
        };
        Poll::Ready(outputs)
    }
}

type Runtime = Scheduler<FixtureAdapter, ControlledBackend, ManualClock>;

struct Fixture {
    runtime: Runtime,
    adapter: Arc<AdapterControl>,
    backend: Arc<BackendControl>,
    clock: ManualClock,
    drops: Arc<Drops>,
}

fn bytes(host_bytes: u64, device_bytes: u64, pinned_bytes: u64) -> Resources {
    Resources {
        host_bytes,
        device_bytes,
        pinned_bytes,
    }
}

fn limits() -> Limits {
    Limits {
        max_requests: 8,
        max_batch_items: 1,
        max_executions: 1,
        max_batch_wait: Duration::ZERO,
        max_queue_age: Duration::from_millis(100),
        deadline_reserve: Duration::from_millis(2),
        memory: bytes(1_000, 1_000, 1_000),
    }
}

impl Fixture {
    fn new(limits: Limits, overhead: Resources) -> Self {
        Self::with_capacity(limits, overhead, 16)
    }

    fn with_capacity(limits: Limits, overhead: Resources, capacity: usize) -> Self {
        let clock = ManualClock::default();
        let drops = Arc::new(Drops::default());
        let adapter = Arc::new(AdapterControl {
            clock: clock.clone(),
            root: AtomicU64::new(1),
            validation: Mutex::new(VecDeque::new()),
            admission_tick: AtomicU64::new(0),
            admission_ids: Mutex::new(Vec::new()),
            execution_tick: AtomicU64::new(0),
            fail_execution_id: AtomicBool::new(false),
            panic_binding: AtomicBool::new(false),
            terminal_ids: Mutex::new(Vec::new()),
            drops: Arc::clone(&drops),
        });
        let backend = Arc::new(BackendControl::default());
        let runtime = Scheduler::new(
            FixtureAdapter {
                control: Arc::clone(&adapter),
                seen: HashSet::new(),
                next_execution: 0,
            },
            ControlledBackend {
                control: Arc::clone(&backend),
                overhead,
                drops: Arc::clone(&drops),
            },
            clock.clone(),
            limits,
            capacity,
        )
        .unwrap();
        Self {
            runtime,
            adapter,
            backend,
            clock,
            drops,
        }
    }

    fn request(&self, id: u64) -> Request {
        Request {
            id,
            root: 1,
            key: 1,
            deadline: 1_000,
            resources: bytes(1, 2, 3),
            canceled: Arc::new(AtomicBool::new(false)),
            drops: Arc::clone(&self.drops),
        }
    }

    fn pending(&self) -> Arc<AtomicBool> {
        let plan = Plan::pending();
        let ready = Arc::clone(&plan.ready);
        self.backend.plans.lock().unwrap().push_back(plan);
        ready
    }

    fn results(&self, results: Vec<(u64, Result<i64, Error>)>) {
        self.backend.plans.lock().unwrap().push_back(Plan {
            ready: Arc::new(AtomicBool::new(true)),
            results: Results::Explicit(results),
        });
    }

    fn batches(&self) -> Vec<Vec<u64>> {
        self.backend.batches.lock().unwrap().clone()
    }
}

fn receive(receiver: &CompletionReceiver<Reply>, id: u64, outcome: Outcome) {
    assert_eq!(
        receiver.try_recv().unwrap(),
        Reply {
            request_id: id,
            outcome,
        }
    );
    assert_eq!(receiver.try_recv(), Err(mpsc::TryRecvError::Disconnected));
}

fn runtime_failure(fault: RuntimeFault) -> Outcome {
    Outcome::Failure(Error {
        request_id: None,
        failure: Failure::Runtime(fault),
    })
}

fn success(id: u64) -> Outcome {
    Outcome::Success(Output {
        request_id: id,
        value: id as i64 * 10,
    })
}

#[test]
fn admission_returns_rejected_ownership_without_terminal_or_reservation() {
    let mut cap = limits();
    cap.max_requests = 1;
    let mut fixture = Fixture::new(cap, Resources::default());
    let first = fixture.runtime.submit(fixture.request(1)).unwrap();
    let duplicate = fixture.runtime.submit(fixture.request(1)).err().unwrap();
    assert_eq!(
        duplicate.error.failure,
        Failure::Runtime(RuntimeFault::DuplicateRequest)
    );
    assert_eq!(duplicate.request.id, 1);
    drop(duplicate);
    let full = fixture.runtime.submit(fixture.request(2)).err().unwrap();
    assert_eq!(
        full.error.failure,
        Failure::Runtime(RuntimeFault::QueueFull)
    );
    drop(full);
    assert_eq!(fixture.runtime.state().reserved, bytes(1, 2, 3));
    assert!(fixture.adapter.terminal_ids.lock().unwrap().is_empty());
    assert!(fixture.runtime.cancel(&1));
    receive(&first, 1, Outcome::Canceled);
    assert!(!fixture.runtime.cancel(&1));
    let reused = fixture.runtime.submit(fixture.request(1)).err().unwrap();
    assert_eq!(reused.error.failure, Failure::Admission);
    drop(reused);
    let refused = fixture.runtime.submit(fixture.request(2)).err().unwrap();
    assert_eq!(refused.error.failure, Failure::Admission);
    drop(refused);
    assert_eq!(
        *fixture.adapter.admission_ids.lock().unwrap(),
        vec![1, 1, 2, 1, 2],
        "capacity and duplicate refusals must each validate exactly once"
    );
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(*fixture.adapter.terminal_ids.lock().unwrap(), vec![1]);
    assert_eq!(fixture.runtime.metrics().rejected, 4);
}

#[test]
fn each_memory_budget_and_checked_addition_are_admission_boundaries() {
    for requested in [bytes(11, 0, 0), bytes(0, 11, 0), bytes(0, 0, 11)] {
        let mut cap = limits();
        cap.memory = bytes(10, 10, 10);
        let mut fixture = Fixture::new(cap, Resources::default());
        let mut request = fixture.request(1);
        request.resources = requested;
        let rejected = fixture.runtime.submit(request).err().unwrap();
        assert_eq!(
            rejected.error.failure,
            Failure::Runtime(RuntimeFault::ResourceLimit)
        );
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert!(fixture.adapter.terminal_ids.lock().unwrap().is_empty());
        drop(rejected);
    }
    for enormous in [
        bytes(u64::MAX, 0, 0),
        bytes(0, u64::MAX, 0),
        bytes(0, 0, u64::MAX),
    ] {
        let mut cap = limits();
        cap.memory = bytes(u64::MAX, u64::MAX, u64::MAX);
        let mut fixture = Fixture::new(cap, Resources::default());
        let mut request = fixture.request(1);
        request.resources = enormous;
        let receiver = fixture.runtime.submit(request).unwrap();
        let mut request = fixture.request(2);
        request.resources = bytes(1, 1, 1);
        let rejected = fixture.runtime.submit(request).err().unwrap();
        assert_eq!(
            rejected.error.failure,
            Failure::Runtime(RuntimeFault::ResourceOverflow)
        );
        assert_eq!(fixture.runtime.state().reserved, enormous);
        drop(rejected);
        fixture.runtime.cancel(&1);
        receive(&receiver, 1, Outcome::Canceled);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
    }
}

#[test]
fn admission_rechecks_time_and_request_generation_after_validation() {
    for invalid in [
        RuntimeFault::Expired,
        RuntimeFault::Canceled,
        RuntimeFault::Stale,
    ] {
        let mut fixture = Fixture::new(limits(), Resources::default());
        let mut request = fixture.request(1);
        match invalid {
            RuntimeFault::Expired => {
                request.deadline = 4;
                fixture.adapter.admission_tick.store(4, Ordering::SeqCst);
            }
            RuntimeFault::Canceled => request.canceled.store(true, Ordering::SeqCst),
            RuntimeFault::Stale => fixture.adapter.root.store(2, Ordering::SeqCst),
            _ => unreachable!(),
        }
        let rejected = fixture.runtime.submit(request).err().unwrap();
        assert_eq!(rejected.error.failure, Failure::Runtime(invalid));
        assert_eq!(fixture.runtime.state().logical_requests, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert!(fixture.adapter.terminal_ids.lock().unwrap().is_empty());
        drop(rejected);
    }
}

#[test]
fn fifo_batches_stop_at_incompatible_head_and_wait_for_size_or_age() {
    let mut cap = limits();
    cap.max_batch_items = 3;
    cap.max_batch_wait = Duration::from_millis(10);
    let mut fixture = Fixture::new(cap, Resources::default());
    let first = fixture.runtime.submit(fixture.request(1)).unwrap();
    let second = fixture.runtime.submit(fixture.request(2)).unwrap();
    fixture.runtime.pump();
    assert!(fixture.batches().is_empty());
    fixture.clock.set(9);
    fixture.runtime.pump();
    assert!(fixture.batches().is_empty());
    let mut incompatible = fixture.request(3);
    incompatible.key = 2;
    let third = fixture.runtime.submit(incompatible).unwrap();
    let fourth = fixture.runtime.submit(fixture.request(4)).unwrap();
    fixture.runtime.pump();
    assert_eq!(fixture.batches(), vec![vec![1, 2]]);
    assert_eq!(fixture.runtime.state().queued, 2);
    fixture.runtime.pump();
    receive(&first, 1, success(1));
    receive(&second, 2, success(2));
    assert_eq!(fixture.batches(), vec![vec![1, 2], vec![3]]);
    fixture.runtime.pump();
    receive(&third, 3, success(3));
    assert_eq!(fixture.batches(), vec![vec![1, 2], vec![3]]);
    fixture.clock.set(19);
    fixture.runtime.pump();
    assert_eq!(fixture.batches(), vec![vec![1, 2], vec![3], vec![4]]);
    fixture.runtime.pump();
    receive(&fourth, 4, success(4));
    assert_eq!(
        *fixture.backend.execution_ids.lock().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
}

#[test]
fn full_batch_dispatches_without_wait_and_near_deadline_flushes_small_batch() {
    let mut cap = limits();
    cap.max_batch_items = 2;
    cap.max_batch_wait = Duration::from_millis(50);
    let mut fixture = Fixture::new(cap, Resources::default());
    let first = fixture.runtime.submit(fixture.request(1)).unwrap();
    let second = fixture.runtime.submit(fixture.request(2)).unwrap();
    fixture.runtime.pump();
    assert_eq!(fixture.batches(), vec![vec![1, 2]]);
    fixture.runtime.pump();
    receive(&first, 1, success(1));
    receive(&second, 2, success(2));
    let mut request = fixture.request(3);
    request.deadline = 10;
    let near = fixture.runtime.submit(request).unwrap();
    fixture.clock.set(7);
    fixture.runtime.pump();
    assert_eq!(fixture.batches().len(), 1);
    fixture.clock.set(8);
    fixture.runtime.pump();
    assert_eq!(fixture.batches(), vec![vec![1, 2], vec![3]]);
    fixture.clock.set(9);
    fixture.runtime.pump();
    receive(&near, 3, success(3));
}

#[test]
fn queued_cancel_releases_input_once_and_queue_age_is_explicit_failure() {
    let mut cap = limits();
    cap.max_batch_items = 4;
    cap.max_batch_wait = Duration::from_millis(100);
    let mut fixture = Fixture::new(cap, Resources::default());
    let canceled = fixture.runtime.submit(fixture.request(1)).unwrap();
    let old = fixture.runtime.submit(fixture.request(2)).unwrap();
    assert!(fixture.runtime.cancel(&1));
    assert!(!fixture.runtime.cancel(&1));
    receive(&canceled, 1, Outcome::Canceled);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.runtime.state().reserved, bytes(1, 2, 3));
    fixture.clock.set(100);
    fixture.runtime.pump();
    receive(&old, 2, runtime_failure(RuntimeFault::QueueAgeExceeded));
    assert!(fixture.batches().is_empty());
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 2);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(*fixture.adapter.terminal_ids.lock().unwrap(), vec![1, 2]);
}

#[test]
fn running_cancel_preserves_lease_inputs_provider_and_overhead_until_physical_ready() {
    let mut fixture = Fixture::new(limits(), bytes(4, 5, 6));
    let ready = fixture.pending();
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.state().reserved, bytes(5, 7, 9));
    assert!(fixture.runtime.cancel(&1));
    assert!(!fixture.runtime.cancel(&1));
    receive(&receiver, 1, Outcome::Canceled);
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.state().logical_requests, 0);
    assert_eq!(fixture.runtime.state().executions, 1);
    assert_eq!(fixture.runtime.state().reserved, bytes(5, 7, 9));
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.drops.providers.load(Ordering::SeqCst), 0);
    ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(fixture.runtime.state().peak_reserved, bytes(5, 7, 9));
    assert_eq!(fixture.runtime.state().peak_reserved_requests, 1);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
    assert_eq!(*fixture.adapter.terminal_ids.lock().unwrap(), vec![1]);
    assert_eq!(fixture.runtime.metrics().canceled, 1);
    assert_eq!(fixture.runtime.metrics().completed, 0);
    assert_eq!(fixture.runtime.metrics().physical_completed, 1);
    let drops = Arc::clone(&fixture.drops);
    drop(fixture);
    assert_eq!(drops.providers.load(Ordering::SeqCst), 1);
}

#[test]
fn canceled_running_reservations_still_block_new_admission() {
    let mut cap = limits();
    cap.max_requests = 2;
    cap.memory = bytes(5, 7, 9);
    let mut fixture = Fixture::new(cap, bytes(4, 5, 6));
    let ready = fixture.pending();
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.cancel(&1);
    receive(&receiver, 1, Outcome::Canceled);
    let rejected = fixture.runtime.submit(fixture.request(2)).err().unwrap();
    assert_eq!(
        rejected.error.failure,
        Failure::Runtime(RuntimeFault::ResourceLimit)
    );
    drop(rejected);
    ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    let third = fixture.runtime.submit(fixture.request(3)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.pump();
    receive(&third, 3, success(3));
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
}

#[test]
fn workspace_limit_or_overflow_fails_batch_before_any_dispatch() {
    for (memory, overhead, expected) in [
        (bytes(3, 3, 3), bytes(3, 0, 0), RuntimeFault::ResourceLimit),
        (
            bytes(u64::MAX, u64::MAX, u64::MAX),
            bytes(u64::MAX, 0, 0),
            RuntimeFault::ResourceOverflow,
        ),
    ] {
        let mut cap = limits();
        cap.memory = memory;
        let mut fixture = Fixture::new(cap, overhead);
        let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
        fixture.runtime.pump();
        receive(&receiver, 1, runtime_failure(expected));
        assert!(fixture.batches().is_empty());
        assert_eq!(fixture.runtime.state().executions, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
        assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn allocation_and_dispatch_failures_preserve_subscriber_context_and_release_reservations() {
    for execution_id_fails in [true, false] {
        let mut cap = limits();
        cap.max_batch_items = 2;
        let mut fixture = Fixture::new(cap, bytes(4, 5, 6));
        if execution_id_fails {
            fixture
                .adapter
                .fail_execution_id
                .store(true, Ordering::SeqCst);
        } else {
            *fixture.backend.dispatch_error.lock().unwrap() = Some(Error::backend("launch"));
        }
        let first = fixture.runtime.submit(fixture.request(1)).unwrap();
        let second = fixture.runtime.submit(fixture.request(2)).unwrap();
        fixture.runtime.pump();
        for (id, receiver) in [(1, first), (2, second)] {
            receive(
                &receiver,
                id,
                Outcome::Failure(Error {
                    request_id: Some(id),
                    failure: if execution_id_fails {
                        Failure::ExecutionId
                    } else {
                        Failure::Backend("launch")
                    },
                }),
            );
        }
        assert!(fixture.batches().is_empty());
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 2);
        assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn execution_id_allocation_crossing_deadline_aborts_launch_and_releases_workspace() {
    let mut fixture = Fixture::new(limits(), bytes(4, 5, 6));
    let mut request = fixture.request(1);
    request.deadline = 10;
    fixture.adapter.execution_tick.store(10, Ordering::SeqCst);
    let receiver = fixture.runtime.submit(request).unwrap();
    fixture.runtime.pump();
    assert!(fixture.batches().is_empty());
    assert_eq!(fixture.clock.now(), 10);
    assert_eq!(fixture.runtime.state().executions, 0);
    assert_eq!(fixture.runtime.state().logical_requests, 0);
    assert_eq!(fixture.runtime.state().queued, 0);
    // The aborted launch's workspace is already released; the expired reply
    // alone retains the per-request delivery reservation until it is consumed.
    assert_eq!(fixture.runtime.state().reserved, bytes(1, 2, 3));
    assert_eq!(fixture.runtime.state().reserved_requests, 1);
    receive(&receiver, 1, Outcome::Expired);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert_eq!(fixture.runtime.metrics().expired, 1);
    assert_eq!(fixture.runtime.metrics().dispatches, 0);
    assert_eq!(fixture.runtime.metrics().physical_completed, 0);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
}

#[test]
fn execution_id_allocation_crossing_queue_age_aborts_launch_at_the_configured_limit() {
    let mut fixture = Fixture::new(limits(), bytes(4, 5, 6));
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.clock.set(99);
    fixture.adapter.execution_tick.store(100, Ordering::SeqCst);
    fixture.runtime.pump();
    assert!(fixture.batches().is_empty());
    assert_eq!(fixture.clock.now(), 100);
    assert_eq!(fixture.runtime.state().executions, 0);
    assert_eq!(fixture.runtime.state().logical_requests, 0);
    assert_eq!(fixture.runtime.state().queued, 0);
    assert_eq!(fixture.runtime.state().reserved, bytes(1, 2, 3));
    receive(
        &receiver,
        1,
        runtime_failure(RuntimeFault::QueueAgeExceeded),
    );
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert_eq!(fixture.runtime.metrics().failed, 1);
    assert_eq!(fixture.runtime.metrics().expired, 0);
    assert_eq!(fixture.runtime.metrics().dispatches, 0);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
}

#[test]
fn unrepresentable_telemetry_capacity_is_a_typed_constructor_failure() {
    let fixture = Fixture::new(limits(), Resources::default());
    let result = Scheduler::new(
        FixtureAdapter {
            control: Arc::clone(&fixture.adapter),
            seen: HashSet::new(),
            next_execution: 0,
        },
        ControlledBackend {
            control: Arc::clone(&fixture.backend),
            overhead: Resources::default(),
            drops: Arc::clone(&fixture.drops),
        },
        fixture.clock.clone(),
        limits(),
        usize::MAX,
    );
    assert!(matches!(result, Err(RuntimeFault::ResourceOverflow)));
    assert_eq!(fixture.drops.providers.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
}

#[test]
fn malformed_completion_ids_fail_the_whole_batch_before_sibling_success() {
    for outputs in [
        vec![(1, Ok(10)), (1, Ok(10))],
        vec![(1, Ok(10))],
        vec![(1, Ok(10)), (99, Ok(990))],
        vec![(1, Ok(10)), (2, Ok(20)), (99, Ok(990))],
    ] {
        let mut cap = limits();
        cap.max_batch_items = 2;
        let mut fixture = Fixture::new(cap, bytes(4, 5, 6));
        fixture.results(outputs);
        let first = fixture.runtime.submit(fixture.request(1)).unwrap();
        let second = fixture.runtime.submit(fixture.request(2)).unwrap();
        fixture.runtime.pump();
        fixture.runtime.pump();
        receive(&first, 1, runtime_failure(RuntimeFault::InvalidCompletion));
        receive(&second, 2, runtime_failure(RuntimeFault::InvalidCompletion));
        assert_eq!(fixture.runtime.metrics().completed, 0);
        assert_eq!(fixture.runtime.metrics().failed, 2);
        assert_eq!(fixture.runtime.metrics().physical_completed, 1);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 2);
        assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn reordered_results_route_to_original_request_and_partial_failure_is_preserved() {
    let mut cap = limits();
    cap.max_batch_items = 3;
    let mut fixture = Fixture::new(cap, Resources::default());
    fixture.results(vec![
        (3, Ok(30)),
        (1, Ok(10)),
        (2, Err(Error::backend("tensor"))),
    ]);
    let first = fixture.runtime.submit(fixture.request(1)).unwrap();
    let second = fixture.runtime.submit(fixture.request(2)).unwrap();
    let third = fixture.runtime.submit(fixture.request(3)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.pump();
    receive(&first, 1, success(1));
    receive(
        &second,
        2,
        Outcome::Failure(Error {
            request_id: Some(2),
            failure: Failure::Backend("tensor"),
        }),
    );
    receive(&third, 3, success(3));
    assert_eq!(*fixture.adapter.terminal_ids.lock().unwrap(), vec![3, 1, 2]);
    assert_eq!(fixture.runtime.metrics().completed, 2);
    assert_eq!(fixture.runtime.metrics().failed, 1);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
}

#[test]
fn output_validation_rechecks_deadline_generation_and_cancel_before_terminal_success() {
    for changed in [Outcome::Expired, Outcome::Stale, Outcome::Canceled] {
        let mut fixture = Fixture::new(limits(), Resources::default());
        let mut request = fixture.request(1);
        request.deadline = 10;
        let cancel_token = Arc::clone(&request.canceled);
        let hook = match changed {
            Outcome::Expired => ValidationHook::AdvanceClock(10),
            Outcome::Stale => ValidationHook::ChangeRoot(2),
            Outcome::Canceled => ValidationHook::Cancel(cancel_token),
            _ => unreachable!(),
        };
        fixture.adapter.validation.lock().unwrap().push_back(hook);
        let receiver = fixture.runtime.submit(request).unwrap();
        fixture.runtime.pump();
        fixture.clock.set(9);
        fixture.runtime.pump();
        receive(&receiver, 1, changed);
        assert_eq!(fixture.runtime.metrics().completed, 0);
        assert_eq!(fixture.runtime.metrics().physical_completed, 1);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
    }
}

#[test]
fn validation_failure_stays_failure_and_lease_is_alive_through_validation() {
    let mut fixture = Fixture::new(limits(), Resources::default());
    fixture.results(vec![(1, Ok(-500))]);
    fixture
        .adapter
        .validation
        .lock()
        .unwrap()
        .push_back(ValidationHook::AssertLeaseAlive);
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.pump();
    receive(
        &receiver,
        1,
        Outcome::Failure(Error {
            request_id: Some(1),
            failure: Failure::InvalidOutput,
        }),
    );
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.runtime.metrics().completed, 0);
}

#[test]
fn exact_deadline_expiry_prevents_dispatch_or_late_completion() {
    for already_running in [false, true] {
        let mut fixture = Fixture::new(limits(), Resources::default());
        let mut request = fixture.request(1);
        request.deadline = 10;
        let receiver = fixture.runtime.submit(request).unwrap();
        if already_running {
            fixture.runtime.pump();
        }
        fixture.clock.set(10);
        fixture.runtime.pump();
        receive(&receiver, 1, Outcome::Expired);
        assert_eq!(fixture.batches().len(), usize::from(already_running));
        assert_eq!(fixture.runtime.metrics().completed, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
    }
}

#[test]
fn shutdown_keeps_first_deadline_and_reclaims_after_timeout_when_physical_work_finishes() {
    let mut fixture = Fixture::new(limits(), bytes(4, 5, 6));
    let ready = fixture.pending();
    let running = fixture.runtime.submit(fixture.request(1)).unwrap();
    let queued = fixture.runtime.submit(fixture.request(2)).unwrap();
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.drain_state(), DrainState::Open);
    fixture.runtime.begin_shutdown(10);
    receive(&running, 1, Outcome::Canceled);
    receive(&queued, 2, Outcome::Canceled);
    assert_eq!(fixture.runtime.drain_state(), DrainState::Draining);
    assert_eq!(fixture.runtime.state().reserved, bytes(5, 7, 9));
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    fixture.runtime.begin_shutdown(100);
    fixture.clock.set(10);
    assert_eq!(
        fixture.runtime.drain_state(),
        DrainState::TimedOut {
            executions: 1,
            reserved: bytes(5, 7, 9),
        }
    );
    let rejected = fixture.runtime.submit(fixture.request(3)).err().unwrap();
    assert_eq!(
        rejected.error.failure,
        Failure::Runtime(RuntimeFault::Closed)
    );
    assert_eq!(
        *fixture.adapter.admission_ids.lock().unwrap(),
        vec![1, 2, 3]
    );
    drop(rejected);
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.state().executions, 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
    ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.drain_state(), DrainState::Drained);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(fixture.runtime.metrics().canceled, 2);
    assert_eq!(fixture.runtime.metrics().physical_completed, 1);
    assert_eq!(fixture.runtime.metrics().completed, 0);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 3);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
}

#[test]
fn physical_drain_can_finish_while_unread_terminal_delivery_still_reserves_budget() {
    for completed in [false, true] {
        let mut fixture = Fixture::new(limits(), Resources::default());
        let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
        if completed {
            fixture.runtime.pump();
            fixture.runtime.pump();
        }
        fixture.runtime.begin_shutdown(10);
        assert_eq!(fixture.runtime.drain_state(), DrainState::Drained);
        assert_eq!(fixture.runtime.state().logical_requests, 0);
        assert_eq!(fixture.runtime.state().executions, 0);
        assert_eq!(fixture.runtime.state().reserved_requests, 1);
        assert_eq!(fixture.runtime.state().reserved, bytes(1, 2, 3));
        // A terminal result already committed before shutdown is immutable.
        // Search must perform its own validity check before using that result.
        assert!(!fixture.runtime.cancel(&1));
        receive(
            &receiver,
            1,
            if completed {
                success(1)
            } else {
                Outcome::Canceled
            },
        );
        assert_eq!(fixture.runtime.drain_state(), DrainState::Drained);
        assert_eq!(fixture.runtime.state().reserved_requests, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        assert_eq!(*fixture.adapter.terminal_ids.lock().unwrap(), vec![1]);
    }
}

#[test]
fn idle_and_pending_pumps_do_not_displace_lifecycle_observations() {
    let mut fixture = Fixture::with_capacity(limits(), bytes(4, 5, 6), 16);
    for _ in 0..128 {
        fixture.runtime.pump();
    }
    let idle = fixture.runtime.take_observations();
    assert!(idle.events.is_empty());
    assert_eq!(idle.dropped, 0);

    let ready = fixture.pending();
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.runtime.pump();
    for _ in 0..128 {
        fixture.runtime.pump();
    }
    let dispatched = fixture.runtime.take_observations();
    assert_eq!(dispatched.dropped, 0);
    assert!(!dispatched.counter_overflow);
    assert_eq!(
        dispatched
            .events
            .iter()
            .filter(|event| { matches!(event.kind, ObservationKind::ReservationChanged) })
            .count(),
        2
    );
    assert!(dispatched
        .events
        .iter()
        .any(|event| { matches!(event.kind, ObservationKind::Admitted { request: 1 }) }));
    assert!(dispatched.events.iter().any(|event| {
        matches!(
            event.kind,
            ObservationKind::RequestDispatched {
                request: 1,
                execution: 1,
                ..
            }
        )
    }));

    assert!(fixture.runtime.cancel(&1));
    receive(&receiver, 1, Outcome::Canceled);
    for _ in 0..128 {
        fixture.runtime.pump();
    }
    let canceled = fixture.runtime.take_observations();
    assert_eq!(canceled.dropped, 0);
    assert_eq!(canceled.events.len(), 1);
    assert!(matches!(
        canceled.events[0].kind,
        ObservationKind::Finished {
            kind: FinishKind::Canceled,
            ..
        }
    ));
    assert_eq!(fixture.runtime.state().reserved, bytes(5, 7, 9));
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);

    ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    for _ in 0..128 {
        fixture.runtime.pump();
    }
    let completed = fixture.runtime.take_observations();
    assert_eq!(completed.dropped, 0);
    assert_eq!(completed.events.len(), 2);
    assert!(matches!(
        completed.events[0].kind,
        ObservationKind::PhysicalCompleted { execution: 1 }
    ));
    assert!(matches!(
        completed.events[1].kind,
        ObservationKind::ReservationChanged
    ));
    assert_eq!(completed.events[1].reserved, Resources::default());
    assert_eq!(completed.events[1].reserved_requests, 0);
    assert_eq!(fixture.runtime.metrics().canceled, 1);
    assert_eq!(fixture.runtime.metrics().physical_completed, 1);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
}

#[test]
fn receiver_release_is_observed_once_on_the_next_owner_pump() {
    let mut fixture = Fixture::with_capacity(limits(), Resources::default(), 16);
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.state().reserved_requests, 1);
    let finished = fixture.runtime.take_observations();
    assert_eq!(finished.dropped, 0);
    for _ in 0..128 {
        fixture.runtime.pump();
    }
    let unread = fixture.runtime.take_observations();
    assert!(unread.events.is_empty());
    assert_eq!(unread.dropped, 0);
    drop(receiver);
    fixture.clock.set(7);
    fixture.runtime.pump();
    for _ in 0..128 {
        fixture.runtime.pump();
    }
    let released = fixture.runtime.take_observations();
    assert_eq!(released.dropped, 0);
    assert_eq!(released.events.len(), 1);
    assert!(matches!(
        released.events[0].kind,
        ObservationKind::ReservationChanged
    ));
    assert_eq!(released.events[0].at, 7);
    assert_eq!(released.events[0].reserved, Resources::default());
    assert_eq!(released.events[0].reserved_requests, 0);
    assert_eq!(released.events[0].peak_reserved, bytes(1, 2, 3));
    assert_eq!(fixture.runtime.metrics().completed, 1);
    assert_eq!(fixture.runtime.metrics().physical_completed, 1);
}

#[test]
fn canceled_observation_precedes_physical_completion_without_output_validation() {
    let mut fixture = Fixture::with_capacity(limits(), bytes(4, 5, 6), 64);
    let ready = fixture.pending();
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.cancel(&1);
    ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    receive(&receiver, 1, Outcome::Canceled);
    let observed = fixture.runtime.take_observations();
    assert_eq!(observed.dropped, 0);
    assert!(!observed.counter_overflow);
    let logical_finished = observed
        .events
        .iter()
        .position(|event| {
            matches!(
                &event.kind,
                ObservationKind::Finished {
                    request: 1,
                    kind: FinishKind::Canceled,
                    delivered: true,
                }
            )
        })
        .unwrap();
    let physical_finished = observed
        .events
        .iter()
        .position(|event| {
            matches!(
                &event.kind,
                ObservationKind::PhysicalCompleted { execution: 1 }
            )
        })
        .unwrap();
    assert!(logical_finished < physical_finished);
    assert!(observed.events.iter().any(|event| {
        matches!(
            &event.kind,
            ObservationKind::RequestDispatched {
                request: 1,
                execution: 1,
                ..
            }
        )
    }));
    assert!(!observed.events.iter().any(|event| {
        matches!(
            &event.kind,
            ObservationKind::ValidationStarted { .. } | ObservationKind::ValidationFinished { .. }
        )
    }));
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
}

#[test]
fn zero_observation_capacity_preserves_outcome_and_reservation_lifecycle() {
    let mut fixture = Fixture::with_capacity(limits(), bytes(4, 5, 6), 0);
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.runtime.pump();
    fixture.runtime.pump();
    receive(&receiver, 1, success(1));
    let observed = fixture.runtime.take_observations();
    assert!(observed.events.is_empty());
    assert!(observed.dropped > 0);
    assert!(!observed.counter_overflow);
    assert_eq!(fixture.runtime.metrics().completed, 1);
    assert_eq!(fixture.runtime.metrics().physical_completed, 1);
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
}

#[test]
fn successful_validation_observation_does_not_imply_terminal_success() {
    for deadline_changes in [false, true] {
        let mut fixture = Fixture::with_capacity(limits(), Resources::default(), 64);
        let mut request = fixture.request(1);
        request.deadline = 10;
        fixture
            .adapter
            .validation
            .lock()
            .unwrap()
            .push_back(if deadline_changes {
                ValidationHook::AdvanceClock(10)
            } else {
                ValidationHook::ChangeRoot(2)
            });
        let receiver = fixture.runtime.submit(request).unwrap();
        fixture.runtime.pump();
        fixture.clock.set(9);
        fixture.runtime.pump();
        receive(
            &receiver,
            1,
            if deadline_changes {
                Outcome::Expired
            } else {
                Outcome::Stale
            },
        );
        let observed = fixture.runtime.take_observations();
        assert_eq!(observed.dropped, 0);
        let validated = observed
            .events
            .iter()
            .position(|event| {
                matches!(
                    &event.kind,
                    ObservationKind::ValidationFinished {
                        request: 1,
                        execution: 1,
                        valid: true,
                    }
                )
            })
            .unwrap();
        let terminal = observed
            .events
            .iter()
            .position(|event| {
                matches!(
                    &event.kind,
                    ObservationKind::Finished {
                        request: 1,
                        kind,
                        ..
                    } if *kind == if deadline_changes {
                        FinishKind::Expired
                    } else {
                        FinishKind::Stale
                    }
                )
            })
            .unwrap();
        assert!(validated < terminal);
        assert!(!observed.events.iter().any(|event| {
            matches!(
                &event.kind,
                ObservationKind::Finished {
                    kind: FinishKind::Completed,
                    ..
                }
            )
        }));
        assert_eq!(fixture.runtime.metrics().completed, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
    }
}

#[test]
fn post_launch_binding_panic_keeps_pending_execution_owned_for_shutdown_quarantine() {
    let mut fixture = Fixture::new(limits(), bytes(4, 5, 6));
    let _ready = fixture.pending();
    fixture.adapter.panic_binding.store(true, Ordering::SeqCst);
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    let pumped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fixture.runtime.pump();
    }));
    assert!(pumped.is_err());
    assert_eq!(fixture.runtime.state().executions, 1);
    assert_eq!(fixture.runtime.state().reserved, bytes(5, 7, 9));
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.drops.providers.load(Ordering::SeqCst), 0);
    let drops = Arc::clone(&fixture.drops);
    let adapter = Arc::clone(&fixture.adapter);
    drop(fixture);
    receive(&receiver, 1, Outcome::Canceled);
    assert_eq!(*adapter.terminal_ids.lock().unwrap(), vec![1]);
    assert_eq!(drops.inputs.load(Ordering::SeqCst), 0);
    assert_eq!(drops.leases.load(Ordering::SeqCst), 0);
    assert_eq!(drops.providers.load(Ordering::SeqCst), 0);
}

#[test]
fn dropping_receiver_does_not_block_scheduler_or_retain_completed_resources() {
    let mut fixture = Fixture::new(limits(), Resources::default());
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    drop(receiver);
    fixture.runtime.pump();
    fixture.runtime.pump();
    assert_eq!(fixture.runtime.metrics().receiver_dropped, 1);
    assert_eq!(fixture.runtime.metrics().completed, 1);
    assert_eq!(fixture.runtime.state().logical_requests, 0);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
}

#[test]
fn unread_completion_keeps_request_slot_and_output_bytes_bounded_until_receive_or_drop() {
    for by_slot in [false, true] {
        let mut cap = limits();
        cap.max_requests = if by_slot { 1 } else { 2 };
        cap.memory = bytes(1, 2, 3);
        let mut fixture = Fixture::new(cap, Resources::default());
        let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
        assert_eq!(receiver.try_recv(), Err(mpsc::TryRecvError::Empty));
        fixture.runtime.pump();
        fixture.runtime.pump();
        // The physical input/lease is gone, but an unread owned output still
        // belongs to the scheduler's delivery budget.
        assert_eq!(fixture.runtime.state().logical_requests, 0);
        assert_eq!(fixture.runtime.state().executions, 0);
        assert_eq!(fixture.runtime.state().reserved_requests, 1);
        assert_eq!(fixture.runtime.state().reserved, bytes(1, 2, 3));
        assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
        assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
        let rejected = fixture.runtime.submit(fixture.request(2)).err().unwrap();
        assert_eq!(
            rejected.error.failure,
            Failure::Runtime(if by_slot {
                RuntimeFault::QueueFull
            } else {
                RuntimeFault::ResourceLimit
            })
        );
        drop(rejected);
        if by_slot {
            assert_eq!(
                receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
                Reply {
                    request_id: 1,
                    outcome: success(1),
                }
            );
        } else {
            drop(receiver);
        }
        assert_eq!(fixture.runtime.state().reserved_requests, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
        let third = fixture.runtime.submit(fixture.request(3)).unwrap();
        fixture.runtime.pump();
        fixture.runtime.pump();
        assert_eq!(
            third.recv().unwrap(),
            Reply {
                request_id: 3,
                outcome: success(3),
            }
        );
        assert_eq!(fixture.runtime.state().reserved_requests, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
    }
}

#[test]
fn cancellation_keeps_a_request_slot_until_both_delivery_and_physical_owners_release_it() {
    for running in [false, true] {
        let mut cap = limits();
        cap.max_requests = 1;
        let mut fixture = Fixture::new(cap, Resources::default());
        let ready = fixture.pending();
        let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
        if running {
            fixture.runtime.pump();
        }
        fixture.runtime.cancel(&1);
        assert_eq!(fixture.runtime.state().logical_requests, 0);
        assert_eq!(fixture.runtime.state().reserved_requests, 1);
        let rejected = fixture.runtime.submit(fixture.request(2)).err().unwrap();
        assert_eq!(
            rejected.error.failure,
            Failure::Runtime(RuntimeFault::QueueFull)
        );
        drop(rejected);
        receive(&receiver, 1, Outcome::Canceled);
        if running {
            assert_eq!(fixture.runtime.state().reserved_requests, 1);
            assert_eq!(fixture.runtime.state().reserved, bytes(1, 2, 3));
            let duplicate = fixture.runtime.submit(fixture.request(1)).err().unwrap();
            assert_eq!(
                duplicate.error.failure,
                Failure::Runtime(RuntimeFault::DuplicateRequest)
            );
            drop(duplicate);
            ready.store(true, Ordering::SeqCst);
            fixture.runtime.pump();
        }
        assert_eq!(fixture.runtime.state().reserved_requests, 0);
        assert_eq!(fixture.runtime.state().reserved, Resources::default());
    }
}

#[test]
fn physical_concurrency_limit_keeps_fifo_tail_queued_until_one_execution_finishes() {
    let mut cap = limits();
    cap.max_executions = 2;
    let mut fixture = Fixture::new(cap, bytes(4, 5, 6));
    let first_ready = fixture.pending();
    let second_ready = fixture.pending();
    let first = fixture.runtime.submit(fixture.request(1)).unwrap();
    let second = fixture.runtime.submit(fixture.request(2)).unwrap();
    let third = fixture.runtime.submit(fixture.request(3)).unwrap();
    fixture.runtime.pump();
    assert_eq!(fixture.batches(), vec![vec![1], vec![2]]);
    assert_eq!(fixture.runtime.state().executions, 2);
    assert_eq!(fixture.runtime.state().queued, 1);
    assert_eq!(fixture.runtime.state().reserved, bytes(11, 16, 21));
    fixture.runtime.pump();
    assert_eq!(fixture.batches(), vec![vec![1], vec![2]]);
    first_ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    assert_eq!(fixture.batches(), vec![vec![1], vec![2], vec![3]]);
    assert_eq!(fixture.runtime.state().executions, 2);
    assert_eq!(fixture.runtime.state().queued, 0);
    receive(&first, 1, success(1));
    second_ready.store(true, Ordering::SeqCst);
    fixture.runtime.pump();
    receive(&second, 2, success(2));
    receive(&third, 3, success(3));
    assert_eq!(fixture.runtime.state().executions, 0);
    assert_eq!(fixture.runtime.state().reserved_requests, 0);
    assert_eq!(fixture.runtime.state().reserved, Resources::default());
    assert_eq!(fixture.runtime.state().peak_reserved, bytes(11, 16, 21));
    assert_eq!(fixture.runtime.state().peak_reserved_requests, 3);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 3);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 3);
}

#[test]
fn dropping_pending_scheduler_quarantines_provider_lease_and_input_instead_of_freeing_borrows() {
    let mut fixture = Fixture::new(limits(), Resources::default());
    let _ready = fixture.pending();
    let receiver = fixture.runtime.submit(fixture.request(1)).unwrap();
    fixture.runtime.pump();
    let drops = Arc::clone(&fixture.drops);
    drop(fixture);
    receive(&receiver, 1, Outcome::Canceled);
    assert_eq!(drops.inputs.load(Ordering::SeqCst), 0);
    assert_eq!(drops.leases.load(Ordering::SeqCst), 0);
    assert_eq!(drops.providers.load(Ordering::SeqCst), 0);
}

struct CpuBackend {
    release: Option<mpsc::Receiver<()>>,
    started: mpsc::SyncSender<()>,
    completed: mpsc::SyncSender<()>,
    drops: Arc<Drops>,
}

struct CpuLease {
    inputs: Vec<Arc<Request>>,
    outputs: mpsc::Receiver<Vec<BackendResult<FixtureAdapter>>>,
    drops: Arc<Drops>,
}

impl Drop for CpuBackend {
    fn drop(&mut self) {
        self.drops.providers.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for CpuLease {
    fn drop(&mut self) {
        self.drops.leases.fetch_add(1, Ordering::SeqCst);
    }
}

impl Backend<FixtureAdapter> for CpuBackend {
    type Lease = CpuLease;

    fn additional_resources(&self, _requests: &[Arc<Request>]) -> Resources {
        bytes(4, 5, 6)
    }

    fn dispatch(
        &mut self,
        _execution_id: &u64,
        requests: &[Arc<Request>],
    ) -> Result<CpuLease, Error> {
        let release = self.release.take().unwrap();
        let inputs = requests.to_vec();
        let worker_inputs = inputs.clone();
        let started = self.started.clone();
        let completed = self.completed.clone();
        let (output_tx, output_rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            started.send(()).unwrap();
            release.recv().unwrap();
            let mut results = Vec::new();
            for request in &worker_inputs {
                // CPU work produces owned host output while input ownership is
                // held by both the worker and the scheduler's opaque lease.
                let value = (0..10).map(|_| request.id as i64).sum();
                results.push(BackendResult {
                    request_id: request.id,
                    output: Ok(Output {
                        request_id: request.id,
                        value,
                    }),
                });
            }
            drop(worker_inputs);
            output_tx.send(results).unwrap();
            completed.send(()).unwrap();
        });
        Ok(CpuLease {
            inputs,
            outputs: output_rx,
            drops: Arc::clone(&self.drops),
        })
    }

    fn poll(&mut self, lease: &mut CpuLease) -> Poll<Vec<BackendResult<FixtureAdapter>>> {
        assert!(!lease.inputs.is_empty());
        match lease.outputs.try_recv() {
            Ok(outputs) => Poll::Ready(outputs),
            Err(mpsc::TryRecvError::Empty) => Poll::Pending,
            Err(mpsc::TryRecvError::Disconnected) => panic!("CPU worker lost its output"),
        }
    }
}

#[test]
fn shared_cancel_token_from_another_thread_terminates_logically_before_cpu_worker_finishes() {
    let fixture = Fixture::new(limits(), Resources::default());
    let request = fixture.request(1);
    let token = Arc::clone(&request.canceled);
    let adapter = FixtureAdapter {
        control: Arc::clone(&fixture.adapter),
        seen: HashSet::new(),
        next_execution: 0,
    };
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (completed_tx, completed_rx) = mpsc::sync_channel(1);
    let mut runtime = Scheduler::new(
        adapter,
        CpuBackend {
            release: Some(release_rx),
            started: started_tx,
            completed: completed_tx,
            drops: Arc::clone(&fixture.drops),
        },
        fixture.clock.clone(),
        limits(),
        8,
    )
    .unwrap();
    let receiver = runtime.submit(request).unwrap();
    runtime.pump();
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    thread::spawn(move || token.store(true, Ordering::SeqCst))
        .join()
        .unwrap();
    runtime.pump();
    receive(&receiver, 1, Outcome::Canceled);
    assert_eq!(runtime.state().logical_requests, 0);
    assert_eq!(runtime.state().executions, 1);
    assert_eq!(runtime.state().reserved, bytes(5, 7, 9));
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 0);
    release_tx.send(()).unwrap();
    completed_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    runtime.pump();
    assert_eq!(runtime.state().executions, 0);
    assert_eq!(runtime.state().reserved, Resources::default());
    assert_eq!(runtime.metrics().physical_completed, 1);
    assert_eq!(runtime.metrics().completed, 0);
    assert_eq!(runtime.metrics().canceled, 1);
    assert_eq!(fixture.drops.inputs.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.drops.leases.load(Ordering::SeqCst), 1);
    drop(runtime);
    assert_eq!(fixture.drops.providers.load(Ordering::SeqCst), 1);
}
