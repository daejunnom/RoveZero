//! Finite CPU/mock trace replay; this is neither neural inference nor a D03 A/B.
//! Numeric IDs and input relationships are example-local fixture metadata.

use rz_runtime::{
    Adapter, Backend, BackendResult, Clock, CompletionReceiver, DrainState, Limits, Observation,
    ObservationKind, Resources, RuntimeFault, Scheduler, TerminalEvent,
};
use rz_telemetry::profile::{
    BackendKind, MemoryReservations, ProfileConfig, Stage, Timepoint, TraceClass, TraceCollector,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

const MAX_REQUESTS: usize = 8;
const MAX_BATCH: usize = 4;
const MAX_EXECUTIONS: usize = 2;
const REQUEST_BYTES: u64 = 512;
const BATCH_BYTES: u64 = 256;
const DOMAIN: u64 = 1;
const ALL_CLASSES: [TraceClass; 7] = [
    TraceClass::Cold,
    TraceClass::Warm,
    TraceClass::ParentChild,
    TraceClass::Sibling,
    TraceClass::Transposition,
    TraceClass::Eviction,
    TraceClass::Long,
];

#[derive(Clone, Default)]
struct FixtureClock(Arc<AtomicU64>);

impl FixtureClock {
    fn advance(&self, ticks: u64) {
        self.0.fetch_add(ticks, Ordering::SeqCst);
    }
}

impl Clock for FixtureClock {
    type Tick = u64;

    fn now(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }

    fn elapsed(&self, earlier: u64, later: u64) -> Duration {
        Duration::from_micros(later.saturating_sub(earlier))
    }
}

fn timepoint(tick: u64) -> Timepoint {
    Timepoint {
        domain: DOMAIN,
        elapsed: Duration::from_micros(tick),
    }
}

#[derive(Debug)]
struct Request {
    id: u64,
    deadline: u64,
    key: u8,
    input: u64,
}

#[derive(Debug)]
struct Output {
    id: u64,
    execution: u64,
    value: u64,
}

#[derive(Clone, Debug)]
enum Failure {
    Runtime(RuntimeFault),
    InvalidOutput,
}

struct Reply {
    id: u64,
    outcome: TerminalEvent<Output, Failure>,
}

struct FixtureAdapter {
    clock: FixtureClock,
    next_execution: u64,
}

impl Adapter for FixtureAdapter {
    type RequestId = u64;
    type ExecutionId = u64;
    type Tick = u64;
    type Request = Request;
    type BatchKey = u8;
    type Output = Output;
    type Error = Failure;
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

    fn resources(&self, _: &Request) -> Resources {
        Resources {
            host_bytes: REQUEST_BYTES,
            ..Resources::default()
        }
    }

    fn validate_admission(&mut self, request: &Request) -> Result<(), Failure> {
        if request.id == 0 {
            Err(Failure::InvalidOutput)
        } else {
            Ok(())
        }
    }

    fn is_current(&self, _: &Request) -> bool {
        true
    }

    fn is_canceled(&self, _: &Request) -> bool {
        false
    }

    fn validate_output(&mut self, request: &Request, output: &Output) -> Result<(), Failure> {
        self.clock.advance(1);
        if output.id == request.id && output.value == evaluate(request.input) {
            Ok(())
        } else {
            Err(Failure::InvalidOutput)
        }
    }

    fn allocate_execution_id(&mut self) -> Result<u64, Failure> {
        self.next_execution += 1;
        Ok(self.next_execution)
    }

    fn runtime_error(&self, fault: RuntimeFault) -> Failure {
        Failure::Runtime(fault)
    }

    fn execution_error(&self, _: &Request, error: &Failure) -> Failure {
        error.clone()
    }

    fn terminal(&mut self, request: &Request, outcome: TerminalEvent<Output, Failure>) -> Reply {
        Reply {
            id: request.id,
            outcome,
        }
    }
}

/// Own the input pins and computed host outputs until the fixture completion tick.
struct CpuLease {
    _inputs: Vec<Arc<Request>>,
    ready_at: u64,
    outputs: Option<Vec<BackendResult<FixtureAdapter>>>,
}

struct CpuFixture {
    clock: FixtureClock,
}

fn evaluate(input: u64) -> u64 {
    let mut value = input;
    for _ in 0..64 {
        value = value.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    std::hint::black_box(value)
}

impl Backend<FixtureAdapter> for CpuFixture {
    type Lease = CpuLease;

    fn additional_resources(&self, _: &[Arc<Request>]) -> Resources {
        Resources {
            host_bytes: BATCH_BYTES,
            ..Resources::default()
        }
    }

    fn dispatch(
        &mut self,
        execution: &u64,
        requests: &[Arc<Request>],
    ) -> Result<CpuLease, Failure> {
        let outputs = requests
            .iter()
            .map(|request| BackendResult {
                request_id: request.id,
                output: Ok(Output {
                    id: request.id,
                    execution: *execution,
                    value: evaluate(request.input),
                }),
            })
            .collect();
        self.clock.advance(1);
        Ok(CpuLease {
            _inputs: requests.to_vec(),
            ready_at: self.clock.now() + 3,
            outputs: Some(outputs),
        })
    }

    fn poll(&mut self, lease: &mut CpuLease) -> Poll<Vec<BackendResult<FixtureAdapter>>> {
        if self.clock.now() < lease.ready_at {
            Poll::Pending
        } else {
            Poll::Ready(lease.outputs.take().expect("each lease completes once"))
        }
    }
}

type Runtime = Scheduler<FixtureAdapter, CpuFixture, FixtureClock>;

struct Active {
    receiver: CompletionReceiver<Reply>,
}

struct ExecutionTrace {
    members: HashSet<u64>,
    physical_done: bool,
}

struct Trace {
    collector: TraceCollector<u64, u64>,
    queued_at: HashMap<u64, u64>,
    request_execution: HashMap<u64, u64>,
    validating_at: HashMap<u64, u64>,
    executions: HashMap<u64, ExecutionTrace>,
    record_errors: u64,
    observation_drops: u64,
    observation_counter_overflow: bool,
    peak_tracked_executions: usize,
}

enum TraceObservation {
    Scheduler(Observation<u64, u64, u64>),
    Queue { request: u64, start: u64, end: u64 },
}

impl TraceObservation {
    fn at(&self) -> u64 {
        match self {
            Self::Scheduler(event) => event.at,
            Self::Queue { end, .. } => *end,
        }
    }
}

impl Trace {
    fn new(sample_capacity: usize) -> Result<Self, String> {
        let collector = TraceCollector::try_new(
            ProfileConfig {
                max_requests: MAX_REQUESTS,
                max_executions: MAX_EXECUTIONS,
                sample_capacity,
            },
            DOMAIN,
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            collector,
            queued_at: HashMap::new(),
            request_execution: HashMap::new(),
            validating_at: HashMap::new(),
            executions: HashMap::new(),
            record_errors: 0,
            observation_drops: 0,
            observation_counter_overflow: false,
            peak_tracked_executions: 0,
        })
    }

    fn record<T>(&mut self, result: Result<T, rz_telemetry::profile::TraceError>) {
        if result.is_err() {
            self.record_errors += 1;
        }
    }

    fn reservations(&mut self, runtime: &Runtime, tick: u64) {
        let state = runtime.state();
        let reserved = state.reserved;
        let peak = state.peak_reserved;
        let result = self.collector.observe_reservation_peak(
            MemoryReservations {
                host_bytes: peak.host_bytes,
                device_bytes: peak.device_bytes,
                pinned_bytes: peak.pinned_bytes,
            },
            timepoint(tick),
        );
        self.record(result);
        let result = self.collector.observe_reservations(
            MemoryReservations {
                host_bytes: reserved.host_bytes,
                device_bytes: reserved.device_bytes,
                pinned_bytes: reserved.pinned_bytes,
            },
            timepoint(tick),
        );
        self.record(result);
    }

    /// Drain after every owner action. Dropped events make completeness explicit.
    fn drain(&mut self, runtime: &mut Runtime, class: TraceClass) -> Option<u64> {
        let drain = runtime.take_observations();
        self.observation_drops = self.observation_drops.saturating_add(drain.dropped);
        self.observation_counter_overflow |= drain.counter_overflow;
        self.collector.note_event_loss(drain.dropped);
        if drain.dropped != 0 {
            // A missing physical completion must not leave an auxiliary lease
            // relationship forever. Clear reconstruction only; the runtime's
            // actual leases and the collector's bounded state remain intact.
            self.queued_at.clear();
            self.request_execution.clear();
            self.validating_at.clear();
            self.executions.clear();
        }
        let mut first_dispatched = None;
        let mut work = Vec::with_capacity(drain.events.len().saturating_mul(2));
        for event in drain.events {
            if let ObservationKind::RequestDispatched {
                request,
                dispatch_started_at,
                ..
            } = &event.kind
            {
                first_dispatched.get_or_insert(*request);
                if let Some(start) = self.queued_at.remove(request) {
                    work.push(TraceObservation::Queue {
                        request: *request,
                        start,
                        end: *dispatch_started_at,
                    });
                }
            }
            work.push(TraceObservation::Scheduler(event));
        }
        // Queue end is dispatch start, while dispatch observations are emitted
        // at dispatch end. Replay interval observations at their actual end tick
        // to preserve the collector's global monotonic observation order.
        work.sort_by_key(TraceObservation::at);
        for observation in work {
            match observation {
                TraceObservation::Scheduler(event) => self.observe(event, class),
                TraceObservation::Queue {
                    request,
                    start,
                    end,
                } => {
                    let result = self.collector.record_stage(
                        &request,
                        Stage::Queue,
                        timepoint(start),
                        timepoint(end),
                    );
                    self.record(result);
                }
            }
        }
        first_dispatched
    }

    fn observe(&mut self, event: Observation<u64, u64, u64>, class: TraceClass) {
        let at = timepoint(event.at);
        let reserved = event.reserved;
        let peak = event.peak_reserved;
        let result = self.collector.observe_reservation_peak(
            MemoryReservations {
                host_bytes: peak.host_bytes,
                device_bytes: peak.device_bytes,
                pinned_bytes: peak.pinned_bytes,
            },
            at,
        );
        self.record(result);
        let result = self.collector.observe_reservations(
            MemoryReservations {
                host_bytes: reserved.host_bytes,
                device_bytes: reserved.device_bytes,
                pinned_bytes: reserved.pinned_bytes,
            },
            at,
        );
        self.record(result);
        match event.kind {
            ObservationKind::Admitted { request } => {
                self.queued_at.insert(request, event.at);
            }
            ObservationKind::Dispatched {
                execution,
                items,
                dispatch_started_at,
            } => {
                let result = self.collector.begin_execution(
                    execution,
                    class,
                    BackendKind::CpuMock,
                    items,
                    at,
                );
                let began = result.is_ok();
                self.record(result);
                if began {
                    let result = self.collector.record_dispatch(
                        &execution,
                        timepoint(dispatch_started_at),
                        at,
                    );
                    self.record(result);
                }
                if began && self.executions.len() < MAX_EXECUTIONS {
                    self.executions.insert(
                        execution,
                        ExecutionTrace {
                            members: HashSet::new(),
                            physical_done: false,
                        },
                    );
                    self.peak_tracked_executions =
                        self.peak_tracked_executions.max(self.executions.len());
                } else if began {
                    // Reconstruction metadata has the same finite execution
                    // bound, including traces with missing completion events.
                    self.record_errors += 1;
                    self.collector.note_event_loss(1);
                }
            }
            ObservationKind::RequestDispatched {
                request, execution, ..
            } => {
                self.request_execution.insert(request, execution);
                if let Some(trace) = self.executions.get_mut(&execution) {
                    trace.members.insert(request);
                }
            }
            ObservationKind::PhysicalCompleted { execution } => {
                let result = self.collector.finish_execution(&execution, at);
                self.record(result);
                if let Some(trace) = self.executions.get_mut(&execution) {
                    trace.physical_done = true;
                }
            }
            ObservationKind::ValidationStarted { request, .. } => {
                self.validating_at.insert(request, event.at);
            }
            ObservationKind::ValidationFinished { request, .. } => {
                if let Some(start) = self.validating_at.remove(&request) {
                    let result = self.collector.record_stage(
                        &request,
                        Stage::Validation,
                        timepoint(start),
                        at,
                    );
                    self.record(result);
                }
            }
            ObservationKind::Finished { request, kind, .. } => {
                let result = self.collector.finish_request(&request, kind, at);
                self.record(result);
                self.queued_at.remove(&request);
            }
            ObservationKind::Rejected { .. } => {
                let result = self.collector.record_rejection(at);
                self.record(result);
            }
            ObservationKind::ReservationChanged => {}
        }
    }

    fn close(&mut self, request: u64, at: u64) {
        let result = self.collector.close_request(&request, timepoint(at));
        self.record(result);
        self.queued_at.remove(&request);
        self.validating_at.remove(&request);
        if let Some(execution) = self.request_execution.remove(&request) {
            if let Some(trace) = self.executions.get_mut(&execution) {
                trace.members.remove(&request);
            }
        }
    }

    fn retire(&mut self, at: u64) {
        let retired: Vec<u64> = self
            .executions
            .iter()
            .filter(|(_, trace)| trace.physical_done && trace.members.is_empty())
            .map(|(id, _)| *id)
            .collect();
        for execution in retired {
            let result = self.collector.retire_execution(&execution, timepoint(at));
            self.record(result);
            self.executions.remove(&execution);
        }
    }
}

fn class_name(class: TraceClass) -> &'static str {
    match class {
        TraceClass::Cold => "cold",
        TraceClass::Warm => "warm",
        TraceClass::ParentChild => "parent-child",
        TraceClass::Sibling => "sibling",
        TraceClass::Transposition => "transposition",
        TraceClass::Eviction => "eviction",
        TraceClass::Long => "long",
    }
}

fn relation(class: TraceClass) -> &'static str {
    match class {
        TraceClass::Cold => "independent-inputs",
        TraceClass::Warm => "repeated-input-fixture-no-warm-state",
        TraceClass::ParentChild => "sequential-parent-child-labels",
        TraceClass::Sibling => "four-inputs-per-parent-label",
        TraceClass::Transposition => "revisited-input-no-dedup",
        TraceClass::Eviction => "revisited-input-after-four-other-labels-no-cache",
        TraceClass::Long => "long-sequential-input-labels",
    }
}

fn fixture_input(class: TraceClass, index: u64) -> u64 {
    match class {
        TraceClass::Cold => index.wrapping_mul(104729),
        TraceClass::Warm => index % 2,
        TraceClass::ParentChild | TraceClass::Long => index,
        TraceClass::Sibling => (index / 4) * 16 + index % 4,
        TraceClass::Transposition => index % 4,
        TraceClass::Eviction => {
            if index % 5 == 4 {
                0
            } else {
                index + 1
            }
        }
    }
}

struct Run {
    trace: Trace,
    metrics: rz_telemetry::Snapshot,
    wall: Duration,
    checksum: u64,
    retained_receipt_checks: u64,
    duplicate_rejections: u64,
    successful_receipts: u64,
    final_reserved: Resources,
}

fn replay(class: TraceClass, count: usize, capacity: usize) -> Result<Run, String> {
    let clock = FixtureClock::default();
    let mut runtime = Scheduler::new(
        FixtureAdapter {
            clock: clock.clone(),
            next_execution: 0,
        },
        CpuFixture {
            clock: clock.clone(),
        },
        clock.clone(),
        Limits {
            max_requests: MAX_REQUESTS,
            max_batch_items: MAX_BATCH,
            max_executions: MAX_EXECUTIONS,
            max_batch_wait: Duration::from_micros(2),
            max_queue_age: Duration::from_micros(100),
            deadline_reserve: Duration::from_micros(2),
            memory: Resources {
                host_bytes: MAX_REQUESTS as u64 * REQUEST_BYTES
                    + MAX_EXECUTIONS as u64 * BATCH_BYTES,
                ..Resources::default()
            },
        },
        capacity,
    )
    .map_err(|error| format!("runtime configuration: {error:?}"))?;
    let mut trace = Trace::new(capacity)?;
    let mut active = HashMap::<u64, Active>::new();
    let mut submitted = 0_usize;
    let mut completed = 0_usize;
    let mut canceled_running = false;
    let mut duplicate_rejections = 0_u64;
    let mut checksum = 0_u64;
    let mut retained_receipt_checks = 0_u64;
    let mut successful_receipts = 0_u64;
    let started = Instant::now();
    let max_steps = count * 16 + 64;
    for _ in 0..max_steps {
        if started.elapsed() > Duration::from_secs(10) {
            return Err("finite replay exceeded 10 second wall limit".into());
        }
        while submitted < count && runtime.state().reserved_requests < MAX_REQUESTS {
            let id = submitted as u64 + 1;
            let prepare_at = clock.now();
            let result = trace
                .collector
                .begin_request(id, class, timepoint(prepare_at));
            trace.record(result);
            let input = fixture_input(class, submitted as u64);
            clock.advance(1);
            let result = trace.collector.record_stage(
                &id,
                Stage::CpuPreparation,
                timepoint(prepare_at),
                timepoint(clock.now()),
            );
            trace.record(result);
            let request = Request {
                id,
                deadline: clock.now() + 1000,
                key: if class == TraceClass::Sibling {
                    (submitted / 4 % 2) as u8
                } else {
                    0
                },
                input,
            };
            let receiver = runtime.submit(request).map_err(|rejected| {
                format!("unexpected admission rejection: {:?}", rejected.error)
            })?;
            active.insert(id, Active { receiver });
            trace.drain(&mut runtime, class);
            if submitted == 0 {
                let duplicate = runtime.submit(Request {
                    id,
                    deadline: clock.now() + 1000,
                    key: 0,
                    input,
                });
                match duplicate {
                    Err(rejected)
                        if matches!(
                            rejected.error,
                            Failure::Runtime(RuntimeFault::DuplicateRequest)
                        ) =>
                    {
                        duplicate_rejections += 1;
                    }
                    _ => return Err("live duplicate request was not rejected".into()),
                }
                trace.drain(&mut runtime, class);
            }
            submitted += 1;
        }
        runtime.pump();
        let first_dispatched = trace.drain(&mut runtime, class);
        if !canceled_running && runtime.state().executions > 0 {
            // When observations are disabled, ID 1 is still the first FIFO input.
            let canceled = first_dispatched.unwrap_or(1);
            if runtime.cancel(&canceled) {
                canceled_running = true;
            }
            trace.drain(&mut runtime, class);
        }
        let mut ids: Vec<u64> = active.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let before = runtime.state();
            match active[&id].receiver.try_recv() {
                Ok(reply) => {
                    if reply.id != id || before.reserved_requests == 0 {
                        return Err("receipt identity or retained reservation is invalid".into());
                    }
                    retained_receipt_checks += 1;
                    match reply.outcome {
                        TerminalEvent::Success(output) => {
                            let after = runtime.state();
                            if before.reserved_requests != after.reserved_requests + 1
                                || before.reserved.host_bytes
                                    != after.reserved.host_bytes + REQUEST_BYTES
                            {
                                return Err(
                                    "successful receipt did not release its exact reservation"
                                        .into(),
                                );
                            }
                            let at = clock.now();
                            let result = trace.collector.consume_request(
                                &id,
                                Some(&output.execution),
                                timepoint(at),
                            );
                            trace.record(result);
                            checksum = checksum.wrapping_add(output.value);
                            successful_receipts += 1;
                            clock.advance(1);
                            let result = trace.collector.record_stage(
                                &id,
                                Stage::Backup,
                                timepoint(at),
                                timepoint(clock.now()),
                            );
                            trace.record(result);
                            let output_at = clock.now();
                            std::hint::black_box(checksum);
                            clock.advance(1);
                            let result = trace.collector.record_stage(
                                &id,
                                Stage::Output,
                                timepoint(output_at),
                                timepoint(clock.now()),
                            );
                            trace.record(result);
                        }
                        TerminalEvent::Canceled => {
                            let after = runtime.state();
                            if before.executions > 0
                                && (before.reserved_requests != after.reserved_requests
                                    || before.reserved != after.reserved)
                            {
                                return Err(
                                    "running cancellation released physical input reservations"
                                        .into(),
                                );
                            }
                        }
                        outcome => return Err(format!("unexpected terminal: {outcome:?}")),
                    }
                    trace.close(id, clock.now());
                    active.remove(&id);
                    completed += 1;
                    trace.reservations(&runtime, clock.now());
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(error) => return Err(format!("terminal mailbox failed: {error}")),
            }
        }
        trace.retire(clock.now());
        if completed == count && runtime.state().executions == 0 {
            runtime.begin_shutdown(clock.now() + 10);
            trace.drain(&mut runtime, class);
            if runtime.drain_state() != DrainState::Drained
                || runtime.state().reserved_requests != 0
                || runtime.state().reserved != Resources::default()
                || !canceled_running
            {
                return Err("replay did not finish, release its receipts and drain".into());
            }
            trace.reservations(&runtime, clock.now());
            return Ok(Run {
                trace,
                metrics: runtime.metrics(),
                wall: started.elapsed(),
                checksum,
                retained_receipt_checks,
                duplicate_rejections,
                successful_receipts,
                final_reserved: runtime.state().reserved,
            });
        }
        clock.advance(1);
    }
    Err(format!("finite replay exhausted {max_steps} owner steps"))
}

struct Options {
    scenario: Option<TraceClass>,
    requests: usize,
    capacity: usize,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        scenario: None,
        requests: 32,
        capacity: 512,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--help" || argument == "-h" {
            println!("cpu_trace --scenario all|cold|warm|parent-child|sibling|transposition|eviction|long --requests 1..65536 --sample-capacity 0..65536");
            std::process::exit(0);
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {argument}"))?;
        match argument.as_str() {
            "--scenario" => {
                options.scenario = if value == "all" {
                    None
                } else {
                    Some(
                        ALL_CLASSES
                            .into_iter()
                            .find(|class| class_name(*class) == value)
                            .ok_or_else(|| format!("unknown scenario {value}"))?,
                    )
                };
            }
            "--requests" => {
                options.requests = value.parse().map_err(|_| "invalid request count")?;
                if options.requests == 0 || options.requests > 65_536 {
                    return Err("request count must be in 1..65536".into());
                }
            }
            "--sample-capacity" => {
                options.capacity = value.parse().map_err(|_| "invalid sample capacity")?;
                if options.capacity > 65_536 {
                    return Err("sample capacity must be in 0..65536".into());
                }
            }
            _ => return Err(format!("unknown argument {argument}")),
        }
    }
    Ok(options)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cpu_trace: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = options()?;
    println!("# format=rove-cpu-trace-tsv-v1 backend=cpu-mock clock=fixture-monotonic-us wall-clock=std-Instant baseline-only=true neural-inference=false cache=false dedup=false warm-state=false gpu=unexecuted");
    println!("# max_requests={MAX_REQUESTS} max_batch_items={MAX_BATCH} max_executions={MAX_EXECUTIONS} sample_and_observation_capacity={} wall_limit_seconds=10 step_limit_per_scenario=requests*16+64", options.capacity);
    println!("scenario\tfield\tvalue\tunit");
    for class in ALL_CLASSES {
        if options.scenario.is_some_and(|selected| selected != class) {
            continue;
        }
        let result = replay(class, options.requests, options.capacity)?;
        print_report(class, &result);
    }
    Ok(())
}

fn print_report(class: TraceClass, result: &Run) {
    let name = class_name(class);
    let report = result.trace.collector.report();
    field(name, "input_relation", relation(class), "fixture");
    field(name, "admitted", result.metrics.admitted, "requests");
    field(name, "completed", result.metrics.completed, "requests");
    field(name, "canceled", result.metrics.canceled, "requests");
    field(
        name,
        "actual_consumed_requests",
        result.successful_receipts,
        "requests",
    );
    field(
        name,
        "duplicate_rejections",
        result.duplicate_rejections,
        "requests",
    );
    field(
        name,
        "retained_receipt_checks",
        result.retained_receipt_checks,
        "requests",
    );
    field(name, "actual_wall", result.wall.as_nanos(), "ns");
    let rate = if result.wall.is_zero() {
        "na".into()
    } else {
        format!(
            "{:.6}",
            result.successful_receipts as f64 / result.wall.as_secs_f64()
        )
    };
    field(name, "actual_wall_consumed_rate", rate, "requests/s");
    field(name, "checksum", result.checksum, "fixture");
    optional_duration(name, "fixture_wall_span", report.wall_span);
    optional_rate(
        name,
        "fixture_physical_rate",
        report.physical_items_per_second,
    );
    optional_rate(
        name,
        "fixture_consumed_rate",
        report.consumed_requests_per_second,
    );
    field(
        name,
        "physical_executions_started",
        report.physical_executions_started,
        "batches",
    );
    field(
        name,
        "physical_executions_completed",
        report.physical_executions_completed,
        "batches",
    );
    field(
        name,
        "physical_items_started",
        report.physical_items_started,
        "requests",
    );
    field(
        name,
        "physical_items_completed",
        report.physical_items_completed,
        "requests",
    );
    field(
        name,
        "consumed_requests",
        report.consumed_requests,
        "requests",
    );
    field(
        name,
        "physical_items_consumed",
        report.physical_items_consumed,
        "requests",
    );
    field(
        name,
        "cache_consumed_requests",
        report.cache_consumed_requests,
        "requests",
    );
    field(
        name,
        "unused_items_finalized",
        report.unused_items_finalized,
        "requests",
    );
    field(
        name,
        "completed_unconsumed_items",
        report.completed_unconsumed_items,
        "requests",
    );
    field(
        name,
        "profile.logical.completed",
        report.logical.completed,
        "requests",
    );
    field(
        name,
        "profile.logical.canceled",
        report.logical.canceled,
        "requests",
    );
    field(
        name,
        "profile.logical.expired",
        report.logical.expired,
        "requests",
    );
    field(
        name,
        "profile.logical.stale",
        report.logical.stale,
        "requests",
    );
    field(
        name,
        "profile.logical.failed",
        report.logical.failed,
        "requests",
    );
    field(
        name,
        "profile.rejected_requests",
        report.rejected_requests,
        "requests",
    );
    field(
        name,
        "batch_samples.total",
        report.batch_samples_total,
        "samples",
    );
    field(
        name,
        "batch_samples.retained",
        report.batch_samples_retained,
        "samples",
    );
    field(
        name,
        "batch_samples.dropped",
        report.batch_samples_dropped,
        "samples",
    );
    for (items, batches) in &report.batch_distribution {
        field(
            name,
            &format!("batch_size.{items}"),
            batches,
            "retained-batches",
        );
    }
    for stage in &report.stages {
        let prefix = format!("stage.{}", stage_name(stage.stage));
        field(
            name,
            &format!("{prefix}.total"),
            stage.latency.total,
            "samples",
        );
        field(
            name,
            &format!("{prefix}.retained"),
            stage.latency.retained,
            "samples",
        );
        field(
            name,
            &format!("{prefix}.dropped"),
            stage.latency.dropped,
            "samples",
        );
        optional_duration(name, &format!("{prefix}.p50"), stage.latency.p50);
        optional_duration(name, &format!("{prefix}.p95"), stage.latency.p95);
        optional_duration(name, &format!("{prefix}.p99"), stage.latency.p99);
    }
    let completeness = &report.completeness;
    field(
        name,
        "completeness.stage_samples_dropped",
        completeness.stage_samples_dropped,
        "bool",
    );
    field(
        name,
        "completeness.batch_samples_dropped",
        completeness.batch_samples_dropped,
        "bool",
    );
    field(
        name,
        "completeness.tracking_capacity_rejections",
        completeness.tracking_capacity_rejections,
        "events",
    );
    field(
        name,
        "completeness.request_ids_evicted",
        completeness.request_ids_evicted,
        "ids",
    );
    field(
        name,
        "completeness.execution_ids_evicted",
        completeness.execution_ids_evicted,
        "ids",
    );
    field(
        name,
        "completeness.observation_errors",
        completeness.observation_errors,
        "events",
    );
    field(
        name,
        "completeness.domain_errors",
        completeness.domain_errors,
        "events",
    );
    field(
        name,
        "completeness.time_errors",
        completeness.time_errors,
        "events",
    );
    field(
        name,
        "completeness.active_requests",
        completeness.active_requests,
        "requests",
    );
    field(
        name,
        "completeness.active_executions",
        completeness.active_executions,
        "batches",
    );
    field(
        name,
        "completeness.external_events_dropped",
        completeness.external_events_dropped,
        "events",
    );
    field(
        name,
        "completeness.caller_ids_must_be_unique",
        completeness.caller_ids_must_be_unique,
        "precondition",
    );
    field(
        name,
        "completeness.duplicate_checks_cover_all_ids",
        completeness.duplicate_checks_cover_all_ids,
        "bool",
    );
    field(
        name,
        "completeness.fixture_record_errors",
        result.trace.record_errors,
        "events",
    );
    field(
        name,
        "completeness.observation_counter_overflow",
        result.trace.observation_counter_overflow,
        "bool",
    );
    field(
        name,
        "completeness.profile_counter_overflow",
        report.counter_overflow,
        "bool",
    );
    field(
        name,
        "completeness.scheduler_counter_overflow",
        result.metrics.counter_overflow,
        "bool",
    );
    field(
        name,
        "auxiliary.peak_executions",
        result.trace.peak_tracked_executions,
        "batches",
    );
    field(
        name,
        "reserved_peak.host",
        report.reserved_peak.host_bytes,
        "bytes",
    );
    field(
        name,
        "reserved_peak.device",
        report.reserved_peak.device_bytes,
        "bytes",
    );
    field(
        name,
        "reserved_peak.pinned",
        report.reserved_peak.pinned_bytes,
        "bytes",
    );
    field(
        name,
        "reserved_final.host",
        result.final_reserved.host_bytes,
        "bytes",
    );
    field(
        name,
        "reserved_final.device",
        result.final_reserved.device_bytes,
        "bytes",
    );
    field(
        name,
        "reserved_final.pinned",
        result.final_reserved.pinned_bytes,
        "bytes",
    );
}

fn field(scenario: &str, name: &str, value: impl std::fmt::Display, unit: &str) {
    println!("{scenario}\t{name}\t{value}\t{unit}");
}

fn optional_duration(scenario: &str, name: &str, value: Option<Duration>) {
    field(
        scenario,
        name,
        value.map_or_else(|| "na".into(), |duration| duration.as_nanos().to_string()),
        "fixture-ns",
    );
}

fn optional_rate(scenario: &str, name: &str, value: Option<f64>) {
    field(
        scenario,
        name,
        value.map_or_else(|| "na".into(), |rate| format!("{rate:.6}")),
        "fixture-requests/s",
    );
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::CpuPreparation => "cpu_preparation",
        Stage::Queue => "queue",
        Stage::Dispatch => "dispatch",
        Stage::Transfer => "transfer",
        Stage::BackendPhysical => "backend_physical",
        Stage::Validation => "validation",
        Stage::Backup => "backup",
        Stage::Output => "output",
        Stage::EndToEnd => "end_to_end",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seven_traces_separate_physical_work_from_backup_and_record_every_boundary() {
        for class in ALL_CLASSES {
            let run = replay(class, 32, 512).unwrap();
            let report = run.trace.collector.report();
            assert_eq!(run.metrics.admitted, 32);
            assert_eq!(report.logical.completed, 31);
            assert_eq!(report.logical.canceled, 1);
            assert_eq!(report.physical_items_completed, 32);
            assert_eq!(report.consumed_requests, 31);
            assert_eq!(run.successful_receipts, 31);
            assert_eq!(report.unused_items_finalized, 1);
            assert_eq!(report.completed_unconsumed_items, 0);
            assert_eq!(report.rejected_requests, 1);
            assert_eq!(report.completeness.external_events_dropped, 0);
            assert_eq!(report.completeness.observation_errors, 0);
            assert_eq!(report.completeness.active_requests, 0);
            assert_eq!(report.completeness.active_executions, 0);
            assert_eq!(report.reserved_peak.host_bytes, 4608);
            assert_eq!(report.reserved_current.host_bytes, 0);
            assert_eq!(run.retained_receipt_checks, 32);
            assert_eq!(run.final_reserved, Resources::default());
            for stage in &report.stages {
                let expected = match stage.stage {
                    Stage::CpuPreparation | Stage::Queue | Stage::EndToEnd => 32,
                    Stage::Dispatch | Stage::BackendPhysical => 8,
                    Stage::Validation | Stage::Backup | Stage::Output => 31,
                    Stage::Transfer => 0,
                };
                assert_eq!(stage.latency.total, expected, "{class:?} {:?}", stage.stage);
                assert_eq!(stage.latency.dropped, 0);
                assert_eq!(stage.latency.p99.is_some(), expected > 0);
            }
        }
    }

    #[test]
    fn zero_event_capacity_preserves_runtime_results_and_reports_incomplete_tracking() {
        let run = replay(TraceClass::Cold, 16, 0).unwrap();
        let report = run.trace.collector.report();
        assert_eq!(run.metrics.completed, 15);
        assert_eq!(run.metrics.canceled, 1);
        assert_eq!(run.successful_receipts, 15);
        assert_eq!(run.metrics.physical_completed, 4);
        assert_eq!(run.final_reserved, Resources::default());
        assert!(report.completeness.external_events_dropped > 0);
        assert!(report.completeness.observation_errors > 0);
        assert!(report.completeness.active_requests > 0);
        assert_eq!(report.physical_items_completed, 0);
        assert_eq!(report.consumed_requests, 0);
        assert!(run.trace.executions.len() <= MAX_EXECUTIONS);
    }

    #[test]
    fn a_single_canceled_item_drains_late_work_without_creating_a_backup_sample() {
        let run = replay(TraceClass::Long, 1, 512).unwrap();
        let report = run.trace.collector.report();
        assert_eq!(report.logical.canceled, 1);
        assert_eq!(report.physical_items_completed, 1);
        assert_eq!(report.consumed_requests, 0);
        assert_eq!(report.unused_items_finalized, 1);
        assert_eq!(report.completeness.observation_errors, 0);
        assert_eq!(run.final_reserved, Resources::default());
        for stage in [Stage::Validation, Stage::Backup, Stage::Output] {
            let summary = &report.stages[stage as usize].latency;
            assert_eq!(summary.total, 0);
            assert_eq!(summary.p50, None);
        }
    }

    #[test]
    fn long_lossy_traces_keep_auxiliary_reconstruction_within_runtime_limits() {
        for capacity in [1, 2, 4, 16] {
            let run = replay(TraceClass::Long, 2048, capacity).unwrap();
            let report = run.trace.collector.report();
            assert_eq!(run.metrics.completed, 2047);
            assert_eq!(run.metrics.canceled, 1);
            assert_eq!(run.successful_receipts, 2047);
            assert_eq!(run.final_reserved, Resources::default());
            assert!(report.completeness.external_events_dropped > 0);
            assert!(run.trace.executions.len() <= MAX_EXECUTIONS);
            assert!(run.trace.peak_tracked_executions <= MAX_EXECUTIONS);
            assert!(run.trace.queued_at.len() <= MAX_REQUESTS);
            assert!(run.trace.request_execution.len() <= MAX_REQUESTS);
            assert!(run.trace.validating_at.len() <= MAX_REQUESTS);
        }
    }
}
