use crate::{
    Adapter, Backend, BackendResult, Clock, Limits, Resources, RuntimeFault, TerminalEvent,
};
use rz_telemetry::{Event, FinishKind, Metrics, Snapshot};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{mpsc, Arc, Mutex};
use std::task::Poll;
use std::time::Duration;

/// Admission failures return ownership to Search. They never also send a terminal result.
#[derive(Debug)]
pub struct Rejected<R, E> {
    pub request: R,
    pub error: E,
}

/// A bounded completion mailbox retaining its pending-output reservation.
/// Receiving transfers payload ownership to the caller and releases this pin.
pub struct CompletionReceiver<T> {
    receiver: mpsc::Receiver<Delivery<T>>,
}

impl<T> CompletionReceiver<T> {
    pub fn recv(&self) -> Result<T, mpsc::RecvError> {
        self.receiver.recv().map(|delivery| delivery.result)
    }

    pub fn try_recv(&self) -> Result<T, mpsc::TryRecvError> {
        self.receiver.try_recv().map(|delivery| delivery.result)
    }

    pub fn recv_timeout(&self, timeout: Duration) -> Result<T, mpsc::RecvTimeoutError> {
        self.receiver
            .recv_timeout(timeout)
            .map(|delivery| delivery.result)
    }
}

pub type SubmitResult<A> = Result<
    CompletionReceiver<<A as Adapter>::TerminalResult>,
    Rejected<<A as Adapter>::Request, <A as Adapter>::Error>,
>;

struct Delivery<T> {
    result: T,
    _reservation: Arc<Reservation>,
}

#[derive(Default)]
struct Budget {
    bytes: Resources,
    requests: usize,
}

struct Reservation {
    budget: Arc<Mutex<Budget>>,
    bytes: Resources,
    request_slot: bool,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut budget = self.budget.lock().expect("reservation ledger");
        budget.bytes = budget.bytes.subtract(self.bytes);
        if self.request_slot {
            budget.requests = budget.requests.checked_sub(1).expect("request reservation");
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct State {
    pub logical_requests: usize,
    /// Includes unread completions and logically canceled physical inputs.
    pub reserved_requests: usize,
    pub queued: usize,
    pub executions: usize,
    pub reserved: Resources,
    pub closed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrainState {
    Open,
    Draining,
    Drained,
    /// Resources remain pinned. The owner may keep pumping to reclaim them later.
    TimedOut {
        executions: usize,
        reserved: Resources,
    },
}

struct Entry<A: Adapter> {
    request: Arc<A::Request>,
    admitted_at: A::Tick,
    reservation: Arc<Reservation>,
    running: bool,
    reply: mpsc::SyncSender<Delivery<A::TerminalResult>>,
}

struct Held<A: Adapter> {
    id: A::RequestId,
    request: Arc<A::Request>,
    _reservation: Arc<Reservation>,
}

struct Execution<A: Adapter, P: Backend<A>> {
    // Physical ID never substitutes for subscriber/selection identity.
    _id: A::ExecutionId,
    inputs: Vec<Held<A>>,
    _overhead: Arc<Reservation>,
    lease: P::Lease,
}

/// Single-owner scheduler. Call `pump` from an embedding event loop/worker.
///
/// All methods linearize logical transitions on this owner. Backend work may
/// run on other threads/devices, while the owner continues cancellation and
/// deadline checks. Result receivers are bounded to one item and do not block
/// the owner. Search must recheck generation/deadline/selection before backup.
///
/// Explicit shutdown is preferred. Dropping with pending physical work cancels
/// subscribers, then intentionally retains the bounded provider/lease/input
/// bundle rather than freeing buffers still borrowed by a device. This is a
/// safety quarantine, not successful drain; use `drain_state` to report failure.
pub struct Scheduler<A, P, C>
where
    A: Adapter,
    P: Backend<A>,
    C: Clock<Tick = A::Tick>,
{
    adapter: A,
    provider: Option<P>,
    clock: C,
    limits: Limits,
    entries: HashMap<A::RequestId, Entry<A>>,
    queue: VecDeque<A::RequestId>,
    executions: Vec<Execution<A, P>>,
    budget: Arc<Mutex<Budget>>,
    closed: bool,
    drain_deadline: Option<A::Tick>,
    metrics: Metrics,
}

impl<A, P, C> Scheduler<A, P, C>
where
    A: Adapter,
    P: Backend<A>,
    C: Clock<Tick = A::Tick>,
{
    pub fn new(
        adapter: A,
        provider: P,
        clock: C,
        limits: Limits,
        metric_sample_capacity: usize,
    ) -> Result<Self, RuntimeFault> {
        let limits = limits.validate()?;
        Ok(Self {
            adapter,
            provider: Some(provider),
            clock,
            limits,
            entries: HashMap::new(),
            queue: VecDeque::new(),
            executions: Vec::new(),
            budget: Arc::new(Mutex::new(Budget::default())),
            closed: false,
            drain_deadline: None,
            metrics: Metrics::new(metric_sample_capacity),
        })
    }

    pub fn state(&self) -> State {
        let budget = self.budget.lock().expect("reservation ledger");
        State {
            logical_requests: self.entries.len(),
            reserved_requests: budget.requests,
            queued: self.queue.len(),
            executions: self.executions.len(),
            reserved: budget.bytes,
            closed: self.closed,
        }
    }

    pub fn metrics(&self) -> Snapshot {
        self.metrics.snapshot()
    }

    pub fn submit(&mut self, request: A::Request) -> SubmitResult<A> {
        let id = self.adapter.request_id(&request);
        let fault = if self.closed {
            Some(RuntimeFault::Closed)
        } else if self.entries.contains_key(&id)
            || self
                .executions
                .iter()
                .any(|execution| execution.inputs.iter().any(|input| input.id == id))
        {
            Some(RuntimeFault::DuplicateRequest)
        } else if self.state().reserved_requests >= self.limits.max_requests {
            Some(RuntimeFault::QueueFull)
        } else {
            None
        };
        if let Some(fault) = fault {
            return self.reject(request, self.adapter.runtime_error(fault));
        }
        if let Err(error) = self.adapter.validate_admission(&request) {
            return self.reject(request, error);
        }
        // Validation itself costs time; sample after it, not before it.
        if let Some(event) = self.invalid(&request, self.clock.now()) {
            let fault = match event {
                TerminalEvent::Canceled => RuntimeFault::Canceled,
                TerminalEvent::Expired => RuntimeFault::Expired,
                TerminalEvent::Stale => RuntimeFault::Stale,
                _ => unreachable!("invalid only returns lifetime facts"),
            };
            return self.reject(request, self.adapter.runtime_error(fault));
        }
        let resources = self.adapter.resources(&request);
        let reservation = match self.reserve(resources, true) {
            Ok(reservation) => reservation,
            Err(fault) => return self.reject(request, self.adapter.runtime_error(fault)),
        };
        let (reply, receiver) = mpsc::sync_channel(1);
        self.entries.insert(
            id.clone(),
            Entry {
                request: Arc::new(request),
                admitted_at: self.clock.now(),
                reservation,
                running: false,
                reply,
            },
        );
        self.queue.push_back(id);
        self.metrics.record(Event::Admitted);
        Ok(CompletionReceiver { receiver })
    }

    fn reject<T>(
        &mut self,
        request: A::Request,
        error: A::Error,
    ) -> Result<T, Rejected<A::Request, A::Error>> {
        self.metrics.record(Event::Rejected);
        Err(Rejected { request, error })
    }

    fn reserve(
        &self,
        bytes: Resources,
        request_slot: bool,
    ) -> Result<Arc<Reservation>, RuntimeFault> {
        let mut budget = self.budget.lock().expect("reservation ledger");
        let total = budget
            .bytes
            .checked_add(bytes)
            .ok_or(RuntimeFault::ResourceOverflow)?;
        if !total.fits(self.limits.memory) {
            return Err(RuntimeFault::ResourceLimit);
        }
        if request_slot && budget.requests >= self.limits.max_requests {
            return Err(RuntimeFault::QueueFull);
        }
        let requests = budget
            .requests
            .checked_add(usize::from(request_slot))
            .ok_or(RuntimeFault::ResourceOverflow)?;
        budget.bytes = total;
        budget.requests = requests;
        Ok(Arc::new(Reservation {
            budget: Arc::clone(&self.budget),
            bytes,
            request_slot,
        }))
    }

    fn invalid(
        &self,
        request: &A::Request,
        now: A::Tick,
    ) -> Option<TerminalEvent<A::Output, A::Error>> {
        if self.adapter.is_canceled(request) {
            Some(TerminalEvent::Canceled)
        } else if !self.adapter.is_current(request) {
            Some(TerminalEvent::Stale)
        } else if now >= self.adapter.deadline(request) {
            Some(TerminalEvent::Expired)
        } else {
            None
        }
    }

    /// Logical cancellation. Running input/Lease bytes remain owned by Execution.
    pub fn cancel(&mut self, request_id: &A::RequestId) -> bool {
        self.finish(request_id, TerminalEvent::Canceled)
    }

    fn finish(&mut self, id: &A::RequestId, event: TerminalEvent<A::Output, A::Error>) -> bool {
        let Some(entry) = self.entries.remove(id) else {
            return false;
        };
        // This owner's final clock/validity read is the success acceptance point.
        // Search still rechecks at its separate backup commit boundary.
        let event = if matches!(event, TerminalEvent::Success(_)) {
            self.invalid(&entry.request, self.clock.now())
                .unwrap_or(event)
        } else {
            event
        };
        if !entry.running {
            self.queue.retain(|queued| queued != id);
        }
        let kind = match &event {
            TerminalEvent::Success(_) => FinishKind::Completed,
            TerminalEvent::Canceled => FinishKind::Canceled,
            TerminalEvent::Expired => FinishKind::Expired,
            TerminalEvent::Stale => FinishKind::Stale,
            TerminalEvent::Failure(_) => FinishKind::Failed,
        };
        let latency = self.clock.elapsed(entry.admitted_at, self.clock.now());
        let result = self.adapter.terminal(&entry.request, event);
        // A sender is used once, on removal from entries, so this never waits.
        if entry
            .reply
            .try_send(Delivery {
                result,
                _reservation: Arc::clone(&entry.reservation),
            })
            .is_err()
        {
            self.metrics.record(Event::ReceiverDropped);
        }
        self.metrics.record(Event::Finished { kind, latency });
        true
    }

    /// Finite maintenance step: sweep logical invalidations, poll existing
    /// executions, then dispatch up to the configured physical concurrency.
    pub fn pump(&mut self) {
        let ids: Vec<_> = self.entries.keys().cloned().collect();
        for id in ids {
            let entry = &self.entries[&id];
            let now = self.clock.now();
            let event = self.invalid(&entry.request, now).or_else(|| {
                (!entry.running
                    && self.clock.elapsed(entry.admitted_at, now) >= self.limits.max_queue_age)
                    .then(|| {
                        TerminalEvent::Failure(
                            self.adapter.runtime_error(RuntimeFault::QueueAgeExceeded),
                        )
                    })
            });
            if let Some(event) = event {
                self.finish(&id, event);
            }
        }
        let mut index = 0;
        while index < self.executions.len() {
            let completion = self
                .provider
                .as_mut()
                .expect("provider alive")
                .poll(&mut self.executions[index].lease);
            match completion {
                Poll::Pending => index += 1,
                Poll::Ready(outputs) => {
                    let execution = self.executions.remove(index);
                    self.metrics.record(Event::PhysicalCompleted);
                    self.complete(execution, outputs);
                }
            }
        }
        while !self.closed && self.executions.len() < self.limits.max_executions {
            let batch = self.next_batch();
            if batch.is_empty() {
                break;
            }
            self.dispatch(batch);
        }
    }

    fn next_batch(&self) -> Vec<A::RequestId> {
        let Some(first_id) = self.queue.front() else {
            return Vec::new();
        };
        let first = &self.entries[first_id];
        let key = self.adapter.batch_key(&first.request);
        let mut batch = Vec::new();
        let mut incompatible = false;
        let now = self.clock.now();
        let mut near_deadline = false;
        for id in &self.queue {
            let entry = &self.entries[id];
            if self.adapter.batch_key(&entry.request) != key {
                incompatible = true;
                break;
            }
            batch.push(id.clone());
            near_deadline |= self
                .clock
                .elapsed(now, self.adapter.deadline(&entry.request))
                <= self.limits.deadline_reserve;
            if batch.len() == self.limits.max_batch_items {
                break;
            }
        }
        let ready = batch.len() == self.limits.max_batch_items
            || incompatible
            || near_deadline
            || self.clock.elapsed(first.admitted_at, now) >= self.limits.max_batch_wait;
        if ready {
            batch
        } else {
            Vec::new()
        }
    }

    fn dispatch(&mut self, batch: Vec<A::RequestId>) {
        // Revalidate immediately before allocating/launching the physical work.
        for id in &batch {
            if let Some(event) = self.invalid(&self.entries[id].request, self.clock.now()) {
                self.finish(id, event);
            }
        }
        let batch: Vec<_> = batch
            .into_iter()
            .filter(|id| self.entries.contains_key(id))
            .collect();
        if batch.is_empty() {
            return;
        }
        let requests: Vec<_> = batch
            .iter()
            .map(|id| Arc::clone(&self.entries[id].request))
            .collect();
        let overhead = self
            .provider
            .as_ref()
            .expect("provider alive")
            .additional_resources(&requests);
        let overhead = match self.reserve(overhead, false) {
            Ok(reservation) => reservation,
            Err(fault) => {
                for id in &batch {
                    self.finish(
                        id,
                        TerminalEvent::Failure(self.adapter.runtime_error(fault)),
                    );
                }
                return;
            }
        };
        let execution_id = match self.adapter.allocate_execution_id() {
            Ok(id) => id,
            Err(error) => {
                drop(overhead);
                self.fail_batch(&batch, &error);
                return;
            }
        };
        // Resource calculation and execution-ID allocation can cross a deadline.
        // Abort this launch if any member became invalid; valid siblings remain
        // queued and are considered again by the bounded dispatch loop.
        let invalid: Vec<_> = batch
            .iter()
            .filter_map(|id| {
                self.invalid(&self.entries[id].request, self.clock.now())
                    .map(|event| (id.clone(), event))
            })
            .collect();
        if !invalid.is_empty() {
            drop(overhead);
            for (id, event) in invalid {
                self.finish(&id, event);
            }
            return;
        }
        let dispatched = self
            .provider
            .as_mut()
            .expect("provider alive")
            .dispatch(&execution_id, &requests);
        let lease = match dispatched {
            Ok(lease) => lease,
            Err(error) => {
                drop(overhead);
                self.fail_batch(&batch, &error);
                return;
            }
        };
        let mut inputs = Vec::with_capacity(batch.len());
        for id in batch {
            let entry = self.entries.get_mut(&id).expect("admitted input");
            entry.running = true;
            inputs.push(Held {
                id,
                request: Arc::clone(&entry.request),
                _reservation: Arc::clone(&entry.reservation),
            });
        }
        self.queue.drain(..inputs.len());
        self.metrics.record(Event::Dispatched {
            items: inputs.len(),
        });
        self.executions.push(Execution {
            _id: execution_id,
            inputs,
            _overhead: overhead,
            lease,
        });
    }

    fn fail_batch(&mut self, batch: &[A::RequestId], error: &A::Error) {
        for id in batch {
            let error = self
                .adapter
                .execution_error(&self.entries[id].request, error);
            self.finish(id, TerminalEvent::Failure(error));
        }
    }

    fn complete(&mut self, execution: Execution<A, P>, outputs: Vec<BackendResult<A>>) {
        // Verify the entire ID set before publishing any sibling success.
        let expected: HashSet<_> = execution.inputs.iter().map(|input| &input.id).collect();
        let mut seen = HashSet::new();
        let well_formed = outputs.len() == execution.inputs.len()
            && outputs.iter().all(|output| {
                expected.contains(&output.request_id) && seen.insert(&output.request_id)
            });
        if well_formed {
            for output in outputs {
                let Some(entry) = self.entries.get(&output.request_id) else {
                    // Already logically canceled/expired/stale. Physical completion
                    // grants no new subscriber/cache/search publication rights.
                    continue;
                };
                let request = Arc::clone(&entry.request);
                let event = match output.output {
                    Ok(value) => match self.adapter.validate_output(&request, &value) {
                        Ok(()) => TerminalEvent::Success(value),
                        Err(error) => TerminalEvent::Failure(error),
                    },
                    Err(error) => {
                        TerminalEvent::Failure(self.adapter.execution_error(&request, &error))
                    }
                };
                // Output validation may take time or observe a changed generation.
                let event = self.invalid(&request, self.clock.now()).unwrap_or(event);
                self.finish(&output.request_id, event);
            }
        } else {
            for input in &execution.inputs {
                if self.entries.contains_key(&input.id) {
                    let event = self
                        .invalid(&input.request, self.clock.now())
                        .unwrap_or_else(|| {
                            TerminalEvent::Failure(
                                self.adapter.runtime_error(RuntimeFault::InvalidCompletion),
                            )
                        });
                    self.finish(&input.id, event);
                }
            }
        }
        // Only Poll Ready reaches this point. C's lease pins are still alive
        // during output validation and are released exactly once afterward.
        drop(execution);
    }

    fn cancel_all(&mut self) {
        let ids: Vec<_> = self.entries.keys().cloned().collect();
        for id in ids {
            self.finish(&id, TerminalEvent::Canceled);
        }
    }

    /// Close admission and cancel subscribers. A repeated call cannot extend drain.
    pub fn begin_shutdown(&mut self, drain_deadline: A::Tick) {
        self.closed = true;
        self.drain_deadline.get_or_insert(drain_deadline);
        self.cancel_all();
    }

    pub fn drain_state(&self) -> DrainState {
        let Some(deadline) = self.drain_deadline else {
            return DrainState::Open;
        };
        if self.executions.is_empty() {
            DrainState::Drained
        } else if self.clock.now() >= deadline {
            DrainState::TimedOut {
                executions: self.executions.len(),
                reserved: self.state().reserved,
            }
        } else {
            DrainState::Draining
        }
    }
}

impl<A, P, C> Drop for Scheduler<A, P, C>
where
    A: Adapter,
    P: Backend<A>,
    C: Clock<Tick = A::Tick>,
{
    fn drop(&mut self) {
        self.closed = true;
        self.cancel_all();
        if !self.executions.is_empty() {
            self.metrics.record(Event::Quarantined {
                executions: self.executions.len(),
            });
            // No bounded safe abort exists in this provider contract. Retain
            // context as well as pins; never Drop a live device borrow on timeout.
            if let Some(provider) = self.provider.take() {
                std::mem::forget(provider);
            }
            for execution in self.executions.drain(..) {
                std::mem::forget(execution);
            }
        }
    }
}
