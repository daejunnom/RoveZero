//! One bounded physical worker, suitable for a runtime Backend lease adapter.
//! No queue policy, deadlines, request IDs, cancellation or logical finalization.

use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::task::Poll;
use std::thread::JoinHandle;

const IDLE: u8 = 0;
const BUSY: u8 = 1;
const CLOSED: u8 = 2;
const QUARANTINED: u8 = 3;

type PhysicalHandle = Arc<Mutex<Option<JoinHandle<Result<(), BackendError>>>>>;
type ShutdownSender = mpsc::SyncSender<Result<(), BackendError>>;

struct Job<J, R> {
    input: Arc<J>,
    completion: mpsc::SyncSender<R>,
    quarantine_cause: Arc<Mutex<Option<BackendError>>>,
}

pub struct SingleWorker<J, R> {
    sender: Option<mpsc::SyncSender<Job<J, R>>>,
    state: Arc<AtomicU8>,
    // The slot, rather than the spawn closure, owns the handle until a reaper
    // actually starts. A failed reaper spawn must not detach the native thread.
    physical_handle: PhysicalHandle,
    shutdown_completion: Option<mpsc::Receiver<Result<(), BackendError>>>,
    shutdown_result: Option<Result<(), BackendError>>,
}

pub struct PhysicalLease<J, R> {
    input: Arc<J>,
    completion: mpsc::Receiver<R>,
    consumed: bool,
    quarantine_cause: Arc<Mutex<Option<BackendError>>>,
}

pub enum PhysicalRun<R> {
    /// Native execution has physically completed and owns its output.
    Complete(R),
    /// Physical completion is unknown. Retain this job and backend until process
    /// exit and publish the original typed failure without permitting reuse.
    Quarantined(BackendError),
}

pub enum PhysicalPoll<R> {
    Pending,
    Ready(R),
    /// Native completion is unknown. The job and backend remain pinned; runtime
    /// must quarantine this lease and must not release/reuse its reservations.
    Quarantined,
    Consumed,
}

impl<J: Send + Sync + 'static, R: Send + 'static> SingleWorker<J, R> {
    /// run must return only after physical completion, with owned outputs. There
    /// is one admitted job and no hidden waiting queue. Dropping this handle
    /// closes admission; an active worker continues owning its job until done.
    pub fn spawn<F>(mut run: F) -> Result<Self, BackendError>
    where
        F: FnMut(&J) -> R + Send + 'static,
    {
        Self::spawn_with_outcome(move |input| PhysicalRun::Complete(run(input)))
    }

    /// Complete must attest physical completion, including owned output. An
    /// explicit Quarantined outcome or unwind permanently closes admission and
    /// retains this job and run closure, preserving the job's original cause.
    /// Resources still in native use must already be owned by J or captured
    /// closure state: callback-local resources are not retained after return.
    pub fn spawn_with_outcome<F>(mut run: F) -> Result<Self, BackendError>
    where
        F: FnMut(&J) -> PhysicalRun<R> + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel::<Job<J, R>>(1);
        let state = Arc::new(AtomicU8::new(IDLE));
        let worker_state = Arc::clone(&state);
        let physical_handle = std::thread::Builder::new()
            .name("rz-maia-physical".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let outcome =
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run(&job.input)
                        })) {
                            Ok(outcome) => outcome,
                            Err(payload) => {
                                PhysicalRun::Quarantined(panic_failure(payload.as_ref()))
                            }
                        };
                    match outcome {
                        PhysicalRun::Complete(output) => {
                            // Even a dropped logical consumer does not cancel native
                            // execution. At this point run has physically completed.
                            worker_state.store(IDLE, Ordering::Release);
                            let _ = job.completion.send(output);
                        }
                        PhysicalRun::Quarantined(cause) => {
                            // Neither unwinding nor a native failure alone attests
                            // physical completion or GPU synchronization.
                            // Deliberately retain backend/inputs until process exit;
                            // no retry, no Ready, no reservation-release permission.
                            // Per-job receipt: a later job's failure must not be
                            // attributed to an earlier completed physical lease.
                            match job.quarantine_cause.lock() {
                                Ok(mut slot) => *slot = Some(cause.clone()),
                                Err(poisoned) => *poisoned.into_inner() = Some(cause.clone()),
                            }
                            std::mem::forget(job);
                            std::mem::forget(run);
                            worker_state.store(QUARANTINED, Ordering::Release);
                            // Thread termination does not prove the quarantined
                            // execution completed, nor release its leaked pins.
                            return Err(cause);
                        }
                    }
                }
                // Session/closure destruction is part of process shutdown,
                // not of an individual physical lease's Ready transition.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(run)))
                    .map_err(|payload| {
                        shutdown_panic_failure(payload.as_ref(), "WorkerShutdownPanic")
                    });
                worker_state.store(CLOSED, Ordering::Release);
                result
            })
            .map_err(|_| {
                BackendError::new(
                    K::BackendUnavailable,
                    S::Backend,
                    "cannot create physical worker",
                )
            })?;
        Ok(Self {
            sender: Some(sender),
            state,
            physical_handle: Arc::new(Mutex::new(Some(physical_handle))),
            shutdown_completion: None,
            shutdown_result: None,
        })
    }

    /// A rejection guarantees this input was not handed to native execution.
    pub fn submit(&mut self, input: J) -> Result<PhysicalLease<J, R>, BackendError> {
        let sender = self.sender.as_ref().ok_or_else(|| {
            BackendError::new(
                K::BackendUnavailable,
                S::Admission,
                "physical worker process shutdown has closed admission",
            )
        })?;
        self.state
            .compare_exchange(IDLE, BUSY, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|state| {
                BackendError::new(
                    if state == BUSY {
                        K::ResourceExhausted
                    } else {
                        K::BackendUnavailable
                    },
                    S::Admission,
                    "physical worker is busy, closed or quarantined",
                )
            })?;
        let (completion, receiver) = mpsc::sync_channel(1);
        let input = Arc::new(input);
        let quarantine_cause = Arc::new(Mutex::new(None));
        if sender
            .try_send(Job {
                input: Arc::clone(&input),
                completion,
                quarantine_cause: Arc::clone(&quarantine_cause),
            })
            .is_err()
        {
            self.state.store(CLOSED, Ordering::Release);
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Admission,
                "physical worker disconnected before launch",
            ));
        }
        Ok(PhysicalLease {
            input,
            completion: receiver,
            consumed: false,
            quarantine_cause,
        })
    }

    /// Terminal, process-only shutdown. Close future admission once, then poll a
    /// single reaper's result without blocking this caller on native destruction
    /// or thread TLS. A successful join is reported only after the native thread
    /// has fully exited; quarantined execution never becomes successful here.
    /// Dropping this owner or reaching a caller deadline does not terminate the
    /// physical thread, invalidate retained inputs, or make a quarantine Ready.
    pub fn try_shutdown(&mut self) -> Poll<Result<(), BackendError>> {
        self.try_shutdown_with_start(|slot, completed| {
            std::thread::Builder::new()
                .name("rz-maia-reaper".into())
                .spawn(move || reap_physical(slot, completed))
        })
    }

    fn try_shutdown_with_start<F>(&mut self, start: F) -> Poll<Result<(), BackendError>>
    where
        F: FnOnce(PhysicalHandle, ShutdownSender) -> std::io::Result<JoinHandle<()>>,
    {
        if let Some(result) = &self.shutdown_result {
            return Poll::Ready(result.clone());
        }
        if self.shutdown_completion.is_none() {
            self.sender.take();
            let slot = Arc::clone(&self.physical_handle);
            let (completed, completion) = mpsc::sync_channel(1);
            let started = start(slot, completed);
            match started {
                Ok(reaper) => {
                    // This small reaper has no native state after it publishes
                    // the joined result. It may detach; the physical handle may
                    // not, and is owned by the retained slot/reaper until joined.
                    drop(reaper);
                    self.shutdown_completion = Some(completion);
                }
                Err(error) => {
                    let error = BackendError::new(
                        K::BackendUnavailable,
                        S::Backend,
                        "cannot create physical worker shutdown reaper",
                    )
                    .with_external_cause(CauseCode::RuntimeInitialize, &error);
                    // The failed spawn closure held only an Arc clone. The
                    // original JoinHandle remains in physical_handle, with no
                    // retry, forced drop or false completion acknowledgement.
                    self.shutdown_result = Some(Err(error.clone()));
                    return Poll::Ready(Err(error));
                }
            }
        }
        let result = match self.shutdown_completion.as_ref().unwrap().try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return Poll::Pending,
            Err(mpsc::TryRecvError::Disconnected) => Err(BackendError::new(
                K::BackendFailure,
                S::Backend,
                "physical worker shutdown reaper disconnected without a result",
            )),
        };
        self.shutdown_result = Some(result.clone());
        Poll::Ready(result)
    }
}

fn reap_physical(slot: PhysicalHandle, completed: ShutdownSender) {
    // Release the slot guard before joining. The caller never waits for this
    // mutex or invokes JoinHandle::join itself, including after is_finished.
    let handle = match slot.lock() {
        Ok(mut slot) => slot.take(),
        Err(error) => {
            let _ = completed.send(Err(BackendError::new(
                K::BackendFailure,
                S::Backend,
                "physical worker join ownership lock is poisoned",
            )
            .with_external_cause(CauseCode::RuntimePanic, &error)));
            return;
        }
    };
    let result = match handle {
        Some(handle) => match handle.join() {
            Ok(result) => result,
            Err(payload) => Err(shutdown_panic_failure(payload.as_ref(), "WorkerJoinPanic")),
        },
        None => Err(BackendError::new(
            K::BackendFailure,
            S::Backend,
            "physical worker join ownership is missing",
        )),
    };
    let _ = completed.send(result);
}

impl<J, R> PhysicalLease<J, R> {
    pub fn input(&self) -> &J {
        &self.input
    }

    /// Bounded per-job evidence, not completion or permission to release pins.
    /// The failure receipt is published before the QUARANTINED Release transition.
    /// Reading raw NativeDiagnostic text remains an explicit local operation.
    pub fn quarantine_cause(&self) -> Result<Option<BackendError>, BackendError> {
        self.quarantine_cause
            .lock()
            .map(|slot| slot.clone())
            .map_err(|error| {
                BackendError::new(
                    K::BackendFailure,
                    S::Backend,
                    "physical worker quarantine receipt lock is poisoned",
                )
                .with_external_cause(CauseCode::RuntimePanic, &error)
            })
    }

    /// Nonblocking. Each owned output can be obtained exactly once, independently
    /// of any logical request cancellation performed by the runtime.
    pub fn poll(&mut self) -> PhysicalPoll<R> {
        if self.consumed {
            return PhysicalPoll::Consumed;
        }
        match self.completion.try_recv() {
            Ok(result) => {
                self.consumed = true;
                PhysicalPoll::Ready(result)
            }
            Err(mpsc::TryRecvError::Empty) => self.poll_empty(),
            // Unexpected disconnection is also not proof of physical completion.
            Err(mpsc::TryRecvError::Disconnected) => PhysicalPoll::Quarantined,
        }
    }

    fn poll_empty(&self) -> PhysicalPoll<R> {
        // The worker's admission state can belong to a later job by now. Only
        // this job's own receipt may quarantine an observed Empty completion.
        // Its output may also have arrived since try_recv: Pending is safe and
        // the next poll will consume it; an unrelated job's panic is irrelevant.
        match self.quarantine_cause.lock() {
            Ok(slot) if slot.is_none() => PhysicalPoll::Pending,
            _ => PhysicalPoll::Quarantined,
        }
    }
}

fn panic_failure(payload: &(dyn std::any::Any + Send)) -> BackendError {
    panic_with_context(
        payload,
        "physical worker unwound; completion is unknown",
        "WorkerPanic",
    )
}

fn shutdown_panic_failure(
    payload: &(dyn std::any::Any + Send),
    code: &'static str,
) -> BackendError {
    panic_with_context(payload, "physical worker process shutdown unwound", code)
}

fn panic_with_context(
    payload: &(dyn std::any::Any + Send),
    detail: &'static str,
    code: &'static str,
) -> BackendError {
    let error = BackendError::new(K::BackendFailure, S::Backend, detail);
    if let Some(message) = payload.downcast_ref::<&str>() {
        error
            .with_external_cause(CauseCode::RuntimePanic, message)
            .with_diagnostic(code, message)
    } else if let Some(message) = payload.downcast_ref::<String>() {
        error
            .with_external_cause(CauseCode::RuntimePanic, message)
            .with_diagnostic(code, message)
    } else {
        // Any need not implement Display/Debug; do not invent its original text.
        let unknown = "non-string physical worker panic payload";
        error
            .with_external_cause(CauseCode::RuntimePanic, &unknown)
            .with_diagnostic(
                if code == "WorkerPanic" {
                    "WorkerPanicNonString"
                } else {
                    code
                },
                unknown,
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait_until(mut check: impl FnMut() -> bool) {
        let stop = Instant::now() + Duration::from_secs(2);
        while !check() {
            assert!(
                Instant::now() < stop,
                "worker race fixture exceeded its budget"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn reaper_spawn_failure_retains_physical_handle_and_never_retries() {
        let mut worker = SingleWorker::spawn(|input: &u8| *input).unwrap();
        let original = match worker.try_shutdown_with_start(|_, _| {
            Err(std::io::Error::other("authored reaper spawn failure"))
        }) {
            Poll::Ready(Err(error)) => error,
            _ => panic!("failed spawn must publish its original typed failure"),
        };
        assert_eq!(original.kind, K::BackendUnavailable);
        assert_eq!(original.cause.unwrap().code, CauseCode::RuntimeInitialize);
        assert!(worker.physical_handle.lock().unwrap().is_some());
        assert!(worker.submit(1).is_err());
        assert!(matches!(worker.try_shutdown_with_start(|_, _| {
            panic!("a sticky spawn failure must never launch another reaper")
        }), Poll::Ready(Err(error)) if error == original));
        assert!(worker.physical_handle.lock().unwrap().is_some());
        // Test cleanup runs off the production caller path. The owned handle
        // stayed in its slot through failure and no production join blocked.
        let handle = worker.physical_handle.lock().unwrap().take().unwrap();
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn an_empty_observation_stays_pending_when_a_later_job_quarantines() {
        let (entered, entering) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        let mut worker = SingleWorker::spawn(move |input: &u8| {
            if *input == 0 {
                entered.send(()).unwrap();
                released.recv_timeout(Duration::from_secs(2)).unwrap();
                return 0;
            }
            panic!("later job's injected unwind");
        })
        .unwrap();
        let mut first = worker.submit(0).unwrap();
        entering.recv_timeout(Duration::from_secs(2)).unwrap();
        // Freeze the exact interleaving between production poll's Empty receipt
        // and its Empty classification, without timing sleeps or a public hook.
        assert!(matches!(
            first.completion.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        release.send(()).unwrap();
        let mut next = None;
        wait_until(|| match worker.submit(1) {
            Ok(lease) => {
                next = Some(lease);
                true
            }
            Err(error) => {
                assert_eq!(error.kind, K::ResourceExhausted);
                false
            }
        });
        let mut next = next.unwrap();
        wait_until(|| matches!(next.poll(), PhysicalPoll::Quarantined));
        wait_until(|| worker.state.load(Ordering::Acquire) == QUARANTINED);
        assert!(next.quarantine_cause().unwrap().is_some());
        assert!(first.quarantine_cause().unwrap().is_none());
        assert!(matches!(first.poll_empty(), PhysicalPoll::Pending));
        assert!(matches!(first.poll(), PhysicalPoll::Ready(0)));
        assert!(matches!(first.poll(), PhysicalPoll::Consumed));
    }
}
