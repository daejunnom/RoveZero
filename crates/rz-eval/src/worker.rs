//! One bounded physical worker, suitable for a runtime Backend lease adapter.
//! No queue policy, deadlines, request IDs, cancellation or logical finalization.

use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{mpsc, Arc, Mutex};

const IDLE: u8 = 0;
const BUSY: u8 = 1;
const CLOSED: u8 = 2;
const QUARANTINED: u8 = 3;

struct Job<J, R> {
    input: Arc<J>,
    completion: mpsc::SyncSender<R>,
    quarantine_cause: Arc<Mutex<Option<BackendError>>>,
}

pub struct SingleWorker<J, R> {
    sender: mpsc::SyncSender<Job<J, R>>,
    state: Arc<AtomicU8>,
}

pub struct PhysicalLease<J, R> {
    input: Arc<J>,
    completion: mpsc::Receiver<R>,
    consumed: bool,
    quarantine_cause: Arc<Mutex<Option<BackendError>>>,
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
        let (sender, receiver) = mpsc::sync_channel::<Job<J, R>>(1);
        let state = Arc::new(AtomicU8::new(IDLE));
        let worker_state = Arc::clone(&state);
        std::thread::Builder::new()
            .name("rz-maia-physical".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&job.input)));
                    match result {
                        Ok(output) => {
                            // Even a dropped logical consumer does not cancel native
                            // execution. At this point run has physically completed.
                            worker_state.store(IDLE, Ordering::Release);
                            let _ = job.completion.send(output);
                        }
                        Err(payload) => {
                            // Rust unwinding alone cannot attest GPU synchronization.
                            // Deliberately retain backend/inputs until process exit;
                            // no retry, no Ready, no reservation-release permission.
                            let cause = panic_failure(payload.as_ref());
                            // Per-job receipt: a later job's panic must not be
                            // attributed to an earlier completed physical lease.
                            match job.quarantine_cause.lock() {
                                Ok(mut slot) => *slot = Some(cause),
                                Err(poisoned) => *poisoned.into_inner() = Some(cause),
                            }
                            std::mem::forget(job);
                            std::mem::forget(run);
                            worker_state.store(QUARANTINED, Ordering::Release);
                            return;
                        }
                    }
                }
                worker_state.store(CLOSED, Ordering::Release);
            })
            .map_err(|_| {
                BackendError::new(
                    K::BackendUnavailable,
                    S::Backend,
                    "cannot create physical worker",
                )
            })?;
        Ok(Self { sender, state })
    }

    /// A rejection guarantees this input was not handed to native execution.
    pub fn submit(&mut self, input: J) -> Result<PhysicalLease<J, R>, BackendError> {
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
        if self
            .sender
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
}

impl<J, R> PhysicalLease<J, R> {
    pub fn input(&self) -> &J {
        &self.input
    }

    /// Bounded per-job evidence, not completion or permission to release pins.
    /// A panic receipt is published before the QUARANTINED Release transition.
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
    let error = BackendError::new(
        K::BackendFailure,
        S::Backend,
        "physical worker unwound; completion is unknown",
    );
    if let Some(message) = payload.downcast_ref::<&str>() {
        error
            .with_external_cause(CauseCode::RuntimePanic, message)
            .with_diagnostic("WorkerPanic", message)
    } else if let Some(message) = payload.downcast_ref::<String>() {
        error
            .with_external_cause(CauseCode::RuntimePanic, message)
            .with_diagnostic("WorkerPanic", message)
    } else {
        // Any need not implement Display/Debug; do not invent its original text.
        let unknown = "non-string physical worker panic payload";
        error
            .with_external_cause(CauseCode::RuntimePanic, &unknown)
            .with_diagnostic("WorkerPanicNonString", unknown)
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
