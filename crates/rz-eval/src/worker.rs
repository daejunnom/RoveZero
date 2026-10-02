//! One bounded physical worker, suitable for a runtime Backend lease adapter.
//! No queue policy, deadlines, request IDs, cancellation or logical finalization.

use crate::error::{BackendError, FailureKind as K, FailureStage as S};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{mpsc, Arc};

const IDLE: u8 = 0;
const BUSY: u8 = 1;
const CLOSED: u8 = 2;
const QUARANTINED: u8 = 3;

struct Job<J, R> {
    input: Arc<J>,
    completion: mpsc::SyncSender<R>,
}

pub struct SingleWorker<J, R> {
    sender: mpsc::SyncSender<Job<J, R>>,
    state: Arc<AtomicU8>,
}

pub struct PhysicalLease<J, R> {
    input: Arc<J>,
    completion: mpsc::Receiver<R>,
    state: Arc<AtomicU8>,
    consumed: bool,
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
                        Err(_) => {
                            // Rust unwinding alone cannot attest GPU synchronization.
                            // Deliberately retain backend/inputs until process exit;
                            // no retry, no Ready, no reservation-release permission.
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
        if self
            .sender
            .try_send(Job {
                input: Arc::clone(&input),
                completion,
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
            state: Arc::clone(&self.state),
            consumed: false,
        })
    }
}

impl<J, R> PhysicalLease<J, R> {
    pub fn input(&self) -> &J {
        &self.input
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
            Err(mpsc::TryRecvError::Empty) if self.state.load(Ordering::Acquire) != QUARANTINED => {
                PhysicalPoll::Pending
            }
            // Unexpected disconnection is also not proof of physical completion.
            Err(_) => PhysicalPoll::Quarantined,
        }
    }
}
