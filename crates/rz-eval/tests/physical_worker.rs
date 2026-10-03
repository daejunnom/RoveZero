use rz_eval::error::{
    BackendError, CauseCode, ExternalCause, FailureKind, FailureStage, CAUSE_PREFIX_BYTES,
};
use rz_eval::worker::{PhysicalPoll, PhysicalRun, SingleWorker};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::task::Poll;
use std::time::{Duration, Instant};

fn wait_until(mut check: impl FnMut() -> bool) {
    let stop = Instant::now() + Duration::from_secs(2);
    while !check() {
        assert!(
            Instant::now() < stop,
            "physical worker did not finish within test budget"
        );
        std::thread::yield_now();
    }
}

struct DelayedDrop {
    entered: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
}

impl Drop for DelayedDrop {
    fn drop(&mut self) {
        self.entered.send(()).unwrap();
        self.release.recv_timeout(Duration::from_secs(2)).unwrap();
    }
}

#[test]
fn process_shutdown_waits_for_captured_backend_drop_and_rejects_new_jobs() {
    let (entered, entering) = mpsc::sync_channel(1);
    let (release, released) = mpsc::sync_channel(1);
    let backend = DelayedDrop {
        entered,
        release: released,
    };
    let mut worker = SingleWorker::spawn(move |input: &u8| {
        let _pin = &backend;
        *input
    })
    .unwrap();
    let mut lease = worker.submit(7).unwrap();
    wait_until(|| matches!(lease.poll(), PhysicalPoll::Ready(7)));
    assert!(matches!(worker.try_shutdown(), Poll::Pending));
    entering.recv_timeout(Duration::from_secs(2)).unwrap();
    for _ in 0..3 {
        assert!(matches!(worker.try_shutdown(), Poll::Pending));
        assert!(worker.submit(8).is_err());
    }
    release.send(()).unwrap();
    wait_until(|| match worker.try_shutdown() {
        Poll::Pending => false,
        Poll::Ready(result) => {
            result.unwrap();
            true
        }
    });
    assert!(matches!(worker.try_shutdown(), Poll::Ready(Ok(()))));
    assert!(worker.submit(9).is_err());
}

#[test]
fn process_shutdown_waits_for_thread_local_destruction_after_closure_returns() {
    std::thread_local! {
        static BACKEND_TLS: std::cell::RefCell<Option<DelayedDrop>> = const {
            std::cell::RefCell::new(None)
        };
    }
    let (entered, entering) = mpsc::sync_channel(1);
    let (release, released) = mpsc::sync_channel(1);
    let mut backend = Some(DelayedDrop {
        entered,
        release: released,
    });
    let mut worker = SingleWorker::spawn(move |input: &u8| {
        BACKEND_TLS.with(|slot| *slot.borrow_mut() = backend.take());
        *input
    })
    .unwrap();
    let mut lease = worker.submit(1).unwrap();
    wait_until(|| matches!(lease.poll(), PhysicalPoll::Ready(1)));
    assert!(matches!(worker.try_shutdown(), Poll::Pending));
    entering.recv_timeout(Duration::from_secs(2)).unwrap();
    // The thread closure has returned, but its TLS still owns native state.
    assert!(matches!(worker.try_shutdown(), Poll::Pending));
    release.send(()).unwrap();
    wait_until(|| match worker.try_shutdown() {
        Poll::Pending => false,
        Poll::Ready(result) => {
            result.unwrap();
            true
        }
    });
}

#[test]
fn captured_backend_drop_panic_is_a_sticky_typed_shutdown_failure() {
    struct PanickingDrop;
    impl Drop for PanickingDrop {
        fn drop(&mut self) {
            panic!("authored backend teardown panic");
        }
    }
    let backend = PanickingDrop;
    let mut worker = SingleWorker::spawn(move |input: &u8| {
        let _pin = &backend;
        *input
    })
    .unwrap();
    let mut original = None;
    wait_until(|| match worker.try_shutdown() {
        Poll::Pending => false,
        Poll::Ready(Err(error)) => {
            original = Some(error);
            true
        }
        Poll::Ready(Ok(())) => panic!("backend Drop panic cannot confirm shutdown"),
    });
    let original = original.unwrap();
    assert_eq!(original.kind, FailureKind::BackendFailure);
    assert_eq!(original.cause.unwrap().code, CauseCode::RuntimePanic);
    assert_eq!(
        original.native.as_ref().unwrap().code,
        "WorkerShutdownPanic"
    );
    assert!(matches!(worker.try_shutdown(), Poll::Ready(Err(error)) if error == original));
    assert!(worker.submit(2).is_err());
}

#[test]
fn busy_and_dropped_consumers_do_not_release_active_inputs() {
    let (entered, entering) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let mut worker = SingleWorker::spawn(move |input: &Arc<usize>| {
        entered.send(()).unwrap();
        released.recv().unwrap();
        **input
    })
    .unwrap();
    let pin = Arc::new(7);
    let weak = Arc::downgrade(&pin);
    let mut lease = worker.submit(pin).unwrap();
    entering.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(matches!(lease.poll(), PhysicalPoll::Pending));
    assert!(worker.submit(Arc::new(9)).is_err());
    drop(lease);
    drop(worker);
    assert!(
        weak.upgrade().is_some(),
        "worker still owns the physical input"
    );
    release.send(()).unwrap();
    wait_until(|| weak.upgrade().is_none());
}

#[test]
fn physical_completion_yields_owned_data_once() {
    let mut worker = SingleWorker::spawn(|input: &String| input.clone()).unwrap();
    #[cfg(feature = "experimental-notify")]
    let signal = worker.completion_signal();
    #[cfg(feature = "experimental-notify")]
    let version = signal.version();
    let mut lease = worker.submit("owned output".into()).unwrap();
    #[cfg(feature = "experimental-notify")]
    assert!(signal.wait_changed(version, std::time::Duration::from_secs(3)));
    let mut output = None;
    wait_until(|| match lease.poll() {
        PhysicalPoll::Ready(value) => {
            output = Some(value);
            true
        }
        PhysicalPoll::Pending => false,
        _ => panic!("unexpected physical state"),
    });
    assert!(matches!(lease.poll(), PhysicalPoll::Consumed));
    drop(lease);
    assert_eq!(output.unwrap(), "owned output");
}

#[test]
fn unwind_quarantines_backend_and_pins_instead_of_claiming_completion() {
    let message = "injected provider unwind α ".repeat(100);
    let injected = message.clone();
    let mut worker = SingleWorker::spawn(move |_: &Arc<usize>| -> () {
        std::panic::panic_any(injected.clone())
    })
    .unwrap();
    let input = Arc::new(5);
    let weak = Arc::downgrade(&input);
    let mut lease = worker.submit(input).unwrap();
    wait_until(|| matches!(lease.poll(), PhysicalPoll::Quarantined));
    let cause = lease.quarantine_cause().unwrap().unwrap();
    assert_eq!(cause.kind, FailureKind::BackendFailure);
    assert_eq!(cause.stage, FailureStage::Backend);
    let receipt = cause.cause.unwrap();
    assert_eq!(receipt.code, CauseCode::RuntimePanic);
    assert_eq!(usize::from(receipt.hashed_bytes), CAUSE_PREFIX_BYTES);
    assert!(receipt.truncated);
    assert_eq!(
        receipt.prefix_sha256,
        ExternalCause::capture(CauseCode::RuntimePanic, &message).prefix_sha256
    );
    let local = cause.native.as_ref().unwrap();
    assert_eq!(local.code, "WorkerPanic");
    assert!(local.message.len() <= 1024);
    assert!(local.truncated);
    assert!(message.starts_with(&local.message));
    assert!(!format!("{cause:?}").contains("injected provider unwind"));
    assert!(!cause.to_string().contains("injected provider unwind"));
    assert_eq!(lease.quarantine_cause().unwrap(), Some(cause));
    assert!(matches!(lease.poll(), PhysicalPoll::Quarantined));
    assert!(worker.submit(Arc::new(6)).is_err());
    drop(lease);
    drop(worker);
    // One tiny allocation intentionally survives this injected unknown completion.
    assert!(weak.upgrade().is_some());
}

#[test]
fn explicit_unknown_completion_preserves_native_cause_and_retains_one_job_and_backend() {
    struct CountDrop(Arc<AtomicUsize>);

    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let input_drops = Arc::new(AtomicUsize::new(0));
    let backend_drops = Arc::new(AtomicUsize::new(0));
    let rejected_drops = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let backend_pin = CountDrop(Arc::clone(&backend_drops));
    let expected = BackendError::new(
        FailureKind::BackendFailure,
        FailureStage::Backend,
        "injected native run completion is unknown",
    )
    .with_external_cause(
        CauseCode::OrtRun,
        &"injected physical completion uncertainty",
    )
    .with_diagnostic("NativeRunUnknown", "injected native execution failure");
    let injected = expected.clone();
    let run_calls = Arc::clone(&calls);
    let mut worker =
        SingleWorker::spawn_with_outcome(move |_: &Arc<CountDrop>| -> PhysicalRun<()> {
            let _backend_pin = &backend_pin;
            run_calls.fetch_add(1, Ordering::SeqCst);
            PhysicalRun::Quarantined(injected.clone())
        })
        .unwrap();
    let input = Arc::new(CountDrop(Arc::clone(&input_drops)));
    let weak = Arc::downgrade(&input);
    let mut lease = worker.submit(input).unwrap();
    wait_until(|| match lease.poll() {
        PhysicalPoll::Pending => false,
        PhysicalPoll::Quarantined => true,
        _ => panic!("unknown native completion cannot yield an owned output"),
    });
    assert_eq!(lease.quarantine_cause().unwrap(), Some(expected.clone()));
    assert_eq!(expected.cause.unwrap().code, CauseCode::OrtRun);
    assert!(matches!(lease.poll(), PhysicalPoll::Quarantined));

    // The per-job receipt may precede the global admission transition. Both
    // Busy and Quarantined must reject this input without executing it.
    let rejected = Arc::new(CountDrop(Arc::clone(&rejected_drops)));
    wait_until(|| match worker.submit(Arc::clone(&rejected)) {
        Err(error) if error.kind == FailureKind::ResourceExhausted => false,
        Err(error) => {
            assert_eq!(error.kind, FailureKind::BackendUnavailable);
            assert_eq!(error.stage, FailureStage::Admission);
            true
        }
        Ok(_) => panic!("quarantined worker must not admit another native job"),
    });
    drop(rejected);
    assert_eq!(rejected_drops.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(matches!(lease.poll(), PhysicalPoll::Quarantined));
    assert_eq!(lease.quarantine_cause().unwrap(), Some(expected.clone()));
    wait_until(|| match worker.try_shutdown() {
        Poll::Pending => false,
        Poll::Ready(Err(error)) => {
            assert_eq!(error, expected);
            true
        }
        Poll::Ready(Ok(())) => panic!("joining a quarantined thread cannot release native pins"),
    });
    assert_eq!(lease.quarantine_cause().unwrap(), Some(expected));
    drop(lease);
    drop(worker);
    assert!(
        weak.upgrade().is_some(),
        "unknown native job remains pinned"
    );
    assert_eq!(input_drops.load(Ordering::SeqCst), 0);
    assert_eq!(backend_drops.load(Ordering::SeqCst), 0);
}

#[test]
fn a_later_panic_cause_is_not_attributed_to_a_completed_lease() {
    let mut worker = SingleWorker::spawn(|input: &u8| {
        if *input == 1 {
            panic!("second job panic");
        }
        *input
    })
    .unwrap();
    let mut completed = worker.submit(0).unwrap();
    // Admit the second job after native completion without consuming the first
    // lease's queued output. Worker admission and each lease's result differ.
    let mut failed = None;
    wait_until(|| match worker.submit(1) {
        Ok(lease) => {
            failed = Some(lease);
            true
        }
        Err(error) => {
            assert_eq!(error.kind, FailureKind::ResourceExhausted);
            false
        }
    });
    let mut failed = failed.unwrap();
    wait_until(|| matches!(failed.poll(), PhysicalPoll::Quarantined));
    assert!(failed.quarantine_cause().unwrap().is_some());
    assert!(completed.quarantine_cause().unwrap().is_none());
    assert!(matches!(completed.poll(), PhysicalPoll::Ready(0)));
    assert!(matches!(completed.poll(), PhysicalPoll::Consumed));
}
