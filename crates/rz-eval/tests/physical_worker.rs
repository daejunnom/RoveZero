use rz_eval::error::{CauseCode, ExternalCause, FailureKind, FailureStage, CAUSE_PREFIX_BYTES};
use rz_eval::worker::{PhysicalPoll, SingleWorker};
use std::sync::{mpsc, Arc};
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
    let mut lease = worker.submit("owned output".into()).unwrap();
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
fn a_later_panic_cause_is_not_attributed_to_a_completed_lease() {
    let mut worker = SingleWorker::spawn(|input: &u8| {
        if *input == 1 {
            panic!("second job panic");
        }
        *input
    })
    .unwrap();
    let mut completed = worker.submit(0).unwrap();
    wait_until(|| match completed.poll() {
        PhysicalPoll::Ready(0) => true,
        PhysicalPoll::Pending => false,
        _ => panic!("unexpected first job state"),
    });
    assert!(completed.quarantine_cause().unwrap().is_none());
    let mut failed = worker.submit(1).unwrap();
    wait_until(|| matches!(failed.poll(), PhysicalPoll::Quarantined));
    assert!(failed.quarantine_cause().unwrap().is_some());
    assert!(completed.quarantine_cause().unwrap().is_none());
    assert!(matches!(completed.poll(), PhysicalPoll::Consumed));
}
