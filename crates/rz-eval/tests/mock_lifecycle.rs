use rz_eval::mock::{
    Callback, EventKind, InjectedFailure, Limits, MockError, ScriptIdentity, ScriptedBackend, Step,
};
use rz_eval::RawOutput;
use std::time::Duration;

fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}

fn identity() -> ScriptIdentity {
    ScriptIdentity {
        name: "c01-lifecycle-v1".into(),
        seed: 73,
    }
}

fn step(after: u64, device_after: u64) -> Step {
    Step {
        callbacks: vec![Callback {
            after: ms(after),
            reply: Ok(RawOutput {
                policy_logits: vec![0.0, 1.0],
                wdl: vec![0.25, 0.5, 0.25],
            }),
        }],
        device_complete_after: ms(device_after),
        cancel_ack_after: Some(Duration::ZERO),
    }
}

#[test]
fn out_of_order_delivery_preserves_opaque_ticket_and_physical_time() {
    // IDs and generation semantics remain the caller's responsibility.
    let mut mock = ScriptedBackend::new(
        identity(),
        vec![step(20, 21), step(5, 6)],
        Limits::default(),
    )
    .unwrap();
    assert_eq!(mock.identity(), &identity());
    assert_eq!(mock.submit((101, 7)), Ok(0));
    assert_eq!(mock.submit((102, 8)), Ok(1));
    assert!(mock.next_event().is_none());
    mock.advance_to(ms(100)).unwrap();
    let actual: Vec<_> = std::iter::from_fn(|| mock.next_event())
        .map(|event| (event.ticket, event.script_step, event.at))
        .collect();
    assert_eq!(
        actual,
        vec![
            ((102, 8), 1, ms(5)),
            ((102, 8), 1, ms(6)),
            ((101, 7), 0, ms(20)),
            ((101, 7), 0, ms(21)),
        ]
    );
    assert_eq!(mock.in_flight(), 0);
    assert_eq!(mock.submit((103, 8)), Err(MockError::ScriptExhausted));
}

#[test]
fn cancel_does_not_release_physical_work_or_suppress_late_output() {
    let mut mock = ScriptedBackend::new(identity(), vec![step(10, 20)], Limits::default()).unwrap();
    mock.submit(11).unwrap();
    assert_eq!(mock.cancel(&11), Ok(true));
    assert_eq!(mock.cancel(&11), Ok(false));
    assert_eq!(
        mock.next_event().unwrap().kind,
        EventKind::CancelAcknowledged
    );
    assert_eq!(mock.in_flight(), 1);
    mock.advance_to(ms(10)).unwrap();
    assert!(matches!(
        mock.next_event().unwrap().kind,
        EventKind::Callback(Ok(_))
    ));
    assert_eq!(mock.in_flight(), 1);
    mock.advance_to(ms(20)).unwrap();
    assert_eq!(mock.next_event().unwrap().kind, EventKind::DeviceCompleted);
    assert_eq!(mock.in_flight(), 0);
    assert!(mock.next_event().is_none());
    assert_eq!(mock.cancel(&11), Err(MockError::UnknownTicket));
}

#[test]
fn callbacks_can_be_interleaved_with_cancel_at_the_same_tick() {
    let mut script = step(10, 11);
    script.callbacks.push(script.callbacks[0].clone());
    let mut mock = ScriptedBackend::new(identity(), vec![script], Limits::default()).unwrap();
    mock.submit(5).unwrap();
    mock.advance_to(ms(10)).unwrap();
    let first = mock.next_event().unwrap();
    mock.cancel(&5).unwrap();
    let duplicate = mock.next_event().unwrap();
    assert_eq!(first, duplicate);
    assert_eq!(
        mock.next_event().unwrap().kind,
        EventKind::CancelAcknowledged
    );
    assert!(mock.next_event().is_none());
    assert_eq!(mock.in_flight(), 1);
}

#[test]
fn missing_heads_non_finite_values_and_backend_failures_reach_the_consumer() {
    let mut script = step(0, 2);
    script.callbacks = vec![
        Callback {
            after: Duration::ZERO,
            reply: Ok(RawOutput {
                policy_logits: vec![f32::NAN, f32::INFINITY],
                wdl: Vec::new(),
            }),
        },
        Callback {
            after: ms(1),
            reply: Err(InjectedFailure {
                code: "device_lost".into(),
                stage: "inference".into(),
            }),
        },
    ];
    script.cancel_ack_after = None;
    let mut mock = ScriptedBackend::new(identity(), vec![script], Limits::default()).unwrap();
    mock.submit(3).unwrap();
    mock.cancel(&3).unwrap();
    let EventKind::Callback(Ok(raw)) = mock.next_event().unwrap().kind else {
        panic!("expected injected raw output");
    };
    assert!(raw.policy_logits[0].is_nan());
    assert!(raw.policy_logits[1].is_infinite());
    assert!(raw.wdl.is_empty());
    mock.advance_to(ms(1)).unwrap();
    assert_eq!(
        mock.next_event().unwrap().kind,
        EventKind::Callback(Err(InjectedFailure {
            code: "device_lost".into(),
            stage: "inference".into(),
        }))
    );
    assert_eq!(mock.in_flight(), 1);
}

#[test]
fn rejected_admission_does_not_consume_the_next_script_step() {
    let limits = Limits {
        max_in_flight: 1,
        ..Limits::default()
    };
    let mut mock = ScriptedBackend::new(identity(), vec![step(0, 0), step(1, 1)], limits).unwrap();
    mock.submit(1).unwrap();
    assert_eq!(mock.submit(1), Err(MockError::DuplicateTicket));
    assert_eq!(mock.submit(2), Err(MockError::InFlightLimit));
    while mock.next_event().is_some() {}
    assert_eq!(mock.submit(2), Ok(1));
}

#[test]
fn bounded_event_admission_and_failed_cancel_are_atomic() {
    let limits = Limits {
        max_pending_events: 2,
        ..Limits::default()
    };
    let mut mock = ScriptedBackend::new(identity(), vec![step(0, 1), step(2, 2)], limits).unwrap();
    mock.submit(1).unwrap();
    assert_eq!(mock.submit(2), Err(MockError::EventLimit));
    assert_eq!(mock.cancel(&1), Err(MockError::EventLimit));
    mock.next_event().unwrap();
    // Failed cancellation did not set cancel_requested or lose device completion.
    assert_eq!(mock.cancel(&1), Ok(true));
    assert_eq!(
        mock.next_event().unwrap().kind,
        EventKind::CancelAcknowledged
    );
    mock.advance_to(ms(1)).unwrap();
    assert_eq!(mock.next_event().unwrap().kind, EventKind::DeviceCompleted);
    assert_eq!(mock.submit(2), Ok(1));
}

#[test]
fn clock_and_overflow_errors_preserve_simulator_state() {
    let mut mock = ScriptedBackend::new(identity(), vec![step(1, 2)], Limits::default()).unwrap();
    mock.advance_to(Duration::MAX).unwrap();
    assert_eq!(mock.advance_to(ms(1)), Err(MockError::ClockMovedBackwards));
    assert_eq!(mock.now(), Duration::MAX);
    assert_eq!(mock.submit(9), Err(MockError::TimeOverflow));
    assert_eq!(mock.in_flight(), 0);
    assert!(mock.next_event().is_none());
    assert_eq!(mock.submit(10), Err(MockError::TimeOverflow));
}

#[test]
fn oversized_scripts_and_payloads_are_rejected_before_execution() {
    let limits = Limits {
        max_script_values: 4,
        ..Limits::default()
    };
    assert_eq!(
        ScriptedBackend::<u64>::new(identity(), vec![step(0, 1)], limits).unwrap_err(),
        MockError::ScriptLimit
    );
    let limits = Limits {
        max_steps: 1,
        ..Limits::default()
    };
    assert_eq!(
        ScriptedBackend::<u64>::new(identity(), vec![step(0, 1), step(0, 1)], limits).unwrap_err(),
        MockError::ScriptLimit
    );
}
