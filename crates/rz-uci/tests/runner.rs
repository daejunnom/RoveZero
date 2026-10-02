use rz_uci::{
    Effect, EngineIdentity, Event, ParserLimits, PositionPort, PositionSpec, PreparedPosition,
    ServeFailure, Session, forward_lines, handle_event, serve_events, serve_events_with_handler,
};
use std::io::{self, Cursor, Write};
use std::sync::mpsc::sync_channel;

struct Fixture;
impl PositionPort for Fixture {
    type Snapshot = ();
    type Error = &'static str;
    fn prepare(&self, _: &PositionSpec) -> Result<PreparedPosition<()>, Self::Error> {
        Ok(PreparedPosition {
            snapshot: (),
            legal_moves: vec!["e2e4".into(), "d2d4".into()],
            exact_terminal: false,
        })
    }
}

fn session() -> Session<Fixture> {
    Session::new(
        Fixture,
        EngineIdentity {
            name: "Fixture".into(),
            author: "B".into(),
        },
        vec![],
        ParserLimits::default(),
    )
    .unwrap()
}

#[test]
fn active_event_loop_handles_ready_stop_quit_without_worker_completion() {
    let (tx, rx) = sync_channel(16);
    for line in ["uci", "go infinite", "isready", "stop", "stop", "quit"] {
        tx.send(Event::Line(line.into())).unwrap();
    }
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut effects = Vec::new();
    let mut session = session();
    serve_events(&mut session, &rx, &mut stdout, &mut stderr, |effect| {
        effects.push(effect);
        Ok::<_, &'static str>(())
    })
    .unwrap();
    assert_eq!(
        String::from_utf8(stdout).unwrap(),
        "id name Fixture\nid author B\nuciok\nreadyok\nbestmove e2e4\n"
    );
    assert!(String::from_utf8(stderr).unwrap().contains("LegalFallback"));
    assert_eq!(effects.len(), 3);
    assert!(matches!(effects[0], Effect::Start { .. }));
    assert!(matches!(effects[1], Effect::Cancel { .. }));
    assert!(matches!(effects[2], Effect::Shutdown));
    assert!(session.is_closed());
}

#[test]
fn channel_disconnect_is_eof_and_suppresses_old_bestmove() {
    let (tx, rx) = sync_channel(1);
    tx.send(Event::Line("go nodes 10".into())).unwrap();
    drop(tx);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut effects = Vec::new();
    serve_events(&mut session(), &rx, &mut stdout, &mut stderr, |effect| {
        effects.push(effect);
        Ok::<_, &'static str>(())
    })
    .unwrap();
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
    assert_eq!(effects.len(), 3);
    assert!(matches!(effects[2], Effect::Shutdown));
}

#[test]
fn bounded_physical_reader_discards_overlong_line_and_recovers_next_command() {
    let (tx, rx) = sync_channel(8);
    let mut input = Cursor::new(b"123456789\nisready\n\xff\nstop".to_vec());
    forward_lines(&mut input, &tx, 8).unwrap();
    assert!(matches!(
        rx.recv().unwrap(),
        Event::RejectedInput {
            code: "LineTooLong",
            ..
        }
    ));
    assert!(matches!(rx.recv().unwrap(), Event::Line(line) if line == "isready\n"));
    assert!(matches!(
        rx.recv().unwrap(),
        Event::RejectedInput {
            code: "InvalidText",
            ..
        }
    ));
    assert!(matches!(rx.recv().unwrap(), Event::Line(line) if line == "stop"));
    assert!(matches!(rx.recv().unwrap(), Event::EndOfInput));
}

struct FailingInput(Cursor<Vec<u8>>);

impl io::Read for FailingInput {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        io::Read::read(&mut self.0, buffer)
    }
}

impl io::BufRead for FailingInput {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.0.position() == self.0.get_ref().len() as u64 {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "input transport disappeared",
            ))
        } else {
            io::BufRead::fill_buf(&mut self.0)
        }
    }

    fn consume(&mut self, amount: usize) {
        io::BufRead::consume(&mut self.0, amount);
    }
}

#[test]
fn input_io_failure_closes_session_even_while_worker_sender_remains_alive() {
    let (tx, rx) = sync_channel(4);
    let _worker_sender = tx.clone();
    let error = forward_lines(
        &mut FailingInput(Cursor::new(b"go infinite\n".to_vec())),
        &tx,
        1024,
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    let mut effects = Vec::new();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    serve_events(&mut session(), &rx, &mut stdout, &mut stderr, |effect| {
        effects.push(effect);
        Ok::<_, &'static str>(())
    })
    .unwrap();
    assert!(stdout.is_empty());
    assert_eq!(
        String::from_utf8(stderr).unwrap(),
        "InputIoError: input transport disappeared\n"
    );
    assert_eq!(effects.len(), 3);
    assert!(matches!(effects[0], Effect::Start { .. }));
    assert!(matches!(effects[1], Effect::Cancel { .. }));
    assert!(matches!(effects[2], Effect::Shutdown));
}

#[test]
fn injected_handler_revalidates_queued_completion_when_owner_dequeues_it() {
    let mut session = session();
    session.handle_line("go nodes 10");
    let ticket = session.active_ticket().unwrap();
    assert!(session.progress(&ticket, "d2d4").accepted);
    let (tx, rx) = sync_channel(4);
    // Producer admission succeeded, but the result expired while queued.
    tx.send(Event::Complete {
        ticket: ticket.clone(),
        completion: rz_uci::SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
    })
    .unwrap();
    tx.send(Event::Line("isready".into())).unwrap();
    tx.send(Event::Line("quit".into())).unwrap();
    let expired_at_dequeue = true;
    let mut checked_completions = 0;
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut effects = Vec::new();
    serve_events_with_handler(
        &mut session,
        &rx,
        &mut stdout,
        &mut stderr,
        |session, event| match event {
            Event::Complete { ticket, completion } => {
                checked_completions += 1;
                if expired_at_dequeue {
                    session.expire(&ticket)
                } else {
                    session.complete(&ticket, completion)
                }
            }
            event => handle_event(session, event),
        },
        |effect| {
            effects.push(effect);
            Ok::<_, &'static str>(())
        },
    )
    .unwrap();
    assert_eq!(checked_completions, 1);
    assert_eq!(
        String::from_utf8(stdout).unwrap(),
        "bestmove d2d4\nreadyok\n"
    );
    assert!(stderr.is_empty());
    assert!(
        matches!(effects[0], Effect::Cancel { ticket: ref canceled, reason: rz_uci::CancelReason::Deadline } if canceled == &ticket)
    );
    assert!(matches!(effects[1], Effect::Shutdown));
}

#[test]
fn injected_loop_preserves_io_failure_cancellation_and_shutdown() {
    let (tx, rx) = sync_channel(2);
    tx.send(Event::Line("go infinite".into())).unwrap();
    tx.send(Event::Line("isready".into())).unwrap();
    let mut effects = Vec::new();
    let mut handled = 0;
    let error = serve_events_with_handler(
        &mut session(),
        &rx,
        &mut BrokenWriter,
        &mut Vec::new(),
        |session, event| {
            handled += 1;
            handle_event(session, event)
        },
        |effect| {
            effects.push(effect);
            Ok::<_, &'static str>(())
        },
    )
    .unwrap_err();
    assert!(matches!(error.failure, ServeFailure::Io(_)));
    // The owner also receives EOF cleanup when the protocol write fails.
    assert_eq!(handled, 3);
    assert!(matches!(effects[0], Effect::Start { .. }));
    assert!(matches!(effects[1], Effect::Cancel { .. }));
    assert!(matches!(effects[2], Effect::Shutdown));
}

#[test]
fn injected_loop_preserves_original_dispatch_failure_and_cleanup_failure() {
    let (tx, rx) = sync_channel(1);
    tx.send(Event::Line("go infinite".into())).unwrap();
    let mut shutdown_called = false;
    let error = serve_events_with_handler(
        &mut session(),
        &rx,
        &mut Vec::new(),
        &mut Vec::new(),
        handle_event,
        |effect| match effect {
            Effect::Start { .. } => Err("start failed"),
            Effect::Cancel { .. } => Err("cancel failed"),
            Effect::Shutdown => {
                shutdown_called = true;
                Ok(())
            }
            _ => Ok(()),
        },
    )
    .unwrap_err();
    assert!(matches!(
        error.failure,
        ServeFailure::Handler("start failed")
    ));
    assert_eq!(error.cleanup_failures, vec!["cancel failed"]);
    assert!(shutdown_called);
}

#[test]
fn dispatch_failure_preserves_original_and_cleanup_errors() {
    let (tx, rx) = sync_channel(1);
    tx.send(Event::Line("go infinite".into())).unwrap();
    let mut shutdown_called = false;
    let error = serve_events(
        &mut session(),
        &rx,
        &mut Vec::new(),
        &mut Vec::new(),
        |effect| match effect {
            Effect::Start { .. } => Err("start failed"),
            Effect::Cancel { .. } => Err("cancel failed"),
            Effect::Shutdown => {
                shutdown_called = true;
                Ok(())
            }
            _ => Ok(()),
        },
    )
    .unwrap_err();
    assert!(matches!(
        error.failure,
        ServeFailure::Handler("start failed")
    ));
    assert_eq!(error.cleanup_failures, vec!["cancel failed"]);
    assert!(shutdown_called);
}

#[test]
fn quit_cancel_failure_still_attempts_shutdown() {
    let (tx, rx) = sync_channel(2);
    tx.send(Event::Line("go infinite".into())).unwrap();
    tx.send(Event::Line("quit".into())).unwrap();
    let mut shutdown_called = false;
    let error = serve_events(
        &mut session(),
        &rx,
        &mut Vec::new(),
        &mut Vec::new(),
        |effect| match effect {
            Effect::Cancel { .. } => Err("cancel failed"),
            Effect::Shutdown => {
                shutdown_called = true;
                Err("shutdown failed")
            }
            _ => Ok(()),
        },
    )
    .unwrap_err();
    assert!(matches!(
        error.failure,
        ServeFailure::Handler("cancel failed")
    ));
    assert_eq!(error.cleanup_failures, vec!["shutdown failed"]);
    assert!(shutdown_called);
}

struct BrokenWriter;
impl Write for BrokenWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("broken pipe"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn transport_failure_still_cancels_worker_and_dispatches_shutdown() {
    let (tx, rx) = sync_channel(2);
    tx.send(Event::Line("go infinite".into())).unwrap();
    tx.send(Event::Line("isready".into())).unwrap();
    let mut effects = Vec::new();
    let mut session = session();
    let error = serve_events(
        &mut session,
        &rx,
        &mut BrokenWriter,
        &mut Vec::new(),
        |effect| {
            effects.push(effect);
            Ok::<_, &'static str>(())
        },
    )
    .unwrap_err();
    assert!(matches!(error.failure, ServeFailure::Io(_)));
    assert!(matches!(effects[1], Effect::Cancel { .. }));
    assert!(matches!(effects[2], Effect::Shutdown));
    assert!(session.is_closed());
}
