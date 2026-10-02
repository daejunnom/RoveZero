//! Actual executable acceptance on CPU/mock, including Windows CI.
//! Five-second bounds catch stalled ownership/drain without asserting millisecond speed.

use rz_position::{BoardMove, Position};
use std::{
    io::{BufRead, BufReader, Read, Write},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const WAIT: Duration = Duration::from_secs(5);
const QUIET: Duration = Duration::from_millis(750);
const MAX_LINE_BYTES: usize = 4096;
const MAX_LINES: usize = 512;
const TERMINAL_FEN: &str = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1";

enum StreamEvent {
    Line(String),
    End,
    Failed(String),
}

#[derive(Debug)]
struct Input {
    line: Option<String>,
    written: SyncSender<Result<(), String>>,
}

/// Pipe producers never share a blocking read with the command/output owner.
/// Dropping this guard on assertion failure closes channels, kills and reaps the child.
struct Engine {
    child: Child,
    input: Option<SyncSender<Input>>,
    stdout: Option<Receiver<StreamEvent>>,
    stderr: Option<Receiver<StreamEvent>>,
    readers: Vec<JoinHandle<()>>,
    protocol: Vec<String>,
    diagnostics: Vec<String>,
    stdout_ended: bool,
    stderr_ended: bool,
}

impl Engine {
    fn spawn(arguments: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_rz-uci"))
            .args(arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn the Cargo-provided rz-uci binary");
        let mut stdin = child.stdin.take().expect("piped stdin");
        let stdout_pipe = child.stdout.take().expect("piped stdout");
        let stderr_pipe = child.stderr.take().expect("piped stderr");
        let (input, commands) = mpsc::sync_channel::<Input>(8);
        let writer = thread::spawn(move || {
            while let Ok(command) = commands.recv() {
                let Some(line) = command.line else {
                    drop(stdin);
                    let _ = command.written.send(Ok(()));
                    return;
                };
                let written = writeln!(stdin, "{line}")
                    .and_then(|()| stdin.flush())
                    .map_err(|error| error.to_string());
                let failed = written.is_err();
                let _ = command.written.send(written);
                if failed {
                    return;
                }
            }
        });
        let (stdout_events, stdout) = mpsc::sync_channel(64);
        let (stderr_events, stderr) = mpsc::sync_channel(64);
        let stdout_reader = thread::spawn(move || read_stream(stdout_pipe, stdout_events));
        let stderr_reader = thread::spawn(move || read_stream(stderr_pipe, stderr_events));
        Self {
            child,
            input: Some(input),
            stdout: Some(stdout),
            stderr: Some(stderr),
            readers: vec![writer, stdout_reader, stderr_reader],
            protocol: Vec::new(),
            diagnostics: Vec::new(),
            stdout_ended: false,
            stderr_ended: false,
        }
    }

    fn send(&mut self, line: &str) {
        assert!(line.len() <= MAX_LINE_BYTES && !line.contains(['\n', '\r']));
        self.write_input(Some(line.to_owned()));
    }

    fn write_input(&mut self, line: Option<String>) {
        let (written, acknowledgment) = mpsc::sync_channel(1);
        self.input
            .as_ref()
            .expect("input remains open")
            .try_send(Input { line, written })
            .expect("bounded input producer accepts one acknowledged command");
        acknowledgment
            .recv_timeout(WAIT)
            .expect("input producer acknowledges before its finite deadline")
            .expect("write the command to child stdin");
    }

    fn close_input(&mut self) {
        if self.input.is_some() {
            self.write_input(None);
            drop(self.input.take());
        }
    }

    fn collect(&mut self) {
        drain_stream(
            self.stdout.as_ref().expect("stdout receiver"),
            &mut self.protocol,
            &mut self.stdout_ended,
        );
        drain_stream(
            self.stderr.as_ref().expect("stderr receiver"),
            &mut self.diagnostics,
            &mut self.stderr_ended,
        );
    }

    fn mark(&mut self) -> usize {
        self.collect();
        self.protocol.len()
    }

    fn wait_line(&mut self, after: usize, predicate: impl Fn(&str) -> bool) -> String {
        let until = Instant::now() + WAIT;
        loop {
            self.collect();
            if let Some(line) = self.protocol[after..].iter().find(|line| predicate(line)) {
                return line.clone();
            }
            assert!(
                Instant::now() < until && !self.stdout_ended,
                "child did not publish the expected protocol; stdout={:?}; stderr={:?}",
                self.protocol,
                self.diagnostics
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn ready(&mut self) {
        let after = self.mark();
        self.send("isready");
        assert_eq!(self.wait_line(after, |line| line == "readyok"), "readyok");
    }

    fn handshake(&mut self) {
        let after = self.mark();
        self.send("uci");
        self.wait_line(after, |line| line == "uciok");
        assert!(
            self.protocol
                .iter()
                .any(|line| line.starts_with("id name "))
        );
        assert!(
            self.protocol
                .iter()
                .any(|line| line.starts_with("id author "))
        );
        self.ready();
    }

    fn bestmove(&mut self, after: usize) -> String {
        let line = self.wait_line(after, |line| line.starts_with("bestmove "));
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        assert_eq!(fields.len(), 2, "unexpected bestmove payload: {line}");
        fields[1].to_owned()
    }

    fn wait_diagnostic(&mut self, code: &str) {
        let until = Instant::now() + WAIT;
        loop {
            self.collect();
            if self.diagnostics.iter().any(|line| line.starts_with(code)) {
                return;
            }
            assert!(
                Instant::now() < until && !self.stderr_ended,
                "missing {code} diagnostic: {:?}",
                self.diagnostics
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn quiet_bestmoves(&mut self, expected: usize) {
        // The window exceeds the deliberately selected 500ms mock callback delay.
        // It checks duplicate/stale output, not a performance or millisecond deadline claim.
        let until = Instant::now() + QUIET;
        while Instant::now() < until {
            self.collect();
            assert_eq!(
                self.bestmove_count(),
                expected,
                "stdout={:?}",
                self.protocol
            );
            thread::sleep(Duration::from_millis(10));
        }
        self.collect();
        assert_eq!(
            self.bestmove_count(),
            expected,
            "stdout={:?}",
            self.protocol
        );
    }

    fn bestmove_count(&self) -> usize {
        self.protocol
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    }

    fn exit(&mut self) -> ExitStatus {
        let until = Instant::now() + WAIT;
        let status = loop {
            self.collect();
            if let Some(status) = self.child.try_wait().expect("poll child exit") {
                break status;
            }
            assert!(
                Instant::now() < until,
                "child failed to exit within the finite bound; stdout={:?}; stderr={:?}",
                self.protocol,
                self.diagnostics
            );
            thread::sleep(Duration::from_millis(10));
        };
        while !self.stdout_ended || !self.stderr_ended {
            self.collect();
            assert!(
                Instant::now() < until,
                "pipe readers did not report child EOF"
            );
            thread::sleep(Duration::from_millis(10));
        }
        status
    }

    fn quit(&mut self) {
        self.send("quit");
        assert!(self.exit().success(), "stderr={:?}", self.diagnostics);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        drop(self.input.take());
        // Unblock producers that were waiting on full output channels before joining.
        drop(self.stdout.take());
        drop(self.stderr.take());
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        let until = Instant::now() + Duration::from_secs(1);
        while self.readers.iter().any(|reader| !reader.is_finished()) && Instant::now() < until {
            thread::sleep(Duration::from_millis(10));
        }
        for reader in self.readers.drain(..) {
            if reader.is_finished() {
                let _ = reader.join();
            }
        }
    }
}

fn read_stream(reader: impl Read, sender: SyncSender<StreamEvent>) {
    let mut reader = BufReader::new(reader);
    loop {
        let mut line = Vec::new();
        let event = match (&mut reader)
            .take((MAX_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
        {
            Ok(0) => StreamEvent::End,
            Ok(_) if line.len() > MAX_LINE_BYTES => {
                StreamEvent::Failed("oversized output line".into())
            }
            Ok(_) => match String::from_utf8(line) {
                Ok(line) => StreamEvent::Line(line.trim_end_matches(['\r', '\n']).to_owned()),
                Err(error) => StreamEvent::Failed(error.to_string()),
            },
            Err(error) => StreamEvent::Failed(error.to_string()),
        };
        let ended = !matches!(event, StreamEvent::Line(_));
        if sender.send(event).is_err() || ended {
            return;
        }
    }
}

fn drain_stream(receiver: &Receiver<StreamEvent>, lines: &mut Vec<String>, ended: &mut bool) {
    loop {
        match receiver.try_recv() {
            Ok(StreamEvent::Line(line)) => {
                assert!(
                    lines.len() < MAX_LINES,
                    "child exceeded the bounded output transcript"
                );
                lines.push(line);
            }
            Ok(StreamEvent::End) | Err(TryRecvError::Disconnected) => {
                *ended = true;
                return;
            }
            Ok(StreamEvent::Failed(error)) => panic!("child pipe reader failed: {error}"),
            Err(TryRecvError::Empty) => return,
        }
    }
}

fn assert_legal(position: &Position, text: &str) {
    let movement = BoardMove::from_uci(text).expect("bestmove has chess move syntax");
    assert!(
        position.legal_moves().contains(&movement),
        "illegal bestmove {text}"
    );
}

#[test]
fn actual_binary_handshake_and_one_simulation_return_a_rules_legal_move() {
    let mut engine = Engine::spawn(&["--cpu-mock", "--mock-delay-ms=0"]);
    engine.handshake();
    engine.send("position startpos");
    let after = engine.mark();
    engine.send("go nodes 1");
    assert_legal(&Position::startpos(), &engine.bestmove(after));
    engine.quit();
    assert_eq!(engine.bestmove_count(), 1);
    assert!(
        !engine
            .diagnostics
            .iter()
            .any(|line| line.contains("SearchFailed")),
        "a fallback after evaluation failure is not successful one-simulation acceptance: {:?}",
        engine.diagnostics
    );
}

#[test]
fn rejected_trace_preserves_the_root_and_root_replacement_newgame_are_distinct() {
    let mut engine = Engine::spawn(&["--cpu-mock"]);
    engine.handshake();
    engine.send("position startpos moves e2e4");
    engine.send("position startpos moves d2d4 d7d5 e1e3");
    engine.ready();
    engine.wait_diagnostic("PositionRejected:");
    let mut black = Position::startpos();
    black
        .make_move(BoardMove::from_uci("e2e4").unwrap())
        .unwrap();
    let after = engine.mark();
    engine.send("go nodes 1");
    assert_legal(&black, &engine.bestmove(after));
    engine.send("go infinite");
    engine.ready();
    engine.send(&format!("position fen {TERMINAL_FEN}"));
    let after = engine.mark();
    engine.send("go nodes 1");
    assert_eq!(engine.bestmove(after), "0000");
    engine.send("ucinewgame");
    engine.ready();
    let after = engine.mark();
    engine.send("go nodes 1");
    assert_legal(&Position::startpos(), &engine.bestmove(after));
    engine.quit();
    assert_eq!(
        engine.bestmove_count(),
        3,
        "old root must not publish an extra move"
    );
}

#[test]
fn tiny_movetime_and_duplicate_stop_publish_the_last_legal_move_once() {
    let mut engine = Engine::spawn(&["--cpu-mock"]);
    engine.handshake();
    let after = engine.mark();
    engine.send("go movetime 1");
    engine.send("stop");
    engine.send("stop");
    engine.ready();
    assert_legal(&Position::startpos(), &engine.bestmove(after));
    engine.quiet_bestmoves(1);
    engine.quit();
    assert_eq!(engine.bestmove_count(), 1);
}

#[test]
fn delayed_mock_stop_and_new_terminal_root_cannot_publish_an_old_move() {
    let mut engine = Engine::spawn(&["--cpu-mock", "--mock-delay-ms=500"]);
    engine.handshake();
    let after = engine.mark();
    engine.send("go infinite");
    engine.ready(); // The input/output owner remains available while mock work is pending.
    engine.send("stop");
    engine.send("stop");
    assert_legal(&Position::startpos(), &engine.bestmove(after));
    engine.send(&format!("position fen {TERMINAL_FEN}"));
    let after = engine.mark();
    engine.send("go nodes 1");
    assert_eq!(engine.bestmove(after), "0000");
    engine.ready();
    engine.quiet_bestmoves(2);
    engine.quit();
    assert_eq!(engine.bestmove_count(), 2);
}

#[test]
fn delayed_mock_movetime_timer_emits_one_legal_move_without_stop() {
    let mut engine = Engine::spawn(&["--cpu-mock", "--mock-delay-ms=500"]);
    engine.handshake();
    let after = engine.mark();
    engine.send("go movetime 200");
    assert_legal(&Position::startpos(), &engine.bestmove(after));
    engine.ready();
    engine.quiet_bestmoves(1);
    engine.quit();
    assert_eq!(engine.bestmove_count(), 1);
}

#[test]
fn quit_and_eof_each_close_an_inflight_search_with_finite_exit() {
    for eof in [false, true] {
        let mut engine = Engine::spawn(&["--cpu-mock", "--mock-delay-ms=500"]);
        engine.handshake();
        engine.send("go infinite");
        engine.ready();
        if eof {
            engine.close_input();
        } else {
            engine.send("quit");
        }
        assert!(engine.exit().success(), "stderr={:?}", engine.diagnostics);
        assert_eq!(
            engine.bestmove_count(),
            0,
            "quit/EOF revokes infinite output ownership"
        );
    }
}

#[test]
fn missing_or_unknown_provider_is_rejected_without_cpu_fallback() {
    for (arguments, cause) in [
        (Vec::new(), "requires --cpu-mock"),
        (
            vec!["--cpu-mock", "--provider=unknown"],
            "unsupported argument; supported flags: --cpu-mock",
        ),
    ] {
        let mut engine = Engine::spawn(&arguments);
        assert_eq!(engine.exit().code(), Some(2));
        assert!(
            engine.protocol.is_empty(),
            "rejected provider started UCI: {:?}",
            engine.protocol
        );
        assert!(
            engine.diagnostics.iter().any(|line| line.contains(cause)),
            "stderr={:?}",
            engine.diagnostics
        );
    }
}
