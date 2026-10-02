//! Opt-in independent legal-set, transition, perft/divide, and undo verification.
//!
//! Install `python-chess==1.999 chess==1.11.2` in a test-only environment, then run:
//! `RZ_CHESS_PYTHON=/path/to/venv/bin/python cargo test --release -p rz-position \
//!     --test reference -- --ignored`
//! The Python dependency is never used by the library or installed automatically.

use std::{
    io::Read,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use rz_position::Position;

const SEED: &str = "20261003";

fn verify_case(lines: &[&str]) -> usize {
    let begin = lines[0].split('\t').collect::<Vec<_>>();
    assert_eq!(begin[0], "BEGIN");
    assert_eq!(begin.len(), 3);
    let name = begin[1];
    let initial_fen = begin[2];
    let mut position = Position::from_fen(initial_fen)
        .unwrap_or_else(|error| panic!("seed={SEED}; case={name}; FEN={initial_fen}: {error:?}"));
    let initial = position.snapshot();
    let mut undo_stack = Vec::new();
    let mut checked_positions = 0;
    let mut current_path = "";

    for line in &lines[1..] {
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(fields[1], name, "oracle case protocol");
        match fields[0] {
            "POSITION" => {
                assert_eq!(fields.len(), 8, "oracle POSITION protocol");
                current_path = fields[2];
                let fen = fields[3];
                assert_eq!(
                    position.to_fen(),
                    fen,
                    "seed={SEED}; case={name}; first differing state after path={current_path}"
                );
                let before = position.snapshot();
                let mut actual = position
                    .legal_moves()
                    .into_iter()
                    .map(|mv| mv.to_string())
                    .collect::<Vec<_>>();
                actual.sort();
                let expected = if fields[4].is_empty() {
                    Vec::new()
                } else {
                    fields[4].split(',').map(str::to_string).collect()
                };
                assert_eq!(
                    actual, expected,
                    "seed={SEED}; case={name}; FEN={fen}; first differing legal set at path={current_path}"
                );
                assert_eq!(
                    position.in_check(),
                    fields[5] == "1",
                    "seed={SEED}; case={name}; check at path={current_path}"
                );
                if fields[6] == "1" || fields[7] == "1" {
                    let reason = if fields[6] == "1" {
                        rz_position::TerminalReason::Checkmate
                    } else {
                        rz_position::TerminalReason::Stalemate
                    };
                    assert!(
                        matches!(position.classify_position().unwrap().play_status, rz_position::PlayStatus::Terminal { reason: actual, .. } if actual == reason),
                        "seed={SEED}; case={name}; terminal at path={current_path}"
                    );
                }
                assert!(
                    position.snapshot().same_state(&before),
                    "seed={SEED}; case={name}; legal generation mutated state at path={current_path}"
                );
                checked_positions += 1;
            }
            "EDGE" => {
                assert_eq!(fields.len(), 5, "oracle EDGE protocol");
                assert_eq!(fields[2], current_path, "oracle EDGE path");
                let before = position.snapshot();
                let parent_fen = position.to_fen();
                let mv = fields[3];
                let undo = position.make_uci(mv).unwrap_or_else(|error| {
                    panic!(
                        "seed={SEED}; case={name}; FEN={parent_fen}; first failing path={current_path}/{mv}: {error:?}"
                    )
                });
                assert_eq!(
                    position.to_fen(), fields[4],
                    "seed={SEED}; case={name}; FEN={parent_fen}; first differing transition path={current_path}/{mv}"
                );
                position.unmake(undo).unwrap_or_else(|error| {
                    panic!("seed={SEED}; case={name}; unmake path={current_path}/{mv}: {error:?}")
                });
                assert!(
                    position.snapshot().same_state(&before),
                    "seed={SEED}; case={name}; FEN={parent_fen}; incomplete undo at path={current_path}/{mv}"
                );
            }
            "DIVIDE" => {
                assert_eq!(fields.len(), 5, "oracle DIVIDE protocol");
                let mv = fields[2];
                let depth = fields[3].parse::<u32>().unwrap();
                let expected_nodes = fields[4].parse::<u64>().unwrap();
                let before = position.snapshot();
                let parent_fen = position.to_fen();
                let undo = position.make_uci(mv).unwrap();
                let actual_nodes = position.perft(depth - 1).unwrap();
                position.unmake(undo).unwrap();
                assert_eq!(
                    actual_nodes, expected_nodes,
                    "seed={SEED}; case={name}; FEN={parent_fen}; depth={depth}; first differing divide path={mv}"
                );
                assert!(position.snapshot().same_state(&before));
            }
            "SELECT" => {
                assert_eq!(fields.len(), 5, "oracle SELECT protocol");
                assert_eq!(fields[2], current_path, "oracle SELECT path");
                let before = position.snapshot();
                let parent_fen = position.to_fen();
                let mv = fields[3];
                let undo = position.make_uci(mv).unwrap_or_else(|error| {
                    panic!(
                        "seed={SEED}; case={name}; FEN={parent_fen}; first failing trace path={current_path}/{mv}: {error:?}"
                    )
                });
                assert_eq!(
                    position.to_fen(), fields[4],
                    "seed={SEED}; case={name}; FEN={parent_fen}; first differing trace path={current_path}/{mv}"
                );
                undo_stack.push((undo, before, mv.to_owned()));
            }
            "END" => {
                assert_eq!(fields.len(), 2, "oracle END protocol");
                while let Some((undo, before, mv)) = undo_stack.pop() {
                    position.unmake(undo).unwrap_or_else(|error| {
                        panic!("seed={SEED}; case={name}; long trace unmake {mv}: {error:?}")
                    });
                    assert!(
                        position.snapshot().same_state(&before),
                        "seed={SEED}; case={name}; full trace path={current_path}; first incomplete undo={mv}"
                    );
                }
                assert!(position.snapshot().same_state(&initial));
                assert_eq!(position.to_fen(), initial_fen);
            }
            unexpected => panic!("unexpected oracle record: {unexpected}"),
        }
    }
    assert_eq!(lines.last().unwrap(), &format!("END\t{name}"));
    checked_positions
}

#[test]
#[ignore = "independent test-only oracle requires pinned python-chess and chess packages"]
fn pinned_python_chess_differential() {
    let python = std::env::var_os("RZ_CHESS_PYTHON").unwrap_or_else(|| "python3".into());
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle.py");
    let mut child = Command::new(python)
        .arg(script)
        .args(["--seed", SEED, "--games", "8", "--plies", "96"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("cannot start independent Python oracle; set RZ_CHESS_PYTHON to the test venv");
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    // Read concurrently to avoid pipe backpressure while applying a process deadline.
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr.take(64 * 1024 + 1).read_to_end(&mut bytes).unwrap();
        bytes
    });
    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() >= Duration::from_secs(30) {
            timed_out = true;
            let _ = child.kill();
            break child.wait().unwrap();
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_reader.join().unwrap();
    let stderr = stderr_reader.join().unwrap();
    assert!(
        !timed_out,
        "independent oracle exceeded its 30-second process deadline"
    );
    assert!(
        status.success(),
        "independent oracle failed; install exactly python-chess==1.999 chess==1.11.2; stderr={}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(stdout.len() <= 8 * 1024 * 1024, "oracle output budget");
    let stdout = String::from_utf8(stdout).expect("non-UTF8 oracle output");
    let mut lines = stdout.lines();
    assert_eq!(
        lines.next().unwrap(),
        format!("ORACLE\tpython-chess\t1.999\tchess\t1.11.2\t{SEED}")
    );
    let mut current_case = Vec::new();
    let mut cases = 0;
    let mut positions = 0;
    for line in lines {
        current_case.push(line);
        if line.starts_with("END\t") {
            positions += verify_case(&current_case);
            cases += 1;
            current_case.clear();
        }
    }
    assert!(current_case.is_empty(), "incomplete oracle case");
    assert_eq!(
        cases, 23,
        "six public cases, nine special cases, eight traces"
    );
    assert!(
        positions > 600,
        "differential trace coverage unexpectedly short"
    );
    eprintln!("oracle seed={SEED}; cases={cases}; checked positions={positions}");
}
