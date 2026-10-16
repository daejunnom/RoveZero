//! Search-driver executable acceptance, separate from NN/GPU evidence.
use rz_position::{BoardMove, Position};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant},
};

const MAX_STDERR_BYTES: usize = 16 * 1024;
const MAX_STDERR_LINES: usize = 64;

struct StderrDiagnostics {
    bytes: Vec<u8>,
    lines: usize,
    truncated: bool,
    eof: bool,
    read_error: Option<io::ErrorKind>,
}
impl StderrDiagnostics {
    fn new() -> Self {
        Self {
            bytes: Vec::with_capacity(MAX_STDERR_BYTES),
            lines: 0,
            truncated: false,
            eof: false,
            read_error: None,
        }
    }
    fn capture(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if self.bytes.len() == MAX_STDERR_BYTES || self.lines == MAX_STDERR_LINES {
                self.truncated = true;
                break;
            }
            self.bytes.push(byte);
            if byte == b'\n' {
                self.lines += 1;
            }
        }
    }
    fn finished(&self) -> bool {
        self.eof || self.read_error.is_some()
    }
}

struct CpuChild {
    process: Child,
    stdout: Receiver<String>,
    lines: Vec<String>,
    stderr: Arc<Mutex<StderrDiagnostics>>,
}
impl CpuChild {
    fn start() -> Self {
        Self::start_with(&[
            "--search=cpu",
            "--cpu-max-depth=64",
            "--cpu-max-nodes=1000000",
            "--cpu-tt-entries=128",
        ])
    }
    fn start_with(arguments: &[&str]) -> Self {
        let mut process = Command::new(env!("CARGO_BIN_EXE_rz-uci"))
            .args(arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let pipe = process.stdout.take().unwrap();
        let (sender, stdout) = mpsc::sync_channel(128);
        thread::spawn(move || {
            for line in BufReader::new(pipe).lines().take(512) {
                let Ok(line) = line else { break };
                if line.len() > 4096 || sender.send(line).is_err() {
                    break;
                }
            }
        });
        let mut stderr_pipe = process.stderr.take().unwrap();
        let stderr = Arc::new(Mutex::new(StderrDiagnostics::new()));
        let reader_diagnostics = Arc::clone(&stderr);
        thread::spawn(move || {
            let mut buffer = [0_u8; 1024];
            loop {
                match stderr_pipe.read(&mut buffer) {
                    Ok(0) => {
                        reader_diagnostics.lock().unwrap().eof = true;
                        break;
                    }
                    Ok(count) => {
                        // Continue draining after the capture cap, so diagnostics
                        // never introduce pipe backpressure into the UCI child.
                        reader_diagnostics.lock().unwrap().capture(&buffer[..count]);
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        reader_diagnostics.lock().unwrap().read_error = Some(error.kind());
                        break;
                    }
                }
            }
        });
        Self {
            process,
            stdout,
            lines: Vec::new(),
            stderr,
        }
    }
    fn send(&mut self, line: &str) {
        writeln!(self.process.stdin.as_mut().unwrap(), "{line}").unwrap();
        self.process.stdin.as_mut().unwrap().flush().unwrap();
    }
    fn wait(&mut self, prefix: &str) -> String {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let line = match self
                .stdout
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok(line) => line,
                Err(error) => self.wait_failed(prefix, error, deadline),
            };
            self.lines.push(line.clone());
            if line.starts_with(prefix) {
                return line;
            }
        }
    }
    fn wait_failed(&mut self, prefix: &str, error: RecvTimeoutError, deadline: Instant) -> ! {
        let mut status = self.process.try_wait();
        // stdout and stderr have independent readers. Give an exiting child
        // time to publish its final diagnostic, within the original wait bound.
        while Instant::now() < deadline
            && (matches!(status, Ok(None)) || !self.stderr.lock().unwrap().finished())
        {
            thread::sleep(
                Duration::from_millis(10).min(deadline.saturating_duration_since(Instant::now())),
            );
            status = self.process.try_wait();
        }
        let stderr = self.stderr.lock().unwrap();
        let executable_bytes =
            std::fs::metadata(env!("CARGO_BIN_EXE_rz-uci")).map(|metadata| metadata.len());
        panic!(
            "CPU UCI did not publish {prefix:?}; stdout_receive={error:?}; \
             child_status={status:?}; stdout={:?}; stderr={:?}; \
             stderr_eof={}; stderr_read_error={:?}; stderr_truncated={}; \
             executable_bytes={executable_bytes:?}",
            self.lines,
            String::from_utf8_lossy(&stderr.bytes),
            stderr.eof,
            stderr.read_error,
            stderr.truncated,
        );
    }
    fn quit(&mut self) {
        self.send("quit");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.process.try_wait().unwrap() {
                assert!(status.success());
                return;
            }
            assert!(
                Instant::now() < deadline,
                "CPU UCI failed to drain and quit"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}
impl Drop for CpuChild {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

#[test]
fn own_cpu_runs_without_a_neural_backend_and_preserves_rules_terminals() {
    let mut child = CpuChild::start();
    child.send("uci");
    child.wait("uciok");
    assert!(
        child
            .lines
            .iter()
            .any(|line| line == "id name RoveZero own Rust CPU_R")
    );
    child.send("isready");
    child.wait("readyok");
    child.send("position startpos");
    child.send("go nodes 32");
    let best = child.wait("bestmove ");
    let movement = BoardMove::from_uci(best.trim_start_matches("bestmove ")).unwrap();
    assert!(
        Position::startpos()
            .ordered_legal_moves()
            .moves()
            .contains(&movement)
    );
    child.send("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    child.send("go nodes 1");
    assert_eq!(child.wait("bestmove "), "bestmove 0000");
    child.quit();
}

#[test]
fn cpu_stop_root_replacement_and_new_game_do_not_publish_an_old_move() {
    let mut child = CpuChild::start();
    child.send("uci");
    child.wait("uciok");
    child.send("position startpos");
    child.send("go infinite");
    child.send("isready");
    child.wait("readyok");
    child.send("position startpos moves e2e4");
    child.send("go nodes 1");
    let best = child.wait("bestmove ");
    let mut black = Position::startpos();
    black
        .make_move(BoardMove::from_uci("e2e4").unwrap())
        .unwrap();
    let movement = BoardMove::from_uci(best.trim_start_matches("bestmove ")).unwrap();
    assert!(black.ordered_legal_moves().moves().contains(&movement));
    child.send("ucinewgame");
    child.send("go infinite");
    child.send("stop");
    child.send("stop");
    let stopped = child.wait("bestmove ");
    let movement = BoardMove::from_uci(stopped.trim_start_matches("bestmove ")).unwrap();
    assert!(
        Position::startpos()
            .ordered_legal_moves()
            .moves()
            .contains(&movement)
    );
    child.send("isready");
    child.wait("readyok");
    assert_eq!(
        child
            .lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count(),
        2
    );
    child.quit();
}

#[test]
fn pals_without_an_injected_session_and_cpu_neural_flags_are_explicitly_rejected() {
    for flags in [
        vec!["--search=pals"],
        vec![
            "--search=pals",
            "--pals-model=legal-order-mock",
            "--pals-provider=cpu",
        ],
        vec!["--search=pals", "--pals-model=onnx", "--pals-provider=cuda"],
        vec!["--search=cpu", "--onnx-cuda"],
        vec!["--search=cpu", "--search=puct"],
        vec!["--search=cpu", "--cpu-max-nodes=0"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_rz-uci"))
            .args(flags)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}

#[test]
fn explicitly_selected_pals_mock_uses_the_search_session_and_exact_history() {
    let mut child = CpuChild::start_with(&[
        "--search=pals",
        "--pals-model=legal-order-mock",
        "--pals-max-rounds=1",
        "--pals-max-cpu-nodes=256",
        "--pals-cpu-depth=1",
    ]);
    child.send("uci");
    child.wait("uciok");
    assert!(
        child
            .lines
            .iter()
            .any(|line| line.contains("PALS explicit legal-order CPU mock"))
    );
    child.send("position startpos moves e2e4");
    child.send("go nodes 32");
    let best = child.wait("bestmove ");
    let mut position = Position::startpos();
    position.make_uci("e2e4").unwrap();
    let movement = BoardMove::from_uci(best.trim_start_matches("bestmove ")).unwrap();
    assert!(position.ordered_legal_moves().moves().contains(&movement));
    child.send("ucinewgame");
    child.send("go nodes 1");
    let best = child.wait("bestmove ");
    let movement = BoardMove::from_uci(best.trim_start_matches("bestmove ")).unwrap();
    assert!(
        Position::startpos()
            .ordered_legal_moves()
            .moves()
            .contains(&movement)
    );
    child.quit();
}

#[cfg(feature = "search-work-receipts")]
#[test]
fn own_cpu_executable_publishes_actual_work_without_a_neural_receipt() {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };
    static ROOT_IDS: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "rz-search-work-{}-{}",
        std::process::id(),
        ROOT_IDS.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let root_arg = format!("--search-work-output-root={}", root.display());
    let launch_arg = format!("--search-work-launch-sha256={}", "ab".repeat(32));
    let mut child = CpuChild::start_with(&[
        "--search=cpu",
        "--cpu-max-depth=2",
        "--cpu-max-nodes=32",
        &root_arg,
        &launch_arg,
        "--search-work-endpoint-id=cpu-fixture",
    ]);
    let pid = child.process.id();
    child.send("uci");
    child.wait("uciok");
    child.send("position startpos");
    child.send("go nodes 32");
    child.wait("bestmove ");
    child.quit();
    let slot = root.join(format!("native-process-{pid}"));
    let startup_path = slot.join("search-work-startup.v3.json");
    let final_path = slot.join("search-work-termination.v3.json");
    let startup: serde_json::Value =
        serde_json::from_slice(&fs::read(&startup_path).unwrap()).unwrap();
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(&final_path).unwrap()).unwrap();
    assert_eq!(startup["search_work"]["go_invocations"], 0);
    assert_eq!(receipt["process_id"], pid);
    assert_eq!(receipt["service_exit_success"], true);
    assert_eq!(receipt["search_work"]["search_kind"], "cpu");
    assert_eq!(receipt["search_work"]["go_invocations"], 1);
    assert_eq!(receipt["search_work"]["active_invocations"], 0);
    assert_eq!(receipt["search_work"]["unobserved_work_invocations"], 0);
    assert_eq!(receipt["search_work"]["cpu"]["tasks_requested"], 1);
    assert!(receipt["search_work"]["cpu"]["nodes"].as_u64().unwrap() > 0);
    assert!(receipt.get("native").is_none());
    fs::remove_file(startup_path).unwrap();
    fs::remove_file(final_path).unwrap();
    fs::remove_dir(slot).unwrap();
    fs::remove_dir(root).unwrap();
}
