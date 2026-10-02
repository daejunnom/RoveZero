#![cfg(target_os = "linux")]

use nix::sys::signal::{SigSet, SigmaskHow, Signal, kill, pthread_sigmask};
use nix::unistd::Pid;
use rz_arena::{
    ArenaError, ArtifactWatch, CleanupStatus, ProcessLimits, ProcessStop, supervise,
    supervise_in_directory, supervise_with_watch,
};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "rz-e02-process-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture_args(name: &str) -> Vec<OsString> {
    [
        "--ignored",
        "--exact",
        name,
        "--nocapture",
        "--test-threads=1",
    ]
    .into_iter()
    .map(OsString::from)
    .collect()
}

fn limits() -> ProcessLimits {
    ProcessLimits {
        wall_ms: 1500,
        shutdown_grace_ms: 100,
        max_output_bytes: 16_384,
        max_child_processes: 1,
    }
}

fn execute(name: &str, limits: ProcessLimits) -> rz_arena::ProcessOutput {
    let scratch = Scratch::new();
    let program = File::open(std::env::current_exe().unwrap()).unwrap();
    let start = Instant::now();
    let result = supervise(&program, &fixture_args(name), &scratch.0, limits, None).unwrap();
    assert!(start.elapsed() < Duration::from_secs(5));
    result
}

#[test]
fn native_runner_exit_preserves_both_outputs_and_closed_stdin() {
    let result = execute("fixture_normal", limits());
    assert_eq!(result.receipt.stop, ProcessStop::Exited);
    assert_eq!(result.receipt.exit_code, Some(0));
    assert_eq!(result.receipt.exit_signal, None);
    assert_eq!(result.receipt.group_cleanup, CleanupStatus::Gone);
    assert!(result.receipt.errors.is_empty());
    assert!(result.pending_child.is_none());
    assert_eq!(result.receipt.watched_artifact_bytes, None);
    assert!(String::from_utf8_lossy(&result.stdout).contains("fixture stdout"));
    assert!(String::from_utf8_lossy(&result.stderr).contains("fixture stderr"));
    assert_eq!(result.receipt.stdout_bytes, result.stdout.len() as u64);
    assert_eq!(result.receipt.stderr_bytes, result.stderr.len() as u64);
    assert_eq!(
        result.receipt.observed_output_bytes,
        result.stdout.len() as u64 + result.stderr.len() as u64
    );
    assert_eq!(
        result.receipt.child_limit_enforcement,
        "process_group_snapshot"
    );
}

#[test]
fn exit_failure_and_signal_are_not_general_success() {
    let result = execute("fixture_nonzero", limits());
    assert_eq!(result.receipt.stop, ProcessStop::Exited);
    assert_eq!(result.receipt.exit_code, Some(7));
    let result = execute("fixture_signal", limits());
    assert_eq!(result.receipt.stop, ProcessStop::Exited);
    assert_eq!(result.receipt.exit_code, None);
    assert_eq!(result.receipt.exit_signal, Some(Signal::SIGKILL as i32));
}

#[test]
fn opened_program_survives_replacement_of_its_former_path() {
    let scratch = Scratch::new();
    let path = scratch.0.join("runner");
    std::os::unix::fs::symlink(std::env::current_exe().unwrap(), &path).unwrap();
    let program = File::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, b"not an executable\n").unwrap();
    let result = supervise(
        &program,
        &fixture_args("fixture_normal"),
        &scratch.0,
        limits(),
        None,
    )
    .unwrap();
    assert_eq!(result.receipt.stop, ProcessStop::Exited);
    assert_eq!(result.receipt.exit_code, Some(0));
    assert!(String::from_utf8_lossy(&result.stdout).contains("fixture stdout"));
}

#[test]
fn directory_handle_remains_bound_after_its_former_path_is_replaced() {
    let scratch = Scratch::new();
    let relocated = Scratch::new();
    let directory = File::open(&scratch.0).unwrap();
    std::fs::remove_dir(&relocated.0).unwrap();
    std::fs::rename(&scratch.0, &relocated.0).unwrap();
    std::fs::create_dir(&scratch.0).unwrap();
    let program = File::open(std::env::current_exe().unwrap()).unwrap();
    let watch = ArtifactWatch {
        relative_files: vec![PathBuf::from("match.pgn")],
        max_total_bytes: 1024,
    };
    let result = supervise_in_directory(
        &program,
        &fixture_args("fixture_cwd_marker"),
        &directory,
        limits(),
        None,
        Some(&watch),
    )
    .unwrap();
    assert_eq!(result.receipt.stop, ProcessStop::Exited);
    assert_eq!(result.receipt.exit_code, Some(0));
    assert_eq!(
        std::fs::read(relocated.0.join("match.pgn")).unwrap(),
        b"pinned working directory\n"
    );
    assert!(!scratch.0.join("match.pgn").exists());
    assert_eq!(result.receipt.watched_artifact_bytes, Some(25));
    let not_directory = File::open(std::env::current_exe().unwrap()).unwrap();
    assert!(matches!(
        supervise_in_directory(&program, &[], &not_directory, limits(), None, None),
        Err(ArenaError::Invalid(_))
    ));
}

#[test]
fn non_utf8_process_name_does_not_break_group_observation() {
    let result = execute("fixture_non_utf8_name", limits());
    assert_eq!(result.receipt.stop, ProcessStop::Exited);
    assert_eq!(result.receipt.exit_code, Some(0));
    assert!(result.receipt.errors.is_empty());
}

#[test]
fn growing_regular_artifact_stops_the_runner_before_wall_deadline() {
    let scratch = Scratch::new();
    let program = File::open(std::env::current_exe().unwrap()).unwrap();
    let watch = ArtifactWatch {
        relative_files: vec![PathBuf::from("match.pgn")],
        max_total_bytes: 1024,
    };
    let result = supervise_with_watch(
        &program,
        &fixture_args("fixture_artifact_growth"),
        &scratch.0,
        limits(),
        None,
        &watch,
    )
    .unwrap();
    assert_eq!(result.receipt.stop, ProcessStop::ArtifactLimit);
    assert_eq!(result.receipt.group_cleanup, CleanupStatus::Gone);
    assert_eq!(
        result.receipt.artifact_limit_enforcement.as_deref(),
        Some("regular_file_snapshot")
    );
    assert!(result.receipt.watched_artifact_bytes.unwrap() > 1024);
    assert!(result.receipt.elapsed_ns < 1_000_000_000);
}

#[test]
fn artifact_fifo_or_symlink_is_rejected_without_opening_its_payload() {
    for fifo in [true, false] {
        let scratch = Scratch::new();
        let program = File::open(std::env::current_exe().unwrap()).unwrap();
        let path = scratch.0.join("match.pgn");
        let outside = scratch.0.join("outside.pgn");
        std::fs::write(&outside, b"outside unchanged\n").unwrap();
        if fifo {
            nix::unistd::mkfifo(&path, nix::sys::stat::Mode::S_IRUSR).unwrap();
        } else {
            std::os::unix::fs::symlink(&outside, &path).unwrap();
        }
        let watch = ArtifactWatch {
            relative_files: vec![PathBuf::from("match.pgn")],
            max_total_bytes: 1024,
        };
        let start = Instant::now();
        let result = supervise_with_watch(
            &program,
            &fixture_args("fixture_sleep"),
            &scratch.0,
            limits(),
            None,
            &watch,
        )
        .unwrap();
        assert_eq!(result.receipt.stop, ProcessStop::IoFailure);
        assert!(
            result
                .receipt
                .errors
                .iter()
                .any(|code| code == "process.artifact_kind")
        );
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(std::fs::read(outside).unwrap(), b"outside unchanged\n");
    }
}

#[test]
fn artifact_watch_rejects_unbounded_or_traversing_configuration() {
    let scratch = Scratch::new();
    let program = File::open(std::env::current_exe().unwrap()).unwrap();
    for paths in [
        vec![],
        vec![PathBuf::from("../match.pgn")],
        vec![PathBuf::from("nested/match.pgn")],
        vec![PathBuf::from("match.pgn"), PathBuf::from("match.pgn")],
    ] {
        let watch = ArtifactWatch {
            relative_files: paths,
            max_total_bytes: 1024,
        };
        assert!(supervise_with_watch(&program, &[], &scratch.0, limits(), None, &watch).is_err());
    }
    let watch = ArtifactWatch {
        relative_files: vec![PathBuf::from("match.pgn")],
        max_total_bytes: 0,
    };
    assert!(matches!(
        supervise_with_watch(&program, &[], &scratch.0, limits(), None, &watch),
        Err(ArenaError::Budget(_))
    ));
}

#[test]
fn wall_limit_kills_a_runner_with_term_blocked_and_drains() {
    // The calling thread's signal mask is inherited across exec, before the
    // harness creates any other threads. This requires no unsafe signal hook.
    let mut mask = SigSet::empty();
    mask.add(Signal::SIGTERM);
    let mut previous = SigSet::empty();
    pthread_sigmask(SigmaskHow::SIG_BLOCK, Some(&mask), Some(&mut previous)).unwrap();
    struct Restore(SigSet);
    impl Drop for Restore {
        fn drop(&mut self) {
            pthread_sigmask(SigmaskHow::SIG_SETMASK, Some(&self.0), None).unwrap();
        }
    }
    let _restore = Restore(previous);
    let mut policy = limits();
    policy.wall_ms = 100;
    let result = execute("fixture_sleep", policy);
    assert_eq!(result.receipt.stop, ProcessStop::WallLimit);
    assert_eq!(result.receipt.exit_signal, Some(Signal::SIGKILL as i32));
    assert_eq!(result.receipt.group_cleanup, CleanupStatus::Gone);
    assert!(result.receipt.elapsed_ns < 1_000_000_000);
}

#[test]
fn combined_output_limit_retains_only_the_budget() {
    let mut policy = limits();
    policy.max_output_bytes = 512;
    let result = execute("fixture_flood", policy);
    assert_eq!(result.receipt.stop, ProcessStop::OutputLimit);
    assert_eq!(result.stdout.len() + result.stderr.len(), 512);
    assert!(result.receipt.observed_output_bytes > 512);
    assert_eq!(result.receipt.group_cleanup, CleanupStatus::Gone);
}

#[test]
fn cancellation_stops_and_reaps_the_owned_runner() {
    let scratch = Scratch::new();
    let program = File::open(std::env::current_exe().unwrap()).unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let sender = Arc::clone(&flag);
    let thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(75));
        sender.store(true, Ordering::Release);
    });
    let result = supervise(
        &program,
        &fixture_args("fixture_sleep"),
        &scratch.0,
        limits(),
        Some(&flag),
    )
    .unwrap();
    thread.join().unwrap();
    assert_eq!(result.receipt.stop, ProcessStop::Cancelled);
    assert_eq!(result.receipt.group_cleanup, CleanupStatus::Gone);
}

#[test]
fn surviving_descendant_pipe_cannot_block_drain_forever() {
    let mut policy = limits();
    policy.max_child_processes = 2;
    let result = execute("fixture_descendant", policy);
    assert_eq!(result.receipt.stop, ProcessStop::Exited);
    assert_eq!(result.receipt.exit_code, Some(0));
    assert!(result.receipt.descendant_cleanup_required);
    // An orphan zombie may be awaiting PID1's reaper. Preserve the real
    // observation rather than pretending the group is absent in that case.
    if result.receipt.group_cleanup == CleanupStatus::Unverified {
        assert!(
            result
                .receipt
                .errors
                .iter()
                .any(|code| code == "process.cleanup_unverified")
        );
    }
    assert!(result.receipt.elapsed_ns < 1_000_000_000);
}

#[test]
fn observed_child_limit_terminates_group() {
    let result = execute("fixture_descendant_alive", limits());
    assert_eq!(result.receipt.stop, ProcessStop::ChildLimit);
    assert!(result.receipt.elapsed_ns < 1_000_000_000);
}

#[test]
fn invalid_limits_scripts_and_fifo_are_rejected_before_spawn() {
    let scratch = Scratch::new();
    let program = File::open(std::env::current_exe().unwrap()).unwrap();
    for policy in [
        ProcessLimits {
            wall_ms: 0,
            ..limits()
        },
        ProcessLimits {
            shutdown_grace_ms: u64::MAX,
            ..limits()
        },
        ProcessLimits {
            max_output_bytes: 0,
            ..limits()
        },
        ProcessLimits {
            max_child_processes: 0,
            ..limits()
        },
    ] {
        assert!(matches!(
            supervise(&program, &[], &scratch.0, policy, None),
            Err(ArenaError::Budget(_))
        ));
    }
    let script = scratch.0.join("script");
    std::fs::write(&script, b"#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(matches!(
        supervise(
            &File::open(script).unwrap(),
            &[],
            &scratch.0,
            limits(),
            None
        ),
        Err(ArenaError::Invalid(_))
    ));
    let fifo = scratch.0.join("fifo");
    nix::unistd::mkfifo(&fifo, nix::sys::stat::Mode::S_IRUSR).unwrap();
    let fifo_file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(fifo)
        .unwrap();
    assert!(matches!(
        supervise(&fifo_file, &[], &scratch.0, limits(), None),
        Err(ArenaError::Invalid(_))
    ));
    let flag = AtomicBool::new(true);
    assert!(matches!(
        supervise(
            &program,
            &fixture_args("fixture_sleep"),
            &scratch.0,
            limits(),
            Some(&flag)
        ),
        Err(ArenaError::Invalid(_))
    ));
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_normal() {
    assert_eq!(std::env::var("LANG").unwrap(), "C");
    assert_eq!(std::env::var("PATH").unwrap(), "/usr/bin:/bin");
    assert_eq!(std::env::vars_os().count(), 2);
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    assert!(bytes.is_empty());
    println!("fixture stdout");
    eprintln!("fixture stderr");
    std::process::exit(0);
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_non_utf8_name() {
    // Rename the process leader, not the harness's worker thread.
    std::fs::write("/proc/self/comm", [b'r', b'z', 0xff, 0xfe]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    std::process::exit(0);
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_artifact_growth() {
    let mut file = File::create("match.pgn").unwrap();
    loop {
        file.write_all(&[b'x'; 256]).unwrap();
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_cwd_marker() {
    std::fs::write("match.pgn", b"pinned working directory\n").unwrap();
    std::process::exit(0);
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_nonzero() {
    std::process::exit(7);
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_signal() {
    kill(Pid::this(), Signal::SIGKILL).unwrap();
    unreachable!();
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_sleep() {
    println!("sleep started");
    std::thread::sleep(Duration::from_secs(3600));
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_flood() {
    let bytes = [b'x'; 4096];
    loop {
        if std::io::stdout().write_all(&bytes).is_err() {
            std::process::exit(1);
        }
        if std::io::stderr().write_all(&bytes).is_err() {
            std::process::exit(1);
        }
    }
}

fn descendant() {
    let child = Command::new(std::env::current_exe().unwrap())
        .args(fixture_args("fixture_sleep"))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    println!("descendant pid {}", child.id());
    drop(child);
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_descendant() {
    descendant();
    std::process::exit(0);
}

#[test]
#[ignore = "native helper; launched only by supervisor tests"]
fn fixture_descendant_alive() {
    descendant();
    std::thread::sleep(Duration::from_secs(3600));
}
