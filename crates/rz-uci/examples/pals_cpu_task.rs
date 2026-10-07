//! One bounded JSON CPU_T request. No UCI engine/GPU/model/teacher is launched.
use rz_uci::pals_cpu_task::{
    CpuTaskError, MAX_REQUEST_BYTES, MAX_SELF_BINARY_BYTES, MAX_WALL_TIME_MS, capabilities,
    dispatch_started, request_admission,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn failure(stage: &'static str, message: impl std::fmt::Display) -> CpuTaskError {
    CpuTaskError {
        code: "cpu_task_cli_failed",
        stage,
        message: message.to_string().chars().take(512).collect::<String>().into_boxed_str(),
        known_nodes: None,
        failed_check_work: None,
        baseline: None,
        after: None,
        elapsed_ms: None,
        deadline_exceeded: false,
        output_limit: 1024,
    }
}

fn read_request(started: Instant) -> Result<Vec<u8>, CpuTaskError> {
    // A blocked stdin cannot defeat the CLI's finite outer bound. This worker
    // owns only stdin, creates no child, and dies with a nonzero process exit.
    let (send, receive) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("cpu-task-stdin".into())
        .spawn(move || {
            let mut bytes = Vec::new();
            let result = io::stdin()
                .lock()
                .take(MAX_REQUEST_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map(|_| bytes);
            let _ = send.send(result);
        })
        .map_err(|e| failure("stdin", e))?;
    let remaining = Duration::from_millis(MAX_WALL_TIME_MS).saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(failure("stdin", "finite outer stdin deadline expired"));
    }
    let bytes = receive
        .recv_timeout(remaining)
        .map_err(|e| failure("stdin", e))?
        .map_err(|e| failure("stdin", e))?;
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        return Err(failure("stdin", "request exceeds 1..=512KiB admission"));
    }
    Ok(bytes)
}

fn own_binary_sha(deadline: Instant) -> Result<String, CpuTaskError> {
    // On Linux the proc handle names the loaded executable inode even if a
    // path is replaced after launch. Other hosts use current_exe's own image.
    #[cfg(target_os = "linux")]
    let path = std::path::PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let path = std::env::current_exe().map_err(|e| failure("self_binary", e))?;
    let mut file = File::open(path).map_err(|e| failure("self_binary", e))?;
    let metadata = file.metadata().map_err(|e| failure("self_binary", e))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_SELF_BINARY_BYTES {
        return Err(failure(
            "self_binary",
            "loaded executable exceeds bounded regular-file admission",
        ));
    }
    let mut sha = Sha256::new();
    let mut block = [0_u8; 32 * 1024];
    let mut total = 0_u64;
    loop {
        if Instant::now() >= deadline {
            return Err(failure(
                "self_binary",
                "absolute request deadline expired while hashing own image",
            ));
        }
        let read = file
            .read(&mut block)
            .map_err(|e| failure("self_binary", e))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| failure("self_binary", "executable byte counter overflow"))?;
        if total > MAX_SELF_BINARY_BYTES || total > metadata.len() {
            return Err(failure(
                "self_binary",
                "executable changed or exceeded finite byte bound",
            ));
        }
        sha.update(&block[..read]);
    }
    let after = file.metadata().map_err(|e| failure("self_binary", e))?;
    if total != metadata.len()
        || after.len() != metadata.len()
        || after.modified().ok() != metadata.modified().ok()
    {
        return Err(failure(
            "self_binary",
            "executable changed during identity verification",
        ));
    }
    Ok(format!("{:x}", sha.finalize()))
}

struct CliContext {
    started: Instant,
    deadline: Instant,
    output_limit: usize,
}

fn annotate(mut error: CpuTaskError, context: &CliContext) -> CpuTaskError {
    error.elapsed_ms = Some(
        context
            .started
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64,
    );
    error.deadline_exceeded = Instant::now() >= context.deadline;
    error.output_limit = context.output_limit;
    error
}

fn run(context: &mut CliContext) -> Result<Vec<u8>, CpuTaskError> {
    let started = context.started;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--capabilities"] {
        let deadline = started
            .checked_add(Duration::from_millis(MAX_WALL_TIME_MS))
            .ok_or_else(|| failure("admission", "capability deadline overflow"))?;
        let binary = own_binary_sha(deadline)?;
        let mut value: serde_json::Value =
            serde_json::from_slice(&capabilities()?).map_err(|e| failure("capabilities", e))?;
        value["cpu_binary_sha256"] = json!(binary);
        let mut result = serde_json::to_vec(&value).map_err(|e| failure("capabilities", e))?;
        result.push(b'\n');
        return Ok(result);
    }
    if !args.is_empty() {
        return Err(failure(
            "arguments",
            "only no arguments or --capabilities are supported",
        ));
    }
    let bytes = read_request(started)?;
    let admission = request_admission(&bytes, started)?;
    context.deadline = admission.deadline;
    context.output_limit = admission.output_limit;
    let binary = own_binary_sha(context.deadline)?;
    dispatch_started(&bytes, &binary, started)
}

fn bounded_write<W: Write + Send + 'static>(
    mut writer: W,
    bytes: Vec<u8>,
    deadline: Instant,
) -> Result<(), CpuTaskError> {
    if Instant::now() >= deadline {
        return Err(failure(
            "output_deadline",
            "original absolute wall allowance expired before output",
        ));
    }
    let (send, receive) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("cpu-task-output".into())
        .spawn(move || {
            let result = writer.write_all(&bytes).and_then(|_| writer.flush());
            let _ = send.send(result);
        })
        .map_err(|e| failure("output", e))?;
    receive
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|e| failure("output_deadline", e))?
        .map_err(|e| failure("output", e))?;
    if Instant::now() >= deadline {
        return Err(failure(
            "output_deadline",
            "late output cannot be accepted as success",
        ));
    }
    Ok(())
}

fn output_file(stdout: bool) -> Result<File, CpuTaskError> {
    // Duplicate the native handle safely and write through File. A blocked
    // worker must never hold stdio's global lock during Rust runtime cleanup.
    #[cfg(unix)]
    {
        use std::os::fd::AsFd;
        let handle = if stdout {
            io::stdout().as_fd().try_clone_to_owned()
        } else {
            io::stderr().as_fd().try_clone_to_owned()
        };
        handle.map(File::from).map_err(|e| failure("output", e))
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsHandle;
        let handle = if stdout {
            io::stdout().as_handle().try_clone_to_owned()
        } else {
            io::stderr().as_handle().try_clone_to_owned()
        };
        handle.map(File::from).map_err(|e| failure("output", e))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = stdout;
        Err(failure(
            "output",
            "deadline-bounded native output is unsupported on this target",
        ))
    }
}

fn main() -> ExitCode {
    let started = Instant::now();
    let Some(deadline) = started.checked_add(Duration::from_millis(MAX_WALL_TIME_MS)) else {
        return ExitCode::from(1);
    };
    let mut context = CliContext {
        started,
        deadline,
        output_limit: 1024,
    };
    match run(&mut context) {
        Ok(bytes) => {
            match output_file(true).and_then(|file| bounded_write(file, bytes, context.deadline)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(_) => ExitCode::from(1),
            }
        }
        Err(error) => {
            let error = annotate(error, &context);
            // One bounded diagnostic on stderr; success emits no diagnostics.
            // If the richer receipt does not fit, retain honest known/unknown
            // work counters rather than replacing the error with a fake result.
            let mut bytes = serde_json::to_vec(&error)
                .unwrap_or_else(|_| b"{\"code\":\"cpu_task_error_serialization_failed\"}".to_vec());
            if bytes.len() + 1 > error.output_limit {
                bytes=serde_json::to_vec(&json!({"code":error.code,"stage":error.stage,
                    "known_nodes":error.known_nodes,"failed_check_work":error.failed_check_work,
                    "baseline_present":error.baseline.is_some(),"after_present":error.after.is_some(),
                    "elapsed_ms":error.elapsed_ms,"deadline_exceeded":error.deadline_exceeded,
                    "full_error_omitted_for_output_bound":true}))
                    .unwrap_or_else(|_|b"{\"code\":\"cpu_task_failed\"}".to_vec());
            }
            bytes.push(b'\n');
            // When the original deadline is exhausted, do not invent an extra
            // output grace. External supervisors retain partial pipe evidence.
            let _ =
                output_file(false).and_then(|file| bounded_write(file, bytes, context.deadline));
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BlockedWriter(mpsc::Receiver<()>);
    impl Write for BlockedWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            self.0
                .recv()
                .map_err(|_| io::Error::other("test writer release failed"))?;
            Err(io::Error::other("released test writer"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn unread_output_cannot_block_beyond_the_absolute_deadline() {
        let (send, receive) = mpsc::sync_channel(1);
        let deadline = Instant::now() + Duration::from_millis(20);
        let error = bounded_write(BlockedWriter(receive), vec![b'x'; 65536], deadline).unwrap_err();
        assert_eq!(error.stage, "output_deadline");
        let _ = send.send(());
    }

    #[test]
    fn cli_failure_uses_admitted_deadline_and_actual_elapsed() {
        let started = Instant::now() - Duration::from_millis(25);
        let context = CliContext {
            started,
            deadline: started + Duration::from_millis(1),
            output_limit: 4096,
        };
        let error = annotate(failure("self_binary", "deadline expired"), &context);
        assert!(error.deadline_exceeded);
        assert!(error.elapsed_ms.unwrap() >= 25);
        assert_eq!(error.output_limit, 4096);
        assert_eq!(error.known_nodes, None);
    }
}
