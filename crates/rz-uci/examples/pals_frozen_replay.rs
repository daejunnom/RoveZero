//! Explicit CPU-only caller for one independently registered frozen replay.
//!
//! The first start instant covers argv, stdin, independent files, native-cache
//! verification, image inspection, manifest preparation, dispatch and delivery.
//! This caller never promotes declared historical binary/source/authority pins.
//! Synchronous filesystem/ORT/pipe operations need an external process supervisor
//! for a hard interruption/resource/loaded-provider/physical-closure witness.
//! The shared credit bounds this caller's owned stdout/stderr frames. Aggregate
//! process output, including unexpected native diagnostics/panic output, requires
//! that separate supervisor's bounded capture; it is not proved by this ledger.

#[cfg(not(feature = "onnx-cpu"))]
mod feature_refusal {
    use std::fs::File;
    use std::io::{self, Write};
    use std::process::ExitCode;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    const MESSAGE: &[u8] = b"{\"code\":\"onnx_cpu_feature_required\",\"native_assets_loaded\":false,\"native_work_observed\":false}\n";
    fn stderr_file() -> io::Result<File> {
        #[cfg(unix)]
        {
            use std::os::fd::AsFd;
            io::stderr().as_fd().try_clone_to_owned().map(File::from)
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsHandle;
            io::stderr()
                .as_handle()
                .try_clone_to_owned()
                .map(File::from)
        }
        #[cfg(not(any(unix, windows)))]
        {
            Err(io::Error::other("native output handle unsupported"))
        }
    }
    pub(super) fn run() -> ExitCode {
        let started = Instant::now();
        let Some(deadline) = started.checked_add(Duration::from_secs(1)) else {
            return ExitCode::from(1);
        };
        // No argv/stdin/assets/dependencies are read on this default path. The
        // bounded wait does not assert interruption of a blocked OS operation.
        if let Ok(mut writer) = stderr_file() {
            let (send, receive) = mpsc::sync_channel(1);
            if std::thread::Builder::new()
                .name("frozen-replay-feature-refusal".into())
                .spawn(move || {
                    let result = (|| -> io::Result<()> {
                        if Instant::now() >= deadline {
                            return Err(io::Error::other("original refusal deadline expired"));
                        }
                        writer.write_all(MESSAGE)?;
                        if Instant::now() >= deadline {
                            return Err(io::Error::other(
                                "original refusal deadline expired before flush",
                            ));
                        }
                        writer.flush()
                    })();
                    drop(writer);
                    let _ = send.send(result);
                })
                .is_ok()
            {
                let _ = receive.recv_timeout(deadline.saturating_duration_since(Instant::now()));
            }
        }
        ExitCode::from(1)
    }
    #[cfg(test)]
    mod tests {
        #[test]
        fn default_refusal_is_fixed_small_and_has_no_native_success_claim() {
            assert!(super::MESSAGE.len() < 1024);
            assert!(super::MESSAGE.ends_with(b"\n"));
            assert!(
                std::str::from_utf8(super::MESSAGE)
                    .unwrap()
                    .contains("\"native_assets_loaded\":false")
            );
        }
    }
}
#[cfg(not(feature = "onnx-cpu"))]
fn main() -> std::process::ExitCode {
    feature_refusal::run()
}
#[cfg(feature = "onnx-cpu")]
fn main() -> std::process::ExitCode {
    cpu_cli::run_main()
}

#[cfg(feature = "onnx-cpu")]
mod cpu_cli {
    use rz_uci::pals_cpu_task::strategic_action::ArtifactPin;
    use rz_uci::pals_cpu_task::strategic_action::replay_inputs::SemanticReceiptProducerScope;
    use rz_uci::pals_cpu_task::strategic_action::replay_launch_preparation::NativeResultLane;
    use serde::Serialize;
    use sha2::{Digest, Sha256};
    #[cfg(any(feature = "onnx-cpu", test))]
    use std::ffi::OsString;
    use std::fs::File;
    #[cfg(feature = "onnx-cpu")]
    use std::fs::Metadata;
    #[cfg(feature = "onnx-cpu")]
    use std::io::Read;
    use std::io::{self, Write};
    #[cfg(feature = "onnx-cpu")]
    use std::path::Path;
    #[cfg(any(feature = "onnx-cpu", test))]
    use std::path::{Component, PathBuf};
    use std::process::ExitCode;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard, mpsc};
    use std::time::{Duration, Instant};

    #[cfg(feature = "onnx-cpu")]
    const EXPECTED_SCHEMA: &str = "rz-pals-frozen-replay-cli-expected/1";
    #[cfg(feature = "onnx-cpu")]
    const EXPECTED_SCHEMA_V2: &str = "rz-pals-frozen-replay-cli-expected/2";
    const EXPECTED_SCHEMA_V3: &str = "rz-pals-frozen-replay-cli-expected/3";
    #[cfg(feature = "onnx-cpu")]
    const LAUNCH_SCHEMA: &str = "rz-pals-frozen-replay-cli-assets/1";
    const CLI_SCHEMA: &str = "rz-pals-frozen-replay-cli-delivery/1";
    const MAX_WALL_MS: u64 = 300_000;
    #[cfg(any(feature = "onnx-cpu", test))]
    const MAX_EXPECTED_BYTES: usize = 64 * 1024;
    #[cfg(feature = "onnx-cpu")]
    const MAX_LAUNCH_BYTES: usize = 128 * 1024;
    #[cfg(feature = "onnx-cpu")]
    const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
    #[cfg(any(feature = "onnx-cpu", test))]
    const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
    #[cfg(feature = "onnx-cpu")]
    const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
    #[cfg(feature = "onnx-cpu")]
    const MAX_RUNTIME_BYTES: u64 = 512 * 1024 * 1024;
    #[cfg(feature = "onnx-cpu")]
    const MAX_SELF_BINARY_BYTES: u64 = 1024 * 1024 * 1024;
    const DIAGNOSTIC_BYTES: usize = 1024;
    const HEADER_RESERVE_BYTES: usize = 4096;
    #[cfg(any(feature = "onnx-cpu", test))]
    const MAX_PATH_BYTES: usize = 4096;
    #[cfg(any(feature = "onnx-cpu", test))]
    const MAX_ARG_BYTES: usize = 16 * 1024;

    #[derive(Debug, Serialize)]
    struct TransportError {
        code: &'static str,
        stage: &'static str,
        message: String,
        #[serde(skip)]
        terminal: Option<Arc<WriteProgress>>,
    }
    impl TransportError {
        fn new(stage: &'static str, message: impl std::fmt::Display) -> Self {
            Self {
                code: "frozen_replay_cli_failed",
                stage,
                message: message.to_string().chars().take(256).collect(),
                terminal: None,
            }
        }
        fn with_terminal(
            stage: &'static str,
            message: impl std::fmt::Display,
            terminal: Arc<WriteProgress>,
        ) -> Self {
            let mut error = Self::new(stage, message);
            error.terminal = Some(terminal);
            error
        }
    }
    type CliResult<T> = Result<T, TransportError>;

    #[cfg(any(feature = "onnx-cpu", test))]
    #[derive(Debug)]
    struct Arguments {
        expected_path: PathBuf,
        expected_pin: ArtifactPin,
        launch_path: PathBuf,
    }
    #[cfg(any(feature = "onnx-cpu", test))]
    fn valid_sha(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }
    #[cfg(any(feature = "onnx-cpu", test))]
    fn absolute_path(value: &str) -> CliResult<PathBuf> {
        let path = PathBuf::from(value);
        if value.is_empty()
            || value.len() > MAX_PATH_BYTES
            || value.chars().any(char::is_control)
            || value
                .split(['/', '\\'])
                .any(|segment| matches!(segment, "." | ".."))
            || !path.is_absolute()
            || path
                .components()
                .any(|p| matches!(p, Component::CurDir | Component::ParentDir))
        {
            return Err(TransportError::new(
                "path_admission",
                "bounded absolute lexical path without traversal required",
            ));
        }
        Ok(path)
    }
    #[cfg(any(feature = "onnx-cpu", test))]
    fn arguments(values: &[OsString]) -> CliResult<Arguments> {
        if values.len() != 8 {
            return Err(TransportError::new(
                "arguments",
                "exactly four explicit option/value pairs required",
            ));
        }
        let mut total = 0_usize;
        let mut expected = None;
        let mut digest = None;
        let mut count = None;
        let mut launch = None;
        for pair in values.chunks_exact(2) {
            let flag = pair[0]
                .to_str()
                .ok_or_else(|| TransportError::new("arguments", "UTF-8 option required"))?;
            let value = pair[1]
                .to_str()
                .ok_or_else(|| TransportError::new("arguments", "UTF-8 value required"))?;
            total = total
                .checked_add(flag.len())
                .and_then(|v| v.checked_add(value.len()))
                .filter(|v| *v <= MAX_ARG_BYTES)
                .ok_or_else(|| TransportError::new("arguments", "argv byte bound exceeded"))?;
            match flag {
                "--expected-pins" if expected.is_none() => expected = Some(absolute_path(value)?),
                "--expected-sha256" if digest.is_none() && valid_sha(value) => {
                    digest = Some(value.to_owned())
                }
                "--expected-bytes" if count.is_none() => {
                    let bytes = value
                        .parse::<u64>()
                        .map_err(|e| TransportError::new("arguments", e))?;
                    if bytes == 0 || bytes > MAX_EXPECTED_BYTES as u64 || bytes.to_string() != value
                    {
                        return Err(TransportError::new(
                            "arguments",
                            "canonical positive expected bytes exceed bound",
                        ));
                    }
                    count = Some(bytes);
                }
                "--asset-profile" if launch.is_none() => launch = Some(absolute_path(value)?),
                _ => {
                    return Err(TransportError::new(
                        "arguments",
                        "unknown, duplicate or invalid option; no implicit modes or fallback",
                    ));
                }
            }
        }
        Ok(Arguments {
            expected_path: expected
                .ok_or_else(|| TransportError::new("arguments", "expected path missing"))?,
            expected_pin: ArtifactPin {
                bytes: count
                    .ok_or_else(|| TransportError::new("arguments", "expected bytes missing"))?,
                sha256: digest
                    .ok_or_else(|| TransportError::new("arguments", "expected SHA missing"))?,
            },
            launch_path: launch
                .ok_or_else(|| TransportError::new("arguments", "asset path missing"))?,
        })
    }
    fn artifact(bytes: &[u8]) -> ArtifactPin {
        ArtifactPin {
            bytes: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    }
    #[cfg(any(feature = "onnx-cpu", test))]
    fn check_pin(pin: &ArtifactPin, maximum: u64) -> CliResult<()> {
        if pin.bytes == 0 || pin.bytes > maximum || !valid_sha(&pin.sha256) {
            return Err(TransportError::new(
                "artifact_admission",
                "positive bounded bytes and lowercase SHA256 required",
            ));
        }
        Ok(())
    }
    #[cfg(any(feature = "onnx-cpu", test))]
    fn check_bytes(bytes: &[u8], expected: &ArtifactPin, maximum: u64) -> CliResult<()> {
        check_pin(expected, maximum)?;
        if artifact(bytes) != *expected {
            return Err(TransportError::new(
                "artifact_identity",
                "actual raw bytes differ from independent pin",
            ));
        }
        Ok(())
    }
    fn check_time(deadline: Instant, stage: &'static str) -> CliResult<()> {
        if Instant::now() >= deadline {
            Err(TransportError::new(
                stage,
                "original absolute allowance expired; no new grace",
            ))
        } else {
            Ok(())
        }
    }
    #[cfg(any(feature = "onnx-cpu", test))]
    fn check_execution(
        deadline: Instant,
        cancel: &AtomicBool,
        stage: &'static str,
    ) -> CliResult<()> {
        if cancel.load(Ordering::Acquire) {
            return Err(TransportError::new(
                stage,
                "cooperative cancellation requested",
            ));
        }
        check_time(deadline, stage)
    }
    #[cfg(any(feature = "onnx-cpu", test))]
    fn hex_digest(bytes: &[u8; 32]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = String::with_capacity(64);
        for byte in bytes {
            text.push(char::from(HEX[usize::from(byte >> 4)]));
            text.push(char::from(HEX[usize::from(byte & 15)]));
        }
        text
    }
    #[cfg(feature = "onnx-cpu")]
    fn is_link(metadata: &Metadata) -> bool {
        if metadata.file_type().is_symlink() {
            return true;
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            metadata.file_attributes() & 0x400 != 0
        }
        #[cfg(not(windows))]
        {
            false
        }
    }
    #[cfg(feature = "onnx-cpu")]
    fn same_open_file(before: &Metadata, after: &Metadata) -> bool {
        let unchanged = before.is_file()
            && after.is_file()
            && before.len() == after.len()
            && before.modified().ok() == after.modified().ok();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            unchanged && before.dev() == after.dev() && before.ino() == after.ino()
        }
        #[cfg(not(unix))]
        {
            unchanged
        }
    }
    #[cfg(feature = "onnx-cpu")]
    fn read_verified(
        path: &Path,
        expected: &ArtifactPin,
        maximum: usize,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> CliResult<Vec<u8>> {
        check_pin(expected, maximum as u64)?;
        check_execution(deadline, cancel, "asset_deadline")?;
        let path_metadata = std::fs::symlink_metadata(path)
            .map_err(|e| TransportError::new("asset_metadata", e))?;
        if is_link(&path_metadata)
            || !path_metadata.is_file()
            || path_metadata.len() != expected.bytes
        {
            return Err(TransportError::new(
                "asset_metadata",
                "independently pinned bounded regular nonlink file required",
            ));
        }
        let mut file = File::open(path).map_err(|e| TransportError::new("asset_open", e))?;
        check_execution(deadline, cancel, "asset_deadline")?;
        let before = file
            .metadata()
            .map_err(|e| TransportError::new("asset_metadata", e))?;
        if !same_open_file(&path_metadata, &before) {
            return Err(TransportError::new(
                "asset_identity",
                "opened file differs from admitted metadata",
            ));
        }
        let size =
            usize::try_from(expected.bytes).map_err(|e| TransportError::new("asset_size", e))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|e| TransportError::new("asset_allocation", e))?;
        if bytes.capacity() > size {
            return Err(TransportError::new(
                "asset_allocation",
                "actual backing capacity exceeds admitted bytes",
            ));
        }
        let mut block = [0_u8; 32 * 1024];
        loop {
            check_execution(deadline, cancel, "asset_deadline")?;
            let count = file
                .read(&mut block)
                .map_err(|e| TransportError::new("asset_read", e))?;
            if count == 0 {
                break;
            }
            if bytes
                .len()
                .checked_add(count)
                .is_none_or(|total| total > size)
            {
                return Err(TransportError::new(
                    "asset_size",
                    "file changed or exceeded independent byte bound",
                ));
            }
            bytes.extend_from_slice(&block[..count]);
        }
        let after = file
            .metadata()
            .map_err(|e| TransportError::new("asset_metadata", e))?;
        if !same_open_file(&before, &after) {
            return Err(TransportError::new(
                "asset_identity",
                "file changed during raw byte verification",
            ));
        }
        check_bytes(&bytes, expected, maximum as u64)?;
        check_execution(deadline, cancel, "asset_deadline")?;
        Ok(bytes)
    }
    fn binary_scope() -> &'static str {
        if cfg!(target_os = "linux") {
            "linux_loaded_executable_inode"
        } else {
            "current_exe_path_hash"
        }
    }
    #[cfg(feature = "onnx-cpu")]
    fn own_binary(
        expected: &ArtifactPin,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> CliResult<ArtifactPin> {
        check_pin(expected, MAX_SELF_BINARY_BYTES)?;
        #[cfg(target_os = "linux")]
        let path = PathBuf::from("/proc/self/exe");
        #[cfg(not(target_os = "linux"))]
        let path = std::env::current_exe().map_err(|e| TransportError::new("self_binary", e))?;
        check_execution(deadline, cancel, "self_binary_deadline")?;
        let mut file = File::open(path).map_err(|e| TransportError::new("self_binary", e))?;
        let before = file
            .metadata()
            .map_err(|e| TransportError::new("self_binary", e))?;
        if !before.is_file() || before.len() != expected.bytes {
            return Err(TransportError::new(
                "self_binary",
                "own image size differs from replay registration",
            ));
        }
        let mut hash = Sha256::new();
        let mut block = [0_u8; 32 * 1024];
        let mut bytes = 0_u64;
        loop {
            check_execution(deadline, cancel, "self_binary_deadline")?;
            let count = file
                .read(&mut block)
                .map_err(|e| TransportError::new("self_binary", e))?;
            if count == 0 {
                break;
            }
            bytes = bytes
                .checked_add(count as u64)
                .filter(|n| *n <= expected.bytes)
                .ok_or_else(|| {
                    TransportError::new("self_binary", "own image exceeded admitted byte bound")
                })?;
            hash.update(&block[..count]);
        }
        let after = file
            .metadata()
            .map_err(|e| TransportError::new("self_binary", e))?;
        let actual = ArtifactPin {
            bytes,
            sha256: format!("{:x}", hash.finalize()),
        };
        if !same_open_file(&before, &after) || actual != *expected {
            return Err(TransportError::new(
                "self_binary",
                "actual image differs from independently registered replay binary",
            ));
        }
        check_execution(deadline, cancel, "self_binary_deadline")?;
        Ok(actual)
    }
    #[cfg(feature = "onnx-cpu")]
    fn read_stdin(deadline: Instant, cancel: &AtomicBool) -> CliResult<Vec<u8>> {
        check_execution(deadline, cancel, "stdin_deadline")?;
        let (send, receive) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("frozen-replay-stdin".into())
            .spawn(move || {
                let mut bytes = Vec::new();
                let result = io::stdin()
                    .lock()
                    .take(MAX_REQUEST_BYTES as u64 + 1)
                    .read_to_end(&mut bytes)
                    .map(|_| bytes);
                let _ = send.send(result);
            })
            .map_err(|e| TransportError::new("stdin", e))?;
        let bytes = receive
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|e| TransportError::new("stdin_deadline", e))?
            .map_err(|e| TransportError::new("stdin", e))?;
        if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
            return Err(TransportError::new(
                "stdin",
                "raw frozen replay input exceeds 1..=2MiB",
            ));
        }
        check_execution(deadline, cancel, "stdin_deadline")?;
        Ok(bytes)
    }

    /// Charged bytes include committed writes and the full possible remainder of
    /// any outstanding attempt. Timeout is not thread exit or a zero-byte write.
    struct StreamBudget {
        limit: usize,
        charged: AtomicUsize,
    }
    impl StreamBudget {
        fn new(limit: usize) -> Self {
            Self {
                limit,
                charged: AtomicUsize::new(0),
            }
        }
        fn reserve(&self, bytes: usize) -> CliResult<()> {
            self.charged
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    current
                        .checked_add(bytes)
                        .filter(|next| *next <= self.limit)
                })
                .map(|_| ())
                .map_err(|_| {
                    TransportError::new(
                        "output_reservation",
                        "stdout/stderr potential bytes exceed original aggregate cap",
                    )
                })
        }
        fn release_definite_unused(&self, bytes: usize) {
            self.charged.fetch_sub(bytes, Ordering::AcqRel);
        }
        fn remaining(&self) -> usize {
            self.limit
                .saturating_sub(self.charged.load(Ordering::Acquire))
        }
    }
    #[derive(Debug)]
    struct WriteTerminal {
        stage: &'static str,
        result: io::Result<()>,
    }
    #[derive(Debug)]
    struct WriteProgress {
        bytes: usize,
        charged: AtomicUsize,
        written: AtomicUsize,
        writer_started: AtomicBool,
        writer_closed: AtomicBool,
        flushed: AtomicBool,
        terminal: Mutex<Option<WriteTerminal>>,
        terminal_poisoned: AtomicBool,
        wait_timed_out: AtomicBool,
        wait_disconnected: AtomicBool,
    }
    #[derive(Serialize)]
    struct WriteObservation {
        prepared_bytes: usize,
        reserved_or_committed_bytes: usize,
        confirmed_bytes_written: usize,
        writer_started: bool,
        writer_closed: bool,
        flush_returned_ok: bool,
        possible_unfinished_bytes: usize,
        terminal_stage: Option<&'static str>,
        terminal_succeeded: Option<bool>,
        terminal_io_kind: Option<String>,
        terminal_raw_os_error: Option<i32>,
        terminal_message_prefix: Option<String>,
        terminal_record_poisoned: bool,
        wait_timed_out: bool,
        wait_disconnected: bool,
    }
    impl WriteProgress {
        fn new(bytes: usize) -> Self {
            Self {
                bytes,
                charged: AtomicUsize::new(0),
                written: AtomicUsize::new(0),
                writer_started: AtomicBool::new(false),
                writer_closed: AtomicBool::new(true),
                flushed: AtomicBool::new(false),
                terminal: Mutex::new(None),
                terminal_poisoned: AtomicBool::new(false),
                wait_timed_out: AtomicBool::new(false),
                wait_disconnected: AtomicBool::new(false),
            }
        }
        fn terminal_guard(&self) -> MutexGuard<'_, Option<WriteTerminal>> {
            match self.terminal.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    self.terminal_poisoned.store(true, Ordering::Release);
                    poisoned.into_inner()
                }
            }
        }
        fn terminal_result(progress: &Arc<Self>) -> CliResult<()> {
            let terminal = progress.terminal_guard();
            if progress.terminal_poisoned.load(Ordering::Acquire) {
                return Err(TransportError::with_terminal(
                    "output_terminal_record",
                    "terminal record poisoned; original record retained",
                    Arc::clone(progress),
                ));
            }
            match terminal.as_ref() {
                Some(WriteTerminal { result: Ok(()), .. }) => Ok(()),
                Some(WriteTerminal {
                    stage,
                    result: Err(error),
                }) => Err(TransportError::with_terminal(
                    stage,
                    error,
                    Arc::clone(progress),
                )),
                None => Err(TransportError::with_terminal(
                    "output_terminal_record",
                    "notification without terminal record",
                    Arc::clone(progress),
                )),
            }
        }
        fn observe(&self) -> WriteObservation {
            let written = self.written.load(Ordering::Acquire);
            let closed = self.writer_closed.load(Ordering::Acquire);
            let terminal = self.terminal_guard();
            let io_error = terminal
                .as_ref()
                .and_then(|record| record.result.as_ref().err());
            WriteObservation {
                prepared_bytes: self.bytes,
                reserved_or_committed_bytes: self.charged.load(Ordering::Acquire),
                confirmed_bytes_written: written,
                writer_started: self.writer_started.load(Ordering::Acquire),
                writer_closed: closed,
                flush_returned_ok: self.flushed.load(Ordering::Acquire),
                possible_unfinished_bytes: if closed {
                    0
                } else {
                    self.bytes.saturating_sub(written)
                },
                terminal_stage: terminal.as_ref().map(|record| record.stage),
                terminal_succeeded: terminal.as_ref().map(|record| record.result.is_ok()),
                terminal_io_kind: io_error.map(|error| format!("{:?}", error.kind())),
                terminal_raw_os_error: io_error.and_then(io::Error::raw_os_error),
                terminal_message_prefix: io_error
                    .map(|error| error.to_string().chars().take(256).collect()),
                terminal_record_poisoned: self.terminal_poisoned.load(Ordering::Acquire),
                wait_timed_out: self.wait_timed_out.load(Ordering::Acquire),
                wait_disconnected: self.wait_disconnected.load(Ordering::Acquire),
            }
        }
    }
    fn native_output_file(stdout: bool) -> CliResult<File> {
        #[cfg(unix)]
        {
            use std::os::fd::AsFd;
            let fd = if stdout {
                io::stdout().as_fd().try_clone_to_owned()
            } else {
                io::stderr().as_fd().try_clone_to_owned()
            };
            fd.map(File::from)
                .map_err(|e| TransportError::new("output_handle", e))
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsHandle;
            let handle = if stdout {
                io::stdout().as_handle().try_clone_to_owned()
            } else {
                io::stderr().as_handle().try_clone_to_owned()
            };
            handle
                .map(File::from)
                .map_err(|e| TransportError::new("output_handle", e))
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = stdout;
            Err(TransportError::new(
                "output_handle",
                "bounded native output unsupported on this target",
            ))
        }
    }
    fn start_write<W: Write + Send + 'static>(
        mut writer: W,
        bytes: Arc<Vec<u8>>,
        deadline: Instant,
        budget: Arc<StreamBudget>,
        progress: Arc<WriteProgress>,
        retained_evidence: Option<SharedEvidence>,
    ) -> CliResult<mpsc::Receiver<()>> {
        check_time(deadline, "output_deadline")?;
        if bytes.len() != progress.bytes {
            return Err(TransportError::new(
                "output_reservation",
                "prepared frame and reservation size differ",
            ));
        }
        budget.reserve(bytes.len())?;
        progress.charged.store(bytes.len(), Ordering::Release);
        progress.writer_closed.store(false, Ordering::Release);
        let thread_budget = Arc::clone(&budget);
        let thread_progress = Arc::clone(&progress);
        let (send, receive) = mpsc::sync_channel(1);
        let spawned = std::thread::Builder::new()
            .name("frozen-replay-output".into())
            .spawn(move || {
                // A timed-out caller cannot drop the original native byte evidence while
                // this owned OS writer is still outstanding. This is retention, not an
                // assertion that the worker thread or an OS operation can be interrupted.
                let _retained_evidence = retained_evidence;
                thread_progress
                    .writer_started
                    .store(true, Ordering::Release);
                let terminal = (|| -> WriteTerminal {
                    let mut written = 0;
                    while written < bytes.len() {
                        if Instant::now() >= deadline {
                            return WriteTerminal {
                                stage: "deadline_before_write",
                                result: Err(io::Error::new(
                                    io::ErrorKind::TimedOut,
                                    "original output deadline expired",
                                )),
                            };
                        }
                        let count = match writer.write(&bytes[written..]) {
                            Ok(count) => count,
                            Err(error) => {
                                return WriteTerminal {
                                    stage: "write",
                                    result: Err(error),
                                };
                            }
                        };
                        if count == 0 {
                            return WriteTerminal {
                                stage: "write",
                                result: Err(io::Error::from(io::ErrorKind::WriteZero)),
                            };
                        }
                        if count > bytes.len() - written {
                            return WriteTerminal {
                                stage: "write_protocol",
                                result: Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "writer returned an impossible byte count",
                                )),
                            };
                        }
                        written += count;
                        thread_progress.written.store(written, Ordering::Release);
                    }
                    if Instant::now() >= deadline {
                        return WriteTerminal {
                            stage: "deadline_before_flush",
                            result: Err(io::Error::new(
                                io::ErrorKind::TimedOut,
                                "original output deadline expired before flush",
                            )),
                        };
                    }
                    if let Err(error) = writer.flush() {
                        return WriteTerminal {
                            stage: "flush",
                            result: Err(error),
                        };
                    }
                    thread_progress.flushed.store(true, Ordering::Release);
                    WriteTerminal {
                        stage: "completed",
                        result: Ok(()),
                    }
                })();
                // One bounded shared slot owns the exact io::Error before any
                // notification. Dropping a timed-out receiver cannot lose a later
                // OS write/flush cause, and observation never consumes that cause.
                *thread_progress.terminal_guard() = Some(terminal);
                // No later native write/flush can consume refunded credit. A pending
                // synchronous OS call cannot reach this point until it actually returns.
                drop(writer);
                let written = thread_progress.written.load(Ordering::Acquire);
                thread_budget
                    .release_definite_unused(thread_progress.bytes.saturating_sub(written));
                thread_progress.charged.store(written, Ordering::Release);
                thread_progress.writer_closed.store(true, Ordering::Release);
                let _ = send.send(());
            });
        if let Err(error) = spawned {
            *progress.terminal_guard() = Some(WriteTerminal {
                stage: "spawn",
                result: Err(error),
            });
            budget.release_definite_unused(progress.bytes);
            progress.charged.store(0, Ordering::Release);
            progress.writer_closed.store(true, Ordering::Release);
            return WriteProgress::terminal_result(&progress).and(Ok(receive));
        }
        Ok(receive)
    }
    fn bounded_write<W: Write + Send + 'static>(
        writer: W,
        bytes: Arc<Vec<u8>>,
        deadline: Instant,
        budget: Arc<StreamBudget>,
        progress: Arc<WriteProgress>,
        retained_evidence: Option<SharedEvidence>,
    ) -> CliResult<()> {
        let receive = start_write(
            writer,
            bytes,
            deadline,
            budget,
            Arc::clone(&progress),
            retained_evidence,
        )?;
        match receive.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(()) => WriteProgress::terminal_result(&progress)?,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                progress.wait_timed_out.store(true, Ordering::Release);
                return Err(TransportError::with_terminal(
                    "output_wait_timeout",
                    "original wait deadline expired; pending writer/terminal record retained",
                    progress,
                ));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                progress.wait_disconnected.store(true, Ordering::Release);
                return Err(TransportError::with_terminal(
                    "output_wait_disconnected",
                    "writer notification disconnected; terminal record retained",
                    progress,
                ));
            }
        }
        check_time(deadline, "output_deadline")
    }

    type SharedEvidence = Arc<Mutex<EvidenceBundle>>;
    struct DeliveryErrors {
        primary: Option<TransportError>,
        secondary: Option<TransportError>,
    }
    struct EvidenceBundle {
        raw_body: Option<Arc<Vec<u8>>>,
        native_error: Option<Box<enabled::NativeReplayError>>,
        input_error: Option<Box<enabled::ReplayInputError>>,
        _delivery_errors: Arc<Mutex<DeliveryErrors>>,
        serialization_error: Option<serde_json::Error>,
    }
    #[derive(Serialize)]
    struct InputErrorClockObservation {
        original_whole_deadline_retained: bool,
        transport_whole_wall_ms: Option<u64>,
        transport_cleanup_reserve_ms: Option<u64>,
        transport_output_limit: usize,
        request_whole_wall_ms: Option<u64>,
        request_cleanup_reserve_ms: Option<u64>,
        request_output_limit: usize,
        transport_execution_allowance_ms: u64,
        request_execution_allowance_ms: Option<u64>,
        effective_execution_allowance_ms: u64,
        effective_whole_wall_ms: Option<u64>,
        effective_output_limit: usize,
        declarations_differ: bool,
        invalid_retained_request_clock: bool,
    }
    #[derive(Serialize)]
    struct ClockDeclarationObservation {
        whole_wall_ms: Option<u64>,
        cleanup_reserve_ms: Option<u64>,
        deadline_after_start_ms: u64,
        execution_deadline_after_start_ms: u64,
        output_bytes: usize,
        output_limit: usize,
    }
    #[derive(Serialize)]
    struct EffectiveClockObservation {
        deadline_after_start_ms: u64,
        execution_deadline_after_start_ms: u64,
        output_limit: usize,
    }
    #[derive(Serialize)]
    struct AdmittedInputClockObservation {
        scope: &'static str,
        transport: ClockDeclarationObservation,
        request: ClockDeclarationObservation,
        effective: EffectiveClockObservation,
        declarations_differ: bool,
    }
    // Private clock observations only. This helper input cannot construct checked
    // replay inputs, a native owner, provider, completion or execution authority.
    struct RequestClockFacts {
        deadline: Instant,
        execution_deadline: Instant,
        output_limit: usize,
        whole_wall_ms: u64,
        cleanup_reserve_ms: u64,
        output_bytes: usize,
    }
    fn clock_after_start_ms(started: Instant, deadline: Instant) -> u64 {
        deadline
            .saturating_duration_since(started)
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    }
    /// A bounded serialization sink uses the same effective original W and cap.
    /// Its error is a preparation failure; it never asserts native completion.
    struct BoundedJsonWriter {
        bytes: Vec<u8>,
        deadline: Instant,
        limit: usize,
    }
    impl Write for BoundedJsonWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if Instant::now() >= self.deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "effective original serialization deadline expired",
                ));
            }
            if self
                .bytes
                .len()
                .checked_add(bytes.len())
                .is_none_or(|size| size > self.limit)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "effective original serialization cap exceeded",
                ));
            }
            self.bytes
                .try_reserve(bytes.len())
                .map_err(io::Error::other)?;
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if Instant::now() >= self.deadline {
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "effective original serialization deadline expired",
                ))
            } else {
                Ok(())
            }
        }
    }
    fn bounded_json<T: Serialize>(
        value: &T,
        deadline: Instant,
        limit: usize,
    ) -> Result<Vec<u8>, serde_json::Error> {
        let mut writer = BoundedJsonWriter {
            bytes: Vec::new(),
            deadline,
            limit,
        };
        serde_json::to_writer(&mut writer, value)?;
        Ok(writer.bytes)
    }
    struct Context {
        started: Instant,
        deadline: Instant,
        execution_deadline: Instant,
        output_limit: usize,
        whole_wall_ms: Option<u64>,
        cleanup_reserve_ms: Option<u64>,
        expected_transport: Option<ArtifactPin>,
        // An independently registered expectation, not a historical image witness.
        expected_semantic_receipt_producer_scope: Option<SemanticReceiptProducerScope>,
        requested_native_result: Option<NativeResultLane>,
        launch_artifact: Option<ArtifactPin>,
        request_artifact: Option<ArtifactPin>,
        replay_binary: Option<ArtifactPin>,
        runtime_library: Option<ArtifactPin>,
        raw_body: Option<Arc<Vec<u8>>>,
        raw_body_kind: &'static str,
        native_dispatch_entered: bool,
        progress: Vec<Arc<WriteProgress>>,
        #[cfg(feature = "onnx-cpu")]
        native_error: Option<Box<enabled::NativeReplayError>>,
        #[cfg(feature = "onnx-cpu")]
        input_error: Option<Box<enabled::ReplayInputError>>,
        shared_evidence: Option<SharedEvidence>,
        input_error_clock: Option<InputErrorClockObservation>,
        admitted_input_clock: Option<AdmittedInputClockObservation>,
        serialization_error: Option<serde_json::Error>,
        delivery_errors: Arc<Mutex<DeliveryErrors>>,
    }
    impl Context {
        fn new(started: Instant) -> Option<Self> {
            let deadline = started.checked_add(Duration::from_millis(MAX_WALL_MS))?;
            Some(Self {
                started,
                deadline,
                execution_deadline: deadline,
                output_limit: DIAGNOSTIC_BYTES,
                whole_wall_ms: None,
                cleanup_reserve_ms: None,
                expected_transport: None,
                expected_semantic_receipt_producer_scope: None,
                requested_native_result: None,
                launch_artifact: None,
                request_artifact: None,
                replay_binary: None,
                runtime_library: None,
                raw_body: None,
                raw_body_kind: "none",
                native_dispatch_entered: false,
                progress: Vec::new(),
                #[cfg(feature = "onnx-cpu")]
                native_error: None,
                #[cfg(feature = "onnx-cpu")]
                input_error: None,
                shared_evidence: None,
                input_error_clock: None,
                admitted_input_clock: None,
                serialization_error: None,
                delivery_errors: Arc::new(Mutex::new(DeliveryErrors {
                    primary: None,
                    secondary: None,
                })),
            })
        }
        #[cfg(any(feature = "onnx-cpu", test))]
        fn install_clock(&mut self, whole: u64, cleanup: u64, output: usize) -> CliResult<()> {
            if !(1..=MAX_WALL_MS).contains(&whole) {
                return Err(TransportError::new(
                    "clock_admission",
                    "finite original whole wall required",
                ));
            }
            self.deadline = self
                .started
                .checked_add(Duration::from_millis(whole))
                .ok_or_else(|| TransportError::new("clock_admission", "whole deadline overflow"))?;
            self.whole_wall_ms = Some(whole);
            // A valid whole wall survives invalid output/cleanup admission.
            if (1024..=MAX_OUTPUT_BYTES).contains(&output) {
                self.output_limit = output;
            }
            if cleanup == 0 || cleanup >= whole || !(1024..=MAX_OUTPUT_BYTES).contains(&output) {
                return Err(TransportError::new(
                    "clock_admission",
                    "positive cleanup below whole wall and 1KiB..=4MiB aggregate output required",
                ));
            }
            self.cleanup_reserve_ms = Some(cleanup);
            self.execution_deadline = self
                .started
                .checked_add(Duration::from_millis(whole - cleanup))
                .ok_or_else(|| {
                    TransportError::new("clock_admission", "execution deadline overflow")
                })?;
            check_time(self.execution_deadline, "admission_deadline")
        }
        #[cfg(any(feature = "onnx-cpu", test))]
        fn store_body(&mut self, bytes: Vec<u8>, kind: &'static str) {
            self.raw_body = Some(Arc::new(bytes));
            self.raw_body_kind = kind;
        }
        fn apply_admitted_request_clock(
            &mut self,
            checked: &enabled::CheckedReplayInputs,
        ) -> CliResult<()> {
            let resources = checked.resources();
            self.constrain_request_clock(RequestClockFacts {
                deadline: checked.deadline(),
                execution_deadline: checked.execution_deadline(),
                output_limit: checked.output_limit(),
                whole_wall_ms: resources.whole_wall_ms,
                cleanup_reserve_ms: resources.cleanup_reserve_ms,
                output_bytes: resources.output_bytes,
            })
        }
        fn constrain_request_clock(&mut self, request: RequestClockFacts) -> CliResult<()> {
            let transport = ClockDeclarationObservation {
                whole_wall_ms: self.whole_wall_ms,
                cleanup_reserve_ms: self.cleanup_reserve_ms,
                deadline_after_start_ms: clock_after_start_ms(self.started, self.deadline),
                execution_deadline_after_start_ms: clock_after_start_ms(
                    self.started,
                    self.execution_deadline,
                ),
                output_bytes: self.output_limit,
                output_limit: self.output_limit,
            };
            let declarations_differ = request.deadline != self.deadline
                || request.execution_deadline != self.execution_deadline
                || request.output_limit != self.output_limit
                || Some(request.whole_wall_ms) != self.whole_wall_ms
                || Some(request.cleanup_reserve_ms) != self.cleanup_reserve_ms
                || request.output_bytes != self.output_limit;
            // Admission success does not let a mismatched transport buy a larger
            // diagnostic allowance. Preserve both originals, constrain first, and
            // then refuse. These are the exact checked deadlines from the same S.
            self.deadline = self.deadline.min(request.deadline);
            self.execution_deadline = self
                .execution_deadline
                .min(request.execution_deadline)
                .min(self.deadline);
            self.output_limit = self
                .output_limit
                .min(request.output_limit)
                .min(MAX_OUTPUT_BYTES);
            self.whole_wall_ms = Some(clock_after_start_ms(self.started, self.deadline));
            // cleanup_reserve_ms remains the original transport declaration. No
            // synthetic cleanup is inferred from independently minimized W and E.
            self.admitted_input_clock = Some(AdmittedInputClockObservation {
                scope: "original_start_clock_limits_only",
                transport,
                request: ClockDeclarationObservation {
                    whole_wall_ms: Some(request.whole_wall_ms),
                    cleanup_reserve_ms: Some(request.cleanup_reserve_ms),
                    deadline_after_start_ms: clock_after_start_ms(self.started, request.deadline),
                    execution_deadline_after_start_ms: clock_after_start_ms(
                        self.started,
                        request.execution_deadline,
                    ),
                    output_bytes: request.output_bytes,
                    output_limit: request.output_limit,
                },
                effective: EffectiveClockObservation {
                    deadline_after_start_ms: clock_after_start_ms(self.started, self.deadline),
                    execution_deadline_after_start_ms: clock_after_start_ms(
                        self.started,
                        self.execution_deadline,
                    ),
                    output_limit: self.output_limit,
                },
                declarations_differ,
            });
            if declarations_differ {
                return Err(TransportError::new(
                    "clock_identity",
                    "frozen request and independent original clock/output declarations differ",
                ));
            }
            Ok(())
        }
        fn retain_input_error(&mut self, error: enabled::ReplayInputError) -> CliResult<()> {
            let old_whole = self.whole_wall_ms;
            let old_cleanup = self.cleanup_reserve_ms;
            let old_output = self.output_limit;
            let transport_execution_allowance_ms =
                self.execution_deadline
                    .saturating_duration_since(self.started)
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64;
            let mut request_execution_allowance_ms = None;
            let declarations_differ = !error.original_whole_deadline_retained
                || error.whole_wall_ms != old_whole
                || error.cleanup_reserve_ms != old_cleanup
                || error.output_limit != old_output;
            let mut invalid_retained_request_clock = false;
            self.output_limit = old_output.min(error.output_limit).min(MAX_OUTPUT_BYTES);
            if error.original_whole_deadline_retained {
                match error
                    .whole_wall_ms
                    .filter(|whole| (1..=MAX_WALL_MS).contains(whole))
                    .and_then(|whole| {
                        self.started
                            .checked_add(Duration::from_millis(whole))
                            .map(|deadline| (whole, deadline))
                    }) {
                    Some((whole, request_deadline)) => {
                        self.deadline = self.deadline.min(request_deadline);
                        self.whole_wall_ms = Some(old_whole.map_or(whole, |old| old.min(whole)));
                        if let Some(cleanup) = error
                            .cleanup_reserve_ms
                            .filter(|cleanup| *cleanup > 0 && *cleanup < whole)
                        {
                            request_execution_allowance_ms = Some(whole - cleanup);
                            if let Some(execution) = self
                                .started
                                .checked_add(Duration::from_millis(whole - cleanup))
                            {
                                self.execution_deadline = self.execution_deadline.min(execution);
                            }
                        }
                    }
                    None => {
                        invalid_retained_request_clock = true;
                        // An invalid retained-clock claim cannot buy the larger
                        // transport allowance or create a fresh diagnostic grace.
                        self.deadline = self.deadline.min(self.started);
                        self.whole_wall_ms = None;
                    }
                }
            }
            self.execution_deadline = self.execution_deadline.min(self.deadline);
            self.input_error_clock = Some(InputErrorClockObservation {
                original_whole_deadline_retained: error.original_whole_deadline_retained,
                transport_whole_wall_ms: old_whole,
                transport_cleanup_reserve_ms: old_cleanup,
                transport_output_limit: old_output,
                request_whole_wall_ms: error.whole_wall_ms,
                request_cleanup_reserve_ms: error.cleanup_reserve_ms,
                request_output_limit: error.output_limit,
                transport_execution_allowance_ms,
                request_execution_allowance_ms,
                effective_execution_allowance_ms: self
                    .execution_deadline
                    .saturating_duration_since(self.started)
                    .as_millis()
                    .min(u128::from(u64::MAX))
                    as u64,
                effective_whole_wall_ms: self.whole_wall_ms,
                effective_output_limit: self.output_limit,
                declarations_differ,
                invalid_retained_request_clock,
            });
            self.input_error = Some(Box::new(error));
            check_time(self.deadline, "input_error_serialization_deadline")?;
            let mut writer = BoundedJsonWriter {
                bytes: Vec::new(),
                deadline: self.deadline,
                limit: self.output_limit,
            };
            if let Err(error) = serde_json::to_writer(&mut writer, &self.input_error) {
                let projected = TransportError::new("input_error_serialization", &error);
                self.serialization_error = Some(error);
                return Err(projected);
            }
            check_time(self.deadline, "input_error_serialization_deadline")?;
            self.store_body(writer.bytes, "replay_input_error");
            Ok(())
        }
        fn evidence_for_writer(&mut self) -> SharedEvidence {
            if let Some(evidence) = self.shared_evidence.as_ref() {
                return Arc::clone(evidence);
            }
            let evidence = Arc::new(Mutex::new(EvidenceBundle {
                raw_body: self.raw_body.clone(),
                native_error: self.native_error.take(),
                input_error: self.input_error.take(),
                _delivery_errors: Arc::clone(&self.delivery_errors),
                serialization_error: self.serialization_error.take(),
            }));
            self.shared_evidence = Some(Arc::clone(&evidence));
            evidence
        }
        fn elapsed_ms(&self) -> u64 {
            self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
        }
        fn typed_native_error_retained(&self) -> bool {
            #[cfg(feature = "onnx-cpu")]
            {
                self.native_error.is_some()
                    || self.shared_evidence.as_ref().is_some_and(|bundle| {
                        bundle
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .native_error
                            .is_some()
                    })
            }
            #[cfg(not(feature = "onnx-cpu"))]
            {
                false
            }
        }
        fn typed_input_error_retained(&self) -> bool {
            #[cfg(feature = "onnx-cpu")]
            {
                self.input_error.is_some()
                    || self.shared_evidence.as_ref().is_some_and(|bundle| {
                        bundle
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .input_error
                            .is_some()
                    })
            }
            #[cfg(not(feature = "onnx-cpu"))]
            {
                false
            }
        }
        fn raw_body_retained(&self) -> bool {
            self.raw_body.is_some()
                || self.shared_evidence.as_ref().is_some_and(|bundle| {
                    bundle
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .raw_body
                        .is_some()
                })
        }
        fn serialization_error_retained(&self) -> bool {
            self.serialization_error.is_some()
                || self.shared_evidence.as_ref().is_some_and(|bundle| {
                    bundle
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .serialization_error
                        .is_some()
                })
        }
    }

    #[derive(Serialize)]
    struct PreparedHeader<'a> {
        schema: &'static str,
        scope: &'static str,
        expected_transport: &'a Option<ArtifactPin>,
        #[serde(skip_serializing_if = "Option::is_none")]
        expected_semantic_receipt_producer_scope: Option<SemanticReceiptProducerScope>,
        #[serde(skip_serializing_if = "Option::is_none")]
        requested_native_result: Option<NativeResultLane>,
        launch_artifact: &'a Option<ArtifactPin>,
        request_artifact: &'a Option<ArtifactPin>,
        replay_binary: &'a Option<ArtifactPin>,
        replay_binary_pin_scope: &'static str,
        runtime_library: &'a Option<ArtifactPin>,
        runtime_pin_scope: &'static str,
        body_artifact: ArtifactPin,
        body_kind: &'static str,
        typed_native_error_retained: bool,
        typed_input_error_retained: bool,
        whole_wall_ms: Option<u64>,
        cleanup_reserve_ms: Option<u64>,
        prepared_elapsed_ms: u64,
        execution_deadline_exceeded_at_preparation: bool,
        stdout_delivered: Option<bool>,
        process_exit_observed: bool,
        loaded_provider_image_observed_by_cli: bool,
        physical_closure_observed_by_cli: bool,
        input_error_clock: &'a Option<InputErrorClockObservation>,
        admitted_input_clock: &'a Option<AdmittedInputClockObservation>,
    }
    fn envelope(context: &Context) -> CliResult<Vec<u8>> {
        check_time(context.deadline, "output_preparation_deadline")?;
        let body = context
            .raw_body
            .as_ref()
            .ok_or_else(|| TransportError::new("output_body", "owned native body missing"))?;
        serde_json::from_slice::<serde::de::IgnoredAny>(body)
            .map_err(|e| TransportError::new("output_json", e))?;
        let header = PreparedHeader {
            schema: CLI_SCHEMA,
            scope: "cli_pins_and_bytes_prepared_before_delivery",
            expected_transport: &context.expected_transport,
            expected_semantic_receipt_producer_scope: context
                .expected_semantic_receipt_producer_scope,
            requested_native_result: context.requested_native_result,
            launch_artifact: &context.launch_artifact,
            request_artifact: &context.request_artifact,
            replay_binary: &context.replay_binary,
            replay_binary_pin_scope: binary_scope(),
            runtime_library: &context.runtime_library,
            runtime_pin_scope: "verified_cpu_library_bytes_not_loaded_provider_image",
            body_artifact: artifact(body),
            body_kind: context.raw_body_kind,
            typed_native_error_retained: context.typed_native_error_retained(),
            typed_input_error_retained: context.typed_input_error_retained(),
            whole_wall_ms: context.whole_wall_ms,
            cleanup_reserve_ms: context.cleanup_reserve_ms,
            prepared_elapsed_ms: context.elapsed_ms(),
            execution_deadline_exceeded_at_preparation: Instant::now()
                >= context.execution_deadline,
            stdout_delivered: None,
            process_exit_observed: false,
            loaded_provider_image_observed_by_cli: false,
            physical_closure_observed_by_cli: false,
            input_error_clock: &context.input_error_clock,
            admitted_input_clock: &context.admitted_input_clock,
        };
        let syntax_bytes = b"{\"cli\":,\"native\":}\n".len();
        let header_limit = context
            .output_limit
            .saturating_sub(body.len())
            .saturating_sub(syntax_bytes)
            .min(HEADER_RESERVE_BYTES);
        let header = bounded_json(&header, context.deadline, header_limit)
            .map_err(|e| TransportError::new("output_header", e))?;
        let overhead = header
            .len()
            .checked_add(syntax_bytes)
            .ok_or_else(|| TransportError::new("output_bound", "header byte overflow"))?;
        if overhead > HEADER_RESERVE_BYTES
            || body
                .len()
                .checked_add(overhead)
                .is_none_or(|size| size > context.output_limit)
        {
            return Err(TransportError::new(
                "output_bound",
                "original native body plus CLI header cannot fit unchanged aggregate output cap",
            ));
        }
        let total = body.len() + overhead;
        let mut output = Vec::new();
        output
            .try_reserve_exact(total)
            .map_err(|e| TransportError::new("output_allocation", e))?;
        output.extend_from_slice(b"{\"cli\":");
        output.extend_from_slice(&header);
        output.extend_from_slice(b",\"native\":");
        output.extend_from_slice(body);
        output.extend_from_slice(b"}\n");
        check_time(context.deadline, "output_preparation_deadline")?;
        Ok(output)
    }
    fn deliver(
        context: &mut Context,
        bytes: Vec<u8>,
        stdout: bool,
        budget: Arc<StreamBudget>,
    ) -> CliResult<()> {
        let progress = Arc::new(WriteProgress::new(bytes.len()));
        context.progress.push(Arc::clone(&progress));
        let file = native_output_file(stdout)?;
        let evidence = context.evidence_for_writer();
        bounded_write(
            file,
            Arc::new(bytes),
            context.deadline,
            budget,
            progress,
            Some(evidence),
        )
    }
    fn diagnose(context: &mut Context, error: TransportError, budget: Arc<StreamBudget>) {
        // Native/input typed errors and raw body stay owned in Context during this
        // attempt. If a richer body cannot fit, disclose omission, never fake work.
        context
            .delivery_errors
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .primary = Some(error);
        let maximum = DIAGNOSTIC_BYTES.min(budget.remaining());
        if maximum == 0 || Instant::now() >= context.deadline {
            return;
        }
        let body = context.raw_body.as_ref().map(|bytes| artifact(bytes));
        let writes: Vec<_> = context
            .progress
            .iter()
            .map(|progress| progress.observe())
            .collect();
        let errors = context
            .delivery_errors
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(error) = errors.primary.as_ref() else {
            return;
        };
        let mut value = serde_json::json!({"schema":CLI_SCHEMA,"error":error,
        "elapsed_ms":context.elapsed_ms(),"deadline_exceeded":Instant::now()>=context.deadline,
        "native_dispatch_entered":context.native_dispatch_entered,"native_work":"unobserved_or_in_owned_native_body",
        "body_artifact":body,"body_kind":context.raw_body_kind,"raw_body_retained_until_delivery_attempt_ends":context.raw_body_retained(),
        "bytes_charged":budget.charged.load(Ordering::Acquire),"writes":writes,
        "typed_native_error_retained":context.typed_native_error_retained(),
        "typed_input_error_retained":context.typed_input_error_retained(),
        "typed_io_terminal_retained":error.terminal.is_some(),
        "input_error_clock":context.input_error_clock,
        "admitted_input_clock":context.admitted_input_clock,
        "serialization_error_retained":context.serialization_error_retained(),
        "secondary_delivery_error_retained":errors.secondary.is_some(),
        "original_native_body_omitted":context.raw_body.is_some(),"physical_closure_observed_by_cli":false});
        if let Some(scope) = context.expected_semantic_receipt_producer_scope {
            value["expected_semantic_receipt_producer_scope"] = serde_json::json!(scope);
        }
        if let Some(lane) = context.requested_native_result {
            value["requested_native_result"] = serde_json::json!(lane);
        }
        let prepared = bounded_json(&value, context.deadline, maximum.saturating_sub(1));
        let mut bytes = match prepared {
            Ok(bytes) => bytes,
            Err(_) => {
                value = serde_json::json!({"schema":CLI_SCHEMA,"code":"frozen_replay_cli_failed",
            "failure_stage":error.stage,
            "input_error_clock":context.input_error_clock,
            "admitted_input_clock":context.admitted_input_clock,
            "elapsed_ms":context.elapsed_ms(),"native_dispatch_entered":context.native_dispatch_entered,
            "body_artifact":body,"original_native_body_omitted":context.raw_body.is_some(),
            "bytes_charged":budget.charged.load(Ordering::Acquire),"work":"unknown_not_zero"});
                if let Some(scope) = context.expected_semantic_receipt_producer_scope {
                    value["expected_semantic_receipt_producer_scope"] = serde_json::json!(scope);
                }
                if let Some(lane) = context.requested_native_result {
                    value["requested_native_result"] = serde_json::json!(lane);
                }
                let Ok(smaller) = bounded_json(&value, context.deadline, maximum.saturating_sub(1))
                else {
                    return;
                };
                smaller
            }
        };
        if bytes.is_empty() || bytes.len() + 1 > maximum {
            return;
        }
        bytes.push(b'\n');
        drop(errors);
        if let Err(error) = deliver(context, bytes, false, budget) {
            context
                .delivery_errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .secondary = Some(error);
        }
    }

    #[cfg(feature = "onnx-cpu")]
    mod enabled {
        use super::*;
        use rz_eval::pals_onnx::PalsOnnxConfig;
        use rz_eval::runtime_pin::RuntimeCache;
        pub(super) use rz_uci::pals_cpu_task::strategic_action::native_replay::NativeReplayError;
        use rz_uci::pals_cpu_task::strategic_action::native_replay::{
            self, CpuFreshAssetProfile, CpuFreshReplayAssets,
        };
        use rz_uci::pals_cpu_task::strategic_action::replay_inputs;
        pub(super) use rz_uci::pals_cpu_task::strategic_action::replay_inputs::CheckedReplayInputs;
        pub(super) use rz_uci::pals_cpu_task::strategic_action::replay_inputs::ReplayInputError;
        use rz_uci::pals_cpu_task::strategic_action::replay_launch_preparation::{
            ExpectedPinsWire, ExpectedTransportV2, ExpectedTransportV3, LaunchAssets,
        };
        use serde::Deserialize;

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ExpectedTransport {
            schema: String,
            request_artifact: ArtifactPin,
            replay_expected: ExpectedPinsWire,
            launch_asset_artifact: ArtifactPin,
            cpu_fresh_profile_artifact: ArtifactPin,
            cpu_fresh_profile: CpuFreshAssetProfile,
            whole_wall_ms: u64,
            cleanup_reserve_ms: u64,
            output_bytes: usize,
            replay_binary_pin_scope: String,
        }
        struct ParsedExpectedTransport {
            transport: ExpectedTransport,
            expected_scope: Option<SemanticReceiptProducerScope>,
            native_result: Option<NativeResultLane>,
        }
        impl ParsedExpectedTransport {
            fn from_v2(
                version: ExpectedTransportV2,
                native_result: Option<NativeResultLane>,
            ) -> Self {
                Self {
                    expected_scope: Some(version.expected_semantic_receipt_producer_scope),
                    native_result,
                    transport: ExpectedTransport {
                        schema: version.schema,
                        request_artifact: version.request_artifact,
                        replay_expected: version.replay_expected,
                        launch_asset_artifact: version.launch_asset_artifact,
                        cpu_fresh_profile_artifact: version.cpu_fresh_profile_artifact,
                        cpu_fresh_profile: version.cpu_fresh_profile,
                        whole_wall_ms: version.whole_wall_ms,
                        cleanup_reserve_ms: version.cleanup_reserve_ms,
                        output_bytes: version.output_bytes,
                        replay_binary_pin_scope: version.replay_binary_pin_scope,
                    },
                }
            }
        }
        // Only these bounded original-clock fields are projected before version
        // admission. Ignored fields grant no scope/identity/native authority; the
        // final version-specific DTO remains closed and rejects duplicate fields.
        #[derive(Deserialize)]
        struct ExpectedClockProjection {
            whole_wall_ms: u64,
            cleanup_reserve_ms: u64,
            output_bytes: usize,
        }
        #[derive(Deserialize)]
        struct ExpectedSchemaProjection {
            schema: String,
        }
        fn parse_expected_transport(
            raw: &[u8],
            context: &mut Context,
        ) -> CliResult<ParsedExpectedTransport> {
            let clock: ExpectedClockProjection = serde_json::from_slice(raw)
                .map_err(|error| TransportError::new("expected_clock_json", error))?;
            context.install_clock(
                clock.whole_wall_ms,
                clock.cleanup_reserve_ms,
                clock.output_bytes,
            )?;
            // A schema/scope refusal keeps the known original S/W/E/cap. No
            // partially parsed or malformed JSON is used to invent a new clock.
            let schema: ExpectedSchemaProjection = serde_json::from_slice(raw)
                .map_err(|error| TransportError::new("expected_schema_json", error))?;
            let parsed = match schema.schema.as_str() {
                EXPECTED_SCHEMA => ParsedExpectedTransport {
                    transport: serde_json::from_slice(raw)
                        .map_err(|error| TransportError::new("expected_json", error))?,
                    expected_scope: None,
                    native_result: None,
                },
                EXPECTED_SCHEMA_V2 => {
                    let version: ExpectedTransportV2 = serde_json::from_slice(raw)
                        .map_err(|error| TransportError::new("expected_json", error))?;
                    ParsedExpectedTransport::from_v2(version, None)
                }
                EXPECTED_SCHEMA_V3 => {
                    let version: ExpectedTransportV3 = serde_json::from_slice(raw)
                        .map_err(|error| TransportError::new("expected_json", error))?;
                    if !version.same_original_clock()
                        || version.expectations.schema != EXPECTED_SCHEMA_V2
                    {
                        return Err(TransportError::new(
                            "expected_result_lane",
                            "observed /3 requires nested /2 expectations and the same original clock",
                        ));
                    }
                    ParsedExpectedTransport::from_v2(
                        version.expectations,
                        Some(version.native_result),
                    )
                }
                _ => {
                    return Err(TransportError::new(
                        "expected_schema",
                        "closed expected transport V1, V2 or explicit V3 required",
                    ));
                }
            };
            finish_expected_transport_admission(parsed, &clock, context)
        }
        fn finish_expected_transport_admission(
            parsed: ParsedExpectedTransport,
            clock: &ExpectedClockProjection,
            context: &mut Context,
        ) -> CliResult<ParsedExpectedTransport> {
            // Closed decoding established this expectation declaration only. Keep
            // it even when the later original-E guard refuses admission; this is
            // not a scope comparison or historical loaded-image success witness.
            context.expected_semantic_receipt_producer_scope = parsed.expected_scope;
            context.requested_native_result = parsed.native_result;
            if parsed.transport.whole_wall_ms != clock.whole_wall_ms
                || parsed.transport.cleanup_reserve_ms != clock.cleanup_reserve_ms
                || parsed.transport.output_bytes != clock.output_bytes
            {
                return Err(TransportError::new(
                    "expected_clock_projection",
                    "closed transport differs from its original budget-only projection",
                ));
            }
            check_time(context.execution_deadline, "expected_admission_deadline")?;
            Ok(parsed)
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum SemanticScopeRoute {
            LibraryOnlyV1,
            Explicit(SemanticReceiptProducerScope),
        }
        fn semantic_scope_route(
            expected: Option<SemanticReceiptProducerScope>,
        ) -> SemanticScopeRoute {
            // Routing an expectation creates neither a checked replay nor an
            // observation of the receipt producer's historical loaded image.
            match expected {
                None => SemanticScopeRoute::LibraryOnlyV1,
                Some(scope) => SemanticScopeRoute::Explicit(scope),
            }
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum NativeDispatchRoute {
            LegacyLibrary,
            LegacyScoped(SemanticReceiptProducerScope),
            ObservedPrior(SemanticReceiptProducerScope),
        }
        fn native_dispatch_route(
            scope: SemanticScopeRoute,
            result: Option<NativeResultLane>,
            mode: replay_inputs::ReplayInputMode,
        ) -> CliResult<NativeDispatchRoute> {
            match (scope, result) {
                (SemanticScopeRoute::LibraryOnlyV1, None) => Ok(NativeDispatchRoute::LegacyLibrary),
                (SemanticScopeRoute::Explicit(scope), None) => {
                    Ok(NativeDispatchRoute::LegacyScoped(scope))
                }
                (SemanticScopeRoute::Explicit(scope), Some(NativeResultLane::QueryPriorV1))
                    if mode == replay_inputs::ReplayInputMode::RepairOpponent4n =>
                {
                    Ok(NativeDispatchRoute::ObservedPrior(scope))
                }
                _ => Err(TransportError::new(
                    "native_result_mode",
                    "query prior requires explicit /3 expectations and registered 4N input",
                )),
            }
        }
        fn validate_runtime_observation(
            bundle: Option<[u8; 32]>,
            actual: &ArtifactPin,
            independently_expected: &ArtifactPin,
        ) -> CliResult<()> {
            if bundle.is_some() || actual != independently_expected {
                return Err(TransportError::new(
                    "runtime_identity",
                    "verified CPU library size/digest differs or a CUDA bundle was supplied",
                ));
            }
            Ok(())
        }
        fn check_binary_domains(
            replay: &ArtifactPin,
            legacy: &ArtifactPin,
            semantic_sha256: &str,
        ) -> CliResult<()> {
            if replay.sha256 == legacy.sha256 || replay.sha256 == semantic_sha256 {
                return Err(TransportError::new(
                    "binary_domain_alias",
                    "replay image SHA must differ from both historical legacy and semantic SHA; bytes-only differences do not separate domains",
                ));
            }
            Ok(())
        }
        pub(super) fn run(context: &mut Context) -> CliResult<bool> {
            // A ninth value is enough to reject an excess option without collecting
            // all OS argv. No flag or path is inferred from the frozen stdin body.
            let args = arguments(&std::env::args_os().skip(1).take(9).collect::<Vec<_>>())?;
            let cancel = AtomicBool::new(false);
            // This one-shot CLI has no signal/control-channel mapping to the flag.
            // It still uses the same flag at preparation boundaries and in dispatch.
            let expected_raw = read_verified(
                &args.expected_path,
                &args.expected_pin,
                MAX_EXPECTED_BYTES,
                context.execution_deadline,
                &cancel,
            )?;
            context.expected_transport = Some(artifact(&expected_raw));
            let parsed = parse_expected_transport(&expected_raw, context)?;
            let scope_route = semantic_scope_route(parsed.expected_scope);
            let native_result = parsed.native_result;
            let transport = parsed.transport;
            if !matches!(
                transport.schema.as_str(),
                EXPECTED_SCHEMA | EXPECTED_SCHEMA_V2
            ) || transport.replay_binary_pin_scope != binary_scope()
                || transport.replay_expected.provider_factory_id
                    != "rz-uci-native-cpu-fresh-invocation-v1"
            {
                return Err(TransportError::new(
                    "expected_admission",
                    "independent transport schema, actual OS image scope or factory ID differs",
                ));
            }
            check_pin(&transport.request_artifact, MAX_REQUEST_BYTES as u64)?;
            check_pin(&transport.launch_asset_artifact, MAX_LAUNCH_BYTES as u64)?;
            check_pin(
                &transport.cpu_fresh_profile_artifact,
                MAX_LAUNCH_BYTES as u64,
            )?;
            let expected = transport.replay_expected.into_expected();
            // This is domain separation only. It creates no historical loaded-image
            // or invocation provenance, and runs before any cache/native dispatch.
            check_binary_domains(
                &expected.registered_artifacts.replay_binary,
                &expected.registered_artifacts.legacy_cpu_binary,
                &expected.semantic_binary_sha256,
            )?;
            let request = read_stdin(context.execution_deadline, &cancel)?;
            check_bytes(
                &request,
                &transport.request_artifact,
                MAX_REQUEST_BYTES as u64,
            )?;
            check_execution(
                context.execution_deadline,
                &cancel,
                "request_admission_deadline",
            )?;
            context.request_artifact = Some(artifact(&request));
            let input_result = match scope_route {
                SemanticScopeRoute::LibraryOnlyV1 => {
                    replay_inputs::check_replay_inputs(&request, &expected, context.started)
                }
                SemanticScopeRoute::Explicit(scope) => {
                    replay_inputs::check_replay_inputs_with_semantic_scope(
                        &request,
                        &expected,
                        scope,
                        context.started,
                    )
                }
            };
            let checked = match input_result {
                Ok(value) => value,
                Err(error) => {
                    // Keep original typed cause/source diagnostics. No native work
                    // count is invented for an input-admission refusal.
                    context.retain_input_error(error)?;
                    return Ok(false);
                }
            };
            context.apply_admitted_request_clock(&checked)?;
            // The selected output lane is checked against the independently
            // admitted input before any asset/cache/native work can begin.
            let dispatch_route = native_dispatch_route(scope_route, native_result, checked.mode())?;
            drop(checked); // No model/provider/completion authority is retained here.
            let launch_raw = read_verified(
                &args.launch_path,
                &transport.launch_asset_artifact,
                MAX_LAUNCH_BYTES,
                context.execution_deadline,
                &cancel,
            )?;
            context.launch_artifact = Some(artifact(&launch_raw));
            let launch: LaunchAssets = serde_json::from_slice(&launch_raw)
                .map_err(|e| TransportError::new("launch_json", e))?;
            if launch.schema != LAUNCH_SCHEMA {
                return Err(TransportError::new(
                    "launch_admission",
                    "explicit launch schema required",
                ));
            }
            let export = absolute_path(&launch.export_manifest_path)?;
            let runtime_source = absolute_path(&launch.runtime_library_path)?;
            let cache_root = absolute_path(&launch.runtime_cache_root)?;
            let profile = &transport.cpu_fresh_profile;
            if profile.schema != "rz-pals-cpu-fresh-replay-assets/1"
                || profile.domain != "cpu_fresh"
                || profile.provider != "cpu"
                || !(1..=2).contains(&profile.intra_threads)
                || profile.device_public_memory
            {
                return Err(TransportError::new(
                    "profile_admission",
                    "independent explicit CPU Fresh profile/thread/host-memory domain required",
                ));
            }
            check_bytes(
                launch.cpu_fresh_profile_raw.as_bytes(),
                &transport.cpu_fresh_profile_artifact,
                MAX_LAUNCH_BYTES as u64,
            )?;
            check_execution(
                context.execution_deadline,
                &cancel,
                "profile_admission_deadline",
            )?;
            check_pin(&profile.runtime_library, MAX_RUNTIME_BYTES)?;
            // Manifest preparation is actual pinned metadata work in the same E.
            let manifest = read_verified(
                &export,
                &profile.export_manifest,
                MAX_MANIFEST_BYTES,
                context.execution_deadline,
                &cancel,
            )?;
            drop(manifest);
            context.replay_binary = Some(own_binary(
                &expected.registered_artifacts.replay_binary,
                context.execution_deadline,
                &cancel,
            )?);
            check_execution(
                context.execution_deadline,
                &cancel,
                "runtime_cache_deadline",
            )?;
            let cache = RuntimeCache::open(&cache_root)
                .map_err(|e| TransportError::new("runtime_cache", e))?;
            check_execution(
                context.execution_deadline,
                &cancel,
                "runtime_cache_deadline",
            )?;
            let runtime = cache
                .library(&runtime_source, &profile.runtime_library.sha256)
                .map_err(|e| TransportError::new("runtime_pin", e))?;
            check_execution(context.execution_deadline, &cancel, "runtime_pin_deadline")?;
            let actual_runtime = ArtifactPin {
                bytes: runtime.storage().bytes,
                sha256: hex_digest(&runtime.binary_digest()),
            };
            validate_runtime_observation(
                runtime.bundle_digest(),
                &actual_runtime,
                &profile.runtime_library,
            )?;
            context.runtime_library = Some(actual_runtime);
            let mut config = PalsOnnxConfig::cpu();
            config.intra_threads = profile.intra_threads;
            config.cache_public_memory = profile.cache_public_memory;
            config.device_public_memory = false;
            check_execution(
                context.execution_deadline,
                &cancel,
                "native_dispatch_deadline",
            )?;
            context.native_dispatch_entered = true;
            let assets = CpuFreshReplayAssets {
                export: &export,
                runtime_pin: &runtime,
                config,
                profile_raw: launch.cpu_fresh_profile_raw.as_bytes(),
                expected_profile_artifact: &transport.cpu_fresh_profile_artifact,
                expected_profile: profile,
            };
            let dispatched = match dispatch_route {
                NativeDispatchRoute::LegacyLibrary => native_replay::dispatch_started(
                    &request,
                    &expected,
                    assets,
                    context.started,
                    &cancel,
                ),
                NativeDispatchRoute::LegacyScoped(scope) => {
                    native_replay::dispatch_started_with_semantic_scope(
                        &request,
                        &expected,
                        scope,
                        assets,
                        context.started,
                        &cancel,
                    )
                }
                NativeDispatchRoute::ObservedPrior(scope) => {
                    native_replay::dispatch_started_observed_with_semantic_scope(
                        &request,
                        &expected,
                        scope,
                        assets,
                        context.started,
                        &cancel,
                    )
                    .map(native_replay::NativeReplayResult::into_bytes)
                }
            };
            match dispatched {
                Ok(bytes) => {
                    context.store_body(bytes, "native_returned_ok");
                    Ok(true)
                }
                Err(error) => {
                    context.native_error = Some(Box::new(error));
                    check_time(context.deadline, "native_error_serialization_deadline")?;
                    let serialized = context
                        .native_error
                        .as_ref()
                        .ok_or_else(|| {
                            TransportError::new(
                                "native_error_retention",
                                "typed native error missing",
                            )
                        })?
                        .serialized();
                    match serialized {
                        Ok(bytes) => {
                            context.store_body(bytes, "native_returned_error");
                            Ok(false)
                        }
                        Err(fault) => Err(TransportError::new("native_error_serialization", fault)),
                    }
                }
            }
        }
        #[cfg(test)]
        mod tests {
            use super::*;
            fn expected_wire(schema: &str, scope_members: &str) -> Vec<u8> {
                // A codec-only declaration fixture, not independently registered
                // history, a checked request, a runtime or a callable provider.
                let h = "c".repeat(64);
                let pin = format!(r#"{{"bytes":17,"sha256":"{h}"}}"#);
                let image_scope = binary_scope();
                format!(r#"{{
                    "schema":"{schema}",
                    "request_artifact":{pin},
                    "replay_expected":{{
                        "registration_artifact":{pin},
                        "prepared_action_artifact":{pin},
                        "semantic_receipt_artifact":{pin},
                        "parent":{{"parent_input_sha256":"{h}","current_view_sha256":"{h}","frozen_admission_sha256":"{h}","encoding_sha256":"{h}"}},
                        "binding":{{
                            "query_sha256":"{h}","catalogue_artifact":{pin},"before_result_artifact":{pin},
                            "prior_ledger_sha256":"{h}","semantic_input_sha256":"{h}","semantic_context_sha256":"{h}",
                            "semantic_branch_meaning_sha256":"{h}","semantic_before_result_anchor_sha256":"{h}","cpu_request_artifact":{pin}
                        }},
                        "registered_artifacts":{{"legacy_cpu_binary":{pin},"replay_binary":{pin},"engine_source":{pin},"wrapper_source":{pin},"replay_source":{pin},"provider_factory_source":{pin}}},
                        "legacy_cpu_profile_sha256":"{h}",
                        "provider_factory_id":"rz-uci-native-cpu-fresh-invocation-v1",
                        "semantic_binary_sha256":"{h}"
                    }},
                    "launch_asset_artifact":{pin},
                    "cpu_fresh_profile_artifact":{pin},
                    "cpu_fresh_profile":{{
                        "schema":"rz-pals-cpu-fresh-replay-assets/1","domain":"cpu_fresh","provider":"cpu",
                        "export_manifest":{pin},"runtime_library":{pin},"model_epoch":"codec-only",
                        "encoding_semantic_sha256":"{h}","intra_threads":1,"cache_public_memory":false,
                        "device_public_memory":false,"context_sha256":"{h}"
                    }},
                    "whole_wall_ms":1000,"cleanup_reserve_ms":100,"output_bytes":1024,
                    "replay_binary_pin_scope":"{image_scope}"{scope_members}
                }}"#).into_bytes()
            }
            fn scope_member(literal: &str) -> String {
                format!(r#", "expected_semantic_receipt_producer_scope":"{literal}""#)
            }
            fn observed_expected() -> ExpectedTransportV3 {
                let mut v2: ExpectedTransportV2 = serde_json::from_slice(&expected_wire(
                    EXPECTED_SCHEMA_V2,
                    &scope_member("dispatcher_compared_verified_argument"),
                ))
                .unwrap();
                // This controlled declaration explicitly reserves header space;
                // no production cap is increased after a serialization failure.
                v2.output_bytes = 8192;
                ExpectedTransportV3::new(v2, NativeResultLane::QueryPriorV1)
            }
            #[test]
            fn explicit_v3_selects_prior_dispatch_and_preserves_original_clock_declaration_only() {
                let raw = serde_json::to_vec(&observed_expected()).unwrap();
                let started = Instant::now();
                let mut context = Context::new(started).unwrap();
                let parsed = parse_expected_transport(&raw, &mut context).unwrap();
                assert_eq!(parsed.transport.schema, EXPECTED_SCHEMA_V2);
                assert_eq!(parsed.native_result, Some(NativeResultLane::QueryPriorV1));
                assert_eq!(context.started, started);
                assert_eq!(context.deadline, started + Duration::from_millis(1000));
                assert_eq!(
                    context.execution_deadline,
                    started + Duration::from_millis(900)
                );
                assert_eq!(context.output_limit, 8192);
                assert_eq!(
                    native_dispatch_route(
                        semantic_scope_route(parsed.expected_scope),
                        parsed.native_result,
                        replay_inputs::ReplayInputMode::RepairOpponent4n
                    )
                    .unwrap(),
                    NativeDispatchRoute::ObservedPrior(
                        SemanticReceiptProducerScope::LibraryDispatcherArgument
                    )
                );
                context.store_body(b"{\"codec_fixture\":true}".to_vec(), "declaration_only");
                let envelope: serde_json::Value =
                    serde_json::from_slice(&envelope(&context).unwrap()).unwrap();
                assert_eq!(envelope["cli"]["requested_native_result"], "query_prior_v1");
                assert_eq!(envelope["cli"]["process_exit_observed"], false);
                assert_eq!(envelope["cli"]["physical_closure_observed_by_cli"], false);
                assert!(!context.native_dispatch_entered);
                assert!(context.runtime_library.is_none() && context.replay_binary.is_none());
            }
            #[test]
            fn observed_result_lane_rejects_mode_or_scope_substitution_before_native_loading() {
                let scope = SemanticReceiptProducerScope::LibraryDispatcherArgument;
                for mode in [
                    replay_inputs::ReplayInputMode::ReplyOnly2n,
                    replay_inputs::ReplayInputMode::RepairEndpoint3n,
                    replay_inputs::ReplayInputMode::RepairOpponent4n,
                ] {
                    assert_eq!(
                        native_dispatch_route(SemanticScopeRoute::LibraryOnlyV1, None, mode)
                            .unwrap(),
                        NativeDispatchRoute::LegacyLibrary
                    );
                    assert_eq!(
                        native_dispatch_route(SemanticScopeRoute::Explicit(scope), None, mode)
                            .unwrap(),
                        NativeDispatchRoute::LegacyScoped(scope)
                    );
                    assert!(
                        native_dispatch_route(
                            SemanticScopeRoute::LibraryOnlyV1,
                            Some(NativeResultLane::QueryPriorV1),
                            mode
                        )
                        .is_err()
                    );
                    if mode != replay_inputs::ReplayInputMode::RepairOpponent4n {
                        assert!(
                            native_dispatch_route(
                                SemanticScopeRoute::Explicit(scope),
                                Some(NativeResultLane::QueryPriorV1),
                                mode
                            )
                            .is_err()
                        );
                    }
                }
                for (schema, member) in [
                    (EXPECTED_SCHEMA, "".to_string()),
                    (
                        EXPECTED_SCHEMA_V2,
                        scope_member("dispatcher_compared_verified_argument"),
                    ),
                ] {
                    let member = member + r#", "native_result":"query_prior_v1""#;
                    let started = Instant::now();
                    let mut context = Context::new(started).unwrap();
                    assert!(
                        parse_expected_transport(&expected_wire(schema, &member), &mut context)
                            .is_err()
                    );
                    assert!(context.requested_native_result.is_none());
                    assert!(!context.native_dispatch_entered);
                }
            }
            #[test]
            fn v3_closed_codec_rejects_clock_version_lane_and_extra_field_mutations() {
                let original = serde_json::to_value(observed_expected()).unwrap();
                let mut variants = Vec::new();
                for field in ["whole_wall_ms", "cleanup_reserve_ms", "output_bytes"] {
                    let mut v = original.clone();
                    v["expectations"][field] =
                        serde_json::json!(v["expectations"][field].as_u64().unwrap() + 1);
                    variants.push(serde_json::to_vec(&v).unwrap());
                }
                for (field, replacement) in [
                    ("schema", serde_json::json!(EXPECTED_SCHEMA)),
                    ("native_result", serde_json::json!("unknown")),
                    ("unregistered_authority", serde_json::json!(true)),
                ] {
                    let mut v = original.clone();
                    v[field] = replacement;
                    variants.push(serde_json::to_vec(&v).unwrap());
                }
                let mut missing = original.clone();
                missing.as_object_mut().unwrap().remove("native_result");
                variants.push(serde_json::to_vec(&missing).unwrap());
                let mut nested = original.clone();
                nested["expectations"]["schema"] = serde_json::json!(EXPECTED_SCHEMA);
                variants.push(serde_json::to_vec(&nested).unwrap());
                let mut nested_extra = original.clone();
                nested_extra["expectations"]["native_result"] = serde_json::json!("query_prior_v1");
                variants.push(serde_json::to_vec(&nested_extra).unwrap());
                let raw = serde_json::to_string(&original).unwrap();
                variants.push(
                    raw.replacen(
                        r#""native_result":"query_prior_v1""#,
                        r#""native_result":"query_prior_v1","native_result":"query_prior_v1""#,
                        1,
                    )
                    .into_bytes(),
                );
                for raw in variants {
                    let started = Instant::now();
                    let mut context = Context::new(started).unwrap();
                    assert!(parse_expected_transport(&raw, &mut context).is_err());
                    assert_eq!(context.started, started);
                    assert_eq!(context.deadline, started + Duration::from_millis(1000));
                    assert_eq!(
                        context.execution_deadline,
                        started + Duration::from_millis(900)
                    );
                    assert_eq!(context.output_limit, 8192);
                    assert!(!context.native_dispatch_entered);
                    assert!(context.runtime_library.is_none() && context.replay_binary.is_none());
                }
            }
            #[test]
            fn expected_v1_codec_retains_wire_and_rejects_any_scope_field() {
                let mut context = Context::new(Instant::now()).unwrap();
                let parsed =
                    parse_expected_transport(&expected_wire(EXPECTED_SCHEMA, ""), &mut context)
                        .unwrap();
                assert_eq!(parsed.transport.schema, EXPECTED_SCHEMA);
                assert_eq!(parsed.expected_scope, None);
                for literal in [
                    "dispatcher_compared_verified_argument",
                    "linux_loaded_executable_inode",
                    "current_exe_path_hash",
                ] {
                    let mut context = Context::new(Instant::now()).unwrap();
                    assert!(
                        parse_expected_transport(
                            &expected_wire(EXPECTED_SCHEMA, &scope_member(literal)),
                            &mut context
                        )
                        .is_err()
                    );
                }
                for member in [
                    r#", "expected_semantic_receipt_producer_scope":null"#,
                    r#", "expected_semantic_receipt_producer_scope":false"#,
                ] {
                    let mut context = Context::new(Instant::now()).unwrap();
                    assert!(
                        parse_expected_transport(
                            &expected_wire(EXPECTED_SCHEMA, member),
                            &mut context
                        )
                        .is_err()
                    );
                }
            }
            #[test]
            fn expected_v2_codec_requires_one_closed_semantic_scope() {
                for (literal, scope) in [
                    (
                        "dispatcher_compared_verified_argument",
                        SemanticReceiptProducerScope::LibraryDispatcherArgument,
                    ),
                    (
                        "linux_loaded_executable_inode",
                        SemanticReceiptProducerScope::LinuxLoadedExecutableInode,
                    ),
                    (
                        "current_exe_path_hash",
                        SemanticReceiptProducerScope::CurrentExePathHash,
                    ),
                ] {
                    let mut context = Context::new(Instant::now()).unwrap();
                    let parsed = parse_expected_transport(
                        &expected_wire(EXPECTED_SCHEMA_V2, &scope_member(literal)),
                        &mut context,
                    )
                    .unwrap();
                    assert_eq!(parsed.transport.schema, EXPECTED_SCHEMA_V2);
                    assert_eq!(parsed.expected_scope, Some(scope));
                    assert!(!context.native_dispatch_entered);
                }
                for member in [
                    "",
                    r#", "expected_semantic_receipt_producer_scope":null"#,
                    r#", "expected_semantic_receipt_producer_scope":true"#,
                    r#", "expected_semantic_receipt_producer_scope":{"linux_loaded_executable_inode":null}"#,
                    r#", "expected_semantic_receipt_producer_scope":"unknown""#,
                    r#", "expected_semantic_receipt_producer_scope":"linux_loaded_executable_inode", "expected_semantic_receipt_producer_scope":"linux_loaded_executable_inode""#,
                    r#", "expected_semantic_receipt_producer_scope":"current_exe_path_hash", "past_image_observed":true"#,
                ] {
                    let mut context = Context::new(Instant::now()).unwrap();
                    assert!(
                        parse_expected_transport(
                            &expected_wire(EXPECTED_SCHEMA_V2, member),
                            &mut context
                        )
                        .is_err()
                    );
                    assert!(context.expected_semantic_receipt_producer_scope.is_none());
                }
            }
            #[test]
            fn closed_v2_scope_survives_final_execution_deadline_with_original_whole_window() {
                let raw = String::from_utf8(expected_wire(
                    EXPECTED_SCHEMA_V2,
                    &scope_member("linux_loaded_executable_inode"),
                ))
                .unwrap()
                .replacen(r#""whole_wall_ms":1000"#, r#""whole_wall_ms":5000"#, 1)
                .replacen(
                    r#""cleanup_reserve_ms":100"#,
                    r#""cleanup_reserve_ms":4000"#,
                    1,
                )
                .replacen(r#""output_bytes":1024"#, r#""output_bytes":8192"#, 1)
                .into_bytes();
                let mut codec_context = Context::new(Instant::now()).unwrap();
                let parsed = parse_expected_transport(&raw, &mut codec_context).unwrap();
                let clock: ExpectedClockProjection = serde_json::from_slice(&raw).unwrap();
                // Reproduce work consuming the declared E before the actual final
                // guard, while W still has three seconds. Only private clock facts
                // are seeded; there is no checked replay, native owner or NN call.
                let started = Instant::now().checked_sub(Duration::from_secs(2)).unwrap();
                let mut context = Context::new(started).unwrap();
                context.deadline = started + Duration::from_millis(clock.whole_wall_ms);
                context.execution_deadline =
                    started + Duration::from_millis(clock.whole_wall_ms - clock.cleanup_reserve_ms);
                context.whole_wall_ms = Some(clock.whole_wall_ms);
                context.cleanup_reserve_ms = Some(clock.cleanup_reserve_ms);
                context.output_limit = clock.output_bytes;
                let original = (
                    context.started,
                    context.deadline,
                    context.execution_deadline,
                    context.output_limit,
                );
                assert!(context.expected_semantic_receipt_producer_scope.is_none());
                assert!(Instant::now() >= context.execution_deadline);
                check_time(context.deadline, "fixture_original_whole_window").unwrap();
                let error = match finish_expected_transport_admission(parsed, &clock, &mut context)
                {
                    Err(error) => error,
                    Ok(_) => panic!("elapsed final admission unexpectedly succeeded"),
                };
                assert_eq!(error.stage, "expected_admission_deadline");
                assert_eq!(
                    context.expected_semantic_receipt_producer_scope,
                    Some(SemanticReceiptProducerScope::LinuxLoadedExecutableInode)
                );
                assert_eq!(
                    (
                        context.started,
                        context.deadline,
                        context.execution_deadline,
                        context.output_limit
                    ),
                    original
                );
                assert_eq!(context.whole_wall_ms, Some(5000));
                assert_eq!(context.cleanup_reserve_ms, Some(4000));
                assert!(!context.native_dispatch_entered);
                assert!(context.native_error.is_none());
                assert!(context.input_error.is_none());
                assert!(context.replay_binary.is_none());
                assert!(context.runtime_library.is_none());
                // This token tests the existing byte-preparation header only, not
                // a native receipt or actual diagnostic delivery/physical closure.
                context.store_body(
                    b"{\"codec_fixture\":true}".to_vec(),
                    "codec_declaration_only",
                );
                let prepared = envelope(&context).unwrap();
                let value: serde_json::Value = serde_json::from_slice(&prepared).unwrap();
                assert_eq!(
                    value["cli"]["expected_semantic_receipt_producer_scope"],
                    "linux_loaded_executable_inode"
                );
                assert_eq!(
                    value["cli"]["execution_deadline_exceeded_at_preparation"],
                    true
                );
                assert_eq!(value["cli"]["loaded_provider_image_observed_by_cli"], false);
                assert_eq!(value["cli"]["physical_closure_observed_by_cli"], false);
                assert_eq!(value["cli"]["process_exit_observed"], false);
                assert!(value["cli"]["stdout_delivered"].is_null());
            }
            #[test]
            fn schema_and_scope_refusals_keep_original_projected_clock_and_cap() {
                let duplicate_schema = String::from_utf8(expected_wire(EXPECTED_SCHEMA_V2, "")).unwrap()
                    .replacen(r#""schema":"rz-pals-frozen-replay-cli-expected/2""#,
                        r#""schema":"rz-pals-frozen-replay-cli-expected/2","schema":"rz-pals-frozen-replay-cli-expected/2""#, 1).into_bytes();
                let boolean_schema = String::from_utf8(expected_wire(EXPECTED_SCHEMA, ""))
                    .unwrap()
                    .replacen(
                        r#""schema":"rz-pals-frozen-replay-cli-expected/1""#,
                        r#""schema":true"#,
                        1,
                    )
                    .into_bytes();
                for raw in [
                    expected_wire(EXPECTED_SCHEMA, &scope_member("current_exe_path_hash")),
                    expected_wire(EXPECTED_SCHEMA_V2, ""),
                    expected_wire(
                        EXPECTED_SCHEMA_V2,
                        r#", "expected_semantic_receipt_producer_scope":null"#,
                    ),
                    expected_wire(
                        EXPECTED_SCHEMA_V2,
                        r#", "expected_semantic_receipt_producer_scope":false"#,
                    ),
                    expected_wire(
                        EXPECTED_SCHEMA_V2,
                        r#", "expected_semantic_receipt_producer_scope":{"linux_loaded_executable_inode":null}"#,
                    ),
                    expected_wire(
                        EXPECTED_SCHEMA_V2,
                        r#", "expected_semantic_receipt_producer_scope":"current_exe_path_hash", "expected_semantic_receipt_producer_scope":"current_exe_path_hash""#,
                    ),
                    expected_wire(EXPECTED_SCHEMA_V2, &scope_member("unknown")),
                    expected_wire("unknown-schema", ""),
                    duplicate_schema,
                    boolean_schema,
                ] {
                    let started = Instant::now();
                    let mut context = Context::new(started).unwrap();
                    let error = match parse_expected_transport(&raw, &mut context) {
                        Err(error) => error,
                        Ok(_) => panic!("closed codec refusal fixture unexpectedly accepted"),
                    };
                    assert!(matches!(
                        error.stage,
                        "expected_json" | "expected_schema" | "expected_schema_json"
                    ));
                    assert_eq!(context.deadline, started + Duration::from_millis(1000));
                    assert_eq!(
                        context.execution_deadline,
                        started + Duration::from_millis(900)
                    );
                    assert_eq!(context.whole_wall_ms, Some(1000));
                    assert_eq!(context.cleanup_reserve_ms, Some(100));
                    assert_eq!(context.output_limit, 1024);
                    assert!(!context.native_dispatch_entered);
                    assert!(context.native_error.is_none());
                    assert!(context.input_error.is_none());
                }
                // Keep the established valid-W/invalid-output refusal behavior.
                let invalid_output = String::from_utf8(expected_wire(EXPECTED_SCHEMA_V2, ""))
                    .unwrap()
                    .replacen(r#""output_bytes":1024"#, r#""output_bytes":0"#, 1)
                    .into_bytes();
                let started = Instant::now();
                let mut context = Context::new(started).unwrap();
                assert!(parse_expected_transport(&invalid_output, &mut context).is_err());
                assert_eq!(context.deadline, started + Duration::from_millis(1000));
                assert_eq!(context.whole_wall_ms, Some(1000));
                assert_eq!(context.output_limit, DIAGNOSTIC_BYTES);
            }
            #[test]
            fn malformed_expected_json_cannot_invent_projected_clock() {
                let started = Instant::now();
                let mut context = Context::new(started).unwrap();
                let before = (
                    context.deadline,
                    context.execution_deadline,
                    context.output_limit,
                );
                let raw = br#"{"whole_wall_ms":1000,"cleanup_reserve_ms":100,"output_bytes":1024,"schema":"#;
                assert!(parse_expected_transport(raw, &mut context).is_err());
                assert_eq!(
                    (
                        context.deadline,
                        context.execution_deadline,
                        context.output_limit
                    ),
                    before
                );
                assert!(context.whole_wall_ms.is_none());
                assert!(context.cleanup_reserve_ms.is_none());
            }
            #[test]
            fn semantic_scope_route_reuses_one_explicit_choice_and_v1_stays_library_only() {
                for selection in [
                    None,
                    Some(SemanticReceiptProducerScope::LibraryDispatcherArgument),
                    Some(SemanticReceiptProducerScope::LinuxLoadedExecutableInode),
                    Some(SemanticReceiptProducerScope::CurrentExePathHash),
                ] {
                    // The same production helper/selection feeds both call sites;
                    // this pure route test does not invoke a dispatcher or provider.
                    let initial_check_route = semantic_scope_route(selection);
                    let dispatch_route = semantic_scope_route(selection);
                    assert_eq!(initial_check_route, dispatch_route);
                    match selection {
                        None => assert_eq!(initial_check_route, SemanticScopeRoute::LibraryOnlyV1),
                        Some(scope) => {
                            assert_eq!(initial_check_route, SemanticScopeRoute::Explicit(scope))
                        }
                    }
                }
            }
            #[test]
            fn expected_semantic_scope_header_is_optional_declaration_not_image_witness() {
                for selection in [
                    None,
                    Some(SemanticReceiptProducerScope::LibraryDispatcherArgument),
                    Some(SemanticReceiptProducerScope::LinuxLoadedExecutableInode),
                    Some(SemanticReceiptProducerScope::CurrentExePathHash),
                ] {
                    let mut context = Context::new(Instant::now()).unwrap();
                    context.install_clock(5000, 500, 16 * 1024).unwrap();
                    context.expected_semantic_receipt_producer_scope = selection;
                    context.store_body(
                        b"{\"codec_fixture\":true}".to_vec(),
                        "codec_declaration_only",
                    );
                    let raw = envelope(&context).unwrap();
                    let value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
                    match selection {
                        None => assert!(
                            value["cli"]
                                .get("expected_semantic_receipt_producer_scope")
                                .is_none()
                        ),
                        Some(scope) => assert_eq!(
                            value["cli"]["expected_semantic_receipt_producer_scope"],
                            serde_json::to_value(scope).unwrap()
                        ),
                    }
                    assert_eq!(value["cli"]["loaded_provider_image_observed_by_cli"], false);
                    assert_eq!(value["cli"]["physical_closure_observed_by_cli"], false);
                    assert_eq!(value["cli"]["process_exit_observed"], false);
                    assert!(value["cli"]["stdout_delivered"].is_null());
                    assert!(!context.native_dispatch_entered);
                }
            }
            #[test]
            fn binary_domains_reject_same_sha_even_when_legacy_bytes_differ() {
                let replay = artifact(b"replay fixture");
                let mut legacy = replay.clone();
                legacy.bytes += 1;
                assert!(check_binary_domains(&replay, &legacy, &"b".repeat(64)).is_err());
                legacy.sha256 = "a".repeat(64);
                assert!(check_binary_domains(&replay, &legacy, &replay.sha256).is_err());
                assert!(check_binary_domains(&replay, &legacy, &"b".repeat(64)).is_ok());
            }
            #[test]
            fn launch_schema_rejects_extra_authority_or_derived_expected_fields() {
                let mut value = serde_json::json!({"schema":LAUNCH_SCHEMA,
                "export_manifest_path":"/manifest.json","runtime_library_path":"/ort.so",
                "runtime_cache_root":"/cache","cpu_fresh_profile_raw":"{}"});
                assert!(serde_json::from_value::<LaunchAssets>(value.clone()).is_ok());
                value["expected_profile"] = serde_json::json!({});
                assert!(serde_json::from_value::<LaunchAssets>(value).is_err());
            }
            #[test]
            fn runtime_observation_requires_cpu_single_library_and_independent_exact_bytes() {
                let expected = artifact(b"registered fixture bytes, not a native library");
                assert!(validate_runtime_observation(None, &expected, &expected).is_ok());
                let mut wrong_size = expected.clone();
                wrong_size.bytes += 1;
                assert!(validate_runtime_observation(None, &wrong_size, &expected).is_err());
                let mut wrong_digest = expected.clone();
                wrong_digest.sha256 = "a".repeat(64);
                assert!(validate_runtime_observation(None, &wrong_digest, &expected).is_err());
                assert!(validate_runtime_observation(Some([0; 32]), &expected, &expected).is_err());
            }
        }
    }

    pub(super) fn run_main() -> ExitCode {
        let started = Instant::now();
        let Some(mut context) = Context::new(started) else {
            return ExitCode::from(1);
        };
        let outcome = enabled::run(&mut context);
        let budget = Arc::new(StreamBudget::new(context.output_limit));
        match outcome {
            Ok(success) => match envelope(&context)
                .and_then(|bytes| deliver(&mut context, bytes, success, Arc::clone(&budget)))
            {
                Ok(()) if success => ExitCode::SUCCESS,
                Ok(()) => ExitCode::from(1),
                Err(error) => {
                    diagnose(&mut context, error, budget);
                    ExitCode::from(1)
                }
            },
            Err(error) => {
                diagnose(&mut context, error, budget);
                ExitCode::from(1)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn malformed_input_error() -> enabled::ReplayInputError {
            use rz_uci::pals_cpu_task::strategic_action::replay_inputs::{
                self, ReplayBindingPins, ReplayExpectedPins, ReplayParentPins,
                ReplayRegisteredArtifacts,
            };
            // Declaration-only pins for a deliberately malformed input. This path
            // cannot admit a provider, native run, completion or historical replay.
            let pin = artifact(b"declaration fixture");
            let h = || "c".repeat(64);
            let expected = ReplayExpectedPins {
                registration_artifact: pin.clone(),
                prepared_action_artifact: pin.clone(),
                semantic_receipt_artifact: pin.clone(),
                parent: ReplayParentPins {
                    parent_input_sha256: h(),
                    current_view_sha256: h(),
                    frozen_admission_sha256: h(),
                    encoding_sha256: h(),
                },
                binding: ReplayBindingPins {
                    query_sha256: h(),
                    catalogue_artifact: pin.clone(),
                    before_result_artifact: pin.clone(),
                    prior_ledger_sha256: h(),
                    semantic_input_sha256: h(),
                    semantic_context_sha256: h(),
                    semantic_branch_meaning_sha256: h(),
                    semantic_before_result_anchor_sha256: h(),
                    cpu_request_artifact: pin.clone(),
                },
                registered_artifacts: ReplayRegisteredArtifacts {
                    legacy_cpu_binary: pin.clone(),
                    replay_binary: pin.clone(),
                    engine_source: pin.clone(),
                    wrapper_source: pin.clone(),
                    replay_source: pin.clone(),
                    provider_factory_source: pin,
                    opponent_recheck_source: None,
                },
                legacy_cpu_profile_sha256: h(),
                provider_factory_id: "rz-uci-native-cpu-fresh-invocation-v1".into(),
                semantic_binary_sha256: h(),
            };
            match replay_inputs::check_replay_inputs(b"{", &expected, Instant::now()) {
                Err(error) => error,
                Ok(_) => panic!("malformed fixture unexpectedly admitted"),
            }
        }
        #[test]
        fn request_error_shorter_clock_and_cap_survive_transport_mismatch() {
            let started = Instant::now();
            let mut context = Context::new(started).unwrap();
            context.install_clock(5000, 500, 64 * 1024).unwrap();
            let mut error = malformed_input_error();
            // Exercise the public input-error clock observation boundary, not a
            // constructor of any native/physical authority.
            error.original_whole_deadline_retained = true;
            error.whole_wall_ms = Some(1000);
            error.cleanup_reserve_ms = Some(100);
            error.output_limit = 1024;
            let _ = context.retain_input_error(error); // Bounded diagnostic can itself exceed 1KiB.
            assert_eq!(context.deadline, started + Duration::from_millis(1000));
            assert_eq!(
                context.execution_deadline,
                started + Duration::from_millis(900)
            );
            assert_eq!(context.output_limit, 1024);
            let observed = context.input_error_clock.as_ref().unwrap();
            assert_eq!(observed.transport_whole_wall_ms, Some(5000));
            assert_eq!(observed.request_whole_wall_ms, Some(1000));
            assert_eq!(observed.transport_execution_allowance_ms, 4500);
            assert_eq!(observed.request_execution_allowance_ms, Some(900));
            assert_eq!(observed.effective_execution_allowance_ms, 900);
            assert!(observed.declarations_differ);
            assert!(context.input_error.is_some());
            let mut writer = BoundedJsonWriter {
                bytes: Vec::new(),
                deadline: context.deadline,
                limit: context.output_limit,
            };
            assert!(writer.write(&vec![0; 1025]).is_err());
            assert!(writer.bytes.is_empty());
        }
        #[test]
        fn expired_request_error_gets_no_new_serialization_grace() {
            let started = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
            let mut context = Context::new(started).unwrap();
            context.install_clock(5000, 500, 64 * 1024).unwrap();
            let mut error = malformed_input_error();
            error.original_whole_deadline_retained = true;
            error.whole_wall_ms = Some(1);
            error.cleanup_reserve_ms = None;
            error.output_limit = 1024;
            assert!(context.retain_input_error(error).is_err());
            assert_eq!(context.deadline, started + Duration::from_millis(1));
            assert_eq!(context.execution_deadline, context.deadline);
            assert_eq!(
                context
                    .input_error_clock
                    .as_ref()
                    .unwrap()
                    .request_execution_allowance_ms,
                None
            );
            assert!(context.raw_body.is_none());
            assert!(context.input_error.is_some());
        }
        fn request_clock_facts(
            started: Instant,
            whole_wall_ms: u64,
            cleanup_reserve_ms: u64,
            output_limit: usize,
        ) -> RequestClockFacts {
            // Pure projection fixture: these scalar observations do not create a
            // CheckedReplayInputs or certify successful/native replay admission.
            RequestClockFacts {
                deadline: started + Duration::from_millis(whole_wall_ms),
                execution_deadline: started
                    + Duration::from_millis(whole_wall_ms - cleanup_reserve_ms),
                output_limit,
                whole_wall_ms,
                cleanup_reserve_ms,
                output_bytes: output_limit,
            }
        }
        #[test]
        fn admitted_clock_mismatch_constrains_before_refusal_and_preserves_both_originals() {
            for (
                transport_whole,
                transport_cleanup,
                transport_output,
                request_whole,
                request_cleanup,
                request_output,
            ) in [
                (5000, 500, 64 * 1024, 1000, 100, 1024),
                (1000, 100, 1024, 5000, 500, 64 * 1024),
                (5000, 4800, 64 * 1024, 1000, 100, 1024),
            ] {
                let started = Instant::now();
                let mut context = Context::new(started).unwrap();
                context
                    .install_clock(transport_whole, transport_cleanup, transport_output)
                    .unwrap();
                let error = context
                    .constrain_request_clock(request_clock_facts(
                        started,
                        request_whole,
                        request_cleanup,
                        request_output,
                    ))
                    .unwrap_err();
                assert_eq!(error.stage, "clock_identity");
                let effective_whole = transport_whole.min(request_whole);
                let effective_execution =
                    (transport_whole - transport_cleanup).min(request_whole - request_cleanup);
                assert_eq!(
                    context.deadline,
                    started + Duration::from_millis(effective_whole)
                );
                assert_eq!(
                    context.execution_deadline,
                    started + Duration::from_millis(effective_execution)
                );
                assert_eq!(context.output_limit, transport_output.min(request_output));
                // Mixed minima do not invent cleanup = effective W - effective E.
                assert_eq!(context.cleanup_reserve_ms, Some(transport_cleanup));
                let observed = context.admitted_input_clock.as_ref().unwrap();
                assert_eq!(observed.transport.whole_wall_ms, Some(transport_whole));
                assert_eq!(
                    observed.transport.cleanup_reserve_ms,
                    Some(transport_cleanup)
                );
                assert_eq!(observed.transport.deadline_after_start_ms, transport_whole);
                assert_eq!(
                    observed.transport.execution_deadline_after_start_ms,
                    transport_whole - transport_cleanup
                );
                assert_eq!(observed.transport.output_limit, transport_output);
                assert_eq!(observed.request.whole_wall_ms, Some(request_whole));
                assert_eq!(observed.request.cleanup_reserve_ms, Some(request_cleanup));
                assert_eq!(observed.request.deadline_after_start_ms, request_whole);
                assert_eq!(
                    observed.request.execution_deadline_after_start_ms,
                    request_whole - request_cleanup
                );
                assert_eq!(observed.request.output_bytes, request_output);
                assert_eq!(observed.request.output_limit, request_output);
                assert_eq!(observed.effective.deadline_after_start_ms, effective_whole);
                assert_eq!(
                    observed.effective.execution_deadline_after_start_ms,
                    effective_execution
                );
                assert_eq!(
                    observed.effective.output_limit,
                    transport_output.min(request_output)
                );
                assert!(observed.declarations_differ);
                let raw = bounded_json(observed, context.deadline, context.output_limit).unwrap();
                let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
                assert_eq!(json["transport"]["whole_wall_ms"], transport_whole);
                assert_eq!(json["request"]["whole_wall_ms"], request_whole);
                assert_eq!(
                    json["effective"]["output_limit"],
                    transport_output.min(request_output)
                );
                assert!(json["effective"].get("cleanup_reserve_ms").is_none());
                assert!(!context.native_dispatch_entered);
                assert!(context.raw_body.is_none());
                assert!(context.input_error.is_none());
                assert!(context.input_error_clock.is_none());
                let mut writer = BoundedJsonWriter {
                    bytes: Vec::new(),
                    deadline: context.deadline,
                    limit: context.output_limit,
                };
                assert!(writer.write(&vec![0; context.output_limit + 1]).is_err());
                assert!(writer.bytes.is_empty());
            }
        }
        #[test]
        fn identical_admitted_clock_observations_leave_original_limits_unchanged() {
            let started = Instant::now();
            let mut context = Context::new(started).unwrap();
            context.install_clock(5000, 500, 64 * 1024).unwrap();
            let before = (
                context.deadline,
                context.execution_deadline,
                context.output_limit,
            );
            context
                .constrain_request_clock(request_clock_facts(started, 5000, 500, 64 * 1024))
                .unwrap();
            assert_eq!(
                (
                    context.deadline,
                    context.execution_deadline,
                    context.output_limit
                ),
                before
            );
            assert_eq!(context.whole_wall_ms, Some(5000));
            assert_eq!(context.cleanup_reserve_ms, Some(500));
            let observed = context.admitted_input_clock.as_ref().unwrap();
            assert!(!observed.declarations_differ);
            assert_eq!(observed.effective.deadline_after_start_ms, 5000);
            assert_eq!(observed.effective.execution_deadline_after_start_ms, 4500);
            assert_eq!(observed.effective.output_limit, 64 * 1024);
            assert!(!context.native_dispatch_entered);
            assert!(context.native_error.is_none());
        }
        #[test]
        fn elapsed_admitted_clock_projection_does_not_create_serialization_grace() {
            let started = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
            let mut context = Context::new(started).unwrap();
            context.install_clock(5000, 500, 64 * 1024).unwrap();
            let error = context
                .constrain_request_clock(request_clock_facts(started, 10, 2, 1024))
                .unwrap_err();
            assert_eq!(error.stage, "clock_identity");
            assert_eq!(context.deadline, started + Duration::from_millis(10));
            assert_eq!(
                context.execution_deadline,
                started + Duration::from_millis(8)
            );
            assert_eq!(context.output_limit, 1024);
            assert!(
                bounded_json(
                    &context.admitted_input_clock,
                    context.deadline,
                    context.output_limit
                )
                .is_err()
            );
            assert!(context.raw_body.is_none());
            assert!(!context.native_dispatch_entered);
        }
        #[derive(Debug)]
        struct OriginalIoCause(Arc<AtomicUsize>);
        impl std::fmt::Display for OriginalIoCause {
            fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                out.write_str("owned original fixture I/O cause")
            }
        }
        impl std::error::Error for OriginalIoCause {}
        impl Drop for OriginalIoCause {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::AcqRel);
            }
        }
        struct LateFailingWriter {
            fail_flush: bool,
            release: mpsc::Receiver<()>,
            entered: mpsc::SyncSender<()>,
            cause: Option<io::Error>,
        }
        impl LateFailingWriter {
            fn failure(&mut self) -> io::Result<usize> {
                self.entered
                    .send(())
                    .map_err(|_| io::Error::other("fixture entry channel"))?;
                self.release
                    .recv()
                    .map_err(|_| io::Error::other("fixture release channel"))?;
                Err(self.cause.take().expect("one original fixture cause"))
            }
        }
        impl Write for LateFailingWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.fail_flush {
                    Ok(bytes.len())
                } else {
                    self.failure()
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                if self.fail_flush {
                    self.failure().map(|_| ())
                } else {
                    Ok(())
                }
            }
        }
        #[test]
        fn actual_bounded_wait_timeout_preserves_late_write_and_flush_causes() {
            for fail_flush in [false, true] {
                let drops = Arc::new(AtomicUsize::new(0));
                let cause = io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    OriginalIoCause(Arc::clone(&drops)),
                );
                let budget = Arc::new(StreamBudget::new(8));
                let progress = Arc::new(WriteProgress::new(8));
                let (release, held) = mpsc::sync_channel(1);
                let (entered_send, entered_receive) = mpsc::sync_channel(1);
                let (returned_send, returned_receive) = mpsc::sync_channel(1);
                let thread_progress = Arc::clone(&progress);
                let deadline = Instant::now() + Duration::from_secs(2);
                std::thread::spawn(move || {
                    let result = bounded_write(
                        LateFailingWriter {
                            fail_flush,
                            release: held,
                            entered: entered_send,
                            cause: Some(cause),
                        },
                        Arc::new(vec![1; 8]),
                        deadline,
                        budget,
                        thread_progress,
                        None,
                    );
                    let _ = returned_send.send(result);
                });
                entered_receive
                    .recv_timeout(Duration::from_secs(2))
                    .unwrap();
                let error = returned_receive
                    .recv_timeout(Duration::from_secs(3))
                    .unwrap()
                    .unwrap_err();
                assert_eq!(error.stage, "output_wait_timeout");
                assert!(error.terminal.is_some());
                assert!(progress.observe().wait_timed_out);
                assert!(progress.terminal_guard().is_none());
                assert_eq!(drops.load(Ordering::Acquire), 0);
                // bounded_write has returned and dropped its real receiver. The
                // later exact error must survive in the one-slot shared record.
                release.send(()).unwrap();
                let stop = Instant::now() + Duration::from_secs(5);
                while !progress.writer_closed.load(Ordering::Acquire) && Instant::now() < stop {
                    std::thread::yield_now();
                }
                assert!(progress.writer_closed.load(Ordering::Acquire));
                let observation = progress.observe();
                assert_eq!(
                    observation.terminal_stage,
                    Some(if fail_flush { "flush" } else { "write" })
                );
                assert_eq!(observation.terminal_succeeded, Some(false));
                assert!(observation.wait_timed_out);
                assert_eq!(
                    observation.confirmed_bytes_written,
                    if fail_flush { 8 } else { 0 }
                );
                assert!(
                    progress
                        .terminal_guard()
                        .as_ref()
                        .unwrap()
                        .result
                        .as_ref()
                        .unwrap_err()
                        .get_ref()
                        .unwrap()
                        .downcast_ref::<OriginalIoCause>()
                        .is_some()
                );
                assert_eq!(drops.load(Ordering::Acquire), 0);
                let weak = Arc::downgrade(&progress);
                drop(error);
                drop(progress);
                let stop = Instant::now() + Duration::from_secs(5);
                while weak.upgrade().is_some() && Instant::now() < stop {
                    std::thread::yield_now();
                }
                assert!(weak.upgrade().is_none());
                assert_eq!(drops.load(Ordering::Acquire), 1);
            }
        }
        #[test]
        fn pending_writer_retains_same_typed_input_box_after_context_drop() {
            let mut context = Context::new(Instant::now()).unwrap();
            context.input_error = Some(Box::new(malformed_input_error()));
            context.raw_body = Some(Arc::new(b"{\"fixture\":true}".to_vec()));
            let original_address = (&**context.input_error.as_ref().unwrap()
                as *const enabled::ReplayInputError) as usize;
            let evidence = context.evidence_for_writer();
            let weak = Arc::downgrade(&evidence);
            let progress = Arc::new(WriteProgress::new(8));
            let (release, held) = mpsc::sync_channel(1);
            let (entered_send, entered_receive) = mpsc::sync_channel(1);
            let receive = start_write(
                WaitingWriter {
                    release: held,
                    entered: entered_send,
                },
                Arc::new(vec![1; 8]),
                Instant::now() + Duration::from_secs(5),
                Arc::new(StreamBudget::new(8)),
                Arc::clone(&progress),
                Some(evidence),
            )
            .unwrap();
            entered_receive
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            drop(context);
            let alive = weak.upgrade().unwrap();
            let guard = alive.lock().unwrap();
            assert!(guard.raw_body.is_some());
            assert_eq!(
                (&**guard.input_error.as_ref().unwrap() as *const enabled::ReplayInputError)
                    as usize,
                original_address
            );
            drop(guard);
            drop(alive);
            release.send(()).unwrap();
            receive.recv_timeout(Duration::from_secs(5)).unwrap();
            let stop = Instant::now() + Duration::from_secs(5);
            while weak.upgrade().is_some() && Instant::now() < stop {
                std::thread::yield_now();
            }
            assert!(weak.upgrade().is_none());
        }
        fn args() -> Vec<OsString> {
            let root = std::env::temp_dir();
            vec![
                "--expected-pins".into(),
                root.join("expected.json").into_os_string(),
                "--expected-sha256".into(),
                "a".repeat(64).into(),
                "--expected-bytes".into(),
                "123".into(),
                "--asset-profile".into(),
                root.join("launch.json").into_os_string(),
            ]
        }
        #[test]
        fn independent_pin_arguments_are_closed_and_explicit() {
            let parsed = arguments(&args()).unwrap();
            assert_eq!(parsed.expected_pin.bytes, 123);
            assert_eq!(
                parsed.expected_path,
                std::env::temp_dir().join("expected.json")
            );
            assert_eq!(parsed.launch_path, std::env::temp_dir().join("launch.json"));
            let mut duplicate = args();
            duplicate[6] = "--expected-pins".into();
            assert!(arguments(&duplicate).is_err());
            let mut hash = args();
            hash[3] = "A".repeat(64).into();
            assert!(arguments(&hash).is_err());
            let mut relative = args();
            relative[1] = "expected.json".into();
            assert!(arguments(&relative).is_err());
            let mut count = args();
            count[5] = "0123".into();
            assert!(arguments(&count).is_err());
            assert!(arguments(&[]).is_err());
            let mut oversized = args();
            oversized.push("excess".into());
            assert!(arguments(&oversized).is_err());
        }
        #[test]
        fn raw_artifact_identity_is_not_json_semantic_equality() {
            let pin = artifact(b"{\"x\":1}");
            assert!(check_bytes(b"{ \"x\": 1 }", &pin, 1024).is_err());
            assert!(check_bytes(b"{\"x\":1}", &pin, 1024).is_ok());
        }
        #[test]
        fn image_scope_is_distinct_from_external_registration() {
            assert_ne!(
                binary_scope(),
                "external_caller_registered_digest_not_loaded_image_observation"
            );
            assert_eq!(
                binary_scope(),
                if cfg!(target_os = "linux") {
                    "linux_loaded_executable_inode"
                } else {
                    "current_exe_path_hash"
                }
            );
        }
        struct WaitingWriter {
            release: mpsc::Receiver<()>,
            entered: mpsc::SyncSender<()>,
        }
        impl Write for WaitingWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                self.entered
                    .send(())
                    .map_err(|_| io::Error::other("fixture entry channel"))?;
                self.release
                    .recv()
                    .map_err(|_| io::Error::other("fixture channel"))?;
                Err(io::Error::other("fixture definite failure"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        #[test]
        fn pending_stdout_keeps_full_credit_until_writer_closes() {
            let budget = Arc::new(StreamBudget::new(12));
            let progress = Arc::new(WriteProgress::new(8));
            let (release, held) = mpsc::sync_channel(1);
            let (entered_send, entered_receive) = mpsc::sync_channel(1);
            let raw = Arc::new(b"{\"raw\":1}".to_vec());
            let raw_weak = Arc::downgrade(&raw);
            let mut context = Context::new(Instant::now()).unwrap();
            context.raw_body = Some(raw);
            let evidence = context.evidence_for_writer();
            let deadline = Instant::now() + Duration::from_secs(5);
            let receive = start_write(
                WaitingWriter {
                    release: held,
                    entered: entered_send,
                },
                Arc::new(vec![1; 8]),
                deadline,
                Arc::clone(&budget),
                Arc::clone(&progress),
                Some(evidence),
            )
            .unwrap();
            entered_receive
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            drop(context);
            // Observe an actually outstanding writer before asking for an immediate
            // receive timeout. The credit property does not rely on sleep/scheduling
            // to get this writer into its blocked operation before a tiny deadline.
            assert!(matches!(
                receive.recv_timeout(Duration::ZERO),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            assert_eq!(budget.remaining(), 4);
            assert!(!progress.writer_closed.load(Ordering::Acquire));
            assert!(raw_weak.upgrade().is_some());
            assert_eq!(progress.observe().reserved_or_committed_bytes, 8);
            assert_eq!(progress.observe().possible_unfinished_bytes, 8);
            assert!(budget.reserve(5).is_err());
            release.send(()).unwrap();
            receive.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(WriteProgress::terminal_result(&progress).is_err());
            assert!(progress.writer_closed.load(Ordering::Acquire));
            assert_eq!(budget.remaining(), 12);
            assert_eq!(progress.observe().confirmed_bytes_written, 0);
        }
        struct PartialWriter {
            writes: usize,
        }
        impl Write for PartialWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                if self.writes == 1 {
                    Ok(3)
                } else {
                    Err(io::Error::other("fixture failed after confirmed prefix"))
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        #[test]
        fn definite_partial_failure_refunds_only_unwritten_credit() {
            let budget = Arc::new(StreamBudget::new(12));
            let progress = Arc::new(WriteProgress::new(8));
            assert!(
                bounded_write(
                    PartialWriter { writes: 0 },
                    Arc::new(vec![1; 8]),
                    Instant::now() + Duration::from_secs(5),
                    Arc::clone(&budget),
                    Arc::clone(&progress),
                    None
                )
                .is_err()
            );
            assert_eq!(progress.observe().confirmed_bytes_written, 3);
            assert_eq!(progress.observe().reserved_or_committed_bytes, 3);
            assert_eq!(progress.observe().possible_unfinished_bytes, 0);
            assert!(progress.observe().writer_closed);
            assert_eq!(budget.remaining(), 9);
            assert!(budget.reserve(10).is_err());
        }
        struct LateSuccessfulWriter {
            release: mpsc::Receiver<()>,
            entered: mpsc::SyncSender<()>,
        }
        impl Write for LateSuccessfulWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.entered
                    .send(())
                    .map_err(|_| io::Error::other("fixture entry channel"))?;
                self.release
                    .recv()
                    .map_err(|_| io::Error::other("fixture channel"))?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        #[test]
        fn late_stdout_completion_and_other_stream_cannot_exceed_shared_cap() {
            let deadline = Instant::now() + Duration::from_secs(5);
            let budget = Arc::new(StreamBudget::new(12));
            let stdout_progress = Arc::new(WriteProgress::new(8));
            let (release, held) = mpsc::sync_channel(1);
            let (entered_send, entered_receive) = mpsc::sync_channel(1);
            let receive = start_write(
                LateSuccessfulWriter {
                    release: held,
                    entered: entered_send,
                },
                Arc::new(vec![1; 8]),
                deadline,
                Arc::clone(&budget),
                Arc::clone(&stdout_progress),
                None,
            )
            .unwrap();
            entered_receive
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            assert!(matches!(
                receive.recv_timeout(Duration::ZERO),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            let stderr_progress = Arc::new(WriteProgress::new(4));
            bounded_write(
                io::sink(),
                Arc::new(vec![2; 4]),
                deadline,
                Arc::clone(&budget),
                Arc::clone(&stderr_progress),
                None,
            )
            .unwrap();
            assert_eq!(budget.remaining(), 0);
            release.send(()).unwrap();
            receive.recv_timeout(Duration::from_secs(5)).unwrap();
            WriteProgress::terminal_result(&stdout_progress).unwrap();
            assert_eq!(stdout_progress.observe().confirmed_bytes_written, 8);
            assert_eq!(stderr_progress.observe().confirmed_bytes_written, 4);
            assert_eq!(budget.remaining(), 0);
            assert!(budget.reserve(1).is_err());
        }
        #[test]
        fn cooperative_cancel_and_digest_encoding_are_explicit() {
            let cancel = AtomicBool::new(true);
            let deadline = Instant::now() + Duration::from_secs(1);
            assert!(check_execution(deadline, &cancel, "fixture_prepare").is_err());
            cancel.store(false, Ordering::Release);
            assert!(check_execution(deadline, &cancel, "fixture_prepare").is_ok());
            assert_eq!(hex_digest(&[0xab; 32]), "ab".repeat(32));
        }
        #[test]
        fn invalid_output_retains_valid_original_whole_clock() {
            let start = Instant::now();
            let mut context = Context::new(start).unwrap();
            assert!(context.install_clock(1000, 100, 0).is_err());
            assert_eq!(context.deadline, start + Duration::from_millis(1000));
            assert_eq!(context.output_limit, DIAGNOSTIC_BYTES);
        }
        #[test]
        fn prepared_envelope_preserves_original_body_without_delivery_claim() {
            let mut context = Context::new(Instant::now()).unwrap();
            context.install_clock(1000, 100, 16 * 1024).unwrap();
            let raw = b"{ \"actual\": 7 }\n".to_vec();
            context.store_body(raw.clone(), "fixture_observation");
            let encoded = envelope(&context).unwrap();
            assert!(encoded.windows(raw.len()).any(|slice| slice == raw));
            let parsed: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
            assert!(parsed["cli"]["stdout_delivered"].is_null());
            assert_eq!(parsed["cli"]["physical_closure_observed_by_cli"], false);
            assert_eq!(artifact(context.raw_body.as_ref().unwrap()), artifact(&raw));
        }
        #[test]
        fn insufficient_envelope_budget_retains_raw_body_and_refuses_delivery() {
            let mut context = Context::new(Instant::now()).unwrap();
            context.install_clock(1000, 100, 1024).unwrap();
            let body = format!("{{\"body\":\"{}\"}}", "x".repeat(1000)).into_bytes();
            let expected = artifact(&body);
            context.store_body(body, "fixture_native_body");
            assert!(envelope(&context).is_err());
            assert_eq!(artifact(context.raw_body.as_ref().unwrap()), expected);
            assert!(context.progress.is_empty());
        }
    }
}
