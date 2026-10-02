//! Bounded native runner supervision on Linux.
//!
//! This owns a fresh process group, not a security sandbox. The child limit is
//! a `/proc` snapshot of that group: transient forks and children escaping with
//! a new session/group require a future cgroup adapter. A leader is not reaped
//! until group cleanup finishes, so its reserved PID cannot name another group.
//! This requires unchanged default SIGCHLD handling and no foreign child reaper.

use crate::ArenaError;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs::File;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::AtomicBool;

#[derive(Debug, Clone, Copy)]
pub struct ProcessLimits {
    pub wall_ms: u64,
    pub shutdown_grace_ms: u64,
    pub max_output_bytes: u64,
    /// Includes the runner leader and its engines, within the owned group.
    pub max_child_processes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStop {
    Exited,
    WallLimit,
    OutputLimit,
    Cancelled,
    IoFailure,
    ChildLimit,
    ArtifactLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupStatus {
    Gone,
    Unverified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessReceipt {
    pub supervisor_version: u32,
    pub pid: u32,
    pub elapsed_ns: u64,
    pub stop: ProcessStop,
    pub exit_code: Option<i32>,
    pub exit_signal: Option<i32>,
    pub group_cleanup: CleanupStatus,
    /// Bytes retained in the output artifact, never above the combined limit.
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
    pub observed_output_bytes: u64,
    pub descendant_cleanup_required: bool,
    /// Deliberately distinguishes observed enforcement from a kernel hard cap.
    pub child_limit_enforcement: String,
    /// Peak total length observed in watched regular output files.
    pub watched_artifact_bytes: Option<u64>,
    pub artifact_limit_enforcement: Option<String>,
    /// Stable bounded step/code evidence; no arguments, paths or child payload.
    pub errors: Vec<String>,
}

#[derive(Debug)]
pub struct ProcessOutput {
    pub receipt: ProcessReceipt,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// A leader still pending after the bounded cleanup interval. Long-lived
    /// callers must retain this handle and later try_wait/reap it. This is not
    /// returned after ownership was lost to an unsupported external reaper.
    pub pending_child: Option<std::process::Child>,
}

#[derive(Debug, Clone)]
pub struct ArtifactWatch {
    /// Supported files are direct children of the pinned working directory.
    pub relative_files: Vec<PathBuf>,
    pub max_total_bytes: u64,
}

impl ArtifactWatch {
    fn validate(&self) -> Result<(), ArenaError> {
        if self.max_total_bytes == 0
            || self.relative_files.is_empty()
            || self.relative_files.len() > 16
        {
            return Err(ArenaError::Budget(
                "artifact watch requires 1..16 files and positive bytes".into(),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for path in &self.relative_files {
            let mut parts = path.components();
            if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
                return Err(ArenaError::Invalid(
                    "artifact watch requires direct relative filenames".into(),
                ));
            }
            let Some(name) = path.to_str() else {
                return Err(ArenaError::Invalid(
                    "artifact watch filenames require UTF-8".into(),
                ));
            };
            if name.starts_with('.')
                || name.len() > 255
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
                || !seen.insert(name)
            {
                return Err(ArenaError::Invalid(
                    "artifact watch filename is unsafe or duplicated".into(),
                ));
            }
        }
        Ok(())
    }
}

impl ProcessLimits {
    fn validate(self) -> Result<(), ArenaError> {
        if self.wall_ms == 0
            || self.shutdown_grace_ms == 0
            || self.max_output_bytes == 0
            || self.max_child_processes == 0
        {
            return Err(ArenaError::Budget("process limits must be positive".into()));
        }
        let total_ms = self
            .shutdown_grace_ms
            .checked_mul(2)
            .and_then(|grace| self.wall_ms.checked_add(grace))
            .ok_or_else(|| ArenaError::Budget("process duration overflow".into()))?;
        if total_ms.checked_mul(1_000_000).is_none()
            || usize::try_from(self.max_output_bytes).is_err()
            || self.max_child_processes > i32::MAX as u32
        {
            return Err(ArenaError::Budget(
                "process limits exceed representable bounds".into(),
            ));
        }
        Ok(())
    }
}

/// Execute exactly the already-opened native ELF artifact. Linux resolves the
/// open descriptor at exec; replacing its former path cannot swap the program.
/// Scripts are rejected because their interpreter may reopen a closed fd.
pub fn supervise(
    program: &File,
    args: &[OsString],
    cwd: &Path,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    #[cfg(target_os = "linux")]
    {
        linux::supervise(
            program,
            args,
            linux::pin_directory(cwd)?,
            limits,
            cancel,
            None,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, cwd, cancel);
        Err(ArenaError::Invalid(
            "native process supervision currently requires Linux".into(),
        ))
    }
}

/// Additionally stop when watched regular artifacts exceed their combined
/// byte cap. Observation is periodic, not a kernel disk quota. Missing files
/// are allowed until created; symlinks/FIFOs/devices cause an explicit failure.
pub fn supervise_with_watch(
    program: &File,
    args: &[OsString],
    cwd: &Path,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &ArtifactWatch,
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    watch.validate()?;
    #[cfg(target_os = "linux")]
    {
        linux::supervise(
            program,
            args,
            linux::pin_directory(cwd)?,
            limits,
            cancel,
            Some(watch),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, cwd, cancel);
        Err(ArenaError::Invalid(
            "native process supervision currently requires Linux".into(),
        ))
    }
}

/// Use an already-pinned working directory capability without reopening its
/// former path. The cloned handle is shared by launch and artifact observation.
pub fn supervise_in_directory(
    program: &File,
    args: &[OsString],
    cwd: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: Option<&ArtifactWatch>,
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    if let Some(watch) = watch {
        watch.validate()?;
    }
    #[cfg(target_os = "linux")]
    {
        if !cwd
            .metadata()
            .map_err(|_| ArenaError::Io("process.cwd_metadata".into()))?
            .is_dir()
        {
            return Err(ArenaError::Invalid(
                "runner cwd handle must be a directory".into(),
            ));
        }
        let directory = cwd
            .try_clone()
            .map_err(|_| ArenaError::Io("process.cwd_clone".into()))?;
        linux::supervise(program, args, directory, limits, cancel, watch)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, cwd, cancel);
        Err(ArenaError::Invalid(
            "native process supervision currently requires Linux".into(),
        ))
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use nix::errno::Errno;
    use nix::fcntl::{FcntlArg, OFlag, fcntl, open, openat};
    use nix::sys::signal::{Signal, killpg};
    use nix::sys::stat::Mode;
    use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid};
    use nix::unistd::Pid;
    use std::io::{self, Read};
    use std::os::fd::{AsFd, AsRawFd};
    use std::os::unix::fs::FileExt;
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    use std::process::{Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    const POLL: Duration = Duration::from_millis(5);

    pub(super) fn pin_directory(cwd: &Path) -> Result<File, ArenaError> {
        open(
            cwd,
            OFlag::O_PATH | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        )
        .map(File::from)
        .map_err(|_| ArenaError::Invalid("runner cwd must be a direct directory".into()))
    }

    pub(super) fn supervise(
        program: &File,
        args: &[OsString],
        cwd_file: File,
        limits: ProcessLimits,
        cancel: Option<&AtomicBool>,
        watch: Option<&ArtifactWatch>,
    ) -> Result<ProcessOutput, ArenaError> {
        native_elf(program)?;
        default_child_disposition()?;
        // Verify required observation support before starting an executable.
        group_snapshot(Pid::this().as_raw() as u32, 0)
            .map_err(|_| ArenaError::Io("process.proc_preflight".into()))?;
        if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Err(ArenaError::Invalid("process.cancelled_before_spawn".into()));
        }
        let started = Instant::now();
        let wall = Duration::from_millis(limits.wall_ms);
        let grace = Duration::from_millis(limits.shutdown_grace_ms);
        started
            .checked_add(wall)
            .and_then(|time| time.checked_add(grace))
            .and_then(|time| time.checked_add(grace))
            .ok_or_else(|| ArenaError::Budget("process monotonic deadline overflow".into()))?;
        let executable = format!("/proc/self/fd/{}", program.as_raw_fd());
        let pinned_cwd = format!("/proc/self/fd/{}", cwd_file.as_raw_fd());
        let mut child = Command::new(executable)
            .args(args)
            .current_dir(pinned_cwd)
            .env_clear()
            .env("LANG", "C")
            .env("PATH", "/usr/bin:/bin")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .map_err(|error| ArenaError::Io(format!("process.spawn:{:?}", error.kind())))?;
        let pid = child.id();
        let group = Pid::from_raw(
            i32::try_from(pid).map_err(|_| ArenaError::Io("process.pid_out_of_range".into()))?,
        );
        let mut receipt = ProcessReceipt {
            supervisor_version: 1,
            pid,
            elapsed_ns: 0,
            stop: ProcessStop::Exited,
            exit_code: None,
            exit_signal: None,
            group_cleanup: CleanupStatus::Unverified,
            stdout_bytes: 0,
            stderr_bytes: 0,
            observed_output_bytes: 0,
            descendant_cleanup_required: false,
            child_limit_enforcement: "process_group_snapshot".into(),
            watched_artifact_bytes: watch.map(|_| 0),
            artifact_limit_enforcement: watch.map(|_| "regular_file_snapshot".into()),
            errors: Vec::new(),
        };
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        let pipe_setup = stdout
            .as_ref()
            .ok_or(Errno::EBADF)
            .and_then(nonblocking)
            .and_then(|()| stderr.as_ref().ok_or(Errno::EBADF).and_then(nonblocking));
        let mut stop = if pipe_setup.is_err() {
            evidence(&mut receipt, "process.pipe_nonblocking");
            // A blocking descriptor is never read by this loop.
            stdout = None;
            stderr = None;
            Some(ProcessStop::IoFailure)
        } else {
            None
        };
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut shutdown = None;
        let mut killed = false;
        let mut leader_done = false;
        let mut ownership_lost = false;
        loop {
            let now = Instant::now();
            if stop.is_none() {
                if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                    stop = Some(ProcessStop::Cancelled);
                } else if now.duration_since(started) >= wall {
                    stop = Some(ProcessStop::WallLimit);
                }
            }
            if let Some(watch) = watch {
                match watched_size(&cwd_file, watch) {
                    Ok(bytes) => {
                        receipt.watched_artifact_bytes =
                            Some(receipt.watched_artifact_bytes.unwrap_or(0).max(bytes));
                        if bytes > watch.max_total_bytes {
                            stop.get_or_insert(ProcessStop::ArtifactLimit);
                        }
                    }
                    Err(code) => {
                        evidence(&mut receipt, code);
                        stop.get_or_insert(ProcessStop::IoFailure);
                    }
                }
            }
            // Each pass drains a finite amount from each pipe, preserving time,
            // cancellation and process checks even under continuous output.
            for result in [
                drain(&mut stdout, &mut out, limits.max_output_bytes, &mut receipt),
                drain(&mut stderr, &mut err, limits.max_output_bytes, &mut receipt),
            ] {
                match result {
                    Ok(true) if stop.is_none() => stop = Some(ProcessStop::OutputLimit),
                    Err(()) => {
                        evidence(&mut receipt, "process.pipe_read");
                        if stop.is_none() {
                            stop = Some(ProcessStop::IoFailure);
                        }
                    }
                    _ => {}
                }
            }
            if !leader_done {
                match waitid(
                    Id::Pid(group),
                    WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
                ) {
                    Ok(WaitStatus::Exited(_, _)) | Ok(WaitStatus::Signaled(_, _, _)) => {
                        leader_done = true;
                    }
                    Ok(WaitStatus::StillAlive) => {}
                    Ok(_) => {
                        evidence(&mut receipt, "process.wait_status");
                        stop.get_or_insert(ProcessStop::IoFailure);
                    }
                    Err(Errno::EINTR) => {}
                    Err(_) => {
                        evidence(&mut receipt, "process.wait_observation");
                        // Ownership is no longer proven. An auto/foreign reaper
                        // may have released this numeric PID: never signal a
                        // potentially recycled group ID after this point.
                        stop = Some(ProcessStop::IoFailure);
                        ownership_lost = true;
                        evidence(&mut receipt, "process.ownership_lost");
                        break;
                    }
                }
            }
            let last_group = match group_snapshot(pid, pid) {
                Ok(snapshot) => {
                    if snapshot.members > limits.max_child_processes {
                        stop.get_or_insert(ProcessStop::ChildLimit);
                    }
                    if leader_done && snapshot.other_members > 0 {
                        receipt.descendant_cleanup_required = true;
                    }
                    Some(snapshot)
                }
                Err(()) => {
                    evidence(&mut receipt, "process.proc_observation");
                    stop.get_or_insert(ProcessStop::IoFailure);
                    None
                }
            };
            let no_others = last_group.is_some_and(|snapshot| snapshot.other_members == 0);
            if leader_done && no_others && stdout.is_none() && stderr.is_none() {
                receipt.group_cleanup = CleanupStatus::Gone;
                break;
            }
            if leader_done {
                stop.get_or_insert(ProcessStop::Exited);
            }
            if stop.is_some() && shutdown.is_none() {
                shutdown = Some(now);
                signal(group, Signal::SIGTERM, &mut receipt, "process.term");
            }
            if let Some(shutdown_at) = shutdown {
                let age = now.duration_since(shutdown_at);
                if age >= grace && !killed {
                    signal(group, Signal::SIGKILL, &mut receipt, "process.kill");
                    killed = true;
                }
                if age >= grace.saturating_mul(2) {
                    // This never joins a reader blocked by an escaped child.
                    evidence(&mut receipt, "process.cleanup_unverified");
                    break;
                }
            }
            std::thread::sleep(POLL);
        }
        // WNOWAIT kept this PID reserved throughout every group signal. Reap
        // only an observed terminal child; never block waiting on an unknown one.
        if leader_done {
            match child.wait() {
                Ok(status) => {
                    receipt.exit_code = status.code();
                    receipt.exit_signal = status.signal();
                }
                Err(_) => evidence(&mut receipt, "process.reap"),
            }
        } else {
            // A successful SIGKILL can still remain pending in uninterruptible
            // kernel IO. Leave cleanup explicitly unverified, without waiting.
            evidence(&mut receipt, "process.leader_unreaped");
        }
        receipt.stop = stop.unwrap_or(ProcessStop::Exited);
        receipt.elapsed_ns = u64::try_from(started.elapsed().as_nanos())
            .map_err(|_| ArenaError::Budget("process elapsed duration overflow".into()))?;
        receipt.stdout_bytes = out.len() as u64;
        receipt.stderr_bytes = err.len() as u64;
        Ok(ProcessOutput {
            receipt,
            stdout: out,
            stderr: err,
            pending_child: if !leader_done && !ownership_lost {
                Some(child)
            } else {
                None
            },
        })
    }

    fn native_elf(program: &File) -> Result<(), ArenaError> {
        let metadata = program
            .metadata()
            .map_err(|_| ArenaError::Io("process.program_metadata".into()))?;
        if !metadata.is_file() {
            return Err(ArenaError::Invalid(
                "runner must be a regular native ELF file".into(),
            ));
        }
        let mut header = [0_u8; 4];
        program
            .read_exact_at(&mut header, 0)
            .map_err(|_| ArenaError::Invalid("runner ELF header unavailable".into()))?;
        if header != *b"\x7fELF" {
            return Err(ArenaError::Invalid(
                "runner scripts and non-ELF files are unsupported".into(),
            ));
        }
        Ok(())
    }

    fn default_child_disposition() -> Result<(), ArenaError> {
        // These masks reject visible ignored/caught handlers. They cannot
        // expose SIG_DFL+SA_NOCLDWAIT: unchanged default flags and exclusive
        // reaping remain a caller precondition, not a claimed /proc proof.
        let file = File::open("/proc/self/status")
            .map_err(|_| ArenaError::Io("process.signal_preflight".into()))?;
        let mut bytes = Vec::new();
        file.take(65_537)
            .read_to_end(&mut bytes)
            .map_err(|_| ArenaError::Io("process.signal_preflight".into()))?;
        if bytes.len() > 65_536 {
            return Err(ArenaError::Io("process.signal_preflight_size".into()));
        }
        let mut ignored = None;
        let mut caught = None;
        for line in bytes.split(|byte| *byte == b'\n') {
            for (prefix, target) in [
                (b"SigIgn:".as_slice(), &mut ignored),
                (b"SigCgt:".as_slice(), &mut caught),
            ] {
                if let Some(value) = line.strip_prefix(prefix) {
                    *target = std::str::from_utf8(value)
                        .ok()
                        .and_then(|value| u64::from_str_radix(value.trim(), 16).ok());
                }
            }
        }
        let mask = 1_u64 << (Signal::SIGCHLD as u32 - 1);
        match (ignored, caught) {
            (Some(ignored), Some(caught)) if (ignored | caught) & mask == 0 => Ok(()),
            (Some(_), Some(_)) => Err(ArenaError::Invalid(
                "process requires default SIGCHLD disposition and exclusive child reaping".into(),
            )),
            _ => Err(ArenaError::Io("process.signal_preflight_fields".into())),
        }
    }

    fn watched_size(cwd: &File, watch: &ArtifactWatch) -> Result<u64, &'static str> {
        let mut bytes = 0_u64;
        for path in &watch.relative_files {
            // O_PATH only obtains an inode reference. No FIFO/device open can
            // block, and NOFOLLOW returns a symlink inode instead of its target.
            let file = match openat(
                cwd,
                path.as_path(),
                OFlag::O_PATH | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            ) {
                Ok(fd) => File::from(fd),
                Err(Errno::ENOENT) => continue,
                Err(_) => return Err("process.artifact_open"),
            };
            let metadata = file.metadata().map_err(|_| "process.artifact_metadata")?;
            if !metadata.is_file() {
                return Err("process.artifact_kind");
            }
            bytes = bytes
                .checked_add(metadata.len())
                .ok_or("process.artifact_size_overflow")?;
        }
        Ok(bytes)
    }

    fn nonblocking<T: AsFd>(pipe: &T) -> Result<(), Errno> {
        let flags = OFlag::from_bits_truncate(fcntl(pipe, FcntlArg::F_GETFL)?);
        fcntl(pipe, FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK))?;
        Ok(())
    }

    fn drain<T: Read>(
        pipe: &mut Option<T>,
        retained: &mut Vec<u8>,
        cap: u64,
        receipt: &mut ProcessReceipt,
    ) -> Result<bool, ()> {
        let Some(reader) = pipe.as_mut() else {
            return Ok(false);
        };
        let mut buffer = [0_u8; 8192];
        for _ in 0..16 {
            let stored = receipt.stdout_bytes + receipt.stderr_bytes;
            let remaining = cap - stored;
            let request = usize::try_from(remaining.saturating_add(1))
                .unwrap_or(usize::MAX)
                .min(buffer.len());
            match reader.read(&mut buffer[..request]) {
                Ok(0) => {
                    *pipe = None;
                    return Ok(false);
                }
                Ok(bytes) => {
                    receipt.observed_output_bytes = receipt
                        .observed_output_bytes
                        .checked_add(bytes as u64)
                        .ok_or(())?;
                    let keep = bytes.min(remaining as usize);
                    retained.extend_from_slice(&buffer[..keep]);
                    // Increment the combined counter without binding a pipe's
                    // identity; final stdout/stderr lengths replace it below.
                    receipt.stdout_bytes += keep as u64;
                    if bytes > keep {
                        *pipe = None;
                        return Ok(true);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    *pipe = None;
                    return Err(());
                }
            }
        }
        Ok(false)
    }

    #[derive(Clone, Copy)]
    struct GroupSnapshot {
        members: u32,
        other_members: u32,
    }

    fn group_snapshot(group: u32, leader: u32) -> Result<GroupSnapshot, ()> {
        let mut members = 0_u32;
        let mut other_members = 0_u32;
        for entry in std::fs::read_dir("/proc").map_err(|_| ())? {
            let entry = entry.map_err(|_| ())?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Ok(pid) = name.parse::<u32>() else {
                continue;
            };
            let stat = match File::open(entry.path().join("stat")) {
                Ok(file) => file,
                Err(error) if disappeared(&error) => continue,
                Err(_) => return Err(()),
            };
            let mut bytes = Vec::new();
            match stat.take(4097).read_to_end(&mut bytes) {
                Ok(_) if bytes.len() <= 4096 => {}
                Err(error) if disappeared(&error) => continue,
                _ => return Err(()),
            }
            // comm may contain spaces, ')' and non-UTF-8 bytes. Only decode the
            // ASCII numeric suffix after its final ')' delimiter.
            let delimiter = bytes.iter().rposition(|byte| *byte == b')').ok_or(())?;
            let fields = std::str::from_utf8(&bytes[delimiter + 1..]).map_err(|_| ())?;
            let pgid = fields
                .split_whitespace()
                .nth(2)
                .and_then(|value| value.parse::<u32>().ok())
                .ok_or(())?;
            if pgid == group {
                members = members.checked_add(1).ok_or(())?;
                if pid != leader {
                    other_members = other_members.checked_add(1).ok_or(())?;
                }
            }
        }
        Ok(GroupSnapshot {
            members,
            other_members,
        })
    }

    fn disappeared(error: &io::Error) -> bool {
        error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(Errno::ESRCH as i32)
    }

    fn signal(group: Pid, signal: Signal, receipt: &mut ProcessReceipt, step: &str) {
        match killpg(group, signal) {
            Ok(()) | Err(Errno::ESRCH) => {}
            Err(error) => evidence(receipt, &format!("{step}:{}", error as i32)),
        }
    }

    fn evidence(receipt: &mut ProcessReceipt, code: &str) {
        if receipt.errors.len() < 16 && !receipt.errors.iter().any(|item| item == code) {
            receipt.errors.push(code.chars().take(256).collect());
        }
    }
}
