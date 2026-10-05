//! Bounded native runner supervision on Linux.
//!
//! This owns a fresh process group, not a security sandbox. The child limit is
//! a `/proc` snapshot of that group: transient forks and children escaping with
//! a new session/group require a future cgroup adapter. A leader is not reaped
//! while group cleanup is in progress, so its reserved PID cannot name another
//! group. The tree API retains the original child after unverified cleanup;
//! a PID reservation is proven only while reaping ownership is intact. The
//! older flat APIs retain their original terminal-child reaping behavior.
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
    /// Peak total length observed in watched regular output files. A tree
    /// observation that stops at a limit may describe only the visited prefix.
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
    /// A leader retained after the bounded cleanup interval. For the tree API,
    /// this includes an observed terminal leader when cleanup is Unverified.
    /// With intact ownership, not reaping reserves its PID/process-group ID.
    /// After ownership_lost, this is only the original handle for quarantine;
    /// it proves no reservation and grants no numeric PID wait/signal authority.
    /// A typed external owner must resolve physical cleanup and authority before
    /// reaping or dropping it. No subsequent kill/reap/drop is automatic.
    /// Flat APIs retain their older pending-only behavior and return no handle
    /// after ownership was lost to an unsupported external reaper.
    pub pending_child: Option<std::process::Child>,
}

#[derive(Debug, Clone)]
pub struct ArtifactWatch {
    /// Supported files are direct children of the pinned working directory.
    pub relative_files: Vec<PathBuf>,
    pub max_total_bytes: u64,
}

/// Periodic observation of an exclusively owned output directory tree.
/// This measures file lengths, not allocated blocks or a kernel disk quota.
/// Runtime-created subdirectories are included; symlinks, regular files with
/// multiple hard links and all non-directory/non-regular kinds are rejected.
#[derive(Debug, Clone, Copy)]
pub struct OwnedArtifactTreeWatch {
    pub max_total_bytes: u64,
    pub max_file_bytes: u64,
    /// Maximum regular files. Directories have a separate equal count limit,
    /// and at most twice this many entries are examined per snapshot, including
    /// entries that disappear before their inode can be opened.
    pub max_files: usize,
    /// Root depth is zero; a direct child directory has depth one.
    pub max_depth: usize,
}

impl OwnedArtifactTreeWatch {
    fn validate(&self) -> Result<(), ArenaError> {
        if self.max_total_bytes == 0
            || self.max_file_bytes == 0
            || !(1..=4096).contains(&self.max_files)
            || self.max_depth > 32
        {
            return Err(ArenaError::Budget(
                "artifact tree requires positive bytes, 1..4096 files and depth at most 32".into(),
            ));
        }
        Ok(())
    }
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
            Some(linux::ArtifactObservation::Flat(watch)),
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
        linux::supervise(
            program,
            args,
            directory,
            limits,
            cancel,
            watch.map(linux::ArtifactObservation::Flat),
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

/// Supervise using a cloned directory capability and bounded recursive tree
/// snapshots. Each inode is opened relative to a pinned directory without
/// following symlinks; regular payloads are never opened for reading. This is
/// observation, not a hard disk quota or an atomic view of a changing tree.
/// When cleanup remains Unverified and leader ownership is intact, even a
/// terminal leader is returned unreaped in pending_child. The caller must keep
/// that reservation until its typed owner confirms physical cleanup; polling
/// Child::try_wait also reaps and therefore releases the reservation.
/// If reaping ownership is lost, the original Child is still returned for
/// quarantine, with ownership_lost evidence: no reservation or further numeric
/// PID wait/signal authority is established by preserving that handle.
pub fn supervise_in_directory_with_tree(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
) -> Result<ProcessOutput, ArenaError> {
    supervise_tree_observed(program, args, directory, limits, cancel, watch, None)
}

/// Passive finite stream observer. It cannot alter output, cancellation or gates.
pub(crate) fn supervise_tree_observed(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    stdout_observer: Option<&mut dyn FnMut(&[u8])>,
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    watch.validate()?;
    #[cfg(target_os = "linux")]
    {
        if !directory
            .metadata()
            .map_err(|_| ArenaError::Io("process.cwd_metadata".into()))?
            .is_dir()
        {
            return Err(ArenaError::Invalid(
                "runner cwd handle must be a directory".into(),
            ));
        }
        let directory = directory
            .try_clone()
            .map_err(|_| ArenaError::Io("process.cwd_clone".into()))?;
        linux::supervise(
            program,
            args,
            directory,
            limits,
            cancel,
            Some(linux::ArtifactObservation::Tree(watch)),
            stdout_observer,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, directory, cancel, stdout_observer);
        Err(ArenaError::Invalid(
            "native process supervision currently requires Linux".into(),
        ))
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use cap_std::fs::Dir;
    use nix::errno::Errno;
    use nix::fcntl::{FcntlArg, OFlag, fcntl, open, openat};
    use nix::sys::signal::{Signal, killpg};
    use nix::sys::stat::Mode;
    use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid};
    use nix::unistd::Pid;
    use std::io::{self, Read};
    use std::os::fd::{AsFd, AsRawFd};
    use std::os::unix::fs::{FileExt, MetadataExt};
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    use std::process::{Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    const POLL: Duration = Duration::from_millis(5);

    #[derive(Clone, Copy)]
    pub(super) enum ArtifactObservation<'a> {
        Flat(&'a ArtifactWatch),
        Tree(&'a OwnedArtifactTreeWatch),
    }

    impl ArtifactObservation<'_> {
        fn enforcement(self) -> String {
            match self {
                Self::Flat(_) => "regular_file_snapshot".into(),
                Self::Tree(_) => "owned_tree_snapshot".into(),
            }
        }
    }

    pub(super) fn pin_directory(cwd: &Path) -> Result<File, ArenaError> {
        open(
            cwd,
            OFlag::O_PATH | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        )
        .map(File::from)
        .map_err(|_| ArenaError::Invalid("runner cwd must be a direct directory".into()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn supervise(
        program: &File,
        args: &[OsString],
        cwd_file: File,
        limits: ProcessLimits,
        cancel: Option<&AtomicBool>,
        watch: Option<ArtifactObservation<'_>>,
        mut stdout_observer: Option<&mut dyn FnMut(&[u8])>,
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
            .envs(source_profile_environment(std::env::var_os(
                "RZ_ARENA_SOURCE_PROFILE",
            )))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .map_err(|error| ArenaError::Io(format!("process.spawn:{:?}", error.kind())))?;
        let pid = child.id();
        crate::emit_native_phase("runner_spawned");
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
            artifact_limit_enforcement: watch.map(ArtifactObservation::enforcement),
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
        let preserve_unverified = matches!(watch, Some(ArtifactObservation::Tree(_)));
        loop {
            let now = Instant::now();
            if stop.is_none() {
                if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                    stop = Some(ProcessStop::Cancelled);
                } else if now.duration_since(started) >= wall {
                    stop = Some(ProcessStop::WallLimit);
                }
            }
            if let Some(ArtifactObservation::Flat(watch)) = watch {
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
            if let Some(ArtifactObservation::Tree(watch)) = watch {
                observe_tree(&cwd_file, watch, &mut receipt, &mut stop);
            }
            // Each pass drains a finite amount from each pipe, preserving time,
            // cancellation and process checks even under continuous output.
            let prior_out = out.len();
            let prior_err = err.len();
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
            // Observe only newly retained bytes; the raw pipe budget and later
            // provider/clock acceptance still own the complete unchanged output.
            if out.len() > prior_out {
                crate::native_diagnostics::observe_stream(true, &out[prior_out..]);
                if let Some(observer) = stdout_observer.as_mut() {
                    observer(&out[prior_out..]);
                }
            }
            if err.len() > prior_err {
                crate::native_diagnostics::observe_stream(false, &err[prior_err..]);
            }
            if !leader_done {
                match waitid(
                    Id::Pid(group),
                    WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
                ) {
                    Ok(WaitStatus::Exited(_, code)) => {
                        leader_done = true;
                        crate::emit_native_phase("runner_exit_observed");
                        if preserve_unverified {
                            receipt.exit_code = Some(code);
                        }
                    }
                    Ok(WaitStatus::Signaled(_, signal, _)) => {
                        leader_done = true;
                        crate::emit_native_phase("runner_exit_observed");
                        if preserve_unverified {
                            receipt.exit_signal = Some(signal as i32);
                        }
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
                Err(error) => {
                    evidence(&mut receipt, "process.proc_observation");
                    evidence(&mut receipt, &error.code());
                    stop.get_or_insert(ProcessStop::IoFailure);
                    None
                }
            };
            let no_others = last_group.is_some_and(|snapshot| snapshot.other_members == 0);
            if leader_done && no_others && stdout.is_none() && stderr.is_none() {
                // The child may have created its final outputs between this
                // pass's snapshot and exit observation. Inspect that final
                // tree before reporting a completed run, without changing the
                // older flat API's observation schedule.
                if let Some(ArtifactObservation::Tree(watch)) = watch {
                    observe_tree(&cwd_file, watch, &mut receipt, &mut stop);
                }
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
        let pending_child = finish_child(
            child,
            &mut receipt,
            &mut stop,
            leader_done,
            ownership_lost,
            preserve_unverified,
        );
        receipt.stop = stop.unwrap_or(ProcessStop::Exited);
        receipt.elapsed_ns = u64::try_from(started.elapsed().as_nanos())
            .map_err(|_| ArenaError::Budget("process elapsed duration overflow".into()))?;
        receipt.stdout_bytes = out.len() as u64;
        receipt.stderr_bytes = err.len() as u64;
        Ok(ProcessOutput {
            receipt,
            stdout: out,
            stderr: err,
            pending_child,
        })
    }

    fn finish_child(
        mut child: std::process::Child,
        receipt: &mut ProcessReceipt,
        stop: &mut Option<ProcessStop>,
        leader_done: bool,
        ownership_lost: bool,
        preserve_unverified: bool,
    ) -> Option<std::process::Child> {
        if preserve_unverified && ownership_lost {
            // Preserve the original handle and evidence, not numeric PID
            // authority: a foreign/automatic reaper may already have freed it.
            receipt.group_cleanup = CleanupStatus::Unverified;
            *stop = Some(ProcessStop::IoFailure);
            evidence(receipt, "process.ownership_lost");
        }
        let retain_child = if preserve_unverified {
            ownership_lost || !leader_done || receipt.group_cleanup == CleanupStatus::Unverified
        } else {
            !ownership_lost && !leader_done
        };
        if leader_done && !retain_child && !ownership_lost {
            match child.wait() {
                Ok(status) => {
                    receipt.exit_code = status.code();
                    receipt.exit_signal = status.signal();
                }
                Err(_) => {
                    evidence(receipt, "process.reap");
                    if preserve_unverified {
                        receipt.group_cleanup = CleanupStatus::Unverified;
                        *stop = Some(ProcessStop::IoFailure);
                        evidence(receipt, "process.ownership_lost");
                        return Some(child);
                    }
                }
            }
        } else {
            // This supervisor did not reap it. With ownership_lost this is
            // not a claim that a foreign reaper left the child unreaped.
            evidence(receipt, "process.leader_unreaped");
        }
        retain_child.then_some(child)
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

    struct TreeWatchFailure {
        bytes: u64,
        stop: ProcessStop,
        code: &'static str,
    }

    #[derive(Default)]
    struct TreeSnapshot {
        bytes: u64,
        files: usize,
        directories: usize,
        entries: usize,
    }

    fn observe_tree(
        cwd: &File,
        watch: &OwnedArtifactTreeWatch,
        receipt: &mut ProcessReceipt,
        stop: &mut Option<ProcessStop>,
    ) {
        let (bytes, failure) = match watched_tree_size(cwd, watch) {
            Ok(bytes) => (bytes, None),
            Err(failure) => (failure.bytes, Some(failure)),
        };
        receipt.watched_artifact_bytes =
            Some(receipt.watched_artifact_bytes.unwrap_or(0).max(bytes));
        if let Some(failure) = failure {
            evidence(receipt, failure.code);
            // A terminal leader is not a successful artifact verdict. A later
            // observed tree failure must not be hidden by the provisional
            // Exited stop while pipes/group cleanup were still in progress.
            if stop.is_none() || *stop == Some(ProcessStop::Exited) {
                *stop = Some(failure.stop);
            }
        }
    }

    fn watched_tree_size(
        cwd: &File,
        watch: &OwnedArtifactTreeWatch,
    ) -> Result<u64, TreeWatchFailure> {
        let mut snapshot = TreeSnapshot::default();
        match visit_tree(cwd, watch, 0, &mut snapshot) {
            Ok(()) => Ok(snapshot.bytes),
            Err((stop, code)) => Err(TreeWatchFailure {
                bytes: snapshot.bytes,
                stop,
                code,
            }),
        }
    }

    fn visit_tree(
        directory: &File,
        watch: &OwnedArtifactTreeWatch,
        depth: usize,
        snapshot: &mut TreeSnapshot,
    ) -> Result<(), (ProcessStop, &'static str)> {
        // Opening '.' obtains a fresh readable directory description from the
        // already pinned inode, including an O_PATH root. It never reopens the
        // entry's former name, and its offset is not shared with the caller.
        let readable = openat(
            directory,
            ".",
            OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        )
        .map(File::from)
        .map_err(|_| {
            (
                ProcessStop::IoFailure,
                "process.artifact_tree.directory_open",
            )
        })?;
        let capability = Dir::from_std_file(readable);
        let entries = capability
            .entries()
            .map_err(|_| (ProcessStop::IoFailure, "process.artifact_tree.enumerate"))?;
        for entry in entries {
            let entry =
                entry.map_err(|_| (ProcessStop::IoFailure, "process.artifact_tree.enumerate"))?;
            // Count even a disappearing entry: directory churn cannot turn a
            // nominal file limit into an unbounded enumeration pass.
            snapshot.entries += 1;
            if snapshot.entries > watch.max_files * 2 {
                return Err((
                    ProcessStop::ArtifactLimit,
                    "process.artifact_tree.entry_count",
                ));
            }
            let name = entry.file_name();
            let mut components = Path::new(&name).components();
            if !matches!(components.next(), Some(Component::Normal(_)))
                || components.next().is_some()
            {
                return Err((ProcessStop::IoFailure, "process.artifact_tree.entry_name"));
            }
            // O_PATH references every inode kind without opening a FIFO,
            // socket or device payload. NOFOLLOW preserves a symlink itself.
            let inode = match openat(
                directory,
                Path::new(&name),
                OFlag::O_PATH | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            ) {
                Ok(fd) => File::from(fd),
                Err(Errno::ENOENT) => continue,
                Err(_) => {
                    return Err((ProcessStop::IoFailure, "process.artifact_tree.inode_open"));
                }
            };
            let metadata = inode
                .metadata()
                .map_err(|_| (ProcessStop::IoFailure, "process.artifact_tree.metadata"))?;
            if metadata.is_dir() {
                snapshot.directories += 1;
                if snapshot.directories > watch.max_files {
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.directory_count",
                    ));
                }
                if depth >= watch.max_depth {
                    return Err((ProcessStop::ArtifactLimit, "process.artifact_tree.depth"));
                }
                visit_tree(&inode, watch, depth + 1, snapshot)?;
            } else if metadata.is_file() {
                if metadata.nlink() != 1 {
                    return Err((ProcessStop::IoFailure, "process.artifact_tree.hard_link"));
                }
                snapshot.files += 1;
                if snapshot.files > watch.max_files {
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.file_count",
                    ));
                }
                let Some(bytes) = snapshot.bytes.checked_add(metadata.len()) else {
                    snapshot.bytes = u64::MAX;
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.size_overflow",
                    ));
                };
                snapshot.bytes = bytes;
                if metadata.len() > watch.max_file_bytes {
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.file_bytes",
                    ));
                }
                if snapshot.bytes > watch.max_total_bytes {
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.total_bytes",
                    ));
                }
            } else {
                return Err((ProcessStop::IoFailure, "process.artifact_tree.kind"));
            }
        }
        Ok(())
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

    #[derive(Debug)]
    struct ProcObservationError {
        stage: &'static str,
        errno: Option<i32>,
    }

    impl ProcObservationError {
        fn invalid(stage: &'static str) -> Self {
            Self { stage, errno: None }
        }

        fn io(stage: &'static str, error: io::Error) -> Self {
            Self {
                stage,
                errno: error.raw_os_error(),
            }
        }

        fn code(&self) -> String {
            format!(
                "process.proc_observation.{}:{}",
                self.stage,
                self.errno.unwrap_or(0)
            )
        }
    }

    fn group_snapshot(group: u32, leader: u32) -> Result<GroupSnapshot, ProcObservationError> {
        let mut members = 0_u32;
        let mut other_members = 0_u32;
        for entry in std::fs::read_dir("/proc")
            .map_err(|error| ProcObservationError::io("enumerate", error))?
        {
            let entry = entry.map_err(|error| ProcObservationError::io("entry", error))?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Ok(pid) = name.parse::<u32>() else {
                continue;
            };
            let stat = match File::open(entry.path().join("stat")) {
                Ok(file) => file,
                Err(error) if disappeared(&error) => continue,
                Err(error) => return Err(ProcObservationError::io("stat_open", error)),
            };
            let mut bytes = Vec::new();
            match stat.take(4097).read_to_end(&mut bytes) {
                Ok(_) if bytes.len() <= 4096 => {}
                Err(error) if disappeared(&error) => continue,
                Err(error) => return Err(ProcObservationError::io("stat_read", error)),
                _ => return Err(ProcObservationError::invalid("stat_size")),
            }
            let Some(pgid) = parse_proc_group(&bytes)? else {
                continue;
            };
            if pgid == group {
                members = members
                    .checked_add(1)
                    .ok_or_else(|| ProcObservationError::invalid("member_count"))?;
                if pid != leader {
                    other_members = other_members
                        .checked_add(1)
                        .ok_or_else(|| ProcObservationError::invalid("member_count"))?;
                }
            }
        }
        Ok(GroupSnapshot {
            members,
            other_members,
        })
    }

    fn parse_proc_group(bytes: &[u8]) -> Result<Option<u32>, ProcObservationError> {
        // comm may contain spaces, ')' and non-UTF-8 bytes. Only decode the
        // ASCII numeric suffix after its final ')' delimiter.
        let delimiter = bytes
            .iter()
            .rposition(|byte| *byte == b')')
            .ok_or_else(|| {
                ProcObservationError::invalid(if bytes.is_empty() {
                    "stat_empty"
                } else {
                    "stat_delimiter"
                })
            })?;
        let fields = std::str::from_utf8(&bytes[delimiter + 1..])
            .map_err(|_| ProcObservationError::invalid("stat_utf8"))?;
        let pgid = fields
            .split_whitespace()
            .nth(2)
            .ok_or_else(|| ProcObservationError::invalid("stat_group_missing"))?
            .parse::<i32>()
            .map_err(|_| ProcObservationError::invalid("stat_group_invalid"))?;
        // Linux do_task_stat starts pgid at -1. If the task has been retired
        // and lock_task_sighand fails, the kernel prints that sentinel instead
        // of a group ID. It cannot name our positive, unreaped leader's group.
        // Malformed records and other negative IDs still fail observation.
        match pgid {
            -1 => Ok(None),
            0.. => Ok(Some(pgid as u32)),
            _ => Err(ProcObservationError::invalid("stat_group_invalid")),
        }
    }

    fn disappeared(error: &io::Error) -> bool {
        error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(Errno::ESRCH as i32)
    }

    /// One explicit passive diagnostic flag; never forward ambient variables,
    /// library search paths, credentials or provider/precision selections.
    fn source_profile_environment(
        value: Option<std::ffi::OsString>,
    ) -> Option<(&'static str, &'static str)> {
        (value.as_deref() == Some(std::ffi::OsStr::new("1")))
            .then_some(("RZ_NATIVE_SOURCE_PROFILE", "1"))
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn source_profile_forwarding_is_one_exact_opt_in_value() {
            assert_eq!(source_profile_environment(None), None);
            for value in ["", "0", "true", "1 ", "--profile"] {
                assert_eq!(source_profile_environment(Some(value.into())), None);
            }
            assert_eq!(
                source_profile_environment(Some("1".into())),
                Some(("RZ_NATIVE_SOURCE_PROFILE", "1"))
            );
        }

        #[test]
        fn retired_proc_group_sentinel_is_not_a_live_group_or_parse_failure() {
            assert_eq!(parse_proc_group(b"12 (rz) X 1 -1 12\n").unwrap(), None);
            assert_eq!(
                parse_proc_group(b"12 (rz )\xff) S 1 42 12\n").unwrap(),
                Some(42)
            );
            assert_eq!(parse_proc_group(b"12 (rz) S 1 0 12\n").unwrap(), Some(0));
            for invalid in [
                b"12 (rz) S 1 -2 12\n".as_slice(),
                b"12 (rz) S 1 invalid 12\n",
                b"12 (rz) S 1 2147483648 12\n",
                b"12 (rz) S 1\n",
                b"12 (rz) S 1 \xff 12\n",
                b"",
            ] {
                assert!(parse_proc_group(invalid).is_err());
            }
        }

        #[test]
        fn proc_snapshot_during_owned_process_churn() {
            let mut child = Command::new("/bin/sh")
                .args([
                    "-c",
                    "i=0; while [ $i -lt 3000 ]; do /bin/true; i=$((i+1)); done",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .process_group(0)
                .spawn()
                .unwrap();
            let pid = child.id();
            let start = Instant::now();
            let mut failure = None;
            while start.elapsed() < Duration::from_secs(3) && child.try_wait().unwrap().is_none() {
                if let Err(error) = group_snapshot(pid, pid) {
                    failure = Some(error);
                    break;
                }
            }
            if child.try_wait().unwrap().is_none() {
                assert!(matches!(
                    killpg(Pid::from_raw(i32::try_from(pid).unwrap()), Signal::SIGKILL),
                    Ok(()) | Err(Errno::ESRCH)
                ));
                child.wait().unwrap();
            }
            assert!(failure.is_none(), "{failure:?}");
        }

        fn externally_reaped_child() -> std::process::Child {
            let child = Command::new("/bin/true")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .process_group(0)
                .spawn()
                .unwrap();
            let pid = Pid::from_raw(i32::try_from(child.id()).unwrap());
            // Consume only this exact test-owned child, without a global child
            // reaper. Child has not cached the status, so its later wait hits
            // the real OS ECHILD boundary that a foreign reaper would create.
            assert_eq!(
                waitid(Id::Pid(pid), WaitPidFlag::WEXITED).unwrap(),
                WaitStatus::Exited(pid, 0)
            );
            child
        }

        fn receipt(pid: u32, cleanup: CleanupStatus) -> ProcessReceipt {
            ProcessReceipt {
                supervisor_version: 1,
                pid,
                elapsed_ns: 0,
                stop: ProcessStop::Exited,
                exit_code: Some(0),
                exit_signal: None,
                group_cleanup: cleanup,
                stdout_bytes: 0,
                stderr_bytes: 0,
                observed_output_bytes: 0,
                descendant_cleanup_required: false,
                child_limit_enforcement: "process_group_snapshot".into(),
                watched_artifact_bytes: None,
                artifact_limit_enforcement: None,
                errors: Vec::new(),
            }
        }

        #[test]
        fn tree_observation_ownership_loss_keeps_original_handle_without_reaping() {
            for tree in [false, true] {
                let child = externally_reaped_child();
                let pid = child.id();
                let mut receipt = receipt(pid, CleanupStatus::Unverified);
                receipt.errors = vec![
                    "process.wait_observation".into(),
                    "process.ownership_lost".into(),
                ];
                let mut stop = Some(ProcessStop::IoFailure);
                let pending = finish_child(child, &mut receipt, &mut stop, false, true, tree);
                assert_eq!(
                    pending.as_ref().map(std::process::Child::id),
                    tree.then_some(pid)
                );
                assert_eq!(receipt.group_cleanup, CleanupStatus::Unverified);
                assert_eq!(stop, Some(ProcessStop::IoFailure));
                assert!(
                    receipt
                        .errors
                        .iter()
                        .any(|code| code == "process.ownership_lost")
                );
                // Ownership loss is evidence for quarantine, never permission
                // to issue a numeric PID wait/signal from this returned handle.
                assert!(!receipt.errors.iter().any(|code| code == "process.reap"));
            }
        }

        #[test]
        fn tree_reap_failure_keeps_original_handle_and_invalidates_cleanup() {
            for tree in [false, true] {
                let child = externally_reaped_child();
                let pid = child.id();
                let mut receipt = receipt(pid, CleanupStatus::Gone);
                let mut stop = Some(ProcessStop::Exited);
                let pending = finish_child(child, &mut receipt, &mut stop, true, false, tree);
                assert_eq!(
                    pending.as_ref().map(std::process::Child::id),
                    tree.then_some(pid)
                );
                assert!(receipt.errors.iter().any(|code| code == "process.reap"));
                assert_eq!(
                    receipt.group_cleanup,
                    if tree {
                        CleanupStatus::Unverified
                    } else {
                        CleanupStatus::Gone
                    }
                );
                assert_eq!(
                    stop,
                    Some(if tree {
                        ProcessStop::IoFailure
                    } else {
                        ProcessStop::Exited
                    })
                );
                assert_eq!(
                    receipt
                        .errors
                        .iter()
                        .any(|code| code == "process.ownership_lost"),
                    tree
                );
            }
        }
    }
}
