//! 준비 완료 경계에서 읽은 Linux 자원 관측. 프로세스의 전체 수명, peak,
//! UCI 옵션 적용 또는 시스템 전체 자원 정책 준수의 증명으로 해석하지 않는다.
//! 파일 읽기는 caller deadline 안에서 최대 500ms의 cooperative 창과 byte
//! 상한을 검사한다. 이는 진행 중인 kernel syscall의 강제 wall 중단 보증이
//! 아니다. nofollow·nonblocking 열기와 실제 procfs/cgroup2 타입을 확인한다.

use rz_search::cpu_checker::ExternalProcessIdentity;
use serde::Serialize;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
#[cfg(target_os = "linux")]
use std::{
    sync::atomic::Ordering,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservedLimit {
    Numeric { value: u64 },
    Max,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CgroupLimitsSnapshot {
    pub mount_point: String,
    pub mount_root: String,
    pub resolved_directory: String,
    /// 해당 cgroup 파일에서 읽은 직접 값이다. ancestor의 effective ceiling,
    /// 전체 수명 동안의 정책 적용 또는 메모리 peak의 관측이 아니다.
    pub memory_high: ObservedLimit,
    pub memory_max: ObservedLimit,
    pub memory_swap_max: ObservedLimit,
    pub pids_max: ObservedLimit,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ThreadAffinitySnapshot {
    pub tid: u32,
    pub proc_start_ticks_before: u64,
    pub proc_start_ticks_after: u64,
    pub cpu_allowed_list: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProcessResourceSnapshot {
    pub pid: u32,
    pub parent_pid: u32,
    pub process_group: u32,
    pub proc_start_ticks_before: u64,
    pub proc_start_ticks_after: u64,
    /// /proc/<pid>/cgroup에서 관측자가 읽은 경로. 대상의 namespace 안에서
    /// 실행한 경로나 시스템 전역 절대 cgroup 이름이라고 주장하지 않는다.
    pub cgroup_v2_membership: String,
    pub membership_path_view: &'static str,
    pub cgroup_namespace_inode: Option<u64>,
    pub cpu_allowed_list: String,
    pub threads: Vec<ThreadAffinitySnapshot>,
    pub thread_observation_scope: &'static str,
    pub cgroup_limits: CgroupLimitsSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReadyBoundaryResources {
    pub schema_version: u32,
    pub domain: &'static str,
    pub scope: &'static str,
    pub observed_unix_us: u64,
    pub observation_elapsed_us: u64,
    pub parent: ProcessResourceSnapshot,
    pub helper: ProcessResourceSnapshot,
    pub cgroup_membership_equal: bool,
    pub cgroup_namespace_equal: Option<bool>,
    pub cpu_allowed_list_equal: bool,
    /// 별도 예산을 선언하지 않는다는 실행 정책과 실제 관측은 구별한다.
    pub additional_allocation: &'static str,
    pub read_consistency: &'static str,
}

/// Pure serialization/lifecycle fixture; it is not an actual OS observation
/// and is never included in non-test binaries or product receipts.
#[cfg(test)]
pub(crate) fn fixture() -> ReadyBoundaryResources {
    let limits = CgroupLimitsSnapshot {
        mount_point: "/sys/fs/cgroup".into(),
        mount_root: "/".into(),
        resolved_directory: "/sys/fs/cgroup/test-only-group".into(),
        memory_high: ObservedLimit::Numeric {
            value: 6 * 1024 * 1024 * 1024,
        },
        memory_max: ObservedLimit::Numeric {
            value: 12 * 1024 * 1024 * 1024,
        },
        memory_swap_max: ObservedLimit::Max,
        pids_max: ObservedLimit::Numeric { value: 128 },
    };
    let parent = ProcessResourceSnapshot {
        pid: 100,
        parent_pid: 99,
        process_group: 100,
        proc_start_ticks_before: 123,
        proc_start_ticks_after: 123,
        cgroup_v2_membership: "/test-only-group".into(),
        membership_path_view: "observer_procfs_and_cgroup2_mount_view",
        cgroup_namespace_inode: Some(12345),
        cpu_allowed_list: "0,2".into(),
        threads: vec![ThreadAffinitySnapshot {
            tid: 100,
            proc_start_ticks_before: 123,
            proc_start_ticks_after: 123,
            cpu_allowed_list: "0,2".into(),
        }],
        thread_observation_scope: "bounded_ready_boundary_thread_snapshot_not_lifetime_enforcement",
        cgroup_limits: limits,
    };
    let mut helper = parent.clone();
    helper.pid = 101;
    helper.parent_pid = 100;
    helper.process_group = 101;
    helper.proc_start_ticks_before = 456;
    helper.proc_start_ticks_after = 456;
    helper.threads = vec![ThreadAffinitySnapshot {
        tid: 101,
        proc_start_ticks_before: 456,
        proc_start_ticks_after: 456,
        cpu_allowed_list: "0,2".into(),
    }];
    ReadyBoundaryResources {
        schema_version: 1,
        domain: "rz-pals-checker-ready-resources/1",
        scope: "linux_ready_boundary_snapshot",
        observed_unix_us: 1,
        observation_elapsed_us: 2,
        parent,
        helper,
        cgroup_membership_equal: true,
        cgroup_namespace_equal: Some(true),
        cpu_allowed_list_equal: true,
        additional_allocation: "none_declared_not_an_enforcement_proof",
        read_consistency: "identity_membership_affinity_bracketed_limits_sequential_not_atomic",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceError {
    #[cfg(target_os = "linux")]
    Unavailable(&'static str),
    #[cfg(target_os = "linux")]
    Invalid(&'static str),
    #[cfg(target_os = "linux")]
    Deadline,
}

pub(crate) struct ResourceObservation {
    pub snapshot: Option<ReadyBoundaryResources>,
    pub unavailable: Option<&'static str>,
}

/// 선택한 helper의 역사적 spawn identity만 읽는다. None은 pre-spawn,
/// 지원되지 않는 OS 또는 이미 사라진 프로세스이며 0 값 관측이 아니다.
pub(crate) fn observe(
    helper: Option<ExternalProcessIdentity>,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<ResourceObservation, ResourceError> {
    let Some(helper) = helper else {
        return Ok(ResourceObservation {
            snapshot: None,
            unavailable: Some("helper_process_identity_not_observed"),
        });
    };
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (helper, deadline, cancel);
        Ok(ResourceObservation {
            snapshot: None,
            unavailable: Some("host_not_linux"),
        })
    }
    #[cfg(target_os = "linux")]
    {
        match linux::observe(helper, deadline, cancel) {
            Err(ResourceError::Unavailable(code)) => Ok(ResourceObservation {
                snapshot: None,
                unavailable: Some(code),
            }),
            value => value.map(|snapshot| ResourceObservation {
                snapshot: Some(snapshot),
                unavailable: None,
            }),
        }
    }
}

#[cfg(target_os = "linux")]
fn controlled(deadline: Instant, cancel: &AtomicBool) -> Result<(), ResourceError> {
    if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
        Err(ResourceError::Deadline)
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::{
        ambient_authority,
        fs::{Dir, OpenOptions, OpenOptionsExt},
    };
    use nix::sys::statfs::{CGROUP2_SUPER_MAGIC, FsType, PROC_SUPER_MAGIC, fstatfs};
    use std::io::{self, Read};
    use std::path::{Component, Path};
    use std::time::Duration;

    const MAX_STAT: usize = 4096;
    const MAX_STATUS: usize = 16 * 1024;
    const MAX_CGROUP: usize = 4096;
    const MAX_MOUNTINFO: usize = 64 * 1024;
    const MAX_THREADS: usize = 64;
    const MAX_TOTAL_READ: usize = 3 * 1024 * 1024;

    struct Budget<'a> {
        bytes: usize,
        deadline: Instant,
        cancel: &'a AtomicBool,
    }
    impl Budget<'_> {
        fn check(&self) -> Result<(), ResourceError> {
            controlled(self.deadline, self.cancel)
        }
        fn read(
            &mut self,
            directory: &Dir,
            name: &str,
            cap: usize,
        ) -> Result<String, ResourceError> {
            self.check()?;
            if self
                .bytes
                .checked_add(cap + 1)
                .is_none_or(|bytes| bytes > MAX_TOTAL_READ)
            {
                return Err(ResourceError::Invalid("resource_read_total_bound"));
            }
            let mut options = OpenOptions::new();
            options
                .read(true)
                .follow(FollowSymlinks::No)
                .custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_CLOEXEC);
            let mut file = directory.open_with(name, &options).map_err(unavailable)?;
            self.check()?;
            let directory_kind = filesystem(directory, self)?;
            let file_kind = filesystem(&file, self)?;
            if directory_kind != file_kind
                || (file_kind != PROC_SUPER_MAGIC && file_kind != CGROUP2_SUPER_MAGIC)
            {
                return Err(ResourceError::Invalid(
                    "resource_kernel_filesystem_mismatch",
                ));
            }
            if !file.metadata().map_err(unavailable)?.is_file() {
                return Err(ResourceError::Invalid("non_regular_resource_file"));
            }
            let bytes = bounded_read(&mut file, cap, self.deadline, self.cancel)?;
            self.bytes = self
                .bytes
                .checked_add(bytes.len())
                .ok_or(ResourceError::Invalid("resource_read_overflow"))?;
            if self.bytes > MAX_TOTAL_READ {
                return Err(ResourceError::Invalid("resource_read_total_bound"));
            }
            self.check()?;
            String::from_utf8(bytes).map_err(|_| ResourceError::Invalid("resource_encoding"))
        }
    }

    fn unavailable(error: io::Error) -> ResourceError {
        ResourceError::Unavailable(match error.kind() {
            io::ErrorKind::NotFound => "resource_file_missing",
            io::ErrorKind::PermissionDenied => "resource_permission_denied",
            _ => "resource_read_io_unavailable",
        })
    }

    fn filesystem(
        fd: &impl std::os::fd::AsFd,
        budget: &Budget<'_>,
    ) -> Result<FsType, ResourceError> {
        budget.check()?;
        let kind = fstatfs(fd)
            .map_err(|_| ResourceError::Unavailable("resource_filesystem_unavailable"))?
            .filesystem_type();
        budget.check()?;
        Ok(kind)
    }

    pub(super) fn bounded_read(
        reader: &mut impl Read,
        cap: usize,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>, ResourceError> {
        controlled(deadline, cancel)?;
        let capacity = cap
            .checked_add(1)
            .filter(|_| cap <= MAX_MOUNTINFO)
            .ok_or(ResourceError::Invalid("resource_file_byte_bound"))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(capacity)
            .map_err(|_| ResourceError::Invalid("resource_read_allocation"))?;
        let mut chunk = [0; 1024];
        loop {
            controlled(deadline, cancel)?;
            let remaining = capacity - output.len();
            let length = chunk.len().min(remaining);
            let count = reader.read(&mut chunk[..length]).map_err(unavailable)?;
            controlled(deadline, cancel)?;
            if count == 0 {
                break;
            }
            output.extend_from_slice(&chunk[..count]);
            if output.len() > cap {
                return Err(ResourceError::Invalid("resource_file_byte_bound"));
            }
        }
        controlled(deadline, cancel)?;
        Ok(output)
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(super) struct ProcIdentity {
        pub pid: u32,
        pub parent: u32,
        pub group: u32,
        pub start: u64,
    }

    pub(super) fn stat(text: &str, expected_pid: u32) -> Result<ProcIdentity, ResourceError> {
        let (head, body) = text
            .rsplit_once(") ")
            .ok_or(ResourceError::Invalid("proc_stat_format"))?;
        let pid = head
            .split_once(" (")
            .and_then(|(pid, _)| pid.parse::<u32>().ok())
            .filter(|pid| *pid == expected_pid)
            .ok_or(ResourceError::Invalid("proc_pid_mismatch"))?;
        let mut fields = body.split_ascii_whitespace();
        let state = fields.next().ok_or(ResourceError::Invalid("proc_state"))?;
        if matches!(state, "Z" | "X" | "x") {
            return Err(ResourceError::Unavailable("process_gone"));
        }
        let parent = fields
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or(ResourceError::Invalid("proc_parent"))?;
        let group = fields
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or(ResourceError::Invalid("proc_group"))?;
        let start = fields
            .nth(16)
            .and_then(|v| v.parse().ok())
            .ok_or(ResourceError::Invalid("proc_start"))?;
        Ok(ProcIdentity {
            pid,
            parent,
            group,
            start,
        })
    }

    pub(super) fn validate_identity(
        actual: ProcIdentity,
        expected: ExternalProcessIdentity,
    ) -> Result<(), ResourceError> {
        if expected.pid == 0
            || expected.process_group == 0
            || expected.proc_start_ticks == 0
            || actual.pid != expected.pid
            || actual.group != expected.process_group
            || actual.start != expected.proc_start_ticks
        {
            Err(ResourceError::Invalid("helper_spawn_identity_mismatch"))
        } else {
            Ok(())
        }
    }

    pub(super) fn cgroup(text: &str) -> Result<String, ResourceError> {
        let mut selected = None;
        for line in text.lines() {
            if let Some(path) = line.strip_prefix("0::") {
                if selected.is_some() {
                    return Err(ResourceError::Invalid("duplicate_cgroup_v2"));
                }
                checked_absolute(path)?;
                selected = Some(path.to_owned());
            }
        }
        selected.ok_or(ResourceError::Unavailable("cgroup_v2_unavailable"))
    }

    fn checked_absolute(path: &str) -> Result<(), ResourceError> {
        if !path.starts_with('/')
            || path.len() > 4096
            || path.chars().any(char::is_control)
            || Path::new(path)
                .components()
                .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
            || path.split('/').any(|part| part == "." || part == "..")
        {
            return Err(ResourceError::Invalid("cgroup_path_not_canonical"));
        }
        Ok(())
    }

    pub(super) fn cpu_list(text: &str) -> Result<String, ResourceError> {
        let mut selected = None;
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("Cpus_allowed_list:") {
                if selected.is_some() {
                    return Err(ResourceError::Invalid("duplicate_cpu_allowed_list"));
                }
                let value = value.trim();
                if value.is_empty() || value.len() > 1024 {
                    return Err(ResourceError::Invalid("cpu_allowed_list_bound"));
                }
                let mut previous = None;
                let mut count = 0u32;
                for part in value.split(',') {
                    let (low, high) = part.split_once('-').map_or((part, part), |v| v);
                    let low = low
                        .parse::<u32>()
                        .map_err(|_| ResourceError::Invalid("cpu_allowed_list_format"))?;
                    let high = high
                        .parse::<u32>()
                        .map_err(|_| ResourceError::Invalid("cpu_allowed_list_format"))?;
                    if high < low || high > 65535 || previous.is_some_and(|prev| low <= prev) {
                        return Err(ResourceError::Invalid("cpu_allowed_list_order"));
                    }
                    count = count
                        .checked_add(high - low + 1)
                        .filter(|count| *count <= 4096)
                        .ok_or(ResourceError::Invalid("cpu_allowed_list_count"))?;
                    previous = Some(high);
                }
                selected = Some(value.to_owned());
            }
        }
        selected.ok_or(ResourceError::Unavailable("cpu_allowed_list_unavailable"))
    }

    pub(super) fn limit(text: &str) -> Result<ObservedLimit, ResourceError> {
        let value = text.trim();
        if value == "max" {
            return Ok(ObservedLimit::Max);
        }
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ResourceError::Invalid("cgroup_limit_format"));
        }
        value
            .parse()
            .map(|value| ObservedLimit::Numeric { value })
            .map_err(|_| ResourceError::Invalid("cgroup_limit_overflow"))
    }

    fn decode_mount(text: &str) -> Result<String, ResourceError> {
        let mut bytes = Vec::new();
        let mut at = 0;
        while at < text.len() {
            if text.as_bytes()[at] == b'\\' {
                let code = text
                    .get(at + 1..at + 4)
                    .ok_or(ResourceError::Invalid("mount_escape"))?;
                let byte = match code {
                    "040" => b' ',
                    "011" => b'\t',
                    "012" => b'\n',
                    "134" => b'\\',
                    _ => return Err(ResourceError::Invalid("mount_escape")),
                };
                bytes.push(byte);
                at += 4;
            } else {
                bytes.push(text.as_bytes()[at]);
                at += 1;
            }
        }
        let decoded =
            String::from_utf8(bytes).map_err(|_| ResourceError::Invalid("mount_encoding"))?;
        checked_absolute(&decoded)?;
        Ok(decoded)
    }

    fn mount(text: &str) -> Result<(String, String), ResourceError> {
        let mut selected = None;
        for line in text.lines() {
            let Some((prefix, suffix)) = line.split_once(" - ") else {
                return Err(ResourceError::Invalid("mountinfo_format"));
            };
            if suffix.split_ascii_whitespace().next() != Some("cgroup2") {
                continue;
            }
            let mut fields = prefix.split_ascii_whitespace();
            let root = fields
                .nth(3)
                .ok_or(ResourceError::Invalid("mountinfo_root"))?;
            let point = fields
                .next()
                .ok_or(ResourceError::Invalid("mountinfo_point"))?;
            let candidate = (decode_mount(point)?, decode_mount(root)?);
            if selected.replace(candidate).is_some() {
                return Err(ResourceError::Unavailable("ambiguous_cgroup2_mount"));
            }
        }
        selected.ok_or(ResourceError::Unavailable("cgroup2_mount_unavailable"))
    }

    fn open_absolute(path: &str, budget: &Budget<'_>) -> Result<Dir, ResourceError> {
        checked_absolute(path)?;
        budget.check()?;
        let mut directory = Dir::open_ambient_dir("/", ambient_authority()).map_err(unavailable)?;
        for part in Path::new(path).components() {
            if let Component::Normal(part) = part {
                budget.check()?;
                directory = directory.open_dir_nofollow(part).map_err(unavailable)?;
            }
        }
        budget.check()?;
        Ok(directory)
    }

    fn namespace(directory: &Dir, budget: &Budget<'_>) -> Result<Option<u64>, ResourceError> {
        budget.check()?;
        let Ok(namespace) = directory.open_dir_nofollow("ns") else {
            return Ok(None);
        };
        budget.check()?;
        if filesystem(&namespace, budget)? != PROC_SUPER_MAGIC {
            return Err(ResourceError::Invalid("resource_proc_filesystem_mismatch"));
        }
        // Read the procfs magic-link name without following it or reading its
        // target. Its bounded `cgroup:[inode]` string is an optional observation.
        let Ok(target) = namespace.read_link("cgroup") else {
            return Ok(None);
        };
        budget.check()?;
        let Some(text) = target.to_str() else {
            return Ok(None);
        };
        if text.len() > 64 {
            return Ok(None);
        }
        Ok(text
            .strip_prefix("cgroup:[")
            .and_then(|v| v.strip_suffix(']'))
            .and_then(|v| v.parse().ok()))
    }

    fn thread_ids(task: &Dir, budget: &Budget<'_>) -> Result<Vec<u32>, ResourceError> {
        let mut ids = Vec::new();
        budget.check()?;
        let mut entries = task.entries().map_err(unavailable)?;
        loop {
            budget.check()?;
            let Some(entry) = entries.next() else { break };
            if ids.len() >= MAX_THREADS {
                return Err(ResourceError::Invalid("thread_observation_bound"));
            }
            let name = entry.map_err(unavailable)?.file_name();
            let tid = name
                .to_str()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|pid| *pid > 0)
                .ok_or(ResourceError::Invalid("proc_task_entry"))?;
            ids.push(tid);
        }
        ids.sort_unstable();
        if ids.is_empty() {
            return Err(ResourceError::Unavailable("process_threads_gone"));
        }
        Ok(ids)
    }

    fn threads(
        directory: &Dir,
        budget: &mut Budget<'_>,
    ) -> Result<Vec<ThreadAffinitySnapshot>, ResourceError> {
        budget.check()?;
        let task = directory.open_dir_nofollow("task").map_err(unavailable)?;
        if filesystem(&task, budget)? != PROC_SUPER_MAGIC {
            return Err(ResourceError::Invalid("resource_proc_filesystem_mismatch"));
        }
        let ids = thread_ids(&task, budget)?;
        let mut snapshots = Vec::new();
        snapshots
            .try_reserve_exact(ids.len())
            .map_err(|_| ResourceError::Invalid("thread_allocation"))?;
        for tid in &ids {
            budget.check()?;
            let directory = task
                .open_dir_nofollow(tid.to_string())
                .map_err(unavailable)?;
            let before = stat(&budget.read(&directory, "stat", MAX_STAT)?, *tid)?;
            let cpu = cpu_list(&budget.read(&directory, "status", MAX_STATUS)?)?;
            let after = stat(&budget.read(&directory, "stat", MAX_STAT)?, *tid)?;
            if before != after {
                return Err(ResourceError::Invalid("thread_identity_changed"));
            }
            snapshots.push(ThreadAffinitySnapshot {
                tid: *tid,
                proc_start_ticks_before: before.start,
                proc_start_ticks_after: after.start,
                cpu_allowed_list: cpu,
            });
        }
        if thread_ids(&task, budget)? != ids {
            return Err(ResourceError::Unavailable("thread_set_changed"));
        }
        Ok(snapshots)
    }

    fn snapshot(
        proc: &Dir,
        pid: u32,
        mount: &(String, String),
        budget: &mut Budget<'_>,
        expected: Option<ExternalProcessIdentity>,
    ) -> Result<ProcessResourceSnapshot, ResourceError> {
        budget.check()?;
        let directory = proc
            .open_dir_nofollow(pid.to_string())
            .map_err(unavailable)?;
        let before = stat(&budget.read(&directory, "stat", MAX_STAT)?, pid)?;
        if let Some(expected) = expected {
            validate_identity(before, expected)?;
        }
        let membership = cgroup(&budget.read(&directory, "cgroup", MAX_CGROUP)?)?;
        let cpu = cpu_list(&budget.read(&directory, "status", MAX_STATUS)?)?;
        let ns = namespace(&directory, budget)?;
        let relative = if mount.1 == "/" {
            membership.trim_start_matches('/')
        } else {
            membership
                .strip_prefix(&mount.1)
                .filter(|v| v.is_empty() || v.starts_with('/'))
                .ok_or(ResourceError::Unavailable(
                    "cgroup_mount_namespace_path_mismatch",
                ))?
                .trim_start_matches('/')
        };
        let resolved = if relative.is_empty() {
            mount.0.clone()
        } else {
            format!("{}/{relative}", mount.0.trim_end_matches('/'))
        };
        let cgroup_directory = open_absolute(&resolved, budget)?;
        if filesystem(&cgroup_directory, budget)? != CGROUP2_SUPER_MAGIC {
            return Err(ResourceError::Invalid(
                "resource_cgroup2_filesystem_mismatch",
            ));
        }
        let limits = CgroupLimitsSnapshot {
            mount_point: mount.0.clone(),
            mount_root: mount.1.clone(),
            resolved_directory: resolved,
            memory_high: limit(&budget.read(&cgroup_directory, "memory.high", 64)?)?,
            memory_max: limit(&budget.read(&cgroup_directory, "memory.max", 64)?)?,
            memory_swap_max: limit(&budget.read(&cgroup_directory, "memory.swap.max", 64)?)?,
            pids_max: limit(&budget.read(&cgroup_directory, "pids.max", 64)?)?,
        };
        let thread_snapshots = threads(&directory, budget)?;
        let after = stat(&budget.read(&directory, "stat", MAX_STAT)?, pid)?;
        if before != after
            || membership != cgroup(&budget.read(&directory, "cgroup", MAX_CGROUP)?)?
            || cpu != cpu_list(&budget.read(&directory, "status", MAX_STATUS)?)?
            || ns != namespace(&directory, budget)?
        {
            return Err(ResourceError::Invalid(
                "resource_process_identity_or_membership_changed",
            ));
        }
        Ok(ProcessResourceSnapshot {
            pid,
            parent_pid: before.parent,
            process_group: before.group,
            proc_start_ticks_before: before.start,
            proc_start_ticks_after: after.start,
            cgroup_v2_membership: membership,
            membership_path_view: "observer_procfs_and_cgroup2_mount_view",
            cgroup_namespace_inode: ns,
            cpu_allowed_list: cpu,
            threads: thread_snapshots,
            thread_observation_scope: "bounded_ready_boundary_thread_snapshot_not_lifetime_enforcement",
            cgroup_limits: limits,
        })
    }

    pub(super) fn observe(
        helper: ExternalProcessIdentity,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<ReadyBoundaryResources, ResourceError> {
        let started = Instant::now();
        // Cooperative controls bracket trusted-kernel filesystem calls; the
        // 500ms window is not a syscall cancellation/enforcement claim.
        let deadline = deadline.min(started + Duration::from_millis(500));
        controlled(deadline, cancel)?;
        if helper.pid == 0
            || helper.process_group == 0
            || helper.proc_start_ticks == 0
            || helper.pid > i32::MAX as u32
            || helper.process_group > i32::MAX as u32
        {
            return Err(ResourceError::Invalid("helper_spawn_identity_mismatch"));
        }
        if helper.pid == std::process::id() {
            return Err(ResourceError::Invalid("helper_identity_is_parent"));
        }
        let mut budget = Budget {
            bytes: 0,
            deadline,
            cancel,
        };
        let proc = open_absolute("/proc", &budget)?;
        if filesystem(&proc, &budget)? != PROC_SUPER_MAGIC {
            return Err(ResourceError::Invalid("resource_proc_filesystem_mismatch"));
        }
        budget.check()?;
        let parent_directory = proc
            .open_dir_nofollow(std::process::id().to_string())
            .map_err(unavailable)?;
        let mount = mount(&budget.read(&parent_directory, "mountinfo", MAX_MOUNTINFO)?)?;
        let observed_unix_us = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| ResourceError::Invalid("observation_clock_before_epoch"))?
                .as_micros(),
        )
        .map_err(|_| ResourceError::Invalid("observation_clock_overflow"))?;
        let parent = snapshot(&proc, std::process::id(), &mount, &mut budget, None)?;
        let helper_snapshot = snapshot(&proc, helper.pid, &mount, &mut budget, Some(helper))?;
        validate_identity(
            ProcIdentity {
                pid: helper_snapshot.pid,
                parent: helper_snapshot.parent_pid,
                group: helper_snapshot.process_group,
                start: helper_snapshot.proc_start_ticks_before,
            },
            helper,
        )?;
        if helper_snapshot.parent_pid != parent.pid {
            return Err(ResourceError::Invalid("helper_parent_identity_mismatch"));
        }
        budget.check()?;
        Ok(ReadyBoundaryResources {
            schema_version: 1,
            domain: "rz-pals-checker-ready-resources/1",
            scope: "linux_ready_boundary_snapshot",
            observed_unix_us,
            observation_elapsed_us: u64::try_from(started.elapsed().as_micros())
                .map_err(|_| ResourceError::Invalid("observation_elapsed_overflow"))?,
            cgroup_membership_equal: parent.cgroup_v2_membership
                == helper_snapshot.cgroup_v2_membership,
            cgroup_namespace_equal: parent
                .cgroup_namespace_inode
                .zip(helper_snapshot.cgroup_namespace_inode)
                .map(|(parent, helper)| parent == helper),
            cpu_allowed_list_equal: parent.cpu_allowed_list == helper_snapshot.cpu_allowed_list,
            parent,
            helper: helper_snapshot,
            additional_allocation: "none_declared_not_an_enforcement_proof",
            read_consistency: "identity_membership_affinity_bracketed_limits_sequential_not_atomic",
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn expired_or_canceled_read_never_probes_the_reader() {
            let mut cursor = std::io::Cursor::new(vec![b'x'; 16]);
            assert_eq!(
                bounded_read(&mut cursor, 8, Instant::now(), &AtomicBool::new(false)),
                Err(ResourceError::Deadline)
            );
            assert_eq!(cursor.position(), 0);
            assert_eq!(
                bounded_read(
                    &mut cursor,
                    8,
                    Instant::now() + Duration::from_secs(1),
                    &AtomicBool::new(true)
                ),
                Err(ResourceError::Deadline)
            );
            assert_eq!(cursor.position(), 0);
        }
        #[test]
        fn proc_start_ticks_and_unavailable_reasons_are_preserved() {
            let text = format!("123 (helper (name)) S 100 123 {}456", "0 ".repeat(16));
            assert_eq!(
                stat(&text, 123),
                Ok(ProcIdentity {
                    pid: 123,
                    parent: 100,
                    group: 123,
                    start: 456
                })
            );
            assert!(stat(&text, 124).is_err());
            assert_eq!(
                unavailable(io::Error::from(io::ErrorKind::PermissionDenied)),
                ResourceError::Unavailable("resource_permission_denied")
            );
            assert_eq!(
                unavailable(io::Error::from(io::ErrorKind::NotFound)),
                ResourceError::Unavailable("resource_file_missing")
            );
        }
        #[test]
        fn bounded_read_never_consumes_more_than_cap_plus_one() {
            let mut cursor = std::io::Cursor::new(vec![b'x'; 100]);
            assert_eq!(
                bounded_read(
                    &mut cursor,
                    8,
                    Instant::now() + Duration::from_secs(1),
                    &AtomicBool::new(false)
                ),
                Err(ResourceError::Invalid("resource_file_byte_bound"))
            );
            assert_eq!(cursor.position(), 9);
        }
        #[test]
        fn parsers_preserve_max_and_reject_ambiguous_or_corrupt_identity() {
            assert_eq!(limit("max\n"), Ok(ObservedLimit::Max));
            assert_eq!(limit("0\n"), Ok(ObservedLimit::Numeric { value: 0 }));
            assert!(limit("unknown").is_err());
            assert_eq!(cgroup("0::/test-group\n"), Ok("/test-group".into()));
            assert!(cgroup("0::/a\n0::/b\n").is_err());
            assert!(cgroup("0::/../escape\n").is_err());
            assert_eq!(cpu_list("Cpus_allowed_list:\t0,2\n"), Ok("0,2".into()));
            assert!(cpu_list("Cpus_allowed_list:\t2,0\n").is_err());
            let actual = ProcIdentity {
                pid: 123,
                parent: 100,
                group: 123,
                start: 456,
            };
            assert!(
                validate_identity(
                    actual,
                    ExternalProcessIdentity {
                        pid: 123,
                        process_group: 123,
                        proc_start_ticks: 457
                    }
                )
                .is_err()
            );
            assert!(
                validate_identity(
                    actual,
                    ExternalProcessIdentity {
                        pid: 123,
                        process_group: 124,
                        proc_start_ticks: 456
                    }
                )
                .is_err()
            );
        }
    }
}
