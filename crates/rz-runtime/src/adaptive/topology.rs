//! Read-only CPU placement evidence and declared thread accounting.
//!
//! This boundary never installs affinity, cgroups, Job Objects, or ORT options.
//! A coordinator must run the probe outside its event loop and keep requested
//! resources separate from these sequential, startup/ready-boundary facts.
//! Linux uses bounded procfs/sysfs reads. Windows native callers supply a typed
//! kernel32 readback, allowing this crate to retain `forbid(unsafe_code)`.

use std::collections::BTreeSet;
use std::fmt;
#[cfg(target_os = "linux")]
use std::io::Read;
#[cfg(any(target_os = "linux", test))]
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub const CPU_PROBE_MAX_LOGICAL_CPUS: usize = 4096;
pub const CPU_PROBE_MAX_THREADS: usize = 1024;
#[cfg(any(target_os = "linux", test))]
const PROBE_FILE_MAX_BYTES: usize = 64 * 1024;
#[cfg(any(target_os = "linux", test))]
const CGROUP_MAX_ANCESTORS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LogicalCpu {
    pub group: u16,
    pub index: u16,
}
impl LogicalCpu {
    /// Windows group-local CPU IDs are flattened only at the manifest edge.
    /// Linux IDs are represented in group zero and are not truncated to 64.
    pub fn manifest_id(self) -> u32 {
        u32::from(self.group) * 64 + u32::from(self.index)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuProbeMethod {
    LinuxProcThreadStatus,
    LinuxProcProcessStatus,
    LinuxSysfsSiblings,
    LinuxCgroupV2Direct,
    LinuxCgroupV2VisibleAncestors,
    WindowsProcessAndThreadGroupAffinity,
    WindowsLogicalProcessorInformationEx,
    WindowsImmediateJobObject,
    WorkerReadyBoundary,
    OrtAppliedSessionSettings,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuProbeUnknown {
    UnsupportedPlatform,
    ReadUnavailable,
    InvalidFormat,
    BudgetExceeded,
    NamespaceOrMountAmbiguous,
    ChangedDuringProbe,
    HiddenAncestors,
    WindowsJobHierarchyUnavailable,
    WindowsGroupAffinityAmbiguous,
    UnobservedThread,
    UnobservedOrtSettings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuReadback<T> {
    Known { value: T, method: CpuProbeMethod },
    Unknown(CpuProbeUnknown),
}
impl<T> CpuReadback<T> {
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Known { value, .. } => Some(value),
            Self::Unknown(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalCpuCore {
    /// Includes observed online siblings outside the caller's affinity. They
    /// must not disappear merely because they are unavailable to this process.
    pub logical_cpus: Vec<LogicalCpu>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuQuotaReadback {
    Unlimited,
    /// An exact rational quota; rounding is never used for equality admission.
    Limited {
        quota_us: u64,
        period_us: u64,
    },
    /// Windows hard cap of the immediate Job Object. This is not a native-node
    /// budget or an assertion about the effective nested Job hierarchy.
    WindowsHardCap {
        rate_per_10000: u16,
    },
}
impl CpuQuotaReadback {
    pub fn equals_millicpus(self, requested: u32) -> bool {
        match self {
            Self::Limited {
                quota_us,
                period_us,
            } => {
                quota_us > 0
                    && period_us > 0
                    && requested > 0
                    && u128::from(quota_us) * 1000 == u128::from(period_us) * u128::from(requested)
            }
            Self::Unlimited | Self::WindowsHardCap { .. } => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuHostKind {
    /// The caller explicitly registered a local environment. The probe does
    /// not infer this from hostname, a GPU name, or the presence of sysfs.
    RegisteredLocal,
    CloudOrUnknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuAffinityScope {
    Process,
    /// `/proc/self/status` describes the Linux thread-group leader. It is not
    /// an aggregate process mask and can differ from worker/thread affinity.
    ThreadGroupLeader,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostCpuSnapshot {
    pub process_id: u32,
    pub caller_thread_id: CpuReadback<u64>,
    pub process_affinity: CpuReadback<Vec<LogicalCpu>>,
    pub process_affinity_scope: CpuAffinityScope,
    pub caller_affinity: CpuReadback<Vec<LogicalCpu>>,
    pub physical_cores: CpuReadback<Vec<PhysicalCpuCore>>,
    pub direct_cpu_quota: CpuReadback<CpuQuotaReadback>,
    /// The tightest readable quota in this cgroup namespace. Even a known
    /// value cannot assert what an inaccessible host ancestor enforces.
    pub visible_ancestor_cpu_quota: CpuReadback<CpuQuotaReadback>,
    pub ancestor_scope: CpuProbeUnknown,
    pub sequential_snapshot: bool,
}
impl HostCpuSnapshot {
    fn unknown() -> Self {
        Self {
            process_id: std::process::id(),
            caller_thread_id: CpuReadback::Unknown(CpuProbeUnknown::UnsupportedPlatform),
            process_affinity: CpuReadback::Unknown(CpuProbeUnknown::UnsupportedPlatform),
            process_affinity_scope: CpuAffinityScope::Unknown,
            caller_affinity: CpuReadback::Unknown(CpuProbeUnknown::UnsupportedPlatform),
            physical_cores: CpuReadback::Unknown(CpuProbeUnknown::UnsupportedPlatform),
            direct_cpu_quota: CpuReadback::Unknown(CpuProbeUnknown::UnsupportedPlatform),
            visible_ancestor_cpu_quota: CpuReadback::Unknown(CpuProbeUnknown::UnsupportedPlatform),
            ancestor_scope: CpuProbeUnknown::HiddenAncestors,
            sequential_snapshot: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlPlacement {
    SharedControl,
    DedicatedControl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrtThreadPolicy {
    pub intra_threads: u16,
    pub inter_threads: u16,
    pub intra_spinning: bool,
    pub inter_spinning: bool,
}
impl Default for OrtThreadPolicy {
    fn default() -> Self {
        Self {
            intra_threads: 1,
            inter_threads: 1,
            intra_spinning: false,
            inter_spinning: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveCpuResourcePlan {
    pub profile_id: String,
    pub placement: ControlPlacement,
    pub host_kind: CpuHostKind,
    pub requested_millicpus: u32,
    pub allowed_cpus: Vec<LogicalCpu>,
    pub manager_cpu: Option<LogicalCpu>,
    pub worker_cpus: Vec<LogicalCpu>,
    pub own_cpu_workers: u8,
    pub manager_threads: u16,
    /// The NN physical worker/caller is counted once, not once per role.
    pub native_caller_threads: u16,
    pub external_engine_threads: u16,
    pub ort: OrtThreadPolicy,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
}
impl Default for AdaptiveCpuResourcePlan {
    fn default() -> Self {
        Self {
            profile_id: "shared-control-cpu2-v1".into(),
            placement: ControlPlacement::SharedControl,
            host_kind: CpuHostKind::CloudOrUnknown,
            requested_millicpus: 2000,
            allowed_cpus: Vec::new(),
            manager_cpu: None,
            worker_cpus: Vec::new(),
            own_cpu_workers: 1,
            manager_threads: 1,
            native_caller_threads: 1,
            external_engine_threads: 1,
            ort: OrtThreadPolicy::default(),
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
        }
    }
}
impl AdaptiveCpuResourcePlan {
    pub fn validate(&self) -> Result<(), CpuResourceError> {
        if self.profile_id.is_empty()
            || self.profile_id.len() > 128
            || !self
                .profile_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            || !(1000..=8000).contains(&self.requested_millicpus)
            || !matches!(self.own_cpu_workers, 1 | 2 | 4)
            || self.manager_threads != 1
            || self.native_caller_threads != 1
            || self.external_engine_threads > 64
            || self.ort != OrtThreadPolicy::default()
            || self.memory_high_bytes != 6 << 30
            || self.memory_max_bytes != 12 << 30
            || !valid_cpu_set(&self.allowed_cpus, true)
            || !valid_cpu_set(&self.worker_cpus, true)
            || self
                .worker_cpus
                .iter()
                .any(|c| !self.allowed_cpus.contains(c))
        {
            return Err(CpuResourceError::InvalidDeclaration);
        }
        if self.own_cpu_workers != 1
            && (self.requested_millicpus <= 2000 || self.profile_id == "shared-control-cpu2-v1")
        {
            return Err(CpuResourceError::SeparateWorkerProfileRequired);
        }
        match self.placement {
            ControlPlacement::SharedControl if self.manager_cpu.is_some() => {
                Err(CpuResourceError::InvalidDeclaration)
            }
            ControlPlacement::DedicatedControl
                if self.host_kind != CpuHostKind::RegisteredLocal
                    || self
                        .manager_cpu
                        .is_none_or(|c| !self.allowed_cpus.contains(&c))
                    || self.worker_cpus.len() < usize::from(self.own_cpu_workers) =>
            {
                Err(CpuResourceError::DedicatedEvidenceRequired)
            }
            _ => Ok(()),
        }
    }

    /// Potential runnable threads, not CPU capacity or observed OS inventory.
    /// Intra/inter=1 add no extra pool threads to the single caller. A caller
    /// must still report runtime-created helper threads separately at ready.
    pub fn declared_runnable_threads(&self) -> u32 {
        u32::from(self.manager_threads)
            + u32::from(self.native_caller_threads)
            + u32::from(self.own_cpu_workers)
            + u32::from(self.external_engine_threads)
            + u32::from(self.ort.intra_threads.saturating_sub(1))
            + u32::from(self.ort.inter_threads.saturating_sub(1))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CpuActor {
    Manager,
    NativeCaller,
    OwnWorker(u8),
    ExternalEngine(u16),
    RuntimeHelper(u16),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuThreadReadback {
    pub actor: CpuActor,
    pub thread_id: u64,
    pub affinity: CpuReadback<Vec<LogicalCpu>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuReadyReadback {
    pub threads: Vec<CpuThreadReadback>,
    /// A bounded complete ready-boundary thread inventory, not lifetime proof.
    pub inventory_complete: bool,
    pub ort_applied_settings: CpuReadback<OrtThreadPolicy>,
}

/// Called by the thread whose affinity is being attested, outside search work.
/// This does not identify another worker from its name or requested placement.
pub fn observe_current_cpu_thread(actor: CpuActor) -> Result<CpuThreadReadback, CpuResourceError> {
    #[cfg(target_os = "linux")]
    {
        let observed = rz_platform::observe_linux_thread_affinity()
            .map_err(|_| CpuResourceError::InvalidReadback)?;
        Ok(CpuThreadReadback {
            actor,
            thread_id: observed.thread_id,
            affinity: CpuReadback::Known {
                value: observed
                    .cpu_ids
                    .into_iter()
                    .map(|index| LogicalCpu { group: 0, index })
                    .collect(),
                method: CpuProbeMethod::LinuxProcThreadStatus,
            },
        })
    }
    #[cfg(target_os = "windows")]
    {
        let observed = rz_platform::observe_windows_thread_affinity()
            .map_err(|_| CpuResourceError::InvalidReadback)?;
        let affinity = map_windows_masks(CpuReadback::Known {
            value: vec![observed.group_mask],
            method: CpuProbeMethod::WindowsProcessAndThreadGroupAffinity,
        })?;
        Ok(CpuThreadReadback {
            actor,
            thread_id: observed.thread_id,
            affinity,
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = actor;
        Err(CpuResourceError::DedicatedEvidenceRequired)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuResourceAdmission {
    pub placement: ControlPlacement,
    pub declared_runnable_threads: u32,
    pub observed_thread_count: usize,
    pub direct_quota_matches_request: Option<bool>,
    pub visible_quota_matches_request: Option<bool>,
    pub ort_settings_match: Option<bool>,
    pub ready_inventory_complete: bool,
    pub dedicated_smt_exclusion_verified: bool,
    /// Never infer host-wide enforcement or performance eligibility here.
    pub resource_settings_installed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuResourceError {
    InvalidDeclaration,
    SeparateWorkerProfileRequired,
    DedicatedEvidenceRequired,
    InvalidReadback,
    AffinityMismatch,
    ManagerSmtOverlap,
    OrtSettingsMismatch,
    ProbeDeadline,
    ProbeCanceled,
    ProbeBudgetExceeded,
}
impl fmt::Display for CpuResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "adaptive CPU resource error: {self:?}")
    }
}
impl std::error::Error for CpuResourceError {}

/// Validate actual ready-boundary placement against the declaration. Unknown
/// evidence permits SharedControl functionality but never DedicatedControl.
pub fn admit_cpu_placement(
    plan: &AdaptiveCpuResourcePlan,
    host: &HostCpuSnapshot,
    ready: &CpuReadyReadback,
) -> Result<CpuResourceAdmission, CpuResourceError> {
    plan.validate()?;
    if ready.threads.len() > CPU_PROBE_MAX_THREADS {
        return Err(CpuResourceError::InvalidReadback);
    }
    let mut actors = BTreeSet::new();
    let mut tids = BTreeSet::new();
    for thread in &ready.threads {
        if thread.thread_id == 0 || !actors.insert(thread.actor) || !tids.insert(thread.thread_id) {
            return Err(CpuResourceError::InvalidReadback);
        }
        if matches!(thread.actor, CpuActor::OwnWorker(id) if id >= plan.own_cpu_workers)
            || matches!(thread.actor, CpuActor::ExternalEngine(id) if id >= plan.external_engine_threads)
        {
            return Err(CpuResourceError::InvalidReadback);
        }
        if let Some(cpus) = thread.affinity.value() {
            if !valid_cpu_set(cpus, false) {
                return Err(CpuResourceError::InvalidReadback);
            }
            if !plan.allowed_cpus.is_empty() && cpus.iter().any(|c| !plan.allowed_cpus.contains(c))
            {
                return Err(CpuResourceError::AffinityMismatch);
            }
        }
    }
    if let Some(cpus) = host.process_affinity.value() {
        if !valid_cpu_set(cpus, false) {
            return Err(CpuResourceError::InvalidReadback);
        }
        if !plan.allowed_cpus.is_empty()
            && (cpus.iter().any(|c| !plan.allowed_cpus.contains(c))
                || (host.process_affinity_scope == CpuAffinityScope::Process
                    && cpus != &plan.allowed_cpus))
        {
            return Err(CpuResourceError::AffinityMismatch);
        }
    }
    for quota in [&host.direct_cpu_quota, &host.visible_ancestor_cpu_quota] {
        if quota.value().is_some_and(|q| !valid_quota(*q)) {
            return Err(CpuResourceError::InvalidReadback);
        }
    }
    let ort_settings_match = ready.ort_applied_settings.value().map(|v| *v == plan.ort);
    if ort_settings_match == Some(false) {
        return Err(CpuResourceError::OrtSettingsMismatch);
    }
    let mut admission = CpuResourceAdmission {
        placement: plan.placement,
        declared_runnable_threads: plan.declared_runnable_threads(),
        observed_thread_count: ready.threads.len(),
        direct_quota_matches_request: host
            .direct_cpu_quota
            .value()
            .map(|q| q.equals_millicpus(plan.requested_millicpus)),
        visible_quota_matches_request: host
            .visible_ancestor_cpu_quota
            .value()
            .map(|q| q.equals_millicpus(plan.requested_millicpus)),
        ort_settings_match,
        ready_inventory_complete: ready.inventory_complete,
        dedicated_smt_exclusion_verified: false,
        resource_settings_installed: false,
    };
    if plan.placement == ControlPlacement::SharedControl {
        return Ok(admission);
    }
    if !ready.inventory_complete
        || ort_settings_match != Some(true)
        || host.process_affinity.value().is_none()
        || host.caller_affinity.value().is_none()
    {
        return Err(CpuResourceError::DedicatedEvidenceRequired);
    }
    let cores = host
        .physical_cores
        .value()
        .ok_or(CpuResourceError::DedicatedEvidenceRequired)?;
    validate_physical_cores(cores)?;
    let manager = plan
        .manager_cpu
        .ok_or(CpuResourceError::DedicatedEvidenceRequired)?;
    let siblings = &cores
        .iter()
        .find(|c| c.logical_cpus.contains(&manager))
        .ok_or(CpuResourceError::DedicatedEvidenceRequired)?
        .logical_cpus;
    let manager_readback = ready
        .threads
        .iter()
        .find(|t| t.actor == CpuActor::Manager)
        .and_then(|t| t.affinity.value())
        .ok_or(CpuResourceError::DedicatedEvidenceRequired)?;
    if manager_readback.as_slice() != [manager] {
        return Err(CpuResourceError::AffinityMismatch);
    }
    for worker in 0..plan.own_cpu_workers {
        if !actors.contains(&CpuActor::OwnWorker(worker)) {
            return Err(CpuResourceError::DedicatedEvidenceRequired);
        }
    }
    if !actors.contains(&CpuActor::NativeCaller)
        || (0..plan.external_engine_threads).any(|i| !actors.contains(&CpuActor::ExternalEngine(i)))
    {
        return Err(CpuResourceError::DedicatedEvidenceRequired);
    }
    for thread in &ready.threads {
        let cpus = thread
            .affinity
            .value()
            .ok_or(CpuResourceError::DedicatedEvidenceRequired)?;
        if thread.actor != CpuActor::Manager && cpus.iter().any(|c| siblings.contains(c)) {
            return Err(CpuResourceError::ManagerSmtOverlap);
        }
        if matches!(thread.actor, CpuActor::OwnWorker(_))
            && cpus.iter().any(|c| !plan.worker_cpus.contains(c))
        {
            return Err(CpuResourceError::AffinityMismatch);
        }
    }
    admission.dedicated_smt_exclusion_verified = true;
    Ok(admission)
}

fn valid_cpu_set(cpus: &[LogicalCpu], empty_allowed: bool) -> bool {
    (empty_allowed || !cpus.is_empty())
        && cpus.len() <= CPU_PROBE_MAX_LOGICAL_CPUS
        && cpus.iter().all(|c| {
            (c.group == 0 && usize::from(c.index) < CPU_PROBE_MAX_LOGICAL_CPUS)
                || (c.group < 64 && c.index < 64)
        })
        && cpus.windows(2).all(|w| w[0] < w[1])
        && cpus
            .iter()
            .map(|c| c.manifest_id())
            .collect::<BTreeSet<_>>()
            .len()
            == cpus.len()
}

fn valid_quota(quota: CpuQuotaReadback) -> bool {
    match quota {
        CpuQuotaReadback::Unlimited => true,
        CpuQuotaReadback::Limited {
            quota_us,
            period_us,
        } => quota_us > 0 && period_us > 0,
        CpuQuotaReadback::WindowsHardCap { rate_per_10000 } => {
            (1..=10000).contains(&rate_per_10000)
        }
    }
}

fn validate_physical_cores(cores: &[PhysicalCpuCore]) -> Result<(), CpuResourceError> {
    if cores.is_empty() || cores.len() > CPU_PROBE_MAX_LOGICAL_CPUS {
        return Err(CpuResourceError::InvalidReadback);
    }
    let mut all = BTreeSet::new();
    let mut flattened = BTreeSet::new();
    for core in cores {
        if !valid_cpu_set(&core.logical_cpus, false)
            || core
                .logical_cpus
                .iter()
                .any(|c| !all.insert(*c) || !flattened.insert(c.manifest_id()))
        {
            return Err(CpuResourceError::InvalidReadback);
        }
    }
    Ok(())
}

/// Windows OS adapters use GetProcessGroupAffinity/GetProcessAffinityMask,
/// GetThreadGroupAffinity, and GetLogicalProcessorInformationEx. All rows must
/// come from the requested process/thread and be bracketed against identity
/// change. A multi-group process without complete process masks stays unknown.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowsCpuApiReadback {
    pub process_id: u32,
    pub caller_thread_id: u64,
    pub identity_stable: bool,
    pub process_group_masks: CpuReadback<Vec<(u16, u64)>>,
    pub caller_group_mask: CpuReadback<(u16, u64)>,
    pub physical_core_group_masks: CpuReadback<Vec<Vec<(u16, u64)>>>,
    pub immediate_job_cpu_quota: CpuReadback<CpuQuotaReadback>,
}

pub fn windows_cpu_snapshot(
    raw: WindowsCpuApiReadback,
) -> Result<HostCpuSnapshot, CpuResourceError> {
    if raw.process_id == 0 || raw.caller_thread_id == 0 {
        return Err(CpuResourceError::InvalidReadback);
    }
    if !raw.identity_stable {
        let mut snapshot = HostCpuSnapshot::unknown();
        snapshot.process_id = raw.process_id;
        snapshot.process_affinity = CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe);
        snapshot.caller_affinity = CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe);
        snapshot.physical_cores = CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe);
        return Ok(snapshot);
    }
    let process_affinity = map_windows_masks(raw.process_group_masks)?;
    let caller_affinity = match raw.caller_group_mask {
        CpuReadback::Known { value, .. } => map_windows_masks(CpuReadback::Known {
            value: vec![value],
            method: CpuProbeMethod::WindowsProcessAndThreadGroupAffinity,
        })?,
        CpuReadback::Unknown(reason) => CpuReadback::Unknown(reason),
    };
    let physical_cores = match raw.physical_core_group_masks {
        CpuReadback::Known { value, .. } => {
            if value.len() > CPU_PROBE_MAX_LOGICAL_CPUS {
                return Err(CpuResourceError::InvalidReadback);
            }
            let mut cores = Vec::with_capacity(value.len());
            for masks in value {
                let logical_cpus = map_windows_masks(CpuReadback::Known {
                    value: masks,
                    method: CpuProbeMethod::WindowsLogicalProcessorInformationEx,
                })?
                .value()
                .cloned()
                .ok_or(CpuResourceError::InvalidReadback)?;
                cores.push(PhysicalCpuCore { logical_cpus });
            }
            validate_physical_cores(&cores)?;
            CpuReadback::Known {
                value: cores,
                method: CpuProbeMethod::WindowsLogicalProcessorInformationEx,
            }
        }
        CpuReadback::Unknown(reason) => CpuReadback::Unknown(reason),
    };
    if raw
        .immediate_job_cpu_quota
        .value()
        .is_some_and(|q| !valid_quota(*q))
    {
        return Err(CpuResourceError::InvalidReadback);
    }
    Ok(HostCpuSnapshot {
        process_id: raw.process_id,
        caller_thread_id: CpuReadback::Known {
            value: raw.caller_thread_id,
            method: CpuProbeMethod::WindowsProcessAndThreadGroupAffinity,
        },
        process_affinity,
        process_affinity_scope: CpuAffinityScope::Process,
        caller_affinity,
        physical_cores,
        direct_cpu_quota: raw.immediate_job_cpu_quota,
        visible_ancestor_cpu_quota: CpuReadback::Unknown(
            CpuProbeUnknown::WindowsJobHierarchyUnavailable,
        ),
        ancestor_scope: CpuProbeUnknown::WindowsJobHierarchyUnavailable,
        sequential_snapshot: true,
    })
}

fn map_windows_masks(
    raw: CpuReadback<Vec<(u16, u64)>>,
) -> Result<CpuReadback<Vec<LogicalCpu>>, CpuResourceError> {
    let CpuReadback::Known {
        value: masks,
        method,
    } = raw
    else {
        return Ok(match raw {
            CpuReadback::Unknown(reason) => CpuReadback::Unknown(reason),
            _ => unreachable!(),
        });
    };
    if masks.is_empty() || masks.len() > 64 {
        return Err(CpuResourceError::InvalidReadback);
    }
    let mut groups = BTreeSet::new();
    let mut cpus = Vec::new();
    for (group, mask) in masks {
        if group >= 64 || mask == 0 || !groups.insert(group) {
            return Err(CpuResourceError::InvalidReadback);
        }
        for index in 0..64 {
            if mask & (1u64 << index) != 0 {
                cpus.push(LogicalCpu { group, index });
            }
        }
    }
    cpus.sort_unstable();
    Ok(CpuReadback::Known {
        value: cpus,
        method,
    })
}

/// Read only fixed OS metadata paths. This function performs no inference and
/// never enumerates arbitrary user files, environment, or credential stores.
#[cfg(target_os = "linux")]
pub fn observe_current_cpu() -> HostCpuSnapshot {
    observe_linux_cpu(|path| {
        let file = std::fs::File::open(path).ok()?;
        let mut bytes = Vec::new();
        file.take((PROBE_FILE_MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .ok()?;
        if bytes.len() > PROBE_FILE_MAX_BYTES {
            return None;
        }
        String::from_utf8(bytes).ok()
    })
}

/// The runtime remains safe Rust. A Windows native embedding must call
/// `windows_cpu_snapshot` with its small OS-boundary API observation; this
/// fallback explicitly reports unavailable facts instead of inventing CPUs.
#[cfg(target_os = "windows")]
pub fn observe_current_cpu() -> HostCpuSnapshot {
    let raw = rz_platform::observe_windows_cpu();
    let map_error = |error: rz_platform::PlatformCpuError| match error {
        rz_platform::PlatformCpuError::AmbiguousProcessGroups => {
            CpuProbeUnknown::WindowsGroupAffinityAmbiguous
        }
        rz_platform::PlatformCpuError::ChangedDuringProbe => CpuProbeUnknown::ChangedDuringProbe,
        rz_platform::PlatformCpuError::NativeBufferBudget => CpuProbeUnknown::BudgetExceeded,
        rz_platform::PlatformCpuError::InvalidNativeBuffer
        | rz_platform::PlatformCpuError::InvalidText => CpuProbeUnknown::InvalidFormat,
        _ => CpuProbeUnknown::ReadUnavailable,
    };
    let wrap_masks = |value: Result<Vec<(u16, u64)>, rz_platform::PlatformCpuError>| match value {
        Ok(value) => CpuReadback::Known {
            value,
            method: CpuProbeMethod::WindowsProcessAndThreadGroupAffinity,
        },
        Err(error) => CpuReadback::Unknown(map_error(error)),
    };
    let mapped = WindowsCpuApiReadback {
        process_id: raw.process_id,
        caller_thread_id: raw.caller_thread_id,
        identity_stable: raw.identity_stable,
        process_group_masks: wrap_masks(raw.process_group_masks),
        caller_group_mask: match raw.caller_group_mask {
            Ok(value) => CpuReadback::Known {
                value,
                method: CpuProbeMethod::WindowsProcessAndThreadGroupAffinity,
            },
            Err(error) => CpuReadback::Unknown(map_error(error)),
        },
        physical_core_group_masks: match raw.physical_core_group_masks {
            Ok(value) => CpuReadback::Known {
                value,
                method: CpuProbeMethod::WindowsLogicalProcessorInformationEx,
            },
            Err(error) => CpuReadback::Unknown(map_error(error)),
        },
        immediate_job_cpu_quota: match raw.immediate_job_cpu_quota {
            Ok(rz_platform::WindowsDirectCpuQuota::Unlimited) => CpuReadback::Known {
                value: CpuQuotaReadback::Unlimited,
                method: CpuProbeMethod::WindowsImmediateJobObject,
            },
            Ok(rz_platform::WindowsDirectCpuQuota::HardCap { rate_per_10000 }) => {
                CpuReadback::Known {
                    value: CpuQuotaReadback::WindowsHardCap { rate_per_10000 },
                    method: CpuProbeMethod::WindowsImmediateJobObject,
                }
            }
            Err(error) => CpuReadback::Unknown(map_error(error)),
        },
    };
    windows_cpu_snapshot(mapped).unwrap_or_else(|_| {
        let mut snapshot = HostCpuSnapshot::unknown();
        snapshot.process_affinity = CpuReadback::Unknown(CpuProbeUnknown::InvalidFormat);
        snapshot.caller_affinity = CpuReadback::Unknown(CpuProbeUnknown::InvalidFormat);
        snapshot.physical_cores = CpuReadback::Unknown(CpuProbeUnknown::InvalidFormat);
        snapshot
    })
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn observe_current_cpu() -> HostCpuSnapshot {
    HostCpuSnapshot::unknown()
}

/// Reuses the same read-only metadata probe with cooperative caller controls.
/// Linux brackets every read/chunk and topology/ancestor traversal. Windows
/// brackets the fixed native call before/after; neither branch interrupts an
/// in-progress kernel syscall or installs a resource policy.
pub fn observe_current_cpu_until(
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<HostCpuSnapshot, CpuResourceError> {
    let control = || cpu_probe_control(deadline, cancel);
    control()?;
    #[cfg(target_os = "linux")]
    {
        use std::cell::Cell;
        const TOTAL_READ_MAX: usize = 8 * 1024 * 1024;
        let total_bytes = Cell::new(0usize);
        let read_failure = Cell::new(None);
        let read = |path: &Path| {
            let result = (|| -> Result<Option<String>, CpuResourceError> {
                control()?;
                let Ok(mut file) = std::fs::File::open(path) else {
                    control()?;
                    return Ok(None);
                };
                control()?;
                let mut bytes = Vec::new();
                let mut chunk = [0; 1024];
                loop {
                    control()?;
                    let remaining = PROBE_FILE_MAX_BYTES + 1 - bytes.len();
                    let limit = chunk.len().min(remaining);
                    let count = match file.read(&mut chunk[..limit]) {
                        Ok(count) => count,
                        Err(_) => {
                            control()?;
                            return Ok(None);
                        }
                    };
                    control()?;
                    if count == 0 {
                        break;
                    }
                    let next = total_bytes
                        .get()
                        .checked_add(count)
                        .filter(|value| *value <= TOTAL_READ_MAX)
                        .ok_or(CpuResourceError::ProbeBudgetExceeded)?;
                    total_bytes.set(next);
                    bytes.extend_from_slice(&chunk[..count]);
                    if bytes.len() > PROBE_FILE_MAX_BYTES {
                        return Err(CpuResourceError::ProbeBudgetExceeded);
                    }
                }
                control()?;
                Ok(String::from_utf8(bytes).ok())
            })();
            match result {
                Ok(value) => value,
                Err(error) => {
                    read_failure.set(Some(error));
                    None
                }
            }
        };
        let guarded_control = || {
            if let Some(error) = read_failure.get() {
                Err(error)
            } else {
                control()
            }
        };
        observe_linux_cpu_controlled(read, guarded_control)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let value = observe_current_cpu();
        control()?;
        Ok(value)
    }
}

fn cpu_probe_control(deadline: Instant, cancel: &AtomicBool) -> Result<(), CpuResourceError> {
    if cancel.load(Ordering::Acquire) {
        Err(CpuResourceError::ProbeCanceled)
    } else if Instant::now() >= deadline {
        Err(CpuResourceError::ProbeDeadline)
    } else {
        Ok(())
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_cpus(raw: &str) -> Option<Vec<LogicalCpu>> {
    if raw.is_empty() || raw.len() > PROBE_FILE_MAX_BYTES {
        return None;
    }
    let mut cpus = BTreeSet::new();
    for field in raw.trim().split(',') {
        let parse = |s: &str| -> Option<u16> {
            if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let n = s.parse::<u16>().ok()?;
            (usize::from(n) < CPU_PROBE_MAX_LOGICAL_CPUS).then_some(n)
        };
        let (lo, hi) = if let Some((lo, hi)) = field.split_once('-') {
            (parse(lo)?, parse(hi)?)
        } else {
            let n = parse(field)?;
            (n, n)
        };
        if hi < lo {
            return None;
        }
        for index in lo..=hi {
            if !cpus.insert(LogicalCpu { group: 0, index }) {
                return None;
            }
        }
    }
    (!cpus.is_empty()).then(|| cpus.into_iter().collect())
}

#[cfg(any(target_os = "linux", test))]
fn parse_cpu_max(raw: &str) -> Option<CpuQuotaReadback> {
    let mut words = raw.split_ascii_whitespace();
    let quota = words.next()?;
    let period = words.next()?;
    if words.next().is_some() || !period.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let period_us = period.parse::<u64>().ok().filter(|v| *v > 0)?;
    if quota == "max" {
        return Some(CpuQuotaReadback::Unlimited);
    }
    if !quota.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let quota_us = quota.parse::<u64>().ok().filter(|v| *v > 0)?;
    Some(CpuQuotaReadback::Limited {
        quota_us,
        period_us,
    })
}

#[cfg(any(target_os = "linux", test))]
fn parse_status(raw: &str, key: &str) -> Option<String> {
    if raw.len() > PROBE_FILE_MAX_BYTES {
        return None;
    }
    let mut rows = raw.lines().filter_map(|line| line.strip_prefix(key));
    let value = rows.next()?.trim().to_owned();
    rows.next().is_none().then_some(value)
}

#[cfg(any(target_os = "linux", test))]
fn safe_absolute_path(raw: &str) -> Option<PathBuf> {
    if !raw.starts_with('/') || raw.len() > 4096 || raw.bytes().any(|b| b.is_ascii_control()) {
        return None;
    }
    let p = Path::new(raw);
    if p.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::CurDir | Component::Prefix(_)
        )
    }) {
        return None;
    }
    // Rust Path components normalize '.', so reject it before normalization.
    if raw.split('/').any(|c| c == "." || c == "..") {
        return None;
    }
    Some(p.to_path_buf())
}

#[cfg(any(target_os = "linux", test))]
fn unified_cgroup(raw: &str) -> Option<PathBuf> {
    if raw.len() > PROBE_FILE_MAX_BYTES {
        return None;
    }
    let mut matches = raw.lines().filter_map(|line| line.strip_prefix("0::"));
    let p = safe_absolute_path(matches.next()?)?;
    matches.next().is_none().then_some(p)
}

#[cfg(any(target_os = "linux", test))]
fn decode_mount_field(raw: &str) -> Option<String> {
    let mut s = String::new();
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            let code = bytes.get(i + 1..i + 4)?;
            let ch = match code {
                b"040" => ' ',
                b"011" => '\t',
                b"012" => '\n',
                b"134" => '\\',
                _ => return None,
            };
            s.push(ch);
            i += 4;
        } else {
            if !bytes[i].is_ascii() {
                return None;
            }
            s.push(bytes[i] as char);
            i += 1;
        }
    }
    Some(s)
}

#[cfg(any(target_os = "linux", test))]
fn cgroup_mount(raw: &str, member: &Path) -> Option<(PathBuf, PathBuf)> {
    if raw.len() > PROBE_FILE_MAX_BYTES {
        return None;
    }
    let mut selected = None;
    for line in raw.lines() {
        let (left, right) = line.split_once(" - ")?;
        if right.split_ascii_whitespace().next() != Some("cgroup2") {
            continue;
        }
        let fields: Vec<_> = left.split_ascii_whitespace().collect();
        if fields.len() < 6 {
            return None;
        }
        let root = safe_absolute_path(&decode_mount_field(fields[3])?)?;
        let mount = safe_absolute_path(&decode_mount_field(fields[4])?)?;
        if let Ok(relative) = member.strip_prefix(&root) {
            if selected.is_some() {
                return None;
            }
            selected = Some((mount.join(relative), mount));
        }
    }
    selected
}

#[cfg(any(target_os = "linux", test))]
fn tighter(a: CpuQuotaReadback, b: CpuQuotaReadback) -> CpuQuotaReadback {
    match (a, b) {
        (CpuQuotaReadback::Unlimited, b) => b,
        (a, CpuQuotaReadback::Unlimited) => a,
        (
            CpuQuotaReadback::Limited {
                quota_us: aq,
                period_us: ap,
            },
            CpuQuotaReadback::Limited {
                quota_us: bq,
                period_us: bp,
            },
        ) => {
            if u128::from(aq) * u128::from(bp) <= u128::from(bq) * u128::from(ap) {
                a
            } else {
                b
            }
        }
        _ => a,
    }
}

#[cfg(any(target_os = "linux", test))]
fn observe_linux_cpu(read: impl Fn(&Path) -> Option<String>) -> HostCpuSnapshot {
    observe_linux_cpu_controlled(read, || Ok(()))
        .expect("the compatibility metadata probe has no failing control callback")
}

#[cfg(any(target_os = "linux", test))]
fn observe_linux_cpu_controlled(
    raw_read: impl Fn(&Path) -> Option<String>,
    control: impl Fn() -> Result<(), CpuResourceError>,
) -> Result<HostCpuSnapshot, CpuResourceError> {
    let read = |path: &Path| {
        control()?;
        let value = raw_read(path);
        control()?;
        Ok::<_, CpuResourceError>(value)
    };
    control()?;
    let mut snapshot = HostCpuSnapshot::unknown();
    let process_before = read(Path::new("/proc/self/status"))?;
    let thread_before = read(Path::new("/proc/thread-self/status"))?;
    snapshot.process_affinity_scope = CpuAffinityScope::ThreadGroupLeader;
    let online_before = read(Path::new("/sys/devices/system/cpu/online"))?;
    let mut observed_files: Vec<(PathBuf, Option<String>)> = Vec::new();
    let process_cpus = process_before
        .as_deref()
        .and_then(|s| parse_status(s, "Cpus_allowed_list:"))
        .and_then(|s| parse_linux_cpus(&s));
    let thread_cpus = thread_before
        .as_deref()
        .and_then(|s| parse_status(s, "Cpus_allowed_list:"))
        .and_then(|s| parse_linux_cpus(&s));
    snapshot.process_affinity = process_cpus.clone().map_or(
        CpuReadback::Unknown(CpuProbeUnknown::ReadUnavailable),
        |value| CpuReadback::Known {
            value,
            method: CpuProbeMethod::LinuxProcProcessStatus,
        },
    );
    snapshot.caller_affinity = thread_cpus.clone().map_or(
        CpuReadback::Unknown(CpuProbeUnknown::ReadUnavailable),
        |value| CpuReadback::Known {
            value,
            method: CpuProbeMethod::LinuxProcThreadStatus,
        },
    );
    snapshot.caller_thread_id = thread_before
        .as_deref()
        .and_then(|s| parse_status(s, "Pid:"))
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map_or(
            CpuReadback::Unknown(CpuProbeUnknown::ReadUnavailable),
            |value| CpuReadback::Known {
                value,
                method: CpuProbeMethod::LinuxProcThreadStatus,
            },
        );
    let topology_cpus = online_before
        .as_deref()
        .and_then(|s| parse_linux_cpus(s.trim()));
    if let Some(cpus) = topology_cpus.as_ref().or(process_cpus.as_ref()) {
        let mut groups = BTreeSet::new();
        let mut complete = true;
        for cpu in cpus {
            control()?;
            if groups
                .iter()
                .any(|siblings: &Vec<LogicalCpu>| siblings.contains(cpu))
            {
                continue;
            }
            let path = PathBuf::from(format!(
                "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
                cpu.index
            ));
            let raw = read(&path)?;
            observed_files.push((path, raw.clone()));
            if let Some(siblings) = raw.and_then(|s| parse_linux_cpus(s.trim())) {
                if !siblings.contains(cpu) {
                    complete = false;
                    break;
                }
                groups.insert(siblings);
            } else {
                complete = false;
                break;
            }
        }
        let cores: Vec<_> = groups
            .into_iter()
            .map(|logical_cpus| PhysicalCpuCore { logical_cpus })
            .collect();
        snapshot.physical_cores = if complete && validate_physical_cores(&cores).is_ok() {
            CpuReadback::Known {
                value: cores,
                method: CpuProbeMethod::LinuxSysfsSiblings,
            }
        } else {
            CpuReadback::Unknown(CpuProbeUnknown::ReadUnavailable)
        };
    }
    let member_before = read(Path::new("/proc/thread-self/cgroup"))?;
    let mounts_before = read(Path::new("/proc/thread-self/mountinfo"))?;
    let mapping = member_before
        .as_deref()
        .and_then(unified_cgroup)
        .and_then(|member| {
            mounts_before
                .as_deref()
                .and_then(|raw| cgroup_mount(raw, &member))
        });
    if let Some((directory, mount)) = mapping {
        let direct_path = directory.join("cpu.max");
        let direct_raw = read(&direct_path)?;
        observed_files.push((direct_path, direct_raw.clone()));
        let direct = direct_raw.and_then(|s| parse_cpu_max(&s));
        snapshot.direct_cpu_quota = direct.map_or(
            CpuReadback::Unknown(CpuProbeUnknown::ReadUnavailable),
            |value| CpuReadback::Known {
                value,
                method: CpuProbeMethod::LinuxCgroupV2Direct,
            },
        );
        let mut current = directory.clone();
        let mut effective = CpuQuotaReadback::Unlimited;
        let mut complete = false;
        for _ in 0..CGROUP_MAX_ANCESTORS {
            control()?;
            // A cgroup namespace/bind mount root can be a real non-root
            // cgroup with cpu.max. Missing root data stays unknown; never
            // substitute Unlimited solely because this is a mount boundary.
            let path = current.join("cpu.max");
            let raw = read(&path)?;
            observed_files.push((path, raw.clone()));
            let Some(quota) = raw.and_then(|s| parse_cpu_max(&s)) else {
                break;
            };
            effective = tighter(effective, quota);
            if current == mount {
                complete = true;
                break;
            }
            let Some(parent) = current.parent().filter(|p| p.starts_with(&mount)) else {
                break;
            };
            current = parent.to_path_buf();
        }
        snapshot.visible_ancestor_cpu_quota = if complete {
            CpuReadback::Known {
                value: effective,
                method: CpuProbeMethod::LinuxCgroupV2VisibleAncestors,
            }
        } else {
            CpuReadback::Unknown(CpuProbeUnknown::ReadUnavailable)
        };
    } else {
        snapshot.direct_cpu_quota =
            CpuReadback::Unknown(CpuProbeUnknown::NamespaceOrMountAmbiguous);
        snapshot.visible_ancestor_cpu_quota =
            CpuReadback::Unknown(CpuProbeUnknown::NamespaceOrMountAmbiguous);
    }
    let process_after = read(Path::new("/proc/self/status"))?;
    let thread_after = read(Path::new("/proc/thread-self/status"))?;
    let member_after = read(Path::new("/proc/thread-self/cgroup"))?;
    let boundary_changed = process_after
        .as_deref()
        .and_then(|s| parse_status(s, "Cpus_allowed_list:"))
        .and_then(|s| parse_linux_cpus(&s))
        != process_cpus
        || thread_after
            .as_deref()
            .and_then(|s| parse_status(s, "Cpus_allowed_list:"))
            .and_then(|s| parse_linux_cpus(&s))
            != thread_cpus
        || process_after
            .as_deref()
            .and_then(|s| parse_status(s, "Pid:"))
            != process_before
                .as_deref()
                .and_then(|s| parse_status(s, "Pid:"))
        || thread_after
            .as_deref()
            .and_then(|s| parse_status(s, "Pid:"))
            != thread_before
                .as_deref()
                .and_then(|s| parse_status(s, "Pid:"))
        || member_after != member_before
        || read(Path::new("/proc/thread-self/mountinfo"))? != mounts_before
        || read(Path::new("/sys/devices/system/cpu/online"))? != online_before;
    // Preserve the compatibility probe's original read/recheck order and
    // short-circuit behavior while bracketing every traversal iteration.
    let mut observed_files_changed = false;
    if !boundary_changed {
        for (path, before) in &observed_files {
            control()?;
            if read(path)? != *before {
                observed_files_changed = true;
                break;
            }
        }
    }
    if boundary_changed || observed_files_changed {
        snapshot.process_affinity = CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe);
        snapshot.caller_affinity = CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe);
        snapshot.physical_cores = CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe);
        snapshot.direct_cpu_quota = CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe);
        snapshot.visible_ancestor_cpu_quota =
            CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe);
    }
    control()?;
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cooperative_cpu_probe_rejects_expiry_and_cancel_before_any_read() {
        assert_eq!(
            observe_current_cpu_until(Instant::now(), &AtomicBool::new(false)),
            Err(CpuResourceError::ProbeDeadline)
        );
        assert_eq!(
            observe_current_cpu_until(
                Instant::now() + std::time::Duration::from_secs(1),
                &AtomicBool::new(true)
            ),
            Err(CpuResourceError::ProbeCanceled)
        );
    }
    #[test]
    fn cooperative_linux_traversal_stops_at_a_read_boundary() {
        use std::cell::Cell;
        let reads = Cell::new(0usize);
        let read = |_: &Path| {
            reads.set(reads.get() + 1);
            Some("Pid:\t100\nCpus_allowed_list:\t0-3\n".into())
        };
        let control = || {
            if reads.get() >= 2 {
                Err(CpuResourceError::ProbeCanceled)
            } else {
                Ok(())
            }
        };
        assert_eq!(
            observe_linux_cpu_controlled(read, control),
            Err(CpuResourceError::ProbeCanceled)
        );
        assert_eq!(reads.get(), 2);
    }
    fn cpu(index: u16) -> LogicalCpu {
        LogicalCpu { group: 0, index }
    }
    fn known<T>(value: T) -> CpuReadback<T> {
        CpuReadback::Known {
            value,
            method: CpuProbeMethod::WorkerReadyBoundary,
        }
    }
    fn local() -> (AdaptiveCpuResourcePlan, HostCpuSnapshot, CpuReadyReadback) {
        let plan = AdaptiveCpuResourcePlan {
            profile_id: "dedicated-control-cpu2-v1".into(),
            host_kind: CpuHostKind::RegisteredLocal,
            placement: ControlPlacement::DedicatedControl,
            allowed_cpus: vec![cpu(0), cpu(1), cpu(2), cpu(3)],
            manager_cpu: Some(cpu(0)),
            worker_cpus: vec![cpu(2), cpu(3)],
            ..AdaptiveCpuResourcePlan::default()
        };
        let mut host = HostCpuSnapshot::unknown();
        host.process_affinity_scope = CpuAffinityScope::Process;
        host.process_affinity = known(plan.allowed_cpus.clone());
        host.caller_affinity = known(vec![cpu(0)]);
        host.physical_cores = known(vec![
            PhysicalCpuCore {
                logical_cpus: vec![cpu(0), cpu(1)],
            },
            PhysicalCpuCore {
                logical_cpus: vec![cpu(2), cpu(3)],
            },
        ]);
        host.direct_cpu_quota = known(CpuQuotaReadback::Limited {
            quota_us: 200000,
            period_us: 100000,
        });
        let threads = [
            CpuActor::Manager,
            CpuActor::NativeCaller,
            CpuActor::OwnWorker(0),
            CpuActor::ExternalEngine(0),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, actor)| CpuThreadReadback {
            actor,
            thread_id: i as u64 + 1,
            affinity: known(vec![if i == 0 { cpu(0) } else { cpu(2) }]),
        })
        .collect();
        (
            plan,
            host,
            CpuReadyReadback {
                threads,
                inventory_complete: true,
                ort_applied_settings: known(OrtThreadPolicy::default()),
            },
        )
    }
    #[test]
    fn shared_unknown_is_not_dedicated_or_enforcement_evidence() {
        let plan = AdaptiveCpuResourcePlan::default();
        let ready = CpuReadyReadback {
            threads: Vec::new(),
            inventory_complete: false,
            ort_applied_settings: CpuReadback::Unknown(CpuProbeUnknown::UnobservedOrtSettings),
        };
        let result = admit_cpu_placement(&plan, &HostCpuSnapshot::unknown(), &ready).unwrap();
        assert_eq!(result.declared_runnable_threads, 4);
        assert_eq!(result.direct_quota_matches_request, None);
        assert!(!result.dedicated_smt_exclusion_verified && !result.resource_settings_installed);
    }
    #[test]
    fn dedicated_control_checks_actual_siblings_and_every_ready_actor() {
        let (plan, host, mut ready) = local();
        assert!(
            admit_cpu_placement(&plan, &host, &ready)
                .unwrap()
                .dedicated_smt_exclusion_verified
        );
        ready.threads[2].affinity = known(vec![cpu(1)]);
        assert_eq!(
            admit_cpu_placement(&plan, &host, &ready),
            Err(CpuResourceError::ManagerSmtOverlap)
        );
        ready.threads[2].affinity = known(vec![cpu(2)]);
        ready.threads[1].affinity = CpuReadback::Unknown(CpuProbeUnknown::UnobservedThread);
        assert_eq!(
            admit_cpu_placement(&plan, &host, &ready),
            Err(CpuResourceError::DedicatedEvidenceRequired)
        );
    }
    #[test]
    fn dedicated_cloud_or_incomplete_inventory_is_rejected() {
        let (mut plan, host, mut ready) = local();
        plan.host_kind = CpuHostKind::CloudOrUnknown;
        assert_eq!(
            admit_cpu_placement(&plan, &host, &ready),
            Err(CpuResourceError::DedicatedEvidenceRequired)
        );
        plan.host_kind = CpuHostKind::RegisteredLocal;
        ready.inventory_complete = false;
        assert_eq!(
            admit_cpu_placement(&plan, &host, &ready),
            Err(CpuResourceError::DedicatedEvidenceRequired)
        );
    }
    #[test]
    fn worker_profiles_and_ort_thread_settings_are_not_silently_scaled() {
        let mut p = AdaptiveCpuResourcePlan {
            own_cpu_workers: 2,
            ..AdaptiveCpuResourcePlan::default()
        };
        assert_eq!(
            p.validate(),
            Err(CpuResourceError::SeparateWorkerProfileRequired)
        );
        p.profile_id = "shared-control-cpu4-workers2-v1".into();
        p.requested_millicpus = 4000;
        assert!(p.validate().is_ok());
        assert_eq!(p.declared_runnable_threads(), 5);
        p.ort.intra_spinning = true;
        assert_eq!(p.validate(), Err(CpuResourceError::InvalidDeclaration));
    }
    #[test]
    fn strict_lists_and_rational_quota_do_not_truncate_or_round() {
        assert_eq!(
            parse_linux_cpus("0-1,4,255").unwrap(),
            vec![cpu(0), cpu(1), cpu(4), cpu(255)]
        );
        for s in ["", "0,0", "1-0", "0-4096", "1,+2", "0--2", "0, 2"] {
            assert!(parse_linux_cpus(s).is_none(), "{s}");
        }
        assert_eq!(
            parse_cpu_max("max 100000"),
            Some(CpuQuotaReadback::Unlimited)
        );
        for s in [
            "0 100000",
            "200000 0",
            "+200000 100000",
            "200000 100000 extra",
        ] {
            assert!(parse_cpu_max(s).is_none());
        }
        assert!(parse_cpu_max("200000 100000")
            .unwrap()
            .equals_millicpus(2000));
        assert!(!parse_cpu_max("200001 100000")
            .unwrap()
            .equals_millicpus(2000));
    }
    #[test]
    fn linux_reads_exact_mount_mapping_and_tightest_visible_ancestor() {
        let result = observe_linux_cpu(|p| match p.to_str().unwrap().replace('\\', "/").as_str() {
            "/proc/self/status" => Some("Pid:\t7\nCpus_allowed_list:\t0,2\n".into()),
            "/proc/thread-self/status" => Some("Pid:\t9\nCpus_allowed_list:\t2\n".into()),
            "/proc/thread-self/cgroup" => Some("0::/parent/worker\n".into()),
            "/proc/thread-self/mountinfo" => {
                Some("23 20 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n".into())
            }
            "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list" => Some("0-1\n".into()),
            "/sys/devices/system/cpu/cpu2/topology/thread_siblings_list" => Some("2-3\n".into()),
            "/sys/fs/cgroup/parent/worker/cpu.max" => Some("400000 100000\n".into()),
            "/sys/fs/cgroup/parent/cpu.max" => Some("200000 100000\n".into()),
            "/sys/fs/cgroup/cpu.max" => Some("max 100000\n".into()),
            _ => None,
        });
        assert_eq!(result.caller_thread_id.value(), Some(&9));
        assert_eq!(result.process_affinity.value(), Some(&vec![cpu(0), cpu(2)]));
        assert_eq!(
            result.physical_cores.value().unwrap()[0].logical_cpus,
            vec![cpu(0), cpu(1)]
        );
        assert_eq!(
            result.direct_cpu_quota.value(),
            Some(&CpuQuotaReadback::Limited {
                quota_us: 400000,
                period_us: 100000
            })
        );
        assert_eq!(
            result.visible_ancestor_cpu_quota.value(),
            Some(&CpuQuotaReadback::Limited {
                quota_us: 200000,
                period_us: 100000
            })
        );
        assert_eq!(result.ancestor_scope, CpuProbeUnknown::HiddenAncestors);
    }
    #[test]
    fn subtree_mount_root_quota_is_included_and_missing_root_is_unknown() {
        let read = |p: &Path| match p.to_str().unwrap().replace('\\', "/").as_str() {
            "/proc/self/status" | "/proc/thread-self/status" => {
                Some("Pid:7\nCpus_allowed_list:0\n".into())
            }
            "/proc/thread-self/cgroup" => Some("0::/parent/worker\n".into()),
            "/proc/thread-self/mountinfo" => {
                Some("23 20 0:1 /parent /sys/fs/cgroup rw - cgroup2 cgroup rw\n".into())
            }
            "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list" => Some("0-1\n".into()),
            "/sys/fs/cgroup/worker/cpu.max" => Some("400000 100000\n".into()),
            "/sys/fs/cgroup/cpu.max" => Some("200000 100000\n".into()),
            _ => None,
        };
        let observed = observe_linux_cpu(read);
        assert!(observed
            .visible_ancestor_cpu_quota
            .value()
            .unwrap()
            .equals_millicpus(2000));
        let missing = observe_linux_cpu(|p| {
            if p == Path::new("/sys/fs/cgroup/cpu.max") {
                None
            } else {
                read(p)
            }
        });
        assert!(matches!(
            missing.visible_ancestor_cpu_quota,
            CpuReadback::Unknown(_)
        ));
    }
    #[test]
    fn pinned_linux_leader_is_not_an_aggregate_process_mask() {
        let (plan, mut host, ready) = local();
        host.process_affinity_scope = CpuAffinityScope::ThreadGroupLeader;
        host.process_affinity = known(vec![cpu(0)]);
        assert!(
            admit_cpu_placement(&plan, &host, &ready)
                .unwrap()
                .dedicated_smt_exclusion_verified
        );
        host.process_affinity_scope = CpuAffinityScope::Process;
        assert_eq!(
            admit_cpu_placement(&plan, &host, &ready),
            Err(CpuResourceError::AffinityMismatch)
        );
    }
    #[test]
    fn undeclared_actors_zero_quota_and_mixed_cpu_ids_are_rejected() {
        let (plan, mut host, mut ready) = local();
        ready.threads.push(CpuThreadReadback {
            actor: CpuActor::OwnWorker(1),
            thread_id: 999,
            affinity: known(vec![cpu(2)]),
        });
        assert_eq!(
            admit_cpu_placement(&plan, &host, &ready),
            Err(CpuResourceError::InvalidReadback)
        );
        ready.threads.pop();
        host.direct_cpu_quota = known(CpuQuotaReadback::Limited {
            quota_us: 0,
            period_us: 0,
        });
        assert!(!host
            .direct_cpu_quota
            .value()
            .unwrap()
            .equals_millicpus(2000));
        assert_eq!(
            admit_cpu_placement(&plan, &host, &ready),
            Err(CpuResourceError::InvalidReadback)
        );
        assert!(!valid_cpu_set(
            &[cpu(64), LogicalCpu { group: 1, index: 0 }],
            false
        ));
    }
    #[test]
    fn changed_quota_or_mount_invalidates_the_sequential_snapshot() {
        use std::cell::Cell;
        let reads = Cell::new(0);
        let observed =
            observe_linux_cpu(|p| match p.to_str().unwrap().replace('\\', "/").as_str() {
                "/proc/self/status" | "/proc/thread-self/status" => {
                    Some("Pid:7\nCpus_allowed_list:0\n".into())
                }
                "/proc/thread-self/cgroup" => Some("0::/worker\n".into()),
                "/proc/thread-self/mountinfo" => {
                    Some("23 20 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n".into())
                }
                "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list" => {
                    Some("0-1\n".into())
                }
                "/sys/fs/cgroup/cpu.max" => Some("max 100000\n".into()),
                "/sys/fs/cgroup/worker/cpu.max" => {
                    reads.set(reads.get() + 1);
                    Some(
                        if reads.get() <= 2 {
                            "200000 100000\n"
                        } else {
                            "400000 100000\n"
                        }
                        .into(),
                    )
                }
                _ => None,
            });
        assert_eq!(
            observed.direct_cpu_quota,
            CpuReadback::Unknown(CpuProbeUnknown::ChangedDuringProbe)
        );
    }
    #[test]
    fn unsafe_or_ambiguous_cgroup_mapping_stays_unknown() {
        assert!(unified_cgroup("0::/../outside").is_none());
        assert!(unified_cgroup("0::/a\n0::/b").is_none());
        let member = Path::new("/a/b");
        assert_eq!(
            cgroup_mount("1 2 0:1 /a /sys/fs/cgroup rw - cgroup2 cgroup rw", member),
            Some((
                PathBuf::from("/sys/fs/cgroup/b"),
                PathBuf::from("/sys/fs/cgroup")
            ))
        );
        assert!(cgroup_mount(
            "1 2 0:1 / /one rw - cgroup2 cgroup rw\n2 2 0:1 / /two rw - cgroup2 cgroup rw",
            member
        )
        .is_none());
    }
    #[test]
    fn windows_groups_keep_full_sibling_masks_and_job_unknown_scope() {
        let result = windows_cpu_snapshot(WindowsCpuApiReadback {
            process_id: 9,
            caller_thread_id: 11,
            identity_stable: true,
            process_group_masks: known(vec![(1, 5)]),
            caller_group_mask: known((1, 1)),
            physical_core_group_masks: known(vec![vec![(1, 3)], vec![(1, 12)]]),
            immediate_job_cpu_quota: known(CpuQuotaReadback::WindowsHardCap {
                rate_per_10000: 2500,
            }),
        })
        .unwrap();
        assert_eq!(
            result.process_affinity.value().unwrap()[1].manifest_id(),
            66
        );
        assert_eq!(
            result.physical_cores.value().unwrap()[0].logical_cpus[1],
            LogicalCpu { group: 1, index: 1 }
        );
        assert_eq!(
            result.visible_ancestor_cpu_quota,
            CpuReadback::Unknown(CpuProbeUnknown::WindowsJobHierarchyUnavailable)
        );
        assert!(!result
            .direct_cpu_quota
            .value()
            .unwrap()
            .equals_millicpus(2000));
    }
    #[test]
    fn windows_rejects_duplicate_groups_or_overlapping_physical_cores() {
        assert!(map_windows_masks(known(vec![(0, 1), (0, 2)])).is_err());
        let raw = WindowsCpuApiReadback {
            process_id: 1,
            caller_thread_id: 2,
            identity_stable: true,
            process_group_masks: known(vec![(0, 3)]),
            caller_group_mask: known((0, 1)),
            physical_core_group_masks: known(vec![vec![(0, 1)], vec![(0, 1)]]),
            immediate_job_cpu_quota: CpuReadback::Unknown(
                CpuProbeUnknown::WindowsJobHierarchyUnavailable,
            ),
        };
        assert_eq!(
            windows_cpu_snapshot(raw),
            Err(CpuResourceError::InvalidReadback)
        );
    }
}
