//! Bounded actual-owner and read-only OS resource observations. No cgroup,
//! affinity, Job Object or engine option is installed here. Each observation
//! keeps its scope; a startup snapshot is never a lifetime/peak assertion.
use crate::pals_adaptive::AdaptiveRootObservationSnapshot;
use crate::pals_attestation::checker::resource::{
    self, ObservedLimit, ProcessResourceSnapshot, ResourceError,
};
use rz_eval::pals_onnx::PalsAppliedSessionSettings;
use rz_experiments::{
    ArtifactRef, ManifestError, PalsCpuActorV5, PalsCpuAffinityScopeV5, PalsCpuQuotaReadbackV5,
    PalsCpuQuotaScopeV5, PalsMemoryLimitReadbackV5, PalsObservedV3, PalsOrtSettingsReadbackV5,
    PalsResourceEvidenceProfileV5, PalsResourceReadbackV5, PalsThreadAffinityReadbackV5, Violation,
};
use rz_runtime::adaptive::{
    CpuActor, CpuAffinityScope, CpuProbeMethod, CpuQuotaReadback, CpuReadback, CpuResourceError,
    CpuThreadReadback, HostCpuSnapshot,
};
use rz_search::cpu_checker::ExternalProcessIdentity;
use std::{
    collections::BTreeSet,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

fn observed<T>(value: T, method: &str) -> PalsObservedV3<T> {
    PalsObservedV3::Observed {
        value,
        method: method.into(),
    }
}
fn invalid(message: &str) -> ManifestError {
    ManifestError::Integrity(format!("V5 actual resource observation: {message}"))
}

pub(super) fn require_original_startup_active(cancel: &AtomicBool) -> Result<(), ManifestError> {
    if cancel.load(Ordering::Acquire) {
        Err(ManifestError::Validation(vec![Violation {
            path: "resource_readback".into(),
            code: "OriginalStartupCanceled",
            message: "the original startup owner revoked resource observation/publication".into(),
        }]))
    } else {
        Ok(())
    }
}

/// Minted inside bootstrap's original native factory, after actual successful
/// loaders return. Session records retain their unforgeable constructor origin.
pub(crate) struct NativeResourceOwnerV5 {
    settings: Option<Arc<[PalsAppliedSessionSettings]>>,
    native_thread: Option<CpuThreadReadback>,
}
impl NativeResourceOwnerV5 {
    pub(crate) fn from_original_native_factory(
        settings: Option<Arc<[PalsAppliedSessionSettings]>>,
        cancel: &AtomicBool,
    ) -> Self {
        Self {
            settings,
            native_thread: if cancel.load(Ordering::Acquire) {
                None
            } else {
                rz_runtime::adaptive::observe_current_cpu_thread(CpuActor::NativeCaller)
                    .ok()
                    .filter(|_| !cancel.load(Ordering::Acquire))
            },
        }
    }
}

/// At most one startup observation and the last settled root's historical
/// ready readings. Root churn never appends an unbounded thread-history list.
pub(crate) struct ActualResourceObservationsV5 {
    owner: NativeResourceOwnerV5,
    startup: Option<PalsResourceReadbackV5>,
    latest_root: Option<PalsResourceReadbackV5>,
    latest_authority: Option<rz_contracts::pals::SearchAuthority>,
    diagnostic: &'static str,
    host_diagnostic: &'static str,
}
impl ActualResourceObservationsV5 {
    pub(crate) fn new(owner: NativeResourceOwnerV5) -> Self {
        Self {
            owner,
            startup: None,
            latest_root: None,
            latest_authority: None,
            diagnostic: "no_ready_boundary_probe",
            host_diagnostic: "host_cpu_probe_not_attempted",
        }
    }
    pub(crate) fn observe_startup(
        &mut self,
        profile: PalsResourceEvidenceProfileV5,
        native_thread: Option<CpuThreadReadback>,
        external: Option<ExternalProcessIdentity>,
        until: Instant,
        cancel: &AtomicBool,
        evidence: ArtifactRef,
    ) -> Result<PalsResourceReadbackV5, ManifestError> {
        if self.startup.is_some() {
            return Err(invalid("startup resources already captured"));
        }
        require_original_startup_active(cancel)?;
        if native_thread
            .as_ref()
            .is_some_and(|thread| thread.actor != CpuActor::NativeCaller)
            || native_thread
                .as_ref()
                .zip(self.owner.native_thread.as_ref())
                .is_some_and(|(ready, factory)| ready.thread_id != factory.thread_id)
        {
            return Err(invalid(
                "native service ready thread differs from its original factory",
            ));
        }
        let native_thread = native_thread.or_else(|| self.owner.native_thread.clone());
        let mut value = PalsResourceReadbackV5::with_unknown_checks(profile, evidence);
        // The bootstrap/main caller is deliberately not labelled Manager.
        // The actual manager is observed on the UCI search owner below.
        if Instant::now() < until {
            match rz_runtime::adaptive::observe_current_cpu_until(until, cancel) {
                Ok(host) => {
                    map_host_cpu(&host, &mut value)?;
                    self.host_diagnostic = "controlled_original_host_cpu_probe";
                }
                Err(CpuResourceError::ProbeDeadline) => {
                    self.host_diagnostic = "host_cpu_probe_deadline";
                }
                Err(CpuResourceError::ProbeCanceled) => {
                    self.host_diagnostic = "host_cpu_probe_canceled";
                }
                Err(CpuResourceError::ProbeBudgetExceeded) => {
                    self.host_diagnostic = "host_cpu_probe_read_budget_exceeded";
                }
                Err(error) => {
                    return Err(invalid(&format!(
                        "original host CPU probe rejected: {error}"
                    )));
                }
            }
            require_original_startup_active(cancel)?;
            let observed_processes = match external {
                Some(identity) => resource::observe(Some(identity), until, cancel).map(|reading| {
                    (
                        reading
                            .snapshot
                            .map(|pair| (pair.parent, Some(pair.helper))),
                        reading.unavailable,
                    )
                }),
                None => resource::observe_current_process(until, cancel)
                    .map(|parent| (parent.map(|parent| (parent, None)), None)),
            };
            match observed_processes {
                Ok((Some((parent, child)), _)) => {
                    map_ready_processes(
                        &parent,
                        child.as_ref(),
                        native_thread.as_ref(),
                        &mut value,
                    )?;
                    self.diagnostic = "bounded_identity_bracketed_ready_processes";
                }
                Ok((None, reason)) => {
                    self.diagnostic = reason.unwrap_or("os_process_inventory_unavailable");
                }
                Err(error) => self.diagnostic = process_probe_diagnostic(error)?,
            }
        } else {
            self.diagnostic = "resource_probe_original_deadline_expired";
            self.host_diagnostic = "host_cpu_probe_original_deadline_expired";
        }
        require_original_startup_active(cancel)?;
        if value.threads.is_empty() {
            if let Some(native) = native_thread.as_ref() {
                value
                    .threads
                    .push(map_owner_thread(native, PalsCpuActorV5::NativeCaller)?);
            }
        }
        map_native_sessions(self.owner.settings.as_deref(), &mut value)?;
        require_original_startup_active(cancel)?;
        self.startup = Some(value.clone());
        Ok(value)
    }
    /// Called from the actual settled root closure. These are historical owner
    /// ready readings, not an inventory of still-alive threads after joins.
    pub(crate) fn observe_closed_root(
        &mut self,
        root: &AdaptiveRootObservationSnapshot,
        manager: Option<CpuThreadReadback>,
    ) -> Result<(), ManifestError> {
        let authority = root.authority();
        if self.latest_authority.is_some_and(|previous| {
            previous.epoch != authority.epoch
                || previous.game.0 > authority.game.0
                || (previous.game == authority.game && previous.root.0 >= authority.root.0)
        }) {
            return Err(invalid(
                "duplicate, stale or different-process root readback",
            ));
        }
        let Some(mut value) = self.startup.clone() else {
            return Err(invalid(
                "root readback precedes actual startup resource publication",
            ));
        };
        value.threads.clear();
        value.thread_inventory_complete = observed(
            false,
            "settled root carries historical ready owner readings; joined workers are not a live OS inventory",
        );
        value.total_observed_kernel_threads = PalsObservedV3::Unknown;
        if let Some(manager) = manager {
            if manager.actor != CpuActor::Manager {
                return Err(invalid("closure caller was not observed as the manager"));
            }
            value
                .threads
                .push(map_owner_thread(&manager, PalsCpuActorV5::Manager)?);
        }
        let receipt = root.shutdown_receipt();
        if let Some(Ok(native)) = &receipt.native_cpu_thread_observation {
            if native.actor != CpuActor::NativeCaller
                || self
                    .owner
                    .native_thread
                    .as_ref()
                    .is_some_and(|factory| factory.thread_id != native.thread_id)
            {
                return Err(invalid("native owner actor mismatch"));
            }
            value
                .threads
                .push(map_owner_thread(native, PalsCpuActorV5::NativeCaller)?);
        }
        if receipt.own_cpu_thread_observations.len() > 4 {
            return Err(invalid(
                "own worker readback exceeds registered worker boundary",
            ));
        }
        for (index, reading) in receipt.own_cpu_thread_observations.iter().enumerate() {
            if let Some(Ok(worker)) = reading {
                // Bootstrap owns CPU WorkerIds 1..=N; receipt order is the
                // actual retained slot order. Manifest actor indices are 0..N.
                if worker.actor != CpuActor::OwnWorker((index + 1) as u8) {
                    return Err(invalid("own worker owner/slot mismatch"));
                }
                value.threads.push(map_owner_thread(
                    worker,
                    PalsCpuActorV5::OwnWorker {
                        index: index as u32,
                    },
                )?);
            }
        }
        require_unique_threads(&value.threads)?;
        self.latest_root = Some(value);
        self.latest_authority = Some(authority);
        Ok(())
    }
    pub(crate) fn final_readback(&self, evidence: ArtifactRef) -> Option<PalsResourceReadbackV5> {
        self.latest_root
            .as_ref()
            .or(self.startup.as_ref())
            .map(|value| {
                let mut value = value.clone();
                value.evidence = evidence;
                value
            })
    }
    pub(crate) fn diagnostic(&self) -> &'static str {
        self.diagnostic
    }
    pub(crate) fn host_diagnostic(&self) -> &'static str {
        self.host_diagnostic
    }
}

fn map_host_cpu(
    host: &HostCpuSnapshot,
    out: &mut PalsResourceReadbackV5,
) -> Result<(), ManifestError> {
    if host.process_id != std::process::id() {
        return Err(invalid("host probe process mismatch"));
    }
    out.affinity_scope = match host.process_affinity_scope {
        CpuAffinityScope::Process => PalsCpuAffinityScopeV5::Process,
        CpuAffinityScope::ThreadGroupLeader => PalsCpuAffinityScopeV5::ThreadGroupLeader,
        CpuAffinityScope::Unknown => PalsCpuAffinityScopeV5::Unknown,
    };
    if let CpuReadback::Known { value, method } = &host.process_affinity {
        out.allowed_logical_cpus = observed(
            cpu_ids(value)?,
            &format!(
                "read-only {method:?}; {:?} scope, sequential OS snapshot",
                host.process_affinity_scope
            ),
        );
    }
    // Linux's visible-ancestor minimum is not complete host ancestry or a
    // process-tree installation proof. Windows exposes only its immediate Job.
    let quota = match &host.direct_cpu_quota {
        CpuReadback::Known {
            value,
            method: CpuProbeMethod::WindowsImmediateJobObject,
        } => Some((
            *value,
            PalsCpuQuotaScopeV5::ImmediateJobObject,
            "Windows immediate Job Object quota; nested hierarchy unknown",
        )),
        _ => match &host.visible_ancestor_cpu_quota {
            CpuReadback::Known { value, .. } => Some((
                *value,
                PalsCpuQuotaScopeV5::VisibleAncestors,
                "Linux tightest readable cgroup quota in visible namespace; hidden ancestors and process-tree application unknown",
            )),
            _ => None,
        },
    };
    if let Some((quota, scope, method)) = quota {
        let value = match quota {
            CpuQuotaReadback::Unlimited => PalsCpuQuotaReadbackV5::Unlimited { scope },
            CpuQuotaReadback::Limited {
                quota_us,
                period_us,
            } if quota_us > 0 && period_us > 0 => PalsCpuQuotaReadbackV5::Limited {
                quota_us,
                period_us,
                scope,
            },
            CpuQuotaReadback::WindowsHardCap { rate_per_10000 }
                if (1..=10_000).contains(&rate_per_10000) =>
            {
                PalsCpuQuotaReadbackV5::WindowsHardCap {
                    rate_per_10000: rate_per_10000 as u32,
                    scope,
                }
            }
            _ => return Err(invalid("invalid native CPU quota reading")),
        };
        out.cpu_quota = observed(value, method);
    }
    // No declaration is copied into limits_applied, ancestor completeness,
    // SMT exclusion, device identity or a physical memory peak.
    Ok(())
}

fn cpu_ids(cpus: &[rz_runtime::adaptive::LogicalCpu]) -> Result<Vec<u32>, ManifestError> {
    if cpus.is_empty() || cpus.len() > 4096 {
        return Err(invalid("CPU mask outside bounded probe"));
    }
    let ids: Vec<_> = cpus.iter().map(|cpu| cpu.manifest_id()).collect();
    if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err(invalid("duplicate CPU identities"));
    }
    Ok(ids)
}
fn map_owner_thread(
    thread: &CpuThreadReadback,
    actor: PalsCpuActorV5,
) -> Result<PalsThreadAffinityReadbackV5, ManifestError> {
    if thread.thread_id == 0 {
        return Err(invalid("zero original owner thread identity"));
    }
    let affinity = match &thread.affinity {
        CpuReadback::Known { value, method } => observed(
            cpu_ids(value)?,
            &format!("original owner ready thread via {method:?}; historical per-thread snapshot"),
        ),
        CpuReadback::Unknown(_) => PalsObservedV3::Unknown,
    };
    Ok(PalsThreadAffinityReadbackV5 {
        actors: BTreeSet::from([actor]),
        process_id: std::process::id() as u64,
        thread_id: thread.thread_id,
        affinity,
    })
}
fn require_unique_threads(threads: &[PalsThreadAffinityReadbackV5]) -> Result<(), ManifestError> {
    let mut ids = BTreeSet::new();
    if threads.len() > 1024
        || threads
            .iter()
            .any(|thread| !ids.insert((thread.process_id, thread.thread_id)))
    {
        return Err(invalid(
            "duplicate physical owner thread or inventory over bound",
        ));
    }
    Ok(())
}
fn limit(value: ObservedLimit) -> PalsMemoryLimitReadbackV5 {
    match value {
        ObservedLimit::Numeric { value } => PalsMemoryLimitReadbackV5::Limited { bytes: value },
        ObservedLimit::Max => PalsMemoryLimitReadbackV5::Unlimited,
    }
}
fn linux_cpu_list(raw: &str) -> Result<Vec<u32>, ManifestError> {
    let mut ids = BTreeSet::new();
    if raw.is_empty() || raw.len() > 64 * 1024 {
        return Err(invalid("invalid Linux CPU list size"));
    }
    let number = |value: &str| -> Result<u32, ManifestError> {
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid("invalid Linux CPU number"));
        }
        value
            .parse::<u32>()
            .ok()
            .filter(|id| *id <= 65535)
            .ok_or_else(|| invalid("Linux CPU id over bound"))
    };
    for span in raw.trim().split(',') {
        let (lo, hi) = match span.split_once('-') {
            Some((lo, hi)) => (number(lo)?, number(hi)?),
            None => {
                let id = number(span)?;
                (id, id)
            }
        };
        if lo > hi {
            return Err(invalid("inverted Linux CPU range"));
        }
        let count = usize::try_from(hi - lo + 1)
            .ok()
            .and_then(|count| ids.len().checked_add(count))
            .filter(|count| *count <= 4096)
            .ok_or_else(|| invalid("Linux CPU count over bound"))?;
        // Validate the selected count before expanding a range. Sparse CPU
        // IDs are valid; the allocation bound is independent of the ID bound.
        debug_assert!(count <= 4096);
        for id in lo..=hi {
            if !ids.insert(id) {
                return Err(invalid("duplicate Linux CPU range"));
            }
        }
    }
    Ok(ids.into_iter().collect())
}

fn process_probe_diagnostic(error: ResourceError) -> Result<&'static str, ManifestError> {
    #[cfg(target_os = "linux")]
    {
        match error {
            ResourceError::Invalid(code) => Err(invalid(&format!(
                "original OS process identity/resource probe rejected: {code}"
            ))),
            ResourceError::Unavailable(code) => Ok(code),
            ResourceError::Deadline => Ok("resource_process_probe_deadline"),
        }
    }
    #[cfg(not(target_os = "linux"))]
    match error {}
}
fn map_ready_processes(
    parent: &ProcessResourceSnapshot,
    child: Option<&ProcessResourceSnapshot>,
    native: Option<&CpuThreadReadback>,
    out: &mut PalsResourceReadbackV5,
) -> Result<(), ManifestError> {
    if parent.pid != std::process::id()
        || parent.proc_start_ticks_before == 0
        || parent.proc_start_ticks_before != parent.proc_start_ticks_after
    {
        return Err(invalid("bracketed self process identity mismatch"));
    }
    if child.is_some_and(|child| child.parent_pid != parent.pid || child.pid == parent.pid) {
        return Err(invalid(
            "admitted direct helper differs from parent identity",
        ));
    }
    out.allowed_logical_cpus = observed(
        linux_cpu_list(&parent.cpu_allowed_list)?,
        "actual identity-bracketed Linux thread-group-leader Cpus_allowed_list at ready boundary; no aggregate worker mask claim",
    );
    out.affinity_scope = PalsCpuAffinityScopeV5::ThreadGroupLeader;
    out.memory_high = observed(
        limit(parent.cgroup_limits.memory_high),
        "direct memory.high of identity-bracketed self cgroup; no ancestor/lifetime/peak claim",
    );
    out.memory_max = observed(
        limit(parent.cgroup_limits.memory_max),
        "direct memory.max of identity-bracketed self cgroup; no ancestor/lifetime/peak claim",
    );
    let mut observed_threads = 0u32;
    for process in std::iter::once(parent).chain(child) {
        if process.proc_start_ticks_before == 0
            || process.proc_start_ticks_before != process.proc_start_ticks_after
            || process.threads.is_empty()
            || process.threads.len() > 64
        {
            return Err(invalid("unbracketed or oversized process thread inventory"));
        }
        for thread in &process.threads {
            if thread.proc_start_ticks_before == 0
                || thread.proc_start_ticks_before != thread.proc_start_ticks_after
            {
                return Err(invalid("thread identity changed in ready inventory"));
            }
            observed_threads += 1;
            let actor = if process.pid == parent.pid
                && native.is_some_and(|native| native.thread_id == thread.tid as u64)
            {
                Some(PalsCpuActorV5::NativeCaller)
            } else if child.is_some_and(|child| process.pid == child.pid)
                && thread.tid == process.pid
            {
                Some(PalsCpuActorV5::ExternalEngine { index: 0 })
            } else {
                // Enumeration proves a kernel thread exists, but supplies no
                // actor provenance. Do not invent ORT/runtime helper actors
                // from a name or assign bootstrap/main to the manager role.
                None
            };
            let Some(actor) = actor else {
                continue;
            };
            out.threads.push(PalsThreadAffinityReadbackV5 {
                actors: BTreeSet::from([actor]), process_id: process.pid as u64, thread_id: thread.tid as u64,
                affinity: observed(linux_cpu_list(&thread.cpu_allowed_list)?, "actual identity-bracketed Linux per-thread Cpus_allowed_list at ready boundary"),
            });
        }
    }
    require_unique_threads(&out.threads)?;
    out.thread_inventory_complete = observed(
        out.threads.len() == observed_threads as usize,
        "actual bounded parent/helper thread enumeration; unknown actor threads omitted from typed owner rows, so owner inventory is incomplete when counts differ",
    );
    out.total_observed_kernel_threads = observed(
        observed_threads,
        "sum of actual ready-boundary kernel threads in parent and admitted direct helper only; no peak or actor attribution claim",
    );
    Ok(())
}
fn map_native_sessions(
    records: Option<&[PalsAppliedSessionSettings]>,
    out: &mut PalsResourceReadbackV5,
) -> Result<(), ManifestError> {
    let Some(records) = records.filter(|records| !records.is_empty()) else {
        return Ok(());
    };
    if records.len() > 4 {
        return Err(invalid(
            "native session setting inventory exceeds supported loader",
        ));
    }
    let first = &records[0];
    let mut roles = BTreeSet::new();
    if records.iter().any(|record| {
        !roles.insert(record.role())
            || record.intra_threads() != first.intra_threads()
            || record.inter_threads() != first.inter_threads()
            || record.intra_op_spinning() != first.intra_op_spinning()
            || record.inter_op_spinning() != first.inter_op_spinning()
            || record.parallel_execution()
            || record.provider() != first.provider()
    }) {
        return Err(invalid(
            "original successful sessions have inconsistent applied settings",
        ));
    }
    let intra =
        u32::try_from(first.intra_threads()).map_err(|_| invalid("ORT intra count overflow"))?;
    let inter =
        u32::try_from(first.inter_threads()).map_err(|_| invalid("ORT inter count overflow"))?;
    if intra == 0 || inter == 0 {
        return Err(invalid("zero applied ORT thread count"));
    }
    out.ort_settings = observed(
        PalsOrtSettingsReadbackV5 {
            intra_threads: intra,
            inter_threads: inter,
            spinning: first.intra_op_spinning() || first.inter_op_spinning(),
        },
        "original native loader SessionOptions setters, EP registration and Session commit succeeded for every loaded session; OS helper count/affinity remain independent",
    );
    if records
        .iter()
        .all(|record| record.cuda_provider_available() == Some(true))
    {
        out.cuda_provider_ready = observed(
            true,
            "original CUDA EP availability check, registration and native session commit succeeded; no first-Run/kernel-placement or physical completion claim",
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn artifact() -> ArtifactRef {
        ArtifactRef {
            path: "runtime/test/resources.json".into(),
            sha256: "0".repeat(64),
            bytes: 1,
            source: "https://github.com/daejunnom/RoveZero".into(),
            license: "MIT".into(),
        }
    }
    #[test]
    fn absent_native_records_never_promote_configured_settings_or_cuda() {
        let mut value = PalsResourceReadbackV5::with_unknown_checks(
            PalsResourceEvidenceProfileV5::CloudObserved,
            artifact(),
        );
        map_native_sessions(None, &mut value).unwrap();
        assert!(matches!(value.ort_settings, PalsObservedV3::Unknown));
        assert!(matches!(value.cuda_provider_ready, PalsObservedV3::Unknown));
        assert!(matches!(value.limits_applied, PalsObservedV3::Unknown));
        assert!(matches!(
            value.smt_sibling_exclusion_verified,
            PalsObservedV3::Unknown
        ));
    }
    #[test]
    fn canceled_original_startup_does_not_probe_or_commit_resources() {
        // This test-only owner contains no native session or OS evidence; the
        // original cancellation must be checked before any OS operation.
        let mut resources = ActualResourceObservationsV5::new(NativeResourceOwnerV5 {
            settings: None,
            native_thread: None,
        });
        let cancel = Arc::new(AtomicBool::new(false));
        require_original_startup_active(&cancel).unwrap();
        let caller = Arc::clone(&cancel);
        caller.store(true, Ordering::Release);
        let error = resources
            .observe_startup(
                PalsResourceEvidenceProfileV5::CloudObserved,
                None,
                None,
                Instant::now() + std::time::Duration::from_secs(1),
                &cancel,
                artifact(),
            )
            .unwrap_err();
        let ManifestError::Validation(violations) = error else {
            panic!("original cancellation must remain a typed admission failure");
        };
        assert_eq!(violations[0].code, "OriginalStartupCanceled");
        assert!(resources.startup.is_none());
        assert_eq!(resources.diagnostic(), "no_ready_boundary_probe");
        assert_eq!(resources.host_diagnostic(), "host_cpu_probe_not_attempted");
        let canceled_factory = NativeResourceOwnerV5::from_original_native_factory(None, &cancel);
        assert!(canceled_factory.native_thread.is_none());
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn original_process_identity_failure_is_not_a_successful_unknown_readback() {
        let error = process_probe_diagnostic(ResourceError::Invalid(
            "resource_process_identity_or_membership_changed",
        ))
        .unwrap_err();
        assert!(matches!(error, ManifestError::Integrity(_)));
        assert!(
            error
                .to_string()
                .contains("resource_process_identity_or_membership_changed")
        );
        assert_eq!(
            process_probe_diagnostic(ResourceError::Deadline).unwrap(),
            "resource_process_probe_deadline"
        );
        assert_eq!(
            process_probe_diagnostic(ResourceError::Unavailable("process_gone")).unwrap(),
            "process_gone"
        );
    }
    #[test]
    fn linux_mask_mapping_preserves_ids_and_rejects_ambiguous_ranges() {
        assert_eq!(
            linux_cpu_list("0,2-3,4095,5000,65535").unwrap(),
            vec![0, 2, 3, 4095, 5000, 65535]
        );
        for invalid in ["", "3-2", "1,1", "0-1,1", "65536", "0-4096", "-1", "1--2"] {
            assert!(linux_cpu_list(invalid).is_err());
        }
    }
    #[test]
    fn original_owner_threads_do_not_become_an_applied_inventory() {
        let thread = CpuThreadReadback {
            actor: CpuActor::NativeCaller,
            thread_id: 7,
            affinity: CpuReadback::Unknown(rz_runtime::adaptive::CpuProbeUnknown::ReadUnavailable),
        };
        let mapped = map_owner_thread(&thread, PalsCpuActorV5::NativeCaller).unwrap();
        assert_eq!(mapped.thread_id, 7);
        assert!(matches!(mapped.affinity, PalsObservedV3::Unknown));
        assert!(require_unique_threads(&[mapped.clone(), mapped]).is_err());
    }
    #[test]
    fn visible_linux_quota_stays_scoped_and_does_not_install_declared_limits() {
        fn unknown<T>() -> CpuReadback<T> {
            CpuReadback::Unknown(rz_runtime::adaptive::CpuProbeUnknown::HiddenAncestors)
        }
        let host = HostCpuSnapshot {
            process_id: std::process::id(),
            caller_thread_id: CpuReadback::Unknown(
                rz_runtime::adaptive::CpuProbeUnknown::UnobservedThread,
            ),
            process_affinity: CpuReadback::Known {
                value: vec![rz_runtime::adaptive::LogicalCpu { group: 0, index: 2 }],
                method: CpuProbeMethod::LinuxProcProcessStatus,
            },
            process_affinity_scope: CpuAffinityScope::ThreadGroupLeader,
            caller_affinity: unknown(),
            physical_cores: unknown(),
            direct_cpu_quota: CpuReadback::Known {
                value: CpuQuotaReadback::Limited {
                    quota_us: 200_000,
                    period_us: 100_000,
                },
                method: CpuProbeMethod::LinuxCgroupV2Direct,
            },
            visible_ancestor_cpu_quota: CpuReadback::Known {
                value: CpuQuotaReadback::Limited {
                    quota_us: 150_000,
                    period_us: 100_000,
                },
                method: CpuProbeMethod::LinuxCgroupV2VisibleAncestors,
            },
            ancestor_scope: rz_runtime::adaptive::CpuProbeUnknown::HiddenAncestors,
            sequential_snapshot: true,
        };
        let mut value = PalsResourceReadbackV5::with_unknown_checks(
            PalsResourceEvidenceProfileV5::CloudObserved,
            artifact(),
        );
        map_host_cpu(&host, &mut value).unwrap();
        assert!(matches!(
            value.cpu_quota,
            PalsObservedV3::Observed {
                value: PalsCpuQuotaReadbackV5::Limited {
                    quota_us: 150_000,
                    period_us: 100_000,
                    scope: PalsCpuQuotaScopeV5::VisibleAncestors
                },
                ..
            }
        ));
        assert_eq!(
            value.affinity_scope,
            PalsCpuAffinityScopeV5::ThreadGroupLeader
        );
        assert!(matches!(value.limits_applied, PalsObservedV3::Unknown));
        assert!(matches!(
            value.quota_ancestor_chain_complete,
            PalsObservedV3::Unknown
        ));
        assert!(matches!(value.memory_max, PalsObservedV3::Unknown));
    }
    #[test]
    fn unattributed_kernel_threads_are_counted_without_inventing_runtime_actors() {
        let mut pair = resource::fixture();
        pair.parent.pid = std::process::id();
        pair.parent.threads[0].tid = pair.parent.pid;
        pair.helper.pid = pair.parent.pid + 1;
        pair.helper.threads[0].tid = pair.helper.pid;
        pair.helper.parent_pid = pair.parent.pid;
        let mut value = PalsResourceReadbackV5::with_unknown_checks(
            PalsResourceEvidenceProfileV5::CloudObserved,
            artifact(),
        );
        map_ready_processes(&pair.parent, Some(&pair.helper), None, &mut value).unwrap();
        assert_eq!(value.threads.len(), 1);
        assert_eq!(
            value.threads[0].actors,
            BTreeSet::from([PalsCpuActorV5::ExternalEngine { index: 0 }])
        );
        assert!(matches!(
            value.total_observed_kernel_threads,
            PalsObservedV3::Observed { value: 2, .. }
        ));
        assert!(matches!(
            value.thread_inventory_complete,
            PalsObservedV3::Observed { value: false, .. }
        ));
        assert!(matches!(value.limits_applied, PalsObservedV3::Unknown));
    }
}
