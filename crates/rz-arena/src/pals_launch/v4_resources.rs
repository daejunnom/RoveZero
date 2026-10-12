//! Explicit cloud observations coexist with the historical strict V4 admission.
//! This module installs no resource settings and never marks requested limits
//! as observed. Only own-checker PALS endpoints enter the functional pilot lane.
use super::*;
pub(super) const FAILURE_METADATA_CAP: u64 = 64 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "profile", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsArenaResourceProfileV4 {
    ColabObserved {},
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsArenaObservedLimitV4 {
    Numeric { value: u64 },
    Max,
    Unknown { reason: String },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsArenaObservedCpuQuotaV4 {
    pub quota: PalsArenaObservedLimitV4,
    pub period_us: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsArenaResourceObservationV4 {
    pub domain: String,
    pub stage: String,
    pub requested: PalsResourcePolicyV3,
    pub actual_affinity: Vec<u32>,
    pub cgroup_v2_membership: Option<String>,
    pub cpu_max: PalsArenaObservedCpuQuotaV4,
    pub memory_high: PalsArenaObservedLimitV4,
    pub memory_max: PalsArenaObservedLimitV4,
    pub memory_swap_max: PalsArenaObservedLimitV4,
    pub pids_max: PalsArenaObservedLimitV4,
    pub method: String,
    pub strict_local_resources_admitted: bool,
    pub resource_settings_installed: bool,
    pub ancestor_effective_limits: String,
    pub performance_evidence_eligible: bool,
    pub pair_eligible: bool,
    pub strength_eligible: bool,
}

pub(super) fn validate_profile(input: &PalsArenaLaunchV4) -> Result<(), ArenaError> {
    if input.resource_profile.is_none() {
        return Ok(());
    }
    let manifest = &input.semantic_lock.manifest;
    require(
        manifest.resources.iter().all(|r| {
            r.cpu_threads == 2
                && r.cpu_affinity.len() == 2
                && r.cpu_affinity[0] != r.cpu_affinity[1]
                && r.memory_high_bytes == 6 << 30
                && r.memory_max_bytes == 12 << 30
                && r.swap_max_bytes == 0
        }) && manifest.engines.iter().all(|e| {
            matches!(e, PalsEngineV4::Pals(p) if p.policies.checker == PalsCheckerPolicyV4::Own && p.base.cpu_r.is_own())
        }),
        "Colab-observed pilot requires CPU2/local requested targets and two own-checker PALS endpoints; external checker admission is unchanged",
    )
}

#[cfg(any(target_os = "linux", test))]
fn unknown(reason: &str) -> PalsArenaObservedLimitV4 {
    PalsArenaObservedLimitV4::Unknown {
        reason: reason.into(),
    }
}

#[cfg(any(target_os = "linux", test))]
fn limit(raw: Option<&str>) -> PalsArenaObservedLimitV4 {
    let Some(raw) = raw else {
        return unknown("read_unavailable");
    };
    let raw = raw.trim();
    if raw == "max" {
        PalsArenaObservedLimitV4::Max
    } else if !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()) {
        raw.parse::<u64>()
            .map(|value| PalsArenaObservedLimitV4::Numeric { value })
            .unwrap_or_else(|_| unknown("limit_overflow"))
    } else {
        unknown("limit_format")
    }
}

#[cfg(any(target_os = "linux", test))]
fn cpu_quota(raw: Option<&str>) -> PalsArenaObservedCpuQuotaV4 {
    let fields: Vec<_> = raw.unwrap_or("").split_ascii_whitespace().collect();
    if fields.len() == 2
        && let Ok(period) = fields[1].parse::<u64>()
        && period > 0
        && fields[1].bytes().all(|b| b.is_ascii_digit())
    {
        let quota = limit(Some(fields[0]));
        if !matches!(quota, PalsArenaObservedLimitV4::Unknown { .. }) {
            return PalsArenaObservedCpuQuotaV4 {
                quota,
                period_us: Some(period),
            };
        }
    }
    PalsArenaObservedCpuQuotaV4 {
        quota: unknown(if raw.is_some() {
            "cpu_quota_format"
        } else {
            "read_unavailable"
        }),
        period_us: None,
    }
}

#[cfg(any(target_os = "linux", test))]
fn membership(raw: Option<&str>) -> Option<String> {
    let mut rows = raw?.lines().filter_map(|line| line.strip_prefix("0::"));
    let value = rows.next()?;
    (rows.next().is_none()
        && value.starts_with('/')
        && !value.bytes().any(|b| b.is_ascii_control())
        && !value.split('/').any(|part| part == "." || part == ".."))
    .then(|| value.into())
}

/// Read this process only. Unknown cgroup values stay unknown; affinity must
/// still match both declared CPU IDs. No limit, namespace or quota is changed.
#[cfg(target_os = "linux")]
fn observe(
    spec: &LockedPalsArenaLaunchV4,
    stage: &str,
    read: impl Fn(&Path) -> Option<String>,
) -> Result<PalsArenaResourceObservationV4, ArenaError> {
    let status = read(Path::new("/proc/self/status"))
        .ok_or_else(|| invalid("Colab actual affinity observation unavailable"))?;
    let cpus = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .ok_or_else(|| invalid("Colab actual affinity evidence absent"))?;
    let actual_affinity = parse_pals_affinity(cpus.trim())?;
    let requested = spec.input.semantic_lock.manifest.resources[0].clone();
    let mut wanted = requested.cpu_affinity.clone();
    wanted.sort_unstable();
    require(
        actual_affinity == wanted && actual_affinity.len() == 2,
        "Colab actual affinity differs from the two registered CPUs",
    )?;
    let cgroup_v2_membership = membership(read(Path::new("/proc/self/cgroup")).as_deref());
    let read_limit = |name: &str| {
        cgroup_v2_membership.as_ref().and_then(|relative| {
            read(
                &Path::new("/sys/fs/cgroup")
                    .join(relative.trim_start_matches('/'))
                    .join(name),
            )
        })
    };
    Ok(PalsArenaResourceObservationV4 {
        domain: "rz-pals-arena-colab-observed-resources-v4/1".into(),
        stage: stage.into(), requested, actual_affinity, cgroup_v2_membership: cgroup_v2_membership.clone(),
        cpu_max: cpu_quota(read_limit("cpu.max").as_deref()),
        memory_high: limit(read_limit("memory.high").as_deref()),
        memory_max: limit(read_limit("memory.max").as_deref()),
        memory_swap_max: limit(read_limit("memory.swap.max").as_deref()),
        pids_max: limit(read_limit("pids.max").as_deref()),
        method: "read-only inherited /proc affinity and direct cgroup-v2 values; sequential snapshot, not atomic or lifetime enforcement".into(),
        strict_local_resources_admitted: false, resource_settings_installed: false,
        ancestor_effective_limits: "unknown".into(), performance_evidence_eligible: false,
        pair_eligible: false, strength_eligible: false,
    })
}

#[cfg(target_os = "linux")]
pub(super) fn admit(spec: &LockedPalsArenaLaunchV4, stage: &str) -> Result<Vec<u32>, ArenaError> {
    if spec.input.resource_profile.is_none() {
        return verify_pals_inherited_resources_v4(spec);
    }
    validate_profile(&spec.input)?;
    let observed = observe(spec, stage, |path| {
        let mut options = std::fs::OpenOptions::new();
        use std::os::unix::fs::OpenOptionsExt;
        options
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let mut file = options.open(path).ok()?;
        let mut bytes = Vec::new();
        file.by_ref()
            .take(64 * 1024 + 1)
            .read_to_end(&mut bytes)
            .ok()?;
        if bytes.len() > 64 * 1024 {
            return None;
        }
        String::from_utf8(bytes).ok()
    })?;
    let actual = observed.actual_affinity.clone();
    let mut observations = spec
        .resource_observations
        .lock()
        .map_err(|_| invalid("Colab resource observation owner poisoned"))?;
    require(
        observations.len() < 8,
        "Colab resource observation snapshot budget exceeded",
    )?;
    observations.push(observed);
    Ok(actual)
}

/// Save the already collected facts through the failed launch's retained
/// directory capability. Saving diagnostics grants no resource or exit admission.
#[cfg(target_os = "linux")]
pub fn save_pals_colab_failure_resources_v4(
    failure: &crate::NativePairFailure<LockedPalsArenaLaunchV4>,
) -> Result<Option<ArtifactRef>, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use std::io::Write;
    let Some(owner) = failure.launch_owner() else {
        return Ok(None);
    };
    let spec = owner.spec();
    if spec.input.resource_profile.is_none() {
        return Ok(None);
    }
    let observations = spec
        .resource_observations
        .lock()
        .map_err(|_| invalid("Colab failure resource observation owner poisoned"))?;
    let cause = failure.cause.to_string();
    require(
        cause.len() <= 8192,
        "Colab failure diagnostic exceeds its reservation",
    )?;
    let value = serde_json::json!({
        "domain": "rz-pals-arena-colab-failure-resources-v4/1",
        "arena_launch_sha256": spec.sha256,
        "native_receipt": failure.receipt_artifact,
        "original_failure": cause,
        "observations": *observations,
        "observation_scope": "already observed before/after admission; failure-time limits not re-read, ancestors unknown",
        "process_exit_is_provider_fence": false,
        "strict_local_resources_admitted": false,
        "performance_evidence_eligible": false,
        "pair_eligible": false,
        "strength_eligible": false,
        "training_executed": false,
    });
    let bytes = serde_json::to_vec(&value).map_err(|e| invalid(e.to_string()))?;
    require(
        bytes.len() as u64 <= FAILURE_METADATA_CAP,
        "Colab failure resource evidence exceeds 64KiB",
    )?;
    require(
        spec.pair_metadata_cap() == V4_METADATA_CAP + FAILURE_METADATA_CAP
            && spec.input.budget.max_output_bytes > spec.pair_metadata_cap(),
        "Colab failure diagnostic lacks its original pre-spawn reservation",
    )?;
    // A failed process can leave its output extent unknown. Reserve its entire
    // original output cap instead of declaring zero or a successful postcheck.
    let retained = failure
        .receipt
        .as_ref()
        .map_or(Some(0u64), |receipt| {
            receipt
                .artifacts
                .iter()
                .try_fold(0u64, |total, artifact| total.checked_add(artifact.bytes))
        })
        .and_then(|total| {
            total.checked_add(failure.receipt_artifact.as_ref().map_or(0, |a| a.bytes))
        });
    let reservation = spec
        .input
        .budget
        .max_output_bytes
        .checked_add(retained.ok_or_else(|| invalid("Colab failure evidence extent overflow"))?)
        .and_then(|total| total.checked_add(bytes.len() as u64));
    require(
        reservation.is_some_and(|total| total <= spec.input.budget.max_artifact_bytes),
        "Colab failure resource evidence reservation exceeds original artifact budget",
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let name = "pals-colab-failure-resources.v4.json";
    let mut file = owner
        .snapshot
        .directory
        .open_with(name, &options)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    Ok(Some(ArtifactRef {
        path: format!("{}/{name}", owner.snapshot.output_directory),
        sha256: digest(&bytes), bytes: bytes.len() as u64,
        source: "V4 Colab observed resource facts from failed retained owner; original failure/receipt retained".into(),
        license: "MIT execution evidence; external asset rights remain separate".into(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn limits_keep_numeric_max_unknown_distinct() {
        assert_eq!(
            limit(Some("15270146048\n")),
            PalsArenaObservedLimitV4::Numeric { value: 15270146048 }
        );
        assert_eq!(limit(Some("max\n")), PalsArenaObservedLimitV4::Max);
        for raw in [
            None,
            Some("unknown"),
            Some("+12"),
            Some("18446744073709551616"),
        ] {
            assert!(matches!(
                limit(raw),
                PalsArenaObservedLimitV4::Unknown { .. }
            ));
        }
        assert_eq!(
            cpu_quota(Some("max 100000")).quota,
            PalsArenaObservedLimitV4::Max
        );
        assert_eq!(cpu_quota(Some("200000 100000")).period_us, Some(100000));
        assert_eq!(cpu_quota(Some("200000 0")).period_us, None);
    }
    #[test]
    fn unsafe_or_ambiguous_membership_is_unknown() {
        assert_eq!(membership(Some("0::/\n")), Some("/".into()));
        assert_eq!(membership(Some("0::/owned\n")), Some("/owned".into()));
        for value in [
            "0::/../other",
            "0::/./other",
            "0::/a\n0::/b",
            "0::relative",
            "0::/a\t",
        ] {
            assert_eq!(membership(Some(value)), None);
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn cloud_actual_limits_are_separate_and_affinity_remains_required() {
        let input = super::super::tests::colab_fixture();
        let lock = input.lock().unwrap();
        let read = |path: &Path| match path.to_str().unwrap() {
            "/proc/self/status" => Some("Cpus_allowed_list:\t0,2\n".into()),
            "/proc/self/cgroup" => Some("0::/\n".into()),
            "/sys/fs/cgroup/memory.high" => Some("max\n".into()),
            "/sys/fs/cgroup/memory.max" | "/sys/fs/cgroup/memory.swap.max" => {
                Some("15270146048\n".into())
            }
            "/sys/fs/cgroup/cpu.max" => Some("max 100000\n".into()),
            _ => None,
        };
        let observation = observe(&lock, "test", read).unwrap();
        assert_eq!(observation.requested.memory_high_bytes, 6 << 30);
        assert_eq!(observation.memory_high, PalsArenaObservedLimitV4::Max);
        assert_eq!(
            observation.memory_max,
            PalsArenaObservedLimitV4::Numeric { value: 15270146048 }
        );
        assert!(matches!(
            observation.pids_max,
            PalsArenaObservedLimitV4::Unknown { .. }
        ));
        assert!(!observation.strict_local_resources_admitted && !observation.pair_eligible);
        assert!(!observation.performance_evidence_eligible && !observation.strength_eligible);
        assert!(
            observe(&lock, "drift", |path| {
                if path == Path::new("/proc/self/status") {
                    Some("Cpus_allowed_list:\t0-3\n".into())
                } else {
                    read(path)
                }
            })
            .is_err()
        );
    }
}
