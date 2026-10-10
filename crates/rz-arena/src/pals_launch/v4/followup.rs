//! V4 selected-owner binding. None keeps the historical Fresh-only recipe.
use super::*;

const METADATA_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;

fn present_binding_option<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d)
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsFollowupExecutionBindingV4 {
    pub domain: String,
    pub schema_version: u32,
    pub endpoints: [Option<PalsEndpointExecutionBindingV4>; 2],
    #[serde(deserialize_with = "present_binding_option")]
    pub archive_output: Option<PalsArchiveRunOutputBudgetV4>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsEndpointExecutionBindingV4 {
    pub owner_methods: PalsFollowupOwnerMethodsV4,
    #[serde(deserialize_with = "present_binding_option")]
    pub cold_archive: Option<PalsColdArchiveExecutionBindingV4>,
    #[serde(deserialize_with = "present_binding_option")]
    pub cuda_warm: Option<PalsCudaWarmExecutionBindingV4>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsColdArchiveExecutionBindingV4 {
    pub runtime_limits: PalsArchiveRuntimeLimitsV4,
    /// Actual repository exclusion boundary, never a synthesized runtime child.
    pub repository_root: String,
    pub root_layout: String,
    pub ram_release_method: String,
    pub cold_index_kind: String,
    pub global_scope_kind: String,
    pub search_io_bytes_max: u64,
    /// Prewrite debit in each restarted session; partial writes keep their cost.
    pub write_output_bytes_max: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaWarmExecutionBindingV4 {
    pub loader_implementation_sha256: String,
    pub export_schema: String,
    pub layout: String,
    pub layout_revision: u32,
    pub execution_domain: String,
    pub query_semantics: String,
    pub device_public_memory: bool,
    pub max_leases: u32,
    pub device_bytes_max: u64,
    pub device_bytes_scope: String,
    pub value_always_fresh: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsArchiveRunOutputBudgetV4 {
    pub metadata_bytes_max: u64,
    pub restarted_sessions_per_endpoint: [u32; 2],
    pub archive_write_bytes_max_per_endpoint: [u64; 2],
    pub total_output_bytes_max: u64,
}

fn selected(p: &PalsPoliciesV4) -> bool {
    p.recheck != PalsRecheckPolicyV4::Disabled
        || !matches!(p.paused_stack, PalsPausedStackPolicyV4::Disabled)
        || !matches!(p.cold_archive, PalsColdArchivePolicyV4::Disabled)
        || !matches!(p.cuda_warm, PalsCudaWarmPolicyV4::Disabled)
}

pub(super) fn validate_lifecycle_pair(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    startup: &serde_json::Value,
    termination: &serde_json::Value,
) -> Result<Option<serde_json::Value>, ArenaError> {
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    {
        super::lifecycle::validate_pair(lock, role, startup, termination)
    }
    #[cfg(not(any(feature = "pals-collection-onnx", feature = "native-cuda")))]
    {
        require(
            lock.input.execution_binding(role_index(role)).is_none()
                && startup["followup_lifecycle"].is_null()
                && termination["followup_lifecycle"].is_null(),
            "actual optional lifecycle needs its explicitly compiled Native producer",
        )?;
        Ok(None)
    }
}

impl PalsArenaLaunchV4 {
    pub fn execution_binding(&self, i: usize) -> Option<&PalsEndpointExecutionBindingV4> {
        self.followup_execution
            .as_ref()
            .and_then(|b| b.endpoints.get(i)?.as_ref())
    }

    pub(super) fn validate_followup_execution(&self) -> Result<u64, ArenaError> {
        let any = self
            .semantic_lock
            .manifest
            .engines
            .iter()
            .any(|e| matches!(e,PalsEngineV4::Pals(p) if selected(&p.policies)));
        let Some(binding) = &self.followup_execution else {
            require(
                !any,
                "selected V4 optional policies need an actual execution binding",
            )?;
            return Ok(METADATA_OUTPUT_BYTES);
        };
        require(
            any && binding.domain == PALS_FOLLOWUP_EXECUTION_V4_DOMAIN
                && binding.schema_version == 1,
            "unselected/unsupported V4 followup execution binding",
        )?;
        let mut archive_caps = [0u64; 2];
        let mut archive_sessions = [0u32; 2];
        for i in 0..2 {
            let expected = match &self.semantic_lock.manifest.engines[i] {
                PalsEngineV4::Pals(p) => Some(p),
                _ => None,
            };
            let Some(e) = expected.filter(|p| selected(&p.policies)) else {
                require(
                    binding.endpoints[i].is_none(),
                    "unselected/non-PALS endpoint has a followup binding",
                )?;
                continue;
            };
            let actual = binding.endpoints[i]
                .as_ref()
                .ok_or_else(|| invalid("selected endpoint owner method binding missing"))?;
            actual.owner_methods.validate_for(&e.policies)?;
            verify_compiled_owner_methods(&actual.owner_methods)?;
            match (&e.policies.cold_archive, &actual.cold_archive) {
                (PalsColdArchivePolicyV4::Disabled, None) => {}
                (PalsColdArchivePolicyV4::Bounded(l), Some(a)) => {
                    let r = &a.runtime_limits;
                    require(
                        r.game_bytes_max == l.game_bytes_max
                            && r.global_bytes_max == l.global_bytes_max
                            && r.index_entries_max == l.index_entries_max
                            && r.index_bytes_max == l.index_bytes_max
                            && r.load_bytes_max == l.load_bytes_max
                            && r.load_deadline_max_ms == l.load_deadline_max_ms
                            && r.record_payload_bytes_max == l.game_bytes_max.min(16 * 1024 * 1024)
                            && r.index_bytes_max <= 16 * 1024 * 1024
                            && r.load_bytes_max <= 16 * 1024 * 1024
                            && r.max_load_pins == 16
                            && a.root_layout == "endpoint_owned_managed_root_v4"
                            && a.ram_release_method == "hot_unique_owned_capacity_subset"
                            && a.cold_index_kind == "directory_scan"
                            && a.global_scope_kind == "canonical_no_link_managed_root"
                            && a.search_io_bytes_max == 16 * 1024 * 1024
                            && a.write_output_bytes_max == l.game_bytes_max
                            && Path::new(&a.repository_root).is_absolute()
                            && a.repository_root.len() <= 4096
                            && !a.repository_root.chars().any(char::is_control)
                            && self.endpoints[i].native_model().is_some(),
                        "archive actual selector/repository/method/original I/O/output bound differs",
                    )?;
                    archive_caps[i] = l
                        .game_bytes_max
                        .checked_mul(2)
                        .ok_or_else(|| invalid("archive session output overflow"))?;
                    archive_sessions[i] = 2;
                }
                _ => return Err(invalid("archive policy and execution owner binding differ")),
            }
            match (&e.policies.cuda_warm, &actual.cuda_warm) {
                (PalsCudaWarmPolicyV4::Disabled, None) => {}
                (PalsCudaWarmPolicyV4::ApproxWarm(l), Some(w)) => {
                    let cuda = self.endpoints[i]
                        .cuda_model()
                        .ok_or_else(|| invalid("CUDA Warm requires explicit CUDA native recipe"))?;
                    require(
                        actual.owner_methods.cuda_warm.as_ref().is_some_and(|m| {
                            w.loader_implementation_sha256 == m.implementation_sha256
                        }) && w.export_schema == "rovezero.pals-private-cuda-warm.v2"
                            && w.layout == "shared_pc_if_approx_cuda_warm_v2"
                            && w.layout_revision == 2
                            && w.execution_domain == "cuda_device_io_binding_v2"
                            && w.query_semantics
                                == "rz-pals-private-cuda-query/2;same-actual-nonrecord-fp32-and-full-lines;logical-game-search-situation-prefix-focus-role-isolated;records-revision-current;original-deadline-cancel-fixed;value-fresh"
                            && w.device_public_memory
                            && w.value_always_fresh
                            && w.max_leases == 1
                            && w.max_leases == l.max_leases
                            && w.device_bytes_max == l.device_bytes_max
                            && w.device_bytes_max
                                <= self.semantic_lock.manifest.resources[i]
                                    .device_allocation_max_bytes
                            && w.device_bytes_scope == "explicit_cuda_kv_payload_bytes"
                            && cuda.cuda_record_pages.is_none()
                            && cuda.cuda_control.is_some()
                            && cuda
                                .cuda_control
                                .as_ref()
                                .is_some_and(|c| c.loading_profile.is_some())
                            && config(e).profile.uses_full_line(),
                        "CUDA Warm requires exact FullLine device owner/export/inventory/loading profile/limits",
                    )?;
                    verify_selected_warm_adapter(&cuda.model)?;
                }
                _ => {
                    return Err(invalid(
                        "CUDA Warm policy and actual execution binding differ",
                    ));
                }
            }
        }
        let archive_total = archive_caps[0]
            .checked_add(archive_caps[1])
            .and_then(|v| v.checked_add(METADATA_OUTPUT_BYTES))
            .ok_or_else(|| invalid("archive output budget overflow"))?;
        if archive_caps != [0, 0] {
            let out = binding
                .archive_output
                .as_ref()
                .ok_or_else(|| invalid("archive actual run output budget missing"))?;
            require(
                out.metadata_bytes_max == METADATA_OUTPUT_BYTES
                    && out.restarted_sessions_per_endpoint == archive_sessions
                    && out.archive_write_bytes_max_per_endpoint == archive_caps
                    && out.total_output_bytes_max == archive_total
                    && archive_total <= 2 * 1024 * 1024 * 1024
                    && self.budget.max_output_bytes == archive_total
                    && self.semantic_lock.manifest.output_bytes_max == archive_total,
                "archive run cap must equal finite metadata plus actual selected quota times two sessions",
            )?;
        } else {
            require(
                binding.archive_output.is_none(),
                "unselected archive output allowance",
            )?;
        }
        Ok(archive_total)
    }
}

pub(super) fn validate_selected_private_execution(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    record: &serde_json::Value,
) -> Result<(), ArenaError> {
    let Some(w) = lock
        .input
        .execution_binding(role_index(role))
        .and_then(|b| b.cuda_warm.as_ref())
    else {
        return require_no_undeclared_private_warm(record);
    };
    let n = lock.native(role)?;
    let declaration = &record["execution"]["private_warm"];
    require(
        declaration.is_object()
            && record["execution"]["provider"] == "cuda"
            && record["execution"]["device_public_memory"] == true
            && declaration["schema"] == w.export_schema
            && declaration["query_semantics"] == w.query_semantics
            && declaration["native_cpu_loaded_capability"] == false
            && declaration["value_always_fresh"] == true
            && declaration["frozen_contexts_per_role"] == 0
            && declaration["accepted_seeds_per_role"] == 1
            && array_hash(&declaration["export_manifest_sha256"])? == n.export.sha256
            && array_hash(&declaration["loader_implementation_sha256"])?
                == w.loader_implementation_sha256
            && declaration["model_epoch"] == record["model_epoch"]
            && declaration["encoding_semantic_sha256"] == record["encoding_semantic_sha256"]
            && declaration["frozen_epoch"] == record["frozen_epoch"],
        "selected CUDA Warm declaration differs from actual loaded device/export/namespace",
    )?;
    verify_selected_warm_bank(record)?;
    // Incomplete legacy bank observations are preserved by the actual lifecycle
    // recorder; absence/unavailability must never be a Fresh-only coercion.
    require(
        (record["private_warm_observation"].is_object()
            && record["private_warm_observation_unavailable"].is_null())
            || (record["private_warm_observation"].is_null()
                && record["private_warm_observation_unavailable"] == true),
        "selected CUDA Warm owner observation missing",
    )
}

#[cfg(feature = "pals-followup-cuda-warm")]
fn verify_selected_warm_adapter(n: &PalsOnnxLaunchV3) -> Result<(), ArenaError> {
    require(
        n.adapter_source_sha256
            == hex(rz_uci::pals_native::pals_native_cuda_warm_adapter_source_digest()),
        "CUDA Warm adapter declaration differs from its actual compiled composite source getter",
    )
}
#[cfg(not(feature = "pals-followup-cuda-warm"))]
fn verify_selected_warm_adapter(_n: &PalsOnnxLaunchV3) -> Result<(), ArenaError> {
    Err(invalid(
        "CUDA Warm adapter requires explicit pals-followup-cuda-warm feature",
    ))
}
#[cfg(feature = "pals-followup-cuda-warm")]
fn verify_selected_warm_bank(record: &serde_json::Value) -> Result<(), ArenaError> {
    require(
        array_hash(&record["execution"]["private_warm"]["implementation_sha256"])?
            == hex(rz_uci::pals_native::pals_native_private_warm_source_digest()),
        "CUDA Warm native bank pin differs from the actual compiled bank boundary",
    )
}
#[cfg(not(feature = "pals-followup-cuda-warm"))]
fn verify_selected_warm_bank(_record: &serde_json::Value) -> Result<(), ArenaError> {
    Err(invalid(
        "CUDA Warm bank verification requires explicit pals-followup-cuda-warm feature",
    ))
}

#[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
fn verify_compiled_owner_methods(m: &PalsFollowupOwnerMethodsV4) -> Result<(), ArenaError> {
    require(
        m.producer_implementation_sha256
            == hex(rz_uci::pals_attestation::followup::compiled_followup_producer_sha256()),
        "followup producer pin differs from the actual compiled wire owner",
    )?;
    if let Some(p) = &m.repair {
        require(
            p.implementation_sha256
                == hex(rz_search::pals::engine::compiled_search_implementation_sha256()),
            "repair observer compiled source pin differs",
        )?;
    }
    if let Some(p) = &m.paused_stack {
        require(
            p.implementation_sha256 == hex(rz_search::cpu::compiled_paused_stack_owner_sha256()),
            "paused frame owner compiled source pin differs",
        )?;
    }
    if let Some(p) = &m.cold_archive {
        require(
            p.implementation_sha256
                == hex(rz_search::pals::store::archive_accounting_source_sha256()),
            "archive actual owner accounting source pin differs",
        )?;
    }
    if m.cuda_warm.is_some() {
        verify_compiled_warm_owner(m)?;
    }
    Ok(())
}
#[cfg(not(any(feature = "pals-collection-onnx", feature = "native-cuda")))]
fn verify_compiled_owner_methods(_m: &PalsFollowupOwnerMethodsV4) -> Result<(), ArenaError> {
    Err(invalid(
        "actual followup producer verification requires an explicit PALS native feature",
    ))
}

#[cfg(all(
    not(feature = "pals-followup-cuda-warm"),
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
fn verify_compiled_warm_owner(_m: &PalsFollowupOwnerMethodsV4) -> Result<(), ArenaError> {
    Err(invalid(
        "CUDA Warm execution requires the explicitly compiled pals-followup-cuda-warm feature",
    ))
}
#[cfg(feature = "pals-followup-cuda-warm")]
fn verify_compiled_warm_owner(m: &PalsFollowupOwnerMethodsV4) -> Result<(), ArenaError> {
    require(
        m.cuda_warm.as_ref().is_some_and(|w| {
            w.implementation_sha256
                == hex(rz_eval::pals_onnx::cuda_private_warm_implementation_digest())
        }),
        "CUDA actual buffer/lease owner source pin differs",
    )
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
pub(super) struct ArchiveRootPins {
    pub root: PathBuf,
    pub repository: PathBuf,
    root_pin: std::fs::File,
    repository_pin: std::fs::File,
}

#[cfg(target_os = "linux")]
impl ArchiveRootPins {
    fn verify(&self) -> Result<(), ArenaError> {
        use std::os::unix::fs::MetadataExt;
        for (path, pin) in [
            (&self.root, &self.root_pin),
            (&self.repository, &self.repository_pin),
        ] {
            let current = std::fs::symlink_metadata(path)
                .map_err(|e| invalid(format!("archive owner root metadata:{e}")))?;
            let held = pin
                .metadata()
                .map_err(|e| invalid(format!("archive owner held root metadata:{e}")))?;
            require(
                current.is_dir()
                    && !current.file_type().is_symlink()
                    && current.dev() == held.dev()
                    && current.ino() == held.ino()
                    && path.canonicalize().is_ok_and(|p| p == *path),
                "archive root/repository ownership or canonical inode changed",
            )?;
        }
        require(
            !self.root.starts_with(&self.repository),
            "archive managed root lies inside repository",
        )
    }
}

impl LockedPalsArenaLaunchV4 {
    #[cfg(target_os = "linux")]
    pub(super) fn archive_runtime_arguments(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        use std::os::unix::fs::OpenOptionsExt;
        let Some(binding) = self
            .input
            .execution_binding(role_index(role))
            .and_then(|b| b.cold_archive.as_ref())
        else {
            return Ok(vec![]);
        };
        let root = runtime_root.join("cold-archive-managed");
        let mut roots = self
            .archive_roots
            .lock()
            .map_err(|_| invalid("archive Arena root owner poisoned"))?;
        if !roots.contains_key(&role_index(role)) {
            require(
                runtime_root.is_absolute()
                    && runtime_root.canonicalize().is_ok_and(|p| p == runtime_root),
                "archive parent must be the actual canonical owned runtime root",
            )?;
            let repository = Path::new(&binding.repository_root)
                .canonicalize()
                .map_err(|e| invalid(format!("actual archive repository boundary:{e}")))?;
            require(
                repository == Path::new(&binding.repository_root)
                    && std::fs::symlink_metadata(repository.join(".git")).is_ok()
                    && !root.starts_with(&repository),
                "archive repository boundary must be the actual canonical Git root outside output",
            )?;
            std::fs::create_dir(&root)
                .map_err(|e| invalid(format!("exclusive archive owner root creation:{e}")))?;
            let open_dir = |path: &Path| {
                std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
                    .open(path)
                    .map_err(|e| invalid(format!("archive owner directory pin:{e}")))
            };
            let pins = ArchiveRootPins {
                root: root.clone(),
                repository: repository.clone(),
                root_pin: open_dir(&root)?,
                repository_pin: open_dir(&repository)?,
            };
            pins.verify()?;
            roots.insert(role_index(role), pins);
        }
        let pins = &roots[&role_index(role)];
        require(
            pins.root == root,
            "archive owner cannot move to another runtime root",
        )?;
        pins.verify()?;
        Ok(vec![
            format!(
                "--pals-archive-root={}",
                pins.root
                    .to_str()
                    .ok_or_else(|| invalid("archive root UTF8"))?
            )
            .into(),
            format!(
                "--pals-archive-repository={}",
                pins.repository
                    .to_str()
                    .ok_or_else(|| invalid("archive repository UTF8"))?
            )
            .into(),
        ])
    }
    #[cfg(not(target_os = "linux"))]
    pub(super) fn archive_runtime_arguments(
        &self,
        _role: NativeEngineRole,
        _runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        require(
            self.input.followup_execution.as_ref().is_none_or(|b| {
                b.endpoints
                    .iter()
                    .flatten()
                    .all(|e| e.cold_archive.is_none())
            }),
            "actual archive execution requires Linux",
        )?;
        Ok(vec![])
    }
    #[cfg(target_os = "linux")]
    pub(super) fn verify_archive_scope_roots(
        &self,
        role: NativeEngineRole,
        scope: &PalsArchiveActualScopeV4,
    ) -> Result<(), ArenaError> {
        let roots = self
            .archive_roots
            .lock()
            .map_err(|_| invalid("archive root owner poisoned"))?;
        let pins = roots
            .get(&role_index(role))
            .ok_or_else(|| invalid("actual archive has no Arena pinned root owner"))?;
        pins.verify()?;
        require(
            scope.root_sha256
                == digest(
                    pins.root
                        .to_str()
                        .ok_or_else(|| invalid("archive root UTF8"))?
                        .as_bytes(),
                )
                && scope.repository_sha256
                    == digest(
                        pins.repository
                            .to_str()
                            .ok_or_else(|| invalid("archive repository UTF8"))?
                            .as_bytes(),
                    ),
            "actual Store canonical root/repository hashes differ from Arena held owners",
        )
    }
    #[cfg(target_os = "linux")]
    pub(super) fn archive_existing_arguments(
        &self,
        role: NativeEngineRole,
    ) -> Result<Vec<OsString>, ArenaError> {
        if self
            .input
            .execution_binding(role_index(role))
            .and_then(|b| b.cold_archive.as_ref())
            .is_none()
        {
            return Ok(vec![]);
        }
        let roots = self
            .archive_roots
            .lock()
            .map_err(|_| invalid("archive root owner poisoned"))?;
        let pins = roots
            .get(&role_index(role))
            .ok_or_else(|| invalid("preflight must use the already held game archive root"))?;
        pins.verify()?;
        Ok(vec![
            format!(
                "--pals-archive-root={}",
                pins.root
                    .to_str()
                    .ok_or_else(|| invalid("archive root UTF8"))?
            )
            .into(),
            format!(
                "--pals-archive-repository={}",
                pins.repository
                    .to_str()
                    .ok_or_else(|| invalid("archive repository UTF8"))?
            )
            .into(),
        ])
    }
    #[cfg(not(target_os = "linux"))]
    pub(super) fn archive_existing_arguments(
        &self,
        _role: NativeEngineRole,
    ) -> Result<Vec<OsString>, ArenaError> {
        Ok(vec![])
    }
}

pub(super) fn selected_policy_arguments(
    e: &PalsEndpointV4,
    binding: Option<&PalsEndpointExecutionBindingV4>,
) -> Result<Vec<String>, ArenaError> {
    let mut args = Vec::new();
    if selected(&e.policies) {
        require(
            binding.is_some(),
            "selected owner binding missing before argv",
        )?;
    }
    if let Some(a) = binding.and_then(|b| b.cold_archive.as_ref()) {
        let r = &a.runtime_limits;
        args.extend([
            format!("--pals-archive-game-bytes-max={}", r.game_bytes_max),
            format!("--pals-archive-global-bytes-max={}", r.global_bytes_max),
            format!("--pals-archive-index-entries-max={}", r.index_entries_max),
            format!("--pals-archive-index-bytes-max={}", r.index_bytes_max),
            format!("--pals-archive-load-bytes-max={}", r.load_bytes_max),
            format!(
                "--pals-archive-load-deadline-max-ms={}",
                r.load_deadline_max_ms
            ),
        ]);
    }
    if let Some(w) = binding.and_then(|b| b.cuda_warm.as_ref()) {
        args.extend([
            format!("--pals-cuda-warm-max-leases={}", w.max_leases),
            format!("--pals-cuda-warm-device-bytes-max={}", w.device_bytes_max),
        ]);
    }
    Ok(args)
}
