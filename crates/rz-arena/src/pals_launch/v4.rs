//! PALS V4 execution through the pinned arena owner and physical lifecycle.
//! V3 recipe DTOs below describe unchanged executable resources only. The V4
//! semantic lock, argv, observed producer policies and receipts retain V4 IDs.
//! V4 base.model.implementation_sha256 pins compiled model-boundary sources;
//! the native recipe's adapter_source_sha256 independently pins the native
//! adapter. Runtime getters attest each against its own declared scope.
use super::*;
use rz_eval::pals_model::{PalsModelConfig, PalsModelProfile};
use rz_experiments::*;

pub const PALS_ARENA_V4_DOMAIN: &str = "rz-pals-arena-launch-v4/1";
pub const PALS_NATIVE_STARTUP_V4_DOMAIN: &str = "rz-pals-native-startup-v4/1";
pub const PALS_NATIVE_TERMINATION_V4_DOMAIN: &str = "rz-pals-native-termination-v4/1";
const V4_METADATA_CAP: u64 = 4 * 1024 * 1024;
pub const PALS_SEARCH_WORK_STARTUP_V4_DOMAIN: &str = "rz-pals-search-work-startup-v4/1";
pub const PALS_SEARCH_WORK_TERMINATION_V4_DOMAIN: &str = "rz-pals-search-work-termination-v4/1";

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
#[cfg(any(feature = "pals-collection", feature = "native-cuda"))]
fn validate_actual_encoding(e: &PalsEndpointV4, n: &PalsOnnxLaunchV3) -> Result<(), ArenaError> {
    let cfg = config(e);
    require(
        n.encoding_semantic_sha256
            == hex(rz_uci::pals_native::pals_rules_encoding_semantic_digest_for_config(&cfg))
            && e.model_v2.encoding_sha256
                == hex(rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(&cfg)),
        "V4 actual model/Rules encoding differs from declared profile",
    )
}
#[cfg(not(any(feature = "pals-collection", feature = "native-cuda")))]
fn validate_actual_encoding(_e: &PalsEndpointV4, _n: &PalsOnnxLaunchV3) -> Result<(), ArenaError> {
    Err(invalid(
        "native V4 encoding verification requires the explicitly selected PALS feature",
    ))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsArenaLaunchV4 {
    pub domain: String,
    pub semantic_lock: PalsInputLockV4,
    pub runner: ToolIdentity,
    pub opening_artifact: ArtifactRef,
    pub endpoints: [PalsEndpointLaunchV3; 2],
    pub budget: NativeResourceBudgetV1,
}
#[derive(Clone, Debug)]
pub struct LockedPalsArenaLaunchV4 {
    input: PalsArenaLaunchV4,
    sha256: String,
    opening: OpeningSpec,
    endpoint_views: [ExternalUciEndpointV2; 2],
}
fn requested_options(engine: &PalsEngineV4) -> &BTreeMap<String, String> {
    match engine {
        PalsEngineV4::Pals(e) => &e.base.requested_options,
        PalsEngineV4::OwnCpu(e) => &e.requested_options,
        PalsEngineV4::ReferenceUci(e) => &e.requested_options,
    }
}
fn config(e: &PalsEndpointV4) -> PalsModelConfig {
    let profile = match e.model_v2.profile {
        PalsModelProfileV4::LegacySummaryV1 => PalsModelProfile::LegacySummaryV1,
        PalsModelProfileV4::FullLineV2 => PalsModelProfile::FullLineV2,
        PalsModelProfileV4::InteractionHeadV2 => PalsModelProfile::InteractionHeadV2,
        PalsModelProfileV4::FullLineInteractionV2 => PalsModelProfile::FullLineInteractionV2,
    };
    PalsModelConfig::for_profile(profile)
}
fn policy_arguments(e: &PalsEndpointV4) -> Result<Vec<String>, ArenaError> {
    if let PalsIterativeRepairPolicyV4::Bounded(l) = &e.policies.iterative_repair {
        require(
            l.max_repairs_per_first_move == 3 && l.max_pending_questions == 64,
            "native queue CLI registers exactly three Repairs and 64 pending questions; lower declared limits need their own actual producer selector",
        )?;
    }
    if let PalsPausedStackPolicyV4::Bounded(l) = &e.policies.paused_stack {
        require(
            l.tokens_max == 1 && l.bytes_max == 8 * 1024 * 1024,
            "native PausedStack selector registers one token and 8MiB; declared limits must equal the actual selected owner",
        )?;
    }
    let mut args = vec![
        "--pals-arena-wire=v4".into(),
        format!("--pals-model-profile={}", config(e).profile.as_str()),
        format!(
            "--pals-resolver-policy={}",
            match e.policies.resolver {
                PalsResolverPolicyV4::OwnRaw => "own-raw-restricted",
                PalsResolverPolicyV4::ModelWdl => "model-wdl-restricted",
            }
        ),
        "--pals-cuda-private-warm=false".into(),
    ];
    if e.base.cpu_r.is_own() {
        args.push(format!(
            "--pals-cpu-resume={}",
            match e.policies.paused_stack {
                PalsPausedStackPolicyV4::Disabled => "completed-iteration",
                PalsPausedStackPolicyV4::Bounded(_) => "paused-stack",
            }
        ));
    }
    match (&e.policies.recheck, &e.policies.iterative_repair) {
        (PalsRecheckPolicyV4::Disabled, PalsIterativeRepairPolicyV4::Disabled) => {}
        (PalsRecheckPolicyV4::FrozenWdl, PalsIterativeRepairPolicyV4::Disabled) => {
            args.push("--pals-post-repair-recheck=frozen-model-wdl-v2".into())
        }
        (PalsRecheckPolicyV4::FrozenWdl, PalsIterativeRepairPolicyV4::Bounded(_)) => {
            args.push("--pals-post-repair-recheck=iterative-frozen-model-wdl-v2".into())
        }
        _ => return Err(invalid("iterative repair requires frozen WDL recheck")),
    }
    // CUDA Warm/archive require actual native capability and owner plumbing.
    // Unsupported non-default selections fail before snapshot/spawn.
    require(
        matches!(e.policies.cuda_warm, PalsCudaWarmPolicyV4::Disabled),
        "CUDA ApproxWarm arena launch capability is not registered by this executable; no host or CPU fallback",
    )?;
    require(
        matches!(e.policies.cold_archive, PalsColdArchivePolicyV4::Disabled),
        "cold archive arena owner binding must be explicitly registered before launch",
    )?;
    Ok(args)
}

impl PalsArenaValidationContext for LockedPalsArenaLaunchV4 {
    fn verify_semantic(&self) -> Result<(), ArenaError> {
        self.input.semantic_lock.verify().map_err(ArenaError::from)
    }
    fn launch_sha256(&self) -> &str {
        &self.sha256
    }
    fn native(&self, role: NativeEngineRole) -> Result<&PalsOnnxLaunchV3, ArenaError> {
        LockedPalsArenaLaunchV4::native(self, role)
    }
    fn recipe(&self, role: NativeEngineRole) -> &PalsEndpointLaunchV3 {
        &self.input.endpoints[role_index(role)]
    }
    fn endpoint_view(&self, role: NativeEngineRole) -> &ExternalUciEndpointV2 {
        &self.endpoint_views[role_index(role)]
    }
    fn pals_endpoint(&self, role: NativeEngineRole) -> Result<&PalsEndpointV3, ArenaError> {
        match &self.input.semantic_lock.manifest.engines[role_index(role)] {
            PalsEngineV4::Pals(e) => Ok(&e.base),
            _ => Err(invalid("native V4 requires actual PALS model identity")),
        }
    }
    fn resource(&self, role: NativeEngineRole) -> &PalsResourcePolicyV3 {
        &self.input.semantic_lock.manifest.resources[role_index(role)]
    }
    fn external_binding(
        &self,
        role: NativeEngineRole,
    ) -> Result<Option<(&PalsExternalCpuRV3, &PalsExternalCpuRLaunchV3)>, ArenaError> {
        self.input.external_cpu_r_binding(role_index(role))
    }
    fn native_wire_version(&self) -> u32 {
        4
    }
    fn native_wire_domains(&self) -> (&'static str, &'static str) {
        (
            PALS_NATIVE_STARTUP_V4_DOMAIN,
            PALS_NATIVE_TERMINATION_V4_DOMAIN,
        )
    }
    fn native_wire_filenames(&self) -> (&'static str, &'static str) {
        (
            "pals-native-startup.v4.json",
            "pals-native-termination.v4.json",
        )
    }
    fn validate_policy_pair(
        &self,
        role: NativeEngineRole,
        startup: &serde_json::Value,
        termination: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        validate_actual_policy_work(self, role, startup, termination)
    }
    fn validate_wire_pair(
        &self,
        role: NativeEngineRole,
        startup: &serde_json::Value,
        termination: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        validate_actual_policy_work(
            self,
            role,
            &startup["search_work"],
            &termination["search_work"],
        )?;
        validate_followup_marker_pair(self, role, startup, termination).map(|_| ())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsActualFollowupMarkerV4 {
    pub model_identity: String,
    pub model_profile: Option<String>,
    pub model_semantics: String,
    pub model_semantics_sha256: Option<[u8; 32]>,
    pub model_implementation_sha256: Option<[u8; 32]>,
    pub encoding_schema: Option<String>,
    pub model_configuration: Option<PalsModelConfig>,
    pub model_epoch: Option<[u8; 32]>,
    pub resolver_policy: String,
    pub cpu_resume: Option<String>,
    pub post_repair_recheck: String,
    pub policy_identity: String,
    pub policy_conditions_sha256: Option<[u8; 32]>,
    pub search_implementation_sha256: [u8; 32],
    pub resolver_implementation_sha256: [u8; 32],
    pub resolver_semantics_sha256: [u8; 32],
    pub cuda_warm: bool,
}

fn expected_recheck(e: &PalsEndpointV4) -> &'static str {
    match (&e.policies.recheck, &e.policies.iterative_repair) {
        (PalsRecheckPolicyV4::Disabled, _) => "disabled",
        (_, PalsIterativeRepairPolicyV4::Disabled) => "frozen-model-wdl-v2",
        _ => "iterative-frozen-model-wdl-v2",
    }
}
fn validate_actual_policy_work(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    s: &serde_json::Value,
    t: &serde_json::Value,
) -> Result<(), ArenaError> {
    let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[role_index(role)] else {
        return require(
            s["pals_resolver"].is_null()
                && t["pals_resolver"].is_null()
                && s["pals_search_policy"].is_null()
                && t["pals_search_policy"].is_null(),
            "non-PALS work cannot claim a PALS policy",
        );
    };
    let (version, semantics) = match e.policies.resolver {
        PalsResolverPolicyV4::OwnRaw => (
            rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
            rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS,
        ),
        PalsResolverPolicyV4::ModelWdl => (
            rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
            rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
        ),
    };
    let semantic_sha: [u8; 32] = Sha256::digest(semantics.as_bytes()).into();
    require(
        e.policies.resolver_identity.semantic_id == version
            && e.policies.resolver_semantics_sha256 == hex(semantic_sha),
        "V4 resolver source registration differs",
    )?;
    for work in [s, t] {
        require(
            work["pals_resolver"]["version"] == version
                && work["pals_resolver"]["semantics_sha256"] == serde_json::json!(semantic_sha),
            "actual V4 resolver differs from locked resolver",
        )?;
        match e.policies.recheck {
            PalsRecheckPolicyV4::Disabled => require(
                work["pals_search_policy"].is_null(),
                "disabled V4 recheck cannot inherit another policy",
            )?,
            PalsRecheckPolicyV4::FrozenWdl => {
                let selected: PalsPolicyIdentityV4 =
                    serde_json::from_value(work["pals_search_policy"].clone())
                        .map_err(|e| invalid(format!("actual V4 policy marker: {e}")))?;
                require(
                    selected == e.policy_identity,
                    "actual V4 recheck registration/conditions differ",
                )?;
            }
        }
    }
    Ok(())
}
fn validate_followup_marker_pair(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    s: &serde_json::Value,
    t: &serde_json::Value,
) -> Result<Option<PalsActualFollowupMarkerV4>, ArenaError> {
    let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[role_index(role)] else {
        require(
            s["v4"].is_null() && t["v4"].is_null(),
            "non-PALS source cannot inherit a followup model marker",
        )?;
        return Ok(None);
    };
    require(
        s["v4"].is_object() && s["v4"] == t["v4"],
        "actual V4 selection marker absent or changed",
    )?;
    let actual: PalsActualFollowupMarkerV4 = serde_json::from_value(s["v4"].clone())
        .map_err(|e| invalid(format!("V4 producer marker: {e}")))?;
    require(
        actual.resolver_policy
            == match e.policies.resolver {
                PalsResolverPolicyV4::OwnRaw => "own-raw-restricted",
                PalsResolverPolicyV4::ModelWdl => "model-wdl-restricted",
            }
            && actual.cpu_resume.as_deref()
                == if e.base.cpu_r.is_own() {
                    Some(match e.policies.paused_stack {
                        PalsPausedStackPolicyV4::Disabled => "completed-iteration",
                        _ => "paused-stack",
                    })
                } else {
                    None
                }
            && actual.post_repair_recheck == expected_recheck(e)
            && actual.policy_identity == e.policy_identity.search_identity
            && actual.policy_conditions_sha256 == Some(e.policy_identity.conditions_sha256)
            && actual.search_implementation_sha256
                == rz_search::pals::engine::compiled_search_implementation_sha256()
            && hex(actual.search_implementation_sha256) == e.base.search.implementation_sha256
            && actual.resolver_implementation_sha256
                == rz_search::pals::engine::compiled_resolver_implementation_sha256()
            && hex(actual.resolver_implementation_sha256)
                == e.policies.resolver_identity.implementation_sha256
            && hex(actual.resolver_semantics_sha256) == e.policies.resolver_semantics_sha256
            && actual.cuda_warm
                == matches!(e.policies.cuda_warm, PalsCudaWarmPolicyV4::ApproxWarm(_)),
        "actual V4 resolver/stack/recheck/conditions/CUDA Warm selection differs",
    )?;
    if lock.input.endpoints[role_index(role)]
        .native_model()
        .is_some()
    {
        let checkpoint = match &e.base.model.weights {
            PalsWeightIdentityV3::Untrained { artifact, .. }
            | PalsWeightIdentityV3::Trained { artifact, .. } => artifact,
            _ => return Err(invalid("native V4 model has no checkpoint")),
        };
        require(
            actual.model_identity == e.model_v2.model_identity
                && actual.model_profile.as_deref() == Some(config(e).profile.as_str())
                && actual.model_semantics == e.model_v2.model_semantics
                && actual.model_semantics_sha256
                    == Some(config(e).profile.registered_semantics_sha256())
                && actual
                    .model_semantics_sha256
                    .is_some_and(|sha| hex(sha) == e.model_v2.model_semantics_sha256)
                && actual.model_implementation_sha256
                    == Some(rz_eval::pals_model::compiled_model_boundary_sha256())
                && actual
                    .model_implementation_sha256
                    .is_some_and(|sha| hex(sha) == e.base.model.implementation_sha256)
                && actual.encoding_schema.as_deref() == Some(e.model_v2.encoding_schema.as_str())
                && actual.model_configuration.as_ref() == Some(&config(e))
                && actual
                    .model_epoch
                    .is_some_and(|epoch| hex(epoch) == checkpoint.sha256),
            "actual loaded V4 model/profile/encoding/checkpoint differs",
        )?;
    } else {
        require(
            actual.model_profile.is_none()
                && actual.model_configuration.is_none()
                && actual.model_epoch.is_none()
                && actual.model_semantics_sha256.is_none()
                && actual.model_implementation_sha256.is_none()
                && actual.encoding_schema.is_none(),
            "explicit legal-order mock cannot claim a loaded neural model",
        )?;
    }
    Ok(Some(actual))
}

fn verify_external_profiles(
    input: &PalsArenaLaunchV4,
    source_root: &Path,
) -> Result<(), ArenaError> {
    for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
        if let Some((_, binding)) = input.external_cpu_r_binding(role_index(role))? {
            #[cfg(all(
                target_os = "linux",
                any(feature = "pals-collection-onnx", feature = "native-cuda")
            ))]
            verify_external_profile(
                &input.semantic_lock,
                role,
                source_root,
                Path::new(&binding.registered_program),
                Path::new(&binding.registered_working_directory),
            )?;
            #[cfg(not(all(
                target_os = "linux",
                any(feature = "pals-collection-onnx", feature = "native-cuda")
            )))]
            {
                let _ = (binding, source_root);
                return Err(invalid(
                    "external V4 checker profile verification requires the explicit native feature",
                ));
            }
        }
    }
    Ok(())
}
impl PalsArenaLaunchV4 {
    fn external_cpu_r_binding(
        &self,
        i: usize,
    ) -> Result<Option<(&PalsExternalCpuRV3, &PalsExternalCpuRLaunchV3)>, ArenaError> {
        let binding = self.endpoints[i]
            .native_model()
            .and_then(|native| native.external_cpu_r.as_ref());
        match &self.semantic_lock.manifest.engines[i] {
            PalsEngineV4::Pals(endpoint) => match &endpoint.base.cpu_r {
                PalsCpuRSelectionV3::Own => {
                    require(
                        binding.is_none(),
                        "own CPU_R cannot inherit an external launch binding",
                    )?;
                    Ok(None)
                }
                PalsCpuRSelectionV3::ExternalUci(declaration) => {
                    let binding = binding.ok_or_else(|| {
                        invalid("external CPU_R requires an explicit native P/C launch binding")
                    })?;
                    binding.validate()?;
                    Ok(Some((declaration, binding)))
                }
            },
            _ => {
                require(
                    binding.is_none(),
                    "non-PALS endpoint cannot inherit an external CPU_R launch binding",
                )?;
                Ok(None)
            }
        }
    }
    pub fn from_json(text: &str) -> Result<Self, ArenaError> {
        require(text.len() <= MAX_JSON_BYTES, "launch JSON budget exceeded")?;
        let input: Self = crate::decode_json(text)?;
        input.validate()?;
        Ok(input)
    }
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.semantic_lock.verify()?;
        require(self.domain == PALS_ARENA_V4_DOMAIN, "wrong V4 arena domain")?;
        let m = &self.semantic_lock.manifest;
        require(
            m.purpose == PalsRunPurposeV4::ArenaPilot,
            "diagnostic declarations cannot launch arena",
        )?;
        require(
            m.pilot.max_plies % 2 == 0,
            "paired runner requires even max ply count",
        )?;
        self.require_reap_status_runner()?;
        self.runner.binary.validate()?;
        self.opening_artifact.validate()?;
        for a in self.declared_artifacts() {
            a.validate()?;
            require_public_artifact(a)?;
        }
        let b = self.budget;
        let unique = self.unique_input_bytes()?;
        require(
            unique <= b.max_input_bytes
                && b.max_input_bytes > 0
                && b.max_output_bytes > V4_METADATA_CAP
                && b.max_output_bytes <= 64 * 1024 * 1024
                && b.max_output_bytes <= m.output_bytes_max
                && b.max_runtime_bytes > 0
                && b.max_runtime_bytes <= 10 * 1024 * 1024 * 1024
                && (3..=16).contains(&b.max_child_processes)
                && (16..=4096).contains(&b.max_runtime_files)
                && (2..=8).contains(&b.max_runtime_depth)
                && b.address_space_per_process_bytes == 0
                && unique
                    .checked_add(b.max_output_bytes)
                    .and_then(|x| x.checked_add(b.max_runtime_bytes))
                    .is_some_and(|x| x <= b.max_artifact_bytes),
            "V4 finite input/output/runtime/process budget invalid",
        )?;
        let a = &m.resources[0];
        let c = &m.resources[1];
        require(
            a.cpu_affinity == c.cpu_affinity
                && a.memory_high_bytes == c.memory_high_bytes
                && a.memory_max_bytes == c.memory_max_bytes
                && a.swap_max_bytes == c.swap_max_bytes,
            "V4 sequential engines require one actual inherited resource policy",
        )?;
        for i in 0..2 {
            self.endpoint(i)?;
            if let PalsEngineV4::Pals(e) = &m.engines[i] {
                policy_arguments(e)?;
                require(
                    e.model_v2.arena_candidate(),
                    "diagnostic checkpoint/export cannot enter arena",
                )?;
                require(
                    e.base.pools.host_bytes == m.resources[i].memory_max_bytes,
                    "V4 host pool differs from inherited memory cap",
                )?;
                if let Some(n) = self.endpoints[i].native_model() {
                    require(
                        n.search.post_repair_recheck.is_none(),
                        "V4 typed policy cannot carry a legacy recheck selector",
                    )?;
                    require(
                        e.base.model.frozen_epoch == 1,
                        "V4 production model requires frozen epoch 1",
                    )?;
                    validate_actual_encoding(e, n)?;
                }
                if let Some(cuda) = self.endpoints[i].cuda_model() {
                    require(
                        cuda.cuda_record_pages.is_none(),
                        "V2 full-line model does not register legacy resident packing",
                    )?;
                    require(
                        m.resources[i].requested_gpu.is_some()
                            && e.base.pools.device_bytes
                                == m.resources[i].device_allocation_max_bytes
                            && cuda
                                .session_arena_bytes
                                .checked_mul(cuda.model.graphs.len() as u64)
                                .is_some_and(|n| n < e.base.pools.device_bytes),
                        "V4 CUDA device/arena reservation differs",
                    )?;
                } else {
                    require(
                        m.resources[i].requested_gpu.is_none() && e.base.pools.device_bytes == 0,
                        "CPU/mock V4 cannot silently inherit CUDA",
                    )?;
                }
            }
            if self.external_cpu_r_binding(i)?.is_some() {
                require(
                    cfg!(all(
                        target_os = "linux",
                        any(feature = "pals-collection-onnx", feature = "native-cuda")
                    )),
                    "external CPU_R needs actual native feature; no fallback",
                )?;
            }
        }
        let mut names = BTreeMap::new();
        for (a, name) in self.named_assets() {
            if let Some(prior) = names.insert(name, a) {
                require(prior == a, "V4 snapshot destination collision")?;
            }
        }
        Ok(())
    }
    pub fn lock(&self) -> Result<LockedPalsArenaLaunchV4, ArenaError> {
        self.validate()?;
        Ok(LockedPalsArenaLaunchV4 {
            input: self.clone(),
            sha256: crate::canonical_sha256(self)?,
            opening: self.opening(),
            endpoint_views: [self.endpoint(0)?, self.endpoint(1)?],
        })
    }
    pub fn runner_status_patch(&self) -> Result<PalsRunnerStatusPatchV3, ArenaError> {
        let patch = self
            .runner
            .dirty_patch
            .as_ref()
            .ok_or_else(|| invalid("runner patch absent"))?;
        match (patch.sha256.as_str(), patch.bytes) {
            (
                rz_experiments::FASTCHESS_CLOCK_PATCH_SHA256,
                rz_experiments::FASTCHESS_CLOCK_PATCH_BYTES,
            ) => Ok(PalsRunnerStatusPatchV3::LegacyClockOnly),
            (PALS_CLOCK_REAP_PATCH_SHA256, PALS_CLOCK_REAP_PATCH_BYTES) => {
                Ok(PalsRunnerStatusPatchV3::ClockAndReapStatus)
            }
            _ => Err(invalid("runner patch hash/bytes not registered")),
        }
    }
    fn require_reap_status_runner(&self) -> Result<(), ArenaError> {
        require(
            self.runner_status_patch()? == PalsRunnerStatusPatchV3::ClockAndReapStatus
                && self.runner.binary.sha256 == PALS_CLOCK_REAP_RUNNER_SHA256
                && self.runner.binary.bytes == PALS_CLOCK_REAP_RUNNER_BYTES,
            "new PALS execution/Core requires the registered clock+reap-status patch and actual runner binary; legacy evidence is preserved without promotion",
        )
    }
    #[cfg(any(target_os = "linux", test))]
    fn require_actual_native_epoch(&self) -> Result<(), ArenaError> {
        for (engine, recipe) in self
            .semantic_lock
            .manifest
            .engines
            .iter()
            .zip(&self.endpoints)
        {
            if recipe.native_model().is_some() {
                require(
                    matches!(engine, PalsEngineV4::Pals(e) if e.base.model.frozen_epoch == 1),
                    "new native PALS execution/Core requires deployment frozen epoch 1; historical epoch 0 locks remain unchanged",
                )?;
            }
        }
        Ok(())
    }
    fn opening(&self) -> OpeningSpec {
        OpeningSpec {
            id: "pals-standard-start".into(),
            initial: InitialPosition::Startpos,
            fen: None,
            moves: vec![],
            history: HistoryCompleteness::Complete,
            history_origin: "startpos-complete-trace".into(),
        }
    }
    fn named_assets(&self) -> Vec<(&ArtifactRef, String)> {
        let mut result = Vec::new();
        for engine in &self.semantic_lock.manifest.engines {
            if let PalsEngineV4::Pals(endpoint) = engine
                && let PalsCpuRSelectionV3::ExternalUci(helper) = &endpoint.base.cpu_r
            {
                result.push((
                    &helper.profile,
                    format!("pals-helper-profile-{}/profile.json", helper.profile.sha256),
                ));
                result.push((
                    &helper.binary,
                    format!("pals-helper-binary-{}/binary", helper.binary.sha256),
                ));
            }
        }
        for recipe in &self.endpoints {
            if let Some(n) = recipe.native_model() {
                let namespace = format!("pals-{}", n.export.sha256);
                result.push((&n.export, format!("{namespace}/{}", n.export_file)));
                result.extend(
                    n.graphs
                        .iter()
                        .map(|g| (&g.artifact, format!("{namespace}/{}", g.file))),
                );
            }
            if let Some(control) = recipe
                .cuda_model()
                .and_then(|cuda| cuda.cuda_control.as_ref())
            {
                result.push((
                    &control.inventory,
                    format!(
                        "pals-control-{}/inventory.v2.json",
                        control.inventory.sha256
                    ),
                ));
            }
            if let Some(pages) = recipe
                .cuda_model()
                .and_then(|cuda| cuda.cuda_record_pages.as_ref())
            {
                let namespace = format!("pals-packing-{}", pages.manifest.sha256);
                result.push((&pages.manifest, format!("{namespace}/device-packing.json")));
                result.push((&pages.graph, format!("{namespace}/device_public_pack.onnx")));
            }
        }
        result
    }
    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut result = vec![&self.runner.binary, &self.opening_artifact];
        if let Some(p) = &self.runner.dirty_patch {
            result.push(p);
        }
        for (engine, recipe) in self
            .semantic_lock
            .manifest
            .engines
            .iter()
            .zip(&self.endpoints)
        {
            match engine {
                PalsEngineV4::Pals(e) => {
                    let e = &e.base;
                    result.push(&e.binary);
                    if let PalsCpuRSelectionV3::ExternalUci(helper) = &e.cpu_r {
                        result.extend([&helper.profile, &helper.binary]);
                    }
                    if let PalsWeightIdentityV3::Untrained { artifact, .. }
                    | PalsWeightIdentityV3::Trained { artifact, .. } = &e.model.weights
                    {
                        result.push(artifact);
                    }
                }
                PalsEngineV4::OwnCpu(e) => result.push(&e.binary),
                PalsEngineV4::ReferenceUci(e) => {
                    result.push(&e.binary);
                    result.extend(&e.assets);
                }
            }
            match recipe {
                PalsEndpointLaunchV3::OnnxCpu(n) => {
                    result.extend([&n.export, &n.runtime]);
                    result.extend(n.graphs.iter().map(|g| &g.artifact));
                }
                PalsEndpointLaunchV3::OnnxCuda(cuda) => {
                    let n = &cuda.model;
                    result.extend([&n.export, &n.runtime, &cuda.cuda_bundle.manifest]);
                    result.extend(n.graphs.iter().map(|g| &g.artifact));
                    result.extend(cuda.cuda_bundle.files.iter().map(|f| &f.artifact));
                    if let Some(control) = &cuda.cuda_control {
                        result.push(&control.inventory);
                    }
                    if let Some(pages) = &cuda.cuda_record_pages {
                        result.extend([&pages.manifest, &pages.graph]);
                    }
                }
                PalsEndpointLaunchV3::ReferenceUci {
                    environment: Some(e),
                    ..
                } => result.push(&e.launcher),
                _ => {}
            }
        }
        result
    }
    fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        let mut by_path = BTreeMap::new();
        let mut total = 0u64;
        for a in self.declared_artifacts() {
            a.validate()?;
            require_public_artifact(a).map_err(|e| ManifestError::Integrity(e.to_string()))?;
            if let Some(prior) = by_path.insert(&a.path, a) {
                if prior != a {
                    return Err(ManifestError::Integrity(
                        "PALS path has inconsistent artifact identity".into(),
                    ));
                }
            } else {
                total = total
                    .checked_add(a.bytes)
                    .ok_or_else(|| ManifestError::Integrity("PALS input byte overflow".into()))?;
            }
        }
        Ok(total)
    }
    fn endpoint(&self, i: usize) -> Result<ExternalUciEndpointV2, ArenaError> {
        let helper = self.external_cpu_r_binding(i)?;
        let m = &self.semantic_lock.manifest;
        let role = if i == 0 {
            NativeEngineRole::Baseline
        } else {
            NativeEngineRole::Candidate
        };

        let (binary, family, version, uci_name, source, mut arguments, assets, environment) = match (
            &m.engines[i],
            &self.endpoints[i],
        ) {
            (PalsEngineV4::Pals(v4), PalsEndpointLaunchV3::LegalOrderMock(s)) => {
                let e = &v4.base;
                require(
                    e.model.backend == PalsModelBackendV3::DeterministicMock
                        && matches!(
                            e.model.weights,
                            PalsWeightIdentityV3::DeterministicMock { seed: 0 }
                        )
                        && e.model.architecture == "explicit-legal-order-role-mock-v1"
                        && e.model.precision == PalsPrecisionV3::Fp32
                        && e.model.max_batch_width == 1
                        && e.model.frozen_epoch == 0,
                    "explicit legal-order mock identity/seed/epoch differs",
                )?;
                validate_search(e, s, m.pilot.wall_time_max_ms)?;
                (
                    &e.binary,
                    "rovezero-pals",
                    "pals/0.1",
                    "RoveZero PALS explicit legal-order CPU mock + own CPU_R".to_string(),
                    Some(ExternalSourceV2 {
                        url: e.binary.source.clone(),
                        commit: e.source_commit.clone(),
                        license: e.binary.license.clone(),
                    }),
                    pals_search_arguments(s, "legal-order-mock"),
                    vec![],
                    None,
                )
            }
            (
                PalsEngineV4::Pals(e),
                recipe @ (PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)),
            ) => {
                let v4 = e;
                let e = &v4.base;
                let n = recipe.native_model().expect("native pattern selected");
                let cuda = recipe.cuda_model();
                require(
                    e.model.backend
                        == if cuda.is_some() {
                            PalsModelBackendV3::OrtCuda
                        } else {
                            PalsModelBackendV3::OrtCpu
                        }
                        && e.model.precision == PalsPrecisionV3::Fp32
                        && e.model.max_batch_width == 1
                        && matches!(e.model.frozen_epoch, 0 | 1),
                    "native metadata requires explicit provider, historical/deployment frozen epoch 0/1 and FP32 B1",
                )?;
                require(
                    e.model.architecture == "pals-width384-latent16-iterations2"
                        && e.model.input_schema == v4.model_v2.encoding_schema
                        && e.model.policy_head == "candidate-policy/1"
                        && e.model.value_head == "stm-wdl/1"
                        && e.model.implementation_sha256
                            == hex(rz_eval::pals_model::compiled_model_boundary_sha256()),
                    "native model topology/input/head/compiled model boundary declaration differs from supported recipe",
                )?;
                require(
                    m.resources[i].cpu_threads == 2,
                    "native CPU model uses two registered ORT intra-op threads",
                )?;
                require(
                    matches!(
                        e.model.weights,
                        PalsWeightIdentityV3::Untrained { .. }
                            | PalsWeightIdentityV3::Trained { .. }
                    ),
                    "native P/C requires actual checkpoint identity",
                )?;
                require(
                    hash(&n.encoding_semantic_sha256)
                        && hash(&n.adapter_source_sha256)
                        && name(&n.export_file)
                        && n.export_file.ends_with(".json")
                        && n.export.bytes <= 64 * 1024
                        && matches!(n.graphs.len(), 2 | 3),
                    "P/C export/adapter identities invalid",
                )?;
                let roles: BTreeSet<_> = n.graphs.iter().map(|g| g.role.as_str()).collect();
                require(
                    (roles == BTreeSet::from(["public", "proposer", "critic"])
                        || roles == BTreeSet::from(["public", "shared_pc"]))
                        && n.graphs.iter().all(|g| {
                            name(&g.file) && g.file.ends_with(".onnx") && g.file != n.export_file
                        }),
                    "only exact public/P/C graphs may execute; V is excluded",
                )?;
                require(
                    n.graphs
                        .iter()
                        .map(|g| &g.file)
                        .collect::<BTreeSet<_>>()
                        .len()
                        == n.graphs.len(),
                    "duplicate graph filename",
                )?;
                validate_search(e, &n.search, m.pilot.wall_time_max_ms)?;
                let mut args = pals_search_arguments(&n.search, "onnx");
                args.extend([
                    format!(
                        "--pals-provider={}",
                        if cuda.is_some() { "cuda" } else { "cpu" }
                    ),
                    "--pals-export-manifest={{asset:0}}".into(),
                    format!("--pals-export-sha256={}", n.export.sha256),
                    "--pals-runtime-path={{asset:1}}".into(),
                    format!("--pals-runtime-sha256={}", n.runtime.sha256),
                ]);
                let mut assets = vec![n.export.clone(), n.runtime.clone()];
                assets.extend(n.graphs.iter().map(|g| g.artifact.clone()));
                if let Some(cuda) = cuda {
                    cuda.cuda_bundle.validate()?;
                    if let Some(probe_ms) = cuda.startup_probe_timeout_ms {
                        require(
                            (1..=MAX_STARTUP_PROBE_TIMEOUT_MS).contains(&probe_ms)
                                && probe_ms <= m.pilot.handshake_max_ms,
                            "explicit CUDA startup probe requires 1..180000ms within the cold-model readiness window",
                        )?;
                    }
                    require(
                        cuda.device_id == 0
                            && cuda.session_arena_bytes == 2 * 1024 * 1024 * 1024
                            && roles == BTreeSet::from(["public", "shared_pc"]),
                        "first 6GiB CUDA recipe requires explicit device 0, per-session 2GiB and two shared-P/C graphs with room for tensor reservations",
                    )?;
                    require(
                        cuda.cuda_bundle.file(
                            rz_experiments::CudaBundleFileRoleV1::Core,
                            "libonnxruntime.so.1.22.0",
                        )? == &n.runtime,
                        "CUDA bundle core identity differs from declared runtime",
                    )?;
                    let index = assets.len();
                    assets.push(cuda.cuda_bundle.manifest.clone());
                    assets.extend(
                        cuda.cuda_bundle
                            .files
                            .iter()
                            .filter(|f| f.artifact != n.runtime)
                            .map(|f| f.artifact.clone()),
                    );
                    args.extend([
                        format!("--pals-cuda-bundle={{{{asset:{index}}}}}"),
                        format!(
                            "--pals-cuda-bundle-sha256={}",
                            cuda.cuda_bundle.manifest.sha256
                        ),
                        format!("--pals-cuda-device={}", cuda.device_id),
                        format!(
                            "--pals-cuda-session-arena-bytes={}",
                            cuda.session_arena_bytes
                        ),
                        "--pals-device-public-memory=false".into(),
                    ]);
                    if let Some(control) = &cuda.cuda_control {
                        control.inventory.validate()?;
                        require_public_artifact(&control.inventory)?;
                        require(
                            (1..=MAX_CONTROL_INVENTORY_BYTES).contains(&control.inventory.bytes),
                            "CUDA control inventory must have a finite one-MiB input bound",
                        )?;
                        let index = assets.len();
                        assets.push(control.inventory.clone());
                        args.extend([
                            "--pals-cuda-control-mode=inventory-v2".into(),
                            format!("--pals-cuda-control-inventory={{{{asset:{index}}}}}"),
                            format!(
                                "--pals-cuda-control-inventory-sha256={}",
                                control.inventory.sha256
                            ),
                        ]);
                        if let Some(loading) = &control.loading_profile {
                            loading.validate()?;
                            args.extend([
                                format!(
                                    "--pals-cuda-loading-profile={}",
                                    loading.profile.native().identifier()
                                ),
                                format!(
                                    "--pals-cuda-loading-profile-sha256={}",
                                    loading.canonical_sha256
                                ),
                            ]);
                        }
                    }
                    if let Some(probe_ms) = cuda.startup_probe_timeout_ms {
                        args.push(format!("--pals-startup-probe-timeout-ms={probe_ms}"));
                    }
                    if let Some(pages) = &cuda.cuda_record_pages {
                        pages.validate(cuda)?;
                        require(
                            helper.is_none(),
                            "resident CUDA cannot launch an external helper",
                        )?;
                        let manifest_index = assets.len();
                        assets.extend([pages.manifest.clone(), pages.graph.clone()]);
                        args.extend([
                            "--pals-cuda-record-pages=registered-packing-v1".into(),
                            format!("--pals-packing-manifest={{{{asset:{manifest_index}}}}}"),
                            format!("--pals-packing-manifest-sha256={}", pages.manifest.sha256),
                            format!("--pals-packing-graph={{{{asset:{}}}}}", manifest_index + 1),
                            format!("--pals-packing-graph-sha256={}", pages.graph.sha256),
                            format!(
                                "--pals-cuda-record-pages-resources={}",
                                pages.resources_json()?
                            ),
                        ]);
                    }
                }
                if let Some((helper, _)) = helper {
                    let index = assets.len();
                    assets.extend([helper.profile.clone(), helper.binary.clone()]);
                    args.extend([
                        "--pals-cpu-checker=external-uci".into(),
                        format!("--pals-cpu-profile={{{{asset:{index}}}}}"),
                        format!("--pals-cpu-profile-sha256={}", helper.profile.sha256),
                    ]);
                }
                (
                    &e.binary,
                    "rovezero-pals",
                    "pals/0.1",
                    format!(
                        "RoveZero PALS P/C ONNX {} + {} CPU_R",
                        if cuda.is_some() { "CUDA" } else { "CPU" },
                        if helper.is_some() {
                            "external UCI"
                        } else {
                            "own"
                        }
                    ),
                    Some(ExternalSourceV2 {
                        url: e.binary.source.clone(),
                        commit: e.source_commit.clone(),
                        license: e.binary.license.clone(),
                    }),
                    args,
                    assets,
                    None,
                )
            }
            (
                PalsEngineV4::OwnCpu(e),
                PalsEndpointLaunchV3::OwnCpu {
                    max_depth,
                    max_nodes,
                    tt_entries,
                },
            ) => {
                require(
                    *max_depth > 0
                        && *max_depth <= 64
                        && u32::from(*max_depth) <= e.cpu.max_depth
                        && *max_nodes > 0
                        && *max_nodes <= e.cpu.max_nodes_per_task
                        && (1..=1_048_576).contains(tt_entries),
                    "own CPU executable limits exceed declaration",
                )?;
                validate_cpu_profile(&e.cpu, m.pilot.wall_time_max_ms)?;
                let config = rz_search::cpu::CpuConfig {
                    max_depth: *max_depth,
                    tt_entries: *tt_entries as usize,
                    ..Default::default()
                };
                require(
                    config
                        .tt_allocation_bytes()
                        .map_err(|e| invalid(e.to_string()))?
                        <= e.cpu.max_tt_bytes,
                    "own CPU TT slot allocation exceeds declared byte limit",
                )?;
                require(
                    e.requested_options.is_empty(),
                    "own CPU closed recipe accepts no unverified UCI overrides",
                )?;
                let args = vec![
                    "--search=cpu".into(),
                    format!("--cpu-max-depth={max_depth}"),
                    format!("--cpu-max-nodes={max_nodes}"),
                    format!("--cpu-tt-entries={tt_entries}"),
                ];
                (
                    &e.binary,
                    "rovezero-own-cpu",
                    "rz-cpu-pvs/0.1",
                    "RoveZero own Rust CPU_R".to_string(),
                    Some(ExternalSourceV2 {
                        url: e.binary.source.clone(),
                        commit: e.source_commit.clone(),
                        license: e.binary.license.clone(),
                    }),
                    args,
                    vec![],
                    None,
                )
            }
            (
                PalsEngineV4::ReferenceUci(e),
                PalsEndpointLaunchV3::ReferenceUci {
                    expected_uci_name,
                    arguments,
                    environment,
                },
            ) => {
                require(
                    !expected_uci_name.is_empty()
                        && expected_uci_name.len() <= 256
                        && !expected_uci_name.chars().any(char::is_control),
                    "reference UCI identity invalid",
                )?;
                (
                    &e.binary,
                    e.family.as_str(),
                    e.version.as_str(),
                    expected_uci_name.clone(),
                    Some(ExternalSourceV2 {
                        url: e.source_url.clone(),
                        commit: e.source_commit.clone(),
                        license: e.binary.license.clone(),
                    }),
                    arguments.clone(),
                    e.assets.clone(),
                    environment.clone(),
                )
            }
            _ => return Err(invalid("semantic endpoint and executable recipe differ")),
        };
        if let PalsEngineV4::Pals(e) = &m.engines[i] {
            arguments.extend(policy_arguments(e)?);
        } else if matches!(self.endpoints[i], PalsEndpointLaunchV3::OwnCpu { .. }) {
            arguments.push("--pals-arena-wire=v4".into());
        }
        require(
            arguments.len() <= 48
                && arguments
                    .iter()
                    .all(|a| !a.is_empty() && a.len() <= 4096 && !a.chars().any(char::is_control)),
            "executable argv budget/injection violation",
        )?;
        Ok(ExternalUciEndpointV2 {
            id: m.engines[i].id().into(),
            role,
            family: family.into(),
            version: version.into(),
            expected_uci_name: uci_name,
            binary: binary.clone(),
            source,
            arguments,
            assets,
            requested_options: requested_options(&m.engines[i]).clone(),
            environment,
            handshake_timeout_ms: m.pilot.handshake_max_ms,
        })
    }
}
fn validate_search(
    e: &rz_experiments::PalsEndpointV3,
    s: &PalsSearchLaunchV3,
    wall: u64,
) -> Result<(), ArenaError> {
    require(
        s.post_repair_recheck.is_none(),
        "V4 executable policy is separately typed",
    )?;
    let (max_depth, max_nodes_per_task) = match &e.cpu_r {
        PalsCpuRSelectionV3::Own => {
            validate_cpu_profile(&e.cpu, wall)?;
            require(
                rz_search::cpu::CpuConfig::default()
                    .tt_allocation_bytes()
                    .map_err(|e| invalid(e.to_string()))?
                    <= e.cpu.max_tt_bytes,
                "PALS own CPU default TT slot allocation exceeds declared byte limit",
            )?;
            (e.cpu.max_depth, e.cpu.max_nodes_per_task)
        }
        PalsCpuRSelectionV3::ExternalUci(helper) => {
            // CPU_T/legacy own declarations remain in the lock; they are not
            // runtime authority for the explicitly selected foreign CPU_R.
            (helper.policy.max_depth, helper.policy.max_nodes_per_task)
        }
    };
    require(
        (1..=1_000_000).contains(&s.max_rounds)
            && s.max_cpu_nodes > 0
            && s.cpu_depth > 0
            && s.cpu_depth <= 16
            && u32::from(s.cpu_depth) <= max_depth
            && (257..=65_536).contains(&s.max_situations)
            && max_nodes_per_task
                >= if e.cpu_r.is_own() {
                    4096
                } else {
                    rz_search::pals::engine::PalsConfig::default()
                        .cpu_nodes_per_task
                        .min(s.max_cpu_nodes)
                },
        "PALS executable work/profile bounds invalid",
    )?;
    let p = &e.pools;
    let n = s.max_situations;
    require(
        p.states >= n
            && p.situations >= n
            && p.line_chunks >= n * 16
            && p.observations >= n * 16
            && p.tasks >= n * 8
            && p.role_states >= 1
            && p.memory_pages >= 1
            && p.queue_requests >= 1,
        "declared pool bounds do not cover the closed PALS executable recipe",
    )?;
    require(
        e.runtime.semantic_id == "single-owner/1" && e.runtime.options.is_empty(),
        "unsupported PALS search/runtime semantic option override",
    )?;
    require(
        e.requested_options.is_empty(),
        "PALS fixed executable recipe accepts no unverified UCI overrides",
    )
}
impl LockedPalsArenaLaunchV4 {
    pub fn input(&self) -> &PalsArenaLaunchV4 {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn to_json(&self) -> Result<String, ArenaError> {
        serde_json::to_string_pretty(&serde_json::json!({"domain":PALS_ARENA_V4_DOMAIN,"sha256":self.sha256,"input":self.input})).map_err(|e|invalid(e.to_string()))
    }
    pub fn from_json(text: &str) -> Result<Self, ArenaError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Envelope {
            domain: String,
            sha256: String,
            input: PalsArenaLaunchV4,
        }
        require(
            text.len() <= MAX_JSON_BYTES,
            "locked launch JSON budget exceeded",
        )?;
        let envelope: Envelope = crate::decode_json(text)?;
        let lock = envelope.input.lock()?;
        require(
            envelope.domain == PALS_ARENA_V4_DOMAIN && envelope.sha256 == lock.sha256,
            "arena lock digest/domain differs",
        )?;
        Ok(lock)
    }
    fn native(&self, role: NativeEngineRole) -> Result<&PalsOnnxLaunchV3, ArenaError> {
        self.input.endpoints[role_index(role)]
            .native_model()
            .ok_or_else(|| invalid("endpoint has no native P/C session"))
    }
    fn validate_export(&self, role: NativeEngineRole, bytes: &[u8]) -> Result<(), ArenaError> {
        require(bytes.len() <= 128 * 1024, "P/C export JSON exceeds bound")?;
        let n = self.native(role)?;
        require(digest(bytes) == n.export.sha256, "P/C export bytes differ")?;
        let value = json(bytes)?;
        let PalsEngineV4::Pals(e) = &self.input.semantic_lock.manifest.engines[role_index(role)]
        else {
            return Err(invalid("native recipe has no PALS identity"));
        };
        let v4 = e;
        let e = &v4.base;
        require(
            v4.model_v2.model_semantics_sha256
                == hex(config(v4).profile.registered_semantics_sha256())
                && e.model.implementation_sha256
                    == hex(rz_eval::pals_model::compiled_model_boundary_sha256()),
            "V4 model semantic/source registration differs from compiled boundary",
        )?;
        let (weights, trained) = match &e.model.weights {
            PalsWeightIdentityV3::Untrained { artifact, .. } => (artifact, false),
            PalsWeightIdentityV3::Trained { artifact, .. } => (artifact, true),
            _ => return Err(invalid("no native checkpoint identity")),
        };
        let shared = n.graphs.iter().any(|g| g.role == "shared_pc");
        let layout_valid = if shared {
            value["schema"] == "rovezero.pals-model.v2"
                && value["layout"] == "shared_pc_if"
                && value["layout_revision"] == 1
                && value["model_semantics"] == v4.model_v2.model_semantics
                && value["role_batching"] == "one_scalar_role_per_physical_batch"
        } else {
            value["schema"] == "rovezero.pals-model.v1"
                && (value["layout"].is_null() || value["layout"] == "separate_pc")
        };
        require(
            layout_valid
                && value["schema"] == v4.model_v2.model_schema
                && value["config"]
                    == serde_json::to_value(config(v4)).map_err(|e| invalid(e.to_string()))?
                && (config(v4).profile.is_legacy()
                    || (value["model_profile"] == config(v4).profile.as_str()
                        && value["encoding_schema"] == v4.model_v2.encoding_schema))
                && value["artifact_purpose"] != "diagnostic"
                && value["diagnostic"] != true
                && value["checkpoint_domain"] != PALS_V4_NONZERO_CHECKPOINT_DOMAIN
                && value["checkpoint_sha256"] == weights.sha256
                && value["trained"] == trained
                && value["validator_present"] == false
                && value["roles"] == serde_json::json!(["proposer", "critic"]),
            "P/C export model/checkpoint/V-free identity differs",
        )?;
        let graphs = value["graphs"]
            .as_array()
            .ok_or_else(|| invalid("missing P/C graph list"))?;
        require(graphs.len() == n.graphs.len(), "export graph count differs")?;
        for declared in &n.graphs {
            require(
                graphs
                    .iter()
                    .filter(|g| {
                        g["role"] == declared.role
                            && g["file"] == declared.file
                            && g["sha256"] == declared.artifact.sha256
                    })
                    .count()
                    == 1,
                "export graph identity differs from pinned input",
            )?;
        }
        Ok(())
    }
    fn validate_cuda_bundle(&self, role: NativeEngineRole, bytes: &[u8]) -> Result<(), ArenaError> {
        let cuda = self.input.endpoints[role_index(role)]
            .cuda_model()
            .ok_or_else(|| invalid("endpoint has no CUDA bundle"))?;
        require(
            bytes.len() <= 64 * 1024 && digest(bytes) == cuda.cuda_bundle.manifest.sha256,
            "CUDA descriptor byte/hash differs",
        )?;
        let parsed = rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(
            std::str::from_utf8(bytes).map_err(|_| invalid("CUDA descriptor not UTF-8"))?,
        )
        .map_err(|e| invalid(format!("CUDA descriptor rejected: {:?}", e.kind)))?;
        let actual = parsed
            .digest()
            .map_err(|e| invalid(format!("CUDA descriptor digest rejected: {:?}", e.kind)))?;
        require(
            actual
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
                == cuda.cuda_bundle.canonical_sha256,
            "CUDA descriptor canonical digest differs from pinned nineteen-file identity",
        )?;
        Ok(())
    }
    fn validate_control_inventory(
        &self,
        role: NativeEngineRole,
        bytes: &[u8],
    ) -> Result<(), ArenaError> {
        let cuda = self.input.endpoints[role_index(role)]
            .cuda_model()
            .ok_or_else(|| invalid("endpoint has no CUDA recipe"))?;
        let control = cuda
            .cuda_control
            .as_ref()
            .ok_or_else(|| invalid("endpoint has no registered control inventory"))?;
        require(
            bytes.len() as u64 == control.inventory.bytes
                && bytes.len() as u64 <= MAX_CONTROL_INVENTORY_BYTES
                && digest(bytes) == control.inventory.sha256,
            "control inventory byte/hash differs",
        )?;
        let value = json(bytes)?;
        require(
            value["schema"] == "rovezero.pals-static-control-inventory.v2"
                && value["scope"] == "read_only_serialized_graph_metadata_no_runtime"
                && value["manifest_sha256"] == cuda.model.export.sha256
                && value["actual_provider_placement"] == "not_observed"
                && value["cpu_allowlist"] == "not_created"
                && (1..=5000).contains(&count(&value, "total_nodes")?),
            "registered control inventory metadata differs",
        )?;
        let graphs = value["graphs"]
            .as_array()
            .ok_or_else(|| invalid("control inventory graph list missing"))?;
        require(
            graphs.len() == 2
                && cuda.model.graphs.iter().all(|declared| {
                    graphs
                        .iter()
                        .filter(|g| {
                            g["role"] == declared.role && g["sha256"] == declared.artifact.sha256
                        })
                        .count()
                        == 1
                }),
            "control inventory is not the pinned public/shared-P/C graph pair",
        )
    }
    fn cuda_profile_argument(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Option<OsString>, ArenaError> {
        if self.input.endpoints[role_index(role)]
            .cuda_model()
            .and_then(|cuda| cuda.cuda_control.as_ref())
            .is_none()
        {
            return Ok(None);
        }
        let path = runtime_root
            .to_str()
            .ok_or_else(|| invalid("CUDA profile parent is not UTF-8"))?;
        require(
            runtime_root.is_absolute() && path.len() <= 4096 && !path.chars().any(char::is_control),
            "CUDA profile parent must be a bounded owned absolute runtime root",
        )?;
        // Fastchess reuses argv between games. The UCI child derives an
        // exclusive PID+nonce directory here for each preflight/game process.
        Ok(Some(format!("--pals-cuda-profile-parent={path}").into()))
    }
}
impl crate::native_launch::sealed::Sealed for LockedPalsArenaLaunchV4 {}
impl NativeLaunchDeclaration for LockedPalsArenaLaunchV4 {
    fn argument_limit(&self) -> usize {
        48
    }
    fn input_sha256(&self) -> &str {
        &self.sha256
    }
    fn view(&self) -> NativePairView<'_> {
        let p = &self.input.semantic_lock.manifest.pilot;
        let ids = &self.input.semantic_lock.manifest.engines;
        let white_order = p.white_order.each_ref().map(|id| {
            if id == ids[0].id() {
                NativeEngineRole::Baseline
            } else {
                NativeEngineRole::Candidate
            }
        });
        NativePairView {
            pair_id: &self.input.semantic_lock.manifest.pair_id,
            white_order,
            opening: &self.opening,
            opening_artifact: &self.input.opening_artifact,
            runner: &self.input.runner,
            clock: NativePairClock::Game(NativeGameClockV3 {
                base_ms: p.base_ms,
                increment_ms: p.increment_ms,
            }),
            max_plies: p.max_plies,
            timeouts: NativeTimeoutsV1 {
                startup_ms: p.handshake_max_ms,
                handshake_ms: p.handshake_max_ms,
                runtime_ms: p.wall_time_max_ms,
                drain_ms: p.cleanup_max_ms,
                shutdown_ms: p.cleanup_max_ms / 2,
            },
            budget: self.input.budget,
        }
    }
    fn declared_inputs(&self) -> Vec<&ArtifactRef> {
        self.input.declared_artifacts()
    }
    fn unique_bytes(&self) -> Result<u64, ManifestError> {
        self.input.unique_input_bytes()
    }
    fn engine_view(&self, role: NativeEngineRole) -> Result<NativeEngineView<'_>, ManifestError> {
        let e = &self.endpoint_views[role_index(role)];
        Ok(NativeEngineView {
            engine_id: &e.id,
            artifacts: &[],
            cuda_bundle: self.input.endpoints[role_index(role)]
                .cuda_model()
                .map(|cuda| &cuda.cuda_bundle),
            search: None,
            batch_experiment: None,
            external: Some(e),
            environment: e.environment.as_ref(),
        })
    }
    fn provider_name(&self) -> &'static str {
        "PALS-V4"
    }
    fn pair_metadata_cap(&self) -> u64 {
        V4_METADATA_CAP
    }
    fn amend_execution_limitations(&self, limitations: &mut Vec<String>) {
        limitations.push("V4 frozen P/C paired pilot; model/profile, resolver, checker, recheck, stack and CUDA flags are separately locked and checked against actual producer getters; no training or Elo promotion".into());
        limitations.push("Shared V3 physical resource DTOs preserve their exact meaning; the V4 semantic lock and raw domains are never coerced to historical IDs".into());
    }
    fn seed(&self) -> u64 {
        self.input.semantic_lock.manifest.pilot.seed
    }
    fn validate_execution(&self) -> Result<(), ArenaError> {
        self.input.validate()?;
        crate::native_launch::pair_stream_cap(
            self.input.budget.max_output_bytes,
            self.pair_metadata_cap(),
        )?;
        Ok(())
    }
    fn advise_drop_input_cache(&self) -> bool {
        true
    }
    fn snapshot_relative_path(&self, a: &ArtifactRef) -> Option<String> {
        self.input
            .named_assets()
            .into_iter()
            .find_map(|(artifact, path)| (artifact == a).then_some(path))
    }
    fn runtime_arguments(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        if matches!(
            &self.input.endpoints[role_index(role)],
            PalsEndpointLaunchV3::ReferenceUci { .. }
        ) {
            return Ok(vec![]);
        }
        let root = runtime_root
            .to_str()
            .ok_or_else(|| invalid("native output root is not UTF-8"))?;
        require(
            runtime_root.is_absolute(),
            "native output root must be absolute",
        )?;
        if !matches!(
            &self.input.endpoints[role_index(role)],
            PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
        ) {
            return Ok(vec![
                format!("--search-work-output-root={root}").into(),
                format!("--search-work-launch-sha256={}", self.sha256).into(),
                format!(
                    "--search-work-endpoint-id={}",
                    self.endpoint_views[role_index(role)].id
                )
                .into(),
            ]);
        }
        let mut arguments = vec![
            format!("--pals-output-root={root}").into(),
            format!("--pals-launch-sha256={}", self.sha256).into(),
            format!(
                "--pals-endpoint-id={}",
                self.endpoint_views[role_index(role)].id
            )
            .into(),
            pals_shared_runtime_argument()?,
        ];
        if let Some(profile) = self.cuda_profile_argument(role, runtime_root)? {
            arguments.push(profile);
        }
        Ok(arguments)
    }
    fn uses_separate_preflight_runtime_root(
        &self,
        role: NativeEngineRole,
    ) -> Result<bool, ArenaError> {
        Ok(self
            .input
            .external_cpu_r_binding(role_index(role))?
            .is_some())
    }
    fn preflight_arguments(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        if !matches!(
            &self.input.endpoints[role_index(role)],
            PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
        ) {
            return Ok(vec![]);
        }
        require(
            runtime_root.is_absolute(),
            "preflight cache root must be absolute",
        )?;
        let isolated = self
            .input
            .external_cpu_r_binding(role_index(role))?
            .is_some();
        // The Native capability owner creates and pins this exact root. Do not
        // derive an unowned sibling from a game path at the argv boundary.
        if isolated {
            let expected = match role {
                NativeEngineRole::Baseline => "external-baseline-preflight-runtime",
                NativeEngineRole::Candidate => "external-candidate-preflight-runtime",
            };
            require(
                runtime_root
                    .file_name()
                    .is_some_and(|name| name == expected),
                "external CPU_R preflight requires the owner-selected role root",
            )?;
        }
        let evidence_root = runtime_root;
        let mut arguments = vec![pals_shared_runtime_argument()?];
        if let Some(profile) = self.cuda_profile_argument(role, evidence_root)? {
            arguments.push(profile);
        }
        if isolated {
            let root = evidence_root
                .to_str()
                .ok_or_else(|| invalid("external CPU_R preflight evidence root is not UTF-8"))?;
            arguments.extend([
                format!("--pals-output-root={root}").into(),
                format!("--pals-launch-sha256={}", self.sha256).into(),
                format!(
                    "--pals-endpoint-id={}",
                    self.endpoint_views[role_index(role)].id
                )
                .into(),
            ]);
        }
        Ok(arguments)
    }
}
impl NativeProviderDeclaration for LockedPalsArenaLaunchV4 {
    type Audit = PalsNativeSessionAuditV3;
    fn scope(&self) -> &'static str {
        "pals_v4_pc_native_or_explicit_mock_cpu_uci_rules_clocks_process_pilot_no_elo"
    }
    fn receipt_filename(&self) -> &'static str {
        "pals-arena-pair-receipt.v4.json"
    }
    fn receipt_version(&self) -> u32 {
        4
    }
    fn contract_revision(&self) -> &str {
        &self.input.semantic_lock.manifest.contract_revision
    }
    #[cfg(target_os = "linux")]
    fn validate_runtime_admission(&self) -> Result<(), ArenaError> {
        self.input.require_reap_status_runner()?;
        self.input.require_actual_native_epoch()?;
        verify_pals_inherited_resources_v4(self).map(|_| ())
    }
    #[cfg(target_os = "linux")]
    fn validate_external_preflight(
        &self,
        role: NativeEngineRole,
        preflight: &crate::ExternalUciPreflight,
        runtime_directory: &cap_std::fs::Dir,
        inputs: &crate::native_runner::NativeProviderInputContext<'_>,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        Ok(
            audit_pals_helper_preflight(self, role, preflight, runtime_directory, inputs)?
                .artifacts,
        )
    }
    #[cfg(target_os = "linux")]
    fn validate_records_with_context(
        &self,
        role: NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
        inputs: &crate::native_runner::NativeProviderInputContext<'_>,
        supervisor_stdout: &[u8],
    ) -> Result<(Self::Audit, u32), ArenaError> {
        if self
            .input
            .external_cpu_r_binding(role_index(role))?
            .is_none()
        {
            return self.validate_records(role, startup, termination, session);
        }
        #[cfg(not(any(feature = "pals-collection-onnx", feature = "native-cuda")))]
        {
            let _ = (inputs, supervisor_stdout);
            Err(invalid(
                "unsupported external CPU_R game validation: native PALS feature is required; no fallback",
            ))
        }
        #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
        {
            let loaded = load_owner_helper_profile(self, role, inputs)?;
            // These role/game IDs originate in the pinned supervisor trace,
            // before any native JSON supplies a purported PID.
            let mapping = validate_pals_game_process_trace_context(
                self,
                supervisor_stdout,
                &[BTreeSet::new(), BTreeSet::new()],
            )?;
            let expected = mapping[role_index(role)]
                .iter()
                .copied()
                .find(|pid| session == format!("native-process-{pid}"))
                .ok_or_else(|| {
                    invalid("native session absent from supervised role/game PID mapping")
                })?;
            let helper = validate_pals_external_cpu_r_session_context(
                self,
                role,
                &loaded,
                startup,
                termination,
                expected,
                PalsExternalCpuRSessionPurposeV3::Game,
            )?;
            validate_pals_native_records_inner(
                self,
                role,
                startup,
                termination,
                session,
                Some(helper),
                true,
            )
        }
    }
    #[cfg(target_os = "linux")]
    fn validate_provider_session_set(
        &self,
        sessions: &[Self::Audit],
        preflight: &[crate::ExternalUciPreflight],
        runtime_root: &cap_std::fs::Dir,
        inputs: &crate::native_runner::NativeProviderInputContext<'_>,
        supervisor_stdout: &[u8],
        artifacts: &[ArtifactRef],
    ) -> Result<(), ArenaError> {
        let mut has_external = false;
        for index in 0..2 {
            has_external |= self.input.external_cpu_r_binding(index)?.is_some();
        }
        if !has_external {
            return Ok(());
        }
        use cap_fs_ext::DirExt;
        let mapping = validate_pals_game_process_trace_context(
            self,
            supervisor_stdout,
            &[BTreeSet::new(), BTreeSet::new()],
        )?;
        let mut leaders: BTreeSet<_> = mapping.iter().flatten().copied().collect();
        for receipt in preflight {
            for process in [&receipt.identification_process, &receipt.readiness_process] {
                require(
                    leaders.insert(process.pid),
                    "preflight/game supervisor PID reused",
                )?;
            }
        }
        let mut audited = sessions.to_vec();
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            let index = role_index(role);
            if self.input.external_cpu_r_binding(index)?.is_none() {
                continue;
            }
            let mut matching = preflight
                .iter()
                .filter(|p| p.engine_id == self.endpoint_views[index].id);
            let receipt = matching
                .next()
                .ok_or_else(|| invalid("external helper preflight receipt absent"))?;
            require(
                matching.next().is_none(),
                "external helper preflight receipt duplicated",
            )?;
            let name = if index == 0 {
                "external-baseline-preflight-runtime"
            } else {
                "external-candidate-preflight-runtime"
            };
            let root = runtime_root
                .open_dir_nofollow(name)
                .map_err(|_| invalid("owner helper preflight root missing"))?;
            let evidence = audit_pals_helper_preflight(self, role, receipt, &root, inputs)?;
            for (relative, bytes) in &evidence.artifacts {
                let suffix = format!("/{name}/{relative}");
                require(
                    artifacts
                        .iter()
                        .filter(|a| {
                            a.path.ends_with(&suffix)
                                && a.sha256 == digest(bytes)
                                && a.bytes == bytes.len() as u64
                        })
                        .count()
                        == 1,
                    "helper preflight bytes differ from retained original evidence",
                )?;
            }
            audited.extend(evidence.sessions);
            let selected: Vec<_> = sessions
                .iter()
                .filter(|s| s.endpoint_id == self.endpoint_views[index].id)
                .collect();
            require(
                selected.len() == 2
                    && selected.iter().all(|s| {
                        mapping[index].contains(&s.process_id)
                            && s.external_cpu_r_session.as_ref().is_some_and(|h| {
                                h.purpose == PalsExternalCpuRSessionPurposeV3::Game
                            })
                    }),
                "two mapped external helper game sessions required",
            )?;
        }
        validate_pals_helper_identity_set(&audited, &leaders)
    }
    fn expected_provider_sessions(&self) -> usize {
        2 * self
            .input
            .endpoints
            .iter()
            .filter(|e| e.native_model().is_some())
            .count()
    }
    fn uses_provider_records(&self, role: NativeEngineRole) -> Result<bool, ArenaError> {
        Ok(matches!(
            &self.input.endpoints[role_index(role)],
            PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
        ))
    }
    #[cfg(target_os = "linux")]
    fn external_game_process_ids(
        &self,
        role: NativeEngineRole,
        external_ids: &[u32],
        root_output: &cap_std::fs::Dir,
    ) -> Result<Vec<u32>, ArenaError> {
        require(
            !self.uses_provider_records(role)?,
            "native role has its own physical record audit",
        )?;
        let own = collect_process_work(self, role, root_output)?;
        let selected: Vec<u32> = if own.is_empty() {
            let other = if role_index(role) == 0 {
                NativeEngineRole::Candidate
            } else {
                NativeEngineRole::Baseline
            };
            let other_records = if self.uses_provider_records(other)? {
                vec![]
            } else {
                collect_process_work(self, other, root_output)?
            };
            // Native counterpart IDs have already been excluded by the generic
            // native auditor. A CPU/mock counterpart remains in external_ids.
            external_ids
                .iter()
                .copied()
                .filter(|pid| !other_records.iter().any(|w| w.process_id == *pid))
                .collect()
        } else {
            require(
                own.iter().all(|w| external_ids.contains(&w.process_id)),
                "own work PID missing from supervised normal exit trace",
            )?;
            external_ids
                .iter()
                .copied()
                .filter(|pid| own.iter().any(|w| w.process_id == *pid))
                .collect()
        };
        require(
            selected.len() == 2 && selected[0] != selected[1],
            "two role-specific external exits required",
        )?;
        Ok(selected)
    }
    fn provider_export_artifact(
        &self,
        role: NativeEngineRole,
    ) -> Result<Option<&ArtifactRef>, ArenaError> {
        if self.uses_provider_records(role)? {
            Ok(Some(&self.native(role)?.export))
        } else {
            Ok(None)
        }
    }
    fn startup_filename(&self) -> &'static str {
        "pals-native-startup.v4.json"
    }
    fn termination_filename(&self) -> &'static str {
        "pals-native-termination.v4.json"
    }
    fn claim_policy(&self) -> rz_experiments::ClaimPolicy {
        rz_experiments::ClaimPolicy::AutomaticAcceptance
    }
    fn validate_clock_trace(
        &self,
        stdout: &[u8],
        pgn: &crate::PairPgnAudit,
    ) -> Result<Option<crate::NativePilotClockAudit>, ArenaError> {
        let p = &self.input.semantic_lock.manifest.pilot;
        crate::native_pilot::validate_pilot_clock_trace(
            stdout,
            pgn,
            &self.opening,
            NativeGameClockV3 {
                base_ms: p.base_ms,
                increment_ms: p.increment_ms,
            },
        )
        .map(Some)
    }
    fn audit_failure_trace(
        &self,
        stdout: &[u8],
        pair: &crate::PairSpec,
    ) -> Result<Option<crate::NativePilotFailureAudit>, ArenaError> {
        crate::native_pilot::audit_pilot_startup_failure_trace(stdout, pair)
    }
    #[cfg(target_os = "linux")]
    fn validate_records(
        &self,
        role: NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
    ) -> Result<(Self::Audit, u32), ArenaError> {
        validate_pals_native_records_inner(self, role, startup, termination, session, None, true)
    }
    #[cfg(target_os = "linux")]
    fn validate_conversion_manifest(
        &self,
        role: NativeEngineRole,
        startup: &[u8],
        manifest: &[u8],
        batch: bool,
    ) -> Result<(), ArenaError> {
        require(
            !batch,
            "PALS native B1 recipe does not authorize experimental batch",
        )?;
        self.validate_export(role, manifest)?;
        let s = json(startup)?;
        require(
            array_hash(&s["native"]["export_manifest_sha256"])? == self.native(role)?.export.sha256,
            "startup export digest differs",
        )
    }
    #[cfg(target_os = "linux")]
    fn external_exit_ids(&self, stdout: &[u8], pids: &[u32]) -> Result<Vec<u32>, ArenaError> {
        crate::native_exit::validate_endpoint_exit_trace(
            stdout,
            pids,
            4 - self.expected_provider_sessions(),
        )
    }
    #[cfg(target_os = "linux")]
    fn validate_process_exit_trace(&self, stdout: &[u8], pids: &[u32]) -> Result<(), ArenaError> {
        self.external_exit_ids(stdout, pids).map(|_| ())
    }
}
pub fn prepare_pals_pair_launch_v4(
    spec: &LockedPalsArenaLaunchV4,
    source_root: &Path,
    output_root: &Path,
    label: &str,
) -> Result<NativeLaunchOwner<LockedPalsArenaLaunchV4>, Box<crate::NativePreparationFailure>> {
    if let Err(cause) = verify_pals_launch_exports_v4(spec, source_root) {
        return Err(Box::new(crate::NativePreparationFailure {
            receipt:crate::NativePreparationReceipt {receipt_version:4,execution_ready:false,input_sha256:spec.sha256.clone(),
                output_directory:"not-created".into(),attempt_created:false,attempted_snapshot_relative_paths:vec![],completed_snapshots:vec![],
                child_spawned:false,writers_closed:true,input_pins_required:false,original_error:cause.to_string(),
                subsequent_file_owner:"source assets retained; descriptor verification failed before snapshot creation or child launch".into(),automatic_retry:false},
            cause,receipt_artifact:None,persistence_error:None,
        }));
    }
    crate::native_launch::prepare_native_launch_for(spec, source_root, output_root, label)
}
/// Uses the existing bounded runner, clocks, Rules replay and cleanup receipts.
pub fn run_pals_pair_v4(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV4>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<
    crate::NativePairOutput<LockedPalsArenaLaunchV4>,
    Box<crate::NativePairFailure<LockedPalsArenaLaunchV4>>,
> {
    crate::native_runner::run_native_pair_for(owner, cancel)
}

/// Time scope starts before UCI preflight and ends after mandatory arena
/// postchecks and receipt persistence. Snapshot preparation remains separate.
pub struct PalsObservedPairOutputV4 {
    pub native: crate::NativePairOutput<LockedPalsArenaLaunchV4>,
    pub wall_time_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsProcessWorkAuditV4 {
    #[serde(flatten)]
    pub base: PalsProcessWorkAuditV3,
    pub marker: Option<PalsActualFollowupMarkerV4>,
}
impl std::ops::Deref for PalsProcessWorkAuditV4 {
    type Target = PalsProcessWorkAuditV3;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
#[cfg(target_os = "linux")]
fn collect_process_work(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    root: &cap_std::fs::Dir,
) -> Result<Vec<PalsProcessWorkAuditV4>, ArenaError> {
    collect_process_work_with_audits(lock, role, root, &[])
}

#[cfg(target_os = "linux")]
fn observed_output_bytes(
    output: &PalsObservedPairOutputV4,
    work: &[PalsProcessWorkAuditV4],
) -> Result<u64, ArenaError> {
    let p = &output.native.receipt.process;
    let owner = output.native.launch_owner();
    let lock = owner.spec();
    let mut files = BTreeMap::<String, (String, u64)>::new();
    let mut admit = |path: String, sha256: String, bytes: u64| -> Result<(), ArenaError> {
        if let Some(previous) = files.insert(path, (sha256.clone(), bytes)) {
            require(
                previous == (sha256, bytes),
                "V4 retained evidence path has conflicting bytes",
            )?;
        }
        Ok(())
    };
    for artifact in output
        .native
        .receipt
        .artifacts
        .iter()
        .chain(std::iter::once(&output.native.receipt_artifact))
    {
        admit(
            artifact.path.clone(),
            artifact.sha256.clone(),
            artifact.bytes,
        )?;
    }
    for work in work {
        let index = lock
            .endpoint_views
            .iter()
            .position(|endpoint| endpoint.id == work.endpoint_id)
            .ok_or_else(|| invalid("V4 work output has unknown endpoint"))?;
        let root = if index == 0 {
            "baseline-runtime"
        } else {
            "candidate-runtime"
        };
        let filenames = if lock.input.endpoints[index].native_model().is_some() {
            (
                "pals-native-startup.v4.json",
                "pals-native-termination.v4.json",
            )
        } else {
            (
                "search-work-startup.v4.json",
                "search-work-termination.v4.json",
            )
        };
        for (name, sha256, bytes) in [
            (filenames.0, &work.startup_sha256, work.startup_bytes),
            (
                filenames.1,
                &work.termination_sha256,
                work.termination_bytes,
            ),
        ] {
            admit(
                format!(
                    "{}/{root}/native-process-{}/{name}",
                    owner.snapshot.output_directory, work.process_id
                ),
                sha256.clone(),
                bytes,
            )?;
        }
    }
    files
        .values()
        .try_fold(p.observed_output_bytes, |total, (_, bytes)| {
            total.checked_add(*bytes)
        })
        .ok_or_else(|| invalid("V4 output byte observation overflow"))
}
fn observed<T>(value: T, method: &str) -> PalsObservedV3<T> {
    PalsObservedV3::Observed {
        value,
        method: method.into(),
    }
}
fn endpoint_admitted(e: &PalsEndpointReceiptV4) -> bool {
    let model_ok = matches!(e.checks.model, PalsObservedV3::Observed { .. })
        || matches!(e.checks.policies, PalsObservedV3::Unknown);
    model_ok && matches!(e.checks.artifacts,PalsObservedV3::Observed{..})
        && matches!(e.checks.resource_limits,PalsObservedV3::Observed{..})
        && matches!(e.process_cleanup,PalsObservedV3::Observed{..})
        // Enabled lanes require their separately captured lifecycle/trace.
        && match &e.checks.policies {
            PalsObservedV3::Observed{value,..}=>
                (matches!(value.policies.cold_archive,PalsColdArchivePolicyV4::Disabled)||matches!(e.archive,PalsObservedV3::Observed{..}))
                && (matches!(value.policies.paused_stack,PalsPausedStackPolicyV4::Disabled)||matches!(e.paused_stack,PalsObservedV3::Observed{..}))
                && (matches!(value.policies.cuda_warm,PalsCudaWarmPolicyV4::Disabled)||matches!(e.cuda_warm,PalsObservedV3::Observed{..}))
                && (matches!(value.policies.recheck,PalsRecheckPolicyV4::Disabled)||matches!(e.repair_trace_total,PalsObservedV3::Observed{value,..} if value==e.repair_traces.len() as u64)),
            PalsObservedV3::Unknown=>true,
        }
}
#[cfg(target_os = "linux")]
fn endpoint_v4_receipt(
    output: &PalsObservedPairOutputV4,
    role: NativeEngineRole,
    work: Option<&PalsProcessWorkAuditV4>,
    native: Option<&PalsNativeSessionAuditV3>,
    preflight: &crate::ExternalUciPreflight,
    cleanup_ms: u64,
) -> Result<PalsEndpointReceiptV4, ArenaError> {
    let owner = output.native.launch_owner();
    let lock = owner.spec();
    let index = role_index(role);
    let engine = &lock.input.semantic_lock.manifest.engines[index];
    let base = endpoint_shared_work_receipt(lock, role, work.map(|w| &w.base), native, preflight)?;
    let mut out = PalsEndpointReceiptV4::with_unknown_checks(base);
    let artifacts: Vec<&ArtifactRef> = match engine {
        PalsEngineV4::Pals(e) => {
            let mut a = vec![&e.base.binary];
            if let PalsWeightIdentityV3::Untrained { artifact, .. }
            | PalsWeightIdentityV3::Trained { artifact, .. } = &e.base.model.weights
            {
                a.push(artifact);
            }
            if let PalsCpuRSelectionV3::ExternalUci(c) = &e.base.cpu_r {
                a.extend([&c.profile, &c.binary]);
            }
            a
        }
        PalsEngineV4::OwnCpu(e) => vec![&e.binary],
        PalsEngineV4::ReferenceUci(e) => {
            let mut a = vec![&e.binary];
            a.extend(&e.assets);
            a
        }
    };
    let verified = artifacts
        .into_iter()
        .map(|a| {
            let snapshot = owner
                .snapshots()
                .iter()
                .find(|s| s.artifact == *a)
                .ok_or_else(|| invalid("V4 required artifact absent from pinned snapshots"))?;
            require(
                snapshot.sha256 == a.sha256
                    && snapshot.bytes == a.bytes
                    && snapshot.distinct_source_inode
                    && snapshot.closed_writer_read_only,
                "V4 pinned snapshot artifact verification differs",
            )?;
            Ok((
                a.path.clone(),
                PalsVerifiedArtifactV4 {
                    sha256: snapshot.sha256.clone(),
                    bytes: snapshot.bytes,
                    pinned_handle_consumed: output.native.receipt.integration_checks_passed,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, ArenaError>>()?;
    out.checks.artifacts = observed(
        verified,
        "owner copied verified original handle into distinct synced read-only inode; pinned executable/input handles consumed by audited runner/native loading and retired after process/physical cleanup",
    );
    let affinity = verify_pals_inherited_resources_v4(lock)?;
    out.base.actual_affinity = observed(
        affinity,
        "actual /proc/self affinity after inherited native admission; child affinity is separately attested for external checker",
    );
    out.checks.resource_limits = observed(
        lock.input.semantic_lock.manifest.resources[index].clone(),
        "actual /proc/self affinity and cgroup-v2 memory.high/max/swap.max independently rechecked; requested device index only, no VRAM peak claim",
    );
    if work.is_some() {
        out.checks.physical_work_accounting = observed(
            true,
            "validated actual producer invocation/CPU task counters and native physical B1 inputs/calls separately; current process/game mapping and no unknown owner",
        );
    }
    if let PalsEngineV4::Pals(e) = engine {
        let actual = work.and_then(|w| w.marker.as_ref()).ok_or_else(|| {
            invalid("V4 policy receipt requires an observed actual immutable marker")
        })?;
        let identity = PalsPolicyIdentityV4 {
            version: e.policy_identity.version.clone(),
            policy: e.policy_identity.policy.clone(),
            search_identity: actual.policy_identity.clone(),
            conditions_sha256: actual
                .policy_conditions_sha256
                .ok_or_else(|| invalid("V4 actual policy conditions unknown"))?,
        };
        let mut actual_policies = e.policies.clone();
        actual_policies.resolver_identity.implementation_sha256 =
            hex(actual.resolver_implementation_sha256);
        actual_policies.resolver_semantics_sha256 = hex(actual.resolver_semantics_sha256);
        out.checks.policies = observed(
            PalsPolicyCheckV4 {
                search: PalsComponentV3 {
                    semantic_id: actual.policy_identity.clone(),
                    implementation_sha256: hex(actual.search_implementation_sha256),
                    options: e.base.search.options.clone(),
                },
                policies: actual_policies,
                identity,
            },
            "actual driver getters + resolver semantic digest, exact recheck condition hash, checker profile/own registration, profile/stack/CUDA argv and loaded marker compared independently with lock; disabled optional lanes remain off",
        );
        if let Some(n) = native {
            require(
                actual.model_configuration.as_ref() == Some(&config(e))
                    && array_hash(&n.raw_native["model_epoch"])?
                        == match &e.base.model.weights {
                            PalsWeightIdentityV3::Untrained { artifact, .. }
                            | PalsWeightIdentityV3::Trained { artifact, .. } => {
                                artifact.sha256.clone()
                            }
                            _ => return Err(invalid("V4 actual checkpoint absent")),
                        },
                "V4 actual model proof drifted",
            )?;
            let mut model = e.base.model.clone();
            model.implementation_sha256 = hex(actual
                .model_implementation_sha256
                .ok_or_else(|| invalid("V4 actual model source pin unknown"))?);
            let mut specification = e.model_v2.clone();
            specification.model_semantics_sha256 = hex(actual
                .model_semantics_sha256
                .ok_or_else(|| invalid("V4 actual model semantics pin unknown"))?);
            out.checks.model = observed(
                PalsModelCheckV4 {
                    model,
                    specification,
                },
                "loaded backend profile/configuration/epoch plus actual registered semantics and compiled model boundary SHA getters compared with independently pinned manifest; original graph/runtime/checkpoint/export bytes validated; diagnostic export rejected before spawn",
            );
        }
    }
    let process = &output.native.receipt.process;
    let exited = out.base.process_exited
        && process.group_cleanup == crate::CleanupStatus::Gone
        && process.exit_code == Some(0)
        && output.native.receipt.cleanup_verified
        && !output.native.receipt.unresolved_owner_retained;
    require(
        exited,
        "V4 supervisor/game-process ownership cleanup incomplete",
    )?;
    out.process_cleanup = observed(
        PalsProcessCleanupV4 {
            owner_identity: format!("arena-process-group-{}-{}", process.pid, lock.sha256),
            process_exited: true,
            exit_code: process.exit_code,
            exit_signal: process.exit_signal,
            child_processes_remaining: 0,
            stdout_drained: true,
            stderr_drained: true,
            stdout_bytes: process.stdout_bytes,
            stderr_bytes: process.stderr_bytes,
            elapsed_ms: cleanup_ms,
            cleanup_complete: true,
            ownership_lost: false,
        },
        "supervisor-owned process-group Gone/reaped normal exit with joined stdout/stderr reader completion; each game PID separately checked against exact renderer/exits and native shutdown; interval scoped to whole pair owner",
    );
    Ok(out)
}
pub fn run_pals_pair_observed_v4(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV4>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<PalsObservedPairOutputV4, Box<crate::NativePairFailure<LockedPalsArenaLaunchV4>>> {
    let started = std::time::Instant::now();
    run_pals_pair_v4(owner, cancel).map(|native| PalsObservedPairOutputV4 {
        native,
        wall_time_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    })
}

pub fn verify_pals_launch_exports_v4(
    spec: &LockedPalsArenaLaunchV4,
    source_root: &Path,
) -> Result<(), ArenaError> {
    verify_external_profiles(&spec.input, source_root)?;
    for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
        if spec.uses_provider_records(role)? {
            let n = spec.native(role)?;
            let mut file = n.export.open_verified(source_root, 128 * 1024)?;
            let mut bytes = Vec::with_capacity(n.export.bytes as usize);
            file.read_to_end(&mut bytes)
                .map_err(|e| ArenaError::Io(e.to_string()))?;
            spec.validate_export(role, &bytes)?;
        }
        if let Some(cuda) = spec.input.endpoints[role_index(role)].cuda_model() {
            let mut file = cuda
                .cuda_bundle
                .manifest
                .open_verified(source_root, 64 * 1024)?;
            let mut bytes = Vec::with_capacity(cuda.cuda_bundle.manifest.bytes as usize);
            file.read_to_end(&mut bytes)
                .map_err(|e| ArenaError::Io(e.to_string()))?;
            spec.validate_cuda_bundle(role, &bytes)?;
            if let Some(control) = &cuda.cuda_control {
                let mut file = control
                    .inventory
                    .open_verified(source_root, MAX_CONTROL_INVENTORY_BYTES)?;
                let mut bytes = Vec::with_capacity(control.inventory.bytes as usize);
                file.read_to_end(&mut bytes)
                    .map_err(|e| ArenaError::Io(e.to_string()))?;
                spec.validate_control_inventory(role, &bytes)?;
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn verify_pals_inherited_resources_v4(
    spec: &LockedPalsArenaLaunchV4,
) -> Result<Vec<u32>, ArenaError> {
    fn read(path: &Path) -> Result<String, ArenaError> {
        let mut f = std::fs::File::open(path).map_err(|e| ArenaError::Io(e.to_string()))?;
        let mut bytes = vec![];
        f.by_ref()
            .take(64 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| ArenaError::Io(e.to_string()))?;
        require(
            bytes.len() <= 64 * 1024,
            "resource observation budget exceeded",
        )?;
        String::from_utf8(bytes).map_err(|_| invalid("resource observation is not UTF-8"))
    }
    let status = read(Path::new("/proc/self/status"))?;
    let cpus = status
        .lines()
        .find_map(|s| s.strip_prefix("Cpus_allowed_list:"))
        .ok_or_else(|| invalid("missing affinity evidence"))?
        .trim();
    let actual = parse_pals_affinity(cpus)?;
    let r = &spec.input.semantic_lock.manifest.resources[0];
    let mut wanted = r.cpu_affinity.clone();
    wanted.sort_unstable();
    require(
        actual == wanted,
        "inherited affinity differs from PALS launch",
    )?;
    let membership = read(Path::new("/proc/self/cgroup"))?;
    let relative = membership
        .lines()
        .find_map(|s| s.strip_prefix("0::"))
        .ok_or_else(|| invalid("cgroup v2 is required"))?;
    require(
        relative.starts_with('/') && !relative.split('/').any(|c| c == ".." || c == "."),
        "cgroup membership path invalid",
    )?;
    let root = Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
    for (file, value) in [
        ("memory.high", r.memory_high_bytes),
        ("memory.max", r.memory_max_bytes),
        ("memory.swap.max", r.swap_max_bytes),
    ] {
        require(
            read(&root.join(file))?.trim() == value.to_string(),
            "inherited cgroup memory limit differs from PALS launch",
        )?;
    }
    Ok(actual)
}

#[cfg(target_os = "linux")]
fn collect_process_work_with_audits(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    root: &cap_std::fs::Dir,
    native_audits: &[PalsNativeSessionAuditV3],
) -> Result<Vec<PalsProcessWorkAuditV4>, ArenaError> {
    use cap_fs_ext::DirExt;
    if matches!(
        &lock.input.endpoints[role_index(role)],
        PalsEndpointLaunchV3::ReferenceUci { .. }
    ) {
        return Ok(vec![]);
    }
    let path = if role_index(role) == 0 {
        "baseline-runtime"
    } else {
        "candidate-runtime"
    };
    let runtime = root
        .open_dir_nofollow(path)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    let native = matches!(
        &lock.input.endpoints[role_index(role)],
        PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
    );
    let (start_name, end_name) = if native {
        (
            "pals-native-startup.v4.json",
            "pals-native-termination.v4.json",
        )
    } else {
        (
            "search-work-startup.v4.json",
            "search-work-termination.v4.json",
        )
    };
    let mut records = vec![];
    for (index, entry) in runtime
        .entries()
        .map_err(|e| ArenaError::Io(e.to_string()))?
        .enumerate()
    {
        require(
            index < lock.input.budget.max_runtime_files as usize,
            "work runtime entry bound exceeded",
        )?;
        let entry = entry.map_err(|e| ArenaError::Io(e.to_string()))?;
        let filename = entry.file_name();
        let name = filename
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 runtime slot"))?;
        if name == "runtime-cache" && native {
            continue;
        }
        require(
            name.starts_with("native-process-") && records.len() < 2,
            "unexpected or excessive work runtime slot",
        )?;
        let directory = runtime
            .open_dir_nofollow(&filename)
            .map_err(|e| ArenaError::Io(e.to_string()))?;
        let start = read_work_file(&directory, start_name)?;
        let end = read_work_file(&directory, end_name)?;
        let mut matching = native_audits.iter().filter(|audit| {
            audit.endpoint_id == lock.endpoint_views[role_index(role)].id
                && name == format!("native-process-{}", audit.process_id)
        });
        let audit = matching.next();
        require(matching.next().is_none(), "native work audit duplicated")?;
        records.push(validate_work_records(
            lock, role, &start, &end, name, audit,
        )?);
    }
    require(
        records.len() == 2 && records[0].process_id != records[1].process_id,
        "two distinct restarted game work processes required",
    )?;
    Ok(records)
}

#[cfg(any(target_os = "linux", test))]
fn endpoint_shared_work_receipt(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    work: Option<&PalsProcessWorkAuditV3>,
    native: Option<&PalsNativeSessionAuditV3>,
    preflight: &crate::ExternalUciPreflight,
) -> Result<PalsEndpointReceiptV3, ArenaError> {
    let index = role_index(role);
    let engine = &lock.input.semantic_lock.manifest.engines[index];
    require(
        preflight.engine_id == engine.id(),
        "work/preflight endpoint differs",
    )?;
    let mut options = BTreeMap::new();
    for (key, requested) in requested_options(engine) {
        let value = preflight
            .options
            .get(key)
            .ok_or_else(|| invalid("requested option missing from preflight"))?;
        require(
            value.value_supported && value.sent_to_preflight && value.readiness_barrier_observed,
            "requested option/preflight barrier unsupported",
        )?;
        options.insert(
            key.clone(),
            PalsOptionReceiptV3 {
                requested: requested.clone(),
                advertised_supported: true,
                observed: value
                    .actual_value
                    .as_ref()
                    .map_or(PalsObservedV3::Unknown, |value| PalsObservedV3::Observed {
                        value: value.clone(),
                        method: "independent option value observation from preflight".into(),
                    }),
            },
        );
    }
    let mut receipt = PalsEndpointReceiptV3 {
        endpoint_id: engine.id().into(),
        external_cpu_r: None,
        uci_ready_observed: true,
        search_failed_go_count: None,
        pals_search_policy: None,
        options,
        actual_affinity: PalsObservedV3::Unknown,
        memory_peak_bytes: PalsObservedV3::Unknown,
        gpu_device: PalsObservedV3::Unknown,
        vram_peak_bytes: PalsObservedV3::Unknown,
        nn_inputs_completed: 0,
        nn_calls_completed: 0,
        nn_inputs_consumed: 0,
        cached_evaluations_consumed: 0,
        proposer_tasks_completed: 0,
        critic_tasks_completed: 0,
        cpu_tasks_requested: 0,
        cpu_tasks_completed: 0,
        cpu_tasks_reused_consumed: 0,
        cpu_tasks_consumed: 0,
        cpu_nodes: 0,
        physical_state: PalsPhysicalStateV3::NotRequired,
        buffers_released: true,
        process_exited: true,
    };
    match engine {
        PalsEngineV4::ReferenceUci(_) => require(
            work.is_none() && native.is_none(),
            "reference UCI cannot inherit RoveZero work",
        )?,
        PalsEngineV4::OwnCpu(_) => {
            let w = &work
                .ok_or_else(|| invalid("own CPU work unobserved"))?
                .search_work;
            validate_pals_process_counter_units(w, "cpu", false, false, false)?;
            receipt.search_failed_go_count = Some(work_count(w, "failed_returns")?);
            let cpu = &w["cpu"];
            receipt.cpu_tasks_requested = work_count(cpu, "tasks_requested")?;
            receipt.cpu_tasks_completed = work_count(cpu, "requested_depth_completed")?
                .checked_add(work_count(cpu, "rules_terminal_reports")?)
                .ok_or_else(|| invalid("CPU completed task overflow"))?;
            // Partial/frontier estimates are retained in raw work evidence; the
            // core completed-task consumer count has the stricter unit.
            receipt.cpu_tasks_consumed =
                work_count(cpu, "completed_reports_accepted_for_uci_output")?;
            receipt.cpu_nodes = work_count(cpu, "nodes")?;
            require(native.is_none(), "own CPU cannot claim native NN evidence")?;
        }
        PalsEngineV4::Pals(endpoint) => {
            let p = &endpoint.base;
            let w = &work
                .ok_or_else(|| invalid("PALS search work unobserved"))?
                .search_work;
            let external = !p.cpu_r.is_own();
            validate_pals_process_counter_units(w, "pals", false, external, false)?;
            receipt.search_failed_go_count = Some(work_count(w, "failed_returns")?);
            let t = &w["pals"];
            receipt.proposer_tasks_completed = work_count(t, "completed_proposer_calls")?
                .checked_add(work_count(t, "completed_repair_calls")?)
                .ok_or_else(|| invalid("proposer task overflow"))?;
            receipt.critic_tasks_completed = work_count(t, "completed_critic_calls")?;
            receipt.cpu_tasks_requested = work_count(t, "cpu_tasks_requested")?;
            receipt.cpu_tasks_completed = work_count(t, "completed_cpu_tasks")?;
            receipt.cpu_tasks_reused_consumed =
                work_count(t, "reused_completed_cpu_tasks_consumed")?;
            receipt.cpu_tasks_consumed = work_count(t, "consumed_cpu_tasks")?;
            receipt.cpu_nodes = work_count(t, "cpu_nodes")?;
            if external {
                let work = work.unwrap();
                let n = native.ok_or_else(|| {
                    invalid("external CPU_R lacks production native session audit")
                })?;
                let helper = n.external_cpu_r_session.as_ref().ok_or_else(|| {
                    invalid("external CPU_R lacks validated independent helper projection")
                })?;
                require(
                    helper.purpose == PalsExternalCpuRSessionPurposeV3::Game
                        && helper.endpoint_id == receipt.endpoint_id
                        && helper.parent_process_id == work.process_id
                        && n.launch_sha256 == lock.sha256
                        && n.startup_sha256 == work.startup_sha256
                        && n.termination_sha256 == work.termination_sha256
                        && n.raw_search_work == work.search_work
                        && helper.startup_sha256 == work.startup_sha256
                        && helper.termination_sha256 == work.termination_sha256
                        && receipt.cpu_tasks_requested == 0
                        && receipt.cpu_tasks_completed == 0
                        && receipt.cpu_tasks_reused_consumed == 0
                        && receipt.cpu_tasks_consumed == 0
                        && receipt.cpu_nodes == 0,
                    "external helper/game/work projection differs or claims Own work",
                )?;
                helper
                    .receipt
                    .validate_against(p, &lock.input.semantic_lock.manifest.resources[index])?;
                receipt.external_cpu_r = Some(helper.receipt.clone());
            } else {
                require(
                    native.is_none_or(|n| n.external_cpu_r_session.is_none()),
                    "own CPU_R cannot inherit validated foreign projection",
                )?;
            }
            if p.model.backend == PalsModelBackendV3::DeterministicMock {
                require(native.is_none(), "explicit mock cannot claim NN inputs")?;
            } else {
                let n =
                    native.ok_or_else(|| invalid("PALS native work missing completion receipt"))?;
                require(
                    n.process_id == work.unwrap().process_id
                        && n.endpoint_id == receipt.endpoint_id
                        && n.physical_shutdown_confirmed
                        && n.native_buffers_released,
                    "native/work process or physical release differs",
                )?;
                require(
                    p.model.frozen_epoch == 1 && count(&n.raw_native, "frozen_epoch")? == 1,
                    "new Core requires actual deployment frozen epoch 1",
                )?;
                require(
                    n.raw_native["backend_stats_observation"] == "exclusive_worker_before_shutdown",
                    "native graph work observation missing",
                )?;
                require(
                    count(&n.raw_native, "observer_failures")? == 0
                        && n.raw_native["last_observer_failure"].is_null(),
                    "Core requires actual successful observer completion",
                )?;
                let stats = &n.raw_native["backend_stats"];
                let (all_inputs, all_calls, all_roles) = graph_stats(stats)?;
                receipt.nn_inputs_completed = all_inputs
                    .checked_sub(n.startup_nn_inputs_completed)
                    .ok_or_else(|| invalid("NN cumulative inputs below startup snapshot"))?;
                receipt.nn_calls_completed = all_calls
                    .checked_sub(n.startup_nn_calls_completed)
                    .ok_or_else(|| invalid("NN cumulative calls below startup snapshot"))?;
                require(
                    receipt.nn_inputs_completed == receipt.nn_calls_completed
                        && all_roles.checked_sub(n.startup_role_inputs_completed)
                            == Some(n.completed_role_inputs),
                    "physical B1 graph inputs/completion or role completion differs",
                )?;
                receipt.nn_inputs_consumed = n.search_consumed_role_inputs;
                require(
                    work_count(t, "consumed_role_outputs")? == receipt.nn_inputs_consumed
                        && receipt
                            .proposer_tasks_completed
                            .checked_add(receipt.critic_tasks_completed)
                            .is_some_and(|completed| completed <= n.completed_role_inputs),
                    "native role completion/search consumption differs from observed search work",
                )?;
                receipt.physical_state = PalsPhysicalStateV3::Completed;
                if let Some(cuda) = lock.input.endpoints[index].cuda_model() {
                    let observed = validate_cuda_placement_witness(
                        cuda,
                        &n.raw_native["startup_probe"]["cuda_placement_witness"],
                    )?;
                    require(
                        observed == n.cuda_placement,
                        "Core CUDA placement projection differs from raw startup evidence",
                    )?;
                    if let Some(placement) = observed {
                        receipt.gpu_device=PalsObservedV3::Observed {
                            value:format!("cuda:{}",placement.device_id),
                            method:"pinned producer CUDA EP index exercised by exact pre-Run coverage and actual public/P/C neural kernel profiles; physical GPU UUID/model and VRAM peak unobserved".into(),
                        };
                    }
                }
            }
        }
    }
    Ok(receipt)
}

#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
fn verify_external_profile(
    semantic_lock: &PalsInputLockV4,
    role: NativeEngineRole,
    source_root: &Path,
    registered_program: &Path,
    registered_cwd: &Path,
) -> Result<PalsExternalCpuRProfileAuditV3, ArenaError> {
    use rz_experiments::PalsCpuRSelectionV3;
    use rz_uci::pals_checker_profile::{PalsCheckerProfile, PalsCheckerSelection};

    semantic_lock.verify()?;
    let manifest = &semantic_lock.manifest;
    let PalsEngineV4::Pals(endpoint) = &manifest.engines[role_index(role)] else {
        return Err(invalid("external CPU_R profile requires a PALS endpoint"));
    };
    let endpoint = &endpoint.base;
    let PalsCpuRSelectionV3::ExternalUci(declaration) = &endpoint.cpu_r else {
        return Err(invalid(
            "own CPU_R selection cannot inherit an external profile",
        ));
    };
    require_public_artifact(&declaration.profile)?;
    require_public_artifact(&declaration.binary)?;
    require(
        source_root.is_absolute()
            && registered_program.is_absolute()
            && registered_cwd.is_absolute(),
        "external CPU_R profile root and registered program/cwd must be absolute",
    )?;
    let program = registered_program
        .to_str()
        .ok_or_else(|| invalid("registered CPU_R program is not UTF-8"))?;
    let cwd = registered_cwd
        .to_str()
        .ok_or_else(|| invalid("registered CPU_R cwd is not UTF-8"))?;
    // Validate only path names here. These are not executable/cwd opens; the
    // owner performs its independent pinned-FD verification before spawn.
    let mut registered_name = declaration.binary.clone();
    registered_name.path = program.trim_start_matches('/').into();
    require_public_artifact(&registered_name)?;
    let loaded = PalsCheckerProfile::load(
        &source_root.join(&declaration.profile.path),
        &declaration.profile.sha256,
    )
    .map_err(|error| invalid(format!("external CPU_R profile load: {error}")))?;
    let profile = loaded.profile();
    let registration = loaded.registration();
    require(
        loaded.canonical_sha256() == declaration.profile_canonical_sha256
            && registration.file_bytes == declaration.profile.bytes
            && profile.selection == PalsCheckerSelection::StockfishEmbeddedNnue
            && profile.program == program
            && profile.working_directory == cwd,
        "external CPU_R actual profile canonical/size/selection or registered paths differ",
    )?;
    require(
        profile.identity.binary_sha256 == declaration.binary.sha256
            && profile.identity.declared_source == declaration.binary.source
            && profile.identity.declared_license == declaration.binary.license,
        "external CPU_R loaded profile differs from independently registered binary/source/license",
    )?;
    require(
        declaration.resolver.version == rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION
            && declaration.resolver.semantics_sha256
                == digest(rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS.as_bytes()),
        "external CPU_R resolver declaration differs from actual model-WDL semantics",
    )?;
    let policy = &declaration.policy;
    let option_number = |name: &str| -> Result<u32, ArenaError> {
        profile
            .identity
            .options
            .get(name)
            .and_then(|value| value.parse::<u32>().ok())
            .ok_or_else(|| invalid(format!("external CPU_R profile {name} cap missing")))
    };
    require(
        option_number("Threads")? <= policy.threads_max
            && option_number("Hash")? <= policy.hash_mib_max
            && u32::from(profile.max_depth) <= policy.max_depth
            && profile.max_prefix_plies as u64 <= u64::from(policy.max_prefix_plies)
            && profile.handshake_timeout_ms <= policy.handshake_max_ms
            && profile.max_task_wall_time_ms <= policy.task_wall_time_max_ms
            && profile.stop_grace_ms <= policy.stop_grace_max_ms
            && profile.shutdown_grace_ms <= policy.shutdown_grace_max_ms
            && profile.max_output_bytes as u64 <= policy.lifetime_output_bytes_max
            && profile.max_line_bytes as u64 <= policy.line_bytes_max,
        "external CPU_R actual profile exceeds registered inherited task/time/output ceilings",
    )?;
    // The loader enforces the exact first-selection option whitelist, empty
    // argv/assets, and unknown embedded model metadata. Preserve that complete
    // declaration with unknown observations; readyok cannot fill those fields.
    let profile_registration = serde_json::to_value(registration)
        .map_err(|error| invalid(format!("external CPU_R profile audit encoding: {error}")))?;
    Ok(PalsExternalCpuRProfileAuditV3 {
        domain: "rz-pals-external-cpu-r-profile-audit-v4/1",
        endpoint_id: endpoint.id.clone(),
        semantic_lock_sha256: semantic_lock.canonical_sha256.clone(),
        profile_file_sha256: loaded.file_sha256().into(),
        profile_canonical_sha256: loaded.canonical_sha256().into(),
        profile_file_bytes: declaration.profile.bytes,
        registered_binary_sha256: declaration.binary.sha256.clone(),
        registered_program: program.into(),
        registered_working_directory: cwd.into(),
        profile_registration,
        profile_file_verification_performed: true,
        binary_file_verification_performed: false,
        execution_admission_completed: false,
        child_spawned: false,
        options_application_observed: None,
        model_loading_observed: None,
        inherited_resource_join_observed: None,
        physical_shutdown_observed: None,
    })
}

fn validate_work_records(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
    audited_native: Option<&PalsNativeSessionAuditV3>,
) -> Result<PalsProcessWorkAuditV4, ArenaError> {
    require(
        startup.len() <= MAX_JSON_BYTES && termination.len() <= MAX_JSON_BYTES,
        "work envelope byte budget exceeded",
    )?;
    let (s, t) = (json(startup)?, json(termination)?);
    lock.validate_wire_pair(role, &s, &t)?;
    let marker = validate_followup_marker_pair(lock, role, &s, &t)?;
    let engine = &lock.endpoint_views[role_index(role)];
    let external = lock
        .input
        .external_cpu_r_binding(role_index(role))?
        .is_some();
    let native = matches!(
        &lock.input.endpoints[role_index(role)],
        PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
    );
    let (sdomain, tdomain, kind) = match &lock.input.endpoints[role_index(role)] {
        PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_) => (
            PALS_NATIVE_STARTUP_V4_DOMAIN,
            PALS_NATIVE_TERMINATION_V4_DOMAIN,
            "pals",
        ),
        PalsEndpointLaunchV3::LegalOrderMock(_) => (
            PALS_SEARCH_WORK_STARTUP_V4_DOMAIN,
            PALS_SEARCH_WORK_TERMINATION_V4_DOMAIN,
            "pals",
        ),
        PalsEndpointLaunchV3::OwnCpu { .. } => (
            PALS_SEARCH_WORK_STARTUP_V4_DOMAIN,
            PALS_SEARCH_WORK_TERMINATION_V4_DOMAIN,
            "cpu",
        ),
        PalsEndpointLaunchV3::ReferenceUci { .. } => {
            return Err(invalid(
                "external reference has no invented RoveZero work envelope",
            ));
        }
    };
    let pid = s["process_id"]
        .as_u64()
        .filter(|v| *v > 0 && *v <= i32::MAX as u64)
        .ok_or_else(|| invalid("work process PID invalid"))? as u32;
    require(
        session == format!("native-process-{pid}"),
        "work directory/PID mismatch",
    )?;
    for (value, domain, ending) in [(&s, sdomain, false), (&t, tdomain, true)] {
        require(
            value["schema_version"] == 4
                && value["domain"] == domain
                && value["endpoint_id"] == engine.id
                && value["launch_sha256"] == lock.sha256
                && value["process_id"] == pid
                && value["binary_sha256"] == engine.binary.sha256
                && value["service_exit_success"] == ending,
            "work envelope identity/start/end differs",
        )?;
        validate_pals_process_counter_units(&value["search_work"], kind, !ending, external, false)?;
    }
    if native {
        // Native worker counters have their own accepted/delivered lifecycle.
        if external {
            let audit = audited_native.ok_or_else(|| {
                invalid("external work requires production owner/supervisor native audit")
            })?;
            require(
                audit.endpoint_id == engine.id
                    && audit.process_id == pid
                    && audit.launch_sha256 == lock.sha256
                    && audit.startup_sha256 == digest(startup)
                    && audit.termination_sha256 == digest(termination)
                    && audit.raw_search_work == t["search_work"]
                    && audit.raw_native == t["native"]
                    && audit.physical_shutdown_confirmed
                    && audit.native_buffers_released
                    && audit.external_cpu_r_session.as_ref().is_some_and(|helper| {
                        helper.purpose == PalsExternalCpuRSessionPurposeV3::Game
                            && helper.parent_process_id == pid
                            && helper.startup_sha256 == audit.startup_sha256
                            && helper.termination_sha256 == audit.termination_sha256
                    }),
                "external work and validated native/helper session differ",
            )?;
        } else {
            validate_pals_native_records_inner(
                lock,
                role,
                startup,
                termination,
                session,
                None,
                true,
            )?;
        }
    }
    Ok(PalsProcessWorkAuditV4 {
        base: PalsProcessWorkAuditV3 {
            endpoint_id: engine.id.clone(),
            process_id: pid,
            startup_sha256: digest(startup),
            termination_sha256: digest(termination),
            startup_bytes: startup.len() as u64,
            termination_bytes: termination.len() as u64,
            search_work: t["search_work"].clone(),
        },
        marker,
    })
}

#[cfg(any(target_os = "linux", test))]
fn project_core_pgn(
    lock: &LockedPalsArenaLaunchV4,
    native_artifact: &ArtifactRef,
) -> Result<(ArtifactRef, PalsPgnProvenanceV3), ArenaError> {
    // Generated native artifacts use source as an execution description. Core's
    // ArtifactRef contract requires a public producer URL instead; retain the
    // original reference and description without changing the recorded bytes.
    let runner = &lock.input.runner;
    let mut artifact = native_artifact.clone();
    artifact.source = runner.source_url.clone();
    artifact.validate()?;
    let provenance = PalsPgnProvenanceV3 {
        native_artifact: native_artifact.clone(),
        producer_source_url: runner.source_url.clone(),
        producer_source_commit: runner.source_commit.clone(),
        producer_binary_sha256: runner.binary.sha256.clone(),
        description: "Generated match PGN from the pinned Fastchess execution; the public URL identifies the producer's source repository, not a public download location for this local PGN. The original native reference preserves its execution description, path, digest, size and license.".into(),
    };
    Ok((artifact, provenance))
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsCoreAssemblyV4 {
    pub domain: String,
    pub arena_launch_sha256: String,
    pub native_receipt: ArtifactRef,
    pub wall_time_ms: Option<u64>,
    pub cleanup_time_ms: Option<u64>,
    pub timing_scope: String,
    pub game_process_correspondence: String,
    pub work: Vec<PalsProcessWorkAuditV4>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pgn_provenance: Option<PalsPgnProvenanceV3>,
    pub core: Option<PalsRunReceiptV4>,
    pub assembly_error: Option<String>,
    pub training_executed: bool,
}

#[cfg(target_os = "linux")]
pub fn assemble_pals_core_receipt_v4(
    output: &PalsObservedPairOutputV4,
    cleanup_time_ms: Option<u64>,
) -> PalsCoreAssemblyV4 {
    let native = &output.native;
    let owner = native.launch_owner();
    let lock = owner.spec();
    let mut assembly = PalsCoreAssemblyV4 { domain: "rz-pals-core-assembly-v4/1".into(), arena_launch_sha256: lock.sha256.clone(),
        native_receipt: native.receipt_artifact.clone(), wall_time_ms: Some(output.wall_time_ms), cleanup_time_ms,
        timing_scope: "UCI preflight through mandatory arena postcheck/receipt; snapshot preparation separate; cleanup is supervisor-owned measured interval".into(),
        game_process_correspondence: "pinned Fastchess 1.8.2 source: one synchronous worker, restart=on, recover=false; TRACE/renderer game id and color plus finished-white-exit-black-exit windows; separate native/owned-group completion gates required".into(),
        work: vec![], pgn_provenance: None, core: None, assembly_error: None, training_executed: false };
    let result: Result<PalsRunReceiptV4, ArenaError> = (|| {
        lock.input.require_reap_status_runner()?;
        lock.input.require_actual_native_epoch()?;
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            assembly.work.extend(collect_process_work_with_audits(
                lock,
                role,
                &owner.snapshot.directory,
                &native.receipt.provider_sessions,
            )?);
        }
        let cleanup = cleanup_time_ms.ok_or_else(|| {
            invalid("cleanup interval unobserved; original arena receipt retained")
        })?;
        let receipts = &native.receipt;
        require(
            receipts.integration_checks_passed
                && receipts.cleanup_verified
                && !receipts.unresolved_owner_retained,
            "arena integration/process/physical gates failed; original failure is preserved",
        )?;
        let pgn = receipts
            .pgn_audit
            .as_ref()
            .ok_or_else(|| invalid("Rules PGN audit absent"))?;
        let clock = receipts
            .clock_audit
            .as_ref()
            .ok_or_else(|| invalid("whole-game clock observation absent"))?;
        require(
            pgn.games.len() == 2 && clock.games.len() == 2,
            "two exchanged audited games required",
        )?;
        let exits =
            crate::native_exit::validate_endpoint_exit_trace(&native.process().stdout, &[], 4)?;
        let mut known = BTreeSet::new();
        for work in &assembly.work {
            require(
                known.insert(work.process_id),
                "work process reused across endpoints",
            )?;
        }
        let known_work_pids: [BTreeSet<u32>; 2] = std::array::from_fn(|index| {
            let id = lock.input.semantic_lock.manifest.engines[index].id();
            assembly
                .work
                .iter()
                .filter(|w| w.endpoint_id == id)
                .map(|w| w.process_id)
                .collect()
        });
        let mapped = validate_pals_game_process_trace_context(
            lock,
            &native.process().stdout,
            &known_work_pids,
        )?;
        let per_role = mapped.map(|pids| pids.to_vec());
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            let index = role_index(role);
            require(
                per_role[index].len() == 2 && per_role[index].iter().all(|pid| exits.contains(pid)),
                "work identity/normal exit game correspondence incomplete",
            )?;
            let view = &lock.endpoint_views[index];
            let options = view
                .requested_options
                .iter()
                .map(|(key, value)| {
                    crate::external_uci::resolve_asset_tokens(value, view, &owner.snapshot.pins)
                        .map(|v| (key.clone(), v))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            crate::external_uci::audit_external_game_protocol(
                &native.process().stdout,
                view,
                &options,
                &per_role[index],
            )?;
        }
        let native_pgn_artifact = receipts
            .artifacts
            .iter()
            .find(|a| a.path.ends_with("/match.pgn"))
            .ok_or_else(|| invalid("retained PGN artifact absent"))?;
        let (pgn_artifact, provenance) = project_core_pgn(lock, native_pgn_artifact)?;
        assembly.pgn_provenance = Some(provenance);
        let m = &lock.input.semantic_lock.manifest;
        let mut failures = BTreeSet::new();
        if output.wall_time_ms > m.pilot.wall_time_max_ms {
            failures.insert(PalsRunFailureV3::WallDeadline);
        }
        if cleanup > m.pilot.cleanup_max_ms {
            failures.insert(PalsRunFailureV3::CleanupTimeout);
        }
        let mut games = vec![];
        for (index, game) in pgn.games.iter().enumerate() {
            require(
                clock.games.iter().any(|c| c.game_id == game.game_id),
                "clock/game identity differs",
            )?;
            let mut engines = vec![];
            for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
                let ri = role_index(role);
                let id = m.engines[ri].id();
                let pid = per_role[ri][index];
                let work = assembly
                    .work
                    .iter()
                    .find(|w| w.endpoint_id == id && w.process_id == pid);
                if game.classification != "engine_loss"
                    && let Some(work) = work
                {
                    let plies = game.uci_moves.len() as u64;
                    let expected_go = if game.white_engine == id {
                        plies / 2 + plies % 2
                    } else {
                        plies / 2
                    };
                    require(
                        work_count(&work.search_work, "go_invocations")? == expected_go,
                        "per-game work invocation count differs from audited moves",
                    )?;
                }
                let nn = receipts
                    .provider_sessions
                    .iter()
                    .find(|n| n.endpoint_id == id && n.process_id == pid);
                let preflight = receipts
                    .external_preflight
                    .iter()
                    .find(|p| p.engine_id == id)
                    .ok_or_else(|| invalid("pinned UCI preflight absent"))?;
                let endpoint = endpoint_v4_receipt(output, role, work, nn, preflight, cleanup)?;
                record_endpoint_search_failure(&endpoint.base, &mut failures);
                engines.push(endpoint);
            }
            let (result, termination, failed_endpoint) = core_game_outcome(game)?;
            if result == PalsResultV3::Incomplete {
                failures.insert(PalsRunFailureV3::IncompleteGame);
            }
            games.push(PalsGameReceiptV4 {
                game_index: index as u32,
                white_endpoint: game.white_engine.clone(),
                black_endpoint: game.black_engine.clone(),
                base_ms: m.pilot.base_ms,
                increment_ms: m.pilot.increment_ms,
                plies: u32::try_from(game.uci_moves.len())
                    .map_err(|_| invalid("ply count overflow"))?,
                result,
                termination,
                failed_endpoint,
                engines: engines
                    .try_into()
                    .map_err(|_| invalid("two endpoint work receipts required"))?,
                pgn: pgn_artifact.clone(),
            });
        }
        let core = PalsRunReceiptV4 {
            schema_version:4,
            domain: PALS_RECEIPT_V4_DOMAIN.into(),
            run_id: m.run_id.clone(),
            pair_id: m.pair_id.clone(),
            lock_sha256: lock.input.semantic_lock.canonical_sha256.clone(),
            training_executed: false,
            training: PalsObservedV3::Unknown,
            output_bytes: PalsObservedV3::Observed {value: observed_output_bytes(output,&assembly.work)?, method: "actual observed supervisor stream bytes plus retained receipt/evidence file lengths counted once per exact path, including separately read CPU/mock work envelopes; pinned inputs and separately capped runtime caches are outside this output-evidence scope".into()},
            policy_failures:BTreeSet::new(),
            wall_time_ms: output.wall_time_ms,
            cleanup_time_ms: cleanup,
            pair_eligible: failures.is_empty() && games.iter().all(|game|game.engines.iter().all(endpoint_admitted)),
            failures,
            games,
        };
        core.validate_against(&lock.input.semantic_lock)?;
        Ok(core)
    })();
    match result {
        Ok(core) => assembly.core = Some(core),
        Err(error) => assembly.assembly_error = Some(error.to_string()),
    }
    assembly
}

#[cfg(target_os = "linux")]
pub fn save_pals_core_assembly_v4(
    output: &PalsObservedPairOutputV4,
    assembly: &PalsCoreAssemblyV4,
) -> Result<ArtifactRef, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use std::io::Write;
    let owner = output.native.launch_owner();
    require(
        assembly.arena_launch_sha256 == owner.spec().sha256
            && assembly.native_receipt == output.native.receipt_artifact,
        "assembly provenance changed",
    )?;
    let bytes = serde_json::to_vec(assembly).map_err(|e| invalid(e.to_string()))?;
    require(
        bytes.len() <= V4_METADATA_CAP as usize,
        "V4 core assembly reservation exceeds 4MiB",
    )?;
    require(
        output
            .native
            .receipt
            .process
            .watched_artifact_bytes
            .is_some_and(|used| {
                used.checked_add(output.native.receipt_artifact.bytes)
                    .and_then(|n| n.checked_add(bytes.len() as u64))
                    .is_some_and(|n| n <= owner.spec().input.budget.max_artifact_bytes)
            }),
        "core assembly output budget unavailable",
    )?;
    // Original provider artifacts may already include their metadata. Reserve
    // all work metadata again conservatively rather than undercount CPU/mock
    // files absent from the generic NN-only artifact list.
    let required = output
        .native
        .receipt
        .artifacts
        .iter()
        .try_fold(output.native.receipt_artifact.bytes, |sum, a| {
            sum.checked_add(a.bytes)
        })
        .and_then(|sum| {
            assembly.work.iter().try_fold(sum, |n, w| {
                n.checked_add(w.startup_bytes)?
                    .checked_add(w.termination_bytes)
            })
        })
        .and_then(|sum| sum.checked_add(bytes.len() as u64));
    require(
        required.is_some_and(|n| n <= owner.spec().input.budget.max_output_bytes),
        "core assembly conservative evidence reservation exceeds output budget",
    )?;
    require(
        observed_output_bytes(output, &assembly.work)?
            .checked_add(bytes.len() as u64)
            .is_some_and(|n| n <= owner.spec().input.budget.max_output_bytes),
        "V4 observed streams and retained evidence plus core assembly exceed output budget",
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = owner
        .snapshot
        .directory
        .open_with("pals-core-assembly.v4.json", &options)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    Ok(ArtifactRef { path: format!("{}/pals-core-assembly.v4.json", owner.snapshot.output_directory), sha256: digest(&bytes), bytes: bytes.len() as u64,
        source: "PALS V4 actual arena work/clock/physical-lifetime assembly; raw native receipt retained".into(), license: "MIT execution evidence; external asset rights remain separate".into() })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> PalsArenaLaunchV4 {
        let old = super::super::tests::fixture();
        let mut base = match &old.semantic_lock.manifest.engines[0] {
            PalsEngineV3::Pals(e) => e.as_ref().clone(),
            _ => unreachable!(),
        };
        let policies = PalsPoliciesV4 {
            resolver: PalsResolverPolicyV4::OwnRaw,
            resolver_identity: PalsComponentV3 {
                semantic_id: PALS_V4_OWN_RAW_RESOLVER_SEMANTICS.into(),
                implementation_sha256: hex(
                    rz_search::pals::engine::compiled_resolver_implementation_sha256(),
                ),
                options: BTreeMap::new(),
            },
            resolver_semantics_sha256: PalsResolverPolicyV4::OwnRaw.expected_semantics_sha256(),
            checker: PalsCheckerPolicyV4::Own,
            recheck: PalsRecheckPolicyV4::Disabled,
            iterative_repair: PalsIterativeRepairPolicyV4::Disabled,
            cold_archive: PalsColdArchivePolicyV4::Disabled,
            paused_stack: PalsPausedStackPolicyV4::Disabled,
            cuda_warm: PalsCudaWarmPolicyV4::Disabled,
        };
        let identity = PalsPolicyIdentityV4::expected_for_policies(&policies).unwrap();
        base.search.semantic_id = identity.search_identity.clone();
        base.search.implementation_sha256 =
            hex(rz_search::pals::engine::compiled_search_implementation_sha256());
        base.model.input_schema = PalsModelProfile::LegacySummaryV1.encoding_schema().into();
        let endpoint = PalsEndpointV4 {
            base,
            policies,
            policy_identity: identity,
            model_v2: PalsModelSpecificationV4 {
                profile: PalsModelProfileV4::LegacySummaryV1,
                model_schema: "rovezero.pals-model.v1".into(),
                model_semantics: "rovezero.pals-model.v1".into(),
                encoding_schema: "rovezero.pals-board-records.v1".into(),
                model_identity: "explicit-legal-order-role-mock-v1".into(),
                model_semantics_sha256: "d".repeat(64),
                encoding_sha256: "e".repeat(64),
                max_records: 128,
                max_line_plies: 0,
                provenance: PalsModelProvenanceV4 {
                    purpose: PalsArtifactPurposeV4::ArenaCandidate,
                    training: PalsTrainingProvenanceV4::Untrained,
                },
            },
        };
        let own = match &old.semantic_lock.manifest.engines[1] {
            PalsEngineV3::OwnCpu(e) => e.clone(),
            _ => unreachable!(),
        };
        let mut manifest = PalsRunManifestV4 {
            schema_version: 4,
            run_id: old.semantic_lock.manifest.run_id.clone(),
            pair_id: old.semantic_lock.manifest.pair_id.clone(),
            contract_revision: PALS_V4_CONTRACT_REVISION.into(),
            rules_profile: old.semantic_lock.manifest.rules_profile.clone(),
            purpose: PalsRunPurposeV4::ArenaPilot,
            comparison: PalsComparisonV3::System,
            declared_changes: BTreeSet::new(),
            training_executed: false,
            engines: [
                PalsEngineV4::Pals(Box::new(endpoint)),
                PalsEngineV4::OwnCpu(own),
            ],
            resources: old.semantic_lock.manifest.resources.clone(),
            pilot: old.semantic_lock.manifest.pilot.clone(),
            output_bytes_max: 64 * 1024 * 1024,
        };
        manifest.declared_changes = manifest.actual_change_axes();
        let mut runner = old.runner;
        runner.binary.sha256 = PALS_CLOCK_REAP_RUNNER_SHA256.into();
        runner.binary.bytes = PALS_CLOCK_REAP_RUNNER_BYTES;
        let patch = runner.dirty_patch.as_mut().unwrap();
        patch.sha256 = PALS_CLOCK_REAP_PATCH_SHA256.into();
        patch.bytes = PALS_CLOCK_REAP_PATCH_BYTES;
        PalsArenaLaunchV4 {
            domain: PALS_ARENA_V4_DOMAIN.into(),
            semantic_lock: manifest.lock().unwrap(),
            runner,
            opening_artifact: old.opening_artifact,
            endpoints: old.endpoints,
            budget: old.budget,
        }
    }
    #[test]
    fn v4_executes_actual_explicit_flags_and_preserves_v3_lock_bytes() {
        let old = super::super::tests::fixture();
        let historical = serde_json::to_string(&old.semantic_lock).unwrap();
        let input = fixture();
        let lock = input.lock().unwrap();
        let args = &lock.endpoint_views[0].arguments;
        for flag in [
            "--pals-arena-wire=v4",
            "--pals-model-profile=legacy_summary_v1",
            "--pals-resolver-policy=own-raw-restricted",
            "--pals-cpu-resume=completed-iteration",
            "--pals-cuda-private-warm=false",
        ] {
            assert!(args.iter().any(|arg| arg == flag));
        }
        assert!(
            !args
                .iter()
                .any(|arg| arg.starts_with("--pals-post-repair-recheck="))
        );
        assert_eq!(lock.argument_limit(), 48);
        assert_eq!(
            LockedPalsArenaLaunchV4::from_json(&lock.to_json().unwrap())
                .unwrap()
                .sha256(),
            lock.sha256()
        );
        assert_ne!(lock.sha256(), old.lock().unwrap().sha256());
        assert_eq!(
            serde_json::to_string(&old.semantic_lock).unwrap(),
            historical
        );
    }
    #[test]
    fn v4_diagnostic_and_undeclared_enabled_owner_fail_before_preparation() {
        let mut input = fixture();
        input.semantic_lock.manifest.purpose = PalsRunPurposeV4::Diagnostic;
        input.semantic_lock = input.semantic_lock.manifest.lock().unwrap();
        assert!(input.lock().is_err());
        let mut input = fixture();
        let PalsEngineV4::Pals(e) = &mut input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model_v2.provenance.purpose = PalsArtifactPurposeV4::Diagnostic;
        assert!(input.semantic_lock.manifest.validate().is_err());
    }
    #[cfg(feature = "pals-collection")]
    #[test]
    fn native_v2_model_and_adapter_keep_independent_source_scopes() {
        let native = super::super::tests::native_fixture();
        let mut input = fixture();
        let PalsEngineV3::Pals(old) = &native.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        let PalsEngineV4::Pals(e) = &mut input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.base.model = old.model.clone();
        let config = PalsModelConfig::full_line_interaction_v2();
        e.base.model.input_schema = config.profile.encoding_schema().into();
        e.base.model.implementation_sha256 =
            hex(rz_eval::pals_model::compiled_model_boundary_sha256());
        e.model_v2.profile = PalsModelProfileV4::FullLineInteractionV2;
        e.model_v2.model_schema = "rovezero.pals-model.v2".into();
        e.model_v2.model_semantics = config.profile.model_semantics().into();
        e.model_v2.model_semantics_sha256 = hex(config.profile.registered_semantics_sha256());
        e.model_v2.encoding_schema = config.profile.encoding_schema().into();
        e.model_v2.encoding_sha256 =
            hex(rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(&config));
        e.model_v2.max_line_plies = 256;
        let PalsEndpointLaunchV3::OnnxCpu(mut n) = native.endpoints[0].clone() else {
            unreachable!()
        };
        n.adapter_source_sha256 = hex(rz_uci::pals_native::pals_native_source_digest());
        n.encoding_semantic_sha256 =
            hex(rz_uci::pals_native::pals_rules_encoding_semantic_digest_for_config(&config));
        n.graphs.retain(|g| g.role != "proposer");
        n.graphs[1].role = "shared_pc".into();
        n.graphs[1].file = "shared_pc_if.onnx".into();
        e.model_v2.model_identity = format!("pals-onnx-pc-fp32-{}", n.export.sha256);
        assert_ne!(e.base.model.implementation_sha256, n.adapter_source_sha256);
        input.endpoints[0] = PalsEndpointLaunchV3::OnnxCpu(n.clone());
        input.semantic_lock.manifest.declared_changes =
            input.semantic_lock.manifest.actual_change_axes();
        input.semantic_lock = input.semantic_lock.manifest.lock().unwrap();
        let lock = input.lock().unwrap();
        assert!(
            lock.endpoint_views[0]
                .arguments
                .iter()
                .any(|a| a == "--pals-model-profile=full_line_interaction_v2")
        );
        let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        let mut marker = PalsActualFollowupMarkerV4 {
            model_identity: e.model_v2.model_identity.clone(),
            model_profile: Some(config.profile.as_str().into()),
            model_semantics: config.profile.model_semantics().into(),
            model_semantics_sha256: Some(config.profile.registered_semantics_sha256()),
            model_implementation_sha256: Some(rz_eval::pals_model::compiled_model_boundary_sha256()),
            encoding_schema: Some(config.profile.encoding_schema().into()),
            model_configuration: Some(config),
            model_epoch: Some([0xaau8; 32]),
            resolver_policy: "own-raw-restricted".into(),
            cpu_resume: Some("completed-iteration".into()),
            post_repair_recheck: "disabled".into(),
            policy_identity: e.policy_identity.search_identity.clone(),
            policy_conditions_sha256: Some(e.policy_identity.conditions_sha256),
            search_implementation_sha256:
                rz_search::pals::engine::compiled_search_implementation_sha256(),
            resolver_implementation_sha256:
                rz_search::pals::engine::compiled_resolver_implementation_sha256(),
            resolver_semantics_sha256: rz_search::pals::engine::ResolverPolicy::OwnRawRestricted
                .semantics_sha256(),
            cuda_warm: false,
        };
        let PalsWeightIdentityV3::Untrained { artifact, .. } = &e.base.model.weights else {
            unreachable!()
        };
        let bytes: Vec<u8> = (0..64)
            .step_by(2)
            .map(|i| u8::from_str_radix(&artifact.sha256[i..i + 2], 16).unwrap())
            .collect();
        marker.model_epoch = Some(bytes.try_into().unwrap());
        let actual = serde_json::json!({"v4":marker});
        assert!(
            validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &actual, &actual)
                .is_ok()
        );
        let mut bad = actual.clone();
        bad["v4"]["model_implementation_sha256"] = serde_json::to_value([0_u8; 32]).unwrap();
        assert!(
            validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &bad, &bad).is_err()
        );
    }
    #[test]
    fn actual_marker_conditions_resolver_and_stack_drift_reject() {
        let lock = fixture().lock().unwrap();
        let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        let marker = PalsActualFollowupMarkerV4 {
            model_identity: "explicit-legal-order-role-mock-v1".into(),
            model_profile: None,
            model_semantics: "explicit-legal-order-role-mock-v1".into(),
            model_semantics_sha256: None,
            model_implementation_sha256: None,
            encoding_schema: None,
            model_configuration: None,
            model_epoch: None,
            resolver_policy: "own-raw-restricted".into(),
            cpu_resume: Some("completed-iteration".into()),
            post_repair_recheck: "disabled".into(),
            policy_identity: e.policy_identity.search_identity.clone(),
            policy_conditions_sha256: Some(e.policy_identity.conditions_sha256),
            search_implementation_sha256:
                rz_search::pals::engine::compiled_search_implementation_sha256(),
            resolver_implementation_sha256:
                rz_search::pals::engine::compiled_resolver_implementation_sha256(),
            resolver_semantics_sha256: rz_search::pals::engine::ResolverPolicy::OwnRawRestricted
                .semantics_sha256(),
            cuda_warm: false,
        };
        let good = serde_json::json!({"v4":marker});
        assert!(
            validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &good, &good).is_ok()
        );
        for field in ["resolver_policy", "cpu_resume", "policy_identity"] {
            let mut bad = good.clone();
            bad["v4"][field] = "different".into();
            assert!(
                validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &bad, &bad)
                    .is_err()
            );
        }
        let mut bad = good.clone();
        bad["v4"]["policy_conditions_sha256"] = serde_json::Value::Null;
        assert!(
            validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &bad, &bad).is_err()
        );
        for field in [
            "search_implementation_sha256",
            "resolver_implementation_sha256",
            "resolver_semantics_sha256",
        ] {
            let mut bad = good.clone();
            bad["v4"][field] = serde_json::to_value([0_u8; 32]).unwrap();
            assert!(
                validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &bad, &bad)
                    .is_err()
            );
        }
    }
}
