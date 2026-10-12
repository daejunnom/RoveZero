//! V2 controls and heterogeneous endpoints. V1 codecs are intentionally unchanged.
use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MANIFEST_V2_DOMAIN: &str = "rz-e01-model-endpoints-v2";
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonV2 {
    Fixture,
    Runtime,
    AdapterEquivalence,
    InternalWeights,
    InternalModel,
    InternalSearch,
    ExternalEngine,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChangeV2 {
    MeaningPreserving,
    Model,
    Search,
    ExternalComparison,
}

/// Semantic configuration, separated from the binary/adapter implementation.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfigurationV2 {
    pub adapter: String,
    pub architecture: String,
    pub encoding: String,
    pub policy_head: String,
    pub value_head: String,
    pub history_policy: String,
    pub weights: Vec<ArtifactRef>,
    pub backend: String,
    /// Runtime/provider configuration, excluding model-bound execution identities.
    pub backend_configuration_sha256: String,
    pub precision: String,
    pub batch_width: u32,
    pub options: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComponentConfigurationV2 {
    pub id: String,
    pub options: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "provider",
    content = "declaration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum RoveLaunchV2 {
    Lc0Cpu(CpuNativeLaunchSpecV1),
    Lc0CudaMaia(CudaNativeLaunchSpec<NativeCudaProfileV1>),
    Lc0Cuda(CudaNativeLaunchSpec<NativeCudaProfileV2>),
}
impl RoveLaunchV2 {
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Lc0Cpu(_) => "onnx-runtime-1.22.0-cpu",
            Self::Lc0CudaMaia(_) | Self::Lc0Cuda(_) => "onnx-runtime-1.22.0-cuda",
        }
    }

    /// V2 comparison identity. Legacy cache/attestation backend hashes include
    /// the model asset and must remain distinct from this configuration hash.
    pub fn backend_configuration_sha256(&self) -> Result<String, ManifestError> {
        let (target, core, profile, bundle) = match self {
            Self::Lc0Cpu(l) => (
                &l.target,
                l.artifact(NativeArtifactRole::OrtLibrary)?,
                serde_json::to_value(&l.profile),
                None,
            ),
            Self::Lc0CudaMaia(l) => (
                &l.target,
                l.artifact(NativeArtifactRole::OrtLibrary)?,
                serde_json::to_value(&l.profile),
                Some(l.cuda_bundle.canonical_sha256.as_str()),
            ),
            Self::Lc0Cuda(l) => (
                &l.target,
                l.artifact(NativeArtifactRole::OrtLibrary)?,
                serde_json::to_value(&l.profile.runtime),
                Some(l.cuda_bundle.canonical_sha256.as_str()),
            ),
        };
        let profile = profile.map_err(|e| ManifestError::Integrity(e.to_string()))?;
        let mut configuration: BTreeMap<String, serde_json::Value> =
            serde_json::from_value(profile).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        configuration.remove("expected_backend_sha256");
        configuration.remove("expected_encoding_sha256");
        let bytes = serde_json::to_vec(&(
            "rz-v2-backend-configuration/1",
            self.backend_name(),
            "ort-wrapper-2.0.0-rc.10",
            target,
            &core.sha256,
            core.bytes,
            bundle,
            configuration,
        ))
        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
        Ok(digest(&bytes))
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoveZeroEndpointV2 {
    pub id: String,
    pub role: NativeEngineRole,
    pub tool: ToolIdentity,
    pub adapter_implementation_sha256: String,
    pub model: ModelConfigurationV2,
    pub search: ComponentConfigurationV2,
    pub runtime: ComponentConfigurationV2,
    /// Public launch settings only; absent preserves the original V2 encoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<EngineEnvironmentV2>,
    /// A future unsupported model can be planned, but cannot be launched by this recipe.
    pub launch: Option<RoveLaunchV2>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExternalSourceV2 {
    pub url: String,
    pub commit: String,
    pub license: String,
}
/// A verified `env` executable clears inherited variables before exec. Settings
/// are public experiment inputs, never ambient credentials or env-file imports.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EngineEnvironmentV2 {
    pub launcher: ArtifactRef,
    pub variables: BTreeMap<String, String>,
}
impl EngineEnvironmentV2 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.launcher.validate()?;
        require_v2(
            self.launcher.bytes <= 4 * 1024 * 1024 && self.variables.len() <= 16,
            "environment launcher/variable budget exceeded",
        )?;
        let mut bytes = 0usize;
        for (name, value) in &self.variables {
            let upper = name.to_ascii_uppercase();
            let sensitive = [
                "SECRET",
                "TOKEN",
                "PASSWORD",
                "CREDENTIAL",
                "AUTHORIZATION",
                "PRIVATE_KEY",
                "API_KEY",
                "ACCESS_KEY",
            ]
            .iter()
            .any(|term| upper.contains(term));
            require_v2(
                !name.is_empty()
                    && name.len() <= 64
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                    && !name.as_bytes()[0].is_ascii_digit()
                    && !sensitive
                    && value.len() <= 4096
                    && name.len() + value.len() < 4096
                    && !value
                        .chars()
                        .any(|c| c.is_control() || matches!(c, '\"' | '\'' | '\\')),
                "environment requires bounded public variable names and literal values",
            )?;
            bytes += name.len() + value.len() + 2;
        }
        require_v2(bytes <= 8192, "environment combined byte budget exceeded")
    }
    pub fn validate_native(&self) -> Result<(), ManifestError> {
        self.validate()?;
        // The closed LC0 recipe owns provider/precision/session configuration.
        // Locale and allocator settings cannot override that semantic identity.
        require_v2(
            self.variables
                .iter()
                .all(|(name, value)| match name.as_str() {
                    "LANG" | "LC_ALL" | "TZ" => true,
                    "MALLOC_ARENA_MAX" => value.parse::<u32>().is_ok_and(|n| (1..=32).contains(&n)),
                    _ => false,
                }),
            "native environment cannot override model/provider/session configuration",
        )
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExternalUciEndpointV2 {
    pub id: String,
    pub role: NativeEngineRole,
    pub family: String,
    pub version: String,
    pub expected_uci_name: String,
    pub binary: ArtifactRef,
    pub source: Option<ExternalSourceV2>,
    pub arguments: Vec<String>,
    pub assets: Vec<ArtifactRef>,
    pub requested_options: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<EngineEnvironmentV2>,
    pub handshake_timeout_ms: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "endpoint",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum EngineEndpointV2 {
    RoveZero(Box<RoveZeroEndpointV2>),
    ExternalUci(Box<ExternalUciEndpointV2>),
}
impl EngineEndpointV2 {
    pub fn id(&self) -> &str {
        match self {
            Self::RoveZero(e) => &e.id,
            Self::ExternalUci(e) => &e.id,
        }
    }
    pub fn role(&self) -> NativeEngineRole {
        match self {
            Self::RoveZero(e) => e.role,
            Self::ExternalUci(e) => e.role,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResourcePolicyV2 {
    pub cpu_threads: u32,
    pub affinity: Vec<u32>,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
    pub swap_max_bytes: u64,
    pub gpu: Option<String>,
    pub gpu_vram_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RunManifestV2 {
    pub schema_version: u32,
    pub run_id: String,
    pub pair_id: String,
    pub comparison: ComparisonV2,
    pub change: ChangeV2,
    pub declared_changes: BTreeSet<String>,
    pub contract_revision: String,
    pub rules_profile: String,
    pub evaluation_policy: String,
    pub resources: ResourcePolicyV2,
    /// Omitted preserves historical V2 locks and ponder-off execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub match_execution: Option<MatchExecutionV1>,
    /// One explicit tree envelope for every RoveZero endpoint in this control.
    /// External UCI engines do not receive this native setting.
    pub rove_tree_max_edges: u32,
    pub engines: [EngineEndpointV2; 2],
    pub white_order: [NativeEngineRole; 2],
    pub opening: OpeningSpec,
    pub opening_artifact: ArtifactRef,
    pub runner: ToolIdentity,
    pub clock: NativeGameClockV3,
    pub max_plies: u32,
    pub seed: u64,
    pub timeouts: NativeTimeoutsV1,
    pub budget: NativeResourceBudgetV1,
}
fn require_v2(ok: bool, reason: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(format!("V2: {reason}")))
    }
}
fn text_v2(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}
fn sha_v2(value: &str, n: usize) -> bool {
    value.len() == n
        && value
            .bytes()
            .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
}
fn map_v2(map: &BTreeMap<String, String>) -> bool {
    map.len() <= 64
        && map
            .iter()
            .all(|(k, v)| text_v2(k) && v.len() <= 1024 && !v.chars().any(char::is_control))
}
fn weight_identity(m: &ModelConfigurationV2) -> Vec<(&str, u64)> {
    m.weights
        .iter()
        .map(|w| (w.sha256.as_str(), w.bytes))
        .collect()
}
fn model_changes(a: &ModelConfigurationV2, b: &ModelConfigurationV2) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    for (name, diff) in [
        ("adapter", a.adapter != b.adapter),
        ("architecture", a.architecture != b.architecture),
        ("encoding", a.encoding != b.encoding),
        ("policy_head", a.policy_head != b.policy_head),
        ("value_head", a.value_head != b.value_head),
        ("history_policy", a.history_policy != b.history_policy),
        ("weights", weight_identity(a) != weight_identity(b)),
        (
            "backend",
            a.backend != b.backend
                || a.backend_configuration_sha256 != b.backend_configuration_sha256,
        ),
        ("precision", a.precision != b.precision),
        ("batch_width", a.batch_width != b.batch_width),
        ("model_options", a.options != b.options),
    ] {
        if diff {
            result.insert(name.into());
        }
    }
    result
}
/// A launchable LC0 endpoint cannot claim metadata different from its closed recipe.
fn validate_executed_configuration(
    e: &RoveZeroEndpointV2,
    launch: &RoveLaunchV2,
) -> Result<(), ManifestError> {
    let (architecture, encoding, threads, workers, search) = match launch {
        RoveLaunchV2::Lc0Cpu(l) => (
            "maia1900",
            &l.profile.expected_encoding_sha256,
            l.profile.intra_threads,
            l.profile.search_workers,
            None,
        ),
        RoveLaunchV2::Lc0CudaMaia(l) => (
            "maia1900",
            &l.profile.expected_encoding_sha256,
            l.profile.intra_threads,
            l.profile.search_workers,
            None,
        ),
        RoveLaunchV2::Lc0Cuda(l) => (
            "bt4-it332",
            &l.profile.runtime.expected_encoding_sha256,
            l.profile.runtime.intra_threads,
            l.profile.runtime.search_workers,
            Some(l.profile.search),
        ),
    };
    let model = &e.model;
    require_v2(
        model.architecture == architecture
            && model.encoding == *encoding
            && model.backend == launch.backend_name()
            && model.backend_configuration_sha256 == launch.backend_configuration_sha256()?
            && model.policy_head == "lc0-policy-1858"
            && model.value_head == "stm-wdl"
            && model.history_policy == "no"
            && model.precision == "fp32"
            && model.batch_width == 1
            && model.options.is_empty(),
        "declared model differs from executed LC0 profile",
    )?;
    let search = search.unwrap_or(NativeCudaSearchV2 {
        simulations: 128,
        final_selection: NativeFinalSelectionV2::Visits,
        policy_temperature_milli: 1000,
        raw_cache: false,
    });
    let expected = BTreeMap::from([
        ("simulations".into(), search.simulations.to_string()),
        (
            "final_selection".into(),
            search.final_selection.cli().into(),
        ),
        (
            "policy_temperature_milli".into(),
            search.policy_temperature_milli.to_string(),
        ),
        ("raw_cache".into(), search.raw_cache.to_string()),
    ]);
    require_v2(
        e.search.id == "puct" && e.search.options == expected,
        "declared search differs from executed recipe",
    )?;
    require_v2(
        e.runtime.id == "single-worker-b1"
            && e.runtime.options
                == BTreeMap::from([
                    ("intra_threads".into(), threads.to_string()),
                    ("search_workers".into(), workers.to_string()),
                ]),
        "declared runtime differs from executed recipe",
    )
}
impl RunManifestV2 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let m: Self = decode_json(input)?;
        m.validate()?;
        Ok(m)
    }
    pub fn engine(&self, role: NativeEngineRole) -> Result<&EngineEndpointV2, ManifestError> {
        self.engines
            .iter()
            .find(|e| e.role() == role)
            .ok_or_else(|| ManifestError::Integrity("V2 missing role".into()))
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        require_v2(
            self.schema_version == 2 && self.contract_revision == "0.1",
            "schema/revision mismatch",
        )?;
        require_v2(
            [
                &self.run_id,
                &self.pair_id,
                &self.rules_profile,
                &self.evaluation_policy,
            ]
            .iter()
            .all(|v| text_v2(v)),
            "empty or unbounded identity",
        )?;
        require_v2(
            self.white_order[0] != self.white_order[1]
                && self.engines[0].role() != self.engines[1].role()
                && self.engines[0].id() != self.engines[1].id(),
            "duplicate roles/identities",
        )?;
        require_v2(
            self.clock.base_ms > 0
                && self.clock.base_ms <= 3_600_000
                && self.clock.increment_ms <= 60_000
                && (1..=4095).contains(&self.max_plies),
            "clock/plies outside bounded pilot",
        )?;
        let r = &self.resources;
        require_v2(
            (1..=4_194_304).contains(&self.rove_tree_max_edges),
            "RoveZero edge envelope invalid",
        )?;
        require_v2(
            (1..=2048).contains(&r.cpu_threads)
                && r.affinity.len() == r.cpu_threads as usize
                && r.affinity.iter().collect::<BTreeSet<_>>().len() == r.affinity.len()
                && r.memory_high_bytes > 0
                && r.memory_high_bytes <= r.memory_max_bytes
                && r.memory_max_bytes <= 128 * 1024 * 1024 * 1024
                && (r.gpu.is_some() == (r.gpu_vram_bytes > 0)),
            "resource envelope invalid",
        )?;
        if let Some(execution) = &self.match_execution {
            let plan = execution.plan()?;
            require_v2(
                self.engines.iter().all(|e| {
                    e.id()
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
                }),
                "match engine IDs must be single bounded protocol tokens",
            )?;
            if execution.ponder {
                let patch = self.runner.dirty_patch.as_ref().ok_or_else(|| {
                    ManifestError::Integrity(
                        "ponder requires the pinned Fastchess clock/ponder V2 patch".into(),
                    )
                })?;
                require_v2(
                    self.runner.dirty
                        && patch.sha256
                            == digest(include_bytes!(
                                "../../../experiments/baselines/fastchess-clock-ponder-v2.patch"
                            )),
                    "ponder runner patch identity differs",
                )?;
            }
            let available: BTreeSet<_> = execution
                .hardware
                .cpu_cores
                .iter()
                .flatten()
                .copied()
                .collect();
            require_v2(
                available == r.affinity.iter().copied().collect(),
                "match CPU inventory differs from inherited envelope",
            )?;
            require_v2(
                execution.hardware.gpus.is_empty() == r.gpu.is_none(),
                "match GPU inventory differs from envelope",
            )?;
            let reserved = plan.engines.iter().try_fold(0u64, |n, a| {
                a.gpu_memory_bytes
                    .checked_mul(a.gpu_ids.len() as u64)
                    .and_then(|m| n.checked_add(m))
            });
            require_v2(
                reserved.is_some_and(|n| n <= r.gpu_vram_bytes),
                "match GPU reservations exceed envelope",
            )?;
            for allocation in &plan.engines {
                let environment = match self.engine(allocation.role)? {
                    EngineEndpointV2::ExternalUci(e) => e.environment.as_ref(),
                    EngineEndpointV2::RoveZero(e) => e.environment.as_ref(),
                };
                require_v2(
                    environment.is_none_or(|e| {
                        !e.variables.contains_key("CUDA_VISIBLE_DEVICES")
                            && !e.variables.contains_key("CUDA_DEVICE_ORDER")
                    }),
                    "resource policy owns CUDA visibility; remove conflicting public environment variables",
                )?;
                if let EngineEndpointV2::RoveZero(e) = self.engine(allocation.role)? {
                    let Some(launch) = e.launch.as_ref() else {
                        continue;
                    };
                    let cpu = matches!(launch, RoveLaunchV2::Lc0Cpu(_));
                    require_v2(
                        cpu == (allocation.kind == EngineComputeKind::Cpu),
                        "native provider differs from compute kind",
                    )?;
                    // Current closed native profiles select logical CUDA device 0.
                    require_v2(
                        allocation.gpu_ids.len() <= 1,
                        "native recipe supports one assigned GPU",
                    )?;
                }
            }
        }
        require_v2(
            self.timeouts.runtime_ms > 0
                && self.timeouts.runtime_ms <= 3_600_000
                && (1..=120_000).contains(&self.timeouts.startup_ms)
                && (1..=30_000).contains(&self.timeouts.handshake_ms)
                && (1..=30_000).contains(&self.timeouts.drain_ms)
                && (1..=15_000).contains(&self.timeouts.shutdown_ms),
            "unbounded lifecycle",
        )?;
        let b = &self.budget;
        require_v2(
            b.max_input_bytes > 0
                && b.max_output_bytes > 0
                && b.max_runtime_bytes > 0
                && b.max_artifact_bytes <= 32 * 1024 * 1024 * 1024
                && b.max_input_bytes
                    .checked_add(b.max_output_bytes)
                    .and_then(|n| n.checked_add(b.max_runtime_bytes))
                    .is_some_and(|n| n <= b.max_artifact_bytes)
                && (3..=16).contains(&b.max_child_processes)
                && (1..=100_000).contains(&b.max_runtime_files)
                && (1..=16).contains(&b.max_runtime_depth)
                && b.address_space_per_process_bytes > 0,
            "artifact/process bounds invalid",
        )?;
        self.opening_artifact.validate()?;
        self.runner.binary.validate()?;
        require_v2(
            sha_v2(&self.runner.source_commit, 40),
            "runner source pin missing",
        )?;
        require_v2(
            text_v2(&self.opening.id)
                && self.opening.moves.len() <= 4095
                && self
                    .opening
                    .moves
                    .iter()
                    .all(|m| m.len() <= 5 && m.is_ascii())
                && (self.opening.initial == InitialPosition::Startpos)
                    == self.opening.fen.is_none(),
            "opening declaration invalid",
        )?;
        for engine in &self.engines {
            require_v2(
                text_v2(engine.id()) && !engine.id().chars().any(char::is_whitespace),
                "engine ID invalid",
            )?;
            match engine {
                EngineEndpointV2::RoveZero(e) => {
                    e.tool.binary.validate()?;
                    if let Some(environment) = &e.environment {
                        environment.validate()?;
                    }
                    require_v2(
                        sha_v2(&e.tool.source_commit, 40)
                            && sha_v2(&e.adapter_implementation_sha256, 64),
                        "RoveZero implementation pin missing",
                    )?;
                    let m = &e.model;
                    require_v2(
                        [
                            &m.adapter,
                            &m.architecture,
                            &m.encoding,
                            &m.policy_head,
                            &m.value_head,
                            &m.history_policy,
                            &m.backend,
                            &m.precision,
                            &e.search.id,
                            &e.runtime.id,
                        ]
                        .iter()
                        .all(|v| text_v2(v))
                            && sha_v2(&m.backend_configuration_sha256, 64)
                            && (1..=16).contains(&m.batch_width)
                            && m.weights.len() <= 8
                            && map_v2(&m.options)
                            && map_v2(&e.search.options)
                            && map_v2(&e.runtime.options),
                        "model configuration invalid",
                    )?;
                    for w in &m.weights {
                        w.validate()?;
                    }
                    if let Some(launch) = &e.launch {
                        if let Some(environment) = &e.environment {
                            environment.validate_native()?;
                        }
                        let (role, id, source, binary) = match launch {
                            RoveLaunchV2::Lc0Cpu(l) => {
                                l.validate()?;
                                (
                                    l.role,
                                    &l.engine_id,
                                    &l.source_commit,
                                    l.artifact(NativeArtifactRole::Binary)?,
                                )
                            }
                            RoveLaunchV2::Lc0CudaMaia(l) => {
                                l.validate()?;
                                (
                                    l.role,
                                    &l.engine_id,
                                    &l.source_commit,
                                    l.artifact(NativeArtifactRole::Binary)?,
                                )
                            }
                            RoveLaunchV2::Lc0Cuda(l) => {
                                l.validate()?;
                                (
                                    l.role,
                                    &l.engine_id,
                                    &l.source_commit,
                                    l.artifact(NativeArtifactRole::Binary)?,
                                )
                            }
                        };
                        require_v2(
                            role == e.role
                                && id == &e.id
                                && source == &e.tool.source_commit
                                && binary == &e.tool.binary
                                && m.adapter == "lc0",
                            "launch identity differs from endpoint",
                        )?;
                        validate_executed_configuration(e, launch)?;
                        let bindings = match launch {
                            RoveLaunchV2::Lc0Cpu(l) => &l.artifacts,
                            RoveLaunchV2::Lc0CudaMaia(l) => &l.artifacts,
                            RoveLaunchV2::Lc0Cuda(l) => &l.artifacts,
                        };
                        for role in [NativeArtifactRole::SourceWeights, NativeArtifactRole::Onnx] {
                            let asset =
                                bindings.iter().find(|a| a.role == role).ok_or_else(|| {
                                    ManifestError::Integrity("V2 model asset absent".into())
                                })?;
                            require_v2(
                                m.weights.iter().any(|w| {
                                    w.sha256 == asset.artifact.sha256
                                        && w.bytes == asset.artifact.bytes
                                }),
                                "model weights omit executed source/export",
                            )?;
                        }
                    }
                }
                EngineEndpointV2::ExternalUci(e) => {
                    e.binary.validate()?;
                    if let Some(environment) = &e.environment {
                        environment.validate()?;
                    }
                    require_v2(
                        text_v2(&e.family)
                            && text_v2(&e.version)
                            && text_v2(&e.expected_uci_name)
                            && e.arguments.len() <= 32
                            && e.environment.as_ref().is_none_or(|environment| {
                                e.arguments.len() + environment.variables.len() + 5 <= 32
                            })
                            && e.arguments.iter().all(|a| text_v2(a))
                            && e.assets.len() <= 16
                            && map_v2(&e.requested_options)
                            && (1..=30_000).contains(&e.handshake_timeout_ms),
                        "external endpoint invalid",
                    )?;
                    for a in &e.assets {
                        a.validate()?;
                    }
                    if let Some(s) = &e.source {
                        require_v2(
                            text_v2(&s.url) && sha_v2(&s.commit, 40) && text_v2(&s.license),
                            "external source invalid",
                        )?;
                    }
                }
            }
        }
        self.validate_control()?;
        self.unique_input_bytes().map(|_| ())
    }
    fn validate_control(&self) -> Result<(), ManifestError> {
        if self.comparison == ComparisonV2::ExternalEngine {
            return require_v2(
                self.engines
                    .iter()
                    .filter(|e| matches!(e, EngineEndpointV2::ExternalUci(_)))
                    .count()
                    == 1
                    && self.change == ChangeV2::ExternalComparison
                    && self.declared_changes == BTreeSet::from(["engine".into()]),
                "external control requires one RoveZero/one UCI engine and an engine comparison declaration",
            );
        }
        if self.comparison == ComparisonV2::Fixture {
            return Ok(());
        }
        self.validate_control_resources()?;
        let (EngineEndpointV2::RoveZero(a), EngineEndpointV2::RoveZero(b)) =
            (&self.engines[0], &self.engines[1])
        else {
            return require_v2(false, "internal comparison requires RoveZero endpoints");
        };
        let mut changes = model_changes(&a.model, &b.model);
        if a.adapter_implementation_sha256 != b.adapter_implementation_sha256 {
            changes.insert("adapter_implementation".into());
        }
        if a.runtime != b.runtime {
            changes.insert("runtime".into());
        }
        if a.search != b.search {
            changes.insert("search".into());
        }
        if a.environment != b.environment {
            changes.insert("environment".into());
        }
        require_v2(
            changes == self.declared_changes && !changes.is_empty(),
            "declared changes differ from actual comparison",
        )?;
        let allowed = match self.comparison {
            ComparisonV2::Runtime => {
                require_v2(self.change == ChangeV2::MeaningPreserving, "runtime is E")?;
                BTreeSet::from(["runtime".into(), "environment".into()])
            }
            ComparisonV2::AdapterEquivalence => {
                require_v2(
                    self.change == ChangeV2::MeaningPreserving,
                    "adapter equivalence is E",
                )?;
                BTreeSet::from(["adapter_implementation".into()])
            }
            ComparisonV2::InternalWeights => {
                require_v2(self.change == ChangeV2::Model, "weights are A")?;
                BTreeSet::from(["weights".into()])
            }
            ComparisonV2::InternalSearch => {
                require_v2(self.change == ChangeV2::Search, "search is S")?;
                BTreeSet::from(["search".into()])
            }
            ComparisonV2::InternalModel => {
                require_v2(
                    self.change == ChangeV2::Model
                        && !changes.contains("runtime")
                        && !changes.contains("search")
                        && changes
                            .iter()
                            .any(|c| c != "adapter_implementation" && c != "environment"),
                    "model configuration keeps search/runtime and changes a model component",
                )?;
                changes.clone()
            }
            _ => unreachable!(),
        };
        require_v2(changes.is_subset(&allowed), "confounded control")?;
        if self.comparison == ComparisonV2::InternalWeights {
            require_v2(
                a.tool == b.tool,
                "weights-only comparison requires the identical executable/build",
            )?;
        } else {
            require_v2(
                a.tool.target == b.tool.target
                    && a.tool.compiler == b.tool.compiler
                    && a.tool.build_mode == b.tool.build_mode
                    && a.tool.isa == b.tool.isa,
                "toolchain/build envelope differs",
            )?;
        }
        Ok(())
    }
    fn validate_control_resources(&self) -> Result<(), ManifestError> {
        if !matches!(
            self.comparison,
            ComparisonV2::Runtime
                | ComparisonV2::AdapterEquivalence
                | ComparisonV2::InternalWeights
                | ComparisonV2::InternalSearch
                | ComparisonV2::InternalModel
        ) {
            return Ok(());
        }
        let Some(execution) = &self.match_execution else {
            // Historical V2 uses the one inherited envelope for both engines;
            // omission remains ponder-off and keeps its original canonical bytes.
            return Ok(());
        };
        let plan = execution.plan()?;
        let allocations = self.engines.each_ref().map(|engine| {
            plan.engines
                .iter()
                .find(|allocation| allocation.role == engine.role())
        });
        let [Some(a), Some(b)] = allocations else {
            return require_v2(false, "controlled match allocation role missing");
        };
        // Compare available physical cores and SMT widths, not physical IDs or
        // logical-CPU counts alone. Shared cores remain part of the signature.
        let cpu_shape = |allocation: &EngineResourceAllocation,
                         other: &EngineResourceAllocation| {
            let mut shape: Vec<_> = execution
                .hardware
                .cpu_cores
                .iter()
                .filter(|core| core.iter().all(|id| allocation.cpu_ids.contains(id)))
                .map(|core| (core.len(), core.iter().any(|id| other.cpu_ids.contains(id))))
                .collect();
            shape.sort_unstable();
            shape
        };
        let gpu_shape = |allocation: &EngineResourceAllocation,
                         other: &EngineResourceAllocation|
         -> Result<Vec<_>, ManifestError> {
            let mut shape = Vec::new();
            for id in &allocation.gpu_ids {
                let gpu = execution
                    .hardware
                    .gpus
                    .iter()
                    .find(|gpu| &gpu.id == id)
                    .ok_or_else(|| {
                        ManifestError::Integrity("controlled match GPU inventory missing".into())
                    })?;
                let shared = other.gpu_ids.contains(id);
                require_v2(
                    shared || gpu.model.is_some(),
                    "controlled comparison on separate GPUs requires known host-checked models",
                )?;
                shape.push((gpu.model.as_deref(), gpu.vram_bytes, shared));
            }
            shape.sort_unstable();
            Ok(shape)
        };
        let model_backend_change = self.comparison == ComparisonV2::InternalModel
            && self.declared_changes.contains("backend")
            && matches!(
                (&self.engines[0], &self.engines[1]),
                (EngineEndpointV2::RoveZero(left), EngineEndpointV2::RoveZero(right))
                    if left.model.backend != right.model.backend
            );
        require_v2(
            (a.kind == b.kind || model_backend_change)
                && a.threads == b.threads
                && cpu_shape(a, b) == cpu_shape(b, a)
                && ((model_backend_change
                    && (a.kind == EngineComputeKind::Cpu || b.kind == EngineComputeKind::Cpu))
                    || (a.gpu_memory_bytes == b.gpu_memory_bytes
                        && gpu_shape(a, b)? == gpu_shape(b, a)?)),
            "controlled comparison requires equivalent CPU/core/SMT/thread and GPU/model/capacity/reservation/sharing allocations; only a declared model backend change permits unused GPU resources",
        )
        // Quota, memory, runner reservation, sharing policy and Ponder are common
        // manifest fields. Ponder applies to both engines; distinct device/core
        // identifiers alone do not change a controlled engine's resource budget.
    }
    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut assets = vec![&self.runner.binary, &self.opening_artifact];
        if let Some(execution) = &self.match_execution {
            assets.push(&execution.executor);
        }
        if self.match_execution.as_ref().is_some_and(|e| e.ponder) {
            if let Some(patch) = &self.runner.dirty_patch {
                assets.push(patch);
            }
        }
        for e in &self.engines {
            match e {
                EngineEndpointV2::ExternalUci(e) => {
                    assets.push(&e.binary);
                    assets.extend(e.assets.iter());
                    if let Some(environment) = &e.environment {
                        assets.push(&environment.launcher);
                    }
                }
                EngineEndpointV2::RoveZero(e) => {
                    assets.push(&e.tool.binary);
                    assets.extend(e.model.weights.iter());
                    if let Some(environment) = &e.environment {
                        assets.push(&environment.launcher);
                    }
                    if let Some(l) = &e.launch {
                        match l {
                            RoveLaunchV2::Lc0Cpu(l) => {
                                assets.extend(l.artifacts.iter().map(|a| &a.artifact))
                            }
                            RoveLaunchV2::Lc0CudaMaia(l) => {
                                assets.extend(l.artifacts.iter().map(|a| &a.artifact));
                                assets.push(&l.cuda_bundle.manifest);
                                assets.extend(l.cuda_bundle.files.iter().map(|a| &a.artifact));
                            }
                            RoveLaunchV2::Lc0Cuda(l) => {
                                assets.extend(l.artifacts.iter().map(|a| &a.artifact));
                                assets.push(&l.cuda_bundle.manifest);
                                assets.extend(l.cuda_bundle.files.iter().map(|a| &a.artifact));
                            }
                        }
                    }
                }
            }
        }
        assets
    }
    pub fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        let mut paths = BTreeMap::new();
        let mut total = 0u64;
        for a in self.declared_artifacts() {
            if let Some(old) = paths.insert(&a.path, a) {
                require_v2(old == a, "conflicting artifact metadata at one path")?;
            } else {
                total = total
                    .checked_add(a.bytes)
                    .ok_or_else(|| ManifestError::Integrity("V2 input overflow".into()))?;
            }
        }
        require_v2(
            total <= self.budget.max_input_bytes,
            "input budget exceeded",
        )?;
        Ok(total)
    }
    pub fn lock(mut self) -> Result<LockedManifestV2, ManifestError> {
        if let Some(execution) = &mut self.match_execution {
            execution.resolved_resources = Some(execution.plan()?);
        }
        self.validate()?;
        let sha256 = digest(&canonical_v2(&self)?);
        Ok(LockedManifestV2 {
            input: self,
            sha256,
        })
    }
}
fn canonical_v2(input: &RunManifestV2) -> Result<Vec<u8>, ManifestError> {
    let mut v = serde_json::json!({"domain":MANIFEST_V2_DOMAIN,"input":input});
    v.sort_all_objects();
    serde_json::to_vec(&v).map_err(|e| ManifestError::Integrity(e.to_string()))
}
#[derive(Clone, Debug)]
pub struct LockedManifestV2 {
    input: RunManifestV2,
    sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EnvelopeV2 {
    lock_version: u32,
    domain: String,
    execution_ready: bool,
    input_sha256: String,
    input: RunManifestV2,
}
impl LockedManifestV2 {
    pub fn input(&self) -> &RunManifestV2 {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        serde_json::to_string_pretty(&EnvelopeV2 {
            lock_version: 2,
            domain: MANIFEST_V2_DOMAIN.into(),
            execution_ready: false,
            input_sha256: self.sha256.clone(),
            input: self.input.clone(),
        })
        .map_err(|e| ManifestError::Integrity(e.to_string()))
    }
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let v: EnvelopeV2 = decode_json(input)?;
        require_v2(
            v.lock_version == 2 && v.domain == MANIFEST_V2_DOMAIN && !v.execution_ready,
            "wrong V2 lock domain/authority",
        )?;
        require_v2(
            v.input
                .match_execution
                .as_ref()
                .is_none_or(|e| e.resolved_resources.is_some()),
            "locked match is missing its resolved placement",
        )?;
        let locked = v.input.lock()?;
        require_v2(locked.sha256 == v.input_sha256, "V2 digest mismatch")?;
        Ok(locked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn environment() -> EngineEnvironmentV2 {
        EngineEnvironmentV2 {
            launcher: ArtifactRef {
                path: "tools/env".into(),
                sha256: "a".repeat(64),
                bytes: 1024,
                source: "https://www.gnu.org/software/coreutils/".into(),
                license: "Synthetic test declaration only".into(),
            },
            variables: BTreeMap::from([("MALLOC_ARENA_MAX".into(), "2".into())]),
        }
    }
    #[test]
    fn absent_environment_preserves_v2_encoding_and_explicit_null_lock_identity() {
        let lock = manifest(ComparisonV2::Fixture).lock().unwrap();
        let json = lock.to_json().unwrap();
        let mut legacy: serde_json::Value = serde_json::from_str(&json).unwrap();
        let engine = &mut legacy["input"]["engines"][0]["configuration"];
        assert!(engine.get("environment").is_none());
        engine["environment"] = serde_json::Value::Null;
        let reread = LockedManifestV2::from_json(&legacy.to_string()).unwrap();
        assert_eq!(reread.sha256(), lock.sha256());
        assert_eq!(reread.to_json().unwrap(), json);
    }
    #[test]
    fn v2_lock_freezes_automatic_placement_and_rejects_missing_or_modified_plan() {
        use crate::{EngineResourceRequest, MatchHardware, ResourceSharing};
        let mut input = manifest(ComparisonV2::Fixture);
        input.resources.cpu_threads = 3;
        input.resources.affinity = vec![0, 1, 2];
        input.match_execution = Some(MatchExecutionV1 {
            ponder: false,
            sharing: ResourceSharing::Isolated,
            hardware: MatchHardware {
                cpu_cores: vec![vec![0], vec![1], vec![2]],
                gpus: vec![],
            },
            engines: [NativeEngineRole::Baseline, NativeEngineRole::Candidate].map(|role| {
                EngineResourceRequest {
                    role,
                    kind: EngineComputeKind::Cpu,
                    cpu_ids: None,
                    threads: None,
                    gpu_ids: None,
                    gpu_memory_bytes: None,
                    cpu_weight: None,
                    gpu_weight: None,
                }
            }),
            executor: input.runner.binary.clone(),
            resolved_resources: None,
        });
        let expected = input.match_execution.as_ref().unwrap().plan().unwrap();
        let lock = input.lock().unwrap();
        assert_eq!(
            lock.input()
                .match_execution
                .as_ref()
                .unwrap()
                .resolved_resources,
            Some(expected)
        );
        let json = lock.to_json().unwrap();
        assert_eq!(
            LockedManifestV2::from_json(&json).unwrap().sha256(),
            lock.sha256()
        );
        let mut changed: serde_json::Value = serde_json::from_str(&json).unwrap();
        changed["input"]["match_execution"]["resolved_resources"]["engines"][0]["cpu_ids"] =
            serde_json::json!([2]);
        assert!(LockedManifestV2::from_json(&changed.to_string()).is_err());
        let mut removed: serde_json::Value = serde_json::from_str(&json).unwrap();
        removed["input"]["match_execution"]
            .as_object_mut()
            .unwrap()
            .remove("resolved_resources");
        assert!(LockedManifestV2::from_json(&removed.to_string()).is_err());
    }
    #[test]
    fn public_environment_budgets_and_native_semantics_are_checked() {
        let base = environment();
        base.validate_native().unwrap();
        for (name, value) in [
            ("9INVALID", "fixture"),
            ("HAS=EQUAL", "fixture"),
            ("CREDENTIAL", "fixture"),
            ("API_KEY", "fixture"),
            ("SAFE", "line\nfeed"),
            ("SAFE", "double\"quote"),
            ("SAFE", "single'quote"),
            ("SAFE", "back\\slash"),
        ] {
            let mut e = base.clone();
            e.variables = BTreeMap::from([(name.into(), value.into())]);
            assert!(e.validate().is_err());
        }
        let mut e = base.clone();
        e.variables = (0..17)
            .map(|i| (format!("PUBLIC_{i}"), "fixture".into()))
            .collect();
        assert!(e.validate().is_err());
        e.variables = BTreeMap::from([
            ("PUBLIC_A".into(), "x".repeat(4096)),
            ("PUBLIC_B".into(), "x".repeat(4096)),
        ]);
        assert!(e.validate().is_err());
        e.variables = BTreeMap::from([("CUDA_VISIBLE_DEVICES".into(), "0".into())]);
        e.validate().unwrap();
        assert!(e.validate_native().is_err());
        e.variables = BTreeMap::from([("MALLOC_ARENA_MAX".into(), "0".into())]);
        assert!(e.validate_native().is_err());
        e.variables = BTreeMap::from([("PUBLIC".into(), "x".repeat(4096))]);
        assert!(e.validate().is_err());
    }
    #[test]
    fn environment_is_an_explicit_runtime_change_and_cannot_confound_weights() {
        let mut m = manifest(ComparisonV2::Fixture);
        let base = match &m.engines[0] {
            EngineEndpointV2::RoveZero(e) => e.clone(),
            _ => unreachable!(),
        };
        let model = base.model.clone();
        candidate(&mut m).model = model;
        candidate(&mut m).environment = Some(environment());
        m.comparison = ComparisonV2::Runtime;
        m.change = ChangeV2::MeaningPreserving;
        m.declared_changes = BTreeSet::from(["environment".into()]);
        m.validate().unwrap();
        assert!(m.declared_artifacts().iter().any(|a| a.path == "tools/env"));
        m.comparison = ComparisonV2::InternalWeights;
        m.change = ChangeV2::Model;
        candidate(&mut m).model.weights[0].sha256 = "c".repeat(64);
        candidate(&mut m).model.weights[0].path = "new-env-comparison-weights.bin".into();
        m.declared_changes.insert("weights".into());
        assert!(m.validate().is_err());
        m.comparison = ComparisonV2::InternalModel;
        m.validate().unwrap();
    }
    fn manifest(comparison: ComparisonV2) -> RunManifestV2 {
        let old = RunManifest::from_json(include_str!(
            "../../../experiments/baselines/fixtures/e01-input.json"
        ))
        .unwrap();
        let make = |index: usize, role: NativeEngineRole| {
            let e = &old.engines[index];
            EngineEndpointV2::RoveZero(Box::new(RoveZeroEndpointV2 {
                id: e.id.clone(),
                role,
                environment: None,
                tool: old.protocol.runner.clone(),
                adapter_implementation_sha256: "a".repeat(64),
                model: ModelConfigurationV2 {
                    adapter: "lc0".into(),
                    architecture: "fixture-network".into(),
                    encoding: "fixture-input".into(),
                    policy_head: "fixture-policy".into(),
                    value_head: "stm-wdl".into(),
                    history_policy: "no".into(),
                    weights: vec![
                        e.weight
                            .clone()
                            .unwrap_or_else(|| old.protocol.runner.binary.clone()),
                    ],
                    backend: "fixture-cpu".into(),
                    backend_configuration_sha256: "c".repeat(64),
                    precision: "fp32".into(),
                    batch_width: 1,
                    options: BTreeMap::new(),
                },
                search: ComponentConfigurationV2 {
                    id: "puct".into(),
                    options: BTreeMap::new(),
                },
                runtime: ComponentConfigurationV2 {
                    id: "baseline".into(),
                    options: BTreeMap::new(),
                },
                launch: None,
            }))
        };
        RunManifestV2 {
            schema_version: 2,
            run_id: "adapter-fixture".into(),
            pair_id: "pair-1".into(),
            comparison,
            change: ChangeV2::Model,
            declared_changes: BTreeSet::new(),
            contract_revision: "0.1".into(),
            rules_profile: "standard".into(),
            evaluation_policy: "paired-no-adjudication".into(),
            resources: ResourcePolicyV2 {
                cpu_threads: 2,
                affinity: vec![0, 1],
                memory_high_bytes: 6 << 30,
                memory_max_bytes: 12 << 30,
                swap_max_bytes: 0,
                gpu: None,
                gpu_vram_bytes: 0,
            },
            match_execution: None,
            rove_tree_max_edges: 262_144,
            engines: [
                make(0, NativeEngineRole::Baseline),
                make(1, NativeEngineRole::Candidate),
            ],
            white_order: [NativeEngineRole::Baseline, NativeEngineRole::Candidate],
            opening: old.input.openings[0].clone(),
            opening_artifact: old.input.opening_artifact.clone(),
            runner: old.protocol.runner.clone(),
            clock: NativeGameClockV3 {
                base_ms: 120_000,
                increment_ms: 1_000,
            },
            max_plies: 256,
            seed: 1,
            timeouts: NativeTimeoutsV1 {
                startup_ms: 60_000,
                handshake_ms: 30_000,
                runtime_ms: 900_000,
                drain_ms: 15_000,
                shutdown_ms: 15_000,
            },
            budget: NativeResourceBudgetV1 {
                max_input_bytes: 1 << 20,
                max_output_bytes: 1 << 20,
                max_runtime_bytes: 1 << 20,
                max_artifact_bytes: 4 << 20,
                max_child_processes: 8,
                max_runtime_files: 1024,
                max_runtime_depth: 8,
                address_space_per_process_bytes: 16 << 30,
            },
        }
    }
    fn candidate(m: &mut RunManifestV2) -> &mut RoveZeroEndpointV2 {
        let EngineEndpointV2::RoveZero(e) = &mut m.engines[1] else {
            panic!()
        };
        e
    }
    const RESOURCE_CONTROLS: [ComparisonV2; 4] = [
        ComparisonV2::Runtime,
        ComparisonV2::AdapterEquivalence,
        ComparisonV2::InternalWeights,
        ComparisonV2::InternalSearch,
    ];
    fn resource_control(comparison: ComparisonV2) -> RunManifestV2 {
        let mut m = manifest(comparison);
        let EngineEndpointV2::RoveZero(base) = &m.engines[0] else {
            unreachable!()
        };
        candidate(&mut m).model = base.model.clone();
        let changed = match comparison {
            ComparisonV2::Runtime => {
                candidate(&mut m).runtime.id = "runtime-variant".into();
                m.change = ChangeV2::MeaningPreserving;
                "runtime"
            }
            ComparisonV2::AdapterEquivalence => {
                candidate(&mut m).adapter_implementation_sha256 = "b".repeat(64);
                m.change = ChangeV2::MeaningPreserving;
                "adapter_implementation"
            }
            ComparisonV2::InternalWeights => {
                candidate(&mut m).model.weights[0].path = "variant-weights.bin".into();
                candidate(&mut m).model.weights[0].sha256 = "b".repeat(64);
                m.change = ChangeV2::Model;
                "weights"
            }
            ComparisonV2::InternalSearch => {
                candidate(&mut m).search.id = "search-variant".into();
                m.change = ChangeV2::Search;
                "search"
            }
            _ => unreachable!(),
        };
        m.declared_changes = BTreeSet::from([changed.into()]);
        m.resources.cpu_threads = 3;
        m.resources.affinity = vec![0, 1, 2];
        m.match_execution = Some(MatchExecutionV1 {
            ponder: false,
            sharing: ResourceSharing::Isolated,
            hardware: MatchHardware {
                cpu_cores: vec![vec![0], vec![1], vec![2]],
                gpus: vec![],
            },
            engines: [NativeEngineRole::Baseline, NativeEngineRole::Candidate]
                .into_iter()
                .enumerate()
                .map(|(index, role)| EngineResourceRequest {
                    role,
                    kind: EngineComputeKind::Cpu,
                    cpu_ids: Some(vec![index as u32 + 1]),
                    threads: Some(1),
                    gpu_ids: None,
                    gpu_memory_bytes: None,
                    cpu_weight: None,
                    gpu_weight: None,
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            executor: m.runner.binary.clone(),
            resolved_resources: None,
        });
        m
    }
    fn resource_inventory_changed(m: &mut RunManifestV2) {
        let execution = m.match_execution.as_ref().unwrap();
        m.resources.affinity = execution
            .hardware
            .cpu_cores
            .iter()
            .flatten()
            .copied()
            .collect();
        m.resources.cpu_threads = m.resources.affinity.len() as u32;
        m.resources.gpu =
            (!execution.hardware.gpus.is_empty()).then(|| "synthetic-gpu-inventory".into());
        m.resources.gpu_vram_bytes = execution.hardware.gpus.iter().map(|g| g.vram_bytes).sum();
    }
    fn gpu_resource_control(comparison: ComparisonV2) -> RunManifestV2 {
        let mut m = resource_control(comparison);
        let execution = m.match_execution.as_mut().unwrap();
        execution.hardware.gpus = ["GPU-a", "GPU-b"]
            .map(|id| GpuCapacity {
                id: id.into(),
                model: Some("Synthetic GPU model".into()),
                vram_bytes: 8 << 30,
            })
            .to_vec();
        for (index, engine) in execution.engines.iter_mut().enumerate() {
            engine.kind = EngineComputeKind::Gpu;
            engine.gpu_ids = Some(vec![execution.hardware.gpus[index].id.clone()]);
            engine.gpu_memory_bytes = Some(2 << 30);
        }
        resource_inventory_changed(&mut m);
        m
    }
    #[test]
    fn internal_controls_reject_unequal_physical_cores_threads_and_smt_widths() {
        for comparison in RESOURCE_CONTROLS {
            let base = resource_control(comparison);
            base.validate().unwrap();
            let mut cores = base.clone();
            let execution = cores.match_execution.as_mut().unwrap();
            execution.hardware.cpu_cores.push(vec![3]);
            execution.engines[1].cpu_ids = Some(vec![2, 3]);
            // Same Threads is insufficient when one engine owns an extra core.
            resource_inventory_changed(&mut cores);
            assert!(cores.validate().is_err(), "{comparison:?}: core count");

            let mut threads = base.clone();
            let execution = threads.match_execution.as_mut().unwrap();
            execution.hardware.cpu_cores = vec![vec![0], vec![1, 2], vec![3, 4]];
            execution.engines[0].cpu_ids = Some(vec![1, 2]);
            execution.engines[1].cpu_ids = Some(vec![3, 4]);
            execution.engines[1].threads = Some(2);
            resource_inventory_changed(&mut threads);
            assert!(threads.validate().is_err(), "{comparison:?}: Threads");

            let mut smt = threads;
            let execution = smt.match_execution.as_mut().unwrap();
            execution.engines[1].threads = Some(1);
            execution.hardware.cpu_cores = vec![vec![0], vec![1, 2], vec![3], vec![4]];
            // Equal logical CPU counts still describe one SMT core versus two cores.
            assert!(
                smt.validate().is_err(),
                "{comparison:?}: physical SMT shape"
            );
        }
    }
    #[test]
    fn internal_controls_accept_equal_placements_with_different_physical_ids() {
        for comparison in RESOURCE_CONTROLS {
            let mut m = resource_control(comparison);
            let execution = m.match_execution.as_mut().unwrap();
            execution.hardware.cpu_cores = vec![vec![7], vec![13, 15], vec![21, 23]];
            execution.engines[0].cpu_ids = Some(vec![15, 13]);
            execution.engines[1].cpu_ids = Some(vec![21, 23]);
            execution.engines.swap(0, 1); // Roles, not array positions, own the budget.
            resource_inventory_changed(&mut m);
            m.validate().unwrap();
            let lock = m.lock().unwrap();
            assert_eq!(
                LockedManifestV2::from_json(&lock.to_json().unwrap())
                    .unwrap()
                    .sha256(),
                lock.sha256()
            );
        }
    }
    #[test]
    fn internal_controls_reject_gpu_kind_model_capacity_and_reservation_confounds() {
        for comparison in RESOURCE_CONTROLS {
            let base = gpu_resource_control(comparison);
            base.validate().unwrap();
            let mut kind = base.clone();
            kind.match_execution.as_mut().unwrap().engines[1].kind = EngineComputeKind::Hybrid;
            assert!(kind.validate().is_err(), "{comparison:?}: compute kind");
            let mut model = base.clone();
            model.match_execution.as_mut().unwrap().hardware.gpus[1].model =
                Some("Different GPU model".into());
            assert!(model.validate().is_err(), "{comparison:?}: GPU model");
            let mut capacity = base.clone();
            capacity.match_execution.as_mut().unwrap().hardware.gpus[1].vram_bytes = 12 << 30;
            resource_inventory_changed(&mut capacity);
            assert!(capacity.validate().is_err(), "{comparison:?}: GPU capacity");
            let mut reservation = base.clone();
            reservation.match_execution.as_mut().unwrap().engines[1].gpu_memory_bytes =
                Some(3 << 30);
            assert!(
                reservation.validate().is_err(),
                "{comparison:?}: GPU reservation"
            );
            let mut unknown = base;
            unknown.match_execution.as_mut().unwrap().hardware.gpus[1].model = None;
            assert!(
                unknown.validate().is_err(),
                "{comparison:?}: unknown separate GPU model"
            );
        }
    }
    #[test]
    fn internal_controls_accept_equivalent_gpu_ids_and_shared_legacy_device() {
        for comparison in RESOURCE_CONTROLS {
            let mut isolated = gpu_resource_control(comparison);
            let execution = isolated.match_execution.as_mut().unwrap();
            execution.engines[0].gpu_ids = Some(vec!["GPU-b".into()]);
            execution.engines[1].gpu_ids = Some(vec!["GPU-a".into()]);
            isolated.validate().unwrap();
            let mut shared = isolated;
            let execution = shared.match_execution.as_mut().unwrap();
            execution.sharing = ResourceSharing::Shared;
            execution.hardware.gpus.truncate(1);
            execution.hardware.gpus[0].model = None;
            for engine in &mut execution.engines {
                engine.gpu_ids = Some(vec!["GPU-a".into()]);
            }
            resource_inventory_changed(&mut shared);
            shared.validate().unwrap(); // Identical UUID identifies the same device.
            shared.match_execution.as_mut().unwrap().engines[1].gpu_memory_bytes = Some(3 << 30);
            assert!(
                shared.validate().is_err(),
                "{comparison:?}: shared reservation"
            );
        }
    }
    #[test]
    fn fixture_and_declared_internal_model_keep_heterogeneous_resource_semantics() {
        let mut m = gpu_resource_control(ComparisonV2::InternalWeights);
        m.match_execution.as_mut().unwrap().engines[1].kind = EngineComputeKind::Hybrid;
        m.comparison = ComparisonV2::Fixture;
        m.validate().unwrap();
        m.comparison = ComparisonV2::InternalModel;
        candidate(&mut m).model.backend = "declared-model-backend".into();
        m.declared_changes.insert("backend".into());
        m.validate().unwrap();
        m.declared_changes.remove("backend");
        assert!(m.validate().is_err());
        m.declared_changes.insert("backend".into());
        let execution = m.match_execution.as_mut().unwrap();
        execution.engines[0].kind = EngineComputeKind::Cpu;
        execution.engines[0].gpu_ids = Some(vec![]);
        execution.engines[0].gpu_memory_bytes = Some(0);
        m.validate().unwrap(); // Same CPU limit, one backend leaves the GPU unused.
        m.match_execution
            .as_mut()
            .unwrap()
            .hardware
            .cpu_cores
            .push(vec![3]);
        m.match_execution.as_mut().unwrap().engines[1].cpu_ids = Some(vec![2, 3]);
        resource_inventory_changed(&mut m);
        assert!(m.validate().is_err()); // Backend changes cannot confer extra CPU cores.
    }
    #[test]
    fn omitted_match_policy_and_gpu_model_keep_historical_canonical_encoding() {
        let old = resource_control(ComparisonV2::InternalWeights);
        let mut m = old;
        m.match_execution = None;
        let lock = m.lock().unwrap();
        let json = lock.to_json().unwrap();
        assert!(!json.contains("match_execution"));
        let reread = LockedManifestV2::from_json(&json).unwrap();
        assert_eq!(reread.sha256(), lock.sha256());
        assert_eq!(reread.to_json().unwrap(), json);
    }
    #[test]
    fn controlled_ponder_is_common_to_both_engines_and_locked_separately() {
        let m = resource_control(ComparisonV2::InternalWeights);
        let off_digest = m.clone().lock().unwrap().sha256().to_owned();
        let mut on = m;
        on.match_execution.as_mut().unwrap().ponder = true;
        let patch_bytes =
            include_bytes!("../../../experiments/baselines/fastchess-clock-ponder-v2.patch");
        let mut patch = on.runner.binary.clone();
        patch.path = "synthetic/fastchess-clock-ponder-v2.patch".into();
        patch.sha256 = digest(patch_bytes);
        patch.bytes = patch_bytes.len() as u64;
        on.runner.dirty = true;
        on.runner.dirty_patch = Some(patch);
        on.validate().unwrap();
        let on_lock = on.lock().unwrap();
        assert_ne!(on_lock.sha256(), off_digest);
        assert_eq!(
            LockedManifestV2::from_json(&on_lock.to_json().unwrap())
                .unwrap()
                .sha256(),
            on_lock.sha256()
        );
    }
    #[test]
    fn weights_comparison_keeps_backend_configuration_but_not_asset_bound_execution_hash() {
        let mut m = manifest(ComparisonV2::InternalWeights);
        m.declared_changes.insert("weights".into());
        for (index, endpoint) in m.engines.iter_mut().enumerate() {
            let EngineEndpointV2::RoveZero(e) = endpoint else {
                unreachable!()
            };
            let artifacts: Vec<_> = NativeArtifactRole::ALL
                .into_iter()
                .map(|role| {
                    let mut artifact = e.tool.binary.clone();
                    if role != NativeArtifactRole::Binary {
                        artifact.path = format!("synthetic/{index}/{role:?}");
                        artifact.bytes = 1;
                        artifact.sha256 = if index == 1
                            && matches!(
                                role,
                                NativeArtifactRole::SourceWeights | NativeArtifactRole::Onnx
                            ) {
                            "b".repeat(64)
                        } else {
                            "a".repeat(64)
                        };
                    }
                    NativeArtifactBinding { role, artifact }
                })
                .collect();
            let launch = RoveLaunchV2::Lc0Cpu(CpuNativeLaunchSpecV1 {
                role: e.role,
                engine_id: e.id.clone(),
                source_commit: e.tool.source_commit.clone(),
                target: e.tool.target.clone(),
                artifacts: artifacts.clone(),
                profile: NativeCpuProfileV1::fixed(
                    if index == 0 {
                        "c".repeat(64)
                    } else {
                        "e".repeat(64)
                    },
                    "d".repeat(64),
                ),
            });
            e.model.architecture = "maia1900".into();
            e.model.encoding = "d".repeat(64);
            e.model.policy_head = "lc0-policy-1858".into();
            e.model.backend = launch.backend_name().into();
            e.model.backend_configuration_sha256 = launch.backend_configuration_sha256().unwrap();
            e.model.weights = artifacts
                .into_iter()
                .filter_map(|a| {
                    matches!(
                        a.role,
                        NativeArtifactRole::SourceWeights | NativeArtifactRole::Onnx
                    )
                    .then_some(a.artifact)
                })
                .collect();
            e.search.options = BTreeMap::from([
                ("simulations".into(), "128".into()),
                ("final_selection".into(), "visits".into()),
                ("policy_temperature_milli".into(), "1000".into()),
                ("raw_cache".into(), "false".into()),
            ]);
            e.runtime.id = "single-worker-b1".into();
            e.runtime.options = BTreeMap::from([
                ("intra_threads".into(), "1".into()),
                ("search_workers".into(), "1".into()),
            ]);
            e.launch = Some(launch);
        }
        m.validate().unwrap();
        let e = candidate(&mut m);
        let Some(RoveLaunchV2::Lc0Cpu(l)) = &mut e.launch else {
            unreachable!()
        };
        l.artifacts
            .iter_mut()
            .find(|a| a.role == NativeArtifactRole::OrtLibrary)
            .unwrap()
            .artifact
            .sha256 = "f".repeat(64);
        // A forged unchanged declaration fails before comparison admission.
        assert!(m.validate().is_err());
        let e = candidate(&mut m);
        e.model.backend_configuration_sha256 = e
            .launch
            .as_ref()
            .unwrap()
            .backend_configuration_sha256()
            .unwrap();
        // Correctly declaring a library change still cannot be weights-only.
        m.declared_changes.insert("backend".into());
        assert!(m.validate().is_err());
        m.comparison = ComparisonV2::InternalModel;
        m.validate().unwrap();
    }
    #[test]
    fn weights_only_never_permits_encoding_head_or_runtime_changes() {
        let mut m = manifest(ComparisonV2::InternalWeights);
        // Shared fixture endpoints initially describe the same weights.
        let base = match &m.engines[0] {
            EngineEndpointV2::RoveZero(e) => e.model.clone(),
            _ => unreachable!(),
        };
        candidate(&mut m).model = base;
        candidate(&mut m).model.weights[0].path = "new-weights.bin".into();
        candidate(&mut m).model.weights[0].sha256 = "b".repeat(64);
        m.declared_changes.insert("weights".into());
        m.validate().unwrap();
        candidate(&mut m).model.policy_head = "candidate-head".into();
        m.declared_changes.insert("policy_head".into());
        assert!(m.validate().is_err());
    }
    #[test]
    fn adapter_equivalence_and_model_configuration_have_distinct_controls() {
        let mut m = manifest(ComparisonV2::AdapterEquivalence);
        m.change = ChangeV2::MeaningPreserving;
        let base = match &m.engines[0] {
            EngineEndpointV2::RoveZero(e) => e.model.clone(),
            _ => unreachable!(),
        };
        candidate(&mut m).model = base;
        candidate(&mut m).adapter_implementation_sha256 = "b".repeat(64);
        m.declared_changes.insert("adapter_implementation".into());
        m.validate().unwrap();
        candidate(&mut m).model.encoding = "entity-input".into();
        m.declared_changes.insert("encoding".into());
        assert!(m.validate().is_err());
        m.comparison = ComparisonV2::InternalModel;
        m.change = ChangeV2::Model;
        candidate(&mut m).model.policy_head = "candidate-head".into();
        candidate(&mut m).model.backend = "different-declared-backend".into();
        m.declared_changes
            .extend(["policy_head".into(), "backend".into()]);
        m.validate().unwrap();
        candidate(&mut m).search.id = "other-search".into();
        m.declared_changes.insert("search".into());
        assert!(m.validate().is_err());
    }
    #[test]
    fn v2_lock_cannot_be_read_as_v1_or_grant_execution() {
        let mut m = manifest(ComparisonV2::AdapterEquivalence);
        m.change = ChangeV2::MeaningPreserving;
        let base = match &m.engines[0] {
            EngineEndpointV2::RoveZero(e) => e.model.clone(),
            _ => unreachable!(),
        };
        candidate(&mut m).model = base;
        candidate(&mut m).adapter_implementation_sha256 = "b".repeat(64);
        m.declared_changes.insert("adapter_implementation".into());
        let lock = m.lock().unwrap();
        let json = lock.to_json().unwrap();
        assert_eq!(
            lock.sha256(),
            LockedManifestV2::from_json(&json).unwrap().sha256()
        );
        assert!(LockedManifest::from_json(&json).is_err());
        assert!(
            LockedManifestV2::from_json(
                &json.replace("\"execution_ready\": false", "\"execution_ready\": true")
            )
            .is_err()
        );
    }

    #[test]
    fn runtime_and_search_code_changes_keep_model_and_build_envelope() {
        for comparison in [ComparisonV2::Runtime, ComparisonV2::InternalSearch] {
            let mut m = manifest(comparison);
            let base = match &m.engines[0] {
                EngineEndpointV2::RoveZero(e) => e.model.clone(),
                _ => unreachable!(),
            };
            let b = candidate(&mut m);
            b.model = base;
            b.tool.source_commit = "b".repeat(40);
            b.tool.binary.path = "bin/runtime-or-search-variant".into();
            b.tool.binary.sha256 = "b".repeat(64);
            if comparison == ComparisonV2::Runtime {
                b.runtime.id = "declared-runtime-variant".into();
                m.change = ChangeV2::MeaningPreserving;
                m.declared_changes.insert("runtime".into());
            } else {
                b.search.id = "declared-search-variant".into();
                m.change = ChangeV2::Search;
                m.declared_changes.insert("search".into());
            }
            m.validate().unwrap();
            candidate(&mut m).tool.compiler = "another-compiler".into();
            assert!(m.validate().is_err());
        }
    }

    #[test]
    fn tree_envelope_is_explicit_finite_and_bound_to_the_v2_digest() {
        let mut m = manifest(ComparisonV2::InternalWeights);
        let EngineEndpointV2::RoveZero(base) = &m.engines[0] else {
            unreachable!()
        };
        let model = base.model.clone();
        candidate(&mut m).model = model;
        candidate(&mut m).model.weights[0].path = "synthetic/tree-candidate-weights".into();
        candidate(&mut m).model.weights[0].sha256 = "b".repeat(64);
        m.declared_changes.insert("weights".into());
        let initial_digest = m.clone().lock().unwrap().sha256().to_owned();
        let mut changed = m.clone();
        changed.rove_tree_max_edges = 100_000;
        assert_ne!(changed.lock().unwrap().sha256(), initial_digest);
        for invalid in [0, 4_194_305] {
            let mut changed = m.clone();
            changed.rove_tree_max_edges = invalid;
            assert!(changed.validate().is_err());
        }
    }

    #[test]
    fn external_endpoint_requires_no_internal_weights_or_neural_observations() {
        let mut m = manifest(ComparisonV2::ExternalEngine);
        m.change = ChangeV2::ExternalComparison;
        m.declared_changes = BTreeSet::from(["engine".into()]);
        m.engines[1] = EngineEndpointV2::ExternalUci(Box::new(ExternalUciEndpointV2 {
            id: "stockfish-19".into(),
            role: NativeEngineRole::Candidate,
            family: "stockfish".into(),
            version: "19".into(),
            expected_uci_name: "Stockfish 19".into(),
            binary: m.runner.binary.clone(),
            source: None,
            arguments: vec![],
            assets: vec![],
            requested_options: BTreeMap::from([("Threads".into(), "2".into())]),
            environment: None,
            handshake_timeout_ms: 30_000,
        }));
        let json = m.clone().lock().unwrap().to_json().unwrap();
        assert_eq!(
            LockedManifestV2::from_json(&json)
                .unwrap()
                .input()
                .comparison,
            ComparisonV2::ExternalEngine
        );
        let mut with_environment = m.clone();
        if let EngineEndpointV2::ExternalUci(endpoint) = &mut with_environment.engines[1] {
            endpoint.environment = Some(environment());
            endpoint.arguments = vec!["bounded".into(); 26];
        }
        with_environment.validate().unwrap();
        if let EngineEndpointV2::ExternalUci(endpoint) = &mut with_environment.engines[1] {
            endpoint.arguments.push("overflow".into());
        }
        assert!(with_environment.validate().is_err());
        m.change = ChangeV2::MeaningPreserving;
        assert!(m.validate().is_err());
        m.change = ChangeV2::ExternalComparison;
        m.timeouts.handshake_ms = 0;
        assert!(m.validate().is_err());
    }
}
