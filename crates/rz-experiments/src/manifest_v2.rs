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
        ("backend", a.backend != b.backend),
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
    let (architecture, backend, encoding, threads, workers, search) = match launch {
        RoveLaunchV2::Lc0Cpu(l) => (
            "maia1900",
            &l.profile.expected_backend_sha256,
            &l.profile.expected_encoding_sha256,
            l.profile.intra_threads,
            l.profile.search_workers,
            None,
        ),
        RoveLaunchV2::Lc0CudaMaia(l) => (
            "maia1900",
            &l.profile.expected_backend_sha256,
            &l.profile.expected_encoding_sha256,
            l.profile.intra_threads,
            l.profile.search_workers,
            None,
        ),
        RoveLaunchV2::Lc0Cuda(l) => (
            "bt4-it332",
            &l.profile.runtime.expected_backend_sha256,
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
            && model.backend == *backend
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
            (1..=64).contains(&r.cpu_threads)
                && r.affinity.len() == r.cpu_threads as usize
                && r.affinity.iter().collect::<BTreeSet<_>>().len() == r.affinity.len()
                && r.memory_high_bytes > 0
                && r.memory_high_bytes <= r.memory_max_bytes
                && r.memory_max_bytes <= 128 * 1024 * 1024 * 1024
                && (r.gpu.is_some() == (r.gpu_vram_bytes > 0)),
            "resource envelope invalid",
        )?;
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
                    require_v2(
                        text_v2(&e.family)
                            && text_v2(&e.version)
                            && text_v2(&e.expected_uci_name)
                            && e.arguments.len() <= 32
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
        require_v2(
            changes == self.declared_changes && !changes.is_empty(),
            "declared changes differ from actual comparison",
        )?;
        let allowed = match self.comparison {
            ComparisonV2::Runtime => {
                require_v2(self.change == ChangeV2::MeaningPreserving, "runtime is E")?;
                BTreeSet::from(["runtime".into()])
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
                        && changes.iter().any(|c| c != "adapter_implementation"),
                    "model configuration keeps search/runtime and changes a model component",
                )?;
                changes.clone()
            }
            _ => unreachable!(),
        };
        require_v2(changes.is_subset(&allowed), "confounded control")?;
        if !matches!(
            self.comparison,
            ComparisonV2::AdapterEquivalence | ComparisonV2::InternalModel
        ) {
            require_v2(
                a.tool == b.tool,
                "binary/build inputs differ outside adapter/model comparison",
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
    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut assets = vec![&self.runner.binary, &self.opening_artifact];
        for e in &self.engines {
            match e {
                EngineEndpointV2::ExternalUci(e) => {
                    assets.push(&e.binary);
                    assets.extend(e.assets.iter());
                }
                EngineEndpointV2::RoveZero(e) => {
                    assets.push(&e.tool.binary);
                    assets.extend(e.model.weights.iter());
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
    pub fn lock(self) -> Result<LockedManifestV2, ManifestError> {
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
        let locked = v.input.lock()?;
        require_v2(locked.sha256 == v.input_sha256, "V2 digest mismatch")?;
        Ok(locked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        m.change = ChangeV2::MeaningPreserving;
        assert!(m.validate().is_err());
        m.change = ChangeV2::ExternalComparison;
        m.timeouts.handshake_ms = 0;
        assert!(m.validate().is_err());
    }
}
