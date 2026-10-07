//! PALS V3 launch envelope over the existing pinned arena lifecycle.
//!
//! V2 UCI endpoint values below are executable views, never a conversion of the
//! V3 manifest or its digest. Native P/C evidence, UCI observations, Rules PGN
//! replay, clocks, process exit and input retirement remain separate gates.
use crate::{
    ArenaError, NativeEngineView, NativeLaunchDeclaration, NativeLaunchOwner, NativePairView,
    NativeProviderDeclaration,
};
use rz_experiments::{
    ArtifactRef, EngineEnvironmentV2, ExternalSourceV2, ExternalUciEndpointV2, HistoryCompleteness,
    InitialPosition, ManifestError, NativeEngineRole, NativeGameClockV3, NativePairClock,
    NativeResourceBudgetV1, NativeTimeoutsV1, OpeningSpec, PALS_RECEIPT_V3_DOMAIN,
    PalsEndpointReceiptV3, PalsEngineV3, PalsGameReceiptV3, PalsInputLockV3, PalsModelBackendV3,
    PalsObservedV3, PalsOptionReceiptV3, PalsPhysicalStateV3, PalsPrecisionV3, PalsResultV3,
    PalsRunFailureV3, PalsRunReceiptV3, PalsTerminationV3, PalsWeightIdentityV3, ToolIdentity,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    io::Read,
    path::{Component, Path},
};

pub const PALS_ARENA_V3_DOMAIN: &str = "rz-pals-arena-launch-v3/1";
pub const PALS_NATIVE_STARTUP_V3_DOMAIN: &str = "rz-pals-native-startup-v3/1";
pub const PALS_NATIVE_TERMINATION_V3_DOMAIN: &str = "rz-pals-native-termination-v3/1";
pub const PALS_SEARCH_WORK_STARTUP_V3_DOMAIN: &str = "rz-pals-search-work-startup-v3/1";
pub const PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN: &str = "rz-pals-search-work-termination-v3/1";
pub const PALS_CLOCK_REAP_PATCH_SHA256: &str =
    "23bc4abfa79fcde2b07bebcd70dba2adc112c6152ae3346188f693dc1b5e9ec5";
pub const PALS_CLOCK_REAP_PATCH_BYTES: u64 = 9907;
pub const PALS_CLOCK_REAP_RUNNER_SHA256: &str =
    "29e89312bc4ec16a32ec185d8b53c60b0cdbad17eaac294f987491e36b8d29ff";
pub const PALS_CLOCK_REAP_RUNNER_BYTES: u64 = 2466608;
const MAX_JSON_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsRunnerStatusPatchV3 {
    LegacyClockOnly,
    ClockAndReapStatus,
}

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("PALS arena V3: {}", reason.into()))
}
fn require(ok: bool, reason: &str) -> Result<(), ArenaError> {
    if ok { Ok(()) } else { Err(invalid(reason)) }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn json(bytes: &[u8]) -> Result<serde_json::Value, ArenaError> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("JSON evidence is not UTF-8"))?;
    crate::decode_json(text)
}
fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        && matches!(
            Path::new(value).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
}
fn role_index(role: NativeEngineRole) -> usize {
    match role {
        NativeEngineRole::Baseline => 0,
        NativeEngineRole::Candidate => 1,
    }
}
/// Resolve the parent user's existing cache before the engine's cleared child
/// environment. A cache open never supplies NN-ready or placement evidence.
fn pals_shared_runtime_argument() -> Result<OsString, ArenaError> {
    let cache = rz_eval::runtime_pin::RuntimeCache::for_user()
        .map_err(|e| invalid(format!("shared runtime cache preparation:{:?}", e.kind)))?;
    let path = cache
        .root()
        .to_str()
        .ok_or_else(|| invalid("shared runtime root is not UTF-8"))?;
    Ok(format!("--pals-runtime-cache-root={path}").into())
}

/// These are actual accepted CLI limits, separately pinned from declarations.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsSearchLaunchV3 {
    pub max_rounds: u64,
    pub max_cpu_nodes: u64,
    pub cpu_depth: u16,
    pub max_situations: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsGraphAssetV3 {
    /// Original single filename in the hashed export descriptor.
    pub file: String,
    pub role: String,
    pub artifact: ArtifactRef,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsOnnxLaunchV3 {
    pub export: ArtifactRef,
    pub export_file: String,
    pub graphs: Vec<PalsGraphAssetV3>,
    pub runtime: ArtifactRef,
    /// Semantic and source identities expected from the actual native producer.
    pub encoding_semantic_sha256: String,
    pub adapter_source_sha256: String,
    pub search: PalsSearchLaunchV3,
}
/// Explicit CUDA execution identity; the session arena is a declaration, never
/// an observed device peak. Host K/V and physical B1 remain the first recipe.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsOnnxCudaLaunchV3 {
    pub model: PalsOnnxLaunchV3,
    pub cuda_bundle: rz_experiments::CudaBundleBindingV1,
    pub device_id: i32,
    pub session_arena_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(
    tag = "recipe",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsEndpointLaunchV3 {
    LegalOrderMock(PalsSearchLaunchV3),
    OnnxCpu(PalsOnnxLaunchV3),
    OnnxCuda(PalsOnnxCudaLaunchV3),
    OwnCpu {
        max_depth: u16,
        max_nodes: u64,
        tt_entries: u32,
    },
    ReferenceUci {
        expected_uci_name: String,
        arguments: Vec<String>,
        environment: Option<EngineEnvironmentV2>,
    },
}
impl PalsEndpointLaunchV3 {
    fn native_model(&self) -> Option<&PalsOnnxLaunchV3> {
        match self {
            Self::OnnxCpu(model) => Some(model),
            Self::OnnxCuda(cuda) => Some(&cuda.model),
            _ => None,
        }
    }
    fn cuda_model(&self) -> Option<&PalsOnnxCudaLaunchV3> {
        if let Self::OnnxCuda(cuda) = self {
            Some(cuda)
        } else {
            None
        }
    }
}

/// A distinct arena envelope pins runner, opening, executable recipes and I/O
/// budgets. A V3 semantic lock alone does not authorize a child process.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsArenaLaunchV3 {
    pub domain: String,
    pub semantic_lock: PalsInputLockV3,
    pub runner: ToolIdentity,
    pub opening_artifact: ArtifactRef,
    pub endpoints: [PalsEndpointLaunchV3; 2],
    pub budget: NativeResourceBudgetV1,
}
#[derive(Clone, Debug)]
pub struct LockedPalsArenaLaunchV3 {
    input: PalsArenaLaunchV3,
    sha256: String,
    opening: OpeningSpec,
    endpoint_views: [ExternalUciEndpointV2; 2],
}
impl PalsArenaLaunchV3 {
    pub fn from_json(text: &str) -> Result<Self, ArenaError> {
        require(text.len() <= MAX_JSON_BYTES, "launch JSON budget exceeded")?;
        let input: Self = crate::decode_json(text)?;
        input.validate()?;
        Ok(input)
    }
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.semantic_lock.verify()?;
        require(
            self.domain == PALS_ARENA_V3_DOMAIN,
            "wrong arena launch domain",
        )?;
        let pilot = &self.semantic_lock.manifest.pilot;
        require(
            pilot.max_plies.is_multiple_of(2),
            "Fastchess max-plies must be even",
        )?;
        require(
            self.runner.source_url == crate::FASTCHESS_SOURCE_URL
                && self.runner.source_commit == crate::FASTCHESS_SOURCE_COMMIT
                && self.runner.version == crate::FASTCHESS_VERSION
                && self.runner.dirty
                && self.runner_status_patch().is_ok(),
            "clock-audited pinned Fastchess and an exact registered patch are required",
        )?;
        self.runner.binary.validate()?;
        self.runner
            .dirty_patch
            .as_ref()
            .ok_or_else(|| invalid("missing runner patch"))?
            .validate()?;
        self.opening_artifact.validate()?;
        let b = self.budget;
        let unique = self.unique_input_bytes()?;
        let cuda_count = self
            .endpoints
            .iter()
            .filter(|e| e.cuda_model().is_some())
            .count() as u64;
        let cpu_native_count = self
            .endpoints
            .iter()
            .filter(|e| matches!(e, PalsEndpointLaunchV3::OnnxCpu(_)))
            .count() as u64;
        // Historical PALS CUDA locks with a finite per-role cache reservation
        // remain readable. New children share the bounded, content-addressed
        // user runtime cache outside this attempt; no per-run bundle copy is
        // included in the owned runtime tree.
        let runtime_ceiling = if cuda_count == 0 {
            2 * 1024 * 1024 * 1024
        } else {
            (4 * cuda_count + cpu_native_count) * 1024 * 1024 * 1024 + 64 * 1024 * 1024
        };
        require(
            b.max_input_bytes > 0
                && unique <= b.max_input_bytes
                && b.max_output_bytes > 0
                && b.max_output_bytes <= 64 * 1024 * 1024
                && b.max_runtime_bytes > 0
                && b.max_runtime_bytes <= runtime_ceiling
                && b.max_child_processes >= 3
                && b.max_child_processes <= 16
                && b.max_runtime_files >= 16
                && b.max_runtime_files <= 4096
                && (2..=8).contains(&b.max_runtime_depth)
                && b.address_space_per_process_bytes == 0,
            "finite I/O/process budgets required; RAM is enforced by the inherited cgroup",
        )?;
        require(
            unique
                .checked_add(b.max_output_bytes)
                .and_then(|n| n.checked_add(b.max_runtime_bytes))
                .is_some_and(|n| n <= b.max_artifact_bytes),
            "artifact budget does not cover inputs and bounded output",
        )?;
        let resources = &self.semantic_lock.manifest.resources;
        require(
            resources[0].cpu_affinity == resources[1].cpu_affinity
                && resources[0].memory_high_bytes == resources[1].memory_high_bytes
                && resources[0].memory_max_bytes == resources[1].memory_max_bytes
                && resources[0].swap_max_bytes == resources[1].swap_max_bytes,
            "this executor supports one identical inherited CPU/memory policy for the two sequential engines",
        )?;
        for (i, resource) in resources.iter().enumerate() {
            self.endpoint(i)?;
            if let PalsEngineV3::Pals(e) = &self.semantic_lock.manifest.engines[i] {
                require(
                    e.pools.host_bytes == resource.memory_max_bytes,
                    "PALS pool host ceiling differs from verified memory cgroup",
                )?;
                if let Some(cuda) = self.endpoints[i].cuda_model() {
                    let declared_arenas = cuda
                        .session_arena_bytes
                        .checked_mul(cuda.model.graphs.len() as u64)
                        .ok_or_else(|| invalid("CUDA session declaration overflow"))?;
                    require(
                        resource.requested_gpu.is_some()
                            && e.pools.device_bytes > 0
                            && e.pools.device_bytes == resource.device_allocation_max_bytes
                            && declared_arenas < e.pools.device_bytes
                            && e.pools.device_bytes <= 6 * 1024 * 1024 * 1024,
                        "CUDA explicit device budget must cover all declared session arenas; observed peak remains unknown",
                    )?;
                } else {
                    require(
                        e.pools.device_bytes == 0
                            && resource.requested_gpu.is_none()
                            && resource.device_allocation_max_bytes == 0,
                        "closed CPU/mock recipe must not declare GPU allocation",
                    )?;
                }
            }
        }
        let cuda_bundles: Vec<_> = self
            .endpoints
            .iter()
            .filter_map(PalsEndpointLaunchV3::cuda_model)
            .map(|c| &c.cuda_bundle)
            .collect();
        require(
            cuda_bundles.windows(2).all(|pair| pair[0] == pair[1]),
            "first CUDA executor requires one exact shared runtime bundle identity for both roles",
        )?;
        let mut names = BTreeMap::new();
        let mut destinations = BTreeMap::new();
        for (artifact, target) in self.named_assets() {
            if let Some(prior) = destinations.insert(&artifact.path, target.clone()) {
                require(
                    prior == target,
                    "one source artifact cannot be placed in multiple export namespaces by this executor",
                )?;
            }
            if let Some(prior) = names.insert(target, artifact) {
                require(prior == artifact, "snapshot destination collision")?;
            }
        }
        Ok(())
    }
    pub fn lock(&self) -> Result<LockedPalsArenaLaunchV3, ArenaError> {
        self.validate()?;
        Ok(LockedPalsArenaLaunchV3 {
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
                    matches!(engine, PalsEngineV3::Pals(e) if e.model.frozen_epoch == 1),
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
                PalsEngineV3::Pals(e) => {
                    result.push(&e.binary);
                    if let PalsWeightIdentityV3::Untrained { artifact, .. }
                    | PalsWeightIdentityV3::Trained { artifact, .. } = &e.model.weights
                    {
                        result.push(artifact);
                    }
                }
                PalsEngineV3::OwnCpu(e) => result.push(&e.binary),
                PalsEngineV3::ReferenceUci(e) => {
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
        let m = &self.semantic_lock.manifest;
        let role = if i == 0 {
            NativeEngineRole::Baseline
        } else {
            NativeEngineRole::Candidate
        };
        let (binary, family, version, uci_name, source, arguments, assets, environment) = match (
            &m.engines[i],
            &self.endpoints[i],
        ) {
            (PalsEngineV3::Pals(e), PalsEndpointLaunchV3::LegalOrderMock(s)) => {
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
                validate_pals_search(e, s, m.pilot.wall_time_max_ms)?;
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
                PalsEngineV3::Pals(e),
                recipe @ (PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)),
            ) => {
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
                        && e.model.input_schema == "rz-pals-rules-fields-v1"
                        && e.model.policy_head == "candidate-policy/1"
                        && e.model.value_head == "stm-wdl/1"
                        && e.model.implementation_sha256 == n.adapter_source_sha256,
                    "native model topology/input/head/adapter declaration differs from supported recipe",
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
                validate_pals_search(e, &n.search, m.pilot.wall_time_max_ms)?;
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
                }
                (
                    &e.binary,
                    "rovezero-pals",
                    "pals/0.1",
                    format!(
                        "RoveZero PALS P/C ONNX {} + own CPU_R",
                        if cuda.is_some() { "CUDA" } else { "CPU" }
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
                PalsEngineV3::OwnCpu(e),
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
                PalsEngineV3::ReferenceUci(e),
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
        require(
            arguments.len() <= 32
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
            requested_options: m.engines[i].requested_options().clone(),
            environment,
            handshake_timeout_ms: m.pilot.handshake_max_ms,
        })
    }
}
fn validate_cpu_profile(cpu: &rz_experiments::PalsOwnCpuV3, wall: u64) -> Result<(), ArenaError> {
    require(
        cpu.core.semantic_id == "rz-cpu-pvs/0.1"
            && cpu.runtime_profile.semantic_id == "cpu-plan-assisted-conservative-v1"
            && cpu.evaluation.semantic_id == "bootstrap-material-pst-v1"
            && cpu.max_task_ms >= wall
            && cpu.core.options.is_empty()
            && cpu.runtime_profile.options.is_empty()
            && cpu.evaluation.options.is_empty(),
        "first executable recipe is PlanAssisted/bootstrap; task time declaration must cover its inherited finite deadline",
    )
}
fn validate_pals_search(
    e: &rz_experiments::PalsEndpointV3,
    s: &PalsSearchLaunchV3,
    wall: u64,
) -> Result<(), ArenaError> {
    validate_cpu_profile(&e.cpu, wall)?;
    require(
        (1..=1_000_000).contains(&s.max_rounds)
            && s.max_cpu_nodes > 0
            && s.cpu_depth > 0
            && s.cpu_depth <= 16
            && u32::from(s.cpu_depth) <= e.cpu.max_depth
            && (257..=65_536).contains(&s.max_situations)
            && e.cpu.max_nodes_per_task >= 4096,
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
        rz_search::cpu::CpuConfig::default()
            .tt_allocation_bytes()
            .map_err(|e| invalid(e.to_string()))?
            <= e.cpu.max_tt_bytes,
        "PALS own CPU default TT slot allocation exceeds declared byte limit",
    )?;
    require(
        e.search.semantic_id == "pals"
            && e.runtime.semantic_id == "single-owner/1"
            && e.search.options.is_empty()
            && e.runtime.options.is_empty(),
        "unsupported PALS search/runtime semantic option override",
    )?;
    require(
        e.requested_options.is_empty(),
        "PALS fixed executable recipe accepts no unverified UCI overrides",
    )
}
fn require_public_artifact(a: &ArtifactRef) -> Result<(), ArenaError> {
    for part in a.path.split('/') {
        let p = part.to_ascii_lowercase();
        require(
            !p.starts_with(".env")
                && !p.ends_with(".key")
                && !p.ends_with(".pem")
                && !p.starts_with("id_rsa")
                && !p.starts_with("id_ed25519")
                && !p.contains("credential")
                && !p.contains("service-account")
                && !p.contains("service_account"),
            "secret artifacts cannot be launch inputs",
        )?;
    }
    Ok(())
}
fn pals_search_arguments(s: &PalsSearchLaunchV3, model: &str) -> Vec<String> {
    vec![
        "--search=pals".into(),
        format!("--pals-model={model}"),
        format!("--pals-max-rounds={}", s.max_rounds),
        format!("--pals-max-cpu-nodes={}", s.max_cpu_nodes),
        format!("--pals-cpu-depth={}", s.cpu_depth),
        format!("--pals-max-situations={}", s.max_situations),
    ]
}
impl LockedPalsArenaLaunchV3 {
    pub fn input(&self) -> &PalsArenaLaunchV3 {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn to_json(&self) -> Result<String, ArenaError> {
        serde_json::to_string_pretty(&serde_json::json!({"domain":PALS_ARENA_V3_DOMAIN,"sha256":self.sha256,"input":self.input})).map_err(|e|invalid(e.to_string()))
    }
    pub fn from_json(text: &str) -> Result<Self, ArenaError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Envelope {
            domain: String,
            sha256: String,
            input: PalsArenaLaunchV3,
        }
        require(
            text.len() <= MAX_JSON_BYTES,
            "locked launch JSON budget exceeded",
        )?;
        let envelope: Envelope = crate::decode_json(text)?;
        let lock = envelope.input.lock()?;
        require(
            envelope.domain == PALS_ARENA_V3_DOMAIN && envelope.sha256 == lock.sha256,
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
        let PalsEngineV3::Pals(e) = &self.input.semantic_lock.manifest.engines[role_index(role)]
        else {
            return Err(invalid("native recipe has no PALS identity"));
        };
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
                && value["model_semantics"] == "rovezero.pals-model.v1"
                && value["role_batching"] == "one_scalar_role_per_physical_batch"
        } else {
            value["schema"] == "rovezero.pals-model.v1"
                && (value["layout"].is_null() || value["layout"] == "separate_pc")
        };
        require(
            layout_valid
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
}
impl crate::native_launch::sealed::Sealed for LockedPalsArenaLaunchV3 {}
impl NativeLaunchDeclaration for LockedPalsArenaLaunchV3 {
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
        "PALS-V3"
    }
    fn amend_execution_limitations(&self, limitations: &mut Vec<String>) {
        if let Some(scope) = limitations.get_mut(0) {
            *scope="PALS V3 whole-system or explicitly controlled paired pilot; P/C native sessions and CPU/reference UCI processes are distinct; no Elo, training or model promotion".into();
        }
        if let Some(cache) = limitations.get_mut(8) {
            *cache="PALS native runtime uses the existing bounded content-addressed shared user cache outside the attempt; hits rehash and retain read-only native file pins through physical completion. Private input snapshots and per-attempt evidence remain owned and bounded separately; cache reuse is not NN inference evidence".into();
        }
        limitations.push("CPU TT byte admission covers the compiled inline slot layout; retained identity heaps, allocator overhead and total peak are separately bounded by the verified inherited memory cgroup".into());
        limitations.push("CPU_T profile is preserved as declared identity only; this pilot executes CPU_R and P/C, with V absent and training_executed=false".into());
        limitations.push(format!("PALS runner status patch: {:?}; legacy clock-only records remain readable, while new execution/Core requires the separately registered clock+reap-status runner", self.input.runner_status_patch()));
        if self
            .input
            .endpoints
            .iter()
            .any(|e| e.cuda_model().is_some())
        {
            limitations.push("PALS CUDA recipe fixes FP32/TF32 off, physical B1, host K/V, device 0 and per-session 2GiB arena declaration; CPU fallback/I/O binding/CUDA Graph are disabled; arena declarations and summed model device budget do not attest observed VRAM peak".into());
        }
    }
    fn seed(&self) -> u64 {
        self.input.semantic_lock.manifest.pilot.seed
    }
    fn validate_execution(&self) -> Result<(), ArenaError> {
        self.input.validate()
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
        Ok(vec![
            format!("--pals-output-root={root}").into(),
            format!("--pals-launch-sha256={}", self.sha256).into(),
            format!(
                "--pals-endpoint-id={}",
                self.endpoint_views[role_index(role)].id
            )
            .into(),
            pals_shared_runtime_argument()?,
        ])
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
        Ok(vec![pals_shared_runtime_argument()?])
    }
}

/// Accepted native evidence is recorded in the existing arena receipt. The
/// original producer fields and hashes remain available without MCTS relabeling.
#[derive(Clone, Debug, Serialize)]
pub struct PalsNativeSessionAuditV3 {
    pub endpoint_id: String,
    pub process_id: u32,
    pub launch_sha256: String,
    pub startup_sha256: String,
    pub termination_sha256: String,
    pub completed_role_inputs: u64,
    pub search_consumed_role_inputs: u64,
    pub completed_new_game_resets: u64,
    pub physical_shutdown_confirmed: bool,
    pub native_buffers_released: bool,
    pub startup_nn_inputs_completed: u64,
    pub startup_nn_calls_completed: u64,
    pub startup_role_inputs_completed: u64,
    pub startup_probe: Option<serde_json::Value>,
    pub execution: Option<serde_json::Value>,
    pub raw_native: serde_json::Value,
    pub raw_search_work: serde_json::Value,
}
impl NativeProviderDeclaration for LockedPalsArenaLaunchV3 {
    type Audit = PalsNativeSessionAuditV3;
    fn scope(&self) -> &'static str {
        "pals_v3_pc_native_or_explicit_mock_cpu_uci_rules_clocks_process_pilot_no_elo"
    }
    fn receipt_filename(&self) -> &'static str {
        "pals-arena-pair-receipt.v3.json"
    }
    fn receipt_version(&self) -> u32 {
        3
    }
    fn contract_revision(&self) -> &str {
        &self.input.semantic_lock.manifest.contract_revision
    }
    #[cfg(target_os = "linux")]
    fn validate_runtime_admission(&self) -> Result<(), ArenaError> {
        self.input.require_reap_status_runner()?;
        self.input.require_actual_native_epoch()?;
        verify_pals_inherited_resources(self).map(|_| ())
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
        let own = collect_pals_process_work(self, role, root_output)?;
        let selected: Vec<u32> = if own.is_empty() {
            let other = if role_index(role) == 0 {
                NativeEngineRole::Candidate
            } else {
                NativeEngineRole::Baseline
            };
            let other_records = collect_pals_process_work(self, other, root_output)?;
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
        "pals-native-startup.v3.json"
    }
    fn termination_filename(&self) -> &'static str {
        "pals-native-termination.v3.json"
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
        validate_pals_native_records(self, role, startup, termination, session)
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
fn array_hash(v: &serde_json::Value) -> Result<String, ArenaError> {
    let bytes: [u8; 32] = serde_json::from_value(v.clone())
        .map_err(|_| invalid("native digest is not exactly 32 bytes"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
fn count(v: &serde_json::Value, k: &str) -> Result<u64, ArenaError> {
    v[k].as_u64()
        .ok_or_else(|| invalid(format!("missing native counter {k}")))
}
/// All graph executions, including public memory, have explicit completion and
/// validation counters. The caller keeps startup and search snapshots distinct.
fn graph_stats(v: &serde_json::Value) -> Result<(u64, u64, u64), ArenaError> {
    let public = count(v, "public_nn_runs_completed")?;
    let role = count(v, "role_nn_runs_completed")?;
    let inputs = count(v, "completed_nn_inputs")?;
    let calls = public
        .checked_add(role)
        .ok_or_else(|| invalid("NN graph counters overflow"))?;
    require(
        inputs == calls
            && count(v, "public_nn_runs_failed_known")? == 0
            && count(v, "role_nn_runs_failed_known")? == 0
            && public <= count(v, "public_nn_runs_attempted")?
            && role <= count(v, "role_nn_runs_attempted")?
            && count(v, "validated_public_outputs")? <= public
            && count(v, "validated_role_outputs")? <= role
            && count(v, "live_public_cache_entries")? <= 1,
        "native graph completion/failure/validation counters inconsistent",
    )?;
    Ok((inputs, calls, role))
}
/// Pure wire validation. No self-reported field replaces process exit or PGN.
pub fn validate_pals_native_records(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
) -> Result<(PalsNativeSessionAuditV3, u32), ArenaError> {
    require(
        startup.len() <= 128 * 1024 && termination.len() <= 128 * 1024,
        "native receipt budget exceeded",
    )?;
    let s = json(startup)?;
    let t = json(termination)?;
    let native = lock.native(role)?;
    let cuda = lock.input.endpoints[role_index(role)].cuda_model();
    let engine = &lock.endpoint_views[role_index(role)];
    require(
        s["schema_version"] == 3
            && t["schema_version"] == 3
            && s["domain"] == PALS_NATIVE_STARTUP_V3_DOMAIN
            && t["domain"] == PALS_NATIVE_TERMINATION_V3_DOMAIN,
        "native wire revision/domain differs",
    )?;
    for field in [
        "endpoint_id",
        "launch_sha256",
        "process_id",
        "binary_sha256",
        "runtime_sha256",
        "runtime_bundle_sha256",
        "provider",
        "precision",
    ] {
        require(
            s[field] == t[field],
            "native startup/termination identity differs",
        )?;
    }
    require(
        s["endpoint_id"] == engine.id
            && s["launch_sha256"] == lock.sha256
            && s["binary_sha256"] == engine.binary.sha256
            && s["runtime_sha256"] == native.runtime.sha256
            && s["provider"] == if cuda.is_some() { "cuda" } else { "cpu" }
            && s["precision"] == "fp32"
            && s["service_exit_success"] == false
            && t["service_exit_success"] == true,
        "native executable/runtime/provider identity differs",
    )?;
    let pid = count(&s, "process_id")?;
    require(
        pid > 0 && pid <= i32::MAX as u64 && session == format!("native-process-{pid}"),
        "native PID/session mismatch",
    )?;
    let sn = &s["native"];
    let tn = &t["native"];
    let mut startup_nn = (0, 0, 0);
    if let Some(cuda) = cuda {
        require(
            s["runtime_bundle_sha256"] == cuda.cuda_bundle.canonical_sha256,
            "CUDA envelope canonical runtime bundle differs",
        )?;
        require(
            sn["execution"] == tn["execution"] && sn["startup_probe"] == tn["startup_probe"],
            "CUDA execution/probe identity changed",
        )?;
        let execution = &sn["execution"];
        require(
            execution["provider"] == "cuda"
                && execution["device_id"] == cuda.device_id
                && execution["session_arena_bytes"] == cuda.session_arena_bytes
                && execution["device_public_memory"] == false
                && array_hash(&execution["runtime_sha256"])? == native.runtime.sha256
                && array_hash(&execution["runtime_bundle_sha256"])?
                    == cuda.cuda_bundle.canonical_sha256,
            "CUDA provider/device/session/runtime bundle identity differs",
        )?;
        let transient = count(execution, "transient_request_device_bytes")?
            .checked_add(count(execution, "transient_execution_device_bytes")?)
            .ok_or_else(|| invalid("CUDA transient reservation overflow"))?;
        let arenas = cuda
            .session_arena_bytes
            .checked_mul(native.graphs.len() as u64)
            .ok_or_else(|| invalid("CUDA session reservation overflow"))?;
        require(
            transient > 0
                && arenas.checked_add(transient).is_some_and(|n| {
                    n <= lock.input.semantic_lock.manifest.resources[role_index(role)]
                        .device_allocation_max_bytes
                })
                && count(execution, "pinned_request_bytes")? == 0,
            "CUDA declared budget does not cover actual tensor reservations plus session declarations",
        )?;
        let probe = &sn["startup_probe"];
        require(
            count(probe, "completed_proposer_calls")? == 1
                && count(probe, "completed_critic_calls")? == 1
                && probe["runtime_mapping_confirmed"] == true
                && probe["reset_completed"] == true,
            "CUDA startup requires actual P/C callbacks, full runtime audit and reset",
        )?;
        startup_nn = graph_stats(&probe["backend_stats"])?;
        require(
            startup_nn.0 > 0
                && startup_nn.2 == 2
                && count(&probe["backend_stats"], "new_game_resets")? >= 1
                && tn["final_runtime_mapping_confirmed"] == true,
            "CUDA startup NN work/reset unavailable or inconsistent",
        )?;
    } else {
        require(
            sn["startup_probe"].is_null()
                && tn["startup_probe"].is_null()
                && s["runtime_bundle_sha256"].is_null(),
            "CPU recipe cannot claim CUDA initialization work",
        )?;
        require(
            (sn["execution"].is_null() && tn["execution"].is_null())
                || (sn["execution"].is_object() && tn["execution"].is_object()),
            "CPU execution identity is malformed or present in only one record",
        )?;
        if sn["execution"].is_object() {
            require(
                sn["execution"] == tn["execution"]
                    && sn["execution"]["provider"] == "cpu"
                    && sn["execution"]["device_id"].is_null()
                    && sn["execution"]["session_arena_bytes"].is_null()
                    && sn["execution"]["runtime_bundle_sha256"].is_null()
                    && count(&sn["execution"], "transient_request_device_bytes")? == 0
                    && count(&sn["execution"], "transient_execution_device_bytes")? == 0
                    && count(&sn["execution"], "pinned_request_bytes")? == 0
                    && sn["execution"]["device_public_memory"] == false
                    && array_hash(&sn["execution"]["runtime_sha256"])? == native.runtime.sha256,
                "CPU native execution metadata differs",
            )?;
        }
    }
    let PalsEngineV3::Pals(e) = &lock.input.semantic_lock.manifest.engines[role_index(role)] else {
        return Err(invalid("missing native model identity"));
    };
    let (checkpoint, trained) = match &e.model.weights {
        PalsWeightIdentityV3::Untrained { artifact, .. } => (artifact, false),
        PalsWeightIdentityV3::Trained { artifact, .. } => (artifact, true),
        _ => return Err(invalid("missing actual weights")),
    };
    for field in [
        "model_epoch",
        "export_manifest_sha256",
        "encoding_semantic_sha256",
        "adapter_source_sha256",
        "trained",
        "residency",
        "process_epoch",
        "frozen_epoch",
    ] {
        require(
            sn[field] == tn[field],
            "native frozen identity changed during game",
        )?;
    }
    require(
        array_hash(&sn["model_epoch"])? == checkpoint.sha256
            && array_hash(&sn["export_manifest_sha256"])? == native.export.sha256
            && array_hash(&sn["encoding_semantic_sha256"])? == native.encoding_semantic_sha256
            && array_hash(&sn["adapter_source_sha256"])? == native.adapter_source_sha256
            && sn["trained"] == trained
            && count(sn, "process_epoch")? == 1,
        "native model/adapter/epoch identity differs",
    )?;
    require(
        (e.model.frozen_epoch == 0 && sn["frozen_epoch"].is_null() && tn["frozen_epoch"].is_null())
            || sn["frozen_epoch"].as_u64() == Some(e.model.frozen_epoch),
        "native frozen deployment epoch differs from immutable semantic lock",
    )?;
    for field in [
        "physically_completed_role_calls",
        "completed_role_inputs",
        "failed_physical_role_calls",
        "invalid_role_outputs",
        "delivered_role_inputs",
        "search_consumed_role_inputs",
        "canceled_requests",
        "expired_requests",
        "completed_new_game_resets",
        "physical_runs_in_flight",
        "request_high_water",
        "execution_high_water",
    ] {
        require(
            count(sn, field)? == 0,
            "startup contains work from another game/run",
        )?;
    }
    require(
        sn["last_failure"].is_null(),
        "startup retained an earlier failure",
    )?;
    require(
        sn["quarantined"] == false
            && sn["physical_shutdown_confirmed"] == false
            && sn["native_buffers_released"] == false,
        "startup owner is not live and fresh",
    )?;
    let physical = count(tn, "physically_completed_role_calls")?;
    let completed = count(tn, "completed_role_inputs")?;
    let failed = count(tn, "failed_physical_role_calls")?;
    let delivered = count(tn, "delivered_role_inputs")?;
    let consumed = count(tn, "search_consumed_role_inputs")?;
    // Historical CPU producer records may omit this additive observation;
    // preserve their wire reader. New Core assembly requires the actual count.
    if !tn["observer_failures"].is_null() {
        require(
            count(tn, "observer_failures")? == 0 && tn["last_observer_failure"].is_null(),
            "native observer failed; physical counters are preserved without Core acceptance",
        )?;
    }
    require(
        physical
            == completed
                .checked_add(failed)
                .ok_or_else(|| invalid("native count overflow"))?
            && failed == 0
            && count(tn, "invalid_role_outputs")? == 0
            && consumed <= delivered
            && delivered <= completed
            && count(tn, "request_high_water")? >= physical,
        "native completion/delivery/consumption counters inconsistent or failed",
    )?;
    require(
        count(tn, "execution_high_water")? >= physical,
        "native physical execution high-water is below completed work",
    )?;
    require(
        count(tn, "completed_new_game_resets")? >= 1
            && count(tn, "game_generation")? > count(sn, "game_generation")?
            && count(tn, "physical_runs_in_flight")? == 0
            && tn["quarantined"] == false
            && tn["physical_shutdown_confirmed"] == true
            && tn["native_buffers_released"] == true,
        "native game reset or physical completion/join/release is unconfirmed",
    )?;
    let graphs = sn["residency"]["graphs"]
        .as_array()
        .ok_or_else(|| invalid("missing native residency graph evidence"))?;
    let shared = native.graphs.iter().any(|g| g.role == "shared_pc");
    require(
        sn["residency"] == tn["residency"]
            && sn["residency"]["native_sessions"] == native.graphs.len()
            && graphs.len() == native.graphs.len()
            && if shared {
                sn["residency"]["layout"] == "shared_pc_if"
                    && sn["residency"]["role_reader_weights_shared"].is_null()
            } else {
                sn["residency"]["role_reader_weights_shared"] == false
                    && (sn["residency"]["layout"].is_null()
                        || sn["residency"]["layout"] == "separate_pc")
            },
        "actual P/C graph layout/residency differs; optimized parameter sharing remains unknown",
    )?;
    for g in &native.graphs {
        require(
            graphs
                .iter()
                .filter(|observed| {
                    observed["role"] == g.role
                        && array_hash(&observed["sha256"]).is_ok_and(|h| h == g.artifact.sha256)
                        && observed["serialized_bytes"] == g.artifact.bytes
                })
                .count()
                == 1,
            "native graph hash/bytes differ",
        )?;
    }
    Ok((
        PalsNativeSessionAuditV3 {
            endpoint_id: engine.id.clone(),
            process_id: pid as u32,
            launch_sha256: lock.sha256.clone(),
            startup_sha256: digest(startup),
            termination_sha256: digest(termination),
            completed_role_inputs: completed,
            search_consumed_role_inputs: consumed,
            completed_new_game_resets: count(tn, "completed_new_game_resets")?,
            physical_shutdown_confirmed: true,
            native_buffers_released: true,
            startup_nn_inputs_completed: startup_nn.0,
            startup_nn_calls_completed: startup_nn.1,
            startup_role_inputs_completed: startup_nn.2,
            startup_probe: sn["startup_probe"]
                .as_object()
                .map(|_| sn["startup_probe"].clone()),
            execution: sn["execution"].as_object().map(|_| sn["execution"].clone()),
            raw_native: tn.clone(),
            raw_search_work: t["search_work"].clone(),
        },
        pid as u32,
    ))
}

/// Snapshot/pin preparation starts no engines and grants no NN-ready claim.
pub fn prepare_pals_pair_launch(
    spec: &LockedPalsArenaLaunchV3,
    source_root: &Path,
    output_root: &Path,
    label: &str,
) -> Result<NativeLaunchOwner<LockedPalsArenaLaunchV3>, Box<crate::NativePreparationFailure>> {
    if let Err(cause) = verify_pals_launch_exports(spec, source_root) {
        return Err(Box::new(crate::NativePreparationFailure {
            receipt:crate::NativePreparationReceipt {receipt_version:3,execution_ready:false,input_sha256:spec.sha256.clone(),
                output_directory:"not-created".into(),attempt_created:false,attempted_snapshot_relative_paths:vec![],completed_snapshots:vec![],
                child_spawned:false,writers_closed:true,input_pins_required:false,original_error:cause.to_string(),
                subsequent_file_owner:"source assets retained; descriptor verification failed before snapshot creation or child launch".into(),automatic_retry:false},
            cause,receipt_artifact:None,persistence_error:None,
        }));
    }
    crate::native_launch::prepare_native_launch_for(spec, source_root, output_root, label)
}
/// Uses the existing bounded runner, clocks, Rules replay and cleanup receipts.
pub fn run_pals_pair(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV3>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<
    crate::NativePairOutput<LockedPalsArenaLaunchV3>,
    Box<crate::NativePairFailure<LockedPalsArenaLaunchV3>>,
> {
    crate::native_runner::run_native_pair_for(owner, cancel)
}

/// Time scope starts before UCI preflight and ends after mandatory arena
/// postchecks and receipt persistence. Snapshot preparation remains separate.
pub struct PalsObservedPairOutputV3 {
    pub native: crate::NativePairOutput<LockedPalsArenaLaunchV3>,
    pub wall_time_ms: u64,
}
pub fn run_pals_pair_observed(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV3>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<PalsObservedPairOutputV3, Box<crate::NativePairFailure<LockedPalsArenaLaunchV3>>> {
    let started = std::time::Instant::now();
    run_pals_pair(owner, cancel).map(|native| PalsObservedPairOutputV3 {
        native,
        wall_time_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    })
}

/// Process-wide work is observed once per restarted game. CPU task reuse and
/// native NN execution have different units; this record preserves both wires.
#[derive(Clone, Debug, Serialize)]
pub struct PalsProcessWorkAuditV3 {
    pub endpoint_id: String,
    pub process_id: u32,
    pub startup_sha256: String,
    pub termination_sha256: String,
    pub startup_bytes: u64,
    pub termination_bytes: u64,
    pub search_work: serde_json::Value,
}

fn work_count(work: &serde_json::Value, field: &str) -> Result<u64, ArenaError> {
    work.get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            invalid(format!(
                "work observation missing, unknown or invalid: {field}"
            ))
        })
}

/// Validate the cumulative work snapshot independently of root-coverage gauges.
/// A failed go can retain observed work. Missing work never becomes zero.
pub fn validate_pals_process_work(
    work: &serde_json::Value,
    kind: &str,
    startup: bool,
) -> Result<(), ArenaError> {
    require(
        work["schema_version"] == 1 && work["search_kind"] == kind,
        "work schema/search kind differs",
    )?;
    // Historical v1 receipts omit this additive identity. Keep their absence
    // observable; an explicit identity must match the registered semantics.
    if let Some(resolver) = work.get("pals_resolver") {
        require(
            kind == "pals"
                && resolver["version"] == rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION
                && resolver["semantics_sha256"]
                    == serde_json::json!(
                        Sha256::digest(
                            rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS.as_bytes()
                        )
                        .to_vec()
                    ),
            "work resolver identity belongs to a different policy/search path",
        )?;
    }
    let go = work_count(work, "go_invocations")?;
    let success = work_count(work, "successful_returns")?;
    let failed = work_count(work, "failed_returns")?;
    let active = work_count(work, "active_invocations")?;
    require(
        active == 0
            && go
                == success
                    .checked_add(failed)
                    .ok_or_else(|| invalid("work return count overflow"))?,
        "work returns/active invocation inconsistent",
    )?;
    require(
        work_count(work, "unobserved_work_invocations")? == 0
            && work_count(work, "physical_unknown_returns")? == 0,
        "work or physical completion observation unknown",
    )?;
    for field in ["canceled_returns", "deadline_returns"] {
        require(
            work_count(work, field)? <= go,
            "work outcome counter exceeds invocations",
        )?;
    }
    let (totals, absent) = if kind == "pals" {
        ("pals", "cpu")
    } else if kind == "cpu" {
        ("cpu", "pals")
    } else {
        return Err(invalid("unsupported work kind"));
    };
    require(
        work[absent].is_null() && work[totals].is_object(),
        "work totals belong to a different search path",
    )?;
    // Only the Core's mandatory units require numeric observations. Partial
    // output ACKs and root gauges can be unknown while completed-task consumers
    // are observed; raw optional counters retain that distinction.
    let mandatory: &[&str] = if kind == "pals" {
        &[
            "completed_proposer_calls",
            "completed_repair_calls",
            "completed_critic_calls",
            "cpu_tasks_requested",
            "completed_cpu_tasks",
            "reused_completed_cpu_tasks_consumed",
            "consumed_cpu_tasks",
            "cpu_nodes",
            "consumed_role_outputs",
        ]
    } else {
        &[
            "tasks_requested",
            "requested_depth_completed",
            "rules_terminal_reports",
            "completed_reports_accepted_for_uci_output",
            "nodes",
        ]
    };
    for field in mandatory {
        work_count(&work[totals], field)?;
    }
    for (field, value) in work[totals].as_object().unwrap() {
        if !matches!(
            field.as_str(),
            "retained_situations_peak" | "unknown_root_children" | "max_completed_depth"
        ) {
            if let Some(observed) = value.as_u64() {
                if startup {
                    require(observed == 0, "startup work is not zero")?;
                }
            } else {
                require(value.is_null(), "work total type invalid")?;
            }
        }
    }
    if startup {
        require(
            go == 0 && failed == 0 && success == 0,
            "startup invocation already executed",
        )?;
    }
    Ok(())
}

pub fn validate_pals_search_work_records(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
) -> Result<PalsProcessWorkAuditV3, ArenaError> {
    require(
        startup.len() <= MAX_JSON_BYTES && termination.len() <= MAX_JSON_BYTES,
        "work envelope byte budget exceeded",
    )?;
    let (s, t) = (json(startup)?, json(termination)?);
    let engine = &lock.endpoint_views[role_index(role)];
    let native = matches!(
        &lock.input.endpoints[role_index(role)],
        PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
    );
    let (sdomain, tdomain, kind) = match &lock.input.endpoints[role_index(role)] {
        PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_) => (
            PALS_NATIVE_STARTUP_V3_DOMAIN,
            PALS_NATIVE_TERMINATION_V3_DOMAIN,
            "pals",
        ),
        PalsEndpointLaunchV3::LegalOrderMock(_) => (
            PALS_SEARCH_WORK_STARTUP_V3_DOMAIN,
            PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN,
            "pals",
        ),
        PalsEndpointLaunchV3::OwnCpu { .. } => (
            PALS_SEARCH_WORK_STARTUP_V3_DOMAIN,
            PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN,
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
            value["schema_version"] == 3
                && value["domain"] == domain
                && value["endpoint_id"] == engine.id
                && value["launch_sha256"] == lock.sha256
                && value["process_id"] == pid
                && value["binary_sha256"] == engine.binary.sha256
                && value["service_exit_success"] == ending,
            "work envelope identity/start/end differs",
        )?;
        validate_pals_process_work(&value["search_work"], kind, !ending)?;
    }
    if native {
        // Native worker counters have their own accepted/delivered lifecycle.
        validate_pals_native_records(lock, role, startup, termination, session)?;
    }
    Ok(PalsProcessWorkAuditV3 {
        endpoint_id: engine.id.clone(),
        process_id: pid,
        startup_sha256: digest(startup),
        termination_sha256: digest(termination),
        startup_bytes: startup.len() as u64,
        termination_bytes: termination.len() as u64,
        search_work: t["search_work"].clone(),
    })
}

#[cfg(target_os = "linux")]
fn read_work_file(directory: &cap_std::fs::Dir, filename: &str) -> Result<Vec<u8>, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsExt, OpenOptionsFollowExt};
    let metadata = directory
        .symlink_metadata(filename)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    require(
        metadata.is_file() && metadata.len() > 0 && metadata.len() <= MAX_JSON_BYTES as u64,
        "work file type/byte bound invalid",
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let mut file = directory
        .open_with(filename, &options)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    require(
        file.metadata()
            .map_err(|e| ArenaError::Io(e.to_string()))?
            .is_file(),
        "pinned work handle is not regular",
    )?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take(MAX_JSON_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    require(
        bytes.len() <= MAX_JSON_BYTES,
        "work file grew past byte bound",
    )?;
    Ok(bytes)
}

/// Only dedicated runtime receipt slots are read; cache/model directories are
/// never scanned as search work and no process is started by this collector.
#[cfg(target_os = "linux")]
pub fn collect_pals_process_work(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    root: &cap_std::fs::Dir,
) -> Result<Vec<PalsProcessWorkAuditV3>, ArenaError> {
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
            "pals-native-startup.v3.json",
            "pals-native-termination.v3.json",
        )
    } else {
        (
            "search-work-startup.v3.json",
            "search-work-termination.v3.json",
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
        records.push(validate_pals_search_work_records(
            lock, role, &start, &end, name,
        )?);
    }
    require(
        records.len() == 2 && records[0].process_id != records[1].process_id,
        "two distinct restarted game work processes required",
    )?;
    Ok(records)
}

fn endpoint_work_receipt(
    lock: &LockedPalsArenaLaunchV3,
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
    for (key, requested) in engine.requested_options() {
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
        uci_ready_observed: true,
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
        PalsEngineV3::ReferenceUci(_) => require(
            work.is_none() && native.is_none(),
            "reference UCI cannot inherit RoveZero work",
        )?,
        PalsEngineV3::OwnCpu(_) => {
            let w = &work
                .ok_or_else(|| invalid("own CPU work unobserved"))?
                .search_work;
            validate_pals_process_work(w, "cpu", false)?;
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
        PalsEngineV3::Pals(p) => {
            let w = &work
                .ok_or_else(|| invalid("PALS search work unobserved"))?
                .search_work;
            validate_pals_process_work(w, "pals", false)?;
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
                // Provider/device declarations and a mapped-runtime audit do
                // not observe graph node placement. Preserve Unknown until a
                // separate actual major-NN CUDA placement witness is supplied.
            }
        }
    }
    Ok(receipt)
}

fn core_game_outcome(
    game: &crate::GamePgnAudit,
) -> Result<(PalsResultV3, PalsTerminationV3, Option<String>), ArenaError> {
    let result = match game.result {
        crate::GameResult::WhiteWin => PalsResultV3::WhiteWin,
        crate::GameResult::BlackWin => PalsResultV3::BlackWin,
        crate::GameResult::Draw => PalsResultV3::Draw,
    };
    match game.classification.as_str() {
        "rules_terminal" | "accepted_claim" => {
            let reason = game
                .terminal_reason
                .as_deref()
                .ok_or_else(|| invalid("Rules termination observation absent"))?;
            Ok((
                result,
                if reason == "checkmate" {
                    PalsTerminationV3::Checkmate
                } else {
                    PalsTerminationV3::RulesDraw
                },
                None,
            ))
        }
        "engine_loss" => {
            let reason = match game
                .engine_failure
                .ok_or_else(|| invalid("engine loss cause absent"))?
            {
                crate::EngineFailureKind::IllegalMove => PalsTerminationV3::IllegalMove,
                crate::EngineFailureKind::Crash => PalsTerminationV3::EngineCrash,
                crate::EngineFailureKind::Timeout => PalsTerminationV3::TimeForfeit,
            };
            Ok((
                result,
                reason,
                Some(
                    game.loser_engine
                        .clone()
                        .ok_or_else(|| invalid("engine failure offender absent"))?,
                ),
            ))
        }
        "incomplete" => Ok((PalsResultV3::Incomplete, PalsTerminationV3::PlyLimit, None)),
        _ => Err(invalid("unsupported PGN outcome classification")),
    }
}

/// The pinned source uses one synchronous game worker, restart=on and recover
/// off: after Finished it destroys white then black before starting game 2.
/// Check those exact game/color windows; exit order alone is insufficient.
/// This codec still requires the separate owned-group/native completion gates.
pub fn validate_pals_game_process_trace(
    lock: &LockedPalsArenaLaunchV3,
    stdout: &[u8],
    known_work_pids: &[BTreeSet<u32>; 2],
) -> Result<[[u32; 2]; 2], ArenaError> {
    require(
        !stdout.is_empty() && stdout.len() <= 64 * 1024 * 1024 && stdout.ends_with(b"\n"),
        "game process trace byte/newline bound invalid",
    )?;
    let text =
        std::str::from_utf8(stdout).map_err(|_| invalid("game process trace is not UTF-8"))?;
    let manifest = &lock.input.semantic_lock.manifest;
    let mut result = [[0u32; 2]; 2];
    let mut seen = BTreeSet::new();
    let mut game = 0usize;
    // trace start, renderer start, running, trace result, renderer finish,
    // white exit, black exit. A later game cannot begin before both exits.
    let mut phase = 0u8;
    for (line_number, line) in text.lines().enumerate() {
        require(
            line_number < 131_072 && line.len() <= 4096,
            "game process trace line bound exceeded",
        )?;
        let trace = crate::native_exit::cuda_exit_trace_message(line);
        let relevant = line.starts_with("Started game ")
            || line.starts_with("Finished game ")
            || trace.is_some_and(|m| {
                m.starts_with("Game ")
                    || m.starts_with("Process with pid:")
                    || m.starts_with("Force terminating process with pid:")
            });
        if !relevant {
            continue;
        }
        require(game < 2, "extra game/process trace after two game windows")?;
        let number = game + 1;
        let white = &manifest.pilot.white_order[game];
        let black = manifest
            .engines
            .iter()
            .find(|e| e.id() != white)
            .ok_or_else(|| invalid("black endpoint absent"))?
            .id();
        let white_role = manifest
            .engines
            .iter()
            .position(|e| e.id() == white)
            .ok_or_else(|| invalid("white endpoint absent"))?;
        let black_role = 1 - white_role;
        let trace_start = format!("Game {number} between {white} and {black} starting");
        let trace_finish = format!("Game {number} between {white} and {black} finished");
        let renderer_start = format!("Started game {number} of 2 ({white} vs {black})");
        let renderer_finish = format!("Finished game {number} ({white} vs {black}): ");
        if trace == Some(trace_start.as_str()) {
            require(phase == 0, "game start outside completed prior window")?;
            phase = 1;
        } else if line == renderer_start {
            require(phase == 1, "renderer start lacks matching TRACE game/color")?;
            phase = 2;
        } else if trace == Some(trace_finish.as_str()) {
            require(phase == 2, "TRACE finished outside running game")?;
            phase = 3;
        } else if trace
            .is_some_and(|m| m.starts_with(&format!("Game {number} finished with result ")))
        {
            require(phase == 3, "result TRACE outside matching finished game")?;
            phase = 4;
        } else if let Some(outcome) = line.strip_prefix(&renderer_finish) {
            require(
                phase == 4
                    && outcome.split_once(' ').is_some_and(|(r, a)| {
                        matches!(r, "1-0" | "0-1" | "1/2-1/2" | "*")
                            && a.starts_with('{')
                            && a.ends_with('}')
                    }),
                "renderer finish/result window invalid",
            )?;
            phase = 5;
        } else if let Some(message) = trace.and_then(|m| m.strip_prefix("Process with pid: ")) {
            let (pid, status) = message
                .split_once(" terminated with status: ")
                .ok_or_else(|| invalid("game exit renderer invalid"))?;
            let parsed = pid
                .parse::<u32>()
                .map_err(|_| invalid("game exit PID invalid"))?;
            require(
                matches!(phase, 5 | 6)
                    && pid == parsed.to_string()
                    && parsed > 0
                    && parsed <= i32::MAX as u32
                    && status == "0"
                    && seen.insert(parsed),
                "game exit outside finished window, failed, or duplicate",
            )?;
            let role = if phase == 5 { white_role } else { black_role };
            require(
                known_work_pids[role].is_empty() || known_work_pids[role].contains(&parsed),
                "white/black exit does not match actual work endpoint PID",
            )?;
            result[role][game] = parsed;
            if phase == 5 {
                phase = 6;
            } else {
                game += 1;
                phase = 0;
            }
        } else {
            return Err(invalid("foreign/unsupported game or exit renderer"));
        }
    }
    require(
        game == 2 && phase == 0 && seen.len() == 4,
        "game process correspondence incomplete; no relative-order fallback",
    )?;
    Ok(result)
}

/// A missing observation leaves the numeric Core receipt absent. The original
/// native receipt and PGN remain authoritative evidence of failures, never an
/// empty-success or an omitted engine loss.
#[derive(Clone, Debug, Serialize)]
pub struct PalsCoreAssemblyV3 {
    pub domain: String,
    pub arena_launch_sha256: String,
    pub native_receipt: ArtifactRef,
    pub wall_time_ms: Option<u64>,
    pub cleanup_time_ms: Option<u64>,
    pub timing_scope: String,
    pub game_process_correspondence: String,
    pub work: Vec<PalsProcessWorkAuditV3>,
    pub core: Option<PalsRunReceiptV3>,
    pub assembly_error: Option<String>,
    pub training_executed: bool,
}

#[cfg(target_os = "linux")]
pub fn assemble_pals_core_receipt(
    output: &PalsObservedPairOutputV3,
    cleanup_time_ms: Option<u64>,
) -> PalsCoreAssemblyV3 {
    let native = &output.native;
    let owner = native.launch_owner();
    let lock = owner.spec();
    let mut assembly = PalsCoreAssemblyV3 { domain: "rz-pals-core-assembly-v3/1".into(), arena_launch_sha256: lock.sha256.clone(),
        native_receipt: native.receipt_artifact.clone(), wall_time_ms: Some(output.wall_time_ms), cleanup_time_ms,
        timing_scope: "UCI preflight through mandatory arena postcheck/receipt; snapshot preparation separate; cleanup is supervisor-owned measured interval".into(),
        game_process_correspondence: "pinned Fastchess 1.8.2 source: one synchronous worker, restart=on, recover=false; TRACE/renderer game id and color plus finished-white-exit-black-exit windows; separate native/owned-group completion gates required".into(),
        work: vec![], core: None, assembly_error: None, training_executed: false };
    let result: Result<PalsRunReceiptV3, ArenaError> = (|| {
        lock.input.require_reap_status_runner()?;
        lock.input.require_actual_native_epoch()?;
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            assembly.work.extend(collect_pals_process_work(
                lock,
                role,
                &owner.snapshot.directory,
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
        let mapped =
            validate_pals_game_process_trace(lock, &native.process().stdout, &known_work_pids)?;
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
        let pgn_artifact = receipts
            .artifacts
            .iter()
            .find(|a| a.path.ends_with("/match.pgn"))
            .ok_or_else(|| invalid("retained PGN artifact absent"))?
            .clone();
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
                engines.push(endpoint_work_receipt(lock, role, work, nn, preflight)?);
            }
            let (result, termination, failed_endpoint) = core_game_outcome(game)?;
            if result == PalsResultV3::Incomplete {
                failures.insert(PalsRunFailureV3::IncompleteGame);
            }
            games.push(PalsGameReceiptV3 {
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
        let core = PalsRunReceiptV3 {
            domain: PALS_RECEIPT_V3_DOMAIN.into(),
            run_id: m.run_id.clone(),
            pair_id: m.pair_id.clone(),
            lock_sha256: lock.input.semantic_lock.canonical_sha256.clone(),
            training_executed: false,
            wall_time_ms: output.wall_time_ms,
            cleanup_time_ms: cleanup,
            pair_eligible: failures.is_empty(),
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
pub fn save_pals_core_assembly(
    output: &PalsObservedPairOutputV3,
    assembly: &PalsCoreAssemblyV3,
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
        bytes.len() <= 64 * 1024,
        "core assembly reservation exceeds 64KiB",
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
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = owner
        .snapshot
        .directory
        .open_with("pals-core-assembly.v3.json", &options)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    Ok(ArtifactRef { path: format!("{}/pals-core-assembly.v3.json", owner.snapshot.output_directory), sha256: digest(&bytes), bytes: bytes.len() as u64,
        source: "PALS V3 actual arena work/clock/physical-lifetime assembly; raw native receipt retained".into(), license: "MIT execution evidence; external asset rights remain separate".into() })
}

/// Independently checks descriptor contents before preparation, using the same
/// bounded no-follow artifact verifier as every actual launch input.
pub fn verify_pals_launch_exports(
    spec: &LockedPalsArenaLaunchV3,
    source_root: &Path,
) -> Result<(), ArenaError> {
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
        }
    }
    Ok(())
}

/// Checks inherited limits; this does not install limits or claim a GPU peak.
/// Both engines are in the same serial-game cgroup in this first executor.
#[cfg(target_os = "linux")]
pub fn verify_pals_inherited_resources(
    spec: &LockedPalsArenaLaunchV3,
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
#[cfg(any(target_os = "linux", test))]
fn parse_pals_affinity(cpus: &str) -> Result<Vec<u32>, ArenaError> {
    require(
        !cpus.is_empty() && cpus.len() <= 4096,
        "affinity observation is empty or oversized",
    )?;
    let mut result = BTreeSet::new();
    for segment in cpus.split(',') {
        if let Some((lo, hi)) = segment.split_once('-') {
            let lo = lo
                .parse::<u32>()
                .map_err(|_| invalid("affinity range invalid"))?;
            let hi = hi
                .parse::<u32>()
                .map_err(|_| invalid("affinity range invalid"))?;
            require(hi >= lo && hi - lo < 64, "affinity range budget exceeded")?;
            for cpu in lo..=hi {
                require(result.insert(cpu), "duplicate CPU affinity observation")?;
            }
        } else {
            require(
                result.insert(
                    segment
                        .parse::<u32>()
                        .map_err(|_| invalid("affinity value invalid"))?,
                ),
                "duplicate CPU affinity observation",
            )?;
        }
        require(result.len() <= 64, "affinity count budget exceeded")?;
    }
    Ok(result.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_experiments::*;
    fn asset(path: &str) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: "a".repeat(64),
            bytes: 1000,
            source: "https://example.org/public-source".into(),
            license: "MIT".into(),
        }
    }
    fn component(id: &str) -> PalsComponentV3 {
        PalsComponentV3 {
            semantic_id: id.into(),
            implementation_sha256: "b".repeat(64),
            options: BTreeMap::new(),
        }
    }
    fn cpu() -> PalsOwnCpuV3 {
        PalsOwnCpuV3 {
            core: component("rz-cpu-pvs/0.1"),
            training_profile: component("cpu-independent-conservative-v1"),
            runtime_profile: component("cpu-plan-assisted-conservative-v1"),
            evaluation: component("bootstrap-material-pst-v1"),
            max_nodes_per_task: 100_000,
            max_depth: 16,
            max_task_ms: 900_000,
            max_tt_bytes: 16 * 1024 * 1024,
        }
    }
    fn fixture() -> PalsArenaLaunchV3 {
        let pals = PalsEndpointV3 {
            id: "pals".into(),
            binary: asset("bin/rove"),
            source_commit: "c".repeat(40),
            model: PalsModelIdentityV3 {
                architecture: "explicit-legal-order-role-mock-v1".into(),
                input_schema: "rz-pals-rules-fields-v1".into(),
                policy_head: "candidate-policy/1".into(),
                value_head: "stm-wdl/1".into(),
                implementation_sha256: "b".repeat(64),
                weights: PalsWeightIdentityV3::DeterministicMock { seed: 0 },
                frozen_epoch: 0,
                backend: PalsModelBackendV3::DeterministicMock,
                precision: PalsPrecisionV3::Fp32,
                max_batch_width: 1,
                exported_roles: BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
            },
            cpu: cpu(),
            search: component("pals"),
            runtime: component("single-owner/1"),
            pools: PalsPoolLimitsV3 {
                states: 4096,
                line_chunks: 65536,
                situations: 4096,
                observations: 65536,
                tasks: 32768,
                role_states: 1,
                memory_pages: 1,
                queue_requests: 1,
                host_bytes: 12 << 30,
                device_bytes: 0,
            },
            requested_options: BTreeMap::new(),
        };
        let resources = PalsResourcePolicyV3 {
            cpu_threads: 2,
            cpu_affinity: vec![0, 2],
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
            swap_max_bytes: 0,
            requested_gpu: None,
            device_allocation_max_bytes: 0,
        };
        let semantic = PalsRunManifestV3 {
            schema_version: 3,
            run_id: "pals-test".into(),
            pair_id: "pair-test".into(),
            contract_revision: "pals/0.1".into(),
            rules_profile: "standard-complete-history/1".into(),
            comparison: PalsComparisonV3::System,
            declared_changes: BTreeSet::from([PalsChangeAxisV3::Endpoint]),
            training_executed: false,
            engines: [
                PalsEngineV3::Pals(Box::new(pals)),
                PalsEngineV3::OwnCpu(Box::new(PalsCpuEndpointV3 {
                    id: "cpu".into(),
                    binary: asset("bin/rove"),
                    source_commit: "c".repeat(40),
                    cpu: cpu(),
                    requested_options: BTreeMap::new(),
                })),
            ],
            resources: [resources.clone(), resources],
            pilot: PalsPilotV3 {
                position_command: "position startpos".into(),
                games: 2,
                white_order: ["pals".into(), "cpu".into()],
                base_ms: 120_000,
                increment_ms: 1000,
                max_plies: 256,
                seed: 1,
                wall_time_max_ms: 900_000,
                cleanup_max_ms: 30_000,
                handshake_max_ms: 30_000,
                concurrent_games: 1,
                restart_processes_each_game: true,
                ponder: false,
                score_adjudication: false,
                elo_claim: false,
            },
        };
        let mut patch = asset("runner/clock.patch");
        patch.sha256 = FASTCHESS_CLOCK_PATCH_SHA256.into();
        patch.bytes = FASTCHESS_CLOCK_PATCH_BYTES;
        PalsArenaLaunchV3 {
            domain: PALS_ARENA_V3_DOMAIN.into(),
            semantic_lock: semantic.lock().unwrap(),
            runner: ToolIdentity {
                version: crate::FASTCHESS_VERSION.into(),
                source_url: crate::FASTCHESS_SOURCE_URL.into(),
                source_commit: crate::FASTCHESS_SOURCE_COMMIT.into(),
                binary: asset("runner/fastchess"),
                dirty: true,
                dirty_patch: Some(patch),
                build_mode: "release".into(),
                compiler: "clang".into(),
                target: "x86_64-unknown-linux-gnu".into(),
                isa: "x86-64".into(),
            },
            opening_artifact: asset("opening/startpos.pgn"),
            endpoints: [
                PalsEndpointLaunchV3::LegalOrderMock(PalsSearchLaunchV3 {
                    max_rounds: 16,
                    max_cpu_nodes: 100_000,
                    cpu_depth: 2,
                    max_situations: 4096,
                }),
                PalsEndpointLaunchV3::OwnCpu {
                    max_depth: 8,
                    max_nodes: 10_000,
                    tt_entries: 128,
                },
            ],
            budget: NativeResourceBudgetV1 {
                max_input_bytes: 32 * 1024 * 1024,
                max_output_bytes: 8 * 1024 * 1024,
                max_runtime_bytes: 1024 * 1024 * 1024,
                max_artifact_bytes: 2 * 1024 * 1024 * 1024,
                max_child_processes: 3,
                max_runtime_files: 64,
                max_runtime_depth: 4,
                address_space_per_process_bytes: 0,
            },
        }
    }
    fn native_fixture() -> PalsArenaLaunchV3 {
        let mut f = fixture();
        let PalsEngineV3::Pals(e) = &mut f.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model.architecture = "pals-width384-latent16-iterations2".into();
        e.model.backend = PalsModelBackendV3::OrtCpu;
        e.model.frozen_epoch = 1;
        e.model.weights = PalsWeightIdentityV3::Untrained {
            artifact: asset("model/checkpoint.pt"),
            initialization_seed: 1,
        };
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        f.endpoints[0] = PalsEndpointLaunchV3::OnnxCpu(PalsOnnxLaunchV3 {
            export: asset("model/export.json"),
            export_file: "export.json".into(),
            graphs: ["public", "proposer", "critic"]
                .into_iter()
                .map(|r| PalsGraphAssetV3 {
                    file: format!("{r}.onnx"),
                    role: r.into(),
                    artifact: asset(&format!("model/{r}.onnx")),
                })
                .collect(),
            runtime: asset("runtime/libonnxruntime.so"),
            encoding_semantic_sha256: "d".repeat(64),
            adapter_source_sha256: "b".repeat(64),
            search: PalsSearchLaunchV3 {
                max_rounds: 16,
                max_cpu_nodes: 100_000,
                cpu_depth: 2,
                max_situations: 4096,
            },
        });
        f
    }
    fn cuda_fixture() -> (PalsArenaLaunchV3, Vec<u8>) {
        let mut f = native_fixture();
        let PalsEndpointLaunchV3::OnnxCpu(mut model) = f.endpoints[0].clone() else {
            unreachable!()
        };
        model.graphs.retain(|g| g.role == "public");
        model.graphs.push(PalsGraphAssetV3 {
            file: "shared_pc_if.onnx".into(),
            role: "shared_pc".into(),
            artifact: asset("model/shared_pc_if.onnx"),
        });
        model.runtime = asset("runtime/libonnxruntime.so.1.22.0");
        let files: Vec<_> = CUDA_BUNDLE_FILENAMES
            .iter()
            .map(|(filename, role)| CudaBundleFileBindingV1 {
                filename: (*filename).into(),
                role: *role,
                artifact: asset(&format!("runtime/{filename}")),
            })
            .collect();
        let descriptor = serde_json::to_vec(&serde_json::json!({"schema_version":1,
            "files":files.iter().map(|f|serde_json::json!({"role":f.role,"filename":f.filename,
                "bytes":f.artifact.bytes,"sha256":f.artifact.sha256})).collect::<Vec<_>>()}))
        .unwrap();
        let mut bundle = CudaBundleBindingV1 {
            manifest: asset("runtime/bundle.json"),
            canonical_sha256: "0".repeat(64),
            files,
        };
        bundle.manifest.sha256 = digest(&descriptor);
        bundle.manifest.bytes = descriptor.len() as u64;
        bundle.canonical_sha256 = bundle.canonical_digest().unwrap();
        let PalsEngineV3::Pals(e) = &mut f.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model.backend = PalsModelBackendV3::OrtCuda;
        e.pools.device_bytes = 6 << 30;
        f.semantic_lock.manifest.resources[0].requested_gpu = Some("RTX 4050 6GB".into());
        f.semantic_lock.manifest.resources[0].device_allocation_max_bytes = 6 << 30;
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        f.endpoints[0] = PalsEndpointLaunchV3::OnnxCuda(PalsOnnxCudaLaunchV3 {
            model,
            cuda_bundle: bundle,
            device_id: 0,
            session_arena_bytes: 2 << 30,
        });
        f.budget.max_runtime_bytes = 4 << 30;
        f.budget.max_artifact_bytes = 8 << 30;
        (f, descriptor)
    }
    fn stats_fixture(public: u64, role: u64) -> serde_json::Value {
        serde_json::json!({"public_nn_runs_completed":public,"role_nn_runs_completed":role,
            "completed_nn_inputs":public+role,"public_nn_runs_attempted":public,"role_nn_runs_attempted":role,
            "public_nn_runs_failed_known":0,"role_nn_runs_failed_known":0,"validated_public_outputs":public,
            "validated_role_outputs":role,"live_public_cache_entries":0,"new_game_resets":1})
    }
    fn hash_array(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }
    fn cuda_records_fixture(
        lock: &LockedPalsArenaLaunchV3,
    ) -> (serde_json::Value, serde_json::Value) {
        let cuda = lock.input.endpoints[0].cuda_model().unwrap();
        let model = &cuda.model;
        let native = serde_json::json!({"physically_completed_role_calls":0,"completed_role_inputs":0,
            "failed_physical_role_calls":0,"invalid_role_outputs":0,"delivered_role_inputs":0,"search_consumed_role_inputs":0,
            "canceled_requests":0,"expired_requests":0,"completed_new_game_resets":0,"process_epoch":1,"game_generation":0,
            "request_high_water":0,"execution_high_water":0,"physical_runs_in_flight":0,"quarantined":false,
            "physical_shutdown_confirmed":false,"native_buffers_released":false,"last_failure":null,
            "model_epoch":hash_array(&"a".repeat(64)),"export_manifest_sha256":hash_array(&model.export.sha256),
            "encoding_semantic_sha256":hash_array(&model.encoding_semantic_sha256),"adapter_source_sha256":hash_array(&model.adapter_source_sha256),
            "trained":false,"frozen_epoch":1,"residency":{"native_sessions":model.graphs.len(),"layout":"shared_pc_if",
                "role_reader_weights_shared":null,"native_resident_parameter_bytes":null,"vram_peak_bytes":null,
                "graphs":model.graphs.iter().map(|g|serde_json::json!({"role":g.role,"sha256":hash_array(&g.artifact.sha256),
                    "serialized_bytes":g.artifact.bytes})).collect::<Vec<_>>()},
            "execution":{"provider":"cuda","device_id":0,"session_arena_bytes":2u64<<30,
                "runtime_sha256":hash_array(&model.runtime.sha256),"runtime_bundle_sha256":hash_array(&cuda.cuda_bundle.canonical_sha256),
                "transient_request_device_bytes":8192,"transient_execution_device_bytes":8192,"pinned_request_bytes":0,"device_public_memory":false},
            "startup_probe":{"completed_proposer_calls":1,"completed_critic_calls":1,"runtime_mapping_confirmed":true,
                "reset_completed":true,"backend_stats":stats_fixture(1,2)}});
        let start = serde_json::json!({"schema_version":3,"domain":PALS_NATIVE_STARTUP_V3_DOMAIN,
            "endpoint_id":"pals","launch_sha256":lock.sha256(),"process_id":100,"binary_sha256":"a".repeat(64),
            "runtime_sha256":model.runtime.sha256,"runtime_bundle_sha256":cuda.cuda_bundle.canonical_sha256,
            "provider":"cuda","precision":"fp32","service_exit_success":false,"native":native});
        let mut end = start.clone();
        end["domain"] = PALS_NATIVE_TERMINATION_V3_DOMAIN.into();
        end["service_exit_success"] = true.into();
        for (key, value) in [
            ("physically_completed_role_calls", 3u64),
            ("completed_role_inputs", 3),
            ("delivered_role_inputs", 2),
            ("search_consumed_role_inputs", 1),
            ("completed_new_game_resets", 1),
            ("game_generation", 1),
            ("request_high_water", 5),
            ("execution_high_water", 3),
        ] {
            end["native"][key] = value.into();
        }
        end["native"]["physical_shutdown_confirmed"] = true.into();
        end["native"]["native_buffers_released"] = true.into();
        end["native"]["final_runtime_mapping_confirmed"] = true.into();
        end["native"]["observer_failures"] = 0.into();
        end["native"]["last_observer_failure"] = serde_json::Value::Null;
        end["native"]["backend_stats_observation"] = "exclusive_worker_before_shutdown".into();
        end["native"]["backend_stats"] = stats_fixture(2, 5);
        (start, end)
    }
    #[test]
    fn pals_v3_launch_lock_has_own_identity_and_actual_endpoint_cli() {
        let lock = fixture().lock().unwrap();
        let decoded = LockedPalsArenaLaunchV3::from_json(&lock.to_json().unwrap()).unwrap();
        assert_eq!(decoded.sha256(), lock.sha256());
        let pals = decoded.engine_view(NativeEngineRole::Baseline).unwrap();
        let cpu = decoded.engine_view(NativeEngineRole::Candidate).unwrap();
        assert!(
            pals.external
                .unwrap()
                .arguments
                .contains(&"--search=pals".into())
        );
        assert!(
            cpu.external
                .unwrap()
                .arguments
                .contains(&"--search=cpu".into())
        );
        assert!(
            !pals
                .external
                .unwrap()
                .arguments
                .iter()
                .any(|a| a.contains("onnx") || a.contains("attestation") || a.contains("weights"))
        );
        assert_eq!(decoded.expected_provider_sessions(), 0);
        let output_root = std::env::temp_dir().join("rovezero-pals-launch-mock-run");
        let arguments = decoded
            .runtime_arguments(NativeEngineRole::Baseline, &output_root)
            .unwrap();
        assert_eq!(arguments.len(), 3);
        assert!(arguments.contains(&OsString::from(format!(
            "--search-work-output-root={}",
            output_root.display()
        ))));
        assert!(
            decoded
                .preflight_arguments(NativeEngineRole::Baseline, &output_root)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            decoded.view().clock,
            NativePairClock::Game(NativeGameClockV3 {
                base_ms: 120_000,
                increment_ms: 1000
            })
        );
        let mut value: serde_json::Value = serde_json::from_str(&lock.to_json().unwrap()).unwrap();
        value["input"]["endpoints"][0]["configuration"]["max_rounds"] = 17.into();
        assert!(LockedPalsArenaLaunchV3::from_json(&value.to_string()).is_err());
    }
    #[test]
    fn pals_runner_registers_new_patch_and_binary_without_rewriting_legacy_locks() {
        let legacy = fixture();
        let old = legacy.lock().unwrap();
        assert_eq!(
            legacy.runner_status_patch().unwrap(),
            PalsRunnerStatusPatchV3::LegacyClockOnly
        );
        assert!(legacy.require_reap_status_runner().is_err());
        let mut current = legacy.clone();
        let patch = current.runner.dirty_patch.as_mut().unwrap();
        patch.sha256 = PALS_CLOCK_REAP_PATCH_SHA256.into();
        patch.bytes = PALS_CLOCK_REAP_PATCH_BYTES;
        current.runner.binary.sha256 = PALS_CLOCK_REAP_RUNNER_SHA256.into();
        current.runner.binary.bytes = PALS_CLOCK_REAP_RUNNER_BYTES;
        let new = current.lock().unwrap();
        assert!(current.require_reap_status_runner().is_ok());
        assert_ne!(old.sha256(), new.sha256());
        assert_eq!(
            LockedPalsArenaLaunchV3::from_json(&old.to_json().unwrap())
                .unwrap()
                .sha256(),
            old.sha256()
        );
        assert_eq!(
            current.semantic_lock.canonical_sha256,
            legacy.semantic_lock.canonical_sha256
        );
        let mut invalid_patch = current.clone();
        invalid_patch.runner.dirty_patch.as_mut().unwrap().bytes -= 1;
        assert!(invalid_patch.lock().is_err());
        let mut unregistered_binary = current;
        unregistered_binary.runner.binary.sha256 = "f".repeat(64);
        assert!(unregistered_binary.lock().is_ok()); // metadata-only declaration remains explicit.
        assert!(unregistered_binary.require_reap_status_runner().is_err());
    }
    #[test]
    fn pals_native_recipe_assets_and_dynamic_arguments_do_not_leak_lc0_flags() {
        let lock = native_fixture().lock().unwrap();
        assert_eq!(lock.expected_provider_sessions(), 2);
        let view = lock.engine_view(NativeEngineRole::Baseline).unwrap();
        let e = view.external.unwrap();
        assert!(
            e.arguments
                .contains(&"--pals-export-manifest={{asset:0}}".into())
        );
        assert_eq!(e.assets.len(), 5);
        assert!(!e.arguments.iter().any(|a| a.starts_with("--onnx-")
            || a == "--attestation"
            || a.starts_with("--source-weights")));
        for a in &e.assets[2..] {
            assert!(lock.snapshot_relative_path(a).unwrap().starts_with("pals-"));
        }
        let output_root = std::env::temp_dir().join("rovezero-pals-launch-native-runtime");
        let other_output_root = std::env::temp_dir().join("rovezero-pals-launch-second-attempt");
        let arguments = lock
            .runtime_arguments(NativeEngineRole::Baseline, &output_root)
            .unwrap();
        assert_eq!(arguments.len(), 4);
        let preflight = lock
            .preflight_arguments(NativeEngineRole::Baseline, &output_root)
            .unwrap();
        assert_eq!(preflight, vec![pals_shared_runtime_argument().unwrap()]);
        let other_runtime = lock
            .runtime_arguments(NativeEngineRole::Baseline, &other_output_root)
            .unwrap();
        let other_preflight = lock
            .preflight_arguments(NativeEngineRole::Baseline, &other_output_root)
            .unwrap();
        assert_eq!(arguments.last(), other_runtime.last());
        assert_eq!(preflight, other_preflight);
        let cache_argument = arguments.last().unwrap().to_str().unwrap();
        assert!(cache_argument.starts_with("--pals-runtime-cache-root="));
        assert!(
            !cache_argument.contains(output_root.to_str().unwrap())
                && !cache_argument.contains("second-attempt")
        );
    }
    #[test]
    fn pals_cuda_recipe_pins_nineteen_libraries_without_cpu_or_lc0_fallback() {
        let (f, descriptor) = cuda_fixture();
        let lock = f.lock().unwrap();
        assert_eq!(lock.expected_provider_sessions(), 2);
        lock.validate_cuda_bundle(NativeEngineRole::Baseline, &descriptor)
            .unwrap();
        let view = lock.engine_view(NativeEngineRole::Baseline).unwrap();
        assert_eq!(view.cuda_bundle.unwrap().files.len(), 19);
        let endpoint = view.external.unwrap();
        assert_eq!(endpoint.assets.len(), 23); // export, core, two graphs, descriptor, eighteen sibling libs.
        assert!(endpoint.arguments.contains(&"--pals-provider=cuda".into()));
        assert!(
            endpoint
                .arguments
                .contains(&"--pals-cuda-bundle={{asset:4}}".into())
        );
        assert!(
            endpoint
                .arguments
                .contains(&"--pals-device-public-memory=false".into())
        );
        assert!(
            !endpoint
                .arguments
                .iter()
                .any(|arg| arg == "--pals-provider=cpu"
                    || arg.starts_with("--onnx-")
                    || arg.starts_with("--source-weights"))
        );
        let serialized = lock.to_json().unwrap();
        assert_eq!(
            LockedPalsArenaLaunchV3::from_json(&serialized)
                .unwrap()
                .sha256(),
            lock.sha256()
        );
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCuda(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.cuda_bundle.files.pop();
        assert!(bad.lock().is_err());
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCuda(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.model.runtime.sha256 = "f".repeat(64);
        assert!(bad.lock().is_err());
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCuda(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.device_id = 1;
        assert!(bad.lock().is_err());
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCuda(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.model.graphs = native_fixture().endpoints[0]
            .native_model()
            .unwrap()
            .graphs
            .clone();
        assert!(bad.lock().is_err());
        let mut bad = f;
        let PalsEngineV3::Pals(e) = &mut bad.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model.backend = PalsModelBackendV3::OrtCpu;
        bad.semantic_lock = bad.semantic_lock.manifest.lock().unwrap();
        assert!(bad.lock().is_err());
        let mut wrong = descriptor;
        wrong[0] = b'[';
        assert!(
            lock.validate_cuda_bundle(NativeEngineRole::Baseline, &wrong)
                .is_err()
        );
    }
    #[test]
    fn pals_cuda_startup_work_is_not_search_work_or_cuda_placement_evidence() {
        let (f, _) = cuda_fixture();
        let lock = f.lock().unwrap();
        let (start, end) = cuda_records_fixture(&lock);
        let validate = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        let (audit, pid) = validate(&start, &end).unwrap();
        assert_eq!(pid, 100);
        assert_eq!(
            (
                audit.startup_nn_inputs_completed,
                audit.startup_nn_calls_completed,
                audit.startup_role_inputs_completed
            ),
            (3, 3, 2)
        );
        assert_eq!(
            (
                audit.completed_role_inputs,
                audit.search_consumed_role_inputs
            ),
            (3, 1)
        );
        let mut work = work_fixture("pals");
        work["pals"]["consumed_role_outputs"] = 1.into();
        work["pals"]["completed_proposer_calls"] = 1.into();
        let search_work = PalsProcessWorkAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 100,
            startup_sha256: "a".repeat(64),
            termination_sha256: "b".repeat(64),
            startup_bytes: 100,
            termination_bytes: 200,
            search_work: work,
        };
        let receipt = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&search_work),
            Some(&audit),
            &preflight_fixture("pals"),
        )
        .unwrap();
        assert_eq!(
            (
                receipt.nn_inputs_completed,
                receipt.nn_calls_completed,
                receipt.nn_inputs_consumed
            ),
            (4, 4, 1)
        );
        assert_eq!(receipt.gpu_device, PalsObservedV3::Unknown);
        let mut incomplete_snapshot = audit.clone();
        incomplete_snapshot.raw_native["backend_stats"] = stats_fixture(0, 1);
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&search_work),
                Some(&incomplete_snapshot),
                &preflight_fixture("pals")
            )
            .is_err()
        );
        let mut failed_observer = end.clone();
        failed_observer["native"]["observer_failures"] = 1.into();
        assert!(validate(&start, &failed_observer).is_err());
        let mut bad = end.clone();
        bad["runtime_bundle_sha256"] = "f".repeat(64).into();
        assert!(validate(&start, &bad).is_err());
        let mut bad_start = start.clone();
        let mut bad_end = end.clone();
        bad_start["native"]["startup_probe"]["runtime_mapping_confirmed"] = false.into();
        bad_end["native"]["startup_probe"] = bad_start["native"]["startup_probe"].clone();
        assert!(validate(&bad_start, &bad_end).is_err());
        let mut bad_start = start.clone();
        let mut bad_end = end;
        bad_start["native"]["execution"]["transient_execution_device_bytes"] = (3u64 << 30).into();
        bad_end["native"]["execution"] = bad_start["native"]["execution"].clone();
        assert!(validate(&bad_start, &bad_end).is_err());
    }
    #[test]
    fn pals_launch_rejects_unsupported_options_small_tt_and_secret_inputs() {
        let mut f = fixture();
        let PalsEngineV3::Pals(e) = &mut f.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.search.options.insert("beam".into(), "100".into());
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        assert!(f.lock().is_err());
        let mut f = fixture();
        let PalsEngineV3::OwnCpu(e) = &mut f.semantic_lock.manifest.engines[1] else {
            unreachable!()
        };
        e.cpu.max_tt_bytes = 1;
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        assert!(f.lock().is_err());
        let mut f = fixture();
        f.opening_artifact.path = "secret/ssh.key".into();
        assert!(f.lock().is_err());
        let mut f = native_fixture();
        let PalsEndpointLaunchV3::OnnxCpu(n) = &mut f.endpoints[0] else {
            unreachable!()
        };
        n.export.bytes = 64 * 1024 + 1;
        assert!(f.lock().is_err());
    }
    #[test]
    fn pals_launch_rejects_same_source_graph_in_two_export_namespaces() {
        let mut f = native_fixture();
        let mut e = f.semantic_lock.manifest.engines[0].clone();
        let PalsEngineV3::Pals(p) = &mut e else {
            unreachable!()
        };
        p.id = "second-pals".into();
        f.semantic_lock.manifest.engines[1] = e;
        f.semantic_lock.manifest.pilot.white_order[1] = "second-pals".into();
        f.semantic_lock.manifest.declared_changes.clear();
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        f.endpoints[1] = f.endpoints[0].clone();
        let PalsEndpointLaunchV3::OnnxCpu(n) = &mut f.endpoints[1] else {
            unreachable!()
        };
        n.export.path = "model/second-export.json".into();
        n.export.sha256 = "e".repeat(64);
        assert!(f.lock().is_err());
    }
    #[test]
    fn pals_native_digest_and_resource_parsing_fail_closed() {
        assert!(array_hash(&serde_json::json!([0, 1])).is_err());
        assert!(
            json(br#"{"native":{"completed_role_inputs":1,"completed_role_inputs":2}}"#).is_err()
        );
        assert_eq!(parse_pals_affinity("2,0").unwrap(), vec![0, 2]);
        assert_eq!(parse_pals_affinity("0-2,4").unwrap(), vec![0, 1, 2, 4]);
        for v in ["", "0,0", "2-1", "0-1000", "0-2,1"] {
            assert!(parse_pals_affinity(v).is_err());
        }
    }
    #[test]
    fn pals_native_records_require_exact_identity_fence_and_real_consumption() {
        let lock = native_fixture().lock().unwrap();
        let graph_hash = [170u8; 32];
        let encoding_hash = [221u8; 32];
        let adapter_hash = [187u8; 32];
        let zero = serde_json::json!({"physically_completed_role_calls":0,"completed_role_inputs":0,"failed_physical_role_calls":0,"invalid_role_outputs":0,"delivered_role_inputs":0,"search_consumed_role_inputs":0,"canceled_requests":0,"expired_requests":0,"completed_new_game_resets":0,"process_epoch":1,"game_generation":0,"request_high_water":0,"execution_high_water":0,"physical_runs_in_flight":0,"quarantined":false,"physical_shutdown_confirmed":false,"native_buffers_released":false,"last_failure":null,
            "model_epoch":graph_hash,"export_manifest_sha256":graph_hash,"encoding_semantic_sha256":encoding_hash,"adapter_source_sha256":adapter_hash,"trained":false,
            "frozen_epoch":1,"residency":{"native_sessions":3,"role_reader_weights_shared":false,"native_resident_parameter_bytes":null,"vram_peak_bytes":null,"graphs":[{"role":"public","sha256":graph_hash,"serialized_bytes":1000},{"role":"proposer","sha256":graph_hash,"serialized_bytes":1000},{"role":"critic","sha256":graph_hash,"serialized_bytes":1000}]}});
        let start = serde_json::json!({"schema_version":3,"domain":PALS_NATIVE_STARTUP_V3_DOMAIN,"endpoint_id":"pals","launch_sha256":lock.sha256(),"process_id":100,"binary_sha256":"a".repeat(64),"runtime_sha256":"a".repeat(64),"provider":"cpu","precision":"fp32","service_exit_success":false,"native":zero});
        let mut end = start.clone();
        end["domain"] = PALS_NATIVE_TERMINATION_V3_DOMAIN.into();
        end["service_exit_success"] = true.into();
        for (key, value) in [
            ("physically_completed_role_calls", 3u64),
            ("completed_role_inputs", 3),
            ("delivered_role_inputs", 2),
            ("search_consumed_role_inputs", 1),
            ("completed_new_game_resets", 1),
            ("game_generation", 1),
            ("request_high_water", 5),
            ("execution_high_water", 3),
        ] {
            end["native"][key] = value.into();
        }
        end["native"]["physical_shutdown_confirmed"] = true.into();
        end["native"]["native_buffers_released"] = true.into();
        let validate = |v: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&start).unwrap(),
                &serde_json::to_vec(v).unwrap(),
                "native-process-100",
            )
        };
        let (audit, pid) = validate(&end).unwrap();
        assert_eq!(pid, 100);
        assert_eq!(audit.completed_role_inputs, 3);
        assert_eq!(audit.search_consumed_role_inputs, 1);
        assert!(lock.input.require_actual_native_epoch().is_ok());
        let mut historical = native_fixture();
        let PalsEngineV3::Pals(e) = &mut historical.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model.frozen_epoch = 0;
        historical.semantic_lock = historical.semantic_lock.manifest.lock().unwrap();
        let old = historical.lock().unwrap();
        assert!(old.input.require_actual_native_epoch().is_err());
        let mut old_start = start.clone();
        let mut old_end = end.clone();
        old_start["launch_sha256"] = old.sha256().into();
        old_end["launch_sha256"] = old.sha256().into();
        old_start["native"]
            .as_object_mut()
            .unwrap()
            .remove("frozen_epoch");
        old_end["native"]
            .as_object_mut()
            .unwrap()
            .remove("frozen_epoch");
        assert!(
            validate_pals_native_records(
                &old,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&old_start).unwrap(),
                &serde_json::to_vec(&old_end).unwrap(),
                "native-process-100"
            )
            .is_ok()
        );
        assert_eq!(
            LockedPalsArenaLaunchV3::from_json(&old.to_json().unwrap())
                .unwrap()
                .sha256(),
            old.sha256()
        );
        let mut missing_start = start.clone();
        let mut missing_end = end.clone();
        missing_start["native"]
            .as_object_mut()
            .unwrap()
            .remove("frozen_epoch");
        missing_end["native"]
            .as_object_mut()
            .unwrap()
            .remove("frozen_epoch");
        assert!(
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&missing_start).unwrap(),
                &serde_json::to_vec(&missing_end).unwrap(),
                "native-process-100"
            )
            .is_err()
        );
        let mut bad = end.clone();
        bad["native"]["search_consumed_role_inputs"] = 4.into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["native"]["physical_runs_in_flight"] = 1.into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["native"]["physical_shutdown_confirmed"] = false.into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["native"]["quarantined"] = true.into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["runtime_sha256"] = "f".repeat(64).into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["native"]["execution"] = serde_json::json!({"provider":"cuda"});
        assert!(validate(&bad).is_err());
        let mut bad = end;
        bad["native"]["execution"] = "not an object".into();
        assert!(validate(&bad).is_err());
    }
    fn work_fixture(kind: &str) -> serde_json::Value {
        let totals = if kind == "pals" {
            serde_json::json!({"completed_proposer_calls":0,"completed_repair_calls":0,"completed_critic_calls":0,"consumed_role_outputs":0,
                "cpu_tasks_requested":0,"completed_cpu_tasks":0,"reused_completed_cpu_tasks_consumed":0,
                "consumed_cpu_tasks":0,"cpu_nodes":0,"consumed_cached_cpu_values":0,"unknown_root_children":null})
        } else {
            serde_json::json!({"tasks_requested":0,"requested_depth_completed":0,"rules_terminal_reports":0,
                "completed_reports_accepted_for_uci_output":0,"nodes":0,"max_completed_depth":null})
        };
        let mut work = serde_json::json!({"schema_version":1,"search_kind":kind,"go_invocations":0,"successful_returns":0,
            "failed_returns":0,"canceled_returns":0,"deadline_returns":0,"physical_unknown_returns":0,"active_invocations":0,
            "unobserved_work_invocations":0,"cpu":null,"pals":null});
        work[kind] = totals;
        work
    }
    fn preflight_fixture(id: &str) -> crate::ExternalUciPreflight {
        let process: crate::ProcessReceipt = serde_json::from_value(serde_json::json!({"supervisor_version":1,"pid":1,
            "elapsed_ns":1,"stop":"exited","exit_code":0,"exit_signal":null,"group_cleanup":"gone","stdout_bytes":1,"stderr_bytes":0,
            "observed_output_bytes":1,"descendant_cleanup_required":false,"child_limit_enforcement":"fixture-only",
            "watched_artifact_bytes":1,"artifact_limit_enforcement":null,"errors":[]})).unwrap();
        crate::ExternalUciPreflight {
            schema_version: 1,
            engine_id: id.into(),
            advertisement: crate::parse_uci_advertisement(b"id name fixture\nuciok\n").unwrap(),
            options: BTreeMap::new(),
            environment: None,
            identification_process: process.clone(),
            readiness_process: process,
            stop_and_legal_bestmove_observed: true,
            compiler_information: vec![],
            selected_isa: None,
            scope: "independent test wire fixture",
        }
    }
    #[test]
    fn pals_work_records_reject_missing_unknown_and_cross_process_observations() {
        let lock = fixture().lock().unwrap();
        let work = work_fixture("pals");
        let start = serde_json::json!({"schema_version":3,"domain":PALS_SEARCH_WORK_STARTUP_V3_DOMAIN,
            "endpoint_id":"pals","launch_sha256":lock.sha256(),"process_id":101,"binary_sha256":"a".repeat(64),
            "service_exit_success":false,"search_work":work});
        let mut end = start.clone();
        end["domain"] = PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN.into();
        end["service_exit_success"] = true.into();
        end["search_work"]["go_invocations"] = 1.into();
        end["search_work"]["successful_returns"] = 1.into();
        let verify = |end: &serde_json::Value| {
            validate_pals_search_work_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&start).unwrap(),
                &serde_json::to_vec(end).unwrap(),
                "native-process-101",
            )
        };
        assert!(verify(&end).is_ok()); // unknown root gauge does not erase observed work.
        for (key, value) in [
            ("active_invocations", 1u64),
            ("unobserved_work_invocations", 1),
            ("physical_unknown_returns", 1),
        ] {
            let mut bad = end.clone();
            bad["search_work"][key] = value.into();
            assert!(verify(&bad).is_err());
        }
        let mut bad = end.clone();
        bad["process_id"] = 102.into();
        assert!(verify(&bad).is_err());
        let mut bad = end.clone();
        bad["search_work"]["pals"]["cpu_nodes"] = serde_json::Value::Null;
        assert!(verify(&bad).is_err());
        let mut bad = end;
        bad["search_work"]["successful_returns"] = 2.into();
        assert!(verify(&bad).is_err());
    }

    #[test]
    fn additive_resolver_identity_checks_semantics_without_rewriting_legacy_work() {
        let mut work = work_fixture("pals");
        assert!(validate_pals_process_work(&work, "pals", true).is_ok());
        assert!(work.get("pals_resolver").is_none());
        work["pals_resolver"] = serde_json::json!({
            "version": rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
            "semantics_sha256": Sha256::digest(
                rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS.as_bytes()
            ).to_vec()
        });
        assert!(validate_pals_process_work(&work, "pals", true).is_ok());
        let mut cpu = work_fixture("cpu");
        cpu["pals_resolver"] = work["pals_resolver"].clone();
        assert!(validate_pals_process_work(&cpu, "cpu", true).is_err());
        work["pals_resolver"]["semantics_sha256"][0] = 256.into();
        assert!(validate_pals_process_work(&work, "pals", true).is_err());
    }
    #[test]
    fn pals_core_projection_separates_cpu_evidence_reuse_from_nn_and_actual_consumption() {
        let lock = fixture().lock().unwrap();
        let mut work = work_fixture("pals");
        for (field, value) in [
            ("cpu_tasks_requested", 1u64),
            ("completed_cpu_tasks", 1),
            ("reused_completed_cpu_tasks_consumed", 2),
            ("consumed_cpu_tasks", 3),
            ("consumed_cached_cpu_values", 4),
            ("completed_proposer_calls", 2),
            ("completed_repair_calls", 1),
            ("completed_critic_calls", 2),
        ] {
            work["pals"][field] = value.into();
        }
        let audit = PalsProcessWorkAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 1,
            startup_sha256: "a".repeat(64),
            termination_sha256: "a".repeat(64),
            startup_bytes: 100,
            termination_bytes: 200,
            search_work: work,
        };
        let receipt = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&audit),
            None,
            &preflight_fixture("pals"),
        )
        .unwrap();
        assert_eq!(
            (
                receipt.cpu_tasks_completed,
                receipt.cpu_tasks_reused_consumed,
                receipt.cpu_tasks_consumed
            ),
            (1, 2, 3)
        );
        assert_eq!(
            (
                receipt.nn_inputs_completed,
                receipt.cached_evaluations_consumed
            ),
            (0, 0)
        );
        assert_eq!(
            (
                receipt.proposer_tasks_completed,
                receipt.critic_tasks_completed
            ),
            (3, 2)
        );
        let mut cpu = work_fixture("cpu");
        cpu["cpu"]["tasks_requested"] = 2.into();
        cpu["cpu"]["requested_depth_completed"] = 2.into();
        cpu["cpu"]["completed_reports_accepted_for_uci_output"] = 1.into();
        let cpu_audit = PalsProcessWorkAuditV3 {
            endpoint_id: "cpu".into(),
            search_work: cpu,
            ..audit
        };
        let receipt = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Candidate,
            Some(&cpu_audit),
            None,
            &preflight_fixture("cpu"),
        )
        .unwrap();
        assert_eq!(
            (receipt.cpu_tasks_completed, receipt.cpu_tasks_consumed),
            (2, 1)
        );
        let mut missing = cpu_audit;
        missing.search_work["cpu"]
            .as_object_mut()
            .unwrap()
            .remove("completed_reports_accepted_for_uci_output");
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Candidate,
                Some(&missing),
                None,
                &preflight_fixture("cpu")
            )
            .is_err()
        );
    }
    #[test]
    fn pals_core_nn_completion_includes_public_graph_without_counting_cache_reuse() {
        let lock = native_fixture().lock().unwrap();
        let mut work = work_fixture("pals");
        work["pals"]["consumed_role_outputs"] = 1.into();
        work["pals"]["completed_proposer_calls"] = 1.into();
        let audit = PalsProcessWorkAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 10,
            startup_sha256: "a".repeat(64),
            termination_sha256: "b".repeat(64),
            startup_bytes: 100,
            termination_bytes: 200,
            search_work: work,
        };
        let native = PalsNativeSessionAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 10,
            launch_sha256: lock.sha256().into(),
            startup_sha256: "a".repeat(64),
            termination_sha256: "b".repeat(64),
            completed_role_inputs: 3,
            search_consumed_role_inputs: 1,
            completed_new_game_resets: 1,
            physical_shutdown_confirmed: true,
            native_buffers_released: true,
            startup_nn_inputs_completed: 0,
            startup_nn_calls_completed: 0,
            startup_role_inputs_completed: 0,
            startup_probe: None,
            execution: None,
            raw_native: serde_json::json!({"backend_stats_observation":"exclusive_worker_before_shutdown","observer_failures":0,"last_observer_failure":null,"frozen_epoch":1,
                "backend_stats":{"public_nn_runs_completed":1,"role_nn_runs_completed":3,"completed_nn_inputs":4,"public_cache_hits":2,
                    "public_nn_runs_failed_known":0,"role_nn_runs_failed_known":0,"public_nn_runs_attempted":1,"role_nn_runs_attempted":3,
                    "validated_public_outputs":1,"validated_role_outputs":3,"live_public_cache_entries":1}}),
            raw_search_work: audit.search_work.clone(),
        };
        let receipt = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&audit),
            Some(&native),
            &preflight_fixture("pals"),
        )
        .unwrap();
        assert_eq!(
            (
                receipt.nn_inputs_completed,
                receipt.nn_calls_completed,
                receipt.nn_inputs_consumed
            ),
            (4, 4, 1)
        );
        assert_eq!(receipt.cached_evaluations_consumed, 0);
        let mut missing_observer = native.clone();
        missing_observer
            .raw_native
            .as_object_mut()
            .unwrap()
            .remove("observer_failures");
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&audit),
                Some(&missing_observer),
                &preflight_fixture("pals")
            )
            .is_err()
        );
        let mut unknown = native;
        unknown.raw_native["backend_stats"] = serde_json::Value::Null;
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&audit),
                Some(&unknown),
                &preflight_fixture("pals")
            )
            .is_err()
        );
    }
    #[test]
    fn pals_shared_pc_recipe_uses_exact_two_graph_layout() {
        let mut f = native_fixture();
        let PalsEndpointLaunchV3::OnnxCpu(n) = &mut f.endpoints[0] else {
            unreachable!()
        };
        n.graphs.retain(|g| g.role == "public");
        n.graphs.push(PalsGraphAssetV3 {
            file: "shared_pc_if.onnx".into(),
            role: "shared_pc".into(),
            artifact: asset("model/shared_pc_if.onnx"),
        });
        let lock = f.lock().unwrap();
        assert_eq!(lock.expected_provider_sessions(), 2); // process count, not graph/session count.
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCpu(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.graphs[1].role = "validator".into();
        assert!(bad.lock().is_err());
    }
    #[test]
    fn pals_game_process_mapping_requires_game_color_and_synchronous_exit_windows() {
        let lock = fixture().lock().unwrap();
        let trace = |message: &str| {
            format!(
                "[TRACE ] [12:34:56.123456] <{:>20}> fastchess --- {message}",
                1
            )
        };
        let exit = |pid| {
            trace(&format!(
                "Process with pid: {pid} terminated with status: 0"
            ))
        };
        let rows = vec![
            "[Engine] fixture pals ---> Started game 1 of 2 (pals vs cpu)".into(),
            trace("Game 1 between pals and cpu starting"),
            "Started game 1 of 2 (pals vs cpu)".into(),
            trace("Game 1 between pals and cpu finished"),
            trace("Game 1 finished with result 0-1"),
            "Finished game 1 (pals vs cpu): 0-1 {Black wins}".into(),
            exit(101),
            exit(201),
            trace("Game 2 between cpu and pals starting"),
            "Started game 2 of 2 (cpu vs pals)".into(),
            trace("Game 2 between cpu and pals finished"),
            trace("Game 2 finished with result 1-0"),
            "Finished game 2 (cpu vs pals): 1-0 {White wins}".into(),
            exit(202),
            exit(102),
        ];
        let pids = [BTreeSet::from([101, 102]), BTreeSet::from([201, 202])];
        let encode = |rows: &[String]| format!("{}\n", rows.join("\n")).into_bytes();
        assert_eq!(
            validate_pals_game_process_trace(&lock, &encode(&rows), &pids).unwrap(),
            [[101, 102], [201, 202]]
        );
        let mut swapped = rows.clone();
        swapped.swap(6, 7);
        assert!(validate_pals_game_process_trace(&lock, &encode(&swapped), &pids).is_err());
        let mut missing = rows.clone();
        missing.remove(7);
        assert!(validate_pals_game_process_trace(&lock, &encode(&missing), &pids).is_err());
        let mut early = rows.clone();
        early.swap(7, 8);
        assert!(validate_pals_game_process_trace(&lock, &encode(&early), &pids).is_err());
        let mut failed = rows.clone();
        failed[6] = trace("Process with pid: 101 terminated with status: 256");
        assert!(validate_pals_game_process_trace(&lock, &encode(&failed), &pids).is_err());
        let mut unknown = rows;
        unknown.remove(3);
        assert!(validate_pals_game_process_trace(&lock, &encode(&unknown), &pids).is_err());
    }
}
