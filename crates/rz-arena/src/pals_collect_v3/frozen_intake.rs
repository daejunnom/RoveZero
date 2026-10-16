//! Read-only named-P/C content intake, issued by the original process owner.
//!
//! A proof JSON is not an authority. Only this private producer, after running
//! the compiled verifier against held files and joining the borrowed native
//! backend, may arm `FrozenPcRegistrationV3`. This proves tensor/export content
//! and loaded asset identity, not optimizer, numerical or arena acceptance.
// The public non-Linux entry point is typed unsupported. Keeping the private
// codecs available there permits CPU-only contract tests without an issuer.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]
use super::{ActionOutcomeCollectionPlanV3, FrozenPcRegistrationV3, check, hex, invalid};
use crate::ArenaError;
use crate::process::{OriginalProcessOutput, ProcessLimits};
use rz_eval::pals_model::{PalsModelConfig, PalsModelProfile};
use rz_eval::pals_onnx::virtual_review::PalsVirtualReviewOnnxBackend;
use rz_experiments::pals_data_v3::{generation_digest, generation_identifier, generation_sha};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::AtomicBool;
#[cfg(target_os = "linux")]
use std::sync::atomic::Ordering;
use std::time::Instant;

const REQUEST_DOMAIN: &str = "rz-pals-frozen-pc-content-request/1";
const PROOF_DOMAIN: &str = "rz-pals-frozen-pc-content-proof/1";
const INVENTORY_DOMAIN: &str = "rz-pals-frozen-pc-parameter-inventory/1";
const MAPPING_DOMAIN: &str = "rz-pals-frozen-pc-initializer-mapping/1";
const INTAKE_DOMAIN: &str = "rz-pals-native-frozen-pc-verified-intake/1";
const CHECKPOINT_DOMAIN: &str = "rz-pals-actual-generation-checkpoint/1";
const MODEL_DOMAIN: &str = "rovezero.pals-model-semantics.v3";
const PROOF_FILE: &str = "frozen-pc-content-proof.json";
const MAX_REQUEST: u64 = 128 * 1024;
const MAX_PROOF: u64 = 4 * 1024 * 1024;
const MAX_SOURCE: u64 = 2 * 1024 * 1024;
const SOURCE_PREFIX: &str = "experiments/model-research/pals/src/rz_pals_model/";
const SOURCE_NAMES: [&str; 13] = [
    "config.py",
    "model.py",
    "onnx_shared.py",
    "onnx_virtual_review.py",
    "nonzero_training.py",
    "generation_training.py",
    "generation_export.py",
    "generation_frozen_pc.py",
    "generation_data.py",
    "virtual_review.py",
    "artifacts.py",
    "training.py",
    "generation_chunks.py",
];

/// File content declaration. Actual authority remains the held descriptor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPcFilePinV3 {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
}
impl FrozenPcFilePinV3 {
    fn validate(&self, maximum: u64) -> Result<(), ArenaError> {
        check(
            relative_file(&self.file)
                && generation_sha(&self.sha256)
                && self.bytes > 0
                && self.bytes <= maximum,
            "frozen intake bounded relative file pin",
        )
    }
}

/// Explicit execution declaration, owning the original request and interpreter
/// handles. `dependency_directories` are registered interpreter library roots;
/// -S bypasses site/.pth execution and -I bypasses inherited Python paths.
/// This is a trusted local verifier invocation, not a security sandbox or a
/// content verification of all installed Torch/ONNX dependencies.
pub struct FrozenPcVerifierLaunchV3 {
    pub interpreter: File,
    pub interpreter_sha256: String,
    pub interpreter_bytes: u64,
    pub request: File,
    pub request_sha256: String,
    pub request_bytes: u64,
    pub registered_source_sha256: String,
    pub dependency_directories: Vec<PathBuf>,
    pub artifact_directory: PathBuf,
    /// Caller-created, exclusively owned empty 0700 directory outside Git and
    /// artifact roots. It is preserved on success and failure for reclamation.
    pub scratch_directory: PathBuf,
    pub started: Instant,
    pub execution_deadline: Instant,
    pub whole_deadline: Instant,
    pub limits: ProcessLimits,
}

/// Successful private production. No deserialization or public successful
/// constructor exists. Source/artifact handles and original child evidence are
/// kept until the integration owner has preserved the receipt and reclaims them.
pub struct VerifiedFrozenPcIntakeV3 {
    registration: FrozenPcRegistrationV3,
    receipt: serde_json::Value,
    proof: Vec<u8>,
    process: OriginalProcessOutput,
    _launch: FrozenPcVerifierLaunchV3,
    #[cfg(target_os = "linux")]
    _held: LinuxHeld,
}
impl VerifiedFrozenPcIntakeV3 {
    pub fn registration(&self) -> &FrozenPcRegistrationV3 {
        &self.registration
    }
    pub fn receipt(&self) -> &serde_json::Value {
        &self.receipt
    }
    pub fn proof_bytes(&self) -> &[u8] {
        &self.proof
    }
    pub fn process(&self) -> &OriginalProcessOutput {
        &self.process
    }
}

/// Keeps pending-child custody and held assets on every post-spawn failure.
/// A caller must inspect/resolve `process.pending_child` through the existing
/// process owner before dropping quarantine. No kill/retry is implicit here.
pub struct FrozenPcIntakeFailureV3 {
    pub error: ArenaError,
    pub process: Option<OriginalProcessOutput>,
    pub launch: FrozenPcVerifierLaunchV3,
    #[cfg(target_os = "linux")]
    _held: Option<LinuxHeld>,
}
impl std::fmt::Debug for FrozenPcIntakeFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenPcIntakeFailureV3")
            .field("error", &self.error)
            .field("process", &self.process)
            .finish_non_exhaustive()
    }
}

/// Source pins to use when constructing the independently registered request.
/// They are derived from the compiled source inventory, never caller JSON.
pub fn compiled_frozen_pc_verifier_pins_v3() -> Result<Vec<FrozenPcFilePinV3>, ArenaError> {
    SOURCE_NAMES
        .into_iter()
        .map(|name| {
            let bytes = compiled_source(name)?;
            Ok(FrozenPcFilePinV3 {
                file: name.to_owned(),
                sha256: hex(&Sha256::digest(bytes)),
                bytes: bytes.len() as u64,
            })
        })
        .collect()
}
fn compiled_source(name: &str) -> Result<&'static [u8], ArenaError> {
    let path = format!("{SOURCE_PREFIX}{name}");
    let mut matches = rz_experiments::SOURCE_COMPONENT_FILES_V5
        .iter()
        .filter(|(component, _)| *component == "model_reference")
        .flat_map(|(_, files)| files.iter())
        .filter(|(file, _)| *file == path);
    let bytes = matches.next().map(|(_, bytes)| *bytes).ok_or_else(|| {
        invalid("compiled frozen verifier source absent from model_reference inventory")
    })?;
    check(
        matches.next().is_none() && !bytes.is_empty() && bytes.len() as u64 <= MAX_SOURCE,
        "compiled frozen verifier source duplicated or exceeds cap",
    )?;
    Ok(bytes)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointPin {
    file: String,
    sha256: String,
    bytes: u64,
    domain: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphPin {
    file: String,
    sha256: String,
    bytes: u64,
    role: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContentLimits {
    checkpoint_bytes: u64,
    graph_bytes: u64,
    tensor_bytes: u64,
    proof_bytes: u64,
    wall_seconds: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContentRequest {
    schema: String,
    source_sha256: String,
    model_epoch: u64,
    config: PalsModelConfig,
    checkpoint: CheckpointPin,
    graphs: Vec<GraphPin>,
    export_manifest: FrozenPcFilePinV3,
    verifier_sources: Vec<FrozenPcFilePinV3>,
    expected_frozen_pc_sha256: Option<String>,
    limits: ContentLimits,
}
impl ContentRequest {
    fn validate(
        &self,
        plan: &ActionOutcomeCollectionPlanV3,
        source: &str,
    ) -> Result<(), ArenaError> {
        plan.validate()?;
        check(
            self.schema == REQUEST_DOMAIN
                && generation_sha(source)
                && self.source_sha256 == source
                && self.model_epoch == plan.frozen_pc_epoch
                && self.expected_frozen_pc_sha256.as_deref()
                    == Some(plan.frozen_pc_weights_sha256.as_str())
                && self.config == PalsModelConfig::for_profile(PalsModelProfile::VirtualReviewV1),
            "frozen request source, named parameter epoch or registered V3 configuration differs",
        )?;
        check(
            self.limits.checkpoint_bytes > 0
                && self.limits.checkpoint_bytes <= 1024 * 1024 * 1024
                && self.limits.graph_bytes > 0
                && self.limits.graph_bytes <= 256 * 1024 * 1024
                && self.limits.tensor_bytes > 0
                && self.limits.tensor_bytes <= 128 * 1024 * 1024
                && self.limits.proof_bytes > 0
                && self.limits.proof_bytes <= MAX_PROOF
                && self.limits.wall_seconds > 0
                && self.limits.wall_seconds <= 300,
            "frozen request finite content/decode limits",
        )?;
        let checkpoint = self.checkpoint.pin();
        checkpoint.validate(self.limits.checkpoint_bytes)?;
        check(
            [CHECKPOINT_DOMAIN, MODEL_DOMAIN].contains(&self.checkpoint.domain.as_str())
                && self.checkpoint.file.ends_with(".pt"),
            "frozen checkpoint explicit tensor domain",
        )?;
        self.export_manifest.validate(MAX_PROOF)?;
        check(
            Path::new(&self.export_manifest.file)
                .file_name()
                .and_then(|s| s.to_str())
                == Some("export.json")
                && self.graphs.len() == 2,
            "frozen original export manifest and graph pair",
        )?;
        let parent = Path::new(&self.export_manifest.file)
            .parent()
            .unwrap_or(Path::new(""));
        let mut paths = BTreeSet::from([
            self.checkpoint.file.as_str(),
            self.export_manifest.file.as_str(),
        ]);
        for (graph, (role, file)) in self.graphs.iter().zip([
            ("public", "public_memory.onnx"),
            ("shared_pc", "shared_pc_if.onnx"),
        ]) {
            graph.pin().validate(self.limits.graph_bytes)?;
            check(
                graph.role == role
                    && Path::new(&graph.file) == parent.join(file)
                    && paths.insert(&graph.file),
                "frozen Fresh graph filename, order or alias",
            )?;
        }
        check(
            self.verifier_sources == compiled_frozen_pc_verifier_pins_v3()?,
            "frozen verifier/builder bytes differ from this executable's compiled source",
        )
    }
    fn digest(&self) -> Result<String, ArenaError> {
        generation_digest(REQUEST_DOMAIN, self).map_err(|e| invalid(e.to_string()))
    }
}
impl CheckpointPin {
    fn pin(&self) -> FrozenPcFilePinV3 {
        FrozenPcFilePinV3 {
            file: self.file.clone(),
            sha256: self.sha256.clone(),
            bytes: self.bytes,
        }
    }
}
impl GraphPin {
    fn pin(&self) -> FrozenPcFilePinV3 {
        FrozenPcFilePinV3 {
            file: self.file.clone(),
            sha256: self.sha256.clone(),
            bytes: self.bytes,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildSummary {
    schema: String,
    file: String,
    sha256: String,
    bytes: u64,
    request_sha256: String,
    frozen_pc_parameter_sha256: String,
    optimizer_content_verified: bool,
    numerical_acceptance: bool,
    native_loaded_assets_verified: bool,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VerificationScope {
    checkpoint_tensor_content_verified: bool,
    export_initializer_content_verified: bool,
    manual_graph_structure_verified: bool,
    optimizer_content_verified: bool,
    numerical_acceptance: bool,
    native_loaded_assets_verified: bool,
    arena_eligible: bool,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterEntry {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    owner: String,
    semantics: String,
    bytes: u64,
    sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InitializerEntry {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    bytes: u64,
    sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphProof {
    role: String,
    file: String,
    sha256: String,
    bytes: u64,
    structure_sha256: String,
    initializers: Vec<InitializerEntry>,
    initializer_inventory_sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappingEntry {
    graph_role: String,
    parameter: String,
    initializer: String,
    transform: String,
    source_sha256: String,
    initializer_sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContentProof {
    schema: String,
    request: ContentRequest,
    request_sha256: String,
    scope: String,
    checkpoint_scope: String,
    frozen_pc_parameter_sha256: String,
    parameter_inventory: Vec<ParameterEntry>,
    parameter_inventory_sha256: String,
    graphs: Vec<GraphProof>,
    mapping: Vec<MappingEntry>,
    mapping_sha256: String,
    encoding_registry_sha256: String,
    verification: VerificationScope,
}
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, ArenaError> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("frozen evidence requires UTF-8"))?;
    rz_experiments::decode_json(text).map_err(|e| invalid(e.to_string()))
}
fn relative_file(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 256
        && !name.contains(['\\', ':'])
        && !name.chars().any(|c| c.is_control())
        && name
            .split('/')
            .all(|s| !s.is_empty() && !s.starts_with('.'))
        && Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}
fn shape_bytes(shape: &[u64], width: u64) -> Option<u64> {
    if shape.len() > 8 || shape.iter().any(|v| !(1..=1_048_576).contains(v)) {
        return None;
    }
    shape.iter().try_fold(width, |a, b| a.checked_mul(*b))
}

// Metadata validation is deliberately private; only the observed source-pinned
// child below can issue content authority. The compiled Python producer also
// checks the full name/shape/owner/transform allowlist and actual tensor bytes.
fn validate_summary(summary: &ChildSummary, request: &ContentRequest) -> Result<(), ArenaError> {
    check(
        summary.schema == PROOF_DOMAIN
            && summary.file == PROOF_FILE
            && generation_sha(&summary.sha256)
            && summary.bytes > 0
            && summary.bytes <= request.limits.proof_bytes
            && summary.request_sha256 == request.digest()?
            && Some(summary.frozen_pc_parameter_sha256.as_str())
                == request.expected_frozen_pc_sha256.as_deref()
            && !summary.optimizer_content_verified
            && !summary.numerical_acceptance
            && !summary.native_loaded_assets_verified,
        "frozen verifier original summary scope/request/parameter mismatch",
    )
}
fn validate_proof(
    proof: &ContentProof,
    request: &ContentRequest,
    summary: &ChildSummary,
) -> Result<(), ArenaError> {
    validate_summary(summary, request)?;
    let v = &proof.verification;
    let scope = if request.checkpoint.domain == MODEL_DOMAIN {
        "own_initialization_tensor_content_only"
    } else {
        "actual_pc_frozen_tensor_content_only"
    };
    check(
        proof.schema == PROOF_DOMAIN
            && proof.request == *request
            && proof.request_sha256 == summary.request_sha256
            && proof.scope == "original_checkpoint_named_shared_pc_to_exact_manual_onnx"
            && proof.checkpoint_scope == scope
            && proof.frozen_pc_parameter_sha256 == summary.frozen_pc_parameter_sha256
            && proof.encoding_registry_sha256
                == hex(&rz_eval::pals_model::virtual_review::registered_encoding_sha256())
            && v.checkpoint_tensor_content_verified
            && v.export_initializer_content_verified
            && v.manual_graph_structure_verified
            && !v.optimizer_content_verified
            && !v.numerical_acceptance
            && !v.native_loaded_assets_verified
            && !v.arena_eligible,
        "frozen proof scope or exact original request/encoding mismatch",
    )?;
    check(
        proof.parameter_inventory.len() == 104
            && proof.mapping.len() == 112
            && proof.graphs.len() == 2,
        "frozen registered V3 complete named tensor and copied initializer inventory",
    )?;
    let mut parameters = BTreeMap::new();
    let mut prior = "";
    let mut total = 0u64;
    for p in &proof.parameter_inventory {
        check(
            generation_identifier(&p.name)
                && p.name.as_str() > prior
                && p.dtype == "fp32_le"
                && ["shared", "proposer", "critic"].contains(&p.owner.as_str())
                && p.semantics == format!("{MODEL_DOMAIN}:{}", p.name)
                && shape_bytes(&p.shape, 4) == Some(p.bytes)
                && generation_sha(&p.sha256),
            "frozen sorted named tensor representation",
        )?;
        total = total
            .checked_add(p.bytes)
            .ok_or_else(|| invalid("frozen tensor total overflow"))?;
        check(
            total <= request.limits.tensor_bytes,
            "frozen named tensor total exceeds registered cap",
        )?;
        parameters.insert(p.name.as_str(), p);
        prior = &p.name;
    }
    check(
        proof.parameter_inventory_sha256
            == generation_digest(INVENTORY_DOMAIN, &proof.parameter_inventory)?,
        "frozen named tensor inventory seal",
    )?;
    let mut graph_entries = BTreeMap::new();
    for (graph, pin) in proof.graphs.iter().zip(&request.graphs) {
        check(
            graph.role == pin.role
                && graph.file == pin.file
                && graph.sha256 == pin.sha256
                && graph.bytes == pin.bytes
                && generation_sha(&graph.structure_sha256)
                && !graph.initializers.is_empty()
                && graph.initializers.len() <= 4096,
            "frozen readback graph original pin",
        )?;
        let mut entries = BTreeMap::new();
        let mut previous = "";
        let mut graph_total = 0u64;
        for entry in &graph.initializers {
            let width = match entry.dtype.as_str() {
                "FLOAT" => 4,
                "INT64" => 8,
                "BOOL" => 1,
                _ => 0,
            };
            check(
                generation_identifier(&entry.name)
                    && entry.name.as_str() > previous
                    && width > 0
                    && shape_bytes(&entry.shape, width) == Some(entry.bytes)
                    && generation_sha(&entry.sha256),
                "frozen bounded sorted initializer",
            )?;
            graph_total = graph_total
                .checked_add(entry.bytes)
                .ok_or_else(|| invalid("frozen initializer total overflow"))?;
            check(
                graph_total <= graph.bytes.min(request.limits.graph_bytes),
                "frozen initializer total exceeds original graph",
            )?;
            entries.insert(entry.name.as_str(), entry);
            previous = &entry.name;
        }
        check(
            graph.initializer_inventory_sha256
                == generation_digest("rz-pals-onnx-readback-initializers/1", &graph.initializers)?,
            "frozen readback initializer inventory seal",
        )?;
        graph_entries.insert(graph.role.as_str(), entries);
    }
    let mut last: Option<(&str, &str)> = None;
    let mut mapped = BTreeSet::new();
    for m in &proof.mapping {
        let key = (m.graph_role.as_str(), m.parameter.as_str());
        check(
            last.is_none_or(|prev| key > prev),
            "frozen copied initializer mapping order/duplicate",
        )?;
        last = Some(key);
        let parameter = parameters
            .get(m.parameter.as_str())
            .ok_or_else(|| invalid("frozen mapping unnamed source parameter"))?;
        let initializer = graph_entries
            .get(m.graph_role.as_str())
            .and_then(|g| g.get(m.initializer.as_str()))
            .ok_or_else(|| invalid("frozen mapping absent loaded graph initializer"))?;
        let transposed = m.transform == "transpose_2d";
        let shape = if transposed {
            parameter.shape.iter().rev().copied().collect()
        } else {
            parameter.shape.clone()
        };
        check(
            (m.transform == "identity" || (transposed && parameter.shape.len() == 2))
                && m.initializer
                    == format!(
                        "{}{}",
                        m.parameter,
                        if transposed { ".linear_transpose" } else { "" }
                    )
                && m.source_sha256 == parameter.sha256
                && m.initializer_sha256 == initializer.sha256
                && initializer.dtype == "FLOAT"
                && initializer.shape == shape
                && initializer.bytes == parameter.bytes
                && (transposed || m.source_sha256 == m.initializer_sha256),
            "frozen source/readback tensor exact mapping",
        )?;
        mapped.insert(m.parameter.as_str());
    }
    check(
        mapped.len() == parameters.len()
            && proof.mapping_sha256 == generation_digest(MAPPING_DOMAIN, &proof.mapping)?,
        "frozen copied tensor coverage and seal",
    )
}
fn validate_native(
    request: &ContentRequest,
    loaded: &PalsVirtualReviewOnnxBackend,
) -> Result<(), ArenaError> {
    check(
        loaded.quarantine_cause().is_none()
            && !loaded.approximate_warm_capable()
            && loaded.model_config() == &request.config
            && hex(&loaded.model_epoch()) == request.checkpoint.sha256
            && hex(&loaded.manifest_digest()) == request.export_manifest.sha256,
        "frozen actual native Fresh backend checkpoint/config/manifest differs",
    )?;
    let actual = &loaded.residency().graphs;
    check(
        actual.len() == (if loaded.validator_present() { 3 } else { 2 })
            && loaded.residency().native_sessions == actual.len(),
        "frozen actual native graph/session roster",
    )?;
    let mut roles = BTreeSet::new();
    for graph in actual {
        check(
            roles.insert(graph.role.as_str()),
            "frozen native duplicate graph role",
        )?;
        if let Some(expected) = request.graphs.iter().find(|p| p.role == graph.role) {
            check(
                expected.sha256 == hex(&graph.sha256) && expected.bytes == graph.serialized_bytes,
                "frozen actual loaded graph byte pin differs from verified content",
            )?;
        } else {
            check(
                loaded.validator_present() && graph.role == "validator",
                "frozen unregistered native graph role",
            )?;
        }
    }
    check(
        request
            .graphs
            .iter()
            .all(|g| roles.contains(g.role.as_str())),
        "frozen native public/shared P/C graph absent",
    )
}

/// Run explicitly in the native startup hook before its backend enters the
/// physical worker. The borrowed backend is actual loaded authority. This
/// entry point never constructs a neural model, runs forward, trains, or renews
/// an expired original window. Non-Linux intake is typed unsupported.
#[allow(clippy::too_many_arguments)]
pub fn run_frozen_pc_intake_v3(
    launch: FrozenPcVerifierLaunchV3,
    plan: &ActionOutcomeCollectionPlanV3,
    loaded: &PalsVirtualReviewOnnxBackend,
    teacher_binary_sha256: String,
    teacher_profile_sha256: String,
    cancel: Option<&AtomicBool>,
) -> Result<VerifiedFrozenPcIntakeV3, Box<FrozenPcIntakeFailureV3>> {
    #[cfg(target_os = "linux")]
    {
        run_linux(
            launch,
            plan,
            loaded,
            teacher_binary_sha256,
            teacher_profile_sha256,
            cancel,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            plan,
            loaded,
            teacher_binary_sha256,
            teacher_profile_sha256,
            cancel,
        );
        Err(Box::new(FrozenPcIntakeFailureV3 {
            error: invalid("frozen verified intake requires Linux native pinned process transport"),
            process: None,
            launch,
        }))
    }
}

// The source-pinned launcher reads no user code and performs no optimizer or
// forward call. Dependency directories are explicit; no .pth is evaluated.
const LAUNCHER: &str = r#"import sys, json, os, hashlib, pathlib, stat
def unique(pairs):
    value={}
    for key,item in pairs:
        if key in value: raise ValueError('duplicate original input key')
        value[key]=item
    return value
e=json.loads(sys.stdin.buffer.read(131072),object_pairs_hook=unique)
if set(e)!={'request','package','dependencies','artifacts','output','sources'}: raise ValueError('launcher envelope')
os.environ.update({'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'})
root=pathlib.Path(e['package'])/'rz_pals_model'
for pin in e['sources']:
    p=root/pin['file']
    st=p.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size!=pin['bytes']: raise ValueError('private source inode')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=pin['sha256']: raise ValueError('private source SHA')
sys.path[:0]=[e['package']]+e['dependencies']
from rz_pals_model.generation_frozen_pc import verify_frozen_pc_content,read_frozen_pc_proof
summary=verify_frozen_pc_content(e['request'],artifact_root=e['artifacts'],output_directory=e['output'],execute=True)
read_frozen_pc_proof(pathlib.Path(e['output'])/summary['file'],expected_file_sha256=summary['sha256'],expected_request_sha256=summary['request_sha256'],expected_parameter_sha256=e['request']['expected_frozen_pc_sha256'])
print(json.dumps(summary,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False),flush=True)
"#;

#[cfg(target_os = "linux")]
struct LinuxHeld {
    artifact: cap_std::fs::Dir,
    scratch: cap_std::fs::Dir,
    directory_paths: Vec<(PathBuf, FileIdentity)>,
    // Retain registered dependency-root inodes while the interpreter imports
    // them. Their contents remain a declared trusted dependency environment,
    // rather than an attestation of all installed third-party libraries.
    _dependency_directories: Vec<File>,
    files: Vec<HeldFile>,
}
#[cfg(target_os = "linux")]
struct HeldFile {
    file: File,
    pin: FrozenPcFilePinV3,
    root: HeldRoot,
}
#[cfg(target_os = "linux")]
enum HeldRoot {
    Artifact,
    Scratch,
}
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}
#[cfg(target_os = "linux")]
impl FileIdentity {
    fn from(metadata: &std::fs::Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            mtime: metadata.mtime(),
            mtime_ns: metadata.mtime_nsec(),
            ctime: metadata.ctime(),
            ctime_ns: metadata.ctime_nsec(),
        }
    }
    fn inode_eq(self, other: Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }
}
#[cfg(target_os = "linux")]
struct ReadCredit {
    remaining: u64,
    until: Instant,
}
#[cfg(target_os = "linux")]
impl ReadCredit {
    fn live(&self, cancel: Option<&AtomicBool>) -> Result<(), ArenaError> {
        check(
            Instant::now() < self.until && !cancel.is_some_and(|c| c.load(Ordering::Acquire)),
            "frozen original intake deadline/cancellation",
        )
    }
    fn read(&mut self, bytes: u64, cancel: Option<&AtomicBool>) -> Result<(), ArenaError> {
        self.live(cancel)?;
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(|| invalid("frozen pinned read credit exceeded"))?;
        Ok(())
    }
}
#[cfg(target_os = "linux")]
fn io_error(stage: &str) -> ArenaError {
    ArenaError::Io(format!("frozen_intake.{stage}"))
}
#[cfg(target_os = "linux")]
fn readable_directory(dir: &cap_std::fs::Dir) -> Result<File, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    dir.open_with(".", &options)
        .map(|f| f.into_std())
        .map_err(|_| io_error("readable_directory"))
}
#[cfg(target_os = "linux")]
fn open_absolute_directory(path: &Path) -> Result<cap_std::fs::Dir, ArenaError> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::OpenOptionsExt;
    check(
        path.is_absolute()
            && path.components().count() <= 64
            && path
                .components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_))),
        "frozen canonical absolute directory",
    )?;
    let root = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open("/")
        .map_err(|_| io_error("root"))?;
    let mut dir = cap_std::fs::Dir::from_std_file(root);
    for component in path.components() {
        if let Component::Normal(name) = component {
            dir = dir
                .open_dir_nofollow(name)
                .map_err(|_| io_error("directory_nofollow"))?;
        }
    }
    Ok(dir)
}
#[cfg(target_os = "linux")]
fn open_relative(dir: &cap_std::fs::Dir, path: &str) -> Result<File, ArenaError> {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    check(relative_file(path), "frozen canonical relative open")?;
    let mut parent = dir.try_clone().map_err(|_| io_error("root_clone"))?;
    let parts = path.split('/').collect::<Vec<_>>();
    for name in &parts[..parts.len() - 1] {
        parent = parent
            .open_dir_nofollow(name)
            .map_err(|_| io_error("relative_directory"))?;
    }
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
    parent
        .open_with(parts[parts.len() - 1], &options)
        .map(|f| f.into_std())
        .map_err(|_| io_error("relative_file"))
}
#[cfg(target_os = "linux")]
fn verified_bytes(
    file: &File,
    pin: &FrozenPcFilePinV3,
    retain: bool,
    credit: &mut ReadCredit,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<u8>, ArenaError> {
    use std::os::unix::fs::{FileExt, MetadataExt};
    let before = file.metadata().map_err(|_| io_error("file_metadata"))?;
    check(
        before.is_file() && before.nlink() == 1 && before.len() == pin.bytes,
        "frozen pinned regular single-link exact file",
    )?;
    let identity = FileIdentity::from(&before);
    let mut hash = Sha256::new();
    let mut result = if retain {
        Vec::with_capacity(
            usize::try_from(pin.bytes).map_err(|_| invalid("frozen read allocation"))?,
        )
    } else {
        Vec::new()
    };
    let mut block = [0u8; 64 * 1024];
    let mut offset = 0u64;
    while offset < pin.bytes {
        let count = (pin.bytes - offset).min(block.len() as u64) as usize;
        credit.read(count as u64, cancel)?;
        let n = file
            .read_at(&mut block[..count], offset)
            .map_err(|_| io_error("read_at"))?;
        check(n > 0, "frozen pinned file premature EOF")?;
        hash.update(&block[..n]);
        if retain {
            result.extend_from_slice(&block[..n]);
        }
        offset += n as u64;
    }
    let after = file.metadata().map_err(|_| io_error("readback_metadata"))?;
    check(
        identity == FileIdentity::from(&after) && pin.sha256 == hex(&hash.finalize()),
        "frozen original file content/identity changed",
    )?;
    Ok(result)
}
#[cfg(target_os = "linux")]
impl LinuxHeld {
    fn revalidate(
        &self,
        credit: &mut ReadCredit,
        cancel: Option<&AtomicBool>,
    ) -> Result<(), ArenaError> {
        for (path, identity) in &self.directory_paths {
            credit.live(cancel)?;
            let dir = open_absolute_directory(path)?;
            let meta = readable_directory(&dir)?
                .metadata()
                .map_err(|_| io_error("directory_readback"))?;
            check(
                identity.inode_eq(FileIdentity::from(&meta)),
                "frozen root directory changed",
            )?;
        }
        for held in &self.files {
            let root = match held.root {
                HeldRoot::Artifact => &self.artifact,
                HeldRoot::Scratch => &self.scratch,
            };
            let current = open_relative(root, &held.pin.file)?;
            check(
                FileIdentity::from(&current.metadata().map_err(|_| io_error("named_file"))?)
                    == FileIdentity::from(
                        &held.file.metadata().map_err(|_| io_error("held_file"))?,
                    ),
                "frozen named/held file inode mismatch",
            )?;
            verified_bytes(&held.file, &held.pin, false, credit, cancel)?;
        }
        Ok(())
    }
}
#[cfg(target_os = "linux")]
fn prepare_roots(launch: &FrozenPcVerifierLaunchV3) -> Result<LinuxHeld, ArenaError> {
    use std::os::unix::fs::MetadataExt;
    let artifact = open_absolute_directory(&launch.artifact_directory)?;
    let scratch = open_absolute_directory(&launch.scratch_directory)?;
    check(
        !launch
            .scratch_directory
            .starts_with(&launch.artifact_directory)
            && !launch
                .artifact_directory
                .starts_with(&launch.scratch_directory),
        "frozen output/artifact roots overlap",
    )?;
    let metadata = readable_directory(&scratch)?
        .metadata()
        .map_err(|_| io_error("scratch_metadata"))?;
    let owner_uid = nix::unistd::Uid::effective().as_raw();
    check(
        metadata.mode() & 0o7777 == 0o700 && metadata.uid() == owner_uid,
        "frozen exclusive 0700 scratch owner",
    )?;
    check(
        scratch
            .entries()
            .map_err(|_| io_error("scratch_entries"))?
            .next()
            .is_none(),
        "frozen scratch must be empty",
    )?;
    for ancestor in launch.scratch_directory.ancestors() {
        let dir = open_absolute_directory(ancestor)?;
        check(
            dir.symlink_metadata(".git").is_err(),
            "frozen proof/source staging must be outside Git",
        )?;
    }
    let mut directory_paths = Vec::new();
    for (path, dir) in [
        (&launch.artifact_directory, &artifact),
        (&launch.scratch_directory, &scratch),
    ] {
        directory_paths.push((
            path.clone(),
            FileIdentity::from(
                &readable_directory(dir)?
                    .metadata()
                    .map_err(|_| io_error("root_identity"))?,
            ),
        ));
    }
    Ok(LinuxHeld {
        artifact,
        scratch,
        directory_paths,
        _dependency_directories: Vec::new(),
        files: Vec::new(),
    })
}
#[cfg(target_os = "linux")]
fn install_sources(held: &mut LinuxHeld) -> Result<Vec<FrozenPcFilePinV3>, ArenaError> {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    held.scratch
        .create_dir("verifier")
        .map_err(|_| io_error("private_source_parent"))?;
    let parent = held
        .scratch
        .open_dir_nofollow("verifier")
        .map_err(|_| io_error("private_source_parent_pin"))?;
    readable_directory(&parent)?
        .set_permissions(std::fs::Permissions::from_mode(0o700))
        .map_err(|_| io_error("private_parent_mode"))?;
    parent
        .create_dir("rz_pals_model")
        .map_err(|_| io_error("private_source_package"))?;
    let package = parent
        .open_dir_nofollow("rz_pals_model")
        .map_err(|_| io_error("private_package_pin"))?;
    readable_directory(&package)?
        .set_permissions(std::fs::Permissions::from_mode(0o700))
        .map_err(|_| io_error("private_package_mode"))?;
    let mut pins = compiled_frozen_pc_verifier_pins_v3()?;
    let init = compiled_source("__init__.py")?;
    pins.push(FrozenPcFilePinV3 {
        file: "__init__.py".into(),
        sha256: hex(&Sha256::digest(init)),
        bytes: init.len() as u64,
    });
    for pin in &pins {
        let relative = format!("verifier/rz_pals_model/{}", pin.file);
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .mode(0o400)
            .follow(FollowSymlinks::No);
        let mut file = held
            .scratch
            .open_with(&relative, &options)
            .map_err(|_| io_error("create_private_source"))?;
        file.write_all(compiled_source(&pin.file)?)
            .map_err(|_| io_error("write_private_source"))?;
        file.sync_all()
            .map_err(|_| io_error("sync_private_source"))?;
        drop(file);
        let file = open_relative(&held.scratch, &relative)?;
        let mut held_pin = pin.clone();
        held_pin.file = relative;
        held.files.push(HeldFile {
            file,
            pin: held_pin,
            root: HeldRoot::Scratch,
        });
    }
    readable_directory(&package)?
        .set_permissions(std::fs::Permissions::from_mode(0o500))
        .map_err(|_| io_error("seal_private_package"))?;
    readable_directory(&parent)?
        .set_permissions(std::fs::Permissions::from_mode(0o500))
        .map_err(|_| io_error("seal_private_parent"))?;
    Ok(pins)
}
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn run_linux(
    launch: FrozenPcVerifierLaunchV3,
    plan: &ActionOutcomeCollectionPlanV3,
    loaded: &PalsVirtualReviewOnnxBackend,
    teacher_binary: String,
    teacher_profile: String,
    cancel: Option<&AtomicBool>,
) -> Result<VerifiedFrozenPcIntakeV3, Box<FrozenPcIntakeFailureV3>> {
    use crate::process::{
        OriginalProcessInput, OriginalProcessWindow, OwnedArtifactTreeWatch,
        supervise_verified_input_in_original_window,
    };
    use std::ffi::OsString;
    let mut held = None;
    let mut process = None;
    let result =
        (|| -> Result<(FrozenPcRegistrationV3, serde_json::Value, Vec<u8>), ArenaError> {
            check(
                generation_sha(&launch.interpreter_sha256)
                    && generation_sha(&launch.request_sha256)
                    && launch.interpreter_bytes > 0
                    && launch.interpreter_bytes <= 64 * 1024 * 1024
                    && launch.request_bytes > 0
                    && launch.request_bytes <= MAX_REQUEST
                    && generation_sha(&teacher_binary)
                    && generation_sha(&teacher_profile)
                    && launch.dependency_directories.len() <= 8
                    && launch.limits.wall_ms > 0
                    && launch.limits.wall_ms <= 300_000
                    && launch.limits.shutdown_grace_ms > 0
                    && launch.limits.shutdown_grace_ms <= 5000
                    && launch.limits.max_output_bytes > 0
                    && launch.limits.max_output_bytes <= 64 * 1024
                    && launch.limits.max_child_processes == 1,
                "frozen bounded registered verifier launch",
            )?;
            let window = OriginalProcessWindow::new(
                launch.started,
                launch.execution_deadline,
                launch.whole_deadline,
            )?;
            check(
                launch
                    .whole_deadline
                    .duration_since(launch.started)
                    .as_millis()
                    <= 310_000,
                "frozen whole intake window exceeds cap",
            )?;
            let mut credit = ReadCredit {
                remaining: 5 * 1024 * 1024 * 1024,
                until: launch.execution_deadline,
            };
            credit.live(cancel)?;
            let request_pin = FrozenPcFilePinV3 {
                file: "original-request.json".into(),
                sha256: launch.request_sha256.clone(),
                bytes: launch.request_bytes,
            };
            let python_pin = FrozenPcFilePinV3 {
                file: "registered-interpreter".into(),
                sha256: launch.interpreter_sha256.clone(),
                bytes: launch.interpreter_bytes,
            };
            let raw = verified_bytes(&launch.request, &request_pin, true, &mut credit, cancel)?;
            let request: ContentRequest = decode(&raw)?;
            request.validate(plan, &launch.registered_source_sha256)?;
            validate_native(&request, loaded)?;
            verified_bytes(&launch.interpreter, &python_pin, false, &mut credit, cancel)?;
            held = Some(prepare_roots(&launch)?);
            let owner = held.as_mut().expect("held roots installed");
            for directory in &launch.dependency_directories {
                let cap = open_absolute_directory(directory)?;
                let file = readable_directory(&cap)?;
                owner.directory_paths.push((
                    directory.clone(),
                    FileIdentity::from(&file.metadata().map_err(|_| io_error("dependency_root"))?),
                ));
                owner._dependency_directories.push(file);
            }
            for pin in std::iter::once(request.checkpoint.pin())
                .chain(std::iter::once(request.export_manifest.clone()))
                .chain(request.graphs.iter().map(GraphPin::pin))
            {
                let file = open_relative(&owner.artifact, &pin.file)?;
                verified_bytes(&file, &pin, false, &mut credit, cancel)?;
                owner.files.push(HeldFile {
                    file,
                    pin,
                    root: HeldRoot::Artifact,
                });
            }
            let sources = install_sources(owner)?;
            owner.revalidate(&mut credit, cancel)?;
            let package = launch.scratch_directory.join("verifier");
            let output = launch.scratch_directory.join("proof");
            let envelope = serde_json::json!({"request":request,"package":package,"dependencies":launch.dependency_directories,
            "artifacts":launch.artifact_directory,"output":output,"sources":sources});
            let input = serde_json::to_vec(&envelope).map_err(|e| invalid(e.to_string()))?;
            check(
                !input.is_empty() && input.len() <= MAX_REQUEST as usize,
                "frozen original input envelope cap",
            )?;
            let args = [
                OsString::from("-I"),
                OsString::from("-S"),
                OsString::from("-B"),
                OsString::from("-c"),
                OsString::from(LAUNCHER),
            ];
            let watch = OwnedArtifactTreeWatch {
                max_total_bytes: (request.limits.checkpoint_bytes
                    + MAX_SOURCE * 14
                    + request.limits.proof_bytes)
                    .min(plan.maximum_output_bytes),
                max_file_bytes: request.limits.checkpoint_bytes.max(MAX_PROOF),
                max_files: 32,
                max_depth: 3,
            };
            let mut loaded_read = |image: &File| {
                verified_bytes(image, &python_pin, false, &mut credit, cancel)
                    .map(|_| ())
                    .map_err(|_| ())
            };
            let cwd = readable_directory(&owner.scratch)?;
            process = Some(supervise_verified_input_in_original_window(
                &launch.interpreter,
                &args,
                &cwd,
                launch.limits,
                cancel,
                &watch,
                OriginalProcessInput {
                    bytes: &input,
                    window,
                },
                &mut loaded_read,
            )?);
            let original = process
                .as_ref()
                .expect("original child observation retained");
            original.checked_loaded_image()?;
            original.checked_timing()?;
            check(
                original.transport_complete(),
                "frozen original verifier exit/pipe/group closure missing",
            )?;
            credit.live(cancel)?;
            verified_bytes(&launch.request, &request_pin, false, &mut credit, cancel)?;
            verified_bytes(&launch.interpreter, &python_pin, false, &mut credit, cancel)?;
            owner.revalidate(&mut credit, cancel)?;
            let summary: ChildSummary = decode(&original.process().stdout)?;
            check(
                summary.file == PROOF_FILE
                    && generation_sha(&summary.sha256)
                    && summary.bytes > 0
                    && summary.bytes <= request.limits.proof_bytes,
                "frozen child proof pin size/name",
            )?;
            let proof_pin = FrozenPcFilePinV3 {
                file: format!("proof/{PROOF_FILE}"),
                sha256: summary.sha256.clone(),
                bytes: summary.bytes,
            };
            let file = open_relative(&owner.scratch, &proof_pin.file)?;
            let proof_bytes = verified_bytes(&file, &proof_pin, true, &mut credit, cancel)?;
            let proof: ContentProof = decode(&proof_bytes)?;
            validate_proof(&proof, &request, &summary)?;
            owner.files.push(HeldFile {
                file,
                pin: proof_pin,
                root: HeldRoot::Scratch,
            });
            validate_native(&request, loaded)?;
            credit.live(cancel)?;
            let receipt = serde_json::json!({"schema":INTAKE_DOMAIN,"request_sha256":request.digest()?,"request_file_sha256":launch.request_sha256,
            "source_sha256":request.source_sha256,"verifier_sources":sources,"launcher_sha256":hex(&Sha256::digest(LAUNCHER.as_bytes())),
            "interpreter_sha256":launch.interpreter_sha256,"proof_sha256":summary.sha256,"proof_bytes":summary.bytes,
            "model_epoch":plan.frozen_pc_epoch,"named_shared_pc_parameter_sha256":plan.frozen_pc_weights_sha256,
            "native_model_epoch_sha256":hex(&loaded.model_epoch()),"native_manifest_sha256":hex(&loaded.manifest_digest()),
            "native_backend_sha256":hex(&loaded.backend_digest()),"native_graphs":loaded.residency().graphs,
            "process":original.process().receipt,"transport":original.observation(),
            "scope":"named_tensor_export_content_and_actual_loaded_asset_identity_only",
            "optimizer_content_verified":false,"numerical_acceptance":false,"arena_eligible":false});
            let verification = generation_digest(INTAKE_DOMAIN, &receipt)?;
            let registration = FrozenPcRegistrationV3::unverified(
                plan,
                hex(&loaded.model_epoch()),
                hex(&loaded.backend_digest()),
                teacher_binary,
                teacher_profile,
            )?;
            let registration =
                FrozenPcRegistrationV3::after_verified_intake(registration, verification)?;
            Ok((registration, receipt, proof_bytes))
        })();
    match result {
        Ok((registration, receipt, proof)) => Ok(VerifiedFrozenPcIntakeV3 {
            registration,
            receipt,
            proof,
            process: process.expect("success requires owned original child"),
            _launch: launch,
            _held: held.expect("success requires held assets"),
        }),
        Err(error) => Err(Box::new(FrozenPcIntakeFailureV3 {
            error,
            process,
            launch,
            _held: held,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Schema-only fixtures never create a process/native admission capability.
    fn request_fixture() -> (ContentRequest, ActionOutcomeCollectionPlanV3) {
        let plan = ActionOutcomeCollectionPlanV3 {
            run_id: "schema-only-frozen-intake".into(),
            desired_contexts: 2048,
            maximum_contexts: 8192,
            frozen_pc_epoch: 2,
            frozen_pc_weights_sha256: "11".repeat(32),
            maximum_output_bytes: 18 * 1024 * 1024 * 1024,
        };
        let request = ContentRequest {
            schema: REQUEST_DOMAIN.into(),
            source_sha256: "22".repeat(32),
            model_epoch: 2,
            config: PalsModelConfig::for_profile(PalsModelProfile::VirtualReviewV1),
            checkpoint: CheckpointPin {
                file: "frozen/model.pt".into(),
                sha256: "33".repeat(32),
                bytes: 1024,
                domain: CHECKPOINT_DOMAIN.into(),
            },
            graphs: vec![
                GraphPin {
                    file: "export/public_memory.onnx".into(),
                    sha256: "44".repeat(32),
                    bytes: 1024,
                    role: "public".into(),
                },
                GraphPin {
                    file: "export/shared_pc_if.onnx".into(),
                    sha256: "55".repeat(32),
                    bytes: 1024,
                    role: "shared_pc".into(),
                },
            ],
            export_manifest: FrozenPcFilePinV3 {
                file: "export/export.json".into(),
                sha256: "66".repeat(32),
                bytes: 512,
            },
            verifier_sources: compiled_frozen_pc_verifier_pins_v3().unwrap(),
            expected_frozen_pc_sha256: Some(plan.frozen_pc_weights_sha256.clone()),
            limits: ContentLimits {
                checkpoint_bytes: 1024 * 1024 * 1024,
                graph_bytes: 256 * 1024 * 1024,
                tensor_bytes: 128 * 1024 * 1024,
                proof_bytes: MAX_PROOF,
                wall_seconds: 300,
            },
        };
        (request, plan)
    }
    #[test]
    fn paths_and_tensor_bytes_are_bounded_without_neural_execution() {
        for path in [
            "",
            "../model.pt",
            ".env",
            "a/.secret",
            "a//b",
            "a/./b",
            "/a",
            "a\\b",
            "a:b",
            "a\n",
        ] {
            assert!(!relative_file(path), "{path:?}");
        }
        assert!(relative_file("frozen/shared_pc_if.onnx"));
        assert_eq!(shape_bytes(&[2, 3], 4), Some(24));
        assert_eq!(shape_bytes(&[0, 3], 4), None);
        assert_eq!(shape_bytes(&[1_048_576; 8], 8), None);
    }
    #[test]
    fn compiled_verifier_sources_have_exact_unique_owned_pins() {
        let pins = compiled_frozen_pc_verifier_pins_v3()
            .expect("compiled model_reference source inventory");
        assert_eq!(pins.len(), SOURCE_NAMES.len());
        for (pin, name) in pins.iter().zip(SOURCE_NAMES) {
            assert_eq!(pin.file, name);
            pin.validate(MAX_SOURCE).unwrap();
        }
        assert!(!compiled_source("__init__.py").unwrap().is_empty());
        assert!(compiled_source("caller_verifier.py").is_err());
    }
    #[test]
    fn strict_summary_rejects_duplicate_unknown_and_claimed_acceptance() {
        let raw = br#"{"schema":"x","schema":"y"}"#;
        assert!(decode::<serde_json::Value>(raw).is_err());
        let raw=br#"{"schema":"x","file":"x","sha256":"x","bytes":1,"request_sha256":"x","frozen_pc_parameter_sha256":"x","optimizer_content_verified":false,"numerical_acceptance":false,"native_loaded_assets_verified":false,"arena_eligible":true}"#;
        assert!(decode::<ChildSummary>(raw).is_err());
        let (request, _) = request_fixture();
        let mut summary = ChildSummary {
            schema: PROOF_DOMAIN.into(),
            file: PROOF_FILE.into(),
            sha256: "77".repeat(32),
            bytes: 512,
            request_sha256: request.digest().unwrap(),
            frozen_pc_parameter_sha256: request.expected_frozen_pc_sha256.clone().unwrap(),
            optimizer_content_verified: false,
            numerical_acceptance: false,
            native_loaded_assets_verified: false,
        };
        validate_summary(&summary, &request).unwrap();
        summary.optimizer_content_verified = true;
        assert!(validate_summary(&summary, &request).is_err());
        summary.optimizer_content_verified = false;
        summary.numerical_acceptance = true;
        assert!(validate_summary(&summary, &request).is_err());
        summary.numerical_acceptance = false;
        summary.native_loaded_assets_verified = true;
        assert!(validate_summary(&summary, &request).is_err());
    }
    #[test]
    fn request_keeps_named_parameters_checkpoint_bytes_and_epoch_distinct() {
        let (request, plan) = request_fixture();
        request.validate(&plan, &request.source_sha256).unwrap();
        assert_ne!(request.checkpoint.sha256, plan.frozen_pc_weights_sha256);
        let raw = serde_json::to_vec(&request).unwrap();
        assert_eq!(decode::<ContentRequest>(&raw).unwrap(), request);
        let mut wrong = request.clone();
        wrong.expected_frozen_pc_sha256 = Some(request.checkpoint.sha256.clone());
        assert!(wrong.validate(&plan, &request.source_sha256).is_err());
        wrong = request.clone();
        wrong.model_epoch += 1;
        assert!(wrong.validate(&plan, &request.source_sha256).is_err());
        wrong = request.clone();
        wrong.verifier_sources[0].sha256 = "88".repeat(32);
        assert!(wrong.validate(&plan, &request.source_sha256).is_err());
        wrong = request.clone();
        wrong.graphs.swap(0, 1);
        assert!(wrong.validate(&plan, &request.source_sha256).is_err());
        wrong = request.clone();
        wrong.config.max_records = 128;
        assert!(wrong.validate(&plan, &request.source_sha256).is_err());
        assert!(request.validate(&plan, &"99".repeat(32)).is_err());
    }
    #[test]
    fn frozen_json_refuses_float_sizes_and_unregistered_cap_overrides() {
        let (request, plan) = request_fixture();
        let mut value = serde_json::to_value(&request).unwrap();
        value["checkpoint"]["bytes"] = serde_json::json!(1024.0);
        assert!(decode::<ContentRequest>(&serde_json::to_vec(&value).unwrap()).is_err());
        value = serde_json::to_value(&request).unwrap();
        value["limits"]["tensor_bytes"] = serde_json::json!(0);
        let wrong: ContentRequest = decode(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(wrong.validate(&plan, &request.source_sha256).is_err());
        value = serde_json::to_value(&request).unwrap();
        value["native_verified"] = serde_json::json!(true);
        assert!(decode::<ContentRequest>(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
