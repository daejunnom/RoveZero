//! Independent numerical evidence admission before any arena child is spawned.
//!
//! This reader never loads a native library or runs a model. It rechecks actual
//! captured tensor bytes and raw results; a status string or export declaration
//! cannot substitute for reference -> native execution evidence. Historical
//! V2 inputs retain their own schema and encoder. Fresh B1 is the first-pair
//! boundary; ApproxWarm needs a separate seeded-reference admission.
use crate::{
    ArtifactRef, ManifestError, PalsAdaptiveStartupConfigV5, PalsEngineModeV5, PalsModelBackendV3,
    PalsPrecisionV3, PalsStartupAssetPurposeV5 as Purpose, decode_json,
};
use rz_eval::pals_model::{
    PalsCandidateToken, PalsModelConfig, PalsModelInput, PalsModelProfile, PalsRawOutput, PalsRole,
    virtual_review::PalsVirtualReviewInput,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PALS_NUMERICAL_ACCEPTANCE_V5_DOMAIN: &str = "rz-pals-numerical-acceptance-v5/1";
pub const PALS_NUMERICAL_RAW_V5_DOMAIN: &str = "rz-pals-numerical-raw-run-v5/1";
pub const PALS_NUMERICAL_TENSOR_V5_DOMAIN: &str = "rz-pals-captured-tensor-bytes-v5/1";
const RAW_LIMIT: usize = crate::MAX_MANIFEST_BYTES;
const ACCEPTANCE_LIMIT: usize = 256 * 1024;
const TOTAL_RAW_LIMIT: u64 = 64 * 1024 * 1024;
const MAX_CASES: usize = 256;
const RAW_ATOL: f64 = 1e-4;
const RAW_RTOL: f64 = 1e-3;
const PROBABILITY_ATOL: f64 = 1e-4;
const NATIVE_COMPONENTS: [&str; 5] = [
    "model_native",
    "model_encoding",
    "nn_runtime",
    "model_reference",
    "model_export",
];

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNumericalBindingsV5 {
    pub model_profile: String,
    pub model_schema: String,
    pub model_semantics: String,
    pub encoding_schema: String,
    pub model_semantic_sha256: String,
    pub encoding_sha256: String,
    pub model_epoch: u64,
    pub weights_sha256: String,
    pub export_manifest_sha256: String,
    pub graphs: BTreeMap<String, String>,
    pub runtime_library_sha256: String,
    pub cuda_libraries: BTreeSet<String>,
    pub cuda_manifest_sha256: Option<String>,
    pub backend: PalsModelBackendV3,
    pub precision: PalsPrecisionV3,
    pub tf32: bool,
    pub source_components: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNumericalRunPairV5 {
    pub reference: ArtifactRef,
    pub native: ArtifactRef,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNumericalAcceptanceV5 {
    pub schema_version: u32,
    pub domain: String,
    pub bindings: PalsNumericalBindingsV5,
    pub invocation: PalsNumericalInvocationV5,
    pub batch: u32,
    pub runs: Vec<PalsNumericalRunPairV5>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsNumericalInvocationV5 {
    Fresh,
    ApproxWarmSeeded,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsNumericalProducerV5 {
    IndependentPytorchFp32,
    RustOrtNative,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "model_input_kind",
    content = "input",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsNumericalInputV5 {
    VirtualReview(PalsVirtualReviewInput),
    LegacyFullLine(PalsModelInput),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsNumericalDtypeV5 {
    F32Le,
    I64Le,
    BoolU8,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNumericalTensorV5 {
    pub name: String,
    pub dtype: PalsNumericalDtypeV5,
    pub shape: Vec<u64>,
    /// Captured exact little-endian bytes. Signed zero is preserved.
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNumericalRawCaseV5 {
    pub id: String,
    pub input: PalsNumericalInputV5,
    pub input_tensors: Vec<PalsNumericalTensorV5>,
    /// Hash of the explicitly framed, sorted name/dtype/shape/byte inventory.
    pub input_tensor_sha256: String,
    /// Original caller order, independently checked against the typed input.
    pub candidate_order: Vec<PalsCandidateToken>,
    pub raw_output: PalsRawOutput,
    /// Native completion identity, never an accepted-search/visit count.
    pub physical_completion_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNumericalRawRunV5 {
    pub schema_version: u32,
    pub domain: String,
    pub run_id: String,
    pub producer: PalsNumericalProducerV5,
    pub producer_source_component: String,
    pub bindings: PalsNumericalBindingsV5,
    pub invocation: PalsNumericalInvocationV5,
    pub batch: u32,
    /// Actual provider of the captured run. Torch reference has no ORT provider.
    pub observed_provider: Option<String>,
    /// Observed native public-cache hits; independent Torch has no native cache.
    /// The independent numeric capture fixes caching off rather than counting
    /// a reused public graph as another completed neural execution.
    pub public_cache_hits: u64,
    /// Native graph input count, including both public and private graphs.
    /// Fresh B1 with cache off completes two graph inputs per role evaluation.
    pub completed_nn_inputs: u64,
    pub physical_completion_known: bool,
    pub cases: Vec<PalsNumericalRawCaseV5>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsNumericalAdmissionV5 {
    pub domain: String,
    pub acceptance: ArtifactRef,
    pub reference_runs: Vec<ArtifactRef>,
    pub native_runs: Vec<ArtifactRef>,
    pub bindings: PalsNumericalBindingsV5,
    pub invocation: PalsNumericalInvocationV5,
    pub batch: u32,
    pub compared_cases: u64,
    pub completed_native_nn_inputs: u64,
    pub input_tensor_bytes_compared: u64,
    pub max_raw_absolute_difference: f64,
    pub max_policy_absolute_difference: f64,
    pub max_wdl_absolute_difference: f64,
    pub evidence_bytes_read: u64,
    /// Tensor fixtures certify model math, not legal game states or strength.
    pub evidence_scope: String,
}

/// Read only the registered raw-evidence artifact requested by the gate. The
/// caller retains source/private FDs and returns bounded actual bytes; this
/// function verifies their SHA and size again. Each run pair is released before
/// the next read so raw fixtures do not remain resident for the whole arena.
pub fn validate_pals_numerical_acceptance_v5(
    config: &PalsAdaptiveStartupConfigV5,
    acceptance_bytes: &[u8],
    mut read_artifact: impl FnMut(&ArtifactRef, usize) -> Result<Vec<u8>, ManifestError>,
) -> Result<PalsNumericalAdmissionV5, ManifestError> {
    let asset = config.asset(Purpose::NumericalAcceptance)?;
    verify_bytes(&asset.artifact, acceptance_bytes, ACCEPTANCE_LIMIT)?;
    let acceptance: PalsNumericalAcceptanceV5 = parse(acceptance_bytes, ACCEPTANCE_LIMIT)?;
    let expected = expected_bindings(config)?;
    if acceptance.schema_version != 5
        || acceptance.domain != PALS_NUMERICAL_ACCEPTANCE_V5_DOMAIN
        || acceptance.bindings != expected
        || acceptance.invocation != PalsNumericalInvocationV5::Fresh
        || acceptance.batch != 1
        || !(1..=32).contains(&acceptance.runs.len())
    {
        return Err(integrity(
            "numerical admission requires exact bound Fresh B1 runs",
        ));
    }
    validate_export_boundary(config, &mut read_artifact)?;
    let declared = config
        .assets
        .iter()
        .filter(|a| a.purpose == Purpose::NumericalRawEvidence)
        .map(|a| (&a.artifact.path, &a.artifact))
        .collect::<BTreeMap<_, _>>();
    if declared.is_empty() {
        return Err(integrity("independent numerical raw evidence absent"));
    }
    let mut used = BTreeSet::new();
    let mut case_ids = BTreeSet::new();
    let mut completion_ids = BTreeSet::new();
    let mut coverage = Coverage::default();
    let mut audit = PalsNumericalAdmissionV5 {
        domain: PALS_NUMERICAL_ACCEPTANCE_V5_DOMAIN.into(), acceptance: asset.artifact.clone(),
        reference_runs: Vec::new(), native_runs: Vec::new(), bindings: expected.clone(),
        invocation: PalsNumericalInvocationV5::Fresh, batch: 1, compared_cases: 0,
        completed_native_nn_inputs: 0, input_tensor_bytes_compared: 0,
        max_raw_absolute_difference: 0., max_policy_absolute_difference: 0.,
        max_wdl_absolute_difference: 0., evidence_bytes_read: acceptance_bytes.len() as u64
            + config.asset(Purpose::ExportManifest)?.artifact.bytes,
        evidence_scope: "independent_reference_vs_actual_native_raw_tensor_math;Rules_legality_and_arena_strength_are_separate".into(),
    };
    for pair in &acceptance.runs {
        let mut read_run =
            |artifact: &ArtifactRef| -> Result<PalsNumericalRawRunV5, ManifestError> {
                if declared.get(&artifact.path) != Some(&artifact)
                    || !used.insert(artifact.path.clone())
                {
                    return Err(integrity(
                        "raw evidence is unregistered, ambiguous or reused",
                    ));
                }
                audit.evidence_bytes_read = audit
                    .evidence_bytes_read
                    .checked_add(artifact.bytes)
                    .ok_or_else(|| {
                    ManifestError::Artifact("numerical raw byte sum overflow".into())
                })?;
                if audit.evidence_bytes_read > TOTAL_RAW_LIMIT {
                    return Err(ManifestError::Artifact(
                        "numerical raw evidence exceeds 64MiB read bound".into(),
                    ));
                }
                let bytes = read_artifact(artifact, RAW_LIMIT)?;
                verify_bytes(artifact, &bytes, RAW_LIMIT)?;
                parse(&bytes, RAW_LIMIT)
            };
        let reference = read_run(&pair.reference)?;
        let native = read_run(&pair.native)?;
        validate_run(
            &reference,
            &expected,
            PalsNumericalProducerV5::IndependentPytorchFp32,
        )?;
        validate_run(&native, &expected, PalsNumericalProducerV5::RustOrtNative)?;
        if reference.run_id == native.run_id || reference.cases.len() != native.cases.len() {
            return Err(integrity("independent runs or case counts differ"));
        }
        for (left, right) in reference.cases.iter().zip(&native.cases) {
            if !case_ids.insert(left.id.clone()) || case_ids.len() > MAX_CASES {
                return Err(integrity(
                    "duplicate case or bounded numerical case roster exceeded",
                ));
            }
            let observation = compare_case(left, right, &expected)?;
            let completion = right
                .physical_completion_id
                .as_ref()
                .ok_or_else(|| integrity("native case has no physical completion identity"))?;
            if !valid_id(completion)
                || !completion_ids.insert((native.run_id.clone(), completion.clone()))
            {
                return Err(integrity(
                    "native physical completion is invalid or duplicated",
                ));
            }
            coverage.add(&observation);
            audit.compared_cases += 1;
            audit.input_tensor_bytes_compared += observation.tensor_bytes;
            audit.max_raw_absolute_difference =
                audit.max_raw_absolute_difference.max(observation.raw_max);
            audit.max_policy_absolute_difference = audit
                .max_policy_absolute_difference
                .max(observation.policy_max);
            audit.max_wdl_absolute_difference =
                audit.max_wdl_absolute_difference.max(observation.wdl_max);
        }
        audit.completed_native_nn_inputs += native.completed_nn_inputs;
        audit.reference_runs.push(pair.reference.clone());
        audit.native_runs.push(pair.native.clone());
    }
    if used.len() != declared.len() || audit.completed_native_nn_inputs != 2 * audit.compared_cases
    {
        return Err(integrity(
            "raw evidence roster or completed NN input count differs",
        ));
    }
    coverage.validate(config)?;
    Ok(audit)
}

fn integrity(message: &str) -> ManifestError {
    ManifestError::Integrity(message.into())
}
fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}
fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8], limit: usize) -> Result<T, ManifestError> {
    if bytes.is_empty() || bytes.len() > limit {
        return Err(ManifestError::Artifact("numerical JSON byte bound".into()));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| integrity("numerical JSON is not UTF-8"))?;
    decode_json(text)
}
fn verify_bytes(artifact: &ArtifactRef, bytes: &[u8], limit: usize) -> Result<(), ManifestError> {
    artifact.validate()?;
    if bytes.is_empty()
        || bytes.len() > limit
        || artifact.bytes != bytes.len() as u64
        || artifact.sha256 != format!("{:x}", Sha256::digest(bytes))
    {
        return Err(integrity(
            "numerical evidence bytes/size/SHA differ from registered artifact",
        ));
    }
    Ok(())
}
fn expected_bindings(
    config: &PalsAdaptiveStartupConfigV5,
) -> Result<PalsNumericalBindingsV5, ManifestError> {
    let m = &config.endpoint.model;
    m.validate(config.endpoint.mode)?;
    if m.profile == "virtual_review_v1"
        && m.encoding_sha256
            != rz_eval::pals_model::virtual_review::registered_encoding_sha256()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
    {
        return Err(integrity(
            "compact numerical encoding pin differs from the actual tensor registry",
        ));
    }
    if !matches!(
        m.backend,
        PalsModelBackendV3::OrtCpu | PalsModelBackendV3::OrtCuda
    ) || m.precision != PalsPrecisionV3::Fp32
        || m.tf32
    {
        return Err(integrity(
            "actual native numerical admission requires ORT FP32 TF32 off",
        ));
    }
    let mut source_components = BTreeMap::new();
    for key in NATIVE_COMPONENTS {
        let pin = config
            .source_components
            .get(key)
            .ok_or_else(|| integrity("native/reference/export source component missing"))?;
        if pin.len() != 64
            || !pin
                .bytes()
                .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
        {
            return Err(integrity(
                "native/reference/export component pin is not lowercase SHA-256",
            ));
        }
        source_components.insert(key.into(), pin.clone());
    }
    let mut graphs = BTreeMap::new();
    for (role, purpose) in [
        ("public", Purpose::PublicGraph),
        ("shared_pc", Purpose::SharedPcGraph),
        ("proposer", Purpose::ProposerGraph),
        ("critic", Purpose::CriticGraph),
        ("validator", Purpose::ValidatorGraph),
    ] {
        let assets = config
            .assets
            .iter()
            .filter(|a| a.purpose == purpose)
            .collect::<Vec<_>>();
        if assets.len() > 1 {
            return Err(integrity("numerical graph role is ambiguous"));
        }
        if let Some(a) = assets.first() {
            graphs.insert(role.into(), a.artifact.sha256.clone());
        }
    }
    let collection = config.endpoint.mode == PalsEngineModeV5::Collection;
    let valid = if m.profile == "virtual_review_v1" {
        BTreeSet::from_iter(
            ["public", "shared_pc"]
                .into_iter()
                .chain(collection.then_some("validator")),
        )
    } else if m.profile == "full_line_interaction_v2" && !collection {
        let separate = graphs.contains_key("proposer") || graphs.contains_key("critic");
        if separate {
            BTreeSet::from(["public", "proposer", "critic"])
        } else {
            BTreeSet::from(["public", "shared_pc"])
        }
    } else {
        return Err(integrity("unsupported numerical model/role capability"));
    };
    if graphs.keys().map(String::as_str).collect::<BTreeSet<_>>() != valid {
        return Err(integrity(
            "numerical graph role set differs from declared model capability",
        ));
    }
    let cuda_manifests = config
        .assets
        .iter()
        .filter(|a| a.purpose == Purpose::CudaManifest)
        .collect::<Vec<_>>();
    let cuda_libraries = config
        .assets
        .iter()
        .filter(|a| a.purpose == Purpose::CudaLibrary)
        .collect::<Vec<_>>();
    if m.backend == PalsModelBackendV3::OrtCuda {
        if cuda_manifests.len() != 1 || cuda_libraries.is_empty() {
            return Err(integrity(
                "CUDA numerical binding requires one manifest and actual library pins",
            ));
        }
    } else if !cuda_manifests.is_empty() || !cuda_libraries.is_empty() {
        return Err(integrity(
            "CPU numerical binding cannot claim CUDA library/provider assets",
        ));
    }
    Ok(PalsNumericalBindingsV5 {
        model_profile: m.profile.clone(),
        model_schema: m.model_schema.clone(),
        model_semantics: m.model_semantics.clone(),
        encoding_schema: m.encoding_schema.clone(),
        model_semantic_sha256: m.semantic_sha256.clone(),
        encoding_sha256: m.encoding_sha256.clone(),
        model_epoch: m.epoch,
        weights_sha256: m.weights.sha256.clone(),
        export_manifest_sha256: config
            .asset(Purpose::ExportManifest)?
            .artifact
            .sha256
            .clone(),
        graphs,
        runtime_library_sha256: config
            .asset(Purpose::RuntimeLibrary)?
            .artifact
            .sha256
            .clone(),
        cuda_libraries: config
            .assets
            .iter()
            .filter(|a| a.purpose == Purpose::CudaLibrary)
            .map(|a| a.artifact.sha256.clone())
            .collect(),
        cuda_manifest_sha256: config
            .assets
            .iter()
            .find(|a| a.purpose == Purpose::CudaManifest)
            .map(|a| a.artifact.sha256.clone()),
        backend: m.backend,
        precision: m.precision,
        tf32: m.tf32,
        source_components,
    })
}

#[cfg(feature = "pals-content-admission")]
fn validate_export_boundary(
    config: &PalsAdaptiveStartupConfigV5,
    read: &mut impl FnMut(&ArtifactRef, usize) -> Result<Vec<u8>, ManifestError>,
) -> Result<(), ManifestError> {
    use rz_eval::pals_onnx::virtual_review::{
        VirtualReviewExecutionMode, validate_pals_legacy_export_manifest,
        validate_virtual_review_export_manifest,
    };
    let artifact = &config.asset(Purpose::ExportManifest)?.artifact;
    let bytes = read(artifact, 256 * 1024)?;
    verify_bytes(artifact, &bytes, 256 * 1024)?;
    let mode = match config.endpoint.mode {
        PalsEngineModeV5::Collection => VirtualReviewExecutionMode::Collection,
        PalsEngineModeV5::Deployment => VirtualReviewExecutionMode::Deployment,
    };
    let expected = expected_bindings(config)?;
    let actual = if expected.model_profile == "virtual_review_v1" {
        let summary = validate_virtual_review_export_manifest(&bytes, mode)
            .map_err(|e| integrity(&e.to_string()))?;
        if summary
            .checkpoint_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            != expected.weights_sha256
        {
            return Err(integrity("numerical export checkpoint differs"));
        }
        summary
            .graphs
            .iter()
            .map(|g| {
                (
                    g.role.clone(),
                    g.sha256
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>(),
                )
            })
            .collect::<BTreeMap<_, _>>()
    } else {
        let summary = validate_pals_legacy_export_manifest(&bytes, mode)
            .map_err(|e| integrity(&e.to_string()))?;
        if summary
            .checkpoint_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            != expected.weights_sha256
        {
            return Err(integrity("legacy numerical export checkpoint differs"));
        }
        summary
            .graphs
            .iter()
            .map(|g| {
                (
                    g.role.clone(),
                    g.sha256
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>(),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    if actual != expected.graphs {
        return Err(integrity(
            "strict export graph pins differ from numerical graphs",
        ));
    }
    Ok(())
}
#[cfg(not(feature = "pals-content-admission"))]
fn validate_export_boundary(
    _: &PalsAdaptiveStartupConfigV5,
    _: &mut impl FnMut(&ArtifactRef, usize) -> Result<Vec<u8>, ManifestError>,
) -> Result<(), ManifestError> {
    Err(integrity(
        "numerical native preflight requires the explicit ONNX feature",
    ))
}

fn validate_run(
    run: &PalsNumericalRawRunV5,
    bindings: &PalsNumericalBindingsV5,
    producer: PalsNumericalProducerV5,
) -> Result<(), ManifestError> {
    if run.schema_version != 5
        || run.domain != PALS_NUMERICAL_RAW_V5_DOMAIN
        || !valid_id(&run.run_id)
        || run.producer != producer
        || &run.bindings != bindings
        || run.invocation != PalsNumericalInvocationV5::Fresh
        || run.batch != 1
        || !(1..=MAX_CASES).contains(&run.cases.len())
    {
        return Err(integrity(
            "numerical raw run identity or execution boundary differs",
        ));
    }
    match producer {
        PalsNumericalProducerV5::IndependentPytorchFp32 => {
            if run.producer_source_component != bindings.source_components["model_reference"]
                || run.observed_provider.is_some()
                || run.completed_nn_inputs != 0
                || run.public_cache_hits != 0
                || run.physical_completion_known
                || run.cases.iter().any(|c| c.physical_completion_id.is_some())
            {
                return Err(integrity(
                    "independent reference is relabelled as native completion",
                ));
            }
        }
        PalsNumericalProducerV5::RustOrtNative => {
            let provider = match bindings.backend {
                PalsModelBackendV3::OrtCpu => "CPUExecutionProvider",
                PalsModelBackendV3::OrtCuda => "CUDAExecutionProvider",
                _ => return Err(integrity("native numerical backend unsupported")),
            };
            if run.producer_source_component != bindings.source_components["model_native"]
                || run.observed_provider.as_deref() != Some(provider)
                || !run.physical_completion_known
                || run.completed_nn_inputs != 2 * run.cases.len() as u64
                || run.public_cache_hits != 0
                || run.cases.iter().any(|case| {
                    case.physical_completion_id
                        .as_deref()
                        .is_none_or(|id| !valid_id(id))
                })
            {
                return Err(integrity(
                    "actual native provider/physical completion/input evidence missing",
                ));
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Coverage {
    combinations: BTreeSet<(u8, usize, usize, i8)>,
    promotions: BTreeMap<u8, BTreeSet<u8>>,
    empty_records: BTreeSet<u8>,
    padded_lines: BTreeSet<(u8, usize)>,
    orders: BTreeMap<(u8, String, BTreeSet<PalsCandidateToken>), BTreeSet<Vec<PalsCandidateToken>>>,
}
struct CaseObservation {
    role: PalsRole,
    bucket: usize,
    anchors: usize,
    perspective: i8,
    candidates: Vec<PalsCandidateToken>,
    same_question_sha256: String,
    empty_records: bool,
    padded_lines: bool,
    tensor_bytes: u64,
    raw_max: f64,
    policy_max: f64,
    wdl_max: f64,
}
impl Coverage {
    fn add(&mut self, o: &CaseObservation) {
        let role = role_index(o.role);
        self.combinations
            .insert((role, o.bucket, o.anchors, o.perspective));
        self.promotions
            .entry(role)
            .or_default()
            .extend(o.candidates.iter().map(|c| c.promotion));
        if o.empty_records {
            self.empty_records.insert(role);
        }
        if o.padded_lines {
            self.padded_lines.insert((role, o.bucket));
        }
        self.orders
            .entry((
                role,
                o.same_question_sha256.clone(),
                o.candidates.iter().copied().collect(),
            ))
            .or_default()
            .insert(o.candidates.clone());
    }
    fn has_candidate_order_variation(&self, role: u8) -> bool {
        self.orders
            .iter()
            .any(|((r, _question, candidates), orders)| {
                *r == role && candidates.len() > 1 && orders.len() > 1
            })
    }
    fn validate(&self, config: &PalsAdaptiveStartupConfigV5) -> Result<(), ManifestError> {
        let roles = if config.endpoint.mode == PalsEngineModeV5::Collection {
            vec![PalsRole::Proposer, PalsRole::Critic, PalsRole::Validator]
        } else {
            vec![PalsRole::Proposer, PalsRole::Critic]
        };
        let compact = config.endpoint.model.profile == "virtual_review_v1";
        let buckets: &[usize] = if compact {
            &[8, 16, 32, 64, 128, 256]
        } else {
            &[256]
        };
        for role in roles.into_iter().map(role_index) {
            for bucket in buckets {
                for anchors in 1..=if compact { 4 } else { 1 } {
                    for perspective in [-1, 1] {
                        if !self
                            .combinations
                            .contains(&(role, *bucket, anchors, perspective))
                        {
                            return Err(integrity(
                                "role/L-bucket/anchor/perspective numerical coverage missing",
                            ));
                        }
                    }
                }
                if !self.padded_lines.contains(&(role, *bucket)) {
                    return Err(integrity("line-mask padding coverage missing"));
                }
            }
            if !self.empty_records.contains(&role)
                || !(1..=4).all(|p| self.promotions.get(&role).is_some_and(|v| v.contains(&p)))
                || !self.has_candidate_order_variation(role)
            {
                return Err(integrity(
                    "empty-mask/four-promotion/candidate-order numerical coverage missing",
                ));
            }
        }
        Ok(())
    }
}

fn role_index(role: PalsRole) -> u8 {
    match role {
        PalsRole::Proposer => 0,
        PalsRole::Critic => 1,
        PalsRole::Validator => 2,
    }
}

fn compare_case(
    reference: &PalsNumericalRawCaseV5,
    native: &PalsNumericalRawCaseV5,
    bindings: &PalsNumericalBindingsV5,
) -> Result<CaseObservation, ManifestError> {
    if !valid_id(&reference.id)
        || reference.id != native.id
        || reference.input != native.input
        || reference.candidate_order != native.candidate_order
        || reference.input_tensors != native.input_tensors
        || reference.input_tensor_sha256 != native.input_tensor_sha256
    {
        return Err(integrity(
            "reference/native case, input bytes or candidate order differ",
        ));
    }
    let (mut observation, expected_tensors, candidate_count, divergence_count) =
        prepare_case_input(&reference.input, &bindings.model_profile)?;
    // Semantic f32 PartialEq deliberately treats +0 and -0 as equal. Rebuild
    // both captured inputs so that equivalence here remains byte exact.
    let (_, native_tensors, _, _) = prepare_case_input(&native.input, &bindings.model_profile)?;
    if reference.candidate_order != observation.candidates
        || reference.input_tensors != expected_tensors
        || native.input_tensors != native_tensors
        || reference.input_tensor_sha256
            != pals_numerical_tensor_digest_v5(&reference.input_tensors, observation.role)?
    {
        return Err(integrity(
            "captured tensors differ from the registered typed encoder",
        ));
    }
    validate_output(
        &reference.raw_output,
        observation.role,
        candidate_count,
        divergence_count,
    )?;
    validate_output(
        &native.raw_output,
        observation.role,
        candidate_count,
        divergence_count,
    )?;
    let left = &reference.raw_output;
    let right = &native.raw_output;
    observation.raw_max = compare_raw(&left.candidate_logits, &right.candidate_logits)?
        .max(compare_raw(&left.wdl_logits, &right.wdl_logits)?)
        .max(compare_raw(&left.private_latent, &right.private_latent)?);
    if let (Some(a), Some(b)) = (&left.divergence_logits, &right.divergence_logits) {
        observation.raw_max = observation.raw_max.max(compare_raw(a, b)?);
    }
    if let (Some(a), Some(b)) = (&left.task_logits, &right.task_logits) {
        observation.raw_max = observation.raw_max.max(compare_raw(a, b)?);
    }
    observation.policy_max =
        compare_probabilities(&left.candidate_logits, &right.candidate_logits)?;
    observation.wdl_max = compare_probabilities(&left.wdl_logits, &right.wdl_logits)?;
    observation.tensor_bytes = reference
        .input_tensors
        .iter()
        .map(|t| t.bytes.len() as u64)
        .sum();
    Ok(observation)
}

/// Pure reusable fixture/reader boundary; no NN session or execution occurs.
/// The versioned input enum chooses an existing encoder, never an automatic
/// model/encoding conversion. Callers retain actual capture provenance.
pub fn pals_numerical_input_tensors_v5(
    input: &PalsNumericalInputV5,
) -> Result<Vec<PalsNumericalTensorV5>, ManifestError> {
    let profile = match input {
        PalsNumericalInputV5::VirtualReview(_) => "virtual_review_v1",
        PalsNumericalInputV5::LegacyFullLine(_) => "full_line_interaction_v2",
    };
    Ok(prepare_case_input(input, profile)?.1)
}

fn prepare_case_input(
    input: &PalsNumericalInputV5,
    profile: &str,
) -> Result<(CaseObservation, Vec<PalsNumericalTensorV5>, usize, usize), ManifestError> {
    let mut tensors = Vec::new();
    let (role, bucket, anchors, perspective, candidates, empty_records, padded_lines, divergences) =
        match input {
            PalsNumericalInputV5::VirtualReview(i) if profile == "virtual_review_v1" => {
                let p = i.prepare().map_err(|e| integrity(&e.to_string()))?;
                let (a, r, c, d, l) = (
                    p.anchors,
                    p.records_count,
                    p.candidates_count,
                    p.divergences_count,
                    p.length_bucket,
                );
                add_i64(&mut tensors, "anchor_board", &[1, a, 64], &p.anchor_board);
                add_bool(&mut tensors, "anchor_mask", &[1, a], &p.anchor_mask);
                add_f32(
                    &mut tensors,
                    "anchor_metadata",
                    &[1, a, 16],
                    &p.anchor_metadata,
                );
                add_bool(&mut tensors, "candidate_mask", &[1, c], &p.candidate_mask);
                add_i64(&mut tensors, "candidates", &[1, c, 3], &p.candidates);
                add_f32(
                    &mut tensors,
                    "divergence_features",
                    &[1, d, 8],
                    &p.divergence_features,
                );
                add_bool(&mut tensors, "divergence_mask", &[1, d], &p.divergence_mask);
                add_f32(&mut tensors, "query", &[1, 8], &p.query);
                add_i64(&mut tensors, "query_anchor", &[1, 3], &p.query_anchor);
                add_bool(
                    &mut tensors,
                    "query_line_mask",
                    &[1, 3, l],
                    &p.query_line_mask,
                );
                add_i64(
                    &mut tensors,
                    "query_line_tokens",
                    &[1, 3, l, 3],
                    &p.query_line_tokens,
                );
                add_i64(&mut tensors, "record_anchor", &[1, r], &p.record_anchor);
                add_bool(
                    &mut tensors,
                    "record_line_mask",
                    &[1, r, l],
                    &p.record_line_mask,
                );
                add_i64(
                    &mut tensors,
                    "record_line_tokens",
                    &[1, r, l, 3],
                    &p.record_line_tokens,
                );
                add_bool(&mut tensors, "record_mask", &[1, r], &p.record_mask);
                add_i64(
                    &mut tensors,
                    "record_relations",
                    &[1, r, 2],
                    &p.record_relations,
                );
                add_f32(&mut tensors, "records", &[1, r, 8], &p.records);
                // WDL is from anchor zero's side to move, independently of
                // the query's owner/line perspective.
                let perspective = match i.anchors[0].metadata[0] {
                    1. => 1,
                    0. => -1,
                    _ => return Err(integrity("compact side-to-move metadata invalid")),
                };
                (
                    i.role,
                    l,
                    a,
                    perspective,
                    i.candidates.clone(),
                    i.records.is_empty(),
                    p.query_line_mask
                        .iter()
                        .chain(&p.record_line_mask)
                        .any(|b| !b),
                    i.divergences.len(),
                )
            }
            PalsNumericalInputV5::LegacyFullLine(i) if profile == "full_line_interaction_v2" => {
                let config = PalsModelConfig::for_profile(PalsModelProfile::FullLineInteractionV2);
                let p = i
                    .prepare_tensors(&config)
                    .map_err(|e| integrity(&e.to_string()))?;
                let (r, c, d) = (
                    p.record_mask.len(),
                    p.candidate_mask.len(),
                    p.divergence_mask.len(),
                );
                let lines = p
                    .full_line
                    .as_ref()
                    .ok_or_else(|| integrity("legacy numeric input has no full-line encoder"))?;
                add_i64(&mut tensors, "board", &[1, 64], &p.board);
                add_f32(&mut tensors, "metadata", &[1, 16], &p.metadata);
                add_bool(&mut tensors, "candidate_mask", &[1, c], &p.candidate_mask);
                add_i64(&mut tensors, "candidates", &[1, c, 3], &p.candidates);
                add_f32(
                    &mut tensors,
                    "divergence_features",
                    &[1, d, 8],
                    &p.divergences,
                );
                add_bool(&mut tensors, "divergence_mask", &[1, d], &p.divergence_mask);
                add_f32(&mut tensors, "query", &[1, 16], &p.query);
                add_bool(
                    &mut tensors,
                    "query_line_mask",
                    &[1, 3, 256],
                    &lines.query_line_mask,
                );
                add_i64(
                    &mut tensors,
                    "query_line_tokens",
                    &[1, 3, 256, 3],
                    &lines.query_line_tokens,
                );
                add_bool(
                    &mut tensors,
                    "record_line_mask",
                    &[1, r, 256],
                    &lines.record_line_mask,
                );
                add_i64(
                    &mut tensors,
                    "record_line_tokens",
                    &[1, r, 256, 3],
                    &lines.record_line_tokens,
                );
                add_bool(&mut tensors, "record_mask", &[1, r], &p.record_mask);
                add_i64(
                    &mut tensors,
                    "record_relations",
                    &[1, r, 2],
                    &lines.record_relations,
                );
                add_f32(&mut tensors, "records", &[1, r, 16], &p.records);
                // Existing Rules metadata axis zero is white-to-move 1/black 0.
                if !matches!(i.metadata[0], 0. | 1.) {
                    return Err(integrity("legacy side-to-move metadata invalid"));
                }
                (
                    i.role,
                    256,
                    1,
                    if i.metadata[0] == 1. { 1 } else { -1 },
                    i.candidates.clone(),
                    i.records.is_empty(),
                    lines
                        .query_line_mask
                        .iter()
                        .chain(&lines.record_line_mask)
                        .any(|b| !b),
                    i.divergence_features.len(),
                )
            }
            _ => {
                return Err(integrity(
                    "historical/compact numerical input domain mismatch",
                ));
            }
        };
    tensors.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    let candidate_count = candidates.len();
    if candidate_count == 0 {
        return Err(integrity("numerical legal-policy case lacks candidates"));
    }
    Ok((
        CaseObservation {
            role,
            bucket,
            anchors,
            perspective,
            candidates,
            same_question_sha256: same_question_digest(input)?,
            empty_records,
            padded_lines,
            tensor_bytes: 0,
            raw_max: 0.,
            policy_max: 0.,
            wdl_max: 0.,
        },
        tensors,
        candidate_count,
        divergences,
    ))
}

/// Coverage may compare candidate permutations only within one exact question.
/// Remove only the candidate list: anchors, board/history, query, selected
/// records, relations, masks, role and input schema remain in the identity.
/// This local coverage key never replaces the captured tensor/cache identity.
fn same_question_digest(input: &PalsNumericalInputV5) -> Result<String, ManifestError> {
    let mut value = serde_json::to_value(input)
        .map_err(|error| integrity(&format!("numerical question encoding: {error}")))?;
    let context = value
        .get_mut("input")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| integrity("numerical question input envelope missing"))?;
    if context.remove("candidates").is_none() {
        return Err(integrity("numerical question candidate field missing"));
    }
    value.sort_all_objects();
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| integrity(&format!("numerical question encoding: {error}")))?;
    let mut hash = Sha256::new();
    frame(&mut hash, b"rz-pals-numerical-same-question-v5/1");
    frame(&mut hash, &bytes);
    Ok(format!("{:x}", hash.finalize()))
}

/// Stable captured-byte protocol; independent Python/native producers use this
/// explicitly versioned framing, not an old semantic cache key.
pub fn pals_numerical_tensor_digest_v5(
    tensors: &[PalsNumericalTensorV5],
    role: PalsRole,
) -> Result<String, ManifestError> {
    if !(1..=32).contains(&tensors.len()) {
        return Err(integrity("numeric tensor roster bound"));
    }
    let mut hash = Sha256::new();
    frame(&mut hash, PALS_NUMERICAL_TENSOR_V5_DOMAIN.as_bytes());
    frame(
        &mut hash,
        match role {
            PalsRole::Proposer => b"proposer",
            PalsRole::Critic => b"critic",
            PalsRole::Validator => b"validator",
        },
    );
    let mut previous: Option<&str> = None;
    for t in tensors {
        if !valid_id(&t.name)
            || previous.is_some_and(|p| p >= t.name.as_str())
            || t.shape.is_empty()
            || t.shape.len() > 4
        {
            return Err(integrity(
                "numeric tensors require unique sorted names and bounded shapes",
            ));
        }
        previous = Some(&t.name);
        let elements = t
            .shape
            .iter()
            .try_fold(1u64, |a, b| if *b > 0 { a.checked_mul(*b) } else { None })
            .ok_or_else(|| integrity("numeric tensor shape overflow/zero"))?;
        let (dtype, width) = match t.dtype {
            PalsNumericalDtypeV5::F32Le => ("f32", 4),
            PalsNumericalDtypeV5::I64Le => ("i64", 8),
            PalsNumericalDtypeV5::BoolU8 => ("bool", 1),
        };
        if elements.checked_mul(width) != Some(t.bytes.len() as u64) || t.bytes.len() > RAW_LIMIT {
            return Err(integrity("numeric tensor dtype/shape/byte length mismatch"));
        }
        if t.dtype == PalsNumericalDtypeV5::F32Le
            && t.bytes
                .chunks_exact(4)
                .any(|c| !f32::from_le_bytes(c.try_into().unwrap()).is_finite())
            || t.dtype == PalsNumericalDtypeV5::BoolU8 && t.bytes.iter().any(|b| *b > 1)
        {
            return Err(integrity("nonfinite numeric input or invalid mask byte"));
        }
        frame(&mut hash, t.name.as_bytes());
        frame(&mut hash, dtype.as_bytes());
        hash.update((t.shape.len() as u64).to_le_bytes());
        for dim in &t.shape {
            hash.update(dim.to_le_bytes());
        }
        frame(&mut hash, &t.bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn add_i64(out: &mut Vec<PalsNumericalTensorV5>, name: &str, shape: &[usize], values: &[i64]) {
    out.push(PalsNumericalTensorV5 {
        name: name.into(),
        dtype: PalsNumericalDtypeV5::I64Le,
        shape: shape.iter().map(|v| *v as u64).collect(),
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    });
}
fn add_f32(out: &mut Vec<PalsNumericalTensorV5>, name: &str, shape: &[usize], values: &[f32]) {
    out.push(PalsNumericalTensorV5 {
        name: name.into(),
        dtype: PalsNumericalDtypeV5::F32Le,
        shape: shape.iter().map(|v| *v as u64).collect(),
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    });
}
fn add_bool(out: &mut Vec<PalsNumericalTensorV5>, name: &str, shape: &[usize], values: &[bool]) {
    out.push(PalsNumericalTensorV5 {
        name: name.into(),
        dtype: PalsNumericalDtypeV5::BoolU8,
        shape: shape.iter().map(|v| *v as u64).collect(),
        bytes: values.iter().map(|v| u8::from(*v)).collect(),
    });
}
fn validate_output(
    raw: &PalsRawOutput,
    role: PalsRole,
    candidates: usize,
    divergences: usize,
) -> Result<(), ManifestError> {
    if raw.candidate_logits.len() != candidates
        || raw.private_latent.len() != 16 * 384
        || raw.divergence_logits.as_ref().map(Vec::len)
            != if role == PalsRole::Critic {
                Some(divergences)
            } else {
                None
            }
        || raw.task_logits.is_some() != (role == PalsRole::Validator)
    {
        return Err(integrity(
            "numeric output shape/private-role ownership differs",
        ));
    }
    for values in [
        &raw.candidate_logits[..],
        &raw.wdl_logits[..],
        &raw.private_latent[..],
        raw.divergence_logits.as_deref().unwrap_or(&[]),
        raw.task_logits.as_ref().map_or(&[][..], |v| &v[..]),
    ] {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(integrity("nonfinite numerical raw output"));
        }
    }
    Ok(())
}
fn compare_raw(reference: &[f32], native: &[f32]) -> Result<f64, ManifestError> {
    if reference.len() != native.len() {
        return Err(integrity("numeric raw shape mismatch"));
    }
    let mut maximum: f64 = 0.;
    for (a, b) in reference.iter().zip(native) {
        if !a.is_finite() || !b.is_finite() {
            return Err(integrity("nonfinite numeric raw output"));
        }
        let difference = (*b as f64 - *a as f64).abs();
        maximum = maximum.max(difference);
        if difference > RAW_ATOL + RAW_RTOL * (*a as f64).abs() {
            return Err(integrity("numeric raw tolerance exceeded"));
        }
    }
    Ok(maximum)
}
fn probabilities(values: &[f32]) -> Result<Vec<f64>, ManifestError> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err(integrity("numeric probability input empty/nonfinite"));
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p = values
        .iter()
        .map(|v| (*v as f64 - maximum).exp())
        .collect::<Vec<_>>();
    let sum: f64 = p.iter().sum();
    if !sum.is_finite() || sum <= 0. {
        return Err(integrity("numeric probability normalization failed"));
    }
    for v in &mut p {
        *v /= sum;
    }
    Ok(p)
}
fn compare_probabilities(reference: &[f32], native: &[f32]) -> Result<f64, ManifestError> {
    if reference.len() != native.len() {
        return Err(integrity("numeric probability shape mismatch"));
    }
    let a = probabilities(reference)?;
    let b = probabilities(native)?;
    let maximum = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    if maximum > PROBABILITY_ATOL {
        return Err(integrity("numeric legal-policy/WDL tolerance exceeded"));
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bindings() -> PalsNumericalBindingsV5 {
        let sha = "11".repeat(32);
        PalsNumericalBindingsV5 {
            model_profile: "virtual_review_v1".into(),
            model_schema: "rovezero.pals-model.v3".into(),
            model_semantics: "rovezero.pals-model-semantics.v3".into(),
            encoding_schema: "rovezero.pals-virtual-review.v1".into(),
            model_semantic_sha256: sha.clone(),
            encoding_sha256: sha.clone(),
            model_epoch: 0,
            weights_sha256: sha.clone(),
            export_manifest_sha256: sha.clone(),
            graphs: BTreeMap::from([
                ("public".into(), sha.clone()),
                ("shared_pc".into(), sha.clone()),
            ]),
            runtime_library_sha256: sha.clone(),
            cuda_libraries: BTreeSet::new(),
            cuda_manifest_sha256: None,
            backend: PalsModelBackendV3::OrtCpu,
            precision: PalsPrecisionV3::Fp32,
            tf32: false,
            source_components: NATIVE_COMPONENTS
                .into_iter()
                .map(|key| (key.into(), sha.clone()))
                .collect(),
        }
    }
    fn case() -> PalsNumericalRawCaseV5 {
        use rz_eval::pals_model::virtual_review::{
            ReviewAnchorInput, ReviewQueryInput, ReviewQueryPurpose,
        };
        let mut metadata = [0.; 16];
        metadata[0] = 1.;
        let candidates = (1..=4)
            .map(|promotion| PalsCandidateToken {
                from: 48,
                to: 56,
                promotion,
            })
            .collect::<Vec<_>>();
        let input = PalsNumericalInputV5::VirtualReview(PalsVirtualReviewInput {
            schema: "rovezero.pals-virtual-review-input.v1".into(),
            role: PalsRole::Proposer,
            anchors: vec![ReviewAnchorInput {
                board: vec![0; 64],
                metadata,
            }],
            records: Vec::new(),
            query: ReviewQueryInput {
                purpose: ReviewQueryPurpose::ProposePolicy,
                perspective: 1,
                prefix: Vec::new(),
                proposal: Vec::new(),
                counter: Vec::new(),
                counter_present: false,
                anchors: [0; 3],
            },
            candidates: candidates.clone(),
            divergences: Vec::new(),
            token_budget: 512,
        });
        let (_, tensors, _, _) = prepare_case_input(&input, "virtual_review_v1").unwrap();
        let sha = pals_numerical_tensor_digest_v5(&tensors, PalsRole::Proposer).unwrap();
        PalsNumericalRawCaseV5 {
            id: "authored-fixture-no-native-execution".into(),
            input,
            input_tensors: tensors,
            input_tensor_sha256: sha,
            candidate_order: candidates,
            raw_output: PalsRawOutput {
                candidate_logits: vec![0.1, 0.2, 0.3, 0.4],
                wdl_logits: [0.1, 0.2, 0.3],
                divergence_logits: None,
                task_logits: None,
                private_latent: vec![0.; 16 * 384],
            },
            physical_completion_id: None,
        }
    }
    #[test]
    fn pure_raw_case_rejects_input_mismatch_candidate_order_missing_head_and_nan() {
        let reference = case();
        assert!(compare_case(&reference, &reference, &bindings()).is_ok());
        let mut native = reference.clone();
        native.input_tensors[0].bytes[0] ^= 1;
        assert!(compare_case(&reference, &native, &bindings()).is_err());
        native = reference.clone();
        if let PalsNumericalInputV5::VirtualReview(input) = &mut native.input {
            input.anchors[0].metadata[1] = -0.;
        }
        assert!(compare_case(&reference, &native, &bindings()).is_err());
        native = reference.clone();
        native.candidate_order.reverse();
        assert!(compare_case(&reference, &native, &bindings()).is_err());
        native = reference.clone();
        native.raw_output.candidate_logits.clear();
        assert!(compare_case(&reference, &native, &bindings()).is_err());
        native = reference.clone();
        native.raw_output.private_latent[0] = f32::NAN;
        assert!(compare_case(&reference, &native, &bindings()).is_err());
        native = reference.clone();
        native.raw_output.task_logits = Some([0.; 7]);
        assert!(compare_case(&reference, &native, &bindings()).is_err());
    }
    #[test]
    fn candidate_order_coverage_requires_the_same_exact_question() {
        let original = case().input;
        let mut permuted = original.clone();
        if let PalsNumericalInputV5::VirtualReview(input) = &mut permuted {
            input.candidates.reverse();
        }
        let original_observation = prepare_case_input(&original, "virtual_review_v1")
            .unwrap()
            .0;
        let permuted_observation = prepare_case_input(&permuted, "virtual_review_v1")
            .unwrap()
            .0;
        assert_eq!(
            original_observation.same_question_sha256,
            permuted_observation.same_question_sha256
        );
        let mut same_question = Coverage::default();
        same_question.add(&original_observation);
        same_question.add(&permuted_observation);
        assert!(same_question.has_candidate_order_variation(role_index(PalsRole::Proposer)));

        for different_anchor in [false, true] {
            let mut other_question = permuted.clone();
            if let PalsNumericalInputV5::VirtualReview(input) = &mut other_question {
                if different_anchor {
                    input.anchors[0].metadata[1] = 1.;
                } else {
                    input.query.perspective = -1;
                }
            }
            let other_observation = prepare_case_input(&other_question, "virtual_review_v1")
                .unwrap()
                .0;
            assert_ne!(
                original_observation.same_question_sha256,
                other_observation.same_question_sha256
            );
            let mut unrelated_questions = Coverage::default();
            unrelated_questions.add(&original_observation);
            unrelated_questions.add(&other_observation);
            assert!(
                !unrelated_questions.has_candidate_order_variation(role_index(PalsRole::Proposer))
            );
        }
    }
    #[test]
    fn reference_cannot_be_promoted_to_cuda_native_completion() {
        let b = bindings();
        let mut run = PalsNumericalRawRunV5 {
            schema_version: 5,
            domain: PALS_NUMERICAL_RAW_V5_DOMAIN.into(),
            run_id: "independent-reference".into(),
            producer: PalsNumericalProducerV5::IndependentPytorchFp32,
            producer_source_component: b.source_components["model_reference"].clone(),
            bindings: b.clone(),
            invocation: PalsNumericalInvocationV5::Fresh,
            batch: 1,
            observed_provider: None,
            public_cache_hits: 0,
            completed_nn_inputs: 0,
            physical_completion_known: false,
            cases: vec![case()],
        };
        assert!(validate_run(&run, &b, PalsNumericalProducerV5::IndependentPytorchFp32).is_ok());
        run.observed_provider = Some("CUDAExecutionProvider".into());
        assert!(validate_run(&run, &b, PalsNumericalProducerV5::IndependentPytorchFp32).is_err());
        run.producer = PalsNumericalProducerV5::RustOrtNative;
        run.observed_provider = None;
        assert!(validate_run(&run, &b, PalsNumericalProducerV5::RustOrtNative).is_err());
        run.producer_source_component = b.source_components["model_native"].clone();
        run.observed_provider = Some("CPUExecutionProvider".into());
        run.completed_nn_inputs = 2;
        run.physical_completion_known = true;
        run.cases[0].physical_completion_id = Some("actual-native-completion".into());
        assert!(validate_run(&run, &b, PalsNumericalProducerV5::RustOrtNative).is_ok());
        run.batch = 16;
        assert!(validate_run(&run, &b, PalsNumericalProducerV5::RustOrtNative).is_err());
        run.batch = 1;
        run.invocation = PalsNumericalInvocationV5::ApproxWarmSeeded;
        assert!(validate_run(&run, &b, PalsNumericalProducerV5::RustOrtNative).is_err());
    }
    #[test]
    fn raw_and_probability_thresholds_are_independent() {
        assert!(compare_raw(&[1.], &[1.001]).is_ok());
        assert!(compare_raw(&[0.], &[0.00011]).is_err());
        assert!(compare_raw(&[1., 1.], &[1.001, 0.999]).is_ok());
        assert!(compare_probabilities(&[1., 1.], &[1.001, 0.999]).is_err());
        assert_eq!(
            compare_probabilities(&[10000., 10000.], &[10001., 10001.]).unwrap(),
            0.
        );
        assert!(compare_raw(&[f32::NAN], &[0.]).is_err());
        assert!(compare_probabilities(&[], &[]).is_err());
    }
    #[test]
    fn captured_bytes_keep_signed_zero_and_reject_invalid_masks_and_shapes() {
        let mut tensors = Vec::new();
        add_f32(&mut tensors, "query", &[1, 1], &[0.]);
        let positive = pals_numerical_tensor_digest_v5(&tensors, PalsRole::Proposer).unwrap();
        assert_ne!(
            positive,
            pals_numerical_tensor_digest_v5(&tensors, PalsRole::Critic).unwrap()
        );
        tensors[0].bytes = (-0f32).to_le_bytes().to_vec();
        assert_ne!(
            positive,
            pals_numerical_tensor_digest_v5(&tensors, PalsRole::Proposer).unwrap()
        );
        tensors[0].bytes = f32::NAN.to_le_bytes().to_vec();
        assert!(pals_numerical_tensor_digest_v5(&tensors, PalsRole::Proposer).is_err());
        tensors.clear();
        add_bool(&mut tensors, "mask", &[1, 1], &[true]);
        tensors[0].bytes[0] = 2;
        assert!(pals_numerical_tensor_digest_v5(&tensors, PalsRole::Proposer).is_err());
        tensors[0].bytes[0] = 1;
        tensors[0].shape = vec![1, 2];
        assert!(pals_numerical_tensor_digest_v5(&tensors, PalsRole::Proposer).is_err());
    }
    #[test]
    fn status_only_and_historical_receipts_cannot_be_new_acceptance() {
        assert!(
            parse::<PalsNumericalAcceptanceV5>(
                br#"{"status":"passed","cuda":"passed"}"#,
                ACCEPTANCE_LIMIT
            )
            .is_err()
        );
        assert!(
            parse::<PalsNumericalAcceptanceV5>(
                br#"{"schema":"rovezero.pals-model.v2","status":"passed"}"#,
                ACCEPTANCE_LIMIT
            )
            .is_err()
        );
        assert!(
            parse::<PalsNumericalRawRunV5>(
                br#"{"schema_version":5,"schema_version":5}"#,
                RAW_LIMIT
            )
            .is_err()
        );
    }
    #[test]
    fn changed_raw_evidence_hash_or_missing_data_is_not_accepted() {
        let bytes = b"independent actual fixture bytes";
        let artifact = ArtifactRef {
            path: "numeric/raw.json".into(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            bytes: bytes.len() as u64,
            source: "https://example.org/authored-numerical-fixture".into(),
            license: "MIT".into(),
        };
        assert!(verify_bytes(&artifact, bytes, RAW_LIMIT).is_ok());
        assert!(verify_bytes(&artifact, b"changed", RAW_LIMIT).is_err());
        assert!(verify_bytes(&artifact, &[], RAW_LIMIT).is_err());
    }
}
