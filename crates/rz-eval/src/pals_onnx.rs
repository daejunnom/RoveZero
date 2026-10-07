//! Frozen, P/C-only PALS ONNX sessions. Chess authority remains with Rules.
//!
//! A synchronous successful Run is a physical completion fence. Failed CUDA
//! execution permanently quarantines this owner and its active native values.
//! Logical cancellation belongs to the existing runtime and never frees these
//! buffers. Public memory is exact and role-neutral; role latents start fresh.
use crate::asset::{self, parse_sha256};
use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use crate::onnx::{OrtRuntime, Provider};
use crate::pals_model::{
    PalsModelConfig, PalsModelInput, PalsRawOutput, PalsRole, PreparedPalsTensors,
    PALS_MODEL_SCHEMA, V_TASK_NAMES,
};
use crate::worker::{PhysicalRun, SingleWorker};
use ort::execution_providers::{
    ArenaExtendStrategy, CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Component, Path};

const MAX_MANIFEST_BYTES: usize = 128 * 1024;
const MAX_GRAPH_BYTES: usize = 256 * 1024 * 1024;
const MAX_MODEL_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct PalsOnnxConfig {
    pub provider: Provider,
    pub intra_threads: usize,
    /// The first exact cache has one bounded public-memory entry. Set false for
    /// independent fresh-versus-cached correctness comparisons.
    pub cache_public_memory: bool,
    /// Device-resident public K/V has a separate explicit admission. The
    /// feature only compiles its code; selecting CPU never enables it.
    pub device_public_memory: bool,
}
impl PalsOnnxConfig {
    pub fn cpu() -> Self {
        Self {
            provider: Provider::Cpu,
            intra_threads: 2,
            cache_public_memory: true,
            device_public_memory: false,
        }
    }
    fn validate(&self) -> Result<(), BackendError> {
        if !(1..=2).contains(&self.intra_threads) {
            return Err(fail(
                K::InvalidInput,
                S::Admission,
                "PALS CPU thread declaration must be 1..=2",
            ));
        }
        if let Provider::Cuda {
            device_id,
            arena_bytes,
        } = self.provider
        {
            if device_id < 0 || arena_bytes == 0 || arena_bytes > 6 * 1024 * 1024 * 1024 {
                return Err(fail(
                    K::InvalidInput,
                    S::Admission,
                    "PALS CUDA device/memory declaration is invalid",
                ));
            }
        }
        if self.device_public_memory
            && (!cfg!(feature = "experimental-io-binding")
                || !matches!(self.provider, Provider::Cuda { .. }))
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "device PALS memory requires explicit CUDA and the I/O binding feature",
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportManifest {
    schema: String,
    #[serde(default)]
    model_semantics: Option<String>,
    #[serde(default)]
    layout: Option<String>,
    #[serde(default)]
    layout_revision: Option<u32>,
    #[serde(default)]
    role_batching: Option<String>,
    #[serde(default)]
    rules_input_profile: Option<String>,
    #[serde(default)]
    rules_input_semantic_sha256: Option<String>,
    #[serde(default)]
    rules_encoder_source_sha256: Option<String>,
    #[serde(default)]
    rules_profile_descriptor_sha256: Option<String>,
    #[serde(default)]
    rules_profile_canonical_sha256: Option<String>,
    #[serde(default)]
    rules_input_declaration: Option<serde_json::Value>,
    #[serde(default)]
    learned_input_compatibility: Option<String>,
    #[serde(default)]
    reader_initializer_bank: Option<ReaderInitializerBank>,
    config: PalsModelConfig,
    checkpoint_sha256: String,
    trained: bool,
    training_steps: u64,
    roles: Vec<String>,
    validator_present: bool,
    graphs: Vec<GraphManifest>,
    runtime: RuntimeManifest,
    candidate_promotion: BTreeMap<String, String>,
    wdl_perspective: String,
    task_names: Vec<String>,
    numeric_status: String,
    cuda_status: String,
}
/// Serialized ONNX initializer ownership only. Native optimization can still
/// allocate transformed/prepacked copies and is not measured by this audit.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReaderInitializerBank {
    pub scope: String,
    pub layout_revision: u32,
    pub shared_parameters: Vec<SharedParameter>,
    pub unique_shared_parameters: usize,
    pub shared_weight_bytes: u64,
    pub if_routes: usize,
    pub branch_local_initializers: usize,
    pub branch_operations: Vec<BranchOperations>,
    pub inactive_private_execution: String,
    pub ort_prepack_copies: String,
    pub device_residency_sharing: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SharedParameter {
    pub parameter: String,
    pub initializer: String,
    pub bytes: u64,
    pub sha256: String,
    pub source_parameter_sha256: String,
    pub shape: Vec<usize>,
}
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BranchOperations {
    pub route: String,
    pub branch: String,
    pub nodes: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    SeparatePc,
    SharedPcIf,
}
impl ExportManifest {
    fn resolved_layout(&self) -> Result<Layout, BackendError> {
        match (
            self.schema.as_str(),
            self.layout.as_deref(),
            self.model_semantics.as_deref(),
        ) {
            (PALS_MODEL_SCHEMA, None | Some("separate_pc"), None | Some(PALS_MODEL_SCHEMA))
                if self.reader_initializer_bank.is_none() =>
            {
                Ok(Layout::SeparatePc)
            }
            ("rovezero.pals-model.v2", Some("shared_pc_if"), Some(PALS_MODEL_SCHEMA))
                if self.layout_revision == Some(1)
                    && self.role_batching.as_deref()
                        == Some("one_scalar_role_per_physical_batch") =>
            {
                Ok(Layout::SharedPcIf)
            }
            _ => Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "unsupported explicit PALS graph layout/semantics",
            )),
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifest {
    onnxruntime: String,
    rust_ort: String,
    compatibility: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphManifest {
    file: String,
    sha256: String,
    role: String,
    inputs: Vec<TensorManifest>,
    outputs: Vec<TensorManifest>,
    opset: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorManifest {
    name: String,
    dtype: String,
    shape: Vec<serde_json::Value>,
}

#[derive(Clone, Copy)]
struct Interface {
    name: &'static str,
    dtype: TensorElementType,
    shape: &'static [i64],
}
fn public_inputs() -> Vec<Interface> {
    vec![
        spec("board", TensorElementType::Int64, &[-1, 64]),
        spec("metadata", TensorElementType::Float32, &[-1, 16]),
        spec("records", TensorElementType::Float32, &[-1, -1, 16]),
        spec("record_mask", TensorElementType::Bool, &[-1, -1]),
    ]
}
fn memory_outputs() -> Vec<Interface> {
    vec![
        spec("memory_key", TensorElementType::Float32, &[-1, 2, -1, 64]),
        spec("memory_value", TensorElementType::Float32, &[-1, 2, -1, 64]),
        spec("memory_mask", TensorElementType::Bool, &[-1, -1]),
    ]
}
fn role_inputs(critic: bool) -> Vec<Interface> {
    let mut result = memory_outputs();
    result.extend([
        spec("candidates", TensorElementType::Int64, &[-1, -1, 3]),
        spec("candidate_mask", TensorElementType::Bool, &[-1, -1]),
        spec("query", TensorElementType::Float32, &[-1, 16]),
    ]);
    if critic {
        result.extend([
            spec(
                "divergence_features",
                TensorElementType::Float32,
                &[-1, -1, 8],
            ),
            spec("divergence_mask", TensorElementType::Bool, &[-1, -1]),
        ]);
    }
    result
}
fn role_outputs(critic: bool) -> Vec<Interface> {
    let mut result = vec![
        spec("candidate_logits", TensorElementType::Float32, &[-1, -1]),
        spec("wdl_logits", TensorElementType::Float32, &[-1, 3]),
        spec("private_latent", TensorElementType::Float32, &[-1, 16, 384]),
    ];
    if critic {
        result.push(spec(
            "divergence_logits",
            TensorElementType::Float32,
            &[-1, -1],
        ));
    }
    result
}
fn shared_inputs() -> Vec<Interface> {
    let mut result = vec![spec("role_is_critic", TensorElementType::Bool, &[])];
    result.extend(role_inputs(true));
    result
}
fn shared_outputs() -> Vec<Interface> {
    let mut result = role_outputs(true);
    result.push(spec("is_critic", TensorElementType::Bool, &[]));
    result
}
const fn spec(name: &'static str, dtype: TensorElementType, shape: &'static [i64]) -> Interface {
    Interface { name, dtype, shape }
}
fn fail(kind: K, stage: S, detail: &'static str) -> BackendError {
    BackendError::new(kind, stage, detail)
}
fn native(code: CauseCode, detail: &'static str, error: ort::Error) -> BackendError {
    fail(K::BackendFailure, S::Backend, detail).with_ort_cause(code, error)
}
fn model_input(error: impl std::fmt::Display) -> BackendError {
    fail(
        K::InvalidInput,
        S::Admission,
        "PALS model request violates its bounded tensor contract",
    )
    .with_external_cause(CauseCode::OutputValidation, &error)
}
fn validate_role_tag(shape: &[i64], values: &[bool], critic: bool) -> Result<(), BackendError> {
    if !shape.is_empty() || values != [critic] {
        return Err(fail(
            K::IdentityMismatch,
            S::Output,
            "shared P/C output role tag differs from admitted request",
        ));
    }
    Ok(())
}
#[derive(Default)]
enum CudaMappingAudit {
    #[default]
    AwaitingFirstRun,
    Confirmed,
    Rejected(BackendError),
}
impl CudaMappingAudit {
    fn allow_run(&self) -> Result<(), BackendError> {
        match self {
            Self::Rejected(cause) => Err(cause.clone()),
            _ => Ok(()),
        }
    }
    fn after_run(
        &mut self,
        completed: bool,
        full_audit: impl FnOnce() -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        self.allow_run()?;
        if !completed || matches!(self, Self::Confirmed) {
            return Ok(());
        }
        match full_audit() {
            Ok(()) => {
                *self = Self::Confirmed;
                Ok(())
            }
            Err(mut failure) => {
                failure.detail = "PALS CUDA full mapping audit failed after completed native Run";
                *self = Self::Rejected(failure.clone());
                Err(failure)
            }
        }
    }
    fn final_audit(
        &mut self,
        full_audit: impl FnOnce() -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        self.allow_run()?;
        if !matches!(self, Self::Confirmed) {
            return Err(fail(
                K::BackendUnavailable,
                S::Backend,
                "PALS full CUDA audit requires a completed native Run",
            ));
        }
        match full_audit() {
            Ok(()) => Ok(()),
            Err(mut failure) => {
                failure.detail =
                    "PALS CUDA final full mapping audit failed after completed native Run";
                *self = Self::Rejected(failure.clone());
                Err(failure)
            }
        }
    }
}

fn validate_declared_interface(
    values: &[TensorManifest],
    expected: &[Interface],
) -> Result<(), BackendError> {
    if values.len() != expected.len() {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "PALS graph descriptor interface differs",
        ));
    }
    for expected in expected {
        let matching: Vec<_> = values.iter().filter(|v| v.name == expected.name).collect();
        if matching.len() != 1 {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS graph descriptor names differ",
            ));
        }
        let value = matching[0];
        let dtype = match expected.dtype {
            TensorElementType::Float32 => "FLOAT",
            TensorElementType::Int64 => "INT64",
            TensorElementType::Bool => "BOOL",
            _ => unreachable!(),
        };
        if value.dtype != dtype
            || value.shape.len() != expected.shape.len()
            || !value
                .shape
                .iter()
                .zip(expected.shape)
                .all(|(actual, expected)| {
                    if *expected == -1 {
                        actual
                            .as_str()
                            .is_some_and(|s| !s.is_empty() && s.len() <= 64)
                    } else {
                        actual.as_i64() == Some(*expected)
                    }
                })
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS descriptor tensor dtype/shape differs",
            ));
        }
    }
    Ok(())
}
fn validate_manifest(manifest: &ExportManifest) -> Result<[u8; 32], BackendError> {
    manifest.config.validate().map_err(model_input)?;
    let layout = manifest.resolved_layout()?;
    match (
        &manifest.rules_input_profile,
        &manifest.rules_input_semantic_sha256,
    ) {
        (None, None) if layout == Layout::SeparatePc => {}
        (Some(profile), Some(digest)) if profile == "rz-pals-rules-fields-v1" => {
            parse_sha256(digest)?;
        }
        _ => {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS Rules semantic profile/digest is missing or invalid",
            ))
        }
    }
    for digest in [
        &manifest.rules_encoder_source_sha256,
        &manifest.rules_profile_descriptor_sha256,
        &manifest.rules_profile_canonical_sha256,
    ]
    .into_iter()
    .flatten()
    {
        parse_sha256(digest)?;
    }
    if let Some(declaration) = &manifest.rules_input_declaration {
        if declaration
            .get("rules_input_profile")
            .and_then(serde_json::Value::as_str)
            != manifest.rules_input_profile.as_deref()
            || declaration
                .get("rules_input_semantic_sha256")
                .and_then(serde_json::Value::as_str)
                != manifest.rules_input_semantic_sha256.as_deref()
            || declaration
                .get("rules_encoder_source_sha256")
                .and_then(serde_json::Value::as_str)
                != manifest.rules_encoder_source_sha256.as_deref()
            || declaration.get("config")
                != Some(&serde_json::to_value(&manifest.config).map_err(model_input)?)
            || manifest.learned_input_compatibility.as_deref()
                != Some("unverified_declaration_only")
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS Rules declaration contradicts registered tensor semantics",
            ));
        }
    }
    if layout == Layout::SharedPcIf {
        if manifest.rules_encoder_source_sha256.is_none() {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "shared P/C artifact has no Rules encoder provenance digest",
            ));
        }
        validate_reader_bank(manifest.reader_initializer_bank.as_ref().ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Asset,
                "shared P/C artifact has no initializer ownership audit",
            )
        })?)?;
    }
    let promotion = ["none", "queen", "rook", "bishop", "knight"]
        .iter()
        .enumerate()
        .map(|(i, name)| (i.to_string(), name.to_string()))
        .collect::<BTreeMap<_, _>>();
    let mut roles = manifest.roles.clone();
    roles.sort();
    if manifest.validator_present
        || roles != ["critic", "proposer"]
        || manifest.graphs.len() != if layout == Layout::SharedPcIf { 2 } else { 3 }
        || manifest.wdl_perspective != "input_side_to_move"
        || !manifest
            .task_names
            .iter()
            .map(String::as_str)
            .eq(V_TASK_NAMES)
        || manifest.candidate_promotion != promotion
        || manifest.runtime.onnxruntime != "1.22.0"
        || manifest.runtime.rust_ort != "2.0.0-rc.10"
        || manifest.runtime.compatibility != "requires_actual_numeric_check"
        || manifest.numeric_status != "not_run"
        || manifest.cuda_status != "not_run"
        || (manifest.trained && manifest.training_steps == 0)
        || (!manifest.trained && manifest.training_steps != 0)
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "expected registered P/C-only FP32 PALS export; V is not a product role",
        ));
    }
    let expected_roles: &[&str] = if layout == Layout::SharedPcIf {
        &["public", "shared_pc"]
    } else {
        &["public", "proposer", "critic"]
    };
    for role in expected_roles.iter().copied() {
        let graphs: Vec<_> = manifest.graphs.iter().filter(|g| g.role == role).collect();
        if graphs.len() != 1 {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS public/P/C graph set differs",
            ));
        }
        let graph = graphs[0];
        let expected_file = if role == "public" {
            "public_memory.onnx".to_owned()
        } else if role == "shared_pc" {
            "shared_pc_if.onnx".to_owned()
        } else {
            format!("role_{role}.onnx")
        };
        if graph.file != expected_file
            || graph.opset != 17
            || !matches!(
                Path::new(&graph.file)
                    .components()
                    .collect::<Vec<_>>()
                    .as_slice(),
                [Component::Normal(_)]
            )
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS graph file/opset is unsupported",
            ));
        }
        parse_sha256(&graph.sha256)?;
        let critic = role == "critic";
        validate_declared_interface(
            &graph.inputs,
            &if role == "public" {
                public_inputs()
            } else if role == "shared_pc" {
                shared_inputs()
            } else {
                role_inputs(critic)
            },
        )?;
        validate_declared_interface(
            &graph.outputs,
            &if role == "public" {
                memory_outputs()
            } else if role == "shared_pc" {
                shared_outputs()
            } else {
                role_outputs(critic)
            },
        )?;
    }
    parse_sha256(&manifest.checkpoint_sha256)
}
fn validate_reader_bank(bank: &ReaderInitializerBank) -> Result<(), BackendError> {
    let mut parameters = std::collections::BTreeSet::new();
    let mut initializers = std::collections::BTreeSet::new();
    let mut total = 0_u64;
    for entry in &bank.shared_parameters {
        let elements = entry
            .shape
            .iter()
            .try_fold(1_u64, |n, dim| n.checked_mul(*dim as u64));
        if entry.parameter.is_empty()
            || entry.initializer.is_empty()
            || entry.parameter.len() > 256
            || entry.initializer.len() > 256
            || entry.shape.len() > 8
            || entry.shape.contains(&0)
            || elements.and_then(|n| n.checked_mul(4)) != Some(entry.bytes)
            || !parameters.insert(entry.parameter.as_str())
            || !initializers.insert(entry.initializer.as_str())
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "shared initializer names/shapes/counts differ",
            ));
        }
        parse_sha256(&entry.sha256)?;
        parse_sha256(&entry.source_parameter_sha256)?;
        total = total.checked_add(entry.bytes).ok_or_else(|| {
            fail(
                K::ResourceExhausted,
                S::Asset,
                "shared initializer budget overflow",
            )
        })?;
    }
    let mut routes = BTreeMap::<&str, std::collections::BTreeSet<&str>>::new();
    for branch in &bank.branch_operations {
        if branch.route.is_empty()
            || branch.route.len() > 128
            || branch.nodes == 0
            || !matches!(branch.branch.as_str(), "then_branch" | "else_branch")
            || !routes
                .entry(&branch.route)
                .or_default()
                .insert(&branch.branch)
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "shared If branch ownership audit differs",
            ));
        }
    }
    if bank.scope != "onnx_serialized_initializers_only"
        || bank.layout_revision != 1
        || bank.shared_parameters.is_empty()
        || bank.unique_shared_parameters != parameters.len()
        || total != bank.shared_weight_bytes
        || total > MAX_MODEL_BYTES
        || bank.if_routes != 6
        || routes.len() != bank.if_routes
        || routes.values().any(|branches| branches.len() != 2)
        || bank.branch_local_initializers != 0
        || bank.inactive_private_execution != "requires_runtime_profile"
        || bank.ort_prepack_copies != "unknown"
        || bank.device_residency_sharing != "unknown"
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "shared P/C initializer audit is unsupported",
        ));
    }
    Ok(())
}
fn validate_session(
    session: &Session,
    graph: &GraphManifest,
    manifest: &ExportManifest,
) -> Result<(), BackendError> {
    let inputs = if graph.role == "public" {
        public_inputs()
    } else if graph.role == "shared_pc" {
        shared_inputs()
    } else {
        role_inputs(graph.role == "critic")
    };
    let outputs = if graph.role == "public" {
        memory_outputs()
    } else if graph.role == "shared_pc" {
        shared_outputs()
    } else {
        role_outputs(graph.role == "critic")
    };
    let valid = |name: &str, ty: &ValueType, expected: &[Interface]| {
        expected.iter().any(|expected| {
        name == expected.name && matches!(ty, ValueType::Tensor { ty, shape, .. } if *ty == expected.dtype && shape.as_ref() == expected.shape)
    })
    };
    if session.inputs.len() != inputs.len()
        || session.outputs.len() != outputs.len()
        || !session
            .inputs
            .iter()
            .all(|v| valid(&v.name, &v.input_type, &inputs))
        || !session
            .outputs
            .iter()
            .all(|v| valid(&v.name, &v.output_type, &outputs))
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "loaded PALS ONNX tensor interface differs",
        ));
    }
    let metadata = session
        .metadata()
        .map_err(|e| native(CauseCode::ModelLoad, "cannot inspect PALS ONNX metadata", e))?;
    let steps = manifest.training_steps.to_string();
    for (key, value) in [
        ("schema", manifest.schema.as_str()),
        ("role", graph.role.as_str()),
        ("checkpoint_sha256", manifest.checkpoint_sha256.as_str()),
        ("trained", if manifest.trained { "true" } else { "false" }),
        ("training_steps", steps.as_str()),
        ("precision", "fp32"),
        ("expected_ort", "1.22.0"),
        ("public_kv", "shared_role_neutral"),
    ] {
        if metadata
            .custom(key)
            .map_err(|e| native(CauseCode::ModelLoad, "cannot read PALS ONNX metadata", e))?
            .as_deref()
            != Some(value)
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS ONNX metadata disagrees with registered export",
            ));
        }
    }
    let shared = manifest.resolved_layout()? == Layout::SharedPcIf;
    for (key, expected) in [
        ("model_semantics", manifest.model_semantics.as_deref()),
        ("layout", manifest.layout.as_deref()),
        (
            "rules_input_profile",
            manifest.rules_input_profile.as_deref(),
        ),
        (
            "rules_input_semantic_sha256",
            manifest.rules_input_semantic_sha256.as_deref(),
        ),
        (
            "rules_encoder_source_sha256",
            manifest.rules_encoder_source_sha256.as_deref(),
        ),
    ] {
        if let Some(expected) = expected {
            if metadata
                .custom(key)
                .map_err(|e| {
                    native(
                        CauseCode::ModelLoad,
                        "cannot read PALS semantic metadata",
                        e,
                    )
                })?
                .as_deref()
                != Some(expected)
            {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Asset,
                    "PALS graph semantic metadata disagrees with export",
                ));
            }
        }
    }
    if shared
        && metadata
            .custom("layout_revision")
            .map_err(|e| native(CauseCode::ModelLoad, "cannot read PALS layout revision", e))?
            .as_deref()
            != Some("1")
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS graph layout revision differs",
        ));
    }
    Ok(())
}
fn load_session(bytes: Vec<u8>, config: PalsOnnxConfig) -> Result<Session, BackendError> {
    let configure = |e| {
        native(
            CauseCode::SessionConfiguration,
            "PALS ORT session configuration failed",
            e,
        )
    };
    let mut builder = Session::builder()
        .map_err(configure)?
        .with_no_environment_execution_providers()
        .map_err(configure)?
        .with_intra_threads(config.intra_threads)
        .map_err(configure)?
        .with_inter_threads(1)
        .map_err(configure)?
        .with_parallel_execution(false)
        .map_err(configure)?
        .with_intra_op_spinning(false)
        .map_err(configure)?
        .with_inter_op_spinning(false)
        .map_err(configure)?
        .with_memory_pattern(false)
        .map_err(configure)?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(configure)?;
    builder = match config.provider {
        Provider::Cpu => builder
            .with_execution_providers([CPUExecutionProvider::default()
                .with_arena_allocator(false)
                .build()
                .error_on_failure()])
            .map_err(configure)?,
        Provider::Cuda {
            device_id,
            arena_bytes,
        } => {
            let cuda = CUDAExecutionProvider::default();
            if !cuda.is_available().map_err(configure)? {
                return Err(fail(
                    K::BackendUnavailable,
                    S::Backend,
                    "PALS CUDA provider unavailable; CPU fallback forbidden",
                ));
            }
            builder
                .with_config_entry("session.disable_cpu_ep_fallback", "1")
                .map_err(configure)?
                .with_execution_providers([cuda
                    .with_device_id(device_id)
                    .with_memory_limit(arena_bytes)
                    .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                    .with_tf32(false)
                    .with_cuda_graph(false)
                    .build()
                    .error_on_failure()])
                .map_err(configure)?
        }
    };
    // Default native copying; no direct-reference initializer option is enabled.
    // Serialized owned bytes are freed immediately after successful/failed load.
    builder
        .commit_from_memory(&bytes)
        .map_err(|e| native(CauseCode::ModelLoad, "PALS ONNX loading failed", e))
}

struct ActiveInputs {
    // ONNX If consumes its scalar condition on CPU, including CUDA sessions.
    role_is_critic: Tensor<bool>,
    board: Tensor<i64>,
    metadata: Tensor<f32>,
    records: Tensor<f32>,
    record_mask: Tensor<bool>,
    candidates: Tensor<i64>,
    candidate_mask: Tensor<bool>,
    query: Tensor<f32>,
    divergences: Tensor<f32>,
    divergence_mask: Tensor<bool>,
}
impl ActiveInputs {
    fn new(value: PreparedPalsTensors, critic: bool) -> Result<Self, BackendError> {
        let tensor_error = |e| native(CauseCode::TensorCreate, "cannot own PALS input tensor", e);
        let r = value.record_mask.len();
        let c = value.candidate_mask.len();
        let d = value.divergence_mask.len();
        Ok(Self {
            role_is_critic: Tensor::from_array((Vec::<usize>::new(), vec![critic]))
                .map_err(tensor_error)?,
            board: Tensor::from_array(([1, 64], value.board)).map_err(tensor_error)?,
            metadata: Tensor::from_array(([1, 16], value.metadata)).map_err(tensor_error)?,
            records: Tensor::from_array(([1, r, 16], value.records)).map_err(tensor_error)?,
            record_mask: Tensor::from_array(([1, r], value.record_mask)).map_err(tensor_error)?,
            candidates: Tensor::from_array(([1, c, 3], value.candidates)).map_err(tensor_error)?,
            candidate_mask: Tensor::from_array(([1, c], value.candidate_mask))
                .map_err(tensor_error)?,
            query: Tensor::from_array(([1, 16], value.query)).map_err(tensor_error)?,
            divergences: Tensor::from_array(([1, d, 8], value.divergences))
                .map_err(tensor_error)?,
            divergence_mask: Tensor::from_array(([1, d], value.divergence_mask))
                .map_err(tensor_error)?,
        })
    }
}
struct PublicMemory {
    key: [u8; 32],
    tokens: usize,
    memory_key: Tensor<f32>,
    memory_value: Tensor<f32>,
    mask: Tensor<bool>,
}
/// A copied diagnostic witness after physical completion, never a product
/// private latent or a second inference. Device-only K/V has no host witness.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsPublicMemoryWitness {
    pub tokens: usize,
    pub memory_key: Vec<f32>,
    pub memory_value: Vec<f32>,
    pub memory_mask: Vec<bool>,
}

/// Cache reset is a physical-worker command so a completed NewGame response
/// proves the old cache was retired after the preceding native execution.
/// It neither reloads weights nor invokes a neural graph.
// Exactly one exclusive physical worker owns at most one in-flight command.
// Keep its bounded input inline rather than adding a Box allocation to every
// role request merely to match the zero-payload control variants' size.
#[allow(clippy::large_enum_variant)]
pub enum PalsNativeCommand {
    Evaluate(PalsModelInput),
    NewGame,
    SnapshotStats,
    VerifyRuntime,
}
pub enum PalsNativeResult {
    Evaluation(PalsRawOutput),
    NewGame,
    Stats(PalsBackendStats),
    RuntimeVerified,
}
/// Native graph evidence, cumulative across games for one frozen backend.
/// These are typed counters; a caller must choose its own receipt schema.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PalsBackendStats {
    pub admitted_role_requests: u64,
    pub public_cache_hits: u64,
    pub public_cache_misses: u64,
    pub public_nn_runs_attempted: u64,
    pub public_nn_runs_completed: u64,
    pub public_nn_runs_failed_known: u64,
    pub role_nn_runs_attempted: u64,
    pub role_nn_runs_completed: u64,
    pub role_nn_runs_failed_known: u64,
    /// Physical B1 inputs from both public and role graphs that returned and
    /// completed their physical fence; output validation is a separate count.
    pub completed_nn_inputs: u64,
    /// Full shape/finite/mask acceptance in the host-cache path. Device K/V
    /// currently has no independent host witness, so this stays zero there.
    pub validated_public_outputs: u64,
    pub validated_role_outputs: u64,
    pub new_game_resets: u64,
    pub live_public_cache_entries: u64,
}
impl PalsBackendStats {
    pub fn validate(&self) -> Result<(), BackendError> {
        if self.live_public_cache_entries > 1
            || self
                .public_nn_runs_completed
                .checked_add(self.public_nn_runs_failed_known)
                .is_none_or(|n| n > self.public_nn_runs_attempted)
            || self
                .role_nn_runs_completed
                .checked_add(self.role_nn_runs_failed_known)
                .is_none_or(|n| n > self.role_nn_runs_attempted)
            || self
                .public_nn_runs_completed
                .checked_add(self.role_nn_runs_completed)
                != Some(self.completed_nn_inputs)
            || self.validated_public_outputs > self.public_nn_runs_completed
            || self.validated_role_outputs > self.role_nn_runs_completed
            || self.role_nn_runs_attempted > self.admitted_role_requests
            || self
                .public_cache_hits
                .checked_add(self.public_cache_misses)
                .is_none_or(|n| n > self.admitted_role_requests)
        {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "PALS native graph counters violate completion accounting",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsGraphIdentity {
    pub role: String,
    pub sha256: [u8; 32],
    /// Actual hash-verified serialized bytes read into native session loading;
    /// this is not an estimate of optimized native/session/VRAM residency.
    pub serialized_bytes: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsSessionResidency {
    pub graphs: Vec<PalsGraphIdentity>,
    pub native_sessions: usize,
    pub layout: String,
    pub reader_initializer_bank: Option<ReaderInitializerBank>,
    /// v1 has independent P/C sessions (Some(false)). v2 has one serialized
    /// reader bank, but optimized native/prepacked storage sharing is unknown
    /// (None). This field never attests actual GPU/VRAM residency.
    pub role_reader_weights_shared: Option<bool>,
    pub native_resident_parameter_bytes: Option<u64>,
    pub vram_peak_bytes: Option<u64>,
}
/// A session's execution weight epoch never changes. It has exactly one cache
/// slot, one active input slot and one exclusive caller/physical worker.
pub struct PalsOnnxBackend {
    public: Option<Session>,
    proposer: Option<Session>,
    critic: Option<Session>,
    shared_pc: Option<Session>,
    layout: Layout,
    runtime: OrtRuntime,
    config: PalsOnnxConfig,
    model_config: PalsModelConfig,
    epoch: [u8; 32],
    manifest_digest: [u8; 32],
    trained: bool,
    rules_input_profile: Option<String>,
    rules_input_semantic_sha256: Option<[u8; 32]>,
    residency: PalsSessionResidency,
    stats: PalsBackendStats,
    cuda_mapping_audit: RefCell<CudaMappingAudit>,
    memory: Option<PublicMemory>,
    active: Option<ActiveInputs>,
    #[cfg(feature = "experimental-io-binding")]
    device_memory: Option<device::DeviceMemory>,
    #[cfg(feature = "experimental-io-binding")]
    device_role: Option<device::DeviceRole>,
    quarantine: Option<BackendError>,
    pub public_encodes: u64,
    pub public_cache_hits: u64,
}
impl PalsOnnxBackend {
    pub fn load(
        path: &Path,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
    ) -> Result<Self, BackendError> {
        config.validate()?;
        if !path.is_absolute() {
            return Err(fail(
                K::InvalidInput,
                S::Asset,
                "PALS manifest path must be absolute",
            ));
        }
        let bytes = asset::read_bounded(path, MAX_MANIFEST_BYTES)?;
        let manifest_digest = asset::sha256(&bytes);
        if manifest_digest != parse_sha256(expected_manifest_sha256)? {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS export manifest hash mismatch",
            ));
        }
        let manifest: ExportManifest = serde_json::from_slice(&bytes).map_err(|e| {
            fail(
                K::UnsupportedModel,
                S::Asset,
                "cannot decode PALS export manifest",
            )
            .with_external_cause(CauseCode::ModelLoad, &e)
        })?;
        let epoch = validate_manifest(&manifest)?;
        let layout = manifest.resolved_layout()?;
        let rules_input_semantic_sha256 = manifest
            .rules_input_semantic_sha256
            .as_deref()
            .map(parse_sha256)
            .transpose()?;
        match (config.provider, runtime.bundle_digest()) {
            (Provider::Cpu, Some(_)) | (Provider::Cuda { .. }, None) => {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Backend,
                    "PALS provider and verified native runtime bundle differ",
                ))
            }
            (Provider::Cuda { .. }, Some(_)) => {
                runtime.verify_cuda_dependencies().map_err(|mut failure| {
                    failure.detail =
                        "PALS CUDA dependency mapping audit failed before session creation";
                    failure
                })?
            }
            _ => {}
        }
        let parent = path.parent().ok_or_else(|| {
            fail(
                K::InvalidInput,
                S::Asset,
                "PALS export manifest has no parent",
            )
        })?;
        let mut sessions = BTreeMap::new();
        let mut graph_identities = Vec::with_capacity(3);
        let mut total = 0_u64;
        for graph in &manifest.graphs {
            let graph_bytes = asset::read_bounded(&parent.join(&graph.file), MAX_GRAPH_BYTES)?;
            total = total.checked_add(graph_bytes.len() as u64).ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "PALS graph byte budget overflow",
                )
            })?;
            if total > MAX_MODEL_BYTES {
                return Err(fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "PALS export exceeds total serialized model budget",
                ));
            }
            let graph_digest = asset::sha256(&graph_bytes);
            if graph_digest != parse_sha256(&graph.sha256)? {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Asset,
                    "PALS ONNX graph hash mismatch",
                ));
            }
            graph_identities.push(PalsGraphIdentity {
                role: graph.role.clone(),
                sha256: graph_digest,
                serialized_bytes: graph_bytes.len() as u64,
            });
            let session = load_session(graph_bytes, config)?;
            validate_session(&session, graph, &manifest)?;
            sessions.insert(graph.role.clone(), session);
        }
        Ok(Self {
            public: sessions.remove("public"),
            proposer: sessions.remove("proposer"),
            critic: sessions.remove("critic"),
            shared_pc: sessions.remove("shared_pc"),
            layout,
            runtime,
            config,
            model_config: manifest.config,
            epoch,
            manifest_digest,
            trained: manifest.trained,
            rules_input_profile: manifest.rules_input_profile,
            rules_input_semantic_sha256,
            residency: PalsSessionResidency {
                graphs: graph_identities,
                native_sessions: if layout == Layout::SharedPcIf { 2 } else { 3 },
                layout: if layout == Layout::SharedPcIf {
                    "shared_pc_if"
                } else {
                    "separate_pc"
                }
                .into(),
                reader_initializer_bank: manifest.reader_initializer_bank,
                role_reader_weights_shared: if layout == Layout::SharedPcIf {
                    None
                } else {
                    Some(false)
                },
                native_resident_parameter_bytes: None,
                vram_peak_bytes: None,
            },
            stats: PalsBackendStats::default(),
            cuda_mapping_audit: RefCell::default(),
            memory: None,
            active: None,
            quarantine: None,
            public_encodes: 0,
            public_cache_hits: 0,
            #[cfg(feature = "experimental-io-binding")]
            device_memory: None,
            #[cfg(feature = "experimental-io-binding")]
            device_role: None,
        })
    }
    pub fn model_epoch(&self) -> [u8; 32] {
        self.epoch
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }
    pub fn runtime_binary_digest(&self) -> [u8; 32] {
        self.runtime.binary_digest()
    }
    pub fn runtime_bundle_digest(&self) -> Option<[u8; 32]> {
        self.runtime.bundle_digest()
    }
    /// Export provenance declaration only. Loading a checkpoint cannot attest
    /// that the declared optimizer steps were actually executed.
    pub fn is_trained(&self) -> bool {
        self.trained
    }
    pub fn residency(&self) -> &PalsSessionResidency {
        &self.residency
    }
    pub fn rules_input_profile(&self) -> Option<&str> {
        self.rules_input_profile.as_deref()
    }
    pub fn rules_input_semantic_sha256(&self) -> Option<[u8; 32]> {
        self.rules_input_semantic_sha256
    }
    pub fn public_memory_witness(&self) -> Result<Option<PalsPublicMemoryWitness>, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        if self.config.device_public_memory {
            return Ok(None);
        }
        self.memory
            .as_ref()
            .map(|memory| {
                let copy = |tensor: &Tensor<f32>| -> Result<Vec<f32>, BackendError> {
                    let (shape, values) = tensor.try_extract_tensor::<f32>().map_err(|e| {
                        native(
                            CauseCode::PolicyExtract,
                            "cannot read completed public-memory witness",
                            e,
                        )
                    })?;
                    if shape.as_ref() != [1, 2, memory.tokens as i64, 64]
                        || values.iter().any(|value| !value.is_finite())
                    {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "completed public-memory witness shape/finite failure",
                        ));
                    }
                    Ok(values.to_vec())
                };
                let (shape, values) = memory.mask.try_extract_tensor::<bool>().map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "cannot read completed public-memory mask witness",
                        e,
                    )
                })?;
                if shape.as_ref() != [1, memory.tokens as i64] {
                    return Err(fail(
                        K::NumericalFailure,
                        S::Output,
                        "completed public-memory mask witness shape differs",
                    ));
                }
                Ok(PalsPublicMemoryWitness {
                    tokens: memory.tokens,
                    memory_key: copy(&memory.memory_key)?,
                    memory_value: copy(&memory.memory_value)?,
                    memory_mask: values.to_vec(),
                })
            })
            .transpose()
    }
    pub fn quarantine_cause(&self) -> Option<&BackendError> {
        self.quarantine.as_ref()
    }
    pub fn clear_public_memory(&mut self) -> Result<(), BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        // Exclusive &mut access and the synchronous Run boundary prove there
        // is no concurrent physical invocation in this direct owner API.
        self.memory = None;
        #[cfg(feature = "experimental-io-binding")]
        {
            self.device_role = None;
            self.device_memory = None;
        }
        self.stats.new_game_resets = self.stats.new_game_resets.saturating_add(1);
        Ok(())
    }
    pub fn snapshot_stats(&self) -> Result<PalsBackendStats, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        if self.active.is_some() {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "PALS stats require the physical invocation boundary",
            ));
        }
        let mut stats = self.stats.clone();
        stats.live_public_cache_entries = u64::from(self.memory.is_some());
        #[cfg(feature = "experimental-io-binding")]
        if self.device_memory.is_some() {
            stats.live_public_cache_entries += 1;
        }
        stats.validate()?;
        Ok(stats)
    }
    pub fn config(&self) -> PalsOnnxConfig {
        self.config
    }
    pub fn verify_runtime(&self) -> Result<(), BackendError> {
        if matches!(self.config.provider, Provider::Cuda { .. }) {
            self.cuda_mapping_audit
                .borrow_mut()
                .final_audit(|| self.runtime.verify_cuda_runtime_mappings())?;
        }
        Ok(())
    }
    pub fn worker(
        mut self,
    ) -> Result<SingleWorker<PalsModelInput, Result<PalsRawOutput, BackendError>>, BackendError>
    {
        SingleWorker::spawn_with_outcome(move |input| {
            let result = self.run(input);
            match self.quarantine.as_ref() {
                Some(cause) => PhysicalRun::Quarantined(cause.clone()),
                None => PhysicalRun::Complete(result),
            }
        })
    }
    pub fn controlled_worker(
        mut self,
    ) -> Result<SingleWorker<PalsNativeCommand, Result<PalsNativeResult, BackendError>>, BackendError>
    {
        SingleWorker::spawn_with_outcome(move |command| {
            let result = match command {
                PalsNativeCommand::Evaluate(input) => {
                    self.run(input).map(PalsNativeResult::Evaluation)
                }
                PalsNativeCommand::NewGame => self
                    .clear_public_memory()
                    .map(|()| PalsNativeResult::NewGame),
                PalsNativeCommand::SnapshotStats => {
                    self.snapshot_stats().map(PalsNativeResult::Stats)
                }
                PalsNativeCommand::VerifyRuntime => self
                    .verify_runtime()
                    .map(|()| PalsNativeResult::RuntimeVerified),
            };
            match self.quarantine.as_ref() {
                Some(cause) => PhysicalRun::Quarantined(cause.clone()),
                None => PhysicalRun::Complete(result),
            }
        })
    }
    pub fn run(&mut self, input: &PalsModelInput) -> Result<PalsRawOutput, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        self.cuda_mapping_audit.borrow().allow_run()?;
        input.validate(&self.model_config).map_err(model_input)?;
        if input.model_epoch != self.epoch {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "PALS request weight epoch differs from frozen session",
            ));
        }
        if input.role == PalsRole::Validator {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "validator inference is absent from P/C product export",
            ));
        }
        let prepared = input
            .prepare_tensors(&self.model_config)
            .map_err(model_input)?;
        let key = prepared.public_memory_key;
        self.active = Some(ActiveInputs::new(prepared, input.role == PalsRole::Critic)?);
        self.stats.admitted_role_requests = self.stats.admitted_role_requests.saturating_add(1);
        let result = {
            #[cfg(feature = "experimental-io-binding")]
            if self.config.device_public_memory {
                self.run_device(input, key)
            } else {
                self.run_active(input, key)
            }
            #[cfg(not(feature = "experimental-io-binding"))]
            self.run_active(input, key)
        };
        if self.quarantine.is_none() {
            self.active = None;
        }
        if matches!(self.config.provider, Provider::Cuda { .. }) {
            // `run_active`'s synchronous Run or `run_device`'s output sync has
            // physically completed. ORT provider images can now be required;
            // this one-time audit must never preempt session creation, and is
            // not a per-node mapping scan. A failed origin audit is latched but
            // does not mislabel this already-fenced execution as unknown.
            self.cuda_mapping_audit
                .borrow_mut()
                .after_run(result.is_ok(), || {
                    self.runtime.verify_cuda_runtime_mappings()
                })?;
        }
        result
    }
    #[cfg(feature = "experimental-io-binding")]
    fn run_device(
        &mut self,
        input: &PalsModelInput,
        key: [u8; 32],
    ) -> Result<PalsRawOutput, BackendError> {
        let device_id = match self.config.provider {
            Provider::Cuda { device_id, .. } => device_id,
            _ => unreachable!("admission validates explicit CUDA"),
        };
        let cached = self.config.cache_public_memory
            && self.device_memory.as_ref().is_some_and(|m| m.key == key);
        if cached {
            self.public_cache_hits = self.public_cache_hits.saturating_add(1);
            self.stats.public_cache_hits = self.stats.public_cache_hits.saturating_add(1);
        } else {
            self.stats.public_cache_misses = self.stats.public_cache_misses.saturating_add(1);
            let tokens = self
                .model_config
                .public_memory_tokens(input.records.len())
                .map_err(model_input)?;
            // Creation performs allocations/bind-output ownership only. Install
            // it before input binding can enqueue a native transfer.
            self.device_memory = Some(
                device::DeviceMemory::new(
                    self.public.as_ref().expect("public session"),
                    key,
                    tokens,
                    device_id,
                )
                .map_err(|e| {
                    native(
                        CauseCode::TensorCreate,
                        "cannot allocate bounded PALS device memory",
                        e,
                    )
                })?,
            );
            let result = self
                .device_memory
                .as_mut()
                .expect("device memory installed")
                .run(
                    self.public.as_mut().expect("public session"),
                    self.active.as_ref().expect("physical inputs pinned"),
                    &mut self.stats,
                );
            if let Err(error) = result {
                let failure = native(
                    CauseCode::OrtRun,
                    "PALS public device binding/copy/fence failed",
                    error,
                );
                self.quarantine = Some(failure.clone());
                return Err(failure);
            }
            if let Err(error) = self
                .device_memory
                .as_ref()
                .expect("device memory completed")
                .validate_mask(input.records.len())
            {
                // Successful Run/fence permits discarding an invalid cache,
                // never admitting its same key as a later successful hit.
                self.device_memory = None;
                return Err(error);
            }
            self.public_encodes = self.public_encodes.saturating_add(1);
        }
        let critic = input.role == PalsRole::Critic;
        let shared = self.layout == Layout::SharedPcIf;
        let session = if shared {
            self.shared_pc.as_ref().expect("shared P/C session")
        } else if critic {
            self.critic.as_ref().expect("critic session")
        } else {
            self.proposer.as_ref().expect("proposer session")
        };
        self.device_role = Some(
            device::DeviceRole::new(
                session,
                input.candidates.len().max(1),
                input.divergence_features.len().max(1),
                critic,
                shared,
            )
            .map_err(|e| {
                native(
                    CauseCode::TensorCreate,
                    "cannot allocate bounded PALS private outputs",
                    e,
                )
            })?,
        );
        let session = if shared {
            self.shared_pc.as_mut().expect("shared P/C session")
        } else if critic {
            self.critic.as_mut().expect("critic session")
        } else {
            self.proposer.as_mut().expect("proposer session")
        };
        let result = self
            .device_role
            .as_mut()
            .expect("private outputs installed")
            .run(
                session,
                self.active.as_ref().expect("physical inputs pinned"),
                self.device_memory.as_ref().expect("public memory pinned"),
                &mut self.stats,
            );
        if let Err(error) = result {
            let failure = native(
                CauseCode::OrtRun,
                "PALS private device binding/copy/fence failed",
                error,
            );
            self.quarantine = Some(failure.clone());
            return Err(failure);
        }
        let output = self
            .device_role
            .as_ref()
            .expect("private outputs completed")
            .extract(input)?;
        output
            .decode(input, &self.model_config)
            .map_err(model_input)?;
        self.stats.validated_role_outputs = self.stats.validated_role_outputs.saturating_add(1);
        // No private role latent is fed into another invocation. After the
        // physical fence the private binding's aliases can be released.
        self.device_role = None;
        Ok(output)
    }
    fn run_active(
        &mut self,
        input: &PalsModelInput,
        key: [u8; 32],
    ) -> Result<PalsRawOutput, BackendError> {
        let cached =
            self.config.cache_public_memory && self.memory.as_ref().is_some_and(|m| m.key == key);
        if cached {
            self.public_cache_hits = self.public_cache_hits.saturating_add(1);
            self.stats.public_cache_hits = self.stats.public_cache_hits.saturating_add(1);
        } else {
            self.stats.public_cache_misses = self.stats.public_cache_misses.saturating_add(1);
            let active = self.active.as_ref().expect("owned input installed");
            self.stats.public_nn_runs_attempted =
                self.stats.public_nn_runs_attempted.saturating_add(1);
            let result = self.public.as_mut().expect("loaded public session").run(ort::inputs!["board" => &active.board, "metadata" => &active.metadata, "records" => &active.records, "record_mask" => &active.record_mask]);
            let outputs = match result {
                Ok(outputs) => outputs,
                Err(error) => {
                    let failure = native(
                        CauseCode::OrtRun,
                        "PALS public-memory native Run failed",
                        error,
                    );
                    if matches!(self.config.provider, Provider::Cuda { .. }) {
                        self.quarantine = Some(failure.clone());
                    } else {
                        self.stats.public_nn_runs_failed_known =
                            self.stats.public_nn_runs_failed_known.saturating_add(1);
                    }
                    return Err(failure);
                }
            };
            self.stats.public_nn_runs_completed =
                self.stats.public_nn_runs_completed.saturating_add(1);
            self.stats.completed_nn_inputs = self.stats.completed_nn_inputs.saturating_add(1);
            let tokens = self
                .model_config
                .public_memory_tokens(input.records.len())
                .map_err(model_input)?;
            let extract = |name: &str| -> Result<Vec<f32>, BackendError> {
                let (shape, values) = outputs[name].try_extract_tensor::<f32>().map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "PALS public-memory extraction failed",
                        e,
                    )
                })?;
                if shape.as_ref() != [1, 2, tokens as i64, 64]
                    || values.iter().any(|v| !v.is_finite())
                {
                    return Err(fail(
                        K::NumericalFailure,
                        S::Output,
                        "PALS public-memory output shape/finite check failed",
                    ));
                }
                Ok(values.to_vec())
            };
            let memory_key = Tensor::from_array(([1, 2, tokens, 64], extract("memory_key")?))
                .map_err(|e| {
                    native(
                        CauseCode::TensorCreate,
                        "PALS public-memory ownership failed",
                        e,
                    )
                })?;
            let memory_value = Tensor::from_array(([1, 2, tokens, 64], extract("memory_value")?))
                .map_err(|e| {
                native(
                    CauseCode::TensorCreate,
                    "PALS public-memory ownership failed",
                    e,
                )
            })?;
            let (shape, values) = outputs["memory_mask"]
                .try_extract_tensor::<bool>()
                .map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "PALS public-memory mask extraction failed",
                        e,
                    )
                })?;
            if shape.as_ref() != [1, tokens as i64]
                || values[..66].iter().any(|v| !*v)
                || values[66..].iter().filter(|v| **v).count() != input.records.len()
            {
                return Err(fail(
                    K::NumericalFailure,
                    S::Output,
                    "PALS public-memory mask differs from admitted records",
                ));
            }
            let mask = Tensor::from_array(([1, tokens], values.to_vec())).map_err(|e| {
                native(
                    CauseCode::TensorCreate,
                    "PALS public-memory mask ownership failed",
                    e,
                )
            })?;
            drop(outputs);
            self.memory = Some(PublicMemory {
                key,
                tokens,
                memory_key,
                memory_value,
                mask,
            });
            self.public_encodes = self.public_encodes.saturating_add(1);
            self.stats.validated_public_outputs =
                self.stats.validated_public_outputs.saturating_add(1);
        }
        let active = self.active.as_ref().expect("owned inputs remain installed");
        let memory = self
            .memory
            .as_ref()
            .expect("public memory installed after successful fence");
        debug_assert_eq!(
            memory.tokens,
            self.model_config
                .public_memory_tokens(input.records.len())
                .expect("validated")
        );
        let critic = input.role == PalsRole::Critic;
        let shared = self.layout == Layout::SharedPcIf;
        self.stats.role_nn_runs_attempted = self.stats.role_nn_runs_attempted.saturating_add(1);
        let result = if shared {
            self.shared_pc.as_mut().expect("loaded shared P/C session").run(ort::inputs!["role_is_critic" => &active.role_is_critic, "memory_key" => &memory.memory_key, "memory_value" => &memory.memory_value, "memory_mask" => &memory.mask, "candidates" => &active.candidates, "candidate_mask" => &active.candidate_mask, "query" => &active.query, "divergence_features" => &active.divergences, "divergence_mask" => &active.divergence_mask])
        } else if critic {
            self.critic.as_mut().expect("loaded critic session").run(ort::inputs!["memory_key" => &memory.memory_key, "memory_value" => &memory.memory_value, "memory_mask" => &memory.mask, "candidates" => &active.candidates, "candidate_mask" => &active.candidate_mask, "query" => &active.query, "divergence_features" => &active.divergences, "divergence_mask" => &active.divergence_mask])
        } else {
            self.proposer.as_mut().expect("loaded proposer session").run(ort::inputs!["memory_key" => &memory.memory_key, "memory_value" => &memory.memory_value, "memory_mask" => &memory.mask, "candidates" => &active.candidates, "candidate_mask" => &active.candidate_mask, "query" => &active.query])
        };
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                let failure = native(
                    CauseCode::OrtRun,
                    "PALS private-role native Run failed",
                    error,
                );
                if matches!(self.config.provider, Provider::Cuda { .. }) {
                    self.quarantine = Some(failure.clone());
                } else {
                    self.stats.role_nn_runs_failed_known =
                        self.stats.role_nn_runs_failed_known.saturating_add(1);
                }
                return Err(failure);
            }
        };
        self.stats.role_nn_runs_completed = self.stats.role_nn_runs_completed.saturating_add(1);
        self.stats.completed_nn_inputs = self.stats.completed_nn_inputs.saturating_add(1);
        let extract = |name: &str, expected: &[i64]| -> Result<Vec<f32>, BackendError> {
            let (shape, values) = outputs[name].try_extract_tensor::<f32>().map_err(|e| {
                native(
                    CauseCode::PolicyExtract,
                    "PALS role output extraction failed",
                    e,
                )
            })?;
            if shape.as_ref() != expected || values.iter().any(|v| !v.is_finite()) {
                return Err(fail(
                    K::NumericalFailure,
                    S::Output,
                    "PALS role output shape/finite check failed",
                ));
            }
            Ok(values.to_vec())
        };
        let mut candidate_logits = extract(
            "candidate_logits",
            &[1, input.candidates.len().max(1) as i64],
        )?;
        candidate_logits.truncate(input.candidates.len());
        let wdl_logits = extract("wdl_logits", &[1, 3])?
            .try_into()
            .map_err(|_| fail(K::NumericalFailure, S::Output, "PALS WDL shape failed"))?;
        let private_latent = extract("private_latent", &[1, 16, 384])?;
        if shared {
            let (shape, tag) = outputs["is_critic"]
                .try_extract_tensor::<bool>()
                .map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "PALS role tag extraction failed",
                        e,
                    )
                })?;
            validate_role_tag(shape.as_ref(), tag, critic)?;
        }
        let divergence_logits = if critic {
            let mut values = extract(
                "divergence_logits",
                &[1, input.divergence_features.len().max(1) as i64],
            )?;
            values.truncate(input.divergence_features.len());
            Some(values)
        } else {
            if shared {
                let values = extract(
                    "divergence_logits",
                    &[1, input.divergence_features.len().max(1) as i64],
                )?;
                if values.iter().any(|value| *value != 0.) {
                    return Err(fail(
                        K::NumericalFailure,
                        S::Output,
                        "shared proposer emitted nonzero critic output",
                    ));
                }
            }
            None
        };
        let output = PalsRawOutput {
            candidate_logits,
            wdl_logits,
            private_latent,
            divergence_logits,
            task_logits: None,
        };
        output
            .decode(input, &self.model_config)
            .map_err(model_input)?;
        self.stats.validated_role_outputs = self.stats.validated_role_outputs.saturating_add(1);
        Ok(output)
    }
}

impl Drop for PalsOnnxBackend {
    fn drop(&mut self) {
        if self.quarantine.is_some() {
            // The worker normally retains the entire closure on quarantine. A
            // direct caller may drop its owner: retain the same resources rather
            // than using Drop as an unproven CUDA completion acknowledgement.
            for session in [
                self.public.take(),
                self.proposer.take(),
                self.critic.take(),
                self.shared_pc.take(),
            ]
            .into_iter()
            .flatten()
            {
                std::mem::forget(session);
            }
            if let Some(active) = self.active.take() {
                std::mem::forget(active);
            }
            if let Some(memory) = self.memory.take() {
                std::mem::forget(memory);
            }
            #[cfg(feature = "experimental-io-binding")]
            {
                if let Some(role) = self.device_role.take() {
                    std::mem::forget(role);
                }
                if let Some(memory) = self.device_memory.take() {
                    std::mem::forget(memory);
                }
            }
        } else {
            // Allocator-backed values and bindings are destroyed before their
            // sessions on the physically complete normal shutdown path.
            #[cfg(feature = "experimental-io-binding")]
            {
                self.device_role.take();
                self.device_memory.take();
            }
            self.memory.take();
            self.active.take();
        }
    }
}

#[cfg(feature = "experimental-io-binding")]
mod device {
    use super::*;
    use ort::io_binding::IoBinding;
    use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
    use rz_native_loader::ort_binding::run_fixed_binding;
    fn alias<T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug>(
        tensor: &Tensor<T>,
    ) -> ort::Result<Tensor<T>> {
        // Upgrading an owned view retains the same OrtValue, unlike Clone's
        // device copy. The alias remains inside this exclusive physical owner.
        tensor
            .view()
            .try_upgrade()
            .map_err(|_| ort::Error::new("PALS owned tensor cannot upgrade its view"))
    }
    pub(super) struct DeviceMemory {
        binding: IoBinding,
        pub key: [u8; 32],
        tokens: usize,
        memory_key: Tensor<f32>,
        memory_value: Tensor<f32>,
        mask: Tensor<bool>,
        // rc.10 allocated tensors retain callbacks rather than this Allocator;
        // keep it last, after all bindings and dependent tensors.
        _allocator: Allocator,
    }
    impl DeviceMemory {
        pub fn new(
            session: &Session,
            key: [u8; 32],
            tokens: usize,
            device: i32,
        ) -> ort::Result<Self> {
            let allocator = Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    device,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
            )?;
            let binding = session.create_binding()?;
            // Stage into options already owned by the aggregate, so a later
            // allocation failure cannot drop its allocator before values.
            let mut owned = DeviceAllocation {
                binding: Some(binding),
                key: None,
                value: None,
                allocator,
            };
            owned.key = Some(Tensor::new(&owned.allocator, [1, 2, tokens, 64])?);
            owned.value = Some(Tensor::new(&owned.allocator, [1, 2, tokens, 64])?);
            let mask = Tensor::new(&Allocator::default(), [1, tokens])?;
            let binding = owned
                .binding
                .as_mut()
                .expect("binding owned before tensors");
            binding.bind_output(
                "memory_key",
                alias(owned.key.as_ref().expect("key allocated"))?,
            )?;
            binding.bind_output(
                "memory_value",
                alias(owned.value.as_ref().expect("value allocated"))?,
            )?;
            binding.bind_output("memory_mask", alias(&mask)?)?;
            Ok(Self {
                binding: owned.binding.take().expect("binding installed"),
                key,
                tokens,
                memory_key: owned.key.take().expect("key allocated"),
                memory_value: owned.value.take().expect("value allocated"),
                mask,
                _allocator: owned.allocator,
            })
        }
        pub fn run(
            &mut self,
            session: &mut Session,
            active: &ActiveInputs,
            stats: &mut PalsBackendStats,
        ) -> ort::Result<()> {
            self.binding.bind_input("board", &active.board)?;
            self.binding.bind_input("metadata", &active.metadata)?;
            self.binding.bind_input("records", &active.records)?;
            self.binding
                .bind_input("record_mask", &active.record_mask)?;
            self.binding.synchronize_inputs()?;
            stats.public_nn_runs_attempted = stats.public_nn_runs_attempted.saturating_add(1);
            run_fixed_binding(session, &self.binding)?;
            self.binding.synchronize_outputs()?;
            stats.public_nn_runs_completed = stats.public_nn_runs_completed.saturating_add(1);
            stats.completed_nn_inputs = stats.completed_nn_inputs.saturating_add(1);
            self.binding.clear_inputs();
            Ok(())
        }
        pub fn validate_mask(&self, records: usize) -> Result<(), BackendError> {
            let (shape, mask) = self.mask.try_extract_tensor::<bool>().map_err(|e| {
                native(
                    CauseCode::PolicyExtract,
                    "cannot extract completed PALS mask",
                    e,
                )
            })?;
            if shape.as_ref() != [1, self.tokens as i64]
                || mask[..66].iter().any(|v| !*v)
                || mask[66..].iter().filter(|v| **v).count() != records
            {
                return Err(fail(
                    K::NumericalFailure,
                    S::Output,
                    "bound PALS memory mask differs",
                ));
            }
            Ok(())
        }
    }
    struct DeviceAllocation {
        binding: Option<IoBinding>,
        key: Option<Tensor<f32>>,
        value: Option<Tensor<f32>>,
        allocator: Allocator,
    }
    pub(super) struct DeviceRole {
        binding: IoBinding,
        candidate: Tensor<f32>,
        wdl: Tensor<f32>,
        latent: Tensor<f32>,
        divergence: Option<Tensor<f32>>,
        is_critic: Option<Tensor<bool>>,
    }
    impl DeviceRole {
        pub fn new(
            session: &Session,
            candidates: usize,
            divergences: usize,
            critic: bool,
            shared: bool,
        ) -> ort::Result<Self> {
            let mut binding = session.create_binding()?;
            let cpu = Allocator::default();
            let candidate = Tensor::new(&cpu, [1, candidates])?;
            let wdl = Tensor::new(&cpu, [1_usize, 3])?;
            let latent = Tensor::new(&cpu, [1_usize, 16, 384])?;
            let divergence = (critic || shared)
                .then(|| Tensor::new(&cpu, [1, divergences]))
                .transpose()?;
            let is_critic = shared
                .then(|| Tensor::new(&cpu, Vec::<usize>::new()))
                .transpose()?;
            binding.bind_output("candidate_logits", alias(&candidate)?)?;
            binding.bind_output("wdl_logits", alias(&wdl)?)?;
            binding.bind_output("private_latent", alias(&latent)?)?;
            if let Some(value) = &divergence {
                binding.bind_output("divergence_logits", alias(value)?)?;
            }
            if let Some(value) = &is_critic {
                binding.bind_output("is_critic", alias(value)?)?;
            }
            Ok(Self {
                binding,
                candidate,
                wdl,
                latent,
                divergence,
                is_critic,
            })
        }
        pub fn run(
            &mut self,
            session: &mut Session,
            active: &ActiveInputs,
            memory: &DeviceMemory,
            stats: &mut PalsBackendStats,
        ) -> ort::Result<()> {
            if self.is_critic.is_some() {
                // CUDA If's condition has OrtMemTypeCPUInput. This is an
                // owned CPU bool scalar, never an uploaded device condition.
                self.binding
                    .bind_input("role_is_critic", &active.role_is_critic)?;
            }
            self.binding.bind_input("memory_key", &memory.memory_key)?;
            self.binding
                .bind_input("memory_value", &memory.memory_value)?;
            self.binding.bind_input("memory_mask", &memory.mask)?;
            self.binding.bind_input("candidates", &active.candidates)?;
            self.binding
                .bind_input("candidate_mask", &active.candidate_mask)?;
            self.binding.bind_input("query", &active.query)?;
            if self.divergence.is_some() {
                self.binding
                    .bind_input("divergence_features", &active.divergences)?;
                self.binding
                    .bind_input("divergence_mask", &active.divergence_mask)?;
            }
            self.binding.synchronize_inputs()?;
            stats.role_nn_runs_attempted = stats.role_nn_runs_attempted.saturating_add(1);
            run_fixed_binding(session, &self.binding)?;
            self.binding.synchronize_outputs()?;
            stats.role_nn_runs_completed = stats.role_nn_runs_completed.saturating_add(1);
            stats.completed_nn_inputs = stats.completed_nn_inputs.saturating_add(1);
            self.binding.clear_inputs();
            Ok(())
        }
        pub fn extract(&self, input: &PalsModelInput) -> Result<PalsRawOutput, BackendError> {
            let extract =
                |tensor: &Tensor<f32>, expected: &[i64]| -> Result<Vec<f32>, BackendError> {
                    let (shape, values) = tensor.try_extract_tensor::<f32>().map_err(|e| {
                        native(
                            CauseCode::PolicyExtract,
                            "PALS bound private output extraction failed",
                            e,
                        )
                    })?;
                    if shape.as_ref() != expected || values.iter().any(|v| !v.is_finite()) {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "bound PALS private output differs",
                        ));
                    }
                    Ok(values.to_vec())
                };
            let mut candidate_logits =
                extract(&self.candidate, &[1, input.candidates.len().max(1) as i64])?;
            candidate_logits.truncate(input.candidates.len());
            let mut divergence_logits = self
                .divergence
                .as_ref()
                .map(|v| {
                    let mut result =
                        extract(v, &[1, input.divergence_features.len().max(1) as i64])?;
                    result.truncate(input.divergence_features.len());
                    Ok(result)
                })
                .transpose()?;
            if let Some(tag) = &self.is_critic {
                let (shape, values) = tag.try_extract_tensor::<bool>().map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "bound shared role tag extraction failed",
                        e,
                    )
                })?;
                validate_role_tag(shape.as_ref(), values, input.role == PalsRole::Critic)?;
                if input.role == PalsRole::Proposer {
                    if divergence_logits
                        .as_ref()
                        .expect("shared divergence output")
                        .iter()
                        .any(|value| *value != 0.)
                    {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "bound shared proposer emitted nonzero critic output",
                        ));
                    }
                    // Validate the padded raw slot too, before semantic trim.
                    let padded = extract(
                        self.divergence.as_ref().expect("shared divergence output"),
                        &[1, input.divergence_features.len().max(1) as i64],
                    )?;
                    if padded.iter().any(|value| *value != 0.) {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "bound shared proposer padded critic output is nonzero",
                        ));
                    }
                    divergence_logits = None;
                }
            }
            Ok(PalsRawOutput {
                candidate_logits,
                wdl_logits: extract(&self.wdl, &[1, 3])?.try_into().expect("fixed WDL"),
                private_latent: extract(&self.latent, &[1, 16, 384])?,
                divergence_logits,
                task_logits: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_model::{PalsCandidateToken, PalsRecordToken};
    fn input() -> PalsModelInput {
        PalsModelInput {
            role: PalsRole::Proposer,
            board: vec![0; 64],
            metadata: [0.; 16],
            records: vec![PalsRecordToken {
                record_id: 1,
                revision: 2,
                critical: true,
                features: [0.; 16],
            }],
            required_critical_records: vec![1],
            candidates: vec![PalsCandidateToken {
                from: 12,
                to: 28,
                promotion: 0,
            }],
            divergence_features: vec![],
            query: [0.; 16],
            situation_revision: 3,
            model_epoch: [4; 32],
            history_digest: [5; 32],
        }
    }
    #[test]
    fn public_cache_role_neutral_private_query_and_candidates_are_not_memory() {
        let a = input();
        let mut b = a.clone();
        b.role = PalsRole::Critic;
        b.divergence_features.push([1.; 8]);
        b.query[0] = 9.;
        b.candidates.clear();
        let key = |input: &PalsModelInput| {
            input
                .public_memory_key(&PalsModelConfig::baseline())
                .unwrap()
        };
        assert_eq!(key(&a), key(&b));
        b.records[0].revision += 1;
        assert_ne!(key(&a), key(&b));
        b = a.clone();
        b.model_epoch[0] += 1;
        assert_ne!(key(&a), key(&b));
        b = a.clone();
        b.metadata[0] = -0.;
        assert_ne!(key(&a), key(&b));
    }
    #[test]
    fn descriptor_rejects_fixed_shapes_and_wrong_dtype() {
        let expected = [spec("board", TensorElementType::Int64, &[-1, 64])];
        let mut fields = vec![TensorManifest {
            name: "board".into(),
            dtype: "INT64".into(),
            shape: vec!["batch".into(), 64.into()],
        }];
        validate_declared_interface(&fields, &expected).unwrap();
        fields[0].shape[0] = 1.into();
        assert!(validate_declared_interface(&fields, &expected).is_err());
        fields[0].shape[0] = "batch".into();
        fields[0].dtype = "FLOAT".into();
        assert!(validate_declared_interface(&fields, &expected).is_err());
    }

    fn manifest() -> ExportManifest {
        let tensor = |v: Interface| {
            serde_json::json!({
                "name":v.name,
                "dtype":match v.dtype { TensorElementType::Float32 => "FLOAT", TensorElementType::Int64 => "INT64", TensorElementType::Bool => "BOOL", _ => unreachable!() },
                "shape":v.shape.iter().enumerate().map(|(axis, dim)| if *dim == -1 { serde_json::json!(format!("dynamic_{axis}")) } else { serde_json::json!(dim) }).collect::<Vec<_>>()
            })
        };
        let graphs = ["public", "proposer", "critic"].map(|role| serde_json::json!({
            "role":role,"file":if role == "public" { "public_memory.onnx".to_owned() } else {format!("role_{role}.onnx")},
            "sha256":"01".repeat(32),"opset":17,
            "inputs":if role == "public" {public_inputs()} else {role_inputs(role=="critic")}.into_iter().map(tensor).collect::<Vec<_>>(),
            "outputs":if role == "public" {memory_outputs()} else {role_outputs(role=="critic")}.into_iter().map(tensor).collect::<Vec<_>>()
        }));
        serde_json::from_value(serde_json::json!({
            "schema":PALS_MODEL_SCHEMA,"config":PalsModelConfig::baseline(),"checkpoint_sha256":"02".repeat(32),
            "trained":false,"training_steps":0,"roles":["proposer","critic"],"validator_present":false,"graphs":graphs,
            "runtime":{"onnxruntime":"1.22.0","rust_ort":"2.0.0-rc.10","compatibility":"requires_actual_numeric_check"},
            "candidate_promotion":{"0":"none","1":"queen","2":"rook","3":"bishop","4":"knight"},
            "wdl_perspective":"input_side_to_move","task_names":V_TASK_NAMES,"numeric_status":"not_run","cuda_status":"not_run"
        })).unwrap()
    }
    #[test]
    fn product_manifest_rejects_v_duplicate_roles_and_path_escape() {
        validate_manifest(&manifest()).unwrap();
        let mut m = manifest();
        m.validator_present = true;
        assert!(validate_manifest(&m).is_err());
        m = manifest();
        m.roles.push("validator".into());
        assert!(validate_manifest(&m).is_err());
        m = manifest();
        m.graphs[1].role = "critic".into();
        assert!(validate_manifest(&m).is_err());
        m = manifest();
        m.graphs[0].file = "../public_memory.onnx".into();
        assert!(validate_manifest(&m).is_err());
        m = manifest();
        m.task_names.swap(0, 1);
        assert!(validate_manifest(&m).is_err());
    }
    #[test]
    fn execution_capability_never_changes_cuda_request_to_cpu() {
        PalsOnnxConfig::cpu().validate().unwrap();
        let mut config = PalsOnnxConfig::cpu();
        config.device_public_memory = true;
        assert!(config.validate().is_err());
        config.device_public_memory = false;
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 0,
        };
        assert!(config.validate().is_err());
        let mut m = manifest();
        m.trained = true;
        assert!(validate_manifest(&m).is_err());
    }
    #[test]
    fn shared_layout_requires_semantic_pin_audit_scalar_and_role_fence() {
        let mut m = manifest();
        m.schema = "rovezero.pals-model.v2".into();
        m.model_semantics = Some(PALS_MODEL_SCHEMA.into());
        m.layout = Some("shared_pc_if".into());
        m.layout_revision = Some(1);
        m.role_batching = Some("one_scalar_role_per_physical_batch".into());
        m.rules_input_profile = Some("rz-pals-rules-fields-v1".into());
        m.rules_input_semantic_sha256 = Some("03".repeat(32));
        m.rules_encoder_source_sha256 = Some("06".repeat(32));
        m.graphs.truncate(1);
        let tensors = |values: Vec<Interface>| {
            values
                .into_iter()
                .map(|v| TensorManifest {
                    name: v.name.into(),
                    dtype: match v.dtype {
                        TensorElementType::Float32 => "FLOAT",
                        TensorElementType::Int64 => "INT64",
                        TensorElementType::Bool => "BOOL",
                        _ => unreachable!(),
                    }
                    .into(),
                    shape: v
                        .shape
                        .iter()
                        .enumerate()
                        .map(|(axis, dim)| {
                            if *dim == -1 {
                                serde_json::json!(format!("dynamic_{axis}"))
                            } else {
                                serde_json::json!(dim)
                            }
                        })
                        .collect(),
                })
                .collect()
        };
        m.graphs.push(GraphManifest {
            file: "shared_pc_if.onnx".into(),
            sha256: "04".repeat(32),
            role: "shared_pc".into(),
            inputs: tensors(shared_inputs()),
            outputs: tensors(shared_outputs()),
            opset: 17,
        });
        m.reader_initializer_bank = Some(ReaderInitializerBank {
            scope: "onnx_serialized_initializers_only".into(),
            layout_revision: 1,
            shared_parameters: vec![SharedParameter {
                parameter: "reader.weight".into(),
                initializer: "reader_weight".into(),
                bytes: 4,
                sha256: "05".repeat(32),
                source_parameter_sha256: "05".repeat(32),
                shape: vec![1],
            }],
            unique_shared_parameters: 1,
            shared_weight_bytes: 4,
            if_routes: 6,
            branch_local_initializers: 0,
            branch_operations: (0..6)
                .flat_map(|route| {
                    ["then_branch", "else_branch"].map(|branch| BranchOperations {
                        route: format!("route_{route}"),
                        branch: branch.into(),
                        nodes: 1,
                    })
                })
                .collect(),
            inactive_private_execution: "requires_runtime_profile".into(),
            ort_prepack_copies: "unknown".into(),
            device_residency_sharing: "unknown".into(),
        });
        validate_manifest(&m).unwrap();
        m.rules_encoder_source_sha256 = None;
        assert!(validate_manifest(&m).is_err());
        m.rules_encoder_source_sha256 = Some("06".repeat(32));
        m.rules_input_semantic_sha256 = None;
        assert!(validate_manifest(&m).is_err());
        m.rules_input_semantic_sha256 = Some("03".repeat(32));
        m.graphs[1].inputs[0].shape.push(1.into());
        assert!(validate_manifest(&m).is_err());
        m.graphs[1].inputs[0].shape.clear();
        m.reader_initializer_bank
            .as_mut()
            .unwrap()
            .shared_weight_bytes = 8;
        assert!(validate_manifest(&m).is_err());
        validate_role_tag(&[], &[true], true).unwrap();
        assert!(validate_role_tag(&[1], &[true], true).is_err());
        assert!(validate_role_tag(&[], &[false], true).is_err());
    }
    #[test]
    fn native_stats_separate_requests_graphs_inputs_and_validated_outputs() {
        let mut stats = PalsBackendStats {
            admitted_role_requests: 2,
            public_cache_hits: 1,
            public_cache_misses: 1,
            public_nn_runs_attempted: 1,
            public_nn_runs_completed: 1,
            role_nn_runs_attempted: 2,
            role_nn_runs_completed: 2,
            completed_nn_inputs: 3,
            validated_public_outputs: 1,
            validated_role_outputs: 2,
            new_game_resets: 2,
            live_public_cache_entries: 1,
            ..Default::default()
        };
        stats.validate().unwrap();
        stats.completed_nn_inputs = 2;
        assert!(stats.validate().is_err());
        stats.completed_nn_inputs = 3;
        stats.validated_role_outputs = 3;
        assert!(stats.validate().is_err());
        stats.validated_role_outputs = 2;
        stats.live_public_cache_entries = 2;
        assert!(stats.validate().is_err());
    }
    #[test]
    fn cuda_full_audit_waits_for_completed_run_and_latches_origin_failure() {
        let mut phase = CudaMappingAudit::default();
        phase.allow_run().unwrap();
        phase
            .after_run(false, || {
                panic!("provider images cannot be required before a completed Run")
            })
            .unwrap();
        assert!(matches!(phase, CudaMappingAudit::AwaitingFirstRun));
        let calls = std::cell::Cell::new(0);
        phase
            .after_run(true, || {
                calls.set(calls.get() + 1);
                Ok(())
            })
            .unwrap();
        phase
            .after_run(true, || panic!("full origin audit is not a per-node scan"))
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert!(matches!(phase, CudaMappingAudit::Confirmed));
        let mut rejected = CudaMappingAudit::default();
        let failure = rejected
            .after_run(true, || {
                Err(fail(
                    K::BackendUnavailable,
                    S::Backend,
                    "missing provider mapping",
                ))
            })
            .unwrap_err();
        assert_eq!(
            failure.detail,
            "PALS CUDA full mapping audit failed after completed native Run"
        );
        assert_eq!(rejected.allow_run().unwrap_err(), failure);
        assert_eq!(
            rejected
                .after_run(true, || panic!(
                    "failed origin audit cannot be retried into success"
                ))
                .unwrap_err(),
            failure
        );
        let mut final_rejected = CudaMappingAudit::default();
        assert!(final_rejected
            .final_audit(|| panic!(
                "final audit cannot require lazy provider images before first Run"
            ))
            .is_err());
        final_rejected.after_run(true, || Ok(())).unwrap();
        let failure = final_rejected
            .final_audit(|| {
                Err(fail(
                    K::BackendUnavailable,
                    S::Backend,
                    "late mapping origin changed",
                ))
            })
            .unwrap_err();
        assert_eq!(
            failure.detail,
            "PALS CUDA final full mapping audit failed after completed native Run"
        );
        assert_eq!(final_rejected.allow_run().unwrap_err(), failure);
        assert_eq!(
            final_rejected
                .after_run(true, || panic!(
                    "no new Run may be admitted after final origin failure"
                ))
                .unwrap_err(),
            failure
        );
        assert_eq!(
            final_rejected
                .final_audit(|| panic!("final origin failure cannot be retried into success"))
                .unwrap_err(),
            failure
        );
    }
}
