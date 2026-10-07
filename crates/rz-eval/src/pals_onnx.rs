//! Frozen, P/C-only PALS ONNX sessions. Chess authority remains with Rules.
//!
//! A synchronous successful Run is a physical completion fence. Failed CUDA
//! execution permanently quarantines this owner and its active native values.
//! Logical cancellation belongs to the existing runtime and never frees these
//! buffers. Public memory is exact and role-neutral; role latents start fresh.
use crate::asset::{self, parse_sha256};
use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use crate::onnx::{OrtRuntime, Provider};
use crate::pals_model::{PalsModelConfig, PalsModelInput, PalsRawOutput, PalsRole, PreparedPalsTensors, PALS_MODEL_SCHEMA};
use crate::worker::{PhysicalRun, SingleWorker};
use ort::execution_providers::{ArenaExtendStrategy, CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use serde::Deserialize;
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
}
impl PalsOnnxConfig {
    pub fn cpu() -> Self {
        Self { provider: Provider::Cpu, intra_threads: 2, cache_public_memory: true }
    }
    fn validate(&self) -> Result<(), BackendError> {
        if !(1..=2).contains(&self.intra_threads) {
            return Err(fail(K::InvalidInput, S::Admission, "PALS CPU thread declaration must be 1..=2"));
        }
        if let Provider::Cuda { device_id, arena_bytes } = self.provider {
            if device_id < 0 || arena_bytes == 0 || arena_bytes > 6 * 1024 * 1024 * 1024 {
                return Err(fail(K::InvalidInput, S::Admission, "PALS CUDA device/memory declaration is invalid"));
            }
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportManifest {
    schema: String,
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
    numeric_status: String,
    cuda_status: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifest { onnxruntime: String, rust_ort: String, compatibility: String }
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
struct TensorManifest { name: String, dtype: String, shape: Vec<serde_json::Value> }

#[derive(Clone, Copy)]
struct Interface { name: &'static str, dtype: TensorElementType, shape: &'static [i64] }
fn public_inputs() -> Vec<Interface> { vec![
    spec("board", TensorElementType::Int64, &[-1, 64]),
    spec("metadata", TensorElementType::Float32, &[-1, 16]),
    spec("records", TensorElementType::Float32, &[-1, -1, 16]),
    spec("record_mask", TensorElementType::Bool, &[-1, -1]),
] }
fn memory_outputs() -> Vec<Interface> { vec![
    spec("memory_key", TensorElementType::Float32, &[-1, 2, -1, 64]),
    spec("memory_value", TensorElementType::Float32, &[-1, 2, -1, 64]),
    spec("memory_mask", TensorElementType::Bool, &[-1, -1]),
] }
fn role_inputs(critic: bool) -> Vec<Interface> {
    let mut result = memory_outputs();
    result.extend([
        spec("candidates", TensorElementType::Int64, &[-1, -1, 3]),
        spec("candidate_mask", TensorElementType::Bool, &[-1, -1]),
        spec("query", TensorElementType::Float32, &[-1, 16]),
    ]);
    if critic { result.extend([
        spec("divergence_features", TensorElementType::Float32, &[-1, -1, 8]),
        spec("divergence_mask", TensorElementType::Bool, &[-1, -1]),
    ]); }
    result
}
fn role_outputs(critic: bool) -> Vec<Interface> {
    let mut result = vec![
        spec("candidate_logits", TensorElementType::Float32, &[-1, -1]),
        spec("wdl_logits", TensorElementType::Float32, &[-1, 3]),
        spec("private_latent", TensorElementType::Float32, &[-1, 16, 384]),
    ];
    if critic { result.push(spec("divergence_logits", TensorElementType::Float32, &[-1, -1])); }
    result
}
const fn spec(name: &'static str, dtype: TensorElementType, shape: &'static [i64]) -> Interface {
    Interface { name, dtype, shape }
}
fn fail(kind: K, stage: S, detail: &'static str) -> BackendError { BackendError::new(kind, stage, detail) }
fn native(code: CauseCode, detail: &'static str, error: ort::Error) -> BackendError {
    fail(K::BackendFailure, S::Backend, detail).with_ort_cause(code, error)
}
fn model_input(error: impl std::fmt::Display) -> BackendError {
    fail(K::InvalidInput, S::Admission, "PALS model request violates its bounded tensor contract")
        .with_external_cause(CauseCode::OutputValidation, &error)
}

fn validate_declared_interface(values: &[TensorManifest], expected: &[Interface]) -> Result<(), BackendError> {
    if values.len() != expected.len() { return Err(fail(K::UnsupportedModel, S::Asset, "PALS graph descriptor interface differs")); }
    for expected in expected {
        let matching: Vec<_> = values.iter().filter(|v| v.name == expected.name).collect();
        if matching.len() != 1 { return Err(fail(K::UnsupportedModel, S::Asset, "PALS graph descriptor names differ")); }
        let value = matching[0];
        let dtype = match expected.dtype { TensorElementType::Float32 => "FLOAT", TensorElementType::Int64 => "INT64", TensorElementType::Bool => "BOOL", _ => unreachable!() };
        if value.dtype != dtype || value.shape.len() != expected.shape.len() || !value.shape.iter().zip(expected.shape).all(|(actual, expected)| {
            if *expected == -1 { actual.as_str().is_some_and(|s| !s.is_empty() && s.len() <= 64) }
            else { actual.as_i64() == Some(*expected) }
        }) { return Err(fail(K::UnsupportedModel, S::Asset, "PALS descriptor tensor dtype/shape differs")); }
    }
    Ok(())
}
fn validate_manifest(manifest: &ExportManifest) -> Result<[u8; 32], BackendError> {
    manifest.config.validate().map_err(model_input)?;
    let promotion = ["none", "queen", "rook", "bishop", "knight"].iter().enumerate()
        .map(|(i, name)| (i.to_string(), name.to_string())).collect::<BTreeMap<_, _>>();
    let mut roles = manifest.roles.clone(); roles.sort();
    if manifest.schema != PALS_MODEL_SCHEMA || manifest.validator_present || roles != ["critic", "proposer"]
        || manifest.graphs.len() != 3 || manifest.wdl_perspective != "input_side_to_move"
        || manifest.candidate_promotion != promotion || manifest.runtime.onnxruntime != "1.22.0"
        || manifest.runtime.rust_ort != "2.0.0-rc.10" || manifest.runtime.compatibility != "requires_actual_numeric_check"
        || manifest.numeric_status != "not_run" || manifest.cuda_status != "not_run"
        || (manifest.trained && manifest.training_steps == 0) || (!manifest.trained && manifest.training_steps != 0)
    { return Err(fail(K::UnsupportedModel, S::Asset, "expected registered P/C-only FP32 PALS export; V is not a product role")); }
    for role in ["public", "proposer", "critic"] {
        let graphs: Vec<_> = manifest.graphs.iter().filter(|g| g.role == role).collect();
        if graphs.len() != 1 { return Err(fail(K::UnsupportedModel, S::Asset, "PALS public/P/C graph set differs")); }
        let graph = graphs[0];
        let expected_file = if role == "public" { "public_memory.onnx".to_owned() } else { format!("role_{role}.onnx") };
        if graph.file != expected_file || graph.opset != 17 || !matches!(Path::new(&graph.file).components().collect::<Vec<_>>().as_slice(), [Component::Normal(_)]) {
            return Err(fail(K::UnsupportedModel, S::Asset, "PALS graph file/opset is unsupported"));
        }
        parse_sha256(&graph.sha256)?;
        let critic = role == "critic";
        validate_declared_interface(&graph.inputs, &if role == "public" { public_inputs() } else { role_inputs(critic) })?;
        validate_declared_interface(&graph.outputs, &if role == "public" { memory_outputs() } else { role_outputs(critic) })?;
    }
    parse_sha256(&manifest.checkpoint_sha256)
}
fn validate_session(session: &Session, graph: &GraphManifest, epoch: &str, trained: bool, training_steps: u64) -> Result<(), BackendError> {
    let inputs = if graph.role == "public" { public_inputs() } else { role_inputs(graph.role == "critic") };
    let outputs = if graph.role == "public" { memory_outputs() } else { role_outputs(graph.role == "critic") };
    let valid = |name: &str, ty: &ValueType, expected: &[Interface]| expected.iter().any(|expected| {
        name == expected.name && matches!(ty, ValueType::Tensor { ty, shape, .. } if *ty == expected.dtype && shape.as_ref() == expected.shape)
    });
    if session.inputs.len() != inputs.len() || session.outputs.len() != outputs.len()
        || !session.inputs.iter().all(|v| valid(&v.name, &v.input_type, &inputs))
        || !session.outputs.iter().all(|v| valid(&v.name, &v.output_type, &outputs))
    { return Err(fail(K::UnsupportedModel, S::Asset, "loaded PALS ONNX tensor interface differs")); }
    let metadata = session.metadata().map_err(|e| native(CauseCode::ModelLoad, "cannot inspect PALS ONNX metadata", e))?;
    let steps = training_steps.to_string();
    for (key, value) in [("schema", PALS_MODEL_SCHEMA), ("role", graph.role.as_str()), ("checkpoint_sha256", epoch), ("trained", if trained { "true" } else { "false" }), ("training_steps", steps.as_str()), ("precision", "fp32"), ("expected_ort", "1.22.0"), ("public_kv", "shared_role_neutral")] {
        if metadata.custom(key).map_err(|e| native(CauseCode::ModelLoad, "cannot read PALS ONNX metadata", e))?.as_deref() != Some(value) {
            return Err(fail(K::IdentityMismatch, S::Asset, "PALS ONNX metadata disagrees with registered export"));
        }
    }
    Ok(())
}
fn load_session(bytes: Vec<u8>, config: PalsOnnxConfig) -> Result<Session, BackendError> {
    let configure = |e| native(CauseCode::SessionConfiguration, "PALS ORT session configuration failed", e);
    let mut builder = Session::builder().map_err(configure)?
        .with_no_environment_execution_providers().map_err(configure)?
        .with_intra_threads(config.intra_threads).map_err(configure)?
        .with_inter_threads(1).map_err(configure)?
        .with_parallel_execution(false).map_err(configure)?
        .with_intra_op_spinning(false).map_err(configure)?
        .with_inter_op_spinning(false).map_err(configure)?
        .with_memory_pattern(false).map_err(configure)?
        .with_optimization_level(GraphOptimizationLevel::Level1).map_err(configure)?;
    builder = match config.provider {
        Provider::Cpu => builder.with_execution_providers([CPUExecutionProvider::default().with_arena_allocator(false).build().error_on_failure()]).map_err(configure)?,
        Provider::Cuda { device_id, arena_bytes } => {
            let cuda = CUDAExecutionProvider::default();
            if !cuda.is_available().map_err(configure)? { return Err(fail(K::BackendUnavailable, S::Backend, "PALS CUDA provider unavailable; CPU fallback forbidden")); }
            builder.with_config_entry("session.disable_cpu_ep_fallback", "1").map_err(configure)?
                .with_execution_providers([cuda.with_device_id(device_id).with_memory_limit(arena_bytes)
                    .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested).with_tf32(false)
                    .with_cuda_graph(false).build().error_on_failure()]).map_err(configure)?
        }
    };
    // Default native copying; no direct-reference initializer option is enabled.
    // Serialized owned bytes are freed immediately after successful/failed load.
    builder.commit_from_memory(&bytes).map_err(|e| native(CauseCode::ModelLoad, "PALS ONNX loading failed", e))
}

struct ActiveInputs {
    board: Tensor<i64>, metadata: Tensor<f32>, records: Tensor<f32>, record_mask: Tensor<bool>,
    candidates: Tensor<i64>, candidate_mask: Tensor<bool>, query: Tensor<f32>,
    divergences: Tensor<f32>, divergence_mask: Tensor<bool>,
}
impl ActiveInputs {
    fn new(value: PreparedPalsTensors) -> Result<Self, BackendError> {
        let tensor_error = |e| native(CauseCode::TensorCreate, "cannot own PALS input tensor", e);
        let r = value.record_mask.len(); let c = value.candidate_mask.len(); let d = value.divergence_mask.len();
        Ok(Self {
            board: Tensor::from_array(([1,64], value.board)).map_err(tensor_error)?,
            metadata: Tensor::from_array(([1,16], value.metadata)).map_err(tensor_error)?,
            records: Tensor::from_array(([1,r,16], value.records)).map_err(tensor_error)?,
            record_mask: Tensor::from_array(([1,r], value.record_mask)).map_err(tensor_error)?,
            candidates: Tensor::from_array(([1,c,3], value.candidates)).map_err(tensor_error)?,
            candidate_mask: Tensor::from_array(([1,c], value.candidate_mask)).map_err(tensor_error)?,
            query: Tensor::from_array(([1,16], value.query)).map_err(tensor_error)?,
            divergences: Tensor::from_array(([1,d,8], value.divergences)).map_err(tensor_error)?,
            divergence_mask: Tensor::from_array(([1,d], value.divergence_mask)).map_err(tensor_error)?,
        })
    }
}
struct PublicMemory {
    key: [u8;32], tokens: usize,
    memory_key: Tensor<f32>, memory_value: Tensor<f32>, mask: Tensor<bool>,
}
/// A session's execution weight epoch never changes. It has exactly one cache
/// slot, one active input slot and one exclusive caller/physical worker.
pub struct PalsOnnxBackend {
    public: Option<Session>, proposer: Option<Session>, critic: Option<Session>,
    runtime: OrtRuntime, config: PalsOnnxConfig, model_config: PalsModelConfig,
    epoch: [u8;32], manifest_digest: [u8;32], trained: bool,
    memory: Option<PublicMemory>, active: Option<ActiveInputs>,
    quarantine: Option<BackendError>, pub public_encodes: u64, pub public_cache_hits: u64,
}
impl PalsOnnxBackend {
    pub fn load(path: &Path, expected_manifest_sha256: &str, runtime: OrtRuntime, config: PalsOnnxConfig) -> Result<Self, BackendError> {
        config.validate()?;
        if !path.is_absolute() { return Err(fail(K::InvalidInput, S::Asset, "PALS manifest path must be absolute")); }
        let bytes = asset::read_bounded(path, MAX_MANIFEST_BYTES)?;
        let manifest_digest = asset::sha256(&bytes);
        if manifest_digest != parse_sha256(expected_manifest_sha256)? { return Err(fail(K::IdentityMismatch, S::Asset, "PALS export manifest hash mismatch")); }
        let manifest: ExportManifest = serde_json::from_slice(&bytes).map_err(|e| fail(K::UnsupportedModel, S::Asset, "cannot decode PALS export manifest").with_external_cause(CauseCode::ModelLoad, &e))?;
        let epoch = validate_manifest(&manifest)?;
        match (config.provider, runtime.bundle_digest()) {
            (Provider::Cpu, Some(_)) | (Provider::Cuda { .. }, None) => return Err(fail(K::IdentityMismatch, S::Backend, "PALS provider and verified native runtime bundle differ")),
            (Provider::Cuda { .. }, Some(_)) => runtime.verify_cuda_runtime_mappings()?,
            _ => {}
        }
        let parent = path.parent().ok_or_else(|| fail(K::InvalidInput, S::Asset, "PALS export manifest has no parent"))?;
        let mut sessions = BTreeMap::new(); let mut total = 0_u64;
        for graph in &manifest.graphs {
            let graph_bytes = asset::read_bounded(&parent.join(&graph.file), MAX_GRAPH_BYTES)?;
            total = total.checked_add(graph_bytes.len() as u64).ok_or_else(|| fail(K::ResourceExhausted, S::Asset, "PALS graph byte budget overflow"))?;
            if total > MAX_MODEL_BYTES { return Err(fail(K::ResourceExhausted, S::Asset, "PALS export exceeds total serialized model budget")); }
            if asset::sha256(&graph_bytes) != parse_sha256(&graph.sha256)? { return Err(fail(K::IdentityMismatch, S::Asset, "PALS ONNX graph hash mismatch")); }
            let session = load_session(graph_bytes, config)?;
            validate_session(&session, graph, &manifest.checkpoint_sha256, manifest.trained, manifest.training_steps)?;
            sessions.insert(graph.role.clone(), session);
        }
        Ok(Self {
            public: sessions.remove("public"), proposer: sessions.remove("proposer"), critic: sessions.remove("critic"),
            runtime, config, model_config: manifest.config, epoch, manifest_digest, trained: manifest.trained,
            memory: None, active: None, quarantine: None, public_encodes: 0, public_cache_hits: 0,
        })
    }
    pub fn model_epoch(&self) -> [u8;32] { self.epoch }
    pub fn manifest_digest(&self) -> [u8;32] { self.manifest_digest }
    pub fn is_trained(&self) -> bool { self.trained }
    pub fn quarantine_cause(&self) -> Option<&BackendError> { self.quarantine.as_ref() }
    pub fn verify_runtime(&self) -> Result<(), BackendError> {
        if matches!(self.config.provider, Provider::Cuda { .. }) { self.runtime.verify_cuda_runtime_mappings()?; }
        Ok(())
    }
    pub fn worker(mut self) -> Result<SingleWorker<PalsModelInput, Result<PalsRawOutput, BackendError>>, BackendError> {
        SingleWorker::spawn_with_outcome(move |input| {
            let result = self.run(input);
            match self.quarantine.as_ref() {
                Some(cause) => PhysicalRun::Quarantined(cause.clone()),
                None => PhysicalRun::Complete(result),
            }
        })
    }
    pub fn run(&mut self, input: &PalsModelInput) -> Result<PalsRawOutput, BackendError> {
        if let Some(cause) = &self.quarantine { return Err(cause.clone()); }
        input.validate(&self.model_config).map_err(model_input)?;
        if input.model_epoch != self.epoch { return Err(fail(K::IdentityMismatch, S::Admission, "PALS request weight epoch differs from frozen session")); }
        if input.role == PalsRole::Validator { return Err(fail(K::UnsupportedModel, S::Admission, "validator inference is absent from P/C product export")); }
        let prepared = input.prepare_tensors(&self.model_config).map_err(model_input)?;
        let key = prepared.public_memory_key;
        self.active = Some(ActiveInputs::new(prepared)?);
        let result = self.run_active(input, key);
        if self.quarantine.is_none() { self.active = None; }
        result
    }
    fn run_active(&mut self, input: &PalsModelInput, key: [u8;32]) -> Result<PalsRawOutput, BackendError> {
        let cached = self.config.cache_public_memory && self.memory.as_ref().is_some_and(|m| m.key == key);
        if cached { self.public_cache_hits = self.public_cache_hits.saturating_add(1); }
        else {
            let active = self.active.as_ref().expect("owned input installed");
            let result = self.public.as_mut().expect("loaded public session").run(ort::inputs!["board" => &active.board, "metadata" => &active.metadata, "records" => &active.records, "record_mask" => &active.record_mask]);
            let outputs = match result {
                Ok(outputs) => outputs,
                Err(error) => {
                    let failure = native(CauseCode::OrtRun, "PALS public-memory native Run failed", error);
                    if matches!(self.config.provider, Provider::Cuda { .. }) { self.quarantine = Some(failure.clone()); }
                    return Err(failure);
                }
            };
            let tokens = self.model_config.public_memory_tokens(input.records.len()).map_err(model_input)?;
            let extract = |name: &str| -> Result<Vec<f32>, BackendError> {
                let (shape, values) = outputs[name].try_extract_tensor::<f32>().map_err(|e| native(CauseCode::PolicyExtract, "PALS public-memory extraction failed", e))?;
                if shape.as_ref() != [1,2,tokens as i64,64] || values.iter().any(|v| !v.is_finite()) { return Err(fail(K::NumericalFailure, S::Output, "PALS public-memory output shape/finite check failed")); }
                Ok(values.to_vec())
            };
            let memory_key = Tensor::from_array(([1,2,tokens,64], extract("memory_key")?)).map_err(|e| native(CauseCode::TensorCreate, "PALS public-memory ownership failed", e))?;
            let memory_value = Tensor::from_array(([1,2,tokens,64], extract("memory_value")?)).map_err(|e| native(CauseCode::TensorCreate, "PALS public-memory ownership failed", e))?;
            let (shape, values) = outputs["memory_mask"].try_extract_tensor::<bool>().map_err(|e| native(CauseCode::PolicyExtract, "PALS public-memory mask extraction failed", e))?;
            if shape.as_ref() != [1,tokens as i64] || values[..66].iter().any(|v| !*v) || values[66..].iter().filter(|v| **v).count() != input.records.len() {
                return Err(fail(K::NumericalFailure, S::Output, "PALS public-memory mask differs from admitted records"));
            }
            let mask = Tensor::from_array(([1,tokens], values.to_vec())).map_err(|e| native(CauseCode::TensorCreate, "PALS public-memory mask ownership failed", e))?;
            drop(outputs);
            self.memory = Some(PublicMemory { key, tokens, memory_key, memory_value, mask });
            self.public_encodes = self.public_encodes.saturating_add(1);
        }
        let active = self.active.as_ref().expect("owned inputs remain installed");
        let memory = self.memory.as_ref().expect("public memory installed after successful fence");
        debug_assert_eq!(memory.tokens, self.model_config.public_memory_tokens(input.records.len()).expect("validated"));
        let critic = input.role == PalsRole::Critic;
        let result = if critic {
            self.critic.as_mut().expect("loaded critic session").run(ort::inputs!["memory_key" => &memory.memory_key, "memory_value" => &memory.memory_value, "memory_mask" => &memory.mask, "candidates" => &active.candidates, "candidate_mask" => &active.candidate_mask, "query" => &active.query, "divergence_features" => &active.divergences, "divergence_mask" => &active.divergence_mask])
        } else {
            self.proposer.as_mut().expect("loaded proposer session").run(ort::inputs!["memory_key" => &memory.memory_key, "memory_value" => &memory.memory_value, "memory_mask" => &memory.mask, "candidates" => &active.candidates, "candidate_mask" => &active.candidate_mask, "query" => &active.query])
        };
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                let failure = native(CauseCode::OrtRun, "PALS private-role native Run failed", error);
                if matches!(self.config.provider, Provider::Cuda { .. }) { self.quarantine = Some(failure.clone()); }
                return Err(failure);
            }
        };
        let extract = |name: &str, expected: &[i64]| -> Result<Vec<f32>, BackendError> {
            let (shape, values) = outputs[name].try_extract_tensor::<f32>().map_err(|e| native(CauseCode::PolicyExtract, "PALS role output extraction failed", e))?;
            if shape.as_ref() != expected || values.iter().any(|v| !v.is_finite()) { return Err(fail(K::NumericalFailure, S::Output, "PALS role output shape/finite check failed")); }
            Ok(values.to_vec())
        };
        let mut candidate_logits = extract("candidate_logits", &[1,input.candidates.len().max(1) as i64])?;
        candidate_logits.truncate(input.candidates.len());
        let wdl_logits = extract("wdl_logits", &[1,3])?.try_into().map_err(|_| fail(K::NumericalFailure, S::Output, "PALS WDL shape failed"))?;
        let private_latent = extract("private_latent", &[1,16,384])?;
        let divergence_logits = if critic {
            let mut values = extract("divergence_logits", &[1,input.divergence_features.len().max(1) as i64])?;
            values.truncate(input.divergence_features.len()); Some(values)
        } else { None };
        let output = PalsRawOutput { candidate_logits, wdl_logits, private_latent, divergence_logits, task_logits: None };
        output.decode(input, &self.model_config).map_err(model_input)?;
        Ok(output)
    }
}

impl Drop for PalsOnnxBackend {
    fn drop(&mut self) {
        if self.quarantine.is_some() {
            // The worker normally retains the entire closure on quarantine. A
            // direct caller may drop its owner: retain the same resources rather
            // than using Drop as an unproven CUDA completion acknowledgement.
            for session in [self.public.take(), self.proposer.take(), self.critic.take()].into_iter().flatten() { std::mem::forget(session); }
            if let Some(active) = self.active.take() { std::mem::forget(active); }
            if let Some(memory) = self.memory.take() { std::mem::forget(memory); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_model::{PalsCandidateToken, PalsRecordToken};
    fn input() -> PalsModelInput {
        PalsModelInput { role: PalsRole::Proposer, board: vec![0;64], metadata: [0.;16],
            records: vec![PalsRecordToken { record_id: 1, revision: 2, critical: true, features: [0.;16] }],
            required_critical_records: vec![1], candidates: vec![PalsCandidateToken { from: 12, to: 28, promotion: 0 }],
            divergence_features: vec![], query: [0.;16], situation_revision: 3, model_epoch: [4;32], history_digest: [5;32] }
    }
    #[test]
    fn public_cache_role_neutral_private_query_and_candidates_are_not_memory() {
        let a = input(); let mut b = a.clone(); b.role = PalsRole::Critic;
        b.divergence_features.push([1.;8]); b.query[0] = 9.; b.candidates.clear();
        let key = |input: &PalsModelInput| input.public_memory_key(&PalsModelConfig::baseline()).unwrap();
        assert_eq!(key(&a), key(&b));
        b.records[0].revision += 1; assert_ne!(key(&a), key(&b));
        b = a.clone(); b.model_epoch[0] += 1; assert_ne!(key(&a), key(&b));
        b = a.clone(); b.metadata[0] = -0.; assert_ne!(key(&a), key(&b));
    }
    #[test]
    fn descriptor_rejects_fixed_shapes_and_wrong_dtype() {
        let expected = [spec("board", TensorElementType::Int64, &[-1,64])];
        let mut fields = vec![TensorManifest { name: "board".into(), dtype: "INT64".into(), shape: vec!["batch".into(),64.into()] }];
        validate_declared_interface(&fields, &expected).unwrap();
        fields[0].shape[0] = 1.into(); assert!(validate_declared_interface(&fields, &expected).is_err());
        fields[0].shape[0] = "batch".into(); fields[0].dtype = "FLOAT".into(); assert!(validate_declared_interface(&fields, &expected).is_err());
    }
}
