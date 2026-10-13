//! Explicit V3 export admission. V1/V2 manifests never enter this parser.
//! Initializer-bank declarations describe serialization, not native residency.
use super::super::{fail, native, GraphManifest, TensorManifest};
use crate::asset::parse_sha256;
use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use crate::pals_model::virtual_review::{
    ENCODING, LENGTH_BUCKETS, MODEL_SCHEMA, MODEL_SEMANTICS, PROFILE, QUERY_FEATURE_NAMES,
    RECORD_FEATURE_NAMES,
};
use crate::pals_model::{PalsModelConfig, PalsModelProfile, V_TASK_NAMES};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

pub(super) const LAYOUT: &str = "virtual_review_shared_pc_if_v1";
pub(super) const WARM_LAYOUT: &str = "virtual_review_shared_pc_if_approx_warm_v1";
// V3 carries three explicit graph/initializer ledgers in Collection. This new
// bounded domain does not alter the legacy V1/V2 128KiB admission.
pub(super) const MAX_MANIFEST_BYTES_V3: usize = 256 * 1024;
const WARM_SEED_POLICY: &str =
    "accepted_final_latent_same_model_role_rules_query_epoch_physical_fence";
const RULES_PROFILE: &str = "rz-pals-rules-virtual-review-v1";
const UNVERIFIED: &str = "unverified_declaration_only";
const REVIEW_FEATURES: [&str; 14] = [
    "review.anchor_board",
    "review.anchor_rules_metadata",
    "review.record_anchor_local",
    "review.record_moves",
    "review.record_lengths",
    "review.record_masks",
    "review.record_kind",
    "review.parent_local",
    "review.supersedes_local",
    "review.query_moves",
    "review.query_lengths",
    "review.query_masks",
    "review.query_anchor_local",
    "review.physical_line_bucket",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub schema: String,
    pub model_semantics: String,
    pub encoding_schema: String,
    pub model_profile: PalsModelProfile,
    pub config: PalsModelConfig,
    pub checkpoint_sha256: String,
    pub trained: bool,
    pub training_steps: u64,
    pub roles: Vec<String>,
    pub validator_present: bool,
    pub layout: String,
    layout_revision: u32,
    pub graphs: Vec<Graph>,
    task_names: Vec<String>,
    physical_line_buckets: Vec<usize>,
    session_policy: String,
    // A bounded declaration is retained for provenance. It is not checked
    // native weight ownership and grants no sharing/cache capability.
    initializer_banks: BTreeMap<String, Value>,
    precision: String,
    tf32: bool,
    #[serde(default)]
    approximate: Option<bool>,
    #[serde(default)]
    warm_seed_policy: Option<String>,
    runtime: Runtime,
    candidate_promotion: BTreeMap<String, String>,
    wdl_perspective: String,
    numeric_status: String,
    cuda_status: String,
    pub rules_input_profile: String,
    pub rules_input_semantic_sha256: String,
    pub rules_encoder_source_sha256: String,
    pub(super) rules_profile_descriptor_sha256: String,
    rules_profile_canonical_sha256: String,
    rules_input_declaration: RulesDeclaration,
    learned_input_compatibility: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Graph {
    pub role: String,
    pub file: String,
    pub sha256: String,
    pub opset: u32,
    inputs: Vec<TensorDescriptor>,
    outputs: Vec<TensorDescriptor>,
}
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorDescriptor {
    name: String,
    dtype: String,
    shape: Vec<Value>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Runtime {
    onnxruntime: String,
    rust_ort: String,
    compatibility: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RulesDeclaration {
    schema: String,
    rules_input_profile: String,
    config: PalsModelConfig,
    rules_base_semantics: String,
    metadata_features: Vec<String>,
    record_features: Vec<String>,
    query_features: Vec<String>,
    divergence_features: Vec<String>,
    virtual_review_features: Vec<String>,
    bounds: Bounds,
    rules_input_semantic_sha256: String,
    rules_encoder_source_sha256: String,
    learned_input_compatibility: String,
    semantic_digest_algorithm: String,
    semantic_fields: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bounds {
    max_records: usize,
    max_anchors: usize,
    max_line_plies: usize,
    line_buckets: Vec<usize>,
    max_candidates: usize,
    token_budgets: Vec<usize>,
    root_anchor: usize,
}

impl Manifest {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self, BackendError> {
        Self::parse_kind(bytes, false)
    }
    /// Explicit separate entry: a Warm artifact can never pass Fresh admission.
    pub(super) fn parse_warm(bytes: &[u8]) -> Result<Self, BackendError> {
        Self::parse_kind(bytes, true)
    }
    pub(super) fn is_warm(&self) -> bool {
        self.layout == WARM_LAYOUT
    }
    fn parse_kind(bytes: &[u8], warm: bool) -> Result<Self, BackendError> {
        if bytes.is_empty() || bytes.len() > MAX_MANIFEST_BYTES_V3 {
            return Err(fail(
                K::ResourceExhausted,
                S::Asset,
                "PALS V3 manifest exceeds bounded admission",
            ));
        }
        // Reject duplicate keys even inside evidence objects before decoding
        // strict typed authority. serde_json::Value alone keeps the last key.
        let unique: UniqueJson = serde_json::from_slice(bytes).map_err(decode_error)?;
        let object = unique.0.as_object().ok_or_else(|| {
            fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS V3 manifest must be an object",
            )
        })?;
        if !warm && (object.contains_key("approximate") || object.contains_key("warm_seed_policy"))
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "Fresh PALS V3 manifest cannot contain Warm capability fields",
            ));
        }
        let manifest: Self = serde_json::from_value(unique.0).map_err(decode_error)?;
        manifest.validate(warm)?;
        Ok(manifest)
    }

    pub(super) fn epoch(&self) -> Result<[u8; 32], BackendError> {
        strict_digest(&self.checkpoint_sha256)
    }

    fn validate(&self, warm: bool) -> Result<(), BackendError> {
        self.config.validate().map_err(|e| {
            fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS V3 configuration differs from registered profile",
            )
            .with_external_cause(CauseCode::ModelLoad, &e)
        })?;
        if self.config.profile != PalsModelProfile::VirtualReviewV1
            || self.model_profile != self.config.profile
            || self.schema != MODEL_SCHEMA
            || self.model_semantics != MODEL_SEMANTICS
            || self.encoding_schema != ENCODING
            || self.layout != if warm { WARM_LAYOUT } else { LAYOUT }
            || self.layout_revision != 1
            || self.precision != "fp32"
            || self.tf32
            || self.physical_line_buckets.as_slice() != LENGTH_BUCKETS
            || self.session_policy != "one_dynamic_L_graph_no_bucket_weight_copies"
            || self.runtime.onnxruntime != "1.22.0"
            || self.runtime.rust_ort != "2.0.0-rc.10"
            || self.runtime.compatibility
                != if warm {
                    "requires_actual_seeded_numeric_check"
                } else {
                    "requires_actual_numeric_check"
                }
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS V3 semantic/layout/runtime declaration differs",
            ));
        }
        self.epoch()?;
        if if warm {
            self.validator_present
                || self.approximate != Some(true)
                || self.warm_seed_policy.as_deref() != Some(WARM_SEED_POLICY)
        } else {
            self.approximate.is_some() || self.warm_seed_policy.is_some()
        } {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS V3 Fresh/Warm capability declaration differs",
            ));
        }
        if self.trained != (self.training_steps > 0) || self.task_names != V_TASK_NAMES
            || self.wdl_perspective != "input_side_to_move"
            // The exporter does not perform either acceptance check. Admission
            // cannot promote a rewritten status into numerical/CUDA evidence.
            || self.numeric_status != "not_run" || self.cuda_status != "not_run"
            || self.learned_input_compatibility != UNVERIFIED
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V3 training/output provenance differs",
            ));
        }
        let roles: Vec<&str> = if self.validator_present {
            vec!["proposer", "critic", "validator"]
        } else {
            vec!["proposer", "critic"]
        };
        let graph_roles: Vec<&str> = if warm {
            vec!["public", "shared_warm"]
        } else if self.validator_present {
            vec!["public", "shared_pc", "validator"]
        } else {
            vec!["public", "shared_pc"]
        };
        if self.roles.iter().map(String::as_str).collect::<Vec<_>>() != roles
            || self
                .graphs
                .iter()
                .map(|g| g.role.as_str())
                .collect::<Vec<_>>()
                != graph_roles
            || self.initializer_banks.len() != graph_roles.len()
            || graph_roles
                .iter()
                .any(|r| !self.initializer_banks.get(*r).is_some_and(Value::is_object))
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS V3 collection/deployment graph ownership differs",
            ));
        }
        let promotion: BTreeMap<String, String> = ["none", "queen", "rook", "bishop", "knight"]
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.to_string()))
            .collect();
        if self.candidate_promotion != promotion {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V3 promotion order differs",
            ));
        }
        for graph in &self.graphs {
            graph.validate()?;
        }
        self.validate_rules()
    }

    fn validate_rules(&self) -> Result<(), BackendError> {
        let d = &self.rules_input_declaration;
        if self.rules_input_profile != RULES_PROFILE
            || d.rules_input_profile != RULES_PROFILE
            || d.schema != "rovezero.pals-rules-descriptor.v3"
            || d.config != self.config
            || d.rules_base_semantics.is_empty()
            || d.rules_base_semantics.len() > 4096
            || d.record_features != RECORD_FEATURE_NAMES
            || d.query_features != QUERY_FEATURE_NAMES
            || d.virtual_review_features != REVIEW_FEATURES
            || d.metadata_features.len() != 16
            || d.divergence_features.len() != 8
            || d.metadata_features
                .iter()
                .chain(&d.divergence_features)
                .any(|s| s.is_empty() || s.len() > 4096)
            || d.learned_input_compatibility != UNVERIFIED
            || d.semantic_digest_algorithm != "sha256_u64le_length_prefixed_utf8_fields"
            || d.bounds.max_records != 64
            || d.bounds.max_anchors != 4
            || d.bounds.max_line_plies != 256
            || d.bounds.line_buckets.as_slice() != LENGTH_BUCKETS
            || d.bounds.max_candidates != 256
            || d.bounds.token_budgets != [128, 256, 512]
            || d.bounds.root_anchor != 0
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V3 Rules vocabulary/bounds differ",
            ));
        }
        let fields: Vec<&str> = std::iter::once(RULES_PROFILE)
            .chain(std::iter::once(d.rules_base_semantics.as_str()))
            .chain(d.metadata_features.iter().map(String::as_str))
            .chain(RECORD_FEATURE_NAMES)
            .chain(QUERY_FEATURE_NAMES)
            .chain(d.divergence_features.iter().map(String::as_str))
            .chain(REVIEW_FEATURES)
            .collect();
        if d.semantic_fields
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != fields
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V3 Rules semantic framing differs",
            ));
        }
        let mut hash = Sha256::new();
        for field in fields {
            hash.update((field.len() as u64).to_le_bytes());
            hash.update(field.as_bytes());
        }
        let semantic: [u8; 32] = hash.finalize().into();
        if semantic != strict_digest(&d.rules_input_semantic_sha256)?
            || semantic != strict_digest(&self.rules_input_semantic_sha256)?
            || strict_digest(&d.rules_encoder_source_sha256)?
                != strict_digest(&self.rules_encoder_source_sha256)?
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V3 Rules source/semantic digest differs",
            ));
        }
        strict_digest(&self.rules_profile_descriptor_sha256)?;
        let value = serde_json::to_value(d).map_err(decode_error)?;
        let canonical = serde_json::to_vec(&canonical_value(value)).map_err(decode_error)?;
        if canonical.len() > 65536
            || <[u8; 32]>::from(Sha256::digest(&canonical))
                != strict_digest(&self.rules_profile_canonical_sha256)?
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V3 Rules canonical descriptor differs",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_session(
        &self,
        session: &Session,
        graph: &Graph,
    ) -> Result<(), BackendError> {
        graph.validate()?;
        if !self.graphs.iter().any(|registered| {
            registered.role == graph.role
                && registered.file == graph.file
                && registered.sha256 == graph.sha256
                && registered.opset == graph.opset
        }) {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V3 session graph is not registered by this manifest",
            ));
        }
        let valid = |name: &str, ty: &ValueType, expected: &TensorDescriptor| {
            name == expected.name
                && matches!(ty, ValueType::Tensor { ty, shape, dimension_symbols }
                if Some(*ty) == element_type(&expected.dtype)
                    && shape.len() == expected.shape.len()
                    && dimension_symbols.len() == expected.shape.len()
                    && expected.native_axes_match(shape.iter().zip(dimension_symbols.iter()).map(|(size,symbol)|(*size,symbol.as_str()))))
        };
        if session.inputs.len() != graph.inputs.len()
            || session.outputs.len() != graph.outputs.len()
            || !session
                .inputs
                .iter()
                .zip(&graph.inputs)
                .all(|(v, e)| valid(&v.name, &v.input_type, e))
            || !session
                .outputs
                .iter()
                .zip(&graph.outputs)
                .all(|(v, e)| valid(&v.name, &v.output_type, e))
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "loaded PALS V3 tensor interface differs",
            ));
        }
        let metadata = session
            .metadata()
            .map_err(|e| native(CauseCode::ModelLoad, "cannot inspect PALS V3 metadata", e))?;
        let steps = self.training_steps.to_string();
        for (key, expected) in [
            ("schema", MODEL_SCHEMA),
            ("model_semantics", MODEL_SEMANTICS),
            ("model_profile", PROFILE),
            ("encoding_schema", ENCODING),
            ("layout", self.layout.as_str()),
            ("layout_revision", "1"),
            ("role", graph.role.as_str()),
            ("checkpoint_sha256", self.checkpoint_sha256.as_str()),
            ("trained", if self.trained { "true" } else { "false" }),
            ("training_steps", steps.as_str()),
            ("precision", "fp32"),
            ("expected_ort", "1.22.0"),
            ("rules_input_profile", RULES_PROFILE),
            (
                "rules_input_semantic_sha256",
                self.rules_input_semantic_sha256.as_str(),
            ),
            (
                "rules_encoder_source_sha256",
                self.rules_encoder_source_sha256.as_str(),
            ),
        ] {
            if metadata
                .custom(key)
                .map_err(|e| native(CauseCode::ModelLoad, "cannot read PALS V3 metadata", e))?
                .as_deref()
                != Some(expected)
            {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Asset,
                    "PALS V3 graph metadata differs from export",
                ));
            }
        }
        let approximate = metadata
            .custom("approximate")
            .map_err(|e| native(CauseCode::ModelLoad, "cannot read PALS V3 Warm metadata", e))?;
        if if self.is_warm() {
            approximate.as_deref() != Some("true")
        } else {
            approximate.is_some()
        } {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V3 Warm approximation metadata differs",
            ));
        }
        Ok(())
    }
}

impl Graph {
    fn validate(&self) -> Result<(), BackendError> {
        let file = match self.role.as_str() {
            "public" => "public_memory.onnx",
            "shared_pc" => "shared_pc_if.onnx",
            "validator" => "role_validator.onnx",
            "shared_warm" => "shared_pc_if_approx_warm.onnx",
            _ => {
                return Err(fail(
                    K::UnsupportedModel,
                    S::Asset,
                    "unsupported PALS V3 graph role",
                ));
            }
        };
        if self.file != file
            || self.opset != 17
            || self.inputs != expected_inputs(&self.role)
            || self.outputs != expected_outputs(&self.role)
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS V3 graph filename/dynamic interface differs",
            ));
        }
        strict_digest(&self.sha256)?;
        Ok(())
    }
    /// Bridges existing loading/profile evidence only. Legacy shape validators
    /// must never validate or execute this V3 graph descriptor.
    pub(super) fn legacy_pin_descriptor(&self) -> GraphManifest {
        let convert = |v: &TensorDescriptor| TensorManifest {
            name: v.name.clone(),
            dtype: v.dtype.clone(),
            shape: v.shape.clone(),
        };
        GraphManifest {
            role: self.role.clone(),
            file: self.file.clone(),
            sha256: self.sha256.clone(),
            opset: self.opset,
            inputs: self.inputs.iter().map(convert).collect(),
            outputs: self.outputs.iter().map(convert).collect(),
        }
    }
}
impl TensorDescriptor {
    fn native_axes_match<'a>(&self, axes: impl ExactSizeIterator<Item = (i64, &'a str)>) -> bool {
        axes.len() == self.shape.len()
            && self
                .shape
                .iter()
                .zip(axes)
                .all(|(expected, (size, symbol))| match expected {
                    Value::String(expected) => size == -1 && symbol == expected,
                    Value::Number(expected) => expected.as_i64() == Some(size) && symbol.is_empty(),
                    _ => false,
                })
    }
}
fn element_type(dtype: &str) -> Option<TensorElementType> {
    match dtype {
        "FLOAT" => Some(TensorElementType::Float32),
        "INT64" => Some(TensorElementType::Int64),
        "BOOL" => Some(TensorElementType::Bool),
        _ => None,
    }
}
fn tensor(name: &str, dtype: &str, shape: &[Value]) -> TensorDescriptor {
    TensorDescriptor {
        name: name.to_string(),
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
    }
}
fn expected_inputs(role: &str) -> Vec<TensorDescriptor> {
    use serde_json::json as j;
    if role == "public" {
        return vec![
            tensor(
                "anchor_board",
                "INT64",
                &[j!("batch"), j!("anchors"), j!(64)],
            ),
            tensor(
                "anchor_metadata",
                "FLOAT",
                &[j!("batch"), j!("anchors"), j!(16)],
            ),
            tensor("anchor_mask", "BOOL", &[j!("batch"), j!("anchors")]),
            tensor("records", "FLOAT", &[j!("batch"), j!("records"), j!(8)]),
            tensor("record_anchor", "INT64", &[j!("batch"), j!("records")]),
            tensor(
                "record_line_tokens",
                "INT64",
                &[j!("batch"), j!("records"), j!("line_bucket"), j!(3)],
            ),
            tensor(
                "record_line_mask",
                "BOOL",
                &[j!("batch"), j!("records"), j!("line_bucket")],
            ),
            tensor("record_mask", "BOOL", &[j!("batch"), j!("records")]),
        ];
    }
    let public = expected_outputs("public");
    let mut out = Vec::new();
    if role == "shared_pc" || role == "shared_warm" {
        out.push(tensor("role_is_critic", "BOOL", &[]));
    }
    out.extend(public[..3].iter().cloned());
    out.extend([
        tensor(
            "candidates",
            "INT64",
            &[j!("batch"), j!("candidates"), j!(3)],
        ),
        tensor("candidate_mask", "BOOL", &[j!("batch"), j!("candidates")]),
        tensor(
            "divergence_features",
            "FLOAT",
            &[j!("batch"), j!("divergences"), j!(8)],
        ),
        tensor("divergence_mask", "BOOL", &[j!("batch"), j!("divergences")]),
        tensor("query", "FLOAT", &[j!("batch"), j!(8)]),
        public[3].clone(),
        tensor("query_anchor", "INT64", &[j!("batch"), j!(3)]),
        tensor(
            "query_line_tokens",
            "INT64",
            &[j!("batch"), j!(3), j!("line_bucket"), j!(3)],
        ),
        tensor(
            "query_line_mask",
            "BOOL",
            &[j!("batch"), j!(3), j!("line_bucket")],
        ),
        tensor(
            "record_relations",
            "INT64",
            &[j!("batch"), j!("records"), j!(2)],
        ),
    ]);
    if role == "shared_warm" {
        out.extend([
            tensor("initial_latent", "FLOAT", &[j!("batch"), j!(16), j!(384)]),
            tensor("warm_start", "BOOL", &[]),
        ]);
    }
    out
}
fn expected_outputs(role: &str) -> Vec<TensorDescriptor> {
    use serde_json::json as j;
    if role == "public" {
        return vec![
            tensor(
                "memory_key",
                "FLOAT",
                &[j!("batch"), j!(2), j!("memory_tokens"), j!(64)],
            ),
            tensor(
                "memory_value",
                "FLOAT",
                &[j!("batch"), j!(2), j!("memory_tokens"), j!(64)],
            ),
            tensor("memory_mask", "BOOL", &[j!("batch"), j!("memory_tokens")]),
            tensor(
                "anchor_context",
                "FLOAT",
                &[j!("batch"), j!("anchors"), j!(384)],
            ),
        ];
    }
    let mut out = vec![
        tensor(
            "candidate_logits",
            "FLOAT",
            &[j!("batch"), j!("candidates")],
        ),
        tensor("wdl_logits", "FLOAT", &[j!("batch"), j!(3)]),
        tensor("private_latent", "FLOAT", &[j!("batch"), j!(16), j!(384)]),
    ];
    if role == "validator" {
        out.push(tensor("task_logits", "FLOAT", &[j!("batch"), j!(7)]));
    } else {
        out.extend([
            tensor(
                "divergence_logits",
                "FLOAT",
                &[j!("batch"), j!("divergences")],
            ),
            tensor("is_critic", "BOOL", &[]),
        ]);
    }
    out
}
fn strict_digest(value: &str) -> Result<[u8; 32], BackendError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS V3 digest must be lowercase SHA-256",
        ));
    }
    parse_sha256(value)
}
fn decode_error(error: serde_json::Error) -> BackendError {
    fail(
        K::UnsupportedModel,
        S::Asset,
        "cannot decode strict PALS V3 export manifest",
    )
    .with_external_cause(CauseCode::ModelLoad, &error)
}
fn canonical_value(value: Value) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .map(|(k, v)| (k, canonical_value(v)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_value).collect()),
        other => other,
    }
}

/// JSON evidence never silently replaces an earlier key with a later one.
struct UniqueJson(Value);
impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueJson;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Bool(v)))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| UniqueJson(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite JSON"))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(v.to_string())))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(v)))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(v) = seq.next_element::<UniqueJson>()? {
                    values.push(v.0);
                }
                Ok(UniqueJson(Value::Array(values)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(k) = map.next_key::<String>()? {
                    if values.contains_key(&k) {
                        return Err(serde::de::Error::custom("duplicate JSON key"));
                    }
                    values.insert(k, map.next_value::<UniqueJson>()?.0);
                }
                Ok(UniqueJson(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture_manifest(validator: bool) -> Value {
        use serde_json::json;
        let config = PalsModelConfig::for_profile(PalsModelProfile::VirtualReviewV1);
        let metadata: Vec<String> = (0..16).map(|i| format!("metadata{i}")).collect();
        let divergences: Vec<String> = (0..8).map(|i| format!("divergence{i}")).collect();
        let fields: Vec<String> = std::iter::once(RULES_PROFILE)
            .chain(std::iter::once("explicit_rules_features"))
            .chain(metadata.iter().map(String::as_str))
            .chain(RECORD_FEATURE_NAMES)
            .chain(QUERY_FEATURE_NAMES)
            .chain(divergences.iter().map(String::as_str))
            .chain(REVIEW_FEATURES)
            .map(str::to_string)
            .collect();
        let mut hash = Sha256::new();
        for field in &fields {
            hash.update((field.len() as u64).to_le_bytes());
            hash.update(field.as_bytes());
        }
        let digest = hash
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let declaration = json!({"schema":"rovezero.pals-rules-descriptor.v3","rules_input_profile":RULES_PROFILE,
            "config":config,"rules_base_semantics":"explicit_rules_features","metadata_features":metadata,
            "record_features":RECORD_FEATURE_NAMES,"query_features":QUERY_FEATURE_NAMES,"divergence_features":divergences,
            "virtual_review_features":REVIEW_FEATURES,"bounds":{"max_records":64,"max_anchors":4,"max_line_plies":256,
                "line_buckets":LENGTH_BUCKETS,"max_candidates":256,"token_budgets":[128,256,512],"root_anchor":0},
            "rules_input_semantic_sha256":digest,"rules_encoder_source_sha256":"11".repeat(32),
            "learned_input_compatibility":UNVERIFIED,"semantic_digest_algorithm":"sha256_u64le_length_prefixed_utf8_fields",
            "semantic_fields":fields});
        let canonical = serde_json::to_vec(&canonical_value(declaration.clone())).unwrap();
        let canonical_pin = Sha256::digest(canonical)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let mut graph_roles = vec!["public", "shared_pc"];
        let mut roles = vec!["proposer", "critic"];
        if validator {
            graph_roles.push("validator");
            roles.push("validator");
        }
        let descriptors = |values: Vec<TensorDescriptor>| {
            values
                .into_iter()
                .map(|v| json!({"name":v.name,"dtype":v.dtype,"shape":v.shape}))
                .collect::<Vec<_>>()
        };
        let graphs=graph_roles.iter().map(|role|json!({"role":role,"file":match *role {
            "public"=>"public_memory.onnx","shared_pc"=>"shared_pc_if.onnx",_=>"role_validator.onnx"},
            "sha256":"22".repeat(32),"opset":17,"inputs":descriptors(expected_inputs(role)),"outputs":descriptors(expected_outputs(role))})).collect::<Vec<_>>();
        let banks = graph_roles
            .iter()
            .map(|role| {
                (
                    role.to_string(),
                    json!({"scope":"onnx_serialized_initializers_only"}),
                )
            })
            .collect::<BTreeMap<_, _>>();
        json!({"schema":MODEL_SCHEMA,"model_semantics":MODEL_SEMANTICS,"encoding_schema":ENCODING,"model_profile":PROFILE,
            "config":config,"checkpoint_sha256":"33".repeat(32),"trained":false,"training_steps":0,"roles":roles,
            "validator_present":validator,"layout":LAYOUT,"layout_revision":1,"graphs":graphs,"task_names":V_TASK_NAMES,
            "physical_line_buckets":LENGTH_BUCKETS,"session_policy":"one_dynamic_L_graph_no_bucket_weight_copies",
            "initializer_banks":banks,"precision":"fp32","tf32":false,"runtime":{"onnxruntime":"1.22.0",
                "rust_ort":"2.0.0-rc.10","compatibility":"requires_actual_numeric_check"},
            "candidate_promotion":{"0":"none","1":"queen","2":"rook","3":"bishop","4":"knight"},
            "wdl_perspective":"input_side_to_move","numeric_status":"not_run","cuda_status":"not_run",
            "rules_input_profile":RULES_PROFILE,"rules_input_semantic_sha256":digest,"rules_encoder_source_sha256":"11".repeat(32),
            "rules_profile_descriptor_sha256":"44".repeat(32),"rules_profile_canonical_sha256":canonical_pin,
            "rules_input_declaration":declaration,"learned_input_compatibility":UNVERIFIED})
    }
    fn fixture_warm() -> Value {
        use serde_json::json;
        let mut value = fixture_manifest(false);
        value["layout"] = json!(WARM_LAYOUT);
        value["approximate"] = json!(true);
        value["warm_seed_policy"] = json!(WARM_SEED_POLICY);
        value["runtime"]["compatibility"] = json!("requires_actual_seeded_numeric_check");
        let descriptors = |values: Vec<TensorDescriptor>| {
            values
                .into_iter()
                .map(|v| json!({"name":v.name,"dtype":v.dtype,"shape":v.shape}))
                .collect::<Vec<_>>()
        };
        value["graphs"][1] = json!({"role":"shared_warm","file":"shared_pc_if_approx_warm.onnx","sha256":"22".repeat(32),
            "opset":17,"inputs":descriptors(expected_inputs("shared_warm")),"outputs":descriptors(expected_outputs("shared_warm"))});
        let bank = value["initializer_banks"].as_object_mut().unwrap();
        let evidence = bank.remove("shared_pc").unwrap();
        bank.insert("shared_warm".to_string(), evidence);
        value
    }
    #[test]
    fn public_preflight_summarizes_actual_manifest_bytes_without_native_execution() {
        let bytes = serde_json::to_vec(&fixture_manifest(false)).unwrap();
        let summary =
            validate_virtual_review_export_manifest(&bytes, VirtualReviewExecutionMode::Deployment)
                .unwrap();
        assert_eq!(summary.manifest_sha256(), asset::sha256(&bytes));
        assert_eq!(summary.manifest_bytes(), bytes.len());
        assert_eq!(summary.artifact_kind, VirtualReviewArtifactKind::Fresh);
        assert_eq!(summary.checkpoint_sha256, [0x33; 32]);
        assert_eq!(summary.graphs.len(), 2);
        assert_eq!(summary.graphs[1].role, "shared_pc");
        assert_eq!(summary.graphs[1].sha256, [0x22; 32]);
        assert!(!summary.validator_present);
    }
    #[test]
    fn public_preflight_rejects_v_deployment_and_requires_explicit_warm_domain() {
        let collection = serde_json::to_vec(&fixture_manifest(true)).unwrap();
        assert!(validate_virtual_review_export_manifest(
            &collection,
            VirtualReviewExecutionMode::Deployment
        )
        .is_err());
        assert!(
            validate_virtual_review_export_manifest(
                &collection,
                VirtualReviewExecutionMode::Collection
            )
            .unwrap()
            .validator_present
        );
        let warm = serde_json::to_vec(&fixture_warm()).unwrap();
        assert!(validate_virtual_review_export_manifest(
            &warm,
            VirtualReviewExecutionMode::Deployment
        )
        .is_err());
        let summary = validate_virtual_review_export_manifest_kind(
            &warm,
            VirtualReviewExecutionMode::Deployment,
            VirtualReviewArtifactKind::ApproxWarm,
        )
        .unwrap();
        assert_eq!(summary.artifact_kind, VirtualReviewArtifactKind::ApproxWarm);
        assert_eq!(summary.graphs[1].role, "shared_warm");
        let fresh = serde_json::to_vec(&fixture_manifest(false)).unwrap();
        assert!(validate_virtual_review_export_manifest_kind(
            &fresh,
            VirtualReviewExecutionMode::Deployment,
            VirtualReviewArtifactKind::ApproxWarm
        )
        .is_err());
    }
    #[test]
    fn explicit_warm_entry_requires_seed_capability_and_v_free_graphs() {
        let fresh = serde_json::to_vec(&fixture_manifest(false)).unwrap();
        let warm = serde_json::to_vec(&fixture_warm()).unwrap();
        assert!(Manifest::parse(&warm).is_err());
        assert!(Manifest::parse_warm(&fresh).is_err());
        let admitted = Manifest::parse_warm(&warm).unwrap();
        assert!(admitted.is_warm());
        assert_eq!(admitted.graphs[1].role, "shared_warm");
        assert_eq!(expected_inputs("shared_warm").len(), 16);
        for change in [0, 1, 2, 3, 4] {
            let mut value = fixture_warm();
            match change {
                0 => value["warm_seed_policy"] = serde_json::json!("unaccepted_seed"),
                1 => value["validator_present"] = serde_json::json!(true),
                2 => value["approximate"] = serde_json::json!(false),
                3 => value["graphs"][1]["inputs"]
                    .as_array_mut()
                    .unwrap()
                    .pop()
                    .map(|_| ())
                    .unwrap(),
                _ => value["graphs"][1]["role"] = serde_json::json!("shared_pc"),
            }
            assert!(Manifest::parse_warm(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        let mut fresh = fixture_manifest(false);
        fresh["approximate"] = Value::Null;
        assert!(Manifest::parse(&serde_json::to_vec(&fresh).unwrap()).is_err());
    }
    #[test]
    fn full_v3_collection_and_v_free_deployment_parse_separately() {
        for validator in [false, true] {
            let manifest =
                Manifest::parse(&serde_json::to_vec(&fixture_manifest(validator)).unwrap())
                    .unwrap();
            assert_eq!(manifest.validator_present, validator);
            assert_eq!(manifest.graphs.len(), if validator { 3 } else { 2 });
            assert_eq!(manifest.epoch().unwrap(), [0x33; 32]);
        }
    }
    #[test]
    fn v3_manifest_rejects_legacy_abi_provenance_and_warm_promotion() {
        let mutations: [fn(&mut Value); 10] = [
            |v| v["schema"] = Value::String("rovezero.pals-model.v2".into()),
            |v| v["config"]["max_records"] = serde_json::json!(128),
            |v| v["graphs"][0]["inputs"][5]["shape"][2] = serde_json::json!(256),
            |v| v["rules_input_declaration"]["semantic_fields"][2] = serde_json::json!("tampered"),
            |v| v["rules_input_declaration"]["bounds"]["root_anchor"] = serde_json::json!(1),
            |v| v["rules_profile_canonical_sha256"] = serde_json::json!("99".repeat(32)),
            |v| v["learned_input_compatibility"] = serde_json::json!("validated"),
            |v| v["initializer_banks"]["validator"] = serde_json::json!({}),
            |v| v["approximate"] = serde_json::json!(true),
            |v| v["training_steps"] = serde_json::json!(1),
        ];
        for mutate in mutations {
            let mut value = fixture_manifest(false);
            mutate(&mut value);
            assert!(Manifest::parse(&serde_json::to_vec(&value).unwrap()).is_err());
        }
    }
    #[test]
    fn graph_contract_uses_exact_symbols_and_separate_v() {
        assert_eq!(expected_inputs("public").len(), 8);
        assert_eq!(expected_inputs("shared_pc").len(), 14);
        assert_eq!(expected_inputs("validator").len(), 13);
        assert_eq!(expected_outputs("shared_pc").len(), 5);
        assert_eq!(expected_outputs("validator").len(), 4);
        let mut graph = Graph {
            role: "public".into(),
            file: "public_memory.onnx".into(),
            sha256: "22".repeat(32),
            opset: 17,
            inputs: expected_inputs("public"),
            outputs: expected_outputs("public"),
        };
        assert!(graph.validate().is_ok());
        graph.inputs[5].shape[2] = serde_json::json!(256);
        assert!(graph.validate().is_err());
        graph.inputs = expected_inputs("public");
        graph.inputs[5].shape[2] = serde_json::json!("other_dynamic_axis");
        assert!(graph.validate().is_err());
        graph.inputs = expected_inputs("public");
        graph.role = "shared_warm".into();
        assert!(graph.validate().is_err());
    }
    #[test]
    fn loaded_axes_must_preserve_dynamic_names_and_fixed_sizes() {
        let expected = expected_inputs("public").remove(5);
        assert!(expected.native_axes_match(
            [(-1, "batch"), (-1, "records"), (-1, "line_bucket"), (3, "")].into_iter()
        ));
        assert!(!expected.native_axes_match(
            [(-1, "batch"), (-1, "records"), (-1, "other_axis"), (3, "")].into_iter()
        ));
        assert!(!expected
            .native_axes_match([(-1, "batch"), (-1, "records"), (256, ""), (3, "")].into_iter()));
        assert!(!expected.native_axes_match(
            [
                (-1, "batch"),
                (-1, "records"),
                (-1, "line_bucket"),
                (3, "fixed_symbol")
            ]
            .into_iter()
        ));
        assert!(!expected
            .native_axes_match([(-1, "batch"), (-1, "records"), (-1, "line_bucket")].into_iter()));
        let scalar = tensor("role_is_critic", "BOOL", &[]);
        assert!(scalar.native_axes_match(std::iter::empty()));
    }
    #[test]
    fn malformed_and_duplicate_evidence_cannot_override_authority() {
        assert!(Manifest::parse(b"{}").is_err());
        assert!(Manifest::parse(&vec![b' '; MAX_MANIFEST_BYTES_V3 + 1]).is_err());
        assert!(serde_json::from_slice::<UniqueJson>(br#"{"schema":"a","schema":"b"}"#).is_err());
        assert!(serde_json::from_slice::<UniqueJson>(
            br#"{"initializer_banks":{"public":{"scope":1,"scope":2}}}"#
        )
        .is_err());
        assert!(strict_digest(&"AA".repeat(32)).is_err());
        assert!(strict_digest(&"ab".repeat(32)).is_ok());
    }
}
