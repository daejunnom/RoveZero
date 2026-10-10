//! Registered bytes and exact declared wire for the fixed public K/V packer.
//!
//! This module performs no file I/O, ONNX decoding/checking, native session
//! construction, allocation/copy on CUDA, Run, provider audit or physical fence.
//! The registration is supplied by the artifact owner; matching its SHA-256 is
//! byte identity, not authentication of that registration. In particular, an
//! exact `learned_weights.count == 0` declaration does not verify graph contents.
//! `RegisteredPackingArtifactBytes` is deliberately not native Run authorization.
//! A future native consumer needs separate graph-body/session/placement gates.
//! JSON parse/comparison workspaces and allocator rounding are not measured here;
//! byte bounds and the declarations below do not establish a process memory cap.
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::{json, Map, Number, Value};
use sha2::{Digest as _, Sha256};
use std::fmt;
use std::mem::size_of;

pub const DEVICE_PACKING_SCHEMA: &str = "rovezero.pals-device-packing.v1";
pub const DEVICE_PACKING_DOMAIN: &str = "rovezero.pals.weight-free-public-routing";
pub const DEVICE_PACKING_GRAPH_FILE: &str = "device_public_pack.onnx";
pub const DEVICE_PACKING_MANIFEST_FILE: &str = "device-packing.json";
pub const DEVICE_PACKING_MAX_GRAPH_BYTES: usize = 2 * 1024 * 1024;
pub const DEVICE_PACKING_MAX_MANIFEST_BYTES: usize = 512 * 1024;
pub const DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM: u64 = 1_142_291;
const BOARD: usize = 66;
const RECORDS: usize = 128;
const HEADS: usize = 2;
const DIM: usize = 64;
const TOKENS: usize = BOARD + RECORDS;
const NODE_OUTPUTS: usize = 275;
// Fixed CPU-owned offset/stop inputs, maximum current mask and finite scalar.
// These are tensor payload declarations, not observed transfer allocations.
const CONTROL_MASK_FINITE_HOST_BYTES: u64 = ((RECORDS + 1) * 8 + TOKENS + 1) as u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingArtifactPart {
    Manifest,
    Graph,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisteredPackingBytePin {
    pub bytes: u64,
    pub sha256: [u8; 32],
}

/// Caller registration only. This module does not mint producer authenticity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingArtifactRegistration {
    pub manifest: RegisteredPackingBytePin,
    pub graph: RegisteredPackingBytePin,
}

/// Finite future packing-stage declarations, not native allocation evidence.
/// Public/private sessions, record-bank owners, source transfers and the full
/// invocation remain the separate device-page/native-owner admission's job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DevicePackingResourceDeclaration {
    pub packing_session_bytes: Option<u64>,
    pub additional_owner_metadata_host_bytes: Option<u64>,
    pub max_owned_artifact_host_bytes: u64,
    pub max_declared_host_bytes: u64,
    pub max_declared_device_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DevicePackingDeclaredResources {
    /// Actual owned Vec capacities plus this capability's inline storage only.
    pub owned_artifact_host_bytes: u64,
    pub controls_mask_and_finite_host_bytes: u64,
    pub declared_additional_owner_metadata_host_bytes: u64,
    pub declared_packing_session_bytes: u64,
    /// Sum of independent maximum node outputs, including joined K/V and BOOL.
    /// It is not peak, allocator/workspace residency, or an actual CUDA count.
    pub maximum_node_output_payload_sum: u64,
    pub declared_host_bytes: u64,
    pub declared_device_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingVerificationScope {
    RegisteredByteIdentityAndExactManifestDeclarationOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingNativeVerification {
    /// Includes the protobuf body/operator/weight-free claim, session interface,
    /// provider placement, numerical results, leases and physical completion.
    NotPerformed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DevicePackingAdmissionError {
    ArtifactBounds(PackingArtifactPart),
    ByteIdentityMismatch(PackingArtifactPart),
    ManifestJson,
    ManifestDeclarationMismatch,
    UnknownResource(&'static str),
    ResourceExceeded(&'static str),
    Overflow,
    Allocation,
}
impl fmt::Display for DevicePackingAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "device packing byte admission: {self:?}")
    }
}
impl std::error::Error for DevicePackingAdmissionError {}
type Result<T> = std::result::Result<T, DevicePackingAdmissionError>;

/// Immutable owned byte identity and validated declarations. No external
/// constructor, mutable byte access, native handles or execution-success flags.
pub struct RegisteredPackingArtifactBytes {
    manifest: Vec<u8>,
    graph: Vec<u8>,
    registration: PackingArtifactRegistration,
    resources: DevicePackingDeclaredResources,
}
impl RegisteredPackingArtifactBytes {
    pub fn admit(
        manifest: &[u8],
        graph: &[u8],
        registration: PackingArtifactRegistration,
        declaration: DevicePackingResourceDeclaration,
    ) -> Result<Self> {
        validate_byte_pin(
            manifest,
            registration.manifest,
            DEVICE_PACKING_MAX_MANIFEST_BYTES,
            PackingArtifactPart::Manifest,
        )?;
        validate_byte_pin(
            graph,
            registration.graph,
            DEVICE_PACKING_MAX_GRAPH_BYTES,
            PackingArtifactPart::Graph,
        )?;
        // Reject missing/zero/overflowed declarations before JSON parsing and
        // owned-byte copying. This is not a bound on JSON allocator workspace.
        let minimum_artifact_bytes = sum(&[
            registration.manifest.bytes,
            registration.graph.bytes,
            size_of::<Self>() as u64,
        ])?;
        declared_resources(minimum_artifact_bytes, declaration)?;
        let expected = declared_manifest(registration.graph);
        let actual: StrictJson = serde_json::from_slice(manifest)
            .map_err(|_| DevicePackingAdmissionError::ManifestJson)?;
        // Number equality preserves integer-vs-float representation. Exact
        // objects reject unknown/missing fields; exact arrays preserve all ports
        // and ledger order. Duplicate keys are rejected recursively below.
        if actual.0 != expected {
            return Err(DevicePackingAdmissionError::ManifestDeclarationMismatch);
        }
        drop(actual);
        drop(expected);
        let manifest = own_bytes(manifest)?;
        let graph = own_bytes(graph)?;
        let owned_bytes = sum(&[
            u64::try_from(manifest.capacity())
                .map_err(|_| DevicePackingAdmissionError::Overflow)?,
            u64::try_from(graph.capacity()).map_err(|_| DevicePackingAdmissionError::Overflow)?,
            size_of::<Self>() as u64,
        ])?;
        let resources = declared_resources(owned_bytes, declaration)?;
        Ok(Self {
            manifest,
            graph,
            registration,
            resources,
        })
    }
    pub fn registration(&self) -> PackingArtifactRegistration {
        self.registration
    }
    pub fn resources(&self) -> DevicePackingDeclaredResources {
        self.resources
    }
    pub fn verification_scope(&self) -> PackingVerificationScope {
        PackingVerificationScope::RegisteredByteIdentityAndExactManifestDeclarationOnly
    }
    pub fn native_verification(&self) -> PackingNativeVerification {
        PackingNativeVerification::NotPerformed
    }
    /// Registered opaque bytes. Borrowing these is not authorization to Run them.
    pub fn graph_bytes(&self) -> &[u8] {
        &self.graph
    }
    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest
    }
}

fn validate_byte_pin(
    bytes: &[u8],
    pin: RegisteredPackingBytePin,
    maximum: usize,
    part: PackingArtifactPart,
) -> Result<()> {
    if bytes.is_empty() || bytes.len() > maximum || pin.bytes == 0 || pin.bytes > maximum as u64 {
        return Err(DevicePackingAdmissionError::ArtifactBounds(part));
    }
    let actual: [u8; 32] = Sha256::digest(bytes).into();
    if pin.bytes != bytes.len() as u64 || pin.sha256 != actual {
        return Err(DevicePackingAdmissionError::ByteIdentityMismatch(part));
    }
    Ok(())
}
fn own_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(bytes.len())
        .map_err(|_| DevicePackingAdmissionError::Allocation)?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}
fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |total, value| {
        total
            .checked_add(*value)
            .ok_or(DevicePackingAdmissionError::Overflow)
    })
}
fn nonzero(value: Option<u64>, name: &'static str) -> Result<u64> {
    value
        .filter(|value| *value > 0)
        .ok_or(DevicePackingAdmissionError::UnknownResource(name))
}
fn declared_resources(
    artifact_bytes: u64,
    declaration: DevicePackingResourceDeclaration,
) -> Result<DevicePackingDeclaredResources> {
    let session = nonzero(declaration.packing_session_bytes, "packing_session_bytes")?;
    let metadata = nonzero(
        declaration.additional_owner_metadata_host_bytes,
        "additional_owner_metadata_host_bytes",
    )?;
    let host = sum(&[artifact_bytes, CONTROL_MASK_FINITE_HOST_BYTES, metadata])?;
    let device = sum(&[DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM, session])?;
    if artifact_bytes > declaration.max_owned_artifact_host_bytes {
        return Err(DevicePackingAdmissionError::ResourceExceeded(
            "owned_artifact_host_bytes",
        ));
    }
    if host > declaration.max_declared_host_bytes {
        return Err(DevicePackingAdmissionError::ResourceExceeded(
            "declared_host_bytes",
        ));
    }
    if device > declaration.max_declared_device_bytes {
        return Err(DevicePackingAdmissionError::ResourceExceeded(
            "declared_device_bytes",
        ));
    }
    Ok(DevicePackingDeclaredResources {
        owned_artifact_host_bytes: artifact_bytes,
        controls_mask_and_finite_host_bytes: CONTROL_MASK_FINITE_HOST_BYTES,
        declared_additional_owner_metadata_host_bytes: metadata,
        declared_packing_session_bytes: session,
        maximum_node_output_payload_sum: DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM,
        declared_host_bytes: host,
        declared_device_bytes: device,
    })
}

fn tensor(name: String, dtype: &str, shape: Value) -> Value {
    json!({"name":name,"dtype":dtype,"shape":shape})
}
fn hex(digest: [u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(64);
    for byte in digest {
        result.push(DIGITS[usize::from(byte >> 4)] as char);
        result.push(DIGITS[usize::from(byte & 15)] as char);
    }
    result
}
fn declared_manifest(graph: RegisteredPackingBytePin) -> Value {
    let mut inputs = vec![
        tensor(
            "board_key".into(),
            "FLOAT",
            json!([1, HEADS, "base_source_tokens", DIM]),
        ),
        tensor(
            "board_value".into(),
            "FLOAT",
            json!([1, HEADS, "base_source_tokens", DIM]),
        ),
    ];
    for index in 0..RECORDS {
        let suffix = format!("{index:03}");
        let source_tokens = format!("source_tokens_{suffix}");
        inputs.push(tensor(
            format!("record_key_{suffix}"),
            "FLOAT",
            json!([1, HEADS, source_tokens, DIM]),
        ));
        inputs.push(tensor(
            format!("record_value_{suffix}"),
            "FLOAT",
            json!([1, HEADS, source_tokens, DIM]),
        ));
        inputs.push(tensor(
            format!("record_offset_{suffix}"),
            "INT64",
            json!([1]),
        ));
    }
    inputs.push(tensor("record_stop".into(), "INT64", json!([1])));
    let outputs = vec![
        tensor(
            "memory_key".into(),
            "FLOAT",
            json!([1, HEADS, "memory_tokens", DIM]),
        ),
        tensor(
            "memory_value".into(),
            "FLOAT",
            json!([1, HEADS, "memory_tokens", DIM]),
        ),
        tensor("finite_output".into(), "BOOL", json!([])),
    ];
    json!({
        "schema":DEVICE_PACKING_SCHEMA,"artifact_domain":DEVICE_PACKING_DOMAIN,"layout_revision":1,
        "graph":{"file":DEVICE_PACKING_GRAPH_FILE,"bytes":graph.bytes,"sha256":hex(graph.sha256),
            "ir_version":10,"opset":17,"inputs":inputs,"outputs":outputs},
        "learned_weights":{"count":0,"bytes":0},
        "integer_control_initializers":{"count":5,"payload_bytes":40},
        "bounds":{"batch":1,"precision":"FP32","board_tokens":BOARD,"kv_heads":HEADS,
            "head_dimension":DIM,"record_capacity":RECORDS,"source_tokens_minimum":67,
            "source_tokens_maximum":TOKENS,"record_count_minimum":0,"record_count_maximum":RECORDS,
            "record_offset_minimum":66,"record_offset_maximum":193,
            "record_stop":"66 + max(record_count, 1)"},
        "routing":{"base":"exactly_one_actual_matching_board_base",
            "records":"ordered_occurrences_duplicates_preserved",
            "inactive_ports":"alias_matching_base_token_66",
            "zero_records":"actual_zero_feature_projection_at_token_66_with_false_mask",
            "mask":"CPU_owned_current_mask_first_66_true_zero_pad_false",
            "finite_output":"joined_visible_key_and_value_only"},
        "intermediate_arithmetic":declared_intermediate_arithmetic(),
        "scope":{"artifact_and_cpu_numeric_only":true,"native_connected":false,
            "cuda_resident_owner":"not_implemented",
            "cuda_physical_completion_and_quarantine":"not_implemented",
            "cuda_validation":"not_run_user_deferred","private_warm":"outside_this_artifact",
            "cpu_projection_origin":"caller_identity_and_actual_feature_bits_not_native_producer_attestation",
            "cpu_owner_accounting":"unique_owned_numpy_payloads_only_not_peak",
            "device_runtime_bytes":"unknown"}
    })
}
fn declared_intermediate_arithmetic() -> Value {
    let mut ledger = Vec::with_capacity(NODE_OUTPUTS);
    let mut append = |name: String, dtype: &str, shape: Vec<usize>, width: u64| {
        let bytes = shape
            .iter()
            .fold(width, |n, dimension| n * *dimension as u64);
        ledger.push(json!({"node":name,"output":name,"dtype":dtype,
            "maximum_shape":shape,"maximum_payload_bytes":bytes}));
    };
    for kind in ["key", "value"] {
        append(
            format!("board_slice_{kind}"),
            "FLOAT",
            vec![1, HEADS, BOARD, DIM],
            4,
        );
        for index in 0..RECORDS {
            append(
                format!("gather_{kind}_{index:03}"),
                "FLOAT",
                vec![1, HEADS, 1, DIM],
                4,
            );
        }
        for stem in ["all_ports", "memory"] {
            append(
                format!("{stem}_{kind}"),
                "FLOAT",
                vec![1, HEADS, TOKENS, DIM],
                4,
            );
        }
        for stem in ["nan", "inf", "bad"] {
            append(
                format!("{stem}_{kind}"),
                "BOOL",
                vec![1, HEADS, TOKENS, DIM],
                1,
            );
        }
        append(
            format!("bad_int_{kind}"),
            "INT64",
            vec![1, HEADS, TOKENS, DIM],
            8,
        );
        append(format!("bad_max_{kind}"), "INT64", vec![], 8);
        append(format!("finite_{kind}"), "BOOL", vec![], 1);
    }
    append("finite_output".into(), "BOOL", vec![], 1);
    json!({"meaning":"sum_of_independent_maximum_node_output_payloads_not_peak",
        "node_output_count":NODE_OUTPUTS,"node_outputs":ledger,
        "maximum_sum_payload_bytes":DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM,
        "allocator_workspace_and_session_bytes":"unknown",
        "physical_stage_completion":"not_implemented"})
}

// serde_json::Value alone accepts duplicate object keys. This recursive visitor
// rejects them before exact declaration comparison, including nested graph and
// node ledger objects. serde_json's default recursion bound remains enabled.
struct StrictJson(Value);
impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}
struct StrictJsonVisitor;
impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bounded JSON without duplicate object keys")
    }
    fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Bool(value)))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Number(value.into())))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Number(value.into())))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Self::Value, E> {
        Number::from_f64(value)
            .map(|n| StrictJson(Value::Number(n)))
            .ok_or_else(|| E::custom("nonfinite JSON number"))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::String(value.into())))
    }
    fn visit_string<E: de::Error>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::String(value)))
    }
    fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Null))
    }
    fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Null))
    }
    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut sequence: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJson>()? {
            values.push(value.0);
        }
        Ok(StrictJson(Value::Array(values)))
    }
    fn visit_map<A: MapAccess<'de>>(
        self,
        mut object: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut values = Map::new();
        while let Some((key, value)) = object.next_entry::<String, StrictJson>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom("duplicate JSON key"));
            }
            values.insert(key, value.0);
        }
        Ok(StrictJson(Value::Object(values)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pin(bytes: &[u8]) -> RegisteredPackingBytePin {
        RegisteredPackingBytePin {
            bytes: bytes.len() as u64,
            sha256: Sha256::digest(bytes).into(),
        }
    }
    // Deliberately opaque, non-ONNX bytes: these source fixtures prove only byte
    // identity and declarations, never native/weight-free graph execution.
    fn fixture() -> (Vec<u8>, Vec<u8>, PackingArtifactRegistration) {
        let graph = b"source-fixture-opaque-graph-body-unverified".to_vec();
        let manifest = serde_json::to_vec(&declared_manifest(pin(&graph))).unwrap();
        let registration = PackingArtifactRegistration {
            manifest: pin(&manifest),
            graph: pin(&graph),
        };
        (manifest, graph, registration)
    }
    fn declaration() -> DevicePackingResourceDeclaration {
        DevicePackingResourceDeclaration {
            packing_session_bytes: Some(1024),
            additional_owner_metadata_host_bytes: Some(256),
            max_owned_artifact_host_bytes: 3 * 1024 * 1024,
            max_declared_host_bytes: 4 * 1024 * 1024,
            max_declared_device_bytes: 2 * 1024 * 1024,
        }
    }
    fn changed_manifest(value: Value, graph: &[u8]) -> (Vec<u8>, PackingArtifactRegistration) {
        let manifest = serde_json::to_vec(&value).unwrap();
        let registration = PackingArtifactRegistration {
            manifest: pin(&manifest),
            graph: pin(graph),
        };
        (manifest, registration)
    }
    #[test]
    fn opaque_registered_bytes_never_mint_native_verification_and_are_owned() {
        let (mut manifest, mut graph, registration) = fixture();
        let admitted =
            RegisteredPackingArtifactBytes::admit(&manifest, &graph, registration, declaration())
                .unwrap();
        assert_eq!(
            admitted.verification_scope(),
            PackingVerificationScope::RegisteredByteIdentityAndExactManifestDeclarationOnly
        );
        assert_eq!(
            admitted.native_verification(),
            PackingNativeVerification::NotPerformed
        );
        assert_eq!(admitted.registration(), registration);
        manifest.fill(0);
        graph.fill(0);
        assert_eq!(pin(admitted.manifest_bytes()), registration.manifest);
        assert_eq!(pin(admitted.graph_bytes()), registration.graph);
        assert_eq!(
            admitted.resources().maximum_node_output_payload_sum,
            1_142_291
        );
    }
    #[test]
    fn actual_byte_and_length_pins_reject_drift_before_declaration_admission() {
        let (manifest, mut graph, registration) = fixture();
        graph[0] ^= 1;
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(&manifest, &graph, registration, declaration()),
            Err(DevicePackingAdmissionError::ByteIdentityMismatch(
                PackingArtifactPart::Graph
            ))
        ));
        let mut bad = registration;
        bad.manifest.bytes -= 1;
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(&manifest, &graph, bad, declaration()),
            Err(DevicePackingAdmissionError::ByteIdentityMismatch(
                PackingArtifactPart::Manifest
            ))
        ));
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(&[], &graph, registration, declaration()),
            Err(DevicePackingAdmissionError::ArtifactBounds(
                PackingArtifactPart::Manifest
            ))
        ));
        let oversized = vec![0; DEVICE_PACKING_MAX_GRAPH_BYTES + 1];
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(
                &manifest,
                &oversized,
                registration,
                declaration()
            ),
            Err(DevicePackingAdmissionError::ArtifactBounds(
                PackingArtifactPart::Graph
            ))
        ));
    }
    #[test]
    fn hash_updated_manifest_cannot_change_port_order_shape_scope_or_node_arithmetic() {
        let (_, graph, _) = fixture();
        let pristine = declared_manifest(pin(&graph));
        let mut alternatives = Vec::new();
        let mut value = pristine.clone();
        value["graph"]["inputs"].as_array_mut().unwrap().swap(2, 5);
        alternatives.push(value);
        let mut value = pristine.clone();
        value["graph"]["outputs"][0]["shape"][1] = json!(6);
        alternatives.push(value);
        let mut value = pristine.clone();
        value["scope"]["native_connected"] = json!(true);
        alternatives.push(value);
        let mut value = pristine.clone();
        value["intermediate_arithmetic"]["node_outputs"][0]["maximum_payload_bytes"] = json!(1);
        alternatives.push(value);
        let mut value = pristine.clone();
        value["bounds"]["batch"] = json!(1.0);
        alternatives.push(value);
        let mut value = pristine.clone();
        value["graph"]["opset"] = json!(18);
        alternatives.push(value);
        let mut value = pristine.clone();
        value["graph"]["ir_version"] = json!(9);
        alternatives.push(value);
        let mut value = pristine.clone();
        value["learned_weights"]["count"] = json!(1);
        alternatives.push(value);
        let mut value = pristine;
        value["extra"] = json!(true);
        alternatives.push(value);
        for value in alternatives {
            let (manifest, registration) = changed_manifest(value, &graph);
            assert!(matches!(
                RegisteredPackingArtifactBytes::admit(
                    &manifest,
                    &graph,
                    registration,
                    declaration()
                ),
                Err(DevicePackingAdmissionError::ManifestDeclarationMismatch)
            ));
        }
    }
    #[test]
    fn nested_duplicate_json_and_nonfinite_or_trailing_json_are_rejected() {
        let (manifest, graph, _) = fixture();
        let original = String::from_utf8(manifest).unwrap();
        let duplicate = original.replacen("\"batch\":1", "\"batch\":1,\"batch\":1", 1);
        assert_ne!(duplicate, original);
        for malformed in [
            duplicate,
            original.replacen("\"batch\":1", "\"batch\":NaN", 1),
            format!("{original} true"),
        ] {
            let manifest = malformed.into_bytes();
            let registration = PackingArtifactRegistration {
                manifest: pin(&manifest),
                graph: pin(&graph),
            };
            assert!(matches!(
                RegisteredPackingArtifactBytes::admit(
                    &manifest,
                    &graph,
                    registration,
                    declaration()
                ),
                Err(DevicePackingAdmissionError::ManifestJson)
            ));
        }
    }
    #[test]
    fn missing_resources_overflow_and_known_payload_limits_never_silently_shrink() {
        let (manifest, graph, registration) = fixture();
        let mut unknown = declaration();
        unknown.packing_session_bytes = None;
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(&manifest, &graph, registration, unknown),
            Err(DevicePackingAdmissionError::UnknownResource(
                "packing_session_bytes"
            ))
        ));
        unknown = declaration();
        unknown.additional_owner_metadata_host_bytes = Some(0);
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(&manifest, &graph, registration, unknown),
            Err(DevicePackingAdmissionError::UnknownResource(
                "additional_owner_metadata_host_bytes"
            ))
        ));
        let mut overflow = declaration();
        overflow.packing_session_bytes = Some(u64::MAX);
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(&manifest, &graph, registration, overflow),
            Err(DevicePackingAdmissionError::Overflow)
        ));
        let mut small = declaration();
        small.max_declared_device_bytes = DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM;
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(&manifest, &graph, registration, small),
            Err(DevicePackingAdmissionError::ResourceExceeded(
                "declared_device_bytes"
            ))
        ));
        small = declaration();
        small.max_owned_artifact_host_bytes = (manifest.len() + graph.len()) as u64;
        assert!(matches!(
            RegisteredPackingArtifactBytes::admit(&manifest, &graph, registration, small),
            Err(DevicePackingAdmissionError::ResourceExceeded(
                "owned_artifact_host_bytes"
            ))
        ));
    }
    #[test]
    fn fixed_declaration_ledger_has_exact_ports_and_non_peak_payload_meaning() {
        let (_, graph, _) = fixture();
        let value = declared_manifest(pin(&graph));
        let inputs = value["graph"]["inputs"].as_array().unwrap();
        assert_eq!(inputs.len(), 387);
        assert_eq!(inputs[383]["name"], "record_key_127");
        assert_eq!(inputs[385]["name"], "record_offset_127");
        assert_eq!(inputs[386]["name"], "record_stop");
        let arithmetic = &value["intermediate_arithmetic"];
        let outputs = arithmetic["node_outputs"].as_array().unwrap();
        assert_eq!(outputs.len(), 275);
        let sum: u64 = outputs
            .iter()
            .map(|row| row["maximum_payload_bytes"].as_u64().unwrap())
            .sum();
        assert_eq!(sum, DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM);
        assert_eq!(arithmetic["physical_stage_completion"], "not_implemented");
        assert_eq!(
            arithmetic["allocator_workspace_and_session_bytes"],
            "unknown"
        );
    }
}
