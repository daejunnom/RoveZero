//! Pure pre-child admission for the registered FullLineInteractionV2 owner.
//!
//! Historical artifact schema and layout remain historical identities. This
//! API admits only the P/C capability of the existing native validator; it
//! never grants V Collection capability, reads assets, or initializes ORT.
use super::super::{fail, validate_manifest, ExportManifest, Layout, MAX_MANIFEST_BYTES};
use super::VirtualReviewExecutionMode;
use crate::asset::parse_sha256;
use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use crate::pals_model::{PalsModelConfig, PalsModelProfile};
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LegacyPalsGraphPin {
    pub role: String,
    pub file: String,
    /// A declaration only: preflight does not read or verify the graph bytes.
    pub sha256: [u8; 32],
    pub opset: u32,
}

/// Validation of the actual bounded manifest bytes and legacy declarations.
/// Graph/checkpoint pins are declarations, not asset or provider acceptance.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LegacyPalsExportManifestSummary {
    manifest_sha256: [u8; 32],
    manifest_bytes: usize,
    /// Original artifact schema; a V2 model can have a V1 separate-PC artifact.
    pub schema: String,
    pub config: PalsModelConfig,
    pub model_semantics: String,
    pub encoding_schema: String,
    pub model_profile: PalsModelProfile,
    pub checkpoint_sha256: [u8; 32],
    pub roles: Vec<String>,
    pub validator_present: bool,
    /// Resolved native layout. `declared_layout` preserves an omitted field.
    pub layout: String,
    pub declared_layout: Option<String>,
    pub layout_revision: Option<u32>,
    pub graphs: Vec<LegacyPalsGraphPin>,
    pub trained: bool,
    pub training_steps: u64,
    /// Separate-PC historical artifacts may omit these declarations.
    pub rules_input_profile: Option<String>,
    pub rules_input_semantic_sha256: Option<[u8; 32]>,
    pub rules_encoder_source_sha256: Option<[u8; 32]>,
    pub rules_profile_descriptor_sha256: Option<[u8; 32]>,
    pub rules_profile_canonical_sha256: Option<[u8; 32]>,
}
impl LegacyPalsExportManifestSummary {
    pub fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }
    pub fn manifest_bytes(&self) -> usize {
        self.manifest_bytes
    }
}

/// Reuses the legacy native manifest/config/interface validator exactly.
///
/// `Collection` does not expand the existing P/C-only legacy capability. The
/// actual V3 Collection artifact has its own explicit admission API. Optional
/// Rules declarations are not synthesized; callers requiring them must reject
/// `None` under their registered startup policy.
pub fn validate_pals_legacy_export_manifest(
    bytes: &[u8],
    mode: VirtualReviewExecutionMode,
) -> Result<LegacyPalsExportManifestSummary, BackendError> {
    if bytes.is_empty() || bytes.len() > MAX_MANIFEST_BYTES {
        return Err(fail(
            K::ResourceExhausted,
            S::Asset,
            "PALS legacy manifest exceeds bounded admission",
        ));
    }
    // ExportManifest has deny_unknown_fields at each typed declaration. Its
    // opaque evidence Value also needs recursive duplicate-key rejection.
    let unique: UniqueJson = serde_json::from_slice(bytes).map_err(decode_error)?;
    if !unique.0.is_object() {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "PALS legacy manifest must be an object",
        ));
    }
    let manifest: ExportManifest = serde_json::from_value(unique.0).map_err(decode_error)?;
    if manifest.validator_present
        || manifest.roles.iter().any(|role| role == "validator")
        || manifest
            .graphs
            .iter()
            .any(|graph| graph.role == "validator")
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Admission,
            match mode {
                VirtualReviewExecutionMode::Deployment => {
                    "legacy Deployment preflight rejects a V-containing manifest"
                }
                VirtualReviewExecutionMode::Collection => {
                    "legacy native Collection has no V execution capability"
                }
            },
        ));
    }
    if manifest.config.profile != PalsModelProfile::FullLineInteractionV2 {
        return Err(fail(
            K::UnsupportedModel,
            S::Admission,
            "legacy preflight supports only registered FullLineInteractionV2",
        ));
    }
    let checkpoint_sha256 = validate_manifest(&manifest)?;
    let layout = match manifest.resolved_layout()? {
        Layout::SeparatePc => "separate_pc",
        Layout::SharedPcIf => "shared_pc_if",
    }
    .to_owned();
    // Native validation requires these identities for every nonlegacy model
    // profile. Keep errors typed rather than treating its invariant as a panic.
    let required_identity = |value: Option<String>| {
        value.ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Asset,
                "FullLineInteractionV2 model identity is missing",
            )
        })
    };
    let model_profile = manifest.model_profile.ok_or_else(|| {
        fail(
            K::IdentityMismatch,
            S::Asset,
            "FullLineInteractionV2 profile identity is missing",
        )
    })?;
    let optional_pin = |value: Option<String>| value.as_deref().map(parse_sha256).transpose();
    let graphs = manifest
        .graphs
        .into_iter()
        .map(|graph| {
            Ok(LegacyPalsGraphPin {
                role: graph.role,
                file: graph.file,
                sha256: parse_sha256(&graph.sha256)?,
                opset: graph.opset,
            })
        })
        .collect::<Result<Vec<_>, BackendError>>()?;
    Ok(LegacyPalsExportManifestSummary {
        manifest_sha256: crate::asset::sha256(bytes),
        manifest_bytes: bytes.len(),
        schema: manifest.schema,
        config: manifest.config,
        model_semantics: required_identity(manifest.model_semantics)?,
        encoding_schema: required_identity(manifest.encoding_schema)?,
        model_profile,
        checkpoint_sha256,
        roles: manifest.roles,
        validator_present: manifest.validator_present,
        layout,
        declared_layout: manifest.layout,
        layout_revision: manifest.layout_revision,
        graphs,
        trained: manifest.trained,
        training_steps: manifest.training_steps,
        rules_input_profile: manifest.rules_input_profile,
        rules_input_semantic_sha256: optional_pin(manifest.rules_input_semantic_sha256)?,
        rules_encoder_source_sha256: optional_pin(manifest.rules_encoder_source_sha256)?,
        rules_profile_descriptor_sha256: optional_pin(manifest.rules_profile_descriptor_sha256)?,
        rules_profile_canonical_sha256: optional_pin(manifest.rules_profile_canonical_sha256)?,
    })
}

fn decode_error(error: serde_json::Error) -> BackendError {
    fail(
        K::UnsupportedModel,
        S::Asset,
        "cannot decode strict legacy PALS export manifest",
    )
    .with_external_cause(CauseCode::ModelLoad, &error)
}

/// Value-like decoder that refuses duplicate keys at every JSON depth. Native
/// authority remains the existing typed ExportManifest, not this intermediate.
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
                Ok(UniqueJson(Value::String(v.to_owned())))
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
                while let Some(value) = seq.next_element::<UniqueJson>()? {
                    values.push(value.0);
                }
                Ok(UniqueJson(Value::Array(values)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate JSON key"));
                    }
                    values.insert(key, map.next_value::<UniqueJson>()?.0);
                }
                Ok(UniqueJson(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        memory_outputs, public_inputs_for, role_inputs_for, role_outputs, shared_inputs_for,
        shared_outputs, Interface,
    };
    use super::*;
    use crate::pals_model::{PALS_MODEL_SCHEMA, V_TASK_NAMES};
    use ort::tensor::TensorElementType;
    use serde_json::json;

    fn tensor(interface: Interface) -> Value {
        json!({
            "name":interface.name,
            "dtype":match interface.dtype {
                TensorElementType::Float32=>"FLOAT",
                TensorElementType::Int64=>"INT64",
                TensorElementType::Bool=>"BOOL",
                _=>unreachable!(),
            },
            "shape":interface.shape.iter().enumerate().map(|(axis,dimension)|{
                if *dimension == -1 {json!(format!("dynamic_{axis}"))}else{json!(dimension)}
            }).collect::<Vec<_>>()
        })
    }
    fn fixture(shared: bool) -> Value {
        let config = PalsModelConfig::full_line_interaction_v2();
        let profile = config.profile;
        let graph_roles: &[&str] = if shared {
            &["public", "shared_pc"]
        } else {
            &["public", "proposer", "critic"]
        };
        let graphs = graph_roles.iter().map(|role| {
            let inputs=if *role=="public"{public_inputs_for(profile)}else if *role=="shared_pc"{
                shared_inputs_for(profile)
            }else{role_inputs_for(*role=="critic",profile)};
            let outputs=if *role=="public"{memory_outputs()}else if *role=="shared_pc"{
                shared_outputs()
            }else{role_outputs(*role=="critic")};
            json!({"role":role,"file":match *role{"public"=>"public_memory.onnx".to_owned(),"shared_pc"=>"shared_pc_if.onnx".to_owned(),_=>format!("role_{role}.onnx")},
                "sha256":"01".repeat(32),"opset":17,"inputs":inputs.into_iter().map(tensor).collect::<Vec<_>>(),
                "outputs":outputs.into_iter().map(tensor).collect::<Vec<_>>()})
        }).collect::<Vec<_>>();
        let mut value = json!({
            "schema":if shared{"rovezero.pals-model.v2"}else{PALS_MODEL_SCHEMA},
            "config":config,"model_profile":profile,"model_semantics":profile.model_semantics(),
            "encoding_schema":profile.encoding_schema(),"checkpoint_sha256":"02".repeat(32),
            "trained":false,"training_steps":0,"roles":["proposer","critic"],"validator_present":false,
            "graphs":graphs,"runtime":{"onnxruntime":"1.22.0","rust_ort":"2.0.0-rc.10","compatibility":"requires_actual_numeric_check"},
            "candidate_promotion":{"0":"none","1":"queen","2":"rook","3":"bishop","4":"knight"},
            "wdl_perspective":"input_side_to_move","task_names":V_TASK_NAMES,"numeric_status":"not_run","cuda_status":"not_run"
        });
        if shared {
            let branch_operations = (0..6)
                .flat_map(|route| {
                    ["then_branch","else_branch"].map(move |branch|{
                json!({"route":format!("route_{route}"),"branch":branch,"nodes":1})
            })
                })
                .collect::<Vec<_>>();
            let object = value.as_object_mut().unwrap();
            object.insert("layout".into(), json!("shared_pc_if"));
            object.insert("layout_revision".into(), json!(1));
            object.insert(
                "role_batching".into(),
                json!("one_scalar_role_per_physical_batch"),
            );
            object.insert(
                "rules_input_profile".into(),
                json!("rz-pals-rules-full-line-v2"),
            );
            object.insert("rules_input_semantic_sha256".into(), json!("03".repeat(32)));
            object.insert("rules_encoder_source_sha256".into(), json!("04".repeat(32)));
            object.insert("reader_initializer_bank".into(),json!({
                "scope":"onnx_serialized_initializers_only","layout_revision":1,
                "shared_parameters":[{"parameter":"reader.weight","initializer":"reader_weight","bytes":4,
                    "sha256":"05".repeat(32),"source_parameter_sha256":"05".repeat(32),"shape":[1]}],
                "unique_shared_parameters":1,"shared_weight_bytes":4,"if_routes":6,
                "branch_local_initializers":0,"branch_operations":branch_operations,
                "inactive_private_execution":"requires_runtime_profile","ort_prepack_copies":"unknown","device_residency_sharing":"unknown"
            }));
        }
        value
    }
    fn bytes(value: &Value) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }
    #[test]
    fn full_line_separate_retains_actual_schema_bytes_and_missing_rules() {
        let encoded = bytes(&fixture(false));
        let summary =
            validate_pals_legacy_export_manifest(&encoded, VirtualReviewExecutionMode::Deployment)
                .unwrap();
        assert_eq!(summary.schema, PALS_MODEL_SCHEMA);
        assert_eq!(
            summary.model_profile,
            PalsModelProfile::FullLineInteractionV2
        );
        assert_eq!(summary.layout, "separate_pc");
        assert_eq!(summary.declared_layout, None);
        assert_eq!(summary.rules_input_semantic_sha256, None);
        assert_eq!(summary.rules_encoder_source_sha256, None);
        assert_eq!(summary.manifest_sha256(), crate::asset::sha256(&encoded));
        assert_eq!(summary.manifest_bytes(), encoded.len());
        assert_eq!(summary.graphs.len(), 3);
        assert_eq!(summary.graphs[0].sha256, [1; 32]);
        let mut whitespace = encoded;
        whitespace.push(b'\n');
        let another = validate_pals_legacy_export_manifest(
            &whitespace,
            VirtualReviewExecutionMode::Collection,
        )
        .unwrap();
        assert_ne!(summary.manifest_sha256(), another.manifest_sha256());
        assert_eq!(summary.checkpoint_sha256, another.checkpoint_sha256);
    }
    #[test]
    fn shared_retains_legacy_layout_and_reuses_source_scalar_ownership_gates() {
        let value = fixture(true);
        let summary = validate_pals_legacy_export_manifest(
            &bytes(&value),
            VirtualReviewExecutionMode::Deployment,
        )
        .unwrap();
        assert_eq!(summary.schema, "rovezero.pals-model.v2");
        assert_eq!(summary.layout, "shared_pc_if");
        assert_eq!(summary.rules_input_semantic_sha256, Some([3; 32]));
        assert_eq!(summary.rules_encoder_source_sha256, Some([4; 32]));
        for case in 0..3 {
            let mut broken = value.clone();
            match case {
                0 => {
                    broken
                        .as_object_mut()
                        .unwrap()
                        .remove("rules_encoder_source_sha256");
                }
                1 => broken["graphs"][1]["inputs"][0]["shape"] = json!([1]),
                _ => broken["reader_initializer_bank"]["shared_weight_bytes"] = json!(8),
            }
            assert!(validate_pals_legacy_export_manifest(
                &bytes(&broken),
                VirtualReviewExecutionMode::Deployment,
            )
            .is_err());
        }
    }
    #[test]
    fn neither_execution_mode_grants_legacy_v_capability() {
        for case in 0..3 {
            let mut value = fixture(false);
            match case {
                0 => value["validator_present"] = json!(true),
                1 => value["roles"] = json!(["proposer", "critic", "validator"]),
                _ => value["graphs"][1]["role"] = json!("validator"),
            }
            for mode in [
                VirtualReviewExecutionMode::Deployment,
                VirtualReviewExecutionMode::Collection,
            ] {
                let failure =
                    validate_pals_legacy_export_manifest(&bytes(&value), mode).unwrap_err();
                assert_eq!(failure.kind, K::UnsupportedModel);
                assert_eq!(failure.stage, S::Admission);
            }
        }
    }
    #[test]
    fn unknown_duplicate_and_oversized_json_never_enter_native_admission() {
        let value = fixture(false);
        for case in 0..3 {
            let mut broken = value.clone();
            match case {
                0 => broken["unexpected"] = json!(1),
                1 => broken["config"]["unexpected"] = json!(1),
                _ => broken["graphs"][0]["inputs"][0]["unexpected"] = json!(1),
            }
            assert!(validate_pals_legacy_export_manifest(
                &bytes(&broken),
                VirtualReviewExecutionMode::Deployment
            )
            .is_err());
        }
        let text = String::from_utf8(bytes(&value)).unwrap();
        let duplicate = text.replacen(
            "\"trained\":false",
            "\"trained\":false,\"trained\":false",
            1,
        );
        assert_ne!(duplicate, text);
        assert!(validate_pals_legacy_export_manifest(
            duplicate.as_bytes(),
            VirtualReviewExecutionMode::Deployment
        )
        .is_err());
        let mut object = value;
        object["rules_input_declaration"] = json!({"opaque":{"same":1}});
        let text = String::from_utf8(bytes(&object)).unwrap();
        let duplicate = text.replacen("\"same\":1", "\"same\":1,\"same\":2", 1);
        let failure = validate_pals_legacy_export_manifest(
            duplicate.as_bytes(),
            VirtualReviewExecutionMode::Deployment,
        )
        .unwrap_err();
        assert_eq!(
            failure.detail,
            "cannot decode strict legacy PALS export manifest"
        );
        assert!(validate_pals_legacy_export_manifest(
            &vec![b' '; MAX_MANIFEST_BYTES + 1],
            VirtualReviewExecutionMode::Deployment
        )
        .is_err());
    }
    #[test]
    fn profile_schema_path_identity_and_training_corruption_remain_rejected() {
        for case in 0..7 {
            let mut broken = fixture(false);
            match case {
                0 => broken["schema"] = json!("rovezero.pals-model.v3"),
                1 => broken["config"]["profile"] = json!("virtual_review_v1"),
                2 => broken["graphs"][0]["file"] = json!("../public_memory.onnx"),
                3 => broken["graphs"][1]["role"] = json!("critic"),
                4 => broken["model_semantics"] = json!("rovezero.pals-model-semantics.v3"),
                5 => broken["checkpoint_sha256"] = json!("not-a-digest"),
                _ => broken["training_steps"] = json!(1),
            }
            assert!(validate_pals_legacy_export_manifest(
                &bytes(&broken),
                VirtualReviewExecutionMode::Deployment
            )
            .is_err());
        }
    }
}
