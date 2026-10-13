//! Bounded, pure V1/V2 Rules declaration join. This reads caller-supplied bytes
//! only; a matching declaration proves neither learned compatibility nor a Run.
use rz_eval::error::{BackendError, FailureKind, FailureStage};
use rz_eval::pals_model::{PalsModelConfig, PalsModelProfile};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyRulesDescriptorSummary {
    descriptor_sha256: [u8; 32],
    pub config: PalsModelConfig,
    pub semantic_sha256: [u8; 32],
    pub encoder_source_sha256: [u8; 32],
}
impl LegacyRulesDescriptorSummary {
    pub fn descriptor_sha256(&self) -> [u8; 32] {
        self.descriptor_sha256
    }
}

// Absence and a declared field are different: V1 has neither extension, while
// V2 requires both. In particular JSON null cannot impersonate an absent field.
#[derive(Default)]
enum Declared<T> {
    #[default]
    Absent,
    Present(T),
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Declared<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Self::Present)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyRulesDescriptorDocument {
    schema: String,
    rules_input_profile: String,
    rules_input_semantic_sha256: String,
    rules_encoder_source_sha256: String,
    encoder_source: String,
    semantic_digest_algorithm: String,
    rules_base_semantics: String,
    semantic_fields: Vec<String>,
    learned_input_compatibility: String,
    metadata_features: Vec<String>,
    record_features: Vec<String>,
    query_features: Vec<String>,
    divergence_features: Vec<String>,
    #[serde(default)]
    full_line_features: Declared<Vec<String>>,
    #[serde(default)]
    bounds: Declared<LegacyRulesDescriptorBounds>,
    config: PalsModelConfig,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyRulesDescriptorBounds {
    max_records: u16,
    max_line_plies: u16,
    max_candidates: u16,
}

fn fail(detail: &'static str) -> BackendError {
    BackendError::new(
        FailureKind::IdentityMismatch,
        FailureStage::Admission,
        detail,
    )
}
fn hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Join the exact existing descriptor schema, feature order and source pin to
/// its registered config. Hash the original bounded bytes without rewriting,
/// normalization or a V3 conversion. No model/runtime/file access occurs here.
pub fn validate_pals_legacy_rules_descriptor_bytes(
    bytes: &[u8],
    config: &PalsModelConfig,
) -> Result<LegacyRulesDescriptorSummary, BackendError> {
    if bytes.is_empty() || bytes.len() > 64 * 1024 {
        return Err(fail("V1/V2 Rules descriptor exceeds bounded bytes"));
    }
    if config.profile == PalsModelProfile::VirtualReviewV1 || config.validate().is_err() {
        return Err(fail(
            "V1/V2 Rules descriptor requires a registered legacy config",
        ));
    }
    // Typed derives reject duplicate known fields as well as unknown fields,
    // including nested config and bounds. No Value/Map projection hides them.
    let document: LegacyRulesDescriptorDocument = serde_json::from_slice(bytes)
        .map_err(|_| fail("cannot decode strict V1/V2 Rules descriptor"))?;
    let semantic_sha256 = super::pals_rules_encoding_semantic_digest_for_config(config);
    let encoder_source_sha256 = super::pals_native_source_digest();
    let full_line = config.profile.uses_full_line();
    let extensions_match = match (&document.full_line_features, &document.bounds, full_line) {
        (Declared::Absent, Declared::Absent, false) => true,
        (Declared::Present(features), Declared::Present(bounds), true) => {
            features
                .iter()
                .map(String::as_str)
                .eq(super::PALS_FULL_LINE_FEATURES)
                && bounds.max_records == 128
                && bounds.max_line_plies == 256
                && bounds.max_candidates == 256
        }
        _ => false,
    };
    let expected_schema = if full_line {
        "rovezero.pals-rules-descriptor.v2"
    } else {
        "rovezero.pals-rules-descriptor.v1"
    };
    if document.schema != expected_schema
        || document.rules_input_profile != super::pals_rules_profile_for_config(config)
        || document.rules_input_semantic_sha256 != hex(semantic_sha256)
        || document.rules_encoder_source_sha256 != hex(encoder_source_sha256)
        || document.encoder_source != "crates/rz-uci/src/pals_native.rs"
        || document.semantic_digest_algorithm != "sha256_u64le_length_prefixed_utf8_fields"
        || document.rules_base_semantics != super::PALS_RULES_BASE_SEMANTICS
        || document.semantic_fields
            != super::pals_rules_semantic_fields_for_config(config).collect::<Vec<_>>()
        || document.learned_input_compatibility != "unverified_declaration_only"
        || document.metadata_features != super::PALS_METADATA_FEATURES
        || document.record_features != super::pals_record_features_for_config(config)
        || document.query_features != super::pals_query_features_for_config(config)
        || document.divergence_features != super::PALS_DIVERGENCE_FEATURES
        || document.config != *config
        || !extensions_match
    {
        return Err(fail(
            "V1/V2 Rules descriptor differs from the compiled source declaration",
        ));
    }
    Ok(LegacyRulesDescriptorSummary {
        descriptor_sha256: Sha256::digest(bytes).into(),
        config: document.config,
        semantic_sha256,
        encoder_source_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The exact fields emitted by examples/pals_rules_descriptor.rs. Source
    // digests come from the compiled source getter, never a saved asset file.
    fn declaration(config: &PalsModelConfig) -> serde_json::Value {
        let full_line = config.profile.uses_full_line();
        let mut value = serde_json::json!({
            "schema": if full_line {"rovezero.pals-rules-descriptor.v2"} else {"rovezero.pals-rules-descriptor.v1"},
            "rules_input_profile":super::super::pals_rules_profile_for_config(config),
            "rules_input_semantic_sha256":hex(super::super::pals_rules_encoding_semantic_digest_for_config(config)),
            "rules_encoder_source_sha256":hex(super::super::pals_native_source_digest()),
            "encoder_source":"crates/rz-uci/src/pals_native.rs",
            "semantic_digest_algorithm":"sha256_u64le_length_prefixed_utf8_fields",
            "rules_base_semantics":super::super::PALS_RULES_BASE_SEMANTICS,
            "semantic_fields":super::super::pals_rules_semantic_fields_for_config(config).collect::<Vec<_>>(),
            "learned_input_compatibility":"unverified_declaration_only",
            "metadata_features":super::super::PALS_METADATA_FEATURES,
            "record_features":super::super::pals_record_features_for_config(config),
            "query_features":super::super::pals_query_features_for_config(config),
            "divergence_features":super::super::PALS_DIVERGENCE_FEATURES,
            "config":config,
        });
        if full_line {
            value["full_line_features"] = serde_json::json!(super::super::PALS_FULL_LINE_FEATURES);
            value["bounds"] =
                serde_json::json!({"max_records":128,"max_line_plies":256,"max_candidates":256});
        }
        value
    }
    fn accepted(value: &serde_json::Value, config: &PalsModelConfig) -> bool {
        validate_pals_legacy_rules_descriptor_bytes(&serde_json::to_vec(value).unwrap(), config)
            .is_ok()
    }
    #[test]
    fn registered_v1_and_v2_profiles_preserve_actual_bytes_and_compiled_pins() {
        for profile in [
            PalsModelProfile::LegacySummaryV1,
            PalsModelProfile::FullLineV2,
            PalsModelProfile::InteractionHeadV2,
            PalsModelProfile::FullLineInteractionV2,
        ] {
            let config = PalsModelConfig::for_profile(profile);
            let bytes = serde_json::to_vec(&declaration(&config)).unwrap();
            let result = validate_pals_legacy_rules_descriptor_bytes(&bytes, &config).unwrap();
            assert_eq!(result.config, config);
            assert_eq!(
                result.descriptor_sha256(),
                <[u8; 32]>::from(Sha256::digest(&bytes))
            );
            assert_eq!(
                result.semantic_sha256,
                super::super::pals_rules_encoding_semantic_digest_for_config(&config)
            );
            assert_eq!(
                result.encoder_source_sha256,
                super::super::pals_native_source_digest()
            );
            let mut whitespace = bytes.clone();
            whitespace.push(b'\n');
            let alternate =
                validate_pals_legacy_rules_descriptor_bytes(&whitespace, &config).unwrap();
            assert_ne!(alternate.descriptor_sha256(), result.descriptor_sha256());
            assert_eq!(alternate.semantic_sha256, result.semantic_sha256);
        }
    }
    #[test]
    fn wrong_or_missing_pins_unknown_and_duplicate_fields_are_rejected() {
        let config = PalsModelConfig::full_line_interaction_v2();
        let original = declaration(&config);
        for pin in ["rules_encoder_source_sha256", "rules_input_semantic_sha256"] {
            let mut changed = original.clone();
            changed[pin] = serde_json::json!("00".repeat(32));
            assert!(!accepted(&changed, &config));
            changed.as_object_mut().unwrap().remove(pin);
            assert!(!accepted(&changed, &config));
        }
        for section in [None, Some("config"), Some("bounds")] {
            let mut changed = original.clone();
            let object = match section {
                Some(section) => changed.get_mut(section).unwrap(),
                None => &mut changed,
            };
            object["extra"] = serde_json::json!(true);
            assert!(!accepted(&changed, &config));
        }
        let text = serde_json::to_string(&original).unwrap();
        for duplicate in [
            text.replacen('{', "{\"schema\":\"rovezero.pals-rules-descriptor.v2\",", 1),
            text.replace("\"config\":{", "\"config\":{\"width\":384,"),
            text.replace("\"bounds\":{", "\"bounds\":{\"max_records\":128,"),
        ] {
            assert!(
                validate_pals_legacy_rules_descriptor_bytes(duplicate.as_bytes(), &config).is_err()
            );
        }
    }
    #[test]
    fn feature_order_bounds_extensions_and_profile_are_exact() {
        let config = PalsModelConfig::full_line_interaction_v2();
        let original = declaration(&config);
        for section in [
            "metadata_features",
            "record_features",
            "query_features",
            "divergence_features",
            "full_line_features",
            "semantic_fields",
        ] {
            let mut changed = original.clone();
            changed[section].as_array_mut().unwrap().swap(0, 1);
            assert!(!accepted(&changed, &config));
        }
        for bound in ["max_records", "max_line_plies", "max_candidates"] {
            let mut changed = original.clone();
            changed["bounds"][bound] = serde_json::json!(1);
            assert!(!accepted(&changed, &config));
        }
        for section in ["full_line_features", "bounds"] {
            let mut changed = original.clone();
            changed.as_object_mut().unwrap().remove(section);
            assert!(!accepted(&changed, &config));
        }
        let legacy = PalsModelConfig::baseline();
        for section in ["full_line_features", "bounds"] {
            let mut changed = declaration(&legacy);
            changed[section] = serde_json::Value::Null;
            assert!(!accepted(&changed, &legacy));
        }
        assert!(!accepted(
            &original,
            &PalsModelConfig::for_profile(PalsModelProfile::FullLineV2)
        ));
        let mut changed = original;
        changed["schema"] = serde_json::json!("rovezero.pals-rules-descriptor.v3");
        assert!(!accepted(&changed, &config));
        assert!(!accepted(
            &declaration(&legacy),
            &PalsModelConfig::for_profile(PalsModelProfile::VirtualReviewV1)
        ));
    }
    #[test]
    fn byte_bound_and_unregistered_config_are_rejected_before_authority_join() {
        let config = PalsModelConfig::baseline();
        assert!(validate_pals_legacy_rules_descriptor_bytes(&[], &config).is_err());
        assert!(
            validate_pals_legacy_rules_descriptor_bytes(&vec![b' '; 64 * 1024 + 1], &config)
                .is_err()
        );
        let mut invalid = config.clone();
        invalid.width = 1;
        assert!(!accepted(&declaration(&config), &invalid));
    }
}
