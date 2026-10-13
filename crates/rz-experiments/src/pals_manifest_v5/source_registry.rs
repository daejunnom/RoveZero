//! Required source-component declarations for V5 execution purposes.
//!
//! These functions inspect declarations only. They neither create source pins
//! nor verify source files, execution, or acceptance. In particular, the
//! registry must not call the manifest validator which calls back into it.

use super::{PalsCpuSelectionV5, PalsModelBackendV3, PalsRunManifestV5, PalsRunPurposeV5};
use crate::ManifestError;
use std::collections::{BTreeMap, BTreeSet};

pub const PALS_V5_BASE_SOURCE_COMPONENTS: [&str; 12] = [
    "rules",
    "store",
    "review_view",
    "cpu_search",
    "cpu_worker",
    "cpu_credit",
    "coordinator",
    "role_tickets",
    "adaptive_costs",
    "resource_boundary",
    "uci",
    "manifest_v5",
];
pub const PALS_V5_NATIVE_SOURCE_COMPONENTS: [&str; 5] = [
    "model_native",
    "model_encoding",
    "nn_runtime",
    "model_reference",
    "model_export",
];
pub const PALS_V5_COLLECTION_SOURCE_COMPONENTS: [&str; 3] =
    ["external_cpu", "collector", "data_v3"];
pub const PALS_V5_TRAINING_SOURCE_COMPONENTS: [&str; 3] = ["trainer", "checkpoint", "loss_recipe"];
pub const PALS_V5_PAIR_SOURCE_COMPONENTS: [&str; 2] = ["arena", "clock_pgn_audit"];

/// Returns the source components required by the declared execution boundary.
/// Additional, honestly registered source components remain permitted. A
/// returned component name is not a pin or proof that its code was executed.
pub fn pals_required_source_components_v5(manifest: &PalsRunManifestV5) -> BTreeSet<&'static str> {
    let native = manifest
        .engines
        .iter()
        .any(|engine| engine.model.backend != PalsModelBackendV3::DeterministicMock);
    let external = manifest.engines.iter().any(|engine| {
        engine.cpu.cpu_t == PalsCpuSelectionV5::Stockfish19Full
            || engine.cpu.cpu_r == PalsCpuSelectionV5::Stockfish19Full
    });
    required_components(manifest.purpose, native, external, manifest.pair.is_some())
}

/// Checks required key presence and lowercase SHA-256 spelling only. The main
/// manifest validator retains its registry-size and key-shape validation; an
/// actual producer separately verifies the declared source bytes.
pub fn validate_source_components_v5(manifest: &PalsRunManifestV5) -> Result<(), ManifestError> {
    validate_source_pins(
        &manifest.source_pins,
        &pals_required_source_components_v5(manifest),
    )
}

fn required_components(
    purpose: PalsRunPurposeV5,
    native_backend: bool,
    external_cpu: bool,
    has_pair: bool,
) -> BTreeSet<&'static str> {
    let mut components = BTreeSet::from(PALS_V5_BASE_SOURCE_COMPONENTS);
    if native_backend
        || matches!(
            purpose,
            PalsRunPurposeV5::NeuralValidation
                | PalsRunPurposeV5::CollectionPair
                | PalsRunPurposeV5::ActionCollection
                | PalsRunPurposeV5::TrainingGeneration
                | PalsRunPurposeV5::EvaluationPair
        )
    {
        components.extend(PALS_V5_NATIVE_SOURCE_COMPONENTS);
    }
    if matches!(
        purpose,
        PalsRunPurposeV5::CollectionPair
            | PalsRunPurposeV5::ActionCollection
            | PalsRunPurposeV5::TrainingGeneration
    ) {
        components.extend(PALS_V5_COLLECTION_SOURCE_COMPONENTS);
    } else if external_cpu {
        components.insert("external_cpu");
    }
    if purpose == PalsRunPurposeV5::TrainingGeneration {
        components.extend(PALS_V5_TRAINING_SOURCE_COMPONENTS);
    }
    if has_pair
        || matches!(
            purpose,
            PalsRunPurposeV5::CollectionPair | PalsRunPurposeV5::EvaluationPair
        )
    {
        components.extend(PALS_V5_PAIR_SOURCE_COMPONENTS);
    }
    components
}

fn validate_source_pins(
    pins: &BTreeMap<String, String>,
    required: &BTreeSet<&'static str>,
) -> Result<(), ManifestError> {
    for component in required {
        if !pins.contains_key(*component) {
            return Err(ManifestError::Integrity(format!(
                "PALS V5 source registry: missing required component {component}"
            )));
        }
    }
    if pins.values().any(|pin| {
        pin.len() != 64
            || !pin
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err(ManifestError::Integrity(
            "PALS V5 source registry: every component pin must be lowercase SHA-256".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_cpu_and_native_backend_have_distinct_source_boundaries() {
        let base = BTreeSet::from(PALS_V5_BASE_SOURCE_COMPONENTS);
        assert_eq!(
            required_components(PalsRunPurposeV5::CpuValidation, false, false, false),
            base
        );
        let mut native = base;
        native.extend(PALS_V5_NATIVE_SOURCE_COMPONENTS);
        assert_eq!(
            required_components(PalsRunPurposeV5::CpuValidation, true, false, false),
            native
        );
        assert_eq!(
            required_components(PalsRunPurposeV5::NeuralValidation, false, false, false),
            native
        );
    }

    #[test]
    fn external_collection_training_and_pair_components_follow_actual_purpose() {
        let mut external_only = BTreeSet::from(PALS_V5_BASE_SOURCE_COMPONENTS);
        external_only.insert("external_cpu");
        assert_eq!(
            required_components(PalsRunPurposeV5::CpuValidation, false, true, false),
            external_only
        );
        for purpose in [
            PalsRunPurposeV5::CollectionPair,
            PalsRunPurposeV5::ActionCollection,
            PalsRunPurposeV5::TrainingGeneration,
            PalsRunPurposeV5::EvaluationPair,
        ] {
            let mut expected = BTreeSet::from(PALS_V5_BASE_SOURCE_COMPONENTS);
            expected.extend(PALS_V5_NATIVE_SOURCE_COMPONENTS);
            if purpose != PalsRunPurposeV5::EvaluationPair {
                expected.extend(PALS_V5_COLLECTION_SOURCE_COMPONENTS);
            }
            if purpose == PalsRunPurposeV5::TrainingGeneration {
                expected.extend(PALS_V5_TRAINING_SOURCE_COMPONENTS);
            }
            if matches!(
                purpose,
                PalsRunPurposeV5::CollectionPair | PalsRunPurposeV5::EvaluationPair
            ) {
                expected.extend(PALS_V5_PAIR_SOURCE_COMPONENTS);
            }
            assert_eq!(required_components(purpose, false, false, false), expected);
        }
        external_only.extend(PALS_V5_PAIR_SOURCE_COMPONENTS);
        assert_eq!(
            required_components(PalsRunPurposeV5::CpuValidation, false, true, true),
            external_only
        );
    }

    #[test]
    fn source_keys_must_cover_required_components_and_extra_pins_stay_strict() {
        let required = BTreeSet::from(["rules", "manifest_v5"]);
        // Test literals exercise format checks; they are not generated source
        // registrations and confer no source or execution verification.
        let mut pins = BTreeMap::from([
            ("rules".into(), "a".repeat(64)),
            ("manifest_v5".into(), "b".repeat(64)),
            ("extra_reviewed_component".into(), "c".repeat(64)),
        ]);
        validate_source_pins(&pins, &required).unwrap();
        pins.remove("rules");
        assert!(validate_source_pins(&pins, &required).is_err());
        pins.insert("rules".into(), "a".repeat(64));
        pins.insert("extra_reviewed_component".into(), "C".repeat(64));
        assert!(validate_source_pins(&pins, &required).is_err());
        pins.insert("extra_reviewed_component".into(), "c".repeat(63));
        assert!(validate_source_pins(&pins, &required).is_err());
    }
}
