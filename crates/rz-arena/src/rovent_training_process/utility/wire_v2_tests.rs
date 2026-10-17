//! Explicit utility execution /2 wire vectors. Original /1 fixtures and bytes
//! are not consumed, re-tagged, default-filled or rewritten by this module.
//! An ancestry declaration is metadata, never proof of source-family coverage,
//! Rules replay, a current initialization owner or execution authority.

use super::execution::{
    RoventUtilityCheckpointReference, RoventUtilityDatasetReference, RoventUtilityExecutionAction,
    RoventUtilityFrozenPcSource, RoventUtilityInitializationSource, RoventUtilityRequiredNullable,
};
use super::execution_v2::{
    ROVENT_UTILITY_EXECUTION_REGISTRATION_V2, RoventUtilityAncestorDatasetRegistration,
    RoventUtilityExecutionRegistrationV2,
};
use crate::rovent_training_process::recovery::metadata_wire_tests::decode_python_fixture;
use crate::rovent_training_process::recovery::{
    RoventHistoricalDatasetRegistration, RoventHistoricalRAuxiliaryDatasetRegistration,
    RoventHistoricalRoleBinding, RoventTrainerRuntimeEndpoint,
    rovent_trainer_dependency_set_sha256,
};
use crate::rovent_training_process::{RoventTrainingFilePin, RoventTrainingResourcePolicy};
use rz_experiments::pals_data_v3::generation_digest;
use rz_experiments::{ArtifactRef, RoventNnuecPin};
use rz_search::cpu_value::nnuec::v2 as nnuec_v2;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, path::Path};

const FIXTURE_DOMAIN: &str = "rz-rovent-utility-metadata-v2-cross-language-fixture/1";
const RESULT_DOMAIN: &str = "rz-rovent-utility-metadata-v2-cross-language-result/1";
const ANCESTOR_WIRE_DOMAIN: &str = "rz-rovent-utility-ancestor-metadata-wire/1";

fn h(label: &str) -> String {
    format!("{:x}", Sha256::digest(label.as_bytes()))
}
fn pin(path: &str, label: &str) -> RoventTrainingFilePin {
    RoventTrainingFilePin {
        path: path.into(),
        sha256: h(label),
        bytes: 128,
    }
}
fn pc_source(label: &str, model_epoch: u64) -> RoventHistoricalDatasetRegistration {
    let root = format!("/metadata-v2/{label}");
    RoventHistoricalDatasetRegistration {
        root: root.clone().into(),
        publication: pin(
            &format!("{root}/publication.json"),
            &format!("{label}-original-publication"),
        ),
        collection_source_commit: "a".repeat(40),
        collection_source_sha256: h(&format!("{label}-source")),
        collection_runtime_library_sha256: h(&format!("{label}-ort-library")),
        model_semantics_sha256: h(&format!("{label}-model-semantics")),
        model_epoch,
        context_epoch: model_epoch + 2,
        recipe_sha256: h(&format!("{label}-canonical-recipe")),
        stage_id: format!("{label}-stage"),
        role_bindings: ["proposer", "critic", "validator"]
            .into_iter()
            .map(|role| {
                (
                    role.into(),
                    RoventHistoricalRoleBinding {
                        model_semantics_sha256: h(&format!("{label}-{role}-model-semantics")),
                        model_identity_sha256: h(&format!("{label}-{role}-ticket-model")),
                        weights_sha256: h(&format!("{label}-{role}-original-raw-weights")),
                        encoding_sha256: h(&format!("{label}-{role}-encoding")),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>(),
    }
}
fn r_source() -> RoventHistoricalRAuxiliaryDatasetRegistration {
    let artifact = |path: &str, label: &str| ArtifactRef {
        path: path.into(),
        sha256: h(label),
        bytes: 128,
        source: "https://github.com/daejunnom/RoveZero".into(),
        license: "MIT".into(),
    };
    RoventHistoricalRAuxiliaryDatasetRegistration {
        root: "/metadata-v2/r-original".into(),
        publication: pin(
            "/metadata-v2/r-original/r-publication.json",
            "r-original-publication",
        ),
        collection_source_commit: "b".repeat(40),
        collection_source_sha256: h("r-original-source"),
        collection_runtime_library_sha256: h("r-original-ort-library"),
        collection_model_semantics_sha256: h("r-original-pc-model-semantics"),
        collection_model_epoch: (1_u64 << 53) + 11,
        collection_context_epoch: (1_u64 << 53) + 13,
        collection_recipe_sha256: h("r-original-canonical-recipe"),
        collection_stage_id: "r-original-stage".into(),
        nnuec: RoventNnuecPin {
            schema: nnuec_v2::SCHEMA.into(),
            semantics: nnuec_v2::SEMANTICS.into(),
            encoding: nnuec_v2::INPUT_SCHEMA.into(),
            implementation_sha256: h("r-original-nnuec-implementation"),
            configuration_sha256: h("r-original-nnuec-config"),
            weights: artifact("nnuec/original-weights.bin", "r-original-raw-nnuec-weights"),
            provenance: artifact(
                "nnuec/original-provenance.json",
                "r-original-raw-nnuec-provenance",
            ),
            diagnostic: true,
            context_namespace_policy_sha256: h("r-original-context-policy"),
            local_namespace_policy_sha256: h("r-original-local-policy"),
            search_conditions_sha256: h("r-original-search-conditions"),
        },
        nnuec_parameter_sha256: h("r-original-nnuec-semantic-parameter-identity"),
    }
}
fn utility_source(label: &str) -> RoventUtilityDatasetReference {
    let root = format!("/metadata-v2/{label}");
    RoventUtilityDatasetReference {
        root: root.clone().into(),
        publication: pin(
            &format!("{root}/publication.json"),
            &format!("{label}-original-publication"),
        ),
        durable_ack: pin(
            &format!("{root}/acknowledgements/original.json"),
            &format!("{label}-original-durable-ack"),
        ),
    }
}

fn fixture() -> Value {
    let legacy = pc_source("pc-legacy", (1_u64 << 53) + 1);
    let submitted = pc_source("pc-submitted", (1_u64 << 53) + 5);
    let r = r_source();
    let utility = utility_source("utility-ancestor");
    let original_payloads = [
        serde_json::to_value(&legacy).unwrap(),
        serde_json::to_value(&submitted).unwrap(),
        serde_json::to_value(&r).unwrap(),
        serde_json::to_value(&utility).unwrap(),
    ];
    let ancestors = vec![
        RoventUtilityAncestorDatasetRegistration::PcLegacy {
            registration: legacy,
        },
        RoventUtilityAncestorDatasetRegistration::PcSubmittedInputV1 {
            registration: submitted,
        },
        RoventUtilityAncestorDatasetRegistration::RAuxiliary {
            registration: r,
            recipe: pin(
                "/metadata-v2/r-original-inputs/recipe.json",
                "r-original-raw-recipe-file",
            ),
        },
        RoventUtilityAncestorDatasetRegistration::UtilityCohort {
            registration: utility,
            preparation: pin(
                "/metadata-v2/utility-ancestor-inputs/preparation.json",
                "utility-ancestor-original-raw-preparation-file",
            ),
        },
    ];
    for (ancestor, original) in ancestors.iter().zip(original_payloads) {
        ancestor.validate().unwrap();
        assert_eq!(
            serde_json::to_value(ancestor).unwrap()["registration"],
            original
        );
    }
    let endpoints = vec![
        RoventTrainerRuntimeEndpoint {
            name: "torch.__file__".into(),
            artifact: pin(
                "/metadata-v2/deps/torch/__init__.py",
                "v2-torch-python-endpoint",
            ),
        },
        RoventTrainerRuntimeEndpoint {
            name: "torch._C.__file__".into(),
            artifact: pin("/metadata-v2/deps/torch/_C.so", "v2-torch-native-endpoint"),
        },
    ];
    let registration = RoventUtilityExecutionRegistrationV2 {
        schema: ROVENT_UTILITY_EXECUTION_REGISTRATION_V2.into(),
        execution_id: "utility-v2-wire-execution".into(),
        action: RoventUtilityExecutionAction::Train,
        export_request: RoventUtilityRequiredNullable(None),
        preparation: pin(
            "/metadata-v2/current-inputs/preparation.json",
            "v2-current-original-raw-preparation",
        ),
        window_spec: pin(
            "/metadata-v2/current-inputs/window.json",
            "v2-current-original-raw-window",
        ),
        trainer_source_sha256: h("v2-current-trainer-source"),
        trainer_python_executable: pin("/metadata-v2/python3", "v2-python-raw-executable"),
        trainer_python_version: "3.12.3".into(),
        dependency_directories: vec!["/metadata-v2/deps".into()],
        trainer_dependency_set_sha256: rovent_trainer_dependency_set_sha256(&endpoints).unwrap(),
        runtime_endpoints: endpoints,
        current_dataset: utility_source("utility-current"),
        ancestor_datasets: ancestors,
        initialization_source: RoventUtilityInitializationSource::Resume {
            checkpoint: RoventUtilityCheckpointReference {
                root: "/metadata-v2/resume-checkpoint".into(),
                publication: pin(
                    "/metadata-v2/resume-checkpoint/publication.json",
                    "v2-original-checkpoint-publication",
                ),
            },
        },
        frozen_nnuec_base: pin(
            "/metadata-v2/current-inputs/frozen-base.json",
            "v2-frozen-base-raw-file",
        ),
        frozen_pc_sources: ["proposer", "critic"]
            .into_iter()
            .map(|role| RoventUtilityFrozenPcSource {
                role: role.into(),
                artifact: pin(
                    &format!("/metadata-v2/current-inputs/{role}.onnx"),
                    &format!("v2-{role}-raw-file"),
                ),
                model_semantics_sha256: h(&format!("v2-{role}-semantics")),
                model_identity_sha256: h(&format!("v2-{role}-ticket-identity")),
                encoding_sha256: h(&format!("v2-{role}-encoding")),
            })
            .collect(),
        resources_policy: RoventTrainingResourcePolicy {
            cpu_quota_millicpus: 2000,
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
            pids_max: 64,
            affinity: vec![0, 2],
            device: "cpu".into(),
        },
        require_total_quota: true,
        output_root: "/metadata-v2/output".into(),
        control_root: "/metadata-v2/control".into(),
        checkpoint_directory: "/metadata-v2/output/checkpoints".into(),
        receipt_path: "/metadata-v2/output/receipt.json".into(),
        max_rpc_messages: 512,
        max_rpc_bytes: 1 << 20,
        max_read_bytes: 16 << 20,
        max_child_read_bytes: 8 << 20,
        max_output_bytes: 2 << 20,
        max_artifact_files: 256,
    };
    registration.validate().unwrap();
    let expected_ancestor_kinds = registration
        .ancestor_datasets
        .iter()
        .map(|ancestor| ancestor.kind())
        .collect::<Vec<_>>();
    let expected_ancestor_metadata_sha256 = registration
        .ancestor_datasets
        .iter()
        .map(|ancestor| generation_digest(ANCESTOR_WIRE_DOMAIN, ancestor).unwrap())
        .collect::<Vec<_>>();
    json!({"schema":FIXTURE_DOMAIN,"registration":registration,
        "expected_registration_sha256":registration.canonical_sha256().unwrap(),
        "expected_ancestor_kinds":expected_ancestor_kinds,
        "expected_ancestor_metadata_sha256":expected_ancestor_metadata_sha256})
}
fn decode(
    value: &Value,
) -> crate::rovent_training_process::recovery::metadata_wire_tests::DecoderOutput {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../experiments/model-research/pals/tests/rovent_utility_metadata_v2_wire_decode.py",
    );
    decode_python_fixture(value, &helper)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Accepted {
    schema: String,
    status: String,
    registration_sha256: String,
    ancestor_kinds: Vec<String>,
    ancestor_metadata_sha256: Vec<String>,
    original_v1_decoder_rejected: bool,
    historical_family_coverage_verified: bool,
    performed_optimizer_updates: u64,
    past_native_authority_reissued: bool,
    execution_authorized: bool,
    neural_imports: bool,
}

#[test]
fn actual_v2_registration_preserves_all_four_original_tagged_payloads() {
    let fixture = fixture();
    let registration = &fixture["registration"];
    assert_eq!(registration.as_object().unwrap().len(), 29);
    assert_eq!(
        registration["schema"],
        ROVENT_UTILITY_EXECUTION_REGISTRATION_V2
    );
    assert_eq!(registration.get("export_request"), Some(&Value::Null));
    assert_eq!(registration["initialization_source"]["mode"], "resume");
    assert_eq!(
        registration["initialization_source"]["checkpoint"]
            .as_object()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        fixture["expected_ancestor_kinds"],
        json!([
            "pc_legacy",
            "pc_submitted_input_v1",
            "r_auxiliary",
            "utility_cohort"
        ])
    );
    let ancestors = registration["ancestor_datasets"].as_array().unwrap();
    assert_eq!(
        ancestors
            .iter()
            .map(|ancestor| ancestor.as_object().unwrap().len())
            .collect::<Vec<_>>(),
        vec![2, 2, 3, 3]
    );
    assert_eq!(
        ancestors[0]["registration"]["model_epoch"].as_u64(),
        Some((1_u64 << 53) + 1)
    );
    assert_eq!(
        ancestors[1]["registration"]["model_epoch"].as_u64(),
        Some((1_u64 << 53) + 5)
    );
    assert_eq!(
        ancestors[2]["registration"]["collection_model_epoch"].as_u64(),
        Some((1_u64 << 53) + 11)
    );
    assert_ne!(
        ancestors[2]["recipe"]["sha256"],
        ancestors[2]["registration"]["collection_recipe_sha256"]
    );
    assert_ne!(
        ancestors[2]["registration"]["nnuec_parameter_sha256"],
        ancestors[2]["registration"]["nnuec"]["weights"]["sha256"]
    );
    assert_ne!(
        ancestors[3]["preparation"]["sha256"],
        registration["preparation"]["sha256"]
    );
    let output = decode(&fixture);
    assert!(
        output.status.success(),
        "strict F /2 decoder rejected actual Rust DTOs: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Accepted = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result.schema, RESULT_DOMAIN);
    assert_eq!(result.status, "accepted_metadata_only");
    assert_eq!(
        fixture["expected_registration_sha256"],
        result.registration_sha256
    );
    assert_eq!(
        fixture["expected_ancestor_kinds"],
        json!(result.ancestor_kinds)
    );
    assert_eq!(
        fixture["expected_ancestor_metadata_sha256"],
        json!(result.ancestor_metadata_sha256)
    );
    assert!(result.original_v1_decoder_rejected);
    assert!(!result.historical_family_coverage_verified);
    assert_eq!(result.performed_optimizer_updates, 0);
    assert!(!result.past_native_authority_reissued);
    assert!(!result.execution_authorized);
    assert!(!result.neural_imports);
}

#[test]
fn independent_v2_decoder_requires_original_r_recipe_and_utility_preparation_pins() {
    let original = fixture();
    for (index, field) in [(2, "recipe"), (3, "preparation")] {
        let mut missing = original.clone();
        missing["registration"]["ancestor_datasets"][index]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(
                missing["registration"].clone()
            )
            .is_err()
        );
        assert!(
            !decode(&missing).status.success(),
            "required original ancestry {field} was silently defaulted"
        );
    }
}

#[test]
fn independent_v2_decoder_rejects_v1_retagging_and_untagged_utility_projection() {
    let original = fixture();
    let mut retagged = original.clone();
    retagged["registration"]["schema"] =
        json!(super::execution::ROVENT_UTILITY_EXECUTION_REGISTRATION);
    let typed: RoventUtilityExecutionRegistrationV2 =
        serde_json::from_value(retagged["registration"].clone()).unwrap();
    assert!(typed.validate().is_err());
    assert!(
        !decode(&retagged).status.success(),
        "/1 schema was silently converted into a /2 registration"
    );
    let mut untagged = original;
    untagged["registration"]["ancestor_datasets"][3] =
        untagged["registration"]["ancestor_datasets"][3]["registration"].clone();
    assert!(
        serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(
            untagged["registration"].clone()
        )
        .is_err()
    );
    assert!(
        !decode(&untagged).status.success(),
        "original utility ancestor was silently projected into an untagged /1 list"
    );
}

#[test]
fn independent_v2_decoder_rejects_cross_family_shape_and_nested_epoch_boolean() {
    let original = fixture();
    let mut wrong_family = original.clone();
    wrong_family["registration"]["ancestor_datasets"][2]["kind"] = json!("pc_legacy");
    assert!(
        serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(
            wrong_family["registration"].clone()
        )
        .is_err()
    );
    assert!(
        !decode(&wrong_family).status.success(),
        "R payload silently became a PC payload through its tag"
    );
    let mut boolean = original;
    boolean["registration"]["ancestor_datasets"][0]["registration"]["model_epoch"] = json!(true);
    assert!(
        serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(
            boolean["registration"].clone()
        )
        .is_err()
    );
    assert!(
        !decode(&boolean).status.success(),
        "nested PC epoch boolean silently became an integer"
    );
}

#[test]
fn independent_v2_decoder_rejects_repeated_publication_and_foreign_canonical_pin() {
    let original = fixture();
    let mut duplicate = original.clone();
    duplicate["registration"]["ancestor_datasets"][1] =
        duplicate["registration"]["ancestor_datasets"][0].clone();
    let typed: RoventUtilityExecutionRegistrationV2 =
        serde_json::from_value(duplicate["registration"].clone()).unwrap();
    assert!(typed.validate().is_err());
    assert!(
        !decode(&duplicate).status.success(),
        "same original publication was registered twice"
    );
    let mut foreign_pin = original;
    foreign_pin["registration"]["ancestor_datasets"][3]["preparation"]["sha256"] =
        json!(h("foreign-ancestor-preparation"));
    assert!(
        !decode(&foreign_pin).status.success(),
        "canonical /2 metadata accepted another original ancestor pin"
    );
}
