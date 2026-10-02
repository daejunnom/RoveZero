use rz_arena::{
    ArenaError, ArenaPlan, ENGINE_CONTRACT_MANIFEST_REVISION, ENGINE_CONTRACT_SOURCE_REVISION,
    PlanLimits, contract_error, validate_engine_contract_revision,
};
use rz_contracts::{CONTRACT_REVISION, ContractError, ErrorCode, SchemaVersion, Stage};
use rz_experiments::{ManifestError, RunManifest};

fn plan(revision: Option<&str>) -> ArenaPlan {
    let mut input = RunManifest::from_json(include_str!(
        "../../../experiments/baselines/fixtures/e01-input.json"
    ))
    .unwrap();
    input.contract_revision = revision.map(str::to_owned);
    ArenaPlan::build(
        &input.lock().unwrap(),
        PlanLimits {
            max_pairs: 1,
            max_json_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
}

#[test]
fn exact_manifest_revision_imports_and_validates_the_native_schema() {
    let plan = plan(Some("0.1"));
    assert_eq!(plan.manifest().input().schema_version, 1);
    let revision = validate_engine_contract_revision(&plan).unwrap();
    assert_eq!(revision, CONTRACT_REVISION);
    assert_eq!(revision, SchemaVersion { major: 0, minor: 1 });
    revision.validate().unwrap();
    assert_eq!(ENGINE_CONTRACT_MANIFEST_REVISION, "0.1");
    assert_eq!(
        ENGINE_CONTRACT_SOURCE_REVISION,
        "67284c4f66f7a7ae9f46fa63dfd50e7410eb6845"
    );
    assert_eq!(
        ENGINE_CONTRACT_SOURCE_REVISION,
        rz_position::contracts::CONTRACT_SOURCE_REVISION
    );
}

#[test]
fn null_noncanonical_and_unsupported_revisions_never_pass_execution_admission() {
    for revision in [
        None,
        Some("0.10"),
        Some("0.0"),
        Some("0.2"),
        Some("1.0"),
        Some("1"),
        Some("00.1"),
        Some("0.01"),
        Some("0.1.0"),
        Some("v0.1"),
        Some("67284c4f66f7a7ae9f46fa63dfd50e7410eb6845"),
    ] {
        let plan = plan(revision); // Structural planning intentionally still works.
        let error = validate_engine_contract_revision(&plan).unwrap_err();
        assert!(matches!(error, ArenaError::Contract(_)));
        let shared = contract_error(&error);
        assert_eq!(shared.code, ErrorCode::UnsupportedContract, "{revision:?}");
        assert_eq!(shared.stage, Stage::Contract, "{revision:?}");
    }
}

#[test]
fn whitespace_revisions_are_rejected_before_a_structural_plan_can_be_locked() {
    for revision in [" 0.1", "0.1 ", "0.1\n"] {
        let mut input = RunManifest::from_json(include_str!(
            "../../../experiments/baselines/fixtures/e01-input.json"
        ))
        .unwrap();
        input.contract_revision = Some(revision.into());
        let error = input.lock().unwrap_err();
        assert_eq!(contract_error(&error.into()).code, ErrorCode::InvalidInput);
    }
}

#[test]
fn native_unsupported_schema_error_is_preserved_without_string_matching() {
    let native = SchemaVersion {
        major: 0,
        minor: 10,
    }
    .validate()
    .unwrap_err();
    let local: ArenaError = native.into();
    assert_eq!(contract_error(&local), native);
    assert_eq!(native.code, ErrorCode::UnsupportedContract);
    assert_eq!(native.stage, Stage::Contract);
}

#[test]
fn common_error_mapping_keeps_stable_codes_stages_and_static_details() {
    let external = "unbounded-external-output-containing-a-private-path";
    for (local, code, stage) in [
        (
            ArenaError::Invalid(external.into()),
            ErrorCode::InvalidInput,
            Stage::Admission,
        ),
        (
            ArenaError::Budget(external.into()),
            ErrorCode::ResourceExhausted,
            Stage::Admission,
        ),
        (
            ArenaError::Integrity(external.into()),
            ErrorCode::IdentityMismatch,
            Stage::Contract,
        ),
        (
            ArenaError::Io(external.into()),
            ErrorCode::BackendFailure,
            Stage::Backend,
        ),
        (
            ArenaError::Manifest(ManifestError::Parse(external.into())),
            ErrorCode::InvalidInput,
            Stage::Admission,
        ),
        (
            ArenaError::Manifest(ManifestError::Integrity(external.into())),
            ErrorCode::IdentityMismatch,
            Stage::Contract,
        ),
        (
            ArenaError::Manifest(ManifestError::Artifact(external.into())),
            ErrorCode::BackendFailure,
            Stage::Backend,
        ),
    ] {
        let shared = contract_error(&local);
        assert_eq!((shared.code, shared.stage), (code, stage));
        assert!(!shared.detail.contains(external));
        assert!(shared.detail.len() < 128);
    }
    let native = ContractError::new(ErrorCode::Expired, Stage::Output, "native expiry");
    assert_eq!(contract_error(&ArenaError::Contract(native)), native);
}
