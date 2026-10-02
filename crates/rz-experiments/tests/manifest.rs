use rz_experiments::*;
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const FIXTURE: &str = include_str!("../../../experiments/baselines/fixtures/e01-input.json");
const IDENTITY: &[u8] = include_bytes!("../../../experiments/baselines/fixtures/e01-identity.txt");
const OPENINGS: &[u8] = include_bytes!("../../../experiments/baselines/fixtures/e01-openings.json");
// Independent Python reference: json.dumps(input, sort_keys=True,
// ensure_ascii=False, separators=(",", ":")).encode("utf-8"), then SHA-256.
const FIXTURE_DIGEST: &str = "3763c90ba72e8bf412a84bd187fa2d6e7973994e0a31df3c2c3795075493e567";
// The three tool references share one 63-byte artifact; openings are 215 bytes.
const UNIQUE_ARTIFACT_BYTES: u64 = 278;
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

fn fixture() -> RunManifest {
    RunManifest::from_json(FIXTURE).expect("parse synthetic input fixture")
}

fn runtime_control() -> RunManifest {
    let mut manifest = fixture();
    manifest.purpose = RunPurpose::Development;
    manifest.comparison = Comparison::Runtime;
    for engine in &mut manifest.engines {
        engine.kind = EngineKind::RoveZero;
    }
    manifest.engines[0].runtime_id = "runtime-variant".into();
    manifest.engines[1].runtime_id = "runtime-baseline".into();
    manifest
}

fn observation(engine: &EngineSpec) -> EngineObservation {
    EngineObservation {
        applied_options: engine.requested_options.clone(),
        backend: engine.backend.clone(),
        precision: engine.precision.clone(),
        device: engine.device,
        device_id: "synthetic-cpu-0".into(),
        evidence: engine.tool.binary.clone(),
    }
}

fn validation_code(manifest: &RunManifest, code: &str) {
    let error = manifest.validate().expect_err("invalid manifest must fail");
    let ManifestError::Validation(issues) = error else {
        panic!("expected structured validation failure, received {error}");
    };
    assert!(
        issues.iter().any(|issue| issue.code == code),
        "missing {code}: {issues:?}"
    );
    assert!(
        manifest.clone().lock().is_err(),
        "validation failure must also block locking"
    );
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let parent = std::env::temp_dir().join("rovezero-e01-manifest-tests");
        fs::create_dir_all(&parent).expect("create test root");
        loop {
            let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!("{}-{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create test directory: {error}"),
            }
        }
    }

    fn with_artifacts() -> Self {
        let directory = Self::new();
        fs::write(directory.0.join("e01-identity.txt"), IDENTITY).expect("write identity fixture");
        fs::write(directory.0.join("e01-openings.json"), OPENINGS).expect("write opening fixture");
        directory
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn artifact_error(locked: &LockedManifest, root: &Path, budget: u64) {
    assert!(matches!(
        locked.verify_artifacts(root, budget),
        Err(ManifestError::Artifact(_))
    ));
}

#[test]
fn canonical_lock_matches_an_independent_digest_and_never_authorizes_execution() {
    let locked = fixture().lock().expect("lock synthetic input");
    assert_eq!(locked.sha256(), FIXTURE_DIGEST);
    let encoded = locked.to_json().expect("serialize input lock");
    let envelope: Value = serde_json::from_str(&encoded).expect("parse envelope");
    assert_eq!(envelope["execution_ready"], false);
    assert_eq!(envelope["input_sha256"], FIXTURE_DIGEST);
    let restored = LockedManifest::from_json(&encoded).expect("verify lock roundtrip");
    assert_eq!(restored.input(), locked.input());
    assert_eq!(restored.sha256(), FIXTURE_DIGEST);

    // Parsing sorted compact JSON changes whitespace and object field order.
    let value: Value = serde_json::from_str(FIXTURE).expect("parse fixture independently");
    let reordered = serde_json::to_string(&value).expect("serialize reordered JSON");
    assert_eq!(
        RunManifest::from_json(&reordered)
            .unwrap()
            .lock()
            .unwrap()
            .sha256(),
        FIXTURE_DIGEST
    );
}

#[test]
fn duplicate_decoded_object_keys_are_rejected_including_option_maps() {
    let duplicate_root = FIXTURE.replacen('{', "{\"schema_version\":1,", 1);
    let duplicate_option = FIXTURE.replacen(
        "\"Ponder\": \"false\"",
        "\"Ponder\": \"false\", \"\\u0050onder\": \"true\"",
        1,
    );
    for input in [duplicate_root, duplicate_option] {
        let error =
            RunManifest::from_json(&input).expect_err("duplicate keys must not be last-wins");
        assert!(matches!(error, ManifestError::Parse(_)));
        assert!(error.to_string().contains("duplicate JSON object key"));
    }
}

#[test]
fn schema_unknown_fields_and_unsupported_versions_are_rejected() {
    let mut unknown: Value = serde_json::from_str(FIXTURE).unwrap();
    unknown["engines"][0]["backned"] = json!("cpu");
    assert!(matches!(
        RunManifest::from_json(&unknown.to_string()),
        Err(ManifestError::Parse(_))
    ));
    let mut wrong_version = fixture();
    wrong_version.schema_version = SCHEMA_VERSION + 1;
    validation_code(&wrong_version, "UnsupportedSchema");

    let mut envelope: Value =
        serde_json::from_str(&fixture().lock().unwrap().to_json().unwrap()).unwrap();
    envelope["unexpected_receipt"] = json!({"result": "1-0"});
    assert!(matches!(
        LockedManifest::from_json(&envelope.to_string()),
        Err(ManifestError::Parse(_))
    ));
}

#[test]
fn modified_inputs_readiness_and_lock_versions_cannot_bypass_integrity() {
    let original: Value =
        serde_json::from_str(&fixture().lock().unwrap().to_json().unwrap()).unwrap();
    for (pointer, replacement) in [
        ("/input/plan/seed", json!(999)),
        ("/input_sha256", json!("1".repeat(64))),
        ("/execution_ready", json!(true)),
        ("/lock_version", json!(2)),
        ("/canonicalization", json!("unrecognized-json")),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            matches!(
                LockedManifest::from_json(&changed.to_string()),
                Err(ManifestError::Integrity(_))
            ),
            "tampering accepted at {pointer}"
        );
    }
    let mut changed = original;
    changed["input"]["budget"]["max_wall_ms"] = json!(0);
    assert!(matches!(
        LockedManifest::from_json(&changed.to_string()),
        Err(ManifestError::Validation(_))
    ));
}

#[test]
fn parse_is_bounded_and_budget_numbers_do_not_round_into_u64() {
    assert!(matches!(
        RunManifest::from_json(&" ".repeat(MAX_MANIFEST_BYTES + 1)),
        Err(ManifestError::Parse(_))
    ));
    for number in ["-1", "18446744073709551616", "1.5", "1e999"] {
        let invalid = FIXTURE.replacen(
            "\"max_wall_ms\": 10000",
            &format!("\"max_wall_ms\": {number}"),
            1,
        );
        assert!(
            matches!(
                RunManifest::from_json(&invalid),
                Err(ManifestError::Parse(_))
            ),
            "invalid unsigned budget accepted: {number}"
        );
    }
}

#[test]
fn card_identifiers_must_belong_to_the_76_card_registry() {
    let mut valid = fixture();
    valid.candidate_ids = [
        "CARD-A01", "CARD-A07", "CARD-B09", "CARD-C08", "CARD-D10", "CARD-E12", "CARD-F30",
    ]
    .map(String::from)
    .to_vec();
    valid.validate().expect("registry boundary IDs");
    for id in [
        "CARD-A08", "CARD-B10", "CARD-C09", "CARD-D11", "CARD-E13", "CARD-F31", "CARD-F99",
        "CARD-A00", "TASK-E01",
    ] {
        let mut invalid = fixture();
        invalid.candidate_ids = vec![id.into()];
        validation_code(&invalid, "InvalidCard");
    }
    valid.candidate_ids.push("CARD-A01".into());
    validation_code(&valid, "InvalidCard");
}

#[test]
fn dirty_source_and_artifact_identities_cannot_be_incomplete_or_conflicting() {
    let mut missing_patch = fixture();
    missing_patch.engines[0].tool.dirty = true;
    validation_code(&missing_patch, "DirtyPatchMismatch");
    let mut conflicting = fixture();
    conflicting.engines[0].tool.binary.sha256 = "1".repeat(64);
    validation_code(&conflicting, "ConflictingArtifact");
    let mut empty_digest = fixture();
    empty_digest.input.opening_artifact.sha256.clear();
    validation_code(&empty_digest, "InvalidDigest");
    let mut mutable_source = fixture();
    mutable_source.engines[0].tool.source_commit = "develop".into();
    validation_code(&mutable_source, "InvalidDigest");
}

#[test]
fn declared_pair_retries_children_and_artifacts_use_checked_arithmetic() {
    let mut too_few_games = fixture();
    too_few_games.budget.max_games = 3;
    validation_code(&too_few_games, "BudgetExceeded");
    let mut pair_overflow = fixture();
    pair_overflow.plan.pairs = u64::MAX;
    pair_overflow.plan.max_retries_per_pair = 0;
    pair_overflow.budget.max_pairs = u64::MAX;
    pair_overflow.budget.max_games = u64::MAX;
    validation_code(&pair_overflow, "BudgetExceeded");
    let mut child_overflow = fixture();
    child_overflow.budget.workers = u32::MAX;
    child_overflow.budget.max_child_processes = u32::MAX;
    validation_code(&child_overflow, "ChildBudget");
    let mut artifact_overflow = fixture();
    artifact_overflow.input.opening_artifact.bytes = u64::MAX;
    artifact_overflow.budget.max_artifact_bytes = u64::MAX;
    validation_code(&artifact_overflow, "ArtifactBudget");
    let mut zero_wall_budget = fixture();
    zero_wall_budget.budget.max_wall_ms = 0;
    validation_code(&zero_wall_budget, "InvalidBudget");
}

#[test]
fn runtime_controls_reject_evaluator_and_undeclared_option_changes() {
    let control = runtime_control();
    control.validate().expect("same evaluator runtime control");
    let mut evaluator_change = control.clone();
    evaluator_change.engines[0].evaluator_id = "different-evaluator".into();
    validation_code(&evaluator_change, "ConfoundedControl");
    let mut option_change = control;
    option_change.engines[0]
        .requested_options
        .insert("Threads".into(), "2".into());
    validation_code(&option_change, "ConfoundedControl");
}

#[test]
fn a_single_predeclared_option_change_preserves_all_other_runtime_inputs() {
    let mut control = runtime_control();
    control.engines[0].runtime_id = control.engines[1].runtime_id.clone();
    control.engines[0]
        .requested_options
        .insert("BatchSize".into(), "1".into());
    control.engines[1]
        .requested_options
        .insert("BatchSize".into(), "2".into());
    control.allowed_option_change = Some("batchsize".into());
    control
        .validate()
        .expect("explicit option-only runtime control");
    let mut extra_change = control.clone();
    extra_change.engines[0]
        .requested_options
        .insert("Threads".into(), "2".into());
    validation_code(&extra_change, "ConfoundedControl");
    control.allowed_option_change = Some("absent-option".into());
    validation_code(&control, "InvalidControlVariable");
}

#[test]
fn control_axes_hold_weights_search_and_runtime_fixed_where_required() {
    let mut search = runtime_control();
    search.comparison = Comparison::InternalSearch;
    search.change = Change::Search;
    search.engines[0].runtime_id = search.engines[1].runtime_id.clone();
    search.engines[0].search_id = "search-variant".into();
    search.validate().expect("isolated search comparison");
    let mut changed_runtime = search.clone();
    changed_runtime.engines[0].runtime_id = "extra-runtime-change".into();
    validation_code(&changed_runtime, "ConfoundedControl");
    search.change = Change::MeaningPreserving;
    validation_code(&search, "ControlChangeMismatch");

    let mut weights = runtime_control();
    weights.comparison = Comparison::InternalWeights;
    weights.change = Change::Model;
    weights.engines[0].runtime_id = weights.engines[1].runtime_id.clone();
    for (index, engine) in weights.engines.iter_mut().enumerate() {
        let mut weight = engine.tool.binary.clone();
        weight.path = format!("weight-{index}.txt");
        weight.sha256 = if index == 0 { "1" } else { "2" }.repeat(64);
        engine.weight = Some(weight);
    }
    weights.validate().expect("isolated weight comparison");
    weights.engines[0].search_id = "extra-search-change".into();
    validation_code(&weights, "ConfoundedControl");
}

#[test]
fn observed_identity_requested_values_and_defaults_are_checked_separately() {
    let mut control = runtime_control();
    for engine in &mut control.engines {
        engine.observation = Some(observation(engine));
    }
    control.validate().expect("matching synthetic observations");
    let mut hidden_default = control.clone();
    hidden_default.engines[0]
        .observation
        .as_mut()
        .unwrap()
        .applied_options
        .insert("HiddenDefault".into(), "enabled".into());
    validation_code(&hidden_default, "ConfoundedControl");
    let mut not_applied = control.clone();
    not_applied.engines[0]
        .observation
        .as_mut()
        .unwrap()
        .applied_options
        .insert("Threads".into(), "2".into());
    validation_code(&not_applied, "OptionNotApplied");
    control.engines[0].observation.as_mut().unwrap().device = Device::Gpu;
    validation_code(&control, "AppliedIdentityMismatch");
}

#[test]
fn duplicate_case_options_ponder_and_protocol_line_injection_are_rejected() {
    let mut duplicate_case = fixture();
    duplicate_case.engines[0]
        .requested_options
        .insert("threads".into(), "1".into());
    validation_code(&duplicate_case, "DuplicateOption");
    let mut ponder = fixture();
    ponder.engines[0]
        .requested_options
        .insert("Ponder".into(), "true".into());
    validation_code(&ponder, "ForbiddenPonder");
    let mut injected_command = fixture();
    injected_command.engines[0]
        .requested_options
        .insert("Threads".into(), "1\ngo infinite".into());
    validation_code(&injected_command, "InvalidOption");
}

#[test]
fn missing_gpu_and_unfair_compute_do_not_silently_fall_back() {
    let mut gpu = fixture();
    gpu.research_path = ResearchPath::Gpu;
    gpu.engines[0].device = Device::Gpu;
    validation_code(&gpu, "MissingGpu");
    let mut unfair = fixture();
    unfair.lifecycle.opponent_turn_compute = true;
    validation_code(&unfair, "UnfairComputation");
    let mut warmup_leak = fixture();
    warmup_leak.lifecycle.warmup_input = Some(warmup_leak.input.opening_artifact.clone());
    warmup_leak.lifecycle.warmup_max_ms = 10;
    validation_code(&warmup_leak, "WarmupLeak");
}

#[test]
fn incomplete_failure_and_unknown_history_cannot_be_promoted_to_normal_results() {
    let mut engine_fault = fixture();
    engine_fault.protocol.engine_failure = OutcomePolicy::Incomplete;
    validation_code(&engine_fault, "InvalidFailurePolicy");
    let mut cancellation_loss = fixture();
    cancellation_loss.protocol.cancellation_outcome = OutcomePolicy::Loss;
    validation_code(&cancellation_loss, "InvalidFailurePolicy");
    let mut unknown_prefix = fixture();
    unknown_prefix.input.openings[0].initial = InitialPosition::Fen;
    unknown_prefix.input.openings[0].fen = Some("8/8/8/8/8/8/4k3/4K3 b - - 0 1".into());
    unknown_prefix.input.openings[0].history = HistoryCompleteness::Complete;
    validation_code(&unknown_prefix, "HistoryMismatch");
}

#[test]
fn formal_inputs_require_contract_holdout_weights_observations_and_statistics() {
    let mut formal = runtime_control();
    formal.purpose = RunPurpose::Formal;
    for code in [
        "MissingContract",
        "HoldoutRequired",
        "FixtureEngine",
        "UnverifiedOptions",
        "MissingStatistics",
    ] {
        validation_code(&formal, code);
    }
    formal.clock = ClockSpec::T3 {
        unique_evaluation_budget: 100,
    };
    validation_code(&formal, "DiagnosticTrack");
}

#[test]
fn artifact_verification_deduplicates_references_and_obeys_an_explicit_byte_ceiling() {
    let directory = TestDirectory::with_artifacts();
    let locked = fixture().lock().unwrap();
    locked
        .verify_artifacts(&directory.0, UNIQUE_ARTIFACT_BYTES)
        .expect("verify exact unique-byte budget");
    artifact_error(&locked, &directory.0, UNIQUE_ARTIFACT_BYTES - 1);
    artifact_error(&locked, &directory.0, 0);
    let mut too_small = fixture();
    too_small.budget.max_artifact_bytes = UNIQUE_ARTIFACT_BYTES - 1;
    validation_code(&too_small, "ArtifactBudget");
}

#[test]
fn artifact_hash_and_byte_length_changes_are_distinct_integrity_failures() {
    let directory = TestDirectory::with_artifacts();
    let locked = fixture().lock().unwrap();
    let mut changed = IDENTITY.to_vec();
    changed[0] ^= 1;
    fs::write(directory.0.join("e01-identity.txt"), &changed).unwrap();
    artifact_error(&locked, &directory.0, UNIQUE_ARTIFACT_BYTES);
    fs::write(
        directory.0.join("e01-identity.txt"),
        &IDENTITY[..IDENTITY.len() - 1],
    )
    .unwrap();
    artifact_error(&locked, &directory.0, UNIQUE_ARTIFACT_BYTES);
    fs::write(directory.0.join("e01-identity.txt"), IDENTITY).unwrap();
    locked
        .verify_artifacts(&directory.0, UNIQUE_ARTIFACT_BYTES)
        .expect("restored original fixture");
}

#[test]
fn traversal_secret_and_cross_platform_device_paths_are_rejected_before_reads() {
    for path in [
        "../outside.bin",
        "/absolute/file.bin",
        "safe/../../outside.bin",
        "C:/file.bin",
        "safe\\file.bin",
        "safe/%2e%2e/file.bin",
        ".env",
        "folder/.env.production",
        "service-account.json",
        "credentials.json",
        "id_ed25519",
        "api_key.txt",
        "certificate.pem",
        "NUL",
        "COM1.txt",
        "folder/trailing.",
    ] {
        let mut invalid = fixture();
        invalid.input.opening_artifact.path = path.into();
        validation_code(&invalid, "UnsafeArtifactPath");
    }
}

#[test]
fn non_regular_artifacts_and_missing_roots_are_rejected() {
    let directory = TestDirectory::with_artifacts();
    let locked = fixture().lock().unwrap();
    fs::remove_file(directory.0.join("e01-identity.txt")).unwrap();
    fs::create_dir(directory.0.join("e01-identity.txt")).unwrap();
    artifact_error(&locked, &directory.0, UNIQUE_ARTIFACT_BYTES);
    artifact_error(
        &locked,
        &directory.0.join("not-created"),
        UNIQUE_ARTIFACT_BYTES,
    );
}

#[cfg(unix)]
#[test]
fn fifo_artifact_without_a_writer_is_rejected_within_a_finite_deadline() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    const CHILD_ROOT: &str = "RZ_E01_MANIFEST_FIFO_CHILD_ROOT";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        artifact_error(
            &fixture().lock().unwrap(),
            Path::new(&root),
            UNIQUE_ARTIFACT_BYTES,
        );
        return;
    }

    let directory = TestDirectory::with_artifacts();
    let fifo = directory.0.join("e01-identity.txt");
    fs::remove_file(&fifo).unwrap();
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // The live CString provides a NUL-terminated path, and mkfifo creates only
    // this isolated test entry. There is deliberately no writer for the FIFO.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    // A regressed blocking open must fail this test without hanging the suite.
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "fifo_artifact_without_a_writer_is_rejected_within_a_finite_deadline",
        ])
        .env(CHILD_ROOT, &directory.0)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "FIFO verification subprocess failed");
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("FIFO artifact verification blocked beyond five seconds");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(unix)]
#[test]
fn symlink_artifacts_and_intermediate_root_escape_are_rejected() {
    use std::os::unix::fs::symlink;
    let root = TestDirectory::with_artifacts();
    let outside = TestDirectory::with_artifacts();
    let locked = fixture().lock().unwrap();
    fs::remove_file(root.0.join("e01-identity.txt")).unwrap();
    symlink(
        outside.0.join("e01-identity.txt"),
        root.0.join("e01-identity.txt"),
    )
    .unwrap();
    artifact_error(&locked, &root.0, UNIQUE_ARTIFACT_BYTES);
    fs::remove_file(root.0.join("e01-identity.txt")).unwrap();
    fs::write(root.0.join("e01-identity.txt"), IDENTITY).unwrap();
    symlink(&outside.0, root.0.join("linked-dir")).unwrap();
    let mut intermediate = fixture();
    intermediate.input.opening_artifact.path = "linked-dir/e01-openings.json".into();
    artifact_error(
        &intermediate.lock().unwrap(),
        &root.0,
        UNIQUE_ARTIFACT_BYTES,
    );
    fs::remove_file(root.0.join("linked-dir")).unwrap();
    fs::create_dir(root.0.join("real-dir")).unwrap();
    fs::write(root.0.join("real-dir/e01-openings.json"), OPENINGS).unwrap();
    symlink("real-dir", root.0.join("linked-dir")).unwrap();
    let mut internal_symlink = fixture();
    internal_symlink.input.opening_artifact.path = "linked-dir/e01-openings.json".into();
    artifact_error(
        &internal_symlink.lock().unwrap(),
        &root.0,
        UNIQUE_ARTIFACT_BYTES,
    );
}
