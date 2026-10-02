use rz_experiments::*;
use serde_json::{Value, json};

// These are declarations with synthetic identities, not executable/model bytes
// or proof that a provider, A rules, a clock or a process limit was exercised.
fn artifact(path: &str, bytes: u64) -> ArtifactRef {
    ArtifactRef {
        path: path.into(),
        sha256: "a".repeat(64),
        bytes,
        source: "https://example.org/rovezero-native-test".into(),
        license: "Synthetic metadata fixture only".into(),
    }
}

fn fixture() -> IntegrationPairSpecV1 {
    let artifacts: Vec<_> = NativeArtifactRole::ALL
        .into_iter()
        .enumerate()
        .map(|(index, role)| NativeArtifactBinding {
            role,
            artifact: artifact(&format!("native/input-{index}"), 10 + index as u64),
        })
        .collect();
    let baseline = CpuNativeLaunchSpecV1 {
        role: NativeEngineRole::Baseline,
        engine_id: "native-baseline".into(),
        source_commit: "b".repeat(40),
        target: "x86_64-unknown-linux-gnu".into(),
        artifacts,
        profile: NativeCpuProfileV1::fixed("c".repeat(64), "d".repeat(64)),
    };
    let mut candidate = baseline.clone();
    candidate.role = NativeEngineRole::Candidate;
    candidate.engine_id = "native-candidate".into();
    IntegrationPairSpecV1 {
        schema_version: 1,
        purpose: NativeIntegrationPurpose::CpuNnIntegration,
        strength_eligible: false,
        contract_revision: "0.1".into(),
        run_id: "native-run-1".into(),
        pair_id: "native-pair-1".into(),
        engines: [baseline, candidate],
        white_order: [NativeEngineRole::Candidate, NativeEngineRole::Baseline],
        opening: OpeningSpec {
            id: "native-opening-1".into(),
            initial: InitialPosition::Startpos,
            fen: None,
            moves: vec!["e2e4".into(), "e7e5".into()],
            history: HistoryCompleteness::Complete,
            history_origin: "startpos-complete-trace".into(),
        },
        opening_artifact: artifact("inputs/opening.pgn", 50),
        runner: ToolIdentity {
            version: "pinned-runner-test-v1".into(),
            source_url: "https://example.org/rovezero-native-test".into(),
            source_commit: "e".repeat(40),
            binary: artifact("runner/fastchess", 100),
            dirty: false,
            dirty_patch: None,
            build_mode: "release".into(),
            compiler: "rustc 1.96.0".into(),
            target: "x86_64-unknown-linux-gnu".into(),
            isa: "x86-64".into(),
        },
        clock: NativeMovetimeV1 { movetime_ms: 100 },
        max_plies: 4,
        timeouts: NativeTimeoutsV1 {
            startup_ms: 1000,
            handshake_ms: 1000,
            runtime_ms: 10000,
            drain_ms: 2000,
            shutdown_ms: 2000,
        },
        budget: NativeResourceBudgetV1 {
            max_input_bytes: 4096,
            max_output_bytes: 128 * 1024,
            max_runtime_bytes: 256 * 1024,
            max_artifact_bytes: 1024 * 1024,
            max_child_processes: 3,
            max_runtime_files: 32,
            max_runtime_depth: 6,
            address_space_per_process_bytes: 2 * 1024 * 1024 * 1024,
        },
    }
}

fn code(error: ManifestError, expected: &str) {
    let ManifestError::Validation(issues) = error else {
        panic!("expected validation error, got {error}")
    };
    assert!(
        issues.iter().any(|issue| issue.code == expected),
        "missing {expected}: {issues:?}"
    );
}

fn invalid(spec: IntegrationPairSpecV1, expected: &str) {
    code(spec.clone().validate().unwrap_err(), expected);
    code(spec.lock().unwrap_err(), expected);
}

#[test]
fn same_native_inputs_have_distinct_roles_swapped_colors_and_one_byte_charge() {
    let spec = fixture();
    assert!(spec.engines[0].same_inputs_as(&spec.engines[1]).unwrap());
    assert_ne!(spec.engines[0].engine_id, spec.engines[1].engine_id);
    assert_eq!(spec.white_order[0].other(), spec.white_order[1]);
    assert_eq!(spec.declared_artifacts().len(), 12);
    // Five shared native inputs total 60 bytes; runner 100 and opening 50.
    assert_eq!(spec.unique_input_bytes().unwrap(), 210);
    let locked = spec.lock().unwrap();
    assert_eq!(locked.unique_input_bytes().unwrap(), 210);
    assert_eq!(locked.input(), locked.spec());
    let wire = locked.to_json().unwrap();
    assert!(wire.len() <= MAX_NATIVE_LAUNCH_JSON_BYTES);
    let envelope: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(envelope["execution_ready"], false);
    assert_eq!(envelope["input"]["strength_eligible"], false);
    assert_eq!(envelope["domain"], NATIVE_LAUNCH_DOMAIN);
    assert_eq!(
        LockedIntegrationPairSpecV1::from_json(&wire)
            .unwrap()
            .sha256(),
        locked.sha256()
    );
}

#[test]
fn canonical_identity_ignores_role_declaration_order_but_binds_execution_order() {
    let spec = fixture();
    let expected = spec.clone().lock().unwrap();
    let mut reordered = spec.clone();
    reordered.engines.reverse();
    for engine in &mut reordered.engines {
        engine.artifacts.reverse();
    }
    assert_eq!(reordered.lock().unwrap().sha256(), expected.sha256());
    let mut changed_order = spec;
    changed_order.white_order.reverse();
    assert_ne!(changed_order.lock().unwrap().sha256(), expected.sha256());
}

#[test]
fn lock_rejects_tampering_execution_permission_and_foreign_domains() {
    let locked = fixture().lock().unwrap();
    let wire = locked.to_json().unwrap();
    let mut value: Value = serde_json::from_str(&wire).unwrap();
    value["input"]["clock"]["movetime_ms"] = json!(101);
    code(
        LockedIntegrationPairSpecV1::from_json(&value.to_string()).unwrap_err(),
        "NativeLockDigestMismatch",
    );
    for (key, replacement) in [
        ("execution_ready", json!(true)),
        ("domain", json!("rz-e01-json-v1")),
        ("lock_version", json!(2)),
        ("canonicalization", json!("different")),
    ] {
        let mut value: Value = serde_json::from_str(&wire).unwrap();
        value[key] = replacement;
        code(
            LockedIntegrationPairSpecV1::from_json(&value.to_string()).unwrap_err(),
            "UnsupportedNativeLock",
        );
    }
}

#[test]
fn readers_reject_nested_duplicate_keys_unknown_fields_and_unapproved_profiles() {
    let raw = serde_json::to_string(&fixture()).unwrap();
    let duplicated = raw.replacen("\"batch_size\":1", "\"batch_size\":1,\"batch_size\":1", 1);
    assert_ne!(raw, duplicated);
    assert!(matches!(
        IntegrationPairSpecV1::from_json(&duplicated),
        Err(ManifestError::Parse(_))
    ));
    for pointer in ["/extra", "/engines/0/profile/extra", "/budget/extra"] {
        let mut value = serde_json::to_value(fixture()).unwrap();
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        value
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(key.into(), json!(true));
        assert!(matches!(
            IntegrationPairSpecV1::from_json(&value.to_string()),
            Err(ManifestError::Parse(_))
        ));
    }
    for (pointer, replacement) in [
        ("/purpose", json!("internal_search")),
        ("/engines/0/profile/provider", json!("cuda")),
        ("/engines/0/profile/precision", json!("fp16")),
        ("/engines/0/profile/history_fill", json!("repeat_oldest")),
    ] {
        let mut value = serde_json::to_value(fixture()).unwrap();
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(matches!(
            IntegrationPairSpecV1::from_json(&value.to_string()),
            Err(ManifestError::Parse(_))
        ));
    }
    let mut extra_clock = serde_json::to_value(fixture()).unwrap();
    extra_clock["clock"]["base_ms"] = json!(1000);
    assert!(matches!(
        IntegrationPairSpecV1::from_json(&extra_clock.to_string()),
        Err(ManifestError::Parse(_))
    ));
    let mut retries = serde_json::to_value(fixture()).unwrap();
    retries["retries"] = json!(1);
    assert!(matches!(
        IntegrationPairSpecV1::from_json(&retries.to_string()),
        Err(ManifestError::Parse(_))
    ));
}

#[test]
fn sixty_four_kib_reader_and_full_lock_envelope_limits_apply() {
    let raw = serde_json::to_string(&fixture()).unwrap();
    let mut exact = raw.clone();
    exact.push_str(&" ".repeat(MAX_NATIVE_LAUNCH_JSON_BYTES - raw.len()));
    assert!(IntegrationPairSpecV1::from_json(&exact).is_ok());
    exact.push(' ');
    code(
        IntegrationPairSpecV1::from_json(&exact).unwrap_err(),
        "NativePayloadLimit",
    );
    code(
        LockedIntegrationPairSpecV1::from_json(&exact).unwrap_err(),
        "NativePayloadLimit",
    );

    // Fill a syntactic declaration near the cap. Its opening is deliberately not
    // replayed: this test exercises wire allocation limits, not A chess legality.
    let mut near_cap = fixture();
    near_cap.opening.moves = vec!["e2e4".into(); 4092];
    near_cap.max_plies = 4094;
    for engine in &mut near_cap.engines {
        for binding in &mut engine.artifacts {
            binding.artifact.license = "L".into();
        }
    }
    let base = serde_json::to_string(&near_cap).unwrap().len();
    let canonical_overhead = json!({ "domain": NATIVE_LAUNCH_DOMAIN,
        "schema_version": NATIVE_LAUNCH_SCHEMA_VERSION, "execution_ready": false,
        "input": null })
    .to_string()
    .len()
        - "null".len();
    let small = fixture();
    let envelope_overhead = small.clone().lock().unwrap().to_json().unwrap().len()
        - serde_json::to_string(&small).unwrap().len();
    assert!(envelope_overhead > canonical_overhead + 32);
    // The canonical payload still fits. Only the complete export envelope is
    // too large, exercising the public lock() gate rather than its hash input.
    let target = MAX_NATIVE_LAUNCH_JSON_BYTES - canonical_overhead - 16;
    let padding = (target - base) / 10;
    assert!(padding < 4096);
    for engine in &mut near_cap.engines {
        for binding in &mut engine.artifacts {
            binding.artifact.license = "L".repeat(1 + padding);
        }
    }
    let raw = serde_json::to_string(&near_cap).unwrap();
    assert!(raw.len() <= MAX_NATIVE_LAUNCH_JSON_BYTES);
    assert!(raw.len() + canonical_overhead <= MAX_NATIVE_LAUNCH_JSON_BYTES);
    assert!(raw.len() + envelope_overhead > MAX_NATIVE_LAUNCH_JSON_BYTES);
    assert!(near_cap.validate().is_ok());
    let error = near_cap.lock().unwrap_err();
    let ManifestError::Validation(issues) = &error else {
        panic!("expected payload boundary")
    };
    assert!(
        issues
            .iter()
            .any(|issue| issue.message == "native declaration JSON exceeds 64 KiB")
    );
    code(error, "NativePayloadLimit");

    let mut huge = fixture();
    huge.opening.moves = vec!["e2e4".into(); 4096];
    invalid(huge, "NativePlyBudget");
}

#[test]
fn artifact_roles_and_identity_conflicts_cannot_be_overwritten_by_dedup() {
    let mut missing = fixture();
    missing.engines[0].artifacts.pop();
    invalid(missing, "ArtifactRoleCount");
    let mut duplicate = fixture();
    duplicate.engines[0].artifacts[4] = duplicate.engines[0].artifacts[0].clone();
    invalid(duplicate, "DuplicateArtifactRole");
    let mut alias = fixture();
    alias.engines[0].artifacts[2].artifact.path =
        alias.engines[0].artifacts[1].artifact.path.clone();
    invalid(alias, "ArtifactRoleAlias");
    let mut conflict = fixture();
    conflict.engines[1].artifacts[0].artifact.license = "different provenance".into();
    code(
        conflict.unique_input_bytes().unwrap_err(),
        "ArtifactIdentityConflict",
    );
    invalid(conflict, "IntegrationInputMismatch");
    let mut opening_alias = fixture();
    opening_alias.opening_artifact = opening_alias.engines[0].artifacts[0].artifact.clone();
    invalid(opening_alias, "ArtifactRoleAlias");
}

#[test]
fn every_artifact_role_has_a_byte_ceiling_and_invalid_hashes_are_refused() {
    for role in NativeArtifactRole::ALL {
        let mut spec = fixture();
        let binding = spec.engines[0]
            .artifacts
            .iter_mut()
            .find(|binding| binding.role == role)
            .unwrap();
        binding.artifact.bytes = role.byte_limit() + 1;
        invalid(spec, "ArtifactRoleBudget");
    }
    for digest in [
        "0".repeat(64),
        "A".repeat(64),
        "a".repeat(63),
        "g".repeat(64),
    ] {
        let mut spec = fixture();
        spec.engines[0].artifacts[0].artifact.sha256 = digest.clone();
        invalid(spec, "InvalidDigest");
        let mut spec = fixture();
        spec.engines[0].profile.expected_backend_sha256 = digest.clone();
        invalid(spec, "InvalidDigest");
        let mut spec = fixture();
        spec.engines[0].profile.expected_encoding_sha256 = digest;
        invalid(spec, "InvalidDigest");
    }
}

#[test]
fn same_integration_profile_rejects_role_id_scope_and_runtime_changes() {
    let mut spec = fixture();
    spec.engines[1].role = NativeEngineRole::Baseline;
    invalid(spec, "DuplicateEngineRole");
    let mut spec = fixture();
    spec.engines[1].engine_id = spec.engines[0].engine_id.clone();
    invalid(spec, "DuplicateEngineId");
    let mut spec = fixture();
    spec.white_order = [NativeEngineRole::Baseline; 2];
    invalid(spec, "InvalidColorOrder");
    let mut spec = fixture();
    spec.strength_eligible = true;
    invalid(spec, "UnsupportedIntegrationScope");
    let mut spec = fixture();
    spec.contract_revision = "0.1.0".into();
    invalid(spec, "UnsupportedContract");
    let mut spec = fixture();
    spec.engines[1].artifacts[1].artifact.bytes += 1;
    invalid(spec, "IntegrationInputMismatch");
    let mut spec = fixture();
    spec.engines[1].profile.expected_backend_sha256 = "f".repeat(64);
    invalid(spec, "IntegrationInputMismatch");
    let mut spec = fixture();
    spec.engines[0].profile.batch_size = 2;
    invalid(spec, "UnsupportedNativeProfile");
    let mut spec = fixture();
    spec.engines[0].profile.fresh_only = false;
    invalid(spec, "UnsupportedNativeProfile");
}

#[test]
fn aggregate_input_output_runtime_and_timeout_bounds_are_checked() {
    let mut spec = fixture();
    spec.budget.max_input_bytes = spec.unique_input_bytes().unwrap() - 1;
    invalid(spec, "NativeInputBudget");
    let mut spec = fixture();
    spec.budget.max_artifact_bytes = spec.unique_input_bytes().unwrap()
        + spec.budget.max_output_bytes
        + spec.budget.max_runtime_bytes
        - 1;
    invalid(spec, "NativeArtifactBudget");
    let mut spec = fixture();
    spec.budget.max_output_bytes = u64::MAX;
    spec.budget.max_artifact_bytes = u64::MAX;
    invalid(spec, "BudgetOverflow");
    let mut spec = fixture();
    spec.budget.max_child_processes = 2;
    invalid(spec, "NativeResourceBudget");
    let mut spec = fixture();
    spec.budget.max_runtime_files = 0;
    invalid(spec, "NativeResourceBudget");
    let mut spec = fixture();
    spec.budget.max_runtime_depth = 1;
    invalid(spec, "NativeResourceBudget");
    let mut spec = fixture();
    spec.budget.address_space_per_process_bytes = 0;
    invalid(spec, "NativeResourceBudget");
    let mut spec = fixture();
    spec.clock.movetime_ms = 0;
    invalid(spec, "NativeMovetimeBudget");
    let mut spec = fixture();
    spec.timeouts.handshake_ms = 0;
    invalid(spec, "NativeTimeoutBudget");
    let mut spec = fixture();
    spec.timeouts.runtime_ms = 86_400_000;
    invalid(spec, "NativeTimeoutBudget");
    let mut overflows = fixture().timeouts;
    overflows.runtime_ms = u64::MAX;
    code(
        overflows.max_supervised_wall_ms().unwrap_err(),
        "BudgetOverflow",
    );
    assert_eq!(fixture().timeouts.max_supervised_wall_ms().unwrap(), 14000);
}

#[test]
fn opening_history_and_fastchess_remainder_are_checked_without_rule_duplication() {
    let mut wrong_history = fixture();
    wrong_history.opening.history = HistoryCompleteness::UnknownPrefix;
    invalid(wrong_history, "HistoryMismatch");
    let mut odd = fixture();
    odd.max_plies = 5;
    invalid(odd, "NativePlyBudget");
    let mut no_search = fixture();
    no_search.max_plies = 2;
    invalid(no_search, "NativePlyBudget");
    let mut malformed = fixture();
    malformed.opening.moves[0] = "a9a1".into();
    invalid(malformed, "MalformedMove");
    let mut unknown_prefix = fixture();
    unknown_prefix.opening.initial = InitialPosition::Fen;
    unknown_prefix.opening.fen = Some("8/8/8/8/8/8/8/8 w - - 0 1".into());
    unknown_prefix.opening.history = HistoryCompleteness::UnknownPrefix;
    // Semantic invalidity belongs to A; the six-field declaration is accepted.
    assert!(unknown_prefix.clone().lock().is_ok());
    unknown_prefix.opening.history = HistoryCompleteness::Complete;
    invalid(unknown_prefix, "HistoryMismatch");
    let mut syntactically_legal_only = fixture();
    syntactically_legal_only.opening.moves[0] = "e2e5".into();
    assert!(
        syntactically_legal_only.lock().is_ok(),
        "declaration gate must not claim A legality"
    );
}
