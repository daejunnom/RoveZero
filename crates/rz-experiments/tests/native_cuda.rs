use rz_experiments::*;
use serde_json::{Value, json};

// Synthetic declarations only: no native/model bytes, CUDA device, provider,
// chess legality, clock or OS resource-enforcement evidence is created here.
fn artifact(path: &str, bytes: u64) -> ArtifactRef {
    ArtifactRef {
        path: path.into(),
        sha256: "a".repeat(64),
        bytes,
        source: "https://example.org/rovezero-cuda-metadata".into(),
        license: "Synthetic metadata fixture only".into(),
    }
}

fn bundle() -> CudaBundleBindingV1 {
    let mut bundle = CudaBundleBindingV1 {
        manifest: artifact("runtime/bundle.json", 128),
        canonical_sha256: "f".repeat(64),
        files: CUDA_BUNDLE_FILENAMES
            .into_iter()
            .map(|(filename, role)| CudaBundleFileBindingV1 {
                role,
                filename: filename.into(),
                artifact: artifact(&format!("runtime/{filename}"), 1),
            })
            .collect(),
    };
    bundle.canonical_sha256 = bundle.canonical_digest().unwrap();
    bundle
}

fn fixture() -> CudaIntegrationPairSpecV1 {
    let cuda_bundle = bundle();
    let core = cuda_bundle
        .file(CudaBundleFileRoleV1::Core, "libonnxruntime.so.1.22.0")
        .unwrap()
        .clone();
    let artifacts = NativeArtifactRole::ALL
        .into_iter()
        .enumerate()
        .map(|(index, role)| NativeArtifactBinding {
            role,
            artifact: if role == NativeArtifactRole::OrtLibrary {
                core.clone()
            } else {
                artifact(&format!("native/input-{index}"), 1)
            },
        })
        .collect();
    let baseline = CudaNativeLaunchSpecV1 {
        role: NativeEngineRole::Baseline,
        engine_id: "cuda-baseline".into(),
        source_commit: "b".repeat(40),
        target: "x86_64-unknown-linux-gnu".into(),
        artifacts,
        profile: NativeCudaProfileV1::fixed("c".repeat(64), "d".repeat(64)),
        cuda_bundle,
    };
    let mut candidate = baseline.clone();
    candidate.role = NativeEngineRole::Candidate;
    candidate.engine_id = "cuda-candidate".into();
    CudaIntegrationPairSpecV1 {
        pilot: None,
        schema_version: 1,
        purpose: NativeCudaIntegrationPurpose::CudaNnIntegration,
        strength_eligible: false,
        contract_revision: "0.1".into(),
        run_id: "cuda-run-1".into(),
        pair_id: "cuda-pair-1".into(),
        engines: [baseline, candidate],
        white_order: [NativeEngineRole::Candidate, NativeEngineRole::Baseline],
        opening: OpeningSpec {
            id: "cuda-opening-1".into(),
            initial: InitialPosition::Startpos,
            fen: None,
            moves: vec!["e2e4".into(), "e7e5".into()],
            history: HistoryCompleteness::Complete,
            history_origin: "startpos-complete-trace".into(),
        },
        opening_artifact: artifact("inputs/opening.pgn", 50),
        runner: ToolIdentity {
            version: "pinned-runner-test-v1".into(),
            source_url: "https://example.org/rovezero-cuda-metadata".into(),
            source_commit: "f".repeat(40),
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
        budget: NativeCudaResourceBudgetV1 {
            max_input_bytes: 4096,
            max_output_bytes: 128 * 1024,
            max_runtime_bytes: 20 * 1024 * 1024,
            max_artifact_bytes: 24 * 1024 * 1024,
            max_child_processes: 3,
            max_runtime_files: 128,
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

fn invalid(spec: CudaIntegrationPairSpecV1, expected: &str) {
    code(spec.validate().unwrap_err(), expected);
    code(spec.lock().unwrap_err(), expected);
}

#[test]
fn locked_gpu_declaration_charges_all_inputs_once_and_cannot_authorize_execution() {
    let spec = fixture();
    assert!(spec.engines[0].same_inputs_as(&spec.engines[1]).unwrap());
    assert_eq!(spec.declared_artifacts().len(), 52);
    // Four common files + nineteen libraries + manifest + runner + opening.
    // Core is explicitly bound twice per engine but is charged just once.
    assert_eq!(spec.unique_input_bytes().unwrap(), 301);
    let locked = spec.lock().unwrap();
    assert_eq!(locked.input(), locked.spec());
    assert_eq!(locked.unique_input_bytes().unwrap(), 301);
    let wire = locked.to_json().unwrap();
    let envelope: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(envelope["execution_ready"], false);
    assert_eq!(envelope["domain"], CUDA_NATIVE_LAUNCH_DOMAIN);
    assert_eq!(envelope["input"]["strength_eligible"], false);
    assert_eq!(envelope["input"]["purpose"], "cuda_nn_integration");
    assert_eq!(
        LockedCudaIntegrationPairSpecV1::from_json(&wire)
            .unwrap()
            .sha256(),
        locked.sha256()
    );
    assert!(matches!(
        IntegrationPairSpecV1::from_json(&serde_json::to_string(locked.input()).unwrap()),
        Err(ManifestError::Parse(_))
    ));
}

#[test]
fn bundle_binary_codec_matches_independently_encoded_c_golden_vector() {
    // Independently encoded using an ordinal filename sort, fixed bytes=1 and
    // 32 decoded 0xaa bytes per entry. The expected digest is not computed by
    // the production implementation and is unrelated to real selected assets.
    assert_eq!(
        bundle().canonical_digest().unwrap(),
        "90518de00ab2a9840d12a3f460b4d049262db62e954e1e6f3b6344eadea9e203"
    );
    let mut reversed = bundle();
    reversed.files.reverse();
    assert_eq!(
        reversed.canonical_digest().unwrap(),
        bundle().canonical_digest().unwrap()
    );
    assert_eq!(reversed.total_bytes().unwrap(), 19);
}

#[test]
fn canonical_lock_normalizes_declarations_but_binds_execution_order() {
    let spec = fixture();
    let expected = spec.clone().lock().unwrap();
    let mut reordered = spec.clone();
    reordered.engines.reverse();
    for engine in &mut reordered.engines {
        engine.artifacts.reverse();
        engine.cuda_bundle.files.reverse();
    }
    assert_eq!(reordered.lock().unwrap().sha256(), expected.sha256());
    let mut one_reordered = spec.clone();
    one_reordered.engines[1].artifacts.reverse();
    one_reordered.engines[1].cuda_bundle.files.reverse();
    assert!(
        one_reordered.engines[0]
            .same_inputs_as(&one_reordered.engines[1])
            .unwrap()
    );
    assert_eq!(one_reordered.lock().unwrap().sha256(), expected.sha256());
    let mut changed_order = spec;
    changed_order.white_order.reverse();
    assert_ne!(changed_order.lock().unwrap().sha256(), expected.sha256());
}

#[test]
fn bundle_rejects_missing_duplicate_foreign_and_misassigned_files() {
    let mut missing = bundle();
    missing.files.pop();
    code(missing.validate().unwrap_err(), "CudaBundleFileCount");
    let mut extra = bundle();
    extra.files.push(extra.files[0].clone());
    code(extra.validate().unwrap_err(), "CudaBundleFileCount");
    let mut duplicate = bundle();
    duplicate.files[18] = duplicate.files[17].clone();
    code(duplicate.validate().unwrap_err(), "DuplicateCudaBundleFile");
    let mut wrong_role = bundle();
    wrong_role.files[3].role = CudaBundleFileRoleV1::Core;
    code(
        wrong_role.validate().unwrap_err(),
        "UnsupportedCudaBundleFile",
    );
    let mut foreign = bundle();
    foreign.files[0].filename = "libonnxruntime.so.1.23.0".into();
    code(foreign.validate().unwrap_err(), "UnsupportedCudaBundleFile");
}

#[test]
fn bundle_paths_are_exact_safe_siblings_of_a_distinct_manifest() {
    for path in [
        "other/libcudart.so.12",
        "runtime/wrong-name.so",
        "../libcudart.so.12",
        "/runtime/libcudart.so.12",
        "runtime\\libcudart.so.12",
        "runtime/%2f-libcudart.so.12",
    ] {
        let mut changed = bundle();
        changed.files[3].artifact.path = path.into();
        assert!(changed.validate().is_err(), "accepted {path}");
    }
    let mut manifest_parent = bundle();
    manifest_parent.manifest.path = "other/bundle.json".into();
    code(
        manifest_parent.validate().unwrap_err(),
        "CudaBundleSiblingPath",
    );
    let mut manifest_alias = bundle();
    manifest_alias.manifest.path = manifest_alias.files[0].artifact.path.clone();
    code(
        manifest_alias.validate().unwrap_err(),
        "CudaBundleSiblingPath",
    );
    let mut root_siblings = bundle();
    root_siblings.manifest.path = "bundle.json".into();
    for file in &mut root_siblings.files {
        file.artifact.path = file.filename.clone();
    }
    assert!(root_siblings.validate().is_ok());
}

#[test]
fn raw_manifest_binding_and_canonical_library_identity_are_distinct() {
    let original = fixture();
    let expected = original.clone().lock().unwrap();
    let mut changed = original.clone();
    for engine in &mut changed.engines {
        engine.cuda_bundle.manifest.sha256 = "b".repeat(64);
    }
    assert_eq!(
        changed.engines[0].cuda_bundle.canonical_sha256,
        original.engines[0].cuda_bundle.canonical_sha256
    );
    assert_ne!(changed.lock().unwrap().sha256(), expected.sha256());
    let mut one_manifest = original.clone();
    one_manifest.engines[1].cuda_bundle.manifest.sha256 = "b".repeat(64);
    invalid(one_manifest, "IntegrationInputMismatch");
    let mut digest = original.clone();
    digest.engines[0].cuda_bundle.canonical_sha256 = "f".repeat(64);
    invalid(digest, "CudaBundleDigestMismatch");
    let mut contents = original;
    contents.engines[0].cuda_bundle.files[3].artifact.bytes = 2;
    invalid(contents, "CudaBundleDigestMismatch");
}

#[test]
fn entire_provenance_and_core_binding_must_match_before_deduplication() {
    let mut spec = fixture();
    spec.engines[1].cuda_bundle.files[3].artifact.license = "Different provenance".into();
    // The C canonical codec has no license field, but complete input identity does.
    assert!(spec.engines[1].cuda_bundle.validate().is_ok());
    code(
        spec.unique_input_bytes().unwrap_err(),
        "ArtifactIdentityConflict",
    );
    invalid(spec, "IntegrationInputMismatch");
    let mut spec = fixture();
    let core = spec.engines[0]
        .artifacts
        .iter_mut()
        .find(|binding| binding.role == NativeArtifactRole::OrtLibrary)
        .unwrap();
    core.artifact.license = "Different Core provenance".into();
    invalid(spec, "CudaCoreBindingMismatch");
    let mut spec = fixture();
    spec.engines[0].artifacts[0].artifact = spec.engines[0].cuda_bundle.manifest.clone();
    invalid(spec, "ArtifactRoleAlias");
    let mut spec = fixture();
    spec.opening_artifact = spec.engines[0].cuda_bundle.files[3].artifact.clone();
    invalid(spec, "ArtifactRoleAlias");
}

#[test]
fn each_bundle_allocation_and_digest_has_a_closed_boundary() {
    let mut manifest = bundle();
    manifest.manifest.bytes = MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES as u64 + 1;
    code(manifest.validate().unwrap_err(), "CudaBundleManifestBudget");
    let mut single = bundle();
    single.files[0].artifact.bytes = CUDA_BUNDLE_FILE_MAX_BYTES;
    single.canonical_sha256 = single.canonical_digest().unwrap();
    assert!(single.validate().is_ok());
    single.files[0].artifact.bytes += 1;
    code(single.validate().unwrap_err(), "CudaBundleFileBudget");
    let mut total = bundle();
    for file in &mut total.files[..3] {
        file.artifact.bytes = CUDA_BUNDLE_FILE_MAX_BYTES;
    }
    total.files[3].artifact.bytes = CUDA_BUNDLE_FILE_MAX_BYTES - 15;
    total.canonical_sha256 = total.canonical_digest().unwrap();
    assert_eq!(total.total_bytes().unwrap(), CUDA_BUNDLE_MAX_BYTES);
    total.files[3].artifact.bytes += 1;
    code(total.validate().unwrap_err(), "CudaBundleBudget");
    for hash in [
        "0".repeat(64),
        "A".repeat(64),
        "a".repeat(63),
        "g".repeat(64),
    ] {
        let mut changed = bundle();
        changed.files[3].artifact.sha256 = hash;
        code(changed.validate().unwrap_err(), "InvalidDigest");
    }
    let mut zero = bundle();
    zero.files[3].artifact.bytes = 0;
    assert!(zero.validate().is_err());
}

#[test]
fn cuda_core_ceiling_is_separate_from_the_unchanged_cpu_role_ceiling() {
    assert_eq!(
        NativeArtifactRole::OrtLibrary.byte_limit(),
        512 * 1024 * 1024
    );
    let mut launch = fixture().engines[0].clone();
    launch.cuda_bundle.files[0].artifact.bytes = CUDA_BUNDLE_FILE_MAX_BYTES;
    launch.cuda_bundle.canonical_sha256 = launch.cuda_bundle.canonical_digest().unwrap();
    let core = launch.cuda_bundle.files[0].artifact.clone();
    launch
        .artifacts
        .iter_mut()
        .find(|binding| binding.role == NativeArtifactRole::OrtLibrary)
        .unwrap()
        .artifact = core;
    assert!(launch.validate().is_ok());
}

#[test]
fn gpu_profile_is_closed_to_the_fixed_recipe_and_explicit_identities() {
    for (pointer, replacement) in [
        ("/engines/0/profile/device_id", json!(1)),
        (
            "/engines/0/profile/arena_bytes",
            json!(CUDA_NATIVE_ARENA_BYTES - 1),
        ),
        ("/engines/0/profile/tf32", json!(true)),
        ("/engines/0/profile/batch_size", json!(2)),
        ("/engines/0/profile/intra_threads", json!(2)),
        ("/engines/0/profile/search_workers", json!(2)),
        ("/engines/0/profile/fresh_only", json!(false)),
        ("/engines/0/profile/full_steps", json!(2)),
    ] {
        let mut value = serde_json::to_value(fixture()).unwrap();
        *value.pointer_mut(pointer).unwrap() = replacement;
        code(
            CudaIntegrationPairSpecV1::from_json(&value.to_string()).unwrap_err(),
            "UnsupportedNativeProfile",
        );
    }
    for field in ["expected_backend_sha256", "expected_encoding_sha256"] {
        let mut value = serde_json::to_value(fixture()).unwrap();
        value["engines"][0]["profile"][field] = json!("0".repeat(64));
        code(
            CudaIntegrationPairSpecV1::from_json(&value.to_string()).unwrap_err(),
            "InvalidDigest",
        );
        let mut value = serde_json::to_value(fixture()).unwrap();
        value["engines"][1]["profile"][field] = json!("f".repeat(64));
        code(
            CudaIntegrationPairSpecV1::from_json(&value.to_string()).unwrap_err(),
            "IntegrationInputMismatch",
        );
    }
}

#[test]
fn all_gpu_resource_reservations_and_finite_ceiling_fields_are_checked() {
    let mut spec = fixture();
    spec.budget.max_input_bytes = spec.unique_input_bytes().unwrap() - 1;
    invalid(spec, "NativeInputBudget");
    let mut spec = fixture();
    spec.budget.max_artifact_bytes =
        spec.budget.max_input_bytes + spec.budget.max_runtime_bytes + spec.budget.max_output_bytes
            - 1;
    invalid(spec, "NativeArtifactBudget");
    let mut spec = fixture();
    // Four starts retain the full bundle, a bounded 4MiB placement trace and
    // 512KiB startup/final receipts each: four bundles plus 18MiB.
    spec.budget.max_runtime_bytes =
        4 * spec.engines[0].cuda_bundle.total_bytes().unwrap() + 18 * 1024 * 1024 - 1;
    invalid(spec, "CudaRuntimeReservation");
    let mut exact = fixture();
    exact.budget.max_runtime_bytes =
        4 * exact.engines[0].cuda_bundle.total_bytes().unwrap() + 18 * 1024 * 1024;
    exact.budget.max_artifact_bytes = exact.budget.max_input_bytes
        + exact.budget.max_runtime_bytes
        + exact.budget.max_output_bytes;
    assert!(exact.clone().lock().is_ok());
    exact.budget.max_artifact_bytes -= 1;
    invalid(exact, "NativeArtifactBudget");
    for (field, replacement, expected) in [
        (
            "max_input_bytes",
            CUDA_BUNDLE_MAX_BYTES + 128 * 1024 * 1024 + 1,
            "NativeInputBudget",
        ),
        (
            "max_output_bytes",
            64 * 1024 * 1024 + 1,
            "NativeResourceBudget",
        ),
        (
            "max_runtime_bytes",
            16 * CUDA_BUNDLE_FILE_MAX_BYTES + 64 * 1024 * 1024 + 1,
            "NativeResourceBudget",
        ),
        (
            "max_artifact_bytes",
            21 * CUDA_BUNDLE_FILE_MAX_BYTES + 1,
            "NativeResourceBudget",
        ),
        (
            "address_space_per_process_bytes",
            128 * CUDA_BUNDLE_FILE_MAX_BYTES + 1,
            "NativeResourceBudget",
        ),
        ("max_child_processes", 2, "NativeResourceBudget"),
        ("max_child_processes", 17, "NativeResourceBudget"),
        ("max_runtime_files", 127, "NativeResourceBudget"),
        ("max_runtime_files", 1025, "NativeResourceBudget"),
        ("max_runtime_depth", 3, "NativeResourceBudget"),
        ("max_runtime_depth", 9, "NativeResourceBudget"),
    ] {
        let mut changed = serde_json::to_value(fixture()).unwrap();
        changed["budget"][field] = json!(replacement);
        code(
            CudaIntegrationPairSpecV1::from_json(&changed.to_string()).unwrap_err(),
            expected,
        );
    }
}

#[test]
fn wire_reader_rejects_duplicates_unknown_fields_foreign_profiles_and_lock_tampering() {
    let raw = serde_json::to_string(&fixture()).unwrap();
    let duplicated = raw.replacen("\"tf32\":false", "\"tf32\":false,\"tf32\":false", 1);
    assert_ne!(raw, duplicated);
    assert!(matches!(
        CudaIntegrationPairSpecV1::from_json(&duplicated),
        Err(ManifestError::Parse(_))
    ));
    let mut trace_pin = serde_json::to_value(fixture()).unwrap();
    trace_pin["engines"][0]["profile"]["expected_cuda_profile_sha256"] = json!("e".repeat(64));
    assert!(matches!(
        CudaIntegrationPairSpecV1::from_json(&trace_pin.to_string()),
        Err(ManifestError::Parse(_))
    ));
    for pointer in [
        "",
        "/engines/0",
        "/engines/0/profile",
        "/engines/0/cuda_bundle",
        "/engines/0/cuda_bundle/files/0",
        "/engines/0/cuda_bundle/files/0/artifact",
        "/budget",
    ] {
        let mut value = serde_json::to_value(fixture()).unwrap();
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("extra".into(), json!(true));
        assert!(matches!(
            CudaIntegrationPairSpecV1::from_json(&value.to_string()),
            Err(ManifestError::Parse(_))
        ));
    }
    for (pointer, replacement) in [
        ("/purpose", json!("cpu_nn_integration")),
        ("/engines/0/profile/provider", json!("onnx_runtime_cpu")),
        ("/engines/0/profile/precision", json!("fp16")),
        ("/engines/0/profile/history_fill", json!("repeat_oldest")),
    ] {
        let mut value = serde_json::to_value(fixture()).unwrap();
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(matches!(
            CudaIntegrationPairSpecV1::from_json(&value.to_string()),
            Err(ManifestError::Parse(_))
        ));
    }
    let wire = fixture().lock().unwrap().to_json().unwrap();
    for (key, replacement) in [
        ("execution_ready", json!(true)),
        ("domain", json!(NATIVE_LAUNCH_DOMAIN)),
        ("canonicalization", json!(NATIVE_LAUNCH_CANONICALIZATION)),
        ("lock_version", json!(2)),
    ] {
        let mut value: Value = serde_json::from_str(&wire).unwrap();
        value[key] = replacement;
        code(
            LockedCudaIntegrationPairSpecV1::from_json(&value.to_string()).unwrap_err(),
            "UnsupportedNativeLock",
        );
    }
    let mut value: Value = serde_json::from_str(&wire).unwrap();
    value["input"]["clock"]["movetime_ms"] = json!(101);
    code(
        LockedCudaIntegrationPairSpecV1::from_json(&value.to_string()).unwrap_err(),
        "NativeLockDigestMismatch",
    );
}

#[test]
fn bounded_wire_and_opening_syntax_remain_separate_from_chess_and_cuda_evidence() {
    let raw = serde_json::to_string(&fixture()).unwrap();
    let mut exact = raw.clone();
    exact.push_str(&" ".repeat(MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES - raw.len()));
    assert!(CudaIntegrationPairSpecV1::from_json(&exact).is_ok());
    exact.push(' ');
    code(
        CudaIntegrationPairSpecV1::from_json(&exact).unwrap_err(),
        "NativePayloadLimit",
    );
    code(
        LockedCudaIntegrationPairSpecV1::from_json(&exact).unwrap_err(),
        "NativePayloadLimit",
    );
    let mut malformed = fixture();
    malformed.opening.moves[0] = "a9a1".into();
    invalid(malformed, "MalformedMove");
    let mut syntax_only = fixture();
    syntax_only.opening.moves[0] = "e2e5".into();
    assert!(
        syntax_only.lock().is_ok(),
        "declaration must not reimplement A legality"
    );
    let mut history = fixture();
    history.opening.history = HistoryCompleteness::UnknownPrefix;
    invalid(history, "HistoryMismatch");
    let mut fen = fixture();
    fen.opening.initial = InitialPosition::Fen;
    fen.opening.fen = Some("8/8/8/8/8/8/8/8 w - - 0 1".into());
    fen.opening.history = HistoryCompleteness::UnknownPrefix;
    assert!(fen.lock().is_ok());
    let mut timeout = fixture();
    timeout.timeouts.runtime_ms = 86_400_000;
    invalid(timeout, "NativeTimeoutBudget");
    let mut odd = fixture();
    odd.max_plies = 5;
    invalid(odd, "NativePlyBudget");
    let mut strength = fixture();
    strength.strength_eligible = true;
    invalid(strength, "UnsupportedIntegrationScope");
}

#[test]
fn full_export_envelope_must_fit_even_if_embedded_input_and_canonical_payload_fit() {
    let mut near_cap = fixture();
    near_cap.opening.moves = vec!["e2e4".into(); 4092];
    near_cap.max_plies = 4094;
    // Reduce one provenance field before sizing the complete declaration.
    for engine in &mut near_cap.engines {
        engine.cuda_bundle.manifest.license = "L".into();
    }
    let base = serde_json::to_string(&near_cap).unwrap().len();
    let canonical_overhead = json!({"domain": CUDA_NATIVE_LAUNCH_DOMAIN,
        "schema_version": CUDA_NATIVE_LAUNCH_SCHEMA_VERSION, "execution_ready": false,
        "input": null})
    .to_string()
    .len()
        - "null".len();
    let small = fixture();
    let envelope_overhead = small.clone().lock().unwrap().to_json().unwrap().len()
        - serde_json::to_string(&small).unwrap().len();
    assert!(envelope_overhead > canonical_overhead + 32);
    let target = MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES - canonical_overhead - 16;
    // Use a bundle library provenance field on every entry to approach the cap
    // within the existing 4096-byte ArtifactRef text ceiling.
    // Nineteen file fields and the additional Core artifact field per engine
    // produce forty added bytes per step.
    let padding = (target - base) / 40;
    assert!(padding < 4000);
    for engine in &mut near_cap.engines {
        for file in &mut engine.cuda_bundle.files {
            file.artifact.license.push_str(&"L".repeat(padding));
        }
        let core = engine
            .cuda_bundle
            .file(CudaBundleFileRoleV1::Core, "libonnxruntime.so.1.22.0")
            .unwrap()
            .clone();
        engine
            .artifacts
            .iter_mut()
            .find(|binding| binding.role == NativeArtifactRole::OrtLibrary)
            .unwrap()
            .artifact = core;
    }
    // Fill the small remaining gap through the two manifest provenance fields.
    let current = serde_json::to_string(&near_cap).unwrap().len();
    let remainder = (target - current) / 2;
    for engine in &mut near_cap.engines {
        engine
            .cuda_bundle
            .manifest
            .license
            .push_str(&"L".repeat(remainder));
    }
    let raw = serde_json::to_string(&near_cap).unwrap();
    assert!(raw.len() <= MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES);
    assert!(raw.len() + canonical_overhead <= MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES);
    assert!(raw.len() + envelope_overhead > MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES);
    assert!(near_cap.validate().is_ok());
    code(near_cap.lock().unwrap_err(), "NativePayloadLimit");
}

fn bt4_fixture() -> CudaIntegrationPairSpecV2 {
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["schema_version"] = json!(2);
    for engine in value["engines"].as_array_mut().unwrap() {
        let profile = NativeCudaProfileV2::fixed("c".repeat(64), "d".repeat(64));
        engine["profile"] = serde_json::to_value(profile).unwrap();
        for binding in engine["artifacts"].as_array_mut().unwrap() {
            match binding["role"].as_str().unwrap() {
                "source_weights" => {
                    binding["artifact"]["sha256"] = json!(BT4_NATIVE_SOURCE_SHA256);
                    binding["artifact"]["bytes"] = json!(BT4_NATIVE_SOURCE_BYTES);
                }
                "onnx" => binding["artifact"]["bytes"] = json!(741_000_000_u64),
                _ => (),
            }
        }
    }
    value["budget"]["max_input_bytes"] = json!(2 * 1024 * 1024 * 1024_u64);
    value["budget"]["max_runtime_bytes"] = json!(19 * 1024 * 1024_u64);
    value["budget"]["max_artifact_bytes"] = json!(3 * 1024 * 1024 * 1024_u64);
    CudaIntegrationPairSpecV2::from_json(&value.to_string()).unwrap()
}

#[test]
fn bt4_v2_has_separate_closed_wire_and_lock_domains() {
    let spec = bt4_fixture();
    let lock = spec.clone().lock().unwrap();
    let wire = lock.to_json().unwrap();
    assert_eq!(
        LockedCudaIntegrationPairSpecV2::from_json(&wire)
            .unwrap()
            .sha256(),
        lock.sha256()
    );
    assert!(LockedCudaIntegrationPairSpecV1::from_json(&wire).is_err());
    assert!(CudaIntegrationPairSpecV1::from_json(&serde_json::to_string(&spec).unwrap()).is_err());
    assert!(
        LockedCudaIntegrationPairSpecV2::from_json(&fixture().lock().unwrap().to_json().unwrap())
            .is_err()
    );
    let mut envelope: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(envelope["domain"], CUDA_NATIVE_LAUNCH_V2_DOMAIN);
    assert_eq!(envelope["execution_ready"], false);
    envelope["input"]["engines"][0]["profile"]["search"]["simulations"] = json!(128);
    envelope["input"]["engines"][1]["profile"]["search"]["simulations"] = json!(128);
    code(
        LockedCudaIntegrationPairSpecV2::from_json(&envelope.to_string()).unwrap_err(),
        "NativeLockDigestMismatch",
    );
}

#[test]
fn bt4_identity_and_bounds_do_not_relax_maia_v1() {
    assert_eq!(NativeArtifactRole::Onnx.byte_limit(), 16 * 1024 * 1024);
    assert_eq!(
        NativeArtifactRole::SourceWeights.byte_limit(),
        4 * 1024 * 1024
    );
    let mut engine = bt4_fixture().engines[0].clone();
    let source = engine
        .artifacts
        .iter_mut()
        .find(|a| a.role == NativeArtifactRole::SourceWeights)
        .unwrap();
    source.artifact.sha256 = "a".repeat(64);
    code(
        engine.validate().unwrap_err(),
        "NativeModelIdentityMismatch",
    );
    let mut engine = bt4_fixture().engines[0].clone();
    let onnx = engine
        .artifacts
        .iter_mut()
        .find(|a| a.role == NativeArtifactRole::Onnx)
        .unwrap();
    onnx.artifact.bytes = BT4_NATIVE_ONNX_LIMIT;
    assert!(engine.validate().is_ok());
    engine
        .artifacts
        .iter_mut()
        .find(|a| a.role == NativeArtifactRole::Onnx)
        .unwrap()
        .artifact
        .bytes += 1;
    code(engine.validate().unwrap_err(), "ArtifactRoleBudget");
}

#[test]
fn bt4_v2_search_runtime_and_a_a_identity_are_locked() {
    for (pointer, value, expected) in [
        (
            "/engines/0/profile/runtime/arena_bytes",
            json!(CUDA_NATIVE_ARENA_BYTES),
            "UnsupportedNativeProfile",
        ),
        (
            "/engines/0/profile/runtime/batch_size",
            json!(16),
            "UnsupportedNativeProfile",
        ),
        (
            "/engines/0/profile/search/simulations",
            json!(4097),
            "UnsupportedNativeSearch",
        ),
        (
            "/engines/0/profile/search/simulations",
            json!(0),
            "UnsupportedNativeSearch",
        ),
        (
            "/engines/0/profile/search/policy_temperature_milli",
            json!(800),
            "UnsupportedNativeSearch",
        ),
        (
            "/engines/0/profile/search/raw_cache",
            json!(true),
            "UnsupportedNativeSearch",
        ),
        (
            "/engines/0/profile/search/final_selection",
            json!("exact_terminal"),
            "IntegrationInputMismatch",
        ),
    ] {
        let mut spec = serde_json::to_value(bt4_fixture()).unwrap();
        *spec.pointer_mut(pointer).unwrap() = value;
        code(
            CudaIntegrationPairSpecV2::from_json(&spec.to_string()).unwrap_err(),
            expected,
        );
    }
    let mut spec = bt4_fixture();
    for engine in &mut spec.engines {
        engine.profile.search.final_selection = NativeFinalSelectionV2::ExactTerminal;
    }
    assert!(spec.lock().is_ok());
}

#[test]
fn bt4_shared_storage_reservation_covers_four_three_file_receipt_sets() {
    let mut spec = bt4_fixture();
    assert_eq!(spec.budget.max_runtime_bytes, 19 * 1024 * 1024);
    assert!(spec.clone().lock().is_ok());
    spec.budget.max_runtime_bytes -= 1;
    code(spec.validate().unwrap_err(), "CudaRuntimeReservation");
    let mut value = serde_json::to_value(bt4_fixture()).unwrap();
    value["engines"][0]["profile"]["arbitrary_option"] = json!(true);
    assert!(matches!(
        CudaIntegrationPairSpecV2::from_json(&value.to_string()),
        Err(ManifestError::Parse(_))
    ));
}

fn pilot_fixture() -> CudaSearchPilotPairSpecV3 {
    let mut value = serde_json::to_value(bt4_fixture()).unwrap();
    value["schema_version"] = json!(3);
    value["purpose"] = json!("cuda_search_pilot");
    value["clock"] = json!({"base_ms":30000,"increment_ms":100});
    value["max_plies"] = json!(256);
    value["white_order"] = json!(["baseline", "candidate"]);
    value["timeouts"]["runtime_ms"] = json!(600000);
    value["engines"][1]["profile"]["search"]["final_selection"] = json!("exact_terminal");
    let mut patch = artifact("runner/clock.patch", FASTCHESS_CLOCK_PATCH_BYTES);
    patch.sha256 = FASTCHESS_CLOCK_PATCH_SHA256.into();
    value["runner"]["source_url"] = json!("https://github.com/Disservin/fastchess");
    value["runner"]["binary"]["source"] = value["runner"]["source_url"].clone();
    value["runner"]["source_commit"] = json!("f618e34540f94f4719ad3817950618dabe441318");
    value["runner"]["dirty"] = json!(true);
    value["runner"]["dirty_patch"] = serde_json::to_value(patch).unwrap();
    value["pilot"] = serde_json::to_value(NativeSearchPilotProtocolV3 {
        pair_ordinal: 0,
        total_pairs: 16,
        opening_cohort: artifact("inputs/cohort.json", 1024),
        total_wall_ms: 7_200_000,
        statistics: NativePilotStatistics::FixedPairedHoeffding95,
        claim_policy: ClaimPolicy::AutomaticAcceptance,
        engine_failure: OutcomePolicy::Loss,
        cutoff: OutcomePolicy::Incomplete,
    })
    .unwrap();
    CudaSearchPilotPairSpecV3::from_json(&value.to_string()).unwrap()
}

#[test]
fn pilot_v3_is_separate_from_both_historical_integration_domains() {
    let lock = pilot_fixture().lock().unwrap();
    let wire = lock.to_json().unwrap();
    assert_eq!(
        LockedCudaSearchPilotPairSpecV3::from_json(&wire)
            .unwrap()
            .sha256(),
        lock.sha256()
    );
    assert!(LockedCudaIntegrationPairSpecV2::from_json(&wire).is_err());
    assert!(LockedCudaIntegrationPairSpecV1::from_json(&wire).is_err());
    assert!(
        LockedCudaSearchPilotPairSpecV3::from_json(
            &bt4_fixture().lock().unwrap().to_json().unwrap()
        )
        .is_err()
    );
    let mut envelope: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(envelope["domain"], CUDA_SEARCH_PILOT_V3_DOMAIN);
    assert_eq!(envelope["execution_ready"], false);
    envelope["input"]["pilot"]["pair_ordinal"] = json!(2);
    code(
        LockedCudaSearchPilotPairSpecV3::from_json(&envelope.to_string()).unwrap_err(),
        "NativeLockDigestMismatch",
    );
    let mut legacy = serde_json::to_value(bt4_fixture()).unwrap();
    legacy["purpose"] = json!("cuda_search_pilot");
    assert!(CudaIntegrationPairSpecV2::from_json(&legacy.to_string()).is_err());
}

#[test]
fn pilot_allows_only_the_predeclared_final_selection_delta() {
    let mut value = serde_json::to_value(pilot_fixture()).unwrap();
    let original = value.clone();
    for (path, replacement) in [
        (
            "/engines/0/profile/search/final_selection",
            json!("exact_terminal"),
        ),
        ("/engines/1/profile/search/final_selection", json!("visits")),
        ("/engines/0/profile/search/simulations", json!(128)),
        (
            "/engines/1/profile/runtime/expected_encoding_sha256",
            json!("e".repeat(64)),
        ),
        (
            "/engines/1/artifacts/0/artifact/license",
            json!("Changed provenance"),
        ),
    ] {
        value = original.clone();
        *value.pointer_mut(path).unwrap() = replacement;
        assert!(
            CudaSearchPilotPairSpecV3::from_json(&value.to_string()).is_err(),
            "{path}"
        );
    }
    let spec = pilot_fixture();
    assert!(!spec.engines[0].same_inputs_as(&spec.engines[1]).unwrap());
    assert!(
        spec.engines[0]
            .same_pinned_inputs_as(&spec.engines[1])
            .unwrap()
    );
}

#[test]
fn pilot_clock_failure_sample_and_runner_policy_are_locked() {
    let original = serde_json::to_value(pilot_fixture()).unwrap();
    for (path, value) in [
        ("/clock/base_ms", json!(30100)),
        ("/clock/increment_ms", json!(0)),
        ("/pilot/total_pairs", json!(8)),
        ("/pilot/pair_ordinal", json!(16)),
        ("/pilot/total_wall_ms", json!(7_200_001)),
        ("/max_plies", json!(258)),
        ("/pilot/claim_policy", json!("explicit_claim")),
        ("/pilot/engine_failure", json!("incomplete")),
        ("/pilot/cutoff", json!("draw")),
        ("/runner/dirty", json!(false)),
        ("/runner/dirty_patch/sha256", json!("a".repeat(64))),
        (
            "/runner/dirty_patch/bytes",
            json!(FASTCHESS_CLOCK_PATCH_BYTES + 1),
        ),
        ("/strength_eligible", json!(true)),
    ] {
        let mut altered = original.clone();
        *altered.pointer_mut(path).unwrap() = value;
        assert!(
            CudaSearchPilotPairSpecV3::from_json(&altered.to_string()).is_err(),
            "{path}"
        );
    }
    let mut missing = original.clone();
    missing.as_object_mut().unwrap().remove("pilot");
    assert!(CudaSearchPilotPairSpecV3::from_json(&missing.to_string()).is_err());
    let mut extra = original.clone();
    extra["clock"]["timeout_grace_ms"] = json!(100);
    assert!(CudaSearchPilotPairSpecV3::from_json(&extra.to_string()).is_err());
    let patch = include_bytes!("../../../experiments/baselines/fastchess-clock-v1.patch");
    use sha2::{Digest, Sha256};
    assert_eq!(patch.len() as u64, FASTCHESS_CLOCK_PATCH_BYTES);
    assert_eq!(
        format!("{:x}", Sha256::digest(patch)),
        FASTCHESS_CLOCK_PATCH_SHA256
    );
}
