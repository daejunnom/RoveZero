#![cfg(all(feature = "native-cuda", target_os = "linux"))]
#![recursion_limit = "256"]

use rz_arena::{
    validate_cuda_bundle_manifest_fields, validate_cuda_placement_trace_fields,
    validate_native_cuda_process_exit_trace, validate_native_cuda_provider_record_fields,
};
use rz_experiments::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

// Synthetic public wire fixtures only. No GPU, model, UCI child or physical
// inference/drain evidence is produced by these tests.
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn exit_trace(pid: u32, status: &str) -> String {
    format!(
        "[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- Process with pid: {pid} terminated with status: {status}\n"
    )
}

fn four_exit_traces() -> String {
    // Synthetic zero statuses using the captured fixed renderer. No child,
    // neural computation, physical drain or Rules evidence is produced here.
    [402, 413, 447, 455]
        .map(|pid| exit_trace(pid, "0"))
        .concat()
}

#[test]
fn cuda_process_exit_trace_requires_four_distinct_validated_native_pids() {
    let trace = four_exit_traces();
    validate_native_cuda_process_exit_trace(trace.as_bytes(), &[455, 402, 447, 413]).unwrap();
    for expected in [
        vec![402, 413, 447],
        vec![402, 413, 447, 447],
        vec![402, 413, 447, 0],
        vec![402, 413, 447, u32::MAX],
    ] {
        assert!(validate_native_cuda_process_exit_trace(trace.as_bytes(), &expected).is_err());
    }
}

#[test]
fn cuda_process_exit_trace_rejects_partial_duplicate_and_foreign_closure() {
    let full = four_exit_traces();
    let cases = [
        full.replace(&exit_trace(455, "0"), ""),
        format!("{full}{}", exit_trace(413, "0")),
        full.replace(&exit_trace(455, "0"), &exit_trace(456, "0")),
        format!("{full}{}", exit_trace(456, "0")),
        full.trim_end_matches('\n').into(),
    ];
    for trace in cases {
        assert!(
            validate_native_cuda_process_exit_trace(trace.as_bytes(), &[402, 413, 447, 455])
                .is_err()
        );
    }
}

#[test]
fn cuda_process_exit_trace_rejects_late_abort_and_unknown_status() {
    let full = four_exit_traces();
    // First failed pair's numeric raw SIGABRT status must reject even when a
    // separately validated native termination record claimed success.
    for status in ["134", "256", "9", "-1", "00", "+0", "0 extra", ""] {
        let trace = full.replace(&exit_trace(413, "0"), &exit_trace(413, status));
        assert!(
            validate_native_cuda_process_exit_trace(trace.as_bytes(), &[402, 413, 447, 455])
                .is_err()
        );
    }
}

#[test]
fn cuda_process_exit_trace_requires_closed_renderer_and_bounds() {
    let full = four_exit_traces();
    let cases = [
        full.replace("[TRACE ]", "[TRACE]"),
        full.replace("10:51:44.181984", "24:51:44.181984"),
        full.replace("10:51:44.181984", "10:60:44.181984"),
        full.replace("     137433152747200", "    137433152747200"),
        full.replace("pid: 413", "pid: 0413"),
        full.replace("pid: 413", "pid: 4294967296"),
        format!(
            "{full}{}",
            exit_trace(456, "0").replace("pid: 456", "pid 456")
        ),
        format!(
            "{full}[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- Force terminating process with pid: 413 0\n"
        ),
        format!("{full}{}\n", "x".repeat(4097)),
        format!("{full}{}", "\n".repeat(131_073)),
        String::new(),
    ];
    for trace in cases {
        assert!(
            validate_native_cuda_process_exit_trace(trace.as_bytes(), &[402, 413, 447, 455])
                .is_err()
        );
    }
}

#[test]
fn cuda_process_exit_trace_does_not_accept_engine_quoted_runner_records() {
    let quoted = four_exit_traces().lines().map(|line| {
        format!("[Engine] [10:51:44.181984] <     137433152747200> cuda-native-baseline ---> {line}\n")
    }).collect::<String>();
    assert!(
        validate_native_cuda_process_exit_trace(quoted.as_bytes(), &[402, 413, 447, 455]).is_err()
    );
    let trace = format!("{}{quoted}", four_exit_traces());
    validate_native_cuda_process_exit_trace(trace.as_bytes(), &[402, 413, 447, 455]).unwrap();
}
fn artifact(path: &str, digest: &str, bytes: u64) -> ArtifactRef {
    ArtifactRef {
        path: path.into(),
        sha256: digest.into(),
        bytes,
        source: "https://example.org/rovezero-cuda-wire".into(),
        license: "Synthetic wire fixture only".into(),
    }
}
fn fixture() -> (CudaNativeLaunchSpecV1, Value, Value) {
    let mut bundle = CudaBundleBindingV1 {
        manifest: artifact("runtime/bundle.json", &"f".repeat(64), 1),
        canonical_sha256: "f".repeat(64),
        files: CUDA_BUNDLE_FILENAMES
            .into_iter()
            .map(|(filename, role)| CudaBundleFileBindingV1 {
                role,
                filename: filename.into(),
                artifact: artifact(&format!("runtime/{filename}"), &"4".repeat(64), 14),
            })
            .collect(),
    };
    bundle.canonical_sha256 = bundle.canonical_digest().unwrap();
    let bundle_wire = json!({"schema_version":1,"files":bundle.files.iter().map(|file|json!({"role":file.role,
        "filename":file.filename,"bytes":file.artifact.bytes,"sha256":file.artifact.sha256})).collect::<Vec<_>>()});
    let manifest = serde_json::to_vec(&bundle_wire).unwrap();
    bundle.manifest.sha256 = sha(&manifest);
    bundle.manifest.bytes = manifest.len() as u64;
    let core = bundle
        .file(CudaBundleFileRoleV1::Core, "libonnxruntime.so.1.22.0")
        .unwrap()
        .clone();
    let engine = CudaNativeLaunchSpecV1 {
        role: NativeEngineRole::Baseline,
        engine_id: "cuda-baseline".into(),
        source_commit: "a".repeat(40),
        target: "x86_64-unknown-linux-gnu".into(),
        artifacts: NativeArtifactRole::ALL
            .into_iter()
            .enumerate()
            .map(|(i, role)| NativeArtifactBinding {
                role,
                artifact: if role == NativeArtifactRole::OrtLibrary {
                    core.clone()
                } else {
                    artifact(
                        &format!("native/input-{i}"),
                        &if i == 0 {
                            "9".repeat(64)
                        } else {
                            format!("{i:x}").repeat(64)
                        },
                        10 + i as u64,
                    )
                },
            })
            .collect(),
        profile: NativeCudaProfileV1::fixed("b".repeat(64), "d".repeat(64)),
        cuda_bundle: bundle,
    };
    let model = json!({"owner":1,"slot":1,"generation":1,"manifest_sha256":"3".repeat(64)});
    let encoding = json!({"owner":1,"slot":2,"generation":1,"manifest_sha256":"d".repeat(64)});
    let startup = json!({"schema_version":1,"kind":"startup","loaded":true,"process_id":123,"process_run_id":"native-process-123","process_epoch":1,
        "executable":{"sha256":"9".repeat(64),"bytes":10,"identity_source":"linux_proc_self_exe"},
        "profile":{"source_weights_gzip_sha256":"1".repeat(64),"source_weights_protobuf_sha256":"e".repeat(64),"onnx_sha256":"2".repeat(64),"onnx_bytes":12,
            "export_manifest_sha256":"3".repeat(64),"converter_commit":"c".repeat(40),"converter_binary_sha256":"f".repeat(64),"ort_library_sha256":"4".repeat(64),
            "ort_build_info_sha256":"a".repeat(64),"ort_release":"1.22.0","backend_sha256":"b".repeat(64),"model":model,"encoding":encoding,
            "action_map_sha256":"a".repeat(64),"history_policy_sha256":"c".repeat(64),"history_length":8,"contract_major":0,"contract_minor":1,
            "provider":"cuda","precision":"fp32","max_batch_items":1,"intra_threads":1,"max_workers":1,"full_steps":1,"min_steps":1,"max_steps":1,
            "require_full":true,"evaluation_mode":"fresh","history_fill":"no","device_id":0,"arena_bytes":1073741824_u64,"tf32":false,
            "runtime_bundle_manifest_sha256":engine.cuda_bundle.manifest.sha256,"runtime_bundle_sha256":engine.cuda_bundle.canonical_sha256,
            "runtime_bundle_files":bundle_wire["files"],"cuda_profile_sha256":"a".repeat(64),"executed_cuda_nodes":1,"runtime_mapping_verified":true,
            "session_resident_admission":{"host_bytes":0,"device_bytes":1073741824_u64,"pinned_bytes":0,"scope":"declaration_not_measured_vram_or_hardcap"}}});
    let completed = json!({"contract_major":0,"contract_minor":1,"request":{"epoch":1,"sequence":1},"selection":{"epoch":1,"sequence":2},"game":1,"root":1,
        "state_owner":2,"state_revision":0,"state_sha256":"a".repeat(64),"legal_order_sha256":"a".repeat(64),"input_sha256":"a".repeat(64),
        "model":startup["profile"]["model"],"encoding":startup["profile"]["encoding"],"backend_sha256":"b".repeat(64),"precision":"fp32","min_steps":1,
        "max_steps":1,"require_full":true,"execution":{"epoch":1,"sequence":3},"actual":{"precision":"fp32","steps":1,"full":true,"backend_sha256":"b".repeat(64),
            "execution":{"epoch":1,"sequence":3},"provenance":"computed","prior_execution":null}});
    let mut second = completed.clone();
    second["execution"]["sequence"] = json!(4);
    second["actual"]["execution"]["sequence"] = json!(4);
    second["request"]["sequence"] = json!(2);
    second["selection"]["sequence"] = json!(3);
    let term = json!({"schema_version":1,"kind":"termination","process_run_id":"native-process-123","startup":startup,"run_succeeded":true,
        "actual_cuda_inference_observed":true,"actual_rules_search_backup_observed":true,"physical_drain":"confirmed",
        "report":{"model_manifest_sha256":"3".repeat(64),"backend_sha256":"b".repeat(64),"origin":"cuda_onnx","completed_by_runtime":2,
            "first_completed":completed,"last_completed":second,"completed_scope":"process_aggregate_normal_poll_first_last_not_full_root_or_physical_journal",
            "search_root_initializations":{"count":1,"first":{"completed":completed,"traversed_edges":0},"last":{"completed":completed,"traversed_edges":0}},
            "search_non_root_backups":{"count":1,"first":{"completed":second,"traversed_edges":1},"last":{"completed":second,"traversed_edges":1}},
            "search_scope":"unchanged_final_tree_guard_accepted_count_first_last_not_every_root_or_bestmove_proof",
            "observations":{"scheduler_events":2,"scheduler_dropped":0,"scheduler_counter_overflow":false,"delivery_events":2,"delivery_dropped":0,
                "delivery_counter_overflow":false,"drain_discarded_results":0,"scope":"drained_event_counts_and_loss_not_full_journal_or_physical_inference_count"},
            "canceled":{"count":0,"first":null,"last":null},"expired":{"count":0,"first":null,"last":null},"failures":[],"overflow":null,"boundary_error":null,"poison_error":null},
        "original_service_failure":null,"collection_failure":null,"runtime_mapping_failure":null,"retained_owner_and_evidence":false});
    (engine, startup, term)
}
fn audit(engine: &CudaNativeLaunchSpecV1, startup: &Value, term: &Value) -> bool {
    validate_native_cuda_provider_record_fields(
        &serde_json::to_vec(startup).unwrap(),
        &serde_json::to_vec(term).unwrap(),
        engine,
        "native-process-123",
    )
    .is_ok()
}

#[test]
fn cuda_wire_profile_is_separate_and_requires_all_native_file_pins() {
    let (engine, startup, term) = fixture();
    let checked = validate_native_cuda_provider_record_fields(
        &serde_json::to_vec(&startup).unwrap(),
        &serde_json::to_vec(&term).unwrap(),
        &engine,
        "native-process-123",
    )
    .unwrap();
    assert_eq!(checked.completed_by_runtime, 2);
    assert_eq!(checked.search_non_root_backups, 1);
    for (pointer, value) in [
        ("/profile/provider", json!("cpu")),
        ("/profile/device_id", json!(1)),
        ("/profile/tf32", json!(true)),
        ("/profile/arena_bytes", json!(2147483648_u64)),
        ("/profile/runtime_bundle_sha256", json!("0".repeat(64))),
        (
            "/profile/runtime_bundle_manifest_sha256",
            json!("0".repeat(64)),
        ),
        (
            "/profile/runtime_bundle_files/0/sha256",
            json!("0".repeat(64)),
        ),
        (
            "/profile/session_resident_admission/scope",
            json!("measured_hardcap"),
        ),
        ("/profile/executed_cuda_nodes", json!(0)),
        ("/profile/runtime_mapping_verified", json!(false)),
        ("/profile/history_fill", json!("repeat_oldest")),
    ] {
        let mut s = startup.clone();
        *s.pointer_mut(pointer).unwrap() = value;
        let mut t = term.clone();
        t["startup"] = s.clone();
        assert!(!audit(&engine, &s, &t), "accepted {pointer}");
    }
    let mut s = startup.clone();
    s["profile"]["runtime_bundle_files"][1] = s["profile"]["runtime_bundle_files"][0].clone();
    let mut t = term.clone();
    t["startup"] = s.clone();
    assert!(!audit(&engine, &s, &t));
}

#[test]
fn cuda_wire_final_requires_guarded_nonroot_compute_and_lossless_drain() {
    let (engine, startup, term) = fixture();
    for (pointer, value) in [
        ("/physical_drain", json!("unconfirmed")),
        ("/actual_cuda_inference_observed", json!(false)),
        ("/actual_rules_search_backup_observed", json!(false)),
        ("/report/origin", json!("cpu_onnx")),
        ("/report/search_non_root_backups/count", json!(0)),
        (
            "/report/search_non_root_backups/first/traversed_edges",
            json!(0),
        ),
        (
            "/report/search_root_initializations/first/traversed_edges",
            json!(1),
        ),
        ("/report/completed_by_runtime", json!(1)),
        (
            "/report/search_non_root_backups/first/completed/actual/provenance",
            json!("raw_eval_hit"),
        ),
        (
            "/report/search_non_root_backups/last/completed/model/generation",
            json!(2),
        ),
        ("/report/observations/scheduler_dropped", json!(1)),
        (
            "/report/observations/delivery_counter_overflow",
            json!(true),
        ),
        ("/retained_owner_and_evidence", json!(true)),
        ("/run_succeeded", json!(false)),
        ("/report", Value::Null),
    ] {
        let mut t = term.clone();
        *t.pointer_mut(pointer).unwrap() = value;
        let preserved = serde_json::to_vec(&t).unwrap();
        assert!(!audit(&engine, &startup, &t), "accepted {pointer}");
        assert_eq!(serde_json::to_vec(&t).unwrap(), preserved);
    }
    for key in [
        "runtime_mapping_failure",
        "original_service_failure",
        "collection_failure",
    ] {
        let mut t = term.clone();
        t.as_object_mut().unwrap().remove(key);
        assert!(!audit(&engine, &startup, &t), "accepted absent {key}");
    }
    let mut t = term.clone();
    t["report"]["last_completed"]["actual"]
        .as_object_mut()
        .unwrap()
        .remove("prior_execution");
    assert!(!audit(&engine, &startup, &t));
}

#[test]
fn cuda_wire_rejects_unknown_duplicate_truncated_records() {
    let (engine, startup, term) = fixture();
    let bytes = serde_json::to_vec(&term).unwrap();
    let raw = serde_json::to_string(&startup).unwrap();
    let duplicate = raw.replacen("\"loaded\":true", "\"loaded\":true,\"loaded\":true", 1);
    assert!(
        validate_native_cuda_provider_record_fields(
            duplicate.as_bytes(),
            &bytes,
            &engine,
            "native-process-123"
        )
        .is_err()
    );
    assert!(
        validate_native_cuda_provider_record_fields(
            &vec![b' '; 256 * 1024 + 1],
            &bytes,
            &engine,
            "native-process-123"
        )
        .is_err()
    );
    let mut s = startup.clone();
    s["profile"]["physical_inferences"] = json!(999);
    let mut t = term.clone();
    t["startup"] = s.clone();
    assert!(!audit(&engine, &s, &t));
    assert!(
        validate_native_cuda_provider_record_fields(
            &serde_json::to_vec(&startup).unwrap(),
            &bytes,
            &engine,
            "native-process-124"
        )
        .is_err()
    );
}

#[test]
fn cuda_retained_manifest_and_trace_require_actual_bytes_and_cuda_kernel_records() {
    let (engine, _, _) = fixture();
    let binding = &engine.cuda_bundle;
    let bytes=serde_json::to_vec(&json!({"schema_version":1,"files":binding.files.iter().map(|f|json!({"role":f.role,"filename":f.filename,
        "bytes":f.artifact.bytes,"sha256":f.artifact.sha256})).collect::<Vec<_>>()})).unwrap();
    validate_cuda_bundle_manifest_fields(&bytes, binding).unwrap();
    let mut changed = bytes.clone();
    changed.push(b' ');
    assert!(validate_cuda_bundle_manifest_fields(&changed, binding).is_err());
    let trace = json!([{"cat":"Session","name":"registration"},{"cat":"Node","name":"conv_fence_before"},
        {"cat":"Node","name":"conv_kernel_time","args":{"provider":"CUDAExecutionProvider"}}]);
    let bytes = serde_json::to_vec(&trace).unwrap();
    validate_cuda_placement_trace_fields(&bytes, &sha(&bytes), 1).unwrap();
    assert!(validate_cuda_placement_trace_fields(&bytes, &sha(&bytes), 2).is_err());
    for trace in [
        json!([]),
        json!([{"cat":"Node","name":"conv_kernel_time","args":{"provider":"CPUExecutionProvider"}}]),
        json!([{"cat":"Node","name":"unknown","args":{"provider":"CUDAExecutionProvider"}}]),
        json!([{"cat":"Node","name":"conv_kernel_time"}]),
    ] {
        let bytes = serde_json::to_vec(&trace).unwrap();
        assert!(validate_cuda_placement_trace_fields(&bytes, &sha(&bytes), 1).is_err());
    }
}

#[test]
fn cuda_private_snapshot_keeps_closed_bundle_names_and_preserves_failed_preparation() {
    use rz_arena::{
        FASTCHESS_SOURCE_COMMIT, FASTCHESS_SOURCE_URL, FASTCHESS_VERSION, native_admission_closed,
        opening_pgn_for_spec, prepare_native_cuda_launch,
    };
    use std::{fs, os::unix::fs::PermissionsExt};
    let base = std::env::temp_dir().join(format!("rovezero-cuda-owner-{}", std::process::id()));
    fs::create_dir(&base).expect("exclusive synthetic CUDA owner fixture root");
    let source = base.join("source");
    let output = base.join("output");
    fs::create_dir(&source).unwrap();
    fs::create_dir(&output).unwrap();
    fs::create_dir(source.join("runtime")).unwrap();
    fs::create_dir(source.join("native")).unwrap();
    let (mut engine, _, _) = fixture();
    let blob = b"synthetic library bytes";
    for file in &mut engine.cuda_bundle.files {
        fs::write(source.join(&file.artifact.path), blob).unwrap();
        file.artifact.sha256 = sha(blob);
        file.artifact.bytes = blob.len() as u64;
    }
    engine.cuda_bundle.canonical_sha256 = engine.cuda_bundle.canonical_digest().unwrap();
    let bundle_bytes=serde_json::to_vec(&json!({"schema_version":1,"files":engine.cuda_bundle.files.iter().map(|f|json!({"role":f.role,
        "filename":f.filename,"bytes":f.artifact.bytes,"sha256":f.artifact.sha256})).collect::<Vec<_>>()})).unwrap();
    fs::write(source.join("runtime/bundle.json"), &bundle_bytes).unwrap();
    engine.cuda_bundle.manifest.sha256 = sha(&bundle_bytes);
    engine.cuda_bundle.manifest.bytes = bundle_bytes.len() as u64;
    let core = engine
        .cuda_bundle
        .file(CudaBundleFileRoleV1::Core, "libonnxruntime.so.1.22.0")
        .unwrap()
        .clone();
    for binding in &mut engine.artifacts {
        if binding.role == NativeArtifactRole::OrtLibrary {
            binding.artifact = core.clone();
        } else {
            fs::write(source.join(&binding.artifact.path), blob).unwrap();
            binding.artifact.sha256 = sha(blob);
            binding.artifact.bytes = blob.len() as u64;
        }
    }
    let opening = OpeningSpec {
        id: "snapshot-opening".into(),
        initial: InitialPosition::Startpos,
        fen: None,
        moves: vec!["e2e4".into(), "e7e5".into()],
        history: HistoryCompleteness::Complete,
        history_origin: "startpos-complete-trace".into(),
    };
    let opening_bytes = opening_pgn_for_spec(&opening, 8).unwrap();
    fs::write(source.join("opening.pgn"), &opening_bytes).unwrap();
    let opening_artifact = artifact(
        "opening.pgn",
        &sha(opening_bytes.as_bytes()),
        opening_bytes.len() as u64,
    );
    fs::write(source.join("fastchess"), blob).unwrap();
    let mut runner_binary = artifact("fastchess", &sha(blob), blob.len() as u64);
    runner_binary.source = FASTCHESS_SOURCE_URL.into();
    let mut candidate = engine.clone();
    candidate.role = NativeEngineRole::Candidate;
    candidate.engine_id = "cuda-candidate".into();
    let spec = CudaIntegrationPairSpecV1 {
        schema_version: 1,
        purpose: NativeCudaIntegrationPurpose::CudaNnIntegration,
        strength_eligible: false,
        contract_revision: "0.1".into(),
        run_id: "snapshot-run".into(),
        pair_id: "snapshot-pair".into(),
        engines: [engine, candidate],
        white_order: [NativeEngineRole::Baseline, NativeEngineRole::Candidate],
        opening,
        opening_artifact,
        runner: ToolIdentity {
            source_url: FASTCHESS_SOURCE_URL.into(),
            source_commit: FASTCHESS_SOURCE_COMMIT.into(),
            version: FASTCHESS_VERSION.into(),
            binary: runner_binary,
            dirty: false,
            dirty_patch: None,
            build_mode: "synthetic-not-built".into(),
            compiler: "synthetic-not-built".into(),
            target: "x86_64-unknown-linux-gnu".into(),
            isa: "x86-64".into(),
        },
        clock: NativeMovetimeV1 { movetime_ms: 100 },
        max_plies: 8,
        timeouts: NativeTimeoutsV1 {
            startup_ms: 1000,
            handshake_ms: 1000,
            runtime_ms: 10000,
            drain_ms: 1000,
            shutdown_ms: 1000,
        },
        budget: NativeCudaResourceBudgetV1 {
            max_input_bytes: 64 * 1024,
            max_output_bytes: 256 * 1024,
            max_runtime_bytes: 20 * 1024 * 1024,
            max_artifact_bytes: 24 * 1024 * 1024,
            max_child_processes: 3,
            max_runtime_files: 128,
            max_runtime_depth: 6,
            address_space_per_process_bytes: 2 * 1024 * 1024 * 1024,
        },
    };
    let locked = spec.clone().lock().unwrap();
    let owner = prepare_native_cuda_launch(&locked, &source, &output, "attempt-one").unwrap();
    for file in &locked.input().engines[0].cuda_bundle.files {
        assert!(owner.snapshots().iter().any(|s| s.snapshot_relative_path
            == format!("inputs/cuda-bundle/{}", file.filename)
            && s.distinct_source_inode
            && s.closed_writer_read_only));
    }
    assert!(
        output
            .join("attempt-one/inputs/cuda-bundle/bundle.v1.json")
            .is_file()
    );
    assert_eq!(
        fs::metadata(output.join("attempt-one/inputs/cuda-bundle"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o500
    );
    assert!(prepare_native_cuda_launch(&locked, &source, &output, "parallel-attempt").is_err());
    drop(owner);
    let mut invalid = spec;
    let broken = b"{\"schema_version\":1,\"files\":[]}";
    fs::write(source.join("runtime/bundle.json"), broken).unwrap();
    for engine in &mut invalid.engines {
        engine.cuda_bundle.manifest.sha256 = sha(broken);
        engine.cuda_bundle.manifest.bytes = broken.len() as u64;
    }
    let error = prepare_native_cuda_launch(
        &invalid.lock().unwrap(),
        &source,
        &output,
        "rejected-manifest",
    )
    .unwrap_err();
    assert!(error.receipt.attempt_created);
    assert!(!error.receipt.child_spawned);
    assert!(!error.receipt.input_pins_required);
    assert!(error.receipt.writers_closed);
    assert!(error.receipt_artifact.is_some());
    assert!(!native_admission_closed());
    assert!(output.join("rejected-manifest").is_dir());
    assert!(!output.join("parallel-attempt").exists());
    assert!(prepare_native_cuda_launch(&locked, &source, &output, "attempt-one").is_err());
    // Only this never-executed, exclusive synthetic test tree is removed.
    for attempt in ["attempt-one", "rejected-manifest"] {
        for relative in ["inputs", "inputs/cuda-bundle"] {
            fs::set_permissions(
                output.join(attempt).join(relative),
                fs::Permissions::from_mode(0o700),
            )
            .unwrap();
        }
    }
    fs::remove_dir_all(base).unwrap();
}

fn bt4_wire_fixture() -> (CudaNativeLaunchSpecV2, Value, Value) {
    let (engine, mut startup, mut termination) = fixture();
    let mut engine = CudaNativeLaunchSpecV2 {
        role: engine.role,
        engine_id: engine.engine_id,
        source_commit: engine.source_commit,
        target: engine.target,
        artifacts: engine.artifacts,
        cuda_bundle: engine.cuda_bundle,
        profile: NativeCudaProfileV2::fixed("b".repeat(64), "d".repeat(64)),
    };
    let source = engine
        .artifacts
        .iter_mut()
        .find(|a| a.role == NativeArtifactRole::SourceWeights)
        .unwrap();
    source.artifact.sha256 = BT4_NATIVE_SOURCE_SHA256.into();
    source.artifact.bytes = BT4_NATIVE_SOURCE_BYTES;
    startup["profile"]["source_weights_gzip_sha256"] = json!(BT4_NATIVE_SOURCE_SHA256);
    startup["profile"]["arena_bytes"] = json!(BT4_NATIVE_ARENA_BYTES);
    startup["profile"]["session_resident_admission"]["device_bytes"] =
        json!(BT4_NATIVE_ARENA_BYTES);
    termination["startup"] = startup.clone();
    (engine, startup, termination)
}

#[test]
fn bt4_provider_consumes_v1_receipt_fields_under_its_separate_v2_profile() {
    let (engine, startup, termination) = bt4_wire_fixture();
    let check = |startup: &Value, termination: &Value| {
        validate_native_cuda_provider_record_fields(
            &serde_json::to_vec(startup).unwrap(),
            &serde_json::to_vec(termination).unwrap(),
            &engine,
            "native-process-123",
        )
    };
    assert!(check(&startup, &termination).is_ok());
    for pointer in [
        "/profile/arena_bytes",
        "/profile/session_resident_admission/device_bytes",
    ] {
        let mut changed = startup.clone();
        *changed.pointer_mut(pointer).unwrap() = json!(CUDA_NATIVE_ARENA_BYTES);
        let mut term = termination.clone();
        term["startup"] = changed.clone();
        assert!(check(&changed, &term).is_err());
    }
    let mut changed = startup.clone();
    changed["profile"]["source_weights_gzip_sha256"] = json!("a".repeat(64));
    let mut term = termination;
    term["startup"] = changed.clone();
    assert!(check(&changed, &term).is_err());
}

#[test]
fn bt4_search_receipt_binds_the_served_settings_to_exact_startup_bytes() {
    use rz_uci::{
        engine::EngineSettings,
        native_cuda_attestation::{CudaSearchReceiptV1, CudaStartupReceiptV1},
    };
    let (mut engine, startup, _) = bt4_wire_fixture();
    let startup: CudaStartupReceiptV1 = serde_json::from_value(startup).unwrap();
    let mut settings = EngineSettings {
        max_workers: 1,
        ..EngineSettings::default()
    };
    settings.search.max_simulations = 4096;
    let mut startup_bytes = serde_json::to_vec(&startup).unwrap();
    startup_bytes.push(b'\n');
    let captured = CudaSearchReceiptV1::capture(&startup, &settings).unwrap();
    let receipt = serde_json::to_value(captured).unwrap();
    let check = |value: &Value, startup: &[u8]| {
        rz_arena::validate_native_cuda_search_record_fields(
            &serde_json::to_vec(value).unwrap(),
            startup,
            engine.profile.search,
        )
    };
    assert!(check(&receipt, &startup_bytes).is_ok());
    for (key, wrong) in [
        ("simulations", json!(128)),
        ("final_selection", json!("exact-terminal")),
        ("startup_sha256", json!("a".repeat(64))),
        ("raw_cache", json!(true)),
        ("policy_temperature_milli", json!(800)),
        ("process_run_id", json!("native-process-124")),
        ("output_margin_ms", json!(0)),
    ] {
        let mut changed = receipt.clone();
        changed[key] = wrong;
        assert!(check(&changed, &startup_bytes).is_err(), "accepted {key}");
    }
    assert!(check(&receipt, &serde_json::to_vec(&startup).unwrap()).is_err());
    let mut absent = receipt.clone();
    absent.as_object_mut().unwrap().remove("raw_cache");
    assert!(check(&absent, &startup_bytes).is_err());
    let mut unknown = receipt.clone();
    unknown["extra"] = json!(true);
    assert!(check(&unknown, &startup_bytes).is_err());
    engine.profile.search.final_selection = NativeFinalSelectionV2::ExactTerminal;
    // Baseline captured Visits cannot satisfy an ExactTerminal lock.
    assert!(
        rz_arena::validate_native_cuda_search_record_fields(
            &serde_json::to_vec(&receipt).unwrap(),
            &startup_bytes,
            engine.profile.search
        )
        .is_err()
    );
}
