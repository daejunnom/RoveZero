#![cfg(target_os = "linux")]

use rz_arena::validate_native_provider_record_fields;
use rz_experiments::*;
use serde_json::{Value, json};

// Public wire-field fixtures only. These tests never create a neural session,
// execute an engine/runner, load an external model or prove physical drain.
fn fixture() -> (CpuNativeLaunchSpecV1, Value, Value) {
    let artifacts = NativeArtifactRole::ALL
        .into_iter()
        .enumerate()
        .map(|(i, role)| NativeArtifactBinding {
            role,
            artifact: ArtifactRef {
                path: format!("native/input-{i}"),
                sha256: if i == 0 {
                    "9".repeat(64)
                } else {
                    format!("{i:x}").repeat(64)
                },
                bytes: 10 + i as u64,
                source: "https://example.org/rovezero-wire-test".into(),
                license: "Synthetic wire fixture only".into(),
            },
        })
        .collect();
    let engine = CpuNativeLaunchSpecV1 {
        role: NativeEngineRole::Baseline,
        engine_id: "wire-baseline".into(),
        source_commit: "a".repeat(40),
        target: "x86_64-unknown-linux-gnu".into(),
        artifacts,
        profile: NativeCpuProfileV1::fixed("b".repeat(64), "d".repeat(64)),
    };
    let model = json!({"owner":1,"slot":1,"generation":1,"manifest_sha256":"3".repeat(64)});
    let encoding = json!({"owner":1,"slot":2,"generation":1,"manifest_sha256":"d".repeat(64)});
    let startup = json!({"schema_version":1,"kind":"startup","loaded":true,"process_id":123,"process_run_id":"native-process-123","process_epoch":1,
        "executable":{"sha256":"9".repeat(64),"bytes":10,"identity_source":"linux_proc_self_exe"},
        "profile":{"source_weights_gzip_sha256":"1".repeat(64),"source_weights_protobuf_sha256":"e".repeat(64),"onnx_sha256":"2".repeat(64),"onnx_bytes":12,"export_manifest_sha256":"3".repeat(64),"converter_commit":"c".repeat(40),"converter_binary_sha256":"f".repeat(64),
            "ort_library_sha256":"4".repeat(64),"ort_build_info_sha256":"a".repeat(64),"ort_release":"1.22.0","backend_sha256":"b".repeat(64),"model":model,"encoding":encoding,
            "action_map_sha256":"a".repeat(64),"history_policy_sha256":"c".repeat(64),"history_length":8,"contract_major":0,"contract_minor":1,"provider":"cpu","precision":"fp32","max_batch_items":1,"intra_threads":1,"max_workers":1,"full_steps":1,"min_steps":1,"max_steps":1,"require_full":true,"evaluation_mode":"fresh","history_fill":"no"}});
    let completion = json!({"contract_major":0,"contract_minor":1,"request":{"epoch":1,"sequence":1},"selection":{"epoch":1,"sequence":2},"game":1,"root":1,"state_owner":2,"state_revision":0,"state_sha256":"a".repeat(64),"legal_order_sha256":"a".repeat(64),"input_sha256":"a".repeat(64),
        "model":startup["profile"]["model"],"encoding":startup["profile"]["encoding"],"backend_sha256":"b".repeat(64),"precision":"fp32","min_steps":1,"max_steps":1,"require_full":true,"execution":{"epoch":1,"sequence":3},
        "actual":{"precision":"fp32","steps":1,"full":true,"backend_sha256":"b".repeat(64),"execution":{"epoch":1,"sequence":3},"provenance":"computed","prior_execution":null}});
    let term = json!({"schema_version":1,"kind":"termination","process_run_id":"native-process-123","startup":startup,"run_succeeded":true,"actual_cpu_inference_observed":true,"physical_drain":"confirmed",
        "report":{"model_manifest_sha256":"3".repeat(64),"backend_sha256":"b".repeat(64),"origin":"cpu_onnx","completed_by_runtime":1,"first_completed":completion,"last_completed":completion,
            "completed_scope":"process_aggregate_normal_poll_first_last_not_full_root_or_physical_journal","canceled":{"count":0,"first":null,"last":null},"expired":{"count":0,"first":null,"last":null},"failures":[],"overflow":null,"boundary_error":null,"poison_error":null},
        "original_service_failure":null,"collection_failure":null,"retained_owner_and_evidence":false});
    (engine, startup, term)
}
fn audit(engine: &CpuNativeLaunchSpecV1, startup: &Value, term: &Value) -> bool {
    validate_native_provider_record_fields(
        &serde_json::to_vec(startup).unwrap(),
        &serde_json::to_vec(term).unwrap(),
        engine,
        "native-process-123",
    )
    .is_ok()
}

#[test]
fn native_wire_field_validator_requires_loaded_and_actual_computed_cpu_completion() {
    let (engine, startup, term) = fixture();
    let checked = validate_native_provider_record_fields(
        &serde_json::to_vec(&startup).unwrap(),
        &serde_json::to_vec(&term).unwrap(),
        &engine,
        "native-process-123",
    )
    .unwrap();
    assert_eq!(checked.completed_by_runtime, 1);
    for (pointer, value) in [
        ("/loaded", json!(false)),
        ("/executable/sha256", json!("8".repeat(64))),
        ("/profile/history_fill", json!("repeat_oldest")),
        ("/profile/evaluation_mode", json!("warm")),
        ("/profile/full_steps", json!(2)),
        ("/profile/provider", json!("cuda")),
    ] {
        let mut s = startup.clone();
        *s.pointer_mut(pointer).unwrap() = value;
        let mut t = term.clone();
        t["startup"] = s.clone();
        assert!(!audit(&engine, &s, &t), "accepted {pointer}");
    }
    for (pointer, value) in [
        ("/report/completed_by_runtime", json!(0)),
        (
            "/report/first_completed/actual/provenance",
            json!("raw_eval_hit"),
        ),
        ("/report/last_completed/actual/execution/sequence", json!(4)),
        ("/report/last_completed/selection/epoch", json!(2)),
        ("/actual_cpu_inference_observed", json!(false)),
    ] {
        let mut t = term.clone();
        *t.pointer_mut(pointer).unwrap() = value;
        assert!(!audit(&engine, &startup, &t), "accepted {pointer}");
    }
}

#[test]
fn native_wire_field_validator_rejects_partial_failure_drain_and_generation_mismatch() {
    let (engine, startup, term) = fixture();
    for (pointer, value) in [
        ("/physical_drain", json!("unconfirmed")),
        ("/report", Value::Null),
        (
            "/original_service_failure",
            json!({"code":"canceled","projection_truncated":true}),
        ),
        ("/collection_failure", json!({"code":"timeout"})),
        ("/run_succeeded", json!(false)),
        ("/retained_owner_and_evidence", json!(true)),
        ("/report/poison_error", json!({"code":"poisoned"})),
        ("/report/first_completed/model/generation", json!(2)),
        ("/report/expired/count", json!(1)),
    ] {
        let mut t = term.clone();
        *t.pointer_mut(pointer).unwrap() = value;
        let bytes = serde_json::to_vec(&t).unwrap();
        let original = bytes.clone();
        assert!(
            validate_native_provider_record_fields(
                &serde_json::to_vec(&startup).unwrap(),
                &bytes,
                &engine,
                "native-process-123"
            )
            .is_err(),
            "accepted {pointer}"
        );
        assert_eq!(bytes, original, "validator changed retained failure bytes");
    }
    let mut wrong = term.clone();
    wrong["process_run_id"] = json!("native-process-124");
    assert!(!audit(&engine, &startup, &wrong));
    let mut truncated = term.clone();
    truncated["report"]["canceled"] = json!({"count":1,
        "first":{"projection":{},"projection_truncated":true,"max_depth":8,"max_nodes":128,"projected_nodes":128},
        "last":{"projection":{},"projection_truncated":true,"max_depth":8,"max_nodes":128,"projected_nodes":128}});
    assert!(!audit(&engine, &startup, &truncated));
}

#[test]
fn native_wire_field_validator_bounds_json_and_rejects_duplicate_unknown_fields() {
    let (engine, startup, term) = fixture();
    let raw = serde_json::to_string(&startup).unwrap();
    let duplicate = raw.replacen("\"loaded\":true", "\"loaded\":true,\"loaded\":true", 1);
    assert_ne!(duplicate, raw);
    assert!(
        validate_native_provider_record_fields(
            duplicate.as_bytes(),
            &serde_json::to_vec(&term).unwrap(),
            &engine,
            "native-process-123"
        )
        .is_err()
    );
    let mut unknown = startup.clone();
    unknown["profile"]["future_field"] = json!(true);
    let mut t = term.clone();
    t["startup"] = unknown.clone();
    assert!(!audit(&engine, &unknown, &t));
    assert!(
        validate_native_provider_record_fields(
            &vec![b' '; 256 * 1024 + 1],
            &serde_json::to_vec(&term).unwrap(),
            &engine,
            "native-process-123"
        )
        .is_err()
    );
    assert!(
        validate_native_provider_record_fields(
            &serde_json::to_vec(&startup).unwrap(),
            &serde_json::to_vec(&term).unwrap(),
            &engine,
            "native-process-124"
        )
        .is_err()
    );
}
