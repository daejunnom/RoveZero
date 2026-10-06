#![cfg(feature = "onnx")]

use rz_eval::onnx::{verify_cuda_profile, BackendConfig, Provider};

#[test]
fn provider_registration_or_mixed_execution_does_not_prove_cuda_execution() {
    for profile in [
        r#"[]"#,
        r#"[{"cat":"Session","args":{"provider":"CUDAExecutionProvider"}}]"#,
        r#"[{"cat":"Node","name":"conv_kernel_time","args":{"provider":"CPUExecutionProvider"}}]"#,
        r#"[{"cat":"Node","name":"conv_kernel_time","args":{"provider":"CUDAExecutionProvider"}},{"cat":"Node","name":"unknown_kernel_time"}]"#,
        r#"[{"cat":"Node","name":"conv_kernel_time","args":{"provider":"CUDAExecutionProvider"}},{"cat":"Node","name":"policy_kernel_time","args":{"provider":"CPUExecutionProvider"}}]"#,
        "invalid json",
    ] {
        assert!(verify_cuda_profile(profile.as_bytes()).is_err());
    }
    assert_eq!(
        verify_cuda_profile(
            br#"[{"cat":"Node","name":"conv_kernel_time","args":{"provider":"CUDAExecutionProvider"}}]"#
        )
        .unwrap(),
        1
    );
}

#[test]
fn real_kernel_placement_cannot_be_missing_malformed_or_unknown() {
    let known = serde_json::json!({
        "cat": "Node",
        "name": "conv_kernel_time",
        "args": {"provider": "CUDAExecutionProvider"},
    });
    for unknown in [
        serde_json::json!({"cat": "Node", "name": "value_kernel_time"}),
        serde_json::json!({"cat": "Node", "name": "value_kernel_time", "args": {}}),
        serde_json::json!({"cat": "Node", "name": "value_kernel_time", "args": {"provider": null}}),
        serde_json::json!({"cat": "Node", "name": "value_kernel_time", "args": {"provider": 17}}),
        serde_json::json!({"cat": "Node", "name": "value_kernel_time", "args": {"provider": "UnknownExecutionProvider"}}),
        serde_json::json!({"cat": "Node", "args": {"provider": "CUDAExecutionProvider"}}),
        serde_json::json!({"cat": "Node", "name": "unknown_event", "args": {"provider": "CUDAExecutionProvider"}}),
    ] {
        // One valid kernel must not hide an unaccounted sibling's placement.
        let profile = serde_json::to_vec(&vec![known.clone(), unknown]).unwrap();
        assert!(verify_cuda_profile(&profile).is_err());
    }
}

#[test]
fn known_fences_are_metadata_and_never_count_as_executed_kernels() {
    let profile = br#"[
        {"cat":"Session","name":"model_run"},
        {"cat":"Node","name":"conv_fence_before","args":{}},
        {"cat":"Node","name":"conv_kernel_time","args":{"provider":"CUDAExecutionProvider"}},
        {"cat":"Node","name":"conv_fence_after","args":{}},
        {"cat":"Node","name":"policy_kernel_time","args":{"provider":"CUDAExecutionProvider"}}
    ]"#;
    assert_eq!(verify_cuda_profile(profile).unwrap(), 2);
    assert!(verify_cuda_profile(
        br#"[{"cat":"Node","name":"conv_fence_before"},{"cat":"Node","name":"conv_fence_after"}]"#
    )
    .is_err());
    for provider in [
        serde_json::json!(null),
        serde_json::json!(false),
        serde_json::json!("CPUExecutionProvider"),
    ] {
        let profile = serde_json::to_vec(&serde_json::json!([
            {"cat":"Node","name":"conv_kernel_time","args":{"provider":"CUDAExecutionProvider"}},
            {"cat":"Node","name":"conv_fence_after","args":{"provider":provider}}
        ]))
        .unwrap();
        assert!(verify_cuda_profile(&profile).is_err());
    }
}

#[test]
fn config_requires_finite_resources_and_cuda_evidence_destination() {
    BackendConfig::cpu().validate().unwrap();
    for invalid in [
        BackendConfig {
            max_batch: 0,
            ..BackendConfig::cpu()
        },
        BackendConfig {
            max_batch: 17,
            ..BackendConfig::cpu()
        },
        BackendConfig {
            host_io_bytes: 1,
            ..BackendConfig::cpu()
        },
        BackendConfig {
            intra_threads: 0,
            ..BackendConfig::cpu()
        },
        BackendConfig {
            provider: Provider::Cuda {
                device_id: 0,
                arena_bytes: 1024,
            },
            ..BackendConfig::cpu()
        },
        BackendConfig {
            provider: Provider::Cuda {
                device_id: 0,
                arena_bytes: 1024,
            },
            profiling_prefix: Some("relative-placement-prefix".into()),
            ..BackendConfig::cpu()
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn experimental_execution_rejects_wrong_provider_shape_and_uncompiled_modes() {
    use rz_eval::onnx::ExecutionExperiments;
    let mut config = BackendConfig::cpu();
    config.experiments = ExecutionExperiments {
        cuda_graph: true,
        ..Default::default()
    };
    assert!(config.validate().is_err());
    config.experiments.io_binding = true;
    assert!(config.validate().is_err()); // CPU cannot capture CUDA graph
    config.provider = Provider::Cuda {
        device_id: 0,
        arena_bytes: 1024,
    };
    config.profiling_prefix = Some(std::env::temp_dir().join("rz-graph-placement"));
    assert!(config.validate().is_err()); // variable batch is forbidden
    config.max_batch = 1;
    assert_eq!(
        config.validate().is_ok(),
        cfg!(feature = "experimental-cuda-graph")
    );
    config.experiments = ExecutionExperiments {
        reuse_buffers: true,
        ..Default::default()
    };
    assert_eq!(
        config.validate().is_ok(),
        cfg!(feature = "experimental-io-buffers")
    );
}

#[test]
fn cuda_cpu_arena_experiment_is_explicit_cuda_only_and_separately_identified() {
    use rz_eval::onnx::declared_backend_identity;
    let mut config = BackendConfig::cpu();
    config.experiments.disable_cuda_cpu_arena = true;
    assert!(config.validate().is_err()); // Does not silently alter CPU inference.
    config.provider = Provider::Cuda {
        device_id: 0,
        arena_bytes: 1024,
    };
    config.profiling_prefix = Some(std::env::temp_dir().join("rz-cpu-arena-placement"));
    assert_eq!(
        config.validate().is_ok(),
        cfg!(feature = "experimental-ort-cpu-arena")
    );
    let experiment = declared_backend_identity([0; 32], [0; 32], Some([0; 32]), &config);
    config.experiments.disable_cuda_cpu_arena = false;
    config.validate().unwrap();
    assert_ne!(
        experiment,
        declared_backend_identity([0; 32], [0; 32], Some([0; 32]), &config)
    );
    assert_eq!(config.experiments, Default::default());
}
