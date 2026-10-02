#![cfg(feature = "onnx")]

use rz_eval::onnx::{verify_cuda_profile, BackendConfig, Provider};

#[test]
fn provider_registration_or_mixed_execution_does_not_prove_cuda_execution() {
    for profile in [
        r#"[]"#,
        r#"[{"cat":"Session","args":{"provider":"CUDAExecutionProvider"}}]"#,
        r#"[{"cat":"Node","args":{"provider":"CPUExecutionProvider"}}]"#,
        r#"[{"cat":"Node","args":{"provider":"CUDAExecutionProvider"}},{"cat":"Node","args":{"provider":"CPUExecutionProvider"}}]"#,
        "invalid json",
    ] {
        assert!(verify_cuda_profile(profile.as_bytes()).is_err());
    }
    assert_eq!(
        verify_cuda_profile(br#"[{"cat":"Node","args":{"provider":"CUDAExecutionProvider"}}]"#)
            .unwrap(),
        1
    );
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
    ] {
        assert!(invalid.validate().is_err());
    }
}
