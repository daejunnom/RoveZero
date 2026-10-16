use super::*;
use crate::*;

fn artifact(path: &str) -> ArtifactRef {
    ArtifactRef {
        path: path.into(),
        sha256: "a".repeat(64),
        bytes: 1,
        source: "https://example.org/explicit-fixture".into(),
        license: "MIT".into(),
    }
}
fn component(id: &str) -> PalsComponentV3 {
    PalsComponentV3 {
        semantic_id: id.into(),
        implementation_sha256: "b".repeat(64),
        options: BTreeMap::new(),
    }
}
fn config() -> (PalsAdaptiveStartupConfigV5, PalsInputLockV5) {
    let native_options = cpu_options();
    let mut engine = PalsEngineV5 {
        id: "g0".into(),
        binary: artifact("bin/engine"),
        source_commit: "f".repeat(40),
        mode: PalsEngineModeV5::Deployment,
        model: PalsModelSpecificationV5 {
            profile: "virtual_review_v1".into(),
            model_schema: PALS_V5_MODEL_SCHEMA.into(),
            model_semantics: PALS_V5_MODEL_SEMANTICS.into(),
            encoding_schema: PALS_V5_ENCODING_SCHEMA.into(),
            semantic_sha256: "c".repeat(64),
            encoding_sha256: "d".repeat(64),
            implementation_sha256: "e".repeat(64),
            weights: artifact("weights/untrained.bin"),
            artifact_domain: PalsArtifactDomainV5::Untrained,
            provenance: None,
            checkpoint_domain: None,
            epoch: 1,
            backend: PalsModelBackendV3::RustCpu,
            precision: PalsPrecisionV3::Fp32,
            tf32: false,
            exported_roles: BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
            width: 384,
            latent_count: 16,
            temporal_width: 64,
            candidate_head_width: 128,
            record_feature_axes: 8,
            query_feature_axes: 8,
            max_records: 64,
            max_anchors: 4,
            line_buckets: PALS_V5_LINE_BUCKETS.to_vec(),
            one_dynamic_line_graph: true,
            ids_as_numeric_features: false,
            fresh_warm_separate_domains: true,
        },
        cpu: PalsCpuEndpointsV5 {
            cpu_t: PalsCpuSelectionV5::Disabled,
            cpu_r: PalsCpuSelectionV5::Own,
            own: component("own/1"),
            external: None,
        },
        context: PalsReviewContextV5::default(),
        coordinator: PalsCoordinatorV5::default(),
        resources: PalsResourcePlanV5::default(),
        search: component("search/1"),
        runtime: component("runtime/1"),
        v_generate: false,
        v_load: false,
        v_call: false,
    };
    engine.runtime.options = native_options.canonical_runtime_options();
    // Synthetic source pins exercise strict transport identity only. They are
    // never a content-verification or native provider receipt.
    let manifest = PalsRunManifestV5 {
        schema_version: 5,
        run_id: "startup-fixture".into(),
        contract_revision: PALS_V5_CONTRACT_REVISION.into(),
        purpose: PalsRunPurposeV5::CpuValidation,
        comparison: PalsComparisonV5::System,
        rules_profile: "standard-chess/1".into(),
        source_pins: PALS_V5_BASE_SOURCE_COMPONENTS
            .into_iter()
            .chain(PALS_V5_NATIVE_SOURCE_COMPONENTS)
            .chain(PALS_V5_COLLECTION_SOURCE_COMPONENTS)
            .chain(PALS_V5_TRAINING_SOURCE_COMPONENTS)
            .chain(PALS_V5_PAIR_SOURCE_COMPONENTS)
            .map(|k| (k.into(), "f".repeat(64)))
            .collect(),
        engines: vec![engine.clone()],
        pair: None,
        generation: Some(PalsGenerationRecipeV5::default()),
        window: PalsExecutionWindowV5 {
            window_id: "fixture-window".into(),
            wall_time_max_ms: 120000,
            cleanup_max_ms: 30000,
            output_bytes_max: 1 << 20,
            agent_renewable: true,
        },
        training_authorized: false,
    };
    let lock = manifest.lock().unwrap();
    let mut assets = Vec::new();
    for (purpose, name) in [
        (PalsStartupAssetPurposeV5::Weights, "weights.bin"),
        (PalsStartupAssetPurposeV5::ExportManifest, "export.json"),
        (PalsStartupAssetPurposeV5::PublicGraph, "public_memory.onnx"),
        (
            PalsStartupAssetPurposeV5::SharedPcGraph,
            "shared_pc_if.onnx",
        ),
        (PalsStartupAssetPurposeV5::RuntimeLibrary, "runtime.so"),
        (PalsStartupAssetPurposeV5::RulesDescriptor, "rules.json"),
        (
            PalsStartupAssetPurposeV5::InitializationCheckpointMetadata,
            "checkpoint.json",
        ),
        (
            PalsStartupAssetPurposeV5::NumericalAcceptance,
            "numerical-acceptance.json",
        ),
        (
            PalsStartupAssetPurposeV5::InitializationEvidence,
            "initialization.json",
        ),
    ] {
        assets.push(PalsStartupAssetV5 {
            artifact: if purpose == PalsStartupAssetPurposeV5::Weights {
                engine.model.weights.clone()
            } else {
                artifact(&format!("fixtures/{name}"))
            },
            snapshot_path: format!("inputs/model/{name}"),
            purpose,
        });
    }
    (
        PalsAdaptiveStartupConfigV5 {
            schema_version: 5,
            domain: PALS_STARTUP_CONFIG_V5_DOMAIN.into(),
            semantic_lock_sha256: lock.canonical_sha256.clone(),
            endpoint: engine,
            runtime_directory: "baseline-runtime".into(),
            native_options,
            assets,
            source_components: lock.manifest.source_pins.clone(),
            training_executed: false,
        },
        lock,
    )
}
fn cpu_options() -> PalsNativeStartupOptionsV5 {
    PalsNativeStartupOptionsV5 {
        native_loading_profile: None,
        cuda_device_id: None,
        cuda_arena_bytes: None,
        cache_public_memory: true,
        device_public_memory: false,
        cudnn_conv_search: PALS_CUDNN_IMPLICIT_EXHAUSTIVE_V5.into(),
        approximate_warm: false,
        startup_ms: 10_000,
        drain_ms: 5_000,
    }
}
fn relock(c: &mut PalsAdaptiveStartupConfigV5, lock: &PalsInputLockV5) -> PalsInputLockV5 {
    let mut manifest = lock.manifest.clone();
    manifest.engines[0] = c.endpoint.clone();
    let next = manifest.lock().unwrap();
    c.semantic_lock_sha256 = next.canonical_sha256.clone();
    next
}
#[test]
fn canonical_startup_seals_endpoint_pins_and_nontraining_mode() {
    let (c, lock) = config();
    let bytes = c.canonical_bytes(&lock).unwrap();
    let parsed =
        PalsAdaptiveStartupConfigV5::from_json(std::str::from_utf8(&bytes).unwrap(), &lock)
            .unwrap();
    assert_eq!(parsed, c);
    let mut other = c.clone();
    other.training_executed = true;
    assert!(other.canonical_bytes(&lock).is_err());
    other = c.clone();
    other.endpoint.id = "different-question-owner".into();
    assert!(other.validate_against(&lock).is_err());
    other = c;
    other
        .source_components
        .insert("own_cpu".into(), "0".repeat(64));
    assert!(other.validate_against(&lock).is_err());
}
#[test]
fn deployment_rejects_private_v_before_native_loading() {
    let (mut c, lock) = config();
    c.assets.push(PalsStartupAssetV5 {
        artifact: artifact("fixtures/role_validator.onnx"),
        snapshot_path: "inputs/model/role_validator.onnx".into(),
        purpose: PalsStartupAssetPurposeV5::ValidatorGraph,
    });
    assert!(c.validate_against(&lock).is_err());
    assert!(!c.endpoint.v_load);
}
#[test]
fn startup_rejects_missing_duplicate_and_escape_assets() {
    let (c, lock) = config();
    let mut missing = c.clone();
    missing
        .assets
        .retain(|a| a.purpose != PalsStartupAssetPurposeV5::RuntimeLibrary);
    assert!(missing.validate_against(&lock).is_err());
    let mut alias = c.clone();
    alias.assets[1].snapshot_path = alias.assets[0].snapshot_path.clone();
    assert!(alias.validate_against(&lock).is_err());
    let mut escape = c;
    escape.assets[1].snapshot_path = "inputs/../external.json".into();
    assert!(escape.validate_against(&lock).is_err());
}
#[test]
fn private_namespace_never_reads_credential_or_absolute_paths() {
    for forbidden in [
        "inputs/.env",
        "inputs/auth-credential.json",
        "inputs/client-secret.json",
        "inputs/ssh-key.key",
        "inputs/service-account.json",
        "/inputs/a",
        "inputs/../a",
        "inputs\\a",
        "C:/inputs/a",
    ] {
        assert!(!pals_private_snapshot_path_v5(forbidden), "{forbidden}");
    }
    assert!(pals_private_snapshot_path_v5(
        "inputs/model/public_memory.onnx"
    ));
}

#[test]
fn native_options_are_required_and_all_registered_runtime_values_are_exact() {
    let (c, lock) = config();
    let raw = serde_json::to_value(&c).unwrap();
    let mut missing = raw.clone();
    missing.as_object_mut().unwrap().remove("native_options");
    assert!(
        PalsAdaptiveStartupConfigV5::from_json(&serde_json::to_string(&missing).unwrap(), &lock)
            .is_err()
    );
    for nullable in [
        "native_loading_profile",
        "cuda_device_id",
        "cuda_arena_bytes",
    ] {
        let mut omitted = raw.clone();
        omitted["native_options"]
            .as_object_mut()
            .unwrap()
            .remove(nullable);
        assert!(
            PalsAdaptiveStartupConfigV5::from_json(
                &serde_json::to_string(&omitted).unwrap(),
                &lock
            )
            .is_err(),
            "missing explicit nullable selection {nullable}"
        );
    }
    let mut unknown = raw;
    unknown["native_options"]["automatic_cpu_fallback"] = serde_json::json!(true);
    assert!(
        PalsAdaptiveStartupConfigV5::from_json(&serde_json::to_string(&unknown).unwrap(), &lock)
            .is_err()
    );
    let mut changed = c.clone();
    changed.native_options.cache_public_memory = false;
    assert!(changed.validate_against(&lock).is_err());
    let mut extra = c;
    extra
        .endpoint
        .runtime
        .options
        .insert("unregistered_default".into(), "true".into());
    let next = relock(&mut extra, &lock);
    assert!(extra.validate_against(&next).is_err());
}

#[test]
fn cpu_and_cuda_select_their_own_explicit_control_inventory() {
    let (c, lock) = config();
    let mut cpu_with_cuda = c.clone();
    cpu_with_cuda.assets.push(PalsStartupAssetV5 {
        purpose: PalsStartupAssetPurposeV5::CudaControlInventory,
        artifact: artifact("fixtures/cuda-control.json"),
        snapshot_path: "inputs/runtime/cuda-control.json".into(),
    });
    assert!(cpu_with_cuda.validate_against(&lock).is_err());
    let mut cuda = c;
    cuda.endpoint.model.backend = PalsModelBackendV3::OrtCuda;
    cuda.endpoint.resources.gpu_count = 1;
    cuda.endpoint.resources.gpu_model = Some("source-only CUDA profile fixture".into());
    cuda.native_options.native_loading_profile = Some(PALS_NATIVE_EAGER_CUDA_PROFILE_V5.into());
    cuda.native_options.cuda_device_id = Some(0);
    cuda.native_options.cuda_arena_bytes = Some(2 << 30);
    cuda.endpoint.runtime.options = cuda.native_options.canonical_runtime_options();
    let next = relock(&mut cuda, &lock);
    assert!(cuda.validate_against(&next).is_err());
    cuda.assets.push(PalsStartupAssetV5 {
        purpose: PalsStartupAssetPurposeV5::CudaControlInventory,
        artifact: artifact("fixtures/cuda-control.json"),
        snapshot_path: "inputs/runtime/cuda-control.json".into(),
    });
    assert!(cuda.validate_against(&next).is_ok());
    let mut duplicate = cuda.clone();
    duplicate.assets.push(PalsStartupAssetV5 {
        purpose: PalsStartupAssetPurposeV5::CudaControlInventory,
        artifact: artifact("fixtures/other-cuda-control.json"),
        snapshot_path: "inputs/runtime/other-cuda-control.json".into(),
    });
    assert!(duplicate.validate_against(&next).is_err());
    for (device, arena, profile) in [
        (
            Some(-1),
            Some(2 << 30),
            Some(PALS_NATIVE_EAGER_CUDA_PROFILE_V5),
        ),
        (None, Some(2 << 30), Some(PALS_NATIVE_EAGER_CUDA_PROFILE_V5)),
        (Some(0), Some(0), Some(PALS_NATIVE_EAGER_CUDA_PROFILE_V5)),
        (
            Some(0),
            Some((6 << 30) + 1),
            Some(PALS_NATIVE_EAGER_CUDA_PROFILE_V5),
        ),
        (
            Some(0),
            Some(2 << 30),
            Some("experimental-cudnn-shim-lazy-v1"),
        ),
        (Some(0), Some(2 << 30), None),
    ] {
        let mut bad = cuda.clone();
        bad.native_options.cuda_device_id = device;
        bad.native_options.cuda_arena_bytes = arena;
        bad.native_options.native_loading_profile = profile.map(str::to_owned);
        bad.endpoint.runtime.options = bad.native_options.canonical_runtime_options();
        let bad_lock = relock(&mut bad, &next);
        assert!(bad.validate_against(&bad_lock).is_err());
    }
}

#[test]
fn startup_drain_and_warm_cannot_expand_registered_authority() {
    let (c, lock) = config();
    for (startup, drain) in [(0, 5_000), (120_001, 5_000), (10_000, 0), (10_000, 30_001)] {
        let mut bad = c.clone();
        bad.native_options.startup_ms = startup;
        bad.native_options.drain_ms = drain;
        bad.endpoint.runtime.options = bad.native_options.canonical_runtime_options();
        let next = relock(&mut bad, &lock);
        assert!(bad.validate_against(&next).is_err());
    }
    let mut unsupported = c.clone();
    unsupported.native_options.cudnn_conv_search = "experimental_default_device_warm".into();
    unsupported.native_options.device_public_memory = true;
    unsupported.endpoint.runtime.options = unsupported.native_options.canonical_runtime_options();
    let next = relock(&mut unsupported, &lock);
    assert!(unsupported.validate_against(&next).is_err());
    let mut warm = c.clone();
    warm.native_options.approximate_warm = true;
    warm.endpoint.runtime.options = warm.native_options.canonical_runtime_options();
    let next = relock(&mut warm, &lock);
    assert!(warm.validate_against(&next).is_err()); // Rust mock profile has no ORT Warm authority.
    warm.endpoint.model.backend = PalsModelBackendV3::OrtCpu;
    let next = relock(&mut warm, &next);
    assert!(warm.validate_against(&next).is_ok()); // Declaration only, no loaded Warm proof.
    let mut collection_endpoint = warm.endpoint.clone();
    collection_endpoint.mode = PalsEngineModeV5::Collection;
    assert!(
        warm.native_options
            .validate_against(&collection_endpoint, &next.manifest.window)
            .is_err()
    );
}

#[test]
fn process_evidence_rejects_other_owner_and_changed_observation() {
    let r = PalsResourceReadbackV5::with_unknown_checks(
        PalsResourceEvidenceProfileV5::CloudObserved,
        artifact("reports/resource.json"),
    );
    let bytes =
        pals_resource_readback_evidence_bytes_v5("endpoint", 47, "native-process-47", &r).unwrap();
    assert!(
        validate_pals_resource_readback_evidence_v5(
            &bytes,
            "endpoint",
            47,
            "native-process-47",
            &r
        )
        .is_ok()
    );
    assert!(
        validate_pals_resource_readback_evidence_v5(&bytes, "other", 47, "native-process-47", &r)
            .is_err()
    );
    assert!(
        validate_pals_resource_readback_evidence_v5(
            &bytes,
            "endpoint",
            48,
            "native-process-48",
            &r
        )
        .is_err()
    );
    let mut changed = r.clone();
    changed.cuda_provider_ready = PalsObservedV3::Observed {
        value: true,
        method: "changed observation fixture".into(),
    };
    assert!(
        validate_pals_resource_readback_evidence_v5(
            &bytes,
            "endpoint",
            47,
            "native-process-47",
            &changed
        )
        .is_err()
    );
    // Updating only the file's self hash cannot change its payload semantics.
    let mut rehashed = r;
    rehashed.evidence.sha256 = "0".repeat(64);
    assert!(
        validate_pals_resource_readback_evidence_v5(
            &bytes,
            "endpoint",
            47,
            "native-process-47",
            &rehashed
        )
        .is_ok()
    );
    let mut injected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    injected["undeclared"] = true.into();
    assert!(
        validate_pals_resource_readback_evidence_v5(
            &serde_json::to_vec(&injected).unwrap(),
            "endpoint",
            47,
            "native-process-47",
            &rehashed
        )
        .is_err()
    );
}
