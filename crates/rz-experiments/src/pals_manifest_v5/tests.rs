use super::*;

fn asset(path: &str) -> ArtifactRef {
    ArtifactRef {
        path: path.into(),
        sha256: "a".repeat(64),
        bytes: 1,
        source: "https://example.org/source".into(),
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
fn model() -> PalsModelSpecificationV5 {
    PalsModelSpecificationV5 {
        profile: "virtual_review_v1".into(),
        model_schema: PALS_V5_MODEL_SCHEMA.into(),
        model_semantics: PALS_V5_MODEL_SEMANTICS.into(),
        encoding_schema: PALS_V5_ENCODING_SCHEMA.into(),
        semantic_sha256: "c".repeat(64),
        encoding_sha256: "d".repeat(64),
        implementation_sha256: "e".repeat(64),
        weights: asset("weights/model.onnx"),
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
    }
}
fn engine() -> PalsEngineV5 {
    PalsEngineV5 {
        id: "a".into(),
        binary: asset("bin/rovezero"),
        source_commit: "f".repeat(40),
        mode: PalsEngineModeV5::Deployment,
        model: model(),
        cpu: PalsCpuEndpointsV5 {
            cpu_t: PalsCpuSelectionV5::Disabled,
            cpu_r: PalsCpuSelectionV5::Own,
            own: component("own-cpu/1"),
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
    }
}
fn manifest() -> PalsRunManifestV5 {
    PalsRunManifestV5 {
        schema_version: 5,
        run_id: "registered-cpu-validation".into(),
        contract_revision: PALS_V5_CONTRACT_REVISION.into(),
        purpose: PalsRunPurposeV5::CpuValidation,
        comparison: PalsComparisonV5::System,
        rules_profile: "standard-chess/1".into(),
        // Synthetic registry coverage for declaration/receipt tests. These
        // placeholders are never emitted as verified source execution data.
        source_pins: PALS_V5_BASE_SOURCE_COMPONENTS
            .into_iter()
            .chain(PALS_V5_NATIVE_SOURCE_COMPONENTS)
            .chain(PALS_V5_COLLECTION_SOURCE_COMPONENTS)
            .chain(PALS_V5_TRAINING_SOURCE_COMPONENTS)
            .chain(PALS_V5_PAIR_SOURCE_COMPONENTS)
            .map(|key| (key.into(), "f".repeat(64)))
            .collect(),
        engines: vec![engine()],
        pair: None,
        generation: Some(PalsGenerationRecipeV5::default()),
        window: PalsExecutionWindowV5 {
            window_id: "cpu-window-1".into(),
            wall_time_max_ms: 120000,
            cleanup_max_ms: 30000,
            output_bytes_max: 1 << 20,
            agent_renewable: true,
        },
        training_authorized: false,
    }
}
fn obs<T>(value: T) -> PalsObservedV3<T> {
    PalsObservedV3::Observed {
        value,
        method: "independent-owner-ledger".into(),
    }
}
fn receipt(lock: &PalsInputLockV5) -> PalsRunReceiptV5 {
    PalsRunReceiptV5 {
        schema_version: 5,
        domain: PALS_RECEIPT_V5_DOMAIN.into(),
        run_id: lock.manifest.run_id.clone(),
        lock_sha256: lock.canonical_sha256.clone(),
        source_verification: obs(PalsSourceVerificationV5 {
            components: lock.manifest.source_pins.clone(),
            binary_sha256_by_endpoint: lock
                .manifest
                .engines
                .iter()
                .map(|e| (e.id.clone(), e.binary.sha256.clone()))
                .collect(),
            content_verified: true,
            evidence: asset("reports/source-verification.json"),
        }),
        wall_time_ms: 100,
        cleanup_time_ms: 1,
        output_bytes: obs(1),
        training_executed: false,
        training: PalsObservedV3::Unknown,
        accepted: false,
        performance_eligible: false,
        strength_eligible: false,
        full_gc_requested: false,
        memory_contention: false,
        failures: BTreeSet::new(),
        endpoints: Vec::new(),
        games: Vec::new(),
    }
}
fn complete_endpoint() -> PalsEndpointReceiptV5 {
    PalsEndpointReceiptV5 {
        endpoint_id: "a".into(),
        runtime_owner: "owner-1".into(),
        model_verification: obs(PalsModelVerificationV5 {
            weights: model().weights,
            provenance: None,
            artifact_domain: PalsArtifactDomainV5::Untrained,
            checkpoint_domain: None,
            semantic_sha256: model().semantic_sha256,
            encoding_sha256: model().encoding_sha256,
            profile: model().profile,
            exported_roles: model().exported_roles,
            backend: model().backend,
            precision: model().precision,
            content_domain_verified: true,
            diagnostic_asset: false,
            evidence: asset("reports/model-verification.json"),
        }),
        resource_readback: obs(PalsResourceReadbackV5 {
            profile: PalsResourceEvidenceProfileV5::StrictLocal,
            limits_applied: obs(true),
            cpu_quota: obs(PalsCpuQuotaReadbackV5::Limited {
                quota_us: 200000,
                period_us: 100000,
                scope: PalsCpuQuotaScopeV5::ProcessTree,
            }),
            quota_ancestor_chain_complete: obs(true),
            allowed_logical_cpus: obs(vec![0, 1]),
            affinity_scope: PalsCpuAffinityScopeV5::Process,
            thread_inventory_complete: obs(true),
            threads: [
                PalsCpuActorV5::Manager,
                PalsCpuActorV5::NativeCaller,
                PalsCpuActorV5::OwnWorker { index: 0 },
            ]
            .into_iter()
            .enumerate()
            .map(|(index, actor)| PalsThreadAffinityReadbackV5 {
                actors: BTreeSet::from([actor]),
                process_id: 1,
                thread_id: index as u64 + 1,
                affinity: obs(vec![0, 1]),
            })
            .collect(),
            memory_high: obs(PalsMemoryLimitReadbackV5::Limited { bytes: 6 << 30 }),
            memory_max: obs(PalsMemoryLimitReadbackV5::Limited { bytes: 12 << 30 }),
            ort_settings: obs(PalsOrtSettingsReadbackV5 {
                intra_threads: 1,
                inter_threads: 1,
                spinning: false,
            }),
            smt_sibling_exclusion_verified: obs(false),
            total_observed_kernel_threads: obs(3),
            gpu_identity: PalsObservedV3::Unknown,
            cuda_provider_ready: PalsObservedV3::Unknown,
            evidence: asset("reports/resource-readback.json"),
        }),
        coordinator: obs(PalsCoordinatorObservationV5 {
            role_request_history_peak: 2,
            cpu_execution_history_peak: 2,
            frame_genealogy_peak: 2,
            ranking_fanout_history_peak: 1,
            validator_source_contexts_peak: PalsObservedV3::Unknown,
            validator_action_keys_peak: PalsObservedV3::Unknown,
            view_submitted: 2,
            eligible_view_consumed: 2,
            tickets_submitted: 2,
            tickets_terminal_before_dispatch: 0,
            tickets_cache_completed: 0,
            tickets_dispatched: 2,
            tickets_queued: 0,
            physical_in_flight: 0,
            physical_completion_unknown: 0,
            tickets_physically_completed: 2,
            tickets_accepted: 2,
            tickets_released: 2,
            tickets_quarantined: 0,
            nn_inputs_completed: 2,
            nn_inputs_accepted: 2,
            cache_evaluations_accepted: 0,
            search_evaluations_consumed: 2,
            unused_results: 0,
            cpu_credit_limit: 4096,
            cpu_spent: 100,
            cpu_outstanding_reserved: 0,
            yield_requests: 1,
            yield_acks: 1,
            natural_completions: 1,
            physical_state: PalsPhysicalStateV3::Completed,
            owner_joined: true,
            archive_closed: true,
            evidence: asset("reports/owner.json"),
        }),
        native_numerical_check: PalsObservedV3::Unknown,
        process_tree_gone: obs(true),
        host_peak_bytes: PalsObservedV3::Unknown,
        vram_peak_bytes: PalsObservedV3::Unknown,
        actual_tf32: PalsObservedV3::Unknown,
        v_generated: obs(false),
        v_loaded: obs(false),
        v_called: obs(false),
    }
}

fn complete_numerical_check(declared: &PalsEngineV5) -> PalsNativeNumericalCheckV5 {
    PalsNativeNumericalCheckV5 {
        weights_sha256: declared.model.weights.sha256.clone(),
        semantic_sha256: declared.model.semantic_sha256.clone(),
        encoding_sha256: declared.model.encoding_sha256.clone(),
        runtime_implementation_sha256: declared.runtime.implementation_sha256.clone(),
        source_commit: declared.source_commit.clone(),
        passed: true,
        logits_absolute_tolerance: "1e-4".into(),
        logits_relative_tolerance: "1e-3".into(),
        policy_wdl_absolute_tolerance: "1e-4".into(),
        evidence: asset("reports/independent-numerical.json"),
    }
}

#[test]
fn standalone_endpoint_pending_is_not_native_session_acceptance() {
    let d = engine();
    let pending = PalsEndpointReceiptV5::with_unknown_checks(d.id.clone(), "owner-1".into());
    validate_pals_endpoint_observation_v5(&pending, &d, false, false, false).unwrap();
    assert!(validate_pals_endpoint_observation_v5(&pending, &d, true, false, false).is_err());
    assert!(validate_pals_endpoint_observation_v5(&pending, &d, false, true, false).is_err());
    let mut wrong = pending;
    wrong.endpoint_id = "other".into();
    assert!(validate_pals_endpoint_observation_v5(&wrong, &d, false, false, false).is_err());
}

#[test]
fn standalone_native_session_binds_numeric_work_and_physical_closure() {
    let d = engine();
    let mut e = complete_endpoint();
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
    e.native_numerical_check = obs(complete_numerical_check(&d));
    validate_pals_endpoint_observation_v5(&e, &d, true, false, false).unwrap();
    if let PalsObservedV3::Observed { value, .. } = &mut e.native_numerical_check {
        value.source_commit = "c".repeat(40);
    }
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
    e.native_numerical_check = obs(complete_numerical_check(&d));
    if let PalsObservedV3::Observed { value, .. } = &mut e.coordinator {
        value.nn_inputs_completed = 0;
        value.nn_inputs_accepted = 0;
        value.search_evaluations_consumed = 0;
    }
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
    e = complete_endpoint();
    e.native_numerical_check = obs(complete_numerical_check(&d));
    if let PalsObservedV3::Observed { value, .. } = &mut e.coordinator {
        value.tickets_released = 1;
    }
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
    e = complete_endpoint();
    e.native_numerical_check = obs(complete_numerical_check(&d));
    e.v_loaded = PalsObservedV3::Unknown;
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
}

#[test]
fn standalone_cuda_session_requires_actual_provider_and_precision_readback() {
    let mut d = engine();
    d.model.backend = PalsModelBackendV3::OrtCuda;
    d.resources.gpu_count = 1;
    d.resources.gpu_model = Some("NVIDIA L4".into());
    let mut e = complete_endpoint();
    if let PalsObservedV3::Observed { value, .. } = &mut e.model_verification {
        value.backend = PalsModelBackendV3::OrtCuda;
    }
    e.native_numerical_check = obs(complete_numerical_check(&d));
    if let PalsObservedV3::Observed { value, .. } = &mut e.resource_readback {
        value.gpu_identity = obs(PalsGpuIdentityReadbackV5 {
            model_id: "NVIDIA L4".into(),
            reported_name: "NVIDIA L4".into(),
            uuid: "GPU-fixture".into(),
            total_bytes: 24 << 30,
            device_count: 1,
        });
        value.cuda_provider_ready = obs(true);
    }
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
    e.actual_tf32 = obs(false);
    validate_pals_endpoint_observation_v5(&e, &d, true, false, false).unwrap();
    if let PalsObservedV3::Observed { value, .. } = &mut e.resource_readback {
        value.cuda_provider_ready = PalsObservedV3::Unknown;
    }
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
}

#[test]
fn root_history_limits_are_explicit_and_not_an_aggregate_budget() {
    let m = manifest();
    assert_eq!(
        m.engines[0].coordinator.replay_history_per_namespace_max,
        65_536
    );
    assert_eq!(m.engines[0].coordinator.frame_genealogy_max, 1_024);
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    let mut e = complete_endpoint();
    if let PalsObservedV3::Observed { value: c, .. } = &mut e.coordinator {
        c.role_request_history_peak = 65_536;
        c.cpu_execution_history_peak = 65_536;
        c.frame_genealogy_peak = 1_024;
        c.ranking_fanout_history_peak = 1_024;
    }
    r.endpoints.push(e);
    r.validate_against(&lock).unwrap();

    let mut changed = m.clone();
    changed.engines[0]
        .coordinator
        .replay_history_per_namespace_max = 65_535;
    assert!(changed.validate().is_err());
    changed = m.clone();
    changed.engines[0].coordinator.frame_genealogy_max = 1_025;
    assert!(changed.validate().is_err());

    let bytes = m.canonical_bytes().unwrap();
    let canonical = std::str::from_utf8(&bytes).unwrap();
    assert!(canonical.contains("\"replay_history_per_namespace_max\":65536"));
    assert!(canonical.contains("\"frame_genealogy_max\":1024"));
    let mut value = serde_json::to_value(m).unwrap();
    value["engines"][0]["coordinator"]
        .as_object_mut()
        .unwrap()
        .remove("replay_history_per_namespace_max");
    assert!(PalsRunManifestV5::from_json(&serde_json::to_string(&value).unwrap()).is_err());
}

#[test]
fn history_overflow_requires_a_preserved_capacity_failure() {
    let lock = manifest().lock().unwrap();
    for namespace in ["role", "cpu", "frames", "ranking"] {
        let mut r = receipt(&lock);
        r.accepted = true;
        let mut e = complete_endpoint();
        if let PalsObservedV3::Observed { value: c, .. } = &mut e.coordinator {
            match namespace {
                "role" => c.role_request_history_peak = 65_537,
                "cpu" => c.cpu_execution_history_peak = 65_537,
                "frames" => c.frame_genealogy_peak = 1_025,
                "ranking" => {
                    c.frame_genealogy_peak = 1_025;
                    c.ranking_fanout_history_peak = 1_025;
                }
                _ => unreachable!(),
            }
        }
        r.endpoints.push(e);
        assert!(r.validate_against(&lock).is_err());
        r.accepted = false;
        assert!(r.validate_against(&lock).is_err());
        r.failures.insert(PalsFailureV5::CoordinatorCapacity);
        r.validate_against(&lock).unwrap();
        let saved = PalsRunReceiptV5::from_json(&r.to_json(&lock).unwrap(), &lock).unwrap();
        assert_eq!(saved.endpoints, r.endpoints);
    }
}

#[test]
fn ranking_peak_preserves_its_frame_genealogy_subset() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    let mut e = complete_endpoint();
    if let PalsObservedV3::Observed { value: c, .. } = &mut e.coordinator {
        c.ranking_fanout_history_peak = c.frame_genealogy_peak + 1;
    }
    r.endpoints.push(e);
    r.failures.insert(PalsFailureV5::CoordinatorCapacity);
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn validator_replay_limits_are_distinct_canonical_fields() {
    let m = manifest();
    assert_eq!(m.engines[0].coordinator.validator_source_contexts_max, 128);
    assert_eq!(m.engines[0].coordinator.validator_action_keys_max, 896);
    let canonical = String::from_utf8(m.canonical_bytes().unwrap()).unwrap();
    assert!(canonical.contains("\"validator_source_contexts_max\":128"));
    assert!(canonical.contains("\"validator_action_keys_max\":896"));
    for field in ["validator_source_contexts_max", "validator_action_keys_max"] {
        let mut value = serde_json::to_value(&m).unwrap();
        value["engines"][0]["coordinator"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(PalsRunManifestV5::from_json(&serde_json::to_string(&value).unwrap()).is_err());
    }
    let mut changed = m.clone();
    changed.engines[0].coordinator.validator_source_contexts_max = 129;
    assert!(changed.validate().is_err());
    changed = m;
    changed.engines[0].coordinator.validator_action_keys_max = 897;
    assert!(changed.validate().is_err());
}

#[test]
fn validator_replay_unknown_is_not_observed_zero_or_v_execution_acceptance() {
    let d = engine();
    let mut e = complete_endpoint();
    e.native_numerical_check = obs(complete_numerical_check(&d));
    // V-free acceptance does not invent a guard observation that never occurred.
    validate_pals_endpoint_observation_v5(&e, &d, true, false, false).unwrap();
    let mut d = collection_engine();
    d.resources.profile_id = "cloud-observed-collection-v1".into();
    d.resources.evidence_profile = PalsResourceEvidenceProfileV5::CloudObserved;
    d.resources.cloud_or_topology_unknown = true;
    let mut e = complete_endpoint();
    e.native_numerical_check = obs(complete_numerical_check(&d));
    if let PalsObservedV3::Observed { value, .. } = &mut e.model_verification {
        value.exported_roles.insert(PalsRoleV3::Validator);
    }
    if let PalsObservedV3::Observed { value, .. } = &mut e.resource_readback {
        value.profile = PalsResourceEvidenceProfileV5::CloudObserved;
    }
    e.v_generated = obs(true);
    e.v_loaded = obs(true);
    e.v_called = obs(true);
    validate_pals_endpoint_observation_v5(&e, &d, false, false, false).unwrap();
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
    if let PalsObservedV3::Observed { value, .. } = &mut e.coordinator {
        value.validator_source_contexts_peak = obs(128);
        value.validator_action_keys_peak = obs(896);
    }
    validate_pals_endpoint_observation_v5(&e, &d, true, false, false).unwrap();
    let c = observed(&e.coordinator).unwrap();
    assert_eq!(c.nn_inputs_completed, 2);
    assert_eq!(c.search_evaluations_consumed, 2);
    if let PalsObservedV3::Observed { value, .. } = &mut e.coordinator {
        value.validator_action_keys_peak = PalsObservedV3::Unknown;
    }
    assert!(validate_pals_endpoint_observation_v5(&e, &d, true, false, false).is_err());
}

#[test]
fn validator_replay_overflow_retains_typed_capacity_failure() {
    let lock = manifest().lock().unwrap();
    for (sources, actions) in [(129, 0), (129, 897)] {
        let mut r = receipt(&lock);
        let mut e = complete_endpoint();
        if let PalsObservedV3::Observed { value, .. } = &mut e.coordinator {
            value.validator_source_contexts_peak = obs(sources);
            value.validator_action_keys_peak = obs(actions);
        }
        r.endpoints.push(e);
        assert!(r.validate_against(&lock).is_err());
        r.failures.insert(PalsFailureV5::CoordinatorCapacity);
        r.validate_against(&lock).unwrap();
        r.accepted = true;
        assert!(r.validate_against(&lock).is_err());
    }
    let mut r = receipt(&lock);
    let mut e = complete_endpoint();
    if let PalsObservedV3::Observed { value, .. } = &mut e.coordinator {
        value.validator_source_contexts_peak = obs(1);
        value.validator_action_keys_peak = obs(8);
    }
    r.endpoints.push(e);
    r.failures.insert(PalsFailureV5::CoordinatorCapacity);
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn generation_domains_match_actual_producer_and_trainer() {
    let domains = BTreeMap::from(PALS_V5_GENERATION_CONSUMPTION_DOMAINS);
    assert_eq!(domains["records"], "rz-pals-generation-data/3");
    assert_eq!(domains["input_seal"], "rz-pals-generation-input/3");
    assert_eq!(domains["train_spec"], "rz-pals-train-generation/1");
    assert_eq!(
        domains["checkpoint"],
        "rz-pals-actual-generation-checkpoint/1"
    );
    assert_eq!(domains["export"], "rz-pals-actual-generation-export/1");
    assert_eq!(
        domains["chunk_manifest"],
        "rz-pals-generation-chunk-manifest/1"
    );
    let mut recipe = PalsGenerationRecipeV5::default();
    recipe.record_domain = "rz-pals-training-record-v3/1".into();
    assert!(recipe.validate().is_err());
}

#[test]
fn source_declarations_require_independent_content_verification() {
    let mut m = manifest();
    m.source_pins.remove("cpu_credit");
    assert!(m.validate().is_err());
    let m = manifest();
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    r.source_verification = PalsObservedV3::Unknown;
    assert!(r.validate_against(&lock).is_err());
    r.accepted = false;
    r.validate_against(&lock).unwrap();
    r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    if let PalsObservedV3::Observed { value, .. } = &mut r.source_verification {
        value.content_verified = false;
    }
    assert!(r.validate_against(&lock).is_err());
    if let PalsObservedV3::Observed { value, .. } = &mut r.source_verification {
        value.content_verified = true;
        value.components.insert("cpu_credit".into(), "e".repeat(64));
    }
    assert!(r.validate_against(&lock).is_err());
    r.accepted = false;
    assert!(r.validate_against(&lock).is_err());
    r.failures.insert(PalsFailureV5::SourceIdentityViolation);
    r.validate_against(&lock).unwrap();
}

#[test]
fn cloud_partial_readbacks_preserve_unknown_without_strict_resource_claims() {
    let mut m = manifest();
    m.engines[0].resources.profile_id = "colab-observed-shared-control-v1".into();
    m.engines[0].resources.evidence_profile = PalsResourceEvidenceProfileV5::CloudObserved;
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    let mut e = complete_endpoint();
    let mut cloud = PalsResourceReadbackV5::with_unknown_checks(
        PalsResourceEvidenceProfileV5::CloudObserved,
        asset("reports/cloud-resource.json"),
    );
    cloud.limits_applied = obs(false);
    cloud.cpu_quota = obs(PalsCpuQuotaReadbackV5::Unlimited {
        scope: PalsCpuQuotaScopeV5::VisibleAncestors,
    });
    cloud.allowed_logical_cpus = obs(vec![0, 1, 2, 3]);
    cloud.affinity_scope = PalsCpuAffinityScopeV5::ThreadGroupLeader;
    e.resource_readback = obs(cloud.clone());
    r.endpoints.push(e);
    r.validate_against(&lock).unwrap();
    let saved = PalsRunReceiptV5::from_json(&r.to_json(&lock).unwrap(), &lock).unwrap();
    assert_eq!(
        observed(&saved.endpoints[0].resource_readback),
        Some(&cloud)
    );
    assert!(matches!(cloud.memory_high, PalsObservedV3::Unknown));
    r.performance_eligible = true;
    r.endpoints[0].host_peak_bytes = obs(100);
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn strict_resource_checks_reject_unknown_limits_and_group_leader_only_masks() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    if let PalsObservedV3::Observed { value, .. } = &mut r.endpoints[0].resource_readback {
        value.memory_high = PalsObservedV3::Unknown;
    }
    assert!(r.validate_against(&lock).is_err());
    r.endpoints[0] = complete_endpoint();
    if let PalsObservedV3::Observed { value, .. } = &mut r.endpoints[0].resource_readback {
        value.affinity_scope = PalsCpuAffinityScopeV5::ThreadGroupLeader;
        value.thread_inventory_complete = PalsObservedV3::Unknown;
    }
    assert!(r.validate_against(&lock).is_err());
    r.accepted = false;
    r.validate_against(&lock).unwrap();
}

#[test]
fn exact_cpu_quota_and_unlimited_memory_have_distinct_readback_types() {
    let exact = PalsCpuQuotaReadbackV5::Limited {
        quota_us: 200000,
        period_us: 100000,
        scope: PalsCpuQuotaScopeV5::ProcessTree,
    };
    assert!(exact.matches_millicpus(2000));
    let close = PalsCpuQuotaReadbackV5::Limited {
        quota_us: 199999,
        period_us: 100000,
        scope: PalsCpuQuotaScopeV5::ProcessTree,
    };
    assert!(!close.matches_millicpus(2000));
    assert!(
        !PalsCpuQuotaReadbackV5::WindowsHardCap {
            rate_per_10000: 5000,
            scope: PalsCpuQuotaScopeV5::ImmediateJobObject,
        }
        .matches_millicpus(2000)
    );
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    if let PalsObservedV3::Observed { value, .. } = &mut r.endpoints[0].resource_readback {
        value.memory_max = obs(PalsMemoryLimitReadbackV5::Unlimited);
    }
    assert!(r.validate_against(&lock).is_err());
    r.accepted = false;
    r.validate_against(&lock).unwrap();
}

#[test]
fn physical_thread_identity_is_unique_even_with_different_actor_labels() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    if let PalsObservedV3::Observed { value, .. } = &mut r.endpoints[0].resource_readback {
        for thread in &mut value.threads {
            thread.thread_id = 1;
        }
        assert!(!value.matches_applied_plan(&lock.manifest.engines[0]));
    }
    assert!(r.validate_against(&lock).is_err());
    r.accepted = false;
    r.failures.insert(PalsFailureV5::ResourceAdmission);
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn private_worker_indices_require_distinct_physical_threads() {
    let mut m = manifest();
    m.engines[0].coordinator.slices.workers = 2;
    m.engines[0].resources.profile_id = "shared-control-cpu3-workers2-v1".into();
    m.engines[0].resources.cpu_quota_millis = 3000;
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    let mut endpoint = complete_endpoint();
    if let PalsObservedV3::Observed { value, .. } = &mut endpoint.resource_readback {
        value.cpu_quota = obs(PalsCpuQuotaReadbackV5::Limited {
            quota_us: 300000,
            period_us: 100000,
            scope: PalsCpuQuotaScopeV5::ProcessTree,
        });
        value.threads[2]
            .actors
            .insert(PalsCpuActorV5::OwnWorker { index: 1 });
        assert!(!value.matches_applied_plan(&m.engines[0]));
    }
    r.endpoints.push(endpoint);
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn synchronous_multi_role_thread_is_counted_once_and_not_an_async_worker() {
    let mut m = manifest();
    m.engines[0].coordinator.policy = PalsCoordinatorPolicyV5::Synchronous;
    m.engines[0].coordinator.manager_executes_blocking_work = true;
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    let mut endpoint = complete_endpoint();
    if let PalsObservedV3::Observed { value, .. } = &mut endpoint.resource_readback {
        value.threads[0].actors.insert(PalsCpuActorV5::NativeCaller);
        value.threads.remove(1);
        value.total_observed_kernel_threads = obs(2);
        assert!(value.matches_applied_plan(&m.engines[0]));
    }
    r.endpoints.push(endpoint);
    r.validate_against(&lock).unwrap();
    m.engines[0].coordinator.policy = PalsCoordinatorPolicyV5::CoordinatorFixed;
    m.engines[0].coordinator.manager_executes_blocking_work = false;
    let async_lock = m.lock().unwrap();
    r.lock_sha256 = async_lock.canonical_sha256.clone();
    assert!(r.validate_against(&async_lock).is_err());
    r.accepted = false;
    r.failures.insert(PalsFailureV5::ResourceAdmission);
    r.validate_against(&async_lock).unwrap();
}

fn collection_engine() -> PalsEngineV5 {
    let mut e = engine();
    e.mode = PalsEngineModeV5::Collection;
    e.model.exported_roles.insert(PalsRoleV3::Validator);
    e.v_generate = true;
    e.v_load = true;
    e.v_call = true;
    e.cpu.cpu_t = PalsCpuSelectionV5::Stockfish19Full;
    let mut binary = asset("bin/stockfish");
    binary.license = "GPL-3.0-or-later".into();
    e.cpu.external = Some(PalsExternalCpuV5 {
        binary,
        profile: asset("profiles/stockfish.json"),
        source_commit: PALS_V5_STOCKFISH_SOURCE.into(),
        version: "Stockfish 19".into(),
        license: "GPL-3.0-or-later".into(),
        options: BTreeMap::from([
            ("Skill Level".into(), "20".into()),
            ("UCI_LimitStrength".into(), "false".into()),
            ("Threads".into(), "1".into()),
            ("Hash".into(), "256".into()),
            ("Ponder".into(), "false".into()),
            ("NumaPolicy".into(), "none".into()),
            ("SyzygyPath".into(), String::new()),
            ("MultiPV".into(), "1".into()),
        ]),
        quantum_min_ms: 100,
        stop_drain_reserve_ms: 500,
        requested_nodes_max: 65536,
        conditional_multipv_max: 4,
        reported_nodes_are_native_credit: false,
        stop_bestmove_readyok_before_next_go: true,
        exact_stack_resume: false,
        sigstop: false,
    });
    e
}

#[test]
fn collection_failure_preserves_actual_v_not_loaded() {
    let mut m = manifest();
    m.engines = vec![collection_engine()];
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    r.failures.insert(PalsFailureV5::Cuda);
    r.endpoints.push(complete_endpoint());
    // Actual V=false may contradict the requested full Collection mode after
    // initialization failure. The failed receipt retains that observation.
    r.validate_against(&lock).unwrap();
    r.accepted = true;
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn training_acceptance_cannot_be_synthesized_from_declarations() {
    let mut m = manifest();
    m.purpose = PalsRunPurposeV5::TrainingGeneration;
    m.training_authorized = true;
    m.engines = vec![collection_engine()];
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn nn_acceptance_requires_numerical_evidence_bound_to_exact_model() {
    let mut m = manifest();
    m.purpose = PalsRunPurposeV5::NeuralValidation;
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    assert!(r.validate_against(&lock).is_err());
    r.endpoints[0].native_numerical_check = obs(PalsNativeNumericalCheckV5 {
        weights_sha256: model().weights.sha256,
        semantic_sha256: model().semantic_sha256,
        encoding_sha256: model().encoding_sha256,
        runtime_implementation_sha256: component("runtime/1").implementation_sha256,
        source_commit: engine().source_commit,
        passed: true,
        logits_absolute_tolerance: "1e-4".into(),
        logits_relative_tolerance: "1e-3".into(),
        policy_wdl_absolute_tolerance: "1e-4".into(),
        evidence: asset("reports/numeric.json"),
    });
    r.validate_against(&lock).unwrap();
    if let PalsObservedV3::Observed { value, .. } = &mut r.endpoints[0].native_numerical_check {
        value.weights_sha256 = "d".repeat(64);
    }
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn cached_evaluation_does_not_fake_new_physical_input() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    let mut e = complete_endpoint();
    if let PalsObservedV3::Observed { value: c, .. } = &mut e.coordinator {
        c.tickets_dispatched = 1;
        c.tickets_physically_completed = 1;
        c.tickets_cache_completed = 1;
        c.nn_inputs_completed = 1;
        c.nn_inputs_accepted = 1;
        c.cache_evaluations_accepted = 1;
    }
    r.endpoints.push(e);
    r.validate_against(&lock).unwrap();
    if let PalsObservedV3::Observed { value: c, .. } = &mut r.endpoints[0].coordinator {
        c.tickets_cache_completed = 0;
    }
    assert!(r.validate_against(&lock).is_err()); // lost second ticket
}

#[test]
fn provenance_unknown_and_tf32_mismatch_cannot_be_accepted() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    r.endpoints[0].model_verification = PalsObservedV3::Unknown;
    assert!(r.validate_against(&lock).is_err());
    r.endpoints[0] = complete_endpoint();
    r.endpoints[0].actual_tf32 = obs(true);
    assert!(r.validate_against(&lock).is_err());
    r.accepted = false;
    r.failures.insert(PalsFailureV5::SelectedModeViolation);
    r.validate_against(&lock).unwrap();
}

#[test]
fn v5_lock_round_trip_and_recipe_are_declarations_only() {
    let m = manifest();
    let lock = m.lock().unwrap();
    assert_eq!(
        PalsInputLockV5::from_json(&lock.to_json().unwrap()).unwrap(),
        lock
    );
    let r = receipt(&lock);
    assert!(!r.training_executed);
    assert_eq!(
        PalsRunReceiptV5::from_json(&r.to_json(&lock).unwrap(), &lock).unwrap(),
        r
    );
}

#[test]
fn duplicate_unknown_fields_and_version_relabelling_are_rejected() {
    let source = serde_json::to_string(&manifest()).unwrap();
    let duplicate = source.replacen("{", "{\"schema_version\":5,", 1);
    assert!(PalsRunManifestV5::from_json(&duplicate).is_err());
    let unknown = source.replacen("{", "{\"unknown\":1,", 1);
    assert!(PalsRunManifestV5::from_json(&unknown).is_err());
    let mut m = manifest();
    m.schema_version = 4;
    assert!(m.validate().is_err());
    assert!(crate::PalsRunManifestV4::from_json(&source).is_err());
    m = manifest();
    m.contract_revision = "pals/0.2".into();
    assert!(m.validate().is_err());
}

#[test]
fn canonical_lock_seals_ordered_semantics_and_domain() {
    let m = manifest();
    let bytes = m.canonical_bytes().unwrap();
    assert!(bytes.starts_with(b"[\"rz-pals-execution-v5/1\",{"));
    let canonical: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        canonical[1]["engines"][0]["model"]["line_buckets"],
        serde_json::json!([8, 16, 32, 64, 128, 256])
    );
    let pretty = serde_json::to_string_pretty(&m).unwrap();
    assert_eq!(
        PalsRunManifestV5::from_json(&pretty)
            .unwrap()
            .lock()
            .unwrap(),
        m.lock().unwrap()
    );
    let mut bad = m.lock().unwrap();
    bad.domain = "rz-pals-execution-v4/1".into();
    assert!(bad.verify().is_err());
    bad = m.lock().unwrap();
    bad.manifest.engines[0].context.token_budget = 128;
    assert!(bad.verify().is_err());
}

#[test]
fn context_domain_has_fixed_canonical_vector() {
    let expected = br#"["rz-pals-execution-v5/1",{"selection":"virtual_review","token_budget":512,"selected_records_max":64,"active_bundles_max":256,"anchors_max":4,"high_water_percent":80,"target_water_percent":55,"protected_eligible_consumptions":2,"count_query_duplicates":true,"include_actual_game_history_tokens":false,"atomic_bundle_selection":true,"rules_rebase_validation":true,"archive_evidence_preserved":true}]"#;
    let bytes = encoded(
        &(PALS_MANIFEST_V5_DOMAIN, PalsReviewContextV5::default()),
        false,
    )
    .unwrap();
    assert_eq!(bytes, expected);
    // SHA computed independently over the 398-byte registered JSON vector.
    assert_eq!(
        digest(&bytes),
        "2e081e8cbe41973a3190e5600599fb49eb4d360431f1534ea87f7d94d7097176"
    );
}

#[test]
fn actual_history_tokens_and_context_limit_violations_are_rejected() {
    for budget in [128, 256, 512] {
        let mut m = manifest();
        m.engines[0].context.token_budget = budget;
        m.validate().unwrap();
    }
    for change in 0..5 {
        let mut m = manifest();
        let c = &mut m.engines[0].context;
        match change {
            0 => c.token_budget = 513,
            1 => c.selected_records_max = 65,
            2 => c.anchors_max = 5,
            3 => c.include_actual_game_history_tokens = true,
            _ => c.count_query_duplicates = false,
        }
        assert!(m.validate().is_err());
    }
}

#[test]
fn v_free_deployment_and_full_collection_are_separate() {
    let mut e = engine();
    e.v_load = true;
    assert!(e.validate().is_err());
    e = engine();
    e.model.exported_roles.insert(PalsRoleV3::Validator);
    assert!(e.validate().is_err());
    e = engine();
    e.mode = PalsEngineModeV5::Collection;
    e.v_generate = true;
    e.v_load = true;
    e.v_call = true;
    assert!(e.validate().is_err()); // missing V export and actual CPU_T selection
}

#[test]
fn cpu_worker_expansion_and_dedicated_control_need_new_resource_profile() {
    let mut m = manifest();
    m.engines[0].coordinator.slices.workers = 2;
    assert!(m.validate().is_err());
    let r = &mut m.engines[0].resources;
    r.profile_id = "cpu4-workers2".into();
    r.cpu_quota_millis = 4000;
    m.validate().unwrap();
    m.engines[0].resources.topology = PalsTopologyPolicyV5::DedicatedControl;
    assert!(m.validate().is_err());
    let r = &mut m.engines[0].resources;
    r.cloud_or_topology_unknown = false;
    r.allowed_logical_cpus = vec![0, 1, 2];
    r.dedicated_manager_cpu = Some(0);
    r.worker_logical_cpus = vec![1, 2];
    m.validate().unwrap();
}

#[test]
fn unsupported_line_bucket_ids_and_precision_do_not_pass_shape_checks() {
    for which in 0..4 {
        let mut m = manifest();
        let model = &mut m.engines[0].model;
        match which {
            0 => model.line_buckets[0] = 7,
            1 => model.ids_as_numeric_features = true,
            2 => model.precision = PalsPrecisionV3::Fp16,
            _ => model.one_dynamic_line_graph = false,
        }
        assert!(m.validate().is_err());
    }
}

#[test]
fn actual_updates_require_explicit_training_authorization() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.training_executed = true;
    assert!(r.validate_against(&lock).is_err());
    let mut m = manifest();
    m.purpose = PalsRunPurposeV5::TrainingGeneration;
    assert!(m.validate().is_err());
    let mut recipe = PalsGenerationRecipeV5::default();
    recipe.shared_optimizer_owners = 2;
    assert!(recipe.validate().is_err());
    recipe = PalsGenerationRecipeV5::default();
    recipe.all_masked_is_update = true;
    assert!(recipe.validate().is_err());
}

#[test]
fn diagnostic_domain_cannot_be_relabelled_as_real_generation() {
    let mut m = manifest();
    let e = &mut m.engines[0].model;
    e.artifact_domain = PalsArtifactDomainV5::RealGeneration;
    e.provenance = Some(asset("reports/diagnostic.json"));
    e.checkpoint_domain = Some(PALS_V5_DIAGNOSTIC_CHECKPOINT_DOMAIN.into());
    assert!(m.validate().is_err());
}

#[test]
fn unknown_observations_remain_valid_ineligible_evidence() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    let mut endpoint = complete_endpoint();
    endpoint.resource_readback = PalsObservedV3::Unknown;
    endpoint.coordinator = PalsObservedV3::Unknown;
    r.endpoints.push(endpoint);
    r.validate_against(&lock).unwrap();
    r.accepted = true;
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn completed_functional_receipt_does_not_claim_unobserved_peaks_or_tf32() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.endpoints.push(complete_endpoint());
    r.accepted = true;
    r.validate_against(&lock).unwrap();
    assert!(matches!(
        r.endpoints[0].vram_peak_bytes,
        PalsObservedV3::Unknown
    ));
    r.performance_eligible = true;
    assert!(r.validate_against(&lock).is_err());
    r.endpoints[0].host_peak_bytes = obs(123);
    r.validate_against(&lock).unwrap();
    r.full_gc_requested = true;
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn credit_overflow_pending_reservation_and_unknown_physical_release_are_rejected() {
    let lock = manifest().lock().unwrap();
    for case in 0..4 {
        let mut r = receipt(&lock);
        r.accepted = true;
        let mut e = complete_endpoint();
        if let PalsObservedV3::Observed { value: c, .. } = &mut e.coordinator {
            match case {
                0 => {
                    c.cpu_spent = u64::MAX;
                    c.cpu_outstanding_reserved = 1;
                }
                1 => c.cpu_outstanding_reserved = 1,
                2 => c.physical_state = PalsPhysicalStateV3::Unknown,
                _ => c.tickets_released = 1,
            }
        }
        r.endpoints.push(e);
        assert!(r.validate_against(&lock).is_err());
    }
}

#[test]
fn readyok_is_not_resource_readback() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    let mut e = complete_endpoint();
    if let PalsObservedV3::Observed { method, .. } = &mut e.resource_readback {
        *method = "readyok".into();
    }
    r.endpoints.push(e);
    assert!(r.validate_against(&lock).is_err());
}

#[test]
fn coordinator_comparison_changes_only_registered_policy() {
    let mut m = manifest();
    let mut a = engine();
    a.coordinator.policy = PalsCoordinatorPolicyV5::Synchronous;
    let mut b = a.clone();
    b.id = "b".into();
    b.coordinator.policy = PalsCoordinatorPolicyV5::CoordinatorFixed;
    m.engines = vec![a, b];
    m.comparison = PalsComparisonV5::SynchronousToFixed;
    m.validate().unwrap();
    m.engines[1].context.token_budget = 256;
    assert!(m.validate().is_err());
}

fn legacy_engine() -> PalsEngineV5 {
    let mut e = engine();
    e.model.profile = "full_line_interaction_v2".into();
    e.model.model_schema = "rovezero.pals-model.v2".into();
    e.model.model_semantics = "rovezero.pals-model-semantics.v2".into();
    e.model.encoding_schema = "rovezero.pals-board-records.v2".into();
    e.model.record_feature_axes = 16;
    e.model.query_feature_axes = 16;
    e.model.max_records = 128;
    e.model.max_anchors = 1;
    e.model.line_buckets = vec![256];
    e.model.one_dynamic_line_graph = false;
    e.model.weights = asset("weights/legacy.onnx");
    e.context.selection = PalsViewSelectionV5::V2Selected;
    e.context.anchors_max = 1;
    e
}

#[test]
fn legacy_view_and_compact_comparisons_preserve_exact_model_domains() {
    let mut m = manifest();
    let mut a = legacy_engine();
    a.context.selection = PalsViewSelectionV5::V2Full;
    a.context.selected_records_max = 128;
    let mut b = legacy_engine();
    b.id = "b".into();
    m.engines = vec![a, b];
    m.comparison = PalsComparisonV5::V2FullToSelected;
    m.validate().unwrap();
    assert_eq!(m.engines[0].model.model_schema, "rovezero.pals-model.v2");
    let mut b = engine();
    b.id = "b".into();
    m.engines = vec![legacy_engine(), b];
    m.comparison = PalsComparisonV5::V2ToCompact;
    m.validate().unwrap();
    m.engines[1].context.token_budget = 256;
    assert!(m.validate().is_err());
}

#[test]
fn quarantined_completion_unknown_keeps_explicit_failure() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    let mut e = complete_endpoint();
    if let PalsObservedV3::Observed { value: c, .. } = &mut e.coordinator {
        c.tickets_physically_completed = 1;
        c.tickets_accepted = 1;
        c.tickets_released = 1;
        c.physical_completion_unknown = 1;
        c.tickets_quarantined = 1;
        c.physical_state = PalsPhysicalStateV3::Quarantined;
    }
    r.endpoints.push(e);
    assert!(r.validate_against(&lock).is_err());
    r.failures.insert(PalsFailureV5::PhysicalCompletionUnknown);
    r.validate_against(&lock).unwrap();
}

#[test]
fn incomplete_game_cannot_create_rules_wdl() {
    let pair = PalsPairPlanV5 {
        opening_id: "opening".into(),
        position_command: "position startpos".into(),
        white_order: ["a".into(), "b".into()],
        base_ms: 120000,
        increment_ms: 1000,
        max_plies: 256,
        seed: 1,
        concurrent_games: 1,
        restart_processes_each_game: true,
        ponder: false,
        score_adjudication: false,
    };
    let mut g = PalsGameReceiptV5 {
        game_index: 0,
        white_endpoint: "a".into(),
        black_endpoint: "b".into(),
        base_ms: 120000,
        increment_ms: 1000,
        plies: 256,
        result: PalsResultV3::Incomplete,
        termination: PalsTerminationV3::PlyLimit,
        failed_endpoint: None,
        pgn: asset("runs/game.pgn"),
        rules_clock_pgn_check: PalsObservedV3::Unknown,
    };
    validate_game_result(&g, &pair).unwrap();
    g.result = PalsResultV3::Draw;
    assert!(validate_game_result(&g, &pair).is_err());
}

#[test]
fn receipt_lock_digest_and_deadline_failures_remain_explicit() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.lock_sha256 = "1".repeat(64);
    assert!(r.validate_against(&lock).is_err());
    r = receipt(&lock);
    r.accepted = true;
    r.endpoints.push(complete_endpoint());
    r.wall_time_ms = lock.manifest.window.wall_time_max_ms + 1;
    assert!(r.validate_against(&lock).is_err());
}
