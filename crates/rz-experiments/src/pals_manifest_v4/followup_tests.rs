fn owner_methods(p: &PalsPoliciesV4) -> PalsFollowupOwnerMethodsV4 {
    let method = |name: &str| {
        Some(PalsOwnerMethodIdentityV4 {
            method: name.into(),
            implementation_sha256: "b".repeat(64),
        })
    };
    PalsFollowupOwnerMethodsV4 {
        producer_domain: PALS_FOLLOWUP_LIFECYCLE_V4_DOMAIN.into(),
        producer_schema_version: 1,
        producer_implementation_sha256: "c".repeat(64),
        trace_capacity: 64,
        trace_bytes_max: 96 * 1024,
        repair: if p.recheck == PalsRecheckPolicyV4::Disabled {
            None
        } else {
            method("engine_actual_followup_search_v4")
        },
        paused_stack: if matches!(p.paused_stack, PalsPausedStackPolicyV4::Disabled) {
            None
        } else {
            method("cpu_actual_paused_frame_owner_v4")
        },
        cold_archive: if matches!(p.cold_archive, PalsColdArchivePolicyV4::Disabled) {
            None
        } else {
            method("archive_actual_owner_events_v4")
        },
        cuda_warm: if matches!(p.cuda_warm, PalsCudaWarmPolicyV4::Disabled) {
            None
        } else {
            method("cuda_actual_backend_and_bank_events_v4")
        },
    }
}
fn lifecycle(p: &PalsPoliciesV4) -> PalsFollowupLifecycleScopeV4 {
    PalsFollowupLifecycleScopeV4 {
        domain: PALS_FOLLOWUP_LIFECYCLE_V4_DOMAIN.into(),
        schema_version: 1,
        process_epoch: Some(7),
        game_generation: 1,
        capture_complete: true,
        owner_shutdown_complete: true,
        trace_capacity: 64,
        trace_bytes_max: 96 * 1024,
        owner_methods: owner_methods(p),
        capture_failure: None,
        partial_facts: None,
        previous_closed_archive_owners: vec![],
    }
}
fn archive_fixture() -> (PalsColdArchiveLimitsV4, PalsArchiveObservationV4) {
    let limits = PalsColdArchiveLimitsV4 {
        game_bytes_max: 4096,
        global_bytes_max: 8192,
        index_entries_max: 16,
        index_bytes_max: 1024,
        load_bytes_max: 1024,
        load_deadline_max_ms: 100,
    };
    let a = PalsArchiveObservationV4 {
        owner_id: 1,
        generation: 1,
        lifecycle: PalsArchiveLifecycleV4::Closed,
        committed_chunks: 1,
        integrity_verified_chunks: 1,
        committed_bytes: 64,
        integrity_verified_bytes: 64,
        ram_released_bytes: 128,
        pending_commit_bytes: 0,
        pending_buffers_retained: false,
        global_managed_bytes: 64,
        index_entries_peak: 0,
        index_bytes_peak: 0,
        pinned_entries_peak: 1,
        loads_requested: 1,
        loads_completed: 1,
        load_bytes_total: 64,
        load_bytes_peak: 64,
        load_elapsed_peak_ms: 1,
        owner_generation_checks: 1,
        quota_failures: 0,
        io_failures: 0,
        pin_saturation_failures: 0,
        cleanup_complete: true,
        actual_scope: Some(PalsArchiveActualScopeV4 {
            measurement_contract: "rz-pals-cold-archive-owner-accounting-v4/1".into(),
            event_sequence: 4,
            root_sha256: "a".repeat(64),
            repository_sha256: "b".repeat(64),
            runtime_limits: PalsArchiveRuntimeLimitsV4 {
                game_bytes_max: limits.game_bytes_max,
                global_bytes_max: limits.global_bytes_max,
                index_entries_max: limits.index_entries_max,
                index_bytes_max: limits.index_bytes_max,
                load_bytes_max: limits.load_bytes_max,
                load_deadline_max_ms: limits.load_deadline_max_ms,
                record_payload_bytes_max: 4096,
                max_load_pins: 16,
            },
            generation_kind: "reserved_generation_counter".into(),
            reserved_generations: 1,
            last_committed_generation: Some(0),
            last_verified_generation: Some(0),
            ram_release_method: "hot_unique_owned_capacity_subset".into(),
            ram_reclaim_events: 1,
            integrity_before_reclaim_events: 1,
            unverified_reclaim_events: 0,
            pending_files_retained: false,
            cold_index_kind: "directory_scan".into(),
            load_pins_max: 16,
            loaded_closure_bytes_peak: Some(128),
            loaded_closure_bytes_method: Some("hot_unique_owned_capacity_subset".into()),
            owner_generation_check_failures: 0,
            archive_write_bytes_total: Some(64),
            session_write_ledger_id: Some(17),
            session_write_bytes_max: Some(limits.game_bytes_max),
            session_write_bytes_consumed: Some(64),
            global_scan_event_sequence: Some(4),
            global_scan_complete: true,
            global_scan_after_last_commit: true,
            global_scope_kind: "canonical_no_link_managed_root".into(),
            admission_closed: true,
            complete: true,
        }),
    };
    (limits, a)
}
fn physical_comparison(p: &PalsEndpointV4, start: u64) -> PalsRecheckObservationV4 {
    let mut a = PalsRecheckObservationV4 {
        before: frozen_value([0.6, 0.3, 0.1]),
        after: frozen_value([0.1, 0.2, 0.7]),
        resolution: PalsRecheckResolutionV4::BeforePreferred,
        fresh_nn_inputs_charged: 3,
        actual_scope: None,
    };
    let provenance = |sequence: u64, cost: u64, wdl: [f32; 3], original_perspective| {
        PalsRecheckModelProvenanceV4 {
        model_value_semantics:"rz-pals-context-wdl/1;side-to-move;restricted-model-estimate;actual-prepared-input;no-cp-calibration;no-rules-proof".into(),
        model_identity:p.model_v2.model_identity.clone(),encoding_identity:p.model_v2.encoding_sha256.clone(),
        model_epoch_sha256:"a".repeat(64),original_perspective,wdl_bits:wdl.map(f32::to_bits),state_sha256:"b".repeat(64),
        input_sha256:"d".repeat(64),context_revision:1,prepared_state:PalsPreparedStateV4 {owner:7,revision:sequence,semantic_sha256:"d".repeat(64)},
        fresh_call_attempted:true,native_request:Some(PalsPhysicalIdV4 {epoch:7,sequence}),
        native_execution:Some(PalsPhysicalIdV4 {epoch:7,sequence}),completed_nn_inputs:Some(cost),
        physical_input_completed:true,accepted_output:true,
    }
    };
    // Identical input digests are legal for two physically distinct Fresh calls.
    a.actual_scope = Some(PalsRecheckActualScopeV4 {
        comparison_perspective: PalsPerspectiveV4::White,
        before: Some(provenance(
            start,
            2,
            [0.1, 0.3, 0.6],
            PalsPerspectiveV4::Black,
        )),
        after: Some(provenance(
            start + 1,
            1,
            [0.1, 0.2, 0.7],
            PalsPerspectiveV4::White,
        )),
        fresh_requests_completed: 2,
        comparison_attempted: true,
        publication: None,
        original_error: None,
    });
    a
}

#[test]
fn default_none_trace_retains_exact_historical_wire_bytes() {
    let expected = concat!(
        "{\"root_generation\":1,\"first_move\":\"e2e4\",\"original_record\":10,\"record_id\":11,\"supersedes\":10,",
        "\"question_sha256\":\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\",\"evidence_revision\":1,\"repair_ordinal\":1,",
        "\"evidence_sha256\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",\"recheck\":{\"status\":\"observed\",\"value\":{",
        "\"before\":{\"kind\":\"frozen_wdl\",\"model_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"model_epoch\":1,",
        "\"encoding_sha256\":\"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\",\"precision\":\"fp32\",\"context_revision\":1,",
        "\"input_sha256\":\"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\",\"perspective\":\"white\",\"fresh\":true,\"wdl\":[0.6,0.3,0.1]},",
        "\"after\":{\"kind\":\"frozen_wdl\",\"model_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"model_epoch\":1,",
        "\"encoding_sha256\":\"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\",\"precision\":\"fp32\",\"context_revision\":1,",
        "\"input_sha256\":\"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\",\"perspective\":\"white\",\"fresh\":true,\"wdl\":[0.1,0.2,0.7]},",
        "\"resolution\":\"before_preferred\",\"fresh_nn_inputs_charged\":2},\"method\":\"independent-fixture-check\"}}"
    );
    assert_eq!(serde_json::to_string(&trace()).unwrap(), expected);
    assert_eq!(
        serde_json::from_str::<PalsRepairTraceV4>(expected).unwrap(),
        trace()
    );
}

#[test]
fn actual_archive_zero_first_generation_and_capacity_units_are_preserved() {
    let (limits, mut a) = archive_fixture();
    a.validate_against(&limits, true, &BTreeSet::new()).unwrap();
    assert!(a.ram_released_bytes > a.integrity_verified_bytes);
    assert!(a.pinned_entries_peak > a.index_entries_peak);
    a.actual_scope
        .as_mut()
        .unwrap()
        .integrity_before_reclaim_events = 0;
    assert!(
        a.validate_against(&limits, false, &BTreeSet::new())
            .is_err()
    );
    a.actual_scope
        .as_mut()
        .unwrap()
        .integrity_before_reclaim_events = 1;
    a.actual_scope.as_mut().unwrap().last_verified_generation = Some(1);
    assert!(
        a.validate_against(&limits, false, &BTreeSet::new())
            .is_err()
    );
    a.actual_scope.as_mut().unwrap().last_verified_generation = Some(0);
    a.pinned_entries_peak = 17;
    assert!(
        a.validate_against(&limits, false, &BTreeSet::new())
            .is_err()
    );
}

#[test]
fn actual_archive_retained_transaction_file_is_not_an_invented_ram_buffer() {
    let (limits, mut a) = archive_fixture();
    a.pending_commit_bytes = 8;
    a.io_failures = 1;
    a.cleanup_complete = false;
    a.lifecycle = PalsArchiveLifecycleV4::Failed;
    let s = a.actual_scope.as_mut().unwrap();
    s.complete = false;
    s.admission_closed = false;
    s.pending_files_retained = true;
    s.archive_write_bytes_total = Some(72);
    s.session_write_bytes_consumed = Some(72);
    s.global_scan_after_last_commit = false;
    let failures = BTreeSet::from([PalsPolicyFailureV4::ArchiveIo]);
    a.validate_against(&limits, false, &failures).unwrap();
    assert!(!a.pending_buffers_retained);
    assert!(a.validate_against(&limits, true, &failures).is_err());
    a.actual_scope.as_mut().unwrap().pending_files_retained = false;
    assert!(a.validate_against(&limits, false, &failures).is_err());
}

#[test]
fn actual_archive_owner_costs_are_counted_once_and_unknown_remains_unknown() {
    let m = manifest();
    let mut o = endpoint_receipt(&m.engines[0], &m.resources[0], 0);
    let (limits, mut current) = archive_fixture();
    current.owner_id = 2;
    current
        .actual_scope
        .as_mut()
        .unwrap()
        .session_write_bytes_consumed = Some(128);
    let mut p = policies();
    p.cold_archive = PalsColdArchivePolicyV4::Bounded(limits);
    let mut s = lifecycle(&p);
    s.previous_closed_archive_owners.push(archive_fixture().1);
    o.followup_lifecycle = Some(s);
    o.archive = observed_fixture(current);
    assert_eq!(
        followup::archive_output_accounting(&o, true).unwrap(),
        (128, true)
    );
    o.followup_lifecycle
        .as_mut()
        .unwrap()
        .previous_closed_archive_owners[0]
        .owner_id = 2;
    assert!(followup::archive_output_accounting(&o, true).is_err());
    o.followup_lifecycle
        .as_mut()
        .unwrap()
        .previous_closed_archive_owners[0]
        .owner_id = 1;
    o.archive = PalsObservedV3::Unknown;
    o.followup_lifecycle.as_mut().unwrap().partial_facts = Some(PalsFollowupPartialFactsV4 {
        archive: Some(PalsArchivePartialFactsV4 {
            owner_id: Some(2),
            archive_write_bytes_total: None,
            complete: false,
            ..Default::default()
        }),
        ..Default::default()
    });
    assert_eq!(
        followup::archive_output_accounting(&o, true).unwrap(),
        (64, false)
    );
}

#[test]
fn actual_session_write_ledger_cannot_reset_with_a_new_game_owner() {
    let m = manifest();
    let mut o = endpoint_receipt(&m.engines[0], &m.resources[0], 0);
    let (limits, mut current) = archive_fixture();
    current.owner_id = 2;
    current
        .actual_scope
        .as_mut()
        .unwrap()
        .session_write_bytes_consumed = Some(128);
    let mut p = policies();
    p.cold_archive = PalsColdArchivePolicyV4::Bounded(limits.clone());
    let mut scope = lifecycle(&p);
    scope
        .previous_closed_archive_owners
        .push(archive_fixture().1);
    o.followup_lifecycle = Some(scope);
    o.archive = observed_fixture(current.clone());
    assert_eq!(
        followup::archive_output_accounting(&o, true).unwrap(),
        (128, true)
    );
    for (id, consumed) in [(18, 128), (17, 64)] {
        let mut changed = current.clone();
        let actual = changed.actual_scope.as_mut().unwrap();
        actual.session_write_ledger_id = Some(id);
        actual.session_write_bytes_consumed = Some(consumed);
        o.archive = observed_fixture(changed);
        assert!(followup::archive_output_accounting(&o, true).is_err());
    }
    current
        .actual_scope
        .as_mut()
        .unwrap()
        .session_write_ledger_id = None;
    o.archive = observed_fixture(current.clone());
    assert_eq!(
        followup::archive_output_accounting(&o, true).unwrap(),
        (128, false)
    );
    assert!(
        current
            .validate_against(&limits, true, &BTreeSet::new())
            .is_err()
    );

    // Known zero is retained; a nullable unknown is never synthesized as zero.
    let (_, mut zero) = archive_fixture();
    zero.owner_id = 3;
    zero.committed_chunks = 0;
    zero.integrity_verified_chunks = 0;
    zero.committed_bytes = 0;
    zero.integrity_verified_bytes = 0;
    zero.ram_released_bytes = 0;
    zero.global_managed_bytes = 0;
    let actual = zero.actual_scope.as_mut().unwrap();
    actual.last_committed_generation = None;
    actual.last_verified_generation = None;
    actual.ram_reclaim_events = 0;
    actual.integrity_before_reclaim_events = 0;
    zero.actual_scope
        .as_mut()
        .unwrap()
        .archive_write_bytes_total = Some(0);
    zero.actual_scope
        .as_mut()
        .unwrap()
        .session_write_bytes_consumed = Some(0);
    o.followup_lifecycle
        .as_mut()
        .unwrap()
        .previous_closed_archive_owners
        .clear();
    o.archive = observed_fixture(zero.clone());
    assert_eq!(
        followup::archive_output_accounting(&o, true).unwrap(),
        (0, true)
    );
    zero.actual_scope
        .as_mut()
        .unwrap()
        .archive_write_bytes_total = None;
    o.archive = observed_fixture(zero);
    assert_eq!(
        followup::archive_output_accounting(&o, true).unwrap(),
        (0, false)
    );

    let mut raw = serde_json::to_value(current).unwrap();
    raw["actual_scope"]
        .as_object_mut()
        .unwrap()
        .remove("session_write_bytes_consumed");
    assert!(serde_json::from_value::<PalsArchiveObservationV4>(raw).is_err());
}

#[test]
fn actual_fresh_two_requests_can_charge_three_graph_inputs_without_reinterpretation() {
    let mut m = manifest();
    let p = pals_mut(&mut m);
    let mut c = physical_comparison(p, 1);
    c.validate_against(p).unwrap();
    c.fresh_nn_inputs_charged = 2;
    assert!(c.validate_against(p).is_err());
    c.fresh_nn_inputs_charged = 3;
    c.actual_scope
        .as_mut()
        .unwrap()
        .after
        .as_mut()
        .unwrap()
        .native_execution
        .as_mut()
        .unwrap()
        .sequence = 1;
    assert!(c.validate_against(p).is_err());
    c.actual_scope
        .as_mut()
        .unwrap()
        .after
        .as_mut()
        .unwrap()
        .native_execution
        .as_mut()
        .unwrap()
        .sequence = 2;
    c.actual_scope
        .as_mut()
        .unwrap()
        .after
        .as_mut()
        .unwrap()
        .accepted_output = false;
    assert!(c.validate_against(p).is_err());
}

#[test]
fn actual_comparison_requires_present_nullable_fields_and_actual_attempt_authority() {
    let mut m = manifest();
    let p = pals_mut(&mut m);
    let c = physical_comparison(p, 1);
    let mut raw = serde_json::to_value(&c).unwrap();
    raw["actual_scope"]
        .as_object_mut()
        .unwrap()
        .remove("publication");
    assert!(serde_json::from_value::<PalsRecheckObservationV4>(raw).is_err());
    let mut c = c;
    c.actual_scope.as_mut().unwrap().comparison_attempted = false;
    assert!(c.validate_against(p).is_err());
    let mut c = physical_comparison(p, 1);
    c.actual_scope.as_mut().unwrap().original_error = Some("actual evaluator failure".into());
    assert!(c.validate_against(p).is_err());
}

#[test]
fn actual_entered_search_scope_preserves_prior_go_supersedes_without_synthetic_generation() {
    let mut m = repair_manifest();
    let p = pals_mut(&mut m).clone();
    let mut o = endpoint_receipt(&m.engines[0], &m.resources[0], 0);
    o.base.nn_inputs_completed = 6;
    o.followup_lifecycle = Some(lifecycle(&p.policies));
    o.repair_trace_total = observed_fixture(2);
    o.pending_questions_peak = observed_fixture(1);
    o.repair_admissions_peak = Some(observed_fixture(3));
    for attempt in [1, 2] {
        let mut t = trace();
        t.root_generation = 0;
        t.record_id = 10 + attempt;
        t.supersedes = if attempt == 1 { None } else { Some(11) };
        t.evidence_revision = attempt;
        t.recheck = observed_fixture(physical_comparison(&p, attempt * 2 - 1));
        t.actual_scope = Some(PalsRepairActualScopeV4 {
            game_generation: 1,
            search_attempt_sequence: attempt,
            store_root_generation: 0,
            lineage_root_record: t.record_id,
            parent_revision: Some(10),
            supersedes_revision: t.supersedes,
        });
        o.repair_traces.push(t);
    }
    followup::validate_actual_traces(&o, &p, true, &BTreeSet::new()).unwrap();
    o.repair_traces[1]
        .actual_scope
        .as_mut()
        .unwrap()
        .search_attempt_sequence = 1;
    assert!(followup::validate_actual_traces(&o, &p, true, &BTreeSet::new()).is_err());
    o.repair_traces[1]
        .actual_scope
        .as_mut()
        .unwrap()
        .search_attempt_sequence = 2;
    o.repair_admissions_peak = Some(observed_fixture(4));
    assert!(followup::validate_actual_traces(&o, &p, true, &BTreeSet::new()).is_err());
}

#[test]
fn actual_incomplete_overflow_capture_can_only_remain_ineligible() {
    let m = repair_manifest();
    let PalsEngineV4::Pals(p) = &m.engines[0] else {
        unreachable!()
    };
    let mut o = endpoint_receipt(&m.engines[0], &m.resources[0], 0);
    let mut s = lifecycle(&p.policies);
    s.capture_complete = false;
    s.owner_shutdown_complete = false;
    s.capture_failure = Some(PalsCaptureFailureV4::CounterOverflow);
    s.partial_facts = Some(PalsFollowupPartialFactsV4 {
        repair: Some(PalsRepairPartialFactsV4 {
            complete: false,
            ..Default::default()
        }),
        ..Default::default()
    });
    o.followup_lifecycle = Some(s);
    o.repair_admissions_peak = Some(PalsObservedV3::Unknown);
    followup::validate_actual_traces(&o, p, false, &BTreeSet::new()).unwrap();
    assert!(followup::validate_actual_traces(&o, p, true, &BTreeSet::new()).is_err());
    let s = o.followup_lifecycle.as_mut().unwrap();
    s.capture_complete = true;
    s.owner_shutdown_complete = true;
    s.capture_failure = None;
    assert!(s.validate_for(&p.policies, true).is_err());
}

#[test]
fn actual_warm_seed_consumption_and_fresh_value_are_independent_fenced_events() {
    let m = manifest();
    let mut base = endpoint_receipt(&m.engines[0], &m.resources[0], 0).base;
    base.nn_inputs_completed = 8;
    let limits = PalsCudaWarmLimitsV4 {
        capability: PALS_V4_CUDA_WARM_CAPABILITY.into(),
        max_leases: 1,
        device_bytes_max: 64,
    };
    let mut a = PalsCudaWarmObservationV4 {
        capability: observed_fixture(PalsCudaWarmCapabilityV4 {
            semantic_id: PALS_V4_CUDA_WARM_CAPABILITY.into(),
            device_identity: "fixture-device".into(),
            available: true,
        }),
        leases_admitted: 8,
        leases_physically_completed: 8,
        leases_quarantined: 0,
        leases_peak: 1,
        device_bytes_peak: 32,
        accepted_seed_consumptions: 5,
        seed_context_checked_consumptions: 5,
        rejected_seed_contexts: 0,
        fresh_value_evaluations: 1,
        buffers_released: true,
        actual_scope: Some(PalsCudaWarmActualScopeV4 {
            measurement_contract: "rz-pals-cuda-warm-owner-accounting-v4/1".into(),
            backend_owner_id: 1,
            bank_owner_id: 2,
            physical_event_sequence: 16,
            seed_event_sequence: 5,
            leases_active: 0,
            device_bytes_current: 0,
            device_bytes_retained: 0,
            device_bytes_scope: "explicit_cuda_kv_payload_bytes".into(),
            lease_accounting_scope: "backend_owner_lifetime_including_startup".into(),
            seed_accounting_scope: "search_only".into(),
            value_accounting_scope: "search_only".into(),
            rejection_scope: "selected_payload_seal_and_bind_validation".into(),
            startup_leases_admitted: 2,
            startup_leases_completed: 2,
            startup_leases_quarantined: 0,
            value_always_fresh: true,
            admission_closed: true,
            bank_closed: true,
            backend_dropped: true,
            worker_joined: true,
            complete: true,
        }),
    };
    a.validate_against(&limits, &base, true, &BTreeSet::new())
        .unwrap();
    a.actual_scope.as_mut().unwrap().worker_joined = false;
    assert!(
        a.validate_against(&limits, &base, false, &BTreeSet::new())
            .is_err()
    );
    a.actual_scope.as_mut().unwrap().worker_joined = true;
    a.actual_scope.as_mut().unwrap().device_bytes_retained = 1;
    assert!(
        a.validate_against(&limits, &base, false, &BTreeSet::new())
            .is_err()
    );
}

#[test]
fn actual_paused_frame_owner_requires_closed_admission_and_retained_zero() {
    let limits = PalsPausedStackLimitsV4 {
        tokens_max: 1,
        bytes_max: 1024,
    };
    let mut a = PalsPausedStackObservationV4 {
        tokens_created: 1,
        tokens_resumed: 1,
        tokens_invalidated: 0,
        tokens_retained: 0,
        tokens_peak: 1,
        bytes_peak: 64,
        stale_context_attempts: 1,
        stale_context_rejections: 1,
        replayed_consumed_work: 0,
        owner_released: true,
        actual_scope: Some(PalsPausedStackActualScopeV4 {
            owner_id: 1,
            event_sequence: 3,
            bytes_current: 0,
            complete: true,
            admission_closed: true,
        }),
    };
    a.validate_against(&limits, true, &BTreeSet::new()).unwrap();
    a.actual_scope.as_mut().unwrap().admission_closed = false;
    assert!(
        a.validate_against(&limits, false, &BTreeSet::new())
            .is_err()
    );
    a.actual_scope.as_mut().unwrap().admission_closed = true;
    a.actual_scope.as_mut().unwrap().bytes_current = 1;
    assert!(
        a.validate_against(&limits, false, &BTreeSet::new())
            .is_err()
    );
}
