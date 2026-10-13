//! Authored counter/file fixtures, not native NN, resource or shutdown proof.
use super::*;
use rz_contracts::{Digest, GameGeneration, ProcessEpoch, RootGeneration};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

fn authored_root(game: u64, root: u64) -> ClosedRootFacts {
    ClosedRootFacts {
        authority: SearchAuthority {
            epoch: ProcessEpoch(7),
            game: GameGeneration(game),
            root: RootGeneration(root),
            implementation: Digest([9; 32]),
        },
        close_mode: AdaptiveCloseMode::RootQuiesce,
        manager: AdaptiveManagerCounters::default(),
        native_accounting: AdaptiveNativeAccountingSnapshot::default(),
        tickets: RoleTicketCumulativeObservation::default(),
        history: CoordinatorHistoryUsage::default(),
        validator_history: ValidatorActionHistoryUsage::default(),
        cpu_limit: 100,
        cpu_spent: 0,
        cpu_reserved: 0,
        registered_validator_profile: false,
        completed_nn_inputs: Some(0),
        accepted_nn_inputs: Some(0),
        cache_evaluations_accepted: None,
    }
}

#[test]
fn exact_verified_factory_roles_observe_v_absence_without_inventing_physical_calls() {
    use std::collections::BTreeSet;

    for validator in [false, true] {
        let mut endpoint = PalsEndpointReceiptV5::with_unknown_checks(
            "authored-endpoint".into(),
            "authored-process".into(),
        );
        let mut roles = BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]);
        if validator {
            roles.insert(PalsRoleV3::Validator);
        }
        observe_verified_factory_roles(&mut endpoint, &roles, validator).unwrap();
        assert!(matches!(
            endpoint.v_loaded,
            PalsObservedV3::Observed { value, .. } if value == validator
        ));
        assert!(matches!(
            endpoint.v_generated,
            PalsObservedV3::Observed { value, .. } if value == validator
        ));
        // Authored role metadata exercises only this helper. No fixture call,
        // cache, resource or numerical result acquires runtime authority.
        assert_eq!(endpoint.v_called, PalsObservedV3::Unknown);
        assert_eq!(endpoint.coordinator, PalsObservedV3::Unknown);
        assert_eq!(endpoint.actual_tf32, PalsObservedV3::Unknown);
        assert_eq!(endpoint.resource_readback, PalsObservedV3::Unknown);
        assert_eq!(endpoint.native_numerical_check, PalsObservedV3::Unknown);
    }
}

#[test]
fn incomplete_export_or_factory_role_mismatch_preserves_unknown_observations() {
    use std::collections::BTreeSet;

    let original = PalsEndpointReceiptV5::with_unknown_checks(
        "authored-endpoint".into(),
        "authored-process".into(),
    );
    for (roles, validator_loaded) in [
        (BTreeSet::new(), false),
        (BTreeSet::from([PalsRoleV3::Proposer]), false),
        (
            BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
            true,
        ),
        (
            BTreeSet::from([
                PalsRoleV3::Proposer,
                PalsRoleV3::Critic,
                PalsRoleV3::Validator,
            ]),
            false,
        ),
    ] {
        let mut endpoint = original.clone();
        assert!(observe_verified_factory_roles(&mut endpoint, &roles, validator_loaded).is_err());
        assert_eq!(endpoint, original);
    }
}

#[test]
fn failed_finish_preserves_known_root_totals_without_inventing_a_process_fence() {
    let mut aggregate = RootAggregateV5::default();
    let mut root = authored_root(1, 1);
    root.manager.completed_nn_inputs = 3;
    root.completed_nn_inputs = Some(3);
    aggregate.add_facts(root, 7).unwrap();
    let facts = failure_observation_facts(
        &aggregate,
        None,
        &SearchSessionFailure::physical_completion_unknown(),
        None,
    );
    assert_eq!(facts["process_finish"]["status"], "unknown");
    assert_eq!(facts["retained_owners"]["status"], "unknown");
    assert_eq!(
        facts["search_failure"]["reported_search_physical_completion"],
        "unknown"
    );
    assert_eq!(facts["root_counter_aggregate"]["roots"], 1);
    assert_eq!(
        facts["root_counter_aggregate"]["manager_original_counters"]["completed_nn_inputs"],
        3
    );
    assert!(facts.get("native_physical_shutdown_confirmed").is_none());
}

#[test]
fn error_classification_and_retention_counts_never_prove_a_successful_finish() {
    let retained = AdaptiveSessionRetention {
        stage: "authored_retention",
        retained_cpu_engines: 2,
        retained_cpu_workers: 1,
        native_pending: 4,
        archive_owner_pending: true,
    };
    let failure = SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code: "AuthoredFailure",
        detail: "한".repeat(400),
    };
    let facts =
        failure_observation_facts(&RootAggregateV5::default(), None, &failure, Some(&retained));
    assert_eq!(
        facts["search_failure"]["reported_search_physical_completion"],
        "confirmed"
    );
    assert_eq!(facts["process_finish"]["status"], "unknown");
    assert_eq!(facts["retained_owners"]["status"], "observed");
    assert_eq!(facts["retained_owners"]["retained_cpu_engines"], 2);
    assert_eq!(facts["retained_owners"]["native_pending"], 4);
    assert_eq!(facts["retained_owners"]["archive_owner_pending"], true);
    assert!(facts["search_failure"]["detail"].as_str().unwrap().len() <= 1024);
    assert_eq!(facts["search_failure"]["detail_truncated"], true);
    assert!(
        facts["process_finish"]
            .get("permits_process_close")
            .is_none()
    );
}

#[test]
fn duplicate_foreign_or_out_of_order_closure_does_not_mutate_the_aggregate() {
    let mut aggregate = RootAggregateV5::default();
    aggregate.add_facts(authored_root(3, 4), 7).unwrap();
    let original = aggregate.metadata();
    for root in [
        authored_root(3, 4),
        authored_root(3, 3),
        authored_root(2, 8),
    ] {
        assert!(aggregate.add_facts(root, 7).is_err());
        assert_eq!(aggregate.metadata(), original);
    }
    let mut foreign = authored_root(3, 5);
    foreign.authority.implementation = Digest([8; 32]);
    assert!(aggregate.add_facts(foreign, 7).is_err());
    assert!(aggregate.add_facts(authored_root(3, 5), 8).is_err());
    assert_eq!(aggregate.metadata(), original);
    // A new game may restart its local root sequence, but cannot replay an old game.
    aggregate.add_facts(authored_root(4, 1), 7).unwrap();
    assert_eq!(aggregate.roots, 2);
}

#[test]
fn nn_completion_owner_acceptance_search_consumption_and_cache_remain_distinct() {
    let mut root = authored_root(1, 1);
    root.manager.completed_nn_inputs = 7;
    root.manager.accepted_nn_inputs = 5;
    root.manager.consumed_nn_inputs = 3;
    root.manager.consumed_role_outputs = 4;
    root.manager.zero_nn_outputs_consumed = 1;
    root.manager.unused_role_results = 2;
    root.completed_nn_inputs = Some(7);
    root.accepted_nn_inputs = Some(5);
    let mut aggregate = RootAggregateV5::default();
    aggregate.add_facts(root, 7).unwrap();
    assert_eq!(aggregate.manager.completed_nn_inputs, 7);
    assert_eq!(aggregate.manager.accepted_nn_inputs, 5);
    assert_eq!(aggregate.manager.consumed_nn_inputs, 3);
    assert_eq!(aggregate.manager.zero_nn_outputs_consumed, 1);
    assert_eq!(aggregate.cache_evaluations_accepted, 0);
    assert_eq!(aggregate.cache_accepted_unknown_roots, 1);
    assert!(aggregate.complete_coordinator(artifact()).is_none());
    // The authored manager totals have no matching actual native command
    // account. Neither V absence nor admission count establishes v_called.
    assert_eq!(
        aggregate.metadata()["v_physical_calls"]["status"],
        "unknown"
    );
}

#[test]
fn independent_phase_peaks_are_not_summed_and_unknown_events_survive_release() {
    let mut root = authored_root(1, 1);
    root.tickets.phase_peaks.prepared = 6;
    root.tickets.phase_peaks.submitted = 6;
    root.tickets.held_peak = 6;
    root.tickets.physical_unknown_events = 1;
    // Original fence release emptied current occupancy, not prior uncertainty.
    root.tickets.released = 1;
    let mut aggregate = RootAggregateV5::default();
    aggregate.add_facts(root, 7).unwrap();
    assert_eq!(aggregate.held_peak, 6);
    assert_eq!(aggregate.phase_peaks.prepared, 6);
    assert_eq!(aggregate.phase_peaks.submitted, 6);
    assert_eq!(aggregate.tickets.physical_unknown_events, 1);
    assert!(
        aggregate
            .require_process_finish_join(aggregate.last_authority)
            .is_err()
    );
}

#[test]
fn overflow_active_reservation_and_bound_failure_leave_original_counters_intact() {
    let mut aggregate = RootAggregateV5::default();
    aggregate.add_facts(authored_root(1, 1), 7).unwrap();
    aggregate.manager.native_preparations = u64::MAX;
    let original = aggregate.metadata();
    let mut overflow = authored_root(1, 2);
    overflow.manager.native_preparations = 1;
    assert!(aggregate.add_facts(overflow, 7).is_err());
    assert_eq!(aggregate.metadata(), original);
    let mut active = authored_root(1, 2);
    active.cpu_reserved = 1;
    assert!(aggregate.add_facts(active, 7).is_err());
    assert_eq!(aggregate.metadata(), original);
    let mut occupied = authored_root(1, 2);
    occupied.tickets.current.physically_running = 1;
    assert!(aggregate.add_facts(occupied, 7).is_err());
    assert_eq!(aggregate.metadata(), original);
    aggregate.roots = ROOT_OBSERVATIONS_MAX;
    let at_bound = aggregate.metadata();
    assert!(aggregate.add_facts(authored_root(1, 2), 7).is_err());
    assert_eq!(aggregate.metadata(), at_bound);
}

#[test]
fn last_process_authority_must_join_the_latest_observed_root() {
    let mut aggregate = RootAggregateV5::default();
    aggregate.add_facts(authored_root(1, 1), 7).unwrap();
    assert!(aggregate.require_process_finish_join(None).is_err());
    let foreign = authored_root(1, 2).authority;
    assert!(
        aggregate
            .require_process_finish_join(Some(foreign))
            .is_err()
    );
    aggregate
        .require_process_finish_join(aggregate.last_authority)
        .unwrap();
}

fn artifact() -> ArtifactRef {
    ArtifactRef {
        path: "runtime/authored-proof.json".into(),
        sha256: "0".repeat(64),
        bytes: 1,
        source: "https://example.org/authored-counter-file-fixture".into(),
        license: "MIT".into(),
    }
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rz-v5-observation-fixture-{}-{id}-{time}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("runtime")).unwrap();
        Self { root }
    }
    fn writer(&self) -> BoundedObservationWriterV5 {
        let root = Dir::open_ambient_dir(&self.root, cap_std::ambient_authority()).unwrap();
        let owner = root.open_dir_nofollow("runtime").unwrap();
        BoundedObservationWriterV5 {
            root,
            owner,
            logical_prefix: "runtime".into(),
            written_files: 0,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // This unique fixture is never an engine/model/cache directory.
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn metadata_is_exclusive_bounded_and_returns_the_actual_byte_digest() {
    let fixture = Fixture::new();
    let mut writer = fixture.writer();
    assert!(
        writer
            .write_bytes(OutputKind::Startup, &vec![0; METADATA_BYTES_MAX + 1])
            .is_err()
    );
    assert!(
        !fixture
            .root
            .join("runtime")
            .join(OutputKind::Startup.name())
            .exists()
    );
    let bytes = b"{\"authored_fixture\":true}";
    let artifact = writer.write_bytes(OutputKind::Startup, bytes).unwrap();
    assert_eq!(artifact.sha256, format!("{:x}", Sha256::digest(bytes)));
    assert_eq!(artifact.bytes, bytes.len() as u64);
    assert_eq!(artifact.path, "runtime/pals-native-startup.v5.json");
    assert!(
        writer
            .write_bytes(OutputKind::Startup, b"overwrite")
            .is_err()
    );
    assert_eq!(fs::read(fixture.root.join(&artifact.path)).unwrap(), bytes);
}

#[test]
fn forensic_output_is_exclusive_and_does_not_create_a_normal_termination() {
    let fixture = Fixture::new();
    let mut writer = fixture.writer();
    let facts = failure_observation_facts(
        &RootAggregateV5::default(),
        None,
        &SearchSessionFailure::physical_completion_unknown(),
        None,
    );
    let proof = writer
        .write_json(OutputKind::FailedTermination, &facts)
        .unwrap();
    let bytes = fs::read(fixture.root.join(&proof.path)).unwrap();
    assert_eq!(proof.sha256, format!("{:x}", Sha256::digest(&bytes)));
    assert_eq!(proof.bytes, bytes.len() as u64);
    assert!(
        writer
            .write_json(OutputKind::FailedTermination, &facts)
            .is_err()
    );
    assert!(
        !fixture
            .root
            .join("runtime")
            .join(OutputKind::Termination.name())
            .exists()
    );
}

#[test]
fn owned_root_quota_and_overflow_are_rejected_without_giant_file_fixtures() {
    require_owned_write_budget(OWNED_ROOT_BYTES_MAX - 2, 2).unwrap();
    assert!(require_owned_write_budget(OWNED_ROOT_BYTES_MAX, 2).is_err());
    assert!(require_owned_write_budget(OWNED_ROOT_BYTES_MAX - 1, 2).is_err());
    assert!(require_owned_write_budget(u64::MAX, 2).is_err());
}

#[test]
fn output_file_bound_prevents_new_metadata_creation() {
    let fixture = Fixture::new();
    let mut writer = fixture.writer();
    writer.written_files = OUTPUT_FILES_MAX;
    assert!(writer.write_bytes(OutputKind::Startup, b"{}").is_err());
    assert!(
        !fixture
            .root
            .join("runtime")
            .join(OutputKind::Startup.name())
            .exists()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn owned_length_scan_rejects_links_and_hard_link_aliases() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("authored.bin"), b"authored fixture").unwrap();
    std::os::unix::fs::symlink("authored.bin", fixture.root.join("alias.bin")).unwrap();
    assert!(owned_root_bytes(&fixture.writer().root).is_err());
    fs::remove_file(fixture.root.join("alias.bin")).unwrap();
    fs::hard_link(
        fixture.root.join("authored.bin"),
        fixture.root.join("alias.bin"),
    )
    .unwrap();
    assert!(owned_root_bytes(&fixture.writer().root).is_err());
}
