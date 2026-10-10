use super::*;
use std::time::Duration;

struct TestRoot(PathBuf);
impl TestRoot {
    fn new() -> Self {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("rz-pals-store-tests-{}-{n}", std::process::id()));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn config(&self) -> ArchiveConfig {
        ArchiveConfig::new(self.0.join("archive"), std::env::current_dir().unwrap())
    }
}
impl Drop for TestRoot {
    fn drop(&mut self) {
        // This unique test-owned root is outside the checkout. Refuse links or
        // paths outside the exact temporary parent before recursive cleanup.
        if self.0.parent() == Some(std::env::temp_dir().as_path())
            && self
                .0
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("rz-pals-store-tests-")
            && no_link(&self.0).is_ok()
        {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}
fn limits() -> StoreLimits {
    StoreLimits {
        states: 4,
        situations: 4,
        line_chunks: 16,
        observations: 8,
        dependency_edges: 16,
        executions: 4,
        consumers: 8,
        conclusions_per_situation: 8,
        root_moves_per_task: 16,
        retained_history_bytes: 64 * 1024,
        ..StoreLimits::default()
    }
}
fn budget() -> ArchiveIoBudget {
    ArchiveIoBudget {
        deadline: Instant::now() + Duration::from_secs(10),
        max_bytes: 8 * 1024 * 1024,
    }
}
fn setup(root: &TestRoot) -> PalsStores {
    let mut store = PalsStores::new(limits());
    store.enable_archive(root.config()).unwrap();
    store
}
fn mv(text: &str) -> BoardMove {
    text.parse().unwrap()
}
fn identity() -> CpuValueIdentity {
    CpuValueIdentity {
        semantics: crate::cpu::BOOTSTRAP_SCORE_VERSION.into(),
        weights_sha256: None,
        training: crate::cpu_value::CpuTrainingState::Bootstrap,
    }
}
fn observation(state: StateId) -> Observation {
    Observation {
        state,
        line: None,
        source: 1,
        epoch: 1,
        value_identity: Some(identity()),
        checker_identity: None,
        checker_work: None,
        external_report: None,
        model_value_identity: None,
        model_value_input: None,
        cpu_condition: Some("archive-fixture-cpu-conditions-v1".into()),
        cpu_pv: None,
        scope: EvidenceScope::DepthLimited {
            depth: 4,
            profile: 1,
            condition: 1,
        },
        score: RawScore::Cpu {
            value: 123,
            perspective: Color::White,
            bound: BoundKind::ExactWithinSearch,
        },
        budget: 128,
        kind: ObservationKind::CpuAnalysis,
        supersedes: None,
        execution: None,
    }
}
fn key(state: StateId) -> TaskKey {
    TaskKey {
        state,
        line: None,
        question: TaskQuestion::AnalyzePosition,
        root_moves: vec![],
        model: 1,
        epoch: 1,
        value_identity: Some(identity()),
        checker_identity: None,
        cpu_condition: Some("archive-fixture-cpu-conditions-v1".into()),
        profile: 1,
        condition: 1,
        input_revision: 0,
        requested_depth: 4,
        node_budget: 128,
    }
}

#[test]
fn multi_move_small_store_preserves_exact_history_lines_and_evidence() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let original = position.snapshot();
    let old_situation = store.focus_actual_moves(original.clone()).unwrap();
    let old_state = store.situations.get(old_situation).unwrap().state;
    let old_line = store
        .append_cpu_pv(&position, &[mv("e2e4"), mv("e7e5"), mv("g1f3")])
        .unwrap();
    let mut raw = observation(old_state);
    raw.cpu_pv = Some(old_line);
    let old_observation = store.append_observation(raw.clone()).unwrap();
    store
        .dependencies
        .add(old_observation, old_situation)
        .unwrap();
    position.make_move(mv("g1f3")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert_eq!(
        store.states.get(old_state).unwrap_err(),
        StoreError::ColdRecord("state")
    );
    let replacement = store.root().unwrap();
    assert!(store.situations.get(old_situation).is_err());
    let old_revision = store.situations.get(replacement).unwrap().revision;
    for movement in [
        "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "a7a6", "e4e5", "d7d5",
        "e5d6",
    ] {
        position.make_move(mv(movement)).unwrap();
        store.focus_actual_moves(position.snapshot()).unwrap();
        store.archive_inactive(budget()).unwrap();
        assert!(store.states.len() <= limits().states);
        assert!(store.lines.len() <= limits().line_chunks);
        assert!(store.situations.len() <= limits().situations);
        assert_eq!(store.states.snapshots.index.len(), store.states.len());
    }
    let handle = store
        .lookup_cold_state(&original, budget())
        .unwrap()
        .unwrap();
    assert_eq!(handle.original_id(), old_state);
    let loaded = store.pin_load(&handle, budget()).unwrap();
    assert!(store.states.get(old_state).unwrap().same_state(&original));
    assert_eq!(
        store.states.get(old_state).unwrap().known_history_fens(),
        original.known_history_fens()
    );
    let evidence = store
        .pin_load(&receipt.observation_handle(old_observation), budget())
        .unwrap();
    assert_eq!(*store.observations.get(old_observation).unwrap(), raw);
    assert_eq!(
        store.lines.moves(old_line).unwrap(),
        vec![mv("e2e4"), mv("e7e5"), mv("g1f3")]
    );
    assert_eq!(
        store
            .dependencies
            .invalidate(old_observation, &mut store.situations)
            .unwrap(),
        0
    );
    if let Ok(replacement) = store.situations.get(replacement) {
        assert_eq!(replacement.revision, old_revision);
    }
    store.release_loaded(evidence.pin).unwrap();
    store.release_loaded(loaded.pin).unwrap();
}

#[test]
fn roots_frontiers_paused_tasks_and_dependency_closure_are_pinned() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let root_id = store.focus_actual_moves(position.snapshot()).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let paused = store.insert_situation(position.snapshot()).unwrap();
    let paused_state = store.situations.get(paused).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            key(paused_state),
            TaskConsumer {
                id: 7,
                situation: paused,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("new task")
    };
    store.pause_task(execution, 19, None).unwrap();
    position.make_move(mv("e7e5")).unwrap();
    let frontier = store.insert_situation(position.snapshot()).unwrap();
    let frontier_state = store.situations.get(frontier).unwrap().state;
    let evidence = store
        .append_observation(observation(frontier_state))
        .unwrap();
    store.dependencies.add(evidence, frontier).unwrap();
    position.make_move(mv("g1f3")).unwrap();
    let inactive = store.insert_situation(position.snapshot()).unwrap();
    let inactive_state = store.situations.get(inactive).unwrap().state;
    let mut pins = StorePins::default();
    pins.situations.insert(frontier);
    store.set_archive_pins(pins).unwrap();
    store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.situations.get(root_id).is_ok());
    assert!(store.situations.get(paused).is_ok());
    assert!(store.situations.get(frontier).is_ok());
    assert!(store.observations.get(evidence).is_ok());
    assert!(store.states.get(inactive_state).is_err());
    assert!(matches!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::Paused { checkpoint: 19, .. }
    ));
    assert_eq!(
        store.dependencies.affected(evidence).collect::<Vec<_>>(),
        vec![frontier]
    );
    assert!(matches!(
        store.archive_inactive(budget()),
        Err(StoreError::PinSaturated(_))
    ));
}

#[test]
fn byte_deadline_checksum_and_owner_failures_do_not_publish_hot_records() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    let handle = receipt.state_handle(state);
    let before = store.hot_stats();
    assert_eq!(
        store
            .pin_load(
                &handle,
                ArchiveIoBudget {
                    deadline: Instant::now(),
                    max_bytes: u64::MAX
                }
            )
            .unwrap_err(),
        StoreError::ArchiveDeadline
    );
    assert_eq!(
        store
            .pin_load(
                &handle,
                ArchiveIoBudget {
                    max_bytes: 4,
                    ..budget()
                }
            )
            .unwrap_err(),
        StoreError::ArchiveByteBudget
    );
    let other_root = TestRoot::new();
    let mut other = setup(&other_root);
    assert!(matches!(
        other.pin_load(&handle, budget()),
        Err(StoreError::InvalidHandle(_))
    ));
    assert_eq!(store.hot_stats(), before);
    let file = store.archive.as_ref().unwrap().path(receipt.generation);
    use std::io::{Seek, SeekFrom};
    let mut output = OpenOptions::new().write(true).open(file).unwrap();
    output.seek(SeekFrom::End(-1)).unwrap();
    output.write_all(b"!").unwrap();
    output.sync_all().unwrap();
    assert_eq!(
        store.pin_load(&handle, budget()).unwrap_err(),
        StoreError::ArchiveIntegrity("archive checksum")
    );
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn quota_and_io_failure_keep_the_original_ram_and_allow_explicit_recovery() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    store.focus_actual_moves(position.snapshot()).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let before = store.hot_stats();
    store.archive.as_mut().unwrap().config.game_bytes = 80;
    assert_eq!(
        store.archive_inactive(budget()).unwrap_err(),
        StoreError::ArchiveQuota("game archive")
    );
    assert_eq!(store.hot_stats(), before);
    store.archive.as_mut().unwrap().config.game_bytes = GAME_ARCHIVE_BYTES;
    let directory = store.archive.as_ref().unwrap().directory.clone();
    fs::remove_dir(&directory).unwrap();
    assert!(matches!(
        store.archive_inactive(budget()),
        Err(StoreError::ArchiveIo { .. })
    ));
    assert_eq!(store.hot_stats(), before);
    fs::create_dir(&directory).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(receipt.committed_bytes > 0);
    assert!(receipt.after.states < receipt.before.states);
    position.make_move(mv("e7e5")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let before = store.hot_stats();
    store.archive.as_mut().unwrap().config.global_bytes = receipt.committed_bytes + 80;
    assert_eq!(
        store.archive_inactive(budget()).unwrap_err(),
        StoreError::ArchiveQuota("global archive")
    );
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn automatic_pressure_and_one_allocation_retry_preserve_serial_identity() {
    for failures in [1, 2] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let mut position = Position::startpos();
        let old = store.focus_actual_moves(position.snapshot()).unwrap();
        let old_state = store.situations.get(old).unwrap().state;
        position.make_move(mv("e2e4")).unwrap();
        store.focus_actual_moves(position.snapshot()).unwrap();
        store.set_archive_io_budget(budget()).unwrap();
        store.states.snapshots.fail_next_reservations(failures);
        position.make_move(mv("e7e5")).unwrap();
        let result = store.insert_situation(position.snapshot());
        assert_eq!(store.archive.as_ref().unwrap().next_generation, 1);
        assert!(store.states.get(old_state).is_err());
        if failures == 1 {
            let new = result.unwrap();
            let state = store.situations.get(new).unwrap().state;
            assert!(state.0 > old_state.0);
        } else {
            assert_eq!(
                result.unwrap_err(),
                StoreError::Capacity("injected hot allocation")
            );
        }
    }
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    for text in ["e2e4", "e7e5", "g1f3"] {
        store.focus_actual_moves(position.snapshot()).unwrap();
        position.make_move(mv(text)).unwrap();
    }
    store.focus_actual_moves(position.snapshot()).unwrap();
    assert_eq!(store.states.len(), 4);
    let receipt = store.reclaim_if_needed(budget()).unwrap().unwrap();
    assert_eq!(receipt.after.states, 1);
}

#[test]
fn stale_hot_pool_owner_and_situation_slot_cannot_alias_cold_evidence() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let old_state = store.situations.get(old).unwrap().state;
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    let loaded = store
        .pin_load(&receipt.situation_handle(old), budget())
        .unwrap();
    let new = loaded.situations[0].1;
    assert_ne!(old, new);
    assert!(store.situations.get(old).is_err());
    assert_eq!(store.situations.get(new).unwrap().state, old_state);
    store.release_loaded(loaded.pin).unwrap();
    store.states = StateStore::new(limits().states, limits().retained_history_bytes);
    store
        .states
        .insert(Position::startpos().snapshot())
        .unwrap();
    let before = store.hot_stats();
    assert_eq!(
        store
            .pin_load(&receipt.state_handle(old_state), budget())
            .unwrap_err(),
        StoreError::InvalidHandle("archive hot-store owner changed")
    );
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn engine_archive_keeps_roles_relationships_and_ordered_node_payloads() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let situation = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(situation).unwrap().state;
    let opaque_cpu = store.append_observation(observation(state)).unwrap();
    let record = crate::pals::engine::RoleRecord {
        revision: 7,
        parent_revision: Some(3),
        supersedes_revision: Some(2),
        origin_state: state,
        kind: crate::pals::engine::RecordKind::Repair,
        line: vec![mv("e2e4"), mv("e7e5")],
        value: Some(19),
        completed_depth: 4,
        score_scope: Some(crate::cpu::CpuScoreScope::CompletedIteration),
        cpu_observation: None,
        perspective: Color::White,
        critical: true,
    };
    let node = EngineArchiveNode {
        state,
        situation,
        edges: vec![(Move16::pack(mv("e2e4")).unwrap(), None)],
        metadata: b"node-v1:visits=9".to_vec(),
    };
    let receipt = store
        .archive_engine_records_with_pins(
            &[record],
            &[node],
            StorePins {
                observations: BTreeSet::from([opaque_cpu]),
                ..StorePins::default()
            },
            budget(),
        )
        .unwrap();
    assert_eq!(receipt.records, 1);
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    store.archive_inactive(budget()).unwrap();
    let loaded = store.load_engine_archive(&receipt, budget()).unwrap();
    assert_eq!(
        *store.observations.get(opaque_cpu).unwrap(),
        observation(state)
    );
    let record = &loaded.engine_records[0];
    assert_eq!(record.revision, 7);
    assert_eq!(record.parent_revision, Some(3));
    assert_eq!(record.supersedes_revision, Some(2));
    assert_eq!(record.value, Some(19));
    assert_eq!(record.line, vec![mv("e2e4"), mv("e7e5")]);
    assert!(record.critical);
    assert_eq!(loaded.engine_nodes[0].metadata, b"node-v1:visits=9");
    assert_eq!(
        loaded.engine_nodes[0].edges[0].0.unpack().unwrap(),
        mv("e2e4")
    );
    assert!(store.situations.get(situation).is_err());
    store.release_loaded(loaded.pin).unwrap();
}

#[test]
fn retired_pauses_preserve_partial_evidence_and_never_resume_old_checkpoints() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old_snapshot = position.snapshot();
    let old = store.focus_actual_moves(old_snapshot.clone()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            key(state),
            TaskConsumer {
                id: 21,
                situation: old,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("new task")
    };
    let mut partial = observation(state);
    partial.execution = Some(execution);
    partial.scope = EvidenceScope::DepthLimited {
        depth: 2,
        profile: 1,
        condition: 1,
    };
    let observation = store.append_observation(partial.clone()).unwrap();
    store.pause_task(execution, 11, Some(observation)).unwrap();
    assert_eq!(
        store.tasks.paused_executions_for_state(state).unwrap(),
        vec![execution]
    );
    assert_eq!(
        store
            .tasks
            .remap_paused_checkpoints(&BTreeMap::from([(state, 7)]))
            .unwrap(),
        1
    );
    assert!(matches!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::Paused { checkpoint: 7, .. }
    ));
    assert_eq!(
        store.tasks.retire_paused_except(&BTreeSet::new()).unwrap(),
        1
    );
    assert_eq!(store.tasks.active_states().count(), 0);
    assert_eq!(
        store
            .tasks
            .remap_paused_checkpoints(&BTreeMap::from([(state, 0)]))
            .unwrap(),
        0
    );
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.tasks.get(execution).is_err());
    let loaded = store
        .pin_load(&receipt.execution_handle(execution), budget())
        .unwrap();
    assert_eq!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::RetiredPaused {
            checkpoint: 7,
            evidence: Some(observation)
        }
    );
    assert_eq!(*store.observations.get(observation).unwrap(), partial);
    let new = store.focus_actual_moves(old_snapshot).unwrap();
    let admission = store
        .request_task(
            key(state),
            TaskConsumer {
                id: 22,
                situation: new,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap();
    assert!(matches!(admission,TaskAdmission::Start(new_execution) if new_execution.0>execution.0));
    assert!(store.complete_task(execution, observation).is_err());
    store.release_loaded(loaded.pin).unwrap();
}

fn model_observation(state: StateId, perspective: Color, wdl: [f32; 3]) -> Observation {
    let mut o = observation(state);
    o.value_identity = None;
    o.cpu_condition = None;
    o.model_value_identity = Some(crate::pals::value::ModelValueIdentity {
        semantics: crate::pals::value::MODEL_WDL_VALUE_SEMANTICS.into(),
        model: "actual-fixture-model".into(),
        encoding: "actual-fixture-encoding".into(),
        precision: "fp32".into(),
        model_epoch: [7; 32],
    });
    o.model_value_input = Some([state.0 as u8; 32]);
    o.scope = EvidenceScope::Model {
        model: 7,
        encoding: 8,
        input: 9,
    };
    o.kind = ObservationKind::Proposal;
    o.score = RawScore::Wdl {
        win: wdl[0],
        draw: wdl[1],
        loss: wdl[2],
        perspective,
    };
    o
}

fn context_model_observation(
    state: StateId,
    perspective: Color,
    wdl: [f32; 3],
    context_revision: u64,
) -> Observation {
    let mut o = model_observation(state, perspective, wdl);
    o.score = RawScore::ContextWdl {
        win: wdl[0],
        draw: wdl[1],
        loss: wdl[2],
        perspective,
        context_revision,
    };
    o
}

#[test]
fn conditional_model_evidence_stays_distinct_and_archives_both_actual_inputs() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let line = store
        .append_line(
            store.situations.get(old).unwrap().focus,
            &[mv("e2e4"), mv("e7e5")],
        )
        .unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let repaired_state = store.states.insert(position.snapshot()).unwrap();
    let repaired_raw = context_model_observation(repaired_state, Color::Black, [0.1, 0.2, 0.7], 43);
    let repaired = store.append_observation(repaired_raw.clone()).unwrap();
    position.make_move(mv("e7e5")).unwrap();
    let counter_state = store.states.insert(position.snapshot()).unwrap();
    let counter_raw = context_model_observation(counter_state, Color::White, [0.2, 0.3, 0.5], 43);
    let counter = store.append_observation(counter_raw.clone()).unwrap();
    let mut conditional = model_observation(state, Color::White, [0.2, 0.3, 0.5]);
    conditional.line = Some(line);
    conditional.kind = ObservationKind::Refutation;
    conditional.model_value_identity = None;
    conditional.model_value_input = None;
    conditional.score = RawScore::ConditionalWdl {
        expectation: 0.2 - 0.5,
        perspective: Color::White,
        repaired,
        counter,
        context_revision: 43,
    };
    let before = store.hot_stats();
    let mut invalid = conditional.clone();
    invalid.model_value_identity = counter_raw.model_value_identity.clone();
    invalid.model_value_input = counter_raw.model_value_input;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = conditional.clone();
    invalid.line = None;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = conditional.clone();
    invalid.score = RawScore::ConditionalWdl {
        expectation: 0.3,
        perspective: Color::White,
        repaired,
        counter,
        context_revision: 43,
    };
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = conditional.clone();
    invalid.score = RawScore::ConditionalWdl {
        expectation: 0.5 - 0.2,
        perspective: Color::Black,
        repaired,
        counter,
        context_revision: 43,
    };
    assert!(store.append_observation(invalid).is_err());
    assert_eq!(store.hot_stats(), before);
    let mut malformed_raw = counter_raw.clone();
    malformed_raw.model_value_input = None;
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &malformed_raw).is_err());
    let another_context =
        context_model_observation(counter_state, Color::White, [0.2, 0.3, 0.5], 44);
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &another_context).is_err());
    let legacy_raw = model_observation(counter_state, Color::White, [0.2, 0.3, 0.5]);
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &legacy_raw).is_err());
    let mut another_revision = conditional.clone();
    if let RawScore::ConditionalWdl {
        context_revision, ..
    } = &mut another_revision.score
    {
        *context_revision = 44;
    }
    assert!(validate_conditional_wdl(&another_revision, &repaired_raw, &counter_raw).is_err());
    let mut mismatched_raw = counter_raw.clone();
    mismatched_raw
        .model_value_identity
        .as_mut()
        .unwrap()
        .model_epoch = [8; 32];
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &mismatched_raw).is_err());
    let mut malformed_raw = counter_raw.clone();
    malformed_raw.score = RawScore::Wdl {
        win: f32::NAN,
        draw: 0.3,
        loss: 0.5,
        perspective: Color::White,
    };
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &malformed_raw).is_err());
    assert!(validate_conditional_wdl(&conditional, &counter_raw, &counter_raw).is_err());
    assert!(validate_conditional_wdl(&conditional, &repaired_raw, &conditional).is_err());

    let derived = store.append_observation(conditional.clone()).unwrap();
    store.dependencies.add(derived, old).unwrap();
    let closure = store
        .expand_pins(StorePins {
            situations: BTreeSet::from([old]),
            ..StorePins::default()
        })
        .unwrap();
    assert!(
        closure
            .observations
            .is_superset(&BTreeSet::from([derived, repaired, counter]))
    );
    assert!(
        closure
            .states
            .is_superset(&BTreeSet::from([state, repaired_state, counter_state]))
    );
    position.make_move(mv("g1f3")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.observations.get(derived).is_err());
    let loaded = store
        .pin_load(&receipt.observation_handle(derived), budget())
        .unwrap();
    assert_eq!(*store.observations.get(repaired).unwrap(), repaired_raw);
    assert_eq!(*store.observations.get(counter).unwrap(), counter_raw);
    let restored = store.observations.get(derived).unwrap();
    assert_eq!(*restored, conditional);
    assert!(restored.model_value_identity.is_none());
    assert!(restored.model_value_input.is_none());
    assert!(matches!(
        restored.score,
        RawScore::ConditionalWdl {
            context_revision: 43,
            ..
        }
    ));
    store.release_loaded(loaded.pin).unwrap();
}

#[test]
fn only_fresh_committed_actual_root_receipt_unlocks_a_full_hot_store() {
    let root = TestRoot::new();
    let mut small = limits();
    small.states = 1;
    small.situations = 1;
    let mut store = PalsStores::new(small);
    store.enable_archive(root.config()).unwrap();
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let old_state = store.situations.get(old).unwrap().state;
    let old_generation = store.generation();
    position.make_move(mv("e2e4")).unwrap();
    assert!(matches!(
        store.focus_actual_moves(position.snapshot()),
        Err(StoreError::Capacity(_))
    ));
    assert_eq!(store.root(), Some(old));
    let unrelated = store.archive_engine_records(&[], &[], budget()).unwrap();
    assert!(store.retire_actual_root_for_archive(&unrelated).is_err());
    assert_eq!(store.generation(), old_generation);
    let receipt = store
        .archive_engine_records(
            &[],
            &[EngineArchiveNode {
                state: old_state,
                situation: old,
                edges: vec![],
                metadata: b"root-v1".to_vec(),
            }],
            budget(),
        )
        .unwrap();
    let lookup = store
        .lookup_engine_archive(old_state, budget())
        .unwrap()
        .unwrap();
    assert!(store.retire_actual_root_for_archive(&lookup).is_err());
    assert_eq!(store.root(), Some(old));
    assert_eq!(
        store.retire_actual_root_for_archive(&receipt).unwrap(),
        Some(old)
    );
    assert_eq!(store.root(), None);
    assert_eq!(store.generation(), old_generation + 1);
    let reclaimed = store.archive_inactive(budget()).unwrap().unwrap();
    assert_eq!(reclaimed.after.states, 0);
    let new = store.focus_actual_moves(position.snapshot()).unwrap();
    assert_ne!(old, new);
    assert!(store.situations.get(old).is_err());
    assert!(store.states.get(old_state).is_err());
    assert!(store.retire_actual_root_for_archive(&receipt).is_err());
}

fn foreign_identity() -> crate::cpu_checker::ExternalCheckerIdentity {
    use crate::cpu_checker::{ExternalModelMetadata, ExternalTrainingKnowledge};
    crate::cpu_checker::ExternalCheckerIdentity {
        adapter_semantics: "fixture-external-archive-raw/1".into(),
        binary_sha256: "a".repeat(64),
        launch_arguments_sha256: "b".repeat(64),
        declared_name: "fixture".into(),
        declared_version: "1".into(),
        declared_source: "test-source".into(),
        declared_license: "MIT".into(),
        options: BTreeMap::new(),
        assets: vec![],
        model_metadata: ExternalModelMetadata {
            weights_sha256: None,
            training: ExternalTrainingKnowledge::Unknown,
            declared_rights: None,
            precision: None,
        },
    }
}
fn foreign_key(state: StateId) -> TaskKey {
    let mut task = key(state);
    task.value_identity = None;
    task.checker_identity = Some(CheckerIdentity::ExternalUci(foreign_identity()));
    task.cpu_condition = Some("archive-fixture-external-conditions".into());
    task
}
fn foreign_observation(
    state: StateId,
    execution: ExecutionId,
    pv: LineId,
    movement: BoardMove,
    perspective: Color,
    request_id: u64,
) -> Observation {
    let report = ExternalCheckerReport {
        identity: foreign_identity(),
        observed_uci: crate::cpu_checker::ExternalUciIdentity {
            name: "fixture".into(),
            author: None,
        },
        request_id,
        best_move: Some(movement),
        pv: vec![movement],
        score: ExternalRawScore::MateMoves(19),
        bound: ExternalBound::Lower,
        wdl_per_mille: Some([900, 50, 50]),
        perspective,
        requested_depth: 4,
        reported_depth: Some(1),
        seldepth: Some(3),
        root_restricted: false,
        completion: ExternalCompletion::BestMove,
        work: CheckerWork {
            nodes: Some(20),
            qnodes: None,
            tt_hits: None,
        },
        elapsed: Duration::from_millis(1),
    };
    let mut o = observation(state);
    o.source = external_source_id(&report.identity.adapter_semantics);
    o.value_identity = None;
    o.checker_identity = Some(CheckerIdentity::ExternalUci(report.identity.clone()));
    o.checker_work = Some(report.work);
    o.cpu_condition = Some("archive-fixture-external-conditions".into());
    o.cpu_pv = Some(pv);
    o.scope = EvidenceScope::ExternalUci {
        requested_depth: report.requested_depth,
        reported_depth: report.reported_depth,
        seldepth: report.seldepth,
        bound: report.bound,
    };
    o.score = RawScore::ExternalUci {
        value: report.score,
        bound: report.bound,
        perspective,
        wdl_per_mille: report.wdl_per_mille,
    };
    o.budget = 20;
    o.kind = ObservationKind::ExternalCpuAnalysis;
    o.execution = Some(execution);
    o.external_report = Some(Box::new(report));
    o
}

#[test]
fn foreign_request_uniqueness_scan_shares_explicit_budget_and_fails_closed() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            foreign_key(state),
            TaskConsumer {
                id: 71,
                situation: old,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("new foreign task")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    let raw = foreign_observation(state, execution, pv, mv("e2e4"), Color::White, 1);
    let before = store.hot_stats();
    assert_eq!(
        store.append_observation(raw.clone()).unwrap_err(),
        StoreError::ArchiveBudgetRequired
    );
    assert_eq!(store.hot_stats(), before);
    let mut shared = budget();
    let accepted = store
        .append_observation_checked_with_archive_budget(raw.clone(), &mut shared)
        .unwrap();
    store.complete_task(execution, accepted).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let current = store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store
        .archive_inactive_with_budget(&mut shared)
        .unwrap()
        .unwrap();
    assert!(store.observations.get(accepted).is_err());
    let current_state = store.situations.get(current).unwrap().state;
    let TaskAdmission::Start(other) = store
        .request_task(
            foreign_key(current_state),
            TaskConsumer {
                id: 72,
                situation: current,
                revision: 0,
                generation: store.generation(),
                deadline_tick: 100,
            },
            0,
        )
        .unwrap()
    else {
        panic!("another foreign task")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e7e5")]).unwrap();
    let duplicate = foreign_observation(current_state, other, pv, mv("e7e5"), Color::Black, 1);
    let before = store.hot_stats();
    let prior_bytes = shared.max_bytes;
    assert_eq!(
        store
            .append_observation_checked_with_archive_budget(duplicate.clone(), &mut shared)
            .unwrap_err(),
        StoreError::InvalidEvidence("foreign physical request belongs to another archived task")
    );
    assert_eq!(prior_bytes - shared.max_bytes, receipt.committed_bytes);
    assert_eq!(store.hot_stats(), before);
    let unique = foreign_observation(current_state, other, pv, mv("e7e5"), Color::Black, 2);
    let mut small = ArchiveIoBudget {
        max_bytes: receipt.committed_bytes - 1,
        ..budget()
    };
    assert_eq!(
        store
            .append_observation_checked_with_archive_budget(unique.clone(), &mut small)
            .unwrap_err(),
        StoreError::ArchiveByteBudget
    );
    assert_eq!(store.hot_stats(), before);
    let mut expired = ArchiveIoBudget {
        deadline: Instant::now(),
        ..budget()
    };
    assert_eq!(
        store
            .append_observation_checked_with_archive_budget(unique.clone(), &mut expired)
            .unwrap_err(),
        StoreError::ArchiveDeadline
    );
    assert_eq!(store.hot_stats(), before);
    let prior_bytes = shared.max_bytes;
    let new_id = store
        .append_observation_checked_with_archive_budget(unique.clone(), &mut shared)
        .unwrap();
    assert_eq!(prior_bytes - shared.max_bytes, receipt.committed_bytes);
    assert_eq!(*store.observations.get(new_id).unwrap(), unique);

    let file = store.archive.as_ref().unwrap().path(receipt.generation);
    use std::io::{Seek, SeekFrom};
    let mut output = OpenOptions::new().write(true).open(&file).unwrap();
    output.seek(SeekFrom::End(-1)).unwrap();
    output.write_all(b"!").unwrap();
    output.sync_all().unwrap();
    let third = foreign_observation(current_state, other, pv, mv("e7e5"), Color::Black, 3);
    let before = store.hot_stats();
    let prior_bytes = shared.max_bytes;
    assert_eq!(
        store
            .append_observation_checked_with_archive_budget(third, &mut shared)
            .unwrap_err(),
        StoreError::ArchiveIntegrity("archive checksum")
    );
    assert_eq!(prior_bytes - shared.max_bytes, receipt.committed_bytes);
    assert_eq!(store.hot_stats(), before);
    drop(output);
    fs::remove_file(file).unwrap();
    let directory = store.archive.as_ref().unwrap().directory.clone();
    fs::remove_dir(&directory).unwrap();
    let third = foreign_observation(current_state, other, pv, mv("e7e5"), Color::Black, 3);
    assert!(matches!(
        store.append_observation_checked_with_archive_budget(third, &mut shared),
        Err(StoreError::ArchiveIo { .. })
    ));
    assert_eq!(store.hot_stats(), before);
}

#[test]
fn bounded_encode_buffer_reserves_from_length_before_crossing_capacity() {
    let mut buffer = BoundedBuffer {
        bytes: Vec::new(),
        maximum: 9000,
        deadline: budget().deadline,
        error: None,
    };
    buffer.write_all(&[1; 8191]).unwrap();
    buffer.write_all(&[2; 2]).unwrap();
    buffer.write_all(&[3; 807]).unwrap();
    assert_eq!(buffer.bytes.len(), 9000);
    assert!(buffer.bytes.capacity() <= 9000);
    assert!(buffer.write_all(&[4]).is_err());
    assert_eq!(buffer.error, Some(StoreError::ArchiveByteBudget));
    assert_eq!(buffer.bytes.len(), 9000);
}

#[test]
fn conditional_repair_preserves_fresh_context_pair_and_active_refutation_link() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let focus = store.situations.get(old).unwrap().focus;
    let refuted_line = store.append_line(focus, &[mv("e2e4"), mv("e7e5")]).unwrap();
    let repaired_line = store.append_line(focus, &[mv("e2e4"), mv("c7c5")]).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let next_actual = position.clone();
    let proposal_state = store.states.insert(position.snapshot()).unwrap();
    let proposal_raw = context_model_observation(proposal_state, Color::Black, [0.1, 0.2, 0.7], 43);
    let proposal = store.append_observation(proposal_raw).unwrap();
    position.make_move(mv("e7e5")).unwrap();
    let counter_state = store.states.insert(position.snapshot()).unwrap();
    let counter = store
        .append_observation(context_model_observation(
            counter_state,
            Color::White,
            [0.2, 0.3, 0.5],
            43,
        ))
        .unwrap();
    let mut refutation = model_observation(state, Color::White, [0.2, 0.3, 0.5]);
    refutation.model_value_identity = None;
    refutation.model_value_input = None;
    refutation.kind = ObservationKind::Refutation;
    refutation.line = Some(refuted_line);
    refutation.score = RawScore::ConditionalWdl {
        expectation: 0.2 - 0.5,
        perspective: Color::White,
        repaired: proposal,
        counter,
        context_revision: 43,
    };
    let refutation_id = store.append_observation(refutation.clone()).unwrap();
    store
        .refute_continuation(old, refuted_line, refutation_id)
        .unwrap();
    let before_raw = context_model_observation(counter_state, Color::White, [0.2, 0.3, 0.5], 44);
    let before = store.append_observation(before_raw.clone()).unwrap();
    let mut repaired_position = next_actual.clone();
    repaired_position.make_move(mv("c7c5")).unwrap();
    let after_state = store.states.insert(repaired_position.snapshot()).unwrap();
    let after_raw = context_model_observation(after_state, Color::White, [0.4, 0.3, 0.3], 44);
    let after = store.append_observation(after_raw.clone()).unwrap();
    let mut repair = refutation.clone();
    repair.kind = ObservationKind::Repair;
    repair.line = Some(repaired_line);
    repair.supersedes = Some(refutation_id);
    repair.score = RawScore::ConditionalRepairWdl {
        expectation: 0.4 - 0.3,
        perspective: Color::White,
        before,
        after,
        context_revision: 44,
    };
    let hot = store.hot_stats();
    let mut invalid = repair.clone();
    invalid.supersedes = None;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = repair.clone();
    invalid.supersedes = Some(before);
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = repair.clone();
    invalid.model_value_identity = before_raw.model_value_identity.clone();
    invalid.model_value_input = before_raw.model_value_input;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = repair.clone();
    invalid.line = None;
    assert!(store.append_observation(invalid).is_err());
    let mut invalid = repair.clone();
    invalid.kind = ObservationKind::Refutation;
    assert!(store.append_observation(invalid).is_err());
    for score in [
        RawScore::ConditionalRepairWdl {
            expectation: 0.2 - 0.5,
            perspective: Color::White,
            before,
            after: before,
            context_revision: 44,
        },
        RawScore::ConditionalRepairWdl {
            expectation: 0.2 - 0.5,
            perspective: Color::White,
            before: after,
            after: before,
            context_revision: 44,
        },
        RawScore::ConditionalRepairWdl {
            expectation: 0.5,
            perspective: Color::White,
            before,
            after,
            context_revision: 44,
        },
        RawScore::ConditionalRepairWdl {
            expectation: 0.4 - 0.3,
            perspective: Color::White,
            before,
            after,
            context_revision: 45,
        },
    ] {
        let mut invalid = repair.clone();
        invalid.score = score;
        assert!(store.append_observation(invalid).is_err());
    }
    let different_context =
        context_model_observation(after_state, Color::White, [0.4, 0.3, 0.3], 45);
    assert!(
        validate_conditional_repair_wdl(&repair, &before_raw, &different_context, &refutation)
            .is_err()
    );
    let mut wrong_model = after_raw.clone();
    wrong_model
        .model_value_identity
        .as_mut()
        .unwrap()
        .model_epoch = [8; 32];
    assert!(
        validate_conditional_repair_wdl(&repair, &before_raw, &wrong_model, &refutation).is_err()
    );
    let legacy_raw = model_observation(counter_state, Color::White, [0.2, 0.3, 0.5]);
    assert!(
        validate_conditional_repair_wdl(&repair, &legacy_raw, &after_raw, &refutation).is_err()
    );
    assert_eq!(store.hot_stats(), hot);
    let repaired_observation = store.append_observation(repair.clone()).unwrap();
    store
        .repair(old, refuted_line, repaired_line, repaired_observation)
        .unwrap();
    let closure = store
        .expand_pins(StorePins {
            observations: BTreeSet::from([repaired_observation]),
            ..StorePins::default()
        })
        .unwrap();
    assert!(closure.observations.is_superset(&BTreeSet::from([
        before,
        after,
        repaired_observation,
        refutation_id,
        proposal,
        counter,
    ])));
    store.focus_actual_moves(next_actual.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    assert!(store.observations.get(repaired_observation).is_err());
    let loaded = store
        .pin_load(&receipt.situation_handle(old), budget())
        .unwrap();
    let restored_situation = loaded
        .situations
        .iter()
        .find(|(prior, _)| *prior == old)
        .unwrap()
        .1;
    assert_ne!(restored_situation, old);
    assert_eq!(*store.observations.get(before).unwrap(), before_raw);
    assert_eq!(*store.observations.get(after).unwrap(), after_raw);
    assert_eq!(
        *store.observations.get(repaired_observation).unwrap(),
        repair
    );
    assert_eq!(
        store
            .situations
            .get(restored_situation)
            .unwrap()
            .conclusions
            .get(refuted_line)
            .unwrap()
            .status,
        ContinuationStatus::RepairedBy(repaired_line)
    );
    assert_eq!(
        store
            .situations
            .get(restored_situation)
            .unwrap()
            .conclusions
            .get(repaired_line)
            .unwrap()
            .evidence,
        Some(repaired_observation)
    );
    store.release_loaded(loaded.pin).unwrap();
}

#[test]
fn legacy_wdl_wire_stays_unchanged_while_context_and_repair_are_distinct() {
    let legacy = model_observation(StateId(0), Color::White, [0.4, 0.3, 0.3]);
    let wire = ObservationWire::encode(&legacy).unwrap();
    let json = serde_json::to_value(&wire).unwrap();
    assert_eq!(
        json["score"],
        serde_json::json!({"Wdl": [[0.4_f32.to_bits(), 0.3_f32.to_bits(), 0.3_f32.to_bits()], 0]})
    );
    let decoded = serde_json::from_value::<ObservationWire>(json)
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(decoded, legacy);
    let context = context_model_observation(StateId(0), Color::White, [0.4, 0.3, 0.3], u64::MAX);
    let json = serde_json::to_value(ObservationWire::encode(&context).unwrap()).unwrap();
    assert!(json["score"].get("ContextWdl").is_some());
    assert_eq!(
        serde_json::from_value::<ObservationWire>(json)
            .unwrap()
            .decode()
            .unwrap(),
        context
    );
    let mut repair = legacy.clone();
    repair.kind = ObservationKind::Repair;
    repair.line = Some(LineId(7));
    repair.supersedes = Some(ObservationId(8));
    repair.model_value_identity = None;
    repair.model_value_input = None;
    repair.score = RawScore::ConditionalRepairWdl {
        expectation: 0.4 - 0.3,
        perspective: Color::White,
        before: ObservationId(3),
        after: ObservationId(4),
        context_revision: 44,
    };
    let wire = ObservationWire::encode(&repair).unwrap();
    let mut pins = StorePins::default();
    wire.add_pins(&mut pins);
    assert_eq!(
        pins.observations,
        BTreeSet::from([ObservationId(3), ObservationId(4), ObservationId(8)])
    );
    let json = serde_json::to_value(wire).unwrap();
    assert!(json["score"].get("ConditionalRepairWdl").is_some());
    assert_eq!(
        serde_json::from_value::<ObservationWire>(json)
            .unwrap()
            .decode()
            .unwrap(),
        repair
    );
}

#[test]
fn cold_load_revalidates_raw_context_revision_before_publishing_evidence() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let line = store
        .append_line(store.situations.get(old).unwrap().focus, &[mv("e2e4")])
        .unwrap();
    let repaired = store
        .append_observation(context_model_observation(
            state,
            Color::White,
            [0.8, 0.1, 0.1],
            43,
        ))
        .unwrap();
    let counter = store
        .append_observation(context_model_observation(
            state,
            Color::White,
            [0.2, 0.3, 0.5],
            43,
        ))
        .unwrap();
    let mut conditional = model_observation(state, Color::White, [0.2, 0.3, 0.5]);
    conditional.line = Some(line);
    conditional.kind = ObservationKind::Refutation;
    conditional.model_value_identity = None;
    conditional.model_value_input = None;
    conditional.score = RawScore::ConditionalWdl {
        expectation: 0.2 - 0.5,
        perspective: Color::White,
        repaired,
        counter,
        context_revision: 43,
    };
    let derived = store.append_observation(conditional).unwrap();
    // Bypass the public writer only in this corruption fixture. The segment
    // receives a valid checksum; load must independently verify its relation.
    let raw = store.observations.observations.get_mut(counter.0).unwrap();
    if let RawScore::ContextWdl {
        context_revision, ..
    } = &mut raw.score
    {
        *context_revision = 44;
    }
    position.make_move(mv("e2e4")).unwrap();
    store.focus_actual_moves(position.snapshot()).unwrap();
    let receipt = store.archive_inactive(budget()).unwrap().unwrap();
    let before = store.hot_stats();
    assert_eq!(
        store
            .pin_load(&receipt.observation_handle(derived), budget())
            .unwrap_err(),
        StoreError::InvalidEvidence(
            "conditional evidence needs actual raw WDL under its context revision"
        )
    );
    assert_eq!(store.hot_stats(), before);
    assert!(store.observations.get(derived).is_err());
    assert!(store.states.get(state).is_err());
}

#[test]
fn node_lookup_ignores_newer_record_only_segment_without_changing_generic_lookup() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let old = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let state = store.situations.get(old).unwrap().state;
    let nodes = store
        .archive_engine_records(
            &[],
            &[EngineArchiveNode {
                state,
                situation: old,
                edges: vec![],
                metadata: b"node-v1".to_vec(),
            }],
            budget(),
        )
        .unwrap();
    let record = crate::pals::engine::RoleRecord {
        revision: 8,
        parent_revision: None,
        supersedes_revision: None,
        origin_state: state,
        kind: crate::pals::engine::RecordKind::Proposal,
        line: vec![],
        value: None,
        completed_depth: 0,
        score_scope: None,
        cpu_observation: None,
        perspective: Color::White,
        critical: false,
    };
    let records = store
        .archive_engine_records(&[record], &[], budget())
        .unwrap();
    let generic = store
        .lookup_engine_archive(state, budget())
        .unwrap()
        .unwrap();
    assert_eq!(generic.archive.generation, records.archive.generation);
    assert_eq!(generic.nodes, 0);
    let mut shared = budget();
    let initial = shared.max_bytes;
    let with_nodes = store
        .lookup_engine_node_archive_with_budget(state, &mut shared)
        .unwrap()
        .unwrap();
    assert_eq!(with_nodes.archive.generation, nodes.archive.generation);
    assert_eq!(with_nodes.nodes, 1);
    assert_eq!(
        initial - shared.max_bytes,
        nodes.archive.committed_bytes + records.archive.committed_bytes
    );
    assert_eq!(with_nodes.archive.io_bytes, initial - shared.max_bytes);
}

#[test]
fn scoped_terminal_admission_keeps_uninstalled_situation_and_pending_raw_ids_hot() {
    let root = TestRoot::new();
    let mut small = limits();
    small.observations = 5;
    let mut store = PalsStores::new(small);
    store.enable_archive(root.config()).unwrap();
    let actual = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let actual_state = store.situations.get(actual).unwrap().state;
    let mate = Position::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let mut shared = budget();
    let original = shared;
    let pending = store
        .with_archive_allocation(StorePins::default(), &mut shared, |stores| {
            stores.insert_situation(mate.snapshot())
        })
        .unwrap();
    assert_eq!(shared.max_bytes, original.max_bytes);
    let pending_state = store.situations.get(pending).unwrap().state;
    let pending_line = store
        .append_line(store.situations.get(actual).unwrap().focus, &[mv("e2e4")])
        .unwrap();
    let mut raw = observation(actual_state);
    raw.line = Some(pending_line);
    let pending_raw = store.append_observation(raw).unwrap();
    let historical = (0..3)
        .map(|_| store.append_observation(observation(actual_state)).unwrap())
        .collect::<Vec<_>>();
    assert!(store.archive_pressure());
    let mut pins = StorePins::default();
    pins.states.insert(pending_state);
    pins.situations.insert(pending);
    pins.lines.insert(pending_line);
    pins.observations.insert(pending_raw);
    let terminal = store
        .with_archive_allocation(pins, &mut shared, |stores| {
            stores.append_rules_terminal_controlled(&mate, 0, 0, || Ok(()))
        })
        .unwrap();
    assert_eq!(store.situations.get(pending).unwrap().state, pending_state);
    assert!(store.lines.get(pending_line).is_ok());
    assert!(store.observations.get(pending_raw).is_ok());
    assert_eq!(
        store.observations.get(terminal).unwrap().state,
        pending_state
    );
    assert_eq!(
        store.observations.get(terminal).unwrap().score,
        RawScore::Terminal {
            winner: Some(Color::White)
        }
    );
    for id in historical {
        assert_eq!(
            store.observations.get(id),
            Err(StoreError::ColdRecord("observation"))
        );
    }
    let manager = store.archive.as_ref().unwrap();
    assert_eq!(manager.next_generation, 1);
    let bytes = fs::metadata(manager.path(0)).unwrap().len();
    assert_eq!(original.max_bytes - shared.max_bytes, 2 * bytes);
    assert_eq!(shared.deadline, original.deadline);
    assert!(!manager.allocation_active);
    assert!(manager.auto_budget.is_none());
    assert_eq!(manager.pins, StorePins::default());
}

#[test]
fn scoped_allocation_has_one_capacity_retry_and_returns_debits_on_both_outcomes() {
    for failures in [1, 2] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let mut position = Position::startpos();
        let old = store.focus_actual_moves(position.snapshot()).unwrap();
        let old_state = store.situations.get(old).unwrap().state;
        position.make_move(mv("e2e4")).unwrap();
        let actual = store.focus_actual_moves(position.snapshot()).unwrap();
        let mut old_pins = StorePins::default();
        old_pins.situations.insert(actual);
        store.set_archive_pins(old_pins.clone()).unwrap();
        store.states.snapshots.fail_next_reservations(failures);
        position.make_move(mv("e7e5")).unwrap();
        let mut shared = budget();
        let original = shared;
        let result = store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
            stores.insert_situation_controlled(position.snapshot(), || Ok(()))
        });
        let manager = store.archive.as_ref().unwrap();
        assert_eq!(manager.next_generation, 1);
        assert_eq!(
            original.max_bytes - shared.max_bytes,
            2 * fs::metadata(manager.path(0)).unwrap().len()
        );
        assert_eq!(shared.deadline, original.deadline);
        assert_eq!(manager.pins, old_pins);
        assert!(!manager.allocation_active);
        assert!(manager.auto_budget.is_none());
        assert!(store.states.get(old_state).is_err());
        assert!(store.situations.get(actual).is_ok());
        if failures == 1 {
            let fresh = result.unwrap();
            assert!(store.situations.get(fresh).unwrap().state.0 > old_state.0);
            assert_ne!(fresh, old);
        } else {
            assert_eq!(result, Err(StoreError::Capacity("injected hot allocation")));
        }
    }
}

#[test]
fn scoped_allowance_rejects_second_budget_and_restores_pins_after_error_and_unwind() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let actual = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let state = store.situations.get(actual).unwrap().state;
    let receipt = store.archive_engine_records(&[], &[], budget()).unwrap();
    let mut old_pins = StorePins::default();
    old_pins.states.insert(state);
    store.set_archive_pins(old_pins.clone()).unwrap();
    let mut extra = StorePins::default();
    extra.situations.insert(actual);
    let mut shared = budget();
    let original = shared;
    let conflict = StoreError::InvalidConditions("archive allocation allowance is active");
    let result: Result<(), StoreError> =
        store.with_archive_allocation(extra, &mut shared, |stores| {
            let mut other = budget();
            let other_original = other;
            assert_eq!(
                stores.with_archive_allocation(StorePins::default(), &mut other, |_| Ok(())),
                Err(conflict.clone())
            );
            assert_eq!(stores.set_archive_io_budget(other), Err(conflict.clone()));
            assert_eq!(
                stores.set_archive_pins(StorePins::default()),
                Err(conflict.clone())
            );
            assert_eq!(
                stores
                    .append_observation_checked_with_archive_budget(observation(state), &mut other),
                Err(conflict.clone())
            );
            assert!(matches!(
                stores.archive_inactive_with_budget(&mut other),
                Err(StoreError::InvalidConditions(_))
            ));
            assert!(matches!(
                stores.lookup_cold_state_id_with_budget(state, &mut other),
                Err(StoreError::InvalidConditions(_))
            ));
            assert!(matches!(
                stores.archive_engine_records_with_pins_and_budget(
                    &[],
                    &[],
                    StorePins::default(),
                    &mut other
                ),
                Err(StoreError::InvalidConditions(_))
            ));
            assert!(matches!(
                stores.read_engine_archive(&receipt, other),
                Err(StoreError::InvalidConditions(_))
            ));
            assert!(matches!(
                stores.pin_load(&receipt.archive.state_handle(state), other),
                Err(StoreError::InvalidConditions(_))
            ));
            assert_eq!(other.max_bytes, other_original.max_bytes);
            Err(StoreError::InvalidConditions("fixture operation failed"))
        });
    assert_eq!(
        result,
        Err(StoreError::InvalidConditions("fixture operation failed"))
    );
    assert_eq!(shared.max_bytes, original.max_bytes);
    assert_eq!(store.archive.as_ref().unwrap().pins, old_pins);
    assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.with_archive_allocation::<()>(StorePins::default(), &mut shared, |_| {
            panic!("fixture unwind")
        })
    }));
    assert!(unwind.is_err());
    let manager = store.archive.as_ref().unwrap();
    assert_eq!(manager.pins, old_pins);
    assert!(!manager.allocation_active);
    assert!(manager.auto_budget.is_none());
    store
        .with_archive_allocation(StorePins::default(), &mut shared, |_| Ok(()))
        .unwrap();
    store.set_archive_io_budget(original).unwrap();
    assert_eq!(
        store.with_archive_allocation(StorePins::default(), &mut shared, |_| Ok(())),
        Err(conflict)
    );
    assert_eq!(
        store
            .archive
            .as_ref()
            .unwrap()
            .auto_budget
            .unwrap()
            .max_bytes,
        original.max_bytes
    );
}

#[test]
fn scoped_quota_io_bytes_and_expired_deadline_failures_preserve_hot_rows() {
    for failure in ["quota", "io", "bytes", "deadline"] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let mut position = Position::startpos();
        store.focus_actual_moves(position.snapshot()).unwrap();
        position.make_move(mv("e2e4")).unwrap();
        store.focus_actual_moves(position.snapshot()).unwrap();
        position.make_move(mv("e7e5")).unwrap();
        let before = store.hot_stats();
        store.states.snapshots.fail_next_reservations(1);
        let directory = store.archive.as_ref().unwrap().directory.clone();
        let mut shared = budget();
        match failure {
            "quota" => store.archive.as_mut().unwrap().config.game_bytes = 80,
            "io" => fs::remove_dir(&directory).unwrap(),
            "bytes" => shared.max_bytes = 0,
            "deadline" => shared.deadline = Instant::now() - Duration::from_millis(1),
            _ => unreachable!(),
        }
        let original = shared;
        let error = store
            .with_archive_allocation(StorePins::default(), &mut shared, |stores| {
                stores.insert_situation(position.snapshot())
            })
            .unwrap_err();
        match failure {
            "quota" => assert_eq!(error, StoreError::ArchiveQuota("game archive")),
            "io" => assert!(matches!(error, StoreError::ArchiveIo { .. })),
            "bytes" => assert_eq!(error, StoreError::ArchiveByteBudget),
            "deadline" => assert_eq!(error, StoreError::ArchiveDeadline),
            _ => unreachable!(),
        }
        assert_eq!(store.hot_stats(), before);
        assert_eq!(shared.max_bytes, original.max_bytes);
        assert_eq!(shared.deadline, original.deadline);
        let manager = store.archive.as_ref().unwrap();
        assert!(!manager.allocation_active);
        assert!(manager.auto_budget.is_none());
        assert_eq!(manager.pins, StorePins::default());
        store.archive.as_mut().unwrap().config.game_bytes = GAME_ARCHIVE_BYTES;
        if failure == "io" {
            fs::create_dir(&directory).unwrap();
        }
        let mut recovered = budget();
        store
            .with_archive_allocation(StorePins::default(), &mut recovered, |stores| {
                stores.insert_situation(position.snapshot())
            })
            .unwrap();
    }
}

#[test]
fn scoped_dependency_refutation_and_repair_retry_do_not_publish_partial_revisions() {
    for operation in ["dependency", "refutation", "repair"] {
        for failures in [1, 2] {
            let root = TestRoot::new();
            let mut store = setup(&root);
            let mut position = Position::startpos();
            let old = store.focus_actual_moves(position.snapshot()).unwrap();
            let old_state = store.situations.get(old).unwrap().state;
            let old_raw = store.append_observation(observation(old_state)).unwrap();
            store.add_dependency(old_raw, old).unwrap();
            position.make_move(mv("e2e4")).unwrap();
            let actual = store.focus_actual_moves(position.snapshot()).unwrap();
            let current = store.situations.get(actual).unwrap().clone();
            let old_line = store.append_line(current.focus, &[mv("e7e5")]).unwrap();
            let repaired_line = store.append_line(current.focus, &[mv("c7c5")]).unwrap();
            let mut raw = observation(current.state);
            raw.line = Some(old_line);
            raw.kind = ObservationKind::Refutation;
            let refutation = store.append_observation(raw).unwrap();
            let evidence = if operation == "repair" {
                store
                    .refute_continuation(actual, old_line, refutation)
                    .unwrap();
                let mut raw = observation(current.state);
                raw.line = Some(repaired_line);
                raw.kind = ObservationKind::Repair;
                raw.supersedes = Some(refutation);
                store.append_observation(raw).unwrap()
            } else {
                refutation
            };
            let revision = store.situations.get(actual).unwrap().revision;
            store.dependencies.fail_allocations = failures;
            let mut shared = budget();
            let original = shared;
            let result =
                store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
                    match operation {
                        "dependency" => {
                            stores.add_dependency_controlled(evidence, actual, || Ok(()))
                        }
                        "refutation" => stores.refute_continuation_controlled(
                            actual,
                            old_line,
                            evidence,
                            || Ok(()),
                        ),
                        "repair" => stores.repair_controlled(
                            actual,
                            old_line,
                            repaired_line,
                            evidence,
                            || Ok(()),
                        ),
                        _ => unreachable!(),
                    }
                });
            assert_eq!(store.archive.as_ref().unwrap().next_generation, 1);
            assert!(shared.max_bytes < original.max_bytes);
            assert!(store.states.get(old_state).is_err());
            assert!(store.observations.get(evidence).is_ok());
            let current = store.situations.get(actual).unwrap();
            if failures == 2 {
                assert_eq!(
                    result,
                    Err(StoreError::Capacity("injected dependency allocation"))
                );
                assert_eq!(current.revision, revision);
                assert_eq!(store.dependencies.affected(evidence).count(), 0);
                if operation == "repair" {
                    assert_eq!(
                        current.conclusions.get(old_line).unwrap().status,
                        ContinuationStatus::Refuted
                    );
                    assert!(current.conclusions.get(repaired_line).is_none());
                } else {
                    assert!(current.conclusions.get(old_line).is_none());
                }
            } else {
                result.unwrap();
                assert_eq!(
                    store.dependencies.affected(evidence).collect::<Vec<_>>(),
                    vec![actual]
                );
                assert_eq!(
                    current.revision,
                    revision + u64::from(operation != "dependency")
                );
                if operation == "repair" {
                    assert_eq!(
                        current.conclusions.get(old_line).unwrap().status,
                        ContinuationStatus::RepairedBy(repaired_line)
                    );
                    assert_eq!(
                        current.conclusions.get(repaired_line).unwrap().status,
                        ContinuationStatus::Supported
                    );
                }
            }
            assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
        }
    }
}

#[test]
fn scoped_controls_stop_after_commit_before_initial_allocation_or_capacity_retry() {
    for forced in [false, true] {
        for cause in [StoreError::ArchiveCanceled, StoreError::ArchiveDeadline] {
            let root = TestRoot::new();
            let mut store = setup(&root);
            let mut position = Position::startpos();
            let old = store.focus_actual_moves(position.snapshot()).unwrap();
            let old_state = store.situations.get(old).unwrap().state;
            for movement in if forced {
                vec!["e2e4"]
            } else {
                vec!["e2e4", "e7e5", "g1f3"]
            } {
                position.make_move(mv(movement)).unwrap();
                store.focus_actual_moves(position.snapshot()).unwrap();
            }
            let actual = store.root().unwrap();
            if forced {
                store.states.snapshots.fail_next_reservations(1);
            }
            position
                .make_move(mv(if forced { "e7e5" } else { "b8c6" }))
                .unwrap();
            let serial_before = store.states.snapshots.next_id();
            let mut shared = budget();
            let original = shared;
            let mut calls = 0;
            let result =
                store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
                    stores.insert_situation_controlled(position.snapshot(), || {
                        calls += 1;
                        if calls == if forced { 4 } else { 2 } {
                            Err(cause.clone())
                        } else {
                            Ok(())
                        }
                    })
                });
            assert_eq!(result, Err(cause));
            assert_eq!(store.states.snapshots.next_id(), serial_before);
            assert!(store.states.find(&position.snapshot()).is_none());
            assert!(store.situations.get(actual).is_ok());
            assert!(store.states.get(old_state).is_err());
            let manager = store.archive.as_ref().unwrap();
            assert_eq!(manager.next_generation, 1);
            assert_eq!(
                original.max_bytes - shared.max_bytes,
                2 * fs::metadata(manager.path(0)).unwrap().len()
            );
            assert_eq!(shared.deadline, original.deadline);
            assert!(!manager.allocation_active);
            assert!(manager.auto_budget.is_none());
        }
    }
}

#[test]
fn scoped_pin_saturation_preserves_last_valid_rows_and_does_not_extend_budget() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let actual = store
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let state = store.situations.get(actual).unwrap().state;
    let mut pins = StorePins::default();
    pins.situations.insert(actual);
    for _ in 0..limits().observations {
        pins.observations
            .insert(store.append_observation(observation(state)).unwrap());
    }
    let before = store.hot_stats();
    let mut shared = budget();
    let original = shared;
    let result = store.with_archive_allocation(pins.clone(), &mut shared, |stores| {
        stores.append_observation(observation(state))
    });
    assert_eq!(
        result,
        Err(StoreError::PinSaturated("all hot records pinned"))
    );
    assert_eq!(store.hot_stats(), before);
    for id in pins.observations {
        assert!(store.observations.get(id).is_ok());
    }
    assert_eq!(shared.max_bytes, original.max_bytes);
    assert_eq!(shared.deadline, original.deadline);
    assert_eq!(store.archive.as_ref().unwrap().next_generation, 0);
    assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
}

#[test]
fn final_control_failure_keeps_raw_evidence_without_publishing_refutation_or_repair() {
    for repairing in [false, true] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let actual = store
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let current = store.situations.get(actual).unwrap().clone();
        let refuted = store.append_line(current.focus, &[mv("e2e4")]).unwrap();
        let repaired = store.append_line(current.focus, &[mv("d2d4")]).unwrap();
        let mut raw = observation(current.state);
        raw.line = Some(refuted);
        raw.kind = ObservationKind::Refutation;
        let refutation = store.append_observation(raw).unwrap();
        let evidence = if repairing {
            store
                .refute_continuation(actual, refuted, refutation)
                .unwrap();
            let mut raw = observation(current.state);
            raw.line = Some(repaired);
            raw.kind = ObservationKind::Repair;
            raw.supersedes = Some(refutation);
            store.append_observation(raw).unwrap()
        } else {
            refutation
        };
        let retained = store.observations.get(evidence).unwrap().clone();
        let revision = store.situations.get(actual).unwrap().revision;
        let mut calls = 0;
        let mut shared = budget();
        let original = shared;
        let result = store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
            let controls = || {
                calls += 1;
                if calls == 3 {
                    Err(StoreError::ArchiveCanceled)
                } else {
                    Ok(())
                }
            };
            if repairing {
                stores.repair_controlled(actual, refuted, repaired, evidence, controls)
            } else {
                stores.refute_continuation_controlled(actual, refuted, evidence, controls)
            }
        });
        assert_eq!(result, Err(StoreError::ArchiveCanceled));
        assert_eq!(store.situations.get(actual).unwrap().revision, revision);
        assert_eq!(store.observations.get(evidence).unwrap(), &retained);
        assert_eq!(
            store.dependencies.affected(evidence).collect::<Vec<_>>(),
            vec![actual]
        );
        let conclusions = &store.situations.get(actual).unwrap().conclusions;
        assert!(conclusions.get(repaired).is_none());
        if repairing {
            assert_eq!(
                conclusions.get(refuted).unwrap().status,
                ContinuationStatus::Refuted
            );
        } else {
            assert!(conclusions.get(refuted).is_none());
        }
        assert_eq!(shared.max_bytes, original.max_bytes);
        assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
    }
}

#[test]
fn unreturned_task_admission_retires_only_new_reservations_and_keeps_physical_facts() {
    for admission_kind in ["start", "join", "reuse", "resume"] {
        let root = TestRoot::new();
        let mut store = setup(&root);
        let actual = store
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = store.situations.get(actual).unwrap().state;
        let consumer = TaskConsumer {
            id: 1,
            situation: actual,
            revision: 0,
            generation: store.generation(),
            deadline_tick: 100,
        };
        let previous = if admission_kind != "start" {
            let TaskAdmission::Start(execution) =
                store.request_task(key(state), consumer, 0).unwrap()
            else {
                panic!("start")
            };
            Some(execution)
        } else {
            None
        };
        let mut completed = None;
        if admission_kind == "reuse" {
            let execution = previous.unwrap();
            let mut raw = observation(state);
            raw.execution = Some(execution);
            let evidence = store.append_observation(raw).unwrap();
            store.complete_task(execution, evidence).unwrap();
            completed = Some(evidence);
        } else if admission_kind == "resume" {
            store.pause_task(previous.unwrap(), 7, None).unwrap();
        }
        let mut calls = 0;
        let mut shared = budget();
        let result = store.with_archive_allocation(StorePins::default(), &mut shared, |stores| {
            stores.request_task_controlled(
                key(state),
                TaskConsumer { id: 2, ..consumer },
                0,
                || {
                    calls += 1;
                    if calls == 3 {
                        Err(StoreError::ArchiveCanceled)
                    } else {
                        Ok(())
                    }
                },
            )
        });
        assert_eq!(result, Err(StoreError::ArchiveCanceled));
        let issued = *store.tasks.latest.get(&key(state)).unwrap();
        let task = store.tasks.get(issued).unwrap();
        assert!(
            task.consumers
                .iter()
                .find(|record| record.consumer.id == 2)
                .unwrap()
                .cancelled
        );
        match admission_kind {
            "start" => {
                assert_eq!(task.status, TaskStatus::Failed);
                assert!(store.tasks.active_states().next().is_none());
            }
            "resume" => {
                assert_eq!(task.status, TaskStatus::Failed);
                assert_eq!(task.resumed_from, previous);
                assert!(matches!(
                    store.tasks.get(previous.unwrap()).unwrap().status,
                    TaskStatus::Paused { checkpoint: 7, .. }
                ));
            }
            "join" => {
                assert_eq!(issued, previous.unwrap());
                assert_eq!(task.status, TaskStatus::InFlight);
                assert!(!task.consumers[0].cancelled);
            }
            "reuse" => {
                assert_eq!(issued, previous.unwrap());
                assert_eq!(task.status, TaskStatus::Completed(completed.unwrap()));
                assert!(store.observations.get(completed.unwrap()).is_ok());
            }
            _ => unreachable!(),
        }
        assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
    }
}

#[test]
fn controlled_foreign_scan_and_pressure_commit_return_shared_debits_on_cancel() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let consumer = TaskConsumer {
        id: 71,
        situation: old,
        revision: 0,
        generation: store.generation(),
        deadline_tick: 100,
    };
    let TaskAdmission::Start(execution) =
        store.request_task(foreign_key(state), consumer, 0).unwrap()
    else {
        panic!("start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    let raw = foreign_observation(state, execution, pv, mv("e2e4"), Color::White, 1);
    let accepted = store
        .append_observation_checked_with_archive_budget(raw, &mut budget())
        .unwrap();
    store.complete_task(execution, accepted).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let actual = store.focus_actual_moves(position.snapshot()).unwrap();
    store.archive_inactive(budget()).unwrap().unwrap();
    let state = store.situations.get(actual).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            foreign_key(state),
            TaskConsumer {
                id: 72,
                situation: actual,
                generation: store.generation(),
                ..consumer
            },
            0,
        )
        .unwrap()
    else {
        panic!("new start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e7e5")]).unwrap();
    let raw = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 2);
    for _ in 0..7 {
        store.append_observation(observation(state)).unwrap();
    }
    let serial_before = store.observations.observations.next_id();
    let mut calls = 0;
    let mut shared = budget();
    let original = shared;
    let result =
        store.append_observation_checked_with_archive_budget_controlled(raw, &mut shared, || {
            calls += 1;
            if calls == 2 {
                Err(StoreError::ArchiveCanceled)
            } else {
                Ok(())
            }
        });
    assert_eq!(result, Err(StoreError::ArchiveCanceled));
    assert_eq!(store.observations.observations.next_id(), serial_before);
    assert_eq!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::InFlight
    );
    assert!(store.lines.get(pv).is_ok());
    assert!(store.states.get(state).is_ok());
    let manager = store.archive.as_ref().unwrap();
    assert_eq!(manager.next_generation, 2);
    let read = fs::metadata(manager.path(0)).unwrap().len();
    let committed = fs::metadata(manager.path(1)).unwrap().len();
    assert_eq!(original.max_bytes - shared.max_bytes, read + 2 * committed);
    assert_eq!(shared.deadline, original.deadline);
    assert!(manager.auto_budget.is_none());
}

#[test]
fn checked_first_foreign_append_keeps_original_budget_and_never_reclaims_or_retries() {
    let root = TestRoot::new();
    let mut store = setup(&root);
    let mut position = Position::startpos();
    let old = store.focus_actual_moves(position.snapshot()).unwrap();
    let state = store.situations.get(old).unwrap().state;
    let consumer = TaskConsumer {
        id: 71,
        situation: old,
        revision: 0,
        generation: store.generation(),
        deadline_tick: 100,
    };
    let TaskAdmission::Start(execution) =
        store.request_task(foreign_key(state), consumer, 0).unwrap()
    else {
        panic!("start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e2e4")]).unwrap();
    let raw = foreign_observation(state, execution, pv, mv("e2e4"), Color::White, 1);
    let accepted = store
        .append_observation_checked_with_archive_budget(raw, &mut budget())
        .unwrap();
    store.complete_task(execution, accepted).unwrap();
    position.make_move(mv("e2e4")).unwrap();
    let actual = store.focus_actual_moves(position.snapshot()).unwrap();
    store.archive_inactive(budget()).unwrap().unwrap();
    let state = store.situations.get(actual).unwrap().state;
    let TaskAdmission::Start(execution) = store
        .request_task(
            foreign_key(state),
            TaskConsumer {
                id: 72,
                situation: actual,
                generation: store.generation(),
                ..consumer
            },
            0,
        )
        .unwrap()
    else {
        panic!("new start")
    };
    let pv = store.append_cpu_pv(&position, &[mv("e7e5")]).unwrap();
    store.tasks.cancel_consumer(execution, 72).unwrap();
    let retained = (0..7)
        .map(|_| store.append_observation(observation(state)).unwrap())
        .collect::<Vec<_>>();
    assert!(store.archive_pressure());
    let before = store.hot_stats();
    let serial_before = store.observations.observations.next_id();
    let mut shared = budget();
    let original = shared;
    let duplicate = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 1);
    assert_eq!(
        store.append_observation_checked_first_with_archive_budget(duplicate, &mut shared),
        Err(StoreError::InvalidEvidence(
            "foreign physical request belongs to another archived task"
        ))
    );
    let cold_bytes = fs::metadata(store.archive.as_ref().unwrap().path(0))
        .unwrap()
        .len();
    assert_eq!(original.max_bytes - shared.max_bytes, cold_bytes);
    let raw = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 2);
    let mut expired = ArchiveIoBudget {
        deadline: Instant::now() - Duration::from_millis(1),
        ..shared
    };
    assert_eq!(
        store.append_observation_checked_first_with_archive_budget(raw.clone(), &mut expired),
        Err(StoreError::ArchiveDeadline)
    );
    assert_eq!(expired.max_bytes, shared.max_bytes);
    let mut bytes = ArchiveIoBudget {
        max_bytes: 1,
        ..shared
    };
    assert_eq!(
        store.append_observation_checked_first_with_archive_budget(raw.clone(), &mut bytes),
        Err(StoreError::ArchiveByteBudget)
    );
    assert_eq!(bytes.max_bytes, 1);
    store.observations.observations.fail_next_reservations(2);
    for _ in 0..2 {
        let remaining = shared.max_bytes;
        assert_eq!(
            store.append_observation_checked_first_with_archive_budget(raw.clone(), &mut shared),
            Err(StoreError::Capacity("injected hot allocation"))
        );
        assert_eq!(remaining - shared.max_bytes, cold_bytes);
        assert_eq!(store.hot_stats(), before);
        assert_eq!(store.observations.observations.next_id(), serial_before);
        assert_eq!(store.archive.as_ref().unwrap().next_generation, 1);
    }
    let accepted = store
        .append_observation_checked_first_with_archive_budget(raw, &mut shared)
        .unwrap();
    assert_eq!(accepted.0, serial_before);
    assert_eq!(store.observations.len(), before.observations + 1);
    for id in retained {
        assert!(store.observations.get(id).is_ok());
    }
    assert_eq!(
        store.tasks.get(execution).unwrap().status,
        TaskStatus::CancellationRequested
    );
    let third = foreign_observation(state, execution, pv, mv("e7e5"), Color::Black, 3);
    assert_eq!(
        store.append_observation_checked_first_with_archive_budget(third, &mut shared),
        Err(StoreError::Capacity("observations"))
    );
    assert_eq!(store.archive.as_ref().unwrap().next_generation, 1);
    assert_eq!(shared.deadline, original.deadline);
    assert!(store.archive.as_ref().unwrap().auto_budget.is_none());
}
