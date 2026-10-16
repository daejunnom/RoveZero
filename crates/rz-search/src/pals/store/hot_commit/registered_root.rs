//! Logical admission for a root generation already issued by the UCI owner.
//!
//! The opaque registration is a single-use capability for this concrete Store,
//! its current root and exact Rules snapshot. It proves no CPU/GPU physical
//! completion: the execution owner must establish its fences separately.

use super::*;

/// A checked, immutable root transition. The private fields and absence of
/// `Clone` keep another caller from manufacturing or duplicating admission.
/// Independently prepared registrations become stale after the first commit.
#[derive(Debug)]
pub struct HotRootRegistration {
    owner: StoreReadOwner,
    current_generation: u64,
    current_root: Option<SituationId>,
    current_root_state: Option<(StateId, LineId, PositionSnapshot)>,
    snapshot: PositionSnapshot,
    target_generation: u64,
    admission: RootGenerationAdmission,
}

#[derive(Debug)]
enum RootGenerationAdmission {
    Advance,
    /// Engine-local retirement attestation is checked by the caller. Store
    /// validates only that retirement has left this generation without a root.
    AfterAttestedRetirement,
}

impl HotRootRegistration {
    /// The original global generation, rather than a Store-local increment.
    pub fn target_generation(&self) -> u64 {
        self.target_generation
    }

    /// The exact state and known history that this registration will adopt.
    pub fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
}

impl PalsStores {
    /// Prepare a hot-only transition to a generation issued by the global UCI
    /// authority. Skipped generations are valid; reuse or rollback is not.
    /// This reads hot records only and performs no archive recovery or I/O.
    pub fn prepare_hot_root_registration(
        &self,
        snapshot: PositionSnapshot,
        target_generation: u64,
    ) -> Result<HotRootRegistration, StoreError> {
        self.prepare_registered_root(
            snapshot,
            target_generation,
            RootGenerationAdmission::Advance,
        )
    }

    /// The engine must first check its actual-root retirement receipt against
    /// the global authority and registered root. This crate-private path only
    /// fills the already advanced, vacant generation; it never authorizes an
    /// equality transition for an ordinary live root or a fresh Store.
    pub(crate) fn prepare_hot_root_registration_after_retirement(
        &self,
        snapshot: PositionSnapshot,
        target_generation: u64,
    ) -> Result<HotRootRegistration, StoreError> {
        self.prepare_registered_root(
            snapshot,
            target_generation,
            RootGenerationAdmission::AfterAttestedRetirement,
        )
    }

    fn prepare_registered_root(
        &self,
        snapshot: PositionSnapshot,
        target_generation: u64,
        admission: RootGenerationAdmission,
    ) -> Result<HotRootRegistration, StoreError> {
        if !registered_generation_valid(&admission, self, target_generation) {
            return Err(StoreError::InvalidConditions(
                "registered root generation must advance",
            ));
        }
        self.validate_registered_root_quiescence()?;
        validate_registered_snapshot(&snapshot)?;
        let current_root_state = self
            .root
            .map(|root| {
                let situation = self.situations.get(root)?;
                self.validate_registered_root_line(situation.state, situation.focus)?;
                Ok::<_, StoreError>((
                    situation.state,
                    situation.focus,
                    self.states.get(situation.state)?.clone(),
                ))
            })
            .transpose()?;
        Ok(HotRootRegistration {
            owner: self.read_owner(),
            current_generation: self.generation,
            current_root: self.root,
            current_root_state,
            snapshot,
            target_generation,
            admission,
        })
    }

    /// Adopt exactly the checked global generation after hot allocation has
    /// succeeded. Capacity and validation errors preserve root/generation;
    /// this path never invokes automatic archive collection or cold loading.
    pub fn hot_focus_registered_root(
        &mut self,
        registration: HotRootRegistration,
    ) -> Result<SituationId, StoreError> {
        if !registration.owner.matches(self)
            || registration.current_generation != self.generation
            || registration.current_root != self.root
            || !registered_generation_valid(
                &registration.admission,
                self,
                registration.target_generation,
            )
        {
            return Err(StoreError::StaleConsumer);
        }
        match (self.root, registration.current_root_state.as_ref()) {
            (None, None) => {}
            (Some(root), Some((state, focus, snapshot))) => {
                let situation = self
                    .situations
                    .get(root)
                    .map_err(|_| StoreError::StaleConsumer)?;
                if situation.state != *state
                    || situation.focus != *focus
                    || !self
                        .states
                        .get(*state)
                        .map_err(|_| StoreError::StaleConsumer)?
                        .same_state(snapshot)
                {
                    return Err(StoreError::StaleConsumer);
                }
                self.validate_registered_root_line(*state, *focus)
                    .map_err(|_| StoreError::StaleConsumer)?;
            }
            _ => return Err(StoreError::StaleConsumer),
        }
        self.validate_registered_root_quiescence()?;
        validate_registered_snapshot(&registration.snapshot)?;
        let matching = self.states.find(&registration.snapshot).and_then(|state| {
            self.situations
                .iter()
                .find(|(_, situation)| {
                    situation.state == state
                        && self
                            .validate_registered_root_line(state, situation.focus)
                            .is_ok()
                })
                .map(|(id, _)| id)
        });
        let root = match matching {
            Some(root) => root,
            None => self.hot_insert_situation(registration.snapshot)?,
        };
        let situation = self.situations.get(root)?;
        self.validate_registered_root_line(situation.state, situation.focus)?;
        // No fallible action follows publication of either authority field.
        self.root = Some(root);
        self.generation = registration.target_generation;
        Ok(root)
    }

    fn validate_registered_root_quiescence(&self) -> Result<(), StoreError> {
        self.reject_archive_allocation_scope()?;
        if self.tasks.active_states().next().is_some() {
            return Err(StoreError::InvalidConditions(
                "registered root requires no active tasks",
            ));
        }
        if self.review_pin_count()? != 0 {
            return Err(StoreError::InvalidConditions(
                "registered root requires no review pins",
            ));
        }
        Ok(())
    }

    fn validate_registered_root_line(
        &self,
        state: StateId,
        focus: LineId,
    ) -> Result<(), StoreError> {
        let line = self.lines.get(focus)?;
        if line.start_state != state || line.parent.is_some() || line.plies() != 0 {
            return Err(StoreError::InvalidConditions(
                "registered root focus is not its exact root line",
            ));
        }
        Ok(())
    }
}

fn registered_generation_valid(
    admission: &RootGenerationAdmission,
    stores: &PalsStores,
    target_generation: u64,
) -> bool {
    match admission {
        RootGenerationAdmission::Advance => target_generation > stores.generation,
        RootGenerationAdmission::AfterAttestedRetirement => {
            stores.root.is_none()
                && stores.generation != 0
                && target_generation == stores.generation
        }
    }
}

fn validate_registered_snapshot(snapshot: &PositionSnapshot) -> Result<(), StoreError> {
    // PositionSnapshot is issued by Rules with private internals. Restoring its
    // checked immutable identity validates the bounded history without losing
    // origin/completeness through a FEN-only reconstruction.
    let restored = Position::from_snapshot(snapshot, rz_position::PositionLimits::default())
        .map_err(|_| StoreError::InvalidEvidence("registered root Rules snapshot"))?;
    if !snapshot.same_state(&restored.snapshot()) {
        return Err(StoreError::InvalidEvidence(
            "registered root Rules snapshot",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestArchiveRoot {
        root: std::path::PathBuf,
        parent: std::path::PathBuf,
    }
    impl TestArchiveRoot {
        fn new() -> Self {
            let parent = std::env::temp_dir().canonicalize().unwrap();
            let serial = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = parent.join(format!(
                "rz-pals-registered-root-tests-{}-{serial}",
                std::process::id()
            ));
            std::fs::create_dir(&root).unwrap();
            Self { root, parent }
        }
    }
    impl Drop for TestArchiveRoot {
        fn drop(&mut self) {
            // Only this unique resolved non-link temporary child is ours.
            if self.root.parent() == Some(self.parent.as_path())
                && self.root.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rz-pals-registered-root-tests-")
                })
                && std::fs::symlink_metadata(&self.root)
                    .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                let _ = std::fs::remove_dir_all(&self.root);
            }
        }
    }

    fn register(
        stores: &mut PalsStores,
        position: &Position,
        generation: u64,
    ) -> Result<SituationId, StoreError> {
        let registration = stores.prepare_hot_root_registration(position.snapshot(), generation)?;
        stores.hot_focus_registered_root(registration)
    }

    fn model_task(stores: &mut PalsStores, root: SituationId) -> ExecutionId {
        let situation = stores.situations.get(root).unwrap();
        let key = TaskKey {
            state: situation.state,
            line: Some(situation.focus),
            question: TaskQuestion::ModelProposal,
            root_moves: Vec::new(),
            model: 1,
            epoch: 0,
            value_identity: None,
            checker_identity: None,
            cpu_condition: None,
            profile: 1,
            condition: 1,
            input_revision: 0,
            requested_depth: 0,
            node_budget: 1,
        };
        let consumer = TaskConsumer {
            id: 1,
            situation: root,
            revision: situation.revision,
            generation: stores.generation(),
            deadline_tick: 100,
        };
        let TaskAdmission::Start(execution) = stores.hot_request_task(key, consumer, 0).unwrap()
        else {
            panic!("new model task");
        };
        execution
    }

    #[test]
    fn registered_root_adopts_skipped_global_generation_and_reuses_exact_state() {
        let mut stores = PalsStores::new(StoreLimits {
            states: 1,
            line_chunks: 1,
            situations: 1,
            ..StoreLimits::default()
        });
        let position = Position::startpos();
        let root = register(&mut stores, &position, 7).unwrap();
        let registration = stores
            .prepare_hot_root_registration(position.snapshot(), 93)
            .unwrap();
        assert_eq!(registration.target_generation(), 93);
        assert!(registration.snapshot().same_state(&position.snapshot()));
        assert_eq!(
            stores.hot_focus_registered_root(registration).unwrap(),
            root
        );
        assert_eq!(stores.generation(), 93);
        assert_eq!(stores.states.len(), 1);
        assert_eq!(stores.lines.len(), 1);
        assert_eq!(stores.situations.len(), 1);
        for invalid in [0, 7, 92, 93] {
            assert!(matches!(
                stores.prepare_hot_root_registration(position.snapshot(), invalid),
                Err(StoreError::InvalidConditions(
                    "registered root generation must advance"
                ))
            ));
        }
        assert_eq!(register(&mut stores, &position, u64::MAX).unwrap(), root);
        assert_eq!(stores.generation(), u64::MAX);
        assert!(
            stores
                .prepare_hot_root_registration(position.snapshot(), u64::MAX)
                .is_err()
        );
    }

    #[test]
    fn registered_root_refuses_another_store_and_replaced_state_owner() {
        let position = Position::startpos();
        let mut first = PalsStores::new(StoreLimits::default());
        let first_root = register(&mut first, &position, 1).unwrap();
        let registration = first
            .prepare_hot_root_registration(position.snapshot(), 2)
            .unwrap();
        let mut second = PalsStores::new(StoreLimits::default());
        let second_root = register(&mut second, &position, 1).unwrap();
        assert_eq!(first_root, second_root);
        assert_eq!(
            second.hot_focus_registered_root(registration),
            Err(StoreError::StaleConsumer)
        );
        assert_eq!((second.root(), second.generation()), (Some(second_root), 1));
        let registration = first
            .prepare_hot_root_registration(position.snapshot(), 2)
            .unwrap();
        first.states = StateStore::new(8_192, 64 * 1024 * 1024);
        first.states.insert(position.snapshot()).unwrap();
        assert_eq!(
            first.hot_focus_registered_root(registration),
            Err(StoreError::StaleConsumer)
        );
        assert_eq!((first.root(), first.generation()), (Some(first_root), 1));
    }

    #[test]
    fn separately_prepared_registration_cannot_repeat_a_committed_transition() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = register(&mut stores, &position, 1).unwrap();
        let first = stores
            .prepare_hot_root_registration(position.snapshot(), 3)
            .unwrap();
        let duplicate = stores
            .prepare_hot_root_registration(position.snapshot(), 3)
            .unwrap();
        assert_eq!(stores.hot_focus_registered_root(first).unwrap(), root);
        let counts = (
            stores.states.len(),
            stores.lines.len(),
            stores.situations.len(),
        );
        assert_eq!(
            stores.hot_focus_registered_root(duplicate),
            Err(StoreError::StaleConsumer)
        );
        assert_eq!((stores.root(), stores.generation()), (Some(root), 3));
        assert_eq!(
            counts,
            (
                stores.states.len(),
                stores.lines.len(),
                stores.situations.len()
            )
        );
    }

    #[test]
    fn registered_root_requires_same_root_authority_and_exact_root_state() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let mut position = Position::startpos();
        let root = register(&mut stores, &position, 1).unwrap();
        let registration = stores
            .prepare_hot_root_registration(position.snapshot(), 4)
            .unwrap();
        position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let other = stores.hot_insert_situation(position.snapshot()).unwrap();
        stores.root = Some(other);
        assert_eq!(
            stores.hot_focus_registered_root(registration),
            Err(StoreError::StaleConsumer)
        );
        stores.root = Some(root);
        let registration = stores
            .prepare_hot_root_registration(Position::startpos().snapshot(), 4)
            .unwrap();
        let other_state = stores.situations.get(other).unwrap().state;
        stores.situations.get_mut(root).unwrap().state = other_state;
        assert_eq!(
            stores.hot_focus_registered_root(registration),
            Err(StoreError::StaleConsumer)
        );
        assert_eq!(stores.generation(), 1);
    }

    #[test]
    fn root_focus_tamper_rejects_existing_proof_and_new_registration() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = register(&mut stores, &position, 1).unwrap();
        let registration = stores
            .prepare_hot_root_registration(position.snapshot(), 2)
            .unwrap();
        let focus = stores.situations.get(root).unwrap().focus;
        let line = stores
            .hot_append_line(focus, &[BoardMove::from_uci("e2e4").unwrap()])
            .unwrap();
        stores.situations.get_mut(root).unwrap().focus = line;
        assert_eq!(
            stores.hot_focus_registered_root(registration),
            Err(StoreError::StaleConsumer)
        );
        assert!(matches!(
            stores.prepare_hot_root_registration(position.snapshot(), 2),
            Err(StoreError::InvalidConditions(
                "registered root focus is not its exact root line"
            ))
        ));
        assert_eq!(stores.generation(), 1);
    }

    #[test]
    fn target_reuse_never_adopts_a_nonempty_focus_as_the_actual_root() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let mut position = Position::startpos();
        register(&mut stores, &position, 1).unwrap();
        position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let stale_target = stores.hot_insert_situation(position.snapshot()).unwrap();
        let target_state = stores.situations.get(stale_target).unwrap().state;
        let target_focus = stores.situations.get(stale_target).unwrap().focus;
        let other_focus = stores
            .hot_append_line(target_focus, &[BoardMove::from_uci("e7e5").unwrap()])
            .unwrap();
        stores.situations.get_mut(stale_target).unwrap().focus = other_focus;
        let actual = register(&mut stores, &position, 3).unwrap();
        assert_ne!(actual, stale_target);
        let actual = stores.situations.get(actual).unwrap();
        assert_eq!(actual.state, target_state);
        assert_eq!(actual.focus, target_focus);
        assert_eq!(stores.states.len(), 2);
    }

    #[test]
    fn registered_root_keeps_same_fen_with_different_history_distinct() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let complete = Position::startpos();
        let root = register(&mut stores, &complete, 1).unwrap();
        let imported = Position::from_fen(&complete.to_fen()).unwrap();
        assert_eq!(complete.to_fen(), imported.to_fen());
        assert!(!complete.snapshot().same_state(&imported.snapshot()));
        let different = register(&mut stores, &imported, 8).unwrap();
        assert_ne!(different, root);
        assert_ne!(
            stores.situations.get(different).unwrap().state,
            stores.situations.get(root).unwrap().state
        );
        assert_eq!(register(&mut stores, &complete, 9).unwrap(), root);
        assert_eq!(stores.states.len(), 2);
    }

    #[test]
    fn active_cancel_requested_and_paused_tasks_reject_prepare_and_commit() {
        for status in 0..3 {
            let mut stores = PalsStores::new(StoreLimits::default());
            let position = Position::startpos();
            let root = register(&mut stores, &position, 1).unwrap();
            let registration = stores
                .prepare_hot_root_registration(position.snapshot(), 2)
                .unwrap();
            let execution = model_task(&mut stores, root);
            match status {
                0 => {}
                1 => stores.tasks.cancel_consumer(execution, 1).unwrap(),
                _ => stores.tasks.pause(execution, 7, None).unwrap(),
            }
            assert!(matches!(
                stores.prepare_hot_root_registration(position.snapshot(), 2),
                Err(StoreError::InvalidConditions(
                    "registered root requires no active tasks"
                ))
            ));
            assert_eq!(
                stores.hot_focus_registered_root(registration),
                Err(StoreError::InvalidConditions(
                    "registered root requires no active tasks"
                ))
            );
            assert_eq!((stores.root(), stores.generation()), (Some(root), 1));
        }
    }

    #[test]
    fn review_pins_reject_prepare_and_late_commit_until_the_owner_releases_them() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = register(&mut stores, &position, 1).unwrap();
        let registration = stores
            .prepare_hot_root_registration(position.snapshot(), 2)
            .unwrap();
        let mut pins = StorePins::default();
        pins.situations.insert(root);
        let pin = stores.pin_review(pins).unwrap();
        assert!(matches!(
            stores.prepare_hot_root_registration(position.snapshot(), 2),
            Err(StoreError::InvalidConditions(
                "registered root requires no review pins"
            ))
        ));
        assert_eq!(
            stores.hot_focus_registered_root(registration),
            Err(StoreError::InvalidConditions(
                "registered root requires no review pins"
            ))
        );
        assert!(pin.validates(&stores));
        assert_eq!((stores.root(), stores.generation()), (Some(root), 1));
        drop(pin);
        assert_eq!(register(&mut stores, &position, 3).unwrap(), root);
    }

    #[test]
    fn completed_task_and_raw_observation_survive_registered_root_transition() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let mut position = Position::startpos();
        let root = register(&mut stores, &position, 1).unwrap();
        let execution = model_task(&mut stores, root);
        let situation = stores.situations.get(root).unwrap();
        let observation = stores
            .hot_append_observation(Observation {
                state: situation.state,
                line: Some(situation.focus),
                source: 1,
                epoch: 0,
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: None,
                cpu_pv: None,
                scope: EvidenceScope::Model {
                    model: 1,
                    encoding: 1,
                    input: 0,
                },
                score: RawScore::Unknown,
                budget: 1,
                kind: ObservationKind::Proposal,
                supersedes: None,
                execution: Some(execution),
            })
            .unwrap();
        stores.tasks.complete(execution, observation).unwrap();
        let raw = stores.observations.get(observation).unwrap().clone();
        position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let next = register(&mut stores, &position, 20).unwrap();
        assert_ne!(next, root);
        assert_eq!(
            stores.tasks.get(execution).unwrap().status,
            TaskStatus::Completed(observation)
        );
        assert_eq!(stores.observations.get(observation).unwrap(), &raw);
        assert_eq!(stores.tasks.len(), 1);
    }

    #[test]
    fn capacity_and_hot_allocation_errors_preserve_root_and_global_generation() {
        for limits in [
            StoreLimits {
                states: 1,
                ..StoreLimits::default()
            },
            StoreLimits {
                line_chunks: 1,
                ..StoreLimits::default()
            },
            StoreLimits {
                situations: 1,
                ..StoreLimits::default()
            },
        ] {
            let mut stores = PalsStores::new(limits);
            let mut position = Position::startpos();
            let root = register(&mut stores, &position, 5).unwrap();
            position
                .make_move(BoardMove::from_uci("e2e4").unwrap())
                .unwrap();
            let registration = stores
                .prepare_hot_root_registration(position.snapshot(), 99)
                .unwrap();
            assert!(matches!(
                stores.hot_focus_registered_root(registration),
                Err(StoreError::Capacity(_))
            ));
            assert_eq!((stores.root(), stores.generation()), (Some(root), 5));
        }
        let mut stores = PalsStores::new(StoreLimits::default());
        let mut position = Position::startpos();
        let root = register(&mut stores, &position, 5).unwrap();
        position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let registration = stores
            .prepare_hot_root_registration(position.snapshot(), 99)
            .unwrap();
        stores.states.snapshots.fail_next_reservations(1);
        assert_eq!(
            stores.hot_focus_registered_root(registration),
            Err(StoreError::Capacity("injected hot allocation"))
        );
        assert_eq!((stores.root(), stores.generation()), (Some(root), 5));
        assert_eq!(
            register(&mut stores, &position, 99).unwrap(),
            stores.root().unwrap()
        );
        assert_eq!(stores.generation(), 99);
    }

    #[test]
    fn line_allocation_failure_leaves_unpublished_state_but_not_new_authority() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let mut position = Position::startpos();
        let root = register(&mut stores, &position, 5).unwrap();
        position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let registration = stores
            .prepare_hot_root_registration(position.snapshot(), 99)
            .unwrap();
        stores.lines.chunks.fail_next_reservations(1);
        assert_eq!(
            stores.hot_focus_registered_root(registration),
            Err(StoreError::Capacity("injected hot allocation"))
        );
        assert_eq!((stores.root(), stores.generation()), (Some(root), 5));
        // Root publication is atomic. The reused hot allocator may nevertheless
        // retain its successfully inserted exact state before line allocation.
        assert!(stores.states.find(&position.snapshot()).is_some());
        assert_eq!(stores.states.len(), 2);
        assert_eq!(stores.situations.len(), 1);
        register(&mut stores, &position, 99).unwrap();
        assert_eq!(stores.states.len(), 2);
    }

    #[test]
    fn attested_retirement_path_only_fills_a_nonzero_vacant_generation() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        assert!(
            stores
                .prepare_hot_root_registration_after_retirement(position.snapshot(), 0)
                .is_err()
        );
        let root = register(&mut stores, &position, 3).unwrap();
        assert!(
            stores
                .prepare_hot_root_registration_after_retirement(position.snapshot(), 3)
                .is_err()
        );
        stores.root = None;
        stores.generation = 4;
        assert!(
            stores
                .prepare_hot_root_registration(position.snapshot(), 4)
                .is_err()
        );
        assert!(
            stores
                .prepare_hot_root_registration_after_retirement(position.snapshot(), 3)
                .is_err()
        );
        assert!(
            stores
                .prepare_hot_root_registration_after_retirement(position.snapshot(), 5)
                .is_err()
        );
        let registration = stores
            .prepare_hot_root_registration_after_retirement(position.snapshot(), 4)
            .unwrap();
        assert_eq!(
            stores.hot_focus_registered_root(registration).unwrap(),
            root
        );
        assert_eq!(stores.generation(), 4);
        assert!(
            stores
                .prepare_hot_root_registration_after_retirement(position.snapshot(), 4)
                .is_err()
        );
    }

    #[test]
    fn retirement_registration_revalidates_that_the_generation_is_still_vacant() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = register(&mut stores, &position, 3).unwrap();
        stores.root = None;
        stores.generation = 4;
        let registration = stores
            .prepare_hot_root_registration_after_retirement(position.snapshot(), 4)
            .unwrap();
        stores.root = Some(root);
        assert_eq!(
            stores.hot_focus_registered_root(registration),
            Err(StoreError::StaleConsumer)
        );
        assert_eq!((stores.root(), stores.generation()), (Some(root), 4));
    }

    #[test]
    fn registration_never_enters_archive_io_or_changes_an_allocation_scope_root() {
        let owned = TestArchiveRoot::new();
        let mut stores = PalsStores::new(StoreLimits::default());
        stores
            .enable_archive(ArchiveConfig::new(
                owned.root.join("archive"),
                std::env::current_dir().unwrap(),
            ))
            .unwrap();
        let mut position = Position::startpos();
        let archive_before = stores.archive_owner_snapshot().unwrap();
        let root = register(&mut stores, &position, 7).unwrap();
        assert_eq!(stores.archive_owner_snapshot().unwrap(), archive_before);
        position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let registration = stores
            .prepare_hot_root_registration(position.snapshot(), 20)
            .unwrap();
        let mut budget = ArchiveIoBudget {
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(10),
            max_bytes: 8 * 1024 * 1024,
        };
        let initial_bytes = budget.max_bytes;
        assert_eq!(
            stores.with_archive_allocation(StorePins::default(), &mut budget, |stores| {
                stores
                    .prepare_hot_root_registration(position.snapshot(), 20)
                    .map(|_| ())
            }),
            Err(StoreError::InvalidConditions(
                "archive allocation allowance is active"
            ))
        );
        assert_eq!(
            stores.with_archive_allocation(StorePins::default(), &mut budget, |stores| {
                stores.hot_focus_registered_root(registration)
            }),
            Err(StoreError::InvalidConditions(
                "archive allocation allowance is active"
            ))
        );
        assert_eq!(budget.max_bytes, initial_bytes);
        assert_eq!((stores.root(), stores.generation()), (Some(root), 7));
        let archive_before = stores.archive_owner_snapshot().unwrap();
        register(&mut stores, &position, 20).unwrap();
        assert_eq!(stores.archive_owner_snapshot().unwrap(), archive_before);
    }
}
