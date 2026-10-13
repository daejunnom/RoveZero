//! Hot-only single-writer commit for the adaptive manager.
//!
//! These methods preserve the ordinary store's exact-state, evidence, task and
//! conclusion checks, but never enter automatic reclamation or archive I/O.
//! Capacity/ColdRecord is a result for the coordinator to handle explicitly;
//! it does not authorize synchronous recovery, a retry, or relaxed validation.
//! The caller still owns logical admission, cancellation and physical fences.

use super::*;
mod registered_root;
pub use registered_root::HotRootRegistration;

impl PalsStores {
    /// Adopt a checked actual root without performing cold restoration. A
    /// failed generation advance or hot allocation leaves root authority intact.
    pub fn hot_focus_actual_moves(
        &mut self,
        checked_state: PositionSnapshot,
    ) -> Result<SituationId, StoreError> {
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        let matching = self.states.find(&checked_state).and_then(|state| {
            self.situations
                .iter()
                .find(|(_, situation)| situation.state == state)
                .map(|(id, _)| id)
        });
        let root = match matching {
            Some(id) => id,
            None => self.hot_insert_situation(checked_state)?,
        };
        self.root = Some(root);
        self.generation = next_generation;
        Ok(root)
    }

    pub fn hot_insert_situation(
        &mut self,
        snapshot: PositionSnapshot,
    ) -> Result<SituationId, StoreError> {
        self.insert_situation_once(snapshot)
    }

    /// Verify the entire immutable prefix and new suffix from its exact Rules
    /// anchor before publishing chunks. Numeric line/state IDs cannot replace
    /// this replay, and a failed replay publishes no partial line.
    pub fn hot_append_line(
        &mut self,
        prefix: LineId,
        moves: &[BoardMove],
    ) -> Result<LineId, StoreError> {
        let head = self.lines.get(prefix)?;
        head.plies
            .checked_add(moves.len())
            .filter(|plies| *plies <= self.limits.line_plies)
            .ok_or(StoreError::Capacity("line plies"))?;
        let snapshot = self.states.get(head.start_state)?;
        let mut replay = Position::from_snapshot(snapshot, rz_position::PositionLimits::default())
            .map_err(|_| StoreError::InvalidEvidence("hot line anchor cannot be restored"))?;
        for movement in self.lines.moves(prefix)?.iter().chain(moves) {
            replay
                .make_move(*movement)
                .map_err(|_| StoreError::InvalidEvidence("hot line failed exact Rules replay"))?;
        }
        self.lines.append(prefix, moves)
    }

    pub fn hot_append_cpu_pv(
        &mut self,
        position: &Position,
        moves: &[BoardMove],
    ) -> Result<LineId, StoreError> {
        self.append_cpu_pv_once(position, moves)
    }

    pub fn hot_append_observation(
        &mut self,
        observation: Observation,
    ) -> Result<ObservationId, StoreError> {
        // A foreign request's uniqueness may depend on archived observations.
        // Preserve the ordinary API's refusal rather than bypassing that scan.
        if self.archive_enabled() && observation.external_report.is_some() {
            return Err(StoreError::ArchiveBudgetRequired);
        }
        self.append_observation_once(observation)
    }

    pub fn hot_append_rules_terminal(
        &mut self,
        position: &Position,
        source: u64,
        epoch: u64,
    ) -> Result<ObservationId, StoreError> {
        self.append_rules_terminal_once(position, source, epoch)
    }

    pub fn hot_request_task(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
    ) -> Result<TaskAdmission, StoreError> {
        self.request_task_once(key, consumer, now_tick)
    }

    /// Retire only a checkpoint whose actual same-worker successor or confirmed
    /// cancellation has already been established by the manager. Evidence and
    /// historical checkpoint provenance survive; other paused owners do not
    /// participate in this local operation and archive I/O is never invoked.
    pub fn hot_retire_paused(&mut self, execution: ExecutionId) -> Result<(), StoreError> {
        let previous = self.tasks.get(execution)?;
        if previous.key.value_identity.is_none()
            || matches!(
                previous.key.checker_identity,
                Some(CheckerIdentity::ExternalUci(_))
            )
        {
            return Err(StoreError::InvalidConditions(
                "retirement requires an owned CPU checkpoint",
            ));
        }
        let TaskStatus::Paused {
            checkpoint,
            evidence,
        } = previous.status
        else {
            return Err(StoreError::InvalidConditions(
                "execution is not a live paused checkpoint",
            ));
        };
        let previous = self
            .tasks
            .tasks
            .get_mut(execution.0)
            .expect("checked hot paused checkpoint");
        previous.status = TaskStatus::RetiredPaused {
            checkpoint,
            evidence,
        };
        for consumer in &mut previous.consumers {
            consumer.cancelled = true;
        }
        Ok(())
    }

    /// Adaptive acceptance belongs to the immutable submitted CPU question,
    /// not a later conclusion revision written by an independent frame. The
    /// caller must already validate its original owner/job/view envelope.
    /// Consumer identity, Rules history, exact task namespace and deadline are
    /// unchanged, and no new consumer or acceptance authority is issued here.
    pub fn hot_resume_sealed_cpu_task(
        &mut self,
        previous: ExecutionId,
        original: TaskConsumer,
        expected_key: TaskKey,
        next_consumer_id: u64,
        exact_state: &PositionSnapshot,
        now_tick: u64,
    ) -> Result<TaskAdmission, StoreError> {
        let task = self.tasks.get(previous)?;
        if task.key != expected_key
            || task.key.value_identity.is_none()
            || matches!(
                task.key.checker_identity,
                Some(CheckerIdentity::ExternalUci(_))
            )
            || self.tasks.latest.get(&expected_key) != Some(&previous)
        {
            return Err(StoreError::InvalidConditions(
                "adaptive resume requires original owned CPU task",
            ));
        }
        let TaskStatus::Paused { evidence, .. } = task.status else {
            return Err(StoreError::InvalidConditions(
                "adaptive resume requires live paused task",
            ));
        };
        let record = task
            .consumers
            .iter()
            .find(|record| record.consumer.id == original.id)
            .ok_or(StoreError::InvalidHandle("consumer"))?;
        let granted = record.consumer;
        if granted.situation != original.situation
            || granted.revision != original.revision
            || granted.generation != original.generation
            || granted.deadline_tick != original.deadline_tick
        {
            return Err(StoreError::StaleConsumer);
        }
        if record.consumed {
            return Err(StoreError::AlreadyConsumed);
        }
        if record.cancelled {
            return Err(StoreError::CancelledConsumer);
        }
        if granted.deadline_tick <= now_tick {
            return Err(StoreError::ExpiredConsumer);
        }
        let situation = self
            .situations
            .get(original.situation)
            .map_err(|_| StoreError::StaleConsumer)?;
        if original.generation != self.generation
            || situation.state != expected_key.state
            || situation.revision < original.revision
            || !self.states.get(expected_key.state)?.same_state(exact_state)
        {
            return Err(StoreError::StaleConsumer);
        }
        if let Some(line) = expected_key.line {
            if self.lines.get(line)?.start_state != expected_key.state {
                return Err(StoreError::InvalidConditions("task line state"));
            }
        }
        if let Some(observation) = evidence {
            let partial = self.observations.get(observation)?;
            if partial.execution != Some(previous)
                || partial.state != expected_key.state
                || partial.line != expected_key.line
                || partial.value_identity != expected_key.value_identity
                || partial.checker_identity != expected_key.checker_identity
                || partial.cpu_condition != expected_key.cpu_condition
                || partial.epoch != expected_key.epoch
                || partial.budget > expected_key.node_budget
                || !matches!(partial.scope, EvidenceScope::DepthLimited { profile, condition, .. }
                    if profile == expected_key.profile && condition == expected_key.condition)
            {
                return Err(StoreError::InvalidEvidence(
                    "adaptive resume original partial evidence",
                ));
            }
            self.validate_observation_handles(partial)?;
        }
        // The new slice has a distinct consumer ID but exactly the original
        // situation/revision/generation/deadline. It does not receive current
        // conclusion authority, extend the deadline or replace the job state.
        self.tasks.request(
            expected_key,
            TaskConsumer {
                id: next_consumer_id,
                ..original
            },
            now_tick,
        )
    }

    /// Completion is revalidated exactly as in the legacy consumer. Only an
    /// independent conclusion revision may differ from the original seal.
    pub fn hot_consume_sealed_cpu_task(
        &mut self,
        execution: ExecutionId,
        original: TaskConsumer,
        expected_key: &TaskKey,
        exact_state: &PositionSnapshot,
        now_tick: u64,
    ) -> Result<ObservationId, StoreError> {
        let task = self.tasks.get(execution)?;
        if task.key != *expected_key
            || (task.key.value_identity.is_none()
                && !matches!(
                    task.key.checker_identity,
                    Some(CheckerIdentity::ExternalUci(_))
                ))
        {
            return Err(StoreError::InvalidConditions(
                "adaptive sealed consumer requires original CPU task",
            ));
        }
        let TaskStatus::Completed(observation) = task.status else {
            return Err(StoreError::NotCompleted);
        };
        self.validate_completed_task(execution, observation)?;
        let record = task
            .consumers
            .iter()
            .find(|record| record.consumer.id == original.id)
            .ok_or(StoreError::InvalidHandle("consumer"))?;
        let granted = record.consumer;
        if granted.situation != original.situation
            || granted.revision != original.revision
            || granted.generation != original.generation
            || granted.deadline_tick != original.deadline_tick
        {
            return Err(StoreError::StaleConsumer);
        }
        if record.consumed {
            return Err(StoreError::AlreadyConsumed);
        }
        if record.cancelled {
            return Err(StoreError::CancelledConsumer);
        }
        if granted.deadline_tick <= now_tick {
            return Err(StoreError::ExpiredConsumer);
        }
        let situation = self
            .situations
            .get(original.situation)
            .map_err(|_| StoreError::StaleConsumer)?;
        if original.generation != self.generation
            || situation.state != expected_key.state
            || situation.revision < original.revision
            || !self.states.get(expected_key.state)?.same_state(exact_state)
        {
            return Err(StoreError::StaleConsumer);
        }
        let raw = self.observations.get(observation)?;
        if raw.execution != Some(execution)
            || raw.state != expected_key.state
            || raw.line != expected_key.line
            || raw.value_identity != expected_key.value_identity
            || raw.checker_identity != expected_key.checker_identity
            || raw.cpu_condition != expected_key.cpu_condition
        {
            return Err(StoreError::InvalidEvidence(
                "adaptive sealed CPU completion namespace",
            ));
        }
        let task = self
            .tasks
            .tasks
            .get_mut(execution.0)
            .expect("checked hot CPU completion");
        let record = task
            .consumers
            .iter_mut()
            .find(|record| record.consumer.id == original.id)
            .expect("checked original consumer");
        record.consumed = true;
        Ok(observation)
    }

    /// A yielded own CPU result may be read without spending the original
    /// checkpoint's resume authority. The caller seals actual receipt use; the
    /// retained task remains Paused and its live consumer remains unconsumed.
    pub fn hot_validate_sealed_paused_cpu_feedback(
        &self,
        execution: ExecutionId,
        original: TaskConsumer,
        expected_key: &TaskKey,
        exact_state: &PositionSnapshot,
        observation: ObservationId,
        now_tick: u64,
    ) -> Result<ObservationId, StoreError> {
        let task = self.tasks.get(execution)?;
        if !matches!(task.status, TaskStatus::Paused { evidence: Some(id), .. } if id == observation)
        {
            return Err(StoreError::InvalidConditions(
                "CPU feedback lacks original paused evidence",
            ));
        }
        self.validate_sealed_feedback_consumer(
            execution,
            original,
            expected_key,
            exact_state,
            now_tick,
            false,
        )?;
        self.validate_owned_partial_feedback(execution, expected_key, observation)?;
        Ok(observation)
    }

    /// Terminal incomplete own CPU feedback is one-use guidance only. Failed
    /// status remains Failed; this does not create a completed-task reuse hit.
    pub fn hot_consume_sealed_cpu_feedback(
        &mut self,
        execution: ExecutionId,
        original: TaskConsumer,
        expected_key: &TaskKey,
        exact_state: &PositionSnapshot,
        observation: ObservationId,
        now_tick: u64,
    ) -> Result<ObservationId, StoreError> {
        if self.tasks.get(execution)?.status != TaskStatus::Failed {
            return Err(StoreError::InvalidConditions(
                "CPU partial feedback is not terminal failed",
            ));
        }
        self.validate_sealed_feedback_consumer(
            execution,
            original,
            expected_key,
            exact_state,
            now_tick,
            false,
        )?;
        self.validate_owned_partial_feedback(execution, expected_key, observation)?;
        self.mark_sealed_feedback_consumed(execution, original.id);
        Ok(observation)
    }

    /// External bestmove/stop-yield feedback may contain unknown or overshot
    /// reported nodes. It remains a raw external observation and cannot become
    /// native credit, a completed search bound or a completed-task reuse hit.
    pub fn hot_consume_sealed_foreign_feedback(
        &mut self,
        execution: ExecutionId,
        original: TaskConsumer,
        expected_key: &TaskKey,
        exact_state: &PositionSnapshot,
        observation: ObservationId,
        now_tick: u64,
    ) -> Result<ObservationId, StoreError> {
        self.validate_foreign_feedback(execution, expected_key, observation)?;
        self.validate_sealed_feedback_consumer(
            execution,
            original,
            expected_key,
            exact_state,
            now_tick,
            false,
        )?;
        self.mark_sealed_feedback_consumed(execution, original.id);
        Ok(observation)
    }

    /// Recheck a previously accepted foreign receipt before a semantic
    /// continuation. Consumption does not survive an explicit revoke, a root
    /// generation change or the original deadline. This only reads authority;
    /// it cannot consume twice or upgrade a partial result to completed work.
    /// The port must first match its private consumed-observation receipt seal:
    /// Failed task records retain consumer consumption, not an observation ID.
    pub fn hot_validate_consumed_foreign_feedback(
        &self,
        execution: ExecutionId,
        original: TaskConsumer,
        expected_key: &TaskKey,
        exact_state: &PositionSnapshot,
        observation: ObservationId,
        now_tick: u64,
    ) -> Result<ObservationId, StoreError> {
        self.validate_sealed_feedback_consumer(
            execution,
            original,
            expected_key,
            exact_state,
            now_tick,
            true,
        )?;
        self.validate_foreign_feedback(execution, expected_key, observation)?;
        Ok(observation)
    }

    /// Recheck actual own CPU feedback without changing its consumer. A
    /// terminal receipt must already be consumed. A retained paused receipt
    /// instead keeps the original unconsumed resume authority and may only be
    /// used while that same paused task and consumer remain live.
    /// The port must first match its private consumed-observation receipt seal;
    /// this getter rechecks Store authority and does not issue that seal.
    pub fn hot_validate_consumed_cpu_feedback(
        &self,
        execution: ExecutionId,
        original: TaskConsumer,
        expected_key: &TaskKey,
        exact_state: &PositionSnapshot,
        observation: ObservationId,
        now_tick: u64,
    ) -> Result<ObservationId, StoreError> {
        match self.tasks.get(execution)?.status {
            TaskStatus::Paused { .. } => self.hot_validate_sealed_paused_cpu_feedback(
                execution,
                original,
                expected_key,
                exact_state,
                observation,
                now_tick,
            ),
            TaskStatus::Completed(id) if id == observation => {
                self.validate_completed_task(execution, observation)?;
                self.validate_sealed_feedback_consumer(
                    execution,
                    original,
                    expected_key,
                    exact_state,
                    now_tick,
                    true,
                )?;
                self.validate_owned_partial_feedback(execution, expected_key, observation)?;
                Ok(observation)
            }
            TaskStatus::Failed => {
                self.validate_sealed_feedback_consumer(
                    execution,
                    original,
                    expected_key,
                    exact_state,
                    now_tick,
                    true,
                )?;
                self.validate_owned_partial_feedback(execution, expected_key, observation)?;
                Ok(observation)
            }
            _ => Err(StoreError::InvalidConditions(
                "accepted CPU feedback is not current settled evidence",
            )),
        }
    }

    fn validate_foreign_feedback(
        &self,
        execution: ExecutionId,
        expected_key: &TaskKey,
        observation: ObservationId,
    ) -> Result<(), StoreError> {
        let task = self.tasks.get(execution)?;
        if !matches!(task.status, TaskStatus::Failed | TaskStatus::Completed(_)) {
            return Err(StoreError::InvalidConditions(
                "foreign feedback is not physically settled",
            ));
        }
        if let TaskStatus::Completed(id) = task.status {
            if id != observation {
                return Err(StoreError::InvalidEvidence(
                    "foreign completed observation differs",
                ));
            }
            self.validate_completed_task(execution, observation)?;
        }
        let raw = self.observations.get(observation)?;
        self.validate_sealed_feedback_raw(execution, expected_key, raw)?;
        let report = raw
            .external_report
            .as_deref()
            .ok_or(StoreError::InvalidEvidence("foreign feedback lacks report"))?;
        if expected_key.value_identity.is_some()
            || !matches!(
                expected_key.checker_identity,
                Some(CheckerIdentity::ExternalUci(_))
            )
            || raw.kind != ObservationKind::ExternalCpuAnalysis
            || report.request_id == 0
            || report.requested_depth != expected_key.requested_depth
            || !matches!(
                report.completion,
                ExternalCompletion::BestMove | ExternalCompletion::StoppedYield
            )
            || report.best_move.is_none()
            || !matches!(raw.scope, EvidenceScope::ExternalUci { requested_depth, .. } if requested_depth == expected_key.requested_depth)
        {
            return Err(StoreError::InvalidEvidence(
                "foreign partial feedback scope",
            ));
        }
        self.validate_observation_handles(raw)
    }

    fn validate_owned_partial_feedback(
        &self,
        execution: ExecutionId,
        key: &TaskKey,
        observation: ObservationId,
    ) -> Result<(), StoreError> {
        let raw = self.observations.get(observation)?;
        self.validate_sealed_feedback_raw(execution, key, raw)?;
        if key.value_identity.is_none()
            || matches!(key.checker_identity, Some(CheckerIdentity::ExternalUci(_)))
            || raw.kind != ObservationKind::CpuAnalysis
            || raw.budget > key.node_budget
            || !matches!(raw.scope, EvidenceScope::DepthLimited { profile, condition, .. }
                if profile == key.profile && condition == key.condition)
        {
            return Err(StoreError::InvalidEvidence(
                "owned partial CPU feedback scope",
            ));
        }
        self.validate_observation_handles(raw)
    }

    fn validate_sealed_feedback_raw(
        &self,
        execution: ExecutionId,
        key: &TaskKey,
        raw: &Observation,
    ) -> Result<(), StoreError> {
        if raw.execution != Some(execution)
            || raw.state != key.state
            || raw.line != key.line
            || raw.value_identity != key.value_identity
            || raw.checker_identity != key.checker_identity
            || raw.cpu_condition != key.cpu_condition
            || raw.epoch != key.epoch
        {
            return Err(StoreError::InvalidEvidence(
                "partial feedback original namespace",
            ));
        }
        Ok(())
    }

    fn validate_sealed_feedback_consumer(
        &self,
        execution: ExecutionId,
        original: TaskConsumer,
        key: &TaskKey,
        exact_state: &PositionSnapshot,
        now_tick: u64,
        expected_consumed: bool,
    ) -> Result<(), StoreError> {
        let task = self.tasks.get(execution)?;
        if task.key != *key {
            return Err(StoreError::InvalidConditions(
                "partial feedback original TaskKey",
            ));
        }
        let record = task
            .consumers
            .iter()
            .find(|record| record.consumer.id == original.id)
            .ok_or(StoreError::InvalidHandle("consumer"))?;
        if record.consumer != original {
            return Err(StoreError::StaleConsumer);
        }
        if record.consumed && !expected_consumed {
            return Err(StoreError::AlreadyConsumed);
        }
        if !record.consumed && expected_consumed {
            return Err(StoreError::InvalidEvidence(
                "feedback consumer has not been consumed",
            ));
        }
        if record.cancelled {
            return Err(StoreError::CancelledConsumer);
        }
        if original.deadline_tick <= now_tick {
            return Err(StoreError::ExpiredConsumer);
        }
        let situation = self
            .situations
            .get(original.situation)
            .map_err(|_| StoreError::StaleConsumer)?;
        if original.generation != self.generation
            || situation.state != key.state
            || situation.revision < original.revision
            || !self.states.get(key.state)?.same_state(exact_state)
        {
            return Err(StoreError::StaleConsumer);
        }
        if let Some(line) = key.line {
            if self.lines.get(line)?.start_state != key.state {
                return Err(StoreError::InvalidConditions("partial feedback line state"));
            }
        }
        Ok(())
    }

    fn mark_sealed_feedback_consumed(&mut self, execution: ExecutionId, consumer_id: u64) {
        self.tasks
            .tasks
            .get_mut(execution.0)
            .expect("checked partial task")
            .consumers
            .iter_mut()
            .find(|record| record.consumer.id == consumer_id)
            .expect("checked partial consumer")
            .consumed = true;
    }

    pub fn hot_add_dependency(
        &mut self,
        observation: ObservationId,
        situation: SituationId,
    ) -> Result<(), StoreError> {
        self.observations.get(observation)?;
        self.situations.get(situation)?;
        self.dependencies.add(observation, situation)
    }

    pub fn hot_refute_continuation(
        &mut self,
        situation: SituationId,
        line: LineId,
        evidence: ObservationId,
    ) -> Result<(), StoreError> {
        let revision = self.refute_continuation_once(situation, line, evidence)?;
        let current = self.situations.get_mut(situation)?;
        current.revision = revision;
        current.dirty = true;
        current.conclusions.continuations.insert(
            line,
            ContinuationConclusion {
                status: ContinuationStatus::Refuted,
                evidence: Some(evidence),
                revision,
            },
        );
        Ok(())
    }

    pub fn hot_repair(
        &mut self,
        situation: SituationId,
        refuted_line: LineId,
        repaired_line: LineId,
        evidence: ObservationId,
    ) -> Result<(), StoreError> {
        let revision = self.repair_once(situation, refuted_line, repaired_line, evidence)?;
        let current = self.situations.get_mut(situation)?;
        current.revision = revision;
        current.dirty = true;
        current.conclusions.continuations.insert(
            refuted_line,
            ContinuationConclusion {
                status: ContinuationStatus::RepairedBy(repaired_line),
                evidence: Some(evidence),
                revision,
            },
        );
        current.conclusions.continuations.insert(
            repaired_line,
            ContinuationConclusion {
                status: ContinuationStatus::Supported,
                evidence: Some(evidence),
                revision,
            },
        );
        Ok(())
    }
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
                "rz-pals-hot-commit-tests-{}-{serial}",
                std::process::id()
            ));
            std::fs::create_dir(&root).unwrap();
            Self { root, parent }
        }
    }
    impl Drop for TestArchiveRoot {
        fn drop(&mut self) {
            // Only this unique, non-link child of the resolved temporary root
            // is eligible for cleanup. No shared cache or archive is touched.
            if self.root.parent() == Some(self.parent.as_path())
                && self.root.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rz-pals-hot-commit-tests-")
                })
                && std::fs::symlink_metadata(&self.root)
                    .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                let _ = std::fs::remove_dir_all(&self.root);
            }
        }
    }

    fn movement(text: &str) -> BoardMove {
        BoardMove::from_uci(text).unwrap()
    }

    #[test]
    fn hot_line_replays_prefix_and_rejects_wrong_side_before_publication() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores
            .hot_insert_situation(Position::startpos().snapshot())
            .unwrap();
        let root = stores.situations.get(situation).unwrap().focus;
        let e4 = stores.hot_append_line(root, &[movement("e2e4")]).unwrap();
        let before = stores.lines.len();
        assert!(matches!(
            stores.hot_append_line(e4, &[movement("d2d4")]),
            Err(StoreError::InvalidEvidence(
                "hot line failed exact Rules replay"
            ))
        ));
        assert_eq!(stores.lines.len(), before);
        let e5 = stores.hot_append_line(e4, &[movement("e7e5")]).unwrap();
        assert_eq!(
            stores.lines.moves(e5).unwrap(),
            [movement("e2e4"), movement("e7e5")]
        );
    }

    #[test]
    fn hot_cpu_pv_requires_exact_history_and_preserves_capacity_error() {
        let position = Position::startpos();
        let mut stores = PalsStores::new(StoreLimits {
            line_plies: 1,
            ..StoreLimits::default()
        });
        stores.hot_insert_situation(position.snapshot()).unwrap();
        let imported = Position::from_fen(&position.to_fen()).unwrap();
        let before = stores.hot_stats();
        assert!(matches!(
            stores.hot_append_cpu_pv(&imported, &[movement("e2e4")]),
            Err(StoreError::InvalidEvidence(
                "CPU PV starts at an unregistered exact state"
            ))
        ));
        assert_eq!(stores.hot_stats(), before);
        assert_eq!(
            stores.hot_append_cpu_pv(&position, &[movement("e2e4"), movement("e7e5")]),
            Err(StoreError::Capacity("CPU PV plies"))
        );
        assert_eq!(stores.hot_stats(), before);
    }

    #[test]
    fn hot_terminal_uses_rules_and_never_issues_a_nonterminal_fact() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let before = stores.hot_stats();
        assert_eq!(
            stores.hot_append_rules_terminal(&Position::startpos(), 1, 0),
            Err(StoreError::InvalidEvidence(
                "Rules position is not terminal"
            ))
        );
        assert_eq!(stores.hot_stats(), before);
        let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
        let id = stores.hot_append_rules_terminal(&mate, 1, 0).unwrap();
        assert_eq!(
            stores.observations.get(id).unwrap().scope,
            EvidenceScope::RulesTerminal
        );
        assert_eq!(
            stores.observations.get(id).unwrap().score,
            RawScore::Terminal {
                winner: Some(Color::White)
            }
        );
    }

    #[test]
    fn hot_root_focus_preserves_authority_when_capacity_or_generation_exhausts() {
        let mut stores = PalsStores::new(StoreLimits {
            states: 1,
            ..StoreLimits::default()
        });
        let mut position = Position::startpos();
        let root = stores.hot_focus_actual_moves(position.snapshot()).unwrap();
        let generation = stores.generation();
        let before = stores.hot_stats();
        assert_eq!(
            stores.hot_focus_actual_moves(position.snapshot()).unwrap(),
            root
        );
        assert_eq!(stores.generation(), generation + 1);
        position.make_uci("e2e4").unwrap();
        let generation = stores.generation();
        assert_eq!(
            stores.hot_focus_actual_moves(position.snapshot()),
            Err(StoreError::Capacity("states"))
        );
        assert_eq!(stores.root(), Some(root));
        assert_eq!(stores.generation(), generation);
        assert_eq!(stores.hot_stats(), before);
        stores.generation = u64::MAX;
        assert_eq!(
            stores.hot_focus_actual_moves(Position::startpos().snapshot()),
            Err(StoreError::RevisionExhausted)
        );
        assert_eq!(stores.root(), Some(root));
        assert_eq!(stores.generation(), u64::MAX);
        assert_eq!(stores.hot_stats(), before);
    }

    #[test]
    fn hot_pressure_keeps_archive_idle_and_does_not_retry_allocation() {
        let owned = TestArchiveRoot::new();
        let mut stores = PalsStores::new(StoreLimits {
            states: 4,
            situations: 4,
            ..StoreLimits::default()
        });
        stores
            .enable_archive(ArchiveConfig::new(
                owned.root.join("archive"),
                std::env::current_dir().unwrap(),
            ))
            .unwrap();
        let mut position = Position::startpos();
        for text in ["e2e4", "e7e5", "g1f3"] {
            stores.hot_focus_actual_moves(position.snapshot()).unwrap();
            position.make_uci(text).unwrap();
        }
        stores
            .set_archive_io_budget(ArchiveIoBudget {
                deadline: std::time::Instant::now() + std::time::Duration::from_secs(10),
                max_bytes: 8 * 1024 * 1024,
            })
            .unwrap();
        let before = stores.archive_owner_snapshot().unwrap();
        let root = stores.root();
        let generation = stores.generation();
        let hot = stores.hot_stats();
        stores.states.snapshots.fail_next_reservations(1);
        assert_eq!(
            stores.hot_focus_actual_moves(position.snapshot()),
            Err(StoreError::Capacity("injected hot allocation"))
        );
        assert_eq!(stores.archive_owner_snapshot().unwrap(), before);
        assert_eq!(stores.hot_stats(), hot);
        assert_eq!(stores.root(), root);
        assert_eq!(stores.generation(), generation);
        // One later explicit attempt succeeds because no hidden retry consumed
        // the injection or archived an inactive state on the failed call.
        stores.hot_focus_actual_moves(position.snapshot()).unwrap();
        assert_eq!(stores.archive_owner_snapshot().unwrap(), before);
        assert_eq!(stores.states.len(), 4);
    }

    #[test]
    fn hot_retirement_preserves_previous_evidence_and_successor_authority() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = stores.hot_focus_actual_moves(position.snapshot()).unwrap();
        let situation = stores.situations.get(root).unwrap().clone();
        let cpu = crate::cpu::CpuEngine::new(crate::cpu::CpuConfig {
            tt_entries: 16,
            ..crate::cpu::CpuConfig::default()
        })
        .unwrap();
        let identity = cpu.value_identity().clone();
        let key = TaskKey {
            state: situation.state,
            line: None,
            question: TaskQuestion::AnalyzePosition,
            root_moves: Vec::new(),
            model: 0,
            epoch: 0,
            value_identity: Some(identity.clone()),
            checker_identity: None,
            cpu_condition: Some("hot-retirement-fixture".into()),
            profile: 1,
            condition: 2,
            input_revision: 0,
            requested_depth: 2,
            node_budget: 100,
        };
        let consumer = TaskConsumer {
            id: 1,
            situation: root,
            revision: situation.revision,
            generation: stores.generation(),
            deadline_tick: 100,
        };
        let TaskAdmission::Start(previous) =
            stores.hot_request_task(key.clone(), consumer, 0).unwrap()
        else {
            panic!("fresh task");
        };
        let observation = stores
            .hot_append_observation(Observation {
                state: situation.state,
                line: None,
                source: 1,
                epoch: 0,
                value_identity: Some(identity),
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: key.cpu_condition.clone(),
                cpu_pv: None,
                scope: EvidenceScope::DepthLimited {
                    depth: 1,
                    profile: 1,
                    condition: 2,
                },
                score: RawScore::Cpu {
                    value: 0,
                    perspective: Color::White,
                    bound: BoundKind::ExactWithinSearch,
                },
                budget: 3,
                kind: ObservationKind::CpuAnalysis,
                supersedes: None,
                execution: Some(previous),
            })
            .unwrap();
        stores.pause_task(previous, 7, Some(observation)).unwrap();
        let evidence = stores.observations.get(observation).unwrap().clone();
        let conclusion = stores
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
                    input: 0,
                    encoding: 1,
                },
                score: RawScore::Estimate {
                    value: -1.0,
                    perspective: Color::White,
                },
                budget: 0,
                kind: ObservationKind::Refutation,
                supersedes: None,
                execution: None,
            })
            .unwrap();
        stores
            .hot_refute_continuation(root, situation.focus, conclusion)
            .unwrap();
        assert_eq!(
            stores.hot_request_task(key.clone(), TaskConsumer { id: 2, ..consumer }, 0),
            Err(StoreError::StaleConsumer)
        );
        let tasks = stores.tasks.len();
        assert_eq!(
            stores.hot_resume_sealed_cpu_task(
                previous,
                consumer,
                key.clone(),
                2,
                &Position::from_fen(&position.to_fen()).unwrap().snapshot(),
                0
            ),
            Err(StoreError::StaleConsumer)
        );
        assert_eq!(
            stores.hot_resume_sealed_cpu_task(
                previous,
                consumer,
                key.clone(),
                2,
                &position.snapshot(),
                100
            ),
            Err(StoreError::ExpiredConsumer)
        );
        assert_eq!(stores.tasks.len(), tasks);
        let TaskAdmission::Resume {
            execution: current,
            previous: actual_previous,
            checkpoint: 7,
        } = stores
            .hot_resume_sealed_cpu_task(previous, consumer, key, 2, &position.snapshot(), 0)
            .unwrap()
        else {
            panic!("resume task");
        };
        assert_eq!(actual_previous, previous);
        let resumed_consumer = stores.tasks.get(current).unwrap().consumers[0].consumer;
        assert_eq!(resumed_consumer, TaskConsumer { id: 2, ..consumer });
        stores.hot_retire_paused(previous).unwrap();
        assert_eq!(
            stores.tasks.get(previous).unwrap().status,
            TaskStatus::RetiredPaused {
                checkpoint: 7,
                evidence: Some(observation)
            }
        );
        assert!(
            stores
                .tasks
                .get(previous)
                .unwrap()
                .consumers
                .iter()
                .all(|consumer| consumer.cancelled)
        );
        assert_eq!(
            stores.tasks.get(current).unwrap().status,
            TaskStatus::InFlight
        );
        assert_eq!(stores.observations.get(observation).unwrap(), &evidence);
        assert_eq!(stores.tasks.active_states().count(), 1);
        assert!(stores.hot_retire_paused(previous).is_err());
        assert!(stores.hot_retire_paused(current).is_err());
    }

    #[test]
    fn sealed_cpu_consume_revalidates_low_level_completion_before_consumption() {
        // Low-level tables remain inspectable. A forged completion must not
        // bypass the same evidence checks used by the ordinary V1 consumer.
        for invalid in 0..5 {
            let mut stores = PalsStores::new(StoreLimits::default());
            let position = Position::startpos();
            let root = stores.hot_focus_actual_moves(position.snapshot()).unwrap();
            let situation = stores.situations.get(root).unwrap().clone();
            let cpu = crate::cpu::CpuEngine::new(crate::cpu::CpuConfig {
                tt_entries: 16,
                ..crate::cpu::CpuConfig::default()
            })
            .unwrap();
            let identity = cpu.value_identity().clone();
            let key = TaskKey {
                state: situation.state,
                line: None,
                question: TaskQuestion::AnalyzePosition,
                root_moves: Vec::new(),
                model: 0,
                epoch: 0,
                value_identity: Some(identity.clone()),
                checker_identity: None,
                cpu_condition: Some("sealed-consume-fixture".into()),
                profile: 1,
                condition: 2,
                input_revision: 0,
                requested_depth: 2,
                node_budget: 100,
            };
            let consumer = TaskConsumer {
                id: 1,
                situation: root,
                revision: situation.revision,
                generation: stores.generation(),
                deadline_tick: 100,
            };
            let TaskAdmission::Start(execution) =
                stores.hot_request_task(key.clone(), consumer, 0).unwrap()
            else {
                panic!("fresh task");
            };
            let (epoch, budget, depth, profile, condition) = match invalid {
                0 => (1, 3, 2, 1, 2),
                1 => (0, 101, 2, 1, 2),
                2 => (0, 3, 1, 1, 2),
                3 => (0, 3, 2, 3, 2),
                _ => (0, 3, 2, 1, 4),
            };
            let observation = stores
                .hot_append_observation(Observation {
                    state: situation.state,
                    line: None,
                    source: 1,
                    epoch,
                    value_identity: Some(identity),
                    checker_identity: None,
                    checker_work: None,
                    external_report: None,
                    model_value_identity: None,
                    model_value_input: None,
                    cpu_condition: key.cpu_condition.clone(),
                    cpu_pv: None,
                    scope: EvidenceScope::DepthLimited {
                        depth,
                        profile,
                        condition,
                    },
                    score: RawScore::Cpu {
                        value: 0,
                        perspective: Color::White,
                        bound: BoundKind::ExactWithinSearch,
                    },
                    budget,
                    kind: ObservationKind::CpuAnalysis,
                    supersedes: None,
                    execution: Some(execution),
                })
                .unwrap();
            stores.tasks.complete(execution, observation).unwrap();
            assert!(matches!(
                stores.hot_consume_sealed_cpu_task(
                    execution,
                    consumer,
                    &key,
                    &position.snapshot(),
                    1
                ),
                Err(StoreError::InvalidEvidence(_))
            ));
            assert!(!stores.tasks.get(execution).unwrap().consumers[0].consumed);
        }
    }
}
