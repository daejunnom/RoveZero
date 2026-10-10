//! Engine-owned topology crosses the archive commit barrier before its RAM is
//! released. Store reads remain hot-only; cold restoration is an explicit call.

use super::super::store::{
    ArchiveConfig, ArchiveIoBudget, EngineArchiveNode, EngineArchiveReceipt, StorePins,
};
use super::*;
use crate::cpu_value::CpuTrainingState;
use std::collections::{BTreeMap, BTreeSet};

const SEARCH_ARCHIVE_IO_BYTES: u64 = 16 * 1024 * 1024;
const NODE_METADATA_BYTES: usize = 4096;
const NODE_METADATA_MAGIC: &[u8; 8] = b"RZENGN01";

/// The directory/configuration is retained across ucinewgame; each new Store
/// owns another game directory and any reopen failure remains a typed error.
#[derive(Default)]
pub(super) struct EngineArchiveLifecycle {
    config: Option<ArchiveConfig>,
    reopen_error: Option<StoreError>,
    deadline: Option<Instant>,
    remaining_io: u64,
    retried_allocation: bool,
    last_receipt: Option<EngineArchiveReceipt>,
    active_search_pins: StorePins,
}

impl<M: RoleModel> PalsEngine<M> {
    /// Explicit opt-in. Paths and the 256MiB/game, 4GiB/global ceilings are
    /// validated by the Store; no game evidence is deleted to satisfy them.
    pub fn enable_archive(&mut self, config: ArchiveConfig) -> Result<(), PalsError> {
        self.stores
            .enable_archive(config.clone())
            .map_err(PalsError::Store)?;
        self.archive_state.config = Some(config);
        self.archive_state.reopen_error = None;
        Ok(())
    }

    pub fn archive_config(&self) -> Option<&ArchiveConfig> {
        self.archive_state.config.as_ref()
    }

    pub fn archive_enabled(&self) -> bool {
        self.stores.archive_enabled()
    }

    /// Bounded receipt of the most recent engine metadata commit, rather than
    /// an unbounded in-memory directory of all historical engine records.
    pub fn last_engine_archive(&self) -> Option<&EngineArchiveReceipt> {
        self.archive_state.last_receipt.as_ref()
    }

    pub(super) fn reset_archive_new_game(&mut self) {
        self.archive_state.deadline = None;
        self.archive_state.remaining_io = 0;
        self.archive_state.retried_allocation = false;
        self.archive_state.last_receipt = None;
        self.archive_state.active_search_pins = StorePins::default();
        self.archive_state.reopen_error = self
            .archive_state
            .config
            .as_ref()
            .and_then(|config| self.stores.enable_archive(config.clone()).err());
    }

    /// Called before focus_actual_moves, while the previous Node indices are
    /// still coherent. This never enables Store's implicit I/O allowance.
    pub(super) fn prepare_archive_root(
        &mut self,
        position: &Position,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        if let Some(error) = &self.archive_state.reopen_error {
            return Err(PalsError::Store(error.clone()));
        }
        if !self.stores.archive_enabled() {
            return Ok(());
        }
        archive_controls(deadline, cancel)?;
        self.archive_state.deadline = Some(deadline);
        self.archive_state.remaining_io = SEARCH_ARCHIVE_IO_BYTES;
        self.archive_state.retried_allocation = false;
        self.archive_state.active_search_pins = StorePins::default();
        self.retire_stale_archive_checkpoints(position)?;
        self.stores
            .set_archive_pins(self.resident_archive_pins()?)?;
        if self.engine_archive_pressure(position) {
            self.reclaim_archive_for_root(position, deadline, cancel)?;
        }
        Ok(())
    }

    /// Exactly one retry is available to the caller after a failed actual-root
    /// focus/intern allocation. Failure to reclaim is an explicit Store error.
    pub(super) fn retry_archive_root_allocation(
        &mut self,
        position: &Position,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        if !self.stores.archive_enabled() {
            return Err(PalsError::Capacity);
        }
        if self.archive_state.retried_allocation {
            return Err(StoreError::PinSaturated("actual root allocation retry exhausted").into());
        }
        self.archive_state.retried_allocation = true;
        self.retire_stale_archive_checkpoints(position)?;
        self.reclaim_archive_for_root(position, deadline, cancel)
    }

    /// Called after the new actual root was interned, before any local traversal
    /// starts. It also removes a former Store root that was temporarily pinned
    /// during preparation. Returned indices are current after compaction.
    pub(super) fn finish_archive_root(
        &mut self,
        root: usize,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<usize, PalsError> {
        if !self.stores.archive_enabled() {
            return Ok(root);
        }
        archive_controls(deadline, cancel)?;
        let state = self
            .nodes
            .get(root)
            .ok_or(StoreError::InvalidHandle("engine actual root"))?
            .state;
        self.stores
            .set_archive_pins(self.resident_archive_pins()?)?;
        let position = self.nodes[root].position.clone();
        if self.engine_archive_pressure(&position) {
            self.reclaim_archive_for_root(&position, deadline, cancel)?;
        }
        self.nodes
            .iter()
            .position(|node| node.state == state)
            .ok_or_else(|| StoreError::ArchiveIntegrity("actual root removed by archive").into())
    }

    /// The existing current-input projection may remove an obsolete record at
    /// its finite cap. Preserve its origin and relationships before that trim.
    pub(super) fn archive_public_record_before_remove(
        &mut self,
        index: usize,
    ) -> Result<(), PalsError> {
        if !self.stores.archive_enabled() {
            return Ok(());
        }
        let record = self
            .records
            .get(index)
            .ok_or(StoreError::InvalidHandle("engine obsolete record"))?
            .clone();
        let mut budget = self.engine_archive_budget()?;
        let initial = budget.max_bytes;
        let result = self.stores.archive_engine_records_with_pins_and_budget(
            &[record],
            &[],
            StorePins::default(),
            &mut budget,
        );
        self.charge_engine_archive_io(initial - budget.max_bytes)?;
        let receipt = result.map_err(PalsError::Store)?;
        self.archive_state.last_receipt = Some(receipt);
        Ok(())
    }

    /// Foreign request IDs are unique across retained hot and cold reports.
    /// The explicitly chosen archive path shares this search's I/O allowance;
    /// a rejected duplicate consumes the actual scan work as well.
    #[cfg(test)]
    pub(super) fn append_engine_observation(
        &mut self,
        observation: Observation,
    ) -> Result<ObservationId, PalsError> {
        self.append_engine_observation_with_controls(observation, None)
    }

    pub(super) fn append_engine_observation_with_controls(
        &mut self,
        observation: Observation,
        controls: Option<(Instant, &AtomicBool)>,
    ) -> Result<ObservationId, PalsError> {
        if self.stores.archive_enabled() && observation.external_report.is_some() {
            self.stores
                .set_archive_pins(self.resident_archive_pins()?)?;
            let mut budget = self.engine_archive_budget()?;
            let initial = budget.max_bytes;
            let result = if controls.is_some_and(|(deadline, cancel)| {
                cancel.load(Ordering::Acquire) || Instant::now() >= deadline
            }) {
                self.stores
                    .append_observation_checked_first_with_archive_budget(observation, &mut budget)
            } else {
                self.stores
                    .append_observation_checked_with_archive_budget_controlled(
                        observation,
                        &mut budget,
                        || allocation_controls(controls),
                    )
            };
            let used =
                initial
                    .checked_sub(budget.max_bytes)
                    .ok_or(StoreError::ArchiveIntegrity(
                        "engine foreign scan accounting",
                    ))?;
            self.charge_engine_archive_io(used)?;
            result.map_err(PalsError::Store)
        } else {
            if controls.is_none() {
                return self
                    .stores
                    .append_observation(observation)
                    .map_err(PalsError::from);
            }
            // A completed late output remains an immutable raw fact if one hot
            // append can admit it. It receives no reclaim/retry or acceptance.
            if controls.is_some_and(|(deadline, cancel)| {
                cancel.load(Ordering::Acquire) || Instant::now() >= deadline
            }) {
                let result = self.stores.append_observation(observation);
                return if self.stores.archive_enabled() {
                    result.map_err(PalsError::Store)
                } else {
                    result.map_err(PalsError::from)
                };
            }
            let controls = controls.filter(|_| self.stores.archive_enabled());
            let mut pins = StorePins::default();
            pins.states.insert(observation.state);
            pins.lines.extend(observation.line);
            pins.lines.extend(observation.cpu_pv);
            pins.observations.extend(observation.supersedes);
            pins.executions.extend(observation.execution);
            match observation.score {
                RawScore::ConditionalWdl {
                    repaired, counter, ..
                } => {
                    pins.observations.extend([repaired, counter]);
                }
                RawScore::ConditionalRepairWdl { before, after, .. } => {
                    pins.observations.extend([before, after]);
                }
                _ => {}
            }
            self.archive_store_allocation(pins, |stores| {
                stores.append_observation_controlled(observation, || allocation_controls(controls))
            })
        }
    }

    /// One hot allocation may archive inactive Store facts while every engine
    /// index and pending raw handle keeps its original identity. No Node arena
    /// compaction is allowed inside this scope.
    pub(super) fn archive_store_allocation<T>(
        &mut self,
        incoming: StorePins,
        operation: impl FnOnce(&mut PalsStores) -> Result<T, StoreError>,
    ) -> Result<T, PalsError> {
        if !self.stores.archive_enabled() {
            return operation(&mut self.stores).map_err(PalsError::from);
        }
        let mut pins = self.resident_archive_pins()?;
        pins.states.extend(incoming.states);
        pins.lines.extend(incoming.lines);
        pins.situations.extend(incoming.situations);
        pins.observations.extend(incoming.observations);
        pins.executions.extend(incoming.executions);
        let mut budget = self.engine_archive_budget()?;
        let initial = budget.max_bytes;
        let result = self
            .stores
            .with_archive_allocation(pins, &mut budget, operation);
        let used = initial
            .checked_sub(budget.max_bytes)
            .ok_or(StoreError::ArchiveIntegrity(
                "engine allocation archive accounting",
            ))?;
        self.charge_engine_archive_io(used)?;
        result.map_err(PalsError::Store)
    }

    /// Public record handles can be held by a local Repair or its pending queue
    /// before they become a conclusion/dependency. Their lifetime is this
    /// bounded search, rather than all roots in the game.
    pub(super) fn pin_archive_active_record(
        &mut self,
        line: LineId,
        observation: ObservationId,
    ) -> Result<(), PalsError> {
        if !self.stores.archive_enabled() {
            return Ok(());
        }
        self.stores.lines.get(line)?;
        if self.stores.observations.get(observation)?.line != Some(line) {
            return Err(StoreError::InvalidEvidence("active archive record line").into());
        }
        let stats = self.stores.hot_stats();
        let pins = &mut self.archive_state.active_search_pins;
        if (!pins.lines.contains(&line) && pins.lines.len() >= stats.lines)
            || (!pins.observations.contains(&observation)
                && pins.observations.len() >= stats.observations)
        {
            return Err(StoreError::PinSaturated("active archive search handles").into());
        }
        pins.lines.insert(line);
        pins.observations.insert(observation);
        Ok(())
    }

    pub(super) fn archive_insert_situation(
        &mut self,
        snapshot: PositionSnapshot,
        controls: Option<(Instant, &AtomicBool)>,
    ) -> Result<SituationId, PalsError> {
        if controls.is_none() {
            return self
                .stores
                .insert_situation(snapshot)
                .map_err(PalsError::from);
        }
        let controls = controls.filter(|_| self.stores.archive_enabled());
        self.archive_store_allocation(StorePins::default(), |stores| {
            stores.insert_situation_controlled(snapshot, || allocation_controls(controls))
        })
    }

    pub(super) fn archive_append_line(
        &mut self,
        prefix: LineId,
        movements: &[BoardMove],
        controls: Option<(Instant, &AtomicBool)>,
    ) -> Result<LineId, PalsError> {
        if controls.is_none() {
            return self
                .stores
                .append_line(prefix, movements)
                .map_err(PalsError::from);
        }
        let controls = controls.filter(|_| self.stores.archive_enabled());
        let mut pins = StorePins::default();
        pins.lines.insert(prefix);
        self.archive_store_allocation(pins, |stores| {
            stores.append_line_controlled(prefix, movements, || allocation_controls(controls))
        })
    }

    pub(super) fn archive_append_cpu_pv(
        &mut self,
        node: usize,
        movements: &[BoardMove],
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<LineId, PalsError> {
        let position = self.nodes[node].position.clone();
        if self.stopped(limits, cancel).is_some() {
            let result = self.stores.append_cpu_pv(&position, movements);
            return if self.stores.archive_enabled() {
                result.map_err(PalsError::Store)
            } else {
                result.map_err(PalsError::from)
            };
        }
        let controls = self
            .stores
            .archive_enabled()
            .then_some((limits.deadline, cancel));
        self.archive_store_allocation(StorePins::default(), |stores| {
            stores.append_cpu_pv_controlled(&position, movements, || allocation_controls(controls))
        })
    }

    pub(super) fn archive_append_rules_terminal(
        &mut self,
        position: &Position,
        situation: SituationId,
        controls: Option<(Instant, &AtomicBool)>,
    ) -> Result<ObservationId, PalsError> {
        if controls.is_none() {
            return self
                .stores
                .append_rules_terminal(position, 0, 0)
                .map_err(PalsError::from);
        }
        let controls = controls.filter(|_| self.stores.archive_enabled());
        let mut pins = StorePins::default();
        pins.situations.insert(situation);
        pins.states
            .insert(self.stores.situations.get(situation)?.state);
        self.archive_store_allocation(pins, |stores| {
            stores
                .append_rules_terminal_controlled(position, 0, 0, || allocation_controls(controls))
        })
    }

    pub(super) fn archive_request_task(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<TaskAdmission, PalsError> {
        let controls = self
            .stores
            .archive_enabled()
            .then_some((limits.deadline, cancel));
        let consumer_id = consumer.id;
        let mut admitted = None;
        let result = self.archive_store_allocation(StorePins::default(), |stores| {
            let admission = stores.request_task_controlled(key, consumer, now_tick, || {
                allocation_controls(controls)
            })?;
            admitted = Some(admission);
            Ok(admission)
        });
        self.finalize_archive_task_admission(result, admitted, consumer_id, controls)
    }
    fn finalize_archive_task_admission(
        &mut self,
        result: Result<TaskAdmission, PalsError>,
        admitted: Option<TaskAdmission>,
        consumer_id: u64,
        controls: Option<(Instant, &AtomicBool)>,
    ) -> Result<TaskAdmission, PalsError> {
        let result = result.and_then(|admission| {
            Self::check_logical_controls(controls)?;
            Ok(admission)
        });
        if result.is_err()
            && let Some(admission) = admitted
        {
            let execution = match admission {
                TaskAdmission::Start(execution)
                | TaskAdmission::Join(execution)
                | TaskAdmission::Resume { execution, .. }
                | TaskAdmission::Reuse { execution, .. } => execution,
            };
            let cleanup = self
                .stores
                .tasks
                .cancel_consumer(execution, consumer_id)
                .and_then(|()| {
                    if matches!(
                        admission,
                        TaskAdmission::Start(_) | TaskAdmission::Resume { .. }
                    ) {
                        self.stores.tasks.fail(execution)
                    } else {
                        Ok(())
                    }
                });
            if let Err(error) = cleanup {
                // Keep the original allocation/control failure and its separate
                // cleanup error. Joined/completed physical facts are never failed.
                self.last_cpu_checkpoint_cleanup_error = Some(PalsError::Store(error));
            }
        }
        result
    }

    pub(super) fn archive_add_dependency(
        &mut self,
        observation: ObservationId,
        situation: SituationId,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        let controls = self
            .stores
            .archive_enabled()
            .then_some((limits.deadline, cancel));
        let mut pins = StorePins::default();
        pins.observations.insert(observation);
        pins.situations.insert(situation);
        self.archive_store_allocation(pins, |stores| {
            stores
                .add_dependency_controlled(observation, situation, || allocation_controls(controls))
        })
    }

    pub(super) fn archive_refute_continuation(
        &mut self,
        situation: SituationId,
        line: LineId,
        evidence: ObservationId,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        let controls = self
            .stores
            .archive_enabled()
            .then_some((limits.deadline, cancel));
        self.archive_store_allocation(StorePins::default(), |stores| {
            stores.refute_continuation_controlled(situation, line, evidence, || {
                allocation_controls(controls)
            })
        })
    }

    pub(super) fn archive_repair(
        &mut self,
        situation: SituationId,
        refuted: LineId,
        repaired: LineId,
        evidence: ObservationId,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        let controls = self
            .stores
            .archive_enabled()
            .then_some((limits.deadline, cancel));
        self.archive_store_allocation(StorePins::default(), |stores| {
            stores.repair_controlled(situation, refuted, repaired, evidence, || {
                allocation_controls(controls)
            })
        })
    }

    fn engine_archive_budget(&self) -> Result<ArchiveIoBudget, PalsError> {
        let deadline = self
            .archive_state
            .deadline
            .ok_or(StoreError::InvalidConditions(
                "engine archive has no search budget",
            ))?;
        if Instant::now() >= deadline {
            return Err(StoreError::ArchiveDeadline.into());
        }
        Ok(ArchiveIoBudget {
            deadline,
            max_bytes: self.archive_state.remaining_io,
        })
    }

    fn charge_engine_archive_io(&mut self, bytes: u64) -> Result<(), PalsError> {
        self.archive_state.remaining_io = self
            .archive_state
            .remaining_io
            .checked_sub(bytes)
            .ok_or(StoreError::ArchiveByteBudget)?;
        Ok(())
    }

    fn engine_archive_pressure(&self, position: &Position) -> bool {
        let target_state = self.stores.states.find(&position.snapshot());
        self.stores.archive_pressure()
            || self.nodes.len().saturating_mul(5) >= self.config.max_nodes.saturating_mul(4)
            || (self.records.len().saturating_mul(5) >= self.config.max_records.saturating_mul(4)
                && self
                    .records
                    .iter()
                    .any(|record| Some(record.origin_state) != target_state))
    }

    fn retire_stale_archive_checkpoints(&mut self, position: &Position) -> Result<(), PalsError> {
        let current_frontier = self.archive_reachable_nodes(position, false)?;
        let mut live = BTreeSet::new();
        for (index, node) in self.nodes.iter_mut().enumerate() {
            if let Some(token) = &node.resume {
                if self.cpu.token_is_current(token)
                    && (token.is_paused_stack() || current_frontier[index])
                {
                    let executions = self.stores.tasks.paused_executions_for_state(node.state)?;
                    // Only the latest accepted token for this exact state can
                    // own its native traversal. Prior checkpoints remain facts.
                    if let Some(execution) = executions.into_iter().max() {
                        live.insert(execution);
                    }
                } else {
                    node.resume = None;
                }
            }
        }
        self.stores.tasks.retire_paused_except(&live)?;
        Ok(())
    }

    fn resident_archive_pins(&self) -> Result<StorePins, PalsError> {
        let mut pins = self.archive_state.active_search_pins.clone();
        for node in &self.nodes {
            if !self
                .stores
                .states
                .get(node.state)?
                .same_state(&node.position.snapshot())
                || self.stores.situations.get(node.situation)?.state != node.state
            {
                return Err(StoreError::ArchiveIntegrity("resident engine exact state").into());
            }
            pins.states.insert(node.state);
            pins.situations.insert(node.situation);
            if let Some((observation, execution)) =
                node.evidence.as_ref().and_then(|e| e.provenance)
            {
                pins.observations.insert(observation);
                pins.executions.insert(execution);
            }
            pins.observations.extend(node.model_observation);
        }
        for record in &self.records {
            pins.states.insert(record.origin_state);
            pins.observations.extend(record.cpu_observation);
        }
        Ok(pins)
    }

    fn archive_reachable_nodes(
        &self,
        position: &Position,
        include_tasks: bool,
    ) -> Result<Vec<bool>, PalsError> {
        let snapshot = position.snapshot();
        let active_states: BTreeSet<_> = if include_tasks {
            self.stores.tasks.active_states().collect()
        } else {
            BTreeSet::new()
        };
        let mut keep = vec![false; self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            keep[index] = node.position.snapshot().same_state(&snapshot)
                || active_states.contains(&node.state)
                || (include_tasks
                    && node.resume.as_ref().is_some_and(|token| {
                        token.is_paused_stack() && self.cpu.token_is_current(token)
                    }));
        }
        self.archive_child_closure(keep)
    }

    /// Closing the same bounded resident graph is useful both for the RAM
    /// frontier and for a self-contained persisted parent topology. Those
    /// selections have different lifetimes and must never share trim rules.
    fn archive_child_closure(&self, mut selected: Vec<bool>) -> Result<Vec<bool>, PalsError> {
        if selected.len() != self.nodes.len() {
            return Err(StoreError::ArchiveIntegrity("engine closure selection").into());
        }
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| StoreError::PinSaturated("archive frontier index allocation"))?;
        for (index, &included) in selected.iter().enumerate() {
            if included {
                pending.push(index);
            }
        }
        while let Some(index) = pending.pop() {
            for edge in &self.nodes[index].edges {
                if edge.child >= self.nodes.len() {
                    return Err(StoreError::ArchiveIntegrity("engine edge index").into());
                }
                if !selected[edge.child] {
                    selected[edge.child] = true;
                    pending.push(edge.child);
                }
            }
        }
        Ok(selected)
    }

    fn reclaim_archive_for_root(
        &mut self,
        position: &Position,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        archive_controls(deadline, cancel)?;
        // Guard every still-resident reference, including immutable raw facts,
        // before producing a DTO. No automatic Store I/O is enabled here.
        self.stores
            .set_archive_pins(self.resident_archive_pins()?)?;
        let keep = self.archive_reachable_nodes(position, true)?;
        let kept_states: BTreeSet<_> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(index, _)| keep[*index])
            .map(|(_, node)| node.state)
            .collect();
        let mut kept_revisions: BTreeSet<_> = self
            .records
            .iter()
            .filter(|record| kept_states.contains(&record.origin_state))
            .map(|record| record.revision)
            .collect();
        // Retain the actual record relationship closure in the bounded public
        // projection; revisions are not substituted with current-root numbers.
        loop {
            let before = kept_revisions.len();
            for record in &self.records {
                if kept_revisions.contains(&record.revision) {
                    kept_revisions.extend(record.parent_revision);
                    kept_revisions.extend(record.supersedes_revision);
                }
            }
            if kept_revisions.len() == before {
                break;
            }
        }
        // A parent can point into the new actual root. Persist the complete
        // resident child metadata in this receipt as well, while preserving
        // that child's independent hot pin and native checkpoint. The bounded
        // closure copies consume the ordinary per-search and archive quotas.
        let mut archive_selection = Vec::new();
        archive_selection
            .try_reserve_exact(keep.len())
            .map_err(|_| StoreError::PinSaturated("engine archive closure allocation"))?;
        archive_selection.extend(keep.iter().map(|&kept| !kept));
        let archive_selection = self.archive_child_closure(archive_selection)?;
        let archived_states: BTreeSet<_> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(index, _)| archive_selection[*index])
            .map(|(_, node)| node.state)
            .collect();
        let mut archived_revisions: BTreeSet<_> = self
            .records
            .iter()
            .filter(|record| {
                !kept_revisions.contains(&record.revision)
                    || archived_states.contains(&record.origin_state)
            })
            .map(|record| record.revision)
            .collect();
        loop {
            let before = archived_revisions.len();
            for record in &self.records {
                if archived_revisions.contains(&record.revision) {
                    archived_revisions.extend(record.parent_revision);
                    archived_revisions.extend(record.supersedes_revision);
                }
            }
            if archived_revisions.len() == before {
                break;
            }
        }
        let archived_records: Vec<_> = self
            .records
            .iter()
            .filter(|record| archived_revisions.contains(&record.revision))
            .cloned()
            .collect();
        let mut archived_nodes = Vec::new();
        let mut archived_pins = StorePins::default();
        archived_nodes
            .try_reserve_exact(
                archive_selection
                    .iter()
                    .filter(|&&selected| selected)
                    .count(),
            )
            .map_err(|_| StoreError::PinSaturated("engine archive DTO allocation"))?;
        for (index, node) in self.nodes.iter().enumerate() {
            if archive_selection[index] {
                archived_nodes.push(if keep[index] {
                    // A native handle belongs only to the retained hot task.
                    // Its raw partial/complete scope stays historical metadata.
                    encode_archive_node_projection(node, &self.nodes)?
                } else {
                    encode_archive_node(node, &self.nodes)?
                });
                if let Some((observation, execution)) =
                    node.evidence.as_ref().and_then(|e| e.provenance)
                {
                    archived_pins.observations.insert(observation);
                    archived_pins.executions.insert(execution);
                }
                archived_pins.observations.extend(node.model_observation);
            }
        }
        let retire_previous_root = self.stores.root().is_some_and(|root| {
            archived_nodes.iter().any(|node| node.situation == root)
                && self.stores.situations.get(root).is_ok_and(|situation| {
                    self.stores
                        .states
                        .get(situation.state)
                        .is_ok_and(|snapshot| !snapshot.same_state(&position.snapshot()))
                })
        });
        // Prepare all compaction allocations before the durable barrier.
        let mut next_nodes = Vec::new();
        next_nodes
            .try_reserve_exact(self.config.max_nodes)
            .map_err(|_| StoreError::PinSaturated("engine compact arena allocation"))?;
        let mut next_records = Vec::new();
        next_records
            .try_reserve_exact(self.config.max_records)
            .map_err(|_| StoreError::PinSaturated("engine compact records allocation"))?;
        let mut remap = vec![usize::MAX; self.nodes.len()];
        let mut next = 0;
        for (index, &kept) in keep.iter().enumerate() {
            if kept {
                remap[index] = next;
                next += 1;
            }
        }
        for (index, node) in self.nodes.iter().enumerate() {
            if keep[index]
                && node
                    .edges
                    .iter()
                    .any(|edge| remap[edge.child] == usize::MAX)
            {
                return Err(
                    StoreError::ArchiveIntegrity("kept engine frontier is not closed").into(),
                );
            }
        }
        if !archived_nodes.is_empty() || !archived_records.is_empty() {
            let mut budget = self.engine_archive_budget()?;
            let initial = budget.max_bytes;
            let result = self.stores.archive_engine_records_with_pins_and_budget(
                &archived_records,
                &archived_nodes,
                archived_pins,
                &mut budget,
            );
            self.charge_engine_archive_io(initial - budget.max_bytes)?;
            let receipt = result.map_err(PalsError::Store)?;
            if retire_previous_root {
                // The old root projection can reference every examined leaf.
                // Only this checked receipt can expire its logical consumers.
                self.stores.retire_actual_root_for_archive(&receipt)?;
            }
            self.archive_state.last_receipt = Some(receipt);
            // Only the checked commit above grants permission to drop metadata.
            for (index, mut node) in std::mem::take(&mut self.nodes).into_iter().enumerate() {
                if keep[index] {
                    for edge in &mut node.edges {
                        edge.child = remap[edge.child];
                    }
                    next_nodes.push(node);
                }
            }
            for record in std::mem::take(&mut self.records) {
                if kept_revisions.contains(&record.revision) {
                    next_records.push(record);
                }
            }
            self.nodes = next_nodes;
            self.records = next_records;
            let checkpoints: BTreeMap<_, _> = self
                .nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (node.state, index as u64))
                .collect();
            self.stores.tasks.remap_paused_checkpoints(&checkpoints)?;
        }
        self.stores
            .set_archive_pins(self.resident_archive_pins()?)?;
        let mut budget = self.engine_archive_budget()?;
        let initial = budget.max_bytes;
        let result = self.stores.archive_inactive_with_budget(&mut budget);
        self.charge_engine_archive_io(initial - budget.max_bytes)?;
        result.map_err(PalsError::Store)?;
        archive_controls(deadline, cancel)
    }

    /// Explicit cold restore. A caller supplies both deadline and aggregate I/O
    /// bound; the original Store owner/generation and exact history are checked.
    /// This performs no CPU/model execution and never invents missing evidence.
    pub fn load_archived_root(
        &mut self,
        position: &Position,
        mut budget: ArchiveIoBudget,
    ) -> Result<usize, PalsError> {
        if budget.max_bytes > SEARCH_ARCHIVE_IO_BYTES {
            return Err(StoreError::InvalidConditions("engine cold load exceeds 16MiB").into());
        }
        let snapshot = position.snapshot();
        let state = self
            .stores
            .lookup_cold_state_with_budget(&snapshot, &mut budget)
            .map_err(PalsError::Store)?
            .ok_or(StoreError::ColdRecord("archived engine root"))?
            .original_id();
        let receipt = self
            .stores
            .lookup_engine_node_archive_with_budget(state, &mut budget)
            .map_err(PalsError::Store)?
            .ok_or(StoreError::ColdRecord("archived engine metadata"))?;
        // Store checks dependency closure, owner/generation and quotas before
        // hot publication. Metadata validation precedes engine publication.
        let loaded = self
            .stores
            .load_engine_archive_for_state(&receipt, state, budget)
            .map_err(PalsError::Store)?;
        let pin = loaded.pin;
        let result = self.install_loaded_archive(
            state,
            position,
            budget.deadline,
            loaded.situations,
            loaded.engine_nodes,
            loaded.engine_records,
        );
        let release = self.stores.release_loaded(pin);
        match result {
            Err(error) => Err(error),
            Ok(root) => {
                release.map_err(PalsError::Store)?;
                self.stores
                    .set_archive_pins(self.resident_archive_pins()?)?;
                Ok(root)
            }
        }
    }

    fn install_loaded_archive(
        &mut self,
        root_state: StateId,
        expected: &Position,
        deadline: Instant,
        situations: Vec<(SituationId, SituationId)>,
        wires: Vec<EngineArchiveNode>,
        records: Vec<RoleRecord>,
    ) -> Result<usize, PalsError> {
        let situation_map: BTreeMap<_, _> = situations.into_iter().collect();
        let mut indices: BTreeMap<_, _> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.state, index))
            .collect();
        let additions = wires
            .iter()
            .filter(|wire| !indices.contains_key(&wire.state))
            .count();
        if self
            .nodes
            .len()
            .checked_add(additions)
            .is_none_or(|n| n > self.config.max_nodes)
        {
            return Err(StoreError::PinSaturated("loaded engine nodes").into());
        }
        let mut new_nodes = Vec::new();
        new_nodes
            .try_reserve_exact(additions)
            .map_err(|_| StoreError::PinSaturated("loaded engine arena allocation"))?;
        for wire in &wires {
            archive_deadline(deadline)?;
            if indices.contains_key(&wire.state) {
                continue;
            }
            let snapshot = self.stores.states.get(wire.state)?;
            let position = position_from_snapshot(snapshot)?;
            let situation = *situation_map
                .get(&wire.situation)
                .ok_or(StoreError::ArchiveIntegrity("engine situation restoration"))?;
            if self.stores.situations.get(situation)?.state != wire.state {
                return Err(StoreError::ArchiveIntegrity("engine restored situation state").into());
            }
            let node = decode_archive_node(wire, position, situation)?;
            self.validate_loaded_node_evidence(&node)?;
            let index = self.nodes.len() + new_nodes.len();
            indices.insert(wire.state, index);
            new_nodes.push(node);
        }
        for wire in &wires {
            archive_deadline(deadline)?;
            let index = *indices
                .get(&wire.state)
                .ok_or(StoreError::ArchiveIntegrity("engine restored node index"))?;
            if index < self.nodes.len() {
                continue;
            }
            let node = &mut new_nodes[index - self.nodes.len()];
            node.edges
                .try_reserve_exact(wire.edges.len())
                .map_err(|_| StoreError::PinSaturated("loaded engine edge allocation"))?;
            for (movement, child) in &wire.edges {
                archive_deadline(deadline)?;
                let child =
                    child.ok_or(StoreError::ArchiveIntegrity("missing exact engine child"))?;
                let movement = movement.unpack()?;
                let mut replay = node.position.clone();
                replay
                    .make_move(movement)
                    .map_err(|_| StoreError::ArchiveIntegrity("engine cold edge Rules legality"))?;
                if !replay.snapshot().same_state(self.stores.states.get(child)?) {
                    return Err(
                        StoreError::ArchiveIntegrity("engine cold edge exact history").into(),
                    );
                }
                let child = *indices
                    .get(&child)
                    .ok_or(StoreError::ColdRecord("engine edge topology"))?;
                node.edges.push(Edge { movement, child });
            }
        }
        let root = *indices
            .get(&root_state)
            .ok_or(StoreError::ColdRecord("engine root topology"))?;
        let restored = if root < self.nodes.len() {
            &self.nodes[root].position
        } else {
            &new_nodes[root - self.nodes.len()].position
        };
        if !restored.snapshot().same_state(&expected.snapshot()) {
            return Err(StoreError::ArchiveIntegrity("engine cold root exact history").into());
        }
        let mut added_records = Vec::new();
        for record in records {
            archive_deadline(deadline)?;
            if self
                .records
                .iter()
                .any(|old| old.revision == record.revision)
            {
                continue;
            }
            added_records.push(record);
        }
        if self
            .records
            .len()
            .checked_add(added_records.len())
            .is_none_or(|n| n > self.config.max_records)
        {
            return Err(StoreError::PinSaturated("loaded engine public records").into());
        }
        self.nodes
            .try_reserve_exact(new_nodes.len())
            .map_err(|_| StoreError::PinSaturated("restored engine node allocation"))?;
        self.records
            .try_reserve_exact(added_records.len())
            .map_err(|_| StoreError::PinSaturated("restored engine record allocation"))?;
        archive_deadline(deadline)?;
        self.nodes.extend(new_nodes);
        self.records.extend(added_records);
        self.records.sort_by_key(|record| record.revision);
        Ok(root)
    }

    fn validate_loaded_node_evidence(&self, node: &Node) -> Result<(), PalsError> {
        if let Some(evidence) = &node.evidence {
            if self.cpu_registered_value.as_ref() != Some(&evidence.value_identity) {
                return Err(StoreError::ArchiveIntegrity("engine restored CPU namespace").into());
            }
            if let Some((observation, execution)) = evidence.provenance {
                let fact = self.stores.observations.get(observation)?;
                let expected_score = if evidence.scope == CpuScoreScope::FrontierOnly {
                    RawScore::Estimate {
                        value: evidence.score as f32,
                        perspective: node.position.side_to_move(),
                    }
                } else {
                    RawScore::Cpu {
                        value: evidence.score,
                        perspective: node.position.side_to_move(),
                        bound: BoundKind::ExactWithinSearch,
                    }
                };
                if fact.state != node.state
                    || fact.execution != Some(execution)
                    || fact.value_identity.as_ref() != Some(&evidence.value_identity)
                    || fact.score != expected_score
                    || !matches!(fact.scope, EvidenceScope::DepthLimited {depth,..} if depth == evidence.depth)
                {
                    return Err(
                        StoreError::ArchiveIntegrity("engine restored CPU observation").into(),
                    );
                }
                self.stores.tasks.get(execution)?;
            }
        }
        if let Some(output) = &node.model_value {
            if self.model_registered_value.as_ref() != Some(&output.identity) {
                return Err(StoreError::ArchiveIntegrity("engine restored model namespace").into());
            }
            let observation = node
                .model_observation
                .ok_or(StoreError::ArchiveIntegrity("engine model observation"))?;
            let fact = self.stores.observations.get(observation)?;
            let same_value = |win, draw, loss, perspective| {
                output.wdl == [win, draw, loss] && output.perspective == perspective
            };
            let supported_score = match fact.score {
                RawScore::ContextWdl {
                    win,
                    draw,
                    loss,
                    perspective,
                    context_revision,
                } => {
                    same_value(win, draw, loss, perspective)
                        && node.model_value_revision == Some(context_revision)
                }
                RawScore::Wdl {
                    win,
                    draw,
                    loss,
                    perspective,
                } if !self.followup_lane && !self.post_repair_recheck.uses_frozen_model_wdl() => {
                    // Match the actual legacy publisher gate. This preserves
                    // an old raw reading without adding a revision or minting
                    // a new ContextWdl/frozen comparison proof.
                    same_value(win, draw, loss, perspective)
                }
                _ => false,
            };
            if fact.state != node.state
                || fact.model_value_identity.as_ref() != Some(&output.identity)
                || fact.model_value_input != Some(output.input_sha256)
                || !supported_score
            {
                return Err(
                    StoreError::ArchiveIntegrity("engine restored model observation").into(),
                );
            }
        }
        Ok(())
    }
}

fn allocation_controls(controls: Option<(Instant, &AtomicBool)>) -> Result<(), StoreError> {
    if let Some((deadline, cancel)) = controls {
        if cancel.load(Ordering::Acquire) {
            return Err(StoreError::ArchiveCanceled);
        }
        if Instant::now() >= deadline {
            return Err(StoreError::ArchiveDeadline);
        }
    }
    Ok(())
}

fn archive_controls(deadline: Instant, cancel: &AtomicBool) -> Result<(), PalsError> {
    if cancel.load(Ordering::Acquire) {
        Err(RoleError::Canceled.into())
    } else if Instant::now() >= deadline {
        Err(StoreError::ArchiveDeadline.into())
    } else {
        Ok(())
    }
}

fn archive_deadline(deadline: Instant) -> Result<(), PalsError> {
    if Instant::now() >= deadline {
        Err(StoreError::ArchiveDeadline.into())
    } else {
        Ok(())
    }
}

fn position_from_snapshot(snapshot: &PositionSnapshot) -> Result<Position, PalsError> {
    let position = Position::from_snapshot(snapshot, rz_position::PositionLimits::default())?;
    if !position.snapshot().same_state(snapshot) {
        return Err(StoreError::ArchiveIntegrity("engine exact Rules restoration").into());
    }
    Ok(position)
}

struct MetadataWriter(Vec<u8>);
impl MetadataWriter {
    fn new() -> Result<Self, PalsError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(NODE_METADATA_MAGIC.len())
            .map_err(|_| StoreError::PinSaturated("engine metadata allocation"))?;
        bytes.extend_from_slice(NODE_METADATA_MAGIC);
        Ok(Self(bytes))
    }
    fn bytes(&mut self, bytes: &[u8]) -> Result<(), PalsError> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > NODE_METADATA_BYTES)
        {
            return Err(PalsError::Store(StoreError::Capacity(
                "engine node metadata 4KiB",
            )));
        }
        self.0
            .try_reserve_exact(bytes.len())
            .map_err(|_| StoreError::PinSaturated("engine metadata allocation"))?;
        self.0.extend_from_slice(bytes);
        Ok(())
    }
    fn tag(&mut self, value: u8) -> Result<(), PalsError> {
        self.bytes(&[value])
    }
    fn u16(&mut self, value: u16) -> Result<(), PalsError> {
        self.bytes(&value.to_le_bytes())
    }
    fn u64(&mut self, value: u64) -> Result<(), PalsError> {
        self.bytes(&value.to_le_bytes())
    }
    fn i32(&mut self, value: i32) -> Result<(), PalsError> {
        self.bytes(&value.to_le_bytes())
    }
    fn text(&mut self, value: &str, maximum: usize) -> Result<(), PalsError> {
        if value.len() > maximum {
            return Err(StoreError::ArchiveIntegrity("engine identity text bound").into());
        }
        self.u16(value.len() as u16)?;
        self.bytes(value.as_bytes())
    }
    fn optional_id(&mut self, value: Option<usize>) -> Result<(), PalsError> {
        self.tag(u8::from(value.is_some()))?;
        if let Some(value) = value {
            self.u64(
                u64::try_from(value)
                    .map_err(|_| StoreError::ArchiveIntegrity("engine ID width"))?,
            )?;
        }
        Ok(())
    }
}

struct MetadataReader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> MetadataReader<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, PalsError> {
        if bytes.len() > NODE_METADATA_BYTES || !bytes.starts_with(NODE_METADATA_MAGIC) {
            return Err(
                StoreError::ArchiveIntegrity("engine node metadata version or bound").into(),
            );
        }
        Ok(Self {
            bytes,
            at: NODE_METADATA_MAGIC.len(),
        })
    }
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], PalsError> {
        let end = self
            .at
            .checked_add(N)
            .ok_or(StoreError::ArchiveIntegrity("engine metadata offset"))?;
        let bytes = self
            .bytes
            .get(self.at..end)
            .ok_or(StoreError::ArchiveIntegrity("engine metadata truncation"))?;
        self.at = end;
        Ok(bytes.try_into().expect("checked fixed width"))
    }
    fn tag(&mut self) -> Result<u8, PalsError> {
        Ok(self.bytes::<1>()?[0])
    }
    fn flag(&mut self) -> Result<bool, PalsError> {
        match self.tag()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(StoreError::ArchiveIntegrity("engine metadata flag").into()),
        }
    }
    fn u16(&mut self) -> Result<u16, PalsError> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, PalsError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn i32(&mut self) -> Result<i32, PalsError> {
        Ok(i32::from_le_bytes(self.bytes()?))
    }
    fn text(&mut self, maximum: usize) -> Result<String, PalsError> {
        let length = usize::from(self.u16()?);
        if length > maximum {
            return Err(StoreError::ArchiveIntegrity("engine identity text bound").into());
        }
        let end = self
            .at
            .checked_add(length)
            .ok_or(StoreError::ArchiveIntegrity("engine metadata offset"))?;
        let bytes = self
            .bytes
            .get(self.at..end)
            .ok_or(StoreError::ArchiveIntegrity(
                "engine metadata text truncation",
            ))?;
        let text = std::str::from_utf8(bytes)
            .map_err(|_| StoreError::ArchiveIntegrity("engine identity UTF-8"))?;
        self.at = end;
        Ok(text.to_owned())
    }
    fn optional_id(&mut self) -> Result<Option<usize>, PalsError> {
        if !self.flag()? {
            return Ok(None);
        }
        Ok(Some(usize::try_from(self.u64()?).map_err(|_| {
            StoreError::ArchiveIntegrity("engine ID width")
        })?))
    }
    fn finish(self) -> Result<(), PalsError> {
        if self.at != self.bytes.len() {
            return Err(StoreError::ArchiveIntegrity("engine metadata trailing bytes").into());
        }
        Ok(())
    }
}

fn encode_cpu_identity(
    writer: &mut MetadataWriter,
    identity: &CpuValueIdentity,
) -> Result<(), PalsError> {
    identity
        .validate()
        .map_err(|_| StoreError::ArchiveIntegrity("engine CPU value identity"))?;
    writer.text(&identity.semantics, 256)?;
    writer.tag(u8::from(identity.weights_sha256.is_some()))?;
    if let Some(weights) = &identity.weights_sha256 {
        writer.text(weights, 64)?;
    }
    match &identity.training {
        CpuTrainingState::Bootstrap => writer.tag(0)?,
        CpuTrainingState::Untrained => writer.tag(1)?,
        CpuTrainingState::Learned {
            run_id,
            steps,
            dataset_sha256,
        } => {
            writer.tag(2)?;
            writer.text(run_id, 128)?;
            writer.u64(*steps)?;
            writer.text(dataset_sha256, 64)?;
        }
    }
    Ok(())
}

fn decode_cpu_identity(reader: &mut MetadataReader<'_>) -> Result<CpuValueIdentity, PalsError> {
    let semantics = reader.text(256)?;
    let weights_sha256 = if reader.flag()? {
        Some(reader.text(64)?)
    } else {
        None
    };
    let training = match reader.tag()? {
        0 => CpuTrainingState::Bootstrap,
        1 => CpuTrainingState::Untrained,
        2 => CpuTrainingState::Learned {
            run_id: reader.text(128)?,
            steps: reader.u64()?,
            dataset_sha256: reader.text(64)?,
        },
        _ => return Err(StoreError::ArchiveIntegrity("engine CPU training kind").into()),
    };
    let identity = CpuValueIdentity {
        semantics,
        weights_sha256,
        training,
    };
    identity
        .validate()
        .map_err(|_| StoreError::ArchiveIntegrity("engine CPU value identity"))?;
    Ok(identity)
}

fn encode_archive_node(node: &Node, nodes: &[Node]) -> Result<EngineArchiveNode, PalsError> {
    if node.resume.is_some() {
        return Err(StoreError::PinSaturated("live CPU resume cannot be archived").into());
    }
    encode_archive_node_projection(node, nodes)
}

/// A projection does not serialize mutable native resume ownership. The
/// lifecycle may use it for a child that remains pinned and resident; dropping
/// a node still goes through the live-token guard above.
fn encode_archive_node_projection(
    node: &Node,
    nodes: &[Node],
) -> Result<EngineArchiveNode, PalsError> {
    let mut metadata = MetadataWriter::new()?;
    metadata.tag(u8::from(node.terminal.is_some()))?;
    if let Some((reason, score)) = node.terminal {
        metadata.tag(match reason {
            TerminalReason::Checkmate => 0,
            TerminalReason::Stalemate => 1,
            TerminalReason::DeadPosition => 2,
            TerminalReason::FivefoldRepetition => 3,
            TerminalReason::SeventyFiveMove => 4,
        })?;
        metadata.i32(score)?;
    }
    metadata.tag(u8::from(node.evidence.is_some()))?;
    if let Some(evidence) = &node.evidence {
        metadata.i32(evidence.score)?;
        metadata.u16(evidence.depth)?;
        metadata.tag(match evidence.scope {
            CpuScoreScope::FrontierOnly => 0,
            CpuScoreScope::CompletedIteration => 1,
            CpuScoreScope::RulesTerminal => 2,
        })?;
        encode_cpu_identity(&mut metadata, &evidence.value_identity)?;
        metadata.tag(u8::from(evidence.provenance.is_some()))?;
        if let Some((observation, execution)) = evidence.provenance {
            metadata.optional_id(Some(observation.0))?;
            metadata.optional_id(Some(execution.0))?;
        }
    }
    metadata.tag(u8::from(node.model_value.is_some()))?;
    if let Some(output) = &node.model_value {
        output.validate(&node.position, &output.identity)?;
        for (text, maximum) in [
            (&output.identity.semantics, 256),
            (&output.identity.model, 1024),
            (&output.identity.encoding, 256),
            (&output.identity.precision, 64),
        ] {
            metadata.text(text, maximum)?;
        }
        metadata.bytes(&output.identity.model_epoch)?;
        metadata.bytes(&output.input_sha256)?;
        metadata.tag(match output.perspective {
            Color::White => 0,
            Color::Black => 1,
        })?;
        for probability in output.wdl {
            metadata.bytes(&probability.to_bits().to_le_bytes())?;
        }
    }
    metadata.optional_id(node.model_observation.map(|id| id.0))?;
    metadata.tag(u8::from(node.model_value_revision.is_some()))?;
    if let Some(revision) = node.model_value_revision {
        metadata.u64(revision)?;
    }
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(node.edges.len())
        .map_err(|_| StoreError::PinSaturated("engine archive edges allocation"))?;
    for edge in &node.edges {
        let child = nodes
            .get(edge.child)
            .ok_or(StoreError::ArchiveIntegrity("engine edge index"))?;
        edges.push((Move16::pack(edge.movement)?, Some(child.state)));
    }
    Ok(EngineArchiveNode {
        state: node.state,
        situation: node.situation,
        edges,
        metadata: metadata.0,
    })
}

fn decode_archive_node(
    wire: &EngineArchiveNode,
    position: Position,
    situation: SituationId,
) -> Result<Node, PalsError> {
    let mut reader = MetadataReader::new(&wire.metadata)?;
    let terminal = if reader.flag()? {
        let reason = match reader.tag()? {
            0 => TerminalReason::Checkmate,
            1 => TerminalReason::Stalemate,
            2 => TerminalReason::DeadPosition,
            3 => TerminalReason::FivefoldRepetition,
            4 => TerminalReason::SeventyFiveMove,
            _ => return Err(StoreError::ArchiveIntegrity("engine terminal reason").into()),
        };
        Some((reason, reader.i32()?))
    } else {
        None
    };
    let actual_terminal = match position.classify_position()?.play_status {
        PlayStatus::Ongoing => None,
        PlayStatus::Terminal { reason, winner } => Some((
            reason,
            match winner {
                Some(winner) if winner == position.side_to_move() => CPU_MATE_SCORE,
                Some(_) => -CPU_MATE_SCORE,
                None => 0,
            },
        )),
    };
    if terminal != actual_terminal {
        return Err(StoreError::ArchiveIntegrity("engine Rules terminal mismatch").into());
    }
    let evidence = if reader.flag()? {
        let score = reader.i32()?;
        let depth = reader.u16()?;
        let scope = match reader.tag()? {
            0 => CpuScoreScope::FrontierOnly,
            1 => CpuScoreScope::CompletedIteration,
            2 => CpuScoreScope::RulesTerminal,
            _ => return Err(StoreError::ArchiveIntegrity("engine CPU score scope").into()),
        };
        let value_identity = decode_cpu_identity(&mut reader)?;
        let provenance = if reader.flag()? {
            let observation = reader
                .optional_id()?
                .ok_or(StoreError::ArchiveIntegrity("engine CPU observation ID"))?;
            let execution = reader
                .optional_id()?
                .ok_or(StoreError::ArchiveIntegrity("engine CPU execution ID"))?;
            Some((ObservationId(observation), ExecutionId(execution)))
        } else {
            None
        };
        if scope == CpuScoreScope::RulesTerminal && terminal.map(|(_, value)| value) != Some(score)
        {
            return Err(StoreError::ArchiveIntegrity("engine CPU Rules-terminal scope").into());
        }
        Some(CpuEvidence {
            score,
            depth,
            scope,
            value_identity,
            provenance,
        })
    } else {
        None
    };
    let model_value = if reader.flag()? {
        let identity = ModelValueIdentity {
            semantics: reader.text(256)?,
            model: reader.text(1024)?,
            encoding: reader.text(256)?,
            precision: reader.text(64)?,
            model_epoch: reader.bytes()?,
        };
        let input_sha256 = reader.bytes()?;
        let perspective = match reader.tag()? {
            0 => Color::White,
            1 => Color::Black,
            _ => return Err(StoreError::ArchiveIntegrity("engine model perspective").into()),
        };
        let mut wdl = [0.0; 3];
        for probability in &mut wdl {
            *probability = f32::from_bits(u32::from_le_bytes(reader.bytes()?));
        }
        let output = ModelValueOutput {
            identity,
            input_sha256,
            state: position.position_identity(),
            perspective,
            wdl,
        };
        output.validate(&position, &output.identity)?;
        Some(output)
    } else {
        None
    };
    let model_observation = reader.optional_id()?.map(ObservationId);
    let model_value_revision = if reader.flag()? {
        Some(reader.u64()?)
    } else {
        None
    };
    reader.finish()?;
    if model_value.is_some() != model_observation.is_some()
        || model_value.is_some() != model_value_revision.is_some()
    {
        return Err(StoreError::ArchiveIntegrity("engine model value provenance fields").into());
    }
    Ok(Node {
        position,
        state: wire.state,
        situation,
        terminal,
        edges: Vec::new(),
        evidence,
        model_value,
        model_observation,
        model_value_revision,
        resume: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;
    use crate::cpu_checker::{
        CheckerCapabilities, CheckerWork, ExternalBound, ExternalCheckerIdentity,
        ExternalModelMetadata, ExternalRawScore, ExternalTrainingKnowledge, ExternalUciIdentity,
    };
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;

    fn archive_config(label: &str) -> ArchiveConfig {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let output = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("RUNNER_TEMP").map(PathBuf::from)
        }
        .unwrap_or_else(std::env::temp_dir);
        let root = output.join("RoveZero").join("runs").join(format!(
            "engine-archive-{label}-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_owned();
        ArchiveConfig::new(root, repository)
    }

    fn engine() -> PalsEngine<LegalOrderRoleMock> {
        PalsEngine::new(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 32,
                max_role_calls: 64,
                cpu_nodes_per_task: 256,
            },
            LegalOrderRoleMock,
            CpuEngine::new(CpuConfig {
                tt_entries: 64,
                max_depth: 4,
                quiescence_ply: 4,
                ..CpuConfig::default()
            })
            .unwrap(),
        )
        .unwrap()
    }

    fn limits() -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 1,
            max_cpu_nodes: 1024,
            cpu_depth: 1,
        }
    }
    fn inactive_position(fullmove: usize) -> Position {
        Position::from_fen(&format!(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {fullmove}"
        ))
        .unwrap()
    }
    fn unknown_observation(state: StateId, line: Option<LineId>) -> Observation {
        Observation {
            state,
            line,
            source: 9,
            epoch: 0,
            scope: EvidenceScope::Model {
                model: 1,
                encoding: 2,
                input: 0,
            },
            score: RawScore::Unknown,
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
            budget: 0,
            kind: ObservationKind::Proposal,
            supersedes: None,
            execution: None,
        }
    }

    fn fill_hot(engine: &mut PalsEngine<LegalOrderRoleMock>) {
        let mut fullmove = 1000;
        while engine.nodes.len() < engine.config.max_nodes {
            engine
                .intern(
                    Position::from_fen(&format!(
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {fullmove}"
                    ))
                    .unwrap(),
                )
                .unwrap();
            fullmove += 1;
        }
        assert_eq!(engine.stores.hot_stats().states, 257);
    }

    fn advanced_root() -> Position {
        let mut position = Position::startpos();
        for movement in ["e2e4", "e7e5", "g1f3", "b8c6"] {
            position.make_move(movement.parse().unwrap()).unwrap();
        }
        position
    }

    fn foreign_identity() -> ExternalCheckerIdentity {
        ExternalCheckerIdentity {
            adapter_semantics: "engine-archive-foreign-fixture/1".into(),
            binary_sha256: "a".repeat(64),
            launch_arguments_sha256: "b".repeat(64),
            declared_name: "typed fixture".into(),
            declared_version: "1".into(),
            declared_source: "test-only".into(),
            declared_license: "MIT".into(),
            options: BTreeMap::new(),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        }
    }

    /// A legacy constructor registration fixture; CPU dispatch is forbidden.
    /// The test exercises only the real model publisher and archive boundary.
    struct ModelOnlyForeignFixture {
        identity: CheckerIdentity,
    }
    impl CpuChecker for ModelOnlyForeignFixture {
        fn identity(&self) -> &CheckerIdentity {
            &self.identity
        }
        fn conditions(&self) -> &str {
            "engine-archive-foreign-fixture-conditions/1"
        }
        fn capabilities(&self) -> CheckerCapabilities {
            CheckerCapabilities {
                max_depth: 4,
                max_prefix_plies: 4,
                max_root_moves: 256,
                root_moves: true,
                divergence: true,
                resume: false,
                selective_search: None,
            }
        }
        fn last_attempt(&self) -> Option<&CheckerAttempt> {
            None
        }
        fn reset_attempt(&mut self) {}
        fn analyze(
            &mut self,
            _: &Position,
            _: CpuLimits,
            _: &AtomicBool,
        ) -> Result<CheckerReport, CheckerError> {
            Err(CheckerError::Unsupported(
                "model-only fixture forbids CPU dispatch",
            ))
        }
        fn new_game(&mut self, _: Instant, _: &AtomicBool) -> Result<(), CheckerError> {
            Ok(())
        }
        fn shutdown(&mut self, _: Instant) -> Result<CheckerShutdown, CheckerError> {
            Ok(CheckerShutdown::default())
        }
    }

    /// Only an in-process typed raw fixture. There is no UCI process, network,
    /// model or physical external-checker completion claim in these tests.
    fn foreign_observation(
        engine: &mut PalsEngine<LegalOrderRoleMock>,
        position: &Position,
        request_id: u64,
        consumer_id: u64,
    ) -> Observation {
        let node = engine
            .nodes
            .iter()
            .find(|node| node.position.snapshot().same_state(&position.snapshot()))
            .unwrap();
        let state = node.state;
        let situation = node.situation;
        let identity = foreign_identity();
        let condition = "engine-archive-foreign-fixture-conditions/1";
        let TaskAdmission::Start(execution) = engine
            .stores
            .request_task(
                TaskKey {
                    state,
                    line: None,
                    question: TaskQuestion::AnalyzePosition,
                    root_moves: Vec::new(),
                    model: 1,
                    epoch: 1,
                    value_identity: None,
                    checker_identity: Some(CheckerIdentity::ExternalUci(identity.clone())),
                    cpu_condition: Some(condition.into()),
                    profile: 1,
                    condition: 1,
                    input_revision: 0,
                    requested_depth: 4,
                    node_budget: 32,
                },
                TaskConsumer {
                    id: consumer_id,
                    situation,
                    revision: engine.stores.situations.get(situation).unwrap().revision,
                    generation: engine.stores.generation(),
                    deadline_tick: 100_000,
                },
                0,
            )
            .unwrap()
        else {
            panic!("fixture foreign task must be new");
        };
        let movement = position.legal_moves()[0];
        let pv = engine.stores.append_cpu_pv(position, &[movement]).unwrap();
        let report = ExternalCheckerReport {
            identity: identity.clone(),
            observed_uci: ExternalUciIdentity {
                name: "typed fixture, process unobserved".into(),
                author: None,
            },
            request_id,
            best_move: Some(movement),
            pv: vec![movement],
            score: ExternalRawScore::MateMoves(19),
            bound: ExternalBound::Lower,
            wdl_per_mille: Some([900, 50, 50]),
            perspective: position.side_to_move(),
            requested_depth: 4,
            reported_depth: Some(1),
            seldepth: Some(3),
            root_restricted: false,
            completion: ExternalCompletion::BestMove,
            work: CheckerWork {
                nodes: Some(1),
                qnodes: None,
                tt_hits: None,
            },
            elapsed: Duration::ZERO,
        };
        Observation {
            state,
            line: None,
            source: stable_id(&identity.adapter_semantics),
            epoch: 1,
            value_identity: None,
            checker_identity: Some(CheckerIdentity::ExternalUci(identity)),
            checker_work: Some(report.work),
            external_report: Some(Box::new(report.clone())),
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: Some(condition.into()),
            cpu_pv: Some(pv),
            scope: EvidenceScope::ExternalUci {
                requested_depth: report.requested_depth,
                reported_depth: report.reported_depth,
                seldepth: report.seldepth,
                bound: report.bound,
            },
            score: RawScore::ExternalUci {
                value: report.score,
                bound: report.bound,
                perspective: report.perspective,
                wdl_per_mille: report.wdl_per_mille,
            },
            budget: 1,
            kind: ObservationKind::ExternalCpuAnalysis,
            supersedes: None,
            execution: Some(execution),
        }
    }

    #[test]
    fn saturated_hot_archive_investigates_the_actual_new_root_and_keeps_old_topology() {
        let mut engine = engine();
        engine.enable_archive(archive_config("saturation")).unwrap();
        let old_position = Position::startpos();
        let initial = engine
            .search(&old_position, limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(initial.counters.cpu_tasks > 0);
        let old_root = engine.stores.root().unwrap();
        let old_state = engine.stores.situations.get(old_root).unwrap().state;
        let old_records: Vec<_> = engine
            .records
            .iter()
            .filter(|record| record.origin_state == old_state)
            .map(|record| {
                (
                    record.revision,
                    record.parent_revision,
                    record.supersedes_revision,
                )
            })
            .collect();
        fill_hot(&mut engine);
        let next = advanced_root();
        let result = engine
            .search(&next, limits(), &AtomicBool::new(false))
            .unwrap();
        assert_ne!(result.completion, PalsCompletion::Capacity);
        assert!(
            result.counters.cpu_tasks > 0,
            "a fallback legal move is not new-root investigation"
        );
        assert!(engine.nodes.len() < 257);
        let root = engine.stores.root().unwrap();
        assert!(
            engine
                .stores
                .states
                .get(engine.stores.situations.get(root).unwrap().state)
                .unwrap()
                .same_state(&next.snapshot())
        );
        assert!(matches!(
            engine.stores.states.get(old_state),
            Err(StoreError::ColdRecord(_))
        ));
        let budget = ArchiveIoBudget {
            deadline: Instant::now() + Duration::from_secs(20),
            max_bytes: SEARCH_ARCHIVE_IO_BYTES,
        };
        let receipt = engine
            .stores
            .lookup_engine_archive(old_state, budget)
            .unwrap()
            .unwrap();
        let (records, nodes, _) = engine.stores.read_engine_archive(&receipt, budget).unwrap();
        let old_node = nodes.iter().find(|node| node.state == old_state).unwrap();
        assert_eq!(old_node.situation, old_root);
        assert!(!old_node.edges.is_empty());
        assert!(old_node.edges.iter().all(|(_, child)| child.is_some()));
        for relationship in old_records {
            assert!(records.iter().any(|record| (
                record.revision,
                record.parent_revision,
                record.supersedes_revision
            ) == relationship));
        }
        for node in &engine.nodes {
            for edge in &node.edges {
                let mut replay = node.position.clone();
                replay.make_move(edge.movement).unwrap();
                assert!(
                    replay
                        .snapshot()
                        .same_state(&engine.nodes[edge.child].position.snapshot())
                );
            }
        }
    }

    #[test]
    fn cold_engine_restore_is_explicit_bounded_and_keeps_exact_history_and_evidence() {
        let mut engine = engine();
        engine.enable_archive(archive_config("restore")).unwrap();
        let old = Position::startpos();
        engine
            .search(&old, limits(), &AtomicBool::new(false))
            .unwrap();
        let state = engine
            .stores
            .situations
            .get(engine.stores.root().unwrap())
            .unwrap()
            .state;
        fill_hot(&mut engine);
        engine
            .search(&advanced_root(), limits(), &AtomicBool::new(false))
            .unwrap();
        let resident_before = engine.nodes.len();
        let insufficient = engine
            .load_archived_root(
                &old,
                ArchiveIoBudget {
                    deadline: Instant::now() + Duration::from_secs(20),
                    max_bytes: 1,
                },
            )
            .unwrap_err();
        assert!(matches!(
            insufficient,
            PalsError::Store(StoreError::ArchiveByteBudget)
        ));
        assert_eq!(engine.nodes.len(), resident_before);
        let root = engine
            .load_archived_root(
                &old,
                ArchiveIoBudget {
                    deadline: Instant::now() + Duration::from_secs(20),
                    max_bytes: SEARCH_ARCHIVE_IO_BYTES,
                },
            )
            .unwrap();
        assert_eq!(engine.nodes[root].state, state);
        assert!(
            engine.nodes[root]
                .position
                .snapshot()
                .same_state(&old.snapshot())
        );
        assert!(!engine.nodes[root].edges.is_empty());
        assert!(engine.nodes.iter().any(|node| {
            node.evidence
                .as_ref()
                .is_some_and(|evidence| evidence.provenance.is_some())
        }));
        // A newer public-record-only receipt must not hide this state's older
        // self-contained topology when the explicit root is restored again.
        let record = engine
            .records
            .iter()
            .position(|record| record.origin_state == state)
            .unwrap();
        engine.archive_public_record_before_remove(record).unwrap();
        assert_eq!(engine.last_engine_archive().unwrap().nodes, 0);
        let restored_again = engine
            .load_archived_root(
                &old,
                ArchiveIoBudget {
                    deadline: Instant::now() + Duration::from_secs(20),
                    max_bytes: SEARCH_ARCHIVE_IO_BYTES,
                },
            )
            .unwrap();
        assert_eq!(engine.nodes[restored_again].state, state);
        let imported = Position::from_fen(&old.snapshot().to_fen()).unwrap();
        assert!(
            !engine.nodes[root]
                .position
                .snapshot()
                .same_state(&imported.snapshot())
        );
    }

    #[test]
    fn actual_model_wdl_archive_restores_raw_context_revision_and_rejects_substitution() {
        let previous = engine();
        let mut engine = PalsEngine::new_with_boxed_checker_and_policies(
            previous.config,
            previous.model,
            previous.cpu,
            ResolverPolicy::ModelWdlRestricted,
            PostRepairRecheckPolicy::Disabled,
        )
        .unwrap();
        engine
            .enable_archive(archive_config("model-context"))
            .unwrap();
        let cancel = AtomicBool::new(false);
        let old = Position::startpos();
        let report = engine.search(&old, limits(), &cancel).unwrap();
        assert!(report.counters.completed_value_calls > 0);
        let published: Vec<_> = engine
            .nodes
            .iter()
            .filter_map(|node| {
                node.model_value.as_ref().map(|output| {
                    (
                        node.state,
                        node.model_observation.unwrap(),
                        node.model_value_revision.unwrap(),
                        output.wdl,
                        output.input_sha256,
                    )
                })
            })
            .collect();
        assert!(!published.is_empty());
        fill_hot(&mut engine);
        engine.search(&advanced_root(), limits(), &cancel).unwrap();
        engine
            .load_archived_root(
                &old,
                ArchiveIoBudget {
                    deadline: Instant::now() + Duration::from_secs(20),
                    max_bytes: SEARCH_ARCHIVE_IO_BYTES,
                },
            )
            .unwrap();
        for (state, observation, revision, wdl, digest) in published {
            let index = engine
                .nodes
                .iter()
                .position(|node| node.state == state)
                .unwrap();
            let node = &engine.nodes[index];
            assert_eq!(node.model_observation, Some(observation));
            assert_eq!(node.model_value_revision, Some(revision));
            assert_eq!(node.model_value.as_ref().unwrap().wdl, wdl);
            assert_eq!(node.model_value.as_ref().unwrap().input_sha256, digest);
            assert!(
                matches!(engine.stores.observations.get(observation).unwrap().score,
                RawScore::ContextWdl {context_revision,..} if context_revision == revision)
            );
            engine.validate_loaded_node_evidence(node).unwrap();
            engine.nodes[index].model_value_revision = Some(revision.checked_add(1).unwrap());
            assert!(matches!(
                engine.validate_loaded_node_evidence(&engine.nodes[index]),
                Err(PalsError::Store(StoreError::ArchiveIntegrity(
                    "engine restored model observation"
                )))
            ));
            engine.nodes[index].model_value_revision = Some(revision);
        }
    }

    #[test]
    fn legacy_model_publisher_archive_reads_stay_legacy_and_cannot_enter_new_proof() {
        let previous = engine();
        let mut engine = PalsEngine::new_with_checker(
            previous.config,
            LegalOrderRoleMock,
            ModelOnlyForeignFixture {
                identity: CheckerIdentity::ExternalUci(foreign_identity()),
            },
        )
        .unwrap();
        assert!(!engine.followup_lane);
        assert_eq!(engine.resolver, ResolverPolicy::ModelWdlRestricted);
        engine
            .enable_archive(archive_config("legacy-model-reading"))
            .unwrap();
        let old = Position::startpos();
        let cancel = AtomicBool::new(false);
        let controls = limits();
        engine
            .prepare_archive_root(&old, controls.deadline, &cancel)
            .unwrap();
        engine.stores.focus_actual_moves(old.snapshot()).unwrap();
        let root = engine.intern(old.clone()).unwrap();
        engine
            .finish_archive_root(root, controls.deadline, &cancel)
            .unwrap();
        let mut counters = PalsCounters::default();
        engine
            .evaluate_model_value(root, &[], controls, &cancel, &mut counters)
            .unwrap();
        assert_eq!(counters.completed_value_calls, 1);
        assert_eq!(counters.cpu_tasks, 0);
        let state = engine.nodes[root].state;
        let observation = engine.nodes[root].model_observation.unwrap();
        assert!(matches!(
            engine.stores.observations.get(observation).unwrap().score,
            RawScore::Wdl { .. }
        ));
        fill_hot(&mut engine);
        let next = advanced_root();
        let controls = limits();
        engine
            .prepare_archive_root(&next, controls.deadline, &cancel)
            .unwrap();
        engine.stores.focus_actual_moves(next.snapshot()).unwrap();
        let next_root = engine.intern(next).unwrap();
        engine
            .finish_archive_root(next_root, controls.deadline, &cancel)
            .unwrap();
        assert!(matches!(
            engine.stores.states.get(state),
            Err(StoreError::ColdRecord(_))
        ));
        let restored = engine
            .load_archived_root(
                &old,
                ArchiveIoBudget {
                    deadline: Instant::now() + Duration::from_secs(20),
                    max_bytes: SEARCH_ARCHIVE_IO_BYTES,
                },
            )
            .unwrap();
        assert_eq!(engine.nodes[restored].model_observation, Some(observation));
        engine
            .validate_loaded_node_evidence(&engine.nodes[restored])
            .unwrap();
        assert!(matches!(
            engine.stores.observations.get(observation).unwrap().score,
            RawScore::Wdl { .. }
        ));
        engine.followup_lane = true;
        assert!(matches!(
            engine.validate_loaded_node_evidence(&engine.nodes[restored]),
            Err(PalsError::Store(StoreError::ArchiveIntegrity(
                "engine restored model observation"
            )))
        ));
        engine.followup_lane = false;
        engine.post_repair_recheck = PostRepairRecheckPolicy::FrozenModelWdlV2;
        assert!(matches!(
            engine.validate_loaded_node_evidence(&engine.nodes[restored]),
            Err(PalsError::Store(StoreError::ArchiveIntegrity(
                "engine restored model observation"
            )))
        ));
    }

    #[test]
    fn production_foreign_append_charges_cold_scan_and_propagates_io_failure() {
        let mut engine = engine();
        let config = archive_config("foreign-append");
        let root = config.root.clone();
        engine.enable_archive(config).unwrap();
        let cancel = AtomicBool::new(false);
        let old = Position::startpos();
        engine.search(&old, limits(), &cancel).unwrap();
        let raw = foreign_observation(&mut engine, &old, 1, 90_001);
        let accepted = engine.append_engine_observation(raw.clone()).unwrap();
        engine
            .stores
            .complete_task(raw.execution.unwrap(), accepted)
            .unwrap();
        fill_hot(&mut engine);
        let current = advanced_root();
        engine.search(&current, limits(), &cancel).unwrap();
        assert!(matches!(
            engine.stores.observations.get(accepted),
            Err(StoreError::ColdRecord(_))
        ));

        let mut incoming = foreign_observation(&mut engine, &current, 1, 90_002);
        let incoming_execution = incoming.execution.unwrap();
        let before = engine.stores.hot_stats();
        let available = engine.archive_state.remaining_io;
        assert!(matches!(
            engine.append_engine_observation(incoming.clone()),
            Err(PalsError::Store(StoreError::InvalidEvidence(
                "foreign physical request belongs to another archived task"
            )))
        ));
        assert!(engine.archive_state.remaining_io < available);
        assert_eq!(engine.stores.hot_stats(), before);
        engine.archive_state.remaining_io = 1;
        incoming.external_report.as_mut().unwrap().request_id = 2;
        assert!(matches!(
            engine.append_engine_observation(incoming.clone()),
            Err(PalsError::Store(StoreError::ArchiveByteBudget))
        ));
        assert_eq!(engine.stores.hot_stats(), before);

        // Fault only this test's unique archive directory. Restore it before
        // assertions so a primary failure never destroys retained evidence.
        engine.archive_state.remaining_io = SEARCH_ARCHIVE_IO_BYTES;
        let unavailable = root.with_extension("unavailable");
        std::fs::rename(&root, &unavailable).unwrap();
        let failed = engine.append_engine_observation(incoming);
        std::fs::rename(&unavailable, &root).unwrap();
        assert!(matches!(
            failed,
            Err(PalsError::Store(StoreError::ArchiveIo { .. }))
        ));
        assert_eq!(engine.stores.hot_stats(), before);
        assert!(matches!(
            engine.stores.tasks.get(raw.execution.unwrap()),
            Err(StoreError::ColdRecord(_))
        ));
        engine.resident_archive_pins().unwrap();
        assert_eq!(
            engine.stores.tasks.get(incoming_execution).unwrap().status,
            TaskStatus::InFlight
        );
    }

    #[test]
    fn archived_parent_restores_actual_descendants_after_two_pressure_transitions() {
        let mut engine = engine();
        engine
            .enable_archive(archive_config("split-descendants"))
            .unwrap();
        let cancel = AtomicBool::new(false);
        let a = Position::startpos();
        let first = engine.search(&a, limits(), &cancel).unwrap();
        assert!(first.counters.cpu_tasks > 0);
        let a_index = engine
            .nodes
            .iter()
            .position(|node| node.position.snapshot().same_state(&a.snapshot()))
            .unwrap();
        let a_state = engine.nodes[a_index].state;
        let b_index = engine.nodes[a_index].edges[0].child;
        let first_move = engine.nodes[a_index].edges[0].movement;
        let b = engine.nodes[b_index].position.clone();
        let b_state = engine.nodes[b_index].state;
        let b_evidence = engine.nodes[b_index]
            .evidence
            .as_ref()
            .map(|evidence| (evidence.scope, evidence.depth, evidence.provenance));
        let mut replay = a.clone();
        replay.make_move(first_move).unwrap();
        assert!(replay.snapshot().same_state(&b.snapshot()));

        fill_hot(&mut engine);
        let second = engine.search(&b, limits(), &cancel).unwrap();
        assert_ne!(second.completion, PalsCompletion::Capacity);
        assert!(matches!(
            engine.stores.states.get(a_state),
            Err(StoreError::ColdRecord(_))
        ));
        let b_index = engine
            .nodes
            .iter()
            .position(|node| node.state == b_state)
            .unwrap();
        let c = engine.nodes[engine.nodes[b_index].edges[0].child]
            .position
            .clone();
        fill_hot(&mut engine);
        let third = engine.search(&c, limits(), &cancel).unwrap();
        assert_ne!(third.completion, PalsCompletion::Capacity);
        assert!(matches!(
            engine.stores.states.get(b_state),
            Err(StoreError::ColdRecord(_))
        ));

        let restored = engine
            .load_archived_root(
                &a,
                ArchiveIoBudget {
                    deadline: Instant::now() + Duration::from_secs(20),
                    max_bytes: SEARCH_ARCHIVE_IO_BYTES,
                },
            )
            .unwrap();
        assert_eq!(engine.nodes[restored].state, a_state);
        let edge = engine.nodes[restored]
            .edges
            .iter()
            .find(|edge| edge.movement == first_move)
            .unwrap();
        let restored_b = &engine.nodes[edge.child];
        assert_eq!(restored_b.state, b_state);
        assert!(restored_b.position.snapshot().same_state(&b.snapshot()));
        assert_eq!(
            restored_b.evidence.as_ref().map(|evidence| (
                evidence.scope,
                evidence.depth,
                evidence.provenance
            )),
            b_evidence
        );
        engine.validate_loaded_node_evidence(restored_b).unwrap();
        let descendants = engine.archive_reachable_nodes(&b, false).unwrap();
        assert!(engine.nodes.iter().enumerate().any(|(index, node)| {
            descendants[index]
                && node
                    .evidence
                    .as_ref()
                    .is_some_and(|evidence| evidence.provenance.is_some())
        }));
    }

    #[test]
    fn archive_quota_failure_preserves_engine_ram_and_propagates_a_typed_error() {
        let mut engine = engine();
        let mut config = archive_config("quota");
        config.game_bytes = 4096;
        config.record_bytes = 4096;
        engine.enable_archive(config).unwrap();
        fill_hot(&mut engine);
        let states: Vec<_> = engine.nodes.iter().map(|node| node.state).collect();
        let error = engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap_err();
        assert!(matches!(
            error,
            PalsError::Store(StoreError::ArchiveQuota(_))
        ));
        assert_eq!(
            engine
                .nodes
                .iter()
                .map(|node| node.state)
                .collect::<Vec<_>>(),
            states
        );
        assert_eq!(engine.stores.hot_stats().states, 257);
        assert!(engine.last_engine_archive().is_none());
    }

    #[test]
    fn long_actual_move_sequence_stays_bounded_and_continues_new_root_work() {
        let mut engine = engine();
        engine.enable_archive(archive_config("long-moves")).unwrap();
        let mut position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let mut played = 0;
        let mut saw_archive = false;
        for ply in 0..48usize {
            if matches!(
                position.classify_position().unwrap().play_status,
                PlayStatus::Terminal { .. }
            ) {
                break;
            }
            let report = engine.search(&position, limits(), &cancel).unwrap();
            assert_ne!(report.completion, PalsCompletion::Capacity);
            assert!(
                report.counters.cpu_tasks > 0
                    || report.counters.reused_completed_cpu_tasks_consumed > 0
            );
            assert!(engine.retained_situations() <= engine.config.max_nodes);
            assert!(engine.records.len() <= engine.config.max_records);
            saw_archive |= engine.last_engine_archive().is_some();
            let legal = position.legal_moves();
            position
                .make_move(legal[(ply * 17 + 5) % legal.len()])
                .unwrap();
            played += 1;
        }
        assert!(played >= 24);
        assert!(saw_archive);
        assert!(engine.nodes.len() < 257);
    }

    #[test]
    fn new_game_reopens_a_unique_archive_owner_and_keeps_startup_config() {
        let mut engine = engine();
        let config = archive_config("new-game");
        let root = config.root.clone();
        engine.enable_archive(config).unwrap();
        fill_hot(&mut engine);
        engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap();
        let handle = engine
            .last_engine_archive()
            .unwrap()
            .archive
            .state_handle(StateId(0));
        let budget = ArchiveIoBudget {
            deadline: Instant::now() + Duration::from_secs(20),
            max_bytes: SEARCH_ARCHIVE_IO_BYTES,
        };
        engine.new_game();
        assert_eq!(engine.archive_config().unwrap().root, root);
        assert!(engine.archive_enabled());
        assert!(engine.last_engine_archive().is_none());
        assert!(matches!(
            engine.stores.pin_load(&handle, budget),
            Err(StoreError::InvalidHandle(_))
        ));
        let report = engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(report.counters.cpu_tasks > 0);
    }

    #[test]
    fn versioned_metadata_roundtrip_preserves_scopes_wdl_revision_and_rejects_corruption() {
        let mut engine = engine();
        let index = engine.intern(Position::startpos()).unwrap();
        let cpu_identity = engine.cpu_registered_value.clone().unwrap();
        let model_identity = engine.model.value_identity().unwrap().clone();
        let node = &mut engine.nodes[index];
        node.evidence = Some(CpuEvidence {
            score: 23,
            depth: 0,
            scope: CpuScoreScope::FrontierOnly,
            value_identity: cpu_identity,
            provenance: Some((ObservationId(9), ExecutionId(7))),
        });
        node.model_value = Some(ModelValueOutput {
            identity: model_identity,
            input_sha256: [17; 32],
            state: node.position.position_identity(),
            perspective: Color::White,
            wdl: [0.3, 0.4, 0.3],
        });
        node.model_observation = Some(ObservationId(11));
        node.model_value_revision = Some(55);
        let wire = encode_archive_node(&engine.nodes[index], &engine.nodes).unwrap();
        assert!(wire.metadata.len() <= 4096);
        let restored = decode_archive_node(&wire, Position::startpos(), wire.situation).unwrap();
        assert_eq!(
            restored.evidence.as_ref().unwrap().scope,
            CpuScoreScope::FrontierOnly
        );
        assert_eq!(restored.evidence.as_ref().unwrap().depth, 0);
        assert_eq!(
            restored.evidence.as_ref().unwrap().provenance,
            Some((ObservationId(9), ExecutionId(7)))
        );
        assert_eq!(restored.model_value.as_ref().unwrap().wdl, [0.3, 0.4, 0.3]);
        assert_eq!(restored.model_value_revision, Some(55));
        assert_eq!(restored.model_observation, Some(ObservationId(11)));
        let mut damaged = wire.clone();
        damaged.metadata[0] ^= 1;
        assert!(matches!(
            decode_archive_node(&damaged, Position::startpos(), wire.situation),
            Err(PalsError::Store(StoreError::ArchiveIntegrity(_)))
        ));
        damaged = wire.clone();
        damaged.metadata.pop();
        assert!(decode_archive_node(&damaged, Position::startpos(), wire.situation).is_err());
        damaged = wire;
        damaged.metadata.push(0);
        assert!(decode_archive_node(&damaged, Position::startpos(), damaged.situation).is_err());
    }
    #[test]
    fn midsearch_pressure_keeps_pending_terminal_and_engine_indices_without_compaction() {
        let mut engine = engine();
        engine.stores = PalsStores::new(StoreLimits {
            states: 4,
            situations: 4,
            line_chunks: 16,
            observations: 16,
            ..StoreLimits::default()
        });
        engine
            .enable_archive(archive_config("midsearch-terminal"))
            .unwrap();
        let position = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1").unwrap();
        let cancel = AtomicBool::new(false);
        let requested = limits();
        engine
            .prepare_archive_root(&position, requested.deadline, &cancel)
            .unwrap();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine
            .intern_checked(position.clone(), requested, &cancel)
            .unwrap();
        let resident = (engine.nodes[root].state, engine.nodes[root].situation);
        let generation = engine.stores.generation();
        let mut inactive = Vec::new();
        for fullmove in [1000, 1001] {
            let situation = engine
                .stores
                .insert_situation(inactive_position(fullmove).snapshot())
                .unwrap();
            inactive.push(engine.stores.situations.get(situation).unwrap().state);
        }
        assert_eq!(engine.stores.hot_stats().states, 3);
        let (movement, child) = position
            .legal_moves()
            .into_iter()
            .find_map(|movement| {
                let mut child = position.clone();
                child.make_move(movement).unwrap();
                matches!(
                    child.classify_position().unwrap().play_status,
                    PlayStatus::Terminal {
                        reason: TerminalReason::Checkmate,
                        ..
                    }
                )
                .then_some((movement, child))
            })
            .unwrap();
        let terminal = engine
            .connect_checked(
                root,
                movement,
                child.clone(),
                &mut PalsCounters::default(),
                requested,
                &cancel,
            )
            .unwrap();
        assert_eq!(root, 0);
        assert_eq!(terminal, 1);
        assert_eq!(engine.nodes.len(), 2);
        assert_eq!(
            (engine.nodes[root].state, engine.nodes[root].situation),
            resident
        );
        assert_eq!(engine.stores.generation(), generation);
        assert_eq!(engine.nodes[root].edges[0].child, terminal);
        assert_eq!(
            engine
                .stores
                .situations
                .get(engine.nodes[terminal].situation)
                .unwrap()
                .state,
            engine.nodes[terminal].state
        );
        assert!(
            engine
                .stores
                .states
                .get(engine.nodes[terminal].state)
                .unwrap()
                .same_state(&child.snapshot())
        );
        assert!(engine.nodes[terminal].terminal.is_some());
        assert!(inactive.iter().all(|state| matches!(
            engine.stores.states.get(*state),
            Err(StoreError::ColdRecord("state"))
        )));
        assert!(engine.archive_state.remaining_io < SEARCH_ARCHIVE_IO_BYTES);
        assert!(engine.last_engine_archive().is_none());
    }
    #[test]
    fn midsearch_active_raw_repair_handles_survive_pressure_until_next_root_prepare() {
        let mut engine = engine();
        engine.stores = PalsStores::new(StoreLimits {
            states: 8,
            situations: 8,
            line_chunks: 16,
            observations: 8,
            ..StoreLimits::default()
        });
        engine
            .enable_archive(archive_config("midsearch-raw-repair"))
            .unwrap();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let requested = limits();
        engine
            .prepare_archive_root(&position, requested.deadline, &cancel)
            .unwrap();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine
            .intern_checked(position.clone(), requested, &cancel)
            .unwrap();
        let resident = (engine.nodes[root].state, engine.nodes[root].situation);
        let mut held = Vec::new();
        for (kind, moves) in [
            (RecordKind::Proposal, ["d2d4", "d7d5"]),
            (RecordKind::Counterexample, ["e2e4", "c7c5"]),
            (RecordKind::Repair, ["e2e4", "e7e5"]),
        ] {
            let line: Vec<_> = moves
                .into_iter()
                .map(|movement| movement.parse().unwrap())
                .collect();
            held.push(
                engine
                    .record_checked(kind, &line, None, 0, None, None, requested, &cancel)
                    .unwrap()
                    .unwrap(),
            );
        }
        let mut inactive_observations = Vec::new();
        for fullmove in 1000..1005 {
            let situation = engine
                .stores
                .insert_situation(inactive_position(fullmove).snapshot())
                .unwrap();
            let (state, focus) = {
                let situation = engine.stores.situations.get(situation).unwrap();
                (situation.state, situation.focus)
            };
            inactive_observations.push(
                engine
                    .stores
                    .append_observation(unknown_observation(state, Some(focus)))
                    .unwrap(),
            );
        }
        assert_eq!(engine.stores.hot_stats().observations, 8);
        engine
            .append_engine_observation_with_controls(
                unknown_observation(resident.0, None),
                Some((requested.deadline, &cancel)),
            )
            .unwrap();
        assert_eq!(engine.nodes.len(), 1);
        assert_eq!(
            (engine.nodes[root].state, engine.nodes[root].situation),
            resident
        );
        for (line, observation) in held {
            assert!(!engine.stores.lines.moves(line).unwrap().is_empty());
            assert_eq!(
                engine.stores.observations.get(observation).unwrap().line,
                Some(line)
            );
            assert!(
                engine
                    .archive_state
                    .active_search_pins
                    .lines
                    .contains(&line)
            );
            assert!(
                engine
                    .archive_state
                    .active_search_pins
                    .observations
                    .contains(&observation)
            );
        }
        assert!(inactive_observations.iter().all(|observation| matches!(
            engine.stores.observations.get(*observation),
            Err(StoreError::ColdRecord("observation"))
        )));
        assert!(engine.archive_state.remaining_io < SEARCH_ARCHIVE_IO_BYTES);
        engine
            .prepare_archive_root(&position, requested.deadline, &cancel)
            .unwrap();
        assert!(engine.archive_state.active_search_pins.lines.is_empty());
        assert!(
            engine
                .archive_state
                .active_search_pins
                .observations
                .is_empty()
        );
    }
    #[test]
    fn canceled_completed_raw_gets_one_hot_append_without_record_or_node_acceptance() {
        let mut engine = engine();
        engine.enable_archive(archive_config("late-raw")).unwrap();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let requested = limits();
        engine
            .prepare_archive_root(&position, requested.deadline, &cancel)
            .unwrap();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine
            .intern_checked(position.clone(), requested, &cancel)
            .unwrap();
        let remaining = engine.archive_state.remaining_io;
        let revision = engine.revision;
        cancel.store(true, Ordering::Release);
        let raw = engine
            .append_engine_observation_with_controls(
                unknown_observation(engine.nodes[root].state, None),
                Some((requested.deadline, &cancel)),
            )
            .unwrap();
        assert!(matches!(
            engine.stores.observations.get(raw).unwrap().score,
            RawScore::Unknown
        ));
        assert_eq!(engine.archive_state.remaining_io, remaining);
        let movement = BoardMove::from_uci("e2e4").unwrap();
        let mut child = position;
        child.make_move(movement).unwrap();
        assert!(matches!(
            engine.connect_checked(
                root,
                movement,
                child,
                &mut PalsCounters::default(),
                requested,
                &cancel
            ),
            Err(PalsError::Role(RoleError::Canceled))
        ));
        assert!(matches!(
            engine.record_checked(
                RecordKind::Proposal,
                &[movement],
                None,
                0,
                None,
                None,
                requested,
                &cancel
            ),
            Err(PalsError::Role(RoleError::Canceled))
        ));
        assert_eq!(engine.nodes.len(), 1);
        assert!(engine.nodes[root].edges.is_empty());
        assert!(engine.records.is_empty());
        assert_eq!(engine.revision, revision);
    }
    #[test]
    fn live_node_pin_saturation_keeps_last_published_legal_move_and_returns_typed_failure() {
        let mut engine = engine();
        // Inject a small live arena after the normal validated constructor; the
        // actual checked search still uses the production allocation boundary.
        engine.config.max_nodes = 3;
        engine.config.line_plies = 1;
        engine
            .enable_archive(archive_config("midsearch-node-pins"))
            .unwrap();
        let position = Position::startpos();
        let requested = PalsLimits {
            max_rounds: 3,
            ..limits()
        };
        let mut published = Vec::new();
        let result = engine.search_with_progress(
            &position,
            requested,
            &AtomicBool::new(false),
            |movement| published.push(movement),
        );
        assert!(matches!(
            result,
            Err(PalsError::Store(StoreError::PinSaturated(
                "resident engine node arena"
            )))
        ));
        let last_valid = published
            .last()
            .copied()
            .expect("valid progress before the arena filled");
        assert!(position.legal_moves().contains(&last_valid));
        assert_eq!(engine.nodes.len(), 3);
        assert!(
            engine.nodes[0]
                .position
                .snapshot()
                .same_state(&position.snapshot())
        );
        for edge in &engine.nodes[0].edges {
            assert!(edge.child < engine.nodes.len());
            assert!(
                engine
                    .stores
                    .situations
                    .get(engine.nodes[edge.child].situation)
                    .is_ok()
            );
        }
        assert!(engine.last_engine_archive().is_none());
    }
    #[test]
    fn canceled_foreign_first_hot_append_keeps_original_scan_budget_and_never_consumes() {
        let mut engine = engine();
        engine
            .enable_archive(archive_config("late-foreign"))
            .unwrap();
        let cancel = AtomicBool::new(false);
        let old = Position::startpos();
        engine.search(&old, limits(), &cancel).unwrap();
        let previous = foreign_observation(&mut engine, &old, 10_001, 91_001);
        let observation = engine.append_engine_observation(previous.clone()).unwrap();
        engine
            .stores
            .complete_task(previous.execution.unwrap(), observation)
            .unwrap();
        fill_hot(&mut engine);
        let position = advanced_root();
        let requested = limits();
        engine.search(&position, requested, &cancel).unwrap();
        assert!(matches!(
            engine.stores.observations.get(observation),
            Err(StoreError::ColdRecord(_))
        ));
        let incoming = foreign_observation(&mut engine, &position, 10_002, 91_002);
        let execution = incoming.execution.unwrap();
        let available = engine.archive_state.remaining_io;
        let generation = engine.stores.generation();
        let states: Vec<_> = engine.nodes.iter().map(|node| node.state).collect();
        cancel.store(true, Ordering::Release);
        let raw = engine
            .append_engine_observation_with_controls(
                incoming.clone(),
                Some((requested.deadline, &cancel)),
            )
            .unwrap();
        assert_eq!(
            engine.stores.observations.get(raw).unwrap().execution,
            Some(execution)
        );
        assert!(matches!(
            engine.stores.tasks.get(execution).unwrap().status,
            TaskStatus::InFlight
        ));
        assert!(engine.archive_state.remaining_io < available);
        assert_eq!(engine.stores.generation(), generation);
        assert_eq!(
            engine
                .nodes
                .iter()
                .map(|node| node.state)
                .collect::<Vec<_>>(),
            states
        );
        let before = engine.stores.hot_stats();
        let mut rejected = incoming;
        rejected.external_report.as_mut().unwrap().request_id = 10_003;
        engine.archive_state.deadline = Some(Instant::now());
        assert!(matches!(
            engine.append_engine_observation_with_controls(
                rejected,
                Some((requested.deadline, &cancel))
            ),
            Err(PalsError::Store(StoreError::ArchiveDeadline))
        ));
        assert_eq!(engine.stores.hot_stats(), before);
        assert!(matches!(
            engine.stores.tasks.get(execution).unwrap().status,
            TaskStatus::InFlight
        ));
    }

    #[test]
    fn post_allocation_task_guard_retires_new_reservations_and_preserves_joined_facts() {
        // Move the flag/deadline after a real Store admission deterministically.
        // These are typed TaskTable fixtures; no backend was dispatched and an
        // opaque pause marker makes no native stack or physical completion claim.
        for kind in ["start", "resume", "join", "reuse"] {
            for expired in [false, true] {
                let mut engine = engine();
                let root = engine.intern(Position::startpos()).unwrap();
                let state = engine.nodes[root].state;
                let situation = engine.nodes[root].situation;
                let generation = engine.stores.generation();
                let revision = engine.stores.situations.get(situation).unwrap().revision;
                let own = (kind == "resume").then(|| engine.cpu_registered_value.clone().unwrap());
                let key = TaskKey {
                    state,
                    line: None,
                    question: if own.is_some() {
                        TaskQuestion::AnalyzePosition
                    } else {
                        TaskQuestion::ModelProposal
                    },
                    root_moves: Vec::new(),
                    model: 1,
                    epoch: 0,
                    value_identity: own.clone(),
                    checker_identity: own.map(CheckerIdentity::Owned),
                    cpu_condition: (kind == "resume").then(|| "post-guard-fixture/1".into()),
                    profile: 1,
                    condition: 1,
                    input_revision: 0,
                    requested_depth: 1,
                    node_budget: 32,
                };
                let consumer = |id| TaskConsumer {
                    id,
                    situation,
                    revision,
                    generation,
                    deadline_tick: 100_000,
                };
                let TaskAdmission::Start(first) = engine
                    .stores
                    .request_task(key.clone(), consumer(1), 0)
                    .unwrap()
                else {
                    panic!("first fixture task must be a new reservation");
                };
                let completed = if kind == "reuse" {
                    let mut raw = unknown_observation(state, None);
                    raw.execution = Some(first);
                    let observation = engine.stores.append_observation(raw).unwrap();
                    engine.stores.complete_task(first, observation).unwrap();
                    Some(observation)
                } else {
                    None
                };
                if kind == "resume" {
                    engine.stores.pause_task(first, 17, None).unwrap();
                }
                let (admission, consumer_id) = if kind == "start" {
                    (TaskAdmission::Start(first), 1)
                } else {
                    (engine.stores.request_task(key, consumer(2), 0).unwrap(), 2)
                };
                let execution = match (kind, admission) {
                    ("start", TaskAdmission::Start(execution))
                    | ("join", TaskAdmission::Join(execution))
                    | ("reuse", TaskAdmission::Reuse { execution, .. }) => execution,
                    (
                        "resume",
                        TaskAdmission::Resume {
                            execution,
                            previous,
                            checkpoint,
                        },
                    ) => {
                        assert_eq!(previous, first);
                        assert_eq!(checkpoint, 17);
                        execution
                    }
                    _ => panic!("fixture admission did not match {kind}"),
                };
                let cancel = AtomicBool::new(!expired);
                let deadline = if expired {
                    Instant::now()
                } else {
                    Instant::now() + Duration::from_secs(20)
                };
                let result = engine.finalize_archive_task_admission(
                    Ok(admission),
                    Some(admission),
                    consumer_id,
                    Some((deadline, &cancel)),
                );
                if expired {
                    assert!(matches!(result, Err(PalsError::RoleDeadline)));
                } else {
                    assert!(matches!(result, Err(PalsError::RoleCanceled)));
                }
                assert!(engine.last_cpu_checkpoint_cleanup_error.is_none());
                match kind {
                    "start" | "resume" => {
                        assert_eq!(
                            engine.stores.tasks.get(execution).unwrap().status,
                            TaskStatus::Failed
                        );
                        if kind == "resume" {
                            assert_eq!(
                                engine.stores.tasks.get(first).unwrap().status,
                                TaskStatus::Paused {
                                    checkpoint: 17,
                                    evidence: None
                                }
                            );
                        }
                    }
                    "join" | "reuse" => {
                        let observation = if let Some(observation) = completed {
                            assert_eq!(
                                engine.stores.tasks.get(first).unwrap().status,
                                TaskStatus::Completed(observation)
                            );
                            observation
                        } else {
                            assert_eq!(
                                engine.stores.tasks.get(first).unwrap().status,
                                TaskStatus::InFlight
                            );
                            let mut raw = unknown_observation(state, None);
                            raw.execution = Some(first);
                            let observation = engine.stores.append_observation(raw).unwrap();
                            engine.stores.complete_task(first, observation).unwrap();
                            observation
                        };
                        assert!(matches!(
                            engine.stores.consume_task(first, consumer_id, 0),
                            Err(StoreError::CancelledConsumer)
                        ));
                        assert_eq!(
                            engine.stores.consume_task(first, 1, 0).unwrap(),
                            observation
                        );
                        assert!(engine.stores.observations.get(observation).is_ok());
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}
