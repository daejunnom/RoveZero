//! Exact archive maintenance while the original semantic owner has no root work.
//! This custody is not a physical fence. The concrete UCI owner must retain its
//! actual native/actor acknowledgements before lending either idle owner here.
use super::adaptive_root::{IdleArchiveGameSeal, UnusedArchiveSetupSeal};
use super::archive_lifecycle::{
    IdleArchiveCheckpoint, SEARCH_ARCHIVE_IO_BYTES, allocation_controls,
    observe_selected_archive_usage,
};
use super::preparing_root::ResearcherPreparingOwner;
use super::*;
use crate::pals::store::{
    ArchiveConfig, ArchiveIoBudget, ArchiveOwnerSnapshot, ArchiveRootCapability,
    ArchiveRuntimeLimits, ArchiveStats, StorePins,
};
use rz_contracts::pals::SearchAuthority;
use std::sync::Weak;

enum IdleOwner<M: RoleModel> {
    Researcher(Box<ResearcherPreparingOwner<M>>),
    Pc(Box<PcSemanticOwner<M>>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum IdleArchiveTransitionKind {
    NewGame(SearchAuthority),
    Finish,
}
struct PendingIdleArchive {
    kind: IdleArchiveTransitionKind,
    seal: Option<IdleArchiveGameSeal>,
    store: crate::pals::store::StoreReadOwner,
    generation: u64,
    revision: u64,
    deadline: Instant,
    remaining_io: u64,
    preflight_complete: bool,
    checkpoint: Option<Arc<IdleArchiveCheckpoint>>,
    replacement: Option<Box<PalsStores>>,
    replacement_inherited: bool,
    replacement_observed: bool,
    completed: bool,
    completion_unknown: bool,
    #[cfg(test)]
    cancel_after_close: bool,
}

/// Read-only bounded transition facts. These grant no root, archive or physical
/// authority; a failed/closed transition remains inside its original wrapper.
#[derive(Clone, Debug)]
pub struct IdleArchiveTransitionSnapshot {
    pub next_authority: Option<SearchAuthority>,
    pub original_deadline: Instant,
    pub remaining_io_bytes: u64,
    pub checkpoint: Option<Arc<IdleArchiveCheckpoint>>,
    pub original_closed: Option<ArchiveOwnerSnapshot>,
    pub replacement: Option<ArchiveOwnerSnapshot>,
    pub completed: bool,
    pub completion_unknown: bool,
}

/// Attribution to the caller's checked registration, never root/file authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnusedIdleArchiveSetupIdentity {
    pub profile: rz_contracts::Digest,
    pub input_lock: rz_contracts::Digest,
    pub registration: rz_contracts::Digest,
}
pub struct UnusedIdleArchiveSetupControl {
    pub deadline: Instant,
    pub cancellation: Arc<AtomicBool>,
    pub max_io_bytes: u64,
}
/// Move-only ownership of the original held capability and finite first-go
/// control. Construction performs no archive I/O and issues no root authority.
pub struct UnusedIdleArchiveSetup {
    pub(super) config: ArchiveConfig,
    pub(super) capability: ArchiveRootCapability,
    pub(super) limits: ArchiveRuntimeLimits,
    pub(super) identity: UnusedIdleArchiveSetupIdentity,
    pub(super) control: UnusedIdleArchiveSetupControl,
}
impl UnusedIdleArchiveSetup {
    pub fn new(
        config: ArchiveConfig,
        capability: ArchiveRootCapability,
        limits: ArchiveRuntimeLimits,
        identity: UnusedIdleArchiveSetupIdentity,
        control: UnusedIdleArchiveSetupControl,
    ) -> Self {
        Self {
            config,
            capability,
            limits,
            identity,
            control,
        }
    }
    pub fn identity(&self) -> UnusedIdleArchiveSetupIdentity {
        self.identity
    }
    pub fn runtime_limits(&self) -> ArchiveRuntimeLimits {
        self.limits
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdleArchiveOwnerKind {
    Pc,
    Researcher,
}
#[derive(Clone, Debug)]
pub struct UnusedIdleArchiveSetupSnapshot {
    pub scope: SearchAuthority,
    pub identity: UnusedIdleArchiveSetupIdentity,
    pub original_deadline: Instant,
    pub remaining_io_bytes: Option<u64>,
    pub configuration_complete: bool,
    pub completion_unknown: bool,
}

/// One original heap owner moves through the existing one-slot archive worker.
/// No checker, TT, model execution, root authority or physical receipt is made.
pub struct IdleSemanticArchiveOwner<M: RoleModel> {
    owner: IdleOwner<M>,
    identity: Arc<()>,
    pending: Option<Box<PendingIdleArchive>>,
    last_checkpoint: Option<Arc<IdleArchiveCheckpoint>>,
    last_transition_remaining: Option<u64>,
    setup: Option<UnusedIdleArchiveSetupSnapshot>,
    #[cfg(test)]
    panic_setup_after_enable: bool,
    #[cfg(test)]
    cancel_after_close: bool,
}

impl<M: RoleModel> ResearcherPreparingOwner<M> {
    #[allow(clippy::result_large_err)]
    pub fn into_idle_archive(
        self: Box<Self>,
    ) -> Result<IdleSemanticArchiveOwner<M>, (Box<Self>, PalsError)> {
        if let Err(error) = self.semantic.ensure_adaptive_logical_quiescence() {
            return Err((self, error));
        }
        Ok(IdleSemanticArchiveOwner {
            owner: IdleOwner::Researcher(self),
            identity: Arc::new(()),
            pending: None,
            last_checkpoint: None,
            last_transition_remaining: None,
            setup: None,
            #[cfg(test)]
            panic_setup_after_enable: false,
            #[cfg(test)]
            cancel_after_close: false,
        })
    }
}
impl<M: RoleModel> PcSemanticOwner<M> {
    #[allow(clippy::result_large_err)]
    pub fn into_idle_archive(
        self: Box<Self>,
    ) -> Result<IdleSemanticArchiveOwner<M>, (Box<Self>, PalsError)> {
        if let Err(error) = self.semantic.ensure_adaptive_logical_quiescence() {
            return Err((self, error));
        }
        Ok(IdleSemanticArchiveOwner {
            owner: IdleOwner::Pc(self),
            identity: Arc::new(()),
            pending: None,
            last_checkpoint: None,
            last_transition_remaining: None,
            setup: None,
            #[cfg(test)]
            panic_setup_after_enable: false,
            #[cfg(test)]
            cancel_after_close: false,
        })
    }
}

impl<M: RoleModel> IdleSemanticArchiveOwner<M> {
    // Rejected kind/setup transitions return the complete original linear
    // owner. Keep this error move-only without a new allocation or lost custody.
    #[allow(clippy::result_large_err)]
    pub fn into_researcher(self) -> Result<Box<ResearcherPreparingOwner<M>>, Self> {
        if self.pending.is_some() || self.setup_retained() {
            return Err(self);
        }
        match self.owner {
            IdleOwner::Researcher(owner) => Ok(owner),
            IdleOwner::Pc(_) => Err(self),
        }
    }
    #[allow(clippy::result_large_err)]
    pub fn into_pc(self) -> Result<Box<PcSemanticOwner<M>>, Self> {
        if self.pending.is_some() || self.setup_retained() {
            return Err(self);
        }
        match self.owner {
            IdleOwner::Pc(owner) => Ok(owner),
            IdleOwner::Researcher(_) => Err(self),
        }
    }
    pub fn stores(&self) -> &PalsStores {
        &self.semantic().stores
    }
    pub fn setup_snapshot(&self) -> Option<&UnusedIdleArchiveSetupSnapshot> {
        self.setup.as_ref()
    }
    pub fn owner_kind(&self) -> IdleArchiveOwnerKind {
        match &self.owner {
            IdleOwner::Pc(_) => IdleArchiveOwnerKind::Pc,
            IdleOwner::Researcher(_) => IdleArchiveOwnerKind::Researcher,
        }
    }
    fn setup_retained(&self) -> bool {
        self.setup
            .as_ref()
            .is_some_and(|state| !state.configuration_complete || state.completion_unknown)
    }
    #[cfg(test)]
    pub(super) fn panic_setup_after_enable_for_test(&mut self) {
        self.panic_setup_after_enable = true;
    }
    pub(super) fn seal_unused_setup(
        &self,
        scope: SearchAuthority,
    ) -> Result<UnusedArchiveSetupSeal, PalsError> {
        self.validate_idle()?;
        let semantic = self.semantic();
        semantic.stores.validate_unused_archive_setup()?;
        if self.setup.is_some()
            || !semantic.nodes.is_empty()
            || !semantic.records.is_empty()
            || semantic.revision != 0
            || semantic.consumer_id != 0
            || !semantic.repair_queue.is_empty()
            || !semantic.repair_questions.is_empty()
            || !semantic.repair_counts.is_empty()
            || semantic.archive_state.config.is_some()
            || semantic.archive_state.root_capability.is_some()
            || semantic.archive_state.previous_closed_owner.is_some()
            || semantic.archive_state.closed_current_owner.is_some()
            || semantic.archive_state.deadline.is_some()
            || semantic.archive_state.remaining_io != 0
            || semantic.archive_state.active_search_pins != StorePins::default()
            || semantic.last_engine_archive().is_some()
            || self.last_checkpoint.is_some()
            || self.last_transition_remaining.is_some()
            || semantic.review_index.as_ref().is_none_or(|index| {
                index.active_bundle_count() != 0 || index.indexed_record_count() != 0
            })
        {
            return Err(
                StoreError::InvalidConditions("archive setup needs unused semantic owner").into(),
            );
        }
        semantic.seal_unused_archive_setup(scope)
    }
    pub(super) fn execute_unused_setup(
        &mut self,
        scope: SearchAuthority,
        seal: &UnusedArchiveSetupSeal,
        setup: &UnusedIdleArchiveSetup,
    ) -> Result<(), PalsError> {
        self.seal_unused_setup(scope)?;
        seal.validate(self.semantic())?;
        self.setup = Some(UnusedIdleArchiveSetupSnapshot {
            scope,
            identity: setup.identity,
            original_deadline: setup.control.deadline,
            remaining_io_bytes: Some(setup.control.max_io_bytes),
            configuration_complete: false,
            completion_unknown: false,
        });
        allocation_controls(Some((setup.control.deadline, &setup.control.cancellation)))?;
        let semantic = self.semantic_mut();
        semantic
            .stores
            .enable_archive_with_capability(setup.config.clone(), setup.capability.clone())?;
        semantic.archive_state.config = Some(setup.config.clone());
        semantic.archive_state.root_capability = Some(setup.capability.clone());
        #[cfg(test)]
        if std::mem::take(&mut self.panic_setup_after_enable) {
            panic!("authored interruption after original archive enable");
        }
        let semantic = self.semantic_mut();
        semantic.stores.set_archive_runtime_limits(setup.limits)?;
        let config = semantic
            .archive_state
            .config
            .as_mut()
            .expect("original setup config");
        config.game_bytes = setup.limits.game_bytes_max;
        config.global_bytes = setup.limits.global_bytes_max;
        config.record_bytes = setup.limits.record_payload_bytes_max;
        config.max_load_pins = setup.limits.max_load_pins as usize;
        allocation_controls(Some((setup.control.deadline, &setup.control.cancellation)))?;
        let maximum_deadline = Instant::now()
            .checked_add(Duration::from_millis(setup.limits.load_deadline_max_ms))
            .ok_or(StoreError::InvalidConditions("archive setup load duration"))?;
        let mut budget = ArchiveIoBudget {
            deadline: setup.control.deadline.min(maximum_deadline),
            max_bytes: setup.control.max_io_bytes.min(setup.limits.load_bytes_max),
        };
        let before = budget.max_bytes;
        let result = self
            .semantic_mut()
            .stores
            .measure_archive_usage_with_budget(&mut budget);
        let remaining = setup
            .control
            .max_io_bytes
            .checked_sub(before.saturating_sub(budget.max_bytes))
            .ok_or(StoreError::ArchiveByteBudget)?;
        self.setup
            .as_mut()
            .expect("original setup state")
            .remaining_io_bytes = Some(remaining);
        result?;
        allocation_controls(Some((setup.control.deadline, &setup.control.cancellation)))?;
        seal.validate(self.semantic())?;
        let actual = self
            .archive_owner_snapshot()?
            .ok_or(StoreError::ArchiveDisabled)?;
        if !actual.complete
            || actual.admission_closed
            || actual.runtime_limits != Some(setup.limits)
        {
            return Err(
                StoreError::InvalidConditions("archive setup observation incomplete").into(),
            );
        }
        self.setup
            .as_mut()
            .expect("original setup state")
            .configuration_complete = true;
        Ok(())
    }
    pub(super) fn retain_unused_setup_failure(&mut self) {
        if let Some(state) = &mut self.setup {
            state.configuration_complete = false;
        }
    }
    pub(super) fn quarantine_unused_setup_panic(
        &mut self,
        scope: SearchAuthority,
        setup: &UnusedIdleArchiveSetup,
    ) {
        let state = self.setup.get_or_insert(UnusedIdleArchiveSetupSnapshot {
            scope,
            identity: setup.identity,
            original_deadline: setup.control.deadline,
            remaining_io_bytes: None,
            configuration_complete: false,
            completion_unknown: true,
        });
        state.configuration_complete = false;
        state.completion_unknown = true;
        state.remaining_io_bytes = None;
    }
    pub fn archive_enabled(&self) -> bool {
        self.stores().archive_enabled()
    }
    pub fn archive_owner_snapshot(&self) -> Result<Option<ArchiveOwnerSnapshot>, PalsError> {
        Ok(self
            .stores()
            .archive_owner_snapshot()?
            .or(self.semantic().archive_state.closed_current_owner))
    }
    /// The concrete caller already owns its physical fence. This closes only
    /// the original Store archive and preserves its closed observation.
    pub fn close_archive_owner(&mut self) -> Result<Option<ArchiveOwnerSnapshot>, PalsError> {
        self.validate_idle()?;
        if self.stores().hot_stats() != ArchiveStats::default() {
            return Err(StoreError::InvalidConditions(
                "hot idle game requires its final checkpoint before close",
            )
            .into());
        }
        let semantic = self.semantic_mut();
        let closed = semantic.stores.close_archive_owner()?;
        semantic.archive_state.closed_current_owner = closed;
        Ok(closed)
    }
    /// Same read-only maintenance namespace as the existing engine adapter.
    /// The next root is still registered only by its original owner afterwards.
    pub fn adaptive_archive_authority(
        &self,
        next: SearchAuthority,
    ) -> Result<SearchAuthority, PalsError> {
        self.semantic().adaptive_archive_authority(next)
    }
    /// Explicit selected storage only. Call on a controlled startup worker;
    /// this may open the original held archive root and never enables itself.
    pub fn enable_archive_with_capability(
        &mut self,
        config: ArchiveConfig,
        capability: ArchiveRootCapability,
    ) -> Result<(), PalsError> {
        self.validate_idle()?;
        let semantic = self.semantic_mut();
        semantic
            .stores
            .enable_archive_with_capability(config.clone(), capability.clone())?;
        semantic.archive_state.config = Some(config);
        semantic.archive_state.root_capability = Some(capability);
        semantic.archive_state.reopen_error = None;
        semantic.archive_state.closed_current_owner = None;
        Ok(())
    }
    pub fn set_archive_runtime_limits(
        &mut self,
        limits: ArchiveRuntimeLimits,
    ) -> Result<(), PalsError> {
        self.validate_idle()?;
        let semantic = self.semantic_mut();
        semantic.stores.set_archive_runtime_limits(limits)?;
        let actual =
            *semantic
                .stores
                .archive_runtime_limits()
                .ok_or(StoreError::InvalidConditions(
                    "idle archive selector missing",
                ))?;
        if let Some(config) = &mut semantic.archive_state.config {
            config.game_bytes = actual.game_bytes_max;
            config.global_bytes = actual.global_bytes_max;
            config.record_bytes = actual.record_payload_bytes_max;
            config.max_load_pins = actual.max_load_pins as usize;
        }
        Ok(())
    }
    pub(super) fn semantic(&self) -> &PalsSemanticState<M> {
        match &self.owner {
            IdleOwner::Researcher(owner) => &owner.semantic,
            IdleOwner::Pc(owner) => &owner.semantic,
        }
    }
    fn semantic_mut(&mut self) -> &mut PalsSemanticState<M> {
        match &mut self.owner {
            IdleOwner::Researcher(owner) => &mut owner.semantic,
            IdleOwner::Pc(owner) => &mut owner.semantic,
        }
    }
    pub(super) fn identity(&self) -> Weak<()> {
        Arc::downgrade(&self.identity)
    }
    fn config(&self) -> PalsSemanticConfig {
        match &self.owner {
            IdleOwner::Researcher(owner) => owner.profile().sequence_config(),
            IdleOwner::Pc(owner) => owner.profile.sequence_config(),
        }
    }
    fn cpu_value(&self) -> Option<&CpuValueIdentity> {
        match &self.owner {
            IdleOwner::Researcher(owner) => Some(owner.source().model().identity()),
            IdleOwner::Pc(_) => None,
        }
    }
    pub(super) fn validate_idle(&self) -> Result<(), PalsError> {
        if self.pending.is_some() || self.setup_retained() {
            return Err(
                StoreError::InvalidConditions("idle archive transition is retained").into(),
            );
        }
        if let Some(error) = &self.semantic().archive_state.reopen_error {
            return Err(PalsError::Store(error.clone()));
        }
        self.semantic().ensure_archive_admission_open()?;
        self.semantic().ensure_adaptive_logical_quiescence()
    }
    pub(super) fn projection_pins(&self) -> Result<StorePins, PalsError> {
        match &self.semantic().review_index {
            Some(index) => Ok(index.archive_projection_pins(self.stores())?),
            None => Err(PalsError::InvalidConfig),
        }
    }
    pub(super) fn prepare_root(
        &mut self,
        position: &Position,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.validate_idle()?;
        allocation_controls(Some((deadline, cancel)))?;
        if !self.archive_enabled() {
            return Err(StoreError::ArchiveDisabled.into());
        }
        let config = self.config();
        let pins = self.projection_pins()?;
        let semantic = self.semantic_mut();
        semantic.archive_state.deadline = Some(deadline);
        semantic.archive_state.remaining_io = SEARCH_ARCHIVE_IO_BYTES;
        semantic.archive_state.retried_allocation = false;
        semantic.archive_state.active_search_pins = StorePins::default();
        if semantic.stores.archive_runtime_limits().is_some() {
            let mut budget = semantic.engine_archive_budget()?;
            let before = budget.max_bytes;
            let result = semantic
                .stores
                .measure_archive_usage_with_budget(&mut budget);
            semantic.charge_engine_archive_io(before.saturating_sub(budget.max_bytes))?;
            result?;
        }
        semantic
            .stores
            .set_archive_pins(semantic.resident_archive_pins_with(&pins)?)?;
        if semantic.engine_archive_pressure(&config, position) {
            // Idle validation rejects every resume token, active task and view.
            semantic.reclaim_archive_for_root(
                &config,
                &pins,
                &|_| false,
                position,
                deadline,
                cancel,
            )?;
        }
        allocation_controls(Some((deadline, cancel)))?;
        Ok(())
    }
    pub(super) fn remaining_io(&self) -> u64 {
        self.semantic().archive_remaining_io()
    }

    pub fn last_checkpoint(&self) -> Option<Arc<IdleArchiveCheckpoint>> {
        self.pending
            .as_ref()
            .and_then(|pending| pending.checkpoint.clone())
            .or_else(|| self.last_checkpoint.clone())
    }
    pub fn transition_snapshot(&self) -> Result<Option<IdleArchiveTransitionSnapshot>, PalsError> {
        self.pending
            .as_ref()
            .map(|pending| {
                Ok(IdleArchiveTransitionSnapshot {
                    next_authority: match pending.kind {
                        IdleArchiveTransitionKind::NewGame(next) => Some(next),
                        IdleArchiveTransitionKind::Finish => None,
                    },
                    original_deadline: pending.deadline,
                    remaining_io_bytes: pending.remaining_io,
                    checkpoint: pending.checkpoint.clone(),
                    original_closed: self.semantic().archive_state.closed_current_owner,
                    replacement: pending
                        .replacement
                        .as_ref()
                        .map(|store| store.archive_owner_snapshot())
                        .transpose()?
                        .flatten(),
                    completed: pending.completed,
                    completion_unknown: pending.completion_unknown,
                })
            })
            .transpose()
    }
    pub(super) fn transition_remaining_io(&self) -> u64 {
        self.pending
            .as_ref()
            .map(|pending| pending.remaining_io)
            .or(self.last_transition_remaining)
            .unwrap_or(SEARCH_ARCHIVE_IO_BYTES)
    }
    pub(super) fn pending_transition_io(&self) -> Option<u64> {
        self.pending.as_ref().map(|pending| pending.remaining_io)
    }
    pub(super) fn quarantine_transition_unknown(&mut self) {
        if let Some(pending) = &mut self.pending {
            pending.completion_unknown = true;
            // A panic can interrupt a real commit before its debit returns.
            // Keep the original owners and reserve, never refund that allowance.
            pending.remaining_io = 0;
        }
    }
    #[cfg(test)]
    pub(super) fn cancel_after_close_for_test(&mut self) {
        self.cancel_after_close = true;
    }
    pub(super) fn validate_transition_request(
        &self,
        kind: IdleArchiveTransitionKind,
        deadline: Instant,
    ) -> Result<(), PalsError> {
        self.semantic().ensure_adaptive_logical_quiescence()?;
        if let Some(pending) = &self.pending {
            if pending.kind != kind
                || deadline > pending.deadline
                || pending.completed
                || pending.completion_unknown
                || !pending.store.matches(self.stores())
                || pending.generation != self.stores().generation()
                || pending.revision != self.semantic().revision
            {
                return Err(StoreError::StaleConsumer.into());
            }
            if let Some(seal) = &pending.seal {
                seal.validate(self.semantic())?;
            }
        } else {
            self.validate_idle()?;
            if let IdleArchiveTransitionKind::NewGame(next) = kind {
                self.semantic().validate_idle_new_game_target(next)?;
            }
        }
        Ok(())
    }
    pub(super) fn advance_transition(
        &mut self,
        kind: IdleArchiveTransitionKind,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.validate_transition_request(kind, deadline)?;
        allocation_controls(Some((deadline, cancel)))?;
        if self.pending.is_none() {
            let seal = match kind {
                IdleArchiveTransitionKind::NewGame(next) => {
                    Some(self.semantic().validate_idle_new_game_target(next)?)
                }
                IdleArchiveTransitionKind::Finish => None,
            };
            self.pending = Some(Box::new(PendingIdleArchive {
                kind,
                seal,
                store: self.stores().read_owner(),
                generation: self.stores().generation(),
                revision: self.semantic().revision,
                deadline,
                remaining_io: SEARCH_ARCHIVE_IO_BYTES,
                preflight_complete: false,
                checkpoint: None,
                replacement: None,
                replacement_inherited: false,
                replacement_observed: false,
                completed: false,
                completion_unknown: false,
                #[cfg(test)]
                cancel_after_close: std::mem::take(&mut self.cancel_after_close),
            }));
        }
        let config = self.config();
        // Borrow separate heap-owned slots; panic/error cannot drop a local
        // replacement or make the original transition lose its remaining bound.
        let semantic = match &mut self.owner {
            IdleOwner::Researcher(owner) => &mut owner.semantic,
            IdleOwner::Pc(owner) => &mut owner.semantic,
        };
        let pending = self.pending.as_mut().expect("original transition slot");
        advance_pending(semantic, &config, pending, deadline, cancel)?;
        self.last_checkpoint = pending.checkpoint.clone();
        self.last_transition_remaining = Some(pending.remaining_io);
        if matches!(kind, IdleArchiveTransitionKind::NewGame(_)) {
            self.pending = None;
        }
        Ok(())
    }

    pub(super) fn restore_root_if_archived(
        &mut self,
        position: &Position,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<usize>, PalsError> {
        self.validate_idle()?;
        let config = self.config();
        let cpu = self.cpu_value().cloned();
        let pins = self.projection_pins()?;
        let semantic = self.semantic_mut();
        semantic
            .stores
            .set_archive_pins(semantic.resident_archive_pins_with(&pins)?)?;
        let result = semantic.load_archived_root_if_present_with_budget(
            &config,
            cpu.as_ref(),
            false,
            position,
            budget,
        );
        let restore = semantic
            .resident_archive_pins_with(&pins)
            .and_then(|pins| semantic.stores.set_archive_pins(pins).map_err(Into::into));
        match result {
            Err(error) => Err(error),
            Ok(root) => {
                restore?;
                Ok(root)
            }
        }
    }
    pub(super) fn load_root(
        &mut self,
        position: &Position,
        budget: &mut ArchiveIoBudget,
    ) -> Result<usize, PalsError> {
        self.validate_idle()?;
        let config = self.config();
        let cpu = self.cpu_value().cloned();
        let pins = self.projection_pins()?;
        let semantic = self.semantic_mut();
        semantic
            .stores
            .set_archive_pins(semantic.resident_archive_pins_with(&pins)?)?;
        // Rovent never accepts a legacy Wdl observation as ContextWdl.
        let result =
            semantic.load_archived_root_with_budget(&config, cpu.as_ref(), false, position, budget);
        let restored_pins = semantic
            .resident_archive_pins_with(&pins)
            .and_then(|restored| {
                semantic
                    .stores
                    .set_archive_pins(restored)
                    .map_err(Into::into)
            });
        match result {
            Err(error) => Err(error),
            Ok(root) => {
                restored_pins?;
                Ok(root)
            }
        }
    }
}

fn advance_pending<M: RoleModel>(
    semantic: &mut PalsSemanticState<M>,
    config: &PalsSemanticConfig,
    pending: &mut PendingIdleArchive,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<(), PalsError> {
    let deadline = deadline.min(pending.deadline);
    allocation_controls(Some((deadline, cancel)))?;
    if !pending.preflight_complete {
        if matches!(pending.kind, IdleArchiveTransitionKind::NewGame(_)) {
            semantic.model.preflight_new_game(deadline, cancel)?;
        }
        pending.preflight_complete = true;
    }
    if pending.checkpoint.is_none() {
        let mut budget = ArchiveIoBudget {
            deadline,
            max_bytes: pending.remaining_io,
        };
        let result = semantic.checkpoint_idle_archive_game(&mut budget);
        pending.remaining_io = budget.max_bytes;
        pending.checkpoint = Some(Arc::new(result?));
    }
    allocation_controls(Some((deadline, cancel)))?;
    if matches!(pending.kind, IdleArchiveTransitionKind::NewGame(_)) {
        if pending.replacement.is_none() {
            // Put the unused Store in its retained slot before any fallible I/O.
            pending.replacement = Some(Box::new(PalsStores::new(config.store_limits())));
        }
        let next = pending
            .replacement
            .as_mut()
            .expect("original unused replacement");
        if !next.archive_enabled() {
            let archive = semantic
                .archive_state
                .config
                .as_ref()
                .ok_or(StoreError::ArchiveDisabled)?
                .clone();
            let held = semantic.archive_state.root_capability.as_ref().ok_or(
                StoreError::InvalidConditions("idle newgame needs original held archive root"),
            )?;
            next.enable_archive_with_capability(archive, held.clone())?;
        }
        if next.archive_runtime_limits().is_none() {
            if let Some(limits) = semantic.stores.archive_runtime_limits().copied() {
                next.set_archive_runtime_limits(limits)?;
            }
        }
        if !pending.replacement_inherited {
            next.inherit_archive_write_budget_from(&semantic.stores)?;
            pending.replacement_inherited = true;
        }
        if !pending.replacement_observed {
            // This existing finite metadata scan has zero payload-byte debit.
            // It retains the original deadline and selected read limits.
            observe_selected_archive_usage(next, deadline, cancel)?;
            pending.replacement_observed = true;
        }
    }
    allocation_controls(Some((deadline, cancel)))?;
    if semantic.archive_state.closed_current_owner.is_none() {
        semantic.archive_state.closed_current_owner = semantic.stores.close_archive_owner()?;
    }
    #[cfg(test)]
    if std::mem::take(&mut pending.cancel_after_close) {
        // Schedule a control change at the real irreversible close, without a
        // fake close result, sleep, replacement receipt or public authority.
        cancel.store(true, Ordering::Release);
    }
    // Closing cannot be undone. A control change here keeps this same original
    // closed Store, checkpoint and replacement in pending, without a new budget.
    allocation_controls(Some((deadline, cancel)))?;
    if matches!(pending.kind, IdleArchiveTransitionKind::NewGame(_)) {
        semantic.commit_idle_new_game_scope(
            pending.seal.as_ref().expect("original newgame seal"),
            &mut pending.replacement,
        )?;
    }
    pending.completed = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparing_idle_transfer_preserves_actual_source_and_rejects_a_live_view() {
        let owner = Box::new(super::super::preparing_root::tests::owner());
        let store = owner.stores().read_owner();
        let model = Arc::clone(owner.source().model());
        let pin = owner.stores().pin_review(StorePins::default()).unwrap();
        let (owner, error) = owner
            .into_idle_archive()
            .err()
            .expect("live original view rejected");
        assert!(matches!(
            error,
            PalsError::Store(StoreError::InvalidConditions(_))
        ));
        assert!(store.matches(owner.stores()));
        assert!(Arc::ptr_eq(owner.source().model(), &model));
        assert_eq!(owner.stores().review_pin_count().unwrap(), 1);
        drop(pin);
        let idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("idle original owner"));
        let idle = idle
            .into_pc()
            .err()
            .expect("wrong typed owner returned intact");
        let owner = idle
            .into_researcher()
            .unwrap_or_else(|_| panic!("original researcher owner"));
        assert!(store.matches(owner.stores()));
        assert!(Arc::ptr_eq(owner.source().model(), &model));
        assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
        assert!(!owner.stores().archive_enabled());
    }
}
