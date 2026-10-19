//! Exact archive maintenance while the original semantic owner has no root work.
//! This custody is not a physical fence. The concrete UCI owner must retain its
//! actual native/actor acknowledgements before lending either idle owner here.
use super::archive_lifecycle::{SEARCH_ARCHIVE_IO_BYTES, allocation_controls};
use super::preparing_root::ResearcherPreparingOwner;
use super::*;
use crate::pals::store::{
    ArchiveConfig, ArchiveIoBudget, ArchiveOwnerSnapshot, ArchiveRootCapability,
    ArchiveRuntimeLimits, StorePins,
};
use rz_contracts::pals::SearchAuthority;
use std::sync::Weak;

enum IdleOwner<M: RoleModel> {
    Researcher(Box<ResearcherPreparingOwner<M>>),
    Pc(Box<PcSemanticOwner<M>>),
}

/// One original heap owner moves through the existing one-slot archive worker.
/// No checker, TT, model execution, root authority or physical receipt is made.
pub struct IdleSemanticArchiveOwner<M: RoleModel> {
    owner: IdleOwner<M>,
    identity: Arc<()>,
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
        })
    }
}

impl<M: RoleModel> IdleSemanticArchiveOwner<M> {
    pub fn into_researcher(self) -> Result<Box<ResearcherPreparingOwner<M>>, Self> {
        match self.owner {
            IdleOwner::Researcher(owner) => Ok(owner),
            IdleOwner::Pc(_) => Err(self),
        }
    }
    pub fn into_pc(self) -> Result<Box<PcSemanticOwner<M>>, Self> {
        match self.owner {
            IdleOwner::Pc(owner) => Ok(owner),
            IdleOwner::Researcher(_) => Err(self),
        }
    }
    pub fn stores(&self) -> &PalsStores {
        &self.semantic().stores
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
