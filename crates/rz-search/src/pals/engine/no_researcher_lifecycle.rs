//! Logical game reset for the typed R-free semantic owner.
//!
//! The concrete lifecycle owner first proves its native role fence and releases
//! old view custody. These checks grant no physical completion by themselves.
//! This child of adaptive_root can preserve its original registry/nonce owner.
use super::*;

impl<M: RoleModel> PcSemanticOwner<M> {
    /// Off-manager logical reset after the caller's real native role fence.
    /// No researcher/worker/TT reset, fake identity or CPU receipt is created.
    pub fn try_adaptive_new_game(
        &mut self,
        next: SearchAuthority,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.semantic.ensure_archive_admission_open()?;
        self.semantic.ensure_adaptive_logical_quiescence()?;
        self.semantic.validate_adaptive_namespace(next)?;
        let old = self
            .semantic
            .game_generation
            .ok_or(StoreError::StaleConsumer)?;
        if next.game.0 <= old || self.semantic.adaptive_root_registry.pending_nonce.is_some() {
            return Err(StoreError::StaleConsumer.into());
        }
        if self.semantic.stores.archive_enabled() || self.semantic.archive_is_configured() {
            return Err(StoreError::InvalidConditions(
                "R-free archive game reset requires its original archive lifecycle owner",
            )
            .into());
        }
        allocation_controls(Some((deadline, cancel)))?;
        self.semantic.model.preflight_new_game(deadline, cancel)?;
        let next_store = PalsStores::new(self.profile.sequence_config().store_limits());
        allocation_controls(Some((deadline, cancel)))?;
        // Validation/preflight finishes before the old logical owner mutates.
        // The registry itself remains in place, including its monotonic nonce.
        self.semantic
            .model
            .new_game_with_generation(Some(next.game.0));
        self.semantic.nodes.clear();
        self.semantic.records.clear();
        self.semantic.reset_review_context();
        self.semantic.stores = next_store;
        self.semantic.game_generation = Some(next.game.0);
        self.semantic.adaptive_root_registry.new_game_ready = Some(next);
        self.semantic.revision = 0;
        self.semantic.consumer_id = 0;
        self.semantic.last_search_counters = None;
        self.semantic.last_recheck_observer_error = None;
        self.semantic.repair_queue.clear();
        self.semantic.repair_questions.clear();
        self.semantic.repair_counts.clear();
        self.semantic.new_game_transition_error = None;
        Ok(())
    }
}
