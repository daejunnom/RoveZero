//! Original startup source selection plus independent per-go root authority.

use super::*;

#[cfg(feature = "pals-adaptive")]
pub(super) struct RoventPcrRootPolicy {
    pub(super) lock: Arc<rz_experiments::RoventInputLock>,
    pub(super) policy: crate::rovent_root_admission::RoventRootAdmissionPolicy,
}

pub(super) struct RoventPcrRootAdmission {
    pub(super) budget: Digest,
    #[cfg(feature = "pals-adaptive")]
    pub(super) original: Option<Arc<crate::rovent_root_admission::RoventRootAdmission>>,
}
pub(super) enum RoventPcrRootSource {
    Legacy(Digest),
    #[cfg(feature = "pals-adaptive")]
    Original(crate::rovent_root_admission::source_seal::SourceSeal),
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// Startup-only opt-in. The original lock's exact workload is unchanged;
    /// DynamicGo is a distinct per-root domain over the same admitted sources.
    #[cfg(feature = "pals-adaptive")]
    pub fn register_rovent_root_policy(
        &mut self,
        lock: Arc<rz_experiments::RoventInputLock>,
        policy: crate::rovent_root_admission::RoventRootAdmissionPolicy,
    ) -> Result<(), SearchSessionFailure> {
        lock.validate()
            .map_err(|error| SearchSessionFailure::debug("RoventStartupLock", &error))?;
        let profile = self
            .rovent
            .as_ref()
            .ok_or_else(|| failure("RoventStartupPolicy", "no actual Rovent PCR profile"))?;
        let profile_lock: String = profile
            .input_lock
            .0
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if self.rovent_root_policy.is_some()
            || self.closed.load(Ordering::Acquire)
            || profile_lock != lock.canonical_sha256
            || profile.mode != lock.manifest.engine.mode
            || profile.maximum_goal_plies != lock.manifest.engine.search.registered_max_chess_plies
            || self
                .latest_rovent_root
                .lock()
                .map_err(|_| failure("RoventStartupPolicy", "root receipt poisoned"))?
                .is_some()
        {
            return Err(failure(
                "RoventStartupPolicy",
                "startup source/mode/policy differs or was already used",
            ));
        }
        let state = self.state.try_lock().map_err(|_| {
            failure(
                "RoventStartupPolicy",
                "startup root policy requires the original idle owner",
            )
        })?;
        if !matches!(&*state, SessionState::Idle(owners) if owners.continuation.is_none() && owners.native_finished.is_none())
        {
            return Err(failure(
                "RoventStartupPolicy",
                "root policy cannot change after physical work",
            ));
        }
        let SessionState::Idle(owners) = &*state else {
            unreachable!()
        };
        let ready = owners
            .native
            .ready_clock()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        if ready.domain() != self.clock.domain()
            || ready.origin() != self.clock.runtime_clock().origin()
            || owners.native.is_quarantined()
        {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pcr-startup-root-policy/1\0");
        hash.update(self.options.implementation.0);
        hash.update(lock.canonical_sha256.as_bytes());
        hash.update([match policy {
            crate::rovent_root_admission::RoventRootAdmissionPolicy::ExactRegistered => 0,
            crate::rovent_root_admission::RoventRootAdmissionPolicy::DynamicGo => 1,
        }]);
        hash.update(include_bytes!("root_admission.rs"));
        self.options.implementation = Digest(hash.finalize().into());
        self.rovent_root_policy = Some(RoventPcrRootPolicy { lock, policy });
        Ok(())
    }

    pub(super) fn prepare_rovent_go(
        &self,
        position: &Position,
        context: &SearchSessionContext,
    ) -> Result<Option<RoventPcrRootSource>, SearchSessionFailure> {
        let Some(profile) = &self.rovent else {
            return Ok(None);
        };
        #[cfg(feature = "pals-adaptive")]
        if let Some(registered) = &self.rovent_root_policy {
            let source = crate::rovent_root_admission::source_seal::SourceSeal::prepare(
                &registered.lock,
                registered.policy,
                position,
                context,
                &self.clock,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventRootAdmission", &error))?;
            return Ok(Some(RoventPcrRootSource::Original(source)));
        }
        Ok(Some(RoventPcrRootSource::Legacy(profile.root_budget)))
    }
}
