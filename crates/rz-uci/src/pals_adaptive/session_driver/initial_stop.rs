//! Normal initial control stops retain the actual idle process owners.
//!
//! This opt-in helper is not a physical fence producer. Its private custody is
//! created by the actual startup/clean-transfer paths, and the same idle mutex
//! remains held while a stopped source is acknowledged. No root registration,
//! model reset, selected view, CPU thread or ledger is made here.

use super::*;
use crate::rovent_root_admission::source_seal::StoppedInitial;
use rz_runtime::contracts::ContractClock as _;

pub const ROVENT_PCR_INITIAL_DOMAIN: &str = "rz-rovent-pcr-native-initial-stop/1";

#[derive(Clone, Debug)]
pub(super) enum RoventPcrIdleCustody {
    StartupReady,
    RetiredRoot(AdaptiveShutdownReceipt),
}

/// Producer-only evidence for a Start that did not create live root work.
/// Actual prior native/CPU evidence is retained, rather than charged to this
/// Start or replaced with a fabricated zero-work shutdown receipt.
#[derive(Clone, Debug)]
pub struct RoventPcrInitialReceipt {
    authority: SearchAuthority,
    source: Arc<StoppedInitial>,
    idle_custody: RoventPcrIdleCustody,
}
impl RoventPcrInitialReceipt {
    pub fn domain(&self) -> &'static str {
        ROVENT_PCR_INITIAL_DOMAIN
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn source(&self) -> &Arc<StoppedInitial> {
        &self.source
    }
    pub fn previous_clean_root(&self) -> Option<&AdaptiveShutdownReceipt> {
        match &self.idle_custody {
            RoventPcrIdleCustody::StartupReady => None,
            RoventPcrIdleCustody::RetiredRoot(receipt) => Some(receipt),
        }
    }
    pub fn used_original_startup_ready(&self) -> bool {
        matches!(self.idle_custody, RoventPcrIdleCustody::StartupReady)
    }
    fn report(&self) -> SearchSessionReport {
        SearchSessionReport {
            best_move: self.source.fallback(),
            work: DriverWork::RoventNative {
                root_entries: 0,
                total_node_visits: Some(0),
                checked_transitions: 0,
                native_nodes: Some(0),
                native_qsearch_nodes: Some(0),
                submitted_role_tickets: 0,
                consumed_role_outputs: 0,
                completed_nn_inputs: Some(0),
                consumed_nn_inputs: Some(0),
                completed_depth: None,
                maximum_examined_ply: 0,
            },
            mate_strategy: None,
        }
    }
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub fn latest_rovent_initial(
        &self,
    ) -> Result<Option<Arc<RoventPcrInitialReceipt>>, ContractError> {
        self.latest_rovent_initial
            .lock()
            .map(|receipt| receipt.clone())
            .map_err(|_| poisoned("Rovent initial receipt owner poisoned"))
    }

    /// Caller must hold the original SessionState mutex and borrow its exact
    /// Idle owners. The source contains no live authority to start new work.
    pub(super) fn finish_rovent_initial(
        &self,
        owners: &IdleOwners<B>,
        context: &SearchSessionContext,
        stopped: StoppedInitial,
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        let original = &self.clock.runtime_clock();
        let ready = owners
            .native
            .ready_clock()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        if self.closed.load(Ordering::Acquire)
            || self.rovent_root_policy.is_none()
            || owners.engine.is_none()
            || owners.native_finished.is_some()
            || owners.native_joined
            || owners.native.is_quarantined()
            || owners.native.occupancy() != 0
            || ready.domain() != original.domain()
            || ready.origin() != original.origin()
            || matches!(
                &owners.rovent_idle_custody,
                RoventPcrIdleCustody::RetiredRoot(receipt) if !receipt.permits_root_reuse()
            )
        {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        // No reset-request sequence or game/root metadata is consumed here.
        // Private custody plus the actual matching Ready clock/idle owner are
        // required; occupancy zero by itself never certifies physical reuse.
        let receipt = Arc::new(RoventPcrInitialReceipt {
            authority: context.authority,
            source: Arc::new(stopped),
            idle_custody: owners.rovent_idle_custody.clone(),
        });
        let report = receipt.report();
        let legal = receipt.source.allowed_root_moves().to_vec();
        *self
            .pending_output
            .lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())? =
            Some(PendingOutput {
                authority: context.authority,
                report: report.best_move,
                legal,
            });
        *self
            .latest_rovent_initial
            .lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())? = Some(receipt);
        Ok(report)
    }
}
