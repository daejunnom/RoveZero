//! Actual manager-to-goal observations. These methods consume original work
//! and physical acceptance; they neither create a stage nor select a move.
use super::*;
use crate::pals::coordinator::AcceptedRoleStep;
use crate::pals::rovent_frontier::{FrontierError, FrontierFrameKind};
use rz_contracts::{ClockDomain, MonotonicTick};

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// A concrete manager invokes this at the first actual admission of a
    /// selected lane. A later continuation cannot replace its original Q.
    pub fn rovent_frame_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        kind: FrontierFrameKind,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, FrontierError> {
        if !self.rovent_goal_work_live(work)? {
            return Ok(false);
        }
        self.with_rovent_frontier(|frontier, stores| {
            frontier.admit_role_input(clock, now)?;
            if let Some(original) = frontier.adaptive_obligation(work.frame, kind)? {
                return if frontier.adaptive_obligation_source_matches(original, work, stores)? {
                    Ok(true)
                } else {
                    Err(FrontierError::Invalid(
                        "responsibility retry differs from original question",
                    ))
                };
            }
            frontier.register_local_frame(work, stores, kind, clock, now)?;
            Ok(true)
        })?
        .ok_or(FrontierError::ForeignRoot)
    }

    /// Only an actual retired native Fresh response can close its lane. A
    /// queued Repair may compare a counter endpoint without owning a new
    /// Counter responsibility; that observation does not invent one.
    #[allow(clippy::too_many_arguments)]
    pub fn rovent_fresh_accepted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, PalsError> {
        if !self.rovent_goal_work_live(work)? {
            return Ok(false);
        }
        let kind = fresh_kind(work.input.stage).ok_or(FrontierError::MissingAcceptance)?;
        // Capacity and original physical/source validation precede head use.
        // The controller is the only place that may close its responsibility.
        let obligation = self
            .rovent_frontier()
            .ok_or(FrontierError::ForeignRoot)?
            .adaptive_obligation(work.frame, kind)?;
        let Some(obligation) = obligation else {
            self.with_rovent_frontier(|frontier, stores| {
                frontier.observe_local_question(work, stores, clock, now)?;
                Ok(())
            })?;
            return Ok(false);
        };
        let permit = self.prepare_fresh_source_use(work, accepted, output, physical)?;
        self.with_rovent_frontier(|frontier, stores| {
            let endpoint = frontier.observe_local_question(work, stores, clock, now)?;
            // Shallow Fresh is real progress and consumption, while only the
            // controller's Closed disposition finishes the original duty.
            frontier.accept_local_fresh_horizon_once(
                obligation, work, stores, endpoint, accepted, output, physical, clock, now,
            )?;
            Ok(())
        })?
        .ok_or(FrontierError::ForeignRoot)?;
        if let Some(permit) = permit {
            self.record_used_fresh_source(permit)?;
        }
        Ok(true)
    }

    /// Rules terminal is exact evidence for this endpoint. CPU/NN values,
    /// missing alternatives and unfinished lines cannot enter this path.
    pub fn rovent_rules_endpoint(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, FrontierError> {
        if !self.rovent_goal_work_live(work)? {
            return Ok(false);
        }
        if !matches!(
            work.input
                .question
                .position
                .classify_position()?
                .play_status,
            PlayStatus::Terminal { .. }
        ) {
            return Ok(false);
        }
        let kind = endpoint_kind(work.input.stage).ok_or(FrontierError::MissingAcceptance)?;
        self.with_rovent_frontier(|frontier, stores| {
            let endpoint = frontier.observe_local_question(work, stores, clock, now)?;
            let Some(obligation) = frontier.adaptive_obligation(work.frame, kind)? else {
                return Ok(false);
            };
            frontier
                .accept_local_rules_endpoint_once(obligation, work, stores, endpoint, clock, now)?;
            Ok(true)
        })?
        .ok_or(FrontierError::ForeignRoot)
    }

    /// Called at the manager's explicit end of selected branch/recheck work,
    /// rather than inferred from a momentarily empty queue.
    pub fn rovent_root_round_sealed(
        &mut self,
        root_budget: ContractDigest,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, FrontierError> {
        if self.rovent_frontier().is_none() {
            return Ok(false);
        }
        self.check_rovent_scan_live()?;
        self.with_rovent_frontier(|frontier, _| {
            if frontier.registration().root_budget != root_budget {
                return Err(FrontierError::ForeignRoot);
            }
            frontier.admit_role_input(clock, now)?;
            if !frontier.snapshot().selection_sealed {
                frontier.seal_selected_policy(clock, now)?;
            }
            Ok(true)
        })?
        .ok_or(FrontierError::ForeignRoot)
    }

    fn rovent_goal_work_live(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<bool, FrontierError> {
        if self.rovent_frontier().is_none() {
            return Ok(false);
        }
        self.check_rovent_scan_live()?;
        if work.frame.authority != self.authority
            || work.original_deadline != self.deadline
            || self.closed.contains(&work.frame)
        {
            return Err(FrontierError::ForeignRoot);
        }
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)
            .map_err(|_| FrontierError::MissingAcceptance)?;
        Ok(true)
    }
}

fn fresh_kind(stage: RefinementStage) -> Option<FrontierFrameKind> {
    match stage {
        RefinementStage::ProposalValueFresh => Some(FrontierFrameKind::Candidate),
        RefinementStage::CounterValueFresh | RefinementStage::FreshCounterEndpoint => {
            Some(FrontierFrameKind::Counter)
        }
        RefinementStage::RepairValueFresh
        | RefinementStage::FreshRepairedEndpoint
        | RefinementStage::RecheckRepairedValueFresh => Some(FrontierFrameKind::Repair),
        _ => None,
    }
}

fn endpoint_kind(stage: RefinementStage) -> Option<FrontierFrameKind> {
    fresh_kind(stage).or(match stage {
        RefinementStage::ProposalContinuation | RefinementStage::CpuValidation => {
            Some(FrontierFrameKind::Candidate)
        }
        RefinementStage::CriticReply
        | RefinementStage::CounterContinuation
        | RefinementStage::CounterValidation
        | RefinementStage::CriticRecheck
        | RefinementStage::RecheckCounterValidation => Some(FrontierFrameKind::Counter),
        RefinementStage::Repair | RefinementStage::RepairValidation => {
            Some(FrontierFrameKind::Repair)
        }
        _ => None,
    })
}
