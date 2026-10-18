//! Actual manager-to-goal observations and same-owner endpoint transitions.
//! An endpoint transition only issues the next real Fresh question; it never
//! invents a physical completion, a value, or a selected move.
use super::*;
use crate::pals::coordinator::AcceptedRoleStep;
use crate::pals::rovent_frontier::{FrontierError, FrontierFrameKind};
use rz_contracts::{ClockDomain, MonotonicTick};

/// An unsubmitted same-owner validation at its original go horizon. Only the
/// live Search port can issue this move-only transition. It is neither a CPU
/// completion nor a Fresh value; the next real role ticket still supplies the
/// physically retired Fresh evidence. Keep its immutable Arc custody on heap.
pub struct AdaptiveGoalBoundary<Q> {
    seal: Box<GoalBoundarySeal<Q>>,
}
struct GoalBoundarySeal<Q> {
    frame: FrameId,
    first: rz_contracts::pals::Move16,
    deadline: Deadline,
    budget: FrameBudget,
    original: FrameInput<ReviewView, Q>,
    root_budget: ContractDigest,
    target: usize,
    evidence: ContractDigest,
    next: FrameInput<ReviewView, Q>,
}
impl<Q> AdaptiveGoalBoundary<Q> {
    pub(crate) fn validate(&self, work: &CoordinatorWork<ReviewView, Q>) -> Result<(), PalsError> {
        let seal = &self.seal;
        let next = &seal.next;
        let original = &seal.original;
        let stage_matches = matches!(
            (original.stage, next.stage),
            (
                RefinementStage::CpuValidation,
                RefinementStage::ProposalValueFresh
            ) | (
                RefinementStage::CounterValidation,
                RefinementStage::CounterValueFresh
            ) | (
                RefinementStage::RepairValidation,
                RefinementStage::RepairValueFresh
            ) | (
                RefinementStage::RecheckCounterValidation,
                RefinementStage::FreshRepairedEndpoint
            )
        );
        if work.frame != seal.frame
            || work.original_first_move != seal.first
            || work.original_deadline != seal.deadline
            || work.budget != seal.budget
            || work.input.stage != original.stage
            || work.input.question_digest != original.question_digest
            || work.input.view_seal != original.view_seal
            || work.input.public_context != original.public_context
            || !Arc::ptr_eq(&work.input.question, &original.question)
            || !Arc::ptr_eq(&work.input.view, &original.view)
            || seal.root_budget == ContractDigest([0; 32])
            || seal.target == 0
            || original.view.query().prefix.len() != seal.target
            || !stage_matches
            || next.view.query().purpose != RoleQueryPurpose::ValueFresh
            || next.view.query().prefix.len() > seal.target
            || next.view_seal.authority != seal.frame.authority
            || next.view_seal.root_state != original.view_seal.root_state
            || next.view.root_state() != original.view.root_state()
            || next.view.generation() != original.view.generation()
            || next.public_context.0 != next.view.record_scope_digest()
            || (original.stage != RefinementStage::RecheckCounterValidation
                && next.view.query().prefix != original.view.query().prefix)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
    pub(crate) fn into_parts(self) -> (ContractDigest, FrameInput<ReviewView, Q>) {
        (self.seal.evidence, self.seal.next)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// The configured generation chunk remains unchanged. A go's own depth
    /// or mate horizon can only shorten it; goal-local work keeps its separate
    /// controller-issued continuation beyond a shorter configured chunk.
    pub(super) fn original_goal_horizon(&self) -> Option<usize> {
        let goal = self.rovent_frontier()?.registration().goal;
        // The admitted goal contains both mate lanes: root win at 2N-1 and
        // root loss at 2N. Keep their maximum endpoint here; the original
        // Frontier still proves each lane and checks Rules terminals itself.
        // SearchGoal::validate already checked the conversion and capability.
        goal.depth_plies
            .map(|depth| usize::from(depth.get()))
            .into_iter()
            .chain(goal.mate_moves.map(|moves| usize::from(moves.get()) * 2))
            .min()
    }
    pub(super) fn continuation_limit(&self) -> usize {
        self.original_goal_horizon()
            .map_or(self.engine.config.line_plies, |horizon| {
                horizon.min(self.engine.config.line_plies)
            })
    }

    /// Replace only an unsubmitted validation exactly at the original goal
    /// endpoint with its real Fresh question. No zero-depth R job is issued.
    pub fn prepare_rovent_goal_boundary(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<AdaptiveGoalBoundary<AdaptivePortQuestion>>, PalsError> {
        self.check_work(work)?;
        let query = &work.input.question;
        if !matches!(
            query.stage,
            RefinementStage::CpuValidation
                | RefinementStage::CounterValidation
                | RefinementStage::RepairValidation
                | RefinementStage::RecheckCounterValidation
        ) {
            return Ok(None);
        }
        let Some(target) = self.original_goal_horizon() else {
            return Ok(None);
        };
        if query.context.prefix.len() < target {
            return Ok(None);
        }
        if query.context.prefix.len() != target {
            return Err(FrontierError::DepthHorizon.into());
        }
        if self.engine.semantic.nodes[query.node].terminal.is_some() {
            // Existing Rules-terminal handling owns this endpoint instead.
            return Ok(None);
        }
        let checked = self.checked_frontier_line(&query.context.prefix)?;
        if !checked
            .endpoint_snapshot()
            .same_state(&query.position.snapshot())
        {
            return Err(FrontierError::ForeignRoot.into());
        }
        let root_budget = self
            .rovent_frontier()
            .ok_or(FrontierError::ForeignRoot)?
            .registration()
            .root_budget;
        let next = if query.stage == RefinementStage::RecheckCounterValidation {
            let (transition, rules_pair) = self.prepare_final_pair(
                work,
                query.repaired.clone().ok_or(RoleError::InvalidOutput)?,
                query.counter.clone().ok_or(RoleError::InvalidOutput)?,
            )?;
            if rules_pair.is_some() {
                return Err(RoleError::InvalidOutput.into());
            }
            let materialized = self.materialize_step(AdaptivePortStep {
                evidence: work.input.question_digest,
                transition,
                consumption_gate: None,
                feedback_gate: None,
                queue_question: None,
                fresh_pair: None,
                rules_pair: None,
            })?;
            let FrameTransition::Continue(next) = materialized.transition else {
                return Err(RoleError::InvalidOutput.into());
            };
            next
        } else {
            let stage = match query.stage {
                RefinementStage::CpuValidation => RefinementStage::ProposalValueFresh,
                RefinementStage::CounterValidation => RefinementStage::CounterValueFresh,
                RefinementStage::RepairValidation => RefinementStage::RepairValueFresh,
                _ => return Err(RoleError::InvalidOutput.into()),
            };
            self.input(
                query.node,
                stage,
                query.first,
                query.context.prefix.clone(),
                query.proposal.clone(),
                query.counter.clone(),
                query.repaired.clone(),
                query.attack_ply,
                query.recheck,
                None,
                None,
            )?
        };
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-original-goal-boundary/1\0");
        hash.update(root_budget.0);
        hash.update(work.input.question_digest.0);
        hash.update(work.input.view_seal.view_digest.0);
        hash.update((target as u64).to_le_bytes());
        hash.update(next.question_digest.0);
        let boundary = AdaptiveGoalBoundary {
            seal: Box::new(GoalBoundarySeal {
                frame: work.frame,
                first: work.original_first_move,
                deadline: work.original_deadline,
                budget: work.budget,
                original: work.input.clone(),
                root_budget,
                target,
                evidence: ContractDigest(hash.finalize().into()),
                next,
            }),
        };
        boundary.validate(work)?;
        Ok(Some(boundary))
    }

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
