//! Original root goal owner for the structurally R-free port.
//!
//! The process owner supplies its existing clock. This module neither creates
//! a CPU credit ledger nor recasts a role question as a CPU refinement frame.
use super::*;
use crate::contract_time::{ContractClock, InstantClock};
use crate::pals::engine::adaptive_port::AdaptiveFrontierReceipt;
use crate::pals::rovent_frontier::{
    FrontierAttachment, FrontierError, FrontierLimits, FrontierObligation, FrontierRegistration,
    RoventFrontier, VerifiedFrontierQuestion,
};
use rz_contracts::MonotonicTick;

/// Independent startup evidence from the real single-writer process owner.
pub struct PcFrontierStartup {
    pub root_budget: ContractDigest,
    pub clock: InstantClock,
}

struct PcFrontierDuty {
    frame: FrameId,
    kind: FrontierFrameKind,
    question: ContractDigest,
    prefix: Vec<BoardMove>,
    obligation: FrontierObligation,
    finished: bool,
}

pub(super) struct PcFrontierState {
    clock: InstantClock,
    root_budget: ContractDigest,
    root_rules_scanned: bool,
    duties: Vec<PcFrontierDuty>,
    last_error: Option<FrontierError>,
}

impl<M: RoleModel> PcSemanticPort<M> {
    /// Attach before any role input exists. Every compared Rules/Store/model
    /// owner comes from this port; the registration cannot certify itself.
    pub fn attach_rovent_frontier(
        &mut self,
        registration: FrontierRegistration,
        limits: FrontierLimits,
        actual: PcFrontierStartup,
    ) -> Result<(), FrontierError> {
        if self.owner.semantic.rovent_frontier.is_some()
            || self.frontier_state.is_some()
            || self.next_question != 1
            || !self.pending.is_empty()
            || !self.consumed.is_empty()
            || self.counters.submitted_role_calls != 0
            || self.mode.has_native_r()
            || actual.clock.domain() != self.deadline.clock
            || actual.clock.instant_at(self.deadline.at)? != self.hard_deadline
        {
            return Err(FrontierError::Invalid("R-free original frontier startup"));
        }
        registration.goal.validate(
            std::num::NonZeroU16::new(self.owner.profile.maximum_root_prefix_plies)
                .ok_or(FrontierError::DepthHorizon)?,
        )?;
        let namespace = self
            .owner
            .semantic
            .model
            .registered_representation_namespace()
            .ok_or(FrontierError::Invalid(
                "actual native model namespace absent",
            ))?;
        namespace
            .verify_identity(
                self.owner.model_identity(),
                self.owner.model_value_identity(),
            )
            .map_err(|_| FrontierError::ForeignRoot)?;
        let now = actual.clock.now()?;
        let frontier = RoventFrontier::new(
            registration,
            limits,
            &self.owner.semantic.stores,
            actual.clock.domain(),
            now,
        )?;
        frontier.validate_attachment(
            FrontierAttachment {
                authority: self.authority,
                mode: self.mode,
                original_deadline: self.deadline,
                root_budget: actual.root_budget,
                root_rules: &self.rules,
                root_moves: &self.root_moves,
                root_state: self.owner.semantic.nodes[self.root].state,
                model_namespace: namespace,
                native_ledger: None,
            },
            &self.owner.semantic.stores,
        )?;
        self.owner.semantic.rovent_frontier = Some(Box::new(frontier));
        self.frontier_state = Some(PcFrontierState {
            clock: actual.clock,
            root_budget: actual.root_budget,
            root_rules_scanned: false,
            duties: Vec::new(),
            last_error: None,
        });
        Ok(())
    }

    pub fn rovent_frontier(&self) -> Option<&RoventFrontier> {
        self.owner.semantic.rovent_frontier.as_deref()
    }

    pub fn rovent_frontier_receipt(&self) -> Option<AdaptiveFrontierReceipt> {
        self.rovent_frontier()
            .map(|frontier| AdaptiveFrontierReceipt {
                root_budget: frontier.registration().root_budget,
                snapshot: frontier.snapshot(),
            })
    }

    pub fn last_frontier_error(&self) -> Option<FrontierError> {
        self.frontier_state
            .as_ref()
            .and_then(|state| state.last_error)
    }

    pub fn original_root_budget(&self) -> Option<ContractDigest> {
        self.frontier_state.as_ref().map(|state| state.root_budget)
    }

    pub(super) fn continuation_limit(&self) -> usize {
        self.rovent_frontier()
            .and_then(|frontier| {
                let goal = frontier.registration().goal;
                goal.depth_plies
                    .map(|depth| usize::from(depth.get()))
                    .or_else(|| goal.mate_moves.map(|moves| usize::from(moves.get()) * 2))
            })
            .unwrap_or(self.owner.config.line_plies)
            .min(usize::from(self.owner.profile.maximum_root_prefix_plies))
    }

    pub(super) fn frontier_failure(&mut self, error: FrontierError) -> PalsError {
        if let Some(state) = &mut self.frontier_state {
            state.last_error = Some(error);
        }
        self.completion = if error == FrontierError::Deadline {
            PcPortCompletion::Deadline
        } else {
            PcPortCompletion::Failed
        };
        match error {
            FrontierError::Deadline => RoleError::Deadline.into(),
            FrontierError::Capacity(_) => PalsError::Capacity,
            _ => StoreError::InvalidEvidence("original P/C frontier rejected goal work").into(),
        }
    }

    pub(super) fn prepare_frontier_root_if_attached(&mut self) -> Result<(), PalsError> {
        if self.frontier_state.is_some() {
            self.prepare_rovent_root_rules()
                .map_err(|error| self.frontier_failure(error))?;
        }
        Ok(())
    }

    /// Only an actual controller-issued missing descriptor can add an imposed
    /// branch. The first query contains no invented proposer/counter opinion.
    fn enqueue_missing_goal_question(&mut self) -> Result<bool, FrontierError> {
        let Some((clock, now)) = self.frontier_time()? else {
            return Ok(false);
        };
        let frontier = self
            .owner
            .semantic
            .rovent_frontier
            .as_deref_mut()
            .ok_or(FrontierError::ForeignRoot)?;
        if matches!(
            frontier.snapshot().goal,
            crate::pals::rovent_frontier::FrontierGoalStatus::MateSatisfied
                | crate::pals::rovent_frontier::FrontierGoalStatus::DepthSatisfied
        ) {
            return Ok(false);
        }
        let Some(question) = frontier.missing_mate_questions(1, clock, now)?.pop() else {
            return Ok(false);
        };
        let origin = Arc::new(goal_local::PcGoalLocalRules::prepare(self, question)?);
        let frame = self.next_frame;
        self.next_frame = frame
            .checked_add(1)
            .ok_or(FrontierError::Capacity("P/C goal frame sequence"))?;
        self.ready
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("P/C missing-goal queue"))?;
        self.ready.push_back(PcPlan {
            frame,
            responsibilities: Vec::new(),
            continuation_end: origin.continuation_end(),
            purpose: origin.purpose(),
            prefix: origin.prefix().to_vec(),
            proposal: Vec::new(),
            counter: None,
            goal_local: Some(origin),
        });
        Ok(true)
    }

    pub(super) fn complete_role_queue(&mut self) -> Result<(), PalsError> {
        if !self.ready.is_empty() || !self.pending.is_empty() {
            return Ok(());
        }
        if self
            .enqueue_missing_goal_question()
            .map_err(|error| self.frontier_failure(error))?
        {
            return Ok(());
        }
        if let Some((clock, now)) = self
            .frontier_time()
            .map_err(|error| self.frontier_failure(error))?
        {
            let frontier = self
                .owner
                .semantic
                .rovent_frontier
                .as_deref_mut()
                .ok_or(PalsError::InvalidConfig)?;
            if !frontier.snapshot().selection_sealed {
                frontier
                    .seal_selected_policy(clock, now)
                    .map_err(|error| self.frontier_failure(error))?;
            }
        }
        self.completion = PcPortCompletion::SelectiveFrontier;
        Ok(())
    }

    /// Actual immediate Rules children are charged once by the original
    /// controller. They do not count as model calls or P/C opinions.
    pub fn prepare_rovent_root_rules(&mut self) -> Result<(), FrontierError> {
        let state = self
            .frontier_state
            .as_ref()
            .ok_or(FrontierError::ForeignRoot)?;
        if state.root_rules_scanned {
            return Ok(());
        }
        let clock = state.clock;
        let moves = self.root_moves.allowed().to_vec();
        for movement in moves {
            let path = self
                .owner
                .semantic
                .rovent_frontier
                .as_deref_mut()
                .ok_or(FrontierError::ForeignRoot)?
                .observe_root_child(movement, clock.domain(), clock.now()?)?;
            let terminal = {
                let frontier = self
                    .owner
                    .semantic
                    .rovent_frontier
                    .as_deref()
                    .ok_or(FrontierError::ForeignRoot)?;
                let child =
                    Position::from_snapshot(frontier.path_snapshot(path)?, Default::default())?;
                matches!(
                    child.classify_position()?.play_status,
                    PlayStatus::Terminal { .. }
                )
            };
            if terminal {
                self.owner
                    .connect_hot(self.root, movement)
                    .map_err(|_| FrontierError::StoreUnavailable)?;
            }
        }
        self.frontier_state.as_mut().unwrap().root_rules_scanned = true;
        Ok(())
    }

    /// A concrete native owner must drain its original role tickets before
    /// calling this. No success, model completion or physical fence is minted.
    pub fn retire_rovent_frontier(
        &mut self,
    ) -> Result<Option<AdaptiveFrontierReceipt>, FrontierError> {
        let Some(receipt) = self.rovent_frontier_receipt() else {
            return Ok(None);
        };
        if !self.pending.is_empty()
            || !receipt.snapshot.work.native_usage_known
            || receipt.snapshot.work.outstanding_native_executions != 0
            || receipt.snapshot.work.reserved_native_nodes != 0
        {
            return Err(FrontierError::NativeUnknown);
        }
        self.owner.semantic.rovent_frontier.take();
        self.frontier_state.take();
        Ok(Some(receipt))
    }

    pub(super) fn frontier_time(
        &self,
    ) -> Result<Option<(rz_contracts::ClockDomain, MonotonicTick)>, FrontierError> {
        self.frontier_state
            .as_ref()
            .map(|state| Ok((state.clock.domain(), state.clock.now()?)))
            .transpose()
    }

    /// This creates no work envelope or role allowance. Only the exact private
    /// input and original accepted action sources reach the common controller.
    pub(super) fn observe_frontier_input(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
    ) -> Result<(), FrontierError> {
        let Some((clock, now)) = self.frontier_time()? else {
            return Ok(());
        };
        if input.question.first.is_none() {
            return Ok(());
        }
        if input
            .question
            .goal_local
            .as_ref()
            .is_some_and(|origin| Some(origin.root_budget()) != self.original_root_budget())
        {
            return Err(FrontierError::ForeignRoot);
        }
        let stores = &self.owner.semantic.stores;
        let cap = VerifiedFrontierQuestion::from_pc_input(input, stores)?;
        let frontier = self
            .owner
            .semantic
            .rovent_frontier
            .as_deref_mut()
            .ok_or(FrontierError::ForeignRoot)?;
        frontier.observe_verified_question(&cap, stores, clock, now)?;
        let state = self
            .frontier_state
            .as_mut()
            .ok_or(FrontierError::ForeignRoot)?;
        for source in input.question.frontier_responsibilities() {
            if state.duties.iter().any(|duty| {
                duty.frame == input.question.frontier_frame()
                    && duty.kind == source.kind()
                    && duty.question == source.question_digest()
                    && duty.prefix == source.prefix()
            }) {
                continue;
            }
            state
                .duties
                .try_reserve(1)
                .map_err(|_| FrontierError::Capacity("P/C responsibility handles"))?;
            let obligation =
                frontier.register_pc_responsibility(&cap, source, stores, clock, now)?;
            state.duties.push(PcFrontierDuty {
                frame: input.question.frontier_frame(),
                kind: source.kind(),
                question: source.question_digest(),
                prefix: source.prefix().to_vec(),
                obligation,
                finished: false,
            });
        }
        Ok(())
    }

    pub(super) fn accept_frontier_endpoint(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        fresh: Option<(&AcceptedRoleStep, &RoleOutput, AdaptivePortRoleEvidence)>,
    ) -> Result<(), FrontierError> {
        let Some((clock, now)) = self.frontier_time()? else {
            return Ok(());
        };
        let stores = &self.owner.semantic.stores;
        let cap = VerifiedFrontierQuestion::from_pc_input(input, stores)?;
        let frontier = self
            .owner
            .semantic
            .rovent_frontier
            .as_deref_mut()
            .ok_or(FrontierError::ForeignRoot)?;
        let endpoint = frontier.observe_verified_question(&cap, stores, clock, now)?;
        let state = self
            .frontier_state
            .as_mut()
            .ok_or(FrontierError::ForeignRoot)?;
        for duty in &mut state.duties {
            if duty.finished
                || duty.frame != input.question.frontier_frame()
                || !input
                    .question
                    .frontier_responsibilities()
                    .iter()
                    .any(|source| {
                        duty.kind == source.kind()
                            && duty.question == source.question_digest()
                            && duty.prefix == source.prefix()
                    })
            {
                continue;
            }
            match fresh {
                Some((accepted, output, physical)) => frontier.accept_verified_fresh_endpoint(
                    duty.obligation,
                    &cap,
                    stores,
                    endpoint,
                    accepted,
                    output,
                    physical,
                    clock,
                    now,
                )?,
                None => frontier.accept_verified_rules_endpoint(
                    duty.obligation,
                    &cap,
                    stores,
                    endpoint,
                    clock,
                    now,
                )?,
            }
            duty.finished = true;
        }
        Ok(())
    }
}
