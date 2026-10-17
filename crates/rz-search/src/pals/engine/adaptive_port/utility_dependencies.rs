//! Owner-private materialized causes. These attest a semantic successor only;
//! no CPU grant, role submission, deadline or completion is issued here.
use super::*;
use crate::pals::store::StoreReadOwner;
use rz_contracts::pals_adaptive::CpuJobEnvelope;
use rz_contracts::pals_adaptive::CpuJobId;
use rz_contracts::rovent::QuestionEnvelope;
use std::sync::Weak;

const MAX_UTILITY_CONTINUATIONS: usize = crate::pals::coordinator::MAX_COORDINATOR_FRAMES;

#[derive(Clone, Copy)]
pub(super) enum UtilityContinuationCause {
    Role {
        request: RequestId,
        frame: FrameId,
        question: ContractDigest,
        view: ContractDigest,
    },
    Cpu {
        job: CpuJobId,
        execution: rz_contracts::ExecutionId,
        frame: FrameId,
        question: ContractDigest,
        observation: ObservationId,
    },
}

impl UtilityContinuationCause {
    fn frame(self) -> FrameId {
        match self {
            Self::Role { frame, .. } | Self::Cpu { frame, .. } => frame,
        }
    }
}

/// Only materialization of a checked private blueprint can mint this source.
/// Weak references do not keep an old query, output tensor or Store lease live.
#[derive(Clone)]
pub struct UtilityContinuationSource {
    owner: StoreReadOwner,
    generation: u64,
    root: StateId,
    authority: SearchAuthority,
    deadline: Deadline,
    hard_deadline: Instant,
    cause: UtilityContinuationCause,
    target_query: Weak<AdaptivePortQuestion>,
    target_view: Weak<ReviewView>,
    stage: RefinementStage,
    question: ContractDigest,
    view: ReviewViewSeal,
    full_view: ContractDigest,
    public_context: ContractDigest,
    first: rz_contracts::pals::Move16,
}
impl UtilityContinuationSource {
    pub fn role_source(&self) -> Option<(RequestId, FrameId, ContractDigest, ContractDigest)> {
        match self.cause {
            UtilityContinuationCause::Role {
                request,
                frame,
                question,
                view,
            } => Some((request, frame, question, view)),
            _ => None,
        }
    }
    pub fn cpu_source(
        &self,
    ) -> Option<(
        CpuJobId,
        rz_contracts::ExecutionId,
        FrameId,
        ContractDigest,
        ObservationId,
    )> {
        match self.cause {
            UtilityContinuationCause::Cpu {
                job,
                execution,
                frame,
                question,
                observation,
            } => Some((job, execution, frame, question, observation)),
            _ => None,
        }
    }
}

/// An actually settled suspension, not a raw publication or completed search.
/// It exists only while the original Store task and same CPU owner are paused.
pub(super) struct UtilityReciprocalAwaitSource {
    job: CpuJobEnvelope,
    execution: rz_contracts::ExecutionId,
    parent: QuestionEnvelope,
}
impl UtilityReciprocalAwaitSource {
    pub(super) fn job(&self) -> CpuJobEnvelope {
        self.job
    }
    pub(super) fn execution(&self) -> rz_contracts::ExecutionId {
        self.execution
    }
    pub(super) fn parent(&self) -> QuestionEnvelope {
        self.parent
    }
}

#[derive(Default)]
pub(super) struct UtilityContinuationRegistry {
    enabled: bool,
    entries: HashMap<ContractDigest, UtilityContinuationSource>,
}
impl UtilityContinuationRegistry {
    pub(super) fn enable(&mut self) {
        self.enabled = true;
    }
    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }
    pub(super) fn reserve(&mut self) -> Result<(), PalsError> {
        // Expired weak metadata has no historical packet, view lease or live
        // continuation authority. Actual action history belongs to its owner.
        self.entries.retain(|_, source| {
            source.target_query.strong_count() != 0 && source.target_view.strong_count() != 0
        });
        if self.entries.len() >= MAX_UTILITY_CONTINUATIONS {
            return Err(PalsError::Capacity);
        }
        self.entries.try_reserve(1).map_err(|_| PalsError::Capacity)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub(super) fn validated_utility_await(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &crate::cpu_worker::CpuSliceReport,
    ) -> Result<UtilityReciprocalAwaitSource, PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        let job = admission.job.ok_or(RoleError::InvalidOutput)?;
        let execution = admission.execution.ok_or(RoleError::InvalidOutput)?;
        let parent = self
            .validated_utility_r_parent(work)?
            .ok_or(RoleError::InvalidOutput)?;
        let request = receipt
            .reciprocal_request
            .as_ref()
            .ok_or(RoleError::InvalidOutput)?;
        let task = self.engine.semantic.stores.tasks.get(execution)?;
        if admission.frame != work.frame
            || admission.stage != work.input.stage
            || admission.view_seal != work.input.view_seal
            || admission.context != work.input.question.context
            || admission.first_move != work.original_first_move
            || admission.grant.is_some()
            || admission.owner != Some(receipt.grant.owner)
            || admission.terminal
            || admission.quarantined
            || !admission.continuation_retained
            || receipt.grant.job != job.id
            || receipt.terminal
            || receipt.admission_rejected
            || !receipt.continuation_retained
            || !receipt.result.as_ref().is_ok_and(|actual| {
                actual.report.completion == crate::cpu::CpuCompletion::AwaitReciprocal
            })
            || receipt.actual_work.is_none_or(|actual| actual.nodes == 0)
            || receipt.cumulative_nodes != admission.cumulative_nodes
            || receipt.progress_sequence != admission.progress_sequence
            || request.scope().job != job
            || request.scope().owner != receipt.grant.owner
            || request.scope().parent != parent
            || admission.settled_receipt
                != Some((
                    receipt.grant.execution,
                    execution,
                    cpu_commit::cpu_receipt_digest(receipt),
                ))
            || task.key != admission.task_key
            || task.status
                != (TaskStatus::Paused {
                    checkpoint: job.id.sequence,
                    evidence: None,
                })
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(UtilityReciprocalAwaitSource {
            job,
            execution: receipt.grant.execution,
            parent,
        })
    }

    pub(super) fn utility_cpu_continuation_cause(
        &self,
        question: ContractDigest,
        observation: ObservationId,
    ) -> Result<Option<UtilityContinuationCause>, PalsError> {
        let Some(admission) = self.cpu_admissions.get(&question) else {
            // Legacy/profiled external feedback has a different authority.
            return Ok(None);
        };
        let job = admission.job.ok_or(RoleError::InvalidOutput)?;
        let (execution, _, actual) = admission
            .committed_execution
            .ok_or(RoleError::InvalidOutput)?;
        if actual != observation
            || !self.cpu_feedback_still_authorized(question, observation)
            || !self.cpu_action_committed(question, job.id, execution)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Some(UtilityContinuationCause::Cpu {
            job: job.id,
            execution,
            frame: admission.frame,
            question,
            observation,
        }))
    }

    pub(super) fn record_utility_continuation(
        &mut self,
        cause: UtilityContinuationCause,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        AdaptivePortQuestion::validate_frontier_input(input, &self.engine.semantic.stores)?;
        let first = input.question.first.ok_or(RoleError::InvalidOutput)?;
        let first = rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).map_err(|_| RoleError::InvalidOutput)?,
        );
        if cause.frame().authority != self.authority
            || self
                .utility_continuations
                .entries
                .contains_key(&input.question_digest)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.utility_continuations.entries.insert(
            input.question_digest,
            UtilityContinuationSource {
                owner: self.engine.semantic.stores.read_owner(),
                generation: self.engine.semantic.stores.generation(),
                root: self.engine.semantic.nodes[self.root].state,
                authority: self.authority,
                deadline: self.deadline,
                hard_deadline: self.limits.deadline,
                cause,
                target_query: Arc::downgrade(&input.question),
                target_view: Arc::downgrade(&input.view),
                stage: input.stage,
                question: input.question_digest,
                view: input.view_seal,
                full_view: ContractDigest(input.view.digest()),
                public_context: input.public_context,
                first,
            },
        );
        Ok(())
    }

    /// The caller supplies actual ready_work after a successful machine
    /// transition. Public frame equality or a reconstructed question cannot
    /// substitute for the exact input this owner materialized.
    pub(super) fn validated_utility_continuation(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<UtilityContinuationSource>, PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let Some(source) = self
            .utility_continuations
            .entries
            .get(&work.input.question_digest)
        else {
            return Ok(None);
        };
        if !source.owner.matches(&self.engine.semantic.stores)
            || source.generation != self.engine.semantic.stores.generation()
            || source.root != self.engine.semantic.nodes[self.root].state
            || source.authority != self.authority
            || source.deadline != work.original_deadline
            || source.hard_deadline != work.input.question.deadline
            || source.cause.frame() != work.frame
            || source.first != work.original_first_move
            || source.stage != work.input.stage
            || source.question != work.input.question_digest
            || source.view != work.input.view_seal
            || source.full_view != ContractDigest(work.input.view.digest())
            || source.public_context != work.input.public_context
            || !source
                .target_query
                .upgrade()
                .is_some_and(|query| Arc::ptr_eq(&query, &work.input.question))
            || !source
                .target_view
                .upgrade()
                .is_some_and(|view| Arc::ptr_eq(&view, &work.input.view))
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Some(source.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::port_tests::{output, setup, work};
    use super::*;

    #[test]
    fn ordinary_utility_cause_requires_consumed_source_and_exact_materialized_target() {
        let mut port = setup(Position::startpos(), 3);
        let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&original, 1);
        port.utility_continuations.enable();
        let step = port
            .commit_role(&original, &ticket, &response, None)
            .unwrap();
        port.role_consumed(&ticket, &original.input.view).unwrap();
        let FrameTransition::Continue(input) = port.materialize_step(step).unwrap().transition
        else {
            panic!("actual semantic continuation")
        };
        let target = CoordinatorWork {
            input,
            ..original.clone()
        };
        let leases = port.engine.semantic.stores.review_pin_count().unwrap();
        let view_owners = Arc::strong_count(&target.input.view);
        let source = port
            .validated_utility_continuation(&target)
            .unwrap()
            .unwrap();
        assert_eq!(
            source.role_source(),
            Some((
                ticket.id,
                original.frame,
                original.input.question_digest,
                original.input.view_seal.view_digest
            ))
        );
        assert!(source.cpu_source().is_none());
        assert_eq!(Arc::strong_count(&target.input.view), view_owners);
        assert_eq!(
            port.engine.semantic.stores.review_pin_count().unwrap(),
            leases
        );
        assert!(
            port.validated_utility_continuation(&original)
                .unwrap()
                .is_none()
        );

        let mut copied_query = target.clone();
        copied_query.input.question = Arc::new(target.input.question.as_ref().clone());
        assert!(port.validated_utility_continuation(&copied_query).is_err());
        let mut other_frame = target.clone();
        other_frame.frame.sequence += 1;
        assert!(port.validated_utility_continuation(&other_frame).is_err());
        let mut changed_scope = target.clone();
        changed_scope.input.public_context = ContractDigest([243; 32]);
        assert!(port.validated_utility_continuation(&changed_scope).is_err());
        assert!(
            port.validated_utility_continuation(&target)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn unconsumed_or_default_off_materialization_grants_no_utility_successor() {
        let mut port = setup(Position::startpos(), 3);
        let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&original, 1);
        let step = port
            .commit_role(&original, &ticket, &response, None)
            .unwrap();
        port.role_consumed(&ticket, &original.input.view).unwrap();
        let FrameTransition::Continue(input) = port.materialize_step(step).unwrap().transition
        else {
            panic!("actual semantic continuation")
        };
        let target = CoordinatorWork {
            input,
            ..original.clone()
        };
        assert!(port.utility_continuations.entries.is_empty());
        assert!(
            port.validated_utility_continuation(&target)
                .unwrap()
                .is_none()
        );

        let mut port = setup(Position::startpos(), 3);
        let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&original, 1);
        port.utility_continuations.enable();
        let step = port
            .commit_role(&original, &ticket, &response, None)
            .unwrap();
        assert!(port.materialize_step(step).is_err());
        assert!(port.utility_continuations.entries.is_empty());
    }
}
