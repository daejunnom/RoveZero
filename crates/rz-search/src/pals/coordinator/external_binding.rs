//! Checked mapping between semantic PALS questions and external UCI namespaces.
//! They remain different digests: only the actual Rules/config producer can
//! issue this proof after checking both representations of the same work.
use super::{CoordinatorError, CoordinatorWork, FrameId};
use crate::external_cpu::worker::{
    external_evaluation_namespace, external_question_digest, external_root_restriction_digest,
    external_search_conditions,
};
use crate::external_cpu::{ExternalCpuConfig, ExternalSlicePolicy};
use crate::pals::review_context::ReviewView;
use rz_contracts::pals_adaptive::{ExternalCpuJobEnvelope, ExternalSliceGrant, ReviewViewSeal};
use rz_contracts::{ClockDomain, Digest, MonotonicTick};
use rz_position::contracts::ContractState;
use rz_position::{BoardMove, Position, PositionLimits};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct CheckedExternalCpuBinding {
    pub(super) frame: FrameId,
    pub(super) question: Digest,
    pub(super) view: ReviewViewSeal,
    pub(super) public_context: Digest,
    pub(super) input_digest: Digest,
    pub(super) job: ExternalCpuJobEnvelope,
    pub(super) grant: ExternalSliceGrant,
}
impl CheckedExternalCpuBinding {
    /// Crate-private issuer used by the actual engine port after its frozen
    /// CPU plan/config admission checks. Callers cannot echo two digests or
    /// supply a public passed flag to construct a namespace bridge.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attest<Q>(
        work: &CoordinatorWork<ReviewView, Q>,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
        state: &ContractState,
        roots: Option<&[BoardMove]>,
        max_depth: u16,
        policy: ExternalSlicePolicy,
        config: &ExternalCpuConfig,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Self, CoordinatorError> {
        config.validate().map_err(|_| {
            CoordinatorError::IdentityMismatch("external binding registered config")
        })?;
        grant.validate(job, clock, now)?;
        work.input
            .view_seal
            .validate(work.frame.authority, work.input.view.token_budget())?;
        super::ranking::validate_actual_view(&work.input.view, work.input.view_seal)?;
        if work.input.stage.role().is_some()
            || job.id.authority != work.frame.authority
            || job.hard_deadline != work.original_deadline
            || state.snapshot().identity() != job.state_identity
            || Digest(work.input.view.record_scope_digest()) != work.input.public_context
            || max_depth == 0
            || max_depth > config.max_depth
            || policy.quantum != grant.soft_quantum
            || policy.requested_reported_nodes != grant.requested_reported_node_hint
        {
            return Err(CoordinatorError::IdentityMismatch(
                "external binding exact scope/state/policy",
            ));
        }
        let anchor = work
            .input
            .view
            .anchors()
            .first()
            .ok_or(CoordinatorError::MissingEvidence)?;
        let mut replay = Position::from_snapshot(&anchor.snapshot, PositionLimits::default())
            .map_err(|_| CoordinatorError::IdentityMismatch("external binding root restoration"))?;
        for movement in &work.input.view.query().prefix {
            replay
                .make_move(*movement)
                .map_err(|_| CoordinatorError::IdentityMismatch("external binding Rules prefix"))?;
        }
        if !replay.snapshot().same_state(state.rules().snapshot()) || state.terminal_wdl().is_some()
        {
            return Err(CoordinatorError::IdentityMismatch(
                "external binding endpoint/history",
            ));
        }
        let legal = replay.legal_moves();
        policy
            .validate(roots.map_or(legal.len(), <[BoardMove]>::len))
            .map_err(|_| CoordinatorError::IdentityMismatch("external binding bounded policy"))?;
        if policy
            .quantum
            .checked_add(policy.stop_drain_reserve)
            .is_none_or(|required| required.as_nanos() > u128::from(job.hard_deadline.at.0 - now.0))
        {
            return Err(CoordinatorError::InvalidTransition(
                "external binding leaves no stop reserve",
            ));
        }
        let declared = &work.input.view.query().legal;
        if declared.len() != legal.len()
            || declared.iter().collect::<HashSet<_>>().len() != declared.len()
            || declared.iter().any(|movement| !legal.contains(movement))
            || roots.is_some_and(|roots| {
                roots.is_empty()
                    || roots.len() > 256
                    || roots.iter().collect::<HashSet<_>>().len() != roots.len()
                    || roots.iter().any(|movement| !legal.contains(movement))
            })
        {
            return Err(CoordinatorError::IdentityMismatch(
                "external binding actual legal restriction",
            ));
        }
        let restriction = external_root_restriction_digest(roots);
        let evaluation = external_evaluation_namespace(&config.identity)
            .map_err(|_| CoordinatorError::IdentityMismatch("external evaluator identity"))?;
        let conditions = external_search_conditions(config, max_depth, policy);
        let external_question = external_question_digest(
            state.snapshot().identity(),
            restriction,
            evaluation,
            conditions,
        );
        if job.root_restriction != restriction
            || job.evaluation_namespace != evaluation
            || job.search_conditions != conditions
            || job.question != external_question
        {
            return Err(CoordinatorError::IdentityMismatch(
                "external physical namespace evidence",
            ));
        }
        Ok(Self {
            frame: work.frame,
            question: work.input.question_digest,
            view: work.input.view_seal,
            public_context: work.input.public_context,
            input_digest: Digest(work.input.view.digest()),
            job: *job,
            grant: *grant,
        })
    }
    pub fn frame(&self) -> FrameId {
        self.frame
    }
    pub fn semantic_question(&self) -> Digest {
        self.question
    }
    pub fn physical_question(&self) -> Digest {
        self.job.question
    }
    pub fn input_digest(&self) -> Digest {
        self.input_digest
    }
}
