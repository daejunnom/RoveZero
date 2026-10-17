//! Exact missing-mate origin preparation. This is not a model opinion, peer
//! question, selected candidate, completed horizon or Rules mate proof.
//!
//! The actual port queue consumes this immutable origin. Its constructor is
//! private to the P/C port and emits no role ticket or reciprocal peer.
use super::*;
use crate::pals::rovent_frontier::{FrontierError, MateQuestionKind, MissingMateQuestion};

pub(super) struct PcGoalLocalRules {
    question: Arc<MissingMateQuestion>,
    owner: Arc<()>,
    authority: SearchAuthority,
    root_budget: ContractDigest,
    prefix: Vec<BoardMove>,
    node: usize,
    snapshot: PositionSnapshot,
    root_side: Color,
    continuation_end: usize,
    digest: ContractDigest,
}

impl PcGoalLocalRules {
    /// A copied key or a guessed prefix cannot call this constructor. The
    /// actual controller-issued descriptor is validated before any Rules work.
    pub(super) fn prepare<M: RoleModel>(
        port: &mut PcSemanticPort<M>,
        question: MissingMateQuestion,
    ) -> Result<Self, FrontierError> {
        let frontier = port
            .owner
            .semantic
            .rovent_frontier
            .as_ref()
            .ok_or(FrontierError::ForeignRoot)?;
        frontier.validate_missing_question(&question)?;
        let prefix = frontier.missing_question_prefix(&question)?;
        if prefix.is_empty()
            || prefix.len() > usize::from(port.owner.profile.maximum_root_prefix_plies)
        {
            return Err(FrontierError::DepthHorizon);
        }
        let continuation_end = prefix
            .len()
            .checked_add(usize::from(
                frontier.missing_question_chunk_limit(
                    &question,
                    u16::try_from(port.owner.config.line_plies)
                        .map_err(|_| FrontierError::DepthHorizon)?,
                )?,
            ))
            .ok_or(FrontierError::DepthHorizon)?;
        let root_budget = frontier.registration().root_budget;
        let (clock, now) = port.frontier_time()?.ok_or(FrontierError::ForeignRoot)?;
        let checked = port
            .owner
            .semantic
            .rovent_frontier
            .as_deref_mut()
            .ok_or(FrontierError::ForeignRoot)?
            .observe_missing_child(&question, clock, now)?;
        if checked.prefix() != prefix.as_slice() {
            return Err(FrontierError::ForeignHandle);
        }
        let snapshot = checked.endpoint_snapshot().clone();
        let node = port
            .adopt_frontier_line(&checked)
            .map_err(|_| FrontierError::StoreUnavailable)?;
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-goal-local-rules-origin/1\0");
        hash.update(root_budget.0);
        hash.update(
            role_line_sha256(b"imposed-root-prefix", &prefix)
                .map_err(|_| FrontierError::ForeignRoot)?,
        );
        hash.update([match question.kind() {
            MateQuestionKind::AttackCandidate => 0,
            MateQuestionKind::RequiredDefense => 1,
        }]);
        hash.update([question.attacker() as u8]);
        hash.update(question.remaining().0.to_le_bytes());
        hash.update(question.original_deadline().at.0.to_le_bytes());
        Ok(Self {
            question: Arc::new(question),
            owner: Arc::clone(&port.identity),
            authority: port.authority,
            root_budget,
            prefix,
            node,
            snapshot,
            root_side: port.owner.semantic.nodes[port.root].position.side_to_move(),
            continuation_end,
            digest: ContractDigest(hash.finalize().into()),
        })
    }

    pub(super) fn root_budget(&self) -> ContractDigest {
        self.root_budget
    }
    pub(super) fn node(&self) -> usize {
        self.node
    }
    pub(super) fn prefix(&self) -> &[BoardMove] {
        &self.prefix
    }
    pub(super) fn digest(&self) -> ContractDigest {
        self.digest
    }
    pub(super) fn continuation_end(&self) -> usize {
        self.continuation_end
    }
    pub(super) fn purpose(&self) -> RoleQueryPurpose {
        // Mate kind is attacker-relative proof metadata. Model roles follow
        // the exact imposed child's turn relative to this engine's root side.
        if self.continuation_end == self.prefix.len() {
            RoleQueryPurpose::ValueFresh
        } else if self.snapshot.side_to_move() == self.root_side {
            RoleQueryPurpose::ProposePolicy
        } else {
            RoleQueryPurpose::ReplyPolicy
        }
    }

    /// Validate the frozen original admission rather than requiring that the
    /// controller still call an already explored child "missing" on delivery.
    pub(super) fn validates_input(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        owner: &Arc<()>,
        authority: SearchAuthority,
        root_budget: ContractDigest,
    ) -> Result<(), PalsError> {
        if !Arc::ptr_eq(&self.owner, owner)
            || self.authority != authority
            || self.root_budget != root_budget
            || self.question.original_deadline() != input.question.source.original_deadline
            || self.prefix != input.question.context.prefix
            || !self
                .snapshot
                .same_state(&input.question.position.snapshot())
            || input.question.context.purpose != self.purpose()
            || !input.view.query().proposal.is_empty()
            || input.view.query().counterexample.is_some()
            || !input.view.query().divergences.is_empty()
            || input.question.context.proposal_sha256 != role_line_sha256(b"proposal", &[])?
            || input.question.context.refutation_sha256.is_some()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}
