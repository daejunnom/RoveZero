//! Exact selected-view bridge for an owner-paused native R question. The
//! existing Store, native ticket service and CPU worker keep their authority.
use super::*;
use crate::cpu::{CpuCompletion, CpuReciprocalReply, CpuReciprocalRequest};
use crate::cpu_worker::CpuSliceReport;
use crate::pals::coordinator::{
    AcceptedRoleStep, MAX_COORDINATOR_FRAMES, MAX_RECIPROCAL_QUESTION_HISTORY,
};
use rz_contracts::pals_adaptive::CpuJobEnvelope;
use rz_contracts::rovent::{
    ChessPly, ExecutionMode, QuestionEnvelope, QuestionPurpose, ReciprocalQuestionId,
    SearchRecipient,
};

#[derive(Default)]
pub(super) struct ReciprocalPortState {
    parents: HashMap<(FrameId, ContractDigest), QuestionEnvelope>,
    pending: HashMap<ReciprocalQuestionId, PendingCpuQuestion>,
    answers: VecDeque<Arc<AdaptiveAcceptedReciprocalAnswer>>,
}

struct PendingCpuQuestion {
    envelope: QuestionEnvelope,
    input: FrameInput<ReviewView, AdaptivePortQuestion>,
    request: CpuReciprocalRequest,
    anchor: Arc<ContractState>,
}

pub struct AdaptiveCpuReciprocalQuestion {
    pub question: QuestionEnvelope,
    pub input: FrameInput<ReviewView, AdaptivePortQuestion>,
}

/// One consumed canonical ticket, shared only with aliases of its exact view.
/// It is a policy answer, never a completed minimax or Rules terminal proof.
pub struct AdaptiveAcceptedReciprocalAnswer {
    question: QuestionEnvelope,
    ticket: RoleTicketEnvelope,
    evidence: ContractDigest,
    ranked: Vec<BoardMove>,
    view: Arc<ReviewView>,
}
impl AdaptiveAcceptedReciprocalAnswer {
    pub fn question(&self) -> QuestionEnvelope {
        self.question
    }
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
    pub fn evidence(&self) -> ContractDigest {
        self.evidence
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// The CPU preparation already issued this Rules owner. Keep its identity
    /// instead of exporting a new root owner or guessing an anchor from a hash.
    pub fn cpu_reciprocal_parent(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        id: ReciprocalQuestionId,
        mode: ExecutionMode,
        root_budget: ContractDigest,
    ) -> Result<QuestionEnvelope, PalsError> {
        self.check_work(work)?;
        let sender = match work.input.stage {
            RefinementStage::CpuValidation | RefinementStage::RepairValidation => {
                SearchRecipient::Proposer
            }
            RefinementStage::CpuDiscovery
            | RefinementStage::CounterValidation
            | RefinementStage::RecheckCounterValidation => SearchRecipient::Critic,
            _ => return Err(RoleError::Unsupported("question is not a native R stage").into()),
        };
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if !mode.has_native_r()
            || id.authority != self.authority
            || id.sequence == 0
            || job.id.authority != self.authority
            || job.question != work.input.question_digest
            || job.state_identity != admission.rules_identity
            || job.hard_deadline != self.deadline
            || admission.frame != work.frame
            || admission.context != work.input.question.context
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let key = (work.frame, work.input.question_digest);
        if let Some(original) = self.reciprocal.parents.get(&key) {
            return if original.id == id && original.root_budget == root_budget {
                Ok(*original)
            } else {
                Err(RoleError::InvalidOutput.into())
            };
        }
        if self.reciprocal.parents.len() >= MAX_RECIPROCAL_QUESTION_HISTORY {
            return Err(PalsError::Capacity);
        }
        let question = QuestionEnvelope {
            id,
            parent: None,
            sender,
            recipient: SearchRecipient::Researcher,
            purpose: QuestionPurpose::TacticalReview,
            anchor: job.state_identity,
            anchor_root_ply: ChessPly(
                u16::try_from(work.input.question.context.prefix.len())
                    .map_err(|_| PalsError::Capacity)?,
            ),
            original_first_move: Some(work.original_first_move),
            prefix_digest: ContractDigest(role_line_sha256(
                b"prefix",
                &work.input.question.context.prefix,
            )?),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: work.input.question_digest,
            view: work.input.view_seal,
            evidence_revision: work.input.question.context.public_revision,
            root_budget,
            original_deadline: work.original_deadline,
        };
        question
            .view
            .validate(self.authority, work.input.view.token_budget())
            .map_err(|_| RoleError::InvalidOutput)?;
        self.reciprocal.parents.insert(key, question);
        Ok(question)
    }

    /// Reuse exact Rules replay, hot graph insertion and Store-pinned selection.
    /// No played history is appended to the model-facing virtual prefix.
    pub fn prepare_cpu_reciprocal(
        &mut self,
        parent: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        request: &CpuReciprocalRequest,
        id: ReciprocalQuestionId,
        mode: ExecutionMode,
        root_budget: ContractDigest,
    ) -> Result<AdaptiveCpuReciprocalQuestion, PalsError> {
        self.check_work(parent)?;
        let scope = request.scope();
        if !mode.has_native_r()
            || self
                .reciprocal
                .parents
                .get(&(parent.frame, parent.input.question_digest))
                != Some(&scope.parent)
            || id.authority != self.authority
            || id.sequence <= scope.parent.id.sequence
            || root_budget != scope.parent.root_budget
            || scope.job.question != parent.input.question_digest
            || scope.parent.view != parent.input.view_seal
            || scope.parent.original_deadline != parent.original_deadline
            || request.relative_prefix().len() != usize::from(request.ply())
            || self.reciprocal.pending.contains_key(&id)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.reciprocal.pending.len() >= MAX_COORDINATOR_FRAMES {
            return Err(PalsError::Capacity);
        }
        let (stage, purpose, envelope_purpose) = match scope.recipient {
            SearchRecipient::Proposer => (
                RefinementStage::ProposalContinuation,
                RoleQueryPurpose::ProposePolicy,
                QuestionPurpose::Propose,
            ),
            SearchRecipient::Critic => (
                RefinementStage::CriticReply,
                RoleQueryPurpose::ReplyPolicy,
                QuestionPurpose::Counter,
            ),
            SearchRecipient::Researcher => return Err(RoleError::InvalidOutput.into()),
        };
        let mut prefix = parent.input.question.context.prefix.clone();
        prefix.extend_from_slice(request.relative_prefix());
        if prefix.len() > 256 {
            return Err(PalsError::Capacity);
        }
        let mut replay = parent.input.question.position.clone();
        for movement in request.relative_prefix() {
            replay.make_move(*movement)?;
        }
        if !replay.snapshot().same_state(&request.position().snapshot()) {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let node = self.replay_node(&prefix)?;
        if !self.engine.nodes[node]
            .position
            .snapshot()
            .same_state(&request.position().snapshot())
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let mut input = self.input_with_purpose(
            node,
            stage,
            parent.input.question.first,
            prefix.clone(),
            Vec::new(),
            None,
            None,
            None,
            false,
            None,
            None,
            purpose,
        )?;
        // The selected input digest covers model-facing contents, not issuance
        // sequence. Keep the old V5 question digest path byte-for-byte intact.
        let semantic = self.reciprocal_semantic_digest(&input, root_budget)?;
        let canonical = self
            .reciprocal
            .pending
            .values()
            .find(|held| held.input.question_digest == semantic);
        let anchor;
        if let Some(canonical) = canonical {
            input = canonical.input.clone();
            anchor = Arc::clone(&canonical.anchor);
        } else {
            let mut query = input.question.as_ref().clone();
            query.question_digest = semantic;
            input.question_digest = semantic;
            input.question = Arc::new(query);
            anchor = Arc::new(
                self.issue_rules_position(request.position().clone())?
                    .export()
                    .map_err(|_| RoleError::InvalidOutput)?,
            );
        }
        let envelope = QuestionEnvelope {
            id,
            parent: Some(scope.parent.id),
            sender: SearchRecipient::Researcher,
            recipient: scope.recipient,
            purpose: envelope_purpose,
            anchor: anchor.snapshot().identity(),
            anchor_root_ply: ChessPly(
                u16::try_from(prefix.len()).map_err(|_| PalsError::Capacity)?,
            ),
            original_first_move: Some(parent.original_first_move),
            prefix_digest: ContractDigest(role_line_sha256(b"prefix", &prefix)?),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: semantic,
            view: input.view_seal,
            evidence_revision: input.question.context.public_revision,
            root_budget,
            original_deadline: parent.original_deadline,
        };
        self.reciprocal.pending.insert(
            id,
            PendingCpuQuestion {
                envelope,
                input: input.clone(),
                request: request.clone(),
                anchor,
            },
        );
        Ok(AdaptiveCpuReciprocalQuestion {
            question: envelope,
            input,
        })
    }

    fn reciprocal_semantic_digest(
        &self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        budget: ContractDigest,
    ) -> Result<ContractDigest, PalsError> {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-exact-reciprocal-question/1\0");
        hash.update([input.stage as u8]);
        hash.update(input.view.digest());
        hash.update(input.view.policy_digest());
        hash.update(input.public_context.0);
        hash.update(input.question.context.focus_prefix_sha256());
        hash.update(self.rules.snapshot().identity().semantic.0);
        hash.update(budget.0);
        let model = self.engine.model.identity().as_bytes();
        hash.update((model.len() as u64).to_le_bytes());
        hash.update(model);
        if let Some(value) = self.engine.model.value_identity() {
            for text in [
                &value.semantics,
                &value.model,
                &value.encoding,
                &value.precision,
            ] {
                hash.update((text.len() as u64).to_le_bytes());
                hash.update(text.as_bytes());
            }
            hash.update(value.model_epoch);
        }
        if let Some(records) = &input.question.legacy_full_records {
            hash.update(records.digest());
        }
        Ok(ContractDigest(hash.finalize().into()))
    }

    /// Admission/acceptance run on original ticket data. The legacy fixed
    /// PALS transition is intentionally not invoked for a one-question answer.
    pub fn consume_cpu_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        _physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<Arc<AdaptiveAcceptedReciprocalAnswer>, PalsError> {
        self.check_work(work)?;
        if let Some(answer) = self
            .reciprocal
            .answers
            .iter()
            .find(|answer| answer.ticket == *accepted.ticket())
        {
            self.verify_registered_model_namespace(accepted.ticket(), output)?;
            if answer.evidence != accepted.output_digest()
                || !accepted.validates_output(output)
                || accepted.ticket().question != work.input.question_digest
                || accepted.ticket().view != work.input.view_seal
                || !Arc::ptr_eq(&answer.view, &work.input.view)
                || !self.role_consumption_recorded(accepted.ticket(), &work.input.view)?
            {
                return Err(RoleError::InvalidOutput.into());
            }
            return Ok(Arc::clone(answer));
        }
        self.validate_role(work, accepted.ticket(), output)?;
        if !accepted.validates_output(output) {
            return Err(RoleError::InvalidOutput.into());
        }
        let pending = self
            .reciprocal
            .pending
            .values()
            .find(|held| {
                held.envelope.question_digest == work.input.question_digest
                    && held.input.view_seal == work.input.view_seal
            })
            .ok_or(RoleError::InvalidOutput)?;
        if !matches!(
            work.input.question.context.purpose,
            RoleQueryPurpose::ProposePolicy | RoleQueryPurpose::ReplyPolicy
        ) {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.counters.role_calls >= self.engine.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        let ranked = role_commit::rank_moves(&work.input.view.query().legal, output)?;
        let answer = Arc::new(AdaptiveAcceptedReciprocalAnswer {
            question: pending.envelope,
            ticket: *accepted.ticket(),
            evidence: accepted.output_digest(),
            ranked,
            view: Arc::clone(&work.input.view),
        });
        self.reciprocal
            .answers
            .retain(|answer| Arc::strong_count(answer) > 1);
        if self.reciprocal.answers.len() >= MAX_COORDINATOR_FRAMES {
            return Err(PalsError::Capacity);
        }
        self.accepted.insert(accepted.ticket().id);
        let previous = self
            .committed_role_questions
            .insert(accepted.ticket().id, accepted.ticket().question);
        if let Err(error) = self.role_consumed(accepted.ticket(), &work.input.view) {
            self.accepted.remove(&accepted.ticket().id);
            if let Some(previous) = previous {
                self.committed_role_questions
                    .insert(accepted.ticket().id, previous);
            } else {
                self.committed_role_questions.remove(&accepted.ticket().id);
            }
            return Err(error);
        }
        self.counters.role_calls += 1;
        self.counters.consumed_role_outputs += 1;
        match accepted.ticket().role {
            Role::Proposer => {
                self.counters.proposer_calls += 1;
                self.counters.completed_proposer_calls += 1;
                self.counters.accepted_proposer_outputs += 1;
            }
            Role::Critic => {
                self.counters.critic_calls += 1;
                self.counters.completed_critic_calls += 1;
                self.counters.accepted_critic_outputs += 1;
            }
            Role::Validator => return Err(RoleError::InvalidOutput.into()),
        }
        self.reciprocal.answers.push_back(Arc::clone(&answer));
        Ok(answer)
    }

    /// A joined caller receives a reply for its own original opaque request;
    /// the shared output remains labelled with the canonical submitted ticket.
    pub fn cpu_reciprocal_reply(
        &mut self,
        answer: &AdaptiveAcceptedReciprocalAnswer,
        request: &CpuReciprocalRequest,
        alias: ReciprocalQuestionId,
    ) -> Result<CpuReciprocalReply, PalsError> {
        self.check_live()?;
        let pending = self
            .reciprocal
            .pending
            .get(&alias)
            .ok_or(RoleError::InvalidOutput)?;
        if pending.envelope.question_digest != answer.question.question_digest
            || pending.input.view_seal != answer.ticket.view
            || !Arc::ptr_eq(&pending.input.view, &answer.view)
            || !pending.request.same_request(request)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let preferred = answer
            .ranked
            .iter()
            .copied()
            .filter(|movement| pending.request.candidates().contains(movement))
            .collect();
        let reply = CpuReciprocalReply::new(
            &pending.request,
            pending.envelope,
            &pending.anchor,
            preferred,
        )?;
        Ok(reply)
    }

    /// Drop only a logical pin/admission. The manager still retains and drains
    /// every native ticket and the R worker's private continuation separately.
    pub fn discard_cpu_reciprocal(&mut self, id: ReciprocalQuestionId) -> bool {
        self.reciprocal.pending.remove(&id).is_some()
    }

    /// Preserve actual CPU usage and the original Store execution as Paused.
    /// No frontier score, incomplete frame or model output is published here.
    pub fn cpu_reciprocal_wait(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
    ) -> Result<(), PalsError> {
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?
            .clone();
        let job = admission.job.ok_or(RoleError::InvalidOutput)?;
        let execution = admission.execution.ok_or(RoleError::InvalidOutput)?;
        let request = receipt
            .reciprocal_request
            .as_ref()
            .ok_or(RoleError::InvalidOutput)?;
        let result = receipt
            .result
            .as_ref()
            .map_err(|error| PalsError::Cpu(error.clone()))?;
        let actual = receipt
            .actual_work
            .ok_or(StoreError::InvalidEvidence("reciprocal CPU work unknown"))?;
        let cumulative = admission
            .cumulative_nodes
            .checked_add(actual.nodes)
            .ok_or(PalsError::Capacity)?;
        if admission.terminal
            || admission.quarantined
            || admission.frame != work.frame
            || work.frame.authority != self.authority
            || work.original_deadline != self.deadline
            || admission.grant != Some(receipt.grant)
            || admission.owner != Some(receipt.grant.owner)
            || job.id != receipt.grant.job
            || request.scope().job != job
            || request.scope().owner != receipt.grant.owner
            || self
                .reciprocal
                .parents
                .get(&(work.frame, work.input.question_digest))
                != Some(&request.scope().parent)
            || result.report.completion != CpuCompletion::AwaitReciprocal
            || result.report.resume.is_some()
            || receipt.terminal
            || receipt.admission_rejected
            || !receipt.continuation_retained
            || result.report.nodes != actual.nodes
            || result.report.quiescence_nodes != actual.quiescence_nodes
            || result.report.tt_hits != actual.tt_hits
            || actual.nodes > receipt.grant.node_credit
            || actual.nodes == 0
            || actual.quiescence_nodes > actual.nodes
            || cumulative != receipt.cumulative_nodes
            || cumulative > admission.credit
            || receipt.progress_sequence < admission.progress_sequence
            || admission.context != work.input.question.context
            || admission.view_seal != work.input.view_seal
            || self.seal(&work.input.view) != admission.view_seal
            || work.input.question.node != admission.node
            || work.input.question.question_digest != work.input.question_digest
            || !work
                .input
                .question
                .position
                .snapshot()
                .same_state(&admission.position.snapshot())
            || admission.first_move != work.original_first_move
            || admission.stage != work.input.stage
            || job.hard_deadline != work.original_deadline
            || self.engine.stores.tasks.get(execution)?.key != admission.task_key
            || self.engine.stores.tasks.get(execution)?.status != TaskStatus::InFlight
        {
            return Err(StoreError::InvalidEvidence(
                "reciprocal CPU owner/usage/Store receipt mismatch",
            )
            .into());
        }
        let nodes = self
            .counters
            .cpu_nodes
            .checked_add(actual.nodes)
            .ok_or(PalsError::Capacity)?;
        let qnodes = self
            .counters
            .cpu_quiescence_nodes
            .checked_add(actual.quiescence_nodes)
            .ok_or(PalsError::Capacity)?;
        let hits = self
            .counters
            .cpu_tt_hits
            .checked_add(actual.tt_hits)
            .ok_or(PalsError::Capacity)?;
        let tasks = self
            .counters
            .cpu_tasks
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        if nodes > self.limits.max_cpu_nodes {
            return Err(
                StoreError::InvalidEvidence("reciprocal wait exceeds root CPU credit").into(),
            );
        }
        self.engine
            .stores
            .pause_task(execution, job.id.sequence, None)?;
        self.counters.cpu_nodes = nodes;
        self.counters.cpu_quiescence_nodes = qnodes;
        self.counters.cpu_tt_hits = hits;
        self.counters.cpu_tasks = tasks;
        let stored = self
            .cpu_admissions
            .get_mut(&work.input.question_digest)
            .expect("checked native R admission");
        stored.grant = None;
        stored.cumulative_nodes = cumulative;
        stored.progress_sequence = receipt.progress_sequence;
        stored.last_started_sequence = receipt.grant.execution.sequence;
        stored.continuation_retained = true;
        stored.settled_receipt = Some((
            receipt.grant.execution,
            execution,
            cpu_commit::cpu_receipt_digest(receipt),
        ));
        Ok(())
    }
}
