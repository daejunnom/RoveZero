//! Full fixed-manager versus synchronous Search trace, with a CPU-authored
//! deterministic neural fixture. No ORT, GPU, NN weights or external engine.
//!
//! This is one complete round, line length two, with ordinary finite CPU
//! verification/discovery and C reply. It must not pass by stopping at ranking,
//! a one-node budget, a matching fallback move, or an unconsumed native output.
//! Preparation runs the real anchor-aware V3 encoder and CPU tensor preparation.
//! Native ack and Search consumption remain separate observed events.
use crate::pals_adaptive::{
    AdaptiveExecutionManager, AdaptiveManagerRegistration, AdaptivePortDriver, ManagerPhase,
};
use crate::pals_native::pals_history_digest;
use crate::pals_native::ticketed::{
    TicketExecutionEvidence, TicketNativeBackend, TicketPhysicalResult, TicketPreparation,
    TicketServiceLimits, TicketedNativeService,
};
use crate::pals_native::virtual_encoding::prepare_selected_virtual_review_input;
use rz_contracts::pals::{RepresentationKey, RoleOutput, RolePayload, SearchAuthority};
use rz_contracts::pals_adaptive::{CpuWorkerId, RoleTicketEnvelope};
use rz_contracts::{
    Digest, ExecutionId, GameGeneration, OwnerId, PrecisionProfile, ProcessEpoch, RootGeneration,
    Wdl,
};
use rz_eval::pals_model::virtual_review::{
    self as virtual_review, PalsVirtualReviewInput, ReviewExecutionIdentity,
};
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_runtime::contracts::{ContractClock as _, ContractSystemClock};
use rz_search::cpu::CpuEngine;
use rz_search::cpu_worker::CpuWorker;
use rz_search::external_cpu::{CpuEndpointSelection, ExternalSlicePolicy};
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveEnginePort, AdaptiveModelNamespace, AdaptivePortModel, AdaptivePortQuestion,
};
use rz_search::pals::engine::{
    DivergenceQuery, LegalOrderRoleMock, PalsCompletion, PalsConfig, PalsCounters, PalsEngine,
    PalsLimits, RoleAcceptance, RoleError, RoleEvaluation, RoleLogicalContext, RoleModel,
    RoleQuery, RoleQueryPurpose,
};
use rz_search::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy, ReviewView};
use rz_search::pals::store::{ObservationId, PalsStores};
use rz_search::pals::value::ModelValueIdentity;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const WINDOW: Duration = Duration::from_secs(5);
const NODE_CREDIT: u64 = 100_000;
const MODEL_EPOCH: [u8; 32] = [51; 32];
const BACKEND: [u8; 32] = [53; 32];

fn encoding() -> [u8; 32] {
    Sha256::digest(virtual_review::ENCODING.as_bytes()).into()
}

/// Only values actually read by the tensor ABI. Exact cache keys and logical
/// Store handles are recorded separately below, not silently normalized into
/// model features. Float bits include padding and masks, not approximate QA.
#[derive(Debug, Eq, PartialEq)]
struct TensorTrace {
    anchors: usize,
    records_count: usize,
    candidates_count: usize,
    divergences_count: usize,
    length_bucket: usize,
    token_count: usize,
    anchor_board: Vec<i64>,
    anchor_metadata: Vec<u32>,
    anchor_mask: Vec<bool>,
    records: Vec<u32>,
    record_anchor: Vec<i64>,
    record_mask: Vec<bool>,
    record_line_tokens: Vec<i64>,
    record_line_mask: Vec<bool>,
    record_relations: Vec<i64>,
    query: [u32; 8],
    query_anchor: [i64; 3],
    query_line_tokens: Vec<i64>,
    query_line_mask: Vec<bool>,
    candidates: Vec<i64>,
    candidate_mask: Vec<bool>,
    divergence_features: Vec<u32>,
    divergence_mask: Vec<bool>,
}
impl TensorTrace {
    fn prepare(input: &PalsVirtualReviewInput) -> Result<Self, RoleError> {
        let tensors = input
            .prepare()
            .map_err(|error| RoleError::Backend(error.to_string()))?;
        let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect();
        Ok(Self {
            anchors: tensors.anchors,
            records_count: tensors.records_count,
            candidates_count: tensors.candidates_count,
            divergences_count: tensors.divergences_count,
            length_bucket: tensors.length_bucket,
            token_count: tensors.token_count,
            anchor_board: tensors.anchor_board,
            anchor_metadata: bits(tensors.anchor_metadata),
            anchor_mask: tensors.anchor_mask,
            records: bits(tensors.records),
            record_anchor: tensors.record_anchor,
            record_mask: tensors.record_mask,
            record_line_tokens: tensors.record_line_tokens,
            record_line_mask: tensors.record_line_mask,
            record_relations: tensors.record_relations,
            query: tensors.query.map(f32::to_bits),
            query_anchor: tensors.query_anchor,
            query_line_tokens: tensors.query_line_tokens,
            query_line_mask: tensors.query_line_mask,
            candidates: tensors.candidates,
            candidate_mask: tensors.candidate_mask,
            divergence_features: bits(tensors.divergence_features),
            divergence_mask: tensors.divergence_mask,
        })
    }
}

#[derive(Debug)]
struct QuestionTrace {
    context: RoleLogicalContext,
    focus_history: Vec<String>,
    anchor_history: Vec<Vec<String>>,
    prefix: Vec<BoardMove>,
    proposal: Vec<BoardMove>,
    counter: Option<Vec<BoardMove>>,
    legal: Vec<BoardMove>,
    divergence_sites: Vec<usize>,
    view_digest: [u8; 32],
    record_scope: [u8; 32],
    context_policy: [u8; 32],
    execution_identity: ReviewExecutionIdentity,
    input_key: [u8; 32],
    tensors: TensorTrace,
}
impl QuestionTrace {
    fn new(
        position: &Position,
        context: &RoleLogicalContext,
        view: &ReviewView,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
    ) -> Result<Self, RoleError> {
        let query = view.query();
        Ok(Self {
            context: context.clone(),
            focus_history: position.snapshot().known_history_fens(),
            anchor_history: view
                .anchors()
                .iter()
                .map(|anchor| anchor.snapshot.known_history_fens())
                .collect(),
            prefix: query.prefix.clone(),
            proposal: query.proposal.clone(),
            counter: query.counterexample.clone(),
            legal: query.legal.clone(),
            divergence_sites: query.divergences.clone(),
            view_digest: view.digest(),
            record_scope: view.record_scope_digest(),
            context_policy: view.policy_digest(),
            execution_identity: identity.clone(),
            input_key: input
                .fresh_key(identity)
                .map_err(|error| RoleError::Backend(error.to_string()))?
                .0,
            tensors: TensorTrace::prepare(input)?,
        })
    }
}

#[derive(Default)]
struct TraceFacts {
    questions: Mutex<Vec<QuestionTrace>>,
    accepted_contexts: Mutex<Vec<RoleLogicalContext>>,
    native_run: Mutex<Vec<RoleTicketEnvelope>>,
    native_accept: Mutex<Vec<RoleTicketEnvelope>>,
    native_discard: Mutex<Vec<RoleTicketEnvelope>>,
    execution_evidence: Mutex<Vec<RoleTicketEnvelope>>,
    native_finished: AtomicBool,
}

fn value_identity() -> ModelValueIdentity {
    let mut identity = LegalOrderRoleMock.value_identity().unwrap().clone();
    identity.model_epoch = MODEL_EPOCH;
    identity
}
fn config() -> PalsConfig {
    PalsConfig {
        beam_width: 1,
        line_plies: 2,
        max_role_calls: 256,
        cpu_nodes_per_task: 4096,
        ..PalsConfig::default()
    }
}
fn cpu() -> CpuEngine {
    // Reuse the actual manager integration's own evaluator/order/resume profile.
    super::cpu_engine()
}
fn policy() -> ReviewSelectionPolicy {
    ReviewSelectionPolicy::default()
}
fn limits(until: Instant) -> PalsLimits {
    PalsLimits {
        deadline: until,
        max_rounds: 1,
        max_cpu_nodes: NODE_CREDIT,
        cpu_depth: 1,
    }
}

struct RecordingModel {
    inner: LegalOrderRoleMock,
    value: ModelValueIdentity,
    pending: Option<(Arc<ReviewView>, RoleLogicalContext)>,
    delivered: Option<RoleLogicalContext>,
    facts: Arc<TraceFacts>,
}
impl RecordingModel {
    fn observe_role(&mut self, query: &RoleQuery<'_>) -> Result<(), RoleError> {
        query.check_control()?;
        let (view, context) = self.pending.take().ok_or(RoleError::InvalidOutput)?;
        if query.legal != view.query().legal
            || query.prefix != view.query().prefix
            || query.proposal != view.query().proposal
            || query.counterexample != view.query().counterexample.as_deref()
            || query.revision != context.public_revision
        {
            return Err(RoleError::InvalidOutput);
        }
        let (input, identity) = prepare_selected_virtual_review_input(
            query.position,
            &context,
            &view,
            MODEL_EPOCH,
            BACKEND,
        )?;
        let trace = QuestionTrace::new(query.position, &context, &view, &input, &identity)?;
        self.facts.questions.lock().unwrap().push(trace);
        self.delivered = Some(context);
        Ok(())
    }
}
impl RoleModel for RecordingModel {
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn supports_selected_review_view(&self) -> bool {
        true
    }
    fn value_identity(&self) -> Option<&ModelValueIdentity> {
        Some(&self.value)
    }
    fn bind_selected_review_view(
        &mut self,
        view: Arc<ReviewView>,
        context: &RoleLogicalContext,
    ) -> Result<(), RoleError> {
        self.inner
            .bind_selected_review_view(Arc::clone(&view), context)?;
        if self.pending.is_some() || self.delivered.is_some() {
            return Err(RoleError::InvalidOutput);
        }
        self.pending = Some((view, context.clone()));
        Ok(())
    }
    fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.observe_role(&query)?;
        self.inner.propose(query)
    }
    fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.observe_role(&query)?;
        self.inner.reply(query)
    }
    fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.observe_role(&query)?;
        self.inner.repair(query)
    }
    fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
        check_preparation(query.deadline, query.cancel)?;
        let (view, context) = self.pending.take().ok_or(RoleError::InvalidOutput)?;
        if query.proposal != view.query().proposal
            || query.candidates != view.query().divergences
            || query.revision != context.public_revision
        {
            return Err(RoleError::InvalidOutput);
        }
        let (input, identity) = prepare_selected_virtual_review_input(
            query.root,
            &context,
            &view,
            MODEL_EPOCH,
            BACKEND,
        )?;
        self.facts
            .questions
            .lock()
            .unwrap()
            .push(QuestionTrace::new(
                query.root, &context, &view, &input, &identity,
            )?);
        self.delivered = Some(context);
        self.inner.divergences(query)
    }
    fn accepted_output_checked(&mut self, acceptance: RoleAcceptance<'_>) -> Result<(), RoleError> {
        acceptance.check_control()?;
        let original = self.delivered.take().ok_or(RoleError::InvalidOutput)?;
        if &original != acceptance.context {
            return Err(RoleError::InvalidOutput);
        }
        self.facts.accepted_contexts.lock().unwrap().push(original);
        Ok(())
    }
}

struct PreparedFixture {
    input: PalsVirtualReviewInput,
    identity: ReviewExecutionIdentity,
    key: RepresentationKey,
}
struct TraceNative {
    clock: ContractSystemClock,
    inner: LegalOrderRoleMock,
    facts: Arc<TraceFacts>,
    pending: BTreeSet<rz_contracts::RequestId>,
}
fn check_preparation(until: Instant, canceled: &AtomicBool) -> Result<(), RoleError> {
    if canceled.load(Ordering::Acquire) {
        return Err(RoleError::Canceled);
    }
    if Instant::now() >= until {
        return Err(RoleError::Deadline);
    }
    Ok(())
}
fn encoded_question(
    question: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
    canceled: &AtomicBool,
) -> Result<(PalsVirtualReviewInput, ReviewExecutionIdentity), RoleError> {
    check_preparation(question.query.deadline, canceled)?;
    prepare_selected_virtual_review_input(
        &question.query.position,
        &question.query.context,
        &question.view,
        MODEL_EPOCH,
        BACKEND,
    )
}
fn failure(error: RoleError) -> RoleFailure {
    match error {
        RoleError::Canceled => RoleFailure::Canceled,
        RoleError::Deadline => RoleFailure::Expired,
        _ => RoleFailure::InvalidOutput,
    }
}
impl TicketNativeBackend<ReviewView, AdaptivePortQuestion> for TraceNative {
    type Prepared = PreparedFixture;
    type Output = RoleOutput;
    fn clock(&self) -> ContractSystemClock {
        self.clock.clone()
    }
    fn prepare(
        &mut self,
        question: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        canceled: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        let (input, identity) = encoded_question(question, canceled).map_err(failure)?;
        let mut candidates = Sha256::new();
        for movement in &input.candidates {
            candidates.update(
                movement
                    .packed()
                    .map_err(|_| RoleFailure::InvalidOutput)?
                    .to_le_bytes(),
            );
        }
        let tensors = input.prepare().map_err(|_| RoleFailure::InvalidOutput)?;
        let mut masks = Sha256::new();
        masks.update(b"rz-pals-virtual-review-physical-masks/1\0");
        for dimension in [
            tensors.anchors,
            tensors.records_count,
            tensors.candidates_count,
            tensors.divergences_count,
            tensors.length_bucket,
        ] {
            masks.update((dimension as u64).to_le_bytes());
        }
        for mask in [
            &tensors.anchor_mask,
            &tensors.record_mask,
            &tensors.record_line_mask,
            &tensors.query_line_mask,
            &tensors.candidate_mask,
            &tensors.divergence_mask,
        ] {
            masks.update((mask.len() as u64).to_le_bytes());
            for present in mask {
                masks.update([u8::from(*present)]);
            }
        }
        let key = RepresentationKey {
            input: Digest(
                input
                    .fresh_key(&identity)
                    .map_err(|_| RoleFailure::InvalidOutput)?
                    .0,
            ),
            // Cache scope retains the request endpoint, while the V3 execution
            // identity separately seals anchor zero and every selected anchor.
            history: Digest(pals_history_digest(&question.query.position).map_err(failure)?),
            candidates: Digest(candidates.finalize().into()),
            mask_positions: Digest(masks.finalize().into()),
            model: Digest(MODEL_EPOCH),
            encoding: Digest(encoding()),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 1,
            record_revision: question.query.context.public_revision,
            role: question.role,
        };
        Ok((
            PreparedFixture {
                input,
                identity,
                key,
            },
            key,
        ))
    }
    fn prepared_host_bytes(&self, prepared: &Self::Prepared) -> u64 {
        // The bounded real tensor preparation supplies a conservative owned
        // declaration. This fixture is not an observed RSS/VRAM peak.
        let tensors = prepared.input.prepare().unwrap();
        (std::mem::size_of::<PreparedFixture>()
            + prepared.input.owned_bytes()
            + prepared.identity.anchor_history_seals.capacity() * std::mem::size_of::<[u8; 32]>()
            + tensors.owned_bytes()) as u64
    }
    fn question_host_bytes(
        &self,
        question: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
    ) -> u64 {
        question.query.owned_host_bytes().unwrap() + question.view.owned_host_bytes().unwrap()
    }
    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        prepared: &Self::Prepared,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        let result = (|| -> Result<RoleOutput, RoleFailure> {
            if ticket.representation != prepared.key
                || ticket.question != question.question
                || ticket.view != question.seal
                || ticket.original_deadline != question.deadline
                || !self.pending.insert(ticket.id)
            {
                return Err(RoleFailure::InvalidOutput);
            }
            let trace = QuestionTrace::new(
                &question.query.position,
                &question.query.context,
                &question.view,
                &prepared.input,
                &prepared.identity,
            )
            .map_err(failure)?;
            self.facts.questions.lock().unwrap().push(trace);
            self.facts.native_run.lock().unwrap().push(*ticket);
            let query = question.view.query();
            let context = &question.query.context;
            let role_query = || RoleQuery {
                position: &question.query.position,
                legal: &query.legal,
                prefix: &query.prefix,
                proposal: &query.proposal,
                counterexample: query.counterexample.as_deref(),
                records: question.view.records(),
                revision: context.public_revision,
                deadline: question.query.deadline,
                cancel: canceled,
            };
            let wdl = Wdl::try_new(0.25, 0.5, 0.25, 1e-4).unwrap();
            let payload = match context.purpose {
                RoleQueryPurpose::DivergencePolicy => RolePayload::Counterexample {
                    candidate_logits: vec![],
                    divergence_logits: self
                        .inner
                        .divergences(DivergenceQuery {
                            root: &question.query.position,
                            proposal: &query.proposal,
                            candidates: &query.divergences,
                            records: question.view.records(),
                            revision: context.public_revision,
                            deadline: question.query.deadline,
                            cancel: canceled,
                        })
                        .map_err(failure)?,
                    wdl,
                },
                RoleQueryPurpose::ReplyPolicy => RolePayload::Counterexample {
                    candidate_logits: self.inner.reply(role_query()).map_err(failure)?.logits,
                    divergence_logits: vec![],
                    wdl,
                },
                RoleQueryPurpose::RepairPolicy => RolePayload::Proposal {
                    candidate_logits: self.inner.repair(role_query()).map_err(failure)?.logits,
                    wdl,
                },
                RoleQueryPurpose::ProposePolicy => RolePayload::Proposal {
                    candidate_logits: self.inner.propose(role_query()).map_err(failure)?.logits,
                    wdl,
                },
                _ => return Err(RoleFailure::Unsupported("fixed own-raw fixture purpose")),
            };
            Ok(RoleOutput {
                id: ticket.id,
                authority: ticket.authority,
                key: ticket.representation,
                payload,
                private_latent: vec![0.; 16 * 384],
            })
        })();
        // The mock's physical computation is wholly complete on this owner;
        // cancellation is not used to synthesize a GPU fence.
        TicketPhysicalResult {
            physical: PhysicalRoleCompletion::Confirmed,
            output: result,
        }
    }
    fn output_host_bytes(&self, output: &Self::Output) -> u64 {
        let head = match &output.payload {
            RolePayload::Proposal {
                candidate_logits, ..
            } => candidate_logits.len(),
            RolePayload::Counterexample {
                candidate_logits,
                divergence_logits,
                ..
            } => candidate_logits.len() + divergence_logits.len(),
            RolePayload::TaskRanking { task_logits } => task_logits.len(),
        };
        ((head + output.private_latent.len()) * 4) as u64
    }
    fn acceptance_output_digest(&self, output: &Self::Output) -> Option<Digest> {
        rz_search::pals::coordinator::native_acceptance_output_digest(output).ok()
    }
    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        self.facts.execution_evidence.lock().unwrap().push(*ticket);
        Some(TicketExecutionEvidence {
            execution: Some(ExecutionId::new(ticket.id.epoch, ticket.id.sequence)),
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
            accounting: None,
        })
    }
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        prepared: &Self::Prepared,
        output: &Self::Output,
    ) -> Result<(), RoleFailure> {
        if output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != prepared.key
            || ticket.view != question.seal
            || !self.pending.remove(&ticket.id)
        {
            return Err(RoleFailure::InvalidOutput);
        }
        self.facts.native_accept.lock().unwrap().push(*ticket);
        self.facts
            .accepted_contexts
            .lock()
            .unwrap()
            .push(question.query.context.clone());
        Ok(())
    }
    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        if !self.pending.remove(&ticket.id) {
            return Err(RoleFailure::InvalidOutput);
        }
        self.facts.native_discard.lock().unwrap().push(*ticket);
        Ok(())
    }
    fn quiesce(&mut self, _: Instant) -> bool {
        self.pending.is_empty()
    }
    fn finish(&mut self, _: Instant) -> bool {
        self.facts.native_finished.store(true, Ordering::Release);
        self.pending.is_empty()
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CpuObservationTrace {
    anchor_history: Vec<String>,
    pv: Vec<BoardMove>,
    score: String,
    scope: String,
    condition: String,
    value_identity: String,
    budget: u64,
}
fn cpu_observations(stores: &PalsStores) -> Vec<CpuObservationTrace> {
    (0..stores.observations.next_observation_sequence())
        .filter_map(|sequence| {
            let observation = stores.observations.get(ObservationId(sequence)).unwrap();
            let condition = observation.cpu_condition.as_ref()?;
            Some(CpuObservationTrace {
                anchor_history: stores
                    .states
                    .get(observation.state)
                    .unwrap()
                    .known_history_fens(),
                pv: observation
                    .cpu_pv
                    .map(|line| stores.lines.moves(line).unwrap())
                    .unwrap_or_default(),
                score: format!("{:?}", observation.score),
                scope: format!("{:?}", observation.scope),
                condition: condition.clone(),
                value_identity: format!("{:?}", observation.value_identity),
                budget: observation.budget,
            })
        })
        .collect()
}

fn compare_questions(legacy: &[QuestionTrace], adaptive: &[QuestionTrace]) {
    assert_eq!(
        legacy.len(),
        adaptive.len(),
        "full role trace cardinality/order changed\nlegacy={legacy:#?}\nadaptive={adaptive:#?}"
    );
    for (index, (a, b)) in legacy.iter().zip(adaptive).enumerate() {
        assert_eq!(a.context.purpose, b.context.purpose, "question {index}");
        assert_eq!(
            a.focus_history, b.focus_history,
            "exact endpoint/history question {index}"
        );
        assert_eq!(
            a.anchor_history, b.anchor_history,
            "exact root/anchor histories question {index}"
        );
        assert_eq!(
            (
                &a.prefix,
                &a.proposal,
                &a.counter,
                &a.legal,
                &a.divergence_sites
            ),
            (
                &b.prefix,
                &b.proposal,
                &b.counter,
                &b.legal,
                &b.divergence_sites
            ),
            "ordered query question {index}"
        );
        assert_eq!(
            a.context_policy, b.context_policy,
            "context policy question {index}"
        );
        assert_eq!(
            a.execution_identity, b.execution_identity,
            "exact V3 root/anchor history and execution identity question {index}"
        );
        assert_eq!(
            a.input_key, b.input_key,
            "exact V3 Fresh key question {index}"
        );
        assert_eq!(
            a.tensors, b.tensors,
            "anchor-aware V3 tensor bits question {index}\nsynchronous context={:?}\nadaptive context={:?}",
            a.context, b.context
        );
        assert_eq!(
            a.context.focus_prefix_sha256(),
            b.context.focus_prefix_sha256(),
            "exact Rules line question {index}"
        );
        assert_eq!(
            (a.context.game_generation, a.context.search_generation),
            (b.context.game_generation, b.context.search_generation),
            "question root generation {index}"
        );
        // These are observed cache/provenance identities, not model features.
        // Preserve any version/owner-layout difference visibly. Equality of
        // tensor bits does not claim cross-route cache-key or handle reuse.
        if a.context != b.context
            || a.input_key != b.input_key
            || a.view_digest != b.view_digest
            || a.record_scope != b.record_scope
        {
            eprintln!(
                "fixed trace question {index} non-tensor metadata:\nlegacy context={:?}; input={:?}; view={:?}; public={:?}\nadaptive context={:?}; input={:?}; view={:?}; public={:?}",
                a.context,
                a.input_key,
                a.view_digest,
                a.record_scope,
                b.context,
                b.input_key,
                b.view_digest,
                b.record_scope
            );
        }
    }
}
fn compare_work(a: PalsCounters, b: PalsCounters) {
    assert_eq!(
        (a.rounds, a.proposals, a.refutations, a.repairs),
        (b.rounds, b.proposals, b.refutations, b.repairs),
        "complete refinement branches"
    );
    assert_eq!(
        (
            a.role_calls,
            a.proposer_calls,
            a.critic_calls,
            a.repair_calls,
            a.consumed_role_outputs
        ),
        (
            b.role_calls,
            b.proposer_calls,
            b.critic_calls,
            b.repair_calls,
            b.consumed_role_outputs
        ),
        "actual role consumption"
    );
    assert_eq!(
        (
            a.cpu_tasks_requested,
            a.cpu_nodes,
            a.cpu_quiescence_nodes,
            a.cpu_tt_hits
        ),
        (
            b.cpu_tasks_requested,
            b.cpu_nodes,
            b.cpu_quiescence_nodes,
            b.cpu_tt_hits
        ),
        "actual own CPU work"
    );
    assert_eq!(
        (
            a.completed_cpu_tasks,
            a.partial_cpu_iterations,
            a.consumed_cpu_tasks,
            a.consumed_partial_cpu_values,
            a.consumed_frontier_cpu_values,
            a.consumed_cached_cpu_values
        ),
        (
            b.completed_cpu_tasks,
            b.partial_cpu_iterations,
            b.consumed_cpu_tasks,
            b.consumed_partial_cpu_values,
            b.consumed_frontier_cpu_values,
            b.consumed_cached_cpu_values
        ),
        "CPU completion/consumption scope"
    );
    assert!(!a.cpu_work_observation_incomplete && !b.cpu_work_observation_incomplete);
}

#[test]
fn complete_line_two_fixed_manager_matches_synchronous_question_tensor_and_work_trace() {
    // Sparse, nonterminal tactical state keeps the full round finite while
    // still providing opponent alternatives to both CPU discovery and C reply.
    let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let legacy_facts = Arc::new(TraceFacts::default());
    let mut legacy = PalsEngine::new(
        config(),
        RecordingModel {
            inner: LegalOrderRoleMock,
            value: value_identity(),
            pending: None,
            delivered: None,
            facts: Arc::clone(&legacy_facts),
        },
        cpu(),
    )
    .unwrap()
    .with_selected_review_context(policy(), ReviewIndexLimits::default())
    .unwrap();
    legacy.new_game();
    let baseline_until = Instant::now() + WINDOW;
    let baseline = legacy
        .search(&position, limits(baseline_until), &AtomicBool::new(false))
        .unwrap();
    assert_eq!(baseline.completion, PalsCompletion::RoundLimit);
    assert_eq!(baseline.counters.rounds, 1);
    assert!(
        baseline.counters.cpu_nodes > 0 && baseline.counters.critic_calls >= 2,
        "baseline fixture did not execute CPU discovery and both C heads"
    );
    assert!(
        legacy_facts
            .questions
            .lock()
            .unwrap()
            .iter()
            .any(|trace| trace.prefix.len() == 1)
    );
    assert!(
        legacy_facts
            .questions
            .lock()
            .unwrap()
            .iter()
            .any(|trace| trace.proposal.len() == 2),
        "baseline fixture did not encode the complete two-ply proposal"
    );
    let legacy_cpu = cpu_observations(legacy.stores());

    let clock = ContractSystemClock::new(ProcessEpoch(331));
    let until = clock.origin() + WINDOW;
    let authority = SearchAuthority {
        epoch: clock.domain().0,
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([61; 32]),
    };
    let value = value_identity();
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest(MODEL_EPOCH),
        Digest(encoding()),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let mirror =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap();
    let mut engine = PalsEngine::new(config(), mirror, cpu())
        .unwrap()
        .with_selected_review_context(policy(), ReviewIndexLimits::default())
        .unwrap();
    engine.new_game();
    let owner_sequence = Arc::new(AtomicU64::new(1));
    let port = AdaptiveEnginePort::new(
        engine,
        position.clone(),
        authority,
        rz_contracts::Deadline {
            clock: clock.domain(),
            at: rz_contracts::MonotonicTick(WINDOW.as_nanos() as u64),
        },
        limits(until),
        Arc::new(move || Ok(OwnerId(owner_sequence.fetch_add(1, Ordering::Relaxed)))),
    )
    .unwrap();
    let facts = Arc::new(TraceFacts::default());
    let owner_facts = Arc::clone(&facts);
    let owner_clock = clock.clone();
    let native = TicketedNativeService::spawn(
        move || {
            Ok(TraceNative {
                clock: owner_clock,
                inner: LegalOrderRoleMock,
                facts: owner_facts,
                pending: BTreeSet::new(),
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    let worker = CpuWorker::spawn_notified(CpuWorkerId(0), cpu(), native.signal()).unwrap();
    let mut driver = AdaptivePortDriver::new(
        port,
        until,
        Arc::new(native.signal()),
        ExternalSlicePolicy::default(),
    )
    .unwrap();
    driver.prepare_root_rules().unwrap();
    let root = driver.root_input().unwrap();
    assert_eq!(
        root.view.anchors()[0].snapshot.known_history_fens(),
        position.snapshot().known_history_fens()
    );
    let mut manager = AdaptiveExecutionManager::new(
        AdaptiveManagerRegistration {
            authority,
            hard_deadline: until,
            token_budget: 512,
            policy: CoordinatorPolicy::Fixed,
            first_request_sequence: 1,
            native_cpu_node_limit: NODE_CREDIT,
            endpoints: CpuEndpointSelection::deployment(),
            controller: ControllerConfig::default(),
            device_digest: Digest([62; 32]),
            cpu_device_digest: Digest([63; 32]),
        },
        clock,
        native,
        vec![worker],
        None,
        driver,
    )
    .unwrap();
    manager.prepare_root_ranking(root).unwrap();
    // Use the actual manager runloop, including root scheduling, not a manually
    // copied refinement state machine or hand-issued stage subset.
    let receipt = manager.run_until(until, &AtomicBool::new(false)).unwrap();
    assert_eq!(
        manager.phase(),
        ManagerPhase::Finished,
        "{:?}",
        manager.failures().collect::<Vec<_>>()
    );
    assert!(receipt.permits_archive_close());
    assert!(
        manager.failures().next().is_none(),
        "{:?}",
        manager.failures().collect::<Vec<_>>()
    );
    assert!(facts.native_finished.load(Ordering::Acquire));
    assert_eq!(
        manager.native_cpu_usage().unwrap(),
        (receipt.native_cpu_spent, 0)
    );
    let actual = manager.driver().result(PalsCompletion::RoundLimit).unwrap();
    let port = manager.driver().port().unwrap();
    let actual_cpu = cpu_observations(port.engine().stores());
    assert_eq!(
        manager.counters().consumed_role_outputs,
        actual.counters.consumed_role_outputs
    );
    assert_eq!(receipt.native_cpu_spent, actual.counters.cpu_nodes);
    let native_run = facts.native_run.lock().unwrap();
    let native_accept = facts.native_accept.lock().unwrap();
    assert_eq!(
        *native_run, *native_accept,
        "all authored physical outputs need original native acknowledgement"
    );
    assert_eq!(
        facts.execution_evidence.lock().unwrap().len(),
        native_run.len()
    );
    assert!(facts.native_discard.lock().unwrap().is_empty());
    assert_eq!(
        native_run.len() as u64,
        manager.counters().consumed_role_outputs
    );
    assert_eq!(
        legacy_facts.accepted_contexts.lock().unwrap().len() as u64,
        baseline.counters.consumed_role_outputs
    );
    compare_questions(
        &legacy_facts.questions.lock().unwrap(),
        &facts.questions.lock().unwrap(),
    );
    assert_eq!(
        legacy_cpu, actual_cpu,
        "CPU state/restriction/score/PV/work evidence trace"
    );
    compare_work(baseline.counters, actual.counters);
    assert_eq!(
        (baseline.best_move, baseline.score, baseline.value_scope),
        (actual.best_move, actual.score, actual.value_scope),
        "final choice after full trace"
    );
    assert!(position.legal_moves().contains(&actual.best_move.unwrap()));
}

#[test]
fn natural_own_cpu_budget_end_drains_same_round_continuation_and_keeps_legal_move() {
    // Run real finite CPU jobs and the existing role/encoding mock. No spent
    // counter, CPU report or node count is injected into the manager or port.
    const CREDIT: u64 = 1024;
    let position = Position::startpos();
    let legal = position.legal_moves();
    let clock = ContractSystemClock::new(ProcessEpoch(335));
    let until = clock.origin() + WINDOW;
    let authority = SearchAuthority {
        epoch: clock.domain().0,
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([65; 32]),
    };
    let make_cpu = || {
        CpuEngine::with_resume_policy(
            rz_search::cpu::CpuConfig {
                tt_entries: 1024,
                max_depth: 4,
                quiescence_ply: 4,
                ..Default::default()
            },
            rz_search::cpu::CpuResumePolicy::PausedStack,
        )
        .unwrap()
    };
    let value = value_identity();
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest(MODEL_EPOCH),
        Digest(encoding()),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let mirror =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap();
    let mut engine = PalsEngine::new(
        PalsConfig {
            beam_width: 1,
            line_plies: 4,
            max_role_calls: 64,
            cpu_nodes_per_task: 128,
            ..PalsConfig::default()
        },
        mirror,
        make_cpu(),
    )
    .unwrap()
    .with_selected_review_context(policy(), ReviewIndexLimits::default())
    .unwrap();
    engine.new_game();
    let next_owner = Arc::new(AtomicU64::new(1));
    let port = AdaptiveEnginePort::new(
        engine,
        position,
        authority,
        rz_contracts::Deadline {
            clock: clock.domain(),
            at: rz_contracts::MonotonicTick(WINDOW.as_nanos() as u64),
        },
        PalsLimits {
            deadline: until,
            max_rounds: 4,
            max_cpu_nodes: CREDIT,
            cpu_depth: 2,
        },
        Arc::new(move || Ok(OwnerId(next_owner.fetch_add(1, Ordering::Relaxed)))),
    )
    .unwrap();
    let facts = Arc::new(TraceFacts::default());
    let owner_facts = Arc::clone(&facts);
    let owner_clock = clock.clone();
    let native = TicketedNativeService::spawn(
        move || {
            Ok(TraceNative {
                clock: owner_clock,
                inner: LegalOrderRoleMock,
                facts: owner_facts,
                pending: BTreeSet::new(),
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    let worker = CpuWorker::spawn_notified(CpuWorkerId(0), make_cpu(), native.signal()).unwrap();
    let mut driver = AdaptivePortDriver::new(
        port,
        until,
        Arc::new(native.signal()),
        ExternalSlicePolicy::default(),
    )
    .unwrap();
    driver.prepare_root_rules().unwrap();
    let root = driver.root_input().unwrap();
    let mut manager = AdaptiveExecutionManager::new(
        AdaptiveManagerRegistration {
            authority,
            hard_deadline: until,
            token_budget: 512,
            policy: CoordinatorPolicy::Fixed,
            first_request_sequence: 1,
            native_cpu_node_limit: CREDIT,
            endpoints: CpuEndpointSelection::deployment(),
            controller: ControllerConfig::default(),
            device_digest: Digest([66; 32]),
            cpu_device_digest: Digest([67; 32]),
        },
        clock,
        native,
        vec![worker],
        None,
        driver,
    )
    .unwrap();
    manager.prepare_root_ranking(root).unwrap();
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(
        manager.failures().next().is_none(),
        "{:?}",
        manager.failures().collect::<Vec<_>>()
    );
    assert!(
        manager.cpu_node_limit_reached(),
        "fixture must hit a new CPU frame after natural credit exhaustion"
    );
    assert_eq!(manager.native_cpu_usage().unwrap(), (CREDIT, 0));
    assert!(receipt.permits_root_reuse());
    assert!(
        !receipt.permits_archive_close(),
        "root fence is not a process shutdown"
    );
    let result = manager
        .driver()
        .result(PalsCompletion::CpuNodeLimit)
        .unwrap();
    assert_eq!(result.completion, PalsCompletion::CpuNodeLimit);
    assert!(
        result
            .best_move
            .is_some_and(|movement| legal.contains(&movement))
    );
    assert_eq!(result.counters.cpu_nodes, CREDIT);
    assert!(manager.counters().cpu_slice_receipts > 1);
    assert!(manager.counters().consumed_role_outputs > 1);
    assert!(!facts.native_finished.load(Ordering::Acquire));
    // Actual root fence returns the original owners. Finish the process in a
    // distinct operation; no pending ticket or same-owner stack is leaked.
    let parts = manager
        .into_persistent_parts()
        .unwrap_or_else(|rejected| panic!("{}", rejected.error));
    let (mut native, cpu, external, driver, _, _) = parts.into_components();
    assert!(external.is_none());
    assert_eq!(cpu.len(), 1);
    assert!(
        cpu[0]
            .engine
            .paused_stack_snapshot()
            .is_some_and(|stack| stack.tokens_retained == 0)
    );
    assert!(driver.port().is_some());
    native.begin_shutdown(until);
    while !native.try_join().unwrap() {
        while native.poll().is_some() {}
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.native_finished.load(Ordering::Acquire));
}
