//! Fresh adapter for the existing NativeRoleModel owner/runtime/physical worker.
//! Warm is explicitly unavailable here until a ticket-owned seed bank is installed.
use super::service::*;
use crate::pals_native::{self, NativeQueryKind, NativeRoleModel};
use rz_contracts::pals::{RepresentationKey, RoleOutput};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{Digest, ExecutionId, RequestId};
use rz_eval::pals_model::PalsModelInput;
use rz_position::Position;
use rz_runtime::contracts::ContractSystemClock;
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::engine::{
    DivergenceQuery, RoleError, RoleLogicalContext, RoleQuery, RoleQueryPurpose,
};
use rz_search::pals::review_context::ReviewView;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

#[derive(Clone)]
pub struct NativeTicketQuestion {
    pub position: Position,
    pub context: RoleLogicalContext,
    pub deadline: Instant,
}
pub struct NativePreparedTicket {
    pub input: PalsModelInput,
    pub representation: RepresentationKey,
    pub kind: NativeQueryKind,
}
pub struct NativeTicketOutput {
    pub output: RoleOutput,
    pub execution: Option<ExecutionId>,
    pub completed_nn_inputs: Option<u64>,
    pub measurement_complete: bool,
    pub physical_fresh: Option<bool>,
}
impl rz_search::pals::coordinator::ProposerRankingOutput for NativeTicketOutput {
    fn proposer_ranking_output(&self) -> &RoleOutput {
        &self.output
    }
}

pub struct NativeFreshTicketBackend {
    model: NativeRoleModel,
    evidence: Option<(RequestId, TicketExecutionEvidence)>,
}
impl NativeFreshTicketBackend {
    /// Call inside the service factory. Model loading/startup remain owned by
    /// the same dedicated thread and existing single native physical worker.
    pub fn new(mut model: NativeRoleModel, startup_deadline: Instant) -> Result<Self, RoleFailure> {
        model.ticketed_require_fresh().map_err(|_| {
            RoleFailure::Unsupported("ticket-owned Warm bank or reused synchronous owner")
        })?;
        model.prepare_startup(startup_deadline).map_err(failure)?;
        Ok(Self {
            model,
            evidence: None,
        })
    }
}
pub type NativeRoleTicketService =
    TicketedNativeService<ReviewView, NativeTicketQuestion, NativeFreshTicketBackend>;

impl TicketNativeBackend<ReviewView, NativeTicketQuestion> for NativeFreshTicketBackend {
    type Prepared = NativePreparedTicket;
    type Output = NativeTicketOutput;
    fn clock(&self) -> ContractSystemClock {
        self.model.ticketed_clock()
    }
    fn prepare(
        &mut self,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        canceled: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        let view = &question.view;
        if question.seal.view_digest.0 != view.seal_digest()
            || question.seal.context_policy.0 != view.policy_digest()
            || question.seal.view_sequence != view.sequence()
            || question.seal.actual_tokens != view.token_count()
            || usize::from(question.seal.selected_records) != view.records().len()
            || usize::from(question.seal.anchors) != view.anchors().len()
            || question.seal.length_bucket != view.length_bucket()
            || view.generation() != question.query.context.search_generation
            || view.query().purpose != question.query.context.purpose
            || question.query.context.prefix != view.query().prefix
        {
            return Err(RoleFailure::InvalidOutput);
        }
        let config = self.model.ticketed_config();
        let epoch = self.model.ticketed_model_epoch();
        let native_kind = kind(question.query.context.purpose)?;
        let input = if question.query.context.purpose == RoleQueryPurpose::DivergencePolicy {
            let query = DivergenceQuery {
                root: &question.query.position,
                proposal: &view.query().proposal,
                candidates: &view.query().divergences,
                records: view.records(),
                revision: question.query.context.public_revision,
                deadline: question.query.deadline,
                cancel: canceled,
            };
            pals_native::prepare_selected_divergence_input_for_config(&query, epoch, &config, view)
        } else {
            let query = role_query(question, canceled);
            pals_native::prepare_selected_role_input_for_config(
                &query,
                native_kind,
                epoch,
                &config,
                view,
            )
        }
        .map_err(failure)?;
        let representation = self
            .model
            .ticketed_representation(&input)
            .map_err(failure)?;
        if representation.role != question.role {
            return Err(RoleFailure::InvalidOutput);
        }
        Ok((
            NativePreparedTicket {
                input,
                representation,
                kind: native_kind,
            },
            representation,
        ))
    }
    fn prepared_host_bytes(&self, prepared: &NativePreparedTicket) -> u64 {
        input_bytes(&prepared.input).saturating_add(
            (std::mem::size_of::<NativePreparedTicket>() - std::mem::size_of::<PalsModelInput>())
                as u64,
        )
    }
    fn question_host_bytes(
        &self,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
    ) -> u64 {
        let owned = || -> Option<u64> {
            let prefix = question
                .query
                .context
                .prefix
                .capacity()
                .checked_mul(std::mem::size_of::<rz_position::BoardMove>())?
                as u64;
            question
                .view
                .owned_host_bytes()?
                .checked_add(std::mem::size_of::<NativeTicketQuestion>() as u64)?
                .checked_add(
                    question
                        .query
                        .position
                        .snapshot()
                        .retained_history_bytes()? as u64,
                )?
                .checked_add(prefix)
        };
        owned().unwrap_or(u64::MAX)
    }
    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &NativePreparedTicket,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<NativeTicketOutput> {
        self.evidence = None;
        let input = prepared.input.clone();
        let result = if question.query.context.purpose == RoleQueryPurpose::DivergencePolicy {
            let query = DivergenceQuery {
                root: &question.query.position,
                proposal: &question.view.query().proposal,
                candidates: &question.view.query().divergences,
                records: question.view.records(),
                revision: question.query.context.public_revision,
                deadline: question.query.deadline,
                cancel: canceled,
            };
            self.model.ticketed_run(
                ticket,
                input,
                &question.query.position,
                question.query.deadline,
                canceled,
                pals_native::NativePreparedContext::Divergence { query: &query },
                &question.query.context,
            )
        } else {
            let query = role_query(question, canceled);
            self.model.ticketed_run(
                ticket,
                input,
                &question.query.position,
                question.query.deadline,
                canceled,
                pals_native::NativePreparedContext::Role {
                    query: &query,
                    kind: prepared.kind,
                },
                &question.query.context,
            )
        };
        self.evidence = match &result {
            Ok(output) => Some(TicketExecutionEvidence {
                execution: output.execution,
                completed_nn_inputs: output.completed_nn_inputs,
                measurement_complete: output.measurement_complete,
                physical_fresh: output.physical_fresh,
                accounting: None,
            }),
            Err(_) => self.model.ticketed_take_execution_evidence(ticket),
        }
        .map(|evidence| (ticket.id, evidence));
        let physical = if matches!(result, Err(RoleError::PhysicalCompletionUnknown)) {
            PhysicalRoleCompletion::Unknown
        } else {
            PhysicalRoleCompletion::Confirmed
        };
        TicketPhysicalResult {
            physical,
            output: result.map_err(failure),
        }
    }
    fn output_host_bytes(&self, output: &NativeTicketOutput) -> u64 {
        let logits = match &output.output.payload {
            rz_contracts::pals::RolePayload::Proposal {
                candidate_logits, ..
            } => candidate_logits.capacity(),
            rz_contracts::pals::RolePayload::Counterexample {
                candidate_logits,
                divergence_logits,
                ..
            } => candidate_logits
                .capacity()
                .saturating_add(divergence_logits.capacity()),
            _ => 0,
        };
        (std::mem::size_of::<NativeTicketOutput>()
            + (logits + output.output.private_latent.capacity()) * 4) as u64
    }
    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        if self
            .evidence
            .as_ref()
            .is_some_and(|(id, _)| *id == ticket.id)
        {
            self.evidence.take().map(|(_, evidence)| evidence)
        } else {
            None
        }
    }
    fn acceptance_output_digest(&self, output: &NativeTicketOutput) -> Option<Digest> {
        rz_search::pals::coordinator::native_acceptance_output_digest(&output.output).ok()
    }
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        _: &NativePreparedTicket,
        _: &NativeTicketOutput,
    ) -> Result<(), RoleFailure> {
        self.model
            .ticketed_acknowledge(ticket, true)
            .map_err(failure)
    }
    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        self.model
            .ticketed_acknowledge(ticket, false)
            .map_err(failure)
    }
    fn quiesce(&mut self, until: Instant) -> bool {
        self.model.ticketed_quiesce(until)
    }
    fn finish(&mut self, until: Instant) -> bool {
        self.model.ticketed_close_pending();
        self.model
            .finish_handle()
            .finish(until)
            .is_ok_and(|receipt| {
                receipt.physical_shutdown_confirmed
                    && receipt.native_buffers_released
                    && !receipt.quarantined
            })
    }
}
fn role_query<'a>(
    question: &'a TicketPreparation<ReviewView, NativeTicketQuestion>,
    canceled: &'a AtomicBool,
) -> RoleQuery<'a> {
    let query = question.view.query();
    RoleQuery {
        position: &question.query.position,
        legal: &query.legal,
        prefix: &query.prefix,
        proposal: &query.proposal,
        counterexample: query.counterexample.as_deref(),
        records: question.view.records(),
        revision: question.query.context.public_revision,
        deadline: question.query.deadline,
        cancel: canceled,
    }
}
fn kind(purpose: RoleQueryPurpose) -> Result<NativeQueryKind, RoleFailure> {
    Ok(match purpose {
        RoleQueryPurpose::ReplyPolicy => NativeQueryKind::Reply,
        RoleQueryPurpose::RepairPolicy => NativeQueryKind::Repair,
        RoleQueryPurpose::DivergencePolicy => NativeQueryKind::Divergence,
        RoleQueryPurpose::ProposePolicy | RoleQueryPurpose::ValueFresh => NativeQueryKind::Propose,
        RoleQueryPurpose::TaskPolicy => {
            return Err(RoleFailure::Unsupported(
                "TaskPolicy requires the explicit V3 collection adapter",
            ));
        }
    })
}
fn input_bytes(input: &PalsModelInput) -> u64 {
    let mut bytes = std::mem::size_of::<PalsModelInput>()
        + input.board.capacity()
        + input.records.capacity() * std::mem::size_of::<rz_eval::pals_model::PalsRecordToken>()
        + input.required_critical_records.capacity() * 8
        + input.candidates.capacity()
            * std::mem::size_of::<rz_eval::pals_model::PalsCandidateToken>()
        + input.divergence_features.capacity() * 8 * 4;
    if let Some(lines) = &input.full_line {
        bytes +=
            lines.records.capacity() * std::mem::size_of::<rz_eval::pals_model::PalsRecordLine>();
        bytes += (lines
            .records
            .iter()
            .map(|line| line.moves.capacity())
            .sum::<usize>()
            + lines.query_prefix.capacity()
            + lines.query_proposal.capacity()
            + lines.query_counter.capacity())
            * std::mem::size_of::<rz_eval::pals_model::PalsCandidateToken>();
    }
    bytes as u64
}
fn failure(error: RoleError) -> RoleFailure {
    match error {
        RoleError::Unsupported(capability) => RoleFailure::Unsupported(capability),
        RoleError::Canceled => RoleFailure::Canceled,
        RoleError::Deadline => RoleFailure::Expired,
        RoleError::InvalidOutput => RoleFailure::InvalidOutput,
        RoleError::PhysicalCompletionUnknown => RoleFailure::PhysicalUnknown,
        other => RoleFailure::Backend(other.to_string()),
    }
}
