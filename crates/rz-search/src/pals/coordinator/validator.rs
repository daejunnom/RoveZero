//! Collection-only task-head publication from an actually accepted ticket.
//! This proof authorizes an offered action; it does not certify a new witness
//! or utility target. Those require the subsequent actual Store commit.
use super::{
    CoordinatorError, PhysicalRoleCompletion, ProposerRankingOutput, RejectedRoleCompletion,
    RoleCompletion, RoleTicketRegistry, TicketPhase, ranking::validate_actual_view,
};
use crate::pals::engine::RoleQueryPurpose;
use crate::pals::review_context::ReviewView;
use rz_contracts::pals::{Role, RoleOutput, RolePayload};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{ClockDomain, Digest, MonotonicTick};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

pub(super) struct HeldValidatorRanking {
    logits: [f32; 7],
    pub(super) output_digest: Digest,
}
impl HeldValidatorRanking {
    fn capture(output: &RoleOutput, ticket: &RoleTicketEnvelope) -> Result<Self, CoordinatorError> {
        if ticket.role != Role::Validator
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
        {
            return Err(CoordinatorError::IdentityMismatch(
                "actual validator task output",
            ));
        }
        let RolePayload::TaskRanking { task_logits } = &output.payload else {
            return Err(CoordinatorError::InvalidTransition("validator task head"));
        };
        let logits: [f32; 7] = task_logits
            .as_slice()
            .try_into()
            .map_err(|_| CoordinatorError::InvalidTransition("validator task head width"))?;
        if logits.iter().any(|value| !value.is_finite())
            || output.private_latent.len() != 16 * 384
            || output.private_latent.iter().any(|value| !value.is_finite())
        {
            return Err(CoordinatorError::InvalidTransition(
                "validator task numerical value",
            ));
        }
        Ok(Self {
            logits,
            output_digest: super::native_acceptance_output_digest(output)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptedValidatorSource {
    pub ticket: RoleTicketEnvelope,
    pub public_context: Digest,
    pub view_input_digest: Digest,
    pub output_digest: Digest,
    pub consumption_digest: Digest,
}

/// The private constructor captures the original completed head before moving
/// the raw output to its consumer, and publishes only after eligible acceptance.
/// A clone does not mint an additional consumption or NN execution. The actual
/// action producer and manager must deduplicate its original ticket/action.
#[derive(Clone, Debug)]
pub struct AcceptedValidatorPublished {
    source: AcceptedValidatorSource,
    view: Arc<ReviewView>,
    logits: [f32; 7],
}
impl AcceptedValidatorPublished {
    pub fn source(&self) -> AcceptedValidatorSource {
        self.source
    }
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.source.ticket
    }
    pub fn view(&self) -> &Arc<ReviewView> {
        &self.view
    }
    pub fn task_logits(&self) -> &[f32; 7] {
        &self.logits
    }
    pub fn output_digest(&self) -> Digest {
        self.source.output_digest
    }
}

impl<V, O: ProposerRankingOutput> RoleTicketRegistry<V, O> {
    /// No native acceptance is implied yet. Unknown physical completion takes
    /// precedence over malformed numerical output, retaining all owned data.
    #[expect(
        clippy::result_large_err,
        reason = "preserve the original bounded completion custody on rejection without allocation under cancellation or memory pressure"
    )]
    pub fn complete_validator_task(
        &mut self,
        completion: RoleCompletion<O>,
    ) -> Result<(), RejectedRoleCompletion<O>> {
        if completion.physical == PhysicalRoleCompletion::Unknown {
            return self.complete(completion);
        }
        let captured = match &completion.result {
            Ok(output) => match HeldValidatorRanking::capture(
                output.proposer_ranking_output(),
                &completion.ticket,
            ) {
                Ok(captured) => Some(captured),
                Err(error) => return Err(RejectedRoleCompletion { error, completion }),
            },
            Err(_) => None,
        };
        let ticket = completion.ticket;
        self.complete(completion)?;
        self.match_ticket_mut(&ticket)
            .expect("original completed ticket")
            .validator_ranking = captured;
        Ok(())
    }
}

impl<O> RoleTicketRegistry<ReviewView, O> {
    /// Call only after the native owner accepted this same delivered output.
    /// Returning a raw ReadyRole or supplying replacement logits grants no proof.
    pub fn accept_validator_task(
        &mut self,
        ticket: &RoleTicketEnvelope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<AcceptedValidatorPublished, CoordinatorError> {
        ticket.validate(clock, now, self.token_budget)?;
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Delivered || held.revoked || !held.successful_payload {
            return Err(CoordinatorError::InvalidTransition(
                "validator publication requires eligible acceptance",
            ));
        }
        if !held.native_acceptance_confirmed {
            return Err(CoordinatorError::MissingEvidence);
        }
        if ticket.role != Role::Validator
            || held.view.query().purpose != RoleQueryPurpose::TaskPolicy
        {
            return Err(CoordinatorError::InvalidTransition(
                "collection task policy question",
            ));
        }
        validate_actual_view(&held.view, ticket.view)?;
        let captured = held
            .validator_ranking
            .as_ref()
            .ok_or(CoordinatorError::MissingEvidence)?;
        let public_context = Digest(held.view.record_scope_digest());
        let view_input_digest = Digest(held.view.digest());
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-published-validator-task/1\0");
        hash.update(ticket.id.epoch.0.to_le_bytes());
        hash.update(ticket.id.sequence.to_le_bytes());
        hash.update(ticket.authority.game.0.to_le_bytes());
        hash.update(ticket.authority.root.0.to_le_bytes());
        hash.update(ticket.authority.implementation.0);
        hash.update(ticket.view.root_state.semantic.0);
        hash.update(ticket.view.root_state.owner.0.to_le_bytes());
        hash.update(ticket.view.root_state.revision.0.to_le_bytes());
        hash.update(ticket.question.0);
        hash.update(ticket.view.view_digest.0);
        hash.update(ticket.representation.input.0);
        hash.update(ticket.representation.model.0);
        hash.update(ticket.representation.encoding.0);
        hash.update(public_context.0);
        hash.update(view_input_digest.0);
        hash.update(captured.output_digest.0);
        let published = AcceptedValidatorPublished {
            source: AcceptedValidatorSource {
                ticket: *ticket,
                public_context,
                view_input_digest,
                output_digest: captured.output_digest,
                consumption_digest: Digest(hash.finalize().into()),
            },
            view: Arc::clone(&held.view),
            logits: captured.logits,
        };
        self.retire_accepted(ticket.id);
        Ok(published)
    }
}

#[cfg(test)]
#[path = "validator_tests.rs"]
mod tests;
