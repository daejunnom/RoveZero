//! A root policy forward is consumed once, then its accepted ranking fans out
//! to continuation frames. No public boolean or caller-supplied replacement
//! logits can certify that the registry accepted the original worker result.
use super::{
    CoordinatorError, PhysicalRoleCompletion, RejectedRoleCompletion, RoleCompletion,
    RoleTicketRegistry, TicketPhase,
};
use crate::pals::engine::RoleQueryPurpose;
use crate::pals::review_context::ReviewView;
use rz_contracts::pals::{Move16, Role, RoleOutput, RolePayload};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{ClockDomain, Digest, LegalOrderIdentity, MonotonicTick, Wdl};
use rz_position::contracts::ContractState;
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

/// Native wrappers expose the actual completed contract output. Implementing
/// this adapter never grants acceptance authority; the registry still checks
/// original IDs, payload, immutable view and Rules-attested legal candidates.
pub trait ProposerRankingOutput {
    fn proposer_ranking_output(&self) -> &RoleOutput;
}
impl ProposerRankingOutput for RoleOutput {
    fn proposer_ranking_output(&self) -> &RoleOutput {
        self
    }
}
impl<T: ProposerRankingOutput + ?Sized> ProposerRankingOutput for Arc<T> {
    fn proposer_ranking_output(&self) -> &RoleOutput {
        self.as_ref().proposer_ranking_output()
    }
}

/// Small sealed data copied from the actual worker output before take_ready
/// transfers that output to the consumer. Later publication cannot substitute
/// an unrelated ReadyRole, a new output, or a public "passed" flag.
pub(super) struct HeldProposerRanking {
    logits: Arc<[f32]>,
    wdl: Wdl,
    pub(super) output_digest: Digest,
}
impl HeldProposerRanking {
    fn capture(output: &RoleOutput, ticket: &RoleTicketEnvelope) -> Result<Self, CoordinatorError> {
        if ticket.role != Role::Proposer
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
        {
            return Err(CoordinatorError::IdentityMismatch(
                "actual proposer ranking output",
            ));
        }
        let RolePayload::Proposal {
            candidate_logits,
            wdl,
        } = &output.payload
        else {
            return Err(CoordinatorError::InvalidTransition("proposer ranking head"));
        };
        if candidate_logits.is_empty()
            || candidate_logits.len() > 256
            || candidate_logits.iter().any(|value| !value.is_finite())
            || output.private_latent.len() != 16 * 384
            || output.private_latent.iter().any(|value| !value.is_finite())
        {
            return Err(CoordinatorError::InvalidTransition(
                "proposer ranking shape or numerical value",
            ));
        }
        let [win, draw, loss] = wdl.probabilities();
        Wdl::try_new(win, draw, loss, 1e-4)?;
        // The source proof also carries the complete original ticket/key. This
        // digest identifies the consumed head, not a model or Rules identity.
        Ok(Self {
            logits: Arc::from(candidate_logits.as_slice()),
            wdl: *wdl,
            output_digest: super::native_acceptance_output_digest(output)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProposerRankingSource {
    pub ticket: RoleTicketEnvelope,
    pub rules_legal_order: LegalOrderIdentity,
    pub ordered_candidate_digest: Digest,
    pub public_context: Digest,
    /// Exact original full model-facing view contents. The ticket carries the
    /// separate ReviewView::seal_digest() acceptance/ownership seal.
    pub view_input_digest: Digest,
    pub output_digest: Digest,
    pub consumption_digest: Digest,
}

/// Only successful eligible registry acceptance constructs this proof. Its
/// immutable original view and Rules snapshot survive every candidate fanout.
/// Cloning the proof is safe: candidate admission deduplicates source/firstmove
/// and does not charge another NN execution or consumed role output.
#[derive(Clone, Debug)]
pub struct ProposerRankingPublished {
    source: ProposerRankingSource,
    view: Arc<ReviewView>,
    rules: Arc<ContractState>,
    candidates: Arc<[Move16]>,
    ranked: Arc<[Move16]>,
    logits: Arc<[f32]>,
    wdl: Wdl,
}
impl ProposerRankingPublished {
    pub fn source(&self) -> ProposerRankingSource {
        self.source
    }
    pub fn view(&self) -> &Arc<ReviewView> {
        &self.view
    }
    pub fn rules(&self) -> &ContractState {
        &self.rules
    }
    pub fn candidates(&self) -> &[Move16] {
        &self.candidates
    }
    pub fn ranked_candidates(&self) -> &[Move16] {
        &self.ranked
    }
    pub fn candidate_logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn wdl(&self) -> Wdl {
        self.wdl
    }
}

impl<V, O: ProposerRankingOutput> RoleTicketRegistry<V, O> {
    /// Capture from the worker completion while its output remains owned by
    /// that message. This does not accept or consume the result. The existing
    /// physical-completion and fixed/adaptive delivery rules remain unchanged.
    #[expect(
        clippy::result_large_err,
        reason = "preserve the original bounded completion custody on rejection without allocation under cancellation or memory pressure"
    )]
    pub fn complete_proposer_ranking(
        &mut self,
        completion: RoleCompletion<O>,
    ) -> Result<(), RejectedRoleCompletion<O>> {
        // Never inspect or numerically validate an output before the worker's
        // physical completion is known. Even a malformed unknown output must
        // close admission and retain its original view/payload in quarantine.
        if completion.physical == PhysicalRoleCompletion::Unknown {
            return self.complete(completion);
        }
        let ranking = match &completion.result {
            Ok(output) => match HeldProposerRanking::capture(
                output.proposer_ranking_output(),
                &completion.ticket,
            ) {
                Ok(ranking) => Some(ranking),
                Err(error) => return Err(RejectedRoleCompletion { error, completion }),
            },
            Err(_) => None,
        };
        let ticket = completion.ticket;
        let physical = completion.physical;
        self.complete(completion)?;
        if physical == PhysicalRoleCompletion::Confirmed {
            self.match_ticket_mut(&ticket)
                .expect("completed original ticket")
                .proposer_ranking = ranking;
        }
        Ok(())
    }
}

impl<O> RoleTicketRegistry<ReviewView, O> {
    /// The root consumer supplies the concrete Rules-owned frozen state. The
    /// candidate order comes from the original immutable ReviewView itself,
    /// rather than an independently supplied array or a later selected view.
    pub fn accept_proposer_ranking(
        &mut self,
        ticket: &RoleTicketEnvelope,
        rules: Arc<ContractState>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<ProposerRankingPublished, CoordinatorError> {
        ticket.validate(clock, now, self.token_budget)?;
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Delivered || held.revoked || !held.successful_payload {
            return Err(CoordinatorError::InvalidTransition(
                "ranking publication requires eligible acceptance",
            ));
        }
        if !held.native_acceptance_confirmed {
            return Err(CoordinatorError::MissingEvidence);
        }
        let ranking = held
            .proposer_ranking
            .as_ref()
            .ok_or(CoordinatorError::MissingEvidence)?;
        let view = &held.view;
        validate_actual_view(view, ticket.view)?;
        let query = view.query();
        if ticket.role != Role::Proposer
            || query.purpose != RoleQueryPurpose::ProposePolicy
            || !query.prefix.is_empty()
            || !query.proposal.is_empty()
            || query.counterexample.is_some()
            || !query.divergences.is_empty()
        {
            return Err(CoordinatorError::InvalidTransition(
                "publication requires root proposer ranking question",
            ));
        }
        let anchor = view
            .anchors()
            .first()
            .ok_or(CoordinatorError::MissingEvidence)?;
        if rules.snapshot().identity() != ticket.view.root_state
            || !rules.rules().snapshot().same_state(&anchor.snapshot)
            || rules.terminal_wdl().is_some()
        {
            return Err(CoordinatorError::IdentityMismatch(
                "ranking exact Rules root/history",
            ));
        }
        let candidates = query
            .legal
            .iter()
            .copied()
            .map(|movement| rz_contracts::Move::try_from(movement).map(Move16::encode))
            .collect::<Result<Vec<_>, _>>()?;
        let legal = rules.legal_moves().moves();
        if candidates.len() != legal.len()
            || candidates.len() != ranking.logits.len()
            || candidates.iter().any(|movement| movement.decode().is_err())
            || candidates
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != candidates.len()
            || candidates
                .iter()
                .any(|movement| !legal.contains(&movement.decode().expect("validated move")))
        {
            return Err(CoordinatorError::IdentityMismatch(
                "ranking exact ordered legal head",
            ));
        }
        let mut candidate_hash = Sha256::new();
        for movement in &candidates {
            candidate_hash.update(movement.bits().to_le_bytes());
        }
        let candidate_digest = Digest(candidate_hash.finalize().into());
        if candidate_digest != ticket.representation.candidates {
            return Err(CoordinatorError::IdentityMismatch(
                "ranking native candidate-order digest",
            ));
        }
        let public_context = Digest(view.record_scope_digest());
        let mut consumption = Sha256::new();
        consumption.update(b"rz-pals-published-proposer-ranking/1\0");
        consumption.update(ticket.id.epoch.0.to_le_bytes());
        consumption.update(ticket.id.sequence.to_le_bytes());
        consumption.update(ticket.authority.game.0.to_le_bytes());
        consumption.update(ticket.authority.root.0.to_le_bytes());
        consumption.update(ticket.authority.implementation.0);
        consumption.update(ticket.view.root_state.semantic.0);
        consumption.update(ticket.view.root_state.owner.0.to_le_bytes());
        consumption.update(ticket.view.root_state.revision.0.to_le_bytes());
        consumption.update(ticket.question.0);
        consumption.update(ticket.view.view_digest.0);
        consumption.update(view.digest());
        consumption.update(ticket.representation.input.0);
        consumption.update(ticket.representation.model.0);
        consumption.update(ticket.representation.encoding.0);
        consumption.update(candidate_digest.0);
        consumption.update(public_context.0);
        consumption.update(ranking.output_digest.0);
        let source = ProposerRankingSource {
            ticket: *ticket,
            rules_legal_order: rules.legal_moves().order(),
            ordered_candidate_digest: candidate_digest,
            public_context,
            view_input_digest: Digest(view.digest()),
            output_digest: ranking.output_digest,
            consumption_digest: Digest(consumption.finalize().into()),
        };
        let mut order: Vec<_> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| {
            ranking.logits[b]
                .total_cmp(&ranking.logits[a])
                .then(a.cmp(&b))
        });
        let ranked = order
            .into_iter()
            .map(|index| candidates[index])
            .collect::<Vec<_>>();
        let published = ProposerRankingPublished {
            source,
            view: Arc::clone(&held.view),
            rules,
            candidates: candidates.into(),
            ranked: ranked.into(),
            logits: Arc::clone(&ranking.logits),
            wdl: ranking.wdl,
        };
        // No arbitrary output supplied here can be made to replace the small
        // seal retained from completion. Acceptance/retirement happens once.
        self.retire_accepted(ticket.id);
        Ok(published)
    }
}

pub(super) fn validate_actual_view(
    view: &ReviewView,
    seal: ReviewViewSeal,
) -> Result<(), CoordinatorError> {
    if seal.view_sequence != view.sequence()
        || seal.view_digest != Digest(view.seal_digest())
        || seal.context_policy != Digest(view.policy_digest())
        || usize::from(seal.selected_records) != view.records().len()
        || usize::from(seal.anchors) != view.anchors().len()
        || seal.actual_tokens != view.token_count()
        || seal.length_bucket != view.length_bucket()
    {
        return Err(CoordinatorError::IdentityMismatch(
            "actual immutable ReviewView seal",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "ranking_tests.rs"]
mod tests;
