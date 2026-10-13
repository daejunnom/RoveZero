//! Linear acceptance permits and private native-owner success receipts.
//! The actual service creates its issuer on the owner thread, publishes only
//! the verifier, and issues receipts only after backend acceptance succeeds.
use super::{CoordinatorError, RoleTicketRegistry, TicketPhase};
use rz_contracts::pals::{Role, RoleOutput, RolePayload};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{ClockDomain, Digest, MonotonicTick, Wdl};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeAcceptanceSeal {
    ticket: RoleTicketEnvelope,
    output_digest: Digest,
}

/// Registry-issued, non-Clone authority to ask the exact native owner to
/// accept one original P/V output. A permit by itself is never success proof.
#[derive(Debug)]
pub struct NativeAcceptancePermit {
    seal: NativeAcceptanceSeal,
}
impl NativeAcceptancePermit {
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.seal.ticket
    }
    pub fn output_digest(&self) -> Digest {
        self.seal.output_digest
    }
}

/// Hold exclusively inside the actual native owner. Other owner pairs have
/// distinct private identities and cannot confirm this registry's requests.
pub struct NativeAcceptanceIssuer {
    identity: Arc<()>,
}
#[derive(Clone, Debug)]
pub struct NativeAcceptanceVerifier {
    identity: Arc<()>,
}
impl NativeAcceptanceIssuer {
    pub fn new_pair() -> (Self, NativeAcceptanceVerifier) {
        let identity = Arc::new(());
        (
            Self {
                identity: Arc::clone(&identity),
            },
            NativeAcceptanceVerifier { identity },
        )
    }
    /// The native service calls this only after exact held ticket/output digest
    /// validation and successful backend.accept; failure never issues a receipt.
    pub fn issue_success(&self, permit: NativeAcceptancePermit) -> NativeAcceptanceReceipt {
        NativeAcceptanceReceipt {
            permit,
            identity: Arc::clone(&self.identity),
        }
    }
}

#[derive(Debug)]
pub struct NativeAcceptanceReceipt {
    permit: NativeAcceptancePermit,
    identity: Arc<()>,
}
impl NativeAcceptanceReceipt {
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        self.permit.ticket()
    }
    pub fn output_digest(&self) -> Digest {
        self.permit.output_digest()
    }
}
#[derive(Debug)]
pub struct RejectedNativeAcceptance {
    pub error: CoordinatorError,
    pub receipt: NativeAcceptanceReceipt,
}

impl<V, O> RoleTicketRegistry<V, O> {
    /// Register only the verifier emitted by the actual spawned service owner.
    /// It cannot be replaced later to make another owner's receipt succeed.
    pub fn register_native_acceptance_verifier(
        &mut self,
        verifier: NativeAcceptanceVerifier,
    ) -> Result<(), CoordinatorError> {
        if !self.admission_open() {
            return Err(CoordinatorError::AdmissionClosed);
        }
        if self.native_acceptance_verifier.is_some() {
            return Err(CoordinatorError::Duplicate);
        }
        self.native_acceptance_verifier = Some(verifier);
        Ok(())
    }
    pub fn issue_native_acceptance(
        &mut self,
        ticket: &RoleTicketEnvelope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<NativeAcceptancePermit, CoordinatorError> {
        ticket.validate(clock, now, self.token_budget)?;
        if self.native_acceptance_verifier.is_none() {
            return Err(CoordinatorError::MissingEvidence);
        }
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Delivered || held.revoked || !held.successful_payload {
            return Err(CoordinatorError::InvalidTransition(
                "native acceptance requires original eligible delivery",
            ));
        }
        if held.native_acceptance_issued.is_some() || held.native_acceptance_confirmed {
            return Err(CoordinatorError::Duplicate);
        }
        let output_digest = match ticket.role {
            Role::Proposer => held
                .proposer_ranking
                .as_ref()
                .map(|head| head.output_digest),
            Role::Validator => held
                .validator_ranking
                .as_ref()
                .map(|head| head.output_digest),
            Role::Critic => None,
        }
        .ok_or(CoordinatorError::MissingEvidence)?;
        let seal = NativeAcceptanceSeal {
            ticket: *ticket,
            output_digest,
        };
        held.native_acceptance_issued = Some(seal);
        Ok(NativeAcceptancePermit { seal })
    }
    pub fn confirm_native_acceptance(
        &mut self,
        receipt: NativeAcceptanceReceipt,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), RejectedNativeAcceptance> {
        let check = (|| {
            let verifier = self
                .native_acceptance_verifier
                .as_ref()
                .ok_or(CoordinatorError::MissingEvidence)?;
            if !Arc::ptr_eq(&verifier.identity, &receipt.identity) {
                return Err(CoordinatorError::IdentityMismatch(
                    "original native acceptance owner",
                ));
            }
            let ticket = receipt.ticket();
            ticket.validate(clock, now, self.token_budget)?;
            let held = self.match_ticket_mut(ticket)?;
            if held.phase != TicketPhase::Delivered
                || held.revoked
                || !held.successful_payload
                || held.native_acceptance_issued != Some(receipt.permit.seal)
                || held.native_acceptance_confirmed
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "original native acceptance permit/head",
                ));
            }
            held.native_acceptance_confirmed = true;
            Ok(())
        })();
        check.map_err(|error| RejectedNativeAcceptance { error, receipt })
    }
}

/// Share the exact completed P/V head identity with the native service. This
/// function is an identity/numerical validator and grants no acceptance rights.
pub fn native_acceptance_output_digest(output: &RoleOutput) -> Result<Digest, CoordinatorError> {
    if output.private_latent.len() != 16 * 384
        || output.private_latent.iter().any(|v| !v.is_finite())
    {
        return Err(CoordinatorError::InvalidTransition(
            "native acceptance latent shape/numerical value",
        ));
    }
    let mut hash = Sha256::new();
    match &output.payload {
        RolePayload::Proposal {
            candidate_logits,
            wdl,
        } if output.key.role == Role::Proposer => {
            if candidate_logits.is_empty()
                || candidate_logits.len() > 256
                || candidate_logits.iter().any(|v| !v.is_finite())
            {
                return Err(CoordinatorError::InvalidTransition(
                    "native acceptance proposer head",
                ));
            }
            let [win, draw, loss] = wdl.probabilities();
            Wdl::try_new(win, draw, loss, 1e-4)?;
            hash.update(b"rz-pals-accepted-proposer-head/1\0");
            hash.update((candidate_logits.len() as u64).to_le_bytes());
            for value in candidate_logits.iter().copied().chain(wdl.probabilities()) {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        RolePayload::TaskRanking { task_logits } if output.key.role == Role::Validator => {
            if task_logits.len() != 7 || task_logits.iter().any(|v| !v.is_finite()) {
                return Err(CoordinatorError::InvalidTransition(
                    "native acceptance validator head",
                ));
            }
            hash.update(b"rz-pals-accepted-validator-head/1\0");
            for value in task_logits {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        _ => {
            return Err(CoordinatorError::InvalidTransition(
                "native acceptance captured P/V head required",
            ));
        }
    }
    Ok(Digest(hash.finalize().into()))
}
