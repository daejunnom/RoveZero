//! Start-time choice between the actual V3 and V2 ticket owners.
//!
//! Dispatch never reconstructs a preparation or completed output. In particular,
//! the V3 output retains its opaque physical completion and accepted-seed
//! authority. A variant mismatch is a boundary failure, not a conversion to the
//! other model's representation or a successful native execution.

use crate::pals_native::ticketed::{
    NativeFreshTicketBackend, NativePreparedTicket, NativeTicketOutput, NativeTicketQuestion,
    NativeVirtualReviewTicketBackend, TicketExecutionEvidence, TicketNativeBackend,
    TicketPhysicalResult, TicketPreparation, VirtualReviewPreparedTicket,
    VirtualReviewTicketOutput,
};
use rz_contracts::Digest;
use rz_contracts::pals::{RepresentationKey, RoleOutput};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_runtime::contracts::ContractSystemClock;
use rz_search::pals::coordinator::{PhysicalRoleCompletion, ProposerRankingOutput, RoleFailure};
use rz_search::pals::review_context::ReviewView;
use std::mem::size_of;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// The selected backend stays unchanged for the process-owned native service.
/// Boxes keep this dispatch value small; they do not share an owner, session,
/// workspace or ticket with another model.
pub enum ActualNativeBackendV5 {
    VirtualReview(Box<NativeVirtualReviewTicketBackend>),
    LegacyV2(Box<NativeFreshTicketBackend>),
}

impl ActualNativeBackendV5 {
    /// Immutable observations emitted by the original successful native
    /// session constructors. This never synthesizes settings from a manifest
    /// request or counts OS helper threads from ORT's requested parallelism.
    pub(crate) fn applied_session_settings(
        &self,
    ) -> Option<std::sync::Arc<[rz_eval::pals_onnx::PalsAppliedSessionSettings]>> {
        match self {
            Self::VirtualReview(backend) => backend.applied_session_settings(),
            Self::LegacyV2(backend) => backend.applied_session_settings(),
        }
    }
}

/// Original backend-owned preparation, moved once rather than re-encoded.
pub enum ActualNativePreparedV5 {
    VirtualReview(Box<VirtualReviewPreparedTicket>),
    LegacyV2(Box<NativePreparedTicket>),
}

/// Original backend-owned output, including the opaque V3 completion proof.
pub enum ActualNativeOutputV5 {
    VirtualReview(Box<VirtualReviewTicketOutput>),
    LegacyV2(Box<NativeTicketOutput>),
}

impl From<NativeVirtualReviewTicketBackend> for ActualNativeBackendV5 {
    fn from(backend: NativeVirtualReviewTicketBackend) -> Self {
        Self::VirtualReview(Box::new(backend))
    }
}

impl From<NativeFreshTicketBackend> for ActualNativeBackendV5 {
    fn from(backend: NativeFreshTicketBackend) -> Self {
        Self::LegacyV2(Box::new(backend))
    }
}

impl ProposerRankingOutput for ActualNativeOutputV5 {
    fn proposer_ranking_output(&self) -> &RoleOutput {
        match self {
            Self::VirtualReview(output) => output.proposer_ranking_output(),
            Self::LegacyV2(output) => output.proposer_ranking_output(),
        }
    }
}

impl TicketNativeBackend<ReviewView, NativeTicketQuestion> for ActualNativeBackendV5 {
    type Prepared = ActualNativePreparedV5;
    type Output = ActualNativeOutputV5;

    fn clock(&self) -> ContractSystemClock {
        match self {
            Self::VirtualReview(backend) => backend.clock(),
            Self::LegacyV2(backend) => backend.clock(),
        }
    }

    fn prepare(
        &mut self,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        canceled: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        match self {
            Self::VirtualReview(backend) => {
                let (prepared, representation) = backend.prepare(question, canceled)?;
                Ok((
                    ActualNativePreparedV5::VirtualReview(Box::new(prepared)),
                    representation,
                ))
            }
            Self::LegacyV2(backend) => {
                let (prepared, representation) = backend.prepare(question, canceled)?;
                Ok((
                    ActualNativePreparedV5::LegacyV2(Box::new(prepared)),
                    representation,
                ))
            }
        }
    }

    fn prepared_host_bytes(&self, prepared: &Self::Prepared) -> u64 {
        let native = match (self, prepared) {
            (Self::VirtualReview(backend), ActualNativePreparedV5::VirtualReview(prepared)) => {
                backend.prepared_host_bytes(prepared)
            }
            (Self::LegacyV2(backend), ActualNativePreparedV5::LegacyV2(prepared)) => {
                backend.prepared_host_bytes(prepared)
            }
            _ => return u64::MAX,
        };
        // The native declaration already includes its inline object, now owned
        // by the Box allocation. Count the additional enum/Box inline storage
        // as well; no underlying tensor or identity capacity is subtracted.
        native.saturating_add(size_of::<ActualNativePreparedV5>() as u64)
    }

    fn question_host_bytes(
        &self,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
    ) -> u64 {
        match self {
            Self::VirtualReview(backend) => backend.question_host_bytes(question),
            Self::LegacyV2(backend) => backend.question_host_bytes(question),
        }
    }

    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &Self::Prepared,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        match (self, prepared) {
            (Self::VirtualReview(backend), ActualNativePreparedV5::VirtualReview(prepared)) => {
                let result = backend.run(ticket, question, prepared, canceled);
                TicketPhysicalResult {
                    physical: result.physical,
                    output: result
                        .output
                        .map(|output| ActualNativeOutputV5::VirtualReview(Box::new(output))),
                }
            }
            (Self::LegacyV2(backend), ActualNativePreparedV5::LegacyV2(prepared)) => {
                let result = backend.run(ticket, question, prepared, canceled);
                TicketPhysicalResult {
                    physical: result.physical,
                    output: result
                        .output
                        .map(|output| ActualNativeOutputV5::LegacyV2(Box::new(output))),
                }
            }
            _ => TicketPhysicalResult {
                // No native call was made by this rejected dispatch. Actual
                // owner quarantine, outstanding leases and finish remain with
                // the selected backend; this does not fence that owner.
                physical: PhysicalRoleCompletion::Confirmed,
                output: Err(RoleFailure::InvalidOutput),
            },
        }
    }

    fn output_host_bytes(&self, output: &Self::Output) -> u64 {
        let native = match (self, output) {
            (Self::VirtualReview(backend), ActualNativeOutputV5::VirtualReview(output)) => {
                backend.output_host_bytes(output)
            }
            (Self::LegacyV2(backend), ActualNativeOutputV5::LegacyV2(output)) => {
                backend.output_host_bytes(output)
            }
            _ => return u64::MAX,
        };
        native.saturating_add(size_of::<ActualNativeOutputV5>() as u64)
    }

    fn acceptance_output_digest(&self, output: &Self::Output) -> Option<Digest> {
        // The ticket service checks this exact original head against the
        // manager-issued permit before invoking the owner's accept callback.
        // Do not replace it with a view/model digest or use the trait default.
        match (self, output) {
            (Self::VirtualReview(backend), ActualNativeOutputV5::VirtualReview(output)) => {
                backend.acceptance_output_digest(output)
            }
            (Self::LegacyV2(backend), ActualNativeOutputV5::LegacyV2(output)) => {
                backend.acceptance_output_digest(output)
            }
            _ => None,
        }
    }

    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        match self {
            Self::VirtualReview(backend) => backend.execution_evidence(ticket),
            Self::LegacyV2(backend) => backend.execution_evidence(ticket),
        }
    }

    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &Self::Prepared,
        output: &Self::Output,
    ) -> Result<(), RoleFailure> {
        match (self, prepared, output) {
            (
                Self::VirtualReview(backend),
                ActualNativePreparedV5::VirtualReview(prepared),
                ActualNativeOutputV5::VirtualReview(output),
            ) => backend.accept(ticket, question, prepared, output),
            (
                Self::LegacyV2(backend),
                ActualNativePreparedV5::LegacyV2(prepared),
                ActualNativeOutputV5::LegacyV2(output),
            ) => backend.accept(ticket, question, prepared, output),
            _ => Err(RoleFailure::InvalidOutput),
        }
    }

    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        match self {
            Self::VirtualReview(backend) => backend.discard(ticket),
            Self::LegacyV2(backend) => backend.discard(ticket),
        }
    }

    fn quiesce(&mut self, until: Instant) -> bool {
        match self {
            Self::VirtualReview(backend) => backend.quiesce(until),
            Self::LegacyV2(backend) => backend.quiesce(until),
        }
    }

    fn finish(&mut self, until: Instant) -> bool {
        match self {
            Self::VirtualReview(backend) => backend.finish(until),
            Self::LegacyV2(backend) => backend.finish(until),
        }
    }
}
