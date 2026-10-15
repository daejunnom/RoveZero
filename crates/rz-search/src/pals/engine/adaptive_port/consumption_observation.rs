//! Opt-in copied scalar facts at the original Search semantic boundaries.
//! No output/model/view payload, callback or coordinator-transition authority.
use super::{PalsError, RoleError, StoreError};
use crate::pals::coordinator::FrameId;
use rz_contracts::pals::{Role, SearchAuthority};
use rz_contracts::{Digest, RequestId};

pub const MAX_ROLE_CONSUMPTION_EVENTS: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoleConsumptionStage {
    SemanticCommit,
    ViewConsumption,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoleConsumptionOutcome {
    /// The original commit_role returned this exact step evidence. This does
    /// not assert view consumption or accepted coordinator continuation.
    SemanticResult,
    /// Original review_index.consume and first consumed_role_views insert
    /// both succeeded. A later materialization/transition can still fail.
    ConsumedView,
    DuplicateNoNewConsumption,
    Rejected(AdaptiveRoleConsumptionRejection),
}

/// Finite original error classes only. The complete original error is returned
/// unchanged; foreign ticket metadata is never promoted to a successful event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoleConsumptionRejection {
    Deadline,
    Canceled,
    ExpiredConsumer,
    StaleConsumer,
    DuplicateAttempt,
    InvalidOutput,
    PhysicalCompletionUnknown,
    RoleCallLimit,
    Capacity,
    OtherOriginalFailure,
}
impl AdaptiveRoleConsumptionRejection {
    pub(super) fn from_error(error: &PalsError) -> Self {
        match error {
            PalsError::Role(RoleError::Deadline) => Self::Deadline,
            PalsError::Role(RoleError::Canceled)
            | PalsError::Store(StoreError::CancelledConsumer) => Self::Canceled,
            PalsError::Store(StoreError::ExpiredConsumer) => Self::ExpiredConsumer,
            PalsError::Store(StoreError::StaleConsumer) => Self::StaleConsumer,
            PalsError::Store(StoreError::AlreadyConsumed)
            | PalsError::Store(StoreError::InvalidEvidence(
                "role semantic commit was already attempted",
            )) => Self::DuplicateAttempt,
            PalsError::Role(RoleError::InvalidOutput) => Self::InvalidOutput,
            PalsError::Role(RoleError::PhysicalCompletionUnknown) => {
                Self::PhysicalCompletionUnknown
            }
            PalsError::RoleCallLimit => Self::RoleCallLimit,
            PalsError::Capacity | PalsError::Store(StoreError::Capacity(_)) => Self::Capacity,
            _ => Self::OtherOriginalFailure,
        }
    }
}

/// RequestId is the original role ticket's id. Every field is a copied scalar
/// identity/digest; neither the ReviewView nor RoleOutput is retained here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveRoleConsumptionEvent {
    pub authority: SearchAuthority,
    pub request: RequestId,
    pub role: Role,
    pub question: Digest,
    /// commit_role: exact submitted work question. ViewConsumption: original
    /// committed_role_questions entry, if present. Never substituted from ticket.
    pub submitted_question: Option<Digest>,
    pub ticket_view: Digest,
    pub submitted_view: Digest,
    /// Only commit_role receives the original coordinator work frame.
    pub frame: Option<FrameId>,
    pub stage: AdaptiveRoleConsumptionStage,
    pub outcome: AdaptiveRoleConsumptionOutcome,
    /// Copied from the successful original AdaptivePortStep, never a native
    /// receipt or an observer-generated digest. None on consume-only events.
    pub completed_evidence: Option<Digest>,
    /// Exact attempted_role_commits membership before this call. A true value
    /// does not assert that the earlier attempt succeeded or consumed a view.
    pub prior_semantic_attempt: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoleConsumptionUnavailable {
    ReservationFailed,
    EventCapacity,
}

/// Root-local fixed record cap. The default port contains None and allocates
/// no journal. Overflow preserves its prefix as incomplete/unavailable and
/// never changes the original search result or error.
#[derive(Debug)]
pub struct AdaptiveRoleConsumptionObservation {
    authority: SearchAuthority,
    events: Vec<AdaptiveRoleConsumptionEvent>,
    unavailable: Option<AdaptiveRoleConsumptionUnavailable>,
}
impl AdaptiveRoleConsumptionObservation {
    pub(super) fn new(authority: SearchAuthority) -> Self {
        let mut events = Vec::new();
        let unavailable = events
            .try_reserve_exact(MAX_ROLE_CONSUMPTION_EVENTS)
            .err()
            .map(|_| AdaptiveRoleConsumptionUnavailable::ReservationFailed);
        Self {
            authority,
            events,
            unavailable,
        }
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn events(&self) -> &[AdaptiveRoleConsumptionEvent] {
        &self.events
    }
    pub fn unavailable(&self) -> Option<AdaptiveRoleConsumptionUnavailable> {
        self.unavailable
    }
    pub(super) fn record(&mut self, event: AdaptiveRoleConsumptionEvent) {
        if self.unavailable.is_some() {
            return;
        }
        if self.events.len() >= MAX_ROLE_CONSUMPTION_EVENTS {
            self.unavailable = Some(AdaptiveRoleConsumptionUnavailable::EventCapacity);
            return;
        }
        // Reservation was made once at opt-in; this push never grows beyond it.
        self.events.push(event);
    }
}
