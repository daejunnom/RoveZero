//! Adaptive search envelopes; EvalRequest/EvalOutput and PALS revision 0.1 stay intact.
//! Owners issue identities and attest actual Rules, view contents and physical fences.
//! No search, Rules, worker implementation or neural tensor is owned by this module.

use crate::pals::{RepresentationKey, Role, SearchAuthority};
use crate::{
    ClockDomain, ContractError, Deadline, Digest, ErrorCode, ExecutionId, MonotonicTick, RequestId,
    Stage, StateIdentity,
};
use std::time::Duration;

pub const COORDINATOR_REVISION: &str = "pals-adaptive/0.1";
pub const REVIEW_LENGTH_BUCKETS: [u16; 6] = [8, 16, 32, 64, 128, 256];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuJobId {
    pub authority: SearchAuthority,
    pub sequence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuWorkerId(pub u16);

/// The question, cumulative credit and hard deadline never change on resume.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuJobEnvelope {
    pub id: CpuJobId,
    pub state_identity: StateIdentity,
    pub question: Digest,
    pub root_restriction: Digest,
    pub evaluation_namespace: Digest,
    pub search_conditions: Digest,
    pub total_native_node_credit: u64,
    pub hard_deadline: Deadline,
}

impl CpuJobEnvelope {
    pub fn validate(&self, clock: ClockDomain, now: MonotonicTick) -> Result<(), ContractError> {
        if self.id.sequence == 0 || self.total_native_node_credit == 0 {
            return Err(invalid(
                "CPU job requires positive sequence and cumulative credit",
            ));
        }
        if self.hard_deadline.clock.0 != self.id.authority.epoch {
            return Err(identity("CPU job deadline and authority epochs differ"));
        }
        self.hard_deadline.accepts(clock, now)
    }
}

/// A manager-issued reservation, not a replacement Job or an external reported-node budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SliceGrant {
    pub job: CpuJobId,
    pub execution: ExecutionId,
    pub owner: CpuWorkerId,
    pub node_credit: u64,
    pub soft_lease: Duration,
    pub control_sequence: u64,
}

impl SliceGrant {
    pub fn validate(
        &self,
        job: &CpuJobEnvelope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), ContractError> {
        job.validate(clock, now)?;
        if self.job != job.id || self.execution.epoch != job.id.authority.epoch {
            return Err(identity("slice is bound to another job or process epoch"));
        }
        if self.execution.sequence == 0
            || self.control_sequence == 0
            || self.node_credit == 0
            || self.node_credit > job.total_native_node_credit
            || self.soft_lease.is_zero()
        {
            return Err(invalid("invalid slice reservation or soft lease"));
        }
        // Soft leases may overrun at an atomic search boundary; admission still
        // cannot schedule an initial lease beyond the immutable hard deadline.
        if self.soft_lease.as_nanos() > u128::from(job.hard_deadline.at.0 - now.0) {
            return Err(invalid(
                "slice lease extends beyond the original hard deadline",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReviewViewSeal {
    pub authority: SearchAuthority,
    pub view_sequence: u64,
    pub context_policy: Digest,
    pub view_digest: Digest,
    pub root_state: StateIdentity,
    /// Includes duplicated query sequences actually read; excludes played-game tokens.
    pub actual_tokens: u16,
    pub selected_records: u16,
    pub anchors: u8,
    pub length_bucket: u16,
}

impl ReviewViewSeal {
    pub fn validate(
        &self,
        authority: SearchAuthority,
        token_budget: u16,
    ) -> Result<(), ContractError> {
        if self.authority != authority {
            return Err(identity("view belongs to another root authority"));
        }
        if self.view_sequence == 0
            || ![128, 256, 512].contains(&token_budget)
            || self.actual_tokens > token_budget
            || self.selected_records > 64
            || self.anchors == 0
            || self.anchors > 4
            || !REVIEW_LENGTH_BUCKETS.contains(&self.length_bucket)
        {
            return Err(invalid("review view exceeds its declared context profile"));
        }
        Ok(())
    }
}

/// Independent submitted questions retain their view even if a later view is published.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoleTicketEnvelope {
    pub id: RequestId,
    pub authority: SearchAuthority,
    pub question: Digest,
    pub view: ReviewViewSeal,
    pub representation: RepresentationKey,
    pub role: Role,
    pub original_deadline: Deadline,
}

impl RoleTicketEnvelope {
    pub fn validate(
        &self,
        clock: ClockDomain,
        now: MonotonicTick,
        token_budget: u16,
    ) -> Result<(), ContractError> {
        self.view.validate(self.authority, token_budget)?;
        if self.id.epoch != self.authority.epoch
            || self.original_deadline.clock.0 != self.authority.epoch
            || self.representation.role != self.role
        {
            return Err(identity("role ticket identity, view or deadline mismatch"));
        }
        self.original_deadline.accepts(clock, now)
    }
}

fn invalid(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::InvalidInput, Stage::Admission, detail)
}
fn identity(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::IdentityMismatch, Stage::Admission, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GameGeneration, OwnerId, ProcessEpoch, RootGeneration, StateRevision};

    fn job() -> CpuJobEnvelope {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(3),
            game: GameGeneration(2),
            root: RootGeneration(4),
            implementation: Digest([1; 32]),
        };
        CpuJobEnvelope {
            id: CpuJobId {
                authority,
                sequence: 1,
            },
            state_identity: StateIdentity {
                owner: OwnerId(1),
                revision: StateRevision(5),
                semantic: Digest([2; 32]),
            },
            question: Digest([3; 32]),
            root_restriction: Digest([4; 32]),
            evaluation_namespace: Digest([5; 32]),
            search_conditions: Digest([6; 32]),
            total_native_node_credit: 8192,
            hard_deadline: Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(20_000_000),
            },
        }
    }

    #[test]
    fn another_slice_does_not_change_job_credit_or_hard_deadline() {
        let original = job();
        let mut slice = SliceGrant {
            job: original.id,
            execution: ExecutionId::new(original.id.authority.epoch, 1),
            owner: CpuWorkerId(0),
            node_credit: 4096,
            soft_lease: Duration::from_millis(8),
            control_sequence: 1,
        };
        let clock = original.hard_deadline.clock;
        slice.validate(&original, clock, MonotonicTick(0)).unwrap();
        slice.execution.sequence = 2;
        slice.control_sequence = 2;
        slice
            .validate(&original, clock, MonotonicTick(8_000_000))
            .unwrap();
        slice.job.sequence = 2;
        assert_eq!(
            slice
                .validate(&original, clock, MonotonicTick(0))
                .unwrap_err()
                .code,
            ErrorCode::IdentityMismatch
        );
        slice.job = original.id;
        assert!(slice
            .validate(&original, clock, MonotonicTick(19_000_000))
            .is_err());
        assert_eq!(original.total_native_node_credit, 8192);
    }

    #[test]
    fn view_and_slice_reject_another_game_or_root() {
        let original = job();
        let view = ReviewViewSeal {
            authority: original.id.authority,
            view_sequence: 1,
            context_policy: Digest([7; 32]),
            view_digest: Digest([8; 32]),
            root_state: original.state_identity,
            actual_tokens: 512,
            selected_records: 64,
            anchors: 4,
            length_bucket: 256,
        };
        view.validate(original.id.authority, 512).unwrap();
        assert!(view.validate(original.id.authority, 256).is_err());
        let mut new_root = original.id.authority;
        new_root.root = RootGeneration(5);
        assert_eq!(
            view.validate(new_root, 512).unwrap_err().code,
            ErrorCode::IdentityMismatch
        );
        let mut invalid_view = view;
        invalid_view.length_bucket = 17;
        assert!(invalid_view.validate(original.id.authority, 512).is_err());
    }
}
