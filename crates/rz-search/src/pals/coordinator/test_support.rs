use super::*;
use rz_contracts::pals::{RepresentationKey, Role, SearchAuthority};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::*;
use std::sync::Arc;

/// CPU/mock owner receipt for registry tests. This exercises the actual linear
/// capability API; it is not a native backend or physical GPU acceptance test.
pub(super) fn confirm_mock_native_acceptance<V, O>(
    registry: &mut RoleTicketRegistry<V, O>,
    ticket: &RoleTicketEnvelope,
) {
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let permit = registry
        .issue_native_acceptance(ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    registry
        .confirm_native_acceptance(
            issuer.issue_success(permit),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
}

pub(super) fn authority() -> SearchAuthority {
    SearchAuthority {
        epoch: ProcessEpoch(3),
        game: GameGeneration(2),
        root: RootGeneration(4),
        implementation: Digest([1; 32]),
    }
}
pub(super) fn state() -> StateIdentity {
    StateIdentity {
        owner: OwnerId(1),
        revision: StateRevision(5),
        semantic: Digest([2; 32]),
    }
}
pub(super) fn deadline() -> Deadline {
    Deadline {
        clock: ClockDomain(authority().epoch),
        at: MonotonicTick(2_000_000_000),
    }
}
pub(super) fn view(sequence: u64) -> ReviewViewSeal {
    ReviewViewSeal {
        authority: authority(),
        view_sequence: sequence,
        context_policy: Digest([7; 32]),
        view_digest: Digest([sequence as u8; 32]),
        root_state: state(),
        actual_tokens: 128,
        selected_records: 4,
        anchors: 1,
        length_bucket: 32,
    }
}
pub(super) fn key(role: Role) -> RepresentationKey {
    RepresentationKey {
        input: Digest([3; 32]),
        history: Digest([4; 32]),
        candidates: Digest([5; 32]),
        mask_positions: Digest([6; 32]),
        model: Digest([7; 32]),
        encoding: Digest([8; 32]),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: 1,
        role,
    }
}
pub(super) fn submission(sequence: u64) -> RoleSubmission {
    RoleSubmission {
        question: Digest([sequence as u8; 32]),
        view: view(sequence),
        representation: key(Role::Proposer),
        role: Role::Proposer,
        original_deadline: deadline(),
    }
}
pub(super) fn ticket(sequence: u64, stage: RefinementStage) -> RoleTicketEnvelope {
    let role = stage.role().unwrap();
    RoleTicketEnvelope {
        id: RequestId::new(authority().epoch, sequence),
        authority: authority(),
        question: Digest([stage as u8; 32]),
        view: view(1),
        representation: key(role),
        role,
        original_deadline: deadline(),
    }
}
pub(super) fn input(stage: RefinementStage) -> FrameInput<u64, RefinementStage> {
    FrameInput {
        stage,
        question_digest: Digest([stage as u8; 32]),
        question: Arc::new(stage),
        view_seal: view(1),
        public_context: Digest([1; 32]),
        view: Arc::new(1),
    }
}
pub(super) fn first_move() -> rz_contracts::pals::Move16 {
    rz_contracts::pals::Move16::encode(
        Move::new(
            Square::try_new(12).unwrap(),
            Square::try_new(28).unwrap(),
            None,
        )
        .unwrap(),
    )
}
pub(super) fn budget() -> FrameBudget {
    FrameBudget {
        role_submissions_remaining: 128,
        original_cpu_credit: 65536,
        original_store_byte_limit: 1 << 20,
    }
}
