use super::*;
use crate::pals::coordinator::RefinementStage;
use crate::pals::coordinator::test_support::*;
use rz_contracts::{ProcessEpoch, RootGeneration};

fn registry(policy: CoordinatorPolicy) -> RoleTicketRegistry<u64, u64> {
    RoleTicketRegistry::new(authority(), 512, policy, 1).unwrap()
}
fn admit(registry: &mut RoleTicketRegistry<u64, u64>, sequence: u64) -> RoleTicketEnvelope {
    registry
        .admit(
            submission(sequence),
            Arc::new(sequence),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap()
}
fn complete(registry: &mut RoleTicketRegistry<u64, u64>, ticket: RoleTicketEnvelope) {
    registry
        .complete(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(ticket.id.sequence),
        })
        .unwrap();
}

#[test]
fn two_pending_questions_keep_the_original_view_and_fixed_order() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    let first = admit(&mut held, 1);
    let second = admit(&mut held, 2);
    assert_eq!(first.authority, second.authority);
    assert_eq!(held.take_dispatch().unwrap().ticket, first);
    assert_eq!(held.take_dispatch().unwrap().ticket, second);
    complete(&mut held, second);
    assert!(
        held.take_ready(deadline().clock, MonotonicTick(0))
            .is_none()
    );
    complete(&mut held, first);
    let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    assert_eq!(ready.ticket.view, view(1));
    assert_eq!(*ready.view, 1);
    assert!(ready.logically_eligible);
    held.accept(&ready.ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    assert_eq!(ready.ticket, second);
    assert_eq!(*ready.view, 2);
}

#[test]
fn adaptive_completion_order_does_not_change_submission_authority() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let first = admit(&mut held, 1);
    let second = admit(&mut held, 2);
    held.take_dispatch().unwrap();
    held.take_dispatch().unwrap();
    complete(&mut held, second);
    complete(&mut held, first);
    let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    assert_eq!(ready.ticket, second);
    held.accept(&ready.ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(
        held.take_ready(deadline().clock, MonotonicTick(0))
            .unwrap()
            .ticket,
        first
    );
}

#[test]
fn changed_seal_and_duplicate_completion_are_rejected_with_payload_returned() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    let first = admit(&mut held, 1);
    held.take_dispatch().unwrap();
    let mut wrong = first;
    wrong.view = view(2);
    let rejected = held
        .complete(RoleCompletion {
            ticket: wrong,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(77),
        })
        .unwrap_err();
    assert!(matches!(
        rejected.error,
        CoordinatorError::IdentityMismatch(_)
    ));
    assert_eq!(rejected.completion.result, Ok(77));
    assert_eq!(held.phase(first.id), Some(TicketPhase::Submitted));
    complete(&mut held, first);
    let rejected = held
        .complete(RoleCompletion {
            ticket: first,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(88),
        })
        .unwrap_err();
    assert_eq!(rejected.error, CoordinatorError::Duplicate);
    assert_eq!(rejected.completion.result, Ok(88));
}

#[test]
fn new_root_revoke_discards_authority_but_waits_for_physical_result() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    let first = admit(&mut held, 1);
    held.take_dispatch().unwrap();
    let prepared = admit(&mut held, 2);
    assert_eq!(held.revoke_all(), vec![first.id]);
    assert!(held.ticket(prepared.id).is_none());
    complete(&mut held, first);
    let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    assert!(!ready.logically_eligible);
    assert!(
        held.accept(&ready.ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    held.discard_delivered(&ready.ticket).unwrap();
    assert_eq!(held.occupancy(), 0);
    assert!(matches!(
        held.admit(
            submission(3),
            Arc::new(3),
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::AdmissionClosed)
    ));
}

#[test]
fn expired_output_is_retired_without_acceptance() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    let first = admit(&mut held, 1);
    held.take_dispatch().unwrap();
    complete(&mut held, first);
    let now = MonotonicTick(deadline().at.0);
    let ready = held.take_ready(deadline().clock, now).unwrap();
    assert!(!ready.logically_eligible);
    assert!(held.accept(&ready.ticket, deadline().clock, now).is_err());
    held.discard_delivered(&ready.ticket).unwrap();
}

#[test]
fn physical_unknown_keeps_original_view_and_payload_until_fence() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let view_owner = Arc::new(1);
    let first = held
        .admit(
            submission(1),
            Arc::clone(&view_owner),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    held.take_dispatch().unwrap();
    held.complete(RoleCompletion {
        ticket: first,
        physical: PhysicalRoleCompletion::Unknown,
        result: Ok(42),
    })
    .unwrap();
    assert_eq!(held.phase(first.id), Some(TicketPhase::Quarantined));
    assert_eq!(Arc::strong_count(&view_owner), 2);
    assert!(
        held.take_ready(deadline().clock, MonotonicTick(0))
            .is_none()
    );
    assert!(held.discard_delivered(&first).is_err());
    held.retire_after_physical_fence(&first).unwrap();
    assert_eq!(Arc::strong_count(&view_owner), 1);
}

#[test]
fn finite_tickets_and_synchronous_lane_preserve_capacity() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    for sequence in 1..=16 {
        admit(&mut held, sequence);
    }
    assert_eq!(held.occupancy(), 16);
    assert!(matches!(
        held.admit(
            submission(17),
            Arc::new(17),
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::Capacity(_))
    ));
    let mut held = registry(CoordinatorPolicy::Synchronous);
    admit(&mut held, 1);
    assert!(matches!(
        held.admit(
            submission(2),
            Arc::new(2),
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::Capacity(_))
    ));
}

#[test]
fn registered_lower_ticket_capacity_owns_admission_and_retires_before_reuse() {
    for capacity in [1, 2, 4] {
        let mut held = RoleTicketRegistry::<u64, u64>::new_with_capacity(
            authority(),
            512,
            CoordinatorPolicy::Fixed,
            1,
            capacity,
        )
        .unwrap();
        assert_eq!(held.capacity(), capacity);
        let tickets: Vec<_> = (1..=capacity as u64).map(|n| admit(&mut held, n)).collect();
        let high_water = held.next_request_sequence();
        assert_eq!(held.occupancy(), capacity);
        assert!(matches!(
            held.admit(
                submission(high_water),
                Arc::new(high_water),
                deadline().clock,
                MonotonicTick(0)
            ),
            Err(CoordinatorError::Capacity(_))
        ));
        assert_eq!(held.next_request_sequence(), high_water);
        assert_eq!(held.take_dispatch().unwrap().ticket, tickets[0]);
        complete(&mut held, tickets[0]);
        let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
        held.accept(&ready.ticket, deadline().clock, MonotonicTick(0))
            .unwrap();
        assert_eq!(held.occupancy(), capacity - 1);
        let next = admit(&mut held, high_water);
        assert_eq!(next.id.sequence, high_water);
        assert_eq!(held.occupancy(), capacity);
        // Cancellation bypasses ordinary admission and preserves completed
        // ownership rules. Prepared descriptors have no physical leases.
        held.revoke_all();
        assert!(!held.admission_open());
    }
}

#[test]
fn explicit_ticket_capacity_rejects_zero_and_above_registered_maximum() {
    for capacity in [0, MAX_ROLE_TICKETS + 1] {
        assert!(
            RoleTicketRegistry::<u64, u64>::new_with_capacity(
                authority(),
                512,
                CoordinatorPolicy::Fixed,
                1,
                capacity,
            )
            .is_err()
        );
    }
    assert_eq!(
        registry(CoordinatorPolicy::Fixed).capacity(),
        MAX_ROLE_TICKETS
    );
}

#[test]
fn process_sequence_high_water_survives_root_replacement() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    let first = admit(&mut held, 1);
    let high_water = held.next_request_sequence();
    let mut next_root = authority();
    next_root.root = RootGeneration(5);
    let mut new =
        RoleTicketRegistry::<u64, u64>::new(next_root, 512, CoordinatorPolicy::Fixed, high_water)
            .unwrap();
    let mut next = submission(2);
    next.view.authority = next_root;
    let second = new
        .admit(next, Arc::new(2), deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_ne!(first.id, second.id);
    assert_eq!(second.id.sequence, 2);
    assert_eq!(second.id.epoch, ProcessEpoch(3));
}

#[test]
fn failed_result_cannot_be_acknowledged_as_consumed() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    let first = admit(&mut held, 1);
    held.take_dispatch().unwrap();
    held.complete(RoleCompletion {
        ticket: first,
        physical: PhysicalRoleCompletion::Confirmed,
        result: Err(RoleFailure::Backend("mock fault".into())),
    })
    .unwrap();
    let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    assert!(ready.result.is_err());
    assert!(
        held.accept(&ready.ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    held.discard_delivered(&ready.ticket).unwrap();
}

#[test]
fn fresh_pair_rejects_changed_view_precision_epoch_and_perspective() {
    let first = FreshPairSeal::from_ticket(
        &ticket(1, RefinementStage::FreshRepairedEndpoint),
        Digest([1; 32]),
        Color::White,
    );
    first.validate_pair(&first).unwrap();
    let mut second = first;
    second.public_context = Digest([2; 32]);
    assert!(first.validate_pair(&second).is_err());
    second = first;
    second.perspective = Color::Black;
    assert!(first.validate_pair(&second).is_err());
    second = first;
    second.frozen_epoch += 1;
    assert!(first.validate_pair(&second).is_err());
}

#[test]
fn fresh_pair_keeps_public_context_while_endpoint_query_identity_differs() {
    let first_ticket = ticket(1, RefinementStage::FreshRepairedEndpoint);
    let mut second_ticket = ticket(2, RefinementStage::FreshCounterEndpoint);
    second_ticket.view = view(2);
    second_ticket.view.actual_tokens = 256;
    second_ticket.view.length_bucket = 64;
    second_ticket.question = Digest([45; 32]);
    second_ticket.representation.input = Digest([46; 32]);
    second_ticket.representation.candidates = Digest([47; 32]);
    let first = FreshPairSeal::from_ticket(&first_ticket, Digest([55; 32]), Color::White);
    let second = FreshPairSeal::from_ticket(&second_ticket, Digest([55; 32]), Color::White);
    first.validate_pair(&second).unwrap();
    assert_ne!(first_ticket.view, second_ticket.view);
    assert_ne!(first_ticket.representation, second_ticket.representation);
    let mut changed = second;
    changed.record_revision += 1;
    assert!(first.validate_pair(&changed).is_err());
    changed = second;
    changed.precision = rz_contracts::PrecisionProfile::Fp16;
    assert!(first.validate_pair(&changed).is_err());
    changed = second;
    changed.root_state.revision.0 += 1;
    assert!(first.validate_pair(&changed).is_err());
}

#[test]
fn owner_join_retirement_requires_revoke_and_never_normalizes_quarantine() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let first = admit(&mut held, 1);
    let second = admit(&mut held, 2);
    held.take_dispatch().unwrap();
    held.take_dispatch().unwrap();
    complete(&mut held, first);
    assert!(held.retire_revoked_after_worker_join().is_err());
    assert_eq!(held.occupancy(), 2);
    assert_eq!(held.revoke_all(), [second.id]);
    // The fixture supplies the owner cleanup call; the API does not itself
    // claim a real GPU fence or thread join. Actual consumers record those.
    assert_eq!(
        held.retire_revoked_after_worker_join().unwrap(),
        [first, second]
    );
    assert_eq!(held.occupancy(), 0);
    assert_eq!(held.physical_outstanding(), 0);
    assert!(
        held.take_ready(deadline().clock, MonotonicTick(0))
            .is_none()
    );

    let mut held = registry(CoordinatorPolicy::Adaptive);
    let ticket = admit(&mut held, 1);
    held.take_dispatch().unwrap();
    held.complete(RoleCompletion {
        ticket,
        physical: PhysicalRoleCompletion::Unknown,
        result: Err(RoleFailure::PhysicalUnknown),
    })
    .unwrap();
    held.revoke_all();
    assert!(held.retire_revoked_after_worker_join().is_err());
    assert_eq!(held.phase(ticket.id), Some(TicketPhase::Quarantined));
    assert_eq!(held.occupancy(), 1);
    assert_eq!(held.physical_outstanding(), 1);
}
