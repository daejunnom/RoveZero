use super::*;
use crate::pals::coordinator::test_support::*;
use crate::pals::coordinator::{
    CoordinatorError, PhysicalRoleCompletion, RoleCompletion, RoleFailure, RoleTicketRegistry,
    TicketPhase,
};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{Digest, MonotonicTick};
use rz_runtime::adaptive::CoordinatorPolicy;
use std::sync::Arc;

fn registry(policy: CoordinatorPolicy) -> RoleTicketRegistry<u64, u64> {
    RoleTicketRegistry::new(authority(), 512, policy, 1).unwrap()
}

fn admit(held: &mut RoleTicketRegistry<u64, u64>, sequence: u64) -> RoleTicketEnvelope {
    held.admit(
        submission(sequence),
        Arc::new(sequence),
        deadline().clock,
        MonotonicTick(0),
    )
    .unwrap()
}

fn complete(held: &mut RoleTicketRegistry<u64, u64>, ticket: RoleTicketEnvelope) {
    held.complete(RoleCompletion {
        ticket,
        physical: PhysicalRoleCompletion::Confirmed,
        result: Ok(ticket.id.sequence),
    })
    .unwrap();
}

fn totals(observation: RoleTicketCumulativeObservation) -> [u64; 10] {
    [
        observation.admitted,
        observation.service_dispatched,
        observation.physical_start_notifications,
        observation.physically_completed,
        observation.delivered,
        observation.logically_accepted,
        observation.terminal_before_service_dispatch,
        observation.released,
        observation.revoked,
        observation.physical_unknown_events,
    ]
}

fn phases(occupancy: RoleTicketPhaseOccupancy) -> [u64; 6] {
    [
        occupancy.prepared,
        occupancy.submitted,
        occupancy.physically_running,
        occupancy.completed,
        occupancy.delivered,
        occupancy.quarantined,
    ]
}

fn assert_current(held: &RoleTicketRegistry<u64, u64>, expected: [u64; 6]) {
    let observation = held.cumulative_observation();
    assert_eq!(phases(observation.current), expected);
    assert_eq!(observation.current.held(), held.occupancy() as u64);
}

#[test]
fn full_cycle_records_each_owned_boundary_and_preserves_peaks_after_release() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    assert_eq!(
        held.cumulative_observation(),
        RoleTicketCumulativeObservation::default()
    );
    let ticket = admit(&mut held, 1);
    assert_current(&held, [1, 0, 0, 0, 0, 0]);
    assert_eq!(
        totals(held.cumulative_observation()),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(held.take_dispatch().unwrap().ticket, ticket);
    assert_current(&held, [0, 1, 0, 0, 0, 0]);
    assert_eq!(
        held.cumulative_observation().physical_start_notifications,
        0
    );
    held.physical_started(&ticket).unwrap();
    assert_current(&held, [0, 0, 1, 0, 0, 0]);
    complete(&mut held, ticket);
    assert_current(&held, [0, 0, 0, 1, 0, 0]);
    let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    assert_eq!(ready.ticket, ticket);
    assert!(ready.logically_eligible);
    assert_eq!(ready.result, Ok(1));
    assert_current(&held, [0, 0, 0, 0, 1, 0]);
    held.accept(&ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_current(&held, [0; 6]);
    let observation = held.cumulative_observation();
    assert_eq!(totals(observation), [1, 1, 1, 1, 1, 1, 0, 1, 0, 0]);
    assert_eq!(phases(observation.phase_peaks), [1, 1, 1, 1, 1, 0]);
    assert_eq!(observation.held_peak, 1);
}

#[test]
fn duplicate_start_completion_delivery_and_retirement_do_not_inflate_observations() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let ticket = admit(&mut held, 1);
    held.take_dispatch().unwrap();
    held.physical_started(&ticket).unwrap();
    let before = held.cumulative_observation();
    assert!(held.physical_started(&ticket).is_err());
    assert!(held.take_dispatch().is_none());
    assert_eq!(held.cumulative_observation(), before);
    complete(&mut held, ticket);
    let before = held.cumulative_observation();
    let rejected = held
        .complete(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(99),
        })
        .unwrap_err();
    assert_eq!(rejected.error, CoordinatorError::Duplicate);
    assert_eq!(rejected.completion.result, Ok(99));
    assert_eq!(held.cumulative_observation(), before);
    held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    let before = held.cumulative_observation();
    assert!(
        held.take_ready_id(ticket.id, deadline().clock, MonotonicTick(0))
            .unwrap()
            .is_none()
    );
    assert_eq!(held.cumulative_observation(), before);
    held.accept(&ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    let before = held.cumulative_observation();
    assert!(
        held.accept(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    assert!(held.discard_delivered(&ticket).is_err());
    assert_eq!(held.cumulative_observation(), before);
    assert_eq!(before.logically_accepted, 1);
    assert_eq!(before.released, 1);
}

#[test]
fn fixed_and_adaptive_delivery_order_share_cumulative_lifecycle_facts() {
    for policy in [CoordinatorPolicy::Fixed, CoordinatorPolicy::Adaptive] {
        let mut held = registry(policy);
        let first = admit(&mut held, 1);
        let second = admit(&mut held, 2);
        held.take_dispatch().unwrap();
        held.take_dispatch().unwrap();
        complete(&mut held, second);
        if policy == CoordinatorPolicy::Fixed {
            let before = held.cumulative_observation();
            assert!(
                held.take_ready(deadline().clock, MonotonicTick(0))
                    .is_none()
            );
            assert!(
                held.take_ready_id(second.id, deadline().clock, MonotonicTick(0))
                    .is_err()
            );
            assert_eq!(held.cumulative_observation(), before);
        }
        complete(&mut held, first);
        let expected_first = if policy == CoordinatorPolicy::Fixed {
            first
        } else {
            second
        };
        let expected_second = if policy == CoordinatorPolicy::Fixed {
            second
        } else {
            first
        };
        for expected in [expected_first, expected_second] {
            let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
            assert_eq!(ready.ticket, expected);
            held.accept(&ready.ticket, deadline().clock, MonotonicTick(0))
                .unwrap();
        }
        let observation = held.cumulative_observation();
        assert_eq!(totals(observation), [2, 2, 0, 2, 2, 2, 0, 2, 0, 0]);
        assert_eq!(phases(observation.phase_peaks), [2, 2, 0, 2, 1, 0]);
        assert_eq!(observation.held_peak, 2);
        assert_current(&held, [0; 6]);
    }
}

#[test]
fn mixed_phase_revoke_counts_original_tickets_once_and_prepared_release_separately() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    let tickets: Vec<_> = (1..=5).map(|sequence| admit(&mut held, sequence)).collect();
    for expected in &tickets[..4] {
        assert_eq!(held.take_dispatch().unwrap().ticket, *expected);
    }
    held.physical_started(&tickets[2]).unwrap();
    complete(&mut held, tickets[0]);
    complete(&mut held, tickets[1]);
    held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    assert_eq!(held.revoke_all(), [tickets[2].id, tickets[3].id]);
    assert_current(&held, [0, 1, 1, 1, 1, 0]);
    let before = held.cumulative_observation();
    assert_eq!(before.revoked, 5);
    assert_eq!(before.terminal_before_service_dispatch, 1);
    assert_eq!(before.released, 1);
    assert!(held.revoke_all().is_empty());
    assert_eq!(held.cumulative_observation(), before);
    assert!(
        held.accept(&tickets[0], deadline().clock, MonotonicTick(0))
            .is_err()
    );
    held.discard_delivered(&tickets[0]).unwrap();
    for ticket in &tickets[1..4] {
        if held.phase(ticket.id) != Some(TicketPhase::Completed) {
            held.complete(RoleCompletion {
                ticket: *ticket,
                physical: PhysicalRoleCompletion::Confirmed,
                result: Err(RoleFailure::Canceled),
            })
            .unwrap();
        }
        let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
        assert_eq!(ready.ticket, *ticket);
        assert!(!ready.logically_eligible);
        held.discard_delivered(ticket).unwrap();
    }
    let observation = held.cumulative_observation();
    assert_eq!(totals(observation), [5, 4, 1, 4, 4, 0, 1, 5, 5, 0]);
    assert_eq!(phases(observation.phase_peaks), [5, 4, 1, 2, 1, 0]);
    assert_eq!(observation.held_peak, 5);
    assert_current(&held, [0; 6]);
}

#[test]
fn exact_identity_unknown_invalid_output_quarantines_and_revokes_all_held_once() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let first = admit(&mut held, 1);
    let second = admit(&mut held, 2);
    let prepared = admit(&mut held, 3);
    held.take_dispatch().unwrap();
    held.take_dispatch().unwrap();
    held.physical_started(&first).unwrap();
    held.complete(RoleCompletion {
        ticket: first,
        physical: PhysicalRoleCompletion::Unknown,
        result: Err(RoleFailure::InvalidOutput),
    })
    .unwrap();
    assert!(!held.admission_open());
    assert_current(&held, [1, 1, 0, 0, 0, 1]);
    let before = held.cumulative_observation();
    assert_eq!(totals(before), [3, 2, 1, 0, 0, 0, 0, 0, 3, 1]);
    assert!(
        held.take_ready(deadline().clock, MonotonicTick(0))
            .is_none()
    );
    assert!(
        held.admit(
            submission(4),
            Arc::new(4),
            deadline().clock,
            MonotonicTick(0)
        )
        .is_err()
    );
    assert_eq!(held.cumulative_observation(), before);
    assert_eq!(held.revoke_all(), [second.id]);
    assert!(held.ticket(prepared.id).is_none());
    let before = held.cumulative_observation();
    assert_eq!(before.revoked, 3);
    assert_eq!(before.released, 1);
    assert_eq!(before.terminal_before_service_dispatch, 1);
    assert!(held.retire_revoked_after_worker_join().is_err());
    assert!(held.revoke_all().is_empty());
    assert_eq!(held.cumulative_observation(), before);
    held.retire_after_physical_fence(&first).unwrap();
    assert_eq!(held.cumulative_observation().released, 2);
    // This mock invokes the owner hook, not an actual native shutdown or join.
    assert_eq!(held.retire_revoked_after_worker_join().unwrap(), [second]);
    let observation = held.cumulative_observation();
    assert_eq!(totals(observation), [3, 2, 1, 0, 0, 0, 1, 3, 3, 1]);
    assert_eq!(phases(observation.phase_peaks), [3, 2, 1, 0, 0, 1]);
    assert_eq!(observation.held_peak, 3);
    assert_current(&held, [0; 6]);
}

#[test]
fn mismatched_unknown_identity_returns_payload_without_observation_or_phase_changes() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let ticket = admit(&mut held, 1);
    held.take_dispatch().unwrap();
    let before = held.cumulative_observation();
    let mut wrong = ticket;
    wrong.question = Digest([77; 32]);
    let rejected = held
        .complete(RoleCompletion {
            ticket: wrong,
            physical: PhysicalRoleCompletion::Unknown,
            result: Err(RoleFailure::InvalidOutput),
        })
        .unwrap_err();
    assert!(matches!(
        rejected.error,
        CoordinatorError::IdentityMismatch(_)
    ));
    assert_eq!(
        rejected.completion.physical,
        PhysicalRoleCompletion::Unknown
    );
    assert_eq!(rejected.completion.result, Err(RoleFailure::InvalidOutput));
    assert_eq!(held.cumulative_observation(), before);
    assert_eq!(held.phase(ticket.id), Some(TicketPhase::Submitted));
    assert!(held.admission_open());
    complete(&mut held, ticket);
    assert_eq!(held.cumulative_observation().physically_completed, 1);
    assert_eq!(held.cumulative_observation().physical_unknown_events, 0);
}

#[test]
fn quarantine_fence_releases_once_without_delivery_acceptance_or_fabricated_completion() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let view_owner = Arc::new(1);
    let ticket = held
        .admit(
            submission(1),
            Arc::clone(&view_owner),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    held.take_dispatch().unwrap();
    let before = held.cumulative_observation();
    assert!(held.retire_after_physical_fence(&ticket).is_err());
    assert_eq!(held.cumulative_observation(), before);
    held.complete(RoleCompletion {
        ticket,
        physical: PhysicalRoleCompletion::Unknown,
        result: Ok(42),
    })
    .unwrap();
    assert_eq!(Arc::strong_count(&view_owner), 2);
    let before = held.cumulative_observation();
    assert!(held.discard_delivered(&ticket).is_err());
    assert!(
        held.accept(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    assert_eq!(held.cumulative_observation(), before);
    held.retire_after_physical_fence(&ticket).unwrap();
    assert_eq!(Arc::strong_count(&view_owner), 1);
    let after = held.cumulative_observation();
    assert_eq!(totals(after), [1, 1, 0, 0, 0, 0, 0, 1, 1, 1]);
    assert_eq!(after.phase_peaks, before.phase_peaks);
    assert_eq!(after.held_peak, before.held_peak);
    assert_current(&held, [0; 6]);
    assert!(held.retire_after_physical_fence(&ticket).is_err());
    assert_eq!(held.cumulative_observation(), after);
}

#[test]
fn joined_owner_cleanup_releases_remaining_phases_and_keeps_historical_high_water() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let tickets: Vec<_> = (1..=4).map(|sequence| admit(&mut held, sequence)).collect();
    for _ in 0..3 {
        held.take_dispatch().unwrap();
    }
    held.physical_started(&tickets[2]).unwrap();
    complete(&mut held, tickets[0]);
    complete(&mut held, tickets[1]);
    held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    let before = held.cumulative_observation();
    assert!(held.retire_revoked_after_worker_join().is_err());
    assert_eq!(held.cumulative_observation(), before);
    assert_eq!(held.revoke_all(), [tickets[2].id]);
    assert_eq!(held.cumulative_observation().released, 1);
    let before = held.cumulative_observation();
    // The owner must possess real shutdown/join receipts outside this fixture.
    assert_eq!(
        held.retire_revoked_after_worker_join().unwrap(),
        tickets[..3]
    );
    let after = held.cumulative_observation();
    assert_eq!(totals(after), [4, 3, 1, 2, 1, 0, 1, 4, 4, 0]);
    assert_eq!(after.phase_peaks, before.phase_peaks);
    assert_eq!(after.held_peak, 4);
    assert_current(&held, [0; 6]);
    assert!(held.retire_revoked_after_worker_join().unwrap().is_empty());
    assert_eq!(held.cumulative_observation(), after);
    assert!(
        held.take_ready(deadline().clock, MonotonicTick(0))
            .is_none()
    );
}

#[test]
fn confirmed_backend_error_counts_delivery_and_release_but_never_logical_acceptance() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    let ticket = admit(&mut held, 1);
    held.take_dispatch().unwrap();
    held.complete(RoleCompletion {
        ticket,
        physical: PhysicalRoleCompletion::Confirmed,
        result: Err(RoleFailure::Backend("mock backend fault".into())),
    })
    .unwrap();
    let ready = held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    assert!(ready.result.is_err());
    let before = held.cumulative_observation();
    assert!(
        held.accept(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    assert_eq!(held.cumulative_observation(), before);
    held.discard_delivered(&ticket).unwrap();
    let after = held.cumulative_observation();
    assert_eq!(totals(after), [1, 1, 0, 1, 1, 0, 0, 1, 0, 0]);
    assert_eq!(after.phase_peaks, before.phase_peaks);
    assert_current(&held, [0; 6]);
}

#[test]
fn failed_admission_preserves_counters_and_phase_peaks_at_capacity() {
    let mut held = registry(CoordinatorPolicy::Fixed);
    for sequence in 1..=16 {
        admit(&mut held, sequence);
    }
    let before = held.cumulative_observation();
    assert_eq!(before.admitted, 16);
    assert_eq!(before.current.prepared, 16);
    assert_eq!(before.phase_peaks.prepared, 16);
    assert_eq!(before.held_peak, 16);
    assert!(matches!(
        held.admit(
            submission(17),
            Arc::new(17),
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::Capacity(_))
    ));
    assert_eq!(held.cumulative_observation(), before);
    held.revoke_all();
    let after = held.cumulative_observation();
    assert_eq!(totals(after), [16, 0, 0, 0, 0, 0, 16, 16, 16, 0]);
    assert_eq!(after.phase_peaks, before.phase_peaks);
    assert_eq!(after.held_peak, 16);
    assert_current(&held, [0; 6]);
}

#[test]
fn internal_retirement_counts_only_actual_removal_even_if_owner_hook_is_repeated() {
    let mut held = registry(CoordinatorPolicy::Adaptive);
    let accepted = admit(&mut held, 1);
    let discarded = admit(&mut held, 2);
    held.take_dispatch().unwrap();
    held.take_dispatch().unwrap();
    complete(&mut held, accepted);
    held.complete(RoleCompletion {
        ticket: discarded,
        physical: PhysicalRoleCompletion::Confirmed,
        result: Err(RoleFailure::Canceled),
    })
    .unwrap();
    held.take_ready_id(accepted.id, deadline().clock, MonotonicTick(0))
        .unwrap()
        .unwrap();
    held.retire_accepted(accepted.id);
    let before = held.cumulative_observation();
    assert_eq!(before.logically_accepted, 1);
    assert_eq!(before.released, 1);
    held.retire_accepted(accepted.id);
    held.retire(accepted.id);
    assert_eq!(held.cumulative_observation(), before);
    held.take_ready_id(discarded.id, deadline().clock, MonotonicTick(0))
        .unwrap()
        .unwrap();
    held.retire(discarded.id);
    let after = held.cumulative_observation();
    assert_eq!(totals(after), [2, 2, 0, 2, 2, 1, 0, 2, 0, 0]);
    held.retire(discarded.id);
    held.retire_accepted(discarded.id);
    assert_eq!(held.cumulative_observation(), after);
    assert_current(&held, [0; 6]);
}

#[test]
fn checked_request_sequence_exhaustion_preserves_cumulative_observations() {
    let mut held = RoleTicketRegistry::<u64, u64>::new(
        authority(),
        512,
        CoordinatorPolicy::Fixed,
        u64::MAX - 1,
    )
    .unwrap();
    let ticket = admit(&mut held, 1);
    assert_eq!(ticket.id.sequence, u64::MAX - 1);
    assert_eq!(held.next_request_sequence(), u64::MAX);
    held.take_dispatch().unwrap();
    complete(&mut held, ticket);
    held.take_ready(deadline().clock, MonotonicTick(0)).unwrap();
    held.accept(&ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    let before = held.cumulative_observation();
    assert_eq!(totals(before), [1, 1, 0, 1, 1, 1, 0, 1, 0, 0]);
    assert_current(&held, [0; 6]);
    assert_eq!(
        held.admit(
            submission(2),
            Arc::new(2),
            deadline().clock,
            MonotonicTick(0),
        ),
        Err(CoordinatorError::SequenceExhausted)
    );
    assert_eq!(held.next_request_sequence(), u64::MAX);
    assert_eq!(held.cumulative_observation(), before);

    let mut exhausted =
        RoleTicketRegistry::<u64, u64>::new(authority(), 512, CoordinatorPolicy::Fixed, u64::MAX)
            .unwrap();
    assert_eq!(
        exhausted.admit(
            submission(1),
            Arc::new(1),
            deadline().clock,
            MonotonicTick(0),
        ),
        Err(CoordinatorError::SequenceExhausted)
    );
    assert_eq!(exhausted.next_request_sequence(), u64::MAX);
    assert_eq!(
        exhausted.cumulative_observation(),
        RoleTicketCumulativeObservation::default()
    );
    assert_current(&exhausted, [0; 6]);
}
