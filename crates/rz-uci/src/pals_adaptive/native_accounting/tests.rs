//! Authored arithmetic/origin fixtures. These exercise private reducers and
//! are not native NN, cache, ticket authority or process-shutdown evidence.
use super::*;

fn evidence(role: Role, public_miss: bool) -> TicketExecutionEvidence {
    let miss = u64::from(public_miss);
    let validator = u64::from(role == Role::Validator);
    TicketExecutionEvidence {
        execution: None,
        completed_nn_inputs: Some(1 + miss),
        measurement_complete: true,
        physical_fresh: Some(true),
        accounting: Some(TicketNativeAccountingEvidence {
            public_cache_hits: 1 - miss,
            public_cache_misses: miss,
            public_nn_runs_attempted: miss,
            public_nn_runs_completed: miss,
            public_nn_runs_failed_known: 0,
            role_nn_runs_attempted: 1,
            role_nn_runs_completed: 1,
            role_nn_runs_failed_known: 0,
            validator_nn_runs_attempted: validator,
            validator_nn_runs_completed: validator,
            validator_nn_runs_failed_known: 0,
        }),
    }
}

#[test]
fn public_kv_hit_is_not_a_full_evaluation_cache_or_a_second_native_ticket() {
    let e = evidence(Role::Proposer, false);
    let mut snapshot = AdaptiveNativeAccountingSnapshot::default();
    snapshot
        .observe_completion_facts(
            Role::Proposer,
            true,
            PhysicalRoleCompletion::Confirmed,
            true,
            Some(e),
        )
        .unwrap();
    snapshot
        .observe_owner_acceptance(Role::Proposer, Some(e))
        .unwrap();
    assert_eq!(snapshot.public_cache_hits(), 1);
    assert_eq!(snapshot.public_nn_runs_completed(), 0);
    assert_eq!(snapshot.role_nn_runs_completed(), 1);
    assert_eq!(snapshot.physical_execution_tickets(), 1);
    assert_eq!(snapshot.cache_evaluations_accepted(), Some(0));
    assert_eq!(snapshot.owner_accepted_outputs(), 1);
    assert_eq!(snapshot.search_consumed_outputs(), 0);
    snapshot
        .observe_search_consumption(Role::Proposer, Some(e))
        .unwrap();
    assert_eq!(snapshot.search_consumed_outputs(), 1);
    assert!(snapshot.physical_partition_complete_for_dispatches(1));
}

#[test]
fn public_and_validator_runs_are_two_inputs_inside_one_original_ticket() {
    let mut snapshot = AdaptiveNativeAccountingSnapshot::default();
    snapshot
        .observe_completion_facts(
            Role::Validator,
            true,
            PhysicalRoleCompletion::Confirmed,
            true,
            Some(evidence(Role::Validator, true)),
        )
        .unwrap();
    assert_eq!(snapshot.physical_execution_tickets(), 1);
    assert_eq!(snapshot.public_nn_runs_completed(), 1);
    assert_eq!(snapshot.role_nn_runs_completed(), 1);
    assert_eq!(snapshot.validator_nn_runs_completed(), 1);
    assert_eq!(snapshot.validator_called_for_dispatches(1), Some(true));
    assert!(snapshot.physical_partition_complete_for_dispatches(1));
}

#[test]
fn public_graph_known_failure_does_not_invent_a_validator_run_or_completed_input() {
    let mut e = evidence(Role::Validator, true);
    e.completed_nn_inputs = Some(0);
    let actual = e.accounting.as_mut().unwrap();
    actual.public_nn_runs_completed = 0;
    actual.public_nn_runs_failed_known = 1;
    actual.role_nn_runs_attempted = 0;
    actual.role_nn_runs_completed = 0;
    actual.validator_nn_runs_attempted = 0;
    actual.validator_nn_runs_completed = 0;
    let mut snapshot = AdaptiveNativeAccountingSnapshot::default();
    snapshot
        .observe_completion_facts(
            Role::Validator,
            true,
            PhysicalRoleCompletion::Confirmed,
            false,
            Some(e),
        )
        .unwrap();
    assert_eq!(snapshot.physical_execution_tickets(), 1);
    assert_eq!(snapshot.public_nn_runs_completed(), 0);
    assert_eq!(snapshot.public_nn_runs_failed_known(), 1);
    assert_eq!(snapshot.validator_called_for_dispatches(1), Some(false));
    assert_eq!(snapshot.owner_accepted_outputs(), 0);
    assert_eq!(snapshot.search_consumed_outputs(), 0);
}

#[test]
fn known_private_forward_failure_is_a_v_call_without_a_completed_nn_input() {
    let mut e = evidence(Role::Validator, false);
    e.completed_nn_inputs = Some(0);
    let actual = e.accounting.as_mut().unwrap();
    actual.role_nn_runs_completed = 0;
    actual.role_nn_runs_failed_known = 1;
    actual.validator_nn_runs_completed = 0;
    actual.validator_nn_runs_failed_known = 1;
    let mut snapshot = AdaptiveNativeAccountingSnapshot::default();
    snapshot
        .observe_completion_facts(
            Role::Validator,
            true,
            PhysicalRoleCompletion::Confirmed,
            false,
            Some(e),
        )
        .unwrap();
    assert_eq!(snapshot.validator_called_for_dispatches(1), Some(true));
    assert_eq!(snapshot.role_nn_runs_completed(), 0);
    assert_eq!(snapshot.validator_nn_runs_failed_known(), 1);
    assert_eq!(snapshot.physical_execution_tickets(), 1);
}

#[test]
fn actual_skip_fact_is_terminal_before_nn_but_callback_attempt_without_evidence_is_unknown() {
    let mut skipped = AdaptiveNativeAccountingSnapshot::default();
    skipped
        .observe_completion_facts(
            Role::Validator,
            false,
            PhysicalRoleCompletion::Confirmed,
            false,
            None,
        )
        .unwrap();
    assert_eq!(skipped.backend_run_skipped_commands(), 1);
    assert_eq!(skipped.terminal_without_nn_dispatch(), 1);
    assert_eq!(skipped.physical_execution_tickets(), 0);
    assert_eq!(skipped.validator_called_for_dispatches(1), Some(false));
    assert!(skipped.physical_partition_complete_for_dispatches(1));

    let mut attempted = AdaptiveNativeAccountingSnapshot::default();
    attempted
        .observe_completion_facts(
            Role::Validator,
            true,
            PhysicalRoleCompletion::Confirmed,
            false,
            None,
        )
        .unwrap();
    assert_eq!(attempted.unknown_native_accounting_commands(), 1);
    assert_eq!(attempted.validator_called_for_dispatches(1), None);
    assert!(!attempted.physical_partition_complete_for_dispatches(1));
}

#[test]
fn inconsistent_counts_unknown_physical_and_missing_original_command_block_process_knowledge() {
    let good = evidence(Role::Critic, false);
    let mut mismatch = good;
    mismatch.completed_nn_inputs = Some(2);
    let mut foreign_role = good;
    foreign_role
        .accounting
        .as_mut()
        .unwrap()
        .validator_nn_runs_attempted = 1;
    for (physical, evidence) in [
        (PhysicalRoleCompletion::Confirmed, Some(mismatch)),
        (PhysicalRoleCompletion::Confirmed, Some(foreign_role)),
        (PhysicalRoleCompletion::Unknown, Some(good)),
        (PhysicalRoleCompletion::Confirmed, None),
    ] {
        let mut snapshot = AdaptiveNativeAccountingSnapshot::default();
        snapshot
            .observe_completion_facts(Role::Critic, true, physical, true, evidence)
            .unwrap();
        assert_eq!(snapshot.validator_called_for_dispatches(1), None);
        assert_eq!(snapshot.unknown_native_accounting_commands(), 1);
    }
    let mut missing = AdaptiveNativeAccountingSnapshot::default();
    missing
        .observe_completion_facts(
            Role::Critic,
            true,
            PhysicalRoleCompletion::Confirmed,
            true,
            Some(good),
        )
        .unwrap();
    assert_eq!(missing.validator_called_for_dispatches(2), None);
}

#[test]
fn owner_ack_and_search_consumption_use_their_own_original_neural_origin() {
    let mut snapshot = AdaptiveNativeAccountingSnapshot::default();
    let e = evidence(Role::Critic, true);
    snapshot
        .observe_owner_acceptance(Role::Critic, Some(e))
        .unwrap();
    assert_eq!(snapshot.owner_accepted_outputs(), 1);
    assert_eq!(snapshot.owner_accepted_neural_outputs(), 1);
    assert_eq!(snapshot.search_consumed_outputs(), 0);
    snapshot
        .observe_search_consumption(Role::Critic, None)
        .unwrap();
    assert_eq!(snapshot.search_consumed_origin_unknown(), 1);
    assert!(!snapshot.search_consumption_origin_known());
    assert_eq!(snapshot.cache_evaluations_accepted(), Some(0));
    snapshot
        .observe_owner_acceptance(Role::Critic, None)
        .unwrap();
    assert_eq!(snapshot.cache_evaluations_accepted(), None);
}

#[test]
fn zero_run_success_does_not_become_a_full_cache_hit() {
    let mut e = evidence(Role::Proposer, false);
    e.completed_nn_inputs = Some(0);
    let actual = e.accounting.as_mut().unwrap();
    actual.public_cache_hits = 0;
    actual.role_nn_runs_attempted = 0;
    actual.role_nn_runs_completed = 0;
    let mut snapshot = AdaptiveNativeAccountingSnapshot::default();
    snapshot
        .observe_completion_facts(
            Role::Proposer,
            true,
            PhysicalRoleCompletion::Confirmed,
            true,
            Some(e),
        )
        .unwrap();
    assert_eq!(snapshot.validator_called_for_dispatches(1), Some(false));
    assert_eq!(snapshot.unknown_evaluation_origin_commands(), 1);
    assert!(!snapshot.physical_partition_complete_for_dispatches(1));
    snapshot
        .observe_owner_acceptance(Role::Proposer, Some(e))
        .unwrap();
    assert_eq!(snapshot.cache_evaluations_accepted(), None);
}

#[test]
fn overflow_is_atomic_and_unknown_merge_never_becomes_known() {
    let mut snapshot = AdaptiveNativeAccountingSnapshot {
        completed_service_commands: u64::MAX,
        ..AdaptiveNativeAccountingSnapshot::default()
    };
    let original = snapshot;
    assert!(
        snapshot
            .observe_completion_facts(
                Role::Proposer,
                false,
                PhysicalRoleCompletion::Confirmed,
                false,
                None
            )
            .is_err()
    );
    assert_eq!(snapshot, original);
    let mut known = AdaptiveNativeAccountingSnapshot::default();
    known
        .observe_completion_facts(
            Role::Proposer,
            true,
            PhysicalRoleCompletion::Confirmed,
            true,
            Some(evidence(Role::Proposer, false)),
        )
        .unwrap();
    let mut unknown = AdaptiveNativeAccountingSnapshot::default();
    unknown
        .observe_completion_facts(
            Role::Proposer,
            true,
            PhysicalRoleCompletion::Confirmed,
            false,
            None,
        )
        .unwrap();
    known.checked_merge(unknown).unwrap();
    assert_eq!(known.completed_service_commands(), 2);
    assert_eq!(known.validator_called_for_dispatches(2), None);
    assert_eq!(known.role_nn_runs_completed(), 1);
}
