use super::*;
use crate::pals::coordinator::test_support::*;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::{Color, ExecutionId};
use std::time::Duration;

fn machine() -> RefinementMachine<u64, RefinementStage> {
    RefinementMachine::new(authority(), deadline(), 512).unwrap()
}
fn clock() -> ClockDomain {
    deadline().clock
}
fn now() -> MonotonicTick {
    MonotonicTick(0)
}
fn bind(
    machine: &mut RefinementMachine<u64, RefinementStage>,
    id: FrameId,
    sequence: u64,
) -> StepEvidence {
    let stage = machine.frame(id).unwrap().input.stage;
    let execution = if stage.role().is_some() {
        let ticket = ticket(sequence, stage);
        if matches!(
            stage,
            RefinementStage::FreshRepairedEndpoint | RefinementStage::FreshCounterEndpoint
        ) {
            machine
                .seal_fresh_pair(
                    id,
                    FreshPairSeal::from_ticket(&ticket, Digest([1; 32]), Color::White),
                )
                .unwrap();
        }
        machine.bind_role(id, &ticket, clock(), now()).unwrap();
        StepExecution::Role(ticket.id)
    } else {
        let job = CpuJobEnvelope {
            id: CpuJobId {
                authority: authority(),
                sequence: stage as u64 + 1,
            },
            state_identity: state(),
            question: Digest([stage as u8; 32]),
            root_restriction: Digest([4; 32]),
            evaluation_namespace: Digest([5; 32]),
            search_conditions: Digest([6; 32]),
            total_native_node_credit: 65536,
            hard_deadline: deadline(),
        };
        let grant = SliceGrant {
            job: job.id,
            execution: ExecutionId::new(authority().epoch, sequence),
            owner: CpuWorkerId(0),
            node_credit: 4096,
            soft_lease: Duration::from_millis(8),
            control_sequence: sequence,
        };
        machine.bind_cpu(id, &job, &grant, clock(), now()).unwrap();
        StepExecution::Cpu {
            job: job.id,
            execution: grant.execution,
        }
    };
    StepEvidence {
        frame: id,
        stage,
        execution,
        submitted_question: Digest([stage as u8; 32]),
        submitted_view: view(1),
        completed_evidence: Digest([sequence as u8; 32]),
    }
}

#[test]
fn actual_completions_drive_entire_fixed_refinement_order() {
    let mut held = machine();
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let stages = [
        RefinementStage::ProposerRanking,
        RefinementStage::ProposalContinuation,
        RefinementStage::CpuValidation,
        RefinementStage::ProposalValueFresh,
        RefinementStage::CriticDivergence,
        RefinementStage::CpuDiscovery,
        RefinementStage::CriticReply,
        RefinementStage::CounterContinuation,
        RefinementStage::CounterValidation,
        RefinementStage::CounterValueFresh,
        RefinementStage::Repair,
        RefinementStage::RepairValidation,
        RefinementStage::RepairValueFresh,
        RefinementStage::CriticRecheck,
        RefinementStage::RecheckCounterValidation,
        RefinementStage::FreshRepairedEndpoint,
        RefinementStage::FreshCounterEndpoint,
    ];
    for (index, stage) in stages.iter().copied().enumerate() {
        assert_eq!(
            held.ready_work(id, clock(), now()).unwrap().input.stage,
            stage
        );
        let evidence = bind(&mut held, id, index as u64 + 1);
        if stage == RefinementStage::CriticRecheck {
            assert!(matches!(
                held.resolve(
                    evidence,
                    FrameTransition::Continue(input(RefinementStage::FreshRepairedEndpoint)),
                    clock(),
                    now(),
                ),
                Err(CoordinatorError::InvalidTransition(_))
            ));
            let frame = held.frame(id).unwrap();
            assert_eq!(frame.input.stage, RefinementStage::CriticRecheck);
            assert_eq!(frame.phase, FramePhase::Waiting(evidence.execution));
            assert_eq!(frame.evidence.len(), index);
        }
        if matches!(
            stage,
            RefinementStage::CriticRecheck | RefinementStage::RecheckCounterValidation
        ) {
            // A generic worker evidence digest is not the Search port's
            // opaque proof of two exact Rules terminal endpoints.
            assert!(matches!(
                held.resolve(
                    evidence,
                    FrameTransition::Complete {
                        scope: FrameCompletionScope::CompletedRulesPair
                    },
                    clock(),
                    now(),
                ),
                Err(CoordinatorError::MissingEvidence)
            ));
            let frame = held.frame(id).unwrap();
            assert_eq!(frame.input.stage, stage);
            assert_eq!(frame.phase, FramePhase::Waiting(evidence.execution));
            assert_eq!(frame.evidence.len(), index);
        }
        if matches!(
            stage,
            RefinementStage::FreshRepairedEndpoint | RefinementStage::FreshCounterEndpoint
        ) {
            held.seal_fresh_pair(
                id,
                FreshPairSeal::from_ticket(
                    &ticket(index as u64 + 1, stage),
                    Digest([1; 32]),
                    Color::White,
                ),
            )
            .unwrap();
        }
        let transition = stages.get(index + 1).map_or(
            FrameTransition::Complete {
                scope: FrameCompletionScope::CompletedFreshPair,
            },
            |next| FrameTransition::Continue(input(*next)),
        );
        let closed = held.resolve(evidence, transition, clock(), now()).unwrap();
        if index == stages.len() - 1 {
            let closed = closed.unwrap();
            assert_eq!(closed.evidence.len(), stages.len());
            assert_eq!(closed.original_first_move, first_move());
            assert_eq!(closed.unknown, None);
        } else {
            assert!(closed.is_none());
        }
    }
    assert_eq!(held.frames().count(), 0);
}

#[test]
fn accepted_branch_can_continue_next_counter_without_reusing_old_pair_seal() {
    let mut held = machine();
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let stages = [
        RefinementStage::ProposerRanking,
        RefinementStage::ProposalContinuation,
        RefinementStage::CpuValidation,
        RefinementStage::CriticDivergence,
        RefinementStage::CpuDiscovery,
        RefinementStage::CriticReply,
        RefinementStage::CounterContinuation,
        RefinementStage::CounterValidation,
        RefinementStage::Repair,
        RefinementStage::RepairValidation,
        RefinementStage::CriticRecheck,
        RefinementStage::RecheckCounterValidation,
        RefinementStage::FreshRepairedEndpoint,
        RefinementStage::FreshCounterEndpoint,
    ];
    for (index, stage) in stages.iter().copied().enumerate() {
        let evidence = bind(&mut held, id, index as u64 + 1);
        let next = stages
            .get(index + 1)
            .copied()
            .unwrap_or(RefinementStage::CounterContinuation);
        held.resolve(
            evidence,
            FrameTransition::Continue(input(next)),
            clock(),
            now(),
        )
        .unwrap();
        assert_eq!(evidence.stage, stage);
    }
    let frame = held.frame(id).unwrap();
    assert_eq!(frame.input.stage, RefinementStage::CounterContinuation);
    assert_eq!(frame.original_first_move, first_move());
    assert_eq!(frame.original_deadline, deadline());
    assert!(frame.fresh_pair.is_none());
    assert!(frame.rules_terminal_pair.is_none());
    assert_eq!(frame.evidence.len(), stages.len());
    // The next distinct CPU/C branch gets a new real request; the earlier
    // pair's policy output is not reused or counted as this question's result.
    let evidence = bind(&mut held, id, stages.len() as u64 + 1);
    held.resolve(
        evidence,
        FrameTransition::Continue(input(RefinementStage::CounterValidation)),
        clock(),
        now(),
    )
    .unwrap();
}

#[test]
fn continuation_root_mismatch_is_atomic_and_new_repair_obeys_existing_cap() {
    let mut held = machine();
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let evidence = bind(&mut held, id, 1);
    let mut wrong = input(RefinementStage::ProposalContinuation);
    wrong.view_seal.root_state.semantic = Digest([233; 32]);
    assert!(
        held.resolve(evidence, FrameTransition::Continue(wrong), clock(), now())
            .is_err()
    );
    assert!(held.frame(id).unwrap().evidence.is_empty());
    assert_eq!(
        held.frame(id).unwrap().phase,
        FramePhase::Waiting(evidence.execution)
    );
    held.resolve(
        evidence,
        FrameTransition::Continue(input(RefinementStage::ProposalContinuation)),
        clock(),
        now(),
    )
    .unwrap();
    let stages = [
        RefinementStage::ProposalContinuation,
        RefinementStage::CpuValidation,
        RefinementStage::CriticDivergence,
        RefinementStage::CpuDiscovery,
        RefinementStage::CriticReply,
        RefinementStage::CounterContinuation,
        RefinementStage::CounterValidation,
        RefinementStage::Repair,
        RefinementStage::RepairValidation,
        RefinementStage::CriticRecheck,
    ];
    for (index, stage) in stages.iter().copied().enumerate() {
        let evidence = bind(&mut held, id, index as u64 + 2);
        assert_eq!(stage, evidence.stage);
        let next = if let Some(stage) = stages.get(index + 1) {
            input(*stage)
        } else {
            let mut next = input(RefinementStage::Repair);
            next.question_digest = Digest([234; 32]);
            next
        };
        held.resolve(evidence, FrameTransition::Continue(next), clock(), now())
            .unwrap();
    }
    assert_eq!(
        held.repair_counts
            .iter()
            .find(|(first, _)| *first == first_move())
            .unwrap()
            .1,
        2
    );
}

#[test]
fn unsolicited_wrong_question_or_stage_cannot_commit_or_advance() {
    let mut held = machine();
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let mut wrong = ticket(1, RefinementStage::ProposerRanking);
    wrong.question = Digest([77; 32]);
    assert!(held.bind_role(id, &wrong, clock(), now()).is_err());
    assert_eq!(held.frame(id).unwrap().budget, budget());
    let evidence = bind(&mut held, id, 1);
    let mut changed = evidence;
    changed.submitted_view = view(2);
    assert!(
        held.resolve(
            changed,
            FrameTransition::Continue(input(RefinementStage::ProposalContinuation)),
            clock(),
            now()
        )
        .is_err()
    );
    assert!(held.frame(id).unwrap().evidence.is_empty());
    assert!(
        held.resolve(
            evidence,
            FrameTransition::Continue(input(RefinementStage::Repair)),
            clock(),
            now()
        )
        .is_err()
    );
    assert!(held.frame(id).unwrap().evidence.is_empty());
    held.resolve(
        evidence,
        FrameTransition::Continue(input(RefinementStage::ProposalContinuation)),
        clock(),
        now(),
    )
    .unwrap();
}

#[test]
fn revoked_generation_never_accepts_old_completed_work() {
    let mut held = machine();
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let evidence = bind(&mut held, id, 1);
    assert_eq!(held.revoke(), vec![evidence.execution]);
    assert!(matches!(
        held.resolve(
            evidence,
            FrameTransition::Complete {
                scope: FrameCompletionScope::RulesTerminal
            },
            clock(),
            now()
        ),
        Err(CoordinatorError::AdmissionClosed)
    ));
    assert_eq!(held.frames().count(), 1);
    held.retired_after_worker_join();
    assert_eq!(held.frames().count(), 0);
}

#[test]
fn repair_is_nonrecursive_bounded_deduplicated_and_keeps_first_move() {
    let mut held = machine();
    let parent = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let repair = input(RefinementStage::Repair);
    let id = held
        .enqueue_repair(
            parent,
            first_move(),
            repair.clone(),
            budget(),
            None,
            clock(),
            now(),
        )
        .unwrap();
    let frame = held.frame(id).unwrap();
    assert_eq!(frame.original_first_move, first_move());
    assert_eq!(frame.parent, Some(parent));
    assert!(matches!(
        held.enqueue_repair(
            parent,
            first_move(),
            repair,
            budget(),
            Some(id),
            clock(),
            now()
        ),
        Err(CoordinatorError::Duplicate)
    ));
    for sequence in 2..=3 {
        let mut next = input(RefinementStage::Repair);
        next.view_seal = view(sequence);
        held.enqueue_repair(
            parent,
            first_move(),
            next,
            budget(),
            Some(id),
            clock(),
            now(),
        )
        .unwrap();
    }
    let mut fourth = input(RefinementStage::Repair);
    fourth.view_seal = view(4);
    assert!(matches!(
        held.enqueue_repair(
            parent,
            first_move(),
            fourth,
            budget(),
            Some(id),
            clock(),
            now()
        ),
        Err(CoordinatorError::Capacity(_))
    ));
}

#[test]
fn fresh_pair_cannot_advance_to_latest_view() {
    let mut held = machine();
    let parent = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let id = held
        .enqueue_repair(
            parent,
            first_move(),
            input(RefinementStage::Repair),
            budget(),
            None,
            clock(),
            now(),
        )
        .unwrap();
    for (sequence, next) in [
        (1, RefinementStage::RepairValidation),
        (2, RefinementStage::CriticRecheck),
        (3, RefinementStage::RecheckCounterValidation),
        (4, RefinementStage::FreshRepairedEndpoint),
    ] {
        let evidence = bind(&mut held, id, sequence);
        held.resolve(
            evidence,
            FrameTransition::Continue(input(next)),
            clock(),
            now(),
        )
        .unwrap();
    }
    let evidence = bind(&mut held, id, 5);
    let seal = FreshPairSeal::from_ticket(
        &ticket(5, RefinementStage::FreshRepairedEndpoint),
        Digest([1; 32]),
        Color::White,
    );
    held.seal_fresh_pair(id, seal).unwrap();
    let mut changed = input(RefinementStage::FreshCounterEndpoint);
    changed.public_context = Digest([2; 32]);
    assert!(
        held.resolve(evidence, FrameTransition::Continue(changed), clock(), now())
            .is_err()
    );
    held.resolve(
        evidence,
        FrameTransition::Continue(input(RefinementStage::FreshCounterEndpoint)),
        clock(),
        now(),
    )
    .unwrap();
    let mut wrong = seal;
    wrong.perspective = Color::Black;
    assert!(held.seal_fresh_pair(id, wrong).is_err());
}

#[test]
fn no_work_yields_stop_after_three_and_original_question_survives_resume() {
    let mut held = machine();
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    for (sequence, next) in [
        (1, RefinementStage::ProposalContinuation),
        (2, RefinementStage::CpuValidation),
    ] {
        let evidence = bind(&mut held, id, sequence);
        held.resolve(
            evidence,
            FrameTransition::Continue(input(next)),
            clock(),
            now(),
        )
        .unwrap();
    }
    let original = held.frame(id).unwrap().input.question_digest;
    for sequence in 3..=5 {
        let evidence = bind(&mut held, id, sequence);
        let retry = held
            .observe_yield(id, evidence.execution, None, 0, false)
            .unwrap();
        assert_eq!(retry, sequence < 5);
        assert_eq!(held.frame(id).unwrap().input.question_digest, original);
        if retry {
            held.make_resumed_ready(id).unwrap();
        }
    }
    assert!(held.make_resumed_ready(id).is_err());
}

#[test]
fn progress_does_not_charge_or_complete_any_work() {
    let mut held = machine();
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let evidence = bind(&mut held, id, 1);
    let after_submission = held.frame(id).unwrap().budget;
    held.publish_progress(id, evidence.execution, 1).unwrap();
    assert!(held.publish_progress(id, evidence.execution, 1).is_err());
    assert_eq!(
        held.frame(id).unwrap().phase,
        FramePhase::Waiting(evidence.execution)
    );
    assert_eq!(held.frame(id).unwrap().budget, after_submission);
    assert!(held.frame(id).unwrap().evidence.is_empty());
}

#[test]
fn frame_capacity_deadline_and_role_budget_are_explicit() {
    let mut held = machine();
    for _ in 0..64 {
        held.admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    }
    assert!(matches!(
        held.admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now()
        ),
        Err(CoordinatorError::Capacity(_))
    ));
    let mut held = machine();
    let mut empty = budget();
    empty.role_submissions_remaining = 0;
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            empty,
            clock(),
            now(),
        )
        .unwrap();
    assert!(matches!(
        held.bind_role(
            id,
            &ticket(1, RefinementStage::ProposerRanking),
            clock(),
            now()
        ),
        Err(CoordinatorError::Capacity(_))
    ));
    assert!(held.ready_work(id, clock(), deadline().at).is_err());
}

fn cpu_frame(held: &mut RefinementMachine<u64, RefinementStage>, first_sequence: u64) -> FrameId {
    let id = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    for (sequence, next) in [
        (first_sequence, RefinementStage::ProposalContinuation),
        (first_sequence + 1, RefinementStage::CpuValidation),
    ] {
        let evidence = bind(held, id, sequence);
        held.resolve(
            evidence,
            FrameTransition::Continue(input(next)),
            clock(),
            now(),
        )
        .unwrap();
    }
    id
}

#[test]
fn single_role_and_cpu_completions_cannot_drive_two_frames() {
    let mut held = machine();
    let first = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let second = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let evidence = bind(&mut held, first, 1);
    assert_eq!(
        held.bind_role(
            second,
            &ticket(1, RefinementStage::ProposerRanking),
            clock(),
            now()
        ),
        Err(CoordinatorError::Duplicate)
    );
    held.resolve(
        evidence,
        FrameTransition::Unknown {
            reason: "bounded test closure",
        },
        clock(),
        now(),
    )
    .unwrap();
    assert_eq!(
        held.bind_role(
            second,
            &ticket(1, RefinementStage::ProposerRanking),
            clock(),
            now()
        ),
        Err(CoordinatorError::Duplicate)
    );
    let mut held = machine();
    let first = cpu_frame(&mut held, 1);
    let second = cpu_frame(&mut held, 3);
    let evidence = bind(&mut held, first, 5);
    let stage = RefinementStage::CpuValidation;
    let job = CpuJobEnvelope {
        id: CpuJobId {
            authority: authority(),
            sequence: stage as u64 + 1,
        },
        state_identity: state(),
        question: Digest([stage as u8; 32]),
        root_restriction: Digest([4; 32]),
        evaluation_namespace: Digest([5; 32]),
        search_conditions: Digest([6; 32]),
        total_native_node_credit: 65536,
        hard_deadline: deadline(),
    };
    let StepExecution::Cpu { execution, .. } = evidence.execution else {
        panic!("CPU test");
    };
    let grant = SliceGrant {
        job: job.id,
        execution,
        owner: CpuWorkerId(0),
        node_credit: 4096,
        soft_lease: Duration::from_millis(8),
        control_sequence: 5,
    };
    assert_eq!(
        held.bind_cpu(second, &job, &grant, clock(), now()),
        Err(CoordinatorError::Duplicate)
    );
    assert_eq!(held.frame(second).unwrap().phase, FramePhase::Ready);
}

#[test]
fn late_previous_execution_yield_and_progress_cannot_suspend_current_slice() {
    let mut held = machine();
    let id = cpu_frame(&mut held, 1);
    let old = bind(&mut held, id, 3);
    held.publish_progress(id, old.execution, 50).unwrap();
    // Manager ack 1 and worker progress 50 are independent sequence domains.
    held.observe_yield(id, old.execution, Some(1), 10, false)
        .unwrap();
    held.make_resumed_ready(id).unwrap();
    let current = bind(&mut held, id, 4);
    assert!(
        held.observe_yield(id, old.execution, Some(99), 10, false)
            .is_err()
    );
    assert!(held.publish_progress(id, old.execution, 100).is_err());
    assert_eq!(
        held.frame(id).unwrap().phase,
        FramePhase::Waiting(current.execution)
    );
    held.publish_progress(id, current.execution, 51).unwrap();
    held.observe_yield(id, current.execution, None, 10, false)
        .unwrap();
    assert_eq!(held.frame(id).unwrap().phase, FramePhase::Suspended);
}

#[test]
fn paused_and_resumed_ready_cpu_jobs_are_both_revoked() {
    for resumed_ready in [false, true] {
        let mut held = machine();
        let id = cpu_frame(&mut held, 1);
        let evidence = bind(&mut held, id, 3);
        held.observe_yield(id, evidence.execution, None, 10, false)
            .unwrap();
        if resumed_ready {
            held.make_resumed_ready(id).unwrap();
        }
        assert_eq!(held.revoke(), vec![evidence.execution]);
        assert_eq!(
            held.make_resumed_ready(id),
            Err(CoordinatorError::AdmissionClosed)
        );
        assert_eq!(
            held.publish_progress(id, evidence.execution, 55),
            Err(CoordinatorError::AdmissionClosed)
        );
        assert_eq!(
            held.observe_yield(id, evidence.execution, None, 0, false),
            Err(CoordinatorError::AdmissionClosed)
        );
    }
}

#[test]
fn closed_parent_genealogy_rejects_fabrication_and_changed_first_move() {
    let mut held = machine();
    let parent = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let evidence = bind(&mut held, parent, 1);
    held.resolve(
        evidence,
        FrameTransition::Complete {
            scope: FrameCompletionScope::RulesTerminal,
        },
        clock(),
        now(),
    )
    .unwrap();
    held.enqueue_repair(
        parent,
        first_move(),
        input(RefinementStage::Repair),
        budget(),
        None,
        clock(),
        now(),
    )
    .unwrap();
    let wrong_first = Move16::encode(
        rz_contracts::Move::new(
            rz_contracts::Square::try_new(11).unwrap(),
            rz_contracts::Square::try_new(27).unwrap(),
            None,
        )
        .unwrap(),
    );
    assert!(
        held.enqueue_repair(
            parent,
            wrong_first,
            input(RefinementStage::Repair),
            budget(),
            None,
            clock(),
            now()
        )
        .is_err()
    );
    let future = FrameId {
        authority: authority(),
        sequence: u64::MAX,
    };
    assert!(
        held.enqueue_repair(
            future,
            first_move(),
            input(RefinementStage::Repair),
            budget(),
            None,
            clock(),
            now()
        )
        .is_err()
    );
}

#[test]
fn external_cpu_admission_has_no_native_node_credit() {
    let mut held = machine();
    let id = cpu_frame(&mut held, 1);
    let stage = RefinementStage::CpuValidation;
    let job = ExternalCpuJobEnvelope {
        id: CpuJobId {
            authority: authority(),
            sequence: 4,
        },
        state_identity: state(),
        question: Digest([stage as u8; 32]),
        root_restriction: Digest([4; 32]),
        evaluation_namespace: Digest([5; 32]),
        search_conditions: Digest([6; 32]),
        requested_reported_node_hint: 65536,
        hard_deadline: deadline(),
    };
    let grant = ExternalSliceGrant {
        job: job.id,
        execution: ExecutionId::new(authority().epoch, 6),
        owner: CpuWorkerId(0),
        requested_reported_node_hint: 65536,
        soft_quantum: Duration::from_millis(100),
        control_sequence: 3,
    };
    let before = held.frame(id).unwrap().budget;
    held.bind_external_cpu(id, &job, &grant, clock(), now())
        .unwrap();
    assert_eq!(held.frame(id).unwrap().budget, before);
    assert_eq!(
        held.frame(id).unwrap().phase,
        FramePhase::Waiting(StepExecution::ExternalCpu {
            job: job.id,
            execution: grant.execution
        })
    );
    assert!(
        held.observe_yield(
            id,
            StepExecution::ExternalCpu {
                job: job.id,
                execution: grant.execution
            },
            None,
            0,
            false
        )
        .is_err()
    );
}

#[test]
fn paused_unknown_closure_requires_real_owner_token_discard_receipt() {
    let mut held = machine();
    let id = cpu_frame(&mut held, 1);
    let evidence = bind(&mut held, id, 3);
    held.observe_yield(id, evidence.execution, None, 0, false)
        .unwrap();
    let StepExecution::Cpu { job, execution } = evidence.execution else {
        panic!("CPU test");
    };
    let mut termination = crate::cpu_worker::CpuJobTermination {
        job,
        owner: CpuWorkerId(0),
        last_execution: execution,
        completion: crate::cpu::CpuCompletion::Canceled,
        controls: rz_runtime::adaptive::ControlSnapshot {
            cancel_sequence: 1,
            cancel_ack: 1,
            terminal: true,
            ..Default::default()
        },
        retained_bytes: 1,
        shutdown: false,
    };
    held.revoke();
    assert!(
        held.retire_suspended_cpu(id, &termination, Digest([99; 32]), "no progress")
            .is_err()
    );
    termination.retained_bytes = 0;
    termination.owner = CpuWorkerId(1);
    assert!(
        held.retire_suspended_cpu(id, &termination, Digest([99; 32]), "no progress")
            .is_err()
    );
    termination.owner = CpuWorkerId(0);
    let closed = held
        .retire_suspended_cpu(id, &termination, Digest([99; 32]), "no progress")
        .unwrap();
    assert_eq!(closed.unknown, Some("no progress"));
    assert!(
        held.retire_suspended_cpu(id, &termination, Digest([99; 32]), "duplicate")
            .is_err()
    );
}

#[test]
fn revoked_waiting_cpu_retires_actual_cancel_without_accepting_late_yield_or_value() {
    let mut held = machine();
    let id = cpu_frame(&mut held, 1);
    let evidence = bind(&mut held, id, 3);
    let original_budget = held.frame(id).unwrap().budget;
    held.revoke();
    assert_eq!(
        held.observe_yield(id, evidence.execution, Some(1), 9, false),
        Err(CoordinatorError::AdmissionClosed)
    );
    let StepExecution::Cpu { job, execution } = evidence.execution else {
        panic!("native CPU");
    };
    let mut termination = crate::cpu_worker::CpuJobTermination {
        job,
        owner: CpuWorkerId(0),
        last_execution: execution,
        completion: crate::cpu::CpuCompletion::Canceled,
        controls: rz_runtime::adaptive::ControlSnapshot {
            cancel_sequence: 2,
            cancel_ack: 2,
            terminal: true,
            ..Default::default()
        },
        retained_bytes: 0,
        shutdown: false,
    };
    termination.last_execution.sequence += 1;
    assert!(
        held.retire_suspended_cpu(id, &termination, Digest([81; 32]), "root revoked")
            .is_err()
    );
    assert_eq!(
        held.frame(id).unwrap().phase,
        FramePhase::Waiting(evidence.execution)
    );
    assert_eq!(held.frame(id).unwrap().budget, original_budget);
    termination.last_execution = execution;
    let closed = held
        .retire_suspended_cpu(id, &termination, Digest([82; 32]), "root revoked")
        .unwrap();
    assert_eq!(closed.unknown, Some("root revoked"));
    assert_eq!(closed.completion_scope, None);
    assert!(held.frame(id).is_none());
}

#[test]
fn iterative_recheck_fresh_value_is_proposer_and_then_enters_critic() {
    assert_eq!(
        RefinementStage::RecheckRepairedValueFresh.role(),
        Some(rz_contracts::pals::Role::Proposer)
    );
    let mut held = machine();
    let parent = held
        .admit(
            first_move(),
            input(RefinementStage::ProposerRanking),
            budget(),
            clock(),
            now(),
        )
        .unwrap();
    let id = held
        .enqueue_repair(
            parent,
            first_move(),
            input(RefinementStage::Repair),
            budget(),
            None,
            clock(),
            now(),
        )
        .unwrap();
    // These are admitted CPU/mock completions, not native NN execution.
    for (sequence, next) in [
        (1, RefinementStage::RepairValidation),
        (2, RefinementStage::RecheckRepairedValueFresh),
    ] {
        let evidence = bind(&mut held, id, sequence);
        held.resolve(
            evidence,
            FrameTransition::Continue(input(next)),
            clock(),
            now(),
        )
        .unwrap();
    }
    let evidence = bind(&mut held, id, 3);
    assert_eq!(evidence.stage, RefinementStage::RecheckRepairedValueFresh);
    assert!(matches!(evidence.execution, StepExecution::Role(_)));
    let remaining = held.frame(id).unwrap().budget;
    held.resolve(
        evidence,
        FrameTransition::Continue(input(RefinementStage::CriticRecheck)),
        clock(),
        now(),
    )
    .unwrap();
    let frame = held.frame(id).unwrap();
    assert_eq!(frame.input.stage, RefinementStage::CriticRecheck);
    assert_eq!(frame.phase, FramePhase::Ready);
    assert_eq!(frame.evidence.len(), 3);
    assert_eq!(frame.budget, remaining);
    assert_eq!(held.repair_counts, vec![(first_move(), 1)]);
}

#[test]
fn queued_recheck_preserves_original_question_remaining_budget_and_repair_count() {
    for queue_after_value in [false, true] {
        let mut held = machine();
        let parent = held
            .admit(
                first_move(),
                input(RefinementStage::ProposerRanking),
                budget(),
                clock(),
                now(),
            )
            .unwrap();
        let id = held
            .enqueue_repair(
                parent,
                first_move(),
                input(RefinementStage::Repair),
                budget(),
                None,
                clock(),
                now(),
            )
            .unwrap();
        let repair = bind(&mut held, id, 1);
        held.resolve(
            repair,
            FrameTransition::Continue(input(RefinementStage::RepairValidation)),
            clock(),
            now(),
        )
        .unwrap();
        let mut sequence = 2;
        if queue_after_value {
            let validation = bind(&mut held, id, sequence);
            held.resolve(
                validation,
                FrameTransition::Continue(input(RefinementStage::RepairValueFresh)),
                clock(),
                now(),
            )
            .unwrap();
            sequence += 1;
        }
        let completion = bind(&mut held, id, sequence);
        let frame = held.frame(id).unwrap();
        assert_eq!(
            frame.input.stage,
            if queue_after_value {
                RefinementStage::RepairValueFresh
            } else {
                RefinementStage::RepairValidation
            }
        );
        let question = frame.input.question_digest;
        let remaining = frame.budget;
        let original_root = frame.input.view_seal.root_state;
        let context_policy = frame.input.view_seal.context_policy;
        let original_deadline = frame.original_deadline;
        let evidence_count = frame.evidence.len();
        let repair_counts = held.repair_counts.clone();
        let repair_questions = held.repair_questions.clone();
        assert!(remaining.role_submissions_remaining < budget().role_submissions_remaining);
        assert!(held.queued_rechecks.is_empty());
        let closed = held
            .resolve(
                completion,
                FrameTransition::Complete {
                    scope: FrameCompletionScope::QueuedRecheck,
                },
                clock(),
                now(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(closed.id, id);
        assert_eq!(closed.original_first_move, first_move());
        assert_eq!(
            closed.completion_scope,
            Some(FrameCompletionScope::QueuedRecheck)
        );
        assert_eq!(closed.unknown, None);
        assert_eq!(closed.evidence.len(), evidence_count + 1);
        assert_eq!(closed.evidence.last(), Some(&completion.completed_evidence));
        assert!(held.frame(id).is_none());
        assert!(held.frame(parent).is_some());
        assert_eq!(held.queued_rechecks.len(), 1);
        let queued = held.queued_rechecks.get(&(id, question)).unwrap();
        assert_eq!(queued.first_move, first_move());
        assert_eq!(queued.question, question);
        assert_eq!(queued.budget, remaining);
        assert_eq!(queued.root_state, original_root);
        assert_eq!(queued.context_policy, context_policy);
        assert_eq!(queued.original_deadline, original_deadline);
        assert_eq!(queued.queued_evidence, completion.completed_evidence);
        assert!(!queued.issued);
        assert!(!queued.producer_checked);
        assert!(queued.committed);
        assert_eq!(held.repair_counts, repair_counts);
        assert_eq!(held.repair_questions, repair_questions);
    }
}

#[test]
fn unrelated_queued_recheck_scope_rejects_without_changing_waiting_frame() {
    let mut held = machine();
    let id = cpu_frame(&mut held, 1);
    let completion = bind(&mut held, id, 3);
    assert_eq!(completion.stage, RefinementStage::CpuValidation);
    let frame = held.frame(id).unwrap();
    let evidence = frame.evidence.clone();
    let remaining = frame.budget;
    let question = frame.input.question_digest;
    let repair_counts = held.repair_counts.clone();
    assert!(held.queued_rechecks.is_empty());
    assert!(matches!(
        held.resolve(
            completion,
            FrameTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck,
            },
            clock(),
            now(),
        ),
        Err(CoordinatorError::InvalidTransition(_))
    ));
    let frame = held.frame(id).unwrap();
    assert_eq!(frame.input.stage, RefinementStage::CpuValidation);
    assert_eq!(frame.input.question_digest, question);
    assert_eq!(frame.phase, FramePhase::Waiting(completion.execution));
    assert_eq!(frame.evidence, evidence);
    assert_eq!(frame.budget, remaining);
    assert!(held.queued_rechecks.is_empty());
    assert_eq!(held.repair_counts, repair_counts);
}

#[test]
fn new_recheck_stages_append_without_reassigning_existing_ordinals() {
    assert_eq!(RefinementStage::FreshCounterEndpoint as u8, 16);
    assert_eq!(RefinementStage::RecheckCounterValidation as u8, 17);
    assert_eq!(RefinementStage::RecheckRepairedValueFresh as u8, 18);
}
