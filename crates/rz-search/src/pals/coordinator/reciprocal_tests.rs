use super::*;
use crate::pals::coordinator::test_support::*;
use rz_contracts::rovent::{ChessPly, QuestionPurpose};

fn machine(mode: ExecutionMode) -> RefinementMachine<u64, RefinementStage> {
    let mut machine = RefinementMachine::new(authority(), deadline(), 512).unwrap();
    machine
        .enable_reciprocal_questions(mode, Digest([31; 32]))
        .unwrap();
    machine
}

fn parent(
    machine: &mut RefinementMachine<u64, RefinementStage>,
    stage: RefinementStage,
) -> FrameId {
    // Only the fixture enters a later existing stage directly. Production
    // frames reach it through the unchanged attested refinement transitions.
    machine
        .admit_internal(
            first_move(),
            input(stage),
            budget(),
            FrameLineage::default(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap()
}

fn question(
    machine: &RefinementMachine<u64, RefinementStage>,
    parent: FrameId,
    stage: RefinementStage,
    sequence: u64,
) -> QuestionEnvelope {
    QuestionEnvelope {
        id: ReciprocalQuestionId {
            authority: authority(),
            sequence,
        },
        parent: machine.reciprocal_question(parent).map(|held| held.id),
        sender: recipient(machine.frame(parent).unwrap().input.stage).unwrap(),
        recipient: recipient(stage).unwrap(),
        purpose: QuestionPurpose::TacticalReview,
        anchor: state(),
        anchor_root_ply: ChessPly(0),
        original_first_move: Some(first_move()),
        prefix_digest: Digest([32; 32]),
        line_digest: None,
        line_offset: ChessPly(0),
        question_digest: input(stage).question_digest,
        view: view(1),
        evidence_revision: 1,
        root_budget: Digest([31; 32]),
        original_deadline: deadline(),
    }
}

fn admit(
    machine: &mut RefinementMachine<u64, RefinementStage>,
    parent: FrameId,
    stage: RefinementStage,
    sequence: u64,
) -> FrameId {
    let question = question(machine, parent, stage, sequence);
    match machine
        .admit_reciprocal_question(
            parent,
            question,
            input(stage),
            4,
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap()
    {
        DependentFrameAdmission::Admitted(frame) | DependentFrameAdmission::Joined(frame) => frame,
    }
}

fn bind_role(
    machine: &mut RefinementMachine<u64, RefinementStage>,
    frame: FrameId,
    sequence: u64,
) -> StepEvidence {
    let stage = machine.frame(frame).unwrap().input.stage;
    let ticket = ticket(sequence, stage);
    machine
        .bind_role(frame, &ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    StepEvidence {
        frame,
        stage,
        execution: StepExecution::Role(ticket.id),
        submitted_question: ticket.question,
        submitted_view: ticket.view,
        completed_evidence: Digest([sequence as u8; 32]),
    }
}

fn cpu_fixture(stage: RefinementStage) -> (CpuJobEnvelope, SliceGrant) {
    let job = CpuJobEnvelope {
        id: CpuJobId {
            authority: authority(),
            sequence: 1,
        },
        state_identity: state(),
        question: input(stage).question_digest,
        root_restriction: Digest([4; 32]),
        evaluation_namespace: Digest([5; 32]),
        search_conditions: Digest([6; 32]),
        total_native_node_credit: budget().original_cpu_credit,
        hard_deadline: deadline(),
    };
    let grant = SliceGrant {
        job: job.id,
        execution: rz_contracts::ExecutionId::new(authority().epoch, 10),
        owner: CpuWorkerId(0),
        node_credit: 4,
        soft_lease: std::time::Duration::from_millis(8),
        control_sequence: 10,
    };
    (job, grant)
}

fn accepted_policy(
    stage: RefinementStage,
) -> (RoleTicketEnvelope, super::super::super::AcceptedRoleStep) {
    use crate::pals::coordinator::{
        NativeAcceptanceIssuer, PhysicalRoleCompletion, RoleCompletion, RoleSubmission,
        RoleTicketRegistry,
    };
    use rz_contracts::pals::{RoleOutput, RolePayload};
    let role = stage.role().unwrap();
    let mut registry = RoleTicketRegistry::<u64, RoleOutput>::new(
        authority(),
        512,
        rz_runtime::adaptive::CoordinatorPolicy::Fixed,
        1,
    )
    .unwrap();
    let ticket = registry
        .admit(
            RoleSubmission {
                question: input(stage).question_digest,
                view: view(1),
                representation: key(role),
                role,
                original_deadline: deadline(),
            },
            Arc::new(1),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    registry.take_dispatch().unwrap();
    let wdl = rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap();
    let payload = if role == Role::Proposer {
        RolePayload::Proposal {
            candidate_logits: vec![0.5],
            wdl,
        }
    } else {
        RolePayload::Counterexample {
            candidate_logits: vec![0.5],
            divergence_logits: Vec::new(),
            wdl,
        }
    };
    registry
        .complete_reciprocal_role(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(RoleOutput {
                id: ticket.id,
                authority: ticket.authority,
                key: ticket.representation,
                payload,
                private_latent: vec![0.0; 16 * 384],
            }),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let permit = registry
        .issue_reciprocal_native_acceptance(&ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    let receipt = issuer.issue_success(permit);
    // An ordinary root/V confirmation cannot consume reciprocal authority.
    let rejected = registry
        .confirm_native_acceptance(receipt, deadline().clock, MonotonicTick(0))
        .unwrap_err();
    assert!(matches!(
        rejected.error,
        CoordinatorError::IdentityMismatch(_)
    ));
    let accepted = registry
        .accept_reciprocal_native(rejected.receipt, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(registry.occupancy(), 0);
    (ticket, accepted)
}

#[test]
fn accepted_role_checkpoint_keeps_original_execution_until_child_returns() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let parent = parent(&mut machine, RefinementStage::ProposalContinuation);
    let (ticket, accepted) = accepted_policy(RefinementStage::ProposalContinuation);
    machine
        .bind_role(parent, &ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    let evidence = StepEvidence {
        frame: parent,
        stage: RefinementStage::ProposalContinuation,
        execution: StepExecution::Role(ticket.id),
        submitted_question: ticket.question,
        submitted_view: ticket.view,
        completed_evidence: accepted.output_digest(),
    };
    let question = question(&machine, parent, RefinementStage::CriticReply, 1);
    let admission = machine
        .admit_after_accepted_role(
            evidence,
            &accepted,
            question,
            input(RefinementStage::CriticReply),
            2,
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    let DependentFrameAdmission::Admitted(child) = admission else {
        panic!("new question unexpectedly joined")
    };
    assert_eq!(
        machine.frame(parent).unwrap().phase,
        FramePhase::Waiting(evidence.execution)
    );
    assert!(
        machine
            .ready_work(parent, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    let response = bind_role(&mut machine, child, 2);
    machine
        .resolve_reciprocal_response(response, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(machine.waiting_on(parent), None);
    assert_eq!(
        machine.frame(parent).unwrap().phase,
        FramePhase::Waiting(evidence.execution)
    );
}

#[test]
fn accepted_checkpoint_rejects_a_different_completed_head_without_spending_credit() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let parent = parent(&mut machine, RefinementStage::ProposalContinuation);
    let (ticket, accepted) = accepted_policy(RefinementStage::ProposalContinuation);
    machine
        .bind_role(parent, &ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    let evidence = StepEvidence {
        frame: parent,
        stage: RefinementStage::ProposalContinuation,
        execution: StepExecution::Role(ticket.id),
        submitted_question: ticket.question,
        submitted_view: ticket.view,
        completed_evidence: Digest([99; 32]),
    };
    let question = question(&machine, parent, RefinementStage::CriticReply, 1);
    assert!(matches!(
        machine.admit_after_accepted_role(
            evidence,
            &accepted,
            question,
            input(RefinementStage::CriticReply),
            2,
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::IdentityMismatch(_))
    ));
    assert_eq!(machine.waiting_on(parent), None);
    assert_eq!(
        machine
            .frame(parent)
            .unwrap()
            .budget
            .role_submissions_remaining,
        budget().role_submissions_remaining - 1
    );
}

#[test]
fn stage_change_registers_a_new_parent_without_reusing_the_old_descriptor() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let frame = parent(&mut machine, RefinementStage::CpuValidation);
    let mut first = question(&machine, frame, RefinementStage::CpuValidation, 1);
    first.sender = SearchRecipient::Proposer;
    machine
        .register_reciprocal_parent(frame, first, deadline().clock, MonotonicTick(0))
        .unwrap();
    // A fixture advances only the semantic stage; production uses checked commit.
    machine.match_frame_mut(frame).unwrap().input = input(RefinementStage::CpuDiscovery);
    assert!(machine.reciprocal_question(frame).is_none());
    let mut next = question(&machine, frame, RefinementStage::CpuDiscovery, 2);
    next.sender = SearchRecipient::Critic;
    machine
        .register_reciprocal_parent(frame, next, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(machine.reciprocal_question(frame), Some(next));
    let mut stale_child = question(&machine, frame, RefinementStage::ProposalContinuation, 3);
    stale_child.parent = Some(first.id);
    assert!(matches!(
        machine.admit_reciprocal_question(
            frame,
            stale_child,
            input(RefinementStage::ProposalContinuation),
            1,
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::IdentityMismatch(_))
    ));
    assert_eq!(machine.waiting_on(frame), None);
}

#[test]
fn a_new_view_issuance_does_not_reopen_completed_identical_semantics() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let parent = parent(&mut machine, RefinementStage::ProposalContinuation);
    let child = admit(&mut machine, parent, RefinementStage::CriticReply, 1);
    let response = bind_role(&mut machine, child, 1);
    machine
        .resolve_reciprocal_response(response, deadline().clock, MonotonicTick(0))
        .unwrap();
    let mut next = question(&machine, parent, RefinementStage::CriticReply, 2);
    let mut changed = input(RefinementStage::CriticReply);
    next.view = view(8);
    changed.view_seal = next.view;
    next.anchor.owner = rz_contracts::OwnerId(88);
    assert_eq!(
        machine.admit_reciprocal_question(
            parent,
            next,
            changed,
            1,
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::InvalidTransition(
            "unchanged reciprocal question made no new progress"
        ))
    );
    assert_eq!(machine.waiting_on(parent), None);
}

#[test]
fn all_six_directions_use_real_frames_and_keep_original_root_limits() {
    let stages = [
        RefinementStage::ProposalContinuation,
        RefinementStage::CriticDivergence,
        RefinementStage::CpuValidation,
    ];
    for from in stages {
        for to in stages {
            if from == to {
                continue;
            }
            let mut machine = machine(ExecutionMode::CpuPcr);
            let parent = parent(&mut machine, from);
            let child = admit(&mut machine, parent, to, 1);
            let waiting = machine.frame(parent).unwrap();
            assert_eq!(machine.waiting_on(parent), Some(child));
            assert_eq!(
                waiting.budget.role_submissions_remaining,
                budget().role_submissions_remaining - 4
            );
            let child = machine
                .ready_work(child, deadline().clock, MonotonicTick(0))
                .unwrap();
            assert_eq!(child.input.stage, to);
            assert_eq!(child.original_deadline, deadline());
            assert_eq!(
                child.budget.original_cpu_credit,
                budget().original_cpu_credit
            );
            assert_eq!(
                child.budget.original_store_byte_limit,
                budget().original_store_byte_limit
            );
            assert!(matches!(
                machine.ready_work(parent, deadline().clock, MonotonicTick(0)),
                Err(CoordinatorError::InvalidTransition(_))
            ));
        }
    }
}

#[test]
fn joined_question_runs_once_and_refunds_only_the_allocating_parent() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let first = parent(&mut machine, RefinementStage::ProposerRanking);
    let second = parent(&mut machine, RefinementStage::ProposerRanking);
    let child = admit(&mut machine, first, RefinementStage::CriticDivergence, 1);
    let joined = question(&machine, second, RefinementStage::CriticDivergence, 2);
    assert_eq!(
        machine.admit_reciprocal_question(
            second,
            joined,
            input(RefinementStage::CriticDivergence),
            4,
            deadline().clock,
            MonotonicTick(0)
        ),
        Ok(DependentFrameAdmission::Joined(child))
    );
    assert_eq!(machine.reciprocal_frame(joined.id), Some(child));
    assert_eq!(machine.reciprocal_question(child).unwrap().id.sequence, 1);
    assert_eq!(machine.frame(second).unwrap().budget, budget());
    let evidence = bind_role(&mut machine, child, 51);
    let response = machine
        .resolve_reciprocal_response(evidence, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(response.evidence, evidence.completed_evidence);
    assert_eq!(machine.waiting_on(first), None);
    assert_eq!(machine.waiting_on(second), None);
    assert_eq!(
        machine
            .frame(first)
            .unwrap()
            .budget
            .role_submissions_remaining,
        127
    );
    assert_eq!(
        machine
            .frame(second)
            .unwrap()
            .budget
            .role_submissions_remaining,
        128
    );
    assert!(matches!(
        machine.resolve_reciprocal_response(evidence, deadline().clock, MonotonicTick(0)),
        Err(CoordinatorError::InvalidTransition(_))
    ));
    let unchanged = question(&machine, first, RefinementStage::CriticDivergence, 3);
    assert!(matches!(
        machine.admit_reciprocal_question(
            first,
            unchanged,
            input(RefinementStage::CriticDivergence),
            4,
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::InvalidTransition(
            "unchanged reciprocal question made no new progress"
        ))
    ));
    assert_eq!(
        machine
            .frame(first)
            .unwrap()
            .budget
            .role_submissions_remaining,
        127
    );
}

#[test]
fn descendant_cannot_request_its_own_ancestor_or_resume_a_waiting_owner() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let parent = parent(&mut machine, RefinementStage::ProposerRanking);
    let child = admit(&mut machine, parent, RefinementStage::CriticDivergence, 1);
    let cycle = question(&machine, child, RefinementStage::ProposerRanking, 2);
    assert!(matches!(
        machine.admit_reciprocal_question(
            child,
            cycle,
            input(RefinementStage::ProposerRanking),
            1,
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::InvalidTransition(
            "reciprocal question would wait on its own ancestor"
        ))
    ));
    assert_eq!(machine.waiting_on(parent), Some(child));
    assert_eq!(machine.waiting_on(child), None);
    assert!(matches!(
        machine.make_resumed_ready(parent),
        Err(CoordinatorError::InvalidTransition(
            "frame awaits an independent reciprocal response"
        ))
    ));
}

#[test]
fn yielded_r_keeps_its_original_job_and_owner_across_a_p_response() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let parent = parent(&mut machine, RefinementStage::CpuValidation);
    let (job, grant) = cpu_fixture(RefinementStage::CpuValidation);
    machine
        .bind_cpu(parent, &job, &grant, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert!(
        machine
            .observe_yield(
                parent,
                StepExecution::Cpu {
                    job: job.id,
                    execution: grant.execution
                },
                None,
                1,
                false
            )
            .unwrap()
    );
    let child = admit(
        &mut machine,
        parent,
        RefinementStage::ProposalContinuation,
        1,
    );
    assert!(machine.make_resumed_ready(parent).is_err());
    let evidence = bind_role(&mut machine, child, 52);
    machine
        .resolve_reciprocal_response(evidence, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(machine.frame(parent).unwrap().phase, FramePhase::Suspended);
    assert_eq!(
        machine.frame(parent).unwrap().cpu_owner,
        Some((job.id, CpuWorkerId(0)))
    );
    machine.make_resumed_ready(parent).unwrap();
    let mut next = grant;
    next.execution.sequence = 11;
    next.control_sequence = 11;
    next.owner = CpuWorkerId(1);
    assert!(matches!(
        machine.bind_cpu(parent, &job, &next, deadline().clock, MonotonicTick(0)),
        Err(CoordinatorError::IdentityMismatch(_))
    ));
    next.owner = CpuWorkerId(0);
    machine
        .bind_cpu(parent, &job, &next, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(
        machine.frame(parent).unwrap().phase,
        FramePhase::Waiting(StepExecution::Cpu {
            job: job.id,
            execution: next.execution
        })
    );
}

#[test]
fn generic_unknown_cannot_hide_a_dependent_r_owner_from_shutdown() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let parent = parent(&mut machine, RefinementStage::ProposerRanking);
    let child = admit(&mut machine, parent, RefinementStage::CpuValidation, 1);
    let (job, grant) = cpu_fixture(RefinementStage::CpuValidation);
    machine
        .bind_cpu(child, &job, &grant, deadline().clock, MonotonicTick(0))
        .unwrap();
    let evidence = StepEvidence {
        frame: child,
        stage: RefinementStage::CpuValidation,
        execution: StepExecution::Cpu {
            job: job.id,
            execution: grant.execution,
        },
        submitted_question: job.question,
        submitted_view: view(1),
        completed_evidence: Digest([88; 32]),
    };
    assert_eq!(
        machine
            .resolve(
                evidence,
                FrameTransition::Unknown {
                    reason: "missing worker report"
                },
                deadline().clock,
                MonotonicTick(0)
            )
            .unwrap_err(),
        CoordinatorError::MissingEvidence
    );
    assert_eq!(machine.waiting_on(parent), Some(child));
    assert_eq!(
        machine.frame(child).unwrap().cpu_owner,
        Some((job.id, grant.owner))
    );
    assert_eq!(
        machine
            .frame(parent)
            .unwrap()
            .budget
            .role_submissions_remaining,
        124
    );
    assert_eq!(machine.revoke(), vec![evidence.execution]);
}

#[test]
fn actual_parent_cancel_detaches_its_wait_without_dropping_a_joined_child() {
    for joined in [false, true] {
        let mut machine = machine(ExecutionMode::CpuPcr);
        let parent = parent(&mut machine, RefinementStage::CpuValidation);
        let (job, grant) = cpu_fixture(RefinementStage::CpuValidation);
        machine
            .bind_cpu(parent, &job, &grant, deadline().clock, MonotonicTick(0))
            .unwrap();
        machine
            .observe_yield(
                parent,
                StepExecution::Cpu {
                    job: job.id,
                    execution: grant.execution,
                },
                None,
                1,
                false,
            )
            .unwrap();
        let child = admit(
            &mut machine,
            parent,
            RefinementStage::ProposalContinuation,
            1,
        );
        let other = joined.then(|| {
            let other = super::tests::parent(&mut machine, RefinementStage::CpuValidation);
            assert_eq!(
                admit(
                    &mut machine,
                    other,
                    RefinementStage::ProposalContinuation,
                    2
                ),
                child
            );
            other
        });
        let terminal = crate::cpu_worker::CpuJobTermination {
            job: job.id,
            owner: grant.owner,
            last_execution: grant.execution,
            completion: crate::cpu::CpuCompletion::Canceled,
            controls: rz_runtime::adaptive::ControlSnapshot {
                cancel_sequence: 11,
                cancel_ack: 11,
                terminal: true,
                ..Default::default()
            },
            retained_bytes: 0,
            shutdown: false,
        };
        let closed = machine
            .retire_suspended_cpu(parent, &terminal, Digest([89; 32]), "parent cancelled")
            .unwrap();
        assert_eq!(closed.unknown, Some("parent cancelled"));
        assert_eq!(machine.waiting_on(parent), None);
        assert!(machine.frame(child).is_some());
        if let Some(other) = other {
            assert_eq!(machine.waiting_on(other), Some(child));
            assert_eq!(machine.orphaned_reciprocal_frames().count(), 0);
        } else {
            assert_eq!(
                machine.orphaned_reciprocal_frames().collect::<Vec<_>>(),
                vec![child]
            );
        }
    }
}

#[test]
fn r_free_mode_foreign_budget_and_running_parent_are_rejected_without_transfer() {
    let mut machine = machine(ExecutionMode::CpuPc);
    let parent = parent(&mut machine, RefinementStage::ProposerRanking);
    let to_r = question(&machine, parent, RefinementStage::CpuValidation, 1);
    assert!(matches!(
        machine.admit_reciprocal_question(
            parent,
            to_r,
            input(RefinementStage::CpuValidation),
            4,
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::Contract(rz_contracts::ContractError {
            code: rz_contracts::ErrorCode::UnsupportedContract,
            ..
        }))
    ));
    let mut to_c = question(&machine, parent, RefinementStage::CriticDivergence, 1);
    to_c.root_budget = Digest([99; 32]);
    assert!(matches!(
        machine.admit_reciprocal_question(
            parent,
            to_c,
            input(RefinementStage::CriticDivergence),
            4,
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::IdentityMismatch(_))
    ));
    to_c.root_budget = Digest([31; 32]);
    let _ = bind_role(&mut machine, parent, 41);
    assert!(matches!(
        machine.admit_reciprocal_question(
            parent,
            to_c,
            input(RefinementStage::CriticDivergence),
            4,
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::InvalidTransition(
            "dependent admission requires a stable parent checkpoint"
        ))
    ));
    assert_eq!(
        machine
            .frame(parent)
            .unwrap()
            .budget
            .role_submissions_remaining,
        127
    );
    assert_eq!(machine.frames().count(), 1);
}

#[test]
fn failure_releases_waiters_as_unknown_and_revoke_never_restores_authority() {
    let mut machine = machine(ExecutionMode::CpuPcr);
    let parent = parent(&mut machine, RefinementStage::ProposerRanking);
    let child = admit(&mut machine, parent, RefinementStage::CriticDivergence, 1);
    let evidence = bind_role(&mut machine, child, 61);
    // A dependent answer cannot claim completion of a legacy PALS workflow.
    assert!(matches!(
        machine.resolve(
            evidence,
            FrameTransition::Complete {
                scope: FrameCompletionScope::NoDivergence
            },
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::InvalidTransition(_))
    ));
    let closed = machine
        .resolve(
            evidence,
            FrameTransition::Unknown {
                reason: "worker failure",
            },
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap()
        .unwrap();
    assert_eq!(closed.unknown, Some("worker failure"));
    assert_eq!(closed.completion_scope, None);
    assert_eq!(machine.waiting_on(parent), None);
    assert_eq!(
        machine
            .frame(parent)
            .unwrap()
            .budget
            .role_submissions_remaining,
        127
    );
    machine.revoke();
    assert!(matches!(
        machine.ready_work(parent, deadline().clock, MonotonicTick(0)),
        Err(CoordinatorError::AdmissionClosed)
    ));
    machine.retired_after_worker_join();
    assert_eq!(
        machine.reciprocal_frame(ReciprocalQuestionId {
            authority: authority(),
            sequence: 1
        }),
        None
    );
}
