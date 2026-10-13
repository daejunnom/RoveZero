//! Authored Rules/view/control fixtures exercise the private queue boundary.
//! They do not attest CPU searches, physical NN completion or performance.
use super::*;
use crate::pals::coordinator::{CoordinatorError, RefinementMachine, StepEvidence, StepExecution};
use crate::pals::engine::adaptive_port::port_tests::{output, setup, work};
use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuJobId, CpuWorkerId, SliceGrant};
use rz_contracts::{
    ClockDomain, ExecutionId, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration,
};

type Port = AdaptiveEnginePort<LegalOrderRoleMock>;
type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;
type Machine = RefinementMachine<ReviewView, AdaptivePortQuestion>;

fn input(port: &mut Port, stage: RefinementStage) -> FrameInput<ReviewView, AdaptivePortQuestion> {
    work(port, "e2e4", stage, 1).input
}

fn bind(
    machine: &mut Machine,
    id: crate::pals::coordinator::FrameId,
    sequence: u64,
) -> (Work, StepEvidence) {
    let clock = rz_contracts::ClockDomain(id.authority.epoch);
    let ready = machine.ready_work(id, clock, MonotonicTick(0)).unwrap();
    let execution = if ready.input.stage.role().is_some() {
        let (ticket, _) = output(&ready, sequence);
        machine
            .bind_role(id, &ticket, clock, MonotonicTick(0))
            .unwrap();
        StepExecution::Role(ticket.id)
    } else {
        let job = CpuJobEnvelope {
            id: CpuJobId {
                authority: id.authority,
                sequence,
            },
            state_identity: ready.input.view_seal.root_state,
            question: ready.input.question_digest,
            root_restriction: ContractDigest([4; 32]),
            evaluation_namespace: ContractDigest([5; 32]),
            search_conditions: ContractDigest([6; 32]),
            total_native_node_credit: ready.budget.original_cpu_credit,
            hard_deadline: ready.original_deadline,
        };
        let grant = SliceGrant {
            job: job.id,
            execution: ExecutionId::new(id.authority.epoch, sequence),
            owner: CpuWorkerId(0),
            node_credit: 16,
            soft_lease: Duration::from_millis(8),
            control_sequence: sequence,
        };
        machine
            .bind_cpu(id, &job, &grant, clock, MonotonicTick(0))
            .unwrap();
        StepExecution::Cpu {
            job: job.id,
            execution: grant.execution,
        }
    };
    let work = machine.bound_work(id, execution).unwrap();
    let evidence = StepEvidence {
        frame: id,
        stage: work.input.stage,
        execution,
        submitted_question: work.input.question_digest,
        submitted_view: work.input.view_seal,
        completed_evidence: ContractDigest([sequence as u8; 32]),
    };
    (work, evidence)
}

fn setup_ready() -> (Port, Machine, crate::pals::coordinator::FrameId) {
    setup_ready_with_port(setup(Position::startpos(), 4))
}

fn setup_fixed_iterative() -> Port {
    let cpu = CpuEngine::new(crate::cpu::CpuConfig {
        tt_entries: 16,
        max_depth: 8,
        ..Default::default()
    })
    .unwrap();
    let engine = PalsEngine::new_with_cpu_and_refinement_policy(
        PalsConfig {
            line_plies: 4,
            ..Default::default()
        },
        LegalOrderRoleMock,
        cpu,
        PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
    )
    .unwrap()
    .with_selected_review_context(
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    let authority = SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(1),
        implementation: ContractDigest([7; 32]),
    };
    let limits = PalsLimits {
        deadline: Instant::now() + Duration::from_secs(20),
        max_rounds: 2,
        max_cpu_nodes: 2000,
        cpu_depth: 1,
    };
    let deadline = Deadline {
        clock: ClockDomain(authority.epoch),
        at: MonotonicTick(engine.tick_at(limits.deadline)),
    };
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    let mut port = Port::new(
        engine,
        Position::startpos(),
        authority,
        deadline,
        limits,
        issuer,
    )
    .unwrap();
    port.configure_root_fanout(AdaptiveRootFanoutPolicy::SequentialFixed)
        .unwrap();
    port
}

fn setup_ready_with_port(mut port: Port) -> (Port, Machine, crate::pals::coordinator::FrameId) {
    let clock = port.deadline.clock;
    let mut machine = Machine::new(port.authority, port.deadline, 512).unwrap();
    let root = work(&mut port, "e2e4", RefinementStage::ProposerRanking, 1);
    let parent = machine
        .admit(
            root.original_first_move,
            root.input,
            root.budget,
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let repair = input(&mut port, RefinementStage::Repair);
    let id = machine
        .enqueue_repair(
            parent,
            root.original_first_move,
            repair,
            root.budget,
            None,
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let (_, repair_evidence) = bind(&mut machine, id, 1);
    machine
        .resolve(
            repair_evidence,
            FrameTransition::Continue(input(&mut port, RefinementStage::RepairValidation)),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    (port, machine, id)
}

fn setup_waiting() -> (Port, Machine, Work, StepEvidence) {
    let (port, mut machine, id) = setup_ready();
    let (bound, evidence) = bind(&mut machine, id, 2);
    (port, machine, bound, evidence)
}

fn queued(port: &Port, work: &Work, evidence: ContractDigest) -> AdaptiveQueuedRecheck {
    AdaptiveQueuedRecheck {
        parent: work.frame,
        question: work.input.question_digest,
        first: work.original_first_move,
        budget: work.budget,
        deadline: work.original_deadline,
        original_view: work.input.view_seal,
        public_context: work.input.public_context,
        root: Arc::clone(&port.rules),
        evidence,
    }
}

fn deferred(port: &mut Port, original: &Work) -> AdaptiveDeferredRepair<AdaptivePortQuestion> {
    let first = BoardMove::from_uci("e2e4").unwrap();
    let repaired = vec![first, BoardMove::from_uci("e7e5").unwrap()];
    let node = port.replay_node(&repaired).unwrap();
    let input = port
        .input(
            node,
            RefinementStage::RecheckRepairedValueFresh,
            Some(first),
            repaired.clone(),
            repaired.clone(),
            None,
            Some(repaired.clone()),
            None,
            false,
            None,
            None,
        )
        .unwrap();
    AdaptiveDeferredRepair {
        parent: original.frame,
        original_question: original.input.question_digest,
        original_view: original.input.view_seal,
        original_public_context: original.input.public_context,
        first: original.original_first_move,
        input,
        budget: original.budget,
        deadline: original.original_deadline,
        root: Arc::clone(&port.rules),
        repaired,
    }
}

#[test]
fn sibling_queued_questions_survive_continue_and_close_with_independent_budget() {
    let (mut port, mut machine, first, first_evidence) = setup_waiting();
    let clock = port.deadline.clock;
    machine
        .register_checked_recheck_queue(
            queued(&port, &first, first_evidence.completed_evidence),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    machine
        .resolve(
            first_evidence,
            FrameTransition::Continue(input(&mut port, RefinementStage::CounterValidation)),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let (_, counter) = bind(&mut machine, first.frame, 3);
    machine
        .resolve(
            counter,
            FrameTransition::Continue(input(&mut port, RefinementStage::Repair)),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let (_, repair) = bind(&mut machine, first.frame, 4);
    machine
        .resolve(
            repair,
            FrameTransition::Continue(input(&mut port, RefinementStage::RepairValidation)),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let (second, second_evidence) = bind(&mut machine, first.frame, 5);
    assert_ne!(first.input.question_digest, second.input.question_digest);
    assert!(second.budget.role_submissions_remaining < first.budget.role_submissions_remaining);
    machine
        .register_checked_recheck_queue(
            queued(&port, &second, second_evidence.completed_evidence),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    machine
        .resolve(
            second_evidence,
            FrameTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck,
            },
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let first_id = machine
        .enqueue_checked_recheck(deferred(&mut port, &first), clock, MonotonicTick(0))
        .unwrap();
    let second_id = machine
        .enqueue_checked_recheck(deferred(&mut port, &second), clock, MonotonicTick(0))
        .unwrap();
    assert_ne!(first_id, second_id);
    assert_eq!(machine.frame(first_id).unwrap().budget, first.budget);
    assert_eq!(machine.frame(second_id).unwrap().budget, second.budget);
    assert_eq!(machine.frame(first_id).unwrap().parent, Some(first.frame));
    assert!(matches!(
        machine.enqueue_checked_recheck(deferred(&mut port, &first), clock, MonotonicTick(0)),
        Err(CoordinatorError::Duplicate)
    ));
}

#[test]
fn queue_registration_rejects_changed_original_scope_and_uncharged_budget() {
    let (port, mut machine, original, evidence) = setup_waiting();
    for changed in ["budget", "view", "public", "question", "first", "deadline"] {
        let mut proof = queued(&port, &original, evidence.completed_evidence);
        match changed {
            "budget" => proof.budget.role_submissions_remaining += 1,
            "view" => proof.original_view.view_digest = ContractDigest([88; 32]),
            "public" => proof.public_context = ContractDigest([88; 32]),
            "question" => proof.question = ContractDigest([88; 32]),
            "first" => {
                proof.first = rz_contracts::pals::Move16::encode(
                    rz_contracts::Move::try_from(BoardMove::from_uci("d2d4").unwrap()).unwrap(),
                )
            }
            _ => proof.deadline.at.0 += 1,
        }
        assert!(
            machine
                .register_checked_recheck_queue(proof, port.deadline.clock, MonotonicTick(0))
                .is_err(),
            "{changed}"
        );
        assert_eq!(
            machine.frame(original.frame).unwrap().budget,
            original.budget
        );
    }
    machine
        .register_checked_recheck_queue(
            queued(&port, &original, evidence.completed_evidence),
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    assert!(matches!(
        machine.register_checked_recheck_queue(
            queued(&port, &original, evidence.completed_evidence),
            port.deadline.clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::Duplicate)
    ));
}

#[test]
fn registered_queue_requires_exact_transition_evidence_and_cannot_close_as_unknown() {
    let (mut port, mut machine, original, evidence) = setup_waiting();
    let clock = port.deadline.clock;
    machine
        .register_checked_recheck_queue(
            queued(&port, &original, evidence.completed_evidence),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let mut forged = evidence;
    forged.completed_evidence = ContractDigest([89; 32]);
    assert!(
        machine
            .resolve(
                forged,
                FrameTransition::Complete {
                    scope: FrameCompletionScope::QueuedRecheck
                },
                clock,
                MonotonicTick(0)
            )
            .is_err()
    );
    assert!(
        machine
            .resolve(
                evidence,
                FrameTransition::Unknown { reason: "raw-only" },
                clock,
                MonotonicTick(0)
            )
            .is_err()
    );
    assert!(
        machine
            .enqueue_checked_recheck(deferred(&mut port, &original), clock, MonotonicTick(0))
            .is_err()
    );
    machine
        .resolve(
            evidence,
            FrameTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck,
            },
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    machine
        .enqueue_checked_recheck(deferred(&mut port, &original), clock, MonotonicTick(0))
        .unwrap();
}

#[test]
fn generic_queue_completion_does_not_mint_private_deferred_admission() {
    let (mut port, mut machine, original, evidence) = setup_waiting();
    let clock = port.deadline.clock;
    machine
        .resolve(
            evidence,
            FrameTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck,
            },
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    assert!(
        machine
            .enqueue_checked_recheck(deferred(&mut port, &original), clock, MonotonicTick(0))
            .is_err()
    );
}

#[test]
fn last_sibling_actual_queue_closure_preserves_prior_unknown_without_granting_generic_completion() {
    let (mut port, mut machine, id) = setup_ready_with_port(setup_fixed_iterative());
    let clock = port.deadline.clock;
    let (earlier, earlier_evidence) = bind(&mut machine, id, 2);
    let moves = |text: &str| {
        text.split_whitespace()
            .map(|movement| BoardMove::from_uci(movement).unwrap())
            .collect::<Vec<_>>()
    };
    let first = BoardMove::from_uci("e2e4").unwrap();
    let proposal = moves("e2e4 e7e5 g1f3 b8c6");
    let counter = moves("e2e4 c7c5 g1f3 b8c6");
    let repaired = moves("e2e4 c7c5 b1c3 b8c6");
    port.record_hot(RecordKind::Proposal, &proposal, None)
        .unwrap();
    assert!(
        port.enqueue_counter(PendingCounter {
            first,
            proposal: proposal.clone(),
            counter: counter.clone(),
            attack: 1,
            source: None,
            repair_direct: false,
        })
        .unwrap()
    );
    let earlier_transition = port
        .finish_branch(
            &earlier,
            AdaptivePortTransition::Unknown {
                reason: "earlier sibling unresolved",
            },
        )
        .unwrap();
    let earlier_step = port
        .materialize_step(AdaptivePortStep {
            evidence: earlier_evidence.completed_evidence,
            transition: earlier_transition,
            consumption_gate: None,
            feedback_gate: None,
            queue_question: None,
            fresh_pair: None,
            rules_pair: None,
        })
        .unwrap();
    assert!(matches!(
        earlier_step.transition,
        FrameTransition::Continue(_)
    ));
    machine
        .resolve(
            earlier_evidence,
            earlier_step.transition,
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let (_, counter_evidence) = bind(&mut machine, id, 3);
    let anchor = port.replay_node(&counter[..=1]).unwrap();
    let repair_input = port
        .input(
            anchor,
            RefinementStage::Repair,
            Some(first),
            counter[..=1].to_vec(),
            proposal.clone(),
            Some(counter.clone()),
            None,
            Some(1),
            false,
            None,
            None,
        )
        .unwrap();
    machine
        .resolve(
            counter_evidence,
            FrameTransition::Continue(repair_input),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let (_, repair_evidence) = bind(&mut machine, id, 4);
    let endpoint = port.replay_node(&repaired).unwrap();
    // The public Repair line and coordinator CPU completions are authored
    // contract fixtures. No physical CPU or neural completion is invented.
    port.record_hot(RecordKind::Repair, &repaired, None)
        .unwrap();
    let repair_validation = port
        .input(
            endpoint,
            RefinementStage::RepairValidation,
            Some(first),
            repaired.clone(),
            proposal,
            Some(counter.clone()),
            Some(repaired.clone()),
            Some(1),
            false,
            None,
            None,
        )
        .unwrap();
    machine
        .resolve(
            repair_evidence,
            FrameTransition::Continue(repair_validation),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let (last, last_evidence) = bind(&mut machine, id, 5);
    let complete = || AdaptivePortTransition::Complete {
        scope: FrameCompletionScope::QueuedRecheck,
    };
    assert!(matches!(
        port.finish_branch(&last, complete()).unwrap(),
        AdaptivePortTransition::Unknown {
            reason: "earlier sibling unresolved"
        }
    ));
    assert!(port.deferred_rechecks.is_empty());
    let blueprint = port
        .blueprint(
            endpoint,
            RefinementStage::RecheckRepairedValueFresh,
            Some(first),
            repaired.clone(),
            repaired.clone(),
            Some(counter),
            Some(repaired),
            Some(1),
            false,
            None,
            None,
        )
        .unwrap();
    let actual = port.defer_fixed_recheck(&last, blueprint).unwrap();
    assert!(matches!(
        port.finish_branch(
            &last,
            AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoAlternative
            }
        )
        .unwrap(),
        AdaptivePortTransition::Unknown { .. }
    ));
    for changed in [
        "frame", "question", "view", "public", "cpu", "roles", "store", "first", "deadline",
    ] {
        let mut mismatched = last.clone();
        match changed {
            "frame" => mismatched.frame.sequence += 1,
            "question" => mismatched.input.question_digest = ContractDigest([88; 32]),
            "view" => mismatched.input.view_seal.view_digest = ContractDigest([88; 32]),
            "public" => mismatched.input.public_context = ContractDigest([88; 32]),
            "cpu" => mismatched.budget.original_cpu_credit += 1,
            "roles" => mismatched.budget.role_submissions_remaining += 1,
            "store" => mismatched.budget.original_store_byte_limit += 1,
            "first" => {
                mismatched.original_first_move = rz_contracts::pals::Move16::encode(
                    rz_contracts::Move::try_from(BoardMove::from_uci("d2d4").unwrap()).unwrap(),
                )
            }
            _ => mismatched.original_deadline.at.0 += 1,
        }
        assert!(
            matches!(
                port.finish_branch(&mismatched, complete()).unwrap(),
                AdaptivePortTransition::Unknown { .. }
            ),
            "{changed}"
        );
    }
    let transition = port.finish_branch(&last, actual).unwrap();
    assert!(matches!(
        transition,
        AdaptivePortTransition::Complete {
            scope: FrameCompletionScope::QueuedRecheck
        }
    ));
    assert_eq!(
        port.branch_issues.get(&first),
        Some(&"earlier sibling unresolved")
    );
    assert!(matches!(
        port.branch_receipts()[0].outcome,
        AdaptiveBranchOutcome::Unknown("earlier sibling unresolved")
    ));
    assert_eq!(
        port.branch_receipts()[0].submitted_question,
        earlier.input.question_digest
    );
    let materialized = port
        .materialize_step(AdaptivePortStep {
            evidence: last_evidence.completed_evidence,
            transition,
            consumption_gate: None,
            feedback_gate: None,
            queue_question: Some((last.frame, last.input.question_digest)),
            fresh_pair: None,
            rules_pair: None,
        })
        .unwrap();
    let proof = materialized.queued_recheck.unwrap();
    proof.validate_work(&last).unwrap();
    machine
        .register_checked_recheck_queue(proof, clock, MonotonicTick(0))
        .unwrap();
    let closed = machine
        .resolve(
            last_evidence,
            materialized.transition,
            clock,
            MonotonicTick(0),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        closed.completion_scope,
        Some(FrameCompletionScope::QueuedRecheck)
    );
    assert_eq!(closed.unknown, None);
    assert!(matches!(
        port.finish_branch(&last, complete()).unwrap(),
        AdaptivePortTransition::Unknown { .. }
    ));
    assert_eq!(port.counters.repair_queue_enqueued, 1);
    assert_eq!(port.counters.cpu_nodes, 0);
    assert_eq!(port.counters.role_calls, 0);
}

#[test]
fn raw_cpu_unknown_has_no_question_queue_proof_or_materialization_authority() {
    let mut port = setup(Position::startpos(), 4);
    let submitted = work(&mut port, "e2e4", RefinementStage::RepairValidation, 1);
    let raw = port
        .after_cpu(&submitted, ObservationId(999), false)
        .unwrap();
    assert!(raw.queue_question.is_none());
    let materialized = port.materialize_step(raw).unwrap();
    assert!(materialized.queued_recheck.is_none());
    assert!(matches!(
        materialized.transition,
        FrameTransition::Unknown { .. }
    ));
    assert!(port.deferred_rechecks.is_empty());
}

#[test]
fn ready_rules_terminal_queue_commits_without_creating_cpu_or_role_execution() {
    let mut port = setup(
        Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap(),
        4,
    );
    let clock = port.deadline.clock;
    let mut machine = Machine::new(port.authority, port.deadline, 512).unwrap();
    let root = work(&mut port, "f7g7", RefinementStage::ProposerRanking, 1);
    let first = root.original_first_move;
    let budget = root.budget;
    let parent = machine
        .admit(first, root.input, budget, clock, MonotonicTick(0))
        .unwrap();
    let repair = work(&mut port, "f7g7", RefinementStage::Repair, 2).input;
    let id = machine
        .enqueue_repair(parent, first, repair, budget, None, clock, MonotonicTick(0))
        .unwrap();
    let (_, evidence) = bind(&mut machine, id, 1);
    let terminal = work(&mut port, "f7g7", RefinementStage::RepairValidation, 2).input;
    machine
        .resolve(
            evidence,
            FrameTransition::Continue(terminal),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let submitted = machine.ready_work(id, clock, MonotonicTick(0)).unwrap();
    let raw_evidence = ContractDigest([91; 32]);
    let usage = machine.history_usage();
    let endpoint = port
        .issue_rules_position(submitted.input.question.position.clone())
        .unwrap()
        .export()
        .unwrap();
    assert!(endpoint.terminal_wdl().is_some());
    machine
        .register_checked_recheck_queue(
            queued(&port, &submitted, raw_evidence),
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    assert!(
        machine
            .resolve_rules_terminal(
                id,
                &endpoint,
                ContractDigest([92; 32]),
                FrameTransition::Complete {
                    scope: FrameCompletionScope::QueuedRecheck
                },
                clock,
                MonotonicTick(0)
            )
            .is_err()
    );
    let closed = machine
        .resolve_rules_terminal(
            id,
            &endpoint,
            raw_evidence,
            FrameTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck,
            },
            clock,
            MonotonicTick(0),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        closed.completion_scope,
        Some(FrameCompletionScope::QueuedRecheck)
    );
    assert_eq!(machine.history_usage(), usage);
    assert_eq!(port.counters.cpu_nodes, 0);
    assert_eq!(port.counters.role_calls, 0);
}

#[test]
fn unsubmitted_ongoing_repair_cannot_claim_rules_terminal_queue() {
    let (port, mut machine, id) = setup_ready();
    // The actual Rules endpoint is ongoing, so an unsubmitted ready Repair
    // cannot substitute a Rules proof for a real worker completion.
    let clock = port.deadline.clock;
    let ready = machine.ready_work(id, clock, MonotonicTick(0)).unwrap();
    assert!(
        machine
            .register_checked_recheck_queue(
                queued(&port, &ready, ContractDigest([93; 32])),
                clock,
                MonotonicTick(0)
            )
            .is_err()
    );
}
