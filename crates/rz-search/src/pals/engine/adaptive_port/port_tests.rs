//! Contract fixtures exercise the actual semantic port. No physical native NN
//! completion or engine performance is inferred from these authored outputs.
use super::*;
use crate::pals::coordinator::{
    CoordinatorError, FrameId, FramePhase, NativeAcceptanceIssuer, PhysicalRoleCompletion,
    ProposerRankingPublished, RefinementMachine, RoleCompletion, RoleSubmission,
    RoleTicketRegistry,
};
use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use rz_contracts::pals::RepresentationKey;
use rz_contracts::{ClockDomain, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration};
use rz_runtime::adaptive::CoordinatorPolicy;

type Port = AdaptiveEnginePort<LegalOrderRoleMock>;
type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;
pub(super) fn setup(position: Position, line_plies: usize) -> Port {
    let cpu = CpuEngine::new(crate::cpu::CpuConfig {
        tt_entries: 16,
        max_depth: 8,
        ..Default::default()
    })
    .unwrap();
    let engine = PalsEngine::new(
        PalsConfig {
            line_plies,
            ..Default::default()
        },
        LegalOrderRoleMock,
        cpu,
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
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    Port::new(
        engine,
        position,
        authority,
        Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(20_000_000_000),
        },
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 2,
            max_cpu_nodes: 2000,
            cpu_depth: 1,
        },
        issuer,
    )
    .unwrap()
}
pub(super) fn work(port: &mut Port, first: &str, stage: RefinementStage, sequence: u64) -> Work {
    let first = BoardMove::from_uci(first).unwrap();
    let node = port.connect_hot(port.root, first).unwrap();
    if !matches!(
        purpose_for_stage(stage),
        RoleQueryPurpose::ProposePolicy | RoleQueryPurpose::ValueFresh
    ) {
        // These authored dependent questions review a Rules-checked proposal
        // already in public memory. This publication is not a CPU/NN receipt.
        port.record_hot(RecordKind::Proposal, &[first], None)
            .unwrap();
    }
    let input = port
        .input(
            node,
            stage,
            Some(first),
            vec![first],
            vec![first],
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    Work {
        frame: FrameId {
            authority: port.authority,
            sequence,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    }
}
pub(super) fn output(work: &Work, id: u64) -> (RoleTicketEnvelope, RoleOutput) {
    let purpose = work.input.question.context.purpose;
    let role = role_for_purpose(purpose);
    let key = RepresentationKey {
        input: ContractDigest(work.input.view.digest()),
        history: ContractDigest([2; 32]),
        candidates: ContractDigest([3; 32]),
        mask_positions: ContractDigest([4; 32]),
        model: ContractDigest([5; 32]),
        encoding: ContractDigest([6; 32]),
        precision: rz_contracts::PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: work.input.question.context.public_revision,
        role,
    };
    let ticket = RoleTicketEnvelope {
        id: RequestId::new(work.frame.authority.epoch, id),
        authority: work.frame.authority,
        question: work.input.question_digest,
        view: work.input.view_seal,
        representation: key,
        role,
        original_deadline: work.original_deadline,
    };
    let logits = vec![0.0; work.input.view.query().legal.len()];
    let wdl = rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap();
    let payload = match role {
        Role::Proposer => RolePayload::Proposal {
            candidate_logits: logits,
            wdl,
        },
        Role::Critic => RolePayload::Counterexample {
            candidate_logits: logits,
            divergence_logits: vec![0.0; work.input.view.query().divergences.len()],
            wdl,
        },
        Role::Validator => RolePayload::TaskRanking {
            task_logits: vec![0.0; 7],
        },
    };
    let output = RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key,
        payload,
        private_latent: vec![0.0; 16 * 384],
    };
    (ticket, output)
}

/// This CPU/mock owner crosses the real linear registry acceptance boundary.
/// Its authored output is not native neural execution or GPU acceptance proof.
fn accepted_root_ranking(
    port: &mut Port,
) -> (
    ProposerRankingPublished,
    RoleTicketRegistry<ReviewView, RoleOutput>,
) {
    let input = port.root_input().unwrap();
    assert_eq!(input.stage, RefinementStage::ProposerRanking);
    let mut ordered_candidates = Sha256::new();
    for movement in &input.view.query().legal {
        let encoded =
            rz_contracts::pals::Move16::encode(rz_contracts::Move::try_from(*movement).unwrap());
        ordered_candidates.update(encoded.bits().to_le_bytes());
    }
    let identity = port.engine.model.value_identity().unwrap();
    let key = RepresentationKey {
        input: ContractDigest(input.view.digest()),
        history: input.view_seal.root_state.semantic,
        candidates: ContractDigest(ordered_candidates.finalize().into()),
        mask_positions: ContractDigest(
            Sha256::digest(vec![1u8; input.view.query().legal.len()]).into(),
        ),
        model: ContractDigest(Sha256::digest(identity.model.as_bytes()).into()),
        encoding: ContractDigest(Sha256::digest(identity.encoding.as_bytes()).into()),
        precision: rz_contracts::PrecisionProfile::Fp32,
        frozen_epoch: 0,
        record_revision: input.question.context.public_revision,
        role: Role::Proposer,
    };
    let mut registry =
        RoleTicketRegistry::new(port.authority, 512, CoordinatorPolicy::Fixed, 1).unwrap();
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let ticket = registry
        .admit(
            RoleSubmission {
                question: input.question_digest,
                view: input.view_seal,
                representation: key,
                role: Role::Proposer,
                original_deadline: port.deadline,
            },
            Arc::clone(&input.view),
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    let dispatched = registry.take_dispatch().unwrap();
    assert_eq!(dispatched.ticket, ticket);
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(RoleOutput {
                id: ticket.id,
                authority: ticket.authority,
                key,
                payload: RolePayload::Proposal {
                    candidate_logits: (0..input.view.query().legal.len())
                        .map(|index| -(index as f32))
                        .collect(),
                    wdl: rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap(),
                },
                private_latent: vec![0.0; 16 * 384],
            }),
        })
        .unwrap();
    let delivered = registry
        .take_ready(port.deadline.clock, MonotonicTick(1))
        .unwrap();
    assert!(delivered.logically_eligible);
    assert_eq!(delivered.ticket, ticket);
    assert!(delivered.result.is_ok());
    let permit = registry
        .issue_native_acceptance(&ticket, port.deadline.clock, MonotonicTick(2))
        .unwrap();
    registry
        .confirm_native_acceptance(
            issuer.issue_success(permit),
            port.deadline.clock,
            MonotonicTick(2),
        )
        .unwrap();
    let ranking = registry
        .accept_proposer_ranking(
            &ticket,
            port.root_rules(),
            port.deadline.clock,
            MonotonicTick(3),
        )
        .unwrap();
    assert_eq!(ranking.source().ticket.representation, key);
    assert_eq!(ranking.source().ordered_candidate_digest, key.candidates);
    (ranking, registry)
}

#[test]
fn actual_one_ply_ranking_enters_cpu_without_another_policy_forward() {
    let mut port = setup(Position::startpos(), 1);
    let (ranking, mut registry) = accepted_root_ranking(&mut port);
    let mut candidates = port.ranking_published(&ranking).unwrap();
    assert_eq!(candidates.len(), port.engine.config.beam_width);
    let mut machine = RefinementMachine::new(port.authority, port.deadline, 512).unwrap();
    for candidate in &mut candidates {
        assert_eq!(candidate.input.stage, RefinementStage::CpuValidation);
        assert_eq!(candidate.input.view.query().prefix.len(), 1);
        assert_eq!(
            candidate.input.view.query().proposal,
            candidate.input.view.query().prefix
        );
        let proof = candidate
            .initialized
            .take()
            .expect("registered one-ply proof");
        proof.validate(&ranking).unwrap();
        let frame = machine
            .admit_checked_candidate(&ranking, proof, port.deadline.clock, MonotonicTick(4))
            .unwrap();
        port.root_candidate_admitted(frame, candidate.first_move, &candidate.input)
            .unwrap();
        let admitted = machine.frame(frame).unwrap();
        assert_eq!(admitted.phase, FramePhase::Ready);
        assert_eq!(admitted.input.stage, RefinementStage::CpuValidation);
        assert_eq!(admitted.ranking_source(), Some(ranking.source()));
    }
    assert_eq!(machine.history_usage().role_requests, 1);
    assert_eq!(machine.history_usage().cpu_executions, 0);
    assert_eq!(machine.history_usage().ranking_fanouts, candidates.len());
    assert_eq!(port.counters.role_calls, 1);
    assert_eq!(port.counters.proposer_calls, 1);
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert_eq!(port.counters.cpu_tasks_requested, 0);
    assert_eq!(registry.next_request_sequence(), 2);
    assert!(
        registry.take_dispatch().is_none(),
        "candidate fanout invented another forward"
    );
}

#[test]
fn longer_registered_proposals_keep_actual_policy_continuation_entry() {
    let mut port = setup(Position::startpos(), 3);
    let (ranking, mut registry) = accepted_root_ranking(&mut port);
    let candidates = port.ranking_published(&ranking).unwrap();
    let mut machine = RefinementMachine::new(port.authority, port.deadline, 512).unwrap();
    for candidate in candidates {
        assert!(candidate.initialized.is_none());
        assert_eq!(candidate.input.stage, RefinementStage::ProposalContinuation);
        assert!(candidate.input.view.query().proposal.is_empty());
        let frame = machine
            .admit_candidate(
                &ranking,
                candidate.first_move,
                candidate.input.clone(),
                candidate.budget,
                port.deadline.clock,
                MonotonicTick(4),
            )
            .unwrap();
        port.root_candidate_admitted(frame, candidate.first_move, &candidate.input)
            .unwrap();
        assert_eq!(
            machine.frame(frame).unwrap().input.stage,
            RefinementStage::ProposalContinuation
        );
    }
    assert_eq!(machine.history_usage().role_requests, 1);
    assert_eq!(port.counters.role_calls, 1);
    assert_eq!(port.counters.proposals, 0);
    assert_eq!(registry.next_request_sequence(), 2);
    assert!(registry.take_dispatch().is_none());
}

#[test]
fn same_actual_one_ply_source_and_first_move_cannot_be_admitted_twice() {
    let mut port = setup(Position::startpos(), 1);
    let (ranking, registry) = accepted_root_ranking(&mut port);
    let mut candidates = port.ranking_published(&ranking).unwrap();
    let candidate = candidates.first_mut().unwrap();
    let proof = candidate.initialized.take().unwrap();
    // Replay only the fields of the actual issued proof; no changed source,
    // view, profile or first move is manufactured for a positive assertion.
    let replay = CheckedRootCandidate {
        source: proof.source,
        first: proof.first,
        input: proof.input.clone(),
        budget: proof.budget,
        line_plies: proof.line_plies,
    };
    let mut machine = RefinementMachine::new(port.authority, port.deadline, 512).unwrap();
    machine
        .admit_checked_candidate(&ranking, proof, port.deadline.clock, MonotonicTick(4))
        .unwrap();
    let before = machine.history_usage();
    assert_eq!(
        machine.admit_checked_candidate(&ranking, replay, port.deadline.clock, MonotonicTick(4)),
        Err(CoordinatorError::Duplicate)
    );
    assert_eq!(machine.history_usage(), before);
    assert_eq!(machine.frames().count(), 1);
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert_eq!(registry.next_request_sequence(), 2);
    assert!(
        port.ranking_published(&ranking).is_err(),
        "accepted root ranking was consumed twice"
    );
    assert_eq!(port.counters.role_calls, 1);
}

#[test]
fn independently_pending_roles_keep_their_original_view_after_publication() {
    let mut port = setup(Position::startpos(), 2);
    let first = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
    let second = work(&mut port, "d2d4", RefinementStage::ProposalContinuation, 2);
    let second_digest = second.input.view.digest();
    let second_revision = second.input.question.context.public_revision;
    let (a, out_a) = output(&first, 1);
    let (b, out_b) = output(&second, 2);
    port.commit_role(&first, &a, &out_a, None).unwrap();
    port.role_consumed(&a, &first.input.view).unwrap();
    assert!(port.engine.revision > second_revision);
    port.validate_role(&second, &b, &out_b).unwrap();
    port.commit_role(&second, &b, &out_b, None).unwrap();
    port.role_consumed(&b, &second.input.view).unwrap();
    assert_eq!(second.input.view.digest(), second_digest);
    assert_eq!(
        second.input.question.context.public_revision,
        second_revision
    );
    assert_eq!(port.counters.consumed_role_outputs, 2);
    assert!(port.commit_role(&first, &a, &out_a, None).is_err());
}

#[test]
fn recorded_role_consumption_is_read_only_and_survives_later_progress_or_deadline() {
    let mut port = setup(Position::startpos(), 2);
    let proposal = vec![
        BoardMove::from_uci("e2e4").unwrap(),
        BoardMove::from_uci("e7e5").unwrap(),
    ];
    let counter = vec![proposal[0], BoardMove::from_uci("c7c5").unwrap()];
    port.record_hot(RecordKind::Proposal, &proposal, None)
        .unwrap();
    port.record_hot(RecordKind::Counterexample, &counter, None)
        .unwrap();
    let witness_revision = port.engine.revision;
    let bundle = port
        .engine
        .review_index
        .as_ref()
        .unwrap()
        .bundle_for_revision(witness_revision)
        .unwrap();
    let protected = |port: &Port| {
        port.engine
            .review_index
            .as_ref()
            .unwrap()
            .bundle(bundle)
            .unwrap()
            .protected_consumptions()
    };
    assert_eq!(protected(&port), 2);
    let input = port
        .input(
            port.root,
            RefinementStage::CriticDivergence,
            Some(proposal[0]),
            Vec::new(),
            proposal.clone(),
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    let original = Work {
        frame: FrameId {
            authority: port.authority,
            sequence: 1,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(proposal[0]).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    let (ticket, out) = output(&original, 1);
    assert!(
        !port
            .role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );
    port.commit_role(&original, &ticket, &out, None).unwrap();
    assert!(
        !port
            .role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );
    port.role_consumed(&ticket, &original.input.view).unwrap();
    assert_eq!(protected(&port), 1);
    assert!(
        port.role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );

    let mut wrong_question = ticket;
    wrong_question.question = ContractDigest([99; 32]);
    assert!(matches!(
        port.role_consumption_recorded(&wrong_question, &original.input.view),
        Err(PalsError::Role(RoleError::InvalidOutput))
    ));
    let other = work(&mut port, "d2d4", RefinementStage::ProposalContinuation, 2);
    assert!(matches!(
        port.role_consumption_recorded(&ticket, &other.input.view),
        Err(PalsError::Role(RoleError::InvalidOutput))
    ));
    let revision_before = port.engine.revision;
    let (other_ticket, other_out) = output(&other, 2);
    port.commit_role(&other, &other_ticket, &other_out, None)
        .unwrap();
    port.role_consumed(&other_ticket, &other.input.view)
        .unwrap();
    assert!(port.engine.revision > revision_before);
    assert!(
        port.role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );

    // An expired current search cannot erase historical recorded consumption.
    // This authored clock condition is not wall-clock performance evidence.
    port.limits.deadline = Instant::now();
    assert!(matches!(
        port.check_live(),
        Err(PalsError::Role(RoleError::Deadline))
    ));
    let counters = port.counters;
    let observations = port.engine.stores.observations.len();
    let pins = port.engine.stores.review_pin_count().unwrap();
    let remaining = protected(&port);
    let revision = port.engine.revision;
    for _ in 0..3 {
        assert!(
            port.role_consumption_recorded(&ticket, &original.input.view)
                .unwrap()
        );
        assert_eq!(protected(&port), remaining);
        assert_eq!(port.counters, counters);
        assert_eq!(port.engine.stores.observations.len(), observations);
        assert_eq!(port.engine.stores.review_pin_count().unwrap(), pins);
        assert_eq!(port.engine.revision, revision);
    }
}
#[test]
fn wrong_question_and_mutated_focus_are_rejected_before_commit() {
    let mut port = setup(Position::startpos(), 2);
    let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
    let (mut ticket, out) = output(&original, 1);
    ticket.question = ContractDigest([99; 32]);
    assert!(port.validate_role(&original, &ticket, &out).is_err());
    let (ticket, out) = output(&original, 1);
    let mut altered = Work {
        frame: original.frame,
        original_first_move: original.original_first_move,
        original_deadline: original.original_deadline,
        budget: original.budget,
        input: original.input.clone(),
    };
    Arc::make_mut(&mut altered.input.question)
        .context
        .focus_sha256 = [99; 32];
    assert!(port.validate_role(&altered, &ticket, &out).is_err());
    assert_eq!(port.counters.consumed_role_outputs, 0);
}
#[test]
fn fresh_counter_rejects_changed_namespace_before_any_hot_publication() {
    let mut port = setup(Position::startpos(), 2);
    let mut pending = work(&mut port, "e2e4", RefinementStage::FreshCounterEndpoint, 1);
    let (mut ticket, mut out) = output(&pending, 1);
    let original = FreshPairSeal::from_ticket(
        &ticket,
        pending.input.public_context,
        port.engine.nodes[port.root].position.side_to_move().into(),
    );
    let query = Arc::make_mut(&mut pending.input.question);
    query.paired_fresh_seal = Some(original);
    // This test exercises the pre-publication guard, not NN completion or the
    // existence of a comparison source observation.
    query.paired_repaired = Some(ObservationId(0));
    port.validate_role(&pending, &ticket, &out).unwrap();
    let observations = port.engine.stores.observations.len();
    for changed in ["model", "encoding", "epoch", "revision"] {
        ticket.representation = out.key;
        match changed {
            "model" => ticket.representation.model = ContractDigest([99; 32]),
            "encoding" => ticket.representation.encoding = ContractDigest([99; 32]),
            "epoch" => ticket.representation.frozen_epoch += 1,
            _ => ticket.representation.record_revision += 1,
        }
        let saved = out.key;
        out.key = ticket.representation;
        assert!(
            port.validate_role(&pending, &ticket, &out).is_err(),
            "{changed}"
        );
        out.key = saved;
    }
    assert_eq!(port.engine.stores.observations.len(), observations);
    assert_eq!(port.counters.consumed_role_outputs, 0);
}
#[test]
fn mate_endpoint_uses_actual_rules_without_cpu_or_nn_receipt() {
    let mut port = setup(
        Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap(),
        2,
    );
    let work = work(&mut port, "f7g7", RefinementStage::CpuValidation, 1);
    let terminal = port.short_circuit(&work).unwrap().unwrap();
    assert!(terminal.endpoint.terminal_wdl().is_some());
    assert!(matches!(
        terminal.checked.transition,
        AdaptivePortTransition::Complete {
            scope: FrameCompletionScope::NoDivergence
        }
    ));
    assert!(terminal.checked.fresh_pair.is_none());
    assert_eq!(port.counters.cpu_nodes, 0);
    assert_eq!(port.counters.role_calls, 0);
    assert_eq!(
        port.result(PalsCompletion::RoundLimit).unwrap().best_move,
        Some(BoardMove::from_uci("f7g7").unwrap())
    );
}
#[test]
fn distinct_cpu_and_critic_lines_remain_two_checked_branches() {
    let mut port = setup(Position::startpos(), 4);
    let first = BoardMove::from_uci("e2e4").unwrap();
    let original = BoardMove::from_uci("c7c5").unwrap();
    let cpu = BoardMove::from_uci("e7e5").unwrap();
    let critic = BoardMove::from_uci("e7e6").unwrap();
    let node = port.connect_hot(port.root, first).unwrap();
    // Coherent authored branch genealogy; no physical CPU/NN evidence is
    // attached to either line before this contract fixture submits its head.
    port.record_hot(RecordKind::Proposal, &[first, original], None)
        .unwrap();
    port.record_hot(RecordKind::Counterexample, &[first, cpu], None)
        .unwrap();
    let mut input = port
        .input(
            node,
            RefinementStage::CriticReply,
            Some(first),
            vec![first],
            vec![first, original],
            Some(vec![first, cpu]),
            None,
            Some(1),
            false,
            None,
            None,
        )
        .unwrap();
    Arc::make_mut(&mut input.question).counter_source = None; // Authored line fixture, not physical CPU evidence.
    let work = Work {
        frame: FrameId {
            authority: port.authority,
            sequence: 1,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    let (ticket, mut out) = output(&work, 1);
    if let RolePayload::Counterexample {
        candidate_logits, ..
    } = &mut out.payload
    {
        candidate_logits[work
            .input
            .view
            .query()
            .legal
            .iter()
            .position(|mv| *mv == critic)
            .unwrap()] = 1.0;
    }
    let first_step = port.commit_role(&work, &ticket, &out, None).unwrap();
    port.role_consumed(&ticket, &work.input.view).unwrap();
    let first_step = port.materialize_step(first_step).unwrap();
    let FrameTransition::Continue(cpu_input) = first_step.transition else {
        panic!("CPU line lost")
    };
    assert_eq!(cpu_input.question.context.prefix, vec![first, cpu]);
    assert_eq!(port.pending_counters[&first].len(), 1);
    let cpu_work = Work {
        frame: work.frame,
        original_first_move: work.original_first_move,
        original_deadline: work.original_deadline,
        budget: work.budget,
        input: cpu_input,
    };
    let next = port
        .finish_branch(
            &cpu_work,
            AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoAlternative,
            },
        )
        .unwrap();
    // This authored branch fixture checks sibling preservation, not a
    // fabricated CPU receipt. Selection uses the already consumed role cause.
    let next = port
        .materialize_step(AdaptivePortStep {
            evidence: first_step.evidence,
            transition: next,
            fresh_pair: None,
            rules_pair: None,
            consumption_gate: None,
            feedback_gate: None,
            queue_question: None,
        })
        .unwrap();
    let FrameTransition::Continue(critic_input) = next.transition else {
        panic!("critic line discarded")
    };
    assert_eq!(critic_input.question.context.prefix, vec![first, critic]);
    assert_eq!(critic_input.question.first, Some(first));
    assert_eq!(critic_input.question.deadline, port.limits.deadline);
}

#[test]
fn main_frames_finish_one_root_round_together() {
    let mut port = setup(Position::startpos(), 1);
    let first = work(&mut port, "e2e4", RefinementStage::CpuValidation, 1);
    let second = work(&mut port, "d2d4", RefinementStage::CpuValidation, 2);
    for pending in [&first, &second] {
        port.round_candidates.insert(
            pending.input.question_digest,
            (pending.input.question.first.unwrap(), None),
        );
        port.root_candidate_admitted(pending.frame, pending.original_first_move, &pending.input)
            .unwrap();
        assert!(
            port.root_candidate_admitted(
                pending.frame,
                pending.original_first_move,
                &pending.input
            )
            .is_err()
        );
    }
    let close = |pending: &Work| ClosedFrame {
        id: pending.frame,
        original_first_move: pending.original_first_move,
        evidence: vec![ContractDigest([33; 32])],
        unknown: None,
        completion_scope: Some(FrameCompletionScope::NoDivergence),
    };
    port.publish_closed(&close(&first)).unwrap();
    assert!(port.next_root_round_input().unwrap().is_none());
    assert_eq!(port.counters.rounds, 0);
    port.publish_closed(&close(&second)).unwrap();
    assert_eq!(port.counters.rounds, 0);
    let next = port.next_root_round_input().unwrap().unwrap();
    assert_eq!(next.stage, RefinementStage::ProposerRanking);
    assert_eq!(port.counters.rounds, 1);
    assert_eq!(port.root_round, 1);
    assert!(port.next_root_round_input().unwrap().is_none());
    assert_eq!(port.counters.rounds, 1);
}

#[test]
fn widening_frontier_preserves_one_unexamined_first_move() {
    let mut port = setup(Position::startpos(), 2);
    port.engine.config.beam_width = 2;
    let ranked: Vec<_> = ["e2e4", "d2d4", "g1f3", "c2c4"]
        .into_iter()
        .map(|text| BoardMove::from_uci(text).unwrap())
        .collect();
    port.connect_hot(port.root, ranked[0]).unwrap();
    port.connect_hot(port.root, ranked[1]).unwrap();
    assert_eq!(
        port.root_frontier(&ranked, 3).unwrap(),
        vec![ranked[0], ranked[2]]
    );
}

#[test]
fn divergence_aging_uses_the_registered_root_round() {
    let mut port = setup(Position::startpos(), 6);
    port.root_round = 1;
    let proposal: Vec<_> = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
        .into_iter()
        .map(|text| BoardMove::from_uci(text).unwrap())
        .collect();
    let first = proposal[0];
    port.record_hot(RecordKind::Proposal, &proposal, None)
        .unwrap();
    let input = port
        .input(
            port.root,
            RefinementStage::CriticDivergence,
            Some(first),
            vec![],
            proposal,
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    let work = Work {
        frame: FrameId {
            authority: port.authority,
            sequence: 1,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    let (ticket, mut out) = output(&work, 1);
    if let RolePayload::Counterexample {
        divergence_logits, ..
    } = &mut out.payload
    {
        *divergence_logits = vec![3.0, 2.0, 1.0];
    }
    let next = port.commit_role(&work, &ticket, &out, None).unwrap();
    port.role_consumed(&ticket, &work.input.view).unwrap();
    let next = port.materialize_step(next).unwrap();
    let FrameTransition::Continue(next) = next.transition else {
        panic!("no discovery")
    };
    assert_eq!(next.question.attack_ply, Some(3));
    assert_eq!(next.question.context.prefix.len(), 3);
}

#[test]
fn immediate_rules_terminal_scan_precedes_every_model_beam() {
    let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    let mates: Vec<_> = position
        .legal_moves()
        .into_iter()
        .filter(|movement| {
            let mut child = position.clone();
            child.make_move(*movement).unwrap();
            matches!(
                child.classify_position().unwrap().play_status,
                PlayStatus::Terminal {
                    reason: TerminalReason::Checkmate,
                    ..
                }
            )
        })
        .collect();
    assert!(!mates.is_empty());
    let mut port = setup(position, 4);
    port.engine.config.beam_width = 1;
    port.prepare_root_rules().unwrap();
    for movement in mates {
        assert!(
            port.engine.nodes[port.root]
                .edges
                .iter()
                .any(|edge| edge.movement == movement
                    && port.engine.nodes[edge.child].terminal.is_some())
        );
    }
    let result = port.result(PalsCompletion::RoundLimit).unwrap();
    let mut child = port.engine.nodes[port.root].position.clone();
    child.make_move(result.best_move.unwrap()).unwrap();
    assert!(matches!(
        child.classify_position().unwrap().play_status,
        PlayStatus::Terminal {
            reason: TerminalReason::Checkmate,
            ..
        }
    ));
    assert_eq!(port.counters.role_calls, 0);
    assert_eq!(port.counters.cpu_nodes, 0);
    let first_edges = port.counters.examined_edges;
    port.prepare_root_rules().unwrap();
    assert_eq!(port.counters.examined_edges, first_edges);
}
