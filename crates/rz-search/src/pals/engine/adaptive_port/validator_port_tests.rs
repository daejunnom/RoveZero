//! Authored V heads cross the actual registry/Search boundary. The private
//! CPU case executes a bounded native worker; no model/optimizer/GPU is used.
use super::*;
use crate::contract_time::{ContractClock, InstantClock};
use crate::cpu::{CpuConfig, CpuResumePolicy};
use crate::cpu_worker::{CpuCreditLedger, CpuWorker, CpuWorkerEvent, cpu_root_restriction_digest};
use crate::pals::coordinator::{
    NativeAcceptanceIssuer, PhysicalRoleCompletion, RefinementMachine, RoleCompletion,
    RoleSubmission, RoleTicketRegistry, StepEvidence, ValidatorActionAdmission,
};
use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use rz_contracts::pals::RepresentationKey;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, SliceGrant};
use rz_contracts::{ClockDomain, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration};
use rz_runtime::adaptive::CoordinatorPolicy;

type Port = AdaptiveEnginePort<LegalOrderRoleMock>;
type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;

fn cpu() -> CpuEngine {
    CpuEngine::with_resume_policy(
        CpuConfig {
            tt_entries: 16,
            max_depth: 4,
            quiescence_ply: 2,
            ..Default::default()
        },
        CpuResumePolicy::PausedStack,
    )
    .unwrap()
}
fn setup() -> Port {
    let engine = PalsEngine::new(
        PalsConfig {
            line_plies: 4,
            beam_width: 1,
            ..Default::default()
        },
        LegalOrderRoleMock,
        cpu(),
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
        Position::startpos(),
        authority,
        Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(20_000_000_000),
        },
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 2,
            max_cpu_nodes: 8192,
            cpu_depth: 1,
        },
        issuer,
    )
    .unwrap()
}
fn budget() -> FrameBudget {
    FrameBudget {
        role_submissions_remaining: 20,
        original_cpu_credit: 8192,
        original_store_byte_limit: 8 * 1024 * 1024,
    }
}
fn target(port: &mut Port, stage: RefinementStage, frame: u64) -> Work {
    let first = BoardMove::from_uci("e2e4").unwrap();
    let response = BoardMove::from_uci("e7e5").unwrap();
    let node = port.connect_hot(port.root, first).unwrap();
    // The authored parent and its dependent V/P/C actions share an existing
    // Rules-checked proposal bundle. It carries no physical model/CPU receipt.
    port.record_hot(RecordKind::Proposal, &[first, response], None)
        .unwrap();
    let input = port
        .input(
            node,
            stage,
            Some(first),
            vec![first],
            vec![first, response],
            None,
            None,
            Some(1),
            false,
            None,
            None,
        )
        .unwrap();
    Work {
        frame: FrameId {
            authority: port.authority,
            sequence: frame,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: budget(),
        input,
    }
}
fn accepted(
    port: &Port,
    question: &AdaptiveValidatorQuestion,
    sequence: u64,
    logits: [f32; 7],
) -> AcceptedValidatorPublished {
    let key = RepresentationKey {
        input: ContractDigest(question.input.view.digest()),
        history: question.input.view_seal.root_state.semantic,
        candidates: ContractDigest([1; 32]),
        mask_positions: ContractDigest([2; 32]),
        model: ContractDigest([3; 32]),
        encoding: ContractDigest([4; 32]),
        precision: rz_contracts::PrecisionProfile::Fp32,
        frozen_epoch: 0,
        record_revision: question.input.question.context.public_revision,
        role: Role::Validator,
    };
    let mut registry =
        RoleTicketRegistry::new(port.authority, 512, CoordinatorPolicy::Fixed, sequence).unwrap();
    // An authored CPU/mock owner capability crosses the actual linear
    // acceptance boundary. It is not real NN/GPU backend acceptance evidence.
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let ticket = registry
        .admit(
            RoleSubmission {
                question: question.input.question_digest,
                view: question.input.view_seal,
                representation: key,
                role: Role::Validator,
                original_deadline: port.deadline,
            },
            Arc::clone(&question.input.view),
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    registry.take_dispatch().unwrap();
    registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(RoleOutput {
                id: ticket.id,
                authority: ticket.authority,
                key,
                payload: RolePayload::TaskRanking {
                    task_logits: logits.to_vec(),
                },
                private_latent: vec![0.; 16 * 384],
            }),
        })
        .unwrap();
    registry
        .take_ready(port.deadline.clock, MonotonicTick(1))
        .unwrap();
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
    registry
        .accept_validator_task(&ticket, port.deadline.clock, MonotonicTick(2))
        .unwrap()
}

#[test]
fn actual_support_masks_unsupported_logits_and_preserves_exact_response_restrictions() {
    let mut port = setup();
    let original = target(&mut port, RefinementStage::CriticReply, 1);
    let q = port
        .prepare_validator_question(&original, budget(), &[], CpuEndpointSelection::collection())
        .unwrap();
    assert_eq!(
        q.support_mask(),
        [true, false, true, false, false, false, true]
    );
    assert_eq!(q.input.view.query().purpose, RoleQueryPurpose::TaskPolicy);
    assert!(q.input.view.query().legal.is_empty());
    assert_eq!(
        q.unsupported_reason(AdaptiveValidatorAction::LowerSelectivity),
        Some(ValidatorUnsupported::NoSelectivityControl)
    );
    let v = accepted(&port, &q, 1, [2., 1., 3., 100., 99., 98., 0.]);
    let plan = port.select_validator_action(&q, &v).unwrap();
    assert_eq!(plan.action(), AdaptiveValidatorAction::WidenResponses);
    assert_eq!(
        plan.selection_provenance(),
        ValidatorActionSelection::ValidatorArgmax
    );
    assert_eq!(plan.accepted_task_logits(), v.task_logits());
    assert_eq!(plan.parent_question(), original.input.question_digest);
    let AdaptiveValidatorEffect::Frame { input, .. } = plan.effect() else {
        panic!("widen must issue an actual CPU question")
    };
    let roots = input.question.validator_root_moves.as_ref().unwrap();
    assert!(!roots.contains(&BoardMove::from_uci("e7e5").unwrap()));
    assert_eq!(roots.len(), input.question.position.legal_moves().len() - 1);
    assert!(
        roots
            .iter()
            .all(|m| input.question.position.legal_moves().contains(m))
    );
    assert!(
        port.plan_validator_action(&q, &v, AdaptiveValidatorAction::WidenResponses)
            .is_err()
    );
    assert!(
        port.plan_validator_action(&q, &v, AdaptiveValidatorAction::LowerSelectivity)
            .is_err()
    );
    // The explicit comparison collector can select another supported action
    // once from the same original V head without charging V a second time.
    let defend = port
        .plan_validator_action(&q, &v, AdaptiveValidatorAction::DefendResponse)
        .unwrap();
    let AdaptiveValidatorEffect::Frame { input, .. } = defend.effect() else {
        panic!("defend")
    };
    assert_eq!(
        input.question.validator_root_moves.as_ref().unwrap(),
        &[BoardMove::from_uci("e7e5").unwrap()]
    );
    assert_eq!(port.counters.consumed_role_outputs, 1);
}

#[test]
fn deployment_and_foreign_exact_resume_have_no_validator_authority() {
    let mut port = setup();
    let original = target(&mut port, RefinementStage::CriticReply, 1);
    assert!(
        port.prepare_validator_question(
            &original,
            budget(),
            &[],
            CpuEndpointSelection::deployment()
        )
        .is_err()
    );
    let foreign = CpuEndpointSelection {
        cpu_r: CpuEndpoint::ExternalUci,
        ..CpuEndpointSelection::collection()
    };
    let q = port
        .prepare_validator_question(&original, budget(), &[], foreign)
        .unwrap();
    assert!(!q.support_mask()[AdaptiveValidatorAction::ResumeTask.index()]);
    let v = accepted(&port, &q, 1, [0., 0., 0., 0., 100., 0., 1.]);
    assert!(
        port.plan_validator_action(&q, &v, AdaptiveValidatorAction::ResumeTask)
            .is_err()
    );
}

#[test]
fn attack_repair_uses_the_exact_repaired_opponent_anchor_and_frozen_input() {
    let mut port = setup();
    let moves = |texts: &[&str]| {
        texts
            .iter()
            .map(|s| BoardMove::from_uci(s).unwrap())
            .collect::<Vec<_>>()
    };
    let proposal = moves(&["e2e4", "e7e5", "g1f3", "b8c6"]);
    let counter = moves(&["e2e4", "c7c5", "g1f3", "b8c6"]);
    let repaired = moves(&["e2e4", "c7c5", "f1c4", "b8c6"]);
    // Preserve the exact Proposal -> Counterexample -> Repair relation that
    // the dependent recheck reviews, without claiming native completion.
    port.record_hot(RecordKind::Proposal, &proposal, None)
        .unwrap();
    port.record_hot(RecordKind::Counterexample, &counter, None)
        .unwrap();
    port.record_hot(RecordKind::Repair, &repaired, None)
        .unwrap();
    let prefix = repaired[..3].to_vec();
    let node = port.replay_node(&prefix).unwrap();
    let input = port
        .input(
            node,
            RefinementStage::CriticRecheck,
            Some(proposal[0]),
            prefix.clone(),
            proposal.clone(),
            Some(counter.clone()),
            Some(repaired.clone()),
            Some(1),
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
        budget: budget(),
        input,
    };
    let q = port
        .prepare_validator_question(&original, budget(), &[], CpuEndpointSelection::collection())
        .unwrap();
    assert!(q.support_mask()[AdaptiveValidatorAction::AttackRepair.index()]);
    let prepared = q.prepared_inputs[AdaptiveValidatorAction::AttackRepair.index()]
        .as_ref()
        .unwrap();
    let prepared_digest = prepared.view.digest();
    let prepared_context = prepared.public_context;
    let accepted = accepted(&port, &q, 1, [0., 100., 0., 0., 0., 0., -1.]);
    // A later authored publication is not retroactively inserted into the
    // P/C action input captured before this V forward was submitted.
    port.record_hot(RecordKind::Proposal, &moves(&["d2d4", "d7d5"]), None)
        .unwrap();
    let plan = port.select_validator_action(&q, &accepted).unwrap();
    assert_eq!(plan.action(), AdaptiveValidatorAction::AttackRepair);
    let AdaptiveValidatorEffect::Frame { input, .. } = plan.effect() else {
        panic!("attack")
    };
    assert_eq!(input.stage, RefinementStage::CriticRecheck);
    assert_eq!(input.question.context.prefix, prefix);
    assert_eq!(input.question.proposal, repaired);
    assert_eq!(input.question.counter, Some(counter));
    assert_eq!(input.view.digest(), prepared_digest);
    assert_eq!(input.public_context, prepared_context);
    assert_eq!(
        input.question.position.side_to_move(),
        rz_position::Color::Black
    );
}

#[test]
fn defer_is_zero_work_and_never_invents_a_positive_witness() {
    let mut port = setup();
    let original = target(&mut port, RefinementStage::CriticReply, 1);
    let mut empty = budget();
    empty.original_cpu_credit = 0;
    empty.role_submissions_remaining = 0;
    let q = port
        .prepare_validator_question(&original, empty, &[], CpuEndpointSelection::collection())
        .unwrap();
    let v = accepted(&port, &q, 1, [100., 100., 100., 100., 100., 100., -1.]);
    let plan = port.select_validator_action(&q, &v).unwrap();
    assert_eq!(plan.action(), AdaptiveValidatorAction::Defer);
    let mask = plan.action_support_mask();
    let outcome = port
        .finish_validator_action_semantics(plan.id(), None)
        .unwrap();
    assert_eq!(outcome.action_support_mask(), mask);
    assert_eq!(outcome.accepted_task_logits(), v.task_logits());
    assert_eq!(outcome.source_output(), v.output_digest());
    assert_eq!(outcome.source_consumption(), v.source().consumption_digest);
    assert_eq!(
        outcome.selection_provenance(),
        ValidatorActionSelection::ValidatorArgmax
    );
    assert_eq!(outcome.parent_question(), original.input.question_digest);
    assert_eq!(outcome.parent_view(), original.input.view_seal);
    assert_eq!(
        outcome.parent_input(),
        ContractDigest(original.input.view.digest())
    );
    assert_eq!(
        outcome.parent_public_context(),
        original.input.public_context
    );
    assert!(outcome.witnesses().is_empty());
    assert!(outcome.executions().is_empty());
    assert!(outcome.role_requests().is_empty());
    assert!(outcome.physical_close_wall_cost() >= outcome.wall_cost());
}

#[test]
fn actual_plan_transfers_parent_credit_and_old_parent_question_is_rejected() {
    let mut port = setup();
    // This authored parent is a machine-state contract fixture; only the V
    // head follows actual registry acceptance. It is not a neural/game run.
    let mut original = target(&mut port, RefinementStage::ProposerRanking, 1);
    let clock = port.deadline.clock;
    let mut machine = RefinementMachine::new(port.authority, port.deadline, 512).unwrap();
    let parent_budget = FrameBudget {
        role_submissions_remaining: 120,
        ..budget()
    };
    original.budget = parent_budget;
    original.frame = machine
        .admit(
            original.original_first_move,
            original.input.clone(),
            parent_budget,
            clock,
            MonotonicTick(0),
        )
        .unwrap();
    let q = port
        .prepare_validator_question(
            &original,
            parent_budget,
            &[],
            CpuEndpointSelection::collection(),
        )
        .unwrap();
    let v = accepted(&port, &q, 10, [0., 0., 1., 0., 0., 0., -1.]);
    let plan = port
        .plan_validator_action(&q, &v, AdaptiveValidatorAction::WidenResponses)
        .unwrap();
    assert_eq!(
        plan.selection_provenance(),
        ValidatorActionSelection::RegisteredSupportedEnumerationV1
    );
    assert_eq!(plan.accepted_task_logits(), v.task_logits());
    let id = match machine
        .admit_validator_action(plan, clock, MonotonicTick(2))
        .unwrap()
    {
        ValidatorActionAdmission::Frame(id) => id,
        _ => panic!("widen child"),
    };
    assert_eq!(machine.frame(id).unwrap().parent, Some(original.frame));
    assert_eq!(
        machine.frame(id).unwrap().budget.role_submissions_remaining,
        64
    );
    assert_eq!(
        machine
            .frame(original.frame)
            .unwrap()
            .budget
            .role_submissions_remaining,
        56
    );
    assert_eq!(
        machine.frame(id).unwrap().budget.original_cpu_credit,
        budget().original_cpu_credit
    );
    assert!(
        port.plan_validator_action(&q, &v, AdaptiveValidatorAction::WidenResponses)
            .is_err()
    );
    // Defer is another supported action on the frozen input and consumes no
    // additional frame/native credit after the child transfer.
    let defer = port
        .plan_validator_action(&q, &v, AdaptiveValidatorAction::Defer)
        .unwrap();
    assert_eq!(
        machine
            .admit_validator_action(defer, clock, MonotonicTick(3))
            .unwrap(),
        ValidatorActionAdmission::Defer
    );
    assert_eq!(machine.frames().count(), 2);
    let held = port
        .plan_validator_action(&q, &v, AdaptiveValidatorAction::DefendResponse)
        .unwrap();
    let first = original.original_first_move.decode().unwrap();
    let movement = BoardMove::try_from(first).unwrap();
    let next = port
        .input(
            original.input.question.node,
            RefinementStage::ProposalContinuation,
            Some(movement),
            vec![movement],
            vec![movement],
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    let representation = RepresentationKey {
        role: Role::Proposer,
        ..v.ticket().representation
    };
    let parent_ticket = RoleTicketEnvelope {
        id: RequestId::new(port.authority.epoch, 40),
        authority: port.authority,
        question: original.input.question_digest,
        view: original.input.view_seal,
        representation,
        role: Role::Proposer,
        original_deadline: port.deadline,
    };
    machine
        .bind_role(original.frame, &parent_ticket, clock, MonotonicTick(4))
        .unwrap();
    machine
        .resolve(
            StepEvidence {
                frame: original.frame,
                stage: original.input.stage,
                execution: StepExecution::Role(parent_ticket.id),
                submitted_question: original.input.question_digest,
                submitted_view: original.input.view_seal,
                completed_evidence: ContractDigest([99; 32]),
            },
            FrameTransition::Continue(next.clone()),
            clock,
            MonotonicTick(5),
        )
        .unwrap();
    // The changed target seal cannot authorize the old action on a new step.
    let mut altered = q.clone();
    altered.target.input = next;
    assert!(
        port.plan_validator_action(&altered, &v, AdaptiveValidatorAction::AttackRepair)
            .is_err()
    );
    assert!(matches!(
        machine.admit_validator_action(held, clock, MonotonicTick(6)),
        Err(
            crate::pals::coordinator::CoordinatorError::IdentityMismatch(
                "actual V action original parent/input/authority"
            )
        )
    ));
}

#[test]
fn bounded_native_cpu_commit_is_causal_and_frontier_does_not_mint_a_witness() {
    let mut port = setup();
    let original = target(&mut port, RefinementStage::CriticReply, 1);
    let worker = CpuWorker::spawn(CpuWorkerId(1), cpu()).unwrap();
    let profile = worker.profile();
    let q = port
        .prepare_validator_question(
            &original,
            budget(),
            &[profile],
            CpuEndpointSelection::collection(),
        )
        .unwrap();
    let v = accepted(&port, &q, 1, [2., 0., 1., 0., 0., 0., 0.]);
    let plan = port.select_validator_action(&q, &v).unwrap();
    let id = plan.id();
    let AdaptiveValidatorEffect::Frame {
        input,
        budget: child_budget,
    } = plan.into_effect()
    else {
        panic!("CPU frame")
    };
    let frame = FrameId {
        authority: port.authority,
        sequence: 2,
    };
    port.bind_validator_action(id, frame).unwrap();
    let work = Work {
        frame,
        original_first_move: original.original_first_move,
        original_deadline: port.deadline,
        budget: child_budget,
        input,
    };
    let eligibility_before = port.validator_review_key(&work).unwrap();
    assert_eq!(
        port.validator_review_key(&work).unwrap(),
        eligibility_before
    );
    let prepared = port.cpu_question(&work, &[profile]).unwrap();
    let clock = InstantClock::new(port.deadline.clock, port.started);
    let envelope = CpuJobEnvelope {
        id: CpuJobId {
            authority: port.authority,
            sequence: 1,
        },
        state_identity: prepared.position.export().unwrap().snapshot().identity(),
        question: work.input.question_digest,
        root_restriction: cpu_root_restriction_digest(prepared.root_moves.as_deref()),
        evaluation_namespace: profile.evaluation_namespace,
        search_conditions: profile.search_conditions,
        total_native_node_credit: prepared.total_native_node_credit,
        hard_deadline: port.deadline,
    };
    let job = worker
        .prepare_job(
            envelope,
            prepared.position,
            prepared.root_moves,
            prepared.target_depth,
            &clock,
        )
        .unwrap();
    let grant = SliceGrant {
        job: envelope.id,
        execution: NativeExecutionId::new(port.authority.epoch, 1),
        owner: profile.owner,
        node_credit: 1,
        soft_lease: Duration::from_secs(1),
        control_sequence: 1,
    };
    let mut ledger = CpuCreditLedger::new(port.authority, 8192, clock.clone()).unwrap();
    ledger.register(envelope).unwrap();
    ledger.reserve(grant).unwrap();
    port.cpu_admitted(&work, &envelope, &grant).unwrap();
    worker.begin(job, grant).unwrap();
    let report = loop {
        let observed = worker.signal.version();
        if let Some(event) = worker.try_event().unwrap() {
            match event {
                CpuWorkerEvent::Slice(report) => break report,
                _ => panic!("unexpected termination"),
            }
        }
        assert!(
            clock.now().unwrap() < port.deadline.at,
            "bounded CPU fixture deadline"
        );
        worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    };
    ledger.settle(&report).unwrap();
    let guard = port
        .begin_validator_action_commit(
            id,
            &work,
            StepExecution::Cpu {
                job: envelope.id,
                execution: grant.execution,
            },
        )
        .unwrap();
    let (obs, complete) = port.commit_cpu_report(&work, &report).unwrap();
    port.after_cpu(&work, obs, complete).unwrap();
    assert!(
        port.cpu_admissions
            .get(&work.input.question_digest)
            .unwrap()
            .grant
            .is_none()
    );
    port.finish_validator_action_commit(guard, &work).unwrap();
    assert_ne!(
        port.validator_review_key(&work).unwrap(),
        eligibility_before
    );
    assert_eq!(
        port.validator_state
            .actions
            .get(&id)
            .unwrap()
            .executions
            .len(),
        1
    );
    assert!(
        port.validator_state
            .actions
            .get(&id)
            .unwrap()
            .witnesses
            .is_empty()
    );
    assert!(
        report.continuation_retained,
        "one-node native slice must retain its own paused stack"
    );
    let resumed_question = port
        .prepare_validator_question(
            &work,
            budget(),
            &[profile],
            CpuEndpointSelection::collection(),
        )
        .unwrap();
    assert!(!resumed_question.support_mask()[AdaptiveValidatorAction::ResumeTask.index()]);
    assert_eq!(
        resumed_question.unsupported_reason(AdaptiveValidatorAction::ResumeTask),
        Some(ValidatorUnsupported::AlreadyOwnedByAction),
        "one paused frame cannot acquire a second causal action owner"
    );
    let wrong_owner = CpuWorkerProfile {
        owner: CpuWorkerId(2),
        ..profile
    };
    let wrong = port
        .prepare_validator_question(
            &work,
            budget(),
            &[wrong_owner],
            CpuEndpointSelection::collection(),
        )
        .unwrap();
    assert!(!wrong.support_mask()[AdaptiveValidatorAction::ResumeTask.index()]);
    let published = accepted(&port, &resumed_question, 2, [0., 0., 0., 0., 100., 0., -1.]);
    let chosen = port
        .select_validator_action(&resumed_question, &published)
        .unwrap();
    assert_ne!(chosen.action(), AdaptiveValidatorAction::ResumeTask);
    assert_eq!(
        ledger.global_usage().unwrap(),
        (report.actual_work.unwrap().nodes, 0)
    );
    assert!(report.actual_work.unwrap().nodes > 0);
    let shutdown = worker.shutdown().unwrap();
    assert!(shutdown.thread_joined && !shutdown.thread_panicked);
    assert_eq!(ledger.global_usage().unwrap().1, 0);
}

#[test]
fn frozen_private_question_seal_catches_budget_query_and_baseline_differences() {
    let mut port = setup();
    let original = target(&mut port, RefinementStage::CriticReply, 1);
    let q = port
        .prepare_validator_question(&original, budget(), &[], CpuEndpointSelection::collection())
        .unwrap();
    let seal = |target: &Work, b: FrameBudget, history: &HashSet<ContractDigest>| {
        port.validator_frozen_context(
            target,
            b,
            CpuEndpointSelection::collection(),
            &q.offers,
            &q.prepared_inputs,
            history,
        )
        .unwrap()
    };
    assert_eq!(
        q.frozen_context_digest(),
        seal(&original, budget(), &q.baseline_witnesses)
    );
    let mut reduced_budget = budget();
    reduced_budget.role_submissions_remaining -= 1;
    assert_ne!(
        q.frozen_context_digest(),
        seal(&original, reduced_budget, &q.baseline_witnesses)
    );
    // This is a pure structural hash regression fixture. An unchanged visible
    // view does not imply that an altered private recheck question is equal;
    // no altered fixture becomes an accepted question or learning sample.
    let mut changed = original.clone();
    Arc::make_mut(&mut changed.input.question).recheck = true;
    assert_eq!(changed.input.view.digest(), original.input.view.digest());
    assert_ne!(
        q.frozen_context_digest(),
        seal(&changed, budget(), &q.baseline_witnesses)
    );
    let mut history_a = q.baseline_witnesses.clone();
    history_a.insert(ContractDigest([1; 32]));
    history_a.insert(ContractDigest([2; 32]));
    let mut history_b = q.baseline_witnesses.clone();
    history_b.insert(ContractDigest([2; 32]));
    history_b.insert(ContractDigest([1; 32]));
    assert_eq!(
        seal(&original, budget(), &history_a),
        seal(&original, budget(), &history_b)
    );
    assert_ne!(
        q.frozen_context_digest(),
        seal(&original, budget(), &history_a)
    );
    let v = accepted(&port, &q, 9, [0., 0., 0., 0., 0., 0., 1.]);
    let plan = port.select_validator_action(&q, &v).unwrap();
    assert_eq!(plan.frozen_context_digest(), q.frozen_context_digest());
    let outcome = port
        .finish_validator_action_semantics(plan.id(), None)
        .unwrap();
    assert_eq!(outcome.frozen_context_digest(), q.frozen_context_digest());
}
