use super::*;
use crate::pals::coordinator::test_support::{
    authority, budget, confirm_mock_native_acceptance, deadline, key,
};
use crate::pals::coordinator::{
    FrameCompletionScope, FrameInput, FramePhase, FrameTransition, RefinementMachine,
    RefinementStage, RoleSubmission,
};
use crate::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
};
use crate::pals::store::{PalsStores, StateId, StoreLimits};
use rz_contracts::{OwnerId, RequestId};
use rz_position::contracts::ContractPosition;
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::CoordinatorPolicy;

struct Fixture {
    stores: PalsStores,
    index: ReviewContextIndex,
    state: StateId,
    rules: Arc<ContractState>,
    view: Arc<ReviewView>,
}
impl Fixture {
    fn new() -> Self {
        Self::from_position(Position::startpos())
    }
    fn from_position(position: Position) -> Self {
        let rules = Arc::new(
            ContractPosition::new(OwnerId(17), position.clone())
                .export()
                .unwrap(),
        );
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(situation).unwrap().state;
        let mut index = ReviewContextIndex::default();
        let legal = position.legal_moves();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ProposePolicy,
                        prefix: &[],
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        Self {
            stores,
            index,
            state,
            rules,
            view,
        }
    }
    fn seal(&self, view: &ReviewView) -> ReviewViewSeal {
        ReviewViewSeal {
            authority: authority(),
            view_sequence: view.sequence(),
            context_policy: Digest(view.policy_digest()),
            view_digest: Digest(view.seal_digest()),
            root_state: self.rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        }
    }
    fn continuation(&mut self, movement: Move16) -> FrameInput<ReviewView, u64> {
        let movement = BoardMove::try_from(movement.decode().unwrap()).unwrap();
        let mut endpoint = Position::from_snapshot(
            self.rules.rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .unwrap();
        endpoint.make_move(movement).unwrap();
        let legal = endpoint.legal_moves();
        let view = Arc::new(
            self.index
                .select(
                    &self.stores,
                    self.state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ProposePolicy,
                        prefix: &[movement],
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        FrameInput {
            stage: RefinementStage::ProposalContinuation,
            question_digest: Digest(view.digest()),
            question: Arc::new(7),
            view_seal: self.seal(&view),
            public_context: Digest(view.record_scope_digest()),
            view,
        }
    }
    fn submit(
        &self,
    ) -> (
        RoleTicketRegistry<ReviewView, RoleOutput>,
        RoleTicketEnvelope,
        RoleOutput,
    ) {
        let mut registry =
            RoleTicketRegistry::new(authority(), 512, CoordinatorPolicy::Fixed, 1).unwrap();
        let mut representation = key(Role::Proposer);
        let mut hash = Sha256::new();
        for movement in &self.view.query().legal {
            let encoded = Move16::encode(rz_contracts::Move::try_from(*movement).unwrap());
            hash.update(encoded.bits().to_le_bytes());
        }
        representation.candidates = Digest(hash.finalize().into());
        let ticket = registry
            .admit(
                RoleSubmission {
                    question: Digest(self.view.digest()),
                    view: self.seal(&self.view),
                    representation,
                    role: Role::Proposer,
                    original_deadline: deadline(),
                },
                Arc::clone(&self.view),
                deadline().clock,
                MonotonicTick(0),
            )
            .unwrap();
        registry.take_dispatch().unwrap();
        let output = RoleOutput {
            id: ticket.id,
            authority: ticket.authority,
            key: ticket.representation,
            payload: RolePayload::Proposal {
                candidate_logits: (0..self.view.query().legal.len())
                    .map(|index| index as f32)
                    .collect(),
                wdl: Wdl::try_new(0.25, 0.5, 0.25, 1e-4).unwrap(),
            },
            private_latent: vec![0.; 16 * 384],
        };
        (registry, ticket, output)
    }
    fn publish(&self) -> ProposerRankingPublished {
        let (mut registry, ticket, output) = self.submit();
        registry
            .complete_proposer_ranking(RoleCompletion {
                ticket,
                physical: PhysicalRoleCompletion::Confirmed,
                result: Ok(output),
            })
            .unwrap();
        registry
            .take_ready(deadline().clock, MonotonicTick(0))
            .unwrap();
        confirm_mock_native_acceptance(&mut registry, &ticket);
        registry
            .accept_proposer_ranking(
                &ticket,
                Arc::clone(&self.rules),
                deadline().clock,
                MonotonicTick(0),
            )
            .unwrap()
    }
}

#[test]
fn one_accepted_root_forward_fans_out_without_duplicate_nn_consumption_or_budget() {
    let mut fixture = Fixture::new();
    let ranking = fixture.publish();
    assert_eq!(
        ranking.ranked_candidates().first(),
        ranking.candidates().last()
    );
    let mut machine = RefinementMachine::new(authority(), deadline(), 512).unwrap();
    let first = ranking.ranked_candidates()[0];
    let second = ranking.ranked_candidates()[1];
    let first_input = fixture.continuation(first);
    let a = machine
        .admit_candidate(
            &ranking,
            first,
            first_input,
            budget(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    let second_input = fixture.continuation(second);
    let b = machine
        .admit_candidate(
            &ranking,
            second,
            second_input,
            budget(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    for id in [a, b] {
        let frame = machine.frame(id).unwrap();
        assert_eq!(frame.input.stage, RefinementStage::ProposalContinuation);
        assert_eq!(frame.phase, FramePhase::Ready);
        assert_eq!(frame.budget, budget());
        assert_eq!(frame.ranking_source(), Some(ranking.source()));
        assert_eq!(frame.evidence, [ranking.source().consumption_digest]);
    }
    let duplicate = fixture.continuation(first);
    assert_eq!(
        machine
            .admit_candidate(
                &ranking,
                first,
                duplicate,
                budget(),
                deadline().clock,
                MonotonicTick(0)
            )
            .unwrap_err(),
        CoordinatorError::Duplicate
    );
    assert_eq!(machine.frames().count(), 2);
    let history = machine.history_usage();
    assert_eq!(history.role_requests, 1);
    assert_eq!(history.cpu_executions, 0);
    assert_eq!(history.frame_genealogy, 2);
    assert_eq!(history.ranking_fanouts, 2);
    machine.revoke();
    machine.retired_after_worker_join();
    assert_eq!(machine.history_peak().frame_genealogy, 2);
    assert_eq!(machine.history_peak().ranking_fanouts, 2);
}

#[test]
fn checked_terminal_endpoint_preserves_c_review_without_fake_cpu_or_nn_execution() {
    let mut fixture =
        Fixture::from_position(Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1").unwrap());
    let ranking = fixture.publish();
    let movement = BoardMove::from_uci("g6g7").unwrap();
    let first = Move16::encode(rz_contracts::Move::try_from(movement).unwrap());
    let mut position = Position::from_snapshot(
        fixture.rules.rules().snapshot(),
        rz_position::PositionLimits::default(),
    )
    .unwrap();
    position.make_move(movement).unwrap();
    let endpoint = ContractPosition::new(OwnerId(19), position)
        .export()
        .unwrap();
    assert!(endpoint.terminal_wdl().is_some());
    let mut machine = RefinementMachine::new(authority(), deadline(), 512).unwrap();
    let input = fixture.continuation(first);
    let mut cpu = input.clone();
    cpu.stage = RefinementStage::CpuValidation;
    let id = machine
        .admit_candidate(
            &ranking,
            first,
            input,
            budget(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    let other = ranking
        .candidates()
        .iter()
        .copied()
        .find(|candidate| *candidate != first)
        .unwrap();
    let mut wrong_cpu = fixture.continuation(other);
    wrong_cpu.stage = RefinementStage::CpuValidation;
    assert!(
        machine
            .resolve_rules_terminal(
                id,
                &endpoint,
                Digest([32; 32]),
                FrameTransition::Continue(wrong_cpu),
                deadline().clock,
                MonotonicTick(0),
            )
            .is_err()
    );
    assert_eq!(
        machine.frame(id).unwrap().input.stage,
        RefinementStage::ProposalContinuation
    );
    assert!(
        machine
            .resolve_rules_terminal(
                id,
                &endpoint,
                Digest([33; 32]),
                FrameTransition::Continue(cpu),
                deadline().clock,
                MonotonicTick(0),
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(
        machine.frame(id).unwrap().input.stage,
        RefinementStage::CpuValidation
    );
    // A terminal of one proposed line does not prove all opponent replies.
    assert!(
        machine
            .resolve_rules_terminal(
                id,
                &endpoint,
                Digest([34; 32]),
                FrameTransition::Complete {
                    scope: FrameCompletionScope::RulesTerminal
                },
                deadline().clock,
                MonotonicTick(0),
            )
            .is_err()
    );
    let closed = machine
        .resolve_rules_terminal(
            id,
            &endpoint,
            Digest([35; 32]),
            FrameTransition::Complete {
                scope: FrameCompletionScope::NoDivergence,
            },
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        closed.completion_scope,
        Some(FrameCompletionScope::NoDivergence)
    );
    assert_eq!(closed.original_first_move, first);
    assert_eq!(closed.evidence.len(), 3);
    assert_eq!(machine.history_usage().role_requests, 1);
    assert_eq!(machine.history_usage().cpu_executions, 0);
}

#[test]
fn rules_shortcircuit_rejects_nonterminal_or_unrelated_history_without_advancing() {
    let mut fixture = Fixture::new();
    let ranking = fixture.publish();
    let first = ranking.ranked_candidates()[0];
    let input = fixture.continuation(first);
    let mut next = input.clone();
    next.stage = RefinementStage::CpuValidation;
    let mut machine = RefinementMachine::new(authority(), deadline(), 512).unwrap();
    let id = machine
        .admit_candidate(
            &ranking,
            first,
            input,
            budget(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    assert!(
        machine
            .resolve_rules_terminal(
                id,
                fixture.rules.as_ref(),
                Digest([33; 32]),
                FrameTransition::Continue(next.clone()),
                deadline().clock,
                MonotonicTick(0),
            )
            .is_err()
    );
    let unrelated_terminal = ContractPosition::new(
        OwnerId(22),
        Position::from_fen("7k/5KQ1/8/8/8/8/8/8 b - - 0 1").unwrap(),
    )
    .export()
    .unwrap();
    assert!(unrelated_terminal.terminal_wdl().is_some());
    assert!(
        machine
            .resolve_rules_terminal(
                id,
                &unrelated_terminal,
                Digest([34; 32]),
                FrameTransition::Continue(next),
                deadline().clock,
                MonotonicTick(0),
            )
            .is_err()
    );
    assert_eq!(machine.frame(id).unwrap().phase, FramePhase::Ready);
    assert_eq!(
        machine.frame(id).unwrap().input.stage,
        RefinementStage::ProposalContinuation
    );
    assert_eq!(machine.frame(id).unwrap().evidence.len(), 1);
    assert_eq!(machine.frame(id).unwrap().budget, budget());
}

#[test]
fn frozen_rules_pair_rejects_neural_seal_without_poisoning_terminal_completion() {
    let mut fixture =
        Fixture::from_position(Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1").unwrap());
    let ranking = fixture.publish();
    let movement = BoardMove::from_uci("g6g7").unwrap();
    let first = Move16::encode(rz_contracts::Move::try_from(movement).unwrap());
    let mut position = Position::from_snapshot(
        fixture.rules.rules().snapshot(),
        rz_position::PositionLimits::default(),
    )
    .unwrap();
    position.make_move(movement).unwrap();
    let endpoint = ContractPosition::new(OwnerId(19), position)
        .export()
        .unwrap();
    let template = fixture.continuation(first);
    let mut machine = RefinementMachine::new(authority(), deadline(), 512).unwrap();
    let parent = machine
        .admit_candidate(
            &ranking,
            first,
            template.clone(),
            budget(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    let at = |stage| {
        let mut input = template.clone();
        input.stage = stage;
        input
    };
    let id = machine
        .enqueue_repair(
            parent,
            first,
            at(RefinementStage::Repair),
            budget(),
            Some(parent),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    for next in [
        RefinementStage::RepairValidation,
        RefinementStage::CriticRecheck,
    ] {
        machine
            .resolve_rules_terminal(
                id,
                &endpoint,
                Digest([35; 32]),
                FrameTransition::Continue(at(next)),
                deadline().clock,
                MonotonicTick(0),
            )
            .unwrap();
    }
    // Deterministic fake-backend C completion supplies the one actual role
    // execution in this state-machine fixture. Rules endpoints use no tickets.
    let current = machine.frame(id).unwrap().input.clone();
    let ticket = RoleTicketEnvelope {
        id: RequestId::new(authority().epoch, 101),
        authority: authority(),
        question: current.question_digest,
        view: current.view_seal,
        representation: key(Role::Critic),
        role: Role::Critic,
        original_deadline: deadline(),
    };
    machine
        .bind_role(id, &ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    machine
        .resolve(
            super::super::StepEvidence {
                frame: id,
                stage: RefinementStage::CriticRecheck,
                execution: super::super::StepExecution::Role(ticket.id),
                submitted_question: ticket.question,
                submitted_view: ticket.view,
                completed_evidence: Digest([36; 32]),
            },
            FrameTransition::Continue(at(RefinementStage::RecheckCounterValidation)),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    machine
        .resolve_rules_terminal(
            id,
            &endpoint,
            Digest([38; 32]),
            FrameTransition::Continue(at(RefinementStage::FreshRepairedEndpoint)),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    machine
        .resolve_rules_terminal(
            id,
            &endpoint,
            Digest([37; 32]),
            FrameTransition::Continue(at(RefinementStage::FreshCounterEndpoint)),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    let before = machine.frame(id).unwrap().evidence.len();
    let mut neural_ticket = ticket;
    neural_ticket.id.sequence = 102;
    neural_ticket.role = Role::Proposer;
    neural_ticket.representation = key(Role::Proposer);
    assert!(
        machine
            .seal_fresh_pair(
                id,
                super::super::FreshPairSeal::from_ticket(
                    &neural_ticket,
                    template.public_context,
                    rz_contracts::Color::White,
                )
            )
            .is_err()
    );
    assert!(
        machine
            .bind_role(id, &neural_ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    assert!(
        machine
            .resolve_rules_terminal(
                id,
                &endpoint,
                Digest([38; 32]),
                FrameTransition::Complete {
                    scope: FrameCompletionScope::CompletedFreshPair
                },
                deadline().clock,
                MonotonicTick(0),
            )
            .is_err()
    );
    assert_eq!(machine.frame(id).unwrap().evidence.len(), before);
    let closed = machine
        .resolve_rules_terminal(
            id,
            &endpoint,
            Digest([39; 32]),
            FrameTransition::Complete {
                scope: FrameCompletionScope::CompletedRulesPair,
            },
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        closed.completion_scope,
        Some(FrameCompletionScope::CompletedRulesPair)
    );
    assert_eq!(machine.history_usage().role_requests, 2);
    assert_eq!(machine.history_usage().cpu_executions, 0);
}

#[test]
fn checked_external_mapping_keeps_namespaces_distinct_and_checks_actual_rules_and_config() {
    use crate::cpu_checker::{
        ExternalCheckerIdentity, ExternalModelMetadata, ExternalTrainingKnowledge,
    };
    use crate::external_cpu::worker::{
        external_evaluation_namespace, external_question_digest, external_root_restriction_digest,
        external_search_conditions,
    };
    use crate::external_cpu::{
        EXTERNAL_UCI_SCORE_SEMANTICS, ExternalCpuConfig, ExternalSlicePolicy, arguments_sha256,
    };
    use crate::pals::coordinator::{CheckedExternalCpuBinding, CoordinatorWork, FrameId};
    use rz_contracts::pals_adaptive::{
        CpuJobId, CpuWorkerId, ExternalCpuJobEnvelope, ExternalSliceGrant,
    };
    use std::time::Duration;
    let mut fixture = Fixture::new();
    let first =
        Move16::encode(rz_contracts::Move::try_from(fixture.view.query().legal[0]).unwrap());
    let mut input = fixture.continuation(first);
    input.stage = RefinementStage::CpuValidation;
    let work = CoordinatorWork {
        frame: FrameId {
            authority: authority(),
            sequence: 1,
        },
        original_first_move: first,
        original_deadline: deadline(),
        budget: super::super::test_support::budget(),
        input,
    };
    let mut position = Position::from_snapshot(
        fixture.rules.rules().snapshot(),
        rz_position::PositionLimits::default(),
    )
    .unwrap();
    position
        .make_move(BoardMove::try_from(first.decode().unwrap()).unwrap())
        .unwrap();
    let state = ContractPosition::new(OwnerId(47), position)
        .export()
        .unwrap();
    // Pure declarations only; this test does not open or spawn the fixture path.
    let directory = std::env::current_dir().unwrap();
    let config = ExternalCpuConfig {
        program: directory.join("fixture-uci"),
        working_directory: directory,
        arguments: Vec::new(),
        expected_uci_name: None,
        identity: ExternalCheckerIdentity {
            adapter_semantics: EXTERNAL_UCI_SCORE_SEMANTICS.into(),
            binary_sha256: "0".repeat(64),
            launch_arguments_sha256: arguments_sha256(&[]),
            declared_name: "fixture".into(),
            declared_version: "fixture/1".into(),
            declared_source: "RoveZero source fixture".into(),
            declared_license: "MIT".into(),
            options: Default::default(),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        },
        max_depth: 8,
        max_prefix_plies: 256,
        handshake_timeout: Duration::from_secs(1),
        max_task_wall_time: Duration::from_secs(2),
        stop_grace: Duration::from_millis(500),
        shutdown_grace: Duration::from_secs(1),
        max_output_bytes: 65536,
        max_line_bytes: 4096,
    };
    let policy = ExternalSlicePolicy::default();
    let restriction = external_root_restriction_digest(None);
    let evaluation = external_evaluation_namespace(&config.identity).unwrap();
    let conditions = external_search_conditions(&config, 8, policy);
    let envelope = ExternalCpuJobEnvelope {
        id: CpuJobId {
            authority: authority(),
            sequence: 1,
        },
        state_identity: state.snapshot().identity(),
        question: external_question_digest(
            state.snapshot().identity(),
            restriction,
            evaluation,
            conditions,
        ),
        root_restriction: restriction,
        evaluation_namespace: evaluation,
        search_conditions: conditions,
        requested_reported_node_hint: 65536,
        hard_deadline: deadline(),
    };
    let grant = ExternalSliceGrant {
        job: envelope.id,
        execution: rz_contracts::ExecutionId::new(authority().epoch, 44),
        owner: CpuWorkerId(0),
        requested_reported_node_hint: 65536,
        soft_quantum: policy.quantum,
        control_sequence: 1,
    };
    let proof = CheckedExternalCpuBinding::attest(
        &work,
        &envelope,
        &grant,
        &state,
        None,
        8,
        policy,
        &config,
        deadline().clock,
        MonotonicTick(0),
    )
    .unwrap();
    assert_eq!(proof.semantic_question(), work.input.question_digest);
    assert_eq!(proof.physical_question(), envelope.question);
    assert_ne!(proof.semantic_question(), proof.physical_question());
    let mut incorrect = envelope;
    incorrect.question = work.input.question_digest;
    assert!(
        CheckedExternalCpuBinding::attest(
            &work,
            &incorrect,
            &grant,
            &state,
            None,
            8,
            policy,
            &config,
            deadline().clock,
            MonotonicTick(0),
        )
        .is_err()
    );
    assert!(
        CheckedExternalCpuBinding::attest(
            &work,
            &envelope,
            &grant,
            fixture.rules.as_ref(),
            None,
            8,
            policy,
            &config,
            deadline().clock,
            MonotonicTick(0),
        )
        .is_err()
    );
    let mut changed = config;
    changed.identity.declared_version = "fixture/2".into();
    assert!(
        CheckedExternalCpuBinding::attest(
            &work,
            &envelope,
            &grant,
            &state,
            None,
            8,
            policy,
            &changed,
            deadline().clock,
            MonotonicTick(0),
        )
        .is_err()
    );
}

#[test]
fn arbitrary_ready_replacement_cannot_replace_sealed_worker_ranking() {
    let fixture = Fixture::new();
    let (mut registry, ticket, output) = fixture.submit();
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    let mut ready = registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    let RolePayload::Proposal {
        candidate_logits, ..
    } = &mut ready.result.as_mut().unwrap().payload
    else {
        unreachable!()
    };
    candidate_logits.fill(-100.);
    confirm_mock_native_acceptance(&mut registry, &ticket);
    let proof = registry
        .accept_proposer_ranking(
            &ticket,
            Arc::clone(&fixture.rules),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    assert_eq!(proof.candidate_logits()[1], 1.);
    assert!(
        registry
            .accept_proposer_ranking(
                &ticket,
                Arc::clone(&fixture.rules),
                deadline().clock,
                MonotonicTick(0)
            )
            .is_err()
    );

    let (mut registry, ticket, output) = fixture.submit();
    registry
        .complete(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(
        registry
            .accept_proposer_ranking(
                &ticket,
                Arc::clone(&fixture.rules),
                deadline().clock,
                MonotonicTick(0)
            )
            .unwrap_err(),
        CoordinatorError::MissingEvidence
    );
}

#[test]
fn ranking_rejects_wrong_head_identity_legal_order_and_expired_or_revoked_acceptance() {
    let fixture = Fixture::new();
    let (mut registry, ticket, mut output) = fixture.submit();
    output.id = RequestId::new(authority().epoch, 99);
    let rejected = registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap_err();
    assert!(matches!(
        rejected.error,
        CoordinatorError::IdentityMismatch(_)
    ));
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Submitted));

    let (mut registry, ticket, output) = fixture.submit();
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    assert!(
        registry
            .accept_proposer_ranking(
                &ticket,
                Arc::clone(&fixture.rules),
                deadline().clock,
                deadline().at
            )
            .is_err()
    );
    registry.revoke_all();
    assert!(
        registry
            .accept_proposer_ranking(
                &ticket,
                Arc::clone(&fixture.rules),
                deadline().clock,
                MonotonicTick(0)
            )
            .is_err()
    );

    let (mut registry, mut ticket, mut output) = fixture.submit();
    // A distinct, internally coherent ticket with a wrong candidate digest is
    // admissible to the generic transport but cannot become a Rules proof.
    registry.revoke_all();
    let mut new_registry =
        RoleTicketRegistry::new(authority(), 512, CoordinatorPolicy::Fixed, 1).unwrap();
    ticket.representation.candidates = Digest([99; 32]);
    ticket = new_registry
        .admit(
            RoleSubmission {
                question: ticket.question,
                view: ticket.view,
                representation: ticket.representation,
                role: ticket.role,
                original_deadline: ticket.original_deadline,
            },
            Arc::clone(&fixture.view),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    output.key = ticket.representation;
    new_registry.take_dispatch().unwrap();
    new_registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    new_registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    confirm_mock_native_acceptance(&mut new_registry, &ticket);
    assert!(matches!(
        new_registry.accept_proposer_ranking(
            &ticket,
            Arc::clone(&fixture.rules),
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::IdentityMismatch(_))
    ));
}

#[test]
fn fanout_first_move_view_and_followup_proposer_namespace_are_sealed() {
    let mut fixture = Fixture::new();
    let ranking = fixture.publish();
    let first = ranking.candidates()[0];
    let different = ranking.candidates()[1];
    let mut machine = RefinementMachine::new(authority(), deadline(), 512).unwrap();
    let wrong = fixture.continuation(different);
    assert!(
        machine
            .admit_candidate(
                &ranking,
                first,
                wrong,
                budget(),
                deadline().clock,
                MonotonicTick(0)
            )
            .is_err()
    );
    assert_eq!(machine.frames().count(), 0);
    let input = fixture.continuation(first);
    let id = machine
        .admit_candidate(
            &ranking,
            first,
            input,
            budget(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    let frame = machine.frame(id).unwrap();
    let mut ticket = RoleTicketEnvelope {
        id: RequestId::new(authority().epoch, 2),
        authority: authority(),
        question: frame.input.question_digest,
        view: frame.input.view_seal,
        representation: key(Role::Proposer),
        role: Role::Proposer,
        original_deadline: deadline(),
    };
    ticket.representation.model = Digest([99; 32]);
    assert!(
        machine
            .bind_role(id, &ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    assert_eq!(machine.frame(id).unwrap().budget, budget());
    ticket.representation.model = ranking.source().ticket.representation.model;
    machine
        .bind_role(id, &ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(
        machine.frame(id).unwrap().budget.role_submissions_remaining,
        budget().role_submissions_remaining - 1
    );
}

#[test]
fn physical_unknown_wrong_rules_history_or_leaf_policy_never_publishes_root_ranking() {
    let fixture = Fixture::new();
    let (mut registry, ticket, mut output) = fixture.submit();
    output.id = RequestId::new(authority().epoch, 999);
    let RolePayload::Proposal {
        candidate_logits, ..
    } = &mut output.payload
    else {
        unreachable!()
    };
    candidate_logits[0] = f32::NAN;
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Unknown,
            result: Ok(output),
        })
        .unwrap();
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Quarantined));
    assert!(!registry.admission_open());
    assert!(
        registry
            .take_ready(deadline().clock, MonotonicTick(0))
            .is_none()
    );
    assert!(
        registry
            .accept_proposer_ranking(
                &ticket,
                Arc::clone(&fixture.rules),
                deadline().clock,
                MonotonicTick(0)
            )
            .is_err()
    );

    let (mut registry, ticket, output) = fixture.submit();
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    let mut position = Position::startpos();
    confirm_mock_native_acceptance(&mut registry, &ticket);
    for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        position.make_uci(text).unwrap();
    }
    let other = Arc::new(
        ContractPosition::new(OwnerId(17), position)
            .export()
            .unwrap(),
    );
    assert!(
        registry
            .accept_proposer_ranking(&ticket, other, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    assert!(
        registry
            .accept_proposer_ranking(
                &ticket,
                Arc::clone(&fixture.rules),
                deadline().clock,
                MonotonicTick(0)
            )
            .is_ok()
    );

    let mut fixture = Fixture::new();
    let first =
        Move16::encode(rz_contracts::Move::try_from(fixture.view.query().legal[0]).unwrap());
    fixture.view = fixture.continuation(first).view;
    let (mut registry, ticket, output) = fixture.submit();
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    confirm_mock_native_acceptance(&mut registry, &ticket);
    assert!(matches!(
        registry.accept_proposer_ranking(
            &ticket,
            Arc::clone(&fixture.rules),
            deadline().clock,
            MonotonicTick(0)
        ),
        Err(CoordinatorError::InvalidTransition(_))
    ));
}

#[test]
fn ranking_request_cannot_be_reused_from_step_binding_or_different_published_source() {
    let mut fixture = Fixture::new();
    let ranking = fixture.publish();
    let first = ranking.candidates()[0];
    let second = ranking.candidates()[1];
    let mut machine = RefinementMachine::new(authority(), deadline(), 512).unwrap();
    let legacy = FrameInput {
        stage: RefinementStage::ProposerRanking,
        question_digest: ranking.source().ticket.question,
        question: Arc::new(1),
        view_seal: ranking.source().ticket.view,
        public_context: ranking.source().public_context,
        view: Arc::clone(ranking.view()),
    };
    let id = machine
        .admit(first, legacy, budget(), deadline().clock, MonotonicTick(0))
        .unwrap();
    machine
        .bind_role(
            id,
            &ranking.source().ticket,
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    let continuation = fixture.continuation(first);
    assert_eq!(
        machine
            .admit_candidate(
                &ranking,
                first,
                continuation,
                budget(),
                deadline().clock,
                MonotonicTick(0)
            )
            .unwrap_err(),
        CoordinatorError::Duplicate
    );

    let mut machine = RefinementMachine::new(authority(), deadline(), 512).unwrap();
    let continuation = fixture.continuation(first);
    machine
        .admit_candidate(
            &ranking,
            first,
            continuation,
            budget(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    // Simulate a buggy producer resetting its RequestId high-water value. A
    // different actual accepted output at that ID cannot replace this source.
    let (mut registry, ticket, mut output) = fixture.submit();
    let RolePayload::Proposal {
        candidate_logits, ..
    } = &mut output.payload
    else {
        unreachable!()
    };
    candidate_logits[0] = 999.;
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    confirm_mock_native_acceptance(&mut registry, &ticket);
    let conflicting = registry
        .accept_proposer_ranking(
            &ticket,
            Arc::clone(&fixture.rules),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    let continuation = fixture.continuation(second);
    assert_eq!(
        machine
            .admit_candidate(
                &conflicting,
                second,
                continuation,
                budget(),
                deadline().clock,
                MonotonicTick(0)
            )
            .unwrap_err(),
        CoordinatorError::Duplicate
    );
    let continuation = fixture.continuation(second);
    machine
        .admit_candidate(
            &ranking,
            second,
            continuation,
            budget(),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    assert_eq!(machine.history_usage().role_requests, 1);
    assert_eq!(machine.history_usage().ranking_fanouts, 2);
}
