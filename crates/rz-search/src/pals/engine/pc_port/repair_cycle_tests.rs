//! Authored native acceptance receipts exercise the real Repair state machine.
//! They do not run a neural model, R/PVS/qsearch/SEE, a teacher or GPU.
//! This module is registered under goal_tests only when the source window opens.
use super::*;

fn moves(text: &str) -> Vec<BoardMove> {
    text.split_ascii_whitespace()
        .map(|text| BoardMove::from_uci(text).unwrap())
        .collect()
}

/// The issuer sees the exact authored output before it signs retirement. The
/// fixture never modifies an already accepted head or constructs a Step itself.
fn accepted_scripted(
    port: &mut GoalPort,
    input: &Input,
    request: u64,
    selected: Option<BoardMove>,
    probabilities: [f32; 3],
) -> (AcceptedRoleStep, RoleOutput) {
    let mut registry = RoleTicketRegistry::<ReviewView, RoleOutput>::new(
        port.authority,
        512,
        CoordinatorPolicy::Fixed,
        request,
    )
    .unwrap();
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let role = role_for_purpose(input.question.context.purpose).unwrap();
    let representation = RepresentationKey {
        input: ContractDigest([request as u8; 32]),
        history: ContractDigest([2; 32]),
        candidates: ContractDigest([3; 32]),
        mask_positions: ContractDigest([4; 32]),
        model: ContractDigest([5; 32]),
        encoding: ContractDigest([6; 32]),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: input.question.context.public_revision,
        role,
    };
    let ticket = registry
        .admit(
            RoleSubmission {
                question: input.question_digest,
                view: input.view_seal,
                representation,
                role,
                original_deadline: port.deadline,
            },
            Arc::clone(&input.view),
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    registry.take_dispatch().unwrap();
    port.role_submitted(input, &ticket).unwrap();
    if let Some(movement) = selected {
        assert!(input.view.query().legal.contains(&movement));
    }
    let logits = input
        .view
        .query()
        .legal
        .iter()
        .map(|movement| {
            if Some(*movement) == selected {
                20.0
            } else {
                -20.0
            }
        })
        .collect();
    let divergences = input
        .view
        .query()
        .divergences
        .iter()
        .map(|ply| if *ply == 1 { 20.0 } else { -20.0 })
        .collect();
    let wdl = rz_contracts::Wdl::try_new(probabilities[0], probabilities[1], probabilities[2], 0.0)
        .unwrap();
    let output = RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key: representation,
        payload: if role == Role::Critic {
            RolePayload::Counterexample {
                candidate_logits: logits,
                divergence_logits: divergences,
                wdl,
            }
        } else {
            RolePayload::Proposal {
                candidate_logits: logits,
                wdl,
            }
        },
        private_latent: vec![0.0; 16 * 384],
    };
    port.validate_role(input, &ticket, &output).unwrap();
    registry
        .complete_reciprocal_role(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output.clone()),
        })
        .unwrap();
    registry
        .take_ready(port.deadline.clock, MonotonicTick(0))
        .unwrap();
    let permit = registry
        .issue_reciprocal_native_acceptance(&ticket, port.deadline.clock, MonotonicTick(0))
        .unwrap();
    let accepted = registry
        .accept_reciprocal_native(
            issuer.issue_success(permit),
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    (accepted, output)
}

fn repair_port() -> GoalPort {
    let roots = moves("e2e4");
    let mut port = goal_port(
        Position::startpos(),
        SearchGoal {
            depth_plies: None,
            mate_moves: None,
        },
        Some(&roots),
    );
    // This is the original registered generation chunk, not an inferred depth.
    port.owner.config.line_plies = 6;
    port.owner.profile.semantic.line_plies = 6;
    port.enable_post_repair_recheck(PostRepairRecheckPolicy::IterativeFrozenModelWdlV2)
        .unwrap();
    port
}

#[test]
fn actual_pc_repair_recheck_freezes_two_endpoints_and_emits_conditional_witness() {
    let proposal = moves("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6");
    let counter = moves("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4");
    let repaired = moves("e2e4 c7c5 b1c3 b8c6 f1b5 g8f6");
    let rechecked = moves("e2e4 c7c5 b1c3 d7d6 g1f3 g8f6");
    let mut port = repair_port();
    assert!(port.take_repair_report().is_err());
    assert_eq!(std::mem::size_of_val(port.researcher()), 0);
    let mut calls = 0;
    let mut visible_source = None;
    let mut repair_acceptance = None;
    let mut recheck_acceptance = None;
    let mut frozen_scope = None;
    let mut frozen_count = 0;
    let mut original_root: Option<(Input, AcceptedRoleStep, RoleOutput)> = None;
    while let Some(input) = port.next_role().unwrap() {
        calls += 1;
        assert!(calls < 80, "the bounded actual cycle must make progress");
        let query = input.view.query();
        let purpose = input.question.context.purpose;
        let cycle = input.question.repair.is_some();
        let is_fresh = purpose == RoleQueryPurpose::ValueFresh;
        let selected = if query.legal.is_empty() || is_fresh {
            None
        } else if cycle {
            let target = match purpose {
                RoleQueryPurpose::ProposePolicy => &proposal,
                RoleQueryPurpose::RepairPolicy => &repaired,
                RoleQueryPurpose::ReplyPolicy if query.counterexample.is_some() => &rechecked,
                RoleQueryPurpose::ReplyPolicy => &counter,
                _ => panic!("unexpected real Repair policy"),
            };
            assert!(target.starts_with(&query.prefix));
            Some(target[query.prefix.len()])
        } else if query.prefix.is_empty() {
            Some(proposal[0])
        } else {
            query.legal.first().copied()
        };
        let probabilities = if cycle && is_fresh && query.prefix == repaired {
            [0.75, 0.125, 0.125]
        } else if cycle && is_fresh && query.prefix == rechecked {
            [0.125, 0.125, 0.75]
        } else {
            [0.25, 0.50, 0.25]
        };
        if cycle && purpose == RoleQueryPurpose::DivergencePolicy && query.counterexample.is_some()
        {
            assert!(query.prefix.is_empty());
            assert_eq!(query.counterexample.as_ref(), Some(&counter));
            let record = input
                .view
                .records()
                .iter()
                .find(|record| record.kind == RecordKind::Counterexample && record.line == counter)
                .expect("the target is visible in this exact C input");
            visible_source = Some((input.question_digest, input.view_seal, record.revision));
            assert!(port.divergence_witnesses().next().is_none());
        }
        if cycle && is_fresh {
            if let Some(scope) = frozen_scope {
                assert_eq!(input.public_context, scope);
            } else {
                frozen_scope = Some(input.public_context);
            }
            assert!(query.prefix == repaired || query.prefix == rechecked);
            assert!(port.divergence_witnesses().next().is_none());
            frozen_count += 1;
        }
        let (accepted, output) =
            accepted_scripted(&mut port, &input, calls, selected, probabilities);
        if cycle && purpose == RoleQueryPurpose::RepairPolicy && query.prefix.len() == 5 {
            repair_acceptance = Some(accepted.ticket().id);
        }
        if cycle
            && purpose == RoleQueryPurpose::ReplyPolicy
            && query.counterexample.is_some()
            && query.prefix.len() == 5
        {
            recheck_acceptance = Some(accepted.ticket().id);
        }
        let physical = evidence(&port, &input, calls);
        if cycle {
            let (_, foreign, foreign_output) = original_root.as_ref().unwrap();
            let before = port.counters;
            assert!(
                port.consume_accepted(&input, foreign, foreign_output, Some(physical))
                    .is_err()
            );
            assert_eq!(port.counters, before);
            assert!(port.pending.contains_key(&input.question_digest));
        }
        port.consume_accepted(&input, &accepted, &output, Some(physical))
            .unwrap();
        assert!(
            port.role_consumption_recorded(accepted.ticket(), &input)
                .unwrap()
        );
        let counters = port.counters;
        let ready = port.ready.len();
        port.consume_accepted(&input, &accepted, &output, Some(physical))
            .unwrap();
        assert_eq!(port.counters, counters);
        assert_eq!(port.ready.len(), ready);
        if !cycle && input.question.context.prefix.is_empty() {
            original_root = Some((input, accepted, output));
        }
    }
    assert_eq!(frozen_count, 2);
    let witnesses: Vec<_> = port.divergence_witnesses().collect();
    assert_eq!(witnesses.len(), 1);
    let witness = witnesses[0];
    assert_eq!(witness.resolution(), PcDivergenceResolution::Refuted);
    assert!(witness.target_known());
    let (question, view, revision) = visible_source.unwrap();
    assert_eq!(witness.source().accepted().ticket().question, question);
    assert_eq!(witness.source().accepted().ticket().view, view);
    assert_eq!(witness.source().counter_revision(), revision);
    assert_eq!(witness.source().divergence_ply(), 1);
    assert_eq!(witness.repair().unwrap().moves(), repaired);
    assert_eq!(
        witness.repair().unwrap().accepted().ticket().id,
        repair_acceptance.unwrap()
    );
    assert_eq!(witness.recheck().unwrap().moves(), rechecked);
    assert_eq!(
        witness.recheck().unwrap().accepted().ticket().id,
        recheck_acceptance.unwrap()
    );
    assert_eq!(witness.repaired_endpoint().unwrap().line(), repaired);
    assert_eq!(witness.counter_endpoint().unwrap().line(), rechecked);
    assert_eq!(port.counters.consumed_role_outputs, calls);
    assert_eq!(port.counters.completed_nn_inputs, Some(calls));
    assert!(port.pending.is_empty());
    assert!(
        port.owner
            .semantic
            .nodes
            .iter()
            .all(|node| node.evidence.is_none() && node.resume.is_none())
    );
    drop(witnesses);
    drop(original_root);
    let original_authority = port.authority;
    let original_budget = port.original_root_budget().unwrap();
    let report = port.take_repair_report().unwrap().unwrap();
    assert_eq!(report.authority(), original_authority);
    assert_eq!(report.root_budget(), original_budget);
    assert_eq!(report.witnesses().count(), 1);
    assert_eq!(report.incomplete_cycle_count(), 0);
    assert!(report.owned_host_bytes().unwrap() > 0);
    assert!(port.take_repair_report().unwrap().is_none());
    port.retire_rovent_frontier().unwrap();
    let mut owner = port.into_owner();
    assert!(owner.retired_repair_report().is_none());
    assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
    let next = SearchAuthority {
        root: RootGeneration(8),
        ..original_authority
    };
    assert!(
        owner
            .register_adaptive_root(&Position::startpos(), next)
            .is_ok()
    );
}

#[test]
fn partial_actual_fresh_report_survives_owner_transfer_and_blocks_reset_until_recovered() {
    let proposal = moves("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6");
    let counter = moves("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4");
    let repaired = moves("e2e4 c7c5 b1c3 b8c6 f1b5 g8f6");
    let rechecked = moves("e2e4 c7c5 b1c3 d7d6 g1f3 g8f6");
    let mut port = repair_port();
    let authority = port.authority;
    let budget = port.original_root_budget().unwrap();
    let same_store = port.owner.stores().read_owner();
    let mut calls = 0;
    let endpoint_ticket = loop {
        let input = port
            .next_role()
            .unwrap()
            .expect("bounded cycle reaches its first actual Fresh");
        calls += 1;
        assert!(calls < 80);
        let query = input.view.query();
        let purpose = input.question.context.purpose;
        let cycle = input.question.repair.is_some();
        let fresh = cycle && purpose == RoleQueryPurpose::ValueFresh;
        let selected = if query.legal.is_empty() || purpose == RoleQueryPurpose::ValueFresh {
            None
        } else if cycle {
            let line = match purpose {
                RoleQueryPurpose::ProposePolicy => &proposal,
                RoleQueryPurpose::RepairPolicy => &repaired,
                RoleQueryPurpose::ReplyPolicy if query.counterexample.is_some() => &rechecked,
                RoleQueryPurpose::ReplyPolicy => &counter,
                _ => panic!("unexpected actual Repair policy"),
            };
            Some(line[query.prefix.len()])
        } else if query.prefix.is_empty() {
            Some(proposal[0])
        } else {
            query.legal.first().copied()
        };
        let (accepted, output) =
            accepted_scripted(&mut port, &input, calls, selected, [0.25, 0.5, 0.25]);
        let physical = evidence(&port, &input, calls);
        port.consume_accepted(&input, &accepted, &output, Some(physical))
            .unwrap();
        if fresh {
            break accepted.ticket().id;
        }
    };
    assert!(port.divergence_witnesses().next().is_none());
    // Root cancellation does not convert the one completed endpoint to a pair.
    port.revoke();
    port.retire_rovent_frontier().unwrap();
    let mut owner = port.into_owner();
    assert!(same_store.matches(owner.stores()));
    assert!(owner.retired_repair_host_bytes().unwrap() > 0);
    assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
    let root_before = owner.stores().generation();
    let next = SearchAuthority {
        root: RootGeneration(8),
        ..authority
    };
    assert!(
        owner
            .register_adaptive_root(&Position::startpos(), next)
            .is_err()
    );
    let next_game = SearchAuthority {
        game: GameGeneration(1),
        ..next
    };
    assert!(
        owner
            .try_adaptive_new_game(
                next_game,
                Instant::now() + Duration::from_secs(5),
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_eq!(owner.stores().generation(), root_before);
    let report = owner.take_retired_repair_report().unwrap();
    assert_eq!(report.authority(), authority);
    assert_eq!(report.root_budget(), budget);
    assert_eq!(report.completion(), PcPortCompletion::Canceled);
    assert_eq!(report.counters().completed_nn_inputs, Some(calls));
    assert_eq!(report.witnesses().count(), 0);
    assert_eq!(report.incomplete_cycle_count(), 1);
    let progress = report.incomplete_progress().next().unwrap();
    let cycle = progress.cycle();
    assert_eq!(
        report.incomplete_resolution(cycle),
        Some(PcDivergenceResolution::Unknown)
    );
    assert!(report.incomplete_target(cycle).is_some());
    assert!(report.incomplete_repair(cycle).is_some());
    assert!(report.incomplete_recheck(cycle).is_some());
    assert_eq!(
        report
            .incomplete_repaired_endpoint(cycle)
            .unwrap()
            .accepted()
            .ticket()
            .id,
        endpoint_ticket
    );
    assert!(report.incomplete_counter_endpoint(cycle).is_none());
    assert_eq!(
        report.incomplete_last_accepted(cycle).unwrap().ticket().id,
        endpoint_ticket
    );
    assert!(owner.take_retired_repair_report().is_none());
    owner
        .try_adaptive_new_game(
            next_game,
            Instant::now() + Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert!(
        owner
            .register_adaptive_root(&Position::startpos(), next_game)
            .is_ok()
    );
    // Report historical evidence remains readable after a legitimate new game.
    assert_eq!(report.incomplete_repair(cycle).unwrap().moves(), repaired);
}
