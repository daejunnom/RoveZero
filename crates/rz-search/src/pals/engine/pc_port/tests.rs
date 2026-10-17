//! Authored contract receipts test custody and semantic control only. These
//! tests do not execute a neural model or establish native/GPU acceptance.
use super::*;
use crate::pals::coordinator::{
    NativeAcceptanceIssuer, PhysicalRoleCompletion, RoleCompletion, RoleSubmission,
    RoleTicketRegistry,
};
use rz_contracts::pals::RepresentationKey;
use rz_contracts::{
    ClockDomain, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch,
    RootGeneration,
};
use rz_runtime::adaptive::CoordinatorPolicy;

type Port = PcSemanticPort<LegalOrderRoleMock>;
type Input = FrameInput<ReviewView, PcPortQuestion>;

fn port(beam: usize, line_plies: usize) -> Port {
    port_with_observation_limit(beam, line_plies, None)
}
fn port_with_observation_limit(
    beam: usize,
    line_plies: usize,
    observations: Option<usize>,
) -> Port {
    let config = PalsSemanticConfig {
        beam_width: beam,
        line_plies,
        max_nodes: 257,
        max_records: 128,
        max_role_calls: 128,
    };
    let mut owner = PcSemanticOwner::new_selected(
        config,
        LegalOrderRoleMock,
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    if let Some(observations) = observations {
        owner.semantic.stores = PalsStores::new(StoreLimits {
            observations,
            ..StoreLimits::default()
        });
    }
    let authority = SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(7),
        implementation: ContractDigest([7; 32]),
    };
    let position = Position::startpos();
    let registration = owner.register_adaptive_root(&position, authority).unwrap();
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: adaptive_port::AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    let roots = [
        BoardMove::from_uci("e2e4").unwrap(),
        BoardMove::from_uci("d2d4").unwrap(),
    ];
    Port::new_registered_retaining(
        owner,
        position,
        authority,
        Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(20_000_000_000),
        },
        Instant::now() + Duration::from_secs(20),
        ExecutionMode::CpuPc,
        issuer,
        registration,
        Some(&roots),
    )
    .map_err(|(_, error)| error)
    .unwrap()
}

fn accepted<M: RoleModel>(
    port: &mut PcSemanticPort<M>,
    input: &Input,
    request: u64,
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
    // Each authored receipt starts at its distinct process request sequence.
    registry.take_dispatch().unwrap();
    port.role_submitted(input, &ticket).unwrap();
    let logits: Vec<_> = input
        .view
        .query()
        .legal
        .iter()
        .map(|movement| match movement.to_string().as_str() {
            "e2e4" => 10.0,
            "d2d4" => 9.0,
            _ => 0.0,
        })
        .collect();
    let wdl = rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap();
    let output = RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key: representation,
        payload: if role == Role::Critic {
            RolePayload::Counterexample {
                candidate_logits: logits,
                divergence_logits: Vec::new(),
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

fn evidence<M: RoleModel>(
    port: &PcSemanticPort<M>,
    input: &Input,
    sequence: u64,
) -> AdaptivePortRoleEvidence {
    AdaptivePortRoleEvidence {
        execution: Some(rz_contracts::ExecutionId {
            epoch: port.authority.epoch,
            sequence,
        }),
        completed_nn_inputs: Some(1),
        measurement_complete: true,
        physical_fresh: Some(input.question.context.purpose == RoleQueryPurpose::ValueFresh),
    }
}

#[test]
fn two_pending_pc_questions_retain_their_original_views() {
    let mut port = port(2, 3);
    assert_eq!(std::mem::size_of_val(port.researcher()), 0);
    let root = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &root, 1);
    port.consume_accepted(&root, &proof, &output, Some(evidence(&port, &root, 1)))
        .unwrap();
    let first = port.next_role().unwrap().unwrap();
    let second = port.next_role().unwrap().unwrap();
    assert_eq!(
        first.question.context.purpose,
        RoleQueryPurpose::ReplyPolicy
    );
    assert_eq!(
        second.question.context.purpose,
        RoleQueryPurpose::ReplyPolicy
    );
    assert_ne!(
        first.question.context.prefix,
        second.question.context.prefix
    );
    assert_ne!(first.view_seal, second.view_seal);
    assert!(port.pending.contains_key(&first.question_digest));
    assert!(port.pending.contains_key(&second.question_digest));
    assert!(first.view.validates(port.owner.stores()));
    assert!(second.view.validates(port.owner.stores()));
    // Publishing a newer source does not relabel the older submitted view.
    let old_scope = first.view.record_scope_digest();
    assert_eq!(old_scope, first.view.record_scope_digest());
    assert!(
        port.owner
            .semantic
            .nodes
            .iter()
            .all(|node| node.evidence.is_none() && node.resume.is_none())
    );
}

#[test]
fn forged_question_and_late_acceptance_do_not_consume_original_view() {
    let mut port = port(1, 2);
    let input = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &input, 1);
    let mut forged = input.clone();
    Arc::make_mut(&mut forged.question).context.public_revision += 1;
    assert!(
        port.validate_role(&forged, proof.ticket(), &output)
            .is_err()
    );
    assert_eq!(port.counters.consumed_role_outputs, 0);
    port.revoke();
    assert!(
        port.consume_accepted(&input, &proof, &output, None)
            .is_err()
    );
    assert_eq!(port.counters.consumed_role_outputs, 0);
    port.role_discarded(&input, Some(proof.ticket())).unwrap();
    assert!(port.pending.is_empty());
}

#[test]
fn exact_accepted_retry_is_not_a_second_semantic_consumption() {
    let mut port = port(1, 2);
    let input = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &input, 1);
    port.consume_accepted(&input, &proof, &output, None)
        .unwrap();
    let before = port.counters;
    let records = port.owner.records().len();
    port.consume_accepted(&input, &proof, &output, None)
        .unwrap();
    assert_eq!(before, port.counters);
    assert_eq!(records, port.owner.records().len());
    let mut changed = output.clone();
    if let RolePayload::Proposal {
        candidate_logits, ..
    } = &mut changed.payload
    {
        candidate_logits[0] += 1.0;
    }
    assert!(
        port.consume_accepted(&input, &proof, &changed, None)
            .is_err()
    );
    assert_eq!(before, port.counters);
}

#[test]
fn role_admission_cap_does_not_revoke_an_already_submitted_question() {
    let mut port = port(2, 2);
    port.owner.config.max_role_calls = 2;
    port.owner.profile.semantic.max_role_calls = 2;
    let root = port.next_role().unwrap().unwrap();
    let (root_proof, root_output) = accepted(&mut port, &root, 1);
    port.consume_accepted(&root, &root_proof, &root_output, None)
        .unwrap();
    let child = port.next_role().unwrap().unwrap();
    let (child_proof, child_output) = accepted(&mut port, &child, 2);
    assert!(port.next_role().unwrap().is_none());
    assert_eq!(port.completion, PcPortCompletion::Active);
    port.consume_accepted(&child, &child_proof, &child_output, None)
        .unwrap();
    assert_eq!(port.counters.consumed_role_outputs, 2);
    assert!(port.next_role().unwrap().is_none());
    assert_eq!(port.completion, PcPortCompletion::RoleBudget);
}

#[test]
fn store_publication_failure_preserves_actual_head_consumption_once() {
    let mut port = port_with_observation_limit(1, 2, Some(0));
    let root = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &root, 1);
    assert!(port.consume_accepted(&root, &proof, &output, None).is_err());
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert_eq!(port.counters.accepted_proposer, 1);
    assert_eq!(port.completion, PcPortCompletion::Failed);
    assert!(port.owner.semantic.nodes[port.root].edges.len() > 0);
    assert!(port.pending[&root.question_digest].failure.is_some());
    let before = port.counters;
    assert!(port.consume_accepted(&root, &proof, &output, None).is_err());
    assert_eq!(port.counters, before);
}

#[test]
fn discarded_live_lane_cannot_be_reported_as_selective_completion() {
    let mut port = port(1, 2);
    let root = port.next_role().unwrap().unwrap();
    port.role_discarded(&root, None).unwrap();
    assert_eq!(
        port.finish_materialized().completion,
        PcPortCompletion::Failed
    );
    assert!(port.is_complete());
}

#[path = "goal_tests.rs"]
mod goals;
