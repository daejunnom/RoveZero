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
fn idle_archive_worker_keeps_actual_pc_projection_and_restores_original_relations() {
    use crate::pals::engine::adaptive_port::archive_worker::{
        ArchiveOperation, ArchiveOwner, ArchiveWork, ArchiveWorker,
    };
    use crate::pals::review_context::ReviewQuerySequences;
    use crate::pals::store::{ArchiveConfig, ArchiveRootCapability};
    use std::sync::atomic::AtomicU64;

    let record_identity = |record: &RoleRecord| {
        (
            record.revision,
            record.parent_revision,
            record.supersedes_revision,
            record.origin_state,
            record.kind,
            record.line.clone(),
            record.value,
            record.completed_depth,
            record.score_scope,
            record.cpu_observation,
            record.perspective,
            record.critical,
        )
    };

    struct Scratch(std::path::PathBuf, bool);
    impl Drop for Scratch {
        fn drop(&mut self) {
            if self.1
                && self.0.parent() == Some(std::env::temp_dir().as_path())
                && self
                    .0
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("rz-idle-archive-"))
                && std::fs::symlink_metadata(&self.0)
                    .is_ok_and(|meta| meta.is_dir() && !meta.file_type().is_symlink())
            {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let mut scratch = Scratch(
        std::env::temp_dir().join(format!(
            "rz-idle-archive-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        )),
        false,
    );
    std::fs::create_dir(&scratch.0).unwrap();
    let mut owner = idle_pc_after_actual_counter();
    assert_eq!(std::mem::size_of_val(owner.researcher()), 0);
    assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
    let original_owner = owner.stores().read_owner();
    let original = Position::startpos();
    let state = owner.semantic.records[0].origin_state;
    let records = owner.records().to_vec();
    assert_eq!(records[1].parent_revision, Some(records[0].revision));
    assert_eq!(records[1].kind, RecordKind::Counterexample);
    let bundle = owner
        .semantic
        .review_index
        .as_ref()
        .unwrap()
        .bundle_for_revision(records[1].revision)
        .unwrap();
    assert_eq!(
        owner
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle(bundle)
            .unwrap()
            .protected_consumptions(),
        2
    );
    // Exact, unrelated Rules states provide real hot pressure outside the next
    // root graph. Current selected evidence remains independently protected.
    for fullmove in 100..310 {
        let position =
            Position::from_fen(&format!("8/7k/8/8/8/8/K7/8 w - - 0 {fullmove}")).unwrap();
        owner.intern_hot(position).unwrap();
    }
    let nodes_before = owner.semantic.nodes.len();
    let query = || ReviewQuerySequences {
        purpose: RoleQueryPurpose::ValueFresh,
        prefix: &[],
        proposal: &[],
        counterexample: None,
        legal: &[],
        divergences: &[],
    };
    let policy = owner.review_policy().unwrap();
    let view = owner
        .semantic
        .review_index
        .as_mut()
        .unwrap()
        .select(&owner.semantic.stores, state, query(), policy)
        .unwrap();
    let digest = view.digest();
    let metadata = view.records_metadata().to_vec();
    let input_records = view.records().to_vec();
    drop(view);
    let mut idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("quiescent PC owner"));
    let config = ArchiveConfig::new(scratch.0.clone(), std::env::current_dir().unwrap());
    let held =
        cap_std::fs::Dir::open_ambient_dir(&scratch.0, cap_std::ambient_authority()).unwrap();
    idle.enable_archive_with_capability(
        config,
        ArchiveRootCapability::from_directory(scratch.0.clone(), held).unwrap(),
    )
    .unwrap();
    let next = SearchAuthority {
        root: RootGeneration(9),
        ..port_authority_for_idle_test()
    };
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let mut position = original.clone();
    position.make_move("e2e4".parse().unwrap()).unwrap();
    let mut worker = ArchiveWorker::start().unwrap();
    let work = ArchiveWork {
        operation: ArchiveOperation::PrepareRoot { position },
        deadline: Instant::now() + Duration::from_secs(20),
        max_io_bytes: 16 * 1024 * 1024,
    };
    let transfer = worker
        .submit(ArchiveOwner::Idle(idle), authority, work)
        .unwrap_or_else(|_| panic!("idle archive admitted"));
    let until = Instant::now() + Duration::from_secs(20);
    let completion = loop {
        if let Some(completion) = worker.try_receive(&transfer).unwrap() {
            break completion;
        }
        assert!(Instant::now() < until, "bounded real archive completion");
        std::thread::yield_now();
    };
    let (ArchiveOwner::Idle(idle), receipt) = completion
        .install(authority, transfer.nonce())
        .unwrap_or_else(|_| panic!("same idle owner return"))
    else {
        panic!("wrong typed owner");
    };
    assert!(receipt.io_bytes.is_some_and(|bytes| bytes > 0));
    assert!(original_owner.matches(idle.stores()));
    let mut owner = idle
        .into_pc()
        .unwrap_or_else(|_| panic!("original PC type"));
    assert!(owner.semantic.nodes.len() < nodes_before);
    assert!(owner.records().is_empty());
    assert_eq!(owner.semantic.revision, records[1].revision);
    assert_eq!(
        owner
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle(bundle)
            .unwrap()
            .protected_consumptions(),
        2
    );
    let view = owner
        .semantic
        .review_index
        .as_mut()
        .unwrap()
        .select(&owner.semantic.stores, state, query(), policy)
        .unwrap();
    assert_eq!(view.digest(), digest);
    assert_eq!(view.records_metadata(), metadata);
    assert_eq!(
        view.records()
            .iter()
            .map(record_identity)
            .collect::<Vec<_>>(),
        input_records
            .iter()
            .map(record_identity)
            .collect::<Vec<_>>()
    );
    drop(view);
    let idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("same returned PC owner"));
    let authority = idle.adaptive_archive_authority(next).unwrap();
    let transfer = worker
        .submit(
            ArchiveOwner::Idle(idle),
            authority,
            ArchiveWork {
                operation: ArchiveOperation::LoadRoot {
                    position: original.clone(),
                },
                deadline: Instant::now() + Duration::from_secs(20),
                max_io_bytes: 16 * 1024 * 1024,
            },
        )
        .unwrap_or_else(|_| panic!("original root restoration admitted"));
    let until = Instant::now() + Duration::from_secs(20);
    let completion = loop {
        if let Some(completion) = worker.try_receive(&transfer).unwrap() {
            break completion;
        }
        assert!(Instant::now() < until, "bounded original root restoration");
        std::thread::yield_now();
    };
    let (ArchiveOwner::Idle(idle), receipt) = completion
        .install(authority, transfer.nonce())
        .unwrap_or_else(|_| panic!("same restored owner"))
    else {
        panic!("wrong restored type");
    };
    assert!(receipt.loaded_root.is_some());
    assert!(original_owner.matches(idle.stores()));
    let owner = idle
        .into_pc()
        .unwrap_or_else(|_| panic!("same restored PC owner"));
    assert_eq!(
        owner
            .records()
            .iter()
            .map(record_identity)
            .collect::<Vec<_>>(),
        records.iter().map(record_identity).collect::<Vec<_>>()
    );
    assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
    assert_eq!(
        owner
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle(bundle)
            .unwrap()
            .protected_consumptions(),
        2
    );
    let mut idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("restored idle"));
    assert!(
        idle.close_archive_owner()
            .unwrap()
            .unwrap()
            .admission_closed
    );
    worker.finish().unwrap();
    scratch.1 = true;
}

fn port_authority_for_idle_test() -> SearchAuthority {
    SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(7),
        implementation: ContractDigest([7; 32]),
    }
}

fn idle_pc_after_actual_counter() -> Box<PcSemanticOwner<LegalOrderRoleMock>> {
    // Reuse the actual registry/accepted-head producer: no fabricated public
    // record, researcher, TT, R worker, or neural execution is needed.
    let mut port = port(1, 2);
    for request in 1..=8 {
        let input = port.next_role().unwrap().expect("real PC question");
        let (accepted, output) = accepted(&mut port, &input, request);
        port.consume_accepted(
            &input,
            &accepted,
            &output,
            Some(evidence(&port, &input, request)),
        )
        .unwrap();
        if port.owner.records().len() == 2 {
            break;
        }
    }
    assert_eq!(port.owner.records().len(), 2);
    port.revoke();
    assert!(port.take_repair_report().unwrap().is_none());
    Box::new(port.into_owner())
}

#[test]
fn idle_archive_rejections_keep_original_pc_owner_and_do_not_enable_storage() {
    use crate::pals::engine::adaptive_port::archive_worker::{
        ArchiveOperation, ArchiveOwner, ArchiveWork, ArchiveWorker,
    };
    let owner = idle_pc_after_actual_counter();
    let store = owner.stores().read_owner();
    let mut idle = owner
        .into_idle_archive()
        .unwrap_or_else(|_| panic!("original idle PC"));
    let authority = port_authority_for_idle_test();
    let mut worker = ArchiveWorker::start().unwrap();
    for case in 0..4 {
        let operation = match case {
            1 => ArchiveOperation::NewGame {
                next_authority: SearchAuthority {
                    game: GameGeneration(1),
                    root: RootGeneration(9),
                    ..authority
                },
            },
            2 => ArchiveOperation::ReclaimHot {
                incoming_pins: Default::default(),
            },
            _ => ArchiveOperation::PrepareRoot {
                position: Position::startpos(),
            },
        };
        let submitted = if case == 0 {
            SearchAuthority {
                root: RootGeneration(8),
                ..authority
            }
        } else {
            authority
        };
        let rejected = worker
            .submit(
                ArchiveOwner::Idle(idle),
                submitted,
                ArchiveWork {
                    operation,
                    deadline: Instant::now() + Duration::from_secs(20),
                    max_io_bytes: 16 * 1024 * 1024,
                },
            )
            .err()
            .expect("exact idle rejection");
        match case {
            0 => assert!(matches!(
                rejected.error,
                PalsError::Store(StoreError::StaleConsumer)
            )),
            1 | 2 => assert!(matches!(
                rejected.error,
                PalsError::Store(StoreError::InvalidConditions(_))
            )),
            _ => assert!(matches!(
                rejected.error,
                PalsError::Store(StoreError::ArchiveDisabled)
            )),
        }
        let ArchiveOwner::Idle(returned) = rejected.owner else {
            panic!("original idle type lost");
        };
        assert!(store.matches(returned.stores()));
        assert_eq!(returned.semantic().records.len(), 2);
        assert!(!returned.archive_enabled());
        assert_eq!(returned.stores().review_pin_count().unwrap(), 0);
        idle = returned;
    }
    worker.finish().unwrap();
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
    assert!(
        port.role_consumption_recorded(proof.ticket(), &root)
            .unwrap()
    );
    assert_eq!(port.completion, PcPortCompletion::Failed);
    assert!(!port.owner.semantic.nodes[port.root].edges.is_empty());
    assert!(port.pending[&root.question_digest].failure.is_some());
    let before = port.counters;
    assert!(port.consume_accepted(&root, &proof, &output, None).is_err());
    assert_eq!(port.counters, before);
    assert!(
        port.role_consumption_recorded(proof.ticket(), &root)
            .unwrap()
    );
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
