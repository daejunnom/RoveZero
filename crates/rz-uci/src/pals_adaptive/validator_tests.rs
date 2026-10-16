//! Authored task heads exercise actual registry acceptance. This never runs a
//! model, GPU, optimizer, cloud, or labels a task result as a valid witness.
use super::*;
use rz_contracts::pals::{RepresentationKey, Role, RoleOutput, RolePayload, SearchAuthority};
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_contracts::{
    ClockDomain, Deadline, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch,
    RootGeneration,
};
use rz_position::{Position, contracts::ContractPosition};
use rz_runtime::adaptive::CoordinatorPolicy;
use rz_search::pals::coordinator::{
    NativeAcceptanceIssuer, PhysicalRoleCompletion, RoleCompletion, RoleSubmission,
    RoleTicketRegistry,
};
use rz_search::pals::engine::RoleQueryPurpose;
use rz_search::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
};
use rz_search::pals::store::{PalsStores, StoreLimits};

#[test]
fn actual_accepted_task_head_keeps_original_view_and_all_seven_logits() {
    let position = Position::startpos();
    let rules = ContractPosition::new(OwnerId(4), position.clone())
        .export()
        .unwrap();
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(position.snapshot()).unwrap();
    let state = stores.situations.get(situation).unwrap().state;
    let view = Arc::new(
        ReviewContextIndex::default()
            .select(
                &stores,
                state,
                ReviewQuerySequences {
                    purpose: RoleQueryPurpose::TaskPolicy,
                    prefix: &[],
                    proposal: &[],
                    counterexample: None,
                    legal: &[],
                    divergences: &[],
                },
                ReviewSelectionPolicy::default(),
            )
            .unwrap(),
    );
    let authority = SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(view.generation()),
        implementation: Digest([7; 32]),
    };
    let clock = ClockDomain(authority.epoch);
    let seal = ReviewViewSeal {
        authority,
        view_sequence: view.sequence(),
        context_policy: Digest(view.policy_digest()),
        view_digest: Digest(view.seal_digest()),
        root_state: rules.snapshot().identity(),
        actual_tokens: view.token_count(),
        selected_records: view.records().len() as u16,
        anchors: view.anchors().len() as u8,
        length_bucket: view.length_bucket(),
    };
    let key = RepresentationKey {
        input: Digest(view.digest()),
        history: seal.root_state.semantic,
        candidates: Digest([1; 32]),
        mask_positions: Digest([2; 32]),
        model: Digest([3; 32]),
        encoding: Digest([4; 32]),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: 0,
        record_revision: 0,
        role: Role::Validator,
    };
    let mut registry =
        RoleTicketRegistry::new(authority, 512, CoordinatorPolicy::Fixed, 1).unwrap();
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let ticket = registry
        .admit(
            RoleSubmission {
                question: Digest([5; 32]),
                view: seal,
                representation: key,
                role: Role::Validator,
                original_deadline: Deadline {
                    clock,
                    at: MonotonicTick(1000),
                },
            },
            Arc::clone(&view),
            clock,
            MonotonicTick(1),
        )
        .unwrap();
    registry.take_dispatch().unwrap();
    let logits = [3., 2., 1., 0., -1., -2., -3.];
    registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(RoleOutput {
                id: ticket.id,
                authority,
                key,
                payload: RolePayload::TaskRanking {
                    task_logits: logits.to_vec(),
                },
                private_latent: vec![0.; 16 * 384],
            }),
        })
        .unwrap();
    registry.take_ready(clock, MonotonicTick(2)).unwrap();
    assert!(
        registry
            .accept_validator_task(&ticket, clock, MonotonicTick(2))
            .is_err(),
        "physical delivery alone is not native acceptance"
    );
    // Mock owner only; no backend/model is executed by this contract fixture.
    let permit = registry
        .issue_native_acceptance(&ticket, clock, MonotonicTick(3))
        .unwrap();
    registry
        .confirm_native_acceptance(issuer.issue_success(permit), clock, MonotonicTick(3))
        .unwrap();
    let published = registry
        .accept_validator_task(&ticket, clock, MonotonicTick(3))
        .unwrap();
    let accepted = AcceptedValidator::from_published(published);
    assert!(Arc::ptr_eq(accepted.view(), &view));
    assert_eq!(accepted.source(), &ticket);
    assert_eq!(*accepted.task_logits(), logits);
    assert_eq!(registry.occupancy(), 0);
}
