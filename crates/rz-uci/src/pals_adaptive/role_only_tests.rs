//! Authored CPU native-service outputs, real registry/acceptance/fences, and
//! structurally absent R owners. These tests execute no neural model or GPU.
use super::*;
use rz_contracts::RequestId;
use rz_search::pals::coordinator::AcceptedRoleStep;
use rz_search::pals::engine::NoResearcher;
use std::collections::VecDeque;

struct RoleOnlyFixture {
    no_r: NoResearcher,
    registration: RoleOnlyManagerRegistration,
    ready: VecDeque<FrameInput<ReviewView, u64>>,
    stores: PalsStores,
    submitted: BTreeMap<RequestId, (RoleTicketEnvelope, FrameInput<ReviewView, u64>)>,
    consumed: Vec<(RoleTicketEnvelope, u64)>,
    discarded: usize,
    revoked: bool,
}
impl RoleOnlySearchDriver<u64> for RoleOnlyFixture {
    fn researcher(&self) -> &NoResearcher {
        &self.no_r
    }
    fn validate_registration(
        &self,
        authority: SearchAuthority,
        mode: rz_contracts::rovent::ExecutionMode,
        hard_deadline: Instant,
        root_budget: Digest,
    ) -> Result<(), AdaptiveManagerError> {
        assert_eq!(authority, self.registration.authority);
        assert_eq!(mode, self.registration.mode);
        assert_eq!(hard_deadline, self.registration.hard_deadline);
        assert_eq!(root_budget, self.registration.root_budget);
        Ok(())
    }
    fn next_role(&mut self) -> Result<Option<FrameInput<ReviewView, u64>>, AdaptiveManagerError> {
        Ok(self.ready.pop_front())
    }
    fn role_submitted(
        &mut self,
        input: &FrameInput<ReviewView, u64>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveManagerError> {
        assert!(
            self.submitted
                .insert(ticket.id, (*ticket, input.clone()))
                .is_none()
        );
        Ok(())
    }
    fn validate_role(
        &self,
        input: &FrameInput<ReviewView, u64>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        let (original, submitted) = self.submitted.get(&ticket.id).unwrap();
        assert_eq!(original, ticket);
        assert_eq!(submitted.question_digest, input.question_digest);
        assert_eq!(submitted.view_seal, input.view_seal);
        assert!(Arc::ptr_eq(&submitted.view, &input.view));
        assert!(input.view.validates(&self.stores));
        assert_eq!(output.id, ticket.id);
        assert_eq!(output.key, ticket.representation);
        Ok(())
    }
    fn consume_accepted(
        &mut self,
        input: &FrameInput<ReviewView, u64>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        self.validate_role(input, accepted.ticket(), output)?;
        assert!(accepted.validates_output(output));
        self.submitted.remove(&accepted.ticket().id).unwrap();
        self.consumed
            .push((*accepted.ticket(), input.view.sequence()));
        Ok(())
    }
    fn role_discarded(
        &mut self,
        _: &FrameInput<ReviewView, u64>,
        ticket: Option<&RoleTicketEnvelope>,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(ticket) = ticket {
            self.submitted.remove(&ticket.id);
        }
        self.discarded += 1;
        Ok(())
    }
    fn is_complete(&self) -> bool {
        self.ready.is_empty() && self.submitted.is_empty()
    }
    fn revoke(&mut self) -> Result<(), AdaptiveManagerError> {
        self.revoked = true;
        self.ready.clear();
        Ok(())
    }
    fn review_lease_admission(
        &self,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveReviewLeaseAdmission,
        AdaptiveManagerError,
    > {
        let current = self.stores.review_pin_count().unwrap();
        let capacity = self.stores.review_pin_capacity();
        Ok(
            rz_search::pals::engine::adaptive_port::AdaptiveReviewLeaseAdmission {
                current,
                capacity,
                required_new_leases: 1,
                frontier_replacement_reserve: 0,
                frontier_owner_present: false,
                may_prepare: current < capacity,
            },
        )
    }
}

fn role_only_fixture(
    block: bool,
) -> (
    RoleOnlyExecutionManager<u64, NativeFixture, RoleOnlyFixture>,
    Arc<NativeFacts>,
    Instant,
) {
    let clock = ContractSystemClock::new(ProcessEpoch(517));
    let until = clock.origin() + WINDOW;
    let authority = SearchAuthority {
        epoch: ProcessEpoch(517),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([91; 32]),
    };
    let registration = RoleOnlyManagerRegistration {
        authority,
        mode: rz_contracts::rovent::ExecutionMode::CpuPc,
        root_budget: Digest([92; 32]),
        hard_deadline: until,
        token_budget: 512,
        first_request_sequence: 1,
        role_tickets_max: 2,
        role_call_limit: 2,
        event_order: CoordinatorPolicy::Fixed,
    };
    let root = Position::startpos();
    let rules = ContractPosition::new(OwnerId(517), root.clone())
        .export()
        .unwrap();
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(root.snapshot()).unwrap();
    let state = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    let mut ready = VecDeque::new();
    for (number, stage, purpose) in [
        (
            1,
            RefinementStage::ProposalContinuation,
            RoleQueryPurpose::ProposePolicy,
        ),
        (
            2,
            RefinementStage::CriticReply,
            RoleQueryPurpose::ReplyPolicy,
        ),
    ] {
        let mut endpoint = root.clone();
        let prefix = if block {
            vec![root.legal_moves()[number as usize]]
        } else {
            Vec::new()
        };
        for movement in &prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let legal = endpoint.legal_moves();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        purpose,
                        prefix: &prefix,
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
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
        ready.push_back(FrameInput {
            stage,
            question: Arc::new(number),
            question_digest: Digest(view.digest()),
            public_context: Digest(view.record_scope_digest()),
            view_seal: seal,
            view,
        });
    }
    let driver = RoleOnlyFixture {
        no_r: NoResearcher,
        registration,
        ready,
        stores,
        submitted: BTreeMap::new(),
        consumed: Vec::new(),
        discarded: 0,
        revoked: false,
    };
    let facts = Arc::new(NativeFacts::default());
    if !block {
        facts.child_gate.store(true, Ordering::Release);
    }
    let owned_facts = Arc::clone(&facts);
    let native_clock = clock.clone();
    let native = TicketedNativeService::spawn(
        move || {
            Ok(NativeFixture {
                clock: native_clock,
                facts: owned_facts,
                finish_ok: true,
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    let manager = RoleOnlyExecutionManager::new_retaining(
        registration,
        clock,
        RoleOnlyManagerSources { native, driver },
    )
    .unwrap_or_else(|rejected| panic!("NoR admission: {}", rejected.error));
    (manager, facts, until)
}

#[test]
fn role_only_native_two_pending_keep_their_original_views_and_return_no_researcher_owner() {
    let (mut manager, facts, until) = role_only_fixture(false);
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(manager.failures().next().is_none());
    assert_eq!(manager.driver().consumed.len(), 2);
    assert_ne!(
        manager.driver().consumed[0].1,
        manager.driver().consumed[1].1
    );
    assert_eq!(facts.accepted.lock().unwrap().len(), 2);
    assert_eq!(receipt.counters.submitted_role_tickets, 2);
    assert_eq!(receipt.counters.consumed_role_outputs, 2);
    assert_eq!(receipt.counters.cpu_slice_receipts, 0);
    assert_eq!(receipt.counters.validator_ticket_admissions, 0);
    assert!(!manager.driver().revoked);
    let mut sources = manager
        .into_sources()
        .unwrap_or_else(|(_, error)| panic!("NoR physical closure: {error}"));
    sources.native.begin_shutdown(Instant::now() + WINDOW);
    let finish = Instant::now() + WINDOW;
    while !sources.native.try_join().unwrap() && Instant::now() < finish {
        sources.native.poll();
        thread::sleep(Duration::from_millis(1));
    }
    assert!(sources.native.try_join().unwrap());
}

#[test]
fn role_only_stop_drains_pending_native_inputs_without_consuming_late_heads() {
    let (mut manager, facts, until) = role_only_fixture(true);
    while (manager.pending_tickets() < 2 || facts.started.lock().unwrap().is_empty())
        && Instant::now() < until
    {
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(manager.pending_tickets(), 2);
    manager.begin_shutdown(Instant::now() + WINDOW).unwrap();
    let receipt = manager
        .run_root_until(Instant::now() + WINDOW, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert_eq!(manager.driver().consumed.len(), 0);
    assert_eq!(facts.accepted.lock().unwrap().len(), 0);
    assert!(manager.driver().revoked);
    assert_eq!(receipt.counters.cpu_slice_receipts, 0);
    assert_eq!(receipt.pending_tickets, 0);
}
