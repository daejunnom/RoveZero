//! Actual CPU owner/native command-loop lifecycle on terminal Rules inputs.
//! The backend cannot prepare or infer: this is not NN/GPU evidence.
use super::*;
use crate::contracts::SessionScope;
use crate::pals_native::ticketed::{TicketPhysicalResult, TicketPreparation, TicketServiceLimits};
use rz_contracts::pals::{RepresentationKey, RoleOutput};
use rz_contracts::pals_adaptive::{CpuWorkerId, RoleTicketEnvelope};
use rz_contracts::{CancelToken, GameGeneration, OwnerId, PrecisionProfile, RootGeneration};
use rz_runtime::contracts::ContractSystemClock;
use rz_search::cpu::{CpuConfig, CpuEngine, CpuOrderingPolicy, CpuResumePolicy};
use rz_search::cpu_value::BootstrapCpuValue;
use rz_search::cpu_worker::{
    CpuWorkerProfile, cpu_evaluation_namespace_digest, cpu_search_conditions_digest,
};
use rz_search::driver::SearchControl;
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::engine::{PalsConfig, PostRepairRecheckPolicy, ResolverPolicy};
use rz_search::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use rz_search::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};

#[derive(Default)]
struct OwnerFacts {
    factories: AtomicU64,
    quiesces: AtomicU64,
    finishes: AtomicU64,
    finish_ok: AtomicBool,
}
struct TerminalOwner {
    clock: ContractSystemClock,
    facts: Arc<OwnerFacts>,
}
impl TicketNativeBackend<ReviewView, AdaptivePortQuestion> for TerminalOwner {
    type Prepared = ();
    type Output = RoleOutput;
    fn clock(&self) -> ContractSystemClock {
        self.clock.clone()
    }
    fn prepare(
        &mut self,
        _: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        _: &AtomicBool,
    ) -> Result<((), RepresentationKey), RoleFailure> {
        Err(RoleFailure::Backend(
            "terminal lifecycle fixture forbids model preparation".into(),
        ))
    }
    fn prepared_host_bytes(&self, _: &()) -> u64 {
        0
    }
    fn run(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        _: &(),
        _: &AtomicBool,
    ) -> TicketPhysicalResult<RoleOutput> {
        TicketPhysicalResult {
            physical: PhysicalRoleCompletion::Confirmed,
            output: Err(RoleFailure::Backend(
                "terminal lifecycle fixture has no neural model".into(),
            )),
        }
    }
    fn output_host_bytes(&self, _: &RoleOutput) -> u64 {
        0
    }
    fn accept(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        _: &(),
        _: &RoleOutput,
    ) -> Result<(), RoleFailure> {
        Err(RoleFailure::InvalidOutput)
    }
    fn discard(&mut self, _: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn quiesce(&mut self, until: Instant) -> bool {
        self.facts.quiesces.fetch_add(1, Ordering::AcqRel);
        Instant::now() < until
    }
    fn finish(&mut self, _: Instant) -> bool {
        self.facts.finishes.fetch_add(1, Ordering::AcqRel);
        self.facts.finish_ok.load(Ordering::Acquire)
    }
}
fn fixture() -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    fixture_with_startup_gate(None)
}
fn fixture_with_startup_gate(
    startup_gate: Option<std::sync::mpsc::Receiver<()>>,
) -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    let clock = ProcessClock::new(ProcessEpoch(31));
    let facts = Arc::new(OwnerFacts::default());
    facts.finish_ok.store(true, Ordering::Release);
    let factory_facts = Arc::clone(&facts);
    let native_clock = clock.runtime_clock();
    let native = TicketedNativeService::spawn(
        move || {
            factory_facts.factories.fetch_add(1, Ordering::AcqRel);
            if let Some(gate) = startup_gate {
                gate.recv_timeout(Duration::from_secs(3)).map_err(|_| {
                    RoleFailure::Backend("terminal fixture startup gate expired".into())
                })?;
            }
            Ok(TerminalOwner {
                clock: native_clock,
                facts: factory_facts,
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    let cpu = CpuEngine::with_evaluator_ordering_and_resume_policy(
        CpuConfig {
            tt_entries: 16,
            max_depth: 2,
            quiescence_ply: 2,
            ..Default::default()
        },
        Arc::new(BootstrapCpuValue::default()),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuResumePolicy::PausedStack,
    )
    .unwrap();
    let actual_owner = cpu.paused_stack_snapshot().unwrap().owner_id;
    let profile = CpuWorkerProfile {
        owner: CpuWorkerId(0),
        max_depth: cpu.config().max_depth,
        evaluation_namespace: cpu_evaluation_namespace_digest(&cpu).unwrap(),
        search_conditions: cpu_search_conditions_digest(&cpu),
    };
    let metadata = AdaptiveCheckerMetadata::capture_owned(&cpu).unwrap();
    let value = ModelValueIdentity {
        semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
        model: "terminal-owned-fixture".into(),
        encoding: "terminal-fixture-v3".into(),
        precision: "fp32".into(),
        model_epoch: [5; 32],
    };
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest([5; 32]),
        Digest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value.clone()), namespace)
            .unwrap();
    let registration = PalsCheckerRegistration {
        identity: metadata.identity().clone(),
        conditions: metadata.conditions().into(),
        capabilities: metadata.capabilities(),
        owned_descriptor: metadata.owned_descriptor(),
        cpu_resume_policy: metadata.resume_policy(),
        role_model: value.model,
        model_value: None,
        resolver_version: rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
        resolver_semantics: rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS,
    };
    let engine = PalsEngine::new_with_boxed_checker_and_policies(
        PalsConfig {
            beam_width: 1,
            line_plies: 1,
            cpu_nodes_per_task: 8,
            ..Default::default()
        },
        model,
        Box::new(metadata.clone().into_mirror()),
        ResolverPolicy::OwnRawRestricted,
        PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
    )
    .unwrap()
    .with_selected_review_context(
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    let next = Arc::new(AtomicU64::new(1));
    let issuer: AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::AcqRel))));
    let options = AdaptiveSessionOptions {
        implementation: Digest([4; 32]),
        identity: EngineIdentity {
            name: "PALS terminal lifecycle fixture".into(),
            author: "RoveZero contributors".into(),
        },
        max_rounds: 1,
        max_cpu_nodes: 64,
        cpu_depth: 1,
        token_budget: 512,
        policy: CoordinatorPolicy::Fixed,
        endpoints: CpuEndpointSelection::deployment(),
        controller: ControllerConfig::default(),
        device_digest: Digest([7; 32]),
        cpu_device_digest: Digest([8; 32]),
        external_policy: ExternalSlicePolicy::default(),
    };
    let driver = AdaptiveSessionDriver::new(
        options,
        AdaptiveSessionSources {
            native,
            engine,
            cpu: vec![PersistentCpuOwner {
                profile,
                engine: cpu,
            }],
            external: None,
            checker_metadata: metadata,
        },
        clock.clone(),
        issuer,
        registration,
        None,
        None,
    )
    .unwrap_or_else(|rejected| panic!("{:?}", rejected.failure));
    (driver, facts, clock, actual_owner)
}
fn context(clock: &ProcessClock, root: u64, game: u64) -> SearchSessionContext {
    let now = Instant::now();
    let authority = SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(game),
        root: RootGeneration(root),
        implementation: Digest([4; 32]),
    };
    SearchSessionContext {
        authority,
        control: SearchControl::new(now + Duration::from_secs(5), 64),
        cancellation: CancelToken::new(),
        shutdown_deadline: now + Duration::from_secs(6),
        current: Arc::new(Mutex::new(SessionScope::Search(authority))),
    }
}
#[test]
fn initial_game_failure_guard_closes_admission_and_retains_owner() {
    let (driver, facts, clock, _) = fixture();
    let driver = Arc::new(driver);
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    let context = context(&clock, 1, 0);
    let until = context.shutdown_deadline;
    let state = driver.state.lock().unwrap();
    // Inject only the failed-initial-binding close boundary. This verifies
    // sticky close and owner retention; begin alone does not prove that the
    // other run has already passed its pre-lock check and started waiting.
    let failed_initial = InitialGameFailureGuard {
        closed: &driver.closed,
        armed: true,
    };
    let waiting_driver = Arc::clone(&driver);
    let waiting = std::thread::spawn(move || {
        waiting_driver.run(&mate, &context, &mut |_| {
            panic!("closed waiting root must never publish a move")
        })
    });
    while driver.work_receipt().unwrap().unwrap().active_invocations != 1 {
        assert!(Instant::now() < until);
        std::thread::yield_now();
    }
    drop(failed_initial);
    assert!(driver.closed.load(Ordering::Acquire));
    drop(state);
    let error = waiting.join().unwrap().unwrap_err();
    assert_eq!(error.code, "PhysicalCompletionUnknown");
    assert!(driver.latest_root_receipt().unwrap().is_none());
    assert_eq!(facts.quiesces.load(Ordering::Acquire), 0);
    let closed = driver.finish_process(until).unwrap();
    assert!(closed.permits_process_close());
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
}

#[test]
fn terminal_roots_and_new_game_preserve_actual_session_and_cpu_owner_then_finish_once() {
    let (driver, facts, clock, actual_owner) = fixture();
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    assert!(mate.legal_moves().is_empty());
    for (root, game) in [(1, 0), (2, 0), (3, 1)] {
        let context = context(&clock, root, game);
        let report = driver
            .run(&mate, &context, &mut |_| {
                panic!("terminal Rules state has no bestmove")
            })
            .unwrap();
        assert_eq!(report.best_move, None);
        let receipt = driver.latest_root_receipt().unwrap().unwrap();
        assert_eq!(receipt.authority, context.authority);
        assert!(receipt.permits_root_reuse());
        assert!(!receipt.permits_archive_close());
        assert_eq!(facts.factories.load(Ordering::Acquire), 1);
        assert_eq!(facts.finishes.load(Ordering::Acquire), 0);
        let state = driver.state.lock().unwrap();
        let SessionState::Idle(owners) = &*state else {
            panic!("not reusable")
        };
        assert_eq!(owners.cpu.len(), 1);
        assert_eq!(
            owners.cpu[0]
                .engine
                .paused_stack_snapshot()
                .unwrap()
                .owner_id,
            actual_owner
        );
        drop(state);
        driver
            .accept_uci_output(context.authority, "0000", true)
            .unwrap();
        assert!(
            driver
                .accept_uci_output(context.authority, "0000", true)
                .is_err()
        );
    }
    let until = Instant::now() + Duration::from_secs(3);
    let closed = driver.finish_process(until).unwrap();
    assert!(closed.permits_process_close());
    assert_eq!(closed.epoch(), clock.epoch());
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    assert_eq!(facts.quiesces.load(Ordering::Acquire), 3);
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
    assert!(
        driver
            .finish_process(until)
            .unwrap()
            .permits_process_close()
    );
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
}
#[test]
fn actual_native_ready_after_terminal_quiesce_keeps_revoked_admission_closed_and_owner_reusable() {
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    let (mut driver, facts, clock, _) = fixture_with_startup_gate(Some(wait));
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    let first = context(&clock, 1, 0);
    let until = first.shutdown_deadline;
    driver.collection_hooks.root_admission = Some(Arc::new(move |manager| {
        // The actual factory cannot publish Ready until the exact terminal
        // Rules root has revoked admission. No sleeps or scheduling assumption
        // decide which event wins this regression.
        manager.driver_mut().prepare_root_rules()?;
        assert!(manager.driver().root_is_terminal()?);
        manager.begin_root_quiesce(until)?;
        release.send(()).map_err(|_| {
            AdaptiveManagerError::Invalid("terminal fixture startup gate receiver lost")
        })?;
        Ok(())
    }));
    let report = driver
        .run(&mate, &first, &mut |_| {
            panic!("terminal Rules root has no bestmove")
        })
        .unwrap();
    assert_eq!(report.best_move, None);
    assert!(
        driver
            .latest_root_receipt()
            .unwrap()
            .unwrap()
            .permits_root_reuse()
    );
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    assert_eq!(facts.finishes.load(Ordering::Acquire), 0);
    driver
        .accept_uci_output(first.authority, "0000", true)
        .unwrap();

    // A new root installs the persistent owner's actual verifier through normal
    // Running admission, without starting another native factory or NN call.
    driver.collection_hooks.root_admission = None;
    let second = context(&clock, 2, 0);
    assert_eq!(
        driver
            .run(&mate, &second, &mut |_| panic!(
                "terminal Rules root has no bestmove"
            ))
            .unwrap()
            .best_move,
        None
    );
    assert!(
        driver
            .latest_root_receipt()
            .unwrap()
            .unwrap()
            .permits_root_reuse()
    );
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    assert!(
        driver
            .finish_process(second.shutdown_deadline)
            .unwrap()
            .permits_process_close()
    );
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
}
#[test]
fn physical_finish_unknown_is_retained_and_never_a_process_receipt() {
    let (driver, facts, _, _) = fixture();
    facts.finish_ok.store(false, Ordering::Release);
    let failure = driver
        .finish_process(Instant::now() + Duration::from_secs(3))
        .unwrap_err();
    assert_eq!(
        failure.physical_completion,
        crate::search_driver::DriverPhysicalCompletion::Unknown
    );
    assert!(!matches!(
        &*driver.state.lock().unwrap(),
        SessionState::Closed(_)
    ));
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
    assert_eq!(
        driver.retained_owner_status().unwrap().retained_cpu_engines,
        1
    );
}
