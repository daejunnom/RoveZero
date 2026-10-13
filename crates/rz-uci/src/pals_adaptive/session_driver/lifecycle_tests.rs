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
    let clock = ProcessClock::new(ProcessEpoch(31));
    let facts = Arc::new(OwnerFacts::default());
    facts.finish_ok.store(true, Ordering::Release);
    let factory_facts = Arc::clone(&facts);
    let native_clock = clock.runtime_clock();
    let native = TicketedNativeService::spawn(
        move || {
            factory_facts.factories.fetch_add(1, Ordering::AcqRel);
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
