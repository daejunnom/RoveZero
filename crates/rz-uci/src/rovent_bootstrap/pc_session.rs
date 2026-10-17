//! Native P/C product owner. No researcher, PVS engine, CPU credit ledger or
//! checker metadata exists in this state machine.

use super::configuration::{ExecutionConfiguration, RoventStartupConfiguration};
use super::{
    PreparedRoventNative, RoventNativeBackend, RoventNativeService, RoventNativeStartupSnapshot,
    RoventPinnedNativeSources,
};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    ManagerPhase, PcPortDriver, RoleOnlyExecutionManager, RoleOnlyManagerRegistration,
    RoleOnlyManagerSources, RoleOnlyShutdownReceipt,
};
use crate::pals_native::ticketed::{TicketNativeEvent, VirtualReviewTicketOutput};
use crate::rovent_root_admission::source_seal::{
    InitialAdmissionOutcome, SourceSeal, StoppedInitial,
};
use crate::rovent_root_admission::{RoventRootAdmission, RoventRootAdmissionPolicy};
use crate::search_driver::{
    DriverPhysicalCompletion, DriverWork, SearchKind, SearchSessionContext, SearchSessionDriver,
    SearchSessionFailure, SearchSessionReport,
};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{ContractError, Digest};
use rz_position::{BoardMove, Position};
use rz_runtime::contracts::ContractClock;
use rz_search::pals::engine::PcSemanticOwner;
use rz_search::pals::engine::adaptive_port::{AdaptivePortModel, AdaptiveRulesOwnerIssuer};
use rz_search::pals::engine::pc_port::{PcFrontierStartup, PcPortQuestion, PcSemanticPort};
use rz_search::pals::rovent_frontier::{
    FrontierError, FrontierGoalStatus, FrontierLimits, FrontierRegistration, FrontierSnapshot,
    FrontierStop,
};
use sha2::{Digest as _, Sha256};
use std::{
    num::NonZeroU16,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

type Backend = RoventNativeBackend<PcPortQuestion>;
type Model = AdaptivePortModel;
type Manager = RoleOnlyExecutionManager<PcPortQuestion, Backend, PcPortDriver<Model>>;

pub const ROVENT_PC_ROOT_DOMAIN: &str = "rz-rovent-pc-native-root/1";
pub const ROVENT_PC_INITIAL_DOMAIN: &str = "rz-rovent-pc-native-initial-stop/1";
#[derive(Clone, Debug)]
pub struct RoventPcRootReceipt {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    /// Original startup lock and each go's actual workload are distinct seals.
    pub root_admission: Arc<RoventRootAdmission>,
    pub frontier: FrontierSnapshot,
    pub physical: RoleOnlyShutdownReceipt,
}
impl RoventPcRootReceipt {
    fn work(&self) -> Result<DriverWork, SearchSessionFailure> {
        if !self.frontier.work.native_usage_known
            || self.frontier.work.native_nodes != 0
            || self.frontier.work.native_qsearch_nodes != 0
            || self.frontier.work.reserved_native_nodes != 0
            || self.frontier.work.outstanding_native_executions != 0
        {
            return Err(failure(
                "RoventPcResearcherPresent",
                "R-free work contains native R accounting",
            ));
        }
        let total = self.frontier.work.total_node_visits().ok_or_else(|| {
            failure(
                "RoventPcWorkOverflow",
                "original node observations overflow",
            )
        })?;
        let counts = &self.physical.counters;
        Ok(DriverWork::RoventNative {
            root_entries: self.frontier.work.root_entries,
            total_node_visits: Some(total),
            checked_transitions: self.frontier.work.checked_transitions,
            native_nodes: Some(0),
            native_qsearch_nodes: Some(0),
            submitted_role_tickets: counts.submitted_role_tickets,
            consumed_role_outputs: counts.consumed_role_outputs,
            completed_nn_inputs: (counts.nn_input_count_unknown == 0)
                .then_some(counts.completed_nn_inputs),
            consumed_nn_inputs: (counts.consumed_nn_input_count_unknown == 0)
                .then_some(counts.consumed_nn_inputs),
            completed_depth: self.frontier.horizon.completed_depth.map(|depth| depth.0),
            maximum_examined_ply: self.frontier.horizon.maximum_examined_ply.0,
        })
    }
}

/// An original Start stopped before a root, view or native ticket was created.
/// The private basis is issued only with the prepared native owner or after a
/// real root's clean physical transfer. This is not a new physical fence.
#[derive(Clone, Debug)]
pub struct RoventPcInitialReceipt {
    authority: SearchAuthority,
    input_lock: Digest,
    source: Arc<StoppedInitial>,
    idle_custody: IdleCustody,
}
impl RoventPcInitialReceipt {
    pub fn domain(&self) -> &'static str {
        ROVENT_PC_INITIAL_DOMAIN
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn input_lock_digest(&self) -> Digest {
        self.input_lock
    }
    pub fn source(&self) -> &Arc<StoppedInitial> {
        &self.source
    }
    pub fn previous_clean_root(&self) -> Option<&Arc<RoventPcRootReceipt>> {
        match &self.idle_custody {
            IdleCustody::StartupReady => None,
            IdleCustody::RetiredRoot(receipt) => Some(receipt),
        }
    }
    pub fn used_original_startup_ready(&self) -> bool {
        matches!(self.idle_custody, IdleCustody::StartupReady)
    }
    fn report(&self) -> SearchSessionReport {
        SearchSessionReport {
            best_move: self.source.fallback(),
            work: DriverWork::RoventNative {
                root_entries: 0,
                total_node_visits: Some(0),
                checked_transitions: 0,
                native_nodes: Some(0),
                native_qsearch_nodes: Some(0),
                submitted_role_tickets: 0,
                consumed_role_outputs: 0,
                completed_nn_inputs: Some(0),
                consumed_nn_inputs: Some(0),
                completed_depth: None,
                maximum_examined_ply: 0,
            },
            mate_strategy: None,
        }
    }
}

#[derive(Clone, Debug)]
enum IdleCustody {
    StartupReady,
    RetiredRoot(Arc<RoventPcRootReceipt>),
}
struct Idle {
    native: NativeOwner,
    owner: Box<PcSemanticOwner<Model>>,
    next_request_sequence: u64,
    game: Option<u64>,
    custody: IdleCustody,
}
struct NativeOwner {
    service: RoventNativeService<PcPortQuestion>,
    finished: Option<bool>,
    joined: bool,
    unexpected: Vec<TicketNativeEvent<VirtualReviewTicketOutput>>,
}
impl NativeOwner {
    fn new(service: RoventNativeService<PcPortQuestion>) -> Self {
        Self {
            service,
            finished: None,
            joined: false,
            unexpected: Vec::new(),
        }
    }
    fn finish(&mut self, until: Instant) -> Result<(), SearchSessionFailure> {
        self.service.begin_shutdown(until);
        loop {
            let signal = self.service.signal();
            let observed = signal.version();
            while let Some(event) = self.service.poll() {
                match event {
                    TicketNativeEvent::Finished {
                        physical_shutdown_confirmed,
                    } => {
                        self.finished =
                            Some(self.finished.unwrap_or(true) && physical_shutdown_confirmed);
                    }
                    TicketNativeEvent::Ready { .. } | TicketNativeEvent::RootQuiesced(_) => {}
                    other => self.unexpected.push(other),
                }
            }
            self.joined = self
                .service
                .try_join()
                .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
            if self.joined && self.finished == Some(true) {
                return if self.unexpected.is_empty() {
                    Ok(())
                } else {
                    Err(failure(
                        "RoventPcNativeProtocol",
                        "unexpected native events retained until physical shutdown",
                    ))
                };
            }
            if Instant::now() >= until || self.finished == Some(false) {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            signal.wait_changed(
                observed,
                until
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(10)),
            );
        }
    }
}
enum State {
    Idle(Box<Idle>),
    Manager(Box<Manager>),
    RejectedPort {
        native: NativeOwner,
        port: PcPortDriver<Model>,
    },
    RejectedOwner {
        native: NativeOwner,
        owner: Box<PcSemanticOwner<Model>>,
    },
    Closed,
    Moving,
}

/// Startup rejection returns the original prepared physical owner intact.
pub struct RoventPcSessionRejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNative<PcPortQuestion>,
}
pub struct RoventPcSessionDriver {
    sources: Arc<RoventPinnedNativeSources>,
    snapshot: RoventNativeStartupSnapshot,
    configuration: RoventStartupConfiguration,
    clock: ProcessClock,
    issuer: AdaptiveRulesOwnerIssuer,
    identity: EngineIdentity,
    implementation: Digest,
    root_policy: RoventRootAdmissionPolicy,
    selective_policy: Digest,
    frontier_limits: FrontierLimits,
    closed: AtomicBool,
    reset_requested: AtomicBool,
    state: Mutex<State>,
    latest: Mutex<Option<Arc<RoventPcRootReceipt>>>,
    latest_initial: Mutex<Option<Arc<RoventPcInitialReceipt>>>,
}
impl RoventPcSessionDriver {
    #[allow(clippy::result_large_err)]
    pub fn new(
        prepared: PreparedRoventNative<PcPortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventPcSessionRejection> {
        let build =
            (|| -> Result<_, SearchSessionFailure> {
                let configuration = prepared.sources().configuration().clone();
                configuration
                    .require_supported_native_capabilities()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcCapability", &e))?;
                if !matches!(
                    configuration.execution,
                    ExecutionConfiguration::RolesOnly { .. }
                ) || prepared.snapshot().mode.has_native_r()
                    || prepared.snapshot().process_epoch != clock.epoch().0
                    || identity.name.is_empty()
                    || identity.author.is_empty()
                {
                    return Err(failure("RoventPcStartup", "invalid actual R-free startup"));
                }
                let native_clock = prepared.service.ready_clock().ok_or_else(|| {
                    failure("RoventPcStartup", "actual native Ready clock is absent")
                })?;
                let original_clock = clock.runtime_clock();
                if prepared.service.is_quarantined()
                    || native_clock.domain() != original_clock.domain()
                    || native_clock.origin() != original_clock.origin()
                {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
                let namespace = prepared
                    .snapshot()
                    .model_namespace()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcNamespace", &e))?;
                let model = AdaptivePortModel::new_with_namespace(
                    prepared.snapshot().role_model_identity(),
                    Some(
                        prepared
                            .snapshot()
                            .model_value_identity()
                            .map_err(|e| SearchSessionFailure::debug("RoventPcNamespace", &e))?,
                    ),
                    namespace,
                )
                .map_err(|e| SearchSessionFailure::debug("RoventPcModelMirror", &e))?;
                let owner =
                    PcSemanticOwner::new_selected_rovent(
                        configuration.semantic.actual().map_err(|e| {
                            SearchSessionFailure::debug("RoventPcConfiguration", &e)
                        })?,
                        model,
                        configuration.review.actual().map_err(|e| {
                            SearchSessionFailure::debug("RoventPcConfiguration", &e)
                        })?,
                        configuration.index.actual().map_err(|e| {
                            SearchSessionFailure::debug("RoventPcConfiguration", &e)
                        })?,
                    )
                    .map_err(|e| SearchSessionFailure::debug("RoventPcSemanticOwner", &e))?;
                let mut hash = Sha256::new();
                hash.update(b"rz-rovent-pc-native-session/1\0");
                hash.update(super::compiled_bootstrap_source_digest());
                hash.update(
                    configuration
                        .configuration_digest()
                        .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
                );
                hash.update(include_bytes!("pc_session.rs"));
                hash.update([match root_policy {
                    RoventRootAdmissionPolicy::ExactRegistered => 0,
                    RoventRootAdmissionPolicy::DynamicGo => 1,
                }]);
                let selective_policy = Digest(
                    configuration
                        .search_configuration_digest()
                        .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
                );
                let frontier_limits = configuration
                    .frontier
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?;
                Ok((
                    configuration,
                    owner,
                    Digest(hash.finalize().into()),
                    selective_policy,
                    frontier_limits,
                ))
            })();
        let (configuration, owner, implementation, selective_policy, frontier_limits) = match build
        {
            Ok(value) => value,
            Err(failure) => return Err(RoventPcSessionRejection { failure, prepared }),
        };
        let (native, snapshot, sources) = prepared.into_parts();
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        Ok(Self {
            sources,
            snapshot,
            configuration,
            clock,
            issuer,
            identity,
            implementation,
            root_policy,
            selective_policy,
            frontier_limits,
            closed: AtomicBool::new(false),
            reset_requested: AtomicBool::new(false),
            state: Mutex::new(State::Idle(Box::new(Idle {
                native: NativeOwner::new(native),
                owner: Box::new(owner),
                next_request_sequence: 1,
                game: None,
                custody: IdleCustody::StartupReady,
            }))),
            latest: Mutex::new(None),
            latest_initial: Mutex::new(None),
        })
    }
    pub fn latest_root(&self) -> Result<Option<Arc<RoventPcRootReceipt>>, ContractError> {
        self.latest
            .lock()
            .map(|value| value.clone())
            .map_err(|_| poisoned())
    }
    pub fn latest_initial(&self) -> Result<Option<Arc<RoventPcInitialReceipt>>, ContractError> {
        self.latest_initial
            .lock()
            .map(|value| value.clone())
            .map_err(|_| poisoned())
    }
    fn run_root(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        if self.closed.load(Ordering::Acquire) {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let cap = NonZeroU16::new(self.configuration.frontier.maximum_goal_plies)
            .expect("admitted nonzero goal cap");
        context
            .goal
            .validate(cap)
            .map_err(|e| SearchSessionFailure::debug("RoventPcGoal", &e))?;
        if context.authority.implementation != self.implementation
            || context.authority.epoch != self.clock.epoch()
        {
            return Err(failure(
                "RoventPcAuthority",
                "foreign original process authority",
            ));
        }
        let legal = position.legal_moves();
        if context.root_moves.as_ref().is_some_and(|moves| {
            moves.is_empty() || moves.iter().any(|movement| !legal.contains(movement))
        }) {
            return Err(failure(
                "RoventPcRootRestriction",
                "invalid original root restriction",
            ));
        }
        let source = SourceSeal::prepare(
            self.sources.lock(),
            self.root_policy,
            position,
            context,
            &self.clock,
        )
        .map_err(|error| SearchSessionFailure::debug("RoventPcOriginalRoot", &error))?;
        let mut state = loop {
            match self.state.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::Poisoned(_)) => {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
                Err(TryLockError::WouldBlock) => {
                    if Instant::now() >= context.shutdown_deadline {
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                    std::thread::yield_now();
                }
            }
        };
        if self.closed.load(Ordering::Acquire) {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        // Hold the same actual idle owner through the final source gate. A
        // stopped Start consumes neither reset intent nor root registration.
        let idle = match &*state {
            State::Idle(idle)
                if idle.native.finished.is_none()
                    && !idle.native.joined
                    && !idle.native.service.is_quarantined() =>
            {
                idle
            }
            _ => return Err(SearchSessionFailure::physical_completion_unknown()),
        };
        let root_admission = match source
            .into_initial()
            .map_err(|error| SearchSessionFailure::debug("RoventPcOriginalRoot", &error))?
        {
            InitialAdmissionOutcome::Admitted(admitted) => Arc::new(admitted),
            InitialAdmissionOutcome::StoppedInitial(stopped) => {
                let receipt = Arc::new(RoventPcInitialReceipt {
                    authority: context.authority,
                    input_lock: self.snapshot.input_lock,
                    source: Arc::new(stopped),
                    idle_custody: idle.custody.clone(),
                });
                let report = receipt.report();
                match self.latest_initial.lock() {
                    Ok(mut latest) => *latest = Some(receipt),
                    Err(_) => {
                        self.closed.store(true, Ordering::Release);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                }
                return Ok(report);
            }
        };
        let retained = std::mem::replace(&mut *state, State::Moving);
        let mut idle = match retained {
            State::Idle(idle) => idle,
            other => {
                *state = other;
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if idle.native.finished.is_some() || idle.native.joined {
            *state = State::Idle(idle);
            self.closed.store(true, Ordering::Release);
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let reset = self.reset_requested.swap(false, Ordering::AcqRel)
            || idle.game != Some(context.authority.game.0);
        if reset {
            if let Err(error) = idle.owner.try_adaptive_new_game(
                context.authority,
                context.control.deadline,
                &context.control.cancellation,
            ) {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcNewGame", &error));
            }
            idle.game = Some(context.authority.game.0);
        }
        let registration = match idle
            .owner
            .register_adaptive_root(position, context.authority)
        {
            Ok(value) => value,
            Err(error) => {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug(
                    "RoventPcRootRegistration",
                    &error,
                ));
            }
        };
        let original_deadline = match self.clock.deadline(context.control.deadline) {
            Ok(value) => value,
            Err(error) => {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcClock", &error));
            }
        };
        let Idle {
            native,
            owner,
            next_request_sequence,
            game: _,
            custody: _,
        } = *idle;
        let mut port = match PcSemanticPort::new_registered_retaining(
            *owner,
            position.clone(),
            context.authority,
            original_deadline,
            context.control.deadline,
            self.snapshot.mode,
            Arc::clone(&self.issuer),
            registration,
            context.root_moves.as_deref(),
        ) {
            Ok(port) => port,
            Err((owner, error)) => {
                *state = State::RejectedOwner {
                    native,
                    owner: Box::new(owner),
                };
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcPort", &error));
            }
        };
        let model_namespace = match port.model_namespace() {
            Ok(namespace) => Arc::new(namespace.clone()),
            Err(error) => {
                *state = State::RejectedPort {
                    native,
                    port: PcPortDriver::new(port),
                };
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcNamespace", &error));
            }
        };
        let frontier = FrontierRegistration {
            authority: context.authority,
            mode: self.snapshot.mode,
            original_deadline,
            root_budget: root_admission.root_budget_digest(),
            goal: context.goal,
            nodes: context.requested_nodes,
            selective_policy: self.selective_policy,
            root_rules: port.root_rules(),
            root_moves: Arc::clone(port.root_move_domain()),
            root_state: port.root_state_id(),
            model_namespace,
            native_ledger: None,
        };
        let root_budget = frontier.root_budget;
        let attached = port.attach_rovent_frontier(
            frontier,
            self.frontier_limits.clone(),
            PcFrontierStartup {
                root_budget,
                clock: self.clock.b_clock(),
            },
        );
        if let Err(error) = attached {
            *state = State::RejectedPort {
                native,
                port: PcPortDriver::new(port),
            };
            self.closed.store(true, Ordering::Release);
            return Err(SearchSessionFailure::debug("RoventPcFrontier", &error));
        }
        let sources = RoleOnlyManagerSources {
            native: native.service,
            driver: PcPortDriver::new(port),
        };
        let registration = RoleOnlyManagerRegistration {
            authority: context.authority,
            mode: self.snapshot.mode,
            root_budget,
            hard_deadline: context.control.deadline,
            token_budget: self.configuration.review.token_budget,
            first_request_sequence: next_request_sequence,
            role_tickets_max: self.configuration.execution.role_tickets_max(),
            role_call_limit: self.configuration.semantic.max_role_calls,
            event_order: self.configuration.execution.event_order().actual(),
        };
        let mut manager =
            match Manager::new_retaining(registration, self.clock.runtime_clock(), sources) {
                Ok(value) => Box::new(value),
                Err(rejected) => {
                    *state = State::RejectedPort {
                        native: NativeOwner::new(rejected.sources.native),
                        port: rejected.sources.driver,
                    };
                    self.closed.store(true, Ordering::Release);
                    return Err(SearchSessionFailure::debug(
                        "RoventPcManagerAdmission",
                        &rejected.error,
                    ));
                }
            };
        let run = catch_unwind(AssertUnwindSafe(
            || -> Result<(), crate::pals_adaptive::AdaptiveManagerError> {
                manager.register_cleanup_deadline(context.shutdown_deadline)?;
                if manager.registered_cleanup_deadline() != Some(context.shutdown_deadline) {
                    return Err(crate::pals_adaptive::AdaptiveManagerError::Invalid(
                        "P/C original cleanup boundary",
                    ));
                }
                let scanned = manager.driver_mut().port_mut().prepare_rovent_root_rules();
                if let Err(error) = scanned {
                    let goal = manager
                        .driver()
                        .port()
                        .rovent_frontier_receipt()
                        .map(|receipt| receipt.snapshot.goal);
                    if !matches!(
                        (error, goal),
                        (
                            FrontierError::NodeBudget,
                            Some(FrontierGoalStatus::Stopped(FrontierStop::WorkBudget))
                        ) | (
                            FrontierError::Deadline,
                            Some(FrontierGoalStatus::Stopped(FrontierStop::Deadline))
                        )
                    ) {
                        return Err(error.into());
                    }
                }
                let close_at = context
                    .control
                    .soft_deadline
                    .min(context.control.admission_deadline);
                let mut published = None;
                loop {
                    let signal = manager.activity_signal();
                    let observed = signal.version();
                    if manager.phase() == ManagerPhase::Running {
                        if !context.accepts().unwrap_or(false) || Instant::now() >= close_at {
                            manager.begin_root_quiesce(context.shutdown_deadline)?;
                        } else {
                            let actual = manager.driver().port().rovent_frontier_receipt().ok_or(
                                crate::pals_adaptive::AdaptiveManagerError::Invalid(
                                    "P/C actual frontier absent",
                                ),
                            )?;
                            match actual.snapshot.goal {
                                FrontierGoalStatus::DepthSatisfied
                                | FrontierGoalStatus::MateSatisfied => {
                                    manager.begin_goal_quiesce(context.shutdown_deadline)?
                                }
                                FrontierGoalStatus::GoalUnresolved
                                | FrontierGoalStatus::Stopped(
                                    FrontierStop::WorkBudget
                                    | FrontierStop::Deadline
                                    | FrontierStop::Canceled,
                                ) => manager.begin_root_quiesce(context.shutdown_deadline)?,
                                FrontierGoalStatus::Stopped(_) => {
                                    return Err(
                                        crate::pals_adaptive::AdaptiveManagerError::PhysicalUnknown,
                                    );
                                }
                                FrontierGoalStatus::Pending => {}
                            }
                        }
                    }
                    let changed = manager.tick()?;
                    if context.accepts().unwrap_or(false) {
                        if let Some(movement) = manager.driver().result().best_move {
                            if !legal.contains(&movement)
                                || context
                                    .root_moves
                                    .as_ref()
                                    .is_some_and(|roots| !roots.contains(&movement))
                            {
                                return Err(crate::pals_adaptive::AdaptiveManagerError::Invalid(
                                    "P/C progress escaped original root restriction",
                                ));
                            }
                            if published != Some(movement) {
                                progress(movement);
                                published = Some(movement);
                            }
                        }
                    }
                    if manager.phase() == ManagerPhase::Finished {
                        return Ok(());
                    }
                    if manager.phase() == ManagerPhase::Quarantined
                        || Instant::now() >= context.shutdown_deadline
                    {
                        return Err(crate::pals_adaptive::AdaptiveManagerError::PhysicalUnknown);
                    }
                    if !changed {
                        signal.wait_changed(
                            observed,
                            context
                                .shutdown_deadline
                                .saturating_duration_since(Instant::now())
                                .min(Duration::from_millis(10)),
                        );
                    }
                }
            },
        ));
        if !matches!(run, Ok(Ok(()))) {
            *state = State::Manager(manager);
            self.closed.store(true, Ordering::Release);
            return Err(match run {
                Ok(Err(error)) => {
                    let mut failure = SearchSessionFailure::debug("RoventPcRoot", &error);
                    failure.physical_completion = DriverPhysicalCompletion::Unknown;
                    failure
                }
                _ => SearchSessionFailure::physical_completion_unknown(),
            });
        }
        let physical = match manager.shutdown_receipt() {
            Ok(receipt) if receipt.permits_semantic_closure() => receipt,
            _ => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        let result = manager.driver().result();
        let next = match physical.request_sequence_high_water.checked_add(1) {
            Some(next) => next,
            None => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(failure(
                    "RoventPcSequence",
                    "actual role request sequence exhausted",
                ));
            }
        };
        let frontier = match manager.driver_mut().port_mut().retire_rovent_frontier() {
            Ok(Some(receipt)) => receipt.snapshot,
            _ => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        let receipt = Arc::new(RoventPcRootReceipt {
            domain: ROVENT_PC_ROOT_DOMAIN,
            authority: context.authority,
            input_lock: self.snapshot.input_lock,
            root_admission,
            frontier,
            physical: physical.clone(),
        });
        let work = match receipt.work() {
            Ok(work) => work,
            Err(error) => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(error);
            }
        };
        // The actual return, not a later global latest observation, owns the
        // same immutable checked strategy through UCI physical publication.
        let mate_strategy = receipt
            .frontier
            .mate_strategy
            .clone()
            .filter(|strategy| result.best_move == Some(strategy.best_move()));
        let sources = match manager.into_sources() {
            Ok(sources) => sources,
            Err((retained, _)) => {
                *state = State::Manager(Box::new(retained));
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        let owner = match sources.driver.into_port(&physical) {
            Ok(port) => port.into_owner(),
            Err((port, _)) => {
                *state = State::RejectedPort {
                    native: NativeOwner::new(sources.native),
                    port,
                };
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        *state = State::Idle(Box::new(Idle {
            native: NativeOwner::new(sources.native),
            owner: Box::new(owner),
            next_request_sequence: next,
            game: Some(context.authority.game.0),
            custody: IdleCustody::RetiredRoot(Arc::clone(&receipt)),
        }));
        match self.latest.lock() {
            Ok(mut latest) => *latest = Some(receipt),
            Err(_) => {
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        }
        if result.best_move.is_some_and(|movement| {
            !legal.contains(&movement)
                || context
                    .root_moves
                    .as_ref()
                    .is_some_and(|moves| !moves.contains(&movement))
        }) {
            self.closed.store(true, Ordering::Release);
            return Err(failure(
                "RoventPcRootRestriction",
                "actual result escaped original root restriction",
            ));
        }
        Ok(SearchSessionReport {
            best_move: result.best_move,
            work,
            mate_strategy,
        })
    }
    fn finish(&self, until: Instant) -> Result<(), SearchSessionFailure> {
        self.closed.store(true, Ordering::Release);
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let mut retained = std::mem::replace(&mut *state, State::Moving);
        loop {
            match retained {
                State::Closed => {
                    *state = State::Closed;
                    return Ok(());
                }
                State::Idle(mut idle) => {
                    let result = idle.native.finish(until);
                    if idle.native.joined && idle.native.finished == Some(true) {
                        *state = State::Closed;
                    } else {
                        *state = State::Idle(idle);
                    }
                    return result;
                }
                State::RejectedOwner { mut native, owner } => {
                    let result = native.finish(until);
                    if native.joined && native.finished == Some(true) {
                        *state = State::Closed;
                    } else {
                        *state = State::RejectedOwner { native, owner };
                    }
                    return result;
                }
                State::RejectedPort { mut native, port } => {
                    let result = native.finish(until);
                    if native.joined && native.finished == Some(true) {
                        *state = State::Closed;
                    } else {
                        *state = State::RejectedPort { native, port };
                    }
                    return result;
                }
                State::Manager(mut manager) => {
                    if manager.phase() != ManagerPhase::Finished {
                        if manager.begin_shutdown(until).is_err()
                            || !manager
                                .run_root_until(until, &AtomicBool::new(false))
                                .is_ok_and(|receipt| receipt.permits_semantic_closure())
                        {
                            *state = State::Manager(manager);
                            return Err(SearchSessionFailure::physical_completion_unknown());
                        }
                    }
                    let receipt = match manager.shutdown_receipt() {
                        Ok(value) if value.permits_semantic_closure() => value,
                        _ => {
                            *state = State::Manager(manager);
                            return Err(SearchSessionFailure::physical_completion_unknown());
                        }
                    };
                    let sources = match manager.into_sources() {
                        Ok(value) => value,
                        Err((manager, _)) => {
                            *state = State::Manager(Box::new(manager));
                            return Err(SearchSessionFailure::physical_completion_unknown());
                        }
                    };
                    let mut native = NativeOwner::new(sources.native);
                    native.finished = receipt.native_physical_shutdown_confirmed.then_some(true);
                    native.joined = receipt.native_process_joined;
                    // Process shutdown does not reuse the semantic owner. Keep its
                    // actual port and every view lease until the native owner joins.
                    retained = State::RejectedPort {
                        native,
                        port: sources.driver,
                    };
                }
                State::Moving => {
                    *state = State::Moving;
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        }
    }
}
impl SearchSessionDriver for RoventPcSessionDriver {
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        crate::bridge::NodeBudgetSemantics::RoventPositionVisits
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        let max = self.search_goal_capability().ok_or_else(poisoned)?;
        crate::rovent_root_admission::preadmission::validate_go(
            position,
            go,
            self.sources.lock(),
            self.root_policy,
            max,
        )
        .map_err(|error| {
            ContractError::new(
                rz_contracts::ErrorCode::InvalidInput,
                rz_contracts::Stage::Admission,
                error.detail,
            )
        })?;
        Ok(())
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.identity.clone()
    }
    fn implementation(&self) -> Digest {
        self.implementation
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        NonZeroU16::new(self.configuration.frontier.maximum_goal_plies)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(ContractError::new(
                rz_contracts::ErrorCode::BackendFailure,
                rz_contracts::Stage::Output,
                "Rovent P/C physical owner is closed or quarantined",
            ));
        }
        self.reset_requested.store(true, Ordering::Release);
        Ok(())
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        self.finish(until)?;
        Ok(Some(DriverPhysicalCompletion::Confirmed))
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.run_root(position, context, progress)
    }
}
impl Drop for RoventPcSessionDriver {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        if !matches!(state, State::Closed) {
            let retained = std::mem::replace(state, State::Moving);
            std::mem::forget(retained);
            // Preserve the original descriptors with the quarantined physical
            // owner instead of releasing startup provenance on an unknown drop.
            std::mem::forget(Arc::clone(&self.sources));
        }
    }
}
fn failure(code: &'static str, detail: &str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}
fn poisoned() -> ContractError {
    ContractError::new(
        rz_contracts::ErrorCode::BackendFailure,
        rz_contracts::Stage::Output,
        "Rovent P/C observation mutex poisoned",
    )
}
