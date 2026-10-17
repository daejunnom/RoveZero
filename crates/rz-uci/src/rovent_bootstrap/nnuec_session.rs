//! Actual native P/C/R NNUEC session, with deferred first C/TT/worker ownership.
//!
//! The loaded model and original source descriptors survive every root. A
//! preparing semantic owner is not an R engine: only the first admitted C can
//! create the session's preparation actor and materialize its real researcher.
//! Unknown transfers retain all original owners rather than replacing them.

mod initial_cleanup;
mod retirement;

use super::configuration::{
    ExecutionConfiguration, ResearchValueConfiguration, StartupConfigurationDocument,
};
use super::nnuec_preparing::{
    InitialNnuecPhase, InitialNnuecPreparing, InitialNnuecReady, InitialNnuecStep,
};
use super::nnuec_source::LoadedRoventNnuecSource;
use super::{
    PreparedRoventNative, RoventNativeBackend, RoventNativeService, RoventNativeStartupSnapshot,
    RoventPinnedNativeSources,
};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    AdaptiveExecutionLimits, AdaptiveExecutionManager, AdaptiveManagerCounters,
    AdaptiveManagerError, AdaptiveManagerRegistration, AdaptiveManagerSources, AdaptivePortDriver,
    AdaptiveRegisteredLedgerSources, AdaptiveShutdownReceipt, ManagerPhase,
};
use crate::pals_native::ticketed::{TicketNativeEvent, VirtualReviewTicketOutput};
use crate::rovent_root_admission::source_seal::{
    InitialAdmissionOutcome, SourceSeal, StoppedInitial,
};
use crate::rovent_root_admission::{RoventRootAdmission, RoventRootAdmissionPolicy};
use crate::search_driver::{
    DriverPhysicalCompletion, DriverWork, PalsCheckerRegistration, SearchKind,
    SearchSessionContext, SearchSessionDriver, SearchSessionFailure, SearchSessionReport,
};
use initial_cleanup::{NnuecInitialCleanup, NnuecInitialCleanupStep};
use retirement::{NnuecRootRetirement, NnuecRootRetirementPins, NnuecRootRetirementStep};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::{ContractError, Digest};
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_runtime::contracts::ContractClock;
use rz_search::cpu_value::nnuec::v2::review::PreparationActor;
use rz_search::cpu_worker::{DeferredNnuecCpuPlan, DeferredNnuecRSource};
use rz_search::external_cpu::{CpuEndpointSelection, ExternalSlicePolicy};
use rz_search::pals::engine::adaptive_port::{
    AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::engine::preparing_root::{
    ResearcherPreparingError, ResearcherPreparingOwner, ResearcherRootStartup,
};
use rz_search::pals::engine::{
    PalsCompletion, PalsConfig, PalsLimits, ResearcherSemanticProfile, ResolverPolicy,
};
use rz_search::pals::rovent_frontier::{
    FrontierError, FrontierGoalStatus, FrontierLimits, FrontierSnapshot, FrontierStop,
};
use sha2::{Digest as _, Sha256};
use std::{
    any::Any,
    num::NonZeroU16,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

type Backend = RoventNativeBackend<AdaptivePortQuestion>;
type Manager =
    AdaptiveExecutionManager<AdaptivePortQuestion, Backend, AdaptivePortDriver<AdaptivePortModel>>;
type SemanticOwner = ResearcherPreparingOwner<AdaptivePortModel>;

pub const ROVENT_NNUEC_ROOT_DOMAIN: &str = "rz-rovent-nnuec-native-root/1";
pub const ROVENT_NNUEC_INITIAL_DOMAIN: &str = "rz-rovent-nnuec-native-initial-stop/1";

/// Original semantic work and an optional actual manager researcher fence.
/// Initial C/TT cleanup is recorded separately from a manager root fence.
#[derive(Clone, Debug)]
pub struct RoventNnuecRootReceipt {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    pub root_admission: Arc<RoventRootAdmission>,
    pub frontier: Option<FrontierSnapshot>,
    pub physical: Option<AdaptiveShutdownReceipt>,
    pub counters: AdaptiveManagerCounters,
    pub stopped_before_research_execution: bool,
    pub deadline_before_frontier: bool,
}
impl RoventNnuecRootReceipt {
    pub fn work(&self) -> Result<DriverWork, SearchSessionFailure> {
        let Some(frontier) = &self.frontier else {
            if self.deadline_before_frontier
                && self.stopped_before_research_execution
                && self.physical.is_none()
            {
                return Ok(zero_work());
            }
            return Err(unknown(
                "RoventNnuecFrontier",
                "original frontier observation absent",
            ));
        };
        let total = if frontier.work.native_usage_known {
            Some(
                frontier
                    .work
                    .total_node_visits()
                    .ok_or_else(|| unknown("RoventNnuecWork", "original work overflow"))?,
            )
        } else {
            None
        };
        Ok(DriverWork::RoventNative {
            root_entries: frontier.work.root_entries,
            total_node_visits: total,
            checked_transitions: frontier.work.checked_transitions,
            native_nodes: frontier
                .work
                .native_usage_known
                .then_some(frontier.work.native_nodes),
            native_qsearch_nodes: frontier
                .work
                .native_usage_known
                .then_some(frontier.work.native_qsearch_nodes),
            submitted_role_tickets: self.counters.submitted_role_tickets,
            consumed_role_outputs: self.counters.consumed_role_outputs,
            completed_nn_inputs: (self.counters.nn_input_count_unknown == 0)
                .then_some(self.counters.completed_nn_inputs),
            consumed_nn_inputs: (self.counters.consumed_nn_input_count_unknown == 0)
                .then_some(self.counters.consumed_nn_inputs),
            completed_depth: frontier.horizon.completed_depth.map(|depth| depth.0),
            maximum_examined_ply: frontier.horizon.maximum_examined_ply.0,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RoventNnuecInitialReceipt {
    authority: SearchAuthority,
    input_lock: Digest,
    source: Arc<StoppedInitial>,
    previous_root: Option<Arc<RoventNnuecRootReceipt>>,
}
impl RoventNnuecInitialReceipt {
    pub fn domain(&self) -> &'static str {
        ROVENT_NNUEC_INITIAL_DOMAIN
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
    pub fn previous_clean_root(&self) -> Option<&Arc<RoventNnuecRootReceipt>> {
        self.previous_root.as_ref()
    }
    pub fn used_original_startup_ready(&self) -> bool {
        self.previous_root.is_none()
    }
}

struct NativeOwner {
    service: RoventNativeService<AdaptivePortQuestion>,
    finished: Option<bool>,
    joined: bool,
    unexpected: Vec<TicketNativeEvent<VirtualReviewTicketOutput>>,
}
impl NativeOwner {
    fn new(service: RoventNativeService<AdaptivePortQuestion>) -> Self {
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
                            Some(self.finished.unwrap_or(true) && physical_shutdown_confirmed)
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
                    Err(unknown(
                        "RoventNnuecNativeProtocol",
                        "unexpected native terminal retained",
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
struct Idle {
    native: NativeOwner,
    owner: Box<SemanticOwner>,
    actor: Option<PreparationActor>,
    next_request_sequence: u64,
    game: Option<u64>,
    previous_root: Option<Arc<RoventNnuecRootReceipt>>,
}
enum State {
    Idle(Box<Idle>),
    /// A move-only owner bundle. Its type has never implied a clean fence.
    Retained {
        _owners: Box<dyn Any + Send>,
    },
    Closed,
    Moving,
}

pub struct RoventNnuecSessionRejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNative<AdaptivePortQuestion>,
    pub loaded: LoadedRoventNnuecSource,
}

/// Dedicated NNUEC session: no bootstrap researcher or legacy V5 receipt.
pub struct RoventNnuecSessionDriver {
    sources: Arc<RoventPinnedNativeSources>,
    loaded: LoadedRoventNnuecSource,
    snapshot: RoventNativeStartupSnapshot,
    configuration: StartupConfigurationDocument,
    clock: ProcessClock,
    issuer: AdaptiveRulesOwnerIssuer,
    identity: EngineIdentity,
    implementation: Digest,
    root_policy: RoventRootAdmissionPolicy,
    selective_policy: Digest,
    frontier_limits: FrontierLimits,
    execution_limits: AdaptiveExecutionLimits,
    controller: ControllerConfig,
    policy: CoordinatorPolicy,
    max_rounds: u64,
    max_native_nodes: u64,
    cpu_depth: u16,
    closed: AtomicBool,
    reset_requested: AtomicBool,
    state: Mutex<State>,
    latest: Mutex<Option<Arc<RoventNnuecRootReceipt>>>,
    latest_initial: Mutex<Option<Arc<RoventNnuecInitialReceipt>>>,
    latest_checker: Mutex<Option<PalsCheckerRegistration>>,
}
impl RoventNnuecSessionDriver {
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventNnuecSessionRejection> {
        let built =
            (|| -> Result<_, SearchSessionFailure> {
                let document = prepared.sources().configuration();
                document
                    .require_supported_native_capabilities()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecCapability", &e))?;
                let fields = document.fields();
                let ExecutionConfiguration::OwnResearcher {
                    event_order,
                    queues,
                    max_rounds,
                    max_native_nodes,
                    cpu_depth,
                    cpu_nodes_per_task,
                    cpu_plan,
                    value,
                    controller,
                    ..
                } = fields.execution
                else {
                    return Err(failure(
                        "RoventNnuecStartup",
                        "actual own-R configuration required",
                    ));
                };
                if !matches!(value, ResearchValueConfiguration::NnuecV2 { .. })
                    || !prepared.snapshot().mode.has_native_r()
                    || prepared.snapshot().process_epoch != clock.epoch().0
                    || prepared.snapshot().input_lock != loaded.input_lock_digest()
                    || !loaded.matches_sources(prepared.sources())
                    || identity.name.is_empty()
                    || identity.author.is_empty()
                {
                    return Err(failure(
                        "RoventNnuecStartup",
                        "loaded NNUEC/source/process identity mismatch",
                    ));
                }
                if prepared.sources().lock().manifest.purpose
                    == rz_experiments::RoventPurpose::Collection
                {
                    return Err(failure(
                        "RoventNnuecCollectionHooks",
                        "actual NNUEC collection callbacks are not yet connected",
                    ));
                }
                let ready = prepared
                    .service
                    .ready_clock()
                    .ok_or_else(|| failure("RoventNnuecStartup", "actual native Ready absent"))?;
                let original = clock.runtime_clock();
                if prepared.service.is_quarantined()
                    || ready.domain() != original.domain()
                    || ready.origin() != original.origin()
                {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
                let control = controller
                    .actual(*event_order)
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecController", &e))?;
                // One initial Bound materializes exactly one physical researcher.
                // Wider declarations are rejected rather than silently resized.
                if control.max_workers != 1 || control.initial_workers != 1 {
                    return Err(failure(
                        "RoventNnuecWorkers",
                        "this same-source assembly requires one declared own worker",
                    ));
                }
                let limits = queues
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecQueues", &e))?;
                if cpu_plan.paused_bytes_limit != limits.paused_bytes_per_worker_max {
                    return Err(failure(
                        "RoventNnuecQueues",
                        "actual paused-owner budget differs from queues",
                    ));
                }
                let plan = DeferredNnuecCpuPlan::new(
                    cpu_plan
                        .actual()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecPlan", &e))?,
                    cpu_plan.ordering(),
                    cpu_plan.paused_bytes_limit,
                    cpu_plan.preserve_paused_side_scope,
                    limits.worker_commands_max,
                    cpu_plan.affinity(),
                )
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecPlan", &e))?;
                let source = DeferredNnuecRSource::new(Arc::clone(loaded.model()), plan);
                let model =
                    AdaptivePortModel::new_with_namespace(
                        prepared.snapshot().role_model_identity(),
                        Some(prepared.snapshot().model_value_identity().map_err(|e| {
                            SearchSessionFailure::debug("RoventNnuecNamespace", &e)
                        })?),
                        prepared
                            .snapshot()
                            .model_namespace()
                            .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?,
                    )
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?;
                let owner = ResearcherPreparingOwner::new_selected(
                    ResearcherSemanticProfile {
                        config: PalsConfig {
                            beam_width: fields.semantic.beam_width,
                            line_plies: fields.semantic.line_plies,
                            max_nodes: fields.semantic.max_nodes,
                            max_records: fields.semantic.max_records,
                            max_role_calls: fields.semantic.max_role_calls,
                            cpu_nodes_per_task: *cpu_nodes_per_task,
                        },
                        maximum_root_prefix_plies: fields.semantic.maximum_root_prefix_plies,
                    },
                    model,
                    ResolverPolicy::ModelWdlRestricted,
                    document.post_repair_policy_for(prepared.snapshot().mode),
                    fields
                        .review
                        .actual()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecReview", &e))?,
                    fields
                        .index
                        .actual()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecIndex", &e))?,
                    source,
                )
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecSemantic", &e))?;
                let mut hash = Sha256::new();
                hash.update(b"rz-rovent-nnuec-native-session/1\0");
                hash.update(super::compiled_bootstrap_source_digest());
                hash.update(include_bytes!("nnuec_session.rs"));
                hash.update(include_bytes!("nnuec_session/retirement.rs"));
                hash.update(include_bytes!("nnuec_session/initial_cleanup.rs"));
                hash.update(
                    document
                        .configuration_digest()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecConfiguration", &e))?,
                );
                hash.update([match root_policy {
                    RoventRootAdmissionPolicy::ExactRegistered => 0,
                    RoventRootAdmissionPolicy::DynamicGo => 1,
                }]);
                Ok((
                    owner,
                    Digest(hash.finalize().into()),
                    Digest(document.search_configuration_digest().map_err(|e| {
                        SearchSessionFailure::debug("RoventNnuecConfiguration", &e)
                    })?),
                    fields
                        .frontier
                        .actual()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecFrontier", &e))?,
                    limits,
                    control,
                    event_order.actual(),
                    *max_rounds,
                    *max_native_nodes,
                    *cpu_depth,
                ))
            })();
        let (
            owner,
            implementation,
            selective_policy,
            frontier_limits,
            execution_limits,
            controller,
            policy,
            max_rounds,
            max_native_nodes,
            cpu_depth,
        ) = match built {
            Ok(value) => value,
            Err(failure) => {
                return Err(RoventNnuecSessionRejection {
                    failure,
                    prepared,
                    loaded,
                });
            }
        };
        let configuration = prepared.sources().configuration().clone();
        let (native, snapshot, sources) = prepared.into_parts();
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        Ok(Self {
            sources,
            loaded,
            snapshot,
            configuration,
            clock,
            issuer,
            identity,
            implementation,
            root_policy,
            selective_policy,
            frontier_limits,
            execution_limits,
            controller,
            policy,
            max_rounds,
            max_native_nodes,
            cpu_depth,
            closed: AtomicBool::new(false),
            reset_requested: AtomicBool::new(false),
            state: Mutex::new(State::Idle(Box::new(Idle {
                native: NativeOwner::new(native),
                owner: Box::new(owner),
                actor: None,
                next_request_sequence: 1,
                game: None,
                previous_root: None,
            }))),
            latest: Mutex::new(None),
            latest_initial: Mutex::new(None),
            latest_checker: Mutex::new(None),
        })
    }

    pub fn latest_root(&self) -> Result<Option<Arc<RoventNnuecRootReceipt>>, ContractError> {
        self.latest
            .lock()
            .map(|r| r.clone())
            .map_err(|_| poisoned())
    }
    pub fn latest_initial(&self) -> Result<Option<Arc<RoventNnuecInitialReceipt>>, ContractError> {
        self.latest_initial
            .lock()
            .map(|r| r.clone())
            .map_err(|_| poisoned())
    }

    fn retain<T: Any + Send>(
        &self,
        owners: T,
        code: &'static str,
        detail: &str,
    ) -> (State, SearchSessionFailure) {
        self.closed.store(true, Ordering::Release);
        (
            State::Retained {
                _owners: Box::new(owners),
            },
            unknown(code, detail),
        )
    }

    fn cleanup_initial(
        &self,
        native: NativeOwner,
        cleanup: super::nnuec_preparing::InitialNnuecCleanup,
        context: &SearchSessionContext,
        code: &'static str,
    ) -> (State, SearchSessionFailure) {
        let signal = native.service.signal();
        let mut cleanup = NnuecInitialCleanup::new(cleanup);
        loop {
            if Instant::now() >= context.shutdown_deadline {
                return self.retain(
                    (native, cleanup),
                    code,
                    "original preparation cleanup deadline expired; exact owners retained",
                );
            }
            let observed = signal.version();
            match cleanup.step() {
                NnuecInitialCleanupStep::Pending(pending) => cleanup = pending,
                NnuecInitialCleanupStep::Retained(retained) => {
                    return self.retain(
                        (native, retained),
                        code,
                        "actual physical cleanup observed; original semantic owners remain quarantined",
                    );
                }
            }
            signal.wait_changed(
                observed,
                context
                    .shutdown_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        }
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
        context
            .goal
            .validate(self.frontier_limits.maximum_goal_plies)
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecGoal", &e))?;
        if context.authority.implementation != self.implementation
            || context.authority.epoch != self.clock.epoch()
        {
            return Err(failure(
                "RoventNnuecAuthority",
                "foreign original authority",
            ));
        }
        let source = SourceSeal::prepare(
            self.sources.lock(),
            self.root_policy,
            position,
            context,
            &self.clock,
        )
        .map_err(|e| SearchSessionFailure::debug("RoventNnuecOriginalRoot", &e))?;
        let mut state = loop {
            match self.state.try_lock() {
                Ok(s) => break s,
                Err(TryLockError::Poisoned(_)) => {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
                Err(TryLockError::WouldBlock) if Instant::now() < context.shutdown_deadline => {
                    std::thread::yield_now()
                }
                Err(_) => return Err(SearchSessionFailure::physical_completion_unknown()),
            }
        };
        let idle = match &*state {
            State::Idle(idle)
                if !self.closed.load(Ordering::Acquire)
                    && idle.native.finished.is_none()
                    && !idle.native.joined
                    && !idle.native.service.is_quarantined()
                    && idle.native.service.occupancy() == 0 =>
            {
                idle
            }
            _ => return Err(SearchSessionFailure::physical_completion_unknown()),
        };
        let admission = match source
            .into_initial()
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecOriginalRoot", &e))?
        {
            InitialAdmissionOutcome::Admitted(value) => Arc::new(value),
            InitialAdmissionOutcome::StoppedInitial(source) => {
                let receipt = Arc::new(RoventNnuecInitialReceipt {
                    authority: context.authority,
                    input_lock: self.snapshot.input_lock,
                    source: Arc::new(source),
                    previous_root: idle.previous_root.clone(),
                });
                let report = SearchSessionReport {
                    best_move: receipt.source.fallback(),
                    work: zero_work(),
                    mate_strategy: None,
                };
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
        let idle = match std::mem::replace(&mut *state, State::Moving) {
            State::Idle(idle) => idle,
            other => {
                *state = other;
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        match self.run_admitted(idle, position, context, admission, progress) {
            Ok((idle, report, receipt)) => {
                *state = State::Idle(idle);
                match self.latest.lock() {
                    Ok(mut latest) => *latest = Some(receipt),
                    Err(_) => {
                        self.closed.store(true, Ordering::Release);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                }
                Ok(report)
            }
            Err((retained, error)) => {
                *state = retained;
                Err(error)
            }
        }
    }

    fn run_admitted(
        &self,
        mut idle: Box<Idle>,
        position: &Position,
        context: &SearchSessionContext,
        admission: Arc<RoventRootAdmission>,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<
        (Box<Idle>, SearchSessionReport, Arc<RoventNnuecRootReceipt>),
        (State, SearchSessionFailure),
    > {
        let reset = self.reset_requested.swap(false, Ordering::AcqRel)
            || idle.game != Some(context.authority.game.0);
        if reset {
            if let Err(e) = idle.owner.try_adaptive_new_game(
                context.authority,
                context.control.deadline,
                &context.control.cancellation,
            ) {
                return Err(self.retain(idle, "RoventNnuecNewGame", &e.to_string()));
            }
            idle.game = Some(context.authority.game.0);
        }
        let deadline = match self.clock.deadline(context.control.deadline) {
            Ok(d) => d,
            Err(e) => return Err(self.retain(idle, "RoventNnuecClock", &e.to_string())),
        };
        let Idle {
            native,
            owner,
            mut actor,
            next_request_sequence,
            game,
            previous_root: _,
        } = *idle;
        let mut root = match (*owner).bind_root(
            position.clone(),
            context.authority,
            deadline,
            context.control.deadline,
            Arc::clone(&context.control.cancellation),
        ) {
            Ok(root) => root,
            Err((owner, e)) => {
                return Err(self.retain(
                    (native, owner, actor),
                    "RoventNnuecBindRoot",
                    &e.to_string(),
                ));
            }
        };
        if let Err(e) = root.attach_original_controls(
            ResearcherRootStartup {
                mode: self.snapshot.mode,
                root_budget: admission.root_budget_digest(),
                goal: context.goal,
                nodes: context.requested_nodes,
                selective_policy: self.selective_policy,
                frontier_limits: self.frontier_limits.clone(),
                clock: self.clock.b_clock(),
                limits: PalsLimits {
                    deadline: context.control.deadline,
                    max_rounds: self.max_rounds,
                    max_cpu_nodes: self.max_native_nodes,
                    cpu_depth: self.cpu_depth,
                },
                cleanup_deadline: context.shutdown_deadline,
            },
            Arc::clone(&self.issuer),
            context.root_moves.as_deref(),
        ) {
            return Err(self.retain((native, root, actor), "RoventNnuecControls", &e.to_string()));
        }
        if let Err(e) = root.prepare_original_root_rules() {
            let controlled = matches!(
                (
                    &e,
                    root.original_frontier_receipt().map(|r| r.snapshot.goal)
                ),
                (
                    ResearcherPreparingError::Frontier(FrontierError::NodeBudget),
                    Some(FrontierGoalStatus::Stopped(FrontierStop::WorkBudget))
                ) | (
                    ResearcherPreparingError::Frontier(FrontierError::Deadline),
                    Some(FrontierGoalStatus::Stopped(FrontierStop::Deadline))
                ) | (
                    ResearcherPreparingError::Frontier(FrontierError::Revoked),
                    Some(FrontierGoalStatus::Stopped(FrontierStop::Canceled))
                )
            );
            if !controlled {
                return Err(self.retain(
                    (native, root, actor),
                    "RoventNnuecRootRules",
                    &e.to_string(),
                ));
            }
        }
        // Every root control is consulted before actor creation and before C.
        if !context.accepts().unwrap_or(false)
            || Instant::now()
                >= context
                    .control
                    .soft_deadline
                    .min(context.control.admission_deadline)
        {
            root.cancel();
        }
        let terminal = root.deadline_before_frontier()
            || root
                .original_frontier_receipt()
                .is_some_and(|r| !matches!(r.snapshot.goal, FrontierGoalStatus::Pending));
        if terminal {
            let retired = match root.retire_before_materialization() {
                Ok(r) => r,
                Err((root, e)) => {
                    return Err(self.retain(
                        (native, root, actor),
                        "RoventNnuecInitialRetire",
                        &e.to_string(),
                    ));
                }
            };
            let receipt = Arc::new(RoventNnuecRootReceipt {
                domain: ROVENT_NNUEC_ROOT_DOMAIN,
                authority: context.authority,
                input_lock: self.snapshot.input_lock,
                root_admission: admission.clone(),
                frontier: retired.frontier.map(|r| r.snapshot),
                physical: None,
                counters: AdaptiveManagerCounters::default(),
                stopped_before_research_execution: true,
                deadline_before_frontier: retired.deadline_before_frontier,
            });
            let work = match receipt.work() {
                Ok(w) => w,
                Err(e) => {
                    return Err(self.retain(
                        (
                            native,
                            retired.owner,
                            actor,
                            retired.ledger,
                            retired.root_rules,
                            retired.root_moves,
                        ),
                        "RoventNnuecInitialWork",
                        &e.detail,
                    ));
                }
            };
            let mate_strategy = receipt
                .frontier
                .as_ref()
                .and_then(|f| f.mate_strategy.clone());
            let best_move = mate_strategy
                .as_ref()
                .map(|s| s.best_move())
                .or_else(|| admission.allowed_root_moves().first().copied());
            let report = SearchSessionReport {
                best_move,
                work,
                mate_strategy,
            };
            return Ok((
                Box::new(Idle {
                    native,
                    owner: Box::new(retired.owner),
                    actor,
                    next_request_sequence,
                    game,
                    previous_root: Some(receipt.clone()),
                }),
                report,
                receipt,
            ));
        }
        let signal = native.service.signal();
        if actor.is_none() {
            actor = match PreparationActor::spawn_notified(
                Arc::clone(self.loaded.model()),
                signal.clone(),
            ) {
                Ok(a) => Some(a),
                Err(e) => {
                    return Err(self.retain(
                        (native, root, actor),
                        "RoventNnuecActor",
                        &format!("{e:?}"),
                    ));
                }
            };
        }
        let mut initial = InitialNnuecPreparing::new(
            root,
            actor.take().expect("actual actor just admitted"),
            CpuWorkerId(0),
            signal.clone(),
        );
        let ready = loop {
            if initial.phase() != InitialNnuecPhase::AdoptPort
                && (!context.accepts().unwrap_or(false)
                    || Instant::now()
                        >= context
                            .control
                            .soft_deadline
                            .min(context.control.admission_deadline))
            {
                return Err(self.cleanup_initial(
                    native,
                    initial.into_cleanup(),
                    context,
                    "RoventNnuecInitialStopped",
                ));
            }
            let observed = signal.version();
            match initial.step() {
                InitialNnuecStep::Pending(pending) => initial = pending,
                InitialNnuecStep::Ready(ready) => break ready,
                InitialNnuecStep::RetainedFailure(failed) => {
                    return Err(self.cleanup_initial(
                        native,
                        failed.into_cleanup(),
                        context,
                        "RoventNnuecInitialPreparation",
                    ));
                }
            }
            signal.wait_changed(
                observed,
                context
                    .shutdown_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        };
        let InitialNnuecReady {
            port,
            custody,
            worker,
            actor,
            metadata,
        } = *ready;
        // The declaration is captured only from this actual materialized
        // engine's metadata and accepted Ready, never at deferred startup.
        let registration = PalsCheckerRegistration {
            identity: metadata.identity().clone(),
            conditions: metadata.conditions().to_owned(),
            capabilities: metadata.capabilities(),
            owned_descriptor: metadata.owned_descriptor(),
            cpu_resume_policy: metadata.resume_policy(),
            role_model: port.engine().model_identity().to_owned(),
            model_value: port.engine().model_value_identity().cloned(),
            resolver_version: rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
            resolver_semantics: rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
        };
        match self.latest_checker.lock() {
            Ok(mut latest) => *latest = Some(registration),
            Err(_) => {
                return Err(self.retain(
                    (native, port, custody, worker, actor, metadata),
                    "RoventNnuecMetadata",
                    "actual metadata observation mutex poisoned",
                ));
            }
        }
        let (ledger, root_budget, cleanup_deadline, pins) =
            NnuecRootRetirementPins::from_custody(custody, metadata);
        let driver = match AdaptivePortDriver::new_with_coordinator_policy_retaining(
            port,
            context.control.deadline,
            Arc::new(signal.clone()),
            ExternalSlicePolicy::default(),
            self.policy,
        ) {
            Ok(d) => d,
            Err((port, e)) => {
                return Err(self.retain(
                    (native, port, worker, actor, pins, ledger),
                    "RoventNnuecPort",
                    &e.to_string(),
                ));
            }
        };
        let registration = AdaptiveManagerRegistration {
            authority: context.authority,
            hard_deadline: context.control.deadline,
            token_budget: self.configuration.fields().review.token_budget,
            policy: self.policy,
            first_request_sequence: next_request_sequence,
            native_cpu_node_limit: self.max_native_nodes,
            endpoints: CpuEndpointSelection::deployment(),
            controller: self.controller,
            device_digest: self.snapshot.backend,
            cpu_device_digest: Digest(Sha256::digest(b"native-own-nnuec-v2-host-cpu").into()),
        };
        let sources = AdaptiveRegisteredLedgerSources {
            sources: AdaptiveManagerSources {
                native: native.service,
                own_workers: vec![worker],
                external: None,
                driver,
            },
            ledger,
            root_budget,
            cleanup_deadline,
        };
        let mut manager = match Manager::new_retaining_with_registered_ledger(
            registration,
            self.clock.runtime_clock(),
            self.execution_limits,
            sources,
        ) {
            Ok(m) => Box::new(m),
            Err(r) => {
                return Err(self.retain(
                    (r.original, actor, pins),
                    "RoventNnuecManager",
                    &r.error.to_string(),
                ));
            }
        };
        let execution = catch_unwind(AssertUnwindSafe(|| -> Result<(), AdaptiveManagerError> {
            manager.enable_nnuec_preparation(Arc::clone(self.loaded.model()))?;
            let fields = self.configuration.fields();
            if let ExecutionConfiguration::OwnResearcher {
                reciprocal, peers, ..
            } = fields.execution
            {
                manager.configure_rovent_reciprocal(
                    reciprocal
                        .actual(self.snapshot.mode, root_budget)
                        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?,
                )?;
                manager.configure_rovent_peer_followups(
                    peers
                        .actual()
                        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?,
                )?;
            }
            // A completed initial Ready is still not permission to create a
            // fresh native input after original UCI admission has closed.
            if context.accepts().unwrap_or(false)
                && Instant::now()
                    < context
                        .control
                        .soft_deadline
                        .min(context.control.admission_deadline)
                && !should_quiesce(&manager)?
            {
                let input = manager.driver_mut().root_input()?;
                manager.prepare_root_ranking(input)?;
            }
            let mut published = None;
            manager.run_root_until_observed(
                context.shutdown_deadline,
                &context.control.cancellation,
                &mut |manager| {
                    // Original UCI scope/cancel/deadline gate precedes all C pumps.
                    let accepts = context.accepts().unwrap_or(false);
                    if manager.phase() == ManagerPhase::Running
                        && (!accepts
                            || Instant::now()
                                >= context
                                    .control
                                    .soft_deadline
                                    .min(context.control.admission_deadline)
                            || should_quiesce(manager)?)
                    {
                        manager.begin_root_quiesce(context.shutdown_deadline)?;
                    }
                    manager.pump_nnuec_preparation(&actor)?;
                    if context.accepts().unwrap_or(false) {
                        if let Ok(result) = manager.driver().result(PalsCompletion::RoundLimit) {
                            if let Some(movement) = result.best_move {
                                if !admission.allowed_root_moves().contains(&movement) {
                                    return Err(AdaptiveManagerError::Invalid(
                                        "NNUEC progress escaped original root moves",
                                    ));
                                }
                                if published != Some(movement) {
                                    progress(movement);
                                    published = Some(movement);
                                }
                            }
                        }
                    }
                    Ok(())
                },
            )?;
            Ok(())
        }));
        match execution {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                let detail = error.to_string();
                return Err(self.retain(
                    (manager, actor, pins, error),
                    "RoventNnuecExecution",
                    &detail,
                ));
            }
            Err(payload) => {
                return Err(self.retain(
                    (manager, actor, pins, payload),
                    "RoventNnuecExecutionPanic",
                    "actual root execution panicked; original panic and physical owners retained",
                ));
            }
        }
        let physical = match manager.shutdown_receipt() {
            Ok(r) if r.permits_root_reuse() && manager.phase() == ManagerPhase::Finished => r,
            _ => {
                return Err(self.retain(
                    (manager, actor, pins),
                    "RoventNnuecFence",
                    "actual root fence absent",
                ));
            }
        };
        let completion = if context.cancellation.is_canceled()
            || context.control.cancellation.load(Ordering::Acquire)
        {
            PalsCompletion::Canceled
        } else if Instant::now() >= context.control.deadline {
            PalsCompletion::Deadline
        } else if manager.driver().root_is_terminal().unwrap_or(false) {
            PalsCompletion::Terminal
        } else if manager.cpu_node_limit_reached() {
            PalsCompletion::CpuNodeLimit
        } else if manager.role_call_limit_reached() {
            PalsCompletion::RoleCallLimit
        } else {
            PalsCompletion::RoundLimit
        };
        let result = match manager.driver().result(completion) {
            Ok(r) => r,
            Err(e) => {
                return Err(self.retain(
                    (manager, actor, pins),
                    "RoventNnuecResult",
                    &e.to_string(),
                ));
            }
        };
        let frontier = match manager.driver_mut().retire_rovent_frontier() {
            Ok(Some(r)) => r.snapshot,
            _ => {
                return Err(self.retain(
                    (manager, actor, pins),
                    "RoventNnuecFrontierRetire",
                    "actual frontier retirement absent",
                ));
            }
        };
        let receipt = Arc::new(RoventNnuecRootReceipt {
            domain: ROVENT_NNUEC_ROOT_DOMAIN,
            authority: context.authority,
            input_lock: self.snapshot.input_lock,
            root_admission: admission,
            frontier: Some(frontier),
            physical: Some(physical),
            counters: manager.counters(),
            stopped_before_research_execution: false,
            deadline_before_frontier: false,
        });
        let work = match receipt.work() {
            Ok(w) => w,
            Err(e) => {
                return Err(self.retain((manager, actor, pins), "RoventNnuecWork", &e.detail));
            }
        };
        if result
            .best_move
            .is_some_and(|m| !receipt.root_admission.allowed_root_moves().contains(&m))
        {
            return Err(self.retain(
                (manager, actor, pins),
                "RoventNnuecRootRestriction",
                "actual result escaped original move domain",
            ));
        }
        let mut parts = match manager.into_nnuec_root_parts() {
            Ok(p) => p,
            Err(r) => {
                return Err(self.retain(
                    (r.manager, actor, pins),
                    "RoventNnuecRootTransfer",
                    &r.error.to_string(),
                ));
            }
        };
        if parts.cpu.len() != 1 || parts.external.is_some() {
            return Err(self.retain(
                (parts, actor, pins),
                "RoventNnuecCpuTransfer",
                "actual researcher owner set differs",
            ));
        }
        let next = parts.continuation.first_request_sequence();
        let port = match parts.driver.into_port(&parts.receipt) {
            Ok(p) => p,
            Err((driver, e)) => {
                return Err(self.retain(
                    (
                        parts.native,
                        parts.cpu,
                        parts.external,
                        parts.continuation,
                        parts.receipt,
                        driver,
                        actor,
                        pins,
                    ),
                    "RoventNnuecSemanticFence",
                    &e.to_string(),
                ));
            }
        };
        let mut retirement = NnuecRootRetirement::new(
            actor,
            parts.cpu.pop().expect("one actual joined researcher"),
            pins,
            port.into_engine(),
        );
        let retired = loop {
            if Instant::now() >= context.shutdown_deadline {
                return Err(self.retain(
                    (parts.native, retirement),
                    "RoventNnuecRetirementDeadline",
                    "original cleanup deadline expired before same-actor release",
                ));
            }
            let observed = signal.version();
            match retirement.step() {
                NnuecRootRetirementStep::Pending(pending) => retirement = pending,
                NnuecRootRetirementStep::Ready(retired) => break retired,
                NnuecRootRetirementStep::RetainedFailure(failed) => {
                    return Err(self.retain(
                        (parts.native, failed),
                        "RoventNnuecRetirement",
                        "same-actor release failed; original owners retained",
                    ));
                }
            }
            signal.wait_changed(
                observed,
                context
                    .shutdown_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        };
        let mate_strategy = receipt
            .frontier
            .as_ref()
            .and_then(|f| f.mate_strategy.clone())
            .filter(|s| result.best_move == Some(s.best_move()));
        Ok((
            Box::new(Idle {
                native: NativeOwner::new(parts.native),
                owner: Box::new(retired.owner),
                actor: Some(retired.actor),
                next_request_sequence: next,
                game,
                previous_root: Some(receipt.clone()),
            }),
            SearchSessionReport {
                best_move: result.best_move,
                work,
                mate_strategy,
            },
            receipt,
        ))
    }

    fn finish(&self, until: Instant) -> Result<(), SearchSessionFailure> {
        self.closed.store(true, Ordering::Release);
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        if matches!(*state, State::Closed) {
            return Ok(());
        }
        let idle = match &mut *state {
            State::Idle(idle) => idle,
            _ => return Err(SearchSessionFailure::physical_completion_unknown()),
        };
        if let Some(actor) = &mut idle.actor {
            actor.request_shutdown();
            loop {
                match actor.try_join() {
                    Ok(Some(())) => break,
                    Ok(None) if Instant::now() < until => std::thread::yield_now(),
                    _ => return Err(SearchSessionFailure::physical_completion_unknown()),
                }
            }
            idle.actor.take();
        }
        idle.native.finish(until)?;
        if idle.native.joined && idle.native.finished == Some(true) {
            *state = State::Closed;
            Ok(())
        } else {
            Err(SearchSessionFailure::physical_completion_unknown())
        }
    }
}

impl SearchSessionDriver for RoventNnuecSessionDriver {
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        crate::bridge::NodeBudgetSemantics::RoventPositionVisits
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        crate::rovent_root_admission::preadmission::validate_go(
            position,
            go,
            self.sources.lock(),
            self.root_policy,
            self.frontier_limits.maximum_goal_plies,
        )
        .map_err(|e| {
            ContractError::new(
                rz_contracts::ErrorCode::InvalidInput,
                rz_contracts::Stage::Admission,
                e.detail,
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
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        self.latest_checker
            .lock()
            .ok()
            .and_then(|registration| registration.clone())
    }
    fn checker_started(&self) -> Option<bool> {
        self.latest_checker
            .lock()
            .ok()
            .map(|registration| registration.is_some())
    }
    fn implementation(&self) -> Digest {
        self.implementation
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        Some(self.frontier_limits.maximum_goal_plies)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(poisoned());
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
impl Drop for RoventNnuecSessionDriver {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|p| p.into_inner());
        if !matches!(state, State::Closed) {
            std::mem::forget(std::mem::replace(state, State::Moving));
            std::mem::forget(Arc::clone(&self.sources));
            std::mem::forget(Arc::clone(self.loaded.model()));
        }
    }
}
fn should_quiesce(manager: &Manager) -> Result<bool, AdaptiveManagerError> {
    let receipt =
        manager
            .driver()
            .rovent_frontier_receipt()?
            .ok_or(AdaptiveManagerError::Invalid(
                "actual NNUEC frontier absent",
            ))?;
    match receipt.snapshot.goal {
        FrontierGoalStatus::Pending => Ok(false),
        FrontierGoalStatus::DepthSatisfied
        | FrontierGoalStatus::MateSatisfied
        | FrontierGoalStatus::GoalUnresolved
        | FrontierGoalStatus::Stopped(
            FrontierStop::Canceled | FrontierStop::Deadline | FrontierStop::WorkBudget,
        ) => Ok(true),
        _ => Err(AdaptiveManagerError::PhysicalUnknown),
    }
}
fn zero_work() -> DriverWork {
    DriverWork::RoventNative {
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
    }
}
fn failure(code: &'static str, detail: &str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}
fn unknown(code: &'static str, detail: &str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Unknown,
        code,
        detail: detail.into(),
    }
}
fn poisoned() -> ContractError {
    ContractError::new(
        rz_contracts::ErrorCode::BackendFailure,
        rz_contracts::Stage::Output,
        "Rovent NNUEC owner closed or observation poisoned",
    )
}
