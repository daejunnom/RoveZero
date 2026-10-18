//! Actual native P/C/R NNUEC session, with deferred first C/TT/worker ownership.
//!
//! The loaded model and original source descriptors survive every root. A
//! preparing semantic owner is not an R engine: only the first admitted C can
//! create the session's preparation actor and materialize its real researcher.
//! Unknown transfers retain all original owners rather than replacing them.

mod initial_cleanup;
mod retirement;
mod root_run;

use super::configuration::{
    ExecutionConfiguration, ResearchValueConfiguration, StartupConfigurationDocument,
};
use super::nnuec_preparing::{
    InitialNnuecFailure, InitialNnuecPhase, InitialNnuecPreparing, InitialNnuecReady,
    InitialNnuecStep,
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
    AdaptiveManagerError, AdaptiveManagerRegistration, AdaptiveManagerSources,
    AdaptiveNnuecRetiredOwner, AdaptivePortDriver, AdaptiveRegisteredLedgerSources,
    AdaptiveSessionCollectionHooks, AdaptiveShutdownReceipt, ManagerPhase,
    RoventCollectionRootAdmission, RoventNativeCollectionOrigin,
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
use initial_cleanup::{
    NnuecInitialCleaned, NnuecInitialCleanup, NnuecInitialCleanupObservations,
    NnuecInitialCleanupStep,
};
use retirement::{NnuecRootRetirement, NnuecRootRetirementPins, NnuecRootRetirementStep};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::{ContractError, Digest};
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_runtime::contracts::ContractClock;
use rz_search::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck;
use rz_search::cpu_value::nnuec::v2::review::{PreparationActor, PreparationError};
use rz_search::cpu_worker::{
    CpuCreditLedger, CpuWorker, CpuWorkerError, DeferredNnuecCpuPlan, DeferredNnuecRSource,
    NnuecAuthorityRevocation,
};
use rz_search::external_cpu::{CpuEndpointSelection, ExternalSlicePolicy};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveEnginePort, AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::engine::preparing_root::{
    InitialResearcherReleaseReceipt, ResearcherPreparingError, ResearcherPreparingOwner,
    ResearcherRootStartup, RetiredPreparingRoot,
};
use rz_search::pals::engine::{
    PalsCompletion, PalsConfig, PalsLimits, ResearcherSemanticProfile, ResolverPolicy,
};
use rz_search::pals::rovent_engine::RootMoveDomain;
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
pub type RoventNnuecCollectionHooks = AdaptiveSessionCollectionHooks<Backend>;
type RootReturn = Result<
    (Box<Idle>, SearchSessionReport, Arc<RoventNnuecRootReceipt>),
    (State, SearchSessionFailure),
>;

pub const ROVENT_NNUEC_ROOT_DOMAIN: &str = "rz-rovent-nnuec-native-root/1";
pub const ROVENT_NNUEC_INITIAL_DOMAIN: &str = "rz-rovent-nnuec-native-initial-stop/1";

/// Producer-only original cleanup witnesses, distinct from a manager root
/// fence. They grant no search result, native execution, or new Ready owner.
pub struct RoventNnuecInitialRetirementReceipt {
    original_phase: InitialNnuecPhase,
    original_failure: Option<InitialNnuecFailure>,
    release: Option<InitialResearcherReleaseReceipt>,
    cache_release: NnuecCacheReleaseAck,
    observations: NnuecInitialCleanupObservations,
}
impl std::fmt::Debug for RoventNnuecInitialRetirementReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventNnuecInitialRetirementReceipt")
            .field("original_phase", &self.original_phase)
            .field("original_failure", &self.original_failure)
            .field("has_original_release", &self.release.is_some())
            .field("cache_release", &self.cache_release)
            .finish_non_exhaustive()
    }
}
impl RoventNnuecInitialRetirementReceipt {
    pub fn original_phase(&self) -> InitialNnuecPhase {
        self.original_phase
    }
    pub fn original_failure(&self) -> Option<&InitialNnuecFailure> {
        self.original_failure.as_ref()
    }
    pub fn release(&self) -> Option<&InitialResearcherReleaseReceipt> {
        self.release.as_ref()
    }
    pub fn cache_release(&self) -> &NnuecCacheReleaseAck {
        &self.cache_release
    }
    pub fn observations(&self) -> &NnuecInitialCleanupObservations {
        &self.observations
    }
}

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
    pub initial_retirement: Option<Arc<RoventNnuecInitialRetirementReceipt>>,
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
    collection_origin: Option<RoventNativeCollectionOrigin>,
    collection_hooks: RoventNnuecCollectionHooks,
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
        Self::new_with_collection_hooks(
            prepared,
            loaded,
            clock,
            owners,
            identity,
            root_policy,
            None,
        )
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new_with_collection_hooks(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: Option<RoventNnuecCollectionHooks>,
    ) -> Result<Self, RoventNnuecSessionRejection> {
        let built = (|| -> Result<_, SearchSessionFailure> {
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
            let collection_origin = match (
                prepared.sources().lock().manifest.purpose,
                &collection_hooks,
            ) {
                (rz_experiments::RoventPurpose::Collection, Some(hooks))
                    if hooks.root_admission.is_some() =>
                {
                    Some(prepared.collection_identity().map_err(|error| {
                        SearchSessionFailure::debug("RoventNnuecCollectionOrigin", &error)
                    })?)
                }
                (rz_experiments::RoventPurpose::Collection, _) | (_, Some(_)) => {
                    return Err(failure(
                        "RoventNnuecCollectionHooks",
                        "explicit Collection and actual root observer must be selected together",
                    ));
                }
                (_, None) => None,
            };
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
            let model = AdaptivePortModel::new_with_namespace(
                prepared.snapshot().role_model_identity(),
                Some(
                    prepared
                        .snapshot()
                        .model_value_identity()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?,
                ),
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
            hash.update(include_bytes!("nnuec_session/root_run.rs"));
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
                collection_origin,
                Digest(hash.finalize().into()),
                Digest(
                    document
                        .search_configuration_digest()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecConfiguration", &e))?,
                ),
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
            collection_origin,
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
            collection_origin,
            collection_hooks: collection_hooks.unwrap_or_default(),
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

    /// Original read-only Frontier/work observation. A later owner transfer
    /// or same-actor retirement may still fail; this is not run-success or a
    /// substitute for the actual returned report and physical completion.
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

    /// Preserve the original observation before any later work projection or
    /// owner transfer can fail. This is neither a successful root return nor
    /// the actor/engine retirement ACK. On poison the caller still owns this
    /// same Arc and must retain it with the original physical owners.
    fn observe_root_receipt(
        &self,
        receipt: &Arc<RoventNnuecRootReceipt>,
    ) -> Result<(), SearchSessionFailure> {
        let mut latest = self.latest.lock().map_err(|_| {
            unknown(
                "RoventNnuecRootObservation",
                "original root receipt observation owner is poisoned",
            )
        })?;
        *latest = Some(Arc::clone(receipt));
        Ok(())
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

    // Keep the original native/cleanup owners, context/admission, sequence and
    // game identity separate while applying their existing cleanup gate.
    #[allow(clippy::too_many_arguments)]
    fn cleanup_initial(
        &self,
        native: NativeOwner,
        cleanup: super::nnuec_preparing::InitialNnuecCleanup,
        context: &SearchSessionContext,
        admission: Arc<RoventRootAdmission>,
        next_request_sequence: u64,
        game: Option<u64>,
        code: &'static str,
    ) -> RootReturn {
        let signal = native.service.signal();
        // Capture the original caller gate before cleanup revokes that same
        // root's cancellation owner. Cleanup itself cannot invent a stop.
        let original_control = context.accepts();
        let original_stop =
            matches!(original_control, Ok(false)) || context.soft_expired(Instant::now());
        let control_error = original_control.err();
        let mut cleanup = NnuecInitialCleanup::new(cleanup);
        loop {
            if Instant::now() >= context.shutdown_deadline {
                return Err(self.retain(
                    (native, cleanup, control_error),
                    code,
                    "original preparation cleanup deadline expired; exact owners retained",
                ));
            }
            let observed = signal.version();
            match cleanup.step() {
                NnuecInitialCleanupStep::Pending(pending) => cleanup = pending,
                NnuecInitialCleanupStep::Ready(cleaned) => {
                    if !original_stop
                        || control_error.is_some()
                        || !matches!(
                            cleaned.original_failure.as_ref(),
                            None | Some(InitialNnuecFailure::Preparation(
                                PreparationError::Cancelled | PreparationError::Expired
                            )) | Some(InitialNnuecFailure::Worker(
                                CpuWorkerError::NnuecAuthorityRevoked(
                                    NnuecAuthorityRevocation::Cancelled
                                        | NnuecAuthorityRevocation::Expired
                                )
                            ))
                        )
                    {
                        return Err(self.retain(
                            (native, cleaned, control_error),
                            code,
                            "actual initial preparation failure is not a normal control stop",
                        ));
                    }
                    let NnuecInitialCleaned {
                        retired,
                        actor,
                        original_phase,
                        original_failure,
                        release,
                        cache_release,
                        observations,
                    } = cleaned;
                    let proof = Arc::new(RoventNnuecInitialRetirementReceipt {
                        original_phase,
                        original_failure,
                        release,
                        cache_release,
                        observations,
                    });
                    return self.finish_preparing_retirement(
                        native,
                        retired,
                        Some(actor),
                        next_request_sequence,
                        game,
                        context,
                        admission,
                        Some(proof),
                    );
                }
                NnuecInitialCleanupStep::Retained(retained) => {
                    return Err(self.retain(
                        (native, retained, control_error),
                        code,
                        "actual physical cleanup observed; original semantic owners remain quarantined",
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
        }
    }

    /// Only an actual preparing-root retirement issues this move-only owner.
    /// This path never constructs a manager, native ticket, R grant or worker.
    #[allow(clippy::too_many_arguments)]
    fn finish_preparing_retirement(
        &self,
        native: NativeOwner,
        retired: RetiredPreparingRoot<AdaptivePortModel>,
        actor: Option<PreparationActor>,
        next_request_sequence: u64,
        game: Option<u64>,
        context: &SearchSessionContext,
        admission: Arc<RoventRootAdmission>,
        initial_retirement: Option<Arc<RoventNnuecInitialRetirementReceipt>>,
    ) -> RootReturn {
        let original_ledger = retired.ledger.registration_observation();
        if retired.root_budget != admission.root_budget_digest()
            || retired.cleanup_deadline != context.shutdown_deadline
            || !matches!(original_ledger, Ok(observed)
                if observed.authority == context.authority
                    && observed.clock_domain == self.clock.domain()
                    && observed.clock_origin == self.clock.runtime_clock().origin()
                    && observed.native_node_limit == self.max_native_nodes
                    && observed.spent_native_nodes == 0
                    && observed.reserved_native_nodes == 0
                    && observed.active_jobs == 0
                    && observed.outstanding_executions == 0
                    && observed.quarantined_jobs == 0)
        {
            return Err(self.retain(
                (native, retired, actor, initial_retirement, original_ledger),
                "RoventNnuecInitialRetirement",
                "actual initial retirement differs from the original root/ledger controls",
            ));
        }
        let receipt = Arc::new(RoventNnuecRootReceipt {
            domain: ROVENT_NNUEC_ROOT_DOMAIN,
            authority: context.authority,
            input_lock: self.snapshot.input_lock,
            root_admission: admission.clone(),
            frontier: retired.frontier.map(|r| r.snapshot),
            physical: None,
            initial_retirement,
            counters: AdaptiveManagerCounters::default(),
            stopped_before_research_execution: true,
            deadline_before_frontier: retired.deadline_before_frontier,
        });
        if let Err(error) = self.observe_root_receipt(&receipt) {
            return Err(self.retain(
                (
                    native,
                    retired.owner,
                    actor,
                    retired.ledger,
                    retired.root_rules,
                    retired.root_moves,
                    receipt,
                ),
                "RoventNnuecRootObservation",
                &error.detail,
            ));
        }
        let work = match receipt.work() {
            Ok(work) => work,
            Err(error) => {
                return Err(self.retain(
                    (
                        native,
                        retired.owner,
                        actor,
                        retired.ledger,
                        retired.root_rules,
                        retired.root_moves,
                        receipt,
                    ),
                    "RoventNnuecInitialWork",
                    &error.detail,
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
        Ok((
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
        ))
    }

    /// The initial handoff already owns a real Ready worker. A control stop
    /// here needs no expired live-driver constructor and no manager admission.
    #[allow(clippy::too_many_arguments)]
    fn cleanup_adopted_initial(
        &self,
        native: NativeOwner,
        mut port: AdaptiveEnginePort<AdaptivePortModel>,
        mut worker: CpuWorker,
        actor: PreparationActor,
        pins: NnuecRootRetirementPins,
        ledger: CpuCreditLedger,
        root_budget: Digest,
        cleanup_deadline: Instant,
        root_moves: Arc<RootMoveDomain>,
        next_request_sequence: u64,
        game: Option<u64>,
        context: &SearchSessionContext,
        admission: Arc<RoventRootAdmission>,
    ) -> RootReturn {
        let signal = native.service.signal();
        let root_rules = port.root_rules();
        context.control.cancellation.store(true, Ordering::Release);
        match port.with_rovent_frontier(|frontier, _| {
            frontier.cancel();
            Ok(())
        }) {
            Ok(Some(())) => {}
            other => {
                return Err(self.retain(
                    (native, port, worker, actor, pins, ledger, root_moves, other),
                    "RoventNnuecAdoptedCancel",
                    "original adopted frontier cancellation was not observed",
                ));
            }
        }
        worker.request_shutdown();
        let shutdown = loop {
            if Instant::now() >= context.shutdown_deadline {
                return Err(self.retain(
                    (native, port, worker, actor, pins, ledger, root_moves),
                    "RoventNnuecAdoptedJoin",
                    "original cleanup deadline expired before actual Ready worker join",
                ));
            }
            let observed = signal.version();
            match worker.try_join() {
                Ok(Some(shutdown)) => break shutdown,
                Ok(None) => {}
                Err(error) => {
                    let detail = format!("{error:?}");
                    return Err(self.retain(
                        (native, port, worker, actor, pins, ledger, root_moves, error),
                        "RoventNnuecAdoptedJoin",
                        &detail,
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
        if !initial_cleanup::permits_initial_reuse(&shutdown) {
            return Err(self.retain(
                (
                    native, port, worker, actor, pins, ledger, root_moves, shutdown,
                ),
                "RoventNnuecAdoptedStartup",
                "actual initial Ready/affinity/join readbacks do not permit reuse",
            ));
        }
        let recovered = match worker.take_nnuec_retirement() {
            Ok(engine) => engine,
            Err(error) => {
                let detail = format!("{error:?}");
                return Err(self.retain(
                    (
                        native, port, worker, actor, pins, ledger, root_moves, shutdown, error,
                    ),
                    "RoventNnuecAdoptedEngine",
                    &detail,
                ));
            }
        };
        let frontier = match port.retire_rovent_frontier() {
            Ok(Some(frontier)) => frontier,
            other => {
                return Err(self.retain(
                    (
                        native, port, worker, actor, pins, ledger, root_moves, shutdown, recovered,
                        other,
                    ),
                    "RoventNnuecAdoptedFrontier",
                    "original adopted frontier retirement was not observed",
                ));
            }
        };
        let mut retirement = NnuecRootRetirement::new(
            actor,
            AdaptiveNnuecRetiredOwner {
                engine: recovered,
                last_installation: None,
            },
            pins,
            port.into_engine(),
        );
        let retired = loop {
            if Instant::now() >= context.shutdown_deadline {
                return Err(self.retain(
                    (
                        native, retirement, worker, ledger, root_moves, shutdown, frontier,
                    ),
                    "RoventNnuecAdoptedRelease",
                    "original cleanup deadline expired before actual same-actor release",
                ));
            }
            let observed = signal.version();
            match retirement.step() {
                NnuecRootRetirementStep::Pending(pending) => retirement = pending,
                NnuecRootRetirementStep::Ready(retired) => break retired,
                NnuecRootRetirementStep::RetainedFailure(failed) => {
                    let detail = format!(
                        "actual initial engine/cache release failed in {:?}: {}; original owners retained",
                        failed.phase(),
                        failed.error(),
                    );
                    return Err(self.retain(
                        (
                            native, failed, worker, ledger, root_moves, shutdown, frontier,
                        ),
                        "RoventNnuecAdoptedRelease",
                        &detail,
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
        let observations = NnuecInitialCleanupObservations {
            root_cancel_requested: true,
            ready_shutdown_requested: true,
            ready_shutdown: Some(shutdown),
            engine_release_matches_original: Some(
                retired
                    .engine_release
                    .matches_source(retired.owner.source()),
            ),
            cache_clears_released_engine: Some(
                retired
                    .engine_release
                    .matches_cache_release(&retired.cache_release),
            ),
            ..Default::default()
        };
        let proof = Arc::new(RoventNnuecInitialRetirementReceipt {
            original_phase: InitialNnuecPhase::AdoptPort,
            original_failure: None,
            release: Some(InitialResearcherReleaseReceipt::Worker(
                retired.engine_release,
            )),
            cache_release: retired.cache_release,
            observations,
        });
        self.finish_preparing_retirement(
            native,
            RetiredPreparingRoot {
                owner: retired.owner,
                ledger,
                root_budget,
                frontier: Some(frontier),
                deadline_before_frontier: false,
                root_rules,
                root_moves,
                cleanup_deadline,
            },
            Some(retired.actor),
            next_request_sequence,
            game,
            context,
            admission,
            Some(proof),
        )
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
            Ok((idle, report, _receipt)) => {
                *state = State::Idle(idle);
                // Each actual receipt was already observed at its original
                // creation boundary, before fallible physical-owner transfers.
                Ok(report)
            }
            Err((retained, error)) => {
                *state = retained;
                Err(error)
            }
        }
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
    fn supports_ponder_clock(&self) -> bool {
        true
    }

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
            // The Frontier was already retired when this read-only receipt
            // was observed. Quarantining the physical owners must not drop
            // their sole original frontier/admission observation on the way
            // out. A normally joined Closed state releases it as before.
            if let Some(receipt) = self
                .latest
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take()
            {
                std::mem::forget(receipt);
            }
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
