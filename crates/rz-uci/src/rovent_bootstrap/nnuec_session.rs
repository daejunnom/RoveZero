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
    ) -> RootReturn {
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
        let accepted = match context.accepts() {
            Ok(accepted) => accepted,
            Err(error) => {
                let detail = error.to_string();
                return Err(self.retain(
                    (native, root, actor, error),
                    "RoventNnuecInitialControl",
                    &detail,
                ));
            }
        };
        if !accepted || context.soft_expired(Instant::now()) {
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
            return self.finish_preparing_retirement(
                native,
                retired,
                actor,
                next_request_sequence,
                game,
                context,
                admission,
                None,
            );
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
            if initial.phase() != InitialNnuecPhase::AdoptPort {
                let accepted = match context.accepts() {
                    Ok(accepted) => accepted,
                    Err(error) => {
                        let detail = error.to_string();
                        return Err(self.retain(
                            (native, initial.into_cleanup(), error),
                            "RoventNnuecInitialControl",
                            &detail,
                        ));
                    }
                };
                if !accepted || context.soft_expired(Instant::now()) {
                    return self.cleanup_initial(
                        native,
                        initial.into_cleanup(),
                        context,
                        admission,
                        next_request_sequence,
                        game,
                        "RoventNnuecInitialStopped",
                    );
                }
            }
            let observed = signal.version();
            match initial.step() {
                InitialNnuecStep::Pending(pending) => initial = pending,
                InitialNnuecStep::Ready(ready) => break ready,
                InitialNnuecStep::RetainedFailure(failed) => {
                    return self.cleanup_initial(
                        native,
                        failed.into_cleanup(),
                        context,
                        admission,
                        next_request_sequence,
                        game,
                        "RoventNnuecInitialPreparation",
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
        let root_moves = Arc::clone(&custody.root_moves);
        let (ledger, root_budget, cleanup_deadline, pins) =
            NnuecRootRetirementPins::from_custody(custody, metadata);
        let accepted = match context.accepts() {
            Ok(accepted) => accepted,
            Err(error) => {
                let detail = error.to_string();
                return Err(self.retain(
                    (native, port, worker, actor, pins, ledger, root_moves, error),
                    "RoventNnuecAdoptedControl",
                    &detail,
                ));
            }
        };
        if !accepted || context.soft_expired(Instant::now()) {
            return self.cleanup_adopted_initial(
                native,
                port,
                worker,
                actor,
                pins,
                ledger,
                root_budget,
                cleanup_deadline,
                root_moves,
                next_request_sequence,
                game,
                context,
                admission,
            );
        }
        let mut driver = match AdaptivePortDriver::new_with_coordinator_policy_retaining(
            port,
            context.control.deadline,
            Arc::new(signal.clone()),
            ExternalSlicePolicy::default(),
            self.policy,
        ) {
            Ok(d) => d,
            Err((port, AdaptiveManagerError::Deadline)) => {
                // The original hard boundary can race the prior live gate.
                // The returned original port follows cleanup, never a fresh
                // driver, deadline, worker, or manager construction.
                return self.cleanup_adopted_initial(
                    native,
                    port,
                    worker,
                    actor,
                    pins,
                    ledger,
                    root_budget,
                    cleanup_deadline,
                    root_moves,
                    next_request_sequence,
                    game,
                    context,
                    admission,
                );
            }
            Err((port, e)) => {
                return Err(self.retain(
                    (native, port, worker, actor, pins, ledger),
                    "RoventNnuecPort",
                    &e.to_string(),
                ));
            }
        };
        if let Some(observer) = &self.collection_hooks.external_teacher {
            if let Err(error) = driver.set_external_teacher_observer(Arc::clone(observer)) {
                return Err(self.retain(
                    (native, driver, worker, actor, pins, ledger),
                    "RoventNnuecTeacherObserver",
                    &error.to_string(),
                ));
            }
        }
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
        let mut collection_admitted = false;
        let execution = catch_unwind(AssertUnwindSafe(|| -> Result<(), AdaptiveManagerError> {
            if manager.registered_cleanup_deadline() != Some(context.shutdown_deadline) {
                return Err(AdaptiveManagerError::Invalid(
                    "NNUEC original cleanup boundary differs",
                ));
            }
            if let Some(clock) = &context.effective_clock {
                manager.register_effective_clock(clock.clone())?;
            }
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
            if context.accepts().unwrap_or(false) && !context.soft_expired(Instant::now()) {
                if let Some(origin) = &self.collection_origin {
                    let frontier = manager
                        .driver()
                        .port()
                        .and_then(|port| port.rovent_frontier())
                        .ok_or(AdaptiveManagerError::Invalid(
                            "actual NNUEC collection frontier absent",
                        ))?;
                    let cap = RoventCollectionRootAdmission::from_attached_frontier(
                        origin,
                        frontier,
                        self.snapshot.input_lock,
                        root_budget,
                        self.selective_policy,
                        Arc::clone(&admission),
                    )?;
                    manager.admit_collection_root(cap)?;
                    if let Some(observer) = &self.collection_hooks.root_admission {
                        observer(&mut manager)?;
                    }
                    collection_admitted = true;
                }
            }
            // A completed initial Ready is still not permission to create a
            // fresh native input after original UCI admission has closed.
            if context.accepts().unwrap_or(false)
                && !context.soft_expired(Instant::now())
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
                            || context.soft_expired(Instant::now())
                            || should_quiesce(manager)?)
                    {
                        manager.begin_root_quiesce(context.shutdown_deadline)?;
                    }
                    if collection_admitted {
                        if let Some(pump) = &self.collection_hooks.cooperative {
                            pump()?;
                        }
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
        if collection_admitted {
            if let Some(observer) = &self.collection_hooks.root_closure {
                match catch_unwind(AssertUnwindSafe(|| observer(&mut manager, result.completion))) {
                    Ok(Ok(())) => {},
                    Ok(Err(error)) => {
                        let detail = error.to_string();
                        return Err(self.retain(
                            (manager, actor, pins, error),
                            "RoventNnuecCollectionClosure",
                            &detail,
                        ));
                    }
                    Err(payload) => return Err(self.retain(
                        (manager, actor, pins, payload),
                        "RoventNnuecCollectionClosurePanic",
                        "root closure panicked; original pending utility and physical owners retained",
                    )),
                }
            }
        }
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
            initial_retirement: None,
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
        // A manager root fence transfers pending cost custody only. Its final
        // completion must still consume the same actor's engine/cache ACKs.
        let utility_route = match manager.take_utility_route_retirement() {
            Ok(pending) => pending,
            Err(error) => {
                let detail = error.to_string();
                return Err(self.retain(
                    (manager, actor, pins, error),
                    "RoventNnuecUtilityRetirement",
                    &detail,
                ));
            }
        };
        let retirement_deadline = context.shutdown_deadline;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let retirement_deadline = utility_route
            .as_ref()
            .and_then(|pending| pending.cohort_cleanup_deadline())
            .map_or(retirement_deadline, |cutoff| {
                retirement_deadline.min(cutoff)
            });
        let mut parts = match manager.into_nnuec_root_parts() {
            Ok(p) => p,
            Err(r) => {
                let detail = r.error.to_string();
                return Err(self.retain(
                    (r, actor, pins, utility_route),
                    "RoventNnuecRootTransfer",
                    &detail,
                ));
            }
        };
        if parts.cpu.len() != 1 || parts.external.is_some() {
            return Err(self.retain(
                (parts, actor, pins, utility_route),
                "RoventNnuecCpuTransfer",
                "actual researcher owner set differs",
            ));
        }
        let next = parts.continuation.first_request_sequence();
        let port = match parts.driver.into_port(&parts.receipt) {
            Ok(p) => p,
            Err((driver, e)) => {
                let detail = e.to_string();
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
                        utility_route,
                        e,
                    ),
                    "RoventNnuecSemanticFence",
                    &detail,
                ));
            }
        };
        let mut retirement = NnuecRootRetirement::new(
            actor,
            parts.cpu.pop().expect("one actual joined researcher"),
            pins,
            port.into_engine(),
        );
        if let Some(pending) = utility_route {
            if let Err((error, pending)) = retirement.attach_utility_route_pending(pending) {
                return Err(self.retain(
                    (parts.native, retirement, pending, error),
                    "RoventNnuecUtilityRetirement",
                    "original utility pending owner was rejected before same-actor retirement",
                ));
            }
        }
        let mut retired = loop {
            if Instant::now() >= retirement_deadline {
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
                    let detail = format!(
                        "same-actor release failed in {:?}: {}; original owners retained",
                        failed.phase(),
                        failed.error(),
                    );
                    return Err(self.retain(
                        (parts.native, failed),
                        "RoventNnuecRetirement",
                        &detail,
                    ));
                }
            }
            signal.wait_changed(
                observed,
                retirement_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        };
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if usize::from(retired.utility_route.is_some())
            + usize::from(retired.utility_cohort.is_some())
            + usize::from(retired.utility_intervention.is_some())
            > 1
        {
            return Err(self.retain(
                (parts.native, retired, receipt),
                "RoventNnuecUtilityRetirementKind",
                "one original retirement returned both root-local and cohort outputs",
            ));
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let Some(trial) = retired.utility_intervention.take() {
            if !collection_admitted
                || trial.authority() != context.authority
                || !trial.matches_root_admission(&receipt.root_admission)
            {
                return Err(self.retain(
                    (parts.native, retired, trial, receipt),
                    "RoventNnuecInterventionTrialScope",
                    "/2 retirement differs from original Collection root",
                ));
            }
            let Some(observer) = &self.collection_hooks.utility_intervention_trial_retired else {
                return Err(self.retain(
                    (parts.native, retired, trial, receipt),
                    "RoventNnuecInterventionTrialObserverMissing",
                    "actual /2 retirement has no original final consumer",
                ));
            };
            match catch_unwind(AssertUnwindSafe(|| observer(&trial))) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    let detail = error.to_string();
                    return Err(self.retain(
                        (parts.native, retired, trial, receipt, error),
                        "RoventNnuecInterventionTrialPublication",
                        &detail,
                    ));
                }
                Err(payload) => {
                    return Err(self.retain(
                        (parts.native, retired, trial, receipt, payload),
                        "RoventNnuecInterventionTrialPublicationPanic",
                        "/2 consumer panicked; original cap and actual owners retained",
                    ));
                }
            }
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let Some(trial) = retired.utility_cohort.take() {
            // This is the physical cap produced from the original cohort
            // admission and actual RetiredResearcherOwner. A root-local
            // Completed action never supplies this independent trial source.
            if !collection_admitted
                || trial.authority() != context.authority
                || !trial.matches_root_admission(&receipt.root_admission)
            {
                return Err(self.retain(
                    (parts.native, retired, trial, receipt),
                    "RoventNnuecUtilityTrialScope",
                    "cohort retirement differs from the original collection root admission",
                ));
            }
            let Some(observer) = &self.collection_hooks.utility_trial_retired else {
                return Err(self.retain(
                    (parts.native, retired, trial, receipt),
                    "RoventNnuecUtilityTrialObserverMissing",
                    "actual cohort retirement has no final trial consumer",
                ));
            };
            match catch_unwind(AssertUnwindSafe(|| observer(&trial))) {
                Ok(Ok(())) => {},
                Ok(Err(error)) => {
                    let detail = error.to_string();
                    return Err(self.retain(
                        (parts.native, retired, trial, receipt, error),
                        "RoventNnuecUtilityTrialPublication",
                        &detail,
                    ));
                }
                Err(payload) => return Err(self.retain(
                    (parts.native, retired, trial, receipt, payload),
                    "RoventNnuecUtilityTrialPublicationPanic",
                    "cohort trial consumer panicked; original physical cap and actual owners retained",
                )),
            }
        }
        if let Some(completed) = retired.utility_route.take() {
            // Cloning this historical Arc grants no execution or preparation
            // authority. Keep the original cap even if its consumer fails.
            let completed = Arc::new(completed);
            if !collection_admitted
                || completed.id().authority() != context.authority
                || completed.semantic().root_budget() != receipt.root_admission.root_budget_digest()
            {
                return Err(self.retain(
                    (parts.native, retired, completed, receipt),
                    "RoventNnuecUtilityScope",
                    "completed utility action differs from the original admitted collection root",
                ));
            }
            let Some(observer) = &self.collection_hooks.utility_route_completed else {
                return Err(self.retain(
                    (parts.native, retired, completed, receipt),
                    "RoventNnuecUtilityObserverMissing",
                    "actual utility completion has no final collection consumer",
                ));
            };
            match catch_unwind(AssertUnwindSafe(|| observer(&completed))) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    let detail = error.to_string();
                    return Err(self.retain(
                        (parts.native, retired, completed, receipt, error),
                        "RoventNnuecUtilityPublication",
                        &detail,
                    ));
                }
                Err(payload) => return Err(self.retain(
                    (parts.native, retired, completed, receipt, payload),
                    "RoventNnuecUtilityPublicationPanic",
                    "final utility consumer panicked; original action and actual owners retained",
                )),
            }
        }
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
