//! Native P/C product owner. No researcher, PVS engine, CPU credit ledger or
//! checker metadata exists in this state machine.

use super::configuration::{ExecutionConfiguration, StartupConfigurationDocument};
use super::nnuec_source::LoadedRoventNnuecSource;
use super::{
    PreparedRoventNative, RoventNativeBackend, RoventNativeService, RoventNativeStartupSnapshot,
    RoventPinnedNativeSources,
};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    AdaptiveManagerError, ManagerPhase, PcPortDriver, RoleOnlyExecutionManager,
    RoleOnlyManagerRegistration, RoleOnlyManagerSources, RoleOnlyShutdownReceipt,
    RoventCollectionRootAdmission, RoventNativeCollectionOrigin,
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
use rz_search::cpu_value::nnuec::v2::review::PreparationActor;
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
pub type RoventPcSessionManager =
    RoleOnlyExecutionManager<PcPortQuestion, Backend, PcPortDriver<Model>>;
type Manager = RoventPcSessionManager;
pub type RoventPcAdmissionObserver = Arc<
    dyn Fn(
            &mut RoventPcSessionManager,
            RoventCollectionRootAdmission,
        ) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;
pub type RoventPcClosureObserver =
    Arc<dyn Fn(&mut RoventPcSessionManager) -> Result<(), AdaptiveManagerError> + Send + Sync>;
#[derive(Default)]
pub struct RoventPcCollectionHooks {
    pub root_admission: Option<RoventPcAdmissionObserver>,
    pub root_closure: Option<RoventPcClosureObserver>,
    pub cooperative: Option<Arc<dyn Fn() -> Result<(), AdaptiveManagerError> + Send + Sync>>,
}

pub const ROVENT_PC_ROOT_DOMAIN: &str = "rz-rovent-pc-native-root/1";
pub const ROVENT_PC_INITIAL_DOMAIN: &str = "rz-rovent-pc-native-initial-stop/1";
pub struct RoventPcRootReceipt {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    /// Original startup lock and each go's actual workload are distinct seals.
    pub root_admission: Arc<RoventRootAdmission>,
    pub frontier: FrontierSnapshot,
    pub physical: RoleOnlyShutdownReceipt,
    pub repair: Option<rz_search::pals::engine::pc_port::PcRepairReport>,
}
impl std::fmt::Debug for RoventPcRootReceipt {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("RoventPcRootReceipt")
            .field("domain", &self.domain)
            .field("authority", &self.authority)
            .field("input_lock", &self.input_lock)
            .field("root_admission", &self.root_admission)
            .field("frontier", &self.frontier)
            .field("physical", &self.physical)
            .field("repair_present", &self.repair.is_some())
            .finish()
    }
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
// One mutex-protected session slot owns the current native/semantic custody.
// RejectedPort retains that exact port until actual cleanup, not in a queue.
#[allow(clippy::large_enum_variant)]
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
/// No preparation thread is started before this constructor has succeeded.
/// Rejection therefore returns both original startup owners without a join.
pub struct RoventPcSharedNnuecSessionRejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNative<PcPortQuestion>,
    pub loaded: LoadedRoventNnuecSource,
}
struct ConstructionRejection {
    failure: SearchSessionFailure,
    prepared: PreparedRoventNative<PcPortQuestion>,
    loaded: Option<LoadedRoventNnuecSource>,
}
/// Exactly one optional RuntimeHelper; no R engine or native node owner.
struct SharedNnuecOwner {
    loaded: LoadedRoventNnuecSource,
    actor: Mutex<SharedNnuecActor>,
}
/// Readback only. An actual thread ID is neither an affinity claim nor a
/// preparation, model-consumption or physical-closure capability.
#[derive(Clone, Debug, Default)]
pub struct RoventPcSharedNnuecReadback {
    pub spawned_runtime_helpers: u8,
    pub worker_thread_id: Option<std::thread::ThreadId>,
    pub worker_joined: bool,
    /// No per-thread OS affinity readback is performed by this owner.
    pub worker_thread_affinity: Option<Vec<u32>>,
}
#[derive(Default)]
struct SharedNnuecActor {
    actor: Option<PreparationActor>,
    signal: rz_runtime::CompletionSignal,
    readback: RoventPcSharedNnuecReadback,
}
impl SharedNnuecActor {
    fn finish(&mut self, until: Instant) -> Result<(), SearchSessionFailure> {
        let Some(actor) = self.actor.as_mut() else {
            return if self.readback.spawned_runtime_helpers == 0 || self.readback.worker_joined {
                Ok(())
            } else {
                Err(SearchSessionFailure::physical_completion_unknown())
            };
        };
        actor.request_shutdown();
        loop {
            let observed = self.signal.version();
            match actor.try_join() {
                Ok(Some(())) => {
                    self.readback.worker_joined = true;
                    self.actor.take();
                    return Ok(());
                }
                Ok(None) if Instant::now() < until => {
                    // The actor notifies for real commands. Thread exit has
                    // no separate notification, so bounded polling still
                    // checks the original join without a busy-spin loop.
                    self.signal.wait_changed(
                        observed,
                        until
                            .saturating_duration_since(Instant::now())
                            .min(Duration::from_millis(10)),
                    );
                }
                _ => return Err(SearchSessionFailure::physical_completion_unknown()),
            }
        }
    }
}
pub struct RoventPcSessionDriver {
    sources: Arc<RoventPinnedNativeSources>,
    snapshot: RoventNativeStartupSnapshot,
    configuration: StartupConfigurationDocument,
    clock: ProcessClock,
    issuer: AdaptiveRulesOwnerIssuer,
    identity: EngineIdentity,
    implementation: Digest,
    root_policy: RoventRootAdmissionPolicy,
    collection_origin: Option<RoventNativeCollectionOrigin>,
    collection_hooks: RoventPcCollectionHooks,
    selective_policy: Digest,
    frontier_limits: FrontierLimits,
    closed: AtomicBool,
    reset_requested: AtomicBool,
    state: Mutex<State>,
    latest: Mutex<Option<Arc<RoventPcRootReceipt>>>,
    latest_initial: Mutex<Option<Arc<RoventPcInitialReceipt>>>,
    /// A report already taken from the port survives any later failed transfer.
    /// The exclusive root owner permits at most one such in-progress report.
    retained_repair: Mutex<Option<rz_search::pals::engine::pc_port::PcRepairReport>>,
    shared_nnuec: Option<SharedNnuecOwner>,
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
        Self::new_with_collection_hooks(
            prepared,
            clock,
            owners,
            identity,
            root_policy,
            RoventPcCollectionHooks::default(),
        )
    }
    #[allow(clippy::result_large_err)]
    pub fn new_with_collection_hooks(
        prepared: PreparedRoventNative<PcPortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: RoventPcCollectionHooks,
    ) -> Result<Self, RoventPcSessionRejection> {
        Self::construct(
            prepared,
            None,
            clock,
            owners,
            identity,
            root_policy,
            collection_hooks,
        )
        .map_err(|rejected| RoventPcSessionRejection {
            failure: rejected.failure,
            prepared: rejected.prepared,
        })
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn from_prepared_with_shared_nnuec(
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventPcSharedNnuecSessionRejection> {
        Self::from_prepared_with_shared_nnuec_and_collection_hooks(
            prepared,
            loaded,
            clock,
            owners,
            identity,
            root_policy,
            RoventPcCollectionHooks::default(),
        )
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn from_prepared_with_shared_nnuec_and_collection_hooks(
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: RoventPcCollectionHooks,
    ) -> Result<Self, RoventPcSharedNnuecSessionRejection> {
        Self::construct(
            prepared,
            Some(loaded),
            clock,
            owners,
            identity,
            root_policy,
            collection_hooks,
        )
        .map_err(|rejected| RoventPcSharedNnuecSessionRejection {
            failure: rejected.failure,
            prepared: rejected.prepared,
            loaded: rejected.loaded.expect("original shared source retained"),
        })
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn construct(
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: Option<LoadedRoventNnuecSource>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: RoventPcCollectionHooks,
    ) -> Result<Self, ConstructionRejection> {
        let build = (|| -> Result<_, SearchSessionFailure> {
            let configuration = prepared.sources().configuration().clone();
            let collection_origin = match prepared.sources().lock().manifest.purpose {
                rz_experiments::RoventPurpose::Collection
                    if collection_hooks.root_admission.is_some() =>
                {
                    Some(prepared.collection_identity().map_err(|error| {
                        SearchSessionFailure::debug("RoventPcCollectionOrigin", &error)
                    })?)
                }
                rz_experiments::RoventPurpose::Collection => {
                    return Err(failure(
                        "RoventPcCollectionHooks",
                        "Collection requires the actual root observer",
                    ));
                }
                _ if collection_hooks.root_admission.is_some()
                    || collection_hooks.root_closure.is_some()
                    || collection_hooks.cooperative.is_some() =>
                {
                    return Err(failure(
                        "RoventPcCollectionHooks",
                        "collection hooks require the explicit Collection source selection",
                    ));
                }
                _ => None,
            };
            configuration
                .require_supported_native_capabilities()
                .map_err(|e| SearchSessionFailure::debug("RoventPcCapability", &e))?;
            match (configuration.fields().execution, &loaded) {
                (
                    ExecutionConfiguration::RolesOnly {
                        shared_nnuec_readout: None,
                        ..
                    },
                    None,
                ) => {}
                (
                    ExecutionConfiguration::RolesOnly {
                        shared_nnuec_readout: Some(readout),
                        ..
                    },
                    Some(loaded),
                ) => {
                    if !declares_shared_nnuec_helper(
                        &prepared.sources().lock().manifest.resources.pools,
                    ) {
                        return Err(failure(
                            "RoventPcNnuecResources",
                            "shared readout requires one declared RuntimeHelper thread",
                        ));
                    }
                    let actual = loaded.model().registration().map_err(|error| {
                        SearchSessionFailure::debug("RoventPcNnuecRegistration", &error)
                    })?;
                    if !loaded.matches_sources(prepared.sources())
                        || loaded.input_lock_digest() != prepared.snapshot().input_lock
                        || loaded.registration() != &actual
                        || readout.context_namespace_policy != actual.context_namespace_policy()
                        || readout.local_namespace_policy != actual.local_namespace_policy()
                    {
                        return Err(failure(
                            "RoventPcNnuecStartup",
                            "original NNUEC source/registration/namespace mismatch",
                        ));
                    }
                }
                _ => {
                    return Err(failure(
                        "RoventPcNnuecStartup",
                        "shared readout declaration and original loaded owner must be selected together",
                    ));
                }
            }
            if !matches!(
                configuration.fields().execution,
                ExecutionConfiguration::RolesOnly { .. }
            ) || prepared.snapshot().mode.has_native_r()
                || prepared.snapshot().process_epoch != clock.epoch().0
                || identity.name.is_empty()
                || identity.author.is_empty()
            {
                return Err(failure("RoventPcStartup", "invalid actual R-free startup"));
            }
            let native_clock = prepared
                .service
                .ready_clock()
                .ok_or_else(|| failure("RoventPcStartup", "actual native Ready clock is absent"))?;
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
            let owner = PcSemanticOwner::new_selected_rovent(
                configuration
                    .fields()
                    .semantic
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
                model,
                configuration
                    .fields()
                    .review
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
                configuration
                    .fields()
                    .index
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
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
                .fields()
                .frontier
                .actual()
                .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?;
            Ok((
                configuration,
                collection_origin,
                owner,
                Digest(hash.finalize().into()),
                selective_policy,
                frontier_limits,
            ))
        })();
        let (
            configuration,
            collection_origin,
            owner,
            implementation,
            selective_policy,
            frontier_limits,
        ) = match build {
            Ok(value) => value,
            Err(failure) => {
                return Err(ConstructionRejection {
                    failure,
                    prepared,
                    loaded,
                });
            }
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
            collection_origin,
            collection_hooks,
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
            retained_repair: Mutex::new(None),
            shared_nnuec: loaded.map(|loaded| SharedNnuecOwner {
                loaded,
                actor: Mutex::new(SharedNnuecActor::default()),
            }),
        })
    }
    pub fn shared_nnuec_readback(
        &self,
    ) -> Result<Option<RoventPcSharedNnuecReadback>, ContractError> {
        self.shared_nnuec
            .as_ref()
            .map(|owner| {
                owner
                    .actor
                    .try_lock()
                    .map(|actor| actor.readback.clone())
                    .map_err(|_| poisoned())
            })
            .transpose()
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
    /// A failed transfer cannot silently discard the already recovered report.
    /// Only a closed root owner exposes this move; live work keeps its custody.
    pub fn take_failed_repair_report(
        &self,
    ) -> Result<Option<rz_search::pals::engine::pc_port::PcRepairReport>, ContractError> {
        if !self.closed.load(Ordering::Acquire) {
            return Err(ContractError::new(
                rz_contracts::ErrorCode::InvalidInput,
                rz_contracts::Stage::Output,
                "repair recovery requires a closed product root owner",
            ));
        }
        self.retained_repair
            .lock()
            .map(|mut report| report.take())
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
        let cap = NonZeroU16::new(self.configuration.fields().frontier.maximum_goal_plies)
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
        // This root retains the same optional actor slot through every manager
        // transfer. A failed/unknown root leaves the actor and loaded model in
        // session custody, including when the manager is quarantined.
        let mut shared_actor = match self
            .shared_nnuec
            .as_ref()
            .map(|owner| owner.actor.try_lock())
            .transpose()
        {
            Ok(actor) => actor,
            Err(_) => {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if let (Some(owner), Some(actor)) = (&self.shared_nnuec, shared_actor.as_mut()) {
            if actor.actor.is_none() {
                if actor.readback.spawned_runtime_helpers != 0 {
                    *state = State::Idle(idle);
                    self.closed.store(true, Ordering::Release);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
                let signal = idle.native.service.signal();
                let created = match PreparationActor::spawn_notified(
                    Arc::clone(owner.loaded.model()),
                    signal.clone(),
                ) {
                    Ok(created) => created,
                    Err(error) => {
                        *state = State::Idle(idle);
                        self.closed.store(true, Ordering::Release);
                        return Err(SearchSessionFailure::debug("RoventPcNnuecActor", &error));
                    }
                };
                actor.readback.spawned_runtime_helpers = 1;
                actor.readback.worker_thread_id = created.worker_thread_id();
                actor.signal = signal;
                actor.actor = Some(created);
            }
        }
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
        let recheck = self
            .configuration
            .post_repair_policy_for(self.snapshot.mode);
        if let Err(error) = port.enable_post_repair_recheck(recheck) {
            *state = State::RejectedPort {
                native,
                port: PcPortDriver::new(port),
            };
            self.closed.store(true, Ordering::Release);
            return Err(SearchSessionFailure::debug("RoventPcRecheckPolicy", &error));
        }
        if port.post_repair_recheck() != recheck {
            *state = State::RejectedPort {
                native,
                port: PcPortDriver::new(port),
            };
            self.closed.store(true, Ordering::Release);
            return Err(failure(
                "RoventPcRecheckPolicy",
                "actual port policy differs from its registered configuration",
            ));
        }
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
        let collection_admission = match &self.collection_origin {
            Some(origin) => {
                let admission = port
                    .rovent_frontier()
                    .ok_or_else(|| failure("RoventPcCollectionRoot", "actual Frontier is absent"))
                    .and_then(|frontier| {
                        RoventCollectionRootAdmission::from_attached_frontier(
                            origin,
                            frontier,
                            self.snapshot.input_lock,
                            root_budget,
                            self.selective_policy,
                            Arc::clone(&root_admission),
                        )
                        .map_err(|error| {
                            SearchSessionFailure::debug("RoventPcCollectionRoot", &error)
                        })
                    });
                match admission {
                    Ok(admission) => Some(admission),
                    Err(error) => {
                        *state = State::RejectedPort {
                            native,
                            port: PcPortDriver::new(port),
                        };
                        self.closed.store(true, Ordering::Release);
                        return Err(error);
                    }
                }
            }
            None => None,
        };
        let sources = RoleOnlyManagerSources {
            native: native.service,
            driver: PcPortDriver::new(port),
        };
        let registration = RoleOnlyManagerRegistration {
            authority: context.authority,
            mode: self.snapshot.mode,
            root_budget,
            hard_deadline: context.control.deadline,
            token_budget: self.configuration.fields().review.token_budget,
            first_request_sequence: next_request_sequence,
            role_tickets_max: self.configuration.fields().execution.role_tickets_max(),
            role_call_limit: self.configuration.fields().semantic.max_role_calls,
            event_order: self.configuration.fields().execution.event_order().actual(),
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
                if let Some(clock) = &context.effective_clock {
                    manager.register_effective_clock(clock.clone())?;
                }
                if let Some(owner) = &self.shared_nnuec {
                    manager.enable_shared_nnuec_preparation(Arc::clone(owner.loaded.model()))?;
                }
                if let (Some(observer), Some(admission)) =
                    (&self.collection_hooks.root_admission, collection_admission)
                {
                    observer(&mut manager, admission)?;
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
                let mut published = None;
                loop {
                    let signal = manager.activity_signal();
                    let observed = signal.version();
                    if manager.phase() == ManagerPhase::Running {
                        if !context.accepts().unwrap_or(false)
                            || context.soft_expired(Instant::now())
                        {
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
                    if let Some(pump) = &self.collection_hooks.cooperative {
                        pump()?;
                    }
                    if let Some(actor) =
                        shared_actor.as_ref().and_then(|owner| owner.actor.as_ref())
                    {
                        manager.pump_shared_nnuec_preparation(actor)?;
                    }
                    let changed = manager.tick()?;
                    if let Some(actor) =
                        shared_actor.as_ref().and_then(|owner| owner.actor.as_ref())
                    {
                        manager.pump_shared_nnuec_preparation(actor)?;
                    }
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
        if let Some(observer) = &self.collection_hooks.root_closure {
            if let Err(error) = observer(&mut manager) {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug(
                    "RoventPcCollectionClosure",
                    &error,
                ));
            }
        }
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
        let repair = match manager.driver_mut().port_mut().take_repair_report() {
            Ok(report) => report,
            Err(error) => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcRepairReport", &error));
            }
        };
        *self
            .retained_repair
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = repair;
        let frontier = match manager.driver_mut().port_mut().retire_rovent_frontier() {
            Ok(Some(receipt)) => receipt.snapshot,
            _ => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        let repair = self
            .retained_repair
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let receipt = Arc::new(RoventPcRootReceipt {
            domain: ROVENT_PC_ROOT_DOMAIN,
            authority: context.authority,
            input_lock: self.snapshot.input_lock,
            root_admission,
            frontier,
            physical: physical.clone(),
            repair,
        });
        // This is an observation of the actual fence/frontier/report. It is
        // retained before fallible owner transfers and is not a run-success flag.
        match self.latest.lock() {
            Ok(mut latest) => *latest = Some(Arc::clone(&receipt)),
            Err(_) => {
                // The root cannot resume after this loss of its receipt slot.
                // Preserve the original report even on this poisoned boundary.
                std::mem::forget(receipt);
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        }
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
        let mut shared_actor = self
            .shared_nnuec
            .as_ref()
            .map(|owner| owner.actor.try_lock())
            .transpose()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let mut retained = std::mem::replace(&mut *state, State::Moving);
        loop {
            match retained {
                State::Closed => {
                    *state = State::Closed;
                    return Ok(());
                }
                State::Idle(mut idle) => {
                    if let Some(actor) = shared_actor.as_mut() {
                        if let Err(error) = actor.finish(until) {
                            *state = State::Idle(idle);
                            return Err(error);
                        }
                    }
                    let result = idle.native.finish(until);
                    if idle.native.joined && idle.native.finished == Some(true) {
                        *state = State::Closed;
                    } else {
                        *state = State::Idle(idle);
                    }
                    return result;
                }
                State::RejectedOwner { mut native, owner } => {
                    if let Some(actor) = shared_actor.as_mut() {
                        if let Err(error) = actor.finish(until) {
                            *state = State::RejectedOwner { native, owner };
                            return Err(error);
                        }
                    }
                    let result = native.finish(until);
                    if native.joined && native.finished == Some(true) {
                        *state = State::Closed;
                    } else {
                        *state = State::RejectedOwner { native, owner };
                    }
                    return result;
                }
                State::RejectedPort { mut native, port } => {
                    if let Some(actor) = shared_actor.as_mut() {
                        if let Err(error) = actor.finish(until) {
                            *state = State::RejectedPort { native, port };
                            return Err(error);
                        }
                    }
                    let result = native.finish(until);
                    if native.joined && native.finished == Some(true) {
                        *state = State::Closed;
                    } else {
                        *state = State::RejectedPort { native, port };
                    }
                    return result;
                }
                State::Manager(mut manager) => {
                    let actor = shared_actor.as_ref().and_then(|owner| owner.actor.as_ref());
                    if manager.phase() != ManagerPhase::Finished
                        && Self::drain_manager(&mut manager, actor, until).is_err()
                    {
                        *state = State::Manager(manager);
                        return Err(SearchSessionFailure::physical_completion_unknown());
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
    fn drain_manager(
        manager: &mut Manager,
        actor: Option<&PreparationActor>,
        until: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        manager.begin_shutdown(until)?;
        if let Some(actor) = actor {
            loop {
                let signal = manager.activity_signal();
                let observed = signal.version();
                manager.pump_shared_nnuec_preparation(actor)?;
                let changed = manager.tick()?;
                manager.pump_shared_nnuec_preparation(actor)?;
                match manager.phase() {
                    ManagerPhase::Finished => return Ok(()),
                    ManagerPhase::Quarantined => return Err(AdaptiveManagerError::PhysicalUnknown),
                    _ if Instant::now() >= until => {
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                    _ => {}
                }
                if !changed {
                    signal.wait_changed(
                        observed,
                        until
                            .saturating_duration_since(Instant::now())
                            .min(Duration::from_millis(10)),
                    );
                }
            }
        } else {
            manager
                .run_root_until(until, &AtomicBool::new(false))
                .and_then(|receipt| {
                    receipt
                        .permits_semantic_closure()
                        .then_some(())
                        .ok_or(AdaptiveManagerError::PhysicalUnknown)
                })
        }
    }
}
impl SearchSessionDriver for RoventPcSessionDriver {
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
        NonZeroU16::new(self.configuration.fields().frontier.maximum_goal_plies)
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
            if let Some(owner) = self.shared_nnuec.take() {
                // Do not run PreparationActor's joining Drop on an unknown
                // manager/root. Its original cache/terminal/model owners stay
                // quarantined with the physical source instead.
                std::mem::forget(owner);
            }
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
/// Declaration gate only. Existing resource admission remains the owner of
/// quota, topology and observed OS affinity; this grants none of those facts.
fn declares_shared_nnuec_helper(pools: &[rz_experiments::RoventThreadPool]) -> bool {
    pools.iter().any(|pool| {
        pool.actors
            .contains(&rz_experiments::RoventActor::RuntimeHelper)
            && pool.max_threads >= 1
    })
}
fn poisoned() -> ContractError {
    ContractError::new(
        rz_contracts::ErrorCode::BackendFailure,
        rz_contracts::Stage::Output,
        "Rovent P/C observation mutex poisoned",
    )
}

#[cfg(test)]
mod shared_nnuec_resource_tests {
    use super::declares_shared_nnuec_helper;
    use rz_experiments::{RoventActor, RoventThreadPool};
    use std::collections::BTreeSet;

    #[test]
    fn shared_readout_rejects_absent_or_zero_thread_helper_declarations() {
        assert!(!declares_shared_nnuec_helper(&[]));
        assert!(!declares_shared_nnuec_helper(&[RoventThreadPool {
            actors: BTreeSet::from([RoventActor::PcCaller]),
            max_threads: 1,
            cpu_charge_millicpus: 1_000,
        }]));
        assert!(!declares_shared_nnuec_helper(&[RoventThreadPool {
            actors: BTreeSet::from([RoventActor::RuntimeHelper]),
            max_threads: 0,
            cpu_charge_millicpus: 0,
        }]));
    }

    #[test]
    fn shared_readout_accepts_one_declared_helper_in_original_pool() {
        assert!(declares_shared_nnuec_helper(&[RoventThreadPool {
            actors: BTreeSet::from([RoventActor::Manager, RoventActor::RuntimeHelper]),
            max_threads: 1,
            cpu_charge_millicpus: 1_000,
        }]));
    }
}
