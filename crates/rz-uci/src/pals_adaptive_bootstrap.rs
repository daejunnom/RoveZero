//! Explicit source-pinned V5 process assembly. Models are loaded once on the
//! native owner; root fences keep the same model, Store and CPU/TT owners.
//! Neither metadata preflight nor a successful UCI handshake is NN acceptance.
pub mod external;
mod observations;
mod protocol;
pub mod runtime_assets;

use crate::EngineIdentity;
use crate::engine::{EngineSettings, OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    AdaptiveCollectionDispatchObserver, AdaptiveCompletedRootObserver, AdaptiveExecutionLimits,
    AdaptiveExternalTeacherObserver, AdaptiveRootAdmissionObserver, AdaptiveRootClosureObserver,
    AdaptiveSessionCollectionHooks, AdaptiveSessionDriver, AdaptiveSessionOptions,
    AdaptiveSessionSources, ExternalManagerWorker, PersistentCpuOwner, ProjectedNativeBackend,
};
use crate::pals_adaptive_startup::{
    PalsStartupOptionsV5, PinnedPalsStartupV5, parse_pals_search_settings_v5,
};
use crate::pals_native::startup_virtual::{
    NativeVirtualReviewStartupResourcesV5, NativeVirtualReviewStartupSnapshotV5,
    load_native_virtual_review_startup_v5,
};
use crate::pals_native::ticketed::{
    NativeVirtualReviewTicketBackend, TicketNativeEvent, TicketServiceLimits,
    TicketedNativeService, VirtualReviewCollectionObserver,
};
use crate::search_driver::{
    PalsCheckerRegistration, SearchSessionContext, SearchSessionDriver, SearchSessionFailure,
};
use cap_fs_ext::DirExt;
use observations::PalsAdaptiveObservationBridgeV5;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::{CancelToken, Digest, GameGeneration, ProcessEpoch, RootGeneration};
use rz_eval::asset::parse_sha256;
use rz_experiments::{
    PalsCoordinatorPolicyV5, PalsEngineModeV5, PalsTopologyPolicyV5, admit_pals_startup_content_v5,
    compiled_source_components_v5,
};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_search::cpu::{CpuEngine, CpuResumePolicy};
use rz_search::cpu_value::BootstrapCpuValue;
use rz_search::cpu_worker::{
    CpuWorkerProfile, cpu_evaluation_namespace_digest, cpu_search_conditions_digest,
};
use rz_search::external_cpu::{
    CpuEndpointSelection, ExternalUciCpuChecker,
    worker::{ExternalCpuWorker, ExternalWorkerClock},
};
use rz_search::pals::engine::PalsEngine;
use rz_search::pals::engine::adaptive_port::{
    AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::review_context::{ReviewAnchorMode, ReviewIndexLimits, ReviewSelectionPolicy};
use rz_search::pals::store::{ArchiveConfig, ArchiveRootCapability};
use std::{
    fmt,
    sync::{Arc, Mutex, atomic::AtomicBool, mpsc},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub type ActualAdaptiveBackendV5 =
    ProjectedNativeBackend<AdaptivePortQuestion, NativeVirtualReviewTicketBackend>;
type ActualBackend = ActualAdaptiveBackendV5;
type ActualService = TicketedNativeService<
    rz_search::pals::review_context::ReviewView,
    AdaptivePortQuestion,
    ActualBackend,
>;
type ActualDriver = AdaptiveSessionDriver<ActualBackend>;

pub type AdaptiveTeacherAdmissionV5 = Arc<
    dyn Fn(&external::ExternalBootstrapAdmissionV5) -> Result<(), AdaptiveBootstrapError>
        + Send
        + Sync,
>;
/// Hooks are installed on actual retained owners, never on reconstructed public
/// job/report records. The source admission callback validates the same helper
/// capability that startup retains for this process's entire physical lifetime.
pub struct AdaptiveCollectionHooksV5 {
    pub native_observer: Box<dyn VirtualReviewCollectionObserver>,
    pub dispatch_observer: AdaptiveCollectionDispatchObserver,
    pub teacher_observer: AdaptiveExternalTeacherObserver,
    pub teacher_admission: AdaptiveTeacherAdmissionV5,
    pub root_admission: AdaptiveRootAdmissionObserver<ActualAdaptiveBackendV5>,
    pub root_closure: AdaptiveCompletedRootObserver<ActualAdaptiveBackendV5>,
    pub native_intake: Option<crate::pals_native::startup_virtual::NativeVirtualReviewIntakeV5>,
}

#[derive(Debug)]
pub struct AdaptiveBootstrapError {
    pub stage: &'static str,
    pub detail: String,
}
impl fmt::Display for AdaptiveBootstrapError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(out, "PALS V5 {}: {}", self.stage, self.detail)
    }
}
impl std::error::Error for AdaptiveBootstrapError {}
fn failure(stage: &'static str, cause: &dyn fmt::Debug) -> AdaptiveBootstrapError {
    AdaptiveBootstrapError {
        stage,
        detail: SearchSessionFailure::debug(stage, cause).detail,
    }
}
fn boundary(stage: &'static str) -> AdaptiveBootstrapError {
    failure(stage, &stage)
}

fn verify_private_working_directory(cwd: &std::path::Path) -> Result<(), AdaptiveBootstrapError> {
    // Archive's immutable-input exclusion and source-checkout exclusion are
    // separate. Inspect only marker metadata, never repository contents.
    for parent in cwd.ancestors() {
        match std::fs::symlink_metadata(parent.join(".git")) {
            Ok(_) => return Err(boundary("PrivateSnapshotInsideSourceCheckout")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(failure("SourceCheckoutExclusion", &error)),
        }
    }
    Ok(())
}

fn verify_actual_process_boundary(
    pinned: &PinnedPalsStartupV5,
) -> Result<[u8; 32], AdaptiveBootstrapError> {
    let cwd = std::env::current_dir().map_err(|e| failure("WorkingDirectory", &e))?;
    verify_private_working_directory(&cwd)?;
    if cwd != pinned.private_cwd() {
        return Err(boundary("PinnedWorkingDirectory"));
    }
    if pinned.config.endpoint.source_commit != env!("RZ_PALS_COMPILED_SOURCE_COMMIT") {
        return Err(boundary("CompiledSourceCommit"));
    }
    let executable = crate::process_receipts::ExecutableIdentityV1::observe()
        .map_err(|e| failure("MappedExecutable", &e))?;
    if executable.sha256 != pinned.config.endpoint.binary.sha256
        || executable.bytes != pinned.config.endpoint.binary.bytes
    {
        return Err(boundary("MappedExecutableIdentity"));
    }
    parse_sha256(&executable.sha256).map_err(|e| failure("MappedExecutableIdentity", &e))
}

/// Admit an actual source-pinned collection process using the same checks and
/// retained startup assembly as the CLI. No metadata/mock or CPU substitution
/// path is admitted through this entry point.
pub fn prepare_collection_actual(
    pinned: Arc<PinnedPalsStartupV5>,
    epoch: ProcessEpoch,
    hooks: AdaptiveCollectionHooksV5,
    until: Instant,
) -> Result<PreparedAdaptiveProcessV5, AdaptiveBootstrapError> {
    let started = Instant::now();
    if started >= until
        || epoch.0 == 0
        || pinned.metadata_preflight
        || pinned.config.endpoint.mode != PalsEngineModeV5::Collection
    {
        return Err(boundary("CollectionActualAdmission"));
    }
    let binary = verify_actual_process_boundary(&pinned)?;
    let clock = ProcessClock::new(epoch);
    let prepared = prepare_actual(
        Arc::clone(&pinned),
        clock,
        Arc::new(OwnerRegistry::default()),
        EngineIdentity {
            name: format!("RoveZero PALS adaptive {}", pinned.config.endpoint.id),
            author: "RoveZero contributors".into(),
        },
        binary,
        started,
        Some(until),
        Some(hooks),
    )?;
    let published = prepared
        .driver
        .start_checker(prepared.startup_deadline, Arc::new(AtomicBool::new(false)))
        .map_err(|e| failure("CheckerStartupFence", &e))
        .and_then(|_checker_fence| {
            prepared
                .observations
                .publish_startup()
                .map_err(|e| failure("StartupReceipt", &e))
        });
    if let Err(error) = published {
        // Store the original bound before Drop; failure cleanup cannot renew it.
        let cleanup = prepared.finish_process(until);
        if let Err(cleanup) = cleanup {
            eprintln!(
                "pals_v5_collection_startup_cleanup {}",
                failure("ProcessDrain", &cleanup)
            );
        }
        return Err(error);
    }
    Ok(prepared)
}

/// A bootstrap failure preserves the native owner until its real finish/join.
/// Timeout leaves the owner detached with its physical leases; it is an error,
/// never an assertion of successful shutdown or permission to reuse buffers.
struct StartupNativeGuard {
    native: Option<ActualService>,
    cleanup_until: Instant,
}
impl Drop for StartupNativeGuard {
    fn drop(&mut self) {
        if let Some(native) = &mut self.native {
            let until = self.cleanup_until;
            native.begin_shutdown(until);
            let signal = native.signal();
            let mut physically_closed = false;
            while Instant::now() < until {
                let sequence = signal.version();
                while let Some(event) = native.poll() {
                    if let TicketNativeEvent::Finished {
                        physical_shutdown_confirmed,
                    } = event
                    {
                        physically_closed = physical_shutdown_confirmed;
                    }
                }
                if native.try_join() == Ok(true) {
                    eprintln!(
                        "pals_v5_bootstrap_cleanup owner_joined=true physical_shutdown_confirmed={physically_closed}"
                    );
                    return;
                }
                signal.wait_changed(sequence, Duration::from_millis(10));
            }
            eprintln!(
                "pals_v5_bootstrap_cleanup owner_joined=unknown physical_shutdown_confirmed=unknown"
            );
        }
    }
}

pub fn run_cli(arguments: Vec<String>) -> Result<(), AdaptiveBootstrapError> {
    if arguments == ["--source-inventory"] {
        let pins = compiled_source_components_v5().map_err(|e| failure("SourceInventory", &e))?;
        let output = serde_json::json!({"domain":"rz-pals-compiled-source-registration-v5/1",
            "source_commit":env!("RZ_PALS_COMPILED_SOURCE_COMMIT"),"components":pins,
            "neural_execution":false,"training_execution":false});
        println!(
            "{}",
            serde_json::to_string(&output).map_err(|e| failure("SourceInventory", &e))?
        );
        return Ok(());
    }
    let started = Instant::now();
    let cwd = std::env::current_dir().map_err(|e| failure("WorkingDirectory", &e))?;
    verify_private_working_directory(&cwd)?;
    let options = PalsStartupOptionsV5::parse(&arguments).map_err(|e| failure("Arguments", &e))?;
    let pinned = Arc::new(
        PinnedPalsStartupV5::open(options, &cwd).map_err(|e| failure("PinnedStartup", &e))?,
    );
    let binary = verify_actual_process_boundary(&pinned)?;
    let epoch = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| failure("ProcessEpoch", &e))?
            .as_nanos(),
    )
    .map_err(|e| failure("ProcessEpoch", &e))?;
    let clock = ProcessClock::new(rz_contracts::ProcessEpoch(epoch));
    let owners = Arc::new(OwnerRegistry::default());
    let mut settings = EngineSettings {
        max_workers: 1,
        shutdown_limit: Duration::from_millis(pinned.config.native_options.drain_ms),
        ..EngineSettings::default()
    };
    settings.search.untimed_limit =
        Duration::from_millis(pinned.lock.manifest.window.wall_time_max_ms);
    let identity = EngineIdentity {
        name: format!("RoveZero PALS adaptive {}", pinned.config.endpoint.id),
        author: "RoveZero contributors".into(),
    };
    if pinned.metadata_preflight {
        pinned
            .publish_metadata_record()
            .map_err(|e| failure("MetadataReceipt", &e))?;
        let implementation = source_digest(&pinned, "coordinator")?;
        return protocol::serve_driver(
            Arc::new(protocol::RulesOnlyPreflightDriver {
                identity,
                implementation,
            }),
            clock,
            owners,
            settings,
        )
        .map_err(|e| failure("MetadataProtocol", &e));
    }
    let prepared = prepare_actual(
        Arc::clone(&pinned),
        clock.clone(),
        Arc::clone(&owners),
        identity,
        binary,
        started,
        None,
        None,
    )?;
    settings.search.max_simulations = prepared.cpu_nodes_limit;
    let published = prepared
        .driver
        .start_checker(prepared.startup_deadline, Arc::new(AtomicBool::new(false)))
        .map_err(|e| failure("CheckerStartupFence", &e))
        .and_then(|_| {
            prepared
                .observations
                .publish_startup()
                .map_err(|e| failure("StartupReceipt", &e))
        });
    if let Err(error) = published {
        // Failed startup keeps its original deadline, including Drop cleanup.
        if let Err(cleanup) = prepared.finish_process(prepared.startup_deadline) {
            eprintln!(
                "pals_v5_cli_startup_cleanup {}",
                failure("ProcessDrain", &cleanup)
            );
        }
        return Err(error);
    }
    let served = protocol::serve_driver(prepared.driver.clone(), clock, owners, settings);
    let until = Instant::now()
        .checked_add(settings.shutdown_limit)
        .ok_or_else(|| boundary("ProcessDrainDeadline"))?;
    let finish = prepared.finish_process(until);
    match &finish {
        Ok(receipt) => {
            prepared
                .observations
                .publish_termination(receipt)
                .map_err(|e| failure("TerminationReceipt", &e))?;
        }
        Err(error) => {
            let retained = prepared.driver.retained_owner_status().ok();
            if let Err(record_error) =
                prepared
                    .observations
                    .publish_failed_termination(None, error, retained.as_ref())
            {
                eprintln!(
                    "pals_v5_failure_record {}",
                    failure("FailureReceipt", &record_error)
                );
            }
            return Err(failure("ProcessDrain", error));
        }
    }
    served.map_err(|e| failure("NativeProtocol", &e))
}

struct CollectionContextOwnerV5 {
    enabled: bool,
    next_root: u64,
    current: Arc<Mutex<crate::contracts::SessionScope>>,
    active: Option<SearchSessionContext>,
}

pub struct PreparedAdaptiveProcessV5 {
    driver: Arc<ActualDriver>,
    observations: Arc<PalsAdaptiveObservationBridgeV5>,
    // Retain the original helper and model/snapshot capabilities through finish.
    _external: Option<external::ExternalBootstrapAdmissionV5>,
    _pinned: Arc<PinnedPalsStartupV5>,
    cpu_nodes_limit: u64,
    startup_deadline: Instant,
    process_cleanup_ceiling: Instant,
    cleanup_deadline: Mutex<Option<Instant>>,
    clock: ProcessClock,
    collection_context: Mutex<CollectionContextOwnerV5>,
}
impl PreparedAdaptiveProcessV5 {
    pub fn driver(&self) -> Arc<AdaptiveSessionDriver<ActualAdaptiveBackendV5>> {
        Arc::clone(&self.driver)
    }
    pub fn clock(&self) -> ProcessClock {
        self.clock.clone()
    }
    /// Only this UCI owner constructs the private current-scope capability.
    /// One prepared process is one game; each actual turn has a new root.
    pub fn context(
        &self,
        hard_deadline: Instant,
        shutdown_deadline: Instant,
    ) -> Result<SearchSessionContext, AdaptiveBootstrapError> {
        if Instant::now() >= hard_deadline || shutdown_deadline < hard_deadline {
            return Err(boundary("CollectionContextDeadline"));
        }
        self.clock
            .deadline(hard_deadline)
            .map_err(|e| failure("CollectionContextClock", &e))?;
        self.clock
            .deadline(shutdown_deadline)
            .map_err(|e| failure("CollectionContextClock", &e))?;
        let mut owner = self
            .collection_context
            .lock()
            .map_err(|_| boundary("CollectionContextOwner"))?;
        if !owner.enabled || owner.active.is_some() {
            return Err(boundary("CollectionContextAdmission"));
        }
        self.driver
            .collection_context_ready()
            .map_err(|e| failure("CollectionContextAdmission", &e))?;
        let root = owner
            .next_root
            .checked_add(1)
            .ok_or_else(|| boundary("CollectionRootSequence"))?;
        let authority = SearchAuthority {
            epoch: self.clock.epoch(),
            game: GameGeneration(1),
            root: RootGeneration(root),
            implementation: self.driver.implementation(),
        };
        let context = SearchSessionContext {
            authority,
            control: rz_search::driver::SearchControl::new(hard_deadline, self.cpu_nodes_limit),
            cancellation: CancelToken::new(),
            shutdown_deadline,
            current: Arc::clone(&owner.current),
        };
        *owner
            .current
            .lock()
            .map_err(|_| boundary("CollectionContextAuthority"))? =
            crate::contracts::SessionScope::Search(authority);
        owner.next_root = root;
        owner.active = Some(context.clone());
        Ok(context)
    }
    /// Retires a real in-process decision without fabricating a UCI stdout
    /// bestmove acknowledgement. The exact private capability must match.
    pub fn retire_context(
        &self,
        context: &SearchSessionContext,
    ) -> Result<(), AdaptiveBootstrapError> {
        let mut owner = self
            .collection_context
            .lock()
            .map_err(|_| boundary("CollectionContextOwner"))?;
        let active = owner
            .active
            .as_ref()
            .ok_or_else(|| boundary("CollectionContextRetirement"))?;
        if active.authority != context.authority || !Arc::ptr_eq(&active.current, &context.current)
        {
            return Err(boundary("CollectionContextRetirement"));
        }
        self.retire_active_context(&mut owner)
    }
    fn retire_active_context(
        &self,
        owner: &mut CollectionContextOwnerV5,
    ) -> Result<(), AdaptiveBootstrapError> {
        if let Some(active) = &owner.active {
            active.cancellation.cancel();
            active.control.cancel();
            *owner
                .current
                .lock()
                .map_err(|_| boundary("CollectionContextAuthority"))? =
                crate::contracts::SessionScope::Search(SearchAuthority {
                    root: RootGeneration(0),
                    ..active.authority
                });
            self.driver
                .retire_collection_output(active.authority)
                .map_err(|e| failure("CollectionContextRetirement", &e))?;
            owner.active = None;
        }
        Ok(())
    }
    pub fn finish_process(
        &self,
        until: Instant,
    ) -> Result<crate::pals_adaptive::AdaptiveSessionProcessReceipt, SearchSessionFailure> {
        let mut owner = self
            .collection_context
            .lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        // A failed active turn retains its original cleanup bound. Once a
        // successful turn is retired, the later whole-game close has its own
        // deadline; it must not inherit an earlier, already completed turn.
        let until = owner
            .active
            .as_ref()
            .map_or(until, |active| until.min(active.shutdown_deadline));
        let mut deadline = self
            .cleanup_deadline
            .lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let until = until.min(self.process_cleanup_ceiling);
        let original = deadline.map_or(until, |previous| previous.min(until));
        *deadline = Some(original);
        drop(deadline);
        self.retire_active_context(&mut owner)
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        owner.enabled = false;
        drop(owner);
        self.driver.finish_process(original)
    }
}
impl Drop for PreparedAdaptiveProcessV5 {
    fn drop(&mut self) {
        // Includes publication/transport failure after construction. An already
        // closed driver returns its original receipt rather than joining twice.
        let drain = Duration::from_millis(self._pinned.config.native_options.drain_ms);
        let until = Instant::now()
            .checked_add(drain)
            .unwrap_or_else(Instant::now)
            .min(self.process_cleanup_ceiling);
        if let Err(error) = self.finish_process(until) {
            eprintln!(
                "pals_v5_failed_process_cleanup {}",
                failure("ProcessDrain", &error)
            );
        }
    }
}

fn source_digest(
    pinned: &PinnedPalsStartupV5,
    component: &str,
) -> Result<Digest, AdaptiveBootstrapError> {
    parse_sha256(
        pinned
            .config
            .source_components
            .get(component)
            .ok_or_else(|| boundary("SourceComponentMissing"))?,
    )
    .map(Digest)
    .map_err(|e| failure("SourceComponent", &e))
}

fn prepare_actual(
    pinned: Arc<PinnedPalsStartupV5>,
    clock: ProcessClock,
    owners: Arc<OwnerRegistry>,
    identity: EngineIdentity,
    actual_binary: [u8; 32],
    started: Instant,
    startup_ceiling: Option<Instant>,
    hooks: Option<AdaptiveCollectionHooksV5>,
) -> Result<PreparedAdaptiveProcessV5, AdaptiveBootstrapError> {
    let process_cleanup_ceiling = match startup_ceiling {
        Some(original) => original,
        None => started
            .checked_add(Duration::from_millis(
                pinned.lock.manifest.window.wall_time_max_ms,
            ))
            .and_then(|end| {
                end.checked_add(Duration::from_millis(pinned.config.native_options.drain_ms))
            })
            .ok_or_else(|| boundary("ProcessCleanupCeiling"))?,
    };
    let startup_until = started
        .checked_add(Duration::from_millis(
            pinned.config.native_options.startup_ms,
        ))
        .ok_or_else(|| boundary("StartupDeadline"))?;
    let startup_until = startup_ceiling.map_or(startup_until, |ceiling| startup_until.min(ceiling));
    if Instant::now() >= startup_until {
        return Err(boundary("StartupExpired"));
    }
    let declaration = &pinned.config.endpoint;
    if declaration.model.profile != "virtual_review_v1"
        || declaration.resources.topology != PalsTopologyPolicyV5::SharedControl
    {
        return Err(boundary("UnsupportedActualStartupProfile"));
    }
    let selections = parse_pals_search_settings_v5(&pinned.config, startup_until)
        .map_err(|e| failure("SearchSettings", &e))?;
    if selections.resume != CpuResumePolicy::PausedStack {
        return Err(boundary("CooperativeCpuRequiresDeclaredPausedStack"));
    }
    let slices = &declaration.coordinator.slices;
    let resource_limits = AdaptiveExecutionLimits {
        waiting_jobs_max: slices.waiting_jobs_max as usize,
        worker_commands_max: slices.worker_commands_max as usize,
        role_tickets_max: slices.role_tickets_max as usize,
        paused_bytes_per_worker_max: slices.paused_bytes_per_worker_max as usize,
    };
    let external = external::load_external_cpu_v5(&pinned, startup_until, &AtomicBool::new(false))
        .map_err(|e| failure("ExternalSnapshot", &e))?;
    let collection_enabled = hooks.is_some();
    let (native_observer, dispatch_observer, native_intake, collection_hooks) = match hooks {
        None => (None, None, None, AdaptiveSessionCollectionHooks::default()),
        Some(hooks) => {
            let admitted = external
                .as_ref()
                .ok_or_else(|| boundary("CollectionExternalTeacher"))?;
            if admitted.selection().execution_use
                != rz_search::external_cpu::CpuExecutionUse::Collection
                || admitted.selection().cpu_t != rz_search::external_cpu::CpuEndpoint::ExternalUci
            {
                return Err(boundary("CollectionExternalTeacher"));
            }
            (hooks.teacher_admission)(admitted)?;
            (
                Some(hooks.native_observer),
                Some(hooks.dispatch_observer),
                hooks.native_intake,
                AdaptiveSessionCollectionHooks {
                    root_admission: Some(hooks.root_admission),
                    root_closure: Some(hooks.root_closure),
                    external_teacher: Some(hooks.teacher_observer),
                },
            )
        }
    };
    let endpoints = external.as_ref().map_or_else(
        || match declaration.mode {
            PalsEngineModeV5::Collection => CpuEndpointSelection::collection(),
            PalsEngineModeV5::Deployment => CpuEndpointSelection::deployment(),
        },
        |external| external.selection(),
    );
    let mut cpu = Vec::new();
    cpu.try_reserve_exact(declaration.coordinator.slices.workers as usize)
        .map_err(|e| failure("CpuOwnerAllocation", &e))?;
    for index in 0..declaration.coordinator.slices.workers {
        let evaluator = Arc::new(BootstrapCpuValue::default());
        let mut engine = if selections.paused_side_scope {
            CpuEngine::with_paused_side_scope(
                selections.own_cpu.clone(),
                evaluator,
                selections.ordering,
            )
        } else {
            CpuEngine::with_evaluator_ordering_and_resume_policy(
                selections.own_cpu.clone(),
                evaluator,
                selections.ordering,
                selections.resume,
            )
        }
        .map_err(|e| failure("CpuOwner", &e))?;
        engine
            .set_paused_stack_max_bytes(resource_limits.paused_bytes_per_worker_max)
            .map_err(|e| failure("PausedOwnerCapacity", &e))?;
        let profile = CpuWorkerProfile {
            owner: CpuWorkerId((index + 1) as u16),
            max_depth: engine.config().max_depth,
            evaluation_namespace: cpu_evaluation_namespace_digest(&engine)
                .map_err(|e| failure("CpuValueNamespace", &e))?,
            search_conditions: cpu_search_conditions_digest(&engine),
        };
        cpu.push(PersistentCpuOwner { engine, profile });
    }
    let metadata = if let Some(external) = &external {
        // Declared external checker has no live process or TT. Actual startup
        // and source/option/name handshake happen later on its one private owner.
        let checker = ExternalUciCpuChecker::create(external.config().clone())
            .map_err(|e| failure("ExternalMetadata", &e))?;
        crate::pals_adaptive::AdaptiveCheckerMetadata::capture(&checker)
            .map_err(|e| failure("ExternalMetadata", &e))?
    } else {
        crate::pals_adaptive::AdaptiveCheckerMetadata::capture_owned(&cpu[0].engine)
            .map_err(|e| failure("CpuMetadata", &e))?
    };
    let asset_refs: Vec<_> = pinned
        .assets()
        .iter()
        .map(|a| (&a.declaration, &a.file))
        .collect();
    let admission = admit_pals_startup_content_v5(&pinned.config, &pinned.lock, &asset_refs)
        .map_err(|e| failure("ContentAdmission", &e))?;
    drop(asset_refs);
    let factory_pinned = Arc::clone(&pinned);
    let factory_admission = admission.clone();
    let factory_clock = clock.runtime_clock();
    let factory_epoch = clock.epoch();
    let (sender, receiver) = mpsc::sync_channel::<
        Result<NativeVirtualReviewStartupSnapshotV5, rz_search::pals::coordinator::RoleFailure>,
    >(1);
    let native = TicketedNativeService::spawn_with_capacity(
        move || {
            let result = (|| {
                let runtime_pin = runtime_assets::pin_runtime_v5(&factory_pinned).map_err(|e| {
                    rz_search::pals::coordinator::RoleFailure::Backend(e.to_string())
                })?;
                let cuda_profile_root =
                    factory_pinned
                        .config
                        .native_options
                        .cuda_device_id
                        .map(|_| {
                            factory_pinned
                                .private_cwd()
                                .join(&factory_pinned.config.runtime_directory)
                                .join(format!(
                                    "cuda-profile-{}-{}",
                                    std::process::id(),
                                    factory_epoch.0
                                ))
                        });
                let mut loaded = load_native_virtual_review_startup_v5(
                    factory_pinned,
                    factory_clock,
                    NativeVirtualReviewStartupResourcesV5 {
                        content_admission: factory_admission,
                        runtime_pin,
                        cuda_profile_root,
                        until: startup_until,
                        content_intake: native_intake,
                    },
                )?;
                if let Some(observer) = native_observer {
                    loaded.backend.set_collection_observer(observer)?;
                }
                Ok(loaded)
            })();
            let notification = result
                .as_ref()
                .map(|loaded| loaded.snapshot.clone())
                .map_err(Clone::clone);
            let _ = sender.send(notification);
            result.map(|loaded| ProjectedNativeBackend::new(loaded.backend))
        },
        TicketServiceLimits {
            review_token_budget: declaration.context.token_budget as u16,
            ..TicketServiceLimits::default()
        },
        resource_limits.role_tickets_max,
        resource_limits.worker_commands_max,
    )
    .map_err(|e| failure("NativeOwnerThread", &e))?;
    let mut native = StartupNativeGuard {
        native: Some(native),
        cleanup_until: startup_until,
    };
    let snapshot = receiver
        .recv_timeout(startup_until.saturating_duration_since(Instant::now()))
        .map_err(|e| failure("NativeFactoryDeadline", &e))?
        .map_err(|e| failure("NativeFactory", &e))?;
    let observations = Arc::new(
        PalsAdaptiveObservationBridgeV5::new(
            Arc::clone(&pinned),
            &admission,
            snapshot.clone(),
            actual_binary,
            pinned.private_cwd(),
        )
        .map_err(|e| failure("ObservationOwner", &e))?,
    );
    let model = AdaptivePortModel::new_with_namespace(
        snapshot.role_model_identity(),
        Some(
            snapshot
                .model_value_identity()
                .map_err(|e| failure("ModelValueNamespace", &e))?,
        ),
        snapshot
            .adaptive_model_namespace()
            .map_err(|e| failure("ModelTicketNamespace", &e))?,
    )
    .map_err(|e| failure("ModelMirror", &e))?;
    let mirror = metadata.clone().into_mirror();
    let registration =
        PalsCheckerRegistration::capture_with_resolver(&mirror, &model, selections.resolver)
            .map_err(|e| failure("CheckerRegistration", &e))?;
    let mut engine = PalsEngine::new_with_boxed_checker_and_policies(
        selections.config,
        model,
        Box::new(mirror),
        selections.resolver,
        selections.post_repair_recheck,
    )
    .map_err(|e| failure("SearchStore", &e))?
    .with_selected_review_context(
        ReviewSelectionPolicy {
            mode: ReviewAnchorMode::AnchoredCompact,
            token_budget: declaration.context.token_budget as u16,
            max_records: declaration.context.selected_records_max as u16,
            max_anchors: declaration.context.anchors_max as u8,
            required_bundle: None,
        },
        ReviewIndexLimits {
            active_bundles: declaration.context.active_bundles_max as usize,
            records_per_bundle: declaration.context.selected_records_max as usize,
        },
    )
    .map_err(|e| failure("ReviewIndex", &e))?;
    let archive_name = format!(
        "adaptive-archive-{}-{}",
        std::process::id(),
        clock.epoch().0
    );
    let archive_parent = pinned
        .runtime_directory_capability()
        .map_err(|e| failure("ArchiveRoot", &e))?;
    archive_parent
        .create_dir(&archive_name)
        .map_err(|e| failure("ArchiveRoot", &e))?;
    let archive_root = archive_parent
        .open_dir_nofollow(&archive_name)
        .map_err(|e| failure("ArchiveRootCapability", &e))?;
    let archive_path = pinned
        .private_cwd()
        .join(&pinned.config.runtime_directory)
        .join(&archive_name);
    let archive_capability =
        ArchiveRootCapability::from_directory(archive_path.clone(), archive_root)
            .map_err(|e| failure("ArchiveRootCapability", &e))?;
    engine
        .enable_archive_with_capability(
            ArchiveConfig::new(archive_path, pinned.private_cwd().join("inputs")),
            archive_capability,
        )
        .map_err(|e| failure("ArchiveOwner", &e))?;
    let external_owner = match &external {
        None => None,
        Some(external) => {
            let owner = CpuWorkerId((declaration.coordinator.slices.workers + 1) as u16);
            let runtime = clock.runtime_clock();
            let worker = ExternalCpuWorker::spawn_pinned(
                external.config().clone(),
                owner,
                ExternalWorkerClock {
                    domain: clock.domain(),
                    origin: runtime.origin(),
                },
                startup_until,
                external.launch_pin(),
            )
            .map_err(|e| failure("ExternalOwner", &e))?;
            Some(ExternalManagerWorker {
                worker,
                config: external.config().clone(),
                owner,
            })
        }
    };
    let policy = match declaration.coordinator.policy {
        PalsCoordinatorPolicyV5::Synchronous => CoordinatorPolicy::Synchronous,
        PalsCoordinatorPolicyV5::CoordinatorFixed => CoordinatorPolicy::Fixed,
        PalsCoordinatorPolicyV5::CoordinatorAdaptive => CoordinatorPolicy::Adaptive,
    };
    let controller = ControllerConfig {
        policy,
        fixed_node_credit: declaration.coordinator.slices.slice_native_nodes_max,
        fixed_soft_lease: Duration::from_micros(declaration.coordinator.slices.soft_lease_micros),
        initial_workers: declaration.coordinator.slices.workers as u8,
        max_workers: declaration.coordinator.slices.workers as u8,
        ..ControllerConfig::default()
    };
    let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
    let callback = Arc::clone(&observations);
    let closure: AdaptiveRootClosureObserver<ActualBackend> = Arc::new(move |manager| {
        let snapshot = manager.root_observation_snapshot()?;
        callback
            .observe_closed_root(&snapshot)
            .map_err(|e| crate::pals_adaptive::AdaptiveManagerError::Semantic(e.to_string()))
    });
    let options = AdaptiveSessionOptions {
        implementation: source_digest(&pinned, "coordinator")?,
        identity,
        max_rounds: selections.limits.max_rounds,
        max_cpu_nodes: selections.limits.max_cpu_nodes,
        cpu_depth: selections.limits.cpu_depth,
        token_budget: declaration.context.token_budget as u16,
        policy,
        endpoints,
        controller,
        device_digest: Digest(snapshot.backend_sha256),
        cpu_device_digest: source_digest(&pinned, "resource_boundary")?,
        external_policy: external
            .as_ref()
            .map_or_else(Default::default, |e| e.slice_policy()),
    };
    let sources = AdaptiveSessionSources {
        native: native.native.take().expect("guard owns native"),
        engine,
        cpu,
        external: external_owner,
        checker_metadata: metadata,
    };
    let driver = AdaptiveSessionDriver::new_with_resource_limits_and_collection_hooks(
        options,
        resource_limits,
        sources,
        clock.clone(),
        issuer,
        registration,
        dispatch_observer,
        Some(closure),
        collection_hooks,
    )
    .map_err(|rejected| {
        let error = failure("SessionAdmission", &rejected.failure);
        cleanup_rejected_sources(rejected.sources, startup_until);
        error
    })?;
    let initial_scope = crate::contracts::SessionScope::Search(SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(1),
        root: RootGeneration(0),
        implementation: driver.implementation(),
    });
    Ok(PreparedAdaptiveProcessV5 {
        driver: Arc::new(driver),
        observations,
        _external: external,
        _pinned: pinned,
        cpu_nodes_limit: selections.limits.max_cpu_nodes,
        startup_deadline: startup_until,
        process_cleanup_ceiling,
        cleanup_deadline: Mutex::new(None),
        clock,
        collection_context: Mutex::new(CollectionContextOwnerV5 {
            enabled: collection_enabled,
            next_root: 0,
            current: Arc::new(Mutex::new(initial_scope)),
            active: None,
        }),
    })
}

/// Rejected declarations never re-enter session admission. These are the same
/// owners created at startup, with no accepted jobs and no synthetic work ledger.
fn cleanup_rejected_sources(mut sources: AdaptiveSessionSources<ActualBackend>, until: Instant) {
    sources.native.begin_shutdown(until);
    if let Some(external) = &sources.external {
        external.worker.request_shutdown(until);
    }
    let signal = sources.native.signal();
    let mut physical = None;
    let mut joined = false;
    let mut external_joined = sources.external.is_none();
    let mut failed = false;
    while Instant::now() < until && !failed {
        let version = signal.version();
        while let Some(event) = sources.native.poll() {
            match event {
                TicketNativeEvent::Finished {
                    physical_shutdown_confirmed,
                } => {
                    physical = Some(physical_shutdown_confirmed);
                }
                TicketNativeEvent::Ready { .. } => {}
                _ => failed = true,
            }
        }
        if !joined {
            match sources.native.try_join() {
                Ok(done) => joined = done,
                Err(_) => failed = true,
            }
        }
        if !external_joined {
            if let Some(external) = &mut sources.external {
                if let Some(result) = external.worker.try_join() {
                    external_joined = result.is_ok_and(|receipt| {
                        receipt.cleanup_complete && !receipt.quarantined && !receipt.ownership_lost
                    });
                    failed |= !external_joined;
                }
            }
        }
        if joined && physical == Some(true) && external_joined {
            let archive_closed =
                sources
                    .engine
                    .finish_followup_owners(until)
                    .is_ok_and(|snapshot| {
                        snapshot.archive.as_ref().is_ok_and(|archive| {
                            archive.as_ref().is_none_or(|owner| owner.admission_closed)
                        })
                    });
            if archive_closed {
                eprintln!(
                    "pals_v5_rejected_startup_cleanup physical=confirmed native_joined=true external_joined=true archive_closed=true"
                );
                return;
            }
            failed = true;
        }
        if physical == Some(false) {
            failed = true;
        }
        if !failed {
            signal.wait_changed(version, Duration::from_millis(1));
        }
    }
    // Preserve every actual owner on uncertain completion. A process-level
    // quarantine is never labelled successful, refunded, or reused.
    let _quarantined = Box::leak(Box::new(sources));
    eprintln!("pals_v5_rejected_startup_cleanup physical=unknown owners_quarantined=true");
}
