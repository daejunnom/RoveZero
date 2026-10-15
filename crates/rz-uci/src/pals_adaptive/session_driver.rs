//! Process-owned adaptive UCI session. A root fence returns the same native
//! session, CPU engine/TT and Search Store; only process finish releases them.
//! Failed ownership transfers are retained in the session, never mapped to an
//! empty successful search or replaced by a synchronous/mock backend.

mod checker_metadata;
pub use checker_metadata::*;

use super::{
    AdaptiveCollectionDispatchObserver, AdaptiveExecutionLimits, AdaptiveExecutionManager,
    AdaptiveExternalTeacherObserver, AdaptiveManagerAdmissionRejection, AdaptiveManagerError,
    AdaptiveManagerRegistration, AdaptiveManagerSources, AdaptivePortDriver,
    AdaptiveSequenceContinuation, AdaptiveShutdownReceipt, ExternalManagerWorker, ManagerPhase,
    PersistentCpuOwner,
};
use crate::EngineIdentity;
use crate::engine::ProcessClock;
use crate::pals_native::ticketed::{TicketNativeBackend, TicketNativeEvent, TicketedNativeService};
use crate::search_driver::work::{AttemptObservation, ProcessWorkJournal};
use crate::search_driver::{
    DriverWork, PalsCheckerRegistration, PalsResolverIdentity, PalsSearchPolicyIdentity,
    ProcessSearchWorkReceipt, SearchKind, SearchSessionContext, SearchSessionDriver,
    SearchSessionFailure, SearchSessionReport,
};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{ContractError, Digest, ErrorCode, ProcessEpoch, Stage};
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_runtime::contracts::ContractClock as _;
use rz_search::cpu_checker::{CheckerIdentity, CheckerShutdown, ExternalUciIdentity};
use rz_search::cpu_worker::CpuWorker;
use rz_search::external_cpu::{CpuEndpointSelection, ExternalSlicePolicy};
use rz_search::pals::coordinator::ProposerRankingOutput;
use rz_search::pals::engine::adaptive_port::archive_worker::{
    ArchiveOperation, ArchiveOwner, ArchiveQuarantine, ArchiveTransfer, ArchiveWork,
    ArchiveWorkReceipt, ArchiveWorker,
};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveEnginePort, AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::engine::{PalsCompletion, PalsCounters, PalsEngine, PalsLimits};
use rz_search::pals::review_context::ReviewView;
use sha2::{Digest as _, Sha256};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

pub type AdaptiveSessionManager<B> =
    AdaptiveExecutionManager<AdaptivePortQuestion, B, AdaptivePortDriver<AdaptivePortModel>>;
type SessionManager<B> = AdaptiveSessionManager<B>;
type AdmissionRejection<B> = AdaptiveManagerAdmissionRejection<
    AdaptivePortQuestion,
    B,
    AdaptivePortDriver<AdaptivePortModel>,
>;

/// Runs on the actual UCI search owner after a real root fence and before the
/// manager is dismantled. Collection can seal actual turns/action outcomes.
pub type AdaptiveRootClosureObserver<B> =
    Arc<dyn Fn(&mut SessionManager<B>) -> Result<(), AdaptiveManagerError> + Send + Sync>;

/// Collection setup executes on the real exclusive Search owner, before root
/// Rules preparation or native ranking admission.
pub type AdaptiveRootAdmissionObserver<B> =
    Arc<dyn Fn(&mut SessionManager<B>) -> Result<(), AdaptiveManagerError> + Send + Sync>;
/// The completion is taken from the actual semantic result after the root fence,
/// rather than inferred by a collection caller from elapsed time or counters.
pub type AdaptiveCompletedRootObserver<B> = Arc<
    dyn Fn(&mut SessionManager<B>, PalsCompletion) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;

pub struct AdaptiveSessionCollectionHooks<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub root_admission: Option<AdaptiveRootAdmissionObserver<B>>,
    pub root_closure: Option<AdaptiveCompletedRootObserver<B>>,
    pub external_teacher: Option<AdaptiveExternalTeacherObserver>,
}
impl<B> Default for AdaptiveSessionCollectionHooks<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    fn default() -> Self {
        Self {
            root_admission: None,
            root_closure: None,
            external_teacher: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveSessionOptions {
    pub implementation: Digest,
    pub identity: EngineIdentity,
    pub max_rounds: u64,
    pub max_cpu_nodes: u64,
    pub cpu_depth: u16,
    pub token_budget: u16,
    pub policy: CoordinatorPolicy,
    pub endpoints: CpuEndpointSelection,
    pub controller: ControllerConfig,
    pub device_digest: Digest,
    pub cpu_device_digest: Digest,
    pub external_policy: ExternalSlicePolicy,
}
impl AdaptiveSessionOptions {
    fn validate(&self) -> Result<(), SearchSessionFailure> {
        if self.implementation == Digest([0; 32])
            || self.identity.name.is_empty()
            || self.identity.author.is_empty()
            || self.max_rounds == 0
            || self.max_cpu_nodes == 0
            || self.cpu_depth == 0
            || self.cpu_depth > 64
            || ![128, 256, 512].contains(&self.token_budget)
            || self.controller.policy != self.policy
        {
            return Err(failure(
                "AdaptiveSessionDeclaration",
                "invalid explicit startup settings",
            ));
        }
        self.endpoints
            .validate()
            .map_err(|e| SearchSessionFailure::debug("AdaptiveSessionDeclaration", &e))?;
        Ok(())
    }
}

/// Supplied by the real startup owner. There is no factory/model loader here.
/// `cpu` contains the already-created engines; no duplicate TT is allocated.
pub struct AdaptiveSessionSources<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    pub native: TicketedNativeService<ReviewView, AdaptivePortQuestion, B>,
    pub engine: PalsEngine<AdaptivePortModel>,
    pub cpu: Vec<PersistentCpuOwner>,
    pub external: Option<ExternalManagerWorker>,
    pub checker_metadata: AdaptiveCheckerMetadata,
}
pub struct AdaptiveSessionAdmissionRejection<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    pub failure: SearchSessionFailure,
    pub sources: AdaptiveSessionSources<B>,
}

struct IdleOwners<B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>> {
    native: TicketedNativeService<ReviewView, AdaptivePortQuestion, B>,
    engine: Option<Box<PalsEngine<AdaptivePortModel>>>,
    cpu: Vec<PersistentCpuOwner>,
    external: Option<ExternalManagerWorker>,
    continuation: Option<AdaptiveSequenceContinuation>,
    native_finished: Option<bool>,
    native_joined: bool,
    external_finished: Option<CheckerShutdown>,
}
// Retained process owners stay on the heap while failure/root transitions move
// the state. Empty variants must not reserve every possible engine payload on
// the caller's stack, and a failed transition keeps the same exclusive owner.
enum SessionState<B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>>
where
    B::Output: ProposerRankingOutput,
{
    Idle(Box<IdleOwners<B>>),
    Manager(Box<SessionManager<B>>),
    AdmissionRejected(Box<AdmissionRejection<B>>),
    PortRejected {
        owners: Box<IdleOwners<B>>,
        driver: AdaptivePortDriver<AdaptivePortModel>,
    },
    SpawnRejected {
        owners: Box<IdleOwners<B>>,
        driver: AdaptivePortDriver<AdaptivePortModel>,
        workers: Vec<CpuWorker>,
    },
    ArchivePending {
        owners: Box<IdleOwners<B>>,
        worker: ArchiveWorker<AdaptivePortModel>,
        transfer: ArchiveTransfer,
        authority: SearchAuthority,
    },
    ArchiveQuarantined {
        owners: Box<IdleOwners<B>>,
        worker: ArchiveWorker<AdaptivePortModel>,
        quarantine: Box<ArchiveQuarantine<AdaptivePortModel>>,
    },
    ArchiveJoinedPending {
        owners: Box<IdleOwners<B>>,
        worker: ArchiveWorker<AdaptivePortModel>,
    },
    Closed(AdaptiveSessionProcessReceipt),
    // Exists only under the exclusive mutex while moving owners; never a
    // reusable state. Poison/panic is a failed ownership boundary.
    Moving,
}

/// Producer-only process receipt. Root quiescence cannot construct this type.
#[derive(Clone, Debug)]
pub struct AdaptiveSessionProcessReceipt {
    epoch: ProcessEpoch,
    last_authority: Option<SearchAuthority>,
    native_owner_joined: bool,
    native_physical_shutdown_confirmed: bool,
    external_shutdown: Option<CheckerShutdown>,
    store_archive_closed: bool,
    latest_root: Option<AdaptiveShutdownReceipt>,
}
impl AdaptiveSessionProcessReceipt {
    pub fn epoch(&self) -> ProcessEpoch {
        self.epoch
    }
    pub fn last_authority(&self) -> Option<SearchAuthority> {
        self.last_authority
    }
    pub fn native_owner_joined(&self) -> bool {
        self.native_owner_joined
    }
    pub fn native_physical_shutdown_confirmed(&self) -> bool {
        self.native_physical_shutdown_confirmed
    }
    pub fn external_shutdown(&self) -> Option<CheckerShutdown> {
        self.external_shutdown
    }
    pub fn store_archive_closed(&self) -> bool {
        self.store_archive_closed
    }
    pub fn latest_root(&self) -> Option<&AdaptiveShutdownReceipt> {
        self.latest_root.as_ref()
    }
    pub fn permits_archive_close(&self) -> bool {
        self.native_owner_joined
            && self.native_physical_shutdown_confirmed
            && self
                .external_shutdown
                .is_none_or(|r| r.cleanup_complete && !r.quarantined && !r.ownership_lost)
            && self
                .latest_root
                .as_ref()
                .is_none_or(AdaptiveShutdownReceipt::permits_root_reuse)
    }
    pub fn permits_process_close(&self) -> bool {
        self.permits_archive_close() && self.store_archive_closed
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveUciOutputObservation {
    pub authority: SearchAuthority,
    pub bestmove: String,
    pub from_report: bool,
}
#[derive(Clone, Debug)]
pub struct AdaptiveSessionRetention {
    pub stage: &'static str,
    pub retained_cpu_engines: usize,
    pub retained_cpu_workers: usize,
    pub native_pending: usize,
    pub archive_owner_pending: bool,
}
struct PendingOutput {
    authority: SearchAuthority,
    report: Option<BoardMove>,
    legal: Vec<BoardMove>,
}

pub struct AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    options: AdaptiveSessionOptions,
    limits: AdaptiveExecutionLimits,
    explicit_native_limits: bool,
    clock: ProcessClock,
    issuer: AdaptiveRulesOwnerIssuer,
    registration: PalsCheckerRegistration,
    checker_metadata: AdaptiveCheckerMetadata,
    external_metadata_bound: AtomicBool,
    state: Mutex<SessionState<B>>,
    work: ProcessWorkJournal,
    reset_requests: AtomicU64,
    closed: AtomicBool,
    dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
    closure_observer: Option<AdaptiveRootClosureObserver<B>>,
    collection_hooks: AdaptiveSessionCollectionHooks<B>,
    latest_root: Mutex<Option<AdaptiveShutdownReceipt>>,
    archive_receipts: Mutex<VecDeque<ArchiveWorkReceipt>>,
    pending_output: Mutex<Option<PendingOutput>>,
    accepted_outputs: Mutex<VecDeque<AdaptiveUciOutputObservation>>,
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    #[allow(
        clippy::result_large_err,
        reason = "admission rejection returns the original native, CPU and Store owners without a failure-path allocation"
    )]
    pub fn new(
        options: AdaptiveSessionOptions,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_with_resource_limits(
            options,
            AdaptiveExecutionLimits::default(),
            sources,
            clock,
            issuer,
            registration,
            dispatch_observer,
            closure_observer,
            AdaptiveSessionCollectionHooks::default(),
            false,
        )
    }
    /// Admit declarations only when the already-created engine ceilings agree.
    /// Root startup configures engines before capturing their immutable profile;
    /// this constructor never modifies a retained TT/token namespace in place.
    #[allow(
        clippy::too_many_arguments,
        clippy::result_large_err,
        reason = "independent admission authorities stay explicit and rejection returns their original owners without allocating"
    )]
    pub fn new_with_resource_limits(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_with_resource_limits(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            dispatch_observer,
            closure_observer,
            AdaptiveSessionCollectionHooks::default(),
            true,
        )
    }
    #[allow(
        clippy::too_many_arguments,
        clippy::result_large_err,
        reason = "independent admission authorities stay explicit and rejection returns their original owners without allocating"
    )]
    pub fn new_with_resource_limits_and_collection_hooks(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
        collection_hooks: AdaptiveSessionCollectionHooks<B>,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_with_resource_limits(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            dispatch_observer,
            closure_observer,
            collection_hooks,
            true,
        )
    }
    #[allow(
        clippy::too_many_arguments,
        clippy::result_large_err,
        reason = "independent admission authorities stay explicit and rejection returns their original owners without allocating"
    )]
    fn admit_with_resource_limits(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
        collection_hooks: AdaptiveSessionCollectionHooks<B>,
        explicit_native_limits: bool,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        let checked = (|| {
            options.validate()?;
            limits
                .validate()
                .map_err(|e| SearchSessionFailure::debug("AdaptiveResourceLimits", &e))?;
            if explicit_native_limits
                && (sources.native.pending_capacity() != limits.role_tickets_max
                    || sources.native.command_capacity() != limits.worker_commands_max)
            {
                return Err(failure(
                    "AdaptiveNativeCapacity",
                    "actual native channel differs from the declared resource limits",
                ));
            }
            if sources.cpu.len() != usize::from(options.controller.max_workers)
                || ![1, 2, 4].contains(&sources.cpu.len())
                || sources.engine.checker_identity() != &registration.identity
                || sources.engine.model_identity() != registration.role_model
                || sources.engine.review_policy().is_none()
                || sources.checker_metadata.identity() != &registration.identity
                || sources.checker_metadata.conditions() != registration.conditions
                || sources.checker_metadata.capabilities() != registration.capabilities
                || !owned_declarations_match(
                    sources.checker_metadata.owned_descriptor(),
                    registration.owned_descriptor.as_ref(),
                )
                || sources.checker_metadata.resume_policy() != registration.cpu_resume_policy
            {
                return Err(failure(
                    "AdaptiveSessionSources",
                    "actual owner/startup metadata mismatch",
                ));
            }
            let resolver = sources.engine.resolver_policy();
            if registration.resolver_version != resolver.version()
                || <[u8; 32]>::from(Sha256::digest(registration.resolver_semantics.as_bytes()))
                    != resolver.semantics_sha256()
            {
                return Err(failure(
                    "AdaptiveResolverDeclaration",
                    "actual resolver differs from registered namespace",
                ));
            }
            let expected_value =
                if resolver == rz_search::pals::engine::ResolverPolicy::ModelWdlRestricted {
                    sources.engine.model_value_identity()
                } else {
                    None
                };
            if expected_value != registration.model_value.as_ref() {
                return Err(failure(
                    "AdaptiveModelValueDeclaration",
                    "actual resolver model value differs from registered namespace",
                ));
            }
            let recheck = sources.engine.post_repair_recheck_policy();
            if sources.engine.search_identity() != recheck.search_identity()
                || sources.engine.refinement_conditions() != recheck.conditions()
            {
                return Err(failure(
                    "AdaptiveSearchLaneUnsupported",
                    "effective lane differs from the registered receipt consumer",
                ));
            }
            for (at, owner) in sources.cpu.iter().enumerate() {
                if owner.engine.paused_stack_max_bytes() != limits.paused_bytes_per_worker_max {
                    return Err(failure(
                        "AdaptiveCpuNamespace",
                        "actual paused owner ceiling differs from the resource declaration",
                    ));
                }
                if matches!(registration.identity, CheckerIdentity::Owned(_)) {
                    if at == 0 {
                        sources
                            .checker_metadata
                            .validate_owned_engine(&owner.engine)
                            .map_err(|e| SearchSessionFailure::debug("AdaptiveCpuNamespace", &e))?;
                    } else {
                        sources
                            .checker_metadata
                            .validate_owned_namespace(&owner.engine)
                            .map_err(|e| SearchSessionFailure::debug("AdaptiveCpuNamespace", &e))?;
                    }
                }
                if owner.profile.max_depth < options.cpu_depth {
                    return Err(failure(
                        "AdaptiveCpuNamespace",
                        "declared depth exceeds actual CPU owner",
                    ));
                }
            }
            if sources.native.ready_clock().is_some_and(|ready| {
                ready.origin() != clock.runtime_clock().origin() || ready.domain() != clock.domain()
            }) {
                return Err(failure(
                    "AdaptiveSessionClock",
                    "actual native process origin differs",
                ));
            }
            if let Some(external) = &sources.external {
                if !external
                    .worker
                    .configuration_matches(&external.config, external.owner)
                {
                    return Err(failure(
                        "AdaptiveExternalNamespace",
                        "actual external owner configuration differs",
                    ));
                }
            }
            if matches!(registration.identity, CheckerIdentity::ExternalUci(_))
                && sources.external.is_none()
            {
                return Err(failure(
                    "AdaptiveExternalNamespace",
                    "external checker has no actual process owner",
                ));
            }
            Ok(())
        })();
        if let Err(failure) = checked {
            return Err(AdaptiveSessionAdmissionRejection { failure, sources });
        }
        let resolver = PalsResolverIdentity::from_semantics(
            registration.resolver_version,
            registration.resolver_semantics,
        );
        let recheck = sources.engine.post_repair_recheck_policy();
        let work = match (
            recheck.registration_version(),
            recheck.registration_policy(),
            recheck.conditions(),
        ) {
            (Some(version), Some(policy), Some(conditions)) => {
                let identity = PalsSearchPolicyIdentity {
                    version: version.into(),
                    policy: policy.into(),
                    search_identity: recheck.search_identity().into(),
                    conditions_sha256: Sha256::digest(conditions.as_bytes()).into(),
                };
                match ProcessWorkJournal::new_pals_with_search_policy(resolver, identity) {
                    Ok(work) => work,
                    Err(error) => {
                        return Err(AdaptiveSessionAdmissionRejection {
                            failure: SearchSessionFailure::debug("AdaptiveSearchPolicy", &error),
                            sources,
                        });
                    }
                }
            }
            _ => ProcessWorkJournal::new_pals(resolver),
        };
        let AdaptiveSessionSources {
            native,
            engine,
            cpu,
            external,
            checker_metadata,
        } = sources;
        Ok(Self {
            options,
            limits,
            explicit_native_limits,
            clock,
            issuer,
            registration,
            checker_metadata,
            external_metadata_bound: AtomicBool::new(false),
            state: Mutex::new(SessionState::Idle(Box::new(IdleOwners {
                native,
                engine: Some(Box::new(engine)),
                cpu,
                external,
                continuation: None,
                native_finished: None,
                native_joined: false,
                external_finished: None,
            }))),
            work,
            reset_requests: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            dispatch_observer,
            closure_observer,
            collection_hooks,
            latest_root: Mutex::new(None),
            archive_receipts: Mutex::new(VecDeque::with_capacity(64)),
            pending_output: Mutex::new(None),
            accepted_outputs: Mutex::new(VecDeque::with_capacity(64)),
        })
    }

    pub fn process_epoch(&self) -> ProcessEpoch {
        self.clock.epoch()
    }
    pub fn execution_limits(&self) -> AdaptiveExecutionLimits {
        self.limits
    }
    /// A nonblocking metadata read of the original idle native owner. This
    /// does not observe the bootstrap caller as the native thread, start any
    /// work, or turn a failed/absent OS reading into an applied placement.
    pub fn native_ready_cpu_thread_observation(
        &self,
    ) -> Result<
        Option<
            Result<rz_runtime::adaptive::CpuThreadReadback, rz_runtime::adaptive::CpuResourceError>,
        >,
        SearchSessionFailure,
    > {
        let state = self.state.try_lock().map_err(|_| {
            failure(
                "AdaptiveResourceOwnerBusy",
                "original owner busy or poisoned",
            )
        })?;
        match &*state {
            SessionState::Idle(owners) if !self.closed.load(Ordering::Acquire) => {
                Ok(owners.native.ready_cpu_thread_observation())
            }
            _ => Err(failure(
                "AdaptiveResourceOwnerUnavailable",
                "native startup owner is not idle and live",
            )),
        }
    }
    /// Only the successful actual startup owner's immutable process identity
    /// crosses this boundary. The read-only resource probe must bracket PID,
    /// process group and start ticks itself; readyok proves no applied option.
    pub fn external_startup_process_identity(
        &self,
    ) -> Result<Option<rz_search::cpu_checker::ExternalProcessIdentity>, SearchSessionFailure> {
        let state = self.state.try_lock().map_err(|_| {
            failure(
                "AdaptiveResourceOwnerBusy",
                "original owner busy or poisoned",
            )
        })?;
        let SessionState::Idle(owners) = &*state else {
            return Err(failure(
                "AdaptiveResourceOwnerUnavailable",
                "external startup owner is not idle",
            ));
        };
        if self.closed.load(Ordering::Acquire) {
            return Err(failure(
                "AdaptiveResourceOwnerUnavailable",
                "admission closed",
            ));
        }
        let Some(external) = &owners.external else {
            return Ok(None);
        };
        let startup = external.worker.startup_report().ok_or_else(|| {
            failure(
                "AdaptiveResourceStartupUnknown",
                "actual startup not observed",
            )
        })?;
        startup.result.map_err(|error| {
            SearchSessionFailure::debug("AdaptiveResourceStartupFailed", &error)
        })?;
        let process = startup
            .attempt
            .and_then(|attempt| attempt.external)
            .map(|attempt| attempt.process)
            .ok_or_else(|| {
                failure(
                    "AdaptiveResourceIdentityUnknown",
                    "original process evidence missing",
                )
            })?;
        if process.quit_sent
            || process.exit_observed
            || process.cleanup_complete
            || process.quarantined
            || process.ownership_lost
        {
            return Err(failure(
                "AdaptiveResourceOwnerUnavailable",
                "historical startup process is no longer reusable",
            ));
        }
        let identity = process.process_identity.filter(|identity| {
            identity.pid > 0
                && identity.process_group > 0
                && identity.proc_start_ticks > 0
                && identity.pid != std::process::id()
                && identity.pid <= i32::MAX as u32
                && identity.process_group <= i32::MAX as u32
        });
        identity.map(Some).ok_or_else(|| {
            failure(
                "AdaptiveResourceIdentityUnknown",
                "native spawn identity unavailable",
            )
        })
    }
    // The actual native bootstrap alone owns the private collection context.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn collection_context_ready(&self) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(poisoned("adaptive process admission closed"));
        }
        let state = self
            .state
            .try_lock()
            .map_err(|_| poisoned("actual adaptive owner is busy or poisoned"))?;
        if !matches!(&*state, SessionState::Idle(_)) {
            return Err(poisoned("actual adaptive owner is retained or unclosed"));
        }
        let pending = self
            .pending_output
            .lock()
            .map_err(|_| poisoned("adaptive collection output authority"))?;
        if pending.is_some() {
            return Err(poisoned("previous adaptive output authority is unretired"));
        }
        Ok(())
    }
    /// Revocation is owned by the UCI collection context wrapper. It grants no
    /// UCI publication observation and cannot acknowledge an emitted bestmove.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn retire_collection_output(
        &self,
        authority: SearchAuthority,
    ) -> Result<(), ContractError> {
        // Match execute's state -> pending lock order. A running Search owner
        // may still publish its final pending report; retire only after it has
        // returned, and retain the outer active context when this guard rejects.
        let _state = self
            .state
            .try_lock()
            .map_err(|_| poisoned("actual adaptive owner has not returned"))?;
        let mut pending = self
            .pending_output
            .lock()
            .map_err(|_| poisoned("adaptive collection output authority"))?;
        if pending
            .as_ref()
            .is_some_and(|registered| registered.authority != authority)
        {
            return Err(poisoned("stale adaptive collection retirement"));
        }
        *pending = None;
        Ok(())
    }
    pub fn retained_owner_status(&self) -> Result<AdaptiveSessionRetention, ContractError> {
        // Failure diagnostics must not wait behind the very search whose
        // original drain deadline has expired. Busy is unobserved retention,
        // never evidence that the retained owners or physical leases are gone.
        let state = self
            .state
            .try_lock()
            .map_err(|_| poisoned("adaptive owner retention unavailable or busy"))?;
        let snapshot = |stage, owners: &IdleOwners<B>, workers, archive| AdaptiveSessionRetention {
            stage,
            retained_cpu_engines: owners.cpu.len(),
            retained_cpu_workers: workers,
            native_pending: owners.native.occupancy(),
            archive_owner_pending: archive,
        };
        Ok(match &*state {
            SessionState::Idle(owners) => snapshot("idle", owners, 0, false),
            SessionState::Manager(manager) => AdaptiveSessionRetention {
                stage: "manager_retained",
                retained_cpu_engines: 0,
                retained_cpu_workers: usize::from(self.options.controller.max_workers),
                native_pending: manager.native_occupancy(),
                archive_owner_pending: manager.driver().port().is_none(),
            },
            SessionState::AdmissionRejected(rejected) => AdaptiveSessionRetention {
                stage: "admission_rejected",
                retained_cpu_engines: 0,
                retained_cpu_workers: rejected.sources.own_workers.len(),
                native_pending: rejected.sources.native.occupancy(),
                archive_owner_pending: rejected.sources.driver.port().is_none(),
            },
            SessionState::PortRejected { owners, driver } => {
                snapshot("port_rejected", owners, 0, driver.port().is_none())
            }
            SessionState::SpawnRejected {
                owners,
                driver,
                workers,
            } => snapshot(
                "cpu_spawn_rejected",
                owners,
                workers.len(),
                driver.port().is_none(),
            ),
            SessionState::ArchivePending {
                owners,
                worker,
                transfer,
                authority,
            } => {
                let _scope = (transfer.nonce(), authority);
                snapshot("archive_pending", owners, 0, worker.is_busy())
            }
            SessionState::ArchiveQuarantined {
                owners,
                worker,
                quarantine,
            } => {
                let _reason = &quarantine.reason;
                snapshot("archive_quarantined", owners, 0, worker.is_busy())
            }
            SessionState::ArchiveJoinedPending { owners, worker } => {
                snapshot("archive_join_pending", owners, 0, worker.is_busy())
            }
            SessionState::Closed(_) => AdaptiveSessionRetention {
                stage: "process_closed",
                retained_cpu_engines: 0,
                retained_cpu_workers: 0,
                native_pending: 0,
                archive_owner_pending: false,
            },
            SessionState::Moving => AdaptiveSessionRetention {
                stage: "owner_transfer_unknown",
                retained_cpu_engines: 0,
                retained_cpu_workers: 0,
                native_pending: 0,
                archive_owner_pending: true,
            },
        })
    }
    pub fn latest_root_receipt(&self) -> Result<Option<AdaptiveShutdownReceipt>, ContractError> {
        self.latest_root
            .lock()
            .map(|r| r.clone())
            .map_err(|_| poisoned("adaptive root receipt"))
    }
    pub fn uci_output_observations(
        &self,
    ) -> Result<Vec<AdaptiveUciOutputObservation>, ContractError> {
        self.accepted_outputs
            .lock()
            .map(|r| r.iter().cloned().collect())
            .map_err(|_| poisoned("adaptive output observations"))
    }

    fn archive_operation(
        &self,
        state: &mut SessionState<B>,
        mut owners: Box<IdleOwners<B>>,
        next: SearchAuthority,
        operation: ArchiveOperation,
        until: Instant,
        context: &SearchSessionContext,
    ) -> Result<Box<IdleOwners<B>>, SearchSessionFailure> {
        let engine = owners.engine.take().expect("exclusive idle Search owner");
        let authority = match engine.adaptive_archive_authority(next) {
            Ok(authority) => authority,
            Err(error) => {
                owners.engine = Some(engine);
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug(
                    "AdaptiveArchiveAuthority",
                    &error,
                ));
            }
        };
        let signal = owners.native.signal();
        let mut worker = match ArchiveWorker::start_with_notifier(move || signal.notify()) {
            Ok(worker) => worker,
            Err(error) => {
                owners.engine = Some(engine);
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug("AdaptiveArchiveOwner", &error));
            }
        };
        let transfer = match worker.submit(
            ArchiveOwner::Engine(*engine),
            authority,
            ArchiveWork {
                operation,
                deadline: until,
                max_io_bytes: 16 * 1024 * 1024,
            },
        ) {
            Ok(transfer) => transfer,
            Err(rejected) => {
                let error =
                    SearchSessionFailure::debug("AdaptiveArchiveAdmission", &rejected.error);
                if let ArchiveOwner::Engine(engine) = rejected.owner {
                    owners.engine = Some(Box::new(engine));
                }
                owners = join_archive_lifecycle(state, owners, worker, context.shutdown_deadline)?;
                *state = SessionState::Idle(owners);
                return Err(error);
            }
        };
        loop {
            match worker.try_receive(&transfer) {
                Ok(Some(completed)) => match completed.install(authority, transfer.nonce()) {
                    Ok((ArchiveOwner::Engine(engine), receipt)) => {
                        owners.engine = Some(Box::new(engine));
                        match self.archive_receipts.lock() {
                            Ok(mut receipts) => {
                                if receipts.len() == 64 {
                                    receipts.pop_front();
                                }
                                receipts.push_back(receipt);
                            }
                            Err(_) => {
                                worker.request_shutdown();
                                *state = SessionState::ArchiveJoinedPending { owners, worker };
                                return Err(SearchSessionFailure::physical_completion_unknown());
                            }
                        }
                        return join_archive_lifecycle(
                            state,
                            owners,
                            worker,
                            context.shutdown_deadline,
                        );
                    }
                    Ok((owner, _)) => {
                        // Submission was Engine-only; retain any invalid owner.
                        let ArchiveOwner::Port(port) = owner else {
                            unreachable!()
                        };
                        owners.engine = Some(Box::new(port.into_engine()));
                        owners = join_archive_lifecycle(
                            state,
                            owners,
                            worker,
                            context.shutdown_deadline,
                        )?;
                        *state = SessionState::Idle(owners);
                        return Err(failure(
                            "AdaptiveArchiveReturn",
                            "returned owner kind differs",
                        ));
                    }
                    Err(quarantine) => {
                        *state = SessionState::ArchiveQuarantined {
                            owners,
                            worker,
                            quarantine: Box::new(quarantine),
                        };
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                },
                Ok(None) if Instant::now() < until && context.accepts().unwrap_or(false) => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                _ => {
                    transfer.cancel();
                    worker.request_shutdown();
                    *state = SessionState::ArchivePending {
                        owners,
                        worker,
                        transfer,
                        authority,
                    };
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        }
    }

    fn execute(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
        observation: &mut AttemptObservation,
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        if context.control.max_simulations == 0
            || context.control.max_simulations > self.options.max_cpu_nodes
        {
            return Err(failure(
                "AdaptiveCpuCredit",
                "go CPU node credit exceeds registered startup maximum",
            ));
        }
        if self.closed.load(Ordering::Acquire) {
            // closed may mean retained failed-lifecycle quarantine, not an
            // actual physical finish. A later go cannot turn it into success.
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        if context.authority.epoch != self.clock.epoch()
            || context.authority.implementation != self.options.implementation
        {
            return Err(failure(
                "AdaptiveSessionAuthority",
                "closed or mismatched UCI process authority",
            ));
        }
        let mut state = loop {
            match self.state.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::WouldBlock) if context.accepts().unwrap_or(false) => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Err(TryLockError::WouldBlock) => {
                    return Err(failure(
                        "AdaptiveSessionBusy",
                        "previous search/lifecycle owner has not returned",
                    ));
                }
                Err(TryLockError::Poisoned(_)) => {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        };
        let mut owners = match std::mem::replace(&mut *state, SessionState::Moving) {
            SessionState::Idle(owners) => owners,
            retained => {
                *state = retained;
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if !context.accepts().unwrap_or(false) {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "AdaptiveSessionCanceled",
                "root revoked before actual admission",
            ));
        }
        let root_legal = position.legal_moves();
        match self.pending_output.lock() {
            Ok(mut output) => {
                *output = Some(PendingOutput {
                    authority: context.authority,
                    report: None,
                    legal: root_legal.clone(),
                })
            }
            Err(_) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        }
        if let Err(error) = self.bind_actual_external(&owners) {
            *state = SessionState::Idle(owners);
            return Err(error);
        }
        let engine = owners.engine.as_ref().expect("idle Search owner");
        let game = match engine.adaptive_archive_authority(context.authority) {
            Ok(authority) => authority.game,
            Err(error) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug("AdaptiveRootNamespace", &error));
            }
        };
        if game != context.authority.game {
            if let Err(error) = self.reset_actual_cpu(&mut owners, context) {
                // A failed reset may have changed an actual external game or
                // cleared a CPU owner. Retain it for process cleanup instead
                // of retrying against the old semantic Store authority.
                self.closed.store(true, Ordering::Release);
                *state = SessionState::Idle(owners);
                return Err(error);
            }
            owners = match self.archive_operation(
                &mut state,
                owners,
                context.authority,
                ArchiveOperation::NewGame {
                    next_authority: context.authority,
                },
                context.control.deadline,
                context,
            ) {
                Ok(owners) => owners,
                Err(error) => {
                    // A reset arm may have been issued but not consumed by the
                    // failed lifecycle. Preserve it with its exact owner and
                    // close further roots; process finish is the only recovery.
                    self.closed.store(true, Ordering::Release);
                    return Err(error);
                }
            };
        }
        if owners
            .engine
            .as_ref()
            .expect("Search owner")
            .archive_enabled()
        {
            owners = self.archive_operation(
                &mut state,
                owners,
                context.authority,
                ArchiveOperation::PrepareRoot {
                    position: position.clone(),
                },
                context.control.deadline,
                context,
            )?;
        }
        let deadline = match self.clock.deadline(context.control.deadline) {
            Ok(deadline) => deadline,
            Err(error) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug("AdaptiveRootDeadline", &error));
            }
        };
        let mut engine = owners.engine.take().expect("Search owner");
        let root = match engine.register_adaptive_root(position, context.authority) {
            Ok(root) => root,
            Err(error) => {
                owners.engine = Some(engine);
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug(
                    "AdaptiveRootRegistration",
                    &error,
                ));
            }
        };
        let limits = PalsLimits {
            deadline: context.control.deadline,
            max_rounds: self.options.max_rounds,
            max_cpu_nodes: context.control.max_simulations,
            cpu_depth: self.options.cpu_depth,
        };
        let port = match AdaptiveEnginePort::new_registered_retaining(
            *engine,
            position.clone(),
            context.authority,
            deadline,
            limits,
            Arc::clone(&self.issuer),
            root,
        ) {
            Ok(port) => port,
            Err((mut engine, error)) => {
                let abandoned = engine.abandon_unconsumed_adaptive_root(context.authority);
                let failure = SearchSessionFailure::debug("AdaptiveRootPort", &(error, abandoned));
                owners.engine = Some(Box::new(engine));
                *state = SessionState::Idle(owners);
                return Err(failure);
            }
        };
        let mut driver = match AdaptivePortDriver::new_with_coordinator_policy_retaining(
            port,
            context.control.deadline,
            Arc::new(owners.native.signal()),
            self.options.external_policy,
            self.options.policy,
        ) {
            Ok(driver) => driver,
            Err((port, error)) => {
                owners.engine = Some(Box::new(port.into_engine()));
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug("AdaptiveSearchOwner", &error));
            }
        };
        if let Some(observer) = &self.dispatch_observer {
            if let Err(error) = driver.set_collection_dispatch_observer(Arc::clone(observer)) {
                *state = SessionState::PortRejected { owners, driver };
                return Err(SearchSessionFailure::debug(
                    "AdaptiveCollectionObserver",
                    &error,
                ));
            }
        }
        if let Some(observer) = &self.collection_hooks.external_teacher {
            if let Err(error) = driver.set_external_teacher_observer(Arc::clone(observer)) {
                *state = SessionState::PortRejected { owners, driver };
                return Err(SearchSessionFailure::debug(
                    "AdaptiveTeacherObserver",
                    &error,
                ));
            }
        }
        let mut workers = Vec::with_capacity(owners.cpu.len());
        while let Some(owner) = owners.cpu.pop() {
            match CpuWorker::spawn_notified_retaining_with_capacity(
                owner.profile.owner,
                owner.engine,
                owners.native.signal(),
                self.limits.worker_commands_max,
            ) {
                Ok(worker) if worker.profile() == owner.profile => workers.push(worker),
                Ok(worker) => {
                    workers.push(worker);
                    return self.retain_spawn_failure(&mut state, owners, driver, workers, None);
                }
                Err((engine, error)) => {
                    owners.cpu.push(PersistentCpuOwner {
                        profile: owner.profile,
                        engine,
                    });
                    return self.retain_spawn_failure(
                        &mut state,
                        owners,
                        driver,
                        workers,
                        Some(error),
                    );
                }
            }
        }
        workers.reverse();
        let first_request_sequence = owners
            .continuation
            .as_ref()
            .map_or(1, |c| c.first_request_sequence());
        let registration = AdaptiveManagerRegistration {
            authority: context.authority,
            hard_deadline: context.control.deadline,
            token_budget: self.options.token_budget,
            policy: self.options.policy,
            first_request_sequence,
            native_cpu_node_limit: context.control.max_simulations,
            endpoints: self.options.endpoints,
            controller: self.options.controller,
            device_digest: self.options.device_digest,
            cpu_device_digest: self.options.cpu_device_digest,
        };
        let sources = AdaptiveManagerSources {
            native: owners.native,
            own_workers: workers,
            external: owners.external,
            driver,
        };
        let admitted = match (owners.continuation, self.explicit_native_limits) {
            (Some(continuation), true) => SessionManager::new_continued_retaining_with_limits(
                registration,
                self.clock.runtime_clock(),
                self.limits,
                sources,
                continuation,
            ),
            (None, true) => SessionManager::new_retaining_with_limits(
                registration,
                self.clock.runtime_clock(),
                self.limits,
                sources,
            ),
            (Some(continuation), false) => SessionManager::new_continued_retaining(
                registration,
                self.clock.runtime_clock(),
                sources,
                continuation,
            ),
            (None, false) => {
                SessionManager::new_retaining(registration, self.clock.runtime_clock(), sources)
            }
        };
        let mut manager = match admitted {
            Ok(manager) => Box::new(manager),
            Err(rejected) => {
                let error =
                    SearchSessionFailure::debug("AdaptiveManagerAdmission", &rejected.error);
                *state = SessionState::AdmissionRejected(Box::new(rejected));
                return Err(error);
            }
        };
        *observation = AttemptObservation::PalsStarted;
        let admitted = catch_unwind(AssertUnwindSafe(|| {
            if let Some(observer) = &self.collection_hooks.root_admission {
                observer(&mut manager)?;
            }
            manager.driver_mut().prepare_root_rules()?;
            if manager.driver().root_is_terminal()? {
                manager.begin_root_quiesce(context.shutdown_deadline)?;
                drain_root(&mut manager, context.shutdown_deadline)?;
                manager.shutdown_receipt()
            } else {
                let close_at = context
                    .control
                    .soft_deadline
                    .min(context.control.admission_deadline);
                if Instant::now() >= close_at || !context.accepts().unwrap_or(false) {
                    manager.begin_root_quiesce(context.shutdown_deadline)?;
                    drain_root(&mut manager, context.shutdown_deadline)?;
                    return manager.shutdown_receipt();
                }
                let input = manager.driver_mut().root_input()?;
                manager.prepare_root_ranking(input)?;
                let mut last_published = None;
                let mut last_semantic_snapshot = None;
                manager.run_root_until_observed(
                    context.shutdown_deadline,
                    &context.control.cancellation,
                    &mut |manager| {
                        if manager.phase() == ManagerPhase::Running
                            && (Instant::now() >= close_at || !context.accepts().unwrap_or(false))
                        {
                            manager.begin_root_quiesce(context.shutdown_deadline)?;
                        }
                        let current = manager.driver().port().map(|port| port.counters());
                        if context.accepts().unwrap_or(false)
                            && current.is_some()
                            && current != last_semantic_snapshot
                        {
                            // Quiet notification ticks inspect only controls.
                            // A Search counter mutation triggers the bounded
                            // resolver; it is not a completed CPU frame claim.
                            last_semantic_snapshot = current;
                            // Only a coherent Search result can publish a move;
                            // callback ticks and partial CPU frames grant none.
                            if let Ok(result) = manager.driver().result(PalsCompletion::RoundLimit)
                            {
                                if let Some(movement) = result.best_move {
                                    if last_published != Some(movement)
                                        && root_legal.contains(&movement)
                                    {
                                        progress(movement);
                                        last_published = Some(movement);
                                    }
                                }
                            }
                        }
                        Ok(())
                    },
                )
            }
        }));
        let admitted = match admitted {
            Ok(result) => result,
            Err(_) => {
                *state = SessionState::Manager(manager);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if let Err(error) = admitted {
            let _cleanup = catch_unwind(AssertUnwindSafe(|| {
                let _ = manager.begin_root_quiesce(context.shutdown_deadline);
                let _ = drain_root(&mut manager, context.shutdown_deadline);
            }));
            // Even cleanup panic retains the whole actual manager before any
            // caller-visible failure; a local owner must not unwind and drop.
            *state = SessionState::Manager(manager);
            return Err(
                SearchSessionFailure::debug("AdaptiveRootExecution", &error).unknown_if_unclosed()
            );
        }
        let receipt = match manager.shutdown_receipt() {
            Ok(receipt) => receipt,
            Err(_) => {
                *state = SessionState::Manager(manager);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if !receipt.permits_root_reuse() || manager.phase() != ManagerPhase::Finished {
            *state = SessionState::Manager(manager);
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
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
        let semantic = match catch_unwind(AssertUnwindSafe(|| manager.driver().result(completion)))
        {
            Ok(result) => result,
            Err(_) => {
                *state = SessionState::Manager(manager);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if let Ok(result) = &semantic {
            *observation = AttemptObservation::Pals {
                counters: result.counters,
                root_coverage_observed: result.counters.root_scope_observation_complete,
            };
        }
        let collection =
            match catch_unwind(AssertUnwindSafe(|| -> Result<(), AdaptiveManagerError> {
                self.closure_observer
                    .as_ref()
                    .map_or(Ok(()), |observer| observer(&mut manager))?;
                if let Some(observer) = &self.collection_hooks.root_closure {
                    let completion = semantic
                        .as_ref()
                        .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?
                        .completion;
                    observer(&mut manager, completion)?;
                }
                Ok(())
            })) {
                Ok(result) => result,
                Err(_) => {
                    *state = SessionState::Manager(manager);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            };
        let retained_failure = manager.failures().next().cloned();
        let parts = match manager.into_persistent_parts() {
            Ok(parts) => parts,
            Err(rejected) => {
                *state = SessionState::Manager(Box::new(rejected.manager));
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        let (native, cpu, external, driver, continuation, receipt) = parts.into_components();
        let mut owners = Box::new(IdleOwners {
            native,
            cpu,
            external,
            engine: None,
            continuation: Some(continuation),
            native_finished: None,
            native_joined: false,
            external_finished: None,
        });
        match driver.into_port(&receipt) {
            Ok(port) => owners.engine = Some(Box::new(port.into_engine())),
            Err((driver, _)) => {
                *state = SessionState::PortRejected { owners, driver };
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        }
        *state = SessionState::Idle(owners);
        *self
            .latest_root
            .lock()
            .map_err(|_| failure("AdaptiveRootReceipt", "receipt mutex poisoned"))? = Some(receipt);
        collection.map_err(|e| SearchSessionFailure::debug("AdaptiveCollectionClosure", &e))?;
        if let Some(error) = retained_failure {
            return Err(SearchSessionFailure::debug("AdaptiveRootFailure", &error));
        }
        let result = semantic.map_err(|e| SearchSessionFailure::debug("AdaptiveRootResult", &e))?;
        if result.resolver_version != self.registration.resolver_version {
            return Err(failure(
                "AdaptiveResolverChanged",
                "actual result resolver differs from startup",
            ));
        }
        *self
            .pending_output
            .lock()
            .map_err(|_| failure("AdaptiveOutputOwner", "output mutex poisoned"))? =
            Some(PendingOutput {
                authority: context.authority,
                report: result.best_move,
                legal: root_legal,
            });
        if context.accepts().unwrap_or(false) {
            if let Some(movement) = result.best_move {
                progress(movement);
            }
        }
        Ok(SearchSessionReport {
            best_move: result.best_move,
            work: pals_work(result.counters),
        })
    }

    fn retain_spawn_failure(
        &self,
        state: &mut SessionState<B>,
        owners: Box<IdleOwners<B>>,
        driver: AdaptivePortDriver<AdaptivePortModel>,
        workers: Vec<CpuWorker>,
        error: Option<rz_search::cpu_worker::CpuWorkerError>,
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        // Every started thread and every not-started CPU engine remains owned.
        // The driver retains these owners in a dedicated failed boundary.
        *state = SessionState::SpawnRejected {
            owners,
            driver,
            workers,
        };
        Err(error
            .as_ref()
            .map_or_else(SearchSessionFailure::physical_completion_unknown, |e| {
                SearchSessionFailure::debug("AdaptiveCpuSpawn", e).unknown_if_unclosed()
            }))
    }

    fn bind_actual_external(&self, owners: &IdleOwners<B>) -> Result<(), SearchSessionFailure> {
        if !matches!(self.registration.identity, CheckerIdentity::ExternalUci(_))
            || self.external_metadata_bound.load(Ordering::Acquire)
        {
            return Ok(());
        }
        let external = owners.external.as_ref().ok_or_else(|| {
            failure(
                "AdaptiveExternalOwnerBinding",
                "actual external owner missing",
            )
        })?;
        self.checker_metadata
            .bind_external_owner(
                &external.worker,
                &external.config,
                external.owner,
                self.clock.epoch(),
                self.options.implementation,
            )
            .map_err(|e| SearchSessionFailure::debug("AdaptiveExternalOwnerBinding", &e))?;
        self.external_metadata_bound.store(true, Ordering::Release);
        Ok(())
    }

    fn reset_actual_cpu(
        &self,
        owners: &mut IdleOwners<B>,
        context: &SearchSessionContext,
    ) -> Result<(), SearchSessionFailure> {
        if let Some(external) = &owners.external {
            external
                .worker
                .request_new_game(context.authority, context.control.deadline)
                .map_err(|e| SearchSessionFailure::debug("AdaptiveExternalNewGame", &e))?;
            loop {
                match external.worker.new_game_fence() {
                    Some(Ok(proof)) => {
                        if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_)) {
                            self.checker_metadata
                                .arm_external_new_game(&external.config, &proof)
                                .map_err(|e| {
                                    SearchSessionFailure::debug("AdaptiveExternalResetProof", &e)
                                })?;
                        }
                        break;
                    }
                    Some(Err(error)) => {
                        return Err(SearchSessionFailure::debug("AdaptiveExternalReset", &error));
                    }
                    None if context.accepts().unwrap_or(false) => {
                        std::thread::sleep(Duration::from_millis(1))
                    }
                    None => return Err(SearchSessionFailure::physical_completion_unknown()),
                }
            }
        }
        for (index, owner) in owners.cpu.iter_mut().enumerate() {
            if index == 0 && matches!(self.registration.identity, CheckerIdentity::Owned(_)) {
                self.checker_metadata
                    .reset_owned_and_arm(
                        &mut owner.engine,
                        context.control.deadline,
                        &context.control.cancellation,
                    )
                    .map_err(|e| SearchSessionFailure::debug("AdaptiveOwnCpuReset", &e))?;
            } else {
                owner.engine.clear();
            }
        }
        Ok(())
    }

    /// Explicit process finish after UCI has retired its output authority. The
    /// same native session receives Finish and is joined once; root receipts do
    /// not stand in for this actual physical release.
    pub fn finish_process(
        &self,
        until: Instant,
    ) -> Result<AdaptiveSessionProcessReceipt, SearchSessionFailure> {
        self.closed.store(true, Ordering::Release);
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        if let SessionState::Closed(receipt) = &*state {
            return Ok(receipt.clone());
        }
        let retained = std::mem::replace(&mut *state, SessionState::Moving);
        let retained = match retained {
            SessionState::Manager(mut manager) => {
                let root_reusable = manager.phase() == ManagerPhase::Finished
                    && manager
                        .shutdown_receipt()
                        .is_ok_and(|r| r.permits_root_reuse());
                if !root_reusable {
                    let finish = catch_unwind(AssertUnwindSafe(|| {
                        manager.begin_shutdown(until)?;
                        manager.run_until(until, &AtomicBool::new(false))
                    }));
                    if !matches!(finish, Ok(Ok(ref receipt)) if receipt.permits_archive_close()) {
                        *state = SessionState::Manager(manager);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                }
                let parts = if root_reusable {
                    manager.into_persistent_parts()
                } else {
                    manager.into_process_parts()
                };
                let parts = match parts {
                    Ok(parts) => parts,
                    Err(rejected) => {
                        *state = SessionState::Manager(Box::new(rejected.manager));
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                };
                let (native, cpu, external, driver, continuation, receipt) =
                    parts.into_components();
                let mut owners = Box::new(IdleOwners {
                    native,
                    cpu,
                    external,
                    engine: None,
                    continuation: Some(continuation),
                    native_finished: if root_reusable {
                        None
                    } else {
                        Some(receipt.native_physical_shutdown_confirmed)
                    },
                    native_joined: receipt.native_owner_joined,
                    external_finished: None,
                });
                if !root_reusable {
                    if let Some(external) = &owners.external {
                        match external.worker.shutdown_snapshot() {
                            Some(Ok(receipt)) => owners.external_finished = Some(receipt),
                            _ => {
                                *state = SessionState::PortRejected { owners, driver };
                                return Err(SearchSessionFailure::physical_completion_unknown());
                            }
                        }
                    }
                }
                match driver.into_port(&receipt) {
                    Ok(port) => owners.engine = Some(Box::new(port.into_engine())),
                    Err((driver, _)) => {
                        *state = SessionState::PortRejected { owners, driver };
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                }
                SessionState::Idle(owners)
            }
            other => other,
        };
        let mut owners = match retained {
            SessionState::Idle(owners) => owners,
            retained => {
                *state = retained;
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        owners.native.begin_shutdown(until);
        if let Some(external) = &owners.external {
            external.worker.request_shutdown(until);
        }
        loop {
            let version = owners.native.signal().version();
            while let Some(event) = owners.native.poll() {
                match event {
                    TicketNativeEvent::Finished {
                        physical_shutdown_confirmed,
                    } => owners.native_finished = Some(physical_shutdown_confirmed),
                    TicketNativeEvent::Ready { .. } | TicketNativeEvent::RootQuiesced(_) => {}
                    _ => {
                        *state = SessionState::Idle(owners);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                }
            }
            if !owners.native_joined {
                match owners.native.try_join() {
                    Ok(joined) => owners.native_joined = joined,
                    Err(_) => {
                        *state = SessionState::Idle(owners);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                }
            }
            if owners.external_finished.is_none() {
                if let Some(external) = &mut owners.external {
                    if let Some(result) = external.worker.try_join() {
                        match result {
                            Ok(receipt) => owners.external_finished = Some(receipt),
                            Err(_) => {
                                *state = SessionState::Idle(owners);
                                return Err(SearchSessionFailure::physical_completion_unknown());
                            }
                        }
                    }
                }
            }
            let external_complete = owners.external.is_none()
                || owners
                    .external_finished
                    .is_some_and(|r| r.cleanup_complete && !r.quarantined && !r.ownership_lost);
            if owners.native_joined && owners.native_finished == Some(true) && external_complete {
                break;
            }
            if Instant::now() >= until || owners.native_finished == Some(false) {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            owners
                .native
                .signal()
                .wait_changed(version, Duration::from_millis(1));
        }
        let engine = owners.engine.as_mut().expect("quiescent Search owner");
        let last_authority = engine.last_adaptive_authority();
        let closed = match catch_unwind(AssertUnwindSafe(|| engine.finish_followup_owners(until))) {
            Ok(Ok(snapshot)) => snapshot
                .archive
                .as_ref()
                .is_ok_and(|archive| archive.as_ref().is_none_or(|a| a.admission_closed)),
            _ => false,
        };
        if !closed {
            *state = SessionState::Idle(owners);
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let latest_root = match self.latest_root.lock() {
            Ok(receipt) => receipt.clone(),
            Err(_) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        let receipt = AdaptiveSessionProcessReceipt {
            epoch: self.clock.epoch(),
            last_authority,
            native_owner_joined: owners.native_joined,
            native_physical_shutdown_confirmed: owners.native_finished == Some(true),
            external_shutdown: owners.external_finished,
            store_archive_closed: closed,
            latest_root,
        };
        if !receipt.permits_process_close() {
            *state = SessionState::Idle(owners);
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        *state = SessionState::Closed(receipt.clone());
        Ok(receipt)
    }
}

impl<B> SearchSessionDriver for AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    fn kind(&self) -> SearchKind {
        SearchKind::Pals
    }
    fn implementation(&self) -> Digest {
        self.options.implementation
    }
    fn identity(&self) -> EngineIdentity {
        self.options.identity.clone()
    }
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        self.work.snapshot().map(Some)
    }
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        Some(self.registration.clone())
    }
    fn checker_started(&self) -> Option<bool> {
        match self.state.try_lock() {
            Ok(state) => {
                match &*state {
                    SessionState::Idle(owners) => Some(owners.external.as_ref().is_none_or(|e| {
                        e.worker.startup_report().is_some_and(|r| r.result.is_ok())
                    })),
                    // This getter never promotes manager admission to an
                    // actual external startup observation.
                    SessionState::Manager(_) => None,
                    _ => None,
                }
            }
            Err(_) => None,
        }
    }
    fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        let state = self
            .state
            .try_lock()
            .map_err(|_| failure("AdaptiveCheckerBusy", "actual owner is busy"))?;
        if let SessionState::Idle(owners) = &*state {
            return owners
                .external
                .as_ref()
                .and_then(|e| e.worker.startup_report())
                .map_or(Ok(None), |r| {
                    r.result
                        .map(Some)
                        .map_err(|e| SearchSessionFailure::debug("AdaptiveCheckerStartup", &e))
                });
        }
        Ok(None)
    }
    fn start_checker(
        &self,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        loop {
            let started = self.checker_started();
            if started == Some(true) {
                let state = self
                    .state
                    .try_lock()
                    .map_err(|_| failure("AdaptiveCheckerBusy", "actual startup owner busy"))?;
                if let SessionState::Idle(owners) = &*state {
                    self.bind_actual_external(owners)?;
                }
                return Ok(started);
            }
            // Preserve an actual startup failure instead of spending the rest
            // of the deadline waiting for an impossible ready response.
            self.checker_startup_uci()?;
            if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err(failure(
                    "AdaptiveCheckerStartup",
                    "actual startup not ready before original deadline",
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    fn finish_checker(
        &self,
        deadline: Instant,
    ) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        self.finish_process(deadline)
            .map(|receipt| receipt.external_shutdown())
    }
    fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        let state = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        Ok(match &*state {
            SessionState::Closed(receipt) => receipt.external_shutdown,
            _ => None,
        })
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(poisoned("adaptive process already closed"));
        }
        self.reset_requests
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(1)
            })
            .map(|_| ())
            .map_err(|_| poisoned("adaptive reset request sequence exhausted"))
    }
    fn accept_uci_output(
        &self,
        authority: SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        let mut pending = self
            .pending_output
            .lock()
            .map_err(|_| poisoned("adaptive output authority"))?;
        let registered = pending
            .as_ref()
            .ok_or_else(|| poisoned("no actual adaptive root output"))?;
        validate_output(registered, authority, bestmove, from_report)?;
        let mut observations = self
            .accepted_outputs
            .lock()
            .map_err(|_| poisoned("adaptive output receipt"))?;
        if observations.len() == 64 {
            observations.pop_front();
        }
        observations.push_back(AdaptiveUciOutputObservation {
            authority,
            bestmove: bestmove.to_owned(),
            from_report,
        });
        *pending = None;
        Ok(())
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.work
            .begin()
            .map_err(|e| SearchSessionFailure::debug("AdaptiveWorkJournal", &e))?;
        let mut observation = AttemptObservation::NoWork;
        let result = self.execute(position, context, progress, &mut observation);
        let recorded = self.work.finish(context, &result, observation);
        match (result, recorded) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(SearchSessionFailure::debug("AdaptiveWorkJournal", &error)),
            (Ok(result), Ok(())) => Ok(result),
        }
    }
}

impl<B> Drop for AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        let retained = std::mem::replace(state, SessionState::Moving);
        if matches!(retained, SessionState::Closed(_)) {
            drop(retained);
        } else {
            // At most one complete process owner per driver is retained. This
            // is failure quarantine, not successful shutdown or a cache. In
            // particular, dropping an Arc must not detach the native service
            // or block while dropping live CPU workers and free their Store.
            let _quarantined_owner: &'static mut SessionState<B> = Box::leak(Box::new(retained));
        }
    }
}

impl SearchSessionFailure {
    fn unknown_if_unclosed(mut self) -> Self {
        self.physical_completion = crate::search_driver::DriverPhysicalCompletion::Unknown;
        self
    }
}
fn drain_root<B>(
    manager: &mut SessionManager<B>,
    until: Instant,
) -> Result<(), AdaptiveManagerError>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    while manager.phase() != ManagerPhase::Finished {
        manager.tick()?;
        if Instant::now() >= until {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}
fn join_archive_lifecycle<B>(
    state: &mut SessionState<B>,
    owners: Box<IdleOwners<B>>,
    mut worker: ArchiveWorker<AdaptivePortModel>,
    until: Instant,
) -> Result<Box<IdleOwners<B>>, SearchSessionFailure>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    worker.request_shutdown();
    loop {
        match worker.try_join() {
            Ok(Some(receipt))
                if receipt.worker_joined && receipt.owners_returned && receipt.admission_closed =>
            {
                return Ok(owners);
            }
            Ok(None) if Instant::now() < until => std::thread::sleep(Duration::from_millis(1)),
            _ => {
                *state = SessionState::ArchiveJoinedPending { owners, worker };
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        }
    }
}
fn owned_declarations_match(
    actual: Option<rz_search::cpu_checker::OwnedCheckerDescriptor>,
    declared: Option<&rz_search::cpu_checker::OwnedCheckerDescriptor>,
) -> bool {
    match (actual, declared) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.config.profile == b.config.profile
                && a.config.tt_entries == b.config.tt_entries
                && a.config.max_depth == b.config.max_depth
                && a.config.quiescence_ply == b.config.quiescence_ply
                && a.search_identity == b.search_identity
                && a.search_conditions == b.search_conditions
                && a.capabilities == b.capabilities
        }
        _ => false,
    }
}
fn pals_work(counters: PalsCounters) -> DriverWork {
    DriverWork::Pals {
        rounds: counters.rounds,
        cpu_nodes: counters.cpu_nodes,
        completed_tasks: counters.completed_cpu_tasks,
        consumed_role_outputs: counters.consumed_role_outputs,
        retained_situations: counters.retained_situations,
    }
}
fn failure(code: &'static str, detail: &'static str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: crate::search_driver::DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}
fn poisoned(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::BackendFailure, Stage::Output, detail)
}
fn validate_output(
    pending: &PendingOutput,
    authority: SearchAuthority,
    bestmove: &str,
    from_report: bool,
) -> Result<(), ContractError> {
    if pending.authority != authority {
        return Err(poisoned("stale adaptive UCI output authority"));
    }
    let movement = if bestmove == "0000" {
        None
    } else {
        Some(
            bestmove
                .parse::<BoardMove>()
                .map_err(|_| poisoned("malformed adaptive bestmove"))?,
        )
    };
    if movement.is_some_and(|m| !pending.legal.contains(&m))
        || (movement.is_none() && !pending.legal.is_empty())
        || (from_report && movement != pending.report)
    {
        return Err(poisoned(
            "adaptive output does not match actual legal root/report",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "session_driver/lifecycle_tests.rs"]
mod lifecycle_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{GameGeneration, RootGeneration};
    fn authority(root: u64) -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(3),
            game: GameGeneration(2),
            root: RootGeneration(root),
            implementation: Digest([4; 32]),
        }
    }
    #[test]
    fn uci_output_checks_exact_root_and_separates_report_from_legal_fallback() {
        let legal = Position::startpos().legal_moves();
        let pending = PendingOutput {
            authority: authority(7),
            report: Some("e2e4".parse().unwrap()),
            legal,
        };
        assert!(validate_output(&pending, authority(7), "e2e4", true).is_ok());
        assert!(validate_output(&pending, authority(8), "e2e4", true).is_err());
        assert!(validate_output(&pending, authority(7), "d2d4", true).is_err());
        assert!(validate_output(&pending, authority(7), "d2d4", false).is_ok());
        assert!(validate_output(&pending, authority(7), "e2e5", false).is_err());
        assert!(validate_output(&pending, authority(7), "0000", false).is_err());
    }
    #[test]
    fn process_receipt_requires_join_and_physical_finish_independently() {
        let mut receipt = AdaptiveSessionProcessReceipt {
            epoch: ProcessEpoch(3),
            last_authority: None,
            native_owner_joined: false,
            native_physical_shutdown_confirmed: false,
            external_shutdown: None,
            store_archive_closed: true,
            latest_root: None,
        };
        assert!(!receipt.permits_process_close());
        receipt.native_owner_joined = true;
        assert!(!receipt.permits_process_close());
        receipt.native_physical_shutdown_confirmed = true;
        assert!(receipt.permits_process_close());
        receipt.store_archive_closed = false;
        assert!(!receipt.permits_process_close());
    }
}
