//! Startup-selected search sessions below UCI. A session returns only after its
//! owned work has completed or been drained; publication remains with UCI.
//! CPU/PALS counters are never translated into PUCT simulations or NN receipts.

use crate::{EngineIdentity, contracts::SessionScope};
use rz_contracts::{CancelToken, ContractError, Digest, ErrorCode, Stage};
use rz_position::{BoardMove, Position};
use rz_search::cpu_checker::{
    CheckerAttempt, CheckerCapabilities, CheckerIdentity, CheckerShutdown, CpuChecker,
    ExternalUciIdentity, OwnedCheckerDescriptor, OwnedCpuChecker,
};
use rz_search::driver::SearchControl;
use rz_search::pals::engine::{ModelValueIdentity, PostRepairRecheckPolicy};
use sha2::{Digest as _, Sha256};
use std::{
    fmt,
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
mod work;
use work::{AttemptObservation, ProcessWorkJournal};
pub use work::{
    CpuWorkTotals, PalsResolverIdentity, PalsSearchPolicyIdentity, PalsWorkTotals,
    ProcessSearchWorkReceipt,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
#[cfg_attr(feature = "search-work-receipts", serde(rename_all = "snake_case"))]
pub enum SearchKind {
    Cpu,
    Pals,
}

#[derive(Clone, Debug)]
pub struct SearchSessionContext {
    pub authority: rz_contracts::pals::SearchAuthority,
    pub control: SearchControl,
    pub cancellation: CancelToken,
    pub shutdown_deadline: Instant,
    pub(crate) current: Arc<Mutex<SessionScope>>,
}
impl SearchSessionContext {
    /// Called at a task/commit boundary. An old root cannot mutate a new root.
    pub fn accepts(&self) -> Result<bool, ContractError> {
        let current = self.current.lock().map_err(|_| {
            ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "UCI search authority poisoned",
            )
        })?;
        Ok(*current == SessionScope::Search(self.authority)
            && !self.cancellation.is_canceled()
            && !self
                .control
                .cancellation
                .load(std::sync::atomic::Ordering::Acquire)
            && Instant::now() < self.control.deadline)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DriverWork {
    Cpu {
        nodes: u64,
        completed_depth: u16,
    },
    Pals {
        rounds: u64,
        cpu_nodes: u64,
        completed_tasks: u64,
        consumed_role_outputs: u64,
        retained_situations: usize,
    },
}

/// Immutable startup declarations. These identify the selected helper, not
/// proof of loaded foreign weights, applied options or physical shutdown.
#[derive(Clone, Debug)]
pub struct PalsCheckerRegistration {
    pub identity: CheckerIdentity,
    pub conditions: String,
    pub capabilities: CheckerCapabilities,
    pub owned_descriptor: Option<OwnedCheckerDescriptor>,
    pub role_model: String,
    pub model_value: Option<ModelValueIdentity>,
    pub resolver_version: &'static str,
    pub resolver_semantics: &'static str,
}
impl PalsCheckerRegistration {
    fn capture<M: rz_search::pals::engine::RoleModel>(
        checker: &dyn CpuChecker,
        model: &M,
    ) -> Result<Self, SearchSessionFailure> {
        checker
            .validate_namespace()
            .map_err(|error| SearchSessionFailure::debug("PalsCheckerRegistration", &error))?;
        let external = matches!(checker.identity(), CheckerIdentity::ExternalUci(_));
        let (resolver_version, resolver_semantics) = if external {
            (
                rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
                rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
            )
        } else {
            (
                rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
                rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS,
            )
        };
        Ok(Self {
            identity: checker.identity().clone(),
            conditions: checker.conditions().to_owned(),
            capabilities: checker.capabilities(),
            owned_descriptor: checker.owned_descriptor(),
            role_model: model.identity().to_owned(),
            model_value: if external {
                model.value_identity().cloned()
            } else {
                None
            },
            resolver_version,
            resolver_semantics,
        })
    }
}

#[derive(Default)]
struct PalsCheckerLifecycle {
    active_cancel: Option<Arc<AtomicBool>>,
    // A bounded latest-search snapshot, replaced instead of appended per go.
    attempts: Vec<CheckerAttempt>,
    last_attempt: Option<CheckerAttempt>,
    shutdown: Option<CheckerShutdown>,
    startup_uci: Option<ExternalUciIdentity>,
    #[cfg(feature = "search-work-receipts")]
    startup_resources: Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
    #[cfg(feature = "search-work-receipts")]
    startup_resource_unavailable: Option<&'static str>,
}
struct PalsActiveLease<'a> {
    lifecycle: &'a Mutex<PalsCheckerLifecycle>,
    cancellation: Arc<AtomicBool>,
}
impl Drop for PalsActiveLease<'_> {
    fn drop(&mut self) {
        let Ok(mut state) = self.lifecycle.lock() else {
            return;
        };
        if state
            .active_cancel
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, &self.cancellation))
        {
            state.active_cancel = None;
        }
    }
}

/// Persistent P/C + startup-selected CPU checker session. The model is selected once at
/// startup; callers must explicitly identify a mock model as mock.
pub struct PalsSessionDriver<M: rz_search::pals::engine::RoleModel + 'static> {
    engine: Mutex<rz_search::pals::engine::PalsEngine<M>>,
    reset_requested: AtomicU64,
    reset_applied: AtomicU64,
    implementation: Digest,
    resolver_version: &'static str,
    identity: EngineIdentity,
    max_rounds: u64,
    max_cpu_nodes: u64,
    cpu_depth: u16,
    work: ProcessWorkJournal,
    registration: PalsCheckerRegistration,
    search_policy: Option<PalsSearchPolicyIdentity>,
    checker_closed: AtomicBool,
    checker_started: AtomicBool,
    checker_lifecycle: Mutex<PalsCheckerLifecycle>,
}
impl<M: rz_search::pals::engine::RoleModel + 'static> PalsSessionDriver<M> {
    pub fn new(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        cpu_config: rz_search::cpu::CpuConfig,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
    ) -> Result<Self, SearchSessionFailure> {
        if max_rounds == 0
            || max_rounds > 1_000_000
            || max_cpu_nodes == 0
            || cpu_depth == 0
            || cpu_depth > cpu_config.max_depth
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsConfiguration",
                detail: "finite PALS round/CPU limits are required".into(),
            });
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-uci-pals-session/1\0");
        hash.update(rz_search::pals::engine::PALS_SEARCH_VERSION.as_bytes());
        hash.update(b"\0value-resolver\0");
        hash.update(rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION.as_bytes());
        hash.update([0]);
        hash.update(rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS.as_bytes());
        hash.update(rz_search::cpu::CPU_SEARCH_VERSION.as_bytes());
        hash.update(rz_search::cpu::BOOTSTRAP_SCORE_VERSION.as_bytes());
        hash.update(model.identity().as_bytes());
        for value in [
            config.beam_width,
            config.line_plies,
            config.max_nodes,
            config.max_records,
        ] {
            hash.update((value as u64).to_le_bytes());
        }
        hash.update(config.max_role_calls.to_le_bytes());
        hash.update(config.cpu_nodes_per_task.to_le_bytes());
        hash.update(cpu_config.profile.identity().as_bytes());
        hash.update((cpu_config.tt_entries as u64).to_le_bytes());
        hash.update(cpu_config.max_depth.to_le_bytes());
        hash.update(cpu_config.quiescence_ply.to_le_bytes());
        hash.update(cpu_depth.to_le_bytes());
        hash.update(max_rounds.to_le_bytes());
        hash.update(max_cpu_nodes.to_le_bytes());
        let cpu = rz_search::cpu::CpuEngine::new(cpu_config)
            .map_err(|error| SearchSessionFailure::debug("PalsCpuConstructor", &error))?;
        let checker = OwnedCpuChecker::new(cpu)
            .map_err(|error| SearchSessionFailure::debug("PalsCheckerConstructor", &error))?;
        let registration = PalsCheckerRegistration::capture(&checker, &model)?;
        let engine = rz_search::pals::engine::PalsEngine::new_with_checker(config, model, checker)
            .map_err(|error| SearchSessionFailure::debug("PalsConstructor", &error))?;
        Ok(Self {
            engine: Mutex::new(engine),
            reset_requested: AtomicU64::new(0),
            reset_applied: AtomicU64::new(0),
            implementation: Digest(hash.finalize().into()),
            resolver_version: rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
            identity,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            work: ProcessWorkJournal::new(SearchKind::Pals),
            registration,
            search_policy: None,
            checker_closed: AtomicBool::new(false),
            checker_started: AtomicBool::new(true),
            checker_lifecycle: Mutex::new(PalsCheckerLifecycle::default()),
        })
    }

    /// The omitted/Disabled selection delegates to the historical own v1
    /// constructor. Only the explicit lane uses the full checker namespace.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_refinement_policy(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        cpu_config: rz_search::cpu::CpuConfig,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, SearchSessionFailure> {
        if policy == PostRepairRecheckPolicy::Disabled {
            return Self::new(
                config,
                model,
                cpu_config,
                max_rounds,
                max_cpu_nodes,
                cpu_depth,
                identity,
            );
        }
        if max_rounds == 0
            || max_rounds > 1_000_000
            || max_cpu_nodes == 0
            || cpu_depth == 0
            || cpu_depth > cpu_config.max_depth
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsConfiguration",
                detail: "finite PALS round/CPU limits are required".into(),
            });
        }
        let cpu = rz_search::cpu::CpuEngine::new(cpu_config)
            .map_err(|error| SearchSessionFailure::debug("PalsCpuConstructor", &error))?;
        let checker = OwnedCpuChecker::new(cpu)
            .map_err(|error| SearchSessionFailure::debug("PalsCheckerConstructor", &error))?;
        Self::new_with_checker_and_refinement_policy(
            config,
            model,
            checker,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            policy,
        )
    }

    /// Explicit checker-selected path with a separate v2 source namespace.
    /// The legacy own constructor above keeps its v1 hash byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_checker<C: CpuChecker + 'static>(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: C,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
    ) -> Result<Self, SearchSessionFailure> {
        Self::new_with_boxed_checker(
            config,
            model,
            Box::new(checker),
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_checker_and_refinement_policy<C: CpuChecker + 'static>(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: C,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, SearchSessionFailure> {
        Self::new_with_boxed_checker_and_refinement_policy(
            config,
            model,
            Box::new(checker),
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            policy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_boxed_checker(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: Box<dyn CpuChecker>,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
    ) -> Result<Self, SearchSessionFailure> {
        Self::new_with_boxed_checker_and_refinement_policy(
            config,
            model,
            checker,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            PostRepairRecheckPolicy::Disabled,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_boxed_checker_and_refinement_policy(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: Box<dyn CpuChecker>,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, SearchSessionFailure> {
        let registration = PalsCheckerRegistration::capture(checker.as_ref(), &model)?;
        if policy != PostRepairRecheckPolicy::Disabled
            && matches!(registration.identity, CheckerIdentity::ExternalUci(_))
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsRefinementUnsupported",
                detail: "post-Repair recheck v1 requires own CPU/Rules evidence; no foreign helper was started".into(),
            });
        }
        if max_rounds == 0
            || max_rounds > 1_000_000
            || max_cpu_nodes == 0
            || cpu_depth == 0
            || cpu_depth > registration.capabilities.max_depth
            || registration.role_model.is_empty()
            || registration.role_model.len() > 1024
            || registration.role_model.chars().any(char::is_control)
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsConfiguration",
                detail: "finite PALS limits and bounded model identity are required".into(),
            });
        }
        let legacy_implementation =
            checker_implementation(&registration, &config, max_rounds, max_cpu_nodes, cpu_depth);
        let engine =
            rz_search::pals::engine::PalsEngine::new_with_boxed_checker_and_refinement_policy(
                config, model, checker, policy,
            )
            .map_err(|error| SearchSessionFailure::debug("PalsConstructor", &error))?;
        let search_policy = selected_search_policy(&engine, policy)?;
        let implementation = search_policy
            .as_ref()
            .map_or(legacy_implementation, |selected| {
                refinement_implementation(legacy_implementation, selected)
            });
        let resolver = PalsResolverIdentity::from_semantics(
            registration.resolver_version,
            registration.resolver_semantics,
        );
        let work = match &search_policy {
            Some(selected) => {
                ProcessWorkJournal::new_pals_with_search_policy(resolver, selected.clone())
                    .map_err(|error| {
                        SearchSessionFailure::debug("PalsSearchPolicyRegistration", &error)
                    })?
            }
            None => ProcessWorkJournal::new_pals(resolver),
        };
        Ok(Self {
            engine: Mutex::new(engine),
            reset_requested: AtomicU64::new(0),
            reset_applied: AtomicU64::new(0),
            implementation,
            resolver_version: registration.resolver_version,
            identity,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            work,
            checker_started: AtomicBool::new(matches!(
                registration.identity,
                CheckerIdentity::Owned(_)
            )),
            registration,
            search_policy,
            checker_closed: AtomicBool::new(false),
            checker_lifecycle: Mutex::new(PalsCheckerLifecycle::default()),
        })
    }

    pub fn checker_registration(&self) -> &PalsCheckerRegistration {
        &self.registration
    }
    /// Available before any go, including when a later result is early/unknown.
    /// This does not attest that a recheck branch ran or refuted a repaired line.
    pub fn search_policy_registration(&self) -> Option<&PalsSearchPolicyIdentity> {
        self.search_policy.as_ref()
    }
    /// Startup success is independent of registration and UCI option claims.
    pub fn checker_started(&self) -> bool {
        self.checker_started.load(Ordering::Acquire)
    }

    /// Starts an unstarted foreign helper on the caller's clock. The shared
    /// signal also permits finish_checker to cancel an active handshake.
    pub fn start_checker(
        &self,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<(), SearchSessionFailure> {
        if self.checker_closed.load(Ordering::Acquire) {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerClosed",
                detail: "checker startup is permanently closed".into(),
            });
        }
        let now = Instant::now();
        if cancel.load(Ordering::Acquire)
            || deadline <= now
            || deadline.duration_since(now) > Duration::from_secs(180)
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerStartupDeadline",
                detail: "checker startup requires an uncanceled finite deadline".into(),
            });
        }
        let mut engine = loop {
            if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsCheckerStartupLockDeadline",
                    detail: "no checker startup admitted before cancellation/deadline".into(),
                });
            }
            match self.engine.try_lock() {
                Ok(engine) => break engine,
                Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(checker_owner_unknown("PalsCheckerStartupOwner"));
                }
            }
        };
        let _active = self.activate_checker(Arc::clone(&cancel))?;
        if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_))
            && self.checker_started()
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerAlreadyStarted",
                detail: "foreign helper startup is not repeated implicitly".into(),
            });
        }
        let startup = engine.start_checker(deadline, &cancel);
        self.capture_checker(&engine)?;
        startup.map_err(|error| pals_failure("PalsCheckerStartup", &error, &engine))?;
        #[cfg(feature = "search-work-receipts")]
        let startup_resources =
            if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_)) {
                let process = engine
                    .checker_last_attempt()
                    .and_then(|attempt| attempt.external.as_ref())
                    .and_then(|external| external.process.process_identity);
                crate::pals_attestation::checker::resource::observe(process, deadline, &cancel)
                    .map_err(|error| {
                        SearchSessionFailure::debug("PalsCheckerStartupResource", &error)
                    })?
            } else {
                crate::pals_attestation::checker::resource::ResourceObservation {
                    snapshot: None,
                    unavailable: None,
                }
            };
        // An adapter returning after control expiry does not gain ready status.
        if cancel.load(Ordering::Acquire)
            || Instant::now() >= deadline
            || self.checker_closed.load(Ordering::Acquire)
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerStartupLate",
                detail: "checker startup returned after its admission controls".into(),
            });
        }
        let mut lifecycle = self
            .checker_lifecycle
            .lock()
            .map_err(|_| checker_owner_unknown("PalsCheckerStartupEvidence"))?;
        lifecycle.startup_uci = engine.checker_startup_uci();
        #[cfg(feature = "search-work-receipts")]
        if lifecycle.startup_resources.is_none() {
            lifecycle.startup_resources = startup_resources.snapshot;
            lifecycle.startup_resource_unavailable = startup_resources.unavailable;
        }
        self.checker_started.store(true, Ordering::Release);
        Ok(())
    }
    pub fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.startup_uci.clone())
            .map_err(|_| checker_owner_unknown("PalsCheckerStartupEvidence"))
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn checker_startup_resources(
        &self,
    ) -> Result<
        Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
        SearchSessionFailure,
    > {
        self.checker_lifecycle
            .lock()
            .map(|state| state.startup_resources.clone())
            .map_err(|_| checker_owner_unknown("PalsCheckerStartupResource"))
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn checker_startup_resource_unavailable(
        &self,
    ) -> Result<Option<&'static str>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.startup_resource_unavailable)
            .map_err(|_| checker_owner_unknown("PalsCheckerStartupResource"))
    }

    /// Latest-search physical attempts only. The latest lifecycle snapshot is
    /// available separately; callers must not sum it again as a new request.
    pub fn checker_attempts(&self) -> Result<Vec<CheckerAttempt>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.attempts.clone())
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))
    }
    pub fn checker_last_attempt(&self) -> Result<Option<CheckerAttempt>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.last_attempt.clone())
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))
    }
    pub fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.shutdown)
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))
    }

    fn activate_checker(
        &self,
        cancellation: Arc<AtomicBool>,
    ) -> Result<PalsActiveLease<'_>, SearchSessionFailure> {
        let mut state = self
            .checker_lifecycle
            .lock()
            .map_err(|_| checker_owner_unknown("PalsCheckerOwner"))?;
        if self.checker_closed.load(Ordering::Acquire) {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerClosed",
                detail: "startup-selected helper is closing or closed; no work admitted".into(),
            });
        }
        state.active_cancel = Some(Arc::clone(&cancellation));
        Ok(PalsActiveLease {
            lifecycle: &self.checker_lifecycle,
            cancellation,
        })
    }

    fn capture_checker(
        &self,
        engine: &rz_search::pals::engine::PalsEngine<M>,
    ) -> Result<(), SearchSessionFailure> {
        let mut state = self
            .checker_lifecycle
            .lock()
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))?;
        state.attempts = engine.checker_attempts().to_vec();
        state.last_attempt = engine.checker_last_attempt().cloned();
        Ok(())
    }

    /// Permanently closes admission, cancels active work and drains the checker
    /// under a finite deadline. This does not shut down the role-model owner.
    /// A successful own result claims no subprocess exit or pipe drain.
    pub fn finish_checker(
        &self,
        deadline: Instant,
    ) -> Result<CheckerShutdown, SearchSessionFailure> {
        self.checker_closed.store(true, Ordering::Release);
        {
            let state = self
                .checker_lifecycle
                .lock()
                .map_err(|_| checker_owner_unknown("PalsCheckerShutdown"))?;
            if let Some(cancel) = &state.active_cancel {
                cancel.store(true, Ordering::Release);
            }
            if let Some(shutdown) = state.shutdown.filter(|value| {
                value.cleanup_complete && !value.quarantined && !value.ownership_lost
            }) {
                return Ok(shutdown);
            }
        }
        let now = Instant::now();
        if deadline <= now || deadline.duration_since(now) > Duration::from_secs(180) {
            return Err(checker_owner_unknown("PalsCheckerShutdownDeadline"));
        }
        let mut engine = loop {
            if Instant::now() >= deadline {
                return Err(checker_owner_unknown("PalsCheckerShutdownLockDeadline"));
            }
            match self.engine.try_lock() {
                Ok(engine) => break engine,
                Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(checker_owner_unknown("PalsCheckerShutdownOwner"));
                }
            }
        };
        let result = engine.shutdown_checker(deadline);
        self.capture_checker(&engine)?;
        let mut state = self
            .checker_lifecycle
            .lock()
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))?;
        // Only actual returned state or actual adapter evidence is retained.
        // Absence of a startup attempt remains None, never fabricated shutdown.
        state.shutdown = result.as_ref().ok().copied().or_else(|| {
            state
                .last_attempt
                .as_ref()
                .and_then(|attempt| attempt.external.as_ref())
                .map(|attempt| attempt.process)
        });
        match result {
            Ok(shutdown)
                if shutdown.cleanup_complete
                    && !shutdown.quarantined
                    && !shutdown.ownership_lost =>
            {
                Ok(shutdown)
            }
            Ok(_) => Err(checker_owner_unknown("PalsCheckerShutdownIncomplete")),
            Err(error) => {
                let mut failure = SearchSessionFailure::debug("PalsCheckerShutdown", &error);
                failure.physical_completion = DriverPhysicalCompletion::Unknown;
                Err(failure)
            }
        }
    }
}
impl<M: rz_search::pals::engine::RoleModel + 'static> SearchSessionDriver for PalsSessionDriver<M> {
    fn kind(&self) -> SearchKind {
        SearchKind::Pals
    }
    fn implementation(&self) -> Digest {
        self.implementation
    }
    fn identity(&self) -> EngineIdentity {
        self.identity.clone()
    }
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        self.work.snapshot().map(Some)
    }
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        Some(self.registration.clone())
    }
    fn checker_started(&self) -> Option<bool> {
        Some(PalsSessionDriver::checker_started(self))
    }
    fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        PalsSessionDriver::checker_startup_uci(self)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resources(
        &self,
    ) -> Result<
        Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
        SearchSessionFailure,
    > {
        PalsSessionDriver::checker_startup_resources(self)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resource_unavailable(
        &self,
    ) -> Result<Option<&'static str>, SearchSessionFailure> {
        PalsSessionDriver::checker_startup_resource_unavailable(self)
    }
    fn start_checker(
        &self,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        PalsSessionDriver::start_checker(self, deadline, cancel).map(|()| Some(true))
    }
    fn checker_attempts(&self) -> Result<Option<Vec<CheckerAttempt>>, SearchSessionFailure> {
        PalsSessionDriver::checker_attempts(self).map(Some)
    }
    fn checker_last_attempt(&self) -> Result<Option<CheckerAttempt>, SearchSessionFailure> {
        PalsSessionDriver::checker_last_attempt(self)
    }
    fn finish_checker(
        &self,
        deadline: Instant,
    ) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        PalsSessionDriver::finish_checker(self, deadline).map(Some)
    }
    fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        PalsSessionDriver::checker_shutdown(self)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        if self.checker_closed.load(Ordering::Acquire) {
            return Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "PALS checker admission is permanently closed",
            ));
        }
        let requested = self
            .reset_requested
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                ContractError::new(
                    ErrorCode::BackendFailure,
                    Stage::Admission,
                    "PALS reset request generation exhausted",
                )
            })?
            + 1;
        // A foreign reset needs real deadline/cancellation and may fail. Do not
        // send ucinewgame, clear its evidence or acknowledge success here.
        if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_)) {
            return Ok(());
        }
        match self.engine.try_lock() {
            Ok(mut engine) => {
                engine.new_game();
                self.reset_applied.fetch_max(requested, Ordering::AcqRel);
                Ok(())
            }
            Err(TryLockError::WouldBlock) => Ok(()), // Closing old work owns the lock; reset before the next admitted search.
            Err(TryLockError::Poisoned(_)) => Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "PALS session owner poisoned",
            )),
        }
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.work
            .begin()
            .map_err(|error| SearchSessionFailure::debug("PalsWorkReceipt", &error))?;
        let mut observation = AttemptObservation::NoWork;
        let result = (|| {
            if self.checker_closed.load(Ordering::Acquire) {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsCheckerClosed",
                    detail: "startup-selected helper is closing or closed; no work admitted".into(),
                });
            }
            if !self.checker_started() {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsCheckerNotStarted",
                    detail: "foreign helper must complete explicit startup before go".into(),
                });
            }
            if context.control.max_simulations > self.max_cpu_nodes {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsNodeLimit",
                    detail: "UCI node request exceeds registered CPU work bound".into(),
                });
            }
            // One owner retains game evidence. Waiting for a canceled previous root
            // is bounded by the new root deadline, not an unbounded Mutex::lock.
            let mut engine = loop {
                if !context.accepts().map_err(|error| SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsAuthority",
                    detail: error.to_string(),
                })? {
                    return Ok(SearchSessionReport {
                        best_move: None,
                        work: DriverWork::Pals {
                            rounds: 0,
                            cpu_nodes: 0,
                            completed_tasks: 0,
                            consumed_role_outputs: 0,
                            retained_situations: 0,
                        },
                    });
                }
                match self.engine.try_lock() {
                    Ok(engine) => break engine,
                    Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                    Err(TryLockError::Poisoned(_)) => {
                        return Err(SearchSessionFailure {
                            physical_completion: DriverPhysicalCompletion::Confirmed,
                            code: "PalsOwner",
                            detail: "persistent PALS owner poisoned".into(),
                        });
                    }
                }
            };
            if !context.accepts().map_err(|error| SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsAuthority",
                detail: error.to_string(),
            })? {
                return Ok(SearchSessionReport {
                    best_move: None,
                    work: DriverWork::Pals {
                        rounds: 0,
                        cpu_nodes: 0,
                        completed_tasks: 0,
                        consumed_role_outputs: 0,
                        retained_situations: engine.retained_situations(),
                    },
                });
            }
            let selected_policy = if self.search_policy.is_some() {
                PostRepairRecheckPolicy::SameRepairedLineOnceV1
            } else {
                PostRepairRecheckPolicy::Disabled
            };
            if selected_search_policy(&engine, selected_policy)? != self.search_policy {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsSearchPolicyIdentity",
                    detail: "effective PALS lane differs from its immutable startup registration"
                        .into(),
                });
            }
            let _active = self.activate_checker(Arc::clone(&context.control.cancellation))?;
            let reset_requested = self.reset_requested.load(Ordering::Acquire);
            if reset_requested != self.reset_applied.load(Ordering::Acquire) {
                let reset = engine.try_new_game(
                    context
                        .control
                        .soft_deadline
                        .min(context.control.admission_deadline),
                    &context.control.cancellation,
                );
                self.capture_checker(&engine)?;
                reset.map_err(|error| pals_failure("PalsReset", &error, &engine))?;
                // A failed reset stays unapplied. A newer concurrent request
                // remains pending even when this older reset succeeds.
                self.reset_applied
                    .fetch_max(reset_requested, Ordering::AcqRel);
            }
            observation = AttemptObservation::PalsStarted;
            let outcome = engine.search_with_progress(
                position,
                rz_search::pals::engine::PalsLimits {
                    // End normal PALS work at the soft target. The existing
                    // hard/output boundaries retain time for report validation,
                    // physical completion, owner notification and UCI output.
                    deadline: context
                        .control
                        .soft_deadline
                        .min(context.control.admission_deadline),
                    max_rounds: self.max_rounds,
                    max_cpu_nodes: context.control.max_simulations,
                    cpu_depth: self.cpu_depth,
                },
                &context.control.cancellation,
                &mut *progress,
            );
            if let Some(counters) = engine.last_search_counters() {
                observation = AttemptObservation::Pals {
                    counters,
                    root_coverage_observed: counters.root_scope_observation_complete,
                };
            }
            self.capture_checker(&engine)?;
            let report = outcome.map_err(|error| pals_failure("PalsSearch", &error, &engine))?;
            if report.resolver_version != self.resolver_version {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsResolverIdentity",
                    detail: "returned PALS resolver version differs from its startup registration"
                        .into(),
                });
            }
            if context.accepts().map_err(|error| SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsAuthority",
                detail: error.to_string(),
            })? {
                if let Some(movement) = report.best_move {
                    progress(movement);
                }
            }
            Ok(SearchSessionReport {
                best_move: report.best_move,
                work: DriverWork::Pals {
                    rounds: report.counters.rounds,
                    cpu_nodes: report.counters.cpu_nodes,
                    completed_tasks: report.counters.completed_cpu_tasks,
                    consumed_role_outputs: report.counters.consumed_role_outputs,
                    retained_situations: report.counters.retained_situations,
                },
            })
        })();
        retain_work_failure(&self.work, context, result, observation)
    }
}
fn checker_owner_unknown(code: &'static str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Unknown,
        code,
        detail: "checker owner completion could not be confirmed within its finite boundary".into(),
    }
}

fn pals_failure<M: rz_search::pals::engine::RoleModel>(
    code: &'static str,
    error: &rz_search::pals::engine::PalsError,
    engine: &rz_search::pals::engine::PalsEngine<M>,
) -> SearchSessionFailure {
    let role_unknown = matches!(
        error,
        rz_search::pals::engine::PalsError::Role(
            rz_search::pals::engine::RoleError::PhysicalCompletionUnknown
        )
    );
    if role_unknown {
        return SearchSessionFailure::physical_completion_unknown();
    }
    // An idle persistent process need not have exited or drained. Only actual
    // quarantine/ownership loss establishes unconfirmed physical completion.
    let checker_unknown = engine
        .checker_last_attempt()
        .and_then(|attempt| attempt.external.as_ref())
        .is_some_and(|attempt| attempt.process.quarantined || attempt.process.ownership_lost);
    let mut failure = SearchSessionFailure::debug(code, error);
    if checker_unknown {
        failure.physical_completion = DriverPhysicalCompletion::Unknown;
    }
    failure
}

/// Capture only actual immutable engine getters. A result's resolver/score does
/// not identify this lane and cannot establish that its recheck branch executed.
fn selected_search_policy<M: rz_search::pals::engine::RoleModel>(
    engine: &rz_search::pals::engine::PalsEngine<M>,
    requested: PostRepairRecheckPolicy,
) -> Result<Option<PalsSearchPolicyIdentity>, SearchSessionFailure> {
    let mismatch = || SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code: "PalsSearchPolicyIdentity",
        detail: "actual PALS search/refinement getters differ from selected startup policy".into(),
    };
    if engine.post_repair_recheck_policy() != requested {
        return Err(mismatch());
    }
    if requested == PostRepairRecheckPolicy::Disabled {
        if engine.search_identity() != rz_search::pals::engine::PALS_SEARCH_VERSION
            || engine.refinement_conditions().is_some()
        {
            return Err(mismatch());
        }
        return Ok(None);
    }
    let conditions = engine.refinement_conditions().ok_or_else(mismatch)?;
    let identity = PalsSearchPolicyIdentity {
        version: PalsSearchPolicyIdentity::VERSION.into(),
        policy: PalsSearchPolicyIdentity::POLICY.into(),
        search_identity: engine.search_identity().into(),
        conditions_sha256: Sha256::digest(conditions.as_bytes()).into(),
    };
    identity
        .validate()
        .map_err(|error| SearchSessionFailure::debug("PalsSearchPolicyIdentity", &error))?;
    Ok(Some(identity))
}

fn refinement_implementation(base: Digest, policy: &PalsSearchPolicyIdentity) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-uci-pals-post-repair-recheck-session/1\0");
    hash.update(base.0);
    for value in [&policy.version, &policy.policy, &policy.search_identity] {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    hash.update(policy.conditions_sha256);
    Digest(hash.finalize().into())
}

/// Canonical length-prefixed declaration hash. It binds every admitted checker
/// field without depending on Debug formatting, platform usize width or paths.
/// It is a source namespace, not independent evidence of foreign execution.
fn checker_implementation(
    registration: &PalsCheckerRegistration,
    config: &rz_search::pals::engine::PalsConfig,
    max_rounds: u64,
    max_cpu_nodes: u64,
    cpu_depth: u16,
) -> Digest {
    fn text(hash: &mut Sha256, value: &str) {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    fn optional_text(hash: &mut Sha256, value: Option<&str>) {
        hash.update([u8::from(value.is_some())]);
        if let Some(value) = value {
            text(hash, value);
        }
    }
    fn sizes(hash: &mut Sha256, depth: u16, prefix: usize, roots: usize) {
        hash.update(depth.to_le_bytes());
        hash.update((prefix as u64).to_le_bytes());
        hash.update((roots as u64).to_le_bytes());
    }
    let mut hash = Sha256::new();
    hash.update(b"rz-uci-pals-checker-session/2\0");
    text(&mut hash, rz_search::pals::engine::PALS_SEARCH_VERSION);
    text(&mut hash, registration.resolver_version);
    text(&mut hash, registration.resolver_semantics);
    text(&mut hash, &registration.role_model);
    text(&mut hash, &registration.conditions);
    match &registration.identity {
        CheckerIdentity::Owned(value) => {
            hash.update([0]);
            text(&mut hash, &value.semantics);
            optional_text(&mut hash, value.weights_sha256.as_deref());
            match &value.training {
                rz_search::cpu_value::CpuTrainingState::Bootstrap => hash.update([0]),
                rz_search::cpu_value::CpuTrainingState::Untrained => hash.update([1]),
                rz_search::cpu_value::CpuTrainingState::Learned {
                    run_id,
                    steps,
                    dataset_sha256,
                } => {
                    hash.update([2]);
                    text(&mut hash, run_id);
                    hash.update(steps.to_le_bytes());
                    text(&mut hash, dataset_sha256);
                }
            }
        }
        CheckerIdentity::ExternalUci(value) => {
            hash.update([1]);
            for value in [
                &value.adapter_semantics,
                &value.binary_sha256,
                &value.launch_arguments_sha256,
                &value.declared_name,
                &value.declared_version,
                &value.declared_source,
                &value.declared_license,
            ] {
                text(&mut hash, value);
            }
            hash.update((value.options.len() as u64).to_le_bytes());
            for (name, value) in &value.options {
                text(&mut hash, name);
                text(&mut hash, value);
            }
            hash.update((value.assets.len() as u64).to_le_bytes());
            for asset in &value.assets {
                text(&mut hash, &asset.purpose);
                text(&mut hash, &asset.sha256);
            }
            optional_text(&mut hash, value.model_metadata.weights_sha256.as_deref());
            match value.model_metadata.training {
                rz_search::cpu_checker::ExternalTrainingKnowledge::Unknown => hash.update([0]),
            }
            optional_text(&mut hash, value.model_metadata.declared_rights.as_deref());
            optional_text(&mut hash, value.model_metadata.precision.as_deref());
        }
    }
    let caps = registration.capabilities;
    sizes(
        &mut hash,
        caps.max_depth,
        caps.max_prefix_plies,
        caps.max_root_moves,
    );
    hash.update([
        u8::from(caps.root_moves),
        u8::from(caps.divergence),
        u8::from(caps.resume),
        match caps.selective_search {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
    ]);
    hash.update([u8::from(registration.owned_descriptor.is_some())]);
    if let Some(owned) = &registration.owned_descriptor {
        text(&mut hash, owned.search_identity);
        text(&mut hash, &owned.search_conditions);
        text(&mut hash, owned.config.profile.identity());
        hash.update((owned.config.tt_entries as u64).to_le_bytes());
        hash.update(owned.config.max_depth.to_le_bytes());
        hash.update(owned.config.quiescence_ply.to_le_bytes());
        let caps = owned.capabilities;
        sizes(
            &mut hash,
            caps.max_depth,
            caps.max_prefix_plies,
            caps.max_root_moves,
        );
        hash.update([
            u8::from(caps.root_moves),
            u8::from(caps.divergence),
            u8::from(caps.completed_iteration_resume),
            u8::from(caps.selective_reductions),
        ]);
    }
    hash.update([u8::from(registration.model_value.is_some())]);
    if let Some(value) = &registration.model_value {
        for text_value in [
            &value.semantics,
            &value.model,
            &value.encoding,
            &value.precision,
        ] {
            text(&mut hash, text_value);
        }
        hash.update(value.model_epoch);
    }
    for value in [
        config.beam_width,
        config.line_plies,
        config.max_nodes,
        config.max_records,
    ] {
        hash.update((value as u64).to_le_bytes());
    }
    hash.update(config.max_role_calls.to_le_bytes());
    hash.update(config.cpu_nodes_per_task.to_le_bytes());
    hash.update(max_rounds.to_le_bytes());
    hash.update(max_cpu_nodes.to_le_bytes());
    hash.update(cpu_depth.to_le_bytes());
    Digest(hash.finalize().into())
}
#[derive(Clone, Debug)]
pub struct SearchSessionReport {
    pub best_move: Option<BoardMove>,
    pub work: DriverWork,
}
#[derive(Clone, Debug)]
pub struct SearchSessionFailure {
    pub physical_completion: DriverPhysicalCompletion,
    pub code: &'static str,
    pub detail: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DriverPhysicalCompletion {
    Confirmed,
    Unknown,
}
impl SearchSessionFailure {
    pub fn physical_completion_unknown() -> Self {
        Self {
            physical_completion: DriverPhysicalCompletion::Unknown,
            code: "PhysicalCompletionUnknown",
            detail: "search driver retains quarantined/unconfirmed physical work".into(),
        }
    }
    /// A role backend may supply a dynamic cause. Preserve a finite receipt
    /// without formatting or retaining an unbounded external backend string.
    pub(crate) fn debug(code: &'static str, value: &dyn fmt::Debug) -> Self {
        struct Detail {
            text: String,
            truncated: bool,
        }
        impl fmt::Write for Detail {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                let available = 512usize.saturating_sub(self.text.len());
                if value.len() <= available {
                    self.text.push_str(value);
                    return Ok(());
                }
                let mut end = available;
                while !value.is_char_boundary(end) {
                    end -= 1;
                }
                self.text.push_str(&value[..end]);
                self.truncated = true;
                Err(fmt::Error)
            }
        }
        let mut detail = Detail {
            text: String::with_capacity(512),
            truncated: false,
        };
        let _ = fmt::write(&mut detail, format_args!("{value:?}"));
        if detail.truncated {
            detail.text.push_str(" [cause truncated]");
        }
        Self {
            physical_completion: DriverPhysicalCompletion::Confirmed,
            code,
            detail: detail.text,
        }
    }
}
impl fmt::Display for SearchSessionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for SearchSessionFailure {}

/// Implementations own their queues, evaluator and physical leases. Returning
/// from `run` acknowledges bounded completion/drain, not just logical cancel.
/// Progress is advisory: UCI checks its exact root legal set and authority.
pub trait SearchSessionDriver: Send + Sync + 'static {
    fn kind(&self) -> SearchKind;
    fn implementation(&self) -> Digest;
    fn identity(&self) -> EngineIdentity;
    /// None is unavailable observation, never an invented all-zero receipt.
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        Ok(None)
    }
    /// None means this driver has no PALS checker registration, not a foreign
    /// helper with an empty or implicitly selected identity.
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        None
    }
    fn checker_started(&self) -> Option<bool> {
        None
    }
    fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        Ok(None)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resources(
        &self,
    ) -> Result<
        Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
        SearchSessionFailure,
    > {
        Ok(None)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resource_unavailable(
        &self,
    ) -> Result<Option<&'static str>, SearchSessionFailure> {
        Ok(None)
    }
    fn start_checker(
        &self,
        _deadline: Instant,
        _cancel: Arc<AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        Ok(None)
    }
    fn checker_attempts(&self) -> Result<Option<Vec<CheckerAttempt>>, SearchSessionFailure> {
        Ok(None)
    }
    fn checker_last_attempt(&self) -> Result<Option<CheckerAttempt>, SearchSessionFailure> {
        Ok(None)
    }
    /// Ends only the CPU-checker owner. The role-model owner has an independent
    /// fence and receipt. None means no checker shutdown interface is present.
    fn finish_checker(
        &self,
        _deadline: Instant,
    ) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        Ok(None)
    }
    fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        Ok(None)
    }
    fn accept_uci_output(
        &self,
        _authority: rz_contracts::pals::SearchAuthority,
        _bestmove: &str,
        _from_report: bool,
    ) -> Result<(), ContractError> {
        Ok(())
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        Ok(())
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure>;
}

/// Own Rust CPU_R. No ModelAdapter, ORT session, CUDA provider or EvalOutput is
/// constructed. One owner retains bounded scratch/TT and exact Rules history.
pub struct CpuSessionDriver {
    config: rz_search::cpu::CpuConfig,
    engine: Mutex<rz_search::cpu::CpuEngine>,
    reset_pending: AtomicBool,
    max_nodes: u64,
    work: ProcessWorkJournal,
}
impl CpuSessionDriver {
    pub fn new(
        config: rz_search::cpu::CpuConfig,
        max_nodes: u64,
    ) -> Result<Self, SearchSessionFailure> {
        if max_nodes == 0 {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "CpuConfiguration",
                detail: "CPU node limit must be positive".into(),
            });
        }
        let engine = rz_search::cpu::CpuEngine::new(config.clone())
            .map_err(|error| SearchSessionFailure::debug("CpuConfiguration", &error))?;
        Ok(Self {
            config,
            engine: Mutex::new(engine),
            reset_pending: AtomicBool::new(false),
            max_nodes,
            work: ProcessWorkJournal::new(SearchKind::Cpu),
        })
    }
}
impl SearchSessionDriver for CpuSessionDriver {
    fn kind(&self) -> SearchKind {
        SearchKind::Cpu
    }
    fn implementation(&self) -> Digest {
        // Semantic search/profile configuration identity, not a weights hash.
        // Source and executable SHA-256 are registered separately by experiments.
        let mut hash = Sha256::new();
        hash.update(b"rz-uci-cpu-session/1\0");
        hash.update(rz_search::cpu::CPU_SEARCH_VERSION.as_bytes());
        hash.update(rz_search::cpu::BOOTSTRAP_SCORE_VERSION.as_bytes());
        hash.update(self.config.profile.identity().as_bytes());
        hash.update((self.config.tt_entries as u64).to_le_bytes());
        hash.update(self.config.max_depth.to_le_bytes());
        hash.update(self.config.quiescence_ply.to_le_bytes());
        hash.update(self.max_nodes.to_le_bytes());
        Digest(hash.finalize().into())
    }
    fn identity(&self) -> EngineIdentity {
        EngineIdentity {
            name: "RoveZero own Rust CPU_R".into(),
            author: "RoveZero contributors".into(),
        }
    }
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        self.work.snapshot().map(Some)
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.work
            .accept_cpu_output(authority, bestmove, from_report)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.reset_pending.store(true, Ordering::Release);
        match self.engine.try_lock() {
            Ok(mut engine) => {
                engine.clear();
                self.reset_pending.store(false, Ordering::Release);
                Ok(())
            }
            Err(TryLockError::WouldBlock) => Ok(()),
            Err(TryLockError::Poisoned(_)) => Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "CPU session owner poisoned",
            )),
        }
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.work
            .begin()
            .map_err(|error| SearchSessionFailure::debug("CpuWorkReceipt", &error))?;
        let mut observation = AttemptObservation::NoWork;
        let result = (|| {
            if context.control.max_simulations > self.max_nodes {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "CpuNodeLimit",
                    detail: "UCI node request exceeds the registered CPU limit".into(),
                });
            }
            let mut engine = loop {
                if !context.accepts().map_err(|error| SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "CpuAuthority",
                    detail: error.to_string(),
                })? {
                    return Ok(SearchSessionReport {
                        best_move: None,
                        work: DriverWork::Cpu {
                            nodes: 0,
                            completed_depth: 0,
                        },
                    });
                }
                match self.engine.try_lock() {
                    Ok(engine) => break engine,
                    Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                    Err(TryLockError::Poisoned(_)) => {
                        return Err(SearchSessionFailure {
                            physical_completion: DriverPhysicalCompletion::Confirmed,
                            code: "CpuOwner",
                            detail: "persistent CPU owner poisoned".into(),
                        });
                    }
                }
            };
            if !context.accepts().map_err(|error| SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "CpuAuthority",
                detail: error.to_string(),
            })? {
                return Ok(SearchSessionReport {
                    best_move: None,
                    work: DriverWork::Cpu {
                        nodes: 0,
                        completed_depth: 0,
                    },
                });
            }
            if self.reset_pending.swap(false, Ordering::AcqRel) {
                engine.clear();
            }
            let limits = rz_search::cpu::CpuLimits {
                max_depth: self.config.max_depth,
                max_nodes: context.control.max_simulations,
                deadline: Some(context.control.admission_deadline),
            };
            observation = AttemptObservation::CpuStarted;
            let attempted = engine.analyze_with_progress(
                position,
                limits,
                &context.control.cancellation,
                |iteration| {
                    if let Some(movement) = iteration.best_move {
                        progress(movement);
                    }
                },
            );
            if let Some(work) = engine.last_attempt_work() {
                observation = AttemptObservation::CpuFailed(work);
            }
            let report =
                attempted.map_err(|error| SearchSessionFailure::debug("CpuSearch", &error))?;
            observation = AttemptObservation::cpu(&report, limits.max_depth);
            if context.accepts().map_err(|error| SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "CpuAuthority",
                detail: error.to_string(),
            })? {
                if let Some(movement) = report.best_move {
                    progress(movement);
                }
            }
            Ok(SearchSessionReport {
                best_move: report.best_move,
                work: DriverWork::Cpu {
                    nodes: report.nodes,
                    completed_depth: report.completed_depth,
                },
            })
        })();
        retain_work_failure(&self.work, context, result, observation)
    }
}
fn retain_work_failure(
    journal: &ProcessWorkJournal,
    context: &SearchSessionContext,
    result: Result<SearchSessionReport, SearchSessionFailure>,
    observation: AttemptObservation,
) -> Result<SearchSessionReport, SearchSessionFailure> {
    if let Err(error) = journal.finish(context, &result, observation) {
        return match result {
            Err(mut primary) => {
                primary.detail.push_str("; work observation unavailable");
                Err(primary)
            }
            Ok(_) => Err(SearchSessionFailure::debug("WorkReceipt", &error)),
        };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration, pals::SearchAuthority};
    use rz_search::{
        cpu::CpuConfig,
        pals::engine::{
            DivergenceQuery, PalsConfig, RoleError, RoleEvaluation, RoleModel, RoleQuery,
        },
        time::{TimeBudget, TimeBudgetConfig, TimeControl},
    };

    struct DeadlineObserver {
        observed: Arc<Mutex<Vec<Instant>>>,
    }
    impl DeadlineObserver {
        fn observe(&self, deadline: Instant) -> RoleError {
            self.observed.lock().unwrap().push(deadline);
            // Stop at the first actual role query, without sleeps or CPU work.
            RoleError::Backend("test role deadline observed".into())
        }
    }
    impl RoleModel for DeadlineObserver {
        fn identity(&self) -> &str {
            "test-only-pals-role-deadline-observer"
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(self.observe(query.deadline))
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(self.observe(query.deadline))
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(self.observe(query.deadline))
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            Err(self.observe(query.deadline))
        }
    }

    #[test]
    fn pals_driver_role_query_uses_soft_target_without_extending_admission() {
        for admission_is_earlier in [false, true] {
            let observed = Arc::new(Mutex::new(Vec::new()));
            let driver = PalsSessionDriver::new(
                PalsConfig::default(),
                DeadlineObserver {
                    observed: Arc::clone(&observed),
                },
                CpuConfig {
                    tt_entries: 0,
                    ..CpuConfig::default()
                },
                1,
                1,
                1,
                EngineIdentity {
                    name: "PALS role deadline consumer test".into(),
                    author: "RoveZero contributors".into(),
                },
            )
            .unwrap();
            // Reproduce the CPU06 low-clock allocation while keeping every
            // timestamp in the future; the probe never waits for a deadline.
            let start = Instant::now() + Duration::from_secs(60);
            let budget = TimeBudget::new(
                start,
                TimeControl::Clock {
                    remaining: Duration::from_millis(1010),
                    increment: Duration::from_millis(1000),
                    moves_to_go: None,
                },
                TimeBudgetConfig::default(),
            )
            .unwrap();
            let mut control = SearchControl::from_budget(&budget, 1);
            let expected = if admission_is_earlier {
                // Shared contracts also permit admission to precede soft.
                control.soft_deadline = control.deadline;
                budget.admission_deadline
            } else {
                budget.soft_deadline
            };
            let authority = SearchAuthority {
                epoch: ProcessEpoch(1),
                game: GameGeneration(1),
                root: RootGeneration(1),
                implementation: driver.implementation(),
            };
            let context = SearchSessionContext {
                authority,
                control,
                cancellation: CancelToken::new(),
                shutdown_deadline: budget.output_deadline,
                current: Arc::new(Mutex::new(SessionScope::Search(authority))),
            };
            let failure = driver
                .run(&Position::startpos(), &context, &mut |_| {})
                .unwrap_err();
            assert_eq!(failure.code, "PalsSearch");
            assert!(failure.detail.contains("test role deadline observed"));
            assert_eq!(*observed.lock().unwrap(), [expected]);
            assert_eq!(context.control.deadline, budget.hard_deadline);
            assert!(!context.cancellation.is_canceled());
        }
    }

    #[derive(Default)]
    struct CheckerProbeState {
        starts: AtomicU64,
        resets: AtomicU64,
        analyses: AtomicU64,
        shutdowns: AtomicU64,
        fail_start: AtomicBool,
        fail_reset: AtomicBool,
        fail_shutdown: AtomicBool,
        late_start: AtomicBool,
        quarantine: AtomicBool,
        block_analysis: AtomicBool,
        analysis_entered: AtomicBool,
        block_reset: AtomicBool,
        reset_entered: AtomicBool,
        reset_deadlines: Mutex<Vec<Instant>>,
    }
    /// In-process lifecycle fixture only; its synthetic shutdown states are
    /// not real UCI subprocess, pipe drain or engine-strength evidence.
    struct CheckerProbe {
        identity: CheckerIdentity,
        state: Arc<CheckerProbeState>,
        started: bool,
        attempt: Option<CheckerAttempt>,
    }
    impl CheckerProbe {
        fn new(state: Arc<CheckerProbeState>) -> Self {
            Self {
                identity: CheckerIdentity::ExternalUci(
                    rz_search::cpu_checker::ExternalCheckerIdentity {
                        adapter_semantics: "test-only-driver-checker/1".into(),
                        binary_sha256: "a".repeat(64),
                        launch_arguments_sha256: "b".repeat(64),
                        declared_name: "declared fixture name".into(),
                        declared_version: "fixture-v1".into(),
                        declared_source: "in-process unit fixture".into(),
                        declared_license: "MIT".into(),
                        options: std::collections::BTreeMap::new(),
                        assets: Vec::new(),
                        model_metadata: rz_search::cpu_checker::ExternalModelMetadata {
                            weights_sha256: None,
                            training: rz_search::cpu_checker::ExternalTrainingKnowledge::Unknown,
                            declared_rights: None,
                            precision: None,
                        },
                    },
                ),
                state,
                started: false,
                attempt: None,
            }
        }
        fn record(&mut self, request: u64, nodes: Option<u64>) {
            self.attempt = Some(CheckerAttempt {
                work: rz_search::cpu_checker::CheckerWork {
                    nodes,
                    qnodes: None,
                    tt_hits: None,
                },
                elapsed: Duration::ZERO,
                external: Some(rz_search::cpu_checker::ExternalAttemptEvidence {
                    request_id: request,
                    partial_report: None,
                    process: CheckerShutdown {
                        quarantined: self.state.quarantine.load(Ordering::Acquire),
                        ..CheckerShutdown::default()
                    },
                }),
            });
        }
    }
    impl CpuChecker for CheckerProbe {
        fn identity(&self) -> &CheckerIdentity {
            &self.identity
        }
        fn conditions(&self) -> &str {
            "test-only-bounded-checker;not-a-real-process"
        }
        fn capabilities(&self) -> CheckerCapabilities {
            CheckerCapabilities {
                max_depth: 64,
                max_prefix_plies: 4096,
                max_root_moves: 256,
                root_moves: true,
                divergence: true,
                resume: false,
                selective_search: None,
            }
        }
        fn start(
            &mut self,
            _: Instant,
            cancel: &AtomicBool,
        ) -> Result<(), rz_search::cpu_checker::CheckerError> {
            self.state.starts.fetch_add(1, Ordering::AcqRel);
            self.record(0, None);
            if self.state.fail_start.load(Ordering::Acquire) {
                return Err(rz_search::cpu_checker::CheckerError::External {
                    stage: "test-startup",
                    code: "preserved-startup-failure",
                });
            }
            self.started = true;
            if self.state.late_start.load(Ordering::Acquire) {
                cancel.store(true, Ordering::Release);
            }
            Ok(())
        }
        fn startup_uci(&self) -> Option<ExternalUciIdentity> {
            self.started.then(|| ExternalUciIdentity {
                name: "actual fixture handshake name".into(),
                author: Some("fixture author".into()),
            })
        }
        fn last_attempt(&self) -> Option<&CheckerAttempt> {
            self.attempt.as_ref()
        }
        fn reset_attempt(&mut self) {
            self.attempt = None;
        }
        fn analyze(
            &mut self,
            _: &Position,
            limits: rz_search::cpu::CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<rz_search::cpu_checker::CheckerReport, rz_search::cpu_checker::CheckerError>
        {
            let request = self.state.analyses.fetch_add(1, Ordering::AcqRel) + 1;
            self.state.analysis_entered.store(true, Ordering::Release);
            while self.state.block_analysis.load(Ordering::Acquire)
                && !cancel.load(Ordering::Acquire)
                && limits
                    .deadline
                    .is_some_and(|deadline| Instant::now() < deadline)
            {
                thread::yield_now();
            }
            self.record(request, Some(7));
            Err(rz_search::cpu_checker::CheckerError::External {
                stage: "test-analysis",
                code: "actual-observed-work-failure",
            })
        }
        fn analyze_root_moves(
            &mut self,
            position: &Position,
            _: &[BoardMove],
            limits: rz_search::cpu::CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<rz_search::cpu_checker::CheckerReport, rz_search::cpu_checker::CheckerError>
        {
            self.analyze(position, limits, cancel)
        }
        fn new_game(
            &mut self,
            deadline: Instant,
            cancel: &AtomicBool,
        ) -> Result<(), rz_search::cpu_checker::CheckerError> {
            self.state.resets.fetch_add(1, Ordering::AcqRel);
            self.state.reset_deadlines.lock().unwrap().push(deadline);
            self.state.reset_entered.store(true, Ordering::Release);
            while self.state.block_reset.load(Ordering::Acquire)
                && !cancel.load(Ordering::Acquire)
                && Instant::now() < deadline
            {
                thread::yield_now();
            }
            if self.state.fail_reset.load(Ordering::Acquire) {
                return Err(rz_search::cpu_checker::CheckerError::External {
                    stage: "test-reset",
                    code: "fallible-reset-not-acknowledged",
                });
            }
            Ok(())
        }
        fn shutdown(
            &mut self,
            _: Instant,
        ) -> Result<CheckerShutdown, rz_search::cpu_checker::CheckerError> {
            self.state.shutdowns.fetch_add(1, Ordering::AcqRel);
            let shutdown = CheckerShutdown {
                cleanup_complete: !self.state.fail_shutdown.load(Ordering::Acquire),
                quarantined: self.state.fail_shutdown.load(Ordering::Acquire),
                ..CheckerShutdown::default()
            };
            if let Some(attempt) = self
                .attempt
                .as_mut()
                .and_then(|value| value.external.as_mut())
            {
                attempt.process = shutdown;
            }
            if shutdown.cleanup_complete {
                Ok(shutdown)
            } else {
                Err(rz_search::cpu_checker::CheckerError::External {
                    stage: "test-shutdown",
                    code: "physical-owner-unconfirmed",
                })
            }
        }
    }
    fn checker_driver(
        state: Arc<CheckerProbeState>,
    ) -> PalsSessionDriver<rz_search::pals::engine::LegalOrderRoleMock> {
        PalsSessionDriver::new_with_checker(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                cpu_nodes_per_task: 8,
                ..PalsConfig::default()
            },
            rz_search::pals::engine::LegalOrderRoleMock,
            CheckerProbe::new(state),
            1,
            8,
            1,
            EngineIdentity {
                name: "PALS driver checker test".into(),
                author: "RoveZero contributors".into(),
            },
        )
        .unwrap()
    }
    fn checker_context(driver: &dyn SearchSessionDriver) -> SearchSessionContext {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: driver.implementation(),
        };
        let now = Instant::now();
        let mut control = SearchControl::new(now + Duration::from_secs(5), 8);
        control.soft_deadline = now + Duration::from_secs(4);
        SearchSessionContext {
            authority,
            control,
            cancellation: CancelToken::new(),
            shutdown_deadline: now + Duration::from_secs(6),
            current: Arc::new(Mutex::new(SessionScope::Search(authority))),
        }
    }

    fn refinement_config() -> PalsConfig {
        PalsConfig {
            beam_width: 1,
            line_plies: 2,
            cpu_nodes_per_task: 8,
            ..PalsConfig::default()
        }
    }
    fn refinement_cpu_config() -> CpuConfig {
        CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        }
    }
    fn refinement_name() -> EngineIdentity {
        EngineIdentity {
            name: "PALS immutable recheck lane fixture".into(),
            author: "RoveZero contributors".into(),
        }
    }
    fn refinement_driver<M: RoleModel + 'static>(model: M) -> PalsSessionDriver<M> {
        PalsSessionDriver::new_with_refinement_policy(
            refinement_config(),
            model,
            refinement_cpu_config(),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        )
        .unwrap()
    }
    fn refinement_checker() -> OwnedCpuChecker {
        OwnedCpuChecker::new(rz_search::cpu::CpuEngine::new(refinement_cpu_config()).unwrap())
            .unwrap()
    }

    #[test]
    fn disabled_policy_keeps_v1_v2_hashes_and_opt_in_has_a_separate_actual_namespace() {
        use rz_search::pals::engine::LegalOrderRoleMock;
        let legacy = PalsSessionDriver::new(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_cpu_config(),
            1,
            8,
            1,
            refinement_name(),
        )
        .unwrap();
        let disabled = PalsSessionDriver::new_with_refinement_policy(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_cpu_config(),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::Disabled,
        )
        .unwrap();
        assert_eq!(legacy.implementation(), disabled.implementation());
        assert_eq!(
            legacy.work_receipt().unwrap(),
            disabled.work_receipt().unwrap()
        );
        assert!(legacy.search_policy_registration().is_none());
        let explicit = PalsSessionDriver::new_with_checker(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_checker(),
            1,
            8,
            1,
            refinement_name(),
        )
        .unwrap();
        let explicit_disabled = PalsSessionDriver::new_with_checker_and_refinement_policy(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_checker(),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::Disabled,
        )
        .unwrap();
        assert_eq!(
            explicit.implementation(),
            explicit_disabled.implementation()
        );
        assert_eq!(
            explicit.work_receipt().unwrap(),
            explicit_disabled.work_receipt().unwrap()
        );
        let selected = refinement_driver(LegalOrderRoleMock);
        assert_ne!(selected.implementation(), legacy.implementation());
        assert_ne!(selected.implementation(), explicit.implementation());
        let policy = selected.search_policy_registration().unwrap();
        policy.validate().unwrap();
        let engine = selected.engine.lock().unwrap();
        assert_eq!(policy.search_identity, engine.search_identity());
        assert_eq!(
            policy.conditions_sha256,
            <[u8; 32]>::from(Sha256::digest(
                engine.refinement_conditions().unwrap().as_bytes(),
            ))
        );
        assert_eq!(
            selected
                .work_receipt()
                .unwrap()
                .unwrap()
                .pals_search_policy
                .as_ref(),
            Some(policy)
        );
        #[cfg(feature = "search-work-receipts")]
        assert_eq!(
            serde_json::to_vec(&legacy.work_receipt().unwrap()).unwrap(),
            serde_json::to_vec(&disabled.work_receipt().unwrap()).unwrap(),
        );
        let changed_budget = PalsSessionDriver::new_with_refinement_policy(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_cpu_config(),
            1,
            9,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        )
        .unwrap();
        assert_ne!(selected.implementation(), changed_budget.implementation());
    }

    #[test]
    fn opted_in_driver_early_cancel_preserves_selection_with_no_recheck_claim() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let driver = refinement_driver(DeadlineObserver {
            observed: Arc::clone(&observed),
        });
        let before = driver.work_receipt().unwrap().unwrap();
        let context = checker_context(&driver);
        context.cancellation.cancel();
        let report = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap();
        assert!(report.best_move.is_none());
        assert!(observed.lock().unwrap().is_empty());
        let after = driver.work_receipt().unwrap().unwrap();
        assert_eq!(after.pals_search_policy, before.pals_search_policy);
        assert_eq!(
            (
                after.go_invocations,
                after.successful_returns,
                after.canceled_returns
            ),
            (1, 1, 1)
        );
        assert_eq!(after.unobserved_work_invocations, 0);
        let totals = after.pals.unwrap();
        assert_eq!(totals.role_calls, Some(0));
        assert_eq!(totals.completed_cpu_tasks, Some(0));
        assert_eq!(totals.consumed_role_outputs, Some(0));
    }

    #[test]
    fn opted_in_driver_preserves_backend_failure_marker_and_original_deadline() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let driver = refinement_driver(DeadlineObserver {
            observed: Arc::clone(&observed),
        });
        let context = checker_context(&driver);
        let hard = context.control.deadline;
        let soft = context
            .control
            .soft_deadline
            .min(context.control.admission_deadline);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap_err();
        assert_eq!(failure.code, "PalsSearch");
        assert!(failure.detail.contains("test role deadline observed"));
        assert_eq!(*observed.lock().unwrap(), [soft]);
        assert_eq!(context.control.deadline, hard);
        let work = driver.work_receipt().unwrap().unwrap();
        assert_eq!(
            work.pals_search_policy.as_ref(),
            driver.search_policy_registration()
        );
        assert_eq!((work.successful_returns, work.failed_returns), (0, 1));
        assert_eq!(work.active_invocations, 0);
        assert_eq!(work.pals.unwrap().accepted_proposer_outputs, Some(0));
    }

    #[test]
    fn selected_external_checker_is_rejected_without_start_or_analysis() {
        let state = Arc::new(CheckerProbeState::default());
        let result = PalsSessionDriver::new_with_checker_and_refinement_policy(
            refinement_config(),
            rz_search::pals::engine::LegalOrderRoleMock,
            CheckerProbe::new(Arc::clone(&state)),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        );
        let Err(failure) = result else {
            panic!("unsupported foreign lane admitted")
        };
        assert_eq!(failure.code, "PalsRefinementUnsupported");
        assert_eq!(state.starts.load(Ordering::Acquire), 0);
        assert_eq!(state.analyses.load(Ordering::Acquire), 0);
    }

    #[test]
    fn registration_drift_fails_before_any_role_work() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut driver = refinement_driver(DeadlineObserver {
            observed: Arc::clone(&observed),
        });
        driver.search_policy.as_mut().unwrap().conditions_sha256[0] ^= 1;
        let context = checker_context(&driver);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap_err();
        assert_eq!(failure.code, "PalsSearchPolicyIdentity");
        assert!(observed.lock().unwrap().is_empty());
        let receipt = driver.work_receipt().unwrap().unwrap();
        receipt
            .pals_search_policy
            .as_ref()
            .unwrap()
            .validate()
            .unwrap();
        assert_eq!(receipt.failed_returns, 1);
        assert_eq!(receipt.pals.unwrap().role_calls, Some(0));
    }

    fn start_probe(driver: &PalsSessionDriver<rz_search::pals::engine::LegalOrderRoleMock>) {
        driver
            .start_checker(
                Instant::now() + Duration::from_secs(2),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
    }

    #[cfg(feature = "search-work-receipts")]
    #[test]
    fn historical_ready_resource_snapshot_is_preserved_after_helper_cleanup() {
        let state = Arc::new(CheckerProbeState::default());
        let driver = checker_driver(state);
        assert!(driver.checker_startup_resources().unwrap().is_none());
        assert!(
            driver
                .checker_startup_resource_unavailable()
                .unwrap()
                .is_none()
        );
        start_probe(&driver);
        // In-process probes have no real helper PID, hence actual observation
        // remains None. Inject a pure snapshot only to test historical storage.
        assert!(driver.checker_startup_resources().unwrap().is_none());
        assert_eq!(
            driver.checker_startup_resource_unavailable().unwrap(),
            Some("helper_process_identity_not_observed")
        );
        let fixture = crate::pals_attestation::checker::resource::fixture();
        {
            let mut lifecycle = driver.checker_lifecycle.lock().unwrap();
            lifecycle.startup_resources = Some(fixture.clone());
            lifecycle.startup_resource_unavailable = None;
        }
        let before = crate::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
            &driver,
            &"a".repeat(64),
            &"b".repeat(64),
        )
        .unwrap();
        driver
            .finish_checker(Instant::now() + Duration::from_secs(1))
            .unwrap();
        assert_eq!(driver.checker_startup_resources().unwrap(), Some(fixture));
        let after = crate::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
            &driver,
            &"a".repeat(64),
            &"b".repeat(64),
        )
        .unwrap();
        let before = serde_json::to_value(before).unwrap();
        let after = serde_json::to_value(after).unwrap();
        assert_eq!(
            before["startup_resource_observation"],
            after["startup_resource_observation"]
        );
        assert_eq!(
            after["startup_resource_observation"]["helper"]["proc_start_ticks_after"],
            456
        );
        assert_eq!(after["applied_option_values"], "unknown");
        assert!(after.get("startup_resource_unavailable").is_none());
    }
    fn wait_probe(flag: &AtomicBool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !flag.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(
            flag.load(Ordering::Acquire),
            "bounded fixture entry not observed"
        );
    }

    #[test]
    fn checker_registration_selects_model_resolver_and_full_v2_namespace() {
        let driver = checker_driver(Arc::new(CheckerProbeState::default()));
        let registration = driver.checker_registration();
        assert_eq!(
            registration.resolver_version,
            rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION
        );
        assert!(registration.model_value.is_some());
        assert!(registration.owned_descriptor.is_none());
        assert_eq!(
            driver
                .work_receipt()
                .unwrap()
                .unwrap()
                .pals_resolver
                .unwrap(),
            PalsResolverIdentity::from_semantics(
                registration.resolver_version,
                registration.resolver_semantics
            )
        );
        let config = PalsConfig {
            beam_width: 1,
            line_plies: 2,
            cpu_nodes_per_task: 8,
            ..PalsConfig::default()
        };
        let original = checker_implementation(registration, &config, 1, 8, 1);
        assert_eq!(original, driver.implementation());
        let changed_digest =
            |changed: &PalsCheckerRegistration| checker_implementation(changed, &config, 1, 8, 1);
        let mut changed = registration.clone();
        changed.conditions.push_str(";different-window");
        assert_ne!(original, changed_digest(&changed));
        changed = registration.clone();
        changed.capabilities.selective_search = Some(false);
        assert_ne!(original, changed_digest(&changed));
        changed = registration.clone();
        changed.model_value.as_mut().unwrap().model_epoch[0] ^= 1;
        assert_ne!(original, changed_digest(&changed));
        changed = registration.clone();
        changed.model_value.as_mut().unwrap().precision = "different-precision".into();
        assert_ne!(original, changed_digest(&changed));
        changed = registration.clone();
        let CheckerIdentity::ExternalUci(identity) = &mut changed.identity else {
            panic!()
        };
        identity.options.insert("Hash".into(), "256".into());
        assert_ne!(original, changed_digest(&changed));
        assert_ne!(
            original,
            checker_implementation(registration, &config, 1, 9, 1)
        );
    }

    #[test]
    fn explicit_start_preserves_failure_zero_request_and_actual_success_identity() {
        let state = Arc::new(CheckerProbeState::default());
        state.fail_start.store(true, Ordering::Release);
        let driver = checker_driver(Arc::clone(&state));
        assert!(!driver.checker_started());
        let failure = driver
            .run(
                &Position::startpos(),
                &checker_context(&driver),
                &mut |_| {},
            )
            .unwrap_err();
        assert_eq!(failure.code, "PalsCheckerNotStarted");
        assert_eq!(state.starts.load(Ordering::Acquire), 0);
        let failure = driver
            .start_checker(
                Instant::now() + Duration::from_secs(2),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap_err();
        assert_eq!(failure.code, "PalsCheckerStartup");
        assert!(failure.detail.contains("preserved-startup-failure"));
        assert!(!driver.checker_started());
        assert_eq!(
            driver
                .checker_last_attempt()
                .unwrap()
                .unwrap()
                .external
                .unwrap()
                .request_id,
            0
        );
        assert!(driver.checker_attempts().unwrap().is_empty());
        assert!(driver.checker_startup_uci().unwrap().is_none());
        assert!(
            driver
                .finish_checker(Instant::now() + Duration::from_secs(2))
                .unwrap()
                .cleanup_complete
        );
        assert_eq!(state.shutdowns.load(Ordering::Acquire), 1);

        let driver = checker_driver(Arc::new(CheckerProbeState::default()));
        start_probe(&driver);
        assert!(driver.checker_started());
        assert_eq!(
            driver.checker_startup_uci().unwrap().unwrap().name,
            "actual fixture handshake name"
        );
        assert_ne!(
            driver.checker_startup_uci().unwrap().unwrap().name.as_str(),
            match &driver.checker_registration().identity {
                CheckerIdentity::ExternalUci(identity) => identity.declared_name.as_str(),
                _ => panic!(),
            }
        );
    }

    #[test]
    fn late_startup_does_not_gain_ready_or_admit_go() {
        let state = Arc::new(CheckerProbeState::default());
        state.late_start.store(true, Ordering::Release);
        let driver = checker_driver(state);
        let cancel = Arc::new(AtomicBool::new(false));
        let failure = driver
            .start_checker(Instant::now() + Duration::from_secs(2), Arc::clone(&cancel))
            .unwrap_err();
        assert_eq!(failure.code, "PalsCheckerStartupLate");
        assert!(cancel.load(Ordering::Acquire));
        assert!(!driver.checker_started());
        assert!(driver.checker_startup_uci().unwrap().is_none());
        assert!(driver.checker_last_attempt().unwrap().is_some());
        assert_eq!(
            driver
                .run(
                    &Position::startpos(),
                    &checker_context(&driver),
                    &mut |_| {}
                )
                .unwrap_err()
                .code,
            "PalsCheckerNotStarted"
        );
        driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap();
    }

    #[test]
    fn foreign_reset_uses_actual_controls_and_failed_reset_remains_unapplied() {
        let state = Arc::new(CheckerProbeState::default());
        let driver = checker_driver(Arc::clone(&state));
        start_probe(&driver);
        state.fail_reset.store(true, Ordering::Release);
        driver.reset_game().unwrap();
        assert_eq!(state.resets.load(Ordering::Acquire), 0);
        let context = checker_context(&driver);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap_err();
        assert_eq!(failure.code, "PalsReset");
        assert!(failure.detail.contains("fallible-reset-not-acknowledged"));
        assert_eq!(
            *state.reset_deadlines.lock().unwrap(),
            [context
                .control
                .soft_deadline
                .min(context.control.admission_deadline)]
        );
        assert_eq!(driver.reset_applied.load(Ordering::Acquire), 0);
        assert_eq!(state.analyses.load(Ordering::Acquire), 0);
        state.fail_reset.store(false, Ordering::Release);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap_err();
        assert_eq!(failure.code, "PalsSearch");
        assert_eq!(driver.reset_applied.load(Ordering::Acquire), 1);
        assert_eq!(state.resets.load(Ordering::Acquire), 2);
        // Normal idle fixture retains no exit/drain event and is not quarantine.
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Confirmed
        );
    }

    #[test]
    fn newer_reset_is_not_erased_by_older_success_and_generation_overflow_is_explicit() {
        let state = Arc::new(CheckerProbeState::default());
        state.block_reset.store(true, Ordering::Release);
        let driver = Arc::new(checker_driver(Arc::clone(&state)));
        start_probe(&driver);
        driver.reset_game().unwrap();
        let running = Arc::clone(&driver);
        let thread = thread::spawn(move || {
            running.run(
                &Position::startpos(),
                &checker_context(&*running),
                &mut |_| {},
            )
        });
        wait_probe(&state.reset_entered);
        driver.reset_game().unwrap();
        state.block_reset.store(false, Ordering::Release);
        assert_eq!(thread.join().unwrap().unwrap_err().code, "PalsSearch");
        assert_eq!(driver.reset_requested.load(Ordering::Acquire), 2);
        assert_eq!(driver.reset_applied.load(Ordering::Acquire), 1);
        assert_eq!(
            driver
                .run(
                    &Position::startpos(),
                    &checker_context(&*driver),
                    &mut |_| {}
                )
                .unwrap_err()
                .code,
            "PalsSearch"
        );
        assert_eq!(driver.reset_applied.load(Ordering::Acquire), 2);
        assert_eq!(state.resets.load(Ordering::Acquire), 2);
        driver.reset_requested.store(u64::MAX, Ordering::Release);
        assert!(
            driver
                .reset_game()
                .unwrap_err()
                .to_string()
                .contains("generation exhausted")
        );
        assert_eq!(state.resets.load(Ordering::Acquire), 2);
    }

    #[test]
    fn finite_finish_cancels_active_owner_preserves_work_and_closes_future_admission() {
        let state = Arc::new(CheckerProbeState::default());
        state.block_analysis.store(true, Ordering::Release);
        let driver = Arc::new(checker_driver(Arc::clone(&state)));
        start_probe(&driver);
        let context = checker_context(&*driver);
        let cancel = Arc::clone(&context.control.cancellation);
        let running = Arc::clone(&driver);
        let thread =
            thread::spawn(move || running.run(&Position::startpos(), &context, &mut |_| {}));
        wait_probe(&state.analysis_entered);
        let shutdown = driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert!(shutdown.cleanup_complete);
        assert!(cancel.load(Ordering::Acquire));
        assert_eq!(thread.join().unwrap().unwrap_err().code, "PalsSearch");
        assert_eq!(state.analyses.load(Ordering::Acquire), 1);
        let attempts = driver.checker_attempts().unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].work.nodes, Some(7));
        assert_eq!(
            driver.checker_last_attempt().unwrap().unwrap().work.nodes,
            Some(7)
        );
        assert_eq!(driver.checker_shutdown().unwrap(), Some(shutdown));
        assert_eq!(
            driver
                .finish_checker(Instant::now() + Duration::from_secs(2))
                .unwrap(),
            shutdown
        );
        assert_eq!(state.shutdowns.load(Ordering::Acquire), 1);
        assert_eq!(
            driver
                .run(
                    &Position::startpos(),
                    &checker_context(&*driver),
                    &mut |_| {}
                )
                .unwrap_err()
                .code,
            "PalsCheckerClosed"
        );
        assert!(driver.reset_game().is_err());
    }

    #[test]
    fn lock_timeout_and_quarantine_never_become_confirmed_shutdown() {
        let state = Arc::new(CheckerProbeState::default());
        let driver = checker_driver(Arc::clone(&state));
        start_probe(&driver);
        let owner = driver.engine.lock().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        driver.checker_lifecycle.lock().unwrap().active_cancel = Some(Arc::clone(&cancel));
        let failure = driver
            .finish_checker(Instant::now() + Duration::from_millis(5))
            .unwrap_err();
        assert_eq!(failure.code, "PalsCheckerShutdownLockDeadline");
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Unknown
        );
        assert!(cancel.load(Ordering::Acquire));
        assert!(driver.checker_shutdown().unwrap().is_none());
        assert_eq!(state.shutdowns.load(Ordering::Acquire), 0);
        drop(owner);
        driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap();

        let state = Arc::new(CheckerProbeState::default());
        state.quarantine.store(true, Ordering::Release);
        state.fail_shutdown.store(true, Ordering::Release);
        let driver = checker_driver(state);
        start_probe(&driver);
        let failure = driver
            .run(
                &Position::startpos(),
                &checker_context(&driver),
                &mut |_| {},
            )
            .unwrap_err();
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Unknown
        );
        let failure = driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap_err();
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Unknown
        );
        let actual = driver.checker_shutdown().unwrap().unwrap();
        assert!(actual.quarantined && !actual.cleanup_complete);
        assert!(!actual.exit_observed && !actual.stdout_drained && !actual.stderr_drained);
        assert_eq!(
            driver.checker_last_attempt().unwrap().unwrap().work.nodes,
            Some(7)
        );
    }

    #[test]
    fn old_own_driver_retains_raw_resolver_and_no_foreign_shutdown_claims() {
        let driver = PalsSessionDriver::new(
            PalsConfig::default(),
            rz_search::pals::engine::LegalOrderRoleMock,
            CpuConfig {
                tt_entries: 0,
                ..CpuConfig::default()
            },
            1,
            8,
            1,
            EngineIdentity {
                name: "own compatibility fixture".into(),
                author: "RoveZero contributors".into(),
            },
        )
        .unwrap();
        assert!(driver.checker_started());
        assert!(matches!(
            driver.checker_registration().identity,
            CheckerIdentity::Owned(_)
        ));
        assert!(driver.checker_registration().model_value.is_none());
        assert_eq!(
            driver.checker_registration().resolver_version,
            rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION
        );
        let shutdown = driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert_eq!(shutdown, CheckerShutdown::owned_no_process());
        let cpu = CpuSessionDriver::new(
            CpuConfig {
                tt_entries: 0,
                ..CpuConfig::default()
            },
            8,
        )
        .unwrap();
        assert!(cpu.pals_checker_registration().is_none());
        assert!(
            cpu.finish_checker(Instant::now() + Duration::from_secs(1))
                .unwrap()
                .is_none()
        );
    }
}
