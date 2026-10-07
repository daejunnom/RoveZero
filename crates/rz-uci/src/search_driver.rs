//! Startup-selected search sessions below UCI. A session returns only after its
//! owned work has completed or been drained; publication remains with UCI.
//! CPU/PALS counters are never translated into PUCT simulations or NN receipts.

use crate::{EngineIdentity, contracts::SessionScope};
use rz_contracts::{CancelToken, ContractError, Digest, ErrorCode, Stage};
use rz_position::{BoardMove, Position};
use rz_search::driver::SearchControl;
use sha2::{Digest as _, Sha256};
use std::{
    fmt,
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
mod work;
use work::{AttemptObservation, ProcessWorkJournal};
pub use work::{CpuWorkTotals, PalsWorkTotals, ProcessSearchWorkReceipt};

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

/// Persistent P/C + own CPU refinement session. The model is selected once at
/// startup; callers must explicitly identify a mock model as mock.
pub struct PalsSessionDriver<M: rz_search::pals::engine::RoleModel + 'static> {
    engine: Mutex<rz_search::pals::engine::PalsEngine<M>>,
    reset_pending: AtomicBool,
    implementation: Digest,
    identity: EngineIdentity,
    max_rounds: u64,
    max_cpu_nodes: u64,
    cpu_depth: u16,
    work: ProcessWorkJournal,
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
        let engine = rz_search::pals::engine::PalsEngine::new(config, model, cpu)
            .map_err(|error| SearchSessionFailure::debug("PalsConstructor", &error))?;
        Ok(Self {
            engine: Mutex::new(engine),
            reset_pending: AtomicBool::new(false),
            implementation: Digest(hash.finalize().into()),
            identity,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            work: ProcessWorkJournal::new(SearchKind::Pals),
        })
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
    fn reset_game(&self) -> Result<(), ContractError> {
        self.reset_pending.store(true, Ordering::Release);
        match self.engine.try_lock() {
            Ok(mut engine) => {
                engine.new_game();
                self.reset_pending.store(false, Ordering::Release);
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
            if self.reset_pending.swap(false, Ordering::AcqRel) {
                engine.new_game();
            }
            observation = AttemptObservation::PalsStarted;
            let outcome = engine.search_with_progress(
                position,
                rz_search::pals::engine::PalsLimits {
                    deadline: context.control.admission_deadline,
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
            let report = outcome.map_err(|error| {
                if matches!(
                    error,
                    rz_search::pals::engine::PalsError::Role(
                        rz_search::pals::engine::RoleError::PhysicalCompletionUnknown
                    )
                ) {
                    SearchSessionFailure::physical_completion_unknown()
                } else {
                    SearchSessionFailure::debug("PalsSearch", &error)
                }
            })?;
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
    fn debug(code: &'static str, value: &dyn fmt::Debug) -> Self {
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
            let report = engine
                .analyze_with_progress(
                    position,
                    limits,
                    &context.control.cancellation,
                    |iteration| {
                        if let Some(movement) = iteration.best_move {
                            progress(movement);
                        }
                    },
                )
                .map_err(|error| SearchSessionFailure::debug("CpuSearch", &error))?;
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
