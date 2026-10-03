//! Concrete Rules → UCI → search → common evaluation assembly.
//! Model loading and physical execution remain the injected factory's responsibility.

use crate::{bridge::BuildSearchSettings, *};
use rz_contracts as contract;
use rz_position::{
    BoardMove, Position, PositionLimits,
    contracts::{ContractPosition, ContractState, RulesState},
};
use rz_search::{
    TreeLimits,
    driver::{CheckedPosition, SearchControl},
};
use std::{
    collections::VecDeque,
    fmt,
    io::Write,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

fn failure(code: contract::ErrorCode, detail: &'static str) -> contract::ContractError {
    contract::ContractError::new(code, contract::Stage::Admission, detail)
}

/// All position forks obtain a checked fresh owner. Clone shares immutable state.
#[derive(Debug, Default)]
pub struct OwnerRegistry(AtomicU64);
impl OwnerRegistry {
    pub fn allocate(&self) -> Result<contract::OwnerId, contract::ContractError> {
        self.0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map(|previous| contract::OwnerId(previous + 1))
            .map_err(|_| {
                failure(
                    contract::ErrorCode::ResourceExhausted,
                    "owner sequence exhausted",
                )
            })
    }
}

#[derive(Clone, Debug)]
pub struct RulesSearchPosition {
    position: Arc<ContractPosition>,
    state: ContractState,
    owners: Arc<OwnerRegistry>,
}
impl RulesSearchPosition {
    fn new(
        position: ContractPosition,
        owners: Arc<OwnerRegistry>,
    ) -> Result<Self, contract::ContractError> {
        let state = position.export()?;
        Ok(Self {
            position: Arc::new(position),
            state,
            owners,
        })
    }
    pub fn state(&self) -> &ContractState {
        &self.state
    }
}
impl CheckedPosition for RulesSearchPosition {
    type Move = contract::Move;
    type Error = contract::ContractError;
    fn classify(&self) -> Result<Option<f64>, Self::Error> {
        Ok(self.state.terminal_wdl().map(|wdl| f64::from(wdl.value())))
    }
    fn legal_moves(&self) -> Result<Vec<Self::Move>, Self::Error> {
        Ok(self.state.legal_moves().moves().to_vec())
    }
    fn play(&self, movement: &Self::Move) -> Result<Self, Self::Error> {
        let mut fork =
            ContractPosition::new(self.owners.allocate()?, self.position.position().clone());
        let fresh = fork.export()?;
        fork.make_from_view(&fresh, *movement)?;
        Self::new(fork, Arc::clone(&self.owners))
    }
}

impl rz_search::contracts::ContractPosition for RulesSearchPosition {
    type State = RulesState;
    fn snapshot(&self) -> &contract::PositionSnapshot<RulesState> {
        self.state.snapshot()
    }
    fn legal(&self) -> &contract::LegalMoveView {
        self.state.legal_moves()
    }
    fn play(&self, movement: &contract::Move) -> Result<Self, contract::ContractError> {
        CheckedPosition::play(self, movement)
    }
    fn validate_authority(&self) -> Result<(), contract::ContractError> {
        if !self
            .position
            .position()
            .matches_snapshot(self.state.rules().snapshot())
            || self.state.snapshot().identity() != self.state.legal_moves().state()
        {
            return Err(failure(
                contract::ErrorCode::IdentityMismatch,
                "Rules state no longer owns its frozen legal view",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct RulesUciPort {
    owners: Arc<OwnerRegistry>,
    limits: PositionLimits,
}
impl RulesUciPort {
    pub fn new(owners: Arc<OwnerRegistry>, limits: PositionLimits) -> Self {
        Self { owners, limits }
    }
}
impl PositionPort for RulesUciPort {
    type Snapshot = RulesSearchPosition;
    type Error = contract::ContractError;
    fn prepare(
        &self,
        spec: &PositionSpec,
    ) -> Result<PreparedPosition<Self::Snapshot>, Self::Error> {
        // The entire trace lives in a temporary owner. No partial state enters Session.
        let position = match &spec.base {
            PositionBase::StartPos => Position::startpos_with_limits(self.limits)?,
            PositionBase::Fen(fen) => Position::from_fen_with_limits(fen, self.limits)?,
        };
        let mut position = ContractPosition::new(self.owners.allocate()?, position);
        for text in &spec.moves {
            let movement = contract::Move::try_from(BoardMove::from_uci(text)?)?;
            let state = position.export()?;
            position.make_from_view(&state, movement)?;
        }
        let snapshot = RulesSearchPosition::new(position, Arc::clone(&self.owners))?;
        let exact_terminal = snapshot.state.terminal_wdl().is_some();
        // Automatic terminals may still have geometric legal moves; UCI emits 0000.
        let legal_moves = if exact_terminal {
            Vec::new()
        } else {
            snapshot
                .state
                .legal_moves()
                .moves()
                .iter()
                .map(|movement| move_text(*movement))
                .collect::<Result<_, _>>()?
        };
        Ok(PreparedPosition {
            snapshot,
            legal_moves,
            exact_terminal,
        })
    }
}

pub fn move_text(movement: contract::Move) -> Result<String, contract::ContractError> {
    Ok(BoardMove::try_from(movement)?.to_string())
}

/// A clone preserves exactly one process origin and domain for every consumer.
#[derive(Clone)]
pub struct ProcessClock {
    epoch: contract::ProcessEpoch,
    origin: Instant,
    ids: Arc<rz_search::contracts::IdAllocator>,
    executions: Arc<AtomicU64>,
}
impl fmt::Debug for ProcessClock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessClock")
            .field("epoch", &self.epoch)
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}
impl ProcessClock {
    pub fn new(epoch: contract::ProcessEpoch) -> Self {
        Self {
            epoch,
            origin: Instant::now(),
            ids: Arc::new(rz_search::contracts::IdAllocator::new(epoch)),
            executions: Arc::new(AtomicU64::new(0)),
        }
    }
    pub fn domain(&self) -> contract::ClockDomain {
        contract::ClockDomain(self.epoch)
    }
    pub fn epoch(&self) -> contract::ProcessEpoch {
        self.epoch
    }
    pub(crate) fn reserve_executions(&self, count: u64) -> Result<u64, contract::ContractError> {
        if count == 0 {
            return Err(failure(
                contract::ErrorCode::InvalidInput,
                "physical ID range must be finite and positive",
            ));
        }
        self.executions
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(count)
            })
            .map_err(|_| {
                failure(
                    contract::ErrorCode::ResourceExhausted,
                    "physical execution ID range exhausted",
                )
            })
    }
    pub fn b_clock(&self) -> rz_search::contract_time::InstantClock {
        rz_search::contract_time::InstantClock::new(self.domain(), self.origin)
    }
    pub fn now(&self) -> Result<contract::MonotonicTick, contract::ContractError> {
        self.tick_at(Instant::now())
    }
    pub fn tick_at(
        &self,
        instant: Instant,
    ) -> Result<contract::MonotonicTick, contract::ContractError> {
        let elapsed = instant.checked_duration_since(self.origin).ok_or_else(|| {
            failure(
                contract::ErrorCode::InvalidInput,
                "tick predates process clock origin",
            )
        })?;
        Ok(contract::MonotonicTick(
            u64::try_from(elapsed.as_nanos()).map_err(|_| {
                failure(
                    contract::ErrorCode::ResourceExhausted,
                    "process clock overflow",
                )
            })?,
        ))
    }
    pub fn deadline(
        &self,
        instant: Instant,
    ) -> Result<contract::Deadline, contract::ContractError> {
        Ok(contract::Deadline {
            clock: self.domain(),
            at: self.tick_at(instant)?,
        })
    }
}
impl rz_search::contract_time::ContractClock for ProcessClock {
    fn domain(&self) -> contract::ClockDomain {
        self.domain()
    }
    fn now(&self) -> Result<contract::MonotonicTick, contract::ContractError> {
        self.now()
    }
}

#[derive(Clone, Debug)]
pub struct SearchAuthority {
    current: Arc<Mutex<contract::AcceptanceScope>>,
    cancel: contract::CancelToken,
}
impl SearchAuthority {
    #[cfg(test)]
    pub(crate) fn isolated(scope: contract::AcceptanceScope) -> Self {
        Self {
            current: Arc::new(Mutex::new(scope)),
            cancel: contract::CancelToken::new(),
        }
    }
    pub fn current_scope(&self) -> Result<contract::AcceptanceScope, contract::ContractError> {
        self.current.lock().map(|scope| *scope).map_err(|_| {
            failure(
                contract::ErrorCode::BackendFailure,
                "acceptance authority poisoned",
            )
        })
    }
    pub fn cancel_token(&self) -> contract::CancelToken {
        self.cancel.clone()
    }
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

#[derive(Clone, Debug)]
pub struct EvaluatorProfile {
    pub model: Arc<contract::ModelDescriptor>,
    pub precision: contract::PrecisionProfile,
    pub backend: contract::Digest,
    pub compute: contract::ComputeBudget,
    pub bytes: contract::ByteBudget,
}

/// Shutdown closes runtime admission and drains physical work within its finite
/// bound. It must preserve the UCI owner's root output permission: a natural
/// completion can still be queued. UCI closes shared cancellation on its own
/// stop/root/deadline/quit or completed-output acceptance boundary.
pub trait ManagedEvaluator: contract::Evaluator<RulesState> + Send {
    fn shutdown(&mut self, deadline: Instant) -> Result<(), contract::ContractError>;
}
pub trait EvaluatorFactory: Send + Sync + 'static {
    fn profile(&self) -> EvaluatorProfile;
    fn create(
        &self,
        clock: ProcessClock,
        authority: SearchAuthority,
    ) -> Result<Box<dyn ManagedEvaluator>, contract::ContractError>;
    /// Derived from the actual encoder/input, never substituted with the rules key.
    fn input_key(
        &self,
        state: &RulesState,
        legal: &[contract::Move],
    ) -> Result<contract::EvalInputKey, contract::ContractError>;
    /// Passive evidence after the existing final backup guard committed. The
    /// observer grants no publication authority and cannot perform another backup.
    fn observe_search_acceptance(
        &self,
        _evaluation: &rz_search::contracts::AcceptedEvaluation,
        _traversed_edges: usize,
    ) -> Result<(), contract::ContractError> {
        Ok(())
    }
}

/// The injected evaluator and Rules owner registry share this process's clock
/// origin and ID allocators across all roots, games and physical worker drains.
pub struct EngineProcess {
    factory: Arc<dyn EvaluatorFactory>,
    owners: Arc<OwnerRegistry>,
    clock: ProcessClock,
    identity: EngineIdentity,
}
impl EngineProcess {
    pub fn new(
        factory: Arc<dyn EvaluatorFactory>,
        owners: Arc<OwnerRegistry>,
        clock: ProcessClock,
    ) -> Self {
        Self {
            factory,
            owners,
            clock,
            identity: EngineIdentity {
                name: "RoveZero CPU mock integration".into(),
                author: "RoveZero contributors".into(),
            },
        }
    }
    /// Session validates the declared identity before starting protocol service.
    pub fn with_identity(mut self, identity: EngineIdentity) -> Self {
        self.identity = identity;
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EngineSettings {
    pub parser: ParserLimits,
    pub position: PositionLimits,
    pub search: BuildSearchSettings,
    pub tree: TreeLimits,
    pub max_workers: usize,
    pub shutdown_limit: Duration,
}
impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            parser: ParserLimits {
                max_legal_moves: 512,
                ..ParserLimits::default()
            },
            position: PositionLimits::default(),
            search: BuildSearchSettings {
                time_config: rz_search::time::TimeBudgetConfig::default(),
                max_simulations: 128,
                untimed_limit: Duration::from_secs(30),
            },
            tree: TreeLimits {
                max_legal_moves: 512,
                max_nodes: 20_000,
                max_edges: 100_000,
                max_depth: 128,
                probability_tolerance: 1e-9,
            },
            max_workers: 2,
            shutdown_limit: Duration::from_secs(2),
        }
    }
}
/// Fixed-size receipts preserve rejected failures without retaining foreign
/// policy/input buffers. The current common failure has context/error/recovery;
/// it has no separate, unbounded external backend cause field.
#[derive(Clone, Debug)]
pub enum WorkerDiagnosticReceipt {
    Boundary(contract::ContractError),
    RejectedFailure {
        rejection: contract::ContractError,
        failure: Box<contract::EvalFailure>,
    },
    RejectedCompletion {
        rejection: contract::ContractError,
        kind: RejectedCompletionKind,
        context: Box<contract::CompletionContext>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectedCompletionKind {
    Completed,
    Canceled,
    Expired,
    Stale,
}

impl PartialEq for WorkerDiagnosticReceipt {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Boundary(left), Self::Boundary(right)) => left == right,
            (
                Self::RejectedFailure {
                    rejection: left,
                    failure: left_failure,
                },
                Self::RejectedFailure {
                    rejection: right,
                    failure: right_failure,
                },
            ) => {
                left == right
                    && left_failure.context == right_failure.context
                    && left_failure.error == right_failure.error
                    && left_failure.recovery == right_failure.recovery
            }
            (
                Self::RejectedCompletion {
                    rejection: left,
                    kind: left_kind,
                    context: left_context,
                },
                Self::RejectedCompletion {
                    rejection: right,
                    kind: right_kind,
                    context: right_context,
                },
            ) => left == right && left_kind == right_kind && left_context == right_context,
            _ => false,
        }
    }
}
impl Eq for WorkerDiagnosticReceipt {}
impl fmt::Display for WorkerDiagnosticReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boundary(error) => write!(formatter, "{error}"),
            Self::RejectedFailure { rejection, failure } => write!(
                formatter,
                "{rejection}; rejected failure: {}; recovery={:?}; context={:?}",
                failure.error, failure.recovery, failure.context,
            ),
            Self::RejectedCompletion {
                rejection,
                kind,
                context,
            } => write!(
                formatter,
                "{rejection}; rejected outcome={kind:?}; context={context:?}",
            ),
        }
    }
}

#[derive(Debug)]
struct DiagnosticRetentionFailure {
    error: contract::ContractError,
    receipt: WorkerDiagnosticReceipt,
}
impl fmt::Display for DiagnosticRetentionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}; original diagnostic: {}",
            self.error, self.receipt
        )
    }
}

/// Original typed failures outlive a closed owner event channel. The protocol
/// completion is a presentation; it cannot replace these source values.
#[derive(Clone, Debug)]
pub enum WorkerFailureSource {
    Factory(contract::ContractError),
    SearchConstructor(contract::ContractError),
    Authority(contract::ContractError),
    Search(Box<rz_search::contracts::ContractSearchFailure>),
    DiagnosticRetention {
        error: contract::ContractError,
        receipt: Box<WorkerDiagnosticReceipt>,
    },
}

#[derive(Clone, Debug)]
pub struct UndeliveredWorkerFailure {
    pub ticket: SearchTicket,
    pub completion: SearchCompletion,
    pub source: WorkerFailureSource,
}

enum WorkerCompletion {
    Completed {
        bestmove: Option<String>,
    },
    Failed {
        completion: SearchCompletion,
        source: WorkerFailureSource,
    },
}
impl WorkerCompletion {
    fn failed(source: WorkerFailureSource) -> Self {
        let completion = match &source {
            WorkerFailureSource::Factory(error)
            | WorkerFailureSource::SearchConstructor(error)
            | WorkerFailureSource::Authority(error) => failed_completion(*error),
            WorkerFailureSource::Search(error) => SearchCompletion::Failed {
                code: "SearchFailed".into(),
                message: format!("{error:?}"),
            },
            WorkerFailureSource::DiagnosticRetention { error, receipt } => {
                SearchCompletion::Failed {
                    code: "DiagnosticRetentionFailure".into(),
                    message: format!("{error}; original diagnostic: {receipt}"),
                }
            }
        };
        Self::Failed { completion, source }
    }
}

#[derive(Debug)]
pub enum EngineError {
    Contract(contract::ContractError),
    Session(SessionError),
    Transport(Box<ServeError<EngineError>>),
    WorkerPanic,
    DrainTimeout,
    UndeliveredDiagnostics(Vec<(WorkerDiagnosticReceipt, u64)>),
    UndeliveredWorkerFailure(Box<UndeliveredWorkerFailure>),
    Cleanup {
        primary: Box<EngineError>,
        cleanup: Vec<EngineError>,
    },
}
impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for EngineError {}
impl From<contract::ContractError> for EngineError {
    fn from(error: contract::ContractError) -> Self {
        Self::Contract(error)
    }
}
struct ActiveBinding {
    ticket: SearchTicket,
    control: SearchControl,
    authority: SearchAuthority,
}
type FailedPublication = Arc<Mutex<Option<Box<UndeliveredWorkerFailure>>>>;
struct Worker {
    handle: JoinHandle<Result<(), EngineError>>,
    failure: FailedPublication,
}
impl Worker {
    fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }
}
struct Owner {
    session_owner: crate::contracts::ContractSessionOwner,
    active: Option<ActiveBinding>,
    scope: Arc<Mutex<contract::AcceptanceScope>>,
    factory: Arc<dyn EvaluatorFactory>,
    clock: ProcessClock,
    ids: Arc<rz_search::contracts::IdAllocator>,
    settings: EngineSettings,
    workers: Vec<Worker>,
    pending_failures: VecDeque<(SearchTicket, SearchCompletion)>,
    diagnostics: Arc<Mutex<Vec<(WorkerDiagnosticReceipt, u64)>>>,
}
impl Owner {
    fn pending_failure(
        &mut self,
        sender: &SyncSender<Event>,
        ticket: SearchTicket,
        completion: SearchCompletion,
    ) {
        self.pending_failures.push_back((ticket, completion));
        // The owner keeps the receipt even if the bounded queue is full.
        let _ = sender.try_send(Event::RejectedInput {
            code: "OwnerWake",
            message: String::new(),
        });
    }
    fn handle(
        &mut self,
        session: &mut Session<RulesUciPort>,
        event: Event,
    ) -> SessionResult<RulesSearchPosition> {
        let mut out = SessionResult::default();
        let completion_receipt = match &event {
            Event::Complete {
                ticket,
                completion: completion @ SearchCompletion::Failed { .. },
            } => Some((ticket.clone(), completion.clone())),
            _ => None,
        };
        match self.diagnostics.lock() {
            Ok(mut receipts) => {
                for (error, count) in std::mem::take(&mut *receipts) {
                    out.diagnostics.push(Diagnostic {
                        code: "SearchDiagnostic",
                        message: format!("{error}; occurrences={count}"),
                    });
                }
            }
            Err(_) => {
                self.cancel();
                out.diagnostics.push(Diagnostic {
                    code: "DiagnosticFailure",
                    message: "worker diagnostic receipt owner poisoned".into(),
                });
                append_result(&mut out, session.end_of_input());
            }
        }
        while let Some((ticket, completion)) = self.pending_failures.pop_front() {
            append_contract_result(
                &mut out,
                self.session_owner
                    .handle_event(session, Event::Complete { ticket, completion }),
            );
        }
        let next = match event {
            Event::RejectedInput {
                code: "OwnerWake", ..
            } => SessionResult::default(),
            Event::Line(line) => {
                let next = self.session_owner.handle_line(session, &line, |snapshot| {
                    Ok(snapshot.state.snapshot().side_to_move())
                });
                let mut out = SessionResult::default();
                append_contract_result(&mut out, next);
                out
            }
            event => {
                let next = self.session_owner.handle_event(session, event);
                let mut out = SessionResult::default();
                append_contract_result(&mut out, next);
                out
            }
        };
        // Enqueueing is not consumption. A typed failed publication stays with
        // its bounded worker slot until this owner has handled the exact ticket.
        if let Some((ticket, completion)) = completion_receipt {
            let diagnosed = match &completion {
                SearchCompletion::Failed { code, message } => {
                    next.diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == "SearchFailed"
                            && diagnostic.message == format!("{code}: {message}")
                    })
                }
                SearchCompletion::Completed { .. } => false,
            };
            for worker in &self.workers {
                match worker.failure.lock() {
                    Ok(mut receipt) => {
                        if diagnosed
                            && receipt.as_ref().is_some_and(|failure| {
                                failure.ticket == ticket && failure.completion == completion
                            })
                        {
                            receipt.take();
                        }
                    }
                    Err(_) => {
                        self.cancel();
                        out.diagnostics.push(Diagnostic {
                            code: "DiagnosticFailure",
                            message: "worker failed publication owner poisoned".into(),
                        });
                        append_result(&mut out, session.end_of_input());
                    }
                }
            }
        }
        // B closes common/tree cancellation before changing its generations.
        match self.scope.lock() {
            Ok(mut scope) => *scope = self.session_owner.scope(),
            Err(_) => {
                self.cancel();
                out.diagnostics.push(Diagnostic {
                    code: "AuthorityFailure",
                    message: "acceptance authority poisoned".into(),
                });
                append_result(&mut out, session.end_of_input());
            }
        }
        append_result(&mut out, next);
        out
    }
    fn cancel(&self) {
        if let Some(active) = &self.active {
            active.control.cancel();
            active.authority.cancel();
        }
    }
    fn reap(&mut self) -> Result<(), EngineError> {
        self.reap_workers(false)
    }
    fn reap_workers(&mut self, closing: bool) -> Result<(), EngineError> {
        let mut index = 0;
        while index < self.workers.len() {
            if self.workers[index].is_finished() {
                if !closing
                    && self.workers[index]
                        .failure
                        .lock()
                        .map_err(|_| {
                            failure(
                                contract::ErrorCode::BackendFailure,
                                "worker failed publication owner poisoned",
                            )
                        })?
                        .is_some()
                {
                    // Keep the unacknowledged slot in max_workers. Starting more
                    // workers cannot grow an orphaned failed-publication history.
                    index += 1;
                    continue;
                }
                let worker = self.workers.swap_remove(index);
                let (unconsumed, poisoned) = take_failed_publication(&worker.failure);
                let joined = worker
                    .handle
                    .join()
                    .map_err(|_| EngineError::WorkerPanic)
                    .and_then(|result| result);
                let joined = match (unconsumed.as_deref(), joined) {
                    (Some(receipt), Err(error)) => {
                        discard_duplicate_delivery_failure(error, receipt).map_or(Ok(()), Err)
                    }
                    (_, result) => result,
                };
                let mut errors = Vec::new();
                if let Some(receipt) = unconsumed {
                    errors.push(EngineError::UndeliveredWorkerFailure(receipt));
                }
                if let Some(error) = poisoned {
                    errors.push(error);
                }
                if let Err(error) = joined {
                    errors.push(error);
                }
                finish_errors(None, errors)?;
            } else {
                index += 1;
            }
        }
        Ok(())
    }
    fn collect_unconsumed_failures(&self) -> Vec<EngineError> {
        let mut errors = Vec::new();
        for worker in &self.workers {
            let (receipt, poisoned) = take_failed_publication(&worker.failure);
            if let Some(receipt) = receipt {
                errors.push(EngineError::UndeliveredWorkerFailure(receipt));
            }
            if let Some(error) = poisoned {
                errors.push(error);
            }
        }
        errors
    }
    fn dispatch(
        &mut self,
        effect: Effect<RulesSearchPosition>,
        sender: &SyncSender<Event>,
    ) -> Result<(), EngineError> {
        match effect {
            Effect::Cancel { ticket, .. } => {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|active| active.ticket == ticket)
                {
                    self.cancel();
                }
            }
            Effect::NewGame | Effect::Shutdown => self.cancel(),
            Effect::Start {
                ticket, snapshot, ..
            } => {
                self.reap()?;
                if self.workers.len() >= self.settings.max_workers {
                    self.pending_failure(
                        sender,
                        ticket,
                        SearchCompletion::Failed {
                            code: "WorkerLimit".into(),
                            message: "previous physical worker has not drained".into(),
                        },
                    );
                    return Ok(());
                }
                let control = self.session_owner.control().cloned().ok_or_else(|| {
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "UCI start has no checked B control",
                    )
                })?;
                let token = self
                    .session_owner
                    .cancellation()
                    .ok_or_else(|| {
                        failure(
                            contract::ErrorCode::IdentityMismatch,
                            "UCI start has no common cancellation",
                        )
                    })?
                    .token();
                let deadlines = *self.session_owner.deadlines().ok_or_else(|| {
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "UCI start has no common deadlines",
                    )
                })?;
                let scope = self.session_owner.active_scope().ok_or_else(|| {
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "UCI start has no current scope",
                    )
                })?;
                let authority = SearchAuthority {
                    current: Arc::clone(&self.scope),
                    cancel: token.clone(),
                };
                self.active = Some(ActiveBinding {
                    ticket: ticket.clone(),
                    control: control.clone(),
                    authority: authority.clone(),
                });
                let profile = self.factory.profile();
                let config = rz_search::contracts::ContractSearchConfig {
                    scope,
                    model: Arc::clone(&profile.model),
                    precision: profile.precision,
                    compute: profile.compute,
                    bytes: profile.bytes,
                    policy_tolerance: 1e-5,
                    wdl_tolerance: 1e-5,
                    cancellation: token,
                    deadlines,
                    ids: Arc::clone(&self.ids),
                    max_simulations: control.max_simulations,
                    tree_limits: self.settings.tree,
                };
                let factory = Arc::clone(&self.factory);
                let clock = self.clock.clone();
                let shutdown_limit = self.settings.shutdown_limit;
                let events = sender.clone();
                let diagnostics = Arc::clone(&self.diagnostics);
                let publication = Arc::new(Mutex::new(None));
                let worker_publication = Arc::clone(&publication);
                let handle = thread::spawn(move || {
                    let mut runtime = match factory.create(clock.clone(), authority.clone()) {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            return finish_worker(
                                &events,
                                ticket,
                                WorkerCompletion::failed(WorkerFailureSource::Factory(error)),
                                &worker_publication,
                                None,
                                shutdown_limit,
                            );
                        }
                    };
                    let mut search =
                        match rz_search::contracts::ContractSearch::new(snapshot, config) {
                            Ok(search) => search,
                            Err(error) => {
                                return finish_worker(
                                    &events,
                                    ticket,
                                    WorkerCompletion::failed(
                                        WorkerFailureSource::SearchConstructor(error),
                                    ),
                                    &worker_publication,
                                    Some(runtime.as_mut()),
                                    shutdown_limit,
                                );
                            }
                        };
                    let mut previous_move = None;
                    let mut authority_error = None;
                    let mut diagnostic_failure = None;
                    while !search.is_finished() {
                        if control.cancellation.load(Ordering::Acquire) {
                            authority.cancel();
                        }
                        if let Err(error) = authority.current_scope() {
                            authority.cancel();
                            authority_error = Some(error);
                        }
                        let mut port = RuntimePort(runtime.as_mut());
                        let event = search.pump(
                            &mut port,
                            &clock,
                            || {
                                authority
                                    .current_scope()
                                    .unwrap_or(contract::AcceptanceScope {
                                        root: contract::RootGeneration(scope.root.0 ^ 1),
                                        ..scope
                                    })
                            },
                            |position, encoding| {
                                if encoding != profile.model.encoding() {
                                    return Err(failure(
                                        contract::ErrorCode::IdentityMismatch,
                                        "search changed encoder descriptor",
                                    ));
                                }
                                factory.input_key(
                                    position.state.rules(),
                                    position.state.legal_moves().moves(),
                                )
                            },
                        );
                        if let rz_search::contracts::ContractPumpEvent::Accepted {
                            evaluation: Some(evaluation),
                            traversed_edges,
                            ..
                        } = &event
                            && let Err(error) =
                                factory.observe_search_acceptance(evaluation, *traversed_edges)
                        {
                            // The already committed tree remains untouched. Preserve
                            // an observer failure and close future logical admission.
                            authority.cancel();
                            authority_error.get_or_insert(error);
                        }
                        if let Err(error) = consume_pump_diagnostic(&event, &diagnostics, &events) {
                            authority.cancel();
                            diagnostic_failure = Some(error);
                        }
                        let best = search.outcome().best_move;
                        if best != previous_move {
                            previous_move = best;
                            if let Some(bestmove) =
                                best.and_then(|movement| move_text(movement).ok())
                            {
                                let _ = events.try_send(Event::Progress {
                                    ticket: ticket.clone(),
                                    bestmove,
                                });
                            }
                        }
                        if matches!(event, rz_search::contracts::ContractPumpEvent::Waiting) {
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                    let outcome = search.outcome();
                    let completion = match (diagnostic_failure, authority_error, outcome.status) {
                        (Some(error), _, _) => {
                            WorkerCompletion::failed(WorkerFailureSource::DiagnosticRetention {
                                error: error.error,
                                receipt: Box::new(error.receipt),
                            })
                        }
                        (_, Some(error), _) => {
                            WorkerCompletion::failed(WorkerFailureSource::Authority(error))
                        }
                        (_, _, rz_search::contracts::ContractSearchStatus::Failed(error)) => {
                            WorkerCompletion::failed(WorkerFailureSource::Search(Box::new(error)))
                        }
                        _ => WorkerCompletion::Completed {
                            bestmove: outcome
                                .best_move
                                .and_then(|movement| move_text(movement).ok()),
                        },
                    };
                    finish_worker(
                        &events,
                        ticket,
                        completion,
                        &worker_publication,
                        Some(runtime.as_mut()),
                        shutdown_limit,
                    )
                });
                self.workers.push(Worker {
                    handle,
                    failure: publication,
                });
            }
            Effect::OptionChanged { .. } | Effect::Button { .. } => {
                return Err(failure(
                    contract::ErrorCode::UnsupportedContract,
                    "bootstrap did not register backend options",
                )
                .into());
            }
        }
        Ok(())
    }
}
fn finish_worker(
    sender: &SyncSender<Event>,
    ticket: SearchTicket,
    completion: WorkerCompletion,
    publication: &Mutex<Option<Box<UndeliveredWorkerFailure>>>,
    runtime: Option<&mut dyn ManagedEvaluator>,
    shutdown_limit: Duration,
) -> Result<(), EngineError> {
    let (completion, source) = match completion {
        WorkerCompletion::Completed { bestmove } => {
            (SearchCompletion::Completed { bestmove }, None)
        }
        WorkerCompletion::Failed { completion, source } => (completion, Some(source)),
    };
    let failed = source.is_some();
    let mut errors = Vec::new();
    if let Some(source) = source {
        let receipt = Box::new(UndeliveredWorkerFailure {
            ticket: ticket.clone(),
            completion: completion.clone(),
            source,
        });
        match publication.lock() {
            Ok(mut slot) if slot.is_none() => *slot = Some(receipt),
            Ok(_) => {
                errors.push(EngineError::UndeliveredWorkerFailure(receipt));
                errors.push(
                    failure(
                        contract::ErrorCode::ResourceExhausted,
                        "worker single failed-publication slot already occupied",
                    )
                    .into(),
                );
            }
            Err(_) => {
                errors.push(EngineError::UndeliveredWorkerFailure(receipt));
                errors.push(
                    failure(
                        contract::ErrorCode::BackendFailure,
                        "worker failed publication owner poisoned",
                    )
                    .into(),
                );
            }
        }
    }
    let delivery_failed =
        errors.is_empty() && sender.send(Event::Complete { ticket, completion }).is_err() && failed;
    // Keep the original owner-reclaimable until actual owner consumption/reap,
    // including the gap between returning this closure and handle.is_finished.
    let cleanup = runtime.and_then(|runtime| shutdown(runtime, shutdown_limit).err());
    if delivery_failed {
        // Failed sends are reclaimed by the join result. Successful sends keep
        // the slot until the owner acknowledges consumption or closing reaps it.
        let (receipt, poisoned) = copy_failed_publication(publication);
        if let Some(receipt) = receipt {
            errors.push(EngineError::UndeliveredWorkerFailure(receipt));
        }
        if let Some(error) = poisoned {
            errors.push(error);
        }
    }
    // Do not return early on failed delivery: physical shutdown always runs when
    // a runtime was created, and its failure remains separate from the source.
    if let Some(error) = cleanup {
        errors.push(EngineError::Contract(error));
    }
    finish_errors(None, errors)
}

fn take_failed_publication(
    publication: &Mutex<Option<Box<UndeliveredWorkerFailure>>>,
) -> (Option<Box<UndeliveredWorkerFailure>>, Option<EngineError>) {
    match publication.lock() {
        Ok(mut slot) => (slot.take(), None),
        Err(poisoned) => (
            poisoned.into_inner().take(),
            Some(
                failure(
                    contract::ErrorCode::BackendFailure,
                    "worker failed publication owner poisoned",
                )
                .into(),
            ),
        ),
    }
}

fn copy_failed_publication(
    publication: &Mutex<Option<Box<UndeliveredWorkerFailure>>>,
) -> (Option<Box<UndeliveredWorkerFailure>>, Option<EngineError>) {
    match publication.lock() {
        Ok(slot) => (slot.clone(), None),
        Err(poisoned) => (
            poisoned.into_inner().clone(),
            Some(
                failure(
                    contract::ErrorCode::BackendFailure,
                    "worker failed publication owner poisoned",
                )
                .into(),
            ),
        ),
    }
}

/// Failed send returns one bounded typed copy while its original stays visible
/// to the owner. Reaping acknowledges the original once and retains all cleanup
/// errors from the copy's join result without reporting the source twice.
fn discard_duplicate_delivery_failure(
    error: EngineError,
    receipt: &UndeliveredWorkerFailure,
) -> Option<EngineError> {
    match error {
        EngineError::UndeliveredWorkerFailure(copy)
            if copy.ticket == receipt.ticket && copy.completion == receipt.completion =>
        {
            None
        }
        EngineError::Cleanup { primary, cleanup } => {
            let primary = discard_duplicate_delivery_failure(*primary, receipt);
            let cleanup = cleanup
                .into_iter()
                .filter_map(|error| discard_duplicate_delivery_failure(error, receipt))
                .collect();
            finish_errors(primary, cleanup).err()
        }
        error => Some(error),
    }
}

struct RuntimePort<'a>(&'a mut dyn ManagedEvaluator);
impl contract::Evaluator<RulesState> for RuntimePort<'_> {
    fn submit(
        &mut self,
        request: Arc<contract::EvalRequest<RulesState>>,
    ) -> Result<(), contract::ContractError> {
        self.0.submit(request)
    }
    fn poll(&mut self) -> Option<contract::EvalResult> {
        self.0.poll()
    }
    fn cancel(&mut self, request: contract::RequestId) -> Result<(), contract::ContractError> {
        self.0.cancel(request)
    }
}
fn shutdown(
    runtime: &mut dyn ManagedEvaluator,
    limit: Duration,
) -> Result<(), contract::ContractError> {
    runtime.shutdown(Instant::now().checked_add(limit).ok_or_else(|| {
        failure(
            contract::ErrorCode::ResourceExhausted,
            "shutdown deadline overflow",
        )
    })?)
}
fn failed_completion(error: contract::ContractError) -> SearchCompletion {
    SearchCompletion::Failed {
        code: format!("{:?}/{:?}", error.stage, error.code),
        message: error.detail.into(),
    }
}
fn append_contract_result<S>(
    out: &mut SessionResult<S>,
    next: crate::contracts::ContractSessionOutcome<S>,
) {
    for error in next.errors {
        out.diagnostics.push(Diagnostic {
            code: "SharedContractError",
            message: error.to_string(),
        });
    }
    append_result(out, next.session);
}
fn append_result<S>(out: &mut SessionResult<S>, next: SessionResult<S>) {
    out.protocol.extend(next.protocol);
    out.diagnostics.extend(next.diagnostics);
    out.effects.extend(next.effects);
    out.accepted &= next.accepted;
}

fn consume_pump_diagnostic(
    event: &rz_search::contracts::ContractPumpEvent,
    receipts: &Mutex<Vec<(WorkerDiagnosticReceipt, u64)>>,
    sender: &SyncSender<Event>,
) -> Result<(), DiagnosticRetentionFailure> {
    use rz_search::contracts::ContractPumpEvent;
    let receipt = match event {
        ContractPumpEvent::Diagnostic(error) => WorkerDiagnosticReceipt::Boundary(*error),
        ContractPumpEvent::RejectedResult { error, result } => match result.as_ref() {
            contract::EvalResult::Failed(failed) => WorkerDiagnosticReceipt::RejectedFailure {
                rejection: *error,
                failure: Box::new(failed.clone()),
            },
            result => {
                let (kind, context) = match result {
                    contract::EvalResult::Completed(output) => (
                        RejectedCompletionKind::Completed,
                        contract::CompletionContext {
                            request: output.context,
                            execution: output.actual.execution,
                        },
                    ),
                    contract::EvalResult::Canceled(context) => {
                        (RejectedCompletionKind::Canceled, *context)
                    }
                    contract::EvalResult::Expired(context) => {
                        (RejectedCompletionKind::Expired, *context)
                    }
                    contract::EvalResult::Stale(context) => {
                        (RejectedCompletionKind::Stale, *context)
                    }
                    contract::EvalResult::Failed(_) => unreachable!("handled failed result above"),
                };
                WorkerDiagnosticReceipt::RejectedCompletion {
                    rejection: *error,
                    kind,
                    context: Box::new(context),
                }
            }
        },
        _ => return Ok(()),
    };
    let retained = retain_diagnostic(receipts, receipt);
    // The typed receipt stays owner-held even if the bounded queue is full.
    let _ = sender.try_send(Event::RejectedInput {
        code: "OwnerWake",
        message: String::new(),
    });
    retained
}

fn retain_diagnostic(
    receipts: &Mutex<Vec<(WorkerDiagnosticReceipt, u64)>>,
    receipt: WorkerDiagnosticReceipt,
) -> Result<(), DiagnosticRetentionFailure> {
    let mut receipts = match receipts.lock() {
        Ok(receipts) => receipts,
        Err(_) => {
            return Err(DiagnosticRetentionFailure {
                error: failure(
                    contract::ErrorCode::BackendFailure,
                    "worker diagnostic receipt owner poisoned",
                ),
                receipt,
            });
        }
    };
    if let Some((_, count)) = receipts
        .iter_mut()
        .find(|(existing, _)| *existing == receipt)
    {
        let Some(next) = count.checked_add(1) else {
            return Err(DiagnosticRetentionFailure {
                error: failure(
                    contract::ErrorCode::ResourceExhausted,
                    "worker diagnostic occurrence count exhausted",
                ),
                receipt,
            });
        };
        *count = next;
    } else if receipts.len() < 32 {
        receipts.push((receipt, 1));
    } else {
        // Overflow remains an explicit completion failure containing the original
        // receipt; it neither overwrites nor silently drops the 32 owned classes.
        return Err(DiagnosticRetentionFailure {
            error: failure(
                contract::ErrorCode::ResourceExhausted,
                "worker diagnostic receipt class limit exhausted",
            ),
            receipt,
        });
    }
    Ok(())
}
/// Independent input, timer, search and physical evaluator owners have bounded
/// event/worker counts. CPU/mock success does not assert NN/GPU support.
pub fn serve<O: Write, D: Write>(
    events: Receiver<Event>,
    sender: SyncSender<Event>,
    protocol: &mut O,
    diagnostics: &mut D,
    process: EngineProcess,
    settings: EngineSettings,
) -> Result<(), EngineError> {
    let EngineProcess {
        factory,
        owners,
        clock,
        identity,
    } = process;
    if settings.max_workers == 0
        || settings.shutdown_limit.is_zero()
        || settings.search.max_simulations > 128
    {
        return Err(failure(
            contract::ErrorCode::InvalidInput,
            "finite workers/drain and at most 128 simulations required",
        )
        .into());
    }
    let profile = factory.profile();
    let scope = contract::AcceptanceScope {
        game: contract::GameGeneration(1),
        root: contract::RootGeneration(0),
        model: profile.model.handle(),
        encoding: profile.model.encoding().handle,
        backend: profile.backend,
    };
    let session_owner = crate::contracts::ContractSessionOwner::new(
        scope,
        clock.epoch(),
        clock.b_clock(),
        settings.search,
    )?;
    let owner = Arc::new(Mutex::new(Owner {
        session_owner,
        active: None,
        scope: Arc::new(Mutex::new(scope)),
        factory,
        ids: Arc::clone(&clock.ids),
        clock,
        settings,
        workers: Vec::new(),
        pending_failures: VecDeque::new(),
        diagnostics: Arc::new(Mutex::new(Vec::new())),
    }));
    let mut session = Session::new(
        RulesUciPort::new(owners, settings.position),
        identity,
        Vec::new(),
        settings.parser,
    )
    .map_err(EngineError::Session)?;
    let timer_owner = Arc::clone(&owner);
    let timer_events = sender.clone();
    let timer_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let timer_flag = Arc::clone(&timer_stop);
    let timer = thread::spawn(move || {
        let mut delivered = None;
        while !timer_flag.load(Ordering::Acquire) {
            if let Ok(owner) = timer_owner.lock() {
                if let Some(active) = &owner.active {
                    if delivered.as_ref() != Some(&active.ticket)
                        && Instant::now() >= active.control.deadline
                        && timer_events
                            .try_send(Event::Deadline(active.ticket.clone()))
                            .is_ok()
                    {
                        delivered = Some(active.ticket.clone());
                    }
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
    });
    let handler_owner = Arc::clone(&owner);
    let dispatch_owner = Arc::clone(&owner);
    let result = serve_events_with_handler(
        &mut session,
        &events,
        protocol,
        diagnostics,
        move |session, event| {
            handler_owner
                .lock()
                .expect("UCI owner")
                .handle(session, event)
        },
        move |effect| {
            dispatch_owner
                .lock()
                .map_err(|_| failure(contract::ErrorCode::BackendFailure, "UCI owner poisoned"))?
                .dispatch(effect, &sender)
        },
    );
    timer_stop.store(true, Ordering::Release);
    drop(events);
    let primary = result
        .err()
        .map(|error| EngineError::Transport(Box::new(error)));
    let mut cleanup = Vec::new();
    if timer.join().is_err() {
        cleanup.push(EngineError::WorkerPanic);
    }
    if let Some(until) = Instant::now().checked_add(settings.shutdown_limit) {
        loop {
            let mut owner = match owner.lock() {
                Ok(owner) => owner,
                Err(_) => {
                    cleanup.push(
                        failure(contract::ErrorCode::BackendFailure, "UCI owner poisoned").into(),
                    );
                    break;
                }
            };
            owner.cancel();
            if let Err(error) = owner.reap_workers(true) {
                cleanup.push(error);
            }
            if owner.workers.is_empty() {
                break;
            }
            if Instant::now() >= until {
                cleanup.push(EngineError::DrainTimeout);
                break;
            }
            drop(owner);
            thread::sleep(Duration::from_millis(1));
        }
    } else {
        cleanup.push(
            failure(
                contract::ErrorCode::ResourceExhausted,
                "shutdown deadline overflow",
            )
            .into(),
        );
    }
    match owner.lock() {
        Ok(owner) => {
            // A timed-out/panicking drain does not discard failures that were
            // already published by workers which have not yet finished joining.
            cleanup.extend(owner.collect_unconsumed_failures());
            match owner.diagnostics.lock() {
                Ok(mut receipts) if !receipts.is_empty() => cleanup.push(
                    EngineError::UndeliveredDiagnostics(std::mem::take(&mut *receipts)),
                ),
                Err(_) => cleanup.push(
                    failure(
                        contract::ErrorCode::BackendFailure,
                        "worker diagnostic receipt owner poisoned",
                    )
                    .into(),
                ),
                _ => {}
            }
        }
        Err(_) => {
            cleanup.push(failure(contract::ErrorCode::BackendFailure, "UCI owner poisoned").into())
        }
    }
    finish_errors(primary, cleanup)
}
fn finish_errors(
    primary: Option<EngineError>,
    mut cleanup: Vec<EngineError>,
) -> Result<(), EngineError> {
    let primary = match primary {
        Some(primary) => primary,
        None if cleanup.is_empty() => return Ok(()),
        None => cleanup.remove(0),
    };
    if cleanup.is_empty() {
        Err(primary)
    } else {
        Err(EngineError::Cleanup {
            primary: Box::new(primary),
            cleanup,
        })
    }
}
pub fn event_channel() -> (SyncSender<Event>, Receiver<Event>) {
    mpsc::sync_channel(128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    enum FailureMode {
        Factory,
        Matching,
        ForeignFlood,
        NoEvaluation,
        PanicDrain,
    }
    struct FailureFactory {
        profile: EvaluatorProfile,
        mode: FailureMode,
        drained: Arc<AtomicU64>,
        cleanup: Option<contract::ContractError>,
    }
    struct FailureRuntime {
        mode: FailureMode,
        context: Option<contract::EvalContext>,
        callbacks: u64,
        drained: Arc<AtomicU64>,
        cleanup: Option<contract::ContractError>,
    }
    fn injected_backend_failure() -> contract::ContractError {
        contract::ContractError::new(
            contract::ErrorCode::BackendFailure,
            contract::Stage::Backend,
            "worker physical failure before owner consumption",
        )
    }
    impl EvaluatorFactory for FailureFactory {
        fn profile(&self) -> EvaluatorProfile {
            self.profile.clone()
        }
        fn create(
            &self,
            _: ProcessClock,
            authority: SearchAuthority,
        ) -> Result<Box<dyn ManagedEvaluator>, contract::ContractError> {
            if matches!(self.mode, FailureMode::Factory) {
                return Err(injected_backend_failure());
            }
            if matches!(self.mode, FailureMode::NoEvaluation) {
                authority.cancel();
            }
            Ok(Box::new(FailureRuntime {
                mode: self.mode,
                context: None,
                callbacks: 0,
                drained: Arc::clone(&self.drained),
                cleanup: self.cleanup,
            }))
        }
        fn input_key(
            &self,
            _: &RulesState,
            _: &[contract::Move],
        ) -> Result<contract::EvalInputKey, contract::ContractError> {
            Ok(contract::EvalInputKey(contract::Digest([9; 32])))
        }
    }
    impl contract::Evaluator<RulesState> for FailureRuntime {
        fn submit(
            &mut self,
            request: Arc<contract::EvalRequest<RulesState>>,
        ) -> Result<(), contract::ContractError> {
            self.context = Some(request.context());
            Ok(())
        }
        fn poll(&mut self) -> Option<contract::EvalResult> {
            let mut context = self.context?;
            self.callbacks += 1;
            if matches!(self.mode, FailureMode::ForeignFlood) {
                context.request.sequence += 100 + self.callbacks;
            } else {
                self.context = None;
            }
            Some(contract::EvalResult::Failed(contract::EvalFailure {
                context: contract::CompletionContext {
                    request: context,
                    execution: Some(contract::ExecutionId::new(
                        context.request.epoch,
                        self.callbacks,
                    )),
                },
                error: injected_backend_failure(),
                recovery: contract::RecoveryOutcome::Failed,
            }))
        }
        fn cancel(&mut self, _: contract::RequestId) -> Result<(), contract::ContractError> {
            Ok(())
        }
    }
    impl ManagedEvaluator for FailureRuntime {
        fn shutdown(&mut self, _: Instant) -> Result<(), contract::ContractError> {
            self.drained.fetch_add(1, Ordering::AcqRel);
            if matches!(self.mode, FailureMode::PanicDrain) {
                panic!("injected drain panic after publication");
            }
            self.cleanup.map_or(Ok(()), Err)
        }
    }

    struct UnusedFactory(EvaluatorProfile);
    impl EvaluatorFactory for UnusedFactory {
        fn profile(&self) -> EvaluatorProfile {
            self.0.clone()
        }
        fn create(
            &self,
            _: ProcessClock,
            _: SearchAuthority,
        ) -> Result<Box<dyn ManagedEvaluator>, contract::ContractError> {
            panic!("these owner refusal tests must not create a runtime")
        }
        fn input_key(
            &self,
            _: &RulesState,
            _: &[contract::Move],
        ) -> Result<contract::EvalInputKey, contract::ContractError> {
            panic!("these owner refusal tests must not encode")
        }
    }
    fn fixture() -> (Owner, Session<RulesUciPort>) {
        let owners = Arc::new(OwnerRegistry::default());
        let encoding = contract::EncodingHandle {
            owner: owners.allocate().unwrap(),
            slot: 0,
            generation: contract::SlotGeneration(1),
            manifest: contract::Digest([1; 32]),
        };
        let model = contract::ModelHandle {
            owner: owners.allocate().unwrap(),
            slot: 0,
            generation: contract::SlotGeneration(1),
            manifest: contract::Digest([2; 32]),
        };
        let descriptor = contract::ModelDescriptor::try_new(
            model,
            contract::EncodingDescriptor {
                handle: encoding,
                history_length: 1,
                action_map: contract::Digest([3; 32]),
                history_policy: contract::Digest([4; 32]),
            },
            vec![contract::PrecisionProfile::Fp32],
            1,
            1,
        )
        .unwrap();
        let profile = EvaluatorProfile {
            model: Arc::new(descriptor),
            precision: contract::PrecisionProfile::Fp32,
            backend: contract::Digest([5; 32]),
            compute: contract::ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            bytes: contract::ByteBudget {
                host: 1,
                device: 0,
                pinned: 0,
            },
        };
        let clock = ProcessClock::new(contract::ProcessEpoch(7));
        let scope = contract::AcceptanceScope {
            game: contract::GameGeneration(1),
            root: contract::RootGeneration(0),
            model,
            encoding,
            backend: profile.backend,
        };
        let settings = EngineSettings {
            max_workers: 1,
            ..EngineSettings::default()
        };
        let owner = Owner {
            session_owner: crate::contracts::ContractSessionOwner::new(
                scope,
                clock.epoch(),
                clock.b_clock(),
                settings.search,
            )
            .unwrap(),
            active: None,
            scope: Arc::new(Mutex::new(scope)),
            factory: Arc::new(UnusedFactory(profile)),
            clock,
            ids: Arc::new(rz_search::contracts::IdAllocator::new(
                contract::ProcessEpoch(7),
            )),
            settings,
            workers: Vec::new(),
            pending_failures: VecDeque::new(),
            diagnostics: Arc::new(Mutex::new(Vec::new())),
        };
        let session = Session::new(
            RulesUciPort::new(owners, settings.position),
            EngineIdentity {
                name: "actual Rules refusal fixture".into(),
                author: "test".into(),
            },
            Vec::new(),
            settings.parser,
        )
        .unwrap();
        (owner, session)
    }

    #[test]
    fn full_event_queue_cannot_lose_worker_refusal_or_checked_fallback() {
        let (mut owner, mut session) = fixture();
        let (release, waiting) = mpsc::channel();
        owner.workers.push(Worker {
            handle: thread::spawn(move || {
                waiting.recv().unwrap();
                Ok(())
            }),
            failure: Arc::new(Mutex::new(None)),
        });
        let out = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, events) = mpsc::sync_channel(1);
        sender.send(Event::Line("isready".into())).unwrap();
        for effect in out.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        assert_eq!(owner.pending_failures.len(), 1);
        let out = owner.handle(&mut session, events.recv().unwrap());
        assert_eq!(
            out.protocol
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1
        );
        assert!(out.protocol.iter().any(|line| line == "readyok"));
        assert!(
            out.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("WorkerLimit"))
        );
        assert!(owner.pending_failures.is_empty());
        assert!(session.active_ticket().is_none());
        release.send(()).unwrap();
        owner.workers.pop().unwrap().handle.join().unwrap().unwrap();
    }

    #[test]
    fn stale_provider_failure_keeps_source_without_emitting_old_bestmove() {
        let (mut owner, mut session) = fixture();
        owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let old = session.active_ticket().unwrap();
        assert!(
            owner
                .handle(
                    &mut session,
                    Event::Line("position startpos moves e2e4".into())
                )
                .accepted
        );
        let out = owner.handle(
            &mut session,
            Event::Complete {
                ticket: old,
                completion: SearchCompletion::Failed {
                    code: "InjectedDeviceFailure".into(),
                    message: "physical callback after root replacement".into(),
                },
            },
        );
        assert!(out.protocol.is_empty());
        assert!(out.diagnostics.iter().any(|diagnostic| {
            diagnostic.message.contains("InjectedDeviceFailure")
                && diagnostic.message.contains("physical callback")
        }));
        assert_eq!(
            session.snapshot().state().snapshot().side_to_move(),
            contract::Color::Black
        );
    }

    #[test]
    fn transport_primary_and_source_diagnostics_survive_physical_drain_failure() {
        let source = Diagnostic {
            code: "SearchFailed",
            message: "original physical backend failure".into(),
        };
        let primary = EngineError::Transport(Box::new(ServeError {
            failure: ServeFailure::Io(std::io::Error::from(std::io::ErrorKind::BrokenPipe)),
            cleanup_failures: Vec::new(),
            diagnostics: vec![source.clone()],
        }));
        let error = finish_errors(Some(primary), vec![EngineError::DrainTimeout]).unwrap_err();
        let EngineError::Cleanup { primary, cleanup } = error else {
            panic!("compound cleanup required")
        };
        assert!(matches!(cleanup.as_slice(), [EngineError::DrainTimeout]));
        let EngineError::Transport(transport) = *primary else {
            panic!("transport primary required")
        };
        assert_eq!(transport.diagnostics, [source]);
        assert!(
            matches!(transport.failure, ServeFailure::Io(error) if error.kind() == std::io::ErrorKind::BrokenPipe)
        );
    }

    #[test]
    fn diagnostic_receipts_survive_full_wakeup_queue_and_preserve_repeated_causes() {
        let (mut owner, mut session) = fixture();
        let (sender, events) = mpsc::sync_channel(1);
        sender.send(Event::Line("isready".into())).unwrap();
        let cause = contract::ContractError::new(
            contract::ErrorCode::IdentityMismatch,
            contract::Stage::Output,
            "foreign response context",
        );
        retain_diagnostic(&owner.diagnostics, WorkerDiagnosticReceipt::Boundary(cause)).unwrap();
        retain_diagnostic(&owner.diagnostics, WorkerDiagnosticReceipt::Boundary(cause)).unwrap();
        assert!(
            sender
                .try_send(Event::RejectedInput {
                    code: "OwnerWake",
                    message: String::new()
                })
                .is_err()
        );
        let out = owner.handle(&mut session, events.recv().unwrap());
        assert_eq!(out.protocol, ["readyok"]);
        assert!(
            out.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "SearchDiagnostic"
                    && diagnostic.message.contains("Output/IdentityMismatch")
                    && diagnostic.message.contains("foreign response context")
                    && diagnostic.message.contains("occurrences=2"))
        );
        assert!(owner.diagnostics.lock().unwrap().is_empty());
    }

    fn foreign_failure_event(
        owner: &mut Owner,
        session: &mut Session<RulesUciPort>,
    ) -> rz_search::contracts::ContractPumpEvent {
        use rz_search::contracts::{ContractPumpEvent, ContractSearch, ContractSearchConfig};
        struct ForeignRuntime(Option<contract::EvalContext>);
        impl contract::Evaluator<RulesState> for ForeignRuntime {
            fn submit(
                &mut self,
                request: Arc<contract::EvalRequest<RulesState>>,
            ) -> Result<(), contract::ContractError> {
                assert!(self.0.replace(request.context()).is_none());
                Ok(())
            }
            fn poll(&mut self) -> Option<contract::EvalResult> {
                let mut context = self.0.take()?;
                context.request.sequence += 100;
                Some(contract::EvalResult::Failed(contract::EvalFailure {
                    context: contract::CompletionContext {
                        request: context,
                        execution: Some(contract::ExecutionId::new(context.request.epoch, 51)),
                    },
                    error: contract::ContractError::new(
                        contract::ErrorCode::BackendFailure,
                        contract::Stage::Backend,
                        "foreign physical callback failed",
                    ),
                    recovery: contract::RecoveryOutcome::Failed,
                }))
            }
            fn cancel(&mut self, _: contract::RequestId) -> Result<(), contract::ContractError> {
                Ok(())
            }
        }
        let start = owner.handle(session, Event::Line("go nodes 1".into()));
        let snapshot = start
            .effects
            .into_iter()
            .find_map(|effect| match effect {
                Effect::Start { snapshot, .. } => Some(snapshot),
                _ => None,
            })
            .unwrap();
        let profile = owner.factory.profile();
        let scope = owner.session_owner.scope();
        let config = ContractSearchConfig {
            scope,
            model: profile.model,
            precision: profile.precision,
            compute: profile.compute,
            bytes: profile.bytes,
            policy_tolerance: 1e-5,
            wdl_tolerance: 1e-5,
            cancellation: owner.session_owner.cancellation().unwrap().token(),
            deadlines: *owner.session_owner.deadlines().unwrap(),
            ids: Arc::clone(&owner.ids),
            max_simulations: 1,
            tree_limits: owner.settings.tree,
        };
        let mut search = ContractSearch::new(snapshot, config).unwrap();
        let mut runtime = ForeignRuntime(None);
        let pump = |search: &mut ContractSearch<RulesSearchPosition>,
                    runtime: &mut ForeignRuntime| {
            search.pump(
                runtime,
                &owner.clock,
                || scope,
                |_, _| Ok(contract::EvalInputKey(contract::Digest([9; 32]))),
            )
        };
        assert!(matches!(
            pump(&mut search, &mut runtime),
            ContractPumpEvent::Submitted { .. }
        ));
        let event = pump(&mut search, &mut runtime);
        assert!(
            matches!(&event, ContractPumpEvent::RejectedResult { result, .. } if matches!(result.as_ref(), contract::EvalResult::Failed(_)))
        );
        assert!(!search.is_finished());
        assert_eq!(search.outcome().counters.completed_visits, 0);
        event
    }

    #[test]
    fn actual_rules_search_rejection_keeps_original_failure_across_full_queue_and_new_root() {
        let (mut owner, mut session) = fixture();
        let event = foreign_failure_event(&mut owner, &mut session);
        // Replacement precedes delivery: a prior worker may only report causes.
        assert!(
            owner
                .handle(
                    &mut session,
                    Event::Line("position startpos moves e2e4".into())
                )
                .accepted
        );
        let (sender, events) = mpsc::sync_channel(1);
        sender.send(Event::Line("isready".into())).unwrap();
        consume_pump_diagnostic(&event, &owner.diagnostics, &sender).unwrap();
        consume_pump_diagnostic(&event, &owner.diagnostics, &sender).unwrap();
        let mut recovered = event.clone();
        let rz_search::contracts::ContractPumpEvent::RejectedResult { result, .. } = &mut recovered
        else {
            panic!("rejected result required")
        };
        let contract::EvalResult::Failed(failure) = result.as_mut() else {
            panic!("original failure required")
        };
        failure.recovery = contract::RecoveryOutcome::Completed;
        consume_pump_diagnostic(&recovered, &owner.diagnostics, &sender).unwrap();
        {
            let receipts = owner.diagnostics.lock().unwrap();
            assert_eq!(receipts.len(), 2);
            assert_eq!(receipts[0].1, 2);
            assert_eq!(receipts[1].1, 1);
            let WorkerDiagnosticReceipt::RejectedFailure { rejection, failure } = &receipts[0].0
            else {
                panic!("original failure required")
            };
            assert_eq!(rejection.code, contract::ErrorCode::IdentityMismatch);
            assert_eq!(failure.error.code, contract::ErrorCode::BackendFailure);
            assert_eq!(failure.error.detail, "foreign physical callback failed");
            assert_eq!(failure.recovery, contract::RecoveryOutcome::Failed);
            assert_eq!(failure.context.execution.unwrap().sequence, 51);
        }
        let out = owner.handle(&mut session, events.recv().unwrap());
        assert_eq!(out.protocol, ["readyok"]);
        assert!(out.diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message
                .contains("foreign physical callback failed")
                && diagnostic.message.contains("recovery=Failed")
                && diagnostic.message.contains("occurrences=2")
        }));
        assert!(session.active_ticket().is_none());
        assert_eq!(
            session.snapshot().state().snapshot().side_to_move(),
            contract::Color::Black
        );
    }

    #[test]
    fn rejection_receipt_limit_retains_all_owned_classes_and_overflow_source() {
        let (mut owner, mut session) = fixture();
        let event = foreign_failure_event(&mut owner, &mut session);
        let rz_search::contracts::ContractPumpEvent::RejectedResult { error, result } = event
        else {
            panic!("rejected result required")
        };
        let contract::EvalResult::Failed(original) = *result else {
            panic!("original failure required")
        };
        for sequence in 1..=32 {
            let mut failure = original.clone();
            failure.context.request.request.sequence = sequence;
            retain_diagnostic(
                &owner.diagnostics,
                WorkerDiagnosticReceipt::RejectedFailure {
                    rejection: error,
                    failure: Box::new(failure),
                },
            )
            .unwrap();
        }
        let mut failure = original;
        failure.context.request.request.sequence = 33;
        failure.recovery = contract::RecoveryOutcome::Completed;
        let overflow = retain_diagnostic(
            &owner.diagnostics,
            WorkerDiagnosticReceipt::RejectedFailure {
                rejection: error,
                failure: Box::new(failure),
            },
        )
        .unwrap_err();
        assert_eq!(overflow.error.code, contract::ErrorCode::ResourceExhausted);
        let WorkerDiagnosticReceipt::RejectedFailure { failure, .. } = &overflow.receipt else {
            panic!("overflow cause required")
        };
        assert_eq!(failure.context.request.request.sequence, 33);
        assert_eq!(failure.recovery, contract::RecoveryOutcome::Completed);
        assert_eq!(failure.error.detail, "foreign physical callback failed");
        assert!(overflow.to_string().contains("recovery=Completed"));
        let receipts = owner.diagnostics.lock().unwrap();
        assert_eq!(receipts.len(), 32);
        assert!(receipts.iter().all(|(_, count)| *count == 1));
    }

    fn closed_receiver_worker(
        mode: FailureMode,
        invalid_constructor: bool,
        cleanup: Option<contract::ContractError>,
    ) -> (Result<(), EngineError>, u64, usize) {
        let (mut owner, mut session) = fixture();
        let drained = Arc::new(AtomicU64::new(0));
        owner.factory = Arc::new(FailureFactory {
            profile: owner.factory.profile(),
            mode,
            drained: Arc::clone(&drained),
            cleanup,
        });
        if invalid_constructor {
            owner.settings.tree.max_nodes = 0;
        }
        let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        assert_eq!(owner.workers.len(), 1);
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until, "worker must finitely terminate");
            thread::sleep(Duration::from_millis(1));
        }
        let result = owner.reap_workers(true);
        assert!(owner.workers.is_empty());
        let receipts = owner.diagnostics.lock().unwrap().len();
        (result, drained.load(Ordering::Acquire), receipts)
    }

    #[test]
    fn closed_receiver_preserves_factory_source_in_actual_worker_join() {
        let (result, drained, _) = closed_receiver_worker(FailureMode::Factory, false, None);
        let EngineError::UndeliveredWorkerFailure(failure) = result.unwrap_err() else {
            panic!("undelivered factory failure required")
        };
        assert!(
            matches!(&failure.completion, SearchCompletion::Failed { message, .. } if message == injected_backend_failure().detail)
        );
        let WorkerFailureSource::Factory(source) = failure.source else {
            panic!("typed factory source required")
        };
        assert_eq!(source, injected_backend_failure());
        assert_eq!(drained, 0, "failed factory created no runtime");
    }

    #[test]
    fn closed_receiver_search_constructor_failure_still_drains_and_keeps_cleanup() {
        let cleanup_cause = failure(
            contract::ErrorCode::BackendFailure,
            "physical drain failed after failed delivery",
        );
        let (result, drained, _) =
            closed_receiver_worker(FailureMode::Matching, true, Some(cleanup_cause));
        let EngineError::Cleanup { primary, cleanup } = result.unwrap_err() else {
            panic!("delivery and drain failures required")
        };
        assert!(
            matches!(cleanup.as_slice(), [EngineError::Contract(error)] if *error == cleanup_cause)
        );
        let EngineError::UndeliveredWorkerFailure(failure) = *primary else {
            panic!("undelivered constructor source required")
        };
        assert!(matches!(
            failure.source,
            WorkerFailureSource::SearchConstructor(_)
        ));
        assert!(matches!(
            failure.completion,
            SearchCompletion::Failed { .. }
        ));
        assert_eq!(drained, 1);
    }

    #[test]
    fn closed_receiver_matching_eval_failure_preserves_typed_context_and_recovery() {
        let (result, drained, _) = closed_receiver_worker(FailureMode::Matching, false, None);
        let EngineError::UndeliveredWorkerFailure(failure) = result.unwrap_err() else {
            panic!("undelivered search failure required")
        };
        let WorkerFailureSource::Search(source) = failure.source else {
            panic!("typed search source required")
        };
        let rz_search::contracts::ContractSearchFailure::Evaluation(evaluation) = *source else {
            panic!("original evaluator failure required")
        };
        assert_eq!(evaluation.error, injected_backend_failure());
        assert_eq!(evaluation.recovery, contract::RecoveryOutcome::Failed);
        assert_eq!(evaluation.context.execution.unwrap().sequence, 1);
        assert_eq!(
            evaluation.context.request.request.epoch,
            contract::ProcessEpoch(7)
        );
        assert_eq!(drained, 1);
    }

    #[test]
    fn closed_receiver_cap33_failure_keeps_original_receipt_and_drain_failure() {
        let cleanup_cause = failure(
            contract::ErrorCode::BackendFailure,
            "physical drain failed after failed delivery",
        );
        let (result, drained, receipts) =
            closed_receiver_worker(FailureMode::ForeignFlood, false, Some(cleanup_cause));
        let EngineError::Cleanup { primary, cleanup } = result.unwrap_err() else {
            panic!("overflow and drain failures required")
        };
        assert!(
            matches!(cleanup.as_slice(), [EngineError::Contract(error)] if *error == cleanup_cause)
        );
        let EngineError::UndeliveredWorkerFailure(failure) = *primary else {
            panic!("undelivered overflow source required")
        };
        let WorkerFailureSource::DiagnosticRetention { error, receipt } = failure.source else {
            panic!("typed overflow receipt required")
        };
        assert_eq!(error.code, contract::ErrorCode::ResourceExhausted);
        let WorkerDiagnosticReceipt::RejectedFailure { failure, .. } = *receipt else {
            panic!("original cap33 evaluator failure required")
        };
        assert_eq!(failure.error, injected_backend_failure());
        assert_eq!(failure.recovery, contract::RecoveryOutcome::Failed);
        assert_eq!(failure.context.execution.unwrap().sequence, 33);
        assert_eq!(
            failure.context.request.request.epoch,
            contract::ProcessEpoch(7)
        );
        assert_eq!(receipts, 32);
        assert_eq!(drained, 1);
    }

    #[test]
    fn closed_receiver_normal_cancel_completion_drains_without_delivery_error() {
        let (result, drained, _) = closed_receiver_worker(FailureMode::NoEvaluation, false, None);
        result.unwrap();
        assert_eq!(drained, 1);
    }

    #[test]
    fn queued_quit_before_successful_failed_send_keeps_unconsumed_typed_source() {
        let (mut owner, mut session) = fixture();
        owner.factory = Arc::new(FailureFactory {
            profile: owner.factory.profile(),
            mode: FailureMode::Factory,
            drained: Arc::new(AtomicU64::new(0)),
            cleanup: None,
        });
        let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, receiver) = mpsc::sync_channel(2);
        sender.send(Event::Line("quit".into())).unwrap();
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        let failure_ticket = owner.workers[0]
            .failure
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .ticket
            .clone();
        // A completed worker whose failed publication is still queued continues
        // occupying its existing worker slot until actual owner consumption.
        owner.reap().unwrap();
        assert_eq!(owner.workers.len(), owner.settings.max_workers);
        let out = owner.handle(&mut session, receiver.recv().unwrap());
        assert!(out.protocol.is_empty());
        drop(receiver); // The successful second send was never consumed.
        let error = owner.reap_workers(true).unwrap_err();
        let EngineError::UndeliveredWorkerFailure(failure) = error else {
            panic!("unconsumed successful publication required")
        };
        assert_eq!(failure.ticket, failure_ticket);
        assert!(matches!(
            failure.completion,
            SearchCompletion::Failed { .. }
        ));
        let WorkerFailureSource::Factory(source) = failure.source else {
            panic!("typed queued factory source required")
        };
        assert_eq!(source, injected_backend_failure());
        assert!(owner.workers.is_empty());
    }

    #[test]
    fn consumed_failed_publication_acknowledges_slot_and_allows_next_root() {
        let (mut owner, mut session) = fixture();
        owner.factory = Arc::new(FailureFactory {
            profile: owner.factory.profile(),
            mode: FailureMode::Factory,
            drained: Arc::new(AtomicU64::new(0)),
            cleanup: None,
        });
        let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, receiver) = mpsc::sync_channel(1);
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        let out = owner.handle(&mut session, receiver.recv().unwrap());
        assert_eq!(
            out.protocol
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1
        );
        assert!(out.diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message
                .contains(injected_backend_failure().detail)
        }));
        assert!(owner.workers[0].failure.lock().unwrap().is_none());
        owner.reap().unwrap();
        assert!(owner.workers.is_empty());
        let next = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        assert!(next.accepted);
        assert!(
            next.effects
                .iter()
                .any(|effect| matches!(effect, Effect::Start { .. }))
        );
    }

    #[test]
    fn closed_receiver_panicking_drain_keeps_source_and_worker_panic() {
        let (result, drained, _) = closed_receiver_worker(FailureMode::PanicDrain, false, None);
        let EngineError::Cleanup { primary, cleanup } = result.unwrap_err() else {
            panic!("source and panic required")
        };
        assert!(matches!(cleanup.as_slice(), [EngineError::WorkerPanic]));
        let EngineError::UndeliveredWorkerFailure(failure) = *primary else {
            panic!("source must survive drain unwind")
        };
        let WorkerFailureSource::Search(source) = failure.source else {
            panic!("typed source required")
        };
        let rz_search::contracts::ContractSearchFailure::Evaluation(evaluation) = *source else {
            panic!("original evaluator failure required")
        };
        assert_eq!(evaluation.error, injected_backend_failure());
        assert_eq!(evaluation.recovery, contract::RecoveryOutcome::Failed);
        assert_eq!(drained, 1);
    }

    #[test]
    fn early_rejected_failure_after_diagnostic_poison_does_not_ack_original() {
        let (mut owner, mut session) = fixture();
        owner.factory = Arc::new(FailureFactory {
            profile: owner.factory.profile(),
            mode: FailureMode::Factory,
            drained: Arc::new(AtomicU64::new(0)),
            cleanup: None,
        });
        let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, receiver) = mpsc::sync_channel(1);
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        let diagnostics = Arc::clone(&owner.diagnostics);
        assert!(
            thread::spawn(move || {
                let _guard = diagnostics.lock().unwrap();
                panic!("injected diagnostic owner poison");
            })
            .join()
            .is_err()
        );
        let out = owner.handle(&mut session, receiver.recv().unwrap());
        assert!(
            !out.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "SearchFailed")
        );
        assert!(owner.workers[0].failure.lock().unwrap().is_some());
        let EngineError::UndeliveredWorkerFailure(failure) = owner.reap_workers(true).unwrap_err()
        else {
            panic!("unconsumed source required")
        };
        assert!(
            matches!(failure.source, WorkerFailureSource::Factory(error) if error == injected_backend_failure())
        );
    }

    #[test]
    fn unfinished_drain_keeps_published_source_reclaimable_by_closing_owner() {
        struct BlockingDrain {
            entered: SyncSender<()>,
            release: Receiver<()>,
        }
        impl contract::Evaluator<RulesState> for BlockingDrain {
            fn submit(
                &mut self,
                _: Arc<contract::EvalRequest<RulesState>>,
            ) -> Result<(), contract::ContractError> {
                unreachable!("only shutdown is exercised")
            }
            fn poll(&mut self) -> Option<contract::EvalResult> {
                None
            }
            fn cancel(&mut self, _: contract::RequestId) -> Result<(), contract::ContractError> {
                Ok(())
            }
        }
        impl ManagedEvaluator for BlockingDrain {
            fn shutdown(&mut self, _: Instant) -> Result<(), contract::ContractError> {
                self.entered.send(()).unwrap();
                self.release.recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(())
            }
        }
        struct ReleaseOnDrop(Option<SyncSender<()>>);
        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                if let Some(sender) = self.0.take() {
                    let _ = sender.try_send(());
                }
            }
        }
        let (mut owner, mut session) = fixture();
        owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let ticket = session.active_ticket().unwrap();
        let expected_ticket = ticket.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        let (entered, waiting) = mpsc::sync_channel(1);
        let (release, gate) = mpsc::sync_channel(1);
        let mut release = ReleaseOnDrop(Some(release));
        let publication = Arc::new(Mutex::new(None));
        let worker_publication = Arc::clone(&publication);
        let handle = thread::spawn(move || {
            let mut runtime = BlockingDrain {
                entered,
                release: gate,
            };
            finish_worker(
                &sender,
                ticket,
                WorkerCompletion::failed(WorkerFailureSource::Factory(injected_backend_failure())),
                &worker_publication,
                Some(&mut runtime),
                Duration::from_secs(5),
            )
        });
        owner.workers.push(Worker {
            handle,
            failure: publication,
        });
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!owner.workers[0].is_finished());
        owner.reap_workers(true).unwrap();
        assert_eq!(owner.workers.len(), 1);
        // The final owner collection also runs when its drain deadline expires.
        // A reclaimed failure does not mean the physical worker has completed.
        let mut recovered = owner.collect_unconsumed_failures();
        assert_eq!(recovered.len(), 1);
        let EngineError::UndeliveredWorkerFailure(failure) = recovered.pop().unwrap() else {
            panic!("registered source required before physical completion")
        };
        assert_eq!(failure.ticket, expected_ticket);
        assert!(
            matches!(failure.source, WorkerFailureSource::Factory(error) if error == injected_backend_failure())
        );
        assert!(!owner.workers[0].is_finished());
        release.0.take().unwrap().send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        owner.reap_workers(true).unwrap();
    }

    #[test]
    fn returned_failure_before_thread_finish_keeps_original_owner_visible() {
        let (mut owner, mut session) = fixture();
        let event = foreign_failure_event(&mut owner, &mut session);
        let rz_search::contracts::ContractPumpEvent::RejectedResult { result, .. } = event else {
            panic!("foreign result fixture")
        };
        let contract::EvalResult::Failed(original) = *result else {
            panic!("original failure fixture")
        };
        let expected = original.context;
        let ticket = session.active_ticket().unwrap();
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        let (entered, waiting) = mpsc::sync_channel(1);
        let (release, gate) = mpsc::sync_channel(1);
        let publication = Arc::new(Mutex::new(None));
        let worker_publication = Arc::clone(&publication);
        let handle = thread::spawn(move || {
            let result = finish_worker(
                &sender,
                ticket,
                WorkerCompletion::failed(WorkerFailureSource::Search(Box::new(
                    rz_search::contracts::ContractSearchFailure::Evaluation(Box::new(original)),
                ))),
                &worker_publication,
                None,
                Duration::from_secs(1),
            );
            // Model preemption after finish_worker returned and before the
            // thread publishes completion through JoinHandle::is_finished.
            entered.send(()).unwrap();
            gate.recv_timeout(Duration::from_secs(5)).unwrap();
            result
        });
        owner.workers.push(Worker {
            handle,
            failure: publication,
        });
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!owner.workers[0].is_finished());
        let mut recovered = owner.collect_unconsumed_failures();
        let EngineError::UndeliveredWorkerFailure(source) = recovered.pop().unwrap() else {
            panic!("original must remain visible after closure return")
        };
        let WorkerFailureSource::Search(source) = source.source else {
            panic!("typed original search failure")
        };
        let rz_search::contracts::ContractSearchFailure::Evaluation(source) = *source else {
            panic!("original evaluator failure")
        };
        assert_eq!(source.context, expected);
        assert_eq!(source.recovery, contract::RecoveryOutcome::Failed);
        release.send(()).unwrap();
        let result = owner.workers.pop().unwrap().handle.join().unwrap();
        let EngineError::UndeliveredWorkerFailure(copy) = result.unwrap_err() else {
            panic!("join must retain typed failure copy")
        };
        let WorkerFailureSource::Search(copy) = copy.source else {
            panic!("typed join search failure")
        };
        let rz_search::contracts::ContractSearchFailure::Evaluation(copy) = *copy else {
            panic!("join evaluator failure")
        };
        assert_eq!(copy.context, expected);
        assert_eq!(copy.recovery, contract::RecoveryOutcome::Failed);
    }

    #[test]
    fn injected_identity_reaches_uci_and_newline_identity_is_rejected_before_output() {
        let run = |name: &str| {
            let (owner, session) = fixture();
            let process = EngineProcess::new(
                Arc::clone(&owner.factory),
                Arc::clone(&session.snapshot().owners),
                owner.clock.clone(),
            )
            .with_identity(EngineIdentity {
                name: name.into(),
                author: "RoveZero identity fixture".into(),
            });
            let (sender, events) = event_channel();
            sender.send(Event::Line("uci".into())).unwrap();
            sender.send(Event::Line("quit".into())).unwrap();
            let mut protocol = Vec::new();
            let mut diagnostics = Vec::new();
            let result = serve(
                events,
                sender,
                &mut protocol,
                &mut diagnostics,
                process,
                owner.settings,
            );
            (result, String::from_utf8(protocol).unwrap(), diagnostics)
        };

        let (result, protocol, diagnostics) = run("RoveZero native CPU identity fixture");
        result.unwrap();
        assert_eq!(
            protocol.lines().collect::<Vec<_>>(),
            [
                "id name RoveZero native CPU identity fixture",
                "id author RoveZero identity fixture",
                "uciok",
            ]
        );
        assert!(diagnostics.is_empty());

        let (result, protocol, diagnostics) = run("RoveZero native CPU\nid name injected");
        let Err(EngineError::Session(error)) = result else {
            panic!("Session must reject the injected newline identity")
        };
        assert!(error.to_string().contains("engine identity requires"));
        assert!(protocol.is_empty());
        assert!(diagnostics.is_empty());
    }
}
