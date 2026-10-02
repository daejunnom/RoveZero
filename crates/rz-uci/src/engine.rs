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

#[derive(Debug)]
pub enum EngineError {
    Contract(contract::ContractError),
    Session(SessionError),
    Transport(Box<ServeError<EngineError>>),
    WorkerPanic,
    DrainTimeout,
    UndeliveredDiagnostics(Vec<(WorkerDiagnosticReceipt, u64)>),
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
struct Owner {
    session_owner: crate::contracts::ContractSessionOwner,
    active: Option<ActiveBinding>,
    scope: Arc<Mutex<contract::AcceptanceScope>>,
    factory: Arc<dyn EvaluatorFactory>,
    clock: ProcessClock,
    ids: Arc<rz_search::contracts::IdAllocator>,
    settings: EngineSettings,
    workers: Vec<JoinHandle<Result<(), contract::ContractError>>>,
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
        let mut index = 0;
        while index < self.workers.len() {
            if self.workers[index].is_finished() {
                self.workers
                    .swap_remove(index)
                    .join()
                    .map_err(|_| EngineError::WorkerPanic)??;
            } else {
                index += 1;
            }
        }
        Ok(())
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
                self.workers.push(thread::spawn(move || {
                    let mut runtime = match factory.create(clock.clone(), authority.clone()) {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            let _ = events.send(Event::Complete {
                                ticket,
                                completion: failed_completion(error),
                            });
                            return Ok(());
                        }
                    };
                    let mut search =
                        match rz_search::contracts::ContractSearch::new(snapshot, config) {
                            Ok(search) => search,
                            Err(error) => {
                                let _ = events.send(Event::Complete {
                                    ticket,
                                    completion: failed_completion(error),
                                });
                                return shutdown(runtime.as_mut(), shutdown_limit);
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
                        (Some(error), _, _) => SearchCompletion::Failed {
                            code: "DiagnosticRetentionFailure".into(),
                            message: error.to_string(),
                        },
                        (_, Some(error), _) => failed_completion(error),
                        (_, _, rz_search::contracts::ContractSearchStatus::Failed(error)) => {
                            SearchCompletion::Failed {
                                code: "SearchFailed".into(),
                                message: format!("{error:?}"),
                            }
                        }
                        _ => SearchCompletion::Completed {
                            bestmove: outcome
                                .best_move
                                .and_then(|movement| move_text(movement).ok()),
                        },
                    };
                    let _ = events.send(Event::Complete { ticket, completion });
                    shutdown(runtime.as_mut(), shutdown_limit)
                }));
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
    factory: Arc<dyn EvaluatorFactory>,
    owners: Arc<OwnerRegistry>,
    clock: ProcessClock,
    settings: EngineSettings,
) -> Result<(), EngineError> {
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
        EngineIdentity {
            name: "RoveZero CPU mock integration".into(),
            author: "RoveZero contributors".into(),
        },
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
                    {
                        if timer_events
                            .try_send(Event::Deadline(active.ticket.clone()))
                            .is_ok()
                        {
                            delivered = Some(active.ticket.clone());
                        }
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
            if let Err(error) = owner.reap() {
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
        Ok(owner) => match owner.diagnostics.lock() {
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
        },
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
        owner.workers.push(thread::spawn(move || {
            waiting.recv().unwrap();
            Ok(())
        }));
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
        owner.workers.pop().unwrap().join().unwrap().unwrap();
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
}
