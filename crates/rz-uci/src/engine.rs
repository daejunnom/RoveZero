//! Concrete Rules → UCI → search → common evaluation assembly.
//! Model loading and physical execution remain the injected factory's responsibility.

use crate::{
    bridge::{BuildSearchSettings, SearchBinding, SideToMove},
    *,
};
use rz_contracts as contract;
use rz_position::{
    BoardMove, Position, PositionLimits,
    contracts::{ContractPosition, ContractState, RulesState},
};
use rz_search::{
    TreeLimits,
    driver::{self, CheckedPosition, SearchControl, SearchStatus},
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
#[derive(Clone, Debug)]
pub struct ProcessClock {
    epoch: contract::ProcessEpoch,
    origin: Instant,
}
impl ProcessClock {
    pub fn new(epoch: contract::ProcessEpoch) -> Self {
        Self {
            epoch,
            origin: Instant::now(),
        }
    }
    pub fn domain(&self) -> contract::ClockDomain {
        contract::ClockDomain(self.epoch)
    }
    pub fn epoch(&self) -> contract::ProcessEpoch {
        self.epoch
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

#[derive(Clone, Debug)]
pub struct SearchAuthority {
    current: Arc<Mutex<contract::AcceptanceScope>>,
    cancel: contract::CancelToken,
}
impl SearchAuthority {
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

/// Shutdown must close admission and wait for physical work within its finite bound.
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

#[derive(Debug)]
pub struct EvaluationFailure {
    pub primary: contract::ContractError,
    pub cleanup: Option<contract::ContractError>,
}
impl fmt::Display for EvaluationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.primary)?;
        if let Some(cleanup) = self.cleanup {
            write!(formatter, "; cancel: {cleanup}")?;
        }
        Ok(())
    }
}
impl std::error::Error for EvaluationFailure {}

/// One worker issues one common request per pending B selection. B's private tree
/// ticket continues to own exactly-once backup. A shared sequence spans workers.
pub struct ContractSearchEvaluator {
    runtime: Box<dyn ManagedEvaluator>,
    factory: Arc<dyn EvaluatorFactory>,
    profile: EvaluatorProfile,
    clock: ProcessClock,
    authority: SearchAuthority,
    scope: contract::AcceptanceScope,
    sequence: Arc<AtomicU64>,
}
impl ContractSearchEvaluator {
    fn stop(
        &mut self,
        request: &contract::EvalRequest<RulesState>,
        primary: contract::ContractError,
    ) -> EvaluationFailure {
        request.cancel_token().cancel();
        EvaluationFailure {
            primary,
            cleanup: self.runtime.cancel(request.context().request).err(),
        }
    }
}
impl driver::Evaluator<RulesSearchPosition> for ContractSearchEvaluator {
    type Error = EvaluationFailure;
    fn evaluate(
        &mut self,
        position: &RulesSearchPosition,
        legal: &[contract::Move],
        control: &SearchControl,
    ) -> Result<driver::Evaluation, Self::Error> {
        let build =
            || -> Result<Arc<contract::EvalRequest<RulesState>>, contract::ContractError> {
                if legal != position.state.legal_moves().moves() {
                    return Err(failure(
                        contract::ErrorCode::IdentityMismatch,
                        "search changed Rules legal order",
                    ));
                }
                let previous = self
                    .sequence
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
                    .map_err(|_| {
                        failure(
                            contract::ErrorCode::ResourceExhausted,
                            "request sequence exhausted",
                        )
                    })?;
                let context = contract::EvalContext {
                    revision: contract::CONTRACT_REVISION,
                    request: contract::RequestId::new(self.clock.epoch(), previous + 1),
                    selection: contract::SelectionId::new(self.clock.epoch(), previous + 1),
                    game: self.scope.game,
                    root: self.scope.root,
                    state: position.state.snapshot().identity(),
                    legal_order: position.state.legal_moves().order(),
                    input: self.factory.input_key(position.state.rules(), legal)?,
                    model: self.profile.model.handle(),
                    encoding: self.profile.model.encoding().handle,
                    precision: self.profile.precision,
                    compute: self.profile.compute,
                    backend: self.profile.backend,
                };
                Ok(Arc::new(contract::EvalRequest::try_new(
                    context,
                    position.state.snapshot().clone(),
                    position.state.legal_moves().clone(),
                    Arc::clone(&self.profile.model),
                    self.clock.deadline(control.deadline)?,
                    self.authority.cancel_token(),
                    self.profile.bytes,
                )?))
            };
        let request = build().map_err(|primary| EvaluationFailure {
            primary,
            cleanup: None,
        })?;
        if let Err(primary) = request.validate_acceptance(
            self.authority
                .current_scope()
                .map_err(|primary| EvaluationFailure {
                    primary,
                    cleanup: None,
                })?,
            self.clock.domain(),
            self.clock.now().map_err(|primary| EvaluationFailure {
                primary,
                cleanup: None,
            })?,
        ) {
            return Err(self.stop(&request, primary));
        }
        if let Err(primary) = self.runtime.submit(Arc::clone(&request)) {
            return Err(self.stop(&request, primary));
        }
        loop {
            if control.cancellation.load(Ordering::Acquire) {
                self.authority.cancel();
            }
            let checked = self.authority.current_scope().and_then(|scope| {
                self.clock
                    .now()
                    .and_then(|now| request.validate_acceptance(scope, self.clock.domain(), now))
            });
            if let Err(primary) = checked {
                return Err(self.stop(&request, primary));
            }
            if let Some(result) = self.runtime.poll() {
                let context = match &result {
                    contract::EvalResult::Completed(output) => output.context,
                    contract::EvalResult::Canceled(context)
                    | contract::EvalResult::Expired(context)
                    | contract::EvalResult::Stale(context) => context.request,
                    contract::EvalResult::Failed(error) => error.context.request,
                };
                if context != request.context() {
                    return Err(self.stop(
                        &request,
                        failure(
                            contract::ErrorCode::IdentityMismatch,
                            "runtime returned another request context",
                        ),
                    ));
                }
                let output = match result {
                    contract::EvalResult::Completed(output) => output,
                    contract::EvalResult::Failed(error) => {
                        return Err(self.stop(&request, error.error));
                    }
                    contract::EvalResult::Canceled(_) => {
                        return Err(self.stop(
                            &request,
                            failure(contract::ErrorCode::Canceled, "runtime canceled request"),
                        ));
                    }
                    contract::EvalResult::Expired(_) => {
                        return Err(self.stop(
                            &request,
                            failure(contract::ErrorCode::Expired, "runtime expired request"),
                        ));
                    }
                    contract::EvalResult::Stale(_) => {
                        return Err(self.stop(
                            &request,
                            failure(contract::ErrorCode::Stale, "runtime rejected stale request"),
                        ));
                    }
                };
                if control.cancellation.load(Ordering::Acquire) {
                    self.authority.cancel();
                }
                let checked = self.authority.current_scope().and_then(|scope| {
                    self.clock.now().and_then(|now| {
                        output.validate_for(&request, scope, self.clock.domain(), now)
                    })
                });
                if let Err(primary) = checked {
                    return Err(self.stop(&request, primary));
                }
                return Ok(driver::Evaluation {
                    priors: output.policy.normalized(),
                    wdl: output.wdl.normalized(),
                });
            }
            thread::sleep(Duration::from_millis(1));
        }
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

#[derive(Debug)]
pub enum EngineError {
    Contract(contract::ContractError),
    Session(SessionError),
    Transport(Box<ServeError<EngineError>>),
    WorkerPanic,
    DrainTimeout,
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
    binding: Arc<SearchBinding>,
    authority: SearchAuthority,
}
struct Owner {
    active: Option<ActiveBinding>,
    scope: Arc<Mutex<contract::AcceptanceScope>>,
    factory: Arc<dyn EvaluatorFactory>,
    clock: ProcessClock,
    sequence: Arc<AtomicU64>,
    settings: EngineSettings,
    workers: Vec<JoinHandle<Result<(), contract::ContractError>>>,
    pending_failures: VecDeque<(SearchTicket, SearchCompletion)>,
}
impl Owner {
    fn pending_failure(
        &mut self,
        sender: &SyncSender<Event>,
        ticket: SearchTicket,
        completion: SearchCompletion,
    ) {
        self.pending_failures.push_back((ticket, completion));
        // The receipt is owner-held. A full bounded queue already contains an
        // event that will trigger its delivery; an empty queue needs a wakeup.
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
        while let Some((ticket, completion)) = self.pending_failures.pop_front() {
            append_result(
                &mut out,
                unbound_event(session, Event::Complete { ticket, completion }),
            );
        }
        let next = if matches!(
            event,
            Event::RejectedInput {
                code: "OwnerWake",
                ..
            }
        ) {
            SessionResult::default()
        } else if let Some(active) = &self.active {
            match event {
                Event::Progress { ticket, bestmove } if &ticket == active.binding.ticket() => {
                    active.binding.progress(session, &bestmove, Instant::now())
                }
                Event::Complete { ticket, completion } if &ticket == active.binding.ticket() => {
                    active.binding.complete(session, completion, Instant::now())
                }
                Event::Deadline(ticket) if &ticket == active.binding.ticket() => {
                    active.binding.expire(session, Instant::now())
                }
                event => unbound_event(session, event),
            }
        } else {
            unbound_event(session, event)
        };
        append_result(&mut out, next);
        out
    }
    fn cancel(&mut self) {
        if let Some(active) = &self.active {
            active.binding.cancel();
            active.authority.cancel();
        }
    }
    fn bump(&mut self, game: bool) -> Result<(), EngineError> {
        self.cancel();
        let mut scope = self.scope.lock().map_err(|_| {
            failure(
                contract::ErrorCode::BackendFailure,
                "acceptance authority poisoned",
            )
        })?;
        scope.root.0 = scope.root.0.checked_add(1).ok_or_else(|| {
            failure(
                contract::ErrorCode::ResourceExhausted,
                "root generation exhausted",
            )
        })?;
        if game {
            scope.game.0 = scope.game.0.checked_add(1).ok_or_else(|| {
                failure(
                    contract::ErrorCode::ResourceExhausted,
                    "game generation exhausted",
                )
            })?;
        }
        Ok(())
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
                    .is_some_and(|active| active.binding.ticket() == &ticket)
                {
                    self.cancel();
                }
            }
            Effect::NewGame => {
                self.bump(true)?;
            }
            Effect::Shutdown => {
                self.bump(false)?;
            }
            Effect::Start {
                ticket,
                snapshot,
                limits,
                ..
            } => {
                self.bump(false)?;
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
                let side = match snapshot.state.snapshot().side_to_move() {
                    contract::Color::White => SideToMove::White,
                    contract::Color::Black => SideToMove::Black,
                };
                let binding = match SearchBinding::new(
                    ticket.clone(),
                    &limits,
                    side,
                    Instant::now(),
                    self.settings.search,
                ) {
                    Ok(binding) => Arc::new(binding),
                    Err(error) => {
                        self.pending_failure(
                            sender,
                            ticket,
                            SearchCompletion::Failed {
                                code: "SearchConfiguration".into(),
                                message: error.to_string(),
                            },
                        );
                        return Ok(());
                    }
                };
                let authority = SearchAuthority {
                    current: Arc::clone(&self.scope),
                    cancel: contract::CancelToken::new(),
                };
                self.active = Some(ActiveBinding {
                    binding: Arc::clone(&binding),
                    authority: authority.clone(),
                });
                let factory = Arc::clone(&self.factory);
                let clock = self.clock.clone();
                let sequence = Arc::clone(&self.sequence);
                let profile = factory.profile();
                let scope = authority.current_scope()?;
                let tree = self.settings.tree;
                let shutdown_limit = self.settings.shutdown_limit;
                let events = sender.clone();
                self.workers.push(thread::spawn(move || {
                    let runtime = match factory.create(clock.clone(), authority.clone()) {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            let _ = events.send(Event::Complete {
                                ticket,
                                completion: SearchCompletion::Failed {
                                    code: format!("{:?}", error.code),
                                    message: error.to_string(),
                                },
                            });
                            return Ok(());
                        }
                    };
                    let mut evaluator = ContractSearchEvaluator {
                        runtime,
                        factory,
                        profile,
                        clock,
                        authority,
                        scope,
                        sequence,
                    };
                    let outcome = driver::run_search_with_progress(
                        &snapshot,
                        &mut evaluator,
                        binding.control(),
                        tree,
                        |progress| {
                            if let Some(movement) = progress
                                .best_move
                                .and_then(|movement| move_text(movement).ok())
                            {
                                let _ = events.try_send(Event::Progress {
                                    ticket: ticket.clone(),
                                    bestmove: movement,
                                });
                            }
                        },
                    );
                    let completion = match outcome.status {
                        SearchStatus::Failed(error) => SearchCompletion::Failed {
                            code: "SearchFailed".into(),
                            message: format!("{error:?}"),
                        },
                        _ => SearchCompletion::Completed {
                            bestmove: outcome
                                .best_move
                                .and_then(|movement| move_text(movement).ok()),
                        },
                    };
                    let _ = events.send(Event::Complete { ticket, completion });
                    let deadline = Instant::now().checked_add(shutdown_limit).ok_or_else(|| {
                        failure(
                            contract::ErrorCode::ResourceExhausted,
                            "shutdown deadline overflow",
                        )
                    })?;
                    evaluator.runtime.shutdown(deadline)
                }));
            }
            Effect::OptionChanged { .. } | Effect::Button { .. } => {
                return Err(EngineError::Contract(failure(
                    contract::ErrorCode::UnsupportedContract,
                    "bootstrap did not register configurable backend options",
                )));
            }
        }
        Ok(())
    }
}

fn append_result<S>(out: &mut SessionResult<S>, next: SessionResult<S>) {
    out.protocol.extend(next.protocol);
    out.diagnostics.extend(next.diagnostics);
    out.effects.extend(next.effects);
    out.accepted &= next.accepted;
}

fn unbound_event(
    session: &mut Session<RulesUciPort>,
    event: Event,
) -> SessionResult<RulesSearchPosition> {
    let source = match &event {
        Event::Complete {
            completion: SearchCompletion::Failed { code, message },
            ..
        } => Some(format!("{code}: {message}")),
        _ => None,
    };
    let mut out = handle_event(session, event);
    if let Some(message) = source {
        if !out
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "SearchFailed")
        {
            out.diagnostics.push(Diagnostic {
                code: "SearchFailed",
                message,
            });
        }
    }
    out
}

/// Runs the owner loop. Input has a bounded independent producer; this owner
/// uses an independent timer so a pending evaluation cannot block stop/deadline.
/// This CPU/mock entry point does not assert neural/GPU acceptance.
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
    if settings.max_workers == 0 || settings.shutdown_limit.is_zero() {
        return Err(failure(
            contract::ErrorCode::InvalidInput,
            "finite positive worker/drain limits required",
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
    let owner = Arc::new(Mutex::new(Owner {
        active: None,
        scope: Arc::new(Mutex::new(scope)),
        factory,
        clock,
        sequence: Arc::new(AtomicU64::new(0)),
        settings,
        workers: Vec::new(),
        pending_failures: VecDeque::new(),
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
    // A single timer observes current binding. No timer is spawned per request.
    let timer_owner = Arc::clone(&owner);
    let timer_events = sender.clone();
    let timer_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let timer_flag = Arc::clone(&timer_stop);
    let timer = thread::spawn(move || {
        let mut delivered = None;
        while !timer_flag.load(Ordering::Acquire) {
            if let Ok(owner) = timer_owner.lock() {
                if let Some(active) = &owner.active {
                    let binding = &active.binding;
                    if delivered.as_ref() != Some(binding.ticket())
                        && Instant::now() >= binding.control().deadline
                    {
                        if timer_events
                            .try_send(Event::Deadline(binding.ticket().clone()))
                            .is_ok()
                        {
                            delivered = Some(binding.ticket().clone());
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
            let mut owner = handler_owner
                .lock()
                .expect("owner is only mutated by UCI event loop");
            owner.handle(session, event)
        },
        move |effect| {
            dispatch_owner
                .lock()
                .map_err(|_| {
                    EngineError::Contract(failure(
                        contract::ErrorCode::BackendFailure,
                        "UCI owner poisoned",
                    ))
                })?
                .dispatch(effect, &sender)
        },
    );
    timer_stop.store(true, Ordering::Release);
    drop(events); // Unblock any bounded completion publisher before joining workers.
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
                    cleanup.push(EngineError::Contract(failure(
                        contract::ErrorCode::BackendFailure,
                        "UCI owner poisoned",
                    )));
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

/// Bounded transport constructor shared by the executable and integration tests.
pub fn event_channel() -> (SyncSender<Event>, Receiver<Event>) {
    mpsc::sync_channel(128)
}
