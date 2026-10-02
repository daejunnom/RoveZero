//! 공통 0.1 game/root·registry·clock·cancel 권한과 UCI 출력 소유권의 연결.
//! 실제 Rules/registry는 caller가 공급한다. 이 owner는 체스 상태나 digest를 만들지
//! 않으며 Runtime의 raw-cache namespace 초기화와 물리 buffer drain을 대체하지 않는다.

use std::time::Instant;

use rz_contracts::{
    AcceptanceScope, Color, ContractError, ErrorCode, GameGeneration, ProcessEpoch, RootGeneration,
    Stage,
};
use rz_search::contract_time::{
    ContractCancellation, ContractClock, ContractDeadlines, InstantClock,
};
use rz_search::driver::SearchControl;

use crate::bridge::{BudgetKind, BuildSearchError, BuildSearchSettings, SearchBinding, SideToMove};
use crate::{
    Command, Diagnostic, Effect, Event, PositionPort, SearchCompletion, SearchTicket, Session,
    SessionResult, handle_event, parse,
};

/// Typed contract causes remain alongside protocol/legacy backend diagnostics.
/// An event-loop handler forwards `session` only after recording `errors`.
#[derive(Debug)]
pub struct ContractSessionOutcome<S> {
    pub session: SessionResult<S>,
    pub errors: Vec<ContractError>,
}

impl<S> From<SessionResult<S>> for ContractSessionOutcome<S> {
    fn from(session: SessionResult<S>) -> Self {
        Self {
            session,
            errors: Vec::new(),
        }
    }
}

struct CurrentSearch {
    binding: SearchBinding,
    cancellation: ContractCancellation,
    deadlines: ContractDeadlines,
    scope: AcceptanceScope,
    infinite: bool,
    work_completed: bool,
}

/// One Session owner's acceptance authority. Epoch, model/encoding/backend
/// handles, and initial generations are explicit inputs from their real owners.
/// Reusing this owner with an independently active Session is rejected.
pub struct ContractSessionOwner {
    scope: AcceptanceScope,
    epoch: ProcessEpoch,
    clock: InstantClock,
    settings: BuildSearchSettings,
    last_dispatch: Instant,
    ticket: Option<SearchTicket>,
    current: Option<CurrentSearch>,
}

impl ContractSessionOwner {
    pub fn new(
        scope: AcceptanceScope,
        epoch: ProcessEpoch,
        clock: InstantClock,
        settings: BuildSearchSettings,
    ) -> Result<Self, ContractError> {
        if clock.domain().0 != epoch {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "process epoch and UCI clock domain differ",
            ));
        }
        clock.now()?;
        Ok(Self {
            scope,
            epoch,
            clock,
            settings,
            last_dispatch: Instant::now(),
            ticket: None,
            current: None,
        })
    }

    pub fn scope(&self) -> AcceptanceScope {
        self.scope
    }
    pub fn process_epoch(&self) -> ProcessEpoch {
        self.epoch
    }
    pub fn clock(&self) -> &InstantClock {
        &self.clock
    }
    pub fn active_ticket(&self) -> Option<&SearchTicket> {
        self.ticket.as_ref()
    }
    pub fn active_scope(&self) -> Option<AcceptanceScope> {
        self.current.as_ref().map(|c| c.scope)
    }
    pub fn active_binding(&self) -> Option<&SearchBinding> {
        self.current.as_ref().map(|c| {
            mirror_cancel(c);
            &c.binding
        })
    }
    pub fn control(&self) -> Option<&SearchControl> {
        self.active_binding().map(SearchBinding::control)
    }
    pub fn cancellation(&self) -> Option<&ContractCancellation> {
        self.current.as_ref().map(|c| &c.cancellation)
    }
    pub fn deadlines(&self) -> Option<&ContractDeadlines> {
        self.current.as_ref().map(|c| &c.deadlines)
    }

    /// Registry replacement is explicit and idle-only. Caller must provide the
    /// live handles/backend while preserving this owner's game/root counters.
    pub fn set_registry<P: PositionPort>(
        &mut self,
        session: &Session<P>,
        scope: AcceptanceScope,
    ) -> Result<(), ContractError> {
        self.ensure_session(session)?;
        if session.active_ticket().is_some() {
            return Err(error(
                ErrorCode::InvalidInput,
                "registry update requires an idle UCI session",
            ));
        }
        if scope.game != self.scope.game || scope.root != self.scope.root {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "registry update cannot replace game/root generations",
            ));
        }
        self.scope = scope;
        Ok(())
    }

    /// Capture go's original Instant before parsing, Rules preparation, or side
    /// validation. Accepted root commands advance checked generations exactly
    /// once; malformed and rejected Rules transitions preserve prior authority.
    pub fn handle_line<P, F>(
        &mut self,
        session: &mut Session<P>,
        line: &str,
        side: F,
    ) -> ContractSessionOutcome<P::Snapshot>
    where
        P: PositionPort,
        F: Fn(&P::Snapshot) -> Result<Color, ContractError>,
    {
        let start = Instant::now();
        if let Err(err) = self.ensure_session(session) {
            return rejected(err);
        }
        let command = match parse(line, session.parser_limits()) {
            Ok(command) => command,
            Err(_) => return session.handle_line(line).into(),
        };
        let root_change = matches!(
            command,
            Command::Go(_) | Command::Position(_) | Command::NewGame
        );
        let new_game = matches!(command, Command::NewGame);
        let next = if root_change {
            match self.next_generations(new_game) {
                Ok(next) => Some(next),
                Err(err) => return rejected(err),
            }
        } else {
            None
        };
        let mut out = ContractSessionOutcome::from(session.handle_line(line));
        if !out.session.accepted {
            return out;
        }
        if let Some((game, root)) = next {
            self.close_current();
            self.scope.game = game;
            self.scope.root = root;
        }
        if matches!(command, Command::Stop | Command::Quit) {
            self.close_current();
        }
        let effects = std::mem::take(&mut out.session.effects);
        for effect in effects {
            if let Effect::Start {
                ticket,
                snapshot,
                limits,
                options,
            } = effect
            {
                let mut build_detail = None;
                let built = side(&snapshot).and_then(|color| {
                    let color = match color {
                        Color::White => SideToMove::White,
                        Color::Black => SideToMove::Black,
                    };
                    let binding =
                        SearchBinding::new(ticket.clone(), &limits, color, start, self.settings)
                            .map_err(|err| {
                                build_detail = Some(err.to_string());
                                map_build_error(err)
                            })?;
                    let deadlines = ContractDeadlines::from_budget(&self.clock, binding.budget())?;
                    let cancellation = ContractCancellation::from_control(binding.control());
                    Ok(CurrentSearch {
                        binding,
                        deadlines,
                        cancellation,
                        scope: self.scope,
                        infinite: limits.infinite,
                        work_completed: false,
                    })
                });
                match built {
                    Ok(current) => {
                        self.current = Some(current);
                        out.session.effects.push(Effect::Start {
                            ticket,
                            snapshot,
                            limits,
                            options,
                        });
                    }
                    Err(err) => {
                        let failed = session.complete(
                            &ticket,
                            SearchCompletion::Failed {
                                code: format!("{:?}/{:?}", err.stage, err.code),
                                message: err.detail.into(),
                            },
                        );
                        merge(&mut out.session, failed);
                        attach_error(&mut out, err);
                        if let Some(message) = build_detail {
                            out.session.diagnostics.push(Diagnostic {
                                code: "SearchConfigurationFailed",
                                message,
                            });
                        }
                    }
                }
            } else {
                out.session.effects.push(effect);
            }
        }
        self.synchronize(session);
        out
    }

    /// Owner dequeue entry point. Search callbacks are never admitted through
    /// the untimed default event reducer; final Session guards re-read this clock.
    pub fn handle_event<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        event: Event,
    ) -> ContractSessionOutcome<P::Snapshot> {
        self.handle_event_at(session, event, Instant::now())
    }

    /// Trusted owner dispatch anchor, useful for deterministic boundary tests.
    /// It cannot admit using an old producer tick: each final guard also checks
    /// actual Instant::now and uses the later time in the owned clock domain.
    pub fn handle_event_at<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        event: Event,
        now: Instant,
    ) -> ContractSessionOutcome<P::Snapshot> {
        if let Err(err) = self.ensure_session(session) {
            return rejected(err);
        }
        let now = now.max(Instant::now()).max(self.last_dispatch);
        self.last_dispatch = now;
        let out = match event {
            Event::Progress { ticket, bestmove } => self.progress(session, &ticket, &bestmove, now),
            Event::Complete { ticket, completion } => {
                self.complete(session, &ticket, completion, now)
            }
            Event::Deadline(ticket) => self.expire(session, &ticket, now),
            Event::Line(_) => {
                return rejected(error(
                    ErrorCode::InvalidInput,
                    "line events require handle_line with the checked Rules side adapter",
                ));
            }
            Event::EndOfInput => {
                self.close_current();
                handle_event(session, Event::EndOfInput).into()
            }
            event => handle_event(session, event).into(),
        };
        self.synchronize(session);
        // The common cancellation clone is authoritative even when an invalid
        // candidate returned before its final guard. Mirror it into legacy work.
        if let Some(current) = &self.current {
            mirror_cancel(current);
        }
        out
    }

    fn progress<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        ticket: &SearchTicket,
        bestmove: &str,
        now: Instant,
    ) -> ContractSessionOutcome<P::Snapshot> {
        let scope = self.scope;
        let clock = self.clock;
        let Some(current) = self.matching_current(ticket) else {
            return stale(None);
        };
        if current.work_completed {
            return stale(None);
        }
        mirror_cancel(current);
        let mut failure = None;
        let prepared = session.progress_with_guard(ticket, bestmove, || {
            failure = acceptance(current, scope, clock, now).err();
            failure.is_none()
        });
        match failure {
            Some(err) => reject_prepared(session, current, prepared, err),
            None => prepared.into(),
        }
    }

    fn complete<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        ticket: &SearchTicket,
        completion: SearchCompletion,
        now: Instant,
    ) -> ContractSessionOutcome<P::Snapshot> {
        let scope = self.scope;
        let clock = self.clock;
        let Some(current) = self.matching_current(ticket) else {
            return stale(Some(completion));
        };
        if current.work_completed {
            return stale(Some(completion));
        }
        mirror_cancel(current);
        let mut failure = None;
        let mut called = false;
        let prepared = session.complete_with_guard(ticket, completion, || {
            called = true;
            failure = acceptance(current, scope, clock, now).err();
            failure.is_none()
        });
        if called && failure.is_none() {
            current.work_completed = true;
        }
        match failure {
            Some(err) => reject_prepared(session, current, prepared, err),
            None => prepared.into(),
        }
    }

    fn expire<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        ticket: &SearchTicket,
        now: Instant,
    ) -> ContractSessionOutcome<P::Snapshot> {
        let scope = self.scope;
        let clock = self.clock;
        let Some(current) = self.matching_current(ticket) else {
            return stale(None);
        };
        // Completed infinite output waiting is passive; old timers never invent
        // a new failure or release a second bestmove.
        if current.work_completed {
            current.cancellation.cancel();
            return SessionResult::default().into();
        }
        match acceptance(current, scope, clock, now) {
            Ok(()) => rejected(error(
                ErrorCode::InvalidInput,
                "deadline event arrived before hard boundary",
            )),
            Err(err) if err.code == ErrorCode::Expired => expire_current(session, current, err),
            Err(err) => rejected(err),
        }
    }

    fn ensure_session<P: PositionPort>(&self, session: &Session<P>) -> Result<(), ContractError> {
        if session.active_ticket().as_ref() != self.ticket.as_ref() {
            return Err(error(
                ErrorCode::Stale,
                "Session active ticket belongs to a different lifecycle owner",
            ));
        }
        Ok(())
    }
    fn next_generations(
        &self,
        new_game: bool,
    ) -> Result<(GameGeneration, RootGeneration), ContractError> {
        let root = self
            .scope
            .root
            .0
            .checked_add(1)
            .ok_or_else(|| error(ErrorCode::ResourceExhausted, "root generation overflow"))?;
        let game =
            if new_game {
                self.scope.game.0.checked_add(1).ok_or_else(|| {
                    error(ErrorCode::ResourceExhausted, "game generation overflow")
                })?
            } else {
                self.scope.game.0
            };
        Ok((GameGeneration(game), RootGeneration(root)))
    }
    fn close_current(&mut self) {
        if let Some(current) = self.current.take() {
            current.cancellation.cancel();
        }
        self.ticket = None;
    }
    fn synchronize<P: PositionPort>(&mut self, session: &Session<P>) {
        self.ticket = session.active_ticket();
        if self.ticket.is_none() {
            self.close_current();
        }
    }
    fn matching_current(&mut self, ticket: &SearchTicket) -> Option<&mut CurrentSearch> {
        self.current
            .as_mut()
            .filter(|current| current.binding.ticket() == ticket)
    }
}

impl Drop for ContractSessionOwner {
    fn drop(&mut self) {
        self.close_current();
    }
}

fn acceptance(
    current: &CurrentSearch,
    scope: AcceptanceScope,
    clock: InstantClock,
    now: Instant,
) -> Result<(), ContractError> {
    if !same_scope(current.scope, scope) {
        return Err(error(
            ErrorCode::Stale,
            "search game/root or registry scope no longer current",
        ));
    }
    if current.cancellation.is_canceled() {
        current.cancellation.cancel();
        return Err(error(
            ErrorCode::Canceled,
            "common logical search cancellation closed admission",
        ));
    }
    current
        .deadlines
        .hard
        .accepts(clock.domain(), clock.tick_at(now.max(Instant::now()))?)
}

fn mirror_cancel(current: &CurrentSearch) {
    if current.cancellation.is_canceled() {
        current.cancellation.cancel();
    }
}
fn same_scope(a: AcceptanceScope, b: AcceptanceScope) -> bool {
    a.game == b.game
        && a.root == b.root
        && a.model == b.model
        && a.encoding == b.encoding
        && a.backend == b.backend
}
fn error(code: ErrorCode, detail: &'static str) -> ContractError {
    ContractError::new(code, Stage::Admission, detail)
}
fn map_build_error(err: BuildSearchError) -> ContractError {
    match err {
        BuildSearchError::SimulationLimitExceeded { .. } => error(
            ErrorCode::ResourceExhausted,
            "go nodes exceeds configured simulation bound",
        ),
        BuildSearchError::UntimedDeadlineOverflow => {
            error(ErrorCode::ResourceExhausted, "untimed deadline overflow")
        }
        BuildSearchError::TimeBudget(_) => error(
            ErrorCode::InvalidInput,
            "UCI time budget could not be constructed",
        ),
        _ => error(
            ErrorCode::InvalidInput,
            "UCI search settings or limits are invalid",
        ),
    }
}
fn attach_error<S>(out: &mut ContractSessionOutcome<S>, err: ContractError) {
    out.errors.push(err);
    out.session.diagnostics.push(Diagnostic {
        code: "ContractFailure",
        message: err.to_string(),
    });
    out.session.accepted = false;
}
fn rejected<S>(err: ContractError) -> ContractSessionOutcome<S> {
    let mut out = SessionResult::default().into();
    attach_error(&mut out, err);
    out
}
fn stale<S>(completion: Option<SearchCompletion>) -> ContractSessionOutcome<S> {
    let mut out = rejected(error(
        ErrorCode::Stale,
        "inactive or foreign common search ticket",
    ));
    if let Some(SearchCompletion::Failed { code, message }) = completion {
        out.session.diagnostics.push(Diagnostic {
            code: "SearchFailed",
            message: format!("{code}: {message}"),
        });
    }
    out
}
fn merge<S>(out: &mut SessionResult<S>, other: SessionResult<S>) {
    out.protocol.extend(other.protocol);
    out.effects.extend(other.effects);
    out.diagnostics.extend(other.diagnostics);
    out.accepted &= other.accepted;
}
fn reject_prepared<P: PositionPort>(
    session: &mut Session<P>,
    current: &mut CurrentSearch,
    prepared: SessionResult<P::Snapshot>,
    err: ContractError,
) -> ContractSessionOutcome<P::Snapshot> {
    let mut out = if err.code == ErrorCode::Expired {
        expire_current(session, current, err)
    } else {
        rejected(err)
    };
    out.session.diagnostics.extend(
        prepared
            .diagnostics
            .into_iter()
            .filter(|d| d.code != "AcceptanceClosed"),
    );
    out
}
fn expire_current<P: PositionPort>(
    session: &mut Session<P>,
    current: &mut CurrentSearch,
    err: ContractError,
) -> ContractSessionOutcome<P::Snapshot> {
    current.cancellation.cancel();
    let mut out = if current.infinite && current.binding.budget_kind() == BudgetKind::ResourceWall {
        session.complete(
            current.binding.ticket(),
            SearchCompletion::Completed { bestmove: None },
        )
    } else {
        session.expire(current.binding.ticket())
    };
    current.work_completed = true;
    let code = if current.binding.budget_kind() == BudgetKind::ResourceWall {
        "ResourceWallLimit"
    } else {
        "SearchExpired"
    };
    out.diagnostics.push(Diagnostic {
        code,
        message: "common hard result boundary closed search work".into(),
    });
    let mut outcome = out.into();
    attach_error(&mut outcome, err);
    outcome
}
