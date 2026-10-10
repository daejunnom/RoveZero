//! Owned continuation of the CPU traversal. Every charged node has an explicit
//! entry stage; resuming starts at the saved stage, never at iteration entry.
//! The engine retains its exclusive TT/history owner while this task is paused.

use super::*;
use crate::cpu_value::{CPU_ACCUMULATOR_WIDTH, CpuAccumulatorUndo};
use rz_position::{PieceChange, RuleMoveDelta, UndoToken};

const MAX_FRAMES: usize = MAX_DEPTH as usize + MAX_QUIESCENCE_PLY as usize + 1;
const MAX_EXCHANGE_FRAMES: usize = 32;

pub(super) struct PausedTask {
    root: PositionIdentity,
    search_conditions: String,
    value_identity: CpuValueIdentity,
    root_moves: Option<Vec<BoardMove>>,
    legal: Vec<BoardMove>,
    work: Position,
    value: Option<CpuAccumulator>,
    frames: Vec<Frame>,
    returned: Option<NodeValue>,
    best: Vec<BoardMove>,
    score: i32,
    completed_depth: u16,
    target_depth: u16,
    iteration_depth: u16,
    window: (i32, i32),
    full_window_retry: bool,
    cumulative: CpuWork,
}

impl PausedTask {
    fn new(
        engine: &CpuEngine,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        legal: Vec<BoardMove>,
        limits: CpuLimits,
        resume: Option<&CpuResumeToken>,
    ) -> Result<Self, CpuError> {
        let value = engine
            .evaluator
            .initialize(position)
            .map_err(CpuError::Value)?;
        let score = if let Some(token) = resume {
            token.score
        } else {
            let score = engine
                .evaluator
                .score(&value, position)
                .map_err(CpuError::Value)?;
            if !(-CPU_FRONTIER_SCORE_LIMIT..=CPU_FRONTIER_SCORE_LIMIT).contains(&score) {
                return Err(CpuError::Value(
                    CpuValueError::ScoreOutsideFrontierNamespace,
                ));
            }
            score
        };
        let best = resume.map_or_else(
            || legal.first().copied().into_iter().collect(),
            |token| token.pv.clone(),
        );
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(MAX_FRAMES)
            .map_err(|_| CpuError::Allocation)?;
        let task = Self {
            root: position.position_identity(),
            search_conditions: engine.search_conditions(),
            value_identity: (*engine.value_identity).clone(),
            root_moves: root_moves.map(<[BoardMove]>::to_vec),
            legal,
            work: position.clone(),
            value: Some(value),
            frames,
            returned: None,
            best,
            score,
            completed_depth: resume.map_or(0, |token| token.completed_depth),
            target_depth: limits.max_depth,
            iteration_depth: 0,
            window: (-INFINITY, INFINITY),
            full_window_retry: false,
            cumulative: CpuWork::default(),
        };
        task.check_memory()?;
        Ok(task)
    }

    /// Conservative retained traversal charge. Shared Rules prefixes count in
    /// each live exchange owner, allocator metadata is reserved per allocation,
    /// and the opaque accumulator reserves its fixed exact192 backing as well.
    /// The configured TT/history and frozen evaluator already belong to the
    /// engine; no copy of either is made or charged as a second paused task.
    pub(super) fn retained_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        for charge in [
            vector_bytes(&self.frames),
            vector_bytes(&self.legal),
            vector_bytes(&self.best),
            self.root_moves.as_ref().map_or(0, vector_bytes),
            self.work
                .snapshot()
                .retained_history_bytes()
                .unwrap_or(usize::MAX),
            self.search_conditions.capacity(),
            2 * CPU_ACCUMULATOR_WIDTH * (4 + 24),
            std::mem::size_of::<CpuAccumulator>(),
            4096, // bounded value namespace/control/Arc and allocator allowance
        ] {
            bytes = bytes.saturating_add(charge);
        }
        for frame in &self.frames {
            bytes = bytes.saturating_add(frame.retained_bytes());
        }
        if let Some(value) = &self.returned {
            bytes = bytes.saturating_add(vector_bytes(&value.pv));
        }
        bytes
    }

    fn check_memory(&self) -> Result<(), CpuError> {
        let retained_bytes = self.retained_bytes();
        if retained_bytes > CPU_PAUSED_STACK_MAX_BYTES {
            Err(CpuError::PausedStackMemoryLimit {
                retained_bytes,
                maximum_bytes: CPU_PAUSED_STACK_MAX_BYTES,
            })
        } else {
            Ok(())
        }
    }

    fn start_iteration(&mut self, full_window: bool) -> Result<(), CpuError> {
        if !full_window {
            self.iteration_depth = self.completed_depth + 1;
            self.full_window_retry = false;
            self.window = if self.completed_depth == 0 || self.score.abs() >= MATE_THRESHOLD {
                (-INFINITY, INFINITY)
            } else {
                (
                    self.score - ASPIRATION_WINDOW,
                    self.score + ASPIRATION_WINDOW,
                )
            };
        } else {
            self.full_window_retry = true;
        }
        let window = if full_window {
            (-INFINITY, INFINITY)
        } else {
            self.window
        };
        let mut frame = Frame::new(Kind::Root, self.iteration_depth, 0, window.0, window.1);
        frame.moves = self.legal.clone();
        frame.previous_best = self.best.first().copied();
        self.frames.push(frame);
        self.check_memory()
    }
}

fn vector_bytes<T>(values: &Vec<T>) -> usize {
    values
        .capacity()
        .saturating_mul(std::mem::size_of::<T>())
        .saturating_add(32)
}

fn undo_bytes(undo: &UndoToken) -> usize {
    vector_bytes(&undo.delta().removals) + vector_bytes(&undo.delta().additions) + 32
}

pub(super) fn token_is_current(engine: &CpuEngine, token: &CpuResumeToken) -> bool {
    let Some(handle) = token.paused_stack.as_ref() else {
        return false;
    };
    let Some(task) = engine.paused.as_ref() else {
        return false;
    };
    let retained_bytes = task.retained_bytes();
    engine.resume_policy == CpuResumePolicy::PausedStack
        && Arc::ptr_eq(&handle.owner, &engine.stack_owner)
        && handle.serial == engine.stack_serial
        && task.root == token.root
        && task.root_moves == token.root_moves
        && task.completed_depth == token.completed_depth
        && task.best == token.pv
        && task.score == token.score
        && task.cumulative == handle.cumulative
        && retained_bytes == handle.retained_bytes
        && retained_bytes <= CPU_PAUSED_STACK_MAX_BYTES
        && task.search_conditions == token.search_conditions
        && token.search_conditions == engine.search_conditions()
        && task.value_identity == token.value_identity
        && token.value_identity == *engine.value_identity
        && engine.evaluator.identity() == engine.value_identity.as_ref()
        && token.profile == engine.config.profile
        && token.quiescence_ply == engine.config.quiescence_ply
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    engine: &mut CpuEngine,
    position: &Position,
    root_moves: Option<&[BoardMove]>,
    limits: CpuLimits,
    cancel: &AtomicBool,
    resume: Option<&CpuResumeToken>,
    observer: &mut dyn FnMut(CpuIterationProgress),
) -> Result<CpuReport, CpuError> {
    engine.last_attempt_work = Some(CpuWork::default());
    if resume.is_none() {
        engine.invalidate_paused_stack();
    }
    if let Err(error) = engine.validate_limits(position, limits) {
        if matches!(error, CpuError::InvalidConfig(_)) {
            engine.observe_paused_stack_rejection(resume);
        }
        if matches!(error, CpuError::InvalidConfig(_)) || cancel.load(Ordering::Acquire) {
            engine.invalidate_paused_stack();
        }
        return Err(error);
    }
    if engine.paused.as_ref().is_some_and(|task| {
        task.search_conditions != engine.search_conditions()
            || task.value_identity != *engine.value_identity
    }) {
        engine.invalidate_paused_stack();
        engine.observe_paused_stack_rejection(resume);
        return Err(CpuError::ResumeMismatch("retained CPU namespace changed"));
    }
    let started = Instant::now();
    if let Some(token) = resume {
        if token.root != position.position_identity() {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "Rules state or full known history changed",
            ));
        }
        if token.profile != engine.config.profile
            || token.quiescence_ply != engine.config.quiescence_ply
            || token.value_identity != *engine.value_identity
            || token.search_conditions != engine.search_conditions()
            || limits.max_depth < token.completed_depth
        {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "CPU search conditions or value namespace changed",
            ));
        }
    }
    let mut task = if let Some(handle) = resume.and_then(|token| token.paused_stack.as_ref()) {
        if !Arc::ptr_eq(&handle.owner, &engine.stack_owner) || handle.serial != engine.stack_serial
        {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "paused stack owner or one-use serial is stale",
            ));
        }
        let Some(retained) = engine.paused.as_ref() else {
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "paused stack was consumed or invalidated",
            ));
        };
        let token = resume.expect("handle came from token");
        if retained.root != token.root
            || retained.root_moves != token.root_moves
            || retained.completed_depth != token.completed_depth
            || retained.best != token.pv
            || retained.score != token.score
            || retained.cumulative != handle.cumulative
            || retained.retained_bytes() != handle.retained_bytes
            || retained.target_depth != limits.max_depth
        {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "paused task metadata, restrictions or target depth changed",
            ));
        }
        let task = engine.paused.take().expect("validated live task");
        engine.observe_paused_stack_resumed();
        task
    } else {
        // Starting a new traversal from a historical completed result also
        // invalidates any unrelated paused owner.
        engine.invalidate_paused_stack();
        let view = position.ordered_legal_moves();
        let status = position.play_status_from_view(&view)?;
        let mut legal = view.moves().to_vec();
        if let Some(requested) = root_moves {
            if requested.is_empty() || requested.len() > MAX_ROOT_MOVES {
                return Err(CpuError::InvalidRootMoves(
                    "root restriction must contain 1..=256 moves",
                ));
            }
            for (index, mv) in requested.iter().enumerate() {
                if !legal.contains(mv) || requested[..index].contains(mv) {
                    return Err(CpuError::InvalidRootMoves(
                        "root restriction contains illegal or duplicate moves",
                    ));
                }
            }
            legal.retain(|mv| requested.contains(mv));
        }
        if let PlayStatus::Terminal { reason, winner } = status {
            return Ok(CpuReport {
                best_move: None,
                pv: Vec::new(),
                score: terminal_score(position, winner, 0),
                completed_depth: 0,
                nodes: 0,
                quiescence_nodes: 0,
                tt_hits: 0,
                completion: CpuCompletion::Terminal(reason),
                score_scope: CpuScoreScope::RulesTerminal,
                profile: engine.config.profile,
                score_provenance: "rz-position/rules-terminal",
                value_identity: (*engine.value_identity).clone(),
                search_version: engine.search_identity(),
                root_restricted: root_moves.is_some(),
                elapsed: started.elapsed(),
                reused_completed_depth: 0,
                resume: None,
            });
        }
        PausedTask::new(engine, position, root_moves, legal, limits, resume)?
    };
    if let Err(error) = task.check_memory() {
        engine.invalidate_paused_stack();
        return Err(error);
    }
    let reused_completed_depth = task.completed_depth;
    let mut control = Control {
        limits,
        cancellation: cancel,
        nodes: 0,
        quiescence_nodes: 0,
        tt_hits: 0,
        value: task.value.take().expect("paused task owns its accumulator"),
    };
    if resume.is_some_and(CpuResumeToken::is_paused_stack) {
        // Observe actual invocation counters before any new work. The saved
        // task.cumulative diagnostics never initialize these counters.
        engine.observe_paused_stack_initial_work(control.actual_work());
    }
    observer(CpuIterationProgress {
        best_move: task.best.first().copied(),
        score: task.score,
        completed_depth: task.completed_depth,
        nodes: 0,
    });
    let result = drive(engine, &mut task, &mut control, observer);
    let work = control.actual_work();
    task.value = Some(control.value);
    engine.last_attempt_work = Some(work);
    let cumulative =
        (|| {
            Ok(CpuWork {
                nodes: task.cumulative.nodes.checked_add(work.nodes).ok_or(
                    CpuError::InvalidLimits("cumulative paused CPU nodes exceed u64"),
                )?,
                quiescence_nodes: task
                    .cumulative
                    .quiescence_nodes
                    .checked_add(work.quiescence_nodes)
                    .ok_or(CpuError::InvalidLimits(
                        "cumulative paused CPU qnodes exceed u64",
                    ))?,
                tt_hits: task.cumulative.tt_hits.checked_add(work.tt_hits).ok_or(
                    CpuError::InvalidLimits("cumulative paused CPU TT hits exceed u64"),
                )?,
            })
        })();
    task.cumulative = match cumulative {
        Ok(cumulative) => cumulative,
        Err(error) => {
            engine.invalidate_paused_stack();
            return Err(error);
        }
    };
    let completion = match result {
        Ok(()) => CpuCompletion::DepthLimit,
        Err(Abort::Stop(reason)) => reason,
        Err(Abort::Error(error)) => {
            engine.invalidate_paused_stack();
            return Err(error);
        }
    };
    let can_pause = matches!(
        completion,
        CpuCompletion::NodeLimit | CpuCompletion::Deadline
    );
    let mut token = None;
    if can_pause {
        if let Err(error) = task.check_memory() {
            engine.invalidate_paused_stack();
            return Err(error);
        }
        engine.stack_serial = engine
            .stack_serial
            .checked_add(1)
            .ok_or(CpuError::Unsupported(
                "paused stack one-use serial exhausted",
            ))?;
        token = Some(make_token(
            engine,
            &task,
            Some(PausedStackToken {
                owner: Arc::clone(&engine.stack_owner),
                serial: engine.stack_serial,
                cumulative: task.cumulative,
                retained_bytes: task.retained_bytes(),
            }),
        ));
    } else if completion == CpuCompletion::DepthLimit && task.completed_depth > 0 {
        token = Some(make_token(engine, &task, None));
    }
    let report = CpuReport {
        best_move: task.best.first().copied(),
        pv: task.best.clone(),
        score: task.score,
        completed_depth: task.completed_depth,
        nodes: work.nodes,
        quiescence_nodes: work.quiescence_nodes,
        tt_hits: work.tt_hits,
        completion,
        score_scope: if task.completed_depth == 0 {
            CpuScoreScope::FrontierOnly
        } else {
            CpuScoreScope::CompletedIteration
        },
        profile: engine.config.profile,
        score_provenance: engine.evaluator.provenance(),
        value_identity: (*engine.value_identity).clone(),
        search_version: engine.search_identity(),
        root_restricted: task.root_moves.is_some(),
        elapsed: started.elapsed(),
        reused_completed_depth,
        resume: token,
    };
    if can_pause {
        engine.paused = Some(task);
        engine.observe_paused_stack_created();
    } else {
        engine.invalidate_paused_stack();
    }
    Ok(report)
}

fn make_token(
    engine: &CpuEngine,
    task: &PausedTask,
    paused_stack: Option<PausedStackToken>,
) -> CpuResumeToken {
    CpuResumeToken {
        root: task.root.clone(),
        root_moves: task.root_moves.clone(),
        profile: engine.config.profile,
        quiescence_ply: engine.config.quiescence_ply,
        value_identity: (*engine.value_identity).clone(),
        search_conditions: engine.search_conditions(),
        completed_depth: task.completed_depth,
        score: task.score,
        pv: task.best.clone(),
        paused_stack,
    }
}

fn drive(
    engine: &mut CpuEngine,
    task: &mut PausedTask,
    control: &mut Control<'_>,
    observer: &mut dyn FnMut(CpuIterationProgress),
) -> Result<(), Abort> {
    loop {
        control.check_stop()?;
        if task.frames.is_empty() {
            if let Some(value) = task.returned.take() {
                if !task.full_window_retry
                    && (value.score <= task.window.0 || value.score >= task.window.1)
                {
                    task.start_iteration(true).map_err(Abort::Error)?;
                    continue;
                }
                if task.work.position_identity() != task.root {
                    return Err(Abort::Error(CpuError::ResumeMismatch(
                        "completed traversal did not restore Rules root",
                    )));
                }
                task.score = value.score;
                task.best = value.pv;
                task.completed_depth = task.iteration_depth;
                observer(CpuIterationProgress {
                    best_move: task.best.first().copied(),
                    score: task.score,
                    completed_depth: task.completed_depth,
                    nodes: control.nodes,
                });
                continue;
            }
            if task.completed_depth >= task.target_depth {
                return Ok(());
            }
            task.start_iteration(false).map_err(Abort::Error)?;
        }
        let mut frame = task.frames.pop().expect("iteration has a frame");
        let action = frame.advance(engine, &mut task.work, control, &mut task.returned);
        match action {
            Ok(Action::Continue) => task.frames.push(frame),
            Ok(Action::Push(child)) => {
                task.frames.push(frame);
                if task.frames.len() >= MAX_FRAMES {
                    return Err(Abort::Error(CpuError::Unsupported(
                        "paused search frame bound exhausted",
                    )));
                }
                task.frames.push(child);
            }
            Ok(Action::Return(value)) => task.returned = Some(value),
            Err(error) => {
                task.frames.push(frame);
                return Err(error);
            }
        }
        task.check_memory().map_err(Abort::Error)?;
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Kind {
    Root,
    Negamax,
    Quiescence,
}

enum Stage {
    Enter,
    Ordering(OrderingState),
    Select,
    Await(ActiveMove),
    Finish,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ChildPhase {
    Scout,
    Full,
    Quiescence,
}

struct ActiveMove {
    mv: BoardMove,
    undo: Option<UndoToken>,
    value_undo: Option<CpuAccumulatorUndo>,
    phase: ChildPhase,
    quiet: bool,
    side: usize,
}

struct Frame {
    kind: Kind,
    depth: u16,
    ply: u16,
    alpha: i32,
    beta: i32,
    alpha_original: i32,
    previous_best: Option<BoardMove>,
    moves: Vec<BoardMove>,
    cursor: usize,
    best: NodeValue,
    tt: Option<(u64, PositionIdentity)>,
    stage: Stage,
}

// A transient result in the nonrecursive driver. Keeping the child inline
// avoids an allocation per searched edge; retained frames live in the one
// fallibly reserved and byte-charged task Vec.
#[allow(clippy::large_enum_variant)]
enum Action {
    Continue,
    Push(Frame),
    Return(NodeValue),
}

impl Frame {
    fn new(kind: Kind, depth: u16, ply: u16, alpha: i32, beta: i32) -> Self {
        Self {
            kind,
            depth,
            ply,
            alpha,
            beta,
            alpha_original: alpha,
            previous_best: None,
            moves: Vec::new(),
            cursor: 0,
            best: NodeValue {
                score: -INFINITY,
                pv: Vec::new(),
            },
            tt: None,
            stage: Stage::Enter,
        }
    }

    fn child(engine: &CpuEngine, depth: u16, ply: u16, alpha: i32, beta: i32) -> Self {
        if depth == 0 {
            Self::new(
                Kind::Quiescence,
                engine.config.quiescence_ply,
                ply,
                alpha,
                beta,
            )
        } else {
            Self::new(Kind::Negamax, depth, ply, alpha, beta)
        }
    }

    fn retained_bytes(&self) -> usize {
        let own = vector_bytes(&self.moves).saturating_add(vector_bytes(&self.best.pv));
        own.saturating_add(match &self.stage {
            Stage::Ordering(ordering) => ordering.retained_bytes(),
            Stage::Await(active) => active.undo.as_ref().map_or(0, undo_bytes),
            _ => 0,
        })
    }

    fn advance(
        &mut self,
        engine: &mut CpuEngine,
        position: &mut Position,
        control: &mut Control<'_>,
        returned: &mut Option<NodeValue>,
    ) -> Result<Action, Abort> {
        match &mut self.stage {
            Stage::Enter => self.enter(engine, position, control),
            Stage::Ordering(ordering) => {
                if let Some(moves) =
                    ordering.advance(engine, position, &self.moves, self.previous_best, control)?
                {
                    self.moves = moves;
                    self.stage = Stage::Select;
                }
                Ok(Action::Continue)
            }
            Stage::Select => {
                if self.cursor >= self.moves.len() {
                    self.stage = Stage::Finish;
                    return Ok(Action::Continue);
                }
                let mv = self.moves[self.cursor];
                let quiet = !is_tactical(position, mv);
                let side = position.side_to_move() as usize;
                let undo = position.make_move(mv)?;
                let value_undo = match engine
                    .evaluator
                    .apply_delta(&mut control.value, undo.delta())
                {
                    Ok(value) => value,
                    Err(error) => {
                        position.unmake(undo)?;
                        return Err(error.into());
                    }
                };
                let phase = if self.kind == Kind::Quiescence {
                    ChildPhase::Quiescence
                } else if self.cursor == 0 {
                    ChildPhase::Full
                } else {
                    ChildPhase::Scout
                };
                let child = if self.kind == Kind::Quiescence {
                    Frame::new(
                        Kind::Quiescence,
                        self.depth - 1,
                        self.ply + 1,
                        -self.beta,
                        -self.alpha,
                    )
                } else {
                    let (alpha, beta) = if phase == ChildPhase::Full {
                        (-self.beta, -self.alpha)
                    } else {
                        (-self.alpha - 1, -self.alpha)
                    };
                    Frame::child(engine, self.depth - 1, self.ply + 1, alpha, beta)
                };
                self.cursor += 1;
                self.stage = Stage::Await(ActiveMove {
                    mv,
                    undo: Some(undo),
                    value_undo: Some(value_undo),
                    phase,
                    quiet,
                    side,
                });
                Ok(Action::Push(child))
            }
            Stage::Await(active) => {
                let child = returned
                    .take()
                    .ok_or(Abort::Error(CpuError::ResumeMismatch(
                        "missing completed child frame",
                    )))?;
                let score = -child.score;
                if active.phase == ChildPhase::Scout && score > self.alpha && score < self.beta {
                    active.phase = ChildPhase::Full;
                    return Ok(Action::Push(Frame::child(
                        engine,
                        self.depth - 1,
                        self.ply + 1,
                        -self.beta,
                        -self.alpha,
                    )));
                }
                let mv = active.mv;
                position.unmake(active.undo.take().expect("active move owns Rules undo"))?;
                engine.evaluator.restore(
                    &mut control.value,
                    active
                        .value_undo
                        .take()
                        .expect("active move owns value undo"),
                    position,
                )?;
                if score > self.best.score {
                    self.best.score = score;
                    self.best.pv.clear();
                    self.best.pv.push(mv);
                    self.best.pv.extend(child.pv);
                }
                self.alpha = self.alpha.max(score);
                if self.alpha >= self.beta {
                    if self.kind == Kind::Negamax && active.quiet {
                        let history = &mut engine.history[active.side][mv.from.index() as usize]
                            [mv.to.index() as usize];
                        *history = history
                            .saturating_add(i32::from(self.depth).pow(2))
                            .min(20_000);
                    }
                    self.stage = Stage::Finish;
                } else {
                    self.stage = Stage::Select;
                }
                Ok(Action::Continue)
            }
            Stage::Finish => {
                control.check_stop()?;
                if let Some((hash, identity)) = self.tt.take() {
                    let bound = if self.best.score <= self.alpha_original {
                        Bound::Upper
                    } else if self.best.score >= self.beta {
                        Bound::Lower
                    } else {
                        Bound::Exact
                    };
                    engine.store(TtEntry {
                        hash,
                        identity,
                        profile: engine.config.profile,
                        value_identity: Arc::clone(&engine.value_identity),
                        depth: self.depth,
                        score: to_tt_score(self.best.score, self.ply),
                        bound,
                        best_move: self.best.pv.first().copied(),
                    });
                }
                Ok(Action::Return(std::mem::replace(
                    &mut self.best,
                    NodeValue {
                        score: -INFINITY,
                        pv: Vec::new(),
                    },
                )))
            }
        }
    }

    fn enter(
        &mut self,
        engine: &mut CpuEngine,
        position: &Position,
        control: &mut Control<'_>,
    ) -> Result<Action, Abort> {
        control.visit(self.kind == Kind::Quiescence)?;
        if self.kind != Kind::Root {
            let view = position.ordered_legal_moves();
            if let PlayStatus::Terminal { winner, .. } = position.play_status_from_view(&view)? {
                return Ok(Action::Return(NodeValue {
                    score: terminal_score(position, winner, self.ply),
                    pv: Vec::new(),
                }));
            }
            self.moves = view.moves().to_vec();
            if self.kind == Kind::Negamax {
                let hash = tt_hash(position);
                let identity = position.position_identity();
                if let Some(entry) = engine.probe(hash, &identity) {
                    control.tt_hits += 1;
                    let score = from_tt_score(entry.score, self.ply);
                    if entry.depth == self.depth
                        && (entry.bound == Bound::Exact
                            || (entry.bound == Bound::Lower && score >= self.beta)
                            || (entry.bound == Bound::Upper && score <= self.alpha))
                    {
                        return Ok(Action::Return(NodeValue {
                            score,
                            pv: entry.best_move.into_iter().collect(),
                        }));
                    }
                    self.previous_best = entry.best_move;
                }
                self.tt = Some((hash, identity));
            } else {
                let in_check = position.in_check();
                if in_check && self.depth == 0 {
                    return Err(Abort::Stop(CpuCompletion::QuiescenceLimit));
                }
                if !in_check {
                    self.best.score = engine.frontier_score(control, position)?;
                    if self.best.score >= self.beta || self.depth == 0 {
                        return Ok(Action::Return(std::mem::replace(
                            &mut self.best,
                            NodeValue {
                                score: -INFINITY,
                                pv: Vec::new(),
                            },
                        )));
                    }
                    self.alpha = self.alpha.max(self.best.score);
                    self.moves.retain(|mv| is_tactical(position, *mv));
                }
            }
        }
        if self.moves.len() > MAX_ROOT_MOVES {
            return Err(Abort::Error(CpuError::Unsupported(
                "Rules move count exceeds bounded paused frame",
            )));
        }
        self.stage = Stage::Ordering(OrderingState::new(self.moves.len())?);
        Ok(Action::Continue)
    }
}

struct OrderingState {
    scored: Vec<(BoardMove, (u8, i32))>,
    cursor: usize,
    exchange_positions: u32,
    exchange: Option<Exchange>,
}

impl OrderingState {
    fn new(moves: usize) -> Result<Self, Abort> {
        let mut scored = Vec::new();
        scored
            .try_reserve_exact(moves)
            .map_err(|_| Abort::Error(CpuError::Allocation))?;
        Ok(Self {
            scored,
            cursor: 0,
            exchange_positions: 0,
            exchange: None,
        })
    }

    fn retained_bytes(&self) -> usize {
        vector_bytes(&self.scored)
            .saturating_add(self.exchange.as_ref().map_or(0, Exchange::retained_bytes))
    }

    fn advance(
        &mut self,
        engine: &CpuEngine,
        position: &Position,
        moves: &[BoardMove],
        best: Option<BoardMove>,
        control: &mut Control<'_>,
    ) -> Result<Option<Vec<BoardMove>>, Abort> {
        if engine.ordering == CpuOrderingPolicy::LegacyMvvLvaV1 {
            let mut ordered = moves.to_vec();
            ordered.sort_by_key(|mv| std::cmp::Reverse(engine.order_score(position, *mv, best)));
            return Ok(Some(ordered));
        }
        if self.cursor == moves.len() {
            control.check_stop()?;
            self.scored.sort_by_key(|(_, key)| std::cmp::Reverse(*key));
            return Ok(Some(self.scored.iter().map(|(mv, _)| *mv).collect()));
        }
        let mv = moves[self.cursor];
        let key = if best == Some(mv) {
            Some((3, 0))
        } else if !is_tactical(position, mv) {
            Some((1, engine.order_score(position, mv, None)))
        } else {
            if self.exchange.is_none() {
                visit_exchange(control, &mut self.exchange_positions)?;
                self.exchange = Some(Exchange::new(position, mv)?);
            }
            let score = self
                .exchange
                .as_mut()
                .expect("created exchange")
                .advance(control, &mut self.exchange_positions)?;
            if let Some(score) = score {
                self.exchange = None;
                Some((if score >= 0 { 2 } else { 0 }, score))
            } else {
                None
            }
        };
        if let Some(key) = key {
            self.scored.push((mv, key));
            self.cursor += 1;
        }
        Ok(None)
    }
}

fn visit_exchange(control: &mut Control<'_>, positions: &mut u32) -> Result<(), Abort> {
    control.check_stop()?;
    if *positions >= MAX_SEE_POSITIONS_PER_ORDER {
        return Err(Abort::Error(CpuError::Unsupported(
            "legal SEE ordering exchange position budget exhausted",
        )));
    }
    control.visit(false)?;
    *positions += 1;
    Ok(())
}

struct Exchange {
    position: Position,
    target: Square,
    initial_gain: i32,
    initial_undo: Option<UndoToken>,
    frames: Vec<ExchangeFrame>,
    returned: Option<i32>,
}

struct ExchangeFrame {
    plies: u16,
    entered: bool,
    moves: Vec<BoardMove>,
    cursor: usize,
    best: i32,
    active: Option<(UndoToken, i32)>,
}

impl ExchangeFrame {
    fn new(plies: u16) -> Self {
        Self {
            plies,
            entered: false,
            moves: Vec::new(),
            cursor: 0,
            best: 0,
            active: None,
        }
    }
}

impl Exchange {
    fn new(position: &Position, mv: BoardMove) -> Result<Self, Abort> {
        let mut position = position.clone();
        let mover = position.side_to_move();
        let undo = position.make_move(mv)?;
        let gain = exchange_gain(undo.delta(), mover);
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(MAX_EXCHANGE_FRAMES)
            .map_err(|_| Abort::Error(CpuError::Allocation))?;
        frames.push(ExchangeFrame::new(1));
        Ok(Self {
            position,
            target: mv.to,
            initial_gain: gain,
            initial_undo: Some(undo),
            frames,
            returned: None,
        })
    }

    fn retained_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        for charge in [
            vector_bytes(&self.frames),
            self.position
                .snapshot()
                .retained_history_bytes()
                .unwrap_or(usize::MAX),
            self.initial_undo.as_ref().map_or(0, undo_bytes),
        ] {
            bytes = bytes.saturating_add(charge);
        }
        for frame in &self.frames {
            bytes = bytes.saturating_add(vector_bytes(&frame.moves));
            if let Some((undo, _)) = &frame.active {
                bytes = bytes.saturating_add(undo_bytes(undo));
            }
        }
        bytes
    }

    fn advance(
        &mut self,
        control: &mut Control<'_>,
        positions: &mut u32,
    ) -> Result<Option<i32>, Abort> {
        if self.frames.is_empty() {
            let reply = self.returned.take().expect("exchange root completed");
            self.position.unmake(
                self.initial_undo
                    .take()
                    .expect("exchange owns initial undo"),
            )?;
            return Ok(Some(self.initial_gain - reply));
        }
        let frame = self.frames.last_mut().expect("live exchange frame");
        if !frame.entered {
            visit_exchange(control, positions)?;
            frame.moves = self
                .position
                .ordered_legal_moves()
                .moves()
                .iter()
                .filter(|mv| mv.to == self.target)
                .copied()
                .collect();
            frame.entered = true;
            return Ok(None);
        }
        if let Some((undo, gain)) = frame.active.take() {
            let reply = self.returned.take().expect("completed exchange child");
            self.position.unmake(undo)?;
            frame.best = frame.best.max(gain - reply);
            return Ok(None);
        }
        if frame.cursor == frame.moves.len() {
            self.returned = Some(frame.best);
            self.frames.pop();
            return Ok(None);
        }
        if frame.plies >= MAX_EXCHANGE_FRAMES as u16 {
            return Err(Abort::Error(CpuError::Unsupported(
                "legal SEE exchange ply bound exhausted",
            )));
        }
        let mv = frame.moves[frame.cursor];
        let mover = self.position.side_to_move();
        let undo = self.position.make_move(mv)?;
        if !undo.delta().kind.capture {
            self.position.unmake(undo)?;
            return Err(Abort::Error(CpuError::Unsupported(
                "legal SEE target recapture was not a capture",
            )));
        }
        let gain = exchange_gain(undo.delta(), mover);
        frame.cursor += 1;
        let plies = frame.plies + 1;
        frame.active = Some((undo, gain));
        self.frames.push(ExchangeFrame::new(plies));
        Ok(None)
    }
}

fn exchange_gain(delta: &RuleMoveDelta, mover: Color) -> i32 {
    let value = |change: &PieceChange| {
        let material = material(change.piece.kind);
        if change.piece.color == mover {
            material
        } else {
            -material
        }
    };
    delta.additions.iter().map(value).sum::<i32>() - delta.removals.iter().map(value).sum::<i32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_value::{CpuFloatCheckpoint, CpuFloatValue};
    use std::sync::atomic::AtomicU64;

    fn config(tt_entries: usize) -> CpuConfig {
        CpuConfig {
            tt_entries,
            max_depth: 4,
            quiescence_ply: 4,
            ..CpuConfig::default()
        }
    }

    fn limits(nodes: u64) -> CpuLimits {
        CpuLimits {
            max_depth: 2,
            max_nodes: nodes,
            deadline: None,
        }
    }

    fn cpu(ordering: CpuOrderingPolicy, tt_entries: usize) -> CpuEngine {
        CpuEngine::with_evaluator_ordering_and_resume_policy(
            config(tt_entries),
            Arc::new(BootstrapCpuValue::default()),
            ordering,
            CpuResumePolicy::PausedStack,
        )
        .unwrap()
    }

    fn add(total: &mut CpuWork, part: CpuWork) {
        total.nodes += part.nodes;
        total.quiescence_nodes += part.quiescence_nodes;
        total.tt_hits += part.tt_hits;
    }

    fn run_slices(
        engine: &mut CpuEngine,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        interval: u64,
    ) -> (CpuReport, CpuWork, bool) {
        let before = position.snapshot();
        let owner = engine.paused_stack_snapshot().unwrap().owner_id;
        let cancel = AtomicBool::new(false);
        let mut report = if let Some(moves) = root_moves {
            engine
                .analyze_root_moves(position, moves, limits(interval), &cancel)
                .unwrap()
        } else {
            engine.analyze(position, limits(interval), &cancel).unwrap()
        };
        let mut total = CpuWork::default();
        let mut saw_exchange = false;
        for _ in 0..100_000 {
            assert_eq!(engine.last_attempt_work().unwrap().nodes, report.nodes);
            assert!(report.nodes <= interval);
            add(&mut total, engine.last_attempt_work().unwrap());
            assert!(before.same_state(&position.snapshot()));
            if report.completion != CpuCompletion::NodeLimit {
                assert_eq!(report.completion, CpuCompletion::DepthLimit);
                assert!(engine.paused.is_none());
                let snapshot = engine.paused_stack_snapshot().unwrap();
                assert!(snapshot.complete);
                assert_eq!(snapshot.owner_id, owner);
                assert_eq!(snapshot.tokens_retained, 0);
                assert_eq!(snapshot.bytes_current, 0);
                assert_eq!(snapshot.replayed_consumed_work, 0);
                assert!(!snapshot.admission_closed && !snapshot.owner_released);
                return (report, total, saw_exchange);
            }
            let token = report
                .resume
                .take()
                .expect("even depth-zero partial frames resume");
            assert!(token.is_paused_stack());
            assert!(engine.token_is_current(&token));
            assert_eq!(token.cumulative_work(), Some(total));
            assert_eq!(token.paused_stack_bytes(), engine.paused_stack_bytes());
            assert!(token.paused_stack_bytes().unwrap() <= CPU_PAUSED_STACK_MAX_BYTES);
            let snapshot = engine.paused_stack_snapshot().unwrap();
            assert_eq!(snapshot, engine.paused_stack_snapshot().unwrap());
            assert!(snapshot.complete);
            assert_eq!(snapshot.owner_id, owner);
            assert_eq!(snapshot.tokens_retained, 1);
            assert_eq!(snapshot.tokens_peak, 1);
            assert_eq!(
                snapshot.bytes_current,
                token.paused_stack_bytes().unwrap() as u64
            );
            assert!(snapshot.bytes_peak >= snapshot.bytes_current);
            assert!(snapshot.bytes_peak <= CPU_PAUSED_STACK_MAX_BYTES as u64);
            assert_eq!(snapshot.replayed_consumed_work, 0);
            assert_eq!(
                snapshot.tokens_created,
                snapshot.tokens_resumed + snapshot.tokens_invalidated + 1
            );
            let task = engine.paused.as_ref().unwrap();
            let fresh = engine.evaluator.initialize(&task.work).unwrap();
            let live = task.value.as_ref().unwrap();
            assert_eq!(
                fresh.values().map(|side| side.map(f32::to_bits)),
                live.values().map(|side| side.map(f32::to_bits))
            );
            assert_eq!(
                engine.evaluator.score(&fresh, &task.work).unwrap(),
                engine.evaluator.score(live, &task.work).unwrap()
            );
            saw_exchange |= task.frames.iter().any(|frame| {
                matches!(&frame.stage,
                Stage::Ordering(order) if order.exchange.is_some())
            });
            report = engine
                .resume(position, &token, limits(interval), &cancel)
                .unwrap();
            let after = engine.paused_stack_snapshot().unwrap();
            assert_eq!(after.tokens_resumed, snapshot.tokens_resumed + 1);
            assert_eq!(after.tokens_invalidated, snapshot.tokens_invalidated);
            assert_eq!(
                after.event_sequence,
                snapshot.event_sequence + 1 + u64::from(after.tokens_retained)
            );
        }
        panic!("finite fixture did not complete within its declared call bound");
    }

    fn compare_tt_and_history(left: &CpuEngine, right: &CpuEngine) {
        assert_eq!(left.history, right.history);
        assert_eq!(left.tt.len(), right.tt.len());
        for (a, b) in left.tt.iter().zip(&right.tt) {
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(
                        (a.hash, a.depth, a.score, a.bound, a.best_move),
                        (b.hash, b.depth, b.score, b.bound, b.best_move)
                    );
                    assert_eq!(a.identity, b.identity);
                    assert_eq!(a.value_identity, b.value_identity);
                }
                _ => panic!("TT completion changed across pause intervals"),
            }
        }
    }

    #[test]
    fn disposing_a_paused_owner_preserves_tt_history_and_observed_work() {
        let position = Position::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let cancel = AtomicBool::new(false);
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 64);
        let completed = engine.analyze(&position, limits(100_000), &cancel).unwrap();
        assert_eq!(completed.completion, CpuCompletion::DepthLimit);
        assert!(engine.tt.iter().any(Option::is_some));
        assert!(
            engine
                .history
                .iter()
                .flatten()
                .flatten()
                .any(|value| *value != 0)
        );
        let partial = engine
            .analyze(
                &position,
                CpuLimits {
                    max_depth: 3,
                    ..limits(1)
                },
                &cancel,
            )
            .unwrap();
        let token = partial.resume.unwrap();
        let work = engine.last_attempt_work();
        let mut unchanged = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 64);
        unchanged.tt = engine.tt.clone();
        unchanged.history = engine.history.clone();
        assert!(engine.token_is_current(&token));
        assert!(engine.supports_paused_stack_discard());
        let before_discard = engine.paused_stack_snapshot().unwrap();
        assert!(engine.discard_paused_stack());
        let discarded = engine.paused_stack_snapshot().unwrap();
        assert!(discarded.complete);
        assert_eq!(
            discarded.tokens_invalidated,
            before_discard.tokens_invalidated + 1
        );
        assert_eq!(discarded.event_sequence, before_discard.event_sequence + 1);
        assert_eq!(discarded.tokens_retained, 0);
        assert_eq!(discarded.bytes_current, 0);
        assert_eq!(discarded.bytes_peak, before_discard.bytes_peak);
        assert!(!discarded.owner_released && !discarded.admission_closed);
        assert!(!engine.token_is_current(&token));
        assert!(engine.paused_stack_bytes().is_none());
        assert_eq!(engine.last_attempt_work(), work);
        compare_tt_and_history(&engine, &unchanged);
        assert!(!engine.discard_paused_stack());
        assert_eq!(discarded, engine.paused_stack_snapshot().unwrap());
        assert_eq!(engine.last_attempt_work(), work);
        compare_tt_and_history(&engine, &unchanged);
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert_eq!(engine.last_attempt_work().unwrap().nodes, 0);
        let rejected = engine.paused_stack_snapshot().unwrap();
        assert_eq!(
            rejected.stale_context_attempts,
            discarded.stale_context_attempts + 1
        );
        assert_eq!(
            rejected.stale_context_rejections,
            discarded.stale_context_rejections + 1
        );
        assert_eq!(rejected.event_sequence, discarded.event_sequence + 1);
    }

    #[test]
    fn paused_owner_lifetime_survives_new_game_and_counter_overflow_is_unknown() {
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let initial = engine.paused_stack_snapshot().unwrap();
        assert!(initial.complete && initial.owner_id != 0);
        assert_eq!(initial.event_sequence, 0);
        assert_eq!(initial.tokens_peak, 0);
        assert_eq!(initial.bytes_peak, 0);
        assert_eq!(initial, engine.paused_stack_snapshot().unwrap());
        let foreign = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        assert_ne!(
            initial.owner_id,
            foreign.paused_stack_snapshot().unwrap().owner_id
        );
        engine.analyze(&position, limits(1), &cancel).unwrap();
        let paused = engine.paused_stack_snapshot().unwrap();
        assert_eq!(paused.tokens_created, 1);
        engine.clear();
        let reset = engine.paused_stack_snapshot().unwrap();
        assert_eq!(reset.owner_id, initial.owner_id);
        assert_eq!(reset.tokens_created, 1);
        assert_eq!(reset.tokens_invalidated, 1);
        assert_eq!(reset.event_sequence, paused.event_sequence + 1);
        assert_eq!(reset.bytes_peak, paused.bytes_peak);
        assert!(reset.complete);
        assert!(engine.last_attempt_work().is_none());
        engine.clear();
        assert_eq!(reset, engine.paused_stack_snapshot().unwrap());

        // Exhaust the local sequence without mutating the process allocator or
        // another owner. Real create/discard continue, but release stays unknown.
        engine
            .paused_stack_telemetry
            .as_mut()
            .unwrap()
            .event_sequence = u64::MAX;
        let report = engine.analyze(&position, limits(1), &cancel).unwrap();
        assert_eq!(report.nodes, 1);
        let overflow = engine.paused_stack_snapshot().unwrap();
        assert!(!overflow.complete);
        assert_eq!(overflow.event_sequence, u64::MAX);
        assert_eq!(overflow.tokens_created, 2);
        assert_eq!(overflow.tokens_retained, 1);
        assert!(engine.discard_paused_stack());
        let released_frames = engine.paused_stack_snapshot().unwrap();
        assert_eq!(released_frames.owner_id, initial.owner_id);
        assert_eq!(released_frames.tokens_invalidated, 2);
        assert_eq!(released_frames.tokens_retained, 0);
        assert_eq!(released_frames.bytes_current, 0);
        assert!(!released_frames.complete);
        assert!(!released_frames.owner_released);
    }

    #[test]
    fn subdivisions_preserve_recursive_score_pv_depth_work_and_tt() {
        for ordering in [
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuOrderingPolicy::LegalSeeV1,
        ] {
            for tt in [0, 64] {
                for fen in [
                    "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
                    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
                    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                ] {
                    let position = Position::from_fen(fen).unwrap();
                    let mut recursive = CpuEngine::with_ordering(config(tt), ordering).unwrap();
                    let expected = recursive
                        .analyze(&position, limits(100_000), &AtomicBool::new(false))
                        .unwrap();
                    assert_eq!(expected.completion, CpuCompletion::DepthLimit);
                    for interval in [1, 3, 13] {
                        let mut sliced = cpu(ordering, tt);
                        let (actual, total, _) = run_slices(&mut sliced, &position, None, interval);
                        assert_eq!(
                            (actual.score, &actual.pv, actual.completed_depth),
                            (expected.score, &expected.pv, expected.completed_depth)
                        );
                        assert_eq!(total, recursive.last_attempt_work().unwrap());
                        compare_tt_and_history(&sliced, &recursive);
                    }
                }
            }
        }
    }

    #[test]
    fn saved_see_frames_and_root_restrictions_resume_without_recharging() {
        let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let moves = [BoardMove::from_uci("e4d5").unwrap()];
        let mut full = cpu(CpuOrderingPolicy::LegalSeeV1, 16);
        let expected = full
            .analyze_root_moves(&position, &moves, limits(100_000), &AtomicBool::new(false))
            .unwrap();
        let mut sliced = cpu(CpuOrderingPolicy::LegalSeeV1, 16);
        let (actual, work, saw_exchange) = run_slices(&mut sliced, &position, Some(&moves), 1);
        assert!(
            saw_exchange,
            "fixture must actually pause in material-exchange frames"
        );
        assert!(actual.root_restricted);
        assert_eq!(
            (actual.score, actual.pv, actual.completed_depth),
            (expected.score, expected.pv, expected.completed_depth)
        );
        assert_eq!(work, full.last_attempt_work().unwrap());
        compare_tt_and_history(&sliced, &full);
    }

    #[test]
    fn pausing_does_not_repeat_evaluator_or_make_unmake_work_without_charging_it() {
        #[derive(Default)]
        struct Counts {
            initializations: AtomicU64,
            scores: AtomicU64,
            deltas: AtomicU64,
            restores: AtomicU64,
        }
        impl Counts {
            fn observed(&self) -> [u64; 4] {
                [
                    &self.initializations,
                    &self.scores,
                    &self.deltas,
                    &self.restores,
                ]
                .map(|count| count.load(Ordering::Relaxed))
            }
        }
        struct CountedValue {
            base: BootstrapCpuValue,
            counts: Arc<Counts>,
        }
        impl CpuValueEvaluator for CountedValue {
            fn identity(&self) -> &CpuValueIdentity {
                self.base.identity()
            }
            fn provenance(&self) -> &'static str {
                self.base.provenance()
            }
            fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
                self.counts.initializations.fetch_add(1, Ordering::Relaxed);
                self.base.initialize(position)
            }
            fn score(
                &self,
                value: &CpuAccumulator,
                position: &Position,
            ) -> Result<i32, CpuValueError> {
                self.counts.scores.fetch_add(1, Ordering::Relaxed);
                self.base.score(value, position)
            }
            fn apply_delta(
                &self,
                value: &mut CpuAccumulator,
                delta: &RuleMoveDelta,
            ) -> Result<CpuAccumulatorUndo, CpuValueError> {
                self.counts.deltas.fetch_add(1, Ordering::Relaxed);
                self.base.apply_delta(value, delta)
            }
            fn restore(
                &self,
                value: &mut CpuAccumulator,
                undo: CpuAccumulatorUndo,
                position: &Position,
            ) -> Result<(), CpuValueError> {
                self.counts.restores.fetch_add(1, Ordering::Relaxed);
                self.base.restore(value, undo, position)
            }
        }
        let position = Position::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let cancel = AtomicBool::new(false);
        let full_counts = Arc::new(Counts::default());
        let sliced_counts = Arc::new(Counts::default());
        let mut full = CpuEngine::with_evaluator(
            config(16),
            Arc::new(CountedValue {
                base: BootstrapCpuValue::default(),
                counts: Arc::clone(&full_counts),
            }),
        )
        .unwrap();
        let expected = full.analyze(&position, limits(100_000), &cancel).unwrap();
        let mut sliced: Box<dyn CpuSearcher> = Box::new(
            CpuEngine::with_evaluator_ordering_and_resume_policy(
                config(16),
                Arc::new(CountedValue {
                    base: BootstrapCpuValue::default(),
                    counts: Arc::clone(&sliced_counts),
                }),
                CpuOrderingPolicy::LegacyMvvLvaV1,
                CpuResumePolicy::PausedStack,
            )
            .unwrap(),
        );
        let mut report = sliced.analyze(&position, limits(1), &cancel).unwrap();
        let mut total = report.nodes;
        for _ in 0..100_000 {
            if report.completion == CpuCompletion::DepthLimit {
                break;
            }
            assert_eq!(report.completion, CpuCompletion::NodeLimit);
            let token = report.resume.unwrap();
            assert!(sliced.token_is_current(&token));
            report = sliced
                .resume(&position, &token, limits(1), &cancel)
                .unwrap();
            assert!(!sliced.token_is_current(&token));
            total += report.nodes;
        }
        assert_eq!(report.completion, CpuCompletion::DepthLimit);
        assert_eq!(
            (report.score, report.pv, report.completed_depth),
            (expected.score, expected.pv, expected.completed_depth)
        );
        assert_eq!(total, expected.nodes);
        assert_eq!(sliced_counts.observed(), full_counts.observed());
        let observed = sliced_counts.observed();
        assert_eq!(
            observed[0], 1,
            "resume must retain the original accumulator"
        );
        assert!(observed[2] > 0);
        assert_eq!(
            observed[2], observed[3],
            "every actual make must restore once"
        );
    }

    #[test]
    fn resumed_value_failure_consumes_one_token_and_preserves_new_work_without_partial_tt() {
        struct FaultingScore {
            base: BootstrapCpuValue,
            fail: Arc<AtomicBool>,
        }
        impl CpuValueEvaluator for FaultingScore {
            fn identity(&self) -> &CpuValueIdentity {
                self.base.identity()
            }
            fn provenance(&self) -> &'static str {
                self.base.provenance()
            }
            fn initialize(&self, p: &Position) -> Result<CpuAccumulator, CpuValueError> {
                self.base.initialize(p)
            }
            fn score(&self, a: &CpuAccumulator, p: &Position) -> Result<i32, CpuValueError> {
                if self.fail.load(Ordering::Acquire) {
                    return Err(CpuValueError::NonFiniteForward);
                }
                self.base.score(a, p)
            }
            fn apply_delta(
                &self,
                a: &mut CpuAccumulator,
                d: &RuleMoveDelta,
            ) -> Result<CpuAccumulatorUndo, CpuValueError> {
                self.base.apply_delta(a, d)
            }
            fn restore(
                &self,
                a: &mut CpuAccumulator,
                u: CpuAccumulatorUndo,
                p: &Position,
            ) -> Result<(), CpuValueError> {
                self.base.restore(a, u, p)
            }
        }
        let position = Position::startpos();
        let before = position.snapshot();
        let restriction = [BoardMove::from_uci("e2e4").unwrap()];
        let cancel = AtomicBool::new(false);
        let fail = Arc::new(AtomicBool::new(false));
        let mut engine = CpuEngine::with_evaluator_ordering_and_resume_policy(
            config(64),
            Arc::new(FaultingScore {
                base: BootstrapCpuValue::default(),
                fail: Arc::clone(&fail),
            }),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuResumePolicy::PausedStack,
        )
        .unwrap();
        let partial = engine
            .analyze_root_moves(&position, &restriction, limits(1), &cancel)
            .unwrap();
        assert_eq!(partial.nodes, 1);
        let token = partial.resume.unwrap();
        let paused = engine.paused_stack_snapshot().unwrap();
        assert!(engine.tt.iter().all(Option::is_none));
        fail.store(true, Ordering::Release);
        // A one-node resume must reach the injected scorer. Seeding Control
        // with the old charged node would stop before that new work and hide
        // this typed failure behind NodeLimit.
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &cancel),
            Err(CpuError::Value(CpuValueError::NonFiniteForward))
        ));
        let failed_work = engine.last_attempt_work().unwrap();
        assert_eq!(failed_work.nodes, 1);
        assert_eq!(failed_work.quiescence_nodes, 1);
        assert_eq!(failed_work.tt_hits, 0);
        assert_eq!(token.cumulative_work().unwrap().nodes, 1);
        assert!(before.same_state(&position.snapshot()));
        assert!(engine.tt.iter().all(Option::is_none));
        assert!(!engine.token_is_current(&token));
        let failed_owner = engine.paused_stack_snapshot().unwrap();
        assert!(failed_owner.complete);
        assert_eq!(failed_owner.owner_id, paused.owner_id);
        assert_eq!(failed_owner.tokens_created, 1);
        assert_eq!(failed_owner.tokens_resumed, 1);
        assert_eq!(failed_owner.tokens_invalidated, 0);
        assert_eq!(failed_owner.tokens_retained, 0);
        assert_eq!(failed_owner.bytes_current, 0);
        assert_eq!(failed_owner.replayed_consumed_work, 0);
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert_eq!(engine.last_attempt_work(), Some(CpuWork::default()));
        let rejected = engine.paused_stack_snapshot().unwrap();
        assert_eq!(rejected.tokens_resumed, 1);
        assert_eq!(rejected.stale_context_attempts, 1);
        assert_eq!(rejected.stale_context_rejections, 1);

        fail.store(false, Ordering::Release);
        let recovered = engine
            .analyze_root_moves(&position, &restriction, limits(100_000), &cancel)
            .unwrap();
        let mut fresh = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 64);
        let expected = fresh
            .analyze_root_moves(&position, &restriction, limits(100_000), &cancel)
            .unwrap();
        assert_eq!(recovered.completion, CpuCompletion::DepthLimit);
        assert_eq!(
            (recovered.score, recovered.pv, recovered.completed_depth),
            (expected.score, expected.pv, expected.completed_depth)
        );
        assert_eq!(engine.last_attempt_work(), fresh.last_attempt_work());
        compare_tt_and_history(&engine, &fresh);
        assert!(before.same_state(&position.snapshot()));
        assert_eq!(
            engine.paused_stack_snapshot().unwrap().owner_id,
            paused.owner_id
        );
    }

    #[test]
    fn aspiration_failure_and_pvs_research_keep_the_same_work_after_one_node_slices() {
        let position = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        checkpoint.output_bias = 0.075;
        let evaluator: Arc<dyn CpuValueEvaluator> =
            Arc::new(CpuFloatValue::new(checkpoint).unwrap());
        let mut recursive = CpuEngine::with_evaluator(config(16), Arc::clone(&evaluator)).unwrap();
        let expected = recursive
            .analyze(&position, limits(100_000), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(expected.completed_depth, 2);
        assert_eq!(expected.score, 75); // depth one is -75, outside its +/-40 aspiration.
        let mut sliced = CpuEngine::with_evaluator_ordering_and_resume_policy(
            config(16),
            evaluator,
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuResumePolicy::PausedStack,
        )
        .unwrap();
        let (actual, total, _) = run_slices(&mut sliced, &position, None, 1);
        assert_eq!(
            (actual.score, actual.pv, actual.completed_depth),
            (expected.score, expected.pv, expected.completed_depth)
        );
        assert_eq!(total, recursive.last_attempt_work().unwrap());
        compare_tt_and_history(&sliced, &recursive);
    }

    #[test]
    fn exact_nonzero_float_accumulator_restores_special_moves_after_partial_frames() {
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        for (index, value) in checkpoint.feature_weights.iter_mut().enumerate() {
            *value = ((index * 13 % 97) as f32 - 48.0) * 0.0001;
        }
        checkpoint.accumulator_bias.fill(0.4);
        checkpoint.hidden_weights.fill(0.001);
        checkpoint.hidden_bias.fill(0.03);
        checkpoint.output_weights.fill(0.02);
        let model: Arc<dyn CpuValueEvaluator> = Arc::new(CpuFloatValue::new(checkpoint).unwrap());
        for (fen, text) in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
        ] {
            let position = Position::from_fen(fen).unwrap();
            let moves = [BoardMove::from_uci(text).unwrap()];
            let mut sliced = CpuEngine::with_evaluator_ordering_and_resume_policy(
                config(16),
                Arc::clone(&model),
                CpuOrderingPolicy::LegacyMvvLvaV1,
                CpuResumePolicy::PausedStack,
            )
            .unwrap();
            let mut full = CpuEngine::with_evaluator_and_ordering(
                config(16),
                Arc::clone(&model),
                CpuOrderingPolicy::LegacyMvvLvaV1,
            )
            .unwrap();
            let expected = full
                .analyze_root_moves(&position, &moves, limits(100_000), &AtomicBool::new(false))
                .unwrap();
            let (actual, work, _) = run_slices(&mut sliced, &position, Some(&moves), 3);
            assert_eq!(
                (actual.score, actual.pv, actual.completed_depth),
                (expected.score, expected.pv, expected.completed_depth)
            );
            assert_eq!(work, full.last_attempt_work().unwrap());
            // Inspect the fully restored root before the owning task is dropped.
            let legal = vec![moves[0]];
            let mut task = PausedTask::new(
                &sliced,
                &position,
                Some(&moves),
                legal,
                limits(100_000),
                None,
            )
            .unwrap();
            let fresh = model.initialize(&position).unwrap();
            let mut control = Control {
                limits: limits(100_000),
                cancellation: &AtomicBool::new(false),
                nodes: 0,
                quiescence_nodes: 0,
                tt_hits: 0,
                value: task.value.take().unwrap(),
            };
            drive(&mut sliced, &mut task, &mut control, &mut |_| {}).unwrap();
            assert_eq!(task.work.position_identity(), position.position_identity());
            assert_eq!(
                control.value.values().map(|side| side.map(f32::to_bits)),
                fresh.values().map(|side| side.map(f32::to_bits))
            );
        }
    }

    #[test]
    fn one_use_foreign_malformed_and_stale_tokens_are_rejected_atomically() {
        let mut position = Position::startpos();
        position
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        let restriction = [BoardMove::from_uci("e2e4").unwrap()];
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let token = engine
            .analyze_root_moves(&position, &restriction, limits(1), &AtomicBool::new(false))
            .unwrap()
            .resume
            .unwrap();
        assert!(token.is_paused_stack());
        assert!(engine.token_is_current(&token));
        let observed_owner = engine.paused_stack_snapshot().unwrap();
        let observed_work = engine.last_attempt_work();
        assert!(engine.token_is_current(&token));
        assert_eq!(engine.last_attempt_work(), observed_work);
        assert_eq!(observed_owner, engine.paused_stack_snapshot().unwrap());
        let mut malformed = token.clone();
        malformed.pv.clear();
        assert!(!engine.token_is_current(&malformed));
        assert!(matches!(
            engine.resume(&position, &malformed, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        malformed = token.clone();
        malformed.root_moves = None;
        assert!(matches!(
            engine.resume(&position, &malformed, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        malformed = token.clone();
        malformed.paused_stack.as_mut().unwrap().retained_bytes += 1;
        assert!(matches!(
            engine.resume(&position, &malformed, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        let imported = Position::from_fen(&position.to_fen()).unwrap();
        assert!(matches!(
            engine.resume(&imported, &token, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert!(matches!(
            engine.resume(
                &position,
                &token,
                CpuLimits {
                    max_depth: 3,
                    ..limits(1)
                },
                &AtomicBool::new(false)
            ),
            Err(CpuError::ResumeMismatch(_))
        ));
        let mut foreign = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        assert!(matches!(
            foreign.resume(&position, &token, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        let rejected = engine.paused_stack_snapshot().unwrap();
        assert_eq!(rejected.tokens_created, 1);
        assert_eq!(rejected.tokens_retained, 1);
        assert_eq!(rejected.tokens_resumed, 0);
        assert_eq!(rejected.tokens_invalidated, 0);
        assert_eq!(rejected.stale_context_attempts, 5);
        assert_eq!(rejected.stale_context_rejections, 5);
        let foreign_owner = foreign.paused_stack_snapshot().unwrap();
        assert_ne!(foreign_owner.owner_id, rejected.owner_id);
        assert_eq!(foreign_owner.tokens_created, 0);
        assert_eq!(foreign_owner.stale_context_attempts, 1);
        assert_eq!(foreign_owner.stale_context_rejections, 1);
        assert_eq!(engine.last_attempt_work(), Some(CpuWork::default()));
        let next = engine
            .resume(&position, &token, limits(1), &AtomicBool::new(false))
            .unwrap()
            .resume
            .unwrap();
        assert!(!engine.token_is_current(&token));
        assert!(engine.token_is_current(&next));
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        engine.clear();
        assert!(!engine.token_is_current(&next));
        assert!(matches!(
            engine.resume(&position, &next, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        let final_owner = engine.paused_stack_snapshot().unwrap();
        assert!(final_owner.complete);
        assert_eq!(final_owner.owner_id, observed_owner.owner_id);
        assert_eq!(final_owner.tokens_created, 2);
        assert_eq!(final_owner.tokens_resumed, 1);
        assert_eq!(final_owner.tokens_invalidated, 1);
        assert_eq!(final_owner.tokens_retained, 0);
        assert_eq!(final_owner.stale_context_attempts, 7);
        assert_eq!(final_owner.stale_context_rejections, 7);
        assert_eq!(final_owner.replayed_consumed_work, 0);
    }

    #[test]
    fn cancellation_new_analysis_and_namespace_mutation_invalidate_retained_work() {
        let position = Position::startpos();
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let cancel = AtomicBool::new(false);
        let token = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        let canceled = engine
            .resume(&position, &token, limits(1), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(canceled.completion, CpuCompletion::Canceled);
        assert_eq!(canceled.nodes, 0);
        assert!(canceled.resume.is_none() && engine.paused.is_none());
        assert!(!engine.token_is_current(&token));
        let canceled_owner = engine.paused_stack_snapshot().unwrap();
        assert!(canceled_owner.complete);
        assert_eq!(canceled_owner.tokens_created, 1);
        assert_eq!(canceled_owner.tokens_resumed, 1);
        assert_eq!(canceled_owner.tokens_invalidated, 0);
        assert_eq!(canceled_owner.tokens_retained, 0);
        assert_eq!(canceled_owner.bytes_current, 0);
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        let old = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        let new = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        assert!(!engine.token_is_current(&old));
        assert!(engine.token_is_current(&new));
        assert!(matches!(
            engine.resume(&position, &old, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        Arc::make_mut(&mut engine.value_identity).semantics = "changed-bootstrap-namespace".into();
        assert!(!engine.token_is_current(&new));
        assert!(matches!(
            engine.resume(&position, &new, limits(1), &cancel),
            Err(CpuError::InvalidConfig(_))
        ));
        assert!(engine.paused.is_none());
        assert_eq!(engine.last_attempt_work(), Some(CpuWork::default()));
        let final_owner = engine.paused_stack_snapshot().unwrap();
        assert!(final_owner.complete);
        assert_eq!(final_owner.owner_id, canceled_owner.owner_id);
        assert_eq!(final_owner.tokens_created, 3);
        assert_eq!(final_owner.tokens_resumed, 1);
        assert_eq!(final_owner.tokens_invalidated, 2);
        assert_eq!(final_owner.tokens_retained, 0);
        assert_eq!(final_owner.stale_context_attempts, 3);
        assert_eq!(final_owner.stale_context_rejections, 3);
    }

    #[test]
    fn expired_deadline_preserves_next_stage_and_memory_refusal_is_explicit() {
        let position = Position::startpos();
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let cancel = AtomicBool::new(false);
        let first = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        let expired = engine
            .resume(
                &position,
                &first,
                CpuLimits {
                    deadline: Some(Instant::now()),
                    ..limits(1)
                },
                &cancel,
            )
            .unwrap();
        assert_eq!(expired.completion, CpuCompletion::Deadline);
        assert_eq!(expired.nodes, 0);
        let next = expired.resume.unwrap();
        assert_eq!(next.cumulative_work(), first.cumulative_work());
        assert!(matches!(
            engine.resume(&position, &first, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert_eq!(
            engine
                .resume(&position, &next, limits(1), &cancel)
                .unwrap()
                .nodes,
            1
        );
        let task = engine.paused.as_mut().unwrap();
        task.best
            .try_reserve_exact(CPU_PAUSED_STACK_MAX_BYTES / std::mem::size_of::<BoardMove>() + 1)
            .unwrap();
        assert!(matches!(
            task.check_memory(),
            Err(CpuError::PausedStackMemoryLimit { .. })
        ));
    }
}
