//! 자체 CPU_T/CPU_R 코어: 유한 iterative deepening, PVS, aspiration, quiescence.
//!
//! 규칙과 종료 판정은 `rz-position`이 소유한다. 이 모듈의 material/PST 점수는
//! `bootstrap-material-pst-v1`의 차례 관점 raw 단위이며 학습 값이나 보정된 CP가 아니다.
//! TT의 짧은 hash는 slot 선택만 담당한다. 재사용은 전체 알려진 Rules 이력의
//! `PositionIdentity`, profile, 남은 깊이와 bound를 함께 검사한다.
//! 중단된 노드는 저장하지 않으며, 체크에서 quiescence 한도에 도달하면 static
//! stand-pat을 반환하는 대신 현재 iteration을 명시적으로 중단한다.

use crate::cpu_value::{
    BootstrapCpuValue, CpuAccumulator, CpuValueError, CpuValueEvaluator, CpuValueIdentity,
};
use rz_position::{
    BoardMove, Color, PieceKind, PlayStatus, Position, PositionError, PositionIdentity, Square,
    TerminalReason,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const CPU_SEARCH_VERSION: &str = "rz-cpu-pvs/0.1";
pub const CPU_SEARCH_CONDITIONS: &str = "iterative-deepening:1..requested;root-window:full-first,aspiration40-following,full-when-mate-or-fail-inclusive;pvs:first-full,following-zero-window,strict-interior-research;qsearch:tactical-capture-ep-promotion,all-check-evasions,no-check-standpat;q-limit:checked-abort;tt:direct-mapped,full-history-value-profile,equal-remaining-depth,completed-nodes-only;selectivity:no-reductions-no-nullmove;ties:Rules-order;score:side-to-move-raw";
pub const BOOTSTRAP_SCORE_VERSION: &str = "bootstrap-material-pst-v1";
pub const CPU_MATE_SCORE: i32 = 30_000;
pub const CPU_MATE_THRESHOLD: i32 = 29_000;
pub const CPU_FRONTIER_SCORE_LIMIT: i32 = 20_000;
const INFINITY: i32 = 32_000;
const MATE_THRESHOLD: i32 = CPU_MATE_THRESHOLD;
const MAX_DEPTH: u16 = 64;
const MAX_QUIESCENCE_PLY: u16 = 32;
const MAX_TT_ENTRIES: usize = 1_048_576;
const MAX_ROOT_MOVES: usize = 256;
const MAX_KNOWN_HISTORY: usize = 4096;
const ASPIRATION_WINDOW: i32 = 40;

/// First implementation has no selective reductions in any profile. Distinct
/// identities prevent independent verification work from sharing TT evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CpuProfile {
    PlanAssisted,
    Independent,
    RelaxedSelectivity,
}

impl CpuProfile {
    pub fn identity(self) -> &'static str {
        match self {
            Self::PlanAssisted => "cpu-plan-assisted-conservative-v1",
            Self::Independent => "cpu-independent-conservative-v1",
            Self::RelaxedSelectivity => "cpu-relaxed-no-reductions-v1",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CpuConfig {
    pub profile: CpuProfile,
    /// Bounded direct-mapped TT slots, never an unbounded map.
    pub tt_entries: usize,
    pub max_depth: u16,
    pub quiescence_ply: u16,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            profile: CpuProfile::PlanAssisted,
            tt_entries: 8192,
            max_depth: 16,
            quiescence_ply: 16,
        }
    }
}

impl CpuConfig {
    /// Requested contiguous TT slot layout in bytes on the current build target.
    /// Zero slots disables the TT and returns zero. This excludes allocations
    /// retained by entry identities, allocator rounding/bookkeeping, the Vec
    /// header and search history; it is neither total TT usage nor memory peak.
    /// `try_reserve_exact` may still receive a larger capacity from the allocator.
    pub fn tt_allocation_bytes(&self) -> Result<u64, CpuError> {
        if self.tt_entries > MAX_TT_ENTRIES {
            return Err(CpuError::InvalidConfig("TT exceeds 1,048,576 slots"));
        }
        checked_inline_slot_bytes(self.tt_entries, std::mem::size_of::<Option<TtEntry>>())
    }
}

fn checked_inline_slot_bytes(entries: usize, slot_bytes: usize) -> Result<u64, CpuError> {
    let bytes = entries
        .checked_mul(slot_bytes)
        .ok_or(CpuError::InvalidConfig(
            "TT slot bytes overflow address space",
        ))?;
    u64::try_from(bytes)
        .map_err(|_| CpuError::InvalidConfig("TT slot bytes cannot be represented as u64"))
}

#[derive(Clone, Copy, Debug)]
pub struct CpuLimits {
    pub max_depth: u16,
    /// Counts both full-search and quiescence positions, including TT probes.
    pub max_nodes: u64,
    pub deadline: Option<Instant>,
}

impl Default for CpuLimits {
    fn default() -> Self {
        Self {
            max_depth: 8,
            max_nodes: 100_000,
            deadline: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuCompletion {
    DepthLimit,
    NodeLimit,
    Deadline,
    Canceled,
    /// A checked quiescence frontier cannot be statically evaluated safely.
    QuiescenceLimit,
    Terminal(TerminalReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuScoreScope {
    /// No complete search iteration; provisional frontier value from the
    /// selected evaluator. Training/weights provenance is recorded separately.
    FrontierOnly,
    /// Completed minimax estimate for the declared depth and qsearch profile;
    /// finite lookahead is not an exact whole-game outcome.
    CompletedIteration,
    /// Rules certified the current state, independent of search depth.
    RulesTerminal,
}

#[derive(Clone, Debug)]
pub struct CpuReport {
    pub best_move: Option<BoardMove>,
    /// Verified legal prefix. TT reuse can shorten this independently of the
    /// completed lookahead depth; length never proves full line coverage.
    pub pv: Vec<BoardMove>,
    /// Root side-to-move value. Never interpreted as calibrated centipawns.
    pub score: i32,
    pub completed_depth: u16,
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub tt_hits: u64,
    pub completion: CpuCompletion,
    pub score_scope: CpuScoreScope,
    pub profile: CpuProfile,
    pub score_provenance: &'static str,
    pub value_identity: CpuValueIdentity,
    pub search_version: &'static str,
    pub root_restricted: bool,
    pub elapsed: Duration,
    pub reused_completed_depth: u16,
    /// Resume only after the last completed iteration. Interrupted node stacks
    /// are discarded; a new finite budget continues at completed_depth + 1.
    pub resume: Option<CpuResumeToken>,
}

#[derive(Clone, Copy, Debug)]
pub struct CpuIterationProgress {
    pub best_move: Option<BoardMove>,
    pub score: i32,
    pub completed_depth: u16,
    pub nodes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuWork {
    /// Actual positions visited by this invocation only, including qsearch.
    /// Reused completed depth does not re-charge the previous invocation.
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub tt_hits: u64,
}

#[derive(Clone, Debug)]
pub struct CpuResumeToken {
    root: PositionIdentity,
    root_moves: Option<Vec<BoardMove>>,
    profile: CpuProfile,
    quiescence_ply: u16,
    value_identity: CpuValueIdentity,
    search_conditions: String,
    completed_depth: u16,
    score: i32,
    pv: Vec<BoardMove>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuError {
    InvalidConfig(&'static str),
    InvalidLimits(&'static str),
    InvalidRootMoves(&'static str),
    ResumeMismatch(&'static str),
    Unsupported(&'static str),
    Allocation,
    Rules(PositionError),
    Value(CpuValueError),
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "CPU configuration: {message}"),
            Self::InvalidLimits(message) => write!(f, "CPU limits: {message}"),
            Self::InvalidRootMoves(message) => write!(f, "CPU root moves: {message}"),
            Self::ResumeMismatch(message) => write!(f, "CPU resume: {message}"),
            Self::Unsupported(message) => write!(f, "CPU capability unavailable: {message}"),
            Self::Allocation => f.write_str("CPU TT allocation refused"),
            Self::Rules(error) => write!(f, "CPU rules failure: {error}"),
            Self::Value(error) => write!(f, "CPU value failure: {error}"),
        }
    }
}

impl std::error::Error for CpuError {}

impl From<PositionError> for CpuError {
    fn from(value: PositionError) -> Self {
        Self::Rules(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Debug)]
struct TtEntry {
    hash: u64,
    identity: PositionIdentity,
    profile: CpuProfile,
    value_identity: Arc<CpuValueIdentity>,
    depth: u16,
    /// Mate distance normalized to this node, not the root that stored it.
    score: i32,
    bound: Bound,
    best_move: Option<BoardMove>,
}

pub struct CpuEngine {
    config: CpuConfig,
    tt: Vec<Option<TtEntry>>,
    /// Saturating side/from/to quiet history; fixed 32 KiB.
    history: Box<[[[i32; 64]; 64]; 2]>,
    evaluator: Arc<dyn CpuValueEvaluator>,
    value_identity: Arc<CpuValueIdentity>,
    last_attempt_work: Option<CpuWork>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuCapabilities {
    pub max_depth: u16,
    pub max_prefix_plies: usize,
    pub max_root_moves: usize,
    pub root_moves: bool,
    pub divergence: bool,
    pub completed_iteration_resume: bool,
    pub selective_reductions: bool,
}

/// Startup-selected own CPU checker. Value identity and search implementation
/// identity are independent immutable namespaces. Consumers validate returned
/// report identities and may not transfer TT or opaque resume tokens between
/// implementations. Unsupported capabilities fail explicitly; no fallback.
pub trait CpuSearcher: Send {
    fn config(&self) -> &CpuConfig;
    fn value_identity(&self) -> &CpuValueIdentity;
    fn search_identity(&self) -> &'static str;
    /// Exact immutable implementation/configuration declaration. Custom
    /// checkers must declare their own conditions rather than inheriting PVS.
    fn search_conditions(&self) -> String;
    fn capabilities(&self) -> CpuCapabilities;
    /// None means work was not observed. Never substitute invented zero work.
    /// The default preserves unknown accounting for custom implementations.
    fn last_attempt_work(&self) -> Option<CpuWork> {
        None
    }
    fn clear(&mut self);
    fn analyze(
        &mut self,
        position: &Position,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CpuReport, CpuError>;
    fn analyze_root_moves(
        &mut self,
        _position: &Position,
        _moves: &[BoardMove],
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        Err(CpuError::Unsupported("root moves"))
    }
    fn analyze_divergence(
        &mut self,
        _position: &Position,
        _prefix: &[BoardMove],
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        Err(CpuError::Unsupported("divergence"))
    }
    fn resume(
        &mut self,
        _position: &Position,
        _token: &CpuResumeToken,
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        Err(CpuError::Unsupported("completed iteration resume"))
    }
}

impl<C: CpuSearcher + ?Sized> CpuSearcher for Box<C> {
    fn config(&self) -> &CpuConfig {
        (**self).config()
    }
    fn value_identity(&self) -> &CpuValueIdentity {
        (**self).value_identity()
    }
    fn search_identity(&self) -> &'static str {
        (**self).search_identity()
    }
    fn search_conditions(&self) -> String {
        (**self).search_conditions()
    }
    fn capabilities(&self) -> CpuCapabilities {
        (**self).capabilities()
    }
    fn last_attempt_work(&self) -> Option<CpuWork> {
        (**self).last_attempt_work()
    }
    fn clear(&mut self) {
        (**self).clear();
    }
    fn analyze(
        &mut self,
        p: &Position,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        (**self).analyze(p, l, c)
    }
    fn analyze_root_moves(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        (**self).analyze_root_moves(p, m, l, c)
    }
    fn analyze_divergence(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        (**self).analyze_divergence(p, m, l, c)
    }
    fn resume(
        &mut self,
        p: &Position,
        t: &CpuResumeToken,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        (**self).resume(p, t, l, c)
    }
}

impl CpuSearcher for CpuEngine {
    fn config(&self) -> &CpuConfig {
        CpuEngine::config(self)
    }
    fn value_identity(&self) -> &CpuValueIdentity {
        CpuEngine::value_identity(self)
    }
    fn search_identity(&self) -> &'static str {
        CPU_SEARCH_VERSION
    }
    fn search_conditions(&self) -> String {
        CpuEngine::search_conditions(self)
    }
    fn capabilities(&self) -> CpuCapabilities {
        CpuCapabilities {
            max_depth: self.config.max_depth,
            max_prefix_plies: 64,
            max_root_moves: MAX_ROOT_MOVES,
            root_moves: true,
            divergence: true,
            completed_iteration_resume: true,
            selective_reductions: false,
        }
    }
    fn last_attempt_work(&self) -> Option<CpuWork> {
        CpuEngine::last_attempt_work(self)
    }
    fn clear(&mut self) {
        CpuEngine::clear(self);
    }
    fn analyze(
        &mut self,
        p: &Position,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        CpuEngine::analyze(self, p, l, c)
    }
    fn analyze_root_moves(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        CpuEngine::analyze_root_moves(self, p, m, l, c)
    }
    fn analyze_divergence(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        CpuEngine::analyze_divergence(self, p, m, l, c)
    }
    fn resume(
        &mut self,
        p: &Position,
        t: &CpuResumeToken,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        CpuEngine::resume(self, p, t, l, c)
    }
}

struct Control<'a> {
    limits: CpuLimits,
    cancellation: &'a AtomicBool,
    nodes: u64,
    quiescence_nodes: u64,
    tt_hits: u64,
    value: CpuAccumulator,
}

#[derive(Debug)]
enum Abort {
    Stop(CpuCompletion),
    Error(CpuError),
}

impl From<PositionError> for Abort {
    fn from(error: PositionError) -> Self {
        Self::Error(CpuError::Rules(error))
    }
}

impl From<CpuValueError> for Abort {
    fn from(error: CpuValueError) -> Self {
        Self::Error(CpuError::Value(error))
    }
}

struct NodeValue {
    score: i32,
    pv: Vec<BoardMove>,
}

impl CpuEngine {
    pub fn new(config: CpuConfig) -> Result<Self, CpuError> {
        Self::with_evaluator(config, Arc::new(BootstrapCpuValue::default()))
    }

    pub fn with_evaluator(
        config: CpuConfig,
        evaluator: Arc<dyn CpuValueEvaluator>,
    ) -> Result<Self, CpuError> {
        config.tt_allocation_bytes()?;
        evaluator.identity().validate().map_err(CpuError::Value)?;
        if config.max_depth == 0 || config.max_depth > MAX_DEPTH {
            return Err(CpuError::InvalidConfig("depth must be in 1..=64"));
        }
        if config.quiescence_ply > MAX_QUIESCENCE_PLY {
            return Err(CpuError::InvalidConfig("quiescence ply must be in 0..=32"));
        }
        let mut tt = Vec::new();
        tt.try_reserve_exact(config.tt_entries)
            .map_err(|_| CpuError::Allocation)?;
        tt.resize_with(config.tt_entries, || None);
        Ok(Self {
            config,
            tt,
            history: Box::new([[[0; 64]; 64]; 2]),
            value_identity: Arc::new(evaluator.identity().clone()),
            evaluator,
            last_attempt_work: None,
        })
    }

    pub fn config(&self) -> &CpuConfig {
        &self.config
    }

    /// Full immutable value namespace used by this engine's TT and resume.
    pub fn value_identity(&self) -> &CpuValueIdentity {
        &self.value_identity
    }

    pub fn search_conditions(&self) -> String {
        format!(
            "{CPU_SEARCH_CONDITIONS};profile={};max_depth={};q_plies={};tt_entries={}",
            self.config.profile.identity(),
            self.config.max_depth,
            self.config.quiescence_ply,
            self.config.tt_entries
        )
    }

    /// Actual counters for the last normally returned attempt, including
    /// errors. Before the first attempt or after clear there is no observation.
    /// An admission error has observed zero searched positions, not a result.
    pub fn last_attempt_work(&self) -> Option<CpuWork> {
        self.last_attempt_work
    }

    /// New game ownership boundary; no TT/history evidence crosses this call.
    pub fn clear(&mut self) {
        self.tt.iter_mut().for_each(|entry| *entry = None);
        *self.history = [[[0; 64]; 64]; 2];
        self.last_attempt_work = None;
    }

    pub fn analyze(
        &mut self,
        position: &Position,
        limits: CpuLimits,
        cancellation: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        self.run(position, None, limits, cancellation, None, &mut |_| {})
    }

    pub fn analyze_with_progress(
        &mut self,
        position: &Position,
        limits: CpuLimits,
        cancellation: &AtomicBool,
        mut observer: impl FnMut(CpuIterationProgress),
    ) -> Result<CpuReport, CpuError> {
        self.run(position, None, limits, cancellation, None, &mut observer)
    }

    /// Restrict only root choices. Every response below the root remains legal
    /// Rules search; restricted root values never enter the unrestricted TT.
    pub fn analyze_root_moves(
        &mut self,
        position: &Position,
        root_moves: &[BoardMove],
        limits: CpuLimits,
        cancellation: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        self.run(
            position,
            Some(root_moves),
            limits,
            cancellation,
            None,
            &mut |_| {},
        )
    }

    /// Apply a checked prefix before analyzing the actual divergence state.
    /// The returned value/PV belong to that state's side to move.
    pub fn analyze_divergence(
        &mut self,
        position: &Position,
        prefix: &[BoardMove],
        limits: CpuLimits,
        cancellation: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        self.last_attempt_work = Some(CpuWork::default());
        if prefix.len() > MAX_DEPTH as usize {
            return Err(CpuError::InvalidLimits("divergence prefix exceeds 64 ply"));
        }
        let mut divergence = position.clone();
        for &mv in prefix {
            divergence.make_move(mv)?;
        }
        self.analyze(&divergence, limits, cancellation)
    }

    pub fn resume(
        &mut self,
        position: &Position,
        token: &CpuResumeToken,
        limits: CpuLimits,
        cancellation: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        self.last_attempt_work = Some(CpuWork::default());
        if token.root != position.position_identity() {
            return Err(CpuError::ResumeMismatch(
                "Rules state or full known history changed",
            ));
        }
        if token.profile != self.config.profile
            || token.quiescence_ply != self.config.quiescence_ply
            || token.value_identity != *self.value_identity
            || token.search_conditions != self.search_conditions()
        {
            return Err(CpuError::ResumeMismatch(
                "CPU search conditions or value namespace changed",
            ));
        }
        if limits.max_depth < token.completed_depth {
            return Err(CpuError::ResumeMismatch(
                "new depth is below completed depth",
            ));
        }
        self.run(
            position,
            token.root_moves.as_deref(),
            limits,
            cancellation,
            Some(token),
            &mut |_| {},
        )
    }

    fn run(
        &mut self,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancellation: &AtomicBool,
        resume: Option<&CpuResumeToken>,
        observer: &mut dyn FnMut(CpuIterationProgress),
    ) -> Result<CpuReport, CpuError> {
        self.last_attempt_work = Some(CpuWork::default());
        self.validate_limits(position, limits)?;
        let started = Instant::now();
        let root_view = position.ordered_legal_moves();
        let status = position.play_status_from_view(&root_view)?;
        let mut legal = root_view.moves().to_vec();
        if let Some(requested) = root_moves {
            if requested.len() > MAX_ROOT_MOVES {
                return Err(CpuError::InvalidRootMoves("restriction exceeds 256 moves"));
            }
            if requested.is_empty() {
                return Err(CpuError::InvalidRootMoves("empty restriction"));
            }
            for (index, mv) in requested.iter().enumerate() {
                if !legal.contains(mv) {
                    return Err(CpuError::InvalidRootMoves("move is not legal at root"));
                }
                if requested[..index].contains(mv) {
                    return Err(CpuError::InvalidRootMoves("duplicate move"));
                }
            }
            // Rules' deterministic order remains the tie breaker.
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
                profile: self.config.profile,
                score_provenance: "rz-position/rules-terminal",
                value_identity: (*self.value_identity).clone(),
                search_version: CPU_SEARCH_VERSION,
                root_restricted: root_moves.is_some(),
                elapsed: started.elapsed(),
                reused_completed_depth: 0,
                resume: None,
            });
        }
        let mut control = Control {
            limits,
            cancellation,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: self
                .evaluator
                .initialize(position)
                .map_err(CpuError::Value)?,
        };
        let mut work = position.clone();
        let mut best = resume
            .map(|token| token.pv.clone())
            .unwrap_or_else(|| legal.first().copied().into_iter().collect());
        let mut score = if let Some(token) = resume {
            token.score
        } else {
            self.frontier_score(&control, position)
                .map_err(CpuError::Value)?
        };
        let mut completed_depth = resume.map_or(0, |token| token.completed_depth);
        let mut completion = CpuCompletion::DepthLimit;
        observer(CpuIterationProgress {
            best_move: best.first().copied(),
            score,
            completed_depth,
            nodes: 0,
        });
        let admission = control.check_stop();
        if let Err(Abort::Stop(reason)) = &admission {
            completion = *reason;
        }

        for depth in
            (completed_depth.saturating_add(1)..=limits.max_depth).filter(|_| admission.is_ok())
        {
            let window = if completed_depth == 0 || score.abs() >= MATE_THRESHOLD {
                (-INFINITY, INFINITY)
            } else {
                (score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW)
            };
            let searched = self.root_search(
                &mut work,
                &legal,
                depth,
                window.0,
                window.1,
                best.first().copied(),
                &mut control,
            );
            let searched = match searched {
                Ok(value) if value.score <= window.0 || value.score >= window.1 => self
                    .root_search(
                        &mut work,
                        &legal,
                        depth,
                        -INFINITY,
                        INFINITY,
                        best.first().copied(),
                        &mut control,
                    ),
                result => result,
            };
            match searched {
                Ok(value) => {
                    score = value.score;
                    best = value.pv;
                    completed_depth = depth;
                    observer(CpuIterationProgress {
                        best_move: best.first().copied(),
                        score,
                        completed_depth,
                        nodes: control.nodes,
                    });
                }
                Err(Abort::Stop(reason)) => {
                    completion = reason;
                    break;
                }
                Err(Abort::Error(error)) => {
                    self.last_attempt_work = Some(control.actual_work());
                    return Err(error);
                }
            }
        }
        // Observers are outside node search and may cancel or consume time.
        // Preserve the completed result while recording the final stop reason.
        if let Err(Abort::Stop(reason)) = control.check_stop() {
            completion = reason;
        }
        self.last_attempt_work = Some(control.actual_work());
        let token = (completed_depth > 0).then(|| CpuResumeToken {
            root: position.position_identity(),
            root_moves: root_moves.map(<[BoardMove]>::to_vec),
            profile: self.config.profile,
            quiescence_ply: self.config.quiescence_ply,
            value_identity: (*self.value_identity).clone(),
            search_conditions: self.search_conditions(),
            completed_depth,
            score,
            pv: best.clone(),
        });
        Ok(CpuReport {
            best_move: best.first().copied(),
            pv: best,
            score,
            completed_depth,
            nodes: control.nodes,
            quiescence_nodes: control.quiescence_nodes,
            tt_hits: control.tt_hits,
            completion,
            score_scope: if completed_depth == 0 {
                CpuScoreScope::FrontierOnly
            } else {
                CpuScoreScope::CompletedIteration
            },
            profile: self.config.profile,
            score_provenance: self.evaluator.provenance(),
            value_identity: (*self.value_identity).clone(),
            search_version: CPU_SEARCH_VERSION,
            root_restricted: root_moves.is_some(),
            elapsed: started.elapsed(),
            reused_completed_depth: resume.map_or(0, |token| token.completed_depth),
            resume: token,
        })
    }

    fn validate_limits(&self, position: &Position, limits: CpuLimits) -> Result<(), CpuError> {
        if self.evaluator.identity() != self.value_identity.as_ref() {
            return Err(CpuError::InvalidConfig(
                "evaluator semantics or frozen weights changed",
            ));
        }
        if limits.max_depth == 0 || limits.max_depth > self.config.max_depth {
            return Err(CpuError::InvalidLimits("depth outside configured bound"));
        }
        if limits.max_nodes == 0 {
            return Err(CpuError::InvalidLimits("node budget must be positive"));
        }
        if position.known_history_len() > MAX_KNOWN_HISTORY {
            return Err(CpuError::InvalidLimits("known history exceeds 4096 states"));
        }
        Ok(())
    }

    fn frontier_score(
        &self,
        control: &Control<'_>,
        position: &Position,
    ) -> Result<i32, CpuValueError> {
        let score = self.evaluator.score(&control.value, position)?;
        if !(-CPU_FRONTIER_SCORE_LIMIT..=CPU_FRONTIER_SCORE_LIMIT).contains(&score) {
            return Err(CpuValueError::ScoreOutsideFrontierNamespace);
        }
        Ok(score)
    }

    #[allow(clippy::too_many_arguments)]
    fn root_search(
        &mut self,
        position: &mut Position,
        legal: &[BoardMove],
        depth: u16,
        mut alpha: i32,
        beta: i32,
        previous_best: Option<BoardMove>,
        control: &mut Control<'_>,
    ) -> Result<NodeValue, Abort> {
        control.visit(false)?;
        let mut moves = legal.to_vec();
        self.order_moves(position, &mut moves, previous_best);
        let mut best = NodeValue {
            score: -INFINITY,
            pv: Vec::new(),
        };
        for (index, mv) in moves.into_iter().enumerate() {
            let undo = position.make_move(mv)?;
            let value_undo = match self.evaluator.apply_delta(&mut control.value, undo.delta()) {
                Ok(undo) => undo,
                Err(error) => {
                    position.unmake(undo)?;
                    return Err(error.into());
                }
            };
            let child = self.pvs_child(position, depth - 1, 1, alpha, beta, index == 0, control);
            position.unmake(undo)?;
            self.evaluator
                .restore(&mut control.value, value_undo, position)?;
            let child = child?;
            let score = -child.score;
            if score > best.score {
                best.score = score;
                best.pv.clear();
                best.pv.push(mv);
                best.pv.extend(child.pv);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        control.check_stop()?;
        Ok(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn pvs_child(
        &mut self,
        position: &mut Position,
        depth: u16,
        ply: u16,
        alpha: i32,
        beta: i32,
        first: bool,
        control: &mut Control<'_>,
    ) -> Result<NodeValue, Abort> {
        if first {
            return self.negamax(position, depth, ply, -beta, -alpha, control);
        }
        let mut value = self.negamax(position, depth, ply, -alpha - 1, -alpha, control)?;
        if -value.score > alpha && -value.score < beta {
            value = self.negamax(position, depth, ply, -beta, -alpha, control)?;
        }
        Ok(value)
    }

    fn negamax(
        &mut self,
        position: &mut Position,
        depth: u16,
        ply: u16,
        mut alpha: i32,
        beta: i32,
        control: &mut Control<'_>,
    ) -> Result<NodeValue, Abort> {
        if depth == 0 {
            return self.quiescence(
                position,
                ply,
                self.config.quiescence_ply,
                alpha,
                beta,
                control,
            );
        }
        control.visit(false)?;
        let view = position.ordered_legal_moves();
        if let PlayStatus::Terminal { winner, .. } = position.play_status_from_view(&view)? {
            return Ok(NodeValue {
                score: terminal_score(position, winner, ply),
                pv: Vec::new(),
            });
        }
        let alpha_original = alpha;
        let hash = tt_hash(position);
        let identity = position.position_identity();
        let tt = self.probe(hash, &identity).cloned();
        if let Some(entry) = &tt {
            control.tt_hits += 1;
            // A depth-8 bound cannot masquerade as a depth-10 result.
            // A deeper static frontier is not mathematically a bound on a
            // shallower frontier. Require the same remaining depth so replay
            // preserves this profile's fixed-depth estimate exactly.
            if entry.depth == depth {
                let score = from_tt_score(entry.score, ply);
                if entry.bound == Bound::Exact
                    || (entry.bound == Bound::Lower && score >= beta)
                    || (entry.bound == Bound::Upper && score <= alpha)
                {
                    return Ok(NodeValue {
                        score,
                        pv: entry.best_move.into_iter().collect(),
                    });
                }
            }
        }
        let mut moves = view.moves().to_vec();
        self.order_moves(
            position,
            &mut moves,
            tt.as_ref().and_then(|entry| entry.best_move),
        );
        let mut best = NodeValue {
            score: -INFINITY,
            pv: Vec::new(),
        };
        for (index, mv) in moves.into_iter().enumerate() {
            let quiet = !is_tactical(position, mv);
            let side = position.side_to_move() as usize;
            let undo = position.make_move(mv)?;
            let value_undo = match self.evaluator.apply_delta(&mut control.value, undo.delta()) {
                Ok(undo) => undo,
                Err(error) => {
                    position.unmake(undo)?;
                    return Err(error.into());
                }
            };
            let child = self.pvs_child(
                position,
                depth - 1,
                ply + 1,
                alpha,
                beta,
                index == 0,
                control,
            );
            position.unmake(undo)?;
            self.evaluator
                .restore(&mut control.value, value_undo, position)?;
            let child = child?;
            let score = -child.score;
            if score > best.score {
                best.score = score;
                best.pv.clear();
                best.pv.push(mv);
                best.pv.extend(child.pv);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                if quiet {
                    let history =
                        &mut self.history[side][mv.from.index() as usize][mv.to.index() as usize];
                    *history = history.saturating_add(i32::from(depth).pow(2)).min(20_000);
                }
                break;
            }
        }
        // A cancellation/deadline after the last child still forbids this
        // parent from publishing a completed TT bound.
        control.check_stop()?;
        let bound = if best.score <= alpha_original {
            Bound::Upper
        } else if best.score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.store(TtEntry {
            hash,
            identity,
            profile: self.config.profile,
            value_identity: Arc::clone(&self.value_identity),
            depth,
            score: to_tt_score(best.score, ply),
            bound,
            best_move: best.pv.first().copied(),
        });
        Ok(best)
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        ply: u16,
        remaining: u16,
        mut alpha: i32,
        beta: i32,
        control: &mut Control<'_>,
    ) -> Result<NodeValue, Abort> {
        control.visit(true)?;
        let view = position.ordered_legal_moves();
        if let PlayStatus::Terminal { winner, .. } = position.play_status_from_view(&view)? {
            return Ok(NodeValue {
                score: terminal_score(position, winner, ply),
                pv: Vec::new(),
            });
        }
        let in_check = position.in_check();
        if in_check && remaining == 0 {
            return Err(Abort::Stop(CpuCompletion::QuiescenceLimit));
        }
        let mut best = NodeValue {
            // Stand-pat is forbidden in check, including at the depth cap.
            score: if in_check {
                -INFINITY
            } else {
                self.frontier_score(control, position)?
            },
            pv: Vec::new(),
        };
        if !in_check {
            if best.score >= beta || remaining == 0 {
                return Ok(best);
            }
            alpha = alpha.max(best.score);
        }
        let mut moves = view.moves().to_vec();
        if !in_check {
            moves.retain(|mv| is_tactical(position, *mv));
        }
        self.order_moves(position, &mut moves, None);
        for mv in moves {
            let undo = position.make_move(mv)?;
            let value_undo = match self.evaluator.apply_delta(&mut control.value, undo.delta()) {
                Ok(undo) => undo,
                Err(error) => {
                    position.unmake(undo)?;
                    return Err(error.into());
                }
            };
            let child = self.quiescence(position, ply + 1, remaining - 1, -beta, -alpha, control);
            position.unmake(undo)?;
            self.evaluator
                .restore(&mut control.value, value_undo, position)?;
            let child = child?;
            let score = -child.score;
            if score > best.score {
                best.score = score;
                best.pv.clear();
                best.pv.push(mv);
                best.pv.extend(child.pv);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        control.check_stop()?;
        Ok(best)
    }

    fn order_moves(&self, position: &Position, moves: &mut [BoardMove], best: Option<BoardMove>) {
        // Stable sorting preserves Rules order when all priorities tie.
        moves.sort_by_key(|mv| std::cmp::Reverse(self.order_score(position, *mv, best)));
    }

    fn order_score(&self, position: &Position, mv: BoardMove, best: Option<BoardMove>) -> i32 {
        if best == Some(mv) {
            return 1_000_000;
        }
        let attacker = position
            .piece_at(mv.from)
            .map_or(0, |piece| material(piece.kind));
        let victim = position.piece_at(mv.to).map_or_else(
            || {
                if is_en_passant(position, mv) {
                    material(PieceKind::Pawn)
                } else {
                    0
                }
            },
            |piece| material(piece.kind),
        );
        if victim > 0 || mv.promotion.is_some() {
            // MVV/LVA is an ordering hint, not an exact SEE or pruning bound.
            return 100_000 + victim * 16 - attacker + mv.promotion.map_or(0, material);
        }
        self.history[position.side_to_move() as usize][mv.from.index() as usize]
            [mv.to.index() as usize]
    }

    fn probe(&self, hash: u64, identity: &PositionIdentity) -> Option<&TtEntry> {
        if self.tt.is_empty() {
            return None;
        }
        self.tt[hash as usize % self.tt.len()]
            .as_ref()
            .filter(|entry| {
                entry.hash == hash
                    && entry.profile == self.config.profile
                    && entry.value_identity == self.value_identity
                    && entry.identity == *identity
            })
    }

    fn store(&mut self, entry: TtEntry) {
        if self.tt.is_empty() {
            return;
        }
        let index = entry.hash as usize % self.tt.len();
        if self.tt[index].as_ref().is_some_and(|old| {
            old.hash == entry.hash && old.identity == entry.identity && old.depth > entry.depth
        }) {
            return;
        }
        self.tt[index] = Some(entry);
    }
}

impl Control<'_> {
    fn actual_work(&self) -> CpuWork {
        CpuWork {
            nodes: self.nodes,
            quiescence_nodes: self.quiescence_nodes,
            tt_hits: self.tt_hits,
        }
    }

    fn check_stop(&self) -> Result<(), Abort> {
        if self.cancellation.load(Ordering::Acquire) {
            Err(Abort::Stop(CpuCompletion::Canceled))
        } else if self
            .limits
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            Err(Abort::Stop(CpuCompletion::Deadline))
        } else {
            Ok(())
        }
    }

    fn visit(&mut self, quiescence: bool) -> Result<(), Abort> {
        self.check_stop()?;
        if self.nodes >= self.limits.max_nodes {
            return Err(Abort::Stop(CpuCompletion::NodeLimit));
        }
        self.nodes += 1;
        if quiescence {
            self.quiescence_nodes += 1;
        }
        Ok(())
    }
}

fn tt_hash(position: &Position) -> u64 {
    let mut hasher = DefaultHasher::new();
    position.repetition_identity().hash(&mut hasher);
    position.halfmove_clock().hash(&mut hasher);
    position.fullmove_number().hash(&mut hasher);
    position.known_history_len().hash(&mut hasher);
    hasher.finish()
}

fn terminal_score(position: &Position, winner: Option<Color>, ply: u16) -> i32 {
    match winner {
        Some(winner) if winner == position.side_to_move() => CPU_MATE_SCORE - i32::from(ply),
        Some(_) => -CPU_MATE_SCORE + i32::from(ply),
        None => 0,
    }
}

fn to_tt_score(score: i32, ply: u16) -> i32 {
    if score >= MATE_THRESHOLD {
        score + i32::from(ply)
    } else if score <= -MATE_THRESHOLD {
        score - i32::from(ply)
    } else {
        score
    }
}

fn from_tt_score(score: i32, ply: u16) -> i32 {
    if score >= MATE_THRESHOLD {
        score - i32::from(ply)
    } else if score <= -MATE_THRESHOLD {
        score + i32::from(ply)
    } else {
        score
    }
}

fn material(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight => 320,
        PieceKind::Bishop => 330,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 0,
    }
}

fn is_en_passant(position: &Position, mv: BoardMove) -> bool {
    position.en_passant_target() == Some(mv.to)
        && position.piece_at(mv.to).is_none()
        && position
            .piece_at(mv.from)
            .is_some_and(|piece| piece.kind == PieceKind::Pawn)
        && mv.from.file() != mv.to.file()
}

fn is_tactical(position: &Position, mv: BoardMove) -> bool {
    mv.promotion.is_some() || position.piece_at(mv.to).is_some() || is_en_passant(position, mv)
}

/// Deterministic, untrained side-to-move raw score, with bounded magnitude far
/// below the exact mate namespace. No win probability or calibrated CP claim.
pub fn evaluate_bootstrap(position: &Position) -> i32 {
    let mut white = 0i32;
    for index in 0..64 {
        let square = Square::new(index).expect("bounded board index");
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        let rank = if piece.color == Color::White {
            square.rank()
        } else {
            7 - square.rank()
        };
        let center = 6 - (2 * i32::from(square.file()) - 7).abs() - (2 * i32::from(rank) - 7).abs();
        let pst = match piece.kind {
            PieceKind::Pawn => i32::from(rank) * 8,
            PieceKind::Knight => center * 4,
            PieceKind::Bishop => center * 2,
            PieceKind::Rook => i32::from(rank) * 2,
            PieceKind::Queen => center,
            PieceKind::King => -center * 2,
        };
        let value = material(piece.kind) + pst;
        white += if piece.color == Color::White {
            value
        } else {
            -value
        };
    }
    let value = if position.side_to_move() == Color::White {
        white
    } else {
        -white
    };
    value.clamp(-20_000, 20_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> CpuEngine {
        CpuEngine::new(CpuConfig {
            tt_entries: 1024,
            max_depth: 10,
            ..CpuConfig::default()
        })
        .unwrap()
    }

    fn limits(depth: u16) -> CpuLimits {
        CpuLimits {
            max_depth: depth,
            max_nodes: 30_000,
            deadline: None,
        }
    }

    fn run(fen: &str, depth: u16) -> (Position, CpuReport) {
        let position = Position::from_fen(fen).unwrap();
        let report = engine()
            .analyze(&position, limits(depth), &AtomicBool::new(false))
            .unwrap();
        (position, report)
    }

    #[test]
    fn mate_in_one_both_colors_is_certified_by_rules() {
        for fen in [
            "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            "8/8/8/8/8/6k1/5q2/7K b - - 0 1",
        ] {
            let (position, report) = run(fen, 1);
            let selected = report.best_move.unwrap();
            let child = position.preview_move(selected).unwrap();
            assert_eq!(
                child.classify_position().unwrap().play_status,
                PlayStatus::Terminal {
                    reason: TerminalReason::Checkmate,
                    winner: Some(position.side_to_move()),
                }
            );
            assert_eq!(report.score, CPU_MATE_SCORE - 1);
            assert_eq!(report.completed_depth, 1);
        }
    }

    #[test]
    fn terminal_and_stalemate_do_not_emit_a_fake_move() {
        let (_, mate) = run("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 2);
        assert_eq!(mate.score, -CPU_MATE_SCORE);
        assert_eq!(
            mate.completion,
            CpuCompletion::Terminal(TerminalReason::Checkmate)
        );
        assert_eq!(mate.best_move, None);
        let (_, stale) = run("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 2);
        assert_eq!(stale.score, 0);
        assert_eq!(
            stale.completion,
            CpuCompletion::Terminal(TerminalReason::Stalemate)
        );
        assert!(stale.pv.is_empty());
    }

    #[test]
    fn avoids_a_verified_opponent_mate_in_one_when_an_escape_exists() {
        let (position, report) = run("6k1/8/5K2/8/8/8/8/6Q1 b - - 0 1", 2);
        let permits_mate = |mv| {
            let child = position.preview_move(mv).unwrap();
            child.legal_moves().into_iter().any(|reply| {
                matches!(
                    child
                        .preview_move(reply)
                        .unwrap()
                        .classify_position()
                        .unwrap()
                        .play_status,
                    PlayStatus::Terminal {
                        reason: TerminalReason::Checkmate,
                        winner: Some(Color::White)
                    }
                )
            })
        };
        let legal = position.legal_moves();
        assert!(legal.iter().copied().any(permits_mate));
        assert!(legal.iter().copied().any(|mv| !permits_mate(mv)));
        assert_eq!(report.completed_depth, 2);
        assert!(!permits_mate(report.best_move.unwrap()));
        assert!(report.score > -MATE_THRESHOLD);
    }

    #[test]
    fn cancellation_deadline_and_node_limit_preserve_a_legal_fallback() {
        let position = Position::startpos();
        let mut engine = engine();
        let canceled = engine
            .analyze(&position, limits(2), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(canceled.completion, CpuCompletion::Canceled);
        assert_eq!(canceled.completed_depth, 0);
        assert!(
            position
                .legal_moves()
                .contains(&canceled.best_move.unwrap())
        );
        assert_eq!(canceled.score_scope, CpuScoreScope::FrontierOnly);
        let expired = engine
            .analyze(
                &position,
                CpuLimits {
                    deadline: Some(Instant::now()),
                    ..limits(2)
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(expired.completion, CpuCompletion::Deadline);
        let bounded = engine
            .analyze(
                &position,
                CpuLimits {
                    max_nodes: 2,
                    ..limits(2)
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(bounded.completion, CpuCompletion::NodeLimit);
        assert_eq!(bounded.nodes, 2);
        assert_eq!(bounded.completed_depth, 0);
        assert!(engine.tt.iter().all(Option::is_none));
    }

    #[test]
    fn all_check_evasions_including_quiet_moves_are_searched() {
        // Black is checked by Ra1 and the only evasion is quiet Kb8.
        let position = Position::from_fen("k7/8/1K6/8/8/8/8/R7 b - - 0 1").unwrap();
        assert!(position.in_check());
        let legal = position.legal_moves();
        assert!(!legal.is_empty());
        assert!(legal.iter().all(|mv| !is_tactical(&position, *mv)));
        let mut position = position;
        let original = position.snapshot();
        let mut cpu = engine();
        let cancel = AtomicBool::new(false);
        let mut control = Control {
            limits: limits(1),
            cancellation: &cancel,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: cpu.evaluator.initialize(&position).unwrap(),
        };
        let value = cpu
            .quiescence(&mut position, 0, 4, -INFINITY, INFINITY, &mut control)
            .unwrap();
        assert!(!value.pv.is_empty());
        assert!(legal.contains(&value.pv[0]));
        assert!(original.same_state(&position.snapshot()));
        let stopped = cpu.quiescence(&mut position, 0, 0, -INFINITY, INFINITY, &mut control);
        assert!(matches!(
            stopped,
            Err(Abort::Stop(CpuCompletion::QuiescenceLimit))
        ));
    }

    #[test]
    fn root_restrictions_cover_castling_en_passant_and_four_promotions() {
        let cases = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", vec!["e1g1", "e1c1"]),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", vec!["e5d6"]),
            (
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                vec!["a7a8q", "a7a8r", "a7a8b", "a7a8n"],
            ),
        ];
        let mut cpu = engine();
        for (fen, moves) in cases {
            let position = Position::from_fen(fen).unwrap();
            let before = position.snapshot();
            for text in moves {
                let mv = BoardMove::from_uci(text).unwrap();
                let report = cpu
                    .analyze_root_moves(&position, &[mv], limits(1), &AtomicBool::new(false))
                    .unwrap();
                assert_eq!(report.best_move, Some(mv));
                assert_eq!(report.completed_depth, 1);
                assert!(report.root_restricted);
                assert!(before.same_state(&position.snapshot()));
            }
        }
    }

    #[test]
    fn exact_history_and_remaining_depth_are_required_for_tt_reuse() {
        let mut played = Position::startpos();
        played
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        let imported = Position::from_fen(&played.to_fen()).unwrap();
        assert_ne!(played.position_identity(), imported.position_identity());
        let mut cpu = engine();
        let hash = tt_hash(&played);
        cpu.store(TtEntry {
            hash,
            identity: played.position_identity(),
            profile: CpuProfile::PlanAssisted,
            value_identity: Arc::clone(&cpu.value_identity),
            depth: 8,
            score: 1234,
            bound: Bound::Exact,
            best_move: played.legal_moves().first().copied(),
        });
        assert!(cpu.probe(hash, &played.position_identity()).is_some());
        assert!(cpu.probe(hash, &imported.position_identity()).is_none());
        let mut deeper = played.clone();
        let cancel = AtomicBool::new(false);
        let mut control = Control {
            limits: CpuLimits {
                max_nodes: 1,
                ..limits(10)
            },
            cancellation: &cancel,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: cpu.evaluator.initialize(&deeper).unwrap(),
        };
        let result = cpu.negamax(&mut deeper, 10, 0, -INFINITY, INFINITY, &mut control);
        assert!(matches!(result, Err(Abort::Stop(CpuCompletion::NodeLimit))));
        assert_eq!(
            cpu.probe(hash, &played.position_identity()).unwrap().depth,
            8
        );
    }

    #[test]
    fn resume_retains_completed_depth_and_rejects_changed_history_or_profile() {
        let mut cpu = engine();
        let position = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let first = cpu
            .analyze(&position, limits(1), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(first.completed_depth, 1);
        let token = first.resume.unwrap();
        let resumed = cpu
            .resume(&position, &token, limits(2), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(resumed.completed_depth, 2);
        let changed = position.preview_move(position.legal_moves()[0]).unwrap();
        assert!(matches!(
            cpu.resume(&changed, &token, limits(2), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        let mut independent = CpuEngine::new(CpuConfig {
            profile: CpuProfile::Independent,
            ..CpuConfig::default()
        })
        .unwrap();
        assert!(matches!(
            independent.resume(&position, &token, limits(2), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        for config in [
            CpuConfig {
                tt_entries: cpu.config.tt_entries + 1,
                ..cpu.config.clone()
            },
            CpuConfig {
                max_depth: cpu.config.max_depth - 1,
                ..cpu.config.clone()
            },
        ] {
            let mut changed_conditions = CpuEngine::new(config).unwrap();
            assert!(matches!(
                changed_conditions.resume(&position, &token, limits(2), &AtomicBool::new(false)),
                Err(CpuError::ResumeMismatch(_))
            ));
        }
        let canceled = cpu
            .resume(&position, &token, limits(1), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(canceled.completion, CpuCompletion::Canceled);
        assert_eq!(canceled.completed_depth, 1);
        assert_eq!(canceled.reused_completed_depth, 1);
        assert_eq!(canceled.nodes, 0);
        let expired = cpu
            .resume(
                &position,
                &token,
                CpuLimits {
                    deadline: Some(Instant::now()),
                    ..limits(1)
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(expired.completion, CpuCompletion::Deadline);
    }

    #[test]
    fn progress_observer_reports_only_fallback_and_completed_iterations() {
        let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let mut progress = Vec::new();
        let report = engine()
            .analyze_with_progress(&position, limits(1), &AtomicBool::new(false), |value| {
                progress.push(value)
            })
            .unwrap();
        assert_eq!(progress.len(), 2);
        assert_eq!(progress[0].completed_depth, 0);
        assert_eq!(progress[1].completed_depth, 1);
        assert_eq!(progress[1].best_move, report.best_move);
        assert_eq!(progress[1].score, report.score);
        let mut stopped = Vec::new();
        let report = engine()
            .analyze_with_progress(
                &position,
                CpuLimits {
                    max_nodes: 1,
                    ..limits(2)
                },
                &AtomicBool::new(false),
                |value| stopped.push(value),
            )
            .unwrap();
        assert_eq!(report.completion, CpuCompletion::NodeLimit);
        assert_eq!(stopped.len(), 1);
        assert_eq!(stopped[0].completed_depth, 0);
        let canceled = AtomicBool::new(false);
        let report = engine()
            .analyze_with_progress(&position, limits(1), &canceled, |value| {
                if value.completed_depth == 1 {
                    canceled.store(true, Ordering::Release);
                }
            })
            .unwrap();
        assert_eq!(report.completion, CpuCompletion::Canceled);
        assert_eq!(report.completed_depth, 1);
        assert_eq!(report.score, CPU_MATE_SCORE - 1);
    }

    #[test]
    fn exact_float_delta_and_fresh_state_reuse_the_same_tt_value() {
        use crate::cpu_value::{CpuFloatCheckpoint, CpuFloatValue};
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        for (square, kind, weight) in [
            (0, PieceKind::Rook, 1_000_000.0),
            (1, PieceKind::Knight, 0.03),
            (6, PieceKind::Knight, -1_000_000.0),
            (18, PieceKind::Knight, 0.04),
        ] {
            checkpoint.feature_weights[(kind as usize * 64 + square) * 256] = weight;
        }
        checkpoint.hidden_weights[0] = 1.0;
        checkpoint.hidden_weights[256] = 1.0;
        checkpoint.output_weights[0] = 1.0;
        let evaluator = Arc::new(CpuFloatValue::new(checkpoint).unwrap());
        let mut position = Position::startpos();
        let mut incremental = evaluator.initialize(&position).unwrap();
        let rules = position.make_uci("b1c3").unwrap();
        evaluator
            .apply_delta(&mut incremental, rules.delta())
            .unwrap();
        let mut cpu = CpuEngine::with_evaluator(
            CpuConfig {
                tt_entries: 128,
                ..CpuConfig::default()
            },
            evaluator.clone(),
        )
        .unwrap();
        let cancel = AtomicBool::new(false);
        let mut control = Control {
            limits: limits(1),
            cancellation: &cancel,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: incremental,
        };
        let first = cpu
            .negamax(&mut position, 1, 0, -INFINITY, INFINITY, &mut control)
            .unwrap();
        let mut fresh = Control {
            limits: limits(1),
            cancellation: &cancel,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: evaluator.initialize(&position).unwrap(),
        };
        assert_eq!(control.value.values(), fresh.value.values());
        let cached = cpu
            .negamax(&mut position, 1, 0, -INFINITY, INFINITY, &mut fresh)
            .unwrap();
        assert_eq!(cached.score, first.score);
        assert_eq!(fresh.tt_hits, 1);
        assert_eq!(fresh.nodes, 1);
        let mut uncached = Control {
            limits: limits(1),
            cancellation: &cancel,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: evaluator.initialize(&position).unwrap(),
        };
        let actual = CpuEngine::with_evaluator(
            CpuConfig {
                tt_entries: 0,
                ..CpuConfig::default()
            },
            evaluator,
        )
        .unwrap()
        .negamax(&mut position, 1, 0, -INFINITY, INFINITY, &mut uncached)
        .unwrap();
        assert_eq!(actual.score, cached.score);
    }

    #[test]
    fn tt_does_not_change_full_search_score_and_pv_stays_legal() {
        let position = Position::from_fen("4k3/8/8/8/8/3p4/4P3/4K3 w - - 0 1").unwrap();
        let before = position.snapshot();
        let with = engine()
            .analyze(&position, limits(3), &AtomicBool::new(false))
            .unwrap();
        let without = CpuEngine::new(CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        })
        .unwrap()
        .analyze(&position, limits(3), &AtomicBool::new(false))
        .unwrap();
        assert_eq!(with.completed_depth, 3);
        assert_eq!(with.score, without.score);
        assert_eq!(with.best_move, without.best_move);
        let mut replay = position.clone();
        for mv in with.pv {
            replay.make_move(mv).unwrap();
        }
        assert!(before.same_state(&position.snapshot()));
    }

    #[test]
    fn mate_distance_round_trip_is_root_independent() {
        for score in [CPU_MATE_SCORE - 5, -CPU_MATE_SCORE + 9, 100, -300] {
            assert_eq!(from_tt_score(to_tt_score(score, 3), 3), score);
        }
        assert_eq!(
            from_tt_score(to_tt_score(CPU_MATE_SCORE - 5, 3), 1),
            CPU_MATE_SCORE - 3
        );
    }

    #[test]
    fn failed_search_retains_actual_work_and_clear_removes_observation() {
        use crate::cpu_value::{CpuAccumulatorUndo, CpuValueEvaluator};
        use rz_position::RuleMoveDelta;
        use std::sync::atomic::AtomicUsize;
        struct InjectFailure {
            base: BootstrapCpuValue,
            scores: AtomicUsize,
        }
        impl CpuValueEvaluator for InjectFailure {
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
                if self.scores.fetch_add(1, Ordering::Relaxed) > 0 {
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
        let mut cpu = CpuEngine::with_evaluator(
            CpuConfig::default(),
            Arc::new(InjectFailure {
                base: BootstrapCpuValue::default(),
                scores: AtomicUsize::new(0),
            }),
        )
        .unwrap();
        assert_eq!(cpu.last_attempt_work(), None);
        assert!(matches!(
            cpu.analyze(&position, limits(1), &AtomicBool::new(false)),
            Err(CpuError::Value(CpuValueError::NonFiniteForward))
        ));
        let actual = cpu.last_attempt_work().unwrap();
        assert!(actual.nodes > 0 && actual.nodes <= limits(1).max_nodes);
        assert!(actual.quiescence_nodes > 0 && actual.quiescence_nodes <= actual.nodes);
        assert!(before.same_state(&position.snapshot()));
        assert!(
            cpu.analyze(
                &position,
                CpuLimits {
                    max_nodes: 0,
                    ..limits(1)
                },
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(cpu.last_attempt_work(), Some(CpuWork::default()));
        cpu.clear();
        assert_eq!(cpu.last_attempt_work(), None);
    }

    #[test]
    fn cpu_checker_is_replaceable_and_unsupported_resume_is_explicit() {
        struct NoResumeMock {
            base: CpuEngine,
        }
        impl CpuSearcher for NoResumeMock {
            fn config(&self) -> &CpuConfig {
                self.base.config()
            }
            fn value_identity(&self) -> &CpuValueIdentity {
                self.base.value_identity()
            }
            fn search_identity(&self) -> &'static str {
                "explicit-cpu-boundary-mock-v1"
            }
            fn search_conditions(&self) -> String {
                format!(
                    "explicit-no-resume-test-wrapper;{}",
                    self.base.search_conditions()
                )
            }
            fn capabilities(&self) -> CpuCapabilities {
                CpuCapabilities {
                    root_moves: false,
                    divergence: false,
                    completed_iteration_resume: false,
                    ..self.base.capabilities()
                }
            }
            fn clear(&mut self) {
                self.base.clear();
            }
            fn analyze(
                &mut self,
                p: &Position,
                l: CpuLimits,
                c: &AtomicBool,
            ) -> Result<CpuReport, CpuError> {
                let mut r = self.base.analyze(p, l, c)?;
                r.search_version = self.search_identity();
                r.resume = None;
                Ok(r)
            }
        }
        let position = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let mut checker: Box<dyn CpuSearcher> = Box::new(engine());
        let first = checker
            .analyze(&position, limits(1), &AtomicBool::new(false))
            .unwrap();
        let token = first.resume.unwrap();
        checker = Box::new(NoResumeMock { base: engine() });
        assert!(!checker.capabilities().completed_iteration_resume);
        assert!(matches!(
            checker.resume(&position, &token, limits(2), &AtomicBool::new(false)),
            Err(CpuError::Unsupported(_))
        ));
        let second = checker
            .analyze(&position, limits(1), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(second.search_version, "explicit-cpu-boundary-mock-v1");
        assert_eq!(second.value_identity, first.value_identity);
        assert_eq!(checker.last_attempt_work(), None);
    }

    #[test]
    fn tt_slot_allocation_bytes_are_bounded_layout_not_peak() {
        let slot_bytes = std::mem::size_of::<Option<TtEntry>>();
        for entries in [0, 1, 8192, MAX_TT_ENTRIES] {
            let config = CpuConfig {
                tt_entries: entries,
                ..CpuConfig::default()
            };
            assert_eq!(
                config.tt_allocation_bytes().unwrap(),
                (entries * slot_bytes) as u64
            );
        }
        for entries in [MAX_TT_ENTRIES + 1, usize::MAX] {
            assert!(matches!(
                CpuConfig {
                    tt_entries: entries,
                    ..CpuConfig::default()
                }
                .tt_allocation_bytes(),
                Err(CpuError::InvalidConfig(_))
            ));
        }
        assert_eq!(checked_inline_slot_bytes(0, usize::MAX).unwrap(), 0);
        assert!(matches!(
            checked_inline_slot_bytes(usize::MAX, 2),
            Err(CpuError::InvalidConfig(
                "TT slot bytes overflow address space"
            ))
        ));
    }

    #[test]
    fn invalid_budgets_and_root_moves_are_not_silent_fallbacks() {
        assert!(
            CpuEngine::new(CpuConfig {
                tt_entries: MAX_TT_ENTRIES + 1,
                ..CpuConfig::default()
            })
            .is_err()
        );
        let position = Position::startpos();
        let mut cpu = engine();
        assert!(
            cpu.analyze(
                &position,
                CpuLimits {
                    max_nodes: 0,
                    ..limits(1)
                },
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(
            cpu.analyze_root_moves(&position, &[], limits(1), &AtomicBool::new(false))
                .is_err()
        );
        let mv = BoardMove::from_uci("e2e4").unwrap();
        assert!(
            cpu.analyze_root_moves(&position, &[mv, mv], limits(1), &AtomicBool::new(false))
                .is_err()
        );
        let illegal = BoardMove::from_uci("e2e5").unwrap();
        assert!(
            cpu.analyze_root_moves(&position, &[illegal], limits(1), &AtomicBool::new(false))
                .is_err()
        );
    }
}
