//! Restricted-game PALS refinement. These edges are examined continuations,
//! not PUCT visits. Only Rules can certify a terminal position.
use super::store::{
    BoundKind, EvidenceScope, LineId, Observation, ObservationId, ObservationKind, PalsStores,
    RawScore, SituationId, StateId, StoreError, StoreLimits, TaskAdmission, TaskConsumer, TaskKey,
    TaskQuestion,
};
use crate::cpu::{CPU_MATE_SCORE, CpuEngine, CpuError, CpuLimits, CpuResumeToken, CpuScoreScope};
use rz_position::{BoardMove, Color, PlayStatus, Position, PositionError, TerminalReason};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const PALS_SEARCH_VERSION: &str = "pals-restricted-refinement/0.1";

#[derive(Clone, Debug)]
pub struct PalsConfig {
    pub beam_width: usize,
    pub line_plies: usize,
    pub max_nodes: usize,
    pub max_records: usize,
    pub max_role_calls: u64,
    pub cpu_nodes_per_task: u64,
}
impl Default for PalsConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            line_plies: 4,
            max_nodes: 4096,
            max_records: 128,
            max_role_calls: 4096,
            cpu_nodes_per_task: 4096,
        }
    }
}
impl PalsConfig {
    pub fn validate(&self) -> Result<(), PalsError> {
        if self.beam_width == 0
            || self.beam_width > 16
            || self.line_plies == 0
            || self.line_plies > 16
            || self.max_nodes < 257
            || self.max_nodes > 65_536
            || self.max_records < 4
            || self.max_records > 128
            || self.max_role_calls == 0
            || self.max_role_calls > 1_000_000
            || self.cpu_nodes_per_task == 0
            || self.cpu_nodes_per_task > 10_000_000
        {
            return Err(PalsError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PalsLimits {
    pub deadline: Instant,
    pub max_rounds: u64,
    pub max_cpu_nodes: u64,
    pub cpu_depth: u16,
}
impl Default for PalsLimits {
    fn default() -> Self {
        Self {
            deadline: Instant::now() + Duration::from_secs(1),
            max_rounds: 16,
            max_cpu_nodes: 100_000,
            cpu_depth: 2,
        }
    }
}

/// Model estimates can guide the restricted frontier but cannot issue Rules facts.
#[derive(Clone, Debug, PartialEq)]
pub struct RoleEvaluation {
    pub logits: Vec<f32>,
    pub wdl: [f32; 3],
}
impl RoleEvaluation {
    fn validate(&self, candidates: usize) -> Result<(), RoleError> {
        if self.logits.len() != candidates
            || self.logits.iter().any(|v| !v.is_finite())
            || self
                .wdl
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || (self.wdl.iter().sum::<f32>() - 1.0).abs() > 1e-4
        {
            return Err(RoleError::InvalidOutput);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RecordKind {
    Proposal,
    Counterexample,
    Repair,
    CpuVerification,
}

/// Public records have explicit examination status; an absent value is unknown.
#[derive(Clone, Debug)]
pub struct RoleRecord {
    pub revision: u64,
    /// Anchor of this relative continuation; old-root facts retain provenance.
    pub origin_state: StateId,
    pub kind: RecordKind,
    pub line: Vec<BoardMove>,
    pub value: Option<i32>,
    pub completed_depth: u16,
    pub score_scope: Option<CpuScoreScope>,
    pub perspective: Color,
    pub critical: bool,
}

pub struct RoleQuery<'a> {
    pub position: &'a Position,
    /// Exact legal order for this prefix, never a neural action vocabulary.
    pub legal: &'a [BoardMove],
    pub prefix: &'a [BoardMove],
    pub proposal: &'a [BoardMove],
    pub counterexample: Option<&'a [BoardMove]>,
    pub records: &'a [RoleRecord],
    pub revision: u64,
    pub deadline: Instant,
    pub cancel: &'a AtomicBool,
}
impl RoleQuery<'_> {
    pub fn check_control(&self) -> Result<(), RoleError> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(RoleError::Canceled);
        }
        if Instant::now() >= self.deadline {
            return Err(RoleError::Deadline);
        }
        Ok(())
    }
}
pub struct DivergenceQuery<'a> {
    pub root: &'a Position,
    pub proposal: &'a [BoardMove],
    /// Ply indices whose side to move differs from the root side to move.
    pub candidates: &'a [usize],
    pub records: &'a [RoleRecord],
    pub revision: u64,
    pub deadline: Instant,
    pub cancel: &'a AtomicBool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RoleError {
    Unavailable,
    InvalidOutput,
    Canceled,
    Deadline,
    /// The provider retains/quarantines its lease; the engine cannot certify
    /// drain, reuse this owner, or emit a normally completed game receipt.
    PhysicalCompletionUnknown,
    Backend(String),
}
impl std::fmt::Display for RoleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RoleError {}

/// Start-up-selected P/C implementation. No default neural-to-mock fallback.
/// Implementations must finish/drain their physical work before returning;
/// logical cancellation alone never permits buffer/session reuse.
pub trait RoleModel: Send {
    fn identity(&self) -> &str;
    fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError>;
    fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError>;
    fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError>;
    fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError>;
    /// Accounting-only acknowledgement for the most recent output. Returning
    /// from inference is not consumption: the search calls this exactly once
    /// after output validation and its final deadline/cancellation acceptance.
    /// This hook must not dispatch work or perform a search.
    fn accepted_output(&mut self) {}
    fn new_game(&mut self) {}
}

impl<M: RoleModel + ?Sized> RoleModel for Box<M> {
    fn identity(&self) -> &str {
        (**self).identity()
    }
    fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        (**self).propose(query)
    }
    fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        (**self).reply(query)
    }
    fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        (**self).repair(query)
    }
    fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
        (**self).divergences(query)
    }
    fn accepted_output(&mut self) {
        (**self).accepted_output();
    }
    fn new_game(&mut self) {
        (**self).new_game();
    }
}

/// Explicit CPU/mock selection for contract and scheduler validation only.
#[derive(Clone, Debug, Default)]
pub struct LegalOrderRoleMock;
impl LegalOrderRoleMock {
    fn evaluate(query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        query.check_control()?;
        Ok(RoleEvaluation {
            logits: (0..query.legal.len()).map(|i| -(i as f32)).collect(),
            wdl: [0.25, 0.5, 0.25],
        })
    }
}
impl RoleModel for LegalOrderRoleMock {
    fn identity(&self) -> &str {
        "explicit-legal-order-role-mock-v1"
    }
    fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        Self::evaluate(query)
    }
    fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        Self::evaluate(query)
    }
    fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        let mut result = Self::evaluate(query)?;
        // The mock exercises a changed continuation without changing its root move.
        result.logits.reverse();
        Ok(result)
    }
    fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
        if query.cancel.load(Ordering::Acquire) {
            return Err(RoleError::Canceled);
        }
        if Instant::now() >= query.deadline {
            return Err(RoleError::Deadline);
        }
        Ok((0..query.candidates.len()).map(|i| -(i as f32)).collect())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PalsCounters {
    pub rounds: u64,
    pub proposals: u64,
    pub refutations: u64,
    pub repairs: u64,
    pub supported_refutations: u64,
    pub supported_repairs: u64,
    pub role_calls: u64,
    pub proposer_calls: u64,
    /// Includes critic divergence ranking and legal reply ranking calls.
    pub critic_calls: u64,
    pub repair_calls: u64,
    /// Shape-valid role outputs returned, before final logical acceptance.
    pub completed_proposer_calls: u64,
    pub completed_critic_calls: u64,
    pub completed_repair_calls: u64,
    pub consumed_role_outputs: u64,
    pub accepted_proposer_outputs: u64,
    pub accepted_critic_outputs: u64,
    pub accepted_repair_outputs: u64,
    /// CPU API calls started, including calls which return CpuError.
    pub cpu_tasks_requested: u64,
    /// Physical CPU reports returned; not a requested-coverage completion count.
    pub cpu_tasks: u64,
    /// A CPU call failed without returning its actual node/work report. Reported
    /// counts are then a lower bound, not a complete observation of CPU work.
    pub cpu_work_observation_incomplete: bool,
    pub cpu_nodes: u64,
    pub cpu_quiescence_nodes: u64,
    pub cpu_tt_hits: u64,
    /// New physical CPU executions satisfying the requested depth and scope.
    /// A completed shallower iteration is retained as partial evidence instead.
    pub completed_cpu_tasks: u64,
    /// Physical reports with a completed iteration below the requested depth.
    pub partial_cpu_iterations: u64,
    /// Completed TaskTable consumer results accepted into the current search;
    /// includes completed-task reuse, excludes partial/frontier estimates.
    pub consumed_cpu_tasks: u64,
    /// Only TaskAdmission::Reuse accepted consumers; excludes the existing-node
    /// evidence fast path. Direct new-completion consumption is total minus this.
    pub reused_completed_cpu_tasks_consumed: u64,
    /// Accepted shallower completed-iteration values, never completed requests.
    pub consumed_partial_cpu_values: u64,
    /// Accepted provisional values with no completed CPU iteration.
    pub consumed_frontier_cpu_values: u64,
    /// Existing depth-sufficient evidence or completed-task reuse accepted here.
    pub consumed_cached_cpu_values: u64,
    pub evidence_cache_hits: u64,
    pub examined_edges: u64,
    pub retained_situations: usize,
    /// Root-value/unknown-child accounting inspected every legal root move.
    /// This is observation coverage, never a claim that every branch is solved.
    pub root_scope_observation_complete: bool,
    pub unknown_root_children: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PalsCompletion {
    RoundLimit,
    CpuNodeLimit,
    RoleCallLimit,
    Deadline,
    Canceled,
    Capacity,
    Terminal,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PalsValueScope {
    Unknown,
    RestrictedEstimate,
    RulesTerminal,
}
#[derive(Clone, Debug)]
pub struct PalsRootValue {
    pub movement: BoardMove,
    pub score: Option<i32>,
    pub scope: PalsValueScope,
    pub examined_replies: usize,
    /// None means the child was not materialized/examined, not zero replies.
    pub unexplored_replies: Option<usize>,
}
#[derive(Clone, Debug)]
pub struct PalsResult {
    pub best_move: Option<BoardMove>,
    pub score: Option<i32>,
    pub value_scope: PalsValueScope,
    pub terminal: Option<TerminalReason>,
    pub completion: PalsCompletion,
    pub counters: PalsCounters,
    pub root_values: Vec<PalsRootValue>,
    pub elapsed: Duration,
    pub model_identity: String,
}
#[derive(Debug)]
pub enum PalsError {
    InvalidConfig,
    InvalidLimits,
    Rules(PositionError),
    Cpu(CpuError),
    Role(RoleError),
    Store(StoreError),
    Capacity,
    RoleCallLimit,
}
impl std::fmt::Display for PalsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS: {self:?}")
    }
}
impl std::error::Error for PalsError {}
impl From<PositionError> for PalsError {
    fn from(e: PositionError) -> Self {
        Self::Rules(e)
    }
}
impl From<CpuError> for PalsError {
    fn from(e: CpuError) -> Self {
        Self::Cpu(e)
    }
}
impl From<RoleError> for PalsError {
    fn from(e: RoleError) -> Self {
        Self::Role(e)
    }
}
impl From<StoreError> for PalsError {
    fn from(e: StoreError) -> Self {
        if matches!(e, StoreError::Capacity(_)) {
            Self::Capacity
        } else {
            Self::Store(e)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    movement: BoardMove,
    child: usize,
}
struct Node {
    position: Position,
    state: StateId,
    situation: SituationId,
    terminal: Option<(TerminalReason, i32)>,
    edges: Vec<Edge>,
    evidence: Option<CpuEvidence>,
    resume: Option<CpuResumeToken>,
}
#[derive(Clone, Debug)]
struct CpuEvidence {
    score: i32,
    depth: u16,
    scope: CpuScoreScope,
}
#[derive(Clone, Copy)]
enum Call {
    Propose,
    Reply,
    Repair,
}

/// Persistent exact situations and public evidence survive a normal root change.
/// `new_game` is the only automatic game-wide invalidation boundary.
pub struct PalsEngine<M: RoleModel> {
    config: PalsConfig,
    model: M,
    cpu: CpuEngine,
    nodes: Vec<Node>,
    records: Vec<RoleRecord>,
    revision: u64,
    stores: PalsStores,
    clock_origin: Instant,
    consumer_id: u64,
    last_search_counters: Option<PalsCounters>,
}
impl<M: RoleModel> PalsEngine<M> {
    pub fn new(config: PalsConfig, model: M, cpu: CpuEngine) -> Result<Self, PalsError> {
        config.validate()?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve(config.max_nodes)
            .map_err(|_| PalsError::Capacity)?;
        let mut records = Vec::new();
        records
            .try_reserve(config.max_records)
            .map_err(|_| PalsError::Capacity)?;
        let stores = PalsStores::new(Self::store_limits(&config));
        Ok(Self {
            config,
            model,
            cpu,
            nodes,
            records,
            revision: 0,
            stores,
            clock_origin: Instant::now(),
            consumer_id: 0,
            last_search_counters: None,
        })
    }
    pub fn new_game(&mut self) {
        self.nodes.clear();
        self.records.clear();
        self.cpu.clear();
        self.model.new_game();
        self.revision = 0;
        self.stores = PalsStores::new(Self::store_limits(&self.config));
        self.consumer_id = 0;
        self.last_search_counters = None;
    }
    pub fn retained_situations(&self) -> usize {
        self.nodes.len()
    }
    pub fn records(&self) -> &[RoleRecord] {
        &self.records
    }
    pub fn model_identity(&self) -> &str {
        self.model.identity()
    }
    /// Snapshot of the most recent entered search, including errors. Read after
    /// that search returns; the caller owns cumulative per-process accounting.
    pub fn last_search_counters(&self) -> Option<PalsCounters> {
        self.last_search_counters
    }
    pub fn stores(&self) -> &PalsStores {
        &self.stores
    }
    /// Derived neural cache eviction does not remove raw CPU/Rules observations.
    pub fn evict_derived_cache(&mut self) -> usize {
        self.stores.evict_derived_cache()
    }

    fn store_limits(config: &PalsConfig) -> StoreLimits {
        StoreLimits {
            states: config.max_nodes,
            situations: config.max_nodes,
            line_chunks: config.max_nodes * 16,
            line_plies: config.line_plies,
            observations: config.max_nodes * 16,
            dependency_edges: config.max_nodes * 16,
            executions: config.max_nodes * 8,
            consumers: config.max_nodes * 16,
            ..StoreLimits::default()
        }
    }

    pub fn search(
        &mut self,
        position: &Position,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<PalsResult, PalsError> {
        self.search_with_progress(position, limits, cancel, |_| {})
    }
    /// Publishes only choices observed before the logical deadline/cancellation.
    /// UCI can preserve this last valid choice while the physical owner drains.
    pub fn search_with_progress<F: FnMut(BoardMove)>(
        &mut self,
        position: &Position,
        limits: PalsLimits,
        cancel: &AtomicBool,
        progress: F,
    ) -> Result<PalsResult, PalsError> {
        let mut counters = PalsCounters::default();
        let result = self.search_inner(position, limits, cancel, progress, &mut counters);
        if let Ok(report) = &result {
            counters = report.counters;
        }
        counters.retained_situations = self.nodes.len();
        self.last_search_counters = Some(counters);
        result
    }
    fn search_inner<F: FnMut(BoardMove)>(
        &mut self,
        position: &Position,
        limits: PalsLimits,
        cancel: &AtomicBool,
        mut progress: F,
        counters: &mut PalsCounters,
    ) -> Result<PalsResult, PalsError> {
        if limits.max_rounds == 0
            || limits.max_cpu_nodes == 0
            || limits.cpu_depth == 0
            || limits.cpu_depth > self.cpu.config().max_depth
        {
            return Err(PalsError::InvalidLimits);
        }
        let started = Instant::now();
        match self.stores.focus_actual_moves(position.snapshot()) {
            Ok(_) => {}
            Err(StoreError::Capacity(_)) => return self.capacity_result(position, started),
            Err(error) => return Err(error.into()),
        }
        let root = match self.intern(position.clone()) {
            Ok(root) => root,
            Err(PalsError::Capacity) => return self.capacity_result(position, started),
            Err(error) => return Err(error),
        };
        self.refresh_projection(self.nodes[root].state);
        let legal = position.legal_moves();
        if let Some((reason, value)) = self.nodes[root].terminal {
            counters.retained_situations = self.nodes.len();
            counters.root_scope_observation_complete = true;
            return Ok(PalsResult {
                best_move: None,
                score: Some(value),
                value_scope: PalsValueScope::RulesTerminal,
                terminal: Some(reason),
                completion: PalsCompletion::Terminal,
                counters: *counters,
                root_values: Vec::new(),
                elapsed: started.elapsed(),
                model_identity: self.model.identity().to_owned(),
            });
        }
        // Check every immediate legal child for actual Rules terminal evidence.
        // A model's ranking can never hide a mate-in-one or promote a nonmate.
        let mut completion = PalsCompletion::RoundLimit;
        for &movement in &legal {
            if self.stopped(limits, cancel).is_some() {
                break;
            }
            let mut child = position.clone();
            child.make_move(movement)?;
            if matches!(
                child.classify_position()?.play_status,
                PlayStatus::Terminal { .. }
            ) {
                match self.connect(root, movement, child, counters) {
                    Ok(_) => {}
                    Err(PalsError::Capacity) => {
                        completion = PalsCompletion::Capacity;
                        break;
                    }
                    Err(error) => return Err(error),
                }
                match self.publish_choice(root, limits, cancel, &mut progress) {
                    Ok(()) => {}
                    Err(PalsError::Role(RoleError::Canceled)) => {
                        completion = PalsCompletion::Canceled;
                        break;
                    }
                    Err(PalsError::Role(RoleError::Deadline)) => {
                        completion = PalsCompletion::Deadline;
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        for round in 0..limits.max_rounds {
            if completion == PalsCompletion::Capacity {
                break;
            }
            if let Some(stop) = self.stopped(limits, cancel) {
                completion = stop;
                break;
            }
            if counters.cpu_nodes >= limits.max_cpu_nodes {
                completion = PalsCompletion::CpuNodeLimit;
                break;
            }
            if counters.role_calls >= self.config.max_role_calls {
                completion = PalsCompletion::RoleCallLimit;
                break;
            }
            let result = self.refine(root, round, limits, cancel, counters, &mut progress);
            match result {
                Ok(()) => counters.rounds += 1,
                Err(PalsError::Role(RoleError::Canceled)) => {
                    completion = PalsCompletion::Canceled;
                    break;
                }
                Err(PalsError::Role(RoleError::Deadline)) => {
                    completion = PalsCompletion::Deadline;
                    break;
                }
                Err(PalsError::Capacity) => {
                    completion = PalsCompletion::Capacity;
                    break;
                }
                Err(PalsError::RoleCallLimit) => {
                    completion = PalsCompletion::RoleCallLimit;
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        let mut root_values = Vec::with_capacity(legal.len());
        let mut best = legal.first().copied();
        let mut score = None;
        let mut scope = PalsValueScope::Unknown;
        for &movement in &legal {
            let edge = self.nodes[root]
                .edges
                .iter()
                .find(|edge| edge.movement == movement);
            let (value, child_scope, examined, unexplored) = if let Some(edge) = edge {
                let child = &self.nodes[edge.child];
                let value = self
                    .value(edge.child, self.config.line_plies.saturating_sub(1))
                    .map(|v| -v);
                let child_scope = if child.terminal.is_some() {
                    PalsValueScope::RulesTerminal
                } else if value.is_some() {
                    PalsValueScope::RestrictedEstimate
                } else {
                    PalsValueScope::Unknown
                };
                let legal_count = if child.terminal.is_some() {
                    0
                } else {
                    child.position.legal_moves().len()
                };
                (
                    value,
                    child_scope,
                    child.edges.len(),
                    Some(legal_count.saturating_sub(child.edges.len())),
                )
            } else {
                (None, PalsValueScope::Unknown, 0, None)
            };
            if value.is_none() {
                counters.unknown_root_children += 1;
            }
            if let Some(value) = value {
                if better_root_choice(value, child_scope, score, scope) {
                    best = Some(movement);
                    score = Some(value);
                    scope = child_scope;
                }
            }
            root_values.push(PalsRootValue {
                movement,
                score: value,
                scope: child_scope,
                examined_replies: examined,
                unexplored_replies: unexplored,
            });
        }
        counters.retained_situations = self.nodes.len();
        counters.root_scope_observation_complete = true;
        Ok(PalsResult {
            best_move: best,
            score,
            value_scope: scope,
            terminal: None,
            completion,
            counters: *counters,
            root_values,
            elapsed: started.elapsed(),
            model_identity: self.model.identity().to_owned(),
        })
    }

    fn stopped(&self, limits: PalsLimits, cancel: &AtomicBool) -> Option<PalsCompletion> {
        if cancel.load(Ordering::Acquire) {
            Some(PalsCompletion::Canceled)
        } else if Instant::now() >= limits.deadline {
            Some(PalsCompletion::Deadline)
        } else {
            None
        }
    }
    fn capacity_result(
        &self,
        position: &Position,
        started: Instant,
    ) -> Result<PalsResult, PalsError> {
        let classification = position.classify_position()?;
        let terminal = match classification.play_status {
            PlayStatus::Ongoing => None,
            PlayStatus::Terminal { reason, winner } => Some((
                reason,
                match winner {
                    Some(color) if color == position.side_to_move() => CPU_MATE_SCORE,
                    Some(_) => -CPU_MATE_SCORE,
                    None => 0,
                },
            )),
        };
        Ok(PalsResult {
            best_move: if terminal.is_none() {
                position.legal_moves().first().copied()
            } else {
                None
            },
            score: terminal.map(|(_, score)| score),
            value_scope: if terminal.is_some() {
                PalsValueScope::RulesTerminal
            } else {
                PalsValueScope::Unknown
            },
            terminal: terminal.map(|(reason, _)| reason),
            completion: if terminal.is_some() {
                PalsCompletion::Terminal
            } else {
                PalsCompletion::Capacity
            },
            counters: PalsCounters {
                retained_situations: self.nodes.len(),
                unknown_root_children: if terminal.is_some() {
                    0
                } else {
                    position.legal_moves().len()
                },
                root_scope_observation_complete: true,
                ..PalsCounters::default()
            },
            root_values: if terminal.is_some() {
                Vec::new()
            } else {
                position
                    .legal_moves()
                    .into_iter()
                    .map(|movement| PalsRootValue {
                        movement,
                        score: None,
                        scope: PalsValueScope::Unknown,
                        examined_replies: 0,
                        unexplored_replies: None,
                    })
                    .collect()
            },
            elapsed: started.elapsed(),
            model_identity: self.model.identity().to_owned(),
        })
    }
    fn intern(&mut self, position: Position) -> Result<usize, PalsError> {
        let snapshot = position.snapshot();
        if let Some(state) = self.stores.states.find(&snapshot) {
            if let Some(index) = self.nodes.iter().position(|node| node.state == state) {
                return Ok(index);
            }
        }
        if self.nodes.len() >= self.config.max_nodes {
            return Err(PalsError::Capacity);
        }
        let situation = if let Some(root) = self.stores.root().filter(|&root| {
            self.stores.situations.get(root).is_ok_and(|s| {
                self.stores
                    .states
                    .get(s.state)
                    .is_ok_and(|s| s.same_state(&snapshot))
            })
        }) {
            root
        } else {
            self.stores.insert_situation(snapshot)?
        };
        let state = self.stores.situations.get(situation)?.state;
        let terminal = match position.classify_position()?.play_status {
            PlayStatus::Ongoing => None,
            PlayStatus::Terminal { reason, winner } => Some((
                reason,
                match winner {
                    Some(color) if color == position.side_to_move() => CPU_MATE_SCORE,
                    Some(_) => -CPU_MATE_SCORE,
                    None => 0,
                },
            )),
        };
        if terminal.is_some() {
            self.stores.append_rules_terminal(&position, 0, 0)?;
        }
        self.nodes.push(Node {
            position,
            state,
            situation,
            terminal,
            edges: Vec::new(),
            evidence: None,
            resume: None,
        });
        Ok(self.nodes.len() - 1)
    }
    fn connect(
        &mut self,
        parent: usize,
        movement: BoardMove,
        child: Position,
        counters: &mut PalsCounters,
    ) -> Result<usize, PalsError> {
        if let Some(edge) = self.nodes[parent]
            .edges
            .iter()
            .find(|edge| edge.movement == movement)
        {
            return Ok(edge.child);
        }
        let child = self.intern(child)?;
        self.nodes[parent].edges.push(Edge { movement, child });
        counters.examined_edges += 1;
        Ok(child)
    }
    // Keep borrowed role context separate from the cancellation/deadline and
    // mutable work ledger; no role input may own or retain those controls.
    #[allow(clippy::too_many_arguments)]
    fn ranked(
        &mut self,
        node: usize,
        call: Call,
        prefix: &[BoardMove],
        proposal: &[BoardMove],
        refutation: Option<&[BoardMove]>,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<Vec<BoardMove>, PalsError> {
        if counters.role_calls >= self.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        let position = &self.nodes[node].position;
        let legal = position.legal_moves();
        if legal.len() > 256 {
            return Err(PalsError::Capacity);
        }
        let query = RoleQuery {
            position,
            legal: &legal,
            prefix,
            proposal,
            counterexample: refutation,
            records: &self.records,
            revision: self.revision,
            deadline: limits.deadline,
            cancel,
        };
        query.check_control()?;
        counters.role_calls += 1;
        match call {
            Call::Propose => counters.proposer_calls += 1,
            Call::Reply => counters.critic_calls += 1,
            Call::Repair => counters.repair_calls += 1,
        }
        let evaluation = match call {
            Call::Propose => self.model.propose(query)?,
            Call::Reply => self.model.reply(query)?,
            Call::Repair => self.model.repair(query)?,
        };
        evaluation.validate(legal.len())?;
        match call {
            Call::Propose => counters.completed_proposer_calls += 1,
            Call::Reply => counters.completed_critic_calls += 1,
            Call::Repair => counters.completed_repair_calls += 1,
        }
        // Reject a late response even if the provider did not observe cancellation.
        if cancel.load(Ordering::Acquire) {
            return Err(RoleError::Canceled.into());
        }
        if Instant::now() >= limits.deadline {
            return Err(RoleError::Deadline.into());
        }
        self.model.accepted_output();
        counters.consumed_role_outputs += 1;
        match call {
            Call::Propose => counters.accepted_proposer_outputs += 1,
            Call::Reply => counters.accepted_critic_outputs += 1,
            Call::Repair => counters.accepted_repair_outputs += 1,
        }
        let mut indices: Vec<_> = (0..legal.len()).collect();
        indices.sort_by(|&a, &b| {
            evaluation.logits[b]
                .total_cmp(&evaluation.logits[a])
                .then(a.cmp(&b))
        });
        Ok(indices.into_iter().map(|i| legal[i]).collect())
    }
    // A bounded continuation mutates only its legal prefix and work ledger;
    // proposal/counterexample slices and logical controls remain borrowed.
    #[allow(clippy::too_many_arguments)]
    fn follow(
        &mut self,
        from: usize,
        prefix: &mut Vec<BoardMove>,
        proposal: &[BoardMove],
        refutation: Option<&[BoardMove]>,
        call: Call,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<usize, PalsError> {
        let mut current = from;
        while prefix.len() < self.config.line_plies && self.nodes[current].terminal.is_none() {
            if self.stopped(limits, cancel).is_some() {
                break;
            }
            let ranked = self.ranked(
                current, call, prefix, proposal, refutation, limits, cancel, counters,
            )?;
            let Some(movement) = ranked.first().copied() else {
                break;
            };
            let mut child = self.nodes[current].position.clone();
            child.make_move(movement)?;
            current = self.connect(current, movement, child, counters)?;
            prefix.push(movement);
        }
        Ok(current)
    }
    fn verify(
        &mut self,
        node: usize,
        line: &[BoardMove],
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<(), PalsError> {
        if self.nodes[node].terminal.is_some() {
            return Ok(());
        }
        if self.stopped(limits, cancel).is_some() {
            return Ok(());
        }
        if self.nodes[node].evidence.as_ref().is_some_and(|e| {
            e.depth >= limits.cpu_depth && e.scope == CpuScoreScope::CompletedIteration
        }) {
            counters.evidence_cache_hits += 1;
            counters.consumed_cached_cpu_values += 1;
            return Ok(());
        }
        let remaining = limits.max_cpu_nodes.saturating_sub(counters.cpu_nodes);
        if remaining == 0 || self.stopped(limits, cancel).is_some() {
            return Ok(());
        }
        let task_nodes = remaining.min(self.config.cpu_nodes_per_task);
        let profile = stable_id(self.cpu.config().profile.identity());
        let condition = stable_id(&format!(
            "{};q={}",
            crate::cpu::CPU_SEARCH_VERSION,
            self.cpu.config().quiescence_ply
        ));
        self.consumer_id = self.consumer_id.checked_add(1).ok_or(PalsError::Capacity)?;
        let consumer_id = self.consumer_id;
        let consumer = TaskConsumer {
            id: consumer_id,
            situation: self.nodes[node].situation,
            revision: self
                .stores
                .situations
                .get(self.nodes[node].situation)?
                .revision,
            generation: self.stores.generation(),
            deadline_tick: self.tick_at(limits.deadline),
        };
        let admission = match self.stores.request_task(
            TaskKey {
                state: self.nodes[node].state,
                line: None,
                question: TaskQuestion::AnalyzePosition,
                root_moves: Vec::new(),
                model: 0,
                epoch: 0,
                profile,
                condition,
                input_revision: 0,
                requested_depth: limits.cpu_depth,
                node_budget: task_nodes,
            },
            consumer,
            self.tick_at(Instant::now()),
        ) {
            Ok(admission) => admission,
            Err(StoreError::ExpiredConsumer) => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        let execution = match admission {
            TaskAdmission::Reuse {
                execution,
                observation,
            } => {
                if self.stopped(limits, cancel).is_none() {
                    match self.stores.consume_task(
                        execution,
                        consumer_id,
                        self.tick_at(Instant::now()),
                    ) {
                        Ok(_) => {}
                        Err(StoreError::ExpiredConsumer) => return Ok(()),
                        Err(error) => return Err(error.into()),
                    }
                    if self.stopped(limits, cancel).is_some() {
                        return Ok(());
                    }
                    if let (
                        RawScore::Cpu { value, .. },
                        EvidenceScope::DepthLimited { depth, .. },
                    ) = (
                        self.stores.observations.get(observation)?.score,
                        self.stores.observations.get(observation)?.scope,
                    ) {
                        self.nodes[node].evidence = Some(CpuEvidence {
                            score: value,
                            depth,
                            scope: CpuScoreScope::CompletedIteration,
                        });
                        counters.evidence_cache_hits += 1;
                        counters.consumed_cpu_tasks += 1;
                        counters.reused_completed_cpu_tasks_consumed += 1;
                        counters.consumed_cached_cpu_values += 1;
                    }
                }
                return Ok(());
            }
            TaskAdmission::Start(execution) => execution,
            TaskAdmission::Resume {
                execution,
                checkpoint,
                ..
            } => {
                if checkpoint != node as u64 || self.nodes[node].resume.is_none() {
                    return Err(
                        StoreError::InvalidConditions("CPU checkpoint owner mismatch").into(),
                    );
                }
                execution
            }
            TaskAdmission::Join(_) => {
                return Err(StoreError::InvalidConditions(
                    "single CPU owner cannot join unknown in-flight work",
                )
                .into());
            }
        };
        let cpu_limits = CpuLimits {
            max_depth: limits.cpu_depth,
            max_nodes: task_nodes,
            deadline: Some(limits.deadline),
        };
        counters.cpu_tasks_requested += 1;
        let report = match if matches!(admission, TaskAdmission::Resume { .. }) {
            self.cpu.resume(
                &self.nodes[node].position,
                self.nodes[node]
                    .resume
                    .as_ref()
                    .expect("validated CPU checkpoint"),
                cpu_limits,
                cancel,
            )
        } else {
            self.cpu
                .analyze(&self.nodes[node].position, cpu_limits, cancel)
        } {
            Ok(report) => report,
            Err(error) => {
                counters.cpu_work_observation_incomplete = true;
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
        };
        counters.cpu_tasks += 1;
        counters.cpu_nodes += report.nodes;
        counters.cpu_quiescence_nodes += report.quiescence_nodes;
        counters.cpu_tt_hits += report.tt_hits;
        let requested_coverage_complete = report.score_scope == CpuScoreScope::CompletedIteration
            && report.completed_depth >= limits.cpu_depth;
        // Actual CPU coverage is complete even when later evidence publication
        // cannot fit in a bounded store. Publication and consumption are separate.
        if requested_coverage_complete {
            counters.completed_cpu_tasks += 1;
        }
        if report.score_scope == CpuScoreScope::CompletedIteration
            && report.completed_depth > 0
            && report.completed_depth < limits.cpu_depth
        {
            counters.partial_cpu_iterations += 1;
        }
        // A stopped task retains completed depth evidence, but never pretends to
        // have examined the requested remaining depth or to prove a mate.
        let observation = match self.stores.append_observation(Observation {
            state: self.nodes[node].state,
            line: None,
            source: stable_id(report.score_provenance),
            epoch: 0,
            scope: EvidenceScope::DepthLimited {
                depth: report.completed_depth,
                profile,
                condition,
            },
            score: if report.score_scope == CpuScoreScope::FrontierOnly {
                RawScore::Estimate {
                    value: report.score as f32,
                    perspective: self.nodes[node].position.side_to_move(),
                }
            } else {
                RawScore::Cpu {
                    value: report.score,
                    perspective: self.nodes[node].position.side_to_move(),
                    bound: BoundKind::ExactWithinSearch,
                }
            },
            budget: report.nodes,
            kind: ObservationKind::CpuAnalysis,
            supersedes: None,
            execution: Some(execution),
        }) {
            Ok(observation) => observation,
            Err(error) => {
                self.nodes[node].resume = None;
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
        };
        if requested_coverage_complete {
            if let Err(error) = self.stores.complete_task(execution, observation) {
                self.nodes[node].resume = None;
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
            self.nodes[node].resume = None;
        } else if let Some(token) = report.resume {
            self.nodes[node].resume = Some(token);
            if let Err(error) = self
                .stores
                .pause_task(execution, node as u64, Some(observation))
            {
                self.nodes[node].resume = None;
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
        } else {
            self.stores.tasks.fail(execution)?;
        }
        if self.stopped(limits, cancel).is_none() {
            if requested_coverage_complete {
                match self
                    .stores
                    .consume_task(execution, consumer_id, self.tick_at(Instant::now()))
                {
                    Ok(_) => {}
                    Err(StoreError::ExpiredConsumer) => return Ok(()),
                    Err(error) => return Err(error.into()),
                }
            }
            if self.stopped(limits, cancel).is_some() {
                return Ok(());
            }
            self.nodes[node].evidence = Some(CpuEvidence {
                score: report.score,
                depth: report.completed_depth,
                scope: report.score_scope,
            });
            if requested_coverage_complete {
                counters.consumed_cpu_tasks += 1;
            } else if report.score_scope == CpuScoreScope::CompletedIteration {
                counters.consumed_partial_cpu_values += 1;
            } else if report.score_scope == CpuScoreScope::FrontierOnly {
                counters.consumed_frontier_cpu_values += 1;
            }
            // Register exactly the situation and active root which depend on this
            // evidence. Evicting derived model memory will not remove these facts.
            self.stores
                .dependencies
                .add(observation, self.nodes[node].situation)?;
            if let Some(root) = self.stores.root() {
                self.stores.dependencies.add(observation, root)?;
            }
            self.record(
                RecordKind::CpuVerification,
                line,
                Some(report.score),
                report.completed_depth,
                Some(report.score_scope),
                Some(self.nodes[node].position.side_to_move()),
            )?;
        }
        Ok(())
    }
    fn tick_at(&self, instant: Instant) -> u64 {
        instant
            .saturating_duration_since(self.clock_origin)
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64
    }
    fn record(
        &mut self,
        kind: RecordKind,
        line: &[BoardMove],
        value: Option<i32>,
        completed_depth: u16,
        score_scope: Option<CpuScoreScope>,
        perspective: Option<Color>,
    ) -> Result<Option<(LineId, ObservationId)>, PalsError> {
        self.revision = self.revision.checked_add(1).ok_or(PalsError::Capacity)?;
        let root = self.stores.root().ok_or(PalsError::Capacity)?;
        let root_state = self.stores.situations.get(root)?.state;
        let perspective = perspective.unwrap_or(self.stores.states.get(root_state)?.side_to_move());
        // This small vector is only the current public input projection. Raw
        // observations remain immutable in the independently bounded store.
        // Pin the newest proposal/counterexample/repair/CPU estimate for every
        // active first move. Drop only an obsolete input projection, never the
        // immutable observation from which it was derived.
        for record in &mut self.records {
            if record.origin_state == root_state
                && record.kind == kind
                && record.line.first() == line.first()
            {
                record.critical = false;
            }
        }
        if self.records.len() == self.config.max_records {
            let obsolete = self
                .records
                .iter()
                .position(|record| !record.critical)
                .ok_or(PalsError::Capacity)?;
            self.records.remove(obsolete);
        }
        self.records.push(RoleRecord {
            revision: self.revision,
            kind,
            line: line.to_vec(),
            origin_state: root_state,
            value,
            completed_depth,
            score_scope,
            perspective,
            critical: true,
        });
        if kind == RecordKind::CpuVerification {
            return Ok(None);
        }
        let line_id = self
            .stores
            .append_line(self.stores.situations.get(root)?.focus, line)?;
        let observation = self.stores.append_observation(Observation {
            state: root_state,
            line: Some(line_id),
            source: stable_id(self.model.identity()),
            epoch: 0,
            scope: EvidenceScope::Model {
                model: stable_id(self.model.identity()),
                encoding: stable_id("pals-role-query-v1"),
                input: self.revision,
            },
            score: RawScore::Unknown,
            budget: 0,
            kind: match kind {
                RecordKind::Proposal => ObservationKind::Proposal,
                RecordKind::Counterexample => ObservationKind::Refutation,
                RecordKind::Repair => ObservationKind::Repair,
                RecordKind::CpuVerification => unreachable!(),
            },
            supersedes: None,
            execution: None,
        })?;
        Ok(Some((line_id, observation)))
    }
    fn refresh_projection(&mut self, root_state: StateId) {
        let mut newest = HashSet::new();
        for record in self.records.iter_mut().rev() {
            record.critical = record.origin_state == root_state
                && newest.insert((record.kind, record.line.first().copied()));
        }
    }
    fn refine(
        &mut self,
        root: usize,
        round: u64,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<(), PalsError> {
        let root_rank = self.ranked(
            root,
            Call::Propose,
            &[],
            &[],
            None,
            limits,
            cancel,
            counters,
        )?;
        // Active-root width grows, while one iteration still handles at most the
        // registered beam. CPU restricted values prioritize the incumbent roots;
        // a reserved slot admits a new first move despite a low neural prior.
        let width = (self.config.beam_width + round as usize).min(root_rank.len());
        let root_color = self.nodes[root].position.side_to_move();
        let mut frontier: Vec<_> = root_rank.iter().take(width).copied().collect();
        frontier.sort_by(|a, b| {
            let value = |movement| {
                self.nodes[root]
                    .edges
                    .iter()
                    .find(|e| e.movement == movement)
                    .and_then(|edge| {
                        self.value(edge.child, self.config.line_plies.saturating_sub(1))
                    })
                    .map(|score| -score)
            };
            value(*b).cmp(&value(*a)).then_with(|| {
                root_rank
                    .iter()
                    .position(|m| m == a)
                    .cmp(&root_rank.iter().position(|m| m == b))
            })
        });
        let new_root = root_rank.iter().take(width).copied().find(|movement| {
            !self.nodes[root]
                .edges
                .iter()
                .any(|edge| edge.movement == *movement)
        });
        frontier.truncate(self.config.beam_width);
        if let Some(movement) = new_root {
            if !frontier.contains(&movement) {
                if frontier.len() == self.config.beam_width {
                    frontier.pop();
                }
                frontier.push(movement);
            }
        }
        for movement in frontier {
            if self.stopped(limits, cancel).is_some() || counters.cpu_nodes >= limits.max_cpu_nodes
            {
                break;
            }
            let mut child = self.nodes[root].position.clone();
            child.make_move(movement)?;
            let first = self.connect(root, movement, child, counters)?;
            let mut proposal = vec![movement];
            let leaf = self.follow(
                first,
                &mut proposal,
                &[],
                None,
                Call::Propose,
                limits,
                cancel,
                counters,
            )?;
            counters.proposals += 1;
            let proposal_record =
                self.record(RecordKind::Proposal, &proposal, None, 0, None, None)?;
            self.verify(leaf, &proposal, limits, cancel, counters)?;
            self.publish_choice(root, limits, cancel, progress)?;
            let proposal_value = self.completed_line_value(leaf, proposal.len(), limits.cpu_depth);
            let mut path = Vec::with_capacity(proposal.len() + 1);
            path.push(root);
            let mut current = root;
            for mv in &proposal {
                current = self.nodes[current]
                    .edges
                    .iter()
                    .find(|edge| edge.movement == *mv)
                    .expect("checked proposal edge")
                    .child;
                path.push(current);
            }
            let divergences: Vec<_> = path
                .iter()
                .enumerate()
                .filter_map(|(ply, &node)| {
                    (ply < proposal.len()
                        && self.nodes[node].terminal.is_none()
                        && self.nodes[node].position.side_to_move() != root_color)
                        .then_some(ply)
                })
                .collect();
            if divergences.is_empty() {
                continue;
            }
            if counters.role_calls >= self.config.max_role_calls {
                return Err(PalsError::RoleCallLimit);
            }
            if let Some(stop) = self.stopped(limits, cancel) {
                return Err(match stop {
                    PalsCompletion::Canceled => RoleError::Canceled,
                    _ => RoleError::Deadline,
                }
                .into());
            }
            counters.role_calls += 1;
            counters.critic_calls += 1;
            let scores = self.model.divergences(DivergenceQuery {
                root: &self.nodes[root].position,
                proposal: &proposal,
                candidates: &divergences,
                records: &self.records,
                revision: self.revision,
                deadline: limits.deadline,
                cancel,
            })?;
            if scores.len() != divergences.len() || scores.iter().any(|v| !v.is_finite()) {
                return Err(RoleError::InvalidOutput.into());
            }
            counters.completed_critic_calls += 1;
            if cancel.load(Ordering::Acquire) {
                return Err(RoleError::Canceled.into());
            }
            if Instant::now() >= limits.deadline {
                return Err(RoleError::Deadline.into());
            }
            self.model.accepted_output();
            counters.consumed_role_outputs += 1;
            counters.accepted_critic_outputs += 1;
            let mut ranked: Vec<_> = (0..divergences.len()).collect();
            ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            // Aging rotates among divergence sites: low-prior sites eventually run.
            let ply = divergences[ranked[round as usize % ranked.len()]];
            let divergence = path[ply];
            let prefix = &proposal[..ply];
            let replies = self.ranked(
                divergence,
                Call::Reply,
                prefix,
                &proposal,
                None,
                limits,
                cancel,
                counters,
            )?;
            let examined: Vec<_> = self.nodes[divergence]
                .edges
                .iter()
                .map(|e| e.movement)
                .collect();
            let response = replies
                .iter()
                .copied()
                .find(|mv| *mv != proposal[ply] && !examined.contains(mv))
                .or_else(|| replies.iter().copied().find(|mv| *mv != proposal[ply]));
            let Some(response) = response else {
                continue;
            };
            let mut state = self.nodes[divergence].position.clone();
            state.make_move(response)?;
            let response_node = self.connect(divergence, response, state, counters)?;
            let mut refutation = prefix.to_vec();
            refutation.push(response);
            let counter_leaf = self.follow(
                response_node,
                &mut refutation,
                &proposal,
                None,
                Call::Reply,
                limits,
                cancel,
                counters,
            )?;
            counters.refutations += 1;
            self.record(RecordKind::Counterexample, &refutation, None, 0, None, None)?;
            self.verify(counter_leaf, &refutation, limits, cancel, counters)?;
            self.publish_choice(root, limits, cancel, progress)?;
            let counter_value =
                self.completed_line_value(counter_leaf, refutation.len(), limits.cpu_depth);
            // The conclusion is conditional on one recorded line and on finite
            // CPU estimates. An unexamined/lower-ranked reply alone cannot refute
            // a proposal, and this never labels the first move permanently lost.
            let supported_refutation = match (proposal_record, proposal_value, counter_value) {
                (Some((line, _)), Some(old), Some(counter))
                    if counter < old && self.stopped(limits, cancel).is_none() =>
                {
                    let evidence = self.conclusion_observation(
                        root,
                        line,
                        ObservationKind::Refutation,
                        counter,
                        None,
                    )?;
                    self.stores
                        .refute_continuation(self.nodes[root].situation, line, evidence)?;
                    counters.supported_refutations += 1;
                    Some((line, evidence, counter))
                }
                _ => None,
            };
            // Repair resumes AFTER the response. The first move remains in the
            // frontier; this conditional counterexample never blacklists it.
            let mut repair = prefix.to_vec();
            repair.push(response);
            let repair_leaf = self.follow(
                response_node,
                &mut repair,
                &proposal,
                Some(&refutation),
                Call::Repair,
                limits,
                cancel,
                counters,
            )?;
            counters.repairs += 1;
            let repair_record = self.record(RecordKind::Repair, &repair, None, 0, None, None)?;
            self.verify(repair_leaf, &repair, limits, cancel, counters)?;
            self.publish_choice(root, limits, cancel, progress)?;
            if let (
                Some((old_line, old_evidence, old_value)),
                Some((new_line, _)),
                Some(new_value),
            ) = (
                supported_refutation,
                repair_record,
                self.completed_line_value(repair_leaf, repair.len(), limits.cpu_depth),
            ) {
                if new_line != old_line
                    && new_value > old_value
                    && self.stopped(limits, cancel).is_none()
                {
                    let evidence = self.conclusion_observation(
                        root,
                        new_line,
                        ObservationKind::Repair,
                        new_value,
                        Some(old_evidence),
                    )?;
                    self.stores
                        .repair(self.nodes[root].situation, old_line, new_line, evidence)?;
                    counters.supported_repairs += 1;
                }
            }
        }
        Ok(())
    }
    fn value(&self, node: usize, remaining: usize) -> Option<i32> {
        let node = &self.nodes[node];
        if let Some((_, score)) = node.terminal {
            return Some(score);
        }
        if remaining > 0 {
            if let Some(value) = node
                .edges
                .iter()
                .filter_map(|edge| self.value(edge.child, remaining - 1).map(|score| -score))
                .max()
            {
                return Some(value);
            }
        }
        node.evidence.as_ref().map(|e| e.score)
    }
    fn completed_line_value(&self, leaf: usize, plies: usize, required_depth: u16) -> Option<i32> {
        let value = self.nodes[leaf]
            .terminal
            .map(|(_, value)| value)
            .or_else(|| {
                self.nodes[leaf]
                    .evidence
                    .as_ref()
                    .filter(|e| {
                        e.scope == CpuScoreScope::CompletedIteration && e.depth >= required_depth
                    })
                    .map(|e| e.score)
            })?;
        Some(if plies % 2 == 0 { value } else { -value })
    }
    fn conclusion_observation(
        &mut self,
        root: usize,
        line: LineId,
        kind: ObservationKind,
        value: i32,
        supersedes: Option<ObservationId>,
    ) -> Result<ObservationId, PalsError> {
        // Restricted-search summaries deliberately retain estimate scope. Even
        // a finite CPU mate score cannot mint RulesTerminal through this route.
        Ok(self.stores.append_observation(Observation {
            state: self.nodes[root].state,
            line: Some(line),
            source: stable_id(PALS_SEARCH_VERSION),
            epoch: 0,
            scope: EvidenceScope::Model {
                model: stable_id(self.model.identity()),
                encoding: stable_id("pals-restricted-summary-v1"),
                input: self.revision,
            },
            score: RawScore::Estimate {
                value: value as f32,
                perspective: self.nodes[root].position.side_to_move(),
            },
            budget: 0,
            kind,
            supersedes,
            execution: None,
        })?)
    }
    fn publish_choice(
        &self,
        root: usize,
        limits: PalsLimits,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<(), PalsError> {
        if let Some(stop) = self.stopped(limits, cancel) {
            return Err(match stop {
                PalsCompletion::Canceled => RoleError::Canceled,
                _ => RoleError::Deadline,
            }
            .into());
        }
        let mut best = None;
        let mut best_score = None;
        let mut best_scope = PalsValueScope::Unknown;
        for edge in &self.nodes[root].edges {
            if let Some(score) = self
                .value(edge.child, self.config.line_plies.saturating_sub(1))
                .map(|v| -v)
            {
                let scope = if self.nodes[edge.child].terminal.is_some() {
                    PalsValueScope::RulesTerminal
                } else {
                    PalsValueScope::RestrictedEstimate
                };
                if better_root_choice(score, scope, best_score, best_scope) {
                    best = Some(edge.movement);
                    best_score = Some(score);
                    best_scope = scope;
                }
            }
        }
        if let Some(movement) = best {
            progress(movement);
        }
        // Progress belongs to the UCI consumer and may itself spend time or
        // trigger stop. A final round must not report RoundLimit after that
        // callback crossed the deadline/cancellation boundary.
        if let Some(stop) = self.stopped(limits, cancel) {
            return Err(match stop {
                PalsCompletion::Canceled => RoleError::Canceled,
                _ => RoleError::Deadline,
            }
            .into());
        }
        Ok(())
    }
}

fn better_root_choice(
    value: i32,
    scope: PalsValueScope,
    previous: Option<i32>,
    previous_scope: PalsValueScope,
) -> bool {
    let proven_win = scope == PalsValueScope::RulesTerminal && value > 0;
    let previous_proven_win =
        previous_scope == PalsValueScope::RulesTerminal && previous.is_some_and(|value| value > 0);
    if proven_win != previous_proven_win {
        return proven_win;
    }
    previous.is_none_or(|old| {
        value > old
            || (value == old
                && scope == PalsValueScope::RulesTerminal
                && previous_scope != PalsValueScope::RulesTerminal)
    })
}

// Local provenance handles only. These never authorize exact NN cache identity;
// neural adapters carry their full canonical model/input digests separately.
fn stable_id(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;

    fn engine() -> PalsEngine<LegalOrderRoleMock> {
        PalsEngine::new(
            PalsConfig::default(),
            LegalOrderRoleMock,
            CpuEngine::new(CpuConfig {
                max_depth: 4,
                tt_entries: 128,
                quiescence_ply: 8,
                ..CpuConfig::default()
            })
            .unwrap(),
        )
        .unwrap()
    }
    fn limits() -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 1,
            max_cpu_nodes: 2000,
            cpu_depth: 1,
        }
    }
    #[test]
    fn mate_in_one_is_rules_checked_for_both_colors_and_never_a_nonmate() {
        for fen in [
            "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1",
            "8/8/8/8/8/6q1/5k2/7K b - - 0 1",
        ] {
            let position = Position::from_fen(fen).unwrap();
            let result = engine()
                .search(&position, limits(), &AtomicBool::new(false))
                .unwrap();
            let mut child = position.clone();
            child.make_move(result.best_move.unwrap()).unwrap();
            assert!(matches!(
                child.classify_position().unwrap().play_status,
                PlayStatus::Terminal {
                    reason: TerminalReason::Checkmate,
                    ..
                }
            ));
            assert_eq!(result.value_scope, PalsValueScope::RulesTerminal);
        }
    }
    #[test]
    fn stalemate_and_cancellation_preserve_explicit_terminal_or_unknown() {
        let position = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();
        let result = engine()
            .search(&position, limits(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result.terminal, Some(TerminalReason::Stalemate));
        assert_eq!(result.best_move, None);
        let position = Position::startpos();
        let result = engine()
            .search(&position, limits(), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(result.completion, PalsCompletion::Canceled);
        assert!(position.legal_moves().contains(&result.best_move.unwrap()));
        assert_eq!(result.score, None);
        assert_eq!(result.counters.cpu_tasks, 0);
    }
    #[test]
    fn counterexamples_and_repairs_keep_first_move_and_game_evidence() {
        let position = Position::startpos();
        let mut engine = engine();
        let result = engine
            .search(&position, limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(result.counters.proposals > 0);
        assert!(result.counters.refutations > 0);
        assert!(result.counters.repairs > 0);
        for record in engine.records() {
            let mut checked = position.clone();
            for &movement in &record.line {
                checked.make_move(movement).unwrap();
            }
        }
        let repair = engine
            .records()
            .iter()
            .find(|r| r.kind == RecordKind::Repair)
            .unwrap();
        let first = repair.line[0];
        assert!(result.root_values.iter().any(|v| v.movement == first));
        let retained = engine.retained_situations();
        let mut next = position.clone();
        next.make_move(first).unwrap();
        engine
            .search(&next, limits(), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(engine.retained_situations(), retained);
        engine.new_game();
        assert_eq!(engine.retained_situations(), 0);
        assert!(engine.records().is_empty());
    }
    #[test]
    fn unavailable_or_invalid_models_are_not_replaced_by_mock() {
        struct Unavailable;
        impl RoleModel for Unavailable {
            fn identity(&self) -> &str {
                "unavailable"
            }
            fn propose(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                Err(RoleError::Unavailable)
            }
            fn reply(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                Err(RoleError::Unavailable)
            }
            fn repair(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                Err(RoleError::Unavailable)
            }
            fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
                Err(RoleError::Unavailable)
            }
        }
        let mut engine = PalsEngine::new(
            PalsConfig::default(),
            Unavailable,
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            engine.search(&Position::startpos(), limits(), &AtomicBool::new(false)),
            Err(PalsError::Role(RoleError::Unavailable))
        ));
    }
    #[test]
    fn minimax_has_correct_alternating_perspective_and_unknown_is_not_draw() {
        let mut engine = engine();
        let root = engine.intern(Position::startpos()).unwrap();
        assert_eq!(engine.value(root, 4), None);
        let mut child = Position::startpos();
        let movement = BoardMove::from_uci("e2e4").unwrap();
        child.make_move(movement).unwrap();
        let child = engine
            .connect(root, movement, child, &mut PalsCounters::default())
            .unwrap();
        engine.nodes[child].evidence = Some(CpuEvidence {
            score: -25,
            depth: 1,
            scope: CpuScoreScope::CompletedIteration,
        });
        assert_eq!(engine.value(root, 4), Some(25));
    }
    #[test]
    fn incomplete_bootstrap_task_is_an_estimate_and_total_nodes_are_bounded() {
        let mut engine = PalsEngine::new(
            PalsConfig {
                cpu_nodes_per_task: 1,
                ..PalsConfig::default()
            },
            LegalOrderRoleMock,
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap();
        let result = engine
            .search(
                &Position::startpos(),
                PalsLimits {
                    max_cpu_nodes: 1,
                    cpu_depth: 2,
                    ..limits()
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(result.counters.cpu_nodes, 1);
        assert_eq!(result.counters.completed_cpu_tasks, 0);
        let cpu_records: Vec<_> = (0..engine.stores.observations.len())
            .filter_map(|index| {
                let record = engine
                    .stores
                    .observations
                    .get(ObservationId(index))
                    .unwrap();
                (record.kind == ObservationKind::CpuAnalysis).then_some(record)
            })
            .collect();
        assert!(!cpu_records.is_empty());
        assert!(
            cpu_records
                .iter()
                .all(|record| matches!(record.score, RawScore::Estimate { .. })
                    && matches!(record.scope, EvidenceScope::DepthLimited { depth: 0, .. }))
        );
        assert_eq!(result.counters.supported_refutations, 0);
        assert_eq!(result.counters.supported_repairs, 0);
    }
    #[test]
    fn completed_shallower_iteration_remains_partial_and_does_not_complete_the_requested_task() {
        let position = Position::startpos();
        let config = CpuConfig {
            max_depth: 4,
            tt_entries: 128,
            quiescence_ply: 8,
            ..CpuConfig::default()
        };
        // Obtain the actual work needed for depth one using a fresh identical
        // CPU owner. Give the depth-two request exactly that finite node budget.
        let depth_one = CpuEngine::new(config.clone())
            .unwrap()
            .analyze(
                &position,
                CpuLimits {
                    max_depth: 1,
                    max_nodes: 100_000,
                    deadline: None,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(depth_one.completed_depth, 1);
        assert_eq!(depth_one.score_scope, CpuScoreScope::CompletedIteration);
        let mut engine = PalsEngine::new(
            PalsConfig {
                cpu_nodes_per_task: depth_one.nodes,
                ..PalsConfig::default()
            },
            LegalOrderRoleMock,
            CpuEngine::new(config).unwrap(),
        )
        .unwrap();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        let mut counters = PalsCounters::default();
        engine
            .verify(
                root,
                &[],
                PalsLimits {
                    cpu_depth: 2,
                    max_cpu_nodes: depth_one.nodes,
                    ..limits()
                },
                &AtomicBool::new(false),
                &mut counters,
            )
            .unwrap();
        assert_eq!(counters.cpu_tasks, 1);
        assert_eq!(counters.completed_cpu_tasks, 0);
        assert_eq!(counters.partial_cpu_iterations, 1);
        assert_eq!(counters.consumed_cpu_tasks, 0);
        assert_eq!(counters.consumed_partial_cpu_values, 1);
        assert_eq!(counters.consumed_frontier_cpu_values, 0);
        let partial = engine.nodes[root].evidence.as_ref().unwrap();
        assert_eq!(partial.depth, 1);
        assert_eq!(partial.scope, CpuScoreScope::CompletedIteration);
        assert_eq!(engine.value(root, 0), Some(partial.score));
        assert_eq!(engine.completed_line_value(root, 0, 2), None);
        let task = engine
            .stores
            .tasks
            .get(super::super::store::ExecutionId(0))
            .unwrap();
        assert_eq!(task.key.requested_depth, 2);
        assert!(matches!(
            task.status,
            super::super::store::TaskStatus::Paused {
                evidence: Some(_),
                ..
            }
        ));
        let partial_record = engine.records().last().unwrap();
        assert_eq!(partial_record.completed_depth, 1);
        assert_eq!(
            partial_record.score_scope,
            Some(CpuScoreScope::CompletedIteration)
        );
    }
    #[test]
    fn role_budget_is_not_storage_failure_and_unexamined_reply_count_is_unknown() {
        let mut engine = PalsEngine::new(
            PalsConfig {
                max_role_calls: 1,
                ..PalsConfig::default()
            },
            LegalOrderRoleMock,
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap();
        let result = engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result.completion, PalsCompletion::RoleCallLimit);
        assert_eq!(result.counters.role_calls, 1);
        assert_eq!(result.counters.cpu_tasks, 0);
        assert!(
            result
                .root_values
                .iter()
                .any(|value| value.unexplored_replies.is_none())
        );
    }
    #[test]
    fn full_persistent_store_returns_safe_capacity_completion_for_a_new_root() {
        let mut engine = PalsEngine::new(
            PalsConfig {
                max_nodes: 257,
                ..PalsConfig::default()
            },
            LegalOrderRoleMock,
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap();
        for fullmove in 1..=257 {
            let position = Position::from_fen(&format!(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {fullmove}"
            ))
            .unwrap();
            engine.intern(position).unwrap();
        }
        let position = Position::startpos();
        let result = engine
            .search(&position, limits(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result.completion, PalsCompletion::Capacity);
        assert!(position.legal_moves().contains(&result.best_move.unwrap()));
        assert_eq!(result.value_scope, PalsValueScope::Unknown);
        assert_eq!(engine.retained_situations(), 257);
        // A capacity fallback still classifies Rules-terminal positions. Legal
        // geometric moves in an automatic draw are not unknown game branches.
        let terminal = Position::from_fen("8/8/8/8/8/8/6k1/K7 w - - 0 1").unwrap();
        assert!(!terminal.legal_moves().is_empty());
        let result = engine
            .search(&terminal, limits(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result.completion, PalsCompletion::Terminal);
        assert!(result.terminal.is_some());
        assert!(result.root_values.is_empty());
        assert_eq!(result.counters.unknown_root_children, 0);
        assert!(result.counters.root_scope_observation_complete);
    }
    #[test]
    fn widening_admits_low_prior_roots_and_derived_eviction_keeps_completed_work() {
        let mut engine = engine();
        let position = Position::startpos();
        let result = engine
            .search(
                &position,
                PalsLimits {
                    max_rounds: 3,
                    max_cpu_nodes: 50_000,
                    ..limits()
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert!(
            result
                .root_values
                .iter()
                .filter(|value| value.unexplored_replies.is_some())
                .count()
                >= 6
        );
        let observations = engine.stores.observations.len();
        let executions = engine.stores.tasks.len();
        engine.evict_derived_cache();
        assert_eq!(engine.stores.observations.len(), observations);
        assert_eq!(engine.stores.tasks.len(), executions);
        let repeated = engine
            .search(&position, limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(repeated.counters.evidence_cache_hits > 0);
    }
    #[test]
    fn last_valid_choice_is_published_before_a_late_role_result_is_rejected() {
        struct CancelingCritic;
        impl RoleModel for CancelingCritic {
            fn identity(&self) -> &str {
                "canceling-critic-test"
            }
            fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                LegalOrderRoleMock::evaluate(query)
            }
            fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                let result = RoleEvaluation {
                    logits: (0..query.legal.len()).map(|i| -(i as f32)).collect(),
                    wdl: [0.25, 0.5, 0.25],
                };
                query.cancel.store(true, Ordering::Release);
                Ok(result)
            }
            fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                LegalOrderRoleMock::evaluate(query)
            }
            fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
                Ok(vec![0.0; query.candidates.len()])
            }
        }
        let position = Position::startpos();
        let mut engine = PalsEngine::new(
            PalsConfig::default(),
            CancelingCritic,
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap();
        let mut published = Vec::new();
        let result = engine
            .search_with_progress(&position, limits(), &AtomicBool::new(false), |movement| {
                published.push(movement)
            })
            .unwrap();
        assert_eq!(result.completion, PalsCompletion::Canceled);
        assert!(!published.is_empty());
        assert!(
            published
                .iter()
                .all(|movement| position.legal_moves().contains(movement))
        );
        assert!(result.counters.role_calls > result.counters.consumed_role_outputs);
        assert_eq!(result.counters.refutations, 0);
        assert_eq!(result.counters.repairs, 0);
    }
    #[test]
    fn actual_rules_win_outranks_arbitrarily_large_finite_cpu_estimates() {
        assert!(better_root_choice(
            CPU_MATE_SCORE,
            PalsValueScope::RulesTerminal,
            Some(100_000),
            PalsValueScope::RestrictedEstimate
        ));
        assert!(!better_root_choice(
            100_000,
            PalsValueScope::RestrictedEstimate,
            Some(CPU_MATE_SCORE),
            PalsValueScope::RulesTerminal
        ));
    }

    #[derive(Clone, Copy)]
    enum AcceptanceBehavior {
        Normal,
        Malformed,
        LateCancel,
        LateDivergence,
        KnownDeadline,
        PhysicalUnknown,
    }

    struct AcceptanceProbe {
        accepted: std::sync::Arc<std::sync::atomic::AtomicU64>,
        returned: std::sync::Arc<std::sync::atomic::AtomicU64>,
        behavior: AcceptanceBehavior,
        bad_divergences: bool,
    }
    impl RoleModel for AcceptanceProbe {
        fn identity(&self) -> &str {
            "acceptance-probe-test"
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let mut output = LegalOrderRoleMock::evaluate(RoleQuery {
                position: query.position,
                legal: query.legal,
                prefix: query.prefix,
                proposal: query.proposal,
                counterexample: query.counterexample,
                records: query.records,
                revision: query.revision,
                deadline: query.deadline,
                cancel: query.cancel,
            })?;
            match self.behavior {
                AcceptanceBehavior::Normal | AcceptanceBehavior::LateDivergence => {}
                AcceptanceBehavior::Malformed => {
                    output.logits.pop();
                }
                AcceptanceBehavior::LateCancel => query.cancel.store(true, Ordering::Release),
                AcceptanceBehavior::KnownDeadline => return Err(RoleError::Deadline),
                AcceptanceBehavior::PhysicalUnknown => {
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
            }
            self.returned.fetch_add(1, Ordering::SeqCst);
            Ok(output)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.propose(query)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.propose(query)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            self.returned.fetch_add(1, Ordering::SeqCst);
            if matches!(self.behavior, AcceptanceBehavior::LateDivergence) {
                query.cancel.store(true, Ordering::Release);
            }
            Ok(if self.bad_divergences {
                Vec::new()
            } else {
                vec![0.0; query.candidates.len()]
            })
        }
        fn accepted_output(&mut self) {
            self.accepted.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn acceptance_probe(
        behavior: AcceptanceBehavior,
        bad_divergences: bool,
    ) -> (
        PalsEngine<Box<dyn RoleModel>>,
        std::sync::Arc<std::sync::atomic::AtomicU64>,
        std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) {
        let accepted = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let returned = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let model: Box<dyn RoleModel> = Box::new(AcceptanceProbe {
            accepted: accepted.clone(),
            returned: returned.clone(),
            behavior,
            bad_divergences,
        });
        (
            PalsEngine::new(
                PalsConfig::default(),
                model,
                CpuEngine::new(CpuConfig::default()).unwrap(),
            )
            .unwrap(),
            accepted,
            returned,
        )
    }

    #[test]
    fn accepted_hook_counts_actual_consumption_including_boxed_divergences() {
        let (mut engine, accepted, returned) = acceptance_probe(AcceptanceBehavior::Normal, false);
        let result = engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(result.counters.refutations > 0);
        assert_eq!(
            accepted.load(Ordering::SeqCst),
            result.counters.consumed_role_outputs
        );
        assert_eq!(
            accepted.load(Ordering::SeqCst),
            returned.load(Ordering::SeqCst)
        );
        assert!(accepted.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn malformed_and_late_outputs_do_not_acknowledge_consumption() {
        for behavior in [
            AcceptanceBehavior::Malformed,
            AcceptanceBehavior::LateCancel,
        ] {
            let (mut engine, accepted, returned) = acceptance_probe(behavior, false);
            let result = engine.search(&Position::startpos(), limits(), &AtomicBool::new(false));
            match behavior {
                AcceptanceBehavior::Malformed => assert!(matches!(
                    result,
                    Err(PalsError::Role(RoleError::InvalidOutput))
                )),
                AcceptanceBehavior::LateCancel => {
                    assert_eq!(result.unwrap().completion, PalsCompletion::Canceled)
                }
                _ => unreachable!(),
            }
            assert_eq!(returned.load(Ordering::SeqCst), 1);
            assert_eq!(accepted.load(Ordering::SeqCst), 0);
        }
        let (mut engine, accepted, returned) = acceptance_probe(AcceptanceBehavior::Normal, true);
        assert!(matches!(
            engine.search(&Position::startpos(), limits(), &AtomicBool::new(false)),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        assert_eq!(
            returned.load(Ordering::SeqCst),
            accepted.load(Ordering::SeqCst) + 1
        );
        let (mut engine, accepted, returned) =
            acceptance_probe(AcceptanceBehavior::LateDivergence, false);
        let result = engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result.completion, PalsCompletion::Canceled);
        assert_eq!(
            returned.load(Ordering::SeqCst),
            accepted.load(Ordering::SeqCst) + 1
        );
        assert_eq!(
            result.counters.consumed_role_outputs,
            accepted.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn known_deadline_and_unknown_physical_completion_have_different_outcomes_and_no_consumption() {
        for behavior in [
            AcceptanceBehavior::KnownDeadline,
            AcceptanceBehavior::PhysicalUnknown,
        ] {
            let (mut engine, accepted, _) = acceptance_probe(behavior, false);
            let result = engine.search(&Position::startpos(), limits(), &AtomicBool::new(false));
            match behavior {
                AcceptanceBehavior::KnownDeadline => {
                    assert_eq!(result.unwrap().completion, PalsCompletion::Deadline)
                }
                AcceptanceBehavior::PhysicalUnknown => assert!(matches!(
                    result,
                    Err(PalsError::Role(RoleError::PhysicalCompletionUnknown))
                )),
                _ => unreachable!(),
            }
            assert_eq!(accepted.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn task_reuse_consumption_is_distinct_from_existing_node_evidence_reuse() {
        let position = Position::startpos();
        let mut engine = engine();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        let mut initial = PalsCounters::default();
        engine
            .verify(root, &[], limits(), &AtomicBool::new(false), &mut initial)
            .unwrap();
        assert_eq!(initial.cpu_tasks_requested, 1);
        assert_eq!(initial.completed_cpu_tasks, 1);
        assert_eq!(initial.consumed_cpu_tasks, 1);
        assert_eq!(initial.reused_completed_cpu_tasks_consumed, 0);
        let mut node_reuse = PalsCounters::default();
        engine
            .verify(
                root,
                &[],
                limits(),
                &AtomicBool::new(false),
                &mut node_reuse,
            )
            .unwrap();
        assert_eq!(node_reuse.evidence_cache_hits, 1);
        assert_eq!(node_reuse.consumed_cached_cpu_values, 1);
        assert_eq!(node_reuse.reused_completed_cpu_tasks_consumed, 0);
        assert_eq!(node_reuse.consumed_cpu_tasks, 0);
        // Evict only the derived node projection, retaining the immutable
        // completed task and observation. A new task consumer can now reuse it.
        engine.nodes[root].evidence = None;
        let mut task_reuse = PalsCounters::default();
        engine
            .verify(
                root,
                &[],
                limits(),
                &AtomicBool::new(false),
                &mut task_reuse,
            )
            .unwrap();
        assert_eq!(task_reuse.cpu_tasks_requested, 0);
        assert_eq!(task_reuse.completed_cpu_tasks, 0);
        assert_eq!(task_reuse.consumed_cpu_tasks, 1);
        assert_eq!(task_reuse.reused_completed_cpu_tasks_consumed, 1);
        assert_eq!(task_reuse.consumed_cached_cpu_values, 1);
    }

    #[test]
    fn completed_cpu_report_survives_failed_bounded_evidence_publication_without_consumption() {
        let position = Position::startpos();
        let mut engine = engine();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        // A zero-capacity evidence store simulates exhausted retained evidence;
        // CPU work must still be observed and its in-flight task must be closed.
        engine.stores.observations = super::super::store::ObservationStore::new(0);
        let mut counters = PalsCounters::default();
        assert!(matches!(
            engine.verify(root, &[], limits(), &AtomicBool::new(false), &mut counters),
            Err(PalsError::Capacity)
        ));
        assert_eq!(counters.cpu_tasks_requested, 1);
        assert_eq!(counters.cpu_tasks, 1);
        assert!(counters.cpu_nodes > 0);
        assert_eq!(counters.completed_cpu_tasks, 1);
        assert_eq!(counters.consumed_cpu_tasks, 0);
        assert!(!counters.cpu_work_observation_incomplete);
        assert!(matches!(
            engine
                .stores
                .tasks
                .get(super::super::store::ExecutionId(0))
                .unwrap()
                .status,
            super::super::store::TaskStatus::Failed
        ));
        assert!(engine.nodes[root].evidence.is_none());
        assert!(engine.nodes[root].resume.is_none());
    }

    #[test]
    fn failed_cpu_execution_keeps_requested_work_snapshot_and_marks_missing_report() {
        use crate::cpu_value::{
            BootstrapCpuValue, CpuAccumulator, CpuAccumulatorUndo, CpuValueError,
            CpuValueEvaluator, CpuValueIdentity,
        };
        struct FailingCpuValue {
            base: BootstrapCpuValue,
            scores: std::sync::Arc<std::sync::atomic::AtomicU64>,
        }
        impl CpuValueEvaluator for FailingCpuValue {
            fn identity(&self) -> &CpuValueIdentity {
                self.base.identity()
            }
            fn provenance(&self) -> &'static str {
                "failing-after-root-value-test"
            }
            fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
                self.base.initialize(position)
            }
            fn score(
                &self,
                accumulator: &CpuAccumulator,
                position: &Position,
            ) -> Result<i32, CpuValueError> {
                let value = self.base.score(accumulator, position)?;
                if self.scores.fetch_add(1, Ordering::SeqCst) > 0 {
                    Err(CpuValueError::NonFiniteForward)
                } else {
                    Ok(value)
                }
            }
            fn apply_delta(
                &self,
                accumulator: &mut CpuAccumulator,
                delta: &rz_position::RuleMoveDelta,
            ) -> Result<CpuAccumulatorUndo, CpuValueError> {
                self.base.apply_delta(accumulator, delta)
            }
            fn restore(
                &self,
                accumulator: &mut CpuAccumulator,
                undo: CpuAccumulatorUndo,
                position: &Position,
            ) -> Result<(), CpuValueError> {
                self.base.restore(accumulator, undo, position)
            }
        }
        let scores = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let cpu = CpuEngine::with_evaluator(
            CpuConfig::default(),
            std::sync::Arc::new(FailingCpuValue {
                base: BootstrapCpuValue::default(),
                scores: scores.clone(),
            }),
        )
        .unwrap();
        let mut engine = PalsEngine::new(PalsConfig::default(), LegalOrderRoleMock, cpu).unwrap();
        assert!(matches!(
            engine.search(&Position::startpos(), limits(), &AtomicBool::new(false)),
            Err(PalsError::Cpu(CpuError::Value(
                CpuValueError::NonFiniteForward
            )))
        ));
        let snapshot = engine.last_search_counters().unwrap();
        assert_eq!(snapshot.cpu_tasks_requested, 1);
        assert_eq!(snapshot.cpu_tasks, 0);
        assert_eq!(snapshot.completed_cpu_tasks, 0);
        assert_eq!(snapshot.consumed_cpu_tasks, 0);
        assert!(snapshot.cpu_work_observation_incomplete);
        assert!(!snapshot.root_scope_observation_complete);
        assert!(scores.load(Ordering::SeqCst) >= 2);
        assert!(snapshot.proposer_calls > 0);
        assert_eq!(
            snapshot.accepted_proposer_outputs,
            snapshot.completed_proposer_calls
        );
        assert_eq!(
            snapshot.role_calls,
            snapshot.proposer_calls + snapshot.critic_calls + snapshot.repair_calls
        );
        engine.new_game();
        assert_eq!(engine.last_search_counters(), None);
    }

    #[test]
    fn role_error_and_late_valid_output_keep_distinct_dispatch_completion_consumption_snapshots() {
        for behavior in [
            AcceptanceBehavior::PhysicalUnknown,
            AcceptanceBehavior::LateCancel,
        ] {
            let (mut engine, accepted, _) = acceptance_probe(behavior, false);
            let result = engine.search(&Position::startpos(), limits(), &AtomicBool::new(false));
            let snapshot = engine.last_search_counters().unwrap();
            assert_eq!(snapshot.role_calls, 1);
            assert_eq!(snapshot.proposer_calls, 1);
            assert_eq!(snapshot.accepted_proposer_outputs, 0);
            assert_eq!(snapshot.consumed_role_outputs, 0);
            assert_eq!(accepted.load(Ordering::SeqCst), 0);
            match behavior {
                AcceptanceBehavior::PhysicalUnknown => {
                    assert!(matches!(
                        result,
                        Err(PalsError::Role(RoleError::PhysicalCompletionUnknown))
                    ));
                    assert_eq!(snapshot.completed_proposer_calls, 0);
                }
                AcceptanceBehavior::LateCancel => {
                    assert_eq!(result.unwrap().completion, PalsCompletion::Canceled);
                    assert_eq!(snapshot.completed_proposer_calls, 1);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn progress_callback_crossing_deadline_rechecks_control_without_new_role_dispatch() {
        let (mut engine, accepted, returned) = acceptance_probe(AcceptanceBehavior::Normal, false);
        let deadline = Instant::now() + Duration::from_millis(500);
        let mut progress_calls = 0;
        let result = engine
            .search_with_progress(
                &Position::startpos(),
                PalsLimits {
                    deadline,
                    ..limits()
                },
                &AtomicBool::new(false),
                |_| {
                    progress_calls += 1;
                    std::thread::sleep(
                        deadline.saturating_duration_since(Instant::now())
                            + Duration::from_millis(2),
                    );
                },
            )
            .unwrap();
        assert_eq!(progress_calls, 1);
        assert_eq!(result.completion, PalsCompletion::Deadline);
        assert_eq!(result.counters.refutations, 0);
        assert_eq!(result.counters.repairs, 0);
        assert_eq!(
            accepted.load(Ordering::SeqCst),
            returned.load(Ordering::SeqCst)
        );
        assert_eq!(
            result.counters.consumed_role_outputs,
            accepted.load(Ordering::SeqCst)
        );
    }
}
