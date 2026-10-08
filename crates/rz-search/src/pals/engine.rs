//! Restricted-game PALS refinement. These edges are examined continuations,
//! not PUCT visits. Only Rules can certify a terminal position.
use super::store::{
    BoundKind, EvidenceScope, LineId, Move16, Observation, ObservationId, ObservationKind,
    PalsStores, RawScore, SituationId, StateId, StoreError, StoreLimits, TaskAdmission,
    TaskConsumer, TaskKey, TaskQuestion,
};
use super::value::MODEL_WDL_RESOLVER_VERSION;
pub use super::value::{
    MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity, ModelValueOutput, PalsResolvedValue,
};
use crate::cpu::{
    CPU_MATE_SCORE, CpuEngine, CpuError, CpuLimits, CpuReport, CpuResumeToken, CpuScoreScope,
    CpuSearcher,
};
use crate::cpu_checker::{
    CheckerAttempt, CheckerError, CheckerIdentity, CheckerReport, CheckerShutdown, CpuChecker,
    ExternalCheckerReport, ExternalCompletion, OwnedCheckerDescriptor, OwnedCpuChecker,
};
use crate::cpu_value::CpuValueIdentity;
use rz_position::{
    BoardMove, Color, PlayStatus, Position, PositionError, PositionSnapshot, TerminalReason,
};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const PALS_SEARCH_VERSION: &str = "pals-restricted-refinement/0.1";

/// Opt-in search semantics; this does not change a model or the value resolver.
pub const POST_REPAIR_RECHECK_SEARCH_VERSION: &str =
    "pals-restricted-refinement-post-repair-recheck/1";
pub const POST_REPAIR_RECHECK_CONDITIONS: &str = "accepted-repair-and-completed-own-evidence;unchanged-first-move;first-opponent-anchor-after-changed-own-response;one-Reply-call-per-repair;prefer-unexamined-different-legal-response;remaining-repaired-suffix-Rules-replay;equal-line-length;equal-completed-CPU-depth-and-namespace-or-both-Rules-terminals;mixed-scope-unresolved;conditional-refutation-only;shared-global-role-cpu-store-deadline-cancel-limits;no-recursive-repair";

/// Selected only at construction. Driver/CLI/manifest registration is a separate
/// integration step; exposing this API does not advertise product CLI support.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PostRepairRecheckPolicy {
    #[default]
    Disabled,
    SameRepairedLineOnceV1,
}
impl PostRepairRecheckPolicy {
    pub fn search_identity(self) -> &'static str {
        match self {
            Self::Disabled => PALS_SEARCH_VERSION,
            Self::SameRepairedLineOnceV1 => POST_REPAIR_RECHECK_SEARCH_VERSION,
        }
    }
    pub fn conditions(self) -> Option<&'static str> {
        match self {
            Self::Disabled => None,
            Self::SameRepairedLineOnceV1 => Some(POST_REPAIR_RECHECK_CONDITIONS),
        }
    }
}

/// Identity of the current value resolver, independent of model, CPU value
/// namespace and search implementation identities. Changing this policy needs
/// a new version; recording it does not calibrate CPU scores or change search.
///
/// Rules terminals are exact current-state facts. Other leaves use accepted
/// side-to-move CPU raw scores from the registered value namespace, including
/// frontier-only and partially completed iterations; their original scope and
/// work counters remain recorded. Examined children propagate max/negation.
/// Missing values stay None and are omitted from that restricted max, never
/// replaced by zero. No P/C WDL averaging or CP/WDL calibration is performed.
///
/// At the root, a Rules-certified win takes priority; equal values prefer a
/// Rules terminal, then existing Rules order resolves ties. A restricted
/// estimate, including a finite-depth mate score, is not a whole-game bound or
/// an independently verified all-defenses proof.
pub const PALS_VALUE_RESOLVER_VERSION: &str = "pals-cpu-raw-restricted/0.1";
pub const PALS_VALUE_RESOLVER_SEMANTICS: &str = "leaf:accepted-registered-cpu-raw-side-to-move-including-frontier-and-partial;propagation:examined-children-max-negation;unknown:None-not-zero;calibration:none;pc-score-average:none;root:rules-certified-win-first,equal-value-terminal-first,Rules-order-ties;nonterminal-scope:restricted-estimate-not-game-bound";

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
    /// CPU-created candidate projection refers to its actual divergence-state
    /// observation, rather than inventing neural generation or a leaf score.
    pub cpu_observation: Option<ObservationId>,
    pub perspective: Color,
    pub critical: bool,
}

/// Search-owned logical question, independent of a native physical execution ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RoleQueryPurpose {
    ProposePolicy,
    ReplyPolicy,
    RepairPolicy,
    DivergencePolicy,
    /// Frontier WDL always uses a fresh evaluation, never a policy warm seed.
    ValueFresh,
}

/// Current checked store handles and complete ordered continuation identity.
/// Numeric handles do not replace exact Rules/history or a prepared model input.
/// Record revisions remain acceptance metadata; they are not warm seed keys.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoleLogicalContext {
    pub game_generation: u64,
    pub search_generation: u64,
    pub situation: SituationId,
    pub state: StateId,
    pub focus: LineId,
    pub purpose: RoleQueryPurpose,
    pub prefix: Vec<BoardMove>,
    pub focus_sha256: [u8; 32],
    pub prefix_sha256: [u8; 32],
    pub proposal_sha256: [u8; 32],
    pub refutation_sha256: Option<[u8; 32]>,
    pub divergence_sha256: [u8; 32],
    pub public_revision: u64,
    pub situation_revision: u64,
}
impl RoleLogicalContext {
    /// Structural digest only. Current records/revisions must still be encoded
    /// in the actual input and checked at acceptance; this grants no cache hit.
    pub fn focus_prefix_sha256(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"rz-pals-role-focus-prefix/1\0");
        digest.update(self.focus_sha256);
        digest.update(self.prefix_sha256);
        digest.finalize().into()
    }
}

/// Final search acknowledgement with the original controls and current Rules
/// snapshot. A provider may commit provisional state only after checking this
/// against its delivered request; this hook grants no physical completion.
pub struct RoleAcceptance<'a> {
    pub snapshot: &'a PositionSnapshot,
    pub context: &'a RoleLogicalContext,
    pub deadline: Instant,
    pub cancel: &'a AtomicBool,
}
impl RoleAcceptance<'_> {
    pub fn check_control(&self) -> Result<(), RoleError> {
        check_role_control(self.deadline, self.cancel)
    }
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
        check_role_control(self.deadline, self.cancel)
    }
}
fn check_role_control(deadline: Instant, cancel: &AtomicBool) -> Result<(), RoleError> {
    if cancel.load(Ordering::Acquire) {
        return Err(RoleError::Canceled);
    }
    if Instant::now() >= deadline {
        return Err(RoleError::Deadline);
    }
    Ok(())
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoleSearchClosure {
    Completed,
    Canceled,
    Deadline,
    Failed,
    PhysicalCompletionUnknown,
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
    /// Separate frontier WDL namespace. Own CPU mode does not require this API.
    /// The provider returns the key from its actual prepared input, rather than
    /// a board-only/public-memory key or a digest reconstructed after inference.
    fn value_identity(&self) -> Option<&ModelValueIdentity> {
        None
    }
    /// Uses the Proposer forward's shared WDL; candidate policy is not consumed.
    /// This estimate cannot calibrate foreign CP/mate or certify Rules facts.
    fn evaluate_value(&mut self, _query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
        Err(RoleError::Unavailable)
    }
    fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError>;
    fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError>;
    fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError>;
    fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError>;
    /// Additive logical context keeps existing callers and Fresh mocks intact.
    /// Opt-in providers must also seal the exact prepared input and query bits.
    fn propose_with_context(
        &mut self,
        query: RoleQuery<'_>,
        _context: &RoleLogicalContext,
    ) -> Result<RoleEvaluation, RoleError> {
        self.propose(query)
    }
    fn reply_with_context(
        &mut self,
        query: RoleQuery<'_>,
        _context: &RoleLogicalContext,
    ) -> Result<RoleEvaluation, RoleError> {
        self.reply(query)
    }
    fn repair_with_context(
        &mut self,
        query: RoleQuery<'_>,
        _context: &RoleLogicalContext,
    ) -> Result<RoleEvaluation, RoleError> {
        self.repair(query)
    }
    fn divergences_with_context(
        &mut self,
        query: DivergenceQuery<'_>,
        _context: &RoleLogicalContext,
    ) -> Result<Vec<f32>, RoleError> {
        self.divergences(query)
    }
    fn evaluate_value_with_context(
        &mut self,
        query: RoleQuery<'_>,
        _context: &RoleLogicalContext,
    ) -> Result<ModelValueOutput, RoleError> {
        self.evaluate_value(query)
    }
    /// Accounting-only acknowledgement for the most recent output. Returning
    /// from inference is not consumption: the search calls this exactly once
    /// after output validation and its final deadline/cancellation acceptance.
    /// This hook must not dispatch work or perform a search.
    fn accepted_output(&mut self) {}
    /// Fallible acknowledgement. Search consumption is recorded only on success.
    /// A warm provider checks its pending context/snapshot and original controls,
    /// relays cancellation to its admitted token, and commits at most once.
    fn accepted_output_checked(&mut self, acceptance: RoleAcceptance<'_>) -> Result<(), RoleError> {
        acceptance.check_control()?;
        self.accepted_output();
        Ok(())
    }
    /// Close only delivered-output accounting at this logical search boundary.
    /// Physical leases, drain, and quarantine remain the backend owner's job.
    fn finish_search(&mut self, _reason: RoleSearchClosure) {}
    fn new_game(&mut self) {}
    /// None means logical generation exhaustion; no later question is admitted.
    /// This reset does not assert native drain or a worker reset acknowledgement.
    fn new_game_with_generation(&mut self, _game_generation: Option<u64>) {
        self.new_game();
    }
}

impl<M: RoleModel + ?Sized> RoleModel for Box<M> {
    fn identity(&self) -> &str {
        (**self).identity()
    }
    fn value_identity(&self) -> Option<&ModelValueIdentity> {
        (**self).value_identity()
    }
    fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
        (**self).evaluate_value(query)
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
    fn propose_with_context(
        &mut self,
        query: RoleQuery<'_>,
        context: &RoleLogicalContext,
    ) -> Result<RoleEvaluation, RoleError> {
        (**self).propose_with_context(query, context)
    }
    fn reply_with_context(
        &mut self,
        query: RoleQuery<'_>,
        context: &RoleLogicalContext,
    ) -> Result<RoleEvaluation, RoleError> {
        (**self).reply_with_context(query, context)
    }
    fn repair_with_context(
        &mut self,
        query: RoleQuery<'_>,
        context: &RoleLogicalContext,
    ) -> Result<RoleEvaluation, RoleError> {
        (**self).repair_with_context(query, context)
    }
    fn divergences_with_context(
        &mut self,
        query: DivergenceQuery<'_>,
        context: &RoleLogicalContext,
    ) -> Result<Vec<f32>, RoleError> {
        (**self).divergences_with_context(query, context)
    }
    fn evaluate_value_with_context(
        &mut self,
        query: RoleQuery<'_>,
        context: &RoleLogicalContext,
    ) -> Result<ModelValueOutput, RoleError> {
        (**self).evaluate_value_with_context(query, context)
    }
    fn accepted_output(&mut self) {
        (**self).accepted_output();
    }
    fn accepted_output_checked(&mut self, acceptance: RoleAcceptance<'_>) -> Result<(), RoleError> {
        (**self).accepted_output_checked(acceptance)
    }
    fn finish_search(&mut self, reason: RoleSearchClosure) {
        (**self).finish_search(reason);
    }
    fn new_game(&mut self) {
        (**self).new_game();
    }
    fn new_game_with_generation(&mut self, game_generation: Option<u64>) {
        (**self).new_game_with_generation(game_generation);
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
    fn value_identity(&self) -> Option<&ModelValueIdentity> {
        static IDENTITY: std::sync::OnceLock<ModelValueIdentity> = std::sync::OnceLock::new();
        Some(IDENTITY.get_or_init(|| ModelValueIdentity {
            semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
            model: "explicit-legal-order-role-mock-v1".into(),
            encoding: "explicit-mock-full-history-role-query-v1".into(),
            precision: "fp32".into(),
            model_epoch: [0; 32],
        }))
    }
    fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
        query.check_control()?;
        // This explicitly selected fixture defines its own prepared input. It
        // is neither an LC0 tensor key nor a learned/native model evaluation.
        let mut prepared = Sha256::new();
        prepared.update(b"explicit-mock-full-history-role-query-v1\0");
        for fen in query.position.snapshot().known_history_fens() {
            prepared.update((fen.len() as u64).to_le_bytes());
            prepared.update(fen.as_bytes());
        }
        prepared.update(
            format!(
                "{:?};{:?};{:?};{:?};{:?};{};{:?}",
                query.legal,
                query.prefix,
                query.proposal,
                query.counterexample,
                query.records,
                query.revision,
                query.deadline
            )
            .as_bytes(),
        );
        let output = ModelValueOutput {
            identity: self.value_identity().ok_or(RoleError::Unavailable)?.clone(),
            input_sha256: prepared.finalize().into(),
            state: query.position.position_identity(),
            perspective: query.position.side_to_move(),
            wdl: [0.25, 0.5, 0.25],
        };
        query.check_control()?;
        Ok(output)
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
    /// Shared Proposer WDL calls are separate from proposal policy calls.
    pub value_calls: u64,
    pub completed_value_calls: u64,
    pub accepted_value_outputs: u64,
    /// Foreign work is reported separately: None/omission never means zero.
    pub external_checker_tasks: u64,
    pub external_checker_reports: u64,
    pub external_checker_nodes_observed: u64,
    pub external_checker_work_incomplete: bool,
    /// Finite admission budget charged once per external request, distinct from
    /// observed nodes; nodes limits in UCI do not guarantee physical hard caps.
    pub external_checker_node_budget_reserved: u64,
    pub consumed_external_checker_tasks: u64,
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
    pub resolved_value: PalsResolvedValue,
    pub scope: PalsValueScope,
    pub examined_replies: usize,
    /// None means the child was not materialized/examined, not zero replies.
    pub unexplored_replies: Option<usize>,
}
#[derive(Clone, Debug)]
pub struct PalsResult {
    pub best_move: Option<BoardMove>,
    pub score: Option<i32>,
    /// Model WDL stays typed and never uses the own raw integer score field.
    pub resolved_value: PalsResolvedValue,
    pub model_value_identity: Option<ModelValueIdentity>,
    pub checker_identity: CheckerIdentity,
    pub value_scope: PalsValueScope,
    pub terminal: Option<TerminalReason>,
    pub completion: PalsCompletion,
    pub counters: PalsCounters,
    pub root_values: Vec<PalsRootValue>,
    pub elapsed: Duration,
    pub model_identity: String,
    /// Policy identity only; the underlying CPU value namespace remains
    /// independent and is preserved in task/observation provenance.
    pub resolver_version: &'static str,
}
#[derive(Debug)]
pub enum PalsError {
    InvalidConfig,
    InvalidLimits,
    Rules(PositionError),
    Cpu(CpuError),
    Checker(CheckerError),
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
impl From<CheckerError> for PalsError {
    fn from(error: CheckerError) -> Self {
        match error {
            CheckerError::Owned(error) => Self::Cpu(error),
            CheckerError::OwnedReportRejected { reason } => {
                Self::Store(StoreError::InvalidEvidence(reason))
            }
            error => Self::Checker(error),
        }
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
    model_value: Option<ModelValueOutput>,
    resume: Option<CpuResumeToken>,
}
#[derive(Clone, Debug)]
struct CpuEvidence {
    score: i32,
    depth: u16,
    scope: CpuScoreScope,
    value_identity: CpuValueIdentity,
}
struct CpuCandidate {
    pv: Vec<BoardMove>,
    observation: ObservationId,
    generation: u64,
    root: SituationId,
    root_revision: u64,
    deadline: Instant,
}
#[derive(Clone, Copy)]
enum Call {
    Propose,
    Reply,
    Repair,
}
#[derive(Clone, Copy)]
struct RoleQuestion<'a> {
    purpose: RoleQueryPurpose,
    prefix: &'a [BoardMove],
    proposal: &'a [BoardMove],
    refutation: Option<&'a [BoardMove]>,
    divergences: &'a [usize],
}

/// One already accepted Repair. All slices remain owned by the current refine;
/// this is neither a persistent queue nor a grant to reopen another root.
struct CompletedRepairRecheck<'a> {
    root: usize,
    original_first: BoardMove,
    attack_ply: usize,
    repaired_line: LineId,
    repair_record_revision: u64,
    repaired_leaf: usize,
    repaired: &'a [BoardMove],
    refutation: &'a [BoardMove],
    completed_value: i32,
}

/// Persistent exact situations and public evidence survive a normal root change.
/// `new_game` is the only automatic game-wide invalidation boundary.
pub struct PalsEngine<M: RoleModel> {
    config: PalsConfig,
    post_repair_recheck: PostRepairRecheckPolicy,
    model: M,
    cpu: Box<dyn CpuChecker>,
    checker_registered_identity: CheckerIdentity,
    cpu_registered_value: Option<CpuValueIdentity>,
    model_registered_value: Option<ModelValueIdentity>,
    cpu_registered_condition: String,
    checker_new_game_pending: bool,
    last_external_attempt: Option<u64>,
    external_attempts: Vec<CheckerAttempt>,
    nodes: Vec<Node>,
    records: Vec<RoleRecord>,
    revision: u64,
    // Store handles may restart from zero after reset; this checked namespace
    // prevents old-game logical questions from acquiring the new authority.
    game_generation: Option<u64>,
    stores: PalsStores,
    clock_origin: Instant,
    consumer_id: u64,
    last_search_counters: Option<PalsCounters>,
}
impl<M: RoleModel> PalsEngine<M> {
    pub fn new(config: PalsConfig, model: M, cpu: CpuEngine) -> Result<Self, PalsError> {
        Self::new_with_cpu(config, model, cpu)
    }
    pub fn new_with_cpu<C: CpuSearcher + 'static>(
        config: PalsConfig,
        model: M,
        cpu: C,
    ) -> Result<Self, PalsError> {
        Self::new_with_boxed_cpu(config, model, Box::new(cpu))
    }
    /// Immutable opt-in lane. Existing constructors keep Disabled semantics.
    pub fn new_with_cpu_and_refinement_policy<C: CpuSearcher + 'static>(
        config: PalsConfig,
        model: M,
        cpu: C,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, PalsError> {
        let cpu: Box<dyn CpuSearcher> = Box::new(cpu);
        Self::new_with_boxed_checker_and_refinement_policy(
            config,
            model,
            Box::new(OwnedCpuChecker::new(cpu)?),
            policy,
        )
    }
    pub fn new_with_boxed_cpu(
        config: PalsConfig,
        model: M,
        cpu: Box<dyn CpuSearcher>,
    ) -> Result<Self, PalsError> {
        Self::new_with_boxed_checker(config, model, Box::new(OwnedCpuChecker::new(cpu)?))
    }
    /// Startup-selected helper, independent of the arena opponent. Foreign raw
    /// scores remain observations; its checked PV expands the candidate graph
    /// whose frontier is evaluated in the model's separately registered WDL.
    pub fn new_with_checker<C: CpuChecker + 'static>(
        config: PalsConfig,
        model: M,
        checker: C,
    ) -> Result<Self, PalsError> {
        Self::new_with_boxed_checker(config, model, Box::new(checker))
    }
    pub fn new_with_checker_and_refinement_policy<C: CpuChecker + 'static>(
        config: PalsConfig,
        model: M,
        checker: C,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, PalsError> {
        Self::new_with_boxed_checker_and_refinement_policy(config, model, Box::new(checker), policy)
    }
    pub fn new_with_boxed_checker(
        config: PalsConfig,
        model: M,
        cpu: Box<dyn CpuChecker>,
    ) -> Result<Self, PalsError> {
        Self::new_with_boxed_checker_and_refinement_policy(
            config,
            model,
            cpu,
            PostRepairRecheckPolicy::Disabled,
        )
    }
    pub fn new_with_boxed_checker_and_refinement_policy(
        config: PalsConfig,
        model: M,
        cpu: Box<dyn CpuChecker>,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, PalsError> {
        config.validate()?;
        cpu.identity().validate()?;
        if policy == PostRepairRecheckPolicy::SameRepairedLineOnceV1
            && matches!(cpu.identity(), CheckerIdentity::ExternalUci(_))
        {
            return Err(CpuError::Unsupported(
                "post-Repair recheck v1 requires completed own CPU evidence",
            )
            .into());
        }
        let capabilities = cpu.capabilities();
        if !capabilities.divergence || !capabilities.root_moves {
            return Err(CpuError::Unsupported(
                "PALS independent CPU divergence/root-move discovery",
            )
            .into());
        }
        if capabilities.max_depth == 0
            || capabilities.max_depth > 64
            || capabilities.max_prefix_plies < config.line_plies.saturating_sub(1)
            || capabilities.max_root_moves < 256
            || cpu.conditions().is_empty()
            || cpu.conditions().len()
                > if matches!(cpu.identity(), CheckerIdentity::ExternalUci(_)) {
                    8192
                } else {
                    2048
                }
            || cpu.conditions().chars().any(char::is_control)
        {
            return Err(CpuError::Unsupported("PALS CPU bounded search capabilities").into());
        }
        let (cpu_registered_value, model_registered_value) = match cpu.identity() {
            CheckerIdentity::Owned(identity) => {
                let descriptor = cpu
                    .owned_descriptor()
                    .ok_or(CheckerError::Invalid("owned configuration missing"))?;
                if descriptor.config.max_depth == 0
                    || descriptor.config.max_depth > 64
                    || descriptor.config.quiescence_ply > 32
                    || capabilities.max_depth < descriptor.config.max_depth
                    || descriptor.search_identity.is_empty()
                {
                    return Err(CpuError::Unsupported("PALS own CPU configuration").into());
                }
                (Some(identity.clone()), None)
            }
            CheckerIdentity::ExternalUci(_) => {
                if cpu.owned_descriptor().is_some() || capabilities.resume {
                    return Err(CheckerError::Invalid(
                        "foreign helper cannot disclose own/resume semantics",
                    )
                    .into());
                }
                let identity = model
                    .value_identity()
                    .ok_or(RoleError::Unavailable)?
                    .clone();
                identity.validate()?;
                if identity.model != model.identity()
                    || identity.semantics != MODEL_WDL_VALUE_SEMANTICS
                {
                    return Err(RoleError::InvalidOutput.into());
                }
                (None, Some(identity))
            }
        };
        let checker_registered_identity = cpu.identity().clone();
        let cpu_registered_condition = Self::cpu_condition_for(cpu.as_ref());
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
            post_repair_recheck: policy,
            model,
            cpu,
            checker_registered_identity,
            cpu_registered_value,
            model_registered_value,
            cpu_registered_condition,
            checker_new_game_pending: false,
            last_external_attempt: None,
            external_attempts: Vec::new(),
            nodes,
            records,
            revision: 0,
            game_generation: Some(0),
            stores,
            clock_origin: Instant::now(),
            consumer_id: 0,
            last_search_counters: None,
        })
    }
    pub fn new_game(&mut self) {
        self.game_generation = self
            .game_generation
            .and_then(|generation| generation.checked_add(1));
        self.nodes.clear();
        self.records.clear();
        // Preserve the old synchronous own clear. A foreign reset can fail and
        // needs caller controls; defer it to search or explicit try_new_game.
        if self.is_external() {
            self.checker_new_game_pending = true;
        } else {
            let never_cancel = AtomicBool::new(false);
            self.checker_new_game_pending = self
                .cpu
                .new_game(Instant::now() + Duration::from_secs(1), &never_cancel)
                .is_err();
        }
        self.model.new_game_with_generation(self.game_generation);
        self.revision = 0;
        self.stores = PalsStores::new(Self::store_limits(&self.config));
        self.consumer_id = 0;
        self.last_search_counters = None;
        self.external_attempts.clear();
        self.last_external_attempt = None;
    }
    pub fn try_new_game(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.new_game();
        if self.checker_new_game_pending {
            self.cpu.new_game(deadline, cancel)?;
        }
        self.checker_new_game_pending = false;
        self.last_external_attempt = None;
        Ok(())
    }
    pub fn shutdown_checker(&mut self, deadline: Instant) -> Result<CheckerShutdown, PalsError> {
        Ok(self.cpu.shutdown(deadline)?)
    }
    pub fn start_checker(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.validate_checker_namespace()?;
        if self.cpu.identity() != &self.checker_registered_identity
            || self.cpu_condition() != self.cpu_registered_condition
        {
            return Err(StoreError::InvalidConditions("checker startup namespace changed").into());
        }
        self.cpu.start(deadline, cancel)?;
        self.validate_checker_namespace()?;
        if self.cpu.identity() != &self.checker_registered_identity
            || self.cpu_condition() != self.cpu_registered_condition
        {
            return Err(StoreError::InvalidConditions("checker startup namespace changed").into());
        }
        Ok(())
    }
    pub fn checker_identity(&self) -> &CheckerIdentity {
        &self.checker_registered_identity
    }
    /// Actual foreign attempt/process evidence, including failed/partial work.
    /// Bounded per search; raw foreign scores do not enter the own resolver.
    pub fn checker_attempts(&self) -> &[CheckerAttempt] {
        &self.external_attempts
    }
    /// Latest actual adapter ledger, including lifecycle updates after a search.
    /// This is not a second physical attempt or an additional work charge.
    pub fn checker_last_attempt(&self) -> Option<&CheckerAttempt> {
        self.cpu.last_attempt()
    }
    pub fn checker_startup_uci(&self) -> Option<crate::cpu_checker::ExternalUciIdentity> {
        self.cpu.startup_uci()
    }
    fn is_external(&self) -> bool {
        matches!(
            self.checker_registered_identity,
            CheckerIdentity::ExternalUci(_)
        )
    }
    fn owned_descriptor(&self) -> Result<OwnedCheckerDescriptor, PalsError> {
        self.cpu
            .owned_descriptor()
            .ok_or(CheckerError::Invalid("own CPU-only operation on foreign helper").into())
    }
    fn owned_identity(&self) -> Result<&CpuValueIdentity, PalsError> {
        match self.cpu.identity() {
            CheckerIdentity::Owned(identity) => Ok(identity),
            CheckerIdentity::ExternalUci(_) => {
                Err(CheckerError::Invalid("foreign helper has no own score identity").into())
            }
        }
    }
    fn validate_checker_namespace(&self) -> Result<(), PalsError> {
        self.cpu.validate_namespace().map_err(|error| match error {
            CheckerError::Invalid(reason) => StoreError::InvalidConditions(reason).into(),
            error => error.into(),
        })
    }
    fn checker_admission_error(error: CheckerError) -> PalsError {
        match error {
            CheckerError::External {
                stage: "admission",
                code: "insufficient_stop_reserve",
            } => RoleError::Deadline.into(),
            error => error.into(),
        }
    }
    fn own_report(report: CheckerReport) -> Result<CpuReport, PalsError> {
        match report {
            CheckerReport::Owned(report) => Ok(report),
            CheckerReport::ExternalUci(_) => {
                Err(CheckerError::Invalid("foreign report delivered to own resolver").into())
            }
        }
    }
    fn resolver_version(&self) -> &'static str {
        if self.is_external() {
            MODEL_WDL_RESOLVER_VERSION
        } else {
            PALS_VALUE_RESOLVER_VERSION
        }
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
    pub fn post_repair_recheck_policy(&self) -> PostRepairRecheckPolicy {
        self.post_repair_recheck
    }
    /// Consumers must bind this effective identity, not only the legacy constant.
    pub fn search_identity(&self) -> &'static str {
        self.post_repair_recheck.search_identity()
    }
    pub fn refinement_conditions(&self) -> Option<&'static str> {
        self.post_repair_recheck.conditions()
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
        if self.game_generation.is_none() {
            return Err(PalsError::Capacity);
        }
        let mut counters = PalsCounters::default();
        self.external_attempts.clear();
        // Actual prepared RoleQuery includes records, revision and deadline.
        // Previous-search WDL is not an exact-input cache hit in this context.
        for node in &mut self.nodes {
            node.model_value = None;
        }
        let result = self.search_inner(position, limits, cancel, progress, &mut counters);
        if let Ok(report) = &result {
            counters = report.counters;
        }
        counters.retained_situations = self.nodes.len();
        self.last_search_counters = Some(counters);
        let closure = match &result {
            Ok(report) => match report.completion {
                PalsCompletion::Canceled => RoleSearchClosure::Canceled,
                PalsCompletion::Deadline => RoleSearchClosure::Deadline,
                _ => RoleSearchClosure::Completed,
            },
            Err(PalsError::Role(RoleError::PhysicalCompletionUnknown)) => {
                RoleSearchClosure::PhysicalCompletionUnknown
            }
            Err(_) => RoleSearchClosure::Failed,
        };
        self.model.finish_search(closure);
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
        self.validate_checker_namespace()?;
        if self.cpu.identity() != &self.checker_registered_identity
            || self.cpu_condition() != self.cpu_registered_condition
            || (self.is_external()
                && self.model.value_identity() != self.model_registered_value.as_ref())
        {
            return Err(StoreError::InvalidConditions(
                "startup-selected CPU search/value identity changed",
            )
            .into());
        }
        if limits.max_rounds == 0
            || limits.max_cpu_nodes == 0
            || limits.cpu_depth == 0
            || limits.cpu_depth > self.cpu.capabilities().max_depth
        {
            return Err(PalsError::InvalidLimits);
        }
        if self.checker_new_game_pending {
            self.cpu.new_game(limits.deadline, cancel)?;
            self.checker_new_game_pending = false;
            self.last_external_attempt = None;
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
                score: if self.is_external() {
                    None
                } else {
                    Some(value)
                },
                resolved_value: self.rules_value(root),
                model_value_identity: self.model_registered_value.clone(),
                checker_identity: self.checker_registered_identity.clone(),
                value_scope: PalsValueScope::RulesTerminal,
                terminal: Some(reason),
                completion: PalsCompletion::Terminal,
                counters: *counters,
                root_values: Vec::new(),
                elapsed: started.elapsed(),
                model_identity: self.model.identity().to_owned(),
                resolver_version: self.resolver_version(),
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
            if self.cpu_budget_used(counters) >= limits.max_cpu_nodes {
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
        if self.is_external() {
            return self.model_result(root, &legal, completion, started, counters);
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
                resolved_value: value.map_or(PalsResolvedValue::Unknown, |value| {
                    PalsResolvedValue::OwnedRaw {
                        value,
                        perspective: position.side_to_move(),
                    }
                }),
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
            resolved_value: score.map_or(PalsResolvedValue::Unknown, |value| {
                PalsResolvedValue::OwnedRaw {
                    value,
                    perspective: position.side_to_move(),
                }
            }),
            model_value_identity: self.model_registered_value.clone(),
            checker_identity: self.checker_registered_identity.clone(),
            value_scope: scope,
            terminal: None,
            completion,
            counters: *counters,
            root_values,
            elapsed: started.elapsed(),
            model_identity: self.model.identity().to_owned(),
            resolver_version: self.resolver_version(),
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
            score: if self.is_external() {
                None
            } else {
                terminal.map(|(_, score)| score)
            },
            resolved_value: match classification.play_status {
                PlayStatus::Terminal { winner, .. } => PalsResolvedValue::RulesTerminal {
                    winner,
                    perspective: position.side_to_move(),
                },
                PlayStatus::Ongoing => PalsResolvedValue::Unknown,
            },
            model_value_identity: self.model_registered_value.clone(),
            checker_identity: self.checker_registered_identity.clone(),
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
                        resolved_value: PalsResolvedValue::Unknown,
                        scope: PalsValueScope::Unknown,
                        examined_replies: 0,
                        unexplored_replies: None,
                    })
                    .collect()
            },
            elapsed: started.elapsed(),
            model_identity: self.model.identity().to_owned(),
            resolver_version: self.resolver_version(),
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
            model_value: None,
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
        let question = RoleQuestion {
            purpose: match call {
                Call::Propose => RoleQueryPurpose::ProposePolicy,
                Call::Reply => RoleQueryPurpose::ReplyPolicy,
                Call::Repair => RoleQueryPurpose::RepairPolicy,
            },
            prefix,
            proposal,
            refutation,
            divergences: &[],
        };
        let context = self.role_context(node, question)?;
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
            Call::Propose => self.model.propose_with_context(query, &context)?,
            Call::Reply => self.model.reply_with_context(query, &context)?,
            Call::Repair => self.model.repair_with_context(query, &context)?,
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
        self.accept_role_output(node, &context, question, limits, cancel)?;
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

    fn role_context(
        &self,
        node: usize,
        question: RoleQuestion<'_>,
    ) -> Result<RoleLogicalContext, PalsError> {
        let game_generation = self.game_generation.ok_or(PalsError::Capacity)?;
        let node = self
            .nodes
            .get(node)
            .ok_or(StoreError::InvalidHandle("role node"))?;
        let situation = self.stores.situations.get(node.situation)?;
        if situation.state != node.state
            || !self
                .stores
                .states
                .get(node.state)?
                .same_state(&node.position.snapshot())
            || self.stores.lines.get(situation.focus)?.start_state != node.state
        {
            return Err(StoreError::InvalidHandle("role checked situation/state/focus").into());
        }
        if question.prefix.len() > self.config.line_plies
            || question.proposal.len() > self.config.line_plies
            || question
                .refutation
                .is_some_and(|line| line.len() > self.config.line_plies)
            || question.divergences.len() > self.config.line_plies
        {
            return Err(PalsError::Capacity);
        }
        let focus = self.stores.lines.moves(situation.focus)?;
        let mut prefix = Vec::new();
        prefix
            .try_reserve_exact(question.prefix.len())
            .map_err(|_| PalsError::Capacity)?;
        prefix.extend_from_slice(question.prefix);
        Ok(RoleLogicalContext {
            game_generation,
            search_generation: self.stores.generation(),
            situation: node.situation,
            state: node.state,
            focus: situation.focus,
            purpose: question.purpose,
            prefix,
            focus_sha256: role_line_sha256(b"focus", &focus)?,
            prefix_sha256: role_line_sha256(b"prefix", question.prefix)?,
            proposal_sha256: role_line_sha256(b"proposal", question.proposal)?,
            refutation_sha256: question
                .refutation
                .map(|line| role_line_sha256(b"refutation", line))
                .transpose()?,
            divergence_sha256: role_divergence_sha256(question.divergences)?,
            public_revision: self.revision,
            situation_revision: situation.revision,
        })
    }

    fn accept_role_output(
        &mut self,
        node: usize,
        prepared: &RoleLogicalContext,
        question: RoleQuestion<'_>,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        let current = self.role_context(node, question)?;
        if current != *prepared {
            return Err(RoleError::InvalidOutput.into());
        }
        let snapshot = self.nodes[node].position.snapshot();
        self.model.accepted_output_checked(RoleAcceptance {
            snapshot: &snapshot,
            context: &current,
            deadline: limits.deadline,
            cancel,
        })?;
        Ok(())
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
        self.validate_checker_namespace()?;
        if self.is_external() {
            let root_state = self
                .stores
                .situations
                .get(self.stores.root().ok_or(StoreError::StaleConsumer)?)?
                .state;
            let root = self
                .nodes
                .iter()
                .position(|candidate| candidate.state == root_state)
                .ok_or(StoreError::StaleConsumer)?;
            // The foreign leaf analysis is retained as raw evidence. Its CP or
            // mate report never supplies the frontier value consumed below.
            self.external_request(root, node, line, None, true, limits, cancel, counters)?;
            return self.evaluate_model_value(node, line, limits, cancel, counters);
        }
        if self.nodes[node].evidence.as_ref().is_some_and(|e| {
            e.depth >= limits.cpu_depth
                && e.scope == CpuScoreScope::CompletedIteration
                && Some(&e.value_identity) == self.cpu_registered_value.as_ref()
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
        let descriptor = self.owned_descriptor()?;
        let profile = stable_id(descriptor.config.profile.identity());
        let value_identity = self.owned_identity()?.clone();
        let cpu_condition = self.cpu_condition();
        let condition = stable_id(&format!(
            "{};q={}",
            descriptor.search_identity, descriptor.config.quiescence_ply
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
                value_identity: Some(value_identity.clone()),
                checker_identity: None,
                cpu_condition: Some(cpu_condition.clone()),
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
                            value_identity: self.owned_identity()?.clone(),
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
                if checkpoint != node as u64
                    || self.nodes[node].resume.is_none()
                    || !self.cpu.capabilities().resume
                {
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
            Ok(report) => match Self::own_report(report) {
                Ok(report) => report,
                Err(error) => {
                    self.observe_cpu_failure(counters, task_nodes);
                    self.stores.tasks.fail(execution)?;
                    return Err(error);
                }
            },
            Err(error) => {
                if matches!(error, CheckerError::OwnedReportRejected { .. }) {
                    counters.cpu_tasks += 1;
                }
                self.observe_cpu_failure(counters, task_nodes);
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
        };
        let admission = self.validate_cpu_report(
            &self.nodes[node].position,
            &report,
            &value_identity,
            &cpu_condition,
            false,
            cpu_limits,
        );
        let requested_coverage_complete =
            Self::observe_cpu_report(counters, &report, limits.cpu_depth, admission.is_ok());
        if let Err(error) = admission {
            self.stores.tasks.fail(execution)?;
            return Err(error);
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
            value_identity: Some(report.value_identity.clone()),
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: Some(cpu_condition.clone()),
            cpu_pv: None,
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
                value_identity: report.value_identity.clone(),
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

    fn cpu_budget_used(&self, counters: &PalsCounters) -> u64 {
        if self.is_external() {
            counters.external_checker_node_budget_reserved
        } else {
            counters.cpu_nodes
        }
    }

    fn evaluate_model_value(
        &mut self,
        node: usize,
        line: &[BoardMove],
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<(), PalsError> {
        if self.nodes[node].terminal.is_some() || self.stopped(limits, cancel).is_some() {
            return Ok(());
        }
        if counters.role_calls >= self.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        let identity = self
            .model_registered_value
            .as_ref()
            .ok_or(RoleError::Unavailable)?
            .clone();
        if self.model.value_identity() != Some(&identity) {
            return Err(RoleError::InvalidOutput.into());
        }
        let legal = self.nodes[node].position.legal_moves();
        if legal.len() > 256 {
            return Err(PalsError::Capacity);
        }
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::ValueFresh,
            prefix: line,
            proposal: line,
            refutation: None,
            divergences: &[],
        };
        let context = self.role_context(node, question)?;
        let query = RoleQuery {
            position: &self.nodes[node].position,
            legal: &legal,
            prefix: line,
            proposal: line,
            counterexample: None,
            records: &self.records,
            revision: self.revision,
            deadline: limits.deadline,
            cancel,
        };
        query.check_control()?;
        counters.role_calls += 1;
        counters.value_calls += 1;
        let output = self.model.evaluate_value_with_context(query, &context)?;
        output.validate(&self.nodes[node].position, &identity)?;
        counters.completed_value_calls += 1;
        // Keep the full actual input identity; numeric scope IDs are metadata.
        // No exact-input cache reuse is granted by a model namespace or board.
        let observation = self.stores.append_observation(Observation {
            state: self.nodes[node].state,
            line: None,
            source: stable_id(&identity.semantics),
            epoch: 0,
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: Some(identity.clone()),
            model_value_input: Some(output.input_sha256),
            cpu_condition: None,
            cpu_pv: None,
            scope: EvidenceScope::Model {
                model: stable_id(&identity.model),
                encoding: stable_id(&identity.encoding),
                input: u64::from_le_bytes(
                    output.input_sha256[..8]
                        .try_into()
                        .map_err(|_| RoleError::InvalidOutput)?,
                ),
            },
            score: RawScore::Wdl {
                win: output.wdl[0],
                draw: output.wdl[1],
                loss: output.wdl[2],
                perspective: output.perspective,
            },
            budget: 1,
            kind: ObservationKind::Proposal,
            supersedes: None,
            execution: None,
        })?;
        if cancel.load(Ordering::Acquire) {
            return Err(RoleError::Canceled.into());
        }
        if Instant::now() >= limits.deadline {
            return Err(RoleError::Deadline.into());
        }
        self.stores
            .dependencies
            .add(observation, self.nodes[node].situation)?;
        if let Some(root) = self.stores.root() {
            self.stores.dependencies.add(observation, root)?;
        }
        // Acknowledgement occurs only after successful publication/dependency
        // admission and final controls; failed/late outputs are not consumed.
        if cancel.load(Ordering::Acquire) {
            return Err(RoleError::Canceled.into());
        }
        if Instant::now() >= limits.deadline {
            return Err(RoleError::Deadline.into());
        }
        self.accept_role_output(node, &context, question, limits, cancel)?;
        counters.consumed_role_outputs += 1;
        counters.accepted_value_outputs += 1;
        self.nodes[node].model_value = Some(output);
        Ok(())
    }

    fn validate_external_report(
        &self,
        position: &Position,
        report: &ExternalCheckerReport,
        alternatives: Option<&[BoardMove]>,
        limits: CpuLimits,
    ) -> Result<(), PalsError> {
        report.identity.validate()?;
        if self.cpu.identity() != &self.checker_registered_identity
            || self.checker_registered_identity
                != CheckerIdentity::ExternalUci(report.identity.clone())
            || self.cpu.conditions() != self.cpu_registered_condition
            || report.perspective != position.side_to_move()
            || report.requested_depth != limits.max_depth
            || report.root_restricted != alternatives.is_some()
            || report.pv.len() > 256
            || report.pv.capacity() > 256
            || report
                .work
                .nodes
                .is_some_and(|nodes| nodes > limits.max_nodes)
            || (report.best_move.is_some() && report.best_move != report.pv.first().copied())
            || (report.completion == ExternalCompletion::BestMove && report.best_move.is_none())
            || alternatives.is_some_and(|moves| {
                report
                    .best_move
                    .is_some_and(|movement| !moves.contains(&movement))
            })
            || report.wdl_per_mille.is_some_and(|wdl| {
                wdl.into_iter().any(|value| value > 1000)
                    || wdl.into_iter().map(u32::from).sum::<u32>() != 1000
            })
        {
            return Err(StoreError::InvalidEvidence(
                "foreign helper report differs from immutable request",
            )
            .into());
        }
        let mut checked = position.clone();
        for movement in &report.pv {
            let legal = checked.ordered_legal_moves();
            if !matches!(checked.play_status_from_view(&legal)?, PlayStatus::Ongoing) {
                return Err(StoreError::InvalidEvidence(
                    "foreign PV continues after Rules terminal",
                )
                .into());
            }
            checked.make_from_view(&legal, *movement)?;
        }
        Ok(())
    }

    /// Capture one physical foreign request once. Budget reservations and actual
    /// reported nodes are different quantities, including on node overshoot.
    fn observe_external_attempt(&mut self, counters: &mut PalsCounters) -> Result<(), PalsError> {
        let Some(attempt) = self.cpu.last_attempt() else {
            counters.external_checker_work_incomplete = true;
            return Ok(());
        };
        let Some(evidence) = &attempt.external else {
            counters.external_checker_work_incomplete = true;
            return Err(CheckerError::Invalid("foreign attempt has own ledger").into());
        };
        if self
            .last_external_attempt
            .is_some_and(|prior| evidence.request_id <= prior)
        {
            return Err(CheckerError::Invalid("foreign physical request ID was reused").into());
        }
        self.last_external_attempt = Some(evidence.request_id);
        if let Some(nodes) = attempt.work.nodes {
            counters.external_checker_nodes_observed = counters
                .external_checker_nodes_observed
                .checked_add(nodes)
                .ok_or(PalsError::Capacity)?;
        }
        if attempt.work.nodes.is_none()
            || attempt.work.qnodes.is_none()
            || attempt.work.tt_hits.is_none()
        {
            counters.external_checker_work_incomplete = true;
        }
        // Bound an arbitrary injected checker before cloning retained payload.
        if let Some(report) = &evidence.partial_report {
            report.identity.validate()?;
            if report.pv.len() > 256
                || report.pv.capacity() > 256
                || report.observed_uci.name.len() > 256
                || report.observed_uci.name.capacity() > 256
                || report
                    .observed_uci
                    .author
                    .as_ref()
                    .is_some_and(|author| author.len() > 256 || author.capacity() > 256)
            {
                return Err(CheckerError::Invalid("unbounded retained foreign attempt").into());
            }
        }
        if self.external_attempts.len() >= self.config.max_records {
            return Err(PalsError::Capacity);
        }
        self.external_attempts.push(attempt.clone());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn external_candidate(
        &mut self,
        root: usize,
        divergence: usize,
        prefix: &[BoardMove],
        alternatives: Option<&[BoardMove]>,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<Option<CpuCandidate>, PalsError> {
        self.external_request(
            root,
            divergence,
            prefix,
            alternatives,
            false,
            limits,
            cancel,
            counters,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn external_request(
        &mut self,
        root: usize,
        divergence: usize,
        prefix: &[BoardMove],
        alternatives: Option<&[BoardMove]>,
        verification: bool,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<Option<CpuCandidate>, PalsError> {
        if self.stopped(limits, cancel).is_some()
            || self.nodes[divergence].terminal.is_some()
            || alternatives.is_some_and(|moves| moves.is_empty())
        {
            return Ok(None);
        }
        self.validate_checker_namespace()?;
        if self.external_attempts.len() >= self.config.max_records {
            return Err(PalsError::Capacity);
        }
        let remaining = limits
            .max_cpu_nodes
            .saturating_sub(self.cpu_budget_used(counters));
        if remaining == 0 {
            return Ok(None);
        }
        let task_nodes = remaining.min(self.config.cpu_nodes_per_task);
        let cpu_limits = CpuLimits {
            max_depth: limits.cpu_depth,
            max_nodes: task_nodes,
            deadline: Some(limits.deadline),
        };
        // A too-short foreign wall closes this search before allocating a task,
        // reserving nodes or recording physical dispatch/unknown work.
        self.cpu
            .preflight_task(cpu_limits)
            .map_err(Self::checker_admission_error)?;
        let generation = self.stores.generation();
        let active_root = self.stores.root().ok_or(StoreError::StaleConsumer)?;
        let root_revision = self.stores.situations.get(active_root)?.revision;
        let identity = self.checker_registered_identity.clone();
        let question = if verification {
            TaskQuestion::AnalyzePosition
        } else if alternatives.is_some() {
            TaskQuestion::AnalyzeRootMoves
        } else {
            TaskQuestion::FindAlternative {
                divergence_ply: prefix.len() as u16,
            }
        };
        let root_moves = alternatives
            .unwrap_or(&[])
            .iter()
            .copied()
            .map(Move16::pack)
            .collect::<Result<Vec<_>, _>>()?;
        let ordered_mask: String = root_moves
            .iter()
            .map(|movement| format!("{:04x}", movement.bits()))
            .collect();
        // Actual wall deadline is part of the request. A completed result from
        // a different time allowance is not an exact-question cache hit.
        let condition = format!(
            "{};question={:?};ordered-root-mask={};prefix-plies={};deadline-tick={};nodes={};depth={}",
            self.cpu_registered_condition,
            question,
            ordered_mask,
            prefix.len(),
            self.tick_at(limits.deadline),
            task_nodes,
            limits.cpu_depth
        );
        if condition.len() > 8192 {
            return Err(
                StoreError::InvalidConditions("foreign full condition exceeds bound").into(),
            );
        }
        self.consumer_id = self.consumer_id.checked_add(1).ok_or(PalsError::Capacity)?;
        let consumer_id = self.consumer_id;
        let consumer = TaskConsumer {
            id: consumer_id,
            situation: self.nodes[divergence].situation,
            revision: self
                .stores
                .situations
                .get(self.nodes[divergence].situation)?
                .revision,
            generation,
            deadline_tick: self.tick_at(limits.deadline),
        };
        let admission = match self.stores.request_task(
            TaskKey {
                state: self.nodes[divergence].state,
                line: None,
                question,
                root_moves,
                value_identity: None,
                checker_identity: Some(identity.clone()),
                cpu_condition: Some(condition.clone()),
                model: 0,
                epoch: 0,
                profile: 0,
                condition: stable_id(&condition),
                input_revision: prefix.len() as u64,
                requested_depth: limits.cpu_depth,
                node_budget: task_nodes,
            },
            consumer,
            self.tick_at(Instant::now()),
        ) {
            Ok(admission) => admission,
            Err(StoreError::ExpiredConsumer) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let (execution, reused) = match admission {
            TaskAdmission::Start(execution) => (execution, None),
            TaskAdmission::Reuse {
                execution,
                observation,
            } => (execution, Some(observation)),
            TaskAdmission::Resume { .. } => {
                return Err(CheckerError::Unsupported("foreign resume checkpoint").into());
            }
            TaskAdmission::Join(_) => {
                return Err(StoreError::InvalidConditions(
                    "single active foreign go cannot join unknown owner",
                )
                .into());
            }
        };
        let observation = if let Some(observation) = reused {
            observation
        } else {
            counters.external_checker_node_budget_reserved = counters
                .external_checker_node_budget_reserved
                .checked_add(task_nodes)
                .ok_or(PalsError::Capacity)?;
            counters.external_checker_tasks += 1;
            self.cpu.reset_attempt();
            let response = if verification {
                self.cpu
                    .analyze(&self.nodes[divergence].position, cpu_limits, cancel)
            } else if let Some(moves) = alternatives {
                self.cpu.analyze_root_moves(
                    &self.nodes[divergence].position,
                    moves,
                    cpu_limits,
                    cancel,
                )
            } else {
                self.cpu
                    .analyze_divergence(&self.nodes[root].position, prefix, cpu_limits, cancel)
            };
            if matches!(
                &response,
                Err(CheckerError::External {
                    stage: "admission",
                    code: "insufficient_stop_reserve"
                })
            ) && self.cpu.last_attempt().is_none()
            {
                // The real adapter rechecks immediately before go. A race can
                // refuse after caller reservation, but it started no physical
                // work: close the task and release only this unused reservation.
                self.stores.tasks.fail(execution)?;
                counters.external_checker_node_budget_reserved = counters
                    .external_checker_node_budget_reserved
                    .checked_sub(task_nodes)
                    .ok_or(StoreError::InvalidConditions(
                        "foreign reservation underflow",
                    ))?;
                counters.external_checker_tasks =
                    counters.external_checker_tasks.checked_sub(1).ok_or(
                        StoreError::InvalidConditions("foreign dispatch count underflow"),
                    )?;
                return Err(RoleError::Deadline.into());
            }
            let capture = self.observe_external_attempt(counters);
            let report = match response {
                Ok(CheckerReport::ExternalUci(report)) => {
                    counters.external_checker_reports += 1;
                    report
                }
                Ok(CheckerReport::Owned(_)) => {
                    self.stores.tasks.fail(execution)?;
                    return Err(CheckerError::Invalid("own report in foreign namespace").into());
                }
                Err(error) => {
                    self.stores.tasks.fail(execution)?;
                    capture?;
                    return Err(error.into());
                }
            };
            if let Err(error) = capture {
                self.stores.tasks.fail(execution)?;
                return Err(error);
            }
            if let Err(error) = self.validate_external_report(
                &self.nodes[divergence].position,
                &report,
                alternatives,
                cpu_limits,
            ) {
                self.stores.tasks.fail(execution)?;
                return Err(error);
            }
            if self
                .cpu
                .last_attempt()
                .and_then(|attempt| attempt.external.as_ref())
                .is_none_or(|attempt| {
                    attempt.request_id != report.request_id
                        || attempt.partial_report.as_ref() != Some(&report)
                })
            {
                self.stores.tasks.fail(execution)?;
                return Err(CheckerError::Invalid(
                    "foreign report has no matching physical attempt",
                )
                .into());
            }
            let publication = (|| -> Result<ObservationId, PalsError> {
                let projection: Vec<_> = report
                    .pv
                    .iter()
                    .copied()
                    .take(self.config.line_plies)
                    .collect();
                let pv_line = if projection.is_empty() {
                    None
                } else {
                    Some(
                        self.stores
                            .append_cpu_pv(&self.nodes[divergence].position, &projection)?,
                    )
                };
                let observation = self.stores.append_observation(Observation {
                    state: self.nodes[divergence].state,
                    line: None,
                    source: stable_id(&report.identity.adapter_semantics),
                    epoch: 0,
                    value_identity: None,
                    checker_identity: Some(identity.clone()),
                    checker_work: Some(report.work),
                    external_report: Some(Box::new(report.clone())),
                    model_value_identity: None,
                    model_value_input: None,
                    cpu_condition: Some(condition.clone()),
                    cpu_pv: pv_line,
                    scope: EvidenceScope::ExternalUci {
                        requested_depth: report.requested_depth,
                        reported_depth: report.reported_depth,
                        seldepth: report.seldepth,
                        bound: report.bound,
                    },
                    score: RawScore::ExternalUci {
                        value: report.score,
                        bound: report.bound,
                        perspective: report.perspective,
                        wdl_per_mille: report.wdl_per_mille,
                    },
                    budget: report.work.nodes.unwrap_or(0),
                    kind: ObservationKind::ExternalCpuAnalysis,
                    supersedes: None,
                    execution: Some(execution),
                })?;
                Ok(observation)
            })();
            let observation = match publication {
                Ok(observation) => observation,
                Err(error) => {
                    self.stores.tasks.fail(execution)?;
                    return Err(error);
                }
            };
            if report.completion == ExternalCompletion::BestMove && report.work.nodes.is_some() {
                if let Err(error) = self.stores.complete_task(execution, observation) {
                    self.stores.tasks.fail(execution)?;
                    return Err(error.into());
                }
            } else {
                self.stores.tasks.fail(execution)?;
            }
            observation
        };
        let evidence = self.stores.observations.get(observation)?;
        let report = evidence
            .external_report
            .as_deref()
            .ok_or(StoreError::InvalidEvidence("foreign task lost report"))?;
        let valid_response =
            report.completion == ExternalCompletion::BestMove && report.best_move.is_some();
        let complete = valid_response && report.work.nodes.is_some();
        let pv: Vec<_> = report
            .pv
            .iter()
            .copied()
            .take(self.config.line_plies.saturating_sub(prefix.len()))
            .collect();
        if self.stopped(limits, cancel).is_some()
            || generation != self.stores.generation()
            || !valid_response
        {
            return Ok(None);
        }
        if complete {
            match self
                .stores
                .consume_task(execution, consumer_id, self.tick_at(Instant::now()))
            {
                Ok(_) => {}
                Err(StoreError::ExpiredConsumer) => return Ok(None),
                Err(error) => return Err(error.into()),
            }
        }
        if self.stopped(limits, cancel).is_some() {
            return Ok(None);
        }
        if complete {
            counters.consumed_external_checker_tasks += 1;
        }
        self.stores
            .dependencies
            .add(observation, self.nodes[divergence].situation)?;
        self.stores
            .dependencies
            .add(observation, self.nodes[root].situation)?;
        Ok(Some(CpuCandidate {
            pv,
            observation,
            generation,
            root: active_root,
            root_revision,
            deadline: limits.deadline,
        }))
    }

    fn cpu_condition(&self) -> String {
        Self::cpu_condition_for(self.cpu.as_ref())
    }
    fn cpu_condition_for(cpu: &dyn CpuChecker) -> String {
        if let Some(descriptor) = cpu.owned_descriptor() {
            format!(
                "{};conditions={};profile={:?};q={};tt={};maxdepth={};capabilities={:?}",
                descriptor.search_identity,
                descriptor.search_conditions,
                descriptor.config.profile,
                descriptor.config.quiescence_ply,
                descriptor.config.tt_entries,
                descriptor.config.max_depth,
                descriptor.capabilities
            )
        } else {
            cpu.conditions().to_owned()
        }
    }

    fn validate_cpu_report(
        &self,
        position: &Position,
        report: &CpuReport,
        value_identity: &CpuValueIdentity,
        cpu_condition: &str,
        restricted: bool,
        limits: CpuLimits,
    ) -> Result<(), PalsError> {
        let descriptor = self.owned_descriptor()?;
        if &report.value_identity != value_identity
            || self.owned_identity()? != value_identity
            || self.cpu_condition() != cpu_condition
            || report.search_version != descriptor.search_identity
            || report.profile != descriptor.config.profile
            || report.root_restricted != restricted
            || report.completed_depth > limits.max_depth
            || report.reused_completed_depth > report.completed_depth
            || report.nodes > limits.max_nodes
            || report.quiescence_nodes > report.nodes
            || report.best_move != report.pv.first().copied()
            || report.score.unsigned_abs() > CPU_MATE_SCORE as u32
            || (report.score_scope == CpuScoreScope::FrontierOnly
                && report.score.unsigned_abs() > crate::cpu::CPU_FRONTIER_SCORE_LIMIT as u32)
            || report.score_scope == CpuScoreScope::RulesTerminal
            || (report.score_scope == CpuScoreScope::CompletedIteration
                && report.completed_depth == 0)
            || (report.score_scope == CpuScoreScope::FrontierOnly && report.completed_depth != 0)
            || report.pv.is_empty()
            || report.pv.len()
                > limits.max_depth as usize + descriptor.config.quiescence_ply as usize
            || (report.resume.is_some() && !self.cpu.capabilities().resume)
        {
            return Err(StoreError::InvalidEvidence(
                "CPU report differs from admitted immutable search/value conditions",
            )
            .into());
        }
        let mut checked = position.clone();
        for &movement in &report.pv {
            if !matches!(
                checked.classify_position()?.play_status,
                PlayStatus::Ongoing
            ) {
                return Err(StoreError::InvalidEvidence(
                    "CPU PV continues after exact Rules terminal",
                )
                .into());
            }
            checked.make_move(movement)?;
        }
        Ok(())
    }

    fn observe_cpu_report(
        counters: &mut PalsCounters,
        report: &CpuReport,
        requested_depth: u16,
        admitted: bool,
    ) -> bool {
        counters.cpu_tasks += 1;
        if admitted {
            counters.cpu_nodes += report.nodes;
            counters.cpu_quiescence_nodes += report.quiescence_nodes;
            counters.cpu_tt_hits += report.tt_hits;
        } else {
            counters.cpu_work_observation_incomplete = true;
        }
        let complete = admitted
            && report.score_scope == CpuScoreScope::CompletedIteration
            && report.completed_depth >= requested_depth;
        // Physical coverage precedes immutable publication and consumer acceptance.
        if complete {
            counters.completed_cpu_tasks += 1;
        } else if admitted
            && report.score_scope == CpuScoreScope::CompletedIteration
            && report.completed_depth > 0
        {
            counters.partial_cpu_iterations += 1;
        }
        complete
    }

    fn observe_cpu_failure(&self, counters: &mut PalsCounters, requested_nodes: u64) {
        if let Some(work) = self
            .cpu
            .last_attempt().map(|attempt| attempt.work)
            .filter(|work| matches!((work.nodes, work.qnodes), (Some(nodes), Some(qnodes)) if nodes <= requested_nodes && qnodes <= nodes) && work.tt_hits.is_some())
        {
            counters.cpu_nodes += work.nodes.unwrap_or(0);
            counters.cpu_quiescence_nodes += work.qnodes.unwrap_or(0);
            counters.cpu_tt_hits += work.tt_hits.unwrap_or(0);
        } else {
            counters.cpu_work_observation_incomplete = true;
        }
    }

    /// CPU discoveries are separate from an unrestricted node-value cache. A
    /// restricted root score belongs to this task, never to its newly found leaf.
    #[allow(clippy::too_many_arguments)]
    fn cpu_candidate(
        &mut self,
        root: usize,
        divergence: usize,
        prefix: &[BoardMove],
        alternatives: Option<&[BoardMove]>,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<Option<CpuCandidate>, PalsError> {
        if self.stopped(limits, cancel).is_some() || self.nodes[divergence].terminal.is_some() {
            return Ok(None);
        }
        self.validate_checker_namespace()?;
        let mut checked = self.nodes[root].position.clone();
        for &movement in prefix {
            checked.make_move(movement)?;
        }
        if checked.position_identity() != self.nodes[divergence].position.position_identity()
            || self.stores.root().is_none_or(|situation| {
                self.stores
                    .situations
                    .get(situation)
                    .map_or(true, |s| s.state != self.nodes[root].state)
            })
        {
            return Err(StoreError::StaleConsumer.into());
        }
        if self.is_external() {
            return self.external_candidate(
                root,
                divergence,
                prefix,
                alternatives,
                limits,
                cancel,
                counters,
            );
        }
        let remaining = limits.max_cpu_nodes.saturating_sub(counters.cpu_nodes);
        if remaining == 0 || alternatives.is_some_and(|moves| moves.is_empty()) {
            return Ok(None);
        }
        let generation = self.stores.generation();
        let active_root = self.stores.root().ok_or(StoreError::StaleConsumer)?;
        let root_revision = self.stores.situations.get(active_root)?.revision;
        let task_nodes = remaining.min(self.config.cpu_nodes_per_task);
        let descriptor = self.owned_descriptor()?;
        let profile = stable_id(descriptor.config.profile.identity());
        let condition = stable_id(&format!(
            "{};q={}",
            descriptor.search_identity, descriptor.config.quiescence_ply
        ));
        let value_identity = self.owned_identity()?.clone();
        let search_condition = self.cpu_condition();
        let question = if alternatives.is_some() {
            TaskQuestion::AnalyzeRootMoves
        } else {
            TaskQuestion::FindAlternative {
                divergence_ply: prefix.len() as u16,
            }
        };
        let root_moves: Vec<_> = alternatives
            .unwrap_or(&[])
            .iter()
            .copied()
            .map(Move16::pack)
            .collect::<Result<_, _>>()?;
        let ordered_mask: String = root_moves
            .iter()
            .map(|movement| format!("{:04x}", movement.bits()))
            .collect();
        let cpu_condition = format!(
            "{};question={:?};ordered-root-mask={};prefix-plies={}",
            search_condition,
            question,
            ordered_mask,
            prefix.len()
        );
        self.consumer_id = self.consumer_id.checked_add(1).ok_or(PalsError::Capacity)?;
        let consumer_id = self.consumer_id;
        let consumer = TaskConsumer {
            id: consumer_id,
            situation: self.nodes[divergence].situation,
            revision: self
                .stores
                .situations
                .get(self.nodes[divergence].situation)?
                .revision,
            generation,
            deadline_tick: self.tick_at(limits.deadline),
        };
        let admission = match self.stores.request_task(
            TaskKey {
                state: self.nodes[divergence].state,
                line: None,
                question,
                root_moves,
                value_identity: Some(value_identity.clone()),
                checker_identity: None,
                cpu_condition: Some(cpu_condition.clone()),
                model: 0,
                epoch: 0,
                profile,
                condition,
                // Projected prefix length is part of the bounded continuation question.
                input_revision: prefix.len() as u64,
                requested_depth: limits.cpu_depth,
                node_budget: task_nodes,
            },
            consumer,
            self.tick_at(Instant::now()),
        ) {
            Ok(admission) => admission,
            Err(StoreError::ExpiredConsumer) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let execution = match admission {
            TaskAdmission::Reuse {
                execution,
                observation,
            } => {
                let evidence = self.stores.observations.get(observation)?;
                let Some(line) = evidence.cpu_pv else {
                    return Ok(None);
                };
                let pv = self.stores.lines.moves(line)?;
                if self.stopped(limits, cancel).is_some() || generation != self.stores.generation()
                {
                    return Ok(None);
                }
                match self
                    .stores
                    .consume_task(execution, consumer_id, self.tick_at(Instant::now()))
                {
                    Ok(_) => {}
                    Err(StoreError::ExpiredConsumer) => return Ok(None),
                    Err(error) => return Err(error.into()),
                }
                if self.stopped(limits, cancel).is_some() {
                    return Ok(None);
                }
                counters.evidence_cache_hits += 1;
                counters.consumed_cpu_tasks += 1;
                counters.reused_completed_cpu_tasks_consumed += 1;
                counters.consumed_cached_cpu_values += 1;
                return Ok(Some(CpuCandidate {
                    pv,
                    observation,
                    generation,
                    root: active_root,
                    root_revision,
                    deadline: limits.deadline,
                }));
            }
            TaskAdmission::Start(execution) => execution,
            // Discovery tasks deliberately do not reuse AnalyzePosition's token.
            TaskAdmission::Join(_) | TaskAdmission::Resume { .. } => {
                return Err(StoreError::InvalidConditions(
                    "CPU discovery has no shared in-flight/resume owner",
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
        let report = match alternatives {
            Some(moves) => self.cpu.analyze_root_moves(
                &self.nodes[divergence].position,
                moves,
                cpu_limits,
                cancel,
            ),
            None => {
                self.cpu
                    .analyze_divergence(&self.nodes[root].position, prefix, cpu_limits, cancel)
            }
        };
        let report = match report {
            Ok(report) => match Self::own_report(report) {
                Ok(report) => report,
                Err(error) => {
                    self.observe_cpu_failure(counters, task_nodes);
                    self.stores.tasks.fail(execution)?;
                    return Err(error);
                }
            },
            Err(error) => {
                if matches!(error, CheckerError::OwnedReportRejected { .. }) {
                    counters.cpu_tasks += 1;
                }
                self.observe_cpu_failure(counters, task_nodes);
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
        };
        let admission = self
            .validate_cpu_report(
                &self.nodes[divergence].position,
                &report,
                &value_identity,
                &search_condition,
                alternatives.is_some(),
                cpu_limits,
            )
            .and_then(|()| {
                if alternatives.is_some_and(|moves| {
                    report
                        .best_move
                        .is_none_or(|movement| !moves.contains(&movement))
                }) {
                    Err(StoreError::InvalidEvidence(
                        "CPU candidate is outside the admitted ordered root mask",
                    )
                    .into())
                } else {
                    Ok(())
                }
            });
        let complete =
            Self::observe_cpu_report(counters, &report, limits.cpu_depth, admission.is_ok());
        if let Err(error) = admission {
            self.stores.tasks.fail(execution)?;
            return Err(error);
        }
        // The store facade independently checks the entire CPU PV with Rules.
        // Only the portion fitting the registered PALS line enters its graph.
        let pv: Vec<_> = report
            .pv
            .iter()
            .copied()
            .take(self.config.line_plies.saturating_sub(prefix.len()))
            .collect();
        let pv_line = match self
            .stores
            .append_cpu_pv(&self.nodes[divergence].position, &pv)
        {
            Ok(line) => line,
            Err(error) => {
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
        };
        let observation = match self.stores.append_observation(Observation {
            state: self.nodes[divergence].state,
            line: None,
            source: stable_id(report.score_provenance),
            epoch: 0,
            scope: EvidenceScope::DepthLimited {
                depth: report.completed_depth,
                profile,
                condition,
            },
            value_identity: Some(report.value_identity.clone()),
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: Some(cpu_condition.clone()),
            cpu_pv: Some(pv_line),
            score: if report.score_scope == CpuScoreScope::FrontierOnly {
                RawScore::Estimate {
                    value: report.score as f32,
                    perspective: self.nodes[divergence].position.side_to_move(),
                }
            } else {
                RawScore::Cpu {
                    value: report.score,
                    perspective: self.nodes[divergence].position.side_to_move(),
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
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
        };
        if complete {
            if let Err(error) = self.stores.complete_task(execution, observation) {
                self.stores.tasks.fail(execution)?;
                return Err(error.into());
            }
        } else {
            // Partial discovery evidence survives, but no generic node resume
            // token is borrowed for a differently restricted CPU question.
            self.stores.tasks.fail(execution)?;
        }
        if self.stopped(limits, cancel).is_some() || generation != self.stores.generation() {
            return Ok(None);
        }
        if complete {
            match self
                .stores
                .consume_task(execution, consumer_id, self.tick_at(Instant::now()))
            {
                Ok(_) => {}
                Err(StoreError::ExpiredConsumer) => return Ok(None),
                Err(error) => return Err(error.into()),
            }
        }
        if self.stopped(limits, cancel).is_some() {
            return Ok(None);
        }
        if complete {
            counters.consumed_cpu_tasks += 1;
        } else if report.score_scope == CpuScoreScope::CompletedIteration {
            counters.consumed_partial_cpu_values += 1;
        } else {
            counters.consumed_frontier_cpu_values += 1;
        }
        self.stores
            .dependencies
            .add(observation, self.nodes[divergence].situation)?;
        self.stores
            .dependencies
            .add(observation, self.nodes[root].situation)?;
        Ok(Some(CpuCandidate {
            pv,
            observation,
            generation,
            root: active_root,
            root_revision,
            deadline: limits.deadline,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_cpu_candidate(
        &mut self,
        root: usize,
        divergence: usize,
        prefix: &[BoardMove],
        proposal: &[BoardMove],
        candidate: &CpuCandidate,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<Option<(usize, usize, Vec<BoardMove>)>, PalsError> {
        if self.stopped(limits, cancel).is_some() || Instant::now() >= candidate.deadline {
            return Ok(None);
        }
        if candidate.generation != self.stores.generation()
            || self.stores.root() != Some(candidate.root)
            || self.stores.situations.get(candidate.root)?.revision != candidate.root_revision
        {
            return Err(StoreError::StaleConsumer.into());
        }
        let Some(response) = candidate.pv.first().copied() else {
            return Ok(None);
        };
        if proposal.get(prefix.len()).copied() == Some(response) {
            return Ok(None);
        }
        if candidate.pv.len() > self.config.line_plies.saturating_sub(prefix.len()) {
            return Err(StoreError::Capacity("CPU candidate exceeds registered PALS line").into());
        }
        let evidence = self.stores.observations.get(candidate.observation)?;
        if (if self.is_external() {
            evidence.checker_identity.as_ref() != Some(&self.checker_registered_identity)
        } else {
            evidence.value_identity.as_ref() != self.cpu_registered_value.as_ref()
        }) || evidence.cpu_pv.is_none_or(|line| {
            self.stores.lines.moves(line).map_or(true, |pv| {
                if self.is_external() {
                    !pv.starts_with(&candidate.pv)
                } else {
                    pv != candidate.pv
                }
            })
        }) {
            return Err(StoreError::InvalidEvidence(
                "CPU candidate lost checked PV or value namespace",
            )
            .into());
        }
        let mut checked = self.nodes[root].position.clone();
        for &movement in prefix {
            checked.make_move(movement)?;
        }
        if checked.position_identity() != self.nodes[divergence].position.position_identity()
            || self.stores.root().is_none_or(|situation| {
                self.stores
                    .situations
                    .get(situation)
                    .map_or(true, |s| s.state != self.nodes[root].state)
            })
            || self.stores.observations.get(candidate.observation)?.state
                != self.nodes[divergence].state
        {
            return Err(StoreError::StaleConsumer.into());
        }
        for &movement in &candidate.pv {
            checked.make_move(movement)?;
        }
        if self.stopped(limits, cancel).is_some() {
            return Ok(None);
        }
        let mut line = prefix.to_vec();
        let mut current = divergence;
        let mut first = None;
        for &movement in &candidate.pv {
            if self.stopped(limits, cancel).is_some() {
                return Ok(None);
            }
            let mut next = self.nodes[current].position.clone();
            next.make_move(movement)?;
            current = self.connect(current, movement, next, counters)?;
            first.get_or_insert(current);
            line.push(movement);
        }
        Ok(Some((first.expect("nonempty checked PV"), current, line)))
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
        self.record_with_cpu(
            kind,
            line,
            value,
            completed_depth,
            score_scope,
            perspective,
            None,
        )
    }
    // Immutable CPU evidence is an explicit source, while the public line is
    // anchored at the active root and does not inherit the CPU state's value.
    #[allow(clippy::too_many_arguments)]
    fn record_with_cpu(
        &mut self,
        kind: RecordKind,
        line: &[BoardMove],
        value: Option<i32>,
        completed_depth: u16,
        score_scope: Option<CpuScoreScope>,
        perspective: Option<Color>,
        cpu_observation: Option<ObservationId>,
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
            cpu_observation,
            perspective,
            critical: true,
        });
        if kind == RecordKind::CpuVerification || cpu_observation.is_some() {
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
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
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
            if self.is_external() {
                let value = |movement| {
                    self.nodes[root]
                        .edges
                        .iter()
                        .find(|edge| edge.movement == movement)
                        .and_then(|edge| model_order_value(self.root_model_value(edge.child).0))
                };
                return value(*b)
                    .partial_cmp(&value(*a))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        root_rank
                            .iter()
                            .position(|movement| movement == a)
                            .cmp(&root_rank.iter().position(|movement| movement == b))
                    });
            }
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
            if self.stopped(limits, cancel).is_some()
                || self.cpu_budget_used(counters) >= limits.max_cpu_nodes
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
            let question = RoleQuestion {
                purpose: RoleQueryPurpose::DivergencePolicy,
                prefix: &[],
                proposal: &proposal,
                refutation: None,
                divergences: &divergences,
            };
            let context = self.role_context(root, question)?;
            let scores = self.model.divergences_with_context(
                DivergenceQuery {
                    root: &self.nodes[root].position,
                    proposal: &proposal,
                    candidates: &divergences,
                    records: &self.records,
                    revision: self.revision,
                    deadline: limits.deadline,
                    cancel,
                },
                &context,
            )?;
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
            self.accept_role_output(root, &context, question, limits, cancel)?;
            counters.consumed_role_outputs += 1;
            counters.accepted_critic_outputs += 1;
            let mut ranked: Vec<_> = (0..divergences.len()).collect();
            ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            // Aging rotates among divergence sites: low-prior sites eventually run.
            let ply = divergences[ranked[round as usize % ranked.len()]];
            let divergence = path[ply];
            let prefix = &proposal[..ply];
            // At most two finite CPU questions per divergence: unrestricted
            // independent discovery, then an explicitly restricted alternative
            // question only if its best move kept the original continuation.
            let mut cpu_discovery =
                self.cpu_candidate(root, divergence, prefix, None, limits, cancel, counters)?;
            if cpu_discovery
                .as_ref()
                .is_none_or(|candidate| candidate.pv.first().copied() == Some(proposal[ply]))
            {
                let alternatives: Vec<_> = self.nodes[divergence]
                    .position
                    .legal_moves()
                    .into_iter()
                    .filter(|movement| *movement != proposal[ply])
                    .collect();
                cpu_discovery = self.cpu_candidate(
                    root,
                    divergence,
                    prefix,
                    Some(&alternatives),
                    limits,
                    cancel,
                    counters,
                )?;
            }
            let cpu_line = if let Some(candidate) = &cpu_discovery {
                self.insert_cpu_candidate(
                    root, divergence, prefix, &proposal, candidate, limits, cancel, counters,
                )?
            } else {
                None
            };
            if let (Some((_, _, line)), Some(candidate)) = (&cpu_line, &cpu_discovery) {
                self.record_with_cpu(
                    RecordKind::Counterexample,
                    line,
                    None,
                    0,
                    None,
                    None,
                    Some(candidate.observation),
                )?;
            }
            // C receives the actual CPU-created candidate projection before its
            // next query; P repair below also starts after that exact response.
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
            // Retain the independent CPU branch and one distinct C-selected
            // branch. Both are bounded, evaluated, and repaired; C output is
            // never acknowledged merely to discard its discovered response.
            let mut branches = Vec::with_capacity(2);
            let cpu_response = cpu_line.as_ref().map(|(_, _, line)| line[ply]);
            if let Some((response_node, leaf, line)) = cpu_line {
                branches.push((
                    response_node,
                    leaf,
                    line,
                    cpu_discovery
                        .as_ref()
                        .map(|candidate| candidate.observation),
                ));
            }
            let examined: Vec<_> = self.nodes[divergence]
                .edges
                .iter()
                .map(|e| e.movement)
                .collect();
            let response = replies
                .iter()
                .copied()
                .find(|mv| {
                    *mv != proposal[ply] && Some(*mv) != cpu_response && !examined.contains(mv)
                })
                .or_else(|| {
                    replies
                        .iter()
                        .copied()
                        .find(|mv| *mv != proposal[ply] && Some(*mv) != cpu_response)
                });
            if let Some(response) = response.filter(|_| self.stopped(limits, cancel).is_none()) {
                let mut state = self.nodes[divergence].position.clone();
                state.make_move(response)?;
                let response_node = self.connect(divergence, response, state, counters)?;
                let mut line = prefix.to_vec();
                line.push(response);
                branches.push((response_node, response_node, line, None));
            }
            for (response_node, counter_leaf, mut refutation, cpu_source) in branches {
                if self.stopped(limits, cancel).is_some() {
                    break;
                }
                // A short checked CPU PV may still need C's bounded continuation.
                let counter_leaf = self.follow(
                    counter_leaf,
                    &mut refutation,
                    &proposal,
                    None,
                    Call::Reply,
                    limits,
                    cancel,
                    counters,
                )?;
                let response = refutation[ply];
                counters.refutations += 1;
                if cpu_source.is_some() {
                    self.record_with_cpu(
                        RecordKind::Counterexample,
                        &refutation,
                        None,
                        0,
                        None,
                        None,
                        cpu_source,
                    )?;
                } else {
                    self.record(RecordKind::Counterexample, &refutation, None, 0, None, None)?;
                }
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
                        self.stores.refute_continuation(
                            self.nodes[root].situation,
                            line,
                            evidence,
                        )?;
                        counters.supported_refutations += 1;
                        Some((line, evidence, counter))
                    }
                    _ => None,
                };
                // Repair resumes AFTER the response. The first move remains in the
                // frontier; this conditional counterexample never blacklists it.
                let mut repair = prefix.to_vec();
                repair.push(response);
                let accepted_repairs_before = counters.accepted_repair_outputs;
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
                let repair_record =
                    self.record(RecordKind::Repair, &repair, None, 0, None, None)?;
                let repair_record_revision = self.revision;
                self.verify(repair_leaf, &repair, limits, cancel, counters)?;
                self.publish_choice(root, limits, cancel, progress)?;
                let completed_repair_value =
                    self.completed_line_value(repair_leaf, repair.len(), limits.cpu_depth);
                if let (
                    Some((old_line, old_evidence, old_value)),
                    Some((new_line, _)),
                    Some(new_value),
                ) = (supported_refutation, repair_record, completed_repair_value)
                {
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
                        self.stores.repair(
                            self.nodes[root].situation,
                            old_line,
                            new_line,
                            evidence,
                        )?;
                        counters.supported_repairs += 1;
                    }
                }
                if let (true, Some((repaired_line, _)), Some(completed_value)) = (
                    counters.accepted_repair_outputs > accepted_repairs_before,
                    repair_record,
                    completed_repair_value,
                ) {
                    self.recheck_repaired_line(
                        CompletedRepairRecheck {
                            root,
                            original_first: movement,
                            attack_ply: ply,
                            repaired_line,
                            repair_record_revision,
                            repaired_leaf: repair_leaf,
                            repaired: &repair,
                            refutation: &refutation,
                            completed_value,
                        },
                        limits,
                        cancel,
                        counters,
                        progress,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// One C policy call against an accepted, completed Repair. The remaining
    /// repaired suffix is only Rules replay, never another C call or Repair.
    /// A shortened/illegal suffix stays a provisional counterexample; only a
    /// full equal-length line with comparable completed evidence may refute.
    fn recheck_repaired_line(
        &mut self,
        repair: CompletedRepairRecheck<'_>,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<(), PalsError> {
        if self.post_repair_recheck == PostRepairRecheckPolicy::Disabled || self.is_external() {
            return Ok(());
        }
        if let Some(stop) = self.stopped(limits, cancel) {
            return Err(match stop {
                PalsCompletion::Canceled => RoleError::Canceled,
                _ => RoleError::Deadline,
            }
            .into());
        }
        if self.cpu_budget_used(counters) >= limits.max_cpu_nodes {
            return Ok(());
        }
        self.validate_checker_namespace()?;
        let root = self
            .nodes
            .get(repair.root)
            .ok_or(StoreError::InvalidHandle("repair recheck root"))?;
        let root_situation = root.situation;
        let root_state = root.state;
        let root_color = root.position.side_to_move();
        if self.stores.root() != Some(root_situation) {
            return Err(StoreError::StaleConsumer.into());
        }
        let current = self.stores.situations.get(root_situation)?;
        if current.state != root_state
            || !self
                .stores
                .states
                .get(root_state)?
                .same_state(&root.position.snapshot())
        {
            return Err(StoreError::InvalidEvidence("repair recheck root state mismatch").into());
        }
        let root_revision = current.revision;
        let generation = self.stores.generation();
        let previous_evidence = current
            .conclusions
            .get(repair.repaired_line)
            .and_then(|conclusion| conclusion.evidence);
        if repair.repaired.len() > self.config.line_plies
            || repair.attack_ply >= repair.repaired.len()
            || repair.attack_ply >= repair.refutation.len()
            || repair.repaired.first().copied() != Some(repair.original_first)
            || repair.repaired[..=repair.attack_ply] != repair.refutation[..=repair.attack_ply]
            || self.stores.lines.get(repair.repaired_line)?.start_state != root_state
            || self.stores.lines.moves(repair.repaired_line)?.as_slice() != repair.repaired
            || !self.records.iter().any(|record| {
                record.origin_state == root_state
                    && record.kind == RecordKind::Repair
                    && record.revision == repair.repair_record_revision
                    && record.line.as_slice() == repair.repaired
            })
        {
            return Err(
                StoreError::InvalidEvidence("repair recheck line/revision mismatch").into(),
            );
        }
        if self.completed_line_value(
            repair.repaired_leaf,
            repair.repaired.len(),
            limits.cpu_depth,
        ) != Some(repair.completed_value)
        {
            return Ok(()); // Partial/foreign value namespaces cannot grant this work.
        }
        let mut path = Vec::new();
        path.try_reserve_exact(repair.repaired.len() + 1)
            .map_err(|_| PalsError::Capacity)?;
        path.push(repair.root);
        let mut node = repair.root;
        for movement in repair.repaired {
            node = self.nodes[node]
                .edges
                .iter()
                .find(|edge| edge.movement == *movement)
                .ok_or(StoreError::InvalidHandle("repair recheck path"))?
                .child;
            path.push(node);
        }
        if node != repair.repaired_leaf {
            return Err(StoreError::InvalidEvidence("repair recheck leaf mismatch").into());
        }
        let changed_own = (repair.attack_ply + 1..repair.repaired.len()).find(|&ply| {
            self.nodes[path[ply]].position.side_to_move() == root_color
                && repair.refutation.get(ply) != Some(&repair.repaired[ply])
        });
        let Some(anchor_ply) = changed_own.and_then(|changed| {
            (changed + 1..repair.repaired.len()).find(|&ply| {
                self.nodes[path[ply]].position.side_to_move() != root_color
                    && self.nodes[path[ply]].terminal.is_none()
            })
        }) else {
            return Ok(()); // No eligible opponent anchor is not a proof of defense.
        };
        let anchor = path[anchor_ply];
        let replies = self.ranked(
            anchor,
            Call::Reply,
            &repair.repaired[..anchor_ply],
            repair.repaired,
            Some(repair.refutation),
            limits,
            cancel,
            counters,
        )?;
        let original_response = repair.repaired[anchor_ply];
        let response = replies
            .iter()
            .copied()
            .find(|movement| {
                *movement != original_response
                    && !self.nodes[anchor]
                        .edges
                        .iter()
                        .any(|edge| edge.movement == *movement)
            })
            .or_else(|| {
                replies
                    .iter()
                    .copied()
                    .find(|movement| *movement != original_response)
            });
        let Some(response) = response else {
            return Ok(()); // No alternative response is not an all-defenses result.
        };
        let mut counter = Vec::new();
        counter
            .try_reserve_exact(repair.repaired.len())
            .map_err(|_| PalsError::Capacity)?;
        counter.extend_from_slice(&repair.repaired[..anchor_ply]);
        let mut state = self.nodes[anchor].position.clone();
        state.make_move(response)?;
        let mut counter_leaf = self.connect(anchor, response, state, counters)?;
        counter.push(response);
        let mut full_suffix = true;
        for movement in &repair.repaired[anchor_ply + 1..] {
            if self.stopped(limits, cancel).is_some()
                || self.nodes[counter_leaf].terminal.is_some()
                || !self.nodes[counter_leaf]
                    .position
                    .legal_moves()
                    .contains(movement)
            {
                full_suffix = false;
                break;
            }
            let mut state = self.nodes[counter_leaf].position.clone();
            state.make_move(*movement)?;
            counter_leaf = self.connect(counter_leaf, *movement, state, counters)?;
            counter.push(*movement);
        }
        counters.refutations += 1;
        self.record(RecordKind::Counterexample, &counter, None, 0, None, None)?;
        self.verify(counter_leaf, &counter, limits, cancel, counters)?;
        self.publish_choice(repair.root, limits, cancel, progress)?;
        if self.stores.root() != Some(root_situation)
            || self.stores.generation() != generation
            || self.stores.situations.get(root_situation)?.revision != root_revision
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.validate_checker_namespace()?;
        if !full_suffix || counter.len() != repair.repaired.len() {
            return Ok(());
        }
        if !self.comparable_recheck_evidence(repair.repaired_leaf, counter_leaf, limits.cpu_depth) {
            return Ok(());
        }
        let Some(counter_value) =
            self.completed_line_value(counter_leaf, counter.len(), limits.cpu_depth)
        else {
            return Ok(());
        };
        if counter_value >= repair.completed_value {
            return Ok(());
        }
        self.publish_recheck_refutation(
            &repair,
            counter_value,
            previous_evidence,
            limits,
            cancel,
            counters,
        )
    }
    fn publish_recheck_refutation(
        &mut self,
        repair: &CompletedRepairRecheck<'_>,
        counter_value: i32,
        previous_evidence: Option<ObservationId>,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<(), PalsError> {
        // The earlier publish_choice guard precedes comparison work. Check again
        // at this final mutation boundary before creating conclusion evidence.
        if let Some(stop) = self.stopped(limits, cancel) {
            return Err(match stop {
                PalsCompletion::Canceled => RoleError::Canceled,
                _ => RoleError::Deadline,
            }
            .into());
        }
        let evidence = self.conclusion_observation(
            repair.root,
            repair.repaired_line,
            ObservationKind::Refutation,
            counter_value,
            previous_evidence,
        )?;
        self.stores.refute_continuation(
            self.nodes[repair.root].situation,
            repair.repaired_line,
            evidence,
        )?;
        counters.supported_refutations += 1;
        Ok(())
    }
    fn comparable_recheck_evidence(&self, repaired: usize, counter: usize, required: u16) -> bool {
        let repaired = &self.nodes[repaired];
        let counter = &self.nodes[counter];
        match (repaired.terminal, counter.terminal) {
            (Some(_), Some(_)) => true,
            (None, None) => match (&repaired.evidence, &counter.evidence) {
                (Some(repaired), Some(counter)) => {
                    repaired.scope == CpuScoreScope::CompletedIteration
                        && counter.scope == CpuScoreScope::CompletedIteration
                        && repaired.depth >= required
                        && repaired.depth == counter.depth
                        && repaired.value_identity == counter.value_identity
                        && self.cpu_registered_value.as_ref() == Some(&repaired.value_identity)
                }
                _ => false,
            },
            _ => false,
        }
    }
    fn value(&self, node: usize, remaining: usize) -> Option<i32> {
        if self.is_external() {
            return None;
        }
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
        node.evidence
            .as_ref()
            .filter(|e| Some(&e.value_identity) == self.cpu_registered_value.as_ref())
            .map(|e| e.score)
    }
    fn completed_line_value(&self, leaf: usize, plies: usize, required_depth: u16) -> Option<i32> {
        if self.is_external() {
            return None;
        }
        let value = self.nodes[leaf]
            .terminal
            .map(|(_, value)| value)
            .or_else(|| {
                self.nodes[leaf]
                    .evidence
                    .as_ref()
                    .filter(|e| {
                        e.scope == CpuScoreScope::CompletedIteration
                            && e.depth >= required_depth
                            && Some(&e.value_identity) == self.cpu_registered_value.as_ref()
                    })
                    .map(|e| e.score)
            })?;
        Some(if plies % 2 == 0 { value } else { -value })
    }
    fn rules_value(&self, node: usize) -> PalsResolvedValue {
        let perspective = self.nodes[node].position.side_to_move();
        match self.nodes[node].terminal {
            Some((_, score)) => PalsResolvedValue::RulesTerminal {
                winner: if score > 0 {
                    Some(perspective)
                } else if score < 0 {
                    Some(perspective.opposite())
                } else {
                    None
                },
                perspective,
            },
            None => PalsResolvedValue::Unknown,
        }
    }
    fn model_value(&self, index: usize, remaining: usize) -> PalsResolvedValue {
        let node = &self.nodes[index];
        if node.terminal.is_some() {
            return self.rules_value(index);
        }
        if remaining > 0 {
            let mut best = PalsResolvedValue::Unknown;
            // Equal W-L distributions/conditional outcomes preserve Rules order.
            for movement in node.position.legal_moves() {
                if let Some(edge) = node.edges.iter().find(|edge| edge.movement == movement) {
                    let value = flip_resolved_value(self.model_value(edge.child, remaining - 1));
                    if better_model_choice(value, best, false, false) {
                        best = value;
                    }
                }
            }
            if !matches!(best, PalsResolvedValue::Unknown) {
                return best;
            }
        }
        node.model_value
            .as_ref()
            .filter(|output| {
                Some(&output.identity) == self.model_registered_value.as_ref()
                    && output.state == node.position.position_identity()
                    && output.perspective == node.position.side_to_move()
            })
            .map_or(PalsResolvedValue::Unknown, |output| {
                PalsResolvedValue::ModelWdl {
                    wdl: output.wdl,
                    perspective: output.perspective,
                }
            })
    }
    fn root_model_value(&self, child: usize) -> (PalsResolvedValue, PalsValueScope) {
        let terminal = self.nodes[child].terminal.is_some();
        let value = self.model_value(child, self.config.line_plies.saturating_sub(1));
        let mut flipped = flip_resolved_value(value);
        if terminal {
            // This is the directly selected child's current-state fact. Deeper
            // terminal lines remain RestrictedRulesLine with estimate scope.
            if let PalsResolvedValue::RestrictedRulesLine {
                winner,
                perspective,
            } = flipped
            {
                flipped = PalsResolvedValue::RulesTerminal {
                    winner,
                    perspective,
                };
            }
        }
        let scope = if terminal {
            PalsValueScope::RulesTerminal
        } else if matches!(flipped, PalsResolvedValue::Unknown) {
            PalsValueScope::Unknown
        } else {
            PalsValueScope::RestrictedEstimate
        };
        (flipped, scope)
    }
    fn model_result(
        &self,
        root: usize,
        legal: &[BoardMove],
        completion: PalsCompletion,
        started: Instant,
        counters: &mut PalsCounters,
    ) -> Result<PalsResult, PalsError> {
        let mut best_move = legal.first().copied();
        let mut resolved_value = PalsResolvedValue::Unknown;
        let mut scope = PalsValueScope::Unknown;
        let mut root_values = Vec::with_capacity(legal.len());
        for &movement in legal {
            let edge = self.nodes[root]
                .edges
                .iter()
                .find(|edge| edge.movement == movement);
            let (value, child_scope, examined, unexplored) = if let Some(edge) = edge {
                let child = &self.nodes[edge.child];
                let (value, scope) = self.root_model_value(edge.child);
                let legal_count = if child.terminal.is_some() {
                    0
                } else {
                    child.position.legal_moves().len()
                };
                (
                    value,
                    scope,
                    child.edges.len(),
                    Some(legal_count.saturating_sub(child.edges.len())),
                )
            } else {
                (PalsResolvedValue::Unknown, PalsValueScope::Unknown, 0, None)
            };
            if matches!(value, PalsResolvedValue::Unknown) {
                counters.unknown_root_children += 1;
            }
            if better_model_choice(
                value,
                resolved_value,
                child_scope == PalsValueScope::RulesTerminal,
                scope == PalsValueScope::RulesTerminal,
            ) {
                best_move = Some(movement);
                resolved_value = value;
                scope = child_scope;
            }
            root_values.push(PalsRootValue {
                movement,
                score: None,
                resolved_value: value,
                scope: child_scope,
                examined_replies: examined,
                unexplored_replies: unexplored,
            });
        }
        counters.retained_situations = self.nodes.len();
        counters.root_scope_observation_complete = true;
        Ok(PalsResult {
            best_move,
            score: None,
            resolved_value,
            model_value_identity: self.model_registered_value.clone(),
            checker_identity: self.checker_registered_identity.clone(),
            value_scope: scope,
            terminal: None,
            completion,
            counters: *counters,
            root_values,
            elapsed: started.elapsed(),
            model_identity: self.model.identity().to_owned(),
            resolver_version: self.resolver_version(),
        })
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
            source: stable_id(self.search_identity()),
            epoch: 0,
            scope: EvidenceScope::Model {
                model: stable_id(self.model.identity()),
                encoding: stable_id("pals-restricted-summary-v1"),
                input: self.revision,
            },
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
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
        if self.is_external() {
            let mut best_value = PalsResolvedValue::Unknown;
            let mut best_terminal = false;
            for movement in self.nodes[root].position.legal_moves() {
                if let Some(edge) = self.nodes[root]
                    .edges
                    .iter()
                    .find(|edge| edge.movement == movement)
                {
                    let (value, scope) = self.root_model_value(edge.child);
                    let terminal = scope == PalsValueScope::RulesTerminal;
                    if better_model_choice(value, best_value, terminal, best_terminal) {
                        best = Some(movement);
                        best_value = value;
                        best_terminal = terminal;
                    }
                }
            }
            if let Some(movement) = best {
                progress(movement);
            }
            if let Some(stop) = self.stopped(limits, cancel) {
                return Err(match stop {
                    PalsCompletion::Canceled => RoleError::Canceled,
                    _ => RoleError::Deadline,
                }
                .into());
            }
            return Ok(());
        }
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

fn flip_resolved_value(value: PalsResolvedValue) -> PalsResolvedValue {
    match value {
        PalsResolvedValue::Unknown => PalsResolvedValue::Unknown,
        // Raw own and model resolver namespaces never meet through this helper.
        PalsResolvedValue::OwnedRaw { .. } => PalsResolvedValue::Unknown,
        PalsResolvedValue::ModelWdl { wdl, perspective } => PalsResolvedValue::ModelWdl {
            // These probabilities were validated before evidence admission.
            wdl: [wdl[2], wdl[1], wdl[0]],
            perspective: perspective.opposite(),
        },
        PalsResolvedValue::RulesTerminal {
            winner,
            perspective,
        }
        | PalsResolvedValue::RestrictedRulesLine {
            winner,
            perspective,
        } => PalsResolvedValue::RestrictedRulesLine {
            winner,
            perspective: perspective.opposite(),
        },
    }
}
/// Ordering only: no foreign CP scaling, terminal-to-neural distribution or
/// averaged P/C prediction is constructed. Unknown is not a zero expectation.
fn model_order_value(value: PalsResolvedValue) -> Option<f32> {
    match value {
        PalsResolvedValue::ModelWdl { wdl, .. } => Some(wdl[0] - wdl[2]),
        PalsResolvedValue::RulesTerminal {
            winner,
            perspective,
        }
        | PalsResolvedValue::RestrictedRulesLine {
            winner,
            perspective,
        } => Some(match winner {
            Some(color) if color == perspective => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }),
        PalsResolvedValue::Unknown | PalsResolvedValue::OwnedRaw { .. } => None,
    }
}
fn better_model_choice(
    candidate: PalsResolvedValue,
    current: PalsResolvedValue,
    terminal: bool,
    current_terminal: bool,
) -> bool {
    let Some(value) = model_order_value(candidate) else {
        return false;
    };
    let previous = model_order_value(current);
    let proven_win = terminal && value > 0.0;
    let previous_proven_win = current_terminal && previous.is_some_and(|value| value > 0.0);
    if proven_win != previous_proven_win {
        return proven_win;
    }
    previous.is_none_or(|previous| {
        value > previous || (value == previous && terminal && !current_terminal)
    })
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
fn role_line_sha256(domain: &[u8], line: &[BoardMove]) -> Result<[u8; 32], StoreError> {
    let mut digest = Sha256::new();
    digest.update(b"rz-pals-role-ordered-move16/1\0");
    digest.update(
        u64::try_from(domain.len())
            .map_err(|_| StoreError::Capacity("role digest domain"))?
            .to_le_bytes(),
    );
    digest.update(domain);
    digest.update(
        u64::try_from(line.len())
            .map_err(|_| StoreError::Capacity("role digest line"))?
            .to_le_bytes(),
    );
    for &movement in line {
        digest.update(Move16::pack(movement)?.bits().to_le_bytes());
    }
    Ok(digest.finalize().into())
}

fn role_divergence_sha256(candidates: &[usize]) -> Result<[u8; 32], StoreError> {
    let mut digest = Sha256::new();
    digest.update(b"rz-pals-role-ordered-divergence/1\0");
    digest.update(
        u64::try_from(candidates.len())
            .map_err(|_| StoreError::Capacity("role divergence length"))?
            .to_le_bytes(),
    );
    for &candidate in candidates {
        digest.update(
            u64::try_from(candidate)
                .map_err(|_| StoreError::Capacity("role divergence index"))?
                .to_le_bytes(),
        );
    }
    Ok(digest.finalize().into())
}

fn stable_id(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;
    use crate::cpu_checker::{
        CheckerCapabilities, CheckerWork, ExternalAttemptEvidence, ExternalBound,
        ExternalCheckerIdentity, ExternalModelMetadata, ExternalRawScore,
        ExternalTrainingKnowledge, ExternalUciIdentity,
    };

    #[derive(Clone, Copy)]
    enum ForeignFixtureMode {
        Normal(i32),
        UnknownWork,
        Overshoot,
        MixedIdentity,
        Pending,
        BudgetRefused,
        BudgetRace,
        AdmissionFailure(&'static str, &'static str),
    }
    /// Typed in-process report fixture, not actual UCI/process evidence. The
    /// native process owner's separate executable fixtures cover that boundary.
    struct ForeignFixture {
        identity: CheckerIdentity,
        mode: ForeignFixtureMode,
        request: u64,
        attempt: Option<CheckerAttempt>,
    }
    impl ForeignFixture {
        fn new(mode: ForeignFixtureMode) -> Self {
            Self {
                identity: CheckerIdentity::ExternalUci(ExternalCheckerIdentity {
                    adapter_semantics: "explicit-foreign-report-fixture/v1".into(),
                    binary_sha256: "1".repeat(64),
                    launch_arguments_sha256: "2".repeat(64),
                    declared_name: "in-process typed fixture".into(),
                    declared_version: "1".into(),
                    declared_source: "test-only".into(),
                    declared_license: "MIT".into(),
                    options: Default::default(),
                    assets: Vec::new(),
                    model_metadata: ExternalModelMetadata {
                        weights_sha256: None,
                        training: ExternalTrainingKnowledge::Unknown,
                        declared_rights: None,
                        precision: None,
                    },
                }),
                mode,
                request: 0,
                attempt: None,
            }
        }
        fn run(
            &mut self,
            position: &Position,
            moves: Option<&[BoardMove]>,
            limits: CpuLimits,
        ) -> Result<CheckerReport, CheckerError> {
            self.attempt = None;
            if matches!(self.mode, ForeignFixtureMode::BudgetRace) {
                return Err(CheckerError::External {
                    stage: "admission",
                    code: "insufficient_stop_reserve",
                });
            }
            self.request += 1;
            let legal = position.legal_moves();
            let best_move = moves.unwrap_or(&legal).first().copied();
            let CheckerIdentity::ExternalUci(identity) = &self.identity else {
                unreachable!()
            };
            let mut identity = identity.clone();
            if matches!(self.mode, ForeignFixtureMode::MixedIdentity) {
                identity.binary_sha256 = "9".repeat(64);
            }
            let report = ExternalCheckerReport {
                identity,
                observed_uci: ExternalUciIdentity {
                    name: "typed fixture, process unobserved".into(),
                    author: None,
                },
                request_id: self.request,
                best_move: if matches!(self.mode, ForeignFixtureMode::Pending) {
                    None
                } else {
                    best_move
                },
                pv: best_move.into_iter().collect(),
                score: match self.mode {
                    ForeignFixtureMode::Normal(value) => ExternalRawScore::MateMoves(value),
                    _ => ExternalRawScore::Centipawns(900_000),
                },
                bound: ExternalBound::Lower,
                wdl_per_mille: Some([999, 1, 0]),
                perspective: position.side_to_move(),
                requested_depth: limits.max_depth,
                reported_depth: Some(0),
                seldepth: None,
                root_restricted: moves.is_some(),
                completion: if matches!(self.mode, ForeignFixtureMode::Pending) {
                    ExternalCompletion::Pending
                } else {
                    ExternalCompletion::BestMove
                },
                work: CheckerWork {
                    nodes: if matches!(self.mode, ForeignFixtureMode::UnknownWork) {
                        None
                    } else if matches!(self.mode, ForeignFixtureMode::Overshoot) {
                        Some(limits.max_nodes + 1)
                    } else {
                        Some(1)
                    },
                    qnodes: None,
                    tt_hits: None,
                },
                elapsed: Duration::ZERO,
            };
            self.attempt = Some(CheckerAttempt {
                work: report.work,
                elapsed: report.elapsed,
                external: Some(ExternalAttemptEvidence {
                    request_id: report.request_id,
                    partial_report: Some(report.clone()),
                    process: CheckerShutdown::default(),
                }),
            });
            if matches!(self.mode, ForeignFixtureMode::Overshoot) {
                Err(CheckerError::External {
                    stage: "fixture",
                    code: "observed_node_overshoot",
                })
            } else {
                Ok(CheckerReport::ExternalUci(report))
            }
        }
    }
    impl CpuChecker for ForeignFixture {
        fn identity(&self) -> &CheckerIdentity {
            &self.identity
        }
        fn conditions(&self) -> &str {
            "test-foreign-helper/v1;process-evidence=unobserved"
        }
        fn capabilities(&self) -> CheckerCapabilities {
            CheckerCapabilities {
                max_depth: 64,
                max_prefix_plies: 64,
                max_root_moves: 256,
                root_moves: true,
                divergence: true,
                resume: false,
                selective_search: None,
            }
        }
        fn preflight_task(&self, limits: CpuLimits) -> Result<(), CheckerError> {
            match self.mode {
                ForeignFixtureMode::BudgetRefused
                    if limits.deadline.is_some_and(|until| {
                        until.saturating_duration_since(Instant::now()) < Duration::from_secs(2)
                    }) =>
                {
                    Err(CheckerError::External {
                        stage: "admission",
                        code: "insufficient_stop_reserve",
                    })
                }
                ForeignFixtureMode::AdmissionFailure(stage, code) => {
                    Err(CheckerError::External { stage, code })
                }
                _ => Ok(()),
            }
        }
        fn last_attempt(&self) -> Option<&CheckerAttempt> {
            self.attempt.as_ref()
        }
        fn reset_attempt(&mut self) {
            self.attempt = None;
        }
        fn analyze(
            &mut self,
            position: &Position,
            limits: CpuLimits,
            _: &AtomicBool,
        ) -> Result<CheckerReport, CheckerError> {
            self.run(position, None, limits)
        }
        fn analyze_root_moves(
            &mut self,
            position: &Position,
            moves: &[BoardMove],
            limits: CpuLimits,
            _: &AtomicBool,
        ) -> Result<CheckerReport, CheckerError> {
            self.run(position, Some(moves), limits)
        }
        fn analyze_divergence(
            &mut self,
            position: &Position,
            prefix: &[BoardMove],
            limits: CpuLimits,
            _: &AtomicBool,
        ) -> Result<CheckerReport, CheckerError> {
            let mut target = position.clone();
            for movement in prefix {
                target
                    .make_move(*movement)
                    .map_err(|_| CheckerError::Invalid("fixture prefix"))?;
            }
            self.run(&target, None, limits)
        }
        fn new_game(&mut self, _: Instant, _: &AtomicBool) -> Result<(), CheckerError> {
            self.attempt = None;
            Ok(())
        }
        fn shutdown(&mut self, _: Instant) -> Result<CheckerShutdown, CheckerError> {
            Ok(CheckerShutdown::default())
        }
    }
    fn foreign_engine(mode: ForeignFixtureMode) -> PalsEngine<LegalOrderRoleMock> {
        PalsEngine::new_with_checker(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                cpu_nodes_per_task: 8,
                ..PalsConfig::default()
            },
            LegalOrderRoleMock,
            ForeignFixture::new(mode),
        )
        .unwrap()
    }

    #[test]
    fn foreign_short_clock_preflight_keeps_last_valid_move_without_dispatch_or_reservation() {
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let mut engine = foreign_engine(ForeignFixtureMode::BudgetRefused);
        // Seed the consumer's legitimate Rules-validated fallback. The first
        // nonterminal proposal reaches foreign admission before any estimate
        // can replace it or manufacture an external/model work observation.
        let prior_choice = position.legal_moves()[0];
        let mut published = vec![prior_choice];
        let report = engine
            .search_with_progress(
                &position,
                PalsLimits {
                    deadline: Instant::now() + Duration::from_secs(1),
                    ..limits()
                },
                &cancel,
                |movement| published.push(movement),
            )
            .unwrap();
        assert_eq!(report.completion, PalsCompletion::Deadline);
        assert_eq!(report.best_move, Some(prior_choice));
        assert_eq!(published, [prior_choice]);
        assert_eq!(report.resolved_value, PalsResolvedValue::Unknown);
        assert_eq!(report.value_scope, PalsValueScope::Unknown);
        assert_eq!(report.counters.proposals, 1);
        assert_eq!(report.counters.value_calls, 0);
        assert_eq!(report.counters.accepted_value_outputs, 0);
        assert_eq!(report.counters.external_checker_tasks, 0);
        assert_eq!(report.counters.external_checker_reports, 0);
        assert_eq!(report.counters.external_checker_node_budget_reserved, 0);
        assert_eq!(report.counters.external_checker_nodes_observed, 0);
        assert_eq!(report.counters.consumed_external_checker_tasks, 0);
        assert!(!report.counters.external_checker_work_incomplete);
        assert_eq!(report.counters.cpu_tasks_requested, 0);
        assert_eq!(report.counters.cpu_nodes, 0);
        assert!(engine.checker_attempts().is_empty());
        assert!(engine.stores.tasks.is_empty());
        assert_eq!(engine.consumer_id, 0);
        assert!(!cancel.load(Ordering::Acquire));
    }

    #[test]
    fn foreign_pre_go_budget_race_releases_unused_reservation_without_unknown_work() {
        let position = Position::startpos();
        let mut engine = foreign_engine(ForeignFixtureMode::BudgetRace);
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        let mut counters = PalsCounters::default();
        assert!(matches!(
            engine.external_candidate(
                root,
                root,
                &[],
                None,
                limits(),
                &AtomicBool::new(false),
                &mut counters,
            ),
            Err(PalsError::Role(RoleError::Deadline))
        ));
        assert_eq!(counters, PalsCounters::default());
        assert!(engine.checker_attempts().is_empty());
        assert!(engine.stores.observations.is_empty());
        assert_eq!(engine.stores.tasks.len(), 1);
        assert!(matches!(
            engine
                .stores
                .tasks
                .get(super::super::store::ExecutionId(0))
                .unwrap()
                .status,
            super::super::store::TaskStatus::Failed
        ));
    }

    #[test]
    fn foreign_other_preflight_failures_keep_exact_error_without_dispatch() {
        for (stage, code) in [
            ("admission", "other_refusal"),
            ("fixture", "insufficient_stop_reserve"),
        ] {
            let position = Position::startpos();
            let mut engine = foreign_engine(ForeignFixtureMode::AdmissionFailure(stage, code));
            engine
                .stores
                .focus_actual_moves(position.snapshot())
                .unwrap();
            let root = engine.intern(position).unwrap();
            let mut counters = PalsCounters::default();
            assert!(matches!(
                engine.external_candidate(
                    root,
                    root,
                    &[],
                    None,
                    limits(),
                    &AtomicBool::new(false),
                    &mut counters,
                ),
                Err(PalsError::Checker(CheckerError::External {
                    stage: actual_stage,
                    code: actual_code,
                })) if (actual_stage, actual_code) == (stage, code)
            ));
            assert_eq!(counters, PalsCounters::default());
            assert!(engine.stores.tasks.is_empty());
            assert!(engine.checker_attempts().is_empty());
        }
    }
    #[test]
    fn externally_mutated_own_namespace_cannot_reuse_evidence_or_dispatch_new_work() {
        struct MutableNamespaceCpu {
            inner: CpuEngine,
            changed: std::sync::Arc<AtomicBool>,
            changed_value: CpuValueIdentity,
            calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
        }
        impl CpuSearcher for MutableNamespaceCpu {
            fn config(&self) -> &CpuConfig {
                self.inner.config()
            }
            fn value_identity(&self) -> &CpuValueIdentity {
                if self.changed.load(Ordering::Acquire) {
                    &self.changed_value
                } else {
                    self.inner.value_identity()
                }
            }
            fn search_identity(&self) -> &'static str {
                self.inner.search_identity()
            }
            fn search_conditions(&self) -> String {
                self.inner.search_conditions()
            }
            fn capabilities(&self) -> crate::cpu::CpuCapabilities {
                self.inner.capabilities()
            }
            fn last_attempt_work(&self) -> Option<crate::cpu::CpuWork> {
                self.inner.last_attempt_work()
            }
            fn clear(&mut self) {
                self.inner.clear();
            }
            fn analyze(
                &mut self,
                position: &Position,
                limits: CpuLimits,
                cancel: &AtomicBool,
            ) -> Result<CpuReport, CpuError> {
                self.calls.fetch_add(1, Ordering::AcqRel);
                self.inner.analyze(position, limits, cancel)
            }
        }
        let changed = std::sync::Arc::new(AtomicBool::new(false));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let inner = CpuEngine::new(CpuConfig::default()).unwrap();
        let mut changed_value = inner.value_identity().clone();
        changed_value.semantics = "externally-mutated-own-value/v1".into();
        let cpu = MutableNamespaceCpu {
            inner,
            changed: changed.clone(),
            changed_value,
            calls: calls.clone(),
        };
        let mut engine =
            PalsEngine::new_with_cpu(PalsConfig::default(), LegalOrderRoleMock, cpu).unwrap();
        let position = Position::startpos();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position.clone()).unwrap();
        let mut counters = PalsCounters::default();
        engine
            .verify(root, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        assert!(engine.nodes[root].evidence.is_some());
        assert_eq!(calls.load(Ordering::Acquire), 1);
        let before = counters;
        let consumer_before = engine.consumer_id;
        changed.store(true, Ordering::Release);
        assert!(matches!(
            engine.verify(root, &[], limits(), &AtomicBool::new(false), &mut counters),
            Err(PalsError::Store(StoreError::InvalidConditions(
                "frozen own value/search conditions changed"
            )))
        ));
        assert_eq!(counters, before);
        assert_eq!(engine.consumer_id, consumer_before);
        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert!(matches!(
            engine.search(&position, limits(), &AtomicBool::new(false)),
            Err(PalsError::Store(StoreError::InvalidConditions(_)))
        ));
        let rejected = engine.last_search_counters().unwrap();
        assert_eq!(rejected.cpu_tasks_requested, 0);
        assert_eq!(rejected.consumed_cached_cpu_values, 0);
        assert_eq!(rejected.cpu_nodes, 0);
        assert_eq!(calls.load(Ordering::Acquire), 1);
    }
    #[test]
    fn foreign_cp_mate_depth_and_wdl_stay_raw_while_model_drives_root_values() {
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let mut positive = foreign_engine(ForeignFixtureMode::Normal(200));
        let mut negative = foreign_engine(ForeignFixtureMode::Normal(-200));
        let first = positive.search(&position, limits(), &cancel).unwrap();
        let second = negative.search(&position, limits(), &cancel).unwrap();
        assert_eq!(first.best_move, second.best_move);
        assert_eq!(first.resolved_value, second.resolved_value);
        assert_eq!(first.score, None);
        assert_eq!(first.resolver_version, MODEL_WDL_RESOLVER_VERSION);
        assert!(matches!(
            first.resolved_value,
            PalsResolvedValue::ModelWdl { .. }
        ));
        assert!(first.counters.external_checker_tasks > 0);
        assert_eq!(first.counters.cpu_tasks_requested, 0);
        assert_eq!(first.counters.cpu_nodes, 0);
        assert!(first.counters.external_checker_work_incomplete);
        assert!(first.counters.accepted_value_outputs > 0);
        assert_eq!(
            first.counters.role_calls,
            first.counters.proposer_calls
                + first.counters.critic_calls
                + first.counters.repair_calls
                + first.counters.value_calls
        );
        assert_eq!(
            first.counters.consumed_role_outputs,
            first.counters.accepted_proposer_outputs
                + first.counters.accepted_critic_outputs
                + first.counters.accepted_repair_outputs
                + first.counters.accepted_value_outputs
        );
        for index in 0..positive.stores.observations.len() {
            let observation = positive
                .stores
                .observations
                .get(ObservationId(index))
                .unwrap();
            assert!(!matches!(observation.score, RawScore::Cpu { .. }));
            if let Some(report) = &observation.external_report {
                assert_eq!(report.score, ExternalRawScore::MateMoves(200));
                assert_eq!(report.reported_depth, Some(0));
                assert_eq!(report.bound, ExternalBound::Lower);
                assert_eq!(report.work.qnodes, None);
            }
        }
        assert!(
            positive
                .nodes
                .iter()
                .all(|node| node.evidence.is_none() && node.resume.is_none())
        );
    }
    #[test]
    fn foreign_unknown_pending_overshoot_and_namespace_failure_preserve_work_without_own_completion()
     {
        for mode in [
            ForeignFixtureMode::UnknownWork,
            ForeignFixtureMode::Pending,
            ForeignFixtureMode::Overshoot,
            ForeignFixtureMode::MixedIdentity,
        ] {
            let position = Position::startpos();
            let mut engine = foreign_engine(mode);
            engine
                .stores
                .focus_actual_moves(position.snapshot())
                .unwrap();
            let root = engine.intern(position).unwrap();
            let mut counters = PalsCounters::default();
            let result = engine.external_candidate(
                root,
                root,
                &[],
                None,
                limits(),
                &AtomicBool::new(false),
                &mut counters,
            );
            assert_eq!(counters.external_checker_tasks, 1);
            assert_eq!(counters.completed_cpu_tasks, 0);
            assert_eq!(counters.consumed_cpu_tasks, 0);
            assert_eq!(counters.consumed_external_checker_tasks, 0);
            assert_eq!(engine.checker_attempts().len(), 1);
            match mode {
                ForeignFixtureMode::Overshoot => {
                    assert!(result.is_err());
                    assert_eq!(counters.external_checker_nodes_observed, 9);
                    assert_eq!(counters.external_checker_node_budget_reserved, 8);
                }
                ForeignFixtureMode::MixedIdentity => {
                    assert!(result.is_err());
                    assert_eq!(counters.external_checker_nodes_observed, 1);
                }
                ForeignFixtureMode::UnknownWork => {
                    assert!(result.unwrap().is_some());
                    assert_eq!(engine.checker_attempts()[0].work.nodes, None);
                    assert!(counters.external_checker_work_incomplete);
                }
                ForeignFixtureMode::Pending => {
                    assert!(result.unwrap().is_none());
                }
                ForeignFixtureMode::Normal(_)
                | ForeignFixtureMode::BudgetRefused
                | ForeignFixtureMode::BudgetRace
                | ForeignFixtureMode::AdmissionFailure(_, _) => unreachable!(),
            }
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
        }
    }
    #[test]
    fn foreign_evidence_capacity_failure_closes_task_but_keeps_physical_attempt() {
        let position = Position::startpos();
        let mut engine = foreign_engine(ForeignFixtureMode::Normal(999));
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        engine.stores.observations = super::super::store::ObservationStore::new(0);
        let mut counters = PalsCounters::default();
        assert!(matches!(
            engine.external_candidate(
                root,
                root,
                &[],
                None,
                limits(),
                &AtomicBool::new(false),
                &mut counters
            ),
            Err(PalsError::Capacity)
        ));
        assert_eq!(engine.checker_attempts().len(), 1);
        assert_eq!(counters.external_checker_nodes_observed, 1);
        assert_eq!(counters.consumed_external_checker_tasks, 0);
        assert!(matches!(
            engine
                .stores
                .tasks
                .get(super::super::store::ExecutionId(0))
                .unwrap()
                .status,
            super::super::store::TaskStatus::Failed
        ));
    }
    #[test]
    fn model_value_final_cancel_never_acknowledges_output_or_assigns_evidence() {
        struct CancelValue;
        impl RoleModel for CancelValue {
            fn identity(&self) -> &str {
                LegalOrderRoleMock.identity()
            }
            fn value_identity(&self) -> Option<&ModelValueIdentity> {
                LegalOrderRoleMock.value_identity()
            }
            fn evaluate_value(
                &mut self,
                query: RoleQuery<'_>,
            ) -> Result<ModelValueOutput, RoleError> {
                let cancel = query.cancel;
                let output = LegalOrderRoleMock.evaluate_value(query)?;
                cancel.store(true, Ordering::Release);
                Ok(output)
            }
            fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                LegalOrderRoleMock.propose(query)
            }
            fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                LegalOrderRoleMock.reply(query)
            }
            fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                LegalOrderRoleMock.repair(query)
            }
            fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
                LegalOrderRoleMock.divergences(query)
            }
            fn accepted_output(&mut self) {
                panic!("canceled model value must not be acknowledged");
            }
        }
        let position = Position::startpos();
        let mut engine = PalsEngine::new_with_checker(
            PalsConfig::default(),
            CancelValue,
            ForeignFixture::new(ForeignFixtureMode::Normal(100)),
        )
        .unwrap();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        let mut counters = PalsCounters::default();
        assert!(matches!(
            engine.evaluate_model_value(
                root,
                &[],
                limits(),
                &AtomicBool::new(false),
                &mut counters
            ),
            Err(PalsError::Role(RoleError::Canceled))
        ));
        assert_eq!(counters.value_calls, 1);
        assert_eq!(counters.completed_value_calls, 1);
        assert_eq!(counters.accepted_value_outputs, 0);
        assert!(engine.nodes[root].model_value.is_none());
    }
    #[test]
    fn restricted_terminal_line_retains_actual_leaf_without_all_defenses_claim() {
        let position = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1").unwrap();
        let mut engine = foreign_engine(ForeignFixtureMode::Normal(-999));
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position.clone()).unwrap();
        let mut counters = PalsCounters::default();
        let (movement, child) = position
            .legal_moves()
            .into_iter()
            .find_map(|movement| {
                let mut child = position.clone();
                child.make_move(movement).unwrap();
                matches!(
                    child.classify_position().unwrap().play_status,
                    PlayStatus::Terminal {
                        reason: TerminalReason::Checkmate,
                        ..
                    }
                )
                .then_some((movement, child))
            })
            .unwrap();
        let leaf = engine
            .connect(root, movement, child, &mut counters)
            .unwrap();
        assert!(matches!(
            engine.rules_value(leaf),
            PalsResolvedValue::RulesTerminal {
                winner: Some(Color::White),
                ..
            }
        ));
        assert!(matches!(
            engine.model_value(root, 1),
            PalsResolvedValue::RestrictedRulesLine {
                winner: Some(Color::White),
                ..
            }
        ));
        assert!(engine.nodes[root].terminal.is_none());
        let result = engine
            .model_result(
                root,
                &position.legal_moves(),
                PalsCompletion::RoundLimit,
                Instant::now(),
                &mut counters,
            )
            .unwrap();
        assert_eq!(result.best_move, Some(movement));
        assert_eq!(result.score, None);
        assert_eq!(result.terminal, None);
        assert!(result.counters.unknown_root_children > 0);
        assert!(matches!(
            result.resolved_value,
            PalsResolvedValue::RulesTerminal {
                winner: Some(Color::White),
                ..
            }
        ));
    }

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

    #[derive(Default)]
    struct RecheckRoleFixture {
        replies: Vec<RoleLogicalContext>,
        cancel_after_reply: bool,
    }
    impl RoleModel for RecheckRoleFixture {
        fn identity(&self) -> &str {
            "post-repair-recheck-context-fixture/1"
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            LegalOrderRoleMock.propose(query)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let preferred = BoardMove::from_uci("g8f6").unwrap();
            let preferred_index = query
                .legal
                .iter()
                .position(|movement| *movement == preferred);
            let mut result = LegalOrderRoleMock.reply(query)?;
            if let Some(index) = preferred_index {
                result.logits[index] = 100.0;
            }
            Ok(result)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let preferred = BoardMove::from_uci("f1c4").unwrap();
            let preferred_index = query
                .legal
                .iter()
                .position(|movement| *movement == preferred);
            let mut result = LegalOrderRoleMock.repair(query)?;
            if let Some(index) = preferred_index {
                result.logits[index] = 100.0;
            }
            Ok(result)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            LegalOrderRoleMock.divergences(query)
        }
        fn reply_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            self.replies.push(context.clone());
            let cancel = query.cancel;
            let result = self.reply(query)?;
            if self.cancel_after_reply {
                cancel.store(true, Ordering::Release);
            }
            Ok(result)
        }
    }

    /// White-box evidence fixture, not an actual CPU/NN/process acceptance run.
    /// Completed cache values isolate conclusion scope from chess strength;
    /// the one-node case below exercises the real bounded CPU incomplete path.
    struct RecheckFixture {
        engine: PalsEngine<RecheckRoleFixture>,
        root: usize,
        leaf: usize,
        old_line: LineId,
        repaired_line: LineId,
        repair_record_revision: u64,
        repair_evidence: ObservationId,
        repaired: Vec<BoardMove>,
        refutation: Vec<BoardMove>,
        counters: PalsCounters,
    }
    impl RecheckFixture {
        fn run(&mut self, limits: PalsLimits, cancel: &AtomicBool) -> Result<(), PalsError> {
            self.engine.recheck_repaired_line(
                CompletedRepairRecheck {
                    root: self.root,
                    original_first: self.repaired[0],
                    attack_ply: 1,
                    repaired_line: self.repaired_line,
                    repair_record_revision: self.repair_record_revision,
                    repaired_leaf: self.leaf,
                    repaired: &self.repaired,
                    refutation: &self.refutation,
                    completed_value: 100,
                },
                limits,
                cancel,
                &mut self.counters,
                &mut |_| {},
            )
        }
    }
    fn recheck_fixture(
        policy: PostRepairRecheckPolicy,
        counter_complete: bool,
        incompatible_suffix: bool,
    ) -> RecheckFixture {
        fn moves(text: &[&str]) -> Vec<BoardMove> {
            text.iter()
                .map(|movement| BoardMove::from_uci(movement).unwrap())
                .collect()
        }
        let proposal = moves(&["e2e4", "c7c5", "g1f3", "d7d6", "d2d3", "b8c6"]);
        let (repaired, refutation, counter) = if incompatible_suffix {
            (
                moves(&["e2e4", "e7e5", "f1c4", "d7d5", "e4d5", "g8f6"]),
                moves(&["e2e4", "e7e5", "g1f3", "d7d5", "e4d5", "g8f6"]),
                moves(&["e2e4", "e7e5", "f1c4", "g8f6"]),
            )
        } else {
            (
                moves(&["e2e4", "e7e5", "f1c4", "b8c6", "d2d3", "d7d6"]),
                moves(&["e2e4", "e7e5", "g1f3", "b8c6", "d2d3", "d7d6"]),
                moves(&["e2e4", "e7e5", "f1c4", "g8f6", "d2d3", "d7d6"]),
            )
        };
        let mut engine = PalsEngine::new_with_cpu_and_refinement_policy(
            PalsConfig {
                line_plies: 6,
                ..PalsConfig::default()
            },
            RecheckRoleFixture::default(),
            CpuEngine::new(CpuConfig {
                max_depth: 4,
                tt_entries: 128,
                ..CpuConfig::default()
            })
            .unwrap(),
            policy,
        )
        .unwrap();
        let position = Position::startpos();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position.clone()).unwrap();
        let mut counters = PalsCounters::default();
        let mut path = vec![root];
        let mut node = root;
        for movement in &repaired {
            let mut child = engine.nodes[node].position.clone();
            child.make_move(*movement).unwrap();
            node = engine
                .connect(node, *movement, child, &mut counters)
                .unwrap();
            path.push(node);
        }
        let leaf = node;
        let identity = engine.owned_identity().unwrap().clone();
        engine.nodes[leaf].evidence = Some(CpuEvidence {
            score: 100,
            depth: 1,
            scope: CpuScoreScope::CompletedIteration,
            value_identity: identity.clone(),
        });
        if counter_complete {
            let mut counter_position = position;
            for movement in &counter {
                counter_position.make_move(*movement).unwrap();
            }
            // Intern the leaf without connecting its anchor edge, so C's
            // preferred alternative remains the first unexamined response.
            let counter_leaf = engine.intern(counter_position).unwrap();
            engine.nodes[counter_leaf].evidence = Some(CpuEvidence {
                score: -100,
                depth: 1,
                scope: CpuScoreScope::CompletedIteration,
                value_identity: identity,
            });
        }
        let (old_line, _) = engine
            .record(RecordKind::Proposal, &proposal, None, 0, None, None)
            .unwrap()
            .unwrap();
        let old_evidence = engine
            .conclusion_observation(root, old_line, ObservationKind::Refutation, 50, None)
            .unwrap();
        engine
            .stores
            .refute_continuation(engine.nodes[root].situation, old_line, old_evidence)
            .unwrap();
        let ranked = engine
            .ranked(
                path[2],
                Call::Repair,
                &repaired[..2],
                &proposal,
                Some(&refutation),
                limits(),
                &AtomicBool::new(false),
                &mut counters,
            )
            .unwrap();
        assert_eq!(ranked[0], repaired[2]);
        let (repaired_line, _) = engine
            .record(RecordKind::Repair, &repaired, None, 0, None, None)
            .unwrap()
            .unwrap();
        let repair_record_revision = engine.revision;
        let repair_evidence = engine
            .conclusion_observation(
                root,
                repaired_line,
                ObservationKind::Repair,
                100,
                Some(old_evidence),
            )
            .unwrap();
        engine
            .stores
            .repair(
                engine.nodes[root].situation,
                old_line,
                repaired_line,
                repair_evidence,
            )
            .unwrap();
        RecheckFixture {
            engine,
            root,
            leaf,
            old_line,
            repaired_line,
            repair_record_revision,
            repair_evidence,
            repaired,
            refutation,
            counters,
        }
    }

    #[test]
    fn post_repair_policy_is_constructor_selected_and_legacy_disabled() {
        assert_eq!(
            engine().post_repair_recheck_policy(),
            PostRepairRecheckPolicy::Disabled
        );
        assert_eq!(engine().search_identity(), PALS_SEARCH_VERSION);
        assert_eq!(engine().refinement_conditions(), None);
        let mut fixture = recheck_fixture(PostRepairRecheckPolicy::Disabled, true, false);
        fixture.run(limits(), &AtomicBool::new(false)).unwrap();
        assert!(fixture.engine.model.replies.is_empty());
        assert_eq!(fixture.counters.critic_calls, 0);
        assert_eq!(fixture.counters.supported_refutations, 0);
        let mut selected =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
        assert_eq!(
            selected.engine.search_identity(),
            POST_REPAIR_RECHECK_SEARCH_VERSION
        );
        assert_eq!(
            selected.engine.refinement_conditions(),
            Some(POST_REPAIR_RECHECK_CONDITIONS)
        );
        selected.engine.new_game();
        assert_eq!(
            selected.engine.post_repair_recheck_policy(),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1
        );
    }

    #[test]
    fn post_repair_one_reply_uses_revised_context_and_refutes_only_that_line() {
        use super::super::store::ContinuationStatus;
        let mut fixture =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
        fixture.run(limits(), &AtomicBool::new(false)).unwrap();
        assert_eq!(fixture.engine.model.replies.len(), 1);
        let context = &fixture.engine.model.replies[0];
        assert_eq!(context.purpose, RoleQueryPurpose::ReplyPolicy);
        assert_eq!(context.prefix, fixture.repaired[..3]);
        assert_eq!(
            context.proposal_sha256,
            role_line_sha256(b"proposal", &fixture.repaired).unwrap()
        );
        assert_eq!(
            context.refutation_sha256,
            Some(role_line_sha256(b"refutation", &fixture.refutation).unwrap())
        );
        assert!(context.public_revision >= fixture.repair_record_revision);
        assert_eq!(fixture.counters.critic_calls, 1);
        assert_eq!(fixture.counters.accepted_critic_outputs, 1);
        assert_eq!(fixture.counters.accepted_repair_outputs, 1);
        assert_eq!(fixture.counters.supported_refutations, 1);
        assert_eq!(fixture.counters.consumed_cached_cpu_values, 1);
        let situation = fixture
            .engine
            .stores
            .situations
            .get(fixture.engine.nodes[fixture.root].situation)
            .unwrap();
        assert!(situation.dirty);
        assert_eq!(
            situation.conclusions.get(fixture.old_line).unwrap().status,
            ContinuationStatus::RepairedBy(fixture.repaired_line)
        );
        let conclusion = situation.conclusions.get(fixture.repaired_line).unwrap();
        assert_eq!(conclusion.status, ContinuationStatus::Refuted);
        let evidence = fixture
            .engine
            .stores
            .observations
            .get(conclusion.evidence.unwrap())
            .unwrap();
        assert_eq!(evidence.supersedes, Some(fixture.repair_evidence));
        assert_eq!(
            evidence.source,
            stable_id(POST_REPAIR_RECHECK_SEARCH_VERSION)
        );
        assert!(matches!(evidence.scope, EvidenceScope::Model { .. }));
        assert!(
            fixture.engine.nodes[fixture.root]
                .edges
                .iter()
                .any(|edge| edge.movement == fixture.repaired[0])
        );
    }

    #[test]
    fn post_repair_illegal_suffix_preserves_provisional_counter_without_refutation() {
        use super::super::store::ContinuationStatus;
        let mut fixture =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, true);
        fixture.run(limits(), &AtomicBool::new(false)).unwrap();
        assert_eq!(fixture.engine.model.replies.len(), 1);
        assert_eq!(fixture.counters.supported_refutations, 0);
        assert_eq!(fixture.counters.consumed_cached_cpu_values, 1);
        let record = fixture
            .engine
            .records
            .iter()
            .rev()
            .find(|record| record.kind == RecordKind::Counterexample)
            .unwrap();
        assert_eq!(record.line.len(), 4);
        assert_eq!(record.line[0], fixture.repaired[0]);
        assert_eq!(record.value, None);
        assert_eq!(
            fixture
                .engine
                .stores
                .situations
                .get(fixture.engine.nodes[fixture.root].situation)
                .unwrap()
                .conclusions
                .get(fixture.repaired_line)
                .unwrap()
                .status,
            ContinuationStatus::Supported
        );
    }

    #[test]
    fn post_repair_incomplete_cpu_and_exhausted_or_canceled_controls_do_not_refute() {
        use super::super::store::ContinuationStatus;
        let mut fixture = recheck_fixture(
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
            false,
            false,
        );
        fixture
            .run(
                PalsLimits {
                    max_cpu_nodes: 1,
                    ..limits()
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(fixture.engine.model.replies.len(), 1);
        assert!(fixture.counters.cpu_nodes <= 1);
        assert_eq!(fixture.counters.supported_refutations, 0);
        assert_eq!(
            fixture
                .engine
                .stores
                .situations
                .get(fixture.engine.nodes[fixture.root].situation)
                .unwrap()
                .conclusions
                .get(fixture.repaired_line)
                .unwrap()
                .status,
            ContinuationStatus::Supported
        );
        let mut exhausted =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
        exhausted.engine.config.max_role_calls = exhausted.counters.role_calls;
        assert!(matches!(
            exhausted.run(limits(), &AtomicBool::new(false)),
            Err(PalsError::RoleCallLimit)
        ));
        assert!(exhausted.engine.model.replies.is_empty());
        let mut cpu_exhausted =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
        cpu_exhausted.counters.cpu_nodes = limits().max_cpu_nodes;
        cpu_exhausted
            .run(limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(cpu_exhausted.engine.model.replies.is_empty());
        assert_eq!(cpu_exhausted.counters.supported_refutations, 0);
        let mut canceled =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
        assert!(matches!(
            canceled.run(limits(), &AtomicBool::new(true)),
            Err(PalsError::Role(RoleError::Canceled))
        ));
        assert!(canceled.engine.model.replies.is_empty());
    }

    #[test]
    fn post_repair_different_completed_depth_is_unresolved_and_foreign_lane_is_unsupported() {
        use super::super::store::ContinuationStatus;
        let mut fixture =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
        for node in &mut fixture.engine.nodes {
            if let Some(evidence) = node
                .evidence
                .as_mut()
                .filter(|evidence| evidence.score == -100)
            {
                evidence.depth = 2;
            }
        }
        fixture.run(limits(), &AtomicBool::new(false)).unwrap();
        assert_eq!(fixture.counters.consumed_cached_cpu_values, 1);
        assert_eq!(fixture.counters.supported_refutations, 0);
        assert_eq!(
            fixture
                .engine
                .stores
                .situations
                .get(fixture.engine.nodes[fixture.root].situation)
                .unwrap()
                .conclusions
                .get(fixture.repaired_line)
                .unwrap()
                .status,
            ContinuationStatus::Supported,
        );
        assert!(matches!(
            PalsEngine::new_with_checker_and_refinement_policy(
                PalsConfig::default(),
                RecheckRoleFixture::default(),
                ForeignFixture::new(ForeignFixtureMode::Normal(1)),
                PostRepairRecheckPolicy::SameRepairedLineOnceV1,
            ),
            Err(PalsError::Cpu(CpuError::Unsupported(_)))
        ));
        let mut stale =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
        stale.repair_record_revision += 1;
        assert!(matches!(
            stale.run(limits(), &AtomicBool::new(false)),
            Err(PalsError::Store(StoreError::InvalidEvidence(_)))
        ));
        assert!(stale.engine.model.replies.is_empty());
    }

    #[test]
    fn post_repair_late_reply_is_completed_but_not_accepted_or_consumed() {
        let mut fixture =
            recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
        fixture.engine.model.cancel_after_reply = true;
        let consumed_before = fixture.counters.consumed_role_outputs;
        assert!(matches!(
            fixture.run(limits(), &AtomicBool::new(false)),
            Err(PalsError::Role(RoleError::Canceled))
        ));
        assert_eq!(fixture.counters.critic_calls, 1);
        assert_eq!(fixture.counters.completed_critic_calls, 1);
        assert_eq!(fixture.counters.accepted_critic_outputs, 0);
        assert_eq!(fixture.counters.consumed_role_outputs, consumed_before);
        assert_eq!(fixture.counters.supported_refutations, 0);
        assert!(
            !fixture
                .engine
                .records
                .iter()
                .any(|record| record.kind == RecordKind::Counterexample)
        );
    }

    #[test]
    fn post_repair_final_refutation_boundary_rejects_late_cancel_and_deadline() {
        use super::super::store::ContinuationStatus;
        for canceled in [true, false] {
            let mut fixture =
                recheck_fixture(PostRepairRecheckPolicy::SameRepairedLineOnceV1, true, false);
            let root_situation = fixture.engine.nodes[fixture.root].situation;
            let before = fixture
                .engine
                .stores
                .situations
                .get(root_situation)
                .unwrap();
            let revision_before = before.revision;
            let dirty_before = before.dirty;
            let observations_before = fixture.engine.stores.observations.len();
            let records_before = fixture.engine.records.len();
            let mut final_limits = limits();
            if !canceled {
                final_limits.deadline = Instant::now();
            }
            // Model a stop arriving after the earlier publish_choice and evidence
            // comparisons: invoke the actual final publication boundary directly.
            let result = fixture.engine.publish_recheck_refutation(
                &CompletedRepairRecheck {
                    root: fixture.root,
                    original_first: fixture.repaired[0],
                    attack_ply: 1,
                    repaired_line: fixture.repaired_line,
                    repair_record_revision: fixture.repair_record_revision,
                    repaired_leaf: fixture.leaf,
                    repaired: &fixture.repaired,
                    refutation: &fixture.refutation,
                    completed_value: 100,
                },
                -100,
                Some(fixture.repair_evidence),
                final_limits,
                &AtomicBool::new(canceled),
                &mut fixture.counters,
            );
            assert!(matches!(
                (canceled, result),
                (true, Err(PalsError::Role(RoleError::Canceled)))
                    | (false, Err(PalsError::Role(RoleError::Deadline)))
            ));
            let after = fixture
                .engine
                .stores
                .situations
                .get(root_situation)
                .unwrap();
            let conclusion = after.conclusions.get(fixture.repaired_line).unwrap();
            assert_eq!(after.revision, revision_before);
            assert_eq!(after.dirty, dirty_before);
            assert_eq!(conclusion.status, ContinuationStatus::Supported);
            assert_eq!(conclusion.evidence, Some(fixture.repair_evidence));
            assert_eq!(
                fixture.engine.stores.observations.len(),
                observations_before
            );
            assert_eq!(fixture.engine.records.len(), records_before);
            assert_eq!(fixture.counters.supported_refutations, 0);
        }
    }

    #[derive(Default)]
    struct LogicalProbe {
        pending: Option<(RoleLogicalContext, PositionSnapshot, Instant, usize)>,
        accepted: Vec<RoleLogicalContext>,
        failure: Option<RoleError>,
        resets: Vec<Option<u64>>,
    }
    impl LogicalProbe {
        fn prepare(
            &mut self,
            position: &Position,
            context: &RoleLogicalContext,
            deadline: Instant,
            cancel: &AtomicBool,
        ) {
            assert!(self.pending.is_none());
            self.pending = Some((
                context.clone(),
                position.snapshot(),
                deadline,
                cancel as *const AtomicBool as usize,
            ));
        }
    }
    impl RoleModel for LogicalProbe {
        fn identity(&self) -> &str {
            LegalOrderRoleMock.identity()
        }
        fn value_identity(&self) -> Option<&ModelValueIdentity> {
            LegalOrderRoleMock.value_identity()
        }
        fn propose(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(RoleError::Backend("contextless test dispatch".into()))
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.propose(query)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.propose(query)
        }
        fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            Err(RoleError::Backend("contextless test divergence".into()))
        }
        fn propose_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            assert_eq!(context.purpose, RoleQueryPurpose::ProposePolicy);
            self.prepare(query.position, context, query.deadline, query.cancel);
            LegalOrderRoleMock::evaluate(query)
        }
        fn reply_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            assert_eq!(context.purpose, RoleQueryPurpose::ReplyPolicy);
            self.prepare(query.position, context, query.deadline, query.cancel);
            LegalOrderRoleMock::evaluate(query)
        }
        fn repair_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            assert_eq!(context.purpose, RoleQueryPurpose::RepairPolicy);
            self.prepare(query.position, context, query.deadline, query.cancel);
            LegalOrderRoleMock::evaluate(query)
        }
        fn divergences_with_context(
            &mut self,
            query: DivergenceQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<Vec<f32>, RoleError> {
            assert_eq!(context.purpose, RoleQueryPurpose::DivergencePolicy);
            self.prepare(query.root, context, query.deadline, query.cancel);
            LegalOrderRoleMock.divergences(query)
        }
        fn evaluate_value_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<ModelValueOutput, RoleError> {
            assert_eq!(context.purpose, RoleQueryPurpose::ValueFresh);
            self.prepare(query.position, context, query.deadline, query.cancel);
            LegalOrderRoleMock.evaluate_value(query)
        }
        fn accepted_output_checked(
            &mut self,
            acceptance: RoleAcceptance<'_>,
        ) -> Result<(), RoleError> {
            acceptance.check_control()?;
            let (context, snapshot, deadline, cancel) =
                self.pending.as_ref().ok_or(RoleError::InvalidOutput)?;
            assert_eq!(context, acceptance.context);
            assert!(snapshot.same_state(acceptance.snapshot));
            assert_eq!(*deadline, acceptance.deadline);
            assert_eq!(*cancel, acceptance.cancel as *const AtomicBool as usize);
            if let Some(error) = &self.failure {
                return Err(error.clone());
            }
            self.accepted.push(context.clone());
            self.pending = None;
            Ok(())
        }
        fn new_game_with_generation(&mut self, generation: Option<u64>) {
            self.pending = None;
            self.resets.push(generation);
        }
        fn finish_search(&mut self, _: RoleSearchClosure) {
            self.pending = None;
        }
    }
    fn logical_probe() -> PalsEngine<Box<LogicalProbe>> {
        PalsEngine::new(
            PalsConfig::default(),
            Box::new(LogicalProbe::default()),
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn logical_context_digests_preserve_order_length_domain_and_promotions() {
        let first = BoardMove::from_uci("e2e4").unwrap();
        let second = BoardMove::from_uci("e7e5").unwrap();
        assert_ne!(
            role_line_sha256(b"prefix", &[first, second]).unwrap(),
            role_line_sha256(b"prefix", &[second, first]).unwrap()
        );
        assert_ne!(
            role_line_sha256(b"prefix", &[first]).unwrap(),
            role_line_sha256(b"proposal", &[first]).unwrap()
        );
        assert_ne!(
            role_line_sha256(b"prefix", &[first]).unwrap(),
            role_line_sha256(b"prefix", &[first, second]).unwrap()
        );
        let promotions: HashSet<_> = ["a7a8q", "a7a8r", "a7a8b", "a7a8n"]
            .into_iter()
            .map(|movement| {
                role_line_sha256(b"proposal", &[BoardMove::from_uci(movement).unwrap()]).unwrap()
            })
            .collect();
        assert_eq!(promotions.len(), 4);
        assert_ne!(
            role_divergence_sha256(&[1, 3]).unwrap(),
            role_divergence_sha256(&[3, 1]).unwrap()
        );
    }

    #[test]
    fn logical_context_uses_current_checked_store_and_separates_record_metadata() {
        let mut engine = engine();
        let position = Position::startpos();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        let prefix = [BoardMove::from_uci("e2e4").unwrap()];
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::ProposePolicy,
            prefix: &prefix,
            proposal: &prefix,
            refutation: None,
            divergences: &[],
        };
        let original = engine.role_context(root, question).unwrap();
        assert_eq!(original.situation, engine.nodes[root].situation);
        assert_eq!(original.state, engine.nodes[root].state);
        assert_eq!(original.search_generation, engine.stores.generation());
        assert_eq!(original.prefix, prefix);
        engine.revision += 1;
        engine
            .stores
            .situations
            .get_mut(original.situation)
            .unwrap()
            .revision += 1;
        let revised = engine.role_context(root, question).unwrap();
        assert_ne!(original, revised);
        assert_eq!(
            original.focus_prefix_sha256(),
            revised.focus_prefix_sha256()
        );
        assert_eq!(original.proposal_sha256, revised.proposal_sha256);
        let empty_refutation = engine
            .role_context(
                root,
                RoleQuestion {
                    refutation: Some(&[]),
                    ..question
                },
            )
            .unwrap();
        assert_ne!(
            revised.refutation_sha256,
            empty_refutation.refutation_sha256
        );
        engine.stores.situations.remove(original.situation).unwrap();
        assert!(matches!(
            engine.role_context(root, question),
            Err(PalsError::Store(StoreError::InvalidHandle(_)))
        ));
    }

    #[test]
    fn boxed_contextual_dispatch_checks_original_controls_and_counts_only_acceptance() {
        let mut engine = logical_probe();
        let report = engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(
            engine
                .model
                .accepted
                .iter()
                .any(|c| c.purpose == RoleQueryPurpose::DivergencePolicy)
        );
        assert_eq!(
            report.counters.consumed_role_outputs as usize,
            engine.model.accepted.len()
        );
        assert!(
            engine
                .model
                .accepted
                .iter()
                .all(|c| c.game_generation == 0 && c.search_generation == 1)
        );
        assert!(engine.model.pending.is_none());
    }

    #[test]
    fn checked_acceptance_failure_is_propagated_without_consumption() {
        let mut engine = logical_probe();
        engine.model.failure = Some(RoleError::Backend("provisional commit rejected".into()));
        assert!(matches!(
            engine.search(&Position::startpos(), limits(), &AtomicBool::new(false)),
            Err(PalsError::Role(RoleError::Backend(message))) if message == "provisional commit rejected"
        ));
        let counters = engine.last_search_counters().unwrap();
        assert_eq!(counters.completed_proposer_calls, 1);
        assert_eq!(counters.consumed_role_outputs, 0);
        assert_eq!(counters.accepted_proposer_outputs, 0);
        assert!(engine.model.accepted.is_empty());
        assert!(engine.model.pending.is_none());
    }

    #[test]
    fn logical_acceptance_rejects_new_focus_and_changed_records_before_provider_commit() {
        let mut engine = logical_probe();
        let position = Position::startpos();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position.clone()).unwrap();
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::ProposePolicy,
            prefix: &[],
            proposal: &[],
            refutation: None,
            divergences: &[],
        };
        let control = limits();
        let cancel = AtomicBool::new(false);
        let prepared = engine.role_context(root, question).unwrap();
        engine
            .model
            .prepare(&position, &prepared, control.deadline, &cancel);
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        assert!(matches!(
            engine.accept_role_output(root, &prepared, question, control, &cancel),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        assert!(engine.model.accepted.is_empty());
        engine.model.pending = None;
        let current = engine.role_context(root, question).unwrap();
        engine
            .model
            .prepare(&position, &current, control.deadline, &cancel);
        engine.revision += 1;
        assert!(matches!(
            engine.accept_role_output(root, &current, question, control, &cancel),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        assert!(engine.model.accepted.is_empty());
    }

    #[test]
    fn checked_acceptance_keeps_deadline_and_cancellation_live_after_delivery() {
        let mut engine = logical_probe();
        let position = Position::startpos();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position.clone()).unwrap();
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::ProposePolicy,
            prefix: &[],
            proposal: &[],
            refutation: None,
            divergences: &[],
        };
        let context = engine.role_context(root, question).unwrap();
        let cancel = AtomicBool::new(false);
        let control = limits();
        engine
            .model
            .prepare(&position, &context, control.deadline, &cancel);
        cancel.store(true, Ordering::Release);
        assert!(matches!(
            engine.accept_role_output(root, &context, question, control, &cancel),
            Err(PalsError::Role(RoleError::Canceled))
        ));
        assert!(engine.model.accepted.is_empty());
        engine.model.pending = None;
        cancel.store(false, Ordering::Release);
        let expired = PalsLimits {
            deadline: Instant::now(),
            ..control
        };
        engine
            .model
            .prepare(&position, &context, expired.deadline, &cancel);
        assert!(matches!(
            engine.accept_role_output(root, &context, question, expired, &cancel),
            Err(PalsError::Role(RoleError::Deadline))
        ));
        assert!(engine.model.accepted.is_empty());
    }

    #[test]
    fn value_dispatch_and_acceptance_are_always_explicitly_fresh() {
        let mut engine = PalsEngine::new_with_checker(
            PalsConfig::default(),
            Box::new(LogicalProbe::default()),
            ForeignFixture::new(ForeignFixtureMode::Normal(0)),
        )
        .unwrap();
        let position = Position::startpos();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        let mut counters = PalsCounters::default();
        engine
            .evaluate_model_value(root, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        assert_eq!(engine.model.accepted.len(), 1);
        assert_eq!(
            engine.model.accepted[0].purpose,
            RoleQueryPurpose::ValueFresh
        );
        assert_eq!(counters.accepted_value_outputs, 1);
        assert_eq!(counters.proposer_calls, 0);
        assert_eq!(counters.consumed_role_outputs, 1);
    }

    #[test]
    fn logical_game_reset_and_generation_exhaustion_cannot_reissue_authority() {
        let mut engine = logical_probe();
        engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap();
        let old = engine.model.accepted[0].clone();
        engine.new_game();
        engine
            .search(&Position::startpos(), limits(), &AtomicBool::new(false))
            .unwrap();
        let new = engine
            .model
            .accepted
            .iter()
            .find(|c| c.game_generation == 1)
            .unwrap();
        assert_eq!(old.situation, new.situation);
        assert_eq!(old.search_generation, new.search_generation);
        assert_ne!(old.game_generation, new.game_generation);
        assert_eq!(engine.model.resets, [Some(1)]);
        engine.game_generation = Some(u64::MAX);
        engine.new_game();
        assert_eq!(engine.game_generation, None);
        assert_eq!(engine.model.resets.last(), Some(&None));
        assert!(matches!(
            engine.search(&Position::startpos(), limits(), &AtomicBool::new(false)),
            Err(PalsError::Capacity)
        ));
        engine.new_game();
        assert_eq!(engine.game_generation, None);
    }
    #[test]
    fn boxed_role_accounting_closes_each_search_once_without_claiming_physical_completion() {
        struct FinishRole {
            failure: Option<RoleError>,
            closed: std::sync::Arc<std::sync::Mutex<Vec<RoleSearchClosure>>>,
        }
        impl RoleModel for FinishRole {
            fn identity(&self) -> &str {
                "search-closure-test"
            }
            fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                if let Some(error) = &self.failure {
                    return Err(error.clone());
                }
                LegalOrderRoleMock.propose(query)
            }
            fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                LegalOrderRoleMock.reply(query)
            }
            fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                LegalOrderRoleMock.repair(query)
            }
            fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
                LegalOrderRoleMock.divergences(query)
            }
            fn finish_search(&mut self, reason: RoleSearchClosure) {
                self.closed.lock().unwrap().push(reason);
            }
        }
        for (failure, cancel, expired, expected) in [
            (None, false, false, RoleSearchClosure::Completed),
            (None, true, false, RoleSearchClosure::Canceled),
            (None, false, true, RoleSearchClosure::Deadline),
            (
                Some(RoleError::Unavailable),
                false,
                false,
                RoleSearchClosure::Failed,
            ),
            (
                Some(RoleError::PhysicalCompletionUnknown),
                false,
                false,
                RoleSearchClosure::PhysicalCompletionUnknown,
            ),
        ] {
            let closed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let role: Box<dyn RoleModel> = Box::new(FinishRole {
                failure,
                closed: closed.clone(),
            });
            let mut engine = PalsEngine::new(
                PalsConfig::default(),
                role,
                CpuEngine::new(CpuConfig::default()).unwrap(),
            )
            .unwrap();
            let mut run_limits = limits();
            if expired {
                run_limits.deadline = Instant::now();
            }
            let _ = engine.search(&Position::startpos(), run_limits, &AtomicBool::new(cancel));
            assert_eq!(*closed.lock().unwrap(), vec![expected]);
            engine.new_game();
            assert_eq!(closed.lock().unwrap().len(), 1);
        }
    }
    #[test]
    fn independent_cpu_defense_enters_graph_and_affected_critic_and_repair_with_exact_history() {
        #[derive(Default)]
        struct Seen {
            critic_saw_cpu: bool,
            repairs: Vec<(
                Vec<BoardMove>,
                Vec<BoardMove>,
                rz_position::PositionSnapshot,
            )>,
        }
        struct BlindRole(std::sync::Arc<std::sync::Mutex<Seen>>);
        impl BlindRole {
            fn rank(query: RoleQuery<'_>, repair: bool) -> Result<RoleEvaluation, RoleError> {
                query.check_control()?;
                let preferred = if repair {
                    &["h3h4", "a1b1", "h8h7"][..]
                } else {
                    &["h2h3", "h8h7", "a1b1", "h7h8"][..]
                };
                let chosen = preferred
                    .iter()
                    .filter_map(|text| BoardMove::from_uci(text).ok())
                    .find(|movement| query.legal.contains(movement));
                Ok(RoleEvaluation {
                    logits: query
                        .legal
                        .iter()
                        .map(|movement| if Some(*movement) == chosen { 10.0 } else { 0.0 })
                        .collect(),
                    wdl: [0.3, 0.4, 0.3],
                })
            }
        }
        impl RoleModel for BlindRole {
            fn identity(&self) -> &str {
                "blind-to-rook-capture-test"
            }
            fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                Self::rank(query, false)
            }
            fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                self.0.lock().unwrap().critic_saw_cpu |= query
                    .records
                    .iter()
                    .any(|record| record.cpu_observation.is_some());
                Self::rank(query, false)
            }
            fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                self.0.lock().unwrap().repairs.push((
                    query.prefix.to_vec(),
                    query.counterexample.unwrap().to_vec(),
                    query.position.snapshot(),
                ));
                Self::rank(query, true)
            }
            fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
                if query.cancel.load(Ordering::Acquire) {
                    return Err(RoleError::Canceled);
                }
                if Instant::now() >= query.deadline {
                    return Err(RoleError::Deadline);
                }
                Ok(query.candidates.iter().map(|ply| -(*ply as f32)).collect())
            }
        }
        let start = Position::from_fen("2r4k/8/8/8/2Q5/8/7P/K7 w - - 0 1").unwrap();
        let first = BoardMove::from_uci("h2h3").unwrap();
        let defense = BoardMove::from_uci("c8c4").unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Seen::default()));
        let mut engine = PalsEngine::new(
            PalsConfig {
                beam_width: 1,
                ..PalsConfig::default()
            },
            BlindRole(seen.clone()),
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap();
        let result = engine
            .search(&start, limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(result.counters.cpu_nodes <= limits().max_cpu_nodes);
        let candidate = engine
            .records()
            .iter()
            .find(|record| {
                record.kind == RecordKind::Counterexample && record.cpu_observation.is_some()
            })
            .unwrap();
        assert_eq!(&candidate.line[..2], &[first, defense]);
        assert_eq!(candidate.value, None);
        let observation = engine
            .stores
            .observations
            .get(candidate.cpu_observation.unwrap())
            .unwrap();
        assert_eq!(observation.kind, ObservationKind::CpuAnalysis);
        assert_eq!(
            observation.value_identity.as_ref(),
            Some(engine.owned_identity().unwrap())
        );
        assert!(matches!(
            observation.scope,
            EvidenceScope::DepthLimited { .. }
        ));
        let mut divergence = start.clone();
        divergence.make_move(first).unwrap();
        assert_eq!(
            engine
                .stores
                .states
                .get(observation.state)
                .unwrap()
                .position_identity(),
            divergence.position_identity()
        );
        assert_eq!(
            engine
                .stores
                .lines
                .moves(observation.cpu_pv.unwrap())
                .unwrap()[0],
            defense
        );
        let seen = seen.lock().unwrap();
        assert!(seen.critic_saw_cpu);
        let repaired = seen
            .repairs
            .iter()
            .find(|(prefix, counterexample, _)| {
                prefix.starts_with(&[first, defense])
                    && counterexample.starts_with(&[first, defense])
            })
            .unwrap();
        let mut repair_state = start.clone();
        for movement in &repaired.0 {
            repair_state.make_move(*movement).unwrap();
        }
        assert_eq!(
            repair_state.position_identity(),
            repaired.2.position_identity()
        );
        for record in engine.records().iter().filter(|record| {
            record.kind == RecordKind::Counterexample || record.kind == RecordKind::Repair
        }) {
            assert_eq!(record.line.first().copied(), Some(first));
            let mut checked = start.clone();
            for &movement in &record.line {
                checked.make_move(movement).unwrap();
            }
            assert!(
                engine
                    .nodes
                    .iter()
                    .any(|node| node.position.position_identity() == checked.position_identity())
            );
        }
    }

    #[test]
    fn cpu_candidate_reuse_preserves_pv_and_restricted_query_namespace() {
        let position = Position::from_fen("2r4k/8/8/8/2Q5/8/7P/K7 b - - 0 1").unwrap();
        let mut engine = engine();
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position.clone()).unwrap();
        let mut first = PalsCounters::default();
        let candidate = engine
            .cpu_candidate(
                root,
                root,
                &[],
                None,
                limits(),
                &AtomicBool::new(false),
                &mut first,
            )
            .unwrap()
            .unwrap();
        assert_eq!(candidate.pv[0], BoardMove::from_uci("c8c4").unwrap());
        assert!(engine.nodes[root].evidence.is_none());
        let mut reused = PalsCounters::default();
        let cached = engine
            .cpu_candidate(
                root,
                root,
                &[],
                None,
                limits(),
                &AtomicBool::new(false),
                &mut reused,
            )
            .unwrap()
            .unwrap();
        assert_eq!(cached.pv, candidate.pv);
        assert_eq!(reused.cpu_tasks_requested, 0);
        assert_eq!(reused.reused_completed_cpu_tasks_consumed, 1);
        let alternatives: Vec<_> = position
            .legal_moves()
            .into_iter()
            .filter(|movement| *movement != candidate.pv[0])
            .collect();
        let mut restricted = PalsCounters::default();
        let other = engine
            .cpu_candidate(
                root,
                root,
                &[],
                Some(&alternatives),
                limits(),
                &AtomicBool::new(false),
                &mut restricted,
            )
            .unwrap()
            .unwrap();
        assert_ne!(other.pv[0], candidate.pv[0]);
        assert_eq!(restricted.cpu_tasks_requested, 1);
        assert_eq!(restricted.reused_completed_cpu_tasks_consumed, 0);
        assert!(engine.nodes[root].evidence.is_none());
        assert!(matches!(
            engine
                .stores
                .tasks
                .get(
                    engine
                        .stores
                        .observations
                        .get(other.observation)
                        .unwrap()
                        .execution
                        .unwrap()
                )
                .unwrap()
                .key
                .question,
            TaskQuestion::AnalyzeRootMoves
        ));
    }

    #[test]
    fn cpu_candidate_rejects_wrong_prefix_cancel_and_old_root_and_distinguishes_unknown_history() {
        let start = Position::from_fen("2r4k/8/8/8/2Q5/8/7P/K7 w - - 0 1").unwrap();
        let movement = BoardMove::from_uci("h2h3").unwrap();
        let mut actual = start.clone();
        actual.make_move(movement).unwrap();
        let mut engine = engine();
        engine.stores.focus_actual_moves(start.snapshot()).unwrap();
        let root = engine.intern(start.clone()).unwrap();
        let divergence = engine.intern(actual.clone()).unwrap();
        let mut counters = PalsCounters::default();
        assert!(matches!(
            engine.cpu_candidate(
                root,
                divergence,
                &[BoardMove::from_uci("h2h4").unwrap()],
                None,
                limits(),
                &AtomicBool::new(false),
                &mut counters
            ),
            Err(PalsError::Store(StoreError::StaleConsumer))
        ));
        assert_eq!(counters.cpu_tasks_requested, 0);
        let mut candidate = engine
            .cpu_candidate(
                root,
                divergence,
                &[movement],
                None,
                limits(),
                &AtomicBool::new(false),
                &mut counters,
            )
            .unwrap()
            .unwrap();
        let proposal = vec![movement, BoardMove::from_uci("h8h7").unwrap()];
        let before = engine.nodes.len();
        assert!(
            engine
                .insert_cpu_candidate(
                    root,
                    divergence,
                    &[movement],
                    &proposal,
                    &candidate,
                    limits(),
                    &AtomicBool::new(true),
                    &mut counters
                )
                .unwrap()
                .is_none()
        );
        assert_eq!(engine.nodes.len(), before);
        let original_deadline = candidate.deadline;
        candidate.deadline = Instant::now();
        assert!(
            engine
                .insert_cpu_candidate(
                    root,
                    divergence,
                    &[movement],
                    &proposal,
                    &candidate,
                    limits(),
                    &AtomicBool::new(false),
                    &mut counters
                )
                .unwrap()
                .is_none()
        );
        candidate.deadline = original_deadline;
        assert_eq!(engine.nodes.len(), before);
        engine.stores.focus_actual_moves(start.snapshot()).unwrap();
        assert!(matches!(
            engine.insert_cpu_candidate(
                root,
                divergence,
                &[movement],
                &proposal,
                &candidate,
                limits(),
                &AtomicBool::new(false),
                &mut counters
            ),
            Err(PalsError::Store(StoreError::StaleConsumer))
        ));
        assert_eq!(engine.nodes.len(), before);
        engine.stores.focus_actual_moves(actual.snapshot()).unwrap();
        assert!(matches!(
            engine.insert_cpu_candidate(
                root,
                divergence,
                &[movement],
                &proposal,
                &candidate,
                limits(),
                &AtomicBool::new(false),
                &mut counters
            ),
            Err(PalsError::Store(StoreError::StaleConsumer))
        ));
        assert_eq!(engine.nodes.len(), before);
        // The same FEN loses earlier known repetition history and is another
        // exact task state; it cannot reuse the old discovery execution/PV.
        let without_history = Position::from_fen(&actual.to_fen()).unwrap();
        assert_ne!(
            actual.position_identity(),
            without_history.position_identity()
        );
        engine
            .stores
            .focus_actual_moves(without_history.snapshot())
            .unwrap();
        let new_root = engine.intern(without_history).unwrap();
        let mut fresh = PalsCounters::default();
        engine
            .cpu_candidate(
                new_root,
                new_root,
                &[],
                None,
                limits(),
                &AtomicBool::new(false),
                &mut fresh,
            )
            .unwrap()
            .unwrap();
        assert_eq!(fresh.cpu_tasks_requested, 1);
        assert_eq!(fresh.reused_completed_cpu_tasks_consumed, 0);
    }

    #[test]
    fn replaceable_cpu_rejects_unsupported_mixed_identity_mask_leak_and_late_discovery() {
        #[derive(Clone, Copy)]
        enum Mode {
            Unsupported,
            ChangedIdentity,
            ChangedIdentityUnknownWork,
            Late,
            UnknownFailure,
            MaskLeak,
            ScoreLeak,
            IllegalPv,
        }
        struct ControlledCpu {
            inner: CpuEngine,
            mode: Mode,
            changed: bool,
            changed_value: CpuValueIdentity,
        }
        impl CpuSearcher for ControlledCpu {
            fn config(&self) -> &CpuConfig {
                self.inner.config()
            }
            fn value_identity(&self) -> &CpuValueIdentity {
                if self.changed {
                    &self.changed_value
                } else {
                    self.inner.value_identity()
                }
            }
            fn search_identity(&self) -> &'static str {
                self.inner.search_identity()
            }
            fn search_conditions(&self) -> String {
                self.inner.search_conditions()
            }
            fn capabilities(&self) -> crate::cpu::CpuCapabilities {
                let mut caps = self.inner.capabilities();
                if matches!(self.mode, Mode::Unsupported) {
                    caps.divergence = false;
                }
                caps
            }
            fn last_attempt_work(&self) -> Option<crate::cpu::CpuWork> {
                if matches!(
                    self.mode,
                    Mode::UnknownFailure | Mode::ChangedIdentityUnknownWork
                ) {
                    None
                } else {
                    self.inner.last_attempt_work()
                }
            }
            fn clear(&mut self) {
                self.inner.clear();
            }
            fn analyze(
                &mut self,
                p: &Position,
                l: CpuLimits,
                c: &AtomicBool,
            ) -> Result<CpuReport, CpuError> {
                let mut report = self.inner.analyze(p, l, c)?;
                if matches!(self.mode, Mode::IllegalPv) {
                    report.best_move = Some(BoardMove::from_uci("e2e5").unwrap());
                    report.pv = vec![report.best_move.unwrap()];
                }
                Ok(report)
            }
            fn analyze_root_moves(
                &mut self,
                p: &Position,
                moves: &[BoardMove],
                l: CpuLimits,
                c: &AtomicBool,
            ) -> Result<CpuReport, CpuError> {
                if matches!(self.mode, Mode::MaskLeak) {
                    let mut report = self.inner.analyze(p, l, c)?;
                    report.root_restricted = true;
                    Ok(report)
                } else {
                    self.inner.analyze_root_moves(p, moves, l, c)
                }
            }
            fn analyze_divergence(
                &mut self,
                p: &Position,
                prefix: &[BoardMove],
                l: CpuLimits,
                c: &AtomicBool,
            ) -> Result<CpuReport, CpuError> {
                if matches!(self.mode, Mode::UnknownFailure) {
                    return Err(CpuError::Unsupported("injected unobserved CPU failure"));
                }
                let mut report = self.inner.analyze_divergence(p, prefix, l, c)?;
                if matches!(
                    self.mode,
                    Mode::ChangedIdentity | Mode::ChangedIdentityUnknownWork
                ) {
                    self.changed = true;
                }
                if matches!(self.mode, Mode::Late) {
                    c.store(true, Ordering::Release);
                }
                if matches!(self.mode, Mode::ScoreLeak) {
                    report.score = i32::MIN;
                }
                Ok(report)
            }
        }
        let make_cpu = |mode| {
            let inner = CpuEngine::new(CpuConfig::default()).unwrap();
            let mut changed_value = inner.value_identity().clone();
            changed_value.semantics = "mixed-checkpoint-test".into();
            ControlledCpu {
                inner,
                mode,
                changed: false,
                changed_value,
            }
        };
        assert!(matches!(
            PalsEngine::new_with_cpu(
                PalsConfig::default(),
                LegalOrderRoleMock,
                make_cpu(Mode::Unsupported)
            ),
            Err(PalsError::Cpu(CpuError::Unsupported(_)))
        ));
        let position = Position::from_fen("2r4k/8/8/8/2Q5/8/7P/K7 b - - 0 1").unwrap();
        for mode in [
            Mode::ChangedIdentity,
            Mode::ChangedIdentityUnknownWork,
            Mode::Late,
            Mode::UnknownFailure,
            Mode::MaskLeak,
            Mode::ScoreLeak,
            Mode::IllegalPv,
        ] {
            let mut engine = PalsEngine::new_with_boxed_cpu(
                PalsConfig::default(),
                LegalOrderRoleMock,
                Box::new(make_cpu(mode)),
            )
            .unwrap();
            engine
                .stores
                .focus_actual_moves(position.snapshot())
                .unwrap();
            let root = engine.intern(position.clone()).unwrap();
            let alternatives: Vec<_> = position
                .legal_moves()
                .into_iter()
                .filter(|movement| *movement != BoardMove::from_uci("c8c4").unwrap())
                .collect();
            let mut counters = PalsCounters::default();
            if matches!(mode, Mode::IllegalPv) {
                assert!(matches!(
                    engine.verify(root, &[], limits(), &AtomicBool::new(false), &mut counters),
                    Err(PalsError::Rules(_))
                ));
                assert_eq!(counters.cpu_tasks_requested, 1);
                assert_eq!(counters.consumed_cpu_tasks, 0);
                assert!(engine.nodes[root].evidence.is_none());
                continue;
            }
            let result = engine.cpu_candidate(
                root,
                root,
                &[],
                if matches!(mode, Mode::MaskLeak) {
                    Some(&alternatives)
                } else {
                    None
                },
                limits(),
                &AtomicBool::new(false),
                &mut counters,
            );
            assert_eq!(counters.cpu_tasks_requested, 1);
            assert_eq!(counters.consumed_cpu_tasks, 0);
            assert!(engine.nodes[root].edges.is_empty());
            assert!(engine.nodes[root].evidence.is_none());
            match mode {
                Mode::Late => {
                    assert!(result.unwrap().is_none());
                    assert_eq!(counters.cpu_tasks, 1);
                    assert_eq!(engine.stores.observations.len(), 1);
                }
                Mode::UnknownFailure => {
                    assert!(matches!(result, Err(PalsError::Cpu(_))));
                    assert_eq!(counters.cpu_tasks, 0);
                    assert!(counters.cpu_work_observation_incomplete);
                }
                _ => {
                    assert!(matches!(
                        result,
                        Err(PalsError::Store(StoreError::InvalidEvidence(_)))
                    ));
                    assert_eq!(counters.cpu_tasks, 1);
                    if matches!(mode, Mode::ChangedIdentity) {
                        // The wrapper rejected an actually returned report;
                        // its independent work snapshot survives exactly once.
                        assert!(counters.cpu_nodes > 0);
                        assert!(counters.cpu_quiescence_nodes <= counters.cpu_nodes);
                        assert!(!counters.cpu_work_observation_incomplete);
                        assert_eq!(counters.completed_cpu_tasks, 0);
                        assert_eq!(engine.stores.observations.len(), 0);
                    }
                    if matches!(mode, Mode::ChangedIdentityUnknownWork) {
                        // Returned-report rejection is known, but independent
                        // actual work is absent. Do not recover nodes from the
                        // rejected report or present the bookkeeping zero as
                        // a measured zero-work execution.
                        assert!(counters.cpu_work_observation_incomplete);
                        assert_eq!(counters.cpu_nodes, 0);
                        assert_eq!(counters.completed_cpu_tasks, 0);
                        assert_eq!(engine.stores.observations.len(), 0);
                    }
                }
            }
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
            assert_eq!(result.resolver_version, PALS_VALUE_RESOLVER_VERSION);
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
        assert_eq!(result.resolver_version, PALS_VALUE_RESOLVER_VERSION);
        assert_eq!(result.terminal, Some(TerminalReason::Stalemate));
        assert_eq!(result.best_move, None);
        let position = Position::startpos();
        let result = engine()
            .search(&position, limits(), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(result.resolver_version, PALS_VALUE_RESOLVER_VERSION);
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
            value_identity: engine.owned_identity().unwrap().clone(),
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
        let previous_generation = engine.stores.generation();
        engine
            .stores
            .focus_actual_moves(engine.nodes[root].position.snapshot())
            .unwrap();
        assert_ne!(engine.stores.generation(), previous_generation);
        let mut resumed = PalsCounters::default();
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
                &mut resumed,
            )
            .unwrap();
        assert_eq!(resumed.cpu_tasks_requested, 1);
        assert_eq!(
            engine
                .stores
                .tasks
                .get(super::super::store::ExecutionId(1))
                .unwrap()
                .resumed_from,
            Some(super::super::store::ExecutionId(0))
        );
        assert_eq!(
            engine
                .stores
                .tasks
                .get(super::super::store::ExecutionId(1))
                .unwrap()
                .key
                .value_identity
                .as_ref(),
            Some(engine.owned_identity().unwrap())
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
        assert_eq!(result.resolver_version, PALS_VALUE_RESOLVER_VERSION);
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
        assert_eq!(result.resolver_version, PALS_VALUE_RESOLVER_VERSION);
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
    fn failed_cpu_execution_keeps_requested_work_snapshot_and_actual_observed_nodes() {
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
        assert!(!snapshot.cpu_work_observation_incomplete);
        assert!(snapshot.cpu_nodes > 0);
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
