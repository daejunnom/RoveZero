//! Independent CPU condition reexecution for a checked captured Repair report.
//! Reported score text is never parsed or promoted. Only a newly returned own
//! CpuReport can create the private conditional endpoint facts in this namespace.

use super::{ConditionalEndpointFact, PriorPhase, ReportedRepairRulesConsistency};
use crate::pals_cpu_task::strategic_action::ArtifactPin;
use crate::pals_cpu_task::strategic_action::replay_inputs::{
    PreparedReplayRequest, ReplayBindingPins, ReplayExpectedPins, ReplayInputError,
    ReplayInputMode, ReplayParentPins, SemanticReceiptProducerScope,
    check_replay_inputs_with_semantic_scope,
};
use rz_position::{BoardMove, PlayStatus, Position, PositionSnapshot};
use rz_search::cpu::{
    CPU_MATE_THRESHOLD, CpuCompletion, CpuEngine, CpuError, CpuLimits, CpuReport, CpuScoreScope,
    CpuWork,
};
use rz_search::pals::engine::replay::{
    FRESH_REPLAY_OPPONENT_SCOPE, FRESH_REPLAY_REPAIR_SCOPE, FRESH_REPLAY_SCOPE,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const SCOPE: &str = "captured_nn_input_reinference_and_independent_cpu_condition_reexecution";
const MAX_AUDIT_BYTES: usize = 32 * 1024;

/// No serde, Clone or caller supplied fact constructor. Each stage owns a newly
/// executed CPU report and exact Rules snapshot; elapsed values are actual costs.
#[derive(Debug)]
pub struct IndependentRepairCpuStage {
    phase: PriorPhase,
    snapshot: PositionSnapshot,
    report: CpuReport,
    attempt: CpuWork,
    registered_condition_sha256: [u8; 32],
    task_condition_sha256: [u8; 32],
}
impl IndependentRepairCpuStage {
    pub fn phase(&self) -> PriorPhase {
        self.phase
    }
    pub fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn report(&self) -> &CpuReport {
        &self.report
    }
    pub fn attempt(&self) -> CpuWork {
        self.attempt
    }
    pub fn registered_condition_sha256(&self) -> [u8; 32] {
        self.registered_condition_sha256
    }
    pub fn task_condition_sha256(&self) -> [u8; 32] {
        self.task_condition_sha256
    }
}

/// Source-bound, actual CPU facts. It admits neither the reported child score
/// nor native inference, Query, whole-cost, utility or training authority.
pub struct CheckedIndependentRepairCpuWitness {
    original_input_artifact: ArtifactPin,
    reported_projection_artifact: ArtifactPin,
    parent: ReplayParentPins,
    binding: ReplayBindingPins,
    work_started_at: Instant,
    work_finished_at: Instant,
    original_started: Instant,
    original_execution: Instant,
    original_deadline: Instant,
    stages: Vec<IndependentRepairCpuStage>,
    repaired_line: Vec<u16>,
    opponent_counterline: Vec<u16>,
    repair_endpoint: Option<ConditionalEndpointFact>,
    opponent_endpoint: Option<ConditionalEndpointFact>,
    actual_cpu_nodes: u64,
    source_role_calls: u64,
    captured_witness_spent: AtomicBool,
}
impl CheckedIndependentRepairCpuWitness {
    pub fn original_input_artifact(&self) -> &ArtifactPin {
        &self.original_input_artifact
    }
    pub fn reported_projection_artifact(&self) -> &ArtifactPin {
        &self.reported_projection_artifact
    }
    pub fn parent(&self) -> &ReplayParentPins {
        &self.parent
    }
    pub fn binding(&self) -> &ReplayBindingPins {
        &self.binding
    }
    pub fn work_started_at(&self) -> Instant {
        self.work_started_at
    }
    pub fn work_finished_at(&self) -> Instant {
        self.work_finished_at
    }
    pub fn original_started(&self) -> Instant {
        self.original_started
    }
    pub fn original_execution(&self) -> Instant {
        self.original_execution
    }
    pub fn original_deadline(&self) -> Instant {
        self.original_deadline
    }
    pub fn stages(&self) -> &[IndependentRepairCpuStage] {
        &self.stages
    }
    pub fn repaired_line(&self) -> &[u16] {
        &self.repaired_line
    }
    pub fn opponent_counterline(&self) -> &[u16] {
        &self.opponent_counterline
    }
    pub fn repair_endpoint(&self) -> Option<&ConditionalEndpointFact> {
        self.repair_endpoint.as_ref()
    }
    pub fn opponent_endpoint(&self) -> Option<&ConditionalEndpointFact> {
        self.opponent_endpoint.as_ref()
    }
    pub fn actual_cpu_nodes(&self) -> u64 {
        self.actual_cpu_nodes
    }
    /// The checked original /4 role bound, separate from observed NN graph rows.
    pub fn source_role_calls(&self) -> u64 {
        self.source_role_calls
    }
    pub fn cpu_nodes_lower_bound(&self) -> u64 {
        self.actual_cpu_nodes
    }
    pub fn final_cpu_completion_checked(&self) -> bool {
        true
    }
    pub fn assurance_scope(&self) -> &'static str {
        SCOPE
    }
    /// The actual CPU work can belong to one captured witness only. A failed
    /// downstream attempt remains spent; no Drop, cancellation or retry resets it.
    pub fn reserve_captured_witness_scope(&self) -> Result<(), IndependentRepairCpuFailure> {
        reserve_once(&self.captured_witness_spent)
    }
    pub fn audit_json(&self) -> Result<Vec<u8>, IndependentRepairCpuFailure> {
        let audit = json!({
            "schema":"rz-pals-independent-repair-cpu-witness/1", "scope": SCOPE,
            "original_input_artifact": self.original_input_artifact,
            "reported_projection_artifact": self.reported_projection_artifact,
            "parent": self.parent, "binding": self.binding,
            "work_start_ns": self.work_started_at.saturating_duration_since(self.original_started).as_nanos(),
            "work_finish_ns": self.work_finished_at.saturating_duration_since(self.original_started).as_nanos(),
            "actual_cpu_nodes":self.actual_cpu_nodes,
            "source_role_calls":self.source_role_calls,
            "source_child_score_equivalence":"unknown_not_structurally_reported",
            "stages": self.stages.iter().map(stage_audit).collect::<Vec<_>>(),
            "repaired_line":self.repaired_line, "opponent_counterline":self.opponent_counterline,
            "final_cpu_completion_checked":true,
            "native_inference_checked":false, "whole_cost_admitted":false,
            "query_admitted":false, "utility_authority":false, "training_authority":false,
        });
        let raw =
            serde_json::to_vec(&audit).map_err(|_| failure("audit", "audit encoding failed"))?;
        if raw.len() > MAX_AUDIT_BYTES {
            return Err(failure("audit", "audit bound exceeded"));
        }
        Ok(raw)
    }
}

#[derive(Debug)]
pub enum IndependentRepairCpuCause {
    Refusal(&'static str),
    Original(Box<ReplayInputError>),
    Cpu(Box<CpuError>),
}
/// A failure retains all normally returned stage reports and the last actual
/// attempt counters. It is never a checked witness or an automatic retry ticket.
#[derive(Debug)]
pub struct IndependentRepairCpuFailure {
    pub stage: &'static str,
    pub cause: IndependentRepairCpuCause,
    pub completed_stages: Vec<IndependentRepairCpuStage>,
    pub last_attempt: Option<CpuWork>,
    /// Normally returned CPU output retained if its attempt counters were
    /// unavailable. It is neither a completed stage nor a checked endpoint fact.
    pub returned_report_without_attempt: Option<CpuReport>,
}
impl std::fmt::Display for IndependentRepairCpuFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "independent Repair CPU witness failed at {}: {:?}",
            self.stage, self.cause
        )
    }
}
impl std::error::Error for IndependentRepairCpuFailure {}
impl IndependentRepairCpuFailure {
    pub fn audit_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&json!({"schema":"rz-pals-independent-repair-cpu-failure/1",
            "scope":SCOPE,"stage":self.stage,"cause":format!("{:?}",self.cause),
            "completed_stages":self.completed_stages.iter().map(stage_audit).collect::<Vec<_>>(),
            "last_attempt":self.last_attempt.map(|w|json!({"nodes":w.nodes,"qnodes":w.quiescence_nodes,"tt_hits":w.tt_hits})),
            "returned_report_without_attempt":self.returned_report_without_attempt.as_ref().map(|report|json!({
                "raw_score":report.score,"completed_depth":report.completed_depth,"nodes":report.nodes,
                "qnodes":report.quiescence_nodes,"tt_hits":report.tt_hits,"pv":packed(&report.pv).ok(),
                "completed_stage_admitted":false})),
            "observed_cpu_nodes_lower_bound":self.completed_stages.iter()
                .try_fold(0u64,|n,stage|n.checked_add(stage.attempt.nodes))
                .and_then(|n|n.checked_add(self.last_attempt.map_or(0,|attempt|attempt.nodes))),
            "whole_cpu_total_known":false,
            "unobserved_work_is_not_zero":true,
            "checked_witness_issued":false,"automatic_retry_authority":false}))
    }
}
fn failure(stage: &'static str, reason: &'static str) -> IndependentRepairCpuFailure {
    IndependentRepairCpuFailure {
        stage,
        cause: IndependentRepairCpuCause::Refusal(reason),
        completed_stages: Vec::new(),
        last_attempt: None,
        returned_report_without_attempt: None,
    }
}
fn reserve_once(spent: &AtomicBool) -> Result<(), IndependentRepairCpuFailure> {
    spent
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map(|_| ())
        .map_err(|_| failure("witness_custody", "actual CPU witness scope already spent"))
}
fn control(deadline: Instant, cancel: &AtomicBool) -> Result<(), IndependentRepairCpuFailure> {
    if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
        Err(failure("control", "original execution window/cancellation"))
    } else {
        Ok(())
    }
}
fn packed(moves: &[BoardMove]) -> Result<Vec<u16>, IndependentRepairCpuFailure> {
    crate::pals_cpu_task::pack_moves(moves)
        .map_err(|_| failure("move_codec", "Rules move does not pack"))
}
fn replay(
    root: &Position,
    moves: &[BoardMove],
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<Position, IndependentRepairCpuFailure> {
    let mut position = root.clone();
    for &movement in moves {
        control(deadline, cancel)?;
        let view = position.ordered_legal_moves();
        if position
            .play_status_from_view(&view)
            .map_err(|_| failure("Rules", "status failure"))?
            != PlayStatus::Ongoing
        {
            return Err(failure("Rules", "line continues after terminal state"));
        }
        position
            .make_from_view(&view, movement)
            .map_err(|_| failure("Rules", "illegal source line"))?;
    }
    control(deadline, cancel)?;
    Ok(position)
}
fn stage_audit(stage: &IndependentRepairCpuStage) -> serde_json::Value {
    json!({"phase":stage.phase,"raw_score":stage.report.score,
        "score_perspective":format!("{:?}",stage.snapshot.side_to_move()),
        "completed_depth":stage.report.completed_depth,"nodes":stage.report.nodes,
        "attempt_nodes":stage.attempt.nodes,"qnodes":stage.report.quiescence_nodes,
        "tt_hits":stage.report.tt_hits,"elapsed_ns":stage.report.elapsed.as_nanos(),
        "root_restricted":stage.report.root_restricted,"reused_completed_depth":stage.report.reused_completed_depth,
        "registered_condition_sha256":stage.registered_condition_sha256,
        "task_condition_sha256":stage.task_condition_sha256,
        "pv":packed(&stage.report.pv).ok(),"value_identity":stage.report.value_identity})
}

/// Revalidates the immutable original against independent pins and its original
/// S/E/W, then runs each already reported CPU condition with a new empty engine.
/// The source child score remains unknown; actual CPU facts have a separate scope.
pub fn observe_independent_repair_cpu_witness(
    prepared: &PreparedReplayRequest,
    reported: &ReportedRepairRulesConsistency,
    expected: &ReplayExpectedPins,
    semantic_scope: SemanticReceiptProducerScope,
    cancel: &AtomicBool,
) -> Result<CheckedIndependentRepairCpuWitness, IndependentRepairCpuFailure> {
    let work_started_at = Instant::now();
    control(prepared.execution_deadline(), cancel)?;
    if reported.rules().original_input_artifact() != prepared.artifact() {
        return Err(failure("binding", "reported original input differs"));
    }
    let original_started = prepared
        .deadline()
        .checked_sub(Duration::from_millis(
            prepared.audit().input_admission.whole_wall_ms,
        ))
        .ok_or_else(|| failure("clock", "original S unavailable"))?;
    let checked = check_replay_inputs_with_semantic_scope(
        prepared.as_bytes(),
        expected,
        semantic_scope,
        original_started,
    )
    .map_err(|error| IndependentRepairCpuFailure {
        stage: "original",
        cause: IndependentRepairCpuCause::Original(Box::new(error)),
        completed_stages: Vec::new(),
        last_attempt: None,
        returned_report_without_attempt: None,
    })?;
    if checked.mode() != ReplayInputMode::RepairOpponent4n
        || checked.audit().input_artifact != *prepared.artifact()
        || checked.deadline() != prepared.deadline()
        || checked.execution_deadline() != prepared.execution_deadline()
    {
        return Err(failure(
            "binding",
            "actual original input mode/pin/clock differs",
        ));
    }
    let projection = reported.rules().reported().reported_projection();
    let projection_raw = serde_json::to_vec(projection)
        .map_err(|_| failure("projection", "projection encoding failed"))?;
    if projection_raw.len() > super::MAX_PROJECTED_JSON_BYTES {
        return Err(failure("projection", "bounded projection exceeds cap"));
    }
    let projection_artifact = ArtifactPin {
        bytes: projection_raw.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&projection_raw)),
    };
    let parent = checked.parent_pins().clone();
    let binding = checked.binding_pins().clone();
    let declared_cpu_nodes = checked.resources().cpu_nodes;
    let source_role_calls = checked
        .mode_requirements()
        .map_err(|_| failure("source_bound", "checked source role arithmetic refused"))?
        .actual_role_calls();
    let (_, root, plan, _) = checked.into_owner_args();
    let execution = prepared.execution_deadline();
    let target = replay(&root, &plan.prefix, execution, cancel)?;
    let repair_moves = crate::pals_cpu_task::decode_moves(projection.repaired_line())
        .map_err(|_| failure("Rules", "repaired move codec"))?;
    let counter_moves = crate::pals_cpu_task::decode_moves(projection.opponent_counterline())
        .map_err(|_| failure("Rules", "counterline move codec"))?;
    let repaired = replay(&root, &repair_moves, execution, cancel)?;
    let opponent = replay(&root, &counter_moves, execution, cancel)?;
    let effective: Vec<_> = plan
        .target_legal_order
        .iter()
        .copied()
        .filter(|movement| plan.response_restriction.contains(movement))
        .collect();
    if effective.is_empty() || !(2..=super::MAX_STAGES).contains(&projection.stages().len()) {
        return Err(failure("scope", "bounded stages/restricted order missing"));
    }
    let mut stages = Vec::with_capacity(super::MAX_STAGES);
    let mut actual_cpu_nodes = 0u64;
    let mut repair_endpoint = None;
    let mut opponent_endpoint = None;
    let mut previous_phase = None;
    for source in projection.stages() {
        if let Err(mut error) = control(execution, cancel) {
            error.completed_stages = stages;
            return Err(error);
        }
        let Some(phase) = source.phase else {
            let mut error = failure("scope", "phase missing");
            error.completed_stages = stages;
            return Err(error);
        };
        let (position, depth, restricted, label, scope) = match phase {
            PriorPhase::Baseline if previous_phase.is_none() => (
                &target,
                plan.baseline_depth,
                true,
                "baseline",
                FRESH_REPLAY_SCOPE,
            ),
            PriorPhase::After if previous_phase == Some(PriorPhase::Baseline) => (
                &target,
                plan.requested_depth,
                true,
                "after",
                FRESH_REPLAY_SCOPE,
            ),
            PriorPhase::RepairEndpoint if previous_phase == Some(PriorPhase::After) => (
                &repaired,
                plan.requested_depth,
                false,
                "repair-endpoint",
                FRESH_REPLAY_REPAIR_SCOPE,
            ),
            PriorPhase::RepairOpponentEndpoint
                if matches!(
                    previous_phase,
                    Some(PriorPhase::After | PriorPhase::RepairEndpoint)
                ) =>
            {
                (
                    &opponent,
                    plan.requested_depth,
                    false,
                    "repair-opponent-endpoint",
                    FRESH_REPLAY_OPPONENT_SCOPE,
                )
            }
            _ => {
                let mut error = failure("scope", "phase order differs");
                error.completed_stages = stages;
                return Err(error);
            }
        };
        let mut cpu = match CpuEngine::new(plan.cpu.clone()) {
            Ok(cpu) => cpu,
            Err(error) => {
                return Err(IndependentRepairCpuFailure {
                    stage: "cpu_create",
                    cause: IndependentRepairCpuCause::Cpu(Box::new(error)),
                    completed_stages: stages,
                    last_attempt: None,
                    returned_report_without_attempt: None,
                });
            }
        };
        let condition = cpu.search_conditions();
        let task_condition = if restricted {
            let encoded = match packed(&effective) {
                Ok(encoded) => encoded,
                Err(mut error) => {
                    error.completed_stages = stages;
                    return Err(error);
                }
            };
            let mask = encoded
                .iter()
                .map(|v| format!("{v:04x}"))
                .collect::<String>();
            format!(
                "{condition};replay={scope};phase={label};question=AnalyzeRootMoves;prefix-plies={};ordered-root-mask={mask}",
                plan.prefix.len()
            )
        } else {
            format!(
                "{condition};replay={scope};phase={label};question=AnalyzePosition;input-revision=0;root-moves=empty"
            )
        };
        let condition_sha: [u8; 32] = Sha256::digest(condition.as_bytes()).into();
        let task_sha: [u8; 32] = Sha256::digest(task_condition.as_bytes()).into();
        if !source.complete()
            || source.requested_depth != depth
            || source.node_budget != plan.nodes_per_check
            || source.registered_condition_sha256 != condition_sha
            || source.task_condition_sha256 != task_sha
        {
            let mut error = failure("condition", "reported phase/depth/node/condition differs");
            error.completed_stages = stages;
            return Err(error);
        }
        let limits = CpuLimits {
            max_depth: depth,
            max_nodes: plan.nodes_per_check,
            deadline: Some(execution),
        };
        let attempt_result = if restricted {
            cpu.analyze_root_moves(position, &effective, limits, cancel)
        } else {
            cpu.analyze(position, limits, cancel)
        };
        let attempt = cpu.last_attempt_work();
        let report = match attempt_result {
            Ok(report) => report,
            Err(error) => {
                return Err(IndependentRepairCpuFailure {
                    stage: "cpu_run",
                    cause: IndependentRepairCpuCause::Cpu(Box::new(error)),
                    completed_stages: stages,
                    last_attempt: attempt,
                    returned_report_without_attempt: None,
                });
            }
        };
        let Some(attempt) = attempt else {
            let mut error = failure("cpu_work", "actual attempt counters missing");
            error.completed_stages = stages;
            error.returned_report_without_attempt = Some(report);
            return Err(error);
        };
        let observation = IndependentRepairCpuStage {
            phase,
            snapshot: position.snapshot(),
            report,
            attempt,
            registered_condition_sha256: condition_sha,
            task_condition_sha256: task_sha,
        };
        let Some(next_cpu_nodes) = actual_cpu_nodes.checked_add(attempt.nodes) else {
            stages.push(observation);
            let mut error = failure("cpu_work", "counter overflow after actual CPU return");
            error.completed_stages = stages;
            return Err(error);
        };
        actual_cpu_nodes = next_cpu_nodes;
        let actual_pv = match packed(&observation.report.pv) {
            Ok(pv) => pv,
            Err(mut error) => {
                stages.push(observation);
                error.completed_stages = stages;
                return Err(error);
            }
        };
        let complete = observation.report.completion == CpuCompletion::DepthLimit
            && observation.report.score_scope == CpuScoreScope::CompletedIteration
            && observation.report.completed_depth == depth
            && observation.report.reused_completed_depth == 0
            && observation.report.root_restricted == restricted
            && observation.report.nodes <= plan.nodes_per_check
            && observation.report.nodes == attempt.nodes
            && observation.report.quiescence_nodes == attempt.quiescence_nodes
            && observation.report.tt_hits == attempt.tt_hits
            && source.report_nodes == Some(observation.report.nodes)
            && source.report_qnodes == Some(observation.report.quiescence_nodes)
            && source.report_tt_hits == Some(observation.report.tt_hits)
            && source.attempt_nodes == Some(attempt.nodes)
            && actual_pv.as_slice() == source.pv()
            && observation.report.value_identity.validate().is_ok();
        stages.push(observation);
        if !complete || actual_cpu_nodes > declared_cpu_nodes {
            return Err(IndependentRepairCpuFailure {
                stage: "cpu_condition_result",
                cause: IndependentRepairCpuCause::Refusal(
                    "actual fresh condition result differs or exceeds exact reservation",
                ),
                completed_stages: stages,
                last_attempt: None,
                returned_report_without_attempt: None,
            });
        }
        let last = stages.last().expect("pushed actual observation");
        if !restricted && last.report.score.unsigned_abs() < CPU_MATE_THRESHOLD as u32 {
            let fact = ConditionalEndpointFact {
                snapshot: last.snapshot.clone(),
                raw_value: last.report.score,
                perspective: last.snapshot.side_to_move(),
                root_perspective: root.side_to_move(),
                completed_depth: last.report.completed_depth,
                registered_condition_sha256: condition_sha,
                value_identity: last.report.value_identity.clone(),
            };
            match phase {
                PriorPhase::RepairEndpoint => repair_endpoint = Some(fact),
                PriorPhase::RepairOpponentEndpoint => opponent_endpoint = Some(fact),
                _ => {}
            }
        }
        previous_phase = Some(phase);
    }
    if let Err(mut error) = control(execution, cancel) {
        error.completed_stages = stages;
        return Err(error);
    }
    let work_finished_at = Instant::now();
    Ok(CheckedIndependentRepairCpuWitness {
        original_input_artifact: prepared.artifact().clone(),
        reported_projection_artifact: projection_artifact,
        parent,
        binding,
        work_started_at,
        work_finished_at,
        original_started,
        original_execution: execution,
        original_deadline: prepared.deadline(),
        stages,
        repaired_line: projection.repaired_line().to_vec(),
        opponent_counterline: projection.opponent_counterline().to_vec(),
        repair_endpoint,
        opponent_endpoint,
        actual_cpu_nodes,
        source_role_calls,
        captured_witness_spent: AtomicBool::new(false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpu_witness_reservation_is_permanent_even_after_downstream_refusal() {
        let spent = AtomicBool::new(false);
        reserve_once(&spent).unwrap();
        let _later_failure = failure("downstream", "cancelled or mismatched source");
        assert!(reserve_once(&spent).is_err());
        assert!(spent.load(Ordering::Acquire));
    }
    #[test]
    fn aliases_cannot_reserve_the_same_actual_cpu_work_twice() {
        let spent = std::sync::Arc::new(AtomicBool::new(false));
        let workers = (0..4)
            .map(|_| {
                let spent = std::sync::Arc::clone(&spent);
                std::thread::spawn(move || reserve_once(&spent).is_ok())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .filter(|won| *won)
                .count(),
            1
        );
    }
}
