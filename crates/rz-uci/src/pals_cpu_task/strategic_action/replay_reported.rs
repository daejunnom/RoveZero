//! Caller report consistency only. No deserialized native witness or Query prior.
use super::super::super::ArtifactPin;
use super::super::super::replay_inputs::{
    ReplayBindingPins, ReplayParentPins, ReplayRegisteredArtifacts, SemanticReceiptProducerScope,
};
use super::*;
use serde_json::Value;

/// A closed, recomputed REPORT, still pending independent native witness and
/// caller chronology. No Serialize/Deserialize/Clone/into-projection constructor.
pub struct ReportedPriorConsistency {
    projection: ReplayQueryPrior,
    readiness: PriorReadiness,
}
impl ReportedPriorConsistency {
    pub fn reported_projection(&self) -> &ReplayQueryPrior {
        &self.projection
    }
    pub fn recomputed_reported_readiness(&self) -> PriorReadiness {
        self.readiness
    }
    pub fn assurance_scope(&self) -> &'static str {
        "reported_consistency_pending_native_witness_and_caller_chronology"
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectionWire {
    schema: String,
    assurance_scope: String,
    projection_source_sha256: [u8; 32],
    outcome: PriorOutcome,
    readiness: PriorReadiness,
    work_returned_elapsed_ms: Option<u64>,
    work_returned_within_execution_window: Option<bool>,
    stage_count: usize,
    stages: Vec<StageWire>,
    opponent_anchor_ply: Option<usize>,
    repaired_line: Vec<u16>,
    opponent_counterline: Vec<u16>,
    opponent_role_steps: usize,
    accepted_opponent_steps: usize,
    endpoint_observed: bool,
    endpoint_rules_terminal: bool,
    actual_utility_groups: u8,
    authorities: ReplayAuthorities,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StageWire {
    phase: Option<PriorPhase>,
    requested_depth: u16,
    node_budget: u64,
    exact_completed: bool,
    report_present: bool,
    completed_depth: Option<u16>,
    completion: Option<PriorCompletion>,
    completed_iteration: bool,
    root_restricted: Option<bool>,
    reused_completed_depth: Option<u16>,
    report_nodes: Option<u64>,
    report_qnodes: Option<u64>,
    report_tt_hits: Option<u64>,
    report_elapsed_ms: Option<u64>,
    attempt_present: bool,
    attempt_nodes: Option<u64>,
    observation_present: bool,
    registered_condition_sha256: [u8; 32],
    task_condition_sha256: [u8; 32],
    pv: Vec<u16>,
}
fn packed(bits: Vec<u16>, maximum: usize) -> Result<PackedMoves, AdmissionFault> {
    if bits.len() > maximum || maximum > MAX_PV_MOVES {
        return Err(fault("reported packed move extent"));
    }
    let mut result = PackedMoves::default();
    for bit in bits {
        Move16::try_from_bits(bit)
            .and_then(Move16::decode)
            .map_err(|_| fault("reported packed move encoding"))?;
        result.moves[result.len] = bit;
        result.len += 1;
    }
    Ok(result)
}
impl StageWire {
    fn into_stage(self) -> Result<PriorStage, AdmissionFault> {
        Ok(PriorStage {
            phase: self.phase,
            requested_depth: self.requested_depth,
            node_budget: self.node_budget,
            exact_completed: self.exact_completed,
            report_present: self.report_present,
            completed_depth: self.completed_depth,
            completion: self.completion,
            completed_iteration: self.completed_iteration,
            root_restricted: self.root_restricted,
            reused_completed_depth: self.reused_completed_depth,
            report_nodes: self.report_nodes,
            report_qnodes: self.report_qnodes,
            report_tt_hits: self.report_tt_hits,
            report_elapsed_ms: self.report_elapsed_ms,
            attempt_present: self.attempt_present,
            attempt_nodes: self.attempt_nodes,
            observation_present: self.observation_present,
            registered_condition_sha256: self.registered_condition_sha256,
            task_condition_sha256: self.task_condition_sha256,
            pv: packed(self.pv, MAX_PV_MOVES)?,
        })
    }
}
impl ProjectionWire {
    fn into_projection(self, source: &ArtifactPin) -> Result<ReplayQueryPrior, AdmissionFault> {
        let digest = self
            .projection_source_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if self.schema != SCHEMA
            || self.assurance_scope != SCOPE
            || digest != source.sha256
            || source.bytes == 0
            || source.bytes > 512 * 1024
            || self.stages.len() > MAX_STAGES
            || self.stage_count != self.stages.len()
            || self.authorities != ReplayAuthorities::default()
            || self.actual_utility_groups != 0
        {
            return Err(fault("reported closed projection/source/authority"));
        }
        let mut stages = std::array::from_fn(|_| PriorStage::default());
        for (target, raw) in stages.iter_mut().zip(self.stages) {
            *target = raw.into_stage()?;
        }
        Ok(ReplayQueryPrior {
            schema: SCHEMA,
            assurance_scope: SCOPE,
            projection_source_sha256: self.projection_source_sha256,
            outcome: self.outcome,
            readiness: self.readiness,
            work_returned_elapsed_ms: self.work_returned_elapsed_ms,
            work_returned_within_execution_window: self.work_returned_within_execution_window,
            stage_count: self.stage_count,
            stages,
            opponent_anchor_ply: self.opponent_anchor_ply,
            repaired_line: packed(self.repaired_line, MAX_LINE_MOVES)?,
            opponent_counterline: packed(self.opponent_counterline, MAX_LINE_MOVES)?,
            opponent_role_steps: self.opponent_role_steps,
            accepted_opponent_steps: self.accepted_opponent_steps,
            endpoint_observed: self.endpoint_observed,
            endpoint_rules_terminal: self.endpoint_rules_terminal,
            actual_utility_groups: self.actual_utility_groups,
            authorities: self.authorities,
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotWire {
    text: String,
    complete: bool,
}
impl SnapshotWire {
    fn into_snapshot(self) -> SnapshotText {
        SnapshotText {
            text: self.text,
            complete: self.complete,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStageWire {
    phase: String,
    cpu_execution: usize,
    requested_depth: u16,
    node_budget: u64,
    exact_completed: bool,
    report_present: bool,
    report_nodes: Option<u64>,
    report_qnodes: Option<u64>,
    report_tt_hits: Option<u64>,
    attempt_present: bool,
    attempt_nodes: Option<u64>,
    attempt_qnodes: Option<u64>,
    attempt_tt_hits: Option<u64>,
    observation_id: Option<usize>,
    raw_stage: SnapshotWire,
    raw_task: SnapshotWire,
    raw_observation: Option<SnapshotWire>,
}
impl RawStageWire {
    fn into_stage(self) -> Result<CpuStageObservation, AdmissionFault> {
        let phase = match self.phase.as_str() {
            "baseline" => "baseline",
            "after" => "after",
            "repair_endpoint" => "repair_endpoint",
            "repair_opponent_endpoint" => "repair_opponent_endpoint",
            "unobserved" => "unobserved",
            _ => return Err(fault("reported raw CPU phase")),
        };
        Ok(CpuStageObservation {
            phase,
            cpu_execution: self.cpu_execution,
            requested_depth: self.requested_depth,
            node_budget: self.node_budget,
            exact_completed: self.exact_completed,
            report_present: self.report_present,
            report_nodes: self.report_nodes,
            report_qnodes: self.report_qnodes,
            report_tt_hits: self.report_tt_hits,
            attempt_present: self.attempt_present,
            attempt_nodes: self.attempt_nodes,
            attempt_qnodes: self.attempt_qnodes,
            attempt_tt_hits: self.attempt_tt_hits,
            observation_id: self.observation_id,
            raw_stage: self.raw_stage.into_snapshot(),
            raw_task: self.raw_task.into_snapshot(),
            raw_observation: self.raw_observation.map(SnapshotWire::into_snapshot),
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputWire {
    schema: String,
    scope: String,
    expected_semantic_receipt_producer_scope: Option<SemanticReceiptProducerScope>,
    mode: ReplayInputMode,
    input_artifact: ArtifactPin,
    registration_artifact: ArtifactPin,
    prepared_action_artifact: ArtifactPin,
    semantic_receipt_artifact: ArtifactPin,
    parent: ReplayParentPins,
    binding: ReplayBindingPins,
    registered_artifacts: ReplayRegisteredArtifacts,
    provider_factory_id: String,
    legacy_cpu_profile_sha256: String,
    binary_pin_scope: String,
    cpu_allowance: u64,
    conservative_repair_role_store_overreservation: bool,
    whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    elapsed_ms: u64,
    cpu_checks: u8,
    cpu_engine_created: bool,
    model_created: bool,
    provider_created: bool,
    actual_utility_groups: u8,
    authorities: ReplayAuthorities,
}
// This is only the receipt fields used by the existing readiness/identity check.
// Remaining native receipt details stay in caller-owned raw bytes, unadmitted.
#[derive(Deserialize)]
struct ReceiptWire {
    model_epoch: [u8; 32],
    export_manifest_sha256: [u8; 32],
    encoding_semantic_sha256: [u8; 32],
    adapter_source_sha256: [u8; 32],
    #[serde(flatten)]
    closure: ClosureReceipt,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeWire {
    schema: String,
    scope: String,
    status: String,
    mode: ReplayInputMode,
    input_admission: InputWire,
    assets_source_admitted: bool,
    asset_profile_artifact: ArtifactPin,
    binary_pin_scope: String,
    model_returned: bool,
    owner_created: bool,
    replay_attempted: bool,
    replay_returned_ok: bool,
    elapsed_ms: u64,
    elapsed_scope: String,
    bytes_prepared_elapsed_ms: Option<u64>,
    bytes_prepared_scope: String,
    original_whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    execution_deadline_exceeded: bool,
    deadline_exceeded: bool,
    canceled: bool,
    cpu_tasks_requested: Option<u64>,
    cpu_tasks_accounted: Option<u64>,
    cpu_reports_returned: Option<u64>,
    cpu_nodes_lower_bound: Option<u64>,
    cpu_work_observation_incomplete: Option<bool>,
    stages: Vec<RawStageWire>,
    replay_state: SnapshotWire,
    opponent_state: Option<SnapshotWire>,
    query_prior: ProjectionWire,
    outcome: SnapshotWire,
    trace_jsonl: String,
    trace_rows: usize,
    trace_complete: bool,
    native_bindings_observed: usize,
    native_completion_unknown_observed: usize,
    native_ready_observed: usize,
    native_terminal_missing: usize,
    native_receipt: Option<ReceiptWire>,
    cleanup_attempted: bool,
    cleanup_returned_ok: bool,
    native_cleanup_missing: bool,
    authorities: ReplayAuthorities,
}

/// Takes ownership of already duplicate-checked, bounded JSON. The caller must
/// separately retain raw bytes and actual process custody. This pure report check
/// cannot return CheckedNativeReplayPrior or a Query/2 capability.
pub fn check_reported_prior_consistency(
    body: Value,
    expected: &ReplayExpectedPins,
    profile_pin: &ArtifactPin,
    profile: &CpuFreshAssetProfile,
    projection_source: &ArtifactPin,
) -> Result<ReportedPriorConsistency, AdmissionFault> {
    bounded_json_extent(&body, super::super::super::replay_inputs::MAX_OUTPUT_BYTES)?;
    if body
        .get("stages")
        .and_then(Value::as_array)
        .is_none_or(|stages| stages.len() > MAX_STAGES)
    {
        return Err(fault("reported raw stage extent"));
    }
    let prior = body
        .get("query_prior")
        .ok_or_else(|| fault("reported projection missing"))?;
    if serde_json::to_vec(prior)
        .map_err(|_| fault("reported projection extent codec"))?
        .len()
        > MAX_PROJECTED_JSON_BYTES
    {
        return Err(fault("reported projection byte bound"));
    }
    let o: NativeWire = serde_json::from_value(body)
        .map_err(|_| fault("reported native closed field/type shape"))?;
    let a = &o.input_admission;
    if o.schema != super::super::QUERY_PRIOR_OBSERVATION_SCHEMA
        || o.scope != super::super::SCOPE
        || o.mode != ReplayInputMode::RepairOpponent4n
        || a.mode != o.mode
        || a.schema != o.mode.input_schema()
        || a.scope != super::super::super::replay_inputs::SCOPE
        || a.registration_artifact != expected.registration_artifact
        || a.prepared_action_artifact != expected.prepared_action_artifact
        || a.semantic_receipt_artifact != expected.semantic_receipt_artifact
        || a.parent != expected.parent
        || a.binding != expected.binding
        || a.registered_artifacts != expected.registered_artifacts
        || expected
            .registered_artifacts
            .opponent_recheck_source
            .is_none()
        || a.provider_factory_id != expected.provider_factory_id
        || a.legacy_cpu_profile_sha256 != expected.legacy_cpu_profile_sha256
        || o.binary_pin_scope != super::super::super::replay_inputs::BINARY_DECLARATION_SCOPE
        || a.binary_pin_scope != o.binary_pin_scope
        || o.asset_profile_artifact != *profile_pin
        || profile.schema != super::super::ASSET_PROFILE_SCHEMA
        || profile.provider != "cpu"
        || profile.domain != "cpu_fresh"
        || o.authorities != ReplayAuthorities::default()
        || a.authorities != ReplayAuthorities::default()
        || a.actual_utility_groups != 0
        || a.cpu_checks != 0
        || a.cpu_engine_created
        || a.model_created
        || a.provider_created
        || a.conservative_repair_role_store_overreservation
        || o.stages.len() > MAX_STAGES
    {
        return Err(fault(
            "reported independent binding/authority/extent differs",
        ));
    }
    if let Some(n) = &o.native_receipt
        && (super::super::digest_string(&profile.model_epoch) != Some(n.model_epoch)
            || super::super::digest_string(&profile.export_manifest.sha256)
                != Some(n.export_manifest_sha256)
            || super::super::digest_string(&profile.encoding_semantic_sha256)
                != Some(n.encoding_semantic_sha256)
            || super::super::digest_string(
                &expected.registered_artifacts.provider_factory_source.sha256,
            ) != Some(n.adapter_source_sha256))
    {
        return Err(fault(
            "reported independent native model/export/encoding/adapter differs",
        ));
    }
    // Preserve rather than reinterpret unused diagnostic snapshots/ticks/IDs.
    let _diagnostics = (
        a.expected_semantic_receipt_producer_scope,
        &a.input_artifact,
        a.elapsed_ms,
        &o.elapsed_scope,
        o.bytes_prepared_elapsed_ms,
        &o.bytes_prepared_scope,
        o.execution_deadline_exceeded,
        &o.trace_jsonl,
        o.trace_rows,
    );
    let stages = o
        .stages
        .into_iter()
        .map(RawStageWire::into_stage)
        .collect::<Result<Vec<_>, _>>()?;
    let projection = o.query_prior.into_projection(projection_source)?;
    let replay_state = o.replay_state.into_snapshot();
    let opponent_state = o.opponent_state.map(SnapshotWire::into_snapshot);
    let outcome = o.outcome.into_snapshot();
    let view = ReadinessObservation {
        canceled: o.canceled,
        deadline_exceeded: o.deadline_exceeded,
        original_whole_wall_ms: o.original_whole_wall_ms,
        cleanup_reserve_ms: o.cleanup_reserve_ms,
        elapsed_ms: o.elapsed_ms,
        input_admission: ReadinessInput {
            whole_wall_ms: a.whole_wall_ms,
            cleanup_reserve_ms: a.cleanup_reserve_ms,
            cpu_allowance: a.cpu_allowance,
        },
        status: &o.status,
        replay_returned_ok: o.replay_returned_ok,
        owner_created: o.owner_created,
        model_returned: o.model_returned,
        replay_attempted: o.replay_attempted,
        assets_source_admitted: o.assets_source_admitted,
        replay_state: &replay_state,
        opponent_state: opponent_state.as_ref(),
        outcome: &outcome,
        trace_complete: o.trace_complete,
        stages: &stages,
        cpu_tasks_requested: o.cpu_tasks_requested,
        cpu_tasks_accounted: o.cpu_tasks_accounted,
        cpu_reports_returned: o.cpu_reports_returned,
        cpu_work_observation_incomplete: o.cpu_work_observation_incomplete,
        cpu_nodes_lower_bound: o.cpu_nodes_lower_bound,
        cleanup_attempted: o.cleanup_attempted,
        cleanup_returned_ok: o.cleanup_returned_ok,
        native_cleanup_missing: o.native_cleanup_missing,
        native_bindings_observed: o.native_bindings_observed,
        native_ready_observed: o.native_ready_observed,
        native_terminal_missing: o.native_terminal_missing,
        native_completion_unknown_observed: o.native_completion_unknown_observed,
        native_receipt: o.native_receipt.map(|n| n.closure),
    };
    let readiness = classify_readiness(&view, &projection);
    if readiness != projection.readiness {
        return Err(fault(
            "reported readiness tag differs from recomputed facts",
        ));
    }
    Ok(ReportedPriorConsistency {
        projection,
        readiness,
    })
}
fn bounded_json_extent(value: &Value, limit: usize) -> Result<(), AdmissionFault> {
    struct Count {
        bytes: usize,
        limit: usize,
    }
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes = self
                .bytes
                .checked_add(bytes.len())
                .filter(|n| *n <= self.limit)
                .ok_or_else(|| std::io::Error::other("reported JSON byte bound"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(&mut Count { bytes: 0, limit }, value)
        .map_err(|_| fault("reported JSON byte bound"))
}
