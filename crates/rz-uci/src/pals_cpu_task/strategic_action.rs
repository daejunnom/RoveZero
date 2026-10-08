//! Caller-bound Query/2 selection to one actual, training-private CPU_T dispatch.
//!
//! This module checks the selected declaration and the original CPU request. It
//! deliberately does not validate Python's full Query/2 capability, current
//! parent, semantic receipt, or prior chronology. Its receipt is neither a native
//! action/witness bridge nor whole-action closure, cost, utility, or a target.

#[cfg(feature = "onnx-cpu")]
pub mod native_replay;
pub mod replay_inputs;

use super::{CpuProfile, CpuTaskAdmission, CpuTaskError, TaskKind};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::time::{Duration, Instant};

pub const SCHEMA: &str = "rz-pals-private-strategic-action/1";
pub const SCOPE: &str = "caller_registered_query_to_actual_own_cpu_dispatch";
pub const MAX_REQUEST_BYTES: usize = super::MAX_REQUEST_BYTES;
pub const MAX_RESPONSE_BYTES: usize = 2 * super::MAX_RESPONSE_BYTES;
const MAX_QUERY_ARTIFACT_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactPin {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategicTaskKind {
    DefendResponse,
    AttackRepair,
    WidenResponses,
    LowerSelectivity,
    ResumeTask,
    CrossProfileRecheck,
    Defer,
}

impl StrategicTaskKind {
    fn legacy(self) -> TaskKind {
        match self {
            Self::DefendResponse => TaskKind::DefendResponse,
            Self::AttackRepair => TaskKind::AttackRepair,
            Self::WidenResponses => TaskKind::WidenResponses,
            Self::LowerSelectivity => TaskKind::LowerSelectivity,
            Self::ResumeTask => TaskKind::ResumeTask,
            Self::CrossProfileRecheck => TaskKind::CrossProfileRecheck,
            Self::Defer => TaskKind::Defer,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategicAction {
    pub slot: u8,
    pub task: StrategicTaskKind,
    pub semantic_input_sha256: String,
    pub profile_registration: u8,
    pub baseline_depth: u16,
    pub requested_depth: u16,
    pub max_nodes_per_check: u64,
    pub max_wall_time_ms: u64,
    pub max_output_bytes: usize,
    pub budget_bucket: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Remaining {
    steps: u64,
    nodes: u64,
    wall_ms: u64,
    output_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    query_sha256: String,
    catalogue_artifact: ArtifactPin,
    before_result_artifact: ArtifactPin,
    prior_ledger_sha256: String,
    action: StrategicAction,
    remaining: Remaining,
    cpu_request_raw: String,
    cpu_request_artifact: ArtifactPin,
    context_sha256: String,
}

/// An actual CPU dispatcher return, including unavailable/deferred returns.
/// The raw CPU wire owns Rules/work facts; these fields never grant utility.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategicReceipt {
    pub schema: String,
    pub status: String,
    pub scope: String,
    pub context_sha256: String,
    pub query_sha256: String,
    pub catalogue_artifact: ArtifactPin,
    pub before_result_artifact: ArtifactPin,
    pub prior_ledger_sha256: String,
    pub action: StrategicAction,
    pub cpu_request_artifact: ArtifactPin,
    pub cpu_response_raw: Option<String>,
    pub cpu_response_artifact: Option<ArtifactPin>,
    /// Entry into the dispatcher, not proof that a physical CPU check started.
    pub cpu_dispatch_attempted: bool,
    /// Failed-run known work can be a lower bound; None is never fabricated zero.
    pub actual_cpu_nodes: Option<u64>,
    pub baseline_present: bool,
    pub after_present: bool,
    /// The library accepts a caller-verified digest; only the CLI observes its image.
    pub binary_pin_scope: String,
    pub elapsed_ms: u64,
    pub deadline_exceeded: bool,
    pub query_revalidated_by_rust: bool,
    pub native_action_causal_bridge_observed: bool,
    pub conditional_witness_validated: bool,
    pub whole_action_cost_observed: bool,
    pub final_search_closure_observed: bool,
    pub action_completion_admitted: bool,
    pub utility_authority: bool,
    pub target_authority: bool,
    pub training_authority: bool,
    pub product_verifier_enabled: bool,
    pub actual_training_executed: bool,
    pub backward_executed: bool,
    pub optimizer_created: bool,
    pub gpu_used: bool,
    pub external_teacher_used: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct StrategicError {
    pub code: &'static str,
    pub stage: &'static str,
    pub message: Box<str>,
    pub elapsed_ms: Option<u64>,
    pub deadline_exceeded: bool,
    #[serde(skip)]
    pub output_limit: usize,
    /// Original failure/work is retained, not converted to a successful zero.
    pub cpu_error: Option<Box<CpuTaskError>>,
    pub receipt: Option<Box<StrategicReceipt>>,
}

impl StrategicError {
    pub fn new(stage: &'static str, message: impl fmt::Display) -> Self {
        Self {
            code: "strategic_action_failed",
            stage,
            message: message
                .to_string()
                .chars()
                .take(512)
                .collect::<String>()
                .into_boxed_str(),
            elapsed_ms: None,
            deadline_exceeded: false,
            output_limit: 1024,
            cpu_error: None,
            receipt: None,
        }
    }

    fn from_cpu(stage: &'static str, error: CpuTaskError) -> Self {
        let mut result = Self::new(stage, &error);
        result.deadline_exceeded = error.deadline_exceeded;
        result.cpu_error = Some(Box::new(error));
        result
    }
}

impl fmt::Display for StrategicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.stage, self.message)
    }
}
impl std::error::Error for StrategicError {}

fn pin(raw: &[u8]) -> ArtifactPin {
    ArtifactPin {
        bytes: raw.len() as u64,
        sha256: super::digest(raw),
    }
}

fn checked_pin(value: &ArtifactPin, maximum: u64) -> Result<(), StrategicError> {
    if value.bytes == 0 || value.bytes > maximum || !super::valid_sha(&value.sha256) {
        return Err(StrategicError::new(
            "artifact_identity",
            "bounded positive bytes and lowercase SHA256 required",
        ));
    }
    Ok(())
}

fn decode(bytes: &[u8], started: Instant) -> Result<(Request, CpuTaskAdmission), StrategicError> {
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        return Err(stamp(
            StrategicError::new("admission", "outer request exceeds 1..=512KiB"),
            started,
            None,
        ));
    }
    let request: Request = serde_json::from_slice(bytes)
        .map_err(|e| stamp(StrategicError::new("json_admission", e), started, None))?;
    // A valid wall establishes its original clock independently of output or
    // identity validity. An invalid output receives only a diagnostic cap.
    if !(1..=super::MAX_WALL_TIME_MS).contains(&request.action.max_wall_time_ms) {
        return Err(stamp(
            StrategicError::new("admission", "finite original action wall required"),
            started,
            None,
        ));
    }
    let admitted = admission(&request, started).map_err(|e| stamp(e, started, None))?;
    if !(1024..=super::MAX_RESPONSE_BYTES).contains(&request.action.max_output_bytes) {
        return Err(stamp(
            StrategicError::new(
                "output_admission",
                "invalid action output; original wall retained with a 1024-byte diagnostic cap",
            ),
            started,
            Some(admitted),
        ));
    }
    if Instant::now() >= admitted.deadline {
        return Err(stamp(
            StrategicError::new(
                "admission_deadline",
                "original wall allowance expired before validation/dispatch",
            ),
            started,
            Some(admitted),
        ));
    }
    let request = validate_request(request).map_err(|e| stamp(e, started, Some(admitted)))?;
    if Instant::now() >= admitted.deadline {
        return Err(stamp(
            StrategicError::new(
                "admission_deadline",
                "original wall allowance expired during validation",
            ),
            started,
            Some(admitted),
        ));
    }
    Ok((request, admitted))
}

fn validate_request(request: Request) -> Result<Request, StrategicError> {
    if request.schema != SCHEMA
        || request.action.slot >= 8
        || request.action.profile_registration >= 8
        || request.action.budget_bucket > 16
        || request.action.task == StrategicTaskKind::LowerSelectivity
    {
        return Err(StrategicError::new(
            "action_admission",
            "unsupported strategic domain, slot, profile, bucket or task",
        ));
    }
    if !(1..=65_536).contains(&request.remaining.steps)
        || request.remaining.nodes > i64::MAX as u64
        || !(1..=super::MAX_WALL_TIME_MS).contains(&request.remaining.wall_ms)
        || !(1..=MAX_QUERY_ARTIFACT_BYTES).contains(&request.remaining.output_bytes)
    {
        return Err(StrategicError::new(
            "remaining_admission",
            "remaining declaration exceeds the supported Query/2 integer limits",
        ));
    }
    for sha in [
        &request.query_sha256,
        &request.prior_ledger_sha256,
        &request.action.semantic_input_sha256,
        &request.context_sha256,
    ] {
        if !super::valid_sha(sha) {
            return Err(StrategicError::new(
                "action_identity",
                "lowercase SHA256 required",
            ));
        }
    }
    checked_pin(&request.catalogue_artifact, MAX_QUERY_ARTIFACT_BYTES)?;
    checked_pin(&request.before_result_artifact, MAX_QUERY_ARTIFACT_BYTES)?;
    checked_pin(
        &request.cpu_request_artifact,
        super::MAX_REQUEST_BYTES as u64,
    )?;
    if request.cpu_request_raw.is_empty()
        || request.cpu_request_raw.len() > super::MAX_REQUEST_BYTES
        || pin(request.cpu_request_raw.as_bytes()) != request.cpu_request_artifact
    {
        return Err(StrategicError::new(
            "cpu_request_identity",
            "original CPU request bytes differ from their independent pin",
        ));
    }
    let mut body =
        serde_json::to_value(&request).map_err(|e| StrategicError::new("context_identity", e))?;
    body.as_object_mut()
        .expect("typed request is an object")
        .remove("context_sha256");
    if super::json_digest(&json!([SCHEMA, body]))
        .map_err(|e| StrategicError::from_cpu("context_identity", e))?
        != request.context_sha256
    {
        return Err(StrategicError::new(
            "context_identity",
            "complete strategic context differs",
        ));
    }
    let cpu = super::decode_request(request.cpu_request_raw.as_bytes())
        .map_err(|e| StrategicError::from_cpu("cpu_request_admission", e))?;
    super::require_ordering(&cpu, super::CpuOrderingPolicy::LegacyMvvLvaV1)
        .map_err(|e| StrategicError::from_cpu("cpu_ordering", e))?;
    // This validates both actual profile configurations, branch and CPU context.
    // It verifies the declared binary here; dispatch checks the caller's image digest.
    super::validate_pins(&cpu, &cpu.cpu_binary_sha256)
        .map_err(|e| StrategicError::from_cpu("cpu_request_pins", e))?;
    let action = &request.action;
    if action.task.legacy() != cpu.task
        || action.baseline_depth != cpu.baseline_depth
        || action.requested_depth != cpu.requested_depth
        || action.max_nodes_per_check != cpu.max_nodes_per_check
        || action.max_wall_time_ms != cpu.max_wall_time_ms
        || action.max_output_bytes != cpu.max_output_bytes
    {
        return Err(StrategicError::new(
            "action_controls",
            "selected task and original CPU controls differ",
        ));
    }
    // Query/2 requires an explicit strict root subset, unlike the legacy CLI's
    // historical first-legal-move fallback. Legality is still checked by Rules.
    if action.task == StrategicTaskKind::WidenResponses && cpu.root_moves.is_empty() {
        return Err(StrategicError::new(
            "action_controls",
            "strategic widening requires an explicit root subset",
        ));
    }
    let effective = if action.task == StrategicTaskKind::CrossProfileRecheck {
        CpuProfile::Independent
    } else {
        CpuProfile::PlanAssisted
    };
    let effective_pin = super::json_digest(&super::profile(&cpu, effective))
        .map_err(|e| StrategicError::from_cpu("effective_profile", e))?;
    let declared = if action.task == StrategicTaskKind::CrossProfileRecheck {
        &cpu.recheck_profile_sha256
    } else {
        &cpu.cpu_profile_sha256
    };
    if &effective_pin != declared {
        return Err(StrategicError::new(
            "effective_profile",
            "selected actual profile differs",
        ));
    }
    let nodes = if action.task == StrategicTaskKind::Defer {
        0
    } else {
        action
            .max_nodes_per_check
            .checked_mul(2)
            .ok_or_else(|| StrategicError::new("reservation", "node reservation overflow"))?
    };
    let output = action
        .max_output_bytes
        .checked_mul(2)
        .filter(|limit| *limit <= MAX_RESPONSE_BYTES)
        .ok_or_else(|| StrategicError::new("reservation", "output reservation overflow"))?;
    if request.remaining.steps == 0
        || nodes > request.remaining.nodes
        || action.max_wall_time_ms > request.remaining.wall_ms
        || output as u64 > request.remaining.output_bytes
    {
        return Err(StrategicError::new(
            "reservation",
            "action does not fit unchanged pre-dispatch allowance",
        ));
    }
    Ok(request)
}

fn admission(request: &Request, started: Instant) -> Result<CpuTaskAdmission, StrategicError> {
    let deadline = started
        .checked_add(Duration::from_millis(request.action.max_wall_time_ms))
        .ok_or_else(|| StrategicError::new("admission", "original deadline overflow"))?;
    let output_limit =
        if (1024..=super::MAX_RESPONSE_BYTES).contains(&request.action.max_output_bytes) {
            request
                .action
                .max_output_bytes
                .checked_mul(2)
                .filter(|limit| *limit <= MAX_RESPONSE_BYTES)
                .ok_or_else(|| StrategicError::new("admission", "outer output limit overflow"))?
        } else {
            // Diagnostic delivery only. decode rejects this output before dispatch.
            1024
        };
    Ok(CpuTaskAdmission {
        deadline,
        output_limit,
    })
}

fn stamp(
    mut error: StrategicError,
    started: Instant,
    admitted: Option<CpuTaskAdmission>,
) -> StrategicError {
    error.elapsed_ms = Some(super::milliseconds(started.elapsed()));
    if let Some(admitted) = admitted {
        error.output_limit = admitted.output_limit;
        error.deadline_exceeded |= Instant::now() >= admitted.deadline;
    }
    error
}

/// The original start covers outer parsing, image inspection, CPU and receipt.
pub fn request_admission(
    bytes: &[u8],
    started: Instant,
) -> Result<CpuTaskAdmission, StrategicError> {
    let (_, admitted) = decode(bytes, started)?;
    Ok(admitted)
}

fn receipt(request: &Request, started: Instant, deadline: Instant) -> StrategicReceipt {
    StrategicReceipt {
        schema: SCHEMA.into(),
        status: "cpu_dispatch_failed".into(),
        scope: SCOPE.into(),
        context_sha256: request.context_sha256.clone(),
        query_sha256: request.query_sha256.clone(),
        catalogue_artifact: request.catalogue_artifact.clone(),
        before_result_artifact: request.before_result_artifact.clone(),
        prior_ledger_sha256: request.prior_ledger_sha256.clone(),
        action: request.action.clone(),
        cpu_request_artifact: request.cpu_request_artifact.clone(),
        cpu_response_raw: None,
        cpu_response_artifact: None,
        cpu_dispatch_attempted: true,
        actual_cpu_nodes: None,
        baseline_present: false,
        after_present: false,
        binary_pin_scope: "library_caller_verified_digest".into(),
        elapsed_ms: super::milliseconds(started.elapsed()),
        deadline_exceeded: Instant::now() >= deadline,
        query_revalidated_by_rust: false,
        native_action_causal_bridge_observed: false,
        conditional_witness_validated: false,
        whole_action_cost_observed: false,
        final_search_closure_observed: false,
        action_completion_admitted: false,
        utility_authority: false,
        target_authority: false,
        training_authority: false,
        product_verifier_enabled: false,
        actual_training_executed: false,
        backward_executed: false,
        optimizer_created: false,
        gpu_used: false,
        external_teacher_used: false,
    }
}

fn finish_receipt(
    mut observed: StrategicReceipt,
    admitted: CpuTaskAdmission,
    started: Instant,
) -> Result<Vec<u8>, StrategicError> {
    let serialize = |value: &StrategicReceipt| -> Result<Vec<u8>, StrategicError> {
        let value = serde_json::to_value(value).map_err(|e| StrategicError::new("receipt", e))?;
        super::canonical(&value).map_err(|e| StrategicError::from_cpu("receipt", e))
    };
    let serialized = serialize(&observed);
    observed.elapsed_ms = super::milliseconds(started.elapsed());
    observed.deadline_exceeded = Instant::now() >= admitted.deadline;
    let result = serialized.and_then(|_| serialize(&observed)).and_then(|mut bytes| {
        bytes.push(b'\n');
        if observed.deadline_exceeded || Instant::now() >= admitted.deadline {
            Err(StrategicError::new("receipt_deadline", "original wall allowance expired; CPU evidence retained only as failed-run evidence"))
        } else if bytes.len() > admitted.output_limit {
            Err(StrategicError::new("receipt", "strategic receipt exceeds original outer output reservation"))
        } else { Ok(bytes) }
    });
    result.map_err(|error| {
        observed.elapsed_ms = super::milliseconds(started.elapsed());
        observed.deadline_exceeded = Instant::now() >= admitted.deadline;
        let mut error = stamp(error, started, Some(admitted));
        error.receipt = Some(Box::new(observed));
        error
    })
}

pub fn dispatch_started(
    bytes: &[u8],
    verified_own_binary_sha256: &str,
    started: Instant,
) -> Result<Vec<u8>, StrategicError> {
    let (request, admitted) = decode(bytes, started)?;
    if Instant::now() >= admitted.deadline {
        return Err(stamp(
            StrategicError::new(
                "admission_deadline",
                "original wall allowance expired before dispatch",
            ),
            started,
            Some(admitted),
        ));
    }
    let result = super::dispatch_started(
        request.cpu_request_raw.as_bytes(),
        verified_own_binary_sha256,
        started,
    );
    let mut observed = receipt(&request, started, admitted.deadline);
    match result {
        Err(cpu_error) => {
            observed.actual_cpu_nodes = cpu_error.known_nodes;
            observed.baseline_present = cpu_error.baseline.is_some();
            observed.after_present = cpu_error.after.is_some();
            let mut error = stamp(
                StrategicError::from_cpu("cpu_dispatch", cpu_error),
                started,
                Some(admitted),
            );
            error.receipt = Some(Box::new(observed));
            Err(error)
        }
        Ok(raw) => {
            observed.status = "cpu_response_observed".into();
            observed.cpu_response_artifact = Some(pin(&raw));
            // Only the existing dispatcher's own UTF-8 JSON return is inspected.
            // No caller report supplies actual work or invents an unavailable zero.
            let facts: Result<Value, _> = serde_json::from_slice(&raw);
            let text = String::from_utf8(raw);
            match (facts, text) {
                (Ok(facts), Ok(text)) => {
                    observed.cpu_response_raw = Some(text);
                    observed.actual_cpu_nodes = facts.get("nodes").and_then(Value::as_u64);
                    observed.baseline_present =
                        facts.get("baseline").is_some_and(|value| !value.is_null());
                    observed.after_present =
                        facts.get("after").is_some_and(|value| !value.is_null());
                    if observed.actual_cpu_nodes.is_none()
                        || facts.get("baseline").is_none()
                        || facts.get("after").is_none()
                    {
                        let mut error = stamp(
                            StrategicError::new(
                                "cpu_response",
                                "own CPU return has no typed work summary",
                            ),
                            started,
                            Some(admitted),
                        );
                        error.receipt = Some(Box::new(observed));
                        return Err(error);
                    }
                    finish_receipt(observed, admitted, started)
                }
                (facts, text) => {
                    if let Ok(text) = text {
                        observed.cpu_response_raw = Some(text);
                    }
                    let mut error = stamp(
                        StrategicError::new(
                            "cpu_response",
                            format!(
                                "own CPU return is not original UTF-8 JSON: {}",
                                facts.err().map_or_else(
                                    || "UTF-8 conversion failed".into(),
                                    |e| e.to_string()
                                )
                            ),
                        ),
                        started,
                        Some(admitted),
                    );
                    error.receipt = Some(Box::new(observed));
                    Err(error)
                }
            }
        }
    }
}

/// Static bridge limits only. Calling this performs no CPU search or model work.
pub fn capabilities() -> Result<Vec<u8>, StrategicError> {
    let value = json!({"schema":SCHEMA,"scope":SCOPE,"cpu_task_schema":super::CPU_TASK_SCHEMA,
        "max_request_bytes":MAX_REQUEST_BYTES,"max_cpu_request_bytes":super::MAX_REQUEST_BYTES,
        "max_response_bytes":MAX_RESPONSE_BYTES,"max_actions":8,"legacy_ordering_only":true,
        "tasks":["defend_response","attack_repair","widen_responses","resume_task","cross_profile_recheck","defer"],
        "query_revalidated_by_rust":false,"native_action_causal_bridge_observed":false,
        "conditional_witness_validated":false,"whole_action_cost_observed":false,
        "final_search_closure_observed":false,"action_completion_admitted":false,
        "utility_authority":false,"target_authority":false,"training_authority":false,
        "product_verifier_enabled":false,"actual_training_executed":false,"backward_executed":false,
        "optimizer_created":false,"gpu_used":false,"external_teacher_used":false});
    let mut bytes =
        super::canonical(&value).map_err(|e| StrategicError::from_cpu("capabilities", e))?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::super::{
        BRANCH_DOMAIN, CPU_TASK_SCHEMA, OwnerRegistry, Position, Request as CpuRequest,
    };
    use super::*;

    const BINARY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn cpu_request(task: StrategicTaskKind) -> CpuRequest {
        let root = Position::startpos();
        let mut request = CpuRequest {
            schema: CPU_TASK_SCHEMA.into(),
            ordering_policy: None,
            task: task.legacy(),
            parent_input_sha256: "b".repeat(64),
            position_command: "position startpos".into(),
            expected_board_fen: root.to_fen(),
            rules_state_sha256: super::super::state_sha(&root, &OwnerRegistry::default()).unwrap(),
            rules_history_sha256: super::super::history_sha(&root).unwrap(),
            cpu_binary_sha256: BINARY.into(),
            branch_sha256: String::new(),
            prefix: vec![],
            root_moves: vec![],
            baseline_depth: 1,
            requested_depth: 2,
            max_nodes_per_check: 4096,
            max_wall_time_ms: 5000,
            max_output_bytes: 65536,
            tt_entries: 64,
            quiescence_ply: 8,
            cpu_profile_sha256: String::new(),
            recheck_profile_sha256: String::new(),
            context_sha256: String::new(),
        };
        if task == StrategicTaskKind::AttackRepair {
            request.prefix =
                super::super::pack_moves(&[super::super::BoardMove::from_uci("e2e4").unwrap()])
                    .unwrap();
        }
        seal_cpu(&mut request);
        request
    }

    fn seal_cpu(request: &mut CpuRequest) {
        request.cpu_profile_sha256 =
            super::super::json_digest(&super::super::profile(request, CpuProfile::PlanAssisted))
                .unwrap();
        request.recheck_profile_sha256 =
            super::super::json_digest(&super::super::profile(request, CpuProfile::Independent))
                .unwrap();
        request.branch_sha256 = super::super::json_digest(&json!([BRANCH_DOMAIN, {"parent_input_sha256":request.parent_input_sha256,"prefix":request.prefix,"root_moves":request.root_moves}])).unwrap();
        let mut body = serde_json::to_value(&*request).unwrap();
        body.as_object_mut().unwrap().remove("context_sha256");
        request.context_sha256 =
            super::super::json_digest(&json!([CPU_TASK_SCHEMA, body])).unwrap();
    }

    fn envelope(cpu: &CpuRequest, task: StrategicTaskKind) -> Request {
        let raw = String::from_utf8(serde_json::to_vec(cpu).unwrap()).unwrap();
        let mut request = Request {
            schema: SCHEMA.into(),
            query_sha256: "c".repeat(64),
            catalogue_artifact: pin(b"independent catalogue"),
            before_result_artifact: pin(b"independent before"),
            prior_ledger_sha256: "d".repeat(64),
            action: StrategicAction {
                slot: 0,
                task,
                semantic_input_sha256: "e".repeat(64),
                profile_registration: 0,
                baseline_depth: cpu.baseline_depth,
                requested_depth: cpu.requested_depth,
                max_nodes_per_check: cpu.max_nodes_per_check,
                max_wall_time_ms: cpu.max_wall_time_ms,
                max_output_bytes: cpu.max_output_bytes,
                budget_bucket: 0,
            },
            remaining: Remaining {
                steps: 1,
                nodes: 8192,
                wall_ms: 5000,
                output_bytes: 131072,
            },
            cpu_request_artifact: pin(raw.as_bytes()),
            cpu_request_raw: raw,
            context_sha256: String::new(),
        };
        seal(&mut request);
        request
    }

    fn seal(request: &mut Request) {
        let mut body = serde_json::to_value(&*request).unwrap();
        body.as_object_mut().unwrap().remove("context_sha256");
        request.context_sha256 = super::super::json_digest(&json!([SCHEMA, body])).unwrap();
    }

    fn run(request: &Request) -> StrategicReceipt {
        serde_json::from_slice(
            &dispatch_started(
                &serde_json::to_vec(request).unwrap(),
                BINARY,
                Instant::now(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn actual_defer_preserves_original_cpu_wire_zero_and_denied_authorities() {
        let mut request = envelope(
            &cpu_request(StrategicTaskKind::Defer),
            StrategicTaskKind::Defer,
        );
        request.remaining.nodes = 0;
        seal(&mut request);
        let observed = run(&request);
        let raw = observed.cpu_response_raw.as_ref().unwrap();
        assert_eq!(observed.cpu_response_artifact, Some(pin(raw.as_bytes())));
        let cpu: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(cpu["status"], "deferred");
        assert_eq!(observed.actual_cpu_nodes, Some(0));
        assert!(!observed.baseline_present && !observed.after_present);
        assert_eq!(observed.context_sha256, request.context_sha256);
        assert!(
            !observed.query_revalidated_by_rust
                && !observed.native_action_causal_bridge_observed
                && !observed.whole_action_cost_observed
                && !observed.final_search_closure_observed
                && !observed.utility_authority
                && !observed.target_authority
                && !observed.training_authority
        );
    }

    #[test]
    fn actual_fresh_cross_and_owned_resume_keep_raw_profiles_and_work() {
        for task in [
            StrategicTaskKind::AttackRepair,
            StrategicTaskKind::CrossProfileRecheck,
            StrategicTaskKind::ResumeTask,
        ] {
            let observed = run(&envelope(&cpu_request(task), task));
            let cpu: Value =
                serde_json::from_str(observed.cpu_response_raw.as_ref().unwrap()).unwrap();
            assert_eq!(cpu["status"], "observed");
            assert!(observed.baseline_present && observed.after_present);
            assert_eq!(observed.actual_cpu_nodes, cpu["nodes"].as_u64());
            assert_eq!(
                cpu["nodes"].as_u64().unwrap(),
                cpu["baseline"]["nodes"].as_u64().unwrap()
                    + cpu["after"]["nodes"].as_u64().unwrap()
            );
            assert!(cpu["elapsed_ms"].as_u64().unwrap() <= observed.elapsed_ms);
            if task == StrategicTaskKind::CrossProfileRecheck {
                assert_ne!(
                    cpu["baseline"]["profile_sha256"],
                    cpu["after"]["profile_sha256"]
                );
            }
            if task == StrategicTaskKind::ResumeTask {
                assert_eq!(cpu["resume_kind"], "completed_iteration");
                assert_eq!(cpu["after"]["reused_completed_depth"], 1);
            }
        }
    }

    #[test]
    fn changed_controls_raw_pins_context_and_reservations_refuse_before_dispatch() {
        let original = envelope(
            &cpu_request(StrategicTaskKind::ResumeTask),
            StrategicTaskKind::ResumeTask,
        );
        let mut candidates = Vec::new();
        let mut changed = original.clone();
        changed.action.requested_depth += 1;
        seal(&mut changed);
        candidates.push(changed);
        let mut changed = original.clone();
        changed.cpu_request_raw.push(' ');
        seal(&mut changed);
        candidates.push(changed);
        let mut changed = original.clone();
        changed.cpu_request_artifact.sha256 = "f".repeat(64);
        seal(&mut changed);
        candidates.push(changed);
        let mut changed = original.clone();
        changed.context_sha256 = "f".repeat(64);
        candidates.push(changed);
        let mut changed = original.clone();
        changed.remaining.nodes -= 1;
        seal(&mut changed);
        candidates.push(changed);
        let mut changed = original.clone();
        changed.remaining.output_bytes -= 1;
        seal(&mut changed);
        candidates.push(changed);
        let mut changed = original.clone();
        changed.remaining.wall_ms -= 1;
        seal(&mut changed);
        candidates.push(changed);
        let mut changed = original.clone();
        changed.remaining.steps = 0;
        seal(&mut changed);
        candidates.push(changed);
        for request in candidates {
            let error = dispatch_started(
                &serde_json::to_vec(&request).unwrap(),
                BINARY,
                Instant::now(),
            )
            .unwrap_err();
            assert!(error.receipt.is_none());
        }
    }

    #[test]
    fn unsupported_remaining_integer_extents_refuse_before_cpu_dispatch() {
        let original = envelope(
            &cpu_request(StrategicTaskKind::Defer),
            StrategicTaskKind::Defer,
        );
        let mut requests = Vec::new();
        let mut request = original.clone();
        request.remaining.steps = 65_537;
        requests.push(request);
        let mut request = original.clone();
        request.remaining.nodes = i64::MAX as u64 + 1;
        requests.push(request);
        let mut request = original.clone();
        request.remaining.wall_ms = super::super::MAX_WALL_TIME_MS + 1;
        requests.push(request);
        let mut request = original.clone();
        request.remaining.output_bytes = MAX_QUERY_ARTIFACT_BYTES + 1;
        requests.push(request);
        for mut request in requests {
            seal(&mut request);
            let error = dispatch_started(
                &serde_json::to_vec(&request).unwrap(),
                BINARY,
                Instant::now(),
            )
            .unwrap_err();
            assert_eq!(error.stage, "remaining_admission");
            assert_eq!(error.output_limit, 2 * request.action.max_output_bytes);
            assert!(error.receipt.is_none() && error.cpu_error.is_none());
        }
    }

    #[test]
    fn exact_typed_wire_rejects_duplicate_unknown_fraction_null_bool_and_negative() {
        let request = envelope(
            &cpu_request(StrategicTaskKind::Defer),
            StrategicTaskKind::Defer,
        );
        let raw = serde_json::to_string(&request).unwrap();
        let duplicate = raw.replacen('{', "{\"schema\":\"duplicate\",", 1);
        let unknown = raw.replacen('{', "{\"unknown\":0,", 1);
        for candidate in [
            duplicate,
            unknown,
            raw.replace("\"slot\":0", "\"slot\":0.5"),
            raw.replace("\"slot\":0", "\"slot\":null"),
            raw.replace("\"slot\":0", "\"slot\":true"),
            raw.replace("\"slot\":0", "\"slot\":-1"),
        ] {
            assert!(request_admission(candidate.as_bytes(), Instant::now()).is_err());
        }
    }

    #[test]
    fn expired_original_start_has_no_dispatch_or_replacement_deadline() {
        let request = envelope(
            &cpu_request(StrategicTaskKind::Defer),
            StrategicTaskKind::Defer,
        );
        let raw = serde_json::to_vec(&request).unwrap();
        let started = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();
        let error = dispatch_started(&raw, BINARY, started).unwrap_err();
        assert!(error.deadline_exceeded && error.receipt.is_none() && error.cpu_error.is_none());
        assert!(request_admission(&raw, started).is_err());
    }

    #[test]
    fn expired_valid_outer_controls_keep_clock_and_output_even_with_bad_identity() {
        let original = envelope(
            &cpu_request(StrategicTaskKind::ResumeTask),
            StrategicTaskKind::ResumeTask,
        );
        let started = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();
        let mut bad_pin = original.clone();
        bad_pin.cpu_request_artifact.sha256 = "f".repeat(64);
        seal(&mut bad_pin);
        let mut bad_control = original.clone();
        bad_control.action.requested_depth += 1;
        seal(&mut bad_control);
        for request in [bad_pin, bad_control] {
            let bytes = serde_json::to_vec(&request).unwrap();
            for error in [
                request_admission(&bytes, started).unwrap_err(),
                dispatch_started(&bytes, BINARY, started).unwrap_err(),
            ] {
                assert!(error.deadline_exceeded);
                assert_eq!(error.output_limit, 2 * original.action.max_output_bytes);
                assert!(error.elapsed_ms.unwrap() >= 10_000);
                assert!(error.receipt.is_none() && error.cpu_error.is_none());
            }
        }
    }

    #[test]
    fn valid_expired_wall_survives_zero_and_overcap_output_rejection() {
        let original = envelope(
            &cpu_request(StrategicTaskKind::Defer),
            StrategicTaskKind::Defer,
        );
        let started = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        for output in [0, super::super::MAX_RESPONSE_BYTES + 1, usize::MAX] {
            let mut request = original.clone();
            request.action.max_wall_time_ms = 1;
            request.action.max_output_bytes = output;
            seal(&mut request);
            let bytes = serde_json::to_vec(&request).unwrap();
            for error in [
                request_admission(&bytes, started).unwrap_err(),
                dispatch_started(&bytes, BINARY, started).unwrap_err(),
            ] {
                assert_eq!(error.stage, "output_admission");
                assert!(error.deadline_exceeded);
                assert_eq!(error.output_limit, 1024);
                assert!(error.elapsed_ms.unwrap() >= 1_000);
                assert!(error.receipt.is_none() && error.cpu_error.is_none());
            }
        }
    }

    #[test]
    fn original_rules_failure_retains_inner_error_and_unknown_work() {
        let mut cpu = cpu_request(StrategicTaskKind::Defer);
        cpu.expected_board_fen = "8/8/8/8/8/8/4K3/7k w - - 0 1".into();
        seal_cpu(&mut cpu);
        let request = envelope(&cpu, StrategicTaskKind::Defer);
        let error = dispatch_started(
            &serde_json::to_vec(&request).unwrap(),
            BINARY,
            Instant::now(),
        )
        .unwrap_err();
        assert_eq!(error.cpu_error.as_ref().unwrap().stage, "rules_identity");
        let observed = error.receipt.as_ref().unwrap();
        assert!(observed.cpu_dispatch_attempted);
        assert_eq!(observed.actual_cpu_nodes, None);
        assert!(observed.cpu_response_raw.is_none() && observed.cpu_response_artifact.is_none());
        assert!(!observed.baseline_present && !observed.after_present);
    }

    #[test]
    fn post_return_output_and_deadline_failures_retain_actual_raw_cpu_observation() {
        let request = envelope(
            &cpu_request(StrategicTaskKind::AttackRepair),
            StrategicTaskKind::AttackRepair,
        );
        let observed = run(&request);
        assert!(observed.actual_cpu_nodes.unwrap() > 0);
        assert!(observed.baseline_present && observed.after_present);
        let started = Instant::now();
        let error = finish_receipt(
            observed.clone(),
            CpuTaskAdmission {
                deadline: started + Duration::from_secs(5),
                output_limit: 1,
            },
            started,
        )
        .unwrap_err();
        assert_eq!(
            error.receipt.as_ref().unwrap().cpu_response_raw,
            observed.cpu_response_raw
        );
        assert_eq!(
            error.receipt.as_ref().unwrap().actual_cpu_nodes,
            observed.actual_cpu_nodes
        );
        let error = finish_receipt(
            observed.clone(),
            CpuTaskAdmission {
                deadline: started,
                output_limit: MAX_RESPONSE_BYTES,
            },
            started,
        )
        .unwrap_err();
        assert!(error.deadline_exceeded);
        assert_eq!(
            error.receipt.as_ref().unwrap().cpu_response_raw,
            observed.cpu_response_raw
        );
        assert_eq!(
            error.receipt.as_ref().unwrap().cpu_response_artifact,
            observed.cpu_response_artifact
        );
        assert_eq!(
            error.receipt.as_ref().unwrap().actual_cpu_nodes,
            observed.actual_cpu_nodes
        );
        assert!(
            error.receipt.as_ref().unwrap().baseline_present
                && error.receipt.as_ref().unwrap().after_present
        );
    }

    #[test]
    fn capability_metadata_has_no_execution_or_product_authority() {
        let value: Value = serde_json::from_slice(&capabilities().unwrap()).unwrap();
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["query_revalidated_by_rust"], false);
        assert_eq!(value["product_verifier_enabled"], false);
        assert_eq!(value["gpu_used"], false);
    }
}
