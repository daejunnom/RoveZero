//! Caller-bound CPU Fresh execution of the separately admitted frozen replay.
//! Input admission, actual source/asset admission and native observations are
//! distinct. No Query, loaded-image, whole-cost, utility or training authority.
//! Synchronous loading is cooperative; an external process supervisor is separate.

use super::ArtifactPin;
use super::replay_inputs::{
    self, ReplayAuthorities, ReplayExpectedPins, ReplayInputAudit, ReplayInputError,
    ReplayInputMode, ReplayRegisteredArtifacts, SemanticReceiptProducerScope,
    replay_mode_requirements,
};
use crate::pals_native::{
    self, NativeInvocationBudget, NativeLoadFailure, NativeOwnerOptions, NativePreparedContext,
    NativeQueryKind, NativeRoleExecutionBinding, NativeRoleFinishHandle, NativeRoleModel,
    NativeRoleObserver, NativeRoleReceipt, NativeRoleRejection, NativeRoleTerminal,
};
use rz_contracts::RequestId;
use rz_eval::onnx::Provider;
use rz_eval::pals_model::{PalsModelInput, PalsRawOutput};
use rz_eval::pals_onnx::{PalsNativeResult, PalsOnnxConfig};
use rz_eval::runtime_pin::RuntimeLibraryPin;
use rz_search::pals::engine::replay::{
    FreshReplayOwner, ReplayError, ReplayOpponentOutcome, ReplayOutcome, ReplayRepairOutcome,
};
use rz_search::pals::engine::{
    PalsLimits, RoleAcceptance, RoleError, RoleLogicalContext, RoleModel,
};
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::fmt::{self, Write as _};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const SCHEMA: &str = "rz-pals-cpu-fresh-native-replay-observation/1";
pub const OPPONENT_SCHEMA: &str = "rz-pals-cpu-fresh-native-replay-observation/2";
pub const ASSET_PROFILE_SCHEMA: &str = "rz-pals-cpu-fresh-replay-assets/1";
pub const SCOPE: &str = "caller_registered_inputs_to_actual_own_cpu_fresh_replay_observations";
// Existing replay registration permits ASCII identifiers without '/'.
pub const FACTORY_ID: &str = "rz-uci-native-cpu-fresh-invocation-v1";
const MAX_PROFILE_BYTES: usize = 8192;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_RUNTIME_BYTES: u64 = 512 * 1024 * 1024;
const HEADER_RESERVE: usize = 512 * 1024;
const ROLE_TRACE_BYTES: usize = 144 * 1024;
const ROWS_PER_ROLE: usize = 12;
const PREPARED_BYTES: usize = 64 * 1024;
const CONTEXT_BYTES: usize = 8 * 1024;
const SMALL_ROW_BYTES: usize = 1024;
const SNAPSHOT_BYTES: usize = 8 * 1024;
const STAGE_TEXT_BYTES: usize = 16 * 1024;
const STATE_TEXT_BYTES: usize = 32 * 1024;
const OUTCOME_TEXT_BYTES: usize = 16 * 1024;
const OPPONENT_STATE_TEXT_BYTES: usize = 32 * 1024;
// Explicit /2 observation profile: two 32KiB large rows, one 8KiB
// accepted-context row, at most nine 1KiB small rows and twelve separators.
// Overflow is an observer failure, never a truncated successful evaluation.
const OPPONENT_LARGE_ROW_BYTES: usize = 32 * 1024;
const OPPONENT_ROLE_TRACE_BYTES: usize = 82 * 1024;

fn role_trace_bytes(checks: usize) -> usize {
    if checks == 4 {
        OPPONENT_ROLE_TRACE_BYTES
    } else {
        ROLE_TRACE_BYTES
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuFreshAssetProfile {
    pub schema: String,
    pub domain: String,
    pub provider: String,
    pub export_manifest: ArtifactPin,
    pub runtime_library: ArtifactPin,
    pub model_epoch: String,
    /// Full Fresh model/input namespace; a Rules-only manifest profile digest
    /// is a distinct declaration and cannot be substituted here.
    pub encoding_semantic_sha256: String,
    pub intra_threads: usize,
    pub cache_public_memory: bool,
    pub device_public_memory: bool,
    pub context_sha256: String,
}

/// The expected profile/pin must come from independent caller registration.
/// These references are never a self-pin fallback or a loaded executable proof.
pub struct CpuFreshReplayAssets<'a> {
    pub export: &'a Path,
    pub runtime_pin: &'a RuntimeLibraryPin,
    pub config: PalsOnnxConfig,
    pub profile_raw: &'a [u8],
    pub expected_profile_artifact: &'a ArtifactPin,
    pub expected_profile: &'a CpuFreshAssetProfile,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct AdmissionFault {
    pub stage: &'static str,
    pub reason: &'static str,
}
fn fault(stage: &'static str, reason: &'static str) -> AdmissionFault {
    AdmissionFault { stage, reason }
}
impl fmt::Display for AdmissionFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.stage, self.reason)
    }
}
impl std::error::Error for AdmissionFault {}

/// The owned original JSON/IO cause survives without dumping local paths.
#[derive(Debug)]
pub enum CpuFreshAssetSourceError {
    Json(serde_json::Error),
    Io(io::Error),
}
#[derive(Debug)]
pub struct CpuFreshAssetError {
    pub fault: AdmissionFault,
    pub source: Option<Box<CpuFreshAssetSourceError>>,
}
impl From<AdmissionFault> for CpuFreshAssetError {
    fn from(fault: AdmissionFault) -> Self {
        Self {
            fault,
            source: None,
        }
    }
}
impl CpuFreshAssetError {
    fn json(stage: &'static str, e: serde_json::Error) -> Self {
        Self {
            fault: fault(stage, "closed profile JSON refused"),
            source: Some(Box::new(CpuFreshAssetSourceError::Json(e))),
        }
    }
    fn io(stage: &'static str, e: io::Error) -> Self {
        Self {
            fault: fault(stage, "actual bounded manifest IO failed"),
            source: Some(Box::new(CpuFreshAssetSourceError::Io(e))),
        }
    }
}
impl Serialize for CpuFreshAssetError {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct View {
            fault: AdmissionFault,
            cause_kind: Option<&'static str>,
            json_category: Option<&'static str>,
            line: Option<usize>,
            column: Option<usize>,
            io_kind: Option<String>,
            raw_os_error: Option<i32>,
            original_cause_retained: bool,
        }
        let mut v = View {
            fault: self.fault,
            cause_kind: None,
            json_category: None,
            line: None,
            column: None,
            io_kind: None,
            raw_os_error: None,
            original_cause_retained: self.source.is_some(),
        };
        match self.source.as_deref() {
            Some(CpuFreshAssetSourceError::Json(e)) => {
                v.cause_kind = Some("json");
                v.json_category = Some(match e.classify() {
                    serde_json::error::Category::Io => "io",
                    serde_json::error::Category::Syntax => "syntax",
                    serde_json::error::Category::Data => "data",
                    serde_json::error::Category::Eof => "eof",
                });
                v.line = Some(e.line());
                v.column = Some(e.column());
            }
            Some(CpuFreshAssetSourceError::Io(e)) => {
                v.cause_kind = Some("io");
                v.io_kind = Some(format!("{:?}", e.kind()));
                v.raw_os_error = e.raw_os_error();
            }
            None => {}
        }
        v.serialize(s)
    }
}

pub enum NativeReplayPrimary {
    Input(Box<ReplayInputError>),
    Admission(AdmissionFault),
    Asset(Box<CpuFreshAssetError>),
    Load(Box<NativeLoadFailure>),
    ObserverInstallation(RoleError),
    Owner(ReplayError),
    Run(ReplayError),
    Control(RoleError),
    Observation(AdmissionFault),
    Output(AdmissionFault),
}
impl NativeReplayPrimary {
    pub fn stage(&self) -> &'static str {
        match self {
            Self::Input(_) => "input",
            Self::Admission(e) => e.stage,
            Self::Asset(e) => e.fault.stage,
            Self::Load(_) => "native_load",
            Self::ObserverInstallation(_) => "observer_installation",
            Self::Owner(_) => "owner_construction",
            Self::Run(_) => "replay_run",
            Self::Control(_) => "original_control",
            Self::Observation(_) => "observation",
            Self::Output(_) => "output",
        }
    }
}
impl fmt::Debug for NativeReplayPrimary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(e) => fmt::Debug::fmt(e, f),
            Self::Admission(e) => fmt::Debug::fmt(e, f),
            Self::Load(e) => fmt::Debug::fmt(e, f),
            Self::Asset(e) => f
                .debug_struct("Asset")
                .field("fault", &e.fault)
                .field("source_retained", &e.source.is_some())
                .finish(),
            Self::ObserverInstallation(e) | Self::Control(e) => fmt::Debug::fmt(e, f),
            Self::Owner(e) | Self::Run(e) => fmt::Debug::fmt(e, f),
            Self::Observation(e) | Self::Output(e) => fmt::Debug::fmt(e, f),
        }
    }
}
impl Serialize for NativeReplayPrimary {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct View<'a> {
            stage: &'static str,
            diagnostic: DebugRef<'a, NativeReplayPrimary>,
            input: Option<&'a ReplayInputError>,
            load: Option<LoadView<'a>>,
            asset: Option<&'a CpuFreshAssetError>,
        }
        #[derive(Serialize)]
        struct BackendView<'a> {
            kind: DebugRef<'a, rz_eval::error::FailureKind>,
            stage: DebugRef<'a, rz_eval::error::FailureStage>,
            detail: &'a str,
            cause: Option<DebugRef<'a, rz_eval::error::ExternalCause>>,
            native_diagnostic_present: bool,
            native_code: Option<&'a str>,
            native_message_bytes: Option<usize>,
            native_truncated: Option<bool>,
            raw_native_diagnostic_archived_in_this_json: bool,
        }
        #[derive(Serialize)]
        struct LoadView<'a> {
            stage: DebugRef<'a, pals_native::NativeLoadStage>,
            elapsed_ms: u64,
            execution_expired: bool,
            whole_expired: bool,
            canceled: bool,
            backend_error: Option<BackendView<'a>>,
            original_backend_error_retained: bool,
        }
        let load = match self {
            Self::Load(e) => Some(LoadView {
                stage: DebugRef(&e.stage),
                elapsed_ms: milliseconds(e.elapsed),
                execution_expired: e.execution_expired,
                whole_expired: e.whole_expired,
                canceled: e.canceled,
                original_backend_error_retained: e.backend_error.is_some(),
                backend_error: e.backend_error.as_deref().map(|b| BackendView {
                    kind: DebugRef(&b.kind),
                    stage: DebugRef(&b.stage),
                    detail: b.detail,
                    cause: b.cause.as_ref().map(DebugRef),
                    native_diagnostic_present: b.native.is_some(),
                    native_code: b.native.as_ref().map(|n| n.code.as_str()),
                    native_message_bytes: b.native.as_ref().map(|n| n.message.len()),
                    native_truncated: b.native.as_ref().map(|n| n.truncated),
                    raw_native_diagnostic_archived_in_this_json: false,
                }),
            }),
            _ => None,
        };
        View {
            stage: self.stage(),
            diagnostic: DebugRef(self),
            input: match self {
                Self::Input(e) => Some(e.as_ref()),
                _ => None,
            },
            load,
            asset: match self {
                Self::Asset(e) => Some(e.as_ref()),
                _ => None,
            },
        }
        .serialize(s)
    }
}

#[derive(Debug, Serialize)]
pub struct NativeReplayError {
    pub code: &'static str,
    /// Independent expected declaration, including failures before comparison.
    /// It grants no historical producer or current loaded-image authority.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_semantic_receipt_producer_scope: Option<SemanticReceiptProducerScope>,
    pub primary: Box<NativeReplayPrimary>,
    pub cleanup_error: Option<Box<RoleErrorView>>,
    pub observer_error: Option<Box<AdmissionFault>>,
    pub output_error: Option<Box<AdmissionFault>>,
    pub elapsed_ms: u64,
    pub deadline_exceeded: bool,
    pub execution_deadline_exceeded: bool,
    pub canceled: bool,
    pub output_limit: usize,
    pub original_whole_deadline_retained: bool,
    pub receipt: Option<Box<NativeReplayObservation>>,
    /// Original W, never a relative serialization/recovery window. Malformed
    /// input without an admitted whole clock keeps this explicitly unknown.
    #[serde(skip)]
    serialization_deadline: Option<Instant>,
}
impl fmt::Display for NativeReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CPU Fresh replay failed at {}", self.primary.stage())
    }
}
impl std::error::Error for NativeReplayError {}
impl NativeReplayError {
    /// Finite serialization; failure leaves this error and all actual rows intact.
    pub fn serialized(&self) -> Result<Vec<u8>, AdmissionFault> {
        serialization_clock(self.serialization_deadline)?;
        let mut out = reserved_bytes(self.output_limit)?;
        serialization_clock(self.serialization_deadline)?;
        serialize_into(
            &mut out,
            self.output_limit,
            self.serialization_deadline,
            self,
        )?;
        serialization_clock(self.serialization_deadline)?;
        Ok(out)
    }
}
fn serialization_clock(until: Option<Instant>) -> Result<(), AdmissionFault> {
    if until.is_some_and(|until| Instant::now() >= until) {
        return Err(fault(
            "error_serialization_clock",
            "original whole deadline exhausted; owned primary and receipt retained",
        ));
    }
    Ok(())
}
fn input_error(
    error: ReplayInputError,
    started: Instant,
    cancel: &AtomicBool,
) -> NativeReplayError {
    let serialization_deadline = if error.original_whole_deadline_retained {
        error
            .whole_wall_ms
            .and_then(|ms| started.checked_add(Duration::from_millis(ms)))
    } else {
        None
    };
    NativeReplayError {
        code: "native_replay_failed",
        expected_semantic_receipt_producer_scope: error.expected_semantic_receipt_producer_scope,
        cleanup_error: None,
        observer_error: None,
        output_error: None,
        elapsed_ms: milliseconds(started.elapsed()),
        deadline_exceeded: error.deadline_exceeded,
        execution_deadline_exceeded: error.execution_deadline_exceeded,
        canceled: cancel.load(Ordering::Acquire),
        output_limit: error.output_limit,
        original_whole_deadline_retained: error.original_whole_deadline_retained,
        receipt: None,
        serialization_deadline,
        primary: Box::new(NativeReplayPrimary::Input(Box::new(error))),
    }
}

#[derive(Debug)]
pub struct RoleErrorView(pub RoleError);
impl Serialize for RoleErrorView {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        DebugRef(&self.0).serialize(s)
    }
}

#[derive(Debug, Serialize)]
pub struct SnapshotText {
    pub text: String,
    pub complete: bool,
}
impl SnapshotText {
    fn reserved(limit: usize) -> Result<Self, AdmissionFault> {
        Ok(Self {
            text: reserved_text(limit)?,
            complete: false,
        })
    }
    fn capture<T: fmt::Debug + ?Sized>(&mut self, value: &T) -> bool {
        self.text.clear();
        let cap = self.text.capacity();
        self.complete = write!(
            TextWriter {
                text: &mut self.text,
                limit: cap
            },
            "{value:?}"
        )
        .is_ok();
        self.complete
    }
}

#[derive(Debug, Serialize)]
pub struct CpuStageObservation {
    pub phase: &'static str,
    /// Search-local store ID. Never joined numerically to native Runtime IDs.
    pub cpu_execution: usize,
    pub requested_depth: u16,
    pub node_budget: u64,
    pub exact_completed: bool,
    pub report_present: bool,
    pub report_nodes: Option<u64>,
    pub report_qnodes: Option<u64>,
    pub report_tt_hits: Option<u64>,
    pub attempt_present: bool,
    pub attempt_nodes: Option<u64>,
    pub attempt_qnodes: Option<u64>,
    pub attempt_tt_hits: Option<u64>,
    pub observation_id: Option<usize>,
    pub raw_stage: SnapshotText,
    pub raw_task: SnapshotText,
    pub raw_observation: Option<SnapshotText>,
}
impl CpuStageObservation {
    fn reserved() -> Result<Self, AdmissionFault> {
        Ok(Self {
            phase: "unobserved",
            cpu_execution: 0,
            requested_depth: 0,
            node_budget: 0,
            exact_completed: false,
            report_present: false,
            report_nodes: None,
            report_qnodes: None,
            report_tt_hits: None,
            attempt_present: false,
            attempt_nodes: None,
            attempt_qnodes: None,
            attempt_tt_hits: None,
            observation_id: None,
            raw_stage: SnapshotText::reserved(STAGE_TEXT_BYTES)?,
            raw_task: SnapshotText::reserved(SNAPSHOT_BYTES)?,
            raw_observation: Some(SnapshotText::reserved(SNAPSHOT_BYTES)?),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct NativeReplayObservation {
    pub schema: &'static str,
    pub scope: &'static str,
    pub status: &'static str,
    pub mode: ReplayInputMode,
    pub input_admission: ReplayInputAudit,
    pub assets_source_admitted: bool,
    pub asset_profile_artifact: ArtifactPin,
    pub binary_pin_scope: &'static str,
    pub model_returned: bool,
    pub owner_created: bool,
    pub replay_attempted: bool,
    pub replay_returned_ok: bool,
    pub elapsed_ms: u64,
    pub elapsed_scope: &'static str,
    /// Main encoding has completed; the final timing member and caller delivery
    /// remain outside this tick. The entire buffer still passes the same W guard.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_prepared_elapsed_ms: Option<u64>,
    pub bytes_prepared_scope: &'static str,
    pub original_whole_wall_ms: u64,
    pub cleanup_reserve_ms: u64,
    pub execution_deadline_exceeded: bool,
    pub deadline_exceeded: bool,
    pub canceled: bool,
    pub cpu_tasks_requested: Option<u64>,
    pub cpu_tasks_accounted: Option<u64>,
    pub cpu_reports_returned: Option<u64>,
    pub cpu_nodes_lower_bound: Option<u64>,
    pub cpu_work_observation_incomplete: Option<bool>,
    pub stages: Vec<CpuStageObservation>,
    /// Actual Rust snapshots; not a legacy raw JSON response or imported record.
    pub replay_state: SnapshotText,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opponent_state: Option<SnapshotText>,
    pub outcome: SnapshotText,
    pub trace_jsonl: String,
    pub trace_rows: usize,
    pub trace_complete: bool,
    pub native_bindings_observed: usize,
    pub native_completion_unknown_observed: usize,
    pub native_ready_observed: usize,
    pub native_terminal_missing: usize,
    pub native_receipt: Option<NativeRoleReceipt>,
    pub cleanup_attempted: bool,
    pub cleanup_returned_ok: bool,
    pub native_cleanup_missing: bool,
    pub authorities: ReplayAuthorities,
}

#[derive(Clone, Copy)]
struct Clock {
    started: Instant,
    execution: Instant,
    whole: Instant,
    output: usize,
}
impl Clock {
    fn check(self, cancel: &AtomicBool) -> Result<(), RoleError> {
        if cancel.load(Ordering::Acquire) {
            Err(RoleError::Canceled)
        } else if Instant::now() >= self.execution {
            Err(RoleError::Deadline)
        } else {
            Ok(())
        }
    }
    fn stamp(self, r: &mut NativeReplayObservation, cancel: &AtomicBool) {
        let now = Instant::now();
        r.elapsed_ms = milliseconds(now.saturating_duration_since(self.started));
        r.execution_deadline_exceeded = now >= self.execution;
        r.deadline_exceeded = now >= self.whole;
        r.canceled = cancel.load(Ordering::Acquire);
    }
}

#[derive(Clone, Copy, Debug)]
struct BindingState {
    binding: NativeRoleExecutionBinding,
    unknown: bool,
    ready: bool,
}
struct Collector {
    bytes: Vec<u8>,
    limit: usize,
    rows: usize,
    row_limit: usize,
    bindings: Vec<BindingState>,
    role_limit: usize,
    whole: Instant,
    failure: Option<AdmissionFault>,
    /// One separately reserved synchronous formatter arena, never a callback
    /// borrow retained after return or format!(large) followed by truncation.
    terminal_scratch: SnapshotText,
    large_row_bytes: usize,
}
impl Collector {
    #[cfg(test)]
    fn reserved(roles: usize, whole: Instant) -> Result<Self, AdmissionFault> {
        Self::reserved_for_checks(roles, 2, whole)
    }
    fn reserved_for_checks(
        roles: usize,
        checks: usize,
        whole: Instant,
    ) -> Result<Self, AdmissionFault> {
        let large_row_bytes = if checks == 4 {
            OPPONENT_LARGE_ROW_BYTES
        } else {
            PREPARED_BYTES
        };
        let limit = roles
            .checked_mul(role_trace_bytes(checks))
            .ok_or(fault("observer_reserve", "byte overflow"))?;
        let row_limit = roles
            .checked_mul(ROWS_PER_ROLE)
            .ok_or(fault("observer_reserve", "row overflow"))?;
        let mut bindings = Vec::new();
        reserve_vec(&mut bindings, roles)?;
        Ok(Self {
            bytes: reserved_bytes(limit)?,
            limit,
            rows: 0,
            row_limit,
            bindings,
            role_limit: roles,
            whole,
            failure: None,
            terminal_scratch: SnapshotText::reserved(large_row_bytes)?,
            large_row_bytes,
        })
    }
    fn fail(&mut self, why: &'static str) -> RoleError {
        self.failure.get_or_insert(fault("observer", why));
        RoleError::Backend("CPU Fresh replay bounded observer failed; actual work retained".into())
    }
    fn row<T: Serialize>(&mut self, event: &T, cap: usize) -> Result<(), RoleError> {
        if self.failure.is_some() {
            return Err(self.fail("prior observer failure"));
        }
        if self.rows >= self.row_limit {
            return Err(self.fail("row reservation exhausted"));
        }
        let old = self.bytes.len();
        let end = old
            .checked_add(cap)
            .and_then(|n| n.checked_add(1))
            .unwrap_or(self.limit)
            .min(self.limit);
        if serialize_into(&mut self.bytes, end, Some(self.whole), event).is_err()
            || self.bytes.len() >= self.limit
        {
            self.bytes.truncate(old);
            return Err(self.fail("row serialization/clock/byte bound"));
        }
        self.bytes.push(b'\n');
        self.rows += 1;
        Ok(())
    }
    fn dispatched(&mut self, binding: NativeRoleExecutionBinding) -> Result<(), RoleError> {
        if self.bindings.len() >= self.role_limit
            || self.bindings.iter().any(|b| {
                b.binding.request == binding.request || b.binding.execution == binding.execution
            })
        {
            return Err(self.fail("duplicate binding or role reservation exhausted"));
        }
        self.bindings.push(BindingState {
            binding,
            unknown: false,
            ready: false,
        });
        self.row(
            &BoundEvent {
                event: "dispatched",
                binding: BindingView::from(binding),
                fact: "actual_lease",
            },
            SMALL_ROW_BYTES,
        )
    }
    fn terminal(
        &mut self,
        binding: NativeRoleExecutionBinding,
        event: NativeRoleTerminal<'_>,
    ) -> Result<(), RoleError> {
        let Some(at) = self.bindings.iter().position(|b| b.binding == binding) else {
            return Err(self.fail("terminal without actual dispatched binding"));
        };
        match event {
            NativeRoleTerminal::CompletionUnknown(reason) => {
                if self.bindings[at].unknown || self.bindings[at].ready {
                    return Err(self.fail("duplicate or late unknown"));
                }
                self.bindings[at].unknown = true;
                self.row(
                    &BoundEvent {
                        event: "completion_unknown",
                        binding: BindingView::from(binding),
                        fact: DebugRef(&reason),
                    },
                    SMALL_ROW_BYTES,
                )
            }
            NativeRoleTerminal::Ready(result) => {
                if self.bindings[at].ready {
                    return Err(self.fail("duplicate actual Ready"));
                }
                self.bindings[at].ready = true;
                // PalsNativeResult has no blanket Debug/Serialize. Record its
                // actual variant, format raw evaluation bits directly into the
                // reserved arena, and leave unsupported control payload missing.
                let (kind, complete) = match result {
                    Ok(PalsNativeResult::Evaluation(raw)) => (
                        "evaluation",
                        self.terminal_scratch.capture(&RawOutputBits(raw)),
                    ),
                    Err(error) => ("backend_error", self.terminal_scratch.capture(error)),
                    Ok(other) => {
                        self.terminal_scratch.text.clear();
                        self.terminal_scratch.complete = false;
                        (native_result_kind(other), false)
                    }
                };
                #[derive(Serialize)]
                struct Fact<'a> {
                    kind: &'static str,
                    raw_projection: &'a SnapshotText,
                    evaluation_scalar_encoding: &'static str,
                    full_native_diagnostic_archive: bool,
                }
                // Move the arena out only during synchronous serialization;
                // row() cannot retain the fact's borrowed reference.
                let scratch = std::mem::replace(
                    &mut self.terminal_scratch,
                    SnapshotText {
                        text: String::new(),
                        complete: false,
                    },
                );
                let row = self.row(
                    &BoundEvent {
                        event: "physical_ready",
                        binding: BindingView::from(binding),
                        fact: Fact {
                            kind,
                            raw_projection: &scratch,
                            evaluation_scalar_encoding: "ordered_ieee754_f32_bits_as_u32_lower_hex",
                            full_native_diagnostic_archive: false,
                        },
                    },
                    self.large_row_bytes,
                );
                self.terminal_scratch = scratch;
                row?;
                if complete {
                    Ok(())
                } else {
                    Err(self.fail("partial/unsupported raw physical terminal projection"))
                }
            }
        }
    }
}

fn native_result_kind(result: &PalsNativeResult) -> &'static str {
    match result {
        PalsNativeResult::Evaluation(_) => "evaluation",
        PalsNativeResult::NewGame => "new_game",
        PalsNativeResult::Stats(_) => "stats",
        PalsNativeResult::RuntimeVerified => "runtime_verified",
        PalsNativeResult::CudaPlacementVerified(_) => "cuda_placement_verified",
        PalsNativeResult::RuntimeMappingsObserved(_) => "runtime_mappings_observed",
        #[cfg(feature = "experimental-io-binding")]
        PalsNativeResult::CudaRecordPagesObserved(_) => "cuda_record_pages_observed",
    }
}
struct FloatBits<'a>(&'a [f32]);
impl fmt::Debug for FloatBits<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "len={};u32hex=", self.0.len())?;
        for &v in self.0 {
            write!(f, "{:08x}", v.to_bits())?;
        }
        Ok(())
    }
}
struct RawOutputBits<'a>(&'a PalsRawOutput);
impl fmt::Debug for RawOutputBits<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PalsRawOutputBits")
            .field("candidate_logits", &FloatBits(&self.0.candidate_logits))
            .field("wdl_logits", &FloatBits(&self.0.wdl_logits))
            .field(
                "divergence_logits",
                &self.0.divergence_logits.as_deref().map(FloatBits),
            )
            .field(
                "task_logits",
                &self.0.task_logits.as_ref().map(|v| FloatBits(v.as_slice())),
            )
            .field("private_latent", &FloatBits(&self.0.private_latent))
            .finish()
    }
}

#[derive(Serialize)]
struct IdView {
    epoch: u64,
    sequence: u64,
}
impl From<RequestId> for IdView {
    fn from(id: RequestId) -> Self {
        Self {
            epoch: id.epoch.0,
            sequence: id.sequence,
        }
    }
}
#[derive(Serialize)]
struct BindingView {
    request: IdView,
    native_execution: IdView,
    execution_namespace: &'static str,
}
impl From<NativeRoleExecutionBinding> for BindingView {
    fn from(b: NativeRoleExecutionBinding) -> Self {
        Self {
            request: b.request.into(),
            native_execution: IdView {
                epoch: b.execution.epoch.0,
                sequence: b.execution.sequence,
            },
            execution_namespace: "rz_runtime_physical_execution_not_search_store_execution",
        }
    }
}
#[derive(Serialize)]
struct BoundEvent<T: Serialize> {
    event: &'static str,
    binding: BindingView,
    fact: T,
}
struct ReplayObserver(Arc<Mutex<Collector>>);
impl ReplayObserver {
    fn with(
        &self,
        operation: impl FnOnce(&mut Collector) -> Result<(), RoleError>,
    ) -> Result<(), RoleError> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| RoleError::Backend("CPU replay collector poisoned".into()))?;
        operation(&mut guard)
    }
}
impl NativeRoleObserver for ReplayObserver {
    fn prepared(
        &mut self,
        id: RequestId,
        input: &PalsModelInput,
        context: NativePreparedContext<'_>,
    ) -> Result<(), RoleError> {
        self.prepared_with_logical_context(id, input, context, None)
    }
    fn prepared_with_logical_context(
        &mut self,
        id: RequestId,
        input: &PalsModelInput,
        context: NativePreparedContext<'_>,
        logical: Option<&RoleLogicalContext>,
    ) -> Result<(), RoleError> {
        #[derive(Serialize)]
        struct Event<'a> {
            event: &'static str,
            request: IdView,
            kind: &'static str,
            input: &'a PalsModelInput,
            logical: Option<DebugRef<'a, RoleLogicalContext>>,
        }
        let kind = match context {
            NativePreparedContext::Divergence { .. } => "divergence",
            NativePreparedContext::Role { kind, .. } => match kind {
                NativeQueryKind::Propose => "propose",
                NativeQueryKind::Reply => "reply",
                NativeQueryKind::Repair => "repair",
                NativeQueryKind::Divergence => "divergence",
            },
        };
        self.with(|c| {
            c.row(
                &Event {
                    event: "prepared",
                    request: id.into(),
                    kind,
                    input,
                    logical: logical.map(DebugRef),
                },
                c.large_row_bytes,
            )
        })
    }
    fn dispatched(&mut self, binding: NativeRoleExecutionBinding) -> Result<(), RoleError> {
        self.with(|c| c.dispatched(binding))
    }
    fn terminal(
        &mut self,
        binding: NativeRoleExecutionBinding,
        event: NativeRoleTerminal<'_>,
    ) -> Result<(), RoleError> {
        self.with(|c| c.terminal(binding, event))
    }
    fn accepted_context(
        &mut self,
        id: RequestId,
        acceptance: &RoleAcceptance<'_>,
    ) -> Result<(), RoleError> {
        #[derive(Serialize)]
        struct Event<'a> {
            event: &'static str,
            request: IdView,
            context: DebugRef<'a, RoleLogicalContext>,
        }
        self.with(|c| {
            c.row(
                &Event {
                    event: "accepted_context",
                    request: id.into(),
                    context: DebugRef(acceptance.context),
                },
                CONTEXT_BYTES,
            )
        })
    }
    fn delivered(&mut self, id: RequestId) -> Result<(), RoleError> {
        self.small("delivered", id, "actual_delivery")
    }
    fn accepted(&mut self, id: RequestId) -> Result<(), RoleError> {
        self.small("accepted", id, "actual_consumption_callback")
    }
    fn rejected(&mut self, id: RequestId, reason: NativeRoleRejection) -> Result<(), RoleError> {
        #[derive(Serialize)]
        struct Event<'a> {
            event: &'static str,
            request: IdView,
            reason: DebugRef<'a, NativeRoleRejection>,
        }
        self.with(|c| {
            c.row(
                &Event {
                    event: "rejected",
                    request: id.into(),
                    reason: DebugRef(&reason),
                },
                SMALL_ROW_BYTES,
            )
        })
    }
}
impl ReplayObserver {
    fn small(
        &self,
        event: &'static str,
        id: RequestId,
        fact: &'static str,
    ) -> Result<(), RoleError> {
        #[derive(Serialize)]
        struct Event {
            event: &'static str,
            request: IdView,
            fact: &'static str,
        }
        self.with(|c| {
            c.row(
                &Event {
                    event,
                    request: id.into(),
                    fact,
                },
                SMALL_ROW_BYTES,
            )
        })
    }
}

/// Immutable serialized-output bound computed from this wrapper's retained
/// observation layout. This is neither an allocator/RSS measurement nor a
/// supported-mode, CPU-budget, provider or execution admission capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayOutputRequirements {
    trace_bytes: usize,
    text_bytes: usize,
    required_output_bytes: usize,
}
impl ReplayOutputRequirements {
    pub const fn trace_bytes(self) -> usize {
        self.trace_bytes
    }
    pub const fn text_bytes(self) -> usize {
        self.text_bytes
    }
    pub const fn required_output_bytes(self) -> usize {
        self.required_output_bytes
    }
    /// Checks a declaration without allocating or extending its allowance.
    pub fn check_output_limit(self, limit: usize) -> Result<(), AdmissionFault> {
        if self.required_output_bytes > limit {
            return Err(fault(
                "output_reserve",
                "declared output cannot retain worst-case observations",
            ));
        }
        Ok(())
    }
}

/// Pure checked arithmetic shared by result-free caller preparation and the
/// actual pre-load reservation. Counts do not attest to any work performed.
/// Supported mode/counts and the original output cap remain caller-checker
/// responsibilities; no new count cap or default is introduced here.
pub fn replay_output_requirements(
    roles: usize,
    checks: usize,
) -> Result<ReplayOutputRequirements, AdmissionFault> {
    let trace = roles
        .checked_mul(role_trace_bytes(checks))
        .ok_or(fault("output_reserve", "trace overflow"))?;
    // Committed compact JSONL contains no unescaped controls except its
    // single newline separators. Wrapping that valid UTF-8 as a JSON string
    // costs at most two bytes per existing byte (quotes/backslashes/newline).
    // Free-text Debug snapshots use the separate six-byte worst-case bound.
    let stage_bytes = checks
        .checked_mul(STAGE_TEXT_BYTES + 2 * SNAPSHOT_BYTES)
        .ok_or(fault("output_reserve", "stage snapshot overflow"))?;
    let text_bytes = stage_bytes
        .checked_add(STATE_TEXT_BYTES + OUTCOME_TEXT_BYTES)
        .and_then(|n| {
            if checks == 4 {
                n.checked_add(OPPONENT_STATE_TEXT_BYTES)
            } else {
                Some(n)
            }
        })
        .ok_or(fault("output_reserve", "retained snapshot overflow"))?;
    let need = trace
        .checked_mul(2)
        .and_then(|n| text_bytes.checked_mul(6).and_then(|t| n.checked_add(t)))
        .and_then(|n| n.checked_add(HEADER_RESERVE))
        .ok_or(fault("output_reserve", "serialized upper bound overflow"))?;
    Ok(ReplayOutputRequirements {
        trace_bytes: trace,
        text_bytes,
        required_output_bytes: need,
    })
}

struct Reserved {
    collector: Arc<Mutex<Collector>>,
    output: Vec<u8>,
    slots: Vec<CpuStageObservation>,
    state: SnapshotText,
    opponent_state: Option<SnapshotText>,
    outcome: SnapshotText,
}
impl Reserved {
    fn new(roles: usize, checks: usize, clock: Clock) -> Result<Self, AdmissionFault> {
        replay_output_requirements(roles, checks)?.check_output_limit(clock.output)?;
        let mut slots = Vec::new();
        reserve_vec(&mut slots, checks)?;
        for _ in 0..checks {
            slots.push(CpuStageObservation::reserved()?);
        }
        Ok(Self {
            collector: Arc::new(Mutex::new(Collector::reserved_for_checks(
                roles,
                checks,
                clock.whole,
            )?)),
            output: reserved_bytes(clock.output)?,
            slots,
            state: SnapshotText::reserved(STATE_TEXT_BYTES)?,
            opponent_state: if checks == 4 {
                Some(SnapshotText::reserved(OPPONENT_STATE_TEXT_BYTES)?)
            } else {
                None
            },
            outcome: SnapshotText::reserved(OUTCOME_TEXT_BYTES)?,
        })
    }
}

#[derive(Debug)]
enum Outcome {
    Reply(ReplayOutcome),
    Repair(ReplayRepairOutcome),
    Opponent(ReplayOpponentOutcome),
}

fn run_owner<M: RoleModel>(
    owner: &mut FreshReplayOwner<M>,
    mode: ReplayInputMode,
    limits: PalsLimits,
    cancel: &AtomicBool,
) -> Result<Outcome, ReplayError> {
    match mode {
        ReplayInputMode::ReplyOnly2n => owner.run(limits, cancel).map(Outcome::Reply),
        ReplayInputMode::RepairEndpoint3n => {
            owner.run_with_repair(limits, cancel).map(Outcome::Repair)
        }
        ReplayInputMode::RepairOpponent4n => owner
            .run_with_opponent_recheck(limits, cancel)
            .map(Outcome::Opponent),
    }
}

pub fn dispatch_started(
    bytes: &[u8],
    expected: &ReplayExpectedPins,
    assets: CpuFreshReplayAssets<'_>,
    started: Instant,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, NativeReplayError> {
    dispatch_started_inner(bytes, expected, None, assets, started, cancel)
}

/// The historical producer scope is independently selected by the caller. It
/// does not change replay binary declaration scope or observe a loaded image.
pub fn dispatch_started_with_semantic_scope(
    bytes: &[u8],
    expected: &ReplayExpectedPins,
    expected_scope: SemanticReceiptProducerScope,
    assets: CpuFreshReplayAssets<'_>,
    started: Instant,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, NativeReplayError> {
    dispatch_started_inner(
        bytes,
        expected,
        Some(expected_scope),
        assets,
        started,
        cancel,
    )
}

fn dispatch_started_inner(
    bytes: &[u8],
    expected: &ReplayExpectedPins,
    expected_scope: Option<SemanticReceiptProducerScope>,
    assets: CpuFreshReplayAssets<'_>,
    started: Instant,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, NativeReplayError> {
    let result = (|| {
        let admission = match expected_scope {
            Some(scope) => replay_inputs::check_replay_inputs_with_semantic_scope(
                bytes, expected, scope, started,
            ),
            None => replay_inputs::check_replay_inputs(bytes, expected, started),
        };
        let checked = match admission {
            Ok(c) => c,
            Err(e) => return Err(input_error(e, started, cancel)),
        };
        let clock = Clock {
            started,
            execution: checked.execution_deadline(),
            whole: checked.deadline(),
            output: checked.output_limit(),
        };
        let mode = checked.mode();
        let audit = checked.audit();
        let reserve = Duration::from_millis(checked.resources().cleanup_reserve_ms);
        let budget = NativeInvocationBudget::new(started, clock.execution, clock.whole, reserve)
            .map_err(|e| bare_error(NativeReplayPrimary::Control(e), clock, cancel))?;
        clock
            .check(cancel)
            .map_err(|e| bare_error(NativeReplayPrimary::Control(e), clock, cancel))?;
        let profile = admit_assets(
            &assets,
            checked.registration().provider_factory_id.as_str(),
            &checked.registration().artifacts,
            clock,
            cancel,
        )
        .map_err(|e| bare_error(NativeReplayPrimary::Asset(Box::new(e)), clock, cancel))?;
        let (config, root, plan, limits) = checked.into_owner_args();
        let requirements = replay_mode_requirements(
            mode,
            config.line_plies,
            plan.prefix.len(),
            plan.nodes_per_check,
        )
        .map_err(|e| bare_error(NativeReplayPrimary::Owner(e), clock, cancel))?;
        let roles = usize::try_from(requirements.actual_role_calls())
            .ok()
            .ok_or_else(|| {
                bare_error(
                    NativeReplayPrimary::Admission(fault(
                        "observer_reserve",
                        "role bound overflow",
                    )),
                    clock,
                    cancel,
                )
            })?;
        let checks = requirements.cpu_checks();
        let mut reserved = Reserved::new(roles, checks, clock)
            .map_err(|e| bare_error(NativeReplayPrimary::Admission(e), clock, cancel))?;
        let mut receipt = NativeReplayObservation {
            schema: if mode == ReplayInputMode::RepairOpponent4n {
                OPPONENT_SCHEMA
            } else {
                SCHEMA
            },
            scope: SCOPE,
            status: "admitted_before_load",
            mode,
            input_admission: audit,
            assets_source_admitted: true,
            asset_profile_artifact: assets.expected_profile_artifact.clone(),
            binary_pin_scope: replay_inputs::BINARY_DECLARATION_SCOPE,
            model_returned: false,
            owner_created: false,
            replay_attempted: false,
            replay_returned_ok: false,
            elapsed_ms: 0,
            original_whole_wall_ms: 0,
            cleanup_reserve_ms: milliseconds(reserve),
            elapsed_scope: "original_start_through_cleanup_and_observation_capture",
            bytes_prepared_elapsed_ms: None,
            bytes_prepared_scope: "original_start_through_main_encoding_before_final_timing_member_and_caller_delivery",
            execution_deadline_exceeded: false,
            deadline_exceeded: false,
            canceled: false,
            cpu_tasks_requested: None,
            cpu_tasks_accounted: None,
            cpu_reports_returned: None,
            cpu_nodes_lower_bound: None,
            cpu_work_observation_incomplete: None,
            stages: Vec::new(),
            replay_state: reserved.state,
            opponent_state: reserved.opponent_state,
            outcome: reserved.outcome,
            trace_jsonl: String::new(),
            trace_rows: 0,
            trace_complete: true,
            native_receipt: None,
            native_bindings_observed: 0,
            native_completion_unknown_observed: 0,
            native_ready_observed: 0,
            native_terminal_missing: 0,
            cleanup_attempted: false,
            cleanup_returned_ok: false,
            native_cleanup_missing: true,
            authorities: ReplayAuthorities::default(),
        };
        receipt.original_whole_wall_ms = receipt.input_admission.whole_wall_ms;
        let mut primary = None;
        let mut cleanup_error = None;
        let mut output_error = None;
        let options = NativeOwnerOptions {
            drain_limit: reserve,
            host_record_pages: None,
        };
        let model_result = NativeRoleModel::load_pinned_cpu_fresh_for_invocation(
            assets.export,
            &profile.export_manifest.sha256,
            assets.runtime_pin,
            assets.config,
            options,
            budget,
            cancel,
        );
        match model_result {
            Err(failure) => {
                if let Some(handle) = failure.finish.as_ref() {
                    cleanup_error = finish(handle, clock.whole, &mut receipt);
                }
                primary = Some(NativeReplayPrimary::Load(Box::new(failure)));
            }
            Ok(mut model) => {
                receipt.model_returned = true;
                let handle = model.finish_handle();
                let identity = model.source_identity();
                if digest_string(&profile.model_epoch) != Some(identity.checkpoint_sha256)
                    || digest_string(&profile.export_manifest.sha256)
                        != Some(identity.export_manifest_sha256)
                    || digest_string(&profile.encoding_semantic_sha256)
                        != Some(identity.encoding_semantic_sha256)
                    || identity.adapter_source_sha256 != pals_native::pals_native_source_digest()
                    || identity.execution.provider != "cpu"
                {
                    primary = Some(NativeReplayPrimary::Admission(fault(
                        "loaded_identity",
                        "actual model/encoding/provider differs",
                    )));
                } else if let Err(e) = clock.check(cancel) {
                    primary = Some(NativeReplayPrimary::Control(e));
                } else if let Err(e) =
                    model.set_observer(Box::new(ReplayObserver(Arc::clone(&reserved.collector))))
                {
                    primary = Some(NativeReplayPrimary::ObserverInstallation(e));
                } else {
                    match FreshReplayOwner::new(config, model, root, plan) {
                        Err(e) => primary = Some(NativeReplayPrimary::Owner(e)),
                        Ok(mut owner) => {
                            receipt.owner_created = true;
                            let result = match clock.check(cancel) {
                                Err(e) => {
                                    primary = Some(NativeReplayPrimary::Control(e));
                                    None
                                }
                                Ok(()) => {
                                    receipt.replay_attempted = true;
                                    Some(run_owner(&mut owner, mode, limits, cancel))
                                }
                            };
                            receipt.replay_returned_ok = result.as_ref().is_some_and(Result::is_ok);
                            if !capture_owner(&owner, &mut reserved.slots, &mut receipt) {
                                output_error = Some(fault(
                                    "observation_capture",
                                    "partial bounded snapshot; actual scalar work retained",
                                ));
                            }
                            match result {
                                Some(Ok(outcome)) => {
                                    let complete = match outcome {
                                        Outcome::Reply(o) => receipt.outcome.capture(&o),
                                        Outcome::Repair(o) => receipt.outcome.capture(&o),
                                        Outcome::Opponent(o) => receipt.outcome.capture(&o),
                                    };
                                    if !complete {
                                        output_error = Some(fault(
                                            "outcome_capture",
                                            "outcome snapshot bound",
                                        ));
                                    }
                                }
                                Some(Err(e)) => primary = Some(NativeReplayPrimary::Run(e)),
                                None => {}
                            }
                        }
                    }
                }
                // Owner/model Drop and this explicit same-W finish do not pump a
                // runtime, retry a failed operation, or grant a recovery window.
                cleanup_error = finish(&handle, clock.whole, &mut receipt);
            }
        }
        let observer_error = drain_collector(&reserved.collector, &mut receipt);
        clock.stamp(&mut receipt, cancel);
        if primary.is_none() && (receipt.deadline_exceeded || receipt.canceled) {
            primary = Some(NativeReplayPrimary::Control(if receipt.canceled {
                RoleError::Canceled
            } else {
                RoleError::Deadline
            }));
        }
        if primary.is_none() && cleanup_error.is_some() {
            primary = Some(NativeReplayPrimary::Observation(fault(
                "native_cleanup",
                "native cleanup did not return success",
            )));
        }
        if primary.is_none()
            && let Some(e) = observer_error
        {
            primary = Some(NativeReplayPrimary::Observation(e));
        }
        if primary.is_none()
            && let Some(e) = output_error
        {
            primary = Some(NativeReplayPrimary::Output(e));
        }
        receipt.status = if primary.is_some() {
            "failed_with_retained_observations"
        } else {
            "observations_returned"
        };
        if let Some(p) = primary {
            return Err(with_receipt(
                p,
                cleanup_error,
                observer_error,
                output_error,
                receipt,
                clock,
                cancel,
            ));
        }
        if let Err(e) = serialize_into(
            &mut reserved.output,
            clock.output,
            Some(clock.whole),
            &receipt,
        ) {
            return Err(with_receipt(
                NativeReplayPrimary::Output(e),
                cleanup_error,
                observer_error,
                Some(e),
                receipt,
                clock,
                cancel,
            ));
        }
        receipt.bytes_prepared_elapsed_ms = Some(milliseconds(started.elapsed()));
        if let Err(e) = append_prepared_tick(
            &mut reserved.output,
            clock,
            receipt.bytes_prepared_elapsed_ms.unwrap_or(u64::MAX),
        ) {
            return Err(with_receipt(
                NativeReplayPrimary::Output(e),
                cleanup_error,
                observer_error,
                Some(e),
                receipt,
                clock,
                cancel,
            ));
        }
        // This actual post-encoding guard covers the terminal timing member as well.
        // receipt.elapsed_ms retains its explicit cleanup/capture scope on success.
        let whole_expired = Instant::now() >= clock.whole;
        let canceled = cancel.load(Ordering::Acquire);
        if whole_expired || canceled {
            return Err(with_receipt(
                NativeReplayPrimary::Control(if canceled {
                    RoleError::Canceled
                } else {
                    RoleError::Deadline
                }),
                cleanup_error,
                observer_error,
                output_error,
                receipt,
                clock,
                cancel,
            ));
        }
        Ok(reserved.output)
    })();
    // Every explicit failure keeps the expectation even if no input/audit/owner
    // was admitted. This is metadata, never a successful scope comparison.
    result.map_err(|mut error: NativeReplayError| {
        error.expected_semantic_receipt_producer_scope = expected_scope;
        error
    })
}

fn capture_owner<M: RoleModel>(
    owner: &FreshReplayOwner<M>,
    slots: &mut Vec<CpuStageObservation>,
    receipt: &mut NativeReplayObservation,
) -> bool {
    let counters = owner.counters();
    receipt.cpu_tasks_requested = Some(counters.cpu_tasks_requested);
    // cpu_tasks includes an OwnedReportRejected accounting path with no owned
    // report. Keep that counter separate from actually retained stage reports.
    receipt.cpu_tasks_accounted = Some(counters.cpu_tasks);
    receipt.cpu_nodes_lower_bound = Some(counters.cpu_nodes);
    receipt.cpu_work_observation_incomplete = Some(counters.cpu_work_observation_incomplete);
    let mut complete = true;
    let stages = owner.stages();
    receipt.cpu_reports_returned = Some(
        stages
            .iter()
            .filter(|stage| stage.report().is_some())
            .count() as u64,
    );
    if stages.len() > slots.len() {
        complete = false;
    }
    let count = stages.len().min(slots.len());
    for (at, stage) in stages.iter().take(count).enumerate() {
        let out = &mut slots[at];
        out.phase = match stage.phase() {
            rz_search::pals::engine::replay::ReplayCpuPhase::Baseline => "baseline",
            rz_search::pals::engine::replay::ReplayCpuPhase::After => "after",
            rz_search::pals::engine::replay::ReplayCpuPhase::RepairEndpoint => "repair_endpoint",
            rz_search::pals::engine::replay::ReplayCpuPhase::RepairOpponentEndpoint => {
                "repair_opponent_endpoint"
            }
        };
        out.cpu_execution = stage.execution().0;
        out.requested_depth = stage.requested_depth();
        out.node_budget = stage.node_budget();
        out.exact_completed = stage.exact_completed();
        out.report_present = stage.report().is_some();
        if let Some(report) = stage.report() {
            out.report_nodes = Some(report.nodes);
            out.report_qnodes = Some(report.quiescence_nodes);
            out.report_tt_hits = Some(report.tt_hits);
        }
        out.attempt_present = stage.attempt().is_some();
        if let Some(attempt) = stage.attempt() {
            out.attempt_nodes = attempt.work.nodes;
            out.attempt_qnodes = attempt.work.qnodes;
            out.attempt_tt_hits = attempt.work.tt_hits;
        }
        out.observation_id = stage.observation().map(|id| id.0);
        complete &= out.raw_stage.capture(stage);
        complete &= out.raw_task.capture(&owner.task(stage.execution()));
        if let Some(id) = stage.observation() {
            if let Some(text) = &mut out.raw_observation {
                complete &= text.capture(&owner.observation(id));
            }
        } else {
            out.raw_observation = None;
        }
    }
    slots.truncate(count);
    receipt.stages = std::mem::take(slots);
    // Two named groups preserve all actual facts without joining CPU/native IDs
    // or promoting the endpoint snapshot into a full Repair recheck.
    complete &= receipt.replay_state.capture(&(
        (
            "reply",
            (
                owner.counters(),
                owner.elapsed(),
                owner.proposal(),
                owner.candidate_line(),
                owner.reply_line(),
                owner.records(),
                owner.reply_prepared_context(),
                owner.reply_accepted_context(),
                owner.publication_revision(),
            ),
        ),
        (
            "repair_endpoint_only",
            (
                owner.model_counterline(),
                owner.repaired_line(),
                owner.repair_record_revision(),
                owner.repair_line_id(),
                owner.repair_endpoint(),
            ),
        ),
    ));
    // This separate /2 snapshot does not relabel the third endpoint or join
    // search-local role contexts/CPU IDs with physical native request IDs.
    if let Some(state) = &mut receipt.opponent_state {
        complete &= state.capture(&(
            "repair_opponent_continuation",
            owner.opponent_anchor_ply(),
            owner.opponent_counterline(),
            owner.opponent_role_steps(),
            owner.opponent_endpoint().map(|endpoint| {
                (
                    endpoint.state,
                    endpoint.situation,
                    endpoint.snapshot,
                    endpoint.terminal,
                    endpoint.stage,
                )
            }),
        ));
    }
    complete
}

// This explicit finish uses only original W. Model/owner Drop followed by
// finish is not a runtime poll pump or recovery/retry admission for unknown work.
fn finish(
    handle: &NativeRoleFinishHandle,
    whole: Instant,
    receipt: &mut NativeReplayObservation,
) -> Option<RoleErrorView> {
    receipt.cleanup_attempted = true;
    receipt.native_cleanup_missing = false;
    match handle.finish(whole) {
        Ok(r) => {
            receipt.cleanup_returned_ok = true;
            receipt.native_receipt = Some(r);
            None
        }
        Err(e) => {
            receipt.native_receipt = Some(handle.receipt());
            Some(RoleErrorView(e))
        }
    }
}
fn drain_collector(
    collector: &Arc<Mutex<Collector>>,
    receipt: &mut NativeReplayObservation,
) -> Option<AdmissionFault> {
    let mut c = match collector.lock() {
        Ok(c) => c,
        Err(p) => {
            let mut c = p.into_inner();
            c.failure
                .get_or_insert(fault("observer", "collector poisoned; prior rows retained"));
            c
        }
    };
    receipt.native_bindings_observed = c.bindings.len();
    receipt.native_completion_unknown_observed = c.bindings.iter().filter(|b| b.unknown).count();
    receipt.native_ready_observed = c.bindings.iter().filter(|b| b.ready).count();
    receipt.native_terminal_missing = c.bindings.iter().filter(|b| !b.ready).count();
    receipt.trace_rows = c.rows;
    receipt.trace_complete = c.failure.is_none() && receipt.native_terminal_missing == 0;
    match String::from_utf8(std::mem::take(&mut c.bytes)) {
        Ok(text) => receipt.trace_jsonl = text,
        Err(_) => {
            receipt.trace_complete = false;
            return Some(fault("observer", "retained JSON UTF8 invariant failed"));
        }
    }
    c.failure
}

fn bare_error(
    primary: NativeReplayPrimary,
    clock: Clock,
    cancel: &AtomicBool,
) -> NativeReplayError {
    NativeReplayError {
        code: "native_replay_failed",
        expected_semantic_receipt_producer_scope: None,
        primary: Box::new(primary),
        cleanup_error: None,
        observer_error: None,
        output_error: None,
        elapsed_ms: milliseconds(clock.started.elapsed()),
        deadline_exceeded: Instant::now() >= clock.whole,
        execution_deadline_exceeded: Instant::now() >= clock.execution,
        canceled: cancel.load(Ordering::Acquire),
        output_limit: clock.output,
        original_whole_deadline_retained: true,
        receipt: None,
        serialization_deadline: Some(clock.whole),
    }
}
fn with_receipt(
    primary: NativeReplayPrimary,
    cleanup_error: Option<RoleErrorView>,
    observer_error: Option<AdmissionFault>,
    output_error: Option<AdmissionFault>,
    mut receipt: NativeReplayObservation,
    clock: Clock,
    cancel: &AtomicBool,
) -> NativeReplayError {
    clock.stamp(&mut receipt, cancel);
    receipt.status = "failed_with_retained_observations";
    let mut e = bare_error(primary, clock, cancel);
    e.expected_semantic_receipt_producer_scope = receipt
        .input_admission
        .expected_semantic_receipt_producer_scope;
    e.cleanup_error = cleanup_error.map(Box::new);
    e.observer_error = observer_error.map(Box::new);
    e.output_error = output_error.map(Box::new);
    e.receipt = Some(Box::new(receipt));
    e
}

fn admit_assets(
    assets: &CpuFreshReplayAssets<'_>,
    factory: &str,
    sources: &ReplayRegisteredArtifacts,
    clock: Clock,
    cancel: &AtomicBool,
) -> Result<CpuFreshAssetProfile, CpuFreshAssetError> {
    clock
        .check(cancel)
        .map_err(|_| fault("assets_clock", "original execution deadline/cancel"))?;
    if factory != FACTORY_ID {
        return Err(fault(
            "factory",
            "separate CPU Fresh factory registration required",
        )
        .into());
    }
    validate_sources(sources)?;
    let profile = check_cpu_fresh_asset_profile(
        assets.profile_raw,
        assets.expected_profile_artifact,
        assets.expected_profile,
    )?;
    validate_config(assets.config, &profile)?;
    if assets.runtime_pin.bundle_digest().is_some()
        || assets.runtime_pin.storage().bytes != profile.runtime_library.bytes
        || digest_string(&profile.runtime_library.sha256)
            != Some(assets.runtime_pin.binary_digest())
    {
        return Err(fault("runtime_asset", "actual verified CPU library pin differs").into());
    }
    verify_file(assets.export, &profile.export_manifest, clock, cancel)?;
    clock
        .check(cancel)
        .map_err(|_| fault("assets_clock", "original execution deadline/cancel"))?;
    Ok(profile)
}
/// Checks original bounded profile bytes and all independently expected facts.
/// The returned value is an owned declaration, not native admission authority.
/// This performs no file IO, source/binary registration, runtime-pin admission,
/// provider/session load or model execution. The actual configuration and
/// source/runtime/export checks remain in the existing `admit_assets` path.
pub fn check_cpu_fresh_asset_profile(
    raw: &[u8],
    expected_pin: &ArtifactPin,
    expected: &CpuFreshAssetProfile,
) -> Result<CpuFreshAssetProfile, CpuFreshAssetError> {
    decode_profile(raw, expected_pin, expected)
}
fn decode_profile(
    raw: &[u8],
    expected_pin: &ArtifactPin,
    expected: &CpuFreshAssetProfile,
) -> Result<CpuFreshAssetProfile, CpuFreshAssetError> {
    if raw.is_empty() || raw.len() > MAX_PROFILE_BYTES || !pin_matches(expected_pin, raw) {
        return Err(fault(
            "asset_profile_pin",
            "independent original profile byte pin differs",
        )
        .into());
    }
    let profile: CpuFreshAssetProfile = serde_json::from_slice(raw)
        .map_err(|e| CpuFreshAssetError::json("asset_profile_json", e))?;
    if !same_profile(&profile, expected) {
        return Err(fault("asset_profile", "independent expected profile differs").into());
    }
    validate_profile(&profile)?;
    if profile.context_sha256 != asset_profile_context(&profile)? {
        return Err(fault("asset_profile_context", "domain context differs").into());
    }
    Ok(profile)
}
fn validate_config(config: PalsOnnxConfig, p: &CpuFreshAssetProfile) -> Result<(), AdmissionFault> {
    if config.provider != Provider::Cpu
        || config.intra_threads != p.intra_threads
        || config.cache_public_memory != p.cache_public_memory
        || config.device_public_memory != p.device_public_memory
    {
        return Err(fault(
            "provider",
            "actual explicit CPU/config selection differs",
        ));
    }
    Ok(())
}
fn validate_sources(p: &ReplayRegisteredArtifacts) -> Result<(), AdmissionFault> {
    let actual: [(&ArtifactPin, &[u8]); 4] = [
        (&p.wrapper_source, include_bytes!("native_replay.rs")),
        (
            &p.provider_factory_source,
            include_bytes!("../../pals_native.rs"),
        ),
        (
            &p.engine_source,
            include_bytes!("../../../../rz-search/src/pals/engine.rs"),
        ),
        (
            &p.replay_source,
            include_bytes!("../../../../rz-search/src/pals/engine/replay.rs"),
        ),
    ];
    let actual_factory_digest: [u8; 32] = Sha256::digest(actual[1].1).into();
    if actual.iter().any(|(pin, bytes)| !pin_matches(pin, bytes))
        || p.opponent_recheck_source.as_ref().is_some_and(|pin| {
            !pin_matches(
                pin,
                include_bytes!("../../../../rz-search/src/pals/engine/replay/opponent_recheck.rs"),
            )
        })
        || actual_factory_digest != pals_native::pals_native_source_digest()
        || p.legacy_cpu_binary.sha256 == p.replay_binary.sha256
    {
        return Err(fault(
            "source_registration",
            "actual source pin or separate binary registration differs",
        ));
    }
    Ok(())
}
fn validate_profile(p: &CpuFreshAssetProfile) -> Result<(), AdmissionFault> {
    if p.schema != ASSET_PROFILE_SCHEMA
        || p.domain != "cpu_fresh"
        || p.provider != "cpu"
        || !(1..=2).contains(&p.intra_threads)
        || p.device_public_memory
        || p.export_manifest.bytes == 0
        || p.export_manifest.bytes > MAX_MANIFEST_BYTES
        || p.runtime_library.bytes == 0
        || p.runtime_library.bytes > MAX_RUNTIME_BYTES
        || digest_string(&p.export_manifest.sha256).is_none()
        || digest_string(&p.runtime_library.sha256).is_none()
        || digest_string(&p.model_epoch).is_none()
        || digest_string(&p.context_sha256).is_none()
        || digest_string(&p.encoding_semantic_sha256)
            != Some(pals_native::pals_fresh_encoding_semantic_digest())
    {
        return Err(fault(
            "asset_profile",
            "unsupported explicit Fresh CPU asset/config/encoding declaration",
        ));
    }
    Ok(())
}
/// Pure domain seal; caller stores the original profile bytes/pin separately.
pub fn asset_profile_context(p: &CpuFreshAssetProfile) -> Result<String, AdmissionFault> {
    if p.schema.len() > 64
        || p.domain.len() > 32
        || p.provider.len() > 32
        || [
            &p.export_manifest.sha256,
            &p.runtime_library.sha256,
            &p.model_epoch,
            &p.encoding_semantic_sha256,
        ]
        .iter()
        .any(|s| s.len() != 64)
    {
        return Err(fault(
            "asset_profile_context",
            "finite seal field extent differs",
        ));
    }
    #[derive(Serialize)]
    struct Body<'a> {
        schema: &'a str,
        domain: &'a str,
        provider: &'a str,
        export_manifest: &'a ArtifactPin,
        runtime_library: &'a ArtifactPin,
        model_epoch: &'a str,
        encoding_semantic_sha256: &'a str,
        intra_threads: usize,
        cache_public_memory: bool,
        device_public_memory: bool,
    }
    let body = Body {
        schema: &p.schema,
        domain: &p.domain,
        provider: &p.provider,
        export_manifest: &p.export_manifest,
        runtime_library: &p.runtime_library,
        model_epoch: &p.model_epoch,
        encoding_semantic_sha256: &p.encoding_semantic_sha256,
        intra_threads: p.intra_threads,
        cache_public_memory: p.cache_public_memory,
        device_public_memory: p.device_public_memory,
    };
    let mut bytes = reserved_bytes(MAX_PROFILE_BYTES)?;
    serialize_into(
        &mut bytes,
        MAX_PROFILE_BYTES,
        None,
        &(ASSET_PROFILE_SCHEMA, body),
    )?;
    // Reuse the existing sorted canonical domain, after bounded serialization
    // has capped all strings and objects. This does not replace the raw pin.
    let value = serde_json::from_slice(&bytes)
        .map_err(|_| fault("asset_profile_context", "bounded seal parse failed"))?;
    let canonical = super::super::canonical(&value)
        .map_err(|_| fault("asset_profile_context", "canonical seal failed"))?;
    if canonical.len() > MAX_PROFILE_BYTES {
        return Err(fault(
            "asset_profile_context",
            "canonical seal exceeds bound",
        ));
    }
    Ok(lower_hex(Sha256::digest(&canonical).into()))
}
fn same_profile(a: &CpuFreshAssetProfile, b: &CpuFreshAssetProfile) -> bool {
    a.schema == b.schema
        && a.domain == b.domain
        && a.provider == b.provider
        && same_pin(&a.export_manifest, &b.export_manifest)
        && same_pin(&a.runtime_library, &b.runtime_library)
        && a.model_epoch == b.model_epoch
        && a.encoding_semantic_sha256 == b.encoding_semantic_sha256
        && a.intra_threads == b.intra_threads
        && a.cache_public_memory == b.cache_public_memory
        && a.device_public_memory == b.device_public_memory
        && a.context_sha256 == b.context_sha256
}
fn same_pin(a: &ArtifactPin, b: &ArtifactPin) -> bool {
    a.bytes == b.bytes && a.sha256 == b.sha256
}
fn pin_matches(p: &ArtifactPin, bytes: &[u8]) -> bool {
    p.bytes == bytes.len() as u64 && digest_string(&p.sha256) == Some(Sha256::digest(bytes).into())
}
fn verify_file(
    path: &Path,
    p: &ArtifactPin,
    clock: Clock,
    cancel: &AtomicBool,
) -> Result<(), CpuFreshAssetError> {
    if !path.is_absolute() {
        return Err(fault("export_asset", "absolute explicit export manifest required").into());
    }
    let named = std::fs::symlink_metadata(path)
        .map_err(|e| CpuFreshAssetError::io("export_asset_metadata", e))?;
    if !named.is_file() || named.file_type().is_symlink() || named.len() != p.bytes {
        return Err(fault("export_asset", "manifest extent/type differs").into());
    }
    let mut file = File::open(path).map_err(|e| CpuFreshAssetError::io("export_asset_open", e))?;
    if file
        .metadata()
        .map_err(|e| CpuFreshAssetError::io("export_asset_open_metadata", e))?
        .len()
        != p.bytes
    {
        return Err(fault("export_asset", "opened manifest extent differs").into());
    }
    let mut count = 0_u64;
    let mut hash = Sha256::new();
    let mut chunk = [0_u8; 8192];
    loop {
        clock
            .check(cancel)
            .map_err(|_| fault("export_asset_clock", "original execution deadline/cancel"))?;
        let n = file
            .read(&mut chunk)
            .map_err(|e| CpuFreshAssetError::io("export_asset_read", e))?;
        if n == 0 {
            break;
        }
        count = count
            .checked_add(n as u64)
            .ok_or(fault("export_asset", "manifest count overflow"))?;
        if count > p.bytes || count > MAX_MANIFEST_BYTES {
            return Err(fault("export_asset", "manifest grew beyond declared bound").into());
        }
        hash.update(&chunk[..n]);
    }
    if count != p.bytes || digest_string(&p.sha256) != Some(hash.finalize().into()) {
        return Err(fault("export_asset", "actual manifest byte pin differs").into());
    }
    Ok(())
}

struct DebugRef<'a, T: fmt::Debug + ?Sized>(&'a T);
impl<T: fmt::Debug + ?Sized> Serialize for DebugRef<'_, T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(&format_args!("{:?}", self.0))
    }
}
struct TextWriter<'a> {
    text: &'a mut String,
    limit: usize,
}
impl fmt::Write for TextWriter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self
            .text
            .len()
            .checked_add(s.len())
            .is_none_or(|n| n > self.limit)
        {
            return Err(fmt::Error);
        }
        self.text.push_str(s);
        Ok(())
    }
}
struct ByteWriter<'a> {
    bytes: &'a mut Vec<u8>,
    limit: usize,
    until: Option<Instant>,
}
impl Write for ByteWriter<'_> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if self.until.is_some_and(|until| Instant::now() >= until)
            || self
                .bytes
                .len()
                .checked_add(b.len())
                .is_none_or(|n| n > self.limit || n > self.bytes.capacity())
        {
            return Err(io::Error::other("bounded replay output/clock"));
        }
        self.bytes.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn serialize_into<T: Serialize>(
    out: &mut Vec<u8>,
    limit: usize,
    until: Option<Instant>,
    value: &T,
) -> Result<(), AdmissionFault> {
    serde_json::to_writer(
        ByteWriter {
            bytes: out,
            limit,
            until,
        },
        value,
    )
    .map_err(|_| {
        fault(
            "serialization",
            "finite serializer/clock refused; original observations retained",
        )
    })
}
fn append_prepared_tick(
    out: &mut Vec<u8>,
    clock: Clock,
    elapsed_ms: u64,
) -> Result<(), AdmissionFault> {
    if out.last() != Some(&b'}') {
        return Err(fault(
            "serialization",
            "owned observation object invariant failed",
        ));
    }
    out.pop();
    write!(
        ByteWriter {
            bytes: out,
            limit: clock.output,
            until: Some(clock.whole)
        },
        ",\"bytes_prepared_elapsed_ms\":{elapsed_ms}}}"
    )
    .map_err(|_| {
        fault(
            "serialization",
            "final original-clock timing member refused",
        )
    })
}
fn reserve_vec<T>(v: &mut Vec<T>, limit: usize) -> Result<(), AdmissionFault> {
    v.try_reserve_exact(limit)
        .map_err(|_| fault("allocation", "fallible retained reservation failed"))?;
    if v.capacity() > limit {
        return Err(fault(
            "allocation",
            "allocator retained capacity exceeds declared bound",
        ));
    }
    Ok(())
}
fn reserved_bytes(limit: usize) -> Result<Vec<u8>, AdmissionFault> {
    let mut v = Vec::new();
    reserve_vec(&mut v, limit)?;
    Ok(v)
}
fn reserved_text(limit: usize) -> Result<String, AdmissionFault> {
    let mut s = String::new();
    s.try_reserve_exact(limit)
        .map_err(|_| fault("allocation", "text reservation failed"))?;
    if s.capacity() > limit {
        return Err(fault(
            "allocation",
            "retained text capacity exceeds declared bound",
        ));
    }
    Ok(s)
}
fn milliseconds(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}
fn lower_hex(d: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in d {
        s.push(HEX[usize::from(b >> 4)] as char);
        s.push(HEX[usize::from(b & 15)] as char);
    }
    s
}
fn digest_string(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let mut out = [0; 32];
    for (at, bytes) in s.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        out[at] = nibble(bytes[0]) * 16 + nibble(bytes[1]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_native::{NativeLoadStage, NativeRoleCompletionUnknown};
    use replay_inputs::{ReplayBindingPins, ReplayParentPins};
    use rz_contracts::{ExecutionId, ProcessEpoch};
    use rz_search::pals::engine::replay::repair_replay_requirements;

    // Admission/observation fixtures never load ORT or launch a child. The
    // explicit dispatch/capture test uses a deterministic RoleModel plus the
    // real bounded CPU checker; it proves no physical native execution.
    fn pin(bytes: &[u8]) -> ArtifactPin {
        ArtifactPin {
            bytes: bytes.len() as u64,
            sha256: lower_hex(Sha256::digest(bytes).into()),
        }
    }
    fn sources() -> ReplayRegisteredArtifacts {
        ReplayRegisteredArtifacts {
            legacy_cpu_binary: pin(b"controlled_old_binary"),
            replay_binary: pin(b"controlled_new_binary"),
            engine_source: pin(include_bytes!("../../../../rz-search/src/pals/engine.rs")),
            wrapper_source: pin(include_bytes!("native_replay.rs")),
            replay_source: pin(include_bytes!(
                "../../../../rz-search/src/pals/engine/replay.rs"
            )),
            provider_factory_source: pin(include_bytes!("../../pals_native.rs")),
            opponent_recheck_source: None,
        }
    }

    #[test]
    fn opponent_source_is_separately_compared_with_compiled_source_bytes() {
        let mut registered = sources();
        assert!(validate_sources(&registered).is_ok());
        registered.opponent_recheck_source = Some(pin(include_bytes!(
            "../../../../rz-search/src/pals/engine/replay/opponent_recheck.rs"
        )));
        assert!(validate_sources(&registered).is_ok());
        registered.opponent_recheck_source.as_mut().unwrap().sha256 = "0".repeat(64);
        assert!(validate_sources(&registered).is_err());
    }

    #[test]
    fn opponent_output_profile_fits_five_ply_without_expanding_original_cap() {
        let req =
            replay_mode_requirements(ReplayInputMode::RepairOpponent4n, 5, 1, 100_000).unwrap();
        let roles = usize::try_from(req.actual_role_calls()).unwrap();
        let declared = replay_output_requirements(roles, req.cpu_checks()).unwrap();
        assert!(declared.required_output_bytes() <= replay_inputs::MAX_OUTPUT_BYTES);
        assert!(
            declared
                .check_output_limit(declared.required_output_bytes() - 1)
                .is_err()
        );
        assert!(Reserved::new(roles, 4, clock(declared.required_output_bytes() - 1)).is_err());
        let reserved = Reserved::new(roles, 4, clock(replay_inputs::MAX_OUTPUT_BYTES)).unwrap();
        assert_eq!(reserved.slots.len(), 4);
        assert!(reserved.opponent_state.is_some());
        let collector = reserved.collector.lock().unwrap();
        assert_eq!(collector.limit, roles * OPPONENT_ROLE_TRACE_BYTES);
        assert_eq!(collector.large_row_bytes, OPPONENT_LARGE_ROW_BYTES);
        assert_eq!(collector.row_limit, roles * ROWS_PER_ROLE);
        const {
            assert!(
                2 * OPPONENT_LARGE_ROW_BYTES + CONTEXT_BYTES + 9 * SMALL_ROW_BYTES + ROWS_PER_ROLE
                    <= OPPONENT_ROLE_TRACE_BYTES
            );
        }
        for checks in [2, 3] {
            let old = replay_output_requirements(5, checks).unwrap();
            assert_eq!(
                old.required_output_bytes(),
                5 * ROLE_TRACE_BYTES * 2
                    + (checks * (STAGE_TEXT_BYTES + 2 * SNAPSHOT_BYTES)
                        + STATE_TEXT_BYTES
                        + OUTCOME_TEXT_BYTES)
                        * 6
                    + HEADER_RESERVE
            );
            assert!(
                Reserved::new(5, checks, clock(replay_inputs::MAX_OUTPUT_BYTES))
                    .unwrap()
                    .opponent_state
                    .is_none()
            );
        }
    }

    #[test]
    fn opponent_observer_overflow_preserves_failure_without_retry_or_success() {
        let mut collector = Collector::reserved_for_checks(1, 4, clock(1024).whole).unwrap();
        let cap = collector.large_row_bytes;
        assert!(collector.row(&"x".repeat(cap + 1), cap).is_err());
        assert!(collector.failure.is_some());
        assert!(collector.bytes.is_empty());
        assert!(collector.row(&"small", SMALL_ROW_BYTES).is_err());
        assert_eq!(collector.rows, 0);
    }

    #[test]
    fn actual_mode_dispatch_and_capture_keep_fourth_cpu_separate_from_native_ids() {
        use rz_position::{BoardMove, Position};
        use rz_search::cpu::{CpuConfig, CpuProfile};
        use rz_search::pals::engine::replay::{DefendResponseReplayPlan, ReplayHistoricalPins};
        use rz_search::pals::engine::{
            DivergenceQuery, PalsConfig, RecordKind, RoleEvaluation, RoleQuery,
        };
        fn line(text: &[&str]) -> Vec<BoardMove> {
            text.iter()
                .map(|text| BoardMove::from_uci(text).unwrap())
                .collect()
        }
        struct Roles;
        impl Roles {
            fn evaluate(query: RoleQuery<'_>, text: &[&str]) -> Result<RoleEvaluation, RoleError> {
                let movement = BoardMove::from_uci(text[query.prefix.len()]).unwrap();
                let at = query
                    .legal
                    .iter()
                    .position(|m| *m == movement)
                    .ok_or(RoleError::InvalidOutput)?;
                let mut logits = vec![0.0; query.legal.len()];
                logits[at] = 1.0;
                Ok(RoleEvaluation {
                    logits,
                    wdl: [0.25, 0.5, 0.25],
                })
            }
        }
        impl RoleModel for Roles {
            fn identity(&self) -> &str {
                "wrapper-controlled-opponent-role/1"
            }
            fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                Self::evaluate(query, &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"])
            }
            fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                let after_repair = query
                    .records
                    .iter()
                    .any(|record| record.kind == RecordKind::Repair);
                Self::evaluate(
                    query,
                    if after_repair {
                        &["e2e4", "e7e6", "d2d3", "c7c5", "b1c3"]
                    } else {
                        &["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"]
                    },
                )
            }
            fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
                Self::evaluate(query, &["e2e4", "e7e6", "d2d3", "d7d5", "g1f3"])
            }
            fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
                Err(RoleError::Unavailable)
            }
        }
        let root = Position::startpos();
        let prefix = line(&["e2e4"]);
        let mut target = root.clone();
        target.make_move(prefix[0]).unwrap();
        let mut owner = FreshReplayOwner::new(
            PalsConfig {
                line_plies: 5,
                max_records: 5,
                cpu_nodes_per_task: 100_000,
                ..PalsConfig::default()
            },
            Roles,
            root.clone(),
            DefendResponseReplayPlan {
                historical: ReplayHistoricalPins {
                    parent_input_sha256: [1; 32],
                    current_view_sha256: [2; 32],
                    frozen_admission_sha256: [3; 32],
                    query_sha256: [4; 32],
                    catalogue_sha256: [5; 32],
                    before_result_sha256: [6; 32],
                    semantic_input_sha256: [7; 32],
                    cpu_request_sha256: [8; 32],
                },
                expected_root: root.snapshot(),
                expected_target: target.snapshot(),
                root_legal_order: root.legal_moves(),
                target_legal_order: target.legal_moves(),
                prefix,
                response_restriction: line(&["c7c5"]),
                claimed_line: line(&["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]),
                baseline_depth: 1,
                requested_depth: 2,
                nodes_per_check: 100_000,
                cpu: CpuConfig {
                    profile: CpuProfile::PlanAssisted,
                    tt_entries: 16,
                    max_depth: 2,
                    quiescence_ply: 16,
                },
            },
        )
        .unwrap();
        let c = clock(replay_inputs::MAX_OUTPUT_BYTES);
        let mode = ReplayInputMode::RepairOpponent4n;
        let outcome = run_owner(
            &mut owner,
            mode,
            PalsLimits {
                deadline: c.execution,
                max_rounds: 1,
                max_cpu_nodes: 400_000,
                cpu_depth: 2,
            },
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(matches!(
            outcome,
            Outcome::Opponent(ReplayOpponentOutcome::CompletedOpponentEndpoint { .. })
        ));
        let mut observation = receipt(c);
        observation.schema = OPPONENT_SCHEMA;
        observation.mode = mode;
        let mut reserved = Reserved::new(14, 4, c).unwrap();
        observation.opponent_state = reserved.opponent_state;
        assert!(capture_owner(&owner, &mut reserved.slots, &mut observation));
        assert_eq!(observation.stages.len(), 4);
        assert_eq!(observation.stages[2].phase, "repair_endpoint");
        assert_eq!(observation.stages[3].phase, "repair_opponent_endpoint");
        assert!(
            observation.stages[3]
                .raw_task
                .text
                .contains("rz-pals-frozen-parent-repair-opponent-continuation-replay/1")
        );
        assert_ne!(
            observation.stages[2].cpu_execution,
            observation.stages[3].cpu_execution
        );
        assert_eq!(observation.cpu_tasks_requested, Some(4));
        assert_eq!(
            owner.opponent_counterline(),
            line(&["e2e4", "e7e6", "d2d3", "c7c5", "b1c3"])
        );
        assert!(observation.opponent_state.as_ref().unwrap().complete);
        assert!(
            observation
                .replay_state
                .text
                .contains("repair_endpoint_only")
        );
        assert_eq!(observation.native_bindings_observed, 0);
        assert!(observation.native_receipt.is_none() && observation.native_cleanup_missing);
        assert_eq!(observation.authorities, ReplayAuthorities::default());
    }
    fn profile(cache: bool) -> CpuFreshAssetProfile {
        let mut p = CpuFreshAssetProfile {
            schema: ASSET_PROFILE_SCHEMA.into(),
            domain: "cpu_fresh".into(),
            provider: "cpu".into(),
            export_manifest: pin(b"controlled_export_manifest"),
            runtime_library: pin(b"controlled_runtime_library"),
            model_epoch: lower_hex([7; 32]),
            encoding_semantic_sha256: lower_hex(pals_native::pals_fresh_encoding_semantic_digest()),
            intra_threads: 1,
            cache_public_memory: cache,
            device_public_memory: false,
            context_sha256: String::new(),
        };
        p.context_sha256 = asset_profile_context(&p).unwrap();
        p
    }
    fn clock(output: usize) -> Clock {
        let started = Instant::now();
        Clock {
            started,
            execution: started + Duration::from_secs(30),
            whole: started + Duration::from_secs(60),
            output,
        }
    }
    fn binding(
        request_epoch: u64,
        execution_epoch: u64,
        sequence: u64,
    ) -> NativeRoleExecutionBinding {
        NativeRoleExecutionBinding {
            request: RequestId::new(ProcessEpoch(request_epoch), sequence),
            execution: ExecutionId::new(ProcessEpoch(execution_epoch), sequence),
        }
    }
    fn receipt(c: Clock) -> NativeReplayObservation {
        let empty = pin(b"controlled_fixture");
        let digest = empty.sha256.clone();
        let parent = ReplayParentPins {
            parent_input_sha256: digest.clone(),
            current_view_sha256: digest.clone(),
            frozen_admission_sha256: digest.clone(),
            encoding_sha256: digest.clone(),
        };
        let binding = ReplayBindingPins {
            query_sha256: digest.clone(),
            catalogue_artifact: empty.clone(),
            before_result_artifact: empty.clone(),
            prior_ledger_sha256: digest.clone(),
            semantic_input_sha256: digest.clone(),
            semantic_context_sha256: digest.clone(),
            semantic_branch_meaning_sha256: digest.clone(),
            semantic_before_result_anchor_sha256: digest.clone(),
            cpu_request_artifact: empty.clone(),
        };
        let audit = ReplayInputAudit {
            schema: replay_inputs::SCHEMA,
            scope: replay_inputs::SCOPE,
            expected_semantic_receipt_producer_scope: None,
            mode: ReplayInputMode::ReplyOnly2n,
            input_artifact: empty.clone(),
            registration_artifact: empty.clone(),
            prepared_action_artifact: empty.clone(),
            semantic_receipt_artifact: empty.clone(),
            parent,
            binding,
            registered_artifacts: sources(),
            provider_factory_id: FACTORY_ID.into(),
            legacy_cpu_profile_sha256: digest,
            binary_pin_scope: replay_inputs::BINARY_DECLARATION_SCOPE,
            cpu_allowance: 64,
            conservative_repair_role_store_overreservation: true,
            whole_wall_ms: 60_000,
            cleanup_reserve_ms: 30_000,
            elapsed_ms: 0,
            cpu_checks: 0,
            cpu_engine_created: false,
            model_created: false,
            provider_created: false,
            actual_utility_groups: 0,
            authorities: ReplayAuthorities::default(),
        };
        let mut r = NativeReplayObservation {
            schema: SCHEMA,
            scope: SCOPE,
            status: "controlled_fixture",
            mode: ReplayInputMode::ReplyOnly2n,
            input_admission: audit,
            assets_source_admitted: false,
            asset_profile_artifact: empty,
            binary_pin_scope: replay_inputs::BINARY_DECLARATION_SCOPE,
            model_returned: false,
            owner_created: false,
            replay_attempted: false,
            replay_returned_ok: false,
            elapsed_ms: 0,
            elapsed_scope: "original_start_through_cleanup_and_observation_capture",
            bytes_prepared_elapsed_ms: None,
            bytes_prepared_scope: "original_start_through_main_encoding_before_final_timing_member_and_caller_delivery",
            original_whole_wall_ms: 60_000,
            cleanup_reserve_ms: 30_000,
            execution_deadline_exceeded: false,
            deadline_exceeded: false,
            canceled: false,
            cpu_tasks_requested: None,
            cpu_tasks_accounted: None,
            cpu_reports_returned: None,
            cpu_nodes_lower_bound: None,
            cpu_work_observation_incomplete: None,
            stages: Vec::new(),
            replay_state: SnapshotText::reserved(STATE_TEXT_BYTES).unwrap(),
            opponent_state: None,
            outcome: SnapshotText::reserved(OUTCOME_TEXT_BYTES).unwrap(),
            trace_jsonl: String::new(),
            trace_rows: 0,
            trace_complete: true,
            native_receipt: None,
            native_bindings_observed: 0,
            native_completion_unknown_observed: 0,
            native_ready_observed: 0,
            native_terminal_missing: 0,
            cleanup_attempted: false,
            cleanup_returned_ok: false,
            native_cleanup_missing: true,
            authorities: ReplayAuthorities::default(),
        };
        c.stamp(&mut r, &AtomicBool::new(false));
        r
    }

    #[test]
    fn independent_profile_and_exact_public_cache_selection_are_required() {
        for cache in [false, true] {
            let p = profile(cache);
            let raw = serde_json::to_vec(&p).unwrap();
            assert!(decode_profile(&raw, &pin(&raw), &p).is_ok());
            let cfg = PalsOnnxConfig {
                provider: Provider::Cpu,
                intra_threads: 1,
                cache_public_memory: cache,
                device_public_memory: false,
            };
            assert!(validate_config(cfg, &p).is_ok());
            assert!(
                validate_config(
                    PalsOnnxConfig {
                        cache_public_memory: !cache,
                        ..cfg
                    },
                    &p
                )
                .is_err()
            );
            let mut expected = p.clone();
            expected.runtime_library = pin(b"independent_other_runtime");
            assert!(decode_profile(&raw, &pin(&raw), &expected).is_err());
        }
    }
    #[test]
    fn closed_profile_rejects_duplicates_unknown_null_fraction_and_bool_numbers() {
        let p = profile(false);
        let text = serde_json::to_string(&p).unwrap();
        let bad = [
            text.replacen('{', "{\"intra_threads\":1,", 1),
            text.replacen('{', "{\"undeclared_mode\":false,", 1),
            text.replace("\"intra_threads\":1", "\"intra_threads\":null"),
            text.replace("\"intra_threads\":1", "\"intra_threads\":1.5"),
            text.replace("\"intra_threads\":1", "\"intra_threads\":true"),
        ];
        for raw in bad {
            let e = decode_profile(raw.as_bytes(), &pin(raw.as_bytes()), &p).unwrap_err();
            assert_eq!(e.fault.stage, "asset_profile_json");
            assert!(matches!(
                e.source.as_deref(),
                Some(CpuFreshAssetSourceError::Json(_))
            ));
        }
    }
    #[test]
    fn original_profile_bytes_and_domain_seal_cannot_be_self_repinned() {
        let p = profile(false);
        let mut raw = serde_json::to_vec(&p).unwrap();
        let expected_pin = pin(&raw);
        raw.push(b' ');
        assert!(decode_profile(&raw, &expected_pin, &p).is_err());
        let mut changed = p.clone();
        changed.cache_public_memory = true;
        let changed_raw = serde_json::to_vec(&changed).unwrap();
        assert!(decode_profile(&changed_raw, &pin(&changed_raw), &changed).is_err());
        let canonical = super::super::super::canonical(&serde_json::json!([ASSET_PROFILE_SCHEMA, {
            "schema": p.schema, "domain": p.domain, "provider": p.provider,
            "export_manifest": p.export_manifest, "runtime_library": p.runtime_library,
            "model_epoch": p.model_epoch, "encoding_semantic_sha256": p.encoding_semantic_sha256,
            "intra_threads": p.intra_threads, "cache_public_memory": p.cache_public_memory,
            "device_public_memory": p.device_public_memory }]))
        .unwrap();
        assert_eq!(
            p.context_sha256,
            lower_hex(Sha256::digest(canonical).into())
        );
    }
    #[test]
    fn actual_sources_and_distinct_legacy_binary_registration_are_required() {
        let p = sources();
        assert!(validate_sources(&p).is_ok());
        for field in ["wrapper", "engine", "replay", "factory", "binary"] {
            let mut changed = p.clone();
            match field {
                "wrapper" => changed.wrapper_source = pin(b"different_wrapper"),
                "engine" => changed.engine_source = pin(b"different_engine"),
                "replay" => changed.replay_source = pin(b"different_replay"),
                "factory" => changed.provider_factory_source = pin(b"different_factory"),
                _ => changed.replay_binary = changed.legacy_cpu_binary.clone(),
            }
            assert!(validate_sources(&changed).is_err());
        }
        let mut same_sha = p.clone();
        same_sha.replay_binary.sha256 = same_sha.legacy_cpu_binary.sha256.clone();
        same_sha.replay_binary.bytes = same_sha.legacy_cpu_binary.bytes.checked_add(1).unwrap();
        assert_ne!(
            same_sha.replay_binary.bytes,
            same_sha.legacy_cpu_binary.bytes
        );
        assert!(validate_sources(&same_sha).is_err());
        assert!(
            FACTORY_ID
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        );
        assert!(!FACTORY_ID.contains('/'));
    }
    #[test]
    fn cuda_device_warm_and_wrong_encoding_are_not_fresh_cpu_assets() {
        let p = profile(false);
        let cfg = PalsOnnxConfig {
            provider: Provider::Cpu,
            intra_threads: 1,
            cache_public_memory: false,
            device_public_memory: false,
        };
        assert!(
            validate_config(
                PalsOnnxConfig {
                    provider: Provider::Cuda {
                        device_id: 0,
                        arena_bytes: 1024
                    },
                    ..cfg
                },
                &p
            )
            .is_err()
        );
        for kind in ["device", "warm", "encoding", "threads"] {
            let mut changed = p.clone();
            match kind {
                "device" => changed.device_public_memory = true,
                "warm" => changed.domain = "private_warm".into(),
                "encoding" => changed.encoding_semantic_sha256 = lower_hex([9; 32]),
                _ => changed.intra_threads = 3,
            }
            assert!(validate_profile(&changed).is_err());
        }
    }
    #[test]
    fn full_fresh_encoding_identity_rejects_rules_only_seal() {
        let p = profile(false);
        let full = pals_native::pals_fresh_encoding_semantic_digest();
        let rules = pals_native::pals_rules_encoding_semantic_digest();
        assert_ne!(full, rules);
        assert_eq!(digest_string(&p.encoding_semantic_sha256), Some(full));
        assert!(validate_profile(&p).is_ok());
        let mut changed = p.clone();
        changed.encoding_semantic_sha256 = lower_hex(rules);
        changed.context_sha256 = asset_profile_context(&changed).unwrap();
        let raw = serde_json::to_vec(&changed).unwrap();
        let refused = decode_profile(&raw, &pin(&raw), &changed).unwrap_err();
        assert_eq!(refused.fault.stage, "asset_profile");
        // Re-pinning a manifest's Rules-only namespace cannot change the
        // independently checked full Fresh owner/model input namespace.
        assert_eq!(p.encoding_semantic_sha256, lower_hex(full));
    }
    #[test]
    fn pre_load_reservation_bounds_include_stages_and_escape_expansion() {
        let c = clock(replay_inputs::MAX_OUTPUT_BYTES);
        for (roles, checks) in [(3, 2), (3, 3), (5, 2), (9, 3)] {
            let r = Reserved::new(roles, checks, c).unwrap();
            assert_eq!(r.slots.capacity(), checks);
            assert!(r.output.capacity() <= c.output);
            assert!(r.state.text.capacity() <= STATE_TEXT_BYTES);
            assert!(r.collector.lock().unwrap().bindings.capacity() <= roles);
        }
        let normal = rz_search::pals::engine::PalsConfig::default();
        let repair =
            repair_replay_requirements(normal.line_plies, 1, normal.cpu_nodes_per_task).unwrap();
        assert!(Reserved::new(usize::try_from(repair.role_calls()).unwrap(), 3, c).is_ok());
        assert!(matches!(
            Reserved::new(3, 2, clock(1024)),
            Err(AdmissionFault {
                stage: "output_reserve",
                ..
            })
        ));
        assert!(Reserved::new(usize::MAX, 3, c).is_err());
    }
    #[test]
    fn bounded_debug_and_failed_observer_row_keep_prior_owned_bytes() {
        struct ManyWrites;
        impl fmt::Debug for ManyWrites {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                for _ in 0..10_000 {
                    f.write_str("payload")?;
                }
                Ok(())
            }
        }
        let mut snapshot = SnapshotText::reserved(28).unwrap();
        assert!(!snapshot.capture(&ManyWrites));
        assert!(!snapshot.complete);
        assert!(!snapshot.text.is_empty());
        assert!(snapshot.text.len() <= 28);
        let mut c = Collector::reserved(1, clock(1024).whole).unwrap();
        c.row(&"first_original_row", 128).unwrap();
        let before = c.bytes.clone();
        assert!(c.row(&DebugRef(&ManyWrites), 32).is_err());
        assert_eq!(c.bytes, before);
        assert_eq!(c.rows, 1);
        assert!(c.failure.is_some());
        let mut escaped = Collector::reserved(1, clock(1024).whole).unwrap();
        escaped
            .row(&"quotes\" slash\\ null\0 newline\n unicode한", 512)
            .unwrap();
        let text = std::str::from_utf8(&escaped.bytes).unwrap();
        assert!(serde_json::to_vec(text).unwrap().len() <= 2 * escaped.bytes.len() + 2);
    }
    #[test]
    fn actual_binding_unknown_and_late_ready_remain_separate_from_consumption() {
        let mut c = Collector::reserved(1, clock(1024).whole).unwrap();
        let b = binding(11, 22, 7);
        c.dispatched(b).unwrap();
        c.terminal(
            b,
            NativeRoleTerminal::CompletionUnknown(NativeRoleCompletionUnknown::Quarantined),
        )
        .unwrap();
        let backend = rz_eval::error::BackendError::new(
            rz_eval::error::FailureKind::BackendFailure,
            rz_eval::error::FailureStage::Backend,
            "controlled terminal failure",
        );
        c.terminal(b, NativeRoleTerminal::Ready(Err(&backend)))
            .unwrap();
        assert!(c.bindings[0].unknown && c.bindings[0].ready);
        assert_eq!(c.rows, 3);
        assert_ne!(b.request.epoch, b.execution.epoch);
        let rows = std::str::from_utf8(&c.bytes).unwrap();
        assert!(!rows.contains("accepted_context"));
        assert!(!rows.contains("actual_consumption_callback"));
        let first: serde_json::Value = serde_json::from_str(rows.lines().next().unwrap()).unwrap();
        assert_eq!(first["binding"]["request"]["epoch"], 11);
        assert_eq!(first["binding"]["native_execution"]["epoch"], 22);
        assert_eq!(
            first["binding"]["execution_namespace"],
            "rz_runtime_physical_execution_not_search_store_execution"
        );
        let before = c.bytes.clone();
        assert!(
            c.terminal(b, NativeRoleTerminal::Ready(Err(&backend)))
                .is_err()
        );
        assert_eq!(c.bytes, before);
        let mut other = Collector::reserved(2, clock(1024).whole).unwrap();
        other.dispatched(b).unwrap();
        let different_epoch = binding(11, 33, 7);
        assert!(
            other
                .terminal(
                    different_epoch,
                    NativeRoleTerminal::CompletionUnknown(NativeRoleCompletionUnknown::Consumed)
                )
                .is_err()
        );
        let output = PalsNativeResult::Evaluation(PalsRawOutput {
            candidate_logits: vec![f32::from_bits(0x7fc0_1234), -0.0],
            wdl_logits: [0.0; 3],
            divergence_logits: None,
            task_logits: None,
            private_latent: vec![
                0.0;
                rz_eval::pals_model::PalsModelConfig::baseline().latent_elements()
            ],
        });
        let mut bits = Collector::reserved(1, clock(1024).whole).unwrap();
        bits.dispatched(b).unwrap();
        bits.terminal(b, NativeRoleTerminal::Ready(Ok(&output)))
            .unwrap();
        let text = std::str::from_utf8(&bits.bytes).unwrap();
        assert!(text.contains("7fc0123480000000"));
        assert!(text.contains("len=6144;u32hex="));
        assert!(bits.terminal_scratch.complete);
        assert_eq!(bits.rows, 2);
        let mut pending = Collector::reserved(1, clock(1024).whole).unwrap();
        pending.dispatched(b).unwrap();
        pending
            .terminal(
                b,
                NativeRoleTerminal::CompletionUnknown(NativeRoleCompletionUnknown::DrainExpired),
            )
            .unwrap();
        let mut r = receipt(clock(4096));
        assert!(drain_collector(&Arc::new(Mutex::new(pending)), &mut r).is_none());
        assert_eq!(r.native_bindings_observed, 1);
        assert_eq!(r.native_completion_unknown_observed, 1);
        assert_eq!(r.native_ready_observed, 0);
        assert_eq!(r.native_terminal_missing, 1);
        assert!(!r.trace_complete);
        assert!(!r.authorities.physical_closure_observed);
    }
    #[test]
    fn original_execution_and_whole_clock_survive_output_errors_and_cleanup_partition() {
        let now = Instant::now();
        let started = now.checked_sub(Duration::from_millis(10)).unwrap();
        let c = Clock {
            started,
            execution: started + Duration::from_millis(1),
            whole: now + Duration::from_secs(60),
            output: 1024,
        };
        assert!(matches!(
            c.check(&AtomicBool::new(false)),
            Err(RoleError::Deadline)
        ));
        let r = receipt(c);
        assert!(r.execution_deadline_exceeded);
        assert!(!r.deadline_exceeded);
        assert!(r.elapsed_ms >= 10);
        let expired = Clock {
            whole: started + Duration::from_millis(2),
            ..c
        };
        let e = bare_error(
            NativeReplayPrimary::Admission(fault("output_reserve", "controlled invalid output")),
            expired,
            &AtomicBool::new(false),
        );
        assert!(
            e.original_whole_deadline_retained
                && e.deadline_exceeded
                && e.execution_deadline_exceeded
        );
        assert_eq!(e.output_limit, 1024);
        assert!(e.elapsed_ms >= 10);
        assert_eq!(e.primary.stage(), "output_reserve");
    }
    #[test]
    fn known_input_failure_retains_original_serialization_deadline() {
        let now = Instant::now();
        let started = now.checked_sub(Duration::from_millis(10)).unwrap();
        let mut source =
            ReplayInputError::new("output_bound", "controlled original invalid output");
        source.whole_wall_ms = Some(1);
        source.original_whole_deadline_retained = true;
        source.deadline_exceeded = true;
        source.execution_deadline_exceeded = true;
        source.output_limit = 8192;
        let e = input_error(source, started, &AtomicBool::new(false));
        assert_eq!(
            e.serialization_deadline,
            started.checked_add(Duration::from_millis(1))
        );
        let refused = e.serialized().unwrap_err();
        assert_eq!(refused.stage, "error_serialization_clock");
        let NativeReplayPrimary::Input(original) = e.primary.as_ref() else {
            panic!("controlled input cause lost")
        };
        assert_eq!(original.stage, "output_bound");
        assert_eq!(original.whole_wall_ms, Some(1));
        assert!(e.original_whole_deadline_retained && e.deadline_exceeded);
        let unknown = input_error(
            ReplayInputError::new("input_json", "controlled unknown-clock malformed input"),
            started,
            &AtomicBool::new(false),
        );
        assert!(unknown.serialization_deadline.is_none());
        assert!(!unknown.original_whole_deadline_retained);
        // Keep the Result's error payload small without changing the transparent
        // JSON wire or cloning the owned cause/receipt into a summary error.
        assert!(std::mem::size_of::<NativeReplayError>() < 128);
    }
    #[test]
    fn scoped_initial_input_failures_preserve_expected_declaration_without_loading() {
        let audit = receipt(clock(8192)).input_admission;
        let expected = ReplayExpectedPins {
            registration_artifact: audit.registration_artifact,
            prepared_action_artifact: audit.prepared_action_artifact,
            semantic_receipt_artifact: audit.semantic_receipt_artifact,
            parent: audit.parent,
            binding: audit.binding,
            registered_artifacts: audit.registered_artifacts,
            legacy_cpu_profile_sha256: audit.legacy_cpu_profile_sha256,
            provider_factory_id: audit.provider_factory_id,
            semantic_binary_sha256: "a".repeat(64),
        };
        let started = Instant::now();
        let scope = SemanticReceiptProducerScope::LinuxLoadedExecutableInode;
        // Exercise the actual scoped input checker and production error adapter
        // before any RuntimeLibraryPin, asset loader, CPU engine or model exists.
        let source =
            replay_inputs::check_replay_inputs_with_semantic_scope(b"{", &expected, scope, started)
                .unwrap_err();
        let error = input_error(source, started, &AtomicBool::new(false));
        assert_eq!(error.expected_semantic_receipt_producer_scope, Some(scope));
        assert!(error.receipt.is_none());
        assert!(!error.original_whole_deadline_retained);
        assert!(error.serialization_deadline.is_none());
        let NativeReplayPrimary::Input(original) = error.primary.as_ref() else {
            panic!("original scoped input cause lost")
        };
        assert_eq!(original.stage, "input_json");
        assert_eq!(
            original.expected_semantic_receipt_producer_scope,
            Some(scope)
        );
        let wire = serde_json::to_value(&error).unwrap();
        assert_eq!(
            wire["expected_semantic_receipt_producer_scope"],
            scope.as_str()
        );
        assert_eq!(
            wire["primary"]["input"]["expected_semantic_receipt_producer_scope"],
            scope.as_str()
        );
        let legacy_source =
            replay_inputs::check_replay_inputs(b"{", &expected, started).unwrap_err();
        let legacy = input_error(legacy_source, started, &AtomicBool::new(false));
        assert!(legacy.expected_semantic_receipt_producer_scope.is_none());
        assert!(
            !serde_json::to_string(&legacy)
                .unwrap()
                .contains("expected_semantic_receipt_producer_scope")
        );
        let wire = serde_json::to_value(&legacy).unwrap();
        assert!(
            !wire
                .as_object()
                .unwrap()
                .contains_key("expected_semantic_receipt_producer_scope")
        );
        assert!(
            !wire["primary"]["input"]
                .as_object()
                .unwrap()
                .contains_key("expected_semantic_receipt_producer_scope")
        );
    }
    #[test]
    fn scoped_retained_output_failure_keeps_expectation_and_false_authorities() {
        let c = clock(512);
        let scope = SemanticReceiptProducerScope::CurrentExePathHash;
        let mut observed = receipt(c);
        observed
            .input_admission
            .expected_semantic_receipt_producer_scope = Some(scope);
        observed.trace_jsonl = "{\"controlled_retained_scope_row\":true}\n".into();
        observed.trace_rows = 1;
        let error = with_receipt(
            NativeReplayPrimary::Output(fault("serialization", "controlled finite output failure")),
            None,
            None,
            Some(fault("serialization", "controlled finite output failure")),
            observed,
            c,
            &AtomicBool::new(false),
        );
        assert!(error.serialized().is_err());
        assert_eq!(error.expected_semantic_receipt_producer_scope, Some(scope));
        let retained = error.receipt.as_ref().unwrap();
        assert_eq!(
            retained
                .input_admission
                .expected_semantic_receipt_producer_scope,
            Some(scope)
        );
        assert_eq!(
            retained.input_admission.binary_pin_scope,
            replay_inputs::BINARY_DECLARATION_SCOPE
        );
        assert_eq!(
            retained.binary_pin_scope,
            replay_inputs::BINARY_DECLARATION_SCOPE
        );
        assert_eq!(retained.trace_rows, 1);
        assert!(
            retained
                .trace_jsonl
                .contains("controlled_retained_scope_row")
        );
        assert_eq!(retained.authorities, ReplayAuthorities::default());
        assert_eq!(
            retained.input_admission.authorities,
            ReplayAuthorities::default()
        );
        assert!(!retained.model_returned && !retained.owner_created && !retained.replay_attempted);
        let wire = serde_json::to_value(&error).unwrap();
        assert_eq!(
            wire["expected_semantic_receipt_producer_scope"],
            scope.as_str()
        );
        assert_eq!(
            wire["receipt"]["input_admission"]["expected_semantic_receipt_producer_scope"],
            scope.as_str()
        );
        let legacy = bare_error(
            NativeReplayPrimary::Admission(fault(
                "source_registration",
                "controlled legacy failure",
            )),
            c,
            &AtomicBool::new(false),
        );
        assert!(
            !serde_json::to_value(&legacy)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("expected_semantic_receipt_producer_scope")
        );
    }
    #[test]
    fn output_failure_keeps_nonzero_work_rows_unknowns_and_negative_authorities() {
        let c = clock(512);
        let mut r = receipt(c);
        r.cpu_tasks_requested = Some(1);
        r.cpu_tasks_accounted = Some(1);
        r.cpu_reports_returned = Some(0);
        r.cpu_nodes_lower_bound = Some(19);
        r.cpu_work_observation_incomplete = Some(true);
        r.trace_jsonl = "{\"controlled_retained_row\":true}\n".into();
        r.trace_rows = 1;
        let mut stage = CpuStageObservation::reserved().unwrap();
        stage.phase = "baseline";
        stage.cpu_execution = 7;
        stage.attempt_present = true;
        stage.attempt_nodes = Some(19);
        stage.attempt_qnodes = None;
        assert!(stage.raw_stage.capture(&"controlled failed stage"));
        r.stages.push(stage);
        let e = with_receipt(
            NativeReplayPrimary::Run(ReplayError::InvalidPlan("controlled original run cause")),
            None,
            Some(fault("observer", "controlled secondary")),
            Some(fault("serialization", "controlled secondary")),
            r,
            c,
            &AtomicBool::new(false),
        );
        assert!(e.serialized().is_err());
        let retained = e.receipt.as_ref().unwrap();
        assert_eq!(retained.cpu_nodes_lower_bound, Some(19));
        assert_eq!(retained.stages[0].attempt_qnodes, None);
        assert_eq!(retained.trace_rows, 1);
        assert!(retained.trace_jsonl.contains("controlled_retained_row"));
        assert!(!retained.replay_returned_ok);
        assert!(retained.native_cleanup_missing);
        let flags = serde_json::to_value(&retained.authorities).unwrap();
        assert!(
            flags
                .as_object()
                .unwrap()
                .values()
                .all(|v| v == &serde_json::Value::Bool(false))
        );
        assert_eq!(e.primary.stage(), "replay_run");
        assert!(e.observer_error.is_some() && e.output_error.is_some());
        let retained = *e.receipt.unwrap();
        let expired = Clock {
            whole: c.started,
            ..c
        };
        let late = with_receipt(
            NativeReplayPrimary::Run(ReplayError::InvalidPlan(
                "controlled same original run cause",
            )),
            None,
            Some(fault("observer", "controlled original secondary")),
            None,
            retained,
            expired,
            &AtomicBool::new(false),
        );
        assert_eq!(late.serialization_deadline, Some(c.started));
        assert_eq!(
            late.serialized().unwrap_err().stage,
            "error_serialization_clock"
        );
        let still_owned = late.receipt.as_ref().unwrap();
        assert_eq!(still_owned.cpu_nodes_lower_bound, Some(19));
        assert_eq!(still_owned.stages[0].attempt_nodes, Some(19));
        assert_eq!(still_owned.stages[0].attempt_qnodes, None);
        assert!(still_owned.trace_jsonl.contains("controlled_retained_row"));
        assert_eq!(late.primary.stage(), "replay_run");
        assert!(late.observer_error.is_some());
    }
    #[test]
    fn original_backend_box_and_bounded_diagnostic_projection_are_distinct() {
        let mut backend = rz_eval::error::BackendError::new(
            rz_eval::error::FailureKind::BackendFailure,
            rz_eval::error::FailureStage::Backend,
            "controlled backend original",
        );
        backend.native = Some(Box::new(rz_eval::error::NativeDiagnostic {
            code: "controlled_code".into(),
            message: "controlled_raw_message_not_archived".into(),
            truncated: false,
        }));
        let failure = NativeLoadFailure {
            stage: NativeLoadStage::BackendLoad,
            cause: RoleError::Backend("controlled role projection".into()),
            elapsed: Duration::from_millis(7),
            execution_expired: false,
            whole_expired: false,
            canceled: false,
            finish: None,
            backend_error: Some(Box::new(backend)),
        };
        let e = bare_error(
            NativeReplayPrimary::Load(Box::new(failure)),
            clock(8192),
            &AtomicBool::new(false),
        );
        let NativeReplayPrimary::Load(load) = e.primary.as_ref() else {
            panic!("controlled load primary lost")
        };
        assert_eq!(
            load.backend_error
                .as_ref()
                .unwrap()
                .native
                .as_ref()
                .unwrap()
                .message,
            "controlled_raw_message_not_archived"
        );
        let raw = e.serialized().unwrap();
        let text = std::str::from_utf8(&raw).unwrap();
        assert!(!text.contains("controlled_raw_message_not_archived"));
        let value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(
            value["primary"]["load"]["backend_error"]["native_diagnostic_present"],
            true
        );
        assert_eq!(
            value["primary"]["load"]["backend_error"]["raw_native_diagnostic_archived_in_this_json"],
            false
        );
        assert!(e.receipt.is_none());
    }
    #[test]
    fn bytes_prepared_tick_is_single_last_field_and_retains_whole_guard() {
        let c = clock(64);
        let mut out = reserved_bytes(c.output).unwrap();
        out.extend_from_slice(b"{\"observed\":true}");
        append_prepared_tick(&mut out, c, 9).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["bytes_prepared_elapsed_ms"], 9);
        assert_eq!(
            std::str::from_utf8(&out)
                .unwrap()
                .matches("bytes_prepared_elapsed_ms")
                .count(),
            1
        );
        let expired = Clock {
            whole: c.started,
            ..c
        };
        let mut refused = reserved_bytes(64).unwrap();
        refused.extend_from_slice(b"{}");
        assert!(append_prepared_tick(&mut refused, expired, 9).is_err());
    }

    #[test]
    fn public_output_exact_cap_and_shortage_match_pre_load_reservation() {
        let requirement = replay_output_requirements(3, 2).unwrap();
        assert_eq!(requirement.trace_bytes(), 432 * 1024);
        assert_eq!(requirement.text_bytes(), 112 * 1024);
        assert_eq!(requirement.required_output_bytes(), 2 * 1024 * 1024);
        let copied = requirement;
        let cap = requirement.required_output_bytes();
        assert!(requirement.check_output_limit(cap).is_ok());
        assert!(Reserved::new(3, 2, clock(cap)).is_ok());
        let refused = copied.check_output_limit(cap - 1).unwrap_err();
        assert_eq!(refused.stage, "output_reserve");
        assert_eq!(
            refused.reason,
            "declared output cannot retain worst-case observations"
        );
        assert!(matches!(
            Reserved::new(3, 2, clock(cap - 1)),
            Err(AdmissionFault {
                stage: "output_reserve",
                reason: "declared output cannot retain worst-case observations"
            })
        ));
        assert_eq!(copied, requirement);
    }

    #[test]
    fn public_output_overflow_faults_preserve_each_arithmetic_stage() {
        let cases = [
            (usize::MAX, 2, "trace overflow"),
            (0, usize::MAX, "stage snapshot overflow"),
            (
                0,
                usize::MAX / (STAGE_TEXT_BYTES + 2 * SNAPSHOT_BYTES),
                "retained snapshot overflow",
            ),
            (
                usize::MAX / ROLE_TRACE_BYTES,
                0,
                "serialized upper bound overflow",
            ),
        ];
        for (roles, checks, reason) in cases {
            let refused = replay_output_requirements(roles, checks).unwrap_err();
            assert_eq!(refused.stage, "output_reserve");
            assert_eq!(refused.reason, reason);
            assert!(
                matches!(Reserved::new(roles, checks, clock(usize::MAX)), Err(error) if error.stage == refused.stage && error.reason == refused.reason)
            );
        }
        // Zero counts retain the old fixed header/state bound. They do not
        // acquire a supported replay mode or an execution capability.
        let empty = replay_output_requirements(0, 0).unwrap();
        assert_eq!(empty.trace_bytes(), 0);
        assert_eq!(empty.text_bytes(), 48 * 1024);
        assert_eq!(empty.required_output_bytes(), 800 * 1024);
    }

    #[test]
    fn public_profile_checker_keeps_originals_and_expected_declaration_immutable() {
        let expected = profile(false);
        let original = serde_json::to_vec(&expected).unwrap();
        let expected_pin = pin(&original);
        let original_copy = original.clone();
        let expected_copy = serde_json::to_vec(&expected).unwrap();
        let pin_copy = expected_pin.clone();
        let mut checked =
            check_cpu_fresh_asset_profile(&original, &expected_pin, &expected).unwrap();
        assert!(same_profile(&checked, &expected));
        checked.cache_public_memory = true;
        checked.context_sha256 = asset_profile_context(&checked).unwrap();
        let changed = serde_json::to_vec(&checked).unwrap();
        let refused =
            check_cpu_fresh_asset_profile(&changed, &pin(&changed), &expected).unwrap_err();
        assert_eq!(refused.fault.stage, "asset_profile");
        assert_eq!(original, original_copy);
        assert_eq!(serde_json::to_vec(&expected).unwrap(), expected_copy);
        assert!(same_pin(&expected_pin, &pin_copy));
        assert!(!expected.cache_public_memory);
    }

    #[test]
    fn public_profile_checker_rejects_pin_extent_scope_and_context_substitution() {
        let expected = profile(false);
        let original = serde_json::to_vec(&expected).unwrap();
        let expected_pin = pin(&original);
        let mut changed_raw = original.clone();
        changed_raw.push(b' ');
        assert_eq!(
            check_cpu_fresh_asset_profile(&changed_raw, &expected_pin, &expected)
                .unwrap_err()
                .fault
                .stage,
            "asset_profile_pin"
        );
        let oversized = vec![b' '; MAX_PROFILE_BYTES + 1];
        assert_eq!(
            check_cpu_fresh_asset_profile(&oversized, &pin(&oversized), &expected)
                .unwrap_err()
                .fault
                .stage,
            "asset_profile_pin"
        );
        for field in ["schema", "domain", "provider", "encoding"] {
            let mut changed = expected.clone();
            match field {
                "schema" => changed.schema = "rz-pals-cpu-fresh-replay-assets/2".into(),
                "domain" => changed.domain = "private_warm".into(),
                "provider" => changed.provider = "cuda".into(),
                _ => {
                    changed.encoding_semantic_sha256 =
                        lower_hex(pals_native::pals_rules_encoding_semantic_digest())
                }
            }
            changed.context_sha256 = asset_profile_context(&changed).unwrap();
            let raw = serde_json::to_vec(&changed).unwrap();
            let refused = check_cpu_fresh_asset_profile(&raw, &pin(&raw), &changed).unwrap_err();
            assert_eq!(refused.fault.stage, "asset_profile");
        }
        let mut changed = expected.clone();
        changed.context_sha256 = lower_hex([9; 32]);
        let raw = serde_json::to_vec(&changed).unwrap();
        assert_eq!(
            check_cpu_fresh_asset_profile(&raw, &pin(&raw), &changed)
                .unwrap_err()
                .fault
                .stage,
            "asset_profile_context"
        );
    }

    #[test]
    fn public_profile_checker_rejects_key_alias_and_duplicate_with_original_json_cause() {
        let expected = profile(false);
        let original = serde_json::to_string(&expected).unwrap();
        let invalid = [
            original.replace("\"domain\":", "\"Domain\":"),
            original.replace("\"context_sha256\":", "\"contextSha256\":"),
            original.replacen('{', "{\"schema\":\"rz-pals-cpu-fresh-replay-assets/1\",", 1),
        ];
        for raw in invalid {
            let refused =
                check_cpu_fresh_asset_profile(raw.as_bytes(), &pin(raw.as_bytes()), &expected)
                    .unwrap_err();
            assert_eq!(refused.fault.stage, "asset_profile_json");
            assert_eq!(refused.fault.reason, "closed profile JSON refused");
            assert!(matches!(
                refused.source.as_deref(),
                Some(CpuFreshAssetSourceError::Json(_))
            ));
        }
    }
}
