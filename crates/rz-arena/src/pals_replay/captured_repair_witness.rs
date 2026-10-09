//! Source-owned exact recorded tensors plus separate actual native/CPU work.
//! A captured deadline feature is historical input data, not a fresh live clock.
//! Child work remains reported; these capabilities attest the new executions.

use super::episode::{digest, normalized_trace};
use super::{ReportedRepairPriorMaterial, control, invalid};
use crate::ArenaError;
use rz_eval::pals_model::{PalsModelConfig, PalsModelInput, PalsRole};
use rz_search::pals::engine::RoleError;
use rz_uci::pals_cpu_task::strategic_action::native_replay::query_prior::PriorReadiness;
use rz_uci::pals_cpu_task::strategic_action::native_replay::query_prior::captured_cpu_witness::CheckedIndependentRepairCpuWitness;
use rz_uci::pals_native::{
    NativeInvocationBudget, NativeRecordedCpuPristineAck, NativeRecordedInputWitness,
    NativeRoleModel, NativeRoleReceipt, NativeRoleSourceIdentity,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

pub const CAPTURED_REPAIR_WITNESS_SCOPE: &str =
    "captured_nn_input_reinference_and_independent_cpu_condition_reexecution";
const MAX_INPUTS: usize = 256;
/// 256 bounded native inputs/raw projections (roughly 52 KiB per captured row)
/// plus the separately bounded 32 KiB CPU audit and final receipt. This is an
/// output policy ceiling, not a measured allocator/RSS peak or an automatic
/// larger replay stdout declaration. Overflow refuses the audit, retaining caps.
pub const MAX_CAPTURED_REPAIR_AUDIT_BYTES: usize = 16 * 1024 * 1024;

/// Keeps the actual error object. A formatted audit is only a bounded diagnostic
/// projection and cannot mint a successful replay, Query or utility capability.
#[derive(Debug)]
pub enum CapturedRepairWitnessCause {
    Refusal(ArenaError),
    Native {
        stage: &'static str,
        error: RoleError,
    },
}
impl From<ArenaError> for CapturedRepairWitnessCause {
    fn from(error: ArenaError) -> Self {
        Self::Refusal(error)
    }
}
#[derive(Default)]
struct WitnessProgress {
    native: Vec<NativeRecordedInputWitness>,
    pristine: Option<NativeRecordedCpuPristineAck>,
    final_receipt: Option<NativeRoleReceipt>,
}
/// Partial actual work remains owned here on refusal, including a returned raw
/// cap that did not match the child. No constructor, Clone or deserializer.
pub struct CapturedRepairWitnessFailure {
    retained: Box<CapturedRepairFailureData>,
}
// Keep the returned error small without dropping or cloning the actual partial
// work. The heap allocation belongs only to this failure and every audit borrows
// its original typed cause, capabilities, receipts and absolute clocks.
struct CapturedRepairFailureData {
    cause: CapturedRepairWitnessCause,
    source_input: rz_uci::pals_cpu_task::strategic_action::ArtifactPin,
    source_output: rz_uci::pals_cpu_task::strategic_action::ArtifactPin,
    budget: NativeInvocationBudget,
    source_started: Instant,
    source_whole: Instant,
    progress: WitnessProgress,
    native_snapshot: NativeRoleReceipt,
}
impl std::fmt::Debug for CapturedRepairWitnessFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturedRepairWitnessFailure")
            .field("cause", &self.retained.cause)
            .field("source_input", &self.retained.source_input)
            .field(
                "completed_native_inputs",
                &self.retained.progress.native.len(),
            )
            .finish()
    }
}
impl std::fmt::Display for CapturedRepairWitnessFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.retained.cause {
            CapturedRepairWitnessCause::Refusal(_) => {
                write!(f, "captured Repair refused; owned diagnostic available")
            }
            CapturedRepairWitnessCause::Native { stage, .. } => write!(
                f,
                "captured Repair native failure at {stage}; owned diagnostic available"
            ),
        }
    }
}
impl std::error::Error for CapturedRepairWitnessFailure {}
impl CapturedRepairWitnessFailure {
    pub fn primary_cause(&self) -> &CapturedRepairWitnessCause {
        &self.retained.cause
    }
    pub fn completed_native_inputs(&self) -> &[NativeRecordedInputWitness] {
        &self.retained.progress.native
    }
    pub fn pristine_ack(&self) -> Option<&NativeRecordedCpuPristineAck> {
        self.retained.progress.pristine.as_ref()
    }
    pub fn final_native_receipt(&self) -> Option<&NativeRoleReceipt> {
        self.retained.progress.final_receipt.as_ref()
    }
    /// Read-only snapshot on failure; physical closure is never implied by it.
    pub fn native_failure_snapshot(&self) -> &NativeRoleReceipt {
        &self.retained.native_snapshot
    }
    pub fn invocation_budget(&self) -> NativeInvocationBudget {
        self.retained.budget
    }
    /// Caller output credit is independently enforced by W. The deadline is for
    /// failure preservation only and may not exceed original W + two seconds.
    pub fn write_audit<W: Write>(
        &self,
        writer: &mut W,
        deadline: Instant,
    ) -> Result<(), ArenaError> {
        let retained = &self.retained;
        let maximum_deadline = retained
            .source_whole
            .checked_add(std::time::Duration::from_secs(2))
            .ok_or_else(|| invalid("captured failure postmortem extent"))?;
        if deadline > maximum_deadline || Instant::now() >= deadline {
            return Err(invalid("captured failure fixed diagnostic window"));
        }
        let audit = CapturedFailureAudit {
            schema: "rz-pals-captured-repair-witness-failure/1",
            cause: DebugDiagnostic(&retained.cause),
            source_input: &retained.source_input,
            source_output: &retained.source_output,
            completed_native_inputs: &retained.progress.native,
            pristine_ack: retained.progress.pristine.as_ref(),
            final_native_receipt: retained.progress.final_receipt.as_ref(),
            native_failure_snapshot: &retained.native_snapshot,
            original_execution_ns: retained
                .budget
                .execution_until()
                .saturating_duration_since(retained.budget.started())
                .as_nanos(),
            original_whole_ns: retained
                .budget
                .whole_until()
                .saturating_duration_since(retained.budget.started())
                .as_nanos(),
            diagnostic_deadline_ns: deadline
                .saturating_duration_since(retained.budget.started())
                .as_nanos(),
            source_original_whole_ns: retained
                .source_whole
                .saturating_duration_since(retained.source_started)
                .as_nanos(),
            postmortem_only: true,
            query_admitted: false,
            utility_authority: false,
            training_authority: false,
            native_cleanup_on_serialization: false,
            maximum_audit_bytes: MAX_CAPTURED_REPAIR_AUDIT_BYTES,
        };
        let mut bounded = ExternalBoundedAudit {
            writer,
            attempted: 0,
            deadline,
        };
        serde_json::to_writer(&mut bounded, &audit)
            .map_err(|_| invalid("captured failure bounded diagnostic serialization"))?;
        bounded
            .flush()
            .map_err(|_| invalid("captured failure diagnostic final clock"))
    }
    pub fn audit_json(&self, deadline: Instant) -> Result<Vec<u8>, ArenaError> {
        let mut writer = BoundedAuditWriter::new(MAX_CAPTURED_REPAIR_AUDIT_BYTES, deadline);
        self.write_audit(&mut writer, deadline)?;
        Ok(writer.bytes)
    }
}
struct DebugDiagnostic<'a>(&'a dyn std::fmt::Debug);
impl std::fmt::Display for DebugDiagnostic<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl Serialize for DebugDiagnostic<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
#[derive(Serialize)]
struct CapturedFailureAudit<'a> {
    schema: &'static str,
    cause: DebugDiagnostic<'a>,
    source_input: &'a rz_uci::pals_cpu_task::strategic_action::ArtifactPin,
    source_output: &'a rz_uci::pals_cpu_task::strategic_action::ArtifactPin,
    completed_native_inputs: &'a [NativeRecordedInputWitness],
    pristine_ack: Option<&'a NativeRecordedCpuPristineAck>,
    final_native_receipt: Option<&'a NativeRoleReceipt>,
    native_failure_snapshot: &'a NativeRoleReceipt,
    original_execution_ns: u128,
    original_whole_ns: u128,
    diagnostic_deadline_ns: u128,
    source_original_whole_ns: u128,
    postmortem_only: bool,
    query_admitted: bool,
    utility_authority: bool,
    training_authority: bool,
    native_cleanup_on_serialization: bool,
    maximum_audit_bytes: usize,
}
struct ExternalBoundedAudit<'a, W: Write> {
    writer: &'a mut W,
    attempted: usize,
    deadline: Instant,
}
impl<W: Write> Write for ExternalBoundedAudit<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if Instant::now() >= self.deadline {
            return Err(io::Error::other(
                "captured failure fixed postmortem expired",
            ));
        }
        let previous = self.attempted;
        self.attempted = previous
            .checked_add(bytes.len())
            .filter(|&n| n <= MAX_CAPTURED_REPAIR_AUDIT_BYTES)
            .ok_or_else(|| io::Error::other("captured failure audit credit exceeded"))?;
        // Failed I/O keeps the entire attempt spent because its partial output
        // is unknown. Successful short writes report an exact n and settle only
        // those bytes before write_all retries the remaining suffix.
        match self.writer.write(bytes) {
            Ok(n) if n <= bytes.len() => {
                self.attempted = previous + n;
                if Instant::now() >= self.deadline {
                    return Err(io::Error::other(
                        "captured failure write returned after fixed postmortem",
                    ));
                }
                Ok(n)
            }
            Ok(_) => Err(io::Error::other(
                "captured failure writer reported an invalid byte count",
            )),
            Err(error) => Err(error),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        if Instant::now() >= self.deadline {
            return Err(io::Error::other(
                "captured failure fixed postmortem expired",
            ));
        }
        self.writer.flush()?;
        if Instant::now() >= self.deadline {
            return Err(io::Error::other(
                "captured failure flush returned after fixed postmortem",
            ));
        }
        Ok(())
    }
}

/// Only the source capture's material owner can issue this wrapper. The native
/// capabilities own immutable actual job inputs; CPU proof comes from a fresh
/// independent engine. No Clone, Deserialize or report-to-capability constructor.
pub struct CheckedCapturedRepairWitness<'a> {
    material: &'a ReportedRepairPriorMaterial<'a>,
    cpu: &'a CheckedIndependentRepairCpuWitness,
    native: Vec<NativeRecordedInputWitness>,
    pristine: NativeRecordedCpuPristineAck,
    final_receipt: NativeRoleReceipt,
    started: Instant,
    finished: Instant,
    fact: String,
    nn_inputs: u64,
    reported_work: Option<(u64, u64)>,
}

impl CheckedCapturedRepairWitness<'_> {
    pub fn material(&self) -> &ReportedRepairPriorMaterial<'_> {
        self.material
    }
    pub fn cpu(&self) -> &CheckedIndependentRepairCpuWitness {
        self.cpu
    }
    pub fn nn_inputs(&self) -> &[NativeRecordedInputWitness] {
        &self.native
    }
    pub fn pristine_ack(&self) -> &NativeRecordedCpuPristineAck {
        &self.pristine
    }
    pub fn final_native_receipt(&self) -> &NativeRoleReceipt {
        &self.final_receipt
    }
    pub fn work_started_at(&self) -> Instant {
        self.started
    }
    pub fn work_finished_at(&self) -> Instant {
        self.finished
    }
    pub fn fact_sha256(&self) -> &str {
        &self.fact
    }
    pub fn assurance_scope(&self) -> &'static str {
        CAPTURED_REPAIR_WITNESS_SCOPE
    }
    /// The new independent executions only. Source child numbers remain reports.
    pub fn actual_work_counts(&self) -> (u64, u64) {
        (self.cpu.actual_cpu_nodes(), self.nn_inputs)
    }
    pub fn reported_work_counts(&self) -> Option<(u64, u64)> {
        self.reported_work
    }
    pub fn audit_json(&self) -> Result<Vec<u8>, ArenaError> {
        let deadline = self.material.original_input().deadline();
        if Instant::now() >= deadline {
            return Err(invalid("captured witness audit original W expired"));
        }
        let cpu_raw = self
            .cpu
            .audit_json()
            .map_err(|_| invalid("captured CPU audit"))?;
        if cpu_raw.len() > 32 * 1024 {
            return Err(invalid("captured CPU audit byte bound"));
        }
        let cpu_audit: Value =
            serde_json::from_slice(&cpu_raw).map_err(|_| invalid("captured CPU audit JSON"))?;
        // Borrow native inputs/raw outputs directly. Building one large serde
        // Value first would duplicate every input and raw latent before applying
        // the output ceiling, defeating the bounded writer's memory policy.
        let audit = CapturedAudit {
            schema: "rz-pals-captured-repair-witness/1",
            assurance_scope: self.assurance_scope(),
            source_input: self.material.original_input().artifact(),
            source_output: self.material.checked_report().body_artifact(),
            parent: self.material.original_parent(),
            binding: self.material.previous_query_binding(),
            conditional_fact_sha256: &self.fact,
            native_inputs: &self.native,
            pristine_ack: &self.pristine,
            independent_cpu: &cpu_audit,
            actual_work_counts: self.actual_work_counts(),
            reported_child_work_counts: self.reported_work,
            source_cpu_scalar_score_equal: None,
            source_physical_execution_attested: false,
            source_context_debug_used_as_rules_proof: false,
            final_native_receipt: &self.final_receipt,
            maximum_audit_bytes: MAX_CAPTURED_REPAIR_AUDIT_BYTES,
            utility_authority: false,
            target_authority: false,
            training_authority: false,
            product_authority: false,
        };
        let mut writer = BoundedAuditWriter::new(MAX_CAPTURED_REPAIR_AUDIT_BYTES, deadline);
        serde_json::to_writer(&mut writer, &audit)
            .map_err(|_| invalid("captured witness audit bound/allocation/original W"))?;
        writer
            .flush()
            .map_err(|_| invalid("captured witness audit final original W"))?;
        Ok(writer.bytes)
    }
}

#[derive(Serialize)]
struct CapturedAudit<'a> {
    schema: &'static str,
    assurance_scope: &'static str,
    source_input: &'a rz_uci::pals_cpu_task::strategic_action::ArtifactPin,
    source_output: &'a rz_uci::pals_cpu_task::strategic_action::ArtifactPin,
    parent: &'a rz_uci::pals_cpu_task::strategic_action::replay_inputs::ReplayParentPins,
    binding: &'a rz_uci::pals_cpu_task::strategic_action::replay_inputs::ReplayBindingPins,
    conditional_fact_sha256: &'a str,
    native_inputs: &'a [NativeRecordedInputWitness],
    pristine_ack: &'a NativeRecordedCpuPristineAck,
    independent_cpu: &'a Value,
    actual_work_counts: (u64, u64),
    reported_child_work_counts: Option<(u64, u64)>,
    source_cpu_scalar_score_equal: Option<bool>,
    source_physical_execution_attested: bool,
    source_context_debug_used_as_rules_proof: bool,
    final_native_receipt: &'a NativeRoleReceipt,
    maximum_audit_bytes: usize,
    utility_authority: bool,
    target_authority: bool,
    training_authority: bool,
    product_authority: bool,
}
struct BoundedAuditWriter {
    bytes: Vec<u8>,
    maximum: usize,
    deadline: Instant,
}
impl BoundedAuditWriter {
    fn new(maximum: usize, deadline: Instant) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
            deadline,
        }
    }
    fn check(&self) -> io::Result<()> {
        if Instant::now() >= self.deadline {
            Err(io::Error::other("original audit W expired"))
        } else {
            Ok(())
        }
    }
}
impl Write for BoundedAuditWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.check()?;
        let required = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|&n| n <= self.maximum)
            .ok_or_else(|| io::Error::other("audit byte ceiling"))?;
        if required > self.bytes.capacity() {
            let target = self
                .bytes
                .capacity()
                .saturating_mul(2)
                .max(64 * 1024)
                .max(required)
                .min(self.maximum);
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|_| io::Error::other("audit allocation refused"))?;
        }
        self.bytes.extend_from_slice(bytes);
        self.check()?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.check()
    }
}

/// Execute the source owner's original P/C inputs unchanged, including query[7].
/// The independently loaded model is already tied to the same absolute S/E/W.
/// Completion/refusal never resets that clock. The caller retains the model's
/// finish handle on refusal; no successful wrapper is issued before final join.
pub fn witness_captured_repair_inputs<'a>(
    material: &'a ReportedRepairPriorMaterial<'a>,
    model: &mut NativeRoleModel,
    cpu: &'a CheckedIndependentRepairCpuWitness,
    budget: NativeInvocationBudget,
    cancel: &AtomicBool,
) -> Result<CheckedCapturedRepairWitness<'a>, CapturedRepairWitnessFailure> {
    let mut progress = WitnessProgress::default();
    let result = (|| -> Result<CheckedCapturedRepairWitness<'a>, CapturedRepairWitnessCause> {
        let invalid = |detail: &str| CapturedRepairWitnessCause::Refusal(super::invalid(detail));
        control(budget.execution_until(), cancel)?;
        let bundle = material.checked_report().caller_timing().capture().bundle();
        let reported = material.checked_report().reported_rules();
        let complete = reported.rules().reported().recomputed_reported_readiness();
        let source_projection = reported.rules().reported().reported_projection();
        let projection_raw = serde_json::to_vec(source_projection)
            .map_err(|_| invalid("captured Repair projection encoding"))?;
        let projection_pin = rz_uci::pals_cpu_task::strategic_action::ArtifactPin {
            bytes: projection_raw.len() as u64,
            sha256: format!("{:x}", Sha256::digest(&projection_raw)),
        };
        if budget.started() != bundle.original_started()
            || budget.whole_until() != bundle.deadline()
            || budget.execution_until() != material.original_input().execution_deadline()
            || !matches!(
                complete,
                PriorReadiness::CompleteCpuScopePendingCallerChronology
            )
            || cpu.original_input_artifact() != material.original_input().artifact()
            || cpu.parent() != material.original_parent()
            || cpu.binding() != material.previous_query_binding()
            || cpu.reported_projection_artifact() != &projection_pin
            || cpu.original_started() != budget.started()
            || cpu.original_execution() != budget.execution_until()
            || cpu.original_deadline() != budget.whole_until()
            || cpu.repaired_line() != source_projection.repaired_line()
            || cpu.opponent_counterline() != source_projection.opponent_counterline()
            || !cpu.final_cpu_completion_checked()
            || cpu.work_started_at() < material.before_next_query_at()
            || cpu.work_finished_at() < cpu.work_started_at()
            || cpu.work_finished_at() > Instant::now()
            || cpu.work_finished_at() >= budget.execution_until()
        {
            return Err(invalid("captured Repair CPU/source/original-clock binding"));
        }
        let reservation = material
            .captured_witness_issue
            .reserve(bundle.deadline(), cancel)?;
        let body: Value = serde_json::from_slice(material.raw_native_bytes())
            .map_err(|_| invalid("captured Repair body JSON"))?;
        let trace = body["trace_jsonl"]
            .as_str()
            .ok_or_else(|| invalid("captured Repair trace missing"))?;
        let normalized = normalized_trace(trace)?;
        let inputs = captured_inputs(&normalized)?;
        check_source_trace_counts(&body, trace, inputs.len())?;
        if inputs.len() as u64 > cpu.source_role_calls() {
            return Err(invalid(
                "captured Repair prepared input count exceeds typed original role allowance",
            ));
        }
        let identity = model.source_identity();
        check_model_binding(
            &body["native_receipt"],
            &identity,
            material.original_parent(),
        )?;
        let (load_started, load_finished) = model
            .invocation_load_interval()
            .ok_or_else(|| invalid("captured Repair actual model initialization unobserved"))?;
        if load_started < budget.started()
            || load_finished < load_started
            || load_finished > Instant::now()
            || load_finished >= budget.execution_until()
        {
            return Err(invalid(
                "captured Repair actual model initialization outside original episode",
            ));
        }
        // A CPU execution is a one-shot witness even across different capture owners
        // or episodes. Refusal after this live claim deliberately leaves it spent.
        cpu.reserve_captured_witness_scope()
            .map_err(|_| invalid("captured Repair CPU witness already consumed"))?;
        let started = cpu.work_started_at().min(load_started);
        let pristine = model
            .witness_recorded_cpu_pristine(budget, cancel)
            .map_err(|error| CapturedRepairWitnessCause::Native {
                stage: "pristine_same_worker_stats",
                error,
            })?;
        progress.pristine = Some(pristine);
        let pristine = progress
            .pristine
            .as_ref()
            .expect("just retained actual pristine ACK");
        if !pristine.physical_completion_confirmed()
            || pristine.invocation_budget() != budget
            || pristine.invocation_load_interval() != (load_started, load_finished)
            || pristine.work_started_at() < load_finished
            || pristine.work_finished_at() < pristine.work_started_at()
            || pristine.work_finished_at() >= budget.execution_until()
        {
            return Err(invalid(
                "captured Repair actual pristine ACK clock/budget differs",
            ));
        }
        progress
            .native
            .try_reserve_exact(inputs.len())
            .map_err(|_| invalid("captured Repair input allocation"))?;
        let mut nn_inputs = 0_u64;
        for (input, raw_bits) in inputs {
            control(budget.execution_until(), cancel)?;
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .map_err(|_| invalid("captured Repair canonical input"))?;
            if input.model_epoch != identity.checkpoint_sha256 {
                return Err(invalid("captured Repair recorded model differs"));
            }
            let cap = model
                .witness_recorded_cpu_input(input, key, budget, cancel)
                .map_err(|error| CapturedRepairWitnessCause::Native {
                    stage: "recorded_input",
                    error,
                })?;
            // A returned physical raw result is retained even when content differs.
            retain_and_check_completed(&mut progress.native, cap, |cap| {
                if cap.raw_output_bit_projection() != raw_bits
                    || !cap.physical_completion_confirmed()
                    || cap.work_started_at() < cpu.work_finished_at()
                    || cap.invocation_load_interval() != Some((load_started, load_finished))
                    || cap.work_started_at() < load_finished
                    || cap.process_epoch() != pristine.process_epoch()
                    || cap.work_started_at() < pristine.work_finished_at()
                    || cap.work_finished_at() < cap.work_started_at()
                    || cap.work_finished_at() >= budget.execution_until()
                {
                    return Err(invalid(
                        "captured Repair exact raw bits/completion chronology differs",
                    ));
                }
                Ok(())
            })?;
            let cap = progress
                .native
                .last()
                .expect("just retained actual completed input");
            let delta = cap
                .backend_stats_after()
                .completed_nn_inputs
                .checked_sub(cap.backend_stats_before().completed_nn_inputs)
                .ok_or_else(|| invalid("captured Repair native counter underflow"))?;
            nn_inputs = nn_inputs
                .checked_add(delta)
                .ok_or_else(|| invalid("captured Repair native counter extent"))?;
        }
        let final_receipt =
            model
                .finish_handle()
                .finish(budget.whole_until())
                .map_err(|error| CapturedRepairWitnessCause::Native {
                    stage: "final_native_shutdown",
                    error,
                })?;
        progress.final_receipt = Some(final_receipt);
        let final_receipt = progress
            .final_receipt
            .as_ref()
            .expect("just retained final native receipt");
        if final_receipt.quarantined
            || final_receipt.physical_runs_in_flight != 0
            || !final_receipt.physical_shutdown_confirmed
            || !final_receipt.native_buffers_released
        {
            return Err(invalid("captured Repair final physical closure"));
        }
        let all_native_inputs = final_receipt
            .backend_stats
            .as_ref()
            .ok_or_else(|| invalid("captured Repair final NN counters unobserved"))?
            .completed_nn_inputs;
        let maximum_native_inputs = cpu
            .source_role_calls()
            .checked_mul(2)
            .ok_or_else(|| invalid("captured Repair typed native input allowance overflow"))?;
        if all_native_inputs != nn_inputs || all_native_inputs > maximum_native_inputs {
            return Err(invalid(
                "captured Repair final pristine counters exceed exact bounded work",
            ));
        }
        // Charge any actual native initialization work as well as the recorded jobs.
        nn_inputs = all_native_inputs;
        control(budget.whole_until(), cancel)?;
        let finished = Instant::now();
        let cpu_json: Value = serde_json::from_slice(
            &cpu.audit_json()
                .map_err(|_| invalid("captured CPU fact encoding"))?,
        )
        .map_err(|_| invalid("captured CPU fact JSON"))?;
        let fact = digest(
            "rz-pals-captured-repair-conditional-fact/1",
            &json!({
                "parent":material.original_parent(),"binding":material.previous_query_binding(),
                "source_input":material.original_input().artifact(),"source_output":material.checked_report().body_artifact(),
                "input_raw_trace":normalized,"independent_cpu":cpu_json,
                "model":identity,"source_scalar_score_equality":null,"assurance_scope":CAPTURED_REPAIR_WITNESS_SCOPE
            }),
        )?;
        let reported_work = if body["cpu_work_observation_incomplete"] == false {
            body["cpu_nodes_lower_bound"]
                .as_u64()
                .zip(body["native_receipt"]["backend_stats"]["completed_nn_inputs"].as_u64())
        } else {
            None
        };
        reservation.issue(budget.whole_until(), cancel)?;
        control(budget.whole_until(), cancel)?;
        Ok(CheckedCapturedRepairWitness {
            material,
            cpu,
            native: std::mem::take(&mut progress.native),
            pristine: progress
                .pristine
                .take()
                .expect("successful actual pristine ACK"),
            final_receipt: progress
                .final_receipt
                .take()
                .expect("successful final native receipt"),
            started,
            finished,
            fact,
            nn_inputs,
            reported_work,
        })
    })();
    match result {
        Ok(witness) => Ok(witness),
        Err(cause) => Err(CapturedRepairWitnessFailure {
            retained: Box::new(CapturedRepairFailureData {
                cause,
                source_input: material.original_input().artifact().clone(),
                source_output: material.checked_report().body_artifact().clone(),
                budget,
                source_started: material
                    .checked_report()
                    .caller_timing()
                    .capture()
                    .bundle()
                    .original_started(),
                source_whole: material.original_input().deadline(),
                progress,
                native_snapshot: model.finish_handle().receipt(),
            }),
        }),
    }
}

fn check_source_trace_counts(body: &Value, trace: &str, inputs: usize) -> Result<(), ArenaError> {
    let count = inputs as u64;
    if body["trace_complete"] != true
        || body["trace_rows"].as_u64() != Some(trace.lines().count() as u64)
    {
        return Err(invalid(
            "captured Repair source trace row/completion report differs",
        ));
    }
    for field in ["native_bindings_observed", "native_ready_observed"] {
        if body[field].as_u64() != Some(count) {
            return Err(invalid(
                "captured Repair source binding/completion count differs from trace",
            ));
        }
    }
    for field in [
        "native_completion_unknown_observed",
        "native_terminal_missing",
    ] {
        if body[field].as_u64() != Some(0) {
            return Err(invalid(
                "captured Repair source trace has unknown/missing physical completion",
            ));
        }
    }
    let receipt = &body["native_receipt"];
    // This source is the registered CPU Fresh /4 invocation. A startup role
    // probe is not folded into the captured search calls. Reset and Stats control
    // commands use their own counters and are not role-input completion groups.
    if !receipt["startup_probe"].is_null() {
        return Err(invalid(
            "captured Repair CPU Fresh source unexpectedly has startup role work",
        ));
    }
    for field in [
        "physically_completed_role_calls",
        "completed_role_inputs",
        "delivered_role_inputs",
        "search_consumed_role_inputs",
    ] {
        if receipt[field].as_u64() != Some(count) {
            return Err(invalid(
                "captured Repair source completed/delivered/consumed role count differs from trace",
            ));
        }
    }
    for field in [
        "failed_physical_role_calls",
        "invalid_role_outputs",
        "canceled_requests",
        "expired_requests",
        "observer_failures",
    ] {
        if receipt[field].as_u64() != Some(0) {
            return Err(invalid(
                "captured Repair source reports failed/canceled/invalid role work",
            ));
        }
    }
    Ok(())
}

fn retain_and_check_completed<T>(
    completed: &mut Vec<T>,
    value: T,
    check: impl FnOnce(&T) -> Result<(), CapturedRepairWitnessCause>,
) -> Result<(), CapturedRepairWitnessCause> {
    completed.push(value);
    check(completed.last().expect("newly retained actual result"))
}

fn check_model_binding(
    reported: &Value,
    actual: &NativeRoleSourceIdentity,
    parent: &rz_uci::pals_cpu_task::strategic_action::replay_inputs::ReplayParentPins,
) -> Result<(), ArenaError> {
    let same = |key: &str, v: Value| reported.get(key) == Some(&v);
    let hex = |v: [u8; 32]| v.iter().map(|b| format!("{b:02x}")).collect::<String>();
    if !same("model_epoch", json!(actual.checkpoint_sha256))
        || !same(
            "export_manifest_sha256",
            json!(actual.export_manifest_sha256),
        )
        || !same(
            "encoding_semantic_sha256",
            json!(actual.encoding_semantic_sha256),
        )
        || !same("frozen_epoch", json!(actual.frozen_epoch))
        || !same("trained", json!(actual.trained))
        || reported["execution"]["runtime_sha256"] != json!(actual.execution.runtime_sha256)
        || reported["execution"]["provider"] != "cpu"
        || actual.execution.provider != "cpu"
        || parent.encoding_sha256 != hex(actual.encoding_semantic_sha256)
        || actual.execution.runtime_bundle_sha256.is_some()
        || actual.execution.device_public_memory
        || actual.execution.host_record_pages.is_some()
        || actual.execution.private_warm.is_some()
        || actual.execution.cuda_record_pages.is_some()
    {
        return Err(invalid(
            "captured Repair checkpoint/export/encoding/frozen/backend binding",
        ));
    }
    // Source adapter implementation is separately pinned by actual material's
    // loaded binary/registered source. A new witness implementation has its own
    // digest and is never presented as the old physical adapter execution.
    Ok(())
}

fn captured_inputs(trace: &Value) -> Result<Vec<(PalsModelInput, String)>, ArenaError> {
    let rows = trace
        .as_array()
        .ok_or_else(|| invalid("captured Repair normalized trace"))?;
    let mut result = Vec::new();
    for row in rows.iter().filter(|r| r["event"] == "prepared") {
        if result.len() >= MAX_INPUTS {
            return Err(invalid("captured Repair maximum native inputs"));
        }
        let ordinal = row["ordinal"]
            .as_u64()
            .ok_or_else(|| invalid("captured Repair request ordinal"))?;
        let input: PalsModelInput = serde_json::from_value(row["input"].clone())
            .map_err(|_| invalid("captured Repair closed typed input"))?;
        if !matches!(
            (row["kind"].as_str(), input.role),
            (Some("propose" | "repair"), PalsRole::Proposer) | (Some("reply"), PalsRole::Critic)
        ) {
            return Err(invalid("captured Repair unsupported role/input kind"));
        }
        input
            .validate(&PalsModelConfig::baseline())
            .map_err(|_| invalid("captured Repair input shape/finite"))?;
        let mut raw = rows
            .iter()
            .filter(|r| r["event"] == "physical_ready" && r["ordinal"].as_u64() == Some(ordinal));
        let value = raw
            .next()
            .ok_or_else(|| invalid("captured Repair physical raw missing"))?;
        if raw.next().is_some() {
            return Err(invalid("captured Repair duplicate physical raw"));
        }
        let bits = value["fact"]["raw_projection"]["text"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("captured Repair raw bit projection missing"))?;
        result.push((input, bits.to_owned()));
    }
    if result.is_empty() {
        return Err(invalid("captured Repair native inputs absent"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> PalsModelInput {
        let mut query = [0.0; 16];
        query[7] = 7.8125;
        PalsModelInput {
            role: PalsRole::Proposer,
            board: vec![0; 64],
            metadata: [0.0; 16],
            records: vec![],
            required_critical_records: vec![],
            candidates: vec![],
            divergence_features: vec![],
            query,
            situation_revision: 9,
            history_digest: [3; 32],
            model_epoch: [4; 32],
        }
    }
    fn trace() -> Value {
        json!([
            {"event":"prepared","ordinal":0,"kind":"repair","input":input()},
            {"event":"physical_ready","ordinal":0,"fact":{"raw_projection":{"text":"actual-bits"}}}
        ])
    }
    // These tests only check bounded immutable report projections, never create
    // a material/native/CPU capability or pretend to have run an actual model.
    #[test]
    fn captured_input_keeps_historical_deadline_and_exact_fingerprint() {
        let entries = captured_inputs(&trace()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0.query[7].to_bits(), input().query[7].to_bits());
        assert_eq!(
            entries[0]
                .0
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap(),
            input()
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap()
        );
        assert_eq!(entries[0].1, "actual-bits");
    }
    #[test]
    fn captured_input_rejects_missing_duplicate_raw_and_wrong_private_role() {
        let mut missing = trace();
        missing.as_array_mut().unwrap().pop();
        assert!(captured_inputs(&missing).is_err());
        let mut duplicate = trace();
        let ready = duplicate[1].clone();
        duplicate.as_array_mut().unwrap().push(ready);
        assert!(captured_inputs(&duplicate).is_err());
        let mut changed = trace();
        changed[0]["kind"] = json!("reply");
        assert!(captured_inputs(&changed).is_err());
        let mut changed = trace();
        changed[0]["input"]["extra"] = json!(0);
        assert!(captured_inputs(&changed).is_err());
    }
    #[test]
    fn audit_serialization_enforces_byte_ceiling_and_original_clock_before_growth() {
        let mut writer =
            BoundedAuditWriter::new(4, Instant::now() + std::time::Duration::from_secs(1));
        writer.write_all(b"1234").unwrap();
        assert!(writer.write_all(b"5").is_err());
        assert_eq!(writer.bytes, b"1234");
        assert!(writer.bytes.capacity() <= 4);
        let mut expired = BoundedAuditWriter::new(16, Instant::now());
        assert!(serde_json::to_writer(&mut expired, &json!({"x":1})).is_err());
        assert!(expired.bytes.is_empty());
        assert_eq!(expired.bytes.capacity(), 0);
    }
    #[test]
    fn raw_completion_and_actual_typed_cause_remain_owned_after_rejection() {
        // Synthetic byte values test the retention policy only. No actual NN
        // capability, owner, physical Ready or native execution is fabricated.
        let mut completed = vec![vec![1, 2, 3]];
        let cause = retain_and_check_completed(&mut completed, vec![0xff, 0x00], |_| {
            Err(CapturedRepairWitnessCause::Native {
                stage: "synthetic_failure_policy",
                error: RoleError::Backend("original backend diagnostic".into()),
            })
        })
        .unwrap_err();
        assert_eq!(completed, vec![vec![1, 2, 3], vec![0xff, 0x00]]);
        assert!(
            matches!(&cause,CapturedRepairWitnessCause::Native{stage:"synthetic_failure_policy",error:RoleError::Backend(s)}
            if s=="original backend diagnostic")
        );
        let mut writer =
            BoundedAuditWriter::new(1024, Instant::now() + std::time::Duration::from_secs(1));
        serde_json::to_writer(&mut writer, &DebugDiagnostic(&cause)).unwrap();
        assert!(
            std::str::from_utf8(&writer.bytes)
                .unwrap()
                .contains("original backend diagnostic")
        );
    }
    #[test]
    fn diagnostic_short_writes_charge_exact_bytes_and_errors_keep_attempt_credit() {
        struct Short(Vec<u8>);
        impl Write for Short {
            fn write(&mut self, v: &[u8]) -> io::Result<usize> {
                let n = v.len().min(2);
                self.0.extend_from_slice(&v[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut target = Short(Vec::new());
        let mut writer = ExternalBoundedAudit {
            writer: &mut target,
            attempted: MAX_CAPTURED_REPAIR_AUDIT_BYTES - 5,
            deadline: Instant::now() + std::time::Duration::from_secs(1),
        };
        writer.write_all(b"12345").unwrap();
        assert_eq!(writer.attempted, MAX_CAPTURED_REPAIR_AUDIT_BYTES);
        assert!(writer.write_all(b"6").is_err());
        assert_eq!(target.0, b"12345");
        struct Failed;
        impl Write for Failed {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("unknown partial output"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut failed = Failed;
        let mut writer = ExternalBoundedAudit {
            writer: &mut failed,
            attempted: 0,
            deadline: Instant::now() + std::time::Duration::from_secs(1),
        };
        assert!(writer.write_all(b"12345").is_err());
        assert_eq!(writer.attempted, 5);
    }
    #[test]
    fn diagnostic_flush_return_after_fixed_deadline_is_refused() {
        struct LateFlush {
            deadline: Instant,
            invoked: bool,
        }
        impl Write for LateFlush {
            fn write(&mut self, v: &[u8]) -> io::Result<usize> {
                Ok(v.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                self.invoked = true;
                while Instant::now() < self.deadline {
                    std::thread::yield_now();
                }
                Ok(())
            }
        }
        let deadline = Instant::now() + std::time::Duration::from_millis(100);
        let mut target = LateFlush {
            deadline,
            invoked: false,
        };
        let mut writer = ExternalBoundedAudit {
            writer: &mut target,
            attempted: 0,
            deadline,
        };
        assert!(writer.flush().is_err());
        assert!(target.invoked);
    }
    #[test]
    fn entire_request_group_omission_and_source_count_mismatch_are_refused() {
        // Report-shape regression only, with no synthetic native capability.
        let text = "first\nsecond\n";
        let mut body = json!({"trace_complete":true,"trace_rows":2,
            "native_bindings_observed":2,"native_ready_observed":2,
            "native_completion_unknown_observed":0,"native_terminal_missing":0,
            "native_receipt":{"startup_probe":null,"physically_completed_role_calls":2,
                "completed_role_inputs":2,"delivered_role_inputs":2,"search_consumed_role_inputs":2,
                "failed_physical_role_calls":0,"invalid_role_outputs":0,"canceled_requests":0,
                "expired_requests":0,"observer_failures":0,"completed_new_game_resets":1}});
        assert!(check_source_trace_counts(&body, text, 2).is_ok());
        assert!(check_source_trace_counts(&body, text, 1).is_err());
        assert!(check_source_trace_counts(&body, "first\n", 2).is_err());
        body["native_receipt"]["completed_role_inputs"] = json!(1);
        assert!(check_source_trace_counts(&body, text, 2).is_err());
        body["native_receipt"]["completed_role_inputs"] = json!(2);
        body["native_receipt"]["expired_requests"] = json!(1);
        assert!(check_source_trace_counts(&body, text, 2).is_err());
    }
}
