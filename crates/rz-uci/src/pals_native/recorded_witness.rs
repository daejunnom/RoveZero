//! Exact recorded-input inference witness, separate from product RoleQuery.
//!
//! Recorded deadline features are historical model data. They never replace the
//! live caller's absolute S/E/W or authorize a native Run from a JSON report.
use super::{NativeBackendStatsReceipt, NativeInvocationBudget, NativeRoleSourceIdentity};
use rz_eval::pals_model::{PalsModelInput, PalsRawOutput};
use sha2::{Digest as _, Sha256};
use std::time::Instant;

pub const RECORDED_INPUT_WITNESS_SCHEMA: &str = "rz-pals-native-recorded-input-witness/1";
pub const RECORDED_INPUT_WITNESS_SCOPE: &str = "captured_nn_input_reinference_and_independent_cpu_condition_reexecution;native_input_only;cpu_condition_and_source_capture_authority_separate";

/// Source identity for this additive witness implementation. The product input
/// semantics and product encoder digest remain unchanged.
pub fn recorded_input_witness_source_digest() -> [u8; 32] {
    Sha256::digest(include_bytes!("recorded_witness.rs")).into()
}

/// Actual NN-zero SnapshotStats ACK from the CPU Fresh invocation owner. This
/// is an observation at a known physical fence, not authority over later work.
/// The caller must keep its exclusive model borrow through the captured-input
/// loop. JSON cannot reconstruct this value or authenticate pristine ownership.
#[derive(serde::Serialize)]
pub struct NativeRecordedCpuPristineAck {
    pub(super) schema: &'static str,
    pub(super) observation: &'static str,
    pub(super) source_identity: NativeRoleSourceIdentity,
    pub(super) process_epoch: u64,
    pub(super) backend_stats: NativeBackendStatsReceipt,
    pub(super) started_elapsed_ns: u64,
    pub(super) completed_elapsed_ns: u64,
    pub(super) physical_completion_confirmed: bool,
    #[serde(skip)]
    pub(super) invocation_budget: NativeInvocationBudget,
    #[serde(skip)]
    pub(super) invocation_load_interval: (Instant, Instant),
    #[serde(skip)]
    pub(super) work_started_at: Instant,
    #[serde(skip)]
    pub(super) work_finished_at: Instant,
}
impl NativeRecordedCpuPristineAck {
    pub const fn schema(&self) -> &'static str {
        self.schema
    }
    pub const fn observation_scope(&self) -> &'static str {
        self.observation
    }
    pub const fn control_kind(&self) -> &'static str {
        "snapshot_stats"
    }
    pub fn source_identity(&self) -> &NativeRoleSourceIdentity {
        &self.source_identity
    }
    pub const fn process_epoch(&self) -> u64 {
        self.process_epoch
    }
    pub fn backend_stats(&self) -> &NativeBackendStatsReceipt {
        &self.backend_stats
    }
    pub const fn invocation_budget(&self) -> NativeInvocationBudget {
        self.invocation_budget
    }
    pub const fn invocation_load_interval(&self) -> (Instant, Instant) {
        self.invocation_load_interval
    }
    pub const fn work_started_at(&self) -> Instant {
        self.work_started_at
    }
    pub const fn work_finished_at(&self) -> Instant {
        self.work_finished_at
    }
    pub const fn physical_completion_confirmed(&self) -> bool {
        self.physical_completion_confirmed
    }
}

/// Issued only after the actual CPU owner's immutable job returns Ready and its
/// output and acknowledged before/after counters have been checked. No public
/// constructor, Clone or Deserialize is supplied. Serializing this observation
/// cannot restore the typed capability or authenticate the source child's data.
#[derive(serde::Serialize)]
pub struct NativeRecordedInputWitness {
    pub(super) schema: &'static str,
    pub(super) scope: &'static str,
    pub(super) implementation_sha256: [u8; 32],
    pub(super) source_identity: NativeRoleSourceIdentity,
    pub(super) process_epoch: u64,
    /// Witness-local high water reserved before the subsequently accepted submit;
    /// this is not a product Runtime ExecutionId or a reconstructed child ID.
    pub(super) lease_sequence: u64,
    pub(super) input_key: [u8; 32],
    pub(super) input: PalsModelInput,
    pub(super) raw_output: PalsRawOutput,
    pub(super) raw_output_bit_projection: String,
    pub(super) historical_deadline_feature_bits: u32,
    pub(super) started_elapsed_ns: u64,
    pub(super) completed_elapsed_ns: u64,
    pub(super) execution_until_elapsed_ns: u64,
    pub(super) whole_until_elapsed_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) invocation_load_elapsed_ns: Option<(u64, u64)>,
    pub(super) physical_completion_confirmed: bool,
    pub(super) backend_stats_before: NativeBackendStatsReceipt,
    pub(super) backend_stats_after: NativeBackendStatsReceipt,
    #[serde(skip)]
    pub(super) invocation_budget: NativeInvocationBudget,
    #[serde(skip)]
    pub(super) invocation_load_interval: Option<(Instant, Instant)>,
    #[serde(skip)]
    pub(super) work_started_at: Instant,
    #[serde(skip)]
    pub(super) work_finished_at: Instant,
}

impl NativeRecordedInputWitness {
    pub const fn schema(&self) -> &'static str {
        self.schema
    }
    pub const fn assurance_scope(&self) -> &'static str {
        self.scope
    }
    pub const fn implementation_sha256(&self) -> [u8; 32] {
        self.implementation_sha256
    }
    pub fn source_identity(&self) -> &NativeRoleSourceIdentity {
        &self.source_identity
    }
    pub const fn process_epoch(&self) -> u64 {
        self.process_epoch
    }
    pub const fn lease_sequence(&self) -> u64 {
        self.lease_sequence
    }
    pub fn input(&self) -> &PalsModelInput {
        &self.input
    }
    pub const fn input_key(&self) -> [u8; 32] {
        self.input_key
    }
    pub fn raw_output(&self) -> &PalsRawOutput {
        &self.raw_output
    }
    pub fn raw_output_bit_projection(&self) -> &str {
        &self.raw_output_bit_projection
    }
    pub const fn historical_deadline_feature_bits(&self) -> u32 {
        self.historical_deadline_feature_bits
    }
    pub const fn invocation_budget(&self) -> NativeInvocationBudget {
        self.invocation_budget
    }
    /// Observed successful CPU factory entry/return, not inferred from S or the
    /// first NN input. Generic loaders and synthetic lease fixtures return None.
    pub const fn invocation_load_interval(&self) -> Option<(Instant, Instant)> {
        self.invocation_load_interval
    }
    pub const fn work_started_at(&self) -> Instant {
        self.work_started_at
    }
    pub const fn work_finished_at(&self) -> Instant {
        self.work_finished_at
    }
    pub const fn physical_completion_confirmed(&self) -> bool {
        self.physical_completion_confirmed
    }
    pub fn backend_stats_before(&self) -> &NativeBackendStatsReceipt {
        &self.backend_stats_before
    }
    pub fn backend_stats_after(&self) -> &NativeBackendStatsReceipt {
        &self.backend_stats_after
    }
}
