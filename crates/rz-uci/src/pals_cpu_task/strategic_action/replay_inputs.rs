//! Caller-bound, result-free admission for a fresh frozen-parent replay.
//!
//! This boundary preserves the original action/1 and complete SemanticReceipt
//! UTF-8 bytes and requires separately supplied expected pins. It reconstructs
//! Rules and compares actual descriptors before constructing owner arguments.
//! It does not create a CPU engine, RoleModel, provider or FreshReplayOwner.
//! Python's Query/2/current/frozen/semantic capability and chronology checks are
//! still the independent caller's responsibility. Historical hashes are binding
//! declarations, never imported records, request IDs, model epochs or authority.
//!
//! ReplyOnly2n reserves conservative Repair role/store requirements, explicitly
//! reported as overreservation; its physical CPU allowance remains exactly 2N.
//! RepairEndpoint3n separately reserves 3N. RepairOpponent4n explicitly uses /2
//! source registration and reserves 4N plus an actual new C continuation.
//! No mode extends old remaining
//! declarations or changes the legacy request. Cleanup is an explicit partition
//! inside the original whole wall, not a fresh clock or whole-cost observation.

use super::super::{self as cpu_task, CpuTaskError, semantic};
use super::{ArtifactPin, StrategicError, StrategicTaskKind};
use crate::engine::OwnerRegistry;
use rz_position::{BoardMove, HistoryCompleteness, HistoryOrigin, Position};
use rz_search::cpu::{CpuConfig, CpuProfile};
use rz_search::pals::engine::replay::{
    DefendResponseReplayPlan, ReplayError, ReplayHistoricalPins,
    repair_opponent_replay_requirements, repair_replay_requirements,
};
use rz_search::pals::engine::{PalsConfig, PalsLimits};
use semantic::{MoveMeaning, MoveTokenKind, SemanticReceipt, SemanticRequest};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::TryReserveError;
use std::fmt;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const SCHEMA: &str = "rz-pals-frozen-replay-inputs/1";
pub const REGISTRATION_SCHEMA: &str = "rz-pals-frozen-replay-consumer-registration/1";
pub const OPPONENT_SCHEMA: &str = "rz-pals-frozen-replay-inputs/2";
pub const OPPONENT_REGISTRATION_SCHEMA: &str = "rz-pals-frozen-replay-consumer-registration/2";
pub const SCOPE: &str = "caller_registered_original_bytes_to_rules_checked_owner_arguments";
pub const BINARY_DECLARATION_SCOPE: &str =
    "external_caller_registered_digest_not_loaded_image_observation";
pub const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_REGISTRATION_BYTES: usize = 16 * 1024;
pub const MAX_SEMANTIC_RECEIPT_BYTES: usize = cpu_task::MAX_RESPONSE_BYTES;
pub const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_SOURCE_BYTES: u64 = 1024 * 1024;
const MAX_KNOWN_HISTORY_POSITIONS: usize = 4096;
const DIAGNOSTIC_OUTPUT_BYTES: usize = 1024;
pub const MAX_CONSTRUCTION_CREDIT_BYTES: usize = 16 * 1024 * 1024;
const CONSTRUCTION_TOPOLOGY_BYTES: usize = 64 * 1024;
const CONSTRUCTION_RAW_PAYLOAD_COPIES: usize = 3;

/// registration is the new ReplayConsumerRegistration wire, not the historical
/// semantic producer's registration. All three byte slices remain original.
#[derive(Clone, Copy)]
pub struct ReplayOriginals<'a> {
    pub registration: &'a [u8],
    pub prepared_action: &'a [u8],
    pub semantic_receipt: &'a [u8],
}
impl fmt::Debug for ReplayOriginals<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplayOriginals")
            .field("registration_bytes", &self.registration.len())
            .field("prepared_action_bytes", &self.prepared_action.len())
            .field("semantic_receipt_bytes", &self.semantic_receipt.len())
            .finish()
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ReplayPreparationDeclaration<'a> {
    pub mode: ReplayInputMode,
    pub config: &'a ReplayConfigInput,
    pub resources: &'a ReplayResourceDeclaration,
}
#[derive(Clone, Copy, Debug)]
pub struct ReplayConstructionBudget {
    pub max_outer_bytes: usize,
    pub max_construction_credit_bytes: usize,
}

/// Policy prepayment for known owned construction backing only. Neither these
/// numbers nor retained Vec capacity measure allocator, validator or RSS peak.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ReplayConstructionCredit {
    pub raw_utf8_bytes: usize,
    /// Source-owned copy allowance, not an observed allocator or RSS peak.
    pub raw_payload_copies: usize,
    pub context_json_bytes: usize,
    pub outer_json_bytes: usize,
    pub fixed_topology_bytes: usize,
    pub policy_bytes: usize,
    pub final_retained_capacity_bytes: usize,
    pub allocator_peak_observed: bool,
    pub validator_peak_observed: bool,
    pub rss_peak_observed: bool,
}
#[derive(Debug, Serialize)]
pub struct ReplayPreparationAudit {
    pub scope: &'static str,
    pub construction: ReplayConstructionCredit,
    pub input_admission: ReplayInputAudit,
    pub elapsed_ms: u64,
}

/// Immutable owned outer wire, checked by the existing scoped input boundary.
/// It is not a Query capability, caller registration or native execution token.
pub struct PreparedReplayRequest {
    bytes: Vec<u8>,
    artifact: ArtifactPin,
    audit: ReplayPreparationAudit,
    whole_deadline: Instant,
    execution_deadline: Instant,
}
impl fmt::Debug for PreparedReplayRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedReplayRequest")
            .field("artifact", &self.artifact)
            .field("construction", &self.audit.construction)
            .finish_non_exhaustive()
    }
}
impl PreparedReplayRequest {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn artifact(&self) -> &ArtifactPin {
        &self.artifact
    }
    pub fn audit(&self) -> &ReplayPreparationAudit {
        &self.audit
    }
    pub fn deadline(&self) -> Instant {
        self.whole_deadline
    }
    pub fn execution_deadline(&self) -> Instant {
        self.execution_deadline
    }

    /// Result verification only, under the original W. This never reopens E or
    /// creates CPU/model/provider work. The reconstructed history is temporary;
    /// no Rules graph is retained throughout native execution or in its report.
    pub(super) fn reconstruct_result_root(
        &self,
        cancel: &AtomicBool,
    ) -> Result<ReplayResultRoot, ReplayInputError> {
        let control = || {
            if cancel.load(Ordering::Acquire) || Instant::now() >= self.whole_deadline {
                Err(ReplayInputError::new(
                    "result_rules_control",
                    "original whole window expired or result verification canceled",
                ))
            } else {
                Ok(())
            }
        };
        control()?;
        if self.bytes.len() > MAX_REQUEST_BYTES || super::pin(&self.bytes) != self.artifact {
            return Err(ReplayInputError::new(
                "result_rules_input",
                "prepared backing differs",
            ));
        }
        let request: ReplayInputRequest = serde_json::from_slice(&self.bytes)
            .map_err(|e| ReplayInputError::new("result_rules_input", e))?;
        let started = self
            .whole_deadline
            .checked_sub(Duration::from_millis(
                self.audit.input_admission.whole_wall_ms,
            ))
            .ok_or_else(|| {
                ReplayInputError::new("result_rules_clock", "original start unavailable")
            })?;
        // Decode the immutable, previously admitted original action with its
        // original S, not a fresh admission clock or an execution extension.
        let (prepared, _) = super::decode(request.prepared_action_raw.as_bytes(), started)
            .map_err(|e| ReplayInputError::strategic("result_rules_action", e))?;
        let cpu = cpu_task::decode_request(prepared.cpu_request_raw.as_bytes())
            .map_err(|e| ReplayInputError::cpu("result_rules_cpu_input", e))?;
        let receipt: SemanticReceipt = serde_json::from_str(&request.semantic_receipt_raw)
            .map_err(|e| ReplayInputError::new("result_rules_semantic", e))?;
        let semantic_request = original_root_request(&cpu, &receipt);
        control()?;
        let owners = Arc::new(OwnerRegistry::default());
        let root = semantic::prepare_root(&semantic_request, &owners, self.whole_deadline)
            .map_err(|e| ReplayInputError::semantic("result_rules_root", e))?;
        if root.history_completeness() != HistoryCompleteness::Complete
            || root.history_origin() != HistoryOrigin::StartPosition
            || root.known_history_len() == 0
            || root.known_history_len() > MAX_KNOWN_HISTORY_POSITIONS
        {
            return Err(ReplayInputError::new(
                "result_rules_history",
                "complete startpos history required",
            ));
        }
        let actual = semantic::describe(
            &root,
            &owners,
            MoveTokenKind::RootLegal,
            self.whole_deadline,
        )
        .map_err(|e| ReplayInputError::semantic("result_rules_descriptor", e))?;
        same_serialized(&actual, &receipt.root, "result_rules_descriptor")?;
        if actual.rules_state_sha256 != cpu.rules_state_sha256
            || actual.rules_history_sha256 != cpu.rules_history_sha256
            || actual.board_fen != cpu.expected_board_fen
            || actual.play_status != "ongoing"
        {
            return Err(ReplayInputError::new(
                "result_rules_identity",
                "original complete Rules root differs",
            ));
        }
        let prefix = cpu_task::decode_moves(&cpu.prefix)
            .map_err(|e| ReplayInputError::cpu("result_rules_prefix", e))?;
        control()?;
        Ok(ReplayResultRoot {
            root,
            prefix,
            line_plies: request.config.line_plies,
            whole_deadline: self.whole_deadline,
        })
    }
}

/// Private temporary input to the report's Rules check, never an execution or
/// historical owner capability. Construction remains with PreparedReplayRequest.
pub(super) struct ReplayResultRoot {
    pub(super) root: Position,
    pub(super) prefix: Vec<BoardMove>,
    pub(super) line_plies: usize,
    pub(super) whole_deadline: Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayOriginalKind {
    Registration,
    PreparedAction,
    SemanticReceipt,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayConstructionWriterFailure {
    Deadline,
    ByteLimit,
    LengthOverflow,
    RetainedCapacity,
}
#[derive(Debug)]
pub enum ReplayPreparationCause {
    Input(Box<ReplayInputError>),
    Utf8 {
        original: ReplayOriginalKind,
        source: std::str::Utf8Error,
    },
    Reserve(TryReserveError),
    Serialization {
        source: serde_json::Error,
        writer: Option<ReplayConstructionWriterFailure>,
    },
    Refusal {
        reason: &'static str,
        actual: Option<usize>,
        limit: Option<usize>,
    },
}

/// Failed bytes are owned evidence, never a successful prepared input. A partial
/// serialization has no complete artifact pin and is explicitly distinguished.
pub struct RejectedReplayBytes {
    bytes: Vec<u8>,
    artifact: Option<ArtifactPin>,
    serialization_complete: bool,
}
impl fmt::Debug for RejectedReplayBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RejectedReplayBytes")
            .field("bytes", &self.bytes.len())
            .field("artifact", &self.artifact)
            .field("serialization_complete", &self.serialization_complete)
            .finish()
    }
}
impl RejectedReplayBytes {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn artifact(&self) -> Option<&ArtifactPin> {
        self.artifact.as_ref()
    }
    pub fn serialization_complete(&self) -> bool {
        self.serialization_complete
    }
}
#[derive(Debug, Serialize)]
pub struct ReplayPreparationDiagnostics {
    pub expected_semantic_receipt_producer_scope: SemanticReceiptProducerScope,
    pub elapsed_ms: u64,
    pub deadline_exceeded: bool,
    pub execution_deadline_exceeded: bool,
    pub original_whole_deadline_retained: bool,
    pub whole_wall_ms: Option<u64>,
    pub cleanup_reserve_ms: Option<u64>,
    pub output_limit: usize,
    #[serde(skip)]
    original_started: Instant,
    #[serde(skip)]
    clock: Option<Clock>,
}
impl ReplayPreparationDiagnostics {
    pub fn original_started(&self) -> Instant {
        self.original_started
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.clock.map(|c| c.whole_deadline)
    }
    pub fn execution_deadline(&self) -> Option<Instant> {
        self.clock.map(|c| c.execution_deadline)
    }
}
#[derive(Debug)]
pub struct ReplayPreparationError {
    pub stage: &'static str,
    pub cause: Box<ReplayPreparationCause>,
    pub diagnostics: Box<ReplayPreparationDiagnostics>,
    pub rejected_outer: Option<Box<RejectedReplayBytes>>,
}
impl fmt::Display for ReplayPreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "replay preparation failed at {}", self.stage)
    }
}
impl std::error::Error for ReplayPreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.cause.as_ref() {
            ReplayPreparationCause::Input(error) => Some(error.as_ref()),
            ReplayPreparationCause::Utf8 { source, .. } => Some(source),
            ReplayPreparationCause::Reserve(source) => Some(source),
            ReplayPreparationCause::Serialization { source, .. } => Some(source),
            ReplayPreparationCause::Refusal { .. } => None,
        }
    }
}

/// Independently registered scope of the historical SemanticReceipt producer.
/// Exact comparison observes a caller declaration, never the current replay
/// image, Query capability or the original producer's loaded-image proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum SemanticReceiptProducerScope {
    #[serde(rename = "dispatcher_compared_verified_argument")]
    LibraryDispatcherArgument,
    #[serde(rename = "linux_loaded_executable_inode")]
    LinuxLoadedExecutableInode,
    #[serde(rename = "current_exe_path_hash")]
    CurrentExePathHash,
}
impl SemanticReceiptProducerScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LibraryDispatcherArgument => "dispatcher_compared_verified_argument",
            Self::LinuxLoadedExecutableInode => "linux_loaded_executable_inode",
            Self::CurrentExePathHash => "current_exe_path_hash",
        }
    }
}
impl<'de> Deserialize<'de> for SemanticReceiptProducerScope {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(SemanticScopeVisitor)
    }
}
struct SemanticScopeVisitor;
impl<'de> serde::de::Visitor<'de> for SemanticScopeVisitor {
    type Value = SemanticReceiptProducerScope;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("one exact registered SemanticReceipt producer scope string")
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
        match value {
            "dispatcher_compared_verified_argument" => Ok(Self::Value::LibraryDispatcherArgument),
            "linux_loaded_executable_inode" => Ok(Self::Value::LinuxLoadedExecutableInode),
            "current_exe_path_hash" => Ok(Self::Value::CurrentExePathHash),
            _ => Err(E::unknown_variant(
                value,
                &[
                    "dispatcher_compared_verified_argument",
                    "linux_loaded_executable_inode",
                    "current_exe_path_hash",
                ],
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayInputMode {
    #[serde(rename = "reply_only_2n")]
    ReplyOnly2n,
    #[serde(rename = "repair_endpoint_3n")]
    RepairEndpoint3n,
    #[serde(rename = "repair_opponent_4n")]
    RepairOpponent4n,
}
impl ReplayInputMode {
    fn checks(self) -> u64 {
        match self {
            Self::ReplyOnly2n => 2,
            Self::RepairEndpoint3n => 3,
            Self::RepairOpponent4n => 4,
        }
    }
    pub const fn input_schema(self) -> &'static str {
        match self {
            Self::RepairOpponent4n => OPPONENT_SCHEMA,
            Self::ReplyOnly2n | Self::RepairEndpoint3n => SCHEMA,
        }
    }
    pub const fn registration_schema(self) -> &'static str {
        match self {
            Self::RepairOpponent4n => OPPONENT_REGISTRATION_SCHEMA,
            Self::ReplyOnly2n | Self::RepairEndpoint3n => REGISTRATION_SCHEMA,
        }
    }
}

/// Source-owned finite reservations shared by admission, launch preparation and
/// native observation. The old 2N lane retains its conservative 3N role/store
/// reservation; actual observer roles and CPU stages are declared separately.
/// These counts establish neither allocation nor work/physical completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayModeRequirements {
    cpu_nodes: u64,
    role_calls: u64,
    actual_role_calls: u64,
    nodes: usize,
    records: usize,
    observations: usize,
    line_chunks: usize,
    stages: usize,
    cpu_checks: usize,
}
impl ReplayModeRequirements {
    pub const fn cpu_nodes(self) -> u64 {
        self.cpu_nodes
    }
    pub const fn role_calls(self) -> u64 {
        self.role_calls
    }
    pub const fn nodes(self) -> usize {
        self.nodes
    }
    pub const fn records(self) -> usize {
        self.records
    }
    pub const fn observations(self) -> usize {
        self.observations
    }
    pub const fn line_chunks(self) -> usize {
        self.line_chunks
    }
    pub const fn stages(self) -> usize {
        self.stages
    }
    pub const fn actual_role_calls(self) -> u64 {
        self.actual_role_calls
    }
    pub const fn cpu_checks(self) -> usize {
        self.cpu_checks
    }
}
pub fn replay_mode_requirements(
    mode: ReplayInputMode,
    line_plies: usize,
    prefix_plies: usize,
    nodes_per_check: u64,
) -> Result<ReplayModeRequirements, ReplayError> {
    let (role_calls, nodes, records, observations, line_chunks, stages) =
        if mode == ReplayInputMode::RepairOpponent4n {
            let required =
                repair_opponent_replay_requirements(line_plies, prefix_plies, nodes_per_check)?;
            (
                required.role_calls(),
                required.nodes(),
                required.records(),
                required.observations(),
                required.line_chunks(),
                required.stages(),
            )
        } else {
            let required = repair_replay_requirements(line_plies, prefix_plies, nodes_per_check)?;
            (
                required.role_calls(),
                required.nodes(),
                required.records(),
                required.observations(),
                required.line_chunks(),
                required.stages(),
            )
        };
    let actual_role_calls = if mode == ReplayInputMode::ReplyOnly2n {
        u64::try_from(line_plies)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or(ReplayError::InvalidPlan("reply role bound overflow"))?
    } else {
        role_calls
    };
    let cpu_nodes = nodes_per_check
        .checked_mul(mode.checks())
        .ok_or(ReplayError::InvalidPlan("mode CPU reservation overflow"))?;
    Ok(ReplayModeRequirements {
        cpu_nodes,
        role_calls,
        actual_role_calls,
        nodes,
        records,
        observations,
        line_chunks,
        stages,
        cpu_checks: mode.checks() as usize,
    })
}

/// Four Query/2 parent identities. Encoding stays here because the existing
/// search-local ReplayHistoricalPins deliberately does not contain it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayParentPins {
    pub parent_input_sha256: String,
    pub current_view_sha256: String,
    pub frozen_admission_sha256: String,
    pub encoding_sha256: String,
}

/// These distinct domains must not be equated: semantic capability identity,
/// SemanticReceipt context/meaning identities, and original receipt byte hash.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayBindingPins {
    pub query_sha256: String,
    pub catalogue_artifact: ArtifactPin,
    pub before_result_artifact: ArtifactPin,
    pub prior_ledger_sha256: String,
    pub semantic_input_sha256: String,
    pub semantic_context_sha256: String,
    pub semantic_branch_meaning_sha256: String,
    pub semantic_before_result_anchor_sha256: String,
    pub cpu_request_artifact: ArtifactPin,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayRegisteredArtifacts {
    pub legacy_cpu_binary: ArtifactPin,
    pub replay_binary: ArtifactPin,
    pub engine_source: ArtifactPin,
    pub wrapper_source: ArtifactPin,
    pub replay_source: ArtifactPin,
    pub provider_factory_source: ArtifactPin,
    // Absent in the old wire and required only by the explicit /2 registration.
    // An explicit null is refused rather than becoming an absent legacy field.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_artifact"
    )]
    pub opponent_recheck_source: Option<ArtifactPin>,
}

fn present_artifact<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ArtifactPin>, D::Error> {
    ArtifactPin::deserialize(deserializer).map(Some)
}

/// Explicit negative authority fields are part of both closed wire schemas.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayAuthorities {
    pub query_revalidated_by_rust: bool,
    pub semantic_capability_revalidated_by_rust: bool,
    pub old_source_admission_verified_by_rust: bool,
    pub loaded_image_observed: bool,
    pub native_action_causal_bridge_observed: bool,
    pub full_repair_recheck_completed: bool,
    pub whole_action_cost_observed: bool,
    pub physical_closure_observed: bool,
    pub utility_authority: bool,
    pub target_authority: bool,
    pub training_authority: bool,
    pub product_verifier_enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConsumerRegistration {
    pub schema: String,
    pub binary_pin_scope: String,
    pub artifacts: ReplayRegisteredArtifacts,
    pub legacy_cpu_max_checks: u8,
    pub legacy_cpu_profile_sha256: String,
    pub provider_factory_id: String,
    pub authorities: ReplayAuthorities,
    pub context_sha256: String,
}

/// Supplied independently of stdin, normally after the caller's existing
/// Query/2 and prepared action checks and literal source registration. This is
/// not a Rust certification of those checks and has no self-pin fallback.
#[derive(Clone, Debug)]
pub struct ReplayExpectedPins {
    pub registration_artifact: ArtifactPin,
    pub prepared_action_artifact: ArtifactPin,
    pub semantic_receipt_artifact: ArtifactPin,
    pub parent: ReplayParentPins,
    pub binding: ReplayBindingPins,
    pub registered_artifacts: ReplayRegisteredArtifacts,
    pub legacy_cpu_profile_sha256: String,
    pub provider_factory_id: String,
    pub semantic_binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfigInput {
    pub beam_width: usize,
    pub line_plies: usize,
    pub max_nodes: usize,
    pub max_records: usize,
    pub max_role_calls: u64,
    pub cpu_nodes_per_task: u64,
}
impl ReplayConfigInput {
    fn config(&self) -> PalsConfig {
        PalsConfig {
            beam_width: self.beam_width,
            line_plies: self.line_plies,
            max_nodes: self.max_nodes,
            max_records: self.max_records,
            max_role_calls: self.max_role_calls,
            cpu_nodes_per_task: self.cpu_nodes_per_task,
        }
    }
}

/// Required observations/line_chunks/stages are source-owned worst-case Repair
/// reservations, not measured peaks or copies of private StoreLimits. Config
/// supplies the existing owner's finite node/record/role caps. The subsequent
/// owner still validates and allocates its own stores; no allocation is proved.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayResourceDeclaration {
    pub cpu_nodes: u64,
    pub role_calls: u64,
    pub store_nodes: usize,
    pub store_records: usize,
    pub required_observations: usize,
    pub required_line_chunks: usize,
    pub required_stages: usize,
    pub max_rounds: u64,
    pub output_bytes: usize,
    pub whole_wall_ms: u64,
    pub cleanup_reserve_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayInputRequest {
    pub schema: String,
    pub mode: ReplayInputMode,
    pub registration_raw: String,
    pub registration_artifact: ArtifactPin,
    pub prepared_action_raw: String,
    pub prepared_action_artifact: ArtifactPin,
    /// Complete original SemanticReceipt, not a re-rendered descriptor subset.
    pub semantic_receipt_raw: String,
    pub semantic_receipt_artifact: ArtifactPin,
    pub parent: ReplayParentPins,
    pub binding: ReplayBindingPins,
    pub config: ReplayConfigInput,
    pub resources: ReplayResourceDeclaration,
    pub authorities: ReplayAuthorities,
    pub context_sha256: String,
}

#[derive(Clone, Copy, Debug)]
struct Clock {
    whole_deadline: Instant,
    execution_deadline: Instant,
    whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    output_limit: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", content = "error", rename_all = "snake_case")]
pub enum ReplayInputSourceError {
    Strategic(Box<StrategicError>),
    Semantic(Box<semantic::SemanticError>),
    Cpu(Box<CpuTaskError>),
}

#[derive(Clone, Serialize)]
pub struct ReplayInputError {
    pub code: &'static str,
    pub stage: &'static str,
    /// Expected declaration only; presence does not mean receipt comparison or
    /// historical loaded-image admission completed successfully.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_semantic_receipt_producer_scope: Option<SemanticReceiptProducerScope>,
    pub message: Box<str>,
    pub elapsed_ms: Option<u64>,
    pub deadline_exceeded: bool,
    pub execution_deadline_exceeded: bool,
    pub original_whole_deadline_retained: bool,
    pub whole_wall_ms: Option<u64>,
    pub cleanup_reserve_ms: Option<u64>,
    #[serde(skip)]
    pub output_limit: usize,
    pub source_error: Option<Box<ReplayInputSourceError>>,
}
impl fmt::Debug for ReplayInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // NativeReplayPrimary archives this Debug view in its JSON diagnostic.
        // Legacy None must retain the old field order and omit the new field.
        let mut view = formatter.debug_struct("ReplayInputError");
        view.field("code", &self.code).field("stage", &self.stage);
        if let Some(scope) = self.expected_semantic_receipt_producer_scope {
            view.field("expected_semantic_receipt_producer_scope", &Some(scope));
        }
        view.field("message", &self.message)
            .field("elapsed_ms", &self.elapsed_ms)
            .field("deadline_exceeded", &self.deadline_exceeded)
            .field(
                "execution_deadline_exceeded",
                &self.execution_deadline_exceeded,
            )
            .field(
                "original_whole_deadline_retained",
                &self.original_whole_deadline_retained,
            )
            .field("whole_wall_ms", &self.whole_wall_ms)
            .field("cleanup_reserve_ms", &self.cleanup_reserve_ms)
            .field("output_limit", &self.output_limit)
            .field("source_error", &self.source_error)
            .finish()
    }
}
impl ReplayInputError {
    pub fn new(stage: &'static str, message: impl fmt::Display) -> Self {
        Self {
            code: "frozen_replay_input_failed",
            stage,
            expected_semantic_receipt_producer_scope: None,
            message: message
                .to_string()
                .chars()
                .take(1024)
                .collect::<String>()
                .into_boxed_str(),
            elapsed_ms: None,
            deadline_exceeded: false,
            execution_deadline_exceeded: false,
            original_whole_deadline_retained: false,
            whole_wall_ms: None,
            cleanup_reserve_ms: None,
            output_limit: DIAGNOSTIC_OUTPUT_BYTES,
            source_error: None,
        }
    }
    fn strategic(stage: &'static str, error: StrategicError) -> Self {
        let mut result = Self::new(stage, &error);
        result.source_error = Some(Box::new(ReplayInputSourceError::Strategic(Box::new(error))));
        result
    }
    fn semantic(stage: &'static str, error: semantic::SemanticError) -> Self {
        let mut result = Self::new(stage, &error);
        result.source_error = Some(Box::new(ReplayInputSourceError::Semantic(Box::new(error))));
        result
    }
    fn cpu(stage: &'static str, error: CpuTaskError) -> Self {
        let mut result = Self::new(stage, &error);
        result.source_error = Some(Box::new(ReplayInputSourceError::Cpu(Box::new(error))));
        result
    }
    fn stamped(
        mut self,
        started: Instant,
        clock: Option<Clock>,
        expected_scope: Option<SemanticReceiptProducerScope>,
    ) -> Self {
        self.expected_semantic_receipt_producer_scope = expected_scope;
        self.elapsed_ms = Some(cpu_task::milliseconds(started.elapsed()));
        if let Some(clock) = clock {
            self.output_limit = clock.output_limit;
            self.original_whole_deadline_retained = true;
            self.whole_wall_ms = Some(clock.whole_wall_ms);
            self.cleanup_reserve_ms = Some(clock.cleanup_reserve_ms);
            self.deadline_exceeded |= Instant::now() >= clock.whole_deadline;
            self.execution_deadline_exceeded |= Instant::now() >= clock.execution_deadline;
        }
        self
    }
}
impl fmt::Display for ReplayInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.stage, self.message)
    }
}
impl std::error::Error for ReplayInputError {}

#[derive(Clone, Debug, Serialize)]
pub struct ReplayInputAudit {
    pub schema: &'static str,
    pub scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_semantic_receipt_producer_scope: Option<SemanticReceiptProducerScope>,
    pub mode: ReplayInputMode,
    pub input_artifact: ArtifactPin,
    pub registration_artifact: ArtifactPin,
    pub prepared_action_artifact: ArtifactPin,
    pub semantic_receipt_artifact: ArtifactPin,
    pub parent: ReplayParentPins,
    pub binding: ReplayBindingPins,
    pub registered_artifacts: ReplayRegisteredArtifacts,
    pub provider_factory_id: String,
    pub legacy_cpu_profile_sha256: String,
    pub binary_pin_scope: &'static str,
    pub cpu_allowance: u64,
    pub conservative_repair_role_store_overreservation: bool,
    pub whole_wall_ms: u64,
    pub cleanup_reserve_ms: u64,
    pub elapsed_ms: u64,
    pub cpu_checks: u8,
    pub cpu_engine_created: bool,
    pub model_created: bool,
    pub provider_created: bool,
    pub actual_utility_groups: u8,
    pub authorities: ReplayAuthorities,
}

/// Private checked fields can only be produced by the replay input check entry
/// points. The
/// consuming conversion supplies existing typed owner inputs, never a provider
/// or a callable dispatch. Readonly raw access preserves original byte evidence.
pub struct CheckedReplayInputs {
    request: ReplayInputRequest,
    registration: ReplayConsumerRegistration,
    expected_semantic_receipt_producer_scope: Option<SemanticReceiptProducerScope>,
    cpu_request_raw: String,
    input_artifact: ArtifactPin,
    config: PalsConfig,
    root: Position,
    plan: DefendResponseReplayPlan,
    limits: PalsLimits,
    clock: Clock,
    started: Instant,
}
impl fmt::Debug for CheckedReplayInputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckedReplayInputs")
            .field("mode", &self.request.mode)
            .field("input_artifact", &self.input_artifact)
            .field("config", &self.config)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}
impl CheckedReplayInputs {
    /// Explicit expected declaration already compared with the original receipt.
    /// None preserves the legacy library-only entry point and its omitted wire.
    pub fn expected_semantic_receipt_producer_scope(&self) -> Option<SemanticReceiptProducerScope> {
        self.expected_semantic_receipt_producer_scope
    }
    pub fn mode(&self) -> ReplayInputMode {
        self.request.mode
    }
    /// Original whole invocation deadline, including cleanup, from caller start.
    pub fn deadline(&self) -> Instant {
        self.clock.whole_deadline
    }
    /// Declared work partition, also from that original start, passed to owner.
    pub fn execution_deadline(&self) -> Instant {
        self.clock.execution_deadline
    }
    pub fn output_limit(&self) -> usize {
        self.clock.output_limit
    }
    pub fn resources(&self) -> &ReplayResourceDeclaration {
        &self.request.resources
    }
    pub fn parent_pins(&self) -> &ReplayParentPins {
        &self.request.parent
    }
    pub fn binding_pins(&self) -> &ReplayBindingPins {
        &self.request.binding
    }
    pub fn original_prepared_action(&self) -> &[u8] {
        self.request.prepared_action_raw.as_bytes()
    }
    pub fn original_cpu_request(&self) -> &[u8] {
        self.cpu_request_raw.as_bytes()
    }
    pub fn original_semantic_receipt(&self) -> &[u8] {
        self.request.semantic_receipt_raw.as_bytes()
    }
    pub fn original_registration(&self) -> &[u8] {
        self.request.registration_raw.as_bytes()
    }
    pub fn registration(&self) -> &ReplayConsumerRegistration {
        &self.registration
    }
    pub fn into_owner_args(self) -> (PalsConfig, Position, DefendResponseReplayPlan, PalsLimits) {
        (self.config, self.root, self.plan, self.limits)
    }
    pub fn audit(&self) -> ReplayInputAudit {
        ReplayInputAudit {
            schema: self.request.mode.input_schema(),
            scope: SCOPE,
            expected_semantic_receipt_producer_scope: self.expected_semantic_receipt_producer_scope,
            mode: self.request.mode,
            input_artifact: self.input_artifact.clone(),
            registration_artifact: self.request.registration_artifact.clone(),
            prepared_action_artifact: self.request.prepared_action_artifact.clone(),
            semantic_receipt_artifact: self.request.semantic_receipt_artifact.clone(),
            parent: self.request.parent.clone(),
            binding: self.request.binding.clone(),
            registered_artifacts: self.registration.artifacts.clone(),
            provider_factory_id: self.registration.provider_factory_id.clone(),
            legacy_cpu_profile_sha256: self.registration.legacy_cpu_profile_sha256.clone(),
            binary_pin_scope: BINARY_DECLARATION_SCOPE,
            cpu_allowance: self.limits.max_cpu_nodes,
            conservative_repair_role_store_overreservation: self.request.mode
                == ReplayInputMode::ReplyOnly2n,
            whole_wall_ms: self.clock.whole_wall_ms,
            cleanup_reserve_ms: self.clock.cleanup_reserve_ms,
            elapsed_ms: cpu_task::milliseconds(self.started.elapsed()),
            cpu_checks: 0,
            cpu_engine_created: false,
            model_created: false,
            provider_created: false,
            actual_utility_groups: 0,
            authorities: ReplayAuthorities::default(),
        }
    }
}

fn check_clock(clock: Clock) -> Result<(), ReplayInputError> {
    if Instant::now() >= clock.execution_deadline {
        Err(ReplayInputError::new(
            "original_deadline",
            "original work partition expired; whole deadline is unchanged",
        ))
    } else {
        Ok(())
    }
}
fn hash(value: &Value, stage: &'static str) -> Result<String, ReplayInputError> {
    cpu_task::json_digest(value).map_err(|e| ReplayInputError::cpu(stage, e))
}
fn closed_context<T: Serialize>(schema: &str, value: &T) -> Result<String, ReplayInputError> {
    let mut body = serde_json::to_value(value)
        .map_err(|e| ReplayInputError::new("context_serialization", e))?;
    body.as_object_mut()
        .ok_or_else(|| ReplayInputError::new("context_serialization", "object required"))?
        .remove("context_sha256");
    hash(&json!([schema, body]), "context_identity")
}
fn digest_bytes(sha: &str) -> Result<[u8; 32], ReplayInputError> {
    if !cpu_task::valid_sha(sha) {
        return Err(ReplayInputError::new(
            "identity",
            "lowercase SHA256 required",
        ));
    }
    let mut result = [0; 32];
    for (at, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&sha[2 * at..2 * at + 2], 16)
            .map_err(|e| ReplayInputError::new("identity", e))?;
    }
    Ok(result)
}
fn bounded_pin(pin: &ArtifactPin, maximum: u64) -> Result<(), ReplayInputError> {
    super::checked_pin(pin, maximum)
        .map_err(|e| ReplayInputError::strategic("artifact_identity", e))
}
fn original_pin(
    raw: &str,
    wire: &ArtifactPin,
    expected: &ArtifactPin,
    maximum: usize,
) -> Result<(), ReplayInputError> {
    bounded_pin(expected, maximum as u64)?;
    bounded_pin(wire, maximum as u64)?;
    if raw.is_empty()
        || raw.len() > maximum
        || wire != expected
        || super::pin(raw.as_bytes()) != *expected
    {
        return Err(ReplayInputError::new(
            "original_artifact_identity",
            "original UTF-8 extent/hash differs from independently supplied pin",
        ));
    }
    Ok(())
}
fn same_serialized<T: Serialize>(
    left: &T,
    right: &T,
    stage: &'static str,
) -> Result<(), ReplayInputError> {
    let left = serde_json::to_value(left).map_err(|e| ReplayInputError::new(stage, e))?;
    let right = serde_json::to_value(right).map_err(|e| ReplayInputError::new(stage, e))?;
    if left != right {
        return Err(ReplayInputError::new(
            stage,
            "actual Rules descriptor/ordered meaning differs from original receipt",
        ));
    }
    Ok(())
}

fn check_registration(
    raw: &str,
    expected: &ReplayExpectedPins,
    mode: ReplayInputMode,
) -> Result<ReplayConsumerRegistration, ReplayInputError> {
    let registration: ReplayConsumerRegistration =
        serde_json::from_str(raw).map_err(|e| ReplayInputError::new("registration_json", e))?;
    if registration.schema != mode.registration_schema()
        || registration.artifacts.opponent_recheck_source.is_some()
            != (mode == ReplayInputMode::RepairOpponent4n)
        || registration.binary_pin_scope != BINARY_DECLARATION_SCOPE
        || registration.legacy_cpu_max_checks != 2
        || registration.authorities != ReplayAuthorities::default()
        || registration.provider_factory_id.is_empty()
        || registration.provider_factory_id.len() > 128
        || !registration
            .provider_factory_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        || registration.provider_factory_id != expected.provider_factory_id
        || registration.legacy_cpu_profile_sha256 != expected.legacy_cpu_profile_sha256
        || registration.artifacts != expected.registered_artifacts
    {
        return Err(ReplayInputError::new(
            "registration_binding",
            "unsupported or independently unregistered source/provider/binary/profile/scope",
        ));
    }
    bounded_pin(
        &registration.artifacts.legacy_cpu_binary,
        cpu_task::MAX_SELF_BINARY_BYTES,
    )?;
    bounded_pin(
        &registration.artifacts.replay_binary,
        cpu_task::MAX_SELF_BINARY_BYTES,
    )?;
    for source in [
        &registration.artifacts.engine_source,
        &registration.artifacts.wrapper_source,
        &registration.artifacts.replay_source,
        &registration.artifacts.provider_factory_source,
    ] {
        bounded_pin(source, MAX_SOURCE_BYTES)?;
    }
    if let Some(source) = &registration.artifacts.opponent_recheck_source {
        bounded_pin(source, MAX_SOURCE_BYTES)?;
    }
    digest_bytes(&registration.legacy_cpu_profile_sha256)?;
    digest_bytes(&registration.context_sha256)?;
    if closed_context(mode.registration_schema(), &registration)? != registration.context_sha256 {
        return Err(ReplayInputError::new(
            "registration_context",
            "closed registration context differs",
        ));
    }
    Ok(registration)
}

fn checked_meanings(
    supplied: &[MoveMeaning],
    moves: &[BoardMove],
    kind: MoveTokenKind,
    legal_tokens: Option<&[MoveMeaning]>,
) -> Result<(), ReplayInputError> {
    if supplied.len() != moves.len() {
        return Err(ReplayInputError::new(
            "ordered_meaning",
            "ordered stream extent differs",
        ));
    }
    let bits =
        cpu_task::pack_moves(moves).map_err(|e| ReplayInputError::cpu("move_encoding", e))?;
    for (at, ((token, movement), bits)) in supplied.iter().zip(moves).zip(bits).enumerate() {
        let legal_order_slot = legal_tokens
            .map(|legal| {
                legal
                    .iter()
                    .position(|candidate| candidate.move16 == bits)
                    .ok_or_else(|| {
                        ReplayInputError::new(
                            "ordered_meaning",
                            "token is not in actual full Rules legal order",
                        )
                    })
                    .and_then(|slot| {
                        u16::try_from(slot).map_err(|e| ReplayInputError::new("ordered_meaning", e))
                    })
            })
            .transpose()?;
        let promotion = match movement.promotion {
            None => None,
            Some(rz_position::PieceKind::Queen) => Some(semantic::PromotionMeaning::Queen),
            Some(rz_position::PieceKind::Rook) => Some(semantic::PromotionMeaning::Rook),
            Some(rz_position::PieceKind::Bishop) => Some(semantic::PromotionMeaning::Bishop),
            Some(rz_position::PieceKind::Knight) => Some(semantic::PromotionMeaning::Knight),
            Some(_) => {
                return Err(ReplayInputError::new(
                    "move_encoding",
                    "unsupported promotion",
                ));
            }
        };
        let wanted = MoveMeaning {
            token_kind: kind,
            slot: u16::try_from(at).map_err(|e| ReplayInputError::new("ordered_meaning", e))?,
            legal_order_slot,
            move16: bits,
            from: movement.from.index(),
            to: movement.to.index(),
            promotion,
            uci: movement.to_string(),
        };
        if token != &wanted {
            return Err(ReplayInputError::new(
                "ordered_meaning",
                "move/slot/side-relative order/promotion meaning differs",
            ));
        }
    }
    Ok(())
}

/// Consumer extents constrain retained Vec capacity as well as length. A
/// collect-created vector may have a minimum capacity larger than a short L.
/// Reserve exactly the required elements fallibly, then explicitly reject any
/// allocator-reported overshoot rather than handing it to the owner unchecked.
fn retained_moves(
    first: &[BoardMove],
    second: &[BoardMove],
    maximum: usize,
) -> Result<Vec<BoardMove>, ReplayInputError> {
    let count = first.len().checked_add(second.len()).ok_or_else(|| {
        ReplayInputError::new("retained_move_extent", "ordered move extent overflow")
    })?;
    if count > maximum {
        return Err(ReplayInputError::new(
            "retained_move_extent",
            "ordered move extent exceeds the actual consumer bound",
        ));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|error| ReplayInputError::new("retained_move_capacity", error))?;
    if result.capacity() > maximum {
        return Err(ReplayInputError::new(
            "retained_move_capacity",
            "allocator retained capacity exceeds the actual consumer bound",
        ));
    }
    result.extend_from_slice(first);
    result.extend_from_slice(second);
    Ok(result)
}

fn original_root_request(cpu: &cpu_task::Request, receipt: &SemanticReceipt) -> SemanticRequest {
    SemanticRequest {
        schema: semantic::SEMANTIC_SCHEMA.into(),
        question: receipt.question,
        parent_input_sha256: cpu.parent_input_sha256.clone(),
        before_result_anchor_sha256: receipt.before_result_anchor_sha256.clone(),
        position_command: cpu.position_command.clone(),
        expected_board_fen: cpu.expected_board_fen.clone(),
        rules_state_sha256: cpu.rules_state_sha256.clone(),
        rules_history_sha256: cpu.rules_history_sha256.clone(),
        prefix: cpu.prefix.clone(),
        root_moves: cpu.root_moves.clone(),
        claimed_line: receipt
            .claimed_line
            .movements
            .iter()
            .map(|token| token.move16)
            .collect(),
        current_binary_sha256: receipt.current_binary_sha256.clone(),
        max_wall_time_ms: receipt.resource_policy.max_wall_time_ms,
        max_output_bytes: receipt.resource_policy.max_output_bytes,
        context_sha256: receipt.context_sha256.clone(),
    }
}

fn prepare_rules(
    request: &ReplayInputRequest,
    cpu: &cpu_task::Request,
    receipt: &SemanticReceipt,
    clock: Clock,
    expected: &ReplayExpectedPins,
    expected_scope: Option<SemanticReceiptProducerScope>,
) -> Result<(Position, DefendResponseReplayPlan), ReplayInputError> {
    if receipt.schema != semantic::SEMANTIC_SCHEMA
        || receipt.implementation != semantic::SEMANTIC_VERSION
        || receipt.binary_pin_scope
            != expected_scope
                .unwrap_or(SemanticReceiptProducerScope::LibraryDispatcherArgument)
                .as_str()
        || receipt.caller_declaration_scope != semantic::DECLARATION_SCOPE
        || receipt.meaning_scope != semantic::MEANING_SCOPE
        || receipt.question != semantic::SemanticQuestion::RestrictedResponse
        || receipt.parent_input_sha256 != request.parent.parent_input_sha256
        || receipt.current_binary_sha256 != expected.semantic_binary_sha256
        || receipt.context_sha256 != request.binding.semantic_context_sha256
        || receipt.branch_meaning_sha256 != request.binding.semantic_branch_meaning_sha256
        || receipt.before_result_anchor_sha256
            != request.binding.semantic_before_result_anchor_sha256
        || receipt.cpu_checks != 0
        || receipt.resource_policy.cpu_checks != 0
        || receipt.search_executed
        || receipt.model_executed
        || receipt.training_target_created
        || receipt.product_verifier_enabled
        || receipt.deadline_exceeded
        || !(1..=cpu_task::MAX_WALL_TIME_MS).contains(&receipt.resource_policy.max_wall_time_ms)
        || !(1024..=MAX_SEMANTIC_RECEIPT_BYTES).contains(&receipt.resource_policy.max_output_bytes)
        || receipt.elapsed_ms > receipt.resource_policy.max_wall_time_ms
        || receipt.resource_policy.max_prefix_plies != semantic::MAX_PREFIX_PLIES
        || receipt.resource_policy.max_claim_plies != semantic::MAX_CLAIM_PLIES
        || receipt.resource_policy.max_root_moves != semantic::MAX_ROOT_MOVES
    {
        return Err(ReplayInputError::new(
            "semantic_binding",
            "original semantic declaration/domain/authority/resource binding differs",
        ));
    }
    // Root preparation uses the actual original CPU position command. Receipt
    // context and its original tool launch remain independently caller-checked;
    // this adapter does not recreate a semantic request/context to certify them.
    let semantic_request = original_root_request(cpu, receipt);
    if semantic_request.claimed_line.len() > semantic::MAX_CLAIM_PLIES {
        return Err(ReplayInputError::new(
            "claim_extent",
            "claim exceeds original semantic bound",
        ));
    }
    check_clock(clock)?;
    let owners = Arc::new(OwnerRegistry::default());
    let root = semantic::prepare_root(&semantic_request, &owners, clock.execution_deadline)
        .map_err(|e| ReplayInputError::semantic("root_rules", e))?;
    if root.history_completeness() != HistoryCompleteness::Complete
        || root.history_origin() != HistoryOrigin::StartPosition
        || root.known_history_len() == 0
        || root.known_history_len() > MAX_KNOWN_HISTORY_POSITIONS
    {
        return Err(ReplayInputError::new(
            "root_history",
            "bounded complete startpos history required; FEN prefixes are not invented",
        ));
    }
    let actual_root = semantic::describe(
        &root,
        &owners,
        MoveTokenKind::RootLegal,
        clock.execution_deadline,
    )
    .map_err(|e| ReplayInputError::semantic("root_descriptor", e))?;
    same_serialized(&actual_root, &receipt.root, "root_descriptor")?;
    if actual_root.rules_state_sha256 != cpu.rules_state_sha256
        || actual_root.rules_history_sha256 != cpu.rules_history_sha256
        || actual_root.board_fen != cpu.expected_board_fen
        || actual_root.play_status != "ongoing"
    {
        return Err(ReplayInputError::new(
            "root_identity",
            "original CPU root differs from actual complete Rules root",
        ));
    }
    let prefix = retained_moves(
        &cpu_task::decode_moves(&cpu.prefix).map_err(|e| ReplayInputError::cpu("prefix", e))?,
        &[],
        request.config.line_plies,
    )?;
    checked_meanings(&receipt.prefix, &prefix, MoveTokenKind::Prefix, None)?;
    let target = cpu_task::replay_checked(&root, &prefix, &owners, clock.execution_deadline)
        .map_err(|e| ReplayInputError::cpu("target_rules", e))?;
    let actual_target = semantic::describe(
        &target,
        &owners,
        MoveTokenKind::TargetLegal,
        clock.execution_deadline,
    )
    .map_err(|e| ReplayInputError::semantic("target_descriptor", e))?;
    same_serialized(&actual_target, &receipt.target, "target_descriptor")?;
    if actual_target.play_status != "ongoing" || root.side_to_move() == target.side_to_move() {
        return Err(ReplayInputError::new(
            "target_side",
            "defend_response needs an ongoing opponent target",
        ));
    }
    let roots = retained_moves(
        &cpu_task::decode_moves(&cpu.root_moves)
            .map_err(|e| ReplayInputError::cpu("restriction", e))?,
        &[],
        semantic::MAX_ROOT_MOVES,
    )?;
    let target_legal = retained_moves(
        &cpu_task::decode_moves(&actual_target.legal_moves)
            .map_err(|e| ReplayInputError::cpu("target_legal", e))?,
        &[],
        semantic::MAX_ROOT_MOVES,
    )?;
    if roots.is_empty()
        || roots.iter().enumerate().any(|(at, movement)| {
            roots[..at].contains(movement) || !target_legal.contains(movement)
        })
    {
        return Err(ReplayInputError::new(
            "restriction",
            "explicit unique actual target legal restriction required",
        ));
    }
    let restriction = receipt.root_restriction.as_ref().ok_or_else(|| {
        ReplayInputError::new("restriction", "original restriction descriptor missing")
    })?;
    checked_meanings(
        &restriction.declared_order,
        &roots,
        MoveTokenKind::RootRestriction,
        Some(&actual_target.legal_tokens),
    )?;
    let effective: Vec<_> = target_legal
        .iter()
        .copied()
        .filter(|movement| roots.contains(movement))
        .collect();
    checked_meanings(
        &restriction.effective_legal_order,
        &effective,
        MoveTokenKind::EffectiveRootLegal,
        Some(&actual_target.legal_tokens),
    )?;
    if restriction.scope
        != "exact_target_root_only;request_order_preserved;effective_order_is_rules_order"
        || restriction.declared_order_sha256
            != hash(
                &json!([semantic::MOVE_ORDER_DOMAIN, cpu.root_moves]),
                "restriction_identity",
            )?
        || restriction.effective_order_sha256
            != hash(
                &json!([
                    semantic::MOVE_ORDER_DOMAIN,
                    cpu_task::pack_moves(&effective)
                        .map_err(|e| ReplayInputError::cpu("effective_order", e))?
                ]),
                "restriction_identity",
            )?
    {
        return Err(ReplayInputError::new(
            "restriction_identity",
            "original/effective restriction order identity differs",
        ));
    }
    let claim = cpu_task::decode_moves(&semantic_request.claimed_line)
        .map_err(|e| ReplayInputError::cpu("claim", e))?;
    checked_meanings(
        &receipt.claimed_line.movements,
        &claim,
        MoveTokenKind::ClaimedContinuation,
        None,
    )?;
    if receipt.claimed_line.claim_truth != "unknown" {
        return Err(ReplayInputError::new(
            "claim_authority",
            "legal hypothesis remains unknown truth",
        ));
    }
    let full_claim = if claim.is_empty() {
        if receipt.claimed_line.status != semantic::ClaimStatus::NoClaim
            || receipt.claimed_line.legality_verified.is_some()
            || receipt.claimed_line.restriction_checked
            || receipt.claimed_line.final_state.is_some()
        {
            return Err(ReplayInputError::new(
                "claim",
                "empty continuation must retain original no-claim fields",
            ));
        }
        retained_moves(&[], &[], request.config.line_plies)?
    } else {
        if receipt.claimed_line.status != semantic::ClaimStatus::LegalContinuation
            || receipt.claimed_line.legality_verified != Some(true)
            || !receipt.claimed_line.restriction_checked
            || !roots.contains(&claim[0])
        {
            return Err(ReplayInputError::new(
                "claim",
                "original legal continuation escapes its target restriction",
            ));
        }
        let end = cpu_task::replay_checked(&target, &claim, &owners, clock.execution_deadline)
            .map_err(|e| ReplayInputError::cpu("claim_rules", e))?;
        let actual_end = semantic::describe(
            &end,
            &owners,
            MoveTokenKind::ClaimEndLegal,
            clock.execution_deadline,
        )
        .map_err(|e| ReplayInputError::semantic("claim_descriptor", e))?;
        same_serialized(
            &actual_end,
            receipt.claimed_line.final_state.as_ref().ok_or_else(|| {
                ReplayInputError::new(
                    "claim_descriptor",
                    "original final Rules descriptor missing",
                )
            })?,
            "claim_descriptor",
        )?;
        // Semantic continuation is target-relative, owner hypothesis root-relative.
        // Prepend prefix exactly once, after its independent descriptor comparison.
        let count = prefix
            .len()
            .checked_add(claim.len())
            .ok_or_else(|| ReplayInputError::new("claim_extent", "prefix+claim overflow"))?;
        if count > request.config.line_plies {
            return Err(ReplayInputError::new(
                "claim_extent",
                "root-relative claim exceeds declared owner L",
            ));
        }
        retained_moves(&prefix, &claim, request.config.line_plies)?
    };
    let parent = &request.parent;
    let binding = &request.binding;
    check_clock(clock)?;
    // expected snapshots are created only after external original descriptors
    // and actual complete Rules state/history/order/claim were compared above.
    let plan = DefendResponseReplayPlan {
        historical: ReplayHistoricalPins {
            parent_input_sha256: digest_bytes(&parent.parent_input_sha256)?,
            current_view_sha256: digest_bytes(&parent.current_view_sha256)?,
            frozen_admission_sha256: digest_bytes(&parent.frozen_admission_sha256)?,
            query_sha256: digest_bytes(&binding.query_sha256)?,
            catalogue_sha256: digest_bytes(&binding.catalogue_artifact.sha256)?,
            before_result_sha256: digest_bytes(&binding.before_result_artifact.sha256)?,
            semantic_input_sha256: digest_bytes(&binding.semantic_input_sha256)?,
            cpu_request_sha256: digest_bytes(&binding.cpu_request_artifact.sha256)?,
        },
        expected_root: root.snapshot(),
        expected_target: target.snapshot(),
        root_legal_order: retained_moves(
            &cpu_task::decode_moves(&actual_root.legal_moves)
                .map_err(|e| ReplayInputError::cpu("root_legal", e))?,
            &[],
            semantic::MAX_ROOT_MOVES,
        )?,
        target_legal_order: target_legal,
        prefix,
        response_restriction: roots,
        claimed_line: full_claim,
        baseline_depth: cpu.baseline_depth,
        requested_depth: cpu.requested_depth,
        nodes_per_check: cpu.max_nodes_per_check,
        cpu: CpuConfig {
            profile: CpuProfile::PlanAssisted,
            tt_entries: cpu.tt_entries,
            max_depth: cpu.requested_depth,
            quiescence_ply: cpu.quiescence_ply,
        },
    };
    Ok((root, plan))
}

fn validate(
    request: ReplayInputRequest,
    expected: &ReplayExpectedPins,
    started: Instant,
    clock: Clock,
    input_artifact: ArtifactPin,
    expected_scope: Option<SemanticReceiptProducerScope>,
) -> Result<CheckedReplayInputs, ReplayInputError> {
    if request.schema != request.mode.input_schema()
        || request.authorities != ReplayAuthorities::default()
    {
        return Err(ReplayInputError::new(
            "domain_authority",
            "closed replay domain and negative authority declarations required",
        ));
    }
    if request.parent != expected.parent || request.binding != expected.binding {
        return Err(ReplayInputError::new(
            "caller_binding",
            "independent parent/encoding/query/catalogue/before/prior/semantic/request pins differ",
        ));
    }
    for sha in [
        &request.parent.parent_input_sha256,
        &request.parent.current_view_sha256,
        &request.parent.frozen_admission_sha256,
        &request.parent.encoding_sha256,
        &request.binding.query_sha256,
        &request.binding.prior_ledger_sha256,
        &request.binding.semantic_input_sha256,
        &request.binding.semantic_context_sha256,
        &request.binding.semantic_branch_meaning_sha256,
        &request.binding.semantic_before_result_anchor_sha256,
        &expected.semantic_binary_sha256,
        &request.context_sha256,
    ] {
        digest_bytes(sha)?;
    }
    for artifact in [
        &request.binding.catalogue_artifact,
        &request.binding.before_result_artifact,
    ] {
        bounded_pin(artifact, MAX_OUTPUT_BYTES as u64)?;
    }
    bounded_pin(
        &request.binding.cpu_request_artifact,
        cpu_task::MAX_REQUEST_BYTES as u64,
    )?;
    original_pin(
        &request.registration_raw,
        &request.registration_artifact,
        &expected.registration_artifact,
        MAX_REGISTRATION_BYTES,
    )?;
    original_pin(
        &request.prepared_action_raw,
        &request.prepared_action_artifact,
        &expected.prepared_action_artifact,
        super::MAX_REQUEST_BYTES,
    )?;
    original_pin(
        &request.semantic_receipt_raw,
        &request.semantic_receipt_artifact,
        &expected.semantic_receipt_artifact,
        MAX_SEMANTIC_RECEIPT_BYTES,
    )?;
    if closed_context(request.mode.input_schema(), &request)? != request.context_sha256 {
        return Err(ReplayInputError::new(
            "context_identity",
            "new replay mode/resources/original raw context differs",
        ));
    }
    check_clock(clock)?;
    let registration = check_registration(&request.registration_raw, expected, request.mode)?;
    let (prepared, _) = super::decode(request.prepared_action_raw.as_bytes(), started)
        .map_err(|e| ReplayInputError::strategic("prepared_action", e))?;
    let binding = &request.binding;
    if prepared.query_sha256 != binding.query_sha256
        || prepared.catalogue_artifact != binding.catalogue_artifact
        || prepared.before_result_artifact != binding.before_result_artifact
        || prepared.prior_ledger_sha256 != binding.prior_ledger_sha256
        || prepared.action.semantic_input_sha256 != binding.semantic_input_sha256
        || prepared.cpu_request_artifact != binding.cpu_request_artifact
        || prepared.action.task != StrategicTaskKind::DefendResponse
    {
        return Err(ReplayInputError::new(
            "prepared_binding",
            "original selected action and new caller binding differ or unsupported task",
        ));
    }
    let cpu = cpu_task::decode_request(prepared.cpu_request_raw.as_bytes())
        .map_err(|e| ReplayInputError::cpu("original_cpu_request", e))?;
    cpu_task::validate_pins(&cpu, &registration.artifacts.legacy_cpu_binary.sha256)
        .map_err(|e| ReplayInputError::cpu("legacy_cpu_binary_profile", e))?;
    if cpu.task != cpu_task::TaskKind::DefendResponse
        || cpu.cpu_profile_sha256 != registration.legacy_cpu_profile_sha256
        || cpu.parent_input_sha256 != request.parent.parent_input_sha256
        || cpu.max_nodes_per_check > 10_000_000
        || cpu.prefix.len() >= request.config.line_plies
        || cpu.max_wall_time_ms != request.resources.whole_wall_ms
    {
        return Err(ReplayInputError::new(
            "fresh_lane",
            "unsupported fresh PlanAssisted H0/H1/N/prefix/original wall binding",
        ));
    }
    let config = request.config.config();
    config
        .validate()
        .map_err(|e| ReplayInputError::new("owner_config", e))?;
    if config.cpu_nodes_per_task != cpu.max_nodes_per_check {
        return Err(ReplayInputError::new(
            "owner_config",
            "config does not retain exact original N",
        ));
    }
    let required = replay_mode_requirements(
        request.mode,
        config.line_plies,
        cpu.prefix.len(),
        cpu.max_nodes_per_check,
    )
    .map_err(|e| ReplayInputError::new("resource_arithmetic", e))?;
    let exact_cpu = cpu
        .max_nodes_per_check
        .checked_mul(request.mode.checks())
        .ok_or_else(|| {
            ReplayInputError::new("resource_arithmetic", "mode CPU reservation overflow")
        })?;
    let minimum_output = cpu
        .max_output_bytes
        .checked_mul(
            usize::try_from(request.mode.checks())
                .map_err(|e| ReplayInputError::new("resource_arithmetic", e))?,
        )
        .ok_or_else(|| {
            ReplayInputError::new("resource_arithmetic", "mode output reservation overflow")
        })?;
    let resources = &request.resources;
    if resources.cpu_nodes != exact_cpu
        || resources.cpu_nodes != required.cpu_nodes
        || resources.cpu_nodes > prepared.remaining.nodes
        || resources.whole_wall_ms > prepared.remaining.wall_ms
        || resources.output_bytes as u64 > prepared.remaining.output_bytes
        || resources.output_bytes < minimum_output
        || resources.role_calls != config.max_role_calls
        || resources.role_calls < required.role_calls
        || resources.store_nodes != config.max_nodes
        || resources.store_nodes < required.nodes
        || resources.store_records != config.max_records
        || resources.store_records < required.records
        || resources.required_observations != required.observations
        || resources.required_line_chunks != required.line_chunks
        || resources.required_stages != required.stages
        || !(1..=65_536).contains(&resources.max_rounds)
    {
        return Err(ReplayInputError::new(
            "resource_reservation",
            "mode fixed CPU/role/store/output/original remaining reservation does not fit",
        ));
    }
    check_clock(clock)?;
    let receipt: SemanticReceipt = serde_json::from_str(&request.semantic_receipt_raw)
        .map_err(|e| ReplayInputError::new("semantic_receipt_json", e))?;
    // Option fields in the legacy DTO must still be present in the original
    // closed receipt. Equality checks shape only; original pin remains raw-byte.
    let raw_receipt: Value = serde_json::from_str(&request.semantic_receipt_raw)
        .map_err(|e| ReplayInputError::new("semantic_receipt_json", e))?;
    if serde_json::to_value(&receipt)
        .map_err(|e| ReplayInputError::new("semantic_receipt_shape", e))?
        != raw_receipt
    {
        return Err(ReplayInputError::new(
            "semantic_receipt_shape",
            "complete original receipt fields required, including explicit nullable fields",
        ));
    }
    let (root, plan) = prepare_rules(&request, &cpu, &receipt, clock, expected, expected_scope)?;
    check_clock(clock)?;
    let limits = PalsLimits {
        deadline: clock.execution_deadline,
        max_rounds: resources.max_rounds,
        max_cpu_nodes: exact_cpu,
        cpu_depth: cpu.requested_depth,
    };
    Ok(CheckedReplayInputs {
        request,
        registration,
        expected_semantic_receipt_producer_scope: expected_scope,
        cpu_request_raw: prepared.cpu_request_raw,
        input_artifact,
        config,
        root,
        plan,
        limits,
        clock,
        started,
    })
}

/// Whole wall is admitted before output, identity or registration rejection.
/// Invalid output retains that clock and a finite diagnostic cap. Malformed
/// outer JSON cannot invent a wall. No I/O, search, CPU/model/provider creation.
pub fn check_replay_inputs(
    bytes: &[u8],
    expected: &ReplayExpectedPins,
    started: Instant,
) -> Result<CheckedReplayInputs, ReplayInputError> {
    check_replay_inputs_inner(bytes, expected, None, started)
}

/// Explicit caller registration is required; the scope is not selected from the
/// receipt, host OS or the current replay executable. All raw bindings and Rules
/// checks remain the same as the legacy library-only entry point.
pub fn check_replay_inputs_with_semantic_scope(
    bytes: &[u8],
    expected: &ReplayExpectedPins,
    expected_scope: SemanticReceiptProducerScope,
    started: Instant,
) -> Result<CheckedReplayInputs, ReplayInputError> {
    check_replay_inputs_inner(bytes, expected, Some(expected_scope), started)
}

fn check_replay_inputs_inner(
    bytes: &[u8],
    expected: &ReplayExpectedPins,
    expected_scope: Option<SemanticReceiptProducerScope>,
    started: Instant,
) -> Result<CheckedReplayInputs, ReplayInputError> {
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        return Err(
            ReplayInputError::new("input_extent", "new replay wire must contain 1..=2MiB").stamped(
                started,
                None,
                expected_scope,
            ),
        );
    }
    let request: ReplayInputRequest = serde_json::from_slice(bytes).map_err(|e| {
        ReplayInputError::new("input_json", e).stamped(started, None, expected_scope)
    })?;
    let mut admitted_clock = None;
    let clock = admit_resource_clock(
        &request.resources,
        started,
        expected_scope,
        &mut admitted_clock,
    )?;
    validate(
        request,
        expected,
        started,
        clock,
        super::pin(bytes),
        expected_scope,
    )
    .map_err(|e| e.stamped(started, Some(clock), expected_scope))
}

/// Keeps the actual partial clock at admission failures without reconstructing
/// it from error strings. Legacy JSON parsing still happens before this helper.
fn admit_resource_clock(
    resources: &ReplayResourceDeclaration,
    started: Instant,
    expected_scope: Option<SemanticReceiptProducerScope>,
    admitted: &mut Option<Clock>,
) -> Result<Clock, ReplayInputError> {
    *admitted = None;
    if !(1..=cpu_task::MAX_WALL_TIME_MS).contains(&resources.whole_wall_ms) {
        return Err(
            ReplayInputError::new("whole_wall", "finite original whole wall required").stamped(
                started,
                None,
                expected_scope,
            ),
        );
    }
    let whole_deadline = started
        .checked_add(Duration::from_millis(resources.whole_wall_ms))
        .ok_or_else(|| {
            ReplayInputError::new("whole_wall", "original whole deadline overflow").stamped(
                started,
                None,
                expected_scope,
            )
        })?;
    let output_valid = (1024..=MAX_OUTPUT_BYTES).contains(&resources.output_bytes);
    let mut clock = Clock {
        whole_deadline,
        execution_deadline: whole_deadline,
        whole_wall_ms: resources.whole_wall_ms,
        cleanup_reserve_ms: resources.cleanup_reserve_ms,
        output_limit: if output_valid {
            resources.output_bytes
        } else {
            DIAGNOSTIC_OUTPUT_BYTES
        },
    };
    *admitted = Some(clock);
    if !output_valid {
        return Err(ReplayInputError::new(
            "output_bound",
            "invalid output; original whole wall retained with diagnostic cap",
        )
        .stamped(started, Some(clock), expected_scope));
    }
    if resources.cleanup_reserve_ms == 0 || resources.cleanup_reserve_ms >= resources.whole_wall_ms
    {
        return Err(ReplayInputError::new(
            "cleanup_reserve",
            "finite positive cleanup reserve must be inside original whole wall",
        )
        .stamped(started, Some(clock), expected_scope));
    }
    clock.execution_deadline = started
        .checked_add(Duration::from_millis(
            resources.whole_wall_ms - resources.cleanup_reserve_ms,
        ))
        .ok_or_else(|| {
            ReplayInputError::new("cleanup_reserve", "original work deadline overflow").stamped(
                started,
                Some(clock),
                expected_scope,
            )
        })?;
    *admitted = Some(clock);
    if started > Instant::now() {
        return Err(ReplayInputError::new(
            "original_clock",
            "caller start cannot be in the future",
        )
        .stamped(started, Some(clock), expected_scope));
    }
    Ok(clock)
}

/// Borrowed shape of the existing closed outer wire. Opaque raw strings are
/// serialized verbatim as JSON string values, never parsed and re-rendered here.
#[derive(Serialize)]
struct ReplayPreparationView<'a> {
    schema: &'static str,
    mode: ReplayInputMode,
    registration_raw: &'a str,
    registration_artifact: &'a ArtifactPin,
    prepared_action_raw: &'a str,
    prepared_action_artifact: &'a ArtifactPin,
    semantic_receipt_raw: &'a str,
    semantic_receipt_artifact: &'a ArtifactPin,
    parent: &'a ReplayParentPins,
    binding: &'a ReplayBindingPins,
    config: &'a ReplayConfigInput,
    resources: &'a ReplayResourceDeclaration,
    authorities: ReplayAuthorities,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_sha256: Option<&'a str>,
}

struct ConstructionWriter<'a> {
    backing: Option<&'a mut Vec<u8>>,
    bytes: usize,
    limit: usize,
    clock: Clock,
    failure: Option<ReplayConstructionWriterFailure>,
}
impl ConstructionWriter<'_> {
    fn reject(&mut self, reason: ReplayConstructionWriterFailure) -> io::Error {
        self.failure = Some(reason);
        io::Error::other("bounded replay construction writer refused write")
    }
}
impl Write for ConstructionWriter<'_> {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if Instant::now() >= self.clock.execution_deadline {
            return Err(self.reject(ReplayConstructionWriterFailure::Deadline));
        }
        let Some(next) = self.bytes.checked_add(input.len()) else {
            return Err(self.reject(ReplayConstructionWriterFailure::LengthOverflow));
        };
        if next > self.limit {
            return Err(self.reject(ReplayConstructionWriterFailure::ByteLimit));
        }
        if self
            .backing
            .as_ref()
            .is_some_and(|bytes| next > bytes.capacity())
        {
            return Err(self.reject(ReplayConstructionWriterFailure::RetainedCapacity));
        }
        if let Some(bytes) = self.backing.as_mut() {
            // Capacity was reserved and inspected before serialization. This
            // extend cannot request a new backing allocation.
            bytes.extend_from_slice(input);
        }
        self.bytes = next;
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if Instant::now() >= self.clock.execution_deadline {
            Err(self.reject(ReplayConstructionWriterFailure::Deadline))
        } else {
            Ok(())
        }
    }
}

fn preparation_input_cause(
    error: ReplayInputError,
    started: Instant,
    clock: Clock,
    scope: SemanticReceiptProducerScope,
) -> ReplayPreparationCause {
    ReplayPreparationCause::Input(Box::new(error.stamped(started, Some(clock), Some(scope))))
}
fn preparation_clock(
    started: Instant,
    clock: Clock,
    scope: SemanticReceiptProducerScope,
) -> Result<(), ReplayPreparationCause> {
    check_clock(clock).map_err(|error| preparation_input_cause(error, started, clock, scope))
}
fn preparation_error(
    stage: &'static str,
    cause: ReplayPreparationCause,
    started: Instant,
    clock: Option<Clock>,
    scope: SemanticReceiptProducerScope,
    rejected_outer: Option<RejectedReplayBytes>,
) -> ReplayPreparationError {
    let mut error = ReplayPreparationError {
        stage,
        cause: Box::new(cause),
        diagnostics: Box::new(ReplayPreparationDiagnostics {
            expected_semantic_receipt_producer_scope: scope,
            elapsed_ms: 0,
            deadline_exceeded: false,
            execution_deadline_exceeded: false,
            original_whole_deadline_retained: clock.is_some(),
            whole_wall_ms: clock.map(|c| c.whole_wall_ms),
            cleanup_reserve_ms: clock.map(|c| c.cleanup_reserve_ms),
            output_limit: clock.map_or(DIAGNOSTIC_OUTPUT_BYTES, |c| c.output_limit),
            original_started: started,
            clock,
        }),
        rejected_outer: rejected_outer.map(Box::new),
    };
    let now = Instant::now();
    error.diagnostics.elapsed_ms = cpu_task::milliseconds(now.saturating_duration_since(started));
    error.diagnostics.deadline_exceeded = clock.is_some_and(|c| now >= c.whole_deadline);
    error.diagnostics.execution_deadline_exceeded =
        clock.is_some_and(|c| now >= c.execution_deadline);
    error
}
fn count_preparation<T: Serialize>(
    value: &T,
    limit: usize,
    clock: Clock,
    started: Instant,
    scope: SemanticReceiptProducerScope,
) -> Result<usize, ReplayPreparationCause> {
    preparation_clock(started, clock, scope)?;
    let mut writer = ConstructionWriter {
        backing: None,
        bytes: 0,
        limit,
        clock,
        failure: None,
    };
    serde_json::to_writer(&mut writer, value).map_err(|source| {
        ReplayPreparationCause::Serialization {
            source,
            writer: writer.failure,
        }
    })?;
    preparation_clock(started, clock, scope)?;
    Ok(writer.bytes)
}

fn preparation_expected_shape(expected: &ReplayExpectedPins) -> Result<(), ReplayInputError> {
    // No cloning or context Value construction until every variable-size
    // expected field is bounded. Numeric declarations have fixed wire topology.
    for sha in [
        &expected.parent.parent_input_sha256,
        &expected.parent.current_view_sha256,
        &expected.parent.frozen_admission_sha256,
        &expected.parent.encoding_sha256,
        &expected.binding.query_sha256,
        &expected.binding.prior_ledger_sha256,
        &expected.binding.semantic_input_sha256,
        &expected.binding.semantic_context_sha256,
        &expected.binding.semantic_branch_meaning_sha256,
        &expected.binding.semantic_before_result_anchor_sha256,
        &expected.legacy_cpu_profile_sha256,
        &expected.semantic_binary_sha256,
    ] {
        digest_bytes(sha)?;
    }
    if expected.provider_factory_id.is_empty()
        || expected.provider_factory_id.len() > 128
        || !expected
            .provider_factory_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ReplayInputError::new(
            "registration_identity",
            "bounded closed factory ID required",
        ));
    }
    for (pin, maximum) in [
        (
            &expected.registration_artifact,
            MAX_REGISTRATION_BYTES as u64,
        ),
        (
            &expected.prepared_action_artifact,
            super::MAX_REQUEST_BYTES as u64,
        ),
        (
            &expected.semantic_receipt_artifact,
            MAX_SEMANTIC_RECEIPT_BYTES as u64,
        ),
        (
            &expected.binding.catalogue_artifact,
            MAX_OUTPUT_BYTES as u64,
        ),
        (
            &expected.binding.before_result_artifact,
            MAX_OUTPUT_BYTES as u64,
        ),
        (
            &expected.binding.cpu_request_artifact,
            cpu_task::MAX_REQUEST_BYTES as u64,
        ),
        (
            &expected.registered_artifacts.legacy_cpu_binary,
            cpu_task::MAX_SELF_BINARY_BYTES,
        ),
        (
            &expected.registered_artifacts.replay_binary,
            cpu_task::MAX_SELF_BINARY_BYTES,
        ),
        (
            &expected.registered_artifacts.engine_source,
            MAX_SOURCE_BYTES,
        ),
        (
            &expected.registered_artifacts.wrapper_source,
            MAX_SOURCE_BYTES,
        ),
        (
            &expected.registered_artifacts.replay_source,
            MAX_SOURCE_BYTES,
        ),
        (
            &expected.registered_artifacts.provider_factory_source,
            MAX_SOURCE_BYTES,
        ),
    ] {
        // checked_pin checks SHA shape before constructing an error projection.
        bounded_pin(pin, maximum)?;
    }
    if let Some(pin) = &expected.registered_artifacts.opponent_recheck_source {
        bounded_pin(pin, MAX_SOURCE_BYTES)?;
    }
    Ok(())
}
fn preparation_original<'a>(
    raw: &'a [u8],
    expected: &ArtifactPin,
    maximum: usize,
    original: ReplayOriginalKind,
) -> Result<&'a str, ReplayPreparationCause> {
    if raw.is_empty() || raw.len() > maximum || expected.bytes != raw.len() as u64 {
        return Err(ReplayPreparationCause::Refusal {
            reason: "original UTF-8 byte extent differs from independent pin or bound",
            actual: Some(raw.len()),
            limit: Some(maximum),
        });
    }
    let text = std::str::from_utf8(raw)
        .map_err(|source| ReplayPreparationCause::Utf8 { original, source })?;
    if super::pin(raw) != *expected {
        return Err(ReplayPreparationCause::Refusal {
            reason: "original bytes differ from independently supplied SHA256",
            actual: None,
            limit: None,
        });
    }
    Ok(text)
}

/// Constructs only the existing outer input wire and its closed context. The
/// three originals stay borrowed and byte-exact; a successful result has passed
/// the existing scoped checker, which is then dropped. No engines are created.
///
/// The prepaid policy is 3*Sraw + 2*C + Q + 64KiB. Unchanged closed_context keeps
/// its original body Value while json!'s array expression converts &body into a
/// second Value with owned raw strings; the existing canonical helper then
/// recursively clones a third, sorted Value. These three raw payload copies can
/// coexist. The remaining allowance covers conservative canonical Vec growth,
/// exact final Vec and fixed metadata topology. It is not an allocator/validator/
/// RSS peak guarantee. The final checker is a separate allocation boundary. No
/// caller certification bool, larger wall or expanded remaining is accepted.
pub fn prepare_replay_request(
    originals: ReplayOriginals<'_>,
    declaration: ReplayPreparationDeclaration<'_>,
    expected: &ReplayExpectedPins,
    expected_scope: SemanticReceiptProducerScope,
    budget: ReplayConstructionBudget,
    started: Instant,
) -> Result<PreparedReplayRequest, ReplayPreparationError> {
    let mut observed_clock = None;
    let clock = admit_resource_clock(
        declaration.resources,
        started,
        Some(expected_scope),
        &mut observed_clock,
    )
    .map_err(|error| {
        preparation_error(
            "resource_clock",
            ReplayPreparationCause::Input(Box::new(error)),
            started,
            observed_clock,
            expected_scope,
            None,
        )
    })?;
    let mut stage = "preflight";
    let mut bytes = Vec::new();
    let mut artifact = None;
    let mut serialization_complete = false;
    let prepared =
        (|| -> Result<(ReplayPreparationAudit, Instant, Instant), ReplayPreparationCause> {
            if !(1..=MAX_REQUEST_BYTES).contains(&budget.max_outer_bytes)
                || !(1..=MAX_CONSTRUCTION_CREDIT_BYTES)
                    .contains(&budget.max_construction_credit_bytes)
            {
                return Err(ReplayPreparationCause::Refusal {
                    reason: "construction outer/credit bounds are outside supported finite limits",
                    actual: None,
                    limit: None,
                });
            }
            preparation_expected_shape(expected)
                .map_err(|error| preparation_input_cause(error, started, clock, expected_scope))?;
            let registration_raw = preparation_original(
                originals.registration,
                &expected.registration_artifact,
                MAX_REGISTRATION_BYTES,
                ReplayOriginalKind::Registration,
            )?;
            let prepared_action_raw = preparation_original(
                originals.prepared_action,
                &expected.prepared_action_artifact,
                super::MAX_REQUEST_BYTES,
                ReplayOriginalKind::PreparedAction,
            )?;
            let semantic_receipt_raw = preparation_original(
                originals.semantic_receipt,
                &expected.semantic_receipt_artifact,
                MAX_SEMANTIC_RECEIPT_BYTES,
                ReplayOriginalKind::SemanticReceipt,
            )?;
            let raw_utf8_bytes = originals
                .registration
                .len()
                .checked_add(originals.prepared_action.len())
                .and_then(|sum| sum.checked_add(originals.semantic_receipt.len()))
                .ok_or(ReplayPreparationCause::Refusal {
                    reason: "original byte sum overflow",
                    actual: None,
                    limit: None,
                })?;
            let mut view = ReplayPreparationView {
                schema: declaration.mode.input_schema(),
                mode: declaration.mode,
                registration_raw,
                registration_artifact: &expected.registration_artifact,
                prepared_action_raw,
                prepared_action_artifact: &expected.prepared_action_artifact,
                semantic_receipt_raw,
                semantic_receipt_artifact: &expected.semantic_receipt_artifact,
                parent: &expected.parent,
                binding: &expected.binding,
                config: declaration.config,
                resources: declaration.resources,
                authorities: ReplayAuthorities::default(),
                context_sha256: None,
            };
            stage = "count";
            // Context has the existing compact [SCHEMA, body] canonical meaning.
            // Sorting changes key order, not compact byte length or string escaping.
            let context_json_bytes = count_preparation(
                &(declaration.mode.input_schema(), &view),
                MAX_REQUEST_BYTES,
                clock,
                started,
                expected_scope,
            )?;
            const PLACEHOLDER: &str =
                "0000000000000000000000000000000000000000000000000000000000000000";
            view.context_sha256 = Some(PLACEHOLDER);
            let outer_json_bytes = count_preparation(
                &view,
                budget.max_outer_bytes,
                clock,
                started,
                expected_scope,
            )?;
            let policy_bytes = raw_utf8_bytes
                .checked_mul(CONSTRUCTION_RAW_PAYLOAD_COPIES)
                .and_then(|amount| {
                    context_json_bytes
                        .checked_mul(2)
                        .and_then(|context| amount.checked_add(context))
                })
                .and_then(|amount| amount.checked_add(outer_json_bytes))
                .and_then(|amount| amount.checked_add(CONSTRUCTION_TOPOLOGY_BYTES))
                .ok_or(ReplayPreparationCause::Refusal {
                    reason: "construction credit arithmetic overflow",
                    actual: None,
                    limit: None,
                })?;
            if policy_bytes > budget.max_construction_credit_bytes {
                return Err(ReplayPreparationCause::Refusal {
                    reason: "known construction backing policy exceeds declared credit",
                    actual: Some(policy_bytes),
                    limit: Some(budget.max_construction_credit_bytes),
                });
            }
            stage = "final_backing";
            preparation_clock(started, clock, expected_scope)?;
            bytes
                .try_reserve_exact(outer_json_bytes)
                .map_err(ReplayPreparationCause::Reserve)?;
            if bytes.capacity() != outer_json_bytes {
                return Err(ReplayPreparationCause::Refusal {
                    reason: "allocator retained capacity differs from prepaid exact final backing",
                    actual: Some(bytes.capacity()),
                    limit: Some(outer_json_bytes),
                });
            }
            stage = "context";
            preparation_clock(started, clock, expected_scope)?;
            let context = closed_context(declaration.mode.input_schema(), &view)
                .map_err(|error| preparation_input_cause(error, started, clock, expected_scope))?;
            preparation_clock(started, clock, expected_scope)?;
            view.context_sha256 = Some(&context);
            stage = "serialize";
            let written = {
                let mut writer = ConstructionWriter {
                    backing: Some(&mut bytes),
                    bytes: 0,
                    limit: outer_json_bytes,
                    clock,
                    failure: None,
                };
                serde_json::to_writer(&mut writer, &view).map_err(|source| {
                    ReplayPreparationCause::Serialization {
                        source,
                        writer: writer.failure,
                    }
                })?;
                writer.bytes
            };
            serialization_complete = true;
            if written != outer_json_bytes {
                return Err(ReplayPreparationCause::Refusal {
                    reason: "actual compact outer differs from prepaid byte count",
                    actual: Some(written),
                    limit: Some(outer_json_bytes),
                });
            }
            preparation_clock(started, clock, expected_scope)?;
            artifact = Some(super::pin(&bytes));
            preparation_clock(started, clock, expected_scope)?;
            stage = "scoped_input_check";
            // Move the actual typed checker cause unchanged on error. Its source,
            // expected scope and original clock are not reconstructed or replaced.
            let checked =
                check_replay_inputs_with_semantic_scope(&bytes, expected, expected_scope, started)
                    .map_err(|error| ReplayPreparationCause::Input(Box::new(error)))?;
            if checked.original_registration() != originals.registration
                || checked.original_prepared_action() != originals.prepared_action
                || checked.original_semantic_receipt() != originals.semantic_receipt
                || checked.deadline() != clock.whole_deadline
                || checked.execution_deadline() != clock.execution_deadline
            {
                return Err(ReplayPreparationCause::Refusal {
                    reason: "scoped checker did not retain original bytes and clock",
                    actual: None,
                    limit: None,
                });
            }
            let input_admission = checked.audit();
            let whole_deadline = checked.deadline();
            let execution_deadline = checked.execution_deadline();
            drop(checked);
            stage = "prepared_publication";
            preparation_clock(started, clock, expected_scope)?;
            Ok((
                ReplayPreparationAudit {
                    scope: "borrowed_original_bytes_to_existing_scoped_input_check_only",
                    construction: ReplayConstructionCredit {
                        raw_utf8_bytes,
                        raw_payload_copies: CONSTRUCTION_RAW_PAYLOAD_COPIES,
                        context_json_bytes,
                        outer_json_bytes,
                        fixed_topology_bytes: CONSTRUCTION_TOPOLOGY_BYTES,
                        policy_bytes,
                        final_retained_capacity_bytes: bytes.capacity(),
                        allocator_peak_observed: false,
                        validator_peak_observed: false,
                        rss_peak_observed: false,
                    },
                    input_admission,
                    elapsed_ms: cpu_task::milliseconds(started.elapsed()),
                },
                whole_deadline,
                execution_deadline,
            ))
        })();
    match prepared {
        Ok((mut audit, whole_deadline, execution_deadline)) => {
            // The checker observed the same artifact. Move that small pin; no
            // rehash, raw clone or fresh deadline is needed during publication.
            let Some(pin) = artifact else {
                return Err(preparation_error(
                    "prepared_publication",
                    ReplayPreparationCause::Refusal {
                        reason: "completed scoped input lacks its observed outer artifact",
                        actual: None,
                        limit: None,
                    },
                    started,
                    Some(clock),
                    expected_scope,
                    Some(RejectedReplayBytes {
                        bytes,
                        artifact: None,
                        serialization_complete,
                    }),
                ));
            };
            if let Err(cause) = preparation_clock(started, clock, expected_scope) {
                return Err(preparation_error(
                    "prepared_publication",
                    cause,
                    started,
                    Some(clock),
                    expected_scope,
                    Some(RejectedReplayBytes {
                        bytes,
                        artifact: Some(pin),
                        serialization_complete,
                    }),
                ));
            }
            audit.elapsed_ms = cpu_task::milliseconds(started.elapsed());
            Ok(PreparedReplayRequest {
                bytes,
                artifact: pin,
                audit,
                whole_deadline,
                execution_deadline,
            })
        }
        Err(cause) => {
            let rejected = (bytes.capacity() != 0).then_some(RejectedReplayBytes {
                bytes,
                artifact,
                serialization_complete,
            });
            Err(preparation_error(
                stage,
                cause,
                started,
                Some(clock),
                expected_scope,
                rejected,
            ))
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::engine::RulesUciPort;
    use crate::{Command, ParserLimits, PositionPort, parse};
    use rz_position::PositionLimits;

    const OLD_BINARY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const NEW_BINARY: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SEMANTIC_BINARY: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const PRODUCER_SCOPES: [SemanticReceiptProducerScope; 3] = [
        SemanticReceiptProducerScope::LibraryDispatcherArgument,
        SemanticReceiptProducerScope::LinuxLoadedExecutableInode,
        SemanticReceiptProducerScope::CurrentExePathHash,
    ];

    struct Fixture {
        request: ReplayInputRequest,
        expected: ReplayExpectedPins,
    }
    fn bits(text: &str) -> u16 {
        cpu_task::pack_moves(&[BoardMove::from_uci(text).unwrap()]).unwrap()[0]
    }
    fn artifact(sha: &str, bytes: u64) -> ArtifactPin {
        ArtifactPin {
            bytes,
            sha256: sha.into(),
        }
    }
    fn seal<T: Serialize>(schema: &str, value: &T) -> String {
        closed_context(schema, value).unwrap()
    }
    fn seal_cpu(cpu: &mut cpu_task::Request) {
        cpu.cpu_profile_sha256 =
            cpu_task::json_digest(&cpu_task::profile(cpu, CpuProfile::PlanAssisted)).unwrap();
        cpu.recheck_profile_sha256 =
            cpu_task::json_digest(&cpu_task::profile(cpu, CpuProfile::Independent)).unwrap();
        cpu.branch_sha256 = cpu_task::json_digest(&json!([cpu_task::BRANCH_DOMAIN, {
            "parent_input_sha256": cpu.parent_input_sha256, "prefix": cpu.prefix, "root_moves": cpu.root_moves,
        }])).unwrap();
        cpu.context_sha256 = seal(cpu_task::CPU_TASK_SCHEMA, cpu);
    }
    fn fixture(mode: ReplayInputMode, claimed: bool, command: &str) -> Fixture {
        let Command::Position(spec) = parse(command, ParserLimits::default()).unwrap() else {
            panic!("position fixture");
        };
        let owners = Arc::new(OwnerRegistry::default());
        let prepared_root = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
            .prepare(&spec)
            .unwrap();
        let root = prepared_root.snapshot.rules_position();
        let prefix = vec![bits("e2e4")];
        let restriction = vec![bits("e7e5"), bits("c7c5")];
        let mut semantic_request = SemanticRequest {
            schema: semantic::SEMANTIC_SCHEMA.into(),
            question: semantic::SemanticQuestion::RestrictedResponse,
            parent_input_sha256: "d".repeat(64),
            before_result_anchor_sha256: "e".repeat(64),
            position_command: command.into(),
            expected_board_fen: root.to_fen(),
            rules_state_sha256: cpu_task::state_sha(root, &owners).unwrap(),
            rules_history_sha256: cpu_task::history_sha(root).unwrap(),
            prefix: prefix.clone(),
            root_moves: restriction.clone(),
            claimed_line: if claimed {
                vec![bits("e7e5"), bits("g1f3")]
            } else {
                Vec::new()
            },
            current_binary_sha256: SEMANTIC_BINARY.into(),
            max_wall_time_ms: 5000,
            max_output_bytes: MAX_SEMANTIC_RECEIPT_BYTES,
            context_sha256: String::new(),
        };
        semantic_request.context_sha256 =
            semantic::request_context_sha256(&semantic_request).unwrap();
        // Fixture preparation invokes Rules only, never an actual CPU/model lane.
        let semantic_raw = semantic::prepare_started(
            &serde_json::to_vec(&semantic_request).unwrap(),
            SEMANTIC_BINARY,
            Instant::now(),
        )
        .unwrap();
        let receipt: SemanticReceipt = serde_json::from_slice(&semantic_raw).unwrap();
        let mut cpu = cpu_task::Request {
            schema: cpu_task::CPU_TASK_SCHEMA.into(),
            ordering_policy: None,
            task: cpu_task::TaskKind::DefendResponse,
            parent_input_sha256: semantic_request.parent_input_sha256.clone(),
            position_command: command.into(),
            expected_board_fen: semantic_request.expected_board_fen,
            rules_state_sha256: semantic_request.rules_state_sha256,
            rules_history_sha256: semantic_request.rules_history_sha256,
            cpu_binary_sha256: OLD_BINARY.into(),
            branch_sha256: String::new(),
            prefix,
            root_moves: restriction,
            baseline_depth: 1,
            requested_depth: 2,
            max_nodes_per_check: 64,
            max_wall_time_ms: 5000,
            max_output_bytes: 65536,
            tt_entries: 32,
            quiescence_ply: 4,
            cpu_profile_sha256: String::new(),
            recheck_profile_sha256: String::new(),
            context_sha256: String::new(),
        };
        seal_cpu(&mut cpu);
        let cpu_raw = serde_json::to_string(&cpu).unwrap() + " \n";
        let binding = ReplayBindingPins {
            query_sha256: "f".repeat(64),
            catalogue_artifact: super::super::pin(b"caller catalogue original"),
            before_result_artifact: super::super::pin(b"caller Query before original"),
            prior_ledger_sha256: "1".repeat(64),
            semantic_input_sha256: "2".repeat(64),
            semantic_context_sha256: receipt.context_sha256.clone(),
            semantic_branch_meaning_sha256: receipt.branch_meaning_sha256.clone(),
            semantic_before_result_anchor_sha256: receipt.before_result_anchor_sha256.clone(),
            cpu_request_artifact: super::super::pin(cpu_raw.as_bytes()),
        };
        let mut action = super::super::Request {
            schema: super::super::SCHEMA.into(),
            query_sha256: binding.query_sha256.clone(),
            catalogue_artifact: binding.catalogue_artifact.clone(),
            before_result_artifact: binding.before_result_artifact.clone(),
            prior_ledger_sha256: binding.prior_ledger_sha256.clone(),
            action: super::super::StrategicAction {
                slot: 0,
                task: StrategicTaskKind::DefendResponse,
                semantic_input_sha256: binding.semantic_input_sha256.clone(),
                profile_registration: 0,
                baseline_depth: 1,
                requested_depth: 2,
                max_nodes_per_check: 64,
                max_wall_time_ms: 5000,
                max_output_bytes: 65536,
                budget_bucket: 0,
            },
            remaining: super::super::Remaining {
                steps: 1,
                nodes: 192,
                wall_ms: 5000,
                output_bytes: 4 * 1024 * 1024,
            },
            cpu_request_raw: cpu_raw,
            cpu_request_artifact: binding.cpu_request_artifact.clone(),
            context_sha256: String::new(),
        };
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        let action_raw = serde_json::to_string(&action).unwrap() + "\n";
        let artifacts = ReplayRegisteredArtifacts {
            legacy_cpu_binary: artifact(OLD_BINARY, 1234),
            replay_binary: artifact(NEW_BINARY, 2345),
            engine_source: artifact(&"3".repeat(64), 344097),
            wrapper_source: artifact(&"4".repeat(64), 12345),
            replay_source: artifact(&"5".repeat(64), 115515),
            provider_factory_source: artifact(&"6".repeat(64), 23456),
            opponent_recheck_source: None,
        };
        let mut registration = ReplayConsumerRegistration {
            schema: REGISTRATION_SCHEMA.into(),
            binary_pin_scope: BINARY_DECLARATION_SCOPE.into(),
            artifacts: artifacts.clone(),
            legacy_cpu_max_checks: 2,
            legacy_cpu_profile_sha256: cpu.cpu_profile_sha256.clone(),
            provider_factory_id: "caller-registered-factory-fixture-v1".into(),
            authorities: ReplayAuthorities::default(),
            context_sha256: String::new(),
        };
        registration.context_sha256 = seal(REGISTRATION_SCHEMA, &registration);
        let registration_raw = serde_json::to_string(&registration).unwrap() + "\n";
        let parent = ReplayParentPins {
            parent_input_sha256: cpu.parent_input_sha256.clone(),
            current_view_sha256: "7".repeat(64),
            frozen_admission_sha256: "8".repeat(64),
            encoding_sha256: "9".repeat(64),
        };
        let expected = ReplayExpectedPins {
            registration_artifact: super::super::pin(registration_raw.as_bytes()),
            prepared_action_artifact: super::super::pin(action_raw.as_bytes()),
            semantic_receipt_artifact: super::super::pin(&semantic_raw),
            parent: parent.clone(),
            binding: binding.clone(),
            registered_artifacts: artifacts,
            legacy_cpu_profile_sha256: cpu.cpu_profile_sha256,
            provider_factory_id: registration.provider_factory_id,
            semantic_binary_sha256: SEMANTIC_BINARY.into(),
        };
        let required = repair_replay_requirements(4, 1, 64).unwrap();
        let mut request = ReplayInputRequest {
            schema: SCHEMA.into(),
            mode,
            registration_raw,
            registration_artifact: expected.registration_artifact.clone(),
            prepared_action_raw: action_raw,
            prepared_action_artifact: expected.prepared_action_artifact.clone(),
            semantic_receipt_raw: String::from_utf8(semantic_raw).unwrap(),
            semantic_receipt_artifact: expected.semantic_receipt_artifact.clone(),
            parent,
            binding,
            config: ReplayConfigInput {
                beam_width: 1,
                line_plies: 4,
                max_nodes: 257,
                max_records: 4,
                max_role_calls: required.role_calls(),
                cpu_nodes_per_task: 64,
            },
            resources: ReplayResourceDeclaration {
                cpu_nodes: 64 * mode.checks(),
                role_calls: required.role_calls(),
                store_nodes: 257,
                store_records: 4,
                required_observations: required.observations(),
                required_line_chunks: required.line_chunks(),
                required_stages: required.stages(),
                max_rounds: 1,
                output_bytes: 3 * 65536,
                whole_wall_ms: 5000,
                cleanup_reserve_ms: 10,
            },
            authorities: ReplayAuthorities::default(),
            context_sha256: String::new(),
        };
        request.context_sha256 = seal(SCHEMA, &request);
        Fixture { request, expected }
    }
    fn admitted(
        fixture: &Fixture,
        started: Instant,
    ) -> Result<CheckedReplayInputs, ReplayInputError> {
        check_replay_inputs(
            &serde_json::to_vec(&fixture.request).unwrap(),
            &fixture.expected,
            started,
        )
    }

    fn opponent_fixture() -> Fixture {
        // A newly constructed original fixture declares 4N; production never
        // upgrades an existing action's remaining allowance.
        let mut value = fixture(ReplayInputMode::RepairEndpoint3n, true, "position startpos");
        let mode = ReplayInputMode::RepairOpponent4n;
        value.request.mode = mode;
        value.request.schema = mode.input_schema().into();
        let mut action: super::super::Request =
            serde_json::from_str(&value.request.prepared_action_raw).unwrap();
        action.remaining.nodes = 256;
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        value.request.prepared_action_raw = serde_json::to_string(&action).unwrap() + "\n";
        value.request.prepared_action_artifact =
            super::super::pin(value.request.prepared_action_raw.as_bytes());
        value.expected.prepared_action_artifact = value.request.prepared_action_artifact.clone();
        let mut registration: ReplayConsumerRegistration =
            serde_json::from_str(&value.request.registration_raw).unwrap();
        registration.schema = mode.registration_schema().into();
        registration.artifacts.opponent_recheck_source = Some(artifact(&"a".repeat(64), 4096));
        value.expected.registered_artifacts = registration.artifacts.clone();
        replace_opponent_registration(&mut value, registration);
        let required = replay_mode_requirements(mode, 4, 1, 64).unwrap();
        value.request.config.max_records = required.records;
        value.request.config.max_role_calls = required.role_calls;
        value.request.resources.cpu_nodes = required.cpu_nodes;
        value.request.resources.role_calls = required.role_calls;
        value.request.resources.store_records = required.records;
        value.request.resources.required_observations = required.observations;
        value.request.resources.required_line_chunks = required.line_chunks;
        value.request.resources.required_stages = required.stages;
        value.request.resources.output_bytes = MAX_OUTPUT_BYTES;
        value.request.context_sha256 = seal(mode.input_schema(), &value.request);
        value
    }
    fn replace_opponent_registration(
        value: &mut Fixture,
        mut registration: ReplayConsumerRegistration,
    ) {
        registration.context_sha256 = seal(&registration.schema, &registration);
        value.request.registration_raw = serde_json::to_string(&registration).unwrap() + "\n";
        value.request.registration_artifact =
            super::super::pin(value.request.registration_raw.as_bytes());
        value.expected.registration_artifact = value.request.registration_artifact.clone();
        value.request.context_sha256 = seal(value.request.mode.input_schema(), &value.request);
    }

    #[test]
    fn opponent_input_v2_admits_exact_four_n_declarations_without_execution() {
        let fixture = opponent_fixture();
        let checked = admitted(&fixture, Instant::now()).unwrap();
        let audit = checked.audit();
        assert_eq!(audit.schema, OPPONENT_SCHEMA);
        assert_eq!(audit.mode, ReplayInputMode::RepairOpponent4n);
        assert_eq!(audit.cpu_allowance, 256);
        assert!(!audit.conservative_repair_role_store_overreservation);
        assert!(!audit.cpu_engine_created && !audit.model_created && !audit.provider_created);
        assert_eq!(audit.authorities, ReplayAuthorities::default());
        assert!(audit.registered_artifacts.opponent_recheck_source.is_some());
        let (_, _, _, limits) = checked.into_owner_args();
        assert_eq!(limits.max_cpu_nodes, 256);
    }

    #[test]
    fn opponent_registration_must_be_independent_present_bounded_and_v2() {
        for fault in 0..5 {
            let mut fixture = opponent_fixture();
            let mut registration: ReplayConsumerRegistration =
                serde_json::from_str(&fixture.request.registration_raw).unwrap();
            match fault {
                0 => registration.schema = REGISTRATION_SCHEMA.into(),
                1 => registration.artifacts.opponent_recheck_source = None,
                2 => {
                    registration
                        .artifacts
                        .opponent_recheck_source
                        .as_mut()
                        .unwrap()
                        .sha256 = "b".repeat(64)
                }
                3 => {
                    registration
                        .artifacts
                        .opponent_recheck_source
                        .as_mut()
                        .unwrap()
                        .bytes = MAX_SOURCE_BYTES + 1;
                    fixture.expected.registered_artifacts = registration.artifacts.clone();
                }
                _ => {
                    registration
                        .artifacts
                        .opponent_recheck_source
                        .as_mut()
                        .unwrap()
                        .bytes = 0
                }
            }
            replace_opponent_registration(&mut fixture, registration);
            assert!(admitted(&fixture, Instant::now()).is_err());
        }
    }

    #[test]
    fn opponent_mode_schema_and_legacy_source_namespace_do_not_cross() {
        let mut new = opponent_fixture();
        new.request.schema = SCHEMA.into();
        new.request.context_sha256 = seal(SCHEMA, &new.request);
        assert!(admitted(&new, Instant::now()).is_err());
        for mode in [
            ReplayInputMode::ReplyOnly2n,
            ReplayInputMode::RepairEndpoint3n,
        ] {
            let mut old = fixture(mode, false, "position startpos");
            let mut registration: ReplayConsumerRegistration =
                serde_json::from_str(&old.request.registration_raw).unwrap();
            registration.artifacts.opponent_recheck_source = Some(artifact(&"a".repeat(64), 4096));
            old.expected.registered_artifacts = registration.artifacts.clone();
            replace_opponent_registration(&mut old, registration);
            assert!(admitted(&old, Instant::now()).is_err());
        }
    }

    #[test]
    fn absent_opponent_pin_preserves_legacy_bytes_and_explicit_null_is_refused() {
        let fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let artifacts = &fixture.expected.registered_artifacts;
        #[derive(Serialize)]
        struct Legacy<'a> {
            legacy_cpu_binary: &'a ArtifactPin,
            replay_binary: &'a ArtifactPin,
            engine_source: &'a ArtifactPin,
            wrapper_source: &'a ArtifactPin,
            replay_source: &'a ArtifactPin,
            provider_factory_source: &'a ArtifactPin,
        }
        let legacy = Legacy {
            legacy_cpu_binary: &artifacts.legacy_cpu_binary,
            replay_binary: &artifacts.replay_binary,
            engine_source: &artifacts.engine_source,
            wrapper_source: &artifacts.wrapper_source,
            replay_source: &artifacts.replay_source,
            provider_factory_source: &artifacts.provider_factory_source,
        };
        assert_eq!(
            serde_json::to_vec(artifacts).unwrap(),
            serde_json::to_vec(&legacy).unwrap()
        );
        let raw = serde_json::to_string(artifacts).unwrap();
        assert!(!raw.contains("opponent_recheck_source"));
        for added in [
            "\"opponent_recheck_source\":null,",
            "\"unknown_opponent_source\":{},",
        ] {
            let bad = raw.replacen('{', &format!("{{{added}"), 1);
            assert!(serde_json::from_str::<ReplayRegisteredArtifacts>(&bad).is_err());
        }
        let mut full = artifacts.clone();
        full.opponent_recheck_source = Some(artifact(&"a".repeat(64), 4096));
        let raw = serde_json::to_string(&full).unwrap();
        let duplicate = raw.replacen('{', "{\"opponent_recheck_source\":{\"bytes\":4096,\"sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"},", 1);
        assert!(serde_json::from_str::<ReplayRegisteredArtifacts>(&duplicate).is_err());
    }

    #[test]
    fn opponent_reservations_and_original_remaining_refuse_before_owner_creation() {
        for fault in 0..7 {
            let mut fixture = opponent_fixture();
            match fault {
                0 => fixture.request.resources.cpu_nodes = 192,
                1 => {
                    fixture.request.resources.role_calls -= 1;
                    fixture.request.config.max_role_calls -= 1;
                }
                2 => {
                    fixture.request.resources.store_records = 4;
                    fixture.request.config.max_records = 4;
                }
                3 => fixture.request.resources.required_stages = 3,
                4 => fixture.request.resources.required_observations -= 1,
                5 => fixture.request.resources.required_line_chunks -= 1,
                _ => {
                    let mut action: super::super::Request =
                        serde_json::from_str(&fixture.request.prepared_action_raw).unwrap();
                    action.remaining.nodes = 192;
                    action.context_sha256 = seal(super::super::SCHEMA, &action);
                    fixture.request.prepared_action_raw = serde_json::to_string(&action).unwrap();
                    fixture.request.prepared_action_artifact =
                        super::super::pin(fixture.request.prepared_action_raw.as_bytes());
                    fixture.expected.prepared_action_artifact =
                        fixture.request.prepared_action_artifact.clone();
                }
            }
            fixture.request.context_sha256 = seal(OPPONENT_SCHEMA, &fixture.request);
            let error = admitted(&fixture, Instant::now()).err().unwrap();
            assert_eq!(error.stage, "resource_reservation");
        }
    }

    #[test]
    fn mode_requirements_keep_legacy_overreservation_and_new_finite_counts() {
        let reply = replay_mode_requirements(ReplayInputMode::ReplyOnly2n, 5, 1, 100_000).unwrap();
        let repair =
            replay_mode_requirements(ReplayInputMode::RepairEndpoint3n, 5, 1, 100_000).unwrap();
        let opponent =
            replay_mode_requirements(ReplayInputMode::RepairOpponent4n, 5, 1, 100_000).unwrap();
        assert_eq!(
            (
                reply.cpu_nodes,
                reply.actual_role_calls(),
                reply.cpu_checks()
            ),
            (200_000, 6, 2)
        );
        assert_eq!(
            (reply.role_calls, reply.nodes, reply.records, reply.stages),
            (
                repair.role_calls,
                repair.nodes,
                repair.records,
                repair.stages
            )
        );
        assert_eq!(
            (
                repair.cpu_nodes,
                repair.actual_role_calls(),
                repair.cpu_checks()
            ),
            (300_000, 12, 3)
        );
        assert_eq!(
            (
                opponent.cpu_nodes,
                opponent.actual_role_calls(),
                opponent.cpu_checks(),
                opponent.records,
                opponent.stages
            ),
            (400_000, 14, 4, 5, 4)
        );
        assert!(replay_mode_requirements(ReplayInputMode::RepairOpponent4n, 3, 1, 1).is_err());
        assert!(
            replay_mode_requirements(ReplayInputMode::RepairOpponent4n, 5, 1, u64::MAX).is_err()
        );
    }
    fn reseal(fixture: &mut Fixture) {
        fixture.request.context_sha256 = seal(SCHEMA, &fixture.request);
    }
    fn changed_receipt(fixture: &mut Fixture, change: impl FnOnce(&mut SemanticReceipt)) {
        let mut receipt: SemanticReceipt =
            serde_json::from_str(&fixture.request.semantic_receipt_raw).unwrap();
        change(&mut receipt);
        fixture.request.semantic_receipt_raw = serde_json::to_string(&receipt).unwrap() + "\n";
        fixture.request.semantic_receipt_artifact =
            super::super::pin(fixture.request.semantic_receipt_raw.as_bytes());
        fixture.expected.semantic_receipt_artifact =
            fixture.request.semantic_receipt_artifact.clone();
        reseal(fixture);
    }

    fn construction_budget() -> ReplayConstructionBudget {
        ReplayConstructionBudget {
            max_outer_bytes: MAX_REQUEST_BYTES,
            max_construction_credit_bytes: MAX_CONSTRUCTION_CREDIT_BYTES,
        }
    }
    fn constructed(
        fixture: &Fixture,
        scope: SemanticReceiptProducerScope,
        budget: ReplayConstructionBudget,
        started: Instant,
    ) -> Result<PreparedReplayRequest, ReplayPreparationError> {
        prepare_replay_request(
            ReplayOriginals {
                registration: fixture.request.registration_raw.as_bytes(),
                prepared_action: fixture.request.prepared_action_raw.as_bytes(),
                semantic_receipt: fixture.request.semantic_receipt_raw.as_bytes(),
            },
            ReplayPreparationDeclaration {
                mode: fixture.request.mode,
                config: &fixture.request.config,
                resources: &fixture.request.resources,
            },
            &fixture.expected,
            scope,
            budget,
            started,
        )
    }

    /// CPU Rules fixture only. No native/role work, witness or caller admission.
    pub(super) fn controlled_prepared_result_rules() -> PreparedReplayRequest {
        constructed(
            &opponent_fixture(),
            PRODUCER_SCOPES[0],
            construction_budget(),
            Instant::now(),
        )
        .unwrap()
    }

    #[test]
    fn result_rules_reconstruction_preserves_full_history_and_original_whole_window() {
        let fixture = fixture(
            ReplayInputMode::RepairEndpoint3n,
            true,
            "position startpos moves g1f3 g8f6 f3g1 f6g8",
        );
        let mut prepared = constructed(
            &fixture,
            PRODUCER_SCOPES[0],
            construction_budget(),
            Instant::now(),
        )
        .unwrap();
        let original_whole = prepared.deadline();
        // Pure result validation is allowed after E, but never changes it.
        prepared.execution_deadline = Instant::now();
        let original_execution = prepared.execution_deadline();
        let reconstructed = prepared
            .reconstruct_result_root(&AtomicBool::new(false))
            .unwrap();
        assert_eq!(
            reconstructed.root.history_completeness(),
            HistoryCompleteness::Complete
        );
        assert_eq!(
            reconstructed.root.history_origin(),
            HistoryOrigin::StartPosition
        );
        assert_eq!(reconstructed.root.known_history_len(), 5);
        assert_ne!(
            cpu_task::history_sha(&reconstructed.root).unwrap(),
            cpu_task::history_sha(&Position::startpos()).unwrap()
        );
        assert_eq!(
            reconstructed.prefix,
            vec![BoardMove::from_uci("e2e4").unwrap()]
        );
        assert_eq!(prepared.deadline(), original_whole);
        assert_eq!(prepared.execution_deadline(), original_execution);
        assert_eq!(prepared.audit().input_admission.cpu_checks, 0);
        assert!(!prepared.audit().input_admission.cpu_engine_created);
    }

    #[test]
    fn result_rules_reconstruction_refuses_cancellation_expiry_and_backing_changes() {
        let mut prepared = controlled_prepared_result_rules();
        assert_eq!(
            prepared
                .reconstruct_result_root(&AtomicBool::new(true))
                .err()
                .unwrap()
                .stage,
            "result_rules_control"
        );
        prepared.bytes.push(b' ');
        assert_eq!(
            prepared
                .reconstruct_result_root(&AtomicBool::new(false))
                .err()
                .unwrap()
                .stage,
            "result_rules_input"
        );
        let mut prepared = controlled_prepared_result_rules();
        prepared.whole_deadline = Instant::now();
        assert_eq!(
            prepared
                .reconstruct_result_root(&AtomicBool::new(false))
                .err()
                .unwrap()
                .stage,
            "result_rules_control"
        );
    }

    #[test]
    fn construction_preserves_all_original_whitespace_and_exact_independent_pins_without_engines() {
        for mode in [
            ReplayInputMode::ReplyOnly2n,
            ReplayInputMode::RepairEndpoint3n,
        ] {
            let mut value = fixture(mode, false, "position startpos");
            value.request.registration_raw =
                format!("\r\n\t{} \r\n", value.request.registration_raw);
            value.request.prepared_action_raw =
                format!("\t{}\r\n ", value.request.prepared_action_raw);
            value.request.semantic_receipt_raw =
                format!("\n{}\t\r\n", value.request.semantic_receipt_raw);
            // Independent fixture registration records the original bytes. The
            // production constructor cannot choose or replace these expected pins.
            value.expected.registration_artifact =
                super::super::pin(value.request.registration_raw.as_bytes());
            value.expected.prepared_action_artifact =
                super::super::pin(value.request.prepared_action_raw.as_bytes());
            value.expected.semantic_receipt_artifact =
                super::super::pin(value.request.semantic_receipt_raw.as_bytes());
            let started = Instant::now();
            let prepared =
                constructed(&value, PRODUCER_SCOPES[0], construction_budget(), started).unwrap();
            let wire: ReplayInputRequest = serde_json::from_slice(prepared.as_bytes()).unwrap();
            assert_eq!(
                wire.registration_raw.as_bytes(),
                value.request.registration_raw.as_bytes()
            );
            assert_eq!(
                wire.prepared_action_raw.as_bytes(),
                value.request.prepared_action_raw.as_bytes()
            );
            assert_eq!(
                wire.semantic_receipt_raw.as_bytes(),
                value.request.semantic_receipt_raw.as_bytes()
            );
            assert_eq!(prepared.artifact(), &super::super::pin(prepared.as_bytes()));
            assert_eq!(wire.context_sha256, seal(SCHEMA, &wire));
            assert_eq!(prepared.deadline(), started + Duration::from_millis(5000));
            assert_eq!(
                prepared.execution_deadline(),
                started + Duration::from_millis(4990)
            );
            let audit = prepared.audit();
            assert_eq!(audit.input_admission.mode, mode);
            assert_eq!(
                audit.input_admission.registration_artifact,
                value.expected.registration_artifact
            );
            assert_eq!(
                audit.input_admission.prepared_action_artifact,
                value.expected.prepared_action_artifact
            );
            assert_eq!(
                audit.input_admission.semantic_receipt_artifact,
                value.expected.semantic_receipt_artifact
            );
            assert_eq!(
                audit.input_admission.authorities,
                ReplayAuthorities::default()
            );
            assert_eq!(audit.input_admission.cpu_checks, 0);
            assert_eq!(audit.input_admission.actual_utility_groups, 0);
            assert!(
                !audit.input_admission.cpu_engine_created
                    && !audit.input_admission.model_created
                    && !audit.input_admission.provider_created
            );
            assert!(
                !audit.construction.allocator_peak_observed
                    && !audit.construction.validator_peak_observed
                    && !audit.construction.rss_peak_observed
            );
            assert_eq!(
                audit.construction.final_retained_capacity_bytes,
                prepared.as_bytes().len()
            );
        }
    }
    #[test]
    fn construction_counts_escaped_wire_and_rejects_exact_outer_or_credit_shortage_before_backing()
    {
        let value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let prepared = constructed(
            &value,
            PRODUCER_SCOPES[0],
            construction_budget(),
            Instant::now(),
        )
        .unwrap();
        let credit = prepared.audit().construction;
        let wire: ReplayInputRequest = serde_json::from_slice(prepared.as_bytes()).unwrap();
        assert_eq!(
            credit.outer_json_bytes,
            serde_json::to_vec(&wire).unwrap().len()
        );
        let mut body = serde_json::to_value(&wire).unwrap();
        body.as_object_mut().unwrap().remove("context_sha256");
        assert_eq!(
            credit.context_json_bytes,
            serde_json::to_vec(&json!([SCHEMA, body])).unwrap().len()
        );
        assert_eq!(
            credit.policy_bytes,
            CONSTRUCTION_RAW_PAYLOAD_COPIES * credit.raw_utf8_bytes
                + 2 * credit.context_json_bytes
                + credit.outer_json_bytes
                + 65536
        );
        // Each original contains quotes, backslashes and line endings; exact
        // JSON escaping is counted rather than a blanket multiplier allowance.
        assert!(credit.outer_json_bytes > credit.raw_utf8_bytes);
        let outer_error = constructed(
            &value,
            PRODUCER_SCOPES[0],
            ReplayConstructionBudget {
                max_outer_bytes: credit.outer_json_bytes - 1,
                max_construction_credit_bytes: credit.policy_bytes,
            },
            Instant::now(),
        )
        .unwrap_err();
        assert_eq!(outer_error.stage, "count");
        assert!(matches!(
            outer_error.cause.as_ref(),
            ReplayPreparationCause::Serialization {
                writer: Some(ReplayConstructionWriterFailure::ByteLimit),
                ..
            }
        ));
        assert!(outer_error.rejected_outer.is_none());
        let credit_error = constructed(
            &value,
            PRODUCER_SCOPES[0],
            ReplayConstructionBudget {
                max_outer_bytes: credit.outer_json_bytes,
                max_construction_credit_bytes: credit.policy_bytes - 1,
            },
            Instant::now(),
        )
        .unwrap_err();
        assert_eq!(credit_error.stage, "count");
        assert!(
            matches!(credit_error.cause.as_ref(), ReplayPreparationCause::Refusal {
            actual: Some(actual), limit: Some(limit), ..
        } if *actual == credit.policy_bytes && *limit + 1 == *actual)
        );
        assert!(credit_error.rejected_outer.is_none());
        let exact = constructed(
            &value,
            PRODUCER_SCOPES[0],
            ReplayConstructionBudget {
                max_outer_bytes: credit.outer_json_bytes,
                max_construction_credit_bytes: credit.policy_bytes,
            },
            Instant::now(),
        )
        .unwrap();
        assert_eq!(exact.as_bytes(), prepared.as_bytes());
    }
    #[test]
    fn construction_three_raw_copy_credit_is_exact_and_prior_two_copy_credit_refuses_before_backing()
     {
        let value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let prepared = constructed(
            &value,
            PRODUCER_SCOPES[0],
            construction_budget(),
            Instant::now(),
        )
        .unwrap();
        let credit = prepared.audit().construction;
        assert_eq!(credit.raw_payload_copies, 3);
        let prior_two_copy_credit = 2 * credit.raw_utf8_bytes
            + 2 * credit.context_json_bytes
            + credit.outer_json_bytes
            + 65536;
        let corrected_three_copy_credit = 3 * credit.raw_utf8_bytes
            + 2 * credit.context_json_bytes
            + credit.outer_json_bytes
            + 65536;
        assert_eq!(credit.policy_bytes, corrected_three_copy_credit);
        assert_eq!(
            corrected_three_copy_credit - prior_two_copy_credit,
            credit.raw_utf8_bytes
        );
        assert!(credit.raw_utf8_bytes > 0);
        let exact_start = Instant::now();
        let exact = constructed(
            &value,
            PRODUCER_SCOPES[0],
            ReplayConstructionBudget {
                max_outer_bytes: credit.outer_json_bytes,
                max_construction_credit_bytes: corrected_three_copy_credit,
            },
            exact_start,
        )
        .unwrap();
        // Credit policy changes no wire, original artifact identity or lifecycle
        // authority. These assertions do not measure the allocator's peak.
        assert_eq!(exact.as_bytes(), prepared.as_bytes());
        assert_eq!(exact.artifact(), prepared.artifact());
        assert_eq!(exact.artifact(), &super::super::pin(exact.as_bytes()));
        assert_eq!(exact.audit().construction.raw_payload_copies, 3);
        assert_eq!(
            exact.audit().construction.policy_bytes,
            corrected_three_copy_credit
        );
        assert_eq!(
            exact.audit().input_admission.authorities,
            ReplayAuthorities::default()
        );
        assert!(!exact.audit().construction.allocator_peak_observed);
        assert!(!exact.audit().construction.validator_peak_observed);
        assert!(!exact.audit().construction.rss_peak_observed);
        assert_eq!(exact.deadline(), exact_start + Duration::from_millis(5000));
        assert_eq!(
            exact.execution_deadline(),
            exact_start + Duration::from_millis(4990)
        );
        let refused_start = Instant::now();
        let error = constructed(
            &value,
            PRODUCER_SCOPES[0],
            ReplayConstructionBudget {
                max_outer_bytes: credit.outer_json_bytes,
                max_construction_credit_bytes: prior_two_copy_credit,
            },
            refused_start,
        )
        .unwrap_err();
        assert_eq!(error.stage, "count");
        assert!(
            matches!(error.cause.as_ref(), ReplayPreparationCause::Refusal {
            actual: Some(actual), limit: Some(limit), ..
        } if *actual == corrected_three_copy_credit && *limit == prior_two_copy_credit)
        );
        // The policy gate precedes final backing, context Value construction,
        // serialization and checker admission, so no candidate bytes are held.
        assert!(error.rejected_outer.is_none());
        assert_eq!(
            error.diagnostics.expected_semantic_receipt_producer_scope,
            PRODUCER_SCOPES[0]
        );
        assert_eq!(
            error.diagnostics.deadline(),
            Some(refused_start + Duration::from_millis(5000))
        );
        assert_eq!(
            error.diagnostics.execution_deadline(),
            Some(refused_start + Duration::from_millis(4990))
        );
        assert_eq!(
            error.diagnostics.output_limit,
            value.request.resources.output_bytes
        );
    }
    #[test]
    fn construction_keeps_scoped_checker_error_and_rejected_outer_as_owned_evidence() {
        let mut value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        changed_receipt(&mut value, |receipt| {
            receipt.binary_pin_scope = PRODUCER_SCOPES[1].as_str().into();
        });
        let started = Instant::now();
        let error =
            constructed(&value, PRODUCER_SCOPES[2], construction_budget(), started).unwrap_err();
        assert_eq!(error.stage, "scoped_input_check");
        let ReplayPreparationCause::Input(original_error) = error.cause.as_ref() else {
            panic!("typed original checker error");
        };
        assert_eq!(original_error.stage, "semantic_binding");
        assert_eq!(
            original_error.expected_semantic_receipt_producer_scope,
            Some(PRODUCER_SCOPES[2])
        );
        assert!(original_error.original_whole_deadline_retained);
        assert_eq!(error.diagnostics.original_started(), started);
        let rejected = error.rejected_outer.as_ref().unwrap();
        assert!(rejected.serialization_complete());
        assert_eq!(
            rejected.artifact(),
            Some(&super::super::pin(rejected.as_bytes()))
        );
        let wire: ReplayInputRequest = serde_json::from_slice(rejected.as_bytes()).unwrap();
        assert_eq!(
            wire.semantic_receipt_raw.as_bytes(),
            value.request.semantic_receipt_raw.as_bytes()
        );
        assert_eq!(wire.authorities, ReplayAuthorities::default());
        let matching = constructed(
            &value,
            PRODUCER_SCOPES[1],
            construction_budget(),
            Instant::now(),
        )
        .unwrap();
        assert_eq!(
            matching
                .audit()
                .input_admission
                .expected_semantic_receipt_producer_scope,
            Some(PRODUCER_SCOPES[1])
        );
    }
    #[test]
    fn construction_preflight_retains_real_utf8_and_rejects_pin_or_oversized_expected_before_count()
    {
        let mut value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let mut invalid = value.request.semantic_receipt_raw.as_bytes().to_vec();
        invalid.push(0xff);
        value.expected.semantic_receipt_artifact = super::super::pin(&invalid);
        let error = prepare_replay_request(
            ReplayOriginals {
                registration: value.request.registration_raw.as_bytes(),
                prepared_action: value.request.prepared_action_raw.as_bytes(),
                semantic_receipt: &invalid,
            },
            ReplayPreparationDeclaration {
                mode: value.request.mode,
                config: &value.request.config,
                resources: &value.request.resources,
            },
            &value.expected,
            PRODUCER_SCOPES[0],
            construction_budget(),
            Instant::now(),
        )
        .unwrap_err();
        assert_eq!(error.stage, "preflight");
        assert!(
            matches!(error.cause.as_ref(), ReplayPreparationCause::Utf8 { original: ReplayOriginalKind::SemanticReceipt, source } if source.valid_up_to() == invalid.len() - 1)
        );
        assert!(std::error::Error::source(&error).is_some());
        assert!(error.rejected_outer.is_none());
        let expired_start = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();
        let expired_utf8 = prepare_replay_request(
            ReplayOriginals {
                registration: value.request.registration_raw.as_bytes(),
                prepared_action: value.request.prepared_action_raw.as_bytes(),
                semantic_receipt: &invalid,
            },
            ReplayPreparationDeclaration {
                mode: value.request.mode,
                config: &value.request.config,
                resources: &value.request.resources,
            },
            &value.expected,
            PRODUCER_SCOPES[2],
            construction_budget(),
            expired_start,
        )
        .unwrap_err();
        assert!(matches!(
            expired_utf8.cause.as_ref(),
            ReplayPreparationCause::Utf8 { .. }
        ));
        assert_eq!(
            expired_utf8
                .diagnostics
                .expected_semantic_receipt_producer_scope,
            PRODUCER_SCOPES[2]
        );
        assert_eq!(
            expired_utf8.diagnostics.deadline(),
            Some(expired_start + Duration::from_millis(5000))
        );
        assert_eq!(
            expired_utf8.diagnostics.execution_deadline(),
            Some(expired_start + Duration::from_millis(4990))
        );
        assert!(
            expired_utf8.diagnostics.deadline_exceeded
                && expired_utf8.diagnostics.execution_deadline_exceeded
        );
        assert_eq!(
            expired_utf8.diagnostics.output_limit,
            value.request.resources.output_bytes
        );
        let mut value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        value.request.semantic_receipt_raw.push(' ');
        let pin_error = constructed(
            &value,
            PRODUCER_SCOPES[0],
            construction_budget(),
            Instant::now(),
        )
        .unwrap_err();
        assert_eq!(pin_error.stage, "preflight");
        assert!(matches!(
            pin_error.cause.as_ref(),
            ReplayPreparationCause::Refusal { .. }
        ));
        assert!(pin_error.rejected_outer.is_none());
        for invalid_field in 0..4 {
            let mut value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            match invalid_field {
                0 => value.expected.parent.encoding_sha256 = "a".repeat(65),
                1 => value.expected.registered_artifacts.wrapper_source.sha256 = "a".repeat(65),
                2 => value.expected.provider_factory_id = "a".repeat(129),
                _ => value.expected.registration_artifact.bytes = MAX_REGISTRATION_BYTES as u64 + 1,
            }
            let error = constructed(
                &value,
                PRODUCER_SCOPES[0],
                construction_budget(),
                Instant::now(),
            )
            .unwrap_err();
            assert_eq!(error.stage, "preflight");
            assert!(matches!(
                error.cause.as_ref(),
                ReplayPreparationCause::Input(_)
            ));
            assert!(error.rejected_outer.is_none());
        }
        for budget in [
            ReplayConstructionBudget {
                max_outer_bytes: MAX_REQUEST_BYTES + 1,
                max_construction_credit_bytes: MAX_CONSTRUCTION_CREDIT_BYTES,
            },
            ReplayConstructionBudget {
                max_outer_bytes: MAX_REQUEST_BYTES,
                max_construction_credit_bytes: MAX_CONSTRUCTION_CREDIT_BYTES + 1,
            },
        ] {
            let value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            let error =
                constructed(&value, PRODUCER_SCOPES[0], budget, Instant::now()).unwrap_err();
            assert_eq!(error.stage, "preflight");
            assert!(matches!(
                error.cause.as_ref(),
                ReplayPreparationCause::Refusal { .. }
            ));
            assert!(error.rejected_outer.is_none());
        }
    }
    #[test]
    fn construction_never_expands_declared_cpu_role_store_or_output_resources() {
        for missing in 0..5 {
            let mut value = fixture(
                ReplayInputMode::RepairEndpoint3n,
                false,
                "position startpos",
            );
            match missing {
                0 => value.request.resources.cpu_nodes = 128,
                1 => value.request.resources.role_calls -= 1,
                2 => value.request.resources.store_nodes -= 1,
                3 => value.request.resources.required_stages -= 1,
                _ => value.request.resources.output_bytes = 2 * 65536,
            }
            let error = constructed(
                &value,
                PRODUCER_SCOPES[0],
                construction_budget(),
                Instant::now(),
            )
            .unwrap_err();
            let ReplayPreparationCause::Input(checker) = error.cause.as_ref() else {
                panic!("existing resource checker");
            };
            assert_eq!(checker.stage, "resource_reservation");
            let rejected = error.rejected_outer.as_ref().unwrap();
            let wire: ReplayInputRequest = serde_json::from_slice(rejected.as_bytes()).unwrap();
            assert_eq!(
                serde_json::to_value(&wire.resources).unwrap(),
                serde_json::to_value(&value.request.resources).unwrap()
            );
            assert_eq!(wire.mode, ReplayInputMode::RepairEndpoint3n);
            assert_eq!(
                wire.prepared_action_raw.as_bytes(),
                value.request.prepared_action_raw.as_bytes()
            );
        }
    }
    #[test]
    fn construction_preserves_old_two_n_remaining_and_requires_explicit_new_mode_resources() {
        let mut value = fixture(
            ReplayInputMode::RepairEndpoint3n,
            false,
            "position startpos",
        );
        let mut action: super::super::Request =
            serde_json::from_str(&value.request.prepared_action_raw).unwrap();
        action.remaining.nodes = 128;
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        value.request.prepared_action_raw = serde_json::to_string(&action).unwrap() + "\r\n";
        value.expected.prepared_action_artifact =
            super::super::pin(value.request.prepared_action_raw.as_bytes());
        let error = constructed(
            &value,
            PRODUCER_SCOPES[0],
            construction_budget(),
            Instant::now(),
        )
        .unwrap_err();
        let ReplayPreparationCause::Input(checker) = error.cause.as_ref() else {
            panic!("original resource checker");
        };
        assert_eq!(checker.stage, "resource_reservation");
        let rejected = error.rejected_outer.as_ref().unwrap();
        let wire: ReplayInputRequest = serde_json::from_slice(rejected.as_bytes()).unwrap();
        assert_eq!(wire.mode, ReplayInputMode::RepairEndpoint3n);
        assert_eq!(wire.resources.cpu_nodes, 192);
        assert_eq!(
            wire.prepared_action_raw.as_bytes(),
            value.request.prepared_action_raw.as_bytes()
        );
        let retained_action: super::super::Request =
            serde_json::from_str(&wire.prepared_action_raw).unwrap();
        assert_eq!(retained_action.remaining.nodes, 128);
        assert_eq!(
            retained_action.cpu_request_raw.as_bytes(),
            action.cpu_request_raw.as_bytes()
        );
        // The caller supplies a separate explicit 2N declaration. The function
        // does not clip 3N or silently mutate the historical action remaining.
        value.request.mode = ReplayInputMode::ReplyOnly2n;
        value.request.resources.cpu_nodes = 128;
        let prepared = constructed(
            &value,
            PRODUCER_SCOPES[0],
            construction_budget(),
            Instant::now(),
        )
        .unwrap();
        let wire: ReplayInputRequest = serde_json::from_slice(prepared.as_bytes()).unwrap();
        assert_eq!(wire.mode, ReplayInputMode::ReplyOnly2n);
        assert_eq!(wire.resources.cpu_nodes, 128);
        assert_eq!(
            wire.prepared_action_raw.as_bytes(),
            value.request.prepared_action_raw.as_bytes()
        );
        assert_eq!(prepared.audit().input_admission.cpu_allowance, 128);
        assert!(
            prepared
                .audit()
                .input_admission
                .conservative_repair_role_store_overreservation
        );
    }
    #[test]
    fn construction_preserves_known_clock_scope_for_early_refusal_and_keeps_malformed_legacy_unknown()
     {
        let scope = PRODUCER_SCOPES[1];
        let mut value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let started = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();
        let error = constructed(
            &value,
            scope,
            ReplayConstructionBudget {
                max_outer_bytes: 0,
                max_construction_credit_bytes: 1,
            },
            started,
        )
        .unwrap_err();
        assert_eq!(error.stage, "preflight");
        assert_eq!(
            error.diagnostics.expected_semantic_receipt_producer_scope,
            scope
        );
        assert_eq!(
            error.diagnostics.deadline(),
            Some(started + Duration::from_millis(5000))
        );
        assert_eq!(
            error.diagnostics.execution_deadline(),
            Some(started + Duration::from_millis(4990))
        );
        assert!(
            error.diagnostics.deadline_exceeded && error.diagnostics.execution_deadline_exceeded
        );
        assert_eq!(
            error.diagnostics.output_limit,
            value.request.resources.output_bytes
        );
        let expired = constructed(&value, scope, construction_budget(), started).unwrap_err();
        assert_eq!(expired.stage, "count");
        assert!(expired.diagnostics.deadline_exceeded);
        assert!(expired.rejected_outer.is_none());
        value.request.resources.output_bytes = 0;
        let output = constructed(&value, scope, construction_budget(), started).unwrap_err();
        assert_eq!(output.stage, "resource_clock");
        assert_eq!(output.diagnostics.output_limit, DIAGNOSTIC_OUTPUT_BYTES);
        assert_eq!(
            output.diagnostics.deadline(),
            Some(started + Duration::from_millis(5000))
        );
        assert!(output.diagnostics.deadline_exceeded);
        let ReplayPreparationCause::Input(original) = output.cause.as_ref() else {
            panic!("original invalid-output cause");
        };
        assert_eq!(original.stage, "output_bound");
        assert!(original.deadline_exceeded);
        let legacy = check_replay_inputs(b"{", &value.expected, started).unwrap_err();
        assert_eq!(legacy.stage, "input_json");
        assert!(!legacy.original_whole_deadline_retained);
        assert!(legacy.elapsed_ms.is_some());
        assert!(
            legacy.whole_wall_ms.is_none()
                && legacy.expected_semantic_receipt_producer_scope.is_none()
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
    fn construction_future_start_and_malformed_original_keep_typed_causes_and_same_partitions() {
        let mut value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let future = Instant::now() + Duration::from_secs(60);
        let error =
            constructed(&value, PRODUCER_SCOPES[0], construction_budget(), future).unwrap_err();
        let ReplayPreparationCause::Input(original) = error.cause.as_ref() else {
            panic!("future start clock cause");
        };
        assert_eq!(original.stage, "original_clock");
        assert_eq!(
            error.diagnostics.deadline(),
            Some(future + Duration::from_millis(5000))
        );
        assert_eq!(
            error.diagnostics.execution_deadline(),
            Some(future + Duration::from_millis(4990))
        );
        assert_eq!(error.diagnostics.elapsed_ms, 0);
        assert!(error.rejected_outer.is_none());
        value.request.semantic_receipt_raw = "{\n".into();
        value.expected.semantic_receipt_artifact =
            super::super::pin(value.request.semantic_receipt_raw.as_bytes());
        let started = Instant::now();
        let error =
            constructed(&value, PRODUCER_SCOPES[0], construction_budget(), started).unwrap_err();
        let ReplayPreparationCause::Input(original) = error.cause.as_ref() else {
            panic!("original malformed receipt cause");
        };
        assert_eq!(original.stage, "semantic_receipt_json");
        assert!(original.original_whole_deadline_retained);
        assert_eq!(
            original.expected_semantic_receipt_producer_scope,
            Some(PRODUCER_SCOPES[0])
        );
        assert_eq!(
            error.diagnostics.deadline(),
            Some(started + Duration::from_millis(5000))
        );
        let rejected = error.rejected_outer.as_ref().unwrap();
        assert!(rejected.serialization_complete());
        let wire: ReplayInputRequest = serde_json::from_slice(rejected.as_bytes()).unwrap();
        assert_eq!(wire.semantic_receipt_raw, "{\n");
    }
    #[test]
    fn construction_writer_preserves_partial_bytes_and_real_serde_failure_without_growth() {
        let started = Instant::now();
        let clock = Clock {
            whole_deadline: started + Duration::from_secs(10),
            execution_deadline: started + Duration::from_secs(9),
            whole_wall_ms: 10000,
            cleanup_reserve_ms: 1000,
            output_limit: 1024,
        };
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(4).unwrap();
        let retained = bytes.capacity();
        let (cause, written) = {
            let mut writer = ConstructionWriter {
                backing: Some(&mut bytes),
                bytes: 0,
                limit: 4,
                clock,
                failure: None,
            };
            let source = serde_json::to_writer(&mut writer, &"abc\n").unwrap_err();
            assert_eq!(
                writer.failure,
                Some(ReplayConstructionWriterFailure::ByteLimit)
            );
            (
                ReplayPreparationCause::Serialization {
                    source,
                    writer: writer.failure,
                },
                writer.bytes,
            )
        };
        assert_eq!(bytes.capacity(), retained);
        assert_eq!(bytes.len(), written);
        assert!(!bytes.is_empty());
        let error = preparation_error(
            "serialize",
            cause,
            started,
            Some(clock),
            PRODUCER_SCOPES[0],
            Some(RejectedReplayBytes {
                bytes,
                artifact: None,
                serialization_complete: false,
            }),
        );
        assert!(std::error::Error::source(&error).is_some());
        let rejected = error.rejected_outer.as_ref().unwrap();
        assert!(!rejected.serialization_complete());
        assert!(rejected.artifact().is_none());
    }

    #[test]
    fn explicit_semantic_scopes_are_exact_and_legacy_remains_library_only() {
        for actual_scope in PRODUCER_SCOPES {
            let mut value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            changed_receipt(&mut value, |receipt| {
                receipt.binary_pin_scope = actual_scope.as_str().into();
            });
            let raw = serde_json::to_vec(&value.request).unwrap();
            for expected_scope in PRODUCER_SCOPES {
                let result = check_replay_inputs_with_semantic_scope(
                    &raw,
                    &value.expected,
                    expected_scope,
                    Instant::now(),
                );
                if actual_scope == expected_scope {
                    let checked = result.unwrap();
                    assert_eq!(
                        checked.expected_semantic_receipt_producer_scope(),
                        Some(expected_scope)
                    );
                    let audit = checked.audit();
                    assert_eq!(
                        audit.expected_semantic_receipt_producer_scope,
                        Some(expected_scope)
                    );
                    assert_eq!(audit.binary_pin_scope, BINARY_DECLARATION_SCOPE);
                    assert_eq!(audit.authorities, ReplayAuthorities::default());
                    assert_eq!(audit.cpu_checks, 0);
                    assert!(
                        !audit.cpu_engine_created
                            && !audit.model_created
                            && !audit.provider_created
                    );
                    assert_eq!(audit.actual_utility_groups, 0);
                    assert_eq!(
                        serde_json::to_value(&audit).unwrap()["expected_semantic_receipt_producer_scope"],
                        expected_scope.as_str(),
                    );
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(error.stage, "semantic_binding");
                    assert_eq!(
                        error.expected_semantic_receipt_producer_scope,
                        Some(expected_scope)
                    );
                }
            }
            let legacy = check_replay_inputs(&raw, &value.expected, Instant::now());
            if actual_scope == SemanticReceiptProducerScope::LibraryDispatcherArgument {
                let checked = legacy.unwrap();
                assert!(checked.expected_semantic_receipt_producer_scope().is_none());
                assert!(
                    !serde_json::to_value(checked.audit())
                        .unwrap()
                        .as_object()
                        .unwrap()
                        .contains_key("expected_semantic_receipt_producer_scope")
                );
            } else {
                let error = legacy.unwrap_err();
                assert_eq!(error.stage, "semantic_binding");
                assert!(error.expected_semantic_receipt_producer_scope.is_none());
                assert!(
                    !serde_json::to_value(&error)
                        .unwrap()
                        .as_object()
                        .unwrap()
                        .contains_key("expected_semantic_receipt_producer_scope")
                );
            }
        }
    }
    #[test]
    fn explicit_scope_cannot_repin_reseal_or_enable_original_authorities() {
        let scope = SemanticReceiptProducerScope::LinuxLoadedExecutableInode;
        for field in 0..3 {
            let mut value = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            changed_receipt(&mut value, |receipt| {
                receipt.binary_pin_scope = scope.as_str().into();
            });
            match field {
                0 => {
                    value.request.semantic_receipt_raw.push(' ');
                    value.request.semantic_receipt_artifact =
                        super::super::pin(value.request.semantic_receipt_raw.as_bytes());
                    // The raw pin from independent registration remains original.
                    reseal(&mut value);
                }
                1 => value.request.context_sha256 = "0".repeat(64),
                _ => {
                    value.request.authorities.loaded_image_observed = true;
                    reseal(&mut value);
                }
            }
            let error = check_replay_inputs_with_semantic_scope(
                &serde_json::to_vec(&value.request).unwrap(),
                &value.expected,
                scope,
                Instant::now(),
            )
            .unwrap_err();
            assert_eq!(error.expected_semantic_receipt_producer_scope, Some(scope));
            assert_eq!(
                error.stage,
                match field {
                    0 => "original_artifact_identity",
                    1 => "context_identity",
                    _ => "domain_authority",
                }
            );
        }
    }
    #[test]
    fn semantic_scope_wire_accepts_only_the_three_exact_string_literals() {
        assert_eq!(
            SemanticReceiptProducerScope::LibraryDispatcherArgument.as_str(),
            semantic::BINARY_PIN_SCOPE
        );
        for scope in PRODUCER_SCOPES {
            let raw = serde_json::to_vec(&scope).unwrap();
            assert_eq!(
                serde_json::from_slice::<SemanticReceiptProducerScope>(&raw).unwrap(),
                scope
            );
            assert_eq!(
                serde_json::from_value::<SemanticReceiptProducerScope>(json!(scope.as_str()))
                    .unwrap(),
                scope
            );
            assert_eq!(
                serde_json::from_slice::<String>(&raw).unwrap(),
                scope.as_str()
            );
        }
        for raw in [
            "null",
            "true",
            "0",
            "-1",
            "0.5",
            "[]",
            "{}",
            r#""LibraryDispatcherArgument""#,
            r#""library_dispatcher_argument""#,
            r#"{"dispatcher_compared_verified_argument":null}"#,
            r#"{"linux_loaded_executable_inode":null,"linux_loaded_executable_inode":null}"#,
        ] {
            assert!(
                serde_json::from_str::<SemanticReceiptProducerScope>(raw).is_err(),
                "{raw}"
            );
        }
    }
    #[test]
    fn explicit_scope_errors_keep_known_original_clock_and_unknown_clock_distinct() {
        let scope = SemanticReceiptProducerScope::CurrentExePathHash;
        let started = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();
        let mut value = fixture(
            ReplayInputMode::RepairEndpoint3n,
            false,
            "position startpos",
        );
        value.request.resources.output_bytes = 0;
        reseal(&mut value);
        let error = check_replay_inputs_with_semantic_scope(
            &serde_json::to_vec(&value.request).unwrap(),
            &value.expected,
            scope,
            started,
        )
        .unwrap_err();
        assert_eq!(error.stage, "output_bound");
        assert_eq!(error.expected_semantic_receipt_producer_scope, Some(scope));
        assert!(error.original_whole_deadline_retained && error.deadline_exceeded);
        assert_eq!(error.whole_wall_ms, Some(5000));
        assert_eq!(error.output_limit, DIAGNOSTIC_OUTPUT_BYTES);
        assert!(error.elapsed_ms.unwrap() >= 10000);
        let unknown =
            check_replay_inputs_with_semantic_scope(b"{", &value.expected, scope, started)
                .unwrap_err();
        assert_eq!(unknown.stage, "input_json");
        assert_eq!(
            unknown.expected_semantic_receipt_producer_scope,
            Some(scope)
        );
        assert!(!unknown.original_whole_deadline_retained);
        assert!(unknown.whole_wall_ms.is_none());
        let legacy = check_replay_inputs(b"{", &value.expected, started).unwrap_err();
        assert!(legacy.expected_semantic_receipt_producer_scope.is_none());
        assert!(
            !serde_json::to_value(&legacy)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("expected_semantic_receipt_producer_scope")
        );
    }

    #[test]
    fn original_bytes_rules_and_two_distinct_modes_admit_without_engines() {
        for mode in [
            ReplayInputMode::ReplyOnly2n,
            ReplayInputMode::RepairEndpoint3n,
        ] {
            let fixture = fixture(mode, false, "position startpos");
            let started = Instant::now();
            let checked = admitted(&fixture, started).unwrap();
            assert_eq!(
                checked.original_prepared_action(),
                fixture.request.prepared_action_raw.as_bytes()
            );
            assert_eq!(
                checked.original_semantic_receipt(),
                fixture.request.semantic_receipt_raw.as_bytes()
            );
            assert_eq!(checked.deadline(), started + Duration::from_millis(5000));
            assert_eq!(
                checked.execution_deadline(),
                started + Duration::from_millis(4990)
            );
            let audit = checked.audit();
            assert_eq!(audit.cpu_allowance, 64 * mode.checks());
            assert_eq!(
                audit.conservative_repair_role_store_overreservation,
                mode == ReplayInputMode::ReplyOnly2n
            );
            assert_eq!(audit.cpu_checks, 0);
            assert!(!audit.cpu_engine_created && !audit.model_created && !audit.provider_created);
            assert_eq!(audit.authorities, ReplayAuthorities::default());
            assert_eq!(audit.actual_utility_groups, 0);
            let (_, root, plan, limits) = checked.into_owner_args();
            assert_eq!(root.side_to_move(), rz_position::Color::White);
            let mut actual_target = root.clone();
            for movement in &plan.prefix {
                actual_target.make_move(*movement).unwrap();
            }
            assert_eq!(actual_target.side_to_move(), rz_position::Color::Black);
            assert!(plan.expected_target.same_state(&actual_target.snapshot()));
            assert_eq!(
                plan.response_restriction,
                vec![
                    BoardMove::from_uci("e7e5").unwrap(),
                    BoardMove::from_uci("c7c5").unwrap()
                ]
            );
            assert!(plan.claimed_line.is_empty());
            assert_eq!(limits.deadline, started + Duration::from_millis(4990));
        }
    }
    #[test]
    fn short_l_two_no_claim_retains_consumer_capacities_and_matching_resources_without_engines() {
        for mode in [
            ReplayInputMode::ReplyOnly2n,
            ReplayInputMode::RepairEndpoint3n,
        ] {
            let mut fixture = fixture(mode, false, "position startpos");
            let original_prepared = fixture.request.prepared_action_raw.clone();
            let required = repair_replay_requirements(2, 1, 64).unwrap();
            fixture.request.config.line_plies = 2;
            fixture.request.config.max_role_calls = required.role_calls();
            fixture.request.resources.role_calls = required.role_calls();
            fixture.request.resources.required_observations = required.observations();
            fixture.request.resources.required_line_chunks = required.line_chunks();
            fixture.request.resources.required_stages = required.stages();
            reseal(&mut fixture);
            let checked = admitted(&fixture, Instant::now()).unwrap();
            assert_eq!(
                checked.original_prepared_action(),
                original_prepared.as_bytes()
            );
            assert_eq!(checked.resources().cpu_nodes, 64 * mode.checks());
            assert_eq!(checked.resources().role_calls, required.role_calls());
            assert_eq!(
                checked.resources().store_nodes,
                fixture.request.config.max_nodes
            );
            assert_eq!(
                checked.resources().store_records,
                fixture.request.config.max_records
            );
            assert_eq!(
                checked.resources().required_observations,
                required.observations()
            );
            assert_eq!(
                checked.resources().required_line_chunks,
                required.line_chunks()
            );
            assert_eq!(checked.resources().required_stages, required.stages());
            let audit = checked.audit();
            assert_eq!(audit.cpu_checks, 0);
            assert!(!audit.cpu_engine_created && !audit.model_created && !audit.provider_created);
            assert_eq!(audit.authorities, ReplayAuthorities::default());
            assert_eq!(audit.actual_utility_groups, 0);
            assert_eq!(audit.cpu_allowance, 64 * mode.checks());
            let (config, root, plan, limits) = checked.into_owner_args();
            assert_eq!(config.line_plies, 2);
            assert_eq!(config.max_role_calls, required.role_calls());
            assert_eq!(plan.prefix.len(), 1);
            assert!(plan.prefix.capacity() <= config.line_plies);
            assert!(plan.claimed_line.is_empty());
            assert!(plan.claimed_line.capacity() <= config.line_plies);
            for moves in [
                &plan.root_legal_order,
                &plan.target_legal_order,
                &plan.response_restriction,
            ] {
                assert!(moves.len() <= semantic::MAX_ROOT_MOVES);
                assert!(moves.capacity() <= semantic::MAX_ROOT_MOVES);
            }
            assert_eq!(plan.root_legal_order, root.legal_moves());
            assert_eq!(plan.baseline_depth, 1);
            assert_eq!(plan.requested_depth, 2);
            assert_eq!(plan.nodes_per_check, 64);
            assert_eq!(limits.max_cpu_nodes, 64 * mode.checks());
            assert_eq!(limits.cpu_depth, plan.requested_depth);
        }
    }
    #[test]
    fn target_relative_claim_is_prefixed_exactly_once_after_actual_rules_check() {
        let fixture = fixture(ReplayInputMode::RepairEndpoint3n, true, "position startpos");
        let (_, _, plan, _) = admitted(&fixture, Instant::now())
            .unwrap()
            .into_owner_args();
        assert_eq!(
            plan.claimed_line
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["e2e4", "e7e5", "g1f3"]
        );
        assert_eq!(plan.prefix.len(), 1);
    }
    #[test]
    fn raw_byte_whitespace_tamper_and_self_repin_do_not_register_originals() {
        for field in 0..3 {
            let mut fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            match field {
                0 => {
                    fixture.request.prepared_action_raw.push(' ');
                    fixture.request.prepared_action_artifact =
                        super::super::pin(fixture.request.prepared_action_raw.as_bytes());
                }
                1 => {
                    fixture.request.semantic_receipt_raw.push(' ');
                    fixture.request.semantic_receipt_artifact =
                        super::super::pin(fixture.request.semantic_receipt_raw.as_bytes());
                }
                _ => {
                    fixture.request.registration_raw.push(' ');
                    fixture.request.registration_artifact =
                        super::super::pin(fixture.request.registration_raw.as_bytes());
                }
            }
            reseal(&mut fixture);
            assert_eq!(
                admitted(&fixture, Instant::now()).unwrap_err().stage,
                "original_artifact_identity"
            );
        }
        let mut fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let mut action: super::super::Request =
            serde_json::from_str(&fixture.request.prepared_action_raw).unwrap();
        action.cpu_request_raw.push(' ');
        action.cpu_request_artifact = super::super::pin(action.cpu_request_raw.as_bytes());
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        fixture.request.prepared_action_raw = serde_json::to_string(&action).unwrap();
        fixture.request.prepared_action_artifact =
            super::super::pin(fixture.request.prepared_action_raw.as_bytes());
        fixture.expected.prepared_action_artifact =
            fixture.request.prepared_action_artifact.clone();
        reseal(&mut fixture);
        assert_eq!(
            admitted(&fixture, Instant::now()).unwrap_err().stage,
            "prepared_binding"
        );
    }
    #[test]
    fn encoding_and_query_domains_require_independent_exact_binding() {
        for field in 0..5 {
            let mut fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            match field {
                0 => fixture.request.parent.encoding_sha256 = "0".repeat(64),
                1 => fixture.request.binding.query_sha256 = "0".repeat(64),
                2 => {
                    fixture.request.binding.semantic_input_sha256 = fixture
                        .request
                        .binding
                        .semantic_branch_meaning_sha256
                        .clone()
                }
                3 => fixture.request.binding.prior_ledger_sha256 = "0".repeat(64),
                _ => {
                    fixture.request.binding.semantic_before_result_anchor_sha256 = fixture
                        .request
                        .binding
                        .before_result_artifact
                        .sha256
                        .clone()
                }
            }
            reseal(&mut fixture);
            assert_eq!(
                admitted(&fixture, Instant::now()).unwrap_err().stage,
                "caller_binding"
            );
        }
    }
    #[test]
    fn old_binary_masked_child_and_provider_sources_cannot_replace_registration() {
        for field in 0..4 {
            let mut fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            let mut registration: ReplayConsumerRegistration =
                serde_json::from_str(&fixture.request.registration_raw).unwrap();
            match field {
                0 => {
                    registration.artifacts.legacy_cpu_binary =
                        registration.artifacts.replay_binary.clone()
                }
                1 => registration.artifacts.replay_source.sha256 = "0".repeat(64),
                2 => registration.artifacts.provider_factory_source.sha256 = "0".repeat(64),
                _ => registration.legacy_cpu_max_checks = 3,
            }
            registration.context_sha256 = seal(REGISTRATION_SCHEMA, &registration);
            fixture.request.registration_raw = serde_json::to_string(&registration).unwrap();
            fixture.request.registration_artifact =
                super::super::pin(fixture.request.registration_raw.as_bytes());
            fixture.expected.registration_artifact = fixture.request.registration_artifact.clone();
            reseal(&mut fixture);
            assert_eq!(
                admitted(&fixture, Instant::now()).unwrap_err().stage,
                "registration_binding"
            );
        }
        let mut fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let mut registration: ReplayConsumerRegistration =
            serde_json::from_str(&fixture.request.registration_raw).unwrap();
        registration.artifacts.legacy_cpu_binary = registration.artifacts.replay_binary.clone();
        registration.context_sha256 = seal(REGISTRATION_SCHEMA, &registration);
        fixture.expected.registered_artifacts = registration.artifacts.clone();
        fixture.request.registration_raw = serde_json::to_string(&registration).unwrap();
        fixture.request.registration_artifact =
            super::super::pin(fixture.request.registration_raw.as_bytes());
        fixture.expected.registration_artifact = fixture.request.registration_artifact.clone();
        reseal(&mut fixture);
        let error = admitted(&fixture, Instant::now()).unwrap_err();
        assert_eq!(error.stage, "legacy_cpu_binary_profile");
        let ReplayInputSourceError::Cpu(source) = error.source_error.as_deref().unwrap() else {
            panic!("original CPU cause");
        };
        assert_eq!(source.stage, "binary_identity");
    }
    #[test]
    fn root_history_side_full_legal_order_and_restriction_fields_are_actual_rules() {
        for field in 0..4 {
            let mut fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            changed_receipt(&mut fixture, |receipt| match field {
                0 => receipt.root.known_history_positions += 1,
                1 => receipt.target.side_to_move = semantic::SideToMove::White,
                2 => receipt.target.legal_moves.swap(0, 1),
                _ => receipt
                    .root_restriction
                    .as_mut()
                    .unwrap()
                    .declared_order
                    .swap(0, 1),
            });
            assert!(admitted(&fixture, Instant::now()).is_err());
        }
    }
    #[test]
    fn fen_unknown_history_and_short_owner_claim_extent_are_rejected() {
        let fixture = fixture(
            ReplayInputMode::ReplyOnly2n,
            false,
            "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        );
        assert_eq!(
            admitted(&fixture, Instant::now()).unwrap_err().stage,
            "root_history"
        );
        let mut fixture = self::fixture(ReplayInputMode::ReplyOnly2n, true, "position startpos");
        fixture.request.config.line_plies = 2;
        let required = repair_replay_requirements(2, 1, 64).unwrap();
        fixture.request.config.max_role_calls = required.role_calls();
        fixture.request.resources.role_calls = required.role_calls();
        fixture.request.resources.required_observations = required.observations();
        fixture.request.resources.required_line_chunks = required.line_chunks();
        reseal(&mut fixture);
        assert_eq!(
            admitted(&fixture, Instant::now()).unwrap_err().stage,
            "claim_extent"
        );
    }
    #[test]
    fn three_n_must_fit_original_two_check_remaining_without_expansion() {
        let mut fixture = fixture(
            ReplayInputMode::RepairEndpoint3n,
            false,
            "position startpos",
        );
        let mut action: super::super::Request =
            serde_json::from_str(&fixture.request.prepared_action_raw).unwrap();
        action.remaining.nodes = 128;
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        fixture.request.prepared_action_raw = serde_json::to_string(&action).unwrap();
        fixture.request.prepared_action_artifact =
            super::super::pin(fixture.request.prepared_action_raw.as_bytes());
        fixture.expected.prepared_action_artifact =
            fixture.request.prepared_action_artifact.clone();
        reseal(&mut fixture);
        assert_eq!(
            admitted(&fixture, Instant::now()).unwrap_err().stage,
            "resource_reservation"
        );
        fixture.request.mode = ReplayInputMode::ReplyOnly2n;
        fixture.request.resources.cpu_nodes = 128;
        reseal(&mut fixture);
        assert!(admitted(&fixture, Instant::now()).is_ok());
    }
    #[test]
    fn finite_role_store_output_and_cleanup_reservations_are_closed() {
        for field in 0..6 {
            let mut fixture = fixture(
                ReplayInputMode::RepairEndpoint3n,
                false,
                "position startpos",
            );
            match field {
                0 => fixture.request.resources.role_calls -= 1,
                1 => fixture.request.resources.required_line_chunks -= 1,
                2 => fixture.request.resources.output_bytes = 2 * 65536,
                3 => fixture.request.resources.cpu_nodes = 128,
                4 => fixture.request.resources.cleanup_reserve_ms = 5000,
                _ => fixture.request.resources.required_stages = 2,
            }
            reseal(&mut fixture);
            assert!(admitted(&fixture, Instant::now()).is_err());
        }
    }
    #[test]
    fn typed_wire_rejects_duplicate_unknown_null_bool_fraction_and_negative() {
        let fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let raw = serde_json::to_string(&fixture.request).unwrap();
        for candidate in [
            raw.replacen('{', "{\"schema\":\"duplicate\",", 1),
            raw.replacen('{', "{\"unknown\":0,", 1),
            raw.replace("\"cpu_nodes\":128", "\"cpu_nodes\":null"),
            raw.replace("\"cpu_nodes\":128", "\"cpu_nodes\":true"),
            raw.replace("\"cpu_nodes\":128", "\"cpu_nodes\":128.5"),
            raw.replace("\"cpu_nodes\":128", "\"cpu_nodes\":-1"),
        ] {
            assert_eq!(
                check_replay_inputs(candidate.as_bytes(), &fixture.expected, Instant::now())
                    .unwrap_err()
                    .stage,
                "input_json"
            );
        }
        let mut fixture = self::fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        let mut receipt: Value =
            serde_json::from_str(&fixture.request.semantic_receipt_raw).unwrap();
        receipt["root"]
            .as_object_mut()
            .unwrap()
            .remove("terminal_reason");
        fixture.request.semantic_receipt_raw = serde_json::to_string(&receipt).unwrap();
        fixture.request.semantic_receipt_artifact =
            super::super::pin(fixture.request.semantic_receipt_raw.as_bytes());
        fixture.expected.semantic_receipt_artifact =
            fixture.request.semantic_receipt_artifact.clone();
        reseal(&mut fixture);
        assert_eq!(
            admitted(&fixture, Instant::now()).unwrap_err().stage,
            "semantic_receipt_shape"
        );
    }
    #[test]
    fn original_expired_wall_survives_bad_output_identity_and_registration() {
        let started = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();
        for field in 0..4 {
            let mut fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
            match field {
                0 => fixture.request.resources.output_bytes = 0,
                1 => fixture.request.resources.output_bytes = MAX_OUTPUT_BYTES + 1,
                2 => fixture.request.parent.encoding_sha256 = "0".repeat(64),
                _ => fixture.request.registration_artifact.sha256 = "0".repeat(64),
            }
            reseal(&mut fixture);
            let error = admitted(&fixture, started).unwrap_err();
            assert!(error.deadline_exceeded && error.original_whole_deadline_retained);
            assert_eq!(error.whole_wall_ms, Some(5000));
            assert!(error.elapsed_ms.unwrap() >= 10000);
            assert_eq!(error.output_limit, if field < 2 { 1024 } else { 3 * 65536 });
        }
    }
    #[test]
    fn authority_escalation_and_original_cpu_failures_remain_typed_failures() {
        let mut fixture = fixture(ReplayInputMode::ReplyOnly2n, false, "position startpos");
        fixture.request.authorities.query_revalidated_by_rust = true;
        reseal(&mut fixture);
        assert_eq!(
            admitted(&fixture, Instant::now()).unwrap_err().stage,
            "domain_authority"
        );
        fixture.request.authorities = ReplayAuthorities::default();
        let mut action: super::super::Request =
            serde_json::from_str(&fixture.request.prepared_action_raw).unwrap();
        action.action.requested_depth += 1;
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        fixture.request.prepared_action_raw = serde_json::to_string(&action).unwrap();
        fixture.request.prepared_action_artifact =
            super::super::pin(fixture.request.prepared_action_raw.as_bytes());
        fixture.expected.prepared_action_artifact =
            fixture.request.prepared_action_artifact.clone();
        reseal(&mut fixture);
        let error = admitted(&fixture, Instant::now()).unwrap_err();
        assert_eq!(error.stage, "prepared_action");
        let ReplayInputSourceError::Strategic(source) = error.source_error.as_deref().unwrap()
        else {
            panic!("original strategic cause");
        };
        assert_eq!(source.stage, "action_controls");
        assert!(error.original_whole_deadline_retained);
    }
}
