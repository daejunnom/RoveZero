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
//! RepairEndpoint3n separately reserves 3N. Neither mode extends old remaining
//! declarations or changes the legacy request. Cleanup is an explicit partition
//! inside the original whole wall, not a fresh clock or whole-cost observation.

use super::super::{self as cpu_task, CpuTaskError, semantic};
use super::{ArtifactPin, StrategicError, StrategicTaskKind};
use crate::engine::OwnerRegistry;
use rz_position::{BoardMove, HistoryCompleteness, HistoryOrigin, Position};
use rz_search::cpu::{CpuConfig, CpuProfile};
use rz_search::pals::engine::replay::{
    DefendResponseReplayPlan, ReplayHistoricalPins, repair_replay_requirements,
};
use rz_search::pals::engine::{PalsConfig, PalsLimits};
use semantic::{MoveMeaning, MoveTokenKind, SemanticReceipt, SemanticRequest};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const SCHEMA: &str = "rz-pals-frozen-replay-inputs/1";
pub const REGISTRATION_SCHEMA: &str = "rz-pals-frozen-replay-consumer-registration/1";
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
}
impl ReplayInputMode {
    fn checks(self) -> u64 {
        match self {
            Self::ReplyOnly2n => 2,
            Self::RepairEndpoint3n => 3,
        }
    }
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
            schema: SCHEMA,
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
) -> Result<ReplayConsumerRegistration, ReplayInputError> {
    let registration: ReplayConsumerRegistration =
        serde_json::from_str(raw).map_err(|e| ReplayInputError::new("registration_json", e))?;
    if registration.schema != REGISTRATION_SCHEMA
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
    digest_bytes(&registration.legacy_cpu_profile_sha256)?;
    digest_bytes(&registration.context_sha256)?;
    if closed_context(REGISTRATION_SCHEMA, &registration)? != registration.context_sha256 {
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
    let semantic_request = SemanticRequest {
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
    };
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
    if request.schema != SCHEMA || request.authorities != ReplayAuthorities::default() {
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
    if closed_context(SCHEMA, &request)? != request.context_sha256 {
        return Err(ReplayInputError::new(
            "context_identity",
            "new replay mode/resources/original raw context differs",
        ));
    }
    check_clock(clock)?;
    let registration = check_registration(&request.registration_raw, expected)?;
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
    let required =
        repair_replay_requirements(config.line_plies, cpu.prefix.len(), cpu.max_nodes_per_check)
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
        || (request.mode == ReplayInputMode::RepairEndpoint3n
            && resources.cpu_nodes != required.cpu_nodes())
        || resources.cpu_nodes > prepared.remaining.nodes
        || resources.whole_wall_ms > prepared.remaining.wall_ms
        || resources.output_bytes as u64 > prepared.remaining.output_bytes
        || resources.output_bytes < minimum_output
        || resources.role_calls != config.max_role_calls
        || resources.role_calls < required.role_calls()
        || resources.store_nodes != config.max_nodes
        || resources.store_nodes < required.nodes()
        || resources.store_records != config.max_records
        || resources.store_records < required.records()
        || resources.required_observations != required.observations()
        || resources.required_line_chunks != required.line_chunks()
        || resources.required_stages != required.stages()
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
    let resources = &request.resources;
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
    if started > Instant::now() {
        return Err(ReplayInputError::new(
            "original_clock",
            "caller start cannot be in the future",
        )
        .stamped(started, Some(clock), expected_scope));
    }
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

#[cfg(test)]
mod tests {
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
