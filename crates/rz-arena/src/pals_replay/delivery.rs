//! Bounded, caller-owned delivery binding. This does not deserialize a native
//! capability, certify model work, or admit the projection as a next Query prior.

use super::OwnedReplayCapture;
use crate::{ArenaError, decode_json};
use rz_uci::pals_cpu_task::strategic_action::ArtifactPin;
use rz_uci::pals_cpu_task::strategic_action::native_replay::{self, query_prior};
use rz_uci::pals_cpu_task::strategic_action::replay_inputs::{
    self, ReplayAuthorities, ReplayInputMode,
};
use rz_uci::pals_cpu_task::strategic_action::replay_launch_preparation::{
    EXPECTED_TRANSPORT_V2_SCHEMA, EXPECTED_TRANSPORT_V3_SCHEMA, ExpectedTransportV3,
    MAX_EXPECTED_BYTES, NativeResultLane,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const HEADER_BYTES: usize = 512 * 1024;
const CLI_SCHEMA: &str = "rz-pals-frozen-replay-cli-delivery/1";

/// Independently reviewed source pin, never inferred from the received digest or
/// the current compiled source. The prepared /3 wire does not register this file.
#[derive(Clone, Debug)]
pub struct ReplayDeliveryRegistration {
    pub query_prior_source: ArtifactPin,
}

/// Only constructible through the actual capture owner, before its original W.
/// The raw body borrows captured stdout; no duplicate body or pending-child owner
/// is created. The small JSON projection is still UNADMITTED native output.
/// There is deliberately no Serialize/Deserialize/Clone or capability accessor.
#[derive(Debug)]
pub struct BoundReplayDelivery<'a> {
    capture: &'a OwnedReplayCapture,
    raw_native: &'a [u8],
    body_artifact: ArtifactPin,
    projection: Value,
    projection_source: ArtifactPin,
}
impl<'a> BoundReplayDelivery<'a> {
    pub fn capture(&self) -> &'a OwnedReplayCapture {
        self.capture
    }
    pub fn raw_native_bytes(&self) -> &'a [u8] {
        self.raw_native
    }
    pub fn body_artifact(&self) -> &ArtifactPin {
        &self.body_artifact
    }
    /// Byte/source/binding checks are not a native readiness or Query/2 check.
    /// The next consumer must validate native work and caller chronology itself.
    pub fn unadmitted_projection(&self) -> &Value {
        &self.projection
    }
    /// Recomputes the producer's reported readiness against the same classifier
    /// used by live native replay. Still pending independent native witness and
    /// next-Query chronology; it never manufactures CheckedNativeReplayPrior.
    pub fn check_reported_consistency(
        &self,
        cancel: &AtomicBool,
    ) -> Result<query_prior::ReportedPriorConsistency, ArenaError> {
        let check_clock = || {
            if cancel.load(Ordering::Acquire) || Instant::now() >= self.capture.bundle().deadline()
            {
                Err(ArenaError::Budget(
                    "replay consistency original window/cancellation".into(),
                ))
            } else {
                Ok(())
            }
        };
        check_clock()?;
        let payload = self
            .capture
            .bundle()
            .payloads()
            .iter()
            .find(|payload| payload.name() == "replay-expected.json")
            .ok_or_else(|| invalid("consistency expected payload missing"))?;
        let expected: ExpectedTransportV3 = decode(payload.as_bytes())?;
        let body: Value = decode(self.raw_native)?;
        let pins = expected.expectations.replay_expected.into_expected();
        let report = query_prior::check_reported_prior_consistency(
            body,
            &pins,
            &expected.expectations.cpu_fresh_profile_artifact,
            &expected.expectations.cpu_fresh_profile,
            &self.projection_source,
        )
        .map_err(|error| invalid(&format!("reported native consistency: {error:?}")))?;
        check_clock()?;
        Ok(report)
    }

    /// Checks reported line legality against this capture's original prepared
    /// Rules input. Still no native witness, Repair record/anchor selection proof,
    /// or subsequent Query capability. Refusal retains raw bytes/child custody.
    pub fn check_reported_rules_consistency(
        &self,
        cancel: &AtomicBool,
    ) -> Result<query_prior::ReportedRulesConsistency, ArenaError> {
        let report = self.check_reported_consistency(cancel)?;
        let mut originals = self
            .capture
            .bundle()
            .payloads()
            .iter()
            .filter_map(|payload| payload.prepared_request());
        let prepared = originals
            .next()
            .ok_or_else(|| invalid("actual prepared input owner missing"))?;
        if originals.next().is_some() || prepared.deadline() != self.capture.bundle().deadline() {
            return Err(invalid(
                "prepared input owner uniqueness/original clock differs",
            ));
        }
        report
            .check_original_rules(prepared, cancel)
            .map_err(|error| invalid(&format!("reported original Rules consistency: {error:?}")))
    }
}

/// Separate independent registration, never inferred from received source hashes.
#[derive(Clone, Debug)]
pub struct RepairReplayDeliveryRegistration {
    pub prior: ReplayDeliveryRegistration,
    pub repair_evidence_source: ArtifactPin,
}
#[derive(Debug)]
pub struct BoundRepairReplayDelivery<'a> {
    inner: BoundReplayDelivery<'a>,
    repair_source: ArtifactPin,
}
impl<'a> BoundRepairReplayDelivery<'a> {
    pub fn capture(&self) -> &'a OwnedReplayCapture {
        self.inner.capture()
    }
    pub fn raw_native_bytes(&self) -> &'a [u8] {
        self.inner.raw_native_bytes()
    }
    pub fn body_artifact(&self) -> &ArtifactPin {
        self.inner.body_artifact()
    }
    pub fn check_reported_consistency(
        &self,
        cancel: &AtomicBool,
    ) -> Result<query_prior::ReportedRepairConsistency, ArenaError> {
        let clock = || {
            if cancel.load(Ordering::Acquire)
                || Instant::now() >= self.capture().bundle().deadline()
            {
                Err(invalid("Repair report original W/cancellation"))
            } else {
                Ok(())
            }
        };
        clock()?;
        let payload = self
            .capture()
            .bundle()
            .payloads()
            .iter()
            .find(|p| p.name() == "replay-expected.json")
            .ok_or_else(|| invalid("Repair expected payload missing"))?;
        let expected: ExpectedTransportV3 = decode(payload.as_bytes())?;
        if expected.native_result != NativeResultLane::RepairAnchorV1 {
            return Err(invalid("Repair report lane differs"));
        }
        let pins = expected.expectations.replay_expected.into_expected();
        let report = query_prior::check_reported_repair_consistency(
            decode(self.raw_native_bytes())?,
            &pins,
            &expected.expectations.cpu_fresh_profile_artifact,
            &expected.expectations.cpu_fresh_profile,
            &self.inner.projection_source,
            &self.repair_source,
        )
        .map_err(|e| invalid(&format!("reported Repair consistency: {e:?}")))?;
        clock()?;
        Ok(report)
    }
    pub fn check_reported_rules_consistency(
        &self,
        cancel: &AtomicBool,
    ) -> Result<query_prior::ReportedRepairRulesConsistency, ArenaError> {
        let report = self.check_reported_consistency(cancel)?;
        let mut originals = self
            .capture()
            .bundle()
            .payloads()
            .iter()
            .filter_map(|p| p.prepared_request());
        let prepared = originals
            .next()
            .ok_or_else(|| invalid("actual Repair prepared owner missing"))?;
        if originals.next().is_some() || prepared.deadline() != self.capture().bundle().deadline() {
            return Err(invalid(
                "Repair prepared owner uniqueness/original W differs",
            ));
        }
        report
            .check_original_rules(prepared, cancel)
            .map_err(|e| invalid(&format!("reported Repair original Rules: {e:?}")))
    }
}
enum DeliveryRegistration<'a> {
    Prior(&'a ReplayDeliveryRegistration),
    Repair(&'a RepairReplayDeliveryRegistration),
}
impl DeliveryRegistration<'_> {
    fn prior(&self) -> &ReplayDeliveryRegistration {
        match self {
            Self::Prior(p) => p,
            Self::Repair(r) => &r.prior,
        }
    }
    fn lane(&self) -> NativeResultLane {
        match self {
            Self::Prior(_) => NativeResultLane::QueryPriorV1,
            Self::Repair(_) => NativeResultLane::RepairAnchorV1,
        }
    }
}

impl OwnedReplayCapture {
    /// Refusal leaves this capture and any child custody intact. Both parser work
    /// and postflight checks must fit the SAME original whole-action deadline.
    pub fn bind_delivery(
        &self,
        registration: &ReplayDeliveryRegistration,
        cancel: &AtomicBool,
    ) -> Result<BoundReplayDelivery<'_>, ArenaError> {
        self.bind_delivery_registered(DeliveryRegistration::Prior(registration), cancel)
    }
    pub fn bind_repair_delivery(
        &self,
        registration: &RepairReplayDeliveryRegistration,
        cancel: &AtomicBool,
    ) -> Result<BoundRepairReplayDelivery<'_>, ArenaError> {
        let inner =
            self.bind_delivery_registered(DeliveryRegistration::Repair(registration), cancel)?;
        Ok(BoundRepairReplayDelivery {
            inner,
            repair_source: registration.repair_evidence_source.clone(),
        })
    }
    fn bind_delivery_registered(
        &self,
        registration: DeliveryRegistration<'_>,
        cancel: &AtomicBool,
    ) -> Result<BoundReplayDelivery<'_>, ArenaError> {
        let check_clock = || {
            if cancel.load(Ordering::Acquire) || Instant::now() >= self.bundle().deadline() {
                Err(ArenaError::Budget(
                    "replay delivery original window/cancellation".into(),
                ))
            } else {
                Ok(())
            }
        };
        check_clock()?;
        if !self.transport_complete() {
            return Err(invalid("actual transport closure required"));
        }
        let mut payloads = self
            .bundle()
            .payloads()
            .iter()
            .filter(|payload| payload.name() == "replay-expected.json");
        let payload = payloads
            .next()
            .ok_or_else(|| invalid("expected payload missing"))?;
        if payloads.next().is_some() || payload.as_bytes().len() > MAX_EXPECTED_BYTES {
            return Err(invalid("expected payload extent/uniqueness"));
        }
        let expected: ExpectedTransportV3 = decode(payload.as_bytes())?;
        let (raw_native, body_artifact, projection) = bind_bytes_registered(
            &self.process().process().stdout,
            &expected,
            payload.artifact(),
            &registration,
        )?;
        check_clock()?;
        Ok(BoundReplayDelivery {
            capture: self,
            raw_native,
            body_artifact,
            projection,
            projection_source: registration.prior().query_prior_source.clone(),
        })
    }
}

fn invalid(reason: &str) -> ArenaError {
    ArenaError::Integrity(format!("replay delivery: {reason}"))
}
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, ArenaError> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("UTF-8"))?;
    // Existing decoder rejects duplicate keys recursively, before typed decode.
    decode_json(text).map_err(|_| invalid("closed JSON/duplicate key"))
}
fn artifact(bytes: &[u8]) -> ArtifactPin {
    ArtifactPin {
        bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(bytes)),
    }
}

/// Exact producer envelope, preserving the body slice used for its original
/// artifact hash. Parsing into Value and serializing again would change the pin.
fn envelope(bytes: &[u8]) -> Result<(&[u8], &[u8]), ArenaError> {
    let rest = bytes
        .strip_prefix(b"{\"cli\":")
        .ok_or_else(|| invalid("envelope prefix"))?;
    let mut values =
        serde_json::Deserializer::from_slice(rest).into_iter::<serde::de::IgnoredAny>();
    values
        .next()
        .ok_or_else(|| invalid("missing CLI value"))?
        .map_err(|_| invalid("CLI syntax"))?;
    let header_end = values.byte_offset();
    if header_end > HEADER_BYTES {
        return Err(invalid("CLI byte bound"));
    }
    let tail = rest[header_end..]
        .strip_prefix(b",\"native\":")
        .ok_or_else(|| invalid("native delimiter"))?;
    let mut values =
        serde_json::Deserializer::from_slice(tail).into_iter::<serde::de::IgnoredAny>();
    values
        .next()
        .ok_or_else(|| invalid("missing native value"))?
        .map_err(|_| invalid("native syntax"))?;
    let body_end = values.byte_offset();
    if &tail[body_end..] != b"}\n" {
        return Err(invalid("envelope trailing bytes"));
    }
    Ok((&rest[..header_end], &tail[..body_end]))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    schema: String,
    scope: String,
    expected_transport: ArtifactPin,
    expected_semantic_receipt_producer_scope: Value,
    requested_native_result: NativeResultLane,
    launch_artifact: ArtifactPin,
    request_artifact: ArtifactPin,
    replay_binary: ArtifactPin,
    replay_binary_pin_scope: String,
    runtime_library: ArtifactPin,
    runtime_pin_scope: String,
    body_artifact: ArtifactPin,
    body_kind: String,
    typed_native_error_retained: bool,
    typed_input_error_retained: bool,
    whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    prepared_elapsed_ms: u64,
    execution_deadline_exceeded_at_preparation: bool,
    #[serde(deserialize_with = "required_option")]
    stdout_delivered: Option<bool>,
    process_exit_observed: bool,
    loaded_provider_image_observed_by_cli: bool,
    physical_closure_observed_by_cli: bool,
    #[serde(deserialize_with = "required_option")]
    input_error_clock: Option<Value>,
    admitted_input_clock: AdmittedClock,
}
fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClockDeclaration {
    whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    deadline_after_start_ms: u64,
    execution_deadline_after_start_ms: u64,
    output_bytes: usize,
    output_limit: usize,
}
impl ClockDeclaration {
    fn matches(&self, expected: &ExpectedTransportV3) -> bool {
        self.whole_wall_ms == expected.whole_wall_ms
            && self.cleanup_reserve_ms == expected.cleanup_reserve_ms
            && self.deadline_after_start_ms == expected.whole_wall_ms
            && Some(self.execution_deadline_after_start_ms)
                == expected
                    .whole_wall_ms
                    .checked_sub(expected.cleanup_reserve_ms)
            && self.output_bytes == expected.output_bytes
            && self.output_limit == expected.output_bytes
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectiveClock {
    deadline_after_start_ms: u64,
    execution_deadline_after_start_ms: u64,
    output_limit: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmittedClock {
    scope: String,
    transport: ClockDeclaration,
    request: ClockDeclaration,
    effective: EffectiveClock,
    declarations_differ: bool,
}

#[cfg(test)]
fn bind_bytes<'a>(
    stdout: &'a [u8],
    expected: &ExpectedTransportV3,
    expected_artifact: &ArtifactPin,
    registration: &ReplayDeliveryRegistration,
) -> Result<(&'a [u8], ArtifactPin, Value), ArenaError> {
    bind_bytes_registered(
        stdout,
        expected,
        expected_artifact,
        &DeliveryRegistration::Prior(registration),
    )
}
fn bind_bytes_registered<'a>(
    stdout: &'a [u8],
    expected: &ExpectedTransportV3,
    expected_artifact: &ArtifactPin,
    registration: &DeliveryRegistration<'_>,
) -> Result<(&'a [u8], ArtifactPin, Value), ArenaError> {
    if expected.schema != EXPECTED_TRANSPORT_V3_SCHEMA
        || expected.expectations.schema != EXPECTED_TRANSPORT_V2_SCHEMA
        || expected.native_result != registration.lane()
        || !expected.same_original_clock()
        || expected.cleanup_reserve_ms == 0
        || expected.cleanup_reserve_ms >= expected.whole_wall_ms
        || expected.output_bytes > replay_inputs::MAX_OUTPUT_BYTES
        || stdout.len() > expected.output_bytes
        || registration.prior().query_prior_source.bytes == 0
        || registration.prior().query_prior_source.bytes > 512 * 1024
    {
        return Err(invalid("explicit /3 registration, clock or byte bound"));
    }
    let (header_bytes, raw_native) = envelope(stdout)?;
    let header: Header = decode(header_bytes)?;
    let profile = &expected.expectations.cpu_fresh_profile;
    let pins = &expected.expectations.replay_expected;
    let semantic_scope = serde_json::to_value(
        expected
            .expectations
            .expected_semantic_receipt_producer_scope,
    )
    .map_err(|_| invalid("semantic scope codec"))?;
    let clock = &header.admitted_input_clock;
    if header.schema != CLI_SCHEMA
        || header.scope != "cli_pins_and_bytes_prepared_before_delivery"
        || header.expected_transport != *expected_artifact
        || header.expected_semantic_receipt_producer_scope != semantic_scope
        || header.requested_native_result != expected.native_result
        || header.launch_artifact != expected.expectations.launch_asset_artifact
        || header.request_artifact != expected.expectations.request_artifact
        || header.replay_binary != pins.registered_artifacts.replay_binary
        || header.replay_binary_pin_scope != expected.expectations.replay_binary_pin_scope
        || header.runtime_library != profile.runtime_library
        || header.runtime_pin_scope != "verified_cpu_library_bytes_not_loaded_provider_image"
        || header.body_artifact != artifact(raw_native)
        || header.body_kind != "native_returned_ok"
        || header.typed_native_error_retained
        || header.typed_input_error_retained
        || header.whole_wall_ms != expected.whole_wall_ms
        || header.cleanup_reserve_ms != expected.cleanup_reserve_ms
        || header.prepared_elapsed_ms > expected.whole_wall_ms
        || header.stdout_delivered.is_some()
        || header.process_exit_observed
        || header.loaded_provider_image_observed_by_cli
        || header.physical_closure_observed_by_cli
        || header.input_error_clock.is_some()
        || clock.scope != "original_start_clock_limits_only"
        || clock.declarations_differ
        || !clock.transport.matches(expected)
        || !clock.request.matches(expected)
        || clock.effective.deadline_after_start_ms != expected.whole_wall_ms
        || Some(clock.effective.execution_deadline_after_start_ms)
            != expected
                .whole_wall_ms
                .checked_sub(expected.cleanup_reserve_ms)
        || clock.effective.output_limit != expected.output_bytes
    {
        return Err(invalid("CLI original bytes/pins/clock or false authority"));
    }
    // Cleanup/serialization can pass E while fitting W. Do not turn this
    // diagnostic into a new execution window, or claim work finished within E.
    let _ = header.execution_deadline_exceeded_at_preparation;
    let mut native: Value = decode(raw_native)?;
    bind_native(&native, expected, &semantic_scope)?;
    match registration {
        DeliveryRegistration::Prior(_) if native.get("repair_anchor_evidence").is_some() => {
            return Err(invalid("legacy /3 binding refuses Repair evidence"));
        }
        DeliveryRegistration::Repair(r) => {
            bind_repair_projection(
                native
                    .get("repair_anchor_evidence")
                    .ok_or_else(|| invalid("Repair evidence required"))?,
                &r.repair_evidence_source,
            )?;
        }
        DeliveryRegistration::Prior(_) => {}
    }
    let projection = native
        .as_object_mut()
        .and_then(|object| object.remove("query_prior"))
        .ok_or_else(|| invalid("structured prior required"))?;
    bind_projection(&projection, registration.prior())?;
    Ok((raw_native, header.body_artifact, projection))
}

fn same<T: serde::Serialize>(value: &Value, key: &str, expected: T) -> Result<(), ArenaError> {
    let expected = serde_json::to_value(expected).map_err(|_| invalid("binding codec"))?;
    if value.get(key) != Some(&expected) {
        return Err(invalid(key));
    }
    Ok(())
}
fn no_authority(value: &Value) -> Result<(), ArenaError> {
    let authorities: ReplayAuthorities = serde_json::from_value(
        value
            .get("authorities")
            .cloned()
            .ok_or_else(|| invalid("authorities missing"))?,
    )
    .map_err(|_| invalid("closed authorities"))?;
    if authorities != ReplayAuthorities::default() {
        return Err(invalid("authority escalation"));
    }
    Ok(())
}
fn bind_native(
    native: &Value,
    expected: &ExpectedTransportV3,
    semantic_scope: &Value,
) -> Result<(), ArenaError> {
    let mode = ReplayInputMode::RepairOpponent4n;
    same(
        native,
        "schema",
        match expected.native_result {
            NativeResultLane::QueryPriorV1 => native_replay::QUERY_PRIOR_OBSERVATION_SCHEMA,
            NativeResultLane::RepairAnchorV1 => native_replay::REPAIR_ANCHOR_OBSERVATION_SCHEMA,
        },
    )?;
    same(native, "scope", native_replay::SCOPE)?;
    same(native, "mode", mode)?;
    same(native, "status", "observations_returned")?;
    same(
        native,
        "asset_profile_artifact",
        &expected.expectations.cpu_fresh_profile_artifact,
    )?;
    same(
        native,
        "binary_pin_scope",
        replay_inputs::BINARY_DECLARATION_SCOPE,
    )?;
    same(native, "original_whole_wall_ms", expected.whole_wall_ms)?;
    same(native, "cleanup_reserve_ms", expected.cleanup_reserve_ms)?;
    no_authority(native)?;
    let audit = native
        .get("input_admission")
        .ok_or_else(|| invalid("input admission missing"))?;
    let pins = &expected.expectations.replay_expected;
    same(audit, "schema", mode.input_schema())?;
    same(audit, "scope", replay_inputs::SCOPE)?;
    same(audit, "mode", mode)?;
    same(
        audit,
        "input_artifact",
        &expected.expectations.request_artifact,
    )?;
    same(audit, "registration_artifact", &pins.registration_artifact)?;
    same(
        audit,
        "prepared_action_artifact",
        &pins.prepared_action_artifact,
    )?;
    same(
        audit,
        "semantic_receipt_artifact",
        &pins.semantic_receipt_artifact,
    )?;
    same(audit, "parent", &pins.parent)?;
    same(audit, "binding", &pins.binding)?;
    same(audit, "registered_artifacts", &pins.registered_artifacts)?;
    same(audit, "provider_factory_id", &pins.provider_factory_id)?;
    same(
        audit,
        "legacy_cpu_profile_sha256",
        &pins.legacy_cpu_profile_sha256,
    )?;
    same(
        audit,
        "expected_semantic_receipt_producer_scope",
        semantic_scope,
    )?;
    same(
        audit,
        "binary_pin_scope",
        replay_inputs::BINARY_DECLARATION_SCOPE,
    )?;
    same(audit, "whole_wall_ms", expected.whole_wall_ms)?;
    same(audit, "cleanup_reserve_ms", expected.cleanup_reserve_ms)?;
    same(audit, "actual_utility_groups", 0u8)?;
    for field in ["cpu_engine_created", "model_created", "provider_created"] {
        same(audit, field, false)?;
    }
    same(audit, "cpu_checks", 0u8)?;
    no_authority(audit)
}

fn bind_repair_projection(value: &Value, source: &ArtifactPin) -> Result<(), ArenaError> {
    closed_keys(
        value,
        &[
            "schema",
            "assurance_scope",
            "source_sha256",
            "source_bytes",
            "model_counterline",
            "repaired_line",
            "repair_record_observed",
            "repair_record_revision",
            "opponent_anchor_ply",
            "actual_utility_groups",
            "authorities",
        ],
    )?;
    same(value, "schema", query_prior::REPAIR_EVIDENCE_SCHEMA)?;
    same(value, "assurance_scope", query_prior::REPAIR_EVIDENCE_SCOPE)?;
    same(value, "source_bytes", source.bytes)?;
    if source.bytes == 0 || source.bytes > 512 * 1024 {
        return Err(invalid("independent Repair source extent"));
    }
    let digest: [u8; 32] = serde_json::from_value(value["source_sha256"].clone())
        .map_err(|_| invalid("Repair source digest shape"))?;
    if digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        != source.sha256
    {
        return Err(invalid("independent Repair source digest"));
    }
    same(value, "actual_utility_groups", 0u8)?;
    no_authority(value)?;
    for key in ["model_counterline", "repaired_line"] {
        let moves = value[key]
            .as_array()
            .ok_or_else(|| invalid("Repair move array"))?;
        if moves.len() > query_prior::MAX_LINE_MOVES {
            return Err(invalid("Repair move extent"));
        }
        for movement in moves {
            let bits = movement
                .as_u64()
                .and_then(|b| u16::try_from(b).ok())
                .ok_or_else(|| invalid("Repair packed move type"))?;
            rz_contracts::pals::Move16::try_from_bits(bits)
                .and_then(rz_contracts::pals::Move16::decode)
                .map_err(|_| invalid("Repair packed move codec"))?;
        }
    }
    Ok(())
}

fn bind_projection(
    prior: &Value,
    registration: &ReplayDeliveryRegistration,
) -> Result<(), ArenaError> {
    closed_keys(
        prior,
        &[
            "schema",
            "assurance_scope",
            "projection_source_sha256",
            "outcome",
            "readiness",
            "work_returned_elapsed_ms",
            "work_returned_within_execution_window",
            "stage_count",
            "stages",
            "opponent_anchor_ply",
            "repaired_line",
            "opponent_counterline",
            "opponent_role_steps",
            "accepted_opponent_steps",
            "endpoint_observed",
            "endpoint_rules_terminal",
            "actual_utility_groups",
            "authorities",
        ],
    )?;
    same(prior, "schema", query_prior::SCHEMA)?;
    same(prior, "assurance_scope", query_prior::SCOPE)?;
    same(prior, "actual_utility_groups", 0u8)?;
    no_authority(prior)?;
    let digest: [u8; 32] = serde_json::from_value(
        prior
            .get("projection_source_sha256")
            .cloned()
            .ok_or_else(|| invalid("projection source missing"))?,
    )
    .map_err(|_| invalid("projection source shape"))?;
    let received = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if received != registration.query_prior_source.sha256 {
        return Err(invalid("independent projection source"));
    }
    let bytes = serde_json::to_vec(prior).map_err(|_| invalid("projection extent codec"))?;
    if bytes.len() > query_prior::MAX_PROJECTED_JSON_BYTES {
        return Err(invalid("projection byte bound"));
    }
    let stages = prior
        .get("stages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("prior stages"))?;
    if stages.len() > query_prior::MAX_STAGES
        || prior.get("stage_count").and_then(Value::as_u64) != Some(stages.len() as u64)
    {
        return Err(invalid("prior stage count"));
    }
    for (field, limit) in [
        ("repaired_line", query_prior::MAX_LINE_MOVES),
        ("opponent_counterline", query_prior::MAX_LINE_MOVES),
    ] {
        bounded_moves(prior.get(field), limit)?;
    }
    for stage in stages {
        closed_keys(
            stage,
            &[
                "phase",
                "requested_depth",
                "node_budget",
                "exact_completed",
                "report_present",
                "completed_depth",
                "completion",
                "completed_iteration",
                "root_restricted",
                "reused_completed_depth",
                "report_nodes",
                "report_qnodes",
                "report_tt_hits",
                "report_elapsed_ms",
                "attempt_present",
                "attempt_nodes",
                "observation_present",
                "registered_condition_sha256",
                "task_condition_sha256",
                "pv",
            ],
        )?;
        bounded_moves(stage.get("pv"), query_prior::MAX_PV_MOVES)?;
    }
    Ok(())
}
fn closed_keys(value: &Value, keys: &[&str]) -> Result<(), ArenaError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("closed projection object"))?;
    if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {
        return Err(invalid("closed projection fields"));
    }
    Ok(())
}
fn bounded_moves(value: Option<&Value>, limit: usize) -> Result<(), ArenaError> {
    let moves = value
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("packed move array"))?;
    if moves.len() > limit
        || moves
            .iter()
            .any(|movement| movement.as_u64().is_none_or(|n| n > u16::MAX as u64))
    {
        return Err(invalid("packed move extent"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_uci::pals_cpu_task::strategic_action::replay_inputs::SemanticReceiptProducerScope;
    use serde_json::json;

    // Wire/parser fixtures only: no process, model, native witness or next Query
    // capability is constructed. Public binding still requires OwnedReplayCapture.
    fn fixture() -> (
        ExpectedTransportV3,
        ArtifactPin,
        ReplayDeliveryRegistration,
        Value,
        Value,
    ) {
        let pin = json!({"bytes": 11, "sha256": "ab".repeat(32)});
        let scope = SemanticReceiptProducerScope::LibraryDispatcherArgument;
        let parent = json!({
            "parent_input_sha256": "10".repeat(32), "current_view_sha256": "20".repeat(32),
            "frozen_admission_sha256": "30".repeat(32), "encoding_sha256": "40".repeat(32),
        });
        let binding = json!({
            "query_sha256": "11".repeat(32), "catalogue_artifact": pin,
            "before_result_artifact": pin, "prior_ledger_sha256": "12".repeat(32),
            "semantic_input_sha256": "13".repeat(32), "semantic_context_sha256": "14".repeat(32),
            "semantic_branch_meaning_sha256": "15".repeat(32),
            "semantic_before_result_anchor_sha256": "16".repeat(32), "cpu_request_artifact": pin,
        });
        let registered = json!({
            "legacy_cpu_binary": pin, "replay_binary": pin, "engine_source": pin,
            "wrapper_source": pin, "replay_source": pin, "provider_factory_source": pin,
            "opponent_recheck_source": pin,
        });
        let profile = json!({
            "schema": native_replay::ASSET_PROFILE_SCHEMA, "domain": "cpu_fresh", "provider": "cpu",
            "export_manifest": pin, "runtime_library": pin, "model_epoch": "50".repeat(32),
            "encoding_semantic_sha256": "60".repeat(32), "intra_threads": 1,
            "cache_public_memory": false, "device_public_memory": false, "context_sha256": "70".repeat(32),
        });
        let expected: ExpectedTransportV3 = serde_json::from_value(json!({
            "schema": EXPECTED_TRANSPORT_V3_SCHEMA, "native_result": "query_prior_v1",
            "whole_wall_ms": 10000, "cleanup_reserve_ms": 1000, "output_bytes": 65536,
            "expectations": {
                "schema": EXPECTED_TRANSPORT_V2_SCHEMA, "request_artifact": pin,
                "replay_expected": {
                    "registration_artifact": pin, "prepared_action_artifact": pin,
                    "semantic_receipt_artifact": pin, "parent": parent, "binding": binding,
                    "registered_artifacts": registered, "legacy_cpu_profile_sha256": "80".repeat(32),
                    "provider_factory_id": native_replay::FACTORY_ID,
                    "semantic_binary_sha256": "90".repeat(32),
                },
                "launch_asset_artifact": pin, "cpu_fresh_profile_artifact": pin,
                "cpu_fresh_profile": profile, "whole_wall_ms": 10000,
                "cleanup_reserve_ms": 1000, "output_bytes": 65536,
                "replay_binary_pin_scope": "linux_loaded_executable_inode",
                "expected_semantic_receipt_producer_scope": scope,
            }
        })).unwrap();
        let expected_artifact = artifact(&serde_json::to_vec(&expected).unwrap());
        let registration = ReplayDeliveryRegistration {
            // Test-only independent fixture choice, never a production fallback.
            query_prior_source: ArtifactPin {
                bytes: 123,
                sha256: format!("{:x}", Sha256::digest(include_bytes!("delivery.rs"))),
            },
        };
        let mut prior = serde_json::to_value(query_prior::ReplayQueryPrior::default()).unwrap();
        prior["projection_source_sha256"] = json!(<[u8; 32]>::from(Sha256::digest(
            include_bytes!("delivery.rs")
        )));
        let native = json!({
            "schema": native_replay::QUERY_PRIOR_OBSERVATION_SCHEMA, "scope": native_replay::SCOPE,
            "mode": "repair_opponent_4n", "status": "observations_returned",
            "asset_profile_artifact": pin, "binary_pin_scope": replay_inputs::BINARY_DECLARATION_SCOPE,
            "original_whole_wall_ms": 10000, "cleanup_reserve_ms": 1000,
            "authorities": ReplayAuthorities::default(),
            "input_admission": {
                "schema": ReplayInputMode::RepairOpponent4n.input_schema(), "scope": replay_inputs::SCOPE,
                "mode": "repair_opponent_4n", "input_artifact": pin,
                "registration_artifact": pin, "prepared_action_artifact": pin,
                "semantic_receipt_artifact": pin, "parent": parent, "binding": binding,
                "registered_artifacts": registered, "provider_factory_id": native_replay::FACTORY_ID,
                "legacy_cpu_profile_sha256": "80".repeat(32),
                "expected_semantic_receipt_producer_scope": scope,
                "binary_pin_scope": replay_inputs::BINARY_DECLARATION_SCOPE,
                "whole_wall_ms": 10000, "cleanup_reserve_ms": 1000,
                "actual_utility_groups": 0, "authorities": ReplayAuthorities::default(),
                "cpu_engine_created": false, "model_created": false, "provider_created": false, "cpu_checks": 0,
            },
            "query_prior": prior,
        });
        let clock = json!({
            "whole_wall_ms": 10000, "cleanup_reserve_ms": 1000,
            "deadline_after_start_ms": 10000, "execution_deadline_after_start_ms": 9000,
            "output_bytes": 65536, "output_limit": 65536,
        });
        let header = json!({
            "schema": CLI_SCHEMA, "scope": "cli_pins_and_bytes_prepared_before_delivery",
            "expected_transport": expected_artifact, "expected_semantic_receipt_producer_scope": scope,
            "requested_native_result": "query_prior_v1", "launch_artifact": pin,
            "request_artifact": pin, "replay_binary": pin,
            "replay_binary_pin_scope": "linux_loaded_executable_inode", "runtime_library": pin,
            "runtime_pin_scope": "verified_cpu_library_bytes_not_loaded_provider_image",
            "body_artifact": pin, "body_kind": "native_returned_ok",
            "typed_native_error_retained": false, "typed_input_error_retained": false,
            "whole_wall_ms": 10000, "cleanup_reserve_ms": 1000, "prepared_elapsed_ms": 9999,
            "execution_deadline_exceeded_at_preparation": true,
            "stdout_delivered": null, "process_exit_observed": false,
            "loaded_provider_image_observed_by_cli": false, "physical_closure_observed_by_cli": false,
            "input_error_clock": null,
            "admitted_input_clock": {"scope": "original_start_clock_limits_only", "transport": clock,
                "request": clock, "effective": {"deadline_after_start_ms": 10000,
                    "execution_deadline_after_start_ms": 9000, "output_limit": 65536},
                "declarations_differ": false},
        });
        (expected, expected_artifact, registration, header, native)
    }
    fn output(header: &Value, raw: &[u8]) -> Vec<u8> {
        let mut header = header.clone();
        header["body_artifact"] = serde_json::to_value(artifact(raw)).unwrap();
        let mut bytes = b"{\"cli\":".to_vec();
        bytes.extend(serde_json::to_vec(&header).unwrap());
        bytes.extend_from_slice(b",\"native\":");
        bytes.extend_from_slice(raw);
        bytes.extend_from_slice(b"}\n");
        bytes
    }
    fn repair_fixture() -> (
        ExpectedTransportV3,
        ArtifactPin,
        RepairReplayDeliveryRegistration,
        Value,
        Value,
    ) {
        let (mut expected, _, prior, mut header, mut native) = fixture();
        expected.native_result = NativeResultLane::RepairAnchorV1;
        let pin = artifact(&serde_json::to_vec(&expected).unwrap());
        header["requested_native_result"] = json!("repair_anchor_v1");
        header["expected_transport"] = serde_json::to_value(&pin).unwrap();
        native["schema"] = json!(native_replay::REPAIR_ANCHOR_OBSERVATION_SCHEMA);
        // Independently selected test fixture, not a production source fallback.
        let source = ArtifactPin {
            bytes: query_prior::repair_evidence_source_bytes(),
            sha256: query_prior::repair_evidence_source_digest()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        };
        native["repair_anchor_evidence"] =
            serde_json::to_value(query_prior::RepairAnchorEvidence::default()).unwrap();
        (
            expected,
            pin,
            RepairReplayDeliveryRegistration {
                prior,
                repair_evidence_source: source,
            },
            header,
            native,
        )
    }
    #[test]
    fn captured_repair_delivery_keeps_original_bytes_and_explicit_independent_lane() {
        let (expected, pin, registration, header, native) = repair_fixture();
        let raw = serde_json::to_vec_pretty(&native).unwrap();
        let bytes = output(&header, &raw);
        let (borrowed, body_pin, _) = bind_bytes_registered(
            &bytes,
            &expected,
            &pin,
            &DeliveryRegistration::Repair(&registration),
        )
        .unwrap();
        assert_eq!(borrowed, raw);
        assert_eq!(body_pin, artifact(&raw));
        assert!(borrowed.as_ptr() >= bytes.as_ptr());
        assert!(borrowed.as_ptr_range().end <= bytes.as_ptr_range().end);
        assert!(bind_bytes(&bytes, &expected, &pin, &registration.prior).is_err());
    }
    #[test]
    fn captured_repair_delivery_refuses_source_shape_move_extent_and_lane_substitution() {
        let (expected, pin, mut registration, header, native) = repair_fixture();
        for case in 0..5 {
            let mut changed = native.clone();
            match case {
                0 => changed["repair_anchor_evidence"]["source_bytes"] = json!(0),
                1 => changed["repair_anchor_evidence"]["extra_authority"] = json!(true),
                2 => changed["repair_anchor_evidence"]["actual_utility_groups"] = json!(1),
                3 => changed["repair_anchor_evidence"]["model_counterline"] = json!(vec![0; 17]),
                _ => changed["schema"] = json!(native_replay::QUERY_PRIOR_OBSERVATION_SCHEMA),
            }
            let raw = serde_json::to_vec(&changed).unwrap();
            assert!(
                bind_bytes_registered(
                    &output(&header, &raw),
                    &expected,
                    &pin,
                    &DeliveryRegistration::Repair(&registration)
                )
                .is_err(),
                "case {case}"
            );
        }
        registration.repair_evidence_source.sha256 = "0".repeat(64);
        let raw = serde_json::to_vec(&native).unwrap();
        assert!(
            bind_bytes_registered(
                &output(&header, &raw),
                &expected,
                &pin,
                &DeliveryRegistration::Repair(&registration)
            )
            .is_err()
        );
    }
    #[test]
    fn captured_delivery_binds_original_body_without_reserializing_or_promoting_prior() {
        let (expected, pin, registration, header, native) = fixture();
        let raw = serde_json::to_vec_pretty(&native).unwrap();
        assert_ne!(
            artifact(&raw),
            artifact(&serde_json::to_vec(&native).unwrap())
        );
        let bytes = output(&header, &raw);
        let (borrowed, body_pin, prior) =
            bind_bytes(&bytes, &expected, &pin, &registration).unwrap();
        assert_eq!(borrowed, raw);
        assert_eq!(body_pin, artifact(&raw));
        assert!(borrowed.as_ptr() >= bytes.as_ptr());
        assert!(borrowed.as_ptr_range().end <= bytes.as_ptr_range().end);
        // Successful byte binding does not promote incomplete/native JSON facts.
        assert_eq!(prior["readiness"], "unobserved");
        assert_eq!(prior["actual_utility_groups"], 0);
        assert_eq!(prior["authorities"], json!(ReplayAuthorities::default()));
    }
    #[test]
    fn captured_delivery_rejects_raw_body_tamper_duplicate_keys_and_trailing_output() {
        let (expected, pin, registration, header, native) = fixture();
        let raw = serde_json::to_vec(&native).unwrap();
        let mut bytes = output(&header, &raw);
        let at = bytes
            .windows(9)
            .position(|part| part == b"\"native\":")
            .unwrap()
            + 9;
        bytes.insert(at, b' '); // Semantically identical JSON, different original bytes.
        assert!(bind_bytes(&bytes, &expected, &pin, &registration).is_err());
        let mut duplicate = raw.clone();
        duplicate.splice(1..1, b"\"schema\":\"injected\",".iter().copied());
        assert!(bind_bytes(&output(&header, &duplicate), &expected, &pin, &registration).is_err());
        let mut duplicate_header = output(&header, &raw);
        duplicate_header.splice(8..8, b"\"schema\":\"injected\",".iter().copied());
        assert!(bind_bytes(&duplicate_header, &expected, &pin, &registration).is_err());
        for suffix in [b"{}".as_slice(), b"\n", b"garbage"] {
            let mut trailing = output(&header, &raw);
            trailing.extend_from_slice(suffix);
            assert!(bind_bytes(&trailing, &expected, &pin, &registration).is_err());
        }
    }
    #[test]
    fn captured_delivery_rejects_independent_pin_and_clock_substitution() {
        let (expected, pin, registration, header, native) = fixture();
        let raw = serde_json::to_vec(&native).unwrap();
        for field in [
            "request_artifact",
            "expected_transport",
            "launch_artifact",
            "runtime_library",
            "replay_binary",
        ] {
            let mut changed = header.clone();
            changed[field]["sha256"] = json!("ff".repeat(32));
            assert!(
                bind_bytes(&output(&changed, &raw), &expected, &pin, &registration).is_err(),
                "{field}"
            );
        }
        for field in ["whole_wall_ms", "cleanup_reserve_ms"] {
            let mut changed = header.clone();
            changed[field] = json!(99);
            assert!(bind_bytes(&output(&changed, &raw), &expected, &pin, &registration).is_err());
        }
        for part in ["transport", "request", "effective"] {
            let mut changed = header.clone();
            changed["admitted_input_clock"][part]["execution_deadline_after_start_ms"] =
                json!(10000);
            assert!(bind_bytes(&output(&changed, &raw), &expected, &pin, &registration).is_err());
        }
        let mut changed = native.clone();
        changed["input_admission"]["parent"]["current_view_sha256"] = json!("fe".repeat(32));
        // Re-pinning an altered body cannot substitute the prepared independent parent.
        assert!(
            bind_bytes(
                &output(&header, &serde_json::to_vec(&changed).unwrap()),
                &expected,
                &pin,
                &registration
            )
            .is_err()
        );
        let mut old_lane = expected.clone();
        old_lane.schema = EXPECTED_TRANSPORT_V2_SCHEMA.into();
        assert!(bind_bytes(&output(&header, &raw), &old_lane, &pin, &registration).is_err());
    }
    #[test]
    fn captured_delivery_refuses_cli_authority_claims_and_native_error_lanes() {
        let (expected, pin, registration, header, native) = fixture();
        let raw = serde_json::to_vec(&native).unwrap();
        for field in [
            "stdout_delivered",
            "process_exit_observed",
            "loaded_provider_image_observed_by_cli",
            "physical_closure_observed_by_cli",
            "typed_native_error_retained",
            "typed_input_error_retained",
        ] {
            let mut changed = header.clone();
            changed[field] = json!(true);
            assert!(
                bind_bytes(&output(&changed, &raw), &expected, &pin, &registration).is_err(),
                "{field}"
            );
        }
        for kind in [
            "native_returned_error",
            "replay_input_error",
            "declaration_only",
        ] {
            let mut changed = header.clone();
            changed["body_kind"] = json!(kind);
            assert!(bind_bytes(&output(&changed, &raw), &expected, &pin, &registration).is_err());
        }
        for path in ["", "input_admission", "query_prior"] {
            let mut changed = native.clone();
            let owner = if path.is_empty() {
                &mut changed
            } else {
                &mut changed[path]
            };
            owner["authorities"]["utility_authority"] = json!(true);
            assert!(
                bind_bytes(
                    &output(&header, &serde_json::to_vec(&changed).unwrap()),
                    &expected,
                    &pin,
                    &registration
                )
                .is_err()
            );
        }
    }
    #[test]
    fn captured_delivery_bounds_projection_and_requires_independent_projection_source() {
        let (expected, pin, registration, header, native) = fixture();
        let raw = serde_json::to_vec(&native).unwrap();
        let mut wrong = registration.clone();
        wrong.query_prior_source.sha256 = "ef".repeat(32);
        assert!(bind_bytes(&output(&header, &raw), &expected, &pin, &wrong).is_err());
        let mut changed = native.clone();
        changed["query_prior"]["repaired_line"] = json!(vec![1u16; 17]);
        assert!(
            bind_bytes(
                &output(&header, &serde_json::to_vec(&changed).unwrap()),
                &expected,
                &pin,
                &registration
            )
            .is_err()
        );
        let mut changed = native.clone();
        changed["query_prior"]["opponent_counterline"] = json!([65536]);
        assert!(
            bind_bytes(
                &output(&header, &serde_json::to_vec(&changed).unwrap()),
                &expected,
                &pin,
                &registration
            )
            .is_err()
        );
        let mut changed = native.clone();
        changed["query_prior"]["stage_count"] = json!(1);
        assert!(
            bind_bytes(
                &output(&header, &serde_json::to_vec(&changed).unwrap()),
                &expected,
                &pin,
                &registration
            )
            .is_err()
        );
        let mut changed = native.clone();
        changed["query_prior"]["unknown_capability"] = json!(true);
        assert!(
            bind_bytes(
                &output(&header, &serde_json::to_vec(&changed).unwrap()),
                &expected,
                &pin,
                &registration
            )
            .is_err()
        );
        let mut bounded = expected.clone();
        bounded.output_bytes = 10;
        bounded.expectations.output_bytes = 10;
        assert!(bind_bytes(&output(&header, &raw), &bounded, &pin, &registration).is_err());
    }
}
