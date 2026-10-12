//! Actual optional-owner evidence. Historical V4 observations omit these
//! scopes; no historical generation, relationship or synthetic cost is rewritten.
use super::*;

pub const PALS_FOLLOWUP_LIFECYCLE_V4_DOMAIN: &str = "rz-pals-followup-lifecycle-v4/1";
pub const PALS_FOLLOWUP_EXECUTION_V4_DOMAIN: &str = "rz-pals-followup-execution-v4/1";
pub const PALS_FOLLOWUP_TRACE_CAPACITY_V4: u32 = 64;
pub const PALS_FOLLOWUP_TRACE_BYTES_V4: u64 = 96 * 1024;
pub const PALS_FOLLOWUP_PARTIAL_BYTES_V4: usize = 8 * 1024;

/// These nullable fields are required inside a present actual scope. Missing
/// data must not be interpreted as a measured absence.
pub(super) fn present_option<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d)
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPhysicalIdV4 {
    pub epoch: u64,
    pub sequence: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPreparedStateV4 {
    pub owner: u64,
    pub revision: u64,
    pub semantic_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsOwnerMethodIdentityV4 {
    pub method: String,
    /// The actual compiled source getter, distinct from a semantic-name hash.
    pub implementation_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsFollowupOwnerMethodsV4 {
    pub producer_domain: String,
    pub producer_schema_version: u32,
    pub producer_implementation_sha256: String,
    pub trace_capacity: u32,
    pub trace_bytes_max: u64,
    #[serde(deserialize_with = "present_option")]
    pub repair: Option<PalsOwnerMethodIdentityV4>,
    #[serde(deserialize_with = "present_option")]
    pub paused_stack: Option<PalsOwnerMethodIdentityV4>,
    #[serde(deserialize_with = "present_option")]
    pub cold_archive: Option<PalsOwnerMethodIdentityV4>,
    #[serde(deserialize_with = "present_option")]
    pub cuda_warm: Option<PalsOwnerMethodIdentityV4>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsCaptureFailureV4 {
    CaptureIncomplete,
    CounterOverflow,
    OwnerUnknown,
    PhysicalCompletionUnknown,
    ContextMismatch,
}

/// Bounded partial source facts never confer admission or replace an Unknown
/// observation. Each metric keeps its actual known/unknown boundary.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsFollowupPartialFactsV4 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive: Option<PalsArchivePartialFactsV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_stack: Option<PalsPausedStackObservationV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_warm: Option<PalsCudaWarmPartialFactsV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair: Option<PalsRepairPartialFactsV4>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub previous_closed_archives: Vec<PalsArchivePartialFactsV4>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsArchivePartialFactsV4 {
    pub owner_id: Option<u64>,
    pub reserved_generations: Option<u64>,
    pub last_committed_generation: Option<u64>,
    pub last_verified_generation: Option<u64>,
    pub ram_released_bytes: Option<u64>,
    pub pending_commit_bytes: Option<u64>,
    pub pending_files_retained: Option<bool>,
    pub global_managed_bytes: Option<u64>,
    pub archive_write_bytes_total: Option<u64>,
    pub session_write_ledger_id: Option<u64>,
    pub session_write_bytes_max: Option<u64>,
    pub session_write_bytes_consumed: Option<u64>,
    pub loaded_closure_bytes_peak: Option<u64>,
    pub complete: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaWarmPartialFactsV4 {
    pub backend_owner_id: Option<u64>,
    pub bank_owner_id: Option<u64>,
    pub physical_event_sequence: Option<u64>,
    pub seed_event_sequence: Option<u64>,
    pub leases_admitted: Option<u64>,
    pub leases_physically_completed: Option<u64>,
    pub leases_quarantined: Option<u64>,
    pub device_bytes_peak: Option<u64>,
    pub accepted_seed_consumptions: Option<u64>,
    pub fresh_value_evaluations: Option<u64>,
    pub backend_dropped: Option<bool>,
    pub worker_joined: Option<bool>,
    pub bank_closed: Option<bool>,
    pub complete: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRepairPartialFactsV4 {
    pub repair_trace_total: Option<u64>,
    pub pending_questions_peak: Option<u32>,
    pub repair_admissions_peak: Option<u32>,
    pub complete: bool,
}

/// Projection of the independently bound producer envelope. Native carries
/// the observations/traces alongside this metadata; Arena retains them in the
/// existing endpoint slots after raw-bit and physical-context verification.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsFollowupLifecycleScopeV4 {
    pub domain: String,
    pub schema_version: u32,
    #[serde(deserialize_with = "present_option")]
    pub process_epoch: Option<u64>,
    pub game_generation: u64,
    pub capture_complete: bool,
    pub owner_shutdown_complete: bool,
    pub trace_capacity: u32,
    pub trace_bytes_max: u64,
    pub owner_methods: PalsFollowupOwnerMethodsV4,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_failure: Option<PalsCaptureFailureV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_facts: Option<PalsFollowupPartialFactsV4>,
    /// Real owners closed during startup/newgame, distinct from the final
    /// per-game owner. Their lifetime counters are never summed per go.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub previous_closed_archive_owners: Vec<PalsArchiveObservationV4>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRepairActualScopeV4 {
    pub game_generation: u64,
    pub search_attempt_sequence: u64,
    pub store_root_generation: u64,
    /// The first actually accepted Repair revision in this entered search.
    pub lineage_root_record: u64,
    #[serde(deserialize_with = "present_option")]
    pub parent_revision: Option<u64>,
    #[serde(deserialize_with = "present_option")]
    pub supersedes_revision: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRecheckModelProvenanceV4 {
    pub model_value_semantics: String,
    pub model_identity: String,
    pub encoding_identity: String,
    pub model_epoch_sha256: String,
    pub original_perspective: PalsPerspectiveV4,
    pub wdl_bits: [u32; 3],
    pub state_sha256: String,
    pub input_sha256: String,
    pub context_revision: u64,
    pub prepared_state: PalsPreparedStateV4,
    pub fresh_call_attempted: bool,
    #[serde(deserialize_with = "present_option")]
    pub native_request: Option<PalsPhysicalIdV4>,
    #[serde(deserialize_with = "present_option")]
    pub native_execution: Option<PalsPhysicalIdV4>,
    #[serde(deserialize_with = "present_option")]
    pub completed_nn_inputs: Option<u64>,
    pub physical_input_completed: bool,
    /// Only the actual Engine accept callback may set this authority bit.
    pub accepted_output: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRecheckActualScopeV4 {
    pub comparison_perspective: PalsPerspectiveV4,
    #[serde(deserialize_with = "present_option")]
    pub before: Option<PalsRecheckModelProvenanceV4>,
    #[serde(deserialize_with = "present_option")]
    pub after: Option<PalsRecheckModelProvenanceV4>,
    /// Actual Fresh request count, distinct from public+role NN input cost.
    pub fresh_requests_completed: u32,
    pub comparison_attempted: bool,
    /// Actual conditional publication, separate from an accepted role output.
    #[serde(deserialize_with = "present_option")]
    pub publication: Option<u64>,
    #[serde(deserialize_with = "present_option")]
    pub original_error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPausedStackActualScopeV4 {
    pub owner_id: u64,
    pub event_sequence: u64,
    pub bytes_current: u64,
    pub complete: bool,
    pub admission_closed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsArchiveRuntimeLimitsV4 {
    pub game_bytes_max: u64,
    pub global_bytes_max: u64,
    pub index_entries_max: u32,
    pub index_bytes_max: u64,
    pub load_bytes_max: u64,
    pub load_deadline_max_ms: u64,
    /// Record payload only; segment header is charged separately by its owner.
    pub record_payload_bytes_max: u64,
    pub max_load_pins: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsArchiveActualScopeV4 {
    pub measurement_contract: String,
    pub event_sequence: u64,
    pub root_sha256: String,
    pub repository_sha256: String,
    pub runtime_limits: PalsArchiveRuntimeLimitsV4,
    pub generation_kind: String,
    pub reserved_generations: u64,
    #[serde(deserialize_with = "present_option")]
    pub last_committed_generation: Option<u64>,
    #[serde(deserialize_with = "present_option")]
    pub last_verified_generation: Option<u64>,
    pub ram_release_method: String,
    pub ram_reclaim_events: u64,
    pub integrity_before_reclaim_events: u64,
    pub unverified_reclaim_events: u64,
    /// Retained transaction files are on disk, not an owned RAM buffer.
    pub pending_files_retained: bool,
    pub cold_index_kind: String,
    pub load_pins_max: u32,
    #[serde(deserialize_with = "present_option")]
    pub loaded_closure_bytes_peak: Option<u64>,
    #[serde(deserialize_with = "present_option")]
    pub loaded_closure_bytes_method: Option<String>,
    pub owner_generation_check_failures: u64,
    #[serde(deserialize_with = "present_option")]
    pub archive_write_bytes_total: Option<u64>,
    /// One original write budget shared by every real per-game Store owner.
    #[serde(deserialize_with = "present_option")]
    pub session_write_ledger_id: Option<u64>,
    #[serde(deserialize_with = "present_option")]
    pub session_write_bytes_max: Option<u64>,
    #[serde(deserialize_with = "present_option")]
    pub session_write_bytes_consumed: Option<u64>,
    #[serde(deserialize_with = "present_option")]
    pub global_scan_event_sequence: Option<u64>,
    pub global_scan_complete: bool,
    pub global_scan_after_last_commit: bool,
    pub global_scope_kind: String,
    pub admission_closed: bool,
    pub complete: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaWarmActualScopeV4 {
    pub measurement_contract: String,
    pub backend_owner_id: u64,
    pub bank_owner_id: u64,
    pub physical_event_sequence: u64,
    pub seed_event_sequence: u64,
    pub leases_active: u32,
    pub device_bytes_current: u64,
    pub device_bytes_retained: u64,
    pub device_bytes_scope: String,
    pub lease_accounting_scope: String,
    pub seed_accounting_scope: String,
    pub value_accounting_scope: String,
    pub rejection_scope: String,
    pub startup_leases_admitted: u64,
    pub startup_leases_completed: u64,
    pub startup_leases_quarantined: u64,
    pub value_always_fresh: bool,
    pub admission_closed: bool,
    pub bank_closed: bool,
    pub backend_dropped: bool,
    pub worker_joined: bool,
    pub complete: bool,
}

impl PalsFollowupOwnerMethodsV4 {
    pub fn validate_for(&self, p: &PalsPoliciesV4) -> Result<(), ManifestError> {
        require(
            self.producer_domain == PALS_FOLLOWUP_LIFECYCLE_V4_DOMAIN
                && self.producer_schema_version == 1
                && sha(&self.producer_implementation_sha256, 64)
                && self.trace_capacity == PALS_FOLLOWUP_TRACE_CAPACITY_V4
                && self.trace_bytes_max == PALS_FOLLOWUP_TRACE_BYTES_V4,
            "actual followup producer domain/schema/capture bounds differ",
        )?;
        for (selected, method, name) in [
            (
                p.recheck != PalsRecheckPolicyV4::Disabled,
                &self.repair,
                "engine_actual_followup_search_v4",
            ),
            (
                !matches!(p.paused_stack, PalsPausedStackPolicyV4::Disabled),
                &self.paused_stack,
                "cpu_actual_paused_frame_owner_v4",
            ),
            (
                !matches!(p.cold_archive, PalsColdArchivePolicyV4::Disabled),
                &self.cold_archive,
                "archive_actual_owner_events_v4",
            ),
            (
                !matches!(p.cuda_warm, PalsCudaWarmPolicyV4::Disabled),
                &self.cuda_warm,
                "cuda_actual_backend_and_bank_events_v4",
            ),
        ] {
            require(
                selected == method.is_some(),
                "unselected/missing actual owner method binding",
            )?;
            if let Some(m) = method {
                require(
                    m.method == name && sha(&m.implementation_sha256, 64),
                    "actual owner method/source pin differs",
                )?;
            }
        }
        Ok(())
    }
}

impl PalsFollowupLifecycleScopeV4 {
    pub(super) fn validate_for(
        &self,
        p: &PalsPoliciesV4,
        eligible: bool,
    ) -> Result<(), ManifestError> {
        require(
            self.domain == PALS_FOLLOWUP_LIFECYCLE_V4_DOMAIN
                && self.schema_version == 1
                && self.process_epoch.is_none_or(|e| e > 0)
                && self.trace_capacity == PALS_FOLLOWUP_TRACE_CAPACITY_V4
                && self.trace_bytes_max == PALS_FOLLOWUP_TRACE_BYTES_V4,
            "actual followup scope/domain/process/capture limits invalid",
        )?;
        self.owner_methods.validate_for(p)?;
        if let Some(partial) = &self.partial_facts {
            require(
                serde_json::to_vec(partial)
                    .map_err(|e| ManifestError::Integrity(e.to_string()))?
                    .len()
                    <= PALS_FOLLOWUP_PARTIAL_BYTES_V4,
                "partial owner facts exceed 8KiB",
            )?;
            require(
                partial.previous_closed_archives.len() <= 2,
                "partial closed archive owner journal exceeds its finite bound",
            )?;
        }
        require(
            self.previous_closed_archive_owners.len() <= 2,
            "actual archive owner transition journal exceeds its finite bound",
        )?;
        let mut closed_ids = BTreeSet::new();
        if let PalsColdArchivePolicyV4::Bounded(limits) = &p.cold_archive {
            for previous in &self.previous_closed_archive_owners {
                require(
                    closed_ids.insert(previous.owner_id)
                        && previous.actual_scope.is_some()
                        && previous.lifecycle == PalsArchiveLifecycleV4::Closed,
                    "previous archive owner is duplicate, synthetic or still open",
                )?;
                previous.validate_against(limits, eligible, &BTreeSet::new())?;
            }
        } else {
            require(
                self.previous_closed_archive_owners.is_empty(),
                "unselected archive has an owner transition journal",
            )?;
        }
        require(
            !eligible
                || (self.process_epoch.is_some()
                    && self.game_generation > 0
                    && self.capture_complete
                    && self.owner_shutdown_complete
                    && self.capture_failure.is_none()
                    && self.partial_facts.is_none()),
            "incomplete actual followup capture/shutdown cannot be admitted",
        )
    }
}

pub(super) fn validate_actual_endpoint_scope(
    o: &PalsEndpointReceiptV4,
    p: &PalsPoliciesV4,
    eligible: bool,
) -> Result<(), ManifestError> {
    let scope = o.followup_lifecycle.as_ref().unwrap();
    let partial = scope.partial_facts.as_ref();
    match &p.cold_archive {
        PalsColdArchivePolicyV4::Disabled => require(
            matches!(o.archive, PalsObservedV3::Unknown)
                && partial
                    .is_none_or(|p| p.archive.is_none() && p.previous_closed_archives.is_empty()),
            "unselected actual archive facts",
        )?,
        PalsColdArchivePolicyV4::Bounded(_) => {
            if let Some(a) = checked_observation(&o.archive, eligible)? {
                require(
                    a.actual_scope.is_some()
                        && scope
                            .previous_closed_archive_owners
                            .iter()
                            .all(|p| p.owner_id != a.owner_id),
                    "actual archive lost its source scope or repeats a predecessor",
                )?;
            }
        }
    }
    match &p.paused_stack {
        PalsPausedStackPolicyV4::Disabled => require(
            matches!(o.paused_stack, PalsObservedV3::Unknown)
                && partial.is_none_or(|p| p.paused_stack.is_none()),
            "unselected actual paused owner facts",
        )?,
        PalsPausedStackPolicyV4::Bounded(_) => {
            if let Some(a) = checked_observation(&o.paused_stack, eligible)? {
                require(
                    a.actual_scope.is_some(),
                    "actual paused owner lost its source scope",
                )?;
            }
        }
    }
    match &p.cuda_warm {
        PalsCudaWarmPolicyV4::Disabled => require(
            matches!(o.cuda_warm, PalsObservedV3::Unknown)
                && partial.is_none_or(|p| p.cuda_warm.is_none()),
            "unselected actual CUDA owner facts",
        )?,
        PalsCudaWarmPolicyV4::ApproxWarm(_) => {
            if let Some(a) = checked_observation(&o.cuda_warm, eligible)? {
                require(
                    a.actual_scope.is_some(),
                    "actual CUDA owner lost its source scope",
                )?;
            }
        }
    }
    require(
        p.recheck != PalsRecheckPolicyV4::Disabled
            || (o.repair_traces.is_empty()
                && matches!(o.repair_trace_total, PalsObservedV3::Unknown)
                && matches!(o.pending_questions_peak, PalsObservedV3::Unknown)
                && o.repair_admissions_peak.is_none()
                && partial.is_none_or(|p| p.repair.is_none())),
        "unselected actual Repair facts",
    )
}

/// Known actual write costs are a lower bound when one owner measurement is
/// absent. Such a partial total cannot become an Observed run output total.
pub(super) fn archive_output_accounting(
    o: &PalsEndpointReceiptV4,
    selected: bool,
) -> Result<(u64, bool), ManifestError> {
    let Some(scope) = &o.followup_lifecycle else {
        return Ok((observed(&o.archive).map_or(0, |a| a.committed_bytes), true));
    };
    if !selected {
        return Ok((0, true));
    }
    let mut owners = BTreeMap::<u64, Option<u64>>::new();
    let mut session = None;
    let mut ledger_complete = true;
    let mut ledger_consumed = Vec::new();
    let mut insert = |id: Option<u64>,
                      bytes: Option<u64>,
                      ledger: Option<(u64, u64, u64)>|
     -> Result<(), ManifestError> {
        let id = id.ok_or_else(|| {
            ManifestError::Integrity(
                "PALS V4: partial archive write has no actual owner identity".into(),
            )
        })?;
        require(
            id > 0 && owners.insert(id, bytes).is_none(),
            "archive write owner counted twice in one process scope",
        )?;
        if let Some((ledger_id, max, consumed)) = ledger {
            require(
                ledger_id > 0 && consumed <= max && bytes.is_none_or(|bytes| bytes <= consumed),
                "actual archive session ledger extent is invalid",
            )?;
            require(
                session.is_none_or(|identity| identity == (ledger_id, max)),
                "archive owners replaced the original process-session write ledger",
            )?;
            session = Some((ledger_id, max));
            ledger_consumed.push(consumed);
        } else {
            ledger_complete = false;
        }
        Ok(())
    };
    for a in &scope.previous_closed_archive_owners {
        insert(
            Some(a.owner_id),
            a.actual_scope
                .as_ref()
                .and_then(|s| s.archive_write_bytes_total),
            a.actual_scope.as_ref().and_then(archive_session_ledger),
        )?;
    }
    let mut current = false;
    let mut final_consumed = None;
    if let Some(a) = observed(&o.archive) {
        insert(
            Some(a.owner_id),
            a.actual_scope
                .as_ref()
                .and_then(|s| s.archive_write_bytes_total),
            a.actual_scope.as_ref().and_then(archive_session_ledger),
        )?;
        current = true;
        final_consumed = a
            .actual_scope
            .as_ref()
            .and_then(|s| s.session_write_bytes_consumed);
    }
    if let Some(partial) = &scope.partial_facts {
        for a in &partial.previous_closed_archives {
            insert(
                a.owner_id,
                a.archive_write_bytes_total,
                partial_session_ledger(a),
            )?;
        }
        if let Some(a) = &partial.archive {
            insert(
                a.owner_id,
                a.archive_write_bytes_total,
                partial_session_ledger(a),
            )?;
            current = true;
            final_consumed = a.session_write_bytes_consumed;
        }
    }
    let mut total = 0u64;
    let mut complete = current
        && ledger_complete
        && scope.capture_complete
        && scope.capture_failure.is_none()
        && scope
            .partial_facts
            .as_ref()
            .is_none_or(|p| p.archive.is_none() && p.previous_closed_archives.is_empty());
    for bytes in owners.values() {
        if let Some(bytes) = bytes {
            total = total.checked_add(*bytes).ok_or_else(|| {
                ManifestError::Integrity("PALS V4: actual archive output overflow".into())
            })?;
        } else {
            complete = false;
        }
    }
    if let Some(consumed) = final_consumed {
        require(
            total <= consumed && ledger_consumed.iter().all(|previous| *previous <= consumed),
            "archive session ledger consumption regressed or lost owner writes",
        )?;
        require(
            !complete || total == consumed,
            "complete archive session ledger and unique actual owner writes differ",
        )?;
    } else {
        complete = false;
    }
    Ok((total, complete))
}

fn archive_session_ledger(s: &PalsArchiveActualScopeV4) -> Option<(u64, u64, u64)> {
    Some((
        s.session_write_ledger_id?,
        s.session_write_bytes_max?,
        s.session_write_bytes_consumed?,
    ))
}

fn partial_session_ledger(s: &PalsArchivePartialFactsV4) -> Option<(u64, u64, u64)> {
    Some((
        s.session_write_ledger_id?,
        s.session_write_bytes_max?,
        s.session_write_bytes_consumed?,
    ))
}

pub(super) fn validate_actual_archive(
    a: &PalsArchiveObservationV4,
    s: &PalsArchiveActualScopeV4,
    l: &PalsColdArchiveLimitsV4,
    eligible: bool,
) -> Result<(), ManifestError> {
    let r = &s.runtime_limits;
    require(
        s.measurement_contract == "rz-pals-cold-archive-owner-accounting-v4/1"
            && sha(&s.root_sha256, 64)
            && sha(&s.repository_sha256, 64)
            && s.generation_kind == "reserved_generation_counter"
            && a.generation == s.reserved_generations
            && s.ram_release_method == "hot_unique_owned_capacity_subset"
            && s.cold_index_kind == "directory_scan"
            && s.global_scope_kind == "canonical_no_link_managed_root"
            && r.game_bytes_max == l.game_bytes_max
            && r.global_bytes_max == l.global_bytes_max
            && r.index_entries_max == l.index_entries_max
            && r.index_bytes_max == l.index_bytes_max
            && r.load_bytes_max == l.load_bytes_max
            && r.load_deadline_max_ms == l.load_deadline_max_ms
            && (1..=16 * 1024 * 1024).contains(&r.record_payload_bytes_max)
            && r.index_bytes_max <= 16 * 1024 * 1024
            && r.load_bytes_max <= 16 * 1024 * 1024
            && (1..=16).contains(&r.max_load_pins)
            && s.load_pins_max == r.max_load_pins
            && a.index_entries_peak == 0
            && a.index_bytes_peak == 0
            && a.pinned_entries_peak <= s.load_pins_max
            && s.owner_generation_check_failures == 0
            && s.unverified_reclaim_events == 0
            && s.integrity_before_reclaim_events == s.ram_reclaim_events
            && (a.ram_released_bytes == 0 || s.ram_reclaim_events > 0)
            && s.loaded_closure_bytes_peak.is_some() == s.loaded_closure_bytes_method.is_some()
            && s.loaded_closure_bytes_method
                .as_ref()
                .is_none_or(|m| m == "hot_unique_owned_capacity_subset"),
        "actual archive owner scope/limits/index/pin/integrity-before-reclaim evidence differs",
    )?;
    require(
        (a.committed_chunks == 0) == s.last_committed_generation.is_none()
            && (a.integrity_verified_chunks == 0) == s.last_verified_generation.is_none()
            && s.last_committed_generation
                .is_none_or(|g| g < s.reserved_generations)
            && s.last_verified_generation
                .is_none_or(|g| Some(g) <= s.last_committed_generation)
            && (!s.global_scan_after_last_commit || s.global_scan_complete)
            && (!s.global_scan_complete || s.global_scan_event_sequence.is_some())
            && (!s.global_scan_after_last_commit || a.global_managed_bytes >= a.committed_bytes)
            && (a.pending_commit_bytes == 0 || s.pending_files_retained),
        "actual reserved/committed/verified generations and managed scan scope conflict",
    )?;
    if let Some(written) = s.archive_write_bytes_total {
        require(
            written >= a.committed_bytes && written <= l.game_bytes_max,
            "actual archive write extent is outside its original finite output budget",
        )?;
    }
    require(
        s.session_write_ledger_id.is_none_or(|id| id > 0)
            && s.session_write_bytes_max
                .is_none_or(|max| max == l.game_bytes_max)
            && s.session_write_bytes_consumed.is_none_or(|consumed| {
                consumed <= l.game_bytes_max
                    && s.archive_write_bytes_total
                        .is_none_or(|written| written <= consumed)
            }),
        "actual archive session write ledger differs from its original finite budget",
    )?;
    require(
        !eligible
            || (s.complete
                && s.admission_closed
                && !s.pending_files_retained
                && !a.pending_buffers_retained
                && s.archive_write_bytes_total.is_some()
                && archive_session_ledger(s).is_some()
                && s.global_scan_complete
                && (a.committed_chunks == 0 || s.global_scan_after_last_commit)
                && s.last_verified_generation == s.last_committed_generation),
        "eligible archive lacks actual write/quota-scan/generation/owner-close evidence",
    )
}

pub(super) fn validate_actual_stack(
    a: &PalsPausedStackObservationV4,
    s: &PalsPausedStackActualScopeV4,
    eligible: bool,
) -> Result<(), ManifestError> {
    require(
        s.owner_id > 0
            && s.bytes_current <= a.bytes_peak
            && (a.tokens_retained == 0) == (s.bytes_current == 0)
            && (!a.owner_released
                || (s.complete
                    && s.admission_closed
                    && a.tokens_retained == 0
                    && s.bytes_current == 0)),
        "actual paused frame owner/current bytes/admission barrier inconsistent",
    )?;
    require(
        !eligible || (s.complete && s.admission_closed && a.owner_released),
        "eligible PausedStack lacks complete actual owner-close evidence",
    )
}

pub(super) fn validate_actual_warm(
    a: &PalsCudaWarmObservationV4,
    s: &PalsCudaWarmActualScopeV4,
    o: &PalsEndpointReceiptV3,
    eligible: bool,
) -> Result<(), ManifestError> {
    require(
        s.measurement_contract == "rz-pals-cuda-warm-owner-accounting-v4/1"
            && s.backend_owner_id > 0
            && s.bank_owner_id > 0
            && s.device_bytes_scope == "explicit_cuda_kv_payload_bytes"
            && s.lease_accounting_scope == "backend_owner_lifetime_including_startup"
            && s.seed_accounting_scope == "search_only"
            && s.value_accounting_scope == "search_only"
            && s.rejection_scope == "selected_payload_seal_and_bind_validation"
            && s.value_always_fresh
            && a.leases_physically_completed
                .checked_add(a.leases_quarantined)
                .and_then(|n| n.checked_add(u64::from(s.leases_active)))
                == Some(a.leases_admitted)
            && s.device_bytes_current <= a.device_bytes_peak
            && s.device_bytes_retained <= s.device_bytes_current
            && s.startup_leases_completed
                .checked_add(s.startup_leases_quarantined)
                .is_some_and(|n| n <= s.startup_leases_admitted)
            && s.startup_leases_admitted <= a.leases_admitted
            && s.startup_leases_completed <= a.leases_physically_completed
            && s.startup_leases_quarantined <= a.leases_quarantined
            && a.accepted_seed_consumptions
                .checked_add(a.fresh_value_evaluations)
                .is_some_and(|n| n <= o.nn_inputs_completed)
            && a.leases_physically_completed
                .checked_sub(s.startup_leases_completed)
                .is_some_and(|n| {
                    a.accepted_seed_consumptions
                        .checked_add(a.fresh_value_evaluations)
                        .is_some_and(|search| search <= n)
                })
            && (!a.buffers_released
                || (s.complete
                    && s.leases_active == 0
                    && s.device_bytes_current == 0
                    && s.device_bytes_retained == 0
                    && s.admission_closed
                    && s.bank_closed
                    && s.backend_dropped
                    && s.worker_joined)),
        "actual CUDA backend/startup/search/seed/Fresh/fence scope conflicts",
    )?;
    require(
        !eligible
            || (s.complete
                && s.admission_closed
                && s.bank_closed
                && s.backend_dropped
                && s.worker_joined
                && a.buffers_released),
        "eligible CUDA Warm lacks actual backend Drop, bank-close and physical join",
    )
}

pub(super) fn validate_actual_recheck(
    a: &PalsRecheckObservationV4,
    s: &PalsRecheckActualScopeV4,
    e: &PalsEndpointV4,
) -> Result<(), ManifestError> {
    require(
        s.original_error
            .as_ref()
            .is_none_or(|error| !error.is_empty() && error.len() <= 4096)
            && (a.resolution == PalsRecheckResolutionV4::Unresolved
                || (s.comparison_attempted && s.original_error.is_none()))
            && s.publication.is_none_or(|id| {
                id > 0
                    && s.comparison_attempted
                    && s.original_error.is_none()
                    && a.resolution != PalsRecheckResolutionV4::Unresolved
            }),
        "actual comparison attempt/error cannot confer resolved value authority",
    )?;
    let mut requests = BTreeSet::new();
    let mut executions = BTreeSet::new();
    let mut charged = 0u64;
    for (value, provenance) in [(&a.before, &s.before), (&a.after, &s.after)] {
        if let Some(p) = provenance {
            require(
                p.model_value_semantics
                    == "rz-pals-context-wdl/1;side-to-move;restricted-model-estimate;actual-prepared-input;no-cp-calibration;no-rules-proof"
                    && p.model_identity == e.model_v2.model_identity
                    && p.encoding_identity == e.model_v2.encoding_sha256
                    && native_checkpoint(e).is_some_and(|c| c.sha256 == p.model_epoch_sha256)
                    && sha(&p.state_sha256, 64)
                    && sha(&p.input_sha256, 64)
                    && p.prepared_state.owner > 0
                    && sha(&p.prepared_state.semantic_sha256, 64)
                    && p.fresh_call_attempted,
                "actual recheck model/prepared-state namespace differs",
            )?;
            let request = p.native_request.as_ref().ok_or_else(|| {
                ManifestError::Integrity("PALS V4: actual Fresh request identity missing".into())
            })?;
            let execution = p.native_execution.as_ref().ok_or_else(|| {
                ManifestError::Integrity("PALS V4: actual Fresh execution identity missing".into())
            })?;
            let cost = p.completed_nn_inputs.ok_or_else(|| {
                ManifestError::Integrity("PALS V4: actual Fresh physical cost unknown".into())
            })?;
            require(
                request.epoch > 0
                    && request.epoch == execution.epoch
                    && request.sequence > 0
                    && execution.sequence > 0
                    && requests.insert((request.epoch, request.sequence))
                    && executions.insert((execution.epoch, execution.sequence))
                    && p.physical_input_completed
                    && (1..=2).contains(&cost),
                "actual Fresh request/execution duplicate or incomplete public+role delta",
            )?;
            charged = charged.checked_add(cost).ok_or_else(|| {
                ManifestError::Integrity("PALS V4: actual Fresh cost overflow".into())
            })?;
            if let PalsRecheckValueV4::FrozenWdl {
                input_sha256,
                context_revision,
                perspective,
                wdl,
                ..
            } = value
            {
                let mut raw = p.wdl_bits.map(f32::from_bits);
                if p.original_perspective != s.comparison_perspective {
                    raw.swap(0, 2);
                }
                require(
                    p.accepted_output
                        && input_sha256 == &p.input_sha256
                        && context_revision == &p.context_revision
                        && *perspective == s.comparison_perspective
                        && raw.map(f32::to_bits) == wdl.map(f32::to_bits),
                    "actual recheck raw WDL/perspective/input/context/accept authority differs",
                )?;
            } else {
                require(
                    matches!(value, PalsRecheckValueV4::Unknown),
                    "Rules terminal cannot inherit a model/Fresh execution",
                )?;
            }
        } else {
            require(
                !matches!(value, PalsRecheckValueV4::FrozenWdl { .. }),
                "actual Frozen WDL has no physical model provenance",
            )?;
            if let PalsRecheckValueV4::RulesTerminal { perspective, .. } = value {
                require(
                    *perspective == s.comparison_perspective,
                    "Rules terminal comparison perspective differs",
                )?;
            }
        }
    }
    require(
        a.fresh_nn_inputs_charged == charged
            && s.fresh_requests_completed as usize == requests.len(),
        "actual Fresh requests and physical NN inputs are distinct accounting units",
    )
}

pub(super) fn validate_actual_traces(
    o: &PalsEndpointReceiptV4,
    p: &PalsEndpointV4,
    eligible: bool,
    failures: &BTreeSet<PalsRunFailureV3>,
) -> Result<(), ManifestError> {
    let capture = o.followup_lifecycle.as_ref().ok_or_else(|| {
        ManifestError::Integrity("PALS V4: actual trace has no lifecycle scope".into())
    })?;
    capture.validate_for(&p.policies, eligible)?;
    require(
        o.repair_traces.len() <= capture.trace_capacity as usize
            && serde_json::to_vec(&o.repair_traces)
                .map_err(|e| ManifestError::Integrity(e.to_string()))?
                .len()
                <= capture.trace_bytes_max as usize,
        "actual trace recorder capacity exceeded",
    )?;
    let active = p.policies.recheck == PalsRecheckPolicyV4::FrozenWdl;
    if let Some(total) = checked_observation(&o.repair_trace_total, eligible && active)? {
        require(
            *total >= o.repair_traces.len() as u64
                && (!eligible || *total == o.repair_traces.len() as u64)
                && (active || *total == 0),
            "actual trace completeness/disabled selection differs",
        )?;
    }
    let iterative = match &p.policies.iterative_repair {
        PalsIterativeRepairPolicyV4::Disabled => None,
        PalsIterativeRepairPolicyV4::Bounded(l) => Some(l),
    };
    if let Some(l) = iterative {
        if let Some(peak) = checked_observation(&o.pending_questions_peak, eligible)? {
            resource_failure(
                *peak <= l.max_pending_questions,
                eligible,
                failures,
                "actual pending queue exceeds finite selector",
            )?;
        }
        let peak = o.repair_admissions_peak.as_ref().ok_or_else(|| {
            ManifestError::Integrity("PALS V4: iterative admission peak absent".into())
        })?;
        if let Some(peak) = checked_observation(peak, eligible)? {
            resource_failure(
                *peak <= l.max_repairs_per_first_move,
                eligible,
                failures,
                "actual failed+accepted Repair admissions exceed selector",
            )?;
        }
    } else {
        require(
            o.repair_admissions_peak.is_none(),
            "unselected iterative admission metric",
        )?;
    }
    require(
        active || o.repair_traces.is_empty(),
        "disabled recheck emitted actual traces",
    )?;
    let mut records = BTreeSet::new();
    let mut questions = BTreeSet::new();
    let mut last = BTreeMap::<(u64, u64, u64, &str), &PalsRepairTraceV4>::new();
    let mut physical = BTreeSet::new();
    let mut requests = BTreeSet::new();
    let mut charged = 0u64;
    for t in &o.repair_traces {
        let s = t.actual_scope.as_ref().ok_or_else(|| {
            ManifestError::Integrity("PALS V4: actual/synthetic trace scope mixed".into())
        })?;
        let scope = (
            s.game_generation,
            s.search_attempt_sequence,
            s.store_root_generation,
        );
        require(
            s.game_generation == capture.game_generation
                && s.search_attempt_sequence > 0
                && t.root_generation == s.store_root_generation
                && t.record_id > 0
                && t.original_record > 0
                && t.record_id != t.original_record
                && s.parent_revision == Some(t.original_record)
                && s.supersedes_revision == t.supersedes
                && t.supersedes != Some(t.record_id)
                && first_move(&t.first_move)
                && sha(&t.question_sha256, 64)
                && sha(&t.evidence_sha256, 64)
                && t.repair_ordinal > 0
                && iterative.is_none_or(|l| t.repair_ordinal <= l.max_repairs_per_first_move)
                && records.insert((s.game_generation, t.record_id))
                && questions.insert((scope, t.question_sha256.as_str(), t.evidence_revision)),
            "actual Repair scope/record/parent/supersedes/question identity invalid",
        )?;
        let key = (scope.0, scope.1, scope.2, t.first_move.as_str());
        if let Some(previous) = last.get(&key) {
            require(
                t.supersedes == Some(previous.record_id)
                    && t.repair_ordinal == previous.repair_ordinal.checked_add(1).unwrap_or(0)
                    && s.lineage_root_record
                        == previous.actual_scope.as_ref().unwrap().lineage_root_record
                    && t.evidence_revision > previous.evidence_revision,
                "actual entered-search Repair lineage did not supersede its latest accepted record",
            )?;
        } else {
            require(
                t.repair_ordinal == 1 && s.lineage_root_record == t.record_id,
                "actual first accepted Repair must anchor its entered-search lineage",
            )?;
        }
        require(
            iterative.is_some() || t.iterative_repair_comparison.is_none(),
            "unselected iterative Repair admission comparison",
        )?;
        require(
            !eligible
                || iterative.is_none()
                || t.repair_ordinal == 1
                || t.iterative_repair_comparison.is_some(),
            "later accepted iterative Repair lost its strict admission comparison",
        )?;
        for (admission, comparison) in t
            .iterative_repair_comparison
            .iter()
            .map(|c| (true, c))
            .chain(std::iter::once((false, &t.recheck)))
        {
            if let Some(recheck) = checked_observation(comparison, eligible)? {
                require(
                    recheck.actual_scope.is_some(),
                    "actual Repair lost physical recheck scope",
                )?;
                recheck.validate_against(p)?;
                if recheck.actual_scope.as_ref().unwrap().publication.is_some() {
                    require(
                        recheck.resolution
                            == if admission {
                                PalsRecheckResolutionV4::AfterPreferred
                            } else {
                                PalsRecheckResolutionV4::BeforePreferred
                            },
                        "actual conditional publication has no strict frozen-value change in its stage",
                    )?;
                }
                for provenance in [
                    &recheck.actual_scope.as_ref().unwrap().before,
                    &recheck.actual_scope.as_ref().unwrap().after,
                ]
                .into_iter()
                .flatten()
                {
                    let id = provenance.native_execution.as_ref().unwrap();
                    let request = provenance.native_request.as_ref().unwrap();
                    require(
                        Some(id.epoch) == capture.process_epoch
                            && Some(request.epoch) == capture.process_epoch
                            && physical.insert((id.epoch, id.sequence))
                            && requests.insert((request.epoch, request.sequence)),
                        "actual Fresh execution belongs to another process or is charged twice",
                    )?;
                }
                charged = charged
                    .checked_add(recheck.fresh_nn_inputs_charged)
                    .ok_or_else(|| {
                        ManifestError::Integrity("PALS V4: actual trace cost overflow".into())
                    })?;
            }
        }
        last.insert(key, t);
    }
    require(
        charged <= o.base.nn_inputs_completed,
        "actual extra Fresh physical inputs absent from NN budget",
    )
}
