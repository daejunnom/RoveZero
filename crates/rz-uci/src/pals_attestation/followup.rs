//! V4-only copies of actual owner events. Selected policy markers remain
//! independent. No capture here owns a tensor, checkpoint, pin or archive file.
use rz_search::{
    cpu::CpuPausedStackSnapshot,
    pals::{
        engine::{
            PalsExecutionEndpointSnapshot, PalsExecutionEndpointValue,
            PalsFollowupExecutionSnapshot, PalsFollowupOwnerSnapshot,
            PalsFrozenComparisonExecution, PalsRepairExecutionTrace, RoleExecutionContext,
            RoleValueExecutionEvidence,
        },
        store::{ArchiveOwnerLifecycle, ArchiveOwnerSnapshot, ArchiveRuntimeLimits},
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

pub const DOMAIN: &str = "rz-pals-followup-lifecycle-v4/1";
pub const PRODUCER_METHOD: &str = "uci_actual_followup_execution_v4";
pub const TRACE_CAPACITY: usize = 64;
pub const TRACE_BYTES_MAX: usize = 96 * 1024;
pub const PARTIAL_BYTES_MAX: usize = 8 * 1024;

/// Exact ordered source bytes compiled into this producer. These pins grant
/// neither executable identity nor authority to declare a completed owner.
pub fn compiled_followup_producer_sha256() -> [u8; 32] {
    let mut hash = Sha256::new();
    let sources = [
        (
            "pals_attestation.rs",
            include_str!("../pals_attestation.rs"),
        ),
        ("pals_attestation/followup.rs", include_str!("followup.rs")),
        ("search_driver.rs", include_str!("../search_driver.rs")),
        (
            "search_driver/work.rs",
            include_str!("../search_driver/work.rs"),
        ),
        ("pals_native.rs", include_str!("../pals_native.rs")),
        (
            "pals_native/private_warm.rs",
            include_str!("../pals_native/private_warm.rs"),
        ),
        ("main.rs", include_str!("../main.rs")),
    ];
    for bytes in std::iter::once(b"rz-pals-followup-producer-sources/1".as_slice()).chain(
        sources
            .iter()
            .flat_map(|(name, source)| [name.as_bytes(), source.as_bytes()]),
    ) {
        hash.update((bytes.len() as u64).to_be_bytes());
        hash.update(bytes);
    }
    hash.finalize().into()
}
pub fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeObservedV4<T> {
    Unknown,
    Observed { value: T, method: String },
}
impl<T> NativeObservedV4<T> {
    fn observed(value: T, method: &str) -> Self {
        Self::Observed {
            value,
            method: method.into(),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeOwnerMethodIdentityV4 {
    pub method: String,
    pub implementation_sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeFollowupOwnerMethodsV4 {
    pub producer_domain: String,
    pub producer_schema_version: u32,
    pub producer_implementation_sha256: String,
    pub trace_capacity: u32,
    pub trace_bytes_max: u64,
    pub repair: Option<NativeOwnerMethodIdentityV4>,
    pub paused_stack: Option<NativeOwnerMethodIdentityV4>,
    pub cold_archive: Option<NativeOwnerMethodIdentityV4>,
    pub cuda_warm: Option<NativeOwnerMethodIdentityV4>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsCaptureFailureV4 {
    CaptureIncomplete,
    CounterOverflow,
    OwnerUnknown,
    PhysicalCompletionUnknown,
    ContextMismatch,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsFollowupLifecycleV4 {
    pub domain: String,
    pub schema_version: u32,
    /// Native's actual process epoch. A synchronous mock has no native epoch.
    pub process_epoch: Option<u64>,
    pub game_generation: u64,
    pub capture_complete: bool,
    pub owner_shutdown_complete: bool,
    pub trace_capacity: u32,
    pub trace_bytes_max: u64,
    pub owner_methods: NativeFollowupOwnerMethodsV4,
    pub archive: NativeObservedV4<NativeArchiveObservationV4>,
    pub paused_stack: NativeObservedV4<NativePausedStackObservationV4>,
    pub cuda_warm: NativeObservedV4<NativeCudaWarmObservationV4>,
    pub repair_traces: Vec<NativeRepairTraceV4>,
    pub repair_trace_total: NativeObservedV4<u64>,
    pub pending_questions_peak: NativeObservedV4<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_admissions_peak: Option<NativeObservedV4<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_facts: Option<NativeFollowupPartialFactsV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_failure: Option<PalsCaptureFailureV4>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub previous_closed_archive_owners: Vec<NativeArchiveObservationV4>,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeFollowupPartialFactsV4 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive: Option<NativeArchiveObservationV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_stack: Option<NativePausedStackObservationV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_warm: Option<NativeCudaWarmObservationV4>,
}

/// Source counters keep nullable measurements intact. Arena can publish a
/// fully observed lane only after checking every mandatory metric and scope.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeArchiveObservationV4 {
    pub measurement_contract: String,
    pub owner_method: String,
    pub compiled_source_sha256: String,
    pub owner_id: u64,
    pub event_sequence: u64,
    pub root_sha256: Option<String>,
    pub repository_sha256: Option<String>,
    pub generation: u64,
    pub reserved_generations: u64,
    pub last_committed_generation: Option<u64>,
    pub last_verified_generation: Option<u64>,
    pub lifecycle: String,
    pub runtime_limits: Option<NativeArchiveRuntimeLimitsV4>,
    pub root_boundary_checked: bool,
    pub committed_chunks: u64,
    pub integrity_verified_chunks: u64,
    pub committed_bytes: u64,
    pub integrity_verified_bytes: u64,
    pub ram_released_bytes: Option<u64>,
    pub ram_release_method: String,
    pub ram_reclaim_events: u64,
    pub integrity_before_reclaim_events: u64,
    pub unverified_reclaim_events: u64,
    pub pending_commit_bytes: Option<u64>,
    pub pending_buffers_retained: bool,
    pub pending_files_retained: bool,
    pub global_managed_bytes: Option<u64>,
    pub global_scan_event_sequence: Option<u64>,
    pub global_scan_complete: bool,
    pub global_scan_after_last_commit: bool,
    pub global_scope_kind: String,
    pub index_entries_peak: u32,
    pub index_bytes_peak: u64,
    pub cold_index_kind: String,
    pub pinned_entries_peak: u32,
    pub load_pins_max: u32,
    pub loaded_closure_bytes_peak: Option<u64>,
    pub loaded_closure_bytes_method: Option<String>,
    pub loads_requested: u64,
    pub loads_completed: u64,
    pub load_bytes_total: u64,
    pub load_bytes_peak: u64,
    pub load_elapsed_peak_ms: u64,
    pub owner_generation_checks: u64,
    pub owner_generation_check_failures: u64,
    pub quota_failures: u64,
    pub io_failures: u64,
    pub pin_saturation_failures: u64,
    pub archive_write_bytes_total: Option<u64>,
    pub session_write_ledger_id: Option<u64>,
    pub session_write_bytes_max: Option<u64>,
    pub session_write_bytes_consumed: Option<u64>,
    pub admission_closed: bool,
    pub cleanup_complete: bool,
    pub complete: bool,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeArchiveRuntimeLimitsV4 {
    pub game_bytes_max: u64,
    pub global_bytes_max: u64,
    pub index_entries_max: u32,
    pub index_bytes_max: u64,
    pub load_bytes_max: u64,
    pub load_deadline_max_ms: u64,
    pub record_payload_bytes_max: u64,
    pub max_load_pins: u32,
}
impl From<ArchiveRuntimeLimits> for NativeArchiveRuntimeLimitsV4 {
    fn from(s: ArchiveRuntimeLimits) -> Self {
        Self {
            game_bytes_max: s.game_bytes_max,
            global_bytes_max: s.global_bytes_max,
            index_entries_max: s.index_entries_max,
            index_bytes_max: s.index_bytes_max,
            load_bytes_max: s.load_bytes_max,
            load_deadline_max_ms: s.load_deadline_max_ms,
            record_payload_bytes_max: s.record_payload_bytes_max,
            max_load_pins: s.max_load_pins,
        }
    }
}
impl From<ArchiveOwnerSnapshot> for NativeArchiveObservationV4 {
    fn from(s: ArchiveOwnerSnapshot) -> Self {
        Self {
            measurement_contract: s.measurement_contract.into(),
            owner_method: s.owner_method.into(),
            compiled_source_sha256: hex(&s.compiled_source_sha256),
            owner_id: s.owner_id,
            event_sequence: s.event_sequence,
            root_sha256: s.root_sha256.map(|v| hex(&v)),
            repository_sha256: s.repository_sha256.map(|v| hex(&v)),
            generation: s.generation,
            reserved_generations: s.reserved_generations,
            last_committed_generation: s.last_committed_generation,
            last_verified_generation: s.last_verified_generation,
            lifecycle: match s.lifecycle {
                ArchiveOwnerLifecycle::Open => "open",
                ArchiveOwnerLifecycle::Closed => "closed",
                ArchiveOwnerLifecycle::Failed => "failed",
            }
            .into(),
            runtime_limits: s.runtime_limits.map(Into::into),
            root_boundary_checked: s.root_boundary_checked,
            committed_chunks: s.committed_chunks,
            integrity_verified_chunks: s.integrity_verified_chunks,
            committed_bytes: s.committed_bytes,
            integrity_verified_bytes: s.integrity_verified_bytes,
            ram_released_bytes: s.ram_released_bytes,
            ram_release_method: s.ram_release_method.into(),
            ram_reclaim_events: s.ram_reclaim_events,
            integrity_before_reclaim_events: s.integrity_before_reclaim_events,
            unverified_reclaim_events: s.unverified_reclaim_events,
            pending_commit_bytes: s.pending_commit_bytes,
            pending_buffers_retained: s.pending_buffers_retained,
            pending_files_retained: s.pending_files_retained,
            global_managed_bytes: s.global_managed_bytes,
            global_scan_event_sequence: s.global_scan_event_sequence,
            global_scan_complete: s.global_scan_complete,
            global_scan_after_last_commit: s.global_scan_after_last_commit,
            global_scope_kind: s.global_scope_kind.into(),
            index_entries_peak: s.index_entries_peak,
            index_bytes_peak: s.index_bytes_peak,
            cold_index_kind: s.cold_index_kind.into(),
            pinned_entries_peak: s.pinned_entries_peak,
            load_pins_max: s.load_pins_max,
            loaded_closure_bytes_peak: s.loaded_closure_bytes_peak,
            loaded_closure_bytes_method: s.loaded_closure_bytes_method.map(Into::into),
            loads_requested: s.loads_requested,
            loads_completed: s.loads_completed,
            load_bytes_total: s.load_bytes_total,
            load_bytes_peak: s.load_bytes_peak,
            load_elapsed_peak_ms: s.load_elapsed_peak_ms,
            owner_generation_checks: s.owner_generation_checks,
            owner_generation_check_failures: s.owner_generation_check_failures,
            quota_failures: s.quota_failures,
            io_failures: s.io_failures,
            pin_saturation_failures: s.pin_saturation_failures,
            archive_write_bytes_total: s.archive_write_bytes_total,
            session_write_ledger_id: s.session_write_ledger_id,
            session_write_bytes_max: s.session_write_bytes_max,
            session_write_bytes_consumed: s.session_write_bytes_consumed,
            admission_closed: s.admission_closed,
            cleanup_complete: s.cleanup_complete,
            complete: s.complete,
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativePausedStackObservationV4 {
    pub owner_id: u64,
    pub event_sequence: u64,
    pub tokens_created: u64,
    pub tokens_resumed: u64,
    pub tokens_invalidated: u64,
    pub tokens_retained: u32,
    pub tokens_peak: u32,
    pub bytes_current: u64,
    pub bytes_peak: u64,
    pub stale_context_attempts: u64,
    pub stale_context_rejections: u64,
    pub replayed_consumed_work: u64,
    pub admission_closed: bool,
    pub owner_released: bool,
    pub complete: bool,
}
impl From<CpuPausedStackSnapshot> for NativePausedStackObservationV4 {
    fn from(s: CpuPausedStackSnapshot) -> Self {
        Self {
            owner_id: s.owner_id,
            event_sequence: s.event_sequence,
            tokens_created: s.tokens_created,
            tokens_resumed: s.tokens_resumed,
            tokens_invalidated: s.tokens_invalidated,
            tokens_retained: s.tokens_retained,
            tokens_peak: s.tokens_peak,
            bytes_current: s.bytes_current,
            bytes_peak: s.bytes_peak,
            stale_context_attempts: s.stale_context_attempts,
            stale_context_rejections: s.stale_context_rejections,
            replayed_consumed_work: s.replayed_consumed_work,
            admission_closed: s.admission_closed,
            owner_released: s.owner_released,
            complete: s.complete,
        }
    }
}

/// Native's bank/context/worker observations and Eval's actual CUDA owner
/// snapshot are joined only for the same execution. All bytes are explicit K/V
/// payloads; ORT staging/workspace/VRAM is outside this measurement.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaWarmObservationV4 {
    pub measurement_contract: String,
    pub backend_owner_id: u64,
    pub bank_owner_id: u64,
    pub physical_event_sequence: u64,
    pub seed_event_sequence: u64,
    pub capability_identity: String,
    pub device_identity: String,
    pub capability_available: Option<bool>,
    pub max_leases: u32,
    pub device_bytes_max: u64,
    pub leases_admitted: u64,
    pub leases_physically_completed: u64,
    pub leases_quarantined: u64,
    pub leases_active: u32,
    pub leases_peak: u32,
    pub startup_leases_admitted: u64,
    pub startup_leases_physically_completed: u64,
    pub startup_leases_quarantined: u64,
    pub device_bytes_current: u64,
    pub device_bytes_peak: u64,
    pub device_bytes_retained: u64,
    pub accepted_seed_consumptions: u64,
    pub seed_context_checked_consumptions: u64,
    pub rejected_seed_contexts: u64,
    pub fresh_value_evaluations: u64,
    pub value_always_fresh: bool,
    pub admission_closed: bool,
    pub bank_closed: bool,
    pub backend_dropped: bool,
    pub worker_joined: bool,
    pub buffers_released: bool,
    pub complete: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRoleContextV4 {
    pub game_generation: u64,
    pub search_generation: u64,
    pub situation_slot: u64,
    pub situation_generation: u64,
    pub state: u64,
    pub focus: u64,
    pub purpose: String,
    pub focus_sha256: String,
    pub prefix_sha256: String,
    pub proposal_sha256: String,
    pub refutation_sha256: Option<String>,
    pub divergence_sha256: String,
    pub public_revision: u64,
    pub situation_revision: u64,
}
impl From<RoleExecutionContext> for NativeRoleContextV4 {
    fn from(s: RoleExecutionContext) -> Self {
        Self {
            game_generation: s.game_generation,
            search_generation: s.search_generation,
            situation_slot: s.situation.slot as u64,
            situation_generation: s.situation.generation,
            state: s.state.0 as u64,
            focus: s.focus.0 as u64,
            purpose: format!("{:?}", s.purpose),
            focus_sha256: hex(&s.focus_sha256),
            prefix_sha256: hex(&s.prefix_sha256),
            proposal_sha256: hex(&s.proposal_sha256),
            refutation_sha256: s.refutation_sha256.map(|v| hex(&v)),
            divergence_sha256: hex(&s.divergence_sha256),
            public_revision: s.public_revision,
            situation_revision: s.situation_revision,
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativePhysicalIdV4 {
    pub epoch: u64,
    pub sequence: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativePreparedStateV4 {
    pub owner: u64,
    pub revision: u64,
    pub semantic_sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRoleValueProvenanceV4 {
    pub request: NativePhysicalIdV4,
    pub execution: NativePhysicalIdV4,
    pub prepared_state: NativePreparedStateV4,
    pub prepared_perspective: String,
    pub input_sha256: String,
    pub model_epoch_sha256: String,
    pub context: NativeRoleContextV4,
    pub fresh: Option<bool>,
    pub physically_completed: Option<bool>,
    pub completed_nn_inputs: Option<u64>,
    pub complete: bool,
}
impl From<RoleValueExecutionEvidence> for NativeRoleValueProvenanceV4 {
    fn from(s: RoleValueExecutionEvidence) -> Self {
        Self {
            request: NativePhysicalIdV4 {
                epoch: s.request.epoch.0,
                sequence: s.request.sequence,
            },
            execution: NativePhysicalIdV4 {
                epoch: s.execution.epoch.0,
                sequence: s.execution.sequence,
            },
            prepared_state: NativePreparedStateV4 {
                owner: s.prepared_state.owner.0,
                revision: s.prepared_state.revision.0,
                semantic_sha256: hex(&s.prepared_state.semantic.0),
            },
            prepared_perspective: match s.prepared_perspective {
                rz_contracts::Color::White => "white",
                rz_contracts::Color::Black => "black",
            }
            .into(),
            input_sha256: hex(&s.input_sha256),
            model_epoch_sha256: hex(&s.model_epoch),
            context: s.context.into(),
            fresh: s.fresh,
            physically_completed: s.physically_completed,
            completed_nn_inputs: s.completed_nn_inputs,
            complete: s.complete,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeRecheckValueV4 {
    Unknown,
    FrozenWdl {
        model_value_semantics: String,
        model_identity: String,
        encoding_identity: String,
        precision: String,
        model_epoch_sha256: String,
        input_sha256: String,
        state_sha256: String,
        perspective: String,
        wdl_bits: [u32; 3],
        context_revision: u64,
        observation: u64,
        accepted_output: bool,
        execution: Option<Box<NativeRoleValueProvenanceV4>>,
    },
    RulesTerminal {
        state_sha256: String,
        perspective: String,
        winner: Option<String>,
        reason: String,
    },
}
fn perspective(color: rz_position::Color) -> String {
    match color {
        rz_position::Color::White => "white",
        rz_position::Color::Black => "black",
    }
    .into()
}
fn endpoint(s: &PalsExecutionEndpointSnapshot) -> NativeRecheckValueV4 {
    let state_sha256 = hex(&rz_position::contracts::rules_snapshot_semantic_digest(&s.snapshot).0);
    match &s.value {
        PalsExecutionEndpointValue::Unknown => NativeRecheckValueV4::Unknown,
        PalsExecutionEndpointValue::FrozenWdl(v) => NativeRecheckValueV4::FrozenWdl {
            model_value_semantics: v.identity.semantics.clone(),
            model_identity: v.identity.model.clone(),
            encoding_identity: v.identity.encoding.clone(),
            precision: v.identity.precision.clone(),
            model_epoch_sha256: hex(&v.identity.model_epoch),
            input_sha256: hex(&v.input_sha256),
            state_sha256,
            perspective: perspective(v.perspective),
            wdl_bits: v.wdl_bits,
            context_revision: v.context_revision,
            observation: v.observation.0 as u64,
            accepted_output: v.logical_accepted,
            execution: v.execution.map(|execution| Box::new(execution.into())),
        },
        PalsExecutionEndpointValue::RulesTerminal {
            reason,
            value,
            perspective: color,
        } => NativeRecheckValueV4::RulesTerminal {
            state_sha256,
            perspective: perspective(*color),
            winner: if *value == 0 {
                None
            } else {
                Some(perspective(if *value > 0 {
                    *color
                } else {
                    color.opposite()
                }))
            },
            reason: format!("{reason:?}"),
        },
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRecheckObservationV4 {
    pub before: NativeRecheckValueV4,
    pub after: NativeRecheckValueV4,
    pub comparison_perspective: String,
    pub context_revision: Option<u64>,
    pub comparable: bool,
    pub fresh_requests_completed: Option<u32>,
    pub completed_nn_inputs: Option<u64>,
    pub publication: Option<u64>,
    pub complete: bool,
    pub comparison_attempted: bool,
    pub original_error: Option<String>,
}
impl From<&PalsFrozenComparisonExecution> for NativeRecheckObservationV4 {
    fn from(s: &PalsFrozenComparisonExecution) -> Self {
        Self {
            before: endpoint(&s.before),
            after: s
                .after
                .as_ref()
                .map_or(NativeRecheckValueV4::Unknown, endpoint),
            comparison_perspective: perspective(s.comparison_perspective),
            context_revision: s.context_revision,
            comparable: s.comparable,
            fresh_requests_completed: s.fresh_requests_completed,
            completed_nn_inputs: s.completed_nn_inputs,
            publication: s.publication.map(|id| id.0 as u64),
            complete: s.complete,
            comparison_attempted: s.comparison_attempted,
            original_error: s.original_error.as_ref().map(|e| format!("{e:?}")),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRepairRecheckV4 {
    pub reply_context: NativeRoleContextV4,
    pub prepared_accepted: bool,
    pub reply_call_attempted: bool,
    pub reply_accepted: bool,
    pub selected_response: Option<String>,
    pub counterline: Vec<String>,
    pub counterline_completed: bool,
    pub disposition: String,
    pub comparison: NativeRecheckObservationV4,
    pub original_error: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRepairTraceV4 {
    pub game_generation: u64,
    pub search_attempt_sequence: u64,
    pub store_root_generation: u64,
    pub root_situation_slot: u64,
    pub root_situation_generation: u64,
    pub root_state: u64,
    pub first_move: String,
    pub lineage_root_record: u64,
    pub parent_revision: Option<u64>,
    pub supersedes_revision: Option<u64>,
    pub record_revision: u64,
    pub repair_ordinal: u32,
    pub repaired_line: u64,
    pub repaired: Vec<String>,
    pub accepted_question: Option<NativeRoleContextV4>,
    pub iterative_repair_comparison: Option<NativeRecheckObservationV4>,
    pub recheck: Option<NativeRepairRecheckV4>,
}
impl From<&PalsRepairExecutionTrace> for NativeRepairTraceV4 {
    fn from(s: &PalsRepairExecutionTrace) -> Self {
        Self {
            game_generation: s.scope.game_generation,
            search_attempt_sequence: s.scope.search_attempt_sequence,
            store_root_generation: s.scope.store_root_generation,
            root_situation_slot: s.scope.root.slot as u64,
            root_situation_generation: s.scope.root.generation,
            root_state: s.scope.root_state.0 as u64,
            first_move: s.first_move.to_string(),
            lineage_root_record: s.lineage_root_record,
            parent_revision: s.parent_revision,
            supersedes_revision: s.supersedes_revision,
            record_revision: s.record_revision,
            repair_ordinal: s.repair_ordinal,
            repaired_line: s.repaired_line.0 as u64,
            repaired: s.repaired.iter().map(ToString::to_string).collect(),
            accepted_question: s.accepted_question.map(Into::into),
            iterative_repair_comparison: s.iterative_repair_comparison.as_ref().map(Into::into),
            recheck: s.recheck.as_ref().map(|v| NativeRepairRecheckV4 {
                reply_context: v.reply_context.into(),
                prepared_accepted: v.prepared_accepted,
                reply_call_attempted: v.reply_call_attempted,
                reply_accepted: v.reply_accepted,
                selected_response: v.selected_response.map(|mv| mv.to_string()),
                counterline: v.counterline.iter().map(ToString::to_string).collect(),
                counterline_completed: v.counterline_completed,
                disposition: format!("{:?}", v.disposition),
                comparison: (&v.comparison).into(),
                original_error: v.original_error.as_ref().map(|error| format!("{error:?}")),
            }),
        }
    }
}

/// Process-cumulative bounded journal. Copy latest owner totals, never sum a
/// per-search snapshot twice. Missing or regressed source facts stay unknown.
pub(crate) struct FollowupJournal {
    pub(crate) lifecycle: Option<PalsFollowupLifecycleV4>,
    process_epoch: Option<u64>,
    methods: Option<NativeFollowupOwnerMethodsV4>,
    last_attempt: Option<(u64, u64)>,
    trace_bytes: usize,
    output_retired: bool,
    selected_repair: bool,
    selected_iterative: bool,
    selected_archive: bool,
    selected_stack: bool,
    selected_warm: bool,
    archive_last: Option<NativeArchiveObservationV4>,
    stack_last: Option<NativePausedStackObservationV4>,
    warm_last: Option<NativeCudaWarmObservationV4>,
    archive_incomplete: bool,
    stack_incomplete: bool,
    warm_incomplete: bool,
    failure: Option<PalsCaptureFailureV4>,
}
impl Default for FollowupJournal {
    fn default() -> Self {
        Self {
            lifecycle: None,
            process_epoch: None,
            methods: None,
            last_attempt: None,
            trace_bytes: 2,
            output_retired: false,
            selected_repair: false,
            selected_iterative: false,
            selected_archive: false,
            selected_stack: false,
            selected_warm: false,
            archive_last: None,
            stack_last: None,
            warm_last: None,
            archive_incomplete: false,
            stack_incomplete: false,
            warm_incomplete: false,
            failure: None,
        }
    }
}
fn stack_progress(
    before: &NativePausedStackObservationV4,
    after: &NativePausedStackObservationV4,
) -> bool {
    before.owner_id == after.owner_id
        && before.event_sequence <= after.event_sequence
        && before.tokens_created <= after.tokens_created
        && before.tokens_resumed <= after.tokens_resumed
        && before.tokens_invalidated <= after.tokens_invalidated
        && before.tokens_peak <= after.tokens_peak
        && before.bytes_peak <= after.bytes_peak
        && before.stale_context_attempts <= after.stale_context_attempts
        && before.stale_context_rejections <= after.stale_context_rejections
        && before.replayed_consumed_work <= after.replayed_consumed_work
        && (!before.admission_closed || after.admission_closed)
        && (!before.owner_released || after.owner_released)
}
fn archive_progress(
    before: &NativeArchiveObservationV4,
    after: &NativeArchiveObservationV4,
) -> bool {
    before.owner_id == after.owner_id
        && before.measurement_contract == after.measurement_contract
        && before.owner_method == after.owner_method
        && before.event_sequence <= after.event_sequence
        && before.root_sha256 == after.root_sha256
        && before.repository_sha256 == after.repository_sha256
        && before.runtime_limits == after.runtime_limits
        && before.compiled_source_sha256 == after.compiled_source_sha256
        && before.ram_release_method == after.ram_release_method
        && before.global_scope_kind == after.global_scope_kind
        && before.cold_index_kind == after.cold_index_kind
        && before.load_pins_max == after.load_pins_max
        && before.session_write_ledger_id == after.session_write_ledger_id
        && before.session_write_bytes_max == after.session_write_bytes_max
        && before.generation <= after.generation
        && before.reserved_generations <= after.reserved_generations
        && before.committed_chunks <= after.committed_chunks
        && before.integrity_verified_chunks <= after.integrity_verified_chunks
        && before.committed_bytes <= after.committed_bytes
        && before.integrity_verified_bytes <= after.integrity_verified_bytes
        && before.ram_reclaim_events <= after.ram_reclaim_events
        && before.integrity_before_reclaim_events <= after.integrity_before_reclaim_events
        && before.unverified_reclaim_events <= after.unverified_reclaim_events
        && before.index_entries_peak <= after.index_entries_peak
        && before.index_bytes_peak <= after.index_bytes_peak
        && before.pinned_entries_peak <= after.pinned_entries_peak
        && before.loads_requested <= after.loads_requested
        && before.loads_completed <= after.loads_completed
        && before.load_bytes_total <= after.load_bytes_total
        && before.load_bytes_peak <= after.load_bytes_peak
        && before.owner_generation_checks <= after.owner_generation_checks
        && before.owner_generation_check_failures <= after.owner_generation_check_failures
        && before.quota_failures <= after.quota_failures
        && before.io_failures <= after.io_failures
        && before.pin_saturation_failures <= after.pin_saturation_failures
        && (!before.admission_closed || after.admission_closed)
        && (!before.cleanup_complete || after.cleanup_complete)
        && [
            (before.ram_released_bytes, after.ram_released_bytes),
            (
                before.archive_write_bytes_total,
                after.archive_write_bytes_total,
            ),
            (
                before.session_write_bytes_consumed,
                after.session_write_bytes_consumed,
            ),
        ]
        .iter()
        .all(|(a, b)| match (a, b) {
            (Some(a), Some(b)) => a <= b,
            (None, None) => true,
            _ => false,
        })
}
fn archive_session_progress(
    before: &NativeArchiveObservationV4,
    after: &NativeArchiveObservationV4,
) -> bool {
    before.root_sha256 == after.root_sha256
        && before.repository_sha256 == after.repository_sha256
        && before.runtime_limits == after.runtime_limits
        && before.compiled_source_sha256 == after.compiled_source_sha256
        && before.session_write_ledger_id.is_some()
        && before.session_write_ledger_id == after.session_write_ledger_id
        && before.session_write_bytes_max.is_some()
        && before.session_write_bytes_max == after.session_write_bytes_max
        && before
            .session_write_bytes_consumed
            .zip(after.session_write_bytes_consumed)
            .is_some_and(|(a, b)| {
                a <= b && after.session_write_bytes_max.is_some_and(|max| b <= max)
            })
}
fn archive_closed(value: &NativeArchiveObservationV4) -> bool {
    value.complete
        && value.lifecycle == "closed"
        && value.admission_closed
        && value.cleanup_complete
        && !value.pending_buffers_retained
        && !value.pending_files_retained
        && value.pending_commit_bytes == Some(0)
}
fn warm_progress(
    before: &NativeCudaWarmObservationV4,
    after: &NativeCudaWarmObservationV4,
) -> bool {
    before.backend_owner_id == after.backend_owner_id
        && before.bank_owner_id == after.bank_owner_id
        && before.measurement_contract == after.measurement_contract
        && before.capability_identity == after.capability_identity
        && before.device_identity == after.device_identity
        && before.max_leases == after.max_leases
        && before.device_bytes_max == after.device_bytes_max
        && before.physical_event_sequence <= after.physical_event_sequence
        && before.seed_event_sequence <= after.seed_event_sequence
        && before.leases_admitted <= after.leases_admitted
        && before.leases_physically_completed <= after.leases_physically_completed
        && before.leases_quarantined <= after.leases_quarantined
        && before.leases_peak <= after.leases_peak
        && before.startup_leases_admitted <= after.startup_leases_admitted
        && before.startup_leases_physically_completed <= after.startup_leases_physically_completed
        && before.startup_leases_quarantined <= after.startup_leases_quarantined
        && before.device_bytes_peak <= after.device_bytes_peak
        && before.accepted_seed_consumptions <= after.accepted_seed_consumptions
        && before.seed_context_checked_consumptions <= after.seed_context_checked_consumptions
        && before.rejected_seed_contexts <= after.rejected_seed_contexts
        && before.fresh_value_evaluations <= after.fresh_value_evaluations
        && (!before.admission_closed || after.admission_closed)
        && (!before.bank_closed || after.bank_closed)
        && (!before.backend_dropped || after.backend_dropped)
        && (!before.worker_joined || after.worker_joined)
        && (!before.buffers_released || after.buffers_released)
        && (before.capability_available.is_none()
            || before.capability_available == after.capability_available)
}
impl FollowupJournal {
    pub(crate) fn enable(
        &mut self,
        process_epoch: Option<u64>,
        methods: NativeFollowupOwnerMethodsV4,
        iterative: bool,
    ) -> bool {
        if self.methods.is_some() {
            return false;
        }
        self.selected_repair = methods.repair.is_some();
        self.selected_iterative = iterative;
        self.selected_archive = methods.cold_archive.is_some();
        self.selected_stack = methods.paused_stack.is_some();
        self.selected_warm = methods.cuda_warm.is_some();
        self.process_epoch = process_epoch;
        self.methods = Some(methods);
        true
    }
    pub(crate) fn enabled(&self) -> bool {
        self.methods.is_some()
    }
    pub(crate) fn requires_native_fence(&self) -> bool {
        self.process_epoch.is_some()
    }
    fn ensure(&mut self, game: Option<u64>) -> Option<&mut PalsFollowupLifecycleV4> {
        let methods = self.methods.as_ref()?;
        if self.lifecycle.is_none() {
            self.lifecycle = Some(PalsFollowupLifecycleV4 {
                domain: DOMAIN.into(),
                schema_version: 1,
                process_epoch: self.process_epoch,
                game_generation: game?,
                capture_complete: self.failure.is_none(),
                owner_shutdown_complete: false,
                trace_capacity: TRACE_CAPACITY as u32,
                trace_bytes_max: TRACE_BYTES_MAX as u64,
                owner_methods: methods.clone(),
                archive: NativeObservedV4::Unknown,
                paused_stack: NativeObservedV4::Unknown,
                cuda_warm: NativeObservedV4::Unknown,
                repair_traces: Vec::new(),
                repair_trace_total: NativeObservedV4::Unknown,
                pending_questions_peak: NativeObservedV4::Unknown,
                repair_admissions_peak: None,
                partial_facts: None,
                capture_failure: self.failure,
                previous_closed_archive_owners: Vec::new(),
            });
        }
        self.lifecycle.as_mut()
    }
    pub(crate) fn incomplete(&mut self, failure: PalsCaptureFailureV4) {
        self.failure.get_or_insert(failure);
        if let Some(l) = self.lifecycle.as_mut() {
            l.capture_complete = false;
            l.owner_shutdown_complete = false;
            l.capture_failure.get_or_insert(failure);
        }
    }
    fn preserve_closed(&mut self, source: Option<ArchiveOwnerSnapshot>) {
        let Some(value) = source.map(NativeArchiveObservationV4::from) else {
            return;
        };
        if !self.selected_archive {
            return;
        }
        let Some(l) = self.lifecycle.as_ref() else {
            return;
        };
        let prior = l
            .previous_closed_archive_owners
            .iter()
            .find(|v| v.owner_id == value.owner_id)
            .or_else(|| {
                self.archive_last
                    .as_ref()
                    .filter(|v| v.owner_id == value.owner_id)
            });
        if !archive_closed(&value) || prior.is_some_and(|p| !archive_progress(p, &value)) {
            self.incomplete(PalsCaptureFailureV4::ContextMismatch);
            return;
        }
        let l = self.lifecycle.as_mut().expect("enabled lifecycle");
        if let Some(prior) = l
            .previous_closed_archive_owners
            .iter_mut()
            .find(|v| v.owner_id == value.owner_id)
        {
            *prior = value;
        } else if l.previous_closed_archive_owners.len() < 2 {
            l.previous_closed_archive_owners.push(value);
        } else {
            self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
        }
    }
    pub(crate) fn capture_search(&mut self, s: &PalsFollowupExecutionSnapshot) {
        if self.ensure(s.game_generation).is_none() {
            if self.enabled() {
                self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
            }
            return;
        }
        let Some((game, attempt)) = s.game_generation.zip(s.search_attempt_sequence) else {
            self.incomplete(PalsCaptureFailureV4::CounterOverflow);
            return;
        };
        if self
            .last_attempt
            .is_some_and(|(g, a)| game < g || (game == g && attempt <= a))
        {
            self.incomplete(PalsCaptureFailureV4::ContextMismatch);
            return;
        }
        self.last_attempt = Some((game, attempt));
        self.output_retired = false;
        self.lifecycle
            .as_mut()
            .expect("enabled lifecycle")
            .game_generation = game;
        self.preserve_closed(s.previous_closed_archive_owner);
        if self.selected_stack
            && !s
                .paused_stack_start
                .zip(s.paused_stack_end)
                .is_some_and(|(a, b)| {
                    a.complete && b.complete && stack_progress(&a.into(), &b.into())
                })
        {
            self.stack_incomplete = true;
            self.incomplete(PalsCaptureFailureV4::ContextMismatch);
        }
        if self.selected_archive
            && !s
                .archive_start
                .as_ref()
                .ok()
                .copied()
                .flatten()
                .zip(s.archive_end.as_ref().ok().copied().flatten())
                .is_some_and(|(a, b)| archive_progress(&a.into(), &b.into()))
        {
            self.archive_incomplete = true;
            self.incomplete(PalsCaptureFailureV4::ContextMismatch);
        }
        if self.selected_repair {
            let l = self.lifecycle.as_mut().expect("enabled lifecycle");
            let total = match l.repair_trace_total {
                NativeObservedV4::Observed { value, .. } => value,
                NativeObservedV4::Unknown => 0,
            };
            if let Some(total) = total.checked_add(s.repair_trace_total) {
                l.repair_trace_total =
                    NativeObservedV4::observed(total, "engine_actual_followup_search_v4");
            } else {
                self.incomplete(PalsCaptureFailureV4::CounterOverflow);
            }
            let l = self.lifecycle.as_mut().expect("enabled lifecycle");
            if let Some(peak) = s.pending_questions_peak {
                let prior = match l.pending_questions_peak {
                    NativeObservedV4::Observed { value, .. } => value,
                    NativeObservedV4::Unknown => 0,
                };
                l.pending_questions_peak =
                    NativeObservedV4::observed(prior.max(peak), "engine_actual_followup_search_v4");
            }
            if self.selected_iterative
                && let Some(peak) = s.repair_admissions_peak
            {
                let prior = match &l.repair_admissions_peak {
                    Some(NativeObservedV4::Observed { value, .. }) => *value,
                    _ => 0,
                };
                l.repair_admissions_peak = Some(NativeObservedV4::observed(
                    prior.max(peak),
                    "engine_actual_followup_search_v4",
                ));
            }
            for trace in &s.repair_traces {
                let raw = NativeRepairTraceV4::from(trace);
                let Ok(bytes) = serde_json::to_vec(&raw) else {
                    self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
                    break;
                };
                let Some(total) = self.trace_bytes.checked_add(bytes.len() + 1) else {
                    self.incomplete(PalsCaptureFailureV4::CounterOverflow);
                    break;
                };
                if total > TRACE_BYTES_MAX
                    || self
                        .lifecycle
                        .as_ref()
                        .is_none_or(|l| l.repair_traces.len() >= TRACE_CAPACITY)
                {
                    self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
                    break;
                }
                self.trace_bytes = total;
                self.lifecycle
                    .as_mut()
                    .expect("enabled lifecycle")
                    .repair_traces
                    .push(raw);
            }
            if !s.complete
                || s.overflow
                || s.pending_questions_peak.is_none()
                || (self.selected_iterative && s.repair_admissions_peak.is_none())
            {
                self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
            }
        }
        self.capture_owner_parts(
            s.archive_end.as_ref().ok().copied().flatten(),
            s.paused_stack_end,
            true,
        );
    }
    fn capture_owner_parts(
        &mut self,
        archive: Option<ArchiveOwnerSnapshot>,
        stack: Option<CpuPausedStackSnapshot>,
        strict: bool,
    ) {
        if self.lifecycle.is_none() {
            return;
        }
        let mut archive = archive.map(NativeArchiveObservationV4::from);
        let mut stack = stack.map(NativePausedStackObservationV4::from);
        if self.selected_archive {
            let progress = archive.as_ref().is_some_and(|current| {
                self.archive_last.as_ref().is_none_or(|prior| {
                    if prior.owner_id == current.owner_id {
                        archive_progress(prior, current)
                    } else {
                        self.lifecycle.as_ref().is_some_and(|l| {
                            l.previous_closed_archive_owners.iter().any(|closed| {
                                closed.owner_id == prior.owner_id
                                    && archive_closed(closed)
                                    && archive_progress(prior, closed)
                                    && archive_session_progress(closed, current)
                            })
                        })
                    }
                })
            });
            if !progress && strict {
                self.incomplete(PalsCaptureFailureV4::ContextMismatch);
            }
            self.archive_incomplete |= !progress
                || archive.as_ref().is_none_or(|value| {
                    !value.complete
                        || value.runtime_limits.is_none()
                        || value.ram_released_bytes.is_none()
                        || value.pending_commit_bytes.is_none()
                        || value.global_managed_bytes.is_none()
                        || value.archive_write_bytes_total.is_none()
                        || value.session_write_ledger_id.is_none()
                        || value.session_write_bytes_max.is_none()
                        || value.session_write_bytes_consumed.is_none()
                });
            if progress {
                self.archive_last = archive.clone();
            }
            if self.archive_incomplete {
                if let Some(value) = archive.as_mut() {
                    value.complete = false;
                }
            }
        }
        if self.selected_stack {
            let progress = stack.as_ref().is_some_and(|current| {
                self.stack_last
                    .as_ref()
                    .is_none_or(|prior| stack_progress(prior, current))
            });
            if !progress && strict {
                self.incomplete(PalsCaptureFailureV4::ContextMismatch);
            }
            self.stack_incomplete |= !progress || stack.is_none_or(|value| !value.complete);
            if progress {
                self.stack_last = stack;
            }
            if self.stack_incomplete {
                if let Some(value) = stack.as_mut() {
                    value.complete = false;
                }
            }
        }
        let l = self.lifecycle.as_mut().expect("enabled lifecycle");
        let mut partial = l.partial_facts.take().unwrap_or_default();
        let mut missing = false;
        if self.selected_archive {
            match archive {
                Some(value)
                    if value.complete
                        && value.runtime_limits.is_some()
                        && value.ram_released_bytes.is_some()
                        && value.pending_commit_bytes.is_some()
                        && value.global_managed_bytes.is_some()
                        && value.archive_write_bytes_total.is_some()
                        && value.session_write_ledger_id.is_some()
                        && value.session_write_bytes_max.is_some()
                        && value.session_write_bytes_consumed.is_some() =>
                {
                    l.archive = NativeObservedV4::observed(value, "archive_actual_owner_events_v4");
                    partial.archive = None;
                }
                value => {
                    l.archive = NativeObservedV4::Unknown;
                    partial.archive = value;
                    missing = true;
                }
            }
        }
        if self.selected_stack {
            match stack {
                Some(value) if value.complete => {
                    l.paused_stack =
                        NativeObservedV4::observed(value, "cpu_actual_paused_frame_owner_v4");
                    partial.paused_stack = None;
                }
                value => {
                    l.paused_stack = NativeObservedV4::Unknown;
                    partial.paused_stack = value;
                    missing = true;
                }
            }
        }
        if partial.archive.is_some()
            || partial.paused_stack.is_some()
            || partial.cuda_warm.is_some()
        {
            if serde_json::to_vec(&partial).is_ok_and(|b| b.len() <= PARTIAL_BYTES_MAX) {
                l.partial_facts = Some(partial);
            } else {
                missing = true;
            }
        }
        if missing {
            self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
        }
    }
    pub(crate) fn capture_initial_owner(&mut self, s: &PalsFollowupOwnerSnapshot) {
        if self.ensure(s.game_generation).is_none() {
            if self.enabled() {
                self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
            }
            return;
        }
        self.capture_owner_parts(
            s.archive.as_ref().ok().copied().flatten(),
            s.paused_stack,
            false,
        );
    }
    pub(crate) fn capture_owner(&mut self, s: &PalsFollowupOwnerSnapshot) {
        if self.ensure(s.game_generation).is_none() {
            if self.enabled() {
                self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
            }
            return;
        }
        self.preserve_closed(s.previous_closed_archive_owner);
        self.capture_owner_parts(
            s.archive.as_ref().ok().copied().flatten(),
            s.paused_stack,
            true,
        );
        if s.archive.is_err() || s.new_game_error.is_some() {
            self.incomplete(PalsCaptureFailureV4::OwnerUnknown);
        }
    }
    pub(crate) fn capture_warm(&mut self, value: Option<NativeCudaWarmObservationV4>) {
        if !self.selected_warm {
            return;
        }
        if self.lifecycle.is_none() {
            self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
            return;
        }
        let mut value = value;
        let progress = value.as_ref().is_some_and(|current| {
            self.warm_last
                .as_ref()
                .is_none_or(|prior| warm_progress(prior, current))
        });
        if !progress {
            self.incomplete(PalsCaptureFailureV4::ContextMismatch);
        } else {
            self.warm_last = value.clone();
        }
        self.warm_incomplete |= !progress || value.as_ref().is_none_or(|value| !value.complete);
        if self.warm_incomplete
            && let Some(value) = value.as_mut()
        {
            value.complete = false;
        }
        let l = self.lifecycle.as_mut().expect("enabled lifecycle");
        match value {
            Some(value) if value.complete => {
                l.cuda_warm =
                    NativeObservedV4::observed(value, "cuda_actual_backend_and_bank_events_v4");
                if let Some(partial) = l.partial_facts.as_mut() {
                    partial.cuda_warm = None;
                }
            }
            value => {
                l.cuda_warm = NativeObservedV4::Unknown;
                let mut partial = l.partial_facts.take().unwrap_or_default();
                partial.cuda_warm = value;
                if serde_json::to_vec(&partial).is_ok_and(|b| b.len() <= PARTIAL_BYTES_MAX) {
                    l.partial_facts = Some(partial);
                }
                self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
            }
        }
    }
    pub(crate) fn output_retired(&mut self) {
        self.output_retired = true;
    }
    pub(crate) fn seal(&mut self, checker_fenced: bool, native_fenced: bool) {
        let Some(l) = self.lifecycle.as_mut() else {
            return;
        };
        let archive_closed = !self.selected_archive
            || matches!(&l.archive,NativeObservedV4::Observed{value,..} if archive_closed(value));
        let stack_closed = !self.selected_stack
            || matches!(&l.paused_stack,NativeObservedV4::Observed{value,..}
            if value.admission_closed && value.owner_released && value.tokens_retained==0 && value.bytes_current==0);
        let warm_closed = !self.selected_warm
            || matches!(&l.cuda_warm,NativeObservedV4::Observed{value,..}
            if value.admission_closed && value.bank_closed && value.backend_dropped && value.worker_joined && value.buffers_released);
        let repair_known = !self.selected_repair
            || matches!(l.repair_trace_total, NativeObservedV4::Observed { .. });
        l.owner_shutdown_complete = l.capture_complete
            && self.output_retired
            && checker_fenced
            && native_fenced
            && archive_closed
            && stack_closed
            && warm_closed
            && repair_known;
        if !l.owner_shutdown_complete {
            l.capture_failure
                .get_or_insert(PalsCaptureFailureV4::OwnerUnknown);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_search::pals::engine::{PostRepairRecheckPolicy, ResolverPolicy};

    #[test]
    fn boxed_recheck_execution_preserves_null_object_wire_and_roundtrip() {
        // Static DTO fixture only; these IDs are not actual execution evidence.
        let digest = "01".repeat(32);
        let provenance = serde_json::json!({
            "request": { "epoch": 7, "sequence": 3 },
            "execution": { "epoch": 7, "sequence": 5 },
            "prepared_state": { "owner": 13, "revision": 17, "semantic_sha256": digest },
            "prepared_perspective": "white",
            "input_sha256": digest,
            "model_epoch_sha256": digest,
            "context": {
                "game_generation": 0,
                "search_generation": 1,
                "situation_slot": 2,
                "situation_generation": 3,
                "state": 4,
                "focus": 5,
                "purpose": "ValueFresh",
                "focus_sha256": digest,
                "prefix_sha256": digest,
                "proposal_sha256": digest,
                "refutation_sha256": null,
                "divergence_sha256": digest,
                "public_revision": 6,
                "situation_revision": 7
            },
            "fresh": true,
            "physically_completed": true,
            "completed_nn_inputs": 2,
            "complete": true
        });
        for execution in [serde_json::Value::Null, provenance] {
            let wire = serde_json::json!({
                "kind": "frozen_wdl",
                "model_value_semantics": "fixture-model-wdl",
                "model_identity": "fixture-model",
                "encoding_identity": digest,
                "precision": "fp32",
                "model_epoch_sha256": digest,
                "input_sha256": digest,
                "state_sha256": digest,
                "perspective": "white",
                "wdl_bits": [0.25_f32.to_bits(), 0.5_f32.to_bits(), 0.25_f32.to_bits()],
                "context_revision": 6,
                "observation": 9,
                "accepted_output": true,
                "execution": execution
            });
            let value: NativeRecheckValueV4 = serde_json::from_value(wire.clone()).unwrap();
            assert_eq!(serde_json::to_value(value.clone()).unwrap(), wire);
            if execution.is_null() {
                let mut historical = wire.clone();
                historical.as_object_mut().unwrap().remove("execution");
                assert_eq!(
                    serde_json::from_value::<NativeRecheckValueV4>(historical).unwrap(),
                    value
                );
            } else {
                let mut unknown = wire.clone();
                unknown["execution"]["unexpected"] = serde_json::Value::Bool(true);
                assert!(serde_json::from_value::<NativeRecheckValueV4>(unknown).is_err());
            }
            let mut unknown = wire;
            unknown["unexpected"] = serde_json::Value::Bool(true);
            assert!(serde_json::from_value::<NativeRecheckValueV4>(unknown).is_err());
        }
    }

    fn journal(repair: bool, stack: bool, warm: bool, iterative: bool) -> FollowupJournal {
        let method = |name: &str| NativeOwnerMethodIdentityV4 {
            method: name.into(),
            implementation_sha256: "01".repeat(32),
        };
        let mut journal = FollowupJournal::default();
        assert!(journal.enable(
            Some(7),
            NativeFollowupOwnerMethodsV4 {
                producer_domain: DOMAIN.into(),
                producer_schema_version: 1,
                producer_implementation_sha256: "02".repeat(32),
                trace_capacity: TRACE_CAPACITY as u32,
                trace_bytes_max: TRACE_BYTES_MAX as u64,
                repair: repair.then(|| method("engine_actual_followup_search_v4")),
                paused_stack: stack.then(|| method("cpu_actual_paused_frame_owner_v4")),
                cold_archive: None,
                cuda_warm: warm.then(|| method("cuda_actual_backend_and_bank_events_v4")),
            },
            iterative
        ));
        journal.ensure(Some(0)).unwrap();
        journal
    }
    fn stack() -> CpuPausedStackSnapshot {
        CpuPausedStackSnapshot {
            owner_id: 11,
            event_sequence: 3,
            tokens_created: 2,
            tokens_resumed: 1,
            tokens_invalidated: 0,
            tokens_retained: 1,
            tokens_peak: 1,
            bytes_current: 1024,
            bytes_peak: 2048,
            stale_context_attempts: 0,
            stale_context_rejections: 0,
            replayed_consumed_work: 0,
            admission_closed: false,
            owner_released: false,
            complete: true,
        }
    }
    fn warm() -> NativeCudaWarmObservationV4 {
        NativeCudaWarmObservationV4 {
            measurement_contract: "rz-pals-cuda-warm-owner-accounting-v4/1".into(),
            backend_owner_id: 21,
            bank_owner_id: 22,
            physical_event_sequence: 5,
            seed_event_sequence: 2,
            capability_identity: "03".repeat(32),
            device_identity: "cuda:0".into(),
            capability_available: Some(true),
            max_leases: 1,
            device_bytes_max: 397312,
            leases_admitted: 3,
            leases_physically_completed: 3,
            leases_quarantined: 0,
            leases_active: 0,
            leases_peak: 1,
            startup_leases_admitted: 2,
            startup_leases_physically_completed: 2,
            startup_leases_quarantined: 0,
            device_bytes_current: 198656,
            device_bytes_peak: 198656,
            device_bytes_retained: 0,
            accepted_seed_consumptions: 1,
            seed_context_checked_consumptions: 1,
            rejected_seed_contexts: 0,
            fresh_value_evaluations: 0,
            value_always_fresh: true,
            admission_closed: false,
            bank_closed: false,
            backend_dropped: false,
            worker_joined: false,
            buffers_released: false,
            complete: true,
        }
    }
    fn search(stack: Option<CpuPausedStackSnapshot>) -> PalsFollowupExecutionSnapshot {
        PalsFollowupExecutionSnapshot {
            version: DOMAIN,
            game_generation: Some(0),
            search_attempt_sequence: Some(1),
            requested_position: rz_position::Position::startpos().snapshot(),
            scope: None,
            selected_recheck: PostRepairRecheckPolicy::Disabled,
            selected_resolver: ResolverPolicy::OwnRawRestricted,
            model_identity: None,
            pending_questions_peak: Some(3),
            repair_admissions_peak: Some(2),
            repair_traces: Vec::new(),
            repair_trace_total: 0,
            overflow: false,
            complete: true,
            issues: Vec::new(),
            closure: None,
            original_error: None,
            archive_start: Ok(None),
            archive_end: Ok(None),
            previous_closed_archive_owner: None,
            paused_stack_start: stack,
            paused_stack_end: stack,
        }
    }
    fn archive_source() -> ArchiveOwnerSnapshot {
        ArchiveOwnerSnapshot {
            measurement_contract: "archive-fixture-accounting/1",
            owner_method: "archive_actual_owner_events_v4",
            compiled_source_sha256: [4; 32],
            root_sha256: Some([5; 32]),
            repository_sha256: Some([6; 32]),
            owner_id: 31,
            event_sequence: 3,
            runtime_limits: Some(ArchiveRuntimeLimits {
                game_bytes_max: 1024 * 1024,
                global_bytes_max: 4 * 1024 * 1024,
                index_entries_max: 128,
                index_bytes_max: 16 * 1024,
                load_bytes_max: 1024 * 1024,
                load_deadline_max_ms: 1000,
                record_payload_bytes_max: 64 * 1024,
                max_load_pins: 1,
            }),
            root_boundary_checked: true,
            ram_released_bytes: Some(0),
            ram_release_method: "fixture_capacity",
            pending_commit_bytes: Some(0),
            global_managed_bytes: Some(0),
            global_scan_event_sequence: Some(3),
            global_scan_complete: true,
            global_scan_after_last_commit: true,
            global_scope_kind: "canonical_no_link_managed_root",
            cold_index_kind: "directory_scan",
            load_pins_max: 1,
            loaded_closure_bytes_peak: Some(0),
            loaded_closure_bytes_method: Some("fixture_capacity"),
            archive_write_bytes_total: Some(0),
            session_write_ledger_id: Some(41),
            session_write_bytes_max: Some(1024 * 1024),
            session_write_bytes_consumed: Some(0),
            complete: true,
            ..ArchiveOwnerSnapshot::default()
        }
    }
    fn archive_journal() -> FollowupJournal {
        let mut journal = journal(false, false, false, false);
        let method = NativeOwnerMethodIdentityV4 {
            method: "archive_actual_owner_events_v4".into(),
            implementation_sha256: "04".repeat(32),
        };
        journal.selected_archive = true;
        journal.methods.as_mut().unwrap().cold_archive = Some(method.clone());
        journal
            .lifecycle
            .as_mut()
            .unwrap()
            .owner_methods
            .cold_archive = Some(method);
        journal
    }
    #[test]
    fn selected_archive_missing_measurement_cannot_be_hidden_by_a_later_some() {
        let actual = archive_source();
        let mut known_zero = archive_journal();
        known_zero.capture_owner_parts(Some(actual), None, false);
        assert!(matches!(known_zero.lifecycle.unwrap().archive,
            NativeObservedV4::Observed { value, .. } if value.archive_write_bytes_total == Some(0)));
        for case in 0..5 {
            let mut journal = archive_journal();
            let mut missing = actual;
            match case {
                0 => missing.archive_write_bytes_total = None,
                1 => missing.global_managed_bytes = None,
                2 => missing.ram_released_bytes = None,
                3 => missing.session_write_bytes_consumed = None,
                _ => {}
            }
            journal.capture_owner_parts((case != 4).then_some(missing), None, false);
            journal.capture_owner_parts(Some(actual), None, true);
            let value = journal.lifecycle.unwrap();
            assert!(!value.capture_complete, "case {case}");
            assert_eq!(value.archive, NativeObservedV4::Unknown);
            assert!(!value.partial_facts.unwrap().archive.unwrap().complete);
        }
    }
    #[test]
    fn replaced_archive_copies_owner_totals_and_keeps_the_same_session_write_ledger() {
        let mut prior = archive_source();
        prior.archive_write_bytes_total = Some(10);
        prior.session_write_bytes_consumed = Some(10);
        let mut closed = prior;
        closed.event_sequence += 1;
        closed.lifecycle = ArchiveOwnerLifecycle::Closed;
        closed.admission_closed = true;
        closed.cleanup_complete = true;
        let mut next = archive_source();
        next.owner_id += 1;
        next.event_sequence = 2;
        next.session_write_bytes_consumed = Some(10);
        for case in 0..4 {
            let mut journal = archive_journal();
            journal.capture_owner_parts(Some(prior), None, true);
            journal.preserve_closed(Some(closed));
            let mut current = next;
            match case {
                1 => current.session_write_ledger_id = Some(42),
                2 => current.session_write_bytes_consumed = Some(0),
                3 => current.session_write_bytes_max = Some(2 * 1024 * 1024),
                _ => {}
            }
            journal.capture_owner_parts(Some(current), None, true);
            let value = journal.lifecycle.unwrap();
            if case == 0 {
                assert!(value.capture_complete);
                assert!(
                    matches!(value.archive, NativeObservedV4::Observed { value, .. }
                    if value.archive_write_bytes_total == Some(0)
                        && value.session_write_bytes_consumed == Some(10))
                );
                assert_eq!(
                    value.previous_closed_archive_owners[0].archive_write_bytes_total,
                    Some(10)
                );
            } else {
                assert!(!value.capture_complete, "case {case}");
                assert_eq!(value.archive, NativeObservedV4::Unknown);
            }
        }
    }
    #[test]
    fn optional_off_journal_is_absent_and_unselected_repair_metrics_stay_unknown() {
        assert!(FollowupJournal::default().lifecycle.is_none());
        let mut journal = journal(false, true, false, false);
        journal.capture_search(&search(Some(stack())));
        let value = journal.lifecycle.unwrap();
        assert!(value.capture_complete);
        assert!(value.repair_traces.is_empty());
        assert_eq!(value.repair_trace_total, NativeObservedV4::Unknown);
        assert_eq!(value.pending_questions_peak, NativeObservedV4::Unknown);
        assert!(value.repair_admissions_peak.is_none());
    }
    #[test]
    fn paused_owner_replacement_regression_or_missing_gap_cannot_be_erased() {
        for case in 0..3 {
            let mut journal = journal(false, true, false, false);
            let initial = stack();
            journal.capture_owner_parts(None, Some(initial), true);
            let mut invalid = initial;
            match case {
                0 => invalid.owner_id += 1,
                1 => invalid.tokens_resumed = 0,
                _ => {}
            }
            journal.capture_owner_parts(None, (case != 2).then_some(invalid), true);
            let mut closed = initial;
            closed.event_sequence += 1;
            closed.tokens_invalidated += 1;
            closed.tokens_retained = 0;
            closed.bytes_current = 0;
            closed.admission_closed = true;
            closed.owner_released = true;
            journal.capture_owner_parts(None, Some(closed), true);
            journal.output_retired();
            journal.seal(true, true);
            let value = journal.lifecycle.unwrap();
            assert!(!value.capture_complete && !value.owner_shutdown_complete);
            assert!(value.capture_failure.is_some());
            assert_eq!(value.paused_stack, NativeObservedV4::Unknown);
            assert!(!value.partial_facts.unwrap().paused_stack.unwrap().complete);
        }
    }
    #[test]
    fn paused_search_start_and_end_must_have_the_same_live_owner() {
        let mut journal = journal(false, true, false, false);
        let mut source = search(Some(stack()));
        source.paused_stack_end.as_mut().unwrap().owner_id += 1;
        journal.capture_search(&source);
        assert!(!journal.lifecycle.unwrap().capture_complete);
    }
    #[test]
    fn cuda_final_release_joins_the_same_cumulative_backend_and_bank_owners() {
        let initial = warm();
        let mut final_value = initial.clone();
        final_value.physical_event_sequence += 1;
        final_value.seed_event_sequence += 1;
        final_value.device_bytes_current = 0;
        final_value.admission_closed = true;
        final_value.bank_closed = true;
        final_value.backend_dropped = true;
        final_value.worker_joined = true;
        final_value.buffers_released = true;
        let mut complete = journal(false, false, true, false);
        complete.capture_warm(Some(initial.clone()));
        complete.capture_warm(Some(final_value.clone()));
        complete.output_retired();
        complete.seal(true, true);
        assert!(complete.lifecycle.unwrap().owner_shutdown_complete);
        for case in 0..3 {
            let mut journal = journal(false, false, true, false);
            journal.capture_warm(Some(initial.clone()));
            let mut invalid = final_value.clone();
            match case {
                0 => invalid.bank_owner_id += 1,
                1 => invalid.accepted_seed_consumptions = 0,
                _ => {}
            }
            journal.capture_warm((case != 2).then_some(invalid));
            journal.capture_warm(Some(final_value.clone()));
            journal.output_retired();
            journal.seal(true, true);
            let value = journal.lifecycle.unwrap();
            assert!(!value.capture_complete && !value.owner_shutdown_complete);
            assert_eq!(value.cuda_warm, NativeObservedV4::Unknown);
            assert!(!value.partial_facts.unwrap().cuda_warm.unwrap().complete);
        }
    }
    #[test]
    fn duplicate_search_capture_and_unknown_game_are_sticky_incomplete() {
        let mut duplicate = journal(true, false, false, false);
        let source = search(None);
        duplicate.capture_search(&source);
        duplicate.capture_search(&source);
        assert!(!duplicate.lifecycle.unwrap().capture_complete);
        let mut journal = journal(true, false, false, false);
        journal.lifecycle = None;
        let mut missing = source.clone();
        missing.game_generation = None;
        journal.capture_search(&missing);
        journal.capture_search(&source);
        assert!(!journal.lifecycle.unwrap().capture_complete);
    }
}
