//! V4-only copies of actual owner events. Selected policy markers remain
//! independent. No capture here owns a tensor, checkpoint, pin or archive file.
use rz_search::{
    cpu::CpuPausedStackSnapshot,
    pals::{
        engine::{
            PalsExecutionEndpointSnapshot, PalsExecutionEndpointValue,
            PalsFollowupExecutionSnapshot, PalsFollowupOwnerSnapshot,
            PalsFrozenComparisonExecution, PalsRepairExecutionTrace,
            RoleExecutionContext, RoleValueExecutionEvidence,
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
        ("pals_attestation.rs", include_str!("../pals_attestation.rs")),
        ("pals_attestation/followup.rs", include_str!("followup.rs")),
        ("search_driver.rs", include_str!("../search_driver.rs")),
        ("search_driver/work.rs", include_str!("../search_driver/work.rs")),
        ("pals_native.rs", include_str!("../pals_native.rs")),
        ("pals_native/private_warm.rs", include_str!("../pals_native/private_warm.rs")),
        ("main.rs", include_str!("../main.rs")),
    ];
    for bytes in std::iter::once(b"rz-pals-followup-producer-sources/1".as_slice())
        .chain(sources.iter().flat_map(|(name, source)| [name.as_bytes(), source.as_bytes()]))
    {
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
        Self::Observed { value, method: method.into() }
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
    CaptureIncomplete, CounterOverflow, OwnerUnknown,
    PhysicalCompletionUnknown, ContextMismatch,
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
    pub admission_closed: bool,
    pub cleanup_complete: bool,
    pub complete: bool,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeArchiveRuntimeLimitsV4 {
    pub game_bytes_max:u64, pub global_bytes_max:u64,
    pub index_entries_max:u32, pub index_bytes_max:u64,
    pub load_bytes_max:u64, pub load_deadline_max_ms:u64,
    pub record_payload_bytes_max:u64, pub max_load_pins:u32,
}
impl From<ArchiveRuntimeLimits> for NativeArchiveRuntimeLimitsV4 {
    fn from(s: ArchiveRuntimeLimits) -> Self {
        Self { game_bytes_max:s.game_bytes_max, global_bytes_max:s.global_bytes_max,
            index_entries_max:s.index_entries_max,index_bytes_max:s.index_bytes_max,
            load_bytes_max:s.load_bytes_max,load_deadline_max_ms:s.load_deadline_max_ms,
            record_payload_bytes_max:s.record_payload_bytes_max,max_load_pins:s.max_load_pins }
    }
}
impl From<ArchiveOwnerSnapshot> for NativeArchiveObservationV4 {
    fn from(s: ArchiveOwnerSnapshot) -> Self {
        Self {
            measurement_contract:s.measurement_contract.into(),owner_method:s.owner_method.into(),
            compiled_source_sha256:hex(&s.compiled_source_sha256),owner_id:s.owner_id,event_sequence:s.event_sequence,
            root_sha256:s.root_sha256.map(|v|hex(&v)),repository_sha256:s.repository_sha256.map(|v|hex(&v)),
            generation:s.generation,reserved_generations:s.reserved_generations,
            last_committed_generation:s.last_committed_generation,last_verified_generation:s.last_verified_generation,
            lifecycle:match s.lifecycle { ArchiveOwnerLifecycle::Open=>"open",ArchiveOwnerLifecycle::Closed=>"closed",ArchiveOwnerLifecycle::Failed=>"failed" }.into(),
            runtime_limits:s.runtime_limits.map(Into::into),root_boundary_checked:s.root_boundary_checked,
            committed_chunks:s.committed_chunks,integrity_verified_chunks:s.integrity_verified_chunks,
            committed_bytes:s.committed_bytes,integrity_verified_bytes:s.integrity_verified_bytes,
            ram_released_bytes:s.ram_released_bytes,ram_release_method:s.ram_release_method.into(),
            ram_reclaim_events:s.ram_reclaim_events,integrity_before_reclaim_events:s.integrity_before_reclaim_events,
            unverified_reclaim_events:s.unverified_reclaim_events,pending_commit_bytes:s.pending_commit_bytes,
            pending_buffers_retained:s.pending_buffers_retained,pending_files_retained:s.pending_files_retained,
            global_managed_bytes:s.global_managed_bytes,
            global_scan_event_sequence:s.global_scan_event_sequence,global_scan_complete:s.global_scan_complete,
            global_scan_after_last_commit:s.global_scan_after_last_commit,global_scope_kind:s.global_scope_kind.into(),
            index_entries_peak:s.index_entries_peak,index_bytes_peak:s.index_bytes_peak,cold_index_kind:s.cold_index_kind.into(),
            pinned_entries_peak:s.pinned_entries_peak,load_pins_max:s.load_pins_max,
            loaded_closure_bytes_peak:s.loaded_closure_bytes_peak,loaded_closure_bytes_method:s.loaded_closure_bytes_method.map(Into::into),
            loads_requested:s.loads_requested,loads_completed:s.loads_completed,load_bytes_total:s.load_bytes_total,
            load_bytes_peak:s.load_bytes_peak,load_elapsed_peak_ms:s.load_elapsed_peak_ms,
            owner_generation_checks:s.owner_generation_checks,owner_generation_check_failures:s.owner_generation_check_failures,
            quota_failures:s.quota_failures,io_failures:s.io_failures,pin_saturation_failures:s.pin_saturation_failures,
            archive_write_bytes_total:s.archive_write_bytes_total,admission_closed:s.admission_closed,
            cleanup_complete:s.cleanup_complete,complete:s.complete,
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativePausedStackObservationV4 {
    pub owner_id:u64,pub event_sequence:u64,
    pub tokens_created:u64,pub tokens_resumed:u64,pub tokens_invalidated:u64,
    pub tokens_retained:u32,pub tokens_peak:u32,pub bytes_current:u64,pub bytes_peak:u64,
    pub stale_context_attempts:u64,pub stale_context_rejections:u64,
    pub replayed_consumed_work:u64,pub admission_closed:bool,pub owner_released:bool,pub complete:bool,
}
impl From<CpuPausedStackSnapshot> for NativePausedStackObservationV4 {
    fn from(s:CpuPausedStackSnapshot)->Self {
        Self { owner_id:s.owner_id,event_sequence:s.event_sequence,tokens_created:s.tokens_created,
            tokens_resumed:s.tokens_resumed,tokens_invalidated:s.tokens_invalidated,tokens_retained:s.tokens_retained,
            tokens_peak:s.tokens_peak,bytes_current:s.bytes_current,bytes_peak:s.bytes_peak,
            stale_context_attempts:s.stale_context_attempts,stale_context_rejections:s.stale_context_rejections,
            replayed_consumed_work:s.replayed_consumed_work,admission_closed:s.admission_closed,
            owner_released:s.owner_released,complete:s.complete }
    }
}

/// Native's bank/context/worker observations and Eval's actual CUDA owner
/// snapshot are joined only for the same execution. All bytes are explicit K/V
/// payloads; ORT staging/workspace/VRAM is outside this measurement.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaWarmObservationV4 {
    pub measurement_contract:String,pub backend_owner_id:u64,pub bank_owner_id:u64,
    pub physical_event_sequence:u64,pub seed_event_sequence:u64,
    pub capability_identity:String,pub device_identity:String,pub capability_available:Option<bool>,
    pub max_leases:u32,pub device_bytes_max:u64,
    pub leases_admitted:u64,pub leases_physically_completed:u64,pub leases_quarantined:u64,
    pub leases_active:u32,pub leases_peak:u32,
    pub startup_leases_admitted:u64,pub startup_leases_physically_completed:u64,pub startup_leases_quarantined:u64,
    pub device_bytes_current:u64,pub device_bytes_peak:u64,pub device_bytes_retained:u64,
    pub accepted_seed_consumptions:u64,pub seed_context_checked_consumptions:u64,
    pub rejected_seed_contexts:u64,pub fresh_value_evaluations:u64,
    pub value_always_fresh:bool,pub admission_closed:bool,pub bank_closed:bool,
    pub backend_dropped:bool,pub worker_joined:bool,pub buffers_released:bool,pub complete:bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRoleContextV4 {
    pub game_generation:u64,pub search_generation:u64,
    pub situation_slot:u64,pub situation_generation:u64,pub state:u64,pub focus:u64,
    pub purpose:String,pub focus_sha256:String,pub prefix_sha256:String,pub proposal_sha256:String,
    pub refutation_sha256:Option<String>,pub divergence_sha256:String,
    pub public_revision:u64,pub situation_revision:u64,
}
impl From<RoleExecutionContext> for NativeRoleContextV4 {
    fn from(s:RoleExecutionContext)->Self {
        Self { game_generation:s.game_generation,search_generation:s.search_generation,
            situation_slot:s.situation.slot as u64,situation_generation:s.situation.generation,
            state:s.state.0 as u64,focus:s.focus.0 as u64,purpose:format!("{:?}",s.purpose),
            focus_sha256:hex(&s.focus_sha256),prefix_sha256:hex(&s.prefix_sha256),proposal_sha256:hex(&s.proposal_sha256),
            refutation_sha256:s.refutation_sha256.map(|v|hex(&v)),divergence_sha256:hex(&s.divergence_sha256),
            public_revision:s.public_revision,situation_revision:s.situation_revision }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativePhysicalIdV4 { pub epoch:u64,pub sequence:u64 }
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativePreparedStateV4 { pub owner:u64,pub revision:u64,pub semantic_sha256:String }
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRoleValueProvenanceV4 {
    pub request:NativePhysicalIdV4,pub execution:NativePhysicalIdV4,
    pub prepared_state:NativePreparedStateV4,pub prepared_perspective:String,
    pub input_sha256:String,pub model_epoch_sha256:String,pub context:NativeRoleContextV4,
    pub fresh:Option<bool>,pub physically_completed:Option<bool>,pub completed_nn_inputs:Option<u64>,pub complete:bool,
}
impl From<RoleValueExecutionEvidence> for NativeRoleValueProvenanceV4 {
    fn from(s:RoleValueExecutionEvidence)->Self {
        Self {request:NativePhysicalIdV4 {epoch:s.request.epoch.0,sequence:s.request.sequence},
            execution:NativePhysicalIdV4 {epoch:s.execution.epoch.0,sequence:s.execution.sequence},
            prepared_state:NativePreparedStateV4 {owner:s.prepared_state.owner.0,revision:s.prepared_state.revision.0,semantic_sha256:hex(&s.prepared_state.semantic.0)},
            prepared_perspective:match s.prepared_perspective {rz_contracts::Color::White=>"white",rz_contracts::Color::Black=>"black"}.into(),
            input_sha256:hex(&s.input_sha256),model_epoch_sha256:hex(&s.model_epoch),context:s.context.into(),
            fresh:s.fresh,physically_completed:s.physically_completed,completed_nn_inputs:s.completed_nn_inputs,complete:s.complete}
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag="kind",rename_all="snake_case",deny_unknown_fields)]
pub enum NativeRecheckValueV4 {
    Unknown,
    FrozenWdl {model_value_semantics:String,model_identity:String,encoding_identity:String,precision:String,
        model_epoch_sha256:String,input_sha256:String,state_sha256:String,perspective:String,wdl_bits:[u32;3],
        context_revision:u64,observation:u64,accepted_output:bool,execution:Option<NativeRoleValueProvenanceV4>},
    RulesTerminal {state_sha256:String,perspective:String,winner:Option<String>,reason:String},
}
fn perspective(color:rz_position::Color)->String {
    match color {rz_position::Color::White=>"white",rz_position::Color::Black=>"black"}.into()
}
fn endpoint(s:&PalsExecutionEndpointSnapshot)->NativeRecheckValueV4 {
    let state_sha256=hex(&rz_position::contracts::rules_snapshot_semantic_digest(&s.snapshot).0);
    match &s.value {
        PalsExecutionEndpointValue::Unknown=>NativeRecheckValueV4::Unknown,
        PalsExecutionEndpointValue::FrozenWdl(v)=>NativeRecheckValueV4::FrozenWdl {
            model_value_semantics:v.identity.semantics.clone(),model_identity:v.identity.model.clone(),
            encoding_identity:v.identity.encoding.clone(),precision:v.identity.precision.clone(),
            model_epoch_sha256:hex(&v.identity.model_epoch),input_sha256:hex(&v.input_sha256),state_sha256,
            perspective:perspective(v.perspective),wdl_bits:v.wdl_bits,context_revision:v.context_revision,
            observation:v.observation.0 as u64,accepted_output:v.logical_accepted,execution:v.execution.map(Into::into),
        },
        PalsExecutionEndpointValue::RulesTerminal {reason,value,perspective:color}=>NativeRecheckValueV4::RulesTerminal {
            state_sha256,perspective:perspective(*color),
            winner:if *value==0 {None} else {Some(perspective(if *value>0 {*color} else {color.opposite()}))},
            reason:format!("{reason:?}"),
        },
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRecheckObservationV4 {
    pub before:NativeRecheckValueV4,pub after:NativeRecheckValueV4,pub comparison_perspective:String,
    pub context_revision:Option<u64>,pub comparable:bool,pub fresh_requests_completed:Option<u32>,
    pub completed_nn_inputs:Option<u64>,pub publication:Option<u64>,pub complete:bool,
}
impl From<&PalsFrozenComparisonExecution> for NativeRecheckObservationV4 {
    fn from(s:&PalsFrozenComparisonExecution)->Self {
        Self {before:endpoint(&s.before),after:s.after.as_ref().map_or(NativeRecheckValueV4::Unknown,endpoint),
            comparison_perspective:perspective(s.comparison_perspective),context_revision:s.context_revision,
            comparable:s.comparable,fresh_requests_completed:s.fresh_requests_completed,completed_nn_inputs:s.completed_nn_inputs,
            publication:s.publication.map(|id|id.0 as u64),complete:s.complete}
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRepairRecheckV4 {
    pub reply_context:NativeRoleContextV4,pub prepared_accepted:bool,pub reply_call_attempted:bool,
    pub reply_accepted:bool,pub selected_response:Option<String>,pub counterline:Vec<String>,
    pub counterline_completed:bool,pub disposition:String,pub comparison:NativeRecheckObservationV4,
    pub original_error:Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeRepairTraceV4 {
    pub game_generation:u64,pub search_attempt_sequence:u64,pub store_root_generation:u64,
    pub root_situation_slot:u64,pub root_situation_generation:u64,pub root_state:u64,
    pub first_move:String,pub lineage_root_record:u64,pub parent_revision:Option<u64>,pub supersedes_revision:Option<u64>,
    pub record_revision:u64,pub repair_ordinal:u32,pub repaired_line:u64,pub repaired:Vec<String>,
    pub accepted_question:Option<NativeRoleContextV4>,pub iterative_repair_comparison:Option<NativeRecheckObservationV4>,
    pub recheck:Option<NativeRepairRecheckV4>,
}
impl From<&PalsRepairExecutionTrace> for NativeRepairTraceV4 {
    fn from(s:&PalsRepairExecutionTrace)->Self {
        Self {game_generation:s.scope.game_generation,search_attempt_sequence:s.scope.search_attempt_sequence,
            store_root_generation:s.scope.store_root_generation,root_situation_slot:s.scope.root.slot as u64,
            root_situation_generation:s.scope.root.generation,root_state:s.scope.root_state.0 as u64,
            first_move:s.first_move.to_string(),lineage_root_record:s.lineage_root_record,parent_revision:s.parent_revision,
            supersedes_revision:s.supersedes_revision,record_revision:s.record_revision,repair_ordinal:s.repair_ordinal,
            repaired_line:s.repaired_line.0 as u64,repaired:s.repaired.iter().map(ToString::to_string).collect(),
            accepted_question:s.accepted_question.map(Into::into),
            iterative_repair_comparison:s.iterative_repair_comparison.as_ref().map(Into::into),
            recheck:s.recheck.as_ref().map(|v|NativeRepairRecheckV4 {
                reply_context:v.reply_context.into(),prepared_accepted:v.prepared_accepted,reply_call_attempted:v.reply_call_attempted,
                reply_accepted:v.reply_accepted,selected_response:v.selected_response.map(|mv|mv.to_string()),
                counterline:v.counterline.iter().map(ToString::to_string).collect(),counterline_completed:v.counterline_completed,
                disposition:format!("{:?}",v.disposition),comparison:(&v.comparison).into(),
                original_error:v.original_error.as_ref().map(|error|format!("{error:?}")),
            })}
    }
}

/// Process-cumulative bounded journal. Snapshot reads never acquire an Engine
/// lock, produce an event, sum the same owner twice or authorize UCI output.
pub(crate) struct FollowupJournal {
    pub(crate) lifecycle: Option<PalsFollowupLifecycleV4>,
    process_epoch:Option<u64>,
    methods:Option<NativeFollowupOwnerMethodsV4>,
    last_attempt:Option<(u64,u64)>,
    trace_bytes:usize,
    output_retired:bool,
    selected_archive:bool,
    selected_stack:bool,
    selected_warm:bool,
}
impl Default for FollowupJournal {
    fn default()->Self {Self {lifecycle:None,process_epoch:None,methods:None,last_attempt:None,trace_bytes:2,
        output_retired:false,selected_archive:false,selected_stack:false,selected_warm:false}}
}
impl FollowupJournal {
    pub(crate) fn enable(&mut self,process_epoch:Option<u64>,methods:NativeFollowupOwnerMethodsV4)->bool {
        if self.methods.is_some() {return false;}
        self.selected_archive=methods.cold_archive.is_some();self.selected_stack=methods.paused_stack.is_some();
        self.selected_warm=methods.cuda_warm.is_some();self.process_epoch=process_epoch;self.methods=Some(methods);true
    }
    pub(crate) fn enabled(&self)->bool {self.methods.is_some()}
    fn ensure(&mut self,game:Option<u64>)->Option<&mut PalsFollowupLifecycleV4> {
        let methods=self.methods.as_ref()?;
        if self.lifecycle.is_none() {
            self.lifecycle=Some(PalsFollowupLifecycleV4 {domain:DOMAIN.into(),schema_version:1,process_epoch:self.process_epoch,
                game_generation:game?,capture_complete:true,owner_shutdown_complete:false,
                trace_capacity:TRACE_CAPACITY as u32,trace_bytes_max:TRACE_BYTES_MAX as u64,owner_methods:methods.clone(),
                archive:NativeObservedV4::Unknown,paused_stack:NativeObservedV4::Unknown,cuda_warm:NativeObservedV4::Unknown,
                repair_traces:Vec::new(),repair_trace_total:NativeObservedV4::Unknown,pending_questions_peak:NativeObservedV4::Unknown,
                repair_admissions_peak:None,partial_facts:None,capture_failure:None});
        }
        self.lifecycle.as_mut()
    }
    pub(crate) fn incomplete(&mut self,failure:PalsCaptureFailureV4) {
        if let Some(lifecycle)=&mut self.lifecycle {lifecycle.capture_complete=false;lifecycle.owner_shutdown_complete=false;
            lifecycle.capture_failure.get_or_insert(failure);}
    }
    pub(crate) fn capture_search(&mut self,s:&PalsFollowupExecutionSnapshot) {
        let Some(game)=s.game_generation else {self.incomplete(PalsCaptureFailureV4::OwnerUnknown);return;};
        if self.ensure(Some(game)).is_none() {return;}
        let Some(attempt)=s.search_attempt_sequence else {self.incomplete(PalsCaptureFailureV4::CounterOverflow);return;};
        if self.last_attempt==Some((game,attempt)) {self.incomplete(PalsCaptureFailureV4::ContextMismatch);return;}
        self.last_attempt=Some((game,attempt));self.output_retired=false;
        let lifecycle=self.lifecycle.as_mut().expect("enabled lifecycle");lifecycle.game_generation=game;
        let total=match lifecycle.repair_trace_total {NativeObservedV4::Observed{value,..}=>value,NativeObservedV4::Unknown=>0};
        if let Some(total)=total.checked_add(s.repair_trace_total) {
            lifecycle.repair_trace_total=NativeObservedV4::observed(total,"engine_actual_repair_recorder_v4");
        } else {self.incomplete(PalsCaptureFailureV4::CounterOverflow);}
        let lifecycle=self.lifecycle.as_mut().expect("enabled lifecycle");
        if let Some(peak)=s.pending_questions_peak {
            let prior=match lifecycle.pending_questions_peak {NativeObservedV4::Observed{value,..}=>value,NativeObservedV4::Unknown=>0};
            lifecycle.pending_questions_peak=NativeObservedV4::observed(prior.max(peak),"engine_actual_repair_recorder_v4");
        }
        if let Some(peak)=s.repair_admissions_peak {
            let prior=match &lifecycle.repair_admissions_peak {Some(NativeObservedV4::Observed{value,..})=>*value,_=>0};
            lifecycle.repair_admissions_peak=Some(NativeObservedV4::observed(prior.max(peak),"engine_actual_repair_recorder_v4"));
        }
        for trace in &s.repair_traces {
            let raw=NativeRepairTraceV4::from(trace);
            let Ok(bytes)=serde_json::to_vec(&raw) else {self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);break;};
            let Some(total)=self.trace_bytes.checked_add(bytes.len()+1) else {self.incomplete(PalsCaptureFailureV4::CounterOverflow);break;};
            if self.lifecycle.as_ref().is_none_or(|l|l.repair_traces.len()>=TRACE_CAPACITY) || total>TRACE_BYTES_MAX {
                self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);break;
            }
            self.trace_bytes=total;self.lifecycle.as_mut().expect("enabled lifecycle").repair_traces.push(raw);
        }
        if !s.complete || s.overflow || s.pending_questions_peak.is_none() || s.repair_admissions_peak.is_none() {
            self.incomplete(PalsCaptureFailureV4::CaptureIncomplete);
        }
        self.capture_owner_parts(s.archive_end.as_ref().ok().copied().flatten(),s.paused_stack_end);
    }
    fn capture_owner_parts(&mut self,archive:Option<ArchiveOwnerSnapshot>,stack:Option<CpuPausedStackSnapshot>) {
        let Some(lifecycle)=&mut self.lifecycle else {return;};
        let mut partial=lifecycle.partial_facts.take().unwrap_or_default();
        if self.selected_archive {
            match archive.map(NativeArchiveObservationV4::from) {
                Some(value) if value.complete && value.runtime_limits.is_some() && value.ram_released_bytes.is_some()
                    && value.pending_commit_bytes.is_some() && value.global_managed_bytes.is_some()
                    && value.archive_write_bytes_total.is_some()=>{
                    lifecycle.archive=NativeObservedV4::observed(value,"archive_actual_owner_events_v4");partial.archive=None;
                }
                value=>{lifecycle.archive=NativeObservedV4::Unknown;partial.archive=value;}
            }
        }
        if self.selected_stack {
            match stack.map(NativePausedStackObservationV4::from) {
                Some(value) if value.complete=>{lifecycle.paused_stack=NativeObservedV4::observed(value,"cpu_actual_paused_frame_owner_v4");partial.paused_stack=None;}
                value=>{lifecycle.paused_stack=NativeObservedV4::Unknown;partial.paused_stack=value;}
            }
        }
        if partial.archive.is_some() || partial.paused_stack.is_some() || partial.cuda_warm.is_some() {
            if serde_json::to_vec(&partial).is_ok_and(|bytes|bytes.len()<=PARTIAL_BYTES_MAX) {lifecycle.partial_facts=Some(partial);}
            lifecycle.capture_complete=false;lifecycle.capture_failure.get_or_insert(PalsCaptureFailureV4::CaptureIncomplete);
        }
    }
    pub(crate) fn capture_owner(&mut self,s:&PalsFollowupOwnerSnapshot) {
        if self.ensure(s.game_generation).is_none(){return;}
        self.capture_owner_parts(s.archive.as_ref().ok().copied().flatten(),s.paused_stack);
        if s.archive.is_err() || s.new_game_error.is_some() {self.incomplete(PalsCaptureFailureV4::OwnerUnknown);}
    }
    pub(crate) fn capture_warm(&mut self,value:Option<NativeCudaWarmObservationV4>) {
        let Some(lifecycle)=&mut self.lifecycle else {return;};
        if !self.selected_warm {return;}
        match value {
            Some(value) if value.complete=>{lifecycle.cuda_warm=NativeObservedV4::observed(value,"cuda_actual_warm_owner_events_v4");}
            value=>{lifecycle.cuda_warm=NativeObservedV4::Unknown;
                let partial=lifecycle.partial_facts.get_or_insert_with(Default::default);partial.cuda_warm=value;
                lifecycle.capture_complete=false;lifecycle.capture_failure.get_or_insert(PalsCaptureFailureV4::CaptureIncomplete);}
        }
    }
    pub(crate) fn output_retired(&mut self) {self.output_retired=true;}
    pub(crate) fn seal(&mut self,checker_fenced:bool,native_fenced:bool) {
        let Some(lifecycle)=&mut self.lifecycle else {return;};
        let archive_closed=!self.selected_archive || matches!(&lifecycle.archive,NativeObservedV4::Observed{value,..}
            if value.admission_closed && value.cleanup_complete && value.lifecycle=="closed"
                && !value.pending_buffers_retained && !value.pending_files_retained);
        let stack_closed=!self.selected_stack || matches!(&lifecycle.paused_stack,NativeObservedV4::Observed{value,..}
            if value.admission_closed && value.owner_released && value.tokens_retained==0 && value.bytes_current==0);
        let warm_closed=!self.selected_warm || matches!(&lifecycle.cuda_warm,NativeObservedV4::Observed{value,..}
            if value.admission_closed && value.bank_closed && value.backend_dropped && value.worker_joined && value.buffers_released);
        lifecycle.owner_shutdown_complete=lifecycle.capture_complete && self.output_retired && checker_fenced
            && native_fenced && archive_closed && stack_closed && warm_closed;
        if !lifecycle.owner_shutdown_complete {lifecycle.capture_failure.get_or_insert(PalsCaptureFailureV4::OwnerUnknown);}
    }
}
