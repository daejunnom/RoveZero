//! Adaptive PALS execution declarations and saved observations.
//!
//! V5 is a new persisted domain. It does not relabel, convert or validate itself
//! through V1--V4 manifests. Shared artifact and observation types retain their
//! original meanings. Declaring a recipe never runs it or attests acceptance.
use crate::pals_manifest as shared;
use crate::{
    ArtifactRef, ManifestError, PalsComponentV3, PalsModelBackendV3, PalsObservedV3,
    PalsPhysicalStateV3, PalsPrecisionV3, PalsResultV3, PalsRoleV3, PalsTerminationV3, decode_json,
    digest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PALS_MANIFEST_V5_DOMAIN: &str = "rz-pals-execution-v5/1";
pub const PALS_RECEIPT_V5_DOMAIN: &str = "rz-pals-receipt-v5/1";
pub const PALS_V5_CONTRACT_REVISION: &str = "0.1";
pub const PALS_V5_COORDINATOR_REVISION: &str = "pals-adaptive/0.1";
pub const PALS_V5_MODEL_SCHEMA: &str = "rovezero.pals-model.v3";
pub const PALS_V5_MODEL_SEMANTICS: &str = "rovezero.pals-model-semantics.v3";
pub const PALS_V5_ENCODING_SCHEMA: &str = "rovezero.pals-virtual-review.v1";
pub const PALS_V5_TRAINING_CHECKPOINT_DOMAIN: &str = "rz-pals-actual-generation-checkpoint/1";
pub const PALS_V5_DIAGNOSTIC_CHECKPOINT_DOMAIN: &str = "rz-pals-python-nonzero-checkpoint/1";
pub const PALS_V5_TRAINING_RECORD_DOMAIN: &str = "rz-pals-generation-data/3";
pub const PALS_V5_GENERATION_INPUT_DOMAIN: &str = "rz-pals-generation-input/3";
pub const PALS_V5_GENERATION_SPEC_DOMAIN: &str = "rz-pals-train-generation/1";
pub const PALS_V5_GENERATION_EXPORT_DOMAIN: &str = "rz-pals-actual-generation-export/1";
pub const PALS_V5_GENERATION_CHUNK_DOMAIN: &str = "rz-pals-generation-chunk-manifest/1";
/// Producer/collector/trainer agree on these persisted domains. Native
/// submission/completion/retirement receipts remain separate evidence domains.
pub const PALS_V5_GENERATION_CONSUMPTION_DOMAINS: [(&str, &str); 6] = [
    ("records", PALS_V5_TRAINING_RECORD_DOMAIN),
    ("input_seal", PALS_V5_GENERATION_INPUT_DOMAIN),
    ("train_spec", PALS_V5_GENERATION_SPEC_DOMAIN),
    ("checkpoint", PALS_V5_TRAINING_CHECKPOINT_DOMAIN),
    ("export", PALS_V5_GENERATION_EXPORT_DOMAIN),
    ("chunk_manifest", PALS_V5_GENERATION_CHUNK_DOMAIN),
];
pub const PALS_V5_FIRST_GENERATION: &str = "first-real-generation-v1";
pub const PALS_V5_STOCKFISH_SOURCE: &str = "edb0d9db6731067ec50ce619ff372b463bc4dd5d";
pub const PALS_V5_LINE_BUCKETS: [u32; 6] = [8, 16, 32, 64, 128, 256];
/// Root-lifetime replay identifiers are bounded independently in the role and
/// CPU execution namespaces. Their sum is an observation, not another limit.
pub const PALS_V5_REPLAY_HISTORY_PER_NAMESPACE_MAX: u32 = 65_536;
/// All issued frames, including closed frames, retain root-lifetime genealogy.
pub const PALS_V5_FRAME_GENEALOGY_MAX: u32 = 1_024;
/// V action replay belongs to a separate retained root-lifetime namespace.
/// It does not represent additional role submissions or completed NN inputs.
pub const PALS_V5_VALIDATOR_SOURCE_CONTEXTS_MAX: u32 = 128;
pub const PALS_V5_VALIDATOR_ACTION_KEYS_MAX: u32 = 896;

macro_rules! choices {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
    };
}
choices!(PalsRunPurposeV5 {
    CpuValidation,
    NeuralValidation,
    CollectionPair,
    ActionCollection,
    TrainingGeneration,
    EvaluationPair
});
choices!(PalsEngineModeV5 {
    Collection,
    Deployment
});
choices!(PalsViewSelectionV5 {
    V2Full,
    V2Selected,
    VirtualReview
});
choices!(PalsCoordinatorPolicyV5 {
    Synchronous,
    CoordinatorFixed,
    CoordinatorAdaptive
});
choices!(PalsTopologyPolicyV5 {
    SharedControl,
    DedicatedControl
});
choices!(PalsComparisonV5 {
    V2FullToSelected,
    SynchronousToFixed,
    FixedToAdaptive,
    V2ToCompact,
    InternalWeights,
    CpuRConnection,
    System
});
choices!(PalsArtifactDomainV5 {
    Untrained,
    Diagnostic,
    RealGeneration
});
choices!(PalsCpuSelectionV5 {
    Disabled,
    Own,
    Stockfish19Full
});
choices!(PalsFailureV5 {
    Cancelled,
    Expired,
    Allocation,
    Cuda,
    PhysicalCompletionUnknown,
    ResourceAdmission,
    CoordinatorCapacity,
    ProcessCleanup,
    Archive,
    Infrastructure,
    InvalidTargets,
    InsufficientValidTargets,
    Training,
    SelectedModeViolation,
    ModelIdentityViolation,
    SourceIdentityViolation,
    IncompleteGame,
    OutputQuota
});

mod resource_readback;
mod source_registry;
pub use resource_readback::*;
pub use source_registry::*;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsReviewContextV5 {
    pub selection: PalsViewSelectionV5,
    pub token_budget: u32,
    pub selected_records_max: u32,
    pub active_bundles_max: u32,
    pub anchors_max: u32,
    pub high_water_percent: u32,
    pub target_water_percent: u32,
    pub protected_eligible_consumptions: u32,
    /// Must include every sequence actually read, including query duplicates.
    pub count_query_duplicates: bool,
    pub include_actual_game_history_tokens: bool,
    pub atomic_bundle_selection: bool,
    pub rules_rebase_validation: bool,
    pub archive_evidence_preserved: bool,
}
impl Default for PalsReviewContextV5 {
    fn default() -> Self {
        Self {
            selection: PalsViewSelectionV5::VirtualReview,
            token_budget: 512,
            selected_records_max: 64,
            active_bundles_max: 256,
            anchors_max: 4,
            high_water_percent: 80,
            target_water_percent: 55,
            protected_eligible_consumptions: 2,
            count_query_duplicates: true,
            include_actual_game_history_tokens: false,
            atomic_bundle_selection: true,
            rules_rebase_validation: true,
            archive_evidence_preserved: true,
        }
    }
}
impl PalsReviewContextV5 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            [128, 256, 512].contains(&self.token_budget)
                && (1..=if self.selection == PalsViewSelectionV5::V2Full {
                    128
                } else {
                    64
                })
                    .contains(&self.selected_records_max)
                && (1..=256).contains(&self.active_bundles_max)
                && (1..=4).contains(&self.anchors_max)
                && self.selected_records_max <= self.active_bundles_max
                && self.high_water_percent == 80
                && self.target_water_percent == 55
                && self.protected_eligible_consumptions == 2
                && self.count_query_duplicates
                && !self.include_actual_game_history_tokens
                && self.atomic_bundle_selection
                && self.rules_rebase_validation
                && self.archive_evidence_preserved,
            "review budget, atomic relations, actual-history exclusion or archive policy differs",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCpuSlicesV5 {
    pub workers: u32,
    pub slice_native_nodes_max: u64,
    pub soft_lease_micros: u64,
    pub paused_jobs_per_worker_max: u32,
    pub paused_bytes_per_worker_max: u64,
    pub waiting_jobs_max: u32,
    pub worker_commands_max: u32,
    pub role_tickets_max: u32,
    pub control_mailbox_independent: bool,
    pub continuation_same_owner: bool,
    pub terminal_events_preserved: bool,
}
impl Default for PalsCpuSlicesV5 {
    fn default() -> Self {
        Self {
            workers: 1,
            slice_native_nodes_max: 4096,
            soft_lease_micros: 8000,
            paused_jobs_per_worker_max: 1,
            paused_bytes_per_worker_max: 8 << 20,
            waiting_jobs_max: 64,
            worker_commands_max: 8,
            role_tickets_max: 16,
            control_mailbox_independent: true,
            continuation_same_owner: true,
            terminal_events_preserved: true,
        }
    }
}
impl PalsCpuSlicesV5 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            [1, 2, 4].contains(&self.workers)
                && (1..=4096).contains(&self.slice_native_nodes_max)
                && (1..=8000).contains(&self.soft_lease_micros)
                && self.paused_jobs_per_worker_max == 1
                && (1..=8 << 20).contains(&self.paused_bytes_per_worker_max)
                && (1..=64).contains(&self.waiting_jobs_max)
                && (1..=8).contains(&self.worker_commands_max)
                && (1..=16).contains(&self.role_tickets_max)
                && self.control_mailbox_independent
                && self.continuation_same_owner
                && self.terminal_events_preserved,
            "CPU slices exceed bounded queues/ownership or allow lossy terminal control",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsAdaptiveCostsV5 {
    pub ewma_numerator: u32,
    pub ewma_denominator: u32,
    pub recent_samples_max: u32,
    pub completed_samples_min: u32,
    pub percentile: u32,
    pub backlog_low_role_p95: u32,
    pub backlog_high_role_p95: u32,
    pub minimum_slice_yield_overhead_factor: u32,
    pub topology_cooldown_ms: u64,
    pub censored_observations_retained: bool,
    pub unknown_remaining_is_zero: bool,
}
impl Default for PalsAdaptiveCostsV5 {
    fn default() -> Self {
        Self {
            ewma_numerator: 1,
            ewma_denominator: 8,
            recent_samples_max: 128,
            completed_samples_min: 8,
            percentile: 95,
            backlog_low_role_p95: 1,
            backlog_high_role_p95: 3,
            minimum_slice_yield_overhead_factor: 4,
            topology_cooldown_ms: 2000,
            censored_observations_retained: true,
            unknown_remaining_is_zero: false,
        }
    }
}
impl PalsAdaptiveCostsV5 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self == &Self::default(),
            "initial adaptive estimator/tail/hysteresis registration differs",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCoordinatorV5 {
    pub revision: String,
    pub policy: PalsCoordinatorPolicyV5,
    pub slices: PalsCpuSlicesV5,
    pub costs: PalsAdaptiveCostsV5,
    pub physical_gpu_workers: u32,
    pub physical_gpu_concurrency: u32,
    pub inference_batch_max: u32,
    pub batch_wait_micros: u64,
    pub manager_executes_blocking_work: bool,
    pub fixed_root_generation: bool,
    pub per_question_ticket: bool,
    pub max_repairs_per_first_move: u32,
    pub pending_repairs_max: u32,
    /// Role RequestId and CPU ExecutionId replay sets each have this limit.
    pub replay_history_per_namespace_max: u32,
    /// All issued frame identities, not only Repair frames. Ranking fanouts
    /// are a bounded subset of these identities and are not added a second time.
    pub frame_genealogy_max: u32,
    /// Original accepted V source contexts, independently of role RequestIds.
    pub validator_source_contexts_max: u32,
    /// Seven supported action identities per retained V source context.
    pub validator_action_keys_max: u32,
}
impl Default for PalsCoordinatorV5 {
    fn default() -> Self {
        Self {
            revision: PALS_V5_COORDINATOR_REVISION.into(),
            policy: PalsCoordinatorPolicyV5::CoordinatorFixed,
            slices: PalsCpuSlicesV5::default(),
            costs: PalsAdaptiveCostsV5::default(),
            physical_gpu_workers: 1,
            physical_gpu_concurrency: 1,
            inference_batch_max: 1,
            batch_wait_micros: 0,
            manager_executes_blocking_work: false,
            fixed_root_generation: true,
            per_question_ticket: true,
            max_repairs_per_first_move: 3,
            pending_repairs_max: 64,
            replay_history_per_namespace_max: PALS_V5_REPLAY_HISTORY_PER_NAMESPACE_MAX,
            frame_genealogy_max: PALS_V5_FRAME_GENEALOGY_MAX,
            validator_source_contexts_max: PALS_V5_VALIDATOR_SOURCE_CONTEXTS_MAX,
            validator_action_keys_max: PALS_V5_VALIDATOR_ACTION_KEYS_MAX,
        }
    }
}
impl PalsCoordinatorV5 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.slices.validate()?;
        self.costs.validate()?;
        require(
            self.revision == PALS_V5_COORDINATOR_REVISION
                && self.physical_gpu_workers == 1
                && self.physical_gpu_concurrency == 1
                && self.inference_batch_max == 1
                && self.batch_wait_micros == 0
                && (self.policy == PalsCoordinatorPolicyV5::Synchronous
                    || !self.manager_executes_blocking_work)
                && self.fixed_root_generation
                && self.per_question_ticket
                && self.max_repairs_per_first_move == 3
                && self.pending_repairs_max == 64
                && self.replay_history_per_namespace_max
                    == PALS_V5_REPLAY_HISTORY_PER_NAMESPACE_MAX
                && self.frame_genealogy_max == PALS_V5_FRAME_GENEALOGY_MAX
                && self.validator_source_contexts_max == PALS_V5_VALIDATOR_SOURCE_CONTEXTS_MAX
                && self.validator_action_keys_max == PALS_V5_VALIDATOR_ACTION_KEYS_MAX,
            "coordinator must preserve question authority and single immediate GPU worker",
        )
    }
}

/// Planned CPU placements grant no claim that affinity, SMT isolation or quotas
/// were applied. Those facts are separately observed in the receipt.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsResourcePlanV5 {
    pub profile_id: String,
    pub evidence_profile: PalsResourceEvidenceProfileV5,
    pub topology: PalsTopologyPolicyV5,
    pub cpu_quota_millis: u32,
    pub allowed_logical_cpus: Vec<u32>,
    pub dedicated_manager_cpu: Option<u32>,
    pub worker_logical_cpus: Vec<u32>,
    pub ort_intra_threads: u32,
    pub ort_inter_threads: u32,
    pub ort_spinning: bool,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
    pub gpu_model: Option<String>,
    pub gpu_count: u32,
    pub caller_threads_max: u32,
    pub kernel_threads_max: u32,
    pub cloud_or_topology_unknown: bool,
}
impl Default for PalsResourcePlanV5 {
    fn default() -> Self {
        Self {
            profile_id: "shared-control-cpu2-v1".into(),
            evidence_profile: PalsResourceEvidenceProfileV5::StrictLocal,
            topology: PalsTopologyPolicyV5::SharedControl,
            cpu_quota_millis: 2000,
            allowed_logical_cpus: Vec::new(),
            dedicated_manager_cpu: None,
            worker_logical_cpus: Vec::new(),
            ort_intra_threads: 1,
            ort_inter_threads: 1,
            ort_spinning: false,
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
            gpu_model: None,
            gpu_count: 0,
            caller_threads_max: 1,
            kernel_threads_max: 128,
            cloud_or_topology_unknown: true,
        }
    }
}
impl PalsResourcePlanV5 {
    pub fn validate(&self, slices: &PalsCpuSlicesV5) -> Result<(), ManifestError> {
        require(
            text(&self.profile_id)
                && (1000..=8000).contains(&self.cpu_quota_millis)
                && self.ort_intra_threads == 1
                && self.ort_inter_threads == 1
                && !self.ort_spinning
                && self.memory_high_bytes == 6 << 30
                && self.memory_max_bytes == 12 << 30
                && (self.gpu_count == 0 || self.gpu_count == 1)
                && (self.gpu_count == 1) == self.gpu_model.is_some()
                && self.gpu_model.as_ref().is_none_or(|s| text(s))
                && self.caller_threads_max == 1
                && (8..=1024).contains(&self.kernel_threads_max)
                && unique_cpus(&self.allowed_logical_cpus)
                && unique_cpus(&self.worker_logical_cpus),
            "invalid declared resources or ORT thread/spinning policy",
        )?;
        require(
            slices.workers == 1
                || (self.cpu_quota_millis > 2000 && self.profile_id != "shared-control-cpu2-v1"),
            "2/4 workers require a separately registered resource profile",
        )?;
        require(
            self.evidence_profile != PalsResourceEvidenceProfileV5::CloudObserved
                || (self.topology == PalsTopologyPolicyV5::SharedControl
                    && self.profile_id != "shared-control-cpu2-v1"),
            "cloud-observed resources require a separate SharedControl resource profile",
        )?;
        require(
            self.worker_logical_cpus
                .iter()
                .all(|c| self.allowed_logical_cpus.contains(c)),
            "worker placement lies outside declared allowed CPUs",
        )?;
        match self.topology {
            PalsTopologyPolicyV5::SharedControl => require(
                self.dedicated_manager_cpu.is_none(),
                "shared control cannot declare a dedicated manager",
            ),
            PalsTopologyPolicyV5::DedicatedControl => require(
                !self.cloud_or_topology_unknown
                    && self.dedicated_manager_cpu.is_some_and(|c| {
                        self.allowed_logical_cpus.contains(&c)
                            && !self.worker_logical_cpus.contains(&c)
                    })
                    && self.worker_logical_cpus.len() >= slices.workers as usize,
                "dedicated control requires known topology and explicit disjoint placement",
            ),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsExternalCpuV5 {
    pub binary: ArtifactRef,
    pub profile: ArtifactRef,
    pub source_commit: String,
    pub version: String,
    pub license: String,
    pub options: BTreeMap<String, String>,
    pub quantum_min_ms: u64,
    pub stop_drain_reserve_ms: u64,
    pub requested_nodes_max: u64,
    pub conditional_multipv_max: u32,
    pub reported_nodes_are_native_credit: bool,
    pub stop_bestmove_readyok_before_next_go: bool,
    pub exact_stack_resume: bool,
    pub sigstop: bool,
}
impl PalsExternalCpuV5 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.binary.validate()?;
        self.profile.validate()?;
        let expected = BTreeMap::from([
            ("Skill Level".into(), "20".into()),
            ("UCI_LimitStrength".into(), "false".into()),
            ("Threads".into(), "1".into()),
            ("Hash".into(), "256".into()),
            ("Ponder".into(), "false".into()),
            ("NumaPolicy".into(), "none".into()),
            ("SyzygyPath".into(), String::new()),
            ("MultiPV".into(), "1".into()),
        ]);
        require(
            self.source_commit == PALS_V5_STOCKFISH_SOURCE
                && self.version == "Stockfish 19"
                && self.license == "GPL-3.0-or-later"
                && self.binary.license == self.license
                && self.options == expected
                && self.quantum_min_ms == 100
                && self.stop_drain_reserve_ms == 500
                && self.requested_nodes_max == 65536
                && self.conditional_multipv_max == 4
                && !self.reported_nodes_are_native_credit
                && self.stop_bestmove_readyok_before_next_go
                && !self.exact_stack_resume
                && !self.sigstop,
            "Stockfish Full options, pin, bounded stop/drain or foreign-credit semantics differ",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCpuEndpointsV5 {
    pub cpu_t: PalsCpuSelectionV5,
    pub cpu_r: PalsCpuSelectionV5,
    pub own: PalsComponentV3,
    pub external: Option<PalsExternalCpuV5>,
}
impl PalsCpuEndpointsV5 {
    pub fn validate(&self, mode: PalsEngineModeV5) -> Result<(), ManifestError> {
        shared::component(&self.own)?;
        require(
            self.cpu_r != PalsCpuSelectionV5::Disabled,
            "CPU_R must select an actual checker",
        )?;
        let selected = self.cpu_t == PalsCpuSelectionV5::Stockfish19Full
            || self.cpu_r == PalsCpuSelectionV5::Stockfish19Full;
        require(
            selected == self.external.is_some(),
            "external identity must be present exactly when selected",
        )?;
        if let Some(external) = &self.external {
            external.validate()?;
        }
        match mode {
            PalsEngineModeV5::Collection => require(
                self.cpu_t == PalsCpuSelectionV5::Stockfish19Full,
                "real collection requires declared Stockfish 19 CPU_T",
            ),
            PalsEngineModeV5::Deployment => require(
                self.cpu_t == PalsCpuSelectionV5::Disabled,
                "deployment cannot run CPU_T",
            ),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsModelSpecificationV5 {
    pub profile: String,
    pub model_schema: String,
    pub model_semantics: String,
    pub encoding_schema: String,
    pub semantic_sha256: String,
    pub encoding_sha256: String,
    pub implementation_sha256: String,
    pub weights: ArtifactRef,
    pub artifact_domain: PalsArtifactDomainV5,
    pub provenance: Option<ArtifactRef>,
    pub checkpoint_domain: Option<String>,
    pub epoch: u64,
    pub backend: PalsModelBackendV3,
    pub precision: PalsPrecisionV3,
    pub tf32: bool,
    pub exported_roles: BTreeSet<PalsRoleV3>,
    pub width: u32,
    pub latent_count: u32,
    pub temporal_width: u32,
    pub candidate_head_width: u32,
    pub record_feature_axes: u32,
    pub query_feature_axes: u32,
    pub max_records: u32,
    pub max_anchors: u32,
    pub line_buckets: Vec<u32>,
    pub one_dynamic_line_graph: bool,
    pub ids_as_numeric_features: bool,
    pub fresh_warm_separate_domains: bool,
}
impl PalsModelSpecificationV5 {
    pub fn validate(&self, mode: PalsEngineModeV5) -> Result<(), ManifestError> {
        self.weights.validate()?;
        if let Some(p) = &self.provenance {
            p.validate()?;
        }
        let roles = match mode {
            PalsEngineModeV5::Collection => BTreeSet::from([
                PalsRoleV3::Proposer,
                PalsRoleV3::Critic,
                PalsRoleV3::Validator,
            ]),
            PalsEngineModeV5::Deployment => {
                BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic])
            }
        };
        let compact = self.profile == "virtual_review_v1";
        let legacy = self.profile == "full_line_interaction_v2";
        require(
            (compact || legacy)
                && self.model_schema
                    == if compact {
                        PALS_V5_MODEL_SCHEMA
                    } else {
                        "rovezero.pals-model.v2"
                    }
                && self.model_semantics
                    == if compact {
                        PALS_V5_MODEL_SEMANTICS
                    } else {
                        "rovezero.pals-model-semantics.v2"
                    }
                && self.encoding_schema
                    == if compact {
                        PALS_V5_ENCODING_SCHEMA
                    } else {
                        "rovezero.pals-board-records.v2"
                    }
                && [
                    &self.semantic_sha256,
                    &self.encoding_sha256,
                    &self.implementation_sha256,
                ]
                .iter()
                .all(|s| sha(s, 64))
                && self.exported_roles == roles
                && self.precision == PalsPrecisionV3::Fp32
                && !self.tf32
                && self.width == 384
                && self.latent_count == 16
                && self.temporal_width == 64
                && self.candidate_head_width == 128
                && self.record_feature_axes == if compact { 8 } else { 16 }
                && self.query_feature_axes == if compact { 8 } else { 16 }
                && self.max_records == if compact { 64 } else { 128 }
                && self.max_anchors == if compact { 4 } else { 1 }
                && self.line_buckets
                    == if compact {
                        PALS_V5_LINE_BUCKETS.to_vec()
                    } else {
                        vec![256]
                    }
                && self.one_dynamic_line_graph == compact
                && !self.ids_as_numeric_features
                && self.fresh_warm_separate_domains,
            "compact model semantics, roles, FP32 or single dynamic-L graph differs",
        )?;
        match self.artifact_domain {
            PalsArtifactDomainV5::Untrained => require(
                self.checkpoint_domain.is_none(),
                "untrained asset cannot claim a training checkpoint",
            ),
            PalsArtifactDomainV5::Diagnostic => require(
                self.provenance.is_some()
                    && self.checkpoint_domain.as_deref()
                        == Some(PALS_V5_DIAGNOSTIC_CHECKPOINT_DOMAIN),
                "diagnostic provenance/checkpoint domain missing",
            ),
            PalsArtifactDomainV5::RealGeneration => require(
                self.provenance.is_some()
                    && self.checkpoint_domain.as_deref()
                        == Some(PALS_V5_TRAINING_CHECKPOINT_DOMAIN),
                "real-generation provenance/checkpoint domain missing",
            ),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsEngineV5 {
    pub id: String,
    pub binary: ArtifactRef,
    pub source_commit: String,
    pub mode: PalsEngineModeV5,
    pub model: PalsModelSpecificationV5,
    pub cpu: PalsCpuEndpointsV5,
    pub context: PalsReviewContextV5,
    pub coordinator: PalsCoordinatorV5,
    pub resources: PalsResourcePlanV5,
    pub search: PalsComponentV3,
    pub runtime: PalsComponentV3,
    pub v_generate: bool,
    pub v_load: bool,
    pub v_call: bool,
}
impl PalsEngineV5 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.binary.validate()?;
        require(
            text(&self.id) && sha(&self.source_commit, 40),
            "invalid engine/source pin",
        )?;
        self.model.validate(self.mode)?;
        self.cpu.validate(self.mode)?;
        self.context.validate()?;
        require(
            match self.context.selection {
                PalsViewSelectionV5::VirtualReview => self.model.profile == "virtual_review_v1",
                PalsViewSelectionV5::V2Full | PalsViewSelectionV5::V2Selected => {
                    self.model.profile == "full_line_interaction_v2"
                        && self.context.anchors_max == 1
                }
            },
            "view selection must retain exact legacy/compact input semantics",
        )?;
        self.coordinator.validate()?;
        self.resources.validate(&self.coordinator.slices)?;
        shared::component(&self.search)?;
        shared::component(&self.runtime)?;
        require(
            match self.mode {
                PalsEngineModeV5::Collection => self.v_generate && self.v_load && self.v_call,
                PalsEngineModeV5::Deployment => !self.v_generate && !self.v_load && !self.v_call,
            },
            "V must be enabled for full collection and absent from deployment",
        )?;
        require(
            self.model.backend != PalsModelBackendV3::OrtCuda || self.resources.gpu_count == 1,
            "CUDA model requires a separately declared GPU",
        )
    }
}

/// First generation registration, not a learning invocation. Numeric decimal
/// strings have one canonical spelling and avoid float-based recipe equality.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsGenerationRecipeV5 {
    pub recipe: String,
    pub checkpoint_domain: String,
    pub record_domain: String,
    pub collection_games: u32,
    pub collection_openings: u32,
    pub train_openings: u32,
    pub validation_openings: u32,
    pub holdout_openings: u32,
    pub p_updates: u32,
    pub c_updates: u32,
    pub v_updates: u32,
    pub v_action_contexts_target: u32,
    pub v_action_contexts_max: u32,
    pub pc_unique_valid_samples_min: u32,
    pub v_known_pairs_min: u32,
    pub v_witness_contexts_min: u32,
    pub evaluation_pairs: u32,
    pub initialization_seed: u64,
    pub collection_seed: u64,
    pub training_seed: u64,
    pub evaluation_seed: u64,
    pub microbatch: u32,
    pub accumulation: u32,
    pub training_cpu_host: u32,
    pub training_gpu: String,
    pub training_gpu_count: u32,
    pub training_precision: PalsPrecisionV3,
    pub training_tf32: bool,
    pub policy_loss_weight: u32,
    pub wdl_loss_weight: u32,
    pub divergence_loss_weight: u32,
    pub v_pairwise_loss_weight: u32,
    pub learning_rate: String,
    pub beta1: String,
    pub beta2: String,
    pub epsilon: String,
    pub weight_decay: String,
    pub gradient_clip: String,
    pub schedule: String,
    pub checkpoint_every_updates: u32,
    pub shared_optimizer_owners: u32,
    pub pc_alternating_updates: bool,
    pub v_private_only: bool,
    pub valid_head_count_normalization: bool,
    pub all_masked_is_update: bool,
    pub wdl_rules_trajectory_only: bool,
    pub teacher_policy: String,
    pub v_loss: String,
    pub real_witness_only: bool,
    pub diagnostic_assets_promotable: bool,
    pub local_staging_bytes_max: u64,
}
impl Default for PalsGenerationRecipeV5 {
    fn default() -> Self {
        Self {
            recipe: PALS_V5_FIRST_GENERATION.into(),
            checkpoint_domain: PALS_V5_TRAINING_CHECKPOINT_DOMAIN.into(),
            record_domain: PALS_V5_TRAINING_RECORD_DOMAIN.into(),
            collection_games: 128,
            collection_openings: 64,
            train_openings: 48,
            validation_openings: 8,
            holdout_openings: 8,
            p_updates: 2000,
            c_updates: 2000,
            v_updates: 1000,
            v_action_contexts_target: 2048,
            v_action_contexts_max: 8192,
            pc_unique_valid_samples_min: 1024,
            v_known_pairs_min: 128,
            v_witness_contexts_min: 32,
            evaluation_pairs: 32,
            initialization_seed: 23,
            collection_seed: 29,
            training_seed: 31,
            evaluation_seed: 1,
            microbatch: 1,
            accumulation: 8,
            training_cpu_host: 2,
            training_gpu: "NVIDIA L4".into(),
            training_gpu_count: 1,
            training_precision: PalsPrecisionV3::Fp32,
            training_tf32: false,
            policy_loss_weight: 1,
            wdl_loss_weight: 1,
            divergence_loss_weight: 1,
            v_pairwise_loss_weight: 1,
            learning_rate: "1e-4".into(),
            beta1: "0.9".into(),
            beta2: "0.999".into(),
            epsilon: "1e-8".into(),
            weight_decay: "0.01".into(),
            gradient_clip: "1.0".into(),
            schedule: "constant".into(),
            checkpoint_every_updates: 250,
            shared_optimizer_owners: 1,
            pc_alternating_updates: true,
            v_private_only: true,
            valid_head_count_normalization: true,
            all_masked_is_update: false,
            wdl_rules_trajectory_only: true,
            teacher_policy: "completed-conditional-multipv-softmax-negative-rank".into(),
            v_loss: "known-pair-pareto-witness-wall-cost-logistic".into(),
            real_witness_only: true,
            diagnostic_assets_promotable: false,
            local_staging_bytes_max: 18 << 30,
        }
    }
}
impl PalsGenerationRecipeV5 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self == &Self::default(),
            "first real generation recipe differs from registration",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPairPlanV5 {
    pub opening_id: String,
    pub position_command: String,
    pub white_order: [String; 2],
    pub base_ms: u64,
    pub increment_ms: u64,
    pub max_plies: u32,
    pub seed: u64,
    pub concurrent_games: u32,
    pub restart_processes_each_game: bool,
    pub ponder: bool,
    pub score_adjudication: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsExecutionWindowV5 {
    pub window_id: String,
    pub wall_time_max_ms: u64,
    pub cleanup_max_ms: u64,
    pub output_bytes_max: u64,
    pub agent_renewable: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRunManifestV5 {
    pub schema_version: u32,
    pub run_id: String,
    pub contract_revision: String,
    pub purpose: PalsRunPurposeV5,
    pub comparison: PalsComparisonV5,
    pub rules_profile: String,
    /// Immutable source/data/opening pins. Actual file verification is separate.
    pub source_pins: BTreeMap<String, String>,
    pub engines: Vec<PalsEngineV5>,
    pub pair: Option<PalsPairPlanV5>,
    pub generation: Option<PalsGenerationRecipeV5>,
    pub window: PalsExecutionWindowV5,
    /// Explicit authorization gate: recipe presence is never training consent.
    pub training_authorized: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsInputLockV5 {
    pub domain: String,
    pub manifest: PalsRunManifestV5,
    pub canonical_sha256: String,
}

impl PalsRunManifestV5 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.schema_version == 5 && self.contract_revision == PALS_V5_CONTRACT_REVISION,
            "schema/revision mismatch; legacy formats are never converted",
        )?;
        require(
            text(&self.run_id)
                && text(&self.rules_profile)
                && (1..=2).contains(&self.engines.len())
                && self
                    .engines
                    .iter()
                    .map(|e| &e.id)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.engines.len()
                && (1..=64).contains(&self.source_pins.len())
                && self.source_pins.iter().all(|(k, v)| text(k) && sha(v, 64)),
            "invalid run/Rules/source identity or duplicate engines",
        )?;
        for e in &self.engines {
            e.validate()?;
        }
        validate_source_components_v5(self)?;
        require(
            text(&self.window.window_id)
                && (1..=86_400_000).contains(&self.window.wall_time_max_ms)
                && (1..=30_000).contains(&self.window.cleanup_max_ms)
                && (1..=64 << 30).contains(&self.window.output_bytes_max)
                && self.window.agent_renewable,
            "execution window must remain finite, separately identified and agent-renewable",
        )?;
        if let Some(recipe) = &self.generation {
            recipe.validate()?;
        }
        let pair_purpose = matches!(
            self.purpose,
            PalsRunPurposeV5::CollectionPair | PalsRunPurposeV5::EvaluationPair
        );
        require(
            pair_purpose == self.pair.is_some(),
            "pair declaration differs from execution purpose",
        )?;
        if let Some(pair) = &self.pair {
            require(
                self.engines.len() == 2
                    && text(&pair.opening_id)
                    && text(&pair.position_command)
                    && pair.position_command.starts_with("position ")
                    && pair.white_order[0] != pair.white_order[1]
                    && pair
                        .white_order
                        .iter()
                        .all(|id| self.engines.iter().any(|e| &e.id == id))
                    && pair.base_ms == 120_000
                    && pair.increment_ms == 1000
                    && pair.concurrent_games == 1
                    && pair.restart_processes_each_game
                    && !pair.ponder
                    && !pair.score_adjudication,
                "pair requires exact opening identity, exchanged colors, Fisher clocks and process restart",
            )?;
            let (mode, max_plies) = if self.purpose == PalsRunPurposeV5::CollectionPair {
                (PalsEngineModeV5::Collection, 512)
            } else {
                (PalsEngineModeV5::Deployment, 256)
            };
            require(
                pair.max_plies == max_plies && self.engines.iter().all(|e| e.mode == mode),
                "collection/evaluation role mode or ply ceiling differs",
            )?;
        }
        if self.purpose == PalsRunPurposeV5::TrainingGeneration {
            require(
                self.training_authorized
                    && self.generation.is_some()
                    && self
                        .engines
                        .iter()
                        .all(|e| e.mode == PalsEngineModeV5::Collection),
                "training requires explicit authorization and registered full collection configuration",
            )?;
        } else {
            require(
                !self.training_authorized,
                "non-training execution cannot authorize optimizer updates",
            )?;
        }
        if matches!(
            self.purpose,
            PalsRunPurposeV5::CollectionPair | PalsRunPurposeV5::ActionCollection
        ) {
            require(
                self.generation.is_some()
                    && self.engines.iter().all(|e| {
                        e.mode == PalsEngineModeV5::Collection
                            && e.model.artifact_domain != PalsArtifactDomainV5::Diagnostic
                    }),
                "real collection requires generation recipe and full P/C/V mode",
            )?;
        }
        if self.purpose == PalsRunPurposeV5::TrainingGeneration {
            require(
                self.engines
                    .iter()
                    .all(|e| e.model.artifact_domain != PalsArtifactDomainV5::Diagnostic),
                "diagnostic initialization/checkpoints cannot become real-generation inputs",
            )?;
        }
        if self.purpose == PalsRunPurposeV5::EvaluationPair {
            require(
                self.engines.iter().all(|e| {
                    e.model.artifact_domain != PalsArtifactDomainV5::Diagnostic
                        && e.cpu.cpu_r == PalsCpuSelectionV5::Own
                }),
                "diagnostic assets or CPU_T/foreign CPU_R cannot enter final own-CPU_R evaluation",
            )?;
        }
        self.validate_comparison()?;
        self.unique_input_bytes()?;
        Ok(())
    }
    fn validate_comparison(&self) -> Result<(), ManifestError> {
        if self.comparison == PalsComparisonV5::System {
            return Ok(());
        }
        require(
            self.engines.len() == 2,
            "controlled comparison requires two registered endpoints",
        )?;
        let (a, b) = (&self.engines[0], &self.engines[1]);
        require(
            a.mode == b.mode && a.resources == b.resources,
            "controlled comparison requires equal mode and resources",
        )?;
        match self.comparison {
            PalsComparisonV5::SynchronousToFixed | PalsComparisonV5::FixedToAdaptive => {
                let expected = if self.comparison == PalsComparisonV5::SynchronousToFixed {
                    (
                        PalsCoordinatorPolicyV5::Synchronous,
                        PalsCoordinatorPolicyV5::CoordinatorFixed,
                    )
                } else {
                    (
                        PalsCoordinatorPolicyV5::CoordinatorFixed,
                        PalsCoordinatorPolicyV5::CoordinatorAdaptive,
                    )
                };
                let mut left = a.coordinator.clone();
                left.policy = b.coordinator.policy;
                if self.comparison == PalsComparisonV5::SynchronousToFixed {
                    left.manager_executes_blocking_work =
                        b.coordinator.manager_executes_blocking_work;
                }
                require(
                    a.coordinator.policy == expected.0
                        && b.coordinator.policy == expected.1
                        && left == b.coordinator
                        && a.model == b.model
                        && a.cpu == b.cpu
                        && a.context == b.context
                        && a.search == b.search
                        && a.runtime == b.runtime,
                    "coordinator comparison changed model/context/CPU/resource policy",
                )
            }
            PalsComparisonV5::InternalWeights => {
                let mut left = a.model.clone();
                left.weights = b.model.weights.clone();
                left.provenance = b.model.provenance.clone();
                left.artifact_domain = b.model.artifact_domain;
                left.checkpoint_domain = b.model.checkpoint_domain.clone();
                left.epoch = b.model.epoch;
                require(
                    a.model.weights != b.model.weights
                        && left == b.model
                        && a.cpu == b.cpu
                        && a.context == b.context
                        && a.coordinator == b.coordinator
                        && a.search == b.search
                        && a.runtime == b.runtime,
                    "weights comparison changed another model/search/runtime condition",
                )
            }
            PalsComparisonV5::CpuRConnection => {
                let mut left = a.cpu.clone();
                left.cpu_r = b.cpu.cpu_r;
                left.external = b.cpu.external.clone();
                require(
                    a.cpu.cpu_r != b.cpu.cpu_r
                        && left == b.cpu
                        && a.model == b.model
                        && a.context == b.context
                        && a.coordinator == b.coordinator
                        && a.search == b.search
                        && a.runtime == b.runtime,
                    "CPU_R connection comparison changed another condition",
                )
            }
            PalsComparisonV5::V2FullToSelected => {
                let mut left = a.context.clone();
                left.selection = b.context.selection;
                left.selected_records_max = b.context.selected_records_max;
                require(
                    a.context.selection == PalsViewSelectionV5::V2Full
                        && b.context.selection == PalsViewSelectionV5::V2Selected
                        && left == b.context
                        && a.model == b.model
                        && a.cpu == b.cpu
                        && a.coordinator == b.coordinator
                        && a.search == b.search
                        && a.runtime == b.runtime,
                    "V2 view selection comparison changed weights/model/CPU/coordinator",
                )
            }
            PalsComparisonV5::V2ToCompact => {
                let mut left = a.context.clone();
                left.selection = b.context.selection;
                left.anchors_max = b.context.anchors_max;
                require(
                    a.model.profile == "full_line_interaction_v2"
                        && b.model.profile == "virtual_review_v1"
                        && a.context.selection == PalsViewSelectionV5::V2Selected
                        && b.context.selection == PalsViewSelectionV5::VirtualReview
                        && left == b.context
                        && a.cpu == b.cpu
                        && a.coordinator == b.coordinator
                        && a.search == b.search
                        && a.runtime == b.runtime,
                    "V2/compact comparison must declare whole model composition while fixing token/selection limits and CPU/search/runtime",
                )
            }
            PalsComparisonV5::System => Ok(()),
        }
    }
    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut result = Vec::new();
        for engine in &self.engines {
            result.extend([&engine.binary, &engine.model.weights]);
            if let Some(p) = &engine.model.provenance {
                result.push(p);
            }
            if let Some(e) = &engine.cpu.external {
                result.extend([&e.binary, &e.profile]);
            }
        }
        result
    }
    pub fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        let mut unique = BTreeMap::new();
        for artifact in self.declared_artifacts() {
            artifact.validate()?;
            if let Some(previous) = unique.insert(&artifact.path, artifact) {
                require(
                    previous == artifact,
                    "one logical artifact path has conflicting identity",
                )?;
            }
        }
        unique
            .values()
            .try_fold(0_u64, |sum, a| sum.checked_add(a.bytes))
            .ok_or_else(|| ManifestError::Integrity("PALS V5 artifact byte overflow".into()))
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        encoded(&(PALS_MANIFEST_V5_DOMAIN, self), false)
    }
    pub fn lock(&self) -> Result<PalsInputLockV5, ManifestError> {
        Ok(PalsInputLockV5 {
            domain: PALS_MANIFEST_V5_DOMAIN.into(),
            manifest: self.clone(),
            canonical_sha256: digest(&self.canonical_bytes()?),
        })
    }
}
impl PalsInputLockV5 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.verify()?;
        Ok(value)
    }
    pub fn verify(&self) -> Result<(), ManifestError> {
        require(
            self.domain == PALS_MANIFEST_V5_DOMAIN
                && sha(&self.canonical_sha256, 64)
                && digest(&self.manifest.canonical_bytes()?) == self.canonical_sha256,
            "lock domain/digest mismatch",
        )
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        self.verify()?;
        json_string(self)
    }
}

/// A separately produced verification report. Declaration of source pins does
/// not establish that these source bytes or installed binaries were verified.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSourceVerificationV5 {
    pub components: BTreeMap<String, String>,
    pub binary_sha256_by_endpoint: BTreeMap<String, String>,
    pub content_verified: bool,
    pub evidence: ArtifactRef,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCoordinatorObservationV5 {
    /// Peaks remain valid after shutdown clears resident frame state. These
    /// count retained root-lifetime identifiers rather than current queue size.
    pub role_request_history_peak: u32,
    pub cpu_execution_history_peak: u32,
    pub frame_genealogy_peak: u32,
    pub ranking_fanout_history_peak: u32,
    /// Read only from the retained V replay guard's actual peak getter. Unknown
    /// means unobserved, including V-free sessions without that observation.
    pub validator_source_contexts_peak: PalsObservedV3<u32>,
    pub validator_action_keys_peak: PalsObservedV3<u32>,
    pub view_submitted: u64,
    pub eligible_view_consumed: u64,
    pub tickets_submitted: u64,
    pub tickets_terminal_before_dispatch: u64,
    pub tickets_cache_completed: u64,
    pub tickets_dispatched: u64,
    pub tickets_queued: u64,
    pub physical_in_flight: u64,
    pub physical_completion_unknown: u64,
    pub tickets_physically_completed: u64,
    pub tickets_accepted: u64,
    pub tickets_released: u64,
    pub tickets_quarantined: u64,
    pub nn_inputs_completed: u64,
    pub nn_inputs_accepted: u64,
    /// Accepted cached-evaluation consumption opportunities, not unique entries.
    pub cache_evaluations_accepted: u64,
    pub search_evaluations_consumed: u64,
    pub unused_results: u64,
    pub cpu_credit_limit: u64,
    pub cpu_spent: u64,
    pub cpu_outstanding_reserved: u64,
    pub yield_requests: u64,
    pub yield_acks: u64,
    pub natural_completions: u64,
    pub physical_state: PalsPhysicalStateV3,
    pub owner_joined: bool,
    pub archive_closed: bool,
    pub evidence: ArtifactRef,
}
impl PalsCoordinatorObservationV5 {
    fn validate(&self) -> Result<(), ManifestError> {
        self.evidence.validate()?;
        shared::observation(&self.validator_source_contexts_peak)?;
        shared::observation(&self.validator_action_keys_peak)?;
        if let (Some(sources), Some(actions)) = (
            observed(&self.validator_source_contexts_peak),
            observed(&self.validator_action_keys_peak),
        ) {
            require(
                u64::from(*actions) <= u64::from(*sources) * 7,
                "V action replay exceeds its seven-action source vocabulary",
            )?;
        }
        require(
            self.ranking_fanout_history_peak <= self.frame_genealogy_peak
                && self.eligible_view_consumed <= self.view_submitted
                && checked_sum(&[
                    self.tickets_terminal_before_dispatch,
                    self.tickets_cache_completed,
                    self.tickets_dispatched,
                    self.tickets_queued,
                ]) == Some(self.tickets_submitted)
                && checked_sum(&[
                    self.tickets_physically_completed,
                    self.physical_in_flight,
                    self.physical_completion_unknown,
                ]) == Some(self.tickets_dispatched)
                && checked_sum(&[
                    self.tickets_physically_completed,
                    self.tickets_cache_completed,
                ])
                .is_some_and(|n| self.tickets_accepted <= n)
                && checked_sum(&[
                    self.tickets_physically_completed,
                    self.tickets_cache_completed,
                    self.tickets_terminal_before_dispatch,
                ])
                .is_some_and(|n| self.tickets_released <= n)
                && self.tickets_quarantined == self.physical_completion_unknown
                && self.nn_inputs_accepted <= self.nn_inputs_completed
                && self
                    .nn_inputs_accepted
                    .checked_add(self.cache_evaluations_accepted)
                    .is_some_and(|n| self.search_evaluations_consumed <= n)
                && self.yield_acks <= self.yield_requests
                && self
                    .cpu_spent
                    .checked_add(self.cpu_outstanding_reserved)
                    .is_some_and(|n| n <= self.cpu_credit_limit),
            "coordinator ticket/view/credit accounting violates authority or reservations",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsEndpointReceiptV5 {
    pub endpoint_id: String,
    pub runtime_owner: String,
    pub model_verification: PalsObservedV3<PalsModelVerificationV5>,
    pub resource_readback: PalsObservedV3<PalsResourceReadbackV5>,
    pub coordinator: PalsObservedV3<PalsCoordinatorObservationV5>,
    pub native_numerical_check: PalsObservedV3<PalsNativeNumericalCheckV5>,
    pub process_tree_gone: PalsObservedV3<bool>,
    pub host_peak_bytes: PalsObservedV3<u64>,
    pub vram_peak_bytes: PalsObservedV3<u64>,
    pub actual_tf32: PalsObservedV3<bool>,
    pub v_generated: PalsObservedV3<bool>,
    pub v_loaded: PalsObservedV3<bool>,
    pub v_called: PalsObservedV3<bool>,
}

/// Independent artifact/content verification binds provenance to actual
/// weights. A manifest-only domain relabel cannot satisfy accepted execution.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsModelVerificationV5 {
    pub weights: ArtifactRef,
    pub provenance: Option<ArtifactRef>,
    pub artifact_domain: PalsArtifactDomainV5,
    pub checkpoint_domain: Option<String>,
    pub semantic_sha256: String,
    pub encoding_sha256: String,
    pub profile: String,
    pub exported_roles: BTreeSet<PalsRoleV3>,
    pub backend: PalsModelBackendV3,
    pub precision: PalsPrecisionV3,
    pub content_domain_verified: bool,
    pub diagnostic_asset: bool,
    pub evidence: ArtifactRef,
}

impl PalsEndpointReceiptV5 {
    /// A producer starts with unknown facts. Declarations never populate an
    /// observed resource/model/physical state or enable eligibility here.
    pub fn with_unknown_checks(endpoint_id: String, runtime_owner: String) -> Self {
        Self {
            endpoint_id,
            runtime_owner,
            model_verification: PalsObservedV3::Unknown,
            resource_readback: PalsObservedV3::Unknown,
            coordinator: PalsObservedV3::Unknown,
            native_numerical_check: PalsObservedV3::Unknown,
            process_tree_gone: PalsObservedV3::Unknown,
            host_peak_bytes: PalsObservedV3::Unknown,
            vram_peak_bytes: PalsObservedV3::Unknown,
            actual_tf32: PalsObservedV3::Unknown,
            v_generated: PalsObservedV3::Unknown,
            v_loaded: PalsObservedV3::Unknown,
            v_called: PalsObservedV3::Unknown,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNativeNumericalCheckV5 {
    pub weights_sha256: String,
    pub semantic_sha256: String,
    pub encoding_sha256: String,
    pub runtime_implementation_sha256: String,
    pub source_commit: String,
    pub passed: bool,
    pub logits_absolute_tolerance: String,
    pub logits_relative_tolerance: String,
    pub policy_wdl_absolute_tolerance: String,
    pub evidence: ArtifactRef,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsGameReceiptV5 {
    pub game_index: u32,
    pub white_endpoint: String,
    pub black_endpoint: String,
    pub base_ms: u64,
    pub increment_ms: u64,
    pub plies: u32,
    pub result: PalsResultV3,
    pub termination: PalsTerminationV3,
    pub failed_endpoint: Option<String>,
    pub pgn: ArtifactRef,
    pub rules_clock_pgn_check: PalsObservedV3<ArtifactRef>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsTrainingObservationV5 {
    pub checkpoint_domain: String,
    pub checkpoint: ArtifactRef,
    pub provenance: ArtifactRef,
    pub p_completed_updates: u32,
    pub c_completed_updates: u32,
    pub v_completed_updates: u32,
    pub p_unique_valid_samples: u32,
    pub c_unique_valid_samples: u32,
    pub v_known_pairs: u32,
    pub v_new_witness_contexts: u32,
    pub shared_optimizer_owners: u32,
    pub optimizer_update_boundary: bool,
    pub pc_frozen_before_v: bool,
    pub resume_exact_state: PalsObservedV3<bool>,
    pub collection_games_completed: u32,
    pub evaluation_pairs_completed: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRunReceiptV5 {
    pub schema_version: u32,
    pub domain: String,
    pub run_id: String,
    pub lock_sha256: String,
    pub source_verification: PalsObservedV3<PalsSourceVerificationV5>,
    pub wall_time_ms: u64,
    pub cleanup_time_ms: u64,
    pub output_bytes: PalsObservedV3<u64>,
    pub training_executed: bool,
    pub training: PalsObservedV3<PalsTrainingObservationV5>,
    pub accepted: bool,
    pub performance_eligible: bool,
    pub strength_eligible: bool,
    pub full_gc_requested: bool,
    pub memory_contention: bool,
    pub failures: BTreeSet<PalsFailureV5>,
    pub endpoints: Vec<PalsEndpointReceiptV5>,
    pub games: Vec<PalsGameReceiptV5>,
}
impl PalsRunReceiptV5 {
    pub fn pending(lock: &PalsInputLockV5) -> Result<Self, ManifestError> {
        lock.verify()?;
        Ok(Self {
            schema_version: 5,
            domain: PALS_RECEIPT_V5_DOMAIN.into(),
            run_id: lock.manifest.run_id.clone(),
            lock_sha256: lock.canonical_sha256.clone(),
            source_verification: PalsObservedV3::Unknown,
            wall_time_ms: 0,
            cleanup_time_ms: 0,
            output_bytes: PalsObservedV3::Unknown,
            training_executed: false,
            training: PalsObservedV3::Unknown,
            accepted: false,
            performance_eligible: false,
            strength_eligible: false,
            full_gc_requested: false,
            memory_contention: false,
            failures: BTreeSet::new(),
            endpoints: Vec::new(),
            games: Vec::new(),
        })
    }
    pub fn from_json(input: &str, lock: &PalsInputLockV5) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.validate_against(lock)?;
        Ok(value)
    }
    pub fn validate_against(&self, lock: &PalsInputLockV5) -> Result<(), ManifestError> {
        lock.verify()?;
        let m = &lock.manifest;
        require(
            self.schema_version == 5
                && self.domain == PALS_RECEIPT_V5_DOMAIN
                && self.run_id == m.run_id
                && self.lock_sha256 == lock.canonical_sha256,
            "receipt schema/identity/lock differs",
        )?;
        shared::observation(&self.source_verification)?;
        if let Some(v) = observed(&self.source_verification) {
            v.evidence.validate()?;
            require(
                v.components.len() <= 64
                    && v.components.iter().all(|(k, v)| text(k) && sha(v, 64))
                    && v.binary_sha256_by_endpoint.len() <= 2
                    && v.binary_sha256_by_endpoint
                        .iter()
                        .all(|(k, v)| text(k) && sha(v, 64)),
                "invalid actual source/binary verification report",
            )?;
            let declared_binaries = m
                .engines
                .iter()
                .map(|e| (e.id.clone(), e.binary.sha256.clone()))
                .collect::<BTreeMap<_, _>>();
            let matches =
                v.components == m.source_pins && v.binary_sha256_by_endpoint == declared_binaries;
            require(
                !self.accepted || (v.content_verified && matches),
                "source declarations do not prove actual source/binary content verification",
            )?;
            require(
                matches
                    || (!self.accepted
                        && self
                            .failures
                            .contains(&PalsFailureV5::SourceIdentityViolation)),
                "actual source/binary mismatch requires a preserved identity failure",
            )?;
        } else {
            require(
                !self.accepted,
                "unknown source verification cannot authorize accepted execution",
            )?;
        }
        require(
            !self.training_executed
                || (m.training_authorized && m.purpose == PalsRunPurposeV5::TrainingGeneration),
            "actual training is unauthorized or attached to a non-training run",
        )?;
        require(
            !self.accepted
                || m.purpose != PalsRunPurposeV5::TrainingGeneration
                || self.training_executed,
            "training-purpose acceptance requires actual optimizer execution",
        )?;
        require(
            !self.accepted
                || (self.failures.is_empty()
                    && self.wall_time_ms <= m.window.wall_time_max_ms
                    && self.cleanup_time_ms <= m.window.cleanup_max_ms),
            "failed/expired cleanup cannot be accepted",
        )?;
        require(
            self.wall_time_ms <= m.window.wall_time_max_ms
                || self.failures.contains(&PalsFailureV5::Expired),
            "expired wall window requires explicit failure",
        )?;
        require(
            self.cleanup_time_ms <= m.window.cleanup_max_ms
                || self.failures.contains(&PalsFailureV5::ProcessCleanup),
            "cleanup deadline overshoot requires explicit failure",
        )?;
        require(
            !self.performance_eligible
                || (self.accepted && !self.full_gc_requested && !self.memory_contention),
            "Full GC/contention cannot support controlled performance evidence",
        )?;
        require(
            !self.strength_eligible
                || (self.accepted
                    && m.purpose == PalsRunPurposeV5::EvaluationPair
                    && self.games.len() == 2),
            "strength evidence requires actual accepted evaluation pair",
        )?;
        shared::observation(&self.output_bytes)?;
        let output = observed(&self.output_bytes);
        require(
            output.is_none_or(|v| *v <= m.window.output_bytes_max)
                || self.failures.contains(&PalsFailureV5::OutputQuota),
            "observed output overshoot requires explicit failure",
        )?;
        require(
            !self.accepted || output.is_some_and(|v| *v <= m.window.output_bytes_max),
            "accepted execution requires observed bounded output",
        )?;
        require(
            self.endpoints.len() <= 2
                && (!self.accepted || self.endpoints.len() == m.engines.len()),
            "endpoint receipt count differs",
        )?;
        let mut ids = BTreeSet::new();
        let mut owners = BTreeSet::new();
        for endpoint in &self.endpoints {
            require(
                ids.insert(&endpoint.endpoint_id)
                    && text(&endpoint.runtime_owner)
                    && owners.insert(&endpoint.runtime_owner),
                "duplicate endpoint/owner receipt",
            )?;
            let declared = m
                .engines
                .iter()
                .find(|e| e.id == endpoint.endpoint_id)
                .ok_or_else(|| {
                    ManifestError::Integrity("PALS V5 unknown receipt endpoint".into())
                })?;
            validate_endpoint_receipt(
                endpoint,
                declared,
                &EndpointAcceptanceContextV5::from(self),
            )?;
            if self.accepted && m.purpose == PalsRunPurposeV5::NeuralValidation {
                require(
                    observed(&endpoint.native_numerical_check).is_some(),
                    "neural acceptance requires independent numerical evidence",
                )?;
            }
        }
        let mut game_indices = BTreeSet::new();
        let mut pgn_bytes = 0_u64;
        require(
            self.games.len() <= 2 && (m.pair.is_some() || self.games.is_empty()),
            "non-pair run cannot attach game results",
        )?;
        for game in &self.games {
            let pair = m
                .pair
                .as_ref()
                .ok_or_else(|| ManifestError::Integrity("missing pair".into()))?;
            game.pgn.validate()?;
            shared::observation(&game.rules_clock_pgn_check)?;
            if let Some(evidence) = observed(&game.rules_clock_pgn_check) {
                evidence.validate()?;
            }
            require(
                game.game_index < 2
                    && game_indices.insert(game.game_index)
                    && game.white_endpoint == pair.white_order[game.game_index as usize]
                    && game.black_endpoint != game.white_endpoint
                    && m.engines.iter().any(|e| e.id == game.black_endpoint)
                    && game.base_ms == pair.base_ms
                    && game.increment_ms == pair.increment_ms
                    && game.plies <= pair.max_plies,
                "game identity/colors/clocks/plies differ from lock",
            )?;
            validate_game_result(game, pair)?;
            require(
                !self.accepted || observed(&game.rules_clock_pgn_check).is_some(),
                "accepted game lacks independent Rules/clock/PGN evidence",
            )?;
            pgn_bytes = pgn_bytes
                .checked_add(game.pgn.bytes)
                .ok_or_else(|| ManifestError::Integrity("PALS V5 PGN byte overflow".into()))?;
        }
        require(
            output.is_none_or(|n| *n >= pgn_bytes),
            "total output omits known PGN bytes",
        )?;
        require(
            !self.accepted
                || m.pair.is_none()
                || (self.games.len() == 2
                    && self
                        .games
                        .iter()
                        .all(|g| g.result != PalsResultV3::Incomplete)),
            "incomplete pair cannot be accepted",
        )?;
        shared::observation(&self.training)?;
        if let Some(training) = observed(&self.training) {
            require(
                self.training_executed,
                "training observation cannot be attached to unexecuted learning",
            )?;
            validate_training(training, m, self.accepted)?;
        } else {
            require(
                !self.training_executed || !self.accepted,
                "accepted actual training requires observed checkpoint/update evidence",
            )?;
        }
        Ok(())
    }
    pub fn canonical_bytes(&self, lock: &PalsInputLockV5) -> Result<Vec<u8>, ManifestError> {
        self.validate_against(lock)?;
        encoded(&(PALS_RECEIPT_V5_DOMAIN, self), false)
    }
    pub fn to_json(&self, lock: &PalsInputLockV5) -> Result<String, ManifestError> {
        self.validate_against(lock)?;
        json_string(self)
    }
}

/// Checks one actual process/session without inventing a completed run or any
/// games. True eligibility flags only select this endpoint's strict checks;
/// they never establish run-level source/binary verification, Rules/clock/PGN
/// validity, paired results, or a numerical acceptance for other sessions.
///
/// Native accepted sessions must have independently bound numerical evidence
/// and positive completed NN input work in this session. Mock/CPU declarations
/// and pending Unknown observations are not promoted to native acceptance.
pub fn validate_pals_endpoint_observation_v5(
    observation: &PalsEndpointReceiptV5,
    declared: &PalsEngineV5,
    accepted: bool,
    performance_eligible: bool,
    strength_eligible: bool,
) -> Result<(), ManifestError> {
    declared.validate()?;
    require(
        observation.endpoint_id == declared.id && text(&observation.runtime_owner),
        "standalone endpoint identity/owner differs from declaration",
    )?;
    require(
        (!performance_eligible && !strength_eligible) || accepted,
        "endpoint eligibility requires accepted session observations",
    )?;
    let failures = BTreeSet::new();
    let context = EndpointAcceptanceContextV5 {
        accepted,
        performance_eligible,
        strength_eligible,
        failures: &failures,
    };
    validate_endpoint_receipt(observation, declared, &context)?;
    if accepted && declared.model.backend != PalsModelBackendV3::DeterministicMock {
        require(
            observed(&observation.native_numerical_check).is_some_and(|value| value.passed)
                && observed(&observation.coordinator).is_some_and(|value| {
                    value.nn_inputs_completed > 0 && value.search_evaluations_consumed > 0
                }),
            "native endpoint acceptance requires bound numerical evidence and actual completed NN inputs",
        )?;
    }
    if accepted && declared.model.backend == PalsModelBackendV3::OrtCuda {
        require(
            observed(&observation.actual_tf32) == Some(&declared.model.tf32),
            "CUDA endpoint acceptance requires actual precision/TF32 observation",
        )?;
    }
    Ok(())
}

/// Non-persisted scope: only endpoint eligibility and the actual run's failures
/// affect the shared endpoint validator. No run ID, games or execution claims
/// are fabricated merely to invoke the accepted branch.
struct EndpointAcceptanceContextV5<'a> {
    accepted: bool,
    performance_eligible: bool,
    strength_eligible: bool,
    failures: &'a BTreeSet<PalsFailureV5>,
}
impl<'a> From<&'a PalsRunReceiptV5> for EndpointAcceptanceContextV5<'a> {
    fn from(run: &'a PalsRunReceiptV5) -> Self {
        Self {
            accepted: run.accepted,
            performance_eligible: run.performance_eligible,
            strength_eligible: run.strength_eligible,
            failures: &run.failures,
        }
    }
}

fn validate_endpoint_receipt(
    o: &PalsEndpointReceiptV5,
    d: &PalsEngineV5,
    run: &EndpointAcceptanceContextV5<'_>,
) -> Result<(), ManifestError> {
    shared::observation(&o.model_verification)?;
    if let Some(v) = observed(&o.model_verification) {
        v.weights.validate()?;
        v.evidence.validate()?;
        if let Some(p) = &v.provenance {
            p.validate()?;
        }
        let matches_declared = v.weights == d.model.weights
            && v.provenance == d.model.provenance
            && v.artifact_domain == d.model.artifact_domain
            && v.checkpoint_domain == d.model.checkpoint_domain
            && v.semantic_sha256 == d.model.semantic_sha256
            && v.encoding_sha256 == d.model.encoding_sha256
            && v.profile == d.model.profile
            && v.exported_roles == d.model.exported_roles
            && v.backend == d.model.backend
            && v.precision == d.model.precision
            && v.diagnostic_asset == (d.model.artifact_domain == PalsArtifactDomainV5::Diagnostic);
        require(
            matches_declared || (!run.accepted && !run.failures.is_empty()),
            "observed model/provenance domain does not bind the declared weights",
        )?;
        require(
            !run.accepted || v.content_domain_verified,
            "accepted execution requires independent model/provenance content verification",
        )?;
    } else {
        require(
            !run.accepted,
            "unknown model/artifact verification cannot authorize execution acceptance",
        )?;
    }
    shared::observation(&o.resource_readback)?;
    shared::observation(&o.coordinator)?;
    shared::observation(&o.native_numerical_check)?;
    shared::observation(&o.process_tree_gone)?;
    shared::observation(&o.host_peak_bytes)?;
    shared::observation(&o.vram_peak_bytes)?;
    shared::observation(&o.actual_tf32)?;
    shared::observation(&o.v_generated)?;
    shared::observation(&o.v_loaded)?;
    shared::observation(&o.v_called)?;
    if let Some(a) = observed(&o.native_numerical_check) {
        a.evidence.validate()?;
        require(
            a.weights_sha256 == d.model.weights.sha256
                && a.semantic_sha256 == d.model.semantic_sha256
                && a.encoding_sha256 == d.model.encoding_sha256
                && a.runtime_implementation_sha256 == d.runtime.implementation_sha256
                && a.source_commit == d.source_commit
                && a.logits_absolute_tolerance == "1e-4"
                && a.logits_relative_tolerance == "1e-3"
                && a.policy_wdl_absolute_tolerance == "1e-4",
            "numerical evidence is not bound to model/encoding/runtime/source and registered tolerance",
        )?;
        require(
            !run.accepted || a.passed,
            "failed numerical evidence cannot become accepted execution",
        )?;
    }
    if let Some(actual) = observed(&o.actual_tf32) {
        require(
            *actual == d.model.tf32 || (!run.accepted && !run.failures.is_empty()),
            "observed TF32 contradicts declaration without preserved failed execution",
        )?;
    }
    if let Some(r) = observed(&o.resource_readback) {
        r.validate_with_context(d, run)?;
    } else {
        require(
            !run.accepted,
            "unknown resource readback is not applied-resource acceptance",
        )?;
    }
    if let Some(c) = observed(&o.coordinator) {
        c.validate()?;
        let histories_in_bounds = c.role_request_history_peak
            <= d.coordinator.replay_history_per_namespace_max
            && c.cpu_execution_history_peak <= d.coordinator.replay_history_per_namespace_max
            && c.frame_genealogy_peak <= d.coordinator.frame_genealogy_max
            && c.ranking_fanout_history_peak <= d.coordinator.frame_genealogy_max
            && observed(&c.validator_source_contexts_peak)
                .is_none_or(|value| *value <= d.coordinator.validator_source_contexts_max)
            && observed(&c.validator_action_keys_peak)
                .is_none_or(|value| *value <= d.coordinator.validator_action_keys_max);
        require(
            !run.accepted
                || !d.v_call
                || (observed(&c.validator_source_contexts_peak).is_some()
                    && observed(&c.validator_action_keys_peak).is_some()),
            "accepted V execution requires actual source/action replay peak observations",
        )?;
        require(
            histories_in_bounds
                || (!run.accepted && run.failures.contains(&PalsFailureV5::CoordinatorCapacity)),
            "coordinator replay/genealogy peaks exceed registered namespace limits without a capacity failure",
        )?;
        require(
            !run.accepted
                || (c.cpu_outstanding_reserved == 0
                    && c.tickets_quarantined == 0
                    && c.tickets_queued == 0
                    && c.physical_in_flight == 0
                    && c.physical_completion_unknown == 0
                    && c.physical_state == PalsPhysicalStateV3::Completed
                    && c.owner_joined
                    && c.archive_closed
                    && c.tickets_released == c.tickets_submitted),
            "accepted coordinator has reservations/live or quarantined physical owners",
        )?;
        require(
            (c.physical_state != PalsPhysicalStateV3::Unknown
                && c.physical_state != PalsPhysicalStateV3::Quarantined
                && c.physical_completion_unknown == 0
                && c.tickets_quarantined == 0)
                || run
                    .failures
                    .contains(&PalsFailureV5::PhysicalCompletionUnknown),
            "physical completion unknown must remain an explicit failure",
        )?;
    } else {
        require(
            !run.accepted,
            "unknown coordinator lifecycle cannot be accepted",
        )?;
    }
    require(
        !run.accepted || observed(&o.process_tree_gone) == Some(&true),
        "accepted execution requires observed process tree closure",
    )?;
    for (actual, declared) in [
        (&o.v_generated, d.v_generate),
        (&o.v_loaded, d.v_load),
        (&o.v_called, d.v_call),
    ] {
        if let Some(value) = observed(actual) {
            require(
                *value == declared || (!run.accepted && !run.failures.is_empty()),
                "V lifecycle mismatch requires preserved failed execution, never acceptance",
            )?;
        } else {
            require(
                !run.accepted,
                "accepted execution must observe V absence/presence",
            )?;
        }
    }
    if run.performance_eligible {
        require(
            observed(&o.host_peak_bytes).is_some(),
            "performance evidence lacks primary host peak",
        )?;
    }
    Ok(())
}
fn validate_game_result(g: &PalsGameReceiptV5, p: &PalsPairPlanV5) -> Result<(), ManifestError> {
    match g.termination {
        PalsTerminationV3::EngineCrash
        | PalsTerminationV3::IllegalMove
        | PalsTerminationV3::TimeForfeit => {
            let expected = if g.failed_endpoint.as_ref() == Some(&g.white_endpoint) {
                PalsResultV3::BlackWin
            } else if g.failed_endpoint.as_ref() == Some(&g.black_endpoint) {
                PalsResultV3::WhiteWin
            } else {
                return require(false, "engine failure lacks actual offender");
            };
            require(
                g.result == expected,
                "engine failure must remain offender loss",
            )
        }
        PalsTerminationV3::Checkmate => require(
            g.failed_endpoint.is_none()
                && matches!(g.result, PalsResultV3::WhiteWin | PalsResultV3::BlackWin),
            "mate result differs",
        ),
        PalsTerminationV3::RulesDraw => require(
            g.failed_endpoint.is_none() && g.result == PalsResultV3::Draw,
            "Rules draw result differs",
        ),
        PalsTerminationV3::PlyLimit => require(
            g.failed_endpoint.is_none()
                && g.plies == p.max_plies
                && g.result == PalsResultV3::Incomplete,
            "ply cutoff is incomplete, not a Rules WDL target",
        ),
        PalsTerminationV3::RunDeadline | PalsTerminationV3::InfrastructureFailure => require(
            g.failed_endpoint.is_none() && g.result == PalsResultV3::Incomplete,
            "infrastructure/deadline cannot be completed game or engine loss",
        ),
    }
}
fn validate_training(
    t: &PalsTrainingObservationV5,
    m: &PalsRunManifestV5,
    accepted: bool,
) -> Result<(), ManifestError> {
    t.checkpoint.validate()?;
    t.provenance.validate()?;
    shared::observation(&t.resume_exact_state)?;
    let recipe = m
        .generation
        .as_ref()
        .ok_or_else(|| ManifestError::Integrity("missing training recipe".into()))?;
    require(
        t.checkpoint_domain == PALS_V5_TRAINING_CHECKPOINT_DOMAIN
            && t.shared_optimizer_owners == 1
            && t.optimizer_update_boundary
            && t.p_completed_updates <= recipe.p_updates
            && t.c_completed_updates <= recipe.c_updates
            && t.v_completed_updates <= recipe.v_updates
            && t.collection_games_completed <= recipe.collection_games
            && t.evaluation_pairs_completed <= recipe.evaluation_pairs,
        "observed training checkpoint/update domain differs",
    )?;
    require(
        !accepted
            || (t.p_completed_updates == recipe.p_updates
                && t.c_completed_updates == recipe.c_updates
                && t.v_completed_updates == recipe.v_updates
                && t.p_unique_valid_samples >= recipe.pc_unique_valid_samples_min
                && t.c_unique_valid_samples >= recipe.pc_unique_valid_samples_min
                && t.v_known_pairs >= recipe.v_known_pairs_min
                && t.v_new_witness_contexts >= recipe.v_witness_contexts_min
                && t.pc_frozen_before_v
                && observed(&t.resume_exact_state) == Some(&true)),
        "accepted training requires real valid targets, complete updates, freeze and exact resume evidence",
    )
}

fn require(ok: bool, message: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(format!("PALS V5: {message}")))
    }
}
fn text(s: &str) -> bool {
    !s.trim().is_empty() && s.len() <= 1024 && !s.chars().any(char::is_control)
}
fn sha(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn unique_cpus(cpus: &[u32]) -> bool {
    cpus.len() <= 64 && cpus.iter().collect::<BTreeSet<_>>().len() == cpus.len()
}
fn checked_sum(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
}
fn observed<T>(value: &PalsObservedV3<T>) -> Option<&T> {
    match value {
        PalsObservedV3::Unknown => None,
        PalsObservedV3::Observed { value, .. } => Some(value),
    }
}
fn encoded<T: Serialize>(value: &T, pretty: bool) -> Result<Vec<u8>, ManifestError> {
    if pretty {
        serde_json::to_vec_pretty(value)
    } else {
        serde_json::to_vec(value)
    }
    .map_err(|error| ManifestError::Integrity(format!("PALS V5 encoding: {error}")))
}
fn json_string<T: Serialize>(value: &T) -> Result<String, ManifestError> {
    String::from_utf8(encoded(value, true)?).map_err(|e| ManifestError::Integrity(e.to_string()))
}

#[cfg(test)]
mod tests;
