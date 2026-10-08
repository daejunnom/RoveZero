//! PALS execution identity. This domain never converts or rewrites V1/V2 records.
//!
//! Declared identity is not an attestation of artifact bytes or GPU readiness.
//! Execution receipts must carry their own observations and physical completion.
use crate::{ArtifactRef, ManifestError, decode_json, digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PALS_MANIFEST_V3_DOMAIN: &str = "rz-pals-execution-v3/1";
// Initial /1 was unpublished WIP. /2 separates CPU completed-task reuse from
// NN cache consumption; historical V1/V2 codecs and execution locks are intact.
pub const PALS_RECEIPT_V3_DOMAIN: &str = "rz-pals-receipt-v3/2";

/// Closed opt-in search lane. These are declaration/observation wire identities,
/// not execution proof, a new model, or a change to the value resolver.
pub const PALS_POST_REPAIR_RECHECK_V3_VERSION: &str = "pals-post-repair-recheck/1";
pub const PALS_POST_REPAIR_RECHECK_V3_OPTION_KEY: &str = "post_repair_recheck";
pub const PALS_POST_REPAIR_RECHECK_V3_OPTION_VALUE: &str = "same-repaired-line-once-v1";
pub const PALS_POST_REPAIR_RECHECK_V3_POLICY: &str = "same_repaired_line_once_v1";
pub const PALS_POST_REPAIR_RECHECK_V3_SEARCH_IDENTITY: &str =
    "pals-restricted-refinement-post-repair-recheck/1";
/// SHA-256 of the exact engine's 425-byte UTF-8 refinement conditions. Arena
/// must independently compare this closed contract pin with its engine constants;
/// experiments never depends on search or UCI to validate persisted identities.
pub const PALS_POST_REPAIR_RECHECK_V3_CONDITIONS_SHA256: [u8; 32] = [
    0xbe, 0xa4, 0x4b, 0x7e, 0x9a, 0xb5, 0x9f, 0x32, 0xdc, 0xff, 0xb1, 0xb4, 0xc5, 0x97, 0xfd, 0x36,
    0xa4, 0xb7, 0x50, 0x37, 0x80, 0x3a, 0xee, 0xac, 0xd6, 0xc1, 0x87, 0x85, 0xce, 0x83, 0x81, 0x66,
];

/// Omission selects the historical lane; there is no serialized Disabled alias.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub enum PalsPostRepairRecheckPolicyV3 {
    #[serde(rename = "same-repaired-line-once-v1")]
    SameRepairedLineOnceV1,
}

/// Four-field identity captured from immutable startup-selected engine getters.
/// A matching identity does not prove a post-Repair Reply ran, completed or won.
#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct PalsSearchPolicyIdentityV3 {
    pub version: String,
    pub policy: String,
    pub search_identity: String,
    pub conditions_sha256: [u8; 32],
}
impl PalsSearchPolicyIdentityV3 {
    pub fn expected_same_repaired_line_once_v1() -> Self {
        Self {
            version: PALS_POST_REPAIR_RECHECK_V3_VERSION.into(),
            policy: PALS_POST_REPAIR_RECHECK_V3_POLICY.into(),
            search_identity: PALS_POST_REPAIR_RECHECK_V3_SEARCH_IDENTITY.into(),
            conditions_sha256: PALS_POST_REPAIR_RECHECK_V3_CONDITIONS_SHA256,
        }
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.version == PALS_POST_REPAIR_RECHECK_V3_VERSION
                && self.policy == PALS_POST_REPAIR_RECHECK_V3_POLICY
                && self.search_identity == PALS_POST_REPAIR_RECHECK_V3_SEARCH_IDENTITY
                && self.conditions_sha256 == PALS_POST_REPAIR_RECHECK_V3_CONDITIONS_SHA256,
            "selected search policy identity differs from the closed post-Repair lane",
        )
    }
}
impl<'de> Deserialize<'de> for PalsSearchPolicyIdentityV3 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            version: String,
            policy: String,
            search_identity: String,
            conditions_sha256: [u8; 32],
        }
        let wire = Wire::deserialize(deserializer)?;
        let identity = Self {
            version: wire.version,
            policy: wire.policy,
            search_identity: wire.search_identity,
            conditions_sha256: wire.conditions_sha256,
        };
        identity.validate().map_err(serde::de::Error::custom)?;
        Ok(identity)
    }
}

/// Used only for the new optional Core observation: absent is unknown, while
/// a present JSON null is invalid. Existing optional fields keep their codecs.
pub fn deserialize_present_pals_search_policy<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<PalsSearchPolicyIdentityV3>, D::Error> {
    PalsSearchPolicyIdentityV3::deserialize(deserializer).map(Some)
}

macro_rules! choices {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
    };
}
choices!(PalsComparisonV3 {
    System,
    InternalModel,
    InternalSearch,
    Runtime
});
choices!(PalsChangeAxisV3 {
    Model,
    Search,
    Runtime,
    CpuCore,
    Endpoint
});
choices!(PalsRoleV3 {
    Proposer,
    Critic,
    Validator
});
choices!(PalsModelBackendV3 {
    DeterministicMock,
    RustCpu,
    OrtCpu,
    OrtCuda
});
choices!(PalsPrecisionV3 { Fp32, Fp16, Bf16 });
choices!(PalsResultV3 {
    WhiteWin,
    BlackWin,
    Draw,
    Incomplete
});
choices!(PalsTerminationV3 {
    Checkmate,
    RulesDraw,
    IllegalMove,
    EngineCrash,
    TimeForfeit,
    PlyLimit,
    RunDeadline,
    InfrastructureFailure
});
choices!(PalsPhysicalStateV3 {
    NotRequired,
    Completed,
    Quarantined,
    Unknown
});
choices!(PalsRunFailureV3 {
    Cancelled,
    WallDeadline,
    CleanupTimeout,
    PhysicalCompletionUnknown,
    ResourceAdmission,
    Infrastructure,
    EngineLaunch,
    IncompleteGame,
    SearchFailure
});

/// Initial parameters and trained parameters cannot share a fictitious weights ID.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsWeightIdentityV3 {
    DeterministicMock {
        seed: u64,
    },
    Untrained {
        artifact: ArtifactRef,
        initialization_seed: u64,
    },
    Trained {
        artifact: ArtifactRef,
        training_run_id: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsModelIdentityV3 {
    pub architecture: String,
    pub input_schema: String,
    pub policy_head: String,
    pub value_head: String,
    pub implementation_sha256: String,
    pub weights: PalsWeightIdentityV3,
    /// Parameters are immutable throughout this run, including self-play rounds.
    pub frozen_epoch: u64,
    pub backend: PalsModelBackendV3,
    pub precision: PalsPrecisionV3,
    pub max_batch_width: u32,
    /// Product inference exports P/C only. V is a training task policy.
    pub exported_roles: BTreeSet<PalsRoleV3>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsComponentV3 {
    pub semantic_id: String,
    pub implementation_sha256: String,
    pub options: BTreeMap<String, String>,
}

/// Same Rust core, with separately declared T/R profile and evaluation semantics.
/// This is a CPU_T/legacy own declaration. It authorizes CPU_R execution only
/// when `PalsEndpointV3::cpu_r` selects Own; it never attests task execution.
/// A CPU problem result is a scoped estimate/bound; this identity grants no proof.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsOwnCpuV3 {
    pub core: PalsComponentV3,
    pub training_profile: PalsComponentV3,
    pub runtime_profile: PalsComponentV3,
    pub evaluation: PalsComponentV3,
    pub max_nodes_per_task: u64,
    pub max_depth: u32,
    pub max_task_ms: u64,
    pub max_tt_bytes: u64,
}

/// Runtime checker selection is independent of the preserved own CPU_T profile.
/// Omitting Own preserves the exact historical V3 serializer and lock bytes.
#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "kind",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsCpuRSelectionV3 {
    #[default]
    Own,
    ExternalUci(Box<PalsExternalCpuRV3>),
}
impl PalsCpuRSelectionV3 {
    pub fn is_own(&self) -> bool {
        matches!(self, Self::Own)
    }
}

choices!(PalsExternalCpuRSelectionV3 {
    StockfishEmbeddedNnue
});
choices!(PalsExternalCpuRResourceScopeV3 {
    InheritedParentCgroup
});

/// Foreign CP/mate/bounds remain external estimates. This declared resolver
/// identifies selected-model WDL resolution, never calibration of those scores.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsExternalCpuRResolverV3 {
    pub version: String,
    /// Declared semantic source digest; a runtime consumer must compare the
    /// actual resolver identity independently before accepting execution.
    pub semantics_sha256: String,
}

/// Finite ceilings to compare against the independently loaded checker profile.
/// These are reservations/declarations, not observations of option application,
/// OS thread count, memory peak, model loading, or successful process cleanup.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsExternalCpuRPolicyV3 {
    /// The helper shares the parent's affinity and memory/swap cgroup limits.
    /// There is no independent CPU, memory or GPU allocation for the helper.
    pub resource_scope: PalsExternalCpuRResourceScopeV3,
    pub max_owners: u32,
    pub max_active_tasks: u32,
    pub max_process_leaders: u32,
    /// Whole inherited cgroup kernel-task ceiling, including Linux threads.
    /// It is not interchangeable with the helper process-leader count.
    pub inherited_kernel_tasks_max: u32,
    pub threads_max: u32,
    pub hash_mib_max: u32,
    pub max_depth: u32,
    pub max_prefix_plies: u32,
    pub max_nodes_per_task: u64,
    pub handshake_max_ms: u64,
    pub task_wall_time_max_ms: u64,
    /// This must be reserved inside the supplied search deadline at admission.
    /// A finite value alone does not authorize extending that deadline.
    pub stop_grace_max_ms: u64,
    /// Helper and native NN owners require separate closure observations;
    /// their combined shutdown must fit the parent's existing cleanup window.
    pub shutdown_grace_max_ms: u64,
    pub lifetime_output_bytes_max: u64,
    pub line_bytes_max: u64,
}

/// First supported external CPU_R declaration: Stockfish with embedded NNUE,
/// no engine argv or separately loaded assets. File and canonical profile pins
/// are distinct. Public source/license declarations prove no model loading or
/// training; those observations remain unknown in the external checker profile.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsExternalCpuRV3 {
    pub selection: PalsExternalCpuRSelectionV3,
    pub profile: ArtifactRef,
    pub profile_canonical_sha256: String,
    pub binary: ArtifactRef,
    pub resolver: PalsExternalCpuRResolverV3,
    pub policy: PalsExternalCpuRPolicyV3,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPoolLimitsV3 {
    pub states: u32,
    pub line_chunks: u32,
    pub situations: u32,
    pub observations: u32,
    pub tasks: u32,
    pub role_states: u32,
    pub memory_pages: u32,
    pub queue_requests: u32,
    pub host_bytes: u64,
    pub device_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsEndpointV3 {
    pub id: String,
    pub binary: ArtifactRef,
    pub source_commit: String,
    pub model: PalsModelIdentityV3,
    /// Own CPU_T and historical own CPU_R declaration; see `cpu_r` for the
    /// actual runtime selection. Keeping it does not mean it was executed.
    pub cpu: PalsOwnCpuV3,
    #[serde(default, skip_serializing_if = "PalsCpuRSelectionV3::is_own")]
    pub cpu_r: PalsCpuRSelectionV3,
    pub search: PalsComponentV3,
    pub runtime: PalsComponentV3,
    pub pools: PalsPoolLimitsV3,
    pub requested_options: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCpuEndpointV3 {
    pub id: String,
    pub binary: ArtifactRef,
    pub source_commit: String,
    pub cpu: PalsOwnCpuV3,
    pub requested_options: BTreeMap<String, String>,
}

/// A comparison opponent is a UCI process, never an internal neural evaluator.
/// BT4/LC0 assets belong here when the whole PALS system is compared with LC0.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsReferenceEndpointV3 {
    pub id: String,
    pub family: String,
    pub version: String,
    pub binary: ArtifactRef,
    pub source_url: String,
    pub source_commit: String,
    pub assets: Vec<ArtifactRef>,
    pub requested_options: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "endpoint",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsEngineV3 {
    Pals(Box<PalsEndpointV3>),
    OwnCpu(Box<PalsCpuEndpointV3>),
    ReferenceUci(Box<PalsReferenceEndpointV3>),
}
impl PalsEngineV3 {
    pub fn id(&self) -> &str {
        match self {
            Self::Pals(e) => &e.id,
            Self::OwnCpu(e) => &e.id,
            Self::ReferenceUci(e) => &e.id,
        }
    }
    pub fn requested_options(&self) -> &BTreeMap<String, String> {
        match self {
            Self::Pals(e) => &e.requested_options,
            Self::OwnCpu(e) => &e.requested_options,
            Self::ReferenceUci(e) => &e.requested_options,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsResourcePolicyV3 {
    pub cpu_threads: u32,
    pub cpu_affinity: Vec<u32>,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
    pub swap_max_bytes: u64,
    /// Requested device is not an observation that the device/provider is ready.
    pub requested_gpu: Option<String>,
    pub device_allocation_max_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPilotV3 {
    /// First pilot: standard start state, two games with the colors exchanged.
    pub position_command: String,
    pub games: u32,
    pub white_order: [String; 2],
    pub base_ms: u64,
    pub increment_ms: u64,
    pub max_plies: u32,
    pub seed: u64,
    pub wall_time_max_ms: u64,
    pub cleanup_max_ms: u64,
    /// External process readiness window, including cold runtime/model
    /// construction and any separately declared post-load startup probe.
    /// This is not a per-command probe limit or a whole-pair time extension;
    /// the launch validator must fit an explicit probe within this window.
    pub handshake_max_ms: u64,
    pub concurrent_games: u32,
    pub restart_processes_each_game: bool,
    pub ponder: bool,
    pub score_adjudication: bool,
    pub elo_claim: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRunManifestV3 {
    pub schema_version: u32,
    pub run_id: String,
    pub pair_id: String,
    pub contract_revision: String,
    pub rules_profile: String,
    pub comparison: PalsComparisonV3,
    pub declared_changes: BTreeSet<PalsChangeAxisV3>,
    /// Actual learning is outside this implementation objective.
    pub training_executed: bool,
    pub engines: [PalsEngineV3; 2],
    pub resources: [PalsResourcePolicyV3; 2],
    pub pilot: PalsPilotV3,
}

fn require(ok: bool, message: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(format!("PALS V3: {message}")))
    }
}
fn text(s: &str) -> bool {
    !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control)
}
fn sha(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn options(m: &BTreeMap<String, String>) -> bool {
    let casefold: BTreeSet<_> = m.keys().map(|key| key.to_ascii_lowercase()).collect();
    m.len() <= 64
        && casefold.len() == m.len()
        && m.iter()
            .all(|(k, v)| text(k) && v.len() <= 1024 && !v.chars().any(char::is_control))
}
fn component(c: &PalsComponentV3) -> Result<(), ManifestError> {
    require(
        text(&c.semantic_id) && sha(&c.implementation_sha256, 64) && options(&c.options),
        "invalid component identity/options",
    )
}
fn pals_search_lane(p: &PalsEndpointV3) -> Result<(), ManifestError> {
    match p.search.semantic_id.as_str() {
        // Keep the historical component/options declaration semantics intact.
        // The executable arena recipe applies its existing stricter closure.
        "pals" => Ok(()),
        PALS_POST_REPAIR_RECHECK_V3_VERSION => require(
            p.cpu_r.is_own()
                && p.search.options.len() == 1
                && p.search
                    .options
                    .get(PALS_POST_REPAIR_RECHECK_V3_OPTION_KEY)
                    .is_some_and(|value| value == PALS_POST_REPAIR_RECHECK_V3_OPTION_VALUE),
            "post-Repair search lane requires its exact single option and Own CPU_R",
        ),
        _ => require(false, "unsupported PALS search semantic lane"),
    }
}
fn cpu(c: &PalsOwnCpuV3) -> Result<(), ManifestError> {
    component(&c.core)?;
    component(&c.training_profile)?;
    component(&c.runtime_profile)?;
    component(&c.evaluation)?;
    require(
        c.core.semantic_id == "rz-cpu-pvs/0.1"
            && c.max_nodes_per_task > 0
            && c.max_depth > 0
            && c.max_depth <= 128
            && c.max_task_ms > 0
            && c.max_task_ms <= 900_000
            && c.max_tt_bytes > 0,
        "CPU core must be own PVS with finite node/depth/time/TT limits",
    )
}
fn external_cpu_r(
    c: &PalsExternalCpuRV3,
    r: &PalsResourcePolicyV3,
    pilot: &PalsPilotV3,
) -> Result<(), ManifestError> {
    c.profile.validate()?;
    c.binary.validate()?;
    require(
        (1..=64 * 1024).contains(&c.profile.bytes)
            && (1..=256 * 1024 * 1024).contains(&c.binary.bytes)
            && sha(&c.profile_canonical_sha256, 64)
            && (c.binary.source == "https://github.com/official-stockfish/Stockfish"
                || c.binary
                    .source
                    .starts_with("https://github.com/official-stockfish/Stockfish/"))
            && c.binary.license == "GPL-3.0-or-later",
        "external CPU_R requires bounded pinned profile/binary and declared Stockfish source/license",
    )?;
    require(
        c.resolver.version == "pals-model-wdl-restricted/0.1"
            && sha(&c.resolver.semantics_sha256, 64),
        "external CPU_R requires declared model-WDL resolver, not own raw/foreign CP calibration",
    )?;
    let p = &c.policy;
    require(
        p.max_owners == 1
            && p.max_active_tasks == 1
            && p.max_process_leaders == 1
            && (1..=65_536).contains(&p.inherited_kernel_tasks_max)
            && (1..=2).contains(&p.threads_max)
            && p.threads_max <= r.cpu_threads
            && p.inherited_kernel_tasks_max >= p.threads_max
            && (1..=1024).contains(&p.hash_mib_max)
            && u64::from(p.hash_mib_max) * 1024 * 1024 <= r.memory_high_bytes
            && (1..=64).contains(&p.max_depth)
            && p.max_prefix_plies <= 4096
            && p.max_nodes_per_task > 0,
        "external CPU_R requires one bounded owner/task and inherited finite CPU/memory/task caps",
    )?;
    require(
        [
            p.handshake_max_ms,
            p.task_wall_time_max_ms,
            p.stop_grace_max_ms,
            p.shutdown_grace_max_ms,
        ]
        .iter()
        .all(|time| (1..=180_000).contains(time))
            && p.handshake_max_ms <= pilot.handshake_max_ms
            && p.task_wall_time_max_ms <= pilot.wall_time_max_ms
            && p.stop_grace_max_ms <= pilot.cleanup_max_ms
            && p.shutdown_grace_max_ms <= pilot.cleanup_max_ms
            && (1..=16 * 1024 * 1024).contains(&p.lifetime_output_bytes_max)
            && p.lifetime_output_bytes_max <= r.memory_high_bytes
            && (1..=65_536).contains(&p.line_bytes_max)
            && p.line_bytes_max <= p.lifetime_output_bytes_max,
        "external CPU_R time/output declarations exceed finite profile or inherited parent ceilings",
    )
}
fn resources(r: &PalsResourcePolicyV3) -> Result<(), ManifestError> {
    let set: BTreeSet<_> = r.cpu_affinity.iter().collect();
    require(
        (1..=64).contains(&r.cpu_threads)
            && r.cpu_affinity.len() <= 64
            && set.len() == r.cpu_affinity.len()
            && !r.cpu_affinity.is_empty()
            && r.cpu_affinity.len() >= r.cpu_threads as usize
            && r.memory_high_bytes > 0
            && r.memory_high_bytes <= r.memory_max_bytes
            && r.memory_max_bytes <= 1 << 50
            && r.requested_gpu.as_ref().is_none_or(|s| text(s))
            && (r.requested_gpu.is_some() || r.device_allocation_max_bytes == 0),
        "invalid resource policy",
    )
}
fn model(m: &PalsModelIdentityV3) -> Result<(), ManifestError> {
    require(
        [
            &m.architecture,
            &m.input_schema,
            &m.policy_head,
            &m.value_head,
        ]
        .iter()
        .all(|s| text(s))
            && sha(&m.implementation_sha256, 64)
            && (1..=16).contains(&m.max_batch_width)
            && m.exported_roles == BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
        "invalid model identity or V leaked into product export",
    )?;
    match &m.weights {
        PalsWeightIdentityV3::DeterministicMock { .. } => require(
            m.backend == PalsModelBackendV3::DeterministicMock,
            "mock identity cannot claim native NN execution",
        ),
        PalsWeightIdentityV3::Untrained { artifact, .. } => {
            artifact.validate()?;
            require(
                m.backend != PalsModelBackendV3::DeterministicMock,
                "untrained tensor asset cannot be a mock",
            )
        }
        PalsWeightIdentityV3::Trained {
            artifact,
            training_run_id,
        } => {
            artifact.validate()?;
            require(
                text(training_run_id) && m.backend != PalsModelBackendV3::DeterministicMock,
                "trained provenance missing",
            )
        }
    }
}
fn endpoint(
    e: &PalsEngineV3,
    r: &PalsResourcePolicyV3,
    pilot: &PalsPilotV3,
) -> Result<(), ManifestError> {
    require(
        text(e.id()) && options(e.requested_options()),
        "invalid engine identity/options",
    )?;
    for (name, value) in e.requested_options() {
        let key = name.to_ascii_lowercase();
        match key.as_str() {
            "ponder" | "uci_chess960" => require(
                value.eq_ignore_ascii_case("false"),
                "pilot forbids ponder and Chess960 overrides",
            )?,
            "threads" => require(
                value.parse::<u32>().ok() == Some(r.cpu_threads),
                "Threads option differs from CPU resource policy",
            )?,
            "hash" => require(
                value
                    .parse::<u64>()
                    .ok()
                    .and_then(|n| n.checked_mul(1024 * 1024))
                    .is_some_and(|n| n > 0 && n <= r.memory_max_bytes),
                "Hash option exceeds memory policy",
            )?,
            _ => {}
        }
        if !matches!(e, PalsEngineV3::ReferenceUci(_)) {
            require(
                !matches!(
                    key.as_str(),
                    "weightsfile"
                        | "weights"
                        | "model"
                        | "modeladapter"
                        | "backend"
                        | "precision"
                        | "searchmode"
                        | "searchpolicy"
                ),
                "native model/search identity cannot be overridden with untyped UCI options",
            )?;
        }
    }
    let (binary, commit) = match e {
        PalsEngineV3::Pals(p) => {
            model(&p.model)?;
            cpu(&p.cpu)?;
            if let PalsCpuRSelectionV3::ExternalUci(c) = &p.cpu_r {
                external_cpu_r(c, r, pilot)?;
            }
            component(&p.search)?;
            component(&p.runtime)?;
            pals_search_lane(p)?;
            let b = &p.pools;
            require(
                [
                    b.states,
                    b.line_chunks,
                    b.situations,
                    b.observations,
                    b.tasks,
                    b.role_states,
                    b.memory_pages,
                    b.queue_requests,
                ]
                .iter()
                .all(|n| *n > 0)
                    && b.host_bytes > 0
                    && b.host_bytes <= r.memory_max_bytes
                    && b.device_bytes <= r.device_allocation_max_bytes
                    && p.cpu.max_tt_bytes <= b.host_bytes,
                "PALS search/pool admission limits invalid",
            )?;
            require(
                p.model.backend != PalsModelBackendV3::OrtCuda
                    || (r.requested_gpu.is_some() && b.device_bytes > 0),
                "CUDA declaration requires explicit bounded device resources",
            )?;
            (&p.binary, &p.source_commit)
        }
        PalsEngineV3::OwnCpu(p) => {
            cpu(&p.cpu)?;
            require(
                p.cpu.max_tt_bytes <= r.memory_max_bytes && r.requested_gpu.is_none(),
                "own CPU endpoint must not claim GPU resources",
            )?;
            (&p.binary, &p.source_commit)
        }
        PalsEngineV3::ReferenceUci(p) => {
            require(
                text(&p.family) && text(&p.version) && text(&p.source_url) && p.assets.len() <= 32,
                "invalid external UCI provenance",
            )?;
            for a in &p.assets {
                a.validate()?;
            }
            (&p.binary, &p.source_commit)
        }
    };
    binary.validate()?;
    require(sha(commit, 40), "source commit must be a full Git SHA")
}
fn changes(a: &PalsEngineV3, b: &PalsEngineV3) -> BTreeSet<PalsChangeAxisV3> {
    let mut result = BTreeSet::new();
    if let (PalsEngineV3::Pals(a), PalsEngineV3::Pals(b)) = (a, b) {
        if a.model != b.model {
            result.insert(PalsChangeAxisV3::Model);
        }
        if a.cpu != b.cpu || a.cpu_r != b.cpu_r {
            result.insert(PalsChangeAxisV3::CpuCore);
        }
        if a.search != b.search {
            result.insert(PalsChangeAxisV3::Search);
        }
        if a.runtime != b.runtime || a.pools != b.pools {
            result.insert(PalsChangeAxisV3::Runtime);
        }
        if a.requested_options != b.requested_options {
            result.insert(PalsChangeAxisV3::Endpoint);
        }
        let binary_changed = (a.binary.sha256.as_str(), a.binary.bytes)
            != (b.binary.sha256.as_str(), b.binary.bytes);
        let implementation_changed = a.model.implementation_sha256 != b.model.implementation_sha256
            || a.search.implementation_sha256 != b.search.implementation_sha256
            || a.runtime.implementation_sha256 != b.runtime.implementation_sha256
            || a.cpu.core.implementation_sha256 != b.cpu.core.implementation_sha256
            || a.cpu.training_profile.implementation_sha256
                != b.cpu.training_profile.implementation_sha256
            || a.cpu.runtime_profile.implementation_sha256
                != b.cpu.runtime_profile.implementation_sha256
            || a.cpu.evaluation.implementation_sha256 != b.cpu.evaluation.implementation_sha256;
        if binary_changed && !implementation_changed {
            result.insert(PalsChangeAxisV3::Endpoint);
        }
    } else {
        result.insert(PalsChangeAxisV3::Endpoint);
    }
    result
}

impl PalsRunManifestV3 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let m: Self = decode_json(input)?;
        m.validate()?;
        Ok(m)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.schema_version == 3 && self.contract_revision == "pals/0.1",
            "schema/revision mismatch; no automatic legacy conversion",
        )?;
        require(
            [&self.run_id, &self.pair_id, &self.rules_profile]
                .iter()
                .all(|s| text(s))
                && !self.training_executed,
            "invalid run identity or actual training attempted",
        )?;
        require(
            self.engines[0].id() != self.engines[1].id()
                && self
                    .engines
                    .iter()
                    .any(|e| matches!(e, PalsEngineV3::Pals(_))),
            "distinct endpoints and a PALS endpoint required",
        )?;
        for (e, r) in self.engines.iter().zip(&self.resources) {
            resources(r)?;
            endpoint(e, r, &self.pilot)?;
        }
        let actual = changes(&self.engines[0], &self.engines[1]);
        require(
            actual == self.declared_changes,
            "declared change axes differ from endpoint configuration",
        )?;
        let expected = match self.comparison {
            PalsComparisonV3::System => None,
            PalsComparisonV3::InternalModel => Some(PalsChangeAxisV3::Model),
            PalsComparisonV3::InternalSearch => Some(PalsChangeAxisV3::Search),
            PalsComparisonV3::Runtime => Some(PalsChangeAxisV3::Runtime),
        };
        if let Some(axis) = expected {
            require(
                actual == BTreeSet::from([axis]) && self.resources[0] == self.resources[1],
                "controlled comparison must change exactly its declared axis and hold resources fixed",
            )?;
        }
        let p = &self.pilot;
        require(
            p.position_command == "position startpos"
                && p.games == 2
                && p.white_order[0] != p.white_order[1]
                && p.white_order
                    .iter()
                    .all(|id| self.engines.iter().any(|e| e.id() == id))
                && p.base_ms == 120_000
                && p.increment_ms == 1_000
                && (1..=256).contains(&p.max_plies)
                && p.wall_time_max_ms > 0
                && p.wall_time_max_ms <= 900_000
                && p.cleanup_max_ms > 0
                && p.cleanup_max_ms <= 30_000
                && p.handshake_max_ms > 0
                && p.handshake_max_ms <= p.wall_time_max_ms
                && p.concurrent_games == 1
                && p.restart_processes_each_game
                && !p.ponder
                && !p.score_adjudication
                && !p.elo_claim,
            "first paired pilot requires 120+1, two exchanged games, bounded 15min+30s and no Elo/ponder/adjudication",
        )
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        serde_json::to_vec(&(PALS_MANIFEST_V3_DOMAIN, self))
            .map_err(|e| ManifestError::Integrity(e.to_string()))
    }
    pub fn lock(&self) -> Result<PalsInputLockV3, ManifestError> {
        let bytes = self.canonical_bytes()?;
        Ok(PalsInputLockV3 {
            domain: PALS_MANIFEST_V3_DOMAIN.into(),
            manifest: self.clone(),
            canonical_sha256: digest(&bytes),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsInputLockV3 {
    pub domain: String,
    pub manifest: PalsRunManifestV3,
    pub canonical_sha256: String,
}
impl PalsInputLockV3 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let l: Self = decode_json(input)?;
        l.verify()?;
        Ok(l)
    }
    pub fn verify(&self) -> Result<(), ManifestError> {
        require(
            self.domain == PALS_MANIFEST_V3_DOMAIN
                && sha(&self.canonical_sha256, 64)
                && digest(&self.manifest.canonical_bytes()?) == self.canonical_sha256,
            "lock domain/digest mismatch",
        )
    }
}

/// `readyok` is a protocol event. An actual applied value needs separate evidence.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsObservedV3<T> {
    Unknown,
    Observed { value: T, method: String },
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsOptionReceiptV3 {
    pub requested: String,
    pub advertised_supported: bool,
    pub observed: PalsObservedV3<String>,
}

/// Historical Linux identity; it is neither a currently live PID nor one of
/// Fastchess's four game UCI leaders.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsHelperProcessIdentityV3 {
    pub pid: u32,
    pub process_group: u32,
    pub proc_start_ticks: u64,
    pub scope: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsReadyLimitV3 {
    Numeric { value: u64 },
    Max,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsReadyCgroupLimitsV3 {
    pub mount_point: String,
    pub mount_root: String,
    pub resolved_directory: String,
    pub memory_high: PalsReadyLimitV3,
    pub memory_max: PalsReadyLimitV3,
    pub memory_swap_max: PalsReadyLimitV3,
    pub pids_max: PalsReadyLimitV3,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsReadyThreadV3 {
    pub tid: u32,
    pub proc_start_ticks_before: u64,
    pub proc_start_ticks_after: u64,
    pub cpu_allowed_list: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsReadyProcessV3 {
    pub pid: u32,
    pub parent_pid: u32,
    pub process_group: u32,
    pub proc_start_ticks_before: u64,
    pub proc_start_ticks_after: u64,
    pub cgroup_v2_membership: String,
    pub membership_path_view: String,
    pub cgroup_namespace_inode: Option<u64>,
    pub cpu_allowed_list: String,
    pub threads: Vec<PalsReadyThreadV3>,
    pub thread_observation_scope: String,
    pub cgroup_limits: PalsReadyCgroupLimitsV3,
}
/// Direct cgroup-file/affinity observations at ready, never applied UCI option
/// values, ancestor effective limits, memory peaks or lifetime enforcement.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCheckerReadyResourcesV3 {
    pub schema_version: u32,
    pub domain: String,
    pub scope: String,
    pub observed_unix_us: u64,
    pub observation_elapsed_us: u64,
    pub parent: PalsReadyProcessV3,
    pub helper: PalsReadyProcessV3,
    pub cgroup_membership_equal: bool,
    pub cgroup_namespace_equal: Option<bool>,
    pub cpu_allowed_list_equal: bool,
    pub additional_allocation: String,
    pub read_consistency: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsHelperShutdownV3 {
    pub process_identity: Option<PalsHelperProcessIdentityV3>,
    pub stop_sent: bool,
    pub quit_sent: bool,
    pub exit_observed: bool,
    pub stdout_drained: bool,
    pub stderr_drained: bool,
    pub exit_code: Option<i32>,
    pub exit_signal: Option<i32>,
    pub cleanup_complete: bool,
    pub quarantined: bool,
    pub ownership_lost: bool,
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsModelValueIdentityV3 {
    pub semantics: String,
    pub model: String,
    pub encoding: String,
    pub precision: String,
    /// Checkpoint digest, separate from numerical frozen/process epochs.
    pub model_epoch: [u8; 32],
}
fn unknown_count() -> PalsObservedV3<u64> {
    PalsObservedV3::Unknown
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsExternalCpuRWorkV3 {
    pub tasks_dispatched: Option<u64>,
    pub reports_returned: Option<u64>,
    pub node_budget_reserved: Option<u64>,
    /// Foreign reported work; missing or incomplete is not zero/own nodes.
    pub nodes_observed: Option<u64>,
    pub consumed_completed_tasks: Option<u64>,
    pub work_incomplete: Option<bool>,
    /// The admitted producer has no totals for these two units. Defaults are
    /// unknown; report counts or total consumption cannot substitute for them.
    #[serde(default = "unknown_count")]
    pub completed_tasks: PalsObservedV3<u64>,
    #[serde(default = "unknown_count")]
    pub reused_completed_task_consumptions: PalsObservedV3<u64>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsExternalCpuRReceiptV3 {
    pub profile_file_sha256: String,
    pub profile_canonical_sha256: String,
    pub registered_binary_sha256: String,
    pub registration_sha256: String,
    pub resolver: PalsExternalCpuRResolverV3,
    pub model_value: PalsModelValueIdentityV3,
    pub ready_resources: PalsCheckerReadyResourcesV3,
    pub shutdown: PalsHelperShutdownV3,
    pub work: PalsExternalCpuRWorkV3,
    pub applied_option_values: PalsObservedV3<BTreeMap<String, String>>,
    pub model_loading: PalsObservedV3<bool>,
}

fn ready_path(path: &str) -> bool {
    path.starts_with('/')
        && path.len() <= 4096
        && !path.chars().any(char::is_control)
        && !path.split('/').any(|part| part == "." || part == "..")
}
fn ready_affinity(text: &str) -> Result<BTreeSet<u32>, ManifestError> {
    require(
        !text.is_empty() && text.len() <= 4096,
        "ready affinity invalid",
    )?;
    let mut cpus = BTreeSet::new();
    for item in text.split(',') {
        let (low, high) = match item.split_once('-') {
            Some((lo, hi)) => (lo.parse::<u32>(), hi.parse::<u32>()),
            None => (item.parse::<u32>(), item.parse::<u32>()),
        };
        let (low, high) = (
            low.map_err(|_| ManifestError::Integrity("ready affinity invalid".into()))?,
            high.map_err(|_| ManifestError::Integrity("ready affinity invalid".into()))?,
        );
        require(
            high >= low && high - low < 64,
            "ready affinity range exceeds bound",
        )?;
        for cpu in low..=high {
            require(
                cpus.insert(cpu) && cpus.len() <= 64,
                "ready affinity duplicate/oversized",
            )?;
        }
    }
    Ok(cpus)
}
impl PalsCheckerReadyResourcesV3 {
    pub fn validate_against(
        &self,
        parent_pid: u32,
        helper: &PalsHelperProcessIdentityV3,
        r: &PalsResourcePolicyV3,
        policy: &PalsExternalCpuRPolicyV3,
    ) -> Result<(), ManifestError> {
        require(
            self.schema_version == 1
                && self.domain == "rz-pals-checker-ready-resources/1"
                && self.scope == "linux_ready_boundary_snapshot"
                && self.additional_allocation == "none_declared_not_an_enforcement_proof"
                && self.read_consistency
                    == "identity_membership_affinity_bracketed_limits_sequential_not_atomic",
            "external ready resource scope/domain differs",
        )?;
        require(
            parent_pid > 0
                && self.parent.pid == parent_pid
                && self.helper.parent_pid == parent_pid
                && helper.pid != parent_pid
                && helper.pid > 0
                && helper.pid <= i32::MAX as u32
                && helper.process_group == helper.pid
                && helper.proc_start_ticks > 0
                && helper.scope == "linux_spawn_observed_identity"
                && self.helper.pid == helper.pid
                && self.helper.process_group == helper.process_group
                && self.helper.process_group != self.parent.process_group
                && self.helper.proc_start_ticks_before == helper.proc_start_ticks,
            "external helper/parent historical identity differs",
        )?;
        let expected: BTreeSet<_> = r.cpu_affinity.iter().copied().collect();
        for process in [&self.parent, &self.helper] {
            require(
                process.pid > 0
                    && process.pid <= i32::MAX as u32
                    && process.parent_pid > 0
                    && process.process_group > 0
                    && process.process_group <= i32::MAX as u32
                    && process.proc_start_ticks_before > 0
                    && process.proc_start_ticks_before == process.proc_start_ticks_after
                    && process.membership_path_view == "observer_procfs_and_cgroup2_mount_view"
                    && ready_path(&process.cgroup_v2_membership)
                    && process
                        .cgroup_namespace_inode
                        .is_some_and(|inode| inode > 0)
                    && ready_affinity(&process.cpu_allowed_list)? == expected
                    && process.thread_observation_scope
                        == "bounded_ready_boundary_thread_snapshot_not_lifetime_enforcement"
                    && !process.threads.is_empty()
                    && process.threads.len() <= 64,
                "external ready process identity/affinity/scope invalid",
            )?;
            let mut tids = BTreeSet::new();
            for thread in &process.threads {
                require(
                    thread.tid > 0
                        && thread.tid <= i32::MAX as u32
                        && tids.insert(thread.tid)
                        && thread.proc_start_ticks_before > 0
                        && thread.proc_start_ticks_before == thread.proc_start_ticks_after
                        && ready_affinity(&thread.cpu_allowed_list)? == expected,
                    "external ready thread identity/affinity invalid",
                )?;
                if thread.tid == process.pid {
                    require(
                        thread.proc_start_ticks_before == process.proc_start_ticks_before,
                        "external ready leader thread identity differs",
                    )?;
                }
            }
            require(
                tids.contains(&process.pid),
                "external ready leader thread missing",
            )?;
            let limits = &process.cgroup_limits;
            let relative = if limits.mount_root == "/" {
                Some(process.cgroup_v2_membership.trim_start_matches('/'))
            } else if process.cgroup_v2_membership == limits.mount_root {
                Some("")
            } else {
                process
                    .cgroup_v2_membership
                    .strip_prefix(&format!("{}/", limits.mount_root.trim_end_matches('/')))
            };
            let resolved = relative.map(|relative| {
                if relative.is_empty() {
                    limits.mount_point.clone()
                } else {
                    format!("{}/{}", limits.mount_point.trim_end_matches('/'), relative)
                }
            });
            require(
                ready_path(&limits.mount_point)
                    && ready_path(&limits.mount_root)
                    && ready_path(&limits.resolved_directory)
                    && resolved.as_ref() == Some(&limits.resolved_directory)
                    && limits.memory_high
                        == (PalsReadyLimitV3::Numeric {
                            value: r.memory_high_bytes,
                        })
                    && limits.memory_max
                        == (PalsReadyLimitV3::Numeric {
                            value: r.memory_max_bytes,
                        })
                    && limits.memory_swap_max
                        == (PalsReadyLimitV3::Numeric {
                            value: r.swap_max_bytes,
                        })
                    && matches!(&limits.pids_max, PalsReadyLimitV3::Numeric { value } if *value > 0 && *value <= u64::from(policy.inherited_kernel_tasks_max)),
                "external ready direct cgroup limits differ from registered finite policy",
            )?;
        }
        require(
            self.parent.cgroup_v2_membership == self.helper.cgroup_v2_membership
                && self.parent.cgroup_namespace_inode == self.helper.cgroup_namespace_inode
                && self.parent.cgroup_limits == self.helper.cgroup_limits
                && ready_affinity(&self.parent.cpu_allowed_list)?
                    == ready_affinity(&self.helper.cpu_allowed_list)?
                && self.cgroup_membership_equal
                && self.cgroup_namespace_equal == Some(true)
                && self.cpu_allowed_list_equal,
            "external ready inherited membership/namespace/affinity differs",
        )
    }
}
impl PalsExternalCpuRReceiptV3 {
    pub fn validate_against(
        &self,
        e: &PalsEndpointV3,
        r: &PalsResourcePolicyV3,
    ) -> Result<(), ManifestError> {
        let PalsCpuRSelectionV3::ExternalUci(declaration) = &e.cpu_r else {
            return Err(ManifestError::Integrity(
                "own CPU_R cannot claim external evidence".into(),
            ));
        };
        require(
            self.profile_file_sha256 == declaration.profile.sha256
                && self.profile_canonical_sha256 == declaration.profile_canonical_sha256
                && self.registered_binary_sha256 == declaration.binary.sha256
                && sha(&self.registration_sha256, 64)
                && self.resolver == declaration.resolver
                && text(&self.model_value.model)
                && self.model_value.model.len() <= 1024
                && text(&self.model_value.semantics)
                && self.model_value.semantics.len() <= 1024
                && sha(&self.model_value.encoding, 64)
                && self.model_value.precision == "fp32",
            "external receipt profile/registration/resolver/model identity differs",
        )?;
        let checkpoint = match &e.model.weights {
            PalsWeightIdentityV3::Untrained { artifact, .. }
            | PalsWeightIdentityV3::Trained { artifact, .. } => artifact,
            _ => {
                return Err(ManifestError::Integrity(
                    "external model-WDL needs actual native checkpoint".into(),
                ));
            }
        };
        require(
            digest_hex(&self.model_value.model_epoch) == checkpoint.sha256,
            "external model-WDL checkpoint epoch differs",
        )?;
        let helper =
            self.shutdown.process_identity.as_ref().ok_or_else(|| {
                ManifestError::Integrity("external shutdown identity unknown".into())
            })?;
        self.ready_resources.validate_against(
            self.ready_resources.parent.pid,
            helper,
            r,
            &declaration.policy,
        )?;
        require(
            self.shutdown.exit_observed
                && self.shutdown.stdout_drained
                && self.shutdown.stderr_drained
                && self.shutdown.exit_code == Some(0)
                && self.shutdown.exit_signal.is_none()
                && self.shutdown.cleanup_complete
                && !self.shutdown.quarantined
                && !self.shutdown.ownership_lost
                && self
                    .shutdown
                    .stdout_bytes
                    .checked_add(self.shutdown.stderr_bytes)
                    .is_some_and(|bytes| bytes <= declaration.policy.lifetime_output_bytes_max),
            "external helper exit/known PGID/pipe closure unconfirmed",
        )?;
        require(
            matches!(self.applied_option_values, PalsObservedV3::Unknown)
                && matches!(self.model_loading, PalsObservedV3::Unknown)
                && matches!(self.work.completed_tasks, PalsObservedV3::Unknown)
                && matches!(
                    self.work.reused_completed_task_consumptions,
                    PalsObservedV3::Unknown
                ),
            "external unobserved options/model/completed/reused work cannot be invented",
        )?;
        if let (Some(tasks), Some(reports)) =
            (self.work.tasks_dispatched, self.work.reports_returned)
        {
            require(reports <= tasks, "external reports exceed dispatched tasks")?;
        }
        if let (Some(tasks), Some(reserved)) =
            (self.work.tasks_dispatched, self.work.node_budget_reserved)
        {
            require(
                u128::from(reserved)
                    <= u128::from(tasks) * u128::from(declaration.policy.max_nodes_per_task),
                "external reserved node budget exceeds declared task ceilings",
            )?;
        }
        Ok(())
    }
}
fn digest_hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsEndpointReceiptV3 {
    pub endpoint_id: String,
    /// Omitted/default None preserves every historical Own V3 receipt byte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_cpu_r: Option<PalsExternalCpuRReceiptV3>,
    pub uci_ready_observed: bool,
    /// Independently retained per-process search-work failed go returns.
    /// Historical absence is unknown, never an observed zero; foreign UCI
    /// endpoints have no RoveZero search-work projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_failed_go_count: Option<u64>,
    /// Actual selected identity only after both raw startup and termination
    /// markers were checked against the manifest/CLI. Absence in old receipts
    /// stays unknown and serializes to the same legacy bytes. This grants no
    /// recheck execution, completion, refutation or repaired-line authority.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present_pals_search_policy"
    )]
    pub pals_search_policy: Option<PalsSearchPolicyIdentityV3>,
    pub options: BTreeMap<String, PalsOptionReceiptV3>,
    pub actual_affinity: PalsObservedV3<Vec<u32>>,
    pub memory_peak_bytes: PalsObservedV3<u64>,
    pub gpu_device: PalsObservedV3<String>,
    pub vram_peak_bytes: PalsObservedV3<u64>,
    /// All physically completed native graph inputs/calls, including shared
    /// public-memory preparation and role reader graphs; not cache hits/visits.
    pub nn_inputs_completed: u64,
    pub nn_calls_completed: u64,
    /// Native role inputs accepted by search. Public-memory reuse is retained
    /// as a separate raw backend preparation/cache count.
    pub nn_inputs_consumed: u64,
    /// Reused native NN evaluations only, never CPU evidence or a visit count.
    pub cached_evaluations_consumed: u64,
    pub proposer_tasks_completed: u64,
    pub critic_tasks_completed: u64,
    /// RoveZero's own CPU_R task units. External UCI engine node/task statistics
    /// remain in external evidence and are not converted to these counters.
    pub cpu_tasks_requested: u64,
    pub cpu_tasks_completed: u64,
    /// Accepted consumers of previously completed TaskTable tasks. Reusing a
    /// depth-sufficient node estimate alone is recorded in raw work evidence.
    pub cpu_tasks_reused_consumed: u64,
    pub cpu_tasks_consumed: u64,
    pub cpu_nodes: u64,
    pub physical_state: PalsPhysicalStateV3,
    /// False while completion is unknown or a lease is quarantined.
    pub buffers_released: bool,
    pub process_exited: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsGameReceiptV3 {
    pub game_index: u32,
    pub white_endpoint: String,
    pub black_endpoint: String,
    pub base_ms: u64,
    pub increment_ms: u64,
    pub plies: u32,
    pub result: PalsResultV3,
    pub termination: PalsTerminationV3,
    /// An engine failure is charged as a loss and never silently omitted.
    pub failed_endpoint: Option<String>,
    pub engines: [PalsEndpointReceiptV3; 2],
    pub pgn: ArtifactRef,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRunReceiptV3 {
    pub domain: String,
    pub run_id: String,
    pub pair_id: String,
    pub lock_sha256: String,
    pub training_executed: bool,
    pub wall_time_ms: u64,
    pub cleanup_time_ms: u64,
    /// Incomplete/infra/quarantined runs preserve evidence without becoming accepted.
    pub pair_eligible: bool,
    /// Failures, including deadline overshoot, remain representable as evidence.
    pub failures: BTreeSet<PalsRunFailureV3>,
    pub games: Vec<PalsGameReceiptV3>,
}

fn observation<T>(o: &PalsObservedV3<T>) -> Result<(), ManifestError> {
    match o {
        PalsObservedV3::Unknown => Ok(()),
        PalsObservedV3::Observed { method, .. } => require(
            text(method) && !method.eq_ignore_ascii_case("readyok"),
            "observation requires an independent measurement method, not readyok",
        ),
    }
}
fn receipt_endpoint(
    o: &PalsEndpointReceiptV3,
    e: &PalsEngineV3,
    r: &PalsResourcePolicyV3,
    eligible: bool,
    failures: &BTreeSet<PalsRunFailureV3>,
) -> Result<(), ManifestError> {
    match e {
        PalsEngineV3::Pals(p) if p.search.semantic_id == PALS_POST_REPAIR_RECHECK_V3_VERSION => {
            pals_search_lane(p)?;
            o.pals_search_policy
                .as_ref()
                .ok_or_else(|| {
                    ManifestError::Integrity(
                        "PALS V3: selected post-Repair lane observation missing".into(),
                    )
                })?
                .validate()?;
        }
        _ => require(
            o.pals_search_policy.is_none(),
            "legacy/non-PALS/external endpoint cannot claim post-Repair lane observation",
        )?,
    }
    match (e, &o.external_cpu_r) {
        (PalsEngineV3::Pals(p), Some(external)) if !p.cpu_r.is_own() => {
            external.validate_against(p, r)?;
            if eligible {
                require(
                    external.work.tasks_dispatched.is_some()
                        && external.work.reports_returned.is_some()
                        && external.work.node_budget_reserved.is_some()
                        && external.work.consumed_completed_tasks.is_some()
                        && external.work.work_incomplete.is_some(),
                    "eligible external CPU_R lacks observed request/report/reservation/consumption accounting",
                )?;
            }
            require(
                o.cpu_tasks_requested == 0
                    && o.cpu_tasks_completed == 0
                    && o.cpu_tasks_reused_consumed == 0
                    && o.cpu_tasks_consumed == 0
                    && o.cpu_nodes == 0,
                "foreign CPU_R cannot be projected into own CPU counters",
            )?;
        }
        (PalsEngineV3::Pals(p), None) if !p.cpu_r.is_own() => {
            return Err(ManifestError::Integrity(
                "external CPU_R receipt projection missing".into(),
            ));
        }
        (_, None) => {}
        _ => {
            return Err(ManifestError::Integrity(
                "own/non-PALS endpoint cannot claim external CPU_R".into(),
            ));
        }
    }
    require(
        o.endpoint_id == e.id() && o.options.keys().eq(e.requested_options().keys()),
        "receipt endpoint/options differ from launch",
    )?;
    if o.search_failed_go_count.is_some_and(|failed| failed > 0) {
        require(
            !eligible && failures.contains(&PalsRunFailureV3::SearchFailure),
            "observed search failed go requires an ineligible run and preserved search failure",
        )?;
    }
    if eligible && !matches!(e, PalsEngineV3::ReferenceUci(_)) {
        require(
            o.search_failed_go_count == Some(0),
            "eligible RoveZero endpoint lacks observed zero search failed go count",
        )?;
    }
    if matches!(e, PalsEngineV3::ReferenceUci(_)) {
        require(
            o.search_failed_go_count.is_none(),
            "reference UCI endpoint cannot inherit RoveZero search failed go count",
        )?;
    }
    for (name, option) in &o.options {
        require(
            e.requested_options().get(name) == Some(&option.requested),
            "receipt requested option changed",
        )?;
        observation(&option.observed)?;
        if let PalsObservedV3::Observed { value, .. } = &option.observed {
            require(
                option.advertised_supported
                    && value.len() <= 1024
                    && !value.chars().any(char::is_control),
                "unsupported/invalid observed option",
            )?;
            if eligible {
                require(
                    value == &option.requested,
                    "eligible option differs from requested value",
                )?;
            }
        }
        if eligible {
            require(
                option.advertised_supported,
                "eligible run requested an unsupported option",
            )?;
        }
    }
    observation(&o.actual_affinity)?;
    observation(&o.memory_peak_bytes)?;
    observation(&o.gpu_device)?;
    observation(&o.vram_peak_bytes)?;
    if let PalsObservedV3::Observed { value, .. } = &o.actual_affinity {
        let observed: BTreeSet<_> = value.iter().collect();
        require(
            !value.is_empty() && value.len() <= 64 && observed.len() == value.len(),
            "observed CPU affinity invalid",
        )?;
        let requested: BTreeSet<_> = r.cpu_affinity.iter().collect();
        require(
            observed == requested
                || (!eligible && failures.contains(&PalsRunFailureV3::ResourceAdmission)),
            "observed CPU affinity differs from admitted resources",
        )?;
    }
    if let PalsObservedV3::Observed { value, .. } = &o.memory_peak_bytes {
        require(
            *value > 0
                && (*value <= r.memory_max_bytes
                    || (!eligible && failures.contains(&PalsRunFailureV3::ResourceAdmission))),
            "observed memory exceeds admitted resources without preserved failure",
        )?;
    }
    if let PalsObservedV3::Observed { value, .. } = &o.gpu_device {
        require(text(value), "empty observed GPU")?;
    }
    require(
        o.nn_inputs_consumed <= o.nn_inputs_completed
            && o.nn_calls_completed <= o.nn_inputs_completed
            && (o.nn_inputs_completed == 0) == (o.nn_calls_completed == 0)
            && o.cpu_tasks_completed
                .checked_add(o.cpu_tasks_reused_consumed)
                .is_some_and(|admitted| o.cpu_tasks_consumed <= admitted)
            && o.cpu_tasks_reused_consumed <= o.cpu_tasks_consumed
            && o.cpu_tasks_completed <= o.cpu_tasks_requested,
        "NN/task completion and consumption counters inconsistent",
    )?;
    let native = matches!(e, PalsEngineV3::Pals(p) if p.model.backend != PalsModelBackendV3::DeterministicMock);
    if let PalsEngineV3::Pals(p) = e {
        require(
            u128::from(o.nn_inputs_completed)
                <= u128::from(o.nn_calls_completed) * u128::from(p.model.max_batch_width),
            "NN input count exceeds declared physical batch width",
        )?;
    }
    if !native {
        require(
            o.nn_inputs_completed == 0 && o.nn_inputs_consumed == 0 && o.nn_calls_completed == 0,
            "mock/CPU/external endpoint cannot invent native NN counters",
        )?;
    }
    if !native {
        require(
            o.cached_evaluations_consumed == 0,
            "mock/CPU/external process cannot claim native NN cache consumption",
        )?;
    }
    if matches!(e, PalsEngineV3::Pals(p) if p.model.backend == PalsModelBackendV3::OrtCuda)
        && o.nn_inputs_completed > 0
    {
        require(
            matches!(o.gpu_device, PalsObservedV3::Observed { .. }),
            "CUDA execution lacks observed device evidence",
        )?;
    }
    if !matches!(e, PalsEngineV3::Pals(_)) {
        require(
            o.proposer_tasks_completed == 0
                && o.critic_tasks_completed == 0
                && o.cached_evaluations_consumed == 0,
            "non-PALS process cannot claim role/cache observations",
        )?;
    }
    require(
        !matches!(
            o.physical_state,
            PalsPhysicalStateV3::Unknown | PalsPhysicalStateV3::Quarantined
        ) || !o.buffers_released,
        "unknown/quarantined physical execution must retain buffers",
    )?;
    require(
        o.nn_inputs_completed == 0 || o.physical_state != PalsPhysicalStateV3::NotRequired,
        "native NN execution requires physical completion evidence",
    )?;
    if eligible {
        require(
            o.uci_ready_observed
                && o.process_exited
                && o.buffers_released
                && matches!(
                    o.physical_state,
                    PalsPhysicalStateV3::Completed | PalsPhysicalStateV3::NotRequired
                ),
            "eligible run lacks readiness, process exit or physical drain",
        )?;
    }
    Ok(())
}
impl PalsRunReceiptV3 {
    pub fn from_json(input: &str, lock: &PalsInputLockV3) -> Result<Self, ManifestError> {
        let r: Self = decode_json(input)?;
        r.validate_against(lock)?;
        Ok(r)
    }
    pub fn validate_against(&self, lock: &PalsInputLockV3) -> Result<(), ManifestError> {
        lock.verify()?;
        let m = &lock.manifest;
        require(
            self.domain == PALS_RECEIPT_V3_DOMAIN
                && self.run_id == m.run_id
                && self.pair_id == m.pair_id
                && self.lock_sha256 == lock.canonical_sha256
                && !self.training_executed,
            "receipt identity/lock mismatch or actual training claimed",
        )?;
        require(
            self.games.len() <= 2
                && (!self.pair_eligible || self.games.len() == 2)
                && self.pair_eligible == self.failures.is_empty(),
            "receipt eligibility/failure/game count inconsistent",
        )?;
        if self.wall_time_ms > m.pilot.wall_time_max_ms {
            require(
                !self.pair_eligible && self.failures.contains(&PalsRunFailureV3::WallDeadline),
                "wall deadline overshoot requires a preserved failure",
            )?;
        }
        if self.cleanup_time_ms > m.pilot.cleanup_max_ms {
            require(
                !self.pair_eligible && self.failures.contains(&PalsRunFailureV3::CleanupTimeout),
                "cleanup overshoot requires a preserved failure",
            )?;
        }
        let mut indices = BTreeSet::new();
        let mut helper_pids = BTreeSet::new();
        let mut helper_groups = BTreeSet::new();
        let mut helper_tuples = BTreeSet::new();
        let mut helper_parents = BTreeSet::new();
        for g in &self.games {
            require(
                g.game_index < 2
                    && indices.insert(g.game_index)
                    && g.white_endpoint == m.pilot.white_order[g.game_index as usize]
                    && g.white_endpoint != g.black_endpoint
                    && m.engines.iter().any(|e| e.id() == g.black_endpoint)
                    && g.base_ms == m.pilot.base_ms
                    && g.increment_ms == m.pilot.increment_ms
                    && g.plies <= m.pilot.max_plies,
                "game color, clock, index or ply limit mismatch",
            )?;
            g.pgn.validate()?;
            require(
                g.engines[0].endpoint_id != g.engines[1].endpoint_id,
                "duplicate endpoint receipts",
            )?;
            for (e, resources) in m.engines.iter().zip(&m.resources) {
                let o = g
                    .engines
                    .iter()
                    .find(|o| o.endpoint_id == e.id())
                    .ok_or_else(|| {
                        ManifestError::Integrity("PALS V3: missing endpoint receipt".into())
                    })?;
                receipt_endpoint(o, e, resources, self.pair_eligible, &self.failures)?;
                if let Some(external) = &o.external_cpu_r {
                    let identity =
                        external.shutdown.process_identity.as_ref().ok_or_else(|| {
                            ManifestError::Integrity(
                                "external helper historical identity missing".into(),
                            )
                        })?;
                    require(
                        helper_pids.insert(identity.pid)
                            && helper_groups.insert(identity.process_group)
                            && helper_tuples.insert((
                                identity.pid,
                                identity.process_group,
                                identity.proc_start_ticks,
                            ))
                            && helper_parents.insert(external.ready_resources.parent.pid),
                        "external helper or restarted parent historical identity reused",
                    )?;
                }
            }
            let incomplete = matches!(
                g.termination,
                PalsTerminationV3::PlyLimit
                    | PalsTerminationV3::RunDeadline
                    | PalsTerminationV3::InfrastructureFailure
            );
            require(
                incomplete == (g.result == PalsResultV3::Incomplete)
                    && (!self.pair_eligible || !incomplete),
                "incomplete/infra/ply-limit game cannot be an accepted draw",
            )?;
            match g.termination {
                PalsTerminationV3::EngineCrash
                | PalsTerminationV3::IllegalMove
                | PalsTerminationV3::TimeForfeit => {
                    let failed = g.failed_endpoint.as_ref().ok_or_else(|| {
                        ManifestError::Integrity("PALS V3: engine failure offender missing".into())
                    })?;
                    let expected = if failed == &g.white_endpoint {
                        PalsResultV3::BlackWin
                    } else if failed == &g.black_endpoint {
                        PalsResultV3::WhiteWin
                    } else {
                        return require(false, "failure offender not in game");
                    };
                    require(
                        g.result == expected,
                        "engine failure must be preserved as offender loss",
                    )?;
                }
                PalsTerminationV3::Checkmate => require(
                    g.failed_endpoint.is_none()
                        && matches!(g.result, PalsResultV3::WhiteWin | PalsResultV3::BlackWin),
                    "mate result inconsistent",
                )?,
                PalsTerminationV3::RulesDraw => require(
                    g.failed_endpoint.is_none() && g.result == PalsResultV3::Draw,
                    "rules draw result inconsistent",
                )?,
                _ => require(
                    g.failed_endpoint.is_none(),
                    "infrastructure/deadline is not an engine loss",
                )?,
            }
        }
        require(
            helper_pids.is_disjoint(&helper_parents) && helper_groups.is_disjoint(&helper_parents),
            "external helper historical identity collides with game parents",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Golden bytes of the own-only V3 fixture before CPU_R selection existed.
    const LEGACY_OWN_CANONICAL: &str = concat!(
        r#"["rz-pals-execution-v3/1",{"schema_version":3,"run_id":"pals-pilot-1","pair_id":"pair-1","contract_revision":"pals/0.1","rules_profile":"standard-complete-histo"#,
        r#"ry/1","comparison":"system","declared_changes":["endpoint"],"training_executed":false,"engines":[{"endpoint":"pals","configuration":{"id":"pals","binary":{"path"#,
        r#"":"bin/rovezero","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bytes":1,"source":"https://example.org/source","license":"MIT"},"s"#,
        r#"ource_commit":"cccccccccccccccccccccccccccccccccccccccc","model":{"architecture":"pals-width384-latent16-iterations2","input_schema":"entity-candidate-records/1"#,
        r#"","policy_head":"candidate-policy/1","value_head":"stm-wdl/1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","weight"#,
        r#"s":{"kind":"deterministic_mock","seed":1},"frozen_epoch":1,"backend":"deterministic_mock","precision":"fp32","max_batch_width":1,"exported_roles":["proposer","c"#,
        r#"ritic"]},"cpu":{"core":{"semantic_id":"rz-cpu-pvs/0.1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"#,
        r#""training_profile":{"semantic_id":"teacher-profile-1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},""#,
        r#"runtime_profile":{"semantic_id":"runtime-profile-1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"ev"#,
        r#"aluation":{"semantic_id":"untrained-material-pst-1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"ma"#,
        r#"x_nodes_per_task":10000,"max_depth":8,"max_task_ms":100,"max_tt_bytes":1024},"search":{"semantic_id":"pals","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbb"#,
        r#"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"runtime":{"semantic_id":"single-owner/1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"#,
        r#"bbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"pools":{"states":64,"line_chunks":64,"situations":64,"observations":256,"tasks":16,"role_states":2,"memory_pages":64,"q"#,
        r#"ueue_requests":16,"host_bytes":4096,"device_bytes":0},"requested_options":{"Ponder":"false"}}},{"endpoint":"own_cpu","configuration":{"id":"cpu","binary":{"path"#,
        r#"":"bin/rovezero","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bytes":1,"source":"https://example.org/source","license":"MIT"},"s"#,
        r#"ource_commit":"cccccccccccccccccccccccccccccccccccccccc","cpu":{"core":{"semantic_id":"rz-cpu-pvs/0.1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"#,
        r#"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"training_profile":{"semantic_id":"teacher-profile-1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"#,
        r#"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"runtime_profile":{"semantic_id":"runtime-profile-1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"#,
        r#"bbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"evaluation":{"semantic_id":"untrained-material-pst-1","implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"#,
        r#"bbbbbbbbbbbbbbbbbbbbbbbbbbbbb","options":{}},"max_nodes_per_task":10000,"max_depth":8,"max_task_ms":100,"max_tt_bytes":1024},"requested_options":{}}}],"resource"#,
        r#"s":[{"cpu_threads":2,"cpu_affinity":[0,2],"memory_high_bytes":6442450944,"memory_max_bytes":12884901888,"swap_max_bytes":0,"requested_gpu":null,"device_allocati"#,
        r#"on_max_bytes":0},{"cpu_threads":2,"cpu_affinity":[0,2],"memory_high_bytes":6442450944,"memory_max_bytes":12884901888,"swap_max_bytes":0,"requested_gpu":null,"de"#,
        r#"vice_allocation_max_bytes":0}],"pilot":{"position_command":"position startpos","games":2,"white_order":["pals","cpu"],"base_ms":120000,"increment_ms":1000,"max_"#,
        r#"plies":256,"seed":1,"wall_time_max_ms":900000,"cleanup_max_ms":30000,"handshake_max_ms":30000,"concurrent_games":1,"restart_processes_each_game":true,"ponder":f"#,
        r#"alse,"score_adjudication":false,"elo_claim":false}}]"#,
    );
    const LEGACY_OWN_SHA256: &str =
        "faf15aca4a8b3f28b4b569cd3d97dd50d57843b4a08c5e067eefe7bac5634219";
    fn asset(path: &str) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: "a".repeat(64),
            bytes: 1,
            source: "https://example.org/source".into(),
            license: "MIT".into(),
        }
    }
    fn comp(id: &str) -> PalsComponentV3 {
        PalsComponentV3 {
            semantic_id: id.into(),
            implementation_sha256: "b".repeat(64),
            options: BTreeMap::new(),
        }
    }
    fn own_cpu() -> PalsOwnCpuV3 {
        PalsOwnCpuV3 {
            core: comp("rz-cpu-pvs/0.1"),
            training_profile: comp("teacher-profile-1"),
            runtime_profile: comp("runtime-profile-1"),
            evaluation: comp("untrained-material-pst-1"),
            max_nodes_per_task: 10_000,
            max_depth: 8,
            max_task_ms: 100,
            max_tt_bytes: 1024,
        }
    }
    fn resource() -> PalsResourcePolicyV3 {
        PalsResourcePolicyV3 {
            cpu_threads: 2,
            cpu_affinity: vec![0, 2],
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
            swap_max_bytes: 0,
            requested_gpu: None,
            device_allocation_max_bytes: 0,
        }
    }
    fn pals(id: &str) -> PalsEngineV3 {
        PalsEngineV3::Pals(Box::new(PalsEndpointV3 {
            id: id.into(),
            binary: asset("bin/rovezero"),
            source_commit: "c".repeat(40),
            model: PalsModelIdentityV3 {
                architecture: "pals-width384-latent16-iterations2".into(),
                input_schema: "entity-candidate-records/1".into(),
                policy_head: "candidate-policy/1".into(),
                value_head: "stm-wdl/1".into(),
                implementation_sha256: "b".repeat(64),
                weights: PalsWeightIdentityV3::DeterministicMock { seed: 1 },
                frozen_epoch: 1,
                backend: PalsModelBackendV3::DeterministicMock,
                precision: PalsPrecisionV3::Fp32,
                max_batch_width: 1,
                exported_roles: BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
            },
            cpu: own_cpu(),
            cpu_r: PalsCpuRSelectionV3::Own,
            search: comp("pals"),
            runtime: comp("single-owner/1"),
            pools: PalsPoolLimitsV3 {
                states: 64,
                line_chunks: 64,
                situations: 64,
                observations: 256,
                tasks: 16,
                role_states: 2,
                memory_pages: 64,
                queue_requests: 16,
                host_bytes: 4096,
                device_bytes: 0,
            },
            requested_options: BTreeMap::from([("Ponder".into(), "false".into())]),
        }))
    }
    fn manifest() -> PalsRunManifestV3 {
        PalsRunManifestV3 {
            schema_version: 3,
            run_id: "pals-pilot-1".into(),
            pair_id: "pair-1".into(),
            contract_revision: "pals/0.1".into(),
            rules_profile: "standard-complete-history/1".into(),
            comparison: PalsComparisonV3::System,
            declared_changes: BTreeSet::from([PalsChangeAxisV3::Endpoint]),
            training_executed: false,
            engines: [
                pals("pals"),
                PalsEngineV3::OwnCpu(Box::new(PalsCpuEndpointV3 {
                    id: "cpu".into(),
                    binary: asset("bin/rovezero"),
                    source_commit: "c".repeat(40),
                    cpu: own_cpu(),
                    requested_options: BTreeMap::new(),
                })),
            ],
            resources: [resource(), resource()],
            pilot: PalsPilotV3 {
                position_command: "position startpos".into(),
                games: 2,
                white_order: ["pals".into(), "cpu".into()],
                base_ms: 120_000,
                increment_ms: 1000,
                max_plies: 256,
                seed: 1,
                wall_time_max_ms: 900_000,
                cleanup_max_ms: 30_000,
                handshake_max_ms: 30_000,
                concurrent_games: 1,
                restart_processes_each_game: true,
                ponder: false,
                score_adjudication: false,
                elo_claim: false,
            },
        }
    }
    fn endpoint_receipt(e: &PalsEngineV3) -> PalsEndpointReceiptV3 {
        PalsEndpointReceiptV3 {
            endpoint_id: e.id().into(),
            external_cpu_r: None,
            uci_ready_observed: true,
            search_failed_go_count: if matches!(e, PalsEngineV3::ReferenceUci(_)) {
                None
            } else {
                Some(0)
            },
            pals_search_policy: None,
            options: e
                .requested_options()
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        PalsOptionReceiptV3 {
                            requested: v.clone(),
                            advertised_supported: true,
                            observed: PalsObservedV3::Unknown,
                        },
                    )
                })
                .collect(),
            actual_affinity: PalsObservedV3::Unknown,
            memory_peak_bytes: PalsObservedV3::Unknown,
            gpu_device: PalsObservedV3::Unknown,
            vram_peak_bytes: PalsObservedV3::Unknown,
            nn_inputs_completed: 0,
            nn_calls_completed: 0,
            nn_inputs_consumed: 0,
            cached_evaluations_consumed: 0,
            proposer_tasks_completed: 0,
            critic_tasks_completed: 0,
            cpu_tasks_requested: 1,
            cpu_tasks_completed: 1,
            cpu_tasks_reused_consumed: 0,
            cpu_tasks_consumed: 1,
            cpu_nodes: 100,
            physical_state: PalsPhysicalStateV3::NotRequired,
            buffers_released: true,
            process_exited: true,
        }
    }
    fn external_receipt_fixture() -> (PalsEndpointV3, PalsExternalCpuRReceiptV3) {
        let PalsEngineV3::Pals(mut endpoint) = pals("pals") else {
            unreachable!()
        };
        endpoint.cpu_r = PalsCpuRSelectionV3::ExternalUci(Box::new(external_cpu_r_fixture()));
        endpoint.model.backend = PalsModelBackendV3::OrtCpu;
        let checkpoint = asset("checkpoint.pt");
        let mut epoch = [0; 32];
        for (i, byte) in epoch.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&checkpoint.sha256[i * 2..i * 2 + 2], 16).unwrap();
        }
        endpoint.model.weights = PalsWeightIdentityV3::Untrained {
            artifact: checkpoint,
            initialization_seed: 1,
        };
        let limits = PalsReadyCgroupLimitsV3 {
            mount_point: "/sys/fs/cgroup".into(),
            mount_root: "/".into(),
            resolved_directory: "/sys/fs/cgroup/test-only".into(),
            memory_high: PalsReadyLimitV3::Numeric { value: 6 << 30 },
            memory_max: PalsReadyLimitV3::Numeric { value: 12 << 30 },
            memory_swap_max: PalsReadyLimitV3::Numeric { value: 0 },
            pids_max: PalsReadyLimitV3::Numeric { value: 128 },
        };
        let process = |pid, parent_pid, ticks| PalsReadyProcessV3 {
            pid,
            parent_pid,
            process_group: pid,
            proc_start_ticks_before: ticks,
            proc_start_ticks_after: ticks,
            cgroup_v2_membership: "/test-only".into(),
            membership_path_view: "observer_procfs_and_cgroup2_mount_view".into(),
            cgroup_namespace_inode: Some(777),
            cpu_allowed_list: "0,2".into(),
            threads: vec![PalsReadyThreadV3 {
                tid: pid,
                proc_start_ticks_before: ticks,
                proc_start_ticks_after: ticks,
                cpu_allowed_list: "0,2".into(),
            }],
            thread_observation_scope:
                "bounded_ready_boundary_thread_snapshot_not_lifetime_enforcement".into(),
            cgroup_limits: limits.clone(),
        };
        let PalsCpuRSelectionV3::ExternalUci(declaration) = &endpoint.cpu_r else {
            unreachable!()
        };
        let receipt = PalsExternalCpuRReceiptV3 {
            profile_file_sha256: declaration.profile.sha256.clone(),
            profile_canonical_sha256: declaration.profile_canonical_sha256.clone(),
            registered_binary_sha256: declaration.binary.sha256.clone(),
            registration_sha256: "f".repeat(64),
            resolver: declaration.resolver.clone(),
            model_value: PalsModelValueIdentityV3 {
                semantics: "fixture-model-value".into(),
                model: "fixture-native-model".into(),
                encoding: "d".repeat(64),
                precision: "fp32".into(),
                model_epoch: epoch,
            },
            ready_resources: PalsCheckerReadyResourcesV3 {
                schema_version: 1,
                domain: "rz-pals-checker-ready-resources/1".into(),
                scope: "linux_ready_boundary_snapshot".into(),
                observed_unix_us: 10,
                observation_elapsed_us: 1,
                parent: process(100, 99, 500),
                helper: process(101, 100, 600),
                cgroup_membership_equal: true,
                cgroup_namespace_equal: Some(true),
                cpu_allowed_list_equal: true,
                additional_allocation: "none_declared_not_an_enforcement_proof".into(),
                read_consistency:
                    "identity_membership_affinity_bracketed_limits_sequential_not_atomic".into(),
            },
            shutdown: PalsHelperShutdownV3 {
                process_identity: Some(PalsHelperProcessIdentityV3 {
                    pid: 101,
                    process_group: 101,
                    proc_start_ticks: 600,
                    scope: "linux_spawn_observed_identity".into(),
                }),
                stop_sent: false,
                quit_sent: true,
                exit_observed: true,
                stdout_drained: true,
                stderr_drained: true,
                exit_code: Some(0),
                exit_signal: None,
                cleanup_complete: true,
                quarantined: false,
                ownership_lost: false,
                stdout_bytes: 100,
                stderr_bytes: 0,
            },
            work: PalsExternalCpuRWorkV3 {
                tasks_dispatched: Some(1),
                reports_returned: Some(1),
                node_budget_reserved: Some(256),
                nodes_observed: Some(20),
                consumed_completed_tasks: Some(3),
                work_incomplete: Some(true),
                completed_tasks: PalsObservedV3::Unknown,
                reused_completed_task_consumptions: PalsObservedV3::Unknown,
            },
            applied_option_values: PalsObservedV3::Unknown,
            model_loading: PalsObservedV3::Unknown,
        };
        (*endpoint, receipt)
    }
    #[test]
    fn external_receipt_preserves_unknown_foreign_units_and_own_json_compatibility() {
        let own = endpoint_receipt(&pals("pals"));
        let bytes = serde_json::to_vec(&own).unwrap();
        assert!(
            !serde_json::from_slice::<serde_json::Value>(&bytes)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("external_cpu_r")
        );
        assert_eq!(
            serde_json::from_slice::<PalsEndpointReceiptV3>(&bytes).unwrap(),
            own
        );
        let (endpoint, receipt) = external_receipt_fixture();
        receipt.validate_against(&endpoint, &resource()).unwrap();
        // Consumption includes reuse and nodes remain foreign reported work.
        let mut observed = receipt.clone();
        observed.work.nodes_observed = Some(512);
        observed.validate_against(&endpoint, &resource()).unwrap();
        observed.work.nodes_observed = None;
        observed.validate_against(&endpoint, &resource()).unwrap();
        let mut wire = serde_json::to_value(&receipt.work).unwrap();
        wire.as_object_mut().unwrap().remove("completed_tasks");
        wire.as_object_mut()
            .unwrap()
            .remove("reused_completed_task_consumptions");
        let decoded: PalsExternalCpuRWorkV3 = serde_json::from_value(wire).unwrap();
        assert!(matches!(decoded.completed_tasks, PalsObservedV3::Unknown));
        assert!(matches!(
            decoded.reused_completed_task_consumptions,
            PalsObservedV3::Unknown
        ));
        observed.work.completed_tasks = PalsObservedV3::Observed {
            value: 1,
            method: "reports".into(),
        };
        assert!(observed.validate_against(&endpoint, &resource()).is_err());
    }
    #[test]
    fn external_ready_resources_and_independent_closure_fail_closed() {
        let (endpoint, receipt) = external_receipt_fixture();
        let mutations: &[fn(&mut PalsExternalCpuRReceiptV3)] = &[
            |r| r.ready_resources.helper.proc_start_ticks_after += 1,
            |r| r.ready_resources.helper.cgroup_namespace_inode = None,
            |r| r.ready_resources.helper.cpu_allowed_list = "0".into(),
            |r| r.ready_resources.helper.cgroup_limits.memory_swap_max = PalsReadyLimitV3::Max,
            |r| r.ready_resources.helper.cgroup_limits.pids_max = PalsReadyLimitV3::Max,
            |r| {
                r.ready_resources.helper.cgroup_limits.resolved_directory =
                    "/sys/fs/cgroup/other".into()
            },
            |r| r.ready_resources.cgroup_membership_equal = false,
            |r| r.ready_resources.scope = "lifetime_enforcement".into(),
            |r| r.shutdown.stderr_drained = false,
            |r| r.shutdown.exit_code = Some(1),
            |r| r.shutdown.exit_signal = Some(9),
            |r| r.shutdown.ownership_lost = true,
            |r| r.model_value.model_epoch[0] ^= 1,
            |r| r.work.reports_returned = Some(2),
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let mut bad = receipt.clone();
            mutate(&mut bad);
            assert!(
                bad.validate_against(&endpoint, &resource()).is_err(),
                "mutation {index}"
            );
        }
        let mut four_threads = receipt;
        for tid in 102..105 {
            four_threads
                .ready_resources
                .helper
                .threads
                .push(PalsReadyThreadV3 {
                    tid,
                    proc_start_ticks_before: 600,
                    proc_start_ticks_after: 600,
                    cpu_allowed_list: "0,2".into(),
                });
        }
        four_threads
            .validate_against(&endpoint, &resource())
            .unwrap();
    }
    #[test]
    fn external_projection_requires_scoped_closure_and_keeps_own_units_separate() {
        let (endpoint, external) = external_receipt_fixture();
        let engine = PalsEngineV3::Pals(Box::new(endpoint));
        let mut output = endpoint_receipt(&engine);
        output.external_cpu_r = Some(external.clone());
        assert!(receipt_endpoint(&output, &engine, &resource(), false, &BTreeSet::new()).is_err());
        output.cpu_tasks_requested = 0;
        output.cpu_tasks_completed = 0;
        output.cpu_tasks_consumed = 0;
        output.cpu_nodes = 0;
        receipt_endpoint(&output, &engine, &resource(), false, &BTreeSet::new()).unwrap();
        assert!(
            receipt_endpoint(&output, &pals("pals"), &resource(), false, &BTreeSet::new()).is_err()
        );
        let mut m = manifest();
        m.engines[0] = engine;
        let lock = m.lock().unwrap();
        assert!(
            receipt(&lock)
                .validate_against(&lock)
                .unwrap_err()
                .to_string()
                .contains("external CPU_R receipt projection missing")
        );
        let mut projected = receipt(&lock);
        for (index, game) in projected.games.iter_mut().enumerate() {
            let mut endpoint = output.clone();
            let observed = endpoint.external_cpu_r.as_mut().unwrap();
            let parent = 100 + index as u32 * 2;
            let helper = parent + 1;
            observed.ready_resources.parent.pid = parent;
            observed.ready_resources.parent.process_group = parent;
            observed.ready_resources.parent.threads[0].tid = parent;
            observed.ready_resources.helper.pid = helper;
            observed.ready_resources.helper.process_group = helper;
            observed.ready_resources.helper.parent_pid = parent;
            observed.ready_resources.helper.threads[0].tid = helper;
            let identity = observed.shutdown.process_identity.as_mut().unwrap();
            identity.pid = helper;
            identity.process_group = helper;
            game.engines[0] = endpoint;
        }
        // This is typed consistency, not arena admission; Native provider,
        // supervised game mapping, PGN/clock and cleanup gates live in arena.
        projected.validate_against(&lock).unwrap();
        let mut duplicated = projected.clone();
        duplicated.games[1].engines[0] = duplicated.games[0].engines[0].clone();
        assert!(duplicated.validate_against(&lock).is_err());
        let mut unknown_closure = projected;
        unknown_closure.games[0].engines[0]
            .external_cpu_r
            .as_mut()
            .unwrap()
            .shutdown
            .stdout_drained = false;
        assert!(unknown_closure.validate_against(&lock).is_err());
    }
    fn receipt(l: &PalsInputLockV3) -> PalsRunReceiptV3 {
        let m = &l.manifest;
        PalsRunReceiptV3 {
            domain: PALS_RECEIPT_V3_DOMAIN.into(),
            run_id: m.run_id.clone(),
            pair_id: m.pair_id.clone(),
            lock_sha256: l.canonical_sha256.clone(),
            training_executed: false,
            wall_time_ms: 800_000,
            cleanup_time_ms: 1000,
            pair_eligible: true,
            failures: BTreeSet::new(),
            games: (0..2)
                .map(|i| PalsGameReceiptV3 {
                    game_index: i,
                    white_endpoint: m.pilot.white_order[i as usize].clone(),
                    black_endpoint: m.pilot.white_order[1 - i as usize].clone(),
                    base_ms: 120_000,
                    increment_ms: 1000,
                    plies: 100,
                    result: PalsResultV3::Draw,
                    termination: PalsTerminationV3::RulesDraw,
                    failed_endpoint: None,
                    engines: [
                        endpoint_receipt(&m.engines[0]),
                        endpoint_receipt(&m.engines[1]),
                    ],
                    pgn: asset(&format!("runs/game-{i}.pgn")),
                })
                .collect(),
        }
    }

    fn select_post_repair(m: &mut PalsRunManifestV3, index: usize) {
        let PalsEngineV3::Pals(p) = &mut m.engines[index] else {
            unreachable!()
        };
        p.search.semantic_id = PALS_POST_REPAIR_RECHECK_V3_VERSION.into();
        p.search.options = BTreeMap::from([(
            PALS_POST_REPAIR_RECHECK_V3_OPTION_KEY.into(),
            PALS_POST_REPAIR_RECHECK_V3_OPTION_VALUE.into(),
        )]);
    }
    fn policy_receipt(l: &PalsInputLockV3) -> PalsRunReceiptV3 {
        let mut r = receipt(l);
        for game in &mut r.games {
            for output in &mut game.engines {
                if l.manifest.engines.iter().any(|e| {
                    e.id() == output.endpoint_id
                        && matches!(e, PalsEngineV3::Pals(p)
                            if p.search.semantic_id == PALS_POST_REPAIR_RECHECK_V3_VERSION)
                }) {
                    // Synthetic typed fixture only: actual raw-pair observation
                    // and source/CLI admission belong to the arena consumer.
                    output.pals_search_policy =
                        Some(PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1());
                }
            }
        }
        r
    }

    #[test]
    fn post_repair_policy_enum_has_only_the_exact_explicit_selection() {
        let policy = PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1;
        let raw = serde_json::to_string(&policy).unwrap();
        assert_eq!(raw, "\"same-repaired-line-once-v1\"");
        assert_eq!(
            decode_json::<PalsPostRepairRecheckPolicyV3>(&raw).unwrap(),
            policy
        );
        for invalid in [
            "null",
            "false",
            "1",
            "\"disabled\"",
            "\"same_repaired_line_once_v1\"",
            "\"same-repaired-line-once-v2\"",
            "\"SAME-REPAIRED-LINE-ONCE-V1\"",
        ] {
            assert!(
                decode_json::<PalsPostRepairRecheckPolicyV3>(invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn post_repair_four_field_identity_is_closed_and_checks_the_fixed_digest() {
        let expected = PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1();
        expected.validate().unwrap();
        let raw = serde_json::to_string(&expected).unwrap();
        assert_eq!(
            decode_json::<PalsSearchPolicyIdentityV3>(&raw).unwrap(),
            expected
        );
        let wire = serde_json::to_value(&expected).unwrap();
        assert_eq!(wire.as_object().unwrap().len(), 4);
        for field in ["version", "policy", "search_identity", "conditions_sha256"] {
            let mut absent = wire.clone();
            absent.as_object_mut().unwrap().remove(field);
            assert!(decode_json::<PalsSearchPolicyIdentityV3>(&absent.to_string()).is_err());
            let mut null = wire.clone();
            null[field] = serde_json::Value::Null;
            assert!(decode_json::<PalsSearchPolicyIdentityV3>(&null.to_string()).is_err());
        }
        for field in ["version", "policy", "search_identity", "conditions"] {
            let mut wrong = expected.clone();
            match field {
                "version" => wrong.version = "pals".into(),
                "policy" => wrong.policy = PALS_POST_REPAIR_RECHECK_V3_OPTION_VALUE.into(),
                "search_identity" => {
                    wrong.search_identity = "pals-restricted-refinement/0.1".into()
                }
                "conditions" => wrong.conditions_sha256[31] ^= 1,
                _ => unreachable!(),
            }
            assert!(wrong.validate().is_err());
            assert!(
                decode_json::<PalsSearchPolicyIdentityV3>(&serde_json::to_string(&wrong).unwrap(),)
                    .is_err()
            );
        }
        let mut extra = wire.clone();
        extra["executed"] = false.into();
        assert!(decode_json::<PalsSearchPolicyIdentityV3>(&extra.to_string()).is_err());
        for invalid in [true.into(), 256.into(), (-1).into(), "190".into()] {
            let mut bad_byte = wire.clone();
            bad_byte["conditions_sha256"][0] = invalid;
            assert!(decode_json::<PalsSearchPolicyIdentityV3>(&bad_byte.to_string()).is_err());
        }
        let mut short = wire;
        short["conditions_sha256"].as_array_mut().unwrap().pop();
        assert!(decode_json::<PalsSearchPolicyIdentityV3>(&short.to_string()).is_err());
    }

    #[test]
    fn post_repair_identity_and_option_duplicate_bytes_are_rejected() {
        let raw = serde_json::to_string(
            &PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1(),
        )
        .unwrap();
        let duplicate = format!(
            "{{\"version\":\"{}\",{}",
            PALS_POST_REPAIR_RECHECK_V3_VERSION,
            &raw[1..],
        );
        assert!(decode_json::<PalsSearchPolicyIdentityV3>(&duplicate).is_err());
        assert!(serde_json::from_str::<PalsSearchPolicyIdentityV3>(&duplicate).is_err());
        let mut m = manifest();
        select_post_repair(&mut m, 0);
        let raw = serde_json::to_string(&m).unwrap();
        let one = "\"post_repair_recheck\":\"same-repaired-line-once-v1\"";
        let duplicate = raw.replacen(one, &format!("{one},{one}"), 1);
        assert_ne!(duplicate, raw);
        assert!(PalsRunManifestV3::from_json(&duplicate).is_err());
    }

    #[test]
    fn post_repair_manifest_requires_exact_option_and_own_cpu_r() {
        let original = manifest().lock().unwrap();
        let mut m = manifest();
        select_post_repair(&mut m, 0);
        let selected = m.lock().unwrap();
        assert_ne!(selected.canonical_sha256, original.canonical_sha256);
        assert_eq!(
            PalsInputLockV3::from_json(&serde_json::to_string(&selected).unwrap()).unwrap(),
            selected,
        );
        for invalid in [
            BTreeMap::new(),
            BTreeMap::from([("post_repair_recheck".into(), "disabled".into())]),
            BTreeMap::from([(
                "post_repair_recheck".into(),
                "same_repaired_line_once_v1".into(),
            )]),
            BTreeMap::from([(
                "Post_Repair_Recheck".into(),
                "same-repaired-line-once-v1".into(),
            )]),
            BTreeMap::from([
                (
                    "post_repair_recheck".into(),
                    "same-repaired-line-once-v1".into(),
                ),
                ("beam".into(), "1".into()),
            ]),
        ] {
            let mut bad = m.clone();
            let PalsEngineV3::Pals(p) = &mut bad.engines[0] else {
                unreachable!()
            };
            p.search.options = invalid;
            assert!(bad.validate().is_err());
        }
        let mut external = m.clone();
        select_external(&mut external);
        assert!(external.validate().is_err());
        let mut unknown = m;
        let PalsEngineV3::Pals(p) = &mut unknown.engines[0] else {
            unreachable!()
        };
        p.search.semantic_id = "pals-post-repair-recheck/2".into();
        assert!(unknown.validate().is_err());
        // Legacy declared search options retain their original generic meaning;
        // this does not make them supported by the closed executable recipe.
        let mut legacy = manifest();
        let PalsEngineV3::Pals(p) = &mut legacy.engines[0] else {
            unreachable!()
        };
        p.search
            .options
            .insert("legacy_declared_experiment".into(), "1".into());
        legacy.validate().unwrap();
    }

    #[test]
    fn post_repair_controlled_comparison_changes_only_search() {
        let mut m = manifest();
        m.engines[1] = pals("new");
        m.pilot.white_order = ["pals".into(), "new".into()];
        m.comparison = PalsComparisonV3::InternalSearch;
        m.declared_changes = BTreeSet::from([PalsChangeAxisV3::Search]);
        select_post_repair(&mut m, 1);
        let lock = m.lock().unwrap();
        assert_eq!(changes(&m.engines[0], &m.engines[1]), m.declared_changes);
        assert_eq!(lock.manifest.resources[0], lock.manifest.resources[1]);
        let (PalsEngineV3::Pals(old), PalsEngineV3::Pals(new)) = (&m.engines[0], &m.engines[1])
        else {
            unreachable!()
        };
        assert_eq!(old.model, new.model);
        assert_eq!(old.cpu, new.cpu);
        assert_eq!(old.runtime, new.runtime);
        assert_eq!(old.pools, new.pools);
        let run = policy_receipt(&lock);
        run.validate_against(&lock).unwrap();
        assert!(run.games[0].engines[0].pals_search_policy.is_none());
        assert!(run.games[0].engines[1].pals_search_policy.is_some());
    }

    #[test]
    fn selected_core_policy_is_required_even_for_failed_or_unknown_work() {
        let mut m = manifest();
        select_post_repair(&mut m, 0);
        let lock = m.lock().unwrap();
        assert!(receipt(&lock).validate_against(&lock).is_err());
        let mut run = policy_receipt(&lock);
        run.validate_against(&lock).unwrap();
        run.pair_eligible = false;
        run.failures.insert(PalsRunFailureV3::Infrastructure);
        run.games[0].engines[0].search_failed_go_count = None;
        run.validate_against(&lock).unwrap();
        run.games[0].engines[0].pals_search_policy = None;
        assert!(run.validate_against(&lock).is_err());
        run.games[0].engines[0].pals_search_policy =
            Some(PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1());
        run.games[0].engines[0].search_failed_go_count = Some(1);
        assert!(run.validate_against(&lock).is_err());
        run.failures.insert(PalsRunFailureV3::SearchFailure);
        run.validate_against(&lock).unwrap();
    }

    #[test]
    fn core_policy_cannot_be_projected_into_legacy_cpu_reference_or_external() {
        let mut engines = vec![pals("legacy"), manifest().engines[1].clone()];
        engines.push(PalsEngineV3::ReferenceUci(Box::new(
            PalsReferenceEndpointV3 {
                id: "reference".into(),
                family: "fixture-reference".into(),
                version: "1".into(),
                binary: asset("bin/reference"),
                source_url: "https://example.org/fixture".into(),
                source_commit: "c".repeat(40),
                assets: vec![],
                requested_options: BTreeMap::new(),
            },
        )));
        for engine in engines {
            let mut output = endpoint_receipt(&engine);
            receipt_endpoint(&output, &engine, &resource(), false, &BTreeSet::new()).unwrap();
            output.pals_search_policy =
                Some(PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1());
            assert!(
                receipt_endpoint(&output, &engine, &resource(), false, &BTreeSet::new()).is_err()
            );
        }
        let (external, helper) = external_receipt_fixture();
        let engine = PalsEngineV3::Pals(Box::new(external));
        let mut output = endpoint_receipt(&engine);
        output.external_cpu_r = Some(helper);
        output.pals_search_policy =
            Some(PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1());
        assert!(receipt_endpoint(&output, &engine, &resource(), false, &BTreeSet::new()).is_err());
    }

    #[test]
    fn core_policy_wire_rejects_present_null_duplicate_and_bad_identity() {
        let mut m = manifest();
        select_post_repair(&mut m, 0);
        let run = policy_receipt(&m.lock().unwrap());
        let output = &run.games[0].engines[0];
        let raw = serde_json::to_string(output).unwrap();
        assert_eq!(decode_json::<PalsEndpointReceiptV3>(&raw).unwrap(), *output);
        let mut null = serde_json::to_value(output).unwrap();
        null["pals_search_policy"] = serde_json::Value::Null;
        assert!(decode_json::<PalsEndpointReceiptV3>(&null.to_string()).is_err());
        let identity = serde_json::to_string(output.pals_search_policy.as_ref().unwrap()).unwrap();
        let duplicate = raw.replacen(
            "\"pals_search_policy\":",
            &format!("\"pals_search_policy\":{identity},\"pals_search_policy\":"),
            1,
        );
        assert_ne!(duplicate, raw);
        assert!(decode_json::<PalsEndpointReceiptV3>(&duplicate).is_err());
        let mut bad = output.clone();
        bad.pals_search_policy.as_mut().unwrap().conditions_sha256[0] ^= 1;
        assert!(
            decode_json::<PalsEndpointReceiptV3>(&serde_json::to_string(&bad).unwrap()).is_err()
        );
    }

    #[test]
    fn legacy_core_omission_preserves_bytes_and_existing_null_unknowns() {
        let engine = pals("pals");
        let output = endpoint_receipt(&engine);
        let bytes = serde_json::to_vec(&output).unwrap();
        let wire: serde_json::Value = decode_json(std::str::from_utf8(&bytes).unwrap()).unwrap();
        assert!(wire.get("pals_search_policy").is_none());
        let decoded: PalsEndpointReceiptV3 =
            decode_json(std::str::from_utf8(&bytes).unwrap()).unwrap();
        assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
        let mut historical_nulls = wire;
        historical_nulls["external_cpu_r"] = serde_json::Value::Null;
        historical_nulls["search_failed_go_count"] = serde_json::Value::Null;
        let unknown: PalsEndpointReceiptV3 = decode_json(&historical_nulls.to_string()).unwrap();
        assert!(unknown.external_cpu_r.is_none());
        assert!(unknown.search_failed_go_count.is_none());
        assert!(unknown.pals_search_policy.is_none());
        receipt_endpoint(&unknown, &engine, &resource(), false, &BTreeSet::new()).unwrap();
        assert!(receipt_endpoint(&unknown, &engine, &resource(), true, &BTreeSet::new()).is_err());
        historical_nulls["pals_search_policy"] = serde_json::Value::Null;
        assert!(decode_json::<PalsEndpointReceiptV3>(&historical_nulls.to_string()).is_err());
    }

    #[test]
    fn selected_policy_does_not_promote_failed_go_or_rewrite_game_facts() {
        let mut m = manifest();
        select_post_repair(&mut m, 0);
        let lock = m.lock().unwrap();
        let mut run = policy_receipt(&lock);
        let original_games = run.games.clone();
        run.games[0].engines[0].search_failed_go_count = Some(6);
        assert!(run.validate_against(&lock).is_err());
        run.pair_eligible = false;
        run.failures.insert(PalsRunFailureV3::Infrastructure);
        assert!(run.validate_against(&lock).is_err());
        run.failures.insert(PalsRunFailureV3::SearchFailure);
        run.validate_against(&lock).unwrap();
        for (actual, original) in run.games.iter().zip(original_games) {
            assert_eq!(actual.pgn, original.pgn);
            assert_eq!(actual.result, original.result);
            assert_eq!(actual.termination, original.termination);
            assert_eq!(
                actual.engines[0].pals_search_policy,
                original.engines[0].pals_search_policy
            );
        }
        let bytes = serde_json::to_string(&run).unwrap();
        assert_eq!(PalsRunReceiptV3::from_json(&bytes, &lock).unwrap(), run);
    }

    #[test]
    fn historical_search_failure_count_stays_unknown_and_rejects_positive_admission() {
        let engine = pals("pals");
        let mut legacy = serde_json::to_value(endpoint_receipt(&engine)).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("search_failed_go_count");
        let decoded: PalsEndpointReceiptV3 = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(decoded.search_failed_go_count, None);
        assert_eq!(serde_json::to_value(&decoded).unwrap(), legacy);
        receipt_endpoint(&decoded, &engine, &resource(), false, &BTreeSet::new()).unwrap();
        assert!(receipt_endpoint(&decoded, &engine, &resource(), true, &BTreeSet::new()).is_err());

        let mut null = legacy.clone();
        null["search_failed_go_count"] = serde_json::Value::Null;
        let decoded: PalsEndpointReceiptV3 = serde_json::from_value(null).unwrap();
        assert_eq!(decoded.search_failed_go_count, None);
        for invalid in [true.into(), (-1).into(), "0".into()] {
            let mut bad = legacy.clone();
            bad["search_failed_go_count"] = invalid;
            assert!(serde_json::from_value::<PalsEndpointReceiptV3>(bad).is_err());
        }
    }

    #[test]
    fn observed_failed_go_cannot_become_an_eligible_completed_pair() {
        let lock = manifest().lock().unwrap();
        let mut run = receipt(&lock);
        run.validate_against(&lock).unwrap();
        let original_games = run.games.clone();
        run.games[0].engines[0].search_failed_go_count = Some(6);
        assert!(run.validate_against(&lock).is_err());
        run.pair_eligible = false;
        run.failures.insert(PalsRunFailureV3::Infrastructure);
        assert!(run.validate_against(&lock).is_err());
        run.failures.insert(PalsRunFailureV3::SearchFailure);
        run.validate_against(&lock).unwrap();
        assert_eq!(run.games[0].pgn, original_games[0].pgn);
        assert_eq!(run.games[0].result, original_games[0].result);
        assert_eq!(run.games[0].termination, original_games[0].termination);
        assert_eq!(run.games[0].engines[0].search_failed_go_count, Some(6));
        let wire = serde_json::to_string(&run).unwrap();
        assert_eq!(PalsRunReceiptV3::from_json(&wire, &lock).unwrap(), run);
        assert!(wire.contains("search_failure"));
        run.pair_eligible = true;
        assert!(run.validate_against(&lock).is_err());
    }

    #[test]
    fn unknown_rove_search_count_is_not_a_zero_observation() {
        let lock = manifest().lock().unwrap();
        let mut run = receipt(&lock);
        run.games[1].engines[1].search_failed_go_count = None;
        assert!(run.validate_against(&lock).is_err());
        run.pair_eligible = false;
        run.failures.insert(PalsRunFailureV3::Infrastructure);
        run.validate_against(&lock).unwrap();
        assert_eq!(run.games[1].engines[1].search_failed_go_count, None);
    }

    fn external_cpu_r_fixture() -> PalsExternalCpuRV3 {
        let mut binary = asset("bin/stockfish");
        binary.source = "https://github.com/official-stockfish/Stockfish".into();
        binary.license = "GPL-3.0-or-later".into();
        PalsExternalCpuRV3 {
            selection: PalsExternalCpuRSelectionV3::StockfishEmbeddedNnue,
            profile: asset("profiles/stockfish.json"),
            profile_canonical_sha256: "d".repeat(64),
            binary,
            resolver: PalsExternalCpuRResolverV3 {
                version: "pals-model-wdl-restricted/0.1".into(),
                semantics_sha256: "e".repeat(64),
            },
            policy: PalsExternalCpuRPolicyV3 {
                resource_scope: PalsExternalCpuRResourceScopeV3::InheritedParentCgroup,
                max_owners: 1,
                max_active_tasks: 1,
                max_process_leaders: 1,
                inherited_kernel_tasks_max: 128,
                threads_max: 2,
                hash_mib_max: 16,
                max_depth: 8,
                max_prefix_plies: 64,
                max_nodes_per_task: 10_000,
                handshake_max_ms: 1000,
                task_wall_time_max_ms: 1000,
                stop_grace_max_ms: 100,
                shutdown_grace_max_ms: 100,
                lifetime_output_bytes_max: 4096,
                line_bytes_max: 1024,
            },
        }
    }
    fn select_external(m: &mut PalsRunManifestV3) {
        let PalsEngineV3::Pals(p) = &mut m.engines[0] else {
            unreachable!()
        };
        p.cpu_r = PalsCpuRSelectionV3::ExternalUci(Box::new(external_cpu_r_fixture()));
    }
    #[test]
    fn own_cpu_r_omission_preserves_legacy_canonical_bytes_and_digest() {
        let (domain, legacy): (String, PalsRunManifestV3) =
            serde_json::from_str(LEGACY_OWN_CANONICAL).unwrap();
        assert_eq!(domain, PALS_MANIFEST_V3_DOMAIN);
        assert_eq!(legacy, manifest());
        assert_eq!(
            legacy.canonical_bytes().unwrap(),
            LEGACY_OWN_CANONICAL.as_bytes()
        );
        assert_eq!(legacy.lock().unwrap().canonical_sha256, LEGACY_OWN_SHA256);
        let mut value = serde_json::to_value(&legacy).unwrap();
        assert!(value["engines"][0]["configuration"].get("cpu_r").is_none());
        value["engines"][0]["configuration"]["cpu_r"] =
            serde_json::to_value(PalsCpuRSelectionV3::Own).unwrap();
        let explicit = PalsRunManifestV3::from_json(&value.to_string()).unwrap();
        assert_eq!(
            explicit.canonical_bytes().unwrap(),
            LEGACY_OWN_CANONICAL.as_bytes()
        );
        assert_eq!(explicit.lock().unwrap().canonical_sha256, LEGACY_OWN_SHA256);
    }
    #[test]
    fn external_cpu_r_is_an_explicit_preparation_lock_not_own_runtime_evidence() {
        let mut m = manifest();
        let original = m.lock().unwrap();
        let PalsEngineV3::Pals(p) = &m.engines[0] else {
            unreachable!()
        };
        let cpu_t_and_legacy = p.cpu.clone();
        select_external(&mut m);
        let lock = m.lock().unwrap();
        assert_ne!(lock.canonical_sha256, original.canonical_sha256);
        assert_eq!(
            PalsInputLockV3::from_json(&serde_json::to_string(&lock).unwrap()).unwrap(),
            lock,
        );
        let PalsEngineV3::Pals(p) = &lock.manifest.engines[0] else {
            unreachable!()
        };
        assert_eq!(p.cpu, cpu_t_and_legacy);
        let value = serde_json::to_value(&lock.manifest).unwrap();
        assert_eq!(
            value["engines"][0]["configuration"]["cpu_r"]["kind"],
            "external_uci"
        );
        let error = receipt(&lock)
            .validate_against(&lock)
            .unwrap_err()
            .to_string();
        assert!(error.contains("external CPU_R receipt projection missing"));
        let mut failed = receipt(&lock);
        failed.pair_eligible = false;
        failed.failures.insert(PalsRunFailureV3::Infrastructure);
        assert!(failed.validate_against(&lock).is_err());
    }
    #[test]
    fn external_cpu_r_declarations_require_finite_inherited_stockfish_caps() {
        let mutations: &[fn(&mut PalsExternalCpuRV3)] = &[
            |c| c.profile.bytes = 0,
            |c| c.profile.bytes = 64 * 1024 + 1,
            |c| c.profile_canonical_sha256 = "not-a-digest".into(),
            |c| c.binary.bytes = 256 * 1024 * 1024 + 1,
            |c| c.binary.source = "https://example.org/other-engine".into(),
            |c| c.binary.license = "MIT".into(),
            |c| c.resolver.version = "pals-cpu-raw-restricted/0.1".into(),
            |c| c.resolver.semantics_sha256 = "e".repeat(63),
            |c| c.policy.max_owners = 2,
            |c| c.policy.max_active_tasks = 2,
            |c| c.policy.max_process_leaders = 2,
            |c| c.policy.inherited_kernel_tasks_max = 0,
            |c| c.policy.inherited_kernel_tasks_max = 65_537,
            |c| c.policy.threads_max = 3,
            |c| c.policy.hash_mib_max = 1025,
            |c| c.policy.max_depth = 65,
            |c| c.policy.max_prefix_plies = 4097,
            |c| c.policy.max_nodes_per_task = 0,
            |c| c.policy.handshake_max_ms = 30_001,
            |c| c.policy.task_wall_time_max_ms = 180_001,
            |c| c.policy.stop_grace_max_ms = 30_001,
            |c| c.policy.shutdown_grace_max_ms = 30_001,
            |c| c.policy.lifetime_output_bytes_max = 16 * 1024 * 1024 + 1,
            |c| c.policy.line_bytes_max = 4097,
        ];
        assert!(external_cpu_r(&external_cpu_r_fixture(), &resource(), &manifest().pilot).is_ok());
        for (index, mutate) in mutations.iter().enumerate() {
            let mut c = external_cpu_r_fixture();
            mutate(&mut c);
            assert!(
                external_cpu_r(&c, &resource(), &manifest().pilot).is_err(),
                "accepted external mutation {index}",
            );
        }
        let mut r = resource();
        r.cpu_threads = 1;
        assert!(external_cpu_r(&external_cpu_r_fixture(), &r, &manifest().pilot).is_err());
        r.cpu_threads = 2;
        r.memory_high_bytes = 1024;
        assert!(external_cpu_r(&external_cpu_r_fixture(), &r, &manifest().pilot).is_err());
    }
    #[test]
    fn external_cpu_r_rejects_unknown_selection_and_nested_policy_fields() {
        let mut m = manifest();
        select_external(&mut m);
        let original = serde_json::to_value(&m).unwrap();
        for (field, value) in [
            ("selection", serde_json::json!("general_uci")),
            ("unexpected", serde_json::json!(true)),
        ] {
            let mut wire = original.clone();
            wire["engines"][0]["configuration"]["cpu_r"]["configuration"][field] = value;
            assert!(PalsRunManifestV3::from_json(&wire.to_string()).is_err());
        }
        let mut wire = original.clone();
        wire["engines"][0]["configuration"]["cpu_r"]["configuration"]["policy"]["resource_scope"] =
            "independent_resources".into();
        assert!(PalsRunManifestV3::from_json(&wire.to_string()).is_err());
        let mut wire = original;
        wire["engines"][0]["configuration"]["cpu_r"]["configuration"]["policy"]["extra_memory"] =
            1.into();
        assert!(PalsRunManifestV3::from_json(&wire.to_string()).is_err());
    }
    #[test]
    fn runtime_checker_change_is_not_hidden_as_a_controlled_model_change() {
        let mut m = manifest();
        m.engines[1] = pals("candidate");
        m.pilot.white_order[1] = "candidate".into();
        select_external(&mut m);
        m.declared_changes = BTreeSet::from([PalsChangeAxisV3::CpuCore]);
        assert!(m.validate().is_ok());
        m.comparison = PalsComparisonV3::InternalModel;
        m.declared_changes = BTreeSet::from([PalsChangeAxisV3::Model]);
        assert!(m.validate().is_err());
    }

    #[test]
    fn legacy_domains_do_not_parse_or_lock_as_pals() {
        let mut m = manifest();
        m.schema_version = 2;
        assert!(m.lock().is_err());
        m.schema_version = 3;
        let mut l = m.lock().unwrap();
        l.domain = "rz-e01-model-endpoints-v2".into();
        assert!(l.verify().is_err());
        let mut l = m.lock().unwrap();
        l.manifest.pilot.seed += 1;
        assert!(l.verify().is_err());
    }
    #[test]
    fn lock_roundtrip_rejects_unknown_and_duplicate_fields() {
        let l = manifest().lock().unwrap();
        let json = serde_json::to_string(&l).unwrap();
        assert_eq!(PalsInputLockV3::from_json(&json).unwrap(), l);
        let duplicate = json.replacen("\"domain\":", "\"domain\":\"x\",\"domain\":", 1);
        assert!(PalsInputLockV3::from_json(&duplicate).is_err());
        let mut value = serde_json::to_value(&l).unwrap();
        value["manifest"]["training_steps"] = 10.into();
        assert!(PalsInputLockV3::from_json(&value.to_string()).is_err());
    }
    #[test]
    fn whole_system_change_is_not_an_internal_model_claim() {
        let mut m = manifest();
        m.comparison = PalsComparisonV3::InternalModel;
        assert!(m.validate().is_err());
        m.engines[1] = pals("candidate");
        m.pilot.white_order[1] = "candidate".into();
        let PalsEngineV3::Pals(b) = &mut m.engines[1] else {
            unreachable!()
        };
        b.model.policy_head = "new-head/2".into();
        m.declared_changes = BTreeSet::from([PalsChangeAxisV3::Model]);
        assert!(m.validate().is_ok());
        let PalsEngineV3::Pals(b) = &mut m.engines[1] else {
            unreachable!()
        };
        b.search.options.insert("widen".into(), "2".into());
        assert!(m.validate().is_err());
        m.comparison = PalsComparisonV3::System;
        m.declared_changes.insert(PalsChangeAxisV3::Search);
        assert!(m.validate().is_ok());
    }
    #[test]
    fn runtime_control_holds_resources_and_model_fixed() {
        let mut m = manifest();
        m.engines[1] = pals("candidate");
        m.pilot.white_order[1] = "candidate".into();
        m.comparison = PalsComparisonV3::Runtime;
        let PalsEngineV3::Pals(b) = &mut m.engines[1] else {
            unreachable!()
        };
        b.runtime.options.insert("queue".into(), "2".into());
        m.declared_changes = BTreeSet::from([PalsChangeAxisV3::Runtime]);
        assert!(m.validate().is_ok());
        m.resources[1].memory_max_bytes += 1;
        assert!(m.validate().is_err());
    }
    #[test]
    fn product_is_v_free_and_execution_finite() {
        let mut m = manifest();
        let PalsEngineV3::Pals(e) = &mut m.engines[0] else {
            unreachable!()
        };
        e.model.exported_roles.insert(PalsRoleV3::Validator);
        assert!(m.validate().is_err());
        let mut m = manifest();
        m.training_executed = true;
        assert!(m.validate().is_err());
        let mut m = manifest();
        let PalsEngineV3::Pals(e) = &mut m.engines[0] else {
            unreachable!()
        };
        e.cpu.max_nodes_per_task = 0;
        assert!(m.validate().is_err());
        let mut m = manifest();
        m.pilot.elo_claim = true;
        assert!(m.validate().is_err());
        let mut m = manifest();
        m.pilot.wall_time_max_ms += 1;
        assert!(m.validate().is_err());
    }
    #[test]
    fn unknown_options_and_vram_remain_unknown() {
        let l = manifest().lock().unwrap();
        let mut r = receipt(&l);
        assert!(r.validate_against(&l).is_ok());
        r.games[0].engines[0]
            .options
            .get_mut("Ponder")
            .unwrap()
            .observed = PalsObservedV3::Observed {
            value: "false".into(),
            method: "readyok".into(),
        };
        assert!(r.validate_against(&l).is_err());
    }
    #[test]
    fn mock_cannot_claim_nn_and_unknown_completion_cannot_free_buffers() {
        let l = manifest().lock().unwrap();
        let mut r = receipt(&l);
        r.games[0].engines[0].nn_inputs_completed = 1;
        r.games[0].engines[0].nn_calls_completed = 1;
        assert!(r.validate_against(&l).is_err());
        let mut r = receipt(&l);
        r.pair_eligible = false;
        r.failures
            .insert(PalsRunFailureV3::PhysicalCompletionUnknown);
        r.games[0].engines[0].physical_state = PalsPhysicalStateV3::Quarantined;
        assert!(r.validate_against(&l).is_err());
        r.games[0].engines[0].buffers_released = false;
        assert!(r.validate_against(&l).is_ok());
        r.pair_eligible = true;
        r.failures.clear();
        assert!(r.validate_against(&l).is_err());
    }
    #[test]
    fn pals_receipt_cpu_task_reuse_is_not_nn_cache_or_new_physical_work() {
        let l = manifest().lock().unwrap();
        let mut r = receipt(&l);
        r.games[0].engines[0].cpu_tasks_requested = 1;
        r.games[0].engines[0].cpu_tasks_completed = 1;
        r.games[0].engines[0].cpu_tasks_reused_consumed = 2;
        r.games[0].engines[0].cpu_tasks_consumed = 3;
        assert!(r.validate_against(&l).is_ok());
        r.games[0].engines[0].cpu_tasks_reused_consumed = 1;
        assert!(r.validate_against(&l).is_err());
        r.games[0].engines[0].cpu_tasks_reused_consumed = 2;
        r.games[0].engines[0].cached_evaluations_consumed = 2;
        assert!(r.validate_against(&l).is_err());
        r.games[0].engines[0].cached_evaluations_consumed = 0;
        r.domain = "rz-pals-receipt-v3/1".into();
        assert!(r.validate_against(&l).is_err());
        r.domain = PALS_RECEIPT_V3_DOMAIN.into();
        let mut value = serde_json::to_value(&r).unwrap();
        value["games"][0]["engines"][0]
            .as_object_mut()
            .unwrap()
            .remove("cpu_tasks_reused_consumed");
        assert!(PalsRunReceiptV3::from_json(&value.to_string(), &l).is_err());
    }
    #[test]
    fn engine_failure_is_a_loss_not_an_omitted_or_drawn_game() {
        let l = manifest().lock().unwrap();
        let mut r = receipt(&l);
        r.games[0].termination = PalsTerminationV3::EngineCrash;
        r.games[0].failed_endpoint = Some("pals".into());
        assert!(r.validate_against(&l).is_err());
        r.games[0].result = PalsResultV3::BlackWin;
        assert!(r.validate_against(&l).is_ok());
        r.games[0].termination = PalsTerminationV3::InfrastructureFailure;
        r.games[0].failed_endpoint = None;
        r.games[0].result = PalsResultV3::Incomplete;
        assert!(r.validate_against(&l).is_err());
        r.pair_eligible = false;
        r.failures.insert(PalsRunFailureV3::Infrastructure);
        assert!(r.validate_against(&l).is_ok());
    }
    #[test]
    fn receipt_binds_exact_clock_pair_colors_and_task_consumption() {
        let l = manifest().lock().unwrap();
        let mut r = receipt(&l);
        r.games[1].white_endpoint = "pals".into();
        assert!(r.validate_against(&l).is_err());
        let mut r = receipt(&l);
        r.games[0].increment_ms = 100;
        assert!(r.validate_against(&l).is_err());
        let mut r = receipt(&l);
        r.games[0].engines[0].cpu_tasks_consumed = 2;
        assert!(r.validate_against(&l).is_err());
        let mut r = receipt(&l);
        r.games.pop();
        assert!(r.validate_against(&l).is_err());
        r.pair_eligible = false;
        r.failures.insert(PalsRunFailureV3::IncompleteGame);
        assert!(r.validate_against(&l).is_ok());
    }
    #[test]
    fn overshoot_is_preserved_as_failure_evidence() {
        let l = manifest().lock().unwrap();
        let mut r = receipt(&l);
        r.wall_time_ms = l.manifest.pilot.wall_time_max_ms + 100;
        assert!(r.validate_against(&l).is_err());
        r.pair_eligible = false;
        r.failures.insert(PalsRunFailureV3::WallDeadline);
        assert!(r.validate_against(&l).is_ok());
        r.cleanup_time_ms = l.manifest.pilot.cleanup_max_ms + 1;
        assert!(r.validate_against(&l).is_err());
        r.failures.insert(PalsRunFailureV3::CleanupTimeout);
        assert!(r.validate_against(&l).is_ok());
    }
    #[test]
    fn unsupported_or_mismatched_options_are_failure_evidence() {
        let l = manifest().lock().unwrap();
        let mut r = receipt(&l);
        r.games[0].engines[0]
            .options
            .get_mut("Ponder")
            .unwrap()
            .advertised_supported = false;
        assert!(r.validate_against(&l).is_err());
        r.pair_eligible = false;
        r.failures.insert(PalsRunFailureV3::EngineLaunch);
        assert!(r.validate_against(&l).is_ok());
        let mut r = receipt(&l);
        r.games[0].engines[0]
            .options
            .get_mut("Ponder")
            .unwrap()
            .observed = PalsObservedV3::Observed {
            value: "true".into(),
            method: "explicit-readback".into(),
        };
        assert!(r.validate_against(&l).is_err());
    }
    #[test]
    fn physical_batch_count_cannot_exceed_declared_width() {
        let mut m = manifest();
        let PalsEngineV3::Pals(p) = &mut m.engines[0] else {
            unreachable!()
        };
        p.model.backend = PalsModelBackendV3::RustCpu;
        p.model.weights = PalsWeightIdentityV3::Untrained {
            artifact: asset("assets/init.bin"),
            initialization_seed: 1,
        };
        let l = m.lock().unwrap();
        let mut r = receipt(&l);
        let o = &mut r.games[0].engines[0];
        o.nn_inputs_completed = 2;
        o.nn_calls_completed = 1;
        o.nn_inputs_consumed = 2;
        o.physical_state = PalsPhysicalStateV3::Completed;
        assert!(r.validate_against(&l).is_err());
        r.games[0].engines[0].nn_calls_completed = 2;
        assert!(r.validate_against(&l).is_ok());
    }
    #[test]
    fn observed_resources_cannot_contradict_admission_in_an_eligible_pair() {
        let l = manifest().lock().unwrap();
        let mut r = receipt(&l);
        r.games[0].engines[0].actual_affinity = PalsObservedV3::Observed {
            value: vec![1, 3],
            method: "sched-getaffinity".into(),
        };
        assert!(r.validate_against(&l).is_err());
        r.pair_eligible = false;
        r.failures.insert(PalsRunFailureV3::ResourceAdmission);
        assert!(r.validate_against(&l).is_ok());
        r.games[0].engines[0].actual_affinity = PalsObservedV3::Observed {
            value: vec![],
            method: "sched-getaffinity".into(),
        };
        assert!(r.validate_against(&l).is_err());
        let mut r = receipt(&l);
        r.games[0].engines[0].memory_peak_bytes = PalsObservedV3::Observed {
            value: l.manifest.resources[0].memory_max_bytes + 1,
            method: "cgroup-memory-peak".into(),
        };
        assert!(r.validate_against(&l).is_err());
        r.pair_eligible = false;
        r.failures.insert(PalsRunFailureV3::ResourceAdmission);
        assert!(r.validate_against(&l).is_ok());
    }
    #[test]
    fn options_cannot_override_fixed_conditions_or_hide_casefold_duplicates() {
        let mut m = manifest();
        let PalsEngineV3::Pals(p) = &mut m.engines[0] else {
            unreachable!()
        };
        p.requested_options.insert("Ponder".into(), "true".into());
        assert!(m.validate().is_err());
        let PalsEngineV3::Pals(p) = &mut m.engines[0] else {
            unreachable!()
        };
        p.requested_options.insert("Ponder".into(), "false".into());
        p.requested_options.insert("ponder".into(), "false".into());
        assert!(m.validate().is_err());
        let mut m = manifest();
        let PalsEngineV3::Pals(p) = &mut m.engines[0] else {
            unreachable!()
        };
        p.requested_options.insert("Threads".into(), "4".into());
        assert!(m.validate().is_err());
        let mut m = manifest();
        let PalsEngineV3::Pals(p) = &mut m.engines[0] else {
            unreachable!()
        };
        p.requested_options.insert("Backend".into(), "cuda".into());
        assert!(m.validate().is_err());
    }
    #[test]
    fn changed_binary_requires_declared_component_implementation_provenance() {
        let mut m = manifest();
        m.engines[1] = pals("candidate");
        m.pilot.white_order[1] = "candidate".into();
        m.comparison = PalsComparisonV3::Runtime;
        m.declared_changes = BTreeSet::from([PalsChangeAxisV3::Runtime]);
        let PalsEngineV3::Pals(p) = &mut m.engines[1] else {
            unreachable!()
        };
        p.runtime.options.insert("queue".into(), "2".into());
        p.binary.path = "another/rovezero".into();
        assert!(m.validate().is_ok());
        let PalsEngineV3::Pals(p) = &mut m.engines[1] else {
            unreachable!()
        };
        p.binary.sha256 = "d".repeat(64);
        assert!(m.validate().is_err());
        let PalsEngineV3::Pals(p) = &mut m.engines[1] else {
            unreachable!()
        };
        p.runtime.implementation_sha256 = "e".repeat(64);
        assert!(m.validate().is_ok());
    }
}
