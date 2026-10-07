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
    IncompleteGame
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
    pub cpu: PalsOwnCpuV3,
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
fn endpoint(e: &PalsEngineV3, r: &PalsResourcePolicyV3) -> Result<(), ManifestError> {
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
            component(&p.search)?;
            component(&p.runtime)?;
            let b = &p.pools;
            require(
                p.search.semantic_id == "pals"
                    && [
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
        if a.cpu != b.cpu {
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
            endpoint(e, r)?;
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

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsEndpointReceiptV3 {
    pub endpoint_id: String,
    pub uci_ready_observed: bool,
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
    require(
        o.endpoint_id == e.id() && o.options.keys().eq(e.requested_options().keys()),
        "receipt endpoint/options differ from launch",
    )?;
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            uci_ready_observed: true,
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
