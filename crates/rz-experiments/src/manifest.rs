use crate::{ManifestError, digest, parse};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;
pub const CANONICALIZATION: &str = "rz-e01-json-v1";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunManifest {
    pub schema_version: u32,
    pub experiment_id: String,
    pub run_id: String,
    pub question: String,
    pub created_utc: String,
    pub parent_run: Option<String>,
    pub candidate_ids: Vec<String>,
    pub purpose: RunPurpose,
    pub comparison: Comparison,
    pub change: Change,
    pub research_path: ResearchPath,
    pub contract_revision: Option<String>,
    pub allowed_option_change: Option<String>,
    pub engines: Vec<EngineSpec>,
    pub hardware: HardwareSpec,
    pub input: InputSpec,
    pub protocol: ProtocolSpec,
    pub clock: ClockSpec,
    pub lifecycle: LifecycleSpec,
    pub plan: PlanSpec,
    pub statistics: StatisticsSpec,
    pub budget: RunBudget,
}

macro_rules! choices {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
    };
}
choices!(RunPurpose {
    Fixture,
    Development,
    Formal
});
choices!(Comparison {
    Fixture,
    InternalSearch,
    InternalWeights,
    Runtime,
    ExternalLc0
});
choices!(Change {
    MeaningPreserving,
    Model,
    Search
});
choices!(ResearchPath {
    Gpu,
    Hybrid,
    Cpu,
    Offline
});
choices!(EngineKind {
    Fixture,
    RoveZero,
    Lc0
});
choices!(Device { Cpu, Gpu });
choices!(HistoryCompleteness {
    Complete,
    UnknownPrefix
});
choices!(InitialPosition { Startpos, Fen });
choices!(InputSplit {
    Fixture,
    Tuning,
    Holdout
});
choices!(Residency {
    CpuOnly,
    Concurrent,
    Swap
});
choices!(ClaimPolicy {
    ExplicitClaim,
    AutomaticAcceptance
});
choices!(OutcomePolicy {
    Loss,
    Incomplete,
    ContractInvalid
});
choices!(SamplePlan { FixedSample });
choices!(EloScale { Logistic });
choices!(ClusterUnit { Opening });
choices!(CiMethod {
    None,
    OpeningClusterBootstrap
});
choices!(PairOrder { Alternating });
choices!(RerunPolicy { WholePair });

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub source: String,
    pub license: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ToolIdentity {
    pub version: String,
    pub source_url: String,
    pub source_commit: String,
    pub binary: ArtifactRef,
    pub dirty: bool,
    pub dirty_patch: Option<ArtifactRef>,
    pub build_mode: String,
    pub compiler: String,
    pub target: String,
    pub isa: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EngineSpec {
    pub id: String,
    pub kind: EngineKind,
    pub tool: ToolIdentity,
    pub model_id: String,
    pub encoding_id: String,
    pub evaluator_id: String,
    pub search_id: String,
    pub runtime_id: String,
    pub backend: String,
    pub precision: String,
    pub device: Device,
    pub weight: Option<ArtifactRef>,
    pub requested_options: BTreeMap<String, String>,
    pub observation: Option<EngineObservation>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EngineObservation {
    pub applied_options: BTreeMap<String, String>,
    pub backend: String,
    pub precision: String,
    pub device: Device,
    pub device_id: String,
    pub evidence: ArtifactRef,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HardwareSpec {
    pub cpu_model: String,
    pub physical_cores: u32,
    pub threads: u32,
    pub affinity: Vec<u32>,
    pub ram_limit_bytes: u64,
    pub gpu: Option<GpuSpec>,
    pub operating_system: String,
    pub runtime_version: String,
    pub assignment_policy: String,
    pub residency: Residency,
    pub helper_core_limit: u32,
    pub helper_ram_limit_bytes: u64,
    pub tablebase_enabled: bool,
    pub tablebase_io_limit_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GpuSpec {
    pub model: String,
    pub count: u32,
    pub vram_limit_bytes: u64,
    pub driver: String,
    pub runtime: String,
    pub power_policy: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    pub opening_artifact: ArtifactRef,
    pub selection_policy: String,
    pub seed: u64,
    pub split: InputSplit,
    pub history_fill_policy: String,
    pub repetition_policy: String,
    pub openings: Vec<OpeningSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpeningSpec {
    pub id: String,
    pub initial: InitialPosition,
    pub fen: Option<String>,
    pub moves: Vec<String>,
    pub history: HistoryCompleteness,
    pub history_origin: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProtocolSpec {
    pub runner: ToolIdentity,
    pub uci_adapter_version: String,
    pub rules_reference_id: String,
    pub rules_version: String,
    pub draw_profile: String,
    pub claim_policy: ClaimPolicy,
    pub terminal_priority: String,
    pub dead_position_scope: String,
    pub adjudication_enabled: bool,
    pub tablebase_policy: String,
    pub max_plies: u32,
    pub max_plies_outcome: OutcomePolicy,
    pub engine_failure: OutcomePolicy,
    pub simultaneous_failure: OutcomePolicy,
    pub cancellation_outcome: OutcomePolicy,
    pub infrastructure_invalid_policy: RerunPolicy,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "track", rename_all = "snake_case", deny_unknown_fields)]
pub enum ClockSpec {
    T1 { base_ms: u64, increment_ms: u64 },
    T2 { movetime_ms: u64 },
    T3 { unique_evaluation_budget: u64 },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LifecycleSpec {
    pub clock_source: String,
    pub clock_boundary: String,
    pub uci_overhead_ms: u64,
    pub timeout_grace_ms: u64,
    pub handshake_timeout_ms: u64,
    pub request_deadline_policy: String,
    pub loading_policy: String,
    pub compile_policy: String,
    pub warmup_policy: String,
    pub warmup_input: Option<ArtifactRef>,
    pub warmup_max_ms: u64,
    pub warmup_cache_reset: bool,
    pub newgame_reset_policy: String,
    pub drain_timeout_ms: u64,
    pub shutdown_timeout_ms: u64,
    pub ponder: bool,
    pub opponent_turn_compute: bool,
    pub online_weights: bool,
    pub cross_game_results: bool,
    pub correction_policy: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanSpec {
    pub seed: u64,
    pub engine_seeds: BTreeMap<String, u64>,
    pub pair_order: PairOrder,
    pub hardware_order: String,
    pub pairs: u64,
    pub max_retries_per_pair: u32,
    pub rerun_policy: RerunPolicy,
    pub incomplete_pair_policy: String,
    pub stop_policy: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StatisticsSpec {
    pub implementation: String,
    pub score_viewpoint: String,
    pub sample_plan: SamplePlan,
    pub elo_scale: EloScale,
    pub cluster_unit: ClusterUnit,
    pub ci_method: CiMethod,
    pub confidence_percent: u8,
    pub bootstrap_replicates: u32,
    pub seed: u64,
    pub minimum_clusters: u32,
    pub promotion_policy: String,
    pub budget_end_policy: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunBudget {
    pub max_pairs: u64,
    pub max_games: u64,
    pub max_wall_ms: u64,
    pub workers: u32,
    pub max_child_processes: u32,
    pub max_output_bytes: u64,
    pub max_artifact_bytes: u64,
}

/// An immutable, structurally valid input lock, not an execution permit.
/// E02 must still perform launch receipts, legality and fairness verification.
#[derive(Clone, Debug)]
pub struct LockedManifest {
    input: RunManifest,
    sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LockEnvelope {
    lock_version: u32,
    canonicalization: String,
    execution_ready: bool,
    input_sha256: String,
    input: RunManifest,
}

impl RunManifest {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        parse(input)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        crate::validation::validate(self)
    }

    pub fn lock(self) -> Result<LockedManifest, ManifestError> {
        self.validate()?;
        let sha256 = digest(&self.canonical_bytes()?);
        Ok(LockedManifest {
            input: self,
            sha256,
        })
    }

    // V1: recursively sorted object keys; UTF-8 compact serde_json;
    // integer-only numbers; explicit null/empty values; no envelope.
    fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        let mut value =
            serde_json::to_value(self).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        value.sort_all_objects();
        serde_json::to_vec(&value).map_err(|e| ManifestError::Integrity(e.to_string()))
    }

    pub(crate) fn artifacts(&self) -> Vec<&ArtifactRef> {
        let mut refs = vec![&self.input.opening_artifact, &self.protocol.runner.binary];
        if let Some(p) = &self.protocol.runner.dirty_patch {
            refs.push(p);
        }
        if let Some(p) = &self.lifecycle.warmup_input {
            refs.push(p);
        }
        for engine in &self.engines {
            refs.push(&engine.tool.binary);
            if let Some(p) = &engine.tool.dirty_patch {
                refs.push(p);
            }
            if let Some(p) = &engine.weight {
                refs.push(p);
            }
            if let Some(o) = &engine.observation {
                refs.push(&o.evidence);
            }
        }
        refs
    }
}

impl LockedManifest {
    pub fn input(&self) -> &RunManifest {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn to_json(&self) -> Result<String, ManifestError> {
        let envelope = LockEnvelope {
            lock_version: 1,
            canonicalization: CANONICALIZATION.into(),
            execution_ready: false,
            input_sha256: self.sha256.clone(),
            input: self.input.clone(),
        };
        serde_json::to_string_pretty(&envelope).map_err(|e| ManifestError::Integrity(e.to_string()))
    }

    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let envelope: LockEnvelope = parse(input)?;
        if envelope.lock_version != 1 || envelope.canonicalization != CANONICALIZATION {
            return Err(ManifestError::Integrity(
                "unsupported lock/canonicalization version".into(),
            ));
        }
        if envelope.execution_ready {
            return Err(ManifestError::Integrity(
                "E01 input lock cannot authorize execution".into(),
            ));
        }
        let locked = envelope.input.lock()?;
        if locked.sha256 != envelope.input_sha256 {
            return Err(ManifestError::Integrity("input SHA-256 mismatch".into()));
        }
        Ok(locked)
    }

    pub fn verify_artifacts(&self, root: &Path, max_total_bytes: u64) -> Result<(), ManifestError> {
        crate::artifact::verify(&self.input, root, max_total_bytes)
    }
}
