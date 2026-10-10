//! PALS V4 execution declarations and independently observed receipts.
//!
//! V4 has its own domain and contract. Shared V3 resource/physical validators
//! keep their original meaning; no V1/V2/V3 asset is converted or reinterpreted.
//! A valid declaration or matching digest is never proof of execution/readiness.
use crate::pals_manifest as shared;
use crate::{
    ArtifactRef, ManifestError, PalsChangeAxisV3, PalsComparisonV3, PalsComponentV3,
    PalsCpuEndpointV3, PalsCpuRSelectionV3, PalsEndpointReceiptV3, PalsEndpointV3, PalsEngineV3,
    PalsModelBackendV3, PalsModelIdentityV3, PalsObservedV3, PalsPhysicalStateV3, PalsPilotV3,
    PalsReferenceEndpointV3, PalsResourcePolicyV3, PalsResultV3, PalsRunFailureV3,
    PalsTerminationV3, PalsWeightIdentityV3, decode_json, digest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod followup;
pub use followup::*;

pub const PALS_MANIFEST_V4_DOMAIN: &str = "rz-pals-execution-v4/1";
pub const PALS_RECEIPT_V4_DOMAIN: &str = "rz-pals-receipt-v4/1";
pub const PALS_V4_CONTRACT_REVISION: &str = "pals/0.2";
pub const PALS_V4_SEARCH_SEMANTICS: &str = "rovezero.pals-full-line-v2/0.2";
pub const PALS_V4_RECHECK_SEARCH_SEMANTICS: &str = "post-repair-frozen-wdl-v2";
pub const PALS_V4_ITERATIVE_SEARCH_SEMANTICS: &str = "post-repair-frozen-wdl-queue-v2";
pub const PALS_V4_OWN_RAW_RESOLVER_SEMANTICS: &str = "pals-cpu-raw-restricted/0.1";
// Numeric WDL resolution retains its existing semantics; V4 separately pins
// explicit resolver selection and the new common-basis recheck/search policy.
pub const PALS_V4_MODEL_WDL_RESOLVER_SEMANTICS: &str = "pals-model-wdl-restricted/0.1";
pub const PALS_V4_OWN_RAW_RESOLVER_CONDITIONS: &str = "leaf:accepted-registered-cpu-raw-side-to-move-including-frontier-and-partial;propagation:examined-children-max-negation;unknown:None-not-zero;calibration:none;pc-score-average:none;root:rules-certified-win-first,equal-value-terminal-first,Rules-order-ties;nonterminal-scope:restricted-estimate-not-game-bound";
pub const PALS_V4_MODEL_WDL_RESOLVER_CONDITIONS: &str = "root:exact-Rules-win-priority;child:accepted-registered-model-W-L-max;perspective:flip-W-L-preserve-D;unknown:None-not-zero;ties:Rules-order;namespace:model-estimate-only;owned-raw:separate;cp-conversion:none;averaging:none;all-defenses-proof:none";
/// Independent closed pins, compared by arena with immutable engine getters.
pub const PALS_V4_BASE_CONDITIONS: &str = "explicit-independent-checker-resolver;restricted-candidate-refinement;Rules-own-legality-terminal;typed-model-profile-encoding;shared-role-cpu-store-deadline-cancel-limits;optional-repair-queue-paused-stack-CUDA-Warm-disabled-unless-declared;no-global-five-percent-gate";
pub const PALS_V4_RECHECK_CONDITIONS: &str = "accepted-repair;unchanged-first-move;C-at-each-actual-counter-prefix;equal-line-length;Fresh-same-frozen-model-precision-context-revision-WDL-or-both-Rules-terminal;mixed-scope-unresolved;strict-value-decrease;conditional-refutation-only;shared-role-cpu-store-deadline-cancel-limits";
pub const PALS_V4_ITERATIVE_CONDITIONS: &str = "accepted-repair;unchanged-first-move;C-at-each-actual-counter-prefix;equal-line-length;Fresh-same-frozen-model-precision-context-revision-WDL-or-both-Rules-terminal;mixed-scope-unresolved;strict-value-decrease;conditional-refutation-only;shared-role-cpu-store-deadline-cancel-limits;pending-at-most-64;Repair-at-most-3-per-first-move;deduplicate-question-revision;stop-without-new-evidence;retain-supersedes";
pub const PALS_V4_MODEL_SCHEMA: &str = "rovezero.pals-model.v2";
pub const PALS_V4_MODEL_SEMANTICS: &str = "rovezero.pals-model-semantics.v2";
pub const PALS_V4_ENCODING_SCHEMA: &str = "rovezero.pals-board-records.v2";
pub const PALS_V4_NONZERO_CHECKPOINT_DOMAIN: &str = "rz-pals-python-nonzero-checkpoint/1";
pub const PALS_V4_CUDA_WARM_CAPABILITY: &str = "rovezero.pals-private-cuda-warm.v2";
pub const PALS_V4_REPAIR_MAX: u32 = 3;
pub const PALS_V4_PENDING_MAX: u32 = 64;
pub const PALS_V4_ARCHIVE_GAME_MAX_BYTES: u64 = 256 << 20;
pub const PALS_V4_ARCHIVE_GLOBAL_MAX_BYTES: u64 = 4 << 30;
pub const PALS_V4_PAUSED_STACK_MAX_BYTES: u64 = 8 << 20;
pub const PALS_V4_REPAIR_TRACE_MAX: usize = 4096;

macro_rules! choices {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
    };
}

choices!(PalsRunPurposeV4 {
    ArenaPilot,
    Diagnostic
});
choices!(PalsArtifactPurposeV4 {
    ArenaCandidate,
    Diagnostic
});
choices!(PalsTargetCoverageV4 {
    RealTargets,
    DiagnosticFixture
});
choices!(PalsModelProfileV4 {
    LegacySummaryV1,
    FullLineV2,
    InteractionHeadV2,
    FullLineInteractionV2
});
choices!(PalsResolverPolicyV4 { OwnRaw, ModelWdl });
impl PalsResolverPolicyV4 {
    pub fn expected_semantics_sha256(self) -> String {
        digest(
            match self {
                Self::OwnRaw => PALS_V4_OWN_RAW_RESOLVER_CONDITIONS,
                Self::ModelWdl => PALS_V4_MODEL_WDL_RESOLVER_CONDITIONS,
            }
            .as_bytes(),
        )
    }
}
choices!(PalsCheckerPolicyV4 { Own, ExternalUci });
choices!(PalsRecheckPolicyV4 {
    Disabled,
    FrozenWdl
});
choices!(PalsPerspectiveV4 { White, Black });
choices!(PalsRecheckResolutionV4 {
    BeforePreferred,
    AfterPreferred,
    Equal,
    Unresolved
});
choices!(PalsArchiveLifecycleV4 {
    Open,
    Closed,
    Failed
});
choices!(PalsPolicyFailureV4 {
    ArtifactCheck,
    ModelIdentity,
    PolicyIdentity,
    ArchiveQuota,
    ArchiveIo,
    ArchivePinSaturated,
    PausedStack,
    CudaCapability,
    CudaCompletionUnknown,
    ProcessCleanup,
    Training,
    OutputQuota
});

/// These are source provenance declarations. Receipt checks must bind them to
/// verified bytes. Nonzero diagnostic training is never an arena promotion.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsTrainingProvenanceV4 {
    Untrained,
    Nonzero {
        checkpoint_domain: String,
        training_run_id: String,
        completed_updates: u32,
        target_coverage: PalsTargetCoverageV4,
        target_coverage_sha256: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsModelProvenanceV4 {
    pub purpose: PalsArtifactPurposeV4,
    pub training: PalsTrainingProvenanceV4,
}

/// The model/encoding identity and checkpoint identity are separate pins.
/// `base.model` retains backend, precision, frozen epoch and P/C export identity.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsModelSpecificationV4 {
    pub profile: PalsModelProfileV4,
    pub model_schema: String,
    pub model_semantics: String,
    pub encoding_schema: String,
    pub model_identity: String,
    pub model_semantics_sha256: String,
    pub encoding_sha256: String,
    pub max_records: u32,
    pub max_line_plies: u32,
    pub provenance: PalsModelProvenanceV4,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsIterativeRepairLimitsV4 {
    pub max_repairs_per_first_move: u32,
    pub max_pending_questions: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "policy",
    content = "limits",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsIterativeRepairPolicyV4 {
    Disabled,
    Bounded(PalsIterativeRepairLimitsV4),
}

/// Bounded cold storage always pins root/frontier/in-flight/paused/evidence;
/// migration triggers at 80% hot admission or allocation failure, retries once,
/// and releases RAM only after archive commit and independent integrity check.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsColdArchiveLimitsV4 {
    pub game_bytes_max: u64,
    pub global_bytes_max: u64,
    pub index_entries_max: u32,
    pub index_bytes_max: u64,
    pub load_bytes_max: u64,
    pub load_deadline_max_ms: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "policy",
    content = "limits",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsColdArchivePolicyV4 {
    Disabled,
    Bounded(PalsColdArchiveLimitsV4),
}

/// Opaque CPU token: exact unfinished frames/state/order/TT are owned by CPU.
/// The manifest does not serialize a frame or confer cross-context reuse rights.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPausedStackLimitsV4 {
    pub tokens_max: u32,
    pub bytes_max: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "policy",
    content = "limits",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsPausedStackPolicyV4 {
    Disabled,
    Bounded(PalsPausedStackLimitsV4),
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaWarmLimitsV4 {
    pub capability: String,
    pub max_leases: u32,
    pub device_bytes_max: u64,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "policy",
    content = "limits",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsCudaWarmPolicyV4 {
    #[default]
    Disabled,
    ApproxWarm(PalsCudaWarmLimitsV4),
}

/// Every new constructor must choose resolver/checker explicitly. CUDA Warm is
/// separately optional; omitting it selects Disabled, never a host Warm alias.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPoliciesV4 {
    pub resolver: PalsResolverPolicyV4,
    /// Actual selected resolver implementation, independently checked at launch.
    pub resolver_identity: PalsComponentV3,
    /// Actual resolver semantic pin, distinct from registered source identity.
    pub resolver_semantics_sha256: String,
    pub checker: PalsCheckerPolicyV4,
    pub recheck: PalsRecheckPolicyV4,
    pub iterative_repair: PalsIterativeRepairPolicyV4,
    pub cold_archive: PalsColdArchivePolicyV4,
    pub paused_stack: PalsPausedStackPolicyV4,
    #[serde(default)]
    pub cuda_warm: PalsCudaWarmPolicyV4,
}

/// Actual immutable runtime getters must be compared with this declaration.
/// Shape-valid pins alone do not establish compatibility with an engine build.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPolicyIdentityV4 {
    pub version: String,
    pub policy: String,
    pub search_identity: String,
    pub conditions_sha256: [u8; 32],
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsEndpointV4 {
    /// Shared fields have the same meaning as V3, but new search/model labels
    /// remain the actual V4 labels and are never coerced to pass V3 validation.
    pub base: PalsEndpointV3,
    pub model_v2: PalsModelSpecificationV4,
    pub policies: PalsPoliciesV4,
    pub policy_identity: PalsPolicyIdentityV4,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "endpoint",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsEngineV4 {
    Pals(Box<PalsEndpointV4>),
    OwnCpu(Box<PalsCpuEndpointV3>),
    ReferenceUci(Box<PalsReferenceEndpointV3>),
}
impl PalsEngineV4 {
    pub fn id(&self) -> &str {
        match self {
            Self::Pals(e) => &e.base.id,
            Self::OwnCpu(e) => &e.id,
            Self::ReferenceUci(e) => &e.id,
        }
    }
    /// Lossless shared physical/resource view; no labels or identities change.
    fn shared_engine(&self) -> PalsEngineV3 {
        match self {
            Self::Pals(e) => PalsEngineV3::Pals(Box::new(e.base.clone())),
            Self::OwnCpu(e) => PalsEngineV3::OwnCpu(e.clone()),
            Self::ReferenceUci(e) => PalsEngineV3::ReferenceUci(e.clone()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRunManifestV4 {
    pub schema_version: u32,
    pub run_id: String,
    pub pair_id: String,
    pub contract_revision: String,
    pub rules_profile: String,
    pub purpose: PalsRunPurposeV4,
    pub comparison: PalsComparisonV3,
    pub declared_changes: BTreeSet<PalsChangeAxisV3>,
    pub training_executed: bool,
    pub engines: [PalsEngineV4; 2],
    pub resources: [PalsResourcePolicyV3; 2],
    pub pilot: PalsPilotV3,
    pub output_bytes_max: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsInputLockV4 {
    pub domain: String,
    pub manifest: PalsRunManifestV4,
    pub canonical_sha256: String,
}

fn require(ok: bool, message: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(format!("PALS V4: {message}")))
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
fn encoded<T: Serialize>(value: &T, pretty: bool) -> Result<Vec<u8>, ManifestError> {
    let bytes = if pretty {
        serde_json::to_vec_pretty(value)
    } else {
        serde_json::to_vec(value)
    }
    .map_err(|e| ManifestError::Integrity(e.to_string()))?;
    require(
        bytes.len() <= crate::MAX_MANIFEST_BYTES,
        "serialized V4 metadata exceeds bounded 4 MiB decoder capacity",
    )?;
    Ok(bytes)
}

impl PalsModelSpecificationV4 {
    pub fn validate_against(&self, e: &PalsEndpointV3) -> Result<(), ManifestError> {
        shared::model(&e.model)?;
        let legacy = self.profile == PalsModelProfileV4::LegacySummaryV1;
        let full_line = matches!(
            self.profile,
            PalsModelProfileV4::FullLineV2 | PalsModelProfileV4::FullLineInteractionV2
        );
        require(
            (self.model_schema == PALS_V4_MODEL_SCHEMA
                || (full_line
                    && e.model.backend == PalsModelBackendV3::OrtCuda
                    && self.model_schema == "rovezero.pals-private-cuda-warm.v2")
                || (legacy && self.model_schema == "rovezero.pals-model.v1"))
                && self.model_semantics
                    == if legacy {
                        "rovezero.pals-model.v1"
                    } else {
                        PALS_V4_MODEL_SEMANTICS
                    }
                && self.encoding_schema
                    == if legacy {
                        "rovezero.pals-board-records.v1"
                    } else {
                        PALS_V4_ENCODING_SCHEMA
                    }
                && e.model.input_schema == self.encoding_schema
                && text(&self.model_identity)
                && sha(&self.model_semantics_sha256, 64)
                && sha(&self.encoding_sha256, 64)
                && self.max_records == 128
                && self.max_line_plies == if full_line { 256 } else { 0 },
            "unsupported model profile/schema/encoding/line bound combination",
        )?;
        match (&self.provenance.training, &e.model.weights) {
            (
                PalsTrainingProvenanceV4::Untrained,
                PalsWeightIdentityV3::DeterministicMock { .. }
                | PalsWeightIdentityV3::Untrained { .. },
            ) => Ok(()),
            (
                PalsTrainingProvenanceV4::Nonzero {
                    checkpoint_domain,
                    training_run_id,
                    completed_updates,
                    target_coverage_sha256,
                    ..
                },
                PalsWeightIdentityV3::Trained {
                    training_run_id: weight_run_id,
                    ..
                },
            ) => require(
                checkpoint_domain == PALS_V4_NONZERO_CHECKPOINT_DOMAIN
                    && text(training_run_id)
                    && training_run_id == weight_run_id
                    && (1..=24).contains(completed_updates)
                    && sha(target_coverage_sha256, 64)
                    && self.provenance.purpose == PalsArtifactPurposeV4::Diagnostic,
                "nonzero checkpoint needs explicit diagnostic domain/run/updates/target provenance",
            ),
            _ => require(false, "checkpoint weights and training provenance differ"),
        }
    }
    pub fn arena_candidate(&self) -> bool {
        self.provenance.purpose == PalsArtifactPurposeV4::ArenaCandidate
            && matches!(
                self.provenance.training,
                PalsTrainingProvenanceV4::Untrained
            )
    }
}

impl PalsPoliciesV4 {
    pub fn validate_against(
        &self,
        e: &PalsEndpointV3,
        r: &PalsResourcePolicyV3,
        pilot: &PalsPilotV3,
    ) -> Result<(), ManifestError> {
        shared::component(&self.resolver_identity)?;
        require(
            self.resolver_identity.semantic_id
                == match self.resolver {
                    PalsResolverPolicyV4::OwnRaw => PALS_V4_OWN_RAW_RESOLVER_SEMANTICS,
                    PalsResolverPolicyV4::ModelWdl => PALS_V4_MODEL_WDL_RESOLVER_SEMANTICS,
                }
                && self.resolver_identity.options.is_empty()
                && self.resolver_semantics_sha256 == self.resolver.expected_semantics_sha256(),
            "selected resolver identity differs from the explicit V4 resolver policy",
        )?;
        require(
            (self.checker == PalsCheckerPolicyV4::Own) == e.cpu_r.is_own()
                && (self.resolver != PalsResolverPolicyV4::OwnRaw
                    || self.checker == PalsCheckerPolicyV4::Own),
            "own raw requires own checker; explicit checker differs from CPU_R declaration",
        )?;
        if let PalsCpuRSelectionV3::ExternalUci(external) = &e.cpu_r {
            shared::external_cpu_r(external, r, pilot)?;
            require(
                self.resolver == PalsResolverPolicyV4::ModelWdl
                    && external.resolver.version == self.resolver_identity.semantic_id
                    && external.resolver.semantics_sha256 == self.resolver_semantics_sha256,
                "external checker requires model WDL resolver",
            )?;
        }
        if let PalsIterativeRepairPolicyV4::Bounded(l) = &self.iterative_repair {
            require(
                (1..=PALS_V4_REPAIR_MAX).contains(&l.max_repairs_per_first_move)
                    && (1..=PALS_V4_PENDING_MAX).contains(&l.max_pending_questions)
                    && self.recheck == PalsRecheckPolicyV4::FrozenWdl,
                "iterative Repair requires frozen-WDL recheck and bounded 3/64 queue",
            )?;
        }
        if let PalsColdArchivePolicyV4::Bounded(l) = &self.cold_archive {
            require(
                (1..=PALS_V4_ARCHIVE_GAME_MAX_BYTES).contains(&l.game_bytes_max)
                    && (1..=PALS_V4_ARCHIVE_GLOBAL_MAX_BYTES).contains(&l.global_bytes_max)
                    && l.game_bytes_max <= l.global_bytes_max
                    && (1..=1_048_576).contains(&l.index_entries_max)
                    && l.index_bytes_max > 0
                    && l.index_bytes_max <= e.pools.host_bytes
                    && l.load_bytes_max > 0
                    && l.load_bytes_max <= l.game_bytes_max
                    && l.load_bytes_max <= e.pools.host_bytes
                    && (1..=pilot.wall_time_max_ms).contains(&l.load_deadline_max_ms),
                "cold archive/index/pin-load limits exceed bounded game/global/host/deadline ceilings",
            )?;
        }
        if let PalsPausedStackPolicyV4::Bounded(l) = &self.paused_stack {
            require(
                self.checker == PalsCheckerPolicyV4::Own
                    && l.tokens_max == 1
                    && (1..=PALS_V4_PAUSED_STACK_MAX_BYTES).contains(&l.bytes_max)
                    && l.bytes_max <= e.pools.host_bytes,
                "PausedStack requires own CPU and one token within 8 MiB/host ceiling",
            )?;
        }
        if let PalsCudaWarmPolicyV4::ApproxWarm(l) = &self.cuda_warm {
            require(
                l.capability == PALS_V4_CUDA_WARM_CAPABILITY
                    && e.model.backend == PalsModelBackendV3::OrtCuda
                    && r.requested_gpu.is_some()
                    && (1..=64).contains(&l.max_leases)
                    && l.device_bytes_max > 0
                    && l.device_bytes_max <= e.pools.device_bytes
                    && l.device_bytes_max <= r.device_allocation_max_bytes,
                "CUDA ApproxWarm needs explicit CUDA device capability and physical lease byte ceiling",
            )?;
        }
        Ok(())
    }
}

impl PalsPolicyIdentityV4 {
    /// Produce the declaration's expected registration. The caller must still
    /// compare actual runtime getters independently; this is not an observation.
    pub fn expected_for_policies(policies: &PalsPoliciesV4) -> Result<Self, ManifestError> {
        let (version, policy, search, conditions) =
            match (&policies.recheck, &policies.iterative_repair) {
                (PalsRecheckPolicyV4::Disabled, PalsIterativeRepairPolicyV4::Disabled) => (
                    PALS_V4_SEARCH_SEMANTICS,
                    "full_line_v2",
                    PALS_V4_SEARCH_SEMANTICS,
                    PALS_V4_BASE_CONDITIONS,
                ),
                (PalsRecheckPolicyV4::FrozenWdl, PalsIterativeRepairPolicyV4::Disabled) => (
                    PALS_V4_RECHECK_SEARCH_SEMANTICS,
                    "frozen_model_wdl_v2",
                    PALS_V4_RECHECK_SEARCH_SEMANTICS,
                    PALS_V4_RECHECK_CONDITIONS,
                ),
                (PalsRecheckPolicyV4::FrozenWdl, PalsIterativeRepairPolicyV4::Bounded(_)) => (
                    PALS_V4_ITERATIVE_SEARCH_SEMANTICS,
                    "iterative_frozen_model_wdl_v2",
                    PALS_V4_ITERATIVE_SEARCH_SEMANTICS,
                    PALS_V4_ITERATIVE_CONDITIONS,
                ),
                _ => {
                    return Err(ManifestError::Integrity(
                        "PALS V4: unsupported recheck/queue identity combination".into(),
                    ));
                }
            };
        use sha2::Digest as _;
        Ok(Self {
            version: version.into(),
            policy: policy.into(),
            search_identity: search.into(),
            conditions_sha256: sha2::Sha256::digest(conditions.as_bytes()).into(),
        })
    }
    pub fn validate_against(&self, e: &PalsEndpointV4) -> Result<(), ManifestError> {
        let expected = Self::expected_for_policies(&e.policies)?;
        require(
            *self == expected
                && e.base.search.semantic_id == expected.search_identity
                && e.base.search.options.is_empty(),
            "V4 search/condition registration differs from the selected closed policy",
        )
    }
}

impl PalsEndpointV4 {
    pub fn validate_against(
        &self,
        r: &PalsResourcePolicyV3,
        pilot: &PalsPilotV3,
    ) -> Result<(), ManifestError> {
        let e = &self.base;
        shared::resources(r)?;
        self.model_v2.validate_against(e)?;
        shared::cpu(&e.cpu)?;
        shared::component(&e.search)?;
        shared::component(&e.runtime)?;
        self.policies.validate_against(e, r, pilot)?;
        require(
            (self.model_v2.model_schema == "rovezero.pals-private-cuda-warm.v2")
                == matches!(self.policies.cuda_warm, PalsCudaWarmPolicyV4::ApproxWarm(_)),
            "CUDA Warm export schema requires its separately selected actual device owner",
        )?;
        self.policy_identity.validate_against(self)?;
        e.binary.validate()?;
        require(
            text(&e.id) && sha(&e.source_commit, 40) && shared::options(&e.requested_options),
            "invalid endpoint/binary/source/options identity",
        )?;
        for (name, value) in &e.requested_options {
            let key = name.to_ascii_lowercase();
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
                        | "resolverpolicy"
                        | "checker"
                        | "recheck"
                        | "iterativerepair"
                        | "coldarchive"
                        | "pausedstack"
                        | "cudawarm"
                        | "palsprofile"
                ),
                "typed V4 model/search/runtime selection cannot be overridden by UCI options",
            )?;
            match key.as_str() {
                "ponder" | "uci_chess960" => require(
                    value.eq_ignore_ascii_case("false"),
                    "pilot forbids ponder and Chess960",
                )?,
                "threads" => require(
                    value.parse::<u32>().ok() == Some(r.cpu_threads),
                    "Threads differs from resource declaration",
                )?,
                "hash" => require(
                    value
                        .parse::<u64>()
                        .ok()
                        .and_then(|n| n.checked_mul(1 << 20))
                        .is_some_and(|n| n > 0 && n <= r.memory_max_bytes),
                    "Hash exceeds memory declaration",
                )?,
                _ => {}
            }
        }
        let b = &e.pools;
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
                && e.cpu.max_tt_bytes <= b.host_bytes
                && (e.model.backend != PalsModelBackendV3::OrtCuda
                    || (r.requested_gpu.is_some() && b.device_bytes > 0)),
            "PALS V4 pools/CPU/device exceed shared physical resource ceilings",
        )
    }
}

impl PalsRunManifestV4 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let m: Self = decode_json(input)?;
        m.validate()?;
        Ok(m)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.schema_version == 4 && self.contract_revision == PALS_V4_CONTRACT_REVISION,
            "schema/revision mismatch; no automatic legacy conversion",
        )?;
        require(
            [&self.run_id, &self.pair_id, &self.rules_profile]
                .iter()
                .all(|s| text(s)),
            "invalid run/pair/Rules identity",
        )?;
        require(
            self.engines[0].id() != self.engines[1].id()
                && self
                    .engines
                    .iter()
                    .any(|e| matches!(e, PalsEngineV4::Pals(_)))
                && (!self.training_executed || self.purpose == PalsRunPurposeV4::Diagnostic)
                && (1..=2 << 30).contains(&self.output_bytes_max),
            "distinct PALS endpoints, bounded output and diagnostic-only actual training required",
        )?;
        for (e, r) in self.engines.iter().zip(&self.resources) {
            shared::resources(r)?;
            match e {
                PalsEngineV4::Pals(p) => {
                    p.validate_against(r, &self.pilot)?;
                    require(
                        self.purpose != PalsRunPurposeV4::ArenaPilot
                            || p.model_v2.arena_candidate(),
                        "diagnostic/nonzero checkpoint cannot be used by arena pilot",
                    )?;
                }
                _ => shared::endpoint(&e.shared_engine(), r, &self.pilot)?,
            }
        }
        let actual = self.actual_change_axes();
        require(
            actual == self.declared_changes,
            "declared axes differ from actual V4 model/policy/configuration changes",
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
                "controlled V4 comparison requires one axis and equal resources",
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
                && (1..=900_000).contains(&p.wall_time_max_ms)
                && (1..=30_000).contains(&p.cleanup_max_ms)
                && (1..=p.wall_time_max_ms).contains(&p.handshake_max_ms)
                && p.concurrent_games == 1
                && p.restart_processes_each_game
                && !p.ponder
                && !p.score_adjudication
                && !p.elo_claim,
            "V4 pilot requires exchanged 120+1 games, bounded 15min+30s and no ponder/adjudication/Elo",
        )?;
        self.unique_input_bytes()?;
        Ok(())
    }
    pub fn actual_change_axes(&self) -> BTreeSet<PalsChangeAxisV3> {
        let mut result = shared::changes(
            &self.engines[0].shared_engine(),
            &self.engines[1].shared_engine(),
        );
        if let (PalsEngineV4::Pals(a), PalsEngineV4::Pals(b)) = (&self.engines[0], &self.engines[1])
        {
            if a.model_v2 != b.model_v2 {
                result.insert(PalsChangeAxisV3::Model);
            }
            if a.policies.resolver != b.policies.resolver
                || a.policies.resolver_identity != b.policies.resolver_identity
                || a.policies.resolver_semantics_sha256 != b.policies.resolver_semantics_sha256
                || a.policies.recheck != b.policies.recheck
                || a.policies.iterative_repair != b.policies.iterative_repair
                || a.policy_identity != b.policy_identity
            {
                result.insert(PalsChangeAxisV3::Search);
            }
            if a.policies.checker != b.policies.checker
                || a.policies.paused_stack != b.policies.paused_stack
            {
                result.insert(PalsChangeAxisV3::CpuCore);
            }
            if a.policies.cold_archive != b.policies.cold_archive
                || a.policies.cuda_warm != b.policies.cuda_warm
            {
                result.insert(PalsChangeAxisV3::Runtime);
            }
        }
        result
    }
    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut result = Vec::new();
        for e in &self.engines {
            match e {
                PalsEngineV4::Pals(p) => {
                    result.push(&p.base.binary);
                    if let PalsWeightIdentityV3::Untrained { artifact, .. }
                    | PalsWeightIdentityV3::Trained { artifact, .. } = &p.base.model.weights
                    {
                        result.push(artifact);
                    }
                    if let PalsCpuRSelectionV3::ExternalUci(external) = &p.base.cpu_r {
                        result.extend([&external.profile, &external.binary]);
                    }
                }
                PalsEngineV4::OwnCpu(p) => result.push(&p.binary),
                PalsEngineV4::ReferenceUci(p) => {
                    result.push(&p.binary);
                    result.extend(&p.assets);
                }
            }
        }
        result
    }
    pub fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        let mut unique = BTreeMap::new();
        for a in self.declared_artifacts() {
            a.validate()?;
            if let Some(previous) = unique.insert(&a.path, a) {
                require(
                    previous == a,
                    "one logical artifact path has conflicting identity/provenance",
                )?;
            }
        }
        unique
            .values()
            .try_fold(0_u64, |total, a| total.checked_add(a.bytes))
            .ok_or_else(|| ManifestError::Integrity("PALS V4: artifact byte total overflow".into()))
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        encoded(&(PALS_MANIFEST_V4_DOMAIN, self), false)
    }
    pub fn lock(&self) -> Result<PalsInputLockV4, ManifestError> {
        Ok(PalsInputLockV4 {
            domain: PALS_MANIFEST_V4_DOMAIN.into(),
            manifest: self.clone(),
            canonical_sha256: digest(&self.canonical_bytes()?),
        })
    }
}
impl PalsInputLockV4 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let l: Self = decode_json(input)?;
        l.verify()?;
        Ok(l)
    }
    pub fn verify(&self) -> Result<(), ManifestError> {
        require(
            self.domain == PALS_MANIFEST_V4_DOMAIN
                && sha(&self.canonical_sha256, 64)
                && digest(&self.manifest.canonical_bytes()?) == self.canonical_sha256,
            "lock domain/digest mismatch",
        )
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        self.verify()?;
        String::from_utf8(encoded(self, true)?).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsVerifiedArtifactV4 {
    pub sha256: String,
    pub bytes: u64,
    /// Historical proof that the loader/launcher consumed the pinned verified
    /// handle; merely re-opening an equal pathname after hashing is insufficient.
    pub pinned_handle_consumed: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsModelCheckV4 {
    pub model: PalsModelIdentityV3,
    pub specification: PalsModelSpecificationV4,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPolicyCheckV4 {
    pub search: PalsComponentV3,
    pub policies: PalsPoliciesV4,
    pub identity: PalsPolicyIdentityV4,
}
/// Unknown is not an observed match. Declarations stay in the locked manifest;
/// these fields require independent producer checks, including actual getters.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRuntimeChecksV4 {
    pub artifacts: PalsObservedV3<BTreeMap<String, PalsVerifiedArtifactV4>>,
    pub model: PalsObservedV3<PalsModelCheckV4>,
    pub policies: PalsObservedV3<PalsPolicyCheckV4>,
    pub resource_limits: PalsObservedV3<PalsResourcePolicyV3>,
    pub physical_work_accounting: PalsObservedV3<bool>,
}
impl Default for PalsRuntimeChecksV4 {
    fn default() -> Self {
        Self {
            artifacts: PalsObservedV3::Unknown,
            model: PalsObservedV3::Unknown,
            policies: PalsObservedV3::Unknown,
            resource_limits: PalsObservedV3::Unknown,
            physical_work_accounting: PalsObservedV3::Unknown,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsArchiveObservationV4 {
    pub owner_id: u64,
    pub generation: u64,
    pub lifecycle: PalsArchiveLifecycleV4,
    pub committed_chunks: u64,
    pub integrity_verified_chunks: u64,
    pub committed_bytes: u64,
    pub integrity_verified_bytes: u64,
    pub ram_released_bytes: u64,
    pub pending_commit_bytes: u64,
    pub pending_buffers_retained: bool,
    pub global_managed_bytes: u64,
    pub index_entries_peak: u32,
    pub index_bytes_peak: u64,
    pub pinned_entries_peak: u32,
    pub loads_requested: u64,
    pub loads_completed: u64,
    pub load_bytes_total: u64,
    pub load_bytes_peak: u64,
    pub load_elapsed_peak_ms: u64,
    pub owner_generation_checks: u64,
    pub quota_failures: u64,
    pub io_failures: u64,
    pub pin_saturation_failures: u64,
    pub cleanup_complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_scope: Option<PalsArchiveActualScopeV4>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPausedStackObservationV4 {
    pub tokens_created: u64,
    pub tokens_resumed: u64,
    pub tokens_invalidated: u64,
    pub tokens_retained: u32,
    pub tokens_peak: u32,
    pub bytes_peak: u64,
    pub stale_context_attempts: u64,
    pub stale_context_rejections: u64,
    /// Resumed consumed work must never be counted a second time.
    pub replayed_consumed_work: u64,
    pub owner_released: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_scope: Option<PalsPausedStackActualScopeV4>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaWarmCapabilityV4 {
    pub semantic_id: String,
    pub device_identity: String,
    pub available: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaWarmObservationV4 {
    pub capability: PalsObservedV3<PalsCudaWarmCapabilityV4>,
    pub leases_admitted: u64,
    pub leases_physically_completed: u64,
    pub leases_quarantined: u64,
    pub leases_peak: u32,
    pub device_bytes_peak: u64,
    pub accepted_seed_consumptions: u64,
    /// Each accepted seed has checked role/model/Rules/query identity; only
    /// public record revisions may differ. No unknown identity is a match.
    pub seed_context_checked_consumptions: u64,
    pub rejected_seed_contexts: u64,
    pub fresh_value_evaluations: u64,
    pub buffers_released: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_scope: Option<PalsCudaWarmActualScopeV4>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsProcessCleanupV4 {
    pub owner_identity: String,
    pub process_exited: bool,
    pub exit_code: Option<i32>,
    pub exit_signal: Option<i32>,
    pub child_processes_remaining: u32,
    pub stdout_drained: bool,
    pub stderr_drained: bool,
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
    pub elapsed_ms: u64,
    pub cleanup_complete: bool,
    pub ownership_lost: bool,
}

/// Values share an explicit comparison perspective. Own raw/foreign CP are not
/// representable here. Rules terminals remain Rules facts, never fake WDL.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsRecheckValueV4 {
    Unknown,
    FrozenWdl {
        model_sha256: String,
        model_epoch: u64,
        encoding_sha256: String,
        precision: String,
        context_revision: u64,
        input_sha256: String,
        perspective: PalsPerspectiveV4,
        fresh: bool,
        wdl: [f32; 3],
    },
    RulesTerminal {
        state_sha256: String,
        perspective: PalsPerspectiveV4,
        #[serde(deserialize_with = "deserialize_present_winner")]
        winner: Option<PalsPerspectiveV4>,
    },
}
fn deserialize_present_winner<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<PalsPerspectiveV4>, D::Error> {
    Option::<PalsPerspectiveV4>::deserialize(deserializer)
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRecheckObservationV4 {
    pub before: PalsRecheckValueV4,
    pub after: PalsRecheckValueV4,
    pub resolution: PalsRecheckResolutionV4,
    /// Extra Fresh inputs are part of original physical NN accounting/budget.
    pub fresh_nn_inputs_charged: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_scope: Option<PalsRecheckActualScopeV4>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRepairTraceV4 {
    pub root_generation: u64,
    pub first_move: String,
    pub original_record: u64,
    pub record_id: u64,
    pub supersedes: Option<u64>,
    pub question_sha256: String,
    pub evidence_revision: u64,
    pub repair_ordinal: u32,
    pub evidence_sha256: String,
    pub recheck: PalsObservedV3<PalsRecheckObservationV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_scope: Option<PalsRepairActualScopeV4>,
    /// Actual candidate admission comparison is independent of the later
    /// accepted Repair → reply C comparison and its physical Fresh inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterative_repair_comparison: Option<PalsObservedV3<PalsRecheckObservationV4>>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsEndpointReceiptV4 {
    /// Complete shared work units: physically completed inputs/calls, accepted
    /// inputs, NN cache consumptions, and own CPU requested/completed/reused/
    /// consumed work remain distinct. Unknown physical completion retains leases.
    pub base: PalsEndpointReceiptV3,
    pub checks: PalsRuntimeChecksV4,
    pub archive: PalsObservedV3<PalsArchiveObservationV4>,
    pub paused_stack: PalsObservedV3<PalsPausedStackObservationV4>,
    pub cuda_warm: PalsObservedV3<PalsCudaWarmObservationV4>,
    pub process_cleanup: PalsObservedV3<PalsProcessCleanupV4>,
    pub repair_traces: Vec<PalsRepairTraceV4>,
    pub repair_trace_total: PalsObservedV3<u64>,
    pub pending_questions_peak: PalsObservedV3<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_admissions_peak: Option<PalsObservedV3<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub followup_lifecycle: Option<PalsFollowupLifecycleScopeV4>,
}
impl PalsEndpointReceiptV4 {
    /// Wrap actual shared facts while explicitly leaving every new check unknown.
    /// This constructor cannot produce an arena-eligible receipt by itself.
    pub fn with_unknown_checks(base: PalsEndpointReceiptV3) -> Self {
        Self {
            base,
            checks: PalsRuntimeChecksV4::default(),
            archive: PalsObservedV3::Unknown,
            paused_stack: PalsObservedV3::Unknown,
            cuda_warm: PalsObservedV3::Unknown,
            process_cleanup: PalsObservedV3::Unknown,
            repair_traces: Vec::new(),
            repair_trace_total: PalsObservedV3::Unknown,
            pending_questions_peak: PalsObservedV3::Unknown,
            repair_admissions_peak: None,
            followup_lifecycle: None,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsGameReceiptV4 {
    pub game_index: u32,
    pub white_endpoint: String,
    pub black_endpoint: String,
    pub base_ms: u64,
    pub increment_ms: u64,
    pub plies: u32,
    pub result: PalsResultV3,
    pub termination: PalsTerminationV3,
    pub failed_endpoint: Option<String>,
    pub engines: [PalsEndpointReceiptV4; 2],
    pub pgn: ArtifactRef,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsTrainingObservationV4 {
    pub updates_completed: u32,
    pub checkpoint: ArtifactRef,
    pub target_coverage: PalsTargetCoverageV4,
    pub target_coverage_sha256: String,
    pub continuous_resume_match: PalsObservedV3<bool>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRunReceiptV4 {
    pub schema_version: u32,
    pub domain: String,
    pub run_id: String,
    pub pair_id: String,
    pub lock_sha256: String,
    pub training_executed: bool,
    pub training: PalsObservedV3<PalsTrainingObservationV4>,
    pub wall_time_ms: u64,
    pub cleanup_time_ms: u64,
    pub output_bytes: PalsObservedV3<u64>,
    /// Ineligible evidence can have unknown checks and no observed failure.
    /// Eligibility requires all independent checks plus completed paired games.
    pub pair_eligible: bool,
    pub failures: BTreeSet<PalsRunFailureV3>,
    pub policy_failures: BTreeSet<PalsPolicyFailureV4>,
    pub games: Vec<PalsGameReceiptV4>,
}

fn observed<T>(o: &PalsObservedV3<T>) -> Option<&T> {
    match o {
        PalsObservedV3::Unknown => None,
        PalsObservedV3::Observed { value, .. } => Some(value),
    }
}
fn checked_observation<T>(
    o: &PalsObservedV3<T>,
    required: bool,
) -> Result<Option<&T>, ManifestError> {
    shared::observation(o)?;
    require(
        !required || observed(o).is_some(),
        "eligible receipt lacks an independent observation",
    )?;
    Ok(observed(o))
}
fn resource_failure(
    ok: bool,
    eligible: bool,
    failures: &BTreeSet<PalsRunFailureV3>,
    message: &str,
) -> Result<(), ManifestError> {
    require(
        ok || (!eligible && failures.contains(&PalsRunFailureV3::ResourceAdmission)),
        message,
    )
}
fn policy_failure(
    ok: bool,
    eligible: bool,
    failures: &BTreeSet<PalsPolicyFailureV4>,
    failure: PalsPolicyFailureV4,
    message: &str,
) -> Result<(), ManifestError> {
    require(ok || (!eligible && failures.contains(&failure)), message)
}
fn native_checkpoint(e: &PalsEndpointV4) -> Option<&ArtifactRef> {
    match &e.base.model.weights {
        PalsWeightIdentityV3::Untrained { artifact, .. }
        | PalsWeightIdentityV3::Trained { artifact, .. } => Some(artifact),
        PalsWeightIdentityV3::DeterministicMock { .. } => None,
    }
}
fn endpoint_artifacts(e: &PalsEngineV4) -> Vec<&ArtifactRef> {
    match e {
        PalsEngineV4::Pals(p) => {
            let mut result = vec![&p.base.binary];
            if let Some(a) = native_checkpoint(p) {
                result.push(a);
            }
            if let PalsCpuRSelectionV3::ExternalUci(c) = &p.base.cpu_r {
                result.extend([&c.profile, &c.binary]);
            }
            result
        }
        PalsEngineV4::OwnCpu(p) => vec![&p.binary],
        PalsEngineV4::ReferenceUci(p) => {
            let mut result = vec![&p.binary];
            result.extend(&p.assets);
            result
        }
    }
}
impl PalsRuntimeChecksV4 {
    fn validate_against(
        &self,
        e: &PalsEngineV4,
        r: &PalsResourcePolicyV3,
        pilot: &PalsPilotV3,
        eligible: bool,
        failures: &BTreeSet<PalsPolicyFailureV4>,
    ) -> Result<(), ManifestError> {
        if let Some(artifacts) = checked_observation(&self.artifacts, eligible)? {
            let expected: BTreeMap<_, _> = endpoint_artifacts(e)
                .into_iter()
                .map(|a| (&a.path, a))
                .collect();
            require(
                artifacts.len() == expected.len() && artifacts.keys().eq(expected.keys().copied()),
                "artifact check set differs from locked endpoint inputs",
            )?;
            for (path, a) in expected {
                let o = &artifacts[path];
                policy_failure(
                    o.sha256 == a.sha256 && o.bytes == a.bytes && o.pinned_handle_consumed,
                    eligible,
                    failures,
                    PalsPolicyFailureV4::ArtifactCheck,
                    "artifact check lacks matching bytes/pinned-handle consumption",
                )?;
            }
        }
        if let Some(limits) = checked_observation(&self.resource_limits, eligible)? {
            shared::resources(limits)?;
            policy_failure(
                limits == r,
                eligible,
                failures,
                PalsPolicyFailureV4::PolicyIdentity,
                "actual resource-limit check differs from declaration",
            )?;
        }
        if let Some(work) = checked_observation(
            &self.physical_work_accounting,
            eligible && !matches!(e, PalsEngineV4::ReferenceUci(_)),
        )? {
            require(
                *work,
                "physical work accounting check cannot be declared as observed false",
            )?;
        }
        match e {
            PalsEngineV4::Pals(p) => {
                if let Some(model) = checked_observation(&self.model, eligible)? {
                    policy_failure(
                        model.model == p.base.model && model.specification == p.model_v2,
                        eligible,
                        failures,
                        PalsPolicyFailureV4::ModelIdentity,
                        "actual model/profile/encoding/checkpoint differs from lock",
                    )?;
                    model.specification.validate_against(&PalsEndpointV3 {
                        model: model.model.clone(),
                        ..p.base.clone()
                    })?;
                }
                if let Some(policy) = checked_observation(&self.policies, eligible)? {
                    policy_failure(
                        policy.search == p.base.search
                            && policy.policies == p.policies
                            && policy.identity == p.policy_identity,
                        eligible,
                        failures,
                        PalsPolicyFailureV4::PolicyIdentity,
                        "actual selected policy registration differs from lock",
                    )?;
                    // An observed unsupported registration is invalid even for a
                    // failed run; raw failure evidence can remain a separate log.
                    let actual = PalsEndpointV4 {
                        policies: policy.policies.clone(),
                        policy_identity: policy.identity.clone(),
                        base: PalsEndpointV3 {
                            search: policy.search.clone(),
                            ..p.base.clone()
                        },
                        ..p.as_ref().clone()
                    };
                    actual.policy_identity.validate_against(&actual)?;
                    actual.policies.validate_against(&actual.base, r, pilot)?;
                }
            }
            _ => require(
                matches!(self.model, PalsObservedV3::Unknown)
                    && matches!(self.policies, PalsObservedV3::Unknown),
                "non-PALS endpoint cannot claim V4 model/policy checks",
            )?,
        }
        Ok(())
    }
}
impl PalsArchiveObservationV4 {
    fn validate_against(
        &self,
        l: &PalsColdArchiveLimitsV4,
        eligible: bool,
        failures: &BTreeSet<PalsPolicyFailureV4>,
    ) -> Result<(), ManifestError> {
        if let Some(scope) = &self.actual_scope {
            followup::validate_actual_archive(self, scope, l, eligible)?;
        }
        require(
            self.owner_id > 0
                && (self.actual_scope.is_some() || self.generation > 0)
                && self.integrity_verified_chunks <= self.committed_chunks
                && self.integrity_verified_bytes <= self.committed_bytes
                && (self.actual_scope.is_some()
                    || self.ram_released_bytes <= self.integrity_verified_bytes)
                && (self.actual_scope.is_some()
                    || self.pending_commit_bytes == 0
                    || self.pending_buffers_retained)
                && self.loads_completed <= self.loads_requested
                && self.owner_generation_checks >= self.loads_requested
                && (self.actual_scope.is_some()
                    || (self.pinned_entries_peak <= self.index_entries_peak
                        && self.global_managed_bytes >= self.committed_bytes)),
            "archive owner/generation/commit/integrity/RAM-release/pin accounting inconsistent",
        )?;
        policy_failure(
            self.committed_bytes
                .checked_add(self.pending_commit_bytes)
                .is_some_and(|n| n <= l.game_bytes_max)
                && self.global_managed_bytes <= l.global_bytes_max
                && self.index_entries_peak <= l.index_entries_max
                && self.index_bytes_peak <= l.index_bytes_max
                && self.load_bytes_peak <= l.load_bytes_max
                && self.load_elapsed_peak_ms <= l.load_deadline_max_ms
                && self.quota_failures == 0,
            eligible,
            failures,
            PalsPolicyFailureV4::ArchiveQuota,
            "archive/index/load quota overrun lacks preserved failure",
        )?;
        policy_failure(
            self.io_failures == 0,
            eligible,
            failures,
            PalsPolicyFailureV4::ArchiveIo,
            "archive I/O failure must remain an explicit failed run",
        )?;
        policy_failure(
            self.pin_saturation_failures == 0,
            eligible,
            failures,
            PalsPolicyFailureV4::ArchivePinSaturated,
            "archive pin saturation must remain an explicit failed run",
        )?;
        if self.lifecycle == PalsArchiveLifecycleV4::Closed {
            require(
                self.pending_commit_bytes == 0 && self.cleanup_complete,
                "closed archive still owns an uncommitted buffer or unfinished cleanup",
            )?;
        }
        if eligible {
            require(
                self.lifecycle == PalsArchiveLifecycleV4::Closed
                    && self.integrity_verified_chunks == self.committed_chunks
                    && self.integrity_verified_bytes == self.committed_bytes
                    && self.loads_completed == self.loads_requested,
                "eligible archive lacks commit/integrity/load/cleanup completion",
            )?;
        }
        Ok(())
    }
}
impl PalsPausedStackObservationV4 {
    fn validate_against(
        &self,
        l: &PalsPausedStackLimitsV4,
        eligible: bool,
        failures: &BTreeSet<PalsPolicyFailureV4>,
    ) -> Result<(), ManifestError> {
        if let Some(scope) = &self.actual_scope {
            followup::validate_actual_stack(self, scope, eligible)?;
        }
        require(
            self.tokens_resumed
                .checked_add(self.tokens_invalidated)
                .and_then(|n| n.checked_add(u64::from(self.tokens_retained)))
                == Some(self.tokens_created)
                && self.stale_context_attempts == self.stale_context_rejections
                && self.replayed_consumed_work == 0
                && self.tokens_retained <= self.tokens_peak
                && (self.tokens_created == 0) == (self.tokens_peak == 0)
                && (self.tokens_peak == 0) == (self.bytes_peak == 0)
                && (!self.owner_released || self.tokens_retained == 0),
            "PausedStack consumed work/context/token ownership inconsistent",
        )?;
        policy_failure(
            self.tokens_peak <= l.tokens_max && self.bytes_peak <= l.bytes_max,
            eligible,
            failures,
            PalsPolicyFailureV4::PausedStack,
            "PausedStack exceeds one token/byte ceiling without explicit failure",
        )?;
        require(
            !eligible || (self.owner_released && self.tokens_retained == 0),
            "eligible CPU process still retains a paused stack owner",
        )
    }
}
impl PalsCudaWarmObservationV4 {
    fn validate_against(
        &self,
        l: &PalsCudaWarmLimitsV4,
        o: &PalsEndpointReceiptV3,
        eligible: bool,
        failures: &BTreeSet<PalsPolicyFailureV4>,
    ) -> Result<(), ManifestError> {
        if let Some(scope) = &self.actual_scope {
            followup::validate_actual_warm(self, scope, o, eligible)?;
        }
        if let Some(capability) =
            checked_observation(&self.capability, eligible || self.leases_admitted > 0)?
        {
            require(
                capability.semantic_id == l.capability && text(&capability.device_identity),
                "CUDA Warm capability identity differs from separate selected device path",
            )?;
            policy_failure(
                capability.available,
                eligible,
                failures,
                PalsPolicyFailureV4::CudaCapability,
                "unavailable CUDA Warm capability cannot be accepted or silently replaced",
            )?;
        }
        require(
            self.leases_physically_completed
                .checked_add(self.leases_quarantined)
                .is_some_and(|n| n <= self.leases_admitted)
                && self.seed_context_checked_consumptions == self.accepted_seed_consumptions
                && (self.actual_scope.is_some()
                    || self.accepted_seed_consumptions <= self.fresh_value_evaluations)
                && self.fresh_value_evaluations <= o.nn_inputs_completed
                && (self.leases_admitted == 0 || self.leases_peak > 0),
            "CUDA Warm lease/accepted-seed/Fresh-value physical accounting inconsistent",
        )?;
        policy_failure(
            self.leases_peak <= l.max_leases && self.device_bytes_peak <= l.device_bytes_max,
            eligible,
            failures,
            PalsPolicyFailureV4::CudaCapability,
            "CUDA Warm lease/device quota exceeded without explicit failure",
        )?;
        let complete = self.leases_physically_completed == self.leases_admitted
            && self.leases_quarantined == 0;
        require(
            complete
                || (!self.buffers_released
                    && !o.buffers_released
                    && matches!(
                        o.physical_state,
                        PalsPhysicalStateV3::Unknown | PalsPhysicalStateV3::Quarantined
                    )),
            "unknown CUDA Warm completion must retain/quarantine physical owner and buffers",
        )?;
        policy_failure(
            complete && self.buffers_released,
            eligible,
            failures,
            PalsPolicyFailureV4::CudaCompletionUnknown,
            "CUDA Warm unknown physical completion lacks preserved failure",
        )
    }
}
impl PalsRecheckObservationV4 {
    fn validate_against(&self, e: &PalsEndpointV4) -> Result<(), ManifestError> {
        if let Some(scope) = &self.actual_scope {
            followup::validate_actual_recheck(self, scope, e)?;
        }
        for value in [&self.before, &self.after] {
            match value {
                PalsRecheckValueV4::Unknown => {}
                PalsRecheckValueV4::FrozenWdl {
                    model_sha256,
                    model_epoch,
                    encoding_sha256,
                    precision,
                    input_sha256,
                    fresh,
                    wdl,
                    ..
                } => {
                    require(
                        native_checkpoint(e).is_some_and(|a| &a.sha256 == model_sha256)
                            && *model_epoch == e.base.model.frozen_epoch
                            && encoding_sha256 == &e.model_v2.encoding_sha256
                            && precision
                                == match e.base.model.precision {
                                    crate::PalsPrecisionV3::Fp32 => "fp32",
                                    crate::PalsPrecisionV3::Fp16 => "fp16",
                                    crate::PalsPrecisionV3::Bf16 => "bf16",
                                }
                            && sha(input_sha256, 64)
                            && *fresh
                            && wdl.iter().all(|n| n.is_finite() && (0.0..=1.0).contains(n))
                            && (wdl.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
                        "recheck requires actual frozen model/precision/encoding and finite Fresh WDL",
                    )?;
                }
                PalsRecheckValueV4::RulesTerminal { state_sha256, .. } => require(
                    sha(state_sha256, 64),
                    "Rules terminal needs exact state identity",
                )?,
            }
        }
        let order = match (&self.before, &self.after) {
            (
                PalsRecheckValueV4::FrozenWdl {
                    perspective: p,
                    context_revision: c,
                    wdl: a,
                    ..
                },
                PalsRecheckValueV4::FrozenWdl {
                    perspective: q,
                    context_revision: d,
                    wdl: b,
                    ..
                },
            ) => {
                require(
                    p == q
                        && c == d
                        && (self.actual_scope.is_some() || self.fresh_nn_inputs_charged == 2),
                    "both Fresh WDL endpoints need the same perspective/context and two budgeted inputs",
                )?;
                (a[0] - a[2]).partial_cmp(&(b[0] - b[2]))
            }
            (
                PalsRecheckValueV4::RulesTerminal {
                    perspective: p,
                    winner: a,
                    ..
                },
                PalsRecheckValueV4::RulesTerminal {
                    perspective: q,
                    winner: b,
                    ..
                },
            ) => {
                require(
                    p == q && self.fresh_nn_inputs_charged == 0,
                    "two Rules terminals compare in one perspective without fabricated NN cost",
                )?;
                let score = |winner: &Option<PalsPerspectiveV4>| match winner {
                    Some(w) if w == p => 1_i32,
                    Some(_) => -1,
                    None => 0,
                };
                Some(score(a).cmp(&score(b)))
            }
            _ => {
                let fresh = [&self.before, &self.after]
                    .iter()
                    .filter(|v| matches!(v, PalsRecheckValueV4::FrozenWdl { .. }))
                    .count() as u64;
                require(
                    self.actual_scope.is_some() || self.fresh_nn_inputs_charged == fresh,
                    "mixed/unknown recheck must retain only its actual Fresh NN cost",
                )?;
                None
            }
        };
        let expected = match order {
            Some(std::cmp::Ordering::Greater) => PalsRecheckResolutionV4::BeforePreferred,
            Some(std::cmp::Ordering::Less) => PalsRecheckResolutionV4::AfterPreferred,
            Some(std::cmp::Ordering::Equal) => PalsRecheckResolutionV4::Equal,
            None => PalsRecheckResolutionV4::Unresolved,
        };
        require(
            self.resolution == expected,
            "recheck resolution differs from common frozen-WDL/Rules basis; mixed/unknown must stay unresolved",
        )
    }
}

fn first_move(s: &str) -> bool {
    let b = s.as_bytes();
    (b.len() == 4 || b.len() == 5)
        && (b'a'..=b'h').contains(&b[0])
        && (b'1'..=b'8').contains(&b[1])
        && (b'a'..=b'h').contains(&b[2])
        && (b'1'..=b'8').contains(&b[3])
        && b[..2] != b[2..4]
        && (b.len() == 4 || b"qrbn".contains(&b[4]))
}
fn validate_traces(
    o: &PalsEndpointReceiptV4,
    p: &PalsEndpointV4,
    eligible: bool,
    failures: &BTreeSet<PalsRunFailureV3>,
) -> Result<(), ManifestError> {
    if o.followup_lifecycle.is_some() {
        return followup::validate_actual_traces(o, p, eligible, failures);
    }
    require(
        o.repair_admissions_peak.is_none()
            && o.repair_traces.iter().all(|t| {
                t.actual_scope.is_none()
                    && t.iterative_repair_comparison.is_none()
                    && observed(&t.recheck).is_none_or(|r| r.actual_scope.is_none())
            }),
        "actual followup evidence cannot inherit a historical synthetic scope",
    )?;
    require(
        o.repair_traces.len() <= PALS_V4_REPAIR_TRACE_MAX,
        "repair trace collection exceeds bounded receipt capacity",
    )?;
    let active = p.policies.recheck == PalsRecheckPolicyV4::FrozenWdl;
    let (repairs_max, pending_max) = match &p.policies.iterative_repair {
        PalsIterativeRepairPolicyV4::Disabled => (1, 0),
        PalsIterativeRepairPolicyV4::Bounded(l) => {
            (l.max_repairs_per_first_move, l.max_pending_questions)
        }
    };
    if let Some(total) = checked_observation(&o.repair_trace_total, eligible && active)? {
        require(
            *total >= o.repair_traces.len() as u64
                && (!eligible || *total == o.repair_traces.len() as u64)
                && (active || *total == 0),
            "repair trace omission cannot become observed complete or execute a disabled lane",
        )?;
    }
    if let Some(peak) = checked_observation(&o.pending_questions_peak, eligible && pending_max > 0)?
    {
        resource_failure(
            *peak <= pending_max,
            eligible,
            failures,
            "pending Repair queue exceeds admitted capacity without explicit resource failure",
        )?;
    }
    require(
        active || o.repair_traces.is_empty(),
        "disabled frozen-WDL recheck cannot claim new repair traces",
    )?;
    let mut records = BTreeMap::<(u64, u64), &PalsRepairTraceV4>::new();
    let mut last = BTreeMap::<(u64, &str), &PalsRepairTraceV4>::new();
    let mut questions = BTreeSet::new();
    let mut charged = 0_u64;
    for t in &o.repair_traces {
        require(
            t.root_generation > 0
                && first_move(&t.first_move)
                && t.record_id != t.original_record
                && sha(&t.question_sha256, 64)
                && sha(&t.evidence_sha256, 64)
                && (1..=repairs_max).contains(&t.repair_ordinal)
                && questions.insert((t.root_generation, &t.question_sha256, t.evidence_revision))
                && !records.contains_key(&(t.root_generation, t.record_id)),
            "repair identity/first move/question-revision/record/ordinal is invalid or duplicated",
        )?;
        if let Some(previous) = t
            .supersedes
            .and_then(|id| records.get(&(t.root_generation, id)))
        {
            require(
                previous.first_move == t.first_move
                    && previous.original_record == t.original_record,
                "supersedes cannot switch original first move or evidence lineage",
            )?;
        }
        let key = (t.root_generation, t.first_move.as_str());
        if let Some(previous) = last.get(&key) {
            require(
                t.original_record == previous.original_record
                    && t.supersedes == Some(previous.record_id)
                    && t.repair_ordinal == previous.repair_ordinal + 1
                    && t.evidence_revision > previous.evidence_revision,
                "repair must retain original first move and supersede latest record only with new evidence",
            )?;
        } else {
            require(
                t.repair_ordinal == 1 && t.supersedes == Some(t.original_record),
                "first Repair must name the original record it supersedes",
            )?;
        }
        if let Some(recheck) = checked_observation(&t.recheck, eligible)? {
            recheck.validate_against(p)?;
            charged = charged
                .checked_add(recheck.fresh_nn_inputs_charged)
                .ok_or_else(|| {
                    ManifestError::Integrity("PALS V4: Fresh recheck accounting overflow".into())
                })?;
        }
        records.insert((t.root_generation, t.record_id), t);
        last.insert(key, t);
    }
    require(
        charged <= o.base.nn_inputs_completed,
        "extra Fresh recheck inputs are missing from physical NN accounting/global budget",
    )
}
fn validate_endpoint(
    o: &PalsEndpointReceiptV4,
    e: &PalsEngineV4,
    r: &PalsResourcePolicyV3,
    m: &PalsRunManifestV4,
    receipt: &PalsRunReceiptV4,
) -> Result<(), ManifestError> {
    let eligible = receipt.pair_eligible;
    // Only shared facts are projected: actual model/search/CPU labels are kept.
    shared::receipt_endpoint_physical(&o.base, &e.shared_engine(), r, eligible, &receipt.failures)?;
    require(
        o.base.pals_search_policy.is_none(),
        "V4 cannot claim a V3 search-policy observation",
    )?;
    match (e, &o.base.external_cpu_r) {
        (PalsEngineV4::Pals(p), external) if !p.base.cpu_r.is_own() => {
            require(
                o.base.cpu_tasks_requested == 0
                    && o.base.cpu_tasks_completed == 0
                    && o.base.cpu_tasks_reused_consumed == 0
                    && o.base.cpu_tasks_consumed == 0
                    && o.base.cpu_nodes == 0,
                "external CPU work cannot be projected as own raw CPU work",
            )?;
            if let Some(proof) = external {
                proof.validate_against(&p.base, r)?;
                require(
                    !eligible
                        || (proof.work.tasks_dispatched.is_some()
                            && proof.work.reports_returned.is_some()
                            && proof.work.node_budget_reserved.is_some()
                            && proof.work.consumed_completed_tasks.is_some()
                            && proof.work.work_incomplete.is_some()),
                    "eligible external CPU_R lacks independently observed work accounting",
                )?;
            } else {
                require(
                    !eligible,
                    "unknown external checker ready/cleanup/work proof cannot become arena eligible",
                )?;
            }
        }
        (_, None) => {}
        _ => {
            return require(
                false,
                "own/non-PALS endpoint cannot claim external checker evidence",
            );
        }
    }
    o.checks
        .validate_against(e, r, &m.pilot, eligible, &receipt.policy_failures)?;
    if matches!(
        o.base.physical_state,
        PalsPhysicalStateV3::Unknown | PalsPhysicalStateV3::Quarantined
    ) {
        require(
            !eligible
                && !o.base.buffers_released
                && receipt
                    .failures
                    .contains(&PalsRunFailureV3::PhysicalCompletionUnknown),
            "unknown physical completion must preserve failure and retain buffers",
        )?;
    }
    if let Some(cleanup) = checked_observation(&o.process_cleanup, eligible)? {
        require(
            text(&cleanup.owner_identity)
                && cleanup.process_exited == o.base.process_exited
                && !(cleanup.exit_code.is_some() && cleanup.exit_signal.is_some()),
            "process cleanup identity/exit observations conflict",
        )?;
        if cleanup.cleanup_complete {
            require(
                cleanup.process_exited
                    && (cleanup.exit_code.is_some() || cleanup.exit_signal.is_some())
                    && cleanup.child_processes_remaining == 0
                    && cleanup.stdout_drained
                    && cleanup.stderr_drained
                    && !cleanup.ownership_lost,
                "process/child/pipe ownership cannot be released before observed cleanup",
            )?;
        }
        policy_failure(
            cleanup.cleanup_complete
                && cleanup.elapsed_ms <= m.pilot.cleanup_max_ms
                && cleanup
                    .stdout_bytes
                    .checked_add(cleanup.stderr_bytes)
                    .is_some_and(|bytes| bytes <= m.output_bytes_max),
            eligible,
            &receipt.policy_failures,
            PalsPolicyFailureV4::ProcessCleanup,
            "incomplete/over-budget process cleanup lacks explicit failure",
        )?;
    }
    match e {
        PalsEngineV4::Pals(p) => {
            if let Some(scope) = &o.followup_lifecycle {
                scope.validate_for(&p.policies, eligible)?;
                followup::validate_actual_endpoint_scope(o, &p.policies, eligible)?;
            } else {
                require(
                    observed(&o.archive).is_none_or(|a| a.actual_scope.is_none())
                        && observed(&o.paused_stack).is_none_or(|s| s.actual_scope.is_none())
                        && observed(&o.cuda_warm).is_none_or(|w| w.actual_scope.is_none()),
                    "actual owner evidence has no independently bound lifecycle scope",
                )?;
            }
            match &p.policies.cold_archive {
                PalsColdArchivePolicyV4::Bounded(l) => {
                    if let Some(archive) = checked_observation(&o.archive, eligible)? {
                        archive.validate_against(l, eligible, &receipt.policy_failures)?;
                    }
                }
                PalsColdArchivePolicyV4::Disabled => {
                    if let Some(a) = checked_observation(&o.archive, false)? {
                        require(
                            a.committed_chunks == 0
                                && a.committed_bytes == 0
                                && a.ram_released_bytes == 0
                                && a.pending_commit_bytes == 0
                                && a.loads_requested == 0
                                && a.loads_completed == 0
                                && a.load_bytes_total == 0,
                            "disabled archive cannot claim archive/RAM-release/load work",
                        )?;
                    }
                }
            }
            match &p.policies.paused_stack {
                PalsPausedStackPolicyV4::Bounded(l) => {
                    if let Some(stack) = checked_observation(&o.paused_stack, eligible)? {
                        stack.validate_against(l, eligible, &receipt.policy_failures)?;
                    }
                }
                PalsPausedStackPolicyV4::Disabled => {
                    if let Some(s) = checked_observation(&o.paused_stack, false)? {
                        require(
                            s.tokens_created == 0
                                && s.tokens_resumed == 0
                                && s.tokens_invalidated == 0
                                && s.tokens_retained == 0
                                && s.tokens_peak == 0
                                && s.bytes_peak == 0
                                && s.replayed_consumed_work == 0,
                            "disabled PausedStack cannot claim retained/resumed CPU frames",
                        )?;
                    }
                }
            }
            match &p.policies.cuda_warm {
                PalsCudaWarmPolicyV4::ApproxWarm(l) => {
                    if let Some(warm) = checked_observation(&o.cuda_warm, eligible)? {
                        warm.validate_against(l, &o.base, eligible, &receipt.policy_failures)?;
                    }
                }
                PalsCudaWarmPolicyV4::Disabled => {
                    if let Some(w) = checked_observation(&o.cuda_warm, false)? {
                        shared::observation(&w.capability)?;
                        require(
                            w.leases_admitted == 0
                                && w.leases_physically_completed == 0
                                && w.leases_quarantined == 0
                                && w.leases_peak == 0
                                && w.device_bytes_peak == 0
                                && w.accepted_seed_consumptions == 0
                                && w.seed_context_checked_consumptions == 0
                                && w.fresh_value_evaluations == 0,
                            "disabled CUDA Warm cannot claim seed/device-lease work",
                        )?;
                    }
                }
            }
            validate_traces(o, p, eligible, &receipt.failures)?;
        }
        _ => require(
            matches!(o.archive, PalsObservedV3::Unknown)
                && matches!(o.paused_stack, PalsObservedV3::Unknown)
                && matches!(o.cuda_warm, PalsObservedV3::Unknown)
                && matches!(o.repair_trace_total, PalsObservedV3::Unknown)
                && matches!(o.pending_questions_peak, PalsObservedV3::Unknown)
                && o.repair_traces.is_empty()
                && o.repair_admissions_peak.is_none()
                && o.followup_lifecycle.is_none(),
            "non-PALS endpoint cannot claim PALS V4 policy work",
        )?,
    }
    Ok(())
}
impl PalsRunReceiptV4 {
    pub fn from_json(input: &str, lock: &PalsInputLockV4) -> Result<Self, ManifestError> {
        let r: Self = decode_json(input)?;
        r.validate_against(lock)?;
        Ok(r)
    }
    pub fn validate_against(&self, lock: &PalsInputLockV4) -> Result<(), ManifestError> {
        lock.verify()?;
        let m = &lock.manifest;
        require(
            self.schema_version == 4
                && self.domain == PALS_RECEIPT_V4_DOMAIN
                && self.run_id == m.run_id
                && self.pair_id == m.pair_id
                && self.lock_sha256 == lock.canonical_sha256
                && self.training_executed == m.training_executed,
            "receipt schema/identity/lock/training differs",
        )?;
        require(
            self.games.len() <= 2
                && (!self.pair_eligible
                    || (self.games.len() == 2
                        && m.purpose == PalsRunPurposeV4::ArenaPilot
                        && !self.training_executed
                        && self.failures.is_empty()
                        && self.policy_failures.is_empty())),
            "diagnostic/incomplete/failed run cannot become arena eligible",
        )?;
        if let Some(training) = checked_observation(&self.training, self.training_executed)? {
            training.checkpoint.validate()?;
            shared::observation(&training.continuous_resume_match)?;
            require(
                self.training_executed
                    && m.purpose == PalsRunPurposeV4::Diagnostic
                    && (1..=24).contains(&training.updates_completed)
                    && sha(&training.target_coverage_sha256, 64),
                "observed nonzero training requires diagnostic purpose and bounded actual updates/checkpoint/targets",
            )?;
            policy_failure(
                !matches!(
                    training.continuous_resume_match,
                    PalsObservedV3::Observed { value: false, .. }
                ),
                false,
                &self.policy_failures,
                PalsPolicyFailureV4::Training,
                "failed continuous/resume match must remain explicit",
            )?;
        }
        if self.wall_time_ms > m.pilot.wall_time_max_ms {
            require(
                !self.pair_eligible && self.failures.contains(&PalsRunFailureV3::WallDeadline),
                "wall deadline overshoot requires explicit failure",
            )?;
        }
        if self.cleanup_time_ms > m.pilot.cleanup_max_ms {
            require(
                !self.pair_eligible && self.failures.contains(&PalsRunFailureV3::CleanupTimeout),
                "cleanup deadline overshoot requires explicit failure",
            )?;
        }
        let output = checked_observation(&self.output_bytes, self.pair_eligible)?;
        if let Some(bytes) = output {
            policy_failure(
                *bytes <= m.output_bytes_max,
                self.pair_eligible,
                &self.policy_failures,
                PalsPolicyFailureV4::OutputQuota,
                "actual output exceeds declared bound without explicit failure",
            )?;
        }
        let mut indices = BTreeSet::new();
        let mut owners = BTreeSet::new();
        let mut helper_tuples = BTreeSet::new();
        let mut accounted_output = 0_u64;
        let mut archive_output_complete = true;
        for g in &self.games {
            require(
                g.game_index < 2
                    && indices.insert(g.game_index)
                    && g.white_endpoint == m.pilot.white_order[g.game_index as usize]
                    && g.white_endpoint != g.black_endpoint
                    && m.engines.iter().any(|e| e.id() == g.black_endpoint)
                    && g.base_ms == m.pilot.base_ms
                    && g.increment_ms == m.pilot.increment_ms
                    && g.plies <= m.pilot.max_plies
                    && g.engines[0].base.endpoint_id != g.engines[1].base.endpoint_id,
                "game index/colors/clocks/plies/endpoint receipt differ from lock",
            )?;
            g.pgn.validate()?;
            accounted_output = accounted_output.checked_add(g.pgn.bytes).ok_or_else(|| {
                ManifestError::Integrity("PALS V4: output byte accounting overflow".into())
            })?;
            for (e, r) in m.engines.iter().zip(&m.resources) {
                let o = g
                    .engines
                    .iter()
                    .find(|o| o.base.endpoint_id == e.id())
                    .ok_or_else(|| {
                        ManifestError::Integrity("PALS V4: missing endpoint receipt".into())
                    })?;
                validate_endpoint(o, e, r, m, self)?;
                if let Some(cleanup) = observed(&o.process_cleanup) {
                    require(
                        owners.insert(&cleanup.owner_identity),
                        "restarted games cannot reuse a historical process owner",
                    )?;
                    accounted_output = accounted_output
                        .checked_add(cleanup.stdout_bytes)
                        .and_then(|n| n.checked_add(cleanup.stderr_bytes))
                        .ok_or_else(|| {
                            ManifestError::Integrity("PALS V4: process output byte overflow".into())
                        })?;
                }
                {
                    let selected = matches!(e,PalsEngineV4::Pals(p) if !matches!(p.policies.cold_archive,PalsColdArchivePolicyV4::Disabled));
                    let (written, complete) = followup::archive_output_accounting(o, selected)?;
                    archive_output_complete &= complete;
                    accounted_output = accounted_output.checked_add(written).ok_or_else(|| {
                        ManifestError::Integrity("PALS V4: archive output byte overflow".into())
                    })?;
                }
                if let Some(external) = &o.base.external_cpu_r {
                    if let Some(h) = &external.shutdown.process_identity {
                        require(
                            helper_tuples.insert((h.pid, h.process_group, h.proc_start_ticks)),
                            "external CPU_R historical owner reused across restarted games",
                        )?;
                    }
                }
            }
            match g.termination {
                PalsTerminationV3::EngineCrash
                | PalsTerminationV3::IllegalMove
                | PalsTerminationV3::TimeForfeit => {
                    let expected = if g.failed_endpoint.as_ref() == Some(&g.white_endpoint) {
                        PalsResultV3::BlackWin
                    } else if g.failed_endpoint.as_ref() == Some(&g.black_endpoint) {
                        PalsResultV3::WhiteWin
                    } else {
                        return require(false, "engine failure lacks its actual offender");
                    };
                    require(
                        g.result == expected,
                        "engine failure must be preserved as an offender loss",
                    )?;
                }
                PalsTerminationV3::Checkmate => require(
                    g.failed_endpoint.is_none()
                        && matches!(g.result, PalsResultV3::WhiteWin | PalsResultV3::BlackWin),
                    "mate result differs",
                )?,
                PalsTerminationV3::RulesDraw => require(
                    g.failed_endpoint.is_none() && g.result == PalsResultV3::Draw,
                    "Rules draw result differs",
                )?,
                PalsTerminationV3::PlyLimit => require(
                    g.failed_endpoint.is_none()
                        && g.result == PalsResultV3::Draw
                        && g.plies == m.pilot.max_plies,
                    "declared ply-limit draw differs",
                )?,
                PalsTerminationV3::RunDeadline => require(
                    g.failed_endpoint.is_none()
                        && g.result == PalsResultV3::Incomplete
                        && !self.pair_eligible
                        && self.failures.contains(&PalsRunFailureV3::WallDeadline),
                    "run deadline cannot be a completed game or engine loss",
                )?,
                PalsTerminationV3::InfrastructureFailure => require(
                    g.failed_endpoint.is_none()
                        && g.result == PalsResultV3::Incomplete
                        && !self.pair_eligible
                        && self.failures.contains(&PalsRunFailureV3::Infrastructure),
                    "infrastructure failure cannot be a completed game or engine loss",
                )?,
            }
        }
        if let Some(bytes) = output {
            require(
                archive_output_complete && *bytes >= accounted_output,
                "total output observation omits known PGN/process/archive bytes",
            )?;
        }
        Ok(())
    }
    pub fn canonical_bytes(&self, lock: &PalsInputLockV4) -> Result<Vec<u8>, ManifestError> {
        self.validate_against(lock)?;
        encoded(&(PALS_RECEIPT_V4_DOMAIN, self), false)
    }
    pub fn to_json(&self, lock: &PalsInputLockV4) -> Result<String, ManifestError> {
        self.validate_against(lock)?;
        String::from_utf8(encoded(self, true)?).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PalsOwnCpuV3, PalsPoolLimitsV3, PalsPrecisionV3, PalsRoleV3};

    fn asset(path: &str) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: "a".repeat(64),
            bytes: 1,
            source: "https://example.org/source".into(),
            license: "MIT".into(),
        }
    }
    fn component(id: &str) -> PalsComponentV3 {
        PalsComponentV3 {
            semantic_id: id.into(),
            implementation_sha256: "b".repeat(64),
            options: BTreeMap::new(),
        }
    }
    fn cpu() -> PalsOwnCpuV3 {
        PalsOwnCpuV3 {
            core: component("rz-cpu-pvs/0.1"),
            training_profile: component("teacher/1"),
            runtime_profile: component("runtime/1"),
            evaluation: component("material/1"),
            max_nodes_per_task: 10_000,
            max_depth: 8,
            max_task_ms: 100,
            max_tt_bytes: 1 << 20,
        }
    }
    fn policies() -> PalsPoliciesV4 {
        PalsPoliciesV4 {
            resolver: PalsResolverPolicyV4::OwnRaw,
            resolver_identity: component(PALS_V4_OWN_RAW_RESOLVER_SEMANTICS),
            resolver_semantics_sha256: PalsResolverPolicyV4::OwnRaw.expected_semantics_sha256(),
            checker: PalsCheckerPolicyV4::Own,
            recheck: PalsRecheckPolicyV4::Disabled,
            iterative_repair: PalsIterativeRepairPolicyV4::Disabled,
            cold_archive: PalsColdArchivePolicyV4::Disabled,
            paused_stack: PalsPausedStackPolicyV4::Disabled,
            cuda_warm: PalsCudaWarmPolicyV4::Disabled,
        }
    }
    fn manifest() -> PalsRunManifestV4 {
        let policies = policies();
        let p = PalsEndpointV4 {
            base: PalsEndpointV3 {
                id: "pals".into(),
                binary: asset("bin/rovezero"),
                source_commit: "c".repeat(40),
                model: PalsModelIdentityV3 {
                    architecture: "pals-width384-latent16-iterations2".into(),
                    input_schema: PALS_V4_ENCODING_SCHEMA.into(),
                    policy_head: "candidate-policy/2".into(),
                    value_head: "stm-wdl/1".into(),
                    implementation_sha256: "b".repeat(64),
                    weights: PalsWeightIdentityV3::Untrained {
                        artifact: asset("weights/model.onnx"),
                        initialization_seed: 1,
                    },
                    frozen_epoch: 1,
                    backend: PalsModelBackendV3::RustCpu,
                    precision: PalsPrecisionV3::Fp32,
                    max_batch_width: 1,
                    exported_roles: BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
                },
                cpu: cpu(),
                cpu_r: PalsCpuRSelectionV3::Own,
                search: component(PALS_V4_SEARCH_SEMANTICS),
                runtime: component("single-owner/1"),
                pools: PalsPoolLimitsV3 {
                    states: 64,
                    line_chunks: 64,
                    situations: 64,
                    observations: 256,
                    tasks: 64,
                    role_states: 2,
                    memory_pages: 64,
                    queue_requests: 64,
                    host_bytes: 16 << 20,
                    device_bytes: 0,
                },
                requested_options: BTreeMap::from([("Ponder".into(), "false".into())]),
            },
            model_v2: PalsModelSpecificationV4 {
                profile: PalsModelProfileV4::FullLineInteractionV2,
                model_schema: PALS_V4_MODEL_SCHEMA.into(),
                model_semantics: PALS_V4_MODEL_SEMANTICS.into(),
                encoding_schema: PALS_V4_ENCODING_SCHEMA.into(),
                model_identity: "fixture-full-line-interaction/2".into(),
                model_semantics_sha256: "d".repeat(64),
                encoding_sha256: "e".repeat(64),
                max_records: 128,
                max_line_plies: 256,
                provenance: PalsModelProvenanceV4 {
                    purpose: PalsArtifactPurposeV4::ArenaCandidate,
                    training: PalsTrainingProvenanceV4::Untrained,
                },
            },
            policy_identity: PalsPolicyIdentityV4::expected_for_policies(&policies).unwrap(),
            policies,
        };
        let resources = PalsResourcePolicyV3 {
            cpu_threads: 2,
            cpu_affinity: vec![0, 1],
            memory_high_bytes: 32 << 20,
            memory_max_bytes: 64 << 20,
            swap_max_bytes: 0,
            requested_gpu: None,
            device_allocation_max_bytes: 0,
        };
        PalsRunManifestV4 {
            schema_version: 4,
            run_id: "v4-fixture".into(),
            pair_id: "v4-pair".into(),
            contract_revision: PALS_V4_CONTRACT_REVISION.into(),
            rules_profile: "standard-complete-history/1".into(),
            purpose: PalsRunPurposeV4::ArenaPilot,
            comparison: PalsComparisonV3::System,
            declared_changes: BTreeSet::from([PalsChangeAxisV3::Endpoint]),
            training_executed: false,
            engines: [
                PalsEngineV4::Pals(Box::new(p)),
                PalsEngineV4::OwnCpu(Box::new(PalsCpuEndpointV3 {
                    id: "cpu".into(),
                    binary: asset("bin/rovezero"),
                    source_commit: "c".repeat(40),
                    cpu: cpu(),
                    requested_options: BTreeMap::new(),
                })),
            ],
            resources: [resources.clone(), resources],
            pilot: PalsPilotV3 {
                position_command: "position startpos".into(),
                games: 2,
                white_order: ["pals".into(), "cpu".into()],
                base_ms: 120_000,
                increment_ms: 1_000,
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
            output_bytes_max: 2 << 30,
        }
    }
    fn observed_fixture<T>(value: T) -> PalsObservedV3<T> {
        PalsObservedV3::Observed {
            value,
            method: "independent-fixture-check".into(),
        }
    }
    fn pals_mut(m: &mut PalsRunManifestV4) -> &mut PalsEndpointV4 {
        let PalsEngineV4::Pals(p) = &mut m.engines[0] else {
            unreachable!()
        };
        p
    }
    fn endpoint_receipt(
        e: &PalsEngineV4,
        r: &PalsResourcePolicyV3,
        game: u32,
    ) -> PalsEndpointReceiptV4 {
        let shared = e.shared_engine();
        let native = matches!(e, PalsEngineV4::Pals(_));
        let base = PalsEndpointReceiptV3 {
            endpoint_id: e.id().into(),
            external_cpu_r: None,
            uci_ready_observed: true,
            search_failed_go_count: Some(0),
            pals_search_policy: None,
            options: shared
                .requested_options()
                .iter()
                .map(|(name, value)| {
                    (
                        name.clone(),
                        crate::PalsOptionReceiptV3 {
                            requested: value.clone(),
                            advertised_supported: true,
                            observed: observed_fixture(value.clone()),
                        },
                    )
                })
                .collect(),
            actual_affinity: observed_fixture(r.cpu_affinity.clone()),
            memory_peak_bytes: observed_fixture(1 << 20),
            gpu_device: PalsObservedV3::Unknown,
            vram_peak_bytes: PalsObservedV3::Unknown,
            nn_inputs_completed: if native { 2 } else { 0 },
            nn_calls_completed: if native { 2 } else { 0 },
            nn_inputs_consumed: if native { 2 } else { 0 },
            cached_evaluations_consumed: 0,
            proposer_tasks_completed: if native { 1 } else { 0 },
            critic_tasks_completed: if native { 1 } else { 0 },
            cpu_tasks_requested: 1,
            cpu_tasks_completed: 1,
            cpu_tasks_reused_consumed: 0,
            cpu_tasks_consumed: 1,
            cpu_nodes: 100,
            physical_state: if native {
                PalsPhysicalStateV3::Completed
            } else {
                PalsPhysicalStateV3::NotRequired
            },
            buffers_released: true,
            process_exited: true,
        };
        let mut o = PalsEndpointReceiptV4::with_unknown_checks(base);
        o.checks.artifacts = observed_fixture(
            endpoint_artifacts(e)
                .into_iter()
                .map(|a| {
                    (
                        a.path.clone(),
                        PalsVerifiedArtifactV4 {
                            sha256: a.sha256.clone(),
                            bytes: a.bytes,
                            pinned_handle_consumed: true,
                        },
                    )
                })
                .collect(),
        );
        o.checks.resource_limits = observed_fixture(r.clone());
        o.checks.physical_work_accounting = observed_fixture(true);
        if let PalsEngineV4::Pals(p) = e {
            o.checks.model = observed_fixture(PalsModelCheckV4 {
                model: p.base.model.clone(),
                specification: p.model_v2.clone(),
            });
            o.checks.policies = observed_fixture(PalsPolicyCheckV4 {
                search: p.base.search.clone(),
                policies: p.policies.clone(),
                identity: p.policy_identity.clone(),
            });
        }
        o.process_cleanup = observed_fixture(PalsProcessCleanupV4 {
            owner_identity: format!("fixture-owner-{game}-{}", e.id()),
            process_exited: true,
            exit_code: Some(0),
            exit_signal: None,
            child_processes_remaining: 0,
            stdout_drained: true,
            stderr_drained: true,
            stdout_bytes: 0,
            stderr_bytes: 0,
            elapsed_ms: 1,
            cleanup_complete: true,
            ownership_lost: false,
        });
        o
    }
    fn receipt(lock: &PalsInputLockV4) -> PalsRunReceiptV4 {
        let m = &lock.manifest;
        PalsRunReceiptV4 {
            schema_version: 4,
            domain: PALS_RECEIPT_V4_DOMAIN.into(),
            run_id: m.run_id.clone(),
            pair_id: m.pair_id.clone(),
            lock_sha256: lock.canonical_sha256.clone(),
            training_executed: false,
            training: PalsObservedV3::Unknown,
            wall_time_ms: 1,
            cleanup_time_ms: 2,
            output_bytes: observed_fixture(2),
            pair_eligible: m.purpose == PalsRunPurposeV4::ArenaPilot,
            failures: BTreeSet::new(),
            policy_failures: BTreeSet::new(),
            games: (0..2)
                .map(|game| PalsGameReceiptV4 {
                    game_index: game,
                    white_endpoint: m.pilot.white_order[game as usize].clone(),
                    black_endpoint: m.pilot.white_order[1 - game as usize].clone(),
                    base_ms: m.pilot.base_ms,
                    increment_ms: m.pilot.increment_ms,
                    plies: 0,
                    result: PalsResultV3::Draw,
                    termination: PalsTerminationV3::RulesDraw,
                    failed_endpoint: None,
                    engines: [
                        endpoint_receipt(&m.engines[0], &m.resources[0], game),
                        endpoint_receipt(&m.engines[1], &m.resources[1], game),
                    ],
                    pgn: asset(&format!("games/{game}.pgn")),
                })
                .collect(),
        }
    }

    #[test]
    fn v4_roundtrip_and_separate_domains_are_stable() {
        let m = manifest();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(PalsRunManifestV4::from_json(&json).unwrap(), m);
        let lock = m.lock().unwrap();
        assert_eq!(
            PalsInputLockV4::from_json(&lock.to_json().unwrap()).unwrap(),
            lock
        );
        let r = receipt(&lock);
        assert_eq!(
            PalsRunReceiptV4::from_json(&r.to_json(&lock).unwrap(), &lock).unwrap(),
            r
        );
        assert_ne!(PALS_MANIFEST_V4_DOMAIN, crate::PALS_MANIFEST_V3_DOMAIN);
        assert_ne!(PALS_RECEIPT_V4_DOMAIN, crate::PALS_RECEIPT_V3_DOMAIN);
        let mut tampered = lock.clone();
        pals_mut(&mut tampered.manifest).model_v2.profile = PalsModelProfileV4::FullLineV2;
        assert!(tampered.verify().is_err());
    }
    #[test]
    fn v4_rejects_unknown_duplicate_and_null_policy_fields() {
        let m = manifest();
        let mut json = serde_json::to_value(&m).unwrap();
        json["engines"][0]["configuration"]["policies"]["invented"] = true.into();
        assert!(PalsRunManifestV4::from_json(&json.to_string()).is_err());
        let json = serde_json::to_string(&m).unwrap();
        assert!(
            PalsRunManifestV4::from_json(&json.replacen(
                "\"schema_version\":4",
                "\"schema_version\":4,\"schema_version\":4",
                1
            ))
            .is_err()
        );
        let mut json = serde_json::to_value(&m).unwrap();
        json["engines"][0]["configuration"]["policies"]["cuda_warm"] = serde_json::Value::Null;
        assert!(PalsRunManifestV4::from_json(&json.to_string()).is_err());
        let mut json = serde_json::to_value(&m).unwrap();
        json["engines"][0]["configuration"]["policies"]
            .as_object_mut()
            .unwrap()
            .remove("cuda_warm");
        assert_eq!(PalsRunManifestV4::from_json(&json.to_string()).unwrap(), m);
        let lock = m.lock().unwrap();
        let mut json = serde_json::to_value(receipt(&lock)).unwrap();
        json["games"][0]["engines"][0]["checks"]["invented"] = 1.into();
        assert!(PalsRunReceiptV4::from_json(&json.to_string(), &lock).is_err());
        let missing_winner = serde_json::json!({ "kind": "rules_terminal", "state_sha256": "a".repeat(64), "perspective": "white" });
        assert!(serde_json::from_value::<PalsRecheckValueV4>(missing_winner.clone()).is_err());
        let mut draw = missing_winner;
        draw["winner"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<PalsRecheckValueV4>(draw).is_ok());
    }
    #[test]
    fn v4_rejects_unsupported_profile_resolver_policy_and_cuda_combinations() {
        let mut m = manifest();
        pals_mut(&mut m).model_v2.encoding_schema = "rovezero.pals-board-records.v1".into();
        assert!(m.validate().is_err());
        let mut m = manifest();
        pals_mut(&mut m).policies.checker = PalsCheckerPolicyV4::ExternalUci;
        assert!(m.validate().is_err());
        let mut m = manifest();
        pals_mut(&mut m).policies.iterative_repair =
            PalsIterativeRepairPolicyV4::Bounded(PalsIterativeRepairLimitsV4 {
                max_repairs_per_first_move: 3,
                max_pending_questions: 64,
            });
        assert!(m.validate().is_err());
        let mut m = manifest();
        pals_mut(&mut m).policies.cuda_warm =
            PalsCudaWarmPolicyV4::ApproxWarm(PalsCudaWarmLimitsV4 {
                capability: PALS_V4_CUDA_WARM_CAPABILITY.into(),
                max_leases: 1,
                device_bytes_max: 1,
            });
        assert!(m.validate().is_err());
        let mut m = manifest();
        pals_mut(&mut m).policy_identity.conditions_sha256[0] ^= 1;
        assert!(m.validate().is_err());
        let mut m = manifest();
        pals_mut(&mut m).policies.resolver_semantics_sha256 = "0".repeat(64);
        assert!(m.validate().is_err());
    }
    #[test]
    fn diagnostic_nonzero_checkpoint_is_valid_evidence_and_never_arena_eligible() {
        let mut m = manifest();
        let p = pals_mut(&mut m);
        p.base.model.weights = PalsWeightIdentityV3::Trained {
            artifact: asset("weights/model.onnx"),
            training_run_id: "diagnostic-run".into(),
        };
        p.model_v2.provenance = PalsModelProvenanceV4 {
            purpose: PalsArtifactPurposeV4::Diagnostic,
            training: PalsTrainingProvenanceV4::Nonzero {
                checkpoint_domain: PALS_V4_NONZERO_CHECKPOINT_DOMAIN.into(),
                training_run_id: "diagnostic-run".into(),
                completed_updates: 4,
                target_coverage: PalsTargetCoverageV4::DiagnosticFixture,
                target_coverage_sha256: "f".repeat(64),
            },
        };
        assert!(m.validate().is_err());
        m.purpose = PalsRunPurposeV4::Diagnostic;
        let lock = m.lock().unwrap();
        let mut r = receipt(&lock);
        assert!(!r.pair_eligible);
        r.validate_against(&lock).unwrap();
        r.pair_eligible = true;
        assert!(r.validate_against(&lock).is_err());
        let mut m = lock.manifest.clone();
        let PalsTrainingProvenanceV4::Nonzero {
            completed_updates, ..
        } = &mut pals_mut(&mut m).model_v2.provenance.training
        else {
            unreachable!()
        };
        *completed_updates = 0;
        assert!(m.validate().is_err());
    }
    #[test]
    fn v4_rejects_declared_archive_stack_queue_overquota() {
        let mut m = manifest();
        pals_mut(&mut m).policies.cold_archive =
            PalsColdArchivePolicyV4::Bounded(PalsColdArchiveLimitsV4 {
                game_bytes_max: PALS_V4_ARCHIVE_GAME_MAX_BYTES + 1,
                global_bytes_max: PALS_V4_ARCHIVE_GLOBAL_MAX_BYTES,
                index_entries_max: 64,
                index_bytes_max: 1024,
                load_bytes_max: 1024,
                load_deadline_max_ms: 100,
            });
        assert!(m.validate().is_err());
        let mut m = manifest();
        pals_mut(&mut m).policies.paused_stack =
            PalsPausedStackPolicyV4::Bounded(PalsPausedStackLimitsV4 {
                tokens_max: 2,
                bytes_max: PALS_V4_PAUSED_STACK_MAX_BYTES,
            });
        assert!(m.validate().is_err());
        let mut m = manifest();
        let p = pals_mut(&mut m);
        p.policies.recheck = PalsRecheckPolicyV4::FrozenWdl;
        p.policies.iterative_repair =
            PalsIterativeRepairPolicyV4::Bounded(PalsIterativeRepairLimitsV4 {
                max_repairs_per_first_move: 4,
                max_pending_questions: 65,
            });
        p.policy_identity = PalsPolicyIdentityV4::expected_for_policies(&p.policies).unwrap();
        p.base.search.semantic_id = p.policy_identity.search_identity.clone();
        assert!(m.validate().is_err());
    }
    #[test]
    fn unknown_completion_retains_buffers_and_cannot_be_success() {
        let lock = manifest().lock().unwrap();
        let mut r = receipt(&lock);
        r.games[0].engines[0].base.physical_state = PalsPhysicalStateV3::Unknown;
        assert!(r.validate_against(&lock).is_err());
        r.pair_eligible = false;
        r.games[0].engines[0].base.buffers_released = false;
        assert!(r.validate_against(&lock).is_err());
        r.failures
            .insert(PalsRunFailureV3::PhysicalCompletionUnknown);
        r.validate_against(&lock).unwrap();
        r.games[0].engines[0].base.buffers_released = true;
        assert!(r.validate_against(&lock).is_err());
    }
    #[test]
    fn actual_checks_and_physical_units_cannot_be_invented() {
        let lock = manifest().lock().unwrap();
        let mut r = receipt(&lock);
        r.games[0].engines[0].checks.policies = PalsObservedV3::Unknown;
        assert!(r.validate_against(&lock).is_err());
        r.pair_eligible = false;
        r.validate_against(&lock).unwrap();
        let mut r = receipt(&lock);
        r.games[0].engines[0].base.nn_inputs_consumed = 3;
        assert!(r.validate_against(&lock).is_err());
        let mut r = receipt(&lock);
        r.games[0].engines[0].base.cpu_tasks_reused_consumed = 2;
        assert!(r.validate_against(&lock).is_err());
        let mut r = receipt(&lock);
        r.games[0].engines[0].checks.resource_limits = PalsObservedV3::Observed {
            value: lock.manifest.resources[0].clone(),
            method: "readyok".into(),
        };
        assert!(r.validate_against(&lock).is_err());
    }
    fn repair_manifest() -> PalsRunManifestV4 {
        let mut m = manifest();
        let p = pals_mut(&mut m);
        p.policies.recheck = PalsRecheckPolicyV4::FrozenWdl;
        p.policies.iterative_repair =
            PalsIterativeRepairPolicyV4::Bounded(PalsIterativeRepairLimitsV4 {
                max_repairs_per_first_move: 3,
                max_pending_questions: 64,
            });
        p.policy_identity = PalsPolicyIdentityV4::expected_for_policies(&p.policies).unwrap();
        p.base.search.semantic_id = p.policy_identity.search_identity.clone();
        m
    }
    fn frozen_value(wdl: [f32; 3]) -> PalsRecheckValueV4 {
        PalsRecheckValueV4::FrozenWdl {
            model_sha256: "a".repeat(64),
            model_epoch: 1,
            encoding_sha256: "e".repeat(64),
            precision: "fp32".into(),
            context_revision: 1,
            input_sha256: "d".repeat(64),
            perspective: PalsPerspectiveV4::White,
            fresh: true,
            wdl,
        }
    }
    fn trace() -> PalsRepairTraceV4 {
        PalsRepairTraceV4 {
            root_generation: 1,
            first_move: "e2e4".into(),
            original_record: 10,
            record_id: 11,
            supersedes: Some(10),
            question_sha256: "f".repeat(64),
            evidence_revision: 1,
            repair_ordinal: 1,
            evidence_sha256: "b".repeat(64),
            recheck: observed_fixture(PalsRecheckObservationV4 {
                before: frozen_value([0.6, 0.3, 0.1]),
                after: frozen_value([0.1, 0.2, 0.7]),
                resolution: PalsRecheckResolutionV4::BeforePreferred,
                fresh_nn_inputs_charged: 2,
                actual_scope: None,
            }),
            actual_scope: None,
            iterative_repair_comparison: None,
        }
    }
    #[test]
    fn frozen_recheck_rejects_mixed_resolution_cached_values_and_broken_supersedes() {
        let lock = repair_manifest().lock().unwrap();
        let PalsEngineV4::Pals(p) = &lock.manifest.engines[0] else {
            unreachable!()
        };
        let mut t = trace();
        let PalsObservedV3::Observed { value: recheck, .. } = &mut t.recheck else {
            unreachable!()
        };
        recheck.validate_against(p).unwrap();
        recheck.after = PalsRecheckValueV4::Unknown;
        recheck.fresh_nn_inputs_charged = 1;
        assert!(recheck.validate_against(p).is_err());
        recheck.resolution = PalsRecheckResolutionV4::Unresolved;
        recheck.validate_against(p).unwrap();
        let PalsRecheckValueV4::FrozenWdl { fresh, .. } = &mut recheck.before else {
            unreachable!()
        };
        *fresh = false;
        assert!(recheck.validate_against(p).is_err());

        let mut r = receipt(&lock);
        for g in &mut r.games {
            let o = &mut g.engines[0];
            o.repair_traces = vec![trace()];
            o.repair_trace_total = observed_fixture(1);
            o.pending_questions_peak = observed_fixture(1);
            o.base.nn_inputs_completed = 4;
            o.base.nn_calls_completed = 4;
            o.base.nn_inputs_consumed = 4;
        }
        r.validate_against(&lock).unwrap();
        let mut second = trace();
        second.record_id = 12;
        second.supersedes = Some(11);
        second.repair_ordinal = 2;
        second.evidence_revision = 2;
        r.games[0].engines[0].repair_traces.push(second);
        r.games[0].engines[0].repair_trace_total = observed_fixture(2);
        r.validate_against(&lock).unwrap();
        r.games[0].engines[0].repair_traces[1].first_move = "d2d4".into();
        assert!(r.validate_against(&lock).is_err());
        r.games[0].engines[0].repair_traces[1].first_move = "e2e4".into();
        r.games[0].engines[0].repair_traces[1].evidence_revision = 1;
        assert!(r.validate_against(&lock).is_err());
        r.games[0].engines[0].repair_traces[1].evidence_revision = 2;
        r.games[0].engines[0].base.nn_inputs_completed = 3;
        r.games[0].engines[0].base.nn_calls_completed = 3;
        r.games[0].engines[0].base.nn_inputs_consumed = 3;
        assert!(r.validate_against(&lock).is_err());
    }
    #[test]
    fn archive_commit_and_integrity_must_precede_ram_release() {
        let limits = PalsColdArchiveLimitsV4 {
            game_bytes_max: 16,
            global_bytes_max: 32,
            index_entries_max: 4,
            index_bytes_max: 1024,
            load_bytes_max: 8,
            load_deadline_max_ms: 100,
        };
        let mut a = PalsArchiveObservationV4 {
            owner_id: 1,
            generation: 1,
            lifecycle: PalsArchiveLifecycleV4::Closed,
            committed_chunks: 1,
            integrity_verified_chunks: 1,
            committed_bytes: 8,
            integrity_verified_bytes: 8,
            ram_released_bytes: 8,
            pending_commit_bytes: 0,
            pending_buffers_retained: true,
            global_managed_bytes: 8,
            index_entries_peak: 1,
            index_bytes_peak: 16,
            pinned_entries_peak: 0,
            loads_requested: 1,
            loads_completed: 1,
            load_bytes_total: 8,
            load_bytes_peak: 8,
            load_elapsed_peak_ms: 1,
            owner_generation_checks: 1,
            quota_failures: 0,
            io_failures: 0,
            pin_saturation_failures: 0,
            cleanup_complete: true,
            actual_scope: None,
        };
        a.validate_against(&limits, true, &BTreeSet::new()).unwrap();
        a.integrity_verified_bytes = 0;
        assert!(
            a.validate_against(
                &limits,
                false,
                &BTreeSet::from([PalsPolicyFailureV4::ArchiveIo])
            )
            .is_err()
        );
        a.integrity_verified_bytes = 8;
        a.pending_commit_bytes = 1;
        a.pending_buffers_retained = false;
        a.lifecycle = PalsArchiveLifecycleV4::Open;
        assert!(
            a.validate_against(&limits, false, &BTreeSet::new())
                .is_err()
        );
        a.pending_commit_bytes = 0;
        a.lifecycle = PalsArchiveLifecycleV4::Closed;
        a.global_managed_bytes = 33;
        assert!(
            a.validate_against(&limits, false, &BTreeSet::new())
                .is_err()
        );
        a.validate_against(
            &limits,
            false,
            &BTreeSet::from([PalsPolicyFailureV4::ArchiveQuota]),
        )
        .unwrap();
    }
    #[test]
    fn paused_cpu_work_cannot_recount_or_reuse_stale_context() {
        let limits = PalsPausedStackLimitsV4 {
            tokens_max: 1,
            bytes_max: 1024,
        };
        let mut s = PalsPausedStackObservationV4 {
            tokens_created: 1,
            tokens_resumed: 1,
            tokens_invalidated: 0,
            tokens_retained: 0,
            tokens_peak: 1,
            bytes_peak: 64,
            stale_context_attempts: 1,
            stale_context_rejections: 1,
            replayed_consumed_work: 0,
            owner_released: true,
            actual_scope: None,
        };
        s.validate_against(&limits, true, &BTreeSet::new()).unwrap();
        s.replayed_consumed_work = 1;
        assert!(
            s.validate_against(
                &limits,
                false,
                &BTreeSet::from([PalsPolicyFailureV4::PausedStack])
            )
            .is_err()
        );
        s.replayed_consumed_work = 0;
        s.stale_context_rejections = 0;
        assert!(
            s.validate_against(&limits, false, &BTreeSet::new())
                .is_err()
        );
    }
    #[test]
    fn cuda_warm_unknown_completion_retains_device_owner() {
        let m = manifest();
        let mut o = endpoint_receipt(&m.engines[0], &m.resources[0], 0).base;
        o.physical_state = PalsPhysicalStateV3::Quarantined;
        o.buffers_released = false;
        let limits = PalsCudaWarmLimitsV4 {
            capability: PALS_V4_CUDA_WARM_CAPABILITY.into(),
            max_leases: 1,
            device_bytes_max: 64,
        };
        let mut w = PalsCudaWarmObservationV4 {
            capability: observed_fixture(PalsCudaWarmCapabilityV4 {
                semantic_id: PALS_V4_CUDA_WARM_CAPABILITY.into(),
                device_identity: "fixture-cuda-device".into(),
                available: true,
            }),
            leases_admitted: 1,
            leases_physically_completed: 0,
            leases_quarantined: 1,
            leases_peak: 1,
            device_bytes_peak: 32,
            accepted_seed_consumptions: 0,
            seed_context_checked_consumptions: 0,
            rejected_seed_contexts: 0,
            fresh_value_evaluations: 0,
            buffers_released: false,
            actual_scope: None,
        };
        let failures = BTreeSet::from([PalsPolicyFailureV4::CudaCompletionUnknown]);
        w.validate_against(&limits, &o, false, &failures).unwrap();
        w.buffers_released = true;
        assert!(w.validate_against(&limits, &o, false, &failures).is_err());
    }
    #[test]
    fn external_checker_unknown_proof_is_preserved_without_own_raw_substitution() {
        let mut m = manifest();
        let p = pals_mut(&mut m);
        let mut binary = asset("bin/stockfish");
        binary.source = "https://github.com/official-stockfish/Stockfish".into();
        binary.license = "GPL-3.0-or-later".into();
        p.base.cpu_r = PalsCpuRSelectionV3::ExternalUci(Box::new(crate::PalsExternalCpuRV3 {
            selection: crate::PalsExternalCpuRSelectionV3::StockfishEmbeddedNnue,
            profile: asset("profiles/stockfish.json"),
            profile_canonical_sha256: "f".repeat(64),
            binary,
            resolver: crate::PalsExternalCpuRResolverV3 {
                version: PALS_V4_MODEL_WDL_RESOLVER_SEMANTICS.into(),
                semantics_sha256: PalsResolverPolicyV4::ModelWdl.expected_semantics_sha256(),
            },
            policy: crate::PalsExternalCpuRPolicyV3 {
                resource_scope: crate::PalsExternalCpuRResourceScopeV3::InheritedParentCgroup,
                max_owners: 1,
                max_active_tasks: 1,
                max_process_leaders: 1,
                inherited_kernel_tasks_max: 128,
                threads_max: 2,
                hash_mib_max: 1,
                max_depth: 8,
                max_prefix_plies: 256,
                max_nodes_per_task: 10_000,
                handshake_max_ms: 100,
                task_wall_time_max_ms: 100,
                stop_grace_max_ms: 10,
                shutdown_grace_max_ms: 10,
                lifetime_output_bytes_max: 4096,
                line_bytes_max: 1024,
            },
        }));
        p.policies.checker = PalsCheckerPolicyV4::ExternalUci;
        assert!(m.validate().is_err()); // Actual external checker + own raw is unsupported.
        let p = pals_mut(&mut m);
        p.policies.resolver = PalsResolverPolicyV4::ModelWdl;
        p.policies.resolver_identity = component(PALS_V4_MODEL_WDL_RESOLVER_SEMANTICS);
        p.policies.resolver_semantics_sha256 =
            PalsResolverPolicyV4::ModelWdl.expected_semantics_sha256();
        let lock = m.lock().unwrap();
        let mut r = receipt(&lock);
        for g in &mut r.games {
            let b = &mut g.engines[0].base;
            b.cpu_tasks_requested = 0;
            b.cpu_tasks_completed = 0;
            b.cpu_tasks_reused_consumed = 0;
            b.cpu_tasks_consumed = 0;
            b.cpu_nodes = 0;
        }
        assert!(r.validate_against(&lock).is_err());
        r.pair_eligible = false;
        r.validate_against(&lock).unwrap(); // Foreign work remains unknown, never own raw or zero foreign nodes.
        r.games[0].engines[0].base.cpu_nodes = 1;
        assert!(r.validate_against(&lock).is_err());
    }

    // Actual owner scopes have different units and authority from the retained
    // historical synthetic fixtures above. Keep their focused regressions here.
    mod actual {
        use super::*;
        include!("pals_manifest_v4/followup_tests.rs");
    }
}
