//! Source-pinned startup transport for the adaptive UCI process.
//! A config is an executable view of one V5 endpoint, never a V4 conversion.
//! It contains only logical private-snapshot names; the launch owner supplies
//! the working directory and retains the actual read-only file descriptors.
use crate::{
    ArtifactRef, ManifestError, PalsEndpointReceiptV5, PalsEngineModeV5, PalsEngineV5,
    PalsExecutionWindowV5, PalsInputLockV5, PalsModelBackendV3, PalsPrecisionV3, decode_json,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod content_admission;
pub use content_admission::*;

pub const PALS_STARTUP_CONFIG_V5_DOMAIN: &str = "rz-pals-adaptive-startup-config-v5/1";
pub const PALS_NATIVE_STARTUP_V5_DOMAIN: &str = "rz-pals-native-startup-v5/1";
pub const PALS_NATIVE_TERMINATION_V5_DOMAIN: &str = "rz-pals-native-termination-v5/1";
pub const PALS_PREFLIGHT_V5_DOMAIN: &str = "rz-pals-adaptive-metadata-preflight-v5/1";
pub const PALS_MODEL_EVIDENCE_V5_DOMAIN: &str = "rz-pals-model-verification-v5/1";
pub const PALS_RESOURCE_EVIDENCE_V5_DOMAIN: &str = "rz-pals-resource-readback-v5/1";
pub const PALS_COORDINATOR_EVIDENCE_V5_DOMAIN: &str = "rz-pals-coordinator-observation-v5/1";
pub const PALS_NATIVE_EAGER_CUDA_PROFILE_V5: &str = "eager-cuda12-cudnn9-v1";
pub const PALS_CUDNN_IMPLICIT_EXHAUSTIVE_V5: &str = "legacy_implicit_exhaustive";
pub const PALS_CUDNN_EXPERIMENTAL_DEFAULT_V5: &str = "experimental_default";

/// Declared startup selections. A selected profile grants no actual loading,
/// GPU readiness, Warm capability or physical-completion authority. The native
/// owner must repeat selection validation against its loaded asset/profile.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNativeStartupOptionsV5 {
    #[serde(deserialize_with = "explicit_optional")]
    pub native_loading_profile: Option<String>,
    #[serde(deserialize_with = "explicit_optional")]
    pub cuda_device_id: Option<i32>,
    #[serde(deserialize_with = "explicit_optional")]
    pub cuda_arena_bytes: Option<u64>,
    pub cache_public_memory: bool,
    pub device_public_memory: bool,
    pub cudnn_conv_search: String,
    pub approximate_warm: bool,
    pub startup_ms: u64,
    pub drain_ms: u64,
}
// Explicit null is meaningful for CPU. A missing field must not receive
// serde's implicit Option::None default in this newly registered V5 transport.
fn explicit_optional<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}
impl PalsNativeStartupOptionsV5 {
    /// The V5 runtime component registers exactly these explicit selections.
    /// Optional CUDA fields use the literal `none`; decimal and bool spellings
    /// have one canonical form. Additional keys and hidden defaults are refused.
    pub fn canonical_runtime_options(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "native_loading_profile".into(),
                self.native_loading_profile
                    .clone()
                    .unwrap_or_else(|| "none".into()),
            ),
            (
                "cuda_device_id".into(),
                self.cuda_device_id
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "none".into()),
            ),
            (
                "cuda_arena_bytes".into(),
                self.cuda_arena_bytes
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "none".into()),
            ),
            (
                "cache_public_memory".into(),
                self.cache_public_memory.to_string(),
            ),
            (
                "device_public_memory".into(),
                self.device_public_memory.to_string(),
            ),
            ("cudnn_conv_search".into(), self.cudnn_conv_search.clone()),
            ("approximate_warm".into(), self.approximate_warm.to_string()),
            ("startup_ms".into(), self.startup_ms.to_string()),
            ("drain_ms".into(), self.drain_ms.to_string()),
        ])
    }
    pub fn validate_against(
        &self,
        endpoint: &PalsEngineV5,
        window: &PalsExecutionWindowV5,
    ) -> Result<(), ManifestError> {
        let fail = |message: &str| ManifestError::Integrity(message.into());
        if self.startup_ms == 0
            || self.startup_ms > window.wall_time_max_ms
            || self.drain_ms == 0
            || self.drain_ms > window.cleanup_max_ms
        {
            return Err(fail(
                "V5 native startup/drain must be finite and within the registered execution/cleanup window",
            ));
        }
        if endpoint.model.precision != PalsPrecisionV3::Fp32 || endpoint.model.tf32 {
            return Err(fail(
                "V5 native startup requires the registered FP32 and TF32-off model",
            ));
        }
        if self.device_public_memory {
            return Err(fail(
                "V5 virtual/selected review native startup supports host public memory; device-public layout is unsupported",
            ));
        }
        if self.approximate_warm && endpoint.mode == PalsEngineModeV5::Collection {
            return Err(fail(
                "V5 Collection requires Fresh input; ApproxWarm has a separate Deployment PC-only capability",
            ));
        }
        match endpoint.model.backend {
            PalsModelBackendV3::OrtCuda => {
                if self.native_loading_profile.as_deref() != Some(PALS_NATIVE_EAGER_CUDA_PROFILE_V5)
                    || !self.cuda_device_id.is_some_and(|id| id >= 0)
                    || !self
                        .cuda_arena_bytes
                        .is_some_and(|bytes| (1..=6u64 << 30).contains(&bytes))
                    || !matches!(
                        self.cudnn_conv_search.as_str(),
                        PALS_CUDNN_IMPLICIT_EXHAUSTIVE_V5 | PALS_CUDNN_EXPERIMENTAL_DEFAULT_V5
                    )
                {
                    return Err(fail(
                        "V5 CUDA native startup requires explicit eager CUDA12/cuDNN9 profile, device/arena and a registered host cuDNN selection",
                    ));
                }
            }
            PalsModelBackendV3::OrtCpu
            | PalsModelBackendV3::RustCpu
            | PalsModelBackendV3::DeterministicMock => {
                if self.native_loading_profile.is_some()
                    || self.cuda_device_id.is_some()
                    || self.cuda_arena_bytes.is_some()
                    || self.cudnn_conv_search != PALS_CUDNN_IMPLICIT_EXHAUSTIVE_V5
                    || (self.approximate_warm
                        && endpoint.model.backend != PalsModelBackendV3::OrtCpu)
                {
                    return Err(fail(
                        "V5 CPU startup cannot inherit CUDA loading/device/arena or CUDA cuDNN selection; CPU Warm requires its explicit ORT Deployment capability",
                    ));
                }
            }
        }
        if endpoint.runtime.options != self.canonical_runtime_options() {
            return Err(fail(
                "V5 startup options differ from exact registered runtime options; missing/extra/default keys are rejected",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsStartupAssetV5 {
    /// Exact artifact identity, independently verified before process admission.
    pub artifact: ArtifactRef,
    /// Relative to the private attempt cwd. No absolute source/user path.
    pub snapshot_path: String,
    pub purpose: PalsStartupAssetPurposeV5,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum PalsStartupAssetPurposeV5 {
    Weights,
    Provenance,
    ExportManifest,
    PublicGraph,
    SharedPcGraph,
    ProposerGraph,
    CriticGraph,
    ValidatorGraph,
    RuntimeLibrary,
    CudaLibrary,
    CudaManifest,
    RulesDescriptor,
    InitializationEvidence,
    ExternalBinary,
    ExternalProfile,
    NumericalAcceptance,
    NumericalRawEvidence,
    GenerationSpecification,
    InitializationCheckpointMetadata,
    CudaControlInventory,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsAdaptiveStartupConfigV5 {
    pub schema_version: u32,
    pub domain: String,
    pub semantic_lock_sha256: String,
    pub endpoint: PalsEngineV5,
    /// Exactly one owner-created role directory, not an arbitrary output path.
    pub runtime_directory: String,
    /// Required in V5; omission never selects an implicit native default.
    pub native_options: PalsNativeStartupOptionsV5,
    pub assets: Vec<PalsStartupAssetV5>,
    pub source_components: BTreeMap<String, String>,
    pub training_executed: bool,
}
impl PalsAdaptiveStartupConfigV5 {
    pub fn from_json(input: &str, lock: &PalsInputLockV5) -> Result<Self, ManifestError> {
        let config: Self = decode_json(input)?;
        config.validate_against(lock)?;
        Ok(config)
    }
    pub fn validate_against(&self, lock: &PalsInputLockV5) -> Result<(), ManifestError> {
        lock.verify()?;
        let fail = |s: &str| ManifestError::Integrity(s.into());
        if self.schema_version != 5
            || self.domain != PALS_STARTUP_CONFIG_V5_DOMAIN
            || self.semantic_lock_sha256 != lock.canonical_sha256
            || self.training_executed
            || self.source_components != lock.manifest.source_pins
            || !lock.manifest.engines.iter().any(|e| e == &self.endpoint)
            || !matches!(
                self.runtime_directory.as_str(),
                "baseline-runtime" | "candidate-runtime"
            )
        {
            return Err(fail(
                "V5 startup identity, source pins, endpoint or non-training transport differs",
            ));
        }
        self.native_options
            .validate_against(&self.endpoint, &lock.manifest.window)?;
        if self.assets.is_empty() || self.assets.len() > 96 {
            return Err(fail("startup asset count exceeds bound"));
        }
        let mut names = BTreeSet::new();
        let mut paths = BTreeMap::new();
        for a in &self.assets {
            a.artifact.validate()?;
            if !pals_private_snapshot_path_v5(&a.snapshot_path)
                || !names.insert(&a.snapshot_path)
                || paths.insert(&a.artifact.path, &a.artifact).is_some()
            {
                return Err(fail(
                    "startup asset has unsafe or duplicate private snapshot identity",
                ));
            }
        }
        let count = |p| self.assets.iter().filter(|a| a.purpose == p).count();
        use PalsStartupAssetPurposeV5 as P;
        if count(P::CudaControlInventory)
            != usize::from(self.endpoint.model.backend == PalsModelBackendV3::OrtCuda)
        {
            return Err(fail(
                "CUDA startup requires exactly one pinned control inventory; CPU cannot carry a CUDA control inventory",
            ));
        }
        for p in [
            P::Weights,
            P::ExportManifest,
            P::RuntimeLibrary,
            P::RulesDescriptor,
            P::NumericalAcceptance,
        ] {
            if count(p) != 1 {
                return Err(fail(
                    "startup lacks an exactly owned required model/runtime asset",
                ));
            }
        }
        if self.endpoint.model.artifact_domain == crate::PalsArtifactDomainV5::Untrained {
            if count(P::InitializationEvidence) != 1
                || count(P::InitializationCheckpointMetadata) != 1
                || count(P::GenerationSpecification) != 0
            {
                return Err(fail(
                    "G0 requires exact initialization evidence/metadata and cannot relabel a generation specification",
                ));
            }
        } else if self.endpoint.model.artifact_domain == crate::PalsArtifactDomainV5::RealGeneration
        {
            if count(P::GenerationSpecification) != 1
                || count(P::InitializationEvidence) != 0
                || count(P::InitializationCheckpointMetadata) != 0
            {
                return Err(fail(
                    "G1 requires its preregistered actual generation specification; G0 evidence cannot stand in for trained provenance",
                ));
            }
        }
        let compact = self.endpoint.model.profile == "virtual_review_v1";
        let shared = count(P::PublicGraph) == 1
            && count(P::SharedPcGraph) == 1
            && count(P::ProposerGraph) == 0
            && count(P::CriticGraph) == 0;
        let separate = count(P::PublicGraph) == 1
            && count(P::SharedPcGraph) == 0
            && count(P::ProposerGraph) == 1
            && count(P::CriticGraph) == 1;
        if (compact && !shared) || (!compact && !shared && !separate) {
            return Err(fail(
                "model graph capability does not match compact/shared or original legacy PC layout",
            ));
        }
        let collection = self.endpoint.mode == PalsEngineModeV5::Collection;
        if count(P::ValidatorGraph) != usize::from(collection) {
            return Err(fail(
                "Collection requires private V; Deployment cannot carry a V graph",
            ));
        }
        let weights = self
            .assets
            .iter()
            .find(|a| a.purpose == P::Weights)
            .unwrap();
        if weights.artifact != self.endpoint.model.weights {
            return Err(fail("startup weights differ from semantic endpoint"));
        }
        match &self.endpoint.model.provenance {
            Some(p)
                if count(P::Provenance) == 1
                    && self
                        .assets
                        .iter()
                        .any(|a| a.purpose == P::Provenance && &a.artifact == p) => {}
            None if count(P::Provenance) == 0 => {}
            _ => {
                return Err(fail(
                    "startup provenance differs from registered model domain",
                ));
            }
        }
        match &self.endpoint.cpu.external {
            Some(p)
                if count(P::ExternalBinary) == 1
                    && count(P::ExternalProfile) == 1
                    && self
                        .assets
                        .iter()
                        .any(|a| a.purpose == P::ExternalBinary && a.artifact == p.binary)
                    && self
                        .assets
                        .iter()
                        .any(|a| a.purpose == P::ExternalProfile && a.artifact == p.profile) => {}
            None if count(P::ExternalBinary) == 0 && count(P::ExternalProfile) == 0 => {}
            _ => {
                return Err(fail(
                    "startup CPU_T/CPU_R assets differ from selected endpoint",
                ));
            }
        }
        Ok(())
    }
    pub fn asset(
        &self,
        purpose: PalsStartupAssetPurposeV5,
    ) -> Result<&PalsStartupAssetV5, ManifestError> {
        let mut found = self.assets.iter().filter(|a| a.purpose == purpose);
        let first = found
            .next()
            .ok_or_else(|| ManifestError::Integrity("startup asset purpose missing".into()))?;
        if found.next().is_some() {
            return Err(ManifestError::Integrity(
                "startup asset purpose ambiguous".into(),
            ));
        }
        Ok(first)
    }
    pub fn canonical_bytes(&self, lock: &PalsInputLockV5) -> Result<Vec<u8>, ManifestError> {
        self.validate_against(lock)?;
        let mut v =
            serde_json::to_value(self).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        v.sort_all_objects();
        serde_json::to_vec(&v).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
}

/// Closed private namespace, including the fixed `inputs/` parent. Symbols
/// commonly used for credentials or private keys are rejected before any read.
pub fn pals_private_snapshot_path_v5(path: &str) -> bool {
    let parts: Vec<_> = path.split('/').collect();
    if !(2..=6).contains(&parts.len()) || parts[0] != "inputs" || path.len() > 512 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && p.len() <= 128
            && !p.starts_with('.')
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
            && !p.to_ascii_lowercase().contains("credential")
            && !p.to_ascii_lowercase().contains("secret")
            && !p.to_ascii_lowercase().contains("ssh-key")
            && !p.to_ascii_lowercase().contains("service-account")
            && !p.to_ascii_lowercase().contains("service_account")
            && !p.to_ascii_lowercase().contains("api-key")
            && !p.to_ascii_lowercase().contains("api_key")
            && !p.to_ascii_lowercase().starts_with("id_rsa")
            && !p.to_ascii_lowercase().starts_with("id_ed25519")
    })
}

/// Producer records retain observations separately from startup declarations.
/// `readyok` is never translated into any of these observed fields.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsAdaptiveStartupRecordV5 {
    pub schema_version: u32,
    pub domain: String,
    pub lock_sha256: String,
    pub startup_config_sha256: String,
    pub endpoint_id: String,
    pub process_id: u32,
    pub process_run_id: String,
    pub source_commit: String,
    pub component_source_pins: BTreeMap<String, String>,
    pub observation: PalsEndpointReceiptV5,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsAdaptiveTerminationRecordV5 {
    pub schema_version: u32,
    pub domain: String,
    pub lock_sha256: String,
    pub startup_config_sha256: String,
    pub endpoint_id: String,
    pub process_id: u32,
    pub process_run_id: String,
    pub observation: PalsEndpointReceiptV5,
    pub admission_closed: bool,
    pub question_authority_revoked: bool,
    pub cpu_workers_joined: bool,
    pub gpu_physical_drain_known: bool,
    pub archive_closed: bool,
    pub training_executed: bool,
}

/// Identification/readiness probes live outside actual game-session evidence.
/// A probe may provide an explicitly labelled Rules-only legal stop witness;
/// it cannot start model sessions, CPU jobs, helper engines or training.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsAdaptivePreflightRecordV5 {
    pub schema_version: u32,
    pub domain: String,
    pub lock_sha256: String,
    pub startup_config_sha256: String,
    pub endpoint_id: String,
    pub process_id: u32,
    pub process_run_id: String,
    pub model_sessions_created: u64,
    pub role_requests_submitted: u64,
    pub cpu_jobs_submitted: u64,
    pub helper_processes_created: u64,
    pub training_executed: bool,
    pub rules_only_stop_witness: bool,
}

/// The single self-referencing evidence ArtifactRef is excluded from the
/// payload. All other model assets and observations remain exact typed data.
/// These pure builders do not create observation authority or write files.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsObservationEvidenceEnvelopeV5 {
    pub domain: String,
    pub endpoint_id: String,
    pub process_id: u32,
    pub process_run_id: String,
    pub observation_without_evidence: serde_json::Value,
}
fn observation_payload<T: Serialize>(observation: &T) -> Result<serde_json::Value, ManifestError> {
    let mut value = serde_json::to_value(observation)
        .map_err(|_| ManifestError::Integrity("V5 observation payload encode failed".into()))?;
    let map = value
        .as_object_mut()
        .ok_or_else(|| ManifestError::Integrity("V5 typed observation is not an object".into()))?;
    if map.remove("evidence").is_none() {
        return Err(ManifestError::Integrity(
            "V5 typed observation has no evidence field".into(),
        ));
    }
    Ok(value)
}
fn evidence_envelope<T: Serialize>(
    domain: &str,
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    observation: &T,
) -> Result<PalsObservationEvidenceEnvelopeV5, ManifestError> {
    if endpoint_id.is_empty()
        || endpoint_id.len() > 128
        || process_id == 0
        || process_id > i32::MAX as u32
        || process_run_id != format!("native-process-{process_id}")
    {
        return Err(ManifestError::Integrity(
            "V5 evidence process owner identity invalid".into(),
        ));
    }
    Ok(PalsObservationEvidenceEnvelopeV5 {
        domain: domain.into(),
        endpoint_id: endpoint_id.into(),
        process_id,
        process_run_id: process_run_id.into(),
        observation_without_evidence: observation_payload(observation)?,
    })
}
fn evidence_bytes<T: Serialize>(
    domain: &str,
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    observation: &T,
) -> Result<Vec<u8>, ManifestError> {
    serde_json::to_vec(&evidence_envelope(
        domain,
        endpoint_id,
        process_id,
        process_run_id,
        observation,
    )?)
    .map_err(|_| ManifestError::Integrity("V5 evidence envelope encode failed".into()))
}
pub fn pals_model_verification_evidence_bytes_v5(
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    value: &crate::PalsModelVerificationV5,
) -> Result<Vec<u8>, ManifestError> {
    evidence_bytes(
        PALS_MODEL_EVIDENCE_V5_DOMAIN,
        endpoint_id,
        process_id,
        process_run_id,
        value,
    )
}
pub fn pals_resource_readback_evidence_bytes_v5(
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    value: &crate::PalsResourceReadbackV5,
) -> Result<Vec<u8>, ManifestError> {
    evidence_bytes(
        PALS_RESOURCE_EVIDENCE_V5_DOMAIN,
        endpoint_id,
        process_id,
        process_run_id,
        value,
    )
}
pub fn pals_coordinator_evidence_bytes_v5(
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    value: &crate::PalsCoordinatorObservationV5,
) -> Result<Vec<u8>, ManifestError> {
    evidence_bytes(
        PALS_COORDINATOR_EVIDENCE_V5_DOMAIN,
        endpoint_id,
        process_id,
        process_run_id,
        value,
    )
}
fn validate_evidence<T: Serialize>(
    bytes: &[u8],
    domain: &str,
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    observation: &T,
) -> Result<(), ManifestError> {
    if bytes.is_empty() || bytes.len() > 128 << 10 {
        return Err(ManifestError::Integrity(
            "V5 evidence envelope exceeds bounded metadata".into(),
        ));
    }
    let input = std::str::from_utf8(bytes)
        .map_err(|_| ManifestError::Integrity("V5 evidence envelope is not UTF8".into()))?;
    let actual: PalsObservationEvidenceEnvelopeV5 = decode_json(input)?;
    if actual != evidence_envelope(domain, endpoint_id, process_id, process_run_id, observation)? {
        return Err(ManifestError::Integrity(
            "V5 evidence payload does not match exact typed observation/process owner".into(),
        ));
    }
    Ok(())
}
pub fn validate_pals_model_verification_evidence_v5(
    bytes: &[u8],
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    value: &crate::PalsModelVerificationV5,
) -> Result<(), ManifestError> {
    validate_evidence(
        bytes,
        PALS_MODEL_EVIDENCE_V5_DOMAIN,
        endpoint_id,
        process_id,
        process_run_id,
        value,
    )
}
pub fn validate_pals_resource_readback_evidence_v5(
    bytes: &[u8],
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    value: &crate::PalsResourceReadbackV5,
) -> Result<(), ManifestError> {
    validate_evidence(
        bytes,
        PALS_RESOURCE_EVIDENCE_V5_DOMAIN,
        endpoint_id,
        process_id,
        process_run_id,
        value,
    )
}
pub fn validate_pals_coordinator_evidence_v5(
    bytes: &[u8],
    endpoint_id: &str,
    process_id: u32,
    process_run_id: &str,
    value: &crate::PalsCoordinatorObservationV5,
) -> Result<(), ManifestError> {
    validate_evidence(
        bytes,
        PALS_COORDINATOR_EVIDENCE_V5_DOMAIN,
        endpoint_id,
        process_id,
        process_run_id,
        value,
    )
}

#[cfg(test)]
mod tests;
