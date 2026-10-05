//! Explicit BT4 integration profile. V1 remains a closed Maia declaration.
//! Shared locking/resource code is parameterized by a sealed profile, not a
//! deserialization fallback or an expanded V1 limit.

use crate::*;
use serde::{Deserialize, Serialize};

pub const CUDA_NATIVE_LAUNCH_V2_DOMAIN: &str = "rz-e-native-cuda-integration-pair-v2";
pub const CUDA_NATIVE_LAUNCH_V2_CANONICALIZATION: &str = "rz-e-native-cuda-json-v2";
pub const BT4_NATIVE_SOURCE_SHA256: &str =
    "e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618";
pub const BT4_NATIVE_SOURCE_BYTES: u64 = 382_645_315;
pub const BT4_NATIVE_ONNX_LIMIT: u64 = 768 * 1024 * 1024;
pub const BT4_NATIVE_ARENA_BYTES: u64 = 3 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeCudaModelV2 {
    Bt4It332,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeFinalSelectionV2 {
    Visits,
    ExactTerminal,
}
impl NativeFinalSelectionV2 {
    pub const fn cli(self) -> &'static str {
        match self {
            Self::Visits => "visits",
            Self::ExactTerminal => "exact-terminal",
        }
    }
}

/// This integration gate remains A/A: both roles must have equal profiles.
/// S0/S1 may each be exercised, but different selection policies require a
/// separate comparison protocol. Policy temperature is fixed at 1.0 by C.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaSearchV2 {
    pub simulations: u64,
    pub final_selection: NativeFinalSelectionV2,
    pub policy_temperature_milli: u32,
    pub raw_cache: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaProfileV2 {
    pub model: NativeCudaModelV2,
    /// Same provider fields with a V2-specific 3GiB arena validation.
    pub runtime: NativeCudaProfileV1,
    pub search: NativeCudaSearchV2,
}
impl NativeCudaProfileV2 {
    pub fn fixed(backend: String, encoding: String) -> Self {
        let mut runtime = NativeCudaProfileV1::fixed(backend, encoding);
        runtime.arena_bytes = BT4_NATIVE_ARENA_BYTES;
        Self {
            model: NativeCudaModelV2::Bt4It332,
            runtime,
            search: NativeCudaSearchV2 {
                simulations: 4096,
                final_selection: NativeFinalSelectionV2::Visits,
                policy_temperature_milli: 1000,
                raw_cache: false,
            },
        }
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.runtime.validate_arena(BT4_NATIVE_ARENA_BYTES)?;
        if !(1..=4096).contains(&self.search.simulations)
            || self.search.policy_temperature_milli != 1000
            || self.search.raw_cache
        {
            return Err(ManifestError::Validation(vec![Violation {
                path: "profile.search".into(),
                code: "UnsupportedNativeSearch",
                message: "requires 1..4096 simulations, policy temperature 1.0 and raw cache off"
                    .into(),
            }]));
        }
        Ok(())
    }
}

impl crate::native_cuda::profile_sealed::Sealed for NativeCudaProfileV2 {}
impl CudaLaunchProfile for NativeCudaProfileV2 {
    const VERSION: u32 = 2;
    const DOMAIN: &'static str = CUDA_NATIVE_LAUNCH_V2_DOMAIN;
    const CANONICALIZATION: &'static str = CUDA_NATIVE_LAUNCH_V2_CANONICALIZATION;
    fn validate_profile(&self) -> Result<(), ManifestError> {
        self.validate()
    }
    fn runtime_profile(&self) -> &NativeCudaProfileV1 {
        &self.runtime
    }
    fn search_options(&self) -> Option<NativeCudaSearchV2> {
        Some(self.search)
    }
    fn artifact_limit(role: NativeArtifactRole) -> u64 {
        match role {
            NativeArtifactRole::SourceWeights => BT4_NATIVE_SOURCE_BYTES,
            NativeArtifactRole::Onnx => BT4_NATIVE_ONNX_LIMIT,
            NativeArtifactRole::OrtLibrary => CUDA_BUNDLE_FILE_MAX_BYTES,
            other => other.byte_limit(),
        }
    }
    fn validate_model(&self, source: &ArtifactRef) -> Result<(), ManifestError> {
        if source.sha256 != BT4_NATIVE_SOURCE_SHA256 || source.bytes != BT4_NATIVE_SOURCE_BYTES {
            return Err(ManifestError::Validation(vec![Violation {
                path: "artifacts.source_weights".into(),
                code: "NativeModelIdentityMismatch",
                message: "V2 requires the selected BT4-it332 source bytes and digest".into(),
            }]));
        }
        Ok(())
    }
}
pub type CudaNativeLaunchSpecV2 = CudaNativeLaunchSpec<NativeCudaProfileV2>;
pub type CudaIntegrationPairSpecV2 = CudaIntegrationPairSpec<NativeCudaProfileV2>;
pub type LockedCudaIntegrationPairSpecV2 = LockedCudaIntegrationPairSpec<NativeCudaProfileV2>;
