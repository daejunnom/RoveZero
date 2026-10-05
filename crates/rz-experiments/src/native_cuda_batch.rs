//! Separate S batch experiment. Existing B1 integration/pilot profiles stay closed.
use crate::*;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaBatchProfileV4 {
    pub runtime: NativeCudaProfileV1,
    pub search: NativeCudaSearchV2,
    pub max_batch_wait_us: u64,
}
impl crate::native_cuda::profile_sealed::Sealed for NativeCudaBatchProfileV4 {}
impl CudaLaunchProfile for NativeCudaBatchProfileV4 {
    type Clock = NativeGameClockV3;
    const VERSION: u32 = 4;
    const DOMAIN: &'static str = "rovezero-cuda-batch-pilot-v4";
    const CANONICALIZATION: &'static str = "sorted-json-cuda-batch-pilot-v4";
    const PURPOSE: NativeCudaIntegrationPurpose = NativeCudaIntegrationPurpose::CudaSearchPilot;
    fn clock_view(clock: Self::Clock) -> NativePairClock {
        NativePairClock::Game(clock)
    }
    fn validate_clock(clock: Self::Clock, runtime_ms: u64) -> Result<(), ManifestError> {
        check(
            clock
                == NativeGameClockV3 {
                    base_ms: 120_000,
                    increment_ms: 1_000,
                }
                && runtime_ms == 900_000,
            "clock",
            "batch pilot requires 120+1 Fischer and 15min pair wall; wall cutoff remains incomplete",
        )
    }
    fn validate_profile(&self) -> Result<(), ManifestError> {
        self.runtime
            .validate_batch_arena(3 * 1024 * 1024 * 1024, self.runtime.batch_size)?;
        let mut fixed = self.runtime.clone();
        fixed.batch_size = 1;
        NativeCudaProfileV2 {
            model: NativeCudaModelV2::Bt4It332,
            runtime: fixed,
            search: self.search,
        }
        .validate()?;
        check(
            self.search.simulations == 4096
                && self.search.final_selection == NativeFinalSelectionV2::Visits
                && self.max_batch_wait_us == if self.runtime.batch_size > 1 { 200 } else { 0 },
            "profile",
            "batch pilot requires visits/4096 and 200us wait (B1 zero wait)",
        )
    }
    fn runtime_profile(&self) -> &NativeCudaProfileV1 {
        &self.runtime
    }
    fn search_options(&self) -> Option<NativeCudaSearchV2> {
        Some(self.search)
    }
    fn max_batch(&self) -> usize {
        self.runtime.batch_size as usize
    }
    fn artifact_limit(role: NativeArtifactRole) -> u64 {
        NativeCudaProfileV2::artifact_limit(role)
    }
    fn validate_model(&self, source: &ArtifactRef) -> Result<(), ManifestError> {
        NativeCudaProfileV2 {
            model: NativeCudaModelV2::Bt4It332,
            runtime: self.runtime.clone(),
            search: self.search,
        }
        .validate_model(source)
    }
    fn validate_runner(runner: &ToolIdentity) -> Result<(), ManifestError> {
        NativeCudaProfileV3::validate_runner(runner)
    }
    fn validate_protocol(
        protocol: Option<&NativeSearchPilotProtocolV3>,
        max_plies: u32,
        _white: [NativeEngineRole; 2],
    ) -> Result<(), ManifestError> {
        check(
            protocol.is_none() && max_plies == 256,
            "pilot",
            "batch pilot is exactly one pair, max256 ply, no V3 cohort or Elo promotion",
        )
    }
    fn pair_profiles_match(baseline: &Self, candidate: &Self) -> bool {
        if baseline.runtime.batch_size != 1 || candidate.runtime.batch_size != 4 {
            return false;
        }
        let mut normalized = candidate.clone();
        normalized.runtime.batch_size = 1;
        normalized.runtime.expected_backend_sha256 =
            baseline.runtime.expected_backend_sha256.clone();
        normalized.max_batch_wait_us = 0;
        &normalized == baseline
    }
}
fn check(valid: bool, path: &str, message: &str) -> Result<(), ManifestError> {
    if valid {
        Ok(())
    } else {
        Err(ManifestError::Validation(vec![Violation {
            path: path.into(),
            code: "UnsupportedBatchPilot",
            message: message.into(),
        }]))
    }
}
pub type CudaBatchPilotPairSpecV4 = CudaIntegrationPairSpec<NativeCudaBatchProfileV4>;
pub type LockedCudaBatchPilotPairSpecV4 = LockedCudaIntegrationPairSpec<NativeCudaBatchProfileV4>;

/// B1 A/A whole-clock memory check; no batch journal and no S1 selection policy.
/// Existing V1/V2 movetime and V3 visits/exact-terminal domains are unchanged.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct NativeCudaB1ClockProfileV5(pub NativeCudaProfileV2);
impl crate::native_cuda::profile_sealed::Sealed for NativeCudaB1ClockProfileV5 {}
impl CudaLaunchProfile for NativeCudaB1ClockProfileV5 {
    type Clock = NativeGameClockV3;
    const VERSION: u32 = 5;
    const DOMAIN: &'static str = "rovezero-bt4-b1-clock-memory-aa-v5";
    const CANONICALIZATION: &'static str = "sorted-json-bt4-b1-clock-memory-aa-v5";
    fn clock_view(clock: Self::Clock) -> NativePairClock {
        NativePairClock::Game(clock)
    }
    fn validate_clock(clock: Self::Clock, runtime_ms: u64) -> Result<(), ManifestError> {
        NativeCudaBatchProfileV4::validate_clock(clock, runtime_ms)
    }
    fn validate_profile(&self) -> Result<(), ManifestError> {
        self.0.validate()?;
        check(
            self.0.search.simulations == 4096
                && self.0.search.final_selection == NativeFinalSelectionV2::Visits,
            "profile",
            "B1 memory A/A requires visits/4096",
        )
    }
    fn runtime_profile(&self) -> &NativeCudaProfileV1 {
        &self.0.runtime
    }
    fn search_options(&self) -> Option<NativeCudaSearchV2> {
        Some(self.0.search)
    }
    fn artifact_limit(role: NativeArtifactRole) -> u64 {
        NativeCudaProfileV2::artifact_limit(role)
    }
    fn validate_model(&self, source: &ArtifactRef) -> Result<(), ManifestError> {
        self.0.validate_model(source)
    }
    fn validate_runner(runner: &ToolIdentity) -> Result<(), ManifestError> {
        NativeCudaProfileV3::validate_runner(runner)
    }
    fn validate_protocol(
        protocol: Option<&NativeSearchPilotProtocolV3>,
        max_plies: u32,
        white: [NativeEngineRole; 2],
    ) -> Result<(), ManifestError> {
        NativeCudaBatchProfileV4::validate_protocol(protocol, max_plies, white)
    }
}
pub type CudaB1ClockPairSpecV5 = CudaIntegrationPairSpec<NativeCudaB1ClockProfileV5>;
pub type LockedCudaB1ClockPairSpecV5 = LockedCudaIntegrationPairSpec<NativeCudaB1ClockProfileV5>;

#[cfg(test)]
mod tests {
    use super::*;
    fn batch_profile(width: u32) -> NativeCudaBatchProfileV4 {
        let fixed = NativeCudaProfileV2::fixed("a".repeat(64), "b".repeat(64));
        let mut runtime = fixed.runtime;
        runtime.batch_size = width;
        NativeCudaBatchProfileV4 {
            runtime,
            search: fixed.search,
            max_batch_wait_us: if width > 1 { 200 } else { 0 },
        }
    }
    #[test]
    fn pair_can_change_only_batch_width_and_corresponding_identity() {
        let baseline = batch_profile(1);
        let mut candidate = batch_profile(4);
        candidate.runtime.expected_backend_sha256 = "c".repeat(64);
        assert!(baseline.validate_profile().is_ok());
        assert!(candidate.validate_profile().is_ok());
        assert!(NativeCudaBatchProfileV4::pair_profiles_match(
            &baseline, &candidate
        ));
        candidate.search.final_selection = NativeFinalSelectionV2::ExactTerminal;
        assert!(!NativeCudaBatchProfileV4::pair_profiles_match(
            &baseline, &candidate
        ));
        assert!(candidate.validate_profile().is_err());
        assert!(batch_profile(0).validate_profile().is_err());
        assert!(batch_profile(17).validate_profile().is_err());
    }
    #[test]
    fn memory_aa_keeps_b1_and_visits_without_changing_v2() {
        let original = NativeCudaProfileV2::fixed("a".repeat(64), "b".repeat(64));
        assert!(
            NativeCudaB1ClockProfileV5(original.clone())
                .validate_profile()
                .is_ok()
        );
        let mut changed = original;
        changed.search.final_selection = NativeFinalSelectionV2::ExactTerminal;
        assert!(changed.validate().is_ok());
        assert!(
            NativeCudaB1ClockProfileV5(changed)
                .validate_profile()
                .is_err()
        );
    }
    #[test]
    fn b1_stays_closed_and_batch_policy_clock_are_separately_locked() {
        let mut runtime = NativeCudaProfileV1::fixed("a".repeat(64), "b".repeat(64));
        runtime.arena_bytes = 3 * 1024 * 1024 * 1024;
        runtime.batch_size = 4;
        assert!(runtime.validate().is_err());
        let mut search = NativeCudaProfileV2::fixed("a".repeat(64), "b".repeat(64)).search;
        search.simulations = 128;
        let profile = NativeCudaBatchProfileV4 {
            runtime,
            search,
            max_batch_wait_us: 200,
        };
        assert!(
            NativeCudaBatchProfileV4::validate_clock(
                NativeGameClockV3 {
                    base_ms: 120_000,
                    increment_ms: 1_000
                },
                900_000
            )
            .is_ok()
        );
        assert!(
            NativeCudaBatchProfileV4::validate_clock(
                NativeGameClockV3 {
                    base_ms: 120_000,
                    increment_ms: 1_000
                },
                899_999
            )
            .is_err()
        );
        assert!(profile.validate_profile().is_err()); // Default 128 is not the BT4 pilot cap.
    }
}
