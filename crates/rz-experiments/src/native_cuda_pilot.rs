//! Separate S0/S1 pilot domain. The historical V1/V2 A/A wire stays closed.
//! This reuses input ownership algorithms, never A/A or strength acceptance.

use crate::*;
use serde::{Deserialize, Serialize};

pub const CUDA_SEARCH_PILOT_V3_DOMAIN: &str = "rz-e-native-cuda-search-pilot-v3";
pub const CUDA_SEARCH_PILOT_V3_CANONICALIZATION: &str = "rz-e-native-cuda-search-pilot-json-v3";
pub const FASTCHESS_CLOCK_PATCH_SHA256: &str =
    "615b49e3a2e19fcd1ef3cd0fdb1a4640e665ae9f8055e7b75428e7b841323b5a";
pub const FASTCHESS_CLOCK_PATCH_BYTES: u64 = 4119;

/// An executor view only; persisted V1/V2 movetime JSON is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativePairClock {
    Movetime(NativeMovetimeV1),
    Game(NativeGameClockV3),
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeGameClockV3 {
    pub base_ms: u64,
    pub increment_ms: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativePilotStatistics {
    FixedPairedHoeffding95,
}

/// The complete cohort is immutable before any game, with no score-based stop,
/// retries, replacement openings or promotion. Engine losses remain recorded;
/// a broken provider/clock gate stops the pilot, rather than deleting a loss.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeSearchPilotProtocolV3 {
    pub pair_ordinal: u32,
    pub total_pairs: u32,
    pub opening_cohort: ArtifactRef,
    pub total_wall_ms: u64,
    pub statistics: NativePilotStatistics,
    pub claim_policy: ClaimPolicy,
    pub engine_failure: OutcomePolicy,
    pub cutoff: OutcomePolicy,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct NativeCudaProfileV3(pub NativeCudaProfileV2);
impl crate::native_cuda::profile_sealed::Sealed for NativeCudaProfileV3 {}
impl CudaLaunchProfile for NativeCudaProfileV3 {
    type Clock = NativeGameClockV3;
    const VERSION: u32 = 3;
    const DOMAIN: &'static str = CUDA_SEARCH_PILOT_V3_DOMAIN;
    const CANONICALIZATION: &'static str = CUDA_SEARCH_PILOT_V3_CANONICALIZATION;
    const PURPOSE: NativeCudaIntegrationPurpose = NativeCudaIntegrationPurpose::CudaSearchPilot;
    fn clock_view(clock: Self::Clock) -> NativePairClock {
        NativePairClock::Game(clock)
    }
    fn validate_clock(clock: Self::Clock, runtime_ms: u64) -> Result<(), ManifestError> {
        check(
            clock.base_ms == 30_000 && clock.increment_ms == 100 && runtime_ms >= 180_000,
            "clock",
            "pilot V3 is locked to 30s+0.1s with a bounded independent pair wall limit",
        )
    }
    fn validate_profile(&self) -> Result<(), ManifestError> {
        self.0.validate()?;
        check(
            self.0.search.simulations == 4096,
            "profile.search.simulations",
            "pilot V3 requires the explicit 4096 cap in both roles",
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
    fn pair_profiles_match(baseline: &Self, candidate: &Self) -> bool {
        if baseline.0.search.final_selection != NativeFinalSelectionV2::Visits
            || candidate.0.search.final_selection != NativeFinalSelectionV2::ExactTerminal
        {
            return false;
        }
        let mut normalized = candidate.clone();
        normalized.0.search.final_selection = NativeFinalSelectionV2::Visits;
        baseline == &normalized
    }
    fn validate_runner(runner: &ToolIdentity) -> Result<(), ManifestError> {
        let patch = runner.dirty_patch.as_ref().ok_or_else(|| {
            fail(
                "runner.dirty_patch",
                "pilot requires the separately pinned whole-engine clock patch",
            )
        })?;
        patch.validate()?;
        check(
            runner.dirty
                && runner.source_commit == "f618e34540f94f4719ad3817950618dabe441318"
                && runner.source_url == "https://github.com/Disservin/fastchess"
                && patch.sha256 == FASTCHESS_CLOCK_PATCH_SHA256
                && patch.bytes == FASTCHESS_CLOCK_PATCH_BYTES,
            "runner",
            "pilot requires the exact audited Fastchess source and clock patch",
        )
    }
    fn validate_protocol(
        protocol: Option<&NativeSearchPilotProtocolV3>,
        max_plies: u32,
    ) -> Result<(), ManifestError> {
        let p = protocol.ok_or_else(|| fail("pilot", "a pilot protocol is required"))?;
        p.opening_cohort.validate()?;
        check(
            p.total_pairs == 16
                && p.pair_ordinal < p.total_pairs
                && p.total_wall_ms == 7_200_000
                && max_plies == 256
                && p.opening_cohort.bytes <= 64 * 1024
                && p.claim_policy == ClaimPolicy::AutomaticAcceptance
                && p.engine_failure == OutcomePolicy::Loss
                && p.cutoff == OutcomePolicy::Incomplete,
            "pilot",
            "requires 16 pairs, 120min wall, 256 plies, automatic claims, retained losses and incomplete cutoffs",
        )
    }
}
fn fail(path: &str, message: &str) -> ManifestError {
    ManifestError::Validation(vec![Violation {
        path: path.into(),
        code: "UnsupportedSearchPilot",
        message: message.into(),
    }])
}
fn check(valid: bool, path: &str, message: &str) -> Result<(), ManifestError> {
    if valid {
        Ok(())
    } else {
        Err(fail(path, message))
    }
}

// Algorithms remain shared behind the sealed profile. The explicit aliases and
// separate wire domain prevent old integrations from accepting a mixed pair.
pub type CudaSearchPilotPairSpecV3 = CudaIntegrationPairSpec<NativeCudaProfileV3>;
pub type LockedCudaSearchPilotPairSpecV3 = LockedCudaIntegrationPairSpec<NativeCudaProfileV3>;
