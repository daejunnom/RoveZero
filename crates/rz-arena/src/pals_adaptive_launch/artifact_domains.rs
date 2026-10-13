//! Arena compatibility boundary for the shared producer-domain metadata verifier.
//!
//! The verifier belongs to `rz-experiments` so startup and arena admit the same
//! metadata meanings. This wrapper preserves the original callback/result shape;
//! its public summary is not a private content-admission token and cannot grant
//! initializer, neural, numerical, GPU, or arena acceptance.
use crate::ArenaError;
use rz_experiments::{ArtifactRef, ManifestError, PalsAdaptiveStartupConfigV5};

pub use rz_experiments::PalsArtifactDomainAdmissionV5;

/// Retain the arena signature while delegating all domain checks to one owner.
/// Small registered metadata reads remain the caller's responsibility; reader
/// failures propagate as artifact failures and never become a metadata success.
pub fn validate_pals_artifact_domain_v5(
    config: &PalsAdaptiveStartupConfigV5,
    export_bytes: &[u8],
    mut read_artifact: impl FnMut(&ArtifactRef, usize) -> Result<Vec<u8>, ArenaError>,
) -> Result<PalsArtifactDomainAdmissionV5, ArenaError> {
    rz_experiments::validate_pals_artifact_domain_v5(config, export_bytes, |artifact, maximum| {
        read_artifact(artifact, maximum)
            .map_err(|error| ManifestError::Artifact(format!("artifact reader: {error}")))
    })
    .map_err(ArenaError::from)
}
