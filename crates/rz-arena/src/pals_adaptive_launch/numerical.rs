//! Arena compatibility facade for the shared read-only numerical admission.
//!
//! Captured tensors, strict Fresh B1 comparison and their fixture tests belong
//! to the startup content admission shared by native launch and arena. This
//! module preserves arena callers' error type without cloning that verifier.
use crate::ArenaError;
use rz_eval::pals_model::PalsRole;
#[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
use rz_experiments::ManifestError;
use rz_experiments::{ArtifactRef, PalsAdaptiveStartupConfigV5};

pub use rz_experiments::{
    PALS_NUMERICAL_ACCEPTANCE_V5_DOMAIN, PALS_NUMERICAL_RAW_V5_DOMAIN,
    PALS_NUMERICAL_TENSOR_V5_DOMAIN, PalsNumericalAcceptanceV5, PalsNumericalAdmissionV5,
    PalsNumericalBindingsV5, PalsNumericalDtypeV5, PalsNumericalInputV5, PalsNumericalInvocationV5,
    PalsNumericalProducerV5, PalsNumericalRawCaseV5, PalsNumericalRawRunV5, PalsNumericalRunPairV5,
    PalsNumericalTensorV5,
};

/// Arena retains its explicit ONNX admission feature boundary. The actual
/// verifier performs no model load or inference, and callback failures retain
/// their diagnostic text in the shared artifact error domain.
pub fn validate_pals_numerical_acceptance_v5(
    config: &PalsAdaptiveStartupConfigV5,
    acceptance_bytes: &[u8],
    mut read_artifact: impl FnMut(&ArtifactRef, usize) -> Result<Vec<u8>, ArenaError>,
) -> Result<PalsNumericalAdmissionV5, ArenaError> {
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    {
        rz_experiments::validate_pals_numerical_acceptance_v5(
            config,
            acceptance_bytes,
            |artifact, limit| {
                read_artifact(artifact, limit)
                    .map_err(|error| ManifestError::Artifact(error.to_string()))
            },
        )
        .map_err(ArenaError::from)
    }
    #[cfg(not(any(feature = "pals-collection-onnx", feature = "native-cuda")))]
    {
        let _ = (config, acceptance_bytes, &mut read_artifact);
        Err(ArenaError::Integrity(
            "numerical native preflight requires the explicit ONNX feature".into(),
        ))
    }
}

/// Pure captured-byte helper; preserves the existing arena signature.
pub fn pals_numerical_input_tensors_v5(
    input: &PalsNumericalInputV5,
) -> Result<Vec<PalsNumericalTensorV5>, ArenaError> {
    rz_experiments::pals_numerical_input_tensors_v5(input).map_err(ArenaError::from)
}

/// Stable framing helper; preserves the existing arena signature.
pub fn pals_numerical_tensor_digest_v5(
    tensors: &[PalsNumericalTensorV5],
    role: PalsRole,
) -> Result<String, ArenaError> {
    rz_experiments::pals_numerical_tensor_digest_v5(tensors, role).map_err(ArenaError::from)
}
