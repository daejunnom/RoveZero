//! Distinct actual factory snapshots retain their original model identities.
//! This sum does not convert the V2 ABI, loaded evidence, or startup probe scope
//! into the compact model's namespace or numerical acceptance.
use crate::pals_native::startup_legacy::NativeLegacySelectedStartupSnapshotV5;
use crate::pals_native::startup_virtual::NativeVirtualReviewStartupSnapshotV5;
use rz_search::pals::engine::RoleError;
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::value::ModelValueIdentity;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
#[allow(
    clippy::large_enum_variant,
    reason = "the two bounded factory snapshots retain their original inline metadata without a startup allocation"
)]
pub(crate) enum ActualStartupSnapshotV5 {
    Virtual(NativeVirtualReviewStartupSnapshotV5),
    Legacy(NativeLegacySelectedStartupSnapshotV5),
}
impl ActualStartupSnapshotV5 {
    pub(crate) fn process_epoch(&self) -> u64 {
        match self {
            Self::Virtual(snapshot) => snapshot.process_epoch,
            Self::Legacy(snapshot) => snapshot.process_epoch,
        }
    }
    pub(crate) fn backend_sha256(&self) -> [u8; 32] {
        match self {
            Self::Virtual(snapshot) => snapshot.backend_sha256,
            Self::Legacy(snapshot) => snapshot.backend_sha256,
        }
    }
    pub(crate) fn validator_loaded(&self) -> bool {
        match self {
            Self::Virtual(snapshot) => snapshot.validator_loaded,
            Self::Legacy(snapshot) => snapshot.validator_loaded,
        }
    }
    pub(crate) fn role_model_identity(&self) -> String {
        match self {
            Self::Virtual(snapshot) => snapshot.role_model_identity(),
            Self::Legacy(snapshot) => snapshot.role_model_identity().into(),
        }
    }
    pub(crate) fn model_value_identity(&self) -> Result<ModelValueIdentity, RoleError> {
        match self {
            Self::Virtual(snapshot) => snapshot.model_value_identity(),
            Self::Legacy(snapshot) => Ok(snapshot.model_value_identity().clone()),
        }
    }
    pub(crate) fn adaptive_model_namespace(&self) -> Result<AdaptiveModelNamespace, RoleError> {
        match self {
            Self::Virtual(snapshot) => snapshot.adaptive_model_namespace(),
            Self::Legacy(snapshot) => Ok(snapshot.adaptive_model_namespace()),
        }
    }
}
