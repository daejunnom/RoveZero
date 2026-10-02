//! Bridge between E's input format and the coordinator-owned native contract.
//! Manifest schema 1 is independent of engine contract revision 0.1. The source
//! commit pins an implementation; it is not itself a protocol/schema version.
use crate::{ArenaError, ArenaPlan};
use rz_contracts::{CONTRACT_REVISION, ContractError, ErrorCode, SchemaVersion, Stage};
use rz_experiments::ManifestError;

pub const ENGINE_CONTRACT_SOURCE_REVISION: &str = "67284c4f66f7a7ae9f46fa63dfd50e7410eb6845";
pub const ENGINE_CONTRACT_MANIFEST_REVISION: &str = "0.1";

/// Admission check for an execution adapter. Structural plans can still carry
/// an absent revision; this bridge cannot authorize execution without exact 0.1.
/// Returning a schema version does not attest evaluator liveness or GPU work.
pub fn validate_engine_contract_revision(plan: &ArenaPlan) -> Result<SchemaVersion, ArenaError> {
    if plan.manifest().input().contract_revision.as_deref()
        != Some(ENGINE_CONTRACT_MANIFEST_REVISION)
    {
        return Err(ContractError::new(
            ErrorCode::UnsupportedContract,
            Stage::Contract,
            "engine contract revision must exactly match pinned 0.1",
        )
        .into());
    }
    // Validate the imported type, preserving its stable code and stage. Do not
    // parse the E manifest schema number into a duplicate contract type.
    CONTRACT_REVISION.validate()?;
    if CONTRACT_REVISION.major != 0 || CONTRACT_REVISION.minor != 1 {
        return Err(ContractError::new(
            ErrorCode::UnsupportedContract,
            Stage::Contract,
            "native engine contract differs from the pinned manifest mapping",
        )
        .into());
    }
    Ok(CONTRACT_REVISION)
}

/// Preserve native contract errors; translate local failures to shared codes.
/// Details are static and bounded, never copied from paths or backend strings.
pub fn contract_error(error: &ArenaError) -> ContractError {
    match error {
        ArenaError::Execution(failure) => contract_error(&failure.cause),
        ArenaError::Contract(error) => *error,
        ArenaError::Invalid(_) => ContractError::new(
            ErrorCode::InvalidInput,
            Stage::Admission,
            "arena input or event is invalid",
        ),
        ArenaError::Budget(_) => ContractError::new(
            ErrorCode::ResourceExhausted,
            Stage::Admission,
            "arena resource limit exceeded",
        ),
        ArenaError::Integrity(_) => ContractError::new(
            ErrorCode::IdentityMismatch,
            Stage::Contract,
            "arena artifact or input identity mismatch",
        ),
        ArenaError::Io(_) => ContractError::new(
            ErrorCode::BackendFailure,
            Stage::Backend,
            "arena process or IO operation failed",
        ),
        ArenaError::Manifest(error) => match error {
            ManifestError::Parse(_) | ManifestError::Validation(_) => ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "arena manifest is invalid",
            ),
            ManifestError::Integrity(_) => ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "arena manifest identity mismatch",
            ),
            ManifestError::Artifact(_) => ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Backend,
                "arena artifact verification failed",
            ),
        },
    }
}
