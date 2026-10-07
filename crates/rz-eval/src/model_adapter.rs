//! Model-independent C boundary. D retains physical leases; B alone owns backup.
use crate::error::{BackendError, FailureKind, FailureStage};
use rz_contracts::*;
use rz_position::contracts::RulesState;
use rz_runtime::Resources;
use std::sync::Arc;

/// Reservations only. They never claim measured peaks or a native allocation cap.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdapterAdmission {
    pub execution: Resources,
    pub session: Resources,
}
impl AdapterAdmission {
    pub fn execution_resources(self) -> Resources {
        self.execution
    }
    pub fn session_resident_admission(self) -> Resources {
        self.session
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ModelCapabilities {
    pub input: &'static str,
    pub policy_head: &'static str,
    pub host_bytes_per_item: u64,
    /// Opt-in only for a full raw head that is projected independently per request.
    pub mixed_legal_batching: bool,
}

/// Concrete selection happens once at bootstrap. No universal tensor/raw shape.
pub trait ModelAdapter: Clone + Send + Sync + 'static {
    type PreparedInput: Send + Sync + 'static;
    type PreparedBatch: Send + Sync + 'static;
    type RawOutput;
    fn model(&self) -> &Arc<ModelDescriptor>;
    fn backend(&self) -> Digest;
    fn capabilities(&self) -> ModelCapabilities;
    fn input_key(&self, state: &RulesState, legal: &[Move]) -> Result<EvalInputKey, ContractError>;
    fn prepare(
        &self,
        request: Arc<EvalRequest<RulesState>>,
    ) -> Result<Self::PreparedInput, ContractError>;
    fn batch(
        &self,
        execution: ExecutionId,
        items: Vec<Self::PreparedInput>,
    ) -> Result<Self::PreparedBatch, ContractError>;
    fn decode(
        &self,
        input: &Self::PreparedInput,
        raw: &Self::RawOutput,
        execution: ExecutionId,
    ) -> Result<EvalOutput, PhysicalFailure>;
    fn validate_runtime(&self, max_batch: usize) -> Result<(), ContractError>;
}

/// Preserve bounded native cause beside the common error. D can retain this
/// local evidence before publishing EvalFailure with its own completion context.
#[derive(Clone, Debug)]
pub struct PhysicalFailure {
    pub contract: ContractError,
    pub backend: Option<BackendError>,
}

impl From<ContractError> for PhysicalFailure {
    fn from(contract: ContractError) -> Self {
        Self {
            contract,
            backend: None,
        }
    }
}

impl From<BackendError> for PhysicalFailure {
    fn from(backend: BackendError) -> Self {
        Self {
            contract: backend_error(&backend),
            backend: Some(backend),
        }
    }
}

impl std::fmt::Display for PhysicalFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.contract.fmt(f)
    }
}
impl std::error::Error for PhysicalFailure {}

pub fn backend_error(failure: &BackendError) -> ContractError {
    let code = match failure.kind {
        FailureKind::InvalidInput => ErrorCode::InvalidInput,
        FailureKind::UnsupportedModel => ErrorCode::UnsupportedContract,
        FailureKind::IdentityMismatch => ErrorCode::IdentityMismatch,
        FailureKind::BackendUnavailable => ErrorCode::BackendUnavailable,
        FailureKind::ResourceExhausted => ErrorCode::ResourceExhausted,
        FailureKind::NumericalFailure => ErrorCode::NumericalFailure,
        FailureKind::BackendFailure | FailureKind::Io => ErrorCode::BackendFailure,
    };
    let stage = match failure.stage {
        FailureStage::Asset => Stage::Contract,
        FailureStage::Admission => Stage::Admission,
        FailureStage::Backend => Stage::Backend,
        FailureStage::Output => Stage::Output,
    };
    ContractError::new(code, stage, failure.detail)
}
