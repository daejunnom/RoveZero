//! Bounded diagnostics for C's internal asset and physical backend boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailureKind {
    InvalidInput,
    UnsupportedModel,
    IdentityMismatch,
    BackendUnavailable,
    ResourceExhausted,
    NumericalFailure,
    BackendFailure,
    Io,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailureStage {
    Asset,
    Admission,
    Backend,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendError {
    pub kind: FailureKind,
    pub stage: FailureStage,
    pub detail: &'static str,
}

impl BackendError {
    pub const fn new(kind: FailureKind, stage: FailureStage, detail: &'static str) -> Self {
        Self {
            kind,
            stage,
            detail,
        }
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}/{:?}: {}", self.stage, self.kind, self.detail)
    }
}

impl std::error::Error for BackendError {}
