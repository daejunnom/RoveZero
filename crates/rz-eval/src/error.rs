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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendError {
    pub kind: FailureKind,
    pub stage: FailureStage,
    pub detail: &'static str,
    /// Bounded local diagnostics, never copied into the common ContractError.
    pub native: Option<Box<NativeDiagnostic>>,
}

#[derive(Clone, Eq, PartialEq)]
pub struct NativeDiagnostic {
    pub code: String,
    pub message: String,
    pub truncated: bool,
}

impl std::fmt::Debug for NativeDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Reading the raw native text is an explicit local diagnostics operation.
        f.debug_struct("NativeDiagnostic")
            .field("code", &self.code)
            .field("message_bytes", &self.message.len())
            .field("truncated", &self.truncated)
            .finish()
    }
}

impl BackendError {
    pub const fn new(kind: FailureKind, stage: FailureStage, detail: &'static str) -> Self {
        Self {
            kind,
            stage,
            detail,
            native: None,
        }
    }

    pub fn with_diagnostic(mut self, code: &str, message: &str) -> Self {
        fn bounded(text: &str, limit: usize) -> String {
            let mut end = text.len().min(limit);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text[..end].to_owned()
        }
        self.native = Some(Box::new(NativeDiagnostic {
            code: bounded(code, 64),
            message: bounded(message, 1024),
            truncated: code.len() > 64 || message.len() > 1024,
        }));
        self
    }

    #[cfg(feature = "onnx")]
    pub(crate) fn with_ort(self, error: ort::Error) -> Self {
        self.with_diagnostic(&format!("{:?}", error.code()), error.message())
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}/{:?}: {}", self.stage, self.kind, self.detail)
    }
}

impl std::error::Error for BackendError {}
