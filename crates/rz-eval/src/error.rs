//! Bounded diagnostics for C's internal asset and physical backend boundary.

use sha2::{Digest, Sha256};
use std::fmt::{self, Write};

/// Diagnostic identity, not an error-message archive or a cryptographic proof
/// of execution. Format at most this many external bytes; never retain or emit
/// a library's raw text in this receipt. Separate bounded local diagnostics are
/// opt-in and are never rendered by BackendError Display/Debug or common errors.
pub const CAUSE_PREFIX_BYTES: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CauseCode {
    RuntimePath,
    RuntimeInitialize,
    RuntimePanic,
    OrtNative,
    SessionBuilder,
    SessionConfiguration,
    ProviderQuery,
    ProviderRegistration,
    ProfilingStart,
    ModelLoad,
    ProfilingFinish,
    InputAllocation,
    TensorCreate,
    OrtRun,
    PolicyExtract,
    WdlExtract,
    OutputValidation,
    ProfileParse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputHead {
    Policy,
    Wdl,
}

/// Nonsecret typed context from the model-head validator. Floating-point sums
/// are stored as IEEE bits so the receipt remains Copy/Eq without rounding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputCause {
    WrongShape {
        head: OutputHead,
        actual: usize,
        expected: usize,
    },
    NonFinite {
        head: OutputHead,
        index: usize,
    },
    WdlOutOfRange {
        index: usize,
    },
    InvalidWdlSum {
        actual_f64_bits: u64,
    },
    EmptyLegalPolicy,
    TooManyLegalIndices,
    InvalidPolicyIndex(usize),
    DuplicatePolicyIndex(usize),
    AllocationFailed,
    InvalidPolicyNormalization,
}

/// Bounded external-cause receipt. Prefixes that differ only after the limit
/// intentionally share a fingerprint; `truncated` prevents calling it a full
/// message digest. No source string or path survives in this Copy value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalCause {
    pub code: CauseCode,
    pub prefix_sha256: [u8; 32],
    pub hashed_bytes: u16,
    pub truncated: bool,
    pub formatting_failed: bool,
    pub output: Option<OutputCause>,
}

impl ExternalCause {
    pub fn capture(code: CauseCode, source: &impl fmt::Display) -> Self {
        let mut sink = CauseSink {
            hash: Sha256::new(),
            bytes: 0,
            truncated: false,
        };
        sink.hash.update(b"rz-eval-cause-prefix-v1\0");
        let formatting_failed =
            fmt::write(&mut sink, format_args!("{source}")).is_err() && !sink.truncated;
        Self {
            code,
            prefix_sha256: sink.hash.finalize().into(),
            hashed_bytes: sink.bytes as u16,
            truncated: sink.truncated,
            formatting_failed,
            output: None,
        }
    }
}

struct CauseSink {
    hash: Sha256,
    bytes: usize,
    truncated: bool,
}

impl Write for CauseSink {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let remaining = CAUSE_PREFIX_BYTES - self.bytes;
        let count = remaining.min(text.len());
        self.hash.update(&text.as_bytes()[..count]);
        self.bytes += count;
        if count != text.len() {
            self.truncated = true;
            // Stop formatting instead of walking an unbounded external message.
            return Err(fmt::Error);
        }
        Ok(())
    }
}

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
    pub cause: Option<ExternalCause>,
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
            cause: None,
            native: None,
        }
    }

    pub fn with_external_cause(mut self, code: CauseCode, source: &impl fmt::Display) -> Self {
        self.cause = Some(ExternalCause::capture(code, source));
        self
    }

    pub fn with_output_cause(mut self, source: &crate::output::OutputError) -> Self {
        use crate::output::{Head, OutputError};
        let head = |value| match value {
            Head::Policy => OutputHead::Policy,
            Head::Wdl => OutputHead::Wdl,
        };
        let context = match *source {
            OutputError::WrongShape {
                head: value,
                actual,
                expected,
            } => OutputCause::WrongShape {
                head: head(value),
                actual,
                expected,
            },
            OutputError::NonFinite { head: value, index } => OutputCause::NonFinite {
                head: head(value),
                index,
            },
            OutputError::WdlOutOfRange { index } => OutputCause::WdlOutOfRange { index },
            OutputError::InvalidWdlSum { actual } => OutputCause::InvalidWdlSum {
                actual_f64_bits: actual.to_bits(),
            },
            OutputError::EmptyLegalPolicy => OutputCause::EmptyLegalPolicy,
            OutputError::TooManyLegalIndices => OutputCause::TooManyLegalIndices,
            OutputError::InvalidPolicyIndex(index) => OutputCause::InvalidPolicyIndex(index),
            OutputError::DuplicatePolicyIndex(index) => OutputCause::DuplicatePolicyIndex(index),
            OutputError::AllocationFailed => OutputCause::AllocationFailed,
            OutputError::InvalidPolicyNormalization => OutputCause::InvalidPolicyNormalization,
        };
        let mut cause = ExternalCause::capture(CauseCode::OutputValidation, source);
        cause.output = Some(context);
        self.cause = Some(cause);
        self.with_diagnostic("OutputValidation", &source.to_string())
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
        let error_with_cause = if self.cause.is_none() {
            self.with_external_cause(CauseCode::OrtNative, &error)
        } else {
            self
        };
        error_with_cause.with_diagnostic(&format!("{:?}", error.code()), error.message())
    }

    #[cfg(feature = "onnx")]
    pub(crate) fn with_ort_cause(self, code: CauseCode, error: ort::Error) -> Self {
        self.with_external_cause(code, &error).with_ort(error)
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}/{:?}: {}", self.stage, self.kind, self.detail)?;
        if let Some(cause) = self.cause {
            write!(f, " [cause={:?} prefix_sha256=", cause.code)?;
            for byte in cause.prefix_sha256 {
                write!(f, "{byte:02x}")?;
            }
            write!(
                f,
                " bytes={} truncated={} formatting_failed={} output={:?}]",
                cause.hashed_bytes, cause.truncated, cause.formatting_failed, cause.output
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for BackendError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_cause_is_bounded_cloneable_and_never_formats_raw_text() {
        let source = "synthetic/private/runtime/path: vendor failure";
        let error = BackendError::new(
            FailureKind::BackendFailure,
            FailureStage::Backend,
            "synchronous ORT Run failed",
        )
        .with_external_cause(CauseCode::OrtRun, &source)
        .with_diagnostic("VendorCode", source);
        let cause = error.cause.unwrap();
        assert_eq!(error.clone(), error);
        assert_eq!(error.native.as_ref().unwrap().message, source);
        assert_eq!(cause.code, CauseCode::OrtRun);
        assert_eq!(usize::from(cause.hashed_bytes), source.len());
        assert!(!cause.truncated);
        assert!(!cause.formatting_failed);
        assert!(error.to_string().contains("OrtRun"));
        assert!(!error.to_string().contains(source));
        assert!(!format!("{error:?}").contains(source));
        assert_ne!(
            cause.prefix_sha256,
            ExternalCause::capture(CauseCode::OrtRun, &"different vendor failure").prefix_sha256
        );

        let oversized = "x".repeat(CAUSE_PREFIX_BYTES * 10);
        let bounded = ExternalCause::capture(CauseCode::OrtRun, &oversized);
        assert_eq!(usize::from(bounded.hashed_bytes), CAUSE_PREFIX_BYTES);
        assert!(bounded.truncated);
        assert!(!bounded.formatting_failed);
    }

    #[test]
    fn source_formatting_failure_is_distinct_from_truncation() {
        struct FailingDisplay;
        impl fmt::Display for FailingDisplay {
            fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
                Err(fmt::Error)
            }
        }
        let cause = ExternalCause::capture(CauseCode::OrtRun, &FailingDisplay);
        assert_eq!(cause.hashed_bytes, 0);
        assert!(!cause.truncated);
        assert!(cause.formatting_failed);
    }

    #[test]
    fn output_validator_preserves_typed_head_index_and_sum() {
        let base = BackendError::new(
            FailureKind::NumericalFailure,
            FailureStage::Output,
            "model output rejected",
        );
        assert_eq!(
            base.clone().with_output_cause(&crate::output::OutputError::NonFinite {
                head: crate::output::Head::Policy,
                index: 17,
            })
            .cause
            .unwrap()
            .output,
            Some(OutputCause::NonFinite {
                head: OutputHead::Policy,
                index: 17,
            })
        );
        let sum = 0.75_f64;
        assert_eq!(
            base.with_output_cause(&crate::output::OutputError::InvalidWdlSum { actual: sum })
                .cause
                .unwrap()
                .output,
            Some(OutputCause::InvalidWdlSum {
                actual_f64_bits: sum.to_bits(),
            })
        );
    }
}
