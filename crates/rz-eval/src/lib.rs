//! Model-specific evaluation and backend test support.
//!
//! The runtime owns logical finalization and search owns backup. Backend calls
//! and mock events do not independently authorize either operation.

#![forbid(unsafe_code)]

pub mod asset;
pub mod error;
pub mod mock;
pub mod output;

#[cfg(feature = "onnx")]
pub mod onnx;

/// Untrusted output at the model adapter boundary. Missing heads, bad shapes,
/// and non-finite values are deliberately representable for fault injection.
#[derive(Clone, Debug, PartialEq)]
pub struct RawOutput {
    pub policy_logits: Vec<f32>,
    pub wdl: Vec<f32>,
}
