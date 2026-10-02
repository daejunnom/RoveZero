//! Model-specific evaluation and backend test support.
//!
//! The shared evaluator trait belongs to `rz-contracts` (TASK-I01). Until that
//! crate is published, this crate exposes backend primitives, not a replacement
//! `EvalRequest`/`EvalResult` contract. In particular, the mock emits physical
//! events; the runtime owns logical finalization and search owns backup.

#![forbid(unsafe_code)]

pub mod mock;
pub mod output;

/// Untrusted output at the model adapter boundary. Missing heads, bad shapes,
/// and non-finite values are deliberately representable for fault injection.
#[derive(Clone, Debug, PartialEq)]
pub struct RawOutput {
    pub policy_logits: Vec<f32>,
    pub wdl: Vec<f32>,
}
