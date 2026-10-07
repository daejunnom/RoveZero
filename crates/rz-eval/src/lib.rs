//! Model-specific evaluation and backend test support.
//!
//! The runtime owns logical finalization and search owns backup. Backend calls
//! and mock events do not independently authorize either operation.

#![forbid(unsafe_code)]

pub mod adapters;
pub use adapters::lc0::{asset, output};
#[cfg(feature = "experimental-batch")]
pub mod batch_journal;
pub mod error;
pub mod mock;
pub mod pals_model;
#[cfg(all(feature = "onnx", feature = "contracts"))]
pub mod pals_onnx;
pub mod runtime_pin;
pub mod worker;

#[cfg(feature = "contracts")]
pub use adapters::lc0::contracts;
#[cfg(feature = "contracts")]
pub mod model_adapter;

#[cfg(feature = "contracts")]
pub use adapters::lc0::rules_projection;

#[cfg(feature = "contracts")]
pub mod runtime_bridge;

#[cfg(feature = "contracts")]
pub mod native_runtime_bridge;

#[cfg(feature = "onnx")]
pub use adapters::lc0::onnx;

#[cfg(feature = "experimental-ort-model")]
pub mod ort_model;

/// Compatibility alias for the supported LC0 raw heads, not a universal model output.
pub use adapters::lc0::RawOutput;
#[cfg(feature = "experimental-raw-cache")]
pub mod raw_cache;
