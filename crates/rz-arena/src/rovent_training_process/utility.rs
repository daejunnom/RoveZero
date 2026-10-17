//! Utility-only training declarations and original process ownership.
//!
//! The utility learner has its own preparation, cohort evidence, checkpoint
//! and execution namespaces. Passive declarations do not mint a comparison,
//! optimizer update, learned model, or past native execution capability.
#![cfg(target_os = "linux")]

#[macro_use]
mod registration;
mod cost;
mod execution;
mod execution_v2;
mod registration_cohort;
mod window_v2;

pub use cost::*;
pub use execution::*;
pub use execution_v2::*;
pub use registration::*;
pub use registration_cohort::*;
pub use window_v2::*;

#[cfg(test)]
mod wire_tests;
#[cfg(test)]
mod wire_v2_tests;
