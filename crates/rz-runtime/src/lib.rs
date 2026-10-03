//! Bounded evaluation scheduling, independent of concrete chess/model contracts.
//!
//! The embedding adapter owns shared identifiers, validation and terminal result
//! construction. This crate owns only admission, batching and execution lifetime.
#![forbid(unsafe_code)]

mod boundary;
#[cfg(feature = "experimental-notify")]
mod notification;
mod observation;
mod scheduler;
#[cfg(feature = "experimental-notify")]
pub use notification::CompletionSignal;

#[cfg(feature = "contracts")]
pub mod contracts;

pub use boundary::*;
pub use observation::*;
pub use scheduler::*;
