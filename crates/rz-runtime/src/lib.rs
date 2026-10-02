//! Bounded evaluation scheduling, independent of concrete chess/model contracts.
//!
//! The embedding adapter owns shared identifiers, validation and terminal result
//! construction. This crate owns only admission, batching and execution lifetime.
#![forbid(unsafe_code)]

mod boundary;

pub use boundary::*;
