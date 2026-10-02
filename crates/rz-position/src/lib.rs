//! Independent standard-chess state, checked transitions and immutable views.
//!
//! This crate owns concrete rule state. Shared evaluator identifiers and request
//! types remain owned by `rz-contracts`; no backend or search types live here.
#![forbid(unsafe_code)]

mod fen;
mod movegen;
mod outcome;
mod position;
mod types;

pub use outcome::*;
pub use position::*;
pub use types::*;
