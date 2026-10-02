//! Model-space encoding only. Rules owns positions and ordered legal moves.
//!
//! These primitives are not replacement shared state/move contracts. An adapter
//! from TASK-I01/A03's checked views will live here once those APIs are published.

#![forbid(unsafe_code)]

pub mod classical;
pub mod policy;

pub const POLICY_SIZE: usize = 1858;
