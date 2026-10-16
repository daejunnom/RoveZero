//! Search-independent adaptive cost and worker control primitives.
//!
//! These primitives suggest scheduling actions; they do not authorize a search
//! result, refund search credit, or claim physical GPU completion. The single
//! coordinator owner applies advice within its original deadline and ledger.

mod controller;
mod cost;
mod mailbox;
mod ready;
mod thread_owner;
mod topology;

pub use controller::*;
pub use cost::*;
pub use mailbox::*;
pub use ready::*;
pub use thread_owner::*;
pub use topology::*;
