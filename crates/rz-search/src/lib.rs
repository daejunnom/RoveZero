//! B-owned search implementation. Shared contracts and concrete adapters are supplied by I/A/C/D.
#![forbid(unsafe_code)]

pub mod driver;
pub mod policy;
pub mod time;
pub mod tree;

pub use policy::{EdgeStats, PolicyIdentity, Puct, SelectionPolicy};
pub use tree::{
    Completion, Leaf, SearchCounters, SearchError, Selection, SelectionTicket, Tree, TreeLimits,
};
