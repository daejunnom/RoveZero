//! B-owned search implementation. Shared contracts and concrete adapters are supplied by I/A/C/D.
#![forbid(unsafe_code)]

pub mod contract_time;
pub mod contracts;
pub mod cpu;
pub mod cpu_checker;
pub mod cpu_value;
pub mod driver;
pub mod external_cpu;
pub mod pals;
pub mod policy;
pub mod time;
pub mod tree;

pub use policy::{EdgeStats, PolicyIdentity, Puct, SelectionPolicy};
pub use tree::{
    Completion, Leaf, SearchCounters, SearchError, Selection, SelectionTicket, Tree, TreeLimits,
};
