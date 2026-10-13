//! Shared pre-provider startup admission. The arena and native UCI owner consume
//! the same producer and numerical verifiers; neither can replace them with a
//! parent status flag. File/source identity and tensor math are separate from
//! the new process's readiness, resource application and physical completion.
mod artifact_domains;
pub use artifact_domains::*;

#[cfg(feature = "pals-content-types")]
mod numerical;
#[cfg(feature = "pals-content-types")]
pub use numerical::*;

#[cfg(feature = "pals-content-admission")]
mod owned_files;
#[cfg(feature = "pals-content-admission")]
pub use owned_files::*;
