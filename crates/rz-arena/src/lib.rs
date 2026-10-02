//! E-owned input planning and event bookkeeping, not an execution permit.
//! Rules reconstruction, runner receipts and physical fairness require adapters.
#![forbid(unsafe_code)]

mod ledger;
mod plan;

pub use ledger::*;
pub use plan::*;

use rz_experiments::ManifestError;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

pub const MAX_JSON_BYTES: usize = rz_experiments::MAX_MANIFEST_BYTES;

#[derive(Debug)]
pub enum ArenaError {
    Manifest(ManifestError),
    Invalid(String),
    Integrity(String),
    Budget(String),
    Io(String),
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "arena validation: {reason}"),
            Self::Integrity(reason) => write!(f, "arena integrity: {reason}"),
            Self::Budget(reason) => write!(f, "arena budget: {reason}"),
            Self::Io(reason) => write!(f, "arena IO: {reason}"),
        }
    }
}

impl std::error::Error for ArenaError {}

impl From<ManifestError> for ArenaError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Strict bounded JSON shared with E01; semantic validation belongs to the reader.
pub fn decode_json<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, ArenaError> {
    rz_experiments::decode_json(input).map_err(ArenaError::Manifest)
}

/// Canonical sorted compact UTF-8 JSON SHA-256 for versioned E02 formats.
/// Persisted schemas use integer numeric fields only. Include a domain/version
/// in the hashed payload; exclude its own digest field.
pub fn canonical_sha256<T: Serialize>(value: &T) -> Result<String, ArenaError> {
    let mut value =
        serde_json::to_value(value).map_err(|error| ArenaError::Integrity(error.to_string()))?;
    value.sort_all_objects();
    let bytes =
        serde_json::to_vec(&value).map_err(|error| ArenaError::Integrity(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
