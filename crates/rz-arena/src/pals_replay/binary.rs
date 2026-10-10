//! Actual loaded executable content checked by the caller before original stdin.
//! This is not native NN, dependency, isolation or next Query admission.

use super::OwnedReplayCapture;
use crate::{ArenaError, OriginalLoadedImage};
use rz_uci::pals_cpu_task::strategic_action::ArtifactPin;
use std::time::Instant;

#[derive(Debug)]
pub(super) struct LoadedReplayBinaryObservation {
    pub artifact: ArtifactPin,
    pub device: u64,
    pub inode: u64,
    pub started_ns: u64,
    pub finished_ns: u64,
}

/// No public constructor, Clone or serde. The actual capture owns the content
/// observation and the original process owns its PID/file/pipe timing evidence.
pub struct CheckedReplayLoadedBinary<'a> {
    capture: &'a OwnedReplayCapture,
    image: OriginalLoadedImage<'a>,
    verification: &'a LoadedReplayBinaryObservation,
}
impl CheckedReplayLoadedBinary<'_> {
    pub fn artifact(&self) -> &ArtifactPin {
        &self.verification.artifact
    }
    pub fn loaded_image(&self) -> &OriginalLoadedImage<'_> {
        &self.image
    }
    pub fn verification_started_ns(&self) -> u64 {
        self.verification.started_ns
    }
    pub fn verification_finished_ns(&self) -> u64 {
        self.verification.finished_ns
    }
    pub fn original_started(&self) -> Instant {
        self.capture.bundle().original_started()
    }
    pub fn assurance_scope(&self) -> &'static str {
        "actual_parent_loaded_executable_sha256_before_stdin_pending_native_and_next_query"
    }
}

impl OwnedReplayCapture {
    /// Requires the actual private callback observation and complete same-owner
    /// closure. A generic verifier returning Ok or reported JSON is insufficient.
    pub fn checked_loaded_binary(&self) -> Result<CheckedReplayLoadedBinary<'_>, ArenaError> {
        if !self.transport_complete() || self.loaded_binary_error.is_some() {
            return Err(invalid(
                "loaded binary requires complete actual caller closure",
            ));
        }
        let image = self.process().checked_loaded_image()?;
        let verification = self
            .loaded_binary
            .as_ref()
            .ok_or_else(|| invalid("actual loaded binary content unobserved"))?;
        if verification.device != image.file_device() || verification.inode != image.file_inode() {
            return Err(invalid("loaded binary file owner differs"));
        }
        check_order([
            image.observed_ns(),
            verification.started_ns,
            verification.finished_ns,
            image.first_input_written_ns(),
        ])?;
        Ok(CheckedReplayLoadedBinary {
            capture: self,
            image,
            verification,
        })
    }
}

fn invalid(detail: &str) -> ArenaError {
    ArenaError::Invalid(detail.into())
}
fn check_order(stamps: [u64; 4]) -> Result<(), ArenaError> {
    if stamps.windows(2).any(|pair| pair[0] > pair[1]) {
        Err(invalid(
            "loaded binary verification must precede original input",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_binary_content_order_refuses_verification_after_input_or_before_image() {
        // Pure chronology boundary only; does not manufacture a capture/token.
        assert!(check_order([1, 2, 3, 4]).is_ok());
        assert!(check_order([1, 1, 1, 1]).is_ok());
        for bad in [[2, 1, 3, 4], [1, 3, 2, 4], [1, 2, 4, 3]] {
            assert!(check_order(bad).is_err());
        }
    }
}
