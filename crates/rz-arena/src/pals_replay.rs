//! Caller-owned observed frozen replay launch. Transport closure never admits a
//! native result, checked Query, utility group or original model/source fact.
//! The prepared byte owner and pending child are retained for caller recovery.

use crate::{ArenaError, OriginalProcessOutput, ProcessLimits};
#[cfg(target_os = "linux")]
use rz_uci::pals_cpu_task::strategic_action::replay_inputs::ReplayInputMode;
use rz_uci::pals_cpu_task::strategic_action::replay_launch_preparation::PreparedReplayLaunchBundle;
#[cfg(target_os = "linux")]
use rz_uci::pals_cpu_task::strategic_action::replay_launch_preparation::{
    EXPECTED_TRANSPORT_V2_SCHEMA, EXPECTED_TRANSPORT_V3_SCHEMA, ExpectedTransportV3,
    NativeResultLane, OBSERVED_PREPARATION_MANIFEST_SCHEMA,
};
use std::fs::File;
use std::sync::atomic::AtomicBool;
#[cfg(target_os = "linux")]
use std::time::Instant;

pub const MAX_REPLAY_CALLER_BINARY_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_REPLAY_CALLER_READ_BYTES: u64 = MAX_REPLAY_CALLER_BINARY_BYTES + 32 * 1024 * 1024;

/// Independent, finite parent policy. The process fields must match the prepared
/// wall/output declarations; verification reads consume the same original E.
#[derive(Clone, Copy, Debug)]
pub struct ReplayCallerPolicy {
    pub process: ProcessLimits,
    pub maximum_binary_bytes: u64,
    pub verification_read_bytes: u64,
}

#[derive(Debug)]
pub struct ReplayCallerFailure {
    pub cause: ArenaError,
    bundle: PreparedReplayLaunchBundle,
}
impl ReplayCallerFailure {
    pub fn bundle(&self) -> &PreparedReplayLaunchBundle {
        &self.bundle
    }
    pub fn into_bundle(self) -> PreparedReplayLaunchBundle {
        self.bundle
    }
}
impl std::fmt::Display for ReplayCallerFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "frozen replay caller refused: {}", self.cause)
    }
}
impl std::error::Error for ReplayCallerFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// No Deserialize/Clone constructor: these fields are actual caller observations.
/// Read-only access does not consume the pending-child custody held by process.
#[derive(Debug)]
pub struct OwnedReplayCapture {
    bundle: PreparedReplayLaunchBundle,
    process: OriginalProcessOutput,
    postflight_error: Option<ArenaError>,
    finished_before_original_whole: bool,
    cancelled_at_return: bool,
}
impl OwnedReplayCapture {
    pub fn bundle(&self) -> &PreparedReplayLaunchBundle {
        &self.bundle
    }
    pub fn process(&self) -> &OriginalProcessOutput {
        &self.process
    }
    pub fn postflight_error(&self) -> Option<&ArenaError> {
        self.postflight_error.as_ref()
    }
    pub fn transport_complete(&self) -> bool {
        self.process.transport_complete()
            && self.postflight_error.is_none()
            && self.finished_before_original_whole
            && !self.cancelled_at_return
    }
    /// The caller remains responsible for any unverified pending child. Drop
    /// neither deletes durable inputs nor declares cleanup/physical completion.
    pub fn into_parts(self) -> (PreparedReplayLaunchBundle, OriginalProcessOutput) {
        (self.bundle, self.process)
    }
}

#[cfg(target_os = "linux")]
fn check_lane_and_policy(
    expected: &ExpectedTransportV3,
    policy: ReplayCallerPolicy,
) -> Result<(), ArenaError> {
    check_resource_policy(
        expected.whole_wall_ms,
        expected.cleanup_reserve_ms,
        expected.output_bytes,
        policy,
    )?;
    if expected.schema != EXPECTED_TRANSPORT_V3_SCHEMA
        || expected.expectations.schema != EXPECTED_TRANSPORT_V2_SCHEMA
        || expected.native_result != NativeResultLane::QueryPriorV1
        || !expected.same_original_clock()
    {
        return Err(ArenaError::Invalid(
            "replay caller requires explicit /3 lane and unchanged clock".into(),
        ));
    }
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn check_resource_policy(
    whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    output_bytes: usize,
    policy: ReplayCallerPolicy,
) -> Result<(), ArenaError> {
    let output_bytes = u64::try_from(output_bytes)
        .map_err(|_| ArenaError::Budget("replay caller output limit conversion overflow".into()))?;
    let execution_ms = whole_wall_ms
        .checked_sub(cleanup_reserve_ms)
        .filter(|ms| *ms > 0)
        .ok_or_else(|| ArenaError::Budget("replay caller original execution window".into()))?;
    let grace = cleanup_reserve_ms.div_ceil(2);
    if cleanup_reserve_ms == 0
        || policy.process.wall_ms != execution_ms
        || policy.process.shutdown_grace_ms != grace
        || policy.process.max_output_bytes != output_bytes
        || policy.process.max_child_processes != 1
        || !(1..=MAX_REPLAY_CALLER_BINARY_BYTES).contains(&policy.maximum_binary_bytes)
        || !(1..=MAX_REPLAY_CALLER_READ_BYTES).contains(&policy.verification_read_bytes)
    {
        return Err(ArenaError::Invalid(
            "replay caller requires explicit /3 lane and unchanged finite resource policy".into(),
        ));
    }
    Ok(())
}

/// The already-open ELF and cwd are caller-owned independent handles. This
/// rechecks prepared /3 bytes, descriptor pins and original deadlines, then uses
/// the existing Linux group/pipe/reap owner. It never substitutes an engine or
/// turns captured JSON into a checked next prior. Default SIGCHLD/exclusive child
/// reaping and exclusive bundle-directory use remain caller preconditions.
pub fn supervise_prepared_observed_replay(
    bundle: PreparedReplayLaunchBundle,
    program: &File,
    directory: &File,
    policy: ReplayCallerPolicy,
    cancel: &AtomicBool,
) -> Result<OwnedReplayCapture, Box<ReplayCallerFailure>> {
    #[cfg(target_os = "linux")]
    {
        match linux::capture(&bundle, program, directory, policy, cancel) {
            Ok((process, postflight_error)) => {
                let finished_before_original_whole = Instant::now() < bundle.deadline();
                let cancelled_at_return = cancel.load(std::sync::atomic::Ordering::Acquire);
                Ok(OwnedReplayCapture {
                    bundle,
                    process,
                    postflight_error,
                    finished_before_original_whole,
                    cancelled_at_return,
                })
            }
            Err(cause) => Err(Box::new(ReplayCallerFailure { cause, bundle })),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, directory, policy, cancel);
        Err(Box::new(ReplayCallerFailure {
            cause: ArenaError::Invalid("frozen replay caller currently requires Linux".into()),
            bundle,
        }))
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use crate::{
        OriginalProcessInput, OriginalProcessWindow, OwnedArtifactTreeWatch,
        supervise_input_in_original_window,
    };
    use nix::fcntl::{OFlag, open, openat};
    use nix::sys::stat::Mode;
    use rz_uci::pals_cpu_task::strategic_action::ArtifactPin;
    use rz_uci::pals_cpu_task::strategic_action::replay_launch_preparation::{
        MAX_EXPECTED_BYTES, MAX_PUBLICATION_BYTES, ReplayLaunchPayload,
    };
    use sha2::{Digest, Sha256};
    use std::ffi::OsString;
    use std::os::unix::fs::{FileExt, MetadataExt};
    use std::sync::atomic::Ordering;

    fn check_clock(
        bundle: &PreparedReplayLaunchBundle,
        cancel: &AtomicBool,
        postflight: bool,
    ) -> Result<(), ArenaError> {
        if cancel.load(Ordering::Acquire) {
            return Err(ArenaError::Invalid("replay caller cancelled".into()));
        }
        let deadline = if postflight {
            bundle.deadline()
        } else {
            bundle.execution_deadline()
        };
        if Instant::now() >= deadline {
            return Err(ArenaError::Budget(
                "replay caller original window expired".into(),
            ));
        }
        Ok(())
    }

    fn payload<'a>(
        bundle: &'a PreparedReplayLaunchBundle,
        name: &str,
    ) -> Result<&'a ReplayLaunchPayload, ArenaError> {
        let mut matches = bundle.payloads().iter().filter(|p| p.name() == name);
        let result = matches
            .next()
            .ok_or_else(|| ArenaError::Invalid("replay caller prepared payload missing".into()))?;
        if matches.next().is_some() {
            return Err(ArenaError::Invalid(
                "replay caller duplicate payload".into(),
            ));
        }
        Ok(result)
    }

    fn same_identity(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
        a.dev() == b.dev()
            && a.ino() == b.ino()
            && a.len() == b.len()
            && a.mtime() == b.mtime()
            && a.mtime_nsec() == b.mtime_nsec()
            && a.ctime() == b.ctime()
            && a.ctime_nsec() == b.ctime_nsec()
    }

    struct ReadCredit {
        remaining: u64,
    }
    impl ReadCredit {
        fn consume(&mut self, bytes: usize) -> Result<(), ArenaError> {
            self.remaining = self.remaining.checked_sub(bytes as u64).ok_or_else(|| {
                ArenaError::Budget("replay caller verification read credit exhausted".into())
            })?;
            Ok(())
        }
    }

    /// read_at leaves the parent's file offset unchanged. Metadata comparison is
    /// cooperative integrity evidence; it is not hostile concurrent-write proof.
    fn verify_file<F: FnMut() -> Result<(), ArenaError>>(
        file: &File,
        pin: &ArtifactPin,
        maximum: u64,
        credit: &mut ReadCredit,
        mut check: F,
    ) -> Result<(), ArenaError> {
        check()?;
        if pin.bytes == 0 || pin.bytes > maximum || pin.bytes > credit.remaining {
            return Err(ArenaError::Budget(
                "replay caller bounded artifact size".into(),
            ));
        }
        let before = file
            .metadata()
            .map_err(|_| ArenaError::Io("replay caller artifact metadata".into()))?;
        if !before.is_file() || before.len() != pin.bytes {
            return Err(ArenaError::Integrity(
                "replay caller artifact kind/size".into(),
            ));
        }
        let mut hasher = Sha256::new();
        let mut block = [0u8; 16 * 1024];
        let mut offset = 0;
        while offset < pin.bytes {
            check()?;
            let take = usize::try_from((pin.bytes - offset).min(block.len() as u64))
                .map_err(|_| ArenaError::Budget("replay caller read count".into()))?;
            credit.consume(take)?;
            file.read_exact_at(&mut block[..take], offset)
                .map_err(|_| ArenaError::Io("replay caller artifact read".into()))?;
            hasher.update(&block[..take]);
            offset += take as u64;
        }
        let after = file
            .metadata()
            .map_err(|_| ArenaError::Io("replay caller artifact metadata after read".into()))?;
        check()?;
        if !same_identity(&before, &after) || format!("{:x}", hasher.finalize()) != pin.sha256 {
            return Err(ArenaError::Integrity(
                "replay caller artifact changed/pin mismatch".into(),
            ));
        }
        Ok(())
    }

    fn verify_publication(
        bundle: &PreparedReplayLaunchBundle,
        directory: &File,
        credit: &mut ReadCredit,
        cancel: &AtomicBool,
        postflight: bool,
    ) -> Result<(), ArenaError> {
        for p in bundle.payloads() {
            let file = File::from(
                openat(
                    directory,
                    p.name(),
                    OFlag::O_RDONLY | OFlag::O_NOFOLLOW | OFlag::O_NONBLOCK | OFlag::O_CLOEXEC,
                    Mode::empty(),
                )
                .map_err(|_| ArenaError::Io("replay caller published file open".into()))?,
            );
            if file
                .metadata()
                .map_err(|_| ArenaError::Io("replay caller published metadata".into()))?
                .nlink()
                != 1
            {
                return Err(ArenaError::Integrity(
                    "replay caller published hardlink".into(),
                ));
            }
            verify_file(
                &file,
                p.artifact(),
                MAX_PUBLICATION_BYTES as u64,
                credit,
                || check_clock(bundle, cancel, postflight),
            )?;
        }
        let manifest = File::from(
            openat(
                directory,
                "preparation-manifest.json",
                OFlag::O_RDONLY | OFlag::O_NOFOLLOW | OFlag::O_NONBLOCK | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(|_| ArenaError::Io("replay caller manifest open".into()))?,
        );
        verify_file(
            &manifest,
            bundle.manifest_artifact(),
            MAX_EXPECTED_BYTES as u64,
            credit,
            || check_clock(bundle, cancel, postflight),
        )
    }

    pub(super) fn capture(
        bundle: &PreparedReplayLaunchBundle,
        program: &File,
        directory: &File,
        policy: ReplayCallerPolicy,
        cancel: &AtomicBool,
    ) -> Result<(OriginalProcessOutput, Option<ArenaError>), ArenaError> {
        check_clock(bundle, cancel, false)?;
        let manifest = bundle.manifest();
        if manifest.schema != OBSERVED_PREPARATION_MANIFEST_SCHEMA
            || manifest.requested_native_result != Some(NativeResultLane::QueryPriorV1)
            || manifest.mode != ReplayInputMode::RepairOpponent4n
        {
            return Err(ArenaError::Invalid(
                "replay caller observed 4N prepared owner required".into(),
            ));
        }
        let expected_payload = payload(bundle, "replay-expected.json")?;
        if expected_payload.as_bytes().len() > MAX_EXPECTED_BYTES {
            return Err(ArenaError::Budget(
                "replay caller expected byte bound".into(),
            ));
        }
        let expected: ExpectedTransportV3 = serde_json::from_slice(expected_payload.as_bytes())
            .map_err(|_| ArenaError::Invalid("replay caller closed /3 expected codec".into()))?;
        check_lane_and_policy(&expected, policy)?;
        if manifest.whole_wall_ms != expected.whole_wall_ms
            || manifest.cleanup_reserve_ms != expected.cleanup_reserve_ms
        {
            return Err(ArenaError::Integrity(
                "replay caller manifest clock mismatch".into(),
            ));
        }
        let window = OriginalProcessWindow::new(
            bundle.original_started(),
            bundle.execution_deadline(),
            bundle.deadline(),
        )?;
        let request = payload(bundle, "replay-input.json")?;
        let assets = payload(bundle, "launch-assets.json")?;
        if *request.artifact() != expected.expectations.request_artifact
            || *assets.artifact() != expected.expectations.launch_asset_artifact
        {
            return Err(ArenaError::Integrity(
                "replay caller request/asset pins differ".into(),
            ));
        }
        let pinned = File::from(
            open(
                bundle.directory(),
                OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(|_| ArenaError::Io("replay caller bundle directory open".into()))?,
        );
        let supplied = directory
            .metadata()
            .map_err(|_| ArenaError::Io("replay caller cwd metadata".into()))?;
        let owned = pinned
            .metadata()
            .map_err(|_| ArenaError::Io("replay caller bundle directory metadata".into()))?;
        if !supplied.is_dir() || supplied.dev() != owned.dev() || supplied.ino() != owned.ino() {
            return Err(ArenaError::Integrity(
                "replay caller cwd is not prepared directory".into(),
            ));
        }
        let mut credit = ReadCredit {
            remaining: policy.verification_read_bytes,
        };
        verify_file(
            program,
            &expected
                .expectations
                .replay_expected
                .registered_artifacts
                .replay_binary,
            policy.maximum_binary_bytes,
            &mut credit,
            || check_clock(bundle, cancel, false),
        )?;
        verify_publication(bundle, directory, &mut credit, cancel, false)?;
        let args = [
            OsString::from("--expected-pins"),
            bundle
                .directory()
                .join("replay-expected.json")
                .into_os_string(),
            OsString::from("--expected-sha256"),
            OsString::from(&expected_payload.artifact().sha256),
            OsString::from("--expected-bytes"),
            OsString::from(expected_payload.artifact().bytes.to_string()),
            OsString::from("--asset-profile"),
            bundle
                .directory()
                .join("launch-assets.json")
                .into_os_string(),
        ];
        let watch = OwnedArtifactTreeWatch {
            max_total_bytes: MAX_PUBLICATION_BYTES as u64,
            max_file_bytes: MAX_PUBLICATION_BYTES as u64,
            max_files: 8,
            max_depth: 0,
        };
        let process = supervise_input_in_original_window(
            program,
            &args,
            directory,
            policy.process,
            Some(cancel),
            &watch,
            OriginalProcessInput {
                bytes: request.as_bytes(),
                window,
            },
        )?;
        // This finite readback also belongs to W. Never return a new Err after
        // spawn that would discard pending-child custody or captured output.
        let postflight_error =
            verify_publication(bundle, directory, &mut credit, cancel, true).err();
        Ok((process, postflight_error))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io::{Seek, SeekFrom, Write};

        #[test]
        fn replay_caller_descriptor_hash_preserves_offset_and_honors_read_credit_and_original_guard()
         {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "rz-replay-caller-pin-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut file = std::fs::OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            // Unlink this exact newly owned regular file immediately. A failed
            // assertion cannot leak the test artifact or delete a replacement.
            std::fs::remove_file(&path).unwrap();
            let bytes = vec![b'x'; 48 * 1024];
            file.write_all(&bytes).unwrap();
            file.seek(SeekFrom::Start(7)).unwrap();
            let pin = ArtifactPin {
                bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&bytes)),
            };
            let mut credit = ReadCredit {
                remaining: pin.bytes,
            };
            verify_file(&file, &pin, pin.bytes, &mut credit, || Ok(())).unwrap();
            assert_eq!(file.stream_position().unwrap(), 7);
            assert_eq!(credit.remaining, 0);
            assert!(verify_file(&file, &pin, pin.bytes, &mut credit, || Ok(())).is_err());
            let bad_pin = ArtifactPin {
                sha256: "0".repeat(64),
                ..pin.clone()
            };
            assert!(
                verify_file(
                    &file,
                    &bad_pin,
                    pin.bytes,
                    &mut ReadCredit {
                        remaining: pin.bytes
                    },
                    || Ok(())
                )
                .is_err()
            );
            let mut calls = 0;
            let mut credit = ReadCredit {
                remaining: pin.bytes,
            };
            assert!(
                verify_file(&file, &pin, pin.bytes, &mut credit, || {
                    calls += 1;
                    if calls >= 3 {
                        Err(ArenaError::Budget("test original E expired".into()))
                    } else {
                        Ok(())
                    }
                })
                .is_err()
            );
            assert_eq!(calls, 3);
            assert_eq!(credit.remaining, pin.bytes - 16 * 1024);
            assert_eq!(file.stream_position().unwrap(), 7);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ReplayCallerPolicy {
        ReplayCallerPolicy {
            process: ProcessLimits {
                wall_ms: 9000,
                shutdown_grace_ms: 500,
                max_output_bytes: 1024 * 1024,
                max_child_processes: 1,
            },
            maximum_binary_bytes: 64 * 1024 * 1024,
            verification_read_bytes: 128 * 1024 * 1024,
        }
    }

    #[test]
    fn replay_caller_policy_never_extends_the_registered_clock_output_or_worker_count() {
        assert!(check_resource_policy(10_000, 1000, 1024 * 1024, policy()).is_ok());
        for p in [
            ReplayCallerPolicy {
                process: ProcessLimits {
                    wall_ms: 10_000,
                    ..policy().process
                },
                ..policy()
            },
            ReplayCallerPolicy {
                process: ProcessLimits {
                    shutdown_grace_ms: 501,
                    ..policy().process
                },
                ..policy()
            },
            ReplayCallerPolicy {
                process: ProcessLimits {
                    max_output_bytes: 2 * 1024 * 1024,
                    ..policy().process
                },
                ..policy()
            },
            ReplayCallerPolicy {
                process: ProcessLimits {
                    max_child_processes: 2,
                    ..policy().process
                },
                ..policy()
            },
            ReplayCallerPolicy {
                maximum_binary_bytes: 0,
                ..policy()
            },
            ReplayCallerPolicy {
                maximum_binary_bytes: MAX_REPLAY_CALLER_BINARY_BYTES + 1,
                ..policy()
            },
            ReplayCallerPolicy {
                verification_read_bytes: 0,
                ..policy()
            },
            ReplayCallerPolicy {
                verification_read_bytes: MAX_REPLAY_CALLER_READ_BYTES + 1,
                ..policy()
            },
        ] {
            assert!(check_resource_policy(10_000, 1000, 1024 * 1024, p).is_err());
        }
        assert!(check_resource_policy(1000, 1000, 1024 * 1024, policy()).is_err());
        assert!(check_resource_policy(10_000, 0, 1024 * 1024, policy()).is_err());
    }
}
