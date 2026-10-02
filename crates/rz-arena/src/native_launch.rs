//! Exclusive, source-verified inputs for the bounded CPU NN integration pair.
//!
//! Copies have new inodes and closed writers before publication. Read-only Unix
//! permissions are an ownership convention, not an immutable OS seal or a
//! same-UID sandbox. The parent exclusively controls the output tree and trusts
//! the pinned native programs not to chmod, replace or modify these inputs.

use crate::ArenaError;
#[cfg(target_os = "linux")]
use crate::FastchessInvocation;
use rz_experiments::{ArtifactRef, LockedIntegrationPairSpecV1};
#[cfg(target_os = "linux")]
use rz_experiments::{NativeArtifactRole, NativeEngineRole};
use serde::Serialize;
use std::{
    ffi::OsString,
    fmt,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub const NATIVE_PAIR_METADATA_CAP: u64 = 64 * 1024;
const MAX_INNER_ARG_BYTES: usize = 16 * 1024;
static NATIVE_ACTIVE: AtomicBool = AtomicBool::new(false);
static NATIVE_ADMISSION_CLOSED: AtomicBool = AtomicBool::new(false);

/// One active lease bounds unresolved retention to one whole launch bundle.
pub(crate) struct NativeAdmissionLease {
    _private: (),
}
impl Drop for NativeAdmissionLease {
    fn drop(&mut self) {
        NATIVE_ACTIVE.store(false, Ordering::Release);
    }
}
impl NativeAdmissionLease {
    #[cfg(target_os = "linux")]
    fn acquire() -> Result<Self, ArenaError> {
        if NATIVE_ADMISSION_CLOSED.load(Ordering::Acquire) {
            return Err(ArenaError::Invalid(
                "native integration admission is closed".into(),
            ));
        }
        NATIVE_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                ArenaError::Budget("one native integration lease is already active".into())
            })?;
        let lease = Self { _private: () };
        if NATIVE_ADMISSION_CLOSED.load(Ordering::Acquire) {
            return Err(ArenaError::Invalid(
                "native integration admission closed during admission".into(),
            ));
        }
        Ok(lease)
    }
}
pub(crate) fn close_native_admission() {
    NATIVE_ADMISSION_CLOSED.store(true, Ordering::Release);
}
pub fn native_admission_closed() -> bool {
    NATIVE_ADMISSION_CLOSED.load(Ordering::Acquire)
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeSnapshotReceipt {
    pub artifact: ArtifactRef,
    pub snapshot_relative_path: String,
    pub bytes: u64,
    pub sha256: String,
    pub distinct_source_inode: bool,
    pub closed_writer_read_only: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativePreparationReceipt {
    pub receipt_version: u32,
    pub execution_ready: bool,
    pub input_sha256: String,
    pub output_directory: String,
    pub attempt_created: bool,
    pub attempted_snapshot_relative_paths: Vec<String>,
    pub completed_snapshots: Vec<NativeSnapshotReceipt>,
    pub child_spawned: bool,
    pub writers_closed: bool,
    pub input_pins_required: bool,
    pub original_error: String,
    pub subsequent_file_owner: String,
    pub automatic_retry: bool,
}
pub struct NativePreparationFailure {
    pub cause: ArenaError,
    pub receipt: NativePreparationReceipt,
    pub receipt_artifact: Option<ArtifactRef>,
    pub persistence_error: Option<ArenaError>,
}
impl fmt::Debug for NativePreparationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativePreparationFailure")
            .field("cause", &self.cause)
            .field("receipt", &self.receipt)
            .field("persistence_error", &self.persistence_error)
            .finish()
    }
}
impl fmt::Display for NativePreparationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "native preparation failed before spawn: {}; attempt_created={}; evidence_save_failed={}",
            self.cause,
            self.receipt.attempt_created,
            self.persistence_error.is_some()
        )
    }
}
impl std::error::Error for NativePreparationFailure {}

/// Owns every input pin, the exact cwd and both runtime output roots. There is
/// deliberately no automatic file deletion. A pending run must retain this
/// owner, even after a terminal leader if its process group remains unverified.
pub struct NativeLaunchOwner {
    pub(crate) spec: LockedIntegrationPairSpecV1,
    pub(crate) _lease: NativeAdmissionLease,
    #[cfg(target_os = "linux")]
    pub(crate) snapshot: linux::Snapshot,
}
impl fmt::Debug for NativeLaunchOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeLaunchOwner")
            .field("input_sha256", &self.spec.sha256())
            .finish_non_exhaustive()
    }
}
impl NativeLaunchOwner {
    pub fn spec(&self) -> &LockedIntegrationPairSpecV1 {
        &self.spec
    }
    #[cfg(target_os = "linux")]
    pub fn snapshots(&self) -> &[NativeSnapshotReceipt] {
        &self.snapshot.receipts
    }
    #[cfg(not(target_os = "linux"))]
    pub fn snapshots(&self) -> &[NativeSnapshotReceipt] {
        &[]
    }
}

/// Encode only the bounded subset supported by the pinned Fastchess
/// `argv_split` parser. This is a single `args=` outer argv element, not shell
/// quoting. Quotes and backslashes are rejected rather than ambiguously escaped.
pub fn encode_fastchess_native_args(tokens: &[OsString]) -> Result<OsString, ArenaError> {
    if tokens.is_empty() || tokens.len() > 32 {
        return Err(ArenaError::Invalid(
            "native engine arguments require 1..32 tokens".into(),
        ));
    }
    let mut encoded = String::from("args=");
    for (index, token) in tokens.iter().enumerate() {
        let value = token
            .to_str()
            .ok_or_else(|| ArenaError::Invalid("native argument is not UTF-8".into()))?;
        if value.is_empty()
            || value.len() > 4096
            || value
                .chars()
                .any(|c| c.is_control() || matches!(c, '\"' | '\'' | '\\'))
        {
            return Err(ArenaError::Invalid(
                "native argument contains unsupported delimiters or length".into(),
            ));
        }
        if index != 0 {
            encoded.push(' ');
        }
        encoded.push('\"');
        encoded.push_str(value);
        encoded.push('\"');
        if encoded.len() > MAX_INNER_ARG_BYTES {
            return Err(ArenaError::Budget(
                "native inner argument byte limit exceeded".into(),
            ));
        }
    }
    Ok(encoded.into())
}

/// Verify declaration and source bytes, then create an exclusive outside-Git
/// attempt. `output_directory` is one safe basename; existing attempts are never
/// reused. No engine or runner is started by this function.
pub fn prepare_native_launch(
    spec: &LockedIntegrationPairSpecV1,
    source_root: &Path,
    output_root: &Path,
    output_directory: &str,
) -> Result<NativeLaunchOwner, Box<NativePreparationFailure>> {
    let mut receipt=NativePreparationReceipt{receipt_version:1,execution_ready:false,input_sha256:spec.sha256().into(),
        output_directory:if safe_output_basename(output_directory){output_directory.into()}else{"rejected-basename".into()},
        attempt_created:false,attempted_snapshot_relative_paths:Vec::new(),completed_snapshots:Vec::new(),child_spawned:false,writers_closed:true,input_pins_required:false,original_error:String::new(),
        subsequent_file_owner:"caller retains exclusively owned outside-Git attempt; all writers are closed and no child was spawned, so input handles may be released; retained historical files are not a process-lifetime aggregate quota".into(),automatic_retry:false};
    #[cfg(target_os = "linux")]
    {
        let mut directory = None;
        let result: Result<NativeLaunchOwner, ArenaError> = (|| {
            let lease = NativeAdmissionLease::acquire()?;
            let snapshot = linux::prepare(
                spec,
                source_root,
                output_root,
                output_directory,
                &mut receipt,
                &mut directory,
            )?;
            Ok(NativeLaunchOwner {
                spec: spec.clone(),
                _lease: lease,
                snapshot,
            })
        })();
        match result {
            Ok(owner) => Ok(owner),
            Err(cause) => {
                receipt.original_error = cause.to_string().chars().take(4096).collect();
                let persisted = if receipt.attempt_created {
                    linux::save_preparation_failure(directory.as_ref(), &receipt)
                } else {
                    Ok(None)
                };
                let (receipt_artifact, persistence_error) = match persisted {
                    Ok(artifact) => (artifact, None),
                    Err(error) => (None, Some(error)),
                };
                Err(Box::new(NativePreparationFailure {
                    cause,
                    receipt,
                    receipt_artifact,
                    persistence_error,
                }))
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (spec, source_root, output_root, output_directory);
        let cause =
            ArenaError::Invalid("CPU NN integration runner currently requires Linux".into());
        receipt.original_error = cause.to_string();
        Err(Box::new(NativePreparationFailure {
            cause,
            receipt,
            receipt_artifact: None,
            persistence_error: None,
        }))
    }
}
fn safe_output_basename(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

#[cfg(target_os = "linux")]
pub(crate) mod linux {
    use super::*;
    use crate::{OwnedArtifactTreeWatch, ProcessLimits};
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        fs::{File, Permissions},
        io::{Read, Seek, SeekFrom, Write},
        os::unix::fs::{MetadataExt, PermissionsExt},
        path::PathBuf,
    };

    pub(crate) struct InputPin {
        pub artifact: ArtifactRef,
        pub path: PathBuf,
        pub file: File,
    }
    pub(crate) struct Snapshot {
        pub directory: Dir,
        pub input_directory: Dir,
        pub cwd: File,
        pub path: PathBuf,
        pub output_directory: String,
        pub pins: Vec<InputPin>,
        pub receipts: Vec<NativeSnapshotReceipt>,
        pub runner_index: usize,
        pub invocation: FastchessInvocation,
        pub watch: OwnedArtifactTreeWatch,
        pub limits: ProcessLimits,
        pub _runtime_root_pins: [File; 2],
    }

    pub(crate) fn pin<'a>(
        pins: &'a [InputPin],
        artifact: &ArtifactRef,
    ) -> Result<&'a InputPin, ArenaError> {
        pins.iter()
            .find(|pin| &pin.artifact == artifact)
            .ok_or_else(|| {
                ArenaError::Integrity("declared native input lacks its owned snapshot".into())
            })
    }
    fn io(detail: &'static str) -> ArenaError {
        ArenaError::Io(detail.into())
    }
    // cap_std's directory capability may use O_PATH. That descriptor can pin
    // cwd/path authority, but Linux fchmod rejects it with EBADF. Open a real
    // readable directory descriptor relative to the same capability instead.
    fn readable_directory_pin(directory: &Dir) -> Result<File, ArenaError> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = directory
            .open_with(".", &options)
            .map_err(|_| io("cannot open readable native directory permissions pin"))?
            .into_std();
        if !file
            .metadata()
            .map_err(|_| io("cannot inspect readable native directory permissions pin"))?
            .is_dir()
        {
            return Err(ArenaError::Integrity(
                "native directory permissions pin is not a directory".into(),
            ));
        }
        Ok(file)
    }
    fn outside_git(path: &Path) -> Result<PathBuf, ArenaError> {
        let metadata =
            std::fs::symlink_metadata(path).map_err(|_| io("native output root unavailable"))?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(ArenaError::Invalid(
                "native output root must be a real owned directory".into(),
            ));
        }
        let path = path
            .canonicalize()
            .map_err(|_| io("cannot resolve native output root"))?;
        for ancestor in path.ancestors() {
            match std::fs::symlink_metadata(ancestor.join(".git")) {
                Ok(_) => {
                    return Err(ArenaError::Invalid(
                        "native outputs must be outside Git checkouts".into(),
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err(io("cannot check native output ownership boundary")),
            }
        }
        Ok(path)
    }
    pub(super) fn save_preparation_failure(
        directory: Option<&Dir>,
        receipt: &NativePreparationReceipt,
    ) -> Result<Option<ArtifactRef>, ArenaError> {
        let directory = directory.ok_or_else(|| {
            io("failed attempt could not retain a directory capability for evidence saving")
        })?;
        let bytes = serde_json::to_vec_pretty(receipt)
            .map_err(|_| io("cannot encode preparation failure receipt"))?;
        if bytes.len() as u64 > NATIVE_PAIR_METADATA_CAP {
            return Err(ArenaError::Budget(
                "preparation failure receipt exceeds 64KiB".into(),
            ));
        }
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
        let mut file = directory
            .open_with("native-preparation-failure.v1.json", &options)
            .map_err(|_| {
                io("cannot reserve preparation failure receipt; partial attempt preserved")
            })?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| {
                io("cannot finish preparation failure receipt; partial bytes preserved")
            })?;
        Ok(Some(ArtifactRef {
            path: format!(
                "{}/native-preparation-failure.v1.json",
                receipt.output_directory
            ),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            bytes: bytes.len() as u64,
            source: "RoveZero CPU NN integration preparation evidence".into(),
            license: "MIT execution evidence; external input rights remain separate".into(),
        }))
    }
    pub(crate) fn readonly_copy(
        artifact: &ArtifactRef,
        source_root: &Path,
        directory: &Dir,
        relative: &str,
        absolute: PathBuf,
        executable: bool,
        max_bytes: u64,
    ) -> Result<(InputPin, NativeSnapshotReceipt), ArenaError> {
        let mut source = artifact.open_verified(source_root, max_bytes)?;
        let source_metadata = source
            .metadata()
            .map_err(|_| io("cannot inspect verified source inode"))?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
        let mut writer = directory
            .open_with(relative, &options)
            .map_err(|_| io("cannot create exclusive native input copy"))?
            .into_std();
        let mut count = 0u64;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 16 * 1024];
        loop {
            let read = source
                .read(&mut buffer)
                .map_err(|_| io("cannot read verified native input"))?;
            if read == 0 {
                break;
            }
            count = count
                .checked_add(read as u64)
                .ok_or_else(|| ArenaError::Budget("native copy length overflow".into()))?;
            if count > artifact.bytes || count > max_bytes {
                return Err(ArenaError::Integrity(
                    "native input changed or exceeded copy budget".into(),
                ));
            }
            hasher.update(&buffer[..read]);
            writer
                .write_all(&buffer[..read])
                .map_err(|_| io("cannot write native input copy"))?;
        }
        if count != artifact.bytes || format!("{:x}", hasher.finalize()) != artifact.sha256 {
            return Err(ArenaError::Integrity(
                "native input changed while copying".into(),
            ));
        }
        writer
            .sync_all()
            .map_err(|_| io("cannot sync native input copy"))?;
        let copied_metadata = writer
            .metadata()
            .map_err(|_| io("cannot inspect native copy inode"))?;
        let distinct = (source_metadata.dev(), source_metadata.ino())
            != (copied_metadata.dev(), copied_metadata.ino());
        if !distinct {
            return Err(ArenaError::Integrity(
                "native copy shares source inode".into(),
            ));
        }
        drop(writer);
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let mut file = directory
            .open_with(relative, &options)
            .map_err(|_| io("cannot pin native copy"))?
            .into_std();
        file.set_permissions(Permissions::from_mode(if executable {
            0o500
        } else {
            0o400
        }))
        .map_err(|_| io("cannot make native input copy read-only"))?;
        verify_copy(&mut file, artifact)?;
        Ok((
            InputPin {
                artifact: artifact.clone(),
                path: absolute,
                file,
            },
            NativeSnapshotReceipt {
                artifact: artifact.clone(),
                snapshot_relative_path: format!("inputs/{relative}"),
                bytes: artifact.bytes,
                sha256: artifact.sha256.clone(),
                distinct_source_inode: distinct,
                closed_writer_read_only: true,
            },
        ))
    }
    pub(crate) fn verify_copy(file: &mut File, artifact: &ArtifactRef) -> Result<(), ArenaError> {
        file.seek(SeekFrom::Start(0))
            .map_err(|_| io("cannot rewind native copy"))?;
        let mut hasher = Sha256::new();
        let mut count = 0u64;
        let mut buffer = [0u8; 16 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|_| io("cannot recheck native copy"))?;
            if read == 0 {
                break;
            }
            count = count
                .checked_add(read as u64)
                .ok_or_else(|| ArenaError::Budget("native pin length overflow".into()))?;
            if count > artifact.bytes {
                return Err(ArenaError::Integrity("native copy grew".into()));
            }
            hasher.update(&buffer[..read]);
        }
        if count != artifact.bytes || format!("{:x}", hasher.finalize()) != artifact.sha256 {
            return Err(ArenaError::Integrity(
                "native copy digest or length differs".into(),
            ));
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|_| io("cannot rewind checked native copy"))?;
        Ok(())
    }

    pub(crate) fn prepare(
        spec: &LockedIntegrationPairSpecV1,
        source_root: &Path,
        output_root: &Path,
        label: &str,
        trace: &mut NativePreparationReceipt,
        owned_directory: &mut Option<Dir>,
    ) -> Result<Snapshot, ArenaError> {
        if !safe_output_basename(label) {
            return Err(ArenaError::Invalid(
                "native attempt requires a safe unique basename".into(),
            ));
        }
        let input = spec.input();
        let opening = crate::opening_pgn_for_spec(&input.opening, input.max_plies)?;
        if opening.len() as u64 != input.opening_artifact.bytes
            || format!("{:x}", Sha256::digest(opening.as_bytes())) != input.opening_artifact.sha256
        {
            return Err(ArenaError::Integrity(
                "opening artifact differs from complete A-validated trace serialization".into(),
            ));
        }
        let total = spec.unique_input_bytes()?;
        if total > input.budget.max_input_bytes {
            return Err(ArenaError::Budget("native input aggregate exceeded".into()));
        }
        let mut unique = BTreeMap::new();
        for artifact in spec.declared_artifacts() {
            if let Some(prior) = unique.insert(artifact.path.clone(), artifact.clone())
                && prior != *artifact
            {
                return Err(ArenaError::Integrity(
                    "same native source path has conflicting metadata".into(),
                ));
            }
        }
        // Verify every source before creating the attempt; subsequently copied
        // bytes are hashed again, so same-inode writes cannot publish torn input.
        for artifact in unique.values() {
            drop(artifact.open_verified(source_root, input.budget.max_input_bytes)?);
        }
        let root_path = outside_git(output_root)?;
        let root = Dir::open_ambient_dir(&root_path, cap_std::ambient_authority())
            .map_err(|_| io("cannot pin native output root"))?;
        root.create_dir(label).map_err(|_| {
            io("cannot create exclusive native attempt; existing outputs preserved")
        })?;
        trace.attempt_created = true;
        let directory = root
            .open_dir_nofollow(label)
            .map_err(|_| io("cannot pin native attempt"))?;
        *owned_directory = Some(
            directory
                .try_clone()
                .map_err(|_| io("cannot retain preparation evidence directory"))?,
        );
        readable_directory_pin(&directory)?
            .set_permissions(Permissions::from_mode(0o700))
            .map_err(|_| io("cannot make native attempt private"))?;
        directory
            .create_dir("inputs")
            .map_err(|_| io("cannot create native input directory"))?;
        let inputs = directory
            .open_dir_nofollow("inputs")
            .map_err(|_| io("cannot pin native inputs directory"))?;
        readable_directory_pin(&inputs)?
            .set_permissions(Permissions::from_mode(0o700))
            .map_err(|_| io("cannot make native input directory private"))?;
        let path = root_path.join(label);
        let mut pins = Vec::new();
        let mut receipts = Vec::new();
        for (index, artifact) in unique.values().enumerate() {
            let name = format!("pin-{index:02}");
            trace
                .attempted_snapshot_relative_paths
                .push(format!("inputs/{name}"));
            let executable = artifact == &input.runner.binary
                || input.engines.iter().any(|e| {
                    e.artifact(NativeArtifactRole::Binary)
                        .is_ok_and(|a| a == artifact)
                });
            let (pin, receipt) = readonly_copy(
                artifact,
                source_root,
                &inputs,
                &name,
                path.join("inputs").join(&name),
                executable,
                input.budget.max_input_bytes,
            )?;
            trace.completed_snapshots.push(receipt.clone());
            pins.push(pin);
            receipts.push(receipt);
        }
        readable_directory_pin(&inputs)?
            .set_permissions(Permissions::from_mode(0o500))
            .map_err(|_| io("cannot close native inputs directory to writes"))?;
        let runner_index = pins
            .iter()
            .position(|p| p.artifact == input.runner.binary)
            .ok_or_else(|| ArenaError::Integrity("runner snapshot absent".into()))?;
        let runtime_roots = [
            path.join("baseline-runtime"),
            path.join("candidate-runtime"),
        ];
        let mut runtime_root_pins = Vec::new();
        for name in ["baseline-runtime", "candidate-runtime"] {
            directory
                .create_dir(name)
                .map_err(|_| io("cannot create private role runtime root"))?;
            let role_directory = directory
                .open_dir_nofollow(name)
                .map_err(|_| io("cannot pin role runtime root"))?;
            let pin = readable_directory_pin(&role_directory)?;
            pin.set_permissions(Permissions::from_mode(0o700))
                .map_err(|_| io("cannot make role runtime root private"))?;
            runtime_root_pins.push(pin);
        }
        let cwd = directory
            .try_clone()
            .map_err(|_| io("cannot clone native cwd"))?
            .into_std_file();
        let invocation = build_invocation(spec, &pins, &path, &runtime_roots)?;
        let watch_bytes = total
            .checked_add(input.budget.max_runtime_bytes)
            .and_then(|n| n.checked_add(input.budget.max_output_bytes))
            .ok_or_else(|| ArenaError::Budget("native watched byte overflow".into()))?;
        if watch_bytes > input.budget.max_artifact_bytes {
            return Err(ArenaError::Budget(
                "native artifact budget needs input/runtime/output reservations".into(),
            ));
        }
        let stream_cap = input
            .budget
            .max_output_bytes
            .checked_sub(NATIVE_PAIR_METADATA_CAP)
            .and_then(|n| n.checked_div(2))
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                ArenaError::Budget(
                    "native output budget needs receipt, PGN/config and stream reservations".into(),
                )
            })?;
        let max_files = usize::try_from(input.budget.max_runtime_files)
            .map_err(|_| ArenaError::Budget("native file count cannot fit host".into()))?
            .checked_add(pins.len())
            .and_then(|n| n.checked_add(8))
            .ok_or_else(|| ArenaError::Budget("native file count overflow".into()))?;
        Ok(Snapshot {
            directory,
            input_directory: inputs,
            cwd,
            path,
            output_directory: label.into(),
            pins,
            receipts,
            runner_index,
            invocation,
            watch: OwnedArtifactTreeWatch {
                max_total_bytes: watch_bytes,
                max_file_bytes: input
                    .budget
                    .max_input_bytes
                    .max(input.budget.max_runtime_bytes)
                    .max(input.budget.max_output_bytes),
                max_files,
                max_depth: usize::from(input.budget.max_runtime_depth)
                    .checked_add(1)
                    .ok_or_else(|| ArenaError::Budget("native tree depth overflow".into()))?,
            },
            limits: ProcessLimits {
                wall_ms: input.timeouts.runtime_ms,
                shutdown_grace_ms: input.timeouts.shutdown_ms,
                max_output_bytes: stream_cap,
                max_child_processes: input.budget.max_child_processes,
            },
            _runtime_root_pins: runtime_root_pins.try_into().map_err(|_| {
                ArenaError::Integrity("native runtime root pin count differs".into())
            })?,
        })
    }

    fn key_path(key: &str, path: &Path) -> Result<OsString, ArenaError> {
        let value = path
            .to_str()
            .ok_or_else(|| ArenaError::Invalid("native path is not UTF-8".into()))?;
        if !path.is_absolute() || value.len() > 4096 || value.chars().any(char::is_control) {
            return Err(ArenaError::Invalid(
                "native path must be a bounded absolute path".into(),
            ));
        }
        Ok(format!("{key}{value}").into())
    }
    pub(crate) fn build_invocation(
        spec: &LockedIntegrationPairSpecV1,
        pins: &[InputPin],
        cwd: &Path,
        roots: &[PathBuf; 2],
    ) -> Result<FastchessInvocation, ArenaError> {
        let input = spec.input();
        if input.runner.source_url != crate::FASTCHESS_SOURCE_URL
            || input.runner.source_commit != crate::FASTCHESS_SOURCE_COMMIT
            || input.runner.version != crate::FASTCHESS_VERSION
            || input.runner.dirty
            || input.runner.dirty_patch.is_some()
        {
            return Err(ArenaError::Invalid(
                "unsupported or dirty native integration Fastchess".into(),
            ));
        }
        let prefix = u32::try_from(input.opening.moves.len())
            .map_err(|_| ArenaError::Budget("native opening ply overflow".into()))?;
        let remaining = input
            .max_plies
            .checked_sub(prefix)
            .filter(|n| *n > 0 && *n % 2 == 0)
            .ok_or_else(|| {
                ArenaError::Invalid(
                    "native pair requires positive even searched plies after opening".into(),
                )
            })?;
        let mut args = Vec::new();
        for role in input.white_order {
            let engine = input.engine(role)?;
            let binary = pin(pins, engine.artifact(NativeArtifactRole::Binary)?)?;
            let root = &roots[match role {
                NativeEngineRole::Baseline => 0,
                NativeEngineRole::Candidate => 1,
            }];
            let mut tokens = vec![
                OsString::from("--onnx-cpu"),
                OsString::from("--attestation"),
            ];
            for (flag, kind) in [
                ("--source-weights=", NativeArtifactRole::SourceWeights),
                ("--onnx-model=", NativeArtifactRole::Onnx),
                ("--export-manifest=", NativeArtifactRole::ExportManifest),
                ("--ort-library=", NativeArtifactRole::OrtLibrary),
            ] {
                tokens.push(key_path(flag, &pin(pins, engine.artifact(kind)?)?.path)?);
            }
            tokens.push(
                format!(
                    "--manifest-sha256={}",
                    engine.artifact(NativeArtifactRole::ExportManifest)?.sha256
                )
                .into(),
            );
            tokens.push(
                format!(
                    "--ort-sha256={}",
                    engine.artifact(NativeArtifactRole::OrtLibrary)?.sha256
                )
                .into(),
            );
            tokens.push(key_path("--output-root=", root)?);
            args.extend([
                OsString::from("-engine"),
                key_path("cmd=", &binary.path)?,
                key_path("dir=", cwd)?,
                format!("name={}", engine.engine_id).into(),
                encode_fastchess_native_args(&tokens)?,
            ]);
        }
        let opening = &pin(pins, &input.opening_artifact)?.path;
        args.extend([
            "-each".into(),
            "proto=uci".into(),
            format!(
                "st={}.{:03}",
                input.clock.movetime_ms / 1000,
                input.clock.movetime_ms % 1000
            )
            .into(),
            "restart=on".into(),
            "timemargin=0".into(),
            "-openings".into(),
            key_path("file=", opening)?,
            "format=pgn".into(),
            "order=sequential".into(),
            format!("plies={prefix}").into(),
            "start=1".into(),
            "-rounds".into(),
            "1".into(),
            "-games".into(),
            "2".into(),
            "-repeat".into(),
            "-concurrency".into(),
            "1".into(),
            "-srand".into(),
            "1".into(),
            "-maxmoves".into(),
            (remaining / 2).to_string().into(),
            "-pgnout".into(),
            key_path("file=", &cwd.join("match.pgn"))?,
            "append=false".into(),
            "notation=uci".into(),
            "min=false".into(),
            "timeleft=true".into(),
            "latency=true".into(),
            "-autosaveinterval".into(),
            "0".into(),
            "-ratinginterval".into(),
            "0".into(),
            "-startup-ms".into(),
            input.timeouts.startup_ms.to_string().into(),
            "-ping-ms".into(),
            input.timeouts.handshake_ms.to_string().into(),
            "-ucinewgame-ms".into(),
            input.timeouts.drain_ms.to_string().into(),
            "-strict".into(),
            "-log".into(),
            "file=/proc/self/fd/1".into(),
            "append=true".into(),
            "level=trace".into(),
            "realtime=true".into(),
            "engine=true".into(),
        ]);
        Ok(FastchessInvocation{args,limitations:[
            "CPU NN integration only; execution_ready=false; strength_eligible=false; same weights and search in both roles",
            "private input copies use separate inodes and closed writers; Unix readonly is an ownership convention, not a same-UID sandbox or immutable seal",
            "trusted pinned programs and parent exclusively own snapshots and runtime output ancestors throughout execution and pending retention",
            "Fastchess movetime and its internal 100ms read margin do not attest the shared whole-engine clock contract",
            "Fastchess automatic draw rules differ from explicit A claims; PGN audit is independent and maxmoves draws remain incomplete",
            "owned-tree/process snapshots are observed limits, not aggregate RAM or kernel disk quotas; escaped/transient children are outside the process-group guarantee",
            "per-process address-space declaration requires separately recorded inherited enforcement; this library does not install RAM/CPU affinity limits",
            "process cleanup alone is not physical NN drain; matched bounded B startup/final provider records are required separately",
        ].map(String::from).to_vec()})
    }
}
