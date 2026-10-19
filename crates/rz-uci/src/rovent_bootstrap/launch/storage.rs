//! Independent opt-in storage admission; no CLI/session currently calls it.
//!
//! The original lock, verified profile/source descriptors and actual directory
//! capability stay together. This starts no model, worker or Store archive IO.
//! Source bytes are verified, but their compiled meaning is deliberately not
//! attested here; the eventual launcher must compare the actual compiled pins.

use crate::rovent_bootstrap::RoventStartupControl;
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_experiments::{ArtifactRef, ManifestError, RoventInputLock, RoventStorageProfile};
use rz_search::pals::store::{ArchiveRootCapability, StoreError};
#[cfg(target_os = "linux")]
use sha2::{Digest as _, Sha256};
use std::{
    error::Error,
    fmt,
    fs::File,
    path::{Component, Path, PathBuf},
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

// Reuse the existing persisted-metadata bound, not an archive runtime selector.
const MAX_INPUT_BYTES: u64 = rz_experiments::MAX_MANIFEST_BYTES as u64;
const MAX_ROOT_PATH_BYTES: usize = 4096;
const ARCHIVE: usize = 0;
const CHECKOUT: usize = 1;
const PRIVATE: usize = 2;
const PROTECTED: usize = 3;

/// Explicit borrowed invocation inputs. No path, raw pin or deadline defaults.
pub struct RoventStorageAdmissionRequest<'a> {
    pub profile_path: &'a str,
    pub profile_sha256: &'a str,
    pub profile_bytes: u64,
    pub archive_root: &'a Path,
    pub checkout_root: &'a Path,
    pub private_snapshot_root: &'a Path,
    pub protected_execution_root: &'a Path,
}

/// Read-only admission custody, not a physical fence or an enabled archive.
/// This owner is intentionally not Clone. Partial failures retain the same
/// original successful descriptor/directory opens rather than reopening them.
pub struct RoventStorageAdmissionOwners {
    input_lock: Arc<RoventInputLock>,
    control: RoventStartupControl,
    artifact: Option<ArtifactRef>,
    profile: Option<RoventStorageProfile>,
    profile_file: Option<VerifiedInput>,
    source_files: [Option<VerifiedInput>; 3],
    directories: [Option<HeldDirectory>; 4],
    capability: Option<ArchiveRootCapability>,
}

impl RoventStorageAdmissionOwners {
    pub fn input_lock(&self) -> &Arc<RoventInputLock> {
        &self.input_lock
    }
    pub fn profile_artifact(&self) -> Option<&ArtifactRef> {
        self.artifact.as_ref()
    }
    pub fn profile(&self) -> Option<&RoventStorageProfile> {
        self.profile.as_ref()
    }
    pub fn profile_file(&self) -> Option<&File> {
        self.profile_file.as_ref().map(|held| &held.file)
    }
    /// 0=storage source, 1=codec, 2=launch. An absent file is never verified.
    pub fn source_file(&self, index: usize) -> Option<&File> {
        self.source_files
            .get(index)
            .and_then(Option::as_ref)
            .map(|held| &held.file)
    }
    pub fn verified_source_files(&self) -> usize {
        self.source_files
            .iter()
            .flatten()
            .filter(|held| held.revalidated)
            .count()
    }
    pub fn retains_root_capability(&self) -> bool {
        self.capability.is_some()
    }
}

/// Move-only byte/directory custody only, with compiled namespace still
/// unverified. This cannot configure/start an archive: a separate Root-owned
/// final compiled/source comparison and checked runtime bridge are required.
/// No such runtime bridge is provided by this independent library entry.
pub struct RoventStorageByteCustody {
    owners: RoventStorageAdmissionOwners,
}

impl RoventStorageByteCustody {
    pub fn owners(&self) -> &RoventStorageAdmissionOwners {
        &self.owners
    }
    pub fn input_lock_sha256(&self) -> &str {
        &self.owners.input_lock.canonical_sha256
    }
    pub fn profile(&self) -> &RoventStorageProfile {
        self.owners.profile.as_ref().expect("admitted profile")
    }
    pub fn retains_root_capability(&self) -> bool {
        self.owners.capability.is_some()
    }
    /// Recheck only within the retained original startup control. This cannot
    /// renew admission or replace Search's later per-operation root checks.
    pub fn revalidate_startup(&self) -> Result<(), RoventStorageAdmissionError> {
        check(&self.owners.control)?;
        for input in self
            .owners
            .profile_file
            .iter()
            .chain(self.owners.source_files.iter().flatten())
        {
            read_rechecked(input, &self.owners.control, false)?;
        }
        for directory in self.owners.directories.iter().flatten() {
            directory.revalidate(&self.owners.control)?;
        }
        require_separation(&self.owners)?;
        check(&self.owners.control)
    }
    pub fn into_owners(self) -> RoventStorageAdmissionOwners {
        self.owners
    }
}

#[derive(Debug)]
pub enum RoventStorageAdmissionError {
    UnsupportedPlatform,
    Cancelled,
    Expired,
    Invalid(&'static str),
    Allocation,
    Input(ManifestError),
    Root(StoreError),
    Io {
        stage: &'static str,
        source: std::io::Error,
    },
}

impl fmt::Display for RoventStorageAdmissionError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(
                output,
                "UnsupportedCapability: native storage admission requires Linux"
            ),
            Self::Cancelled => write!(output, "storage admission canceled"),
            Self::Expired => write!(output, "original storage startup deadline expired"),
            Self::Invalid(reason) => write!(output, "storage admission: {reason}"),
            Self::Allocation => write!(output, "storage admission allocation failed"),
            Self::Input(error) => write!(output, "storage original input: {error}"),
            Self::Root(error) => write!(output, "storage root capability: {error}"),
            Self::Io { stage, source } => write!(output, "storage {stage}: {source}"),
        }
    }
}
impl Error for RoventStorageAdmissionError {}

/// The exact original lock and each successfully opened owner survive failure.
pub struct RoventStorageAdmissionFailure {
    pub error: RoventStorageAdmissionError,
    pub owners: RoventStorageAdmissionOwners,
}

impl fmt::Debug for RoventStorageAdmissionFailure {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output
            .debug_struct("RoventStorageAdmissionFailure")
            .field("error", &self.error)
            .field("profile_fd_retained", &self.owners.profile_file.is_some())
            .field("source_fds_retained", &self.owners.verified_source_files())
            .field(
                "root_capability_retained",
                &self.owners.capability.is_some(),
            )
            .finish()
    }
}

/// Library entry only: no existing CLI, startup assembly or session selects it.
/// The caller supplies the original Arc lock, actual private Dir and original
/// startup control. No renewed or cleanup deadline is used for admission reads.
#[allow(clippy::result_large_err)]
pub fn open_rovent_storage_byte_custody(
    input_lock: Arc<RoventInputLock>,
    private_root: &Dir,
    request: RoventStorageAdmissionRequest<'_>,
    control: RoventStartupControl,
) -> Result<RoventStorageByteCustody, RoventStorageAdmissionFailure> {
    let mut owners = RoventStorageAdmissionOwners {
        input_lock,
        control,
        artifact: None,
        profile: None,
        profile_file: None,
        source_files: [None, None, None],
        directories: [None, None, None, None],
        capability: None,
    };
    let result = admit(&mut owners, private_root, request);
    match result {
        Ok(()) => Ok(RoventStorageByteCustody { owners }),
        Err(error) => Err(RoventStorageAdmissionFailure { error, owners }),
    }
}

fn admit(
    owners: &mut RoventStorageAdmissionOwners,
    private_root: &Dir,
    request: RoventStorageAdmissionRequest<'_>,
) -> Result<(), RoventStorageAdmissionError> {
    if !cfg!(target_os = "linux") {
        return Err(RoventStorageAdmissionError::UnsupportedPlatform);
    }
    check(&owners.control)?;
    if owners.control.cleanup_until < owners.control.until
        || !rz_experiments::pals_private_snapshot_path_v5(request.profile_path)
        || request.profile_bytes == 0
        || request.profile_bytes > MAX_INPUT_BYTES
    {
        return Err(invalid(
            "original bounded profile/control selection is required",
        ));
    }
    owners.input_lock.validate().map_err(input)?;
    let artifact = unique_provenance(&owners.input_lock, request.profile_path)?.clone();
    owners.artifact = Some(artifact.clone());
    if artifact.sha256 != request.profile_sha256 || artifact.bytes != request.profile_bytes {
        return Err(invalid(
            "explicit profile raw pin differs from original provenance",
        ));
    }
    let raw = retain_verified_input(
        &mut owners.profile_file,
        private_root,
        artifact,
        &owners.control,
        true,
    )?;
    let text = std::str::from_utf8(&raw).map_err(|_| invalid("profile is not UTF-8 JSON"))?;
    let profile = RoventStorageProfile::from_json(text).map_err(input)?;
    owners.profile = Some(profile);
    owners
        .profile
        .as_ref()
        .unwrap()
        .validate_for_manifest(&owners.input_lock.manifest)
        .map_err(input)?;

    let declared_sources = &owners.profile.as_ref().unwrap().source_pins;
    let sources = [
        declared_sources.source.clone(),
        declared_sources.codec.clone(),
        declared_sources.launch.clone(),
    ];
    for (index, artifact) in sources.into_iter().enumerate() {
        check(&owners.control)?;
        if unique_provenance(&owners.input_lock, &artifact.path)? != &artifact {
            return Err(invalid(
                "source pin metadata differs from original provenance",
            ));
        }
        retain_verified_input(
            &mut owners.source_files[index],
            private_root,
            artifact,
            &owners.control,
            false,
        )?;
    }
    // Capture every named boundary without following a component. These are
    // metadata/root capabilities only, never a new archive game or segment.
    for (index, path) in [
        request.archive_root,
        request.checkout_root,
        request.private_snapshot_root,
        request.protected_execution_root,
    ]
    .into_iter()
    .enumerate()
    {
        owners.directories[index] = Some(open_directory(path, &owners.control)?);
    }
    if owners.directories[PRIVATE].as_ref().unwrap().identity != directory_identity(private_root)? {
        return Err(invalid(
            "private snapshot path differs from the original held Dir",
        ));
    }
    require_separation(owners)?;
    let archive = owners.directories[ARCHIVE].as_ref().unwrap();
    let directory = archive
        .directory
        .try_clone()
        .map_err(|source| io("archive capability clone", source))?;
    owners.capability = Some(
        ArchiveRootCapability::from_directory(archive.path.clone(), directory)
            .map_err(RoventStorageAdmissionError::Root)?,
    );
    // Revalidate the same successfully held inputs after all opens, including
    // growth/EOF and root-name replacement races. No compiled match is emitted.
    for held in owners
        .profile_file
        .iter()
        .chain(owners.source_files.iter().flatten())
    {
        read_rechecked(held, &owners.control, false)?;
    }
    for directory in owners.directories.iter().flatten() {
        directory.revalidate(&owners.control)?;
    }
    check(&owners.control)
}

fn unique_provenance<'a>(
    lock: &'a RoventInputLock,
    path: &str,
) -> Result<&'a ArtifactRef, RoventStorageAdmissionError> {
    let mut matching = lock
        .manifest
        .provenance
        .iter()
        .filter(|artifact| artifact.path == path);
    let artifact = matching
        .next()
        .ok_or_else(|| invalid("input is absent from original provenance"))?;
    if matching.next().is_some() {
        return Err(invalid("input path is not unique in original provenance"));
    }
    Ok(artifact)
}

struct VerifiedInput {
    artifact: ArtifactRef,
    file: File,
    identity: Option<FileIdentity>,
    revalidated: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    links: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

fn retain_verified_input(
    destination: &mut Option<VerifiedInput>,
    root: &Dir,
    artifact: ArtifactRef,
    control: &RoventStartupControl,
    collect: bool,
) -> Result<Vec<u8>, RoventStorageAdmissionError> {
    check(control)?;
    let file = artifact
        .open_verified_at(root, MAX_INPUT_BYTES)
        .map_err(input)?;
    // ArtifactRef has returned the actual verified descriptor. Retain it
    // before every further check, so cancellation/partial rejection cannot
    // silently discard a successfully acquired original owner.
    *destination = Some(VerifiedInput {
        artifact,
        file,
        identity: None,
        revalidated: false,
    });
    let held = destination.as_mut().unwrap();
    held.identity = Some(file_identity(&held.file)?);
    let raw = read_rechecked(held, control, collect)?;
    held.revalidated = true;
    Ok(raw)
}

#[cfg(target_os = "linux")]
fn file_identity(file: &File) -> Result<FileIdentity, RoventStorageAdmissionError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file
        .metadata()
        .map_err(|source| io("original input metadata", source))?;
    if !metadata.is_file() {
        return Err(invalid("original input descriptor is not regular"));
    }
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        links: metadata.nlink(),
        modified: (metadata.mtime(), metadata.mtime_nsec()),
        changed: (metadata.ctime(), metadata.ctime_nsec()),
    })
}
#[cfg(not(target_os = "linux"))]
fn file_identity(_: &File) -> Result<FileIdentity, RoventStorageAdmissionError> {
    Err(RoventStorageAdmissionError::UnsupportedPlatform)
}

#[cfg(target_os = "linux")]
fn read_rechecked(
    input: &VerifiedInput,
    control: &RoventStartupControl,
    collect: bool,
) -> Result<Vec<u8>, RoventStorageAdmissionError> {
    use std::os::unix::fs::FileExt;
    check(control)?;
    let before = file_identity(&input.file)?;
    if Some(before) != input.identity
        || before.bytes != input.artifact.bytes
        || before.bytes == 0
        || before.bytes > MAX_INPUT_BYTES
    {
        return Err(invalid(
            "original descriptor identity or byte bound changed",
        ));
    }
    let mut raw = Vec::new();
    if collect {
        raw.try_reserve_exact(
            usize::try_from(before.bytes).map_err(|_| RoventStorageAdmissionError::Allocation)?,
        )
        .map_err(|_| RoventStorageAdmissionError::Allocation)?;
    }
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut offset = 0u64;
    while offset < before.bytes {
        check(control)?;
        let length = (before.bytes - offset).min(buffer.len() as u64) as usize;
        let read = input
            .file
            .read_at(&mut buffer[..length], offset)
            .map_err(|source| io("original descriptor read", source))?;
        if read == 0 {
            return Err(invalid("original descriptor ended before its raw pin"));
        }
        hash.update(&buffer[..read]);
        if collect {
            raw.extend_from_slice(&buffer[..read]);
        }
        offset += read as u64;
    }
    check(control)?;
    let mut tail = [0u8; 1];
    let tail_len = input
        .file
        .read_at(&mut tail, offset)
        .map_err(|source| io("original descriptor EOF", source))?;
    if tail_len != 0
        || file_identity(&input.file)? != before
        || format!("{:x}", hash.finalize()) != input.artifact.sha256
    {
        return Err(invalid("original descriptor bytes/identity/EOF changed"));
    }
    check(control)?;
    Ok(raw)
}
#[cfg(not(target_os = "linux"))]
fn read_rechecked(
    input: &VerifiedInput,
    _: &RoventStartupControl,
    _: bool,
) -> Result<Vec<u8>, RoventStorageAdmissionError> {
    let _ = (&input.artifact, &input.identity);
    Err(RoventStorageAdmissionError::UnsupportedPlatform)
}

struct HeldDirectory {
    path: PathBuf,
    directory: Dir,
    identity: (u64, u64),
    ancestors: Vec<(u64, u64)>,
}
impl HeldDirectory {
    fn revalidate(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), RoventStorageAdmissionError> {
        let named = open_directory(&self.path, control)?;
        if named.identity != self.identity
            || named.ancestors != self.ancestors
            || directory_identity(&self.directory)? != self.identity
        {
            return Err(invalid("held directory was renamed or replaced"));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn directory_identity(directory: &Dir) -> Result<(u64, u64), RoventStorageAdmissionError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = directory
        .try_clone()
        .map_err(|source| io("directory clone", source))?
        .into_std_file()
        .metadata()
        .map_err(|source| io("directory metadata", source))?;
    if !metadata.is_dir() {
        return Err(invalid("held root is not a directory"));
    }
    Ok((metadata.dev(), metadata.ino()))
}
#[cfg(not(target_os = "linux"))]
fn directory_identity(_: &Dir) -> Result<(u64, u64), RoventStorageAdmissionError> {
    Err(RoventStorageAdmissionError::UnsupportedPlatform)
}

fn open_directory(
    path: &Path,
    control: &RoventStartupControl,
) -> Result<HeldDirectory, RoventStorageAdmissionError> {
    check(control)?;
    if !cfg!(target_os = "linux") {
        return Err(RoventStorageAdmissionError::UnsupportedPlatform);
    }
    if !path.is_absolute()
        || path.as_os_str().as_encoded_bytes().len() > MAX_ROOT_PATH_BYTES
        || !path
            .components()
            .any(|part| matches!(part, Component::Normal(_)))
        || path
            .components()
            .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
    {
        return Err(invalid(
            "explicit bounded absolute roots without traversal are required",
        ));
    }
    let mut directory = Dir::open_ambient_dir("/", cap_std::ambient_authority())
        .map_err(|source| io("directory filesystem anchor", source))?;
    let mut ancestors = vec![directory_identity(&directory)?];
    for part in path.components() {
        check(control)?;
        if let Component::Normal(name) = part {
            directory = directory
                .open_dir_nofollow(name)
                .map_err(|source| io("no-follow directory component", source))?;
            ancestors.push(directory_identity(&directory)?);
        }
    }
    let identity = directory_identity(&directory)?;
    Ok(HeldDirectory {
        path: path.to_path_buf(),
        directory,
        identity,
        ancestors,
    })
}

fn require_separation(
    owners: &RoventStorageAdmissionOwners,
) -> Result<(), RoventStorageAdmissionError> {
    let archive = owners.directories[ARCHIVE]
        .as_ref()
        .ok_or_else(|| invalid("archive root is not held"))?;
    for index in [CHECKOUT, PRIVATE, PROTECTED] {
        let protected = owners.directories[index]
            .as_ref()
            .ok_or_else(|| invalid("protected directory is not held"))?;
        if archive.path.starts_with(&protected.path)
            || protected.path.starts_with(&archive.path)
            || archive.ancestors.contains(&protected.identity)
            || protected.ancestors.contains(&archive.identity)
        {
            return Err(invalid(
                "archive root overlaps checkout/private/protected execution root",
            ));
        }
    }
    Ok(())
}

fn check(control: &RoventStartupControl) -> Result<(), RoventStorageAdmissionError> {
    if control.cancellation.load(Ordering::Acquire) {
        return Err(RoventStorageAdmissionError::Cancelled);
    }
    if Instant::now() >= control.until {
        return Err(RoventStorageAdmissionError::Expired);
    }
    Ok(())
}
fn invalid(reason: &'static str) -> RoventStorageAdmissionError {
    RoventStorageAdmissionError::Invalid(reason)
}
fn input(error: ManifestError) -> RoventStorageAdmissionError {
    RoventStorageAdmissionError::Input(error)
}
fn io(stage: &'static str, source: std::io::Error) -> RoventStorageAdmissionError {
    RoventStorageAdmissionError::Io { stage, source }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::open_rovent_storage_byte_custody as admit_rovent_storage;
    use super::*;
    use rz_experiments::{
        ROVENT_STORAGE_MEASUREMENT_CONTRACT, ROVENT_STORAGE_PROFILE_DOMAIN,
        RoventArchiveRuntimeLimits, RoventExecutionManifest, RoventStorageHelpers,
        RoventStoragePolicy, RoventStorageSourcePins,
    };
    use std::{
        fs,
        sync::atomic::{AtomicBool, AtomicU64},
        time::Duration,
    };

    static NEXT: AtomicU64 = AtomicU64::new(1);

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rovent-storage-admission-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            // Exactly this fixture-created tree; no source/cache/foreign tree.
            if self.0.parent() == Some(std::env::temp_dir().as_path())
                && self.0.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rovent-storage-admission-")
                })
                && fs::symlink_metadata(&self.0)
                    .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
    }

    struct Fixture {
        _scratch: Scratch,
        private_path: PathBuf,
        checkout: PathBuf,
        protected: PathBuf,
        archive: PathBuf,
        private: Dir,
        profile: ArtifactRef,
        lock: Arc<RoventInputLock>,
    }
    impl Fixture {
        fn new() -> Self {
            let scratch = Scratch::new();
            let private_path = scratch.0.join("private");
            let checkout = scratch.0.join("checkout");
            let protected = scratch.0.join("execution");
            let archive = scratch.0.join("archive");
            for directory in [&private_path, &checkout, &protected, &archive] {
                fs::create_dir(directory).unwrap();
            }
            fs::create_dir(private_path.join("inputs")).unwrap();
            let mut pins = Vec::new();
            for (name, body) in [
                ("source.json", "source-fixture/1\n"),
                ("codec.rs", "codec-fixture/1\n"),
                ("launch.rs", "launch-fixture/1\n"),
            ] {
                let path = format!("inputs/{name}");
                fs::write(private_path.join(&path), body).unwrap();
                pins.push(artifact(&path, body.as_bytes()));
            }
            let declaration = RoventStorageProfile {
                schema: ROVENT_STORAGE_PROFILE_DOMAIN.into(),
                policy: RoventStoragePolicy::ArchiveOnV1,
                archive_limits: RoventArchiveRuntimeLimits {
                    game_bytes_max: 1 << 20,
                    global_bytes_max: 2 << 20,
                    index_entries_max: 16,
                    index_bytes_max: 1024,
                    load_bytes_max: 1024,
                    load_deadline_max_ms: 100,
                    record_payload_bytes_max: 1024,
                    max_load_pins: 1,
                },
                maintenance_io_bytes_max: 1 << 20,
                worker_threads: 1,
                command_capacity: 1,
                completion_capacity: 1,
                helpers: RoventStorageHelpers {
                    shared_nnuec: 0,
                    archive: 1,
                },
                source_pins: RoventStorageSourcePins {
                    source: pins[0].clone(),
                    codec: pins[1].clone(),
                    launch: pins[2].clone(),
                },
                measurement_contract: ROVENT_STORAGE_MEASUREMENT_CONTRACT.into(),
            };
            let raw = serde_json::to_vec(&declaration).unwrap();
            fs::write(private_path.join("inputs/storage-profile.json"), &raw).unwrap();
            let profile = artifact("inputs/storage-profile.json", &raw);
            let mut provenance = vec![profile.clone()];
            provenance.extend(pins);
            let manifest = manifest(provenance);
            let lock = Arc::new(manifest.lock().unwrap());
            let private =
                Dir::open_ambient_dir(&private_path, cap_std::ambient_authority()).unwrap();
            Self {
                _scratch: scratch,
                private_path,
                checkout,
                protected,
                archive,
                private,
                profile,
                lock,
            }
        }
        fn request(&self) -> RoventStorageAdmissionRequest<'_> {
            RoventStorageAdmissionRequest {
                profile_path: &self.profile.path,
                profile_sha256: &self.profile.sha256,
                profile_bytes: self.profile.bytes,
                archive_root: &self.archive,
                checkout_root: &self.checkout,
                private_snapshot_root: &self.private_path,
                protected_execution_root: &self.protected,
            }
        }
        fn admit(&self) -> Result<RoventStorageByteCustody, RoventStorageAdmissionFailure> {
            admit_rovent_storage(
                Arc::clone(&self.lock),
                &self.private,
                self.request(),
                control(),
            )
        }
    }

    fn artifact(path: &str, raw: &[u8]) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: format!("{:x}", Sha256::digest(raw)),
            bytes: raw.len() as u64,
            source: "https://example.org/storage-admission-fixture".into(),
            license: "MIT".into(),
        }
    }
    fn control() -> RoventStartupControl {
        let until = Instant::now() + Duration::from_secs(10);
        RoventStartupControl {
            until,
            cleanup_until: until + Duration::from_secs(1),
            cancellation: Arc::new(AtomicBool::new(false)),
        }
    }
    fn manifest(provenance: Vec<ArtifactRef>) -> RoventExecutionManifest {
        let pin = serde_json::json!({
            "semantics":"storage-fixture-model/1","encoding":"storage-fixture-input/1",
            "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
            "implementation_sha256":"b".repeat(64),"weights":artifact("inputs/weights.bin",b"x"),
            "provenance":artifact("inputs/model-provenance.json",b"x"),"diagnostic":false
        });
        let value = serde_json::json!({
            "domain":"rz-rovent-execution/1","contract_revision":rz_contracts::rovent::ROVENT_SEARCH_REVISION,
            "run_id":"storage-fd-admission-fixture","purpose":"validation",
            "engine":{
                "source_commit":"a".repeat(40),"source_components":{"storage":"c".repeat(64)},
                "binary":artifact("inputs/rovent",b"x"),"mode":"cpu_pc",
                "pc":{"proposer":pin.clone(),"critic":pin,"validator":null,"backend":"rust_cpu",
                    "device":{"device":"cpu"},"backend_configuration_sha256":"d".repeat(64),
                    "runtime_artifacts":[],"fp32":true,"tf32":false,"batch_width":1,
                    "ort_intra_threads":null,"ort_inter_threads":null,"ort_spinning":null},
                "researcher":null,"teachers":[],"nnuec":null,
                "search":{"node_unit":"rovent-position-visits/1","requested_nodes":null,
                    "depth_plies":null,"mate_moves":null,"registered_max_chess_plies":64,"searchmoves":null,
                    "search_configuration_sha256":"e".repeat(64),"root_state_and_history_sha256":"f".repeat(64),
                    "rules_profile_sha256":"1".repeat(64)}
            },
            "resources":{"cpu_quota_millicpus":2000,"pools":[
                {"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000},
                {"actors":["runtime_helper"],"max_threads":1,"cpu_charge_millicpus":1000}],
                "memory_high_bytes":3u64<<30,"memory_max_bytes":4u64<<30,"gpu_devices":0,
                "physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
            "coordinator":{"policy_sha256":"2".repeat(64),"questions_max":64,"ancestor_depth_max":64,
                "waiting_jobs_max":64,"role_tickets_max":16,"cpu_credit_limit":0,
                "original_root_budget_sha256":"3".repeat(64),"original_deadline_seal_sha256":"4".repeat(64)},
            "wall_time_max_ms":120000,"cleanup_max_ms":30000,"output_bytes_max":1u64<<20,
            "agent_renewable":true,"provenance":provenance
        });
        RoventExecutionManifest::from_json(&value.to_string()).unwrap()
    }

    #[test]
    fn admission_holds_original_lock_fds_and_empty_root_without_enabling_archive() {
        let fixture = Fixture::new();
        let original_control = control();
        let admitted = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            fixture.request(),
            original_control.clone(),
        )
        .unwrap();
        assert!(Arc::ptr_eq(admitted.owners().input_lock(), &fixture.lock));
        assert_eq!(admitted.input_lock_sha256(), fixture.lock.canonical_sha256);
        assert_eq!(admitted.owners().verified_source_files(), 3);
        assert!(admitted.owners().profile_file().is_some());
        assert!(admitted.owners().retains_root_capability());
        assert_eq!(admitted.owners.control.until, original_control.until);
        assert!(Arc::ptr_eq(
            &admitted.owners.control.cancellation,
            &original_control.cancellation
        ));
        assert!(admitted.revalidate_startup().is_ok());
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn wrong_lock_raw_pin_and_duplicate_provenance_reject_without_profile_open() {
        let fixture = Fixture::new();
        let mut changed = (*fixture.lock).clone();
        changed.canonical_sha256 = "b".repeat(64);
        let wrong = Arc::new(changed);
        let failed = admit_rovent_storage(
            Arc::clone(&wrong),
            &fixture.private,
            fixture.request(),
            control(),
        )
        .err()
        .unwrap();
        assert!(Arc::ptr_eq(failed.owners.input_lock(), &wrong));
        assert!(failed.owners.profile_file().is_none());
        let mut request = fixture.request();
        request.profile_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            request,
            control(),
        )
        .err()
        .unwrap();
        assert!(failed.owners.profile_artifact().is_some());
        assert!(failed.owners.profile_file().is_none());
        let mut repeated = fixture.lock.manifest.clone();
        repeated.provenance.push(fixture.profile.clone());
        let repeated = Arc::new(repeated.lock().unwrap());
        let failed = admit_rovent_storage(repeated, &fixture.private, fixture.request(), control())
            .err()
            .unwrap();
        assert!(failed.owners.profile_file().is_none());
    }

    #[test]
    fn missing_second_source_preserves_profile_and_first_verified_fd() {
        let fixture = Fixture::new();
        fs::remove_file(fixture.private_path.join("inputs/codec.rs")).unwrap();
        let failed = fixture.admit().err().unwrap();
        assert!(failed.owners.profile_file().is_some());
        assert!(failed.owners.profile().is_some());
        assert_eq!(failed.owners.verified_source_files(), 1);
        assert!(failed.owners.source_file(0).is_some());
        assert!(failed.owners.source_file(1).is_none());
        assert!(!failed.owners.retains_root_capability());
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn exact_source_metadata_and_actual_private_directory_are_required() {
        let fixture = Fixture::new();
        let mut changed = fixture.lock.manifest.clone();
        changed
            .provenance
            .iter_mut()
            .find(|a| a.path == "inputs/codec.rs")
            .unwrap()
            .source = "https://example.org/different-origin".into();
        let failed = admit_rovent_storage(
            Arc::new(changed.lock().unwrap()),
            &fixture.private,
            fixture.request(),
            control(),
        )
        .err()
        .unwrap();
        assert_eq!(failed.owners.verified_source_files(), 1);
        let mut request = fixture.request();
        request.private_snapshot_root = &fixture.checkout;
        // The actual private capability still reads the original files; a
        // different declared private root must not impersonate that same Dir.
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            request,
            control(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            failed.error,
            RoventStorageAdmissionError::Invalid(_)
        ));
        assert_eq!(failed.owners.verified_source_files(), 3);
    }

    #[test]
    fn overlapping_roots_and_symlink_components_never_make_a_capability() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        for root in [&fixture.checkout, &fixture.private_path, &fixture.protected] {
            let mut request = fixture.request();
            request.archive_root = root;
            let failed = admit_rovent_storage(
                Arc::clone(&fixture.lock),
                &fixture.private,
                request,
                control(),
            )
            .err()
            .unwrap();
            assert!(matches!(
                failed.error,
                RoventStorageAdmissionError::Invalid(_)
            ));
            assert!(!failed.owners.retains_root_capability());
        }
        let alias = fixture._scratch.0.join("archive-alias");
        symlink(&fixture.archive, &alias).unwrap();
        let mut request = fixture.request();
        request.archive_root = &alias;
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            request,
            control(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            failed.error,
            RoventStorageAdmissionError::Io { .. }
        ));
        assert!(!failed.owners.retains_root_capability());
    }

    #[test]
    fn root_name_replacement_is_rejected_while_original_capability_is_retained() {
        let fixture = Fixture::new();
        let admitted = fixture.admit().unwrap();
        let old = fixture._scratch.0.join("original-archive");
        fs::rename(&fixture.archive, &old).unwrap();
        fs::create_dir(&fixture.archive).unwrap();
        assert!(matches!(
            admitted.revalidate_startup(),
            Err(RoventStorageAdmissionError::Invalid(_))
        ));
        assert!(admitted.owners().retains_root_capability());
        assert_eq!(fs::read_dir(&old).unwrap().count(), 0);
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn original_file_mutation_and_original_cancellation_are_not_renewed() {
        let fixture = Fixture::new();
        let original = control();
        let admitted = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            fixture.request(),
            original.clone(),
        )
        .unwrap();
        fs::write(
            fixture.private_path.join("inputs/source.json"),
            b"changed-source\n",
        )
        .unwrap();
        assert!(matches!(
            admitted.revalidate_startup(),
            Err(RoventStorageAdmissionError::Invalid(_))
        ));
        original.cancellation.store(true, Ordering::Release);
        assert!(matches!(
            admitted.revalidate_startup(),
            Err(RoventStorageAdmissionError::Cancelled)
        ));
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            fixture.request(),
            original,
        )
        .err()
        .unwrap();
        assert!(matches!(
            failed.error,
            RoventStorageAdmissionError::Cancelled
        ));
        assert!(failed.owners.profile_file().is_none());
        let mut expired = control();
        expired.until = Instant::now();
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            fixture.request(),
            expired,
        )
        .err()
        .unwrap();
        assert!(matches!(failed.error, RoventStorageAdmissionError::Expired));
        assert!(failed.owners.profile_file().is_none());
    }
}
