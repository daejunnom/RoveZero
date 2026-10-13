//! Explicit owned-root admission. It grants no delete permission and never
//! evicts failure evidence to admit another run. Byte scans do not read data.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsOwnedStoragePlanV5 {
    /// User-created public marker binding this staging/cache root to one owner.
    pub owner_id: String,
    pub marker: ArtifactRef,
    pub byte_limit: u64,
    pub external_serialization_required: bool,
    pub automatic_eviction: bool,
}
impl PalsOwnedStoragePlanV5 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.marker.validate()?;
        require(
            !self.owner_id.is_empty()
                && self.owner_id.len() <= 128
                && self
                    .owner_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
                && self.marker.path == "pals-owned-storage.v5.json"
                && self.marker.bytes <= 4096
                && self.byte_limit == PALS_STAGING_BYTES_V5
                && self.external_serialization_required
                && !self.automatic_eviction,
            "V5 storage requires one explicit owner marker, 18GiB cap and no automatic evidence deletion",
        )
    }
}
#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct PalsOwnedStorageAuditV5 {
    pub domain: String,
    pub owner_id: String,
    pub marker_sha256: String,
    pub byte_limit: u64,
    pub reserved_attempt_bytes: u64,
    pub no_follow_byte_scan_completed: bool,
    pub external_serialization_observed: bool,
    pub limitations: Vec<String>,
}
/// Held across snapshots, child execution, failure recovery and quarantine.
/// The descriptor is not serializable and cannot be recovered from a receipt.
#[derive(Debug)]
pub struct PalsOwnedStorageLeaseV5 {
    #[cfg(target_os = "linux")]
    _marker: nix::fcntl::Flock<std::fs::File>,
    #[cfg(target_os = "linux")]
    _root: std::fs::File,
    #[cfg(not(target_os = "linux"))]
    _marker: std::fs::File,
}

/// Re-admit another serial pair under the exact still-held marker flock. This
/// never constructs a guard from metadata, unlocks it, or deletes old evidence.
#[cfg(target_os = "linux")]
pub(crate) fn readmit_pals_owned_storage_v5(
    plan: &PalsOwnedStoragePlanV5,
    root: &Path,
    reservation: u64,
    lease: &PalsOwnedStorageLeaseV5,
) -> Result<PalsOwnedStorageAuditV5, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
    use std::os::unix::fs::{FileExt, MetadataExt};
    plan.validate()?;
    require(
        root.is_absolute()
            && root
                .components()
                .all(|p| !matches!(p, std::path::Component::ParentDir))
            && root.canonicalize().is_ok_and(|p| p == root),
        "continued storage root must retain its exact canonical absolute path",
    )?;
    let directory = Dir::open_ambient_dir(root, cap_std::ambient_authority())
        .map_err(|_| invalid("continued storage root cannot be pinned"))?;
    let mut root_options = OpenOptions::new();
    root_options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let named_root = directory
        .open_with(".", &root_options)
        .map_err(|_| invalid("continued root identity unavailable"))?
        .into_std();
    let old_root = lease
        ._root
        .metadata()
        .map_err(|_| invalid("held storage root metadata unavailable"))?;
    let now_root = named_root
        .metadata()
        .map_err(|_| invalid("continued storage root metadata unavailable"))?;
    require(
        old_root.is_dir()
            && now_root.is_dir()
            && (old_root.dev(), old_root.ino()) == (now_root.dev(), now_root.ino()),
        "continued storage root inode differs from original held root",
    )?;
    require(
        directory.symlink_metadata(".git").is_err(),
        "source checkout cannot hold continued storage",
    )?;
    let mut opts = OpenOptions::new();
    opts.read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let named = directory
        .open_with(&plan.marker.path, &opts)
        .map_err(|_| invalid("continued marker pathname unavailable"))?
        .into_std();
    let held = lease
        ._marker
        .metadata()
        .map_err(|_| invalid("held storage marker metadata unavailable"))?;
    let current = named
        .metadata()
        .map_err(|_| invalid("continued marker metadata unavailable"))?;
    require(
        held.is_file()
            && held.nlink() == 1
            && current.is_file()
            && current.nlink() == 1
            && held.len() == plan.marker.bytes
            && (held.dev(), held.ino(), held.len())
                == (current.dev(), current.ino(), current.len()),
        "continued marker no longer names the exact original live guard",
    )?;
    let mut bytes = vec![0u8; plan.marker.bytes as usize];
    lease
        ._marker
        .read_exact_at(&mut bytes, 0)
        .map_err(|_| invalid("held continued marker read failed"))?;
    require(
        format!("{:x}", Sha256::digest(&bytes)) == plan.marker.sha256,
        "continued storage marker content differs",
    )?;
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Marker {
        domain: String,
        owner_id: String,
        purpose: String,
    }
    let marker: Marker = crate::decode_json(
        std::str::from_utf8(&bytes).map_err(|_| invalid("continued marker not UTF8"))?,
    )?;
    require(
        marker.domain == "rz-pals-owned-storage-v5/1"
            && marker.owner_id == plan.owner_id
            && marker.purpose == "staging_and_cache",
        "continued marker owner/profile differs",
    )?;
    crate::native_retention::admit_bounded(&directory, reservation, plan.byte_limit)?;
    let final_named = directory
        .open_with(&plan.marker.path, &opts)
        .map_err(|_| invalid("continued marker changed during quota check"))?
        .into_std();
    let final_metadata = final_named
        .metadata()
        .map_err(|_| invalid("continued marker final metadata unavailable"))?;
    require(
        (held.dev(), held.ino(), held.len())
            == (
                final_metadata.dev(),
                final_metadata.ino(),
                final_metadata.len(),
            ),
        "continued marker replaced during admission",
    )?;
    let final_directory = Dir::open_ambient_dir(root, cap_std::ambient_authority())
        .map_err(|_| invalid("continued root final pathname unavailable"))?;
    let final_root = final_directory
        .open_with(".", &root_options)
        .map_err(|_| invalid("continued root final identity unavailable"))?
        .into_std();
    let final_root_metadata = final_root
        .metadata()
        .map_err(|_| invalid("continued root final metadata unavailable"))?;
    require(
        (old_root.dev(), old_root.ino()) == (final_root_metadata.dev(), final_root_metadata.ino()),
        "continued root replaced during quota/marker admission",
    )?;
    Ok(PalsOwnedStorageAuditV5 {
        domain: "rz-pals-storage-admission-v5/1".into(),
        owner_id: plan.owner_id.clone(), marker_sha256: plan.marker.sha256.clone(),
        byte_limit: plan.byte_limit, reserved_attempt_bytes: reservation,
        no_follow_byte_scan_completed: true, external_serialization_observed: true,
        limitations: vec!["same original marker flock retained across serial normal closures; this is not a kernel quota or same-UID sandbox".into(), "all old reports, PGN and raw failures remain counted; no evidence eviction".into()],
    })
}

#[cfg(target_os = "linux")]
pub fn acquire_pals_owned_storage_v5(
    plan: &PalsOwnedStoragePlanV5,
    root: &Path,
    reservation: u64,
) -> Result<(PalsOwnedStorageAuditV5, PalsOwnedStorageLeaseV5), ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
    use nix::fcntl::{Flock, FlockArg};
    use std::{io::Read, os::unix::fs::MetadataExt};
    plan.validate()?;
    require(
        root.is_absolute()
            && root
                .components()
                .all(|p| !matches!(p, std::path::Component::ParentDir)),
        "storage root must be an explicit absolute path",
    )?;
    let canonical = root
        .canonicalize()
        .map_err(|_| invalid("owned storage root unavailable"))?;
    require(
        canonical == root,
        "owned storage root aliases a different resolved path",
    )?;
    let directory = Dir::open_ambient_dir(root, cap_std::ambient_authority())
        .map_err(|_| invalid("owned storage root cannot be pinned"))?;
    let mut root_options = OpenOptions::new();
    root_options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let root_pin = directory
        .open_with(".", &root_options)
        .map_err(|_| invalid("owned storage root identity cannot be retained"))?
        .into_std();
    let mut opts = OpenOptions::new();
    opts.read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let marker = directory
        .open_with(&plan.marker.path, &opts)
        .map_err(|_| invalid("storage ownership marker unavailable"))?
        .into_std();
    let held = marker
        .metadata()
        .map_err(|_| invalid("storage marker metadata unavailable"))?;
    require(
        held.is_file() && held.nlink() == 1 && held.len() == plan.marker.bytes,
        "storage marker is not a single bounded regular owner file",
    )?;
    // The owned guard retains the bounded regular marker and its exclusive lock
    // through the final launch owner's lifetime. Nonblocking acquisition refuses
    // concurrent admissions without waiting or retrying; failed admission drops
    // the guard rather than publishing a lease.
    let mut marker = Flock::lock(marker, FlockArg::LockExclusiveNonblock).map_err(|(_file, _error)| {
        invalid("owned staging/cache marker lease is busy or unavailable; no child or retry admitted")
    })?;
    let mut bytes = Vec::new();
    Read::take(&mut *marker, 4097)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid("storage marker read failed"))?;
    require(
        bytes.len() == plan.marker.bytes as usize
            && format!("{:x}", Sha256::digest(&bytes)) == plan.marker.sha256,
        "storage ownership marker content changed",
    )?;
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Marker {
        domain: String,
        owner_id: String,
        purpose: String,
    }
    let parsed: Marker = crate::decode_json(
        std::str::from_utf8(&bytes).map_err(|_| invalid("storage owner marker not UTF8"))?,
    )?;
    require(
        parsed.domain == "rz-pals-owned-storage-v5/1"
            && parsed.owner_id == plan.owner_id
            && parsed.purpose == "staging_and_cache",
        "storage marker does not name the registered owner/purpose",
    )?;
    // Reject source checkout roots without traversing them, then reuse the
    // established bounded no-follow entry/length walker with the stricter cap.
    require(
        directory.symlink_metadata(".git").is_err(),
        "source checkout cannot be an owned staging/cache root",
    )?;
    crate::native_retention::admit_bounded(&directory, reservation, plan.byte_limit)?;
    let named = directory
        .open_with(&plan.marker.path, &opts)
        .map_err(|_| invalid("storage marker pathname changed"))?
        .into_std();
    let now = named
        .metadata()
        .map_err(|_| invalid("storage marker final identity unavailable"))?;
    require(
        (held.dev(), held.ino(), held.len()) == (now.dev(), now.ino(), now.len()),
        "storage ownership marker was replaced during admission",
    )?;
    let audit=PalsOwnedStorageAuditV5{domain:"rz-pals-storage-admission-v5/1".into(),owner_id:plan.owner_id.clone(),marker_sha256:plan.marker.sha256.clone(),byte_limit:plan.byte_limit,reserved_attempt_bytes:reservation,no_follow_byte_scan_completed:true,external_serialization_observed:true,limitations:vec!["exclusive ownership-marker flock serializes participating launch/cache writers; this is not a kernel disk quota or protection against uncooperative same-UID writers".into(),"existing reports, PGN, failed attempts and shared runtime contents are preserved; no automatic eviction was performed".into()]};
    Ok((
        audit,
        PalsOwnedStorageLeaseV5 {
            _marker: marker,
            _root: root_pin,
        },
    ))
}
#[cfg(not(target_os = "linux"))]
pub fn acquire_pals_owned_storage_v5(
    _plan: &PalsOwnedStoragePlanV5,
    _root: &Path,
    _reservation: u64,
) -> Result<(PalsOwnedStorageAuditV5, PalsOwnedStorageLeaseV5), ArenaError> {
    Err(invalid(
        "V5 paired native ownership currently requires Linux",
    ))
}
