//! A bootstrap-owned copy of the verified native library, outside the checkout.
//!
//! Hashing a caller's mutable path does not bind the bytes subsequently dlopened.
//! This capability copies verified bytes into a newly created private directory,
//! closes the writer, makes the copy read-only and keeps an OS file pin alive.
//! The caller must supply a private output root whose ancestors it controls.
//! Its owner must not chmod, replace or remove bootstrap copies. This is an
//! ownership contract, not protection against hostile code with the same OS UID.
//! Runtime copies are retained for process lifetime once passed to ORT. No
//! runtime/provider assets are downloaded, and sibling provider libraries are
//! deliberately not copied: absent native dependencies remain a bootstrap error.

use crate::{
    asset,
    error::{BackendError, FailureKind as K, FailureStage as S},
};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

const MAX_LIBRARY_BYTES: usize = 512 * 1024 * 1024;
static NEXT_COPY: AtomicU64 = AtomicU64::new(0);

struct OwnedCopy {
    path: PathBuf,
    digest: [u8; 32],
    // Windows denies write/delete sharing; Unix retains the exact owned inode.
    _pin: File,
}

/// Capability issued only after an owned copy has been verified and pinned.
#[derive(Clone)]
pub struct RuntimeLibraryPin(Arc<OwnedCopy>);

impl std::fmt::Debug for RuntimeLibraryPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Private filesystem paths are not shared diagnostics.
        f.debug_struct("RuntimeLibraryPin")
            .field("digest", &self.0.digest)
            .finish_non_exhaustive()
    }
}

impl RuntimeLibraryPin {
    /// `output_root` must exist outside Git and be owned by this bootstrap.
    /// Source mutation after this returns cannot change the loaded copy.
    pub fn copy_verified(
        source: &Path,
        output_root: &Path,
        expected_sha256: &str,
    ) -> Result<Self, BackendError> {
        let expected = asset::parse_sha256(expected_sha256)?;
        let root_metadata = fs::symlink_metadata(output_root)
            .map_err(|_| io_error("runtime output root is unavailable"))?;
        if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
            return Err(io_error(
                "runtime output root must be a real owned directory",
            ));
        }
        let root = output_root
            .canonicalize()
            .map_err(|_| io_error("cannot resolve runtime output root"))?;
        let bytes = asset::read_bounded(source, MAX_LIBRARY_BYTES)?;
        if bytes.is_empty() || asset::sha256(&bytes) != expected {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Backend,
                "native runtime source digest differs",
            ));
        }
        let filename = source
            .file_name()
            .ok_or_else(|| io_error("runtime source has no library filename"))?;
        let sequence = NEXT_COPY
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| {
                BackendError::new(
                    K::ResourceExhausted,
                    S::Backend,
                    "runtime copy sequence exhausted",
                )
            })?;
        let directory = root.join(format!("ort-bootstrap-{}-{}", std::process::id(), sequence));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        // create, never create_dir_all: a collision cannot be reused.
        builder
            .create(&directory)
            .map_err(|_| io_error("cannot create exclusive runtime directory"))?;
        let path = directory.join(filename);
        let result = (|| {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut writer = options
                .open(&path)
                .map_err(|_| io_error("cannot create exclusive runtime copy"))?;
            writer
                .write_all(&bytes)
                .map_err(|_| io_error("cannot write verified runtime copy"))?;
            writer
                .sync_all()
                .map_err(|_| io_error("cannot sync verified runtime copy"))?;
            drop(writer);
            let mut permissions = fs::metadata(&path)
                .map_err(|_| io_error("runtime copy metadata failed"))?
                .permissions();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                permissions.set_mode(0o400);
            }
            #[cfg(not(unix))]
            permissions.set_readonly(true);
            fs::set_permissions(&path, permissions)
                .map_err(|_| io_error("cannot make runtime copy read-only"))?;
            let mut options = OpenOptions::new();
            options.read(true);
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                // FILE_SHARE_READ: denies all writers and delete/rename handles.
                options.share_mode(1);
            }
            let pin = options
                .open(&path)
                .map_err(|_| io_error("cannot pin verified runtime copy"))?;
            // Re-read after closing the writer and acquiring the durable pin.
            if asset::sha256(&asset::read_bounded(&path, MAX_LIBRARY_BYTES)?) != expected {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "runtime owned copy digest differs",
                ));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&directory, fs::Permissions::from_mode(0o500))
                    .map_err(|_| io_error("cannot seal runtime directory"))?;
            }
            Ok(Self(Arc::new(OwnedCopy {
                path: path.clone(),
                digest: expected,
                _pin: pin,
            })))
        })();
        // Only a failed, unpublished copy is eligible for local cleanup.
        if result.is_err() {
            #[cfg(windows)]
            if let Ok(metadata) = fs::metadata(&path) {
                let mut permissions = metadata.permissions();
                permissions.set_readonly(false);
                let _ = fs::set_permissions(&path, permissions);
            }
            let _ = fs::remove_file(&path);
            let _ = fs::remove_dir(&directory);
        }
        result
    }

    pub fn binary_digest(&self) -> [u8; 32] {
        self.0.digest
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0.path
    }
}

fn io_error(detail: &'static str) -> BackendError {
    BackendError::new(K::Io, S::Backend, detail)
}
