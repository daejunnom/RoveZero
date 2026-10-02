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
        Self::copy_with(source, output_root, expected_sha256, |writer, bytes| {
            writer.write_all(bytes)?;
            writer.sync_all()
        })
    }

    fn copy_with(
        source: &Path,
        output_root: &Path,
        expected_sha256: &str,
        write: impl FnOnce(&mut File, &[u8]) -> std::io::Result<()>,
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
        let builder = fs::DirBuilder::new();
        #[cfg(unix)]
        let builder = {
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = builder;
            builder.mode(0o700);
            builder
        };
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
            write(&mut writer, &bytes).map_err(|error| {
                io_error("cannot write and sync verified runtime copy")
                    .with_external_cause(crate::error::CauseCode::RuntimePath, &error)
            })?;
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
            let _ = clear_owned_windows_readonly_attribute(&path);
            let _ = fs::remove_file(&path);
            let _ = fs::remove_dir(&directory);
        }
        result
    }

    pub fn binary_digest(&self) -> [u8; 32] {
        self.0.digest
    }

    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    pub(crate) fn path(&self) -> &Path {
        &self.0.path
    }
}

fn io_error(detail: &'static str) -> BackendError {
    BackendError::new(K::Io, S::Backend, detail)
}

// Windows only: clear FILE_ATTRIBUTE_READONLY on this bootstrap's own copy.
// Unlike Unix mode changes, this does not grant world write permission. Cleanup
// calls this after its writer/pin has closed; the sharing regression also clears
// the attribute while pinned to verify that write/delete sharing stays denied.
#[cfg(windows)]
#[allow(clippy::permissions_set_readonly_false)]
fn clear_owned_windows_readonly_attribute(path: &Path) -> std::io::Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(false);
    fs::set_permissions(path, permissions)
}

#[cfg(test)]
mod tests {
    use super::*;
    static NEXT_TEST: AtomicU64 = AtomicU64::new(0);
    const FAKE_LIBRARY: &[u8] = b"fake native bootstrap bytes: no actual ORT execution";

    fn fixture_root() -> (PathBuf, PathBuf) {
        #[cfg(windows)]
        let base =
            PathBuf::from(std::env::var_os("APPDATA").expect("Windows tests require APPDATA"));
        #[cfg(not(windows))]
        let base = std::env::var_os("RUNNER_TEMP")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let base = base.join("RoveZero").join("tmp").join("native-pin-tests");
        fs::create_dir_all(&base).unwrap();
        let base = base.canonicalize().unwrap();
        let root = base.join(format!(
            "{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let source = root.join("library.fixture");
        fs::write(&source, FAKE_LIBRARY).unwrap();
        (root, source)
    }
    fn release_copy(pin: RuntimeLibraryPin, root: &Path) {
        let path = pin.path().to_owned();
        let directory = path.parent().unwrap().to_owned();
        assert_eq!(directory.parent().unwrap(), root);
        drop(pin);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        }
        #[cfg(windows)]
        clear_owned_windows_readonly_attribute(&path).unwrap();
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
    fn release_root(root: PathBuf, source: PathBuf) {
        assert_eq!(source.parent().unwrap(), root);
        fs::remove_file(source).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn mutable_source_does_not_change_owned_bytes_or_disclose_private_path() {
        let (root, source) = fixture_root();
        let digest = asset::sha256(FAKE_LIBRARY);
        let pin =
            RuntimeLibraryPin::copy_verified(&source, &root, &asset::hex_sha256(FAKE_LIBRARY))
                .unwrap();
        fs::write(&source, b"source was subsequently replaced").unwrap();
        assert_eq!(asset::sha256(&fs::read(pin.path()).unwrap()), digest);
        assert_eq!(pin.binary_digest(), digest);
        let diagnostic = format!("{pin:?}");
        assert!(!diagnostic.contains(root.to_str().unwrap()));
        assert!(!diagnostic.contains("library.fixture"));
        release_copy(pin, &root);
        release_root(root, source);
    }

    #[test]
    fn digest_rejection_and_partial_write_failure_leave_no_copy() {
        let (root, source) = fixture_root();
        let failure =
            RuntimeLibraryPin::copy_verified(&source, &root, &"01".repeat(32)).unwrap_err();
        assert_eq!(failure.kind, K::IdentityMismatch);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        let failure = RuntimeLibraryPin::copy_with(
            &source,
            &root,
            &asset::hex_sha256(FAKE_LIBRARY),
            |writer, bytes| {
                writer.write_all(&bytes[..8])?;
                Err(std::io::Error::other(
                    "injected finite partial-write failure",
                ))
            },
        )
        .unwrap_err();
        assert_eq!(failure.kind, K::Io);
        assert!(failure.cause.is_some());
        assert_eq!(
            fs::read_dir(&root).unwrap().count(),
            1,
            "only the caller's source remains after failure cleanup"
        );
        release_root(root, source);
    }

    #[cfg(windows)]
    #[test]
    fn live_pin_denies_write_and_delete_sharing() {
        use std::os::windows::fs::OpenOptionsExt;
        let (root, source) = fixture_root();
        let pin =
            RuntimeLibraryPin::copy_verified(&source, &root, &asset::hex_sha256(FAKE_LIBRARY))
                .unwrap();
        // Clear readonly to test OS sharing rather than just the attribute.
        clear_owned_windows_readonly_attribute(pin.path()).unwrap();
        assert!(OpenOptions::new()
            .write(true)
            .share_mode(7)
            .open(pin.path())
            .is_err());
        assert!(fs::remove_file(pin.path()).is_err());
        release_copy(pin, &root);
        release_root(root, source);
    }
}
