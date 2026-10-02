use rz_experiments::{ArtifactRef, ManifestError, RunManifest};
use std::fs;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const FIXTURE: &str = include_str!("../../../experiments/baselines/fixtures/e01-input.json");
const IDENTITY: &[u8] = include_bytes!("../../../experiments/baselines/fixtures/e01-identity.txt");
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

fn artifact() -> ArtifactRef {
    RunManifest::from_json(FIXTURE).unwrap().engines[0]
        .tool
        .binary
        .clone()
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let parent = std::env::temp_dir().join("rovezero-e01-verified-handle-tests");
        fs::create_dir_all(&parent).unwrap();
        loop {
            let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!("{}-{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create test directory: {error}"),
            }
        }
    }

    fn with_identity() -> Self {
        let directory = Self::new();
        fs::write(directory.0.join("e01-identity.txt"), IDENTITY).unwrap();
        directory
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn rejected(artifact: &ArtifactRef, root: &Path, budget: u64) {
    assert!(matches!(
        artifact.open_verified(root, budget),
        Err(ManifestError::Artifact(_))
    ));
}

#[test]
fn verified_handle_is_rewound_read_only_and_contains_the_hashed_bytes() {
    let directory = TestDirectory::with_identity();
    let reference = artifact();
    let mut pinned = reference
        .open_verified(&directory.0, reference.bytes)
        .unwrap();
    assert_eq!(pinned.stream_position().unwrap(), 0);
    assert!(
        pinned
            .write_all(b"must not modify a verified artifact")
            .is_err()
    );
    let mut actual = Vec::new();
    pinned.read_to_end(&mut actual).unwrap();
    assert_eq!(actual, IDENTITY);
    assert_eq!(
        fs::read(directory.0.join(&reference.path)).unwrap(),
        IDENTITY
    );
}

#[test]
fn replacing_a_verified_path_cannot_redirect_the_returned_file_handle() {
    let directory = TestDirectory::with_identity();
    let reference = artifact();
    let original_path = directory.0.join(&reference.path);
    let mut pinned = reference
        .open_verified(&directory.0, reference.bytes)
        .unwrap();
    fs::rename(&original_path, directory.0.join("original-pinned.txt")).unwrap();
    let replacement = vec![b'x'; IDENTITY.len()];
    fs::write(&original_path, &replacement).unwrap();
    assert_eq!(fs::read(&original_path).unwrap(), replacement);
    let mut actual = Vec::new();
    pinned.read_to_end(&mut actual).unwrap();
    assert_eq!(
        actual, IDENTITY,
        "must consume the verified inode, not reopen its former name"
    );
    rejected(&reference, &directory.0, reference.bytes);
}

#[test]
fn replacing_a_parent_directory_does_not_redirect_a_verified_nested_file() {
    let directory = TestDirectory::new();
    fs::create_dir(directory.0.join("engine")).unwrap();
    fs::write(directory.0.join("engine/binary.txt"), IDENTITY).unwrap();
    let mut reference = artifact();
    reference.path = "engine/binary.txt".into();
    let mut pinned = reference
        .open_verified(&directory.0, reference.bytes)
        .unwrap();
    fs::rename(
        directory.0.join("engine"),
        directory.0.join("pinned-engine"),
    )
    .unwrap();
    fs::create_dir(directory.0.join("engine")).unwrap();
    fs::write(
        directory.0.join("engine/binary.txt"),
        vec![b'x'; IDENTITY.len()],
    )
    .unwrap();
    let mut actual = Vec::new();
    pinned.read_to_end(&mut actual).unwrap();
    assert_eq!(actual, IDENTITY);
    rejected(&reference, &directory.0, reference.bytes);
}

#[test]
fn declared_size_digest_and_explicit_positive_byte_ceiling_are_enforced() {
    let directory = TestDirectory::with_identity();
    let reference = artifact();
    rejected(&reference, &directory.0, 0);
    rejected(&reference, &directory.0, reference.bytes - 1);
    for declared_size in [reference.bytes - 1, reference.bytes + 1] {
        let mut invalid = reference.clone();
        invalid.bytes = declared_size;
        rejected(&invalid, &directory.0, declared_size);
    }
    let mut changed_digest = reference.clone();
    changed_digest.sha256 = "1".repeat(64);
    rejected(&changed_digest, &directory.0, reference.bytes);
    let mut changed_bytes = IDENTITY.to_vec();
    changed_bytes[0] ^= 1;
    fs::write(directory.0.join(&reference.path), &changed_bytes).unwrap();
    rejected(&reference, &directory.0, reference.bytes);
}

#[test]
fn direct_open_validates_metadata_before_accessing_the_artifact_root() {
    let directory = TestDirectory::new();
    let missing_root = directory.0.join("does-not-exist");
    let mut variants = [artifact(), artifact(), artifact(), artifact(), artifact()];
    variants[0].path = "../outside.bin".into();
    variants[1].path = ".env.production".into();
    variants[2].sha256 = "0".repeat(64);
    variants[3].bytes = 0;
    variants[4].source = "https://user:credential@example.test/file".into();
    for invalid in variants {
        assert!(matches!(
            invalid.open_verified(&missing_root, u64::MAX),
            Err(ManifestError::Validation(_))
        ));
    }
}

#[test]
fn missing_roots_files_and_directory_artifacts_are_rejected() {
    let directory = TestDirectory::new();
    let reference = artifact();
    rejected(
        &reference,
        &directory.0.join("missing-root"),
        reference.bytes,
    );
    rejected(&reference, &directory.0, reference.bytes);
    fs::create_dir(directory.0.join(&reference.path)).unwrap();
    rejected(&reference, &directory.0, reference.bytes);
}

#[cfg(unix)]
#[test]
fn final_symlinks_and_internal_or_external_parent_symlinks_are_rejected() {
    use std::os::unix::fs::symlink;
    let directory = TestDirectory::with_identity();
    let outside = TestDirectory::with_identity();
    let reference = artifact();
    fs::remove_file(directory.0.join(&reference.path)).unwrap();
    symlink(
        outside.0.join(&reference.path),
        directory.0.join(&reference.path),
    )
    .unwrap();
    rejected(&reference, &directory.0, reference.bytes);
    fs::remove_file(directory.0.join(&reference.path)).unwrap();
    fs::write(directory.0.join(&reference.path), IDENTITY).unwrap();
    let mut nested = reference.clone();
    nested.path = "linked/e01-identity.txt".into();
    symlink(&outside.0, directory.0.join("linked")).unwrap();
    rejected(&nested, &directory.0, nested.bytes);
    fs::remove_file(directory.0.join("linked")).unwrap();
    fs::create_dir(directory.0.join("internal")).unwrap();
    fs::write(directory.0.join("internal/e01-identity.txt"), IDENTITY).unwrap();
    symlink("internal", directory.0.join("linked")).unwrap();
    rejected(&nested, &directory.0, nested.bytes);
}

#[cfg(unix)]
#[test]
fn fifo_without_a_writer_cannot_block_verified_open() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    const CHILD_ROOT: &str = "RZ_E01_VERIFIED_HANDLE_FIFO_ROOT";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let reference = artifact();
        rejected(&reference, Path::new(&root), reference.bytes);
        return;
    }
    let directory = TestDirectory::new();
    let name = CString::new(directory.0.join("e01-identity.txt").as_os_str().as_bytes()).unwrap();
    // CString remains live and owns the NUL-terminated isolated test path.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "fifo_without_a_writer_cannot_block_verified_open",
        ])
        .env(CHILD_ROOT, &directory.0)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("verified FIFO open exceeded its five-second test deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
