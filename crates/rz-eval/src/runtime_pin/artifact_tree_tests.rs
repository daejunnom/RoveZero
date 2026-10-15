//! 작은 regular 텍스트 fixture만 사용한다. native 코드나 신경망은 실행하지 않는다.

use super::*;
use std::os::unix::fs::{symlink, PermissionsExt};

const BYTES: &[u8] = b"authored non-executable bytes for bounded runtime cache custody and read accounting; no ELF or CUDA library is used by this fixture.";
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    source: PathBuf,
    cache: RuntimeCache,
    spec: CudaRuntimeBundleSpec,
}
impl Fixture {
    fn new() -> Self {
        let base = std::env::var_os("RUNNER_TEMP")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("RoveZero/tmp/runtime-artifact-tree-tests");
        fs::create_dir_all(&base).unwrap();
        let root = base.join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let source = root.join("source");
        fs::create_dir(&source).unwrap();
        let mut files = vec![
            (RuntimeBundleFileRole::Core, CORE_FILENAME),
            (RuntimeBundleFileRole::ProvidersShared, SHARED_FILENAME),
            (RuntimeBundleFileRole::ProvidersCuda, CUDA_FILENAME),
        ];
        files.extend(
            NVIDIA_FILENAMES
                .iter()
                .map(|name| (RuntimeBundleFileRole::NvidiaDependency, *name)),
        );
        let files: Vec<_> = files
            .into_iter()
            .map(|(role, filename)| {
                fs::write(source.join(filename), BYTES).unwrap();
                RuntimeBundleFile {
                    role,
                    filename: filename.into(),
                    bytes: BYTES.len() as u64,
                    sha256: asset::hex_sha256(BYTES),
                }
            })
            .collect();
        let loader_aliases = Some(
            files
                .iter()
                .filter(|file| file.filename.starts_with("libcudnn"))
                .map(|file| RuntimeBundleAlias {
                    filename: format!("{}.8.0", file.filename),
                    target: file.filename.clone(),
                })
                .collect(),
        );
        let spec = CudaRuntimeBundleSpec {
            schema_version: 2,
            files,
            loader_aliases,
        };
        spec.validate().unwrap();
        let cache = RuntimeCache::open(&root.join("cache")).unwrap();
        Self {
            root,
            source,
            cache,
            spec,
        }
    }
    fn prepare(&self, read: &mut u64) -> PreparedCudaArtifactTree {
        self.cache
            .prepare_cuda_artifact_tree(&self.source, &self.spec, &mut |bytes| {
                *read += bytes;
                Ok(())
            })
            .unwrap()
    }
    fn entry(&self, tree: &PreparedCudaArtifactTree) -> PathBuf {
        self.cache.root().join(&tree.0.name)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fn remove_owned(root: &Path, path: &Path, depth: usize) {
            assert!(depth <= 4 && path.starts_with(root));
            let metadata = fs::symlink_metadata(path).unwrap();
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
                let entries: Vec<_> = fs::read_dir(path)
                    .unwrap()
                    .map(|e| e.unwrap().path())
                    .collect();
                assert!(entries.len() <= 64);
                for entry in entries {
                    remove_owned(root, &entry, depth + 1);
                }
                fs::remove_dir(path).unwrap();
            } else {
                fs::remove_file(path).unwrap();
            }
        }
        remove_owned(&self.root, &self.root, 0);
    }
}

#[test]
fn final_entry_has_original_fds_and_exact_alias_bytes_without_tick_reads() {
    let fixture = Fixture::new();
    let mut reads = 0;
    let first = fixture.prepare(&mut reads);
    let sum = BYTES.len() as u64 * 19;
    assert_eq!(reads, 3 * (sum + 19));
    let after_prepare = reads;
    let expected = first.observe().unwrap();
    assert_eq!(
        expected.bytes(),
        sum + fixture
            .spec
            .loader_aliases
            .as_ref()
            .unwrap()
            .iter()
            .map(|a| a.target.len() as u64)
            .sum::<u64>()
    );
    assert_eq!(
        (
            expected.files(),
            expected.entries(),
            expected.regular_files(),
            expected.aliases()
        ),
        (27, 27, 19, 8)
    );
    assert_eq!(
        (
            expected.directories(),
            expected.depth(),
            expected.max_file_bytes()
        ),
        (0, 0, BYTES.len() as u64)
    );
    assert!(first.matches_directory(&first.0.directory).unwrap());
    assert!(!first.matches_directory(&first.0.parent).unwrap());
    for _ in 0..8 {
        assert_eq!(first.observe().unwrap(), expected);
    }
    assert_eq!(reads, after_prepare);
    let clone = first.clone();
    drop(first);
    assert_eq!(clone.observe().unwrap(), expected);
    // cache hit는 mutable source를 읽지 않고 최종 19개 pin만 hash한다.
    fs::write(
        fixture.source.join(CORE_FILENAME),
        b"source changed after publication",
    )
    .unwrap();
    let before = reads;
    let second = fixture.prepare(&mut reads);
    assert_eq!(reads - before, sum + 19);
    assert_eq!(second.observe().unwrap(), expected);
    assert!(second.matches_directory(&clone.0.directory).unwrap());
    assert!(!format!("{second:?}").contains(fixture.root.to_str().unwrap()));
}

#[test]
fn metadata_observer_rejects_replacement_extra_missing_modes_links_and_wrong_target() {
    for kind in [
        "replace",
        "extra",
        "missing",
        "writable",
        "hardlink",
        "wrong-target",
        "absolute-alias",
        "extra-directory",
    ] {
        let fixture = Fixture::new();
        let tree = fixture.prepare(&mut 0);
        let entry = fixture.entry(&tree);
        let regular = entry.join(CORE_FILENAME);
        if kind == "writable" {
            fs::set_permissions(&regular, fs::Permissions::from_mode(0o600)).unwrap();
        } else {
            fs::set_permissions(&entry, fs::Permissions::from_mode(0o700)).unwrap();
            match kind {
                "replace" => {
                    fs::remove_file(&regular).unwrap();
                    fs::write(&regular, BYTES).unwrap();
                    fs::set_permissions(&regular, fs::Permissions::from_mode(0o400)).unwrap();
                }
                "extra" => fs::write(entry.join("unregistered.fixture"), BYTES).unwrap(),
                "missing" => fs::remove_file(&regular).unwrap(),
                "hardlink" => {
                    fs::hard_link(&regular, fixture.root.join("second-link.fixture")).unwrap()
                }
                "wrong-target" | "absolute-alias" => {
                    let alias = fixture
                        .spec
                        .loader_aliases
                        .as_ref()
                        .unwrap()
                        .first()
                        .unwrap();
                    fs::remove_file(entry.join(&alias.filename)).unwrap();
                    symlink(
                        if kind == "absolute-alias" {
                            fixture.source.join(&alias.target)
                        } else {
                            PathBuf::from(CORE_FILENAME)
                        },
                        entry.join(&alias.filename),
                    )
                    .unwrap();
                }
                "extra-directory" => fs::create_dir(entry.join("unexpected-child")).unwrap(),
                _ => unreachable!(),
            }
            fs::set_permissions(&entry, fs::Permissions::from_mode(0o500)).unwrap();
        }
        assert!(tree.observe().is_err(), "{kind}");
    }
}

#[test]
fn named_entry_and_parent_replacement_are_not_restored_by_original_fds() {
    for replace_parent in [false, true] {
        let fixture = Fixture::new();
        let tree = fixture.prepare(&mut 0);
        let original = if replace_parent {
            fixture.cache.root().to_owned()
        } else {
            fixture.entry(&tree)
        };
        let renamed = fixture.root.join("renamed-original");
        fs::rename(&original, &renamed).unwrap();
        fs::create_dir(&original).unwrap();
        fs::set_permissions(
            &original,
            fs::Permissions::from_mode(if replace_parent { 0o700 } else { 0o500 }),
        )
        .unwrap();
        assert!(tree.matches_directory(&tree.0.directory).is_err());
        assert!(tree.observe().is_err());
    }
}

#[test]
fn interrupted_read_is_rejected_and_only_unpublished_owned_files_are_removed() {
    let fixture = Fixture::new();
    let mut charged = 0;
    let failure = fixture
        .cache
        .prepare_cuda_artifact_tree(&fixture.source, &fixture.spec, &mut |count| {
            charged += count;
            if charged > BYTES.len() as u64 {
                Err(BackendError::new(
                    K::ResourceExhausted,
                    S::Backend,
                    "fixture original read budget exhausted",
                ))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(failure.kind, K::ResourceExhausted);
    assert!(charged > 0);
    assert_eq!(fs::read_dir(fixture.cache.root()).unwrap().count(), 0);
    assert_eq!(fs::read(fixture.source.join(CORE_FILENAME)).unwrap(), BYTES);
}

#[test]
fn canceled_checkpoint_during_lock_wait_is_not_recovered_as_cache_success() {
    let fixture = Fixture::new();
    let lock = fixture.cache.root().join(".publish-lock");
    fs::create_dir(&lock).unwrap();
    fs::set_permissions(&lock, fs::Permissions::from_mode(0o700)).unwrap();
    let started = std::time::Instant::now();
    let mut checkpoints = 0;
    let failure = fixture
        .cache
        .prepare_cuda_artifact_tree(&fixture.source, &fixture.spec, &mut |count| {
            assert_eq!(count, 0);
            checkpoints += 1;
            if checkpoints >= 3 {
                Err(BackendError::new(
                    K::ResourceExhausted,
                    S::Backend,
                    "fixture original deadline expired",
                ))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(failure.kind, K::ResourceExhausted);
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
    assert!(lock.is_dir());
    assert_eq!(fs::read_dir(fixture.cache.root()).unwrap().count(), 1);
}

#[test]
fn reused_entry_still_charges_reads_and_rejects_before_exceeding_original_credit() {
    let fixture = Fixture::new();
    let original = fixture.prepare(&mut 0);
    let mut charged = 0;
    let failure = fixture
        .cache
        .prepare_cuda_artifact_tree(&fixture.source, &fixture.spec, &mut |count| {
            if charged + count > BYTES.len() as u64 {
                return Err(BackendError::new(
                    K::ResourceExhausted,
                    S::Backend,
                    "fixture cache-hit credit exhausted",
                ));
            }
            charged += count;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(failure.kind, K::ResourceExhausted);
    assert_eq!(charged, BYTES.len() as u64);
    assert!(original.observe().is_ok());
    assert_eq!(fs::read_dir(fixture.cache.root()).unwrap().count(), 1);
}
