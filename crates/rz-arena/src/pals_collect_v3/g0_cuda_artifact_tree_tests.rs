//! 실제 cache 소유권 접점의 작은 filesystem fixture다. 모든 library 파일은
//! 비실행 텍스트이며 GPU, DSO loader, 신경망, child process를 호출하지 않는다.

use super::{
    GenerationTrainingFilePinV3 as Pin, g0_cuda_artifact_tree as cuda_tree, selfplay_cli,
    training_process as common,
};
use crate::ArenaError;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{
    fs,
    os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt, symlink},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const PAYLOAD: &[u8] = b"authored non-executable bytes for the G0 cache ownership adapter, bounded metadata and original read accounting; no ELF or CUDA library is present";
const CORE: &str = "libonnxruntime.so.1.22.0";
const OUTPUT_CAP: u64 = 1 << 20;
const READ_CAP: u64 = 1 << 20;

struct Fixture {
    root: PathBuf,
    parent: PathBuf,
    identity: (u64, u64),
    parent_identity: (u64, u64),
    source: PathBuf,
    cache: PathBuf,
    manifest: PathBuf,
    pins: Vec<Pin>,
    manifest_bytes: u64,
    alias_bytes: u64,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = std::env::temp_dir().canonicalize().unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = parent.join(format!(
            "rz-g0-cuda-adapter-test-{}-{}-{nonce}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        let root_metadata = fs::symlink_metadata(&root).unwrap();
        let parent_metadata = fs::symlink_metadata(&parent).unwrap();
        let source = root.join("source");
        fs::DirBuilder::new().mode(0o700).create(&source).unwrap();
        let cache = root.join("cache");
        let names = [
            ("core", CORE),
            ("providers_shared", "libonnxruntime_providers_shared.so"),
            ("providers_cuda", "libonnxruntime_providers_cuda.so"),
            ("nvidia_dependency", "libcudart.so.12"),
            ("nvidia_dependency", "libcublas.so.12"),
            ("nvidia_dependency", "libcublasLt.so.12"),
            ("nvidia_dependency", "libcufft.so.11"),
            ("nvidia_dependency", "libcurand.so.10"),
            ("nvidia_dependency", "libnvrtc.so.12"),
            ("nvidia_dependency", "libnvrtc-builtins.so.12.8"),
            ("nvidia_dependency", "libnvJitLink.so.12"),
            ("nvidia_dependency", "libcudnn.so.9"),
            ("nvidia_dependency", "libcudnn_adv.so.9"),
            ("nvidia_dependency", "libcudnn_cnn.so.9"),
            ("nvidia_dependency", "libcudnn_ops.so.9"),
            ("nvidia_dependency", "libcudnn_engines_precompiled.so.9"),
            (
                "nvidia_dependency",
                "libcudnn_engines_runtime_compiled.so.9",
            ),
            ("nvidia_dependency", "libcudnn_graph.so.9"),
            ("nvidia_dependency", "libcudnn_heuristic.so.9"),
        ];
        let mut pins = Vec::with_capacity(20);
        let files = names
            .iter()
            .map(|(role, filename)| {
                let path = source.join(filename);
                fs::write(&path, PAYLOAD).unwrap();
                let pin = pin(&path, PAYLOAD);
                let value = serde_json::json!({"role":role,"filename":filename,
                    "bytes":pin.bytes,"sha256":pin.sha256});
                pins.push(pin);
                value
            })
            .collect::<Vec<_>>();
        let aliases = names
            .iter()
            .filter(|(_, filename)| filename.starts_with("libcudnn"))
            .map(|(_, filename)| {
                serde_json::json!({"filename":format!("{filename}.8.0"),"target":filename})
            })
            .collect::<Vec<_>>();
        let alias_bytes = aliases
            .iter()
            .map(|alias| alias["target"].as_str().unwrap().len() as u64)
            .sum();
        let raw = serde_json::to_vec(&serde_json::json!({
            "schema_version":2,"files":files,"loader_aliases":aliases,
        }))
        .unwrap();
        rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&raw).unwrap())
            .unwrap();
        let manifest = source.join("cuda-bundle.json");
        fs::write(&manifest, &raw).unwrap();
        pins.push(pin(&manifest, &raw));
        Self {
            root,
            parent,
            identity: (root_metadata.dev(), root_metadata.ino()),
            parent_identity: (parent_metadata.dev(), parent_metadata.ino()),
            source,
            cache,
            manifest,
            pins,
            manifest_bytes: raw.len() as u64,
            alias_bytes,
        }
    }
    fn held(&self, name: &str) -> common::LinuxHeld {
        common::prepare_original_root(
            &self.root,
            self.root.join(format!("not-installed-{name}")),
            &self.root.join(format!("not-written-{name}.json")),
            &[],
        )
        .unwrap()
    }
    fn hold_inputs(&self, held: &mut common::LinuxHeld, credit: &mut common::ReadCredit<'_>) {
        for pin in &self.pins {
            common::hold_input(pin.clone(), false, held, credit).unwrap();
        }
    }
    fn input_bytes(&self) -> u64 {
        self.pins.iter().map(|pin| pin.bytes).sum()
    }
    fn prepare(
        &self,
        held: &mut common::LinuxHeld,
        credit: &mut common::ReadCredit<'_>,
    ) -> Result<(), ArenaError> {
        cuda_tree::prepare(
            held,
            &self.cache,
            &self.source.join(CORE),
            Some(&self.manifest),
            OUTPUT_CAP,
            credit,
        )
    }
    fn entry(&self, observation: &Value) -> PathBuf {
        let path = PathBuf::from(observation["directory"]["path"].as_str().unwrap());
        assert_eq!(path.parent(), Some(self.cache.as_path()));
        assert!(path.starts_with(&self.root));
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let (Ok(root), Ok(parent)) = (
            fs::symlink_metadata(&self.root),
            fs::symlink_metadata(&self.parent),
        ) else {
            return;
        };
        if !self.root.is_absolute()
            || self.root.parent() != Some(self.parent.as_path())
            || !root.is_dir()
            || root.file_type().is_symlink()
            || !parent.is_dir()
            || parent.file_type().is_symlink()
            || (root.dev(), root.ino()) != self.identity
            || (parent.dev(), parent.ino()) != self.parent_identity
        {
            return;
        }
        fn remove_owned(root: &Path, path: &Path, depth: usize, total: &mut usize) {
            assert!(path.starts_with(root) && depth <= 4);
            *total += 1;
            assert!(*total <= 256);
            let metadata = fs::symlink_metadata(path).unwrap();
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
                let entries = fs::read_dir(path)
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .collect::<Vec<_>>();
                assert!(entries.len() <= 64);
                for entry in entries {
                    remove_owned(root, &entry, depth + 1, total);
                }
                fs::remove_dir(path).unwrap();
            } else {
                // symlink 자체만 제거하며 target을 따라가지 않는다.
                fs::remove_file(path).unwrap();
            }
        }
        remove_owned(&self.root, &self.root, 0, &mut 0);
    }
}

fn pin(path: &Path, raw: &[u8]) -> Pin {
    Pin {
        path: path.to_path_buf(),
        bytes: raw.len() as u64,
        sha256: super::hex(&Sha256::digest(raw)),
    }
}
fn read_credit(cancel: &AtomicBool, until: Instant, maximum: u64) -> common::ReadCredit<'_> {
    common::ReadCredit {
        remaining: maximum,
        maximum,
        until,
        cancel,
        account: common::ReadAccounting::new(),
    }
}
fn expect_invalid(result: Result<(), ArenaError>, expected: &str) {
    assert!(matches!(result, Err(ArenaError::Invalid(detail)) if detail == expected));
}

#[test]
fn actual_g0_prepare_scans_exact_bytes_and_transfers_only_live_capability() {
    let fixture = Fixture::new();
    let cancel = AtomicBool::new(false);
    let until = Instant::now() + Duration::from_secs(30);
    let mut credit = read_credit(&cancel, until, READ_CAP);
    let mut held = fixture.held("original");
    fixture.hold_inputs(&mut held, &mut credit);
    fixture.prepare(&mut held, &mut credit).unwrap();
    let observation = cuda_tree::child_observation_for_dispatch(&held)
        .unwrap()
        .unwrap();
    assert_eq!(observation["files"].as_array().unwrap().len(), 19);
    assert_eq!(observation["aliases"].as_array().unwrap().len(), 8);
    let tree = held.cuda_artifact_tree.as_ref().unwrap().observe().unwrap();
    assert_eq!((tree.regular_files(), tree.aliases()), (19, 8));
    assert_eq!(
        tree.bytes(),
        19 * PAYLOAD.len() as u64 + fixture.alias_bytes
    );
    let remaining = credit.remaining;
    let charged = credit.account.charged.load(Ordering::Acquire);
    let account = Arc::clone(&credit.account);
    let scan = cuda_tree::scan_staging(&held, OUTPUT_CAP, until, &cancel).unwrap();
    assert_eq!(scan["delegated"]["nondirectory_entries"], 27);
    assert_eq!(scan["delegated"]["bytes"], tree.bytes());
    assert_eq!(
        scan["accounted_bytes"].as_u64().unwrap(),
        scan["ordinary"]["unique_inode_bytes"].as_u64().unwrap() + tree.bytes()
    );
    assert!(selfplay_cli::scan_staging(&held.staging, OUTPUT_CAP, until, &cancel).is_err());

    let mut transferred = fixture.held("transferred");
    transferred.cuda_artifact_tree = held.cuda_artifact_tree.clone();
    transferred.cuda_artifact_tree_observation = held.cuda_artifact_tree_observation.clone();
    drop(held);
    assert_eq!(
        cuda_tree::scan_staging(&transferred, OUTPUT_CAP, until, &cancel).unwrap(),
        scan
    );
    assert_eq!(
        cuda_tree::child_observation_for_dispatch(&transferred)
            .unwrap()
            .unwrap(),
        observation
    );
    assert_eq!(
        (credit.remaining, credit.maximum, credit.until),
        (remaining, READ_CAP, until)
    );
    assert!(Arc::ptr_eq(&credit.account, &account));
    assert_eq!(account.charged.load(Ordering::Acquire), charged);

    // 같은 JSON이라도 live Rust owner가 없으면 권한이 되지 않는다.
    let mut json_only = fixture.held("json-only");
    json_only.cuda_artifact_tree_observation = Some(observation);
    assert!(json_only.cuda_artifact_tree.is_none());
    assert!(cuda_tree::child_observation_for_dispatch(&json_only).is_err());
    assert!(cuda_tree::scan_staging(&json_only, OUTPUT_CAP, until, &cancel).is_err());
}

#[test]
fn actual_g0_prepare_rejects_wrong_core_and_replaced_original_source() {
    for replaced_source in [false, true] {
        let fixture = Fixture::new();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let mut credit = read_credit(&cancel, until, READ_CAP);
        let mut held = fixture.held("original");
        fixture.hold_inputs(&mut held, &mut credit);
        if replaced_source {
            fs::remove_file(fixture.source.join(CORE)).unwrap();
            fs::write(fixture.source.join(CORE), PAYLOAD).unwrap();
            expect_invalid(
                fixture.prepare(&mut held, &mut credit),
                "training held/named original file identity differs",
            );
        } else {
            expect_invalid(
                cuda_tree::prepare(
                    &mut held,
                    &fixture.cache,
                    &fixture.source.join("libonnxruntime_providers_shared.so"),
                    Some(&fixture.manifest),
                    OUTPUT_CAP,
                    &mut credit,
                ),
                "G0 CUDA core differs from the original runtime library",
            );
        }
        assert!(held.cuda_artifact_tree.is_none());
        assert!(held.cuda_artifact_tree_observation.is_none());
        assert!(
            cuda_tree::child_observation_for_dispatch(&held)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn actual_g0_prepare_preserves_exhausted_original_credit_and_cancel_failure() {
    for canceled in [false, true] {
        let fixture = Fixture::new();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        // 원 input read와 manifest 재독만으로 소진되는 등록값이다.
        let maximum = fixture.input_bytes() + fixture.manifest_bytes;
        let mut credit = read_credit(&cancel, until, maximum);
        let mut held = fixture.held("original");
        fixture.hold_inputs(&mut held, &mut credit);
        let charged_before = credit.account.charged.load(Ordering::Acquire);
        if canceled {
            cancel.store(true, Ordering::Release);
        }
        expect_invalid(
            fixture.prepare(&mut held, &mut credit),
            if canceled {
                "training original read deadline/cancel"
            } else {
                "training registered read credit exhausted"
            },
        );
        assert_eq!((credit.maximum, credit.until), (maximum, until));
        assert_eq!(
            credit.account.charged.load(Ordering::Acquire),
            if canceled { charged_before } else { maximum }
        );
        assert!(!credit.account.uncertain.load(Ordering::Acquire));
        assert!(held.cuda_artifact_tree.is_none());
        assert!(held.cuda_artifact_tree_observation.is_none());
        assert!(
            cuda_tree::child_observation_for_dispatch(&held)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn actual_g0_capability_rejects_renamed_directory_and_unregistered_alias() {
    for rename_directory in [false, true] {
        let fixture = Fixture::new();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let mut credit = read_credit(&cancel, until, READ_CAP);
        let mut held = fixture.held("original");
        fixture.hold_inputs(&mut held, &mut credit);
        fixture.prepare(&mut held, &mut credit).unwrap();
        let observation = cuda_tree::child_observation_for_dispatch(&held)
            .unwrap()
            .unwrap();
        let entry = fixture.entry(&observation);
        if rename_directory {
            fs::rename(&entry, fixture.cache.join("retired-cache-entry")).unwrap();
            fs::DirBuilder::new().mode(0o500).create(&entry).unwrap();
        } else {
            fs::set_permissions(&entry, fs::Permissions::from_mode(0o700)).unwrap();
            symlink("libcudnn.so.9", entry.join("unregistered-alias.so")).unwrap();
            fs::set_permissions(&entry, fs::Permissions::from_mode(0o500)).unwrap();
        }
        assert!(cuda_tree::scan_staging(&held, OUTPUT_CAP, until, &cancel).is_err());
        assert!(cuda_tree::child_observation_for_dispatch(&held).is_err());
    }
}
