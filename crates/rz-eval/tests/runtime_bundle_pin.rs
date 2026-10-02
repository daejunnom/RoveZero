//! Authored, tiny fake-library fixtures. No test executes ORT, CUDA or native code.

use rz_eval::runtime_pin::{
    CudaRuntimeBundleSpec, RuntimeBundleFile, RuntimeBundleFileRole as Role, RuntimeLibraryPin,
};
use sha2::{Digest, Sha256};

const CORE: &str = "libonnxruntime.so.1.22.0";
const SHARED: &str = "libonnxruntime_providers_shared.so";
const CUDA: &str = "libonnxruntime_providers_cuda.so";
const CUDART: &str = "libcudart.so.12";
const NVIDIA: &[&str] = &[
    CUDART, "libnvJitLink.so.12", "libnvrtc-builtins.so.12.8", "libnvrtc.so.12",
    "libcublasLt.so.12", "libcublas.so.12", "libcurand.so.10", "libcufft.so.11",
    "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn_adv.so.9", "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9", "libcudnn.so.9",
];
const FAKE_BYTES: &[u8] = b"authored fake bundle bytes; no native execution";

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn entry(role: Role, filename: &str) -> RuntimeBundleFile {
    RuntimeBundleFile {
        role,
        filename: filename.into(),
        bytes: FAKE_BYTES.len() as u64,
        sha256: hex(FAKE_BYTES),
    }
}

fn spec() -> CudaRuntimeBundleSpec {
    let mut files = vec![
        entry(Role::Core, CORE),
        entry(Role::ProvidersShared, SHARED),
        entry(Role::ProvidersCuda, CUDA),
    ];
    files.extend(NVIDIA.iter().map(|filename| entry(Role::NvidiaDependency, filename)));
    CudaRuntimeBundleSpec {
        schema_version: 1,
        files,
    }
}

fn manifest_json(spec: &CudaRuntimeBundleSpec) -> String {
    let files: Vec<_> = spec.files.iter().map(|file| {
        let role = match file.role {
            Role::Core => "core",
            Role::ProvidersShared => "providers_shared",
            Role::ProvidersCuda => "providers_cuda",
            Role::NvidiaDependency => "nvidia_dependency",
        };
        format!(
            "{{\"role\":\"{role}\",\"filename\":\"{}\",\"bytes\":{},\"sha256\":\"{}\"}}",
            file.filename, file.bytes, file.sha256
        )
    }).collect();
    format!(
        "{{\"schema_version\":{},\"files\":[{}]}}",
        spec.schema_version, files.join(",")
    )
}

#[test]
fn manifest_rejects_unknown_duplicate_fields_and_unbounded_json() {
    let source = manifest_json(&spec());
    assert!(CudaRuntimeBundleSpec::from_json(&source).is_ok());
    let rejected = [
        source.replacen("\"schema_version\":1", "\"schema_version\":1,\"unknown\":0", 1),
        source.replacen("\"schema_version\":1", "\"schema_version\":1,\"schema_version\":1", 1),
        source.replacen("\"bytes\":", "\"unknown\":0,\"bytes\":", 1),
        source.replacen("\"role\":\"core\"", "\"role\":\"core\",\"role\":\"core\"", 1),
        source.replacen("\"sha256\":", "\"bytes\":1,\"sha256\":", 1),
        format!("{source}{}", " ".repeat(64 * 1024)),
    ];
    for json in rejected {
        assert!(CudaRuntimeBundleSpec::from_json(&json).is_err());
    }
}

#[test]
fn canonical_identity_survives_order_and_binds_size_and_content() {
    let original = spec();
    let expected = original.digest().unwrap();
    let mut reordered = original.clone();
    reordered.files.reverse();
    assert_eq!(expected, reordered.digest().unwrap());
    assert_eq!(expected, CudaRuntimeBundleSpec::from_json(&manifest_json(&reordered)).unwrap().digest().unwrap());
    let mut changed_size = original.clone();
    changed_size.files[0].bytes += 1;
    assert_ne!(expected, changed_size.digest().unwrap());
    let mut changed_content = original.clone();
    changed_content.files[0].sha256 = hex(b"another authored fixture");
    assert_ne!(expected, changed_content.digest().unwrap());
    let mut changed_content = original.clone();
    changed_content.files[3].sha256 = hex(b"another NVIDIA fixture");
    assert_ne!(expected, changed_content.digest().unwrap());
}

#[test]
fn manifest_rejects_unsafe_names_wrong_roles_missing_core_and_bad_hashes() {
    for filename in ["../libcudart.so.12", "/tmp/libcudart.so.12", "..\\libcudart.so.12", "libunknown.so.12"] {
        let mut invalid = spec();
        invalid.files[3].filename = filename.into();
        assert!(invalid.validate().is_err());
        assert!(invalid.digest().is_err());
    }
    let mut invalid = spec();
    invalid.files[0].role = Role::NvidiaDependency;
    assert!(invalid.validate().is_err());
    let mut invalid = spec();
    invalid.files.remove(0);
    assert!(invalid.validate().is_err());
    let mut invalid = spec();
    invalid.files.pop();
    assert!(invalid.validate().is_err(), "the first GPU profile requires all 16 NVIDIA entries");
    let mut invalid = spec();
    invalid.files.push(invalid.files[3].clone());
    assert!(invalid.validate().is_err());
    for hash in ["f".repeat(63), "G".repeat(64), "A".repeat(64)] {
        let mut invalid = spec();
        invalid.files[0].sha256 = hash;
        assert!(invalid.validate().is_err());
    }
    let mut invalid = spec();
    invalid.schema_version = 2;
    assert!(invalid.validate().is_err());
}

#[test]
fn manifest_enforces_file_count_individual_and_total_byte_budgets() {
    for bytes in [0, 1024 * 1024 * 1024 + 1, u64::MAX] {
        let mut invalid = spec();
        invalid.files[0].bytes = bytes;
        assert!(invalid.validate().is_err());
    }
    let mut excessive = spec();
    excessive.files.resize(33, excessive.files[3].clone());
    assert!(excessive.validate().is_err());
    let mut excessive = spec();
    for file in &mut excessive.files {
        file.bytes = 1;
    }
    for file in &mut excessive.files[..4] {
        file.bytes = 1024 * 1024 * 1024;
    }
    assert!(excessive.validate().is_err());
    let remaining_file_bytes = (excessive.files.len() - 4) as u64;
    excessive.files[3].bytes -= remaining_file_bytes;
    assert!(excessive.validate().is_ok(), "exactly 4 GiB is within the manifest budget");
}

#[test]
fn debug_does_not_disclose_raw_hashes_or_unvalidated_paths() {
    let original = spec();
    let debug = format!("{original:?}");
    assert!(!debug.contains(&original.files[0].sha256));
    let mut invalid = original;
    invalid.files[0].filename = "/private/bootstrap/source.so".into();
    assert!(!format!("{invalid:?}").contains("/private/bootstrap"));
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{symlink, PermissionsExt},
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
    };
    static NEXT_TEST: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        source: PathBuf,
        output: PathBuf,
    }

    impl Fixture {
        fn new(spec: &CudaRuntimeBundleSpec) -> Self {
            let base = std::env::var_os("RUNNER_TEMP")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir)
                .join("RoveZero").join("tmp").join("native-bundle-tests");
            fs::create_dir_all(&base).unwrap();
            let root = base.canonicalize().unwrap().join(format!(
                "{}-{}", std::process::id(), NEXT_TEST.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            let source = root.join("source");
            let output = root.join("output");
            fs::create_dir(&source).unwrap();
            fs::create_dir(&output).unwrap();
            for file in &spec.files {
                fs::write(source.join(&file.filename), FAKE_BYTES).unwrap();
            }
            Self { root, source, output }
        }

        fn bundle_directory(&self) -> PathBuf {
            let entries: Vec<_> = fs::read_dir(&self.output).unwrap()
                .map(|entry| entry.unwrap().path()).collect();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].parent(), Some(self.output.as_path()));
            assert!(entries[0].is_dir());
            entries[0].clone()
        }

        fn remove_owned_files(directory: &Path) {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).unwrap();
            for entry in fs::read_dir(directory).unwrap() {
                let path = entry.unwrap().path();
                assert_eq!(path.parent(), Some(directory));
                // Only authored files in this test's directly owned directories.
                assert!(!fs::symlink_metadata(&path).unwrap().is_dir());
                fs::remove_file(path).unwrap();
            }
            fs::remove_dir(directory).unwrap();
        }

        fn release(self) {
            for entry in fs::read_dir(&self.output).unwrap() {
                let path = entry.unwrap().path();
                assert_eq!(path.parent(), Some(self.output.as_path()));
                if path.is_dir() {
                    Self::remove_owned_files(&path);
                } else {
                    fs::remove_file(path).unwrap();
                }
            }
            Self::remove_owned_files(&self.source);
            fs::remove_dir(&self.output).unwrap();
            fs::remove_dir(&self.root).unwrap();
        }
    }

    #[test]
    fn copied_bundle_is_immutable_from_source_and_survives_pin_clone() {
        let spec = spec();
        let fixture = Fixture::new(&spec);
        let pin = RuntimeLibraryPin::copy_cuda_bundle(&fixture.source, &fixture.output, &spec).unwrap();
        let directory = fixture.bundle_directory();
        assert_eq!(fs::metadata(&directory).unwrap().permissions().mode() & 0o777, 0o500);
        let expected_core_digest: [u8; 32] = Sha256::digest(FAKE_BYTES).into();
        assert_eq!(pin.binary_digest(), expected_core_digest);
        assert_eq!(pin.bundle_digest(), Some(spec.digest().unwrap()));
        let mut expected_files = spec.files.clone();
        expected_files.sort_unstable_by(|left, right| left.filename.cmp(&right.filename));
        assert_eq!(pin.bundle_files().unwrap(), expected_files.as_slice());
        for file in &spec.files {
            let owned = directory.join(&file.filename);
            assert_eq!(fs::metadata(&owned).unwrap().permissions().mode() & 0o777, 0o400);
            fs::write(fixture.source.join(&file.filename), b"source replaced after issue").unwrap();
            assert_eq!(fs::read(owned).unwrap(), FAKE_BYTES);
        }
        let debug = format!("{pin:?}");
        assert!(!debug.contains(fixture.root.to_str().unwrap()));
        assert!(!debug.contains(&spec.files[0].sha256));
        let cloned = pin.clone();
        drop(pin);
        assert_eq!(cloned.bundle_digest(), Some(spec.digest().unwrap()));
        assert_eq!(fs::read(directory.join(CUDA)).unwrap(), FAKE_BYTES);
        drop(cloned);
        fixture.release();
    }

    #[test]
    fn late_hash_or_length_failure_cleans_only_unpublished_bundle() {
        for wrong_length in [false, true] {
            let mut spec = spec();
            if wrong_length {
                spec.files[1].bytes += 1;
            } else {
                spec.files[1].sha256 = hex(b"different authored fixture");
            }
            let fixture = Fixture::new(&spec);
            let preserved = fixture.output.join("unrelated-authored.fixture");
            fs::write(&preserved, b"preserve this unrelated test file").unwrap();
            assert!(RuntimeLibraryPin::copy_cuda_bundle(&fixture.source, &fixture.output, &spec).is_err());
            assert_eq!(fs::read_dir(&fixture.output).unwrap().count(), 1);
            assert_eq!(fs::read(preserved).unwrap(), b"preserve this unrelated test file");
            assert_eq!(fs::read(fixture.source.join(CORE)).unwrap(), FAKE_BYTES);
            fixture.release();
        }
    }

    #[test]
    fn source_symlink_and_symlink_root_are_rejected_without_copy() {
        let spec = spec();
        let fixture = Fixture::new(&spec);
        let library = fixture.source.join(CUDA);
        fs::remove_file(&library).unwrap();
        symlink(fixture.source.join(CORE), &library).unwrap();
        assert!(RuntimeLibraryPin::copy_cuda_bundle(&fixture.source, &fixture.output, &spec).is_err());
        assert_eq!(fs::read_dir(&fixture.output).unwrap().count(), 0);
        let alias = fixture.root.join("source-alias");
        symlink(&fixture.source, &alias).unwrap();
        assert!(RuntimeLibraryPin::copy_cuda_bundle(&alias, &fixture.output, &spec).is_err());
        assert_eq!(fs::read_dir(&fixture.output).unwrap().count(), 0);
        fs::remove_file(alias).unwrap();
        fixture.release();
    }
}

#[cfg(not(target_os = "linux"))]
#[test]
fn nonlinux_gpu_bundle_is_explicitly_unavailable_before_any_copy() {
    let error = RuntimeLibraryPin::copy_cuda_bundle(
        std::path::Path::new("unopened-source"),
        std::path::Path::new("unopened-output"),
        &spec(),
    ).unwrap_err();
    assert!(error.detail.contains("require Linux"));
}
