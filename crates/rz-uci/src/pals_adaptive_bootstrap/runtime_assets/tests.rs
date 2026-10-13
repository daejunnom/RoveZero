//! Source-only fixtures; fake library bytes are never dlopened or sent to ORT.
use super::*;
use rz_eval::runtime_pin::RuntimeStorageOrigin;
use rz_experiments::{ArtifactRef, PalsStartupAssetV5};
use std::{
    fs,
    io::{Seek, SeekFrom},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let unique = NEXT.fetch_add(1, Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rz-v5-runtime-fixture-{}-{unique}-{now}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("inputs")).unwrap();
        fs::create_dir(root.join("inputs/runtime")).unwrap();
        Self { root }
    }
    fn root_cap(&self) -> Dir {
        Dir::open_ambient_dir(&self.root, cap_std::ambient_authority()).unwrap()
    }
    fn asset(&self, filename: &str, purpose: Purpose, bytes: &[u8]) -> PinnedStartupAssetV5 {
        let snapshot_path = format!("inputs/runtime/{filename}");
        let path = self.root.join(&snapshot_path);
        fs::write(&path, bytes).unwrap();
        PinnedStartupAssetV5 {
            declaration: PalsStartupAssetV5 {
                artifact: ArtifactRef {
                    path: format!("runtime/{filename}"),
                    bytes: bytes.len() as u64,
                    sha256: format!("{:x}", Sha256::digest(bytes)),
                    source: "https://example.org/fixtures/rovezero/unexecuted-runtime-fixture"
                        .into(),
                    license: "fixture-only".into(),
                },
                snapshot_path,
                purpose,
            },
            file: File::open(path).unwrap(),
        }
    }
    fn cpu(&self) -> Vec<PinnedStartupAssetV5> {
        vec![self.asset(
            "libonnxruntime.so.1.22.0",
            Purpose::RuntimeLibrary,
            b"fake, unexecuted CPU library bytes",
        )]
    }
    fn cuda(&self, aliases: bool) -> Vec<PinnedStartupAssetV5> {
        let names = [
            ("libonnxruntime.so.1.22.0", "core"),
            ("libonnxruntime_providers_shared.so", "providers_shared"),
            ("libonnxruntime_providers_cuda.so", "providers_cuda"),
            ("libcudart.so.12", "nvidia_dependency"),
            ("libcublas.so.12", "nvidia_dependency"),
            ("libcublasLt.so.12", "nvidia_dependency"),
            ("libcufft.so.11", "nvidia_dependency"),
            ("libcurand.so.10", "nvidia_dependency"),
            ("libnvrtc.so.12", "nvidia_dependency"),
            ("libnvrtc-builtins.so.12.8", "nvidia_dependency"),
            ("libnvJitLink.so.12", "nvidia_dependency"),
            ("libcudnn.so.9", "nvidia_dependency"),
            ("libcudnn_adv.so.9", "nvidia_dependency"),
            ("libcudnn_cnn.so.9", "nvidia_dependency"),
            ("libcudnn_ops.so.9", "nvidia_dependency"),
            ("libcudnn_engines_precompiled.so.9", "nvidia_dependency"),
            (
                "libcudnn_engines_runtime_compiled.so.9",
                "nvidia_dependency",
            ),
            ("libcudnn_graph.so.9", "nvidia_dependency"),
            ("libcudnn_heuristic.so.9", "nvidia_dependency"),
        ];
        let mut result = Vec::new();
        let mut files = Vec::new();
        for (name, role) in names {
            let bytes = format!("fake unexecuted CUDA file: {name}");
            let purpose = if role == "core" {
                Purpose::RuntimeLibrary
            } else {
                Purpose::CudaLibrary
            };
            let asset = self.asset(name, purpose, bytes.as_bytes());
            files.push(serde_json::json!({ "role": role, "filename": name, "bytes": asset.declaration.artifact.bytes, "sha256": asset.declaration.artifact.sha256 }));
            result.push(asset);
        }
        // Deliberately noncanonical input ordering exercises semantic digest
        // equivalence when the cache stores its filename-sorted specification.
        files.reverse();
        let mut value =
            serde_json::json!({ "schema_version": if aliases { 2 } else { 1 }, "files": files });
        if aliases {
            value["loader_aliases"] = serde_json::json!([
                { "filename": "libcudnn.so.9.8.0", "target": "libcudnn.so.9" },
                { "filename": "libcudnn_cnn.so.9.8.0", "target": "libcudnn_cnn.so.9" }
            ]);
        }
        let bytes = serde_json::to_vec(&value).unwrap();
        result.push(self.asset("cuda-bundle.json", Purpose::CudaManifest, &bytes));
        result
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Only this create-new fixture is removed. Never follow a replacement
        // root or clean a shared RuntimeCache::for_user directory.
        if self
            .root
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("rz-v5-runtime-fixture-"))
            && fs::symlink_metadata(&self.root)
                .is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
        {
            let _ = remove_fixture_tree(&self.root, 0);
        }
    }
}
fn remove_fixture_tree(path: &std::path::Path, depth: u8) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if depth > 6 {
        return Err(std::io::Error::other("unexpected fixture depth"));
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return fs::remove_file(path);
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    let entries: Vec<_> = fs::read_dir(path)?
        .take(65)
        .collect::<std::io::Result<_>>()?;
    if entries.len() > 64 {
        return Err(std::io::Error::other("unexpected fixture entries"));
    }
    for entry in entries {
        remove_fixture_tree(&entry.path(), depth + 1)?;
    }
    fs::remove_dir(path)
}

#[test]
fn positional_runtime_verification_preserves_the_original_cursor() {
    let fixture = Fixture::new();
    let mut assets = fixture.cpu();
    assets[0].file.seek(SeekFrom::Start(7)).unwrap();
    verify_descriptor(&assets[0]).unwrap();
    let plan = RuntimePlan::from_assets(PalsModelBackendV3::OrtCpu, &assets).unwrap();
    let guards = SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).unwrap();
    guards.verify(&fixture.root).unwrap();
    assert_eq!(assets[0].file.stream_position().unwrap(), 7);
}

#[test]
fn changed_bytes_and_same_bytes_replacement_are_both_rejected() {
    let fixture = Fixture::new();
    let assets = fixture.cpu();
    let plan = RuntimePlan::from_assets(PalsModelBackendV3::OrtCpu, &assets).unwrap();
    let guards = SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).unwrap();
    let path = fixture.root.join(&assets[0].declaration.snapshot_path);
    let original = fs::read(&path).unwrap();
    let original_modified = assets[0].file.metadata().unwrap().modified().unwrap();
    fs::write(&path, vec![b'x'; original.len()]).unwrap();
    assert_eq!(
        verify_descriptor(&assets[0]).unwrap_err().kind,
        K::IdentityMismatch
    );
    // SHA verification above rejects the same-length corruption independently.
    // The snapshot guard compares metadata, so make that change explicit rather
    // than assuming back-to-back writes have distinct filesystem timestamps.
    assets[0]
        .file
        .set_modified(original_modified + std::time::Duration::from_secs(60))
        .unwrap();
    assert_eq!(
        guards.verify(&fixture.root).unwrap_err().kind,
        K::IdentityMismatch
    );

    fs::write(&path, &original).unwrap();
    let plan = RuntimePlan::from_assets(PalsModelBackendV3::OrtCpu, &assets).unwrap();
    let guards = SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).unwrap();
    fs::rename(&path, fixture.root.join("original-retained.bin")).unwrap();
    fs::write(&path, &original).unwrap();
    // Matching bytes do not authorize a new inode as the original snapshot.
    assert_eq!(
        guards.verify(&fixture.root).unwrap_err().kind,
        K::IdentityMismatch
    );
}

#[test]
fn linked_snapshot_or_replaced_parent_is_rejected() {
    let fixture = Fixture::new();
    let assets = fixture.cpu();
    let plan = RuntimePlan::from_assets(PalsModelBackendV3::OrtCpu, &assets).unwrap();
    let guards = SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).unwrap();
    let path = fixture.root.join(&assets[0].declaration.snapshot_path);
    fs::rename(&path, fixture.root.join("original-retained.bin")).unwrap();
    std::os::unix::fs::symlink(fixture.root.join("original-retained.bin"), &path).unwrap();
    assert!(guards.verify(&fixture.root).is_err());
    fs::remove_file(path).unwrap();
    fs::rename(
        fixture.root.join("inputs/runtime"),
        fixture.root.join("old-runtime"),
    )
    .unwrap();
    fs::create_dir(fixture.root.join("inputs/runtime")).unwrap();
    assert_eq!(
        guards.verify(&fixture.root).unwrap_err().kind,
        K::IdentityMismatch
    );
}

#[test]
fn cpu_cache_is_shared_by_content_and_hit_still_checks_the_original() {
    let fixture = Fixture::new();
    let assets = fixture.cpu();
    let cache = RuntimeCache::open(&fixture.root.join("shared-cache")).unwrap();
    let plan = RuntimePlan::from_assets(PalsModelBackendV3::OrtCpu, &assets).unwrap();
    let first = pin_plan(
        &cache,
        &fixture.root,
        &plan,
        SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).unwrap(),
    )
    .unwrap();
    assert_eq!(first.storage().origin, RuntimeStorageOrigin::CacheCreated);
    let second = pin_plan(
        &cache,
        &fixture.root,
        &plan,
        SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).unwrap(),
    )
    .unwrap();
    assert_eq!(second.storage().origin, RuntimeStorageOrigin::CacheReused);
    assert_eq!(first.binary_digest(), second.binary_digest());
    assert_eq!(first.storage().cache_key, second.storage().cache_key);
    let path = fixture.root.join(&assets[0].declaration.snapshot_path);
    fs::write(path, b"invalid original despite a still valid cache").unwrap();
    assert!(SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).is_err());
}

#[test]
fn backend_is_explicit_and_cpu_cannot_carry_cuda_assets() {
    let fixture = Fixture::new();
    let assets = fixture.cpu();
    assert!(
        matches!(RuntimePlan::from_assets(PalsModelBackendV3::DeterministicMock, &assets), Err(e) if e.kind == K::BackendUnavailable)
    );
    assert!(
        matches!(RuntimePlan::from_assets(PalsModelBackendV3::RustCpu, &assets), Err(e) if e.kind == K::BackendUnavailable)
    );
    let cuda_fixture = Fixture::new();
    let cuda_assets = cuda_fixture.cuda(false);
    assert!(RuntimePlan::from_assets(PalsModelBackendV3::OrtCpu, &cuda_assets).is_err());
}

#[test]
fn cuda_roster_and_aliases_are_preserved_through_content_cache_reuse() {
    let fixture = Fixture::new();
    let assets = fixture.cuda(true);
    let plan = RuntimePlan::from_assets(PalsModelBackendV3::OrtCuda, &assets).unwrap();
    assert_eq!(plan.cuda.as_ref().unwrap().spec.files.len(), 19);
    let cache = RuntimeCache::open(&fixture.root.join("shared-cache")).unwrap();
    let pin = pin_plan(
        &cache,
        &fixture.root,
        &plan,
        SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).unwrap(),
    )
    .unwrap();
    assert_eq!(pin.storage().origin, RuntimeStorageOrigin::CacheCreated);
    assert_eq!(pin.bundle_spec().unwrap().loader_aliases.unwrap().len(), 2);
    let reused = pin_plan(
        &cache,
        &fixture.root,
        &plan,
        SnapshotGuards::capture(&fixture.root, fixture.root_cap(), &plan).unwrap(),
    )
    .unwrap();
    assert_eq!(reused.storage().origin, RuntimeStorageOrigin::CacheReused);
    assert_eq!(pin.bundle_digest(), reused.bundle_digest());
}

#[test]
fn cuda_missing_extra_wrong_size_sha_and_role_are_rejected() {
    let fixture = Fixture::new();
    let mut assets = fixture.cuda(false);
    assets.pop();
    assert!(RuntimePlan::from_assets(PalsModelBackendV3::OrtCuda, &assets).is_err());

    let fixture = Fixture::new();
    let mut assets = fixture.cuda(false);
    assets[1].declaration.artifact.bytes += 1;
    assert!(RuntimePlan::from_assets(PalsModelBackendV3::OrtCuda, &assets).is_err());
    assets[1].declaration.artifact.bytes -= 1;
    assets[1].declaration.artifact.sha256 = "a".repeat(64);
    assert!(RuntimePlan::from_assets(PalsModelBackendV3::OrtCuda, &assets).is_err());
    assets[1].declaration.purpose = Purpose::RuntimeLibrary;
    assert!(RuntimePlan::from_assets(PalsModelBackendV3::OrtCuda, &assets).is_err());

    let fixture = Fixture::new();
    let mut assets = fixture.cuda(false);
    assets.push(fixture.asset(
        "undeclared-library.so",
        Purpose::CudaLibrary,
        b"not in the closed CUDA roster",
    ));
    assert!(RuntimePlan::from_assets(PalsModelBackendV3::OrtCuda, &assets).is_err());
}

#[test]
fn cuda_sources_cannot_use_an_alias_as_a_regular_pin_or_different_directory() {
    let fixture = Fixture::new();
    let mut assets = fixture.cuda(true);
    assets[1].declaration.snapshot_path = "inputs/runtime/libcudnn.so.9.8.0".into();
    assert!(RuntimePlan::from_assets(PalsModelBackendV3::OrtCuda, &assets).is_err());
    assets[1].declaration.snapshot_path =
        "inputs/another/libonnxruntime_providers_shared.so".into();
    assert!(RuntimePlan::from_assets(PalsModelBackendV3::OrtCuda, &assets).is_err());
}
