//! Retained V5 snapshot descriptors to the existing shared runtime cache.
//!
//! No library is loaded here. The native factory owner calls this before ORT
//! initialization, away from the coordinator. A cache hit still verifies the
//! cached bytes and retains the cache's file pins. The original snapshot FDs
//! are hashed positionally and never sought, cloned for sequential reading, or
//! replaced by `/proc` paths. Source pathname reopening is confined to the
//! existing RuntimeCache API: its real directory and named files are compared
//! with the held owner capabilities before and after that call, and the final
//! cache pin is checked against the original declaration. Participating owners
//! must not mutate these private snapshots; this is not a hostile same-UID
//! filesystem sandbox or an assertion that every transient pathname race can
//! be observed.

use crate::pals_adaptive_startup::PinnedPalsStartupV5;
use rz_eval::{
    error::{BackendError, FailureKind as K, FailureStage as S},
    runtime_pin::RuntimeLibraryPin,
};
use rz_experiments::PalsModelBackendV3;

/// Provider selection comes only from the admitted endpoint. No environment
/// variable, path suffix or automatic CPU fallback changes that selection.
pub fn pin_runtime_v5(owner: &PinnedPalsStartupV5) -> Result<RuntimeLibraryPin, BackendError> {
    owner
        .config
        .validate_against(&owner.lock)
        .map_err(|_| invalid("V5 runtime config no longer matches its semantic lock"))?;
    #[cfg(target_os = "linux")]
    {
        let root = owner
            .private_root_capability()
            .map_err(|_| invalid("cannot retain the original V5 snapshot root"))?;
        verify_owner_runtime_roster(owner)?;
        let plan = RuntimePlan::from_assets(owner.config.endpoint.model.backend, owner.assets())?;
        let guards = SnapshotGuards::capture(owner.private_cwd(), root, &plan)?;
        let cache = RuntimeCache::for_user()?;
        pin_plan(&cache, owner.private_cwd(), &plan, guards)
    }
    #[cfg(not(target_os = "linux"))]
    {
        if owner.config.endpoint.model.backend == PalsModelBackendV3::OrtCuda {
            return Err(unavailable(
                "V5 CUDA runtime pins require the Linux owned-descriptor profile",
            ));
        }
        Err(unavailable(
            "V5 retained runtime identity requires the Linux positional-descriptor profile",
        ))
    }
}

fn invalid(detail: &'static str) -> BackendError {
    BackendError::new(K::IdentityMismatch, S::Backend, detail)
}
fn unavailable(detail: &'static str) -> BackendError {
    BackendError::new(K::BackendUnavailable, S::Backend, detail)
}
#[cfg(target_os = "linux")]
fn io(detail: &'static str) -> BackendError {
    BackendError::new(K::Io, S::Backend, detail)
}
#[cfg(target_os = "linux")]
fn require(ok: bool, detail: &'static str) -> Result<(), BackendError> {
    if ok { Ok(()) } else { Err(invalid(detail)) }
}

#[cfg(target_os = "linux")]
use {
    crate::pals_adaptive_startup::PinnedStartupAssetV5,
    cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt},
    cap_std::fs::{Dir, OpenOptions},
    rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole, RuntimeCache},
    rz_experiments::PalsStartupAssetPurposeV5 as Purpose,
    sha2::{Digest as _, Sha256},
    std::{
        collections::BTreeSet,
        fs::File,
        os::unix::fs::{FileExt, MetadataExt},
        path::{Component, Path},
    },
};

#[cfg(target_os = "linux")]
fn verify_owner_runtime_roster(owner: &PinnedPalsStartupV5) -> Result<(), BackendError> {
    let relevant = |purpose| {
        matches!(
            purpose,
            Purpose::RuntimeLibrary | Purpose::CudaLibrary | Purpose::CudaManifest
        )
    };
    let declared: Vec<_> = owner
        .config
        .assets
        .iter()
        .filter(|a| relevant(a.purpose))
        .collect();
    let held: Vec<_> = owner
        .assets()
        .iter()
        .filter(|a| relevant(a.declaration.purpose))
        .collect();
    require(
        declared.len() == held.len()
            && declared
                .iter()
                .all(|a| held.iter().filter(|h| &h.declaration == *a).count() == 1),
        "runtime config roster differs from the retained owner declarations",
    )
}

#[cfg(target_os = "linux")]
struct CudaPlan<'a> {
    manifest: &'a PinnedStartupAssetV5,
    spec: CudaRuntimeBundleSpec,
    files: Vec<&'a PinnedStartupAssetV5>,
    parent: String,
}
#[cfg(target_os = "linux")]
struct RuntimePlan<'a> {
    core: &'a PinnedStartupAssetV5,
    cuda: Option<CudaPlan<'a>>,
}
#[cfg(target_os = "linux")]
impl<'a> RuntimePlan<'a> {
    fn from_assets(
        backend: PalsModelBackendV3,
        assets: &'a [PinnedStartupAssetV5],
    ) -> Result<Self, BackendError> {
        if !matches!(
            backend,
            PalsModelBackendV3::OrtCpu | PalsModelBackendV3::OrtCuda
        ) {
            return Err(unavailable(
                "this V5 runtime bridge requires an explicitly selected ORT backend",
            ));
        }
        let exactly_one = |purpose| {
            let mut entries = assets.iter().filter(|a| a.declaration.purpose == purpose);
            let first = entries
                .next()
                .ok_or_else(|| invalid("missing runtime asset purpose"))?;
            require(entries.next().is_none(), "ambiguous runtime asset purpose")?;
            Ok::<_, BackendError>(first)
        };
        let core = exactly_one(Purpose::RuntimeLibrary)?;
        let cuda_files: Vec<_> = assets
            .iter()
            .filter(|a| a.declaration.purpose == Purpose::CudaLibrary)
            .collect();
        let manifests: Vec<_> = assets
            .iter()
            .filter(|a| a.declaration.purpose == Purpose::CudaManifest)
            .collect();
        if backend == PalsModelBackendV3::OrtCpu {
            require(
                cuda_files.is_empty() && manifests.is_empty(),
                "CPU runtime cannot carry CUDA libraries or a bundle manifest",
            )?;
            require(
                core.declaration.artifact.bytes > 0 && core.declaration.artifact.bytes <= 512 << 20,
                "CPU runtime exceeds the existing library bound",
            )?;
            return Ok(Self { core, cuda: None });
        }
        require(
            manifests.len() == 1 && cuda_files.len() == 18,
            "CUDA runtime requires exactly one manifest and eighteen non-core files",
        )?;
        let manifest = manifests[0];
        let raw = read_small(manifest, 64 << 10)?;
        let spec = CudaRuntimeBundleSpec::from_json(
            std::str::from_utf8(&raw).map_err(|_| invalid("CUDA runtime manifest is not UTF-8"))?,
        )?;
        let (parent, basename) = split_snapshot(core)?;
        let declared_core = spec
            .files
            .iter()
            .find(|f| f.role == RuntimeBundleFileRole::Core)
            .ok_or_else(|| invalid("CUDA runtime manifest lacks its core"))?;
        require(
            declared_core.filename == basename
                && declared_core.bytes == core.declaration.artifact.bytes
                && declared_core.sha256 == core.declaration.artifact.sha256,
            "CUDA core name, size or SHA differs from retained runtime declaration",
        )?;
        let mut seen = BTreeSet::new();
        for asset in &cuda_files {
            let (file_parent, basename) = split_snapshot(asset)?;
            require(
                file_parent == parent && seen.insert(basename),
                "CUDA runtime files require one real directory and unique basenames",
            )?;
            let entry = spec
                .files
                .iter()
                .find(|f| f.filename == basename)
                .ok_or_else(|| invalid("undeclared CUDA runtime file"))?;
            require(
                entry.role != RuntimeBundleFileRole::Core
                    && entry.bytes == asset.declaration.artifact.bytes
                    && entry.sha256 == asset.declaration.artifact.sha256,
                "CUDA runtime role, size or SHA differs from its exact manifest entry",
            )?;
        }
        require(
            spec.files.len() == 19
                && seen.len() == 18
                && spec
                    .files
                    .iter()
                    .filter(|f| f.role != RuntimeBundleFileRole::Core)
                    .all(|f| seen.contains(f.filename.as_str())),
            "missing or extra CUDA runtime basename",
        )?;
        // Spec validation owns the closed V2 alias allowlist. Alias names are
        // generated inside the immutable cache, never accepted as source files.
        Ok(Self {
            core,
            cuda: Some(CudaPlan {
                manifest,
                spec,
                files: cuda_files,
                parent: parent.to_owned(),
            }),
        })
    }
    fn originals(&self) -> Vec<&'a PinnedStartupAssetV5> {
        let mut originals = vec![self.core];
        if let Some(cuda) = &self.cuda {
            originals.push(cuda.manifest);
            originals.extend(cuda.files.iter().copied());
        }
        originals
    }
}

#[cfg(target_os = "linux")]
fn split_snapshot(asset: &PinnedStartupAssetV5) -> Result<(&str, &str), BackendError> {
    require(
        rz_experiments::pals_private_snapshot_path_v5(&asset.declaration.snapshot_path),
        "unsafe runtime snapshot name",
    )?;
    asset
        .declaration
        .snapshot_path
        .rsplit_once('/')
        .ok_or_else(|| invalid("runtime snapshot requires a real parent"))
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}
#[cfg(target_os = "linux")]
fn identity(file: &File) -> Result<FileIdentity, BackendError> {
    let metadata = file
        .metadata()
        .map_err(|_| io("runtime descriptor metadata is unavailable"))?;
    require(
        metadata.is_file() && metadata.nlink() == 1,
        "runtime descriptor is not a singly owned regular file",
    )?;
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        modified: (metadata.mtime(), metadata.mtime_nsec()),
        changed: (metadata.ctime(), metadata.ctime_nsec()),
    })
}
#[cfg(target_os = "linux")]
fn verify_descriptor(asset: &PinnedStartupAssetV5) -> Result<FileIdentity, BackendError> {
    asset
        .declaration
        .artifact
        .validate()
        .map_err(|_| invalid("runtime artifact declaration is invalid"))?;
    let before = identity(&asset.file)?;
    require(
        before.bytes == asset.declaration.artifact.bytes
            && before.bytes > 0
            && before.bytes <= 4 << 30,
        "runtime descriptor length differs from its bound",
    )?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 << 10];
    let mut offset = 0u64;
    while offset < before.bytes {
        let length = usize::try_from((before.bytes - offset).min(buffer.len() as u64))
            .map_err(|_| invalid("runtime length conversion failed"))?;
        let count = asset
            .file
            .read_at(&mut buffer[..length], offset)
            .map_err(|_| io("runtime positional hash read failed"))?;
        require(
            count > 0,
            "runtime descriptor ended before its declared size",
        )?;
        hash.update(&buffer[..count]);
        offset += count as u64;
    }
    require(
        asset
            .file
            .read_at(&mut buffer[..1], offset)
            .map_err(|_| io("runtime descriptor end check failed"))?
            == 0,
        "runtime descriptor grew beyond its declared size",
    )?;
    require(
        format!("{:x}", hash.finalize()) == asset.declaration.artifact.sha256
            && identity(&asset.file)? == before,
        "runtime descriptor SHA or identity changed during verification",
    )?;
    Ok(before)
}
#[cfg(target_os = "linux")]
fn read_small(asset: &PinnedStartupAssetV5, limit: u64) -> Result<Vec<u8>, BackendError> {
    require(
        asset.declaration.artifact.bytes > 0 && asset.declaration.artifact.bytes <= limit,
        "runtime manifest exceeds its bounded descriptor budget",
    )?;
    let before = verify_descriptor(asset)?;
    let mut bytes = vec![0; before.bytes as usize];
    asset
        .file
        .read_exact_at(&mut bytes, 0)
        .map_err(|_| io("runtime manifest positional read failed"))?;
    require(
        format!("{:x}", Sha256::digest(&bytes)) == asset.declaration.artifact.sha256
            && identity(&asset.file)? == before,
        "runtime manifest changed while read",
    )?;
    Ok(bytes)
}

#[cfg(target_os = "linux")]
struct NamedGuard<'a> {
    asset: &'a PinnedStartupAssetV5,
    parent: Dir,
    basename: String,
    identity: FileIdentity,
}
#[cfg(target_os = "linux")]
struct SnapshotGuards<'a> {
    root: Dir,
    root_identity: (u64, u64),
    files: Vec<NamedGuard<'a>>,
}
#[cfg(target_os = "linux")]
fn directory_identity(directory: &Dir) -> Result<(u64, u64), BackendError> {
    let metadata = directory
        .try_clone()
        .map_err(|_| io("cannot clone runtime directory capability"))?
        .into_std_file()
        .metadata()
        .map_err(|_| io("runtime directory identity unavailable"))?;
    require(metadata.is_dir(), "runtime capability is not a directory")?;
    Ok((metadata.dev(), metadata.ino()))
}
#[cfg(target_os = "linux")]
fn open_real_absolute_directory(path: &Path) -> Result<Dir, BackendError> {
    require(path.is_absolute(), "runtime source root is not absolute")?;
    let mut directory = Dir::open_ambient_dir("/", cap_std::ambient_authority())
        .map_err(|_| io("cannot open the filesystem root capability"))?;
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(name) => {
                directory = directory.open_dir_nofollow(name).map_err(|_| {
                    invalid("runtime source directory contains a link or unavailable parent")
                })?
            }
            _ => {
                return Err(invalid(
                    "runtime source root contains traversal or unsupported prefix",
                ));
            }
        }
    }
    Ok(directory)
}
#[cfg(target_os = "linux")]
fn open_named(parent: &Dir, basename: &str) -> Result<File, BackendError> {
    require(
        parent
            .symlink_metadata(basename)
            .map_err(|_| io("runtime snapshot name is missing"))?
            .is_file(),
        "runtime snapshot name is linked or nonregular",
    )?;
    let mut options = OpenOptions::new();
    use cap_std::fs::OpenOptionsExt;
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(0x20000 | 0x800);
    parent
        .open_with(basename, &options)
        .map(|f| f.into_std())
        .map_err(|_| invalid("cannot open a no-follow runtime snapshot name"))
}
#[cfg(target_os = "linux")]
impl<'a> SnapshotGuards<'a> {
    fn capture(path: &Path, root: Dir, plan: &RuntimePlan<'a>) -> Result<Self, BackendError> {
        let root_identity = directory_identity(&root)?;
        require(
            directory_identity(&open_real_absolute_directory(path)?)? == root_identity,
            "ambient runtime source root differs from its original capability",
        )?;
        let mut files = Vec::new();
        for asset in plan.originals() {
            let held_identity = verify_descriptor(asset)?;
            let (parents, basename) = split_snapshot(asset)?;
            let mut parent = root
                .try_clone()
                .map_err(|_| io("cannot retain runtime snapshot parent"))?;
            for part in parents.split('/') {
                parent = parent
                    .open_dir_nofollow(part)
                    .map_err(|_| invalid("runtime snapshot parent changed or is linked"))?;
            }
            require(
                identity(&open_named(&parent, basename)?)? == held_identity,
                "runtime snapshot path no longer names its original descriptor",
            )?;
            files.push(NamedGuard {
                asset,
                parent,
                basename: basename.to_owned(),
                identity: held_identity,
            });
        }
        let guards = Self {
            root,
            root_identity,
            files,
        };
        guards.verify(path)?;
        Ok(guards)
    }
    fn verify(&self, path: &Path) -> Result<(), BackendError> {
        require(
            directory_identity(&self.root)? == self.root_identity
                && directory_identity(&open_real_absolute_directory(path)?)? == self.root_identity,
            "runtime snapshot root was replaced",
        )?;
        for guard in &self.files {
            let (parents, _) = split_snapshot(guard.asset)?;
            let ambient_parent = open_real_absolute_directory(&path.join(parents))?;
            require(
                directory_identity(&ambient_parent)? == directory_identity(&guard.parent)?,
                "runtime snapshot parent was replaced",
            )?;
            require(
                identity(&guard.asset.file)? == guard.identity
                    && identity(&open_named(&ambient_parent, &guard.basename)?)? == guard.identity,
                "runtime snapshot named file or original descriptor changed",
            )?;
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn pin_plan(
    cache: &RuntimeCache,
    root: &Path,
    plan: &RuntimePlan<'_>,
    guards: SnapshotGuards<'_>,
) -> Result<RuntimeLibraryPin, BackendError> {
    guards.verify(root)?;
    let result = if let Some(cuda) = &plan.cuda {
        cache.cuda_bundle(&root.join(&cuda.parent), &cuda.spec)
    } else {
        cache.library(
            &root.join(&plan.core.declaration.snapshot_path),
            &plan.core.declaration.artifact.sha256,
        )
    };
    // Even cache failure must not hide an observed snapshot replacement. A
    // cache-created immutable entry remains preserved under cache ownership.
    guards.verify(root)?;
    let pin = result?;
    let actual_core = pin
        .binary_digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    require(
        actual_core == plan.core.declaration.artifact.sha256,
        "final runtime pin differs from the retained core SHA",
    )?;
    if let Some(cuda) = &plan.cuda {
        let expected = cuda.spec.digest()?;
        let actual = pin
            .bundle_spec()
            .ok_or_else(|| invalid("final CUDA pin is missing its exact bundle specification"))?;
        require(
            pin.bundle_digest() == Some(expected)
                && actual.files.len() == 19
                && actual.digest()? == expected,
            "final CUDA pin lost its exact nineteen-file roster or declared aliases",
        )?;
        require(
            pin.storage().bytes == cuda.spec.files.iter().map(|f| f.bytes).sum::<u64>(),
            "final CUDA pin storage byte count differs",
        )?;
    } else {
        require(
            pin.bundle_digest().is_none()
                && pin.bundle_files().is_none()
                && pin.storage().bytes == plan.core.declaration.artifact.bytes,
            "CPU pin unexpectedly carries a CUDA bundle or different length",
        )?;
    }
    Ok(pin)
}

#[cfg(all(test, target_os = "linux"))]
#[path = "runtime_assets/tests.rs"]
mod tests;
