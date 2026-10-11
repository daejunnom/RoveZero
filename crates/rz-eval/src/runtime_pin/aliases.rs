//! Closed loader aliases for an explicitly registered cuDNN 9.8.0 profile.
//! The nineteen regular pins keep their inode and single-link ownership.

use super::*;

const CUDNN_980_ALIASES: &[(&str, &str)] = &[
    ("libcudnn.so.9.8.0", "libcudnn.so.9"),
    ("libcudnn_adv.so.9.8.0", "libcudnn_adv.so.9"),
    ("libcudnn_cnn.so.9.8.0", "libcudnn_cnn.so.9"),
    ("libcudnn_ops.so.9.8.0", "libcudnn_ops.so.9"),
    (
        "libcudnn_engines_precompiled.so.9.8.0",
        "libcudnn_engines_precompiled.so.9",
    ),
    (
        "libcudnn_engines_runtime_compiled.so.9.8.0",
        "libcudnn_engines_runtime_compiled.so.9",
    ),
    ("libcudnn_graph.so.9.8.0", "libcudnn_graph.so.9"),
    ("libcudnn_heuristic.so.9.8.0", "libcudnn_heuristic.so.9"),
];

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBundleAlias {
    pub filename: String,
    pub target: String,
}

impl std::fmt::Debug for RuntimeBundleAlias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let allowed = allowed(&self.filename, &self.target);
        f.debug_struct("RuntimeBundleAlias")
            .field(
                "filename",
                &if allowed {
                    &*self.filename
                } else {
                    "<invalid alias>"
                },
            )
            .field(
                "target",
                &if allowed {
                    &*self.target
                } else {
                    "<invalid alias target>"
                },
            )
            .finish()
    }
}

fn allowed(filename: &str, target: &str) -> bool {
    CUDNN_980_ALIASES.contains(&(filename, target))
}

pub(super) fn validate(spec: &CudaRuntimeBundleSpec) -> Result<(), BackendError> {
    let aliases = match (spec.schema_version, &spec.loader_aliases) {
        (1, None) => return Ok(()),
        (2, Some(aliases)) if !aliases.is_empty() && aliases.len() <= CUDNN_980_ALIASES.len() => {
            aliases
        }
        _ => {
            return Err(bundle_error(
                "CUDA loader aliases require an explicit nonempty V2 declaration",
            ))
        }
    };
    for (index, alias) in aliases.iter().enumerate() {
        if !allowed(&alias.filename, &alias.target)
            || aliases[..index]
                .iter()
                .any(|previous| previous.filename == alias.filename)
            || !spec.files.iter().any(|file| {
                file.role == RuntimeBundleFileRole::NvidiaDependency
                    && file.filename == alias.target
            })
        {
            return Err(bundle_error(
                "CUDA loader alias is undeclared, duplicate or outside the closed profile",
            ));
        }
    }
    Ok(())
}

pub(super) fn allowed_original_basename(filename: &str) -> bool {
    matches!(filename, CORE_FILENAME | SHARED_FILENAME | CUDA_FILENAME)
        || NVIDIA_FILENAMES.contains(&filename)
}

/// Validate only declared relative symlinks. No external or chained target is
/// followed: the target basename has a regular, single-link readonly pin.
#[cfg(target_os = "linux")]
pub(super) fn verify(
    directory: &Path,
    aliases: &[RuntimeBundleAlias],
    files: &[RuntimeBundleFile],
) -> Result<(), BackendError> {
    for alias in aliases {
        let target = files
            .iter()
            .find(|file| file.filename == alias.target)
            .ok_or_else(|| bundle_identity_error("CUDA loader alias target is not declared"))?;
        verify_one(
            directory,
            &alias.filename,
            &alias.target,
            Some(target.bytes),
        )?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub(super) fn verify_one(
    directory: &Path,
    filename: &str,
    target: &str,
    expected_bytes: Option<u64>,
) -> Result<(), BackendError> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
    if !allowed(filename, target) {
        return Err(bundle_identity_error(
            "CUDA loader alias is outside the closed profile",
        ));
    }
    let path = directory.join(filename);
    let named = fs::symlink_metadata(&path)
        .map_err(|error| bundle_io_error("CUDA loader alias metadata failed", &error))?;
    if !named.file_type().is_symlink() || named.nlink() != 1 {
        return Err(bundle_identity_error(
            "CUDA loader alias must be a single declared relative symlink",
        ));
    }
    let relative = fs::read_link(&path)
        .map_err(|error| bundle_io_error("CUDA loader alias target is unavailable", &error))?;
    if relative != Path::new(target) {
        return Err(bundle_identity_error(
            "CUDA loader alias target differs from its declaration",
        ));
    }
    let target_path = directory.join(target);
    let target_named = fs::symlink_metadata(&target_path).map_err(|error| {
        bundle_io_error("CUDA loader alias regular target is unavailable", &error)
    })?;
    if !target_named.is_file()
        || target_named.file_type().is_symlink()
        || target_named.nlink() != 1
        || target_named.permissions().mode() & 0o777 != 0o400
        || target_named.len() == 0
        || target_named.len() > MAX_BUNDLE_FILE_BYTES
        || expected_bytes.is_some_and(|bytes| bytes != target_named.len())
    {
        return Err(bundle_identity_error(
            "CUDA loader alias target is not the readonly single-link regular file",
        ));
    }
    // O_NOFOLLOW protects the regular target. Only the declared alias is
    // followed, and O_NONBLOCK prevents a swapped FIFO from blocking a check.
    let target_pin = OpenOptions::new()
        .read(true)
        .custom_flags(0x20000 | 0x800)
        .open(&target_path)
        .map_err(|error| bundle_io_error("cannot open CUDA loader alias regular target", &error))?;
    let alias_pin = OpenOptions::new()
        .read(true)
        .custom_flags(0x800)
        .open(&path)
        .map_err(|error| bundle_io_error("cannot open declared CUDA loader alias", &error))?;
    let target_opened = target_pin
        .metadata()
        .map_err(|error| bundle_io_error("CUDA loader alias target pin metadata failed", &error))?;
    let opened = alias_pin
        .metadata()
        .map_err(|error| bundle_io_error("CUDA loader alias pin metadata failed", &error))?;
    for metadata in [&target_opened, &opened] {
        if !metadata.is_file()
            || metadata.dev() != target_named.dev()
            || metadata.ino() != target_named.ino()
            || metadata.len() != target_named.len()
            || metadata.nlink() != 1
            || metadata.permissions().mode() & 0o777 != 0o400
        {
            return Err(bundle_identity_error(
                "CUDA loader alias and pinned target inode differ",
            ));
        }
    }
    let after = fs::symlink_metadata(&path)
        .map_err(|error| bundle_io_error("CUDA loader alias final metadata failed", &error))?;
    if !after.file_type().is_symlink()
        || named.dev() != after.dev()
        || named.ino() != after.ino()
        || after.nlink() != 1
        || fs::read_link(&path)
            .map_err(|error| bundle_io_error("CUDA loader alias final target failed", &error))?
            != relative
    {
        return Err(bundle_identity_error(
            "CUDA loader alias changed during verification",
        ));
    }
    Ok(())
}

/// A quota scan may encounter entries published by an earlier owner. Accept
/// links only inside the distinct V2 namespace and only for this closed pair.
#[cfg(target_os = "linux")]
pub(super) fn verify_cached_name(directory: &Path, filename: &str) -> Result<(), BackendError> {
    let target = CUDNN_980_ALIASES
        .iter()
        .find_map(|(alias, target)| (*alias == filename).then_some(*target))
        .ok_or_else(|| {
            bundle_identity_error("runtime cache contains an undeclared loader alias")
        })?;
    verify_one(directory, filename, target, None)
}
