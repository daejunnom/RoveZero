use crate::{ArtifactRef, ManifestError, RunManifest};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub(crate) fn verify(
    input: &RunManifest,
    root: &Path,
    max_total_bytes: u64,
) -> Result<(), ManifestError> {
    input.validate()?;
    if max_total_bytes == 0 {
        return Err(ManifestError::Artifact(
            "explicit positive byte ceiling required".into(),
        ));
    }
    let unique: BTreeMap<_, _> = input
        .artifacts()
        .into_iter()
        .map(|a| (&a.path, a))
        .collect();
    let total = unique
        .values()
        .try_fold(0_u64, |sum, a| sum.checked_add(a.bytes))
        .ok_or_else(|| ManifestError::Artifact("artifact byte total overflow".into()))?;
    if total > max_total_bytes || total > input.budget.max_artifact_bytes {
        return Err(ManifestError::Artifact(
            "declared artifacts exceed verification byte ceiling".into(),
        ));
    }
    let directory = open_root(root)?;
    for artifact in unique.values() {
        // Use the same pinned-handle verification as launch consumers, while
        // retaining the aggregate budget and logical-path deduplication above.
        drop(open_verified_in(&directory, artifact, max_total_bytes)?);
    }
    // Launch must reverify; a standalone preflight cannot bind future opens.
    Ok(())
}

pub(crate) fn open_verified(
    artifact: &ArtifactRef,
    root: &Path,
    max_bytes: u64,
) -> Result<File, ManifestError> {
    check_metadata_budget(artifact, max_bytes)?;
    open_verified_in(&open_root(root)?, artifact, max_bytes)
}

fn open_root(root: &Path) -> Result<Dir, ManifestError> {
    // All subsequent operations stay relative to this pinned root handle.
    Dir::open_ambient_dir(root, cap_std::ambient_authority()).map_err(|_| {
        ManifestError::Artifact("artifact root must be an accessible directory".into())
    })
}

fn check_metadata_budget(artifact: &ArtifactRef, max_bytes: u64) -> Result<(), ManifestError> {
    artifact.validate()?;
    if max_bytes == 0 || artifact.bytes > max_bytes {
        return Err(ManifestError::Artifact(
            "declared artifact exceeds its explicit positive byte ceiling".into(),
        ));
    }
    Ok(())
}

fn open_verified_in(
    directory: &Dir,
    artifact: &ArtifactRef,
    max_bytes: u64,
) -> Result<File, ManifestError> {
    check_metadata_budget(artifact, max_bytes)?;
    let fail = |reason: &str| ManifestError::Artifact(format!("{}: {reason}", artifact.path));
    let components: Vec<_> = artifact.path.split('/').collect();
    let (filename, parents) = components
        .split_last()
        .ok_or_else(|| fail("empty artifact path"))?;
    let mut scoped = directory
        .try_clone()
        .map_err(|_| fail("cannot pin artifact root"))?;
    // Single-component no-follow opens pin each directory before considering
    // the next name, including when an ancestor is concurrently renamed.
    for component in parents {
        scoped = scoped
            .open_dir_nofollow(component)
            .map_err(|_| fail("parent directory is unavailable or a symlink"))?;
    }
    let before = scoped
        .symlink_metadata(filename)
        .map_err(|_| fail("cannot inspect declared file"))?;
    if !before.is_file() {
        return Err(fail("artifact must be a regular file"));
    }
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    #[cfg(unix)]
    {
        use cap_std::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = scoped
        .open_with(filename, &options)
        .map_err(|_| fail("cannot open declared file within artifact root"))?;
    let metadata = file
        .metadata()
        .map_err(|_| fail("cannot inspect opened file"))?;
    if !metadata.is_file() || metadata.len() != artifact.bytes {
        return Err(fail("file type or byte length mismatch"));
    }
    let mut reader = file.take(artifact.bytes);
    let mut hasher = Sha256::new();
    let mut consumed = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|_| fail("cannot read declared file"))?;
        if count == 0 {
            break;
        }
        consumed += count as u64;
        hasher.update(&buffer[..count]);
    }
    let after = reader
        .get_ref()
        .metadata()
        .map_err(|_| fail("cannot inspect verified file"))?;
    if consumed != artifact.bytes
        || after.len() != artifact.bytes
        || format!("{:x}", hasher.finalize()) != artifact.sha256
    {
        return Err(fail("SHA-256 or byte length mismatch"));
    }
    // Move the verified handle out of the bounded reader and rewind it. Never
    // reopen its pathname after digest verification.
    let mut verified = reader.into_inner();
    verified
        .seek(SeekFrom::Start(0))
        .map_err(|_| fail("cannot rewind verified file"))?;
    Ok(verified.into_std())
}
