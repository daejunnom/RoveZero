use crate::{ManifestError, RunManifest};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
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
    // All subsequent operations stay relative to this directory handle.
    // cap-std confines intermediate symlink resolution even during a rename race.
    let directory = Dir::open_ambient_dir(root, cap_std::ambient_authority()).map_err(|_| {
        ManifestError::Artifact("artifact root must be an accessible directory".into())
    })?;
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
    for (logical, artifact) in unique {
        let fail = |reason: &str| ManifestError::Artifact(format!("{logical}: {reason}"));
        let components: Vec<_> = logical.split('/').collect();
        let (filename, parents) = components
            .split_last()
            .ok_or_else(|| fail("empty artifact path"))?;
        let mut scoped = directory
            .try_clone()
            .map_err(|_| fail("cannot pin artifact root"))?;
        // Pin each single directory component without following a symlink.
        // Later name replacement cannot redirect an already-open handle.
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
    }
    // Launch must reverify; a standalone preflight cannot bind future opens.
    Ok(())
}
