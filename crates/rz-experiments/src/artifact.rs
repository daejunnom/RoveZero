use crate::{ManifestError, RunManifest};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::Path;

pub(crate) fn verify(input: &RunManifest, root: &Path, max_total_bytes: u64) -> Result<(), ManifestError> {
    input.validate()?;
    if max_total_bytes == 0 { return Err(ManifestError::Artifact("explicit positive byte ceiling required".into())); }
    let root = root.canonicalize().map_err(|_| ManifestError::Artifact("artifact root is unavailable".into()))?;
    if !root.is_dir() { return Err(ManifestError::Artifact("artifact root must be a directory".into())); }
    let unique: BTreeMap<_, _> = input.artifacts().into_iter().map(|a| (&a.path, a)).collect();
    let total = unique.values().try_fold(0_u64, |sum, a| sum.checked_add(a.bytes))
        .ok_or_else(|| ManifestError::Artifact("artifact byte total overflow".into()))?;
    if total > max_total_bytes || total > input.budget.max_artifact_bytes {
        return Err(ManifestError::Artifact("declared artifacts exceed verification byte ceiling".into()));
    }
    for (logical, artifact) in unique {
        let fail = |reason: &str| ManifestError::Artifact(format!("{logical}: {reason}"));
        let mut path = root.clone();
        // Refuse all symlink components, including ones pointing inside the root.
        for component in logical.split('/') {
            path.push(component);
            let metadata = fs::symlink_metadata(&path).map_err(|_| fail("declared file is unavailable"))?;
            if metadata.file_type().is_symlink() { return Err(fail("symlink artifacts are unsupported")); }
        }
        let resolved = path.canonicalize().map_err(|_| fail("cannot resolve declared file"))?;
        if !resolved.starts_with(&root) { return Err(fail("file escapes artifact root")); }
        let before = fs::symlink_metadata(&resolved).map_err(|_| fail("cannot inspect declared file"))?;
        if !before.is_file() { return Err(fail("artifact must be a regular file")); }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)] {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
        }
        let file = options.open(&resolved).map_err(|_| fail("cannot open declared file"))?;
        let metadata = file.metadata().map_err(|_| fail("cannot inspect opened file"))?;
        if !metadata.is_file() || metadata.len() != artifact.bytes { return Err(fail("file type or byte length mismatch")); }
        let mut reader = file.take(artifact.bytes);
        let mut hasher = Sha256::new();
        let mut consumed = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = reader.read(&mut buffer).map_err(|_| fail("cannot read declared file"))?;
            if count == 0 { break; }
            consumed += count as u64;
            hasher.update(&buffer[..count]);
        }
        let after = reader.get_ref().metadata().map_err(|_| fail("cannot inspect verified file"))?;
        if consumed != artifact.bytes || after.len() != artifact.bytes
            || format!("{:x}", hasher.finalize()) != artifact.sha256 {
            return Err(fail("SHA-256 or byte length mismatch"));
        }
    }
    // Launch must reverify; a standalone preflight cannot bind future opens.
    Ok(())
}
