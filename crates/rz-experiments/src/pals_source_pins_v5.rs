//! Source component framing shared by registration, native receipts and runner.
//! This pure function hashes an explicit logical inventory; it never scans files
//! or turns a declaration into evidence that a component actually executed.
use crate::ManifestError;
use sha2::{Digest, Sha256};

pub const PALS_SOURCE_COMPONENT_V5_DOMAIN: &str = "rz-pals-source-component-v5/1";
pub const PALS_SOURCE_COMPONENT_V5_ALGORITHM: &str =
    "sha256-domain-u64le;sorted-path-u64le-utf8;size-u64le;sha256-raw32";

/// Sizes and hashes must come from actual owned source bytes. The caller keeps
/// their explicit inventory beside the component pin as independent evidence.
pub fn source_component_digest_v5(
    files: &[(String, u64, [u8; 32])],
) -> Result<[u8; 32], ManifestError> {
    if !(1..=256).contains(&files.len()) {
        return Err(ManifestError::Integrity(
            "bounded source inventory required".into(),
        ));
    }
    let mut ordered = files.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut hash = Sha256::new();
    hash.update((PALS_SOURCE_COMPONENT_V5_DOMAIN.len() as u64).to_le_bytes());
    hash.update(PALS_SOURCE_COMPONENT_V5_DOMAIN.as_bytes());
    let mut previous: Option<&str> = None;
    for (path, size, digest) in ordered {
        validate_source_path_v5(path)?;
        if previous == Some(path.as_str()) || *size == 0 {
            return Err(ManifestError::Integrity(
                "duplicate or empty source file".into(),
            ));
        }
        previous = Some(path);
        hash.update((path.len() as u64).to_le_bytes());
        hash.update(path.as_bytes());
        hash.update(size.to_le_bytes());
        hash.update(digest);
    }
    Ok(hash.finalize().into())
}

/// Logical source paths only. In particular, inventories cannot grant access
/// to secret files, the separate document WIP or generated artifact folders.
pub fn validate_source_path_v5(path: &str) -> Result<(), ManifestError> {
    let allowed = path.starts_with("crates/")
        || path.starts_with("scripts/")
        || path.starts_with("experiments/model-research/pals/src/")
        || path.starts_with(".github/workflows/");
    let lower = path.to_ascii_lowercase();
    let forbidden_component = path.split('/').any(|part| {
        part.is_empty()
            || part == "."
            || part == ".."
            || part.to_ascii_lowercase().starts_with(".env")
            || matches!(
                part,
                "node_modules"
                    | "dist"
                    | "dist-server"
                    | "build"
                    | "coverage"
                    | "models"
                    | "checkpoints"
                    | ".cache"
            )
    });
    let suffix = path.rsplit('.').next().unwrap_or("");
    if !allowed
        || forbidden_component
        || path.len() > 512
        || path.contains(['\\', ':'])
        || path.chars().any(char::is_control)
        || (!matches!(suffix, "rs" | "py" | "toml" | "yml" | "yaml")
            && path != "crates/rz-experiments/src/pals_source_inventory_v5.json")
        || lower.contains("credential")
        || lower.contains("service-account")
        || lower.contains("service_account")
        || lower.contains("api-key")
        || lower.contains("api_key")
        || lower.contains("ssh-key")
    {
        return Err(ManifestError::Integrity(
            "source inventory path is outside its allowed boundary".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn file(path: &str, size: u64, value: u8) -> (String, u64, [u8; 32]) {
        (path.into(), size, [value; 32])
    }
    #[test]
    fn inventory_is_order_independent_but_fences_size_path_and_content() {
        let a = file("crates/rz-search/src/cpu.rs", 123, 1);
        let b = file("crates/rz-runtime/src/lib.rs", 456, 2);
        let original = source_component_digest_v5(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(
            original,
            source_component_digest_v5(&[b.clone(), a.clone()]).unwrap()
        );
        let mut different = a.clone();
        different.1 += 1;
        assert_ne!(
            original,
            source_component_digest_v5(&[different, b.clone()]).unwrap()
        );
        different = a.clone();
        different.2[31] += 1;
        assert_ne!(
            original,
            source_component_digest_v5(&[different, b.clone()]).unwrap()
        );
        different = a.clone();
        different.0 = "crates/rz-search/src/cpu_worker.rs".into();
        assert_ne!(
            original,
            source_component_digest_v5(&[different, b]).unwrap()
        );
        assert!(source_component_digest_v5(&[a.clone(), a]).is_err());
    }
    #[test]
    fn source_inventory_never_authorizes_secret_generated_or_unrelated_paths() {
        for path in [
            ".env",
            "crates/a/../../.env",
            "C:/source.py",
            "crates/a/src/credential.rs",
            "crates/a/models/a.rs",
            "crates/a/src/service-account.json",
            "docs/research/PALS-IMPLEMENTATION.md",
            "crates/a/src\\a.rs",
            "crates//a.rs",
        ] {
            assert!(validate_source_path_v5(path).is_err(), "{path}");
        }
        assert!(source_component_digest_v5(&[]).is_err());
        assert!(source_component_digest_v5(&[file("crates/a/src/lib.rs", 0, 1)]).is_err());
    }
}
