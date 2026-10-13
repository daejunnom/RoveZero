//! Opaque content admission issued from this executable's source inventory and
//! the owner's retained file descriptors. It cannot be deserialized, filled by
//! a caller with passing flags, or issued from model metadata alone.
use super::{
    PalsArtifactDomainAdmissionV5, PalsNumericalAdmissionV5, validate_pals_artifact_domain_v5,
    validate_pals_numerical_acceptance_v5,
};
use crate::{
    ArtifactRef, ManifestError, PalsAdaptiveStartupConfigV5, PalsInputLockV5,
    PalsStartupAssetPurposeV5, PalsStartupAssetV5, compiled_source_inventory_v5,
    verify_compiled_source_components_v5,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File};

/// A borrowed declaration and the original retained, owner-opened descriptor.
/// The constructor never opens paths and never requests credentials or keys.
pub type PalsStartupOwnedAssetV5<'a> = (&'a PalsStartupAssetV5, &'a File);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PalsVerifiedStartupAssetV5 {
    pub artifact: ArtifactRef,
    pub snapshot_path: String,
    pub purpose: PalsStartupAssetPurposeV5,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

/// Successful content checks, bound to one canonical endpoint and its retained
/// files. Getters expose observations, not setters or a public passing flag.
/// This does not inspect checkpoint tensors/optimizers, establish readiness or
/// resource application, or certify this process's physical completion.
#[derive(Clone, Debug)]
pub struct PalsStartupContentAdmissionV5 {
    config_sha256: String,
    lock_sha256: String,
    endpoint_id: String,
    source_components: BTreeMap<String, String>,
    assets: BTreeMap<String, PalsVerifiedStartupAssetV5>,
    file_identities: BTreeMap<String, FileIdentity>,
    artifact_domain: PalsArtifactDomainAdmissionV5,
    numerical: PalsNumericalAdmissionV5,
    frozen_pc_collection: Option<PalsFrozenPcCollectionContentV5>,
}

/// Issued only after the source inventory, original descriptor roster, strong
/// frozen producer metadata and independent numerical inputs were joined.
/// This is not a Torch optimizer decoder or an original trainer execution token.
/// The collection pipeline retains that independent actual export owner.
#[derive(Clone, Debug)]
pub struct PalsFrozenPcCollectionContentV5 {
    checkpoint_sha256: String,
    named_shared_pc_sha256: String,
    model_epoch: u64,
    context_epoch: u64,
    provenance_sha256: String,
}
impl PalsFrozenPcCollectionContentV5 {
    pub fn checkpoint_sha256(&self) -> &str {
        &self.checkpoint_sha256
    }
    pub fn named_shared_pc_sha256(&self) -> &str {
        &self.named_shared_pc_sha256
    }
    pub fn model_epoch(&self) -> u64 {
        self.model_epoch
    }
    pub fn context_epoch(&self) -> u64 {
        self.context_epoch
    }
    pub fn provenance_sha256(&self) -> &str {
        &self.provenance_sha256
    }
}

impl PalsStartupContentAdmissionV5 {
    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }
    pub fn semantic_lock_sha256(&self) -> &str {
        &self.lock_sha256
    }
    pub fn endpoint_id(&self) -> &str {
        &self.endpoint_id
    }
    pub fn source_components(&self) -> &BTreeMap<String, String> {
        &self.source_components
    }
    pub fn assets(&self) -> &BTreeMap<String, PalsVerifiedStartupAssetV5> {
        &self.assets
    }
    pub fn artifact_domain(&self) -> &PalsArtifactDomainAdmissionV5 {
        &self.artifact_domain
    }
    pub fn numerical(&self) -> &PalsNumericalAdmissionV5 {
        &self.numerical
    }
    pub fn frozen_pc_collection(&self) -> Option<&PalsFrozenPcCollectionContentV5> {
        self.frozen_pc_collection.as_ref()
    }
    pub fn tensor_optimizer_content_verified(&self) -> bool {
        false
    }

    /// Check that a native consumer still owns the same canonical endpoint and
    /// descriptor roster. This is an ownership binding, not a second numerical
    /// run. Changed file identity/time/size is refused. The backend must still
    /// check each model/graph's expected SHA when materializing its actual bytes.
    pub fn verify_binding(
        &self,
        config: &PalsAdaptiveStartupConfigV5,
        lock: &PalsInputLockV5,
        assets: &[PalsStartupOwnedAssetV5<'_>],
    ) -> Result<(), ManifestError> {
        config.validate_against(lock)?;
        need(
            self.config_sha256 == digest(&config.canonical_bytes(lock)?)
                && self.lock_sha256 == lock.canonical_sha256
                && self.endpoint_id == config.endpoint.id
                && self.source_components == config.source_components,
            "opaque startup content admission belongs to another config, lock or source",
        )?;
        let roster = checked_roster(config, assets)?;
        need(
            roster.len() == self.assets.len(),
            "content admission asset roster differs",
        )?;
        for (path, (declared, file)) in roster {
            let expected = self
                .assets
                .get(path)
                .ok_or_else(|| fail("unbound startup asset"))?;
            need(
                expected.artifact == declared.artifact
                    && expected.snapshot_path == declared.snapshot_path
                    && expected.purpose == declared.purpose
                    && self.file_identities.get(path) == Some(&identity(file)?),
                "content admission descriptor owner or artifact identity differs",
            )?;
        }
        Ok(())
    }
}

fn fail(message: &str) -> ManifestError {
    ManifestError::Artifact(format!("V5 owned startup content: {message}"))
}
fn need(ok: bool, message: &str) -> Result<(), ManifestError> {
    if ok { Ok(()) } else { Err(fail(message)) }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(64);
    for byte in bytes {
        write!(&mut result, "{byte:02x}").expect("String write");
    }
    result
}

fn checked_roster<'a>(
    config: &PalsAdaptiveStartupConfigV5,
    assets: &'a [PalsStartupOwnedAssetV5<'a>],
) -> Result<BTreeMap<&'a str, (&'a PalsStartupAssetV5, &'a File)>, ManifestError> {
    need(
        assets.len() == config.assets.len(),
        "missing or extra owned startup descriptors",
    )?;
    let mut result = BTreeMap::new();
    for (declared, file) in assets {
        need(
            config.assets.iter().any(|asset| asset == *declared),
            "descriptor declaration is not in the canonical startup config",
        )?;
        need(
            result
                .insert(declared.snapshot_path.as_str(), (*declared, *file))
                .is_none(),
            "duplicate owned snapshot descriptor",
        )?;
    }
    Ok(result)
}

#[cfg(target_os = "linux")]
fn identity(file: &File) -> Result<FileIdentity, ManifestError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file
        .metadata()
        .map_err(|_| fail("retained descriptor metadata failed"))?;
    need(
        metadata.is_file() && metadata.nlink() == 1,
        "regular singly owned asset descriptor required",
    )?;
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}
#[cfg(not(target_os = "linux"))]
fn identity(_: &File) -> Result<FileIdentity, ManifestError> {
    Err(fail(
        "retained startup content admission requires the Linux owned-descriptor profile",
    ))
}

#[cfg(target_os = "linux")]
fn stream_identity(file: &File, artifact: &ArtifactRef) -> Result<FileIdentity, ManifestError> {
    use std::os::unix::fs::FileExt;
    artifact.validate()?;
    let before = identity(file)?;
    need(
        before.bytes == artifact.bytes && artifact.bytes <= 18u64 << 30,
        "asset size differs or exceeds managed bound",
    )?;
    let mut hash = Sha256::new();
    let mut count = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read_at(&mut buffer, count)
            .map_err(|_| fail("positional asset hash read failed"))?;
        if read == 0 {
            break;
        }
        count = count
            .checked_add(read as u64)
            .ok_or_else(|| fail("asset byte count overflow"))?;
        need(
            count <= artifact.bytes,
            "asset grew past its declared bound",
        )?;
        hash.update(&buffer[..read]);
    }
    need(
        count == artifact.bytes && hex(&hash.finalize().into()) == artifact.sha256,
        "actual descriptor SHA or length differs",
    )?;
    need(
        identity(file)? == before,
        "asset changed during content admission",
    )?;
    Ok(before)
}
#[cfg(not(target_os = "linux"))]
fn stream_identity(_: &File, _: &ArtifactRef) -> Result<FileIdentity, ManifestError> {
    Err(fail(
        "retained startup content admission requires Linux positional reads",
    ))
}

#[cfg(target_os = "linux")]
fn read_bounded(
    file: &File,
    artifact: &ArtifactRef,
    limit: usize,
) -> Result<Vec<u8>, ManifestError> {
    use std::os::unix::fs::FileExt;
    need(
        artifact.bytes > 0 && artifact.bytes <= limit as u64 && limit <= 4 << 20,
        "small startup evidence exceeds bounded read",
    )?;
    let before = identity(file)?;
    need(
        before.bytes == artifact.bytes,
        "small startup evidence length differs",
    )?;
    let mut result = vec![0; artifact.bytes as usize];
    file.read_exact_at(&mut result, 0)
        .map_err(|_| fail("small startup evidence positional read failed"))?;
    need(
        digest(&result) == artifact.sha256 && identity(file)? == before,
        "small startup evidence changed or SHA differs",
    )?;
    Ok(result)
}
#[cfg(not(target_os = "linux"))]
fn read_bounded(_: &File, _: &ArtifactRef, _: usize) -> Result<Vec<u8>, ManifestError> {
    Err(fail(
        "retained startup content admission requires Linux positional reads",
    ))
}

/// Mint content admission before provider/session creation from the exact owned
/// descriptors. Small reference captures are released pair by pair by the
/// shared numerical verifier. Weights and graphs are only streamed, never
/// materialized here; optimizer/tensor contents remain explicitly unverified.
pub fn admit_pals_startup_content_v5(
    config: &PalsAdaptiveStartupConfigV5,
    lock: &PalsInputLockV5,
    assets: &[PalsStartupOwnedAssetV5<'_>],
) -> Result<PalsStartupContentAdmissionV5, ManifestError> {
    config.validate_against(lock)?;
    need(
        !config.native_options.approximate_warm,
        "first-generation content admission is Fresh B1; seeded Warm is separate",
    )?;
    verify_compiled_source_components_v5(&config.source_components)?;
    let inventory = compiled_source_inventory_v5()?;
    let roster = checked_roster(config, assets)?;
    let mut file_identities = BTreeMap::new();
    let mut verified_assets = BTreeMap::new();
    let mut bytes = 0u64;
    for (path, (declared, file)) in &roster {
        bytes = bytes
            .checked_add(declared.artifact.bytes)
            .ok_or_else(|| fail("asset inventory size overflow"))?;
        need(bytes <= 18u64 << 30, "owned asset inventory exceeds 18GiB")?;
        file_identities.insert(
            (*path).to_owned(),
            stream_identity(file, &declared.artifact)?,
        );
        verified_assets.insert(
            (*path).to_owned(),
            PalsVerifiedStartupAssetV5 {
                artifact: declared.artifact.clone(),
                snapshot_path: declared.snapshot_path.clone(),
                purpose: declared.purpose,
            },
        );
    }
    let mut read = |artifact: &ArtifactRef, limit: usize| {
        let mut matches = roster
            .values()
            .filter(|(declared, _)| &declared.artifact == artifact);
        let (_, file) = matches
            .next()
            .ok_or_else(|| fail("requested evidence has no retained registered descriptor"))?;
        // Identical artifact identities may be present in two explicit snapshots;
        // each was independently streamed above. Read one original owned FD.
        read_bounded(file, artifact, limit)
    };
    let export = read(
        &config
            .asset(PalsStartupAssetPurposeV5::ExportManifest)?
            .artifact,
        256 << 10,
    )?;
    let mut domain = validate_pals_artifact_domain_v5(config, &export, &mut read)?;
    for (name, expected) in &domain.initializer_source_pins {
        let logical_path = format!("experiments/model-research/pals/src/rz_pals_model/{name}");
        let matches = inventory
            .values()
            .flatten()
            .filter(|row| row.0 == logical_path)
            .collect::<Vec<_>>();
        need(
            !matches.is_empty() && matches.iter().all(|row| hex(&row.2) == *expected),
            "initializer producer capture differs from embedded actual source bytes",
        )?;
    }
    domain.registered_artifact_bytes_verified_by_launch_owner = true;
    let numerical_bytes = read(
        &config
            .asset(PalsStartupAssetPurposeV5::NumericalAcceptance)?
            .artifact,
        256 << 10,
    )?;
    let numerical = validate_pals_numerical_acceptance_v5(config, &numerical_bytes, &mut read)?;
    domain.numerical_acceptance_verified = true;
    // Numerical and provenance evidence only. This retains the existing lower
    // scope of metadata summaries and never claims decoded optimizer validity.
    need(
        !domain.tensor_optimizer_content_verified && !domain.arena_accepted,
        "content verifier scope unexpectedly expanded",
    )?;
    let frozen_pc_collection =
        domain
            .frozen_pc_collection
            .as_ref()
            .map(|frozen| PalsFrozenPcCollectionContentV5 {
                checkpoint_sha256: frozen.checkpoint_sha256().to_owned(),
                named_shared_pc_sha256: frozen.named_shared_pc_sha256().to_owned(),
                model_epoch: frozen.model_epoch(),
                context_epoch: frozen.context_epoch(),
                provenance_sha256: domain
                    .provenance_file_sha256
                    .clone()
                    .expect("validated frozen provenance identity"),
            });
    let result = PalsStartupContentAdmissionV5 {
        config_sha256: digest(&config.canonical_bytes(lock)?),
        lock_sha256: lock.canonical_sha256.clone(),
        endpoint_id: config.endpoint.id.clone(),
        source_components: config.source_components.clone(),
        assets: verified_assets,
        file_identities,
        artifact_domain: domain,
        numerical,
        frozen_pc_collection,
    };
    result.verify_binding(config, lock, assets)?;
    Ok(result)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    #[test]
    fn positional_admission_preserves_cursor_and_refuses_changed_or_aliased_files() {
        let root = std::env::temp_dir().join(format!(
            "rz-v5-content-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("owned.bin");
        std::fs::write(&path, b"registered-content").unwrap();
        let mut file = File::open(&path).unwrap();
        file.seek(SeekFrom::Start(3)).unwrap();
        let artifact = ArtifactRef {
            path: "fixtures/owned.bin".into(),
            sha256: digest(b"registered-content"),
            bytes: 18,
            source: "https://example.org/authored-filesystem-fixture".into(),
            license: "MIT".into(),
        };
        let admitted = stream_identity(&file, &artifact).unwrap();
        assert_eq!(file.stream_position().unwrap(), 3);
        assert_eq!(
            read_bounded(&file, &artifact, 1024).unwrap(),
            b"registered-content"
        );
        assert_eq!(file.stream_position().unwrap(), 3);
        std::fs::write(&path, b"changed---content!").unwrap();
        assert!(stream_identity(&file, &artifact).is_err());
        assert_eq!(file.stream_position().unwrap(), 3);
        // A same-size rewrite is rejected by the hash even when the host's
        // metadata timestamp resolution cannot distinguish the two writes.
        // Use a distinct length to test identity changes deterministically.
        std::fs::write(&path, b"changed---content!!").unwrap();
        assert_ne!(file.metadata().unwrap().len(), artifact.bytes);
        assert_ne!(identity(&file).unwrap(), admitted);
        std::fs::hard_link(&path, root.join("alias.bin")).unwrap();
        assert!(identity(&file).is_err());
        drop(file);
        std::fs::remove_file(root.join("alias.bin")).unwrap();
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
